"""Overlap graph over a cluster system, and its structural metrics.

Two clusters ``A != B`` are *nested* if one contains the other, *disjoint* if ``A & B`` is empty,
and *overlapping* otherwise.  Only overlapping pairs are edges of the overlap graph ``O(C)``.
With ``ic = |A & B|``::

    overlap(A, B)  <=>  0 < ic < min(|A|, |B|)

Two kernels compute this and are cross-tested against each other in ``tests/test_relations.py``:

``"popcount"``  ``np.bitwise_count`` over packed ``uint64`` words.  Reference implementation.
``"gemm"``      a ``float32`` matrix product of the 0/1 membership matrices.  Production kernel:
                identical throughput single-threaded, faster at ``n=200``, and it scales with
                BLAS threads.  Exact because intersection counts are integers below ``2**24``.

Scaling notes (SPEC-AMENDMENTS A4, A5):

* The edge list is materialised only while the running edge count stays under ``edge_cap``.
  Past that the scan degrades to counting and sets ``dense_bailout``.  A cell that trips the cap
  has mean degree above ``N/10``, which *is* the RED signature -- it is recorded, never skipped.
* ``(L)`` violations are counted **exactly** in ``O(E)`` by grouping each apex's neighbours by the
  value of ``A & B``, so no sampling path is needed.  Triangles remain size-gated and may be
  estimated for components above ``dense_triangle_max``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _cc

from opc import bitset

__all__ = [
    "EDGE_CAP",
    "BLOCK",
    "OverlapGraph",
    "build_overlap_graph",
    "overlap_pairs_bruteforce",
    "component_metrics",
    "ComponentMetrics",
    "cluster_weighted_p99",
    "edge_intersections",
    "assignment_pairs",
    "l_violations_per_vertex",
]

#: Largest edge list we will materialise.  1e8 edges = 800 MB as two int32 arrays.
EDGE_CAP = 100_000_000

#: Block edge length for the tiled scan.  2048 measured optimum; 4096 was ~30% worse.
BLOCK = 2048

#: Components at or below this size get an exact dense triangle count; larger ones are sampled.
DENSE_TRIANGLE_MAX = 2000


def _tile_gemm(dense_i, dense_j, sz_i, sz_j):
    """Boolean overlap tile via float32 GEMM."""
    ic = dense_i @ dense_j.T
    return (ic > 0) & (ic < sz_i[:, None]) & (ic < sz_j[None, :])


def _tile_popcount(words_i, words_j, sz_i, sz_j):
    """Boolean overlap tile via ``np.bitwise_count``.  Reference kernel."""
    ic = np.bitwise_count(words_i[:, None, :] & words_j[None, :, :]).sum(axis=2)
    return (ic > 0) & (ic < sz_i[:, None]) & (ic < sz_j[None, :])


@dataclass
class OverlapGraph:
    """Result of scanning a cluster system for overlapping pairs."""

    n_vertices: int
    n_edges: int
    degrees: np.ndarray
    ii: np.ndarray | None
    jj: np.ndarray | None
    dense_bailout: bool
    words: np.ndarray = field(repr=False)
    sizes: np.ndarray = field(repr=False)

    @property
    def n_isolated(self) -> int:
        return int(np.count_nonzero(self.degrees == 0))

    @property
    def kernel_fraction(self) -> float:
        """Fraction of clusters overlapping nothing -- accepted unconditionally by the solver."""
        return self.n_isolated / self.n_vertices if self.n_vertices else float("nan")

    def csr(self) -> sp.csr_matrix:
        """Symmetric CSR adjacency.  Neighbour indices are sorted within each row."""
        if self.ii is None:
            raise RuntimeError("edge list was not materialised (dense_bailout)")
        n = self.n_vertices
        data = np.ones(2 * self.ii.size, dtype=np.int8)
        rows = np.concatenate([self.ii, self.jj])
        cols = np.concatenate([self.jj, self.ii])
        out = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        out.sort_indices()
        return out


def build_overlap_graph(
    words: np.ndarray,
    n_taxa: int,
    *,
    block: int = BLOCK,
    edge_cap: int = EDGE_CAP,
    kernel: str = "gemm",
) -> OverlapGraph:
    """Scan every unordered pair of clusters and collect the overlapping ones.

    ``words`` is ``uint64[N, W]`` as produced by :mod:`opc.bitset`.  Rows must already be
    deduplicated -- duplicate clusters would appear as nested, not overlapping, but they also
    inflate every component-size metric, so the caller is responsible for dedup.
    """
    if kernel not in ("gemm", "popcount"):
        raise ValueError(f"unknown kernel {kernel!r}")
    n = words.shape[0]
    sizes = bitset.popcounts(words)
    degrees = np.zeros(n, dtype=np.int64)
    if n < 2:
        return OverlapGraph(n, 0, degrees.astype(np.int32), np.empty(0, np.int32),
                            np.empty(0, np.int32), False, words, sizes)

    dense = bitset.to_dense(words, n_taxa) if kernel == "gemm" else None
    sz_f = sizes.astype(np.float32)

    total = 0
    collecting = True
    chunks_i: list[np.ndarray] = []
    chunks_j: list[np.ndarray] = []

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            if kernel == "gemm":
                ov = _tile_gemm(dense[i0:i1], dense[j0:j1], sz_f[i0:i1], sz_f[j0:j1])
            else:
                ov = _tile_popcount(words[i0:i1], words[j0:j1], sizes[i0:i1], sizes[j0:j1])
            if i0 == j0:
                # Diagonal tile: keep the strict upper triangle so each pair is seen once.
                ov = np.triu(ov, k=1)
                degrees[i0:i1] += ov.sum(axis=1) + ov.sum(axis=0)
            else:
                degrees[i0:i1] += ov.sum(axis=1)
                degrees[j0:j1] += ov.sum(axis=0)
            cnt = int(ov.sum())
            total += cnt
            if collecting and cnt:
                if total > edge_cap:
                    collecting = False
                    chunks_i.clear()
                    chunks_j.clear()
                else:
                    a, b = np.nonzero(ov)
                    chunks_i.append((a + i0).astype(np.int32))
                    chunks_j.append((b + j0).astype(np.int32))

    if collecting:
        ii = np.concatenate(chunks_i) if chunks_i else np.empty(0, np.int32)
        jj = np.concatenate(chunks_j) if chunks_j else np.empty(0, np.int32)
    else:
        ii = jj = None

    return OverlapGraph(n, total, degrees.astype(np.int32), ii, jj, not collecting, words, sizes)


def overlap_pairs_bruteforce(words: np.ndarray, n_taxa: int) -> set[tuple[int, int]]:
    """Every overlapping pair, by explicit Python set arithmetic.  Tests only."""
    sets = [frozenset(bitset.to_indices(words[i], n_taxa)) for i in range(words.shape[0])]
    out = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            inter = a & b
            if inter and inter != a and inter != b:
                out.add((i, j))
    return out


# --------------------------------------------------------------------------------------------
# (L) violations -- exact, O(E)
# --------------------------------------------------------------------------------------------


def _sum_choose2_of_groups(sorted_keys: np.ndarray) -> int:
    """``sum_i C(g_i, 2)`` over runs of equal values in a sorted array."""
    if sorted_keys.size < 2:
        return 0
    bounds = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    edges = np.concatenate(([0], bounds, [sorted_keys.size]))
    g = np.diff(edges).astype(np.int64)
    return int((g * (g - 1) // 2).sum())


def edge_intersections(graph: OverlapGraph) -> tuple[np.ndarray, np.ndarray]:
    """``(hcand, edge_class)`` -- the distinct values of ``A & B`` over the edges, and a label.

    Under (L) every edge of a connected component must carry the *same* intersection, so these
    values are exactly the candidate hybrid clusters an ILP assigns blocks to.  They come free:
    :func:`l_violations_per_vertex` already materialises the same intersections in order to group
    a vertex's neighbours by them.

    Grouped by the **exact bitset**, not by :func:`opc.bitset.fnv1a_many`.  A 64-bit collision is
    harmless when the grouping only feeds a count, but here it would merge two candidate hybrids
    into one variable and silently produce a different model.

    Note ``hcand`` deliberately keeps singleton intersections.  They are excluded from ``C_tau``
    by construction, but they are still the intersection that defines a block -- 53-61% of the
    grid's overlap edges meet in a singleton, and 40-80% of true hybrid clusters *are*
    singletons, so dropping them would forbid exactly the reticulations the study is about.
    """
    if graph.ii is None:
        raise RuntimeError("edge list was not materialised (dense_bailout)")
    if graph.ii.size == 0:
        return np.zeros((0, graph.words.shape[1]), dtype=np.uint64), np.zeros(0, dtype=np.int64)
    inter = graph.words[graph.ii] & graph.words[graph.jj]
    hcand, edge_class = np.unique(inter, axis=0, return_inverse=True)
    return hcand, np.asarray(edge_class).ravel()


def assignment_pairs(
    ii: np.ndarray, jj: np.ndarray, edge_class: np.ndarray, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex, class)`` pairs -- one per ``z[A, h]`` variable the ILP would create.

    ``A`` only ever needs a ``z`` for an intersection value some incident edge actually realises,
    so this is ``sum_A g_A`` in SPEC-AMENDMENTS A5's notation and it is the honest size proxy for
    the assignment formulation.  A5's ``n_l_violations`` counts rows of a *different* (triple)
    formulation and runs 1.3x to 146x larger.
    """
    if ii.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    verts = np.concatenate([ii, jj])
    classes = np.concatenate([edge_class, edge_class])
    pairs = np.unique(np.stack([verts, classes], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def l_violations_per_vertex(graph: OverlapGraph, csr: sp.csr_matrix | None = None) -> np.ndarray:
    """``bad[A]`` = number of pairs ``{B, C}`` of ``A``'s neighbours with ``A&B != A&C``.

    An ordered ``(L)`` violation is a triple ``(A; B, C)`` where ``A`` overlaps both and the two
    intersections differ, so ``sum(bad)`` is exactly the number of ``(L)`` constraints an ILP over
    this component set would emit (SPEC-AMENDMENTS A5).  Computed as
    ``C(deg A, 2) - sum_i C(g_i, 2)`` where ``g_i`` are the sizes of the groups of neighbours
    sharing the same intersection value -- ``O(deg A)`` per apex instead of ``O(deg A ** 2)``.
    """
    if csr is None:
        csr = graph.csr()
    n = graph.n_vertices
    words = graph.words
    indptr, indices = csr.indptr, csr.indices
    bad = np.zeros(n, dtype=np.int64)
    if indices.size == 0:
        return bad

    # Intersection of every (row, neighbour) pair, hashed to a full 64-bit key.  Full width, not
    # packed with the row id: at E ~ 1e7 a 40-bit key would collide ~89 times.
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    inter = words[rows] & words[indices]
    keys = bitset.fnv1a_many(inter)

    for a in range(n):
        lo, hi = indptr[a], indptr[a + 1]
        d = hi - lo
        if d < 2:
            continue
        bad[a] = (d * (d - 1)) // 2 - _sum_choose2_of_groups(np.sort(keys[lo:hi]))
    return bad


# --------------------------------------------------------------------------------------------
# Components and per-component metrics
# --------------------------------------------------------------------------------------------


@dataclass
class ComponentMetrics:
    """Per-cell structural summary of ``O(C_tau)``."""

    n_components: int
    comp_sizes: np.ndarray
    labels: np.ndarray
    giant_size: int
    n_triangles: int
    n_triangles_exact: bool
    n_l_violations: int
    n_l_violations_max_comp: int
    n_triangles_max_comp: int
    n_comp_n3o_free: int
    n_comp_l_free: int
    comp_edge_density_max: float

    @property
    def n_nontrivial(self) -> int:
        """Components with at least two vertices, per the spec's definition of *component*."""
        return int(np.count_nonzero(self.comp_sizes >= 2))


def _triangles_dense(sub_ii, sub_jj, size) -> int:
    adj = np.zeros((size, size), dtype=np.float32)
    adj[sub_ii, sub_jj] = 1.0
    adj[sub_jj, sub_ii] = 1.0
    return int(round(float((adj * (adj @ adj)).sum()) / 6.0))


def _triangles_sampled(csr, verts, sub_ii, sub_jj, rng, n_sample) -> tuple[int, bool]:
    """Unbiased wedge estimate of the triangle count of one component."""
    n_edges = sub_ii.size
    if n_edges == 0:
        return 0, True
    take = min(n_sample, n_edges)
    pick = rng.choice(n_edges, size=take, replace=False) if take < n_edges else np.arange(n_edges)
    indptr, indices = csr.indptr, csr.indices
    shared = 0
    for e in pick:
        u = verts[sub_ii[e]]
        v = verts[sub_jj[e]]
        nu = indices[indptr[u]:indptr[u + 1]]
        nv = indices[indptr[v]:indptr[v + 1]]
        shared += np.intersect1d(nu, nv, assume_unique=True).size
    est = shared / take * n_edges / 3.0
    return int(round(est)), False


def component_metrics(
    graph: OverlapGraph,
    *,
    dense_triangle_max: int = DENSE_TRIANGLE_MAX,
    triangle_sample: int = 20_000,
    seed: int = 0,
) -> ComponentMetrics:
    """Connected components of ``O(C)`` plus the triangle / (L) statistics the spec asks for."""
    n = graph.n_vertices
    if graph.dense_bailout:
        # Mean degree above N/10.  Report the RED signature rather than skipping the cell.
        return ComponentMetrics(
            n_components=1, comp_sizes=np.array([n]), labels=np.zeros(n, np.int32),
            giant_size=n, n_triangles=-1, n_triangles_exact=False, n_l_violations=-1,
            n_l_violations_max_comp=-1, n_triangles_max_comp=-1, n_comp_n3o_free=0,
            n_comp_l_free=0, comp_edge_density_max=1.0,
        )

    ii, jj = graph.ii, graph.jj
    adj = sp.coo_matrix(
        (np.ones(ii.size, dtype=np.int8), (ii, jj)), shape=(n, n)
    ) if ii.size else sp.coo_matrix((n, n), dtype=np.int8)
    n_comp, labels = _cc(adj, directed=False)
    comp_sizes = np.bincount(labels, minlength=n_comp)
    giant = int(comp_sizes.max()) if n_comp else 0

    csr = graph.csr()
    bad = l_violations_per_vertex(graph, csr)
    bad_by_comp = np.bincount(labels, weights=bad, minlength=n_comp).astype(np.int64)

    rng = np.random.default_rng(seed)
    tri_by_comp = np.zeros(n_comp, dtype=np.int64)
    all_exact = True
    density_max = 0.0

    if ii.size:
        elab = labels[ii]
        order = np.argsort(elab, kind="stable")
        elab_sorted = elab[order]
        ii_sorted, jj_sorted = ii[order], jj[order]
        starts = np.searchsorted(elab_sorted, np.arange(n_comp), side="left")
        stops = np.searchsorted(elab_sorted, np.arange(n_comp), side="right")
        for c in range(n_comp):
            size = int(comp_sizes[c])
            lo, hi = starts[c], stops[c]
            if size < 3 or hi <= lo:
                continue
            n_e = hi - lo
            density_max = max(density_max, 2.0 * n_e / (size * (size - 1)))
            verts = np.flatnonzero(labels == c)
            remap = np.full(n, -1, dtype=np.int32)
            remap[verts] = np.arange(size, dtype=np.int32)
            sub_i, sub_j = remap[ii_sorted[lo:hi]], remap[jj_sorted[lo:hi]]
            if size <= dense_triangle_max:
                tri_by_comp[c] = _triangles_dense(sub_i, sub_j, size)
            else:
                tri_by_comp[c], exact = _triangles_sampled(
                    csr, verts, sub_i, sub_j, rng, triangle_sample
                )
                all_exact &= exact

    nontrivial = comp_sizes >= 2
    biggest = int(np.argmax(comp_sizes)) if n_comp else 0
    return ComponentMetrics(
        n_components=int(n_comp),
        comp_sizes=comp_sizes,
        labels=labels.astype(np.int32),
        giant_size=giant,
        n_triangles=int(tri_by_comp.sum()),
        n_triangles_exact=all_exact,
        n_l_violations=int(bad.sum()),
        n_l_violations_max_comp=int(bad_by_comp[biggest]) if n_comp else 0,
        n_triangles_max_comp=int(tri_by_comp[biggest]) if n_comp else 0,
        n_comp_n3o_free=int(np.count_nonzero((tri_by_comp == 0) & nontrivial)),
        n_comp_l_free=int(np.count_nonzero((bad_by_comp == 0) & nontrivial)),
        comp_edge_density_max=float(density_max),
    )


def cluster_weighted_p99(comp_sizes: np.ndarray, q: float = 0.99) -> float:
    """Size of the component holding the ``q``-quantile cluster (SPEC-AMENDMENTS D6).

    The spec's plain ``p99`` over components degenerates: with fewer than 100 components the 99th
    percentile *is* the maximum.  Weighting by cluster count asks the operationally meaningful
    question -- how big is the subproblem a randomly chosen cluster lands in.
    """
    sizes = np.asarray(comp_sizes, dtype=np.int64)
    if sizes.size == 0:
        return float("nan")
    order = np.sort(sizes)
    cum = np.cumsum(order)
    target = q * cum[-1]
    return float(order[min(int(np.searchsorted(cum, target, side="left")), order.size - 1)])
