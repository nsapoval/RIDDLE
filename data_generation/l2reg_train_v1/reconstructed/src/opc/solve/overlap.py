"""``O(C_tau)`` decomposed into the subproblems the solver actually sees (WS1-2).

A thin driver over :mod:`opc.clusters.relations` -- no new overlap kernel, no second
component-finder.  What it adds is the per-component view the ILP needs: local vertex indexing,
the candidate hybrid values on each edge, the ``z`` variables those imply, and the triangle rows.

Two things the handoff spec gets wrong about this stage.

**A component is not a block.**  The spec's feasibility condition reads as though ``O(C_tau)``
arrives already decomposed into blocks with one hybrid each.  It does not: the number of distinct
edge intersections in a component runs 1.1-1.5x its vertex count, so a component is a
*superposition* of candidate blocks and choosing at most ``j`` of them is precisely the ILP's job.

**Only monochromatic triangles are needed.**  Under (L) every triangle of a feasible system has
all three pairwise intersections equal, and a triangle whose intersections differ is already cut
at integrality by the assignment rows.  So :func:`monochromatic_triangles` enumerates within an
intersection class rather than over the whole component -- 10-13% of the triangles, and on the
grid's largest envelope component that is 1,691 rather than 17,037.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc import bitset
from opc.clusters import relations

__all__ = ["Component", "Decomposition", "decompose", "monochromatic_triangles",
           "monochromatic_cliques", "edge_clique_cover"]


@dataclass
class Component:
    """One connected component of ``O(C_tau)``, indexed locally from zero."""

    vertices: np.ndarray  # int32[V] -- rows of the filtered ground set, ascending
    ii: np.ndarray  # int32[E] local endpoints, ii < jj
    jj: np.ndarray
    edge_class: np.ndarray  # int64[E] -- index into hcand
    hcand: np.ndarray  # uint64[D, W] -- the distinct A & B values realised here
    z_vertex: np.ndarray  # int64[Z] \ the (A, h) pairs an ILP would create
    z_class: np.ndarray  # int64[Z] /

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.size)

    @property
    def n_edges(self) -> int:
        return int(self.ii.size)

    @property
    def n_hcand(self) -> int:
        return int(self.hcand.shape[0])

    @property
    def n_z(self) -> int:
        return int(self.z_vertex.size)

    def model_size(self, n_triangles: int) -> dict:
        """Binary count and row count of the ILP this component would produce."""
        return {
            "binaries": self.n_vertices + self.n_z + self.n_hcand,
            "rows": self.n_vertices + 2 * self.n_edges + self.n_z + 1 + n_triangles,
        }


@dataclass
class Decomposition:
    """``O(C_tau)`` split into the kernel and the non-trivial components."""

    n_clusters: int
    kernel: np.ndarray  # int32[] -- isolated vertices, accepted unconditionally
    components: list[Component]

    @property
    def kernel_fraction(self) -> float:
        return self.kernel.size / self.n_clusters if self.n_clusters else float("nan")


def decompose(words: np.ndarray, n_taxa: int, **kwargs) -> Decomposition:
    """Build ``O(C_tau)`` and split it.

    ``words`` must be deduplicated, as :func:`relations.build_overlap_graph` requires.  A cell
    that trips the edge cap has no edge list at all, so it cannot be decomposed -- the caller is
    told rather than handed a silently empty answer.
    """
    graph = relations.build_overlap_graph(words, n_taxa, **kwargs)
    if graph.dense_bailout:
        raise RuntimeError(
            "overlap graph exceeded the edge cap; no edge list to decompose "
            f"({graph.n_edges:,} edges over {graph.n_vertices:,} clusters)"
        )
    hcand_all, edge_class_all = relations.edge_intersections(graph)

    n = graph.n_vertices
    labels = np.full(n, -1, dtype=np.int64)
    if graph.ii.size:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components as _cc

        adj = sp.coo_matrix(
            (np.ones(graph.ii.size, np.int8), (graph.ii, graph.jj)), shape=(n, n)
        )
        _, labels = _cc(adj, directed=False)
        labels = labels.astype(np.int64)
    else:
        labels = np.arange(n, dtype=np.int64)

    degrees = graph.degrees
    kernel = np.flatnonzero(degrees == 0).astype(np.int32)

    components: list[Component] = []
    if graph.ii.size:
        order = np.argsort(labels[graph.ii], kind="stable")
        elab = labels[graph.ii][order]
        ii_s, jj_s, ec_s = graph.ii[order], graph.jj[order], edge_class_all[order]
        bounds = np.flatnonzero(np.diff(elab)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, elab.size]):
            verts = np.flatnonzero(labels == elab[lo]).astype(np.int32)
            remap = np.full(n, -1, dtype=np.int32)
            remap[verts] = np.arange(verts.size, dtype=np.int32)
            li, lj = remap[ii_s[lo:hi]], remap[jj_s[lo:hi]]
            # Renumber the intersection classes densely inside this component.
            used, local_class = np.unique(ec_s[lo:hi], return_inverse=True)
            zv, zc = relations.assignment_pairs(li, lj, local_class, verts.size)
            components.append(Component(
                vertices=verts, ii=li, jj=lj, edge_class=np.asarray(local_class).ravel(),
                hcand=np.ascontiguousarray(hcand_all[used]), z_vertex=zv, z_class=zc,
            ))
    return Decomposition(n_clusters=n, kernel=kernel, components=components)


def monochromatic_triangles(comp: Component) -> np.ndarray:
    """``int32[T, 3]`` of local vertex triples whose three edges share one intersection value.

    Enumerated per intersection class.  A triangle with two different intersections is an (L)
    violation that the assignment rows already forbid at integrality, so adding it as an (N3O)
    row would be redundant; a monochromatic one satisfies (L) and violates (N3O), which is
    exactly the case that needs its own constraint.  ``{0,1}, {0,2}, {0,3}`` -- the sunflower --
    is the canonical example, and it is why (L) does not imply (N3O).
    """
    if comp.n_edges == 0:
        return np.zeros((0, 3), dtype=np.int32)
    out: list[tuple[int, int, int]] = []
    for c in range(comp.n_hcand):
        sel = np.flatnonzero(comp.edge_class == c)
        if sel.size < 3:
            continue
        ii, jj = comp.ii[sel], comp.jj[sel]
        nodes = np.unique(np.concatenate([ii, jj]))
        idx = {int(v): k for k, v in enumerate(nodes)}
        adj: list[set[int]] = [set() for _ in nodes]
        for a, b in zip(ii, jj):
            adj[idx[int(a)]].add(idx[int(b)])
            adj[idx[int(b)]].add(idx[int(a)])
        for a, b in zip(ii, jj):
            u, v = (int(a), int(b)) if a < b else (int(b), int(a))
            for w in adj[idx[u]] & adj[idx[v]]:
                if int(nodes[w]) > v:
                    out.append((u, v, int(nodes[w])))
    if not out:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(sorted(set(out)), dtype=np.int32)


def monochromatic_cliques(comp: Component) -> list[tuple[np.ndarray, int]]:
    """Maximal monochromatic cliques covering every monochromatic triangle.

    A clique ``Q`` whose edges all meet in the same ``h`` admits at most two accepted members --
    three would be an (N3O) violation -- and two accepted members force ``y[h] = 1``.  So one row
    ``sum_{A in Q} x[A] <= 1 + y[h]`` subsumes all ``C(|Q|, 3)`` triangle rows inside ``Q`` and is
    strictly stronger: unlike ``sum x <= |Q| - 1`` it cuts the all-halves LP point, which is the
    only thing that moves the bound on these instances.

    Grown greedily from each uncovered triangle rather than enumerated exhaustively -- a maximal
    clique cover is all the constraint set needs, and Bron-Kerbosch on a dense class subgraph is
    not worth its cost here.
    """
    out: list[tuple[np.ndarray, int]] = []
    for c in range(comp.n_hcand):
        sel = np.flatnonzero(comp.edge_class == c)
        if sel.size < 3:
            continue
        ii, jj = comp.ii[sel], comp.jj[sel]
        nodes = np.unique(np.concatenate([ii, jj]))
        adj: dict[int, set[int]] = {int(v): set() for v in nodes}
        for a, b in zip(ii, jj):
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

        covered: set[tuple[int, ...]] = set()
        for a, b in zip(ii, jj):
            u, v = (int(a), int(b)) if a < b else (int(b), int(a))
            for w in sorted(adj[u] & adj[v]):
                if w <= v:
                    continue
                tri = (u, v, w)
                if tri in covered:
                    continue
                # Grow the triangle to a maximal clique inside this class.
                q = [u, v, w]
                cand = adj[u] & adj[v] & adj[w]
                while cand:
                    nxt = min(cand)
                    q.append(nxt)
                    cand &= adj[nxt]
                q.sort()
                for i in range(len(q)):
                    for j in range(i + 1, len(q)):
                        for k in range(j + 1, len(q)):
                            covered.add((q[i], q[j], q[k]))
                out.append((np.array(q, dtype=np.int32), c))
    return out


def edge_clique_cover(comp: Component) -> list[np.ndarray]:
    """Maximal cliques of the whole component covering every edge, ignoring intersection classes.

    At ``j = 0`` every overlapping pair is forbidden, so a clique admits **at most one** accepted
    member and ``sum_{A in Q} x[A] <= 1`` replaces all ``C(|Q|, 2)`` edge rows.  This is the
    standard clique reformulation of maximum-weight independent set, and it is much tighter: the
    edge formulation's LP relaxation has the all-halves point at half the total weight, which a
    clique row of size ``q`` cuts down to ``1/q``.
    """
    adj: dict[int, set[int]] = {v: set() for v in range(comp.n_vertices)}
    for a, b in zip(comp.ii, comp.jj):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    covered: set[tuple[int, int]] = set()
    out: list[np.ndarray] = []
    # Densest vertices first: they seed the largest cliques, so fewer rows cover the same edges.
    for a, b in sorted(zip(comp.ii, comp.jj),
                       key=lambda e: -(len(adj[int(e[0])]) + len(adj[int(e[1])]))):
        u, v = (int(a), int(b)) if a < b else (int(b), int(a))
        if (u, v) in covered:
            continue
        q = [u, v]
        cand = adj[u] & adj[v]
        while cand:
            nxt = max(cand, key=lambda x: len(adj[x] & cand))
            q.append(nxt)
            cand &= adj[nxt]
        q.sort()
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                covered.add((q[i], q[j]))
        out.append(np.array(q, dtype=np.int32))
    return out


def hcand_in_ground_set(comp: Component, words: np.ndarray) -> np.ndarray:
    """Which candidate hybrids are themselves members of ``C_tau``.

    Reported, never enforced.  The handoff spec proposes constraining ``x[h] >= y[h]``, but
    ``h`` is nested inside both clusters it separates, so it overlaps neither and is provably
    never a vertex of the component it labels -- the constraint would reference a variable that
    is not in the model, and 88% of blocks in feasible systems have ``h`` outside the accepted
    set entirely.  The fraction is worth recording because it bounds what the spec's alternative
    "recoverable ``h``" design could ever have bought.
    """
    if comp.n_hcand == 0:
        return np.zeros(0, dtype=bool)
    present = {bytes(row) for row in np.ascontiguousarray(words)}
    return np.array([bytes(h) in present for h in np.ascontiguousarray(comp.hcand)], dtype=bool)


def size_report(dec: Decomposition, words: np.ndarray) -> list[dict]:
    """One row per component: what the ILP will actually be asked to solve."""
    rows = []
    for k, comp in enumerate(dec.components):
        tri = monochromatic_triangles(comp)
        rows.append({
            "component": k,
            "n_vertices": comp.n_vertices,
            "n_edges": comp.n_edges,
            "n_hcand": comp.n_hcand,
            "n_hcand_in_ctau": int(hcand_in_ground_set(comp, words[comp.vertices]).sum()),
            "n_hcand_singleton": int(
                (bitset.popcounts(comp.hcand) == 1).sum()
            ),
            "n_z": comp.n_z,
            "n_triangles_mono": int(tri.shape[0]),
            **comp.model_size(int(tri.shape[0])),
        })
    return rows
