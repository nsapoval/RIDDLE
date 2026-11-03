"""``I(S*)``, its validation, and the Hasse diagram as a network (WS1-5).

The solver returns an accepted set ``S*``; the *answer* is the network it stands for.  By
Hellmuth-Schaller-Stadler that network is the Hasse diagram of the closed system ``I(S*)``, and
recovering it is three steps: close, take the cover relation of the inclusion order, and read the
result as a DAG in which a cluster with two covering parents is a hybrid.

Three details that are easy to get wrong, all of them load-bearing:

**The trivial clusters are mandatory.**  ``I(S*)`` alone is not enough -- ``F`` must also contain
the full taxon set and every singleton.  With the singletons present, any member of size >= 2 has
at least two covers (one cover would have to contain all of ``A``'s singletons, i.e. be ``A``), so
``H(F)`` has no unifurcations.  Without them the emitter produces degree-2 nodes and
``to_enewick`` renders a malformed tree.  This is SPEC-AMENDMENTS A9 arriving from the other side:
the closedness claim holds for the *full* system, not the non-trivial restriction.

**The hybrid cluster is added by the closure, not chosen by the solver.**  ``h = A & B`` is nested
inside both clusters it separates, so it overlaps neither and never appears in their overlap
component.  The ILP never has a variable for it; ``closure`` is where it enters.

**The isomorphism is to the *regular version* of the network.**  A hybrid node has indegree 2 and
outdegree 1, so its cluster equals its child's and the node-to-cluster map is not injective.
Contracting exactly the edges with ``c(u) == c(v)`` is that regularisation, which is why
:func:`hasse_covers` compares against ``{(c(u), c(v)) : c(u) != c(v)}`` and not against the raw
edge set.  ``tests/test_networks.py`` has asserted this since §8 test 2b; the implementation now
lives here so there is one of it.
"""

from __future__ import annotations

import numpy as np

from opc import bitset
from opc.simulate import networks as N

__all__ = [
    "closure_words",
    "hasse_covers",
    "hasse_covers_words",
    "with_trivial_clusters",
    "network_from_hasse",
    "reticulation_number",
]


def closure_words(words: np.ndarray, n_taxa: int) -> np.ndarray:
    """``I(S)`` on packed clusters: add ``A & B`` for every overlapping pair, to a fixpoint.

    Inside the target class one pass suffices -- no third member overlaps both of an overlapping
    pair under (N3O) -- but the loop is kept so this is also correct on a raw ``C_tau``, where it
    is a descriptive measurement rather than a theorem (SPEC-AMENDMENTS A10).
    """
    out = {bytes(row) for row in np.ascontiguousarray(words)}
    while True:
        cur = np.stack([np.frombuffer(b, dtype=np.uint64) for b in sorted(out)])
        sizes = bitset.popcounts(cur)
        dense = bitset.to_dense(cur, n_taxa)
        ic = dense @ dense.T
        overlapping = (ic > 0) & (ic < sizes[:, None]) & (ic < sizes[None, :])
        ii, jj = np.nonzero(np.triu(overlapping, k=1))
        if ii.size == 0:
            return cur
        new = {bytes(r) for r in (cur[ii] & cur[jj])} - out
        if not new:
            return cur
        out |= new


def with_trivial_clusters(words: np.ndarray, n_taxa: int) -> np.ndarray:
    """``F = words | {X} | {singletons}`` -- the object the Hasse diagram is taken of."""
    extra = [bitset.from_indices(range(n_taxa), n_taxa)]
    extra += [bitset.from_indices([t], n_taxa) for t in range(n_taxa)]
    have = {bytes(r) for r in np.ascontiguousarray(words)}
    rows = list(np.ascontiguousarray(words))
    rows += [e for e in extra if bytes(e) not in have]
    return np.stack(rows) if rows else np.zeros((0, bitset.n_words(n_taxa)), np.uint64)


def hasse_covers(clusters) -> set[tuple[frozenset, frozenset]]:
    """Covering pairs ``(bigger, smaller)`` of the inclusion order.  Reference implementation.

    ``O(k^3)`` in frozenset comparisons -- the oracle :func:`hasse_covers_words` is checked
    against, and what ``tests/test_networks.py`` uses.
    """
    cs = sorted(clusters, key=len)
    covers = set()
    for i, small in enumerate(cs):
        for big in cs[i + 1:]:
            if not small < big:
                continue
            if any(small < mid < big for mid in cs if mid is not small and mid is not big):
                continue
            covers.add((big, small))
    return covers


def hasse_covers_words(words: np.ndarray, n_taxa: int) -> tuple[np.ndarray, np.ndarray]:
    """``(parent, child)`` index arrays of the cover relation, via GEMM + transitive reduction.

    ``P[i, j]`` is "``c_j`` is a proper subset of ``c_i``", from the same exact float32
    intersection product the overlap kernel uses.  A pair is a *cover* when no third member sits
    strictly between, i.e. when ``(P @ P)[i, j] == 0``.
    """
    k = words.shape[0]
    if k == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    sizes = bitset.popcounts(words).astype(np.float32)
    dense = bitset.to_dense(words, n_taxa)
    ic = dense @ dense.T
    contains = (ic == sizes[None, :]) & (sizes[:, None] > sizes[None, :])  # c_j proper-sub c_i
    p = contains.astype(np.float32)
    between = (p @ p) > 0
    ii, jj = np.nonzero(contains & ~between)
    return ii.astype(np.int64), jj.astype(np.int64)


def reticulation_number(parent: np.ndarray, child: np.ndarray, n_nodes: int) -> int:
    """``|E| - |V| + 1`` on the Hasse diagram: the number of reticulations it carries.

    Computed from the cover relation alone, independently of any budget the solver spent, so it
    is a genuine cross-check on ``sum_K j_K`` rather than a restatement of it.
    """
    return int(parent.size - n_nodes + 1)


def network_from_hasse(words: np.ndarray, n_taxa: int, labels=None) -> N.Network:
    """Read ``H(F)`` as a :class:`~opc.simulate.networks.Network`.

    ``F`` must already be closed and carry the trivial clusters (:func:`with_trivial_clusters`).
    A cluster with two covering parents is a hybrid; one parent becomes its tree parent and the
    other its donor.  Node times are ``n_taxa - |c(v)|``, which is monotone along every edge --
    the Hasse diagram carries no real times, and any monotone rank keeps the edge lengths
    non-negative so the structure round-trips through ``to_enewick``.

    ``gamma`` is set to 0.5 and is **not identifiable from clusters**: the clustering system fixes
    the topology, never the inheritance proportions.
    """
    parent, child = hasse_covers_words(words, n_taxa)
    k = words.shape[0]
    sizes = bitset.popcounts(words)

    covers: dict[int, list[int]] = {v: [] for v in range(k)}
    for p, c in zip(parent, child):
        covers[int(c)].append(int(p))

    roots = [v for v in range(k) if not covers[v]]
    if len(roots) != 1 or int(sizes[roots[0]]) != n_taxa:
        raise ValueError(f"H(F) has {len(roots)} maximal elements; expected exactly the taxon set")
    for v in range(k):
        if len(covers[v]) > 2:
            raise ValueError(
                f"cluster {sorted(bitset.to_indices(words[v], n_taxa))} has {len(covers[v])} "
                "covering parents; H(F) is not level-1"
            )

    labels = list(labels) if labels is not None else [f"t{i + 1}" for i in range(n_taxa)]
    net = N.Network(
        n_taxa=n_taxa, labels=labels,
        time=[float(n_taxa - int(sizes[v])) for v in range(k)],
        tree_parent=[-1] * k, tree_children=[[] for _ in range(k)],
        leaf_taxon={}, horiz_child={}, donor_of={}, retics=[], root=int(roots[0]),
    )
    for v in range(k):
        if int(sizes[v]) == 1:
            net.leaf_taxon[v] = bitset.to_indices(words[v], n_taxa)[0]

    # Order hybrids deterministically by their cluster, so the output does not depend on the
    # row order of `words`.
    order = sorted(range(k), key=lambda v: bitset.to_indices(words[v], n_taxa))
    for v in order:
        ps = sorted(covers[v], key=lambda p: bitset.to_indices(words[p], n_taxa))
        if not ps:
            continue
        if len(ps) == 1:
            net.tree_parent[v] = ps[0]
            net.tree_children[ps[0]].append(v)
            continue
        # A node can donate to at most one hybrid; prefer a free donor.
        free = [p for p in ps if p not in net.horiz_child]
        if not free:
            raise ValueError("both covering parents already donate; H(F) is not level-1")
        donor = free[-1]
        tree = ps[0] if ps[0] != donor else ps[1]
        net.tree_parent[v] = tree
        net.tree_children[tree].append(v)
        net.horiz_child[donor] = v
        net.donor_of[v] = donor
        net.retics.append(N.Reticulation(v, donor, len(net.retics), net.time[v]))
    return net
