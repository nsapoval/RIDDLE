"""B1's T1: ``k = 0`` is Than-Nakhleh, checked two independent ways.

The claim is a *reduction*, so testing it against another part of the solver proves nothing.  Two
oracles, neither of which shares a line with :mod:`opc.solve`:

**T1a -- the graph claim.**  MW-GTCS at ``k = 0`` is maximum-weight clique in the *compatibility*
graph, which is Than-Nakhleh's own formulation.  Checked against a branch-and-bound weighted
clique solver written here, on realistic components built from simulated gene trees with real MDC
weights.  It reaches components several times larger than the subset enumeration in
``test_ilp.py`` can, which is the point: brute force stops at about 20 vertices.

**T1b -- the phylogenetic claim.**  Than-Nakhleh's answer is the MDC species *tree*.  So the
``k = 0`` optimum must equal the best rooted binary tree that the ground set can express, scored
by counting deep coalescences directly on dendropy's gene trees -- no ``alpha_sum``, no telescoped
subset-sum, no bitsets.  Every topology on six taxa is enumerated, so this is exhaustive rather
than sampled.

**Where the gate stops being checkable.**  T1's wording is "all envelope cells".  An independent
clique solver cannot close the components that the ILP cannot close either -- maximum-weight
clique in the complement *is* the same problem -- so on the grid's hardest components there is no
oracle to compare against, only a second incumbent.  The reduction is verified where verification
is possible, and the boundary is recorded rather than papered over.
"""

from __future__ import annotations

import numpy as np
import pytest

from opc import bitset
from opc.clusters import extract
from opc.simulate import msnc_oracle
from opc.simulate import networks as N
from opc.solve import ilp, instance, overlap

# ------------------------------------------------------------------ an independent clique solver


def _max_weight_clique(adj: list[set[int]], w: np.ndarray) -> tuple[float, list[int]]:
    """Exact maximum-weight clique by branch and bound with a greedy-colouring bound.

    Colour classes are independent sets of the compatibility graph, so each can contribute at
    most its heaviest vertex -- summing those maxima over the classes bounds any clique inside
    the candidate set.  Written here, in plain adjacency sets, precisely so that it shares no
    code with the model under test.
    """
    best = [0.0, []]

    def bound(cand: list[int]) -> float:
        classes: list[list[int]] = []
        for v in cand:
            for cl in classes:
                if all(u not in adj[v] for u in cl):
                    cl.append(v)
                    break
            else:
                classes.append([v])
        return sum(max(w[u] for u in cl) for cl in classes)

    def expand(cand: list[int], cur: list[int], val: float) -> None:
        if val > best[0]:
            best[0], best[1] = val, list(cur)
        if not cand or val + bound(cand) <= best[0]:
            return
        order = sorted(cand, key=lambda v: -w[v])
        for i, v in enumerate(order):
            if val + bound(order[i:]) <= best[0]:
                return
            expand([u for u in order[i + 1:] if u in adj[v]], cur + [v], val + float(w[v]))

    expand(list(range(len(adj))), [], 0.0)
    return best[0], sorted(best[1])


def _instance_from_simulation(n_taxa: int, r: int, m: int, tau: int, seed: int, cu: float = 1.0):
    """``(words, weights)`` at ``tau``, through exactly the pipeline the grid uses."""
    base = N.build_base(n_taxa, r, seed=seed)
    net = N.with_gamma(N.scaled(N.with_reticulations(base, r), cu), 0.3)
    trees = msnc_oracle.simulate_gene_trees(net, m, seed=seed + 1)
    idx = {lab: i for i, lab in enumerate(net.labels)}
    counter = extract.SupportCounter(n_taxa, (m,))
    buf: list[int] = []
    for nwk in trees:
        buf.clear()
        counter.add_locus(extract.clusters_from_newick(nwk, idx, buf))
    inst = instance.build_instance(counter, n_taxa)
    words, support, alpha = inst.at(m)
    keep = np.flatnonzero(support >= tau)
    w = instance.weight_vector(
        alpha[keep], support[keep], bitset.popcounts(words[keep]), m, scheme="mdc",
    )
    return net, np.ascontiguousarray(words[keep]), w, trees


# ------------------------------------------------------------------------------------- T1a


@pytest.mark.parametrize("n_taxa,r,tau,seed", [(14, 2, 4, 17), (16, 2, 4, 41), (20, 2, 4, 7)])
def test_k_zero_is_maximum_weight_clique_in_the_compatibility_graph(n_taxa, r, tau, seed):
    """T1a.  Same objective *and* same accepted set, on real MDC weights."""
    net, words, w, _ = _instance_from_simulation(n_taxa, r, m=60, tau=tau, seed=seed)
    dec = overlap.decompose(words, n_taxa)
    checked = big = 0
    for comp in dec.components:
        if comp.n_vertices > 45:  # beyond this the oracle is no better off than the ILP
            continue
        local_w = w[comp.vertices]
        # The compatibility graph is the complement of the overlap graph on these vertices.
        over = [set() for _ in range(comp.n_vertices)]
        for a, b in zip(comp.ii, comp.jj):
            over[int(a)].add(int(b))
            over[int(b)].add(int(a))
        compat = [
            {u for u in range(comp.n_vertices) if u != v and u not in over[v]}
            for v in range(comp.n_vertices)
        ]
        want, want_set = _max_weight_clique(compat, local_w)

        sol = ilp.solve_component(comp, local_w, 0, time_limit=60)
        assert sol.optimal
        assert sol.objective == pytest.approx(want)
        if len({round(float(local_w[v]), 9) for v in range(comp.n_vertices)}) == comp.n_vertices:
            assert sorted(int(a) for a in sol.accepted) == want_set
        checked += 1
        big += comp.n_vertices > 20
    assert checked >= 1
    assert big >= 1, "every component was small enough for brute force -- T1a adds nothing here"


# ------------------------------------------------------------------------------------- T1b


def _rooted_topologies(leaves: tuple[int, ...]):
    """Every rooted binary topology on ``leaves``, as sets of internal-node leaf-sets."""
    if len(leaves) == 1:
        yield frozenset(), frozenset(leaves)
        return
    head, rest = leaves[0], leaves[1:]
    for mask in range(1 << len(rest)):
        left = (head,) + tuple(rest[i] for i in range(len(rest)) if mask >> i & 1)
        right = tuple(rest[i] for i in range(len(rest)) if not mask >> i & 1)
        if not right:
            continue
        for lc, lset in _rooted_topologies(left):
            for rc, rset in _rooted_topologies(right):
                yield lc | rc | {lset, rset}, lset | rset


def _lineages(node, cluster: frozenset) -> int:
    """Maximal subtrees of ``node`` whose leaf set fits inside ``cluster`` -- Maddison's count.

    Top-down over dendropy's own tree object: a subtree that fits contributes one lineage,
    otherwise the count is the sum over its children.  No parent links, no bitsets, no
    telescoping -- a different computation of the same quantity.
    """
    leaves = frozenset(int(lf.taxon.label[1:]) - 1 for lf in node.leaf_iter())
    if leaves <= cluster:
        return 1
    return sum(_lineages(ch, cluster) for ch in node.child_node_iter())


def test_k_zero_is_the_mdc_species_tree_by_direct_deep_coalescence_counting():
    """T1b.  Exhaustive over all 945 rooted topologies on six taxa.

    The ``than-nakhleh`` shift keeps every weight at least 1, so the optimum is a *maximal*
    compatible set -- a fully resolved tree -- and comparing against tree topologies is exactly
    the right comparison.  Under the handoff's ``m + 1 - sum alpha`` reading it would not be
    (A19): most weights go negative and the optimum stops resolving.
    """
    import dendropy

    # 0.5 CU: enough discordance that the ground set expresses 123 of the 945 topologies.  At
    # 1 CU it expresses 45, and the enumeration stops being a search.
    n_taxa, m, tau = 6, 60, 2
    net, words, w, newicks = _instance_from_simulation(n_taxa, 0, m=m, tau=tau, seed=5, cu=0.5)
    have = {
        frozenset(bitset.to_indices(words[i], n_taxa)): i for i in range(words.shape[0])
    }
    assert len(have) > n_taxa, "ground set too thin for the enumeration to be meaningful"

    gene_trees = [
        dendropy.Tree.get(data=nwk, schema="newick", rooting="force-rooted") for nwk in newicks
    ]

    #: alpha(A, T) = (lineages crossing A) - 1, summed over loci, counted on dendropy's trees.
    def alpha_sum(cluster: frozenset) -> int:
        return sum(_lineages(t.seed_node, cluster) - 1 for t in gene_trees)

    shift = int(max(alpha_sum(c) for c in have))
    best, best_clusters = -1.0, None
    considered = 0
    for clusters, _ in _rooted_topologies(tuple(range(n_taxa))):
        nontrivial = {c for c in clusters if 2 <= len(c) < n_taxa}
        if not nontrivial <= set(have):
            continue
        score = sum(shift + 1 - alpha_sum(c) for c in nontrivial)
        considered += 1
        if score > best:
            best, best_clusters = score, nontrivial
    assert considered > 50, f"only {considered} topologies were expressible -- test is thin"

    # The same weights the solver uses, re-derived from the independent count, so the two sides
    # differ only in how the optimum is found.
    w_indep = np.array([shift + 1 - alpha_sum(c) for c in have], dtype=float)
    order = np.array([have[c] for c in have])
    weights = np.zeros(words.shape[0])
    weights[order] = w_indep
    assert np.allclose(weights, w), "the two alpha computations disagree before any solving"

    dec = overlap.decompose(words, n_taxa)
    total = float(weights[dec.kernel].sum())
    accepted = set(dec.kernel.tolist())
    for comp in dec.components:
        sol = ilp.solve_component(comp, weights[comp.vertices], 0, time_limit=60)
        assert sol.optimal
        total += sol.objective
        accepted |= {int(comp.vertices[a]) for a in sol.accepted}

    assert total == pytest.approx(best)
    got = {frozenset(bitset.to_indices(words[i], n_taxa)) for i in accepted}
    assert got == best_clusters
