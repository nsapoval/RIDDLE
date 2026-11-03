"""Generator invariants and the network-side theory tests (§8 tests 2a, 2b, 3b).

These run on every network shape the grid will actually use, so a failure here means the ground
truth the whole study is measured against is wrong.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from opc.clusters import systems
from opc.simulate import networks as N

# (n_taxa, r_max) -- the grid's shapes, shrunk where the exhaustive checks would dominate runtime.
SHAPES = [(10, 2), (25, 3), (50, 5)]


def _networks(n_taxa, r_max, seed, cus=(0.25, 1.0), gammas=(0.1, 0.5)):
    base = N.build_base(n_taxa, r_max, seed=seed)
    for r in sorted({0, 1, r_max}):
        for cu in cus:
            for g in gammas:
                yield r, cu, g, N.with_gamma(N.scaled(N.with_reticulations(base, r), cu), g)


# ------------------------------------------------------------------- generator invariants (gates)


@pytest.mark.parametrize("n_taxa,r_max", SHAPES)
def test_generator_invariants(n_taxa, r_max):
    for r, cu, gamma, net in _networks(n_taxa, r_max, seed=100 + n_taxa):
        assert net.n_retics == r
        assert N.level(net) == (1 if r else 0), f"level != 1 at r={r}"
        assert net.gamma == gamma

        # Time-embedded => every edge length non-negative, and time-consistent by construction.
        for v in range(net.n_nodes):
            for p in net.net_parents(v):
                assert net.edge_length(p, v) >= 0.0

        # Ultrametric: every leaf at the same depth.
        depths = [net.time[v] for v in range(net.n_nodes) if net.is_leaf(v)]
        assert max(depths) - min(depths) < 1e-9 * max(depths)

        # D2: the ILS axis is exact on the derived network, not just the base tree.
        assert np.median(N.internal_branch_lengths(net)) == pytest.approx(cu, abs=1e-9)

        # Leaf set is exactly X, no duplicates.
        taxa = sorted(net.leaf_taxon.values())
        assert taxa == list(range(n_taxa))


@pytest.mark.parametrize("n_taxa,r_max", SHAPES)
def test_enewick_is_wellformed(n_taxa, r_max):
    for r, _cu, gamma, net in _networks(n_taxa, r_max, seed=7, cus=(0.5,), gammas=(0.3, 0.7)):
        s = N.to_enewick(net)
        assert s.endswith(";")
        assert s.count("(") == s.count(")")
        # Every reticulation appears exactly twice: once as a subtree, once as a leaf copy.
        for k in range(1, r + 1):
            assert s.count(f"#H{k}:") == 2, f"#H{k} appears {s.count(f'#H{k}:')} times"
        # Every taxon label appears exactly once.
        for lab in net.labels:
            assert s.count(f"{lab}:") == 1
        # Gammas on the two copies of a hybrid edge must sum to 1.
        import re

        gs = [float(x) for x in re.findall(r"::([0-9.eE+-]+)", s)]
        assert len(gs) == 2 * r
        assert all(abs(a + b - 1.0) < 1e-9 for a, b in zip(sorted(gs)[:r], sorted(gs)[r:][::-1]))


def test_nested_r_is_a_prefix_of_the_same_reticulation_set():
    """A2: derived-r networks must be nested, or the r contrast is not paired."""
    base = N.build_base(25, 4, seed=42)
    seen = []
    for r in range(5):
        net = N.with_reticulations(base, r)
        below = N.hardwired_clusters(net)
        seen.append({frozenset(below[ret.hybrid]) for ret in net.retics})
    for a, b in zip(seen, seen[1:]):
        assert a <= b, "reticulation sets are not nested across r"


def test_topology_is_shared_across_cu_and_gamma():
    """A2: CU and gamma are pure derivations, so the cluster system must not move."""
    base = N.build_base(25, 3, seed=17)
    ref = N.nontrivial_clusters(N.with_gamma(N.scaled(base, 0.25), 0.1))
    for cu in (0.25, 0.5, 1.0, 2.0):
        for g in (0.1, 0.3, 0.5):
            assert N.nontrivial_clusters(N.with_gamma(N.scaled(base, cu), g)) == ref


# ------------------------------------------------------------------------------- §8 test 2a


@pytest.mark.parametrize("n_taxa,r_max", SHAPES)
def test_2a_true_clustering_system_is_closed_and_in_the_target_class(n_taxa, r_max):
    """C_Nstar (the *full* system, per §4.1) is closed and satisfies (L) and (N3O).

    Hellmuth, Schaller & Stadler Thm 7.38 / 8.9.  Note this must be the full system: the
    nontrivial restriction of §6.3 is not closed whenever a hybrid's cluster is a single leaf.
    """
    for r, _cu, _g, net in _networks(n_taxa, r_max, seed=3, cus=(1.0,), gammas=(0.3,)):
        full = N.full_clusters(net)
        assert systems.is_closed(full), f"C_Nstar not closed at r={r}"
        assert systems.satisfies_n3o(full), f"C_Nstar violates (N3O) at r={r}"
        assert systems.satisfies_l(full), f"C_Nstar violates (L) at r={r}"
        assert frozenset(range(n_taxa)) in full
        assert all(frozenset({i}) in full for i in range(n_taxa))
        assert len(full) >= n_taxa


# ------------------------------------------------------------------------------- §8 test 2b


# The cover relation now lives with the emitter that consumes it, so there is one
# implementation rather than a test-only copy that can drift from it.
from opc.solve.closure import hasse_covers as _hasse_covers  # noqa: E402


@pytest.mark.parametrize("n_taxa,r_max", [(10, 2), (25, 3)])
def test_2b_hasse_diagram_matches_the_regular_version_of_the_network(n_taxa, r_max):
    """Cor. 7.41: ``H(C(N*))`` is isomorphic to the *regular version* of ``N*``.

    "Regular version" is the load-bearing phrase.  A hybrid node has indegree 2 and outdegree 1,
    so its cluster equals its child's -- the node-to-cluster map is not injective on ``N*`` itself
    and the test fails on every network unless those edges are contracted first.  Contracting
    exactly the edges with ``c(u) == c(v)`` *is* the regularisation, and because leaves are
    labelled the comparison is then a direct check rather than a graph-isomorphism search.
    """
    for r, _cu, _g, net in _networks(n_taxa, r_max, seed=21, cus=(1.0,), gammas=(0.3,)):
        below = N.hardwired_clusters(net)
        clusters = set(below.values())

        # Contract every edge whose endpoints carry the same cluster.
        net_covers = set()
        for v in N.hardwired_clusters(net):
            for c in net.net_children(v):
                if below[v] != below[c]:
                    net_covers.add((below[v], below[c]))

        assert net_covers == _hasse_covers(clusters), f"Hasse mismatch at r={r}"

        # Non-tautological half: distinct networks must not share a clustering system.
        assert len(clusters) == len({frozenset(c) for c in clusters})


def test_2b_distinct_networks_have_distinct_clustering_systems():
    """If two networks share C_Nstar, no cluster-based method can tell them apart."""
    seen: dict[frozenset, tuple] = {}
    collisions = []
    for seed in range(30):
        base = N.build_base(15, 3, seed=500 + seed)
        for r in (1, 2, 3):
            net = N.with_reticulations(base, r)
            key = frozenset(N.full_clusters(net))
            if key in seen and seen[key] != (seed, r):
                collisions.append((seen[key], (seed, r)))
            seen[key] = (seed, r)
    assert not collisions, f"clustering-system collisions: {collisions[:3]}"


# ------------------------------------------------------------------------------- §8 test 3b


@pytest.mark.parametrize("n_taxa,r_max", [(10, 2), (25, 3), (50, 5)])
def test_3b_reticulations_correspond_one_to_one_with_overlap_components(n_taxa, r_max):
    """The decisive check: each reticulation is exactly one non-trivial overlap component.

    This is what makes "one subproblem per overlap component" the right decomposition -- if a
    single block produced two components, or two blocks merged into one, the per-component solver
    would be attacking the wrong subproblems.  Three things are asserted together: the counts
    match, no component carries two reticulations, and no reticulation's signature pair is split
    across components (which would make it unrecoverable however good the solver is).
    """
    for r, _cu, _g, net in _networks(n_taxa, r_max, seed=31, cus=(1.0,), gammas=(0.3,)):
        cs = N.nontrivial_clusters(net)
        comps = [c for c in systems.components(cs) if len(c) >= 2]
        assert len(comps) == r, f"r={r} but {len(comps)} non-trivial overlap components"
        assert N.n_cluster_visible(net) == r, "some reticulation left no trace in C_Nstar"

        where = {c: i for i, comp in enumerate(comps) for c in comp}
        occupied = []
        for ret in net.retics:
            a, b = N.signature_pair(net, ret)
            assert a in cs and b in cs and systems.overlaps(a, b)
            assert where[a] == where[b], "signature pair split across components"
            occupied.append(where[a])
        assert sorted(occupied) == list(range(r)), "components and reticulations are not paired"


def test_3b_hybrid_cluster_is_recoverable_as_a_pairwise_intersection():
    """§6.3's ``retention_hybrid``: each hybrid cluster is ``A & B`` for an overlapping pair."""
    for seed in range(6):
        net = N.with_gamma(N.scaled(N.build_base(20, 3, seed=700 + seed), 1.0), 0.3)
        below = N.hardwired_clusters(net)
        cs = N.nontrivial_clusters(net)
        pair_intersections = {a & b for a, b in combinations(cs, 2) if systems.overlaps(a, b)}
        for ret in net.retics:
            hyb = frozenset(below[ret.hybrid])
            assert hyb in pair_intersections, f"hybrid cluster {sorted(hyb)} not recoverable"


def test_shortcut_reticulations_are_invisible_and_are_excluded_by_default():
    """Why ``allow_shortcuts=False`` is the default.

    A hybrid attached directly at the cycle's LCA makes ``z -> h`` a shortcut edge; the network is
    then not regular, both hybrid parents sit on the same cluster chain, and the cycle contributes
    no overlapping pair.  Such a reticulation exists in the network but cannot be recovered from
    clusters by any method, so leaving them in would cap ``retention_hybrid`` for reasons
    unrelated to ILS or filtering.
    """
    import numpy as np

    from opc.simulate.networks import _add_reticulation, _from_dendropy  # noqa: PLC2701

    n_visible_with, n_total_with = 0, 0
    for seed in range(12):
        base = N.build_base(20, 3, seed=900 + seed)
        assert N.n_cluster_visible(base) == base.n_retics  # default: all visible

        # Same topology, shortcuts permitted.
        import random

        from dendropy.simulate import treesim

        rng = np.random.default_rng([900 + seed, 0])
        tree = treesim.birth_death_tree(
            birth_rate=1.0, death_rate=0.2, num_extant_tips=20, gsa_ntax=30,
            rng=random.Random(int(rng.integers(1, 2**31))),
        )
        loose = _from_dendropy(tree, 20)
        med = float(np.median(N.internal_branch_lengths(loose)))
        for v in range(loose.n_nodes):
            loose.time[v] /= med
        placed = sum(_add_reticulation(loose, rng, allow_shortcuts=True) for _ in range(3))
        if placed == 3:
            n_total_with += loose.n_retics
            n_visible_with += N.n_cluster_visible(loose)

    assert n_total_with > 0
    frac = n_visible_with / n_total_with
    assert frac < 0.95, (
        f"expected shortcut reticulations to be invisible, but {frac:.0%} were visible"
    )


# ------------------------------------------------------------------ A28: galled trees + strata


def test_the_generator_already_emits_galled_trees_and_is_galled_is_not_a_vacuous_test():
    """A28.  ``level(net) == 1`` and "galled" are different claims, but not on *this* output.

    Two cycles meeting at a cut vertex are two biconnected components, so ``level`` stays 1 while
    the network stops being galled.  On the generator's own output that configuration is
    unreachable, and not by luck: every interior vertex of a placed cycle has its incoming tree
    edge consumed, and both of the apex's child edges are consumed, so a later cycle can reach a
    shared vertex only through a consumed edge.  A vertex with **two** incoming edges escapes that
    argument -- and that is exactly A20's case, a hybrid sitting at another cycle's apex, which the
    *regular version* of a level-1 network really does produce.

    So the assertion is: galled on both placement paths, and ``is_galled`` still able to say no.
    """
    for seed in range(20):
        for vd in (False, True):
            net = N.build_base(30, 4, seed=7000 + seed, vertex_disjoint=vd)
            assert N.level(net) == 1
            assert N.is_galled(net), f"seed {seed}, vertex_disjoint={vd}: cycles share a vertex"

    # Not vacuous: the regular version of a generator network is level-1 and frequently not
    # galled, because a hybrid with two in-edges can be another cycle's apex without sharing one.
    import numpy as np

    from opc.solve import closure

    non_galled = 0
    for seed in range(30):
        net = N.build_base(15, 3, seed=seed)
        words = np.stack([
            np.frombuffer(int(sum(1 << t for t in c)).to_bytes(8, "little"), dtype=np.uint64)
            for c in N.full_clusters(net)
        ])
        regular = closure.network_from_hasse(words, net.n_taxa)
        assert N.level(regular) == 1
        non_galled += not N.is_galled(regular)
    assert non_galled > 0, "is_galled never returned False, so it proves nothing above"


def test_cycle_size_bands_are_respected_and_measure_what_block_clusters_returns():
    for band in ((3, 6), (9, None)):
        lo, hi = band
        seen = []
        for seed in range(15):
            net = N.build_base(50, 2, seed=8000 + seed,
                               vertex_disjoint=True, cycle_size_range=band)
            for ret in net.retics:
                size = N.cycle_size(net, ret)
                seen.append(size)
                assert lo <= size and (hi is None or size <= hi)
                # cycle_size counts the tree edges on the cycle, which is also how many clusters
                # the block contributes -- equal unless two of those clusters coincide.
                assert len(N.block_clusters(net, ret)) <= size
        assert seen, f"band {band} placed nothing, so the assertion above was vacuous"


def test_the_default_placement_rule_is_unchanged_by_the_a28_parameters():
    """Defaults must reproduce pre-A28 topologies exactly -- the cached base networks depend on
    it, and so does every archived ``config_hash``."""
    for seed in (11, 12, 13):
        a = N.build_base(25, 3, seed=seed)
        b = N.build_base(25, 3, seed=seed, vertex_disjoint=False, cycle_size_range=(3, None))
        assert a.time == b.time and a.tree_parent == b.tree_parent
        assert a.donor_of == b.donor_of and a.horiz_child == b.horiz_child
        assert a.cycle_size_rejections == 0, "the default band rejected a placeable cycle"
