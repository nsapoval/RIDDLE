"""E3's enumeration: is it exhaustive, is it galled, and does it agree with the network path?

Everything downstream of :mod:`opc.theory.galled` -- the fiber distribution, gate G3, and E4's
brute-force validation -- is a statement about an enumeration being *complete and correct*.  So the
tests here are not about the API; they are about those two words.

Three independent implementations meet in this file, which is the point:

* the fast cluster-level enumeration under test;
* :func:`_closed_form_k1`, path arithmetic on the base tree, written here and nowhere else, which
  says what a one-reticulation network's clusters and displayed trees must be;
* the *network* path -- ``theory.galled.to_network`` into
  :mod:`opc.simulate.networks` and :mod:`opc.mpl.msnc3`, code that predates E3 entirely.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from opc import bitset
from opc.mpl import msnc3
from opc.simulate import networks as N
from opc.solve import closure
from opc.theory import galled as G


def _triples_of_masks(masks, n_taxa: int) -> int:
    table = G.triple_bits(n_taxa)
    bits = 0
    for m in masks:
        bits |= table[m]
    return bits


def _closed_form_k1(tree: G.BaseTree, cy: G.Cycle) -> tuple[int, int]:
    """``(system, triples)`` for a single reticulation, by path arithmetic on the base tree.

    An independent implementation of what :func:`opc.theory.galled.instance` computes with a DAG
    traversal, and deliberately a *different* argument.  Adding ``d -> r``:

    * every strict ancestor of ``d`` below ``z`` gains ``c(r)``, and the new node ``p_d`` above
      ``d`` has cluster ``c(d) | c(r)``.  Nothing else moves, so that is ``C(Psi)``;
    * the displayed tree that takes the tree edge is ``T`` itself;
    * the one that takes the donor edge additionally *removes* ``c(r)`` from every strict ancestor
      of ``r`` below ``z``, because the clade moved out from under them.

    ``instance`` does not use this because its ``k = 2`` composition would need an argument about
    the two cycles not interfering, which is exactly the kind of thing this repo checks rather than
    argues.
    """
    cr = tree.clus[cy.hybrid]
    dp, hp = set(cy.donor_path), set(cy.hybrid_path)

    hard = {tree.clus[v] | cr if v in dp else tree.clus[v] for v in range(tree.n_nodes)}
    hard.add(tree.clus[cy.donor] | cr)

    donor_side = {tree.clus[cy.donor] | cr}
    for v in range(tree.n_nodes):
        if v in dp:
            donor_side.add(tree.clus[v] | cr)
        elif v in hp:
            donor_side.add(tree.clus[v] & ~cr)
        else:
            donor_side.add(tree.clus[v])

    system = 0
    for m in hard:
        system |= 1 << m
    return system, tree.triples | _triples_of_masks(donor_side, tree.n_taxa)


def _system_words(system: int, n_taxa: int) -> np.ndarray:
    return np.stack([
        bitset.from_indices([t for t in range(n_taxa) if m >> t & 1], n_taxa)
        for m in sorted(G.decode_system(system, n_taxa))
    ])


def _triples_via_network(net, n_taxa: int) -> int:
    bits = 0
    for choice in itertools.product((False, True), repeat=net.n_retics):
        cur = net
        for i in range(net.n_retics - 1, -1, -1):
            cur = msnc3.resolve_one_reticulation(cur, i, take_donor=choice[i])
        bits |= _triples_of_masks(
            (sum(1 << t for t in c) for c in N.hardwired_clusters(cur).values()), n_taxa)
    return bits


# ------------------------------------------------------------------------------ the base trees


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_every_labelled_rooted_binary_tree_appears_exactly_once(n):
    """``(2n-3)!!`` of them, and no duplicates -- both halves, because the split rule that avoids
    double-counting is the only thing standing between this and an enumeration that is not one."""
    trees = list(G.rooted_binary_trees(n))
    assert len(trees) == G.n_rooted_binary_trees(n)
    assert len({t.system for t in trees}) == len(trees)


def test_a_trees_clusters_and_triples_agree_with_the_network_path():
    for n in (5, 6):
        for tree in G.rooted_binary_trees(n):
            net = G.to_network(tree, ())
            want = 0
            for c in N.full_clusters(net):
                want |= 1 << sum(1 << t for t in c)
            assert want == tree.system
            assert _triples_via_network(net, n) == tree.triples
            # A rooted binary tree resolves every triple exactly once.
            assert bin(tree.triples).count("1") == len(G.triple_slots(n))


# -------------------------------------------------------------------------------- placements


def test_a_placement_is_never_a_shortcut_and_always_leaves_an_overlapping_pair():
    """A11: a reticulation whose donor already reaches its hybrid leaves no trace in ``C(N*)``.

    Disjoint clusters is what forbids it here, and the trace it must leave instead is
    ``networks.signature_pair`` -- the donor's cluster and the cluster of the hybrid's tree parent,
    which have to *overlap* and to meet in exactly the hybrid's own cluster.  That is the
    reticulation as a cluster-based method sees it, and it is the property A11 found missing on
    33-53% of the unconstrained generator's placements.

    ``n_cluster_visible`` is the S50 generator's version of the same counter, and it must be 1 on
    *every* placement with no exceptions -- that is what makes the enumeration an enumeration of
    regular networks.
    """
    from opc.clusters.systems import overlaps

    seen = 0
    for tree in G.rooted_binary_trees(6):
        for cy in G.cycles_of(tree):
            assert tree.clus[cy.donor] & tree.clus[cy.hybrid] == 0
            net = G.to_network(tree, (cy,))
            a, b = N.signature_pair(net, net.retics[0])
            assert overlaps(a, b)
            assert a & b == frozenset(t for t in range(6) if tree.clus[cy.hybrid] >> t & 1)
            assert N.n_cluster_visible(net) == 1
            seen += 1
    assert seen > 0


def test_the_excluded_shortcut_placements_are_the_ones_a11_is_about():
    """The rule is only worth having if what it excludes is really invisible.  Measured, not
    assumed.

    Every placement ``cycles_of`` drops is rebuilt here and shown to fail the property E3 depends
    on: the reticulation number read off the Hasse diagram of its clustering system comes back
    **0**, so the clustering system is that of a *tree* and the network is outside the
    correspondence entirely.  The size-2 cycles go further and reproduce the base tree's clustering
    system exactly.
    """
    n = 5
    dropped = flat = identical = 0
    for tree in G.rooted_binary_trees(n):
        kept = {(c.donor, c.hybrid) for c in G.cycles_of(tree)}
        for cy in G.cycles_of(tree, allow_shortcuts=True):
            if (cy.donor, cy.hybrid) in kept:
                continue
            dropped += 1
            assert cy.hybrid_path == ()
            inst = G.instance(tree, (cy,))
            words = closure.with_trivial_clusters(
                closure.closure_words(_system_words(inst.system, n), n), n)
            p, c = closure.hasse_covers_words(words, n)
            flat += closure.reticulation_number(p, c, words.shape[0]) == 0
            identical += inst.system == tree.system
    assert dropped > 0
    assert flat == dropped, "a dropped placement was cluster-visible after all"
    assert identical > 0, "no size-2 cycle in the dropped set; the starkest case went untested"


def test_compatible_placements_are_exactly_the_galled_ones():
    """``compatible`` must agree with ``networks.is_galled`` on every pair, both ways.

    Non-vacuous by construction: the counts assert that the sample contains both accepted and
    rejected pairs, so a rule that said "always true" could not pass.
    """
    accepted = rejected = 0
    for tree in G.rooted_binary_trees(5):
        cycles = G.cycles_of(tree)
        for a, b in itertools.combinations(cycles, 2):
            galled = N.is_galled(G.to_network(tree, (a, b)))
            assert G.compatible(a, b) == galled
            accepted += galled
            rejected += not galled
    assert accepted > 0 and rejected > 0


def test_node_disjointness_already_implies_edge_disjointness_on_this_construction():
    """Measured, and it is the mirror of A28 rather than a coincidence.

    A28 found that on the S50 *generator*, edge-disjointness (level-1) already gives
    vertex-disjointness (galled).  Here the implication runs the other way and is a property of
    placing cycles on a base tree: two cycles can only share an *edge* while sharing no *node* if
    that edge's lower endpoint is an endpoint of both, i.e. it lies in ``{d1, r1} & {d2, r2}``.
    But both cycles' apexes are then strict ancestors of that one node, so the lower apex sits on
    the other cycle's path and the node sets meet after all.

    The edge test is kept in :func:`~opc.theory.galled.compatible` regardless -- it costs one
    integer ``AND``, and it is the invariant that would catch the day the construction changes.
    """
    for n in (5, 6):
        pairs = separating = 0
        for tree in G.rooted_binary_trees(n):
            cycles = G.cycles_of(tree)
            for a, b in itertools.combinations(cycles, 2):
                if a.nodes & b.nodes:
                    continue
                pairs += 1
                separating += bool(a.edges & b.edges)
        assert pairs > 0
        assert separating == 0


# -------------------------------------------------------------------- the enumeration's output


def test_the_closed_form_and_the_dag_traversal_agree_at_k_equals_one():
    for n in (5, 6):
        seen = 0
        for tree in G.rooted_binary_trees(n):
            for cy in G.cycles_of(tree):
                inst = G.instance(tree, (cy,))
                system, triples = _closed_form_k1(tree, cy)
                assert inst.system == system
                assert inst.triples == triples
                seen += 1
        assert seen > 0


def test_the_enumeration_agrees_with_the_network_path():
    """Clusters, galledness, level, cycle sizes and ``R(Psi)``, against code E3 does not share.

    Strided rather than exhaustive: the same check over all 14,925 instances at n = 5 and 334,035
    at n = 6 is ``scripts/probe_enumeration.py``'s job, not a unit test's.
    """
    for n, stride in ((5, 7), (6, 101)):
        checked = k2 = singleton_hybrid = 0
        i = 0
        for tree in G.rooted_binary_trees(n):
            cycles = G.cycles_of(tree)
            for k in (0, 1, 2):
                for combo in G.galled_placements(tree, k, cycles):
                    i += 1
                    if i % stride:
                        continue
                    inst = G.instance(tree, combo)
                    net = G.to_network(tree, combo)
                    want = 0
                    for c in N.full_clusters(net):
                        want |= 1 << sum(1 << t for t in c)
                    assert want == inst.system
                    assert N.is_galled(net)
                    assert N.level(net) == min(k, 1)
                    assert sorted(N.cycle_sizes(net)) == sorted(inst.cycle_sizes)
                    assert _triples_via_network(net, n) == inst.triples
                    checked += 1
                    k2 += k == 2
                    singleton_hybrid += inst.singleton_hybrids > 0
        assert checked > 0
        assert k2 > 0, "no k=2 instance was checked, so the composition was never exercised"
        assert singleton_hybrid > 0, (
            "no instance had a singleton hybrid cluster, so A38's case was never exercised"
        )


def test_the_reticulation_number_read_off_the_closure_matches_the_placements():
    """``|E| - |V| + 1`` on the Hasse diagram of ``I(C)``, which never sees how many were placed.

    This is the claim E3 quotients over: a clustering system determines its regular network, and
    therefore its reticulation count.  Computed from the clusters alone, it cannot restate its own
    input.
    """
    n = 6
    i = 0
    checked = 0
    for tree in G.rooted_binary_trees(n):
        cycles = G.cycles_of(tree)
        for k in (0, 1, 2):
            for combo in G.galled_placements(tree, k, cycles):
                i += 1
                if i % 149:
                    continue
                inst = G.instance(tree, combo)
                words = closure.with_trivial_clusters(
                    closure.closure_words(_system_words(inst.system, n), n), n)
                p, c = closure.hasse_covers_words(words, n)
                assert closure.reticulation_number(p, c, words.shape[0]) == k
                checked += 1
    assert checked > 0


def test_the_clustering_system_is_the_canonical_form_so_dedup_is_up_to_isomorphism():
    """Two placements sharing a clustering system must realise the same regular network.

    This is what lets the enumeration deduplicate on ``C(Psi)`` instead of on a DAG canonical form
    (Hellmuth-Schaller-Stadler).  Asserted the only way that means anything: find distinct
    placements with equal systems, rebuild the network from the *clusters* both times, and require
    the results to have the same reticulation count and the same triple system.
    """
    n = 6
    by_system: dict[int, G.GalledInstance] = {}
    collisions = 0
    for inst in G.enumerate_galled(n, 1):
        prev = by_system.get(inst.system)
        if prev is None:
            by_system[inst.system] = inst
            continue
        collisions += 1
        assert prev.k == inst.k
        assert prev.triples == inst.triples, (
            "two placements with one clustering system disagree about R(Psi), so the clustering "
            "system is not the canonical form the quotient assumes"
        )
    assert collisions > 0, "no two placements shared a clustering system; nothing was tested"


def test_the_triple_system_determines_the_clustering_system_on_shortcut_free_galled_trees():
    """E3's headline, pinned at the smallest ``n`` where it is cheap: the fibers are singletons.

    ``C(Psi) -> R(Psi)`` is injective over every galled tree with k <= 2 on 5 and 6 taxa, so the
    clustering system -- and therefore the regular network -- is recoverable from the displayed
    triple system alone.  ``scripts/e3_fibers.py`` carries it to n = 8; this is the version that
    fails fast if the enumeration or the encoding is ever broken.
    """
    for n in (5, 6):
        fibers: dict[int, set[int]] = {}
        for inst in G.enumerate_galled(n, 2):
            fibers.setdefault(inst.triples, set()).add(inst.system)
        sizes = sorted(len(v) for v in fibers.values())
        assert len(fibers) > 1, "the enumeration produced nothing to quotient"
        assert sizes[-1] == 1, f"n={n}: largest fiber is {sizes[-1]}, expected all singletons"


def test_allowing_shortcuts_breaks_the_map_and_reproduces_yu_nakhlehs_figure_2():
    """The counterexample that makes the previous test a *finding* rather than an artefact.

    Yu-Nakhleh Fig. 2 exhibits three distinct single-reticulation 4-taxon networks with identical
    triple systems, which the plan's E3 takes as the reason to expect non-singleton fibers.  Put
    the shortcut placements back and it appears here too -- fibers of size 3 at n = 4 -- so the
    injectivity above is a property of the **regular** class and not of our encoding being blunt.

    It also breaks something stronger, and this is A11 from a direction the previous study never
    looked from: with shortcuts allowed ``C -> R`` is not a *function* at all.  A shortcut
    reticulation adds displayed triples while leaving the clustering system untouched, so one
    clustering system carries several triple systems.  That, and not merely "hard to recover", is
    why the correspondence is stated for regular networks.
    """
    fibers: dict[int, set[int]] = {}
    per_system: dict[int, set[int]] = {}
    for tree in G.rooted_binary_trees(4):
        cycles = G.cycles_of(tree, allow_shortcuts=True)
        for k in (0, 1, 2):
            for combo in G.galled_placements(tree, k, cycles):
                inst = G.instance(tree, combo)
                fibers.setdefault(inst.triples, set()).add(inst.system)
                per_system.setdefault(inst.system, set()).add(inst.triples)
    assert max(len(v) for v in fibers.values()) == 3, "expected Fig. 2's three-network fiber"
    assert max(len(v) for v in per_system.values()) > 1, (
        "with shortcuts allowed one clustering system must carry more than one triple system"
    )


def test_triple_bits_matches_a_direct_definition():
    """The lookup table the whole inner loop rests on, against the definition it stands for."""
    for n in (4, 5, 6):
        table = G.triple_bits(n)
        slots = G.triple_slots(n)
        for mask in range(1 << n):
            want = 0
            for rank, (a, b, c) in enumerate(slots):
                inside = [t for t in (a, b, c) if mask >> t & 1]
                if len(inside) != 2:
                    continue
                # Slots run (ab|c, ac|b, bc|a), so the resolution index is 2 minus the position of
                # the taxon left *outside* the cluster.
                outside = next(t for t in (a, b, c) if t not in inside)
                want |= 1 << (3 * rank + 2 - (a, b, c).index(outside))
            assert table[mask] == want
