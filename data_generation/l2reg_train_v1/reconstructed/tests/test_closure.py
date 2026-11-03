"""B1's T2 and T5: the closure, the Hasse diagram, and the network it stands for (WS1-5).

The round trip is the real gate.  Generate a level-1 network ``N``, take its clustering system,
and rebuild: the rebuilt network must carry the *same* clustering system, the same number of
reticulations, and level 1.  Anything less and the solver could return a right-looking answer for
a network nobody asked about.

Note what the isomorphism is **to**.  ``H(C(N)) ~= N`` is false as literally written on every
network with a reticulation: a hybrid node has indegree 2 and outdegree 1, so its cluster equals
its child's and the node-to-cluster map is not injective.  The claim is about the *regular
version* of ``N``, obtained by contracting exactly the edges with ``c(u) == c(v)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import cluster_systems, pack
from hypothesis import HealthCheck, given, settings

from opc import bitset
from opc.clusters import systems
from opc.simulate import networks as N
from opc.solve import closure

SLOW = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _frozensets(words, n_taxa):
    return {frozenset(bitset.to_indices(w, n_taxa)) for w in np.ascontiguousarray(words)}


# ------------------------------------------------------------------------------------- closure


@given(cluster_systems(max_taxa=8, max_clusters=10))
@SLOW
def test_closure_words_matches_the_frozenset_reference(sysdef):
    n_taxa, cs = sysdef
    got = _frozensets(closure.closure_words(pack(cs, n_taxa), n_taxa), n_taxa)
    assert got == systems.closure(set(cs))


@given(cluster_systems(max_taxa=8, max_clusters=10))
@SLOW
def test_closure_is_idempotent_and_preserves_the_target_class(sysdef):
    """T5.  The hypothesis is (L)+(N3O), not closedness -- see SPEC-AMENDMENTS A10."""
    n_taxa, cs = sysdef
    once = closure.closure_words(pack(cs, n_taxa), n_taxa)
    twice = closure.closure_words(once, n_taxa)
    assert _frozensets(once, n_taxa) == _frozensets(twice, n_taxa)
    if systems.is_feasible(set(cs)):
        closed = _frozensets(once, n_taxa)
        assert systems.is_feasible(closed) and systems.is_closed(closed)


@given(cluster_systems(max_taxa=7, max_clusters=9))
@SLOW
def test_hasse_covers_fast_path_matches_the_reference(sysdef):
    """The GEMM + transitive-reduction path against the O(k^3) frozenset one."""
    n_taxa, cs = sysdef
    words = pack(cs, n_taxa)
    p, c = closure.hasse_covers_words(words, n_taxa)
    got = {(frozenset(bitset.to_indices(words[a], n_taxa)),
            frozenset(bitset.to_indices(words[b], n_taxa))) for a, b in zip(p, c)}
    assert got == closure.hasse_covers(list(cs))


def test_trivial_clusters_remove_every_unifurcation():
    """Why ``X`` and the singletons are mandatory rather than cosmetic.

    With all singletons present, a member of size >= 2 cannot have a single cover: that cover
    would have to contain every one of its singletons, i.e. be the member itself.  Without them
    the Hasse diagram grows degree-2 nodes and ``to_enewick`` renders a malformed tree.
    """
    n = 6
    cs = [frozenset({0, 1, 2}), frozenset({0, 1})]
    bare = pack(cs, n)
    p, _ = closure.hasse_covers_words(bare, n)
    assert np.bincount(p, minlength=bare.shape[0]).max() == 1, "fixture should be a chain"

    full = closure.with_trivial_clusters(bare, n)
    p, c = closure.hasse_covers_words(full, n)
    n_children = np.bincount(p, minlength=full.shape[0])
    internal = np.flatnonzero(bitset.popcounts(full) >= 2)
    assert (n_children[internal] >= 2).all()


# ----------------------------------------------------------------------------------------- T2


@pytest.mark.parametrize("n_taxa,r_max,seed", [(10, 2, 401), (15, 3, 402), (25, 3, 403),
                                               (50, 5, 412)])
def test_theory_round_trip_rebuilds_the_network_from_its_clusters(n_taxa, r_max, seed):
    """T2: ``H(I(C(N)))`` carries ``C(N)``, ``r(N)`` reticulations, and level 1.

    ``seed=402`` and ``seed=412`` are here on purpose -- their regular versions put a hybrid node
    at the apex of a second cycle, which is the configuration that used to be miscounted as
    level 2 (see :func:`test_level_counts_a_reticulation_once`).
    """
    base = N.build_base(n_taxa, r_max, seed=seed)
    for r in range(r_max + 1):
        net = N.with_gamma(N.scaled(N.with_reticulations(base, r), 1.0), 0.3)
        full = N.full_clusters(net)
        words = np.stack([bitset.from_indices(c, n_taxa) for c in full])

        rebuilt = closure.network_from_hasse(words, n_taxa, net.labels)
        assert set(N.hardwired_clusters(rebuilt).values()) == full
        assert rebuilt.n_retics == r
        assert N.level(rebuilt) == (1 if r else 0)

        p, c = closure.hasse_covers_words(words, n_taxa)
        assert closure.reticulation_number(p, c, words.shape[0]) == r

        # The isomorphism, exactly as §8 test 2b states it: against the *regular version*.
        below = N.hardwired_clusters(net)
        net_covers = {(below[v], below[ch]) for v in below for ch in net.net_children(v)
                      if below[v] != below[ch]}
        assert net_covers == closure.hasse_covers(full)


def test_closure_recovers_the_hybrid_the_solver_never_had_a_variable_for():
    """``h = A & B`` is nested in both, so it is not in the component the ILP solved.

    The accepted set is the overlapping pair; the closure supplies the hybrid.  This is why the
    handoff's ``x[h] >= y[h]`` row is unnecessary as well as unsatisfiable.
    """
    n = 6
    accepted = [frozenset({0, 1, 2}), frozenset({2, 3, 4})]
    closed = _frozensets(closure.closure_words(pack(accepted, n), n), n)
    assert closed == set(accepted) | {frozenset({2})}


def test_network_from_hasse_rejects_a_system_that_is_not_level_1():
    """Three covering parents is an indegree-3 hybrid -- outside the target class."""
    n = 8
    cs = [frozenset(range(n)), frozenset({0, 1, 2}), frozenset({0, 1, 3}), frozenset({0, 1, 4}),
          frozenset({0, 1})]
    words = closure.with_trivial_clusters(pack(cs, n), n)
    with pytest.raises(ValueError, match="covering parents"):
        closure.network_from_hasse(words, n)


def test_level_counts_a_reticulation_once_even_when_its_hybrid_is_a_cut_vertex():
    """Regression: a hybrid at the apex of a *second* cycle used to be counted in both.

    ``level`` walked the biconnected components and counted hybrid **nodes** per component.  A cut
    vertex belongs to every component incident to it, so such a hybrid was counted twice and a
    level-1 network came back as level-2.  The generator never produces this shape -- its
    ``consumed`` edge set keeps cycles apart -- but the *regular version* does, and that is the
    object a cluster-based method reconstructs.  Reticulations are now counted by their in-edges.
    """
    n_taxa = 15
    base = N.build_base(n_taxa, 3, seed=402)
    net = N.with_gamma(N.scaled(N.with_reticulations(base, 3), 1.0), 0.3)
    assert N.level(net) == 1

    words = np.stack([bitset.from_indices(c, n_taxa) for c in N.full_clusters(net)])
    rebuilt = closure.network_from_hasse(words, n_taxa, net.labels)
    assert rebuilt.n_retics == 3
    assert N.level(rebuilt) == 1

    # The shape that triggers it: some hybrid strictly contains another hybrid's covers, i.e. it
    # sits at the apex of that second cycle.  Without this the test would pass vacuously.
    below = N.hardwired_clusters(rebuilt)
    hybrids = {frozenset(below[ret.hybrid]) for ret in rebuilt.retics}
    apex = [
        h for h in hybrids
        for ret in rebuilt.retics
        if frozenset(below[ret.hybrid]) != h
        and all(frozenset(below[p]) < h for p in rebuilt.net_parents(ret.hybrid))
    ]
    assert apex, "fixture no longer contains a hybrid at another cycle's apex"
