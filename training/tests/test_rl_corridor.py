"""The swap corridor: the construction's identities, and the lattice statistics read off it.

The corridor's whole value rests on two equalities -- ``S({}) == greedy`` and ``S(D) == truth`` --
and on every node between them being a *legal terminal state*.  If any of those slips, the object
being measured is not "the path from the baseline to the answer" and every number taken along it
means something else.  So they are asserted exhaustively over the whole hypercube rather than
sampled.

``monotone_reachable`` is checked against brute-force enumeration of all ``d!`` orderings, per the
house rule that nothing is checked against itself: the DP factorises the count over predecessors
and the brute force does not.

**What is deliberately not here.**  Nothing in this file computes an MPL.  The claim that some
truth-ward swap is *downhill* is a property of the surface, not of the construction, and it belongs
to ``scripts/w7_corridor.py``'s smoke check where a real cell and a real evaluator are available.
A unit test that asserted it would need either a fixture with no bearing on the study or a GPU.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from opc import bitset
from opc.rl import corridor as CO
from opc.rl.env import ClusterEnv, greedy_support_rollout
from opc.solve import candidates


class _Instance:
    """The five attributes :func:`corridor.frame` reads, and nothing else.

    A real :class:`opc.rl.instances.Instance` needs a simulated grid cell behind it; the corridor's
    construction depends on none of that, so standing one up here would test the fixture rather
    than the module.
    """

    def __init__(self, gs, k, truth_rows):
        self.gs = gs
        self.k = int(k)
        self.graph = None
        self.truth_rows = np.asarray(truth_rows, dtype=np.int64)
        self.n_true = int(self.truth_rows.size)


def _ground_set(n_taxa=12, n_clusters=90, seed=0):
    """A dense-ish random ground set -- ``tests/test_rl_env.py``'s, so the two share a fixture."""
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < n_clusters:
        size = int(rng.integers(2, n_taxa - 1))
        c = frozenset(rng.choice(n_taxa, size=size, replace=False).tolist())
        if c in seen:
            continue
        seen.add(c)
        rows.append(bitset.from_indices(sorted(c), n_taxa))
    words = bitset.pack_rows(rows, n_taxa)
    support = rng.integers(1, 40, size=len(rows)).astype(np.int32)
    return candidates.build(words, support, n_taxa, m=1000, tau=0)


def _maximal_under(env, weights) -> np.ndarray:
    """A maximal feasible selection, accepting in descending ``weights`` order."""
    env.reset()
    for v in np.argsort(-np.asarray(weights, dtype=float), kind="stable"):
        ok, klass = env.state.can_accept(int(v))
        if ok:
            env.state.accept(int(v), int(klass))
    return env.selection.copy()


def fixture(seed=0, budget=2):
    """A ground set with a "truth" that is maximal and that greedy-by-support does **not** find.

    The truth is the maximal set a *different* weight vector selects.  That is enough to exercise
    every identity here -- they are statements about the lattice, not about phylogenetics -- and it
    is the only way to get a non-empty ``D`` without simulating a grid cell.
    """
    gs = _ground_set(seed=seed)
    env = ClusterEnv(gs, budget=budget)
    rng = np.random.default_rng(seed + 7)
    truth = _maximal_under(env, rng.random(gs.n_clusters))
    inst = _Instance(gs, budget, truth)
    return inst, ClusterEnv(gs, budget=budget)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fixture_is_not_degenerate(seed):
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    assert sum(c.n_edges for c in inst.gs.components) > 200, "no overlap: nothing can be blocked"
    assert fr.d >= 2, "greedy already found the truth: there is no corridor to measure"
    assert fr.false_pos.size >= 1, "greedy selected no wrong cluster: nothing is being swapped out"
    assert fr.truth_feasible and fr.truth_maximal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_two_poles_are_exact(seed):
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    assert np.array_equal(np.sort(CO.corridor_node(env, fr, [])),
                          np.sort(greedy_support_rollout(env)))
    assert np.array_equal(np.sort(CO.corridor_node(env, fr, fr.misses)), np.sort(fr.truth))


@pytest.mark.parametrize("seed", [0, 1])
def test_every_corridor_node_is_a_legal_terminal_state(seed):
    """Feasible by the *independent* validator, and maximal -- so it is a state the env can end in.

    ``state.verify()`` runs ``heuristics.check``, which re-derives (L) and (N3O) from the accepted
    set with its own triangle count and shares nothing with the incremental mask.
    """
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    if 2 ** fr.d > 512:
        pytest.skip(f"d = {fr.d}: the whole hypercube is not the point of this test")
    seen = set()
    for A in CO.enumerate_nodes(fr.d, budget=2 ** fr.d):
        sel = CO.corridor_node(env, fr, fr.misses[list(A)])
        assert env.state.verify(), f"node {A} fails the independent feasibility check"
        assert not env.legal_mask().any(), f"node {A} is not maximal"
        seen.add(tuple(sel.tolist()))
    assert len(seen) > 2, "every subset collapsed onto the two poles: the corridor is vacuous"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_S_of_A_does_not_depend_on_the_order_A_is_presented_in(seed):
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        A = fr.misses[rng.permutation(fr.d)[:max(1, fr.d // 2)]]
        first = CO.corridor_node(env, fr, A)
        for _ in range(3):
            other = CO.corridor_node(env, fr, rng.permutation(A))
            assert np.array_equal(np.sort(first), np.sort(other))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_force_move_lands_on_a_maximal_feasible_state_that_contains_v(seed):
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    rng = np.random.default_rng(seed + 3)
    moved = 0
    for v in rng.choice(inst.gs.n_clusters, size=25, replace=False):
        if v in set(fr.greedy.tolist()):
            continue
        sel = CO.force_move(env, fr.greedy, int(v))
        assert int(v) in set(sel.tolist()), "the forced cluster is not in the result"
        assert env.state.verify() and not env.legal_mask().any()
        moved += int(not np.array_equal(np.sort(sel), np.sort(fr.greedy)))
    assert moved > 0, "no forced cluster changed the selection: the operator is a no-op here"


def test_complete_raises_when_tier_zero_cannot_be_taken():
    """A silently skipped forced row would make ``S(D)`` a subset of the truth and look fine."""
    inst, env = fixture(seed=0)
    fr = CO.frame(inst, env)
    # The truth plus a cluster the truth already excludes: the truth is maximal, so any row
    # outside it is blocked once the truth is in -- but tier 0 presents them together.
    outside = [v for v in range(inst.gs.n_clusters) if v not in set(fr.truth.tolist())]
    for v in outside:
        try:
            CO.complete(env, [np.append(fr.truth, np.int64(v))])
        except ValueError:
            return
    pytest.fail("no row outside a maximal truth was refused as a forced tier-0 member")


def _brute_force_increasing_orderings(values, d):
    """Every ordering, walked start to finish -- the deliberately stupid version of the DP."""
    n = 0
    for perm in itertools.permutations(range(d)):
        cur = ()
        for a in perm:
            nxt = CO.subset_key(cur + (a,))
            if values[nxt] <= values[cur]:
                break
            cur = nxt
        else:
            n += 1
    return n


@pytest.mark.parametrize("seed", range(6))
def test_monotone_reachable_against_brute_force(seed):
    d = 4
    rng = np.random.default_rng(seed)
    values = {CO.subset_key(c): float(rng.normal())
              for r in range(d + 1) for c in itertools.combinations(range(d), r)}
    got = CO.monotone_reachable(values, d)
    want = _brute_force_increasing_orderings(values, d)
    assert got["monotone_n_orderings"] == want
    assert got["monotone_exists"] is (want > 0)
    assert got["monotone_frac_orderings"] == pytest.approx(want / 24)


def test_monotone_reachable_is_certain_on_a_monotone_table_and_refuses_a_partial_one():
    d = 4
    full = {CO.subset_key(c): float(len(c))
            for r in range(d + 1) for c in itertools.combinations(range(d), r)}
    got = CO.monotone_reachable(full, d)
    assert got["monotone_n_orderings"] == 24 and got["monotone_frac_orderings"] == 1.0
    partial = dict(list(full.items())[:-1])
    assert CO.monotone_reachable(partial, d)["monotone_exists"] is None, \
        "a sampled corridor cannot answer this and must not appear to"


def test_epistasis_is_zero_on_an_additive_table():
    d = 4
    gains = np.array([3.0, -1.0, 7.0, 0.5])
    values = {CO.subset_key(c): float(gains[list(c)].sum())
              for r in range(d + 1) for c in itertools.combinations(range(d), r)}
    got = CO.epistasis(values, d)
    assert got["epistasis_max"] == pytest.approx(0.0)
    assert got["solo_gain_max"] == pytest.approx(7.0)
    assert got["solo_gain_n_downhill"] == 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_legal_windows_matches_a_naive_per_step_replay(seed):
    """The same question asked through ``SearchState.legal()`` instead of the vectorised mask."""
    inst, env = fixture(seed=seed)
    fr = CO.frame(inst, env)
    got = CO.legal_windows(env, fr.misses)

    env.reset()
    naive: dict[int, list[int]] = {int(v): [] for v in fr.misses}
    step = 0
    while True:
        legal = {v for v, _k in env.state.legal()}
        if not legal:
            break
        for v in fr.misses:
            if int(v) in legal:
                naive[int(v)].append(step)
        order = np.argsort(-env.support, kind="stable")
        rank = np.empty(env.n, dtype=np.int64)
        rank[order] = np.arange(env.n)
        cand = np.fromiter(legal, dtype=np.int64)
        env.step(int(cand[np.argmin(rank[cand])]))
        step += 1

    assert got["n_steps"] == step
    assert np.array_equal(np.sort(got["selection"]), np.sort(greedy_support_rollout(env)))
    for v in fr.misses:
        assert got["live"][int(v)] == naive[int(v)]
    assert got["n_never_legal"] == sum(1 for s in naive.values() if not s)


def test_enumerate_nodes_takes_the_hypercube_when_it_fits_and_both_poles_when_it_does_not():
    whole = CO.enumerate_nodes(5, budget=256)
    assert len(whole) == 32 and len(set(whole)) == 32
    sample = CO.enumerate_nodes(16, budget=200, seed=0)
    assert len(sample) == len(set(sample))
    assert () in sample and tuple(range(16)) in sample
    for i in range(16):
        assert (i,) in sample, "a missing singleton would silently drop a marginal"
        assert CO.subset_key([j for j in range(16) if j != i]) in sample
