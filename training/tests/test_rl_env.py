"""W2 -- ``ClusterEnv``, against the machinery it is a fast path for.  RL_PLAN gate G-R0.

Two independent obligations, and they are not the same one twice:

* the **mask** must equal ``SearchState.legal()`` at *every* state of a trajectory, not only at the
  empty one -- the conditions that are hard to vectorise ((L) at a committed neighbour, and the
  (N3O) triangle) are exactly the ones that only appear once something has been accepted;
* the **terminal state** must satisfy ``heuristics.check``, which shares no state with ``_State``:
  it re-derives (L) by a lexsort groupby and (N3O) as ``nnz((A @ A) . A) / 6`` from the accepted
  set alone.  That is the house rule -- nothing is checked against itself -- applied to the env.

Every test here asserts that its fixture produced the case it means to test.  A ground set whose
overlap graph is empty would pass a mask-parity check while exercising none of it, which is the
shape of the defect the budget-DP test carried for a whole sprint.
"""

from __future__ import annotations

import numpy as np
import pytest

from opc import bitset
from opc.followup import search
from opc.rl.env import ClusterEnv, OverlapCSR, greedy_support_rollout
from opc.solve import candidates, heuristics


def ground_set(n_taxa=12, n_clusters=90, seed=0):
    """A dense-ish random ground set: enough overlap that (L) and (N3O) both bite."""
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


def _reference_mask(state, n):
    out = np.zeros(n, dtype=bool)
    out[[v for v, _ in state.legal()]] = True
    return out


def test_the_fixture_is_not_degenerate():
    gs = ground_set()
    assert gs.n_clusters > 50
    assert sum(c.n_edges for c in gs.components) > 200, "no overlap: the mask would test nothing"
    assert max(c.n_hcand for c in gs.components) > 1, "one intersection class: (L) cannot bite"


@pytest.mark.parametrize("seed,budget", [(0, 1), (1, 2), (2, 3), (3, 0)])
def test_mask_equals_search_state_at_every_state(seed, budget):
    gs = ground_set(seed=seed)
    env = ClusterEnv(gs, budget=budget)
    rng = np.random.default_rng(seed + 100)
    states = blocked_seen = 0
    while True:
        mask = env.legal_mask()
        assert np.array_equal(mask, _reference_mask(env.state, env.n)), f"state {states}"
        states += 1
        # Did anything actually get *blocked* by a constraint rather than by already being in S?
        blocked_seen += int((~mask & ~env.state.in_s).sum() > 0)
        if not mask.any():
            break
        env.step(int(rng.choice(np.flatnonzero(mask))))
    assert states > 3, f"vacuous: the episode was {states} states long"
    assert blocked_seen > 0, "no candidate was ever blocked; the constraints were never exercised"
    assert env.verify()


def test_budget_binds_and_is_counted_across_components():
    gs = ground_set(seed=7)
    for budget in (0, 1, 2):
        env = ClusterEnv(gs, budget=budget)
        greedy_support_rollout(env)
        assert env.state.n_hybrids <= budget
        assert env.verify()
    # The budget must not be inert -- but "bigger budget, more clusters" is **not** the check to
    # make: it fails on this fixture, where greedy at k = 3 selects 3 and k = 0 selects 4.
    # Greedy is order-dependent, so admitting a cluster that opens a reticulation early can block
    # more later than it lets in.  What has to hold is that the budget *changes* the answer and
    # that it is actually spent.
    zero = ClusterEnv(gs, budget=0)
    greedy_support_rollout(zero)
    three = ClusterEnv(gs, budget=3)
    greedy_support_rollout(three)
    assert zero.state.n_hybrids == 0
    assert three.state.n_hybrids > 0, "no reticulation was ever opened; the budget is untested"
    assert not np.array_equal(np.sort(zero.selection), np.sort(three.selection))
    assert zero.verify() and three.verify()
    assert not zero.legal_mask().any() and not three.legal_mask().any()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gr0_greedy_parity_with_the_landed_heuristic(seed):
    """argmax-support under the env == ``search._greedy_support``.  Exactly, not nearly."""
    gs = ground_set(seed=seed)
    env = ClusterEnv(gs, budget=2)
    got = np.sort(greedy_support_rollout(env))

    st = search.SearchState(gs, 2)
    for v in np.argsort(-np.asarray(gs.support, dtype=float), kind="stable"):
        ok, klass = st.can_accept(int(v))
        if ok:
            st.accept(int(v), klass)
    assert np.array_equal(got, np.sort(st.selection))
    assert got.size > 2, "vacuous: the greedy selected almost nothing"


def test_terminal_states_pass_the_independent_validator():
    gs = ground_set(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(4):
        env = ClusterEnv(gs, budget=2)
        while True:
            mask = env.legal_mask()
            if not mask.any():
                break
            env.step(int(rng.choice(np.flatnonzero(mask))))
        sel = env.selection
        for comp in gs.components:
            mine = np.flatnonzero(env.state.in_s[comp.vertices]).astype(np.int64)
            assert heuristics.check(comp, mine, 2).ok
        assert sel.size > 0


def test_init_policies():
    gs = ground_set(seed=9)
    env = ClusterEnv(gs, budget=2)

    env.reset(init="empty")
    assert env.selection.size == gs.kernel_size

    env.reset(init="greedy")
    greedy = env.selection.copy()
    assert greedy.size > gs.kernel_size
    assert not env.legal_mask().any(), "a greedy start must be maximal, or it is not the baseline"

    env.reset(init="given", given=greedy[:3])
    assert set(env.selection.tolist()) >= set(greedy[:3].tolist())
    assert env.verify()

    with pytest.raises(ValueError, match="unknown init"):
        env.reset(init="warp")


def test_step_refuses_an_illegal_action():
    gs = ground_set(seed=11)
    env = ClusterEnv(gs, budget=1)
    mask = env.legal_mask()
    env.step(int(np.flatnonzero(mask)[0]))
    illegal = np.flatnonzero(~env.legal_mask())
    assert illegal.size, "fixture left nothing illegal to try"
    with pytest.raises(ValueError, match="illegal action"):
        env.step(int(illegal[0]))


def test_overlap_csr_classes_are_globally_unique():
    """Two components may reuse a local class number for different intersections."""
    gs = ground_set(seed=13)
    g = OverlapCSR.of(gs)
    assert g.n_classes == sum(c.n_hcand for c in gs.components)
    for ci, comp in enumerate(gs.components):
        v = np.asarray(comp.vertices)
        assert np.all(g.comp_of[v] == ci)
    # Symmetry: every edge appears from both endpoints.
    for v in range(g.n):
        for u, c in zip(*g.neighbours(v), strict=True):
            back = dict(zip(*[x.tolist() for x in g.neighbours(int(u))], strict=True))
            assert back[v] == c


def test_step_reward_is_zero_and_done_is_maximality():
    """RL_PLAN §2: per-step reward 0, no STOP action, termination is ``legal(S_t) = {}``."""
    gs = ground_set(seed=17)
    env = ClusterEnv(gs, budget=2)
    rng = np.random.default_rng(0)
    done = False
    steps = 0
    while not done:
        mask = env.legal_mask()
        if not mask.any():
            break
        _obs, reward, done, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert reward == 0.0
        assert info["class"] >= -1
        steps += 1
    assert done and steps > 3
    assert not env.legal_mask().any()


def test_with_features_agrees_with_the_single_return_form_and_recounts_correctly():
    """``deg_sel`` is the decoder's "overlap-degree within ``S_t``" (§3), free from the mask.

    Two obligations.  The mask must be **unchanged** -- G-R0's parity subject is the single-return
    form, and a refactor that quietly altered it would be invisible in a test that only checked the
    new return.  And ``deg_sel`` must be recounted from the CSR *independently*, not read back out
    of the same intermediate that produced it: the whole point of the extra return is that the two
    were computed once, so checking one against the other would be checking nothing.
    """
    gs = ground_set(n_taxa=13, n_clusters=110, seed=23)
    env = ClusterEnv(gs, budget=3)
    g = env.graph
    rng = np.random.default_rng(5)
    seen_nonzero = False
    for _ in range(40):
        mask_only = env.legal_mask()
        mask, deg = env.legal_mask(with_features=True)
        assert np.array_equal(mask, mask_only), "the mask changed when features were asked for"

        accepted = set(np.flatnonzero(env.state.in_s).tolist())
        want = np.zeros(env.n, dtype=np.int64)
        for v in range(env.n):
            nbr, _cls = g.neighbours(v)
            want[v] = sum(1 for u in nbr.tolist() if u in accepted)
        assert np.array_equal(deg, want), "deg_sel is not the accepted-neighbour count"
        seen_nonzero |= bool(deg.any())

        if not mask.any():
            break
        obs, _r, done, _i = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert np.array_equal(obs["legal"], env.legal_mask())
        assert obs["deg_sel"] is not None
        if done:
            break
    # Assert the fixture produced the case being tested: a run where nothing was ever adjacent to
    # the accepted set would pass every check above while exercising none of it.
    assert seen_nonzero, "no candidate ever had an accepted neighbour -- the fixture is vacuous"
