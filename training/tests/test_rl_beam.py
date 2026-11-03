"""W12 -- the top-K beam decoder over cluster systems.  ``opc.rl.beam``.

Five obligations, each guarding something this repository has already paid for once:

* **Width 1 is the incumbent, exactly.**  ``beam_decode(width=1, max_dev=0)`` must be
  :func:`opc.rl.nodecls.decode` selection for selection, which chains onward to
  :func:`opc.rl.env.greedy_support_rollout` through ``nodecls``' own parity test.  If it were not,
  every ``f1_selected_vs_greedy`` W12 reports would be measuring the decoder rather than the pool.
* **The canonical order loses nothing.**  Cross-validated against brute-force enumeration over
  every subset of a small ground set, with feasibility decided by
  :func:`opc.solve.truth.feasible_and_cost` -- the independent validator, which re-decomposes and
  runs ``heuristics.check`` and shares no state with the mask the beam walks.
* **Everything returned is a cluster system.**  ``env.verify()`` and the budget, per member.
* **The env is left as it was found.**  An ``InstancePool`` built with ``swap_tabu`` hands one
  ``ClusterEnv`` to a ``SwapEnv``; a state left swapped in would surface as a wrong walk, not as an
  error.
* **The fixtures are not vacuous.**  Every test asserts the object it means to exercise exists --
  that the beam returned more than one system, that the brute force found more than one, that the
  deviation classes were actually populated.  ``CLAUDE.md`` records the budget-DP test passing
  1,188 budgets over *zero* multi-component systems.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from opc import bitset
from opc.rl import beam as BM
from opc.rl import nodecls as NC
from opc.rl.env import ClusterEnv, greedy_support_rollout
from opc.solve import candidates
from opc.solve import truth as T


def ground_set(n_taxa=12, n_clusters=90, seed=0):
    """A random ground set.  Mirrors ``tests/test_rl_nodecls.ground_set`` deliberately."""
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
    return candidates.build(words, support, n_taxa, m=40, tau=0)


def scores_for(gs, seed=0):
    return np.random.default_rng(seed).random(int(gs.n_clusters))


# ---------------------------------------------------------------------------- parity


def test_width_one_reproduces_decode_exactly():
    """**The parity the whole arm rests on**, and the reason ``width=1`` is a supported setting."""
    for seed in (0, 1, 2):
        gs = ground_set(seed=seed)
        p = scores_for(gs, seed + 10)
        want = NC.decode(ClusterEnv(gs, budget=2), p)
        got = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=1, width=1, max_dev=0)
        assert len(got) == 1
        assert want.size > 0, "decode selected nothing: the parity would be vacuous"
        assert np.array_equal(np.sort(got[0].selection), np.sort(want))
        assert got[0].deviations == 0
        assert got[0].recompleted == 0, "greedy is maximal; nothing to recomplete"


def test_width_one_on_support_reproduces_greedy_support_rollout():
    """Closes the chain ``beam -> decode -> greedy`` that ``test_rl_nodecls`` opens."""
    for seed in (0, 1, 2):
        gs = ground_set(seed=seed)
        s = np.asarray(gs.support, dtype=np.float64)
        want = greedy_support_rollout(ClusterEnv(gs, budget=2))
        got = BM.beam_decode(ClusterEnv(gs, budget=2), s, k=1, width=1, max_dev=0)
        assert want.size > 0, "greedy selected nothing: the parity would be vacuous"
        assert np.array_equal(np.sort(got[0].selection), np.sort(want))


def test_the_incumbent_is_always_in_the_pool():
    """Whatever the beam explores, ``decode``'s own answer must be reachable and returned."""
    gs = ground_set(seed=1)
    p = scores_for(gs, 4)
    want = np.sort(NC.decode(ClusterEnv(gs, budget=2), p))
    pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=16, width=4, max_dev=2)
    assert len(pool) > 1, "beam returned one system: every diversity claim below would be vacuous"
    assert BM.rank_of(pool, want) is not None


# ---------------------------------------------------------------------------- what it returns


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_every_returned_system_is_feasible_maximal_and_within_budget(budget):
    """Checked through ``env.verify()`` and ``legal_mask``, which share no state at all."""
    gs = ground_set(seed=2)
    p = scores_for(gs, 7)
    pool = BM.beam_decode(ClusterEnv(gs, budget=budget), p, k=12, width=4, max_dev=2)
    assert pool, "no system returned: the test would be vacuous"
    for sysm in pool:
        env = ClusterEnv(gs, budget=budget)
        env.reset(init="given", given=[v for v in sysm.selection if not env.state.in_s[int(v)]])
        assert env.verify()
        assert env.state.n_hybrids <= budget
        assert sysm.maximal
        assert not env.legal_mask().any(), "returned system is not maximal"


def test_returned_systems_are_pairwise_distinct():
    gs = ground_set(seed=0)
    p = scores_for(gs, 11)
    pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=24, width=6, max_dev=2)
    assert len(pool) > 2, "fewer than three systems: distinctness would be near-vacuous"
    keys = {sysm.key() for sysm in pool}
    assert len(keys) == len(pool)


def test_the_pool_is_ordered_by_the_requested_set_score():
    gs = ground_set(seed=1)
    p = scores_for(gs, 5)
    for kind in BM.SET_SCORES:
        pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=8, width=4, max_dev=2, score=kind)
        vals = [sysm.score(kind) for sysm in pool]
        assert vals == sorted(vals, reverse=True)


def test_deviation_classes_are_actually_populated():
    """``max_dev`` is a quota, and a quota nothing reaches is a setting that does nothing."""
    gs = ground_set(seed=2)
    p = scores_for(gs, 9)
    pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=64, width=6, max_dev=2)
    devs = {sysm.deviations for sysm in pool}
    assert max(devs) >= 1, f"only deviation classes {sorted(devs)} were reached"
    assert 0 in devs, "the incumbent was cut from the pool"


def test_the_incumbent_is_kept_even_when_k_others_outscore_it():
    """The case the guard exists for, and it is the common one rather than an edge case.

    Greedy maximises the score of the *next* cluster; the beam maximises the score of the *system*.
    Those differ, so the incumbent routinely does not rank first -- on this fixture it is 18th, so
    a pool of eight drops it.  Were it dropped, ``f1_ceiling`` would stop bounding ``f1_selected``.
    """
    gs = ground_set(seed=2)
    p = scores_for(gs, 9)
    want = np.sort(NC.decode(ClusterEnv(gs, budget=2), p))
    kept = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=8, beam=64, width=6, max_dev=2)
    cut = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=8, beam=64, width=6, max_dev=2,
                         include_incumbent=False)
    assert BM.rank_of(cut, want) is None, "the fixture does not exercise the guard"
    assert BM.rank_of(kept, want) is not None
    assert len(kept) == len(cut)


def test_a_wider_beam_never_makes_the_top_scoring_system_worse():
    """Monotone in ``width`` at fixed ``max_dev``: a wider search sees a superset of the paths."""
    gs = ground_set(seed=0)
    p = scores_for(gs, 3)
    best = []
    for width in (1, 2, 4, 8):
        pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=64, beam=64, width=width, max_dev=2)
        best.append(pool[0].sum_calibrated)
    assert all(b >= a - 1e-9 for a, b in zip(best, best[1:])), best


# ---------------------------------------------------------------------------- the bijection


def _brute_force(gs, budget, kernel):
    """Every feasible system over a small ground set, by the independent validator."""
    words = np.asarray(gs.words)
    free = [v for v in range(int(gs.n_clusters)) if v not in set(kernel.tolist())]
    feasible = {}
    for r in range(len(free) + 1):
        for comb in itertools.combinations(free, r):
            sel = tuple(sorted(set(comb) | set(kernel.tolist())))
            ok, j = T.feasible_and_cost(np.ascontiguousarray(words[list(sel)]), int(gs.n_taxa))
            if ok and j <= budget:
                feasible[sel] = j
    return feasible


def test_the_canonical_order_enumerates_every_maximal_system_exactly_once():
    """**The cross-validation.**  Unbounded, the beam is the brute force, once each.

    Under ``flashlight`` a terminal survives only if every candidate it passed over is now
    illegal, so every survivor is maximal *and* distinct paths still give distinct systems.  That
    the survivors are exactly the brute force's maximal systems is what says the canonical
    descending-rank restriction loses nothing -- and that ``len(got) == len(pool)`` is what says
    the bijection is intact, with no dedup doing the work.
    """
    gs = ground_set(n_taxa=6, n_clusters=10, seed=0)
    budget = 2
    kernel = np.asarray(gs.decomposition.kernel, dtype=np.int64)
    feasible = _brute_force(gs, budget, kernel)
    assert len(feasible) > 4, "brute force found almost nothing: the fixture is vacuous"

    maximal = set()
    for sel in feasible:
        rest = set(range(int(gs.n_clusters))) - set(sel)
        if not any(tuple(sorted(set(sel) | {v})) in feasible for v in rest):
            maximal.add(sel)
    assert len(maximal) > 1, "one maximal system only: the enumeration would be vacuous"
    assert max(len(s) for s in maximal) > 1, "every maximal system is a singleton: vacuous"

    p = scores_for(gs, 21)
    for terminal in ("flashlight", "complete"):
        pool = BM.beam_decode(ClusterEnv(gs, budget=budget), p, k=10_000, beam=10_000,
                              width=0, max_dev=-1, terminal=terminal)
        got = {tuple(sorted(int(v) for v in sysm.selection)) for sysm in pool}
        assert got == maximal, terminal
        assert all(sysm.maximal for sysm in pool)
    # only ``flashlight`` gets there without deduplicating anything
    pool = BM.beam_decode(ClusterEnv(gs, budget=budget), p, k=10_000, beam=10_000,
                          width=0, max_dev=-1, terminal="flashlight")
    assert len({s.key() for s in pool}) == len(pool), "the bijection needed dedup"


def test_without_a_terminal_rule_the_paths_are_a_bijection_onto_forward_maximal_systems():
    """The strict form: no dedup happens, because no two paths can reach the same set.

    A path ends when nothing legal ranks *above* its cursor, so the output is exactly
    ``{S feasible : no addable v ranks above max(S)}`` -- maximal systems plus those whose only
    additions were passed over.  Enumerating that from the brute force is what pins the canonical
    order itself rather than the recompletion that follows it.
    """
    gs = ground_set(n_taxa=6, n_clusters=10, seed=0)
    budget = 2
    kernel = np.asarray(gs.decomposition.kernel, dtype=np.int64)
    feasible = _brute_force(gs, budget, kernel)
    _order, rank = BM.rank_order(scores_for(gs, 21))

    want = set()
    for sel in feasible:
        top = max((rank[v] for v in sel), default=-1)
        rest = set(range(int(gs.n_clusters))) - set(sel)
        if not any(rank[v] > top and tuple(sorted(set(sel) | {v})) in feasible for v in rest):
            want.add(sel)
    assert len(want) > 1, "fewer than two forward-maximal systems: vacuous"

    pool = BM.beam_decode(ClusterEnv(gs, budget=budget), scores_for(gs, 21),
                          k=10_000, beam=10_000, width=0, max_dev=-1, terminal="none")
    got = [tuple(sorted(int(v) for v in sysm.selection)) for sysm in pool]
    assert len(set(got)) == len(got), "distinct paths produced the same set"
    assert set(got) == want
    assert not all(sysm.maximal for sysm in pool), (
        "every raw terminal was maximal: the flashlight rule would be guarding nothing")


def test_the_raw_cursor_rule_leaves_non_maximal_terminals_and_flashlight_drops_them():
    """**The negative test.**  ``env.verify()`` cannot see this: the terminals are feasible.

    Measured on the a97 checkpoint at 22.5% of terminals (b25) and 15.1% (b50), on 20 of 20 cells
    at both phases -- and in every one of those cases the cause was the same, a candidate the path
    passed over that was still legal at the end.  A pool quietly containing non-maximal systems
    would depress every ``f1_ceiling`` it is quoted in.
    """
    gs = ground_set(n_taxa=6, n_clusters=10, seed=0)
    p = scores_for(gs, 21)
    raw = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=10_000, beam=10_000,
                         width=0, max_dev=-1, terminal="none")
    lit = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=10_000, beam=10_000,
                         width=0, max_dev=-1, terminal="flashlight")
    bad = [s for s in raw if not s.maximal]
    assert bad, "the fixture produced no non-maximal terminal: the test would be vacuous"
    for sysm in bad:
        env = ClusterEnv(gs, budget=2)
        env.reset(init="given", given=[v for v in sysm.selection if not env.state.in_s[int(v)]])
        assert env.verify(), "a non-maximal terminal is still a feasible system"
        assert env.legal_mask().any()
    kept = {s.key() for s in lit}
    assert all(s.key() not in kept for s in bad)
    assert all(s.maximal for s in lit)


# ---------------------------------------------------------------------------- the set score


def test_the_calibrated_score_is_the_set_log_posterior_up_to_a_constant():
    """``sum log p + sum log(1-p)`` over the ground set -- what the head was trained on."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=40)
    sel = np.array([1, 5, 9, 30], dtype=np.int64)
    w = 42.0
    assert sel.size > 1, "the fixture selected nothing to sum"
    _, _, calibrated = BM.set_scores(sel, p, pos_weight=w)
    want = float(np.sum(np.log(p[sel]) - np.log1p(-p[sel]) - np.log(w)))
    assert calibrated == pytest.approx(want, rel=1e-12)
    # and the same number read as a log-posterior over the whole ground set, up to its constant
    q = p / (w - (w - 1.0) * p)                       # invert odds(p) = w * odds(q)
    inside = float(np.sum(np.log(q[sel]) - np.log1p(-q[sel])))
    assert calibrated == pytest.approx(inside, rel=1e-9)


def test_the_prior_shift_reorders_across_cardinalities_but_not_within_one():
    """Why the shift is not cosmetic: it is the only thing that stops the beam maximising size."""
    # 0.7 and 0.6 sit between the raw-logit cut (0.5) and the calibrated one (w/(1+w) = 0.977),
    # which is exactly the band the shift exists to exclude.
    p = np.array([0.99, 0.985, 0.7, 0.6], dtype=np.float64)
    big, small = np.array([0, 1, 2, 3]), np.array([0, 1])
    _, big_logit, big_cal = BM.set_scores(big, p, pos_weight=42.0)
    _, small_logit, small_cal = BM.set_scores(small, p, pos_weight=42.0)
    assert big_logit > small_logit, "uncorrected, a cluster above sigmoid 0.5 always pays"
    assert big_cal < small_cal, "corrected, a true posterior below 0.5 costs score"
    a, b = np.array([0, 2]), np.array([1, 2])
    assert (BM.set_scores(a, p)[2] > BM.set_scores(b, p)[2]) == \
           (BM.set_scores(a, p)[1] > BM.set_scores(b, p)[1])


def test_logits_are_taken_when_offered_rather_than_reconstructed():
    p = np.array([1.0, 0.5, 0.0], dtype=np.float64)   # saturated: unrecoverable by log
    z = np.array([37.0, 0.0, -37.0], dtype=np.float64)
    assert np.allclose(BM.logits_of(p, z), z)
    rebuilt = BM.logits_of(p)
    assert rebuilt[0] > 20.0 and rebuilt[2] < -20.0     # clipped, not inf
    assert np.isfinite(rebuilt).all()
    assert not np.allclose(rebuilt, z), "the fixture does not exercise the difference"


# ---------------------------------------------------------------------------- hygiene


def test_the_env_state_is_restored():
    """A ``SwapEnv`` may share this ``ClusterEnv``; a foreign state would be a wrong walk."""
    gs = ground_set(seed=0)
    env = ClusterEnv(gs, budget=2)
    before = env.state
    BM.beam_decode(env, scores_for(gs, 1), k=8, width=4, max_dev=1)
    assert env.state is before


def test_shape_and_degenerate_inputs():
    gs = ground_set(seed=0)
    p = scores_for(gs, 1)
    with pytest.raises(ValueError, match="scores must be"):
        BM.beam_decode(ClusterEnv(gs, budget=2), p[:-1])
    with pytest.raises(ValueError, match="unknown set score"):
        BM.beam_decode(ClusterEnv(gs, budget=2), p, score="posterior")
    assert BM.beam_decode(ClusterEnv(gs, budget=2), p, k=0) == []


def test_the_beam_is_deterministic():
    gs = ground_set(seed=1)
    p = scores_for(gs, 2)
    a = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=8, width=4, max_dev=2)
    b = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=8, width=4, max_dev=2)
    assert [x.key() for x in a] == [x.key() for x in b]


# ---------------------------------------------------------------------------- the two controls


def test_gumbel_draw_zero_is_the_incumbent_and_the_pool_is_diverse():
    gs = ground_set(seed=0)
    p = scores_for(gs, 6)
    want = np.sort(NC.decode(ClusterEnv(gs, budget=2), p))
    pool = BM.gumbel_decode(ClusterEnv(gs, budget=2), p, k=16, temperature=1.0,
                            rng=np.random.default_rng(0))
    assert BM.rank_of(pool, want) is not None
    assert len(pool) > 1, "the Gumbel pool collapsed: it would price nothing"
    assert len({s.key() for s in pool}) == len(pool)


def test_force_decode_contains_the_incumbent_and_returns_systems():
    gs = ground_set(n_taxa=8, n_clusters=24, seed=0)
    p = scores_for(gs, 8)
    want = np.sort(NC.decode(ClusterEnv(gs, budget=2), p))
    pool = BM.force_decode(ClusterEnv(gs, budget=2), p, k=12, width=8)
    assert BM.rank_of(pool, want) is not None
    assert len(pool) > 1
    for sysm in pool:
        env = ClusterEnv(gs, budget=2)
        env.reset(init="given", given=[v for v in sysm.selection if not env.state.in_s[int(v)]])
        assert env.verify()


def test_corridor_complete_is_unchanged_when_no_key_is_given():
    """The ``key=`` parameter added for ``force_decode`` must not move the landed default."""
    from opc.rl import corridor as CO
    gs = ground_set(seed=0)
    env = ClusterEnv(gs, budget=2)
    a = CO.complete(env, [[]])
    b = CO.complete(env, [[]], key=np.asarray(env.support, dtype=np.float64))
    c = greedy_support_rollout(ClusterEnv(gs, budget=2))
    assert a.size > 0
    assert np.array_equal(np.sort(a), np.sort(b))
    assert np.array_equal(np.sort(a), np.sort(c))


# ---------------------------------------------------------------------------- heredity

def test_the_decode_accept_sequence_is_strictly_rank_increasing():
    """The concrete form of the property the canonical order rests on.

    It is a corollary of illegality being monotone -- greedy takes the best legal cluster, and a
    cluster it passed over cannot become legal later -- and it fails *before* the width-1 parity
    would if heredity ever broke, which is why it is a test of its own.
    """
    for seed in (0, 1, 2):
        gs = ground_set(seed=seed)
        p = scores_for(gs, seed + 30)
        _order, rank = BM.rank_order(p)
        env = ClusterEnv(gs, budget=2)
        picks = []
        obs = env.reset()
        while obs["legal"].any():
            cand = np.flatnonzero(obs["legal"])
            a = int(cand[np.argmin(rank[cand])])
            picks.append(int(rank[a]))
            obs, _r, done, _i = env.step(a)
            if done:
                break
        assert len(picks) > 2, "fewer than three picks: the monotonicity claim would be vacuous"
        assert picks == sorted(picks) and len(set(picks)) == len(picks)


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_the_legal_set_only_shrinks(budget):
    """Including the budget term -- the one condition that is not obviously downward-closed."""
    gs = ground_set(seed=1)
    p = scores_for(gs, 41)
    _order, rank = BM.rank_order(p)
    env = ClusterEnv(gs, budget=budget)
    obs = env.reset()
    prev, steps = obs["legal"].copy(), 0
    while obs["legal"].any():
        cand = np.flatnonzero(obs["legal"])
        a = int(cand[np.argmin(rank[cand])])
        obs, _r, done, _i = env.step(a)
        gained = obs["legal"] & ~prev
        assert not gained.any(), f"the legal set gained {int(gained.sum())} bits at step {steps}"
        prev = obs["legal"].copy()
        steps += 1
        if done:
            break
    assert steps > 0, "no step taken: the test would be vacuous"


# ---------------------------------------------------------------------------- the option profile


def test_the_option_profile_sizes_a_beam_that_then_reaches_the_system():
    """Round trip: profile a known feasible system, then reach it at exactly that ``(w, D)``.

    The target is a *different* score vector's decode, so it is a genuine alternative under the
    scores being profiled rather than the incumbent itself.
    """
    gs = ground_set(seed=0)
    p = scores_for(gs, 51)
    target = np.sort(NC.decode(ClusterEnv(gs, budget=2), scores_for(gs, 52)))
    base = np.sort(NC.decode(ClusterEnv(gs, budget=2), p))
    assert not np.array_equal(target, base), "target is the incumbent: profile is trivial"

    prof = BM.truth_option_profile(ClusterEnv(gs, budget=2), p, target)
    assert prof["truth_path_feasible"], "heredity says any order of a feasible set is acceptable"
    assert prof["dev_needed"] >= 1, "target reached with no deviation: vacuous"

    pool = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=100_000, beam=100_000,
                          width=prof["width_needed"], max_dev=prof["dev_needed"])
    assert BM.rank_of(pool, target) is not None
    narrow = BM.beam_decode(ClusterEnv(gs, budget=2), p, k=100_000, beam=100_000,
                            width=prof["width_needed"], max_dev=prof["dev_needed"] - 1)
    assert BM.rank_of(narrow, target) is None, "one deviation fewer still reached it: not tight"


def test_the_option_profile_restores_the_env():
    gs = ground_set(seed=0)
    env = ClusterEnv(gs, budget=2)
    before = env.state
    BM.truth_option_profile(env, scores_for(gs, 1), NC.decode(ClusterEnv(gs, budget=2),
                                                              scores_for(gs, 2)))
    assert env.state is before


# ---------------------------------------------------------------------------- shared-env hygiene


def test_a_terminal_env_is_reset_before_the_beam_snapshots_it():
    """``decode`` leaves the env in its **terminal** state and its docstring does not say so.

    A beam that snapshotted after a decode would search from a maximal system, return one member,
    and report a plausible ceiling equal to the incumbent -- with nothing raising.
    """
    gs = ground_set(seed=0)
    p = scores_for(gs, 61)
    env = ClusterEnv(gs, budget=2)
    NC.decode(env, p)                       # leaves env at a maximal system
    assert not env.legal_mask().any()
    pool = BM.beam_decode(env, p, k=16, beam=16, width=4, max_dev=1)
    assert len(pool) > 1, "the beam searched from the terminal state"


def test_gumbel_refuses_a_collapsed_pool():
    gs = ground_set(seed=0)
    p = scores_for(gs, 71)
    with pytest.raises(ValueError, match="collapsed"):
        BM.gumbel_decode(ClusterEnv(gs, budget=2), p, k=16, temperature=1e-6,
                         rng=np.random.default_rng(0))


def test_beam_zero_is_unbounded_and_beam_none_is_k():
    """``--beam 0`` must mean 'no pruning', not 'keep k' -- see the comment it guards."""
    gs = ground_set(n_taxa=6, n_clusters=10, seed=0)
    p = scores_for(gs, 21)
    kw = dict(k=10_000, width=0, max_dev=-1)
    unbounded = BM.beam_decode(ClusterEnv(gs, budget=2), p, beam=0, **kw)
    explicit = BM.beam_decode(ClusterEnv(gs, budget=2), p, beam=100_000, **kw)
    assert [s.key() for s in unbounded] == [s.key() for s in explicit]
    assert len(unbounded) > 2, "the fixture is too small to tell pruning from not"
    narrow = BM.beam_decode(ClusterEnv(gs, budget=2), p, beam=1, **kw)
    assert len(narrow) < len(unbounded), "beam=1 pruned nothing: the distinction is vacuous"
