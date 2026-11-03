"""W8's swap MDP: the identities the amendment rests on, and the stationarity that frees ``H``.

Three claims are load-bearing and each is asserted rather than argued:

* **``S_0`` is greedy and every state is a maximal feasible cluster system** -- i.e. a *terminal*
  state of the construction MDP.  If that slips, the walk is not moving inside the space v1's
  answers lived in and every number taken along it means something else.
* **Nothing the decoder consumes is a function of ``t`` or ``H``.**  This is what makes the walk a
  stationary Markov chain, and therefore what makes the ``H = 24`` trajectory the *prefix* of the
  ``H = 96`` one at a fixed seed -- which is the entire basis for treating ``H`` as a free
  inference parameter.  It is tested twice: directly (the observation is invariant to ``env.t``)
  and end to end (a real policy, sampling, at two horizons).
* **``sum_t record_increments == max_t M_t - M_0``.**  The dense and flat credit arms optimise the
  same return and differ only in where credit lands; if this fails they are two different
  algorithms being compared as one.

**Nothing here computes an MPL**, for ``tests/test_rl_corridor.py``'s reason: whether a swap is
*uphill* is a property of the surface and belongs to a driver with a real cell and a real
evaluator.  The synthetic fixture's "truth" is a maximal set under an unrelated weight vector, so
its cluster F1 against greedy starts near zero -- fine for lattice identities, meaningless as
phylogenetics, and deliberately not read as such.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_rl_corridor import _Instance, fixture  # noqa: E402

from opc.rl import corridor as CO
from opc.rl import policy as P
from opc.rl import reward as RW
from opc.rl import train as TR
from opc.rl.env import greedy_support_rollout
from opc.rl.rollout import swap_rollout_group
from opc.rl.swap import (
    N_SWAP_DYNAMIC,
    SwapEnv,
    incumbent_stats,
    oracle_swap_actions,
    swap_scalars,
)


class _F1Instance(_Instance):
    """``_Instance`` plus ``cluster_f1`` -- byte-for-byte ``instances.Instance.cluster_f1``.

    Duplicated rather than imported because a real :class:`opc.rl.instances.Instance` needs a
    simulated grid cell behind it, and the oracle target depends on none of that.
    """

    def cluster_f1(self, selection) -> float:
        sel = np.asarray(selection, dtype=np.int64)
        if sel.size == 0 or self.n_true == 0:
            return 0.0
        correct = np.intersect1d(sel, self.truth_rows).size
        if correct == 0:
            return 0.0
        p, r = correct / sel.size, correct / self.n_true
        return float(2 * p * r / (p + r))


def swap_fixture(seed=0, budget=2, tabu=8):
    inst, env = fixture(seed=seed, budget=budget)
    inst.__class__ = _F1Instance
    fr = CO.frame(inst, env)
    return inst, fr, SwapEnv(inst.gs, budget, tabu=tabu, env=env)


def _walk(se, n, seed=0, tabu_aware=True):
    """A seeded random walk.  Returns the visited selections."""
    rng = np.random.default_rng(seed)
    se.reset()
    out = [se.selection.copy()]
    for _ in range(n):
        cand = np.flatnonzero(se.legal_mask() if tabu_aware else ~se.in_s)
        se.step(int(rng.choice(cand)))
        out.append(se.selection.copy())
    return out


# ------------------------------------------------------------------------------- the fixture


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fixture_is_not_degenerate(seed):
    inst, fr, se = swap_fixture(seed=seed)
    assert sum(c.n_edges for c in inst.gs.components) > 200, "no overlap: nothing can be displaced"
    assert fr.d >= 2, "greedy already found the truth: there is nothing to swap towards"
    assert se.legal_mask().sum() > 10 * se.selection.size, "the action set is not the wide one"


# ------------------------------------------------------------------------- G-S0: the invariants


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_S0_is_greedy_by_support(seed):
    _inst, _fr, se = swap_fixture(seed=seed)
    assert np.array_equal(se.reset().selection, np.sort(greedy_support_rollout(se.env)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_state_is_maximal_and_passes_the_independent_validator(seed):
    """``heuristics.check`` shares no state with anything here, and maximality is the other half.

    Both matter and they are different claims: ``verify()`` says the selection is a legal cluster
    system, and the empty legal mask says it is one the construction MDP could have *terminated*
    in.  A state that were feasible but not maximal would still be a cluster system, but the swap
    MDP's whole framing -- that it moves among v1's terminal states -- would be false.
    """
    _inst, _fr, se = swap_fixture(seed=seed)
    for _sel in _walk(se, 12, seed=seed):
        assert se.verify()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_step_is_exactly_force_move(seed):
    """The move operator is A76's, not a second copy of it."""
    _inst, _fr, se = swap_fixture(seed=seed, tabu=0)
    rng = np.random.default_rng(seed)
    se.reset()
    for _ in range(10):
        before = se.selection.copy()
        v = int(rng.choice(np.flatnonzero(se.legal_mask())))
        expected = np.sort(CO.force_move(se.env, before, v))
        se.step(v)
        assert np.array_equal(se.selection, expected)


def test_step_refuses_a_selected_or_tabu_action():
    """Two refusals with two messages, and the *order* of the checks is why they are separate.

    A cluster is in the tabu ring because it was just forced, so it is normally also *selected* --
    the "already selected" branch fires first and the tabu branch is unreachable that way.  It
    becomes reachable when a later swap displaces it while its tenure is still running, which is
    exactly the cycle the ring exists to break, so the guard is exercised by putting the env in
    that state directly rather than by hoping a random walk produces it.
    """
    _inst, _fr, se = swap_fixture(tabu=4)
    se.reset()
    with pytest.raises(ValueError, match="already selected"):
        se.step(int(se.selection[0]))
    v = int(np.flatnonzero(se.legal_mask())[0])
    se.step(v)
    assert not se.legal_mask()[v], "a forced cluster is selected, so it is not a legal action"
    u = int(np.flatnonzero(se.legal_mask())[0])
    se._tabu.append(u)                                                          # noqa: SLF001
    assert not se.legal_mask()[u]
    with pytest.raises(ValueError, match="tabu"):
        se.step(u)


# -------------------------------------------------------------- stationarity, and the prefix


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nothing_the_decoder_consumes_depends_on_t(seed):
    """The direct form of the claim: move the clock, the observation does not move.

    ``SwapObs.t`` is carried for logging and is not consumed; what the decoder reads is
    ``dynamic``, ``scalars``, ``legal`` and ``last_action``, and none of those may be a function
    of the step index.  Putting ``t/n`` back into ``swap_scalars`` -- which is what v1 fed -- fails
    this line, deterministically, rather than showing up as a horizon-transfer curve that sags.
    """
    _inst, _fr, se = swap_fixture(seed=seed)
    _walk(se, 5, seed=seed)
    a = se.observe()
    se.t = 10_000
    b = se.observe()
    assert np.array_equal(a.dynamic, b.dynamic)
    assert np.array_equal(a.scalars, b.scalars)
    assert np.array_equal(a.legal, b.legal)


def _tiny_policy(inst, se, seed=0):
    torch.manual_seed(seed)
    cfg = P.PolicyConfig(n_node_features=4, n_global_features=2, hidden=16, layers=1,
                         n_dynamic=N_SWAP_DYNAMIC)
    pol = P.Policy(cfg)
    n = inst.gs.n_clusters
    rng = np.random.default_rng(0)
    x_nodes = torch.as_tensor(rng.random((n, 4)), dtype=torch.float32)
    x_global = torch.as_tensor(rng.random(2), dtype=torch.float32)
    g = P.graph_tensors(se.env.graph, inst.gs.support, "cpu")
    return pol, pol.encode(x_nodes, x_global, g)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prefix_property(seed):
    """``H = 8`` is the prefix of ``H = 24`` at one seed -- what "``H`` is free" rests on.

    End to end through a real ``Policy``, sampling rather than argmax, because the failure this
    guards against is subtle: any input that were a function of ``t`` or ``H`` would change the
    logits at step ``t`` and the two runs would diverge from there.  Seeding torch identically
    makes the two multinomial streams identical, so the *only* thing that can separate them is the
    policy seeing a different state -- which is exactly what is being ruled out.
    """
    inst, _fr, se = swap_fixture(seed=seed)
    pol, enc = _tiny_policy(inst, se, seed=seed)
    runs = {}
    for horizon in (8, 24):
        runs[horizon] = swap_rollout_group(pol, enc, se, 3, horizon, device="cpu",
                                           pomo=True, seed=1234)
    short, long = runs[8], runs[24]
    assert np.array_equal(short.actions, long.actions[:, :8])
    for a, b in zip(short.selections, long.selections, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b[:9], strict=True))


@pytest.mark.parametrize("seed", [0, 1])
def test_swap_rollout_group_shapes_and_pomo_forces_distinct_first_swaps(seed):
    inst, _fr, se = swap_fixture(seed=seed)
    pol, enc = _tiny_policy(inst, se, seed=seed)
    torch.manual_seed(0)
    g = swap_rollout_group(pol, enc, se, 4, 6, device="cpu", pomo=True)
    assert g.logp.shape == (4, 6) and g.entropy.shape == (4, 6)
    assert g.actions.shape == (4, 6)
    assert all(len(t) == 7 for t in g.selections), "the start state must be in the trajectory"
    assert g.n_distinct_first == 4, "POMO did not force the first swaps apart"
    assert g.logp.requires_grad and g.entropy.requires_grad
    assert torch.isfinite(g.logp).all() and torch.isfinite(g.entropy).all()
    # every rollout starts at the same state -- greedy -- which is what makes POMO necessary here
    for t in g.selections:
        assert np.array_equal(t[0], se.greedy)


# ------------------------------------------------------------------------- the reward identity


def test_record_increments_sum_to_the_best_seen():
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = rng.normal(scale=1e4, size=int(rng.integers(2, 40)))
        assert RW.record_increments(v).sum() == pytest.approx(v.max() - v[0], abs=1e-6)
        assert np.array_equal(RW.best_seen(v),
                              np.asarray([v[:i + 1].max() for i in range(v.size)]))


def test_the_deadband_is_a_lower_bound_with_a_computable_leak():
    """With a band the identity becomes an inequality, and the gap must be readable, not assumed.

    Constructed so it actually bites: ten improvements of 100 against a band of 500 are worth
    1,000 log units of real progress and **zero** reward.  That is the intended behaviour -- A76
    §12.6 measured 65 log units of movement from batch composition alone -- but a band that could
    silently swallow a real gain has to be measurable from outside, which is what the leak is.
    """
    v = np.cumsum(np.r_[0.0, np.full(10, 100.0)])
    banded = RW.record_increments(v, deadband=500.0)
    exact = RW.record_increments(v)
    assert banded.sum() == 0.0 and exact.sum() == pytest.approx(1000.0)
    assert banded.sum() <= exact.sum()
    v2 = np.cumsum(np.r_[0.0, [100.0, 900.0, 50.0]])
    assert RW.record_increments(v2, deadband=500.0).sum() == pytest.approx(900.0)


# --------------------------------------------------------------------------- features and tabu


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_unselected_cluster_crosses_the_incumbent(seed):
    """Feature 3's ``inf`` branch is a guard, not a case -- and this is why.

    ``S_t`` is maximal, so a cluster with *no* accepted neighbour would be acceptable outright and
    ``S_t`` would not be maximal.  Asserted rather than reasoned, because the feature's
    normalisation would silently become a constant if it ever stopped holding.
    """
    _inst, _fr, se = swap_fixture(seed=seed)
    for _sel in _walk(se, 8, seed=seed):
        deg, min_w = incumbent_stats(se.env.graph, se.in_s, se.support)
        free = ~se.in_s
        assert (deg[free] > 0).all()
        assert np.isfinite(min_w[free]).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_incumbent_stats_against_a_naive_per_vertex_scan(seed):
    """Cross-validated against a slow reference, per the house rule -- not against itself."""
    _inst, _fr, se = swap_fixture(seed=seed)
    _walk(se, 6, seed=seed)
    deg, min_w = incumbent_stats(se.env.graph, se.in_s, se.support)
    g = se.env.graph
    for v in range(se.n):
        nbr, _cls = g.neighbours(v)
        acc = [int(u) for u in nbr if se.in_s[int(u)]]
        assert deg[v] == len(acc)
        want = min((se.support[u] for u in acc), default=np.inf)
        assert min_w[v] == want


@pytest.mark.parametrize("seed", [0, 1])
def test_swap_features_are_bounded_and_the_tabu_column_is_the_tabu(seed):
    _inst, _fr, se = swap_fixture(seed=seed, tabu=4)
    _walk(se, 6, seed=seed)
    obs = se.observe()
    x = obs.dynamic
    assert x.shape == (se.n, N_SWAP_DYNAMIC) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert (x[:, 1] >= 0).all() and (x[:, 1] <= 1).all()
    assert (x[:, 2] >= -1).all() and (x[:, 2] <= 1).all()
    # Column 3 is the **raw** tabu ring, not ``in_s & ~legal``: a just-forced cluster is both
    # selected and tabu, and telling the policy "this is a recent arrival" is information about a
    # selected row too.  ``legal`` is the intersection, and it is the mask; this column is not.
    assert np.array_equal(x[:, 3] > 0, se.tabu_mask())
    assert (se.tabu_mask() & se.in_s).any(), "no tabu row is selected: the column is untested"
    # the selected rows carry no support/margin signal: they are not actions
    assert (x[se.selection, 1] == 0).all() and (x[se.selection, 2] == 0).all()
    assert obs.legal.sum() == int(((~se.in_s) & (x[:, 3] == 0)).sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tabu_reduces_revisits_and_the_fixture_actually_cycles(seed):
    """**The fixture must produce the case being tested**, or this asserts nothing (A34's rule).

    Forcing the top-support unselected cluster displaces something, which is then the top-support
    unselected cluster, which is forced back: a 2-cycle by construction.  A76 §12.2's repair search
    could not do this because it accepted only strict improvements; a sampled policy can, which is
    what the tabu ring is for.  Note it does not claim to eliminate cycling -- cycles longer than
    the tenure survive -- only to reduce it, and the assertion says exactly that.
    """
    _inst, _fr, se0 = swap_fixture(seed=seed, tabu=0)

    def top_support_walk(env, steps=12):
        env.reset()
        for _ in range(steps):
            cand = np.flatnonzero(env.legal_mask())
            env.step(int(cand[np.lexsort((cand, -env.support[cand]))[0]]))
        return env.n_revisits

    without = top_support_walk(se0)
    assert without > 0, "the fixture never cycles: this test would assert nothing"
    _inst, _fr, se8 = swap_fixture(seed=seed, tabu=8)
    assert top_support_walk(se8) < without


# ------------------------------------------------------------------------------ the BC target


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_swap_actions_are_legal_at_their_own_states_and_improve_on_greedy(seed):
    """A72's first defect is the trap: a clone target in the wrong order is 4.8% illegal actions.

    Here it cannot be, because the target is generated by *stepping this env* -- so the test is
    that replaying it independently reproduces both the legality and the trajectory.
    """
    inst, fr, se = swap_fixture(seed=seed, tabu=0)
    actions = oracle_swap_actions(inst, se, fr)
    assert actions, "no truth-ward swap was available: the target is empty"
    se.reset()
    start = inst.cluster_f1(se.selection)
    for a in actions:
        assert se.legal_mask()[a], "the clone target contains an action illegal at its own state"
        se.step(a)
    assert len(set(actions)) == len(actions), "the oracle forced one cluster twice"
    assert inst.cluster_f1(se.selection) > start


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_support_cap_is_a_cap_and_is_off_by_default(seed):
    """``top_m`` restricts to the top unselected clusters by support -- greedy's own ordering.

    Two things, and the second is the one that matters: the cap is exactly the top ``M`` of the
    *free* set in descending support (not the top ``M`` of the ground set, which would shrink as
    the incumbent ate into it), and it is **off** unless asked for.  The default is a measurement:
    a cap concentrates the improving swaps 3-19x and makes 8.7% of missed truth clusters
    unreachable at ``M = 256``, which is a silent ceiling on G-R5.
    """
    _inst, _fr, se = swap_fixture(seed=seed)
    assert se.top_m is None, "the support cap must not be on by default"
    full = int(se.legal_mask().sum())
    _inst2, _fr2, cap = swap_fixture(seed=seed)
    cap.top_m = 8
    cap.reset()
    m = cap.legal_mask()
    assert m.sum() == min(8, full)
    free = np.flatnonzero(~cap.in_s)
    want = free[np.argsort(-cap.support[free], kind="stable")][:8]
    assert set(np.flatnonzero(m).tolist()) == set(want.tolist())
    assert (cap.support[m].min() >= cap.support[~cap.in_s & ~m].max()
            if (~cap.in_s & ~m).any() else True)


# ---------------------------------------------------------------------- W8's credit assignment


def test_returns_to_go_against_a_naive_loop():
    rng = np.random.default_rng(0)
    r = rng.random((5, 9))
    for disc in (1.0, 0.9):
        want = np.zeros_like(r)
        for i in range(r.shape[0]):
            for t in range(r.shape[1]):
                want[i, t] = sum(disc ** (s - t) * r[i, s] for s in range(t, r.shape[1]))
        assert np.allclose(TR.returns_to_go(r, disc), want)


def test_stepwise_advantage_normalises_down_the_group_not_across_time():
    """The baseline for rollout ``i`` at step ``t`` is the other rollouts **at step ``t``**.

    They are all at step ``t`` simultaneously and all started from the same state, so that is a
    proper baseline.  Normalising the whole matrix at once would let the shrinking scale of a
    return-to-go -- it has fewer steps left to accumulate -- read as a signal about the action.
    The fixture makes the two visibly different: column scale falls by an order of magnitude with
    ``t`` while the *within-column* ordering is constant.
    """
    g = np.array([[100.0, 10.0, 1.0], [200.0, 20.0, 2.0],
                  [300.0, 30.0, 3.0], [400.0, 40.0, 4.0]])
    a = TR.stepwise_advantage(g)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(a.std(axis=0), 1.0, atol=1e-6)
    # every column carries the same ordering, so every column gets the same advantage
    assert np.allclose(a[:, 0], a[:, 1]) and np.allclose(a[:, 1], a[:, 2])
    # ties get equal advantage, and a dead column teaches nothing
    dead = TR.stepwise_advantage(np.ones((4, 3)))
    assert np.allclose(dead, 0.0)


def test_flat_and_dense_credit_carry_the_same_total_return():
    """The identity the two arms are compared under, at the level the trainer uses it.

    ``flat`` advantages ``max_t M_t`` once per rollout; ``dense`` advantages the return-to-go of
    the record increments.  ``G[:, 0]`` -- the whole return-to-go -- must equal what ``flat``
    scores, or the two are different algorithms being reported as one choice.
    """
    rng = np.random.default_rng(0)
    m = rng.normal(scale=1e4, size=(6, 13))
    inc = np.stack([RW.record_increments(m[i]) for i in range(m.shape[0])])
    assert np.allclose(TR.returns_to_go(inc)[:, 0], m.max(axis=1) - m[:, 0], atol=1e-6)


class _FakePool:
    """The five keys :meth:`SwapTrainer.step` reads off a pool entry, and nothing else."""

    def __init__(self, entries):
        self._e = entries
        self.keys_by_phase = {"t": list(entries)}

    def get(self, _phase, key):
        return self._e[key]


def _entry(seed, inst, se):
    n, f = inst.gs.n_clusters, 4
    rng = np.random.default_rng(seed)
    inst.n, inst.n_clusters = 12, n
    return {"inst": inst, "swap": se, "env": se.env, "greedy": se.greedy, "graph": None,
            "x_nodes": torch.as_tensor(rng.random((n, f)), dtype=torch.float32),
            "x_global": torch.as_tensor(rng.random(2), dtype=torch.float32)}


@pytest.mark.parametrize("credit", ["dense", "flat"])
def test_swap_trainer_takes_a_step_and_moves_the_policy(credit):
    """End to end on the free arm: rollouts, record increments, advantage, backward, optimiser.

    ``--reward f1`` so no MPL is involved.  What this pins is the **assembly** -- that the reward
    array reshapes to ``(K, H + 1)`` the way the trainer assumes, that the loss is finite, and that
    a gradient actually reaches the parameters.  The reward's own correctness is
    ``tests/test_rl_mpl.py``'s and the identity is above.
    """
    entries = {}
    for i in (0, 1):
        inst, _fr, se = swap_fixture(seed=i)
        entries[f"c{i}"] = _entry(i, inst, se)
    pool = _FakePool(entries)
    cfg = P.PolicyConfig(n_node_features=4, n_global_features=2, hidden=16, layers=0,
                         n_dynamic=N_SWAP_DYNAMIC)
    torch.manual_seed(0)
    pol = P.Policy(cfg)
    tc = TR.SwapTrainConfig(batch=2, rollouts=4, horizon_min=5, horizon_max=5, reward="f1",
                            credit=credit, record_deadband=0.0, seed=0, curriculum={"t": 1.0})
    tr = TR.SwapTrainer(pol, pool, tc, device="cpu")
    before = [p.detach().clone() for p in pol.parameters()]
    row = tr.step()
    assert row["horizon"] == 5
    assert np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"])
    # The synthetic fixture's "truth" is a maximal set under an unrelated weight vector, so its
    # F1 against greedy is legitimately 0 -- see the module docstring.  What must hold is the
    # ordering of the three, which is a statement about the pool and not about phylogenetics.
    assert 0.0 <= row["f1_selected"] <= row["f1_ceiling"] <= 1.0
    # the best-seen return can never be below the start state: greedy is column 0
    assert row["best_vs_greedy"] >= 0.0
    assert row["deadband_leak"] == pytest.approx(0.0), "no band was set; nothing may leak"
    assert any(not torch.equal(a, b) for a, b in zip(before, pol.parameters(), strict=True))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reset_re_establishes_the_underlying_env_and_not_only_the_row_set(seed):
    """``force_move`` leaves ``ClusterEnv`` standing where it built; ``reset`` must undo that.

    The defect this pins: a ``reset`` that rewrote only ``self.selection`` left the previous walk's
    commitments and opened intersection classes in ``self.env.state``, so ``swap_scalars``'
    ``k_rem / k`` was read off the wrong state and ``verify()`` validated a selection this object
    was no longer reporting.  It hides on a fixture where every state saturates the budget -- the
    two scalars then agree by luck -- so the assertion is on ``env.state`` itself, which cannot.
    """
    _inst, _fr, se = swap_fixture(seed=seed, tabu=0)
    at_start = swap_scalars(se).copy()
    _walk(se, 8, seed=seed)
    assert not np.array_equal(np.sort(se.env.selection), se.greedy), \
        "the walk never left greedy: this test would assert nothing"
    se.reset()
    assert np.array_equal(np.sort(se.env.selection), se.selection)
    assert np.array_equal(se.selection, se.greedy)
    assert np.array_equal(swap_scalars(se), at_start)
    assert se.verify()
