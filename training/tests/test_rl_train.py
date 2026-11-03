"""W5 -- the rollout engine, the two reward arms, and the group-normalised advantage.

The theme is the same one the rest of this suite holds to: **nothing is checked against itself**.
The batched log-probs are checked against a per-episode recomputation; the ``f1`` reward against
``opc.solve.exact.recovery_columns``, which re-derives the truth's clusters from the network; the
reward cache against the uncached path; the deduplicated fit against the undeduplicated one.

Two of these guard failures that would be *invisible* in a training curve rather than loud:

* a reward that has drifted from the metric it is supposed to be still produces a smooth curve;
* a cache key missing a component that changes the answer returns a stale value forever, and the
  run converges beautifully to the wrong thing.  That is A52's defect, and it has now appeared in
  four places in this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opc import arms as A  # noqa: E402
from opc import bitset  # noqa: E402
from opc.mpl import triples  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl import reward as RW  # noqa: E402
from opc.rl import train as TR  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR  # noqa: E402
from opc.rl.rollout import rollout_group  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.solve import candidates, exact  # noqa: E402
from opc.solve import truth as T  # noqa: E402


def fake_tree_stats(gs, seed: int = 0):
    """W13's tree-derived block for a **synthetic** ground set that has no gene trees behind it.

    Random rather than zeros on purpose: a zero block would leave every test below vacuous with
    respect to the twenty-one columns W13 added -- the encoder would be exercised on the same ten
    features it always was, and a bug in how the wider matrix is projected would pass.

    ``node_features`` takes this as a required argument rather than defaulting it, so that a
    production caller which forgets it fails loudly instead of training on zeros.
    """
    import numpy as _np

    from opc.clusters import blen as _blen
    return _np.random.default_rng(seed).random(
        (int(gs.n_clusters), len(_blen.TREE_FEATURES))).astype(_np.float32)

def tiny_instance(n=9, seed=5, k=2, n_noise=60, m=50):
    """A real network's clusters, plus noise, wrapped as an :class:`opc.rl.instances.Instance`.

    Small enough for a unit test and *real* enough to exercise the parts that matter: the truth is
    an actual galled tree, its clusters are actually in the ground set, and the noise clusters
    actually overlap it, so (L) and (N3O) both bite.
    """
    net = N.with_gamma(N.scaled(N.with_reticulations(N.build_base(n, r_max=k, seed=seed), k), 0.5),
                       0.4)
    rows = [bitset.from_indices(sorted(int(t) for t in c), n) for c in N.nontrivial_clusters(net)]
    rng = np.random.default_rng(seed)
    seen = {r.tobytes() for r in rows}
    # ``len(seen) == len(rows)`` always, so ``while len(rows) < len(seen) + n_noise`` never
    # terminates.  Fixing a test fixture that hangs is cheaper than one that passes vacuously,
    # but only just: a hang looks like a slow test until someone times it.
    target = len(rows) + n_noise
    while len(rows) < target:
        size = int(rng.integers(2, n - 1))
        w = bitset.from_indices(sorted(rng.choice(n, size=size, replace=False).tolist()), n)
        if w.tobytes() in seen:
            continue
        seen.add(w.tobytes())
        rows.append(w)
    words = bitset.pack_rows(rows, n)
    support = rng.integers(1, m, size=len(rows)).astype(np.int32)
    support[:len(N.nontrivial_clusters(net))] = m          # the truth is well supported
    gs = candidates.build(words, support, n, m=m, tau=0)
    graph = OverlapCSR.of(gs)
    index = triples.triple_index(n)
    obs = rng.multinomial(m, [1 / 3, 1 / 3, 1 / 3], size=index.shape[0]).astype(float)
    return I.Instance(
        key=f"t{n}_{seed}", n=n, k=k, tau=0, m=m, cu=0.5, gamma=0.4, stratum="small", rep=0,
        gs=gs, graph=graph, obs=obs, index=index, truth=net,
        truth_rows=T.true_cluster_rows(gs.words, net),
        n_true=len(N.nontrivial_clusters(net)), labels=tuple(net.labels),
        x_nodes=I.node_features(gs, graph, fake_tree_stats(gs)), x_global=I.global_features(gs, graph, k))


def small_policy(inst, **kw):
    torch.manual_seed(0)
    cfg = P.PolicyConfig(n_node_features=inst.x_nodes.shape[1],
                         n_global_features=inst.x_global.shape[0],
                         hidden=kw.pop("hidden", 32), layers=kw.pop("layers", 2), **kw)
    return P.Policy(cfg)


# --------------------------------------------------------------------------- the advantage


def test_z_advantage_is_centred_and_scaled():
    r = np.array([1.0, 2.0, 3.0, 10.0])
    a = TR.advantage(r, "z")
    assert abs(a.mean()) < 1e-9
    assert abs(a.std() - 1.0) < 1e-6


def test_rank_advantage_is_bounded_and_ties_get_the_same_advantage():
    """Two rollouts that reach the same selection get the same reward and must not be split."""
    r = np.array([5.0, 5.0, 1.0, 9.0])
    a = TR.advantage(r, "rank")
    assert a.min() >= -1.0 and a.max() <= 1.0
    assert a[0] == a[1], "tied rewards were given different advantages"
    assert a[2] < a[0] < a[3]
    # A degenerate group must produce no gradient signal at all, under either rule.
    for kind in ("z", "rank"):
        flat = TR.advantage(np.full(8, 3.3), kind)
        assert np.abs(flat).max() < 1e-6, f"{kind} invented a signal in a collapsed group"


def test_split_is_disjoint_and_by_replicate():
    keys = [f"n50_rep{r}_r{k}_cu0.5_g0.5_small" for r in range(20) for k in (1, 2)]
    s = TR.split_by_rep(keys)
    reps = {name: {int(x.split("_rep")[1].split("_")[0]) for x in v} for name, v in s.items()}
    assert reps["train"] & reps["val"] == set()
    assert reps["train"] & reps["test"] == set()
    assert reps["val"] & reps["test"] == set()
    assert sum(len(v) for v in s.values()) == len(keys)
    assert len(reps["test"]) == 2 and len(reps["val"]) == 3


def test_split_by_rep_refuses_a_cut_that_belongs_to_another_grid():
    """The defaults are the S-grids' twenty replicates.  On a 900-replicate phase they give
    15 train / 3 val / 882 test -- *silently*, which is the failure this guard exists to convert
    into a loud one.  ``w4_bc``, ``w5_train``, ``w8_swap``, ``w6_eval`` and ``w7_corridor`` all
    still call in with the defaults, and adding the x-phases to ``grid.PHASES`` made them
    reachable."""
    big = [f"n25_rep{r}_r1_cu0.5_gU0.25-0.5_mid" for r in range(900)]
    with pytest.raises(ValueError, match="validation cut"):
        TR.split_by_rep(big)
    got = TR.split_by_rep(big, 700, 800)
    assert [len(got[s]) for s in ("train", "val", "test")] == [700, 100, 100]
    # ... and it is a no-op on every grid the defaults were written for.
    assert TR.split_by_rep([f"n50_rep{r}_r1_cu0.5_g0.5_small" for r in range(20)])
    assert TR.split_by_rep([f"n25_rep{r}_r1_cu0.5_gU0.25-0.5_mid" for r in range(100)], 70, 80)


def test_lazy_env_defers_construction_and_yields_the_same_objects(tmp_path):
    """``lazy_env`` is a memory switch, so what has to be pinned is that it changes *nothing else*.

    An eager pool and a lazy one must produce the same greedy selection and the same legal mask;
    if they did not, W10's validation decode would depend on a flag that is only about RAM.
    """
    import pickle

    inst = tiny_instance()
    cache = I.cache_dir(tmp_path, "unit", 0)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{inst.key}.pkl").write_bytes(pickle.dumps(inst))

    keys = {"unit": [inst.key]}
    eager = TR.InstancePool(keys, tmp_path, 0, None, size=4)
    lazy = TR.InstancePool(keys, tmp_path, 0, None, size=4, lazy_env=True)

    assert eager.get("unit", inst.key)["env"] is not None
    e = lazy.get("unit", inst.key)
    assert e["env"] is None and e["greedy"] is None, "lazy_env built the env on the miss"

    assert np.array_equal(lazy.greedy_of("unit", inst.key), eager.greedy_of("unit", inst.key))
    assert lazy.get("unit", inst.key)["env"] is not None, "env_of did not memoise"
    assert np.array_equal(lazy.env_of("unit", inst.key).reset()["legal"],
                          eager.env_of("unit", inst.key).reset()["legal"])
    # A swap pool must stay eager: SwapEnv takes the env as a constructor argument.
    assert TR.InstancePool(keys, tmp_path, 0, None, size=4, lazy_env=True,
                           swap_tabu=4).lazy_env is False


# --------------------------------------------------------------------------- the rewards


def test_f1_reward_is_the_metric_it_claims_to_be():
    inst = tiny_instance()
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    rng = np.random.default_rng(0)
    sels = []
    from opc.rl.env import greedy_support_rollout, rollout
    sels.append(greedy_support_rollout(env))
    for _ in range(4):
        sels.append(rollout(env, lambda m: rng.choice(np.flatnonzero(m))))
    got = RW.f1_rewards(inst, sels)
    want = [exact.recovery_columns(inst.gs, s, inst.truth, inst.n)["align_f1"] for s in sels]
    assert np.allclose(got, want, atol=0, rtol=0), "the reward and the metric have diverged"
    assert got.max() > got.min(), "every selection scored the same -- the fixture is vacuous"


def test_topology_key_identifies_the_closed_system_not_the_selection():
    from opc.followup import search
    inst = tiny_instance(seed=7)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout
    sel = greedy_support_rollout(env)
    a = search.realise(inst.gs.words, sel, inst.n, list(inst.labels))
    assert a is not None
    # The same closed system reached from a permuted selection must hash identically.
    perm = np.random.default_rng(0).permutation(sel)
    b = search.realise(inst.gs.words, perm, inst.n, list(inst.labels))
    assert RW.topology_key(a[1]) == RW.topology_key(b[1])
    # A different system must not.
    other = sel[:-2]
    c = search.realise(inst.gs.words, other, inst.n, list(inst.labels))
    if c is not None and c[1].shape != a[1].shape:
        assert RW.topology_key(c[1]) != RW.topology_key(a[1])


def test_mpl_reward_caches_on_everything_that_changes_the_answer(tmp_path):
    """A52's rule, made a test: change any key component and the cache must **miss**."""
    inst = tiny_instance(seed=11)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout
    sels = [greedy_support_rollout(env)]

    base = RW.MPLReward(device="cpu", iters=12, restarts=1, seed=0, cache_dir=tmp_path)
    v0, s0 = base(inst, sels)
    assert s0.n_cached == 0 and s0.n_distinct == 1
    v1, s1 = base(inst, sels)
    assert s1.n_cached == 1 and np.array_equal(v0, v1), "a cache hit changed the value"

    # A fresh object, same key -> reads the value off disk rather than recomputing.
    again = RW.MPLReward(device="cpu", iters=12, restarts=1, seed=0, cache_dir=tmp_path)
    v2, s2 = again(inst, sels)
    assert s2.n_cached == 1 and np.allclose(v0, v2)

    for kw in ({"iters": 24}, {"restarts": 2}, {"seed": 1}, {"optimiser": "lbfgs"}):
        opts = {"device": "cpu", "iters": 12, "restarts": 1, "seed": 0,
                "cache_dir": tmp_path, **kw}
        other = RW.MPLReward(**opts)
        _v, st = other(inst, sels)
        assert st.n_cached == 0, f"the cache key ignores {list(kw)[0]} -- a stale value survives"


def test_dedup_broadcasts_the_same_values_the_undeduplicated_path_gives(tmp_path):
    inst = tiny_instance(seed=13)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout, rollout
    rng = np.random.default_rng(3)
    one = greedy_support_rollout(env)
    two = rollout(env, lambda m: rng.choice(np.flatnonzero(m)))
    dup = [one, two, one, two, one]                      # 5 selections, 2 distinct

    r = RW.MPLReward(device="cpu", iters=12, restarts=1, cache_dir=tmp_path)
    vals, stats = r(inst, dup)
    assert stats.n_scored == 5
    assert stats.n_distinct <= 2, "dedup did not collapse the repeats"
    assert vals[0] == vals[2] == vals[4]
    assert vals[1] == vals[3]

    solo = RW.MPLReward(device="cpu", iters=12, restarts=1, cache_dir=tmp_path / "b")
    v_one, _ = solo(inst, [one])
    assert np.isclose(v_one[0], vals[0]), "the deduplicated value differs from the direct one"


def test_chunking_dedups_first_and_is_exact_within_one_chunk(tmp_path):
    """W8's chunking: **dedup before chunk**, and a batch that fits in one chunk is unchanged.

    ``MPLReward`` fitted every distinct network in a single ``profile_mpl`` call until W8, and the
    swap MDP presents ``B x K x (H + 1)`` of them -- ~450, where ``probe_mpl_fast.py`` at 128
    already reached 38-42 GB on a 48 GB card.  ``scripts/w7_corridor.py`` got the same effect by
    wrapping this class from *outside*, and that is the ordering this pins: chunking before dedup
    re-fits a topology once per chunk it appears in, which for the swap MDP is the common case --
    the ``K`` walks of a group all begin at greedy.

    **What is deliberately not asserted** is that a split batch gives the same numbers as an
    unsplit one.  A76 §12.6 measured a network's fitted MPL moving **65 log units** with batch
    composition alone (``_initial_vectors`` draws each restart's perturbation as one contiguous
    block over the whole batch), and the cache key does not include the batch.  Chunking therefore
    *is* a change of batch composition and the honest statement is the one below: exact when
    everything fits in one chunk, and within the §12.6 band otherwise.
    """
    inst = tiny_instance(seed=19)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout, rollout
    rng = np.random.default_rng(5)
    uniq = {}
    for _ in range(40):
        sel = rollout(env, lambda m: rng.choice(np.flatnonzero(m)))
        uniq.setdefault(tuple(sel.tolist()), sel)
    distinct = list(uniq.values())[:4]
    assert len(distinct) >= 3, "the fixture produced too few distinct selections to split"
    sels = [greedy_support_rollout(env), *distinct, distinct[0], distinct[1]]

    whole = RW.MPLReward(device="cpu", iters=10, restarts=1, cache_dir=tmp_path / "w", chunk=64)
    v_whole, s_whole = whole(inst, sels)
    assert s_whole.n_chunks == 1

    split = RW.MPLReward(device="cpu", iters=10, restarts=1, cache_dir=tmp_path / "s", chunk=2)
    v_split, s_split = split(inst, sels)

    # (a) dedup ran first: the repeats at the end cost no extra fit, whatever the chunk size.
    assert s_split.n_fitted == s_whole.n_fitted, "chunking changed how many topologies were fitted"
    assert s_split.n_chunks == -(-s_split.n_fitted // 2) > 1, "the batch did not actually split"

    # (b) every position still gets a value, and repeats still agree **across a chunk boundary**.
    assert np.isfinite(v_split).all()
    assert v_split[-2] == v_split[1] and v_split[-1] == v_split[2]

    # (c) exact within one chunk, and within A76 §12.6's band when split.
    assert np.array_equal(v_whole, whole(inst, sels)[0])
    assert np.allclose(v_split, v_whole, atol=200.0), (
        "a split batch moved a fitted MPL by more than A76 §12.6's measured band")

    # (d) the per-chunk shape count is what memory tracks, and it is reported separately.
    assert 0 < s_split.shapes_max_chunk <= s_split.n_shapes_pooled


def test_an_unrealisable_selection_scores_at_a_finite_floor(tmp_path):
    """``nan``/``-inf`` would poison a group mean; a dropped rollout would change ``K``."""
    assert np.isfinite(RW.UNREALISABLE)
    inst = tiny_instance(seed=17)
    r = RW.MPLReward(device="cpu", iters=8, restarts=1, cache_dir=tmp_path)
    # The **empty** selection is not unrealisable: it is the star (A49), whose parameter vector is
    # empty, so it must be scored in closed form rather than handed to an optimiser with nothing
    # to descend -- which raises rather than being slow.
    vals, stats = r(inst, [np.array([], dtype=np.int64)])
    assert len(vals) == 1 and np.isfinite(vals[0])
    assert stats.n_parameter_free == 1, "the star was not recognised as parameter-free"
    assert np.isclose(vals[0], inst.obs.sum() * np.log(1 / 3)), "the star's closed form (A49)"


# --------------------------------------------------------------------------- rollouts


def test_rollout_logps_match_a_per_episode_recomputation():
    """The group's log-probs must be the policy's, not an artefact of how they were accumulated."""
    inst = tiny_instance(seed=19)
    policy = small_policy(inst)
    dev = torch.device("cpu")
    g = P.graph_tensors(inst.graph, inst.gs.support, dev)
    xn = torch.as_tensor(inst.x_nodes)
    xg = torch.as_tensor(inst.x_global)
    enc = policy.encode(xn, xg, g)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    out = rollout_group(policy, enc, inst, env, 4, device=dev,
                        rng=np.random.default_rng(0), pomo=True)
    assert len(out) == 4
    assert out.n_distinct >= 2, "POMO produced a collapsed group -- nothing is being tested"
    assert out.logp.shape == (4,) and torch.isfinite(out.logp).all()
    assert torch.isfinite(out.entropy).all()
    assert float(out.entropy.detach().min()) >= 0.0

    # Every terminal state is a valid cluster system, checked by the independent validator.
    for sel in out.selections:
        env.reset(init="given", given=sel)
        assert env.verify()
        assert not env.legal_mask().any(), "a terminal state was not maximal"


def test_pomo_forces_distinct_first_actions():
    inst = tiny_instance(seed=23)
    policy = small_policy(inst)
    dev = torch.device("cpu")
    g = P.graph_tensors(inst.graph, inst.gs.support, dev)
    enc = policy.encode(torch.as_tensor(inst.x_nodes), torch.as_tensor(inst.x_global), g)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    with torch.no_grad():                # a near-deterministic policy is the case that matters
        policy.key.weight.mul_(30.0)
    off = rollout_group(policy, enc, inst, env, 6, device=dev,
                        rng=np.random.default_rng(0), pomo=False, temperature=0.05)
    on = rollout_group(policy, enc, inst, env, 6, device=dev,
                       rng=np.random.default_rng(0), pomo=True, temperature=0.05)
    assert on.n_distinct >= off.n_distinct
    assert on.n_distinct >= 2


def test_the_star_is_a_provable_floor_and_a_failed_fit_is_repaired(tmp_path, monkeypatch):
    """``sup MPL(N) >= MPL(star)`` for every level-1 ``N``, so a value below it is a failed fit.

    Driving every internal branch length to zero sends every triple probability to 1/3 whatever
    the topology, and that point is inside the box the optimiser works in.  The bound therefore
    involves no modelling judgement, which is what makes it usable as a correctness check rather
    than as a heuristic -- and W8 needed one: at 100 x 2 over the topologies an exploring swap
    policy visits, **277 of 8,434 fits (3.3%) landed below the star, the worst by 9.6e8 log
    units**, against a star-to-optimum range of ~1.6e7.  Landing on the *start* state, that reads
    as an enormous reward improvement and is invisible in a training curve (A78).

    The repair is forced here rather than waited for: a real failure needs a pathological
    topology, so the optimiser is stubbed to return a value below the star once and the true value
    after, which tests the *mechanism* -- detect, re-fit, and clamp only if the re-fit fails too.
    """
    inst = tiny_instance(seed=23)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout
    sels = [greedy_support_rollout(env)]
    star = RW.star_value(inst.obs)
    assert star == pytest.approx(float(inst.obs.sum() * np.log(1 / 3)))

    honest = RW.MPLReward(device="cpu", iters=12, restarts=1, cache_dir=tmp_path / "a")
    truth, _ = honest(inst, sels)
    assert truth[0] >= star, "the fixture's own fit is already below the star"

    from opc.rl import mpl_torch as MT
    calls = {"n": 0}
    real = MT.profile_mpl

    def flaky(models, obs, **kw):
        calls["n"] += 1
        out = real(models, obs, **kw)
        if calls["n"] == 1:                      # the first fit "fails" catastrophically
            out.value = torch.full_like(out.value, star - 1e6)
        return out

    monkeypatch.setattr(MT, "profile_mpl", flaky)
    r = RW.MPLReward(device="cpu", iters=12, restarts=1, cache_dir=tmp_path / "b")
    vals, stats = r(inst, sels)
    assert calls["n"] == 2, "the failed fit was not re-fitted"
    assert stats.n_nonconverged == 1 and stats.n_unrepaired == 0
    assert vals[0] >= star
    # The repair runs at REPAIR_ITERS x REPAIR_RESTARTS the schedule, so it converges *better*
    # than the honest fit rather than identically to it -- asserting equality here was wrong.
    assert vals[0] >= truth[0] - 1e-6, "the repair came back worse than the unstubbed fit"


def test_a_fit_that_cannot_be_repaired_is_clamped_to_the_star(tmp_path, monkeypatch):
    """A loose lower bound is survivable; one 9.6e8 log units wrong is not."""
    inst = tiny_instance(seed=29)
    env = ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    from opc.rl.env import greedy_support_rollout
    sels = [greedy_support_rollout(env)]
    star = RW.star_value(inst.obs)

    from opc.rl import mpl_torch as MT
    real = MT.profile_mpl

    def always_bad(models, obs, **kw):
        out = real(models, obs, **kw)
        out.value = torch.full_like(out.value, star - 1e9)
        return out

    monkeypatch.setattr(MT, "profile_mpl", always_bad)
    r = RW.MPLReward(device="cpu", iters=12, restarts=1, cache_dir=tmp_path)
    vals, stats = r(inst, sels)
    assert stats.n_nonconverged == 1 and stats.n_unrepaired == 1
    assert vals[0] == pytest.approx(star), "an unrepairable fit must be clamped to the floor"


# ------------------------------------------------------------- gene-tree arms (W14, A100)


def test_arm_dirs_reproduces_the_true_arms_names_exactly():
    """``arm=""`` must return the three directory names the study has always used.

    Everything landed lives under them, so a change here silently orphans ~50 GB of caches and
    every parquet beside them.
    """
    d = A.arm_dirs(Path("/r/b25"), "julia")
    assert d.trees == Path("/r/b25/genetrees_julia")
    assert d.cells == Path("/r/b25/cells_julia")
    assert d.clusters == Path("/r/b25/clusters_julia")


def test_arm_dirs_word_order_differs_between_trees_and_the_rest():
    """The asymmetry this module exists for: ``genetrees_est_julia_bp1000`` but
    ``cells_julia_est_bp1000``.  A single ``backend`` suffix cannot address both, which is what
    ``instances.build`` assumed before it went through here."""
    d = A.arm_dirs(Path("/r/x50"), "julia", "est_bp1000")
    assert d.trees == Path("/r/x50/genetrees_est_julia_bp1000")
    assert d.cells == Path("/r/x50/cells_julia_est_bp1000")
    assert d.clusters == Path("/r/x50/clusters_julia_est_bp1000")
    assert d.trees.name != f"genetrees_{d.cells.name.split('cells_', 1)[1]}", (
        "if these ever agreed, the module's reason for existing would be gone -- but so would "
        "this test's, and the string surgery it replaced would start working again by accident")


def test_an_unknown_arm_raises_rather_than_inventing_a_directory():
    """``opc.paths.tier_of``'s rule one level down: a made-up cache directory is *empty*, and an
    empty cache reads as 'nothing has been built yet' rather than as an error."""
    for bad in ("est", "est_bp", "true", "iq_l500", "est_bp1000x"):
        with pytest.raises(KeyError):
            A.arm_dirs(Path("/r"), "julia", bad)


def test_arm_slug_has_no_underscore_and_round_trips():
    """``make_gnn_report.ARM_RE`` splits a filename on ``_`` and takes the arch greedily up to
    ``_s<digits>``; an arm id with an underscore in it would re-partition the name into a
    different phase, arch and tag (A92/A94)."""
    for arm in ("", "est_bp500", "est_bp1000"):
        slug = A.arm_slug(arm)
        assert "_" not in slug
        assert A.arm_of_slug(slug) == arm


def test_cache_dir_is_unchanged_for_the_true_arm_and_keyed_for_any_other():
    """A52 in a fifth place -- and the true arm keeps the bare name, so nothing landed moves."""
    assert I.cache_dir(Path("/r"), "b25", 2) == Path("/r/b25/instances_tau2")
    assert I.cache_dir(Path("/r"), "b25", 2, "") == Path("/r/b25/instances_tau2")
    assert I.cache_dir(Path("/r"), "b25", 2, "est_bp1000") == \
        Path("/r/b25/instances_tau2_est_bp1000")


def test_summary_name_is_the_only_spelling_of_the_instance_parquet():
    """``rl_instances.py`` writes it and ``w10_gnn.pos_weight_of`` reads it; a miss there is a
    silent fallback that materialises every training instance (A89 -- a hang, not a failure)."""
    assert I.summary_name("x25", 2) == "rl_instances_x25_tau2.parquet"
    assert I.summary_name("x25", 2, "est_bp1000") == "rl_instances_x25_tau2_est_bp1000.parquet"
    assert I.summary_name("x25", 2, "", "_t") == "rl_instances_x25_tau2_t.parquet"


def test_the_mpl_cache_key_separates_gene_tree_arms():
    """**A100.** The likelihood is a function of the triple tensor, not of the topology alone.

    ``inst.key`` is the *cell*, and the same cell exists under both arms with the same taxon count
    and the same number of triples -- so every other part of the key matches while ``inst.obs``
    does not. Without the arm an estimated-arm run reads the simulated arm's likelihoods back and
    nothing raises. The true arm's key must stay byte-identical, or every landed cache entry in
    ``results/<phase>/mpl_cache_tau2/`` is orphaned.
    """
    class _Fake:
        def __init__(self, arm):
            self.key, self.n, self.arm = "n25_rep80_x", 25, arm
            self.index = np.zeros((2300, 3), dtype=np.int64)

    r = RW.MPLReward.__new__(RW.MPLReward)
    r.optimiser, r.iters, r.restarts, r.seed = "adam", 1500, 8, 0
    true_key = r._key(_Fake(""), "TOPO")
    est_key = r._key(_Fake("est_bp1000"), "TOPO")

    assert true_key != est_key, "two arms of one cell must not share an MPL cache entry"
    assert true_key == "TOPO|n25_rep80_x|25|2300|adam|1500|8|0", (
        "the true arm's key must be exactly what it was before arms existed, or every landed "
        "mpl_cache entry is orphaned")
    assert est_key.endswith("|est_bp1000")

    # ...and a pickle that predates the field reads as the true arm, so it keeps its entries.
    class _Old:
        key, n = "n25_rep80_x", 25
        index = np.zeros((2300, 3), dtype=np.int64)
    assert r._key(_Old(), "TOPO") == true_key
