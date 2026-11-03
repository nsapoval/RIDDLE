"""W10 -- the supervised node-classification arm.  ``opc.rl.nodecls`` and the two new convs.

Five obligations, and each one guards a defect this repository has actually paid for:

* **The registry is a refactor, not a rewrite.**  ``conv="segment"`` must be bit-identical to the
  path that existed before :data:`opc.rl.policy.CONVS`, and every landed checkpoint must still
  construct through ``PolicyConfig(**ck["policy_config"])`` -- which is how all six existing drivers
  build a policy.  A new dataclass field is exactly the kind of change that breaks that silently.
* **The decode is one variable changed.**  :func:`opc.rl.nodecls.decode` fed ``support`` must equal
  :func:`opc.rl.env.greedy_support_rollout` selection for selection.  If it does not, every cluster-F1
  delta W10 reports is measuring the decoder rather than the model, and nothing would say so.
* **Whatever the decode returns is a cluster system.**  Checked with ``env.verify()``, which
  re-derives (L) and (N3O) through ``heuristics.check`` and shares no state with the mask.
* **The tie handling is not cosmetic.**  Support is heavily tied and a model's float scores are not,
  so a ranking metric with an arbitrary tie convention would systematically flatter one of them.
  :func:`~opc.rl.nodecls.average_precision` is checked against brute-force enumeration of every
  permutation of the tied blocks -- an independent implementation, not a restatement.
* **The fixtures are not vacuous.**  Every test that means to exercise labels asserts the label
  vector has both classes, and the decode tests assert the trajectory actually moved.  ``CLAUDE.md``
  records the budget-DP test passing 1,188 budgets over *zero* multi-component systems.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opc import bitset  # noqa: E402
from opc.rl import bc, nodecls as NC, policy as P  # noqa: E402
from opc.rl.rollout import _scalars  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR, greedy_support_rollout  # noqa: E402
from opc.solve import candidates  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def ground_set(n_taxa=12, n_clusters=90, seed=0):
    """A random ground set.  Mirrors ``tests/test_rl_policy.ground_set`` deliberately."""
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

def tensors(gs, n_taxa=12, k=2):
    from opc.rl import instances as I
    g = OverlapCSR.of(gs)
    return (torch.as_tensor(I.node_features(gs, g, fake_tree_stats(gs))),
            torch.as_tensor(I.global_features(gs, g, k)),
            P.graph_tensors(g, gs.support, torch.device("cpu")), g)


# ---------------------------------------------------------------------------- the conv registry


@pytest.mark.parametrize("conv", sorted(P.CONVS))
def test_every_conv_runs_and_is_permutation_equivariant(conv):
    """Relabelling the nodes permutes the output and changes nothing else.

    The property that makes a message-passing layer a message-passing layer, and the one a
    hand-written ``index_add`` gets wrong by mixing up ``src`` and ``dst``.  A layer that aggregated
    over the wrong endpoint would still produce finite numbers of the right shape on this fixture.
    """
    gs = ground_set()
    xn, xg, g, graph = tensors(gs)
    pairs = bool(getattr(P.CONVS[conv], "needs_pairs", False))
    if pairs:
        g = P.graph_tensors(graph, gs.support, torch.device("cpu"), words=gs.words,
                            n_taxa=gs.n_taxa, m=gs.m)
    assert int(g.src.shape[0]) > 0, "fixture has no edges: the test would be vacuous"
    torch.manual_seed(0)
    pol = P.Policy(P.PolicyConfig(n_node_features=xn.shape[1], n_global_features=xg.shape[0],
                                  hidden=32, conv=conv))
    pol.eval()
    with torch.no_grad():
        base = pol.encode(xn, xg, g).h

    rng = np.random.default_rng(1)
    perm = torch.as_tensor(rng.permutation(int(graph.n)))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    if pairs:
        # A conv that reads the clusters themselves needs the relabelled *ground set*, not a
        # relabelled index: its hash table, containment relation and CSR order all follow the rows.
        from opc.solve import candidates
        p_np = perm.numpy()
        gsp = candidates.build(gs.words[p_np], np.asarray(gs.support)[p_np], gs.n_taxa, m=gs.m,
                               tau=gs.tau)
        gp = P.graph_tensors(OverlapCSR.of(gsp), gsp.support, torch.device("cpu"),
                             words=gsp.words, n_taxa=gs.n_taxa, m=gs.m)
    else:
        gp = P.GraphTensors(src=inv[g.src], dst=inv[g.dst], deg=g.deg[perm],
                            w_norm=g.w_norm, deg1=g.deg1[perm])
    with torch.no_grad():
        got = pol.encode(xn[perm], xg, gp).h
    assert torch.allclose(got, base[perm], atol=1e-5)


def test_an_unknown_conv_raises_rather_than_falling_back():
    with pytest.raises(ValueError, match="unknown conv"):
        P.Policy(P.PolicyConfig(n_node_features=10, n_global_features=8, conv="gat"))


def test_the_landed_checkpoints_still_load_and_default_to_segment():
    """The one change that could break every existing driver.

    Six readers build a policy as ``PolicyConfig(**ck["policy_config"])`` from a dict written before
    ``conv`` existed.  The dataclass default is the only thing making that dict still construct --
    and making it come back as the layer it was actually trained with.
    """
    found = sorted(REPO.glob("results/rl/rl_run_*/[bl][ca]*.pt"))
    if not found:
        pytest.skip("no landed checkpoints on this box")
    for path in found[:4]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = P.PolicyConfig(**ck["policy_config"])
        assert cfg.conv == "segment", f"{path.name} came back as {cfg.conv!r}"
        P.Policy(cfg).load_state_dict(ck["policy"])


def test_segment_is_unchanged_by_the_registry():
    """``CONVS["segment"]`` *is* ``SegmentConv``, and a default config selects it."""
    assert P.CONVS["segment"] is P.SegmentConv
    cfg = P.PolicyConfig(n_node_features=10, n_global_features=8)
    assert cfg.conv == "segment"
    assert all(isinstance(c, P.SegmentConv) for c in P.Policy(cfg).convs)


@pytest.mark.parametrize("conv", sorted(P.CONVS))
def test_layers_zero_touches_no_edge_whatever_the_conv(conv):
    """The ``mlp`` arm must be a strict features-only path under every conv name.

    ``layers = 0`` is the control that decides whether the overlap graph buys anything over the node
    features.  If it quietly used the graph, W10's G-N4 would answer a question nobody asked.
    Passing ``g=None`` is the strongest available statement: it segfaults nothing, it raises.
    """
    gs = ground_set()
    xn, xg, _g, _graph = tensors(gs)
    pol = P.Policy(P.PolicyConfig(n_node_features=xn.shape[1], n_global_features=xg.shape[0],
                                  hidden=32, layers=0, conv=conv))
    with torch.no_grad():
        assert pol.encode(xn, xg, None).h.shape == (int(gs.n_clusters), 32)


def test_gcn_normalisation_uses_the_augmented_degree():
    """``deg1`` is ``deg + 1``, not ``max(deg, 1)`` -- they differ wherever a vertex has neighbours."""
    gs = ground_set()
    _xn, _xg, g, graph = tensors(gs)
    raw = np.diff(np.asarray(graph.indptr)).astype(np.float64)
    assert np.allclose(g.deg1.numpy(), raw + 1.0)
    assert np.allclose(g.deg.numpy(), np.maximum(raw, 1.0))
    assert (raw > 0).any(), "fixture has no vertex with neighbours: the distinction is vacuous"
    assert not np.allclose(g.deg1.numpy(), g.deg.numpy())


# ---------------------------------------------------------------------------- labels


def test_labels_recover_truth_rows_and_are_not_vacuous():
    gs = ground_set()

    class Fake:
        n_clusters = int(gs.n_clusters)
        truth_rows = np.array([3, 17, 42, 61], dtype=np.int64)

    y = NC.labels(Fake())
    assert y.dtype == bool and y.shape == (Fake.n_clusters,)
    assert np.array_equal(np.flatnonzero(y), Fake.truth_rows)
    assert 0 < y.sum() < y.size


def test_the_size_features_are_gone_and_the_index_is_derived_not_written():
    """**The trap A97 had to avoid.** ``SCALE_GLOBAL_IDX`` used to be the literal ``(2, 3, 4)``.

    Removing the three size features narrowed ``GLOBAL_FEATURES`` from 8 entries to 5, and those
    same indices now address ``density``, ``mean_degree_frac`` and ``kernel_frac`` -- so a literal
    would have gone on zeroing three features, silently, and any ablation run against it would
    have been measuring something else. Deriving it by name is what makes it go empty instead.
    """
    from opc.rl import instances as I
    assert set(I.LEGACY_SIZE_FEATURES).isdisjoint(I.GLOBAL_FEATURES)
    assert NC.SCALE_GLOBAL_IDX == ()
    # the literal would still be in range, which is exactly why it could not be left as one
    assert len(I.GLOBAL_FEATURES) > 4
    assert [I.GLOBAL_FEATURES[i] for i in (2, 3, 4)] != list(I.LEGACY_SIZE_FEATURES)
    # `k` survives: it extrapolates too, but it is the budget the mask enforces, not a size proxy
    assert I.GLOBAL_FEATURES[0] == "k"


def test_zero_scale_globals_refuses_rather_than_silently_doing_nothing():
    """An identity return would let ``--zero-scale-globals`` report an ablation it did not do."""
    from opc.rl import instances as I
    with pytest.raises(SystemExit, match="A97"):
        NC.zero_scale_globals(np.ones(len(I.GLOBAL_FEATURES), dtype=np.float32))


def test_global_features_vector_matches_its_names():
    """The vector and the name tuple must not drift; every driver sizes its policy from the one
    and every report reads the other."""
    from opc.rl import instances as I
    gs = ground_set(seed=0)
    g = OverlapCSR.of(gs)
    x = I.global_features(gs, g, k=2)
    assert x.shape == (len(I.GLOBAL_FEATURES),)
    assert np.isfinite(x).all()


def test_a_stale_instance_is_refused_with_the_rebuild_instruction():
    """A pre-A97 pickle would otherwise build a policy of the wrong width and fail in a matmul."""
    from opc.rl import instances as I

    class Stale:
        x_global = np.zeros(len(I.GLOBAL_FEATURES) + 3, dtype=np.float32)

    with pytest.raises(SystemExit, match="rl_instances.py"):
        I.check_features(Stale(), "results/b50/instances_tau2/x.pkl")


def test_a_stale_node_vector_is_refused_too():
    """W13 took the node width 10 -> 31.  Until then ``check_features`` looked at ``x_global``
    only, which made it a guard on one path and therefore a guard on none: a pre-W13 pickle would
    have passed here and died inside ``Policy.embed`` several frames later."""
    from opc.rl import instances as I

    class Stale:
        x_global = np.zeros(len(I.GLOBAL_FEATURES), dtype=np.float32)
        x_nodes = np.zeros((7, len(I.BASE_NODE_FEATURES)), dtype=np.float32)

    with pytest.raises(SystemExit, match="node features"):
        I.check_features(Stale(), "results/b50/instances_tau2/x.pkl")


def test_node_blocks_are_addressed_by_name_and_cover_the_w13_columns():
    """A48's rule and A97's near-miss, one level down.

    W13's blocks are *appended* in order, so a literal slice ``NODE_FEATURES[10:]`` would work
    today -- which is exactly when it is cheapest not to write one.  The moment a block is
    reordered, renamed or inserted, a literal keeps returning in-range indices addressing the
    wrong columns and every ablation run against it silently measures something else.
    """
    from opc.rl import instances as I
    seen = []
    for name, cols in I.NODE_BLOCKS.items():
        idx = NC.block_idx(name)
        assert len(idx) == len(cols)
        assert tuple(I.NODE_FEATURES[i] for i in idx) == tuple(cols)
        seen.extend(idx)
    assert sorted(seen) == list(range(len(I.BASE_NODE_FEATURES), len(I.NODE_FEATURES)))
    assert len(set(seen)) == len(seen), "blocks overlap"
    with pytest.raises(SystemExit, match="unknown node block"):
        NC.block_idx("not_a_block")


def test_zeroing_a_block_touches_that_block_and_nothing_else():
    from opc.rl import instances as I
    rng = np.random.default_rng(0)
    x = rng.normal(size=(11, len(I.NODE_FEATURES))).astype(np.float32) + 5.0
    out = NC.zero_node_blocks(x, ["bconf"])
    cols = list(NC.block_idx("bconf"))
    other = [i for i in range(x.shape[1]) if i not in cols]
    assert (out[:, cols] == 0).all()
    assert np.array_equal(out[:, other], x[:, other])
    assert (x[:, cols] != 0).any(), "vacuous: the fixture block was already zero"
    assert NC.zero_node_blocks(x, []) is x        # the no-op path must not copy


def test_zeroing_a_block_works_on_a_torch_tensor():
    """The training loop holds device tensors; the probes hold numpy.  One function, both."""
    from opc.rl import instances as I
    x = torch.arange(3 * len(I.NODE_FEATURES), dtype=torch.float32).reshape(
        3, len(I.NODE_FEATURES)) + 1.0
    out = NC.zero_node_blocks(x, ["blen", "brank"])
    assert isinstance(out, torch.Tensor)
    cols = sorted(set(NC.block_idx("blen")) | set(NC.block_idx("brank")))
    assert float(out[:, cols].abs().sum()) == 0.0
    assert float(x[:, cols].abs().sum()) > 0.0, "the input was mutated"


# ---------------------------------------------------------------------------- the decode


def test_decode_on_support_reproduces_greedy_exactly():
    """**The parity the whole comparison rests on.**

    If ``decode`` and ``greedy_support_rollout`` disagree when handed the same scores, then a
    reported cluster-F1 delta is a property of the decoder and not of the model -- and nothing
    downstream would reveal it.
    """
    for seed in (0, 1, 2):
        gs = ground_set(seed=seed)
        env = ClusterEnv(gs, budget=2)
        want = greedy_support_rollout(env)
        got = NC.decode(env, np.asarray(gs.support, dtype=np.float64))
        assert want.size > 0, "greedy selected nothing: the parity would be vacuous"
        assert np.array_equal(np.sort(got), np.sort(want))


def test_every_decoded_selection_is_a_feasible_system_within_budget():
    """Checked through ``env.verify()`` -- ``heuristics.check``, which shares no state with the mask."""
    rng = np.random.default_rng(0)
    for seed in (0, 1, 2):
        gs = ground_set(seed=seed)
        for budget in (0, 1, 3):
            env = ClusterEnv(gs, budget=budget)
            sel = NC.decode(env, rng.random(int(gs.n_clusters)))
            assert env.verify()
            assert env.state.n_hybrids <= budget
            assert np.array_equal(np.sort(sel), np.sort(env.selection))


def test_decode_is_deterministic_and_rejects_a_wrong_shape():
    gs = ground_set()
    p = np.random.default_rng(3).random(int(gs.n_clusters))
    a = NC.decode(ClusterEnv(gs, budget=2), p)
    b = NC.decode(ClusterEnv(gs, budget=2), p)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="scores must be"):
        NC.decode(ClusterEnv(gs, budget=2), p[:-1])


def test_raw_set_diagnostics_can_report_an_infeasible_prediction():
    """The diagnostic has to be able to *fail*, or it measures nothing.

    A score vector of all ones thresholds to the whole ground set, which on this fixture is not a
    cluster system.  If this came back feasible the diagnostic would be reporting a constant.
    """
    gs = ground_set()

    class Fake:
        n = 12
        n_clusters = int(gs.n_clusters)
        truth_rows = np.array([1, 2, 3], dtype=np.int64)

    Fake.gs = gs
    d = NC.raw_set_diagnostics(Fake(), np.ones(int(gs.n_clusters)))
    assert d["raw_size"] == int(gs.n_clusters)
    assert d["raw_feasible"] is False
    empty = NC.raw_set_diagnostics(Fake(), np.zeros(int(gs.n_clusters)))
    assert empty["raw_size"] == 0 and empty["raw_feasible"] is True and empty["raw_k_needed"] == 0
    # ``top`` is threshold-free and takes exactly |truth_rows|, whatever the scores are -- which
    # is the whole reason it is the reading to trust when the head is trained under a pos_weight.
    for scores in (np.ones(int(gs.n_clusters)), np.zeros(int(gs.n_clusters)),
                   np.random.default_rng(0).random(int(gs.n_clusters))):
        got = NC.raw_set_diagnostics(Fake(), scores)
        assert got["top_size"] == Fake.truth_rows.size
        assert got["top_tp"] <= got["top_size"]


# ---------------------------------------------------------------------------- ranking metrics


def _brute_force_ap(scores, y):
    """Average precision averaged over **every** ordering consistent with the scores.

    The independent implementation :func:`~opc.rl.nodecls.average_precision`'s closed form is checked
    against.  Exponential, so it is only ever run on a handful of items -- which is the point: the
    closed form is what makes the same statement affordable at |C| = 14,124.
    """
    scores, y = np.asarray(scores, float), np.asarray(y).astype(bool)
    order, starts = NC._tied_blocks(scores)
    blocks = [order[lo:(int(starts[b + 1]) if b + 1 < starts.size else y.size)]
              for b, lo in enumerate(starts)]
    total, count = 0.0, 0
    for combo in itertools.product(*[list(itertools.permutations(b)) for b in blocks]):
        idx = np.concatenate([np.asarray(c, dtype=np.int64) for c in combo])
        yo = y[idx]
        hits, ap = 0, 0.0
        for i, lab in enumerate(yo, start=1):
            if lab:
                hits += 1
                ap += hits / i
        total += ap / max(1, int(y.sum()))
        count += 1
    return total / count


@pytest.mark.parametrize("scores,y", [
    ([5.0, 4.0, 3.0, 2.0, 1.0], [1, 0, 1, 0, 0]),          # no ties at all
    ([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]),                   # one big tie
    ([3.0, 2.0, 2.0, 2.0, 1.0, 1.0], [1, 1, 0, 1, 0, 1]),   # several tied blocks
    ([2.0, 2.0, 1.0], [0, 0, 1]),                           # a single positive, at the back
])
def test_average_precision_matches_brute_force_over_tied_orderings(scores, y):
    assert NC.average_precision(scores, y) == pytest.approx(_brute_force_ap(scores, y), abs=1e-12)


def test_average_precision_and_auc_are_undefined_on_one_class():
    assert np.isnan(NC.average_precision([1.0, 2.0, 3.0], [0, 0, 0]))
    assert np.isnan(NC.average_precision([1.0, 2.0, 3.0], [1, 1, 1]))
    assert np.isnan(bc.auc(np.array([1.0, 2.0]), np.array([True, True])))


def test_precision_at_k_splits_a_straddling_tied_block_in_proportion():
    # top-2 of [3, 2, 2, 2] with labels [1, 1, 0, 0]: the whole first item plus one third of the
    # tied block of three, which holds one positive -> (1 + 1/3) / 2.
    assert NC.precision_at_k([3.0, 2.0, 2.0, 2.0], [1, 1, 0, 0], 2) == pytest.approx((1 + 1 / 3) / 2)
    assert NC.precision_at_k([4.0, 3.0, 2.0, 1.0], [1, 1, 0, 0], 2) == pytest.approx(1.0)
    assert np.isnan(NC.precision_at_k([1.0, 2.0], [1, 0], 0))
    assert np.isnan(NC.precision_at_k([1.0, 2.0], [1, 0], 5))


def test_error_reduction_is_scripts_w4_bc_s_own_formula():
    """Identical to ``w4_bc.py:96``, so the two arms' numbers are the same quantity.

    And a caution the numbers themselves make: ``rl_run_bc_oracle/verdict.json`` records
    ``auc_support`` 0.991808 and ``auc_gain`` 0.001840 -- whose **ratio** is 0.225 -- beside
    ``error_reduction`` **0.378**.  They disagree because both are medians over validation cells and
    a ratio of medians is not a median of ratios.  This function is the *per-cell* statistic, which
    is the one that composes; anything reporting a pooled figure has to median the ratios, not
    divide the medians.
    """
    assert NC.error_reduction(0.991808 + 0.001840, 0.991808) == pytest.approx(0.22465, abs=1e-4)
    assert NC.error_reduction(1.0, 0.5) == pytest.approx(1.0)
    assert NC.error_reduction(0.5, 0.5) == pytest.approx(0.0)
    assert np.isnan(NC.error_reduction(1.0, 1.0))


def test_pos_weight_is_pooled_and_refuses_a_vacuous_pool():
    class Fake:
        def __init__(self, n, pos):
            self.n_clusters, self.truth_rows = n, np.arange(pos, dtype=np.int64)

    assert NC.pos_weight([Fake(100, 2), Fake(100, 3)]) == pytest.approx(195 / 5)
    assert NC.positive_rate(Fake(100, 2)) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="vacuous"):
        NC.pos_weight([Fake(100, 0)])


# ---------------------------------------------------------------------------- W11: the two targets


def _fixture(seed=0, budget=2):
    gs = ground_set(seed=seed)
    xn, xg, g, _graph = tensors(gs, k=budget)
    torch.manual_seed(0)
    pol = P.Policy(P.PolicyConfig(n_node_features=xn.shape[1], n_global_features=xg.shape[0],
                                  hidden=32))
    env = ClusterEnv(gs, budget=budget)
    return gs, pol, pol.encode(xn, xg, g), env


def _feasible_sequence(env, n_take):
    """``n_take`` rows that are legal **in order**, found by walking the env.

    Taking the first ``n`` rows of ``legal_mask()`` at reset does *not* work and is how this
    fixture was first written: accepting one cluster makes its overlap partners illegal, so the
    sequence is rejected from step 2 and every assertion below becomes a statement about
    ``n_illegal`` instead of about the loss.  Feasibility is hereditary going *down*, not across.
    """
    env.reset()
    out = []
    while len(out) < n_take:
        legal = np.flatnonzero(env.legal_mask())
        if not legal.size:
            break
        out.append(int(legal[0]))
        env.step(int(legal[0]))
    env.reset()
    return np.asarray(out, dtype=np.int64)


class _Inst:
    """The three attributes ``bc``'s losses read off an instance."""
    def __init__(self, gs, n_taxa, k, truth_rows):
        self.gs, self.n, self.k = gs, n_taxa, k
        self.n_clusters = int(gs.n_clusters)
        self.truth_rows = np.asarray(truth_rows, dtype=np.int64)


def test_step_logits_reproduces_the_samplers_own_context():
    """**The guard `swap_clone_loss`'s docstring asks for, one MDP over.**

    Two clone targets now share this MDP, so the context arithmetic lives in one function.  This
    asserts that function still agrees with :func:`opc.rl.rollout.rollout_group`'s own inline
    construction -- if it drifts, a target and a rollout disagree about what state the policy is
    standing at, and nothing else would say so.
    """
    gs, pol, enc, env = _fixture()
    inst = _Inst(gs, 12, 2, [])
    support = np.asarray(gs.support, dtype=np.float64)
    env.reset()
    for _ in range(3):
        mask, deg = env.legal_mask(with_features=True)
        assert mask.any(), "fixture terminated early: the comparison would be vacuous"
        mine = bc._step_logits(pol, enc, inst, env, mask, deg, device=torch.device("cpu"))
        theirs = pol.logits(
            enc,
            accepted_sum=(enc.h[env.selection].sum(0, keepdim=True) if env.selection.size
                          else torch.zeros(1, pol.cfg.hidden)),
            accepted_n=torch.tensor([float(env.selection.size)]),
            last_action=torch.tensor([env.last_action], dtype=torch.long),
            scalars=_scalars(env, inst, int(mask.sum()), torch.device("cpu")),
            dynamic=torch.as_tensor(
                P.dynamic_features(support, mask, deg, int(env.selection.size))).unsqueeze(0),
            mask=torch.as_tensor(mask).unsqueeze(0),
        )[0]
        fin = torch.isfinite(mine)
        assert torch.allclose(mine[fin], theirs[fin], atol=1e-6)
        assert torch.equal(fin, torch.isfinite(theirs))
        env.step(int(np.flatnonzero(mask)[0]))


def test_the_set_target_never_exceeds_the_ordered_one_and_matches_it_when_the_set_is_a_singleton():
    """``-log sum_{i in T} p_i <= -log p_a`` for any ``a in T``, with equality iff ``|T| == 1``.

    Which is the whole claim: the ordered target charges the policy for preferring a *different*
    true cluster, and the set target does not.  The singleton case pins that the two are the same
    statement when the objective really does fix an order.
    """
    gs, pol, enc, env = _fixture()
    dev = torch.device("cpu")
    rows = _feasible_sequence(env, 4)
    assert rows.size == 4, "fixture could not build a 4-step feasible sequence"
    inst_many = _Inst(gs, 12, 2, rows)
    inst_one = _Inst(gs, 12, 2, rows[:1])
    lo, n_o, bad_o = bc.clone_loss(pol, enc, inst_many, env, rows, device=dev)
    ls, n_s, bad_s = bc.set_clone_loss(pol, enc, inst_many, env, rows, device=dev)
    assert bad_o == bad_s == 0 and n_o == n_s == len(rows)
    assert float(ls) < float(lo), "several true clusters legal, so the set target must be looser"
    l1, _n, _b = bc.clone_loss(pol, enc, inst_one, env, rows[:1], device=dev)
    s1, _n, _b = bc.set_clone_loss(pol, enc, inst_one, env, rows[:1], device=dev)
    assert float(s1) == pytest.approx(float(l1), abs=1e-6)


def test_both_targets_carry_a_gradient_and_visit_the_same_states():
    gs, pol, enc, env = _fixture()
    dev = torch.device("cpu")
    rows = _feasible_sequence(env, 5)
    assert rows.size == 5, "fixture could not build a 5-step feasible sequence"
    inst = _Inst(gs, 12, 2, rows)
    seen = []
    for fn in (bc.clone_loss, bc.set_clone_loss):
        loss, n, bad = fn(pol, enc, inst, env, rows, device=dev)
        assert bad == 0 and n == len(rows) and loss.requires_grad
        loss.backward(retain_graph=True)
        seen.append(np.sort(env.selection).tolist())
    assert seen[0] == seen[1], "teacher forcing must put both targets on the same trajectory"


def test_dynamic_extra_defaults_to_the_three_column_output_byte_for_byte():
    """``extra=None`` must reproduce the landed behaviour exactly.

    Four drivers call ``dynamic_features`` on the construction MDP and none of them passes ``extra``;
    if the default widened or reordered, every one of them would keep running and quietly compute a
    different policy.  So the guard is equality, not shape.
    """
    rng = np.random.default_rng(0)
    n = 60
    support = rng.integers(1, 40, n).astype(np.float64)
    mask = rng.random(n) > 0.3
    deg = rng.integers(0, 5, n).astype(np.int64)
    base = P.dynamic_features(support, mask, deg, 7)
    assert base.shape == (n, P.N_DYNAMIC)
    assert np.array_equal(base, P.dynamic_features(support, mask, deg, 7, extra=None))

    prior = rng.normal(size=(n, 1)).astype(np.float32)
    wide = P.dynamic_features(support, mask, deg, 7, extra=prior)
    assert wide.shape == (n, P.N_DYNAMIC + 1)
    assert np.array_equal(wide[:, :P.N_DYNAMIC], base)     # the first three are untouched
    assert np.array_equal(wide[:, P.N_DYNAMIC:], prior)


# ------------------------------------------------------- the decode-derived loss terms (W10 v2)
#
# The claim these have to support is narrow: the term is computed from what the decode *did*, the
# selection is a constant inside it, and it is exactly zero on a cell the decode already got right.
# Anything looser and a `+lambda * term` arm would be reporting a constant offset as a finding.


def _labelled(seed=0, n_true=8):
    """A ground set plus a label vector that the decode is *guaranteed* to get partly wrong.

    The truth is the ``n_true`` lowest-support rows, so greedy-by-support -- which is what the
    decode reduces to when handed support -- must pass some of them over.  A fixture where the
    decode is already right would make every assertion below vacuous, so it is asserted.
    """
    gs = ground_set(seed=seed)
    y = np.zeros(gs.n_clusters, dtype=bool)
    y[np.argsort(np.asarray(gs.support), kind="stable")[:n_true]] = True
    return gs, y


def test_decode_trace_records_the_same_episode_decode_runs():
    """One rollout implementation, three scores.  If the trace came from a second loop, every
    number built on it would describe that loop rather than the decode."""
    gs, y = _labelled()
    scores = np.asarray(gs.support, dtype=np.float64)
    sel_a = NC.decode(ClusterEnv(gs, budget=2), scores)
    sel_b, trace = NC.decode_trace(ClusterEnv(gs, budget=2), scores, y)
    assert np.array_equal(sel_a, sel_b)
    assert len(trace) == sel_a.size, "one trace entry per accepted cluster"
    assert {a for a, _ in trace} == set(sel_a.tolist())


def test_the_trace_finds_avoidable_errors_and_the_fixture_has_some():
    gs, y = _labelled()
    _, trace = NC.decode_trace(ClusterEnv(gs, budget=2), np.asarray(gs.support, float), y)
    avoidable = [(a, q) for a, q in trace if not y[a] and q.size]
    assert avoidable, "the decode made no avoidable error: every assertion below would be vacuous"
    for a, q in avoidable:
        assert y[q].all(), "legal_true must contain only true clusters"
        assert a not in set(q.tolist())


def test_a_cap_truncates_and_none_reproduces_the_uncapped_episode():
    gs, y = _labelled()
    scores = np.asarray(gs.support, dtype=np.float64)
    full, _ = NC.decode_trace(ClusterEnv(gs, budget=2), scores, y)
    assert full.size > 3, "fixture too short to truncate"
    cut, trace = NC.decode_trace(ClusterEnv(gs, budget=2), scores, y, cap=3)
    assert cut.size == 3 and len(trace) == 3
    assert set(cut.tolist()) <= set(full.tolist()), "the cap must take a prefix of the episode"
    again, _ = NC.decode_trace(ClusterEnv(gs, budget=2), scores, y, cap=None)
    assert np.array_equal(again, full)


@pytest.mark.parametrize("kind", ("step", "set"))
def test_the_decode_term_is_exactly_zero_when_the_decode_is_right(kind):
    """A term that added a constant on a correct cell would shift every arm's loss and none of
    its gradients, which is the sort of thing a loss curve cannot show."""
    gs = ground_set(seed=1)
    env = ClusterEnv(gs, budget=2)
    scores = np.asarray(gs.support, dtype=np.float64)
    sel, trace = NC.decode_trace(env, scores, np.zeros(gs.n_clusters, dtype=bool))
    y = np.zeros(gs.n_clusters, dtype=bool)
    y[sel] = True                                   # the decode's own answer is now the truth
    _, trace = NC.decode_trace(ClusterEnv(gs, budget=2), scores, y)
    logits = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    term = NC.decode_loss_terms(logits, trace, y, kind, selection=sel)
    assert float(term) == 0.0
    term.backward()                                 # must still be differentiable
    assert torch.count_nonzero(logits.grad) == 0


def test_the_step_term_pushes_a_passed_over_true_cluster_above_the_wrong_pick():
    """The whole point of ``step``: it is a *comparison*, so its gradient must raise the true
    cluster's logit and lower the wrong pick's.  A per-node BCE cannot state that."""
    gs, y = _labelled()
    env = ClusterEnv(gs, budget=2)
    scores = np.asarray(gs.support, dtype=np.float64)
    _, trace = NC.decode_trace(env, scores, y)
    bad = [(a, q) for a, q in trace if not y[a] and q.size]
    assert bad, "vacuous fixture"
    logits = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    NC.decode_loss_terms(logits, trace, y, "step").backward()
    g = logits.grad
    a, q = bad[0]
    assert g[a] > 0, "the wrong pick's logit must be pushed down (loss gradient positive)"
    assert (g[torch.as_tensor(q.astype(np.int64))] < 0).any(), "no true cluster was pushed up"
    assert torch.count_nonzero(g) < g.numel(), "a step term must not touch every node"


def test_the_set_term_only_touches_the_decodes_mistakes():
    gs, y = _labelled()
    sel, trace = NC.decode_trace(ClusterEnv(gs, budget=2), np.asarray(gs.support, float), y)
    logits = torch.zeros(gs.n_clusters, requires_grad=True)
    NC.decode_loss_terms(logits, trace, y, "set", selection=sel).backward()
    touched = set(torch.nonzero(logits.grad).flatten().tolist())
    mistakes = set(np.flatnonzero(np.isin(np.arange(gs.n_clusters), sel) ^ y).tolist())
    assert touched and touched == mistakes


def test_the_band_term_needs_no_env_and_ranks_inside_the_band():
    """``band`` is the control that prices the decode machinery, so what matters is that it is a
    ranking term over the confusable band, that it reaches nothing outside it, and that it never
    pushes a true logit down."""
    gs = ground_set(seed=0)
    order = np.argsort(-np.asarray(gs.support), kind="stable")
    y = np.zeros(gs.n_clusters, dtype=bool)
    y[order[[1, 4, 7, 11, 16, 22]]] = True      # inside the band, interleaved with negatives
    logits = torch.tensor(np.asarray(gs.support, np.float64), dtype=torch.float32,
                          requires_grad=True)
    band = torch.argsort(logits.detach(), descending=True, stable=True)[:3 * int(y.sum())]
    assert y[band.numpy()].any() and (~y[band.numpy()]).any(), "band has one class: vacuous"
    term = NC.band_loss_terms(logits, y, rng=np.random.default_rng(0))
    assert float(term) > 0
    term.backward()
    touched = torch.nonzero(logits.grad).flatten()
    assert set(touched.tolist()) <= set(band.tolist()), "the term reached outside the band"
    assert (logits.grad[torch.as_tensor(np.flatnonzero(y))] <= 0).all(), "a true logit pushed down"


def test_the_band_term_is_zero_when_the_band_holds_one_class():
    """No positive in the confusable band means there is no pair to rank, and the honest value is
    zero rather than a constant -- the same property the decode terms have on a correct cell."""
    gs, y = _labelled()                      # truth is the LOWEST-support rows, so none is in band
    logits = torch.tensor(np.asarray(gs.support, np.float64), dtype=torch.float32,
                          requires_grad=True)
    band = torch.argsort(logits.detach(), descending=True, stable=True)[:3 * int(y.sum())]
    assert not y[band.numpy()].any(), "fixture put a positive in the band: vacuous"
    term = NC.band_loss_terms(logits, y, rng=np.random.default_rng(0))
    assert float(term) == 0.0
    term.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_pos_weight_from_counts_agrees_with_the_instance_form():
    """``w10_gnn.pos_weight_of`` reads the two counts from the instance parquet instead of
    materialising 1,400 instances for two scalars.  The two forms must agree exactly, because the
    landed runs used the slow one and a drift would change the loss scale with nothing saying so.
    """
    class FakeInst:
        def __init__(self, n_clusters, n_pos):
            self.n_clusters = n_clusters
            self.truth_rows = np.arange(n_pos)
    insts = [FakeInst(2000, 46), FakeInst(1500, 35), FakeInst(3000, 70)]
    slow = NC.pos_weight(insts)
    fast = NC.pos_weight_from_counts(sum(len(i.truth_rows) for i in insts),
                                     sum(i.n_clusters for i in insts))
    assert slow == fast
    assert abs(slow - 41.3) < 1.0, "fixture drifted off the ~2.3% positive rate this is about"


def test_pos_weight_refuses_a_pool_with_no_positives():
    """A vacuous fixture must raise rather than divide by zero and train on a silent NaN."""
    with pytest.raises(ValueError, match="vacuous"):
        NC.pos_weight_from_counts(0, 1000)
