"""W3 -- the encoder and the attention decoder.  ``RL_PLAN.md`` §3.

Three obligations, and only the first is about the network being a network:

* **The mask is absolute.**  ``ClusterEnv.step`` raises on an illegal action, so a policy that puts
  any probability on one is a run that dies at a random step hours in.  Checked at *every* state of
  a trajectory, not only the empty one -- (L) at a committed neighbour and the (N3O) triangle only
  bite once something has been accepted, which is the same reason ``test_rl_env`` walks a
  trajectory.
* **The encoder is size-agnostic.**  One set of weights must run at two ``|C|`` and two ``n``.
  That is what makes **G-R6** a transfer test rather than a rescaling one, and it is easy to
  break by accident: a layer whose width came from the instance would pass every other test here.
* **``layers = 0`` touches no edge.**  It is the ablation that measures whether the overlap graph
  buys anything over the node features, so it has to be a *strict* features-only path; if it
  quietly used the graph the ablation would answer a question nobody asked.

Every test asserts its fixture produced the case it means to test.  A ground set whose overlap
graph is empty, or a trajectory that terminates immediately, would pass most of this while
exercising none of it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opc import bitset  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR  # noqa: E402
from opc.solve import candidates  # noqa: E402


def ground_set(n_taxa=12, n_clusters=90, seed=0):
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

def features(gs):
    from opc.rl import instances as I
    g = OverlapCSR.of(gs)
    return (torch.as_tensor(I.node_features(gs, g, fake_tree_stats(gs))),
            torch.as_tensor(I.global_features(gs, g, 2)),
            P.graph_tensors(g, gs.support, torch.device("cpu")), g)


def make_policy(xn, xg, **kw):
    cfg = P.PolicyConfig(n_node_features=xn.shape[1], n_global_features=xg.shape[0],
                         hidden=kw.pop("hidden", 32), **kw)
    torch.manual_seed(0)
    return P.Policy(cfg)


def decode(policy, enc, env, support, mask, deg):
    n_legal = int(mask.sum())
    return policy.logits(
        enc,
        accepted_sum=(enc.h[env.selection].sum(0, keepdim=True) if env.selection.size
                      else torch.zeros(1, policy.cfg.hidden)),
        accepted_n=torch.tensor([float(env.selection.size)]),
        last_action=torch.tensor([env.last_action], dtype=torch.long),
        scalars=torch.tensor([[1.0, 0.1, env.t / 12, n_legal / env.n]]),
        dynamic=torch.as_tensor(
            P.dynamic_features(support, mask, deg, int(env.selection.size))).unsqueeze(0),
        mask=torch.as_tensor(mask).unsqueeze(0),
    )[0]


def test_no_probability_on_an_illegal_action_at_any_state():
    gs = ground_set(seed=3)
    xn, xg, g, _csr = features(gs)
    policy = make_policy(xn, xg)
    enc = policy.encode(xn, xg, g)
    env = ClusterEnv(gs, budget=2)
    support = np.asarray(gs.support, dtype=float)
    mask, deg = env.legal_mask(with_features=True)
    steps, saw_illegal_candidates = 0, False
    while mask.any():
        saw_illegal_candidates |= bool((~mask & ~env.state.in_s).any())
        lg = decode(policy, enc, env, support, mask, deg)
        p = torch.softmax(lg, dim=-1).detach()
        assert torch.isfinite(p).all()
        assert float(p[~torch.as_tensor(mask)].sum()) == 0.0
        assert abs(float(p.sum()) - 1.0) < 1e-5
        a = int(torch.multinomial(p, 1))
        obs, _r, done, _i = env.step(a)          # raises if the mask and the policy disagree
        mask, deg = obs["legal"], obs["deg_sel"]
        steps += 1
        if done:
            break
    assert steps > 3, "the trajectory terminated too early to test anything"
    assert saw_illegal_candidates, "nothing was ever illegal -- the mask was never exercised"


def test_the_encoder_is_size_agnostic():
    """One set of weights, two ``|C|`` and two ``n``.  G-R6 is meaningless without this."""
    a = ground_set(n_taxa=12, n_clusters=90, seed=4)
    b = ground_set(n_taxa=17, n_clusters=160, seed=5)
    xa, ga_, gra, _ = features(a)
    xb, gb_, grb, _ = features(b)
    assert a.n_clusters != b.n_clusters and a.n_taxa != b.n_taxa
    policy = make_policy(xa, ga_)
    ha = policy.encode(xa, ga_, gra).h
    hb = policy.encode(xb, gb_, grb).h
    assert ha.shape == (a.n_clusters, policy.cfg.hidden)
    assert hb.shape == (b.n_clusters, policy.cfg.hidden)
    assert torch.isfinite(ha).all() and torch.isfinite(hb).all()


def test_layers_zero_is_a_strict_features_only_path():
    gs = ground_set(seed=6)
    xn, xg, g, _ = features(gs)
    policy = make_policy(xn, xg, layers=0)
    h_none = policy.encode(xn, xg, None).h        # no graph at all
    # Feeding a *different* graph must change nothing: if it did, the ablation is not an ablation.
    other = ground_set(n_taxa=12, n_clusters=90, seed=99)
    _xn2, _xg2, g2, _ = features(other)
    assert torch.equal(h_none, policy.encode(xn, xg, g).h)
    assert torch.equal(h_none, policy.encode(xn, xg, g2).h)
    # ...and with layers > 0 it must matter, or the check above proves nothing.
    deep = make_policy(xn, xg, layers=2)
    assert not torch.allclose(deep.encode(xn, xg, g).h, policy.encode(xn, xg, None).h)
    with pytest.raises(ValueError, match="needs the graph"):
        deep.encode(xn, xg, None)


def test_graph_tensors_normalise_per_destination():
    gs = ground_set(seed=7)
    g = OverlapCSR.of(gs)
    gt = P.graph_tensors(g, gs.support, torch.device("cpu"))
    deg = np.diff(np.asarray(g.indptr))
    assert int((deg > 0).sum()) > 0.5 * g.n, "fixture is too sparse to test normalisation"
    tot = np.bincount(gt.dst.numpy(), weights=gt.w_norm.numpy(), minlength=g.n)
    assert np.allclose(tot[deg > 0], 1.0, atol=1e-5)
    assert np.array_equal(gt.deg.numpy(), np.maximum(deg, 1).astype(np.float32))
    # src/dst really are the CSR's own edges, both directions.
    assert gt.src.numel() == int(deg.sum())


def test_dynamic_features_say_what_they_claim():
    gs = ground_set(seed=8)
    env = ClusterEnv(gs, budget=2)
    support = np.asarray(gs.support, dtype=float)
    rng = np.random.default_rng(1)
    for _ in range(6):
        mask, deg = env.legal_mask(with_features=True)
        if not mask.any():
            break
        n_acc = int(env.selection.size)
        f = P.dynamic_features(support, mask, deg, n_acc)
        assert f.shape == (env.n, P.N_DYNAMIC)
        assert np.allclose(f[:, 0], deg / max(1.0, float(n_acc)))
        legal = np.flatnonzero(mask)
        assert np.isclose(f[legal, 1].max(), 1.0)
        assert f[~mask.astype(bool), 1].sum() == 0.0, "an illegal row carries a support feature"
        assert f[:, 2].sum() == 1.0
        assert support[int(np.argmax(f[:, 2]))] == support[legal].max()
        env.step(int(rng.choice(legal)))


def test_logits_are_bounded_by_the_tanh_clip():
    """§4's entropy schedule is meaningless against an unbounded dot product."""
    gs = ground_set(seed=9)
    xn, xg, g, _ = features(gs)
    policy = make_policy(xn, xg)
    with torch.no_grad():                     # blow the keys up: the clip must still hold
        policy.key.weight.mul_(50.0)
    enc = policy.encode(xn, xg, g)
    env = ClusterEnv(gs, budget=2)
    mask, deg = env.legal_mask(with_features=True)
    lg = decode(policy, enc, env, np.asarray(gs.support, float), mask, deg).detach()
    finite = lg[torch.isfinite(lg)]
    assert float(finite.abs().max()) <= policy.cfg.tanh_clip + 1e-5
    assert float(finite.abs().max()) > 1.0, "the fixture did not push the logits anywhere"


def test_encoder_checkpointing_changes_memory_not_meaning():
    """``checkpoint_encoder`` recomputes each layer in the backward instead of keeping it.

    It is what makes the encoder fit -- measured at n = 50, tau = 2: 771 MiB retained *per layer*
    against 171 MiB for the whole forward with checkpointing on -- so it is on by default, and a
    default that silently changed what the policy computes would be the worst kind.

    The tolerance is loose on purpose and the reason is not checkpointing: ``index_add`` on CUDA
    is an atomic scatter and is non-deterministic *anyway*.  What this pins is that the two paths
    agree to float32 accumulation order, not to the bit.
    """
    gs = ground_set(seed=31)
    xn, xg, g, _ = features(gs)
    kw = {"n_node_features": xn.shape[1], "n_global_features": xg.shape[0], "hidden": 32,
          "layers": 2}
    torch.manual_seed(0)
    plain = P.Policy(P.PolicyConfig(**kw, checkpoint_encoder=False))
    torch.manual_seed(0)
    ckpt = P.Policy(P.PolicyConfig(**kw, checkpoint_encoder=True))

    ha, hb = plain.encode(xn, xg, g).h, ckpt.encode(xn, xg, g).h
    assert torch.allclose(ha, hb, atol=1e-5), "checkpointing changed the embeddings"
    ga = torch.autograd.grad(ha.square().sum(), list(plain.parameters()), allow_unused=True)
    gb = torch.autograd.grad(hb.square().sum(), list(ckpt.parameters()), allow_unused=True)
    pairs = [(x, y) for x, y in zip(ga, gb, strict=True) if x is not None and y is not None]
    assert pairs, "no parameter received a gradient -- the fixture proves nothing"
    for x, y in pairs:
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-3), "checkpointing changed the gradient"
    # ...and under no_grad it must not try to checkpoint at all.
    with torch.no_grad():
        assert torch.allclose(ckpt.encode(xn, xg, g).h, ha, atol=1e-5)


def test_edge_chunking_is_exact():
    """The gather is chunked to bound the transient; the answer must not depend on the chunk."""
    gs = ground_set(seed=33)
    xn, xg, g, csr = features(gs)
    n_edges = int(g.src.numel())
    assert n_edges > 400, "fixture has too few edges to chunk"
    policy = make_policy(xn, xg, layers=2)
    old = P.EDGE_CHUNK
    try:
        P.EDGE_CHUNK = 1 << 20                       # one chunk
        whole = policy.encode(xn, xg, g).h
        P.EDGE_CHUNK = 97                            # many, and 97 does not divide n_edges
        assert n_edges % 97 != 0, "pick a chunk size that leaves a partial tail"
        split = policy.encode(xn, xg, g).h
    finally:
        P.EDGE_CHUNK = old
    assert torch.allclose(whole, split, atol=1e-5)
