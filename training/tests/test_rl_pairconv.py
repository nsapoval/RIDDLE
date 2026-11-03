"""PairConv (S1-S3, A108 addendum): the pair features, the containment relation, the attention.

Every device-side quantity is checked against Python ints and frozensets, because a hashed lookup
that silently misses, or a popcount that leaks a sign bit, would hand the gate wrong inputs and
nothing downstream would raise.  The fixture is intervals of a line plus random sets over 70 taxa
-- two words per cluster -- so crossing pairs exist, their intersections are often candidate rows
and their twins never are, and nested pairs are common; each test asserts the case it needs exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opc import bitset  # noqa: E402
from opc.rl import partner as PT  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl.env import OverlapCSR  # noqa: E402
from opc.solve import candidates  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N_TAXA, M = 70, 50
CPU = torch.device("cpu")


def fixture(seed=0, n_random=40, max_len=9):
    rng = np.random.default_rng(seed)
    sets = []
    for i in range(N_TAXA):
        for ln in range(2, max_len):
            if i + ln <= N_TAXA and rng.random() < 0.6:
                sets.append(frozenset(range(i, i + ln)))
    while len(sets) < 400 + n_random:
        sets.append(frozenset(rng.choice(N_TAXA, size=int(rng.integers(2, 12)), replace=False).tolist()))
    sets = list(dict.fromkeys(sets))
    words = bitset.pack_rows([bitset.from_indices(sorted(c), N_TAXA) for c in sets], N_TAXA)
    support = rng.integers(1, M, size=len(sets)).astype(np.int32)
    gs = candidates.build(words, support, N_TAXA, m=M, tau=0)
    return gs, OverlapCSR.of(gs), [frozenset(bitset.to_indices(w, N_TAXA)) for w in gs.words]


def pair_graph(gs, g):
    return P.graph_tensors(g, gs.support, CPU, words=gs.words, n_taxa=N_TAXA, m=M)


def test_popcount_and_hash_lookup_are_exact():
    rng = np.random.default_rng(1)
    x = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=500, dtype=np.int64)
    got = P._popcount64(torch.as_tensor(x)).numpy()
    ref = np.array([bin(int(v) & ((1 << 64) - 1)).count("1") for v in x])
    assert (x < 0).any(), "the sign bit must be exercised"
    np.testing.assert_array_equal(got, ref)

    gs, g, sets = fixture()
    gt = pair_graph(gs, g)
    rows = P._lookup_rows(gt, gt.words)
    np.testing.assert_array_equal(rows.numpy(), np.arange(len(sets)))
    absent = torch.as_tensor(bitset.pack_rows([bitset.from_indices([0, 69], N_TAXA)], N_TAXA)
                             .view(np.int64))
    assert frozenset([0, 69]) not in set(sets)
    assert int(P._lookup_rows(gt, absent)[0]) == -1


def test_row_hash_is_pinned_on_one_word_and_does_not_collide_on_four():
    """One word must keep the formula the n <= 64 checkpoints were trained on; four words are the
    width that collided under the old chain, on sets confined to single words (A108 addendum)."""
    rng = np.random.default_rng(4)
    one = torch.as_tensor(rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                       size=(300, 1), dtype=np.int64))
    ref = ((torch.full((300,), P._FNV_OFFSET, dtype=torch.int64) ^ one[:, 0]) * P._FNV_PRIME)
    assert torch.equal(P._hash_rows(one), ref)

    n = 200
    sets = {frozenset(range(i, i + ln)) for i in range(n) for ln in range(2, 14) if i + ln <= n}
    sets |= {frozenset(rng.choice(64, size=int(rng.integers(2, 9)), replace=False) + 64 * wd)
             for wd in range(3) for _ in range(1500)}
    words = bitset.pack_rows([bitset.from_indices(sorted(c), n) for c in sets], n)
    h = P._hash_rows(torch.as_tensor(np.ascontiguousarray(words).view(np.int64)))
    assert words.shape[1] == 4 and len(sets) > 5000
    assert torch.unique(h).numel() == len(sets), "row hash collided on a four-word set"


def test_containment_is_the_strict_subset_relation():
    gs, g, sets = fixture()
    gt = pair_graph(gs, g)
    got = set(zip(gt.cont_parent.tolist(), gt.cont_child.tolist()))
    ref = {(i, j) for i, a in enumerate(sets) for j, b in enumerate(sets) if b < a}
    assert len(ref) > 200, "the fixture must have nested pairs"
    assert got == ref
    for par, w in ((gt.cont_parent, gt.cont_w_child), (gt.cont_child, gt.cont_w_parent)):
        tot = torch.zeros(len(sets)).index_add(0, par, w)
        has = torch.bincount(par, minlength=len(sets)) > 0
        np.testing.assert_allclose(tot[has].numpy(), 1.0, rtol=1e-5)


def _reference(gs, g, sets, e, row_hash):
    idx = {s: i for i, s in enumerate(sets)}
    sup = np.asarray(gs.support, dtype=np.float64)
    deg = np.diff(g.indptr).astype(np.float64)
    q = int(np.searchsorted(g.indptr, e, side="right") - 1)
    c = int(g.nbr[e])
    A, B = sets[q], sets[c]

    def look(s):
        if len(s) in (1, N_TAXA):
            return 1.0, float(M)
        r = idx.get(frozenset(s), -1)
        return (1.0, sup[r]) if r >= 0 else (0.0, 0.0)

    h_in, sh = look(A & B)
    u_in, su = look(A | B)
    d_in, sd = look(A ^ B)
    x_in, sx = look(A - B)
    w_in, sw = look(B - A)
    lo, hi = int(g.indptr[q]), int(g.indptr[q + 1])
    row_cls = g.cls[lo:hi]
    c_lo, c_hi = int(g.indptr[c]), int(g.indptr[c + 1])
    ns, nh = sup[g.nbr[lo:hi]], row_hash[g.nbr[lo:hi]]
    rank = int((ns > sup[c]).sum() + ((ns == sup[c]) & (nh < row_hash[c])).sum())
    sq, sc = sup[q], sup[c]
    return np.array([
        sq / M, sc / M, np.log((sc + 1) / (sq + 1)) / np.log(M + 1),
        len(A) / N_TAXA, len(B) / N_TAXA, len(A & B) / len(A), len(A & B) / len(B),
        len(A & B) / N_TAXA, h_in, sh / M, u_in, su / M, d_in, sd / M,
        (sq - sd) / max(sq + sd, 1), (sc - sd) / max(sc + sd, 1), x_in, sx / M, w_in, sw / M,
        deg[q] / len(sets), deg[c] / len(sets),
        (row_cls == g.cls[e]).sum() / max(deg[q], 1), (g.cls[c_lo:c_hi] == g.cls[e]).sum() / max(deg[c], 1),
        rank / max(deg[q], 1),
    ])


def test_pair_edge_features_match_set_arithmetic():
    gs, g, sets = fixture()
    gt = pair_graph(gs, g)
    ef = P.pair_edge_features(gt).numpy()
    assert ef.shape == (len(g.nbr), len(P.PAIR_EDGE_FEATURES))
    col = {n: i for i, n in enumerate(P.PAIR_EDGE_FEATURES)}
    h_row = (ef[:, col["h_in"]] == 1) & (ef[:, col["h_over_n"]] * N_TAXA > 1)
    assert h_row.any() and (ef[:, col["h_in"]] == 0).any(), "both intersection cases must occur"
    assert (ef[:, col["d_in"]] == 0).any()
    row_hash = P._hash_rows(gt.words).numpy()
    rng = np.random.default_rng(2)
    for e in rng.choice(len(g.nbr), size=300, replace=False).tolist():
        np.testing.assert_allclose(ef[e], _reference(gs, g, sets, e, row_hash), rtol=1e-5,
                                   atol=1e-6, err_msg=f"entry {e}")


def _policy(containment=True, layers=2, seed=0):
    torch.manual_seed(seed)
    return P.Policy(P.PolicyConfig(n_node_features=31, n_global_features=5, hidden=32,
                                   layers=layers, conv="pair", containment=containment))


def _inputs(gs, seed=0):
    rng = np.random.default_rng(seed)
    return (torch.as_tensor(rng.normal(size=(gs.n_clusters, 31)).astype(np.float32)),
            torch.as_tensor(rng.normal(size=5).astype(np.float32)))


def test_attention_rows_and_sink_sum_to_one():
    gs, g, _ = fixture()
    gt = pair_graph(gs, g)
    pol = _policy()
    x, xg = _inputs(gs)
    ef = P.pair_edge_features(gt)
    h = pol.embed(torch.cat([x, xg.expand(x.shape[0], -1)], 1))
    w, sink = pol.convs[0].attention(h, gt, ef)
    tot = torch.zeros(gs.n_clusters).index_add(0, gt.dst, w) + sink
    np.testing.assert_allclose(tot.detach().numpy(), 1.0, rtol=1e-5)
    assert (w >= 0).all() and (sink > 0).all()


def test_chunking_and_checkpointing_do_not_change_the_answer(monkeypatch):
    gs, g, _ = fixture()
    gt = pair_graph(gs, g)
    x, xg = _inputs(gs)
    pol = _policy()
    ref = pol.encode(x, xg, gt).h.detach()
    monkeypatch.setattr(P, "EDGE_CHUNK", 97)
    np.testing.assert_allclose(pol.encode(x, xg, gt).h.detach().numpy(), ref.numpy(), atol=1e-5)
    grads = []
    for ck in (True, False):
        pol.zero_grad()
        pol.cfg.checkpoint_encoder = ck
        pol.cluster_logits(pol.encode(x, xg, gt)).sum().backward()
        grads.append(pol.convs[0].gate[0].weight.grad.clone())
    np.testing.assert_allclose(grads[0].numpy(), grads[1].numpy(), atol=1e-5)
    assert grads[0].abs().sum() > 0


def test_edge_logits_are_the_gates_and_carry_gradient():
    gs, g, _ = fixture()
    gt = pair_graph(gs, g)
    x, xg = _inputs(gs)
    pol = _policy(layers=3)
    enc = pol.encode(x, xg, gt, keep_layer_inputs=True)
    pos = torch.arange(0, len(g.nbr), 7)
    zs = pol.edge_logits(enc, gt, pos)
    assert len(zs) == 3 and all(z.shape == pos.shape for z in zs)
    torch.stack([z.sum() for z in zs]).sum().backward()
    assert pol.convs[2].gate[0].weight.grad.abs().sum() > 0
    assert pol.embed[0].weight.grad.abs().sum() > 0, "the gate BCE must reach the encoder"


def test_containment_off_is_smaller_and_size_agnostic():
    gs, g, _ = fixture(seed=0)
    gs2, g2, _ = fixture(seed=3, n_random=120, max_len=12)
    assert gs2.n_clusters != gs.n_clusters
    on, off = _policy(True), _policy(False)
    assert sum(p.numel() for p in off.parameters()) < sum(p.numel() for p in on.parameters())
    for gsx, gx in ((gs, g), (gs2, g2)):
        x, xg = _inputs(gsx)
        for pol in (on, off):
            assert pol.encode(x, xg, pair_graph(gsx, gx)).h.shape == (gsx.n_clusters, 32)


def test_segment_is_untouched_by_the_extras():
    gs, g, _ = fixture()
    x, xg = _inputs(gs)
    torch.manual_seed(0)
    pol = P.Policy(P.PolicyConfig(n_node_features=31, n_global_features=5, hidden=32, layers=2))
    assert pol.cfg.conv == "segment"
    a = pol.encode(x, xg, P.graph_tensors(g, gs.support, CPU)).h
    b = pol.encode(x, xg, pair_graph(gs, g)).h
    np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())
    old = {"n_node_features": 31, "n_global_features": 5, "hidden": 32, "layers": 2}
    assert P.PolicyConfig(**old).conv == "segment"
    with pytest.raises(ValueError):
        pol.edge_logits(pol.encode(x, xg, P.graph_tensors(g, gs.support, CPU)), None, None)


def _real_instances(k=4):
    from opc.rl import instances as I
    d = I.cache_dir(REPO / "results", "l2t25", 2)
    paths = sorted(d.glob("*.pkl"))[:k]
    if not paths:
        pytest.skip("no l2t25 instances cached")
    return [I.load(p) for p in paths]


def test_coparent_entries_on_real_instances():
    from opc.rl import hybrid as HY
    n_pos = 0
    for inst in _real_instances():
        th = HY.truth_ha(inst.truth, inst.gs.words)
        tg = PT.edge_targets(inst, th, per_query=8)
        assert tg.n_missing == 0
        rows = np.searchsorted(inst.graph.indptr, tg.pos, side="right") - 1
        cols = inst.graph.nbr[tg.pos]
        wit = [set(int(r) for r in w if r >= 0) for w in th.witness]
        for a, b in zip(rows.tolist(), cols.tolist()):
            assert any(a in w and b in w for w in wit)
        assert not np.isin(tg.hard, tg.pos).any()
        idx, lab = PT.sample_edges(tg, len(inst.graph.nbr), 64, np.random.default_rng(0))
        np.testing.assert_array_equal(lab.astype(bool), np.isin(idx, tg.pos))
        n_pos += tg.pos.size
        gt = P.graph_tensors(inst.graph, inst.gs.support, CPU, words=inst.gs.words,
                             n_taxa=inst.n, m=inst.m)
        assert P.pair_edge_features(gt).shape[0] == len(inst.graph.nbr)
    assert n_pos > 0, "no co-parent pair on any fixture instance -- the test is vacuous"
