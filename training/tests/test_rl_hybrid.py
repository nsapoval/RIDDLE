"""Pins :mod:`opc.rl.hybrid` -- hybrid ancestry off a cluster system, and A108's loss term.

* **The training-time metric is the benchmark's metric.**  ``l2reg_eval/make_csv.py`` reads hybrid
  ancestry off an eNewick string with no parser (``hybrid_ancestry_text``) and scores it with
  ``ha_metrics``; this module reads it off a Hasse diagram.  They must agree on real truths and on
  real relaxed decodes -- including decodes that actually reach level 2, which a support ranking
  does not (A107's G-L24 lesson), so an oracle ranking is run too and its level-2 reach is asserted.
* **Each hand-built fixture asserts it produced its case** before testing what the term does there.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from opc import bitset
from opc.rl import hybrid as HY
from opc.rl import instances as I
from opc.rl import relaxed as RX
from opc.simulate import networks as N

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------------------------------ fixture

#: One reticulation: (b, c) is a hybrid under (a, b, c) and (b, c, e).
ENEWICK = "(((a,(b,c)#H1),d),((#H1,e),f));"


def _fixture():
    net = N.from_enewick(ENEWICK)
    lab = {x: i for i, x in enumerate(net.labels)}
    n = net.n_taxa

    def w(*names):
        return bitset.from_indices(sorted(lab[x] for x in names), n)

    rows = {"bc": w("b", "c"), "abc": w("a", "b", "c"), "abcd": w("a", "b", "c", "d"),
            "bce": w("b", "c", "e"), "bcef": w("b", "c", "e", "f"), "de": w("d", "e")}
    names = list(rows)
    words = np.stack([rows[k] for k in names])
    idx = {k: i for i, k in enumerate(names)}
    truth = {frozenset(c) for c in N.nontrivial_clusters(net)}
    y = np.array([frozenset(bitset.to_indices(words[i], n)) in truth for i in range(len(names))])
    return net, lab, n, words, idx, y


def test_the_fixture_is_the_case_it_claims():
    net, lab, n, words, idx, y = _fixture()
    assert net.n_retics == 1
    assert y.tolist() == [True, True, True, True, True, False]
    t = HY.truth_ha(net, words)
    assert sorted(np.flatnonzero(t.ancestry).tolist()) == sorted([lab["b"], lab["c"]])
    assert len(t.witness) == 1
    assert sorted(t.witness[0].tolist()) == sorted([idx["abc"], idx["bce"]])


def test_a_tree_scored_reticulation_raises_exactly_its_unselected_parent():
    net, lab, n, words, idx, y = _fixture()
    sel = np.array(sorted([idx["bc"], idx["abc"], idx["abcd"]]))
    tg = HY.ha_targets(words, n, sel, y, HY.truth_ha(net, words))
    assert tg.stats["ha_n_pred"] == 0 and tg.stats["ha_fnr"] == 1.0       # the case: P empty
    assert tg.rows.tolist() == [idx["bce"]] and tg.sign.tolist() == [1]
    assert tg.weight.tolist() == [1.0]
    z = torch.randn(len(y), dtype=torch.float64, requires_grad=True)
    loss = HY.ha_loss_terms(z, tg)
    assert torch.isclose(loss, 0.5 * torch.nn.functional.softplus(-z[idx["bce"]]))
    loss.backward()
    assert z.grad[idx["bce"]] < 0                                          # descent raises z
    assert torch.count_nonzero(z.grad) == 1


def test_a_spurious_hybrid_lowers_its_false_parent_and_both_sides_fire():
    net, lab, n, words, idx, y = _fixture()
    sel = np.array(sorted([idx["abc"], idx["abcd"], idx["de"]]))
    tg = HY.ha_targets(words, n, sel, y, HY.truth_ha(net, words))
    assert tg.stats["ha_fp"] == 1 and tg.stats["ha_fn"] == 2              # the case: d is spurious
    got = dict(zip(tg.rows.tolist(), zip(tg.sign.tolist(), tg.weight.tolist())))
    assert got == {idx["bce"]: (1, 1.0), idx["de"]: (-1, 1.0)}
    z = torch.zeros(len(y), requires_grad=True)
    HY.ha_loss_terms(z, tg).backward()
    assert z.grad[idx["bce"]] < 0 < z.grad[idx["de"]]


def test_a_correct_selection_gives_an_exactly_zero_term_that_still_backpropagates():
    net, lab, n, words, idx, y = _fixture()
    sel = np.flatnonzero(y)
    tg = HY.ha_targets(words, n, sel, y, HY.truth_ha(net, words))
    assert tg.stats["ha_ber"] == 0.0 and tg.rows.size == 0
    z = torch.randn(len(y), requires_grad=True)
    loss = HY.ha_loss_terms(z, tg)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.count_nonzero(z.grad) == 0


# ------------------------------------------------------------------ parity with make_csv, real data

def _make_csv():
    here = REPO / "l2reg_eval"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location("l2reg_make_csv", here / "make_csv.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                    # pragma: no cover - environment
        pytest.skip(f"l2reg_eval/make_csv.py does not import here: {e}")
    return mod


def _instances(phase, arm="", n_cells=4):
    d = I.cache_dir(RESULTS, phase, 2, arm)
    files = sorted(d.glob("*.pkl"))
    if not files:
        pytest.skip(f"no instances for {phase} arm={arm or 'true'}")
    return [I.load(f, arm=arm) for f in files[::max(1, len(files) // n_cells)][:n_cells]]


def _indices(labels, names):
    at = {x: i for i, x in enumerate(labels)}
    return {at[x] for x in names}


@pytest.mark.parametrize("phase,arm", [("l2t50", ""), ("l2t25", "est_bp1000"), ("x50", "")])
def test_ancestry_and_rates_match_make_csv_on_real_decodes(phase, arm):
    mc = _make_csv()
    reached_level2 = scored = 0
    for inst in _instances(phase, arm):
        n, labels = int(inst.n), list(inst.labels)
        t = HY.truth_ha(inst.truth, inst.gs.words)
        truth_txt = mc.hybrid_ancestry_text(mc.C.strip_gamma(N.to_enewick(inst.truth)))
        assert set(np.flatnonzero(t.ancestry).tolist()) == _indices(labels, truth_txt)
        y = np.zeros(inst.n_clusters, bool)
        y[inst.truth_rows] = True
        support = np.asarray(inst.gs.support, dtype=np.float64)
        oracle = support + y * (support.max() + 1.0)          # true clusters first, then support
        for scores in (support, oracle):
            dec = RX.RelaxedDecoder(inst.gs.words, n, int(inst.k), 2)
            sel = dec.decode(scores)
            reached_level2 += int(dec.level() >= 2)
            got = RX.realise(inst.gs.words, sel, n, labels)
            assert got is not None
            P = HY.ancestry(HY.selection_full(inst.gs.words, sel, n), n)
            txt = mc.hybrid_ancestry_text(mc.C.strip_gamma(N.to_enewick(got[0])))
            assert set(np.flatnonzero(P).tolist()) == _indices(labels, txt)
            if not t.ancestry.any():
                continue
            ours = HY.rates(P, t.ancestry)
            theirs = mc.ha_metrics(frozenset(txt), frozenset(truth_txt))
            assert ours["ha_fn"] == theirs["ha_fn"] and ours["ha_fp"] == theirs["ha_fp"]
            assert ours["ha_fnr"] == pytest.approx(theirs["ha_fnr"])
            if theirs["ha_balanced_error"] is None:
                assert np.isnan(ours["ha_ber"])
            else:
                assert ours["ha_ber"] == pytest.approx(theirs["ha_balanced_error"])
            tg = HY.ha_targets(inst.gs.words, n, sel, y, t)
            fnr = ours["ha_fnr"] if np.isfinite(ours["ha_fnr"]) else 0.0
            fpr = ours["ha_fpr"] if np.isfinite(ours["ha_fpr"]) else 0.0
            up = float(tg.weight[tg.sign > 0].sum()) + tg.stats["ha_fn_dropped"]
            down = float(tg.weight[tg.sign < 0].sum()) + tg.stats["ha_fp_dropped"]
            assert up == pytest.approx(fnr, abs=1e-5) and down == pytest.approx(fpr, abs=1e-5)
            assert not (y[tg.rows[tg.sign > 0]] == 0).any()      # raised rows are true clusters
            assert not y[tg.rows[tg.sign < 0]].any()             # lowered rows are false ones
            scored += 1
    if phase.startswith("l2t"):
        assert reached_level2 > 0, "no decode reached level 2: the parity never met a level-2 net"
    assert scored > 0


def test_pool_workers_return_what_the_parent_computes():
    inst = _instances("l2t25", "", n_cells=3)
    tasks = []
    for x in inst:
        s = np.asarray(x.gs.support, dtype=np.float32)
        tasks.append((x.gs.words, int(x.n), int(x.k), 2, s, x.truth_rows,
                      HY.truth_ha(x.truth, x.gs.words)))
    here = [HY.decode_and_target(t) for t in tasks]
    with mp.get_context("fork").Pool(2, initializer=HY.worker_init) as pool:
        there = pool.map(HY.decode_and_target, tasks)
    for (s1, t1), (s2, t2) in zip(here, there):
        assert np.array_equal(s1, s2)
        assert np.array_equal(t1.rows, t2.rows) and np.array_equal(t1.weight, t2.weight)
