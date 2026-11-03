"""``opc.rl.tob_resolve``: the GNN as a tree-of-blobs resolver.

Neither venv on this Mac has pytest, so this also runs as a script::

    .venv/bin/python tests/test_tob_resolve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pylib"))

from opc import bitset  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl import nodecls as NC  # noqa: E402
from opc.rl import tob_resolve as TR  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.solve import candidates  # noqa: E402

LABELS = ["a", "b", "c", "d", "e", "f"]
NT = len(LABELS)
TOB = "(OUT,((a,b,c),(d,e,f)));"
#: {c,d} crosses both TOB clusters; {a,b}/{b,c} and {d,e}/{e,f} overlap inside one blob each.
CANDS = [("a", "b"), ("b", "c"), ("a", "b", "c"), ("d", "e"), ("e", "f"), ("d", "e", "f"),
         ("c", "d")]


def W(names):
    return bitset.from_indices([LABELS.index(x) for x in names], NT)


def gs_of(cands):
    words = np.stack([W(c) for c in cands])
    return candidates.build(words, np.full(len(cands), 5, dtype=np.int32), NT, m=10, tau=0)


def labels_of(gs, rows):
    return {frozenset(LABELS[i] for i in bitset.to_indices(gs.words[r], NT)) for r in rows}


def net_clusters(enewick):
    net = N.from_enewick(enewick)
    return CN.restrict_clusters(net, list(net.labels))


def test_tob_words_and_crossing():
    gs = gs_of(CANDS)
    tw = TR.tob_words(TOB, LABELS)
    assert labels_of(SimpleGS(tw), range(tw.shape[0])) == {frozenset("abc"), frozenset("def")}
    R = TR.restrict(gs, tw)
    assert labels_of(gs, np.flatnonzero(~R.keep)) == {frozenset("cd")}
    assert R.forced_rows.size == 2 and R.missing.shape[0] == 0


class SimpleGS:
    def __init__(self, words):
        self.words = words


def test_crossing_row_never_taken_even_ranked_first():
    gs = gs_of(CANDS)
    R = TR.restrict(gs, TR.tob_words(TOB, LABELS))
    scores = np.linspace(0.1, 0.9, len(CANDS))            # {c,d}, the last row, ranks first
    a = TR.resolve_level1(gs, OverlapCSR.of(gs), scores, 2, R, LABELS)
    x = TR.resolve_relaxed(gs.words, NT, scores, 2, 2, R, LABELS)
    for got in (a, x):
        assert frozenset("cd") not in labels_of(gs, got["selection"])
        assert {frozenset("abc"), frozenset("def")} <= net_clusters(got["enewick"])
        assert got["k_pred"] == 2, got["k_pred"]          # one reticulation per blob


def test_budget_zero_is_a_refinement_of_the_tob():
    gs = gs_of(CANDS)
    R = TR.restrict(gs, TR.tob_words(TOB, LABELS))
    scores = np.linspace(0.9, 0.1, len(CANDS))
    a = TR.resolve_level1(gs, OverlapCSR.of(gs), scores, 0, R, LABELS)
    x = TR.resolve_relaxed(gs.words, NT, scores, 0, 2, R, LABELS)
    for got in (a, x):
        assert got["k_pred"] == 0
        c = net_clusters(got["enewick"])
        assert {frozenset("abc"), frozenset("def")} <= c
        assert {frozenset("ab"), frozenset("de")} <= c     # the higher-ranked of each pair


def test_missing_tob_cluster_is_appended():
    cands = [c for c in CANDS if c != ("d", "e", "f")]
    gs = gs_of(cands)
    R = TR.restrict(gs, TR.tob_words(TOB, LABELS))
    assert R.missing.shape[0] == 1, "fixture: {d,e,f} should be missing from the candidates"
    scores = np.linspace(0.9, 0.1, len(cands))
    a = TR.resolve_level1(gs, OverlapCSR.of(gs), scores, 1, R, LABELS)
    x = TR.resolve_relaxed(gs.words, NT, scores, 1, 2, R, LABELS)
    for got in (a, x):
        assert frozenset("def") in net_clusters(got["enewick"])
        assert got["k_pred"] <= 1


def test_own_tob_reproduces_the_decode():
    gs = gs_of(CANDS)
    graph = OverlapCSR.of(gs)
    rng = np.random.default_rng(7)
    for _ in range(20):
        scores = rng.random(len(CANDS))
        for b in (0, 1, 2):
            sel = np.sort(NC.decode(ClusterEnv(gs, budget=b, graph=graph), scores).astype(np.int64))
            net, _ = search.realise(gs.words, sel, NT, LABELS)
            own = CN.tree_of_blobs(N.from_enewick(N.to_enewick(net)))
            R = TR.restrict(gs, TR.tob_words(own, LABELS))
            a = TR.resolve_level1(gs, graph, scores, b, R, LABELS)
            assert np.array_equal(a["selection"], sel), (scores, b)


def test_taxon_mismatch_raises():
    try:
        TR.tob_words("(OUT,((a,b,c),(d,e,g)));", LABELS)
    except ValueError:
        return
    raise AssertionError("a TOB over other taxa was accepted")


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(fns)} passed")
