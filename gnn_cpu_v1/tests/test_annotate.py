"""``opc.rl.annotate`` and ``to_enewick(node_label=...)``.

Neither venv on this Mac has pytest, so this also runs as a script::

    .venv/bin/python tests/test_annotate.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pylib"))

from opc import bitset  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import annotate as AN  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

LABELS = ["a", "b", "c", "d", "e"]
NT = len(LABELS)


def W(*names):
    return bitset.from_indices([LABELS.index(x) for x in names], NT)


def _overlapping():
    """{a,b,c} and {b,c,d} overlap: the closure adds {b,c}, and {b,c} is a hybrid."""
    return np.stack([W("a", "b", "c"), W("b", "c", "d")])


def test_level1_closure_added_node_maps_to_minus_one():
    cand = _overlapping()
    net, full = search.realise(cand, [0, 1], NT, LABELS)
    rows = AN.node_rows(full, cand)
    bc = [v for v in range(full.shape[0])
          if bitset.to_indices(full[v], NT) == [1, 2]]
    assert bc, "fixture: the closure did not add {b,c} -- the case under test is absent"
    (v,) = bc
    assert rows[v] == -1 and net.is_hybrid(v)
    scored = sorted(int(r) for r in rows if r >= 0)
    assert scored == [0, 1]
    # every scored node's word is the candidate's word
    for u in range(full.shape[0]):
        if rows[u] >= 0:
            assert full[u].tobytes() == cand[rows[u]].tobytes()


def test_relaxed_node_ids_are_selection_rows():
    cand = np.stack([W("a", "b"), W("a", "b", "c"), W("d", "e"), W("c", "d")])
    sel = np.array([1, 0, 2], dtype=np.int64)              # not sorted: the identity must hold
    got = RX.realise(cand, sel, NT, LABELS)
    assert got is not None
    net, full = got
    rows = AN.node_rows(full, cand)
    assert np.array_equal(rows[:len(sel)], sel)
    assert (rows[len(sel):] == -1).all()                   # the trivial clusters


def test_unlabelled_enewick_is_unchanged_and_labels_round_trip():
    cand = _overlapping()
    net, full = search.realise(cand, [0, 1], NT, LABELS)
    rows = AN.node_rows(full, cand)
    z = np.array([7.25, -1.5])
    plain = N.to_enewick(net)
    assert plain == N.to_enewick(net, node_label=None) == N.to_enewick(net, node_label={})
    lab = AN.tree_labels(net, rows, z)
    assert lab, "fixture: no internal tree node is scored"
    text = N.to_enewick(net, node_label=lab)
    assert text != plain
    assert re.sub(r"\)(-?\d+\.\d\d):", "):", text) == plain
    a, b = N.from_enewick(text), N.from_enewick(plain)
    assert a.labels == b.labels and N.full_clusters(a) == N.full_clusters(b)
    assert all(not net.is_hybrid(v) for v in lab)


def test_hybrid_leaf_network_unchanged():
    """A singleton hybrid (written ``(x:0)#Hk``) is the eNewick branch most likely to break."""
    cand = np.stack([W("a", "b"), W("b", "c")])
    net, full = search.realise(cand, [0, 1], NT, LABELS)
    assert any(net.is_hybrid(v) and net.is_leaf(v) for v in range(net.n_nodes)), \
        "fixture: no hybrid leaf"
    rows = AN.node_rows(full, cand)
    plain = N.to_enewick(net)
    text = N.to_enewick(net, node_label=AN.tree_labels(net, rows, np.array([3.0, 4.0])))
    assert re.sub(r"\)(-?\d+\.\d\d):", "):", text) == plain


if __name__ == "__main__":
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(fns)} passed")
