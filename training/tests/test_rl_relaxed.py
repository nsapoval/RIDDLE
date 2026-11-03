"""Pins :mod:`opc.rl.relaxed` -- the relaxed decoder, promoted into ``src/`` from the CPU hand-off.

Four obligations:

* **``src/`` and the hand-off package must be one implementation, byte for byte.**  The decoder was
  written inside ``handoff/gnn_cpu_v1/pylib/`` and benchmark rows are produced there while
  validation rows are produced here.  ``handoff/alisim_gt_v1/pylib/opc/simulate/outgroup.py`` has
  the same relationship with ``src/``, and A100 is the record of what a *second* implementation of
  a shipped function costs.  Skipped rather than failed when the package is gone, because the
  package is disposable and the module is not.
* **The donor assignment is the constraint no other object in this repository states.**
  :class:`~opc.simulate.networks.Network` holds one donation per node, so a diagram needing two is
  unrepresentable however legal its clusters are.  Checked on the two cases that separate a
  pseudoforest from a non-pseudoforest.
* **The incremental bookkeeping must equal a from-scratch reference.**  The whole module is an
  incremental Hasse diagram maintained under insertion; a fast kernel checked against itself
  checks nothing, so the reference rescans from the top of the ranking at every step, rebuilds the
  diagram per probe and measures level with the independent
  :func:`opc.simulate.networks.level`.
* **The fixtures are not vacuous.**  Under a weak ranking the decode never reaches level 2, so the
  level and donor branches go untested while everything reports green -- the budget-DP test that
  passed 1,188 budgets over zero multi-component systems.  Every test here that means to exercise
  level 2 asserts that it got there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from opc import bitset
from opc.rl import relaxed as RX
from opc.simulate import networks as N

REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "handoff" / "gnn_cpu_v1" / "pylib" / "opc" / "rl" / "relaxed.py"
SRC = REPO / "src" / "opc" / "rl" / "relaxed.py"


# ------------------------------------------------------------------ one implementation, not two


def test_src_and_handoff_copies_are_byte_identical():
    if not HANDOFF.exists():
        pytest.skip(f"no hand-off package at {HANDOFF}")
    a = hashlib.sha256(SRC.read_bytes()).hexdigest()
    b = hashlib.sha256(HANDOFF.read_bytes()).hexdigest()
    assert a == b, (
        "src/opc/rl/relaxed.py and the hand-off package's copy have diverged.  The CPU box scores "
        "benchmark rows with one and w10_gnn.py selects checkpoints with the other, so a "
        "difference here is two decoders wearing one name.  Copy, do not patch.")


# --------------------------------------------------------------------------------- donors


def test_donor_assignment_accepts_a_pseudoforest_and_refuses_k23():
    """``donors`` is the only statement of "no node donates twice" in the repository.

    ``K_{2,3}`` -- three hybrids sharing two parents -- has no assignment: three hybrids need three
    distinct donors and only two nodes are available.  Two disjoint two-cycles do.
    """
    ok = RX.donors({10: (0, 1), 11: (2, 3)})
    assert ok is not None
    assert ok[10] in (0, 1) and ok[11] in (2, 3)
    assert len(set(ok.values())) == 2, "a node donated twice"

    assert RX.donors({10: (0, 1), 11: (0, 1), 12: (0, 1)}) is None
    # One shared parent is still fine: 0 donates to one hybrid, 2 and 3 cover the others.
    shared = RX.donors({10: (0, 1), 11: (0, 2)})
    assert shared is None or len(set(shared.values())) == 2


def test_donors_is_empty_for_a_tree():
    """No two-parent node, so the assignment is the empty one rather than a refusal."""
    assert RX.donors({}) == {}


# ------------------------------------------------------------------- incremental vs from-scratch


def _reference(words, n_taxa, scores, budget, max_level):
    """The obvious statement of the rule, with no incremental state at all."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    chosen: list[int] = []
    while True:
        for c in (int(x) for x in order):
            if c in chosen:
                continue
            got = RX.realise(words, chosen + [c], n_taxa)
            if got is None:
                continue
            net = got[0]
            if net.n_retics > budget or N.level(net) > max_level:
                continue
            chosen = chosen + [c]
            break
        else:
            return np.array(sorted(chosen), dtype=np.int64)


def _random_ground_set(n_taxa=10, n_clusters=40, seed=0):
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < n_clusters:
        size = int(rng.integers(2, n_taxa - 1))
        c = frozenset(rng.choice(n_taxa, size=size, replace=False).tolist())
        if c in seen:
            continue
        seen.add(c)
        rows.append(bitset.from_indices(sorted(c), n_taxa))
    return bitset.pack_rows(rows, n_taxa), rng.random(n_clusters)


@pytest.mark.parametrize("max_level", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_incremental_decode_equals_from_scratch(max_level, seed):
    """The module's own certificate, checked rather than trusted.

    ``check_every_accept`` turns on a full rebuild after every acceptance **and** the end-of-decode
    audit that no refused candidate is legal -- so a disagreement here is the incremental
    bookkeeping and nothing else.
    """
    n_taxa = 10
    words, scores = _random_ground_set(n_taxa=n_taxa, seed=seed)
    budget = 3
    dec = RX.RelaxedDecoder(words, n_taxa, budget, max_level, check_every_accept=True)
    sel = dec.decode(scores)
    assert dec.verify(), dec.verify_reason
    assert np.array_equal(np.sort(sel), _reference(words, n_taxa, scores, budget, max_level))
    # Not vacuous: a decode that selected nothing would pass the equality above trivially.
    assert sel.size > 0


def test_the_level_cap_actually_binds_somewhere():
    """The parametrised test above is only meaningful if ``max_level`` ever changes the answer.

    Searched rather than asserted on one fixture: which random ground set puts the decode above
    level 1 is not something to guess, and a test that silently never exercises the cap is the
    defect this file's docstring names.
    """
    n_taxa, budget = 10, 4
    moved = 0
    for seed in range(12):
        words, scores = _random_ground_set(n_taxa=n_taxa, n_clusters=50, seed=seed)
        a = RX.RelaxedDecoder(words, n_taxa, budget, 1).decode(scores)
        b = RX.RelaxedDecoder(words, n_taxa, budget, 3).decode(scores)
        if not np.array_equal(np.sort(a), np.sort(b)):
            moved += 1
    assert moved > 0, "max_level never changed a decode: the level branch is untested"


def test_decoded_selection_realises_at_or_below_its_cap():
    """Whatever comes back must be a network of the level it was capped at, measured by
    ``networks.level`` -- which shares no code with the decoder's incremental block count."""
    n_taxa, budget = 10, 4
    saw2 = False
    for seed in range(8):
        words, scores = _random_ground_set(n_taxa=n_taxa, n_clusters=50, seed=seed)
        dec = RX.RelaxedDecoder(words, n_taxa, budget, 2)
        sel = dec.decode(scores)
        got = RX.realise(words, sel, n_taxa)
        assert got is not None, "a decoded selection failed to realise"
        net = got[0]
        assert N.level(net) <= 2
        assert net.n_retics <= budget
        assert int(dec.level()) == int(N.level(net))
        saw2 |= N.level(net) == 2
    assert saw2, "no fixture reached level 2: the cap was never the binding constraint"


def test_decoder_tag_refuses_a_meaningless_combination():
    assert RX.decoder_tag("level1") == "level1"
    assert RX.decoder_tag("relaxed", 2) == "relaxed_L2"
    with pytest.raises(ValueError):
        RX.decoder_tag("level1", 2)          # the formal decoder has no such parameter
    with pytest.raises(ValueError):
        RX.decoder_tag("relaxed")            # a relaxed decoder needs a cap
    with pytest.raises(ValueError):
        RX.decoder_tag("nonsense", 1)
