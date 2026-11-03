"""Pins :mod:`opc.rl.l2reg` -- the level-2 training grids, and the plumbing they needed.

The dataset itself is not in the repo, so everything that reads it skips without it.  The parts
that can be pinned without it -- the arm table, the key grammar, the phase registry's absence from
``grid.PHASES`` -- are not skipped, because those are the ones that would silently mis-file a cache
or mis-split a grid.

The one that matters most is the arm table.  ``estimated_bp1000/<key>.nwk.gz`` is topology-only and
``<key>.len.nwk.gz`` is not; A100 measured what the wrong one costs -- every W13 length block
degenerates to a constant while the support cross-check downstream still passes -- so the arm
pointing at the right file is a correctness property, not a spelling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opc import grid
from opc.rl import instances as I
from opc.rl import l2reg as L2
from opc.rl import train as TR
from opc.simulate import networks as N

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "l2reg_train_v1"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _need_data():
    if not DATA.exists():
        pytest.skip(f"no l2reg dataset at {DATA}")


# --------------------------------------------------------------------------- names and plumbing


def test_the_inferred_arm_reads_the_length_bearing_file():
    """A100's failure mode, pinned by name.

    The bare ``.nwk.gz`` under ``estimated_bp1000/`` carries no branch length at all.  Fed to W13,
    ``blen_*`` collapses to 0, ``bdepth_*`` to 1, ``bsib_*`` to 0 and ``brank_*`` becomes the order
    the nodes happen to appear in the string -- and ``_tree_stats``' support cross-check still
    passes, because it counts loci and not lengths.
    """
    assert L2.ARMS["est_bp1000"].endswith(".len.nwk.gz")
    assert L2.ARMS[""].startswith("trees/")


def test_phases_are_not_in_the_grid_registry():
    """``grid.PHASES`` means "a phase ``opc.rungrid`` can generate".

    Every driver that enumerates a phase resolves through that table, and
    :func:`opc.rl.train.select_split` refuses ``split="all"`` on anything in it -- which is exactly
    the convenience an external dataset needs.  :mod:`opc.rl.camus` makes the same argument.
    """
    for phase in L2.PHASES:
        assert phase not in grid.PHASES
        assert TR.select_split(["n25_rep000_x"], "all", phase=phase) == ["n25_rep000_x"]


def test_discover_refuses_an_unknown_phase_or_arm():
    with pytest.raises(KeyError):
        L2.discover(DATA, "l2t99")
    with pytest.raises(KeyError):
        L2.discover(DATA, "l2t25", "est_bp500")


def test_the_key_grammar_survives_split_by_rep_and_sorts_numerically():
    """Two properties of the dataset's own key, both load-bearing and neither ours to choose.

    ``split_by_rep`` reads the replicate out of ``_rep<digits>_``; and six readers glob ``*.pkl``
    and take a **lexicographic** prefix, so the padding is what makes that prefix the numeric one.
    ``x75``'s unpadded keys are the counter-example: there ``rep100`` sorts before ``rep80``.
    """
    keys = [f"n50_rep{r:03d}_l2x1_l1x0_cu0.25" for r in (0, 5, 80, 100, 699, 700, 899)]
    sp = TR.split_by_rep(keys, 700, 800)
    assert [k for k in sp["train"]] == keys[:5]
    assert sp["val"] == [keys[5]]
    assert sp["test"] == [keys[6]]
    assert sorted(keys) == keys, "zero padding must make lexicographic order numeric"


def test_cache_dir_gives_each_arm_its_own_directory():
    """Six readers glob ``*.pkl``; two arms in one directory would silently double all of them,
    and ``w10_gnn_eval`` takes ``test[:max_cells]`` from a sorted list, so it would also change
    *which cells are the test set*."""
    a = I.cache_dir(Path("/r"), "l2t25", 2, "")
    b = I.cache_dir(Path("/r"), "l2t25", 2, "est_bp1000")
    assert a != b and a.name == "instances_tau2"
    assert b.name == "instances_tau2_est_bp1000"


# ----------------------------------------------------------------------------- against the data


def test_truth_matches_the_shipped_ground_truth():
    """The dataset ships ``regular_network.clusters``, computed by a different program.

    That is the repository's cross-validation convention with a genuinely independent
    implementation rather than a restatement of ours.  Compared as **label** sets: a taxon's index
    is a property of the parse, and A103 is the record of what comparing index sets costs.
    """
    _need_data()
    cells = L2.discover(DATA, "l2t25")[:5] + L2.discover(DATA, "l2t50")[:5]
    assert cells, "no cells discovered"
    saw_level2 = False
    for cell in cells:
        net = L2.truth_of(cell)
        meta = L2.spec_of(cell)
        mine = {frozenset(net.labels[i] for i in c) for c in N.nontrivial_clusters(net)}
        theirs = {frozenset(c) for c in meta["regular_network"]["clusters"]}
        theirs = {c for c in theirs if 1 < len(c) < len(net.labels)}
        assert mine == theirs, cell.key
        assert int(net.n_retics) == int(meta["spec"]["k"]), cell.key
        saw_level2 |= N.level(net) == 2
    # Not vacuous: this file exists because the truths are level 2, so say so.
    assert saw_level2, "no fixture was level 2; these tests would prove nothing about this dataset"


def test_a_built_instance_has_this_commit_s_feature_widths():
    """``check_features`` is what refuses a stale cache, so a freshly built instance must pass it.

    Also pins that the level-2 truth reaches ``Instance`` intact -- ``truth_rows`` are rows of the
    ground set, so an empty label set would look like a hard cell rather than a broken build.
    """
    _need_data()
    import tempfile

    cell = L2.discover(DATA, "l2t25")[0]
    with tempfile.TemporaryDirectory() as tmp:
        inst = L2.build(cell, Path(tmp), m=200, tau=2)
    I.check_features(inst)
    assert inst.x_nodes.shape[1] == len(I.NODE_FEATURES)
    assert inst.x_global.shape[0] == len(I.GLOBAL_FEATURES)
    assert inst.n == 25 and inst.k >= 2 and inst.tau == 2 and inst.m == 200
    assert inst.arm == "" and inst.rep == cell.rep
    assert inst.stratum.startswith("l2x")
    assert N.level(inst.truth) == 2
    assert inst.truth_rows.size > 0 and inst.n_true > 0
    y = np.zeros(inst.n_clusters, dtype=bool)
    y[inst.truth_rows] = True
    assert y.any() and not y.all(), "the label vector must carry both classes"


def test_the_two_arms_are_different_objects_over_one_truth():
    """The arms differ in the candidate set, the support and every node feature -- which is why
    they get separate cache directories and why ``instances.load(arm=)`` refuses a cross-arm read.
    The truth is the same network, so the labels must describe the same clusters."""
    _need_data()
    import tempfile

    key = L2.discover(DATA, "l2t25")[0].key
    built = {}
    for arm in ("", "est_bp1000"):
        cell = L2.L2Cell("l2t25", key, arm, DATA)
        with tempfile.TemporaryDirectory() as tmp:
            built[arm] = L2.build(cell, Path(tmp), m=200, tau=2)
    a, b = built[""], built["est_bp1000"]
    assert a.n_true == b.n_true
    assert {frozenset(a.labels[i] for i in c) for c in N.nontrivial_clusters(a.truth)} == \
           {frozenset(b.labels[i] for i in c) for c in N.nontrivial_clusters(b.truth)}
    assert b.arm == "est_bp1000"
    # Not vacuous: if the two arms read the same file this test would pass while proving nothing.
    assert a.gs.words.shape != b.gs.words.shape or not np.array_equal(a.gs.support, b.gs.support)


def test_the_inferred_arm_carries_real_branch_lengths():
    """The other half of A100: ``_tree_stats`` refuses a length-free locus, so a build that
    succeeds already proves the lengths were there -- but the W13 blocks must also be non-constant,
    which is what "degenerate" actually looks like downstream."""
    _need_data()
    import tempfile

    cell = L2.L2Cell("l2t25", L2.discover(DATA, "l2t25")[0].key, "est_bp1000", DATA)
    with tempfile.TemporaryDirectory() as tmp:
        inst = L2.build(cell, Path(tmp), m=200, tau=2)
    for block in ("blen", "bdepth", "bsib"):
        idx = [i for i, name in enumerate(I.NODE_FEATURES) if name.startswith(block + "_")]
        assert idx, block
        col = inst.x_nodes[:, idx]
        assert np.ptp(col) > 0, f"{block}_* is constant -- the length-free file was read"
