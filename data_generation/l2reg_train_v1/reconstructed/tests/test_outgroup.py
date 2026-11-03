"""Pins :mod:`opc.simulate.outgroup` -- the graft/reroot/prune round trip the arm rests on.

Everything here is exercised against dendropy as an independent oracle, the same division
``tests/test_extract.py`` uses: hand-rolled in ``src/`` because it runs over millions of trees,
dendropy in the tests because correctness matters more than speed there.
"""

from __future__ import annotations

import gzip
import random
import time
from pathlib import Path

import dendropy
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from opc.clusters import extract
from opc.simulate import outgroup as og

REPO = Path(__file__).resolve().parents[1]
CACHED = REPO / "results" / "full" / "genetrees_julia" / "n25_rep0_r0_cu0.5_g0.5.nwk.gz"

SLOW = settings(max_examples=200, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])


def clusters(newick: str, n_taxa: int) -> set[int]:
    index = {f"t{i}": i - 1 for i in range(1, n_taxa + 1)}
    return set(extract.clusters_from_newick(newick, index, []))


def random_ultrametric(n_taxa: int, rng: random.Random) -> str:
    """A random ultrametric rooted tree, built the way a coalescent builds one."""
    nodes = [(f"t{i}", 0.0) for i in range(1, n_taxa + 1)]
    height = 0.0
    while len(nodes) > 1:
        height += rng.uniform(0.05, 1.0)
        i, j = rng.sample(range(len(nodes)), 2)
        (a, ha), (b, hb) = nodes[i], nodes[j]
        merged = f"({a}:{height - ha:.8f},{b}:{height - hb:.8f})"
        nodes = [nd for k, nd in enumerate(nodes) if k not in (i, j)] + [(merged, height)]
    return nodes[0][0] + ";"


@st.composite
def ultrametric_trees(draw, min_taxa: int = 2, max_taxa: int = 12):
    n = draw(st.integers(min_value=min_taxa, max_value=max_taxa))
    seed = draw(st.integers(min_value=0, max_value=2 ** 30))
    return n, random_ultrametric(n, random.Random(seed))


def cached_trees(count: int = 40):
    if not CACHED.exists():
        pytest.skip(f"no cached gene trees at {CACHED}")
    with gzip.open(CACHED, "rt") as fh:
        return [next(fh).strip() for _ in range(count)]


# ------------------------------------------------------------------- the load-bearing property


@given(ultrametric_trees())
@SLOW
def test_graft_then_root_and_prune_is_the_identity_on_the_ingroup_topology(case):
    n, tree = case
    p, ln, _, ch = og.parse_newick(tree)
    h = og.root_height(p, ln, ch)
    grafted = og.graft_outgroup(tree, d=0.5 * h + 0.1, t_out=1.5 * h + 0.1)
    assert clusters(og.reroot_on_outgroup(grafted), n) == clusters(tree, n)


def test_graft_then_root_and_prune_is_the_identity_on_real_gene_trees():
    for tree in cached_trees():
        p, ln, _, ch = og.parse_newick(tree)
        h = og.root_height(p, ln, ch)
        grafted = og.graft_outgroup(tree, d=0.5 * h, t_out=1.5 * h, scale=0.018)
        assert clusters(og.reroot_on_outgroup(grafted), 25) == clusters(tree, 25)


# ------------------------------------------------------------------------------ vs dendropy


def test_root_height_matches_dendropy():
    for tree in cached_trees(20):
        p, ln, _, ch = og.parse_newick(tree)
        d = dendropy.Tree.get(data=tree, schema="newick", rooting="force-rooted")
        d.calc_node_ages(ultrametricity_precision=1e-6)
        assert og.root_height(p, ln, ch) == pytest.approx(d.seed_node.age, abs=1e-7)


def test_node_masks_agree_with_clusters_from_newick():
    index = {f"t{i}": i - 1 for i in range(1, 26)}
    for tree in cached_trees(20):
        p, _, lab, ch = og.parse_newick(tree)
        masks = og.node_masks(p, lab, ch, index)
        mine = {masks[i] for i in range(len(p)) if ch[i]}
        assert mine == set(extract.clusters_from_newick(tree, index, []))


def test_rerooting_matches_dendropys_outgroup_rooting():
    for tree in cached_trees(20):
        p, ln, _, ch = og.parse_newick(tree)
        h = og.root_height(p, ln, ch)
        grafted = og.graft_outgroup(tree, d=0.5 * h, t_out=1.5 * h)

        d = dendropy.Tree.get(data=grafted, schema="newick", rooting="force-unrooted")
        d.to_outgroup_position(d.find_node_with_taxon_label(og.OUTGROUP_LABEL),
                               update_bipartitions=False)
        d.prune_taxa_with_labels([og.OUTGROUP_LABEL])
        theirs = d.as_string(schema="newick").replace("[&U]", "").replace("[&R]", "").strip()

        assert clusters(og.reroot_on_outgroup(grafted), 25) == clusters(theirs, 25)


@given(ultrametric_trees(min_taxa=3))
@SLOW
def test_the_grafted_tree_is_still_ultrametric(case):
    _, tree = case
    p, ln, _, ch = og.parse_newick(tree)
    h = og.root_height(p, ln, ch)
    grafted = og.graft_outgroup(tree, d=0.25 * h + 0.1, t_out=1.25 * h + 0.1, scale=0.018)
    d = dendropy.Tree.get(data=grafted, schema="newick", rooting="force-rooted")
    d.calc_node_ages(ultrametricity_precision=1e-6)  # raises if not ultrametric
    # Lengths are written at 8 decimal digits, so a root-to-tip path of k branches carries up to
    # k * 1e-8 of rounding.
    assert d.seed_node.age == pytest.approx((1.25 * h + 0.1) * 0.018, abs=5e-7)


# ------------------------------------------------------------------------------- edge cases


def test_reroot_handles_an_outgroup_that_is_already_a_child_of_the_root():
    # A rooted binary tree with OUT as one of two root children: suppressing the old root must
    # leave no unary node behind.
    tree = "(OUT:0.9,((t1:0.1,t2:0.1):0.2,t3:0.3):0.6);"
    assert og.reroot_on_outgroup(tree) == "((t1,t2),t3);"


def test_reroot_handles_the_unrooted_trifurcation_iqtree_emits():
    tree = "(OUT:0.1,(t1:0.1,t2:0.1):0.1,((t3:0.1,t4:0.1):0.1,t5:0.1):0.1);"
    assert clusters(og.reroot_on_outgroup(tree), 5) == clusters("((t1,t2),((t3,t4),t5));", 5)


def test_reroot_handles_a_nested_outgroup():
    tree = "((OUT:0.1,t1:0.1):0.1,t2:0.1,(t3:0.1,(t4:0.1,t5:0.1):0.1):0.1);"
    assert clusters(og.reroot_on_outgroup(tree), 5) == clusters("(t1,(t2,(t3,(t4,t5))));", 5)


def test_reroot_handles_a_polytomous_root():
    tree = "(OUT:0.1,t1:0.1,t2:0.1,(t3:0.1,t4:0.1):0.1);"
    rooted = og.reroot_on_outgroup(tree)
    _, _, _, ch = og.parse_newick(rooted)
    assert len(ch[0]) == 3
    assert clusters(rooted, 4) == clusters("(t1,t2,(t3,t4));", 4)


def test_reroot_rejects_a_tree_without_the_outgroup():
    # Silence here would be the dangerous outcome: an unrooted tree yields wrong clusters and
    # nothing downstream raises, because clusters_from_newick only rejects *unknown* labels.
    with pytest.raises(ValueError, match="not found"):
        og.reroot_on_outgroup("((t1:0.1,t2:0.1):0.1,t3:0.1);")


def test_reroot_rejects_an_outgroup_that_is_not_a_leaf():
    with pytest.raises(ValueError):
        og.reroot_on_outgroup("((t1:0.1,t2:0.1)OUT:0.1,t3:0.1,t4:0.1);")


def test_parser_tolerates_support_values_and_internal_labels():
    a = og.reroot_on_outgroup("(OUT:0.1,(t1:0.1,t2:0.1)100:0.1,(t3:0.1,t4:0.1)i7:0.1);")
    assert clusters(a, 4) == clusters("((t1,t2),(t3,t4));", 4)


# ------------------------------------------------------------------------------ file level


def test_graft_file_fixes_the_outgroup_divergence_over_the_whole_cell(tmp_path):
    """T_out must not depend on which slice of the cell a chunk happens to cover."""
    whole = og.graft_file(CACHED if CACHED.exists() else pytest.skip("no cached trees"),
                          tmp_path / "a.nwk", alpha=0.5, count=40)
    slice_ = og.graft_file(CACHED, tmp_path / "b.nwk", alpha=0.5, start=20, count=20, cell_m=40)
    assert slice_["t_out"] == whole["t_out"]
    assert slice_["n_loci"] == 20
    # and the grafted trees for the shared range are byte-identical
    tail = (tmp_path / "a.nwk").read_text().split()[20:]
    assert (tmp_path / "b.nwk").read_text().split() == tail


def test_root_and_prune_rejects_a_tree_with_a_missing_taxon(tmp_path):
    src = tmp_path / "raw.nwk"
    src.write_text("(OUT:0.1,(t1:0.1,t2:0.1):0.1,t3:0.1);\n")
    with pytest.raises(ValueError, match="expected"):
        og.root_and_prune_file(src, tmp_path / "out.nwk.gz", [f"t{i}" for i in range(1, 5)])


def test_root_and_prune_scores_zero_rf_against_the_trees_it_came_from(tmp_path):
    trees = cached_trees(30)
    true_gz = tmp_path / "true.nwk.gz"
    with gzip.open(true_gz, "wt") as fh:
        fh.write("\n".join(trees) + "\n")
    grafted = tmp_path / "grafted.nwk"
    og.graft_file(true_gz, grafted, alpha=0.5)
    stats = og.root_and_prune_file(grafted, tmp_path / "out.nwk.gz",
                                   [f"t{i}" for i in range(1, 26)], true_gz=true_gz)
    assert stats["n_loci"] == 30
    assert stats["rf_norm_mean"] == 0.0
    assert stats["frac_rf_zero"] == 1.0
    assert stats["frac_root_split_correct"] == 1.0


def test_rooted_output_is_consumable_by_extract(tmp_path):
    trees = cached_trees(20)
    true_gz = tmp_path / "true.nwk.gz"
    with gzip.open(true_gz, "wt") as fh:
        fh.write("\n".join(trees) + "\n")
    og.graft_file(true_gz, tmp_path / "g.nwk", alpha=0.5)
    og.root_and_prune_file(tmp_path / "g.nwk", tmp_path / "out.nwk.gz",
                           [f"t{i}" for i in range(1, 26)])
    index = {f"t{i}": i - 1 for i in range(1, 26)}
    counter = extract.count_newick_file(tmp_path / "out.nwk.gz", index, (10, 20))
    masks, support, n_raw = counter.finalize(min_support=2)
    assert len(masks) > 0 and n_raw[-1] > 0


def test_throughput_is_adequate():
    """Millions of trees pass through here; dendropy would be milliseconds each."""
    trees = cached_trees(40)
    grafted = []
    for t in trees:
        p, ln, _, ch = og.parse_newick(t)
        h = og.root_height(p, ln, ch)
        grafted.append(og.graft_outgroup(t, 0.5 * h, 1.5 * h))
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        for g in grafted:
            og.reroot_on_outgroup(g)
    rate = reps * len(grafted) / (time.perf_counter() - t0)
    assert rate > 2000, f"only {rate:.0f} reroots/s at n=25"
