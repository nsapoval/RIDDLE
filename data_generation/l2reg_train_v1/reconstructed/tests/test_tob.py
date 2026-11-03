"""The tree-of-blobs stage of the level-2 benchmark (A63).

The true tree of blobs is checked against edge deletion, not against the Hopcroft-Tarjan code it
is computed with.  TREE-QMC is checked end to end on a network whose answer is known -- and the
defect it has there, a root branch its own test rejected surviving into its output, is pinned, so
a TREE-QMC that fixes it fails this file instead of silently changing the data.
"""

from __future__ import annotations

import subprocess
from collections import deque

import pytest

from opc import l2reg
from opc.simulate import level2 as L2
from opc.simulate import msnc_oracle
from opc.simulate import networks as N
from opc.simulate import outgroup as og

requires_treeqmc = pytest.mark.skipif(not l2reg.TREEQMC.exists(),
                                      reason="TREE-QMC not installed")


def _nets():
    for shape in L2.SHAPES:
        yield L2.build_level2(40, L2.Layout(1, 1, (shape,)), seed=11, cu=0.5)[0]
    yield L2.build_level2(60, L2.Layout(2, 1), seed=3, cu=0.5)[0]


def _bridge_clusters_by_deletion(net: N.Network) -> set[frozenset[str]]:
    """Reference: an edge is a bridge iff deleting it disconnects the underlying graph."""
    edges = L2._edges(net)  # noqa: SLF001
    assert len({frozenset(e) for e in edges}) == len(edges), "not a simple graph"
    adj: dict[int, set[int]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    below = N.hardwired_clusters(net)
    out = set()
    for u, v in edges:
        seen, todo = {u}, deque([u])
        while todo:
            x = todo.popleft()
            for y in adj[x]:
                if {x, y} != {u, v} and y not in seen:
                    seen.add(y)
                    todo.append(y)
        if v not in seen and 1 < len(below[v]) < net.n_taxa:
            out.add(frozenset(net.labels[t] for t in below[v]))
    return out


def test_true_tob_matches_edge_deletion():
    polytomous = 0
    for net in _nets():
        true = l2reg.true_tob_clusters(net)
        assert true == _bridge_clusters_by_deletion(net)
        assert true, "no bridges at all -- the comparison would be vacuous"
        polytomous += len(true) < net.n_taxa - 2
    assert polytomous == 5, "every network has a blob, so none of its TOBs can be binary"


def test_laminar_newick_round_trips_the_true_tob():
    for net in _nets():
        true = l2reg.true_tob_clusters(net)
        newick = l2reg.laminar_newick(true, net.labels)
        assert l2reg.newick_clusters(newick) == true
        assert l2reg.tob_problems(newick, set(net.labels)) == []


def test_rooting_a_tree_of_blobs_on_out_keeps_its_polytomies():
    raw = "((t1,t2),t3,(t4,(t5,t6),OUT),t7);"   # unrooted, written from an arbitrary node
    rooted = og.reroot_on_outgroup(raw)
    assert l2reg.newick_clusters(rooted) == {
        frozenset({"t5", "t6"}), frozenset({"t1", "t2"}), frozenset({"t1", "t2", "t3", "t7"})}
    assert rooted.count(",") == 6 and "OUT" not in rooted


def test_tob_problems_fire():
    taxa = {"a", "b", "c", "d"}
    assert l2reg.tob_problems("((a,b),(c,d));", taxa) == []
    assert l2reg.tob_problems("(a,b,c,d);", taxa) == []      # adjacent labels in a polytomy
    assert l2reg.tob_problems("((a,b),(c,c));", taxa)        # duplicate, and d missing
    assert l2reg.tob_problems("((a,b),(c,d);", taxa)         # unbalanced
    assert l2reg.tob_problems("((a,b),(c,d))", taxa)         # no semicolon
    assert l2reg.tob_problems("((a,b),c,d,e);", taxa)        # a label the input never had


# ------------------------------------------------------------ TREE-QMC's contraction, as splits


def _ann(qtt: float, qst: float = 0.0) -> str:
    """A TREE-QMC annotation; note the ``;`` inside the quotes."""
    return f"'[blob_id=0;qtt_p={qtt:e};qst_p={qst:e};qcf_1=1;qcf_2=1;qcf_3=1;minimizer=A/B/C/D]'"


def _pv(root_qtt: float, inner_qtt: float = 0.5, inner_qst: float = 0.0) -> str:
    # get_bipartitions' shape: a binary root whose branch is annotated on its FIRST child only.
    return f"((A,B){_ann(root_qtt)},(C,(D,OUT){_ann(inner_qtt, inner_qst)}));"


def test_a_rejected_root_branch_is_contracted_although_treeqmc_would_keep_it():
    kept, info = l2reg.contract_tob(_pv(root_qtt=1e-9), alpha=1e-7, beta=0.95)
    assert kept == {frozenset("ABC")}
    assert info["root_branch"] == [{"cluster": ["A", "B"], "qtt_p": 1e-9, "qst_p": 0.0,
                                    "contracted": True}]
    # What build_refinement does instead: splice (A,B) into the root and keep (C,(D,OUT)) --
    # whose edge to the root *is* the {A,B} branch.
    assert frozenset("AB") in l2reg.newick_clusters(og.reroot_on_outgroup("(A,B,(C,(D,OUT)));"))


def test_a_kept_root_branch_stays_and_beta_contracts():
    kept, info = l2reg.contract_tob(_pv(root_qtt=0.5), alpha=1e-7, beta=0.95)
    assert kept == {frozenset("AB"), frozenset("ABC")} and not info["root_branch"][0]["contracted"]
    kept, info = l2reg.contract_tob(_pv(root_qtt=0.5, inner_qst=0.99), alpha=1e-7, beta=0.95)
    assert kept == {frozenset("AB")} and info["n_contracted_beta"] == 1


def test_a_branch_treeqmc_never_tested_is_an_error():
    with pytest.raises(ValueError, match="never tested"):
        l2reg.contract_tob(f"((A,B){_ann(0.5)},(C,(D,OUT)));", alpha=1e-7, beta=0.95)


def test_pvalue_tree_is_read_from_the_log():
    log = f"...\nPrinting output tree with pvalues:\n{_pv(0.5)}\nContracting...\n"
    assert l2reg.pvalue_tree(log) == _pv(0.5)


@requires_treeqmc
def test_treeqmc_recovers_a_known_tree_of_blobs_once_its_root_is_contracted(tmp_path):
    net, _ = L2.build_level2(8, L2.Layout(1, 0, ("S0",)), seed=5, cu=0.5)
    true = l2reg.true_tob_clusters(net)
    assert 0 < len(true) < net.n_taxa - 2, "the fixture must have both a blob and a bridge"
    trees = msnc_oracle.simulate_gene_trees(net, 2000, seed=3)
    gt = tmp_path / "gt.nwk"
    gt.write_text("".join(f"({t.strip().rstrip(';')},{og.OUTGROUP_LABEL});\n" for t in trees))
    out = tmp_path / "tob"
    res = subprocess.run([str(l2reg.TREEQMC), *l2reg.TOB_ARGS, "-i", str(gt), "-o", str(out)],
                         capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    raw = out.read_text()
    assert l2reg.tob_problems(raw, set(net.labels) | {og.OUTGROUP_LABEL}) == []

    est, info = l2reg.contract_tob(l2reg.pvalue_tree(res.stdout), alpha=1e-7, beta=0.95)
    assert est == true
    # The defect, pinned: TREE-QMC's own output keeps exactly the root branch its test rejected.
    got = l2reg.newick_clusters(og.reroot_on_outgroup(raw))
    assert got - est == {frozenset({"t3", "t4", "t5", "t8"})} and est <= got
    assert [r["contracted"] for r in info["root_branch"]] == [True]
