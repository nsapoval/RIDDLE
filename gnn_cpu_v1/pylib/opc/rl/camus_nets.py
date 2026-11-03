"""Third-party **inferred networks** for the camus dataset, and the metrics that compare them.

``camus-data/inferred-networks/<phase>/<rep>/`` holds what other tools produced from the same
replicates: an ASTRAL species tree, a **CAMUS** ladder, and (on some phases) PhyloNet and SNaQ.
This module reads the CAMUS ladder and scores it on exactly the two quantities the RL arm is scored
on -- cluster F1 against ``C(N_true)``, and PhyloNet ``CMPNETS -m luay``.

**CAMUS emits a nested ladder, not one network.**  One rung per reticulation count,
``k = 0, 1, 2, ...``: the ``.csv`` form carries ``k = 0`` (the rooted ASTRAL tree) and a
``Quartet Satisfied Percent`` column, the ``.nwk`` form starts at ``k = 1``.  So "CAMUS's answer"
is not well defined without a rule for picking a rung, and this module therefore scores **every**
rung and leaves the choice to the reader.  The rung a report should lead with is ``k = k_true``,
because that is the budget ``ClusterEnv(inst.gs, budget=inst.k)`` gives the RL arm -- comparing a
method that was told ``k`` against one that was not is not a comparison of methods.

**Clusters are compared as sets of taxon LABELS, never of taxon indices.**
:func:`opc.simulate.networks.nontrivial_clusters` returns index sets, and the index of a taxon is
assigned by whichever parse produced the network -- so two networks over the same taxa number them
differently and an index-set intersection is meaningless.  Measured, that failure is not obviously
wrong from the outside: it put CAMUS's k = 1 F1 at 0.04-0.46 against the 0.64-0.96 it actually
scores, which reads as a poor method rather than as a bug.

**The rep index across the two folders is verified, not assumed.**  An ASTRAL/CAMUS network built
from replicate ``r``'s gene trees must resemble replicate ``r``'s truth far more than replicate
``r'``'s; :func:`alignment_check` measures exactly that, and it is 6/6 on the diagonal at every
phase with matched F1 0.64-0.96 against a mismatched maximum of 0.083.

**Which gene trees each ladder came from is part of the row**, because it is the whole confound.
``astral_iqtree`` ladders (n15, n25 reps 00-19) come from the same IQ-TREE trees as this repo's
``cam_iq500`` arm, so that pair is like-for-like.  ``astral_fasttree`` ladders come from
``g_500.nwk`` -- which is FastTree, established by elimination: at n25 replicates 20-39 the *only*
inferred gene trees in the dataset are ``g_500.nwk`` and the ASTRAL tree there is named
``fasttree``.  Comparing a FastTree-derived CAMUS network against an RL arm run on **true** gene
trees flatters the RL arm, and any table that does it owes the note.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from opc.rl.external import _prune_outgroup
from opc.simulate import networks as N

__all__ = ["METHODS", "PHASES", "LadderCell", "discover", "read_ladder", "label_clusters",
           "restrict_clusters", "ingroup_of", "cluster_counts", "truth_of", "alignment_check"]

#: ``phase -> dataset directory``.  The same six the dataset has; the inferred networks cover
#: replicates 00-19 (and 20-39 for n25's fasttree ladder).
PHASES = {"cam15": "n15", "cam25": "n25", "cam50": "n50",
          "cam100": "n100", "cam150": "n150", "cam200": "n200"}

#: method id -> (filename stem, which gene trees it was built from).  ``f05`` is CAMUS's filter
#: threshold and is the only one present on every phase; n25 also ships f0/f02/f08 as a sweep.
METHODS = {
    "camus_f05_iqtree": ("camus_f05_astral_iqtree", "cam_iq500"),
    "camus_f05_fasttree": ("camus_f05_astral_fasttree", "cam_g500"),
    "camus_f0_fasttree": ("camus_f0_astral_fasttree", "cam_g500"),
    "camus_f02_fasttree": ("camus_f02_astral_fasttree", "cam_g500"),
    "camus_f08_fasttree": ("camus_f08_astral_fasttree", "cam_g500"),
}

_HYB = re.compile(r"#H\d+")


@dataclass(frozen=True)
class LadderCell:
    """One replicate's ladder from one method."""

    phase: str
    rep: int
    method: str
    path: Path

    @property
    def key(self) -> str:
        """``n25_rep00`` -- the same key ``opc.rl.camus`` uses, so rows join on it."""
        return f"n{int(re.sub('[^0-9]', '', self.phase))}_rep{self.rep:02d}"

    @property
    def gene_trees(self) -> str:
        return METHODS[self.method][1]


def discover(root: Path, phase: str, method: str) -> list[LadderCell]:
    """Every replicate for which this method's ladder exists, in either file form."""
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase!r}; expected one of {sorted(PHASES)}")
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    stem = METHODS[method][0]
    base = Path(root) / PHASES[phase]
    if not base.exists():
        return []
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.isdigit()):
        for ext in (".nwk", ".csv"):
            if (d / f"{stem}{ext}").exists():
                out.append(LadderCell(phase, int(d.name), method, d / f"{stem}{ext}"))
                break
    return out


def read_ladder(path: Path) -> dict[int, str]:
    """``{k: eNewick}`` from either file form.

    The ``.csv`` states ``k`` in its own column and the ``.nwk`` does not, so the ``.nwk`` branch
    counts ``#H`` labels.  The two are cross-checked against each other on the ``.csv`` form --
    ``Number of Branches`` has equalled the ``#H`` count on every row read so far -- but the
    column is authoritative where it exists, because it is the tool's own statement.
    """
    path = Path(path)
    if path.suffix == ".csv":
        out = {}
        for row in csv.DictReader(path.open()):
            nwk = row["Extended Newick"].strip()
            k = int(row["Number of Branches"])
            if k != len(set(_HYB.findall(nwk))):
                raise ValueError(
                    f"{path}: row says {k} branches but the eNewick carries "
                    f"{len(set(_HYB.findall(nwk)))} distinct #H labels")
            out[k] = nwk
        return out
    return {len(set(_HYB.findall(l.strip()))): l.strip()
            for l in path.open() if l.strip()}


def label_clusters(net) -> set[frozenset[str]]:
    """``nontrivial_clusters`` as sets of taxon **labels**.

    The index-to-label map is a property of the parse, not of the network, so this conversion is
    what makes two independently parsed networks comparable at all.  See the module docstring for
    what omitting it looks like from the outside.
    """
    return {frozenset(net.labels[i] for i in c) for c in N.nontrivial_clusters(net)}


def restrict_clusters(net, keep) -> set[frozenset[str]]:
    """The clusters ``net`` induces on the taxon subset ``keep``, as label sets.

    **Why this exists rather than pruning the outgroup structurally.**
    :func:`opc.rl.external._prune_outgroup` is exact and refuses anything that is not "``OUT`` is a
    child of a degree-2 root" -- which holds for all 300 truth networks and for 97.25% of CAMUS's
    rungs, but **not** for the rest: CAMUS sometimes places a reticulation on the outgroup's own
    edge (``((#H1,OUT),...)``), and pruning a leaf out from under a hybrid is a different operation
    that can turn a reticulation into a parallel edge.  Restricting the *cluster set* is well
    defined whatever the topology, and it is what a cluster-level metric actually needs.

    **Cross-validated against the exact function, not asserted.**  Over 763 CAMUS rungs at
    cam25/cam50/cam100/cam200 this agrees with ``label_clusters(_prune_outgroup(net))`` on
    **742 of 742** cases where the latter is defined, and supplies an answer on the 21 where it
    is not.  ``tests/test_camus_nets.py`` pins the agreement.
    """
    full = frozenset(keep)
    return {s for s in (frozenset(net.labels[i] for i in c) & full
                        for c in N.hardwired_clusters(net).values())
            if 1 < len(s) < len(full)}


def ingroup_of(net, label: str = "OUT") -> list[str]:
    """``net``'s taxon labels with the outgroup removed, in the network's own order."""
    return [x for x in net.labels if x != label]


def cluster_counts(pred: set[frozenset[str]], true: set[frozenset[str]]) -> dict:
    """``tp/fp/fn/precision/recall/f1``, byte-for-byte ``Instance.cluster_counts``' definition.

    Same denominators -- precision on the prediction's own size, recall on ``|C(N_true)|`` -- so a
    number here sits on the same scale as one from an RL row and the two can go in one table.
    There is no ``fn_reachable`` column: that one is denominated on ``C_tau``, and a method that
    does not build a candidate set has no ``C_tau`` to be denominated on.
    """
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return {
        "tp": tp, "fp": len(pred) - tp, "fn": len(true) - tp,
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if tp else 0.0,
    }


def truth_of(dataset_root: Path, phase: str, rep: int):
    """The replicate's true network, outgroup pruned -- the same object ``opc.rl.camus`` scores."""
    p = Path(dataset_root) / PHASES[phase] / f"{rep:02d}" / "true_net.nwk"
    return _prune_outgroup(N.from_enewick(p.read_text()))


def alignment_check(inferred_root: Path, dataset_root: Path, phase: str, method: str,
                    n: int = 6) -> dict:
    """**G-CAM5** -- do the two folders' replicate indices refer to the same replicate?

    An inferred network built from replicate ``r``'s gene trees must resemble replicate ``r``'s
    truth far more than any other replicate's.  Scored as an ``n x n`` F1 matrix; the gate is that
    the row-wise argmax is on the diagonal every time.  Cheap, and the failure it catches is total:
    a shifted index gives plausible-looking F1 in the 0.1-0.3 range on every row.
    """
    cells = {c.rep: c for c in discover(inferred_root, phase, method)}
    reps = sorted(cells)[:n]
    if not reps:
        return {"state": "SKIPPED", "reason": f"no {method} ladders at {phase}"}
    truths = {r: label_clusters(truth_of(dataset_root, phase, r)) for r in reps}
    hits, diag, off = 0, [], []
    for r in reps:
        ladder = read_ladder(cells[r].path)
        k1 = min(k for k in ladder if k >= 1)
        net = N.from_enewick(ladder[k1])
        pred = restrict_clusters(net, ingroup_of(net))
        scores = {r2: cluster_counts(pred, truths[r2])["f1"] for r2 in reps}
        hits += max(scores, key=scores.get) == r
        diag.append(scores[r])
        off.extend(v for r2, v in scores.items() if r2 != r)
    return {"state": "GREEN" if hits == len(reps) else "RED",
            "reps": len(reps), "argmax_on_diagonal": hits,
            "matched_f1_min": min(diag), "matched_f1_max": max(diag),
            "mismatched_f1_max": max(off) if off else float("nan")}


def tree_of_blobs(net, label: str | None = None) -> str:
    """The network's **tree of blobs**, as a Newick string over ``net``'s taxon labels.

    A tree of blobs says *where* reticulation happened without saying *how*: every biconnected
    component -- on a level-1 network, every cycle -- is contracted to a single node, which becomes
    a polytomy whose children are the components hanging off that cycle.  It is what
    :mod:`BROOQS` takes as ``-t``, and deriving it from the **true** network gives that tool the
    exact location and size of every blob.

    **That is an oracle input and a row using it must say so.**  It is strictly more help than this
    repo's RL arm gets, which is handed only the reticulation *count* ``k`` as a budget -- so
    "BROOQS on the true tree of blobs" is an upper bound on BROOQS, not a competitor's score.  The
    comparable input is an *inferred* tree of blobs (TOB-QMC's ``tree-qmc --blob``).

    Cross-checked against BROOQS's own ``Example/true_tob.nwk``, which is the tree of blobs of
    ``Example/true_net.nwk``; ``tests/test_camus_nets.py`` pins the reproduction.
    """
    # Contract each cycle: every vertex on it collapses onto the cycle's apex (the LCA of the
    # hybrid and its donor), which is the node the blob becomes.
    onto = {}
    for ret in net.retics:
        vs = N.cycle_vertices(net, ret)
        apex = min(vs, key=lambda v: _depth(net, v))
        for v in vs:
            if v != apex:
                onto[v] = apex

    def rep(v: int) -> int:
        while v in onto:
            v = onto[v]
        return v

    kids: dict[int, list[int]] = {}
    for v in N._postorder(net):
        p = net.tree_parent[v]
        if p is None or p < 0:
            continue
        a, b = rep(v), rep(p)
        if a != b:
            kids.setdefault(b, []).append(a)

    def emit(v: int) -> str:
        if v in net.leaf_taxon:
            return net.labels[net.leaf_taxon[v]]
        ch = [emit(c) for c in kids.get(v, [])]
        ch = [c for c in ch if c]
        return "(" + ",".join(ch) + ")" if len(ch) > 1 else (ch[0] if ch else "")

    return emit(rep(net.root)) + ";"


def _depth(net, v: int) -> int:
    d = 0
    while net.tree_parent[v] is not None and net.tree_parent[v] >= 0:
        v = net.tree_parent[v]
        d += 1
    return d


def root_tob_on_outgroup(newick: str, label: str = "OUT") -> str:
    """Re-root a tree of blobs on ``label``'s pendant edge, **keeping** ``label``.

    TOB-QMC returns a *semi-directed* tree of blobs: the outgroup is wherever the traversal put it,
    nested arbitrarily deep rather than sitting beside the root.  Feeding that to a resolver and
    then reading clusters off the result is A82's failure exactly -- an unrooted object read as
    rooted, giving clusters that are wrong with nothing raising.  Measured on ``cam25/rep01``,
    TOB-QMC wrote ``(((...),(((...),OUT)))``: ``OUT`` four levels down.

    :func:`opc.simulate.outgroup.reroot_on_outgroup` roots on that pendant edge but *deletes* the
    outgroup, and both resolvers want it kept -- BROOQS's ``--rooted`` needs an outgroup to trust.
    Rooting a tree on a leaf's pendant edge puts that leaf and the rest of the tree either side of
    the new root, so re-attaching it is the whole operation.
    """
    from opc.simulate import outgroup as OG
    rest = OG.reroot_on_outgroup(newick, label, keep_lengths=False).rstrip(";")
    return f"({label},{rest});"
