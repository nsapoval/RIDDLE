#!/usr/bin/env python3
"""The l2reg_n200_v1 per-network CSV, in ``camus_a104_per_network.csv``'s shape -- and its parity gates.

Rows (``gene_trees = est_bp1000``, ``n = 200``, ``network_idx = rep``, ``seed = 0`` but for the GNN):

    ours/prior, ours/2x-data-full, ours/+n75   GNN runs (runs/gnn*/), seeds 0-2; ours/2x-data too
                                               when a _mix2 run exists
    greedy                                     greedy runs (runs/greedy*/, not greedy_camus*)
    TOB-QMC, TOB-QMC (fixed), true TOB*        the staged trees of blobs, read as trees (k_pred = 0)
    NetCS, BROOQS                              on the raw TOB -- the camus protocol, camus's labels
    NetCS (fixed TOB), BROOQS (fixed TOB)      on the corrected TOB
    NetCS*(true TOB), BROOQS*(true TOB)        on the true TOB (oracle)

**Decoders.**  Every GNN and greedy cell records the decoder it was run with (``opc.rl.relaxed``).
The formal level-1 decoder keeps the plain labels; a relaxed run is its own set of rows labelled
``ours/+n75 [relaxed L2]``, ``greedy [relaxed L2]`` and so on, and the ``decoder`` column says
``level1`` or ``relaxed_L<L>`` (empty for the tree-of-blobs methods, which have no decoder).  The
level-1 GNN and greedy runs are required; a relaxed variant is optional, but must be complete for
every checkpoint tag it carries.

Columns: camus_a104's eleven in its order, then ``decoder, key, cu, n_l2, n_l1, n_blobs_tob, tp,
fp, fn, precision, recall, mass_pred, mass_truth, cluster_f1_rooted, status``, then hybrid
ancestry's ``ha_n_true, ha_n_pred, ha_fp, ha_fn, ha_fpr, ha_fnr, ha_balanced_error``.  Empty
fields are ``''``, as in camus_a104.

Cluster F1
----------
The landed definitions, one per source: the GNN's own ``Instance.cluster_counts`` on its selection
(the landed ``f1_selected``), the same for greedy, and ``camus_nets.cluster_counts`` over the
OUT-restricted clusters for everything else (the landed third-party definition).  A resolver
network with OUT below one of its own hybrids has no OUT-rooted reading: ``cluster_f1`` then takes
the value read off the network as emitted, as camus_a104's 15 + 2 such NetCS rows did, and
``cluster_f1_rooted`` -- the principled one -- is empty, with ``status = out_not_rootable``.

Luay: PhyloNet and nothing else
-------------------------------
Every distance and every network size in this file comes from PhyloNet's ``CMPNETS -m luay``, in
one JVM per invocation.  ``opc.mpl.netdist`` is deliberately not imported: its distance disagrees
with the jar on 374 of 961 camus GNN and greedy networks -- 370 of them carrying a hybrid with two
or more children, the regular-form hybrid ``closure.network_from_hasse`` builds -- while the jar
reproduces every landed luay (900 of 900 GNN rows, 61 of 61 greedy cells; measured 2026-09-11).

* ``luay``: ``CMPNETS -m luay`` between the prediction and the truth, both OUT-free and
  gamma-stripped.  The truth is ``to_enewick(truth_of(cell))``; a prediction is the GNN/greedy
  eNewick as written, or ``_prune_outgroup`` of the OUT-rooted resolver/TOB string.
* ``mass_pred``, ``mass_truth``: each network's multiset size as the jar counts it, measured with
  the jar.  CMPNETS returns ``floor(|M1 symdiff M2| / 2)``, and a leaf matches only the leaf of the
  same name.  So compare a network N on leaves L with the star S on L, and with S', the star under
  one unary node: neither shares an internal class with N, the two distances are
  ``floor((X + 1) / 2)`` and ``floor((X + 2) / 2)`` for ``X = |M_N| - |L|``, and
  ``X = luay(N, S) + luay(N, S') - 1`` exactly.  ``mass = X + |L|``.  On the camus networks this
  equals netdist's ``traversal_mass`` for 1,544 of 1,544 predictions -- netdist's sizes were never
  the problem, its class matching is -- and the landed ``mass_*`` columns everywhere but one
  truth, cam150 rep04, whose size-2 cycle the jar sizes differently from the landed number.  No
  l2reg network has a cycle that short (girth >= 5).
* ``luay_norm = luay / (n - 1 + floor(3 (k_true + k_pred) / 2))``: the closed-form maximum for
  binary networks, as camus_a104 used.
* ``luay_norm_exact = luay / floor((X_pred + X_truth) / 2)``: the most two networks on one leaf
  set can differ, counted the jar's way, so it is at most 1 by construction (1,544 of 1,544 camus
  pairs).  It is camus_a104's ``luay_norm_exact`` wherever the sizes agree, i.e. everywhere above.

Hybrid ancestry
---------------
Dai & Molloy, *Is level-1 blob reconstruction under the network multispecies coalescent easy?*
(bioRxiv 10.64898/2026.06.06.730607, Fig. 3): "A species has hybrid ancestry if it is the
descendant of one or more hybrids. [...] FPR is the number of FPs divided by the number of species
with hybrid ancestry according to the estimated network. FNR is the number of FNs divided by the
number of species with hybrid ancestry according to the true network."  With P and T the ingroup
taxa below at least one hybrid of the prediction and of the truth:

* ``ha_n_pred = |P|``, ``ha_n_true = |T|``, ``ha_fp = |P - T|``, ``ha_fn = |T - P|``.
  ``ha_fpr = ha_fp / |P|`` -- a false *discovery* rate, whatever its name -- and
  ``ha_fnr = ha_fn / |T|``; ``ha_balanced_error`` is their mean, the number the paper reports.
* **An empty P leaves ``ha_fpr`` and ``ha_balanced_error`` empty.**  The paper's FPR is 0/0 there,
  a case it never meets (every method in its Fig. 3 emits hybrids); here it is every tree-of-blobs
  row, which still gets its counts and ``ha_fnr = 1``.  T is never empty: k >= 3.
* The species are the 200 ingroup taxa; OUT is dropped from P.  Each row is read off the network
  its luay compares.  The four ``out_not_rootable`` NetCS rows have no luay and are read off the
  network as emitted, and that is their value, not a stand-in for one: hybrid ancestry is a
  property of the semi-directed network.  A valid root lies in the tree part above every hybrid,
  so re-rooting reverses only edges no hybrid reaches (``blob_cpu_v1/run_blob_cpu.py`` makes the
  same argument for OUT).  NetCS put OUT below a hybrid in those four, and 110-172 of the 200
  ingroup taxa with it.

Two derivations, cross-checked on every network, the truths included: the hybrids' hardwired
clusters on the parsed ``Network``, and a scan of the eNewick string for the leaves inside each
``(...)#Hk`` body, which needs no parser.  A disagreement empties that row's ``ha_*`` and is a
problem; ``status`` is never touched, so every earlier column re-derives unchanged.  They agree
on 800 of 800 (the 20 truths and every row's network; 2026-09-11).  The rows cannot test the
rooting argument -- all 116 rootable resolver networks were emitted already rooted on OUT -- so it
was checked by re-rooting instead, with ``run_blob_cpu.reroot_enewick``: 7,400 re-roots of the
truth, GNN, greedy and resolver networks on taxa outside P left P unchanged every time, and the
1,479 taxa inside P it was offered were all refused, as they must be.

Gene-tree scaling (``REG_GT``)
------------------------------
``REG_GT=gt5k`` / ``gt10k`` builds the same table for the 5,000- / 10,000-tree dataset
(``common.py``), written to ``results/rl/<dataset>_per_network.csv`` with ``gene_trees =
est_bp1000_gt5k`` / ``_gt10k``.  Its runs were scoped narrower, and completeness follows the scope:
``_l2mix_pair`` (both checkpoints) at the headline relaxed decoder, greedy at level 1 and that
decoder, and the resolvers on the raw and fixed trees of blobs (``common.GNN_REQUIRED``,
``GREEDY_REQUIRED``, ``RESOLVER_VARIANTS``).  **Every GNN and greedy cell must show it read this
dataset's trees at its threshold** -- ``n_loci == m == common.M`` and ``tau == common.TAU``, and a
GNN cell's ``tau_frac == common.TAU_FRAC`` -- because until 2026-09-21 every driver read the first
1,000 trees of any file and said nothing.  The same check holds the v1 cells to (1000, 2).

--parity
--------
Against the landed camus rows, before trusting any of the above.  Two misses are known and
explained, and are allowed exactly the fields they explain:

    cam150 rep04, any row     mass_truth       -- the size-2-cycle truth (above)
    TOB-QMC cam200 rep15      luay, mass_pred  -- the Mac tree of blobs is not the Linux one

(a) GATE  the landed NetCS/BROOQS strings, the Mac TOB-QMC strings and the true trees of blobs
          reproduce the landed luay and both masses.
(b) GATE  the Mac GNN networks in ``gnn_cpu_v1/runs/C4537CLFY0_t1`` reproduce the landed GNN luay
          and masses.
(c) GATE  greedy on the 100 camus cells (``run_greedy.py --camus``, the formal decoder) reproduces
          the landed tp, fp, k_pred, F1, mass and luay exactly -- it is deterministic.

Usage (``gnn_cpu_v1/.venv/bin/python``)
-----
    python l2reg_eval/make_csv.py --parity
    python l2reg_eval/make_csv.py

``--phylonet`` defaults to ``$OPC_PHYLONET``, else ``~/research/stochastic-search/tools/PhyloNet.jar``;
the system Java 8 runs it (``phylonet.JAVA``, ``$OPC_JAVA``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import common as C

import pyarrow.parquet as pq  # noqa: E402

from opc.mpl import phylonet as PN  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

CAMUS_COLUMNS = ("method", "gene_trees", "n", "network_idx", "seed", "cluster_f1", "luay",
                 "luay_norm", "luay_norm_exact", "k_pred", "k_true")
#: Hybrid ancestry (module docstring): appended, so every column before it stays where it was.
HA_COLUMNS = ("ha_n_true", "ha_n_pred", "ha_fp", "ha_fn", "ha_fpr", "ha_fnr", "ha_balanced_error")
EXTRA_COLUMNS = ("decoder", "key", "cu", "n_l2", "n_l1", "n_blobs_tob", "tp", "fp", "fn",
                 "precision", "recall", "mass_pred", "mass_truth", "cluster_f1_rooted",
                 "status") + HA_COLUMNS
METHOD_ORDER = ("TOB-QMC", "TOB-QMC (fixed)", "true TOB*", "greedy", "ours/prior", "ours/2x-data",
                "ours/2x-data-full", "ours/+n75", "ours/l2mix", "ours/l2mix75", "ours/l2mixHA/selF1",
                "ours/l2mixHA/selHA", "ours/l2mix_pair/selF1", "ours/l2mix_pair/selHA", "NetCS",
                "NetCS (fixed TOB)", "NetCS*(true TOB)",
                "BROOQS", "BROOQS (fixed TOB)", "BROOQS*(true TOB)")
#: Checkpoint tag -> the seeds it must have, read off ``gnn_cpu_v1/checkpoints/<tag>/s*/`` rather
#: than assumed flat -- ``l3reg_eval/make_csv.py``'s rule, ported 2026-09-13.  The flat ``SEEDS``
#: this file had turned every family shipping fewer than three seeds (``_l2mix``: two,
#: ``_l2mix75``: one) into spurious "missing (seed, rep)" problems, which is why the 2026-09-12 build
#: records ``"complete": false``.  A tag with no checkpoint directory falls back to ``SEEDS``.
SEEDS = (0, 1, 2)
def _required_seeds(root: Path) -> dict[str, tuple[int, ...]]:
    """Model tag -> seeds, one entry per checkpoint *file*: ``<tag>/s<N>/best.pt`` is ``<tag>``, and
    any other ``best_<x>.pt`` beside it is ``<tag>:best_<x>`` -- ``run_gnn_cpu.py --models``' naming,
    so ``_l2mixha/s0/best_ha.pt`` (2026-09-14) is required as a model in its own right."""
    out: dict[str, set[int]] = {}
    for d in sorted(root.glob("_*")):
        for s in sorted(d.glob("s*")):
            if s.is_dir() and s.name[1:].isdigit():
                for f in sorted(s.glob("best*.pt")):
                    key = d.name if f.stem == "best" else f"{d.name}:{f.stem}"
                    out.setdefault(key, set()).add(int(s.name[1:]))
    return {k: tuple(sorted(v)) for k, v in sorted(out.items())}


REQUIRED_SEEDS = _required_seeds(C.GNN / "checkpoints")
REQUIRED_TAGS = tuple(sorted(REQUIRED_SEEDS) or ("_mix", "_mix2L", "_mix75"))
COUNTS = ("tp", "fp", "fn", "precision", "recall")
#: camus cell arm -> the suffix in ``rl_gnn_eval_cam{N}{suffix}_...`` file names.
LANDED_ARM = {"cam_g500": "g500", "cam_iq500": "iq500"}
MAC_GNN_RUN = C.GNN / "runs" / "C4537CLFY0_t1"
MAC_BLOB_RUN = C.BLOB / "runs" / "C4537CLFY0_j8"
DEFAULT_JAR = Path.home() / "research" / "stochastic-search" / "tools" / "PhyloNet.jar"
#: A leaf label: a name right after "(" or ",".  Never a "#H" reference or an internal label.
_LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")
#: What closes a hybrid's body: an optional internal label, then the ``#Hk`` tag.
_BODY_TAG = re.compile(r"[^(),:;#\s\[\]]*#[A-Za-z]*\d+")
#: A leaf that is itself a hybrid: ``label#Hk``.
_LEAF_HYBRID = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)#[A-Za-z]*\d+")
_FIELDS = ("luay", "mass_pred", "mass_truth")


def missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pruned(rooted: str) -> str:
    """An OUT-rooted network, OUT pruned and gammas stripped: what the landed luay compared."""
    return C.strip_gamma(N.to_enewick(_prune_outgroup(N.from_enewick(rooted))))


def leaf_set(enewick: str) -> tuple[str, ...]:
    return tuple(sorted(set(_LEAF.findall(enewick))))


def closed_form_max(n: int, k1: int, k2: int) -> int:
    """``n - 1 + floor(3 (k1 + k2) / 2)``: camus_a104's ``luay_norm`` denominator -- the most two
    rooted binary networks on n leaves with k1 and k2 reticulations can differ.  The derivation is
    in ``opc.mpl.netdist``'s docstring; only the arithmetic is used here."""
    return max(0, n - 1 + (3 * (k1 + k2)) // 2)


def variant_rank(tag: str) -> int:
    """``level1`` first, then relaxed variants by level cap."""
    return 0 if tag in ("", "level1") else int(tag.rsplit("_L", 1)[1])


def row_order(row: dict, order: dict) -> tuple:
    return (order.get(row["_base"], len(order)), variant_rank(row["decoder"]))


class PhyloNetLuay:
    """Every luay and every network size, from PhyloNet alone, in one JVM (module docstring)."""

    def __init__(self, jar: Path):
        self.jar = jar
        self.pairs: list[tuple[str, str]] = []

    def add(self, pred: str, truth: str) -> int:
        self.pairs.append((pred, truth))
        return len(self.pairs) - 1

    def resolve(self, workdir: Path) -> None:
        PN.PHYLONET = Path(self.jar)                  # read at call time by ``available()``
        if not PN.available():
            raise SystemExit(f"PhyloNet not usable: jar {PN.PHYLONET}, java {PN.JAVA}")
        nets = list(dict.fromkeys(s for pair in self.pairs for s in pair))
        self.n_leaves = {s: len(leaf_set(s)) for s in nets}
        cmds, at = list(self.pairs), {}
        for s in nets:
            star = "(" + ",".join(leaf_set(s)) + ")"
            at[s] = len(cmds)
            cmds += [(s, star + ";"), (s, f"({star});")]
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        got = [r["luay"] for r in PN.cmpnets_many(cmds, methods=("luay",), workdir=workdir)]
        self.luay = [None if missing(v) else int(v) for v in got[:len(self.pairs)]]
        self.size = {s: (None if missing(got[i]) or missing(got[i + 1])
                         else int(got[i] + got[i + 1] - 1)) for s, i in at.items()}
        print(f"[luay] PhyloNet: {len(self.pairs)} distances + {2 * len(nets)} size probes "
              f"({len(nets)} networks) in {time.perf_counter() - t0:.1f} s", flush=True)

    def result(self, i: int) -> dict:
        pred, truth = self.pairs[i]
        xp, xt = self.size[pred], self.size[truth]
        return {"luay": self.luay[i],
                "mass_pred": None if xp is None else xp + self.n_leaves[pred],
                "mass_truth": None if xt is None else xt + self.n_leaves[truth],
                "max_exact": None if xp is None or xt is None else (xp + xt) // 2}


def fmt(x) -> str:
    """camus_a104's formatting: ``%g`` floats, empty for missing."""
    if missing(x):
        return ""
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


# ---------------------------------------------------------------------------------- hybrid ancestry

def hybrid_ancestry(net, drop: str = C.OUT) -> frozenset[str]:
    """Taxa below at least one hybrid: the union of the hybrids' hardwired clusters, as labels."""
    below = N.hardwired_clusters(net)
    return frozenset(net.labels[i] for v in below if net.is_hybrid(v) for i in below[v]) - {drop}


def hybrid_ancestry_text(enewick: str, drop: str = C.OUT) -> frozenset[str]:
    """:func:`hybrid_ancestry` off the string, with no parser -- the independent derivation.

    Every hybrid is written once with its body, ``(...)#Hk`` or a leaf ``label#Hk``, and elsewhere
    as a bare ``#Hk``.  A hybrid inside another's body is only a tag there, but its own body is
    counted in its own right, so the union of the bodies' leaves is the whole set.
    """
    out, opened = set(), []
    for i, ch in enumerate(enewick):
        if ch == "(":
            opened.append(i)
        elif ch == ")":
            j = opened.pop()
            if _BODY_TAG.match(enewick, i + 1):
                out.update(_LEAF.findall(enewick, j, i + 1))
    out.update(_LEAF_HYBRID.findall(enewick))
    return frozenset(out) - {drop}


def ha_metrics(pred: frozenset[str], true: frozenset[str]) -> dict:
    """The ``ha_*`` columns.  An empty ``pred`` has no FPR, so no balanced error either."""
    fp, fn = len(pred - true), len(true - pred)
    fpr, fnr = (fp / len(pred) if pred else None), fn / len(true)
    return {"ha_n_true": len(true), "ha_n_pred": len(pred), "ha_fp": fp, "ha_fn": fn,
            "ha_fpr": fpr, "ha_fnr": fnr,
            "ha_balanced_error": None if fpr is None else (fpr + fnr) / 2}


# ------------------------------------------------------------------------------------------ rows

def base_row(rec: dict, method: str, seed: int, decoder: str = "") -> dict:
    return {"method": method + C.decoder_suffix(decoder) if decoder else method, "_base": method,
            "decoder": decoder, "gene_trees": C.GENE_TREES, "n": C.N_TAXA,
            "network_idx": rec["rep"], "seed": seed, "k_true": rec["k"], "key": rec["key"],
            "cu": rec["cu"], "n_l2": rec["n_l2"], "n_l1": rec["n_l1"], "status": "ok"}


def model_row(rec: dict, label: str, seed: int, m: dict, decoder: str) -> dict:
    row = base_row(rec, label, seed, decoder)
    row.update(cluster_f1=m["f1"], cluster_f1_rooted=m["f1"], k_pred=int(m["k_pred"]),
               **{c: m[c] for c in COUNTS})
    if m.get("enewick") is None or int(m["k_pred"]) < 0:
        row["status"] = "no_network"          # realise() refused the selection; no luay exists
    else:
        row["_pred"] = C.strip_gamma(m["enewick"])
    return row


def sampling_problems(f: Path, cell: dict, *, gnn: bool) -> list[str]:
    """A GNN or greedy cell must have read this dataset's trees at its threshold (module docstring).

    Fields a cell lacks take the landed values: ``run_gnn_cpu.py`` and ``run_greedy.py`` write
    ``tau`` / ``tau_frac`` (and greedy ``m``) only off the landed (1000, 2).  ``tau_frac`` is
    compared as the decimal it names, so ``0.0020`` is ``0.002``.
    """
    have = (int(cell["n_loci"]), int(cell.get("m", 1000)), int(cell.get("tau", 2)))
    out = []
    if have != (C.M, C.M, C.TAU):
        out.append(f"{f}: (n_loci, m, tau) = {have}, this dataset is {(C.M, C.M, C.TAU)}")
    if gnn:
        frac = lambda x: None if x is None else Fraction(str(x))       # noqa: E731
        if frac(cell.get("tau_frac")) != frac(C.TAU_FRAC):
            out.append(f"{f}: tau_frac = {cell.get('tau_frac')!r}, this dataset is {C.TAU_FRAC!r}")
    return out


def cell_files(run: Path) -> list[Path]:
    return sorted((run / "cells").glob(f"{C.PHASE}_rep*_{C.ARM}.json"))


def gnn_rows(runs: list[Path], by_rep: dict) -> tuple[list[dict], list[str]]:
    rows, problems, seen = [], [], {}
    for run in runs:
        for f in cell_files(run):
            cell = json.loads(f.read_text())
            dec = C.cell_decoder(cell)
            rec = by_rep[int(cell["rep"])]
            if int(cell["k"]) != rec["k"]:
                problems.append(f"{f}: k = {cell['k']}, staged k = {rec['k']}")
            problems += sampling_problems(f, cell, gnn=True)
            for m in cell["models"]:
                label = C.GNN_LABELS.get(m["tag"])
                if label is None:
                    problems.append(f"{f}: checkpoint tag {m['tag']!r} has no label")
                    continue
                ident = (dec, m["tag"], int(m["seed"]), rec["rep"])
                if ident in seen:
                    problems.append(f"{ident} is in both {seen[ident]} and {run}")
                    continue
                seen[ident] = str(run)
                rows.append(model_row(rec, label, int(m["seed"]), m, dec))
    present = {(d, t) for d, t, _s, _r in seen}
    if C.GNN_REQUIRED is None:          # v1: every shipped checkpoint, at level 1
        need = {("level1", t) for t in REQUIRED_TAGS} | present
    else:                               # gt5k / gt10k: what those runs were scoped to (common.py)
        tags, dec_required = C.GNN_REQUIRED
        need = {(dec_required, t) for t in tags} | present
    for dec, tag in sorted(need, key=lambda x: (variant_rank(x[0]), x[1])):
        seeds = REQUIRED_SEEDS.get(tag, SEEDS)
        gaps = [(s, r) for s in seeds for r in by_rep if (dec, tag, s, r) not in seen]
        if gaps:
            problems.append(f"GNN {tag} ({dec}): {len(gaps)} of {len(seeds) * len(by_rep)} "
                            f"(seed, rep) missing, e.g. {gaps[:3]}")
    return rows, problems


def greedy_rows(runs: list[Path], by_rep: dict) -> tuple[list[dict], list[str]]:
    rows, got, problems = [], {}, []
    for run in runs:
        for f in cell_files(run):
            cell = json.loads(f.read_text())
            dec = C.cell_decoder(cell)
            rec = by_rep[int(cell["rep"])]
            if (dec, rec["rep"]) in got:
                problems.append(f"greedy {dec} rep {rec['rep']} is in both {got[(dec, rec['rep'])]} "
                                f"and {run}")
                continue
            problems += sampling_problems(f, cell, gnn=False)
            got[(dec, rec["rep"])] = str(run)
            rows.append(model_row(rec, "greedy", 0, cell["models"][0], dec))
    for dec in sorted(set(C.GREEDY_REQUIRED) | {d for d, _r in got}, key=variant_rank):
        gaps = sorted(r for r in by_rep if (dec, r) not in got)
        if gaps:
            problems.append(f"greedy ({dec}): {len(gaps)} reps missing, e.g. {gaps[:5]}")
    return rows, problems


def tob_rows(stage_cells: list[dict], truths: dict) -> list[dict]:
    rows = []
    for rec in stage_cells:
        for v in C.VARIANTS:
            rooted = (C.TOB / f"{rec['key']}.{v}.rooted.nwk").read_text().strip()
            net = N.from_enewick(rooted)
            counts = CN.cluster_counts(CN.restrict_clusters(net, CN.ingroup_of(net)),
                                       truths[rec["rep"]]["clusters"])
            row = base_row(rec, C.TOB_LABELS[v], 0)
            row.update(cluster_f1=counts["f1"], cluster_f1_rooted=counts["f1"], k_pred=0,
                       n_blobs_tob=rec["n_blobs"][v], **{c: counts[c] for c in COUNTS})
            row["_pred"] = pruned(rooted)
            rows.append(row)
    return rows


def resolver_rows(run: Path, stage_cells: list[dict]) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for rec in stage_cells:
        for v in C.RESOLVER_VARIANTS:
            f = run / "cells" / f"{rec['key']}.{v}.json"
            if not f.exists():
                problems.append(f"resolvers: no {f.name}")
                continue
            cell = json.loads(f.read_text())
            for name in C.RESOLVERS:
                m = cell["methods"].get(name)
                if m is None:
                    problems.append(f"resolvers: {f.name} has no {name}")
                    continue
                row = base_row(rec, C.RESOLVER_LABELS[(name, v)], 0)
                row["n_blobs_tob"] = cell["tob"]["n_blobs"]
                if "error" in m:
                    row["status"] = f"error: {m['error']}"
                elif m["out_rootable"]:
                    row.update(cluster_f1=m["f1"], cluster_f1_rooted=m["f1"],
                               k_pred=int(m["k_pred"]), **{c: m[c] for c in COUNTS})
                    row["_pred"] = pruned(m["enewick_rooted"])
                else:
                    row.update(cluster_f1=m["f1_as_emitted"], k_pred=int(m["k_pred"]),
                               status="out_not_rootable")
                    if m.get("enewick"):          # hybrid ancestry only: it needs no rooting
                        row["_emitted"] = C.strip_gamma(m["enewick"])
                rows.append(row)
    return rows, problems


def attach_hybrid_ancestry(rows: list[dict], truths: dict) -> tuple[dict, list[str]]:
    """Fill the ``ha_*`` columns (module docstring); return the cross-check counts and problems.

    Runs before :func:`attach_luay`, which pops ``_pred``.  A row is read off ``_pred``, the
    network its luay compares, or for an ``out_not_rootable`` resolver row off ``_emitted``.
    """
    stats = Counter(networks=len(truths), agree=len(truths), rows=0)   # truths: checked in build()
    problems = []
    for row in rows:
        emitted = row.pop("_emitted", None)
        src = row.get("_pred") or emitted
        if src is None:
            continue
        truth = truths[row["network_idx"]]
        where = f"hybrid ancestry: {row['method']} {row['key']} seed {row['seed']}"
        try:
            net = N.from_enewick(src)
            got, text = hybrid_ancestry(net), hybrid_ancestry_text(src)
        except (ValueError, IndexError) as e:
            problems.append(f"{where}: {e}")
            continue
        stats["networks"] += 1
        if got != text:
            problems.append(f"{where}: the two derivations differ on {sorted(got ^ text)[:5]}")
            continue
        stats["agree"] += 1
        if set(net.labels) - {C.OUT} != truth["taxa"]:
            problems.append(f"{where}: not the truth's taxon set")
            continue
        row.update(ha_metrics(got, truth["ha"]))
        stats["rows"] += 1
    return dict(stats), problems


def attach_luay(rows: list[dict], truths: dict, luay: PhyloNetLuay) -> None:
    for row in rows:
        pred = row.pop("_pred", None)
        if pred is not None:
            row["_i"] = luay.add(pred, truths[row["network_idx"]]["enewick"])
    luay.resolve(C.RUNS / "phylonet")
    for row in rows:
        i = row.pop("_i", None)
        if i is None:
            continue
        r = luay.result(i)
        row["mass_pred"], row["mass_truth"] = r["mass_pred"], r["mass_truth"]
        if r["luay"] is None:
            row["status"] = "phylonet_failed"
            continue
        row["luay"] = r["luay"]
        row["luay_norm"] = r["luay"] / closed_form_max(C.N_TAXA, row["k_true"], row["k_pred"])
        if r["max_exact"]:
            row["luay_norm_exact"] = r["luay"] / r["max_exact"]


def per_method(rows: list[dict], order: dict) -> dict:
    def med(col, rs):
        xs = [r[col] for r in rs if not missing(r.get(col))]
        return statistics.median(xs) if xs else None
    labels = sorted({r["method"]: r for r in rows}.values(), key=lambda r: row_order(r, order))
    out = {}
    for first in labels:
        rs = [r for r in rows if r["method"] == first["method"]]
        out[first["method"]] = {
            "rows": len(rs), "decoder": first["decoder"],
            "status": dict(Counter(r["status"] for r in rs)),
            **{f"median_{c}": med(c, rs) for c in
               ("cluster_f1", "cluster_f1_rooted", "luay", "luay_norm", "luay_norm_exact",
                "k_pred", "k_true", "ha_fpr", "ha_fnr", "ha_balanced_error")}}
    return out


def build(args, jar: Path) -> int:
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run l2reg_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    if stage["dataset"] != str(C.PKG):
        raise SystemExit(f"{C.STAGE} stages {stage['dataset']}, not {C.PKG}: re-run stage.py")
    loci = sorted({int(r["n_loci"]) for r in C.manifest()})
    if loci != [C.M]:
        raise SystemExit(f"{C.PKG / 'MANIFEST.tsv'} has n_loci {loci}; common.M is {C.M}")
    print(f"[csv] {C.PKG.name}: m = {C.M:,} trees, tau = {C.TAU}"
          f"{'' if C.TAU_FRAC is None else f' (tau-frac {C.TAU_FRAC})'} -> {args.out_csv}",
          flush=True)
    cells = stage["cells"]
    by_rep = {c["rep"]: c for c in cells}
    truths = {}
    for rec in cells:
        t = C.truth_of(rec["rep"])
        if t.n_retics != rec["k"]:
            raise SystemExit(f"{rec['key']}: truth n_retics {t.n_retics} != staged k {rec['k']}")
        enewick, ha = C.strip_gamma(N.to_enewick(t)), hybrid_ancestry(t)
        if not ha or ha != hybrid_ancestry_text(enewick):             # k >= 3: never empty
            raise SystemExit(f"{rec['key']}: the truth's hybrid ancestry is empty, or its two "
                             f"derivations differ")
        truths[rec["rep"]] = {"enewick": enewick, "clusters": CN.label_clusters(t),
                              "taxa": set(t.labels), "ha": ha}

    def runs_like(pattern: str, exclude: str = "") -> list[Path]:
        return sorted(p for p in C.RUNS.glob(pattern) if (p / "cells").is_dir()
                      and not (exclude and p.name.startswith(exclude)))

    gnn_runs = [Path(p) for p in args.gnn_runs] if args.gnn_runs else runs_like("gnn*")
    greedy_runs = ([Path(p) for p in args.greedy_runs] if args.greedy_runs
                   else runs_like("greedy*", exclude="greedy_camus"))
    rows, problems = gnn_rows(gnn_runs, by_rep)
    g_rows, g_prob = greedy_rows(greedy_runs, by_rep)
    r_rows, r_prob = resolver_rows(Path(args.resolver_run), cells)
    rows += g_rows + tob_rows(cells, truths) + r_rows
    problems += g_prob + r_prob
    if problems:
        print("[csv] incomplete:\n" + "\n".join(f"  {p}" for p in problems))
        if not args.allow_partial:
            print("[csv] nothing written; --allow-partial writes what exists")
            return 1

    ha_stats, ha_problems = attach_hybrid_ancestry(rows, truths)
    problems += ha_problems
    print(f"[ha] hybrid ancestry: the two derivations agree on {ha_stats['agree']} of "
          f"{ha_stats['networks']} networks; {ha_stats['rows']} rows scored", flush=True)
    luay = PhyloNetLuay(jar)
    attach_luay(rows, truths, luay)
    over = [r for r in rows if not missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    if over:                                     # impossible by construction; say so if it happens
        problems.append(f"{len(over)} rows with luay_norm_exact > 1, e.g. "
                        f"{[(r['method'], r['key'], r['seed']) for r in over[:3]]}")
    offsets = sorted({r["mass_truth"] - (2 * C.N_TAXA + 3 * r["k_true"] - 1)
                      for r in rows if not missing(r.get("mass_truth"))})

    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (row_order(r, order), r["network_idx"], r["seed"]))
    out_csv, out_json = Path(args.out_csv), Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")          # camus_a104's line ending, not csv's \r\n
        w.writerow(CAMUS_COLUMNS + EXTRA_COLUMNS)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in CAMUS_COLUMNS + EXTRA_COLUMNS])

    parity_p = C.RUNS / "parity.json"
    summary = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "sampling": {"m": C.M, "tau": C.TAU, "tau_frac": C.TAU_FRAC,
                     "resolver_variants": list(C.RESOLVER_VARIANTS),
                     "gnn_required": C.GNN_REQUIRED, "greedy_required": list(C.GREEDY_REQUIRED)},
        "stage": {"built": stage["built"],
                  "dataset_manifest_sha256": stage["dataset_manifest_sha256"]},
        "rows": len(rows), "complete": not problems, "problems": problems,
        "decoders": sorted({r["decoder"] for r in rows if r["decoder"]}, key=variant_rank),
        "luay": {"engine": "PhyloNet CMPNETS -m luay", "jar": str(jar),
                 "sizes": "two star probes per network (module docstring)",
                 "truth_mass_offset_vs_2n_plus_3k_minus_1": offsets},
        "hybrid_ancestry": {
            "source": "Dai & Molloy, bioRxiv 10.64898/2026.06.06.730607, Fig. 3",
            "definition": "P, T = the ingroup taxa below >= 1 hybrid of the prediction, of the "
                          "truth; ha_fpr = |P - T| / |P|, ha_fnr = |T - P| / |T|, "
                          "ha_balanced_error = their mean",
            "empty_prediction": "ha_fpr and ha_balanced_error empty (0/0)",
            "out_not_rootable": "read off the network as emitted: hybrid ancestry is "
                                "rooting-invariant",
            "cross_check": "hardwired clusters of the parsed Network vs an eNewick string scan",
            **ha_stats},
        "runs": {"gnn": [str(p) for p in gnn_runs], "greedy": [str(p) for p in greedy_runs],
                 "resolvers": str(args.resolver_run)},
        "labels": {"gnn": C.GNN_LABELS, "tob": C.TOB_LABELS,
                   "resolvers": {f"{a}/{b}": v for (a, b), v in C.RESOLVER_LABELS.items()}},
        "per_method": per_method(rows, order),
        "parity": json.loads(parity_p.read_text()) if parity_p.exists() else None,
    }
    out_json.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"[csv] {len(rows)} rows -> {out_csv}\n[csv] summary -> {out_json}\n"
          f"[csv] truth mass - (2n + 3k - 1): {offsets}; rows with luay_norm_exact > 1: {len(over)}")
    for label, s in summary["per_method"].items():
        f1, ln, ber = (s[f"median_{c}"] for c in ("cluster_f1", "luay_norm_exact",
                                                   "ha_balanced_error"))
        print(f"  {label:32s} {s['rows']:3d} rows | F1 med {'' if f1 is None else f'{f1:.4f}':6s}"
              f" | luay_norm_exact med {'' if ln is None else f'{ln:.4f}':6s}"
              f" | ha BER med {'' if ber is None else f'{ber:.4f}':6s} | "
              f"k_pred med {s['median_k_pred']} (k {s['median_k_true']}) | {s['status']}")
    return 0 if not problems else 2


# ---------------------------------------------------------------------------------------- parity

def landed(name: str) -> list[dict]:
    return pq.read_table(C.LANDED / name).to_pylist()


def allowed(it: dict) -> set[str]:
    """The fields a known, explained miss may differ in (module docstring); empty for any other."""
    out = set()
    if (it["phase"], it["rep"]) == ("cam150", 4):
        out.add("mass_truth")                          # the size-2-cycle truth
    if it["what"] == "tob_tqmc" and (it["phase"], it["rep"]) == ("cam200", 15):
        out |= {"luay", "mass_pred"}                   # a different tree of blobs on the Mac
    return out


def parity(args, jar: Path) -> int:
    t0 = time.perf_counter()
    tcache: dict[tuple[str, int], tuple[str, object]] = {}

    def truth(phase: str, rep: int):
        if (phase, rep) not in tcache:
            t = C.truth_of(rep, inputs=C.GNN / "inputs", phase=phase)
            tcache[(phase, rep)] = (C.strip_gamma(N.to_enewick(t)), t)
        return tcache[(phase, rep)]

    luay = PhyloNetLuay(jar)
    items: list[dict] = []          # one per pair: which gate, and what the landed row says

    def add(gate, what, phase, rep, pred, landed_luay, mass_pred, mass_truth, **extra):
        items.append({"gate": gate, "what": what, "phase": phase, "rep": int(rep),
                      "i": luay.add(pred, truth(phase, int(rep))[0]), "luay": landed_luay,
                      "mass_pred": mass_pred, "mass_truth": mass_truth, **extra})

    # (a) resolver rows: the landed strings themselves
    for what in ("brooqs_tqmctob", "netcs_tqmctob", "brooqs_truetob", "netcs_truetob"):
        for r in landed(f"rl_camus_{what}_tau2.parquet"):
            if missing(r["luay"]):
                continue                    # landed unrootable: nothing to compare
            raw = r["enewick_raw"]
            add("a", what, r["phase"], r["rep"], pruned(raw) if C.OUT in raw else C.strip_gamma(raw),
                r["luay"], r["mass_pred"], r["mass_truth"])
    # (a) TOB-QMC: blob_cpu_v1's Mac trees of blobs against the landed tob_tqmc rows
    tq = {(r["phase"], r["rep"]): r for r in landed("rl_camus_tob_tqmc_tau2.parquet")}
    for p in sorted((MAC_BLOB_RUN / "work").glob("*/tob.rooted.nwk")):
        m = re.match(r"(cam\d+)_rep(\d+)_", p.parent.name)
        r = tq[(m.group(1), int(m.group(2)))]
        add("a", "tob_tqmc", m.group(1), m.group(2), pruned(p.read_text().strip()),
            r["luay"], r["mass_pred"], r["mass_truth"])
    # (a) true TOB: tree_of_blobs on the camus truths, which are level-1, so it is valid here
    for r in landed("rl_camus_tob_true_tau2.parquet"):
        add("a", "tob_true", r["phase"], r["rep"], CN.tree_of_blobs(truth(r["phase"], r["rep"])[1]),
            r["luay"], r["mass_pred"], r["mass_truth"])
    # (b) the Mac GNN networks against the landed GNN rows
    gnn_landed: dict[str, dict] = {}
    for f in sorted((Path(args.gnn_camus_run) / "cells").glob("*.json")):
        cell = json.loads(f.read_text())
        for mdl in cell["models"]:
            if not mdl.get("enewick"):
                continue
            name = (f"rl_gnn_eval_{cell['phase']}{LANDED_ARM[cell['arm']]}_tau2_segment_"
                    f"s{mdl['seed']}{mdl['tag']}.parquet")
            if name not in gnn_landed:
                gnn_landed[name] = ({r["key"]: r for r in landed(name)}
                                    if (C.LANDED / name).exists() else {})
            r = gnn_landed[name].get(cell["key"])
            if r is None or missing(r.get("luay")):
                continue
            add("b", f"{mdl['tag']}/s{mdl['seed']}", cell["phase"], cell["rep"],
                C.strip_gamma(mdl["enewick"]), r["luay"], r["mass_pred"], r["mass_truth"])
    # (c) greedy on the camus cells -- against the landed rows, so only the formal decoder
    greedy_cells = sorted((Path(args.greedy_camus_run) / "cells").glob("*.json"))
    ref = {(r["phase"], int(r["rep"]), r["arm"]): r for r in C.read_tsv(C.GNN / "reference.tsv")
           if r["tag"] == "_mix" and r["seed"] == "0"}
    for f in greedy_cells:
        cell = json.loads(f.read_text())
        if C.cell_decoder(cell) != "level1":
            raise SystemExit(f"{f} was decoded as {C.cell_decoder(cell)}; parity (c) compares "
                             f"against the landed level-1 greedy -- point --greedy-camus-run at a "
                             f"formal-decoder run")
        m = cell["models"][0]
        r = {x["key"]: x for x in landed(f"rl_gnn_eval_{cell['phase']}{LANDED_ARM[cell['arm']]}"
                                         f"_tau2_segment_s0_mix.parquet")}[cell["key"]]
        counts = {k: (m[k], r[f"{k}_greedy"]) for k in ("tp", "fp", "k_pred")}
        counts["f1"] = (round(m["f1"], 12),
                        round(float(ref[(cell["phase"], int(cell["rep"]), cell["arm"])]
                                    ["f1_greedy"]), 12))
        if not m.get("enewick"):
            items.append({"gate": "c", "what": "greedy", "phase": cell["phase"],
                          "rep": int(cell["rep"]), "i": None, "counts": counts})
            continue
        add("c", "greedy", cell["phase"], cell["rep"], C.strip_gamma(m["enewick"]),
            r["luay_greedy"], r["mass_greedy"], r["mass_truth"], counts=counts)

    luay.resolve(C.RUNS / "phylonet_parity")
    report: dict = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jar": str(jar)}
    ok = True

    def diff(it) -> set[str]:
        """The landed fields this pair fails to reproduce."""
        if it["i"] is None:
            return set(_FIELDS)
        res = luay.result(it["i"])
        return {f for f in _FIELDS if missing(it[f]) or res[f] is None or int(res[f]) != int(it[f])}

    def gate(name: str, its: list[dict], label: str) -> bool:
        misses = [(it, d) for it in its if (d := diff(it))]
        known = [(it, d) for it, d in misses if d <= allowed(it)]
        bad = [(it, d) for it, d in misses if not d <= allowed(it)]
        by_field = Counter(f for _it, d in misses for f in d)
        report[name] = {"pairs": len(its), "exact": len(its) - len(misses),
                        "known": [{**{k: it[k] for k in ("what", "phase", "rep")},
                                   "fields": sorted(d)} for it, d in known],
                        "unexplained": [{**{k: it[k] for k in ("what", "phase", "rep")},
                                         "fields": sorted(d)} for it, d in bad[:50]],
                        "misses_by_field": dict(by_field)}
        print(f"[parity] ({name}) {label}: {len(its) - len(misses)} of {len(its)} exact on luay + "
              f"both masses; {len(known)} known misses {sorted({(it['phase'], it['rep']) for it, _d in known})}"
              f", {len(bad)} unexplained -> {'PASS' if not bad else 'FAIL'}")
        for it, d in bad[:10]:
            print(f"      {it['what']} {it['phase']} rep{it['rep']:02d}: {sorted(d)}")
        return not bad

    ok &= gate("a", [it for it in items if it["gate"] == "a"], "resolver/TOB rows")
    g = [it for it in items if it["gate"] == "b"]
    if g:
        ok &= gate("b", g, "Mac GNN networks")
    else:
        report["b"] = None
        print(f"[parity] (b) skipped: no Mac GNN cells under {args.gnn_camus_run}")
    c_items = [it for it in items if it["gate"] == "c"]
    if c_items:
        c_ok = gate("c", c_items, "greedy networks")
        bad_counts = [(it["phase"], it["rep"], {k: v for k, v in it["counts"].items() if v[0] != v[1]})
                      for it in c_items if any(a != b for a, b in it["counts"].values())]
        complete = len(c_items) == 100
        report["c"]["counts_bad"] = bad_counts
        report["c"]["cells"] = len(c_items)
        c_ok &= not bad_counts and (complete or args.allow_partial)
        ok &= c_ok
        print(f"[parity] (c) greedy tp/fp/k_pred/F1 exact on {len(c_items) - len(bad_counts)} of "
              f"{len(c_items)} camus cells{'' if complete else ' (of 100 expected)'} -> "
              f"{('PASS' if complete else 'PASS (partial)') if c_ok else 'FAIL'}")
        if not c_ok and not bad_counts:
            print("      the landed realise->luay path differs from this one: ask for "
                  "scripts/w10_gnn_eval.py and scripts/w9_swap_eval.py from the Linux box")
    else:
        report["c"] = None
        print(f"[parity] (c) skipped: no {args.greedy_camus_run}/cells -- run run_greedy.py --camus")

    report["pass"] = bool(ok)
    out = C.RUNS / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"[parity] {'PASS' if ok else 'FAIL'} -> {out}  ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true", help="the gates against landed camus rows")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET") or DEFAULT_JAR),
                    help="PhyloNet.jar (Java-8 classes); default $OPC_PHYLONET or %(default)s")
    ap.add_argument("--gnn-runs", nargs="*", default=None,
                    help="GNN run directories; default every runs/gnn*/ with cells")
    ap.add_argument("--greedy-runs", nargs="*", default=None,
                    help="greedy run directories; default every runs/greedy*/ with cells, "
                         "greedy_camus* excluded")
    ap.add_argument("--resolver-run", type=Path, default=C.RUNS / "resolvers")
    ap.add_argument("--gnn-camus-run", type=Path, default=MAC_GNN_RUN,
                    help="parity (b): the Mac GNN run on the camus cells")
    ap.add_argument("--greedy-camus-run", type=Path, default=C.RUNS / "greedy_camus")
    # Named after the dataset, so v1 keeps its landed names and gt5k / gt10k get their own.
    ap.add_argument("--out-csv", type=Path, default=C.LANDED / f"{C.PKG.name}_per_network.csv")
    ap.add_argument("--out-json", type=Path, default=C.LANDED / f"{C.PKG.name}.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    jar = args.phylonet.expanduser().resolve()
    if not jar.exists():
        raise SystemExit(f"no PhyloNet jar at {jar}: pass --phylonet or set OPC_PHYLONET")
    return parity(args, jar) if args.parity else build(args, jar)


if __name__ == "__main__":
    raise SystemExit(main())
