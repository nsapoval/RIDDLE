#!/usr/bin/env python3
"""The subsample study's per-network CSV, one per dataset, with the full-data rows beside it.

Rows
----
One per (draw, cell, method): ``draw`` is ``full`` or ``0``..``4``, and ``method`` is one of
``ours/l2mix_pair/selF1``, ``ours/l2mix_pair/selHA`` and ``greedy``, each with the landed decoder
suffix, e.g. ``ours/l2mix_pair/selF1 [relaxed L2]``.  The labels are the landed CSVs' own.

* **Subsample rows** come from ``runs/<ds>/d<k>/{gnn_t1_l2mix_pair, greedy}{,_relaxedL<L>}/``.
* **``full`` rows** come from the landed 1,000-tree runs (``common.DATASETS[..].full``).  They are
  re-scored here by the same code as the subsample rows, so a paired difference is between two
  numbers computed one way.  The one full-data run no benchmark had -- camus's relaxed greedy --
  is ``runs/camus/full/greedy_relaxedL2`` (``run_sub.sh camus full``).

Columns
-------
* **Identity:** ``dataset, draw, sub_seed, m, n_loci_source, phase, rep, arm, key, n, n_taxa,
  k_true``, the dataset's stratum (camus none -- ``phase`` is one; ``cu`` on l2reg/l3reg;
  ``level, ils`` on netcs), then ``decoder, method, tag``.
  * ``n`` is the phase's nominal size.
  * ``n_taxa`` is the truth's ingroup count; on netcs that is n..n+5.
  * ``sub_seed`` is ``MANIFEST.tsv``'s seed, empty on ``full``.
* **Scores:** ``n_clusters`` (|C| at tau = 2), ``n_selected, tp, fp, fn, precision, recall,
  cluster_f1, k_pred, luay, mass_pred, mass_truth, luay_norm, luay_norm_exact``, hybrid
  ancestry's ``ha_*``, ``prep_wall, model_wall`` (seconds), ``status``.
  * ``prep_wall`` is the shared stages: root, clusters, candidates, overlap, tree statistics and
    features for the GNN; greedy's ``prep``.
  * ``model_wall`` is forward + decode + network for the GNN, greedy's ``decode``.
* **``landed``** (``full`` rows only): ``match`` or ``MISMATCH`` against the landed CSV, empty where
  it holds no such row.

Scoring: the landed definitions
-------------------------------
* **Cluster F1** is the driver's own ``Instance.cluster_counts`` on the selection -- the landed
  ``f1_selected`` -- for the GNN and greedy alike.
* **luay and both masses** are PhyloNet ``CMPNETS -m luay`` only, never ``opc.mpl.netdist``.
  * A network's size comes from two star probes: ``X = luay(N, star) + luay(N, (star)) - 1``,
    ``mass = X + n``.
  * ``luay_norm`` is camus_a104's closed form on ``n_taxa``.
  * ``luay_norm_exact = luay / floor((X_pred + X_truth) / 2)``, at most 1 by construction.
  * The truth is ``strip_gamma(to_enewick(camus.truth_of(cell)))``, except on netcs, where it is the
    corpus string ``true_net.phylonet.nwk``, as ``netcs_eval/make_csv.py`` has it.
* **Hybrid ancestry** follows Dai & Molloy (bioRxiv 10.64898/2026.06.06.730607, Fig. 3).
  * P and T are the ingroup taxa below a hybrid.
  * ``ha_fpr = |P - T| / |P|``, ``ha_fnr = |T - P| / |T|``, and the balanced error is their mean.
  * An empty P leaves the FPR and the balanced error empty.
  * It is derived twice, from the parsed network and from a scan of the string, and the two must
    agree.
* These pieces are **copied, not imported**: the cached, parallel ``PhyloNetLuay`` from
  ``netcs_eval/make_csv.py``, and the hybrid-ancestry helpers from ``l2reg_eval/make_csv.py``.
  Each of those scripts imports a module named ``common``, which would shadow this package's own.
  The parity gate below is what licenses the copy.
* **The luay cache** is ``runs/phylonet/cache.jsonl``, keyed by the SHA-256 of the (pred, truth)
  string pair.  netcs also reads ``netcs_eval/runs/phylonet/cache.jsonl`` read-only
  (``common.SEED_CACHES``): the same key scheme, so its landed networks are not re-measured.

Checks
------
**Completeness** (nothing is written unless ``--allow-partial``):
* a cell JSON is missing from a run;
* a model is missing from a cell.

**Integrity** (the CSV is written, and the exit code is 2):
* a subsample cell whose ``m`` or ``n_loci`` is not 500, i.e. a forgotten ``--m-from-trees`` or
  the wrong ``--inputs``;
* a staged subsample whose SHA-256 no longer matches ``MANIFEST.tsv``;
* a full-data cell whose ``n_loci`` is not the source's;
* a cell decoded by another decoder, or with another ``k``, or naming another tree file;
* the two hybrid-ancestry derivations disagree;
* ``luay_norm_exact > 1``.

**Parity:** every ``full`` row must reproduce the landed CSV's ``cluster_f1, tp, fp, fn,
precision, recall, k_pred, luay, luay_norm, luay_norm_exact, mass_pred, mass_truth`` and ``ha_*``,
compared as the landed file prints them.
* A field the landed row leaves empty is not compared. camus's greedy rows come from the
  2026-09 base file, which carries only camus_a104's eleven columns.
* ``ALLOWED`` lists the known, explained misses, each with exactly the fields it explains.

Outputs
-------
``results/<ds>_sub500.csv``, ``results/<ds>_sub500.json`` (runs, counts, parity, problems,
summary) and ``results/<ds>_sub500_summary.txt``.  The summary is per method, subsample against
full:
* the full-data median;
* the pooled subsample median and the per-draw medians;
* the **median paired delta**: per cell, the mean over its draws minus its full value;
* the fraction of cells that got worse;
* the **mean within-cell SD across draws**;
* the median |C| ratio of sub to full;
* then Delta F1 by stratum.

camus's main table is the headline 80 ``cam_g500`` cells; cam25 (``cam_iq500``) shows up only in
the per-phase block.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/make_sub_csv.py --dataset l2reg
    python subsample_eval/make_sub_csv.py --dataset netcs --jobs 4
    python subsample_eval/make_sub_csv.py --dataset l2reg --draws full          # parity alone
    python subsample_eval/make_sub_csv.py --dataset camus --draws full,0 --phases cam50 --reps 0 \\
        --runs-root /tmp/smoke/runs --allow-partial --out-dir /tmp/smoke/results   # a smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common as C

from opc.mpl import phylonet as PN  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

GNN_TAGS = ("_l2mix_pair", "_l2mix_pair:best_ha")
SHARED = ("root", "clusters", "candidates", "overlap", "tree_stats", "features")   # run_gnn_cpu.py
PER_MODEL = ("forward", "decode", "network")
HA_COLUMNS = ("ha_n_true", "ha_n_pred", "ha_fp", "ha_fn", "ha_fpr", "ha_fnr", "ha_balanced_error")
STRATA = {"camus": (), "l2reg": ("cu",), "l3reg": ("cu",), "netcs": ("level", "ils")}
SCORE_COLUMNS = ("n_clusters", "n_selected", "tp", "fp", "fn", "precision", "recall", "cluster_f1",
                 "k_pred", "luay", "mass_pred", "mass_truth", "luay_norm", "luay_norm_exact",
                 *HA_COLUMNS, "prep_wall", "model_wall", "status", "landed")
PARITY_FIELDS = ("cluster_f1", "tp", "fp", "fn", "precision", "recall", "k_pred", "luay",
                 "luay_norm", "luay_norm_exact", "mass_pred", "mass_truth", *HA_COLUMNS)
#: (dataset, phase, rep) -> the fields a known, explained miss may differ in.  Empty for anything else.
#: cam150 rep04: its truth has a size-2 cycle, which the jar sizes differently from the landed
#: (Linux) number -- ``l2reg_eval/make_csv.py``'s parity notes -- and luay_norm_exact's denominator
#: is that size.  Measured 2026-09-18: the landed greedy row (the Linux base file) has
#: luay_norm_exact 0.349693 where the jar gives 0.360759, luay equal; the four pair rows, sized by
#: the jar on the Mac, match in every field.
ALLOWED = {("camus", "cam150", 4): {"mass_truth", "luay_norm_exact"}}
#: (metric, "higher" | "lower" is better)
METRICS = (("cluster_f1", "higher"), ("luay_norm_exact", "lower"), ("ha_balanced_error", "lower"))

# --------------------------------------------------------------------- copied scoring (see docstring)

#: A leaf label: a name right after "(" or ",".  Never a "#H" reference or an internal label.
_LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")
#: What closes a hybrid's body: an optional internal label, then the ``#Hk`` tag.
_BODY_TAG = re.compile(r"[^(),:;#\s\[\]]*#[A-Za-z]*\d+")
#: A leaf that is itself a hybrid: ``label#Hk``.
_LEAF_HYBRID = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)#[A-Za-z]*\d+")
_GAMMA = re.compile(r"::[0-9.eE+-]+")


def strip_gamma(enewick: str) -> str:
    return _GAMMA.sub("", enewick)


def missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt(x) -> str:
    """camus_a104's formatting: ``%g`` floats, empty for missing."""
    if missing(x):
        return ""
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def leaf_set(enewick: str) -> tuple[str, ...]:
    return tuple(sorted(set(_LEAF.findall(enewick))))


def closed_form_max(n: int, k1: int, k2: int) -> int:
    """``n - 1 + floor(3 (k1 + k2) / 2)``: camus_a104's ``luay_norm`` denominator."""
    return max(0, n - 1 + (3 * (k1 + k2)) // 2)


def pair_key(a: str, b: str) -> str:
    return hashlib.sha256(f"{a}\n{b}".encode()).hexdigest()[:40]


class PhyloNetLuay:
    """``netcs_eval/make_csv.py``'s cached, parallel jar runner, plus read-only seed caches."""

    def __init__(self, jar: Path, cache: Path | None = None, seeds=(), jobs: int = 1,
                 chunk: int = 2000):
        self.jar, self.cache, self.seeds = jar, cache, tuple(seeds)
        self.jobs, self.chunk = max(1, jobs), max(1, chunk)
        self.pairs: list[tuple[str, str]] = []

    def add(self, pred: str, truth: str) -> int:
        self.pairs.append((pred, truth))
        return len(self.pairs) - 1

    def _load(self) -> tuple[dict[str, int], int]:
        known: dict[str, int] = {}
        for path in (*self.seeds, self.cache):
            if path is None or not Path(path).exists():
                continue
            with open(path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:          # a torn last line from an interrupted run
                        continue
                    known[r["k"]] = int(r["luay"])
        return known, len(known)

    def resolve(self, workdir: Path) -> None:
        PN.PHYLONET = Path(self.jar)                  # read at call time by ``available()``
        nets = list(dict.fromkeys(s for pair in self.pairs for s in pair))
        self.n_leaves = {s: len(leaf_set(s)) for s in nets}
        cmds, at = list(self.pairs), {}
        for s in nets:
            star = "(" + ",".join(leaf_set(s)) + ")"
            at[s] = len(cmds)
            cmds += [(s, star + ";"), (s, f"({star});")]
        keys = [pair_key(a, b) for a, b in cmds]
        by_key = dict(zip(keys, cmds))
        known, _ = self._load()
        todo = [k for k in dict.fromkeys(keys) if k not in known]
        if todo and not PN.available():
            raise SystemExit(f"PhyloNet not usable: jar {PN.PHYLONET}, java {PN.JAVA}")
        chunks = [todo[i:i + self.chunk] for i in range(0, len(todo), self.chunk)]
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"[luay] {len(self.pairs)} distances + {2 * len(nets)} size probes ({len(nets)} "
              f"networks): {len(set(keys)) - len(todo)} cached, {len(todo)} to measure in "
              f"{len(chunks)} chunk(s) x {self.jobs} JVM(s)", flush=True)

        def one(i: int, ks: list[str]):
            wd = workdir / f"chunk{i:05d}"
            got = PN.cmpnets_many([by_key[k] for k in ks], methods=("luay",), workdir=wd,
                                  timeout=6 * 3600.0)
            shutil.rmtree(wd, ignore_errors=True)
            return ks, [r["luay"] for r in got]

        n_failed = 0
        if chunks:
            fh = open(self.cache, "a") if self.cache is not None else None
            try:
                with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                    futs = [ex.submit(one, i, ks) for i, ks in enumerate(chunks)]
                    for done, fut in enumerate(as_completed(futs), 1):
                        ks, vals = fut.result()
                        for k, v in zip(ks, vals):
                            if missing(v):
                                n_failed += 1
                                continue
                            known[k] = int(v)
                            if fh is not None:
                                fh.write(json.dumps({"k": k, "luay": int(v)}) + "\n")
                        if fh is not None:
                            fh.flush()
                        if done % max(1, len(chunks) // 20) == 0 or done == len(chunks):
                            print(f"[luay]   chunk {done}/{len(chunks)} | "
                                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            finally:
                if fh is not None:
                    fh.close()
        got = [known.get(k) for k in keys]
        self.luay = got[:len(self.pairs)]
        self.size = {s: (None if got[i] is None or got[i + 1] is None else got[i] + got[i + 1] - 1)
                     for s, i in at.items()}
        self.measured, self.failed = len(todo) - n_failed, n_failed
        print(f"[luay] PhyloNet done in {time.perf_counter() - t0:.1f} s; {n_failed} commands "
              f"failed (not cached)", flush=True)

    def result(self, i: int) -> dict:
        pred, truth = self.pairs[i]
        xp, xt = self.size[pred], self.size[truth]
        return {"luay": self.luay[i],
                "mass_pred": None if xp is None else xp + self.n_leaves[pred],
                "mass_truth": None if xt is None else xt + self.n_leaves[truth],
                "max_exact": None if xp is None or xt is None else (xp + xt) // 2}


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
    fpr = fp / len(pred) if pred else None
    fnr = fn / len(true) if true else None
    return {"ha_n_true": len(true), "ha_n_pred": len(pred), "ha_fp": fp, "ha_fn": fn,
            "ha_fpr": fpr, "ha_fnr": fnr,
            "ha_balanced_error": None if fpr is None or fnr is None else (fpr + fnr) / 2}


# ------------------------------------------------------------------------------------------- cells

def load_cells(ds: C.Dataset, phases: set[str], reps: set[int] | None) -> list[dict]:
    """The dataset's CELLS.tsv rows (filtered), each with its stratum fields attached."""
    rows = [r for r in C.read_tsv(ds.cells)
            if (not phases or r["phase"] in phases) and (reps is None or int(r["rep"]) in reps)]
    extra: dict[tuple, dict] = {}
    if ds.stage is not None:
        stage = json.loads(ds.stage.read_text())
        for c in stage["cells"]:
            ident = (c.get("phase", stage.get("phase")), int(c["rep"]), c.get("arm", stage.get("arm")))
            extra[ident] = ({"cu": c["cu"]} if ds.name in ("l2reg", "l3reg", "n1000")
                            else {"level": c["level"], "ils": c["ils"]})
    out = []
    for r in rows:
        ident = (r["phase"], int(r["rep"]), r["arm"])
        out.append({"phase": r["phase"], "rep": int(r["rep"]), "arm": r["arm"],
                    "n_loci_source": int(r["n_loci"]), "name": C.cell_name(*ident),
                    **extra.get(ident, {})})
    return out


def load_truths(ds: C.Dataset, cells: list[dict], problems: list[str]) -> dict:
    truths = {}
    for c in cells:
        tkey = (c["phase"], c["rep"])
        if tkey in truths:
            continue
        cell = CAMUS.CamusCell(c["phase"], c["rep"], c["arm"], ds.source)
        t = CAMUS.truth_of(cell)
        if ds.name == "netcs":
            enewick = (cell.cell_dir / "true_net.phylonet.nwk").read_text().strip()
        else:
            enewick = strip_gamma(N.to_enewick(t))
        ha = hybrid_ancestry(t)
        if ha != hybrid_ancestry_text(enewick):
            problems.append(f"truth {tkey}: the two hybrid-ancestry derivations differ")
        truths[tkey] = {"enewick": enewick, "taxa": set(t.labels), "ha": ha,
                        "k": int(t.n_retics), "n_taxa": len(t.labels), "n": cell.n}
    return truths


def load_manifest(ds: C.Dataset) -> dict[tuple, dict]:
    if not ds.manifest.exists():
        return {}
    return {(int(r["draw"]), r["cell"]): r for r in C.read_tsv(ds.manifest)}


# -------------------------------------------------------------------------------------------- rows

def build_rows(ds: C.Dataset, draws: list, cells: list[dict], truths: dict, runs_root: Path,
               ) -> tuple[list[dict], list[str], list[str], dict]:
    """Every (draw, cell, method) row; returns rows, completeness gaps, integrity problems, and the
    run directories read."""
    manifest = load_manifest(ds)
    rows, gaps, problems, runs_used = [], [], [], {}
    sha_checked: set[tuple] = set()
    for draw in draws:
        for kind in ("gnn", "greedy"):
            for dec in ds.decoders:
                run = ds.run_dir(draw, kind, dec, runs_root)
                runs_used[f"{C.draw_label(draw)}/{kind}/{dec}"] = str(run)
                n_missing = 0
                for c in cells:
                    f = run / "cells" / f"{c['name']}.json"
                    if not f.exists():
                        n_missing += 1
                        continue
                    rows += cell_rows(ds, draw, kind, dec, c, json.loads(f.read_text()), f,
                                      truths[(c["phase"], c["rep"])], manifest, sha_checked,
                                      gaps, problems)
                if n_missing:
                    gaps.append(f"{C.draw_label(draw)} {kind} {dec}: {n_missing} of {len(cells)} "
                                f"cells missing from {run}")
    return rows, gaps, problems, runs_used


def cell_rows(ds, draw, kind, dec, c, cell, f, truth, manifest, sha_checked, gaps, problems):
    where = f"{C.draw_label(draw)} {f.parent.parent.name}/{f.name}"
    if C.cell_decoder(cell) != dec:
        problems.append(f"{where}: decoded as {C.cell_decoder(cell)}, expected {dec}")
    if int(cell["k"]) != truth["k"]:
        problems.append(f"{where}: k = {cell['k']}, the truth has {truth['k']}")
    trees = CAMUS.ARMS[c["arm"]][0]
    if cell.get("trees") != trees:
        problems.append(f"{where}: read {cell.get('trees')}, the arm's file is {trees}")
    n_loci = int(cell["n_loci"])
    m = int(cell.get("m", min(1000, n_loci)))           # l3reg's greedy_camus predates the field
    seed = ""
    if draw == "full":
        if n_loci != c["n_loci_source"] or m != min(1000, n_loci):
            problems.append(f"{where}: n_loci {n_loci}, m {m}; the source holds "
                            f"{c['n_loci_source']}")
    else:
        if n_loci != C.N_SUB or m != C.N_SUB:
            problems.append(f"{where}: n_loci {n_loci}, m {m}, not {C.N_SUB} -- a forgotten "
                            f"--m-from-trees, or the wrong --inputs")
        rec = manifest.get((int(draw), c["name"]))
        if rec is None:
            problems.append(f"{where}: no MANIFEST.tsv row")
        else:
            seed = int(rec["seed"])
            if (draw, c["name"]) not in sha_checked:
                sha_checked.add((draw, c["name"]))
                staged = C.HERE / rec["path"]
                if not staged.exists() or C.sha256_bytes(staged.read_bytes()) != rec["sha256_sub"]:
                    problems.append(f"{where}: {staged} is not the file MANIFEST.tsv records")
    if kind == "gnn":
        prep = sum(cell["wall"].get(s, 0.0) for s in SHARED)
        want_tags = GNN_TAGS
    else:
        prep = cell["wall"]["prep"]
        want_tags = ("greedy",)
    by_tag = {mr["tag"]: mr for mr in cell["models"]}
    out = []
    for tag in want_tags:
        mr = by_tag.get(tag)
        if mr is None:
            gaps.append(f"{where}: no model {tag}")
            continue
        model = (sum(mr["wall"].get(s, 0.0) for s in PER_MODEL) if kind == "gnn"
                 else cell["wall"]["decode"])
        label = C.LABELS[tag]
        row = {"dataset": ds.name, "draw": str(draw),
               "sub_seed": seed, "m": m, "n_loci_source": c["n_loci_source"],
               "phase": c["phase"], "rep": c["rep"], "arm": c["arm"], "key": cell["key"],
               "n": truth["n"], "n_taxa": truth["n_taxa"], "k_true": truth["k"],
               **{s: c[s] for s in STRATA[ds.name]},
               "decoder": dec, "method": label + C.decoder_suffix(dec), "_base": label, "tag": tag,
               "n_clusters": int(cell["n_clusters"]), "n_selected": int(mr["n_selected"]),
               **{k: mr[k] for k in ("tp", "fp", "fn", "precision", "recall")},
               "cluster_f1": mr["f1"], "k_pred": int(mr["k_pred"]),
               "prep_wall": prep, "model_wall": model, "status": "ok"}
        if mr.get("enewick") is None or int(mr["k_pred"]) < 0:
            row["status"] = "no_network"          # realise() refused the selection; no luay exists
        else:
            row["_pred"] = strip_gamma(mr["enewick"])
        out.append(row)
    return out


def attach_hybrid_ancestry(rows: list[dict], truths: dict, problems: list[str]) -> dict:
    stats = Counter(networks=0, agree=0)
    for row in rows:
        src = row.get("_pred")
        if src is None:
            continue
        truth = truths[(row["phase"], row["rep"])]
        where = f"hybrid ancestry: {row['draw']} {row['method']} {row['key']} {row['arm']}"
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
    return dict(stats)


def attach_luay(rows: list[dict], truths: dict, luay: PhyloNetLuay, workdir: Path) -> None:
    for row in rows:
        pred = row.pop("_pred", None)
        if pred is not None:
            row["_i"] = luay.add(pred, truths[(row["phase"], row["rep"])]["enewick"])
    luay.resolve(workdir)
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
        row["luay_norm"] = r["luay"] / closed_form_max(row["n_taxa"], row["k_true"], row["k_pred"])
        if r["max_exact"]:
            row["luay_norm_exact"] = r["luay"] / r["max_exact"]


# ------------------------------------------------------------------------------------------ parity

def landed_index(ds: C.Dataset) -> tuple[dict, set[str]]:
    """The landed CSV keyed as our ``full`` rows are, and its column set."""
    with open(ds.landed_csv, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = set(rd.fieldnames or ())
        idx = {}
        for r in rd:
            if r["method"] not in {C.LABELS[t] + C.decoder_suffix(d) for t in C.LABELS
                                   for d in ds.decoders}:
                continue
            idx[landed_key(ds.name, r)] = r
    return idx, cols


def landed_key(name: str, r: dict) -> tuple:
    if name == "camus":        # gene_trees holds the arm; n is the phase's size; seed 0 here
        return (r["method"], r.get("gene_trees", r.get("arm")), int(r["n"]),
                int(r.get("network_idx", r.get("rep"))), int(r.get("seed", 0)))
    if name == "netcs":
        return (r["method"], r["phase"], int(r.get("network_idx", r.get("rep"))), r["arm"])
    return (r["method"], int(r.get("network_idx", r.get("rep"))))


def parity(ds: C.Dataset, rows: list[dict], problems: list[str]) -> dict:
    idx, cols = landed_index(ds)
    fields = [f for f in PARITY_FIELDS if f in cols]
    stats = Counter()
    field_hits = Counter()
    for row in rows:
        if row["draw"] != "full":
            continue
        land = idx.get(landed_key(ds.name, row))
        if land is None:
            stats["no_landed_row"] += 1
            row["landed"] = ""
            continue
        allowed = ALLOWED.get((ds.name, row["phase"], row["rep"]), set())
        bad = []
        for f in fields:
            if land.get(f, "") == "":
                continue
            field_hits[f] += 1
            if fmt(row.get(f)) != land[f] and f not in allowed:
                bad.append(f"{f} {fmt(row.get(f))!r} vs landed {land[f]!r}")
        stats["compared"] += 1
        if bad:
            stats["mismatched"] += 1
            row["landed"] = "MISMATCH"
            problems.append(f"parity {row['method']} {row['phase']} rep{row['rep']} {row['arm']}: "
                            + "; ".join(bad))
        else:
            row["landed"] = "match"
    no_row = Counter(r["method"] for r in rows if r["draw"] == "full" and r.get("landed") == "")
    return {"landed_csv": str(ds.landed_csv), "rows_compared": stats["compared"],
            "rows_mismatched": stats["mismatched"], "rows_without_landed_row": dict(no_row),
            "fields_compared": dict(field_hits),
            "allowed": {f"{k[1]} rep{k[2]}": sorted(v) for k, v in ALLOWED.items() if k[0] == ds.name}}


# ----------------------------------------------------------------------------------------- summary

def _num(x):
    return None if x is None or x == "" or missing(x) else float(x)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarise(ds: C.Dataset, rows: list[dict], sub_draws: list[int]) -> dict:
    """Per method: full vs subsample, paired by cell (module docstring)."""
    main = [r for r in rows if ds.name != "camus" or r["arm"] == "cam_g500"]
    order = {m: i for i, m in enumerate(C.METHOD_ORDER)}
    methods = sorted({r["method"]: r for r in rows}.values(),
                     key=lambda r: (order.get(r["_base"], 99), r["decoder"]))
    labels = [r["method"] for r in methods]

    def block(rs: list[dict]) -> dict:
        out = {}
        for label in labels:
            by_cell: dict[tuple, dict] = defaultdict(dict)
            for r in rs:
                if r["method"] == label:
                    by_cell[(r["phase"], r["rep"], r["arm"])][r["draw"]] = r
            paired = {k: v for k, v in by_cell.items()
                      if "full" in v and any(str(d) in v for d in sub_draws)}
            rec = {"cells": len(paired)}
            for metric, better in METRICS:
                full = [_num(v["full"].get(metric)) for v in paired.values()]
                per_draw = {d: _med([_num(v[str(d)].get(metric)) for v in paired.values()
                                     if str(d) in v]) for d in sub_draws}
                pooled = _med([_num(v[str(d)].get(metric)) for v in paired.values()
                               for d in sub_draws if str(d) in v])
                deltas, sds = [], []
                for v in paired.values():
                    f0 = _num(v["full"].get(metric))
                    xs = [_num(v[str(d)].get(metric)) for d in sub_draws if str(d) in v]
                    xs = [x for x in xs if x is not None]
                    if f0 is None or not xs:
                        continue
                    deltas.append(statistics.fmean(xs) - f0)
                    if len(xs) >= 2:
                        sds.append(statistics.stdev(xs))
                worse = [d for d in deltas if (d < 0 if better == "higher" else d > 0)]
                better_ = [d for d in deltas if (d > 0 if better == "higher" else d < 0)]
                rec[metric] = {"full_median": _med(full), "sub_median": pooled,
                               "sub_median_per_draw": per_draw,
                               "delta_median": _med(deltas),
                               "delta_mean": statistics.fmean(deltas) if deltas else None,
                               "cells_paired": len(deltas),
                               "frac_worse": len(worse) / len(deltas) if deltas else None,
                               "frac_better": len(better_) / len(deltas) if deltas else None,
                               "within_cell_sd_mean": statistics.fmean(sds) if sds else None}
            ratios = []
            for v in paired.values():
                c0 = v["full"]["n_clusters"]
                cs = [v[str(d)]["n_clusters"] for d in sub_draws if str(d) in v]
                if c0 and cs:
                    ratios.append(statistics.fmean(cs) / c0)
            rec["n_clusters_ratio_median"] = _med(ratios)
            rec["k_pred_eq_k"] = {
                "full": _frac([v["full"] for v in paired.values()]),
                "sub": _frac([v[str(d)] for v in paired.values() for d in sub_draws if str(d) in v])}
            out[label] = rec
        return out

    strata = defaultdict(list)
    for r in rows:
        strata[stratum_of(ds.name, r)].append(r)
    return {"main_cells": "cam_g500 only (the headline 80)" if ds.name == "camus" else "all",
            "main": block(main),
            "by_stratum": {s: {m: v["cluster_f1"] for m, v in block(rs).items()}
                           for s, rs in sorted(strata.items())}}


def _frac(rs: list[dict]):
    rs = [r for r in rs if r["status"] == "ok"]
    return sum(int(r["k_pred"]) == int(r["k_true"]) for r in rs) / len(rs) if rs else None


def stratum_of(name: str, r: dict) -> str:
    """camus: the phase (n); l2reg/l3reg: the ILS level cu; netcs: level x ILS."""
    if name == "camus":
        return r["phase"]
    if name == "netcs":
        return f"L{r['level']} ils{r['ils']}"
    return f"cu{r['cu']}"


def render(ds: C.Dataset, summary: dict, sub_draws: list[int], header: list[str]) -> str:
    def f(x, w=7, p=4):
        return f"{'':>{w}}" if x is None else f"{x:{w}.{p}f}"

    lines = [*header, ""]
    for metric, better in METRICS:
        lines.append(f"## {metric} ({better} is better) -- {summary['main_cells']}; delta = mean "
                     f"over a cell's draws - its full value")
        lines.append(f"{'method':38s} {'cells':>5s} {'full':>7s} {'sub':>7s} {'delta md':>8s} "
                     f"{'delta mn':>8s} {'worse':>6s} {'better':>6s} {'draw SD':>7s}  per-draw medians")
        for label, rec in summary["main"].items():
            m = rec[metric]
            draws = " ".join(f(m["sub_median_per_draw"].get(d), 6, 3) for d in sub_draws)
            lines.append(f"{label:38s} {rec['cells']:5d} {f(m['full_median'])} {f(m['sub_median'])} "
                         f"{f(m['delta_median'], 8)} {f(m['delta_mean'], 8)} "
                         f"{f(m['frac_worse'], 6, 2)} {f(m['frac_better'], 6, 2)} "
                         f"{f(m['within_cell_sd_mean'])}  {draws}")
        lines.append("")
    lines.append("## candidate set and budget")
    lines.append(f"{'method':38s} {'|C| sub/full':>12s} {'k_pred=k full':>13s} {'k_pred=k sub':>12s}")
    for label, rec in summary["main"].items():
        lines.append(f"{label:38s} {f(rec['n_clusters_ratio_median'], 12, 3)} "
                     f"{f(rec['k_pred_eq_k']['full'], 13, 3)} {f(rec['k_pred_eq_k']['sub'], 12, 3)}")
    lines.append("")
    lines.append("## cluster_f1 delta median (cells) by stratum")
    labels = list(summary["main"])
    lines.append(f"{'stratum':14s} " + " ".join(f"{lab[:26]:>26s}" for lab in labels))
    for s, recs in summary["by_stratum"].items():
        lines.append(f"{s:14s} " + " ".join(
            f"{(f(recs[lab]['delta_median'], 8) + ' (' + str(recs[lab]['cells_paired']) + ')') if lab in recs else '':>26s}"
            for lab in labels))
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main

def parse_reps(spec: str) -> set[int] | None:
    if not spec:
        return None
    want: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(C.DATASETS))
    ap.add_argument("--draws", default="", help="comma list of full and draw numbers "
                                                "(default: full and every staged draw)")
    ap.add_argument("--phases", default="", help="comma list; default all")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--runs-root", type=Path, default=C.RUNS,
                    help="where the subsample runs live (a smoke points it elsewhere)")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even when cells or models are missing")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET", str(C.DEFAULT_JAR))))
    ap.add_argument("--jobs", type=int, default=4, help="PhyloNet JVMs at once (4 GB heap each)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true", help="measure every luay afresh")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ds = C.DATASETS[args.dataset]
    draws: list = []
    for tok in (args.draws.split(",") if args.draws else ["full", *range(ds.draws)]):
        tok = str(tok).strip()
        if tok == "full":
            draws.append("full")
        elif tok:
            if not 0 <= int(tok) < ds.draws:
                raise SystemExit(f"--draws: {ds.name} has draws 0..{ds.draws - 1}")
            draws.append(int(tok))
    sub_draws = [d for d in draws if d != "full"]
    cells = load_cells(ds, {p for p in args.phases.split(",") if p}, parse_reps(args.reps))
    if not cells:
        raise SystemExit("no cells selected")

    problems: list[str] = []
    truths = load_truths(ds, cells, problems)
    rows, gaps, integrity, runs_used = build_rows(ds, draws, cells, truths, args.runs_root)
    problems += integrity
    if gaps:
        print("[csv] incomplete:\n" + "\n".join(f"  {g}" for g in gaps[:30]))
        if not args.allow_partial:
            print("[csv] nothing written; --allow-partial writes what exists")
            return 1
    if not rows:
        raise SystemExit("[csv] no rows: nothing has been run for this selection yet")

    ha = attach_hybrid_ancestry(rows, truths, problems)
    print(f"[ha] the two derivations agree on {ha.get('agree', 0)} of {ha.get('networks', 0)} "
          f"networks", flush=True)
    cache_dir = C.RUNS / "phylonet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    luay = PhyloNetLuay(args.phylonet, cache=None if args.no_cache else cache_dir / "cache.jsonl",
                        seeds=() if args.no_cache else C.SEED_CACHES.get(ds.name, ()),
                        jobs=args.jobs, chunk=args.chunk)
    attach_luay(rows, truths, luay, cache_dir / "work")
    over = [r for r in rows if not missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    if over:
        problems.append(f"{len(over)} rows with luay_norm_exact > 1, e.g. "
                        f"{[(r['draw'], r['method'], r['key']) for r in over[:3]]}")
    par = parity(ds, rows, problems) if "full" in draws else None

    order = {m: i for i, m in enumerate(C.METHOD_ORDER)}
    draw_rank = lambda d: -1 if d == "full" else int(d)  # noqa: E731
    rows.sort(key=lambda r: (order.get(r["_base"], 99), r["decoder"], r["phase"], r["rep"],
                             r["arm"], draw_rank(r["draw"])))
    ident = ["dataset", "draw", "sub_seed", "m", "n_loci_source", "phase", "rep", "arm", "key", "n",
             "n_taxa", "k_true", *STRATA[ds.name], "decoder", "method", "tag"]
    columns = ident + list(SCORE_COLUMNS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{ds.name}_sub{C.N_SUB}"
    out_csv = stem.with_suffix(".csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in columns])

    summary = summarise(ds, rows, sub_draws) if sub_draws and "full" in draws else None
    header = [f"# {ds.name}: _l2mix_pair/s0 (selF1, selHA) and greedy on {C.N_SUB} of each cell's "
              f"gene trees, against the full data",
              f"# draws {', '.join(C.draw_label(d) for d in draws)} | {len(cells)} cells | "
              f"{len(rows)} rows | built {time.strftime('%Y-%m-%d %H:%M')}",
              f"# parity vs {ds.landed_csv.name}: " + (
                  f"{par['rows_compared'] - par['rows_mismatched']}/{par['rows_compared']} full rows "
                  f"match; no landed row for {par['rows_without_landed_row'] or 'none'}"
                  if par else "not run (no full draw)"),
              f"# problems: {len(problems)}; incomplete: {len(gaps)}"]
    if summary:
        (stem.parent / f"{stem.name}_summary.txt").write_text(render(ds, summary, sub_draws, header))
    rec = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": ds.name, "m": C.N_SUB,
           "draws": [C.draw_label(d) for d in draws], "cells": len(cells), "rows": len(rows),
           "rows_by_status": dict(Counter(r["status"] for r in rows)),
           "complete": not gaps and not problems, "incomplete": gaps, "problems": problems,
           "runs": runs_used, "parity": par, "hybrid_ancestry": ha,
           "luay": {"jar": str(args.phylonet), "cache": str(cache_dir / "cache.jsonl"),
                    "seed_caches": [str(p) for p in C.SEED_CACHES.get(ds.name, ())],
                    "measured": getattr(luay, "measured", None), "failed": getattr(luay, "failed", None)},
           "summary": summary, "seconds": round(time.perf_counter() - t0, 1)}
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=str) + "\n")

    print(f"[csv] {len(rows)} rows -> {out_csv}")
    if par:
        print(f"[parity] {par['rows_compared'] - par['rows_mismatched']}/{par['rows_compared']} full "
              f"rows reproduce {ds.landed_csv.name}; fields compared {par['fields_compared']}; "
              f"no landed row: {par['rows_without_landed_row'] or 'none'}")
    if summary:
        print((stem.parent / f"{stem.name}_summary.txt").read_text())
    for p in problems[:40]:
        print(f"  PROBLEM {p}")
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more (the JSON has them all)")
    print(f"[csv] {'clean' if not problems else f'{len(problems)} problems'}"
          f"{'' if not gaps else f', {len(gaps)} gaps (--allow-partial)'} in "
          f"{time.perf_counter() - t0:.0f} s")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
