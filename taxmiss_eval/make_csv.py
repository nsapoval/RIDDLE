#!/usr/bin/env python3
"""The missing-taxa per-network table: one CSV per family, a row per (condition, cell, method).

Rows
----
* **GNN** -- ``ours/l2mix_pair/selF1 [relaxed L<L>]``: ``run_gnn.sh``'s cell JSONs under
  ``runs/<fam>/<cond>/gnn_t1_l2mix_pair_relaxedL<L>/``.
* **Resolvers** -- ``BROOQS``, ``NetCS`` (raw TOB) and, where ``run_res.py --variants raw,fixed`` ran
  them, ``... (fixed TOB)``: ``run_res.py``'s task records, looked up by content id re-derived from
  the current TOB and gene-tree files, so a record made from other inputs is not found.

This file is what ``nni_ensemble`` reads as the **landed** rows of a taxmiss dataset
(``cluster_anatomy/common.py:_derived``, ``extra["landed_csv"]``): ``bench.py`` takes the GNN and the
two resolvers' ``cluster_f1`` / ``luay_norm_exact`` / ``ha_*`` from it (gate B1 holds the emitted
network to the GNN row), and ``make_csv.py``'s luay parity compares against its ``luay``.  Both key
on (phase, rep, arm) within ``cond``.

Columns
-------
``subsample_eval/make_sub_csv.py``'s identity and score columns with ``draw`` / ``sub_seed`` replaced
by the condition (``cond, p, cap, M, tier``) and what it did to the cell (``mean_missing`` -- the
MANIFEST's, taxa absent per tree over n -- ``n_modified``, and the GNN cell's ``n_partial`` /
``min_taxa``); then ``make_res_csv.py``'s resolver columns, its ``cond`` (the landed condition key)
renamed ``landed_cond``; the TOB columns are on every row of a cell, the GNN's included.  Last,
``ctrl_tob_fn_raw`` / ``ctrl_tob_fp_raw``: the **control** (published, full-taxon) raw TOB's error
against the same true TOB, by splits rooted on OUT -- the comparison that reproduces the JSON's own
``fn_raw`` / ``fp_raw`` on the condition's TOB, which is checked on every cell.  Scoring is theirs, imported: luay and both masses by the PhyloNet jar
(``PhyloNetLuay``, on ``runs/phylonet/cache.jsonl``), hybrid ancestry by the two derivations, which
must agree; a resolver output with OUT below one of its own hybrids is ``out_not_rootable``, scored
as emitted and given no luay (the landed rule, ``make_res_csv.apply_score``).

Checks
------
* **Completeness** (nothing written, exit 1, unless ``--allow-partial``): a GNN cell JSON or a raw-TOB
  resolver record is missing.  A fixed-TOB record is optional and its absence is only counted.
* **Integrity** (written; exit 2): the split comparison does not reproduce a TOB JSON's ``fn_raw`` /
  ``fp_raw`` / ``n_true_edges``; the GNN cell was decoded otherwise, read another file, or has
  ``n_loci`` / ``m`` other than the MANIFEST's ``n_loci``, ``partial`` other than ``tm_common.PARTIAL``
  or ``n_partial`` other than the MANIFEST's ``n_modified``; ``k`` is not the truth's; a resolver record
  was scored by another ``SCORE_VERSION`` or breaks a hard invariant; the two hybrid-ancestry
  derivations disagree; a network's luay failed; ``luay_norm_exact > 1``.

Outputs: ``results/<fam>_taxmiss.csv``, ``results/<fam>_taxmiss_gates.json`` and
``results/<fam>_taxmiss_summary.txt`` (per condition and method: medians, statuses, and the paired
difference from the published control, ``nni_ensemble/results/<fam>_bench.csv``).

**Never run two of these at once**: they append to one jar cache.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python taxmiss_eval/make_csv.py --dataset camus,l2reg,l3reg --jobs 4
    python taxmiss_eval/make_csv.py --dataset camus --conds p0.90_c24 --allow-partial
"""

from __future__ import annotations

import tm_common as TM  # first: BLAS pin, gnn_cpu_v1/pylib at sys.path[0], subsample_eval's common

import argparse  # noqa: E402
import csv  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import importlib.util  # noqa: E402

import make_sub_csv as S  # noqa: E402  -- subsample_eval's, on sys.path via tm_common


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: this package's run_res, by path: ``import run_res`` would find subsample_eval's, which tm_common
#: put first on sys.path
TR = _load("tm_run_res", TM.HERE / "run_res.py")
#: make_res_csv imports ``run_res`` by name and means subsample_eval's -- the one TR loaded
sys.modules["run_res"] = TR.SR
MR = _load("sub_make_res_csv", TM.SUB / "make_res_csv.py")
P, B = TR.P, TR.B
C = TM.SC

GNN_TAG = "_l2mix_pair"
IDENTITY = ("dataset", "cond", "p", "cap", "M", "tier", "phase", "rep", "arm", "cell", "key", "n",
            "n_taxa", "k_true")
CONDITION = ("mean_missing", "n_modified", "n_partial", "min_taxa", "m", "n_loci_source")
RESOLVER_COLUMNS = tuple("landed_cond" if c == "cond" else c for c in MR.RESOLVER_COLUMNS)
CONTROL_TOB = ("ctrl_tob_fn_raw", "ctrl_tob_fp_raw")
METRICS = S.METRICS


def columns(fam: str) -> list[str]:
    return [*IDENTITY, *S.STRATA[fam], *CONDITION, "decoder", "method", "tag",
            *[c for c in S.SCORE_COLUMNS if c != "landed"], *RESOLVER_COLUMNS, *CONTROL_TOB]


def record(fam: str, cond: str):
    """``subsample_eval``-shaped record for ``make_sub_csv.load_cells`` / ``load_truths``: the
    parent's name (they dispatch on it) and stage.json (strata), this condition's CELLS and inputs."""
    return dataclasses.replace(C.DATASETS[fam], source=TM.staged_dir(fam, cond), cells=TM.cells_file(fam, cond))


def base_row(fam: str, cond: str, c: dict, mrow: dict, truth: dict) -> dict:
    return {"dataset": fam, "cond": cond, "p": mrow["p"], "cap": mrow["cap"], "M": mrow["M"],
            "tier": mrow["tier"], "phase": c["phase"], "rep": c["rep"], "arm": c["arm"],
            "cell": c["name"], "key": mrow["key"], "n": truth["n"], "n_taxa": truth["n_taxa"],
            "k_true": truth["k"], **{s: c[s] for s in S.STRATA[fam]},
            "mean_missing": mrow["mean_missing"], "n_modified": mrow["n_modified"],
            "n_loci_source": c["n_loci_source"], "status": "ok"}


# --------------------------------------------------------------------------------------- the TOBs

def tob_columns(fam: str, cond: str, row: dict, control: dict, problems: list[str]) -> dict:
    """The condition's TOB fields (its JSON) and the control TOB's error, both by splits against the
    JSON's ``true_tob`` -- the split comparison is held to the JSON's own ``fn_raw`` / ``fp_raw``."""
    paths = TM.tob_paths(fam, cond, row["name"])
    meta = json.loads(paths["json"].read_text())
    true = P.tree_splits(f"({C.OUT},{meta['true_tob'].strip().rstrip(';')});", C.OUT)
    raw = P.tree_splits(TR.SR.wrap_raw(paths["raw"].read_text()), C.OUT)
    if (len(true - raw), len(raw - true), len(true)) != (meta["fn_raw"], meta["fp_raw"], meta["n_true_edges"]):
        problems.append(f"{cond} {row["name"]}: splits give fn/fp/true {len(true - raw)}/{len(raw - true)}/"
                        f"{len(true)}, the TOB json {meta['fn_raw']}/{meta['fp_raw']}/{meta['n_true_edges']}")
    out = MR.tob_fields(meta)
    ctrl = control.get(TR.SR.cond_key(fam, row["phase"], row["rep"], row["arm"]))
    if ctrl is not None:
        out.update(ctrl_tob_fn_raw=len(true - ctrl), ctrl_tob_fp_raw=len(ctrl - true))
    return out


# --------------------------------------------------------------------------------------- GNN rows

def gnn_rows(fam: str, cond: str, cells: list[dict], truths: dict, manifest: dict, tobs: dict,
             gaps: list[str], problems: list[str]) -> list[dict]:
    L = TM.FAMILIES[fam].level
    dec = f"relaxed_L{L}"
    run = TM.gnn_run(fam, cond)
    rows = []
    for c in cells:
        f = run / "cells" / f"{c['name']}.json"
        if not f.exists():
            gaps.append(f"{cond} {c['name']}: no GNN cell JSON in {run}")
            continue
        cell = json.loads(f.read_text())
        mrow = manifest[c["name"]]
        truth = truths[(c["phase"], c["rep"])]
        where = f"{cond} {run.name}/{f.name}"
        if C.cell_decoder(cell) != dec:
            problems.append(f"{where}: decoded as {C.cell_decoder(cell)}, expected {dec}")
        if int(cell["k"]) != truth["k"]:
            problems.append(f"{where}: k = {cell['k']}, the truth has {truth['k']}")
        if cell.get("trees") != S.CAMUS.ARMS[c["arm"]][0]:
            problems.append(f"{where}: read {cell.get('trees')}, the arm's file is {S.CAMUS.ARMS[c['arm']][0]}")
        if int(cell["n_loci"]) != mrow["n_loci"] or int(cell["m"]) != mrow["n_loci"]:
            problems.append(f"{where}: n_loci {cell['n_loci']}, m {cell['m']}; the MANIFEST says {mrow['n_loci']}")
        if int(cell.get("partial", 0)) != TM.PARTIAL:
            problems.append(f"{where}: --partial {cell.get('partial', 0)}, not {TM.PARTIAL}")
        if cell.get("n_partial") != mrow["n_modified"]:
            problems.append(f"{where}: {cell.get('n_partial')} partial trees read, the MANIFEST "
                            f"modified {mrow['n_modified']}")
        mr = next((m for m in cell["models"] if m["tag"] == GNN_TAG and int(m["seed"]) == 0), None)
        if mr is None:
            gaps.append(f"{where}: no model {GNN_TAG}/s0")
            continue
        row = {**base_row(fam, cond, c, mrow, truth), **tobs[c["name"]], "n_partial": cell.get("n_partial"),
               "min_taxa": cell.get("min_taxa"), "m": int(cell["m"]), "decoder": dec,
               "method": C.LABELS[GNN_TAG] + C.decoder_suffix(dec), "tag": GNN_TAG,
               "n_clusters": int(cell["n_clusters"]), "n_selected": int(mr["n_selected"]),
               **{k: mr[k] for k in ("tp", "fp", "fn", "precision", "recall")},
               "cluster_f1": mr["f1"], "k_pred": int(mr["k_pred"]),
               "prep_wall": sum(cell["wall"].get(s, 0.0) for s in S.SHARED),
               "model_wall": sum(mr["wall"].get(s, 0.0) for s in S.PER_MODEL)}
        if mr.get("enewick") is None or int(mr["k_pred"]) < 0:
            row["status"] = "no_network"
        else:
            row["_pred"] = S.strip_gamma(mr["enewick"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------- resolver rows

def resolver_rows(fam: str, cond: str, cells: list[dict], truths: dict, manifest: dict, tobs: dict,
                  gaps: list[str], problems: list[str], stats: Counter) -> list[dict]:
    names = {c["name"]: c for c in cells}
    tasks = TR.build_tasks(fam, [cond], set(TR.VARIANTS[fam]), set(names))
    index = {(t["cell"], v, t["resolver"]): t for t in tasks for v in t["variants"]}
    rows = []
    for name, c in names.items():
        mrow = manifest[name]
        truth = truths[(c["phase"], c["rep"])]
        for variant in TR.VARIANTS[fam]:
            for resolver in P.RESOLVERS:
                t = index[(name, variant, resolver)]
                f = TR.RES / fam / "tasks" / f"{t['task_id']}.json"
                if not f.exists():
                    if variant in TR.DEFAULT_VARIANTS:
                        gaps.append(f"{cond} {name} {variant} {resolver}: no task record")
                    else:
                        stats[f"optional_{variant}_not_run"] += 1
                    continue
                rec = json.loads(f.read_text())
                lab = TR.label(fam, variant, resolver)
                where = f"{cond} {name} {lab}"
                if rec.get("score_version") != TR.SR.SCORE_VERSION:
                    problems.append(f"{where}: scored by version {rec.get('score_version')}")
                st = P.tob_stats(t["tob_text"])
                row = {**base_row(fam, cond, c, mrow, truth), "m": mrow["n_loci"], "decoder": "",
                       "method": lab, "tag": resolver, "variant": variant, "protocol": "rooted",
                       "landed_cond": t["cond"], "n_elig": rec["n_elig"], "n_blobs_tob": st["n_poly"],
                       "task_id": t["task_id"], **tobs[name]}
                row["status"] = rec["status"]
                tm = rec.get("timing", {})
                row.update(model_wall=tm.get("wall"), resolver_cpu=tm.get("cpu"),
                           resolver_rss_mb=tm.get("rss_mb"), timed_out=tm.get("timed_out"),
                           note=tm.get("note"))
                if rec.get("violations"):
                    row["violations"] = "; ".join(rec["violations"])
                    problems.append(f"{where}: hard invariant violated: {rec['violations']}")
                MR.apply_score(row, rec.get("score"), rec.get("enewick"))
                rows.append(row)
    return rows


def attach_hybrid_ancestry(rows: list[dict], truths: dict, problems: list[str]) -> dict:
    """``make_res_csv.attach_hybrid_ancestry``, keyed by condition: a GNN or rooted resolver network
    is read as rooted (``_pred``), an out_not_rootable one as emitted (``_emitted``).  Before luay,
    which pops ``_pred``."""
    stats = Counter(networks=0, agree=0)
    for row in rows:
        emitted = row.pop("_emitted", None)
        src = row.get("_pred") or emitted
        if src is None:
            continue
        truth = truths[(row["phase"], row["rep"])]
        where = f"hybrid ancestry: {row['cond']} {row['method']} {row['cell']}"
        try:
            net = S.N.from_enewick(src)
            got, text = S.hybrid_ancestry(net), S.hybrid_ancestry_text(src)
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
        row.update(S.ha_metrics(got, truth["ha"]))
    return dict(stats)


# ------------------------------------------------------------------------------------------ summary

def num(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if v != v else v


def control_rows(fam: str) -> dict:
    """``(method, cell) -> row`` of the published ztc bench, the control every condition is read
    against (GNN, BROOQS, NetCS; the MAP's taxmiss rows are nni_ensemble's own)."""
    p = TM.REPO / "nni_ensemble" / "results" / f"{fam}_bench.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as fh:
        return {(r["method"], r["cell"]): r for r in csv.DictReader(fh)}


def summarise(fam: str, rows: list[dict]) -> list[str]:
    ctrl = control_rows(fam)
    bench_name = {f"ours/l2mix_pair/selF1 [relaxed L{TM.FAMILIES[fam].level}]": "GNN",
                  "BROOQS": "BROOQS", "NetCS": "NetCS", "BROOQS*": "BROOQS", "NetCS*": "NetCS"}
    out = [f"[{fam}] {len(rows)} rows; control = nni_ensemble/results/{fam}_bench.csv"
           f"{'' if ctrl else ' (MISSING)'}"]
    for cond in dict.fromkeys(r["cond"] for r in rows):
        rc = [r for r in rows if r["cond"] == cond]
        mm = statistics.fmean(float(r["mean_missing"]) for r in rc)
        out.append(f"  {cond}: mean missing {mm:.4f}")
        for meth in dict.fromkeys(r["method"] for r in rc):
            rm = [r for r in rc if r["method"] == meth]
            st = Counter(r["status"] for r in rm)
            parts = []
            for m, better in METRICS:
                xs = [num(r.get(m)) for r in rm if num(r.get(m)) is not None]
                d = []
                for r in rm:
                    cr = ctrl.get((bench_name.get(meth, ""), r["cell"]))
                    a, b = num(r.get(m)), num(cr.get(m)) if cr else None
                    if a is not None and b is not None:
                        d.append(a - b)
                parts.append(f"{m} med {statistics.median(xs) if xs else float('nan'):.4f}"
                             + (f" (vs control d {statistics.fmean(d):+.4f}, n {len(d)})" if d else ""))
            out.append(f"    {meth:36s} n {len(rm):4d} {dict(st)} | " + " | ".join(parts))
    return out


# --------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="comma list of families, or 'all' (every one staged)")
    ap.add_argument("--conds", default="all")
    ap.add_argument("--jobs", type=int, default=4, help="PhyloNet JVMs")
    ap.add_argument("--phylonet", type=Path, default=C.DEFAULT_JAR)
    ap.add_argument("--allow-partial", action="store_true", help="write despite missing runs")
    ap.add_argument("--out-dir", type=Path, default=TM.RESULTS)
    args = ap.parse_args()
    fams = TM.families_staged() if args.dataset == "all" else [f for f in args.dataset.split(",") if f]
    if not fams:
        raise SystemExit("nothing staged -- run taxmiss_eval/stage.py")
    rc = 0
    for fam in fams:
        conds = TM.parse_conds(fam, args.conds)
        if not conds:
            raise SystemExit(f"{fam}: nothing staged")
        t0 = time.perf_counter()
        gaps, problems, stats = [], [], Counter()
        rows, truths = [], {}
        man = {(r["cond"], r["cell"]): r for r in TM.manifest(fam)}
        control = TR.SR.full_tob_splits(fam)
        for cond in conds:
            SD = record(fam, cond)
            cells = S.load_cells(SD, set(), None)
            tr = S.load_truths(SD, cells, problems)
            truths.update(tr)
            manifest = {name: r for (cd, name), r in man.items() if cd == cond}
            tobs = {c["name"]: tob_columns(fam, cond, c, control, problems) for c in cells}
            stats["control_tobs"] += sum(1 for t in tobs.values() if "ctrl_tob_fn_raw" in t)
            rows += gnn_rows(fam, cond, cells, tr, manifest, tobs, gaps, problems)
            rows += resolver_rows(fam, cond, cells, tr, manifest, tobs, gaps, problems, stats)
        if gaps and not args.allow_partial:
            for g in gaps[:30]:
                print(f"  GAP {g}")
            print(f"[make_csv] {fam}: {len(gaps)} runs missing -- nothing written "
                  f"(--allow-partial writes what exists)", file=sys.stderr)
            rc = max(rc, 1)
            continue
        ha = attach_hybrid_ancestry(rows, truths, problems)
        cache = TM.RUNS / "phylonet" / "cache.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        seeds = [p for p in (C.RUNS / "phylonet" / "cache.jsonl",
                             TM.REPO / "ens_anatomy" / "runs" / "phylonet" / "cache.jsonl",
                             TM.REPO / "nni_ensemble" / "runs" / "phylonet" / "cache.jsonl",
                             *C.SEED_CACHES.get(fam, ())) if p.exists()]
        luay = S.PhyloNetLuay(args.phylonet, cache, seeds=seeds, jobs=args.jobs)
        S.attach_luay(rows, truths, luay, TM.RUNS / "phylonet" / "work")
        for r in rows:
            if r["status"] == "phylonet_failed":
                problems.append(f"{r['cond']} {r['cell']} {r['method']}: PhyloNet failed")
            v = num(r.get("luay_norm_exact"))
            if v is not None and v > 1 + 1e-9:
                problems.append(f"{r['cond']} {r['cell']} {r['method']}: luay_norm_exact {v} > 1")
        cols = columns(fam)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda r: (conds.index(r["cond"]), r["phase"], int(r["rep"]), r["arm"], r["method"]))
        tmp = out / f"{fam}_taxmiss.csv.tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: S.fmt(r.get(k)) for k in cols})
        tmp.replace(out / f"{fam}_taxmiss.csv")
        by = Counter((r["cond"], r["method"], r["status"]) for r in rows)
        gates = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "family": fam, "conds": conds,
                 "rows": len(rows), "by_cond_method_status": {"|".join(k): v for k, v in sorted(by.items())},
                 "hybrid_ancestry": ha, "luay": {"measured": luay.measured, "failed": luay.failed},
                 "stats": dict(stats), "gaps": gaps, "problems": problems,
                 "seconds": round(time.perf_counter() - t0, 1)}
        TM.write_json(out / f"{fam}_taxmiss_gates.json", gates)
        text = summarise(fam, rows)
        text.append(f"=== {len(rows)} rows, {len(gaps)} gaps, {len(problems)} problems")
        (out / f"{fam}_taxmiss_summary.txt").write_text("\n".join(text) + "\n")
        print("\n".join(text), flush=True)
        for p in problems[:20]:
            print(f"  PROBLEM {p}")
        if problems:
            rc = max(rc, 2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
