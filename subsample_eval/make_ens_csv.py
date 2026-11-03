#!/usr/bin/env python3
"""The ensemble rows, scored as the sub500 rows are, and read against them.

Rows
----
One per (cell, method, aggregator), with ``draw = ens``.  ``method`` is the landed label with its
decoder suffix (``ours/l2mix_pair/selF1 [relaxed L2]``, ``greedy``, ...), and ``agg`` is ``pool``,
``pool_present`` or ``vote`` -- greedy has ``vote`` only, the greedy-vote control.  The cells are
``run_ens.py decode``'s, under ``<runs-root>/<ds>/ens/<agg>/{gnn_t1_l2mix_pair,greedy}{,_relaxedL<L>}/``.

Columns are ``results/<ds>_sub500.csv``'s, then the ensemble block: ``agg, n_draws, n_majority,
majority_refused, sel_equals_a_draw, sel_in_all_draws, sel_in_one_draw, net_equals_full,
net_jaccard_full``.  ``n_clusters`` is the union's size.  ``prep_wall`` is the five draws'
preparation plus the union and its instance; ``model_wall`` is the five forwards (none for greedy),
the five per-draw decodes when the aggregator is ``vote``, and the ensemble decode.

Scoring
-------
Exactly ``make_sub_csv.py``'s, imported rather than copied (this file sits beside it, so the
``common`` it imports is this package's): F1 is the driver's own ``Instance.cluster_counts``; luay
and the masses come from the PhyloNet jar through ``make_sub_csv.PhyloNetLuay`` and the shared
cache ``runs/phylonet/cache.jsonl``; hybrid ancestry is derived twice and the two must agree.  **Do
not run this beside ``make_sub_csv.py``**: both append to that cache.

The full and subsample rows are **read** from ``results/<ds>_sub500.csv``, which this never
rewrites, so every comparison is between numbers computed by one set of functions.

Summary
-------
Per method x aggregator, for ``cluster_f1`` (higher is better), ``luay_norm_exact`` and
``ha_balanced_error`` (lower is better), over the dataset's main cells (camus: the 80 ``cam_g500``):

* the medians of the full-data value, of each cell's mean over its five draws, and of the ensemble;
* **ens - full**, the headline: median, mean, the counts better / worse / tied, and a two-sided
  Wilcoxon signed-rank p (``zero_method="pratt"``) with the number of non-zero pairs;
* **ens - draws**, the ensemble against the mean of its own five draws;
* k_pred = k, and ens - full on F1 by stratum;
* the draw-count sweep (``results/<ds>_ens_sweep.tsv``): median F1 and HA error at D = 1..5;
* the gates' tallies from ``run_ens.py decode``.

**How to read ens - full.**  Five random halves of 1,000 trees together cover ~97% of them, so
ens - full compares two ways of *combining* nearly the same data -- one decode on all of it, against
five half-data decodes merged -- not two amounts of data.  ens - draws mixes more data with the
combining; the sweep separates the two.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/make_ens_csv.py --dataset l2reg --jobs 4
    python subsample_eval/make_ens_csv.py --dataset l2reg --no-luay            # seconds, no jar
    python subsample_eval/make_ens_csv.py --dataset camus --phases cam50 --reps 0 \\
        --runs-root /tmp/smoke/runs --out-dir /tmp/smoke/results --allow-partial
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import common as C
import make_sub_csv as MS

if str(C.GNN) not in sys.path:           # gnn_cpu_v1/ensemble.py; pylib stays at sys.path[0]
    sys.path.insert(1, str(C.GNN))

import ensemble as E  # noqa: E402

ENS_DATASETS = ("camus", "l2reg", "l3reg")
GNN_TAGS = MS.GNN_TAGS
ENS_COLUMNS = ("agg", "n_draws", "n_majority", "majority_refused", "sel_equals_a_draw",
               "sel_in_all_draws", "sel_in_one_draw", "net_equals_full", "net_jaccard_full")


def aggs_for(kind: str) -> tuple[str, ...]:
    return ("vote",) if kind == "greedy" else E.AGGREGATORS


def run_dir(runs_root: Path, ds: C.Dataset, agg: str, kind: str, dec: str) -> Path:
    base = C.GNN_RUN if kind == "gnn" else C.GREEDY_RUN
    sfx = "" if dec == "level1" else "_relaxedL" + dec.rsplit("_L", 1)[1]
    return Path(runs_root) / ds.name / "ens" / agg / (base + sfx)


def build_rows(ds: C.Dataset, cells: list[dict], truths: dict, runs_root: Path):
    manifest = MS.load_manifest(ds)
    rows, gaps, problems, runs_used = [], [], [], {}
    sha_checked: set[tuple] = set()
    for kind in ("gnn", "greedy"):
        for agg in aggs_for(kind):
            for dec in ds.decoders:
                run = run_dir(runs_root, ds, agg, kind, dec)
                runs_used[f"{agg}/{kind}/{dec}"] = str(run)
                n_missing = 0
                for c in cells:
                    f = run / "cells" / f"{c['name']}.json"
                    if not f.exists():
                        n_missing += 1
                        continue
                    rows += ens_rows(ds, kind, agg, dec, c, json.loads(f.read_text()), f,
                                     truths[(c["phase"], c["rep"])], manifest, sha_checked,
                                     gaps, problems)
                if n_missing:
                    gaps.append(f"{agg} {kind} {dec}: {n_missing} of {len(cells)} cells missing "
                                f"from {run}")
    return rows, gaps, problems, runs_used


def ens_rows(ds, kind, agg, dec, c, cell, f, truth, manifest, sha_checked, gaps, problems):
    """``make_sub_csv.cell_rows`` for an ensemble cell: the same fields, its own integrity checks."""
    where = f"ens {agg} {f.parent.parent.name}/{f.name}"
    if C.cell_decoder(cell) != dec:
        problems.append(f"{where}: decoded as {C.cell_decoder(cell)}, expected {dec}")
    if int(cell["k"]) != truth["k"]:
        problems.append(f"{where}: k = {cell['k']}, the truth has {truth['k']}")
    if cell.get("trees") != MS.CAMUS.ARMS[c["arm"]][0]:
        problems.append(f"{where}: read {cell.get('trees')}, the arm's file is "
                        f"{MS.CAMUS.ARMS[c['arm']][0]}")
    per_draw = cell.get("n_loci_per_draw", [])
    if int(cell["n_draws"]) != ds.draws or per_draw != [C.N_SUB] * ds.draws or int(cell["m"]) != C.N_SUB:
        problems.append(f"{where}: n_draws {cell['n_draws']}, n_loci {per_draw}, m {cell['m']}; "
                        f"expected {ds.draws} draws of {C.N_SUB}")
    for d in range(ds.draws):
        if (d, c["name"]) in sha_checked:
            continue
        sha_checked.add((d, c["name"]))
        rec = manifest.get((d, c["name"]))
        staged = None if rec is None else C.HERE / rec["path"]
        if staged is None or not staged.exists() or C.sha256_bytes(staged.read_bytes()) != rec["sha256_sub"]:
            problems.append(f"{where}: draw {d}'s staged file is not the one MANIFEST.tsv records")
    want_tags = GNN_TAGS if kind == "gnn" else ("greedy",)
    by_tag = {mr["tag"]: mr for mr in cell["models"]}
    prep = sum(cell["wall"].get(s, 0.0) for s in ("draws_prep", "union", "instance"))
    out = []
    for tag in want_tags:
        mr = by_tag.get(tag)
        if mr is None:
            gaps.append(f"{where}: no model {tag}")
            continue
        ens = mr["ensemble"]
        if ens["agg"] != agg or ens["source"] != tag or int(ens["n_draws"]) != ds.draws:
            problems.append(f"{where} {tag}: its ensemble block says {ens['agg']}/{ens['source']}/"
                            f"{ens['n_draws']}")
        dw = mr.get("draws_wall", {})
        model = (dw.get("forward", 0.0) + (dw.get("decode", 0.0) if agg == "vote" else 0.0)
                 + mr["wall"]["decode"])
        label = C.LABELS[tag]
        row = {"dataset": ds.name, "draw": "ens", "sub_seed": "", "m": int(cell["m"]),
               "n_loci_source": c["n_loci_source"], "phase": c["phase"], "rep": c["rep"],
               "arm": c["arm"], "key": cell["key"], "n": truth["n"], "n_taxa": truth["n_taxa"],
               "k_true": truth["k"], **{s: c[s] for s in MS.STRATA[ds.name]},
               "decoder": dec, "method": label + C.decoder_suffix(dec), "_base": label, "tag": tag,
               "agg": agg, "n_draws": int(ens["n_draws"]),
               **{k: ens.get(k) for k in ENS_COLUMNS[2:]},
               "n_clusters": int(cell["n_clusters"]), "n_selected": int(mr["n_selected"]),
               **{k: mr[k] for k in ("tp", "fp", "fn", "precision", "recall")},
               "cluster_f1": mr["f1"], "k_pred": int(mr["k_pred"]),
               "prep_wall": prep, "model_wall": model, "status": "ok"}
        if mr.get("enewick") is None or int(mr["k_pred"]) < 0:
            row["status"] = "no_network"
        else:
            row["_pred"] = MS.strip_gamma(mr["enewick"])
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------- summary

def reference(path: Path) -> dict:
    """``(phase, rep, arm, method) -> {"full": row, "draws": [rows]}`` from the sub500 CSV."""
    ref: dict[tuple, dict] = defaultdict(lambda: {"full": None, "draws": []})
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["phase"], int(r["rep"]), r["arm"], r["method"])
            if r["draw"] == "full":
                ref[key]["full"] = r
            else:
                ref[key]["draws"].append(r)
    return ref


def wilcoxon(deltas: list[float]):
    nz = [d for d in deltas if d != 0]
    if not nz:
        return None, 0
    from scipy.stats import wilcoxon as _w
    try:
        return float(_w(deltas, zero_method="pratt").pvalue), len(nz)
    except ValueError:
        return None, len(nz)


def summarise(ds: C.Dataset, rows: list[dict], ref: dict, decoded: list[dict]) -> dict:
    main = [r for r in rows if ds.name != "camus" or r["arm"] == "cam_g500"]
    order = {m: i for i, m in enumerate(C.METHOD_ORDER)}
    keys = sorted({(r["method"], r["agg"]): r for r in rows}.values(),
                  key=lambda r: (order.get(r["_base"], 99), r["decoder"],
                                 E.AGGREGATORS.index(r["agg"])))
    labels = [(r["method"], r["agg"]) for r in keys]

    def block(rs: list[dict]) -> dict:
        out = {}
        for method, agg in labels:
            recs = []
            for r in rs:
                if r["method"] != method or r["agg"] != agg:
                    continue
                rf = ref.get((r["phase"], r["rep"], r["arm"], method))
                if rf is None or rf["full"] is None or not rf["draws"]:
                    continue
                recs.append((r, rf))
            rec = {"cells": len(recs)}
            for metric, better in MS.METRICS:
                sign = 1 if better == "higher" else -1
                full, draws, ens, d_full, d_draws = [], [], [], [], []
                for r, rf in recs:
                    e = MS._num(r.get(metric))
                    f0 = MS._num(rf["full"].get(metric))
                    xs = [x for x in (MS._num(d.get(metric)) for d in rf["draws"]) if x is not None]
                    mdraw = statistics.fmean(xs) if xs else None
                    if f0 is not None:
                        full.append(f0)
                    if mdraw is not None:
                        draws.append(mdraw)
                    if e is not None:
                        ens.append(e)
                    if e is not None and f0 is not None:
                        d_full.append(e - f0)
                    if e is not None and mdraw is not None:
                        d_draws.append(e - mdraw)
                p, nz = wilcoxon(d_full)
                rec[metric] = {
                    "full_median": MS._med(full), "draws_median": MS._med(draws),
                    "ens_median": MS._med(ens),
                    "vs_full": {"median": MS._med(d_full),
                                "mean": statistics.fmean(d_full) if d_full else None,
                                "better": sum(sign * d > 0 for d in d_full),
                                "worse": sum(sign * d < 0 for d in d_full),
                                "tied": sum(d == 0 for d in d_full),
                                "wilcoxon_p": p, "nonzero": nz, "n": len(d_full)},
                    "vs_draws": {"median": MS._med(d_draws),
                                 "mean": statistics.fmean(d_draws) if d_draws else None,
                                 "better": sum(sign * d > 0 for d in d_draws),
                                 "worse": sum(sign * d < 0 for d in d_draws),
                                 "n": len(d_draws)}}
            ens_rows_ = [r for r, _ in recs]
            rec["k_pred_eq_k"] = MS._frac(ens_rows_)
            rec["n_clusters_ratio_median"] = MS._med([
                r["n_clusters"] / float(rf["full"]["n_clusters"]) for r, rf in recs
                if rf["full"].get("n_clusters")])
            rec["sel_differs_from_every_draw"] = sum(not r["sel_equals_a_draw"] for r in ens_rows_)
            nf = [r["net_equals_full"] for r in ens_rows_ if r.get("net_equals_full") is not None]
            rec["net_equals_full"] = sum(nf) if nf else None
            mr = [r["majority_refused"] for r in ens_rows_ if r.get("majority_refused") is not None]
            rec["majority_refused_mean"] = statistics.fmean(mr) if mr else None
            out[f"{method} {{{agg}}}"] = rec
        return out

    strata = defaultdict(list)
    for r in rows:
        strata[MS.stratum_of(ds.name, r)].append(r)
    main_names = {C.cell_name(r["phase"], r["rep"], r["arm"]) for r in main}
    return {"main_cells": "cam_g500 only (the headline 80)" if ds.name == "camus" else "all",
            "main": block(main),
            "by_stratum": {s: {m: v["cluster_f1"]["vs_full"] for m, v in block(rs).items()}
                           for s, rs in sorted(strata.items())},
            "sweep": sweep_summary(ds, decoded, main_names, ref),
            "gates": gate_tally(decoded)}


def sweep_rows(ds: C.Dataset, decoded: list[dict]) -> list[dict]:
    out = []
    for rec in decoded:
        for s in rec["sweep"]:
            out.append({"dataset": ds.name, "phase": rec["phase"], "rep": rec["rep"],
                        "arm": rec["arm"], "cell": rec["cell"], "n": rec["n"], "k_true": rec["k"],
                        **s, "method": C.LABELS[s["source"]] + C.decoder_suffix(s["decoder"])})
    return out


def sweep_summary(ds: C.Dataset, decoded: list[dict], main_names: set, ref: dict) -> dict:
    by: dict[tuple, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    full: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for r in sweep_rows(ds, [d for d in decoded if d["cell"] in main_names]):
        key = (r["method"], r["agg"])
        by[key][r["D"]]["f1"].append(r["f1"])
        if r["ha_balanced_error"] is not None:
            by[key][r["D"]]["ha"].append(r["ha_balanced_error"])
        if r["D"] == 1:
            rf = ref.get((r["phase"], r["rep"], r["arm"], r["method"]))
            if rf and rf["full"]:
                full[key]["f1"].append(MS._num(rf["full"]["cluster_f1"]))
                ha = MS._num(rf["full"].get("ha_balanced_error"))
                if ha is not None:
                    full[key]["ha"].append(ha)
    return {f"{m} {{{a}}}": {"by_D": {D: {"f1": MS._med(v["f1"]), "ha": MS._med(v["ha"]),
                                          "cells": len(v["f1"])} for D, v in sorted(Ds.items())},
                             "full": {"f1": MS._med(full[(m, a)]["f1"]),
                                      "ha": MS._med(full[(m, a)]["ha"])}}
            for (m, a), Ds in by.items()}


def gate_tally(decoded: list[dict]) -> dict:
    ident = Counter()
    for r in decoded:
        ident.update(r["identity"])
    return {"cells": len(decoded), "identity": dict(ident),
            "problems": sum(len(r["problems"]) for r in decoded),
            "union_bigger_than_every_draw": sum(r["n_union"] > r["n_full_draw_max"] for r in decoded),
            "pool_order_sensitive": dict(Counter(k for r in decoded
                                                 for k, ok in r["order_invariant"].items() if not ok))}


def render(ds: C.Dataset, summary: dict, header: list[str]) -> str:
    def f(x, w=7, p=4):
        return f"{'':>{w}}" if x is None else f"{x:{w}.{p}f}"

    def i(x, w=4):
        return f"{'':>{w}}" if x is None else f"{x:{w}d}"

    lines = [*header, "",
             "How to read ens - full: five random halves of 1,000 trees cover ~97% of them, so this",
             "compares two ways of combining nearly the same data, not two amounts of data. ens - draws",
             "is the ensemble against the mean of its own five draws. better/worse count cells.", ""]
    for metric, better in MS.METRICS:
        lines.append(f"## {metric} ({better} is better) -- {summary['main_cells']}")
        lines.append(f"{'method {agg}':50s} {'cells':>5s} {'full':>7s} {'draws':>7s} {'ens':>7s} | "
                     f"{'ens-full md':>11s} {'mean':>8s} {'bet':>4s} {'wor':>4s} {'tie':>4s} "
                     f"{'p':>7s} | {'ens-draws md':>12s} {'bet':>4s} {'wor':>4s}")
        for label, rec in summary["main"].items():
            m = rec[metric]
            vf, vd = m["vs_full"], m["vs_draws"]
            lines.append(f"{label:50s} {rec['cells']:5d} {f(m['full_median'])} "
                         f"{f(m['draws_median'])} {f(m['ens_median'])} | {f(vf['median'], 11)} "
                         f"{f(vf['mean'], 8)} {i(vf['better'])} {i(vf['worse'])} {i(vf['tied'])} "
                         f"{f(vf['wilcoxon_p'], 7, 3)} | {f(vd['median'], 12)} {i(vd['better'])} "
                         f"{i(vd['worse'])}")
        lines.append("")
    lines.append("## the ensemble itself")
    lines.append(f"{'method {agg}':50s} {'|U|/|C_full|':>12s} {'k_pred=k':>8s} {'unlike draws':>12s} "
                 f"{'net = full':>10s} {'maj refused':>11s}")
    for label, rec in summary["main"].items():
        lines.append(f"{label:50s} {f(rec['n_clusters_ratio_median'], 12, 3)} "
                     f"{f(rec['k_pred_eq_k'], 8, 3)} {i(rec['sel_differs_from_every_draw'], 12)} "
                     f"{i(rec['net_equals_full'], 10)} {f(rec['majority_refused_mean'], 11, 2)}")
    lines.append("")
    lines.append("## draw-count sweep (main cells): median F1 / HA error over draws 0..D-1; "
                 "D = 1 is draw 0 alone")
    Ds = sorted({D for v in summary["sweep"].values() for D in v["by_D"]})
    lines.append(f"{'method {agg}':50s} " + " ".join(f"{'D=' + str(D):>13s}" for D in Ds)
                 + f" {'full':>13s}")
    for label, v in sorted(summary["sweep"].items()):
        cells = " ".join(f"{f(v['by_D'][D]['f1'], 6, 3)}/{f(v['by_D'][D]['ha'], 6, 3)}"
                         if D in v["by_D"] else f"{'':>13s}" for D in Ds)
        lines.append(f"{label:50s} {cells} {f(v['full']['f1'], 6, 3)}/{f(v['full']['ha'], 6, 3)}")
    lines.append("")
    lines.append("## cluster_f1: ens - full median (cells) by stratum")
    labels = list(summary["main"])
    for s, recs in summary["by_stratum"].items():
        lines.append(f"{s}")
        for lab in labels:
            if lab in recs:
                lines.append(f"  {lab:48s} {f(recs[lab]['median'], 8)} ({recs[lab]['n']})")
    g = summary["gates"]
    lines += ["", f"## gates (run_ens.py decode): {g['cells']} cells | G-identity "
                  f"{g['identity'].get('ok', 0)}/{g['identity'].get('ok', 0) + g['identity'].get('failed', 0)}"
                  f" | problems {g['problems']} | union > every draw {g['union_bigger_than_every_draw']}"
                  f" | pool order-sensitive {g['pool_order_sensitive'] or 'none'}"]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=ENS_DATASETS)
    ap.add_argument("--phases", default="", help="comma list; default all")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--runs-root", type=Path, default=C.RUNS)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--sub-csv", type=Path, default=None,
                    help="the full + subsample rows to compare against (default "
                         "results/<ds>_sub500.csv)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even when cells or models are missing")
    ap.add_argument("--no-luay", action="store_true",
                    help="skip the jar: F1 and hybrid ancestry only (luay columns empty)")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET", str(C.DEFAULT_JAR))))
    ap.add_argument("--jobs", type=int, default=4, help="PhyloNet JVMs at once (4 GB heap each)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true", help="measure every luay afresh")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ds = C.DATASETS[args.dataset]
    cells = MS.load_cells(ds, {p for p in args.phases.split(",") if p}, MS.parse_reps(args.reps))
    if not cells:
        raise SystemExit("no cells selected")
    sub_csv = args.sub_csv or C.RESULTS / f"{ds.name}_sub{C.N_SUB}.csv"
    if not sub_csv.exists():
        raise SystemExit(f"no {sub_csv}: run make_sub_csv.py --dataset {ds.name} first")

    problems: list[str] = []
    truths = MS.load_truths(ds, cells, problems)
    rows, gaps, integrity, runs_used = build_rows(ds, cells, truths, args.runs_root)
    problems += integrity
    names = {c["name"] for c in cells}
    decoded = [json.loads(p.read_text()) for p in
               sorted((Path(args.runs_root) / ds.name / "ens" / "decoded").glob("*.json"))]
    decoded = [d for d in decoded if d["cell"] in names]
    if len(decoded) != len(cells):
        gaps.append(f"decoded: {len(decoded)} of {len(cells)} cells")
    problems += [p for d in decoded for p in d["problems"]]
    if gaps:
        print("[ens] incomplete:\n" + "\n".join(f"  {g}" for g in gaps[:30]))
        if not args.allow_partial:
            print("[ens] nothing written; --allow-partial writes what exists")
            return 1
    if not rows:
        raise SystemExit("[ens] no rows: run run_ens.py decode first")

    ha = MS.attach_hybrid_ancestry(rows, truths, problems)
    print(f"[ha] the two derivations agree on {ha.get('agree', 0)} of {ha.get('networks', 0)} "
          f"networks", flush=True)
    luay = None
    if args.no_luay:
        for r in rows:
            r.pop("_pred", None)
    else:
        cache_dir = C.RUNS / "phylonet"
        cache_dir.mkdir(parents=True, exist_ok=True)
        luay = MS.PhyloNetLuay(args.phylonet,
                               cache=None if args.no_cache else cache_dir / "cache.jsonl",
                               jobs=args.jobs, chunk=args.chunk)
        MS.attach_luay(rows, truths, luay, cache_dir / "work")
        over = [r for r in rows
                if not MS.missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
        if over:
            problems.append(f"{len(over)} rows with luay_norm_exact > 1")

    order = {m: i for i, m in enumerate(C.METHOD_ORDER)}
    rows.sort(key=lambda r: (order.get(r["_base"], 99), r["decoder"],
                             E.AGGREGATORS.index(r["agg"]), r["phase"], r["rep"], r["arm"]))
    ident = ["dataset", "draw", "sub_seed", "m", "n_loci_source", "phase", "rep", "arm", "key", "n",
             "n_taxa", "k_true", *MS.STRATA[ds.name], "decoder", "method", "tag"]
    columns = ident + list(ENS_COLUMNS) + [c for c in MS.SCORE_COLUMNS if c != "landed"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{ds.name}_ens"
    with open(stem.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([MS.fmt(r.get(c)) for c in columns])
    sw = sweep_rows(ds, decoded)
    sw_cols = ["dataset", "phase", "rep", "arm", "cell", "n", "k_true", "method", "source",
               "decoder", "agg", "D", "n_union", "n_selected", "tp", "fp", "fn", "f1", "k_pred",
               "ha_balanced_error"]
    with open(args.out_dir / f"{ds.name}_ens_sweep.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(sw_cols)
        for r in sw:
            w.writerow([MS.fmt(r.get(c)) for c in sw_cols])

    ref = reference(sub_csv)
    summary = summarise(ds, rows, ref, decoded)
    header = [f"# {ds.name}: ensembles of _l2mix_pair/s0 (selF1, selHA) and greedy over {ds.draws} "
              f"draws of {C.N_SUB} trees, against the full data and the draws",
              f"# {len(cells)} cells | {len(rows)} ensemble rows | reference {sub_csv.name} | "
              f"luay {'skipped (--no-luay)' if args.no_luay else 'PhyloNet jar'} | built "
              f"{time.strftime('%Y-%m-%d %H:%M')}",
              f"# problems: {len(problems)}; incomplete: {len(gaps)}"]
    text = render(ds, summary, header)
    (stem.parent / f"{stem.name}_summary.txt").write_text(text)
    rec = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": ds.name, "cells": len(cells),
           "rows": len(rows), "rows_by_status": dict(Counter(r["status"] for r in rows)),
           "complete": not gaps and not problems, "incomplete": gaps, "problems": problems,
           "runs": runs_used, "reference": str(sub_csv), "hybrid_ancestry": ha,
           "luay": None if luay is None else {
               "jar": str(args.phylonet), "measured": getattr(luay, "measured", None),
               "failed": getattr(luay, "failed", None)},
           "summary": summary, "seconds": round(time.perf_counter() - t0, 1)}
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"[ens] {len(rows)} rows -> {stem.with_suffix('.csv')}; sweep {len(sw)} rows")
    print(text)
    for p in problems[:40]:
        print(f"  PROBLEM {p}")
    print(f"[ens] {'clean' if not problems else f'{len(problems)} problems'}"
          f"{'' if not gaps else f', {len(gaps)} gaps (--allow-partial)'} in "
          f"{time.perf_counter() - t0:.0f} s")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
