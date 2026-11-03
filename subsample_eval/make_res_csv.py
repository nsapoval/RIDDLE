#!/usr/bin/env python3
"""The resolver rows of the subsample study, one CSV per dataset, with the full-data rows beside them.

Rows
----
One per (draw, cell, method).  ``draw`` is ``full`` or ``0``..``4``.  ``method`` is the landed label:

| dataset | subsample rows (``run_res.py``) | ``full`` rows (the landed resolver runs) |
|---|---|---|
| camus | ``BROOQS``, ``NetCS`` | the same (``results/rl/rl_camus_*_tqmctob_tau2.parquet``, Linux) |
| l2reg, l3reg | ``BROOQS``, ``NetCS``, ``... (fixed TOB)`` | the same (``<ds>_eval/runs/resolvers/``) |
| netcs | ``BROOQS*``, ``NetCS*`` -- rooted, oracle OUT | ``BROOQS``, ``NetCS`` -- unrooted (``netcs_eval/runs/resolvers/``) |

netcs's two protocols differ (``run_res.py`` docstring), so its sub - full deltas are **cross-protocol**
and the summary says so on every such line.

The ``full`` rows are re-scored here from the landed outputs by the landed scoring
(``pte_bench.score``: ``run_blob_cpu.score``, with OUT placed by the true-root oracle on netcs), so a
paired difference is between two numbers computed one way -- and the parity gate below checks that
this reproduces the landed CSV.

Columns
-------
``make_sub_csv.py``'s identity and score columns, in its order, so the two CSVs concatenate:
``n_clusters`` is empty (a resolver has no candidate set), ``decoder`` is empty, ``tag`` is the
resolver, ``prep_wall`` is empty (the TOB was built on the Linux box: ``tob_seconds``) and
``model_wall`` is the resolver's wall time.  Then:

* ``variant`` (raw / fixed), ``protocol`` (rooted / unrooted), ``cond`` (the landed condition key);
* ``n_elig`` (the TOB's polytomies the resolver would resolve), ``n_blobs_tob`` (its polytomies);
* ``out_rootable``, ``f1_as_emitted``, ``cluster_f1_rooted`` -- the landed resolver semantics: an
  output with OUT below one of its own hybrids has no OUT-rooted reading, so ``cluster_f1`` is the
  as-emitted value, ``cluster_f1_rooted`` and luay are empty, ``status = out_not_rootable``, and
  hybrid ancestry is read off the network as emitted (it needs no root);
* ``resolver_cpu``, ``resolver_rss_mb``, ``timed_out``, ``note``, ``root_cost`` (netcs full only);
* the TOB's own quality, from its JSON: ``tob_true_edges, tob_est_edges, tob_fn, tob_fp,
  tob_fn_raw, tob_fp_raw, tob_a63, tob_seconds``;
* ``violations`` (hard invariants; must be empty) and ``task_id``.

Scoring
-------
luay, both masses and hybrid ancestry are ``make_sub_csv.py``'s, imported: ``PhyloNetLuay`` on the
shared ``runs/phylonet/cache.jsonl``, luay on ``pruned(enewick_rooted)`` -- OUT pruned, gammas
stripped, the landed ``pruned()`` -- and the two hybrid-ancestry derivations, which must agree.

Checks
------
* **Completeness** (nothing is written unless ``--allow-partial``): a task record is missing.  Records
  are looked up by content id, re-derived from the current TOB and gene-tree files, so a record made
  from other inputs is not found and counts as missing.
* **Integrity** (written; exit 2): ``m != 500``; a record scored by another ``SCORE_VERSION``; a hard
  invariant violated; the two hybrid-ancestry derivations disagree; ``luay_norm_exact > 1``.
* **Parity:** every ``full`` row must reproduce the landed CSV's ``cluster_f1, cluster_f1_rooted,
  tp, fp, fn, precision, recall, k_pred, luay, luay_norm, luay_norm_exact, mass_pred, mass_truth,
  ha_*, status`` as it prints them; a field the landed row leaves empty is not compared.
  ``ALLOWED`` lists the known, explained misses.

Outputs
-------
``results/<ds>_sub500_res.{csv,json}`` and ``results/<ds>_sub500_res_summary.txt``: per label, the
full median, the pooled and per-draw subsample medians, the per-cell delta (mean over draws - full),
the fractions of cells worse and better, and the draw SD, for ``cluster_f1`` (higher is better),
``luay_norm_exact`` and ``ha_balanced_error`` (lower); then statuses and k; then each resolver
against ``ours/l2mix_pair/selF1`` on the same (draw, cell), from ``results/<ds>_sub500.csv``;
a tie counts for neither side, so the two directions do not sum to 100%.

**Never run beside ``make_sub_csv.py`` or ``make_ens_csv.py``**: all three append to one cache.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/make_res_csv.py --dataset l2reg
    python subsample_eval/make_res_csv.py --dataset netcs --jobs 4
    python subsample_eval/make_res_csv.py --dataset camus --draws full       # parity alone
"""

from __future__ import annotations

import run_res as R  # first: common (BLAS pin, library path), then prefix_tob_eval's pte_*

import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import common as C
import make_sub_csv as S

P, B = R.P, R.B

RESOLVER_COLUMNS = ("variant", "protocol", "cond", "n_elig", "n_blobs_tob", "out_rootable",
                    "f1_as_emitted", "cluster_f1_rooted", "resolver_cpu", "resolver_rss_mb",
                    "timed_out", "note", "root_cost", "tob_true_edges", "tob_est_edges", "tob_fn",
                    "tob_fp", "tob_fn_raw", "tob_fp_raw", "tob_a63", "tob_seconds", "violations",
                    "task_id")
PARITY_FIELDS = (*S.PARITY_FIELDS, "cluster_f1_rooted", "status")
#: (dataset, phase, rep, method) -> the fields a known, explained miss may differ in.
#: cam150 rep04 is make_sub_csv's: a size-2-cycle truth the jar sizes differently from Linux.
ALLOWED: dict[tuple, set[str]] = {("camus", "cam150", 4, m): {"mass_truth", "luay_norm_exact"}
                                  for m in ("BROOQS", "NetCS")}
COUNTS = ("n_selected", "tp", "fp", "fn", "precision", "recall")
GNN = "ours/l2mix_pair/selF1"


def full_label(ds: str, sub_label: str) -> str:
    """The landed label a subsample label is paired with: itself, but netcs's ``X*`` -> ``X``."""
    return sub_label.replace("*", "") if ds == "netcs" else sub_label


def protocol(ds: str, draw) -> str:
    return "unrooted" if ds == "netcs" and draw == "full" else "rooted"


# ------------------------------------------------------------------------------------------- rows

def base_row(ds: C.Dataset, draw, c: dict, truth: dict, label: str, variant: str, resolver: str,
             m: int, seed) -> dict:
    return {"dataset": ds.name, "draw": str(draw), "sub_seed": seed, "m": m,
            "n_loci_source": c["n_loci_source"], "phase": c["phase"], "rep": c["rep"],
            "arm": c["arm"], "key": C_key(ds, c), "n": truth["n"], "n_taxa": truth["n_taxa"],
            "k_true": truth["k"], **{s: c[s] for s in S.STRATA[ds.name]},
            "decoder": "", "method": label, "_base": label, "tag": resolver,
            "variant": variant, "protocol": protocol(ds.name, draw),
            "cond": R.cond_key(ds.name, c["phase"], c["rep"], c["arm"]), "status": "ok"}


def C_key(ds: C.Dataset, c: dict) -> str:
    """``CamusCell.key`` (``n100_rep00``): the ``key`` column make_sub_csv's rows carry."""
    return B.lib().CAMUS.CamusCell(c["phase"], c["rep"], c["arm"], ds.source).key


def apply_score(row: dict, s: dict | None, enewick: str | None) -> None:
    """The landed ``resolver_rows`` semantics (``netcs_eval/make_csv.py:442``)."""
    if s is None:
        return
    row["k_pred"] = int(s["k_pred"])
    row["out_rootable"] = s["out_rootable"]
    row["f1_as_emitted"] = s["f1_as_emitted"]
    if "root_cost" in s:
        row["root_cost"] = s["root_cost"]
    if s["out_rootable"]:
        row.update(cluster_f1=s["f1"], cluster_f1_rooted=s["f1"], **{k: s[k] for k in COUNTS})
        row["_pred"] = pruned(s["enewick_rooted"])
    else:
        row.update(cluster_f1=s["f1_as_emitted"], status="out_not_rootable")
        if enewick:
            row["_emitted"] = S.strip_gamma(enewick)


def pruned(rooted: str) -> str:
    """``l2reg_eval/make_csv.py:pruned``: OUT pruned, gammas stripped -- what the landed luay compared."""
    from opc.rl.external import _prune_outgroup
    L = B.lib()
    return S.strip_gamma(L.N.to_enewick(_prune_outgroup(L.N.from_enewick(rooted))))


def tob_fields(meta: dict) -> dict:
    return {"tob_true_edges": meta.get("n_true_edges"), "tob_est_edges": meta.get("n_est_edges"),
            "tob_fn": meta.get("fn"), "tob_fp": meta.get("fp"), "tob_fn_raw": meta.get("fn_raw"),
            "tob_fp_raw": meta.get("fp_raw"),
            "tob_a63": meta.get("treeqmc_kept_a_contracted_root_branch"),
            "tob_seconds": meta.get("seconds")}


def sub_rows(ds: C.Dataset, draws: list[int], cells: list[dict], truths: dict, runs_root: Path,
             gaps: list[str], problems: list[str]) -> list[dict]:
    names = {c["name"]: c for c in cells}
    tasks = R.build_tasks(ds.name, set(names), set(draws))
    index = {(t["draw"], t["cell"], v, t["resolver"]): t for t in tasks for v in t["variants"]}
    manifest = {(int(r["draw"]), r["cell"]): r for r in C.read_tsv(ds.manifest)}
    rows = []
    for draw in draws:
        for name, c in names.items():
            mrow = manifest.get((draw, name))
            if mrow is None:
                problems.append(f"d{draw} {name}: no MANIFEST row")
                continue
            meta = json.loads(R.tob_paths(ds.name, draw, name)["json"].read_text())
            truth = truths[(c["phase"], c["rep"])]
            for variant in R.VARIANTS[ds.name]:
                for resolver in P.RESOLVERS:
                    t = index[(draw, name, variant, resolver)]
                    f = runs_root / ds.name / "tasks" / f"{t['task_id']}.json"
                    if not f.exists():
                        gaps.append(f"d{draw} {name} {variant} {resolver}: no task record")
                        continue
                    rec = json.loads(f.read_text())
                    lab = R.label(ds.name, variant, resolver)
                    row = base_row(ds, draw, c, truth, lab, variant, resolver, int(mrow["m"]),
                                   int(mrow["seed"]))
                    where = f"d{draw} {name} {lab}"
                    if int(mrow["m"]) != C.N_SUB:
                        problems.append(f"{where}: m = {mrow['m']}")
                    if rec.get("score_version") != R.SCORE_VERSION:
                        problems.append(f"{where}: scored by version {rec.get('score_version')}")
                    st = P.tob_stats(t["tob_text"])
                    row.update(n_elig=rec["n_elig"], n_blobs_tob=st["n_poly"],
                               task_id=t["task_id"], status=rec["status"], **tob_fields(meta))
                    tm = rec.get("timing", {})
                    row.update(model_wall=tm.get("wall"), resolver_cpu=tm.get("cpu"),
                               resolver_rss_mb=tm.get("rss_mb"), timed_out=tm.get("timed_out"),
                               note=tm.get("note"))
                    if rec.get("violations"):
                        row["violations"] = "; ".join(rec["violations"])
                        problems.append(f"{where}: hard invariant violated: {rec['violations']}")
                    apply_score(row, rec.get("score"), rec.get("enewick"))
                    rows.append(row)
    return rows


# --------------------------------------------------------------------------------- full-data rows

def landed_outputs(ds: str) -> dict[tuple[str, str, str], dict]:
    """(condition key, variant, resolver) -> {enewick, error, wall, cpu, rss_mb, note} of the landed
    1,000-tree resolver run."""
    out = {}
    if ds == "camus":
        import pyarrow.parquet as pq
        km = B.key_map(ds)
        for r in ("brooqs", "netcs"):
            t = pq.read_table(C.LANDED / f"rl_camus_{r}_tqmctob_tau2.parquet").to_pylist()
            for x in t:
                key = km[(x["phase"], int(x["rep"]), x["gene_trees"])]
                text = (x.get("enewick_raw") or "").strip()
                out[(key, "raw", r)] = {"enewick": text or None,
                                        "error": None if text else f"landed status {x['status']}",
                                        "wall": float(x["seconds"]) if x.get("seconds") else None}
        return out
    cells = C.REPO / f"{ds}_eval" / "runs" / "resolvers" / "cells"
    for f in sorted(cells.glob("*.json")):
        rec = json.loads(f.read_text())
        variant = rec.get("variant", "raw")
        if variant not in R.VARIANTS[ds]:
            continue                                      # l2reg/l3reg's true-TOB oracle rows
        for r, m in rec["methods"].items():
            out[(rec["key"], variant, r)] = {
                "enewick": m.get("enewick"), "error": m.get("error"), "wall": m.get("wall"),
                "cpu": m.get("cpu"), "rss_mb": m.get("rss_mb"), "note": m.get("note")}
    return out


def landed_fed_tob(ds: str, key: str, variant: str) -> str | None:
    """The tree of blobs the landed run was fed, where it is on this box."""
    if ds in ("l2reg", "l3reg"):
        f = C.REPO / f"{ds}_eval" / "tob" / f"{key}.{variant}.rooted.nwk"
    elif ds == "netcs":
        f = C.REPO / "netcs_eval" / "tob" / f"{key}.tqmc.nwk"
    else:
        return None                   # camus: the Linux TOB is not here (blob_cpu_v1's is the Mac's)
    return f.read_text().strip() if f.exists() else None


def full_rows(ds: C.Dataset, cells: list[dict], truths: dict, gaps: list[str],
              problems: list[str]) -> list[dict]:
    landed = landed_outputs(ds.name)
    rows = []
    for c in cells:
        key = R.cond_key(ds.name, c["phase"], c["rep"], c["arm"])
        cond = B.conditions(ds.name)[key]
        truth = truths[(c["phase"], c["rep"])]
        for variant in R.VARIANTS[ds.name]:
            fed = landed_fed_tob(ds.name, key, variant)
            st = P.tob_stats(fed) if fed else None
            for resolver in P.RESOLVERS:
                lab = full_label(ds.name, R.label(ds.name, variant, resolver))
                x = landed.get((key, variant, resolver))
                if x is None:
                    gaps.append(f"full {key} {variant} {resolver}: no landed output")
                    continue
                row = base_row(ds, "full", c, truth, lab, variant, resolver,
                               c["n_loci_source"], "")
                row.update(model_wall=x.get("wall"), resolver_cpu=x.get("cpu"),
                           resolver_rss_mb=x.get("rss_mb"), note=x.get("note"))
                if st is not None:
                    row.update(n_blobs_tob=st["n_poly"], n_elig=st["elig"][P.rule_of(resolver, None)])
                if x["error"]:
                    row["status"] = f"error: {x['error']}"
                elif not x["enewick"]:
                    row["status"] = "error: no output"
                else:
                    try:
                        s = B.score(ds.name, x["enewick"], cond)
                    except Exception as exc:
                        row["status"] = f"error: score: {type(exc).__name__}: {exc}"
                        rows.append(row)
                        continue
                    apply_score(row, s, x["enewick"])
                rows.append(row)
    return rows


# ------------------------------------------------------------------------------ hybrid ancestry

def attach_hybrid_ancestry(rows: list[dict], truths: dict, problems: list[str]) -> dict:
    """``make_sub_csv.attach_hybrid_ancestry`` plus the landed ``_emitted`` rule: an
    out_not_rootable output is read as emitted.  Runs before ``attach_luay`` pops ``_pred``."""
    stats = Counter(networks=0, agree=0)
    for row in rows:
        emitted = row.pop("_emitted", None)
        src = row.get("_pred") or emitted
        if src is None:
            continue
        truth = truths[(row["phase"], row["rep"])]
        where = f"hybrid ancestry: {row['draw']} {row['method']} {row['phase']} rep{row['rep']} {row['arm']}"
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


# ------------------------------------------------------------------------------------------ parity

def landed_index(ds: C.Dataset) -> tuple[dict, set[str]]:
    labels = {R.label(ds.name, v, r).replace("*", "") for v in R.VARIANTS[ds.name] for r in P.RESOLVERS}
    with open(ds.landed_csv, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = set(rd.fieldnames or ())
        idx = {S.landed_key(ds.name, r): r for r in rd if r["method"] in labels}
    return idx, cols


def parity(ds: C.Dataset, rows: list[dict], problems: list[str]) -> dict:
    idx, cols = landed_index(ds)
    fields = [f for f in PARITY_FIELDS if f in cols]
    stats, hits = Counter(), Counter()
    for row in rows:
        if row["draw"] != "full":
            continue
        land = idx.get(S.landed_key(ds.name, row))
        if land is None:
            stats["no_landed_row"] += 1
            row["landed"] = ""
            continue
        allowed = ALLOWED.get((ds.name, row["phase"], row["rep"], row["method"]), set())
        bad = []
        for f in fields:
            if land.get(f, "") == "":
                continue
            hits[f] += 1
            if S.fmt(row.get(f)) != land[f] and f not in allowed:
                bad.append(f"{f} {S.fmt(row.get(f))!r} vs landed {land[f]!r}")
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
            "fields_compared": dict(hits),
            "allowed": {f"{k[1]} rep{k[2]} {k[3]}": sorted(v) for k, v in ALLOWED.items()
                        if k[0] == ds.name}}


# ----------------------------------------------------------------------------------------- summary

def summarise(ds: C.Dataset, rows: list[dict], sub_draws: list[int], gnn_csv: Path) -> dict:
    main = [r for r in rows if ds.name != "camus" or r["arm"] == "cam_g500"]
    sub_labels = sorted({r["method"] for r in rows if r["draw"] != "full"})
    num = S._num

    def block(rs: list[dict]) -> dict:
        out = {}
        for lab in sub_labels:
            flab = full_label(ds.name, lab)
            by_cell: dict[tuple, dict] = defaultdict(dict)
            for r in rs:
                ident = (r["phase"], r["rep"], r["arm"])
                if r["draw"] == "full" and r["method"] == flab:
                    by_cell[ident]["full"] = r
                elif r["draw"] != "full" and r["method"] == lab:
                    by_cell[ident][r["draw"]] = r
            paired = {k: v for k, v in by_cell.items()
                      if "full" in v and any(str(d) in v for d in sub_draws)}
            rec = {"full_label": flab, "cross_protocol": flab != lab, "cells": len(paired)}
            for metric, better in S.METRICS:
                full = [num(v["full"].get(metric)) for v in paired.values()]
                per_draw = {d: S._med([num(v[str(d)].get(metric)) for v in paired.values()
                                       if str(d) in v]) for d in sub_draws}
                pooled = S._med([num(v[str(d)].get(metric)) for v in paired.values()
                                 for d in sub_draws if str(d) in v])
                deltas, sds = [], []
                for v in paired.values():
                    f0 = num(v["full"].get(metric))
                    xs = [num(v[str(d)].get(metric)) for d in sub_draws if str(d) in v]
                    xs = [x for x in xs if x is not None]
                    if f0 is None or not xs:
                        continue
                    deltas.append(statistics.fmean(xs) - f0)
                    if len(xs) >= 2:
                        sds.append(statistics.stdev(xs))
                worse = [d for d in deltas if (d < 0 if better == "higher" else d > 0)]
                better_ = [d for d in deltas if (d > 0 if better == "higher" else d < 0)]
                rec[metric] = {"full_median": S._med(full), "sub_median": pooled,
                               "sub_median_per_draw": per_draw, "delta_median": S._med(deltas),
                               "delta_mean": statistics.fmean(deltas) if deltas else None,
                               "cells_paired": len(deltas),
                               "frac_worse": len(worse) / len(deltas) if deltas else None,
                               "frac_better": len(better_) / len(deltas) if deltas else None,
                               "within_cell_sd_mean": statistics.fmean(sds) if sds else None}
            out[lab] = rec
        return out

    def outcomes(rs: list[dict]) -> dict:
        out = {}
        for lab in sorted({r["method"] + ("" if r["draw"] != "full" else " [full]") for r in rs}):
            base, full = (lab[:-7], True) if lab.endswith(" [full]") else (lab, False)
            xs = [r for r in rs if r["method"] == base and (r["draw"] == "full") == full]
            ok = [r for r in xs if r["status"] in ("ok", "out_not_rootable")]
            out[lab] = {"rows": len(xs), "status": dict(Counter(r["status"][:40] for r in xs)),
                        "k_pred_mean": statistics.fmean(int(r["k_pred"]) for r in ok) if ok else None,
                        "k_true_mean": statistics.fmean(int(r["k_true"]) for r in ok) if ok else None,
                        "frac_k_eq_k": (sum(int(r["k_pred"]) == int(r["k_true"]) for r in ok) / len(ok))
                        if ok else None,
                        "cpu_h": sum(num(r.get("resolver_cpu")) or 0 for r in xs) / 3600}
        return out

    strata = defaultdict(list)
    for r in rows:
        strata[S.stratum_of(ds.name, r)].append(r)
    return {"main_cells": "cam_g500 only (the headline 80)" if ds.name == "camus" else "all",
            "main": block(main), "outcomes": outcomes(main),
            "by_stratum": {s: {m: v["cluster_f1"] for m, v in block(rs).items()}
                           for s, rs in sorted(strata.items())},
            "vs_gnn": vs_gnn(ds, main, gnn_csv)}


def vs_gnn(ds: C.Dataset, rows: list[dict], gnn_csv: Path) -> dict | None:
    """Each resolver label against ``ours/l2mix_pair/selF1`` (level-1 and relaxed) on the same
    (draw, cell): median of resolver - GNN, and the fraction of pairs the GNN wins."""
    if not gnn_csv.exists():
        return None
    gnn: dict[tuple, dict] = {}
    with open(gnn_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["method"].startswith(GNN) and (ds.name != "camus" or r["arm"] == "cam_g500"):
                gnn[(r["method"], r["draw"], r["phase"], int(r["rep"]), r["arm"])] = r
    gnn_labels = sorted({k[0] for k in gnn})
    out = {}
    for lab in sorted({(r["method"], r["draw"] == "full") for r in rows}):
        method, is_full = lab
        xs = [r for r in rows if r["method"] == method and (r["draw"] == "full") == is_full]
        for g in gnn_labels:
            rec = {"pairs": 0}
            for metric, better in S.METRICS:
                diffs = []
                for r in xs:
                    o = gnn.get((g, r["draw"], r["phase"], r["rep"], r["arm"]))
                    if o is None:
                        continue
                    # The GNN side is read from a written CSV, so it carries ``fmt``'s %g rounding.
                    # Round ours the same way before differencing: full precision against six
                    # significant digits turns an exact tie into a win.  Measured 2026-09-23 on
                    # camus BROOQS all-trees cluster F1 -- 3 of 6 ties counted as GNN wins, 0.7375
                    # against the true 0.700.
                    a, b = S._num(S.fmt(r.get(metric))), S._num(o.get(metric))
                    if a is None or b is None:
                        continue
                    diffs.append(a - b)
                gnn_wins = [d for d in diffs if (d < 0 if better == "higher" else d > 0)]
                rec[metric] = {"pairs": len(diffs), "median_res_minus_gnn": S._med(diffs),
                               "frac_gnn_better": len(gnn_wins) / len(diffs) if diffs else None}
                rec["pairs"] = max(rec["pairs"], len(diffs))
            out[f"{method} {'[full]' if is_full else '[sub]'} vs {g}"] = rec
    return out


def render(ds: C.Dataset, summary: dict, sub_draws: list[int], header: list[str]) -> str:
    def f(x, w=7, p=4):
        return f"{'':>{w}}" if x is None else f"{x:{w}.{p}f}"

    lines = [*header, ""]
    for metric, better in S.METRICS:
        lines.append(f"## {metric} ({better} is better) -- {summary['main_cells']}; delta = mean over "
                     f"a cell's draws - its full value")
        lines.append(f"{'method (vs full label)':40s} {'cells':>5s} {'full':>7s} {'sub':>7s} "
                     f"{'delta md':>8s} {'delta mn':>8s} {'worse':>6s} {'better':>6s} {'draw SD':>7s}"
                     f"  per-draw medians")
        for lab, rec in summary["main"].items():
            m = rec[metric]
            name = lab if not rec["cross_protocol"] else f"{lab} (vs {rec['full_label']}: X-PROTOCOL)"
            draws = " ".join(f(m["sub_median_per_draw"].get(d), 6, 3) for d in sub_draws)
            lines.append(f"{name:40s} {rec['cells']:5d} {f(m['full_median'])} {f(m['sub_median'])} "
                         f"{f(m['delta_median'], 8)} {f(m['delta_mean'], 8)} "
                         f"{f(m['frac_worse'], 6, 2)} {f(m['frac_better'], 6, 2)} "
                         f"{f(m['within_cell_sd_mean'])}  {draws}")
        lines.append("")
    lines.append("## outcomes and budget")
    lines.append(f"{'method':32s} {'rows':>5s} {'k=k_true':>8s} {'k mean':>7s} {'k_true':>7s} "
                 f"{'CPU h':>7s}  statuses")
    for lab, rec in summary["outcomes"].items():
        lines.append(f"{lab:32s} {rec['rows']:5d} {f(rec['frac_k_eq_k'], 8, 3)} "
                     f"{f(rec['k_pred_mean'], 7, 2)} {f(rec['k_true_mean'], 7, 2)} "
                     f"{f(rec['cpu_h'], 7, 2)}  {rec['status']}")
    lines.append("")
    lines.append("## cluster_f1 delta median (cells) by stratum")
    labels = list(summary["main"])
    lines.append(f"{'stratum':14s} " + " ".join(f"{lab[:26]:>26s}" for lab in labels))
    for s, recs in summary["by_stratum"].items():
        lines.append(f"{s:14s} " + " ".join(
            f"{(f(recs[lab]['delta_median'], 8) + ' (' + str(recs[lab]['cells_paired']) + ')') if lab in recs else '':>26s}"
            for lab in labels))
    if summary.get("vs_gnn"):
        lines.append("")
        lines.append(f"## resolver - {GNN} on the same (draw, cell): median difference, and the "
                     f"fraction of pairs the GNN wins")
        lines.append(f"{'pair':78s} {'pairs':>5s} " + " ".join(
            f"{m:>17s} {'GNN wins':>8s}" for m, _ in S.METRICS))
        for name, rec in summary["vs_gnn"].items():
            lines.append(f"{name:78s} {rec['pairs']:5d} " + " ".join(
                f"{f(rec[m]['median_res_minus_gnn'], 17)} {f(rec[m]['frac_gnn_better'], 8, 2)}"
                for m, _ in S.METRICS))
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(C.DATASETS))
    ap.add_argument("--draws", default="", help="comma list of full and draw numbers (default: all)")
    ap.add_argument("--phases", default="")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges")
    ap.add_argument("--cells", default="", help="comma list of cell names")
    ap.add_argument("--runs-root", type=Path, default=R.RES, help="run_res.py's --out")
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS)
    ap.add_argument("--gnn-csv", type=Path, default=None,
                    help="make_sub_csv's CSV for the GNN comparison (default results/<ds>_sub500.csv)")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET", str(C.DEFAULT_JAR))))
    ap.add_argument("--jobs", type=int, default=4, help="PhyloNet JVMs at once")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--no-cache", action="store_true")
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
    cells = S.load_cells(ds, {p for p in args.phases.split(",") if p}, S.parse_reps(args.reps))
    want = set(filter(None, args.cells.split(",")))
    if want:
        cells = [c for c in cells if c["name"] in want]
    if not cells:
        raise SystemExit("no cells selected")

    problems: list[str] = []
    gaps: list[str] = []
    truths = S.load_truths(ds, cells, problems)
    rows = []
    if "full" in draws:
        rows += full_rows(ds, cells, truths, gaps, problems)
    if sub_draws:
        rows += sub_rows(ds, sub_draws, cells, truths, args.runs_root, gaps, problems)
    if gaps:
        print("[csv] incomplete:\n" + "\n".join(f"  {g}" for g in gaps[:30])
              + (f"\n  ... and {len(gaps) - 30} more" if len(gaps) > 30 else ""))
        if not args.allow_partial:
            print("[csv] nothing written; --allow-partial writes what exists")
            return 1
    if not rows:
        raise SystemExit("[csv] no rows")

    ha = attach_hybrid_ancestry(rows, truths, problems)
    print(f"[ha] the two derivations agree on {ha.get('agree', 0)} of {ha.get('networks', 0)} networks")
    cache_dir = C.RUNS / "phylonet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    luay = S.PhyloNetLuay(args.phylonet, cache=None if args.no_cache else cache_dir / "cache.jsonl",
                          seeds=() if args.no_cache else C.SEED_CACHES.get(ds.name, ()),
                          jobs=args.jobs, chunk=args.chunk)
    S.attach_luay(rows, truths, luay, cache_dir / "work")
    over = [r for r in rows if not S.missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    if over:
        problems.append(f"{len(over)} rows with luay_norm_exact > 1, e.g. "
                        f"{[(r['draw'], r['method'], r['key']) for r in over[:3]]}")
    par = parity(ds, rows, problems) if "full" in draws else None

    draw_rank = lambda d: -1 if d == "full" else int(d)  # noqa: E731
    rows.sort(key=lambda r: (r["tag"], r["variant"], r["phase"], r["rep"], r["arm"],
                             draw_rank(r["draw"])))
    ident = ["dataset", "draw", "sub_seed", "m", "n_loci_source", "phase", "rep", "arm", "key", "n",
             "n_taxa", "k_true", *S.STRATA[ds.name], "decoder", "method", "tag"]
    columns = ident + list(S.SCORE_COLUMNS) + list(RESOLVER_COLUMNS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{ds.name}_sub{C.N_SUB}_res"
    out_csv = stem.with_suffix(".csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([S.fmt(r.get(c)) for c in columns])

    gnn_csv = args.gnn_csv or (C.RESULTS / f"{ds.name}_sub{C.N_SUB}.csv")
    summary = summarise(ds, rows, sub_draws, gnn_csv) if sub_draws and "full" in draws else None
    header = [f"# {ds.name}: BROOQS and NetCS on TREE-QMC trees of blobs of {C.N_SUB} of each cell's "
              f"gene trees, against the landed full-data resolver rows",
              f"# draws {', '.join(C.draw_label(d) for d in draws)} | {len(cells)} cells | "
              f"{len(rows)} rows | built {time.strftime('%Y-%m-%d %H:%M')}",
              f"# parity vs {ds.landed_csv.name}: " + (
                  f"{par['rows_compared'] - par['rows_mismatched']}/{par['rows_compared']} full rows "
                  f"match; no landed row for {par['rows_without_landed_row'] or 'none'}"
                  if par else "not run (no full draw)"),
              f"# problems: {len(problems)}; incomplete: {len(gaps)}"]
    if ds.name == "netcs":
        header.append("# netcs: subsample rows are ROOTED (oracle OUT, BROOQS* / NetCS*); the full rows "
                      "are the landed UNROOTED ones -- every sub - full delta crosses protocols")
    if summary:
        (stem.parent / f"{stem.name}_summary.txt").write_text(render(ds, summary, sub_draws, header))
    rec = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": ds.name, "m": C.N_SUB,
           "draws": [C.draw_label(d) for d in draws], "cells": len(cells), "rows": len(rows),
           "rows_by_status": dict(Counter(r["status"][:40] for r in rows)),
           "complete": not gaps and not problems, "incomplete": gaps, "problems": problems,
           "runs_root": str(args.runs_root), "parity": par, "hybrid_ancestry": ha,
           "luay": {"jar": str(args.phylonet), "cache": str(cache_dir / "cache.jsonl"),
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
