#!/usr/bin/env python3
"""The l2_n1000_gt_50k_v1 per-network CSV, in l2reg's shape.

Rows (``gene_trees = est_bp1000_gt10k_n1000``, ``n = 1000``, ``network_idx = rep``):

    ours/l2mix_pair/selF1 [relaxed L2]   the GNN (runs/gnn*/), seed 0
    ours/l2mix_pair/selHA [relaxed L2]
    greedy, greedy [relaxed L2]          runs/greedy*/
    GNN TOB, true TOB*                   the trees of blobs, read as trees (k_pred = 0)
    BROOQS (GNN TOB), NetCS (GNN TOB)    resolvers on the selF1 network's tree of blobs
    BROOQS*(true TOB), NetCS*(true TOB)  the oracle arm, taken when it has been run (optional)

**This reuses l2reg's code rather than copying it.** ``l2reg_eval/make_csv.py`` is loaded by path
with ``common`` bound to *this* package's module (its ``import common as C`` finds this module in
``sys.modules``). That brings in its luay (PhyloNet ``CMPNETS``, sizes by star probes), its hybrid
ancestry (two derivations cross-checked), the m/tau gate (``sampling_problems``), and its GNN and
greedy row builders, unchanged; their docstrings are the definitions. Only the TOB and resolver rows
are written here, because the variants differ.

Columns: l2reg's (camus_a104's eleven, then ``EXTRA_COLUMNS`` and the hybrid-ancestry columns), and
then these:

    tob_fn, tob_fp     TOB rows and resolver rows: the true-TOB clusters this TOB misses and adds
    n_eligible         resolver rows: the polytomies that resolver can turn into a reticulation
    cpu_s              CPU seconds for the row's method on this network. For the GNN this is the
                       shared preparation plus the model's own forward and decode. For greedy it is
                       the wall time (one thread). For a resolver it is the child's CPU, from wait4.
    rss_mb             resolver rows: the child's peak RSS
    note               resolver rows: the tool note (rerooted:/swaps:), and any invariant violation

Gates (any failure is a problem, and nothing is written without ``--allow-partial``):

* ``stage.json`` stages this dataset.
* MANIFEST ``n_est_loci`` == ``common.M``.
* Every GNN and greedy cell read all 10,000 trees at tau = 20, and the GNN at tau-frac 0.002.
* The required GNN, greedy and GNN-TOB resolver cells are complete.
* Every resolver cell ran on the TOB that ``tob/gnn_tob.json`` records now.
* No resolver output violates an invariant.

Usage (``gnn_cpu_v1/.venv/bin/python``)
-----
    python n1000_eval/make_csv.py [--allow-partial] [--no-resolvers] [--out-csv P --out-json P]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import common as C

from opc.rl import camus_nets as CN  # noqa: E402
from opc.simulate import networks as N  # noqa: E402


def _load_l2reg_make_csv():
    """``l2reg_eval/make_csv.py``, with ``common`` resolving to this package (module docstring)."""
    if sys.modules.get("common") is not C:
        raise SystemExit("common is not n1000_eval/common.py; run this file directly")
    spec = importlib.util.spec_from_file_location("l2reg_make_csv",
                                                  C.REPO / "l2reg_eval" / "make_csv.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["l2reg_make_csv"] = mod
    spec.loader.exec_module(mod)
    if mod.C is not C:
        raise SystemExit("l2reg's make_csv bound a different common")
    return mod


MC = _load_l2reg_make_csv()

NEW_COLUMNS = ("tob_fn", "tob_fp", "n_eligible", "cpu_s", "rss_mb", "note")
COLUMNS = MC.CAMUS_COLUMNS + MC.EXTRA_COLUMNS + NEW_COLUMNS
METHOD_ORDER = ("GNN TOB", "true TOB*", "greedy", "ours/l2mix_pair/selF1", "ours/l2mix_pair/selHA",
                "BROOQS (GNN TOB)", "NetCS (GNN TOB)", "BROOQS*(true TOB)", "NetCS*(true TOB)")


def tob_clusters(rooted: str) -> set:
    net = N.from_enewick(rooted)
    return CN.restrict_clusters(net, CN.ingroup_of(net))


def cpu_of_cells(runs: list[Path], *, gnn: bool) -> dict:
    """(label, rep) -> CPU seconds, read off the same cells the rows came from."""
    out = {}
    for run in runs:
        for f in MC.cell_files(run):
            cell = json.loads(f.read_text())
            dec = C.cell_decoder(cell)
            for m in cell["models"]:
                if gnn:
                    label = C.GNN_LABELS.get(m["tag"])
                    if label is None:
                        continue
                    secs = sum((cell.get("cpu") or {}).values()) + sum((m.get("cpu") or {}).values())
                else:
                    label, secs = "greedy", sum(cell["wall"].values())
                out[(label + C.decoder_suffix(dec), int(cell["rep"]))] = secs
    return out


def tob_rows(stage_cells: list[dict], truths: dict, gnn_index: dict, tob_dir: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for rec in stage_cells:
        true_c = tob_clusters((tob_dir / f"{rec['key']}.true.rooted.nwk").read_text().strip())
        for v in C.VARIANTS:
            p = tob_dir / f"{rec['key']}.{v}.rooted.nwk"
            if not p.exists():
                problems.append(f"TOB: no {p.name}")
                continue
            rooted = p.read_text().strip()
            got = tob_clusters(rooted)
            counts = CN.cluster_counts(got, truths[rec["rep"]]["clusters"])
            row = MC.base_row(rec, C.TOB_LABELS[v], 0)
            n_blobs = (gnn_index.get(rec["key"], {}).get("n_blobs") if v == "gnn"
                       else rec["n_blobs"]["true"])
            row.update(cluster_f1=counts["f1"], cluster_f1_rooted=counts["f1"], k_pred=0,
                       n_blobs_tob=n_blobs, tob_fn=len(true_c - got), tob_fp=len(got - true_c),
                       **{c: counts[c] for c in MC.COUNTS})
            row["_pred"] = MC.pruned(rooted)
            rows.append(row)
    return rows, problems


def resolver_rows(run: Path, stage_cells: list[dict], gnn_index: dict, tob_dir: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for rec in stage_cells:
        true_c = tob_clusters((tob_dir / f"{rec['key']}.true.rooted.nwk").read_text().strip())
        for v in C.VARIANTS:
            f = run / "cells" / f"{rec['key']}.{v}.json"
            if not f.exists():
                if v in C.RESOLVER_VARIANTS:
                    problems.append(f"resolvers: no {f.name}")
                continue
            cell = json.loads(f.read_text())
            if v == "gnn":
                want = gnn_index.get(rec["key"], {}).get("tob_sha256")
                if cell["tob"]["sha256"] != want:
                    problems.append(f"resolvers: {f.name} ran on a TOB that gnn_tob.json no longer "
                                    f"records: re-run run_resolvers.py --keys {rec['rep']} --force")
            got = tob_clusters(cell["tob"]["newick_rooted"])
            for name in C.RESOLVERS:
                m = cell["methods"].get(name)
                if m is None:
                    problems.append(f"resolvers: {f.name} has no {name}")
                    continue
                row = MC.base_row(rec, C.RESOLVER_LABELS[(name, v)], 0)
                row.update(n_blobs_tob=cell["tob"]["n_blobs"], tob_fn=len(true_c - got),
                           tob_fp=len(got - true_c), n_eligible=cell["tob"]["n_eligible"][name],
                           cpu_s=m.get("cpu"), rss_mb=m.get("rss_mb"), note=m.get("note", ""))
                if m.get("violations"):
                    problems.append(f"resolvers: {f.name} {name}: {m['violations']}")
                    row["note"] = (row["note"] + " VIOLATION " + "; ".join(m["violations"])).strip()
                if m.get("not_run"):
                    # score-only on a partial pull: the tool had not finished when out/ was synced
                    problems.append(f"resolvers: {f.name} {name}: not run yet (no record)")
                if "error" in m:
                    row["status"] = ("timeout" if m.get("timed_out") else f"error: {m['error']}")
                elif m["out_rootable"]:
                    row.update(cluster_f1=m["f1"], cluster_f1_rooted=m["f1"],
                               k_pred=int(m["k_pred"]), **{c: m[c] for c in MC.COUNTS})
                    row["_pred"] = MC.pruned(m["enewick_rooted"])
                else:
                    row.update(cluster_f1=m["f1_as_emitted"], k_pred=int(m["k_pred"]),
                               status="out_not_rootable")
                    if m.get("enewick"):
                        row["_emitted"] = C.strip_gamma(m["enewick"])
                rows.append(row)
    return rows, problems


def build(args, jar: Path) -> int:
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run n1000_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    if stage["dataset"] != str(C.PKG):
        raise SystemExit(f"{C.STAGE} stages {stage['dataset']}, not {C.PKG}: re-run stage.py")
    loci = sorted({int(r["n_est_loci"]) for r in C.manifest()})
    if loci != [C.M]:
        raise SystemExit(f"MANIFEST n_est_loci {loci}; common.M is {C.M}")
    print(f"[csv] {C.PKG.name}: m = {C.M:,} trees, tau = {C.TAU} (tau-frac {C.TAU_FRAC}) -> "
          f"{args.out_csv}", flush=True)
    cells = stage["cells"]
    by_rep = {c["rep"]: c for c in cells}
    truths = {}
    for rec in cells:
        t = C.truth_of(rec["rep"])
        if t.n_retics != rec["k"]:
            raise SystemExit(f"{rec['key']}: truth n_retics {t.n_retics} != staged k {rec['k']}")
        enewick, ha = C.strip_gamma(N.to_enewick(t)), MC.hybrid_ancestry(t)
        if not ha or ha != MC.hybrid_ancestry_text(enewick):
            raise SystemExit(f"{rec['key']}: the truth's hybrid ancestry is empty, or its two "
                             f"derivations differ")
        truths[rec["rep"]] = {"enewick": enewick, "clusters": CN.label_clusters(t),
                              "taxa": set(t.labels), "ha": ha}
    index_p = Path(args.tob_dir) / "gnn_tob.json"
    gnn_index = json.loads(index_p.read_text())["cells"] if index_p.exists() else {}

    def runs_like(pattern: str) -> list[Path]:
        return sorted(p for p in C.RUNS.glob(pattern) if (p / "cells").is_dir())

    gnn_runs = [Path(p) for p in args.gnn_runs] if args.gnn_runs else runs_like("gnn*")
    greedy_runs = [Path(p) for p in args.greedy_runs] if args.greedy_runs else runs_like("greedy*")
    rows, problems = MC.gnn_rows(gnn_runs, by_rep)
    g_rows, g_prob = MC.greedy_rows(greedy_runs, by_rep)
    t_rows, t_prob = tob_rows(cells, truths, gnn_index, Path(args.tob_dir))
    # --no-resolvers: a GNN + greedy (+ TOB) build that is complete without them; the scope says so.
    r_rows, r_prob = (([], []) if args.no_resolvers else
                      resolver_rows(Path(args.resolver_run), cells, gnn_index, Path(args.tob_dir)))
    cpu = {**cpu_of_cells(gnn_runs, gnn=True), **cpu_of_cells(greedy_runs, gnn=False)}
    for row in rows + g_rows:
        row["cpu_s"] = cpu.get((row["method"], row["network_idx"]))
    rows += g_rows + t_rows + r_rows
    problems += g_prob + t_prob + r_prob
    if problems:
        print("[csv] incomplete:\n" + "\n".join(f"  {p}" for p in problems))
        if not args.allow_partial:
            print("[csv] nothing written; --allow-partial writes what exists")
            return 1

    ha_stats, ha_problems = MC.attach_hybrid_ancestry(rows, truths)
    problems += ha_problems
    print(f"[ha] hybrid ancestry: the two derivations agree on {ha_stats['agree']} of "
          f"{ha_stats['networks']} networks; {ha_stats['rows']} rows scored", flush=True)
    MC.attach_luay(rows, truths, MC.PhyloNetLuay(jar))
    over = [r for r in rows if not MC.missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    if over:
        problems.append(f"{len(over)} rows with luay_norm_exact > 1")

    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (MC.row_order(r, order), r["network_idx"], r["seed"]))
    out_csv, out_json = Path(args.out_csv), Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([MC.fmt(r.get(c)) for c in COLUMNS])

    summary = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "sampling": {"m": C.M, "tau": C.TAU, "tau_frac": C.TAU_FRAC,
                     "resolver_variants": [] if args.no_resolvers else list(C.RESOLVER_VARIANTS),
                     "gnn_required": C.GNN_REQUIRED, "greedy_required": list(C.GREEDY_REQUIRED)},
        "stage": {"built": stage["built"],
                  "dataset_manifest_sha256": stage["dataset_manifest_sha256"]},
        "gnn_tob": {"source": C.TOB_SOURCE, "index": str(index_p),
                    "built": json.loads(index_p.read_text())["built"] if index_p.exists() else None},
        "rows": len(rows), "complete": not problems, "problems": problems,
        "scope": ("gnn, greedy, tob (resolvers not included: --no-resolvers)" if args.no_resolvers
                  else "gnn, greedy, tob, resolvers"),
        "luay": {"engine": "PhyloNet CMPNETS -m luay", "jar": str(jar)},
        "hybrid_ancestry": ha_stats,
        "runs": {"gnn": [str(p) for p in gnn_runs], "greedy": [str(p) for p in greedy_runs],
                 "resolvers": None if args.no_resolvers else str(args.resolver_run)},
        "labels": {"gnn": C.GNN_LABELS, "tob": C.TOB_LABELS,
                   "resolvers": {f"{a}/{b}": v for (a, b), v in C.RESOLVER_LABELS.items()}},
        "per_method": MC.per_method(rows, order),
    }
    out_json.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"[csv] {len(rows)} rows -> {out_csv}\n[csv] summary -> {out_json}")
    for label, s in summary["per_method"].items():
        f1, ln, ber = (s[f"median_{c}"] for c in ("cluster_f1", "luay_norm_exact",
                                                   "ha_balanced_error"))
        print(f"  {label:36s} {s['rows']:3d} rows | F1 med {'' if f1 is None else f'{f1:.4f}':6s}"
              f" | luay_norm_exact med {'' if ln is None else f'{ln:.4f}':6s}"
              f" | ha BER med {'' if ber is None else f'{ber:.4f}':6s} | "
              f"k_pred med {s['median_k_pred']} (k {s['median_k_true']}) | {s['status']}")
    return 0 if not problems else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET") or MC.DEFAULT_JAR))
    ap.add_argument("--gnn-runs", nargs="*", default=None)
    ap.add_argument("--greedy-runs", nargs="*", default=None)
    ap.add_argument("--resolver-run", type=Path, default=C.RUNS / "resolvers")
    ap.add_argument("--tob-dir", type=Path, default=C.TOB)
    ap.add_argument("--out-csv", type=Path, default=C.LANDED / "l2_n1000_gt50k_v1_per_network.csv")
    ap.add_argument("--out-json", type=Path, default=C.LANDED / "l2_n1000_gt50k_v1.json")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--no-resolvers", action="store_true",
                    help="build without resolver rows (GNN, greedy and TOB rows only); complete "
                         "without them, and the summary's scope says so")
    args = ap.parse_args()
    jar = args.phylonet.expanduser().resolve()
    if not jar.exists():
        raise SystemExit(f"no PhyloNet jar at {jar}: pass --phylonet or set OPC_PHYLONET")
    return build(args, jar)


if __name__ == "__main__":
    raise SystemExit(main())
