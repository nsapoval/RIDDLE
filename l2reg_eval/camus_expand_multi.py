#!/usr/bin/env python3
"""Widen ``camus_a104_per_network.csv`` with several gnn_cpu_v1 camus runs at once -- luay by PhyloNet.

``camus_expand.py`` appends **one** relaxed run and emits the camus eleven.  This does the same
scoring for **any number** of runs, formal or relaxed, and writes a single wide table: the camus
eleven exactly as camus_a104 holds them, then the extras (which decoder, which checkpoint, the
selection counts, both masses, the formal comparison, the network as eNewick).

Why a formal run is allowed here and refused there: ``camus_expand.py``'s refusal assumes every
checkpoint's level-1 rows are already landed, which is true of ``_mix`` / ``_mix2L`` / ``_mix75``
and false of any family added since -- ``_l2mix``, whose formal rows are as new as its relaxed
ones.  The check is therefore made per row rather than per run: a run is refused only if a row it
would emit already exists in the base under the same (method, gene_trees, n, network_idx, seed).

Every base line is carried through **byte for byte** in its eleven columns, with the extras empty,
and that is verified by re-splitting the file that was written.

Scoring is ``camus_expand.py``'s, which is camus_a104's:

* ``cluster_f1``: the driver's ``f1`` -- ``Instance.cluster_counts`` on the *selection*, the landed
  ``f1_selected``.  For the formal decoder that is not the network's F1, because the network is the
  selection plus its closure; ``f1_formal_network`` carries the network reading beside it.
* ``luay``: PhyloNet ``CMPNETS -m luay`` against ``camus.truth_of``, both OUT-pruned and gamma-free.
* ``luay_norm`` = luay / (n - 1 + floor(3 (k_true + k_pred) / 2)), camus_a104's closed form;
  ``luay_norm_exact`` = luay / floor((X_pred + X_truth) / 2) with both sizes measured by the jar.

Usage::

    gnn_cpu_v1/.venv/bin/python l2reg_eval/camus_expand_multi.py \\
        --run gnn_cpu_v1/runs/C4537CLFY0_t1_relaxedL1 \\
        --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2 \\
        --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2_relaxedL1 \\
        --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2_relaxedL2 \\
        --formal-run gnn_cpu_v1/runs/C4537CLFY0_t1 \\
        --formal-run gnn_cpu_v1/runs/C4537CLFY0_t1_l2 \\
        --out results/rl/camus_a104_per_network_l2mix.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import common as C
from camus_expand import EXTRA, cells_of
from make_csv import CAMUS_COLUMNS, DEFAULT_JAR, PhyloNetLuay, closed_form_max, fmt, missing

from opc.rl import camus_nets as CN  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

COUNTS = ("n_selected", "tp", "fp", "fn", "precision", "recall")


def variant_rank(tag: str) -> int:
    """``level1`` first, then the relaxed variants by level cap -- ``make_csv.variant_rank``."""
    return 0 if tag in ("", "level1") else int(tag.rsplit("_L", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, action="append", required=True, metavar="DIR",
                    help="a gnn_cpu_v1 camus run directory; repeatable")
    ap.add_argument("--formal-run", type=Path, action="append", default=[], metavar="DIR",
                    help="a level-1 run supplying the formal network each relaxed row is compared "
                         "against; repeatable.  A --run that is itself level-1 is added to these.")
    ap.add_argument("--base", type=Path, default=C.LANDED / "camus_a104_per_network.csv")
    ap.add_argument("--out", type=Path, default=C.LANDED / "camus_a104_per_network_l2mix.csv")
    ap.add_argument("--json", type=Path, default=None,
                    help="provenance and per-(n, checkpoint, decoder) medians; default <out>.json")
    ap.add_argument("--work", type=Path, default=None,
                    help="PhyloNet scratch; default <out parent>/phylonet_camus_expand")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET") or DEFAULT_JAR))
    args = ap.parse_args()
    args.base, args.out = args.base.resolve(), args.out.resolve()

    jar = args.phylonet.expanduser().resolve()
    if not jar.exists():
        raise SystemExit(f"no PhyloNet jar at {jar}: pass --phylonet or set OPC_PHYLONET")

    # ---- the base, kept line by line so the eleven columns survive byte for byte
    base_text = args.base.read_text()
    base_lines = base_text.rstrip("\n").split("\n")
    header = base_lines[0].split(",")
    if tuple(header) != CAMUS_COLUMNS:
        raise SystemExit(f"{args.base} has columns {header}, expected {CAMUS_COLUMNS}")
    base_rows = list(csv.DictReader(io.StringIO(base_text)))
    if len(base_rows) != len(base_lines) - 1:
        raise SystemExit(f"{args.base}: {len(base_lines) - 1} lines but {len(base_rows)} parsed rows")
    base_key = {(r["method"], r["gene_trees"], int(r["n"]), int(r["network_idx"]), int(r["seed"])): r
                for r in base_rows}

    # ---- the runs, each one decoder
    runs: list[tuple[Path, str, list[dict]]] = []
    for d in args.run:
        d = d.resolve()
        cells = cells_of(d)
        if not cells:
            raise SystemExit(f"no cells under {d}/cells")
        tags = {C.cell_decoder(c) for c in cells}
        if len(tags) != 1:
            raise SystemExit(f"{d} mixes decoders {sorted(tags)}")
        runs.append((d, tags.pop(), cells))
    formal_dirs = list(dict.fromkeys(
        [d.resolve() for d in args.formal_run] + [d for d, t, _ in runs if t == "level1"]))

    truths: dict[tuple, tuple[str, set]] = {}

    def truth(phase: str, rep: int, k: int):
        if (phase, rep) not in truths:
            t = C.truth_of(rep, inputs=C.GNN / "inputs", phase=phase)
            if int(t.n_retics) != k:
                raise SystemExit(f"{phase} rep{rep}: truth has {t.n_retics} reticulations, cell {k}")
            truths[(phase, rep)] = (C.strip_gamma(N.to_enewick(t)), CN.label_clusters(t))
        return truths[(phase, rep)]

    def network_clusters(enewick: str) -> set:
        net = N.from_enewick(enewick)
        return CN.restrict_clusters(net, CN.ingroup_of(net))

    # ---- the formal decoder's networks, keyed as the cells are
    formal: dict[tuple, tuple[frozenset, float]] = {}
    for d in formal_dirs:
        if not d.is_dir():
            print(f"[expand] no formal run at {d}: skipping", flush=True)
            continue
        for c in cells_of(d):
            if C.cell_decoder(c) != "level1":
                raise SystemExit(f"--formal-run {d} is not a level-1 run")
            tc = truth(c["phase"], int(c["rep"]), int(c["k"]))[1]
            for m in c["models"]:
                if m.get("enewick"):
                    got = network_clusters(m["enewick"])
                    formal[(c["phase"], int(c["rep"]), c["arm"], m["tag"], int(m["seed"]))] = (
                        frozenset(got), CN.cluster_counts(got, tc)["f1"])

    # ---- one row per (cell, model, run)
    luay = PhyloNetLuay(jar)
    rows, f1_mismatch, clash = [], [], []
    for d, tag, cells in runs:
        for c in cells:
            phase, rep, arm = c["phase"], int(c["rep"]), c["arm"]
            n, k = int(c["n"]), int(c["k"])
            t_str, tc = truth(phase, rep, k)
            for m in c["models"]:
                label = C.GNN_LABELS.get(m["tag"])
                if label is None:
                    raise SystemExit(f"checkpoint tag {m['tag']!r} has no camus_a104 label: "
                                     f"add it to common.GNN_LABELS")
                method = label + C.decoder_suffix(tag)
                seed = int(m["seed"])
                if (method, arm, n, rep, seed) in base_key:
                    clash.append((method, arm, n, rep, seed))
                    continue
                row = {"method": method, "_base": label, "_run": d.name, "gene_trees": arm,
                       "n": n, "network_idx": rep, "seed": seed, "cluster_f1": m["f1"],
                       "k_pred": int(m["k_pred"]), "k_true": k, "phase": phase, "arm": arm,
                       "tag": m["tag"], "decoder": tag, **{x: m[x] for x in COUNTS},
                       "status": "ok", "enewick": m.get("enewick")}
                if m.get("enewick") and int(m["k_pred"]) >= 0:
                    row["_i"] = luay.add(C.strip_gamma(m["enewick"]), t_str)
                    got = network_clusters(m["enewick"])
                    net_f1 = CN.cluster_counts(got, tc)["f1"]
                    # The relaxed decoder does not close its selection, so its selection F1 and
                    # its network F1 are the same number; the formal decoder's are not.
                    if tag != "level1" and abs(net_f1 - m["f1"]) > 1e-9:
                        f1_mismatch.append((d.name, phase, rep, m["tag"], seed))
                    key = (phase, rep, arm, m["tag"], seed)
                    if key in formal:
                        row["same_network_as_formal"] = frozenset(got) == formal[key][0]
                        row["f1_formal_network"] = formal[key][1]
                else:
                    row["status"] = "no_network"
                rows.append(row)
    if clash:
        raise SystemExit(f"{len(clash)} rows already in {args.base} -- e.g. {clash[:3]}.  "
                         f"camus_a104 already holds those; drop that run from --run.")
    if not rows:
        raise SystemExit("no rows to add")

    # **Absolute.**  ``phylonet.run`` writes ``<workdir>/run.nex`` and launches java with
    # ``cwd=workdir`` while passing that same path on the command line, so a relative workdir
    # resolves twice and the jar reports ``No such file 'run.nex'`` -- with a zero exit status,
    # which surfaces only as "parsed 0 distances".
    work = (args.work or args.out.parent / "phylonet_camus_expand").resolve()
    luay.resolve(work)
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
        row["luay_norm"] = r["luay"] / closed_form_max(row["n"], row["k_true"], row["k_pred"])
        if r["max_exact"]:
            row["luay_norm_exact"] = r["luay"] / r["max_exact"]

    # ---- each relaxed row's formal counterpart: a landed base row, or a level-1 row written here
    mine = {(r["_base"], r["gene_trees"], r["n"], r["network_idx"], r["seed"]): r
            for r in rows if r["decoder"] == "level1"}
    for row in rows:
        if row["decoder"] == "level1":
            row["f1_formal_selection"] = row["cluster_f1"]
            continue
        k = (row["_base"], row["gene_trees"], row["n"], row["network_idx"], row["seed"])
        b = base_key.get(k) or mine.get(k)
        if b is None:
            continue
        row["_formal"] = b
        row["f1_formal_selection"] = float(b["cluster_f1"])

    order = {label: i for i, label in enumerate(C.GNN_LABELS.values())}
    rows.sort(key=lambda r: (order.get(r["_base"], len(order)), variant_rank(r["decoder"]),
                             r["n"], r["gene_trees"], r["network_idx"], r["seed"]))

    # ---- write: every base line verbatim with the extras empty, then the new rows
    pad = "," * len(EXTRA)
    buf = io.StringIO()
    buf.write(",".join(CAMUS_COLUMNS + EXTRA) + "\n")
    for line in base_lines[1:]:
        buf.write(line + pad + "\n")
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow([fmt(r.get(col)) for col in CAMUS_COLUMNS + EXTRA])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(buf.getvalue())

    # ---- the base must re-derive byte for byte from what was written
    got = args.out.read_text().rstrip("\n").split("\n")
    if len(got) != len(base_lines) + len(rows):
        raise SystemExit(f"{args.out}: {len(got)} lines, expected {len(base_lines) + len(rows)}")
    for i, line in enumerate(base_lines[1:], 1):
        if got[i] != line + pad:
            raise SystemExit(f"{args.out} line {i + 1} is not {args.base}'s line byte for byte")
    for i, r in enumerate(csv.DictReader(io.StringIO(args.out.read_text()))):
        if i >= len(base_rows):
            break
        if {c: r[c] for c in CAMUS_COLUMNS} != base_rows[i]:
            raise SystemExit(f"{args.out} row {i} differs from {args.base} once parsed")

    bad = [r for r in rows if r["status"] != "ok"]
    over = [r for r in rows if not missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    checked = [r for r in rows if r["decoder"] != "level1" and r["status"] == "ok"]
    print(f"\n[expand] {len(base_rows)} camus_a104 rows (kept byte for byte) + {len(rows)} new "
          f"rows -> {args.out}")
    for d, tag, cells in runs:
        g = [r for r in rows if r["_run"] == d.name]
        print(f"[expand]   {len(g):4d} {tag:10s} {sorted({r['method'] for r in g})} <- {d.name}")
    print(f"[expand] status: {len(rows) - len(bad)} ok, {len(bad)} not "
          f"{sorted({r['status'] for r in bad})}; luay_norm_exact > 1 on {len(over)}; "
          f"relaxed selection F1 == network F1 on {len(checked) - len(f1_mismatch)} of "
          f"{len(checked)}" + (f" (mismatch: {f1_mismatch[:5]})" if f1_mismatch else ""))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["n"], r["_base"], r["decoder"])].append(r)
    med, summary = statistics.median, []
    print(f"\n  {'n':>4s} {'checkpoint':16s} {'decoder':11s} {'rows':>4s} | {'sel F1':>7s} "
          f"{'net F1':>7s} | {'luay':>6s} {'norm_ex':>7s} | {'k_pred':>6s} {'k_true':>6s} | "
          f"same as formal")
    for (n, label, tag), g in sorted(groups.items(),
                                     key=lambda kv: (order.get(kv[0][1], 99),
                                                     variant_rank(kv[0][2]), kv[0][0])):
        ok = [r for r in g if r["status"] == "ok"]
        nf = [r["f1_formal_network"] for r in g if not missing(r.get("f1_formal_network"))]
        ex = [r["luay_norm_exact"] for r in ok if not missing(r.get("luay_norm_exact"))]
        same = [r for r in g if r.get("same_network_as_formal") is not None]
        rec = {"n": n, "checkpoint": label, "decoder": tag, "rows": len(g), "ok": len(ok),
               "cluster_f1": med([r["cluster_f1"] for r in g]),
               "f1_formal_network": med(nf) if nf else None,
               "luay": med([r["luay"] for r in ok]) if ok else None,
               "luay_norm_exact": med(ex) if ex else None,
               "k_pred": med([r["k_pred"] for r in g]), "k_true": med([r["k_true"] for r in g]),
               "same_network_as_formal": (sum(bool(r["same_network_as_formal"]) for r in same),
                                          len(same))}
        summary.append(rec)
        print(f"  {n:4d} {label:16s} {tag:11s} {len(g):4d} | {rec['cluster_f1']:7.4f} "
              f"{(rec['f1_formal_network'] if nf else float('nan')):7.4f} | "
              f"{(rec['luay'] if ok else float('nan')):6.1f} "
              f"{(rec['luay_norm_exact'] if ex else float('nan')):7.4f} | "
              f"{rec['k_pred']:6.1f} {rec['k_true']:6.1f} | "
              f"{rec['same_network_as_formal'][0]}/{rec['same_network_as_formal'][1]}")

    out_json = args.json or args.out.with_suffix(".json")
    out_json.write_text(json.dumps(
        {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": str(args.base),
         "base_rows": len(base_rows), "out": str(args.out), "new_rows": len(rows),
         "columns": list(CAMUS_COLUMNS + EXTRA), "phylonet_jar": str(jar),
         "runs": [{"dir": str(d), "decoder": t, "cells": len(cs),
                   "methods": sorted({r["method"] for r in rows if r["_run"] == d.name})}
                  for d, t, cs in runs],
         "formal_runs": [str(d) for d in formal_dirs],
         "labels": {k: v for k, v in C.GNN_LABELS.items()
                    if v in {r["_base"] for r in rows}},
         "status": {s: sum(r["status"] == s for r in rows)
                    for s in sorted({r["status"] for r in rows})},
         "relaxed_selection_f1_equals_network_f1": [len(checked) - len(f1_mismatch), len(checked)],
         "medians": summary}, indent=1))
    print(f"\n[expand] provenance -> {out_json}")
    return 1 if f1_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
