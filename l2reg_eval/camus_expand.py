#!/usr/bin/env python3
"""Append a gnn_cpu_v1 camus run's rows to a copy of ``camus_a104_per_network.csv`` -- luay by PhyloNet.

``gnn_cpu_v1/run_gnn_cpu.py`` writes one JSON per camus cell with every checkpoint's selection
counts, cluster F1, ``k_pred`` and network (eNewick), but no luay.  This scores those networks the
way camus_a104's GNN rows were scored and appends them to a copy of that CSV:

* ``cluster_f1``: the driver's ``f1`` -- ``Instance.cluster_counts`` on the selection, which is the
  landed ``f1_selected`` camus_a104 carries for every GNN row;
* ``luay``: PhyloNet ``CMPNETS -m luay`` between the network and the camus truth with OUT pruned
  (``camus.truth_of``), gammas stripped -- the convention under which the jar reproduces all 900
  landed camus GNN luay values;
* ``luay_norm``: ``luay / (n - 1 + floor(3 (k_true + k_pred) / 2))``, camus_a104's closed form;
* ``luay_norm_exact``: ``luay / floor((X_pred + X_truth) / 2)`` with both sizes measured by the jar
  (``make_csv.PhyloNetLuay``'s star probes).  They equal camus_a104's sizes on every camus network
  but one truth, cam150 rep04 (a size-2 cycle), which the jar sizes differently; on that cell's
  rows this column follows the jar.

Rows are labelled as ``make_csv`` labels them: ``ours/prior [relaxed L1]`` and so on for a relaxed
run.  A formal (level-1) run is refused -- camus_a104 already holds those rows.  The copy keeps
every original line byte for byte (checked) and appends.  Beside it, ``<decoder>_per_network.csv``
carries the same rows with the extras and every network's eNewick.

**Selection F1 is not network F1 for the formal decoder.**  camus_a104's GNN ``cluster_f1`` scores
the *selection*; the formal decoder's network is the selection **plus its closure** -- the hybrid
clusters ``A & B`` it adds -- so its selection F1 leaves those clusters out, in both directions (a
true hybrid cluster is not credited, a false one not charged).  The relaxed decoder has no closure:
its selection is its network, and the two F1s are the same number (asserted, row by row).  So the
sidecar also carries ``f1_formal_network``: the formal network (``--formal-run``, default the Mac
run ``C4537CLFY0_t1``) scored the way camus_a104 scores third-party networks
(``camus_nets.cluster_counts`` on its clusters).  Compare decoders on that, or on luay, which was
always computed on the network.

Usage:  gnn_cpu_v1/.venv/bin/python l2reg_eval/camus_expand.py --run gnn_cpu_v1/runs/<RUN_ID>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import common as C
from make_csv import CAMUS_COLUMNS, DEFAULT_JAR, PhyloNetLuay, closed_form_max, fmt, missing

from opc.rl import camus_nets as CN  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

EXTRA = ("phase", "arm", "tag", "decoder", "n_selected", "tp", "fp", "fn", "precision", "recall",
         "mass_pred", "mass_truth", "f1_formal_selection", "f1_formal_network",
         "same_network_as_formal", "status", "enewick")
COUNTS = ("n_selected", "tp", "fp", "fn", "precision", "recall")


def cells_of(run: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted((run / "cells").glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a gnn_cpu_v1 run directory")
    ap.add_argument("--base", type=Path, default=C.LANDED / "camus_a104_per_network.csv")
    ap.add_argument("--formal-run", type=Path, default=C.GNN / "runs" / "C4537CLFY0_t1",
                    help="the formal decoder's run on the same cells, for the network comparison")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <run>/camus_a104_per_network_<decoder>.csv")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET") or DEFAULT_JAR))
    args = ap.parse_args()

    run = args.run.resolve()
    cells = cells_of(run)
    if not cells:
        raise SystemExit(f"no cells under {run}/cells")
    tags = {C.cell_decoder(c) for c in cells}
    if len(tags) != 1:
        raise SystemExit(f"{run} mixes decoders {sorted(tags)}")
    (tag,) = tags
    if tag == "level1":
        raise SystemExit("a formal-decoder run: camus_a104 already holds those rows")
    jar = args.phylonet.expanduser().resolve()
    if not jar.exists():
        raise SystemExit(f"no PhyloNet jar at {jar}: pass --phylonet or set OPC_PHYLONET")
    base_bytes = args.base.read_bytes()
    header = base_bytes.split(b"\n", 1)[0].decode().strip().split(",")
    if tuple(header) != CAMUS_COLUMNS:
        raise SystemExit(f"{args.base} has columns {header}, expected {CAMUS_COLUMNS}")

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

    formal: dict[tuple, tuple[frozenset, float]] = {}
    if args.formal_run.is_dir():
        for c in cells_of(args.formal_run):
            if C.cell_decoder(c) != "level1":
                continue
            tc = truth(c["phase"], int(c["rep"]), int(c["k"]))[1]
            for m in c["models"]:
                if m.get("enewick"):
                    got = network_clusters(m["enewick"])
                    formal[(c["phase"], int(c["rep"]), c["arm"], m["tag"], int(m["seed"]))] = (
                        frozenset(got), CN.cluster_counts(got, tc)["f1"])

    luay = PhyloNetLuay(jar)
    rows, f1_mismatch = [], []
    for c in cells:
        phase, rep, arm, n, k = c["phase"], int(c["rep"]), c["arm"], int(c["n"]), int(c["k"])
        t_str, tc = truth(phase, rep, k)
        for m in c["models"]:
            label = C.GNN_LABELS.get(m["tag"])
            if label is None:
                raise SystemExit(f"checkpoint tag {m['tag']!r} has no camus_a104 label")
            row = {"method": label + C.decoder_suffix(tag), "_base": label, "gene_trees": arm,
                   "n": n, "network_idx": rep, "seed": int(m["seed"]), "cluster_f1": m["f1"],
                   "k_pred": int(m["k_pred"]), "k_true": k, "phase": phase, "arm": arm,
                   "tag": m["tag"], "decoder": tag, **{x: m[x] for x in COUNTS},
                   "status": "ok", "enewick": m.get("enewick")}
            if m.get("enewick") and int(m["k_pred"]) >= 0:
                row["_i"] = luay.add(C.strip_gamma(m["enewick"]), t_str)
                got = network_clusters(m["enewick"])
                if abs(CN.cluster_counts(got, tc)["f1"] - m["f1"]) > 1e-9:
                    f1_mismatch.append((phase, rep, m["tag"], m["seed"]))
                key = (phase, rep, arm, m["tag"], int(m["seed"]))
                if key in formal:
                    row["same_network_as_formal"] = frozenset(got) == formal[key][0]
                    row["f1_formal_network"] = formal[key][1]
            else:
                row["status"] = "no_network"
            rows.append(row)

    luay.resolve(run / "phylonet")
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

    base = {}
    for r in csv.DictReader(io.StringIO(base_bytes.decode())):
        base[(r["method"], r["gene_trees"], int(r["n"]), int(r["network_idx"]), int(r["seed"]))] = r
    for row in rows:
        b = base.get((row["_base"], row["gene_trees"], row["n"], row["network_idx"], row["seed"]))
        if b is not None:
            row["_formal"] = b
            row["f1_formal_selection"] = float(b["cluster_f1"])
    rows.sort(key=lambda r: (r["method"], r["n"], r["gene_trees"], r["network_idx"], r["seed"]))

    def csv_text(columns) -> str:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for r in rows:
            w.writerow([fmt(r.get(col)) for col in columns])
        return buf.getvalue()

    suffix = tag.replace("_", "")
    out = args.out or run / f"camus_a104_per_network_{suffix}.csv"
    body = (base_bytes if base_bytes.endswith(b"\n") else base_bytes + b"\n")
    out.write_bytes(body + csv_text(CAMUS_COLUMNS).encode())
    if not out.read_bytes().startswith(base_bytes):
        raise SystemExit(f"{out} does not begin with {args.base} byte for byte")
    side = run / f"{suffix}_per_network.csv"
    side.write_text(",".join(CAMUS_COLUMNS + EXTRA) + "\n" + csv_text(CAMUS_COLUMNS + EXTRA))
    n_base = body.count(b"\n") - 1
    bad = [r for r in rows if r["status"] != "ok"]
    over = [r for r in rows if not missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    print(f"[expand] {n_base} camus_a104 rows (kept byte for byte) + {len(rows)} {tag} rows -> {out}\n"
          f"[expand] with extras and networks -> {side}\n"
          f"[expand] status: {len(rows) - len(bad)} ok, {len(bad)} not {sorted({r['status'] for r in bad})}; "
          f"luay_norm_exact > 1 on {len(over)}; relaxed selection F1 == network F1 on "
          f"{len(rows) - len(bad) - len(f1_mismatch)} of {len(rows) - len(bad)}"
          + (f" (mismatch: {f1_mismatch[:5]})" if f1_mismatch else ""))

    groups = defaultdict(list)
    for r in rows:
        b = r.get("_formal")
        if b is None or missing(r.get("luay_norm_exact")) or not b["luay_norm_exact"]:
            continue
        groups[(r["n"], r["_base"])].append(r)
    med = statistics.median
    print(f"  {'n':>4s} {'checkpoint':18s} {'pairs':>5s} | F1: formal selection (camus_a104) | formal "
          f"network | relaxed | luay_norm_exact: formal -> relaxed | same network")
    for (n, label), g in sorted(groups.items()):
        fn = [r["f1_formal_network"] for r in g if not missing(r.get("f1_formal_network"))]
        print(f"  {n:4d} {label:18s} {len(g):5d} | {med([r['f1_formal_selection'] for r in g]):.4f} | "
              f"{med(fn) if fn else float('nan'):.4f} | {med([r['cluster_f1'] for r in g]):.4f} | "
              f"{med([float(r['_formal']['luay_norm_exact']) for r in g]):.4f} -> "
              f"{med([r['luay_norm_exact'] for r in g]):.4f} | "
              f"{sum(bool(r.get('same_network_as_formal')) for r in g)}/{len(g)}")
    return 1 if f1_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
