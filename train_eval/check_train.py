#!/usr/bin/env python3
"""T1: a scored training cell is what the GNN was trained on -- checked against training's own builder.

``cluster_anatomy/score_cells.py`` reaches a ``trn`` cell through ``scoring.build_cell`` (camus layout,
``prepare_gene_trees``'s ``as_is`` copy of the trees).  Training reached the same network through
``opc.rl.instances`` -- a different entry point onto the same stages.  Two comparisons per cell:

* **exact**, under the scoring path's taxon order (``camus.truth_of``: OUT pruned, taxa renumbered in
  leaf postorder): ``instances.build_from_trees`` on the *source* ``.nwk.gz`` must give bit-equal
  ``words``, ``support``, ``x_nodes``, ``x_global`` and truth rows, and its forward the logits
  ``score_cells`` cached.  This is the gate.
* **drift**, under training's own taxon order (x/y: ``N.derive`` of the grid cell, as
  ``instances.build`` did; l2t: the source eNewick's order): the same clusters by label set, their
  support, and the per-cluster change in ``x_nodes`` and in the logit.  Candidate-row order and the
  pair conv's hash tie-break both follow taxon order, so a small drift is expected; it is reported,
  not gated.

No flag may start with ``--t``: ``run_gnn_cpu`` prefix-matches ``--threads`` from ``sys.argv`` at import.

T2 (``--val-f1``): the scored validation cells read against ``verdict.json``.  Training's
``val_decoder: auto`` is level 1 on the 800 x/y val cells and relaxed L2 (max level 2) on the 400 l2t
ones, so each decoder's mean F1 is compared with its own verdict number, and the pooled mean with
``best_val_f1``; greedy likewise.  Exact agreement is not expected -- taxon order differs (above), and
training's val ran through ``instances.build`` -- so it warns past ``WARN`` and fails past ``FAIL``.
Every cell must also say epoch 30.

Usage:  gnn_cpu_v1/.venv/bin/python train_eval/check_train.py --anat <scored root> --cells trn_x25:0,trn_l2t50:16
        (both arms of each rep; exit 1 on any exact-parity failure)
        gnn_cpu_v1/.venv/bin/python train_eval/check_train.py --val-f1      (T2, after the full scoring run)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "cluster_anatomy"))

import common as C  # noqa: E402

import numpy as np  # noqa: E402

import run_gnn_cpu as R  # noqa: E402
import scoring as S  # noqa: E402
import stage as ST  # noqa: E402
from opc import grid  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.simulate import networks as N  # noqa: E402


def label_sets(words: np.ndarray, taxa) -> list[frozenset]:
    return [frozenset(taxa[i] for i in range(len(taxa)) if (mk >> i) & 1) for mk in S.words_to_masks(words)]


def training_truth(row: dict):
    """The network ``instances`` built this cell's targets from, in training's taxon order."""
    phase = row["phase"]
    src, fam, _pkg = ST.PHASES[phase]
    rec = next(r for r in ST.sources(phase) if r["rep"] == int(row["rep"]))
    if fam == "l2t":
        return N.from_enewick(rec["net"].read_text())
    c = {c.key: c for c in grid.enumerate_cells(**grid.PHASES[src])}[rec["key"]]
    with open(ST.XY / src / "networks" / f"{c.base_key}.pkl", "rb") as fh:
        base = pickle.load(fh)
    return N.derive(base, r=c.r, cu=c.cu, gamma=c.gamma, gamma_range=c.gamma_range, gamma_seed=c.gamma_seed)


def check(ds, row: dict, net, anat: Path, work: Path) -> dict:
    stem = C.cell_name(row["phase"], int(row["rep"]), row["arm"])
    cam = CAMUS.CamusCell(row["phase"], int(row["rep"]), row["arm"], ds.inputs)
    src = cam.trees_path.resolve()
    cell = S.build_cell(ds, row, work)
    z_cell, _p = S.model_scores(net, cell)
    cached = np.load(anat / "trn" / "cells" / f"{stem}.npz")
    out = {"cell": stem, "problems": []}

    inst = I.build_from_trees(cam.key, CAMUS.truth_of(cam), src, work / f"{stem}.parquet",
                              m=1000, tau=2, arm=row["arm"])
    z_inst, _ = S.model_scores(net, SimpleNamespace(gs=inst.gs, graph=inst.graph, xn=inst.x_nodes,
                                                     xg=inst.x_global))
    for name, a, b in (("words", inst.gs.words, cell.gs.words),
                       ("support", np.asarray(inst.gs.support), np.asarray(cell.gs.support)),
                       ("x_nodes", inst.x_nodes, cell.xn), ("x_global", inst.x_global, cell.xg),
                       ("truth_rows", np.sort(inst.truth_rows), np.sort(cell.truth_rows)),
                       ("logits (instances vs scoring)", z_inst, z_cell),
                       ("logits (scoring vs cached npz)", z_cell, cached["logits"]),
                       ("words (scoring vs cached npz)", cell.gs.words, cached["words"])):
        if a.shape != b.shape or not np.array_equal(a, b):
            out["problems"].append(f"exact: {name} differ")
    out["n_clusters"] = int(cell.gs.n_clusters)

    # drift under training's own taxon order, matched by label set
    tt = training_truth(row)
    inst2 = I.build_from_trees(cam.key, tt, src, work / f"{stem}.train_order.parquet",
                               m=1000, tau=2, arm=row["arm"])
    z2, _ = S.model_scores(net, SimpleNamespace(gs=inst2.gs, graph=inst2.graph, xn=inst2.x_nodes,
                                                 xg=inst2.x_global))
    a_sets, b_sets = label_sets(cell.gs.words, cell.taxa), label_sets(inst2.gs.words, list(tt.labels))
    ia = {s: i for i, s in enumerate(a_sets)}
    same_set = set(a_sets) == set(b_sets)
    idx = np.array([ia[s] for s in b_sets]) if same_set else None
    out["drift"] = {
        "taxon_order_same": list(tt.labels) == list(cell.taxa),
        "same_cluster_set": same_set,
        "support_equal": bool(same_set and np.array_equal(np.asarray(inst2.gs.support),
                                                          np.asarray(cell.gs.support)[idx])),
        "x_nodes_max_abs": float(np.abs(inst2.x_nodes - cell.xn[idx]).max()) if same_set else None,
        "logit_max_abs": float(np.abs(z2 - z_cell[idx]).max()) if same_set else None,
        "p_half_flips": int(np.count_nonzero((z2 > 0) != (z_cell[idx] > 0))) if same_set else None,
    }
    return out


WARN, FAIL = 0.005, 0.01
VERDICT = C.CHECKPOINTS / C.TAG / f"s{C.SEED}" / "verdict.json"


def val_f1(anat: Path, out: Path | None) -> int:
    """T2: mean val F1 per training decoder, and pooled, against verdict.json."""
    v = json.loads(VERDICT.read_text())
    ds = C.DATASETS["trn"]
    rows = [r for r in ds.rows() if r["split"] == "val"]
    recs = {r["phase"] + ":" + r["rep"] + ":" + r["arm"]:
            json.loads((anat / "trn" / "cells" / f"{C.cell_name(r['phase'], int(r['rep']), r['arm'])}.json").read_text())
            for r in rows}
    epochs = {rec["epoch"] for rec in recs.values()}
    got = {}
    for method in ("gnn", "greedy"):
        f = {"level1": [], "relaxed_L2": []}
        for r in rows:
            dtag = "relaxed_L2" if r["phase"].startswith("trn_l2t") else "level1"
            f[dtag].append(recs[r["phase"] + ":" + r["rep"] + ":" + r["arm"]]["decodes"][f"{method}|{dtag}"]["f1"])
        got[method] = {d: (float(np.mean(x)), len(x)) for d, x in f.items()}
        got[method]["pooled"] = (float(np.mean(f["level1"] + f["relaxed_L2"])), len(f["level1"]) + len(f["relaxed_L2"]))
    want = {"gnn": {"level1": v["val_f1_level1"], "relaxed_L2": v["val_f1_relaxed_L2"], "pooled": v["best_val_f1"]},
            "greedy": {"pooled": v["best_val_f1_greedy"]}}
    bad = warn = 0
    for method, ws in want.items():
        for key, w in ws.items():
            g, n = got[method][key]
            d = abs(g - w)
            tag = "FAIL" if d > FAIL else "warn" if d > WARN else "ok  "
            bad += d > FAIL
            warn += WARN < d <= FAIL
            print(f"[check_train] T2 {tag} {method:6s} {key:10s} scored {g:.5f} (n {n}) | verdict {w:.5f} | |d| {d:.5f}")
    if epochs != {30}:
        bad += 1
        print(f"[check_train] T2 FAIL epochs {sorted(epochs)}, expected 30")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"scored": got, "verdict": want, "epochs": sorted(epochs),
                                   "warn": WARN, "fail": FAIL}, indent=1) + "\n")
    print(f"[check_train] T2: {bad} failed, {warn} warnings ({len(rows)} val cells)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anat", type=Path, default=C.RUNS, help="the root score_cells wrote trn/ under")
    ap.add_argument("--cells", default="", help="phase:rep,... (both arms of each)")
    ap.add_argument("--val-f1", action="store_true", help="T2 instead of T1")
    ap.add_argument("--out", type=Path, default=None, help="write the report here as JSON")
    args = ap.parse_args()
    if args.val_f1:
        return val_f1(args.anat, args.out)
    if not args.cells:
        raise SystemExit("[check_train] --cells is required for T1")
    ds = C.DATASETS["trn"]
    want = [(p, int(r)) for p, r in (x.split(":") for x in args.cells.split(",") if x)]
    rows = [r for r in ds.rows() if (r["phase"], int(r["rep"])) in want]
    if len(rows) != 2 * len(want):
        raise SystemExit(f"[check_train] {len(rows)} CELLS rows for {len(want)} reps; expected two arms each")
    mdl = R.load_models(C.CHECKPOINTS, [C.TAG], [C.SEED])[0]
    if mdl.epoch != 30:
        raise SystemExit(f"[check_train] loaded epoch {mdl.epoch}; selF1 (verdict.json best_epoch) is 30")
    recs, n_bad = [], 0
    with tempfile.TemporaryDirectory(prefix="check_train_") as tmp:
        for row in rows:
            r = check(ds, row, mdl.net, args.anat, Path(tmp))
            recs.append(r)
            n_bad += bool(r["problems"])
            d = r["drift"]
            print(f"[check_train] {'ok  ' if not r['problems'] else 'FAIL'} {r['cell']:36s} |C| {r['n_clusters']:5d} "
                  f"| drift: order same {d['taxon_order_same']}, set {d['same_cluster_set']}, "
                  f"support {d['support_equal']}, max|dx| {d['x_nodes_max_abs']:.3g}, "
                  f"max|dz| {d['logit_max_abs']:.3g}, sign flips {d['p_half_flips']}"
                  + ("".join(f"\n    {p}" for p in r["problems"])), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"epoch": mdl.epoch, "cells": recs}, indent=1) + "\n")
    print(f"[check_train] T1: {len(rows) - n_bad} ok, {n_bad} failed", flush=True)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
