#!/usr/bin/env python3
"""The choice model fitted on the checkpoint's own training networks, selected on its validation ones.

``calib.py`` cross-fits ``beta`` on the benchmarks themselves (pools ``sim`` and ``netcs``).  This fits
the same conditional logit, with the same blocks, design, moments, ridge-Newton and log-likelihood
(``calib.configs / option_matrix / moments / design / fit / loglik``, called unchanged), on the ``trn``
dataset (``train_eval/``): the GNN's training grids x25 x50 y25 y50 l2t25 l2t50, both gene-tree arms.

* **train** = reps 0-699, **val** = reps 700-799: ``_l2mix_pair/s0/config.json``'s own split.  A network
  (group ``trn:<phase>:<rep>``) is in exactly one of them.  Nothing from a benchmark is read.
* **One pooled model, no dataset offsets** (``calib.columns(v, "train")``: offsets are ``sim``'s only).
* **Pre-registered grid**, per variant: ridge in ``RIDGES`` x training arms in ``ARMSETS`` (both
  arms, as the GNN trained; or the estimated arm only).  Each candidate is fitted on its train blocks
  with its own standardisation moments.
* **Selection**: the largest held-out log-likelihood of ``P_b(r*)`` on the **estimated-arm val
  blocks** -- every benchmark runs on estimated gene trees.  The final model is that candidate's
  train-only fit, so every val number stays held out.
* **Reported, not selected on**: all val blocks; the uniform baseline; the published ``sim`` /
  ``netcs`` all-data fits scored on the same val blocks (they never saw these networks); the same
  grid without the k = 0 networks (z+tc only; ~22% of x/y, and no benchmark has k = 0); and D1, how
  in-sample the training logits are (GNN F1, low-z count, trainable blocks, share of blocks whose r*
  is the emitted configuration -- per phase x arm x split).

Writes ``<runs-root>/calib/train__z<thr>.json`` in calib's schema -- ``folds: 5``,
``fold_of_group: {}``, ``variants[v] = {columns, folds: [], all: {beta, mean, sd, ll}}`` -- so
``credible.py --calib-pool train`` reads it as a transfer, plus ``selection`` and ``d1``.  Gates (exit 1):
L0 train and val groups disjoint and every block from ``trn``; L1 r* is one of R_b (calib's own
assertion); no trainable block in either split.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python nni_ensemble/trainfit.py --runs-root nni_ensemble/runs_trainfit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import ne_common as NE

import numpy as np  # noqa: E402

import calib as CB  # noqa: E402

DS = "trn"
SPLIT = (700, 800)
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
ARMSETS = {"both": ("trn_true", "trn_est_bp1000"), "est": ("trn_est_bp1000",)}
VAL_ARM = "trn_est_bp1000"
PUBLISHED = ("sim", "netcs")


def split_of(rep: int) -> str:
    return "train" if rep < SPLIT[0] else "val" if rep < SPLIT[1] else "test"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root: Path, thr: float) -> tuple[list[dict], list[dict], list[Path]]:
    """Every trainable block (calib.load_training's filter and assertion), and one record per cell."""
    files = sorted((Path(root) / DS / "feat").glob(f"*__{NE.thr_tag(thr)}.json"))
    exs, cells = [], []
    for p in files:
        d = json.loads(p.read_text())
        rep, n_tr, n_star_emitted = int(d["rep"]), 0, 0
        for blk in d["blocks"]:
            if blk["excluded"] or blk["category"] != "A" or not 2 <= blk["r_count"] <= NE.R_CAP:
                continue
            cf = CB.configs(blk)
            want = sum(1 << i for i in blk["rstar"])
            star = [i for i, c in enumerate(cf) if c == want]
            if len(star) != 1:
                raise AssertionError(f"{d['cell']} block {blk['id']}: r* is not one of R_b (L1)")
            n_tr += 1
            n_star_emitted += sorted(blk["rstar"]) == sorted(blk["members"])
            exs.append({"ds": DS, "cell": d["cell"], "group": d["group"], "arm": d["arm"], "rep": rep,
                        "split": split_of(rep), "k": int(d["k"]), "block": blk, "configs": cf,
                        "star": star[0]})
        cells.append({"cell": d["cell"], "phase": d["phase"], "arm": d["arm"], "split": split_of(rep),
                      "k": int(d["k"]), "f1": d["emitted"]["f1"], "M": d["M"],
                      "n_blocks": len(d["blocks"]), "n_trainable": n_tr, "n_star_emitted": n_star_emitted})
    return exs, cells, files


def ll_of(beta, mean, sd, cols, exs) -> float:
    if not exs:
        return float("nan")
    return CB.loglik(np.asarray(beta), [CB.design(ex, cols, np.asarray(mean), np.asarray(sd)) for ex in exs],
                     [ex["star"] for ex in exs])[0]


def uniform(exs) -> float:
    return -sum(math.log(len(ex["configs"])) for ex in exs)


def d1(cells: list[dict]) -> list[dict]:
    """How in-sample the training logits are: per phase x arm x split."""
    by = defaultdict(list)
    for c in cells:
        by[(c["phase"], c["arm"], c["split"])].append(c)
    out = []
    for (ph, arm, sp), cs in sorted(by.items()):
        tr = sum(c["n_trainable"] for c in cs)
        out.append({"phase": ph, "arm": arm, "split": sp, "cells": len(cs),
                    "gnn_f1_mean": float(np.mean([c["f1"] for c in cs])),
                    "M_mean": float(np.mean([c["M"] for c in cs])),
                    "blocks_per_cell": float(np.mean([c["n_blocks"] for c in cs])),
                    "trainable_per_cell": tr / len(cs),
                    "rstar_emitted_share": (sum(c["n_star_emitted"] for c in cs) / tr) if tr else float("nan")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="where trn/feat lives and calib/train__z<thr>.json is written (never runs/)")
    ap.add_argument("--published-root", type=Path, default=NE.RUNS,
                    help="read-only: the published calib/{sim,netcs}__z<thr>.json, scored on val for reference")
    ap.add_argument("--thr", type=float, default=NE.Z_THR)
    ap.add_argument("--variants", default=",".join(NE.VARIANTS))
    args = ap.parse_args()
    if Path(args.runs_root).resolve() == Path(NE.RUNS).resolve():
        raise SystemExit("[trainfit] --runs-root must not be the published runs/ directory")
    t0 = time.perf_counter()
    exs, cells, files = load(args.runs_root, args.thr)
    train = [ex for ex in exs if ex["split"] == "train"]
    val = [ex for ex in exs if ex["split"] == "val"]
    val_est = [ex for ex in val if ex["arm"] == VAL_ARM]
    problems = []
    if not train or not val_est:
        problems.append(f"no trainable blocks: {len(train)} train, {len(val_est)} val (est arm)")
    if {ex["group"] for ex in train} & {ex["group"] for ex in val}:
        problems.append("L0: a network has blocks in both train and val")
    if any(ex["ds"] != DS or not ex["group"].startswith(f"{DS}:") for ex in exs):
        problems.append("L0: a block from outside trn")
    if any(ex["split"] == "test" for ex in exs):
        problems.append("L0: a test-split block (rep >= 800) was read")
    print(f"[trainfit] {len(files)} feat files; trainable blocks: {len(train)} train "
          f"({len({ex['group'] for ex in train})} networks), {len(val)} val ({len(val_est)} est arm)", flush=True)
    if problems:
        for p in problems:
            print("[trainfit] PROBLEM " + p, file=sys.stderr)
        return 1

    pub = {}
    for pool in PUBLISHED:
        cp = Path(args.published_root) / "calib" / f"{pool}__{NE.thr_tag(args.thr)}.json"
        if cp.exists():
            pub[pool] = (json.loads(cp.read_text()), sha256(cp))

    out = {"pool": "train", "thr": args.thr, "folds": 5, "fold_of_group": {}, "split": list(SPLIT),
           "val_arm": VAL_ARM, "ridges": list(RIDGES), "armsets": {k: list(v) for k, v in ARMSETS.items()},
           "n_blocks": {"train": len(train), "val": len(val), "val_est": len(val_est)},
           "variants": {}, "selection": {}, "d1": d1(cells), "problems": problems}
    uni = {"val_est": uniform(val_est), "val": uniform(val)}
    for v in [x for x in args.variants.split(",") if x]:
        cols = CB.columns(v, "train")
        grid = []
        ksets = (("all", 0), ("k>=1", 1)) if v == "z+tc" else (("all", 0),)
        for kname, kmin in ksets:
            for aname, arms in ARMSETS.items():
                tr = [ex for ex in train if ex["arm"] in arms and ex["k"] >= kmin]
                mean, sd = CB.moments(tr, cols)
                Phis = [CB.design(ex, cols, mean, sd) for ex in tr]
                stars = [ex["star"] for ex in tr]
                vPhis = [CB.design(ex, cols, mean, sd) for ex in val_est]
                aPhis = [CB.design(ex, cols, mean, sd) for ex in val]
                for ridge in RIDGES:
                    beta, ll = CB.fit(Phis, stars, ridge)
                    grid.append({"k": kname, "arms": aname, "ridge": ridge, "n_train": len(tr),
                                 "train_ll": ll, "train_ll_per_block": ll / len(tr),
                                 "val_est_ll": CB.loglik(beta, vPhis, [ex["star"] for ex in val_est])[0],
                                 "val_ll": CB.loglik(beta, aPhis, [ex["star"] for ex in val])[0],
                                 "beta": beta, "mean": mean, "sd": sd})
        cands = [g for g in grid if g["k"] == "all"]
        best = max(cands, key=lambda g: (g["val_est_ll"], g["ridge"]))
        out["variants"][v] = {"columns": list(cols), "folds": [], "ridge": best["ridge"], "arms": best["arms"],
                              "all": {"beta": best["beta"], "mean": best["mean"], "sd": best["sd"],
                                      "ll": best["train_ll"]}}
        ref = {}
        for pool, (cal, h) in pub.items():
            prec = cal["variants"].get(v)
            if prec:
                a = prec["all"]
                ref[pool] = {"sha256": h, "columns": prec["columns"],
                             "val_est_ll": ll_of(a["beta"], a["mean"], a["sd"], prec["columns"], val_est),
                             "val_ll": ll_of(a["beta"], a["mean"], a["sd"], prec["columns"], val)}
        out["selection"][v] = {
            "criterion": "max val_est_ll over k = all; ties to the larger ridge",
            "chosen": {"arms": best["arms"], "ridge": best["ridge"], "val_est_ll": best["val_est_ll"],
                       "val_ll": best["val_ll"]},
            "uniform": uni, "published_all_fits": ref,
            "grid": [{k: g[k] for k in g if k not in ("beta", "mean", "sd")} | {"beta": g["beta"]} for g in grid]}
        vals = sorted({round(g["val_est_ll"], 3) for g in cands})
        print(f"[trainfit]   {v:10s} chosen arms={best['arms']:4s} ridge={best['ridge']:g}: "
              f"val(est) ll {best['val_est_ll']:.2f} (uniform {uni['val_est']:.2f}; grid range "
              f"{vals[0]:.2f}..{vals[-1]:.2f}; "
              + ", ".join(f"{p} all-fit {r['val_est_ll']:.2f}" for p, r in ref.items())
              + f"); beta = {dict(zip(cols, np.round(best['beta'], 3)))}", flush=True)
    out["inputs"] = {"feat_files": len(files),
                     "feat_manifest_sha256": hashlib.sha256("".join(
                         f"{p.name} {sha256(p)}\n" for p in files).encode()).hexdigest(),
                     "code_sha256": {f: sha256(Path(__file__).parent / f)
                                     for f in ("calib.py", "trainfit.py", "blocks.py", "ne_common.py")}}
    dst = Path(args.runs_root) / "calib" / f"train__{NE.thr_tag(args.thr)}.json"
    NE.write_json(dst, out, indent=1)
    print(f"[trainfit] wrote {dst} in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
