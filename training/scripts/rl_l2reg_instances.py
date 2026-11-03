#!/usr/bin/env python
"""Build RL instances from **data/l2reg_train_v1** -- the level-2 training grids.

    .venv/bin/python -u scripts/rl_l2reg_instances.py --phase l2t25 --tau 2 --reps 800 --jobs 12
    .venv/bin/python -u scripts/rl_l2reg_instances.py --phase l2t50 --tau 2 --arm est_bp1000 \
        --reps 800 --jobs 12

The counterpart of ``scripts/rl_instances.py`` (the grid), ``scripts/rl_external_instances.py``
(data/N100-2) and ``scripts/rl_camus_instances.py`` (camus-data) for a fourth dataset this repo did
not generate.  See :mod:`opc.rl.l2reg` for what the data is, what was checked about it, and why the
inferred arm reads ``.len.nwk.gz``.

Writes ``results/<phase>/instances_tau<t>[_<arm>]/<key>.pkl``, which is exactly where
:func:`opc.rl.instances.cache_dir` looks, so ``w10_gnn.py --phases l2t25`` reads them with no
further plumbing.  Each arm gets its **own directory**, as camus does and as ``n100ext`` pointedly
does not.

**It writes the instance summary parquet, and unlike ``rl_camus_instances.py`` that is not
optional.**  ``w10_gnn.pos_weight_of`` reads ``results/rl/rl_instances_<phase>_tau<t>[_<arm>]
.parquet`` for the pooled positive rate, and above 400 training cells it **refuses** the run rather
than fall back -- because the fallback materialises every training instance before epoch 0, which
A89 records as a silent hang at 0% CPU rather than a failure.  The camus phases are only ever
evaluated, so that driver never needed one; these phases are training data, so this one does.  The
name comes from :func:`opc.rl.instances.summary_name` -- one writer, one reader, one spelling.

**The summary MERGES rather than overwrites**, then reconciles against the cache directory, exactly
as ``rl_instances.py`` does: ``--min-rep``/``--reps`` exist so a phase can be built in rep
*windows*, and an overwrite would leave the file describing the last window alone.
``pos_weight_of`` then refuses the run because the summary does not cover every training key.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from opc import paths  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import l2reg as L2  # noqa: E402


def _row(inst, key: str, seconds: float, nbytes: int) -> dict:
    """One summary row.  **The same columns ``rl_instances.py`` writes**, because one reader --
    ``w10_gnn.pos_weight_of`` -- takes ``n_true_in_ctau`` and ``n_clusters`` off both, and a report
    that concatenates the two must not have to know which driver produced a row."""
    return {
        "key": key, "arm": inst.arm, "n": inst.n, "rep": inst.rep, "k": inst.k, "cu": inst.cu,
        "gamma": inst.gamma, "stratum": inst.stratum, "tau": inst.tau, "m": inst.m,
        "n_clusters": inst.n_clusters, "kernel": int(inst.gs.kernel_size),
        "n_components": len(inst.gs.components),
        "largest_component": max((c.n_vertices for c in inst.gs.components), default=0),
        "n_edges": int(sum(c.n_edges for c in inst.gs.components)),
        "n_true": inst.n_true, "n_true_in_ctau": int(inst.truth_rows.size),
        "truth_coverage": inst.truth_coverage, "n_triples": int(inst.index.shape[0]),
        "seconds": seconds, "bytes": nbytes,
    }


def _one(payload) -> dict:
    phase, key, arm, data, root, cache, m, tau = payload
    cell = L2.L2Cell(phase, key, arm, Path(data))
    path = Path(cache) / f"{key}.pkl"
    t0 = time.perf_counter()
    try:
        if path.exists():
            # ``arm=`` is checked, not trusted: the two arms are different candidate sets over one
            # truth, so a cross-arm read is not a cache that can be shared.
            inst = I.load(path, arm=arm)
            if inst.tau != int(tau) or inst.m != int(m):
                raise SystemExit(f"{path} holds tau={inst.tau} m={inst.m}; "
                                 f"this run wants tau={tau} m={m}")
            return {**_row(inst, key, 0.0, path.stat().st_size), "status": "cached"}
        inst = L2.build(cell, Path(root), m=m, tau=tau)
        Path(cache).mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(inst, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        return {"key": key, "error": traceback.format_exc(limit=4)}
    return {**_row(inst, key, time.perf_counter() - t0, path.stat().st_size), "status": "built"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "l2reg_train_v1")
    ap.add_argument("--phase", required=True, choices=sorted(L2.PHASES))
    ap.add_argument("--arm", default="", choices=sorted(L2.ARMS),
                    help="'' is the simulated gene trees; 'est_bp1000' the IQ-TREE re-estimates. "
                         "It selects the instance cache DIRECTORY, not just a column.")
    # Rep WINDOW, spelled as `rl_instances.py` spells it: --reps is an exclusive upper bound and
    # --min-rep an inclusive lower one, so a phase can be built in two disjoint windows into one
    # cache and one summary.
    ap.add_argument("--reps", type=int, default=900,
                    help="exclusive upper bound on the replicate; the phases carry 0-899")
    ap.add_argument("--min-rep", type=int, default=0, help="inclusive lower bound")
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--m", type=int, default=1000, help="loci; every cell carries 1,000")
    ap.add_argument("--jobs", type=int, default=min((mp.cpu_count() or 2) - 4, 12))
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cells = [c for c in L2.discover(args.data, args.phase, args.arm)
             if args.min_rep <= c.rep < args.reps]
    if not cells:
        raise SystemExit(
            f"no replicates for {args.phase} arm {args.arm or 'true'} in reps "
            f"[{args.min_rep}, {args.reps}) under {args.data}")

    cache = I.cache_dir(args.out, args.phase, args.tau, args.arm)
    cache.mkdir(parents=True, exist_ok=True)
    root = args.out / args.phase
    root.mkdir(parents=True, exist_ok=True)
    have = sum((cache / f"{c.key}.pkl").exists() for c in cells)
    print(f"[l2] {args.phase} tau={args.tau} arm={args.arm or 'true'} "
          f"({L2.ARMS[args.arm]}): {len(cells)} cells in reps [{args.min_rep}, {args.reps}), "
          f"{have} cached, {args.jobs} jobs -> {cache}", flush=True)

    payload = [(args.phase, c.key, args.arm, str(args.data), str(root), str(cache), args.m,
                args.tau) for c in cells]
    t0, rows = time.perf_counter(), []
    with mp.Pool(min(args.jobs, len(cells))) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, payload, chunksize=1), 1):
            rows.append(r)
            if "error" not in r and (i % 50 == 0 or i == len(payload)):
                print(f"[l2] {i}/{len(payload)}  last {r['key']} k={r['k']} "
                      f"|C|={r['n_clusters']} cov={r['truth_coverage']:.4f} "
                      f"{r['seconds']:.1f}s", flush=True)
    print(f"[l2] {len(rows)} cells in {time.perf_counter() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values("key")
    if "error" in df and df["error"].notna().any():
        print(df.loc[df["error"].notna(), "error"].iloc[0])
        print(f"[l2] {int(df['error'].notna().sum())} cells FAILED")
        return 1
    df = df.drop(columns=[c for c in ("status", "error") if c in df.columns])

    out = paths.artifact(args.out, I.summary_name(args.phase, args.tau, args.arm, args.tag))
    if out.exists():
        prev = pd.read_parquet(out)
        keep = prev[~prev["key"].isin(set(df["key"]))]
        if len(keep):
            df = pd.concat([df, keep], ignore_index=True).sort_values("key")
    on_disk = {p.stem for p in cache.glob("*.pkl")}
    gone = set(df["key"]) - on_disk
    if gone:
        print(f"[l2] dropping {len(gone)} summary rows with no .pkl in {cache}")
        df = df[df["key"].isin(on_disk)].reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"[l2] summary covers {len(df)} cells -> {out}")

    print(df.groupby(["stratum", "k"]).agg(
        cells=("n_clusters", "size"), nC=("n_clusters", "median"),
        nC_max=("n_clusters", "max"), edges=("n_edges", "median"),
        largest=("largest_component", "max"), cov=("truth_coverage", "median"),
        cov_min=("truth_coverage", "min"), seconds=("seconds", "median")).to_string())

    full = float((df["truth_coverage"] >= 1.0).mean())
    band = "GREEN" if full >= 0.99 else ("AMBER" if full >= 0.90 else "RED")
    print(f"\n[G-R0c] cells whose candidate set holds all of C(N_true) at tau={args.tau}: "
          f"{full:.1%} -> {band}  (min per-cell {df['truth_coverage'].min():.4f})")
    big = int(df["largest_component"].max())
    b8 = int(df["n_edges"].nlargest(8).sum())
    big_band = "GREEN" if big <= 5e3 else "AMBER" if big <= 2e4 else "RED"
    print(f"[G-R0b] largest overlap component {big} -> {big_band}; worst batch-8 edge count "
          f"{b8:,} (RL_PLAN section 1.3 calls 2e8 unviable)")
    print(f"[l2] {df['bytes'].sum() / 2**30:.2f} GiB on disk, "
          f"{np.median(df['bytes']) / 2**20:.1f} MiB median -> {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
