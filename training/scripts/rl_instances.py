#!/usr/bin/env python
"""Build the RL training instances once, into ``results/<phase>/instances_tau<t>/``.

    .venv/bin/python -u scripts/rl_instances.py --phase s50 --tau 2 --jobs 12
    .venv/bin/python -u scripts/rl_instances.py --phase s20 --tau 2 --jobs 12

``RL_PLAN.md`` §7's ``instances/{cell_id}_tau{tau}`` file, as a pickle rather than an npz: the
ground set is a dataclass of ragged per-component arrays, and flattening it into an npz would be a
second representation of something ``opc.solve.candidates`` already owns.

**Why it is a separate stage and not a lazy build inside the trainer.**  Two reasons, and the
second is the one that bites:

* The **oracle-F1 arm** spends no MPL at all, so its whole step is rollouts plus a policy
  backward.  At 0.4-2.9 s to build a ground set, a lazy build would make instance preparation the
  dominant cost of the arm whose entire purpose is to be cheap enough to carry the ablations.
* A build that happens in a pool worker inside a training loop fails *there*, mid-run, on cell 700
  of 960.  Here it fails in a stage whose only job is to fail early, and the summary below says
  which cells are missing before a run is launched against them.

The design filters default to the **full S50 design** v1 trains on -- both strata, k in {1, 2, 4},
gamma in {0.5, 0.3}, every CU -- not to the sixty-cell corner W0 measured.  Over 192 cells at
tau = 2, greedy-by-support reaches cluster F1 0.90-0.96 on the *small* stratum and **0.69-0.86** on
the *large* one, bottoming at 0.538 at k = 4.  The corner is where a policy has two to nine
clusters of forty-nine to win; the large stratum is where it has sixteen to twenty-four of
fifty-two.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from opc import arms as armlayout  # noqa: E402
from opc import grid, paths  # noqa: E402
from opc.rl import instances as I  # noqa: E402


def _one(payload) -> dict:
    cell, root, cache, backend, m, tau, arm = payload
    t0 = time.perf_counter()
    try:
        inst = I.load_or_build(cell, root, cache, backend=backend, m=m, tau=tau, arm=arm)
    except Exception:  # noqa: BLE001
        return {"key": cell.key, "error": traceback.format_exc(limit=4)}
    return {
        "key": cell.key, "arm": inst.arm, "n": inst.n, "rep": inst.rep, "k": inst.k,
        "cu": inst.cu,
        "gamma": inst.gamma, "stratum": inst.stratum, "tau": inst.tau, "m": inst.m,
        "n_clusters": inst.n_clusters, "kernel": int(inst.gs.kernel_size),
        "n_components": len(inst.gs.components),
        "largest_component": max((c.n_vertices for c in inst.gs.components), default=0),
        "n_edges": int(sum(c.n_edges for c in inst.gs.components)),
        "n_true": inst.n_true, "n_true_in_ctau": int(inst.truth_rows.size),
        "truth_coverage": inst.truth_coverage, "n_triples": int(inst.index.shape[0]),
        "seconds": time.perf_counter() - t0,
        "bytes": (Path(cache) / f"{cell.key}.pkl").stat().st_size,
    }


def select(phase: str, args) -> list:
    """The cells of ``phase`` this run covers.

    ``k = 0`` is excluded **by default** rather than by the ``--k`` filter: a tree has no
    reticulation, so the budget mask has nothing to be about and ``RL_PLAN`` §2's "k is given at
    inference" degenerates.  The r = 0 cells stay in the grid because A2's blocked design needs
    them as the per-stratum null; they are not training instances.

    ``--include-k0`` lifts that for **evaluation**, where a zero reticulation budget is a case with
    an answer rather than a degenerate one: no swap can improve anything, so the question is
    whether a policy knows to leave the greedy selection alone.  A84 measured that as the clearest
    thing training learned -- BC lands 0.0102 *below* greedy on the 59 pure trees of the external
    n = 100 set and the trained policy is neutral -- and it was invisible at n = 50 precisely
    because no S-grid instance set has a k = 0 cell in it.

    The ``--gamma`` filter is skipped for a cell drawn from a gamma *range*: gamma is not a factor
    of those designs, and ``Cell.gamma`` there is a summary rather than a level to select on.
    """
    cells = grid.enumerate_cells(**grid.PHASES[phase])
    cus = None if args.cu == "all" else {float(x) for x in args.cu.split(",")}
    ks = None if args.k == "all" else {int(x) for x in args.k.split(",")}
    gam = None if args.gamma == "all" else {float(x) for x in args.gamma.split(",")}
    strata = None if args.stratum == "all" else set(args.stratum.split(","))
    out = [c for c in cells
           if c.m_max == args.m and (args.include_k0 or c.r > 0)
           and args.min_rep <= c.rep < args.reps
           and (cus is None or c.cu in cus)
           and (ks is None or c.r in ks)
           and (gam is None or c.gamma_range is not None or c.gamma in gam)
           and (strata is None or c.stratum in strata)]
    return sorted(out, key=lambda c: c.key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=sorted(grid.PHASES), default="s50")
    ap.add_argument("--backend", default="julia")
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--cu", default="all")
    ap.add_argument("--k", default="all",
                    help="every reticulate k the phase defines: {1,2,4} at n=50, {1,2} at n=20, "
                         "{1,2,3} at n=30.  k = 0 is excluded unconditionally -- a tree has no "
                         "reticulation for the budget mask to be about")
    ap.add_argument("--gamma", default="0.5,0.3",
                    help="gamma = 0.1 is the plan's negative control and is where E0's only "
                         "coverage loss sits; excluded from training by default")
    ap.add_argument("--stratum", default="all")
    ap.add_argument("--include-k0", action="store_true",
                    help="keep the k = 0 (pure tree) cells.  Off for training -- a tree has no "
                         "reticulation for the budget mask to be about -- and on for the "
                         "benchmark grids, where a zero budget is the stratum A84's restraint "
                         "finding lives in, and no S-grid has one")
    ap.add_argument("--reps", type=int, default=20,
                    help="replicates to take.  The S-grids have 20; the benchmark grids have 100 "
                         "and the default would silently take a fifth of one")
    ap.add_argument("--min-rep", type=int, default=0,
                    help="lower bound on the replicate, so a rep WINDOW can be selected.  The "
                         "estimated arm exists only at reps 80-99 on the b-grids (the test "
                         "splits), and nothing else expresses that: --reps is an upper bound")
    ap.add_argument("--arm", default="",
                    help="which gene trees to build from: '' is the simulated ones, 'est_bp1000' "
                         "the IQ-TREE re-estimates unpacked by scripts/unpack_alisim.py.  It is "
                         "in the cache DIRECTORY name, not just in a column -- every reader here "
                         "globs *.pkl, so two arms in one directory would silently double them")
    ap.add_argument("--jobs", type=int, default=min((mp.cpu_count() or 2) - 4, 12))
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    root = args.out / args.phase
    cache = I.cache_dir(args.out, args.phase, args.tau, args.arm)
    cells = select(args.phase, args)
    if not cells:
        print("[inst] no cells selected")
        return 2

    # Fail before the pool, not inside it.  A missing arm directory is the likely mistake here --
    # `--arm est_bp1000` without having run scripts/unpack_alisim.py -- and `build` would raise it
    # once per worker, 900 times, with the real cause buried.
    d = armlayout.arm_dirs(root, args.backend, args.arm)
    missing = [c.key for c in cells
               if not (d.trees / f"{c.key}.nwk.gz").exists()
               or not (d.clusters / f"{c.key}.parquet").exists()]
    if missing:
        print(f"[inst] {len(missing)} of {len(cells)} cells have no {args.arm or 'true'} gene "
              f"trees or cluster table under {d.trees.parent}, e.g. {missing[:3]}.\n"
              f"       For an estimated arm: .venv/bin/python scripts/unpack_alisim.py "
              f"--phases {args.phase}" + (" --min-rep 80" if args.min_rep else ""))
        return 2
    cache.mkdir(parents=True, exist_ok=True)
    have = sum((cache / f"{c.key}.pkl").exists() for c in cells)
    print(f"[inst] {args.phase} tau={args.tau} arm={args.arm or 'true'}: {len(cells)} cells, "
          f"{have} cached, {args.jobs} jobs -> {cache}", flush=True)

    t0 = time.perf_counter()
    payload = [(c, root, cache, args.backend, args.m, args.tau, args.arm) for c in cells]
    with mp.Pool(min(args.jobs, len(cells))) as pool:
        rows = list(pool.imap_unordered(_one, payload, chunksize=1))
    print(f"[inst] {len(rows)} cells in {time.perf_counter() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values("key")
    bad = int(df["error"].notna().sum()) if "error" in df else 0
    if bad:
        print(df.loc[df["error"].notna(), "error"].iloc[0])
        print(f"[inst] {bad} cells FAILED")
        return 1

    # **The summary MERGES rather than overwrites**, and ``rungrid``'s estimated-arm collect is
    # the precedent one level up: "without this, a later ``--bp 200`` run would overwrite the
    # collected file with only its own rows."  The same hazard is reachable here through
    # ``--min-rep``, which exists precisely so a phase can be built in rep *windows* -- ``x75`` is
    # 300 train cells plus a disjoint 100-cell validation window, two invocations into one cache.
    # An overwrite leaves the file describing the last window alone, and ``w10_gnn.pos_weight_of``
    # then refuses the run because the summary does not cover every training key.
    #
    # Reconciled against the cache directory rather than merged blindly: a row whose ``.pkl`` is
    # gone is dropped, so the file never claims an instance that is not there.
    out = paths.artifact(args.out, I.summary_name(args.phase, args.tau, args.arm, args.tag))
    if out.exists():
        prev = pd.read_parquet(out)
        keep = prev[~prev["key"].isin(set(df["key"]))]
        if len(keep):
            df = pd.concat([df, keep], ignore_index=True).sort_values("key")
    on_disk = {p.stem for p in cache.glob("*.pkl")}
    missing = set(df["key"]) - on_disk
    if missing:
        print(f"[inst] dropping {len(missing)} summary rows with no .pkl in {cache}")
        df = df[df["key"].isin(on_disk)].reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"[inst] summary covers {len(df)} cells -> {out}")

    print(df.groupby(["stratum", "k"]).agg(
        cells=("n_clusters", "size"), nC=("n_clusters", "median"),
        nC_max=("n_clusters", "max"), edges=("n_edges", "median"),
        largest=("largest_component", "max"), cov=("truth_coverage", "median"),
        cov_min=("truth_coverage", "min"), seconds=("seconds", "median")).to_string())

    full = float((df["truth_coverage"] >= 1.0).mean())
    band = "GREEN" if full >= 0.99 else ("AMBER" if full >= 0.90 else "RED")
    print(f"\n[G-R0c] cells whose candidate set holds all of C(N_true) at tau={args.tau}: "
          f"{full:.1%} -> {band}  (min per-cell {df['truth_coverage'].min():.4f})")
    # G-R0b is a statement about what the encoder has to carry, so state it at the batch it will
    # actually carry rather than one cell at a time.
    big = int(df["largest_component"].max())
    b8 = int(df["n_edges"].nlargest(8).sum())
    big_band = "GREEN" if big <= 5e3 else "AMBER" if big <= 2e4 else "RED"
    print(f"[G-R0b] largest overlap component {big} -> {big_band}; worst batch-8 edge count "
          f"{b8:,} (RL_PLAN section 1.3 calls 2e8 unviable)")
    print(f"[inst] {df['bytes'].sum() / 2**30:.2f} GiB on disk, "
          f"{np.median(df['bytes']) / 2**20:.1f} MiB median -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
