#!/usr/bin/env python
"""Grid driver: networks -> gene trees -> the tau = 0 candidate set.

Usage::

    .venv/bin/python -m opc.rungrid --phase smoke
    .venv/bin/python -m opc.rungrid --phase s50 --dry-run
    .venv/bin/python -m opc.rungrid --phase s50 --backend julia --jobs 48 --keep-trees

Three stages, each idempotent and resumable (a cell whose output is newer than its input is
skipped, the ``needs_rerun`` pattern from
``/shared/ns58/MPPhySMC/scripts/run_phase3_budget_sweep.py``):

1. ``networks``  -- one base topology per ``(n, replicate, stratum)``; every other axis derives.
2. ``simulate``  -- gene trees, then the cluster/support table at **every** observed cluster.
3. ``collect``   -- one summary row per cell to ``results/clusters_<phase>.parquet``.

**Stage 2 stores support >= 1, not support >= 2.**  The percolation study only ever read the
table at ``tau >= 2``, so pruning there was lossless.  ``exp-plan-n50-mpl-v0.1`` makes tau = 0 --
every cluster induced by at least one gene tree -- the headline setting, and the clusters that
appear exactly once are disproportionately the hybrid-signature ones E0 is measuring.  Pruning
them would delete the measurement.

Children are pinned to one thread each so a large pool does not oversubscribe the box.

``--estimate`` adds the gene-tree-estimation-error arm (SPEC-AMENDMENTS A15): alignments from the
cached true gene trees with AliSim, re-estimation with IQ-TREE, and the same metrics recomputed on
the estimated trees.  It runs alongside the true arm, never instead of it -- the true arm's
outputs, ``config_hash`` included, are untouched::

    .venv/bin/python -m opc.rungrid --phase full --backend julia --estimate --est-only \
        --bp 500 --est-m 1000 --jobs 190
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

#: repo root: ``src/opc/rungrid.py`` -> ``src/opc`` -> ``src`` -> here.
REPO = Path(__file__).resolve().parents[2]

# Pin BLAS/OpenMP before numpy is imported anywhere in a worker.
for _var in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS", "JULIA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from opc import arms as armlayout  # noqa: E402
from opc import grid, paths  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.simulate import msnc_oracle as oracle  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

PHASES = grid.PHASES


@dataclass(frozen=True)
class Arm:
    """Which gene trees ``run_cell`` reads.

    ``ARM_TRUE`` is the pre-existing behaviour and the default, so a run without ``--estimate``
    produces byte-identical output to before this arm existed.
    """

    name: str = "true"
    bp: int = 0
    tree_dir: Path | None = None
    cell_dir: Path | None = None
    clu_dir: Path | None = None
    est_m: int = 0
    sidecar_dir: Path | None = None

    @property
    def estimated(self) -> bool:
        return self.name != "true"


ARM_TRUE = Arm()


# --------------------------------------------------------------------------------- stage 1


def build_base_network(args):
    key, spec, out_dir, master_seed = args
    path = out_dir / f"{key}.pkl"
    if path.exists():
        return key, "cached", 0.0
    t0 = time.perf_counter()
    # The stratum enters the seed as well as the placement rule.  Without it the two strata would
    # start from the same birth-death draw, and "small" and "large" would be correlated samples of
    # the same 20 topologies rather than independent ones.
    parts = ("base", spec.n, spec.rep) + ((spec.stratum,) if spec.stratum else ())
    try:
        net = N.build_base(
            spec.n, spec.r_max,
            seed=grid.chain_seed(master_seed, *parts),
            vertex_disjoint=spec.vertex_disjoint,
            cycle_size_range=spec.cycle_size_range,
        )
    except Exception:
        return key, "FAILED\n" + traceback.format_exc(limit=3), time.perf_counter() - t0
    # Gates that must hold before a network is allowed into the grid.
    assert N.level(net) == (1 if spec.r_max else 0), f"{key}: level != 1"
    assert net.n_retics == spec.r_max, f"{key}: wrong reticulation count"
    assert N.n_cluster_visible(net) == spec.r_max, f"{key}: a reticulation left no trace"
    lo, hi = spec.cycle_size_range
    sizes = N.cycle_sizes(net)
    assert all(lo <= s and (hi is None or s <= hi) for s in sizes), f"{key}: cycle sizes {sizes}"
    if spec.vertex_disjoint:
        assert N.is_galled(net), f"{key}: cycles share a vertex, so this is level-1 but not galled"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(net, fh)
    tmp.rename(path)
    return key, "built", time.perf_counter() - t0


# --------------------------------------------------------------------------------- stage 2

# Written by ``seqsim.run_est_job`` into ``<cell.key>.json``; carried into the parquet so a row
# is self-describing -- what model made the sequences, and how wrong the resulting trees were.
_EST_SIDECAR_COLS = (
    "rf_norm_mean", "rf_norm_median", "rf_norm_p90", "frac_rf_zero",
    "frac_root_split_correct", "n_rf_scored", "n_root_polytomy", "n_loci", "n_alisim_retries",
    "aln_len_mean", "aln_len_min", "aln_len_max", "gap_fraction",
    "alisim_seconds", "iqtree_seconds", "loci_per_iqtree_batch", "iqtree_model_used",
    "graft_t_out", "graft_outgroup_subs_per_site", "graft_root_height_mean",
    "graft_root_height_p99", "graft_n_outgroup_depth_clamped",
    "seed_alisim", "seed_iqtree", "iqtree_version",
)
_EST_MODEL_COLS = (
    "branch_scale", "theta", "kappa", "base_freqs", "gamma_shape", "gamma_cats",
    "indel_ins", "indel_del", "indel_size", "outgroup_alpha", "inference_model",
    "fast", "keep_ident", "sim_model",
)


def _add_estimation_columns(df, cell, arm) -> None:
    side = json.loads((arm.sidecar_dir / f"{cell.key}.json").read_text())
    df["arm"] = "estimated"
    df["bp"] = arm.bp
    for k in _EST_SIDECAR_COLS:
        df["est_" + k] = side.get(k)
    for k in _EST_MODEL_COLS:
        df[k] = side.get(k)
    # The single number that says how much signal a locus carried.
    df["expected_subs_per_internal_branch"] = cell.cu * side.get("branch_scale", 0.0) * arm.bp
    # A *separate* identity for the estimation arm -- ``config_hash`` above is untouched, so
    # every published true-arm hash keeps its value (SPEC-AMENDMENTS A15).
    df["est_hash"] = grid.config_hash({
        **cell.as_dict(), "bp": arm.bp, "est_m": arm.est_m,
        **{k: side.get(k) for k in _EST_MODEL_COLS},
    })


def run_cell(args):
    cell, net_dir, out_dir, backend, keep_trees, tree_dir, clu_dir = args[:7]
    arm = args[7] if len(args) > 7 else ARM_TRUE
    if arm.estimated:
        # The three directory names come from ``opc.arms``, which owns the convention -- this used
        # to re-derive the cluster directory from the cell directory's *name*, which worked only
        # because those two happen to share a suffix that the tree directory does not.
        out_dir, tree_dir, clu_dir = arm.cell_dir, arm.tree_dir, arm.clu_dir
        clu_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cell.key}.parquet"
    base_path = net_dir / f"{cell.base_key}.pkl"
    # For the estimated arm the input is the estimated tree file, not the base network.
    src_path = base_path if not arm.estimated else tree_dir / f"{cell.key}.nwk.gz"
    if out.exists() and src_path.exists() and out.stat().st_mtime > src_path.stat().st_mtime:
        return cell.key, "cached", 0.0
    t0 = time.perf_counter()
    try:
        with open(base_path, "rb") as fh:
            base = pickle.load(fh)
        net = N.derive(base, r=cell.r, cu=cell.cu, gamma=cell.gamma,
                       gamma_range=cell.gamma_range, gamma_seed=cell.gamma_seed)
        checkpoints = cell.checkpoints
        if arm.estimated and arm.est_m:
            checkpoints = tuple(c for c in cell.checkpoints if c <= arm.est_m) or (arm.est_m,)

        # ---- gene trees -> cluster support table
        if arm.estimated:
            # Rooted, outgroup-pruned, bare t1..tN labels -- consumed unchanged.
            counter = extract.count_newick_file(
                tree_dir / f"{cell.key}.nwk.gz",
                {lab: i for i, lab in enumerate(net.labels)},
                checkpoints,
            )
        elif backend == "oracle":
            counter = extract.SupportCounter(cell.n, checkpoints)
            idx = {lab: i for i, lab in enumerate(net.labels)}
            trees = oracle.simulate_gene_trees(net, cell.m_max, seed=cell.seed)
            buf: list[int] = []
            for nwk in trees:
                buf.clear()
                counter.add_locus(extract.clusters_from_newick(nwk, idx, buf))
            if keep_trees:
                import gzip

                with gzip.open(tree_dir / f"{cell.key}.nwk.gz", "wt") as fh:
                    fh.write("\n".join(trees) + "\n")
        elif backend == "julia":
            # The Julia stage already ran and wrote the trees; just consume them.
            counter = extract.count_newick_file(
                tree_dir / f"{cell.key}.nwk.gz",
                {lab: i for i, lab in enumerate(net.labels)},
                cell.checkpoints,
            )
        else:
            raise ValueError(f"unknown backend {backend!r}")

        # tau = 0: keep every cluster induced by at least one gene tree.  See the module
        # docstring -- pruning at support >= 2 would delete the measurement E0 exists to make.
        masks, support, n_raw = counter.finalize(min_support=1)
        meta = N.metadata(net)
        extract.write_cluster_table(
            clu_dir / f"{cell.key}.parquet", masks, support, checkpoints, cell.n, n_raw,
            backend=backend, seed=cell.seed, config_hash=grid.config_hash(cell.as_dict()),
            **{k: v for k, v in cell.as_dict().items()},
        )

        # One summary row per (cell, m).  The cluster table itself is the artifact; this is the
        # index over it, so a census can be planned without opening 1,600 files.
        rows = []
        for i, m in enumerate(checkpoints):
            sup = support[:, i]
            rows.append({
                "m": m,
                "n_clusters_tau0": int((sup >= 1).sum()),
                "n_clusters_tau2": int((sup >= 2).sum()),
                "n_clusters_raw": int(n_raw[i]),
                "n_singleton_support": int((sup == 1).sum()),
                "max_support": int(sup.max()) if sup.size else 0,
            })
        df = pd.DataFrame(rows)
        for k, v in cell.as_dict().items():
            df[k] = v
        df["backend"] = backend
        df["seed"] = cell.seed
        df["config_hash"] = grid.config_hash(cell.as_dict())
        df["realized_median_cu"] = meta["realized_median_internal_cu"]
        df["singleton_hybrid_fraction"] = meta["singleton_hybrid_fraction"]
        df["n_clusters_true"] = meta["n_clusters_true"]
        df["n_retics_visible"] = meta["n_retics_cluster_visible"]
        df["cycle_size_min"] = min(meta["cycle_sizes"], default=0)
        df["cycle_size_max"] = max(meta["cycle_sizes"], default=0)
        df["n_nonbinary_loci"] = counter.n_nonbinary_loci
        # New columns land only in the estimated arm's parquets: the true arm keeps the exact
        # schema it published, and the two arms are collected into separate files anyway.
        if arm.estimated:
            _add_estimation_columns(df, cell, arm)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(out)
        return cell.key, "ok", time.perf_counter() - t0
    except Exception:
        return cell.key, "FAILED\n" + traceback.format_exc(limit=5), time.perf_counter() - t0


# --------------------------------------------------------------------------------- driver


def _select_est_cells(cells, args):
    """Filter the cell list for the estimation arm.  Never touches the true arm's cell list."""
    def keep(c):
        for spec, value in ((args.est_n, c.n), (args.est_r, c.r),
                            (args.est_cu, c.cu), (args.est_gamma, c.gamma)):
            if spec and str(value) not in {s.strip() for s in spec.split(",")}:
                return False
        return not (args.est_max_rep and c.rep >= args.est_max_rep)

    return [c for c in cells if keep(c)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), default="smoke")
    # 48, not cpu_count: the shared RLIMIT_NPROC on this box counts *threads*, and a pool past
    # ~100 workers hits it during a fork, which kills multiprocessing's _handle_workers thread
    # and hangs the pool silently and indefinitely.  See CLAUDE.md.
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() - 2, 48))
    ap.add_argument("--backend", choices=("oracle", "julia"), default="oracle")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--root-phase", default=None,
                    help="share another phase's caches (networks, gene trees, cluster tables) "
                         "so a phase can extend another in place rather than re-simulating it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-trees", action="store_true", help="persist gzipped Newick gene trees")

    est = ap.add_argument_group(
        "gene-tree estimation error (SPEC-AMENDMENTS A15)",
        "AliSim alignments from the cached true gene trees, re-estimated with IQ-TREE. "
        "Requires --backend julia: the oracle backend emits topology only, with no branch "
        "lengths to evolve sequences along.")
    est.add_argument("--estimate", action="store_true", help="enable the estimated-tree arm")
    est.add_argument("--bp", default="500", help="comma-separated root sequence lengths")
    est.add_argument("--est-m", type=int, default=0,
                     help="use only the first M loci of each cell (0 = all); the m checkpoints "
                          "are nested prefixes, so the arms stay paired")
    est.add_argument("--est-only", action="store_true", help="skip the true arm's stage 2")
    est.add_argument("--branch-scale", type=float, default=0.018,
                     help="substitutions/site per coalescent unit (theta = 2 x this)")
    est.add_argument("--kappa", type=float, default=2.0)
    est.add_argument("--gamma-shape", type=float, default=1.0, help="+G4 alpha; 0 disables")
    est.add_argument("--indel", default="0.03,0.03", help="insertion,deletion rate")
    est.add_argument("--indel-size", default="POW{1.7/100}")
    est.add_argument("--outgroup-alpha", type=float, default=0.5,
                     help="outgroup divergence as (1+alpha) x p99 of the root heights")
    est.add_argument("--est-model", default="HKY+F+G4", help="IQ-TREE inference model")
    est.add_argument("--no-fast", action="store_true", help="drop IQ-TREE --fast")
    est.add_argument("--est-n", default="", help="restrict to these n (comma-separated)")
    est.add_argument("--est-r", default="", help="restrict to these r")
    est.add_argument("--est-cu", default="", help="restrict to these cu")
    est.add_argument("--est-gamma", default="", help="restrict to these gamma")
    est.add_argument("--est-max-rep", type=int, default=0, help="use only the first R replicates")
    est.add_argument("--est-scratch", type=Path, default=Path("/tmp"),
                     help="scratch for alignments and IQ-TREE aux files (tmpfs by default)")
    est.add_argument("--est-target-seconds", type=float, default=900.0,
                     help="target wall time per task; longer cells are split into chunks")
    est.add_argument("--iq-target-ns", type=float, default=4e6,
                     help="n_taxa x sites budget per IQ-TREE process")
    est.add_argument("--iqtree", type=Path, default=None)
    args = ap.parse_args()

    if args.estimate and args.backend != "julia":
        ap.error("--estimate requires --backend julia (the oracle emits no branch lengths)")

    cells = grid.enumerate_cells(**PHASES[args.phase])
    bases = grid.base_networks(cells)
    n_gene_trees = sum(c.m_max for c in cells)

    print(f"phase={args.phase} backend={args.backend} jobs={args.jobs}")
    print(f"  base topologies : {len(bases)}")
    print(f"  replicate-runs  : {len(cells)}")
    print(f"  gene trees      : {n_gene_trees:,}")
    print(f"  summary rows    : {sum(len(c.checkpoints) for c in cells):,}")
    strata = sorted({c.stratum for c in cells})
    if strata != [""]:
        print(f"  strata          : {', '.join(f'{s} {grid.STRATA[s]}' for s in strata)}"
              f"  (vertex-disjoint: {cells[0].vertex_disjoint})")

    est_cells, bp_list, seq_model = [], [], None
    if args.estimate:
        from opc.simulate import seqsim

        if args.iqtree:
            seqsim.IQTREE = args.iqtree
        ins, _, dele = args.indel.partition(",")
        seq_model = seqsim.SeqModel(
            branch_scale=args.branch_scale, kappa=args.kappa, gamma_shape=args.gamma_shape,
            indel_ins=float(ins), indel_del=float(dele or ins), indel_size=args.indel_size,
            outgroup_alpha=args.outgroup_alpha, inference_model=args.est_model,
            fast=not args.no_fast,
        )
        bp_list = [int(x) for x in args.bp.split(",") if x.strip()]
        est_cells = _select_est_cells(cells, args)
        print(f"  -- estimation arm: {len(est_cells)} cells x bp {bp_list} "
              f"(m<={args.est_m or 'all'})")
        print(f"     {seq_model.sim_model_string()} -> {seq_model.inference_model}, "
              f"branch_scale={seq_model.branch_scale} (theta={seq_model.theta:g})")
        total = 0.0
        for bp in bp_list:
            secs = sum((min(args.est_m, c.m_max) if args.est_m else c.m_max)
                       * seqsim.seconds_per_locus(c.n, bp) for c in est_cells)
            total += secs
            subs = sorted({round(seq_model.expected_subs_per_internal_branch(c.cu, bp), 2)
                           for c in est_cells})
            print(f"     bp={bp:<5d} {secs / 3600:8.1f} CPU-h  "
                  f"({secs / 3600 / max(1, args.jobs):5.1f} h at --jobs {args.jobs})  "
                  f"expected subs/internal branch {subs}")
        pen = seqsim.CONCURRENCY_PENALTY if args.jobs >= 64 else 1.0
        print(f"     total {total / 3600:.1f} CPU-h "
              f"({total / 3600 / max(1, args.jobs):.1f} h at --jobs {args.jobs})")
        if pen > 1.0:
            print(f"     realistically ~{total * pen / 3600:.0f} CPU-h "
                  f"(~{total * pen / 3600 / max(1, args.jobs):.1f} h): the per-locus cost is "
                  f"measured single-process, and at --jobs {args.jobs} this workload is "
                  f"memory-bandwidth bound (~{pen:g}x)")
    if args.dry_run:
        return 0

    root = args.out / (args.root_phase or args.phase)
    # The tree cache MUST be keyed by backend: without it a julia run silently re-reads trees
    # the oracle wrote, and the two backends stop being independent.
    net_dir = root / "networks"
    true_arm = armlayout.arm_dirs(root, args.backend)
    cell_dir, tree_dir, clu_dir = true_arm.cells, true_arm.trees, true_arm.clusters
    for d in (net_dir, cell_dir, tree_dir, clu_dir):
        d.mkdir(parents=True, exist_ok=True)

    from multiprocessing import Pool

    t0 = time.perf_counter()
    jobs = [(k, spec, net_dir, cells[0].master_seed) for k, spec in sorted(bases.items())]
    with Pool(args.jobs) as pool:
        results = pool.map(build_base_network, jobs)
    failures = [(k, s) for k, s, _ in results if s.startswith("FAILED")]
    print(f"[networks] {len(results) - len(failures)}/{len(results)} ok "
          f"in {time.perf_counter() - t0:.1f}s")
    for k, s in failures[:5]:
        print(f"  FAILED {k}: {s.splitlines()[-1]}")
    if failures:
        return 1

    if args.backend == "julia":
        from opc.simulate import julia_backend

        if not julia_backend.available():
            print("julia backend unavailable (no toolchain or Manifest.toml)")
            return 1
        enw_dir = root / "enewick"
        enw_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        jl_jobs = []
        for c in cells:
            out_gz = tree_dir / f"{c.key}.nwk.gz"
            if out_gz.exists():
                continue
            with open(net_dir / f"{c.base_key}.pkl", "rb") as fh:
                base = pickle.load(fh)
            net = N.derive(base, r=c.r, cu=c.cu, gamma=c.gamma,
                           gamma_range=c.gamma_range, gamma_seed=c.gamma_seed)
            enw = enw_dir / f"{c.key}.enewick"
            enw.write_text(N.to_enewick(net) + "\n")
            jl_jobs.append(dict(
                enewick=str(enw), out=str(out_gz), n_loci=c.m_max, seed=c.seed % (2**31),
                n_taxa=c.n, r=c.r, cu=c.cu,
            ))
        if jl_jobs:
            fails = julia_backend.run_shards(jl_jobs, root / "manifests", args.jobs)
            print(f"[julia]    {len(jl_jobs) - len(fails)}/{len(jl_jobs)} networks simulated "
                  f"in {time.perf_counter() - t0:.1f}s")
            for line in fails[:5]:
                print("  " + line)
            if fails:
                return 1
        else:
            print("[julia]    all gene trees cached")

    # ---------------------------------------------------------------- stage 1.75: sequences
    arms = [] if args.est_only else [ARM_TRUE]
    if args.estimate:
        from opc.simulate import seqsim

        if not seqsim.available():
            print(f"iqtree2 not found at {seqsim.IQTREE}")
            return 1
        missing = [c.key for c in est_cells if not (tree_dir / f"{c.key}.nwk.gz").exists()]
        if missing:
            print(f"[estimate] {len(missing)} cells have no cached gene trees, e.g. {missing[:3]}")
            return 1

        est_jobs = [j for bp in bp_list
                    for j in seqsim.plan_jobs(est_cells, bp, seq_model, root, args.backend,
                                              args.est_m, args.est_scratch,
                                              cells[0].master_seed,
                                              target_seconds=args.est_target_seconds,
                                              iq_target_ns=args.iq_target_ns)]
        # Longest-first with imap_unordered: task cost spans three orders of magnitude across n,
        # and pool.map's static chunking would strand one worker on the n=200 tail.
        est_jobs.sort(key=lambda j: -j.predicted_seconds())
        t0 = time.perf_counter()
        done = 0
        failures = []
        with Pool(args.jobs) as pool:
            for key, status, _ in pool.imap_unordered(seqsim.run_est_job, est_jobs, chunksize=1):
                done += 1
                if status.startswith("FAILED"):
                    failures.append((key, status))
                if done % 200 == 0 or done == len(est_jobs):
                    print(f"[estimate] {done}/{len(est_jobs)} tasks, {len(failures)} failed, "
                          f"{time.perf_counter() - t0:.0f}s elapsed", flush=True)
        for k, s in failures[:5]:
            print(f"  FAILED {k}:\n{s}")
        if failures:
            return 1
        for bp in bp_list:
            for c in est_cells:
                seqsim.merge_chunks(c.key, bp, root, args.backend)
            d = armlayout.arm_dirs(root, args.backend, armlayout.arm_of_bp(bp))
            d.cells.mkdir(parents=True, exist_ok=True)
            arms.append(Arm(name=f"est_bp{bp}", bp=bp, tree_dir=d.trees, cell_dir=d.cells,
                            clu_dir=d.clusters, est_m=args.est_m, sidecar_dir=d.trees))
        print(f"[estimate] {len(est_jobs)} tasks in {time.perf_counter() - t0:.0f}s")

    for arm in arms:
        t0 = time.perf_counter()
        subset = cells if not arm.estimated else est_cells
        jobs = [(c, net_dir, cell_dir, args.backend, args.keep_trees, tree_dir, clu_dir, arm)
                for c in subset]
        with Pool(args.jobs) as pool:
            results = pool.map(run_cell, jobs, chunksize=1)
        failures = [(k, s) for k, s, _ in results if s.startswith("FAILED")]
        times = [t for _, s, t in results if not s.startswith("FAILED")]
        print(f"[simulate] {arm.name}: {len(results) - len(failures)}/{len(results)} ok "
              f"in {time.perf_counter() - t0:.1f}s "
              f"(per-cell median {np.median(times) if times else 0:.2f}s, "
              f"max {max(times) if times else 0:.2f}s)")
        for k, s in failures[:5]:
            print(f"  FAILED {k}:\n{s}")
        if failures:
            return 1

    # The index over the cluster tables, not the tables themselves: one row per (cell, m).  The
    # estimated arms go to a separate file rather than an ``arm`` column in the same one, so that
    # any groupby over the true arm cannot silently double-count.
    if not args.est_only:
        parts = sorted(cell_dir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        out = paths.artifact(args.out, f"clusters_{args.phase}.parquet")
        df.to_parquet(out, index=False)
        print(f"[collect]  {len(df):,} rows -> {out}")
        big = df[df.m == df.m.max()]
        print(f"           |C| at tau=0, m={df.m.max()}: median {big.n_clusters_tau0.median():.0f}"
              f"  p95 {big.n_clusters_tau0.quantile(0.95):.0f}"
              f"  max {big.n_clusters_tau0.max():.0f}"
              f"  (support-1 share {1 - (big.n_clusters_tau2 / big.n_clusters_tau0).mean():.3f})")

    # Collect *every* estimated arm on disk, not just the ones this invocation ran -- the
    # per-cell parquets are the source of truth, exactly as they are for the true arm.  Without
    # this, a later `--bp 200` run would overwrite the collected file with only its own rows.
    est_dirs = sorted(root.glob(f"cells_{args.backend}_est_bp*"))
    if est_dirs:
        frames = [pd.read_parquet(p) for d in est_dirs for p in sorted(d.glob("*.parquet"))]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            out = paths.artifact(args.out, f"clusters_{args.phase}_est.parquet")
            df.to_parquet(out, index=False)
            print(f"[collect]  {len(df):,} estimated rows -> {out}")
            for bp, g in df.groupby("bp"):
                print(f"           bp={bp}: RF {g['est_rf_norm_mean'].mean():.3f}, "
                      f"root split correct {g['est_frac_root_split_correct'].mean():.3f}, "
                      f"gaps {g['est_gap_fraction'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
