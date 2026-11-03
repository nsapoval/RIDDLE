#!/usr/bin/env python3
"""Take the AliSim hand-off's estimated gene trees back into ``results/`` -- the return leg of A99.

``scripts/pack_alisim.py`` shipped 1,880 cells of true gene trees to a CPU box; this reads back
what came home under ``handoff/alisim_gt_v1/estimated_bp1000/`` and lands it as a second gene-tree
**arm** beside the true one, so ``rl_instances.py --arm est_bp1000`` can build instances from it.

**It re-derives the rooted trees rather than copying them, and that is the whole reason this
exists.**  The package's ``<key>.nwk.gz`` is rooted and outgroup-pruned but carries **no branch
lengths**: ``outgroup.reroot_on_outgroup`` dropped them unconditionally until W13, a decision
taken when ``clusters_from_newick`` stripped ``:length`` anyway.  W13 then built twenty-one node
features out of those lengths, and a length-free tree does not fail -- it yields ``blen_* = 0``,
``bdepth_* = 1``, ``bsib_* = 0`` and a ``brank_*`` that is the order the nodes happen to appear
in the string.  So this reads ``<key>.raw.nwk.gz`` -- what IQ-TREE actually returned, unrooted
with ``OUT`` still attached -- and reroots it with ``keep_lengths=True``.

The lengths are in substitutions per site rather than coalescent units, and that needs no
conversion: every W13 statistic is a ratio *within one tree* (``opc.clusters.blen``'s module
docstring pre-registers exactly this), so the per-tree scalar separating the two systems cancels.

    .venv/bin/python -u scripts/unpack_alisim.py --phases x25,x50 --jobs 12
    .venv/bin/python -u scripts/unpack_alisim.py --phases b25,b50,b100,b250 --min-rep 80 --jobs 12

Three stages, each idempotent and skippable:

``reroot``
    ``<key>.raw.nwk.gz`` -> ``results/<phase>/genetrees_est_<backend>_bp<bp>/<key>.nwk.gz``, with
    the sidecar copied beside it (``rungrid._add_estimation_columns`` reads it from there).
``cells``
    ``opc.rungrid.run_cell`` on an estimated :class:`~opc.rungrid.Arm` -> the ``cells_`` and
    ``clusters_`` parquets.  That code path already existed and wants only
    ``networks/<base_key>.pkl`` plus the trees: **no AliSim, no IQ-TREE, no Julia.**
``report``
    ``results/validation/alisim_ingest_bp<bp>.json``.

**Why not ``python -m opc.rungrid --estimate --est-only``**, which would find the trees "cached"
(``seqsim.run_est_job`` skips on mtime) and run the same ``run_cell``: four reasons, any one of
them fatal.  The shipped trees have to be rewritten before they are usable at all, so the cached
path would be skipping work already done by hand; ``--est-max-rep R`` keeps reps ``0..R-1`` and
there is **no flag that selects reps 80-99**, which is what the b-grids have; the cached check is
an mtime comparison against files that arrived by ``rsync``, which is a property of the transfer
flags rather than of the data; and its ``[collect]`` step would write
``clusters_<phase>_est.parquet`` holding **20 of 100** cells under a name that reads as the whole
phase.

**G-IG0 is on by default.** Every emitted tree is compared against the package's own
``<key>.nwk.gz`` after stripping lengths, and the first disagreement refuses the cell.  Note what
that gate is and is not: ``handoff/alisim_gt_v1/pylib/opc/simulate/outgroup.py`` is
**byte-identical** to ``src/opc/simulate/outgroup.py``, so the shipped file was produced by the
*same function*.  This is therefore a self-consistency check between two branches of one function
at 1.88 M-tree scale -- which is what catches a bug in the new branch, and is worth having -- and
**not** the repository's cross-validate-against-an-independent-implementation convention.  That
one is discharged by dendropy, in ``tests/test_outgroup.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# Before numpy.  This driver runs no solver; the only thing that could take every core per worker
# is the BLAS behind numpy -- CLAUDE.md's second route into the silent pool hang (A44).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from opc import arms as armlayout  # noqa: E402
from opc import grid, rungrid  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.simulate import outgroup as og  # noqa: E402

#: Only the phases the package was built for.  A phase not here has no trees to unpack, and
#: guessing a directory name is how an empty cache reads as "nothing built yet" (``opc.arms``).
PHASES = ("x25", "x50", "b25", "b50", "b100", "b250")


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def reroot_cell(src_raw: Path, src_rooted: Path, dst: Path, taxa: set[str],
                verify: bool = True) -> dict:
    """One cell: reroot with lengths, check the taxon set, and (by default) check topology parity.

    The taxon-set check is not optional and is not paranoia.  ``clusters_from_newick`` raises on
    an *unknown* label and is **silent on a missing one**, and IQ-TREE's ``--keep-ident`` default
    drops identical sequences -- common at short alignments -- so a short taxon set would corrupt
    every cluster in the cell without erroring anywhere downstream.
    """
    n_loci = n_zero_len = 0
    min_bl, max_bl, n_zero_edges, n_edges = float("inf"), 0.0, 0, 0
    tmp = dst.with_name(dst.name + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with _open(src_raw) as fr, gzip.open(tmp, "wt") as out:
        ref = _open(src_rooted) if verify else None
        try:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                kept = og.reroot_on_outgroup(line, keep_lengths=True)
                bare = extract._STRIP.sub("", kept)

                _, ln, lab, ch = og.parse_newick(kept)
                got = {lab[i] for i in range(len(lab)) if not ch[i]}
                if got != taxa:
                    raise ValueError(
                        f"{src_raw}: locus {n_loci} has {len(got)} taxa, expected {len(taxa)} "
                        f"(missing {sorted(taxa - got)[:5]}, extra {sorted(got - taxa)[:5]}). "
                        f"clusters_from_newick is SILENT on a missing taxon, so this would have "
                        f"corrupted every cluster in the cell without raising.")

                if verify:
                    want = ref.readline().strip()
                    if bare != want:
                        raise ValueError(
                            f"{src_raw}: locus {n_loci} disagrees with the package's own rooted "
                            f"tree after stripping lengths.\n  got  {bare[:120]}\n  want "
                            f"{want[:120]}")

                edges = [ln[i] for i in range(1, len(ln))]
                if edges:
                    n_edges += len(edges)
                    n_zero_edges += sum(1 for e in edges if e <= 0.0)
                    min_bl = min(min_bl, min(edges))
                    max_bl = max(max_bl, max(edges))
                    if max(edges) <= 0.0:
                        n_zero_len += 1
                out.write(kept + "\n")
                n_loci += 1
        finally:
            if ref is not None:
                ref.close()
    tmp.rename(dst)
    return {
        "n_loci": n_loci,
        "n_zero_length_loci": n_zero_len,
        "frac_zero_length_edges": (n_zero_edges / n_edges) if n_edges else 0.0,
        "min_branch_length": 0.0 if min_bl == float("inf") else min_bl,
        "max_branch_length": max_bl,
    }


def _reroot_job(payload) -> dict:
    cell, src_dir, dst_dir, verify, force = payload
    raw = src_dir / f"{cell.key}.raw.nwk.gz"
    rooted = src_dir / f"{cell.key}.nwk.gz"
    dst = dst_dir / f"{cell.key}.nwk.gz"
    t0 = time.perf_counter()
    try:
        if not raw.exists():
            return {"key": cell.key, "error": f"no {raw}"}
        if (not force and dst.exists()
                and dst.stat().st_mtime > raw.stat().st_mtime):
            return {"key": cell.key, "status": "cached", "seconds": 0.0}
        taxa = {f"t{i}" for i in range(1, cell.n + 1)}
        stats = reroot_cell(raw, rooted, dst, taxa, verify=verify)
        side = src_dir / f"{cell.key}.json"
        if side.exists():
            shutil.copyfile(side, dst_dir / f"{cell.key}.json")
        return {"key": cell.key, "status": "ok",
                "seconds": time.perf_counter() - t0, **stats}
    except Exception:                                             # noqa: BLE001
        return {"key": cell.key, "error": traceback.format_exc(limit=4)}


def select(phase: str, min_rep: int, max_rep: int, src_dir: Path):
    """The phase's cells that are both in the rep window and actually present in the package."""
    cells = [c for c in grid.enumerate_cells(**grid.PHASES[phase])
             if min_rep <= c.rep < (max_rep or 10 ** 9)]
    return [c for c in cells if (src_dir / f"{c.key}.raw.nwk.gz").exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=None,
                    help="the package's estimated_bp<BP>/ directory "
                         "(default handoff/alisim_gt_v1/estimated_bp<bp>)")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--bp", type=int, default=1000)
    ap.add_argument("--backend", default="julia", choices=("julia", "oracle"))
    ap.add_argument("--min-rep", type=int, default=0)
    ap.add_argument("--max-rep", type=int, default=0, help="0 = no upper bound")
    ap.add_argument("--stage", default="all", choices=("all", "reroot", "cells"))
    ap.add_argument("--no-verify-parity", dest="verify", action="store_false",
                    help="skip G-IG0.  It is a per-locus string compare against the package's own "
                         "rooted tree and costs one extra gzip read; there is no good reason.")
    ap.add_argument("--jobs", "-j", type=int, default=min((os.cpu_count() or 4) - 4, 12))
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arm = armlayout.arm_of_bp(args.bp)
    src_root = args.src or (REPO / "handoff" / "alisim_gt_v1" / f"estimated_bp{args.bp}")
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bad = [p for p in phases if p not in grid.PHASES]
    if bad:
        ap.error(f"unknown phase(s) {bad}; known: {sorted(grid.PHASES)}")

    plan, report = [], {"arm": arm, "bp": args.bp, "src": str(src_root), "phases": {}}
    for phase in phases:
        src_dir = src_root / phase
        if not src_dir.is_dir():
            print(f"[unpack] {phase}: no {src_dir}, skipping")
            continue
        cells = select(phase, args.min_rep, args.max_rep, src_dir)
        n_window = len([c for c in grid.enumerate_cells(**grid.PHASES[phase])
                        if args.min_rep <= c.rep < (args.max_rep or 10 ** 9)])
        print(f"[unpack] {phase}: {len(cells)} of {n_window} cells in the rep window are in the "
              f"package")
        if cells:
            plan.append((phase, src_dir, cells))

    if args.dry_run or not plan:
        for phase, _, cells in plan:
            d = armlayout.arm_dirs(args.out / phase, args.backend, arm)
            print(f"  {phase}: {len(cells)} cells -> {d.trees}")
        return 0

    t_all = time.perf_counter()
    for phase, src_dir, cells in plan:
        root = args.out / phase
        d = armlayout.arm_dirs(root, args.backend, arm)
        entry = {"cells": len(cells)}

        # ---------------------------------------------------------------- stage: reroot
        if args.stage in ("all", "reroot"):
            d.trees.mkdir(parents=True, exist_ok=True)
            work = [(c, src_dir, d.trees, args.verify, args.force) for c in cells]
            t0 = time.perf_counter()
            rows, fails = [], []
            with Pool(args.jobs) as pool:
                for i, r in enumerate(pool.imap_unordered(_reroot_job, work, chunksize=1), 1):
                    (fails if "error" in r else rows).append(r)
                    if i % 100 == 0 or i == len(work):
                        print(f"[reroot]   {phase}: {i}/{len(work)}, {len(fails)} failed, "
                              f"{time.perf_counter() - t0:.0f}s", flush=True)
            for f in fails[:3]:
                print(f"  FAILED {f['key']}:\n{f['error']}")
            if fails:
                print(f"[reroot]   {phase}: G-IG0 RED -- {len(fails)} cells failed")
                return 2
            done = [r for r in rows if r.get("status") == "ok"]
            n_zero = sum(r["n_zero_length_loci"] for r in done)
            entry |= {
                "cached": sum(1 for r in rows if r.get("status") == "cached"),
                "rerooted": len(done),
                "loci": sum(r["n_loci"] for r in done),
                "n_zero_length_loci": n_zero,
                "frac_zero_length_edges": (
                    sum(r["frac_zero_length_edges"] for r in done) / len(done)) if done else 0.0,
                "min_branch_length": min((r["min_branch_length"] for r in done), default=0.0),
                "max_branch_length": max((r["max_branch_length"] for r in done), default=0.0),
                "verified_parity": bool(args.verify),
            }
            if n_zero:
                # The one thing that must never reach an instance build: every W13 tree feature
                # would go constant (or, for brank_*, become the Newick write order) and the
                # support cross-check would still pass.
                print(f"[reroot]   {phase}: RED -- {n_zero} loci carry no branch length at all")
                return 2
            print(f"[reroot]   {phase}: {len(done)} rerooted, {entry['cached']} cached, "
                  f"{entry['loci']:,} loci, zero-length edges "
                  f"{entry['frac_zero_length_edges']:.4f}, branch lengths "
                  f"[{entry['min_branch_length']:.3g}, {entry['max_branch_length']:.3g}]")

        # ---------------------------------------------------------------- stage: cells
        if args.stage in ("all", "cells"):
            d.cells.mkdir(parents=True, exist_ok=True)
            d.clusters.mkdir(parents=True, exist_ok=True)
            arm_obj = rungrid.Arm(name=arm, bp=args.bp, tree_dir=d.trees, cell_dir=d.cells,
                                  clu_dir=d.clusters, est_m=0, sidecar_dir=d.trees)
            net_dir = root / "networks"
            work = [(c, net_dir, d.cells, args.backend, False, d.trees, d.clusters, arm_obj)
                    for c in cells]
            t0 = time.perf_counter()
            n_ok = n_cached = 0
            errs = []
            with Pool(args.jobs) as pool:
                for i, (key, status, _) in enumerate(
                        pool.imap_unordered(rungrid.run_cell, work, chunksize=1), 1):
                    if status == "cached":
                        n_cached += 1
                    elif status == "ok":
                        n_ok += 1
                    else:
                        errs.append((key, status))
                    if i % 100 == 0 or i == len(work):
                        print(f"[cells]    {phase}: {i}/{len(work)}, {len(errs)} failed, "
                              f"{time.perf_counter() - t0:.0f}s", flush=True)
            for k, st in errs[:3]:
                print(f"  FAILED {k}: {st}")
            if errs:
                return 2
            entry |= {"cells_built": n_ok, "cells_cached": n_cached}
            print(f"[cells]    {phase}: {n_ok} built, {n_cached} cached -> {d.clusters}")

        report["phases"][phase] = entry

    # **Merged with what is already on disk, not overwritten.**  The b-grids and the x-grids need
    # different `--min-rep`, so the ingest is at least two invocations and a plain write would
    # leave a record of whichever ran last -- rungrid's collect stage makes the same point about
    # reporting every arm on disk rather than the ones this invocation touched.
    out = args.out / "validation" / f"alisim_ingest_bp{args.bp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            if prev.get("arm") == report["arm"]:
                report["phases"] = {**prev.get("phases", {}), **report["phases"]}
        except (OSError, ValueError):
            pass
    report["phases"] = dict(sorted(report["phases"].items()))
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n[unpack] {time.perf_counter() - t_all:.0f}s total -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
