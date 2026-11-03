#!/usr/bin/env python3
"""Alignments and estimated gene trees from this package's true gene trees.

Self-contained: stdlib plus the ``pylib/`` copy of ``opc.simulate`` that ships beside it, plus an
IQ-TREE 2 binary (AliSim ships inside it).  Nothing else, no network, no GPU.

What one cell costs
-------------------

    graft an outgroup  ->  AliSim once per locus  ->  IQ-TREE ``-S`` per batch  ->  root on the
    outgroup, prune it, check the taxon set

and it writes three files per cell:

* ``rooted/<phase>/<key>.nwk.gz``   -- **the deliverable**: one rooted, outgroup-pruned, ingroup-only
  estimated gene tree per line, in locus order, paired 1:1 with ``trees/<phase>/<key>.nwk.gz``;
* ``rooted/<phase>/<key>.raw.nwk.gz`` -- what IQ-TREE actually returned, unrooted, ``OUT`` still on;
* ``rooted/<phase>/<key>.json``     -- alignment widths, gap fraction, seeds, timings, the model
  IQ-TREE reported using, and the RF of the estimate against the true tree it came from.

Three things here are not conveniences and removing them silently corrupts the output; all three
are measured, and ``../README.md`` gives the numbers.

* **AliSim runs once per locus.**  ``-t multi.nwk`` simulates from the *first* tree only, with no
  error, and the batched ``-Q`` mode both ignores ``--branch-scale`` and aborts on indels + ``+G4``.
* **The substitution model goes in the partition file**, where it overrides ``-m``.  Writing the
  conventional ``DNA`` there makes IQ-TREE infer under GTR whatever ``-m`` says, exit code 0.
* **``--keep-ident``**, or IQ-TREE drops identical sequences and returns a tree short a taxon.
  ``root_and_prune_file`` is what catches that, by checking the emitted taxon set.

Resumable at cell granularity: a cell whose output is newer than its input is skipped, so an
interrupted run costs only the cells still missing.  Failures are written to
``rooted/<phase>/<key>.failed.txt`` with the tool's own stderr tail, and never stop the pool.

Usage
-----

    python3 run_estimate.py --iqtree /path/to/iqtree2 --jobs 32 --dry-run
    python3 run_estimate.py --iqtree /path/to/iqtree2 --jobs 32
    python3 run_estimate.py --iqtree /path/to/iqtree2 --jobs 32 --phases b250 --bp 1000

``--stage graft`` stops after the outgroup graft and the coalescent-unit rescale, writing
AliSim-ready Newicks to ``alisim_ready/<phase>/<key>.nwk.gz`` -- one tree per line, branch lengths
already in substitutions per site, ready to be fed to AliSim one line at a time.  Use it if you
would rather drive the sequence simulator yourself.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The vendored modules are byte-for-byte copies and they use ``dict |= dict`` (PEP 584), so 3.9
# is a hard floor.  Checked here rather than left to fail: on 3.8 everything up to and including
# AliSim and IQ-TREE succeeds, and the TypeError lands in ``root_and_prune_file`` *after* the
# whole cell has been simulated and inferred -- i.e. it looks like a rooting bug and costs the run.
if sys.version_info < (3, 9):
    raise SystemExit(
        f"needs Python 3.9 or newer (running {sys.version.split()[0]}).  The failure without this "
        f"check comes only after a cell's alignments and trees are computed, so it is not worth "
        f"discovering the slow way.")


# ---------------------------------------------------------------------------------------------
# seeds
#
# Verbatim from ``opc.grid.chain_seed``.  The seeds are derived here rather than shipped in the
# manifest because they depend on ``--bp``, which is a free parameter of this script -- but a
# re-derivation that drifts is worse than no derivation at all, so ``manifest.json`` carries a
# handful of (phase, key, bp) -> (alisim, iqtree) triples computed on the source machine and
# ``_check_seeds`` refuses to start if this file disagrees with them.
# ---------------------------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def chain_seed(*parts) -> int:
    payload = ":".join(str(p) for p in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MASK64


def cell_seeds(master_seed: int, key: str, bp: int) -> tuple[int, int]:
    return (chain_seed(master_seed, "alisim", key, bp) % (2 ** 31),
            chain_seed(master_seed, "iqtree", key, bp) % (2 ** 31))


def _check_seeds(manifest: dict) -> None:
    for row in manifest.get("seed_check", []):
        got = cell_seeds(row["master_seed"], row["cell_key"], row["bp"])
        want = (row["seed_alisim"], row["seed_iqtree"])
        if got != want:
            raise SystemExit(
                f"seed check FAILED for {row['cell_key']} at bp={row['bp']}: this script derives "
                f"{got}, the package was built expecting {want}.  Do not run: the arm would not "
                f"reproduce.")


# ---------------------------------------------------------------------------------------------


def load_manifest(pkg: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((pkg / "manifest.json").read_text())
    rows = []
    with open(pkg / "MANIFEST.tsv") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    for r in rows:
        r["n_taxa"] = int(r["n_taxa"])
        r["k"] = int(r["k"])
        r["rep"] = int(r["rep"])
        r["cu"] = float(r["cu"])
        r["n_loci"] = int(r["n_loci"])
        r["master_seed"] = int(r["master_seed"])
        r["requested"] = r["requested"] == "1"
    return meta, rows


def select(rows, args) -> list[dict]:
    keep = rows
    if not args.all:
        keep = [r for r in keep if r["requested"]]
    if args.phases:
        want = set(args.phases.split(","))
        keep = [r for r in keep if r["phase"] in want]
    if args.splits:
        want = set(args.splits.split(","))
        keep = [r for r in keep if r["split"] in want]
    if args.cells:
        want = {ln.strip() for ln in Path(args.cells).read_text().split() if ln.strip()}
        keep = [r for r in keep if r["cell_key"] in want]
    if args.max_k is not None:
        keep = [r for r in keep if r["k"] <= args.max_k]
    keep.sort(key=lambda r: (r["n_taxa"], r["phase"], r["rep"]))
    if args.limit:
        keep = keep[:args.limit]
    return keep


def build_model(meta, seqsim):
    """Rebuild the :class:`seqsim.SeqModel` from ``manifest.json``, then check the rebuild.

    ``SeqModel.as_dict`` is a *report*, not a constructor kwargs dict: it flattens ``base_freqs``
    to a comma-joined string and adds three derived keys.  Feeding it straight back gives a model
    whose frequencies are a ``str``, which raises inside the first f-string that formats one -- so
    the parse is explicit, and the result is checked against the ``sim_model`` string the package
    shipped with.  That round trip is the only thing that would catch this drifting again.
    """
    d = dict(meta["model"])
    fields = seqsim.SeqModel.__dataclass_fields__
    kw = {k: v for k, v in d.items() if k in fields}
    kw["base_freqs"] = tuple(float(x) for x in str(d["base_freqs"]).split(","))
    model = seqsim.SeqModel(**kw)
    if model.sim_model_string() != d["sim_model"]:
        raise SystemExit(f"model rebuild disagrees with the manifest: this script derives "
                         f"{model.sim_model_string()!r}, the package shipped "
                         f"{d['sim_model']!r}.  Do not run.")
    return model


def make_jobs(rows, args, meta, seqsim):
    """One :class:`seqsim.EstJob` per cell -- ``n_chunks=1``, so no merge step afterwards."""
    model = build_model(meta, seqsim)
    jobs = []
    for r in rows:
        m = min(args.loci, r["n_loci"]) if args.loci else r["n_loci"]
        seed_a, seed_i = cell_seeds(r["master_seed"], r["cell_key"], args.bp)
        jobs.append(seqsim.EstJob(
            cell_key=r["cell_key"], n_taxa=r["n_taxa"], cu=r["cu"], bp=args.bp,
            chunk=0, start=0, count=m, cell_m=m,
            per_batch=seqsim.batch_loci(r["n_taxa"] + 1, args.bp, args.iq_target_ns),
            true_gz=args.pkg / r["genetrees"],
            out_dir=args.out / r["phase"],
            seed_alisim=seed_a, seed_iqtree=seed_i, model=model,
            scratch=args.scratch, iq_target_ns=args.iq_target_ns, n_chunks=1,
        ))
    return jobs


def graft_only(rows, args, meta, outgroup) -> None:
    """Materialise the AliSim-ready trees and stop.  Deterministic: no RNG anywhere in it."""
    scale = meta["model"]["branch_scale"]
    alpha = meta["model"]["outgroup_alpha"]
    for i, r in enumerate(rows, 1):
        dst = args.out / r["phase"] / f"{r['cell_key']}.nwk.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.force:
            continue
        with tempfile.TemporaryDirectory(dir=str(args.scratch), prefix="opc-graft-") as tmp:
            plain = Path(tmp) / "grafted.nwk"
            stats = outgroup.graft_file(args.pkg / r["genetrees"], plain,
                                        alpha=alpha, branch_scale=scale,
                                        cell_m=r["n_loci"])
            with open(plain, "rb") as fh, gzip.open(str(dst) + ".tmp", "wb") as out:
                shutil.copyfileobj(fh, out)
        os.replace(str(dst) + ".tmp", dst)
        # Built as a string, never with_suffix(): a cell key contains dots, so Path.suffix
        # is ".25_gU0" and with_suffix would rewrite the key rather than the extension.
        (dst.parent / f"{r['cell_key']}.json").write_text(
            json.dumps({"cell_key": r["cell_key"], "phase": r["phase"],
                        "branch_scale": scale, **stats}, indent=1) + "\n")
        if i % 100 == 0 or i == len(rows):
            print(f"[graft] {i}/{len(rows)}", flush=True)
    print(f"[graft] wrote {len(rows)} AliSim-ready cells to {args.out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iqtree", type=Path, default=os.environ.get("OPC_IQTREE"),
                    help="path to the iqtree2 binary (AliSim ships inside it)")
    ap.add_argument("--pkg", type=Path, default=HERE, help="package root (default: this directory)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: <pkg>/estimated_bp<BP>, or <pkg>/alisim_ready "
                         "for --stage graft)")
    ap.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()),
                    help="per-job working directory; needs ~%%(loci x taxa x bp) bytes per job")
    ap.add_argument("--bp", type=int, default=500, help="ROOT sequence length, not alignment width")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--loci", type=int, default=0, help="cap loci per cell (0 = all)")
    ap.add_argument("--stage", choices=("run", "graft"), default="run")
    ap.add_argument("--phases", default="", help="comma list, e.g. b25,b50")
    ap.add_argument("--splits", default="", help="comma list of train,val,test")
    ap.add_argument("--cells", type=Path, help="file of cell keys, one per line")
    ap.add_argument("--max-k", type=int, default=None, help="keep cells with at most k reticulations")
    ap.add_argument("--all", action="store_true",
                    help="ignore the manifest's `requested` flag and take every cell in the package")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iq-target-ns", type=float, default=4e6,
                    help="IQ-TREE batch footprint knob; lower it if concurrency is thrashing")
    ap.add_argument("--force", action="store_true", help="--stage graft: overwrite existing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.jobs >= 100:
        print("refusing --jobs >= 100: a multiprocessing pool that large deadlocks silently "
              "under a shared RLIMIT_NPROC -- see README.")
        return 2

    meta, rows = load_manifest(args.pkg)
    _check_seeds(meta)
    rows = select(rows, args)
    if not rows:
        print("no cells selected")
        return 1

    if args.out is None:
        args.out = args.pkg / ("alisim_ready" if args.stage == "graft"
                               else f"estimated_bp{args.bp}")
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.pkg / "pylib"))
    if args.stage == "graft":
        from opc.simulate import outgroup
        graft_only(rows, args, meta, outgroup)
        return 0

    if not args.iqtree:
        print("--iqtree is required (or set OPC_IQTREE)")
        return 2
    os.environ["OPC_IQTREE"] = str(args.iqtree)
    from opc.simulate import seqsim
    if not seqsim.available():
        print(f"no IQ-TREE at {args.iqtree}")
        return 2

    jobs = make_jobs(rows, args, meta, seqsim)
    total_loci = sum(j.count for j in jobs)
    serial = sum(j.predicted_seconds() for j in jobs)
    print(f"IQ-TREE: {seqsim.iqtree_version()}")
    print(f"model:   sim {build_model(meta, seqsim).sim_model_string()}"
          f"  indel {meta['model']['indel_ins']},{meta['model']['indel_del']}"
          f"  infer {meta['model']['inference_model']}")
    print(f"cells:   {len(jobs)}  loci: {total_loci:,}  bp: {args.bp}  jobs: {args.jobs}")
    print(f"budget:  {serial / 3600:.1f} CPU-h nominal, "
          f"{serial * seqsim.CONCURRENCY_PENALTY / 3600:.1f} CPU-h with the measured concurrency "
          f"penalty ({seqsim.CONCURRENCY_PENALTY}x)")
    print(f"         ~{serial * seqsim.CONCURRENCY_PENALTY / 3600 / args.jobs:.1f} h wall at "
          f"--jobs {args.jobs}")
    by_phase: dict[str, list] = {}
    for j, r in zip(jobs, rows):
        by_phase.setdefault(r["phase"], []).append(j)
    for ph, js in sorted(by_phase.items()):
        s = sum(j.predicted_seconds() for j in js)
        print(f"           {ph:>5}: {len(js):>4} cells  {s * seqsim.CONCURRENCY_PENALTY / 3600:8.1f} CPU-h")
    print(f"out:     {args.out}")
    print(f"scratch: {args.scratch}")
    if args.dry_run:
        return 0

    from multiprocessing import Pool
    t0 = time.perf_counter()
    n_ok = n_cached = n_fail = 0
    fails: list[str] = []
    log = args.out / f"run_bp{args.bp}.tsv"
    new = not log.exists()
    with open(log, "a") as lf:
        if new:
            lf.write("cell_key\tstatus\tseconds\n")
        with Pool(processes=args.jobs) as pool:
            for i, (stem, status, secs) in enumerate(
                    pool.imap_unordered(seqsim.run_est_job, jobs, chunksize=1), 1):
                short = status.split("\n")[0]
                if status == "ok":
                    n_ok += 1
                elif status == "cached":
                    n_cached += 1
                else:
                    n_fail += 1
                    if len(fails) < 5:
                        fails.append(f"--- {stem}\n{status}")
                lf.write(f"{stem}\t{short}\t{secs:.2f}\n")
                lf.flush()
                if i % 10 == 0 or i == len(jobs):
                    el = time.perf_counter() - t0
                    rate = i / el
                    print(f"[{i}/{len(jobs)}] ok={n_ok} cached={n_cached} failed={n_fail} "
                          f"{el / 60:.1f} min elapsed, ~{(len(jobs) - i) / rate / 60:.1f} min left",
                          flush=True)
    print(f"\ndone: {n_ok} computed, {n_cached} cached, {n_fail} FAILED in "
          f"{(time.perf_counter() - t0) / 3600:.2f} h")
    for f in fails:
        print(f)
    if n_fail:
        print(f"per-cell tracebacks are in {args.out}/<phase>/<key>.failed.txt")
    print(f"\nsend back: {args.out}  (the .nwk.gz and .json files; the .raw.nwk.gz too if you "
          f"want the unrooted originals kept)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
