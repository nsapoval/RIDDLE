"""Sequences from gene trees, and gene trees back from those sequences.

The estimation-error arm: take a cell's cached true gene trees, graft an outgroup, simulate an
alignment per locus with AliSim under HKY+G with indels, re-estimate each locus tree with
IQ-TREE, root it on the outgroup and prune it.  Feeding the result back through
:func:`opc.clusters.extract.count_newick_file` gives the same percolation metrics computed on
*estimated* rather than true gene trees.

Why AliSim runs once per locus
------------------------------

AliSim can batch, but not in a way this study can use -- both defects measured in
``scripts/probe_alisim.py`` and recorded in ``results/validation/alisim_probe.json``:

* plain ``-t multi.nwk`` **silently simulates from the first tree only**, so the obvious batched
  form would yield one locus where m were asked for, with no error;
* ``--alisim -Q charsets.nex`` does simulate one locus per tree line and in order, but it
  **ignores ``--branch-scale``** (identical TreeLen with and without it) and **aborts** inside
  ``AliSimulator::updateNewGenomeIndels`` whenever indels are combined with ``+G4``.

Since the study needs indels *and* rate heterogeneity, AliSim is called once per locus, where
both work.  It costs 0.013-0.038 s per locus -- under 1% of IQ-TREE's cost -- and the
coalescent-unit rescale moves into Python (:func:`opc.simulate.outgroup.graft_file`), which is
where it is testable in isolation anyway.

IQ-TREE, by contrast, *is* batched: the per-locus alignments are concatenated into one
superalignment and inferred with ``-S``, one process per batch rather than per locus.  Batch size
is chosen to bound the likelihood-vector footprint.

Everything intermediate lives in a per-job scratch directory (tmpfs by default), so the ~10 aux
files IQ-TREE writes per invocation never touch NFS.  Only three files per (cell, bp) persist.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from opc.simulate import outgroup, phylip

__all__ = [
    "IQTREE",
    "SeqModel",
    "EstJob",
    "available",
    "iqtree_version",
    "alisim_cmd",
    "iqtree_cmd",
    "batch_loci",
    "seconds_per_locus",
    "plan_jobs",
    "run_est_job",
    "merge_chunks",
    "arm_dirs",
]

IQTREE = Path(os.environ.get("OPC_IQTREE", "/home/ns58/miniconda3/envs/fusang-env/bin/iqtree2"))

# Measured by probe P3 (--fast, -T 1, HKY+F+G4, 500 bp, ingroup + outgroup), recorded in
# ``results/validation/alisim_probe.json``.  Interpolated log-linearly in n.
#
# These are *single-process* numbers, measured under the study's own inference model.  (They were
# briefly measured under GTR, because a partition file's model field overrides -m -- see
# ``phylip.write_partition_raxml``; GTR ran ~25% slower, so the old constants were conservative.)
# ``CONCURRENCY_PENALTY`` is calibrated against the one real full-scale run: the 500 bp full grid
# spent 1,651 CPU-h where this model predicts 612 -- the workload is memory-bandwidth bound, and
# 180 concurrent processes each streaming ~1.4 GB of likelihood vectors do not each get the whole
# machine's bandwidth.  It is applied to the budget ``--dry-run`` prints so they do not read 2-3x
# optimistic, and deliberately *not* to ``seconds_per_locus``, which sizes chunks *relatively*,
# where a constant factor cancels.
_SEC_PER_LOCUS_500 = {10: 0.016, 25: 0.062, 50: 0.229, 100: 0.646, 200: 2.182}
CONCURRENCY_PENALTY = 2.7
# Cost grows *sublinearly* in sequence length -- measured 0.407 / 0.646 / 1.060 s at
# 200 / 500 / 1000 bp for n=100, i.e. (bp/500)**0.6, not (bp/500).
_BP_EXPONENT = 0.6
_ALISIM_SEC_PER_LOCUS = 0.025  # 0.013 (n=10) to 0.038 (n=200); under 1% of the IQ-TREE cost

# Two rare indel-driven failures, both from the same cause.  AliSim can delete every site of a
# locus and give up:
#   "Could not select a valid position (not a deleted-site) for insertion/deletion events."
# Indel load compounds over *total tree length*, so the deep cells (n=200, cu=2.0 is ~25 subs/site)
# combined with a short root sequence can lose the whole alignment.  Measured rate is ~1 locus in
# 35,000 -- but a chunk rolls that dice ~800 times, so ~2% of *chunks* died at 200 bp before this
# retry existed, each throwing away the hundreds of loci it had already simulated.
#
# The same load has a second, quieter effect: a single *taxon* can lose every site in a locus.
# IQ-TREE then drops that row -- ``--keep-ident`` protects identical sequences, not empty ones --
# and returns a tree one taxon short, which ``root_and_prune_file`` rejects.  Measured at ~1 locus
# in 792 in the gappiest cell (n=200, cu=2.0, 200 bp, 81% gaps), rarer elsewhere.
#
# Re-drawing the indel process with a fresh seed is the honest response to both rather than a
# fudge: neither a fully deleted locus nor a taxon with zero observed sites is data you could have
# in hand, so conditioning on survival is what ascertainment already does.  At these rates the
# conditioning is negligible, the retry is deterministic, and ``n_alisim_retries`` records it.
# Only these two conditions are retried; any other AliSim failure still propagates.
_ALISIM_ATTEMPTS = 4
_RETRY_SEED_STRIDE = 7_919_657
_DELETED_SITE_ERROR = "not a deleted-site"


def available() -> bool:
    return IQTREE.exists()


def iqtree_version() -> str:
    try:
        res = subprocess.run([str(IQTREE), "--version"], capture_output=True, text=True,
                             timeout=60)
        return res.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - a missing/broken binary is reported, not raised
        return "unknown"


@dataclass(frozen=True)
class SeqModel:
    """Everything that defines the sequence-evolution and inference side of the arm.

    ``branch_scale`` converts coalescent units to substitutions/site: a pair of lineages
    coalescing at 1 CU are separated by two branches of ``branch_scale``, so the expected pairwise
    divergence at 1 CU is ``theta = 2 * branch_scale``.
    """

    branch_scale: float = 0.018
    kappa: float = 2.0
    base_freqs: tuple[float, ...] = (0.3, 0.2, 0.2, 0.3)
    gamma_shape: float = 1.0          # 0 disables rate heterogeneity on the simulation side
    gamma_cats: int = 4
    indel_ins: float = 0.03
    indel_del: float = 0.03
    indel_size: str = "POW{1.7/100}"
    outgroup_alpha: float = 0.5
    inference_model: str = "HKY+F+G4"
    fast: bool = True
    keep_ident: bool = True

    @property
    def theta(self) -> float:
        return 2.0 * self.branch_scale

    @property
    def indel(self) -> bool:
        return self.indel_ins > 0 or self.indel_del > 0

    def sim_model_string(self) -> str:
        freqs = ",".join(f"{f:g}" for f in self.base_freqs)
        s = f"HKY{{{self.kappa:g}}}+F{{{freqs}}}"
        if self.gamma_shape > 0:
            s += f"+G{self.gamma_cats}{{{self.gamma_shape:g}}}"
        return s

    def expected_subs_per_internal_branch(self, cu: float, bp: int) -> float:
        """The one number that says how much signal a locus carries.  Printed by ``--dry-run``."""
        return cu * self.branch_scale * bp

    def as_dict(self) -> dict:
        d = asdict(self)
        d["base_freqs"] = ",".join(f"{f:g}" for f in self.base_freqs)
        d["theta"] = self.theta
        d["sim_model"] = self.sim_model_string()
        return d


def arm_dirs(root: Path, backend: str, bp: int) -> tuple[Path, Path]:
    """``(tree_dir, cell_dir)`` for one estimated arm.

    Keyed by backend *and* bp for the reason ``CLAUDE.md`` gives for ``genetrees_<backend>/``: an
    un-keyed cache silently makes two arms read each other's trees.
    """
    return (root / f"genetrees_est_{backend}_bp{bp}",
            root / f"cells_{backend}_est_bp{bp}")


def seconds_per_locus(n_taxa: int, bp: int) -> float:
    """Cost model: log-linear in ``n`` between the measured anchors, ``bp**0.6`` in length."""
    import math

    ks = sorted(_SEC_PER_LOCUS_500)
    if n_taxa <= ks[0]:
        base = _SEC_PER_LOCUS_500[ks[0]]
    elif n_taxa >= ks[-1]:
        base = _SEC_PER_LOCUS_500[ks[-1]]
    else:
        hi = next(k for k in ks if k >= n_taxa)
        lo = max(k for k in ks if k <= n_taxa)
        if lo == hi:
            base = _SEC_PER_LOCUS_500[lo]
        else:
            w = (math.log(n_taxa) - math.log(lo)) / (math.log(hi) - math.log(lo))
            base = math.exp((1 - w) * math.log(_SEC_PER_LOCUS_500[lo])
                            + w * math.log(_SEC_PER_LOCUS_500[hi]))
    return base * (bp / 500.0) ** _BP_EXPONENT + _ALISIM_SEC_PER_LOCUS


def batch_loci(n_taxa: int, bp: int, target_ns: float = 4e6) -> int:
    """How many loci one IQ-TREE ``-S`` process handles.

    Bounds the likelihood-vector footprint while keeping process startup well under 1% of the
    batch's work at every ``(n, bp)``.  With ``+G4`` a site costs 4 states x 4 rate categories x
    8 bytes = **128 B per taxon**, and IQ-TREE allocates every partition in the batch at once:
    n=201 with 39 loci of ~1234 bp measures 1.4 GB resident.  ``target_ns`` is therefore a
    footprint knob as much as a batching one -- lower it (``--iq-target-ns``) if 180-way
    concurrency is thrashing.
    """
    return max(10, min(2000, int(target_ns / max(1, n_taxa * bp))))


def alisim_cmd(tree_file: Path, prefix: Path, bp: int, model: SeqModel, seed: int) -> list[str]:
    cmd = [str(IQTREE), "--alisim", str(prefix), "-t", str(tree_file), "--length", str(bp),
           "-m", model.sim_model_string()]
    if model.indel:
        cmd += ["--indel", f"{model.indel_ins:g},{model.indel_del:g}",
                "--indel-size", f"{model.indel_size},{model.indel_size}", "--no-unaligned"]
    cmd += ["-af", "phy", "--seed", str(seed), "-T", "1", "--quiet"]
    return cmd


def iqtree_cmd(aln: Path, parts: Path, prefix: Path, model: SeqModel, seed: int,
               threads: int = 1) -> list[str]:
    cmd = [str(IQTREE), "-s", str(aln), "-S", str(parts), "-m", model.inference_model,
           "-T", str(threads), "--prefix", str(prefix), "--seed", str(seed), "--redo", "--quiet"]
    if model.keep_ident:
        cmd.append("--keep-ident")
    if model.fast:
        cmd.append("--fast")
    return cmd


# "   1  HKY+F+G4       0.5742  HKY{1.4}+F{...}+G4{...}" in the .iqtree report.
# The substitution-model table is space-aligned; the partition-info table above it is
# tab-separated and would otherwise match with its "DNA" Type column.
_MODEL_ROW = re.compile(r"^ *\d+ {2,}([A-Za-z][A-Za-z0-9+*]*) {2,}[0-9.]+ ", re.M)


def _models_used(report: Path, want: str) -> set[str]:
    """The model IQ-TREE actually applied, checked against the one asked for.

    IQ-TREE accepts an unknown ``-m`` string and runs anyway with exit code 0, and a partition
    file's model field silently overrides ``-m`` -- so "did we infer under the model we think"
    is not something the exit code answers.
    """
    if not report.exists():
        return set()
    used = set(_MODEL_ROW.findall(report.read_text()))
    want_base = want.split("+")[0].upper()
    wrong = {u for u in used if u.split("+")[0].upper() != want_base}
    if wrong:
        raise RuntimeError(f"asked IQ-TREE for {want!r} but it used {sorted(wrong)}")
    return used


def _env() -> dict:
    """One thread per process.

    ``-T AUTO`` would both oversubscribe the pool and make the result non-deterministic.
    """
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def _run(cmd, cwd, timeout=None):
    res = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=_env(),
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-12:])
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return res


@dataclass(frozen=True)
class EstJob:
    """One contiguous slice of one cell's loci at one sequence length.

    Chunking is an execution detail, not part of the result: seeds key off the *global* locus and
    batch index rather than the position within a chunk, chunk boundaries are multiples of the
    IQ-TREE batch size, and ``T_out`` is taken over the whole cell.  So re-running a cell with a
    different ``--est-target-seconds`` reproduces it byte for byte.
    """

    cell_key: str
    n_taxa: int
    cu: float
    bp: int
    chunk: int
    start: int
    count: int
    cell_m: int
    per_batch: int
    true_gz: Path
    out_dir: Path
    seed_alisim: int          # per-cell base; the locus seed adds the global locus index
    seed_iqtree: int          # per-cell base; the batch seed adds the global batch index
    model: SeqModel
    scratch: Path
    iq_target_ns: float = 4e6
    n_chunks: int = 1

    @property
    def stem(self) -> str:
        return self.cell_key if self.n_chunks == 1 else f"{self.cell_key}.c{self.chunk:03d}"

    @property
    def rooted(self) -> Path:
        return self.out_dir / f"{self.stem}.nwk.gz"

    @property
    def raw(self) -> Path:
        return self.out_dir / f"{self.stem}.raw.nwk.gz"

    @property
    def sidecar(self) -> Path:
        return self.out_dir / f"{self.stem}.json"

    @property
    def failure_log(self) -> Path:
        return self.out_dir / f"{self.stem}.failed.txt"

    def predicted_seconds(self) -> float:
        return self.count * seconds_per_locus(self.n_taxa, self.bp)


def run_est_job(job: EstJob) -> tuple[str, str, float]:
    """graft -> AliSim per locus -> IQ-TREE ``-S`` per batch -> reroot/prune -> sidecar.

    Returns the ``(key, status, seconds)`` triple the other stages use, with the traceback folded
    into the status so a pool worker never loses it.
    """
    t0 = time.perf_counter()
    try:
        # Either this chunk's own output, or the merged whole-cell file it was folded into --
        # merge_chunks deletes the chunk parts, so without the second check a completed cell
        # would be recomputed from scratch on every resume.
        merged = job.out_dir / f"{job.cell_key}.nwk.gz"
        merged_side = job.out_dir / f"{job.cell_key}.json"
        for done, side in ((job.rooted, job.sidecar), (merged, merged_side)):
            if done.exists() and side.exists() and \
                    done.stat().st_mtime > job.true_gz.stat().st_mtime:
                return job.stem, "cached", 0.0
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.scratch.mkdir(parents=True, exist_ok=True)
        taxa = [f"t{i}" for i in range(1, job.n_taxa + 1)]

        with tempfile.TemporaryDirectory(dir=str(job.scratch), prefix="opc-est-") as tmp:
            work = Path(tmp)
            graft = outgroup.graft_file(job.true_gz, work / "grafted.nwk",
                                        alpha=job.model.outgroup_alpha,
                                        branch_scale=job.model.branch_scale,
                                        start=job.start, count=job.count, cell_m=job.cell_m)
            trees = (work / "grafted.nwk").read_text().split()
            if len(trees) != job.count:
                raise RuntimeError(f"grafted {len(trees)} trees, expected {job.count}")

            t_sim = time.perf_counter()
            one = work / "one.nwk"
            n_retries = 0
            for i, tree in enumerate(trees):
                one.write_text(tree + "\n")
                base = (job.seed_alisim + job.start + i) % (2 ** 31)
                last = _ALISIM_ATTEMPTS - 1
                phy = work / f"l{i:06d}.phy"
                for attempt in range(_ALISIM_ATTEMPTS):
                    seed = (base + attempt * _RETRY_SEED_STRIDE) % (2 ** 31)
                    try:
                        _run(alisim_cmd(one, work / f"l{i:06d}", job.bp, job.model, seed), work)
                    except RuntimeError as exc:
                        if _DELETED_SITE_ERROR not in str(exc) or attempt == last:
                            raise
                        n_retries += 1
                        continue
                    if not phylip.has_empty_sequence(phy):
                        break
                    if attempt == last:
                        raise RuntimeError(
                            f"locus {job.start + i}: a taxon has no observed site after "
                            f"{_ALISIM_ATTEMPTS} attempts")
                    n_retries += 1
            sim_seconds = time.perf_counter() - t_sim

            per_batch = job.per_batch
            t_inf = time.perf_counter()
            widths_all, gaps, raw_lines = [], [], []
            models_used: set[str] = set()
            for b, lo in enumerate(range(0, job.count, per_batch)):
                hi = min(lo + per_batch, job.count)
                srcs = [work / f"l{i:06d}.phy" for i in range(lo, hi)]
                aln = work / f"b{b:04d}.phy"
                parts = work / f"b{b:04d}.parts"
                widths = phylip.write_superalignment(srcs, aln)
                # The model goes in the partition file, where it overrides -m; see
                # write_partition_raxml.  Passing it in both places keeps them from drifting.
                phylip.write_partition_raxml(parts, widths, model=job.model.inference_model)
                widths_all += widths
                gaps.append(phylip.gap_counts(aln))
                # Global batch index: chunk starts are multiples of per_batch, so a batch has the
                # same composition and the same seed however the cell was chunked.
                _run(iqtree_cmd(aln, parts, work / f"b{b:04d}", job.model,
                                (job.seed_iqtree + (job.start + lo) // per_batch) % (2 ** 31)),
                     work)
                models_used |= _models_used(work / f"b{b:04d}.iqtree",
                                            job.model.inference_model)
                lines = [ln.strip() for ln in
                         (work / f"b{b:04d}.treefile").read_text().splitlines() if ln.strip()]
                if len(lines) != hi - lo:
                    raise RuntimeError(
                        f"batch {b}: IQ-TREE returned {len(lines)} trees for {hi - lo} partitions")
                raw_lines += lines
                for p in (aln, parts):
                    p.unlink(missing_ok=True)
                for p in work.glob(f"b{b:04d}.*"):
                    p.unlink(missing_ok=True)
            infer_seconds = time.perf_counter() - t_inf

            if len(raw_lines) != job.count:
                raise RuntimeError(f"{len(raw_lines)} estimated trees, expected {job.count}")
            (work / "raw.nwk").write_text("\n".join(raw_lines) + "\n")

            stats = outgroup.root_and_prune_file(work / "raw.nwk", work / "rooted.nwk.gz", taxa,
                                                 true_gz=job.true_gz, start=job.start)
            with open(work / "raw.nwk", "rb") as fh, gzip.open(work / "raw.nwk.gz", "wb") as out:
                shutil.copyfileobj(fh, out)

            n_gap = sum(g for g, _ in gaps)
            n_char = sum(t for _, t in gaps)
            sidecar = {
                "cell_key": job.cell_key, "chunk": job.chunk, "n_chunks": job.n_chunks,
                "start": job.start, "count": job.count, "bp": job.bp, "n_taxa": job.n_taxa,
                "cu": job.cu,
                "aln_len_total": sum(widths_all),
                "aln_len_mean": sum(widths_all) / len(widths_all),
                "aln_len_min": min(widths_all), "aln_len_max": max(widths_all),
                "n_gap_chars": n_gap, "n_aln_chars": n_char,
                "gap_fraction": n_gap / n_char if n_char else 0.0,
                "alisim_seconds": sim_seconds, "iqtree_seconds": infer_seconds,
                "loci_per_iqtree_batch": per_batch,
                "n_alisim_retries": n_retries,
                "iqtree_model_used": ",".join(sorted(models_used)),
                "seed_alisim": job.seed_alisim, "seed_iqtree": job.seed_iqtree,
                "iqtree_version": iqtree_version(),
                **{f"graft_{k}": v for k, v in graft.items()},
                **stats,
                **job.model.as_dict(),
            }
            # Copy to a sibling .tmp then rename, so a reader never sees a partial file even
            # though the move crosses tmpfs -> NFS.
            for src, dst in ((work / "rooted.nwk.gz", job.rooted),
                             (work / "raw.nwk.gz", job.raw)):
                tmp_dst = dst.with_name(dst.name + ".tmp")
                shutil.copyfile(src, tmp_dst)
                tmp_dst.rename(dst)
            tmp_side = job.sidecar.with_name(job.sidecar.name + ".tmp")
            tmp_side.write_text(json.dumps(sidecar, indent=1) + "\n")
            tmp_side.rename(job.sidecar)
        job.failure_log.unlink(missing_ok=True)   # a previous failure that has now succeeded
        return job.stem, "ok", time.perf_counter() - t0
    except Exception:  # noqa: BLE001 - reported through the status triple, as the other stages do
        tb = traceback.format_exc()
        # Persist it.  The driver only prints the first few failures once every task has
        # finished, and the scratch directory is gone by then -- so without this a failure 40 min
        # into a 19 h run is undiagnosable until the run ends.  ``_run`` folds the last lines of
        # AliSim/IQ-TREE stderr into the exception, so this file is the whole story.
        try:
            job.out_dir.mkdir(parents=True, exist_ok=True)
            job.failure_log.write_text(
                f"{job.cell_key} bp={job.bp} chunk={job.chunk}/{job.n_chunks} "
                f"start={job.start} count={job.count} per_batch={job.per_batch}\n\n{tb}")
        except OSError:
            pass
        return job.stem, "FAILED\n" + tb, time.perf_counter() - t0


def plan_jobs(cells, bp: int, model: SeqModel, root: Path, backend: str, est_m: int,
              scratch: Path, master_seed: int, target_seconds: float = 900.0,
              iq_target_ns: float = 4e6) -> list[EstJob]:
    """One job per contiguous chunk of each cell's loci, sized to ``target_seconds``."""
    from opc import grid

    tree_dir = root / f"genetrees_{backend}"
    out_dir, _ = arm_dirs(root, backend, bp)
    jobs: list[EstJob] = []
    for c in cells:
        m = c.m_max if not est_m else min(est_m, c.m_max)
        per_batch = batch_loci(c.n + 1, bp, iq_target_ns)
        # Chunks are whole numbers of IQ-TREE batches, so batch composition -- and therefore the
        # estimated trees -- does not depend on how the cell was chunked.
        want = max(1, int(target_seconds / max(1e-6, seconds_per_locus(c.n, bp))))
        size = max(per_batch, -(-want // per_batch) * per_batch)
        n_chunks = max(1, -(-m // size))
        # Seeds are per-cell bases; run_est_job adds the global locus / batch index.
        base_a = grid.chain_seed(master_seed, "alisim", c.key, bp) % (2 ** 31)
        base_i = grid.chain_seed(master_seed, "iqtree", c.key, bp) % (2 ** 31)
        for k in range(n_chunks):
            start = k * size
            count = min(size, m - start)
            if count <= 0:
                continue
            jobs.append(EstJob(
                cell_key=c.key, n_taxa=c.n, cu=c.cu, bp=bp, chunk=k, start=start, count=count,
                cell_m=m, per_batch=per_batch,
                true_gz=tree_dir / f"{c.key}.nwk.gz", out_dir=out_dir,
                seed_alisim=base_a, seed_iqtree=base_i,
                model=model, scratch=scratch, iq_target_ns=iq_target_ns, n_chunks=n_chunks,
            ))
    return jobs


def _cat_gz(parts, dst: Path) -> None:
    """gzip is a multi-member format (RFC 1952), so concatenating members is a byte copy."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out)
    tmp.rename(dst)


def _count_lines(gz: Path) -> int:
    with gzip.open(gz, "rt") as fh:
        return sum(1 for ln in fh if ln.strip())


def merge_chunks(cell_key: str, bp: int, root: Path, backend: str) -> dict | None:
    """Concatenate a cell's chunks in order and pool their sidecars.  Idempotent."""
    out_dir, _ = arm_dirs(root, backend, bp)
    final = out_dir / f"{cell_key}.nwk.gz"
    sides = sorted(out_dir.glob(f"{cell_key}.c[0-9][0-9][0-9].json"))
    if final.exists() and not sides:
        return json.loads((out_dir / f"{cell_key}.json").read_text())
    if not sides:
        return None

    metas = [json.loads(p.read_text()) for p in sides]
    if len(metas) != metas[0]["n_chunks"]:
        return None  # still incomplete
    metas.sort(key=lambda d: d["chunk"])
    # Built as strings, never with_suffix(): a cell key contains dots ("...cu0.5_g0.5"), so
    # Path.suffix is ".c000" and with_suffix would rewrite a chunk path into the final path.
    stems = [f"{cell_key}.c{d['chunk']:03d}" for d in metas]
    _cat_gz([out_dir / f"{s}.nwk.gz" for s in stems], final)
    _cat_gz([out_dir / f"{s}.raw.nwk.gz" for s in stems], out_dir / f"{cell_key}.raw.nwk.gz")

    total = sum(d["count"] for d in metas)
    got = _count_lines(final)
    if got != total:
        raise RuntimeError(f"{cell_key}: merged {got} trees, expected {total}")

    def wmean(key):
        vals = [(d[key], d["count"]) for d in metas if key in d]
        return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None

    n_gap = sum(d["n_gap_chars"] for d in metas)
    n_char = sum(d["n_aln_chars"] for d in metas)
    aln_total = sum(d["aln_len_total"] for d in metas)
    pooled = dict(metas[0])
    pooled |= {
        "chunk": None, "n_chunks": len(metas), "start": 0, "count": total, "n_loci": total,
        "aln_len_min": min(d["aln_len_min"] for d in metas),
        "aln_len_max": max(d["aln_len_max"] for d in metas),
        "aln_len_total": aln_total, "aln_len_mean": aln_total / total,
        "n_gap_chars": n_gap, "n_aln_chars": n_char,
        "gap_fraction": n_gap / n_char if n_char else 0.0,
        "alisim_seconds": sum(d["alisim_seconds"] for d in metas),
        "iqtree_seconds": sum(d["iqtree_seconds"] for d in metas),
        "n_root_polytomy": sum(d["n_root_polytomy"] for d in metas),
        "n_alisim_retries": sum(d.get("n_alisim_retries", 0) for d in metas),
    }
    # RF summaries are per-locus means, so a locus-count weighting is exact for the mean and an
    # approximation for the median/p90 -- flagged rather than recomputed, which would need the
    # per-locus values kept.
    for key in ("rf_norm_mean", "rf_norm_median", "rf_norm_p90", "frac_rf_zero",
                "frac_root_split_correct"):
        v = wmean(key)
        if v is not None:
            pooled[key] = v
    tmp = out_dir / f"{cell_key}.json.tmp"
    tmp.write_text(json.dumps(pooled, indent=1) + "\n")
    tmp.rename(out_dir / f"{cell_key}.json")
    for s in stems:
        for suf in ("nwk.gz", "raw.nwk.gz", "json"):
            (out_dir / f"{s}.{suf}").unlink(missing_ok=True)
    return pooled
