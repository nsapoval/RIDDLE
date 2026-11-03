"""Driving ``gene_trees.jl`` from the Python grid.

PhyloCoalSimulations is single-threaded and Julia's package load costs ~10-15 s, so the unit of
work is a *manifest shard*, not a cell: one process validates and simulates many networks before
exiting.  Sharding across processes is the only parallelism available (the package has no
threading; the documented hook is the explicit ``rng`` argument, which the script uses).
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["JULIA", "DEPOT", "available", "write_manifest", "run_shards"]

REPO = Path(__file__).resolve().parents[3]
JULIA = Path(os.environ.get("OPC_JULIA", "/home/ns58/software/julia-1.12.6/bin/julia"))
DEPOT = Path(os.environ.get("JULIA_DEPOT_PATH", str(REPO / ".julia")).split(":")[0])
SCRIPT = REPO / "src" / "opc" / "simulate" / "gene_trees.jl"
PROJECT = REPO / "julia"


def available() -> bool:
    return JULIA.exists() and (PROJECT / "Manifest.toml").exists()


def write_manifest(jobs, path: Path) -> Path:
    path.write_text(json.dumps(jobs, indent=1))
    return path


def _run_one(manifest: Path) -> tuple[int, str]:
    env = dict(
        os.environ,
        JULIA_DEPOT_PATH=str(DEPOT),
        JULIA_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    res = subprocess.run(
        [str(JULIA), f"--project={PROJECT}", str(SCRIPT), str(manifest)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    return res.returncode, (res.stdout + res.stderr)


def run_shards(jobs, workdir: Path, n_shards: int) -> list[str]:
    """Split ``jobs`` across ``n_shards`` Julia processes.  Returns the failure lines."""
    workdir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, min(n_shards, len(jobs)))
    shards = [jobs[i::n_shards] for i in range(n_shards)]
    paths = [
        write_manifest(s, workdir / f"manifest_{i:04d}.json")
        for i, s in enumerate(shards)
        if s
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        for code, out in pool.map(_run_one, paths):
            for line in out.splitlines():
                if line.startswith("FAILED") or (code != 0 and "ERROR" in line):
                    failures.append(line)
    return failures
