#!/usr/bin/env python3
"""Assemble ``remote_pkg/`` (the resolver hand-off) from the staged GNN and true TOBs.

The package contains:

    remote_pkg/
      README.md              how to set up, check, run, monitor and send back (hand-written)
      run_remote.py          the scheduler: standard library only (hand-written)
      setup_tools.sh         builds TREE-QMC e577f8c + patch, and BROOQS 17b25b2 (hand-written)
      treeqmc-e577f8c.patch  the Mac build's patch, written here from blob_cpu_v1's clone
      MANIFEST.json          written here (below)
      tobs/<key>.{gnn,true}.rooted.nwk   copied byte for byte from n1000_eval/tob/

MANIFEST.json holds, for each network:

* ``raw_gz_sha256``: the dataset's ``.raw.nwk.gz``;
* ``trees_sha256``: its decompression, which is what the Mac's GNN read;
* each TOB's sha256 and eligible polytomy counts;
* the GNN cell sha256 that the ``gnn`` TOB came from.

It also records the tool pins the Mac ran. The remote runner checks every sha256 before any tool
starts.

Gates (exit 1, nothing written):

* the GNN TOBs are current against the GNN cells (``run_resolvers.stale_problems``);
* every staged sha256 in ``stage.json`` still matches the file on disk.

Usage:  gnn_cpu_v1/.venv/bin/python n1000_eval/make_remote_pkg.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import common as C
import run_resolvers as R

PKG = C.HERE / "remote_pkg"


def sh(cmd, cwd=None) -> str:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    return (r.stdout + r.stderr).strip()


def main() -> int:
    stage = json.loads(C.STAGE.read_text())
    cells = stage["cells"]
    bad = R.stale_problems(cells, C.TOB, C.RUNS / C.GNN_RUN)
    for c in cells:
        for name, rel in c["files"].items():
            if R.sha_file(C.ROOT / rel) != c["sha256"][name]:
                bad.append(f"{c['key']}: {rel} changed since staging")
        gz = C.PKG / "estimated_bp1000" / f"{c['key']}.raw.nwk.gz"
        if R.sha_file(gz) != c["sha256"]["source_estimated_raw"]:
            bad.append(f"{c['key']}: {gz.name} differs from the staged source")
    if bad:
        print("[pkg] FAILED:\n" + "\n".join(f"  {b}" for b in bad))
        return 1

    index = json.loads((C.TOB / "gnn_tob.json").read_text())["cells"]
    (PKG / "tobs").mkdir(parents=True, exist_ok=True)
    man_cells = {}
    for c in cells:
        tobs = {}
        for v in C.VARIANTS:
            src = C.TOB / f"{c['key']}.{v}.rooted.nwk"
            dst = PKG / "tobs" / src.name
            shutil.copyfile(src, dst)
            from opc.simulate import outgroup as OG
            _p, _l, _lab, children = OG.parse_newick(src.read_text().strip())
            degs = sorted((len(ch) for ch in children if len(ch) > 2), reverse=True)
            tobs[v] = {"sha256": R.sha_file(dst), "blob_degrees": degs,
                       "n_eligible": {m: sum(d >= t for d in degs) for m, t in R.ELIGIBLE_AT.items()}}
        tobs["gnn"]["gnn_cell_sha256"] = index[c["key"]]["gnn_cell_sha256"]
        man_cells[c["key"]] = {"rep": c["rep"], "k": c["k"], "cu": c["cu"], "n_loci": c["n_loci"],
                               "raw_gz_sha256": c["sha256"]["source_estimated_raw"],
                               "trees_sha256": c["sha256"]["trees"], "tobs": tobs}

    qmc = C.BLOB / "tools" / "TREE-QMC"
    # Byte for byte: never through sh(), whose .strip() ate the last hunk's whitespace-only CRLF
    # context line (tree_of_blobs.cpp is CRLF) and shipped a patch `git apply` calls corrupt.
    live = subprocess.run(["git", "diff"], cwd=qmc, capture_output=True, check=True).stdout
    shipped = (C.BLOB / "tools" / "treeqmc-e577f8c-macos-clang.patch").read_bytes()
    if live != shipped:
        print("[pkg] FAILED: blob_cpu_v1's TREE-QMC clone differs from its committed patch")
        return 1
    (PKG / "treeqmc-e577f8c.patch").write_bytes(shipped)
    brq_py = C.BLOB / ".venv-brooqs" / "bin" / "python"
    tools = {
        "tree_qmc_commit": sh(["git", "rev-parse", "HEAD"], cwd=qmc),
        "tree_qmc_version": (qmc / "version.txt").read_text().strip(),
        "tree_qmc_patch_sha256": hashlib.sha256(shipped).hexdigest(),     # raw `git diff` bytes
        "brooqs_commit": sh(["git", "rev-parse", "HEAD"], cwd=C.BLOB / "tools" / "BROOQS"),
        "brooqs_env": sh([brq_py, "-c", "import sys; from importlib.metadata import version as v; "
                          "print(sys.version.split()[0], *(f'{p}=={v(p)}' for p in "
                          "('brooqs', 'numpy', 'treeswift')))"]),
        "brooqs_args": "--rooted --dist-var", "netcs_args": "",
    }
    man = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": C.PKG.name,
           "dataset_manifest_sha256": stage["dataset_manifest_sha256"],
           "gnn_tob_source": {"tag": C.TOB_SOURCE[0], "seed": C.TOB_SOURCE[1],
                              "decoder": C.TOB_SOURCE[2], "run": C.GNN_RUN},
           "mac_measured": {
               "note": "Mac M-series, 4 workers under swap pressure, 2026-10-06/07",
               "brooqs_hours": [1.05, 3.27, 1.54, 1.25], "brooqs_rss_gb_max": 5.4,
               "netcs_hours": "rep02: 3.37; rep00, rep01, rep03: > 12 (timed out)",
               "netcs_rss_gb_max": 12.1},
           "tools": tools, "cells": man_cells}
    (PKG / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    n = sum(1 for _ in (PKG / "tobs").glob("*.nwk"))
    print(f"[pkg] {PKG}: {len(man_cells)} networks, {n} TOBs, MANIFEST.json, patch "
          f"({tools['tree_qmc_patch_sha256'][:12]}) | tools: TREE-QMC {tools['tree_qmc_commit'][:7]}, "
          f"BROOQS {tools['brooqs_commit'][:7]} ({tools['brooqs_env']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
