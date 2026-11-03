#!/usr/bin/env python3
"""Stage the checkpoint's own training and validation networks in camus-data's layout, and gate them.

``_l2mix_pair/s0/config.json`` trained on phases x25 x50 y25 y50 l2t25 l2t50, arms ``true`` and
``est_bp1000``, split ``[700, 800]`` -- reps 0-699 train, 700-799 validation (model selection), 800-899
test.  ``nni_ensemble``'s choice model is fitted here on 0-699 and selected on 700-799 (``trainfit.py``)
instead of being cross-fitted on the benchmarks.  The test reps are not staged.

Writes under this directory; the staged data is only read:

    inputs/<x25..l2t50>/<RR>/true_net.nwk        the truth with OUT grafted at the root: (OUT:1,<net>:0);
    inputs/<x25..l2t50>/<RR>/g_true.nwk.gz       symlink: 1,000 true gene trees (rooted, ultrametric)
    inputs/<x25..l2t50>/<RR>/est_bp1000.nwk.gz   symlink: 1,000 IQ-TREE trees (rooted on OUT, OUT pruned,
                                                 lengths kept) -- l2t's ``.len.nwk.gz``, since its plain
                                                 ``.nwk.gz`` is topology only
    CELLS.tsv, stage.json                        written only when every gate passes

The trees are symlinked, not rewritten: ``opc.rl.camus``'s ``trn_true`` / ``trn_est_bp1000`` arms read
them ``as_is`` -- already rooted and OUT-free, each line passed through untouched -- which is exactly
what ``instances.build`` fed the GNN in training.  Only the truth needs OUT, because ``camus.truth_of``
prunes it with ``external._prune_outgroup``, which refuses a network without one.

Gates -- exit 1, and neither CELLS.tsv nor stage.json is written:

    split    reps are exactly 0..799 per phase; the split column (x/y MANIFEST.tsv) or ``spec.split``
             (l2t JSON) agrees with rep < 700 / < 800, the checkpoint's own cut
    names    opc.rl.camus resolves phase/rep/arm to exactly the staged truth and trees, rooting as_is,
             n the phase's
    truth    staged-and-pruned == direct parse (label clusters, node count); n_retics == the key's r
             (x/y) or the JSON's k (l2t); taxa t1..tn; x/y: also == ``N.derive`` of training's own
             grid cell (``opc.grid.PHASES``), the network instances.build scored against; l2t:
             ``.enewick`` and ``.regular.enewick`` have the same label clusters
    trees    every file: 1,000 trees, each with branch lengths, a bifurcating root and leaf set exactly
             t1..tn (so no OUT); x/y true trees match the MANIFEST sha256; x/y estimated trees with
             lengths stripped equal ``alisim_gt_v*/estimated_bp1000`` line for line; l2t's
             ``.len.nwk.gz`` with lengths stripped equals its ``.nwk.gz``

Usage:  gnn_cpu_v1/.venv/bin/python train_eval/stage.py [--jobs 6]       (~5 min at --jobs 6)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "gnn_cpu_v1" / "pylib"), str(REPO / "gnn_cpu_v1")]

from opc import grid  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

XY = REPO / "results" / "home" / "ns58" / "net-opt" / "results"
L2T = REPO / "data" / "home" / "ns58" / "net-opt" / "data" / "l2reg_train_v1"
INPUTS = HERE / "inputs"
CELLS = HERE / "CELLS.tsv"
STAGE = HERE / "stage.json"
#: ``config.json``'s ``"split": [700, 800]``: train rep < 700, val rep < 800.
SPLIT = (700, 800)
N_LOCI = 1000
#: camus phase -> (source phase, family, alisim package holding its MANIFEST and topology-only estimates)
PHASES = {"trn_x25": ("x25", "xy", REPO / "alisim_gt_v1"), "trn_x50": ("x50", "xy", REPO / "alisim_gt_v1"),
          "trn_y25": ("y25", "xy", REPO / "alisim_gt_v2"), "trn_y50": ("y50", "xy", REPO / "alisim_gt_v2"),
          "trn_l2t25": ("l2t25", "l2t", None), "trn_l2t50": ("l2t50", "l2t", None)}
ARMS = ("trn_true", "trn_est_bp1000")
COLUMNS = ("phase", "rep", "arm", "n", "k", "trees", "n_loci", "key", "split", "cu", "level")
_LEN = re.compile(r":[^,();]+")


def split_of(rep: int) -> str:
    return "train" if rep < SPLIT[0] else "val" if rep < SPLIT[1] else "test"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def body(newick: str) -> str:
    return newick.strip().rstrip(";")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def link(dst: Path, src: Path) -> None:
    """``dst`` -> ``src`` (absolute), replacing a link that points elsewhere."""
    src = src.resolve()
    if dst.is_symlink() and Path(os.readlink(dst)) == src:
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst)


# ------------------------------------------------------------------------------ the sources

def sources(phase: str) -> list[dict]:
    """Every rep < 800 of one phase: key, k, cu, split as the source records them, and its files."""
    src, fam, pkg = PHASES[phase]
    out = []
    if fam == "xy":
        with open(pkg / "MANIFEST.tsv") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["phase"] != src or int(r["rep"]) >= SPLIT[1]:
                    continue
                key = r["cell_key"]
                out.append({"rep": int(r["rep"]), "key": key, "base_key": r["base_key"], "k": int(r["k"]),
                            "cu": float(r["cu"]), "split_src": r["split"], "level": 1,
                            "net": XY / src / "enewick" / f"{key}.enewick",
                            "true": XY / src / "genetrees_julia" / f"{key}.nwk.gz",
                            "est": XY / src / "genetrees_est_julia_bp1000" / f"{key}.nwk.gz",
                            "true_sha": r["sha256"],
                            "est_topo": pkg / "estimated_bp1000" / src / f"{key}.nwk.gz"})
    else:
        for p in sorted((L2T / src / "networks").glob("*.json")):
            spec = json.loads(p.read_text())["spec"]
            if int(spec["rep"]) >= SPLIT[1]:
                continue
            key = spec["key"]
            d = L2T / src
            out.append({"rep": int(spec["rep"]), "key": key, "k": int(spec["k"]), "cu": float(spec["cu"]),
                        "split_src": spec["split"], "level": 2,
                        "net": d / "networks" / f"{key}.enewick",
                        "regular": d / "networks" / f"{key}.regular.enewick",
                        "true": d / "trees" / f"{key}.nwk.gz",
                        "est": d / "estimated_bp1000" / f"{key}.len.nwk.gz",
                        "est_topo": d / "estimated_bp1000" / f"{key}.nwk.gz"})
    return sorted(out, key=lambda r: r["rep"])


# ------------------------------------------------------------------------------ the gates

def gate_split(phase: str, recs: list[dict]) -> list[str]:
    bad = []
    reps = [r["rep"] for r in recs]
    if reps != list(range(SPLIT[1])):
        bad.append(f"split: {phase} has {len(reps)} reps < {SPLIT[1]}, not exactly 0..{SPLIT[1] - 1}")
    for r in recs:
        if r["split_src"] != split_of(r["rep"]):
            bad.append(f"split: {phase} rep {r['rep']} is '{r['split_src']}' in its source, "
                       f"'{split_of(r['rep'])}' by the checkpoint's cut")
    return bad


def stage_truth(phase: str, rec: dict, derived: dict) -> list[str]:
    n = CAMUS.PHASES[phase][1]
    cell = CAMUS.CamusCell(phase, rec["rep"], ARMS[0], INPUTS)
    text = rec["net"].read_text()
    staged = f"({OG.OUTGROUP_LABEL}:1,{body(text)}:0);\n"
    if not cell.base_path.exists() or cell.base_path.read_text() != staged:
        write_atomic(cell.base_path, staged)
    direct = N.from_enewick(text)
    pruned = _prune_outgroup(N.from_enewick(cell.base_path.read_text()))
    taxa = {f"t{i}" for i in range(1, n + 1)}
    where = f"truth: {phase} rep {rec['rep']}"
    bad = []
    if set(direct.labels) != taxa:
        bad.append(f"{where}: {len(set(direct.labels) ^ taxa)} labels differ from t1..t{n}")
    lc = CN.label_clusters(direct)
    if CN.label_clusters(pruned) != lc:
        bad.append(f"{where}: the pruned staged network and the direct parse differ in label clusters")
    if pruned.n_nodes != direct.n_nodes:
        bad.append(f"{where}: {pruned.n_nodes} nodes staged-and-pruned, {direct.n_nodes} parsed directly")
    if not pruned.n_retics == direct.n_retics == rec["k"]:
        bad.append(f"{where}: n_retics {pruned.n_retics} staged / {direct.n_retics} direct, source k {rec['k']}")
    if "base_key" in rec:
        c = derived[rec["key"]]
        with open(XY / PHASES[phase][0] / "networks" / f"{c.base_key}.pkl", "rb") as fh:
            base = pickle.load(fh)
        truth = N.derive(base, r=c.r, cu=c.cu, gamma=c.gamma, gamma_range=c.gamma_range,
                         gamma_seed=c.gamma_seed)
        if CN.label_clusters(truth) != lc or c.r != rec["k"]:
            bad.append(f"{where}: differs from N.derive of training's grid cell {c.key}")
    else:
        if CN.label_clusters(N.from_enewick(rec["regular"].read_text())) != lc:
            bad.append(f"{where}: .enewick and .regular.enewick differ in label clusters")
    return bad


def stage_names(phase: str, rec: dict) -> list[str]:
    bad = []
    for arm, src in zip(ARMS, (rec["true"], rec["est"])):
        cell = CAMUS.CamusCell(phase, rec["rep"], arm, INPUTS)
        link(cell.trees_path, src)
        if cell.trees_path.resolve() != src.resolve() or cell.trees_path.parent != cell.base_path.parent:
            bad.append(f"names: {phase} rep {rec['rep']} {arm} resolves to {cell.trees_path.resolve()}")
        if cell.rooting != "as_is" or cell.n != CAMUS.PHASES[phase][1]:
            bad.append(f"names: {phase} {arm}: rooting {cell.rooting!r}, n {cell.n}")
    return bad


def _strip(line: str) -> str:
    return _LEN.sub("", line)


def check_trees(job: tuple) -> dict:
    """One tree file: count, lengths, root, leaf set; plus its provenance check.  A worker."""
    path, n, mode, ref = job
    taxa = frozenset(f"t{i}" for i in range(1, n + 1))
    bad, n_trees, n_nolen, n_poly, n_taxa_bad = [], 0, 0, 0, 0
    lines = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_trees += 1
            parent, _elen, lab, children = OG.parse_newick(line)
            got = frozenset(lab[i] for i in range(len(parent)) if not children[i])
            n_taxa_bad += got != taxa
            n_nolen += ":" not in line
            n_poly += len(children[0]) != 2
            if mode == "strip":
                lines.append(_strip(line))
    if n_trees != N_LOCI:
        bad.append(f"{n_trees} trees, expected {N_LOCI}")
    if n_nolen:
        bad.append(f"{n_nolen} trees without branch lengths")
    if n_poly:
        bad.append(f"{n_poly} trees whose root is not bifurcating")
    if n_taxa_bad:
        bad.append(f"{n_taxa_bad} trees whose leaf set is not exactly t1..t{n}")
    if mode == "sha" and sha256(path) != ref:
        bad.append("sha256 differs from the MANIFEST")
    if mode == "strip":
        with gzip.open(ref, "rt") as fh:
            want = [x.strip() for x in fh if x.strip()]
        if lines != want:
            diff = sum(a != b for a, b in zip(lines, want)) + abs(len(lines) - len(want))
            bad.append(f"lengths stripped, {diff} lines differ from {ref.name}")
    return {"path": str(path), "problems": bad}


# ------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    t0 = time.perf_counter()
    problems, rows, jobs, counts = [], [], [], {}
    for phase, (src, fam, _pkg) in PHASES.items():
        n = CAMUS.PHASES[phase][1]
        recs = sources(phase)
        problems += gate_split(phase, recs)
        derived = ({c.key: c for c in grid.enumerate_cells(**grid.PHASES[src])} if fam == "xy" else {})
        for rec in recs:
            problems += stage_truth(phase, rec, derived)
            problems += stage_names(phase, rec)
            jobs.append((rec["true"], n, "sha" if fam == "xy" else "none", rec.get("true_sha")))
            jobs.append((rec["est"], n, "strip", rec["est_topo"]))
            for arm in ARMS:
                rows.append({"phase": phase, "rep": rec["rep"], "arm": arm, "n": n, "k": rec["k"],
                             "trees": CAMUS.ARMS[arm][0], "n_loci": N_LOCI, "key": rec["key"],
                             "split": split_of(rec["rep"]), "cu": f"{rec['cu']:g}", "level": rec["level"]})
        counts[phase] = {s: sum(1 for r in recs if split_of(r["rep"]) == s) for s in ("train", "val")}
        print(f"[stage] {phase}: {len(recs)} networks staged ({counts[phase]}), "
              f"{len(problems)} problems so far, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"[stage] checking {len(jobs)} tree files at --jobs {args.jobs}", flush=True)
    with get_context("fork").Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(check_trees, jobs, chunksize=8), 1):
            problems += [f"trees: {r['path']}: {p}" for p in r["problems"]]
            if i % 2000 == 0:
                print(f"[stage]   {i}/{len(jobs)} files, {time.perf_counter() - t0:.0f} s", flush=True)
    for p in problems[:50]:
        print("[stage] PROBLEM " + p, file=sys.stderr)
    if problems:
        print(f"[stage] FAILED: {len(problems)} problems; CELLS.tsv and stage.json not written", file=sys.stderr)
        return 1
    text = "\t".join(COLUMNS) + "\n" + "".join("\t".join(str(r[c]) for c in COLUMNS) + "\n" for r in rows)
    write_atomic(CELLS, text)
    write_atomic(STAGE, json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "split": list(SPLIT), "n_loci": N_LOCI,
        "phases": {p: {"source": str((XY if f == "xy" else L2T) / s), "alisim": str(k) if k else None, **counts[p]}
                   for p, (s, f, k) in PHASES.items()},
        "arms": {a: CAMUS.ARMS[a][0] for a in ARMS}, "n_cells": len(rows), "n_tree_files": len(jobs),
        "cells_sha256": sha256(CELLS), "gates": ["split", "names", "truth", "trees"], "problems": 0},
        indent=1) + "\n")
    print(f"[stage] wrote {CELLS} ({len(rows)} cells) and {STAGE} in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
