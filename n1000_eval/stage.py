#!/usr/bin/env python3
"""Stage ``l2_n1000_gt_50k_v1`` for the benchmark, and gate every staged file.

This writes only under this directory. The dataset itself is only read.

    inputs/n1000/<RR>/true_net.nwk    the network with OUT grafted at the root: (OUT:1,<net>:0);
    inputs/n1000/<RR>/est_bp1000.nwk  estimated_bp1000/<key>.raw.nwk.gz, decompressed (~330 MB)
    tob/<key>.true.rooted.nwk         (OUT,<true TOB>), where the true TOB has one edge per bridge
    CELLS.tsv, stage.json             written only when every gate passes

This is ``l2reg_eval/stage.py`` with two changes.

* **There is no TREE-QMC tree of blobs.** The resolvers' inferred TOB is the GNN's, staged after
  the GNN grid by ``stage_gnn_tob.py``. The dataset's JSON carries no ``true_tob`` either, so the
  true one is built here from the network's bridges (``common.tree_of_blobs_general``).
* **The tree count is checked against MANIFEST ``n_est_loci`` (10,000).** ``n_loci`` (50,000)
  counts the true trees.

Gates (exit 1, and neither CELLS.tsv nor stage.json is written):

    names       opc.rl.camus resolves phase l2_n1000_gt50k / arm est_bp1000 to exactly the staged
                truth and trees, rooted on OUT, with n = 1000
    gene trees  the count equals n_est_loci; every tree has branch lengths; rooted on OUT (as
                prepare_gene_trees does), every tree's leaf set is exactly t1..t1000
    truth       staged-and-pruned == direct parse in label clusters, n_retics == MANIFEST k, and
                node count; level == 2
    true TOB    it parses, carries t1..t1000 + OUT and prunes; every cluster is a hardwired
                cluster of the truth; it has exactly n_l2 + n_l1 polytomies (one per blob), and
                as many as the truth has cycle components

Usage:  gnn_cpu_v1/.venv/bin/python n1000_eval/stage.py [--force] [--jobs 8]   (~5-10 min, 6.6 GB)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import common as C

from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

TAXA = frozenset(f"t{i}" for i in range(1, C.N_TAXA + 1))


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


def blob_degrees(rooted: str) -> list[int]:
    """The child count of every polytomy, as ``run_blob_cpu.tob_stats`` counts ``blob_degrees``."""
    _parent, _len, _lab, children = OG.parse_newick(rooted)
    return sorted((len(c) for c in children if len(c) > 2), reverse=True)


def stage_names(rep: int, truth_file: Path, trees_file: Path) -> list[str]:
    cell = CAMUS.CamusCell(C.PHASE, rep, C.ARM, C.INPUTS)
    bad = []
    if cell.base_path != truth_file:
        bad.append(f"names: {C.PHASE} rep {rep} resolves its truth to {cell.base_path}")
    if cell.trees_path != trees_file:
        bad.append(f"names: {C.ARM} resolves the trees to {cell.trees_path}, staged {trees_file}")
    if cell.rooting != "outgroup" or cell.n != C.N_TAXA:
        bad.append(f"names: rooting {cell.rooting!r}, n {cell.n}; expected outgroup, {C.N_TAXA}")
    return bad


def stage_truth(row: dict) -> tuple[list[str], dict]:
    dst = C.INPUTS / "n1000" / f"{int(row['rep']):02d}" / "true_net.nwk"
    text = (C.PKG / row["network"]).read_text()
    staged = f"({C.OUT}:1,{body(text)}:0);\n"
    if not dst.exists() or dst.read_text() != staged:
        write_atomic(dst, staged)
    direct = N.from_enewick(text)
    pruned = _prune_outgroup(N.from_enewick(dst.read_text()))
    k = int(row["k"])
    bad = []
    if set(direct.labels) != TAXA:
        bad.append(f"truth: {len(set(direct.labels) ^ TAXA)} labels differ from t1..t{C.N_TAXA}")
    if CN.label_clusters(pruned) != CN.label_clusters(direct):
        bad.append("truth: the pruned staged network and the direct parse differ in label clusters")
    if not pruned.n_retics == direct.n_retics == k:
        bad.append(f"truth: n_retics {pruned.n_retics} staged / {direct.n_retics} direct, "
                   f"MANIFEST k = {k}")
    if pruned.n_nodes != direct.n_nodes:
        bad.append(f"truth: {pruned.n_nodes} nodes staged-and-pruned, {direct.n_nodes} parsed "
                   f"directly -- the pruning left a node of the graft behind")
    if N.level(direct) != C.LEVEL:
        bad.append(f"truth: level {N.level(direct)}, expected {C.LEVEL}")
    return bad, {"truth": direct, "file": dst, "n_nodes": direct.n_nodes}


def stage_trees(row: dict, force: bool) -> tuple[list[str], dict]:
    src = C.PKG / row["estimated_raw"]
    dst = C.trees_path(int(row["rep"]))
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with gzip.open(src, "rt") as fi, open(tmp, "w") as fo:
            shutil.copyfileobj(fi, fo)
        tmp.replace(dst)
    bad, n = [], 0
    with open(dst) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            rooted = OG.reroot_on_outgroup(line, keep_lengths=True)
            parent, _len, lab, children = OG.parse_newick(rooted)
            got = frozenset(lab[i] for i in range(len(parent)) if not children[i])
            if got != TAXA:
                bad.append(f"trees: locus {n - 1} carries {len(got)} taxa "
                           f"({len(got ^ TAXA)} differ from t1..t{C.N_TAXA})")
            if ":" not in rooted:
                bad.append(f"trees: locus {n - 1} has no branch lengths")
            if len(bad) >= 5:
                bad.append("trees: stopped after 5 problems")
                break
    if n != int(row["n_est_loci"]) and len(bad) < 5:
        bad.append(f"trees: {n} in the file, MANIFEST n_est_loci = {row['n_est_loci']}")
    return bad, {"file": dst, "source": src, "n_loci": n}


def stage_true_tob(row: dict, truth) -> tuple[list[str], dict]:
    key = row["key"]
    rooted = f"({C.OUT},{body(C.tree_of_blobs_general(truth))});"
    dst = C.TOB / f"{key}.true.rooted.nwk"
    write_atomic(dst, rooted + "\n")
    bad = []
    net = N.from_enewick(rooted)
    if set(net.labels) != TAXA | {C.OUT}:
        return [f"true TOB: {len(set(net.labels) ^ (TAXA | {C.OUT}))} labels differ"], {}
    try:
        _prune_outgroup(net)
    except ValueError as exc:
        bad.append(f"true TOB: {exc}")
    clusters = CN.restrict_clusters(net, CN.ingroup_of(net))
    stray = clusters - CN.label_clusters(truth)
    if stray:
        bad.append(f"true TOB: {len(stray)} clusters are not hardwired clusters of the network")
    degrees = blob_degrees(rooted)
    n_blobs = int(row["n_l2"]) + int(row["n_l1"])
    n_cyc = C.pte().n_cycle_components(truth)
    if not len(degrees) == n_blobs == n_cyc:
        bad.append(f"true TOB: {len(degrees)} polytomies, n_l2 + n_l1 = {n_blobs}, "
                   f"cycle components = {n_cyc}")
    return bad, {"file": dst, "n_blobs": len(degrees), "blob_degrees": degrees}


def stage_one(row: dict, force: bool) -> dict:
    """One network, end to end.  Top-level so a spawned worker can unpickle it."""
    rep = int(row["rep"])
    bad, truth = stage_truth(row)
    tbad, trees = stage_trees(row, force)
    obad, tob = stage_true_tob(row, truth["truth"]) if not bad else ([], {})
    bad += tbad + obad + stage_names(rep, truth["file"], trees["file"])
    if bad:
        return {"key": row["key"], "bad": bad}
    files = {"true_net.nwk": truth["file"], "trees": trees["file"], "tob.true": tob["file"]}
    return {
        "key": row["key"], "bad": [], "rep": rep, "k": int(row["k"]), "cu": float(row["cu"]),
        "n_l2": int(row["n_l2"]), "n_l1": int(row["n_l1"]), "shapes": row["shapes"],
        "n_loci": trees["n_loci"], "n_nodes": truth["n_nodes"],
        "n_blobs": {"true": tob["n_blobs"]}, "blob_degrees": {"true": tob["blob_degrees"]},
        "files": {name: str(p.relative_to(C.ROOT)) for name, p in files.items()},
        "sha256": {**{name: sha256(p) for name, p in files.items()},
                   "source_estimated_raw": sha256(trees["source"])},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="decompress the gene trees again even when already staged")
    ap.add_argument("--jobs", type=int, default=8, help="networks staged concurrently")
    ap.add_argument("--keys", default="", help="reps, e.g. 0-4; default all (CELLS.tsv needs all)")
    args = ap.parse_args()

    want = C.parse_reps(args.keys)
    rows = [r for r in C.manifest() if want is None or int(r["rep"]) in want]
    t0 = time.perf_counter()
    print(f"[stage] {len(rows)} networks: {C.PKG} -> {C.ROOT} (phase {C.PHASE}, arm {C.ARM}), "
          f"jobs {args.jobs}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(stage_one, rows, [args.force] * len(rows)))

    failures = [f"{r['key']}: {b}" for r in results for b in r["bad"]]
    for r in results:
        if r["bad"]:
            print(f"  {r['key']:30s} FAILED: {r['bad'][0]}")
        else:
            print(f"  {r['key']:30s} k={r['k']} trees={r['n_loci']} blobs={r['n_blobs']['true']} "
                  f"degrees={r['blob_degrees']['true']}")
    if failures:
        print(f"[stage] FAILED on {len({f.split(':')[0] for f in failures})} network(s); "
              f"CELLS.tsv and stage.json not written:")
        print("\n".join(f"  {f}" for f in failures))
        return 1
    if want is not None:
        print(f"[stage] OK on {len(results)} networks; --keys given, so CELLS.tsv and stage.json "
              f"are not written")
        return 0

    recs = [{k: v for k, v in r.items() if k != "bad"} for r in results]
    with open(C.CELLS, "w", newline="") as fh:
        fields = ("phase", "rep", "arm", "n", "k", "trees", "n_loci")
        fh.write("\t".join(fields) + "\n")
        for r in recs:
            c = {"phase": C.PHASE, "rep": r["rep"], "arm": C.ARM, "n": C.N_TAXA, "k": r["k"],
                 "trees": C.TREES, "n_loci": r["n_loci"]}
            fh.write("\t".join(str(c[f]) for f in fields) + "\n")
    C.STAGE.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "dataset_manifest_sha256": sha256(C.PKG / "MANIFEST.tsv"),
        "phase": C.PHASE, "arm": C.ARM, "gene_trees": C.GENE_TREES, "cells": recs,
    }, indent=1) + "\n")
    print(f"[stage] OK: {len(recs)} networks, every gate passed | "
          f"{time.perf_counter() - t0:.0f} s -> {C.CELLS.name}, {C.STAGE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
