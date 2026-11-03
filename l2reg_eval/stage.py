#!/usr/bin/env python3
"""Stage ``l2reg_n200_v1`` for the benchmark, and gate every staged file.

Writes under this directory; the dataset itself is only read:

    inputs/n200/<RR>/true_net.nwk   the network with OUT grafted at the root: (OUT:1,<net>:0);
    inputs/n200/<RR>/est_bp1000.nwk estimated_bp1000/<key>.raw.nwk.gz, decompressed
    tob/<key>.raw.rooted.nwk        root_tob_on_outgroup(<key>.raw.tob) -- the camus protocol
    tob/<key>.fixed.rooted.nwk      (OUT,<key>.tob) -- the dataset's corrected tree of blobs
    tob/<key>.true.rooted.nwk       (OUT,true_tob) -- one edge per bridge of the network
    CELLS.tsv, stage.json           written only when every gate passes

``CELLS.tsv`` names every cell with phase ``l2reg_n200`` and arm ``est_bp1000``, which
``opc.rl.camus`` resolves to the two ``inputs/`` files above -- checked, not assumed.  A staging
from before those names existed (the trees as ``g_500.nwk``, cells as ``cam200`` / ``cam_g500``) is
migrated in place: the decompressed trees are renamed, not rewritten, so their bytes and the
resolver runs that read them stay valid.

Why each file is shaped the way it is
-------------------------------------
* **OUT on the truth.**  ``camus.truth_of`` prunes OUT with ``external._prune_outgroup``, which
  refuses a network without it.  Grafted at a degree-2 root, pruning is an exact inverse -- which
  the gate checks against a direct parse by **label** cluster, because ``_prune_outgroup``
  renumbers taxa in postorder and index sets would not compare, and by node count, so that no
  node the graft added survives the pruning.
* **Decompressed raw trees.**  ``camus.prepare_gene_trees`` and BROOQS both ``open()`` the file as
  text, and only the ``.raw`` trees carry OUT and branch lengths (the ``.nwk.gz`` has neither).
  They are also exactly what TREE-QMC ``--blob`` read to build these trees of blobs.
* **Three trees of blobs.**  ``raw`` is what the camus NetCS/BROOQS rows ran on, TREE-QMC's
  root-branch defect included (12 of 20 networks here); ``fixed`` is ``<key>.tob``, TREE-QMC's own
  contraction rule applied to that branch; ``true`` is the oracle.  The true one comes from the
  dataset's JSON, built from the network's bridges.  ``camus_nets.tree_of_blobs`` contracts one
  cycle per reticulation and is wrong at level 2; nothing here calls it.

Gates -- exit 1, and neither CELLS.tsv nor stage.json is written:

    names            opc.rl.camus resolves phase l2reg_n200 / arm est_bp1000 to exactly the
                     staged truth and trees, rooted on OUT, n = 200
    gene trees       MANIFEST n_loci per network; every tree has branch lengths, and rooted on
                     OUT (as prepare_gene_trees does) its leaf set is exactly t1..t200
    truth            staged-and-pruned == direct parse: label clusters, n_retics == MANIFEST k,
                     node count
    trees of blobs   all three parse, carry t1..t200 + OUT, and prune (OUT a child of a degree-2
                     root); missed/extra clusters against the true TOB equal the JSON's fn/fp
                     (fixed) and fn_raw/fp_raw (raw); raw != fixed exactly where TREE-QMC kept a
                     contracted root branch; every true-TOB cluster is a hardwired cluster of the
                     truth; the true TOB has one polytomy per blob

No distance is computed here.  Every luay, and every network size, is PhyloNet's, in make_csv.py.

Usage:  gnn_cpu_v1/.venv/bin/python l2reg_eval/stage.py [--force]      (~20 s)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import common as C

from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

TAXA = frozenset(f"t{i}" for i in range(1, C.N_TAXA + 1))
LEGACY_TREES = "g_500.nwk"          # the staged trees' name while cells borrowed cam200 / cam_g500


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
    """Children of every polytomy of an OUT-rooted tree of blobs -- ``run_blob_cpu.tob_stats``'s
    ``blob_degrees``, so a count here and a count there mean the same thing."""
    _parent, _len, _lab, children = OG.parse_newick(rooted)
    return sorted((len(c) for c in children if len(c) > 2), reverse=True)


def stage_names(rep: int, truth_file: Path, trees_file: Path) -> list[str]:
    """The camus tables must resolve this cell to exactly the files staged for it."""
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
    dst = C.INPUTS / "n200" / f"{int(row['rep']):02d}" / "true_net.nwk"
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
    return bad, {"truth": direct, "file": dst, "n_nodes": direct.n_nodes}


def stage_trees(row: dict, force: bool) -> tuple[list[str], dict]:
    src = C.PKG / row["estimated_raw"]
    dst = C.trees_path(int(row["rep"]))
    legacy = dst.with_name(LEGACY_TREES)
    migrated = False
    if legacy.exists() and not dst.exists() and not force:
        legacy.rename(dst)                    # same bytes: the resolver runs that read it stay valid
        migrated = True
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with gzip.open(src, "rt") as fi, open(tmp, "w") as fo:
            shutil.copyfileobj(fi, fo)
        tmp.replace(dst)
    if legacy.exists():
        legacy.unlink()                       # superseded by dst, decompressed from the same source
    bad, n = [], 0
    with open(dst) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            # What ``camus.prepare_gene_trees`` does to every line on this arm, and the two checks
            # it makes -- here, so a bad file fails in seconds rather than mid-run.
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
    if n != int(row["n_loci"]) and len(bad) < 5:
        bad.append(f"trees: {n} in the file, MANIFEST n_loci = {row['n_loci']}")
    return bad, {"file": dst, "source": src, "n_loci": n, "migrated": migrated}


def stage_tobs(row: dict, truth) -> tuple[list[str], dict]:
    key = row["key"]
    side = json.loads((C.PKG / "tob_bp1000" / f"{key}.json").read_text())
    rooted = {
        "raw": CN.root_tob_on_outgroup((C.PKG / row["tob_raw"]).read_text().strip()),
        "fixed": f"({C.OUT},{body((C.PKG / row['tob']).read_text())});",
        "true": f"({C.OUT},{body(side['true_tob'])});",
    }
    expect = set(truth.labels) | {C.OUT}
    bad, clusters, degrees, files = [], {}, {}, {}
    for v, s in rooted.items():
        dst = C.TOB / f"{key}.{v}.rooted.nwk"
        write_atomic(dst, s + "\n")
        files[v] = dst
        net = N.from_enewick(s)
        if set(net.labels) != expect:
            bad.append(f"{v} TOB: {len(set(net.labels) ^ expect)} labels differ from t1..t200 + OUT")
            continue
        try:
            _prune_outgroup(net)
        except ValueError as exc:
            bad.append(f"{v} TOB: {exc}")
        clusters[v] = CN.restrict_clusters(net, CN.ingroup_of(net))
        degrees[v] = blob_degrees(s)
    if bad:
        return bad, {}

    true, fixed, raw = clusters["true"], clusters["fixed"], clusters["raw"]
    for name, got in (("fn", len(true - fixed)), ("fp", len(fixed - true)),
                      ("fn_raw", len(true - raw)), ("fp_raw", len(raw - true))):
        if got != side[name]:
            bad.append(f"TOB: {name} = {got} against the true TOB; the dataset's JSON says {side[name]}")
    kept = bool(side["treeqmc_kept_a_contracted_root_branch"])
    if (raw != fixed) != kept:
        bad.append(f"TOB: raw {'differs from' if raw != fixed else 'equals'} fixed "
                   f"({len(raw ^ fixed)} clusters) but treeqmc_kept_a_contracted_root_branch={kept}")
    stray = true - CN.label_clusters(truth)
    if stray:
        bad.append(f"TOB: {len(stray)} true-TOB clusters are not hardwired clusters of the network")
    n_blobs = int(row["n_l2"]) + int(row["n_l1"])
    if not len(degrees["true"]) == side["n_blobs"] == n_blobs:
        bad.append(f"TOB: true TOB has {len(degrees['true'])} polytomies, JSON n_blobs = "
                   f"{side['n_blobs']}, n_l2 + n_l1 = {n_blobs}")
    return bad, {"files": files, "n_blobs": {v: len(d) for v, d in degrees.items()},
                 "blob_degrees": degrees, "raw_vs_fixed": len(raw ^ fixed),
                 **{name: side[name] for name in ("fn", "fp", "fn_raw", "fp_raw")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="decompress the gene trees again even when already staged")
    args = ap.parse_args()

    rows = C.manifest()
    t0 = time.perf_counter()
    print(f"[stage] {len(rows)} networks: {C.PKG} -> {C.ROOT} (phase {C.PHASE}, arm {C.ARM})",
          flush=True)
    print(f"  {'key':30s} {'k':>2s} {'blobs raw/fixed/true':>20s} {'fn/fp':>6s} "
          f"{'raw fn/fp':>9s} {'raw~fixed':>9s}", flush=True)
    cells, recs, failures, migrated = [], [], [], 0
    for row in rows:
        key, rep, k = row["key"], int(row["rep"]), int(row["k"])
        bad, truth = stage_truth(row)
        tbad, trees = stage_trees(row, args.force)
        obad, tobs = stage_tobs(row, truth["truth"])
        bad += tbad + obad + stage_names(rep, truth["file"], trees["file"])
        migrated += trees["migrated"]
        if bad:
            failures += [f"{key}: {b}" for b in bad]
            print(f"  {key:30s} FAILED: {bad[0]}", flush=True)
            continue
        files = {"true_net.nwk": truth["file"], "trees": trees["file"],
                 **{f"tob.{v}": p for v, p in tobs["files"].items()}}
        recs.append({
            "key": key, "rep": rep, "k": k, "cu": float(row["cu"]), "n_l2": int(row["n_l2"]),
            "n_l1": int(row["n_l1"]), "shapes": row["shapes"], "n_loci": trees["n_loci"],
            "n_nodes": truth["n_nodes"], "n_blobs": tobs["n_blobs"],
            "blob_degrees": tobs["blob_degrees"], "raw_vs_fixed": tobs["raw_vs_fixed"],
            **{name: tobs[name] for name in ("fn", "fp", "fn_raw", "fp_raw")},
            "files": {name: str(p.relative_to(C.ROOT)) for name, p in files.items()},
            "sha256": {**{name: sha256(p) for name, p in files.items()},
                       "source_estimated_raw": sha256(trees["source"])},
        })
        cells.append({"phase": C.PHASE, "rep": rep, "arm": C.ARM, "n": C.N_TAXA, "k": k,
                      "trees": C.TREES, "n_loci": trees["n_loci"]})
        nb = tobs["n_blobs"]
        print(f"  {key:30s} {k:2d} {nb['raw']:>8d}/{nb['fixed']}/{nb['true']:<8d}"
              f"{tobs['fn']:>3d}/{tobs['fp']:<3d}{tobs['fn_raw']:>5d}/{tobs['fp_raw']:<4d}"
              f"{tobs['raw_vs_fixed']:>6d}", flush=True)

    if failures:
        print(f"[stage] FAILED on {len({f.split(':')[0] for f in failures})} network(s); "
              f"CELLS.tsv and stage.json not written:")
        print("\n".join(f"  {f}" for f in failures))
        return 1

    with open(C.CELLS, "w", newline="") as fh:
        fields = ("phase", "rep", "arm", "n", "k", "trees", "n_loci")
        fh.write("\t".join(fields) + "\n")
        for c in cells:
            fh.write("\t".join(str(c[f]) for f in fields) + "\n")
    C.STAGE.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "dataset_manifest_sha256": sha256(C.PKG / "MANIFEST.tsv"),
        "phase": C.PHASE, "arm": C.ARM, "gene_trees": C.GENE_TREES, "cells": recs,
    }, indent=1) + "\n")
    n_diff = sum(r["raw_vs_fixed"] > 0 for r in recs)
    print(f"[stage] OK: {len(recs)} networks, every gate passed; raw TOB differs from fixed on "
          f"{n_diff} of {len(recs)}"
          + (f"; {migrated} gene-tree files renamed {LEGACY_TREES} -> {C.TREES}" if migrated else "")
          + f" | {time.perf_counter() - t0:.0f} s -> {C.CELLS.name}, {C.STAGE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
