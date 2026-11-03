#!/usr/bin/env python3
"""Stage ``sptree_n200_gt10k_v1`` for the GNN and ASTRAL-IV, and gate every file.

The dataset is only read.  Writes, under this directory:

    inputs/n200/<RR>/true_net.nwk    the species tree with OUT grafted at the root: (OUT:1,<tree>:0);
    inputs/n200/<RR>/est_bp1000.nwk  estimated_bp1000/<key>.raw.nwk.gz, decompressed byte for byte
    CELLS.tsv, stage.json            written only when every gate passes

``CELLS.tsv`` names each cell with phase ``sptree_n200`` and arm ``est_bp1000``, which
``opc.rl.camus`` resolves to the two files above, so the GNN reads the same arm, through the same
rooting, as every landed benchmark (``l2reg_eval/stage.py``, whose truth recipe this copies).
ASTRAL-IV reads the same staged file (``run_astral.py``).

Gates -- exit 1, and neither CELLS.tsv nor stage.json is written:

    D0 integrity   exactly the 50 keys n200_repRR_cu<0.1|0.25|0.5>, reps 0..49; no dot-file or
                   partial-copy file in networks/ or estimated_bp1000/; every .raw / .nwk / .len
                   gzip read to its end (CRC and trailer), 10,000 lines, none blank, each ending
                   ';'; the sidecar's n_loci == count == graft_n_loci == 10,000 at start 0, its
                   key and cu the file's; the network JSON's check.ok / no_reticulation / binary,
                   and .regular.enewick byte-identical to .enewick
    S1 truth       staged-and-pruned == direct parse (label clusters, node count), labels exactly
                   t1..t200, 0 reticulations, 198 non-trivial clusters
    S2 trees       the staged file is the decompressed raw file (sha256); every line rooted on OUT
                   as prepare_gene_trees roots it has leaf set t1..t200 and branch lengths, and is
                   **string-identical** to the shipped .len.nwk.gz line (so its topology and lengths
                   are the dataset's own rooting), and its topology is the .nwk.gz line's
    S3 names       CamusCell(sptree_n200, rep, est_bp1000) resolves to the staged files, rooted on
                   OUT, n = 200

``--check-only`` runs D0 and writes nothing -- the check to poll while a copy is in flight.
``--verify-sums`` also runs the dataset's SHA256SUMS, once one ships (none has yet).

Usage:  gnn_cpu_v1/.venv/bin/python sptree_eval/stage.py [--check-only] [--jobs 8] [--force]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import sp_common as SP

from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

VARIANTS = {"raw": ".raw.nwk.gz", "nwk": ".nwk.gz", "len": ".len.nwk.gz"}
PARTIAL_COPY = re.compile(r"^\.|\.(part|tmp)$|~$")
LENGTHS = re.compile(r":[^,();]+")


def body(newick: str) -> str:
    return newick.strip().rstrip(";")


def keys() -> tuple[list[str], list[str]]:
    """The dataset's keys, and every D0 problem with the key set or the directories."""
    bad = []
    for sub in ("networks", "estimated_bp1000"):
        stray = sorted(p.name for p in (SP.PKG / sub).iterdir() if PARTIAL_COPY.search(p.name))
        if stray:
            bad.append(f"D0: {sub}/ holds {len(stray)} dot-file or partial-copy names, e.g. {stray[:3]} "
                       f"-- is a copy still running?")
    ks = sorted(p.name.removesuffix(".json") for p in (SP.PKG / "networks").glob("n200_*.json"))
    reps = []
    for k in ks:
        m = SP.KEY_RE.match(k)
        if not m:
            bad.append(f"D0: unexpected key {k}")
        else:
            reps.append(int(m.group(1)))
    if sorted(reps) != list(range(50)):
        bad.append(f"D0: reps {sorted(reps)[:5]}... ({len(reps)}), expected exactly 0..49 once each")
    return ks, bad


def d0(key: str) -> tuple[list[str], dict]:
    """Integrity of one cell's source files."""
    bad, info = [], {}
    rep, cu = int(SP.KEY_RE.match(key).group(1)), float(SP.KEY_RE.match(key).group(2))
    net = SP.PKG / "networks"
    est = SP.PKG / "estimated_bp1000"
    for v, suf in VARIANTS.items():
        p = est / f"{key}{suf}"
        if not p.exists():
            bad.append(f"D0: no {p.name}")
            continue
        n = blank = unterminated = 0
        h = hashlib.sha256()
        try:
            with gzip.open(p, "rb") as fh:
                for line in fh:
                    if v == "raw":
                        h.update(line)
                    s = line.strip()
                    if not s:
                        blank += 1
                        continue
                    n += 1
                    unterminated += not s.endswith(b";")
        except (OSError, EOFError) as exc:
            bad.append(f"D0: {p.name} does not decompress to its end ({type(exc).__name__}: {exc})")
            continue
        if (n, blank, unterminated) != (SP.N_LOCI, 0, 0):
            bad.append(f"D0: {p.name}: {n} trees, {blank} blank lines, {unterminated} without ';' "
                       f"(want {SP.N_LOCI}, 0, 0)")
        if v == "raw":
            info["raw_decompressed_sha256"] = h.hexdigest()
    side = json.loads((est / f"{key}.json").read_text())
    want = {"cell_key": key, "start": 0, "n_taxa": SP.N_TAXA, "n_loci": SP.N_LOCI,
            "count": SP.N_LOCI, "graft_n_loci": SP.N_LOCI}
    for f, w in want.items():
        if side.get(f) != w:
            bad.append(f"D0: sidecar {f} = {side.get(f)!r}, want {w!r}")
    if float(side.get("cu", -1)) != cu:
        bad.append(f"D0: sidecar cu {side.get('cu')} against the key's {cu}")
    info["side"] = {f: side.get(f) for f in ("rf_norm_mean", "rf_norm_median", "frac_root_split_correct",
                                              "gap_fraction", "iqtree_model_used", "n_root_polytomy")}
    nj = json.loads((net / f"{key}.json").read_text())
    chk = nj.get("check", {})
    for f in ("ok", "no_reticulation", "binary"):
        if chk.get(f) is not True:
            bad.append(f"D0: network check.{f} = {chk.get(f)!r}")
    if nj.get("n_taxa") != SP.N_TAXA or nj.get("key") != key:
        bad.append(f"D0: network JSON n_taxa {nj.get('n_taxa')} key {nj.get('key')}")
    if (net / f"{key}.regular.enewick").read_bytes() != (net / f"{key}.enewick").read_bytes():
        bad.append("D0: .regular.enewick differs from .enewick -- a species tree is its own regular form")
    info.update(rep=rep, cu=cu, root_height=nj.get("root_height"),
                median_internal_branch=nj.get("median_internal_branch"))
    return bad, info


def s1_truth(key: str, rep: int) -> tuple[list[str], dict]:
    dst = SP.truth_path(rep)
    text = (SP.PKG / "networks" / f"{key}.enewick").read_text()
    staged = f"({SP.OUT}:1,{body(text)}:0);\n"
    if not dst.exists() or dst.read_text() != staged:
        SP.write_atomic(dst, staged)
    direct = N.from_enewick(text)
    pruned = _prune_outgroup(N.from_enewick(dst.read_text()))
    bad = []
    if set(direct.labels) != SP.TAXA:
        bad.append(f"S1: {len(set(direct.labels) ^ SP.TAXA)} labels differ from t1..t{SP.N_TAXA}")
    lc = CN.label_clusters(direct)
    if CN.label_clusters(pruned) != lc:
        bad.append("S1: the pruned staged tree and the direct parse differ in label clusters")
    if pruned.n_retics or direct.n_retics:
        bad.append(f"S1: n_retics {pruned.n_retics} staged / {direct.n_retics} direct, want 0")
    if pruned.n_nodes != direct.n_nodes:
        bad.append(f"S1: {pruned.n_nodes} nodes staged-and-pruned, {direct.n_nodes} parsed directly")
    if len(lc) != SP.N_TRUE:
        bad.append(f"S1: {len(lc)} non-trivial clusters, want {SP.N_TRUE}")
    return bad, {"file": dst, "n_nodes": direct.n_nodes, "n_clusters": len(lc)}


def s2_trees(key: str, rep: int, raw_sha: str, force: bool) -> tuple[list[str], dict]:
    est = SP.PKG / "estimated_bp1000"
    src = est / f"{key}.raw.nwk.gz"
    dst = SP.trees_path(rep)
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with gzip.open(src, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        tmp.replace(dst)
    bad = []
    staged_sha = SP.sha256(dst)
    if staged_sha != raw_sha:
        bad.append(f"S2: the staged trees ({staged_sha[:12]}) are not the decompressed raw file "
                   f"({raw_sha[:12]}) -- rerun with --force")
        return bad, {}
    n = n_len = n_topo = 0
    with open(dst) as fs, gzip.open(est / f"{key}.len.nwk.gz", "rt") as fl, \
            gzip.open(est / f"{key}.nwk.gz", "rt") as ft:
        for line, lline, tline in zip(fs, fl, ft):
            rooted = OG.reroot_on_outgroup(line.strip(), keep_lengths=True)
            parent, _len, lab, children = OG.parse_newick(rooted)
            got = frozenset(lab[i] for i in range(len(parent)) if not children[i])
            if got != SP.TAXA:
                bad.append(f"S2: locus {n} carries {len(got)} taxa ({len(got ^ SP.TAXA)} differ from t1..t200)")
            if ":" not in rooted:
                bad.append(f"S2: locus {n} has no branch lengths")
            n_len += rooted != lline.strip()
            n_topo += LENGTHS.sub("", rooted) != tline.strip()
            n += 1
            if len(bad) >= 5:
                bad.append("S2: stopped after 5 problems")
                break
    if n != SP.N_LOCI and len(bad) < 5:
        bad.append(f"S2: compared {n} lines, want {SP.N_LOCI}")
    if n_len:
        bad.append(f"S2: {n_len} of {n} rooted trees differ from the shipped .len.nwk.gz line")
    if n_topo:
        bad.append(f"S2: {n_topo} of {n} rooted topologies differ from the shipped .nwk.gz line")
    return bad, {"file": dst, "n_loci": n, "sha256": staged_sha,
                 "prefix_sha256": {t: hashlib.sha256(SP.prefix_bytes(dst, m)).hexdigest()
                                   for t, m, _ in SP.SIZES}}


def s3_names(rep: int, truth_file: Path, trees_file: Path) -> list[str]:
    cell = CAMUS.CamusCell(SP.PHASE, rep, SP.ARM, SP.INPUTS)
    bad = []
    if cell.base_path != truth_file:
        bad.append(f"S3: {SP.PHASE} rep {rep} resolves its truth to {cell.base_path}")
    if cell.trees_path != trees_file:
        bad.append(f"S3: {SP.ARM} resolves the trees to {cell.trees_path}, staged {trees_file}")
    if cell.rooting != "outgroup" or cell.n != SP.N_TAXA:
        bad.append(f"S3: rooting {cell.rooting!r}, n {cell.n}; expected outgroup, {SP.N_TAXA}")
    return bad


def one(payload) -> dict:
    key, check_only, force = payload
    t0 = time.perf_counter()
    try:
        bad, info = d0(key)
        if check_only or bad:
            return {"key": key, "bad": bad, "info": info, "seconds": time.perf_counter() - t0}
        rep = info["rep"]
        tbad, truth = s1_truth(key, rep)
        sbad, trees = s2_trees(key, rep, info["raw_decompressed_sha256"], force)
        bad += tbad + sbad
        if not bad:
            bad += s3_names(rep, truth["file"], trees["file"])
        rec = None
        if not bad:
            sources = {v: SP.PKG / "estimated_bp1000" / f"{key}{suf}" for v, suf in VARIANTS.items()}
            sources["enewick"] = SP.PKG / "networks" / f"{key}.enewick"
            rec = {"key": key, "rep": rep, "cu": info["cu"], "k": 0, "n_loci": trees["n_loci"],
                   "n_nodes": truth["n_nodes"], "n_true_clusters": truth["n_clusters"],
                   "root_height": info["root_height"],
                   "median_internal_branch": info["median_internal_branch"], **info["side"],
                   "files": {"true_net.nwk": str(truth["file"].relative_to(SP.HERE)),
                             "trees": str(trees["file"].relative_to(SP.HERE))},
                   "sha256": {"staged": {"true_net.nwk": SP.sha256(truth["file"]), "trees": trees["sha256"]},
                              "sources": {v: SP.sha256(p) for v, p in sources.items()}},
                   "prefix_sha256": trees["prefix_sha256"],
                   "xcheck": {"len_string_identical": trees["n_loci"], "topology_identical": trees["n_loci"]}}
        return {"key": key, "bad": bad, "info": info, "rec": rec, "seconds": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001 -- reported per cell, and the run fails
        return {"key": key, "bad": [f"{type(exc).__name__}: {exc}"], "info": {}, "seconds": time.perf_counter() - t0}


def verify_sums() -> list[str]:
    p = SP.PKG / "SHA256SUMS"
    if not p.exists():
        print("[stage] --verify-sums: the dataset ships no SHA256SUMS yet; nothing to verify")
        return []
    r = subprocess.run(["shasum", "-a", "256", "-c", "--quiet", "SHA256SUMS"], cwd=SP.PKG,
                       capture_output=True, text=True)
    return [] if r.returncode == 0 else [f"SHA256SUMS: {r.stdout.strip()[:500]} {r.stderr.strip()[:200]}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="D0 only; write nothing")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="decompress the gene trees again")
    ap.add_argument("--verify-sums", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ks, bad = keys()
    if args.verify_sums:
        bad += verify_sums()
    print(f"[stage] {len(ks)} keys in {SP.PKG.name}/networks | {'D0 only' if args.check_only else 'D0 + S1-S3'}"
          f" | jobs {args.jobs}", flush=True)
    payloads = [(k, args.check_only, args.force) for k in ks if SP.KEY_RE.match(k)]
    with get_context("fork").Pool(args.jobs) as pool:
        res = sorted(pool.imap_unordered(one, payloads), key=lambda r: r["key"])
    failures = list(bad) + [f"{r['key']}: {b}" for r in res for b in r["bad"]]
    for r in res:
        s = r["info"].get("side", {})
        print(f"  {r['key']:22s} {'FAILED: ' + r['bad'][0] if r['bad'] else 'ok'}"
              + (f"  gt-RF {s.get('rf_norm_mean', float('nan')):.3f} root-split {s.get('frac_root_split_correct', float('nan')):.3f}"
                 if not r["bad"] else "") + f"  {r['seconds']:.0f}s", flush=True)
    missing = [f for f in ("README.md", "MANIFEST.tsv", "manifest.json", "SHA256SUMS", "code", "tob_bp1000")
               if not (SP.PKG / f).exists()]
    n_true = sum(1 for k in ks if (SP.PKG / "trees" / f"{k}.nwk.gz").exists())
    print(f"[stage] the dataset lacks {missing or 'nothing'}; true gene trees present for {n_true}/{len(ks)} "
          f"(not used: the estimated arm only)")
    if failures:
        print(f"[stage] FAILED ({len(failures)} problems); CELLS.tsv and stage.json not written:")
        print("\n".join(f"  {f}" for f in failures[:40]))
        return 1
    if args.check_only:
        print(f"[stage] D0 OK on {len(res)} cells | {time.perf_counter() - t0:.0f} s (nothing written)")
        return 0

    recs = [r["rec"] for r in res]
    fields = ("phase", "rep", "arm", "n", "k", "trees", "n_loci", "cu", "key")
    lines = ["\t".join(fields)]
    for c in sorted(recs, key=lambda c: c["rep"]):
        row = {"phase": SP.PHASE, "rep": c["rep"], "arm": SP.ARM, "n": SP.N_TAXA, "k": 0,
               "trees": SP.TREES, "n_loci": c["n_loci"], "cu": c["cu"], "key": c["key"]}
        lines.append("\t".join(str(row[f]) for f in fields))
    SP.write_atomic(SP.CELLS, "\n".join(lines) + "\n")
    SP.write_atomic(SP.STAGE, json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(SP.PKG), "phase": SP.PHASE,
        "arm": SP.ARM, "sizes": {t: {"m": m, "tau": tau} for t, m, tau in SP.SIZES},
        "missing_from_dataset": missing, "true_trees_present": n_true,
        "cells": sorted(recs, key=lambda c: c["rep"]),
    }, indent=1) + "\n")
    cus = {}
    for c in recs:
        cus[c["cu"]] = cus.get(c["cu"], 0) + 1
    print(f"[stage] OK: {len(recs)} cells (cu {dict(sorted(cus.items()))}), every gate passed "
          f"| {time.perf_counter() - t0:.0f} s -> {SP.CELLS.name}, {SP.STAGE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
