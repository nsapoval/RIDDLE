#!/usr/bin/env python3
"""Stage the missing-taxa conditions: ``staged/<fam>/<cond>/<phase-dir>/<RR>/`` in camus-data's layout.

For every (family, condition) of ``inputs/<fam>/MANIFEST.tsv`` whose files are present -- ``control``
excepted, which is the published benchmark itself -- each cell gets:

* ``<arm file>`` (``g_500.nwk``, ``iqtree_500.nwk``, ``est_bp1000.nwk``, ...): a **relative symlink**
  to the MANIFEST's ``path`` under ``inputs/``.  Nothing is copied: one condition is ~0.5 GB of
  trees and nine are coming.  The name is the arm's own (``opc.rl.camus.ARMS``), which is the one
  rename l2reg/l3reg need (``est_bp1000.raw.nwk`` -> ``est_bp1000.nwk``).
* ``true_net.nwk`` (and netcs's ``true_net.phylonet.nwk``): the **parent benchmark's** truth, copied
  byte for byte -- the object every published row was scored against.
* ``missing.tsv``: a symlink to the per-locus record of what was deleted -- on netcs, whose four ILS
  arms share a cell directory, one per arm, ``<arm stem>.missing.tsv`` (:func:`missing_path`).

plus ``staged/<fam>/<cond>/CELLS.tsv`` (the parent's CELLS rows for this condition's cells, so
``run_gnn_cpu.py --inputs staged/<fam>/<cond> --cells .../CELLS.tsv`` resolves them the usual way)
and ``record.json`` (when, what, every gate's numbers).  Re-running is idempotent; a condition synced
later is staged by running this again, and every downstream driver finds it by its ``CELLS.tsv``.

Gates (exit 1 on any failure, and when nothing is staged at all)
-----
T1  the MANIFEST ``path`` hashes to ``sha256_staged``, with ``n_loci`` non-blank lines; the parent's
    source hashes to ``sha256_source`` -- camus: the parent cell's tree file itself; l2reg/l3reg:
    the package's ``estimated_bp1000/<key>.raw.nwk.gz``, which decompresses to the parent cell's
    ``est_bp1000.nwk`` byte for byte (``l2reg_eval/stage.py``'s rule); netcs: the corpus's own
    unrooted ``.estimated_gene_trees``, which ``netcs_eval/stage.json`` records as the source of the
    parent cell's ``est_ils<x>.nwk`` (OUT inserted by the true-root oracle), both by SHA-256.
T1b the condition is a **deletion** from that file and nothing else: on ``--prune-sample`` trees per
    cell (the first, the last and a seeded draw; 0 = every tree), the parent's tree pruned to the
    staged tree's leaves (``infer_gnn_cpu.prune_to``) has the staged tree's rooted clusters, both
    rooted on OUT.  **netcs** (:data:`T1B_INGROUP`) is held to its **ingroup** only -- OUT deleted,
    the same unrooted splits -- because the Linux box placed OUT anew on the pruned trees: on 4.7% of
    them (2-10% by phase and ILS; mostly among the oracle's tied root edges) OUT sits on another edge
    than pruning the parent gives (README, NetCS).  Those trees are counted, not failed: ``out_moved``
    in ``t1b.tsv`` and ``record.json``.
T2  **every tree's** leaf set is the truth's + OUT, minus that locus's ``removed`` in
    ``missing.tsv`` -- whose ``n_missing`` agrees, whose loci are 0..n_loci-1 in order, and which
    reproduces the MANIFEST's ``n_modified`` and ``mean_missing`` (mean n_missing / n).
T3  the tree-of-blobs sidecars: the JSON's ``cell`` / ``cond`` / ``input`` are the row's,
    ``sha256_input`` is ``sha256_staged``, ``n_trees`` is ``n_loci``; the raw TOB's leaves are the
    truth's + OUT; ``.tob`` and ``.pvalues.tob`` exist.
T4  the staged truth is the parent's byte for byte and parses (``camus.truth_of``) with the TOB's
    taxa; camus's own ``true_net.nwk`` in ``inputs/`` is the parent's too, and l2reg/l3reg's
    ``true_tob.nwk`` has the TOB JSON's ``true_tob`` clusters.
T5  read-back: ``camus.prepare_gene_trees(partial=PARTIAL)`` -- what the GNN driver runs first --
    reads one staged cell per (condition, phase) with ``n_loci`` trees and the MANIFEST's mean
    missing, **and the strict call refuses the same cell** (the fixture is partial: G4's lesson).
T6  ``subsample_eval/NO_REFERENCE.tsv`` does not exist (``run_gnn.sh`` points ``--reference`` at it).

``--scan`` also records, per cell, the candidate set the partial count gives at tau = 2 beside the
published one (``|C|`` from the landed GNN cell), in ``staged/<fam>/<cond>/scan.tsv``: what sizes
the grid, and which cells the tail smokes take.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python taxmiss_eval/stage.py                          # every family, every condition present
    python taxmiss_eval/stage.py --families camus --conds p0.90_c24
    python taxmiss_eval/stage.py --check                  # every gate; writes nothing
    python taxmiss_eval/stage.py --scan --jobs 6          # + scan.tsv
"""

from __future__ import annotations

import tm_common as TM  # first: BLAS pin, gnn_cpu_v1/pylib at sys.path[0]

import argparse  # noqa: E402
import functools  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from opc.clusters import extract  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

if str(TM.SC.GNN) not in sys.path:      # infer_gnn_cpu, for T1b
    sys.path.insert(1, str(TM.SC.GNN))
TRUTH_FILES = ("true_net.nwk", "true_net.phylonet.nwk")
#: Families whose condition trees carry an OUT placed anew after the deletion: T1b holds their
#: ingroup to the parent's and counts the trees where OUT moved (module docstring).
T1B_INGROUP = frozenset({"netcs"})
#: ``infer_gnn_cpu.py``'s leaf regex: a label right after ``(`` or ``,`` -- internal-node support
#: values follow ``)`` and are not matched.
_LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")
TAU = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_lines(path: Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def leaves_of(newick: str) -> set[str]:
    return set(_LEAF.findall(newick))


def label_clusters(newick: str) -> set[frozenset]:
    """Every non-trivial internal node's leaf-label set, by a real parser (no taxon index)."""
    parent, _l, lab, children = OG.parse_newick(newick.strip())
    below: dict[int, frozenset] = {}
    for i in range(len(parent) - 1, -1, -1):
        below[i] = (frozenset(x for c in children[i] for x in below[c]) if children[i]
                    else frozenset([lab[i]]))
    allx = below[0]
    return {s for i, s in below.items() if children[i] and 1 < len(s) < len(allx)}


def parent_cell(fam: str, row: dict) -> CAMUS.CamusCell:
    return CAMUS.CamusCell(row["phase"], row["rep"], row["arm"], TM.FAMILIES[fam].parent_inputs)


def staged_cell(fam: str, cond: str, row: dict) -> CAMUS.CamusCell:
    return CAMUS.CamusCell(row["phase"], row["rep"], row["arm"], TM.staged_dir(fam, cond))


def source_check(fam: str, row: dict, pc: CAMUS.CamusCell) -> list[str]:
    """T1's link to the parent: what ``sha256_source`` names, and that it is the parent's file."""
    if fam in ("l2reg", "l3reg"):
        gz = TM.REPO / f"{fam}_n200_v1" / "estimated_bp1000" / f"{row['key']}.raw.nwk.gz"
        if not gz.exists():
            return [f"no {gz}"]
        bad = [] if sha256_file(gz) == row["sha256_source"] else [f"{gz.name} does not hash to sha256_source"]
        with gzip.open(gz, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != sha256_file(pc.trees_path):
                bad.append(f"{gz.name} does not decompress to the parent's {pc.trees_path}")
        return bad
    if fam == "netcs":
        rec = netcs_stage().get((row["phase"], row["rep"], row["arm"]))
        if rec is None:
            return [f"no netcs_eval/stage.json cell for {row['cell']}"]
        bad = []
        if rec["sha256"]["source_trees"] != row["sha256_source"]:
            bad.append("netcs_eval/stage.json's source_trees is not sha256_source")
        if rec["sha256"]["trees"] != sha256_file(pc.trees_path):
            bad.append(f"the parent's {pc.trees_path} is not the file netcs_eval/stage.json records")
        corpus = (TM.REPO / "netcs_bench_corpus" / f"LEVEL_{rec['level']}" / f"TAXA_{rec['n']}"
                  / f"network_{row['rep']}" / f"network_{row['rep']}_ils_{rec['ils']}.estimated_gene_trees")
        if not corpus.exists():
            bad.append(f"no {corpus}")
        elif sha256_file(corpus) != row["sha256_source"]:
            bad.append(f"{corpus.name} does not hash to sha256_source")
        return bad
    return ([] if sha256_file(pc.trees_path) == row["sha256_source"]
            else [f"the parent's {pc.trees_path} does not hash to sha256_source"])


@functools.cache
def netcs_stage() -> dict[tuple[str, int, str], dict]:
    """``netcs_eval/stage.json``'s cell records by (phase, rep, arm): the parent's file hashes."""
    rec = json.loads((TM.REPO / "netcs_eval" / "stage.json").read_text())
    return {(c["phase"], c["rep"], c["arm"]): c for c in rec["cells"]}


def missing_path(src: Path) -> Path:
    """The per-locus deletion record of the trees file ``src``: ``<stem>.missing.tsv`` beside it where
    one exists (netcs: four ILS arms share a cell directory), else the directory's ``missing.tsv``."""
    own = src.with_name(src.name.removesuffix(".nwk") + ".missing.tsv")
    return own if own.exists() else src.parent / "missing.tsv"


def rooted_clusters(newick: str) -> set[frozenset]:
    """Rooted on OUT, OUT deleted: the clusters an estimated tree contributes."""
    return label_clusters(OG.reroot_on_outgroup(newick, keep_lengths=False))


def ingroup_splits(newick: str) -> set[frozenset]:
    """The ingroup's unrooted splits, OUT deleted: one canonical side per split."""
    ing = frozenset(leaves_of(newick)) - {TM.OUT}
    out = set()
    for c in label_clusters(newick):
        s = c - {TM.OUT}
        t = ing - s
        if len(s) > 1 and len(t) > 1:
            out.add(min(s, t, key=sorted))
    return out


def prune_check(parent_lines: list[str], lines: list[str], sample: int, seed: str,
                ingroup: bool = False) -> tuple[list[int], list[int], int]:
    """T1b: the loci where pruning the parent's tree does not give the staged one, the loci where
    only OUT's edge differs (counted, not failed, with ``ingroup``), and how many trees were checked."""
    import infer_gnn_cpu as IG          # lazily: it imports torch
    n = len(lines)
    if sample and sample < n:
        pick = sorted({0, n - 1, *random.Random(seed).sample(range(n), sample - 2)})
    else:
        pick = range(n)
    bad, moved = [], []
    for i in pick:
        keep = leaves_of(lines[i])
        pruned = IG.prune_to(parent_lines[i], keep)
        if rooted_clusters(pruned) == rooted_clusters(lines[i]):
            continue
        if ingroup and ingroup_splits(pruned) == ingroup_splits(lines[i]):
            moved.append(i)
        else:
            bad.append(i)
    return bad, moved, len(pick)


def missing_rows(path: Path) -> list[dict]:
    rows = TM.read_tsv(path)
    for r in rows:
        r["locus"], r["n_missing"] = int(r["locus"]), int(r["n_missing"])
        r["removed"] = frozenset(x for x in r["removed"].split(",") if x)
    return rows


# --------------------------------------------------------------------------------------- one cell

def stage_cell(fam: str, cond: str, row: dict, check: bool, problems: list[str],
               prune_sample: int) -> dict:
    where = f"{fam} {cond} {row['cell']}"
    src = TM.INPUTS / row["path"]
    here = src.parent
    pc = parent_cell(fam, row)
    sc = staged_cell(fam, cond, row)
    stats = {"cell": row["cell"]}

    # T1
    if not src.exists():
        problems.append(f"T1 {where}: no {src}")
        return stats
    if sha256_file(src) != row["sha256_staged"]:
        problems.append(f"T1 {where}: {src.name} is not the file the MANIFEST records")
    problems += [f"T1 {where}: {b}" for b in source_check(fam, row, pc)]
    lines = tree_lines(src)
    if len(lines) != row["n_loci"]:
        problems.append(f"T1 {where}: {len(lines)} trees, the MANIFEST says {row['n_loci']}")
    parent_lines = tree_lines(pc.trees_path)
    if len(parent_lines) < len(lines):
        problems.append(f"T1b {where}: the parent has {len(parent_lines)} trees, fewer than {len(lines)}")
    else:
        t1b, moved, checked = prune_check(parent_lines, lines, prune_sample, row["cell"],
                                          ingroup=fam in T1B_INGROUP)
        if t1b:
            problems.append(f"T1b {where}: {len(t1b)} trees are not the parent's pruned"
                            f"{' (ingroup)' if fam in T1B_INGROUP else ''}, e.g. loci {t1b[:5]}")
        if fam in T1B_INGROUP:
            stats.update(t1b_checked=checked, out_moved=len(moved))

    # T4 (the truth first: T2 and T3 need its taxa)
    truth_bytes = {f: (pc.cell_dir / f).read_bytes() for f in TRUTH_FILES if (pc.cell_dir / f).exists()}
    if "true_net.nwk" not in truth_bytes:
        problems.append(f"T4 {where}: the parent has no true_net.nwk at {pc.cell_dir}")
        return stats
    own = here / "true_net.nwk"
    if own.exists() and own.read_bytes() != truth_bytes["true_net.nwk"]:
        problems.append(f"T4 {where}: inputs' true_net.nwk differs from the parent's")
    truth = CAMUS.truth_of(pc)
    taxa = set(truth.labels)
    full = taxa | {TM.OUT}

    # T2
    miss = missing_rows(missing_path(src))
    if [r["locus"] for r in miss] != list(range(len(lines))):
        problems.append(f"T2 {where}: {missing_path(src).name} loci are not 0..{len(lines) - 1} in order")
    else:
        bad = []
        for i, (t, r) in enumerate(zip(lines, miss)):
            if r["n_missing"] != len(r["removed"]):
                bad.append(f"locus {i}: n_missing {r['n_missing']} but {len(r['removed'])} removed")
            elif leaves_of(t) != full - r["removed"] or not r["removed"] <= taxa:
                bad.append(f"locus {i}: leaf set is not truth + OUT - removed")
            if len(bad) >= 3:
                break
        if bad:
            problems.append(f"T2 {where}: {'; '.join(bad)}")
        n_mod = sum(1 for r in miss if r["n_missing"])
        mean_miss = statistics.fmean(r["n_missing"] for r in miss) / row["n"]
        if n_mod != row["n_modified"]:
            problems.append(f"T2 {where}: {n_mod} modified trees, the MANIFEST says {row['n_modified']}")
        if abs(mean_miss - float(row["mean_missing"])) > 1e-5:
            problems.append(f"T2 {where}: mean missing {mean_miss:.6f}, the MANIFEST says {row['mean_missing']}")
        stats.update(n_modified=n_mod, mean_missing=mean_miss,
                     max_missing=max(r["n_missing"] for r in miss),
                     min_taxa=min(len(taxa) - r["n_missing"] for r in miss))

    # T3
    tp = TM.tob_paths(fam, cond, row["cell"])
    if not all(p.exists() for p in tp.values()):
        problems.append(f"T3 {where}: missing TOB files {[k for k, p in tp.items() if not p.exists()]}")
    else:
        meta = json.loads(tp["json"].read_text())
        for k, want in (("cell", row["cell"]), ("cond", cond), ("input", row["path"]),
                        ("sha256_input", row["sha256_staged"]), ("n_trees", row["n_loci"])):
            if meta.get(k) != want:
                problems.append(f"T3 {where}: TOB json {k} = {meta.get(k)!r}, expected {want!r}")
        if leaves_of(tp["raw"].read_text()) != full:
            problems.append(f"T3 {where}: the raw TOB's leaves are not the truth's + OUT")
        true_tob = here / "true_tob.nwk"
        if true_tob.exists() and meta.get("true_tob") and \
                label_clusters(true_tob.read_text()) != label_clusters(meta["true_tob"]):
            problems.append(f"T4 {where}: true_tob.nwk is not the TOB json's true_tob")
        stats.update(tob_fn=meta.get("fn_raw"), tob_fp=meta.get("fp_raw"),
                     tob_true_edges=meta.get("n_true_edges"), tob_seconds=meta.get("seconds"))

    # write
    if not check:
        sc.cell_dir.mkdir(parents=True, exist_ok=True)
        link(sc.trees_path, src)
        link(sc.cell_dir / missing_path(src).name, missing_path(src))
        for f, data in truth_bytes.items():
            dst = sc.cell_dir / f
            if not dst.exists() or dst.read_bytes() != data:
                tmp = dst.with_name(dst.name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(dst)
    if sc.cell_dir.exists():
        for f, data in truth_bytes.items():
            if (sc.cell_dir / f).read_bytes() != data:
                problems.append(f"T4 {where}: staged {f} differs from the parent's")
        if sc.trees_path.resolve() != src.resolve():
            problems.append(f"T1 {where}: staged {sc.trees_path.name} does not point at {src}")
    return stats


def link(dst: Path, target: Path) -> None:
    """A relative symlink ``dst -> target``, replaced atomically when it points elsewhere."""
    rel = os.path.relpath(target, dst.parent)
    if dst.is_symlink() and os.readlink(dst) == rel:
        return
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(rel, tmp)
    tmp.replace(dst)


def read_back(fam: str, cond: str, row: dict, problems: list[str]) -> dict:
    """T5 on one staged cell: partial reads it, strict refuses it."""
    where = f"T5 {fam} {cond} {row['cell']}"
    sc = staged_cell(fam, cond, row)
    taxa = list(CAMUS.truth_of(sc).labels)
    work = Path(tempfile.mkdtemp(prefix="tm_stage_"))
    try:
        info = CAMUS.prepare_gene_trees(sc, work / "t.nwk.gz", taxa, max_loci=0, partial=TM.PARTIAL)
        if info["n_loci"] != row["n_loci"]:
            problems.append(f"{where}: read {info['n_loci']} trees, not {row['n_loci']}")
        if abs(info["mean_missing"] / row["n"] - float(row["mean_missing"])) > 1e-5:
            problems.append(f"{where}: mean missing {info['mean_missing'] / row['n']:.6f}, the "
                            f"MANIFEST says {row['mean_missing']}")
        strict_refused = False
        try:
            CAMUS.prepare_gene_trees(sc, work / "s.nwk.gz", taxa, max_loci=0)
        except ValueError:
            strict_refused = True
        if not strict_refused and row["n_modified"]:
            problems.append(f"{where}: the strict read accepted a cell with {row['n_modified']} partial trees")
        return {"cell": row["cell"], "n_loci": info["n_loci"], "n_partial": info["n_partial"],
                "min_taxa": info["min_taxa"], "strict_refused": strict_refused}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------------------------------ scan

def scan_cell(payload) -> dict:
    """|C| at tau = 2 under the partial count, against the published cell's."""
    fam, cond, row = payload
    sc = staged_cell(fam, cond, row)
    truth = CAMUS.truth_of(sc)
    taxa = list(truth.labels)
    ti = {t: i for i, t in enumerate(taxa)}
    work = Path(tempfile.mkdtemp(prefix="tm_scan_"))
    t0 = time.perf_counter()
    try:
        info = CAMUS.prepare_gene_trees(sc, work / "t.nwk.gz", taxa, max_loci=1000, partial=TM.PARTIAL)
        counter = extract.count_newick_file(work / "t.nwk.gz", ti, [1000], drop_partial_root=True)
        masks, support, _ = counter.finalize(min_support=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    n_tau = int((support[:, -1] >= TAU).sum()) if len(masks) else 0
    landed = TM.SC.DATASETS[fam].full.get(("gnn", f"relaxed_L{TM.FAMILIES[fam].level}"))
    f = landed / "cells" / f"{row['cell']}.json" if landed else None
    pub = json.loads(f.read_text())["n_clusters"] if f is not None and f.exists() else ""
    return {"cell": row["cell"], "phase": row["phase"], "rep": row["rep"], "arm": row["arm"],
            "n": row["n"], "k": int(truth.n_retics), "mean_missing": row["mean_missing"],
            "n_partial": info["n_partial"], "min_taxa": info["min_taxa"], "n_clusters": n_tau,
            "n_clusters_published": pub,
            "ratio": round(n_tau / pub, 3) if pub else "", "seconds": round(time.perf_counter() - t0, 2)}


# ------------------------------------------------------------------------------------------ main

def stage_condition(fam: str, cond: str, rows: list[dict], args, problems: list[str]) -> dict:
    F = TM.FAMILIES[fam]
    t0 = time.perf_counter()
    before = len(problems)
    stats = [stage_cell(fam, cond, r, args.check, problems, args.prune_sample) for r in rows]
    want = {(r["phase"], r["rep"], r["arm"]) for r in rows}
    cells = [c for c in TM.read_tsv(F.parent_cells) if (c["phase"], int(c["rep"]), c["arm"]) in want]
    if len(cells) != len(rows):
        problems.append(f"{fam} {cond}: {len(rows)} MANIFEST rows but {len(cells)} parent CELLS rows")
    if not args.check:
        TM.write_tsv(TM.cells_file(fam, cond), cells, list(cells[0]) if cells else [])
    # T5: one cell per phase
    seen, back = set(), []
    if TM.staged_dir(fam, cond).exists():
        for r in rows:
            if r["phase"] not in seen and staged_cell(fam, cond, r).trees_path.exists():
                seen.add(r["phase"])
                back.append(read_back(fam, cond, r, problems))
    scan = None
    if args.scan and not args.check:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            scan = list(ex.map(scan_cell, [(fam, cond, r) for r in rows]))
        TM.write_tsv(TM.staged_dir(fam, cond) / "scan.tsv", scan, list(scan[0]))
    num = lambda k: [s[k] for s in stats if s.get(k) is not None]  # noqa: E731
    rec = {"family": fam, "cond": cond, "cells": len(rows),
           "phases": sorted({r["phase"] for r in rows}), "p": rows[0]["p"], "cap": rows[0]["cap"],
           "M": sorted({r["M"] for r in rows}),
           "mean_missing": statistics.fmean(num("mean_missing")) if num("mean_missing") else None,
           "max_missing": max(num("max_missing"), default=None),
           "min_taxa": min(num("min_taxa"), default=None),
           "tob_fn_mean": statistics.fmean(num("tob_fn")) if num("tob_fn") else None,
           "tob_fp_mean": statistics.fmean(num("tob_fp")) if num("tob_fp") else None,
           "t1b_trees_per_cell": args.prune_sample or "all", "read_back": back, "problems": problems[before:],
           "seconds": round(time.perf_counter() - t0, 1)}
    if scan:
        ratios = [s["ratio"] for s in scan if s["ratio"] != ""]
        rec["scan"] = {"n_clusters_max": max(s["n_clusters"] for s in scan),
                       "ratio_median": statistics.median(ratios) if ratios else None,
                       "ratio_max": max(ratios) if ratios else None,
                       "largest": [s["cell"] for s in sorted(scan, key=lambda s: -s["n_clusters"])[:5]]}
    if fam in T1B_INGROUP:
        moved = [s for s in stats if "out_moved" in s]
        n_chk, n_mv = sum(s["t1b_checked"] for s in moved), sum(s["out_moved"] for s in moved)
        rec["t1b_mode"] = "ingroup"
        rec["out_moved"] = {"trees_checked": n_chk, "moved": n_mv, "frac": n_mv / n_chk if n_chk else None,
                            "cells_with_moves": sum(1 for s in moved if s["out_moved"])}
        if not args.check:
            TM.write_tsv(TM.staged_dir(fam, cond) / "t1b.tsv",
                         [{k: s[k] for k in ("cell", "t1b_checked", "out_moved")} for s in moved],
                         ["cell", "t1b_checked", "out_moved"])
    if not args.check:
        rec["built"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        TM.write_json(TM.staged_dir(fam, cond) / "record.json", rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(TM.FAMILIES))
    ap.add_argument("--conds", default="all", help="comma list, or all (every condition present)")
    ap.add_argument("--check", action="store_true", help="every gate; write nothing")
    ap.add_argument("--scan", action="store_true", help="also write scan.tsv (|C| under the partial count)")
    ap.add_argument("--jobs", type=int, default=6, help="--scan workers")
    ap.add_argument("--prune-sample", type=int, default=100,
                    help="T1b: trees per cell held to the parent's, pruned (0 = every tree)")
    args = ap.parse_args()

    problems: list[str] = []
    if TM.SC.NO_REFERENCE.exists():                                                     # T6
        problems.append(f"T6 {TM.SC.NO_REFERENCE} exists; it must not")
    n_staged = 0
    for fam in [f for f in args.families.split(",") if f]:
        if fam not in TM.FAMILIES:
            raise SystemExit(f"unknown family {fam!r}; known {list(TM.FAMILIES)}")
        if not TM.FAMILIES[fam].manifest.exists():
            print(f"[stage] {fam}: no {TM.FAMILIES[fam].manifest.relative_to(TM.REPO)} -- skipped", flush=True)
            continue
        rows = TM.manifest(fam)
        for cond in TM.conditions(fam):
            if args.conds != "all" and cond not in args.conds.split(","):
                continue
            crow = [r for r in rows if r["cond"] == cond]
            present = [r for r in crow if (TM.INPUTS / r["path"]).exists()]
            if not present:
                continue
            if len(present) != len(crow):
                problems.append(f"{fam} {cond}: only {len(present)} of {len(crow)} cells' inputs are present "
                                f"-- a sync in progress?  Not staged.")
                continue
            rec = stage_condition(fam, cond, present, args, problems)
            n_staged += 1
            sc = rec.get("scan")
            print(f"[stage] {fam} {cond}: {rec['cells']} cells, phases {','.join(rec['phases'])} | "
                  f"mean missing {rec['mean_missing']:.4f}, max {rec['max_missing']}, min taxa "
                  f"{rec['min_taxa']} | TOB fn/fp mean {rec['tob_fn_mean']:.2f}/{rec['tob_fp_mean']:.2f} | "
                  f"read back {len(rec['read_back'])} | "
                  + (f"|C| max {sc['n_clusters_max']}, x{sc['ratio_median']} median / "
                     f"x{sc['ratio_max']} max of published | " if sc else "")
                  + (f"T1b ingroup: OUT moved on {rec['out_moved']['moved']} of {rec['out_moved']['trees_checked']} "
                     f"trees checked | " if "out_moved" in rec else "")
                  + f"{len(rec['problems'])} problems, {rec['seconds']} s", flush=True)
    if not n_staged:
        problems.append("nothing staged: no condition has its inputs present")
    for p in problems[:50]:
        print(f"  {p}")
    if len(problems) > 50:
        print(f"  ... and {len(problems) - 50} more")
    print(f"[stage] {'PASS' if not problems else 'FAIL'}: T1-T6 over {n_staged} conditions"
          f"{' (--check)' if args.check else ''}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
