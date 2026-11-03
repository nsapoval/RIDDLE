#!/usr/bin/env python3
"""Stage ``netcs_bench_corpus`` for the benchmark, and gate every staged file.

Writes under this directory; the corpus itself is only read:

    inputs/L<l>_n<n>/<id>/true_net.nwk           (OUT:1,<normalised truth>:0);
    inputs/L<l>_n<n>/<id>/true_net.phylonet.nwk  the truth PhyloNet compares against (common.py)
    inputs/L<l>_n<n>/<id>/est_ils<x>.nwk         the corpus gene trees with OUT inserted where the
                                                 true-root oracle says, lengths kept, supports dropped
    tob/<key>.tqmc.nwk                           TOB-QMC default exactly as shipped: the resolvers' input
    tob/<key>.<kind>.rooted.nwk                  each tree of blobs rooted by the same oracle, OUT
                                                 kept: the tree-of-blobs rows
    CELLS.tsv, shards/CELLS_XX.tsv, stage.json   written only when every gate passes on the whole
                                                 corpus -- never by a --only / --limit run

``CELLS.tsv`` names every condition with phase ``netcs_L<l>_n<n>``, rep = the corpus network id and
arm ``netcs_ils<x>``, which ``opc.rl.camus`` resolves to the two ``inputs/`` files above -- checked,
not assumed.  ``shards/`` splits it twelve ways, balanced on n_taxa^2, for parallel GNN and greedy
runs.

Gates -- exit 1, and no table is written:

    truth        the four ILS files share one topology; the leaf set is the corpus's; staged-and-
                 pruned == direct parse by label cluster and node count; n_retics == distinct #H
                 tags; level == the directory's level; clusters == those an independent parser
                 (outgroup.parse_newick) reads off the corpus string -- which is what vouches for
                 the unary-node wrap on the two double-donor truths; and, as a control, wrapping a
                 bare tag on every *other* reticulate truth leaves clusters and k unchanged
    names        opc.rl.camus resolves every (phase, id, arm) to exactly the staged files, rooted
                 on OUT
    gene trees   every locus has lengths and exactly the truth's taxa; rooting the staged line on OUT
                 with reroot_on_outgroup -- prepare_gene_trees' own call -- gives a root whose two
                 sides are exactly the oracle's pick; and prepare_gene_trees itself runs clean on
                 every staged file and counts every locus
    trees of     all present kinds parse with the truth's taxa (TINNIK required at n <= 100 and
    blobs        absent at n = 200), root on the oracle and prune

Reported, not gated: the oracle's cost per condition (fraction of loci at an exact displayed root,
median and max cost, ties), each tree of blobs' blob count and root cost, and its clusters against
the truth's once rooted -- the true tree of blobs included, which need not be all-true when the root
lies inside a blob.

Usage (``gnn_cpu_v1/.venv/bin/python``):
    python netcs_eval/stage.py --jobs 12                    # the corpus, ~5 min
    python netcs_eval/stage.py --only L2_n50 --limit 2      # a smoke: gates only, no tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import statistics
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common as C

from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

N_SHARDS = 12
MAX_PROBLEMS = 5
TMP = C.RUNS / "stage_tmp"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_if_changed(path: Path, text: str) -> None:
    """Atomic, and a no-op when the file already holds ``text``, so a re-run rewrites nothing."""
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def rel(path: Path) -> str:
    return str(Path(path).relative_to(C.HERE))


# ------------------------------------------------------------------------------------------ truth

def stage_truth(level: int, n: int, netid: int) -> tuple[list[str], dict]:
    phase = C.phase_name(level, n)
    texts = {ils: C.corpus_network(level, n, netid, ils).read_text() for ils in C.ILS}
    bad = []
    if len({C.topology(t) for t in texts.values()}) != 1:
        bad.append("truth: the four ILS files differ in topology")
    text = texts["1.0"]
    norm = C.normalise_truth(text)
    direct = N.from_enewick(norm["body"] + ";")
    dst = C.cell_dir(phase, netid) / "true_net.nwk"
    write_if_changed(dst, f"({C.OUT}:1,{norm['body']}:0);\n")
    write_if_changed(C.phylonet_truth_path(phase, netid), norm["phylonet"] + "\n")
    pruned = _prune_outgroup(N.from_enewick(dst.read_text()))
    taxa = frozenset(direct.labels)
    index = {lab: i for i, lab in enumerate(direct.labels)}
    clusters = CN.label_clusters(direct)
    n_tags = len(set(C.HTAG.findall(text)))

    if C.leaf_labels(text) != taxa:
        bad.append(f"truth: {len(C.leaf_labels(text) ^ taxa)} labels differ from the corpus string")
    if CN.label_clusters(pruned) != clusters:
        bad.append("truth: the pruned staged network and the direct parse differ in label clusters")
    if pruned.n_nodes != direct.n_nodes:
        bad.append(f"truth: {pruned.n_nodes} nodes staged-and-pruned, {direct.n_nodes} parsed "
                   f"directly -- the pruning left a node of the graft behind")
    if direct.n_retics != n_tags:
        bad.append(f"truth: n_retics {direct.n_retics}, distinct #H tags {n_tags}")
    got_level = N.level(direct)
    if got_level != level:
        bad.append(f"truth: level {got_level} in a LEVEL_{level} directory")
    independent = C.string_clusters(norm["corpus_body"] + ";", index)
    if independent != clusters:
        bad.append(f"truth: from_enewick and parse_newick disagree on {len(independent ^ clusters)} "
                   f"clusters (wrapped: {norm['wrapped']})")
    control = None
    tags = C.bare_tags(norm["body"])
    if not norm["wrapped"] and tags:
        wrapped = N.from_enewick(C.wrap_tags(norm["body"], tags[:1]) + ";")
        control = C.HTAG.match(tags[0].group(0)).group(0)
        if CN.label_clusters(wrapped) != clusters or wrapped.n_retics != direct.n_retics:
            bad.append(f"truth: the wrap control changed clusters or k (tag {control})")
    targets = C.root_targets(direct)
    if not targets:
        bad.append("truth: no root bipartition")
    return bad, {"direct": direct, "pruned": pruned, "index": index, "taxa": taxa,
                 "clusters": clusters, "targets": targets, "file": dst, "rec": {
                     "phase": phase, "rep": netid, "level": level, "n": n,
                     "n_taxa": len(taxa), "k": direct.n_retics,
                     "shortcut_retics": CAMUS.shortcut_retics(direct),
                     "unary_root_removed": norm["unary_removed"], "wrapped_tags": norm["wrapped"],
                     "wrap_control_tag": control, "n_root_targets": len(targets),
                     "n_nodes": direct.n_nodes}}


# ------------------------------------------------------------------------------------- gene trees

def stage_trees(t: dict, level: int, n: int, netid: int, ils: str) -> tuple[list[str], dict]:
    phase, arm = C.phase_name(level, n), C.arm_name(ils)
    index, targets = t["index"], t["targets"]
    n_taxa = len(index)
    full = (1 << n_taxa) - 1
    src = C.corpus_trees(level, n, netid, ils)
    bad, lines, costs, ties = [], [], [], 0
    with open(src) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            locus = len(lines)
            if ":" not in raw:
                bad.append(f"trees: locus {locus} has no branch lengths")
            try:
                r = C.insert_out_at_truth(raw, index, targets, keep_lengths=True)
            except KeyError as exc:
                bad.append(f"trees: locus {locus} carries a taxon the truth does not: {exc}")
                break
            if r["root_mask"] != full or r["n_leaves"] != n_taxa:
                bad.append(f"trees: locus {locus} carries {r['n_leaves']} leaves, not exactly the "
                           f"truth's {n_taxa} taxa")
            # The independent check: prepare_gene_trees' own rooting call recovers the oracle's root.
            rooted = OG.reroot_on_outgroup(r["newick"], keep_lengths=True)
            p, _e, lb, ch = OG.parse_newick(rooted)
            masks = OG.node_masks(p, lb, ch, index)
            if len(ch[0]) != 2 or {masks[c] for c in ch[0]} != {r["side"], full ^ r["side"]}:
                bad.append(f"trees: locus {locus} does not root where the oracle put OUT")
            lines.append(r["newick"])
            costs.append(r["cost"])
            ties += r["ties"] > 1
            if len(bad) >= MAX_PROBLEMS:
                bad.append("trees: stopped after 5 problems")
                break
    dst = C.trees_path(phase, netid, arm)
    if not bad:
        write_if_changed(dst, "\n".join(lines) + "\n")
    cell = CAMUS.CamusCell(phase, netid, arm, C.INPUTS)
    if cell.base_path != t["file"] or cell.trees_path != dst or cell.rooting != "outgroup":
        bad.append(f"names: {phase}/{netid}/{arm} resolves to {cell.base_path}, {cell.trees_path}, "
                   f"rooting {cell.rooting!r}")
    info = {}
    if not bad:
        TMP.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=TMP))
        try:
            info = CAMUS.prepare_gene_trees(cell, work / "t.nwk.gz", list(t["pruned"].labels),
                                            max_loci=1000)
        except ValueError as exc:
            bad.append(f"trees: prepare_gene_trees refused the staged file: {exc}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if info and info["n_loci"] != len(lines):
            bad.append(f"trees: prepare_gene_trees read {info['n_loci']} of {len(lines)} loci")
    stats = {
        "n_loci": len(lines),
        "root_exact_frac": sum(c == 0 for c in costs) / max(len(costs), 1),
        "root_cost_median": statistics.median(costs) if costs else None,
        "root_cost_mean": statistics.fmean(costs) if costs else None,
        "root_cost_max": max(costs, default=None), "root_ties_frac": ties / max(len(costs), 1),
        **{k: info.get(k) for k in ("n_ultrametric", "n_polytomy", "min_branch", "max_branch")},
    }
    return bad, {"file": dst, "source": src, **stats}


# ---------------------------------------------------------------------------------- trees of blobs

def stage_tobs(t: dict, level: int, n: int, netid: int, ils: str, key: str) -> tuple[list[str], dict]:
    bad, out = [], {}
    for kind in C.TOB_KINDS:
        src = C.corpus_tob(level, n, netid, ils, kind)
        if kind == "tinnik" and n == 200:
            if src.exists():
                bad.append(f"TOB: a TINNIK tree at n = 200, where the corpus was read to have none")
            continue
        if not src.exists():
            bad.append(f"TOB: no {kind} tree of blobs at {src.name}")
            continue
        raw = src.read_text().strip()
        if C.leaf_labels(raw) != t["taxa"]:
            bad.append(f"TOB {kind}: {len(C.leaf_labels(raw) ^ t['taxa'])} labels differ from the truth")
            continue
        r = C.insert_out_at_truth(raw, t["index"], t["targets"], keep_lengths=False)
        rooted = CN.root_tob_on_outgroup(r["newick"])
        net = N.from_enewick(rooted)
        if set(net.labels) != set(t["taxa"]) | {C.OUT}:
            bad.append(f"TOB {kind}: the rooted tree has the wrong taxon set")
            continue
        try:
            _prune_outgroup(net)
        except ValueError as exc:
            bad.append(f"TOB {kind}: {exc}")
            continue
        got = CN.restrict_clusters(net, CN.ingroup_of(net))
        _p, _l, _lab, children = OG.parse_newick(rooted)
        degrees = sorted((len(c) for c in children if len(c) > 2), reverse=True)
        dst = C.tob_rooted_path(key, kind)
        write_if_changed(dst, rooted + "\n")
        out[kind] = {"file": rel(dst), "source": str(src), "n_blobs": len(degrees),
                     "blob_degrees": degrees, "root_cost": r["cost"], "root_ties": r["ties"],
                     "fp_vs_truth": len(got - t["clusters"]), "fn_vs_truth": len(t["clusters"] - got)}
        if kind == C.RESOLVER_TOB:
            write_if_changed(C.tob_input_path(key), raw + "\n")
            out[kind]["input_file"] = rel(C.tob_input_path(key))
    return bad, out


# --------------------------------------------------------------------------------------- a network

def stage_network(task: dict) -> dict:
    """One truth and its four conditions.  Top-level so a spawned worker can unpickle it."""
    level, n, netid = task["level"], task["n"], task["netid"]
    t0 = time.perf_counter()
    where = f"L{level}_n{n}/{netid}"
    try:
        bad, t = stage_truth(level, n, netid)
    except Exception as exc:                          # a gate failure, not a crash
        return {"where": where, "failures": [f"truth: {type(exc).__name__}: {exc}"], "cells": []}
    failures = [f"{where}: {b}" for b in bad]
    cells = []
    for ils in C.ILS:
        key = C.key_of(level, n, netid, ils)
        try:
            tbad, trees = stage_trees(t, level, n, netid, ils)
            obad, tobs = stage_tobs(t, level, n, netid, ils, key)
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        failures += [f"{key}: {b}" for b in tbad + obad]
        if tbad or obad:
            continue
        files = {"true_net.nwk": t["file"], "true_net.phylonet.nwk":
                 C.phylonet_truth_path(t["rec"]["phase"], netid), "trees": trees["file"]}
        cells.append({
            "key": key, **{k: t["rec"][k] for k in ("phase", "rep", "level", "n", "n_taxa", "k",
                                                    "shortcut_retics")},
            "arm": C.arm_name(ils), "ils": float(ils),
            **{k: v for k, v in trees.items() if k not in ("file", "source")},
            "tob": tobs, "files": {name: rel(p) for name, p in files.items()},
            "sha256": {**{name: sha256(p) for name, p in files.items()},
                       "source_trees": sha256(trees["source"])},
        })
    return {"where": where, "failures": failures, "network": t["rec"], "cells": cells,
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------- main

def write_tables(cells: list[dict], networks: list[dict], t_run: float) -> None:
    fields = ("phase", "rep", "arm", "n", "k", "trees", "n_loci")
    cells = sorted(cells, key=lambda c: (c["phase"], c["rep"], c["arm"]))

    def row(c):
        return {"phase": c["phase"], "rep": c["rep"], "arm": c["arm"], "n": c["n_taxa"], "k": c["k"],
                "trees": C.trees_name(C.ARMS[c["arm"]]), "n_loci": c["n_loci"]}

    def write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as fh:
            fh.write("\t".join(fields) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[f]) for f in fields) + "\n")
        tmp.replace(path)

    write(C.CELLS, [row(c) for c in cells])
    # Twelve shards, balanced on n_taxa^2 -- the candidate set, and so a cell's cost, grows with it.
    loads, shards = [0] * N_SHARDS, [[] for _ in range(N_SHARDS)]
    for c in sorted(cells, key=lambda c: (-c["n_taxa"] ** 2, c["phase"], c["rep"], c["arm"])):
        i = loads.index(min(loads))
        shards[i].append(c)
        loads[i] += c["n_taxa"] ** 2
    for old in C.SHARDS.glob("CELLS_*.tsv"):
        old.unlink()
    for i, s in enumerate(shards):
        write(C.SHARDS / f"CELLS_{i:02d}.tsv",
              [row(c) for c in sorted(s, key=lambda c: (c["phase"], c["rep"], c["arm"]))])
    listing = hashlib.sha256()
    for p in sorted(C.PKG.rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            listing.update(f"{p.relative_to(C.PKG)}\t{p.stat().st_size}\n".encode())
    C.STAGE.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "dataset_listing_sha256": listing.hexdigest(), "seconds": round(time.perf_counter() - t_run),
        "gene_trees": C.GENE_TREES, "rooting": "true-root oracle (common.py)",
        "resolver_tob": C.RESOLVER_TOB, "shard_loads": loads,
        "networks": sorted(networks, key=lambda r: (r["phase"], r["rep"])), "cells": cells,
    }, indent=1) + "\n")


def summary(cells: list[dict]) -> None:
    med = lambda xs: statistics.median(xs) if xs else float("nan")   # noqa: E731
    print(f"  {'phase':14s} {'arm':13s} {'cells':>5s} {'k':>3s} {'loci':>5s} {'exact':>6s} "
          f"{'cost':>5s} {'max':>4s} {'blobs tqmc/true':>15s} {'trueTOB fp':>10s} {'tqmc cost':>9s}")
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        groups.setdefault((c["phase"], c["arm"]), []).append(c)
    for (phase, arm), g in sorted(groups.items()):
        print(f"  {phase:14s} {arm:13s} {len(g):5d} {med([c['k'] for c in g]):3.0f} "
              f"{med([c['n_loci'] for c in g]):5.0f} {med([c['root_exact_frac'] for c in g]):6.2f} "
              f"{med([c['root_cost_mean'] for c in g]):5.1f} {max(c['root_cost_max'] for c in g):4d} "
              f"{med([c['tob']['tqmc']['n_blobs'] for c in g]):7.1f}/"
              f"{med([c['tob']['true']['n_blobs'] for c in g]):<7.1f}"
              f"{med([c['tob']['true']['fp_vs_truth'] for c in g]):10.1f}"
              f"{med([c['tob']['tqmc']['root_cost'] for c in g]):9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1, help="networks staged concurrently")
    ap.add_argument("--only", default="", help="comma list of L<l>_n<n>; implies no tables")
    ap.add_argument("--limit", type=int, default=0, help="networks per (level, n); implies no tables")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    partial = bool(only or args.limit)
    tasks = []
    for level in C.LEVELS:
        for n in C.SIZES:
            if only and C.phase_dir(level, n) not in only:
                continue
            ids = C.network_ids(level, n)
            tasks += [{"level": level, "n": n, "netid": i} for i in (ids[:args.limit] if args.limit else ids)]
    t_run = time.perf_counter()
    print(f"[stage] {len(tasks)} networks x {len(C.ILS)} ILS: {C.PKG} -> {C.HERE} | jobs {args.jobs}"
          + (" | partial: no tables" if partial else ""), flush=True)

    results = []

    def report(i: int, res: dict) -> None:
        results.append(res)
        state = "FAILED: " + res["failures"][0] if res["failures"] else (
            f"k={res['network']['k']} taxa={res['network']['n_taxa']} "
            f"targets={res['network']['n_root_targets']} exact "
            + "/".join(f"{c['root_exact_frac']:.2f}" for c in res["cells"])
            + f" | {res['seconds']:.1f}s")
        if not res["failures"] or i % 25 == 0 or i == len(tasks):
            print(f"[stage] {i:3d}/{len(tasks)} {res['where']:16s} {state}", flush=True)
        elif res["failures"]:
            print(f"[stage] {i:3d}/{len(tasks)} {res['where']:16s} {state}", flush=True)

    if args.jobs <= 1:
        for i, task in enumerate(tasks, 1):
            report(i, stage_network(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(stage_network, task) for task in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                report(i, fut.result())
    shutil.rmtree(TMP, ignore_errors=True)

    failures = [f for r in results for f in r["failures"]]
    cells = [c for r in results for c in r["cells"]]
    networks = [r["network"] for r in results if "network" in r]
    if cells:
        summary(cells)
    special = [(r["phase"], r["rep"], {k: r[k] for k in ("wrapped_tags", "shortcut_retics",
                                                           "unary_root_removed") if r[k]})
               for r in networks if r["wrapped_tags"] or r["shortcut_retics"]]
    print(f"[stage] truths needing care: {special or 'none'}")
    if failures:
        print(f"[stage] FAILED ({len(failures)} problems); no table written:")
        print("\n".join(f"  {f}" for f in failures[:60]))
        return 1
    if partial:
        print(f"[stage] OK on {len(cells)} conditions, every gate passed | partial run: "
              f"CELLS.tsv, shards/ and stage.json not written | {time.perf_counter() - t_run:.0f} s")
        return 0
    if len(cells) != len(C.LEVELS) * len(C.SIZES) * 50 * len(C.ILS):
        print(f"[stage] FAILED: {len(cells)} conditions staged, 1,200 expected; no table written")
        return 1
    write_tables(cells, networks, t_run)
    print(f"[stage] OK: {len(cells)} conditions on {len(networks)} truths, every gate passed | "
          f"{time.perf_counter() - t_run:.0f} s -> {C.CELLS.name}, shards/, {C.STAGE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
