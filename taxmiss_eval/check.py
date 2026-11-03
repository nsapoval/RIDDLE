#!/usr/bin/env python3
"""The partial-tree path, held to code that did not write it.  Seconds to a few minutes; exit 1 on
any failure.

P1  the cluster count ``run_gnn_cpu --partial`` and ``score_cells`` use --
    ``extract.count_newick_file(drop_partial_root=True)`` -- equals ``infer_gnn_cpu.count_clusters``,
    a separate implementation of the same rule, mask for mask and support for support; and differs
    from the full-taxon count (the fixture has partial roots to drop, or P1 says nothing).
P2  end to end: ``infer_gnn_cpu.infer(partial=4, taxa = order = the truth's, outgroups=("OUT",),
    m=1000)`` -- the other driver, through its own scan / select / prepare -- gives the network and
    selection size ``run_gnn.sh`` wrote, on every checked cell that has a run_gnn cell JSON.
P3  the partial triple kernels agree: ``concord.block_atom_tensors(partial=True)`` equals the atom
    sums of ``concord.partial_triple_counts`` (dense), and both equal
    ``slow_atom_tensor(partial=True)`` on sampled atom triples; the full-taxon kernel exceeds them by
    exactly ``absent_c_counts`` (N3'(c)), which is non-zero.
P4  on the **published** full-taxon trees of the same cells, every partial counter is the published
    one bit for bit: extract (masks and support), dense (the whole n^3 tensor) and block (the atom
    tensors).  This is what licenses running them only on the taxmiss datasets.

The cells: the lowest rep of every staged (family, condition, phase, arm) -- rep 0 on camus / l2reg /
l3reg; netcs's reps are network ids, so its first network per ILS arm -- plus the largest ``scan.tsv``
cell per (family, condition) -- the tail.  ``--cells`` names others.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python taxmiss_eval/check.py
    python taxmiss_eval/check.py --only p1,p3,p4 --trees 100
"""

from __future__ import annotations

import tm_common as TM  # first: BLAS pin, gnn_cpu_v1/pylib at sys.path[0]

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

for _p in (TM.SC.GNN, TM.REPO / "nni_ensemble"):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))

from opc.clusters import extract  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402


def pick_cells(fams: list[str], extra: set[str]) -> list[tuple[str, str, dict]]:
    out, seen = [], set()
    for fam in fams:
        man = {(r["cond"], r["cell"]): r for r in TM.manifest(fam)}
        for cond in TM.staged_conds(fam):
            rows = [r for (c, _), r in man.items() if c == cond]
            # the lowest rep of every (phase, arm): rep 0 on camus / l2reg / l3reg, and on netcs, whose
            # reps are the corpus's network ids, the first network of each ILS arm
            first: dict[tuple[str, str], dict] = {}
            for r in rows:
                if (r["phase"], r["arm"]) not in first or r["rep"] < first[(r["phase"], r["arm"])]["rep"]:
                    first[(r["phase"], r["arm"])] = r
            want = {r["cell"] for r in first.values()} | (extra & {r["cell"] for r in rows})
            scan = TM.staged_dir(fam, cond) / "scan.tsv"
            if scan.exists():
                big = max(TM.read_tsv(scan), key=lambda s: int(s["n_clusters"]))
                want.add(big["cell"])
            for r in rows:
                if r["cell"] in want and (fam, cond, r["cell"]) not in seen:
                    seen.add((fam, cond, r["cell"]))
                    out.append((fam, cond, r))
    return out


def prepared(cell: CAMUS.CamusCell, taxa: list[str], work: Path, partial: int) -> Path:
    gz = work / f"{'p' if partial else 'f'}_{cell.phase}_{cell.rep}_{cell.arm}.nwk.gz"
    CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=1000, partial=partial)
    return gz


def table(counter) -> dict[int, int]:
    masks, support, _ = counter.finalize(min_support=1)
    return {int(m): int(s) for m, s in zip(masks, support[:, -1])}


def atoms_from(truth, n: int, rng: random.Random, k: int = 4) -> list[int] | None:
    """``k`` pairwise-disjoint parts of one true cluster: its children, the largest refined into its
    own children until there are ``k`` -- ``features.check_tc``'s construction, on the truth."""
    clusters = sorted({sum(1 << t for t in c) for c in NW.nontrivial_clusters(truth)}, key=int.bit_count)

    def children(X: int) -> list[int]:
        kids = [c for c in clusters if c != X and c & X == c]
        top = [c for c in kids if not any(c != d and c & d == c for d in kids)]
        parts: list[int] = []
        for c in sorted(top, key=lambda c: -c.bit_count()):       # a network's children can overlap
            if not any(c & d for d in parts):
                parts.append(c)
        covered = 0
        for c in parts:
            covered |= c
        return parts + [1 << t for t in range(n) if X >> t & 1 and not covered >> t & 1]

    big = [c for c in clusters if c.bit_count() >= 8]
    rng.shuffle(big)
    for P in big:
        parts = children(P)
        while len(parts) < k:
            top = max(parts, key=int.bit_count)
            sub = children(top)
            if top.bit_count() < 2 or len(sub) < 2:
                break
            parts = [x for x in parts if x != top] + sub
        if len(parts) >= 3:
            rng.shuffle(parts)
            return sorted(parts[:k], key=int.bit_count)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--only", default="p1,p2,p3,p4")
    ap.add_argument("--cells", default="", help="also these cells (names), in every staged condition")
    ap.add_argument("--trees", type=int, default=200, help="trees for the P3/P4 kernel checks")
    args = ap.parse_args()
    only = set(args.only.split(","))
    fams = TM.families_staged() if args.dataset == "all" else args.dataset.split(",")
    cells = pick_cells(fams, set(filter(None, args.cells.split(","))))
    if not cells:
        raise SystemExit("no cells: nothing staged")

    import concord as K                 # nni_ensemble's: imports ne_common, which is torch-free
    IG = None
    if "p1" in only or "p2" in only:
        import infer_gnn_cpu as IG       # noqa: F811 -- torch
    problems: list[str] = []
    stats: dict = {"cells": [f"{f} {c} {r['cell']}" for f, c, r in cells]}
    work = Path(tempfile.mkdtemp(prefix="tm_check_"))
    model = {}
    t0 = time.perf_counter()
    try:
        for fam, cond, row in cells:
            where = f"{fam} {cond} {row['cell']}"
            sc = CAMUS.CamusCell(row["phase"], row["rep"], row["arm"], TM.staged_dir(fam, cond))
            pc = CAMUS.CamusCell(row["phase"], row["rep"], row["arm"], TM.FAMILIES[fam].parent_inputs)
            truth = CAMUS.truth_of(sc)
            taxa = list(truth.labels)
            ti = {t: i for i, t in enumerate(taxa)}
            gz = prepared(sc, taxa, work, TM.PARTIAL)
            line = [where]

            if "p1" in only:
                mine = table(extract.count_newick_file(gz, ti, [1000], drop_partial_root=True))
                theirs = table(IG.count_clusters(gz, ti, 1000))
                naive = table(extract.count_newick_file(gz, ti, [1000]))
                if mine != theirs:
                    problems.append(f"P1 {where}: extract's partial count != infer_gnn_cpu.count_clusters "
                                    f"({len(set(mine.items()) ^ set(theirs.items()))} entries differ)")
                if naive == mine:
                    problems.append(f"P1 {where}: vacuous -- no partial root to drop")
                line.append(f"P1 |C1| {len(mine)} (naive {len(naive)})")

            if "p2" in only:
                f = TM.gnn_run(fam, cond) / "cells" / f"{row['cell']}.json"
                if f.exists():
                    rec = json.loads(f.read_text())
                    mr = next(m for m in rec["models"] if m["tag"] == "_l2mix_pair")
                    if "net" not in model:
                        model["net"] = IG.load_model(TM.SC.GNN / "checkpoints" / "_l2mix_pair" / "s0" / "best.pt",
                                                     log=open("/dev/null", "w"))
                    L = TM.FAMILIES[fam].level
                    got = IG.infer(sc.trees_path, model["net"], [int(truth.n_retics)], outgroups=(TM.OUT,),
                                   taxa=taxa, partial=TM.PARTIAL, m=1000, order=taxa, decoder="relaxed",
                                   max_level=L, log=open("/dev/null", "w"))[0]
                    if got["enewick"] != mr["enewick"] or got["n_selected"] != mr["n_selected"]:
                        problems.append(f"P2 {where}: infer_gnn_cpu's network differs from run_gnn_cpu's "
                                        f"(n_selected {got['n_selected']} vs {mr['n_selected']})")
                    line.append(f"P2 ok={got['enewick'] == mr['enewick']}")
                    stats["p2_cells"] = stats.get("p2_cells", 0) + 1
                else:
                    line.append("P2 no run_gnn cell")

            if "p3" in only or "p4" in only:
                rng = random.Random(row["cell"])
                atoms = atoms_from(truth, len(taxa), rng)
                if atoms is None:
                    line.append("P3/P4 no atom set")
                else:
                    trip = [(0, 1, 2), (0, 2, 1), (1, 2, 0)] + ([(0, 1, 3), (2, 3, 0)] if len(atoms) > 3 else [])
                    if "p3" in only:
                        dense = K.atom_tensor(K.partial_triple_counts(gz, ti, args.trees).count, atoms)
                        block = K.block_atom_tensors(gz, taxa, args.trees, [atoms], partial=True)["T"][0]
                        full = K.block_atom_tensors(gz, taxa, args.trees, [atoms])["T"][0]
                        slow = K.slow_atom_tensor(gz, taxa, args.trees, atoms, trip, partial=True)
                        excess = K.absent_c_counts(gz, taxa, args.trees, atoms, trip)
                        if not np.array_equal(dense, block):
                            problems.append(f"P3 {where}: block-partial != dense-partial atom tensor")
                        bad = [t for t in trip if int(round(block[t])) != slow[t]]
                        if bad:
                            problems.append(f"P3 {where}: kernel != slow reference on {bad}")
                        bad = [t for t in trip if int(round(full[t] - block[t])) != excess[t]]
                        if bad:
                            problems.append(f"P3 {where}: full - partial != absent-c count on {bad} (N3'(c))")
                        if not sum(excess.values()):
                            problems.append(f"P3 {where}: vacuous -- no absent-c triple on the test atoms")
                        line.append(f"P3 atoms {[a.bit_count() for a in atoms]} absent-c {sum(excess.values())}")
                    if "p4" in only:
                        fz = prepared(pc, taxa, work, 0)
                        a = table(extract.count_newick_file(fz, ti, [1000], drop_partial_root=True))
                        b = table(extract.count_newick_file(fz, ti, [1000]))
                        if a != b:
                            problems.append(f"P4 {where}: extract's partial count changes a full-taxon file")
                        d_part = K.partial_triple_counts(fz, ti, args.trees)
                        d_pub = K.triple_counts(fz, taxa, args.trees)
                        if not (np.array_equal(d_part.count, d_pub.count)
                                and d_part.n_nonbinary_loci == d_pub.n_nonbinary_loci):
                            problems.append(f"P4 {where}: the partial dense counter changes a full-taxon file")
                        b_part = K.block_atom_tensors(fz, taxa, args.trees, [atoms], partial=True)
                        b_pub = K.block_atom_tensors(fz, taxa, args.trees, [atoms])
                        if not (np.array_equal(b_part["T"][0], b_pub["T"][0])
                                and b_part["n_nonbinary_loci"] == b_pub["n_nonbinary_loci"]):
                            problems.append(f"P4 {where}: the partial block kernel changes a full-taxon file")
                        line.append("P4 done")
            for p in sorted(work.glob("*.nwk.gz")):
                p.unlink()
            print(f"[check] {' | '.join(line)} | {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stats["problems"] = problems
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    TM.write_json(TM.RUNS / "check.json", stats)
    for p in problems[:40]:
        print(f"  PROBLEM {p}")
    print(f"[check] {'PASS' if not problems else 'FAIL'}: {','.join(sorted(only))} on {len(cells)} cells "
          f"in {stats['seconds']:.0f} s -> {TM.RUNS / 'check.json'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
