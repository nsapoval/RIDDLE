#!/usr/bin/env python3
"""Greedy-by-support -- the ``greedy`` rows -- on the staged netcs cells.

The GNN rows come from ``gnn_cpu_v1/run_gnn_cpu.py``, which runs no baseline.  This runs that
driver's shared stages function for function, then a decode ranked by support instead of by model
score:

    root on OUT, prune it  ->  count clusters  ->  tau = 2 candidate set
    ->  decode by support, budget k  ->  realise  ->  eNewick

``OUT`` is where ``stage.py``'s true-root oracle put it, so these rows -- like the GNN rows, and
unlike NetCS and BROOQS -- are handed the root (``common.py``).

Two decoders, chosen as in the GNN driver:

``--decoder level1`` (default) -- ``greedy_support_rollout(ClusterEnv(gs, budget=k))``, argmax-support
    under the formal (L)/(N3O)/budget mask, realised through the closure.
``--decoder relaxed --max-level L`` -- ``opc.rl.relaxed.RelaxedDecoder`` with support as the scores
    and the same tie rule: Hasse-diagram constraints (reticulations <= k, in-degree <= 2, level <= L,
    a donor assignment), realised with no closure.  This corpus is level 1 and level 2, so the
    ladder is L1 and L2.

Budget and taxon order both come from the true network, as in the GNN driver.

**m is the file's own locus count** (at most 1,000), as ``run_gnn_cpu.py --m-from-trees`` sets it
for the GNN rows.  Greedy reads no feature normalised by m -- support ranks the same either way,
and tau = 2 is an absolute count -- so this only keeps the recorded ground set honest.

l3reg_eval's ``--camus`` mode is not copied: ``make_csv.py --parity`` borrows
``l3reg_eval/runs/greedy_camus``, which is dataset-independent.

Usage (``gnn_cpu_v1/.venv/bin/python``)
-----
    python netcs_eval/run_greedy.py --dry-run
    python netcs_eval/run_greedy.py --cells netcs_eval/shards/CELLS_00.tsv      # one shard
    python netcs_eval/run_greedy.py --decoder relaxed --max-level 2             # -> runs/greedy_relaxedL2/

Resumable per cell; ``--force`` re-runs.  A run directory holds one decoder: resuming into a
directory whose cells were decoded differently is refused.  Shards may share a run directory:
every cell is written atomically under its own name.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import common as C

from opc.clusters import extract  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR, greedy_support_rollout  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.solve import candidates  # noqa: E402
from opc.solve import truth as T  # noqa: E402

#: Every checkpoint trained at 1,000 loci and tau = 2 (``run_gnn_cpu.M, TAU``).
M, TAU = 1000, 2


def run_cell(row: dict, inputs: Path, work_root: Path, decoder: str, max_level) -> dict:
    phase, rep, arm = row["phase"], int(row["rep"]), row["arm"]
    cell = CAMUS.CamusCell(phase, rep, arm, Path(inputs))
    truth = CAMUS.truth_of(cell)
    taxa = list(truth.labels)
    k = int(truth.n_retics)
    taxon_index = {lab: i for i, lab in enumerate(taxa)}

    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{phase}_rep{rep:02d}_", dir=work_root))
    gz, table = work / "trees.nwk.gz", work / "clusters.parquet"
    t0 = time.perf_counter()
    try:
        tree_info = CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=M)
        m = min(M, int(tree_info["n_loci"]))
        counter = extract.count_newick_file(gz, taxon_index, [m])
        masks, support, n_raw = counter.finalize(min_support=1)
        extract.write_cluster_table(table, masks, support, [m], len(taxa), n_raw,
                                    source=str(cell.trees_path), arm=arm)
        words, sup, meta = extract.read_cluster_table(table)
        gs = candidates.build(words, sup[:, meta["checkpoints"].index(m)], len(taxa),
                              m=m, tau=TAU)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    prep_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    stats = None
    if decoder == "level1":
        env = ClusterEnv(gs, budget=k, graph=OverlapCSR.of(gs))
        sel = greedy_support_rollout(env)
        if not env.verify():
            raise AssertionError(f"{cell.key}: the greedy selection is not a feasible cluster system")
        got = search.realise(gs.words, sel, len(taxa), taxa)
    else:
        dec = RX.RelaxedDecoder(gs.words, len(taxa), k, max_level)
        sel = dec.decode(gs.support)
        if not dec.verify():
            raise AssertionError(f"{cell.key}: relaxed state fails verify: {dec.verify_reason}")
        got = RX.realise(gs.words, sel, len(taxa), taxa)
        stats = dec.stats
    decode_s = time.perf_counter() - t1
    enewick = None if got is None else NW.to_enewick(got[0])
    k_pred = -1 if got is None else int(got[0].n_retics)
    # ``run_gnn_cpu.run_cell``'s scorer, so a greedy F1 and a GNN F1 are one definition.
    scorer = SimpleNamespace(truth_rows=T.true_cluster_rows(gs.words, truth),
                             n_true=len(NW.nontrivial_clusters(truth)))
    counts = I.Instance.cluster_counts(scorer, sel)
    return {"phase": phase, "rep": rep, "arm": arm, "key": cell.key,
            "trees": cell.trees_path.name, "n": len(taxa), "k": k,
            "decoder": decoder, "max_level": max_level,
            "n_loci": int(tree_info["n_loci"]), "m": m, "n_clusters": int(gs.n_clusters),
            "host": socket.gethostname(), "wall": {"prep": prep_s, "decode": decode_s},
            "models": [{"tag": "greedy", "seed": 0, "decoder": decoder, "max_level": max_level,
                        "n_selected": int(sel.size), **counts, "k_pred": k_pred,
                        "enewick": enewick, "decode_stats": stats}]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", choices=RX.DECODERS, default="level1",
                    help="level1: the formal (L)/(N3O) decoder (default); relaxed: Hasse-diagram "
                         "constraints, needs --max-level")
    ap.add_argument("--max-level", type=int, default=None,
                    help="the relaxed decoder's level cap (>= 1); required with --decoder relaxed")
    ap.add_argument("--inputs", type=Path, default=C.INPUTS)
    ap.add_argument("--cells", type=Path, default=C.CELLS, help="CELLS.tsv, or one shards/CELLS_XX.tsv")
    ap.add_argument("--out", type=Path, default=C.RUNS)
    ap.add_argument("--run-name", default="", help="default greedy, with _relaxedL<L> for relaxed")
    ap.add_argument("--phases", default="", help="comma list; default all")
    ap.add_argument("--reps", default="", help="network ids: comma list or a-b ranges; default all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        tag = RX.decoder_tag(args.decoder, args.max_level)
    except ValueError as exc:
        raise SystemExit(f"--decoder {args.decoder}: {exc}")

    if not Path(args.cells).exists():
        raise SystemExit(f"no {args.cells}: run netcs_eval/stage.py first")
    want_phases = {p for p in args.phases.split(",") if p}
    want_reps = C.parse_reps(args.reps)
    cells = [c for c in C.read_tsv(args.cells)
             if (not want_phases or c["phase"] in want_phases)
             and (want_reps is None or int(c["rep"]) in want_reps)]
    if not cells:
        raise SystemExit("no cells selected")
    run = Path(args.out) / (args.run_name or ("greedy" + (
        "" if tag == "level1" else f"_relaxedL{args.max_level}")))
    dst_of = lambda c: run / "cells" / f"{C.cell_name(c['phase'], int(c['rep']), c['arm'])}.json"  # noqa: E731
    if not args.force:
        for c in cells:
            if dst_of(c).exists():
                have = C.cell_decoder(json.loads(dst_of(c).read_text()))
                if have != tag:
                    raise SystemExit(f"{dst_of(c)} was decoded as {have}, this run is {tag}: "
                                     f"pick another --run-name, or --force to overwrite")
    todo = [c for c in cells if args.force or not dst_of(c).exists()]
    print(f"[greedy] {len(cells)} cells ({len(cells) - len(todo)} done) | decoder {tag} | "
          f"{args.cells.name} -> {run}", flush=True)
    if args.dry_run:
        for c in todo:
            print(f"  {c['phase']:14s} rep{int(c['rep']):02d} {c['arm']:13s} n={c['n']} k={c['k']}")
        return 0

    (run / "cells").mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    for i, row in enumerate(todo, 1):
        rec = run_cell(row, args.inputs, run / "work", args.decoder, args.max_level)
        dst = dst_of(row)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(dst)
        m = rec["models"][0]
        print(f"[greedy] {i:4d}/{len(todo)} {rec['phase']:14s} rep{rec['rep']:02d} {rec['arm']:13s} "
              f"n={rec['n']} k={rec['k']} |C|={rec['n_clusters']:6d} | prep {rec['wall']['prep']:6.2f}s "
              f"decode {rec['wall']['decode']:5.2f}s | k_pred={m['k_pred']} F1={m['f1']:.4f}",
              flush=True)
    print(f"[greedy] done: {len(todo)} cells in {time.perf_counter() - t_run:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
