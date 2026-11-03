#!/usr/bin/env python3
"""Greedy-by-support -- the ``greedy`` rows -- on the staged l2_n1000_gt_50k_v1 cells.

This is a copy of ``l2reg_eval/run_greedy.py`` with ``common`` pointing here. It reads every one of
the 10,000 estimated trees at tau = 20 (``common.M, TAU``), and always records ``m`` and ``tau``.
``--camus`` (parity) is l2reg's job and still works from there.  The original docstring follows.

Greedy-by-support -- the ``greedy`` rows -- on the staged l2reg cells, or on camus for parity.

The GNN rows come from ``gnn_cpu_v1/run_gnn_cpu.py``, which runs no baseline.  This runs that
driver's shared stages (``run_cell``, lines 243-253) function for function, then a decode ranked by
support instead of by model score:

    root on OUT, prune it  ->  count clusters  ->  tau = 2 candidate set
    ->  decode by support, budget k  ->  realise  ->  eNewick

Two decoders, chosen as in the GNN driver:

``--decoder level1`` (default) -- ``greedy_support_rollout(ClusterEnv(gs, budget=k))``, argmax-support
    under the formal (L)/(N3O)/budget mask, realised through the closure.  ``nodecls.decode`` is the
    same loop with model scores in place of support.  It has no floating-point step, so it has to
    reproduce the landed rows exactly on any CPU; ``--camus`` runs gnn_cpu_v1's 100 camus cells so
    ``make_csv.py --parity`` can hold it to that.
``--decoder relaxed --max-level L`` -- ``opc.rl.relaxed.RelaxedDecoder`` with support as the scores
    and the same tie rule: Hasse-diagram constraints (reticulations <= k, in-degree <= 2, level <= L,
    a donor assignment), realised with no closure.  See that module's docstring.

Budget and taxon order both come from the true network, as in the GNN driver: ``k`` is the budget
every landed row was handed, and the taxon order fixes the candidate rows and so every support tie.

Usage (``gnn_cpu_v1/.venv/bin/python``)
-----
    python n1000_eval/run_greedy.py --dry-run
    python n1000_eval/run_greedy.py                                   # -> runs/greedy/
    python n1000_eval/run_greedy.py --decoder relaxed --max-level 2   # -> runs/greedy_relaxedL2/
    python n1000_eval/run_greedy.py --camus                           # -> runs/greedy_camus/

Resumable per cell; ``--force`` re-runs.  A run directory holds one decoder: resuming into a
directory whose cells were decoded differently is refused.
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

#: Every landed row and every checkpoint: 1,000 loci, tau = 2 (``run_gnn_cpu.M, TAU``).  The
#: staged cells use ``C.M, C.TAU`` instead -- the same pair unless ``REG_GT`` names the 5,000- or
#: 10,000-tree dataset, where every tree is read at tau = ceil(0.002 m).  ``--camus`` always uses
#: these, since its only job is parity with the landed camus rows.
M, TAU = 1000, 2


def run_cell(row: dict, inputs: Path, work_root: Path, decoder: str, max_level,
             m: int = M, tau: int = TAU) -> dict:
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
        tree_info = CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=m)
        counter = extract.count_newick_file(gz, taxon_index, [m])
        masks, support, n_raw = counter.finalize(min_support=1)
        extract.write_cluster_table(table, masks, support, [m], len(taxa), n_raw,
                                    source=str(cell.trees_path), arm=arm)
        words, sup, meta = extract.read_cluster_table(table)
        gs = candidates.build(words, sup[:, meta["checkpoints"].index(m)], len(taxa),
                              m=m, tau=tau)
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
            "n_loci": int(tree_info["n_loci"]), "n_clusters": int(gs.n_clusters),
            "m": m, "tau": tau,
            "host": socket.gethostname(), "wall": {"prep": prep_s, "decode": decode_s},
            "models": [{"tag": "greedy", "seed": 0, "decoder": decoder, "max_level": max_level,
                        "n_selected": int(sel.size), **counts, "k_pred": k_pred,
                        "enewick": enewick, "decode_stats": stats}]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camus", action="store_true",
                    help="gnn_cpu_v1's own 100 camus cells, the input to make_csv.py --parity")
    ap.add_argument("--decoder", choices=RX.DECODERS, default="level1",
                    help="level1: the formal (L)/(N3O) decoder (default); relaxed: Hasse-diagram "
                         "constraints, needs --max-level")
    ap.add_argument("--max-level", type=int, default=None,
                    help="the relaxed decoder's level cap (>= 1); required with --decoder relaxed")
    ap.add_argument("--inputs", type=Path, default=None, help="default: staged, or gnn_cpu_v1's")
    ap.add_argument("--cells", type=Path, default=None, help="default: staged, or gnn_cpu_v1's")
    ap.add_argument("--out", type=Path, default=C.RUNS)
    ap.add_argument("--run-name", default="",
                    help="default greedy or greedy_camus, with _relaxedL<L> for the relaxed decoder")
    ap.add_argument("--phases", default="", help="comma list; default all")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        tag = RX.decoder_tag(args.decoder, args.max_level)
    except ValueError as exc:
        raise SystemExit(f"--decoder {args.decoder}: {exc}")

    inputs = args.inputs or (C.GNN / "inputs" if args.camus else C.INPUTS)
    cells_p = args.cells or (C.GNN / "CELLS.tsv" if args.camus else C.CELLS)
    if not cells_p.exists():
        raise SystemExit(f"no {cells_p}: run n1000_eval/stage.py first")
    want_phases = {p for p in args.phases.split(",") if p}
    want_reps = C.parse_reps(args.reps)
    cells = [c for c in C.read_tsv(cells_p)
             if (not want_phases or c["phase"] in want_phases)
             and (want_reps is None or int(c["rep"]) in want_reps)]
    if not cells:
        raise SystemExit("no cells selected")
    default_name = ("greedy_camus" if args.camus else "greedy") + (
        "" if tag == "level1" else f"_relaxedL{args.max_level}")
    run = Path(args.out) / (args.run_name or default_name)
    # Not ``m``: the print loop below binds that name to each cell's model record.
    loci, tau = (M, TAU) if args.camus else (C.M, C.TAU)
    dst_of = lambda c: run / "cells" / f"{C.cell_name(c['phase'], int(c['rep']), c['arm'])}.json"  # noqa: E731
    if not args.force:
        for c in cells:
            if dst_of(c).exists():
                have = C.cell_decoder(json.loads(dst_of(c).read_text()))
                if have != tag:
                    raise SystemExit(f"{dst_of(c)} was decoded as {have}, this run is {tag}: "
                                     f"pick another --run-name, or --force to overwrite")
                prior = json.loads(dst_of(c).read_text())
                if (prior.get("m", M), prior.get("tau", TAU)) != (loci, tau):
                    raise SystemExit(f"{dst_of(c)} was built on (m, tau) = "
                                     f"{(prior.get('m', M), prior.get('tau', TAU))}, this run is "
                                     f"{(loci, tau)}: pick another --run-name, or --force")
    todo = [c for c in cells if args.force or not dst_of(c).exists()]
    print(f"[greedy] {C.PKG.name if not args.camus else 'camus'} | m {loci} tau {tau} | "
          f"{len(cells)} cells ({len(cells) - len(todo)} done) | decoder {tag} | "
          f"{inputs} -> {run}", flush=True)
    if args.dry_run:
        for c in todo:
            print(f"  {c['phase']:7s} rep{int(c['rep']):02d} {c['arm']:10s} n={c['n']} k={c['k']}")
        return 0

    (run / "cells").mkdir(parents=True, exist_ok=True)
    t_run = time.perf_counter()
    for i, row in enumerate(todo, 1):
        rec = run_cell(row, inputs, run / "work", args.decoder, args.max_level, loci, tau)
        dst = dst_of(row)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(dst)
        m = rec["models"][0]
        print(f"[greedy] {i:3d}/{len(todo)} {rec['phase']:7s} rep{rec['rep']:02d} n={rec['n']} "
              f"k={rec['k']} |C|={rec['n_clusters']:6d} | prep {rec['wall']['prep']:6.2f}s "
              f"decode {rec['wall']['decode']:5.2f}s | k_pred={m['k_pred']} F1={m['f1']:.4f}",
              flush=True)
    print(f"[greedy] done: {len(todo)} cells in {time.perf_counter() - t_run:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
