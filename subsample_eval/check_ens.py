#!/usr/bin/env python3
"""Gates that tie ``gnn_cpu_v1/infer_gnn_ens.py`` (the real-data driver) to ``infer_gnn_cpu.py`` and
to the benchmark's ensembles.  Seconds to minutes; exits 1 on any failure.

R1   ``infer_gnn_ens.py --draws 1`` writes exactly what ``infer_gnn_cpu.py`` writes with the same
     flags -- byte for byte, on ``gnn_cpu_v1/inputs/n50/00/g_500.nwk`` at ``--m 500 --seed 3
     --k 0,1,2``, under level1 (``best.pt``) and relaxed L2 (``best_ha.pt``).  This licenses
     ``build_draw``, the copy of ``infer()``'s stage chain.
R1b  with ``--draws 5 --seed 3``, ``--draws-out`` line b is exactly ``infer_gnn_cpu.py --seed 3+b``.
R2   end to end, through the Python API (the taxon order can only be set there): on camus cells,
     with the five staged draws' ``chain_seed``s, the truth's taxon order, ``outgroups=("OUT",)``
     and m = 500, every draw's network is the landed subsample cell's, and the ``pool`` and ``vote``
     networks are ``run_ens.py decode``'s for that cell -- both checkpoints, both decoders.  The
     default cells are cam25 rep 0 and cam50 rep 0 (k = 1) and cam100 rep 9 (k = 4), on which
     both ensembles differ from every draw, so R2 is not satisfied by a draw passing through.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/check_ens.py                          # R1, R1b, R2 against runs/
    python subsample_eval/check_ens.py --only r1,r1b
    python subsample_eval/check_ens.py --only r2 --runs-root /tmp/smoke/runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import common as C

PY = sys.executable
GNN = C.GNN
TREES = GNN / "inputs" / "n50" / "00" / "g_500.nwk"
CK = GNN / "checkpoints" / "_l2mix_pair" / "s0"
R1_CASES = ((CK / "best.pt", ["--decoder", "level1"]),
            (CK / "best_ha.pt", ["--decoder", "relaxed", "--max-level", "2"]))
R2_CELLS = (("cam25", 0), ("cam50", 0), ("cam100", 9))
R2_MODELS = (("_l2mix_pair", "best.pt"), ("_l2mix_pair:best_ha", "best_ha.pt"))
R2_AGGS = ("pool", "vote")


def run(args: list[str]) -> str:
    got = subprocess.run([PY, *args], cwd=GNN, capture_output=True, text=True)
    if got.returncode:
        raise SystemExit(f"{' '.join(args[:3])} ... exited {got.returncode}:\n{got.stderr[-2000:]}")
    return got.stdout


def r1(problems: list[str]) -> int:
    n = 0
    common = ["--trees", str(TREES), "--outgroup", "OUT", "--m", "500"]
    for ck, dec in R1_CASES:
        want = run(["infer_gnn_cpu.py", "--model", str(ck), *common, "--seed", "3", "--k", "0,1,2",
                    *dec])
        got = run(["infer_gnn_ens.py", "--model", str(ck), *common, "--seed", "3", "--k", "0,1,2",
                   "--draws", "1", *dec])
        n += 1
        if got != want:
            problems.append(f"R1 {ck.name} {' '.join(dec)}: --draws 1 differs from infer_gnn_cpu.py")
        print(f"[R1] {ck.parent.parent.name}/{ck.name} {' '.join(dec)}: "
              f"{'identical' if got == want else 'DIFFERS'} ({len(want.splitlines())} lines)",
              flush=True)
    return n


def r1b(problems: list[str]) -> int:
    ck, dec = R1_CASES[0]
    common = ["--trees", str(TREES), "--outgroup", "OUT", "--m", "500", "--k", "1"]
    with tempfile.TemporaryDirectory(prefix="check_ens_") as work:
        draws_out = Path(work) / "draws.tsv"
        run(["infer_gnn_ens.py", "--model", str(ck), *common, "--seed", "3", "--draws", "5",
             "--draws-out", str(draws_out), *dec])
        lines = draws_out.read_text().splitlines()
    n = 0
    for b, line in enumerate(lines):
        k, bb, seed, enewick = line.split("\t")
        want = run(["infer_gnn_cpu.py", "--model", str(ck), *common, "--seed", str(3 + b), *dec])
        n += 1
        ok = int(bb) == b and int(seed) == 3 + b and want == f"{k}\t{enewick}\n"
        if not ok:
            problems.append(f"R1b draw {b}: not infer_gnn_cpu.py --seed {3 + b}")
        print(f"[R1b] draw {b} (seed {seed}): {'identical' if ok else 'DIFFERS'}", flush=True)
    if len(lines) != 5:
        problems.append(f"R1b: {len(lines)} draw lines, expected 5")
    return n


def r2(problems: list[str], runs_root: Path) -> int:
    if str(GNN) not in sys.path:
        sys.path.insert(1, str(GNN))
    import infer_gnn_cpu as IG                      # after common: pylib is already first
    import infer_gnn_ens as IE
    from opc.rl import camus as CAMUS

    ds = C.DATASETS["camus"]
    n = 0
    for phase, rep in R2_CELLS:
        row = next(r for r in C.read_tsv(ds.cells) if r["phase"] == phase and int(r["rep"]) == rep)
        arm = row["arm"]
        name = C.cell_name(phase, rep, arm)
        cell = CAMUS.CamusCell(phase, rep, arm, ds.source)
        truth = CAMUS.truth_of(cell)
        seeds = [C.draw_seed("camus", phase, rep, arm, d) for d in range(ds.draws)]
        for tag, fname in R2_MODELS:
            net = IG.load_model(CK / fname, log=open("/dev/null", "w"))
            for dec in ds.decoders:
                decoder, max_level = (("level1", None) if dec == "level1"
                                      else ("relaxed", int(dec.rsplit("_L", 1)[1])))
                for agg in R2_AGGS:
                    t0 = time.perf_counter()
                    got = IE.infer_ensemble(cell.trees_path, net, [int(truth.n_retics)],
                                            outgroups=("OUT",), m=C.N_SUB, seeds=seeds,
                                            order=list(truth.labels), decoder=decoder,
                                            max_level=max_level, agg=agg,
                                            log=open("/dev/null", "w"))[0]
                    bad = []
                    for d, dr in enumerate(got["draws"]):
                        f = ds.run_dir(d, "gnn", dec) / "cells" / f"{name}.json"
                        landed = {m["tag"]: m for m in json.loads(f.read_text())["models"]}[tag]
                        if dr["enewick"] != landed["enewick"]:
                            bad.append(f"draw {d}")
                    f = (runs_root / "camus" / "ens" / agg
                         / (C.GNN_RUN + ("" if dec == "level1" else f"_relaxedL{max_level}"))
                         / "cells" / f"{name}.json")
                    if not f.exists():
                        bad.append(f"no run_ens cell {f}")
                    else:
                        ens = {m["tag"]: m for m in json.loads(f.read_text())["models"]}[tag]
                        if got["enewick"] != ens["enewick"]:
                            bad.append("ensemble")
                        if got["ensemble"]["sel_equals_a_draw"]:
                            print(f"[R2]   note: {name} {tag} {dec} {agg} equals a draw", flush=True)
                    n += 1
                    if bad:
                        problems.append(f"R2 {name} {tag} {dec} {agg}: {', '.join(bad)} differ")
                    print(f"[R2] {name} {tag:20s} {dec:10s} {agg:5s}: "
                          f"{'identical' if not bad else 'DIFFERS: ' + ', '.join(bad)} | "
                          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="r1,r1b,r2", help="comma list of r1, r1b, r2")
    ap.add_argument("--runs-root", type=Path, default=C.RUNS,
                    help="where run_ens.py decode wrote camus/ens/ (R2)")
    args = ap.parse_args()
    want = {x.strip() for x in args.only.split(",") if x.strip()}
    problems: list[str] = []
    counts = {}
    if "r1" in want:
        counts["R1"] = r1(problems)
    if "r1b" in want:
        counts["R1b"] = r1b(problems)
    if "r2" in want:
        counts["R2"] = r2(problems, args.runs_root)
    for p in problems:
        print(f"  FAIL {p}")
    print(f"[check_ens] {'PASS' if not problems else 'FAIL'}: "
          + ", ".join(f"{k} {v} checks" for k, v in counts.items()))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
