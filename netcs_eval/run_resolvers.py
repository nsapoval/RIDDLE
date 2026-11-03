#!/usr/bin/env python3
"""NetCS and BROOQS on the corpus's own TOB-QMC trees of blobs, native and unrooted: 1,200 tasks.

``blob_cpu_v1/run_blob_cpu.py`` ran these two resolvers on the camus cells, on a tree of blobs
rooted on a simulated OUT.  This corpus has no OUT, and the user chose (2026-09-12) to run both
tools **as the corpus ships them**:

    netcs    tree-qmc --network -i <corpus gene trees> --at tob.nwk -o netcs.enewick
    brooqs   brooqs -i <corpus gene trees> -t tob.nwk -o brooqs.enewick

on the IQ-TREE trees exactly as the corpus has them (read in place, unrooted, no OUT) and
``tob/<key>.tqmc.nwk``, TOB-QMC default's unrooted tree of blobs byte for byte.  **BROOQS runs
without ``--rooted``/``--dist-var``**, which differs from every landed camus row: ``--rooted``
tells it to trust a root, and there is none to trust; ``--dist-var`` requires ``--rooted``.  Both
tools accept the unrooted input (smoke-tested on ``L2_n50`` network 1: rc 0, ~9 s each).

Timing and logs reuse ``run_blob_cpu``'s own pieces -- ``run_timed`` (wall, CPU and peak RSS of
the child, by ``wait4``) and its log regexes.  ``--blob`` is never called.

**Neither resolver is told k.**  Each places one reticulation per blob it resolves.

Scoring, after every clock has stopped
--------------------------------------
Each output is written with whatever root the tool chose, which is not a rooting of the truth.
For the rooted metrics -- cluster F1 here, luay in ``make_csv.py`` -- ``OUT`` is inserted on the
network's tree edge nearest the true root (``common.insert_out_at_truth``, the same oracle that
rooted the GNN's gene trees; only edges outside every hybrid's subtree are eligible), and the
string then goes through ``run_blob_cpu.score`` unchanged: re-rooted on OUT, refused
(``out_rootable`` false, ``f1`` empty) if that is impossible.  ``f1_as_emitted`` is the network read
as written.  ``root_cost`` records how far the chosen edge is from a displayed true root, in taxa.
Hybrid ancestry, in ``make_csv.py``, is read off the emitted network and needs no root.

Usage (``gnn_cpu_v1/.venv/bin/python``; long runs from a terminal, under ``caffeinate -i``)
-----
    python netcs_eval/run_resolvers.py --self-check
    python netcs_eval/run_resolvers.py --dry-run
    python netcs_eval/run_resolvers.py --phases netcs_L2_n50 --limit 2 --run-name smoke    # ~40 s
    python netcs_eval/run_resolvers.py --jobs 12                                            # all 1,200

Resumable per condition and, inside one, per method; ``--force`` re-runs the resolvers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import common as C

TREE_QMC = C.BLOB / "tools" / "TREE-QMC" / "build" / "tree-qmc"
BROOQS = C.BLOB / ".venv-brooqs" / "bin" / "brooqs"

_RB = None


def rb():
    """``run_blob_cpu`` as a module, loaded once per process by absolute path.

    Registered in ``sys.modules`` under its own name so anything of it that crosses a process
    boundary pickles.  Its module level pins BLAS to one thread and compiles regexes, nothing more,
    so importing it starts no tool.
    """
    global _RB
    if _RB is None:
        spec = importlib.util.spec_from_file_location("run_blob_cpu", C.BLOB / "run_blob_cpu.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_blob_cpu"] = mod
        spec.loader.exec_module(mod)
        _RB = mod
    return _RB


def run_task(task: dict) -> dict:
    """One condition: both resolvers, then scoring after every clock has stopped.

    Top-level so a spawned worker can unpickle it.
    """
    RB = rb()
    L = RB.lib(str(C.PYLIB))
    key, rep, phase = task["key"], int(task["rep"]), task["phase"]
    run = Path(task["run"])
    work = run / "work" / key
    work.mkdir(parents=True, exist_ok=True)
    trees = Path(task["trees"]).resolve()
    tob_text = C.tob_input_path(key).read_text().strip()
    tob_p = work / "tob.nwk"
    tob_p.write_text(tob_text + "\n")
    t_start = time.time()

    methods = {}
    for name in task["methods"]:
        out_p, mrec = work / f"{name}.enewick", work / f"{name}.json"
        if not task["force"] and mrec.exists() and out_p.exists():
            methods[name] = json.loads(mrec.read_text())
            continue
        out_p.unlink(missing_ok=True)          # tree-qmc will not overwrite -o; it prints instead
        if name == "netcs":
            cmd = [task["tree_qmc"], "--network", "-i", trees, "--at", tob_p.name,
                   "-o", out_p.name, *task["netcs_args"]]
        else:
            cmd = [task["brooqs"], "-i", trees, "-t", tob_p.name, "-o", out_p.name,
                   *task["brooqs_args"]]
        m0 = time.time()
        t = RB.run_timed(cmd, work / f"{name}.log", work)
        text = (work / f"{name}.log").read_text(errors="replace")
        if name == "netcs":
            t["self_s"] = RB.first_float(RB._QMC_TIME, text)
            hit = RB._REROOTED.search(text)
            t["note"] = (f"rerooted:{hit.group(1)}" if hit else
                         "rooting-valid" if "rooting is valid" in text else
                         "not-semi-directed" if "not a valid semi-directed" in text else "")
        else:
            t["self_s"] = RB.first_float(RB._BRQ_TIME, text)
            t["note"] = f"swaps:{text.count('[hybrid swap] reason')}"
        t.update(t_start=m0, t_end=time.time())
        mrec.write_text(json.dumps(t, indent=1))
        methods[name] = t

    # ---- scoring, after every clock has stopped
    truth = C.truth_of(rep, phase=phase)
    tc = L.CN.label_clusters(truth)
    index = {lab: i for i, lab in enumerate(truth.labels)}
    targets = C.root_targets(truth)
    expect = set(truth.labels) | {C.OUT}
    for name, t in methods.items():
        for stale in ("error", "error_tail", "enewick", "root_cost", "root_ties"):
            t.pop(stale, None)
        out_p = work / f"{name}.enewick"
        if t["rc"] != 0 or not out_p.exists() or not out_p.read_text().strip():
            t.update(error=f"rc={t['rc']}", error_tail=RB.tail(work / f"{name}.log"))
            continue
        text = out_p.read_text().strip()
        t["enewick"] = text
        try:
            placed = C.insert_out_at_truth(text, index, targets, keep_lengths=False, network=True)
            t.update(root_cost=placed["cost"], root_ties=placed["ties"])
            t.update(RB.score(L, placed["newick"], tc, expect))
        except Exception as exc:
            t["error"] = f"score: {type(exc).__name__}: {exc}"
    _parent, _len, _lab, children = L.OG.parse_newick(tob_text)
    degs = sorted((len(c) for c in children if len(c) > 2), reverse=True)
    rec = {"key": key, "phase": phase, "rep": rep, "arm": task["arm"], "k": int(task["k"]),
           "n": len(truth.labels), "trees": str(trees), "host": socket.gethostname(),
           "jobs": task["jobs"], "tob": {"newick": tob_text, "kind": C.RESOLVER_TOB,
                                         "degrees_as_written": degs, "n_blobs": task["n_blobs"]},
           "n_true_clusters": len(tc), "methods": methods, "t_start": t_start, "t_end": time.time()}
    dst = run / "cells" / f"{key}.json"
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(dst)
    return rec


def summarise(run: Path) -> str:
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in sorted((run / "cells").glob("*.json")):
        rec = json.loads(f.read_text())
        for name, m in rec["methods"].items():
            groups.setdefault((rec["phase"], name), []).append({**m, "k": rec["k"],
                                                                "n_blobs": rec["tob"]["n_blobs"]})
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lines = [f"{'phase':14s} {'method':7s} {'cells':>5s} {'ok':>4s} {'unroot':>6s} {'err':>3s} "
             f"{'k_pred med':>10s} {'k med':>5s} {'blobs med':>9s} {'F1 med':>7s} "
             f"{'F1 emit':>7s} {'cost med':>8s} {'wall med':>8s} {'cpu sum h':>9s}"]
    for (phase, name), ms in sorted(groups.items()):
        ok = [m for m in ms if "error" not in m]
        f1 = [m["f1"] for m in ok if m.get("f1") is not None]
        lines.append(
            f"{phase:14s} {name:7s} {len(ms):5d} {len(ok):4d} "
            f"{sum(m.get('out_rootable') is False for m in ok):6d} {len(ms) - len(ok):3d} "
            f"{med([m['k_pred'] for m in ok]):10.1f} {med([m['k'] for m in ms]):5.1f} "
            f"{med([m['n_blobs'] for m in ms]):9.1f} {med(f1):7.4f} "
            f"{med([m['f1_as_emitted'] for m in ok]):7.4f} {med([m['root_cost'] for m in ok]):8.1f} "
            f"{med([m['wall'] for m in ms]):8.1f} {sum(m['cpu'] for m in ms) / 3600:9.2f}")
    errs = Counter(m.get("error", "")[:60] for ms in groups.values() for m in ms if "error" in m)
    if errs:
        lines.append("errors: " + "; ".join(f"{n} x {e}" for e, n in errs.most_common()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1, help="tasks run concurrently, each tool 1 thread")
    ap.add_argument("--tree-qmc", type=Path, default=TREE_QMC)
    ap.add_argument("--brooqs", type=Path, default=BROOQS)
    ap.add_argument("--methods", default=",".join(C.RESOLVERS))
    ap.add_argument("--phases", default="", help="comma list of netcs_L<l>_n<n>; default all")
    ap.add_argument("--keys", default="", help="comma list of condition keys; default all")
    ap.add_argument("--limit", type=int, default=0, help="first N selected conditions (a smoke)")
    ap.add_argument("--netcs-args", default="", help="extra arguments for tree-qmc --network")
    ap.add_argument("--brooqs-args", default="",
                    help="extra arguments for brooqs; default none -- native unrooted mode")
    ap.add_argument("--out", type=Path, default=C.RUNS)
    ap.add_argument("--run-name", default="resolvers")
    ap.add_argument("--force", action="store_true", help="re-run resolvers already done")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    args.jobs = max(1, args.jobs)
    methods = [m for m in args.methods.split(",") if m]
    if set(methods) - set(C.RESOLVERS):
        raise SystemExit(f"methods must be in {C.RESOLVERS}")
    for tool in (args.tree_qmc, args.brooqs):
        if not Path(tool).exists():
            raise SystemExit(f"missing {tool}; see blob_cpu_v1/README.md for the build")
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run netcs_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())

    want_phases = {p for p in args.phases.split(",") if p}
    want_keys = {k for k in args.keys.split(",") if k}
    cells = [c for c in stage["cells"] if (not want_phases or c["phase"] in want_phases)
             and (not want_keys or c["key"] in want_keys)]
    if args.limit:
        cells = cells[:args.limit]
    trees_of = lambda c: C.corpus_trees(c["level"], c["n"], c["rep"], C.ARMS[c["arm"]])  # noqa: E731
    missing = [str(p) for c in cells for p in (trees_of(c), C.tob_input_path(c["key"]))
               if not p.exists()]

    if args.self_check:
        RB = rb()
        L = RB.lib(str(C.PYLIB))
        rc = {Path(t).name: subprocess.run([str(t), "-h"], capture_output=True).returncode
              for t in (args.tree_qmc, args.brooqs)}
        print(f"[resolvers] opc from {L.path}")
        print(f"[resolvers] tree-qmc -h rc={rc['tree-qmc']} | brooqs -h rc={rc['brooqs']}")
        print(f"[resolvers] {len(cells)} staged conditions; inputs missing: {len(missing)}")
        ok = not missing and all(v == 0 for v in rc.values())
        print(f"[resolvers] self-check {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if missing:
        raise SystemExit(f"{len(missing)} inputs missing, e.g. {missing[0]}: run stage.py")

    run = Path(args.out) / args.run_name
    tasks = [{"key": c["key"], "phase": c["phase"], "rep": c["rep"], "arm": c["arm"], "k": c["k"],
              "n_blobs": c["tob"][C.RESOLVER_TOB]["n_blobs"], "trees": str(trees_of(c)),
              "run": str(run), "methods": methods, "tree_qmc": str(args.tree_qmc),
              "brooqs": str(args.brooqs), "netcs_args": args.netcs_args.split(),
              "brooqs_args": args.brooqs_args.split(), "force": args.force, "jobs": args.jobs}
             for c in cells]
    todo = [t for t in tasks if args.force or not (run / "cells" / f"{t['key']}.json").exists()]
    print(f"[resolvers] {len(tasks)} tasks ({len(tasks) - len(todo)} done) | methods "
          f"{','.join(methods)} | jobs {args.jobs} | -> {run}", flush=True)
    if args.dry_run:
        for t in todo[:40]:
            print(f"  {t['key']:34s} k={t['k']} blobs={t['n_blobs']}")
        if len(todo) > 40:
            print(f"  ... and {len(todo) - 40} more")
        return 0

    (run / "cells").mkdir(parents=True, exist_ok=True)
    info = rb().machine(args)          # tool versions, commits, the TREE-QMC patch hash
    t_run = time.perf_counter()

    def report(i: int, rec: dict) -> None:
        parts = [f"[resolvers] {i:4d}/{len(todo)} {rec['key']:34s} k={rec['k']} "
                 f"blobs={rec['tob']['n_blobs']}"]
        for name, m in rec["methods"].items():
            f1 = m.get("f1")
            parts.append(f"{name} {m['wall']:6.1f}s " + (
                f"ERR {m['error']}" if "error" in m else
                f"k={m['k_pred']} F1={'unrootable' if f1 is None else f'{f1:.4f}'}"))
        print(" | ".join(parts), flush=True)

    crashed = 0
    if args.jobs == 1:
        for i, t in enumerate(todo, 1):
            report(i, run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t): t for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    report(i, fut.result())
                except Exception as exc:
                    crashed += 1
                    t = futs[fut]
                    print(f"[resolvers] {i:4d}/{len(todo)} {t['key']} CRASHED: "
                          f"{type(exc).__name__}: {exc}", flush=True)

    header = {**info, "tasks_run_this_invocation": len(todo),
              "wall_this_invocation": time.perf_counter() - t_run,
              "args": {k: str(v) for k, v in vars(args).items()}}
    (run / f"header_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(header, indent=1))
    text = summarise(run)
    (run / "summary.txt").write_text(text + "\n")
    print(text)
    return 0 if not crashed and "errors:" not in text else 3


if __name__ == "__main__":
    raise SystemExit(main())
