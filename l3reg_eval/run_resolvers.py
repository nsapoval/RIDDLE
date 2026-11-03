#!/usr/bin/env python3
"""NetCS and BROOQS on the staged l3reg trees of blobs: 20 networks x three TOB variants.

``blob_cpu_v1/run_blob_cpu.py`` ran these two resolvers on the camus cells.  This runs them on
``stage.py``'s trees of blobs with **that driver's own pieces** -- ``run_timed`` (wall, CPU and
peak RSS of the child, by ``wait4``), its log regexes, ``reroot_enewick`` and ``score`` -- and the
exact commands it ran:

    netcs    tree-qmc --network -i <est_bp1000.nwk> --at tob.rooted.nwk -o netcs.enewick
    brooqs   brooqs -i <est_bp1000.nwk> -t tob.rooted.nwk -o brooqs.enewick --rooted --dist-var

on the gene trees TREE-QMC ``--blob`` read (IQ-TREE's, OUT included) and one of three OUT-rooted
trees of blobs: ``raw`` (TREE-QMC's output through ``root_tob_on_outgroup``, the camus protocol),
``fixed`` (the dataset's corrected ``<key>.tob``) and ``true`` (the oracle).

**Why not ``run_blob_cpu.run_cell`` on pre-seeded work directories.**  It would run, but a
``--force`` or one missing seed file silently launches ``tree-qmc --blob`` -- 5.4-6.3 h and ~1 GB
per network at n = 200 on this dataset -- and its ``tob_stats`` scores against
``camus_nets.tree_of_blobs``, which contracts one cycle per reticulation and is wrong at level 3
for the same reason it is wrong at level 2, so it would write plausible, wrong columns.  This loop
never calls ``--blob``.

**Neither resolver is told k**, and each places one reticulation per blob it resolves, so on these
level-3 networks ``k_pred`` is bounded by the tree of blobs' own blob count -- below ``k`` on the
true TOB, by **two** per level-3 blob, where l2reg lost only one per level-2 blob.  The true TOB
has ``n_l3 + n_l1`` polytomies while ``k = 3 n_l3 + n_l1``, so on rep16 (3 level-3 blobs, 3
level-1, k = 12) the ceiling is 6.  Read a low ``k_pred`` here as the method's design, not as a
failure to find reticulations.

Scoring is ``run_blob_cpu.score``: every output is re-rooted on OUT before its clusters are read,
and refused (``out_rootable`` false, ``f1`` empty) when OUT lies below one of its own hybrids.

Usage (``gnn_cpu_v1/.venv/bin/python``; long runs from a terminal, under ``caffeinate -i``)
-----
    python l3reg_eval/run_resolvers.py --self-check
    python l3reg_eval/run_resolvers.py --dry-run
    python l3reg_eval/run_resolvers.py --keys 0 --variants fixed         # smoke test, ~2 min
    python l3reg_eval/run_resolvers.py --jobs 8                          # all 60 tasks

Resumable per (network, variant) and, inside one, per method; ``--force`` re-runs the resolvers.
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


def task_name(key: str, variant: str) -> str:
    return f"{key}.{variant}"


def run_task(task: dict) -> dict:
    """One network x one TOB variant: both resolvers, then scoring after every clock has stopped.

    Top-level so a spawned worker can unpickle it.
    """
    RB = rb()
    L = RB.lib(str(C.PYLIB))
    key, rep, variant = task["key"], int(task["rep"]), task["variant"]
    run = Path(task["run"])
    work = run / "work" / key / variant
    work.mkdir(parents=True, exist_ok=True)
    trees = C.trees_path(rep).resolve()
    rooted = (C.TOB / f"{key}.{variant}.rooted.nwk").read_text().strip()
    rooted_p = work / "tob.rooted.nwk"
    rooted_p.write_text(rooted + "\n")
    t_start = time.time()

    methods = {}
    for name in task["methods"]:
        out_p, mrec = work / f"{name}.enewick", work / f"{name}.json"
        if not task["force"] and mrec.exists() and out_p.exists():
            methods[name] = json.loads(mrec.read_text())
            continue
        out_p.unlink(missing_ok=True)          # tree-qmc will not overwrite -o; it prints instead
        if name == "netcs":
            cmd = [task["tree_qmc"], "--network", "-i", trees, "--at", rooted_p.name,
                   "-o", out_p.name, *task["netcs_args"]]
        else:
            cmd = [task["brooqs"], "-i", trees, "-t", rooted_p.name, "-o", out_p.name,
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

    # ---- scoring, after every clock has stopped: run_blob_cpu.run_cell's, verbatim in effect
    truth = L.CAMUS.truth_of(L.CAMUS.CamusCell(C.PHASE, rep, C.ARM, C.INPUTS))
    tc = L.CN.label_clusters(truth)
    expect = set(truth.labels) | {C.OUT}
    for name, t in methods.items():
        out_p = work / f"{name}.enewick"
        if t["rc"] != 0 or not out_p.exists() or not out_p.read_text().strip():
            t.update(error=f"rc={t['rc']}", error_tail=RB.tail(work / f"{name}.log"))
            continue
        text = out_p.read_text().strip()
        t["enewick"] = text
        try:
            t.update(RB.score(L, text, tc, expect))
        except Exception as exc:
            t["error"] = f"score: {type(exc).__name__}: {exc}"
    _parent, _len, _lab, children = L.OG.parse_newick(rooted)
    degs = sorted((len(c) for c in children if len(c) > 2), reverse=True)
    rec = {"key": key, "rep": rep, "variant": variant, "k": int(task["k"]), "n": C.N_TAXA,
           "trees": str(trees), "host": socket.gethostname(), "jobs": task["jobs"],
           "tob": {"newick_rooted": rooted, "blob_degrees": degs, "n_blobs": len(degs)},
           "n_true_clusters": len(tc), "methods": methods, "t_start": t_start,
           "t_end": time.time()}
    dst = run / "cells" / f"{task_name(key, variant)}.json"
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(dst)
    return rec


def summarise(run: Path) -> str:
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in sorted((run / "cells").glob("*.json")):
        rec = json.loads(f.read_text())
        for name, m in rec["methods"].items():
            groups.setdefault((rec["variant"], name), []).append({**m, "k": rec["k"],
                                                                  "n_blobs": rec["tob"]["n_blobs"]})
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lines = [f"{'variant':8s} {'method':7s} {'cells':>5s} {'ok':>3s} {'unroot':>6s} {'err':>3s} "
             f"{'k_pred med':>10s} {'k med':>5s} {'blobs med':>9s} {'F1 med':>7s} {'wall med':>8s}"]
    for (variant, name), ms in sorted(groups.items()):
        ok = [m for m in ms if "error" not in m]
        f1 = [m["f1"] for m in ok if m.get("f1") is not None]
        lines.append(
            f"{variant:8s} {name:7s} {len(ms):5d} {len(ok):3d} "
            f"{sum(m.get('out_rootable') is False for m in ok):6d} {len(ms) - len(ok):3d} "
            f"{med([m['k_pred'] for m in ok]):10.1f} {med([m['k'] for m in ms]):5.1f} "
            f"{med([m['n_blobs'] for m in ms]):9.1f} {med(f1):7.4f} "
            f"{med([m['wall'] for m in ms]):8.1f}")
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
    # The gt5k/gt10k datasets run raw + fixed only (common.RESOLVER_VARIANTS); v1 runs all three.
    ap.add_argument("--variants", default=",".join(C.RESOLVER_VARIANTS))
    ap.add_argument("--keys", default="", help="reps, e.g. 0-4,7; default all 20")
    ap.add_argument("--netcs-args", default="", help="extra arguments for tree-qmc --network")
    ap.add_argument("--brooqs-args", default="--rooted --dist-var",
                    help="extra arguments for brooqs; the default is what the landed rows ran")
    ap.add_argument("--out", type=Path, default=C.RUNS)
    ap.add_argument("--run-name", default="resolvers")
    ap.add_argument("--force", action="store_true", help="re-run resolvers already done")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    args.jobs = max(1, args.jobs)
    methods = [m for m in args.methods.split(",") if m]
    variants = [v for v in args.variants.split(",") if v]
    if set(methods) - set(C.RESOLVERS) or set(variants) - set(C.VARIANTS):
        raise SystemExit(f"methods must be in {C.RESOLVERS}, variants in {C.VARIANTS}")
    for tool in (args.tree_qmc, args.brooqs):
        if not Path(tool).exists():
            raise SystemExit(f"missing {tool}; see blob_cpu_v1/README.md for the build")
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run l3reg_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())

    want = C.parse_reps(args.keys)
    cells = [c for c in stage["cells"] if want is None or c["rep"] in want]
    missing = [str(p) for c in cells for p in
               [C.trees_path(c["rep"]), *(C.TOB / f"{c['key']}.{v}.rooted.nwk" for v in variants)]
               if not p.exists()]

    if args.self_check:
        RB = rb()
        L = RB.lib(str(C.PYLIB))
        rc = {Path(t).name: subprocess.run([str(t), "-h"], capture_output=True).returncode
              for t in (args.tree_qmc, args.brooqs)}
        print(f"[resolvers] opc from {L.path}")
        print(f"[resolvers] tree-qmc -h rc={rc['tree-qmc']} | brooqs -h rc={rc['brooqs']}")
        print(f"[resolvers] {len(cells)} staged networks; staged inputs missing: {len(missing)}")
        ok = not missing and all(v == 0 for v in rc.values())
        print(f"[resolvers] self-check {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if missing:
        raise SystemExit(f"{len(missing)} staged inputs missing, e.g. {missing[0]}: run stage.py")

    run = Path(args.out) / args.run_name
    tasks = [{"key": c["key"], "rep": c["rep"], "k": c["k"], "variant": v, "run": str(run),
              "methods": methods, "tree_qmc": str(args.tree_qmc), "brooqs": str(args.brooqs),
              "netcs_args": args.netcs_args.split(), "brooqs_args": args.brooqs_args.split(),
              "force": args.force, "jobs": args.jobs}
             for c in cells for v in variants]
    todo = [t for t in tasks if args.force or not (
        run / "cells" / f"{task_name(t['key'], t['variant'])}.json").exists()]
    print(f"[resolvers] {C.PKG.name} | {len(tasks)} tasks ({len(tasks) - len(todo)} done) | "
          f"methods {','.join(methods)} | jobs {args.jobs} | -> {run}", flush=True)
    if args.dry_run:
        for t in todo:
            print(f"  {t['key']:30s} {t['variant']:6s} k={t['k']}")
        return 0

    (run / "cells").mkdir(parents=True, exist_ok=True)
    info = rb().machine(args)          # tool versions, commits, the TREE-QMC patch hash
    t_run = time.perf_counter()

    def report(i: int, rec: dict) -> None:
        parts = [f"[resolvers] {i:3d}/{len(todo)} {rec['key']:30s} {rec['variant']:6s} "
                 f"k={rec['k']} blobs={rec['tob']['n_blobs']}"]
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
                    print(f"[resolvers] {i:3d}/{len(todo)} {t['key']} {t['variant']} CRASHED: "
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
