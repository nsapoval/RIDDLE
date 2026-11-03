#!/usr/bin/env python3
"""NetCS and BROOQS on the GNN's tree of blobs (and optionally the true one): 20 networks x variant.

This is ``l2reg_eval/run_resolvers.py`` with three changes.

* **The TOB variants are ``gnn`` and ``true``.**
  * ``gnn`` is the selF1 GNN network collapsed to its tree of blobs (``stage_gnn_tob.py``). It is
    the only inferred TOB this dataset has, since TREE-QMC ``--blob`` does not reach n = 1000.
  * ``true`` is the oracle. It is optional.
* **Every tool runs under a timeout** (``pte_common.run_timed_to``). The child's whole process
  group is killed, and its CPU and RSS are still read exactly by ``wait4``. Nothing has measured
  either resolver at n = 1000 with 10,000 trees, so a hang must not be able to hold a worker
  forever. A timeout is recorded as ``error: timeout``, never dropped.
* **A stale-TOB guard.** A ``gnn`` task records the sha256 of the TOB it ran on and of the GNN cell
  that TOB came from. Before anything runs, both are checked against ``tob/gnn_tob.json`` and the
  cell on disk now. A GNN re-run without a re-collapse is refused.

The commands are the landed ones, on the staged IQ-TREE trees (OUT included):

    netcs    tree-qmc --network -i <est_bp1000.nwk> --at tob.rooted.nwk -o netcs.enewick
    brooqs   brooqs -i <est_bp1000.nwk> -t tob.rooted.nwk -o brooqs.enewick --rooted --dist-var

Scoring is ``run_blob_cpu.score``. The output is re-rooted on OUT, and is ``out_not_rootable`` when
OUT lies below one of its hybrids. Every output is then held to ``pte_common``'s invariants, and a
violation is recorded on the method:

* ``k_pred`` <= the eligible polytomies;
* BROOQS ``k_pred`` == eligible;
* the TOB's splits == the output's bridge splits.

Usage (``gnn_cpu_v1/.venv/bin/python``; run long jobs from a terminal under ``caffeinate -i``)
-----
    python n1000_eval/run_resolvers.py --self-check
    python n1000_eval/run_resolvers.py --dry-run
    python n1000_eval/run_resolvers.py --keys 0                 # one network: the timing probe
    python n1000_eval/run_resolvers.py --jobs 8                 # all 20 GNN-TOB tasks
    python n1000_eval/run_resolvers.py --variants true --jobs 8 # the optional oracle arm
    python n1000_eval/run_resolvers.py --run-name resolvers_remote --score-only --jobs 4
                                                                # score remote_pkg results

It resumes per (network, variant), and inside one, per method. ``--force`` re-runs.
"""

from __future__ import annotations

import argparse
import hashlib
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
#: Non-root eligibility thresholds (``pte_common``'s module docstring).  The TOB is fed as
#: ``(OUT,<tob>)``, so its root has two children and every blob is a non-root node.
ELIGIBLE_AT = {"brooqs": 3, "netcs": 4}

_RB = None


def rb():
    """``run_blob_cpu`` as a module, loaded once per process by absolute path (l2reg's ``rb``)."""
    global _RB
    if _RB is None:
        spec = importlib.util.spec_from_file_location("run_blob_cpu", C.BLOB / "run_blob_cpu.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_blob_cpu"] = mod
        spec.loader.exec_module(mod)
        _RB = mod
    return _RB


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def task_name(key: str, variant: str) -> str:
    return f"{key}.{variant}"


def gnn_cell_path(gnn_run: Path, rep: int) -> Path:
    return gnn_run / "cells" / f"{C.cell_name(C.PHASE, rep, C.ARM)}.json"


def stale_problems(cells: list[dict], tob_dir: Path, gnn_run: Path) -> list[str]:
    """The ``gnn`` TOBs must be the ones ``stage_gnn_tob.py`` wrote from the GNN cells on disk now."""
    index_p = tob_dir / "gnn_tob.json"
    if not index_p.exists():
        return [f"no {index_p}: run n1000_eval/stage_gnn_tob.py after the GNN grid"]
    index = json.loads(index_p.read_text())["cells"]
    out = []
    for c in cells:
        rec = index.get(c["key"])
        tob = tob_dir / f"{c['key']}.gnn.rooted.nwk"
        cell = gnn_cell_path(gnn_run, c["rep"])
        if rec is None or not tob.exists():
            out.append(f"{c['key']}: no GNN TOB staged")
        elif sha_file(tob) != rec["tob_sha256"]:
            out.append(f"{c['key']}: {tob.name} differs from what gnn_tob.json recorded")
        elif not cell.exists() or sha_file(cell) != rec["gnn_cell_sha256"]:
            out.append(f"{c['key']}: the GNN cell changed since its TOB was staged: re-run "
                       f"stage_gnn_tob.py")
    return out


def invariants(L, method: str, rooted_tob: str, rec: dict, n_eligible: int) -> list[str]:
    """``pte_common``'s three invariants, on a scored output."""
    bad = []
    k = rec.get("k_pred")
    if k is None:
        return bad
    if k > n_eligible:
        bad.append(f"k_pred {k} > {n_eligible} eligible polytomies")
    if method == "brooqs" and k != n_eligible:
        bad.append(f"BROOQS k_pred {k} != {n_eligible} eligible polytomies")
    _taxa, tob_splits = rb().splits(L, rooted_tob)
    if C.pte().bridge_splits(rec["enewick"], C.OUT) != tob_splits:
        bad.append("the output's bridge splits differ from the TOB's splits")
    return bad


def run_task(task: dict) -> dict:
    """One network x one TOB variant: both resolvers, then scoring after every clock has stopped."""
    RB = rb()
    L = RB.lib(str(C.PYLIB))
    P = C.pte()
    key, rep, variant = task["key"], int(task["rep"]), task["variant"]
    run = Path(task["run"])
    work = run / "work" / key / variant
    work.mkdir(parents=True, exist_ok=True)
    trees = C.trees_path(rep).resolve()
    tob_src = Path(task["tob_dir"]) / f"{key}.{variant}.rooted.nwk"
    rooted = tob_src.read_text().strip()
    rooted_p = work / "tob.rooted.nwk"
    rooted_p.write_text(rooted + "\n")
    tob_sha = sha_file(rooted_p)
    _parent, _len, _lab, children = L.OG.parse_newick(rooted)
    degs = sorted((len(c) for c in children if len(c) > 2), reverse=True)
    eligible = {m: sum(d >= t for d in degs) for m, t in ELIGIBLE_AT.items()}
    t_start = time.time()

    methods = {}
    for name in task["methods"]:
        out_p, mrec = work / f"{name}.enewick", work / f"{name}.json"
        if task.get("score_only"):
            # Results produced elsewhere (remote_pkg/run_remote.py), rsynced into this layout:
            # score what is there, launch nothing.  The record must be for this exact TOB.
            if not mrec.exists():
                methods[name] = {"rc": None, "wall": 0.0, "cpu": None, "rss_mb": 0.0,
                                 "timed_out": False, "not_run": True}
            else:
                methods[name] = json.loads(mrec.read_text())
                if methods[name].get("tob_sha256") != tob_sha:
                    methods[name]["tob_mismatch"] = True
            continue
        if not task["force"] and mrec.exists():
            prior = json.loads(mrec.read_text())
            # Keep a finished run, or a timeout at least as long as the one asked for now.
            if prior.get("tob_sha256") == tob_sha and (
                    (out_p.exists() and not prior.get("timed_out"))
                    or (prior.get("timed_out") and prior["timeout_s"] >= task["timeout"][name])):
                methods[name] = prior
                continue
        out_p.unlink(missing_ok=True)          # tree-qmc will not overwrite -o; it prints instead
        if name == "netcs":
            cmd = [task["tree_qmc"], "--network", "-i", trees, "--at", rooted_p.name,
                   "-o", out_p.name, *task["netcs_args"]]
        else:
            cmd = [task["brooqs"], "-i", trees, "-t", rooted_p.name, "-o", out_p.name,
                   *task["brooqs_args"]]
        m0 = time.time()
        t = P.run_timed_to(cmd, work / f"{name}.log", work, task["timeout"][name], out_p)
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
        t.update(t_start=m0, t_end=time.time(), tob_sha256=tob_sha)
        mrec.write_text(json.dumps(t, indent=1))
        methods[name] = t

    # ---- scoring, after every clock has stopped: run_blob_cpu.run_cell's
    truth = L.CAMUS.truth_of(L.CAMUS.CamusCell(C.PHASE, rep, C.ARM, C.INPUTS))
    tc = L.CN.label_clusters(truth)
    expect = set(truth.labels) | {C.OUT}
    for name, t in methods.items():
        for stale in ("error", "error_tail", "violations", "enewick"):
            t.pop(stale, None)
        out_p = work / f"{name}.enewick"
        if t.get("not_run"):
            t["error"] = "not run (score-only: no record)"
            continue
        if t.get("tob_mismatch"):
            t["error"] = "record is for a different TOB (tob_sha256 mismatch)"
            continue
        if t.get("timed_out"):
            t["error"] = f"timeout after {t['timeout_s']:.0f} s"
            continue
        if t["rc"] != 0 or not out_p.exists() or not out_p.read_text().strip():
            t.update(error=f"rc={t['rc']}", error_tail=RB.tail(work / f"{name}.log"))
            continue
        text = out_p.read_text().strip()
        t["enewick"] = text
        try:
            t.update(RB.score(L, text, tc, expect))
            t["violations"] = invariants(L, name, rooted, t, eligible[name])
        except Exception as exc:
            t["error"] = f"score: {type(exc).__name__}: {exc}"
    rec = {"key": key, "rep": rep, "variant": variant, "k": int(task["k"]), "n": C.N_TAXA,
           "trees": str(trees), "host": socket.gethostname(), "jobs": task["jobs"],
           "tob": {"newick_rooted": rooted, "source": str(tob_src), "sha256": tob_sha,
                   "blob_degrees": degs, "n_blobs": len(degs), "n_eligible": eligible,
                   "gnn_cell_sha256": task.get("gnn_cell_sha256")},
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
            groups.setdefault((rec["variant"], name), []).append(
                {**m, "k": rec["k"], "n_blobs": rec["tob"]["n_blobs"],
                 "n_elig": rec["tob"]["n_eligible"][name]})
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lines = [f"{'variant':8s} {'method':7s} {'cells':>5s} {'ok':>3s} {'unroot':>6s} {'err':>3s} "
             f"{'viol':>4s} {'k_pred med':>10s} {'elig med':>8s} {'k med':>5s} {'F1 med':>7s} "
             f"{'wall med':>8s} {'wall max':>8s} {'RSS max MB':>10s}"]
    for (variant, name), ms in sorted(groups.items()):
        ok = [m for m in ms if "error" not in m]
        f1 = [m["f1"] for m in ok if m.get("f1") is not None]
        lines.append(
            f"{variant:8s} {name:7s} {len(ms):5d} {len(ok):3d} "
            f"{sum(m.get('out_rootable') is False for m in ok):6d} {len(ms) - len(ok):3d} "
            f"{sum(bool(m.get('violations')) for m in ok):4d} "
            f"{med([m['k_pred'] for m in ok]):10.1f} {med([m['n_elig'] for m in ms]):8.1f} "
            f"{med([m['k'] for m in ms]):5.1f} {med(f1):7.4f} "
            f"{med([m['wall'] for m in ms]):8.1f} {max(m['wall'] for m in ms):8.1f} "
            f"{max(m['rss_mb'] for m in ms):10.0f}")
    errs = Counter(m.get("error", "")[:60] for ms in groups.values() for m in ms if "error" in m)
    if errs:
        lines.append("errors: " + "; ".join(f"{n} x {e}" for e, n in errs.most_common()))
    viol = Counter(v for ms in groups.values() for m in ms for v in m.get("violations") or [])
    if viol:
        lines.append("VIOLATIONS: " + "; ".join(f"{n} x {v}" for v, n in viol.most_common()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1, help="tasks run concurrently, each tool 1 thread")
    ap.add_argument("--tree-qmc", type=Path, default=TREE_QMC)
    ap.add_argument("--brooqs", type=Path, default=BROOQS)
    ap.add_argument("--methods", default=",".join(C.RESOLVERS))
    ap.add_argument("--variants", default=",".join(C.RESOLVER_VARIANTS))
    ap.add_argument("--keys", default="", help="reps, e.g. 0-4,7; default all 20")
    ap.add_argument("--netcs-args", default="", help="extra arguments for tree-qmc --network")
    ap.add_argument("--brooqs-args", default="--rooted --dist-var",
                    help="extra arguments for brooqs; the default is what the landed rows ran")
    ap.add_argument("--timeout-brooqs", type=float, default=28800.0, help="seconds per run")
    ap.add_argument("--timeout-netcs", type=float, default=28800.0, help="seconds per run")
    ap.add_argument("--tob-dir", type=Path, default=C.TOB)
    ap.add_argument("--gnn-run", type=Path, default=C.RUNS / C.GNN_RUN,
                    help="the GNN run the gnn TOBs were collapsed from (stale-TOB guard)")
    ap.add_argument("--out", type=Path, default=C.RUNS)
    ap.add_argument("--run-name", default="resolvers")
    ap.add_argument("--force", action="store_true", help="re-run resolvers already done")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="launch no tool: score the method records already under <run>/work/ "
                         "(remote_pkg results rsynced back); a missing record is an error row")
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
        raise SystemExit(f"no {C.STAGE}: run n1000_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())

    want = C.parse_reps(args.keys)
    cells = [c for c in stage["cells"] if want is None or c["rep"] in want]
    missing = [str(p) for c in cells for p in
               [C.trees_path(c["rep"]),
                *(args.tob_dir / f"{c['key']}.{v}.rooted.nwk" for v in variants)]
               if not p.exists()]
    stale = stale_problems(cells, args.tob_dir, args.gnn_run) if "gnn" in variants else []

    if args.self_check:
        RB = rb()
        L = RB.lib(str(C.PYLIB))
        rc = {Path(t).name: subprocess.run([str(t), "-h"], capture_output=True).returncode
              for t in (args.tree_qmc, args.brooqs)}
        print(f"[resolvers] opc from {L.path}")
        print(f"[resolvers] tree-qmc -h rc={rc['tree-qmc']} | brooqs -h rc={rc['brooqs']}")
        print(f"[resolvers] {len(cells)} staged networks; inputs missing: {len(missing)}; "
              f"stale GNN TOBs: {len(stale)}")
        for s in (missing + stale)[:5]:
            print(f"    {s}")
        ok = not missing and not stale and all(v == 0 for v in rc.values())
        print(f"[resolvers] self-check {'OK' if ok else 'FAILED'}")
        return 0 if ok else 1
    if missing:
        raise SystemExit(f"{len(missing)} staged inputs missing, e.g. {missing[0]}")
    if stale:
        raise SystemExit("stale GNN TOBs:\n" + "\n".join(f"  {s}" for s in stale))

    gnn_index = (json.loads((args.tob_dir / "gnn_tob.json").read_text())["cells"]
                 if "gnn" in variants else {})
    run = Path(args.out) / args.run_name
    timeout = {"brooqs": args.timeout_brooqs, "netcs": args.timeout_netcs}
    tasks = [{"key": c["key"], "rep": c["rep"], "k": c["k"], "variant": v, "run": str(run),
              "tob_dir": str(args.tob_dir), "methods": methods, "tree_qmc": str(args.tree_qmc),
              "brooqs": str(args.brooqs), "netcs_args": args.netcs_args.split(),
              "brooqs_args": args.brooqs_args.split(), "timeout": timeout,
              "gnn_cell_sha256": gnn_index.get(c["key"], {}).get("gnn_cell_sha256")
              if v == "gnn" else None,
              "force": args.force, "jobs": args.jobs, "score_only": args.score_only}
             for c in cells for v in variants]
    todo = [t for t in tasks if args.force or args.score_only or not (
        run / "cells" / f"{task_name(t['key'], t['variant'])}.json").exists()]
    print(f"[resolvers] {C.PKG.name} | {len(tasks)} tasks ({len(tasks) - len(todo)} done) | "
          f"variants {','.join(variants)} | methods {','.join(methods)} | jobs {args.jobs} | "
          f"timeouts brooqs {args.timeout_brooqs:.0f}s netcs {args.timeout_netcs:.0f}s | -> {run}",
          flush=True)
    if args.dry_run:
        for t in todo:
            extra = gnn_index.get(t["key"], {})
            print(f"  {t['key']:30s} {t['variant']:5s} k={t['k']}"
                  + (f" blobs={extra.get('blob_degrees')}" if extra else ""))
        return 0

    (run / "cells").mkdir(parents=True, exist_ok=True)
    # Score-only: the tools ran elsewhere, and their header (remote_header_*.json) came back with
    # the results; this machine's tool versions would be the wrong ones to record.
    info = ({"score_only": True, "host": socket.gethostname()} if args.score_only
            else rb().machine(args))     # tool versions, commits, the TREE-QMC patch hash
    t_run = time.perf_counter()

    def report(i: int, rec: dict) -> None:
        parts = [f"[resolvers] {i:3d}/{len(todo)} {rec['key']:30s} {rec['variant']:5s} "
                 f"k={rec['k']} blobs={rec['tob']['n_blobs']}"]
        for name, m in rec["methods"].items():
            f1 = m.get("f1")
            parts.append(f"{name} {m['wall']:7.1f}s {m['rss_mb']:6.0f}MB " + (
                f"ERR {m['error']}" if "error" in m else
                f"k={m['k_pred']} F1={'unrootable' if f1 is None else f'{f1:.4f}'}"
                + (f" VIOLATION {m['violations']}" if m.get("violations") else "")))
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
    return 0 if not crashed and "errors:" not in text and "VIOLATIONS" not in text else 3


if __name__ == "__main__":
    raise SystemExit(main())
