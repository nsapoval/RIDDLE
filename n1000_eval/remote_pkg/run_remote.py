#!/usr/bin/env python3
"""NetCS and BROOQS on the n = 1000 trees of blobs, on a remote box -- see README.md.

This needs only the Python 3.9+ standard library; no repository code. It runs one tool per task:

    task = (network, TOB variant, method)      20 networks x {gnn[, true]} x {netcs, brooqs}

    netcs    tree-qmc --network -i <trees> --at tob.rooted.nwk -o netcs.enewick
    brooqs   brooqs -i <trees> -t tob.rooted.nwk -o brooqs.enewick --rooted --dist-var

These are byte-for-byte the commands ``n1000_eval/run_resolvers.py`` runs on the Mac. Each tool
writes into ``<out>/<key>/<variant>/{tob.rooted.nwk, <method>.log, <method>.enewick,
<method>.json}``. That is ``n1000_eval/runs/<run>/work/``'s layout, so rsyncing ``<out>/`` back
and running ``run_resolvers.py --score-only`` scores it with no tool launched. The ``.json`` record
carries the fields that driver reads: rc, wall, cpu, rss_mb, timed_out, timeout_s, tob_sha256,
note and self_s.

**Scheduling.** There is a single scheduler process and no worker pool. Tools are started with
``start_new_session`` and reaped with ``wait4``, which gives the exact CPU and peak RSS of each one.
A task is admitted while both of these hold:

* the running tasks number fewer than ``--jobs``;
* the sum of their memory estimates fits ``--mem-gb``.

The memory estimates are ``--mem-netcs`` / ``--mem-brooqs``, measured on the Mac: NetCS 6.6-12.1
GB, BROOQS 5.0-5.4 GB. NetCS runs first and longest-first (more eligible blobs first), because it is
the long pole: on the Mac it took 3.4 h on its one finished network and passed 12 h on the other
three.

**Stopping is safe.** SIGINT, SIGTERM and SIGHUP kill every running tool's whole process group,
then exit. Nothing is left running. (On the Mac, Ctrl-C on the pool-based driver orphaned 16 tools
that ran on for hours.) A killed task leaves no ``.json``, so the next start re-runs it. A finished
task, or a timeout at least as long as the one asked for now, is kept.

Usage (``$PY`` from env.sh: the tools env's Python 3.12; the box's system python3 is 3.6)
-----
    $PY run_remote.py --check                     # inputs, sha256s, tools: run this first
    $PY run_remote.py --dry-run --jobs 32
    $PY run_remote.py --jobs 32                   # the grid (inside tmux / nohup)
    $PY run_remote.py --status                    # progress from disk, any time
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"
PIN = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
METHODS = ("netcs", "brooqs")
# run_blob_cpu.py's log regexes, so ``self_s`` and ``note`` mean what they mean on the Mac.
_QMC_TIME = re.compile(r"Execution time:\s*([0-9.]+)\s*secs")
_BRQ_TIME = re.compile(r"runtime:\s+([0-9.eE+-]+)")
_REROOTED = re.compile(r"Perform rerooting on the edge above taxon (\S+)")

RUNNING: dict[int, dict] = {}      # pid -> task, for the signal handler


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def mem_total_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:                                    # macOS, for testing the package locally
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                  text=True).stdout) / 2**30
    except (OSError, ValueError):
        return 0.0


def first_float(rx, text):
    m = rx.search(text)
    return float(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------ inputs

def trees_file(args, key: str) -> Path:
    return Path(args.inputs) / f"{key}.est_bp1000.nwk"


def stage_inputs(args, man: dict, keys: list[str]) -> list[str]:
    """Decompress each network's ``.raw.nwk.gz`` and check both sha256s against the manifest.

    The decompressed file must be byte-identical to what the Mac's GNN and resolvers read.
    A ``.ok`` marker caches the check, keyed by the decompressed sha, so a restart costs seconds.
    """
    bad = []
    Path(args.inputs).mkdir(parents=True, exist_ok=True)
    for key in keys:
        cell = man["cells"][key]
        dst = trees_file(args, key)
        ok = dst.with_name(dst.name + ".ok")
        if dst.exists() and ok.exists() and ok.read_text().strip() == cell["trees_sha256"]:
            continue
        gz = Path(args.dataset) / "estimated_bp1000" / f"{key}.raw.nwk.gz"
        if not gz.exists():
            bad.append(f"{key}: no {gz}")
            continue
        got = sha_file(gz)
        if got != cell["raw_gz_sha256"]:
            bad.append(f"{key}: {gz.name} sha256 {got[:12]} != manifest {cell['raw_gz_sha256'][:12]}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        with gzip.open(gz, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
        got = sha_file(tmp)
        if got != cell["trees_sha256"]:
            tmp.unlink()
            bad.append(f"{key}: decompressed sha256 {got[:12]} != manifest {cell['trees_sha256'][:12]}")
            continue
        tmp.replace(dst)
        ok.write_text(got + "\n")
        log(f"staged {dst.name} ({dst.stat().st_size / 2**20:.0f} MB), sha256 verified")
    return bad


def check_tobs(man: dict, keys: list[str], variants: list[str]) -> list[str]:
    bad = []
    for key in keys:
        for v in variants:
            p = HERE / "tobs" / f"{key}.{v}.rooted.nwk"
            want = man["cells"][key]["tobs"][v]["sha256"]
            if not p.exists():
                bad.append(f"{key}: no {p.name}")
            elif sha_file(p) != want:
                bad.append(f"{key}: {p.name} sha256 differs from the manifest")
    return bad


def tool_info(args) -> dict:
    def run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr).strip()
        except Exception as exc:                                     # noqa: BLE001
            return f"ERROR {exc}"
    info = {"host": socket.gethostname(), "python": sys.version.split()[0],
            "nproc": os.cpu_count(), "mem_total_gb": round(mem_total_gb(), 1),
            "tree_qmc": args.tree_qmc, "brooqs": args.brooqs,
            "tree_qmc_version": run([args.tree_qmc, "--version"])[:200],
            "tree_qmc_sha256": sha_file(Path(args.tree_qmc)) if Path(args.tree_qmc).exists() else None}
    qmc_src = Path(args.tree_qmc).resolve().parent.parent
    if (qmc_src / ".git").exists():
        info["tree_qmc_commit"] = run(["git", "-C", str(qmc_src), "rev-parse", "HEAD"])
        # raw bytes, as the manifest hashes them (a stripped diff is not the patch: CRLF context)
        patch = subprocess.run(["git", "-C", str(qmc_src), "diff"], capture_output=True).stdout
        info["tree_qmc_patch_sha256"] = hashlib.sha256(patch).hexdigest() if patch else ""
    py = Path(args.brooqs).parent / "python"
    if py.exists():
        info["brooqs_env"] = run([str(py), "-c", "import sys; from importlib.metadata import version as v; "
                                  "print(sys.version.split()[0], *(f'{p}=={v(p)}' for p in "
                                  "('brooqs', 'numpy', 'treeswift')))"])
    return info


# ------------------------------------------------------------------------------------------- tasks

def task_dir(args, t: dict) -> Path:
    return Path(args.out) / t["key"] / t["variant"]


def is_done(args, t: dict) -> bool:
    d = task_dir(args, t)
    rec_p, out_p = d / f"{t['method']}.json", d / f"{t['method']}.enewick"
    if not rec_p.exists():
        return False
    rec = json.loads(rec_p.read_text())
    if rec.get("tob_sha256") != t["tob_sha256"]:
        return False
    if rec.get("timed_out"):
        return rec["timeout_s"] >= t["timeout"]
    if rec.get("rc") == 0 and out_p.exists() and out_p.stat().st_size > 0:
        return True
    return not args.retry_failed


def launch(args, t: dict) -> subprocess.Popen:
    d = task_dir(args, t)
    d.mkdir(parents=True, exist_ok=True)
    tob = d / "tob.rooted.nwk"
    shutil.copyfile(HERE / "tobs" / f"{t['key']}.{t['variant']}.rooted.nwk", tob)
    out_p = d / f"{t['method']}.enewick"
    out_p.unlink(missing_ok=True)          # tree-qmc will not overwrite -o
    (d / f"{t['method']}.json").unlink(missing_ok=True)
    trees = str(trees_file(args, t["key"]).resolve())
    if t["method"] == "netcs":
        cmd = [args.tree_qmc, "--network", "-i", trees, "--at", tob.name, "-o", out_p.name,
               *args.netcs_args.split()]
    else:
        cmd = [args.brooqs, "-i", trees, "-t", tob.name, "-o", out_p.name,
               *args.brooqs_args.split()]
    fh = open(d / f"{t['method']}.log", "w")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=d,
                         env={**os.environ, **PIN}, start_new_session=True)
    fh.close()
    t.update(cmd=" ".join(cmd), t0=time.perf_counter(), t_start=time.time(), proc=p)
    return p


def finish(args, t: dict, status: int, ru, timed_out: bool) -> dict:
    d = task_dir(args, t)
    out_p = d / f"{t['method']}.enewick"
    if timed_out:
        out_p.unlink(missing_ok=True)
    text = (d / f"{t['method']}.log").read_text(errors="replace")
    rec = {"rc": os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode")
           else status, "wall": time.perf_counter() - t["t0"], "cpu": ru.ru_utime + ru.ru_stime,
           "user": ru.ru_utime, "sys": ru.ru_stime,
           "rss_mb": ru.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10),
           "cmd": t["cmd"], "timed_out": timed_out, "timeout_s": t["timeout"],
           "t_start": t["t_start"], "t_end": time.time(), "tob_sha256": t["tob_sha256"],
           "host": socket.gethostname(), "runner": "remote_pkg/run_remote.py"}
    if t["method"] == "netcs":
        rec["self_s"] = first_float(_QMC_TIME, text)
        hit = _REROOTED.search(text)
        rec["note"] = (f"rerooted:{hit.group(1)}" if hit else
                       "rooting-valid" if "rooting is valid" in text else
                       "not-semi-directed" if "not a valid semi-directed" in text else "")
    else:
        rec["self_s"] = first_float(_BRQ_TIME, text)
        rec["note"] = f"swaps:{text.count('[hybrid swap] reason')}"
    tmp = d / f"{t['method']}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(d / f"{t['method']}.json")
    return rec


def kill_all(signum=None, _frame=None):
    for pid in list(RUNNING):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid, t in list(RUNNING.items()):
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        (task_dir(ARGS, t) / f"{t['method']}.enewick").unlink(missing_ok=True)
    if signum is not None:
        log(f"signal {signum}: killed {len(RUNNING)} running tool(s) and their process groups; "
            f"they re-run on the next start")
        sys.exit(130)


def build_tasks(args, man: dict, keys: list[str], variants: list[str], methods: list[str]) -> list[dict]:
    timeout = {"netcs": args.timeout_netcs, "brooqs": args.timeout_brooqs}
    mem = {"netcs": args.mem_netcs, "brooqs": args.mem_brooqs}
    tasks = []
    for key in keys:
        for v in variants:
            tob = man["cells"][key]["tobs"][v]
            for m in methods:
                tasks.append({"key": key, "variant": v, "method": m, "tob_sha256": tob["sha256"],
                              "eligible": tob["n_eligible"][m], "timeout": timeout[m],
                              "mem": mem[m]})
    # NetCS first (the long pole), then most eligible blobs first; BROOQS fills around them.
    tasks.sort(key=lambda t: (t["method"] != "netcs", -t["eligible"], t["key"], t["variant"]))
    return tasks


def status(args, tasks: list[dict]) -> str:
    rows = []
    for t in tasks:
        d = task_dir(args, t)
        rec_p = d / f"{t['method']}.json"
        if rec_p.exists():
            r = json.loads(rec_p.read_text())
            st = ("timeout" if r.get("timed_out") else "ok" if r.get("rc") == 0 else f"rc={r.get('rc')}")
            rows.append((t, st, r["wall"] / 3600, r["rss_mb"] / 1024))
        elif (d / f"{t['method']}.log").exists():
            age = (time.time() - (d / f"{t['method']}.log").stat().st_mtime) / 60
            rows.append((t, f"started, no record: running or killed (log {age:.0f} min old)", None, None))
        else:
            rows.append((t, "pending", None, None))
    lines = [f"{'key':30s} {'var':5s} {'method':7s} {'elig':>4s}  {'status':52s} {'wall h':>7s} {'RSS GB':>7s}"]
    for t, st, h, g in rows:
        lines.append(f"{t['key']:30s} {t['variant']:5s} {t['method']:7s} {t['eligible']:4d}  {st:52s} "
                     f"{'' if h is None else f'{h:7.2f}'} {'' if g is None else f'{g:7.1f}'}")
    from collections import Counter
    c = Counter(st.split(" ")[0] for _t, st, _h, _g in rows)
    lines.append("totals: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    return "\n".join(lines)


ARGS = None


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("N1000_DATASET", "/shared/ns58/rl-net-opt/l2_n1000_gt50k_v1"),
                    help="the dataset directory holding estimated_bp1000/<key>.raw.nwk.gz (%(default)s)")
    ap.add_argument("--inputs", default=str(HERE / "inputs"),
                    help="where the decompressed gene trees go, ~330 MB each (%(default)s)")
    ap.add_argument("--out", default=str(HERE / "out"), help="results; rsync this back (%(default)s)")
    ap.add_argument("--tree-qmc", default=os.environ.get("TREE_QMC", str(HERE / "tools" / "TREE-QMC" / "build" / "tree-qmc")))
    ap.add_argument("--brooqs", default=os.environ.get("BROOQS", str(HERE / "tools" / ".venv-brooqs" / "bin" / "brooqs")))
    ap.add_argument("--variants", default="gnn", help="gnn, true or gnn,true (default gnn)")
    ap.add_argument("--methods", default="netcs,brooqs")
    ap.add_argument("--keys", default="", help="reps, e.g. 0-4,7; default all 20")
    ap.add_argument("--jobs", type=int, default=16, help="tools running at once (keep <= 48 on the shared box)")
    ap.add_argument("--mem-gb", type=float, default=0.0,
                    help="memory budget for running tools; default 85%% of MemTotal")
    ap.add_argument("--mem-netcs", type=float, default=14.0, help="GB reserved per NetCS run (Mac max 12.1)")
    ap.add_argument("--mem-brooqs", type=float, default=6.0, help="GB reserved per BROOQS run (Mac max 5.4)")
    ap.add_argument("--timeout-netcs", type=float, default=7 * 86400.0, help="seconds (default 7 days)")
    ap.add_argument("--timeout-brooqs", type=float, default=2 * 86400.0, help="seconds (default 2 days)")
    ap.add_argument("--netcs-args", default="")
    ap.add_argument("--brooqs-args", default="--rooted --dist-var", help="what every landed row ran")
    ap.add_argument("--retry-failed", action="store_true", help="re-run tools that exited non-zero")
    ap.add_argument("--check", action="store_true", help="stage + verify inputs and tools, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    ARGS = args

    man = json.loads(MANIFEST.read_text())
    want = None
    if args.keys:
        want = set()
        for part in args.keys.split(","):
            lo, _, hi = part.partition("-")
            want.update(range(int(lo), int(hi or lo) + 1))
    keys = [k for k, c in sorted(man["cells"].items()) if want is None or c["rep"] in want]
    variants = [v for v in args.variants.split(",") if v]
    methods = [m for m in args.methods.split(",") if m]
    if set(variants) - {"gnn", "true"} or set(methods) - set(METHODS):
        raise SystemExit("variants must be gnn/true, methods netcs/brooqs")
    tasks = build_tasks(args, man, keys, variants, methods)

    if args.status:
        print(status(args, tasks))
        return 0

    budget = args.mem_gb or 0.85 * mem_total_gb()
    if args.jobs > 48:
        log("WARNING: --jobs > 48: on the shared box a per-user process/thread limit has hung "
            "pools before (CLAUDE.md); 48 is the tested ceiling")
    bad = check_tobs(man, keys, variants)
    for tool in (args.tree_qmc, args.brooqs):
        if not (Path(tool).exists() and os.access(tool, os.X_OK)):
            bad.append(f"tool not executable: {tool}  (README: Tools)")
    if bad:
        print("\n".join(bad))
        return 1
    todo = [t for t in tasks if not is_done(args, t)]
    log(f"{len(tasks)} tasks ({len(tasks) - len(todo)} done) | {len(keys)} networks x "
        f"{variants} x {methods} | jobs {args.jobs} | memory budget {budget:.0f} GB "
        f"(netcs {args.mem_netcs:g}, brooqs {args.mem_brooqs:g}) | timeouts netcs "
        f"{args.timeout_netcs / 3600:g} h, brooqs {args.timeout_brooqs / 3600:g} h")
    if args.dry_run:
        for t in todo:
            print(f"  {t['key']:30s} {t['variant']:5s} {t['method']:7s} eligible={t['eligible']}")
        return 0

    bad = stage_inputs(args, man, keys)
    if bad:
        print("\n".join(bad))
        return 1
    info = tool_info(args)
    info["expected"] = man["tools"]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"remote_header_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps({**info, "args": vars(args)}, indent=1, default=str))
    log(f"tree-qmc: {info['tree_qmc_version'][:80]!r} commit {info.get('tree_qmc_commit', '?')[:12]} "
        f"patch {info.get('tree_qmc_patch_sha256', '?')[:12]} (Mac: {man['tools']['tree_qmc_commit'][:12]} "
        f"patch {man['tools']['tree_qmc_patch_sha256'][:12]})")
    log(f"brooqs env: {info.get('brooqs_env', '?')} (Mac: {man['tools']['brooqs_env']})")
    if args.check:
        log("check OK: TOBs verified, gene trees staged and verified, tools present")
        return 0

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, kill_all)
    pending = list(todo)
    n_done = 0
    while pending or RUNNING:
        # admit
        used = sum(t["mem"] for t in RUNNING.values())
        for t in list(pending):
            if len(RUNNING) >= args.jobs:
                break
            if used + t["mem"] > budget and RUNNING:
                continue
            p = launch(args, t)
            RUNNING[p.pid] = t
            used += t["mem"]
            pending.remove(t)
            log(f"start  {t['key']} {t['variant']} {t['method']} (eligible {t['eligible']}) pid {p.pid} "
                f"| running {len(RUNNING)}, reserved {used:.0f}/{budget:.0f} GB")
        # reap / time out
        for pid, t in list(RUNNING.items()):
            got, status_, ru = os.wait4(pid, os.WNOHANG)
            timed_out = False
            if not got and time.perf_counter() - t["t0"] > t["timeout"]:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                got, status_, ru = os.wait4(pid, 0)
                timed_out = True
            if got:
                del RUNNING[pid]
                rec = finish(args, t, status_, ru, timed_out)
                n_done += 1
                st = "TIMEOUT" if timed_out else ("ok" if rec["rc"] == 0 else f"FAILED rc={rec['rc']}")
                log(f"finish {t['key']} {t['variant']} {t['method']}: {st} | {rec['wall'] / 3600:.2f} h "
                    f"{rec['rss_mb'] / 1024:.1f} GB | {n_done}/{len(todo)} done, {len(pending)} pending")
        time.sleep(2)
    log("all tasks finished")
    print(status(args, tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
