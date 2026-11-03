#!/usr/bin/env python3
"""ASTRAL-IV on the staged gene trees: one species tree per (size, cell) -- see README.md.

Input is exactly what the GNN reads: the first 1k / 5k / 10k lines of the staged
``inputs/n200/<RR>/est_bp1000.nwk`` (IQ-TREE's unrooted trees with ``OUT`` and branch lengths).  A
prefix's sha256 must be ``stage.json``'s ``prefix_sha256[size]``, so both methods provably read the
same trees.  ASTRAL is unrooted; its tree is rooted in Python on ``OUT`` with
``outgroup.reroot_on_outgroup`` -- the function ``prepare_gene_trees`` roots the gene trees with --
and ``OUT`` pruned, so its clusters compare with the GNN's on the same footing.

    astral4 -i <prefix> -o species.tre -t 1 --seed 233 -u 1 -r 4 -s 4 --genelength 1000

(every value ASTER 1.22's default, written out so ``run.json`` records it).  Per unit,
``runs/astral/<size>/<stem>/``: ``species.tre`` (ASTRAL's output, local posteriors as labels),
``species.rooted.nwk`` (internal labels stripped, rooted on OUT, OUT pruned), ``astral.log``, and
``run.json`` -- written last, with the binary's sha256 and version, argv, seed, input sha, wall /
CPU / peak RSS, and gate A1:

    A1   exit 0, not timed out; the leaf set is t1..t200 + OUT; binary (198 non-trivial clusters
         after rooting); the rooted tree carries exactly t1..t200

A unit is skipped when its ``run.json`` passed A1 with the same input sha, argv, binary and version.  Exits 1
if any unit fails, and when nothing was selected.

Usage:  gnn_cpu_v1/.venv/bin/python sptree_eval/run_astral.py [--sizes 1k,5k,10k] [--reps 0-49] [--jobs 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import sp_common as SP

from opc.simulate import outgroup as OG  # noqa: E402

ASTRAL = Path.home() / "miniconda3" / "envs" / "phylo-env" / "bin" / "astral4"
ARGS = ("-t", "1", "--seed", "233", "-u", "1", "-r", "4", "-s", "4", "--genelength", "1000")
ROOT = SP.RUNS / "astral"
INTERNAL_LABEL = re.compile(r"\)[^:,();]+")


def version(binary: Path) -> str:
    r = subprocess.run([str(binary), "-h"], capture_output=True, text=True)
    m = re.search(r"Version:\s*(\S+)", r.stdout + r.stderr)     # ASTER prints its help on stderr
    return m.group(1) if m else "unknown"


def run_timed(cmd, log: Path, timeout_s: float, out_path: Path) -> dict:
    """``prefix_tob_eval/pte_common.run_timed_to``: wall, and CPU and peak RSS of this child alone by
    ``wait4``, with a timeout that kills the child's process group and removes a partial output."""
    timed_out = False
    with open(log, "w") as fh:
        t0 = time.perf_counter()
        p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_s:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, ru = os.wait4(p.pid, 0)
                timed_out = True
                break
            time.sleep(0.05 if elapsed < 5 else 1.0)
        wall = time.perf_counter() - t0
    if timed_out:
        Path(out_path).unlink(missing_ok=True)
    rss = ru.ru_maxrss / 2**20 if sys.platform == "darwin" else ru.ru_maxrss / 2**10
    return {"rc": os.waitstatus_to_exitcode(status), "wall": wall, "cpu": ru.ru_utime + ru.ru_stime,
            "rss_mb": rss, "timed_out": timed_out, "timeout_s": timeout_s}


def clusters(rooted: str) -> tuple[set[frozenset[str]], frozenset[str]]:
    """Non-trivial clusters of a rooted Newick, as label sets, and its leaf set."""
    parent, _len, lab, children = OG.parse_newick(rooted)
    leaves: list = [None] * len(parent)
    for i in reversed(range(len(parent))):
        leaves[i] = (frozenset([lab[i]]) if not children[i]
                     else frozenset().union(*(leaves[c] for c in children[i])))
    full = leaves[0]
    return {s for s in leaves if 1 < len(s) < len(full)}, full


def unit(payload) -> dict:
    size, rep, prefix_sha, binary_sha, ver, timeout_s, force = payload
    stem = SP.stem(rep)
    d = ROOT / size / stem
    run_json = d / "run.json"
    m = SP.SIZE_M[size]
    staged = SP.trees_path(rep)
    if m == SP.N_LOCI:
        inp, tmp_input = staged, None
    else:
        inp = tmp_input = ROOT / "work" / f"{size}_{stem}.nwk"
    argv = [str(ASTRAL), "-i", str(inp), "-o", str(d / "species.tre"), *ARGS]
    if run_json.exists() and not force:
        old = json.loads(run_json.read_text())
        if (old.get("ok") and old["input_sha256"] == prefix_sha and old["argv"] == argv
                and old["binary_sha256"] == binary_sha and old.get("version") == ver):
            return {"size": size, "rep": rep, "skipped": True, "ok": True}
    d.mkdir(parents=True, exist_ok=True)
    problems = []
    try:
        if tmp_input is not None:
            data = SP.prefix_bytes(staged, m)
            tmp_input.parent.mkdir(parents=True, exist_ok=True)
            tmp_input.write_bytes(data)
            got = hashlib.sha256(data).hexdigest()
        else:
            got = SP.sha256(staged)
        if got != prefix_sha:
            raise RuntimeError(f"input sha256 {got[:12]} is not stage.json's {size} prefix {prefix_sha[:12]}")
        run_json.unlink(missing_ok=True)
        rec = run_timed(argv, d / "astral.log", timeout_s, d / "species.tre")
    finally:
        if tmp_input is not None:
            tmp_input.unlink(missing_ok=True)
    out = {"size": size, "rep": rep, "stem": stem, "m": m, "input": str(inp), "input_sha256": prefix_sha,
           "binary": str(ASTRAL), "binary_sha256": binary_sha, "version": ver, "argv": argv,
           "seed": 233, "threads": 1, **rec}
    if rec["rc"] != 0 or rec["timed_out"]:
        problems.append(f"A1: exit {rec['rc']}{' (timed out)' if rec['timed_out'] else ''}")
    else:
        text = (d / "species.tre").read_text().strip()
        plain = INTERNAL_LABEL.sub(")", text)
        _unrooted, leaves = clusters(plain)
        if leaves != SP.TAXA | {SP.OUT}:
            problems.append(f"A1: leaf set has {len(leaves)} labels, {len(leaves ^ (SP.TAXA | {SP.OUT}))} "
                            f"differ from t1..t200 + OUT")
        else:
            rooted = OG.reroot_on_outgroup(plain, keep_lengths=True)
            cl, full = clusters(rooted)
            if full != SP.TAXA:
                problems.append(f"A1: the rooted tree carries {len(full)} taxa")
            if len(cl) != SP.N_TRUE:
                problems.append(f"A1: {len(cl)} non-trivial clusters after rooting, want {SP.N_TRUE} (binary)")
            SP.write_atomic(d / "species.rooted.nwk", rooted + "\n")
            out["species_rooted_sha256"] = SP.sha256(d / "species.rooted.nwk")
            out["n_clusters"] = len(cl)
        out["species_sha256"] = SP.sha256(d / "species.tre")
    out["problems"] = problems
    out["ok"] = not problems
    SP.write_atomic(run_json, json.dumps(out, indent=1) + "\n")
    return {"size": size, "rep": rep, "skipped": False, "ok": out["ok"], "problems": problems,
            "wall": rec["wall"], "rss_mb": rec["rss_mb"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=",".join(t for t, _, _ in SP.SIZES))
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=12 * 3600,
                    help="seconds per run (default 12 h: 10k on rep17 passed 3 h still in the subsample stage)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not ASTRAL.exists():
        raise SystemExit(f"no ASTRAL-IV at {ASTRAL}")
    st = SP.stage()
    want = SP.parse_reps(args.reps)
    sizes = [s for s in args.sizes.split(",") if s]
    if not sizes or set(sizes) - set(SP.SIZE_M):
        raise SystemExit(f"--sizes {args.sizes!r}: expected some of {list(SP.SIZE_M)}")
    binary_sha, ver = SP.sha256(ASTRAL), version(ASTRAL)
    payloads = [(s, c["rep"], c["prefix_sha256"][s], binary_sha, ver, args.timeout, args.force)
                for s in sizes for c in st["cells"] if want is None or c["rep"] in want]
    if not payloads:
        raise SystemExit("no cells selected")
    # largest inputs first, so the long runs start early and the tail is short
    payloads.sort(key=lambda p: (-SP.SIZE_M[p[0]], p[1]))
    print(f"[astral] ASTRAL-IV {ver} ({binary_sha[:12]}) | {len(payloads)} runs over sizes "
          f"{','.join(sizes)} | jobs {args.jobs} | -> {ROOT}", flush=True)
    t0, bad, n_skip = time.perf_counter(), [], 0
    ctx = get_context("fork")
    with ctx.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(unit, payloads), 1):
            if r["skipped"]:
                n_skip += 1
                continue
            if not r["ok"]:
                bad.append(r)
            print(f"[astral] {i:4d}/{len(payloads)} {r['size']:>3s} rep{r['rep']:02d} "
                  f"{'ok' if r['ok'] else 'FAILED ' + '; '.join(r['problems'])} | {r['wall']:.0f} s, "
                  f"{r['rss_mb']:.0f} MB | {time.perf_counter() - t0:.0f} s elapsed", flush=True)
    print(f"[astral] done: {len(payloads) - len(bad)} ok ({n_skip} already done), {len(bad)} failed | "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
