#!/usr/bin/env python3
"""NetCS and BROOQS on the 500-tree subsample trees of blobs -- one task per (draw, cell, TOB
variant, resolver).  See README.md, "Resolvers on the subsample TOBs".

Inputs
------
* **Gene trees:** the staged subsample, ``inputs/<ds>/d<k>/.../<arm file>`` (``MANIFEST.tsv``).
  Every line carries OUT, on netcs too (``stage.py``: the true-root OUT, an oracle).
* **Trees of blobs:** ``runs/tob/<ds>/d<k>/<cell>.{raw.tob,tob,json}``, TREE-QMC 4.1.5 ``--blob`` on
  exactly those files (built on the Linux box; ``json['input']`` / ``sha256_input`` name them).
  ``.raw.tob`` is TREE-QMC's output as written; ``.tob`` is rooted on OUT, OUT removed, with the A63
  root-branch contraction applied -- the landed l2reg/l3reg "fixed" TOB.

The protocol -- the landed camus / l2reg / l3reg one, on all four corpora
------------------------------------------------------------------------
::

    raw     tob.nwk = camus_nets.root_tob_on_outgroup(<cell>.raw.tob)      (l2reg_eval/stage.py:183)
    fixed   tob.nwk = (OUT,<cell>.tob);                                      (l2reg_eval/stage.py:184)
    netcs   tree-qmc --network -i <gene trees> --at tob.nwk -o out.enewick
    brooqs  brooqs -i <gene trees> -t tob.nwk -o out.enewick --rooted --dist-var

``raw`` runs on every corpus, ``fixed`` on l2reg and l3reg only: exactly the landed labels
(``BROOQS``, ``NetCS``, ``BROOQS (fixed TOB)``, ``NetCS (fixed TOB)``).  **netcs is run rooted**
(decided 2026-09-22): its gene trees and TOBs carry the oracle OUT, so these rows are ``BROOQS*`` /
``NetCS*`` -- the same root information the GNN's netcs subsample rows had, and **not** the landed
netcs resolver protocol, which ran unrooted with no OUT anywhere (``netcs_eval/run_resolvers.py``).

Scoring is the landed camus/l2reg/l3reg drivers' own on every corpus: ``run_blob_cpu.score`` against
``C(N_true)`` with taxon set truth + OUT, after every clock has stopped.  ``pte_bench.score`` is not
used: its netcs branch places OUT by the oracle, and these outputs already carry one.

Machinery borrowed, not copied
------------------------------
``prefix_tob_eval``'s ``pte_common`` (the timed runner with its process-group timeout, content ids,
the tool fingerprint, eligibility rules) and ``pte_bench`` (the landed modules loaded by path, the
truths, the invariants).  Their ``pte_`` names cannot collide with this package's ``common``, and
``pte_bench.load_by_path`` restores ``sys.modules['common']`` after each load.  ``run_task`` is this
file's own, modelled on ``prefix_tob_eval/run_resolvers.py:run_task``.

Gates (``--check``; also run before any grid)
---------------------------------------------
    S1  every MANIFEST row has its three TOB files; json input == MANIFEST path, sha256_input ==
        sha256_sub == the staged file's SHA-256 now; n_trees == m == 500
    S2  leaf sets: raw TOB == first/last gene tree == truth + OUT; fixed TOB == truth.  Both
        wrapped trees have OUT as one of two root children, and wrapping moves no split
    S3  A63: fixed splits == raw splits, minus exactly the contracted root-branch split where the
        JSON says TREE-QMC kept one
    S4  every cell maps 1:1 to a landed condition, whose truth (pte_bench.truth, the object the
        landed resolver rows were scored on) equals the staged copy's, by label cluster and k
    S5  reported, not gated: draws whose raw TOB equals the landed full-data one by splits, and
        cells whose five draws all agree

``--parity`` pushes landed full-data TOBs through this driver on the landed gene trees: our
wrapping must rebuild the landed ``tob.nwk`` byte for byte, BROOQS must rebuild the landed eNewick
byte for byte and its score, NetCS its k and (where both are OUT-rootable) its score.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root; long runs from a terminal, caffeinate -i)
-----
    python subsample_eval/run_res.py --check
    python subsample_eval/run_res.py --parity --jobs 6
    python subsample_eval/run_res.py --smoke --jobs 8 --out /tmp/res_smoke
    python subsample_eval/run_res.py --dataset all --dry-run --jobs 12
    python subsample_eval/run_res.py --dataset camus,l2reg,l3reg --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200
    python subsample_eval/run_res.py --dataset all --summary-only
"""

from __future__ import annotations

import common as C  # first: BLAS pin, gnn_cpu_v1/pylib at sys.path[0]

import sys

_PTE = str(C.REPO / "prefix_tob_eval")
if _PTE not in sys.path:
    sys.path.insert(1, _PTE)

import pte_common as P  # noqa: E402
import pte_bench as B  # noqa: E402

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import functools  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

TOBS = C.RUNS / "tob"
RES = C.RUNS / "res"
BROOQS_ARGS = ("--rooted", "--dist-var")
VARIANTS = {"camus": ("raw",), "l2reg": ("raw", "fixed"), "l3reg": ("raw", "fixed"),
            "netcs": ("raw",)}
DEFAULT_TIMEOUT = 7200.0
#: CPU of a 500-tree run over the landed 1,000-tree run of the same condition, for the LPT order and
#: the --dry-run projection.  Set from the smoke (README); only the order and the estimate use it.
RATIO = {"brooqs": 0.55, "netcs": 0.8}     # the 2026-09-23 grid measured 0.49-0.58 and 0.66-0.89
#: Bumped whenever scoring changes; ``--rescore`` re-scores saved outputs.
SCORE_VERSION = 1
#: One d0 cell per dataset and size, both variants where there are two.
SMOKE = {"camus": ("cam50_rep00_cam_g500", "cam200_rep00_cam_g500"),
         "l2reg": ("l2reg_n200_rep00_est_bp1000",), "l3reg": ("l3reg_n200_rep00_est_bp1000",),
         "netcs": ("netcs_L1_n50_rep10_netcs_ils1.0", "netcs_L2_n200_rep00_netcs_ils1.0")}


def label(ds: str, variant: str, resolver: str) -> str:
    """The row label: the landed spelling, with netcs's oracle ``*`` (module docstring)."""
    name = P.RESOLVER_NAMES[resolver] + ("*" if ds == "netcs" else "")
    return name + (" (fixed TOB)" if variant == "fixed" else "")


@functools.lru_cache(maxsize=None)
def corpus(ds: str) -> P.Corpus:
    """``pte_common.CORPORA[ds]`` with this round's conventions: OUT carried everywhere, BROOQS
    ``--rooted --dist-var``.  Only ``wrap_out`` is read downstream (``pte_bench.reference_taxon``)."""
    return dataclasses.replace(P.CORPORA[ds], wrap_out=True, brooqs_args=BROOQS_ARGS,
                               oracle_root=ds == "netcs")


# ----------------------------------------------------------------------------------- trees of blobs

def wrap_raw(text: str) -> str:
    """``l2reg_eval/stage.py:183``: TREE-QMC's tree of blobs re-rooted on OUT's pendant edge."""
    return B.lib().CN.root_tob_on_outgroup(text.strip())


def wrap_fixed(text: str) -> str:
    """``l2reg_eval/stage.py:184``: the corrected tree (rooted on OUT, OUT removed) under OUT."""
    return f"({C.OUT},{text.strip().rstrip(';')});"


WRAP = {"raw": (".raw.tob", wrap_raw), "fixed": (".tob", wrap_fixed)}


def tob_paths(ds: str, draw: int, cell: str) -> dict[str, Path]:
    base = TOBS / ds / f"d{draw}" / cell
    return {"json": base.with_name(cell + ".json"), "raw": base.with_name(cell + ".raw.tob"),
            "fixed": base.with_name(cell + ".tob")}


def leaves(text: str) -> set[str]:
    _p, _l, lab, children = P.parse(text)
    return {lab[i] for i in range(len(lab)) if not children[i]}


def root_children_ok(text: str) -> bool:
    """Two root children, one of them the leaf OUT: the shape ``--rooted`` trusts."""
    parent, _l, lab, children = P.parse(text)
    kids = children[0]
    return len(kids) == 2 and any(lab[k] == C.OUT and not children[k] for k in kids)


# ------------------------------------------------------------------------------------------- cells

def manifest(ds: str) -> list[dict]:
    rows = C.read_tsv(C.DATASETS[ds].manifest)
    for r in rows:
        r["draw"], r["rep"], r["m"] = int(r["draw"]), int(r["rep"]), int(r["m"])
    return rows


@functools.lru_cache(maxsize=None)
def cond_key(ds: str, phase: str, rep: int, arm: str) -> str:
    return B.key_map(ds)[(phase, rep, arm)]


@functools.lru_cache(maxsize=None)
def landed_cpu(ds: str) -> dict[tuple[str, str, str], float]:
    """(condition key, variant, resolver) -> CPU seconds of the landed 1,000-tree run."""
    out: dict[tuple[str, str, str], float] = {}
    if ds == "camus":
        for r in C.read_tsv(C.REPO / "blob_cpu_v1" / "runs" / "C4537CLFY0_j8" / "timings.tsv"):
            key = cond_key(ds, r["phase"], int(r["rep"]), r["arm"]) if (
                r["phase"], int(r["rep"]), r["arm"]) in B.key_map(ds) else None
            if key is None:          # cam25 is recorded under the other arm name there
                key = next((k for (ph, rp, _a), k in B.key_map(ds).items()
                            if ph == r["phase"] and rp == int(r["rep"])), None)
            if key and r["method_cpu"]:
                out[(key, "raw", r["method"])] = float(r["method_cpu"])
        return out
    cells = C.REPO / f"{ds}_eval" / "runs" / "resolvers" / "cells"
    for f in cells.glob("*.json"):
        rec = json.loads(f.read_text())
        variant = rec.get("variant", "raw")
        for m, t in rec["methods"].items():
            if t.get("cpu") is not None:
                out[(rec["key"], variant, m)] = float(t["cpu"])
    return out


def make_task(ds: str, row: dict, variant: str, resolver: str, tob_text: str) -> dict:
    cell, draw = row["cell"], row["draw"]
    key = cond_key(ds, row["phase"], row["rep"], row["arm"])
    argv = list(BROOQS_ARGS) if resolver == "brooqs" else []
    spec = {"dataset": ds, "key": f"d{draw}/{cell}", "trees_sha": row["sha256_sub"],
            "tob_sha": P.sha(tob_text), "resolver": resolver, "argv": argv,
            "tool_fp": P.tool_fp()}
    rule = P.rule_of(resolver, None)
    stats = P.tob_stats(tob_text)
    land = landed_cpu(ds).get((key, variant, resolver))
    proj = RATIO[resolver] * land if land is not None else P.brooqs_cpu(stats, "brooqs4")
    return {**spec, "task_id": P.task_id(spec), "draw": draw, "cell": cell, "cond": key,
            "phase": row["phase"], "rep": row["rep"], "arm": row["arm"], "variants": [variant],
            "method": resolver, "rule": rule, "n_elig": stats["elig"][rule],
            "trees": str(C.HERE / row["path"]), "proj_cpu": round(proj, 1),
            "landed_cpu": land, "tob_text": tob_text}


def build_tasks(ds: str, cells: set[str] | None = None, draws: set[int] | None = None,
                resolvers=P.RESOLVERS) -> list[dict]:
    """Every task for ``ds``.  Two variants whose ``tob.nwk`` is byte-identical are one task (same
    content id), serving both labels."""
    tasks: dict[str, dict] = {}
    for row in manifest(ds):
        if (cells and row["cell"] not in cells) or (draws is not None and row["draw"] not in draws):
            continue
        paths = tob_paths(ds, row["draw"], row["cell"])
        for variant in VARIANTS[ds]:
            text = WRAP[variant][1](paths[variant].read_text())
            for r in resolvers:
                t = make_task(ds, row, variant, r, text)
                if t["task_id"] in tasks:
                    tasks[t["task_id"]]["variants"].append(variant)
                else:
                    tasks[t["task_id"]] = t
    return list(tasks.values())


# ------------------------------------------------------------------------------------------- a task

def run_task(task: dict) -> dict:
    """One resolver on one tree of blobs, then scoring after the clock has stopped.  Top-level so
    a spawned worker can unpickle it."""
    RB = B.rb()
    run = Path(task["run_dir"])
    work = run / "work" / f"d{task['draw']}" / task["cell"] / f"{task['variants'][0]}.{task['method']}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "tob.nwk").write_text(task["tob_text"] + "\n")
    out_p, log = work / "out.enewick", work / "run.log"
    trees = Path(task["trees"]).resolve()
    if task["resolver"] == "netcs":
        cmd = [P.TREE_QMC, "--network", "-i", trees, "--at", "tob.nwk", "-o", out_p.name, *task["argv"]]
        timeout = task["timeout_netcs"]
    else:
        cmd = [P.BROOQS, "-i", trees, "-t", "tob.nwk", "-o", out_p.name, *task["argv"]]
        timeout = task["timeout_brooqs"]
    t0 = time.time()
    t = P.run_timed_to(cmd, log, work, timeout, out_p)
    text = log.read_text(errors="replace")
    if task["resolver"] == "netcs":
        t["self_s"] = RB.first_float(RB._QMC_TIME, text)
        hit = RB._REROOTED.search(text)
        t["note"] = (f"rerooted:{hit.group(1)}" if hit else
                     "rooting-valid" if "rooting is valid" in text else
                     "not-semi-directed" if "not a valid semi-directed" in text else "")
    else:
        t["self_s"] = RB.first_float(RB._BRQ_TIME, text)
        t["note"] = f"swaps:{text.count('[hybrid swap] reason')}"
    rec = {k: v for k, v in task.items() if k != "tob_text"}
    rec.update(work=str(work.relative_to(run)), timing=t, host=socket.gethostname(), t_start=t0,
               t_end=time.time(), score_version=SCORE_VERSION)
    rec.update(score_output(task, task["tob_text"], out_p, t))
    P.write_json(run / "tasks" / f"{task['task_id']}.json", rec)
    return rec


def score_output(task: dict, tob_text: str, out_p: Path, t: dict) -> dict:
    ds = task["dataset"]
    if t["timed_out"]:
        return {"status": "timeout"}
    if t["rc"] != 0 or not out_p.exists() or not out_p.read_text().strip():
        return {"status": f"error: rc={t['rc']}", "error_tail": B.rb().tail(out_p.parent / "run.log")}
    text = out_p.read_text().strip()
    out = {"enewick": text}
    L = B.lib()
    truth = B.truth(ds, B.conditions(ds)[task["cond"]])
    expect = set(truth.labels) | {C.OUT}
    try:
        s = B.rb().score(L, text, L.CN.label_clusters(truth), expect)
    except Exception as exc:
        return {**out, "status": f"error: score: {type(exc).__name__}: {exc}"}
    out["score"] = s
    out["status"] = "ok" if s["out_rootable"] else "out_not_rootable"
    out["invariants"] = B.invariants(corpus(ds), tob_text, text, task["rule"], int(s["k_pred"]),
                                     expect)
    out["violations"] = B.hard_violations(out["invariants"], task["resolver"])
    return out


def rescore(rec_path: Path) -> dict:
    rec = json.loads(rec_path.read_text())
    work = rec_path.parent.parent / rec["work"]
    tob_text = (work / "tob.nwk").read_text().strip()
    for k in ("enewick", "score", "invariants", "violations", "error_tail"):
        rec.pop(k, None)
    rec.update(score_output(rec, tob_text, work / "out.enewick", rec["timing"]))
    rec["score_version"] = SCORE_VERSION
    P.write_json(rec_path, rec)
    return rec


# ------------------------------------------------------------------------------------- scheduling

def lpt_makespan(costs: list[float], jobs: int) -> float:
    heap = [0.0] * max(1, jobs)
    for c in sorted(costs, reverse=True):
        heapq.heapreplace(heap, heap[0] + c)
    return max(heap)


def task_file(t: dict) -> Path:
    return Path(t["run_dir"]) / "tasks" / f"{t['task_id']}.json"


def pending(tasks: list[dict], force: bool) -> tuple[list[dict], list[dict]]:
    """(to run, done).  A ``timeout`` re-runs when a longer timeout is asked for; anything else is
    final unless ``--force``."""
    todo, done = [], []
    for t in tasks:
        f = task_file(t)
        if f.exists() and not force:
            rec = json.loads(f.read_text())
            want = t["timeout_netcs"] if t["resolver"] == "netcs" else t["timeout_brooqs"]
            if not (rec.get("status") == "timeout" and (rec["timing"].get("timeout_s") or 0) < want):
                done.append(rec)
                continue
        todo.append(t)
    return todo, done


def execute(todo: list[dict], jobs: int) -> list[dict]:
    todo = sorted(todo, key=lambda t: -t["proj_cpu"])                  # LPT
    out: list[dict] = []
    if not todo:
        return out
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=ctx) as ex:
        futs = {ex.submit(run_task, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:                                    # a driver bug, not a tool error
                rec = {**{k: v for k, v in t.items() if k != "tob_text"},
                       "status": f"crash: {type(exc).__name__}: {exc}"}
            out.append(rec)
            s = rec.get("score", {})
            f1 = "  -  " if s.get("f1") is None else f"{s['f1']:.3f}"
            print(f"[run] {i}/{len(todo)} {t['dataset']} d{t['draw']} {t['cell']:36s} "
                  f"{'+'.join(t['variants']):9s} {t['method']:6s} {rec['status'][:24]:24s} "
                  f"k {s.get('k_pred', '')!s:>3}/{t['n_elig']:<3} f1 {f1} "
                  f"cpu {rec.get('timing', {}).get('cpu', 0):7.1f}s (proj {t['proj_cpu']:.0f}) | "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return out


def summarise(recs: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["dataset"], "+".join(r.get("variants", [])), r["method"])].append(r)
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lines = [f"{'dataset':6s} {'variant':9s} {'method':6s} {'n':>5s} {'ok':>5s} {'unroot':>6s} "
             f"{'t/out':>5s} {'error':>5s} {'k med':>6s} {'elig':>5s} {'k==el':>8s} {'split':>8s} "
             f"{'viol':>4s} {'F1 med':>7s} {'cpu med':>8s} {'CPU h':>7s} {'/landed':>7s}"]
    for (ds, variant, method), rs in sorted(groups.items()):
        st = Counter("error" if r["status"].startswith(("error", "crash")) else r["status"] for r in rs)
        sc = [r for r in rs if "invariants" in r]
        kk = [r["score"]["k_pred"] for r in sc]
        el = [r["invariants"]["n_elig"] for r in sc]
        eq = sum(1 for r in sc if r["invariants"]["k_eq_elig"])
        sp = sum(1 for r in sc if r["invariants"].get("splits_equal"))
        viol = sum(1 for r in rs if r.get("violations"))
        f1 = [r["score"]["f1"] for r in sc if r["score"].get("f1") is not None]
        cpu = [r["timing"]["cpu"] for r in rs if "timing" in r]
        ratio = [r["timing"]["cpu"] / r["landed_cpu"] for r in rs
                 if "timing" in r and r.get("landed_cpu")]
        lines.append(f"{ds:6s} {variant:9s} {method:6s} {len(rs):5d} {st['ok']:5d} "
                     f"{st['out_not_rootable']:6d} {st['timeout']:5d} {st['error']:5d} "
                     f"{med(kk):6.1f} {med(el):5.1f} {eq:4d}/{len(sc):<4d}{sp:4d}/{len(sc):<4d}"
                     f"{viol:4d} {med(f1):7.3f} {med(cpu):8.1f} {sum(cpu) / 3600:7.2f} {med(ratio):7.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- gates

def read_staged(path: Path, want_hash: bool) -> tuple[str | None, list[set[str]]]:
    """The staged file's SHA-256 (one read) and the leaf sets of its first and last tree."""
    data = path.read_bytes()
    lines = [ln for ln in data.decode().splitlines() if ln.strip()]
    return (C.sha256_bytes(data) if want_hash else None), [leaves(lines[0]), leaves(lines[-1])]


def check(datasets: list[str], hash_files: bool = True) -> tuple[bool, dict]:
    """S1-S5 (module docstring).  Returns (pass, report)."""
    L = B.lib()
    problems: list[str] = []
    report: dict = {"checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "datasets": {}}
    t0 = time.perf_counter()
    for ds in datasets:
        rows = manifest(ds)
        stats = Counter()
        truths: dict[tuple, object] = {}
        conds: dict[tuple, str] = {}
        full_splits = full_tob_splits(ds)
        draw_splits: dict[str, list[frozenset]] = defaultdict(list)
        for row in rows:
            where = f"{ds} d{row['draw']} {row['cell']}"
            paths = tob_paths(ds, row["draw"], row["cell"])
            # S1 -- provenance
            if not all(p.exists() for p in paths.values()):
                problems.append(f"S1 {where}: TOB files missing")
                continue
            meta = json.loads(paths["json"].read_text())
            if meta["input"] != row["path"] or meta["sha256_input"] != row["sha256_sub"]:
                problems.append(f"S1 {where}: TOB built from {meta['input']} "
                                f"({meta['sha256_input'][:12]}), MANIFEST says {row['path']}")
            if int(meta["n_trees"]) != row["m"] or row["m"] != C.N_SUB:
                problems.append(f"S1 {where}: n_trees {meta['n_trees']}, m {row['m']}")
            staged = C.HERE / row["path"]
            digest, ends = read_staged(staged, hash_files)
            if hash_files and digest != row["sha256_sub"]:
                problems.append(f"S1 {where}: {staged} no longer hashes to MANIFEST")
            stats["s1"] += 1
            # S4 -- the condition and its truth
            ident = (row["phase"], row["rep"], row["arm"])
            if ident not in B.key_map(ds):
                problems.append(f"S4 {where}: no landed condition")
                continue
            key = cond_key(ds, *ident)
            if key in conds.values() and conds.get(ident) != key:
                problems.append(f"S4 {where}: condition {key} claimed twice")
            conds[ident] = key
            if (row["phase"], row["rep"]) not in truths:
                staged_t = L.CAMUS.truth_of(L.CAMUS.CamusCell(
                    row["phase"], row["rep"], row["arm"], C.DATASETS[ds].draw_inputs(row["draw"])))
                landed_t = B.truth(ds, B.conditions(ds)[key])
                if (set(staged_t.labels) != set(landed_t.labels)
                        or L.CN.label_clusters(staged_t) != L.CN.label_clusters(landed_t)
                        or staged_t.n_retics != landed_t.n_retics):
                    problems.append(f"S4 {where}: staged truth != landed truth")
                truths[(row["phase"], row["rep"])] = landed_t
                stats["s4_truths"] += 1
            taxa = set(truths[(row["phase"], row["rep"])].labels)
            # S2 -- leaf sets and wrapping
            raw_t, fixed_t = paths["raw"].read_text().strip(), paths["fixed"].read_text().strip()
            if leaves(raw_t) != taxa | {C.OUT}:
                problems.append(f"S2 {where}: raw TOB leaves != truth + OUT")
            if leaves(fixed_t) != taxa:
                problems.append(f"S2 {where}: fixed TOB leaves != truth")
            for i, ls in enumerate(ends):
                if ls != taxa | {C.OUT}:
                    problems.append(f"S2 {where}: gene tree {'first' if i == 0 else 'last'} "
                                    f"leaves != truth + OUT")
            wr, wf = wrap_raw(raw_t), wrap_fixed(fixed_t)
            for v, w in (("raw", wr), ("fixed", wf)):
                if not root_children_ok(w):
                    problems.append(f"S2 {where}: wrapped {v} TOB is not (OUT,<body>)")
            s_raw, s_wr = P.tree_splits(raw_t, C.OUT), P.tree_splits(wr, C.OUT)
            s_wf = P.tree_splits(wf, C.OUT)
            if s_raw != s_wr:
                problems.append(f"S2 {where}: wrapping the raw TOB changed its splits")
            stats["s2"] += 1
            # S3 -- the A63 pairing
            kept = bool(meta.get("treeqmc_kept_a_contracted_root_branch"))
            removed, added = s_wr - s_wf, s_wf - s_wr
            contracted = {frozenset(b["cluster"]) for b in meta.get("root_branch", [])
                          if b.get("contracted")}
            contracted |= {frozenset(taxa) - c for c in contracted}
            if added or (kept and (len(removed) != 1 or not removed <= contracted)) or (
                    not kept and removed):
                problems.append(f"S3 {where}: kept={kept}, raw-only splits {len(removed)}, "
                                f"fixed-only {len(added)}")
            stats["s3_kept" if kept else "s3_same"] += 1
            # S5 -- vacuity, reported
            draw_splits[row["cell"]].append(frozenset(s_wr))
            f = full_splits.get(key)
            if f is not None:
                stats["s5_compared"] += 1
                stats["s5_equal_full"] += s_wr == f
        stats["s5_cells_all_draws_equal"] = sum(1 for v in draw_splits.values()
                                                if len(v) > 1 and len(set(v)) == 1)
        stats["s5_cells_multi_draw"] = sum(1 for v in draw_splits.values() if len(v) > 1)
        report["datasets"][ds] = dict(stats, manifest_rows=len(rows))
        print(f"[check] {ds}: {dict(stats)} | {time.perf_counter() - t0:.0f} s", flush=True)
    report["problems"] = problems
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return not problems, report


def full_tob_splits(ds: str) -> dict[str, set]:
    """Condition key -> splits of the landed full-data raw TOB (OUT-rooted), where one is here:
    camus the Mac ``blob_cpu_v1`` run's, l2reg/l3reg the staged ``.raw.rooted.nwk``.  netcs's
    landed TOB is unrooted and OUT-free, a different object: none."""
    out = {}
    if ds == "camus":
        for key in B.conditions(ds):
            f = C.REPO / "blob_cpu_v1" / "runs" / "C4537CLFY0_j8" / "work" / key / "tob.rooted.nwk"
            if f.exists():
                out[key] = P.tree_splits(f.read_text(), C.OUT)
    elif ds in ("l2reg", "l3reg"):
        for key in B.conditions(ds):
            f = C.REPO / f"{ds}_eval" / "tob" / f"{key}.raw.rooted.nwk"
            if f.exists():
                out[key] = P.tree_splits(f.read_text(), C.OUT)
    return out


# ------------------------------------------------------------------------------------------ parity

#: (dataset, condition key or None for the first, raw TOB, fixed TOB or None, the landed fed tob.nwk
#: per variant, landed output per (variant, resolver), landed cell JSON per variant).
def parity_cases() -> list[dict]:
    blob = C.REPO / "blob_cpu_v1" / "runs" / "C4537CLFY0_j8"
    cases = []
    for key in ("cam25_rep01_cam_iq500", "cam100_rep09_cam_g500"):
        w = blob / "work" / key
        cases.append({"ds": "camus", "key": key, "variant": "raw", "source": w / "tob.raw.tob",
                      "fed": w / "tob.rooted.nwk", "out": lambda r, w=w: w / f"{r}.enewick",
                      "cell": blob / "cells" / f"{key}.json"})
    for ds in ("l2reg", "l3reg"):
        conds = B.conditions(ds)
        key = sorted(conds, key=lambda k: conds[k]["rep"])[0]
        pkg = C.REPO / f"{ds}_n200_v1" / "tob_bp1000"
        ev = C.REPO / f"{ds}_eval"
        for variant, src in (("raw", pkg / f"{key}.raw.tob"), ("fixed", pkg / f"{key}.tob")):
            w = ev / "runs" / "resolvers" / "work" / key / variant
            cases.append({"ds": ds, "key": key, "variant": variant, "source": src,
                          "fed": ev / "tob" / f"{key}.{variant}.rooted.nwk",
                          "out": lambda r, w=w: w / f"{r}.enewick",
                          "cell": ev / "runs" / "resolvers" / "cells" / f"{key}.{variant}.json"})
    return cases


def parity(args) -> int:
    tasks, expect, bad = [], {}, 0
    print("[parity] wrapping")
    for c in parity_cases():
        text = WRAP[c["variant"]][1](c["source"].read_text())
        landed_fed = c["fed"].read_text().strip()
        same = text == landed_fed
        bad += not same
        print(f"  {'PASS' if same else 'FAIL'} {c['ds']:6s} {c['key']:30s} {c['variant']:5s} "
              f"wrap({c['source'].name}) {'==' if same else '!='} {c['fed'].name}")
        cond = B.conditions(c["ds"])[c["key"]]
        trees = B.gene_trees(c["ds"], cond)
        row = {"cell": c["key"], "draw": "full", "phase": cond["phase"], "rep": cond["rep"],
               "arm": cond["arm"], "sha256_sub": P.file_sha(trees), "path": str(trees)}
        cell = json.loads(c["cell"].read_text())
        for r in P.RESOLVERS:
            t = make_task(c["ds"], row, c["variant"], r, text)
            t.update(trees=str(trees), run_dir=str(RES / "_parity" / c["ds"]),
                     timeout_brooqs=args.timeout_brooqs, timeout_netcs=args.timeout_netcs)
            task_file(t).unlink(missing_ok=True)                 # parity always re-runs
            tasks.append(t)
            expect[t["task_id"]] = (c, c["out"](r), cell["methods"][r])
    recs = execute(tasks, args.jobs)
    print("\n[parity] outputs")
    for rec in sorted(recs, key=lambda r: (r["dataset"], r["cond"], r["variants"][0], r["resolver"])):
        c, out_f, landed = expect[rec["task_id"]]
        s = rec.get("score", {})
        if rec["status"] not in ("ok", "out_not_rootable"):
            print(f"  FAIL {rec['dataset']} {rec['cond']} {rec['resolver']}: {rec['status']}")
            bad += 1
            continue
        same_k = s["k_pred"] == landed["k_pred"]
        if rec["resolver"] == "brooqs":
            same_bytes = rec["enewick"] == out_f.read_text().strip()
            same_score = all(s.get(k) == landed.get(k) for k in ("tp", "fp", "f1", "k_pred"))
            good = same_bytes and same_score
            what = f"bytes {'==' if same_bytes else '!='} landed, score {'==' if same_score else '!='}"
        else:
            both = s["out_rootable"] and landed.get("out_rootable")
            same_score = (not both) or all(s.get(k) == landed.get(k) for k in ("tp", "fp", "f1"))
            good = same_k and same_score
            what = (f"k {s['k_pred']} vs {landed['k_pred']}, "
                    f"{'score ==' if both and same_score else 'score !=' if both else 'not both rootable'}")
        v = rec.get("violations", [])
        good &= not v
        bad += not good
        print(f"  {'PASS' if good else 'FAIL'} {rec['dataset']:6s} {rec['cond']:30s} "
              f"{rec['variants'][0]:5s} {rec['resolver']:6s} {what}; invariants {'ok' if not v else v}"
              f"; cpu {rec['timing']['cpu']:.1f}s")
    print(f"[parity] {'PASS' if not bad else f'FAIL ({bad})'}")
    return 0 if not bad else 1


# -------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="comma list, or 'all'")
    ap.add_argument("--resolvers", default="netcs,brooqs")
    ap.add_argument("--cells", default="", help="comma list of cell names")
    ap.add_argument("--draws", default="", help="comma list of draw numbers")
    ap.add_argument("--out", type=Path, default=RES, help="run root: <out>/<ds>/{tasks,work}/")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout-brooqs", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--timeout-netcs", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="re-score finished tasks from their outputs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="tally the task records on disk")
    ap.add_argument("--smoke", action="store_true", help="the SMOKE cells, draw 0; use with --out")
    ap.add_argument("--no-check", action="store_true", help="skip gates S1-S4 before a run")
    ap.add_argument("--check", action="store_true", help="gates S1-S5 only")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()

    datasets = [c.name for c in P.corpora(args.dataset)]
    if args.check:
        ok, report = check(datasets)
        P.write_json(RES / "check.json", report)
        for p in report["problems"][:40]:
            print(f"  PROBLEM {p}")
        print(f"[check] {'PASS: S1-S4' if ok else f'FAIL ({len(report['problems'])} problems)'} "
              f"in {report['seconds']:.0f} s -> {RES / 'check.json'}")
        return 0 if ok else 1
    if args.parity:
        return parity(args)

    resolvers = tuple(r for r in P.RESOLVERS if r in args.resolvers.split(","))
    cells = set(filter(None, args.cells.split(",")))
    draws = {int(d) for d in args.draws.split(",") if d.strip()} or None
    tasks: list[dict] = []
    for ds in datasets:
        want = cells
        dr = draws
        if args.smoke:
            want, dr = set(SMOKE[ds]), {0}
        built = build_tasks(ds, want or None, dr, resolvers)
        for t in built:
            t.update(run_dir=str(args.out / ds), timeout_brooqs=args.timeout_brooqs,
                     timeout_netcs=args.timeout_netcs)
            tasks.append(t)
        if args.smoke:
            unknown = set(want) - {t["cell"] for t in built}
            if unknown:
                raise SystemExit(f"--smoke: {ds} has no cell(s) {sorted(unknown)}")
    if cells:
        unknown = cells - {t["cell"] for t in tasks}
        if unknown:
            raise SystemExit(f"--cells: no such cell(s) in {datasets}: {sorted(unknown)}")
    if args.summary_only or args.rescore:
        recs = []
        for t in tasks:
            f = task_file(t)
            if f.exists():
                recs.append(rescore(f) if args.rescore else json.loads(f.read_text()))
        print(f"[summary] {len(recs)} of {len(tasks)} tasks have a record under {args.out}")
        print(summarise(recs))
        viol = [r for r in recs if r.get("violations")]
        print(f"[summary] hard invariant violations: {len(viol)}"
              + (f", e.g. {[(r['cell'], r['method'], r['violations']) for r in viol[:5]]}" if viol else ""))
        return 4 if viol else 0 if len(recs) == len(tasks) else 1

    todo, done = pending(tasks, args.force)
    by = Counter((t["dataset"], "+".join(t["variants"]), t["method"]) for t in tasks)
    cpu = defaultdict(float)
    for t in todo:
        cpu[(t["dataset"], t["resolver"])] += t["proj_cpu"]
    print(f"[run] {len(tasks)} tasks ({len(done)} done, {len(todo)} to run) under {args.out}")
    for k, n in sorted(by.items()):
        print(f"      {k[0]:6s} {k[1]:11s} {k[2]:6s} {n:5d}")
    for k, s in sorted(cpu.items()):
        print(f"      projected {k[0]:6s} {k[1]:6s} {s / 3600:7.1f} CPU-h")
    make = lpt_makespan([t["proj_cpu"] for t in todo], args.jobs) / 3600
    longest = max((t["proj_cpu"] for t in todo), default=0) / 3600
    print(f"      projected LPT makespan at --jobs {args.jobs}: {make:.1f} h "
          f"(longest task {longest:.2f} h; ratio to landed CPU {RATIO})")
    if args.dry_run:
        return 0

    if todo and not args.no_check:
        ok, report = check(sorted({t["dataset"] for t in todo}))
        if not ok:
            for p in report["problems"][:40]:
                print(f"  PROBLEM {p}")
            print("[run] gates failed; nothing run")
            return 1
    for ds in {t["dataset"] for t in todo}:
        rd = args.out / ds
        rd.mkdir(parents=True, exist_ok=True)
        P.write_json(rd / f"header_{time.strftime('%Y%m%dT%H%M%S')}.json",
                     {"argv": sys.argv, "host": socket.gethostname(), "tool_fp": P.tool_fp(),
                      "jobs": args.jobs, "timeout_brooqs": args.timeout_brooqs,
                      "timeout_netcs": args.timeout_netcs,
                      "n_tasks": sum(1 for t in todo if t["dataset"] == ds)})
    recs = execute(todo, args.jobs) + done
    text = summarise(recs)
    print("\n" + text)
    for ds in {r["dataset"] for r in recs}:
        (args.out / ds).mkdir(parents=True, exist_ok=True)
        (args.out / ds / "summary.txt").write_text(
            summarise([r for r in recs if r["dataset"] == ds]) + "\n")
    viol = [r for r in recs if r.get("violations")]
    failed = [r for r in recs if not r["status"].startswith(("ok", "out_not_rootable"))]
    if viol:
        print(f"\n[run] {len(viol)} outputs break an invariant, e.g. "
              f"{[(r['dataset'], r['cell'], r['method'], r['violations']) for r in viol[:5]]}")
    if failed:
        print(f"[run] {len(failed)} tasks did not finish ok (tool errors are outcomes): "
              f"{Counter(r['status'][:40] for r in failed)}")
    return 4 if viol else 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
