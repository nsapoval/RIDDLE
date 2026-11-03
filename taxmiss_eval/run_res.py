#!/usr/bin/env python3
"""NetCS and BROOQS on the missing-taxa trees of blobs -- one task per (condition, cell, TOB variant,
resolver).  See README.md.

Inputs
------
* **Gene trees:** each cell's MANIFEST ``path`` under ``inputs/``, as the Linux box wrote it: the
  published trees with taxa deleted per ``missing.tsv``, OUT kept (``stage.py`` T1/T1b/T2).
* **Trees of blobs:** ``runs/tob/<fam>/<cond>/<cell>.{raw.tob,tob,json}``, TREE-QMC 4.1.5 ``--blob``
  on exactly those files (``json['input']`` / ``sha256_input``, gate S1).

The protocol is ``subsample_eval/run_res.py``'s, which is the landed camus / l2reg / l3reg one::

    raw     tob.nwk = camus_nets.root_tob_on_outgroup(<cell>.raw.tob)
    fixed   tob.nwk = (OUT,<cell>.tob);
    netcs   tree-qmc --network -i <gene trees> --at tob.nwk -o out.enewick
    brooqs  brooqs -i <gene trees> -t tob.nwk -o out.enewick --rooted --dist-var

Both resolvers take the partial gene trees as they are: they read only the TOB and the quartets the
trees carry (TREE-QMC warns about, and skips, a tree with fewer than 4 species; none comes close).
``raw`` is the default variant everywhere -- the comparator ``nni_ensemble/bench.py`` writes as
``BROOQS`` / ``NetCS`` -- and ``--variants raw,fixed`` adds the fixed TOB on l2reg/l3reg (labels
``... (fixed TOB)``).  netcs's labels carry ``*``, as in ``subsample_eval``.

Scoring is the landed drivers' own, ``run_blob_cpu.score`` against ``C(N_true)`` with taxon set
truth + OUT, after the clock has stopped -- ``subsample_eval/run_res.py:score_output``, loaded from
that file together with the wrapping, the condition keys, ``pending`` and ``rescore``.  ``run_task``,
``execute`` and the gates are this file's own: the subsample's are keyed on an integer draw.

Gates (``--check``; also run before any grid)
---------------------------------------------
    S1  every staged cell of the asked conditions has its three TOB files; json input == MANIFEST
        path, sha256_input == sha256_staged == the file's SHA-256 now; n_trees == n_loci
    S2  leaf sets: raw TOB == truth + OUT, fixed TOB == truth; the first and last gene tree carry
        OUT and only truth taxa (stage.py T2 checked every tree); both wrapped TOBs are
        (OUT,<body>), and wrapping moves no split
    S3  A63: fixed splits == raw splits, minus exactly the contracted root-branch split where the
        JSON says TREE-QMC kept one
    S4  every cell maps to a landed condition whose truth (pte_bench.truth) equals the staged copy's,
        by label cluster and k

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root; long runs from a terminal, caffeinate -i)
-----
    python taxmiss_eval/run_res.py --check
    python taxmiss_eval/run_res.py --smoke --jobs 4 --out <scratch>
    python taxmiss_eval/run_res.py --dataset camus,l2reg,l3reg --dry-run --jobs 12
    python taxmiss_eval/run_res.py --dataset camus,l2reg,l3reg --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200
    python taxmiss_eval/run_res.py --dataset all --summary-only
"""

from __future__ import annotations

import tm_common as TM  # first: BLAS pin, gnn_cpu_v1/pylib at sys.path[0], subsample_eval's common

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: subsample_eval/run_res.py, loaded by path (this file has its name); it imports pte_common /
#: pte_bench and ``common`` -- subsample_eval's, which tm_common already loaded under that name.
SR = _load("sub_run_res", TM.SUB / "run_res.py")
P, B = SR.P, SR.B

RES = TM.RUNS / "res"
VARIANTS = {"camus": ("raw",), "l2reg": ("raw", "fixed"), "l3reg": ("raw", "fixed"), "netcs": ("raw",)}
DEFAULT_VARIANTS = ("raw",)
DEFAULT_TIMEOUT = 7200.0
#: CPU of a run on a condition's 1,000 partial trees over the landed run on the full 1,000, for the
#: LPT order and the --dry-run projection only.  1.0 until the smoke says otherwise (README).
RATIO = {"brooqs": 1.0, "netcs": 1.0}
SMOKE = {"camus": ("cam25_rep00_cam_iq500", "cam50_rep00_cam_g500"),
         "l2reg": ("l2reg_n200_rep00_est_bp1000",), "l3reg": ("l3reg_n200_rep00_est_bp1000",),
         "netcs": ("netcs_L1_n50_rep10_netcs_ils1.0", "netcs_L2_n200_rep00_netcs_ils1.0")}  # subsample_eval's


def label(fam: str, variant: str, resolver: str) -> str:
    return SR.label(fam, variant, resolver)


def rows_for(fam: str, conds: list[str]) -> list[dict]:
    """The MANIFEST rows of the asked conditions, every one staged."""
    want = set(conds)
    return [r for r in TM.manifest(fam) if r["cond"] in want]


# ------------------------------------------------------------------------------------------- tasks

def make_task(fam: str, row: dict, variant: str, resolver: str, tob_text: str) -> dict:
    """One task.  ``cond`` is the **landed** condition key (``pte_bench.key_map``), because
    ``score_output`` looks the truth up by it; the missing-taxa condition is ``tm_cond``."""
    cell = row["cell"]
    key = SR.cond_key(fam, row["phase"], row["rep"], row["arm"])
    argv = list(SR.BROOQS_ARGS) if resolver == "brooqs" else []
    spec = {"dataset": fam, "key": f"{row['cond']}/{cell}", "trees_sha": row["sha256_staged"],
            "tob_sha": P.sha(tob_text), "resolver": resolver, "argv": argv, "tool_fp": P.tool_fp()}
    rule = P.rule_of(resolver, None)
    stats = P.tob_stats(tob_text)
    land = SR.landed_cpu(fam).get((key, variant, resolver))
    proj = RATIO[resolver] * land if land is not None else P.brooqs_cpu(stats, "brooqs4")
    return {**spec, "task_id": P.task_id(spec), "tm_cond": row["cond"], "cell": cell, "cond": key,
            "phase": row["phase"], "rep": row["rep"], "arm": row["arm"], "variants": [variant],
            "method": resolver, "rule": rule, "n_elig": stats["elig"][rule],
            "trees": str(TM.INPUTS / row["path"]), "proj_cpu": round(proj, 1),
            "landed_cpu": land, "tob_text": tob_text}


def build_tasks(fam: str, conds: list[str], variants, cells: set[str] | None = None,
                resolvers=P.RESOLVERS) -> list[dict]:
    """Every task; two variants whose ``tob.nwk`` is byte-identical are one task serving both."""
    tasks: dict[str, dict] = {}
    for row in rows_for(fam, conds):
        if cells and row["cell"] not in cells:
            continue
        paths = TM.tob_paths(fam, row["cond"], row["cell"])
        for variant in [v for v in VARIANTS[fam] if v in variants]:
            text = SR.WRAP[variant][1](paths[variant].read_text())
            for r in resolvers:
                t = make_task(fam, row, variant, r, text)
                if t["task_id"] in tasks:
                    tasks[t["task_id"]]["variants"].append(variant)
                else:
                    tasks[t["task_id"]] = t
    return list(tasks.values())


def run_task(task: dict) -> dict:
    """One resolver on one tree of blobs, then scoring after the clock has stopped --
    ``subsample_eval/run_res.py:run_task`` with the condition in the work path.  Top-level so a
    spawned worker can unpickle it."""
    RB = B.rb()
    run = Path(task["run_dir"])
    work = run / "work" / task["tm_cond"] / task["cell"] / f"{task['variants'][0]}.{task['method']}"
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
               t_end=time.time(), score_version=SR.SCORE_VERSION)
    rec.update(SR.score_output(task, task["tob_text"], out_p, t))
    P.write_json(run / "tasks" / f"{task['task_id']}.json", rec)
    return rec


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
            print(f"[run] {i}/{len(todo)} {t['dataset']} {t['tm_cond']} {t['cell']:30s} "
                  f"{'+'.join(t['variants']):9s} {t['method']:6s} {rec['status'][:24]:24s} "
                  f"k {s.get('k_pred', '')!s:>3}/{t['n_elig']:<3} f1 {f1} "
                  f"cpu {rec.get('timing', {}).get('cpu', 0):7.1f}s (proj {t['proj_cpu']:.0f}) | "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return out


def summarise(recs: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["dataset"], r["tm_cond"], "+".join(r.get("variants", [])), r["method"])].append(r)
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lines = [f"{'dataset':6s} {'cond':10s} {'variant':9s} {'method':6s} {'n':>4s} {'ok':>4s} "
             f"{'unroot':>6s} {'t/out':>5s} {'error':>5s} {'k med':>6s} {'elig':>5s} {'viol':>4s} "
             f"{'F1 med':>7s} {'cpu med':>8s} {'CPU h':>6s} {'/landed':>7s}"]
    for (ds, cond, variant, method), rs in sorted(groups.items()):
        st = Counter("error" if r["status"].startswith(("error", "crash")) else r["status"] for r in rs)
        sc = [r for r in rs if "invariants" in r]
        kk = [r["score"]["k_pred"] for r in sc]
        el = [r["invariants"]["n_elig"] for r in sc]
        viol = sum(1 for r in rs if r.get("violations"))
        f1 = [r["score"]["f1"] for r in sc if r["score"].get("f1") is not None]
        cpu = [r["timing"]["cpu"] for r in rs if "timing" in r]
        ratio = [r["timing"]["cpu"] / r["landed_cpu"] for r in rs if "timing" in r and r.get("landed_cpu")]
        lines.append(f"{ds:6s} {cond:10s} {variant:9s} {method:6s} {len(rs):4d} {st['ok']:4d} "
                     f"{st['out_not_rootable']:6d} {st['timeout']:5d} {st['error']:5d} {med(kk):6.1f} "
                     f"{med(el):5.1f} {viol:4d} {med(f1):7.3f} {med(cpu):8.1f} {sum(cpu) / 3600:6.2f} "
                     f"{med(ratio):7.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- gates

def check(fams: list[str], conds_of: dict[str, list[str]], hash_files: bool = True) -> tuple[bool, dict]:
    """S1-S4 (module docstring).  Returns (pass, report)."""
    L = B.lib()
    problems: list[str] = []
    report: dict = {"checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "datasets": {}}
    t0 = time.perf_counter()
    for fam in fams:
        stats = Counter()
        truths: dict[tuple, object] = {}
        for row in rows_for(fam, conds_of[fam]):
            where = f"{fam} {row['cond']} {row['cell']}"
            paths = TM.tob_paths(fam, row["cond"], row["cell"])
            paths = {k: paths[k] for k in ("json", "raw", "fixed")}
            # S1 -- provenance
            if not all(p.exists() for p in paths.values()):
                problems.append(f"S1 {where}: TOB files missing")
                continue
            meta = json.loads(paths["json"].read_text())
            if meta["input"] != row["path"] or meta["sha256_input"] != row["sha256_staged"]:
                problems.append(f"S1 {where}: TOB built from {meta['input']} "
                                f"({meta['sha256_input'][:12]}), MANIFEST says {row['path']}")
            if int(meta["n_trees"]) != row["n_loci"]:
                problems.append(f"S1 {where}: n_trees {meta['n_trees']}, n_loci {row['n_loci']}")
            src = TM.INPUTS / row["path"]
            data = src.read_bytes()
            if hash_files and TM.SC.sha256_bytes(data) != row["sha256_staged"]:
                problems.append(f"S1 {where}: {src} no longer hashes to MANIFEST")
            lines = [ln for ln in data.decode().splitlines() if ln.strip()]
            ends = [SR.leaves(lines[0]), SR.leaves(lines[-1])]
            stats["s1"] += 1
            # S4 -- the condition and its truth
            ident = (row["phase"], row["rep"], row["arm"])
            if ident not in B.key_map(fam):
                problems.append(f"S4 {where}: no landed condition")
                continue
            key = SR.cond_key(fam, *ident)
            if (row["phase"], row["rep"]) not in truths:
                staged_t = L.CAMUS.truth_of(L.CAMUS.CamusCell(
                    row["phase"], row["rep"], row["arm"], TM.staged_dir(fam, row["cond"])))
                landed_t = B.truth(fam, B.conditions(fam)[key])
                if (set(staged_t.labels) != set(landed_t.labels)
                        or L.CN.label_clusters(staged_t) != L.CN.label_clusters(landed_t)
                        or staged_t.n_retics != landed_t.n_retics):
                    problems.append(f"S4 {where}: staged truth != landed truth")
                truths[(row["phase"], row["rep"])] = landed_t
                stats["s4_truths"] += 1
            taxa = set(truths[(row["phase"], row["rep"])].labels)
            # S2 -- leaf sets and wrapping
            raw_t, fixed_t = paths["raw"].read_text().strip(), paths["fixed"].read_text().strip()
            if SR.leaves(raw_t) != taxa | {TM.OUT}:
                problems.append(f"S2 {where}: raw TOB leaves != truth + OUT")
            if SR.leaves(fixed_t) != taxa:
                problems.append(f"S2 {where}: fixed TOB leaves != truth")
            for i, ls in enumerate(ends):
                if TM.OUT not in ls or not ls <= taxa | {TM.OUT}:
                    problems.append(f"S2 {where}: gene tree {'first' if i == 0 else 'last'} lacks OUT "
                                    f"or carries a taxon the truth does not")
            wr, wf = SR.wrap_raw(raw_t), SR.wrap_fixed(fixed_t)
            for v, w in (("raw", wr), ("fixed", wf)):
                if not SR.root_children_ok(w):
                    problems.append(f"S2 {where}: wrapped {v} TOB is not (OUT,<body>)")
            s_raw, s_wr = P.tree_splits(raw_t, TM.OUT), P.tree_splits(wr, TM.OUT)
            s_wf = P.tree_splits(wf, TM.OUT)
            if s_raw != s_wr:
                problems.append(f"S2 {where}: wrapping the raw TOB changed its splits")
            stats["s2"] += 1
            # S3 -- the A63 pairing
            kept = bool(meta.get("treeqmc_kept_a_contracted_root_branch"))
            removed, added = s_wr - s_wf, s_wf - s_wr
            contracted = {frozenset(b["cluster"]) for b in meta.get("root_branch", []) if b.get("contracted")}
            contracted |= {frozenset(taxa) - c for c in contracted}
            if added or (kept and (len(removed) != 1 or not removed <= contracted)) or (not kept and removed):
                problems.append(f"S3 {where}: kept={kept}, raw-only splits {len(removed)}, "
                                f"fixed-only {len(added)}")
            stats["s3_kept" if kept else "s3_same"] += 1
        report["datasets"][fam] = dict(stats, conds=conds_of[fam])
        print(f"[check] {fam} {','.join(conds_of[fam])}: {dict(stats)} | {time.perf_counter() - t0:.0f} s",
              flush=True)
    report["problems"] = problems
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return not problems, report


# -------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", help="comma list of families, or 'all' (every one staged)")
    ap.add_argument("--conds", default="all", help="comma list, or 'all' (every staged condition)")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help="raw (default), or raw,fixed (fixed applies to l2reg/l3reg only)")
    ap.add_argument("--resolvers", default="netcs,brooqs")
    ap.add_argument("--cells", default="", help="comma list of cell names")
    ap.add_argument("--out", type=Path, default=RES, help="run root: <out>/<fam>/{tasks,work}/")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout-brooqs", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--timeout-netcs", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="re-score finished tasks from their outputs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="tally the task records on disk")
    ap.add_argument("--smoke", action="store_true", help="the SMOKE cells; use with --out")
    ap.add_argument("--no-check", action="store_true", help="skip gates S1-S4 before a run")
    ap.add_argument("--check", action="store_true", help="gates S1-S4 only")
    args = ap.parse_args()
    args.out = args.out.resolve()

    fams = TM.families_staged() if args.dataset == "all" else [f for f in args.dataset.split(",") if f]
    unknown = [f for f in fams if f not in TM.FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; known {list(TM.FAMILIES)}")
    conds_of = {f: TM.parse_conds(f, args.conds) for f in fams}
    fams = [f for f in fams if conds_of[f]]
    if not fams:
        raise SystemExit("nothing staged for the asked families -- run taxmiss_eval/stage.py")
    variants = {v for v in args.variants.split(",") if v}
    if not variants <= {"raw", "fixed"}:
        raise SystemExit(f"--variants: {sorted(variants)}; known raw, fixed")

    if args.check:
        ok, report = check(fams, conds_of)
        P.write_json(RES / "check.json", report)
        for p in report["problems"][:40]:
            print(f"  PROBLEM {p}")
        print(f"[check] {'PASS: S1-S4' if ok else f'FAIL ({len(report['problems'])} problems)'} "
              f"in {report['seconds']:.0f} s -> {RES / 'check.json'}")
        return 0 if ok else 1

    resolvers = tuple(r for r in P.RESOLVERS if r in args.resolvers.split(","))
    cells = set(filter(None, args.cells.split(",")))
    tasks: list[dict] = []
    for fam in fams:
        want = set(SMOKE[fam]) if args.smoke else cells
        if args.smoke and not want:
            continue
        built = build_tasks(fam, conds_of[fam], variants, want or None, resolvers)
        for t in built:
            t.update(run_dir=str(args.out / fam), timeout_brooqs=args.timeout_brooqs,
                     timeout_netcs=args.timeout_netcs)
            tasks.append(t)
    if cells:
        missing = cells - {t["cell"] for t in tasks}
        if missing:
            raise SystemExit(f"--cells: no such cell(s) in {fams}: {sorted(missing)}")
    if not tasks:
        raise SystemExit("no tasks")
    if args.summary_only or args.rescore:
        recs = []
        for t in tasks:
            f = SR.task_file(t)
            if f.exists():
                recs.append(SR.rescore(f) if args.rescore else json.loads(f.read_text()))
        print(f"[summary] {len(recs)} of {len(tasks)} tasks have a record under {args.out}")
        print(summarise(recs))
        viol = [r for r in recs if r.get("violations")]
        print(f"[summary] hard invariant violations: {len(viol)}"
              + (f", e.g. {[(r['cell'], r['method'], r['violations']) for r in viol[:5]]}" if viol else ""))
        return 4 if viol else 0 if len(recs) == len(tasks) else 1

    todo, done = SR.pending(tasks, args.force)
    by = Counter((t["dataset"], t["tm_cond"], "+".join(t["variants"]), t["method"]) for t in tasks)
    cpu = defaultdict(float)
    for t in todo:
        cpu[(t["dataset"], t["resolver"])] += t["proj_cpu"]
    print(f"[run] {len(tasks)} tasks ({len(done)} done, {len(todo)} to run) under {args.out}")
    for k, n in sorted(by.items()):
        print(f"      {k[0]:6s} {k[1]:10s} {k[2]:11s} {k[3]:6s} {n:5d}")
    for k, s in sorted(cpu.items()):
        print(f"      projected {k[0]:6s} {k[1]:6s} {s / 3600:7.1f} CPU-h")
    make = SR.lpt_makespan([t["proj_cpu"] for t in todo], args.jobs) / 3600
    longest = max((t["proj_cpu"] for t in todo), default=0) / 3600
    print(f"      projected LPT makespan at --jobs {args.jobs}: {make:.1f} h "
          f"(longest task {longest:.2f} h; ratio to landed CPU {RATIO})")
    if args.dry_run:
        return 0

    if todo and not args.no_check:
        ok, report = check(sorted({t["dataset"] for t in todo}), conds_of)
        if not ok:
            for p in report["problems"][:40]:
                print(f"  PROBLEM {p}")
            print("[run] gates failed; nothing run")
            return 1
    for fam in {t["dataset"] for t in todo}:
        rd = args.out / fam
        rd.mkdir(parents=True, exist_ok=True)
        P.write_json(rd / f"header_{time.strftime('%Y%m%dT%H%M%S')}.json",
                     {"argv": sys.argv, "host": socket.gethostname(), "tool_fp": P.tool_fp(),
                      "jobs": args.jobs, "timeout_brooqs": args.timeout_brooqs,
                      "timeout_netcs": args.timeout_netcs, "variants": sorted(variants),
                      "n_tasks": sum(1 for t in todo if t["dataset"] == fam)})
    recs = execute(todo, args.jobs) + done
    text = summarise(recs)
    print("\n" + text)
    for fam in {r["dataset"] for r in recs}:
        (args.out / fam).mkdir(parents=True, exist_ok=True)
        (args.out / fam / "summary.txt").write_text(summarise([r for r in recs if r["dataset"] == fam]) + "\n")
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
