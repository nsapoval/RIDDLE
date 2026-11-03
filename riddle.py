#!/usr/bin/env python3
"""RIDDLE: a phylogenetic network per reticulation budget k from a file of gene trees.

RIDDLE is the GNN's emitted network, re-decided block by block by a calibrated choice model on the
GNN logit (z) and gene-tree triplet concordance (tc) -- "GNN MAP [z+tc]" under the train-fit beta.
This driver chains the two stages that produce it.  

    gnn_cpu_v1/.venv/bin/python riddle.py --trees genes.tre --outgroup OUT --k 1-5 --out run/

Steps (every one is skipped when its outputs already exist, unless ``--force``):

* **A1 root** -- the trees Stage A scores, re-derived with ``infer_gnn_cpu``'s own ``scan``,
  ``select``, ``prepare`` and ``resolve_tau`` in a subprocess.  ``infer`` writes them into a
  temporary directory it deletes, and Stage B needs them for the triple tensor.
* **A2 GNN** -- ``infer_gnn_cpu.py --decoder relaxed --max-level 2 --annotate``: the GNN's network
  and every candidate's logit, per k.  ``--stage-a PREFIX`` reuses a finished run instead.
* **B1 trees** -- the rooted trees as ``fz_trees.py`` writes them (deterministic gzip, presence
  matrix), and gate **N0-support**: the trees, counted at support >= tau, reproduce Stage A's
  candidate words and support row for row.  That is what proves A1 and A2 read the same trees.
* **B2 triples** -- ``fz_triples.build`` and ``check``: the partial-aware rooted-triple tensor and
  its gates N3'.
* **B4 features** -- ``fz_features.run_cell`` per k: the low-z blocks and their z / tc features.
* **B6 credible** -- ``fz_credible.run_cell`` per k, pool ``train``: the block distributions under
  ``nni_ensemble/runs_trainfit/calib/train__z6.json`` and the MAP network.
* **B7 tables** -- gates (no problems, feat and cred from one npz, N8: no truth-shaped key), then:

  - ``OUT/riddle.tsv``: ``k<TAB>eNewick``, the RIDDLE network per k (``fz_tables``' L2map rows);
  - ``OUT/gnn.tsv``: the GNN's own network per k, before the z+tc re-decision;
  - ``OUT/riddle.json``: arguments, hashes, the taxon set, m, tau, and per-k summaries.

Fixed, because Stage B was built and calibrated for exactly this: checkpoint
``gnn_cpu_v1/checkpoints/_l2mix_pair/s0/best.pt``, the relaxed level-2 decoder, z < 6, radius 2.
Stage B's caps (``fz_common.CAPS``) are not overridden: a k that exceeds one fails, is named, and
the run exits 1.  The triple tensor is n^3 uint16 (2 GB at n = 1,000), and m must be below 65,535.
Internal files carry ``fungi_`` names, which is the naming ``fz_common`` derives paths from.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
GNN = REPO / "gnn_cpu_v1"
MODEL_REL = Path("checkpoints") / "_l2mix_pair" / "s0" / "best.pt"
CALIB_SHA = "f47199e9746d8468f7c62ee145544f1525de176c94dbe7e4558adadf5c03a610"
SCHEME = "all"
POOL = "train"
#: Read by ``fz_paths`` / ``ne_common`` at import; any of them set would move Stage B's paths.
SCRUB = ("FZ_VARIANT", "FZ_BETA", "FZ_ROOT", "NE_ANAT_RUNS")
STAGE_A_EXT = ("tsv", "znet.tsv", "clusters.tsv", "logits.npz", "manifest.json")


def ks_of(spec: str) -> list[int]:
    """``1-5,8`` -> [1, 2, 3, 4, 5, 8]: sorted, distinct, each >= 0."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    if not out or min(out) < 0:
        raise ValueError(f"--k {spec!r}: expected non-negative integers, e.g. 1-5 or 1,3,5")
    return sorted(out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[riddle] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------------------- A1, subprocess

def root_helper(spec_path: Path) -> int:
    """A1: ``infer_gnn_cpu``'s own tree selection and rooting, written where Stage B can read it.

    ``infer_gnn_cpu`` reads ``--threads`` / ``--opc`` from ``sys.argv`` at import and imports torch,
    so this runs in its own process with ``sys.argv`` set first."""
    spec = json.loads(Path(spec_path).read_text())
    sys.argv = [sys.argv[0], "--threads", str(spec["threads"])]
    sys.path.insert(0, str(GNN))
    import infer_gnn_cpu as G
    species = G.RT.SpeciesTree.read(spec["species_tree"]) if spec["species_tree"] else None
    outgroups = tuple(spec["outgroups"])
    G.rooting_scheme(outgroups, spec["midpoint"], species)
    trees = Path(spec["trees"])
    masks, labels, freq = G.scan(trees)
    S, chosen, reserve, stats = G.select(masks, labels, freq, outgroups=outgroups, taxa=spec["taxa"],
                                         n_taxa=spec["n_taxa"], partial=spec["partial"], m=spec["m"],
                                         seed=spec["seed"], species=species)
    prep = G.prepare(trees, chosen, Path(spec["dst"]), masks, labels, S, outgroups,
                     midpoint=spec["midpoint"], species=species, reserve=reserve)
    m = prep["n_loci"]
    if not m:
        raise SystemExit(f"no tree could be rooted ({prep['scheme']}): rejected {prep['rejected']}")
    tau = G.resolve_tau(m, spec["tau"], spec["tau_frac"])
    out = {"S": S, "m": m, "tau": tau, "prep": prep,
           "scan": {k: v for k, v in stats.items() if isinstance(v, (int, float, str))}}
    Path(spec["root_json"]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


# ------------------------------------------------------------------------------------- driver

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--_root", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--trees", type=Path, help="one Newick gene tree per line, plain or .gz")
    ap.add_argument("--k", help="reticulation budgets, e.g. 1-5 or 1,3,5")
    ap.add_argument("--out", type=Path, help="output directory")
    root = ap.add_mutually_exclusive_group()
    root.add_argument("--outgroup", default="", help="comma list, ranked: each tree roots on the "
                      "first one it carries, which is then pruned")
    root.add_argument("--midpoint", action="store_true")
    root.add_argument("--species-tree", type=Path, default=None, metavar="FILE",
                      help="a rooted species tree; each gene tree roots on its root split")
    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--n-taxa", type=int, default=0, help="S = the N most frequent taxa")
    sub.add_argument("--taxa", type=Path, default=None, help="S = the labels in FILE, one per line")
    ap.add_argument("--partial", type=int, default=0, metavar="MIN",
                    help="accept trees carrying at least MIN taxa of S (default: every taxon)")
    ap.add_argument("--m", type=int, default=0, help="trees to sample; 0 (the default) uses all")
    ap.add_argument("--seed", type=int, default=0)
    cut = ap.add_mutually_exclusive_group()
    cut.add_argument("--tau", type=int, default=None, metavar="N",
                     help="candidate support threshold (infer_gnn_cpu's default: 2)")
    cut.add_argument("--tau-frac", default=None, metavar="F", help="tau = ceil(F * m)")
    ap.add_argument("--jobs", type=int, default=5, help="Stage B workers (~0.7 GB each)")
    ap.add_argument("--threads", type=int, default=1, help="Stage A's torch / BLAS threads")
    ap.add_argument("--stage-a", type=Path, default=None, metavar="PREFIX",
                    help="reuse a finished infer_gnn_cpu.py --annotate PREFIX run (relaxed L2)")
    ap.add_argument("--triples-check", choices=("sample", "full"), default="sample")
    ap.add_argument("--force", action="store_true", help="recompute every step")
    args = ap.parse_args(argv)
    if args._root is None:
        missing = [f for f in ("trees", "k", "out") if getattr(args, f) is None]
        if missing:
            ap.error("required: " + ", ".join(f"--{f}" for f in missing))
        try:
            args.ks = ks_of(args.k)
        except ValueError as exc:
            ap.error(str(exc))
    return args


def fingerprint(args) -> dict:
    """What changes the answer; a run directory refuses to resume under different values."""
    return {"trees": str(args.trees.resolve()), "trees_sha256": sha256(args.trees),
            "outgroups": [o.strip() for o in args.outgroup.split(",") if o.strip()],
            "midpoint": args.midpoint,
            "species_tree": str(args.species_tree.resolve()) if args.species_tree else None,
            "n_taxa": args.n_taxa, "taxa": str(args.taxa.resolve()) if args.taxa else None,
            "partial": args.partial, "m": args.m, "seed": args.seed, "tau": args.tau,
            "tau_frac": args.tau_frac, "ks": args.ks,
            "stage_a": str(args.stage_a.resolve()) if args.stage_a else None}


def run_logged(cmd: list[str], logf: Path, cwd: Path) -> subprocess.Popen:
    fh = open(logf, "w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)


def wait_ok(proc: subprocess.Popen, logf: Path, what: str) -> None:
    if proc.wait() != 0:
        tail = "".join(Path(logf).read_text().splitlines(keepends=True)[-15:])
        raise SystemExit(f"[riddle] {what} failed (exit {proc.returncode}); {logf}:\n{tail}")


def main() -> int:
    args = parse_args()
    if args._root is not None:
        return root_helper(args._root)
    for var in SCRUB:
        os.environ.pop(var, None)
    out = args.out.resolve()
    work = out / "work"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args)
    fp_path = work / "args.json"
    if fp_path.exists() and not args.force:
        old = json.loads(fp_path.read_text())
        if old != fp:
            diff = sorted(k for k in fp if old.get(k) != fp[k])
            raise SystemExit(f"[riddle] {out} holds a run with different {diff}: use another --out, "
                             f"or --force to recompute")
    fp_path.write_text(json.dumps(fp, indent=1) + "\n")

    sys.path.insert(0, str(REPO / "fungi" / "scripts"))
    import fz_common as FZ                          # first: BLAS pin, pylib on sys.path, no torch
    import fz_credible
    import fz_features
    import fz_tables
    import fz_trees
    import fz_triples
    import numpy as np

    if sha256(FZ.MODEL) != FZ.MODEL_SHA:
        raise SystemExit(f"[riddle] {FZ.MODEL}: not the checkpoint Stage B was calibrated for")
    calib = FZ.calib_path(POOL, FZ.NE.Z_THR)
    if sha256(calib) != CALIB_SHA:
        raise SystemExit(f"[riddle] {calib}: not the train-fit beta (sha256 {CALIB_SHA[:12]}...)")
    t0 = time.perf_counter()

    # ---- A1 + A2, side by side
    root_json, raw_gz = work / "root.json", work / "rooted.nwk.gz"
    stage_a = work / "stage_a" / "gnn"
    stage_a.parent.mkdir(exist_ok=True)
    procs = []
    if args.force or not root_json.exists():
        spec = {"trees": str(args.trees.resolve()), "outgroups": fp["outgroups"],
                "midpoint": args.midpoint, "species_tree": fp["species_tree"],
                "taxa": (None if args.taxa is None else
                         [s.strip() for s in args.taxa.read_text().splitlines() if s.strip()]),
                "n_taxa": args.n_taxa, "partial": args.partial, "m": args.m, "seed": args.seed,
                "tau": args.tau, "tau_frac": args.tau_frac, "threads": args.threads,
                "dst": str(raw_gz), "root_json": str(root_json)}
        (work / "root_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_root", str(work / "root_spec.json")]
        procs.append((run_logged(cmd, work / "logs" / "root.log", REPO), work / "logs" / "root.log",
                      "A1 rooting"))
        log("A1: rooting the trees as infer_gnn_cpu does")
    if args.stage_a is not None:
        for ext in STAGE_A_EXT:
            src = args.stage_a.resolve().with_name(f"{args.stage_a.name}.{ext}")
            dst = stage_a.with_name(f"gnn.{ext}")
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        log(f"A2: reusing Stage A at {args.stage_a}")
    elif args.force or not stage_a.with_name("gnn.manifest.json").exists():
        cmd = [sys.executable, "infer_gnn_cpu.py", "--model", str(MODEL_REL),
               "--trees", fp["trees"]]
        if fp["outgroups"]:
            cmd += ["--outgroup", ",".join(fp["outgroups"])]
        if args.midpoint:
            cmd += ["--midpoint"]
        if args.species_tree:
            cmd += ["--species-tree", fp["species_tree"]]
        if args.n_taxa:
            cmd += ["--n-taxa", str(args.n_taxa)]
        if args.taxa:
            cmd += ["--taxa", fp["taxa"]]
        if args.partial:
            cmd += ["--partial", str(args.partial)]
        cmd += ["--m", str(args.m), "--seed", str(args.seed)]
        if args.tau is not None:
            cmd += ["--tau", str(args.tau)]
        if args.tau_frac is not None:
            cmd += ["--tau-frac", str(args.tau_frac)]
        cmd += ["--decoder", "relaxed", "--max-level", "2", "--k", ",".join(map(str, args.ks)),
                "--threads", str(args.threads), "--out", str(stage_a.with_name("gnn.tsv")),
                "--annotate", str(stage_a)]
        procs.append((run_logged(cmd, work / "logs" / "stage_a.log", GNN),
                      work / "logs" / "stage_a.log", "A2 Stage A"))
        log(f"A2: infer_gnn_cpu.py, k {args.ks[0]}..{args.ks[-1]} ({len(args.ks)} budgets)")
    for proc, logf, what in procs:
        wait_ok(proc, logf, what)
    R = json.loads(root_json.read_text())
    S, m, tau = R["S"], int(R["m"]), int(R["tau"])
    log(f"A done: |S| {len(S)}, m {m:,}, tau {tau}, {R['prep']['scheme']} rooting")

    # ---- Stage A, where fz_common looks for it, and its manifest against A1
    gnn_root = work / "gnn"
    pre = FZ.gnn_prefix(gnn_root, SCHEME, tau, FZ.HEADLINE_DEC)
    pre.parent.mkdir(parents=True, exist_ok=True)
    for ext in STAGE_A_EXT:
        dst = pre.with_name(f"{pre.name}.{ext}")
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(stage_a.with_name(f"gnn.{ext}").resolve())
    man = FZ.read_json(pre.with_name(pre.name + ".manifest.json"))
    bad = []
    if man["model_sha256"] != FZ.MODEL_SHA:
        bad.append("model_sha256")
    if int(man["m"]) != m or man["scan"]["eligible"] != R["scan"]["eligible"]:
        bad.append(f"m/eligible {man['m']}/{man['scan']['eligible']} vs {m}/{R['scan']['eligible']}")
    if man["n_taxa"] != len(S):
        bad.append(f"n_taxa {man['n_taxa']} vs {len(S)}")
    ks_a = [int(r["k"]) for r in man["per_k"]]
    if sorted(ks_a) != args.ks:
        bad.append(f"ks {ks_a}")
    if any(r["decoder"] != FZ.DECODERS[FZ.HEADLINE_DEC] or int(r["tau"]) != tau for r in man["per_k"]):
        bad.append("a k not decoded relaxed_L2 at this tau")
    if bad:
        raise SystemExit(f"[riddle] Stage A disagrees with A1: {bad}")

    # ---- B1: trees, presence, and N0-support
    tdir = FZ.trees_dir(work, SCHEME)
    tdir.mkdir(parents=True, exist_ok=True)
    trees = [ln.strip() for ln in gzip.open(raw_gz, "rt") if ln.strip()]
    if len(trees) != m:
        raise SystemExit(f"[riddle] {raw_gz}: {len(trees)} trees, A1 recorded {m}")
    gz = tdir / "trees.nwk.gz"
    tmp = gz.with_name(gz.name + ".tmp")
    # fz_trees.py's writer: mtime 0 and no name in the header, so the sha256 is the content's
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as fh:
        fh.write("".join(t + "\n" for t in trees).encode())
    tmp.replace(gz)
    ti = {t: i for i, t in enumerate(S)}
    pres = np.zeros((m, len(S)), dtype=bool)
    for r, t in enumerate(trees):
        pres[r, [ti[x] for x in fz_trees._LEAF.findall(t)]] = True
    np.save(tdir / "presence.npy.tmp.npy", pres)
    (tdir / "presence.npy.tmp.npy").replace(tdir / "presence.npy")
    trees_sha = FZ.sha256(gz)
    FZ.write_json(tdir / "trees.json", {"scheme": SCHEME, "src": fp["trees"],
                                        "src_sha256": fp["trees_sha256"], "m": m,
                                        "n_labels": len(S), "partial": args.partial,
                                        "trees_sha256": trees_sha,
                                        "leaves_min": int(pres.sum(1).min()),
                                        "n_partial_trees": int((~pres.all(1)).sum()),
                                        "labels": S}, indent=1)
    counter, _ = fz_trees.count_clusters(trees, ti, m)
    masks, sup, _ = counter.finalize(min_support=1)
    sup = sup[:, -1].astype(np.int64)
    keep = sup >= tau
    W, Sup = fz_trees.words_of(masks[keep], len(S)), sup[keep]
    npz = pre.with_name(pre.name + ".logits.npz")
    arr = np.load(npz)
    same = (arr["words"].shape == W.shape and np.array_equal(arr["words"], W)
            and np.array_equal(arr["support"].astype(np.int64), Sup)
            and [str(t) for t in arr["taxa_order"]] == S)
    par = {"scheme": SCHEME, "m": m, "trees_sha256": trees_sha, "n_clusters": int(len(Sup)),
           "runs": {pre.name: {"npz_sha256": FZ.sha256(npz), "n_clusters": int(len(Sup)),
                               "rows_identical": bool(same)}}, "ok": bool(same)}
    FZ.write_json(tdir / "parity.json", par, indent=1)
    if not same:
        raise SystemExit("[riddle] N0-support fails: the rooted trees do not reproduce Stage A's "
                         "candidate words / support / taxon order")
    log(f"B1: {m:,} trees ({par['n_clusters']:,} candidates at tau {tau}) reproduce Stage A (N0)")

    # ---- B2: the triple tensor and N3'
    fz_triples.build(work, SCHEME, args.jobs, args.jobs, 0, args.force)
    n3 = fz_triples.check(work, SCHEME, args.triples_check)
    if n3["problems"]:
        # N3'(c) asks that the published counter's spurious counts -- from taxa a partial tree lacks
        # -- be removed, and refuses when there are none to remove.  On complete trees there are
        # none by construction, so the guard against a vacuous test has nothing to guard.
        rest = [p for p in n3["problems"] if not p.startswith("N3'(c)")]
        vac = (not rest and bool(pres.all()) and n3["n3c"]["n_mismatch"] == 0
               and n3["n3c"]["spurious_total"] == 0)
        if not vac:
            raise SystemExit(f"[riddle] N3' fails: {n3['problems']}")
        n3.update({"ok": True, "riddle_note": "N3'(c) not applicable: every tree carries every "
                   "taxon of S, so there is no spurious count to remove (0 mismatches)"})
        FZ.write_json(tdir / "n3.json", n3, indent=1)
        log("B2: N3'(c) not applicable (no partial tree); every other N3' gate passes")
    log(f"B2: triple tensor {len(S)}^3 built and checked ({args.triples_check})")

    # ---- B4 features, B6 credible: fz_*.run_cell per k
    thr, rho = FZ.NE.Z_THR, FZ.NE.RHO
    ns_f = argparse.Namespace(out_root=work, gnn_root=gnn_root, zthr=thr)
    units = [(SCHEME, tau, k) for k in args.ks]
    todo = [u for u in units if args.force or not FZ.feat_path(work, *u, thr).exists()]
    ok, err_f = FZ.run_units(fz_features.order_units(ns_f, todo),
                             lambda u: fz_features.run_cell(ns_f, u, False), args.jobs, "features")
    failed = {e["unit"]: e["error"] for e in err_f}
    ns_c = argparse.Namespace(out_root=work, gnn_root=gnn_root, zthr=thr, rho=rho, pools=[POOL],
                              variants=list(FZ.VARIANTS))
    have = [u for u in units if FZ.feat_path(work, *u, thr).exists()]
    todo = [u for u in have if args.force or not FZ.cred_path(work, *u, thr, rho, POOL).exists()]
    ok_c, err_c = FZ.run_units(todo, lambda u: fz_credible.run_cell(ns_c, u), args.jobs, "credible")
    failed.update({e["unit"]: e["error"] for e in err_c})
    log(f"B4/B6: {len(units) - len(failed)} of {len(units)} budgets scored under the train-fit beta")

    # ---- B7: gates and the networks
    problems, truth, rows, per_k = [], set(), [], []
    for u in units:
        f_p, c_p = FZ.feat_path(work, *u, thr), FZ.cred_path(work, *u, thr, rho, POOL)
        cid = FZ.cell_id(*u)
        if not (f_p.exists() and c_p.exists()):
            failed.setdefault(cid, "no feat / cred file")
            continue
        f, c = FZ.read_json(f_p), FZ.read_json(c_p)
        fz_tables.walk_keys(f, truth)
        fz_tables.walk_keys(c, truth)
        problems += [f"{cid}: {p}" for p in c["problems"]]
        if c["npz_sha256"] != f["npz_sha256"]:
            problems.append(f"{cid}: cred and feat are for different npz")
        mp = c["variants"]["z+tc"]["map"]
        rows.append((u[2], mp["enewick"]))
        per_k.append({"k": u[2], "n_blocks": c["n_blocks"], "n_moved": mp["n_moved"],
                      "log10_p_map": mp["log10_p_map"], "r": mp["r"], "level": mp["level"],
                      "moved": mp["enewick"] != c["emitted_net"]["enewick"],
                      "n_selected_gnn": f["emitted"]["n_selected"],
                      "n_retics_gnn": f["emitted"]["n_retics"]})
    if truth:
        problems.append(f"N8: truth-shaped keys present: {sorted(truth)}")
    if problems:
        for p in problems:
            print(f"[riddle] PROBLEM {p}", file=sys.stderr)
        raise SystemExit("[riddle] Stage B gates failed; nothing written to riddle.tsv")
    rows.sort()
    (out / "riddle.tsv").write_text("".join(f"{k}\t{enw}\n" for k, enw in rows))
    gnn_lines = FZ.tsv_lines(pre.with_name(pre.name + ".tsv"))
    (out / "gnn.tsv").write_text("".join(f"{k}\t{gnn_lines[k]}\n" for k in sorted(gnn_lines)))
    summary = {"args": fp, "model": str(FZ.MODEL), "model_sha256": FZ.MODEL_SHA,
               "calib": str(calib), "calib_sha256": CALIB_SHA, "n_taxa": len(S), "taxa": S,
               "m": m, "tau": tau, "rooting": R["prep"], "scan": R["scan"],
               "n_partial_trees": int((~pres.all(1)).sum()), "n_candidates": par["n_clusters"],
               "decoder": "relaxed_L2", "zthr": thr, "rho": rho, "pool": POOL,
               "per_k": per_k, "failed": failed, "seconds": time.perf_counter() - t0}
    FZ.write_json(out / "riddle.json", summary, indent=1)
    moved = sum(r["moved"] for r in per_k)
    log(f"done: {len(rows)} networks in {out / 'riddle.tsv'} ({moved} re-decided from the GNN's), "
        f"{len(failed)} failed | {summary['seconds']:.0f} s")
    for cid, why in sorted(failed.items()):
        print(f"[riddle] FAILED {cid}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
