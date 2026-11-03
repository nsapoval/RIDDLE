#!/usr/bin/env python3
"""Per cell and variant: the block model's configuration probabilities, and every compact form of
the credible set the plan reports.

For each non-excluded block: the option weights ``w_Y = beta . phi(Y)`` from the fold the cell's
network is held out of; configurations best-first (:func:`blocks.bk_best`) to ``KEEP_MAX`` or until
the unlisted mass is below ``MASS_TAIL``; ``P_b`` normalised by the upper bracket of ``Z``.  A block
with more than ``R_CAP`` configurations (``mode = "radius"``) lists instead every configuration within
``--rho`` swaps of the emitted one (:func:`blocks.radius_configs`), exactly, and ``P_b`` is conditional
on that set; ``mass_by_radius`` says how much of it sits at each radius.  Then:

* **per-block** ``CS_b(alpha)`` sizes, where ``r*`` and the emitted configuration rank, the mass
  ranked above ``r*`` (the randomised PIT and the conformal score);
* **marginals** ``P(Y in network)`` for every option;
* the **MAP** network, and **Hamming balls** -- the MAP plus up to ``d`` blocks moved inside their
  ``CS_b(ALPHA_BALL)`` -- with their mass (a Poisson-binomial DP) and their best F1;
* the **joint lazy k-best** to ``K_CAP``: best-in-top-K F1, and the mass covered;
* the exact **per-block oracle** (Dinkelbach over the product of enumerated configurations).

F1 is additive across blocks (``tp = tp_E + sum dtp_b``, ``|sel| = |E| + sum dsel_b``), so none of that
needs a network.  Networks are realised for: the MAP, the best of the top 4 and top 16, and every
member of the ``d = 1`` ball, through ``RX.realise`` and ``check_relaxed``'s legality recipe.

Gates: **N1** -- every realised member is legal with the emitted network's reticulation number and
level; **N5** -- F1 from block deltas equals the realised selection's counts, and on cells with
``prod |R_b| <= 10^4`` brute-force product enumeration equals the lazy k-best prefix.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python nni_ensemble/credible.py --dataset camus --jobs 8
    python nni_ensemble/credible.py --dataset netcs --cells netcs_eval/shards/CELLS_00.tsv --jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import ne_common as NE

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402
import calib as CB  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.solve import closure  # noqa: E402

KEEP_STORED = 256
N5_MAX = 10_000


def cred_path(root: Path, ds: str, name: str, thr: float, rho: int) -> Path:
    return Path(root) / ds / "cred" / f"{name}__{NE.cred_tag(thr, rho)}.json"


def lse(xs) -> float:
    xs = [x for x in xs if x != -math.inf]
    if not xs:
        return -math.inf
    mx = max(xs)
    return mx + math.log(sum(math.exp(x - mx) for x in xs))


def f1(tp: int, n_sel: int, n_true: int) -> float:
    return 2.0 * tp / (n_sel + n_true) if n_sel + n_true else 0.0


# ------------------------------------------------------------------------------------- one block

def block_dist(blk: dict, ds: str, model: dict, cols, adj: list[int], radius: list[int] | None,
               rho: int) -> dict:
    """``P_b`` over the block's configurations, best first, with everything the tables need.

    ``radius`` is the block's configurations within ``rho`` swaps of the emitted one when the block
    is oversized (``mode = "radius"``; ``P_b`` is then conditional on that set, normalised exactly),
    else ``None`` (``mode = "full"``: best-first over all of ``R_b``)."""
    o = blk["options"]
    X = (CB.option_matrix(blk, ds, cols) - np.asarray(model["mean"])) / np.asarray(model["sd"])
    w = (X @ np.asarray(model["beta"])).tolist()
    ab = [NE.unhex(x) for x in o["abits"]]
    members = sum(1 << i for i in blk["members"])
    if radius is None:
        got = B.bk_best(adj, w, NE.KEEP_MAX, NE.MASS_TAIL, max_nodes=NE.NODE_CAP)
    else:
        ws = sorted(((sum(w[i] for i in B.bits(c)), c) for c in radius), key=lambda t: -t[0])
        got = {"cliques": [c for _, c in ws], "logw": [x for x, _ in ws], "exact": True,
               "log_frontier": -math.inf}
    cl, lw = got["cliques"], got["logw"]
    z_lo = lse(lw)
    z_hi = lse(lw + [got["log_frontier"]])
    p = [math.exp(x - z_hi) for x in lw]
    true = o["true"]
    n_mem_true = sum(1 for i in blk["members"] if true[i])
    dtp = [sum(1 for i in B.bits(c) if true[i]) - n_mem_true for c in cl]
    dsel = [c.bit_count() - len(blk["members"]) for c in cl]
    rank_of = {c: i for i, c in enumerate(cl)}
    star = sum(1 << i for i in blk["rstar"]) if blk["rstar"] is not None else None
    r_rank = rank_of.get(star, -1) if star is not None else -1
    e_rank = rank_of.get(members, -1)
    cum = np.cumsum(p)
    cs = {}
    for a in NE.ALPHAS:
        k = int(np.searchsorted(cum, a - 1e-12) + 1)
        cs[f"{a:g}"] = k if k <= len(p) else -1          # -1: not reached within what was listed
    marg = [0.0] * len(ab)
    for c, pc in zip(cl, p):
        for i in B.bits(c):
            marg[i] += pc
    mass_r = None
    if radius is not None:
        mass_r = [0.0] * (rho + 1)
        for c, pc in zip(cl, p):
            mass_r[(c & ~members).bit_count()] += pc
    return {"cl": cl, "logw": lw, "p": p, "dtp": dtp, "dsel": dsel,
            "mode": "full" if radius is None else "radius", "rho": None if radius is None else rho,
            "mass_by_radius": mass_r, "map_radius": (cl[0] & ~members).bit_count() if cl else None,
            "exact": got["exact"], "z_gap": 1.0 - math.exp(z_lo - z_hi) if z_hi > -math.inf else 0.0,
            "n_listed": len(cl), "rstar_rank": r_rank,
            "rstar_p": p[r_rank] if r_rank >= 0 else None,
            "rstar_above": float(sum(p[:r_rank])) if r_rank >= 0 else None,
            "emit_rank": e_rank, "emit_p": p[e_rank] if e_rank >= 0 else None,
            "cs": cs, "marg": marg}


# ---------------------------------------------------------------------------------------- one cell

def realise_selection(words: np.ndarray, rows, n: int, taxa) -> dict:
    """Legality as ``l3reg_eval/check_relaxed.py:155-166`` states it, plus the network."""
    rows = np.asarray(sorted(rows), dtype=np.int64)
    full = closure.with_trivial_clusters(words[rows], n)
    parent, child = closure.hasse_covers_words(full, n)
    indeg = int(np.bincount(child, minlength=full.shape[0]).max())
    r = closure.reticulation_number(parent, child, full.shape[0])
    got = RX.realise(words, rows, n, taxa)
    if got is None:
        return {"ok": False, "indeg": indeg, "r": r, "level": None, "enewick": None}
    net = got[0]
    return {"ok": True, "indeg": indeg, "r": r, "level": int(NW.level(net)),
            "enewick": NW.to_enewick(net), "k_pred": int(net.n_retics)}


def run_cell(args, row: dict, calib: dict) -> dict:
    ds, name, thr = args.dataset, row["name"], args.thr
    t0 = time.perf_counter()
    feat = json.loads((Path(args.runs_root) / ds / "feat" / f"{name}__{NE.thr_tag(thr)}.json").read_text())
    c = NE.load_cell(ds, name)
    taxa = list(c["meta"]["taxa"])
    n = len(taxa)
    W = NE.masks_of(c["words"])
    true_rows = set(int(r) for r in c["true_row"] if r >= 0)
    sel = [int(r) for r in c["sel"]]
    n_true, tp_E, nsel_E = feat["n_true"], feat["emitted"]["tp"], feat["emitted"]["n_selected"]
    k_true = int(c["meta"]["k"])
    L = int(c["headline"].rsplit("_L", 1)[1])
    pool = args.calib_pool or CB.pool_of(ds)
    # a network that contributed no trainable block is in no fold's training set either way; its
    # hash fold is where calib.py would have put it
    # a transferred dataset (calib.TRANSFER, or any dataset under --calib-pool) is in no fold and
    # takes the all-data fit
    transfer = bool(args.calib_pool) or ds in CB.TRANSFER
    fold = (None if transfer
            else calib["fold_of_group"].get(feat["group"], CB.fold_of(feat["group"], calib["folds"])))
    blocks = [b for b in feat["blocks"] if not b["excluded"]]
    adjs = [B.compat_graph([NE.unhex(x) for x in b["options"]["abits"]]) for b in blocks]
    # oversized blocks: the configurations within rho swaps, once -- they do not depend on the variant
    t_r = time.perf_counter()
    radii = [B.radius_configs(a, sum(1 << i for i in b["members"]), args.rho)
             if b["r_count"] > NE.R_CAP else None for b, a in zip(blocks, adjs)]
    t_r = time.perf_counter() - t_r
    out = {"dataset": ds, "cell": name, "group": feat["group"], "thr": thr, "rho": args.rho,
           "n_radius_blocks": sum(1 for r in radii if r is not None), "seconds_radius": t_r,
           "pool": pool,
           "fold": fold, "n_true": n_true, "emitted_tp": tp_E, "emitted_n_sel": nsel_E,
           "emitted_f1": f1(tp_E, nsel_E, n_true), "variants": {}, "problems": [], "warnings": []}
    real_cache: dict = {}
    emitted = realise_selection(c["words"], sel, n, taxa)
    if emitted["enewick"] != c["decode"]["enewick"]:
        out["problems"].append("N0: the emitted selection does not realise the landed eNewick")

    def realise(chosen: dict) -> dict:
        """``chosen``: block index -> configuration bitmask (absent = emitted configuration)."""
        key = tuple(sorted(chosen.items()))
        if key in real_cache:
            return real_cache[key]
        rows = set(sel)
        for bi, cfg in chosen.items():
            blk = blocks[bi]
            o = blk["options"]["row"]
            rows -= {o[i] for i in blk["members"]}
            rows |= {o[i] for i in B.bits(cfg)}
        res = realise_selection(c["words"], rows, n, taxa)
        res["tp"] = len(rows & true_rows)
        res["n_sel"] = len(rows)
        real_cache[key] = res
        return res

    for v in args.variants:
        if v not in calib["variants"]:
            continue
        crec = calib["variants"][v]
        cols = crec["columns"]
        model = crec["folds"][fold] if fold is not None and fold < len(crec["folds"]) else None
        if model is None:
            model = crec["all"]
            if not transfer:
                out["warnings"].append(f"{v}: no held-out fold for {feat['group']}; used the all-data fit")
        dists = [block_dist(b, ds, model, cols, a, r, args.rho) for b, a, r in zip(blocks, adjs, radii)]
        vr = {"blocks": [], "fit": ("transfer" if transfer
                                    else "fold" if model is not crec["all"] else "all")}
        for b, d in zip(blocks, dists):
            vr["blocks"].append({
                "id": b["id"], "category": b["category"], "r_count": b["r_count"],
                "mode": d["mode"], "rho": d["rho"], "mass_by_radius": d["mass_by_radius"],
                "map_radius": d["map_radius"],
                "n_listed": d["n_listed"], "exact": d["exact"], "z_gap": d["z_gap"],
                "rstar_rank": d["rstar_rank"], "rstar_p": d["rstar_p"],
                "rstar_above": d["rstar_above"], "emit_rank": d["emit_rank"], "emit_p": d["emit_p"],
                "cs": d["cs"], "p": d["p"][:KEEP_STORED], "dtp": d["dtp"][:KEEP_STORED],
                "dsel": d["dsel"][:KEEP_STORED], "marg": d["marg"]})
        multi = [i for i, d in enumerate(dists) if d["n_listed"] >= 2]

        # MAP and the per-block oracle
        tp_map = tp_E + sum(d["dtp"][0] for d in dists)
        ns_map = nsel_E + sum(d["dsel"][0] for d in dists)
        vr["map"] = {"tp": tp_map, "n_sel": ns_map, "f1": f1(tp_map, ns_map, n_true),
                     "n_moved": sum(1 for d in dists if d["emit_rank"] != 0),
                     "p_map": math.exp(sum(math.log(d["p"][0]) for d in dists)) if dists else 1.0}
        vr["oracle"] = dinkelbach(dists, tp_E, nsel_E, n_true)

        # Hamming balls inside CS_b(ALPHA_BALL)
        csz = []
        for d in dists:
            k = d["cs"][f"{NE.ALPHA_BALL:g}"]
            csz.append(k if k > 0 else d["n_listed"])
        q0 = [d["p"][0] for d in dists]
        q1 = [sum(d["p"][1:k]) for d, k in zip(dists, csz)]
        dp = [1.0]
        for a0, a1 in zip(q0, q1):
            nxt = [0.0] * (len(dp) + 1)
            for j, x in enumerate(dp):
                nxt[j] += x * a0
                nxt[j + 1] += x * a1
            dp = nxt
        dev = [(i, j) for i, k in enumerate(csz) for j in range(1, k)]
        # F1 sees a deviation only through (dtp, dsel) relative to the MAP: dedupe per block first
        moves = [{(d["dtp"][j] - d["dtp"][0], d["dsel"][j] - d["dsel"][0]) for j in range(1, k)}
                 for d, k in zip(dists, csz)]
        best1 = vr["map"]["f1"]
        for mv in moves:
            for a_, b_ in mv:
                best1 = max(best1, f1(tp_map + a_, ns_map + b_, n_true))
        best2 = best1
        for mv, mv2 in itertools.combinations([mv for mv in moves if mv], 2):
            for (a_, b_), (a2, b2) in itertools.product(mv, mv2):
                best2 = max(best2, f1(tp_map + a_ + a2, ns_map + b_ + b2, n_true))
        s1 = 1 + len(dev)
        s2 = s1 + sum((csz[i] - 1) * (csz[i2] - 1) for i, i2 in itertools.combinations(range(len(csz)), 2))
        vr["ball"] = {"alpha": NE.ALPHA_BALL, "size_d1": s1, "size_d2": s2,
                      "mass_d0": dp[0] if dp else 1.0, "mass_d1": sum(dp[:2]), "mass_d2": sum(dp[:3]),
                      "best_f1_d1": best1, "best_f1_d2": best2,
                      "sum_cs": {f"{a:g}": sum((d["cs"][f"{a:g}"] if d["cs"][f"{a:g}"] > 0 else d["n_listed"])
                                               for d in dists) for a in NE.ALPHAS},
                      "log10_prod_cs": {f"{a:g}": sum(math.log10(d["cs"][f"{a:g}"] if d["cs"][f"{a:g}"] > 0
                                                                 else d["n_listed"]) for d in dists)
                                        for a in NE.ALPHAS}}

        # joint lazy k-best
        lists = [[math.log(x) for x in dists[i]["p"][:NE.K_CAP]] for i in multi]
        curve, mass, best, best_at = [], 0.0, -1.0, {}
        top = []
        for rank, (s, idx) in enumerate(B.kbest_product(lists, NE.K_CAP), 1):
            tp = tp_map + sum(dists[i]["dtp"][j] - dists[i]["dtp"][0] for i, j in zip(multi, idx))
            ns = ns_map + sum(dists[i]["dsel"][j] - dists[i]["dsel"][0] for i, j in zip(multi, idx))
            fv = f1(tp, ns, n_true)
            mass += math.exp(s + sum(math.log(dists[i]["p"][0]) for i in range(len(dists)) if i not in multi))
            if fv > best:
                best, best_at[rank] = fv, idx
            top.append((s, fv, idx))
            if rank & (rank - 1) == 0:
                curve.append({"K": rank, "best_f1": best, "mass": mass})
        # a product with fewer than K_CAP members is exhausted: its top K, for every larger K, is the
        # whole product.  Record those K too, or the per-K means run over ever fewer (and larger) cells.
        K = 1 << len(top).bit_length()
        while len(top) < NE.K_CAP and K <= NE.K_CAP:
            curve.append({"K": K, "best_f1": best, "mass": mass, "exhausted": True})
            K <<= 1
        vr["kbest"] = curve
        vr["kbest_n"] = len(top)
        if len(multi) and math.prod(len(lst) for lst in lists) <= N5_MAX and all(dists[i]["exact"] for i in multi):
            brute = sorted((sum(t) for t in itertools.product(*lists)), reverse=True)[:len(top)]
            if any(abs(a - b[0]) > 1e-9 for a, b in zip(brute, top)):
                out["problems"].append(f"N5 {v}: lazy k-best differs from brute-force enumeration")
            vr["n5_checked"] = True

        # realise: MAP, best of the top 4 / 16, and every d = 1 member
        def chosen_of(idx_by_block: dict) -> dict:
            return {i: dists[i]["cl"][j] for i, j in idx_by_block.items()}

        mapc = {i: dists[i]["cl"][0] for i in range(len(dists))}
        nets = {"map": realise({i: c_ for i, c_ in mapc.items()})}
        for K in NE.K_LUAY:
            cand = top[:K]
            if cand:
                s, fv, idx = max(cand, key=lambda t: (t[1], t[0]))
                ch = dict(mapc)
                ch.update(chosen_of(dict(zip(multi, idx))))
                nets[f"best{K}"] = realise(ch)
        n_ball, n_ill = 0, 0
        for i, j in dev:
            ch = dict(mapc)
            ch[i] = dists[i]["cl"][j]
            got = realise(ch)
            n_ball += 1
            add_tp = tp_map - dists[i]["dtp"][0] + dists[i]["dtp"][j]
            add_ns = ns_map - dists[i]["dsel"][0] + dists[i]["dsel"][j]
            if got["tp"] != add_tp or got["n_sel"] != add_ns:
                out["problems"].append(f"N5 {v}: block {blocks[i]['id']} config {j}: realised counts "
                                       f"({got['tp']}, {got['n_sel']}) != additive ({add_tp}, {add_ns})")
            if not legal_as_emitted(got, emitted, k_true, L):
                n_ill += 1
        for key_, got in nets.items():
            if not legal_as_emitted(got, emitted, k_true, L):
                n_ill += 1
                out["problems"].append(f"N1 {v}: {key_} is not legal as the emitted network is")
        if n_ill:
            out["problems"].append(f"N1 {v}: {n_ill} realised members are illegal or change r/level")
        if nets["map"]["tp"] != tp_map or nets["map"]["n_sel"] != ns_map:
            out["problems"].append(f"N5 {v}: the MAP's realised counts differ from the additive ones")
        vr["nets"] = {k_: {f: g.get(f) for f in ("enewick", "tp", "n_sel", "r", "level", "k_pred")}
                      for k_, g in nets.items()}
        vr["n_ball_realised"] = n_ball
        out["variants"][v] = vr
    out["emitted_net"] = {f: emitted.get(f) for f in ("enewick", "r", "level", "k_pred")}
    out["seconds"] = time.perf_counter() - t0
    NE.write_json(cred_path(args.runs_root, ds, name, thr, args.rho), out)
    return {"cell": name, "seconds": out["seconds"], "problems": out["problems"],
            "warnings": out["warnings"]}


def legal_as_emitted(got: dict, emitted: dict, k: int, L: int) -> bool:
    return (got["ok"] and got["indeg"] <= 2 and got["r"] <= k and got["level"] is not None
            and got["level"] <= L and got["r"] == emitted["r"] and got["level"] == emitted["level"])


def dinkelbach(dists: list[dict], tp_E: int, ns_E: int, n_true: int) -> dict:
    """``max F1`` over the product of every block's listed configurations: maximise the ratio
    ``2 tp / (n_sel + n_true)`` exactly by Dinkelbach iterations on the separable ``2 dtp - lam dsel``."""
    lam = f1(tp_E, ns_E, n_true)
    pick = [d["emit_rank"] if d["emit_rank"] >= 0 else 0 for d in dists]
    for _ in range(100):
        new = [max(range(len(d["dtp"])), key=lambda j: (2 * d["dtp"][j] - lam * d["dsel"][j], -j))
               for d in dists]
        tp = tp_E + sum(d["dtp"][j] for d, j in zip(dists, new))
        ns = ns_E + sum(d["dsel"][j] for d, j in zip(dists, new))
        val = f1(tp, ns, n_true)
        if val <= lam + 1e-12:
            break
        lam, pick = val, new
    tp = tp_E + sum(d["dtp"][j] for d, j in zip(dists, pick))
    ns = ns_E + sum(d["dsel"][j] for d, j in zip(dists, pick))
    return {"f1": f1(tp, ns, n_true), "tp": tp, "n_sel": ns,
            "n_moved": sum(1 for d, j in zip(dists, pick) if j != d["emit_rank"])}


def _worker(payload):
    args, row, calib = payload
    try:
        return run_cell(args, row, calib)
    except Exception as exc:  # noqa: BLE001
        return {"cell": row["name"], "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=NE.ALL_DATASETS)
    ap.add_argument("--phases", default="")
    ap.add_argument("--reps", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--cells", type=Path, default=None)
    ap.add_argument("--thr", type=float, default=NE.Z_THR)
    ap.add_argument("--rho", type=int, default=NE.RHO,
                    help="radius of an oversized block's configuration set (in the cache key)")
    ap.add_argument("--variants", default=",".join(NE.VARIANTS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runs-root", type=Path, default=NE.RUNS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib-pool", default=None,
                    help="score every cell with <runs-root>/calib/<POOL>__z<thr>.json's all-data fit, "
                         "as a transfer (fit = transfer, no fold): trainfit.py's 'train'.  Off by "
                         "default, and then the pool is calib.pool_of(dataset), as published")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    pool = args.calib_pool or CB.pool_of(args.dataset)
    cpath = Path(args.runs_root) / "calib" / f"{pool}__{NE.thr_tag(args.thr)}.json"
    if not cpath.exists():
        raise SystemExit(f"no model at {cpath}: run calib.py --pool {pool} first")
    calib = json.loads(cpath.read_text())
    rows = NE.cells(args.dataset, args.phases, args.reps, args.arms, args.cells)
    rows = [r for r in rows if (Path(args.runs_root) / args.dataset / "feat"
                                / f"{r['name']}__{NE.thr_tag(args.thr)}.json").exists()]
    todo = [r for r in rows if args.force or not cred_path(args.runs_root, args.dataset, r["name"],
                                                            args.thr, args.rho).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[credible] {args.dataset} {NE.cred_tag(args.thr, args.rho)}: {len(rows)} featured cells, "
          f"{len(todo)} to run, jobs {args.jobs}", flush=True)
    t0 = time.perf_counter()
    payloads = [(args, r, calib) for r in todo]
    errors, probs, warns, done = [], [], 0, 0
    it = (get_context("fork").Pool(args.jobs).imap_unordered(_worker, payloads)
          if args.jobs > 1 and len(todo) > 1 else map(_worker, payloads))
    for r in it:
        done += 1
        if "error" in r:
            errors.append(r)
        else:
            probs += [f"{r['cell']}: {p}" for p in r["problems"]]
            warns += len(r["warnings"])
        if done % max(1, len(todo) // 20) == 0 or done == len(todo):
            print(f"[credible]   {done}/{len(todo)} | {time.perf_counter() - t0:.0f} s", flush=True)
    for e in errors:
        print(f"[credible] FAILED {e['cell']}: {e['error']}\n{e['trace']}", file=sys.stderr)
    for p in probs:
        print(f"[credible] PROBLEM {p}", file=sys.stderr)
    print(f"[credible] done: {done - len(errors)} ok, {len(errors)} failed, {len(probs)} problems, "
          f"{warns} all-data-fit fallbacks (degenerate folds), {time.perf_counter() - t0:.0f} s", flush=True)
    return 1 if errors or probs else 0


if __name__ == "__main__":
    sys.exit(main())
