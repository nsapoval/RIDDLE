#!/usr/bin/env python3
"""Stage B6: per relaxed-L2 cell and calibration pool, the z+tc ensemble -- ``P_b`` over every block's
configurations, the MAP network, credible sets, Hamming balls, marginals and the joint top-K -- and
the networks they stand for.  ``nni_ensemble/credible.py``, truth-free.

The model is the calibration's **all-data** fit, ``calib/<pool>__z6.json`` ``variants[v]["all"]``:
fungi is in no fold, so no fold applies.  Under ``FZ_BETA=train`` the one pool is ``train``,
``nni_ensemble/runs_trainfit/calib/train__z6.json`` (``fz_common.calib_path``), whose columns carry no
dataset offsets.  Its columns must equal ``calib.columns(v, pool)``; the
dataset offsets of the sim pool are 0 here (``calib.option_matrix`` with ``ds = "fungi"``), the camus
baseline.  ``w_Y = beta . (phi(Y) - mean) / sd`` with the fit's own moments, never fungi's.

What is removed, not faked: ``true``, ``dtp``, ``rstar*``, every F1, the oracle (``dinkelbach``) and the
best-of-K picks (an oracle by construction).  What is added: per block, ``ood`` (the largest
|standardised feature| over its options, per column) and ``delta_by_feature`` (beta_f times the MAP's
minus the emitted configuration's summed standardised feature -- which feature moved it); the listed
configurations as ``add`` / ``drop`` option indices; and the realised networks of the MAP, the joint
top-16 and every d = 1 ball member, each with its model probability.

Gates: **N0** (the emitted selection realises the run's eNewick; the feat file is for this npz),
**N1** (every realised network is legal with the emitted network's reticulation number and level,
r <= k, level <= 2), **N5** (the realised selection's size equals the additive count; the lazy top-K
equals brute-force product enumeration where the product is <= 10^4 and every block exact).

    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_credible.py --cells short:tau2:k29 --pools sim --jobs 1
    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_credible.py --pools sim,netcs --jobs 5
    FZ_BETA=train gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_credible.py --pools train --jobs 5
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time

import fz_common as FZ

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402
import calib as CB  # noqa: E402
import credible as CR  # noqa: E402
import fz_blocks as FB  # noqa: E402

SCHEMA = "fz-cred-v1"
KEEP_STORED = CR.KEEP_STORED
KEEP_CFG = 64
TOP_NETS = 16
LOG10 = math.log(10)

_CAL: dict = {}


def load_calib(pool: str, thr: float) -> tuple[dict, str]:
    p = FZ.calib_path(pool, thr)
    if p not in _CAL:
        _CAL[p] = (FZ.read_json(p), FZ.sha256(p))
    return _CAL[p]


def block_dist_tf(blk: dict, model: dict, cols, adj: list[int], radius: list[int] | None,
                  rho: int) -> dict:
    """``credible.block_dist`` (``credible.py:78-130``) without ``true``/``dtp``/``rstar``."""
    o = blk["options"]
    mean, sd, beta = (np.asarray(model[x]) for x in ("mean", "sd", "beta"))
    Xs = (CB.option_matrix(blk, "fungi", cols) - mean) / sd
    w = (Xs @ beta).tolist()
    members = sum(1 << i for i in blk["members"])
    if radius is None:
        got = B.bk_best(adj, w, FZ.NE.KEEP_MAX, FZ.NE.MASS_TAIL, max_nodes=FZ.NE.NODE_CAP)
    else:
        ws = sorted(((sum(w[i] for i in B.bits(c)), c) for c in radius), key=lambda t: -t[0])
        got = {"cliques": [c for _, c in ws], "logw": [x for x, _ in ws], "exact": True,
               "log_frontier": -math.inf}
    cl, lw = got["cliques"], got["logw"]
    z_lo = CR.lse(lw)
    z_hi = CR.lse(lw + [got["log_frontier"]])
    p = [math.exp(x - z_hi) for x in lw]
    logp = [x - z_hi for x in lw]
    dsel = [c.bit_count() - len(blk["members"]) for c in cl]
    rank_of = {c: i for i, c in enumerate(cl)}
    e_rank = rank_of.get(members, -1)
    cum = np.cumsum(p)
    cs = {}
    for a in FZ.NE.ALPHAS:
        k = int(np.searchsorted(cum, a - 1e-12) + 1)
        cs[f"{a:g}"] = k if k <= len(p) else -1
    marg = [0.0] * len(o["row"])
    for c, pc in zip(cl, p):
        for i in B.bits(c):
            marg[i] += pc
    mass_r = None
    if radius is not None:
        mass_r = [0.0] * (rho + 1)
        for c, pc in zip(cl, p):
            mass_r[(c & ~members).bit_count()] += pc
    phi = lambda c: Xs[list(B.bits(c))].sum(axis=0) if c else np.zeros(len(cols))  # noqa: E731
    delta = (beta * (phi(cl[0]) - phi(members))).tolist() if cl else [0.0] * len(cols)
    return {"cl": cl, "logw": lw, "p": p, "logp": logp, "dsel": dsel,
            "mode": "full" if radius is None else "radius", "rho": None if radius is None else rho,
            "mass_by_radius": mass_r, "map_radius": (cl[0] & ~members).bit_count() if cl else None,
            "exact": got["exact"], "z_gap": 1.0 - math.exp(z_lo - z_hi) if z_hi > -math.inf else 0.0,
            "n_listed": len(cl), "emit_rank": e_rank, "emit_p": p[e_rank] if e_rank >= 0 else None,
            "cs": cs, "marg": marg, "members_mask": members,
            "delta_by_feature": dict(zip(cols, delta)),
            "ood": dict(zip(cols, np.abs(Xs).max(axis=0).tolist())) if Xs.size else {},
            "ood_share_gt4": dict(zip(cols, (np.abs(Xs) > 4).mean(axis=0).tolist())) if Xs.size else {}}


def run_cell(args, unit) -> dict:
    scheme, tau, k = unit
    t0 = time.perf_counter()
    thr, rho, L = args.zthr, args.rho, FZ.LEVEL
    feat = FZ.read_json(FZ.feat_path(args.out_root, scheme, tau, k, thr))
    cell = FZ.load_cell(args.gnn_root, scheme, tau, k)
    if feat.get("schema") != "fz-feat-v1" or feat["npz_sha256"] != cell["npz_sha"]:
        raise AssertionError(f"{cell['id']}: the feat file is not for this npz (re-run fz_features) (N0)")
    words, n, taxa = cell["words"], cell["n"], cell["taxa"]
    sel = [int(r) for r in cell["sel"]]
    nsel_E = len(sel)
    blocks = [b for b in feat["blocks"] if not b["excluded"]]
    abits = [[FZ.NE.unhex(x) for x in b["options"]["abits"]] for b in blocks]
    adjs = [FB.compat_graph_np(ab, b["m"]) for b, ab in zip(blocks, abits)]
    t_r = time.perf_counter()
    radii = []
    for b, a in zip(blocks, adjs):
        if b["r_count"] > FZ.NE.R_CAP:
            rc = b.get("radius_candidates") or FB.radius_candidates(a, sum(1 << i for i in b["members"]), rho)
            if rc > FZ.CAPS["RAD_CAND_CAP"]:
                raise FZ.CapExceeded(cell["id"], b["id"], "radius candidates", rc, FZ.CAPS["RAD_CAND_CAP"])
            r = B.radius_configs(a, sum(1 << i for i in b["members"]), rho)
            if len(r) > FZ.CAPS["RAD_CONF_CAP"]:
                raise FZ.CapExceeded(cell["id"], b["id"], "radius configurations", len(r),
                                     FZ.CAPS["RAD_CONF_CAP"])
            radii.append(r)
        else:
            radii.append(None)
    t_r = time.perf_counter() - t_r
    emitted = CR.realise_selection(words, sel, n, taxa)
    problems_all = []
    if emitted["enewick"] != cell["enewick"]:
        problems_all.append("N0: the emitted selection does not realise the run's eNewick")
    real_cache: dict = {}

    def realise(chosen: dict) -> dict:
        key = tuple(sorted(chosen.items()))
        if key in real_cache:
            return real_cache[key]
        rows = set(sel)
        for bi, cfg in chosen.items():
            o = blocks[bi]["options"]["row"]
            rows -= {o[i] for i in blocks[bi]["members"]}
            rows |= {o[i] for i in B.bits(cfg)}
        res = CR.realise_selection(words, rows, n, taxa)
        res["n_sel"] = len(rows)
        real_cache[key] = res
        return res

    out_files = []
    for pool in args.pools:
        calib, csha = load_calib(pool, thr)
        out = {"schema": SCHEMA, "truth_free": True, "cell": cell["id"], "scheme": scheme, "tau": tau,
               "k": k, "thr": thr, "rho": rho, "pool": pool, "calib_sha256": csha, "fit": "all",
               "npz_sha256": cell["npz_sha"], "triples_sha256": feat.get("triples_sha256"),
               "n_blocks": len(blocks), "n_radius_blocks": sum(1 for r in radii if r is not None),
               "seconds_radius": t_r, "variants": {}, "problems": list(problems_all), "warnings": [],
               "emitted_net": {f: emitted.get(f) for f in ("enewick", "r", "level", "k_pred")},
               "emitted_n_sel": nsel_E,
               "note": "model probabilities from a choice model calibrated on simulations, not on real data",
               **FZ.P.beta_tag()}
        for v in args.variants:
            crec = calib["variants"][v]
            cols = crec["columns"]
            if list(cols) != list(CB.columns(v, pool)):
                raise AssertionError(f"{pool} {v}: calibration columns {cols} != calib.columns")
            model = crec["all"]
            dists = [block_dist_tf(b, model, cols, a, r, rho) for b, a, r in zip(blocks, adjs, radii)]
            vr = {"blocks": []}
            for b, d in zip(blocks, dists):
                mem = d["members_mask"]
                cfgs = [{"rank": i, "logp": d["logp"][i],
                         "add": [j for j in B.bits(c & ~mem)], "drop": [j for j in B.bits(mem & ~c)]}
                        for i, c in enumerate(d["cl"][:KEEP_CFG])]
                vr["blocks"].append({
                    "id": b["id"], "r_count": b["r_count"], "mode": d["mode"], "rho": d["rho"],
                    "mass_by_radius": d["mass_by_radius"], "map_radius": d["map_radius"],
                    "n_listed": d["n_listed"], "exact": d["exact"], "z_gap": d["z_gap"],
                    "emit_rank": d["emit_rank"], "emit_p": d["emit_p"], "cs": d["cs"],
                    "p": d["p"][:KEEP_STORED], "dsel": d["dsel"][:KEEP_STORED], "marg": d["marg"],
                    "configs": cfgs, "delta_by_feature": d["delta_by_feature"], "ood": d["ood"],
                    "ood_share_gt4": d["ood_share_gt4"]})
            multi = [i for i, d in enumerate(dists) if d["n_listed"] >= 2]
            ns_map = nsel_E + sum(d["dsel"][0] for d in dists)
            log_map = sum(d["logp"][0] for d in dists)
            vr["map"] = {"n_sel": ns_map, "n_moved": sum(1 for d in dists if d["emit_rank"] != 0),
                         "log10_p_map": log_map / LOG10,
                         "log10_p_emitted": (sum(d["logp"][d["emit_rank"]] for d in dists) / LOG10
                                             if all(d["emit_rank"] >= 0 for d in dists) else None)}

            # Hamming balls inside CS_b(ALPHA_BALL) (credible.py:233-267, masses and sizes only)
            csz = []
            for d in dists:
                kk = d["cs"][f"{FZ.NE.ALPHA_BALL:g}"]
                csz.append(kk if kk > 0 else d["n_listed"])
            q0 = [d["p"][0] for d in dists]
            q1 = [sum(d["p"][1:kk]) for d, kk in zip(dists, csz)]
            dp = [1.0]
            for a0, a1 in zip(q0, q1):
                nxt = [0.0] * (len(dp) + 1)
                for j, x in enumerate(dp):
                    nxt[j] += x * a0
                    nxt[j + 1] += x * a1
                dp = nxt
            dev = [(i, j) for i, kk in enumerate(csz) for j in range(1, kk)]
            s1 = 1 + len(dev)
            s2 = s1 + sum((csz[i] - 1) * (csz[i2] - 1) for i, i2 in itertools.combinations(range(len(csz)), 2))
            vr["ball"] = {"alpha": FZ.NE.ALPHA_BALL, "size_d1": s1, "size_d2": s2,
                          "mass_d0": dp[0] if dp else 1.0, "mass_d1": sum(dp[:2]), "mass_d2": sum(dp[:3]),
                          "sum_cs": {f"{a:g}": sum((d["cs"][f"{a:g}"] if d["cs"][f"{a:g}"] > 0
                                                    else d["n_listed"]) for d in dists)
                                     for a in FZ.NE.ALPHAS},
                          "log10_prod_cs": {f"{a:g}": sum(math.log10(d["cs"][f"{a:g}"] if d["cs"][f"{a:g}"] > 0
                                                                     else d["n_listed"]) for d in dists)
                                            for a in FZ.NE.ALPHAS}}

            # the joint lazy top-K: the mass curve, and the first TOP_NETS members
            lists = [[d_ for d_ in dists[i]["logp"][:FZ.NE.K_CAP]] for i in multi]
            fixed = sum(dists[i]["logp"][0] for i in range(len(dists)) if i not in multi)
            curve, mass, top = [], 0.0, []
            for rank, (s, idx) in enumerate(B.kbest_product(lists, FZ.NE.K_CAP), 1):
                mass += math.exp(s + fixed)
                top.append((s + fixed, idx))
                if rank & (rank - 1) == 0:
                    curve.append({"K": rank, "mass": mass})
            K = 1 << len(top).bit_length()
            while len(top) < FZ.NE.K_CAP and K <= FZ.NE.K_CAP:
                curve.append({"K": K, "mass": mass, "exhausted": True})
                K <<= 1
            vr["kbest"], vr["kbest_n"] = curve, len(top)
            if multi and math.prod(len(x) for x in lists) <= CR.N5_MAX and all(dists[i]["exact"] for i in multi):
                brute = sorted((sum(t) for t in itertools.product(*lists)), reverse=True)[:len(top)]
                if any(abs(a - (b[0] - fixed)) > 1e-9 for a, b in zip(brute, top)):
                    out["problems"].append(f"N5 {v}: lazy k-best differs from brute-force enumeration")
                vr["n5_checked"] = True

            # realise: the MAP, the top-16, every d = 1 ball member
            mapc = {i: d["cl"][0] for i, d in enumerate(dists)}
            n_ill = 0

            def check(got, what):
                nonlocal n_ill
                if not CR.legal_as_emitted(got, emitted, k, L):
                    n_ill += 1
                    out["problems"].append(f"N1 {v}: {what} is not legal as the emitted network is")

            got = realise(dict(mapc))
            check(got, "the MAP")
            if got["n_sel"] != ns_map:
                out["problems"].append(f"N5 {v}: the MAP's realised size {got['n_sel']} != additive {ns_map}")
            vr["map"].update({f: got.get(f) for f in ("enewick", "r", "level", "k_pred")})
            nets_top = []
            for rank, (s, idx) in enumerate(top[:TOP_NETS], 1):
                ch = dict(mapc)
                ch.update({i: dists[i]["cl"][j] for i, j in zip(multi, idx)})
                g = realise(ch)
                check(g, f"top-K member {rank}")
                nets_top.append({"rank": rank, "log10_p": s / LOG10,
                                 "n_moved_vs_map": sum(1 for j in idx if j), "enewick": g.get("enewick")})
            vr["top"] = nets_top
            order = sorted(dev, key=lambda ij: -(dists[ij[0]]["logp"][ij[1]] - dists[ij[0]]["logp"][0]))
            if len(order) > FZ.CAPS["BALL_CAP"]:
                out["warnings"].append(f"{v}: {len(order)} d=1 ball members; realised the top "
                                       f"{FZ.CAPS['BALL_CAP']} by probability")
            ball = []
            for i, j in order[:FZ.CAPS["BALL_CAP"]]:
                ch = dict(mapc)
                ch[i] = dists[i]["cl"][j]
                g = realise(ch)
                check(g, f"ball block {blocks[i]['id']} config {j}")
                want = ns_map - dists[i]["dsel"][0] + dists[i]["dsel"][j]
                if g["n_sel"] != want:
                    out["problems"].append(f"N5 {v}: block {blocks[i]['id']} config {j}: size "
                                           f"{g['n_sel']} != additive {want}")
                ball.append({"block": blocks[i]["id"], "cfg_rank": j,
                             "log10_p": (log_map - dists[i]["logp"][0] + dists[i]["logp"][j]) / LOG10,
                             "enewick": g.get("enewick")})
            vr["ball1"] = ball
            vr["ball1_truncated"] = len(order) > FZ.CAPS["BALL_CAP"]
            vr["n_illegal"] = n_ill
            out["variants"][v] = vr
        out["seconds"] = time.perf_counter() - t0
        out["rss_mb"] = FZ.peak_rss_mb()
        FZ.write_json(FZ.cred_path(args.out_root, scheme, tau, k, thr, rho, pool), out)
        out_files.append(out)
    return {"unit": cell["id"], "seconds": time.perf_counter() - t0, "rss_mb": FZ.peak_rss_mb(),
            "problems": [p for o in out_files for p in o["problems"]],
            "warnings": [w for o in out_files for w in o["warnings"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    FZ.add_cell_args(ap)
    ap.add_argument("--pools", default=FZ.POOLS[0],
                    help="calibration pools: sim (headline), netcs; train alone under FZ_BETA=train")
    ap.add_argument("--variants", default=",".join(FZ.VARIANTS))
    ap.add_argument("--rho", type=int, default=FZ.NE.RHO)
    args = ap.parse_args()
    args.pools = [p for p in args.pools.split(",") if p]
    args.variants = [v for v in args.variants.split(",") if v]
    for p in args.pools:
        if p not in FZ.POOLS:
            raise SystemExit(f"--pools {p}: expected one of {FZ.POOLS}")
    units = FZ.select_cells(args)
    have = [u for u in units if FZ.feat_path(args.out_root, *u, args.zthr).exists()]
    todo = [u for u in have if args.force or not all(
        FZ.cred_path(args.out_root, *u, args.zthr, args.rho, p).exists() for p in args.pools)]
    print(f"[credible] {len(units)} cells, {len(have)} with features, {len(todo)} to run, pools "
          f"{args.pools}, variants {args.variants}, jobs {args.jobs}", flush=True)
    t0 = time.perf_counter()
    ok, err = FZ.run_units(todo, lambda u: run_cell(args, u), args.jobs, "credible")
    probs = [f"{r['unit']}: {p}" for r in ok for p in r["problems"]]
    for p in probs:
        print(f"[credible] PROBLEM {p}", file=sys.stderr)
    if ok:
        sec = np.array([r["seconds"] for r in ok])
        print(f"[credible] per cell: median {np.median(sec):.1f} s, max {sec.max():.1f} s; peak RSS "
              f"{max(r['rss_mb'] for r in ok):.0f} MB; warnings {sum(len(r['warnings']) for r in ok)}",
              flush=True)
    print(f"[credible] done: {len(ok)} ok, {len(err)} failed, {len(probs)} problems, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return 1 if err or probs else 0


if __name__ == "__main__":
    sys.exit(main())
