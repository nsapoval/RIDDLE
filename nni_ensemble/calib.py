#!/usr/bin/env python3
"""The block-level choice model: a cross-fitted conditional logit over each block's configurations.

``P_b(r) = exp(beta . Phi_r) / sum_{r' in R_b} exp(beta . Phi_r')``, ``Phi_r = sum_{Y in r} phi(Y)``.
Fitted by maximising ``sum_b log P_b(r*_b)`` -- each block once -- over the blocks whose truth is
reachable (category A), whose ``R_b`` is enumerated exactly (``r_count <= R_CAP``) and has a choice in
it (``r_count >= 2``).  Newton's method with a small ridge; ``beta``'s scale is the temperature.

Calibrating the configuration probability directly, rather than a per-cluster ``P(true)``, is the
point: a calibrated per-cluster ``pi`` does not give calibrated ``P_b`` (``pi = (0.6, 0.5, 0.1)`` on a
three-option block gives ``(0.57, 0.38, 0.04)``), and pooled cluster rows would be dominated by the
few blocks with hundreds of options.

Variants (``phi`` columns; the ``1`` is a per-cluster intercept, since maximal families differ in size):

* ``z``         -- 1, z, margin_z, freq, k
* ``tc``        -- 1, tc, margin_tc
* ``z+tc``      -- ``z`` + tc, margin_tc
* ``z+tc+emit`` -- ``z+tc`` + emitted (observable at inference; its coefficient is what the decoder's
  own choice adds beyond z)

In the ``sim`` pool (camus, l2reg, l3reg) every variant adds per-cluster dataset offsets.  No feature is
derived from the truth.

Folds are **grouped by network** (``group`` = dataset:phase:rep): NetCS has four ILS arms per network,
and splitting them would leak the topology.  Gate N4 checks no group sits on both sides of a fold.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python nni_ensemble/calib.py --pool sim
    python nni_ensemble/calib.py --pool netcs --thr 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import ne_common as NE

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402

POOLS = {"sim": ("camus", "l2reg", "l3reg"), "netcs": ("netcs",)}
BASE = {"z": ("one", "z", "margin_z", "freq", "k"),
        "tc": ("one", "tc", "margin_tc"),
        "z+tc": ("one", "z", "margin_z", "freq", "k", "tc", "margin_tc"),
        "z+tc+emit": ("one", "z", "margin_z", "freq", "k", "tc", "margin_tc", "emitted")}
NO_SCALE = ("one", "emitted", "ds_l2reg", "ds_l3reg")


#: Datasets scored with a pool's **all-data** fit and never fitted on -- so every published fit stays
#: byte-identical.  ``credible`` marks their variants ``fit = "transfer"``.
TRANSFER = {"n1000": "sim"}
#: The per-cluster dataset offset a transferred dataset takes: n1000 is l2reg's generator family at
#: n = 1000, decoded by the same relaxed L2 (the offsets are ~0.003 in z+tc; README, n1000).
OFFSET_AS = {"n1000": "l2reg"}


def pool_of(ds: str) -> str:
    """The pool whose fit scores ``ds``: TRANSFER's, else the pool holding it -- or, for a derived
    dataset, its base (``NE.DERIVED``: never fitted on, scored by its parent network's fold)."""
    if ds in TRANSFER:
        return TRANSFER[ds]
    fam = NE.family(ds)
    return next(p for p, dss in POOLS.items() if fam in dss)


def columns(variant: str, pool: str) -> tuple[str, ...]:
    extra = ("ds_l2reg", "ds_l3reg") if pool == "sim" else ()
    return BASE[variant] + extra


def option_matrix(blk: dict, ds: str, cols) -> np.ndarray:
    """``phi`` for every option of a block, raw (unstandardised)."""
    o = blk["options"]
    a = len(o["row"])
    dso = OFFSET_AS.get(ds, NE.family(ds))      # a derived dataset takes its base's offset
    out = np.zeros((a, len(cols)))
    for j, c in enumerate(cols):
        if c == "one":
            v = np.ones(a)
        elif c == "ds_l2reg":
            v = np.full(a, 1.0 if dso == "l2reg" else 0.0)
        elif c == "ds_l3reg":
            v = np.full(a, 1.0 if dso == "l3reg" else 0.0)
        elif c == "emitted":
            v = np.asarray(o["emitted"], dtype=float)
        else:
            v = np.asarray(o[c], dtype=float)
            if c == "tc":
                v = np.where(np.isnan(v), 1.0 / 3.0, v)
            else:
                v = np.where(np.isnan(v), 0.0, v)
        out[:, j] = v
    return out


def configs(blk: dict) -> list[int]:
    """Every maximal laminar family of the block's options, as option bitmasks (exact)."""
    ab = [NE.unhex(x) for x in blk["options"]["abits"]]
    return B.bk_all(B.compat_graph(ab))


def fold_of(group: str, n_folds: int) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % n_folds


def load_training(root: Path, pool: str, thr: float) -> list[dict]:
    """Every trainable block in the pool: options' raw features, its configurations, r*'s index."""
    out = []
    for ds in POOLS[pool]:
        for p in sorted((Path(root) / ds / "feat").glob(f"*__{NE.thr_tag(thr)}.json")):
            d = json.loads(p.read_text())
            for blk in d["blocks"]:
                if (blk["excluded"] or blk["category"] != "A"
                        or not 2 <= blk["r_count"] <= NE.R_CAP):
                    continue
                cf = configs(blk)
                want = sum(1 << i for i in blk["rstar"])
                star = [i for i, c in enumerate(cf) if c == want]
                if len(star) != 1:
                    raise AssertionError(f"{d['cell']} block {blk['id']}: r* is not one of R_b")
                out.append({"ds": ds, "cell": d["cell"], "group": d["group"], "block": blk,
                            "configs": cf, "star": star[0]})
    return out


def design(ex: dict, cols, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``Phi`` (configurations x features) for one block, standardised with the given moments."""
    X = (option_matrix(ex["block"], ex["ds"], cols) - mean) / sd
    a = X.shape[0]
    S = np.zeros((len(ex["configs"]), a))
    for r, c in enumerate(ex["configs"]):
        for i in B.bits(c):
            S[r, i] = 1.0
    return S @ X


def moments(train: list[dict], cols) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([option_matrix(ex["block"], ex["ds"], cols) for ex in train])
    mean, sd = X.mean(axis=0), X.std(axis=0)
    for j, c in enumerate(cols):
        if c in NO_SCALE or sd[j] < 1e-12:
            mean[j], sd[j] = 0.0, 1.0
    return mean, sd


def loglik(beta, Phis, stars) -> tuple[float, np.ndarray, np.ndarray]:
    ll, g, H = 0.0, np.zeros_like(beta), np.zeros((beta.size, beta.size))
    for Phi, s in zip(Phis, stars):
        u = Phi @ beta
        mx = u.max()
        e = np.exp(u - mx)
        Z = e.sum()
        p = e / Z
        ll += u[s] - mx - math.log(Z)
        mu = p @ Phi
        g += Phi[s] - mu
        H -= (Phi * p[:, None]).T @ Phi - np.outer(mu, mu)
    return ll, g, H


def fit(Phis, stars, ridge: float, iters: int = 50) -> tuple[np.ndarray, float]:
    d = Phis[0].shape[1]
    beta = np.zeros(d)
    prev = -math.inf
    for _ in range(iters):
        ll, g, H = loglik(beta, Phis, stars)
        obj = ll - 0.5 * ridge * beta @ beta
        g = g - ridge * beta
        H = H - ridge * np.eye(d)
        step = np.linalg.solve(H, -g)
        t = 1.0
        while t > 1e-6:
            nb = beta + t * step
            nll = loglik(nb, Phis, stars)[0] - 0.5 * ridge * nb @ nb
            if nll >= obj - 1e-12:
                break
            t *= 0.5
        beta = nb
        if abs(nll - prev) < 1e-9 * max(1.0, abs(nll)):
            break
        prev = nll
    return beta, loglik(beta, Phis, stars)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--thr", type=float, default=NE.Z_THR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--variants", default=",".join(NE.VARIANTS))
    ap.add_argument("--runs-root", type=Path, default=NE.RUNS)
    args = ap.parse_args()
    t0 = time.perf_counter()
    exs = load_training(args.runs_root, args.pool, args.thr)
    if not exs:
        raise SystemExit("no trainable blocks: run features.py first")
    groups = sorted({ex["group"] for ex in exs})
    fold = {g: fold_of(g, args.folds) for g in groups}
    print(f"[calib] {args.pool} {NE.thr_tag(args.thr)}: {len(exs)} blocks in {len(groups)} networks, "
          f"{args.folds} folds", flush=True)

    out = {"pool": args.pool, "thr": args.thr, "folds": args.folds, "ridge": args.ridge,
           "n_blocks": len(exs), "fold_of_group": fold, "variants": {}}
    problems = []
    for v in [x for x in args.variants.split(",") if x]:
        cols = columns(v, args.pool)
        rec = {"columns": list(cols), "folds": []}
        for f in range(args.folds):
            train = [ex for ex in exs if fold[ex["group"]] != f]
            test = [ex for ex in exs if fold[ex["group"]] == f]
            tg, sg = {ex["group"] for ex in train}, {ex["group"] for ex in test}
            if tg & sg:
                problems.append(f"{v} fold {f}: {len(tg & sg)} networks on both sides (N4)")
            if not train:
                rec["folds"].append(None)
                continue
            mean, sd = moments(train, cols)
            Phis = [design(ex, cols, mean, sd) for ex in train]
            beta, ll = fit(Phis, [ex["star"] for ex in train], args.ridge)
            tPhis = [design(ex, cols, mean, sd) for ex in test]
            tll = loglik(beta, tPhis, [ex["star"] for ex in test])[0] if test else float("nan")
            null = -sum(math.log(len(ex["configs"])) for ex in test)
            rec["folds"].append({"fold": f, "beta": beta, "mean": mean, "sd": sd,
                                 "n_train": len(train), "n_test": len(test),
                                 "train_ll": ll, "test_ll": tll, "test_ll_uniform": null})
        mean, sd = moments(exs, cols)
        Phis = [design(ex, cols, mean, sd) for ex in exs]
        beta, ll = fit(Phis, [ex["star"] for ex in exs], args.ridge)
        rec["all"] = {"beta": beta, "mean": mean, "sd": sd, "ll": ll}
        out["variants"][v] = rec
        tl = sum(fr["test_ll"] for fr in rec["folds"] if fr and fr["n_test"])
        tn = sum(fr["test_ll_uniform"] for fr in rec["folds"] if fr and fr["n_test"])
        print(f"[calib]   {v:10s} held-out log-lik {tl:9.2f} (uniform {tn:9.2f}); "
              f"beta(all) = {dict(zip(cols, np.round(beta, 3)))}", flush=True)
    out["problems"] = problems
    dst = Path(args.runs_root) / "calib" / f"{args.pool}__{NE.thr_tag(args.thr)}.json"
    NE.write_json(dst, out, indent=1)
    for p in problems:
        print("[calib] PROBLEM " + p, file=sys.stderr)
    print(f"[calib] wrote {dst} in {time.perf_counter() - t0:.1f} s", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
