#!/usr/bin/env python3
"""The sptree tables: every method against the true species tree, per (size, k, cell) -- see README.md.

``nni_ensemble/make_csv.py`` and ``bench.py`` do not apply here: both pool calibration and luay
parity against a landed table this dataset does not have, and HAE is undefined against a truth with
no reticulation.  So this reads the three sources directly:

* the scored cells (``score_ksweep``), for the emitted decodes;
* the cred files (``credible --calib-pool train``), for **RIDDLE** = ``GNN MAP [z+tc]``;
* ``runs/astral/<size>/<stem>/``, for ASTRAL-IV.

Rows (``results/sptree_rows.csv``), one per (size, k, rep, method):

    RIDDLE               the z+tc MAP of the relaxed-L1 decode, train-fit beta     (k = 0..10)
    GNN [relaxed L1]     the decode RIDDLE starts from (the emitted network)       (k = 0..10)
    GNN strict L1        the formal level-1 decode, same forward                   (k = 0..10)
    greedy [relaxed L1]  the relaxed-L1 decode ranked by support                   (k = 0..10)
    ASTRAL-IV            one species tree per (size, rep); ``k`` is empty

Columns: ``tp fp fn n_sel precision recall f1`` -- the landed definition, the selection's clusters
against the 198 non-trivial true clusters (``Instance.cluster_counts``; ASTRAL: its tree's clusters);
``net_tp net_n net_f1`` -- the same for the realised network's hardwired clusters (the strict-L1
closure can add clusters a selection lacks); ``k_pred`` and ``level`` of the realised network;
``rf`` / ``rf_norm`` -- unrooted RF on the 200 taxa over ``2 (n - 3)``, when the prediction is a tree;
``map_moved`` / ``p_map`` (RIDDLE); ``luay`` / ``luay_norm_exact`` with ``--luay`` (the PhyloNet jar,
cached in ``<runs-root>/phylonet/cache.jsonl``).

Gates (``results/sptree_gates.json``; exit 1 on any, ``--partial`` relaxes only C1's counts):

    C1  1,650 scored cells, 1,650 creds, 150 ASTRAL trees
    C2  every cred is pool ``train``, z+tc fit ``transfer``, no problems, no warnings
    C3  each cred's emitted network and F1 are the scored cell's relaxed-L1 decode
    C4  ``k_pred <= k`` for every decode and RIDDLE (its MAP's ``r`` too); all 0 at k = 0
    C5  a MAP that moved no block is the emitted network
    C6  at k = 0 every prediction is a tree, and its network counts are its selection counts
    C7  every truth has 198 non-trivial clusters and no reticulation
    C8  every ASTRAL run passed A1, and its rooted tree is the one run.json hashed

Usage:  gnn_cpu_v1/.venv/bin/python sptree_eval/collate.py [--partial] [--luay] [--jobs 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import sp_common as SP

os.environ.setdefault("NE_ANAT_RUNS", str(SP.RUNS / "anat"))
sys.path.insert(0, str(SP.REPO / "nni_ensemble"))

import ne_common as NE  # noqa: E402  -- after sp_common: BLAS pinned, pylib on sys.path

from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402

DECODES = (("GNN [relaxed L1]", "gnn", "relaxed_L1"), ("GNN strict L1", "gnn", "level1"),
           ("greedy [relaxed L1]", "greedy", "relaxed_L1"))
RIDDLE = "RIDDLE"
ASTRAL = "ASTRAL-IV"
METHODS = (RIDDLE, "GNN [relaxed L1]", "GNN strict L1", "greedy [relaxed L1]", ASTRAL)
COLS = ("size", "m", "k", "rep", "cu", "key", "method", "tp", "fp", "fn", "n_sel", "precision", "recall",
        "f1", "net_tp", "net_n", "net_f1", "k_pred", "level", "rf", "rf_norm", "map_moved", "p_map",
        "luay", "luay_norm_exact", "enewick")


def counts(tp: int, n_sel: int, n_true: int = SP.N_TRUE) -> dict:
    p = tp / n_sel if n_sel else 0.0
    r = tp / n_true
    return {"tp": tp, "fp": n_sel - tp, "fn": n_true - tp, "n_sel": n_sel, "precision": p, "recall": r,
            "f1": 2 * tp / (n_sel + n_true) if tp else 0.0}


def splits(clusters: set[frozenset[str]]) -> set[frozenset[str]]:
    """Unrooted non-trivial splits of a rooted tree on t1..t200, each as the side without ``t1``."""
    out = set()
    for c in clusters:
        s = c if "t1" not in c else SP.TAXA - c
        if 1 < len(s) < SP.N_TAXA - 1:
            out.add(frozenset(s))
    return out


def network_metrics(enewick: str | None, T: set, TS: set) -> dict:
    if not enewick:
        return {"net_tp": None, "net_n": None, "net_f1": None, "k_pred": None, "level": None,
                "rf": None, "rf_norm": None}
    net = NW.from_enewick(enewick)
    cl = CN.label_clusters(net)
    tp = len(cl & T)
    out = {"net_tp": tp, "net_n": len(cl), "net_f1": 2 * tp / (len(cl) + len(T)) if tp else 0.0,
           "k_pred": int(net.n_retics), "level": int(NW.level(net)), "rf": None, "rf_norm": None}
    if net.n_retics == 0:
        rf = len(splits(cl) ^ TS)
        out.update(rf=rf, rf_norm=rf / (2 * (SP.N_TAXA - 3)))
    return out


def truth_of(rep: int):
    t = CAMUS.truth_of(CAMUS.CamusCell(SP.PHASE, rep, SP.ARM, SP.INPUTS))
    return t, CN.label_clusters(t)


def unit(payload) -> dict:
    """Every row of one (size, rep): 11 k x 4 GNN-side methods, and ASTRAL."""
    size, cell, runs_root = payload
    rep, stem = cell["rep"], SP.stem(cell["rep"])
    truth, T = truth_of(rep)
    TS = splits(T)
    base = {"size": size, "m": SP.SIZE_M[size], "rep": rep, "cu": cell["cu"], "key": cell["key"]}
    rows, gates, have = [], {f"C{i}": [] for i in range(1, 9)}, {"score": 0, "cred": 0, "astral": 0}
    if len(T) != SP.N_TRUE or truth.n_retics:
        gates["C7"].append(f"{stem}: truth has {len(T)} clusters, {truth.n_retics} reticulations")
    for k in SP.KS:
        ds = NE.CA.sptree_name(size, k)
        sp = NE.cell_paths(ds, stem)[1]
        if not sp.exists():
            continue
        have["score"] += 1
        sc = json.loads(sp.read_text())
        tag = f"{ds} {stem}"
        if sc["n_true"] != SP.N_TRUE:
            gates["C7"].append(f"{tag}: the scored cell's n_true is {sc['n_true']}")
        for label, method, dtag in DECODES:
            d = sc["decodes"][f"{method}|{dtag}"]
            r = {**base, "k": k, "method": label, **counts(d["tp"], d["n_selected"]),
                 **network_metrics(d["enewick"], T, TS), "enewick": d["enewick"]}
            if abs(r["f1"] - d["f1"]) > 1e-12:
                gates["C3"].append(f"{tag} {label}: f1 {r['f1']} recomputed, {d['f1']} scored")
            if not 0 <= d["k_pred"] <= k or (r["k_pred"] is not None and r["k_pred"] != d["k_pred"]):
                gates["C4"].append(f"{tag} {label}: k_pred {d['k_pred']} (network {r['k_pred']}) at k {k}")
            if k == 0 and (r["k_pred"] != 0 or (r["net_tp"], r["net_n"]) != (r["tp"], r["n_sel"])):
                gates["C6"].append(f"{tag} {label}: at k = 0 k_pred {r['k_pred']}, net {r['net_tp']}/"
                                   f"{r['net_n']} against selection {r['tp']}/{r['n_sel']}")
            rows.append(r)
        cp = runs_root / ds / "cred" / f"{stem}__{NE.cred_tag(NE.Z_THR, NE.RHO)}.json"
        if not cp.exists():
            continue
        have["cred"] += 1
        cr = json.loads(cp.read_text())
        v = cr["variants"]["z+tc"]
        if cr["pool"] != "train" or v["fit"] != "transfer" or cr["problems"] or cr["warnings"]:
            gates["C2"].append(f"{tag}: pool {cr['pool']} fit {v['fit']} problems {cr['problems'][:2]} "
                               f"warnings {cr['warnings'][:2]}")
        emit = sc["decodes"]["gnn|relaxed_L1"]
        if cr["emitted_net"]["enewick"] != emit["enewick"] or abs(cr["emitted_f1"] - emit["f1"]) > 1e-12:
            gates["C3"].append(f"{tag}: the cred's emitted network / F1 is not the scored decode's")
        mp, mnet = v["map"], v["nets"]["map"]
        r = {**base, "k": k, "method": RIDDLE, **counts(mp["tp"], mp["n_sel"]),
             **network_metrics(mnet["enewick"], T, TS), "map_moved": mp["n_moved"], "p_map": mp["p_map"],
             "enewick": mnet["enewick"]}
        if mnet["r"] is None or mnet["r"] > k or (r["k_pred"] is not None and r["k_pred"] > k):
            gates["C4"].append(f"{tag} RIDDLE: MAP r {mnet['r']} k_pred {r['k_pred']} at k {k}")
        if mp["n_moved"] == 0 and mnet["enewick"] != emit["enewick"]:
            gates["C5"].append(f"{tag}: the MAP moved no block but is not the emitted network")
        if k == 0 and (r["k_pred"] != 0 or (r["net_tp"], r["net_n"]) != (r["tp"], r["n_sel"])):
            gates["C6"].append(f"{tag} RIDDLE: at k = 0 k_pred {r['k_pred']}, net {r['net_tp']}/{r['net_n']} "
                               f"against selection {r['tp']}/{r['n_sel']}")
        rows.append(r)
    aj = SP.RUNS / "astral" / size / stem / "run.json"
    if aj.exists():
        a = json.loads(aj.read_text())
        rp = aj.parent / "species.rooted.nwk"
        if not a.get("ok") or not rp.exists() or SP.sha256(rp) != a.get("species_rooted_sha256"):
            gates["C8"].append(f"{size} {stem}: ASTRAL run not ok ({a.get('problems')}) or its rooted tree "
                               f"is not the one run.json hashed")
        else:
            have["astral"] += 1
            nw = rp.read_text().strip()
            net = network_metrics(nw, T, TS)
            rows.append({**base, "k": "", "method": ASTRAL, **counts(net["net_tp"], net["net_n"]), **net,
                         "enewick": nw})
    return {"rows": rows, "gates": gates, "have": have, "size": size, "rep": rep}


def attach_luay(rows: list[dict], runs_root: Path, jobs: int) -> dict:
    import make_sub_csv as MS
    cache = runs_root / "phylonet" / "cache.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    luay = MS.PhyloNetLuay(Path(os.environ.get("OPC_PHYLONET", str(NE.SC.DEFAULT_JAR))), cache, jobs=jobs)
    truths, idx = {}, []
    for r in rows:
        if not r["enewick"]:
            continue
        if r["rep"] not in truths:
            truths[r["rep"]] = MS.strip_gamma(NW.to_enewick(truth_of(r["rep"])[0]))
        idx.append((r, luay.add(MS.strip_gamma(r["enewick"]), truths[r["rep"]])))
    luay.resolve(runs_root / "phylonet" / "work")
    for r, i in idx:
        got = luay.result(i)
        r["luay"] = got["luay"]
        r["luay_norm_exact"] = got["luay"] / got["max_exact"] if got["luay"] is not None and got["max_exact"] else None
    return {"pairs": len(idx), "measured": luay.measured, "failed": luay.failed}


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict = {}
    for r in rows:
        for cu in ("all", r["cu"]):
            groups.setdefault((r["size"], r["k"], r["method"], cu), []).append(r)
    order = {t: i for i, (t, _, _) in enumerate(SP.SIZES)}
    for (size, k, method, cu), g in sorted(groups.items(), key=lambda kv: (
            order[kv[0][0]], -1 if kv[0][1] == "" else kv[0][1], METHODS.index(kv[0][2]), str(kv[0][3]))):
        rec = {"size": size, "k": k, "method": method, "cu": cu, "n": len(g)}
        for col in ("f1", "precision", "recall", "net_f1", "k_pred", "rf_norm", "map_moved", "luay_norm_exact"):
            v = [r[col] for r in g if r.get(col) is not None]
            rec[f"{col}_mean"] = st.fmean(v) if v else None
            rec[f"{col}_median"] = st.median(v) if v else None
            rec[f"{col}_se"] = (st.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else None
            rec[f"{col}_n"] = len(v)
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict], cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    tmp.replace(path)


def text_summary(summ: list[dict]) -> str:
    f = lambda x: "    -" if x is None else f"{x:.3f}"  # noqa: E731
    lines = ["sptree_n200_gt10k_v1: mean cluster F1 against the true species tree (198 clusters), "
             "all 50 cells; k_pred mean in brackets", ""]
    by = {(s["size"], s["k"], s["method"]): s for s in summ if s["cu"] == "all"}
    for size, _, _ in SP.SIZES:
        a = by.get((size, "", ASTRAL))
        lines.append(f"== {size} gene trees" + (f" | ASTRAL-IV F1 {f(a['f1_mean'])} (n {a['n']}), "
                                                 f"RF_norm {f(a['rf_norm_mean'])}" if a else " | ASTRAL-IV not run"))
        lines.append(f"  {'k':>3s} " + " ".join(f"{m:>22s}" for m in METHODS[:-1]))
        for k in SP.KS:
            cells = []
            for m in METHODS[:-1]:
                s = by.get((size, k, m))
                if s is None:
                    cells.append("-")
                    continue
                kp = "-" if s["k_pred_mean"] is None else f"{s['k_pred_mean']:.1f}"
                cells.append(f"{f(s['f1_mean'])} [{kp}] n{s['n']}")
            lines.append(f"  {k:>3d} " + " ".join(f"{c:>22s}" for c in cells))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=SP.RUNS / "ne", help="the NE runs root (creds)")
    ap.add_argument("--out-dir", type=Path, default=SP.RESULTS)
    ap.add_argument("--partial", action="store_true", help="a smoke: C1 asks for >= 1 of each, not all")
    ap.add_argument("--luay", action="store_true")
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    runs_root = args.runs_root.resolve()

    t0 = time.perf_counter()
    stg = SP.stage()
    payloads = [(size, c, runs_root) for size, _, _ in SP.SIZES for c in stg["cells"]]
    with get_context("fork").Pool(args.jobs) as pool:
        res = list(pool.imap_unordered(unit, payloads))
    rows = sorted((r for u in res for r in u["rows"]),
                  key=lambda r: ([t for t, _, _ in SP.SIZES].index(r["size"]), r["rep"],
                                 -1 if r["k"] == "" else r["k"], METHODS.index(r["method"])))
    gates = {f"C{i}": [] for i in range(1, 9)}
    for u in res:
        for g, v in u["gates"].items():
            gates[g] += v
    have = {x: sum(u["have"][x] for u in res) for x in ("score", "cred", "astral")}
    want = {"score": 50 * 3 * len(SP.KS), "cred": 50 * 3 * len(SP.KS), "astral": 50 * 3}
    for x in have:
        if have[x] < (1 if args.partial else want[x]):
            gates["C1"].append(f"{x}: {have[x]} of {want[x]}")
    luay = attach_luay(rows, runs_root, args.jobs) if args.luay else None

    summ = summarise(rows)
    out = args.out_dir
    write_csv(out / "sptree_rows.csv", rows, COLS)
    write_csv(out / "sptree_summary.csv", summ, list(summ[0]) if summ else ["size"])
    txt = text_summary(summ)
    SP.write_atomic(out / "sptree_summary.txt", txt)
    bad = {g: v for g, v in gates.items() if v}
    SP.write_atomic(out / "sptree_gates.json", json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "partial": args.partial, "have": have, "want": want,
        "luay": luay, "gates": {g: {"ok": not v, "n": len(v), "first": v[:20]} for g, v in gates.items()},
    }, indent=1) + "\n")
    print(txt)
    print(f"[collate] {len(rows)} rows | have {have} of {want} | gates "
          + " ".join(f"{g}:{'ok' if not v else f'FAIL({len(v)})'}" for g, v in gates.items())
          + f" | {time.perf_counter() - t0:.0f} s -> {out}")
    for g, v in bad.items():
        for x in v[:5]:
            print(f"[collate]   {g} {x}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
