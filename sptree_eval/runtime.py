#!/usr/bin/env python3
"""Seconds per network: RIDDLE (k=0), RIDDLE (sweep) and ASTRAL-IV, per (size, rep) -- see README.md.

What each method is charged, as ``nni_ensemble/runtime.py`` charges RIDDLE where no clean add-on
timing exists (its ``recorded_upper``):

* **RIDDLE at k** = the scored cell's ``seconds`` (``build``, ``forward``, ``decode``) + the feat file's
  ``seconds.total`` + the cred file's ``seconds``.  Every one is a ``perf_counter`` wall time of a
  single-threaded worker, so CPU <= wall.  Two of them are upper bounds:
  * ``decode`` times all four decodes (gnn / greedy x relaxed_L1 / level1), and RIDDLE uses one;
  * features regenerate the rooted gene trees, which the build already holds in production.
  Import, model load, and the calibration fit are not charged.  The train-fit beta is fitted once on
  the GNN's training networks and reused unchanged, so it is a one-off cost of the method, like
  training the GNN.
* **RIDDLE (k=0)** = the build + RIDDLE at k = 0.
* **RIDDLE (sweep)** = the build once (it does not depend on k; ``score_ksweep`` shares it) + RIDDLE
  at every k the vote reads: the k = 0 backbone and ``sweep.VOTE_KS``.  Each k is its own forward
  pass.  ``sweep_k10_s`` is the same sum over the whole k = 0..10 sweep that was run.
* **ASTRAL-IV** = ``run.json``'s ``wall`` (``cpu`` beside it), single-threaded (``-t 1``).

Hosts and load: all of it ran on the M4 Max, the RIDDLE stages 6 at a time beside at most 5 ASTRAL-IV
runs, with at most 11 of the 12 P-cores busy (README, Costs).  So the times are uncontended.

Rows (``results/sptree_runtime.csv``): one per (size, rep, method), with ``wall_s`` (what the figure
draws), ``cpu_s`` (ASTRAL-IV only), and RIDDLE's parts ``build_s forward_s decode_s feat_s cred_s``,
summed over the k it is charged for.

Gates (``results/sptree_runtime_gates.json``; exit 1 on any):

    T1  every (size, rep) has both RIDDLE rows; ASTRAL-IV has a row wherever its run.json is ok
    T2  the build is shared: the same seconds, flagged ``build_shared``, at every k of a (size, rep)
    T3  every RIDDLE part, at every k charged, is a positive finite time
    T4  every ASTRAL-IV run used one thread, and cpu <= 1.05 x wall + 0.05 s

Usage:  gnn_cpu_v1/.venv/bin/python sptree_eval/runtime.py   (after score / rest / astral)
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter

import collate as CO
import sweep as SW

SP, NE = CO.SP, CO.NE
PARTS = ("build_s", "forward_s", "decode_s", "feat_s", "cred_s")
COLS = ("size", "m", "rep", "cu", "key", "method", "wall_s", "cpu_s", *PARTS, "sweep_k10_s", "rss_mb")


def at_k(size: str, stem: str, k: int, runs_root) -> dict:
    ds = NE.CA.sptree_name(size, k)
    sc = json.loads(NE.cell_paths(ds, stem)[1].read_text())["seconds"]
    feat = json.loads((runs_root / ds / "feat" / f"{stem}__{NE.thr_tag(NE.Z_THR)}.json").read_text())
    cred = json.loads((runs_root / ds / "cred" / f"{stem}__{NE.cred_tag(NE.Z_THR, NE.RHO)}.json").read_text())
    return {"build": sc["build"], "shared": sc.get("build_shared", False), "forward_s": sc["forward"],
            "decode_s": sc["decode"], "feat_s": feat["seconds"]["total"], "cred_s": cred["seconds"]}


def main() -> int:
    t0 = time.perf_counter()
    runs_root = SP.RUNS / "ne"
    gates = {g: [] for g in ("T1", "T2", "T3", "T4")}
    rows = []
    for size, m, _ in SP.SIZES:
        for c in SP.stage()["cells"]:
            rep, stem = c["rep"], SP.stem(c["rep"])
            base = {"size": size, "m": m, "rep": rep, "cu": c["cu"], "key": c["key"]}
            tag = f"{size} rep{rep:02d}"
            try:
                ks = {k: at_k(size, stem, k, runs_root) for k in SP.KS}
            except FileNotFoundError as e:
                gates["T1"].append(f"{tag}: {e.filename}")
                continue
            build = ks[0]["build"]
            bad = [(k, p) for k, v in ks.items() for p in PARTS[1:] if not (math.isfinite(v[p]) and v[p] > 0)]
            if bad or not (math.isfinite(build) and build > 0):
                gates["T3"].append(f"{tag}: non-positive times at (k, part) {bad[:4]}, build {build}")
            if any(not v["shared"] or v["build"] != build for v in ks.values()):
                gates["T2"].append(f"{tag}: build seconds differ across k, or are not shared")
            for method, charged in (("RIDDLE (k=0)", (0,)), ("RIDDLE (sweep)", (0, *SW.VOTE_KS))):
                r = {**base, "method": method, "build_s": build,
                     **{p: sum(ks[k][p] for k in charged) for p in PARTS[1:]}}
                r["wall_s"] = sum(r[p] for p in PARTS)
                if method == "RIDDLE (sweep)":
                    r["sweep_k10_s"] = build + sum(ks[k][p] for k in SP.KS for p in PARTS[1:])
                rows.append(r)
            aj = SP.RUNS / "astral" / size / stem / "run.json"
            if aj.exists():
                a = json.loads(aj.read_text())
                if not a.get("ok"):
                    continue
                if a["threads"] != 1 or a["cpu"] > 1.05 * a["wall"] + 0.05:
                    gates["T4"].append(f"{tag}: threads {a['threads']}, cpu {a['cpu']:.1f} wall {a['wall']:.1f}")
                rows.append({**base, "method": "ASTRAL-IV", "wall_s": a["wall"], "cpu_s": a["cpu"],
                             "rss_mb": a["rss_mb"]})

    CO.write_csv(SP.RESULTS / "sptree_runtime.csv", rows, COLS)
    have = Counter((r["size"], r["method"]) for r in rows)
    SP.write_atomic(SP.RESULTS / "sptree_runtime_gates.json", json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "vote_ks": list(SW.VOTE_KS),
        "have": {f"{s} {m}": n for (s, m), n in sorted(have.items())},
        "gates": {g: {"ok": not v, "n": len(v), "first": v[:20]} for g, v in gates.items()},
    }, indent=1) + "\n")
    print(f"[runtime] {len(rows)} rows | " + " ".join(f"{s} {m}: {n}" for (s, m), n in sorted(have.items()))
          + " | gates " + " ".join(f"{g}:{'ok' if not v else f'FAIL({len(v)})'}" for g, v in gates.items())
          + f" | {time.perf_counter() - t0:.0f} s")
    for g, v in gates.items():
        for x in v[:5]:
            print(f"[runtime]   {g} {x}")
    return 1 if any(gates.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
