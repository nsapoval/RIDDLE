#!/usr/bin/env python3
"""Stage B7: every table, network file and gate tally of the z+tc ensemble, rebuilt from the feat and
cred JSONs each time (nothing here is cached, so re-running it is always safe).

Writes under ``<out-root>/ens/``:

* ``cells.tsv`` -- one row per (cell, pool, variant): the low-z set and its classes, blocks
  (multi-cluster, radius-mode, hybrid-excluded, merged), the size ladder (``log10_3M``, ``log10_nni``,
  ``log10_prod_R``), the MAP (``map_moved``, ``n_moved``, ``log10_p_map``, ``log10_p_emitted``), the
  balls, credible-set sizes, the top-K mass, ``tc`` coverage and the out-of-distribution maxima;
* ``blocks.tsv.gz`` -- one row per (cell, pool, variant, block);
* ``options.tsv.gz`` -- one row per (cell, pool, variant, allowed cluster): z, tc, support, its
  marginal ``P(Y in network)`` and whether it is emitted / in the MAP; its taxa as a hex mask over the
  sorted label order;
* ``lowz.tsv.gz`` -- one row per low-z emitted cluster (class, block, NNI counts);
* ``kbest.tsv`` -- the joint top-K mass curve;
* ``agreement.tsv`` -- per cell: does the MAP agree across variants and pools;
* ``<scheme>/nets/fungi_<scheme>_tau<T>_L2map-<ztc|z>-<pool>.tsv`` -- ``k<TAB>eNewick``, the format
  Stage A's ``.tsv`` (and the consensus) reads; ``..._L2top16-*.tsv.gz`` (k, rank, log10_p, eNewick)
  and ``..._L2ball1-*.tsv.gz`` (k, block, cfg_rank, log10_p, eNewick) -- gzipped, because a ball holds
  up to 5,000 full 646-taxon networks per cell;
* ``gates.json`` and ``summary.txt``.

Gates: **N8** -- no truth-shaped key anywhere in a feat or cred JSON; **vacuity** -- blocks of size >= 2
exist and some cell's MAP differs from the emitted network (``--strict`` exits 2 on either being
zero); and every cred file's own ``problems`` list must be empty.

    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_tables.py
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import fz_common as FZ

TRUTH_KEYS = {"true", "n_fp", "n_tp", "category", "rstar", "rstar_rank", "rstar_p", "rstar_above",
              "dtp", "well_posed", "oracle", "best_f1_d1", "best_f1_d2", "best_f1", "f1", "tp",
              "fold", "nni_true", "nni_true_any", "best_alt_true", "classes_fp", "n_low_fp",
              "n_fp_emitted", "member_true", "n_true", "T_region", "n_true_region",
              "n_true_not_allowed", "n_true_not_in_ctau"}
VTAG = {"z+tc": "ztc", "z": "z"}


def walk_keys(obj, found: set) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in TRUTH_KEYS:
                found.add(k)
            walk_keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            walk_keys(v, found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=FZ.ANALYSIS)
    ap.add_argument("--zthr", type=float, default=FZ.NE.Z_THR)
    ap.add_argument("--rho", type=int, default=FZ.NE.RHO)
    ap.add_argument("--strict", action="store_true", help="exit 2 on a vacuous gate")
    args = ap.parse_args()
    ttag, ctag = FZ.NE.thr_tag(args.zthr), FZ.NE.cred_tag(args.zthr, args.rho)
    ens = FZ.ens_dir(args.out_root)
    cells, blocks_t, opts_t, lowz_t, kbest_t = [], [], [], [], []
    agree = defaultdict(dict)
    nets = defaultdict(list)          # (scheme, tau, kind, vtag, pool) -> rows
    truth_found: set = set()
    problems, warnings = [], []
    tally = defaultdict(int)
    for scheme in FZ.SCHEMES:
        for fp in sorted((ens / scheme / "feat").glob(f"*__{ttag}.json")):
            f = FZ.read_json(fp)
            walk_keys(f, truth_found)
            tally["feat_files"] += 1
            bl = f["blocks"]
            tally["blocks"] += len(bl)
            tally["blocks_multi"] += sum(1 for b in bl if b["size"] >= 2)
            tally["blocks_hybrid_excluded"] += sum(1 for b in bl if b["excluded"] == "hybrid_atom")
            tally["n2a_checked"] += sum(1 for b in bl if b.get("n2a") is not None)
            tally["blocks_merged"] += sum(1 for b in bl if b["n_merged"])
            tally["blocks_p_crosses"] += sum(1 for b in bl if b["p_crosses"])
            for r in f["lowz"]:
                lowz_t.append({"cell": f["cell"], "scheme": scheme, "tau": f["tau"], "k": f["k"], **r})
            for pool in FZ.POOLS:
                cp = FZ.cred_path(args.out_root, scheme, f["tau"], f["k"], args.zthr, args.rho, pool)
                if not cp.exists():
                    continue
                c = FZ.read_json(cp)
                walk_keys(c, truth_found)
                tally["cred_files"] += 1
                problems += [f"{c['cell']} {pool}: {p}" for p in c["problems"]]
                warnings += [f"{c['cell']} {pool}: {w}" for w in c["warnings"]]
                if c["npz_sha256"] != f["npz_sha256"]:
                    problems.append(f"{c['cell']} {pool}: cred and feat are for different npz")
                live = [b for b in bl if not b["excluded"]]
                for v, vr in c["variants"].items():
                    vt = VTAG[v]
                    moved = vr["map"]["enewick"] != c["emitted_net"]["enewick"]
                    tally[f"map_moved_{pool}_{vt}"] += moved
                    agree[c["cell"]][f"{pool}_{vt}"] = vr["map"]["enewick"]
                    cs9 = [b["cs"]["0.9"] for b in vr["blocks"]]
                    oodmax = defaultdict(float)
                    for b in vr["blocks"]:
                        for col, x in b["ood"].items():
                            oodmax[col] = max(oodmax[col], x)
                    cells.append({
                        "cell": c["cell"], "scheme": scheme, "tau": f["tau"], "k": f["k"], "pool": pool,
                        "variant": v, "n_clusters": f["n_clusters"], "m": f["m"],
                        "n_emitted": f["n_emitted"], "n_retics": f["emitted"]["n_retics"],
                        "level": f["emitted"]["level"], "M": f["M"],
                        "frac_low": f["M"] / max(1, f["n_emitted"]),
                        **{f"n_{cl}": f["classes"].get(cl, 0) for cl in ("Q", "crosses", "multi_parent", "blob_top")},
                        "n_blocks": len(bl), "n_blocks_multi": sum(1 for b in bl if b["size"] >= 2),
                        "largest_block": max((b["size"] for b in bl), default=0),
                        "max_m": max((b["m"] for b in bl), default=0),
                        "max_a": max((b.get("n_allowed") or 0 for b in bl), default=0),
                        "n_hybrid_excluded": sum(1 for b in bl if b["excluded"] == "hybrid_atom"),
                        "n_merged": sum(1 for b in bl if b["n_merged"]),
                        "n_radius": c["n_radius_blocks"],
                        "log10_3M": f["log10_3M"], "log10_nni": f["log10_nni"],
                        "log10_prod_R": sum(math.log10(max(1, b["r_count"])) for b in live),
                        "map_moved": moved, "n_moved": vr["map"]["n_moved"],
                        "log10_p_map": vr["map"]["log10_p_map"],
                        "log10_p_emitted": vr["map"]["log10_p_emitted"],
                        "map_n_sel": vr["map"]["n_sel"], "map_r": vr["map"]["r"],
                        "ball_size_d1": vr["ball"]["size_d1"], "ball_size_d2": vr["ball"]["size_d2"],
                        "ball_mass_d0": vr["ball"]["mass_d0"], "ball_mass_d1": vr["ball"]["mass_d1"],
                        "ball_mass_d2": vr["ball"]["mass_d2"],
                        **{f"sum_cs{a}": vr["ball"]["sum_cs"][a] for a in vr["ball"]["sum_cs"]},
                        **{f"log10_prod_cs{a}": vr["ball"]["log10_prod_cs"][a] for a in vr["ball"]["log10_prod_cs"]},
                        "kbest_mass_16": next((x["mass"] for x in vr["kbest"] if x["K"] == 16), None),
                        "kbest_mass_1024": next((x["mass"] for x in vr["kbest"] if x["K"] == 1024), None),
                        "n_truncated": sum(1 for b in vr["blocks"] if not b["exact"]),
                        "max_z_gap": max((b["z_gap"] for b in vr["blocks"]), default=0.0),
                        "max_cs0.9": max(cs9, default=0),
                        "tc_nan": sum(b.get("tc_nan", 0) for b in live),
                        **{f"ood_{col}": x for col, x in oodmax.items()},
                        "ball1_truncated": vr["ball1_truncated"], "n_illegal": vr["n_illegal"],
                        "seconds_feat": f["seconds"]["total"], "seconds_cred": c["seconds"]})
                    for b, fb in zip(vr["blocks"], live):
                        if b["id"] != fb["id"]:
                            problems.append(f"{c['cell']}: cred block order differs from feat")
                            break
                        blocks_t.append({
                            "cell": c["cell"], "scheme": scheme, "tau": f["tau"], "k": f["k"], "pool": pool,
                            "variant": v, "block": b["id"], "size": fb["size"], "m": fb["m"],
                            "n_merged": fb["n_merged"], "p_size": fb["p_size"], "p_is_root": fb["p_is_root"],
                            "p_crosses": fb["p_crosses"], "n_allowed": fb["n_allowed"],
                            "r_count": fb["r_count"], "mode": b["mode"], "n_listed": b["n_listed"],
                            "exact": b["exact"], "z_gap": b["z_gap"], "emit_rank": b["emit_rank"],
                            "emit_p": b["emit_p"], "map_p": b["p"][0] if b["p"] else None,
                            "map_radius": b["map_radius"],
                            **{f"cs{a}": b["cs"][a] for a in b["cs"]},
                            **{f"mass_r{i}": x for i, x in enumerate(b["mass_by_radius"] or [])},
                            **{f"delta_{col}": x for col, x in b["delta_by_feature"].items()},
                            "tc_nan": fb.get("tc_nan"), "R_min": fb.get("R_min"),
                            "R_median": fb.get("R_median"),
                            "member_z_min": min(fb["member_z"]) if fb["member_z"] else None})
                        o = fb["options"]
                        mapset = set()
                        if b["configs"]:
                            mem = set(fb["members"])
                            mapset = (mem - set(b["configs"][0]["drop"])) | set(b["configs"][0]["add"])
                        for i in range(len(o["row"])):
                            opts_t.append({
                                "cell": c["cell"], "pool": pool, "variant": v, "block": b["id"],
                                "cand_row": o["row"][i], "abits": o["abits"][i], "z": o["z"][i],
                                "tc": o["tc"][i], "support": o["support"][i], "freq": o["freq"][i],
                                "k_atoms": o["k"][i], "emitted": o["emitted"][i], "in_map": i in mapset,
                                "marg": b["marg"][i], "margin_z": o["margin_z"][i],
                                "margin_tc": o["margin_tc"][i]})
                    for x in vr["kbest"]:
                        kbest_t.append({"cell": c["cell"], "pool": pool, "variant": v, **x})
                    nets[(scheme, f["tau"], "L2map", vt, pool)].append(
                        {"k": f["k"], "enewick": vr["map"]["enewick"]})
                    for t in vr["top"]:
                        nets[(scheme, f["tau"], "L2top16", vt, pool)].append({"k": f["k"], **t})
                    for t in vr["ball1"]:
                        nets[(scheme, f["tau"], "L2ball1", vt, pool)].append({"k": f["k"], **t})
    for (scheme, tau, kind, vt, pool), rows in nets.items():
        rows.sort(key=lambda r: (r["k"], r.get("rank", 0), r.get("block", 0), r.get("cfg_rank", 0)))
        path = ens / scheme / "nets" / (f"fungi_{scheme}_tau{tau}_{kind}-{vt}-{pool}.tsv"
                                        + ("" if kind == "L2map" else ".gz"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "L2map":
            path.write_text("".join(f"{r['k']}\t{r['enewick']}\n" for r in rows))
        else:
            FZ.write_tsv(path, rows)
    ag = []
    for cell, d in sorted(agree.items()):
        vals = list(d.values())
        ag.append({"cell": cell, **{k: v == d.get(f"{FZ.P.HEADLINE_POOL}_ztc") for k, v in d.items()},
                   "all_agree": len(set(vals)) == 1})
    FZ.write_tsv(ens / "cells.tsv", cells)
    FZ.write_tsv(ens / "blocks.tsv.gz", blocks_t)
    FZ.write_tsv(ens / "options.tsv.gz", opts_t)
    FZ.write_tsv(ens / "lowz.tsv.gz", lowz_t)
    FZ.write_tsv(ens / "kbest.tsv", kbest_t)
    FZ.write_tsv(ens / "agreement.tsv", ag)
    if truth_found:
        problems.append(f"N8: truth-shaped keys present: {sorted(truth_found)}")
    vac = []
    if tally["feat_files"] and not tally["blocks_multi"]:
        vac.append("no block of size >= 2")
    if tally["cred_files"] and not any(v for k, v in tally.items() if k.startswith("map_moved_")):
        vac.append("no cell's MAP differs from the emitted network")
    gates = {"ok": not problems, "problems": problems, "vacuous": vac, "warnings": warnings,
             "tally": dict(tally), "n_cells_rows": len(cells)}
    FZ.write_json(ens / "gates.json", gates, indent=1)
    lines = [f"z+tc ensemble tables: {tally['feat_files']} feat files, {tally['cred_files']} cred files",
             f"blocks {tally['blocks']} (size >= 2: {tally['blocks_multi']}, hybrid-excluded "
             f"{tally['blocks_hybrid_excluded']}, merged {tally['blocks_merged']}, P crosses "
             f"{tally['blocks_p_crosses']}), N2(a) checked on {tally['n2a_checked']}",
             "MAP != emitted: " + ", ".join(f"{k[10:]} {v}" for k, v in sorted(tally.items())
                                             if k.startswith("map_moved_")),
             f"problems {len(problems)}, warnings {len(warnings)}, vacuous {vac or 'none'}"]
    (ens / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for p in problems[:50]:
        print(f"[tables] PROBLEM {p}", file=sys.stderr)
    if problems:
        return 1
    return 2 if (vac and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
