#!/usr/bin/env python3
"""Stage B4 (and B3's scan): per relaxed-L2 cell, the low-z clusters, the co-dependence blocks, and
each block's allowed clusters with their z / tc features -- ``nni_ensemble/features.py``, truth-free.

A cell is one (scheme, tau, k) of Stage A's ``fungi_<scheme>_tau<T>_L2`` runs.  Gates (a failure
raises naming the cell; the driver exits non-zero):

* **N0-inputs** -- the run's npz is the one ``fz_trees.py`` proved support parity for
  (``trees/<scheme>/parity.json``, by sha256), and the triple tensor was built from those trees;
* **N0-realise** -- ``RX.realise`` of the cached selection is the run's eNewick for that k, with the
  manifest's ``n_selected`` and ``n_retics``;
* **N0-structure** -- one polytomy node per block, not questioned; atoms cover it; no allowed cluster
  splits an atom; the emitted members are in ``A_b`` and maximal there; options of different blocks
  never cross;
* **N2(a)** -- a lone cluster over three unmerged atoms has exactly its 1-NNI set as ``R_b``;
* **N3-tc** -- on a three-atom block, the three options' ``tc`` sum to 1.

Caps (``fz_common.CAPS``) raise ``CapExceeded``; ``--scan`` records them instead, skips the triple
tensor, and writes ``ens/<scheme>/scan/<cell>.json`` -- which ``fz_scan.py`` summarises.

    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_features.py --cells short:tau2:k29 --jobs 1
    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_features.py --jobs 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import fz_common as FZ

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402
import fz_blocks as FB  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402

SCHEMA = "fz-feat-v1"


def scan_path(out_root: Path, scheme: str, tau: int, k: int, thr: float) -> Path:
    return FZ.ens_dir(out_root, scheme) / "scan" / f"{FZ.cell_name(tau, k)}__{FZ.NE.thr_tag(thr)}.json"


def check_inputs(args, cell: dict, need_triples: bool) -> dict:
    """N0-inputs: the npz is the one support parity was proved for; the tensor is from those trees."""
    d = FZ.trees_dir(args.out_root, cell["scheme"])
    par_path = d / "parity.json"
    if not par_path.exists():
        raise SystemExit(f"{par_path}: run fz_trees.py first")
    par = FZ.read_json(par_path)
    name = FZ.run_name(cell["scheme"], cell["tau"], FZ.HEADLINE_DEC)
    rec = par["runs"].get(name)
    if rec is None or rec["npz_sha256"] != cell["npz_sha"] or not rec["rows_identical"]:
        raise AssertionError(f"{cell['id']}: {name}.logits.npz has no support-parity record for its "
                             f"current sha in {par_path} (re-run fz_trees.py) (N0-inputs)")
    out = {"npz_sha256": cell["npz_sha"], "trees_sha256": par["trees_sha256"]}
    if need_triples:
        tm = FZ.read_json(d / "triples.json")
        if tm["trees_sha256"] != par["trees_sha256"] or tm["limit_trees"]:
            raise AssertionError(f"{cell['id']}: the triple tensor is not from the parity trees (N0-inputs)")
        if not FZ.read_json(d / "n3.json")["ok"]:
            raise AssertionError(f"{cell['id']}: the triple tensor failed N3' (n3.json)")
        out["triples_sha256"] = tm["sha256"]
    return out


def run_cell(args, unit: tuple[str, int, int], scan: bool = False) -> dict:
    scheme, tau, k = unit
    t0 = time.perf_counter()
    cell = FZ.load_cell(args.gnn_root, scheme, tau, k)
    shas = check_inputs(args, cell, need_triples=not scan)
    n, taxa = cell["n"], cell["taxa"]
    sel = cell["sel"]
    got = RX.realise(cell["words"], sel, n, taxa)
    if got is None or NW.to_enewick(got[0]) != cell["enewick"]:
        raise AssertionError(f"{cell['id']}: the npz selection does not realise the run's eNewick (N0)")
    if int(cell["per_k"]["n_selected"]) != len(sel) or int(cell["per_k"]["n_retics"]) != got[0].n_retics:
        raise AssertionError(f"{cell['id']}: n_selected / n_retics disagree with the manifest (N0)")
    count = None if scan else FZ.triples_mm(args.out_root, scheme)
    ctx = FB.make_ctx(cell, count)
    D = B.Diagram([ctx["W"][i] for i in sel], n)
    t1 = time.perf_counter()
    rec = FB.analyse_threshold_tf(D, ctx, args.zthr, cell=cell["id"], with_tc=not scan, scan=scan)
    rec.update({"schema": SCHEMA, "cell": cell["id"], "scheme": scheme, "tau": tau, "k": k, "n": n,
                "m": cell["m"], "n_clusters": int(cell["words"].shape[0]), "decoder": cell["decoder"],
                "emitted": {"n_selected": int(len(sel)), "n_retics": int(got[0].n_retics),
                            "level": int(NW.level(got[0]))},
                **shas, "seconds": {"setup": t1 - t0, "total": time.perf_counter() - t0},
                "rss_mb": FZ.peak_rss_mb()})
    path = (scan_path if scan else FZ.feat_path)(args.out_root, scheme, tau, k, args.zthr)
    FZ.write_json(path, rec)
    return {"unit": cell["id"], "seconds": rec["seconds"]["total"], "rss_mb": rec["rss_mb"],
            "cap_hits": len(rec["cap_hits"])}


def order_units(args, units: list) -> list:
    """Heaviest first: by the scan's sum of a^2 when it exists, else tau ascending, k descending."""
    def cost(u):
        p = scan_path(args.out_root, u[0], u[1], u[2], args.zthr)
        if p.exists():
            s = FZ.read_json(p)
            return -sum((b.get("n_allowed") or 0) ** 2 for b in s["blocks"])
        return (u[1], -u[2])
    return sorted(units, key=cost)


def main(scan: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    FZ.add_cell_args(ap)
    args = ap.parse_args()
    units = FZ.select_cells(args)
    path_of = scan_path if scan else FZ.feat_path
    todo = [u for u in units if args.force or not path_of(args.out_root, *u, args.zthr).exists()]
    todo = order_units(args, todo)
    tag = "scan" if scan else "features"
    print(f"[{tag}] {len(units)} cells, {len(todo)} to run, z < {args.zthr:g}, jobs {args.jobs}",
          flush=True)
    t0 = time.perf_counter()
    ok, err = FZ.run_units(todo, lambda u: run_cell(args, u, scan), args.jobs, tag)
    if ok:
        sec = np.array([r["seconds"] for r in ok])
        print(f"[{tag}] per cell: median {np.median(sec):.1f} s, max {sec.max():.1f} s; peak RSS per "
              f"worker {max(r['rss_mb'] for r in ok):.0f} MB; cap hits {sum(r['cap_hits'] for r in ok)}",
              flush=True)
    print(f"[{tag}] done: {len(ok)} ok, {len(err)} failed, {time.perf_counter() - t0:.0f} s", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
