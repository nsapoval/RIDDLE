#!/usr/bin/env python3
"""sptree_eval's standalone gates -- see README.md.

    cmp A B              every ``*.json`` / ``*.npz`` under A has an identical twin under B: JSON
                         equal once every key starting ``seconds`` is dropped (timings), npz equal
                         array by array.  G-PUB (the default-off edits leave published cells,
                         feats and creds unchanged) and G-SWEEP (a shared-build cell equals a fresh
                         build) are both this.
    cred DS [DS ...]     every cred of each dataset is a clean train-fit transfer -- pool ``train``,
                         z+tc fit ``transfer``, no problems, no warnings -- and its emitted F1 is
                         the scored cell's (C2, C3).  ``--runs-root`` is the NE runs root.
    sweep DS [DS ...]    G-K on every scored cell: ``k == extra.k_decode``, ``k_true == 0``,
                         ``k_pred <= k`` for every decode, all of them 0 at k = 0.
    kernels DS           the tail gate per cell (derived.TAIL) and the triple kernel each cell's
                         features gets: block, or dense past derived.BLOCK_MAX_M atoms.
    status all           cells scored / featured / credible per (size, k); ASTRAL runs per size.

Every command exits 1 on a failure and on an empty input, so a chain cannot pass on nothing.
Usage: ``gnn_cpu_v1/.venv/bin/python sptree_eval/check_sptree.py cmp <before> <after>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "nni_ensemble"))
# the scored cells live under sptree_eval/runs/anat (run_sptree.sh exports it; a bare call needs it too)
os.environ.setdefault("NE_ANAT_RUNS", str(HERE / "runs" / "anat"))

import numpy as np  # noqa: E402


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not k.startswith("seconds")}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _diff_json(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in {'A' if k in a else 'B'}" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += _diff_json(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff_json(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}"]


def cmp(a: Path, b: Path) -> int:
    files = sorted(p for p in a.rglob("*") if p.suffix in (".json", ".npz") and ".tmp" not in p.name)
    files = [p for p in files if "summary" not in p.name]
    if not files:
        print(f"[cmp] nothing under {a}")
        return 1
    bad = 0
    for p in files:
        q = b / p.relative_to(a)
        if not q.exists():
            print(f"[cmp] MISSING {q}")
            bad += 1
            continue
        if p.suffix == ".json":
            d = _diff_json(_strip(json.loads(p.read_text())), _strip(json.loads(q.read_text())))
        else:
            x, y = np.load(p), np.load(q)
            d = [f"key sets differ: {sorted(set(x.files) ^ set(y.files))}"] if set(x.files) != set(y.files) else []
            d += [f"{k}: arrays differ" for k in sorted(set(x.files) & set(y.files))
                  if x[k].dtype != y[k].dtype or not np.array_equal(x[k], y[k])]
        if d:
            bad += 1
            print(f"[cmp] DIFFERS {p.relative_to(a)}: {d[:5]}")
    print(f"[cmp] {len(files)} files under {a} against {b}: {len(files) - bad} identical, {bad} differ"
          f" -- {'ok' if not bad else 'FAILED'}")
    return 1 if bad else 0


def cred(dss: list[str], runs_root: Path) -> int:
    import ne_common as NE
    n = bad = 0
    for ds in dss:
        for r in NE.cells(ds):
            p = runs_root / ds / "cred" / f"{r['name']}__z6_rho2.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            score = json.loads(NE.cell_paths(ds, r["name"])[1].read_text())
            emit = score["decodes"][f"gnn|{NE.headline(ds)}"]
            why = []
            if d["pool"] != "train" or d["variants"]["z+tc"]["fit"] != "transfer":
                why.append(f"pool {d['pool']} fit {d['variants']['z+tc']['fit']}")
            if d["warnings"] or d["problems"]:
                why.append(f"warnings {d['warnings'][:2]} problems {d['problems'][:2]}")
            if abs(d["emitted_f1"] - emit["f1"]) > 1e-12 or d["emitted_n_sel"] != emit["n_selected"]:
                why.append(f"emitted f1 {d['emitted_f1']} / n_sel {d['emitted_n_sel']} against the "
                           f"scored {emit['f1']} / {emit['n_selected']}")
            n += 1
            if why:
                bad += 1
                print(f"[cred] {ds} {r['name']}: {'; '.join(why)}")
    print(f"[cred] {len(dss)} datasets, {n} creds, {bad} not clean -- {'ok' if n and not bad else 'FAILED'}")
    return 0 if n and not bad else 1


def sweep(dss: list[str]) -> int:
    import ne_common as NE
    n = bad = 0
    for ds in dss:
        k = int(NE.CA.DATASETS[ds].extra["k_decode"])
        for r in NE.cells(ds):
            p = NE.cell_paths(ds, r["name"])[1]
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            n += 1
            why = []
            if d["k"] != k or d.get("k_true") != 0:
                why.append(f"k {d['k']} (want {k}), k_true {d.get('k_true')}")
            for dk, dec in d["decodes"].items():
                if dec["k_pred"] < 0 or dec["k_pred"] > k:
                    why.append(f"{dk} k_pred {dec['k_pred']} outside [0, {k}]")
            if why:
                bad += 1
                print(f"[sweep] {ds} {r['name']}: {'; '.join(why)}")
    print(f"[sweep] {len(dss)} datasets, {n} scored cells, {bad} failing G-K -- "
          f"{'ok' if n and not bad else 'FAILED'}")
    return 0 if n and not bad else 1


def kernels(ds: str, probe_dir: Path, reps: str) -> int:
    """The tail gate, per cell, and which triple kernel each cell's ``features`` gets.

    Every cell must sit inside ``derived.TAIL`` -- the published tail ``features`` / ``credible`` have
    been measured to, dense kernel (``max_m`` 113).  The block kernel's chunk holds ``trees x m_b^3``
    floats, so a cell with a block past ``derived.BLOCK_MAX_M`` atoms (24) runs dense instead: dense
    and block counts are equal (gate G-TC), and dense is n^3 int32, 33 MB at n = 200 (measured on
    sp5k_k06 rep17, 55 atoms: 104 s, 0.39 GB).  Prints ``block <reps>`` and ``dense <reps>``
    (comma lists, possibly empty); exits 1 past the tail, or when the probe holds no cell."""
    import derived as DV
    import ne_common as NE
    p = probe_dir / f"{ds}_probe_blocks.csv"
    if not p.exists():
        print(f"[kernels] {ds}: no {p} -- run the probe stage", file=sys.stderr)
        return 1
    import csv
    want = NE.CA.parse_reps(reps)
    rows = [r for r in csv.DictReader(open(p, newline=""))
            if want is None or int(r["cell"].split("_rep")[1][:2]) in want]
    if not rows:
        print(f"[kernels] {ds}: no probed cell selected", file=sys.stderr)
        return 1
    over = {r["cell"]: {k: int(r[k]) for k, lim in DV.TAIL.items() if int(r[k]) > lim} for r in rows}
    over = {c: v for c, v in over.items() if v}
    if over:
        print(f"[kernels] {ds}: {len(over)} cells past the published tail {DV.TAIL}: "
              f"{dict(list(sorted(over.items()))[:5])}", file=sys.stderr)
        return 1
    rep = lambda r: int(r["cell"].split("_rep")[1][:2])  # noqa: E731
    dense = sorted(rep(r) for r in rows if int(r["max_m"]) > DV.BLOCK_MAX_M)
    block = sorted(rep(r) for r in rows if int(r["max_m"]) <= DV.BLOCK_MAX_M)
    print(f"block {','.join(map(str, block))}")
    print(f"dense {','.join(map(str, dense))}")
    print(f"[kernels] {ds}: {len(block)} block, {len(dense)} dense (max_m > {DV.BLOCK_MAX_M}); "
          f"max {({k: max(int(r[k]) for r in rows) for k in DV.TAIL})} inside {DV.TAIL}", file=sys.stderr)
    return 0


def status(runs_root: Path) -> int:
    """Cells scored / featured / credible per (size, k), and ASTRAL runs per size."""
    import ne_common as NE
    sizes = [t for t, _, _ in NE.CA.SPTREE_SIZES]
    print(f"{'size':>4s} {'stage':>7s} " + " ".join(f"k{k:<3d}" for k in NE.CA.SPTREE_KS))
    for size in sizes:
        cnt = {"scored": [], "feat": [], "cred": []}
        for k in NE.CA.SPTREE_KS:
            ds = NE.CA.sptree_name(size, k)
            names = [r["name"] for r in NE.cells(ds)]
            cnt["scored"].append(sum(NE.cell_paths(ds, n)[1].exists() for n in names))
            cnt["feat"].append(sum((runs_root / ds / "feat" / f"{n}__z6.json").exists() for n in names))
            cnt["cred"].append(sum((runs_root / ds / "cred" / f"{n}__z6_rho2.json").exists() for n in names))
        for st, v in cnt.items():
            print(f"{size:>4s} {st:>7s} " + " ".join(f"{x:<4d}" for x in v))
    astral = HERE / "runs" / "astral"
    for size in sizes:
        runs = list((astral / size).glob("*/run.json")) if (astral / size).exists() else []
        ok = sum(json.loads(p.read_text()).get("ok", False) for p in runs)
        print(f"ASTRAL-IV {size:>3s}: {ok} ok of {len(runs)} run (of 50)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("cmp", "cred", "sweep", "kernels", "status"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--runs-root", type=Path, default=HERE / "runs" / "ne")
    ap.add_argument("--probe-dir", type=Path, default=None, help="kernels: default <runs-root>/probe")
    ap.add_argument("--reps", default="", help="kernels: a rep filter")
    a = ap.parse_args()
    if a.what == "kernels":
        return kernels(a.args[0], a.probe_dir or a.runs_root / "probe", a.reps)
    if a.what == "cmp":
        if len(a.args) != 2:
            ap.error("cmp takes two directories")
        return cmp(Path(a.args[0]), Path(a.args[1]))
    if a.what == "cred":
        return cred(a.args, a.runs_root)
    if a.what == "status":
        return status(a.runs_root)
    return sweep(a.args)


if __name__ == "__main__":
    raise SystemExit(main())
