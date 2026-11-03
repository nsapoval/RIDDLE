#!/usr/bin/env python3
"""Stage the 500-tree subsamples: ``inputs/<ds>/d<k>/<phase-dir>/<rep>/`` in camus-data's layout.

For every cell of a benchmark's ``CELLS.tsv`` and every draw ``k``, the cell's staged gene-tree
file is read, ``common.draw_indices`` picks 500 of its non-blank lines, and those lines are written
**unchanged and in source order** under the arm's own file name.  The truth files beside it
(``true_net.nwk``, and netcs's ``true_net.phylonet.nwk``) are copied byte for byte.  Phase and arm
names do not change: ``run_gnn_cpu.py --inputs inputs/<ds>/d<k>`` resolves every cell exactly as it
does on the full data, and only the directory says that it is a subsample.

The draw is taken from the **staged** file, not the raw corpus.  On camus, l2reg and l3reg the two
are the same trees; on netcs the staged file carries ``stage.py``'s true-root ``OUT``, which is
applied line by line, so drawing staged lines is drawing corpus lines and rooting them.

Outputs, per dataset
--------------------
* ``inputs/<ds>/d<k>/...``: the tree files and the truth copies.
* ``inputs/<ds>/MANIFEST.tsv``: one row per (draw, cell): ``seed``, ``n_source``, ``m``, and the
  SHA-256 of the source file and of the subsample.  ``make_sub_csv.py`` re-hashes every subsample
  it scores against it.
* ``inputs/<ds>/d<k>/indices.json``: cell -> the 0-based source line indices drawn.
* ``inputs/<ds>/stage.json``: when, what, and every gate's numbers.

Gates (exit 1 on any failure; nothing half-written survives -- every file is replaced atomically)
-----
G1  the source file's SHA-256 is the one its benchmark recorded (``gnn_cpu_v1/SHA256SUMS`` for
    camus, ``stage.json`` for the rest), and its non-blank line count is ``CELLS.tsv``'s
    ``n_loci``.
G2  every written file, re-read, is exactly the drawn source lines in order: ``N_SUB`` of them.
G3  every truth copy is byte-identical to its source.
G4  vacuity: a cell's draws are pairwise distinct and none is the first-``N_SUB`` prefix -- the
    subsample the drivers' own ``max_loci`` cut would have taken.
G5  ``opc.rl.camus.prepare_gene_trees`` -- what both drivers run first -- reads one staged cell per
    (phase, draw) back with ``n_loci == N_SUB``.  It raises on any leaf set other than the truth's.
G6  ``NO_REFERENCE.tsv`` does not exist (``common.NO_REFERENCE``).
G7  ``gnn_cpu_v1/infer_gnn_cpu.py``'s own ``scan`` + ``select(m=500, seed=<seed>, outgroups=OUT)``
    on the source file picks exactly the staged indices, for the G5 cells.  The draw is that
    function's formula; this holds it to the function itself rather than to a copy of it.

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/stage.py                       # every dataset; idempotent
    python subsample_eval/stage.py --datasets camus,l2reg
    python subsample_eval/stage.py --check               # re-derive and compare; writes nothing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import common as C

from opc.rl import camus as CAMUS  # noqa: E402

TRUTH_FILES = ("true_net.nwk", "true_net.phylonet.nwk")


def recorded_shas(ds: C.Dataset) -> dict[tuple, dict[str, str]]:
    """``(phase, rep, arm)`` -> {``trees``, ``true_net.nwk``, ...} -> the SHA its benchmark recorded."""
    out: dict[tuple, dict[str, str]] = {}
    if ds.name == "camus":
        sums = {}
        for line in (C.GNN / "SHA256SUMS").read_text().splitlines():
            h, _, p = line.partition("  ")
            sums[p.strip()] = h
        for r in C.read_tsv(ds.cells):
            cell = CAMUS.CamusCell(r["phase"], int(r["rep"]), r["arm"], ds.source)
            rel = cell.cell_dir.relative_to(C.GNN)
            out[(r["phase"], int(r["rep"]), r["arm"])] = {
                "trees": sums.get(str(rel / cell.trees_path.name)),
                "true_net.nwk": sums.get(str(rel / "true_net.nwk"))}
        return out
    stage = json.loads(ds.stage.read_text())
    for c in stage["cells"]:
        phase = c.get("phase", stage.get("phase"))
        arm = c.get("arm", stage.get("arm"))
        out[(phase, int(c["rep"]), arm)] = dict(c["sha256"])
    return out


def atomic_write(dst: Path, data: bytes) -> bool:
    """Write ``data`` unless ``dst`` already holds it.  True if the file was (re)written."""
    if dst.exists() and dst.read_bytes() == data:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return True


def stage_dataset(ds: C.Dataset, check: bool) -> tuple[dict, list[str]]:
    t0 = time.perf_counter()
    cells = C.read_tsv(ds.cells)
    shas = recorded_shas(ds)
    problems: list[str] = []
    manifest: list[dict] = []
    indices: dict[int, dict[str, list[int]]] = {d: {} for d in range(ds.draws)}
    overlap_draws, overlap_prefix = [], []
    n_written = n_bytes = 0
    truth_done: set[tuple] = set()

    for i, r in enumerate(cells, 1):
        phase, rep, arm = r["phase"], int(r["rep"]), r["arm"]
        name = C.cell_name(phase, rep, arm)
        src = CAMUS.CamusCell(phase, rep, arm, ds.source)
        raw = src.trees_path.read_bytes()
        sha_src = C.sha256_bytes(raw)
        want = shas.get((phase, rep, arm), {})
        # G1
        if want.get("trees") != sha_src:
            problems.append(f"G1 {ds.name} {name}: source SHA {sha_src[:12]} is not the recorded "
                            f"{str(want.get('trees'))[:12]}")
            continue
        lines = [ln for ln in raw.split(b"\n") if ln.strip()]
        if len(lines) != int(r["n_loci"]):
            problems.append(f"G1 {ds.name} {name}: {len(lines)} non-blank lines, CELLS.tsv says "
                            f"{r['n_loci']}")
            continue

        drawn = []
        for d in range(ds.draws):
            seed = C.draw_seed(ds.name, phase, rep, arm, d)
            idx = C.draw_indices(seed, len(lines))
            drawn.append(idx)
            data = b"".join(lines[j] + b"\n" for j in idx)
            dst_dir = ds.draw_inputs(d) / src.cell_dir.relative_to(ds.source)
            dst = dst_dir / src.trees_path.name
            if check:
                if not dst.exists() or dst.read_bytes() != data:
                    problems.append(f"--check {ds.name} d{d} {name}: {dst} is missing or differs")
            elif atomic_write(dst, data):
                n_written += 1
            n_bytes += len(data)
            # G2, on what is actually on disk
            if dst.exists():
                back = [ln for ln in dst.read_bytes().split(b"\n") if ln.strip()]
                if back != [lines[j] for j in idx] or len(back) != C.N_SUB:
                    problems.append(f"G2 {ds.name} d{d} {name}: {dst} is not the drawn lines")
            # G3: the truth files, once per (draw, cell directory)
            if (d, src.cell_dir) not in truth_done:
                truth_done.add((d, src.cell_dir))
                for fname in TRUTH_FILES:
                    s = src.cell_dir / fname
                    if not s.exists():
                        continue
                    sb = s.read_bytes()
                    if fname in want and want[fname] != C.sha256_bytes(sb):
                        problems.append(f"G1 {ds.name} {name}: {fname} is not the recorded file")
                    t = dst_dir / fname
                    if not check:
                        atomic_write(t, sb)
                    if not t.exists() or t.read_bytes() != sb:
                        problems.append(f"G3 {ds.name} d{d} {name}: {t} is not {s}")
            indices[d][name] = idx
            manifest.append({"dataset": ds.name, "draw": d, "phase": phase, "rep": rep, "arm": arm,
                             "cell": name, "seed": seed, "n_source": len(lines), "m": C.N_SUB,
                             "sha256_source": sha_src, "sha256_sub": C.sha256_bytes(data),
                             "path": str(dst.relative_to(C.HERE))})
        # G4
        prefix = list(range(C.N_SUB))
        sets = [set(x) for x in drawn]
        for a in range(len(drawn)):
            if drawn[a] == prefix:
                problems.append(f"G4 {ds.name} d{a} {name}: the draw is the first-{C.N_SUB} prefix")
            overlap_prefix.append(len(sets[a] & set(prefix)))
            for b in range(a + 1, len(drawn)):
                if drawn[a] == drawn[b]:
                    problems.append(f"G4 {ds.name} {name}: draws {a} and {b} are identical")
                overlap_draws.append(len(sets[a] & sets[b]))
        if i % 100 == 0 or i == len(cells):
            print(f"[stage] {ds.name}: {i}/{len(cells)} cells | {time.perf_counter() - t0:.0f} s",
                  flush=True)

    if not check:
        C.write_tsv(ds.manifest, manifest, list(manifest[0]) if manifest else ["dataset"])
        for d, table in indices.items():
            p = ds.draw_inputs(d) / "indices.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(table, separators=(",", ":")) + "\n")
    elif ds.manifest.exists():
        on_disk = C.read_tsv(ds.manifest)
        want_rows = [{k: str(v) for k, v in row.items()} for row in manifest]
        if on_disk != want_rows:
            problems.append(f"--check {ds.name}: {ds.manifest} differs from a fresh derivation")

    g5 = gate_g5(ds, cells, problems)
    stats = {
        "cells": len(cells), "draws": ds.draws, "m": C.N_SUB, "files_written": n_written,
        "bytes_staged": n_bytes, "seconds": round(time.perf_counter() - t0, 1),
        "n_source": dict(sorted(_counts(int(r["n_loci"]) for r in cells).items())),
        "g4_overlap_between_draws_mean": statistics.fmean(overlap_draws) if overlap_draws else None,
        "g4_overlap_with_prefix_mean": statistics.fmean(overlap_prefix) if overlap_prefix else None,
        "g5_cells_read_back": g5,
    }
    return stats, problems


def _counts(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def gate_g5(ds: C.Dataset, cells: list[dict], problems: list[str]) -> int:
    """G5 and G7: one staged cell per (phase, draw) through ``prepare_gene_trees``, as both drivers
    run it, and through ``infer_gnn_cpu.select``."""
    if str(C.GNN) not in sys.path:
        sys.path.append(str(C.GNN))
    import infer_gnn_cpu as INF
    first: dict[str, dict] = {}
    for r in cells:
        first.setdefault(r["phase"], r)
    n = 0
    with tempfile.TemporaryDirectory(prefix="sub_g5_") as work:
        for phase, r in sorted(first.items()):
            for d in range(ds.draws):
                cell = CAMUS.CamusCell(phase, int(r["rep"]), r["arm"], ds.draw_inputs(d))
                if not cell.trees_path.exists():
                    problems.append(f"G5 {ds.name} d{d} {phase}: no {cell.trees_path}")
                    continue
                taxa = list(CAMUS.truth_of(cell).labels)
                try:
                    info = CAMUS.prepare_gene_trees(cell, Path(work) / f"{phase}_d{d}.nwk.gz", taxa,
                                                    max_loci=1000)
                except ValueError as e:
                    problems.append(f"G5 {ds.name} d{d} {phase}: {e}")
                    continue
                if int(info["n_loci"]) != C.N_SUB:
                    problems.append(f"G5 {ds.name} d{d} {phase}: prepare_gene_trees read "
                                    f"{info['n_loci']} loci, not {C.N_SUB}")
                src = CAMUS.CamusCell(phase, int(r["rep"]), r["arm"], ds.source).trees_path
                masks, labels, freq = INF.scan(src)
                seed = C.draw_seed(ds.name, phase, int(r["rep"]), r["arm"], d)
                chosen = INF.select(masks, labels, freq, outgroups=(C.OUT,), m=C.N_SUB, seed=seed)[1]
                if chosen != C.draw_indices(seed, len(masks)):
                    problems.append(f"G7 {ds.name} d{d} {phase}: infer_gnn_cpu.select draws "
                                    f"differently")
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(C.DATASETS),
                    help=f"comma list (default all: {','.join(C.DATASETS)})")
    ap.add_argument("--check", action="store_true", help="re-derive and compare; write nothing")
    args = ap.parse_args()
    names = [x for x in args.datasets.split(",") if x]
    unknown = sorted(set(names) - set(C.DATASETS))
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; known: {sorted(C.DATASETS)}")

    problems: list[str] = []
    if C.NO_REFERENCE.exists():                                             # G6
        problems.append(f"G6 {C.NO_REFERENCE} exists: the camus GNN runs would check parity "
                        f"against it; it must not exist")
    for name in names:
        ds = C.DATASETS[name]
        stats, probs = stage_dataset(ds, args.check)
        problems += probs
        print(f"[stage] {name}: {stats['cells']} cells x {stats['draws']} draws | "
              f"{stats['files_written']} files written, {stats['bytes_staged'] / 1e9:.2f} GB staged | "
              f"n_source {stats['n_source'] if name == 'netcs' else list(stats['n_source'])} | "
              f"mean overlap: between draws {stats['g4_overlap_between_draws_mean']}, with the "
              f"prefix {stats['g4_overlap_with_prefix_mean']:.1f} | G5/G7 read back "
              f"{stats['g5_cells_read_back']} cells | {len(probs)} problems", flush=True)
        if not args.check:
            rec = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": name,
                   "source": str(ds.source), "cells_tsv": str(ds.cells), "salt": C.SALT,
                   "draw": "sorted(random.Random(seed).sample(list(range(n_source)), m)); "
                           "seed = opc.grid.chain_seed(salt, dataset, phase, rep, arm, draw)",
                   **stats, "problems": probs}
            (ds.inputs / "stage.json").write_text(json.dumps(rec, indent=1) + "\n")
    for p in problems[:50]:
        print(f"  {p}")
    if len(problems) > 50:
        print(f"  ... and {len(problems) - 50} more")
    print(f"[stage] {'PASS' if not problems else 'FAIL'}: G1-G7 over {', '.join(names)}"
          f"{' (--check)' if args.check else ''}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
