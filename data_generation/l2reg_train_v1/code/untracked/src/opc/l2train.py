"""``l2reg_train_v1``: regular, exactly-level-2 *training* data at n = 25 and n = 50 (A65).

    python -m opc.l2train probe --jobs 48        # BLOCKING gate, before anything is built
    python -m opc.l2train networks --jobs 48     # 1,800 networks: N, R, and their ground truth
    python -m opc.l2train genetrees --jobs 48    # PhyloNetworks audit (BLOCKING), then the MSNC
    python -m opc.l2train pilot                  # per phase: s/locus, kB/locus, RF
    python -m opc.l2train estimate --jobs 48     # AliSim + IQ-TREE, alignments kept
    python -m opc.l2train lengths --jobs 48      # the rooted estimates again, lengths kept
    python -m opc.l2train tob --jobs 48          # TREE-QMC --blob per network
    python -m opc.l2train package                # MANIFEST.tsv, manifest.json, README, SHA256SUMS
    python -m opc.l2train all --jobs 48

``l2reg_n200_v1`` (:mod:`opc.l2reg`) is a 20-network *benchmark* at n = 200; this is the training
set that goes with it.  net-opt's mixed true/inferred regime trains on ``alisim_gt_v1/v2``'s x/y
phases, which are **level-1 only**, so nothing the policy has seen carries a blob with two
reticulations.  The design was fixed with the user on 2026-09-11:

* two phases, ``l2t25`` (n = 25) and ``l2t50`` (n = 50), 900 networks each, split by replicate
  index exactly as the x/y grids are -- train < 700, val < 800, test < 900;
* layouts chosen so ``k`` sits near the level-1 training ladders (x25 has k <= 3, x50 k <= 5) and
  starts at the smallest level-2 network there is, ``(1 level-2 blob, 0 level-1 blobs)``, k = 2.
  Each ``(layout, cu)`` cell is a whole number of times in every split, because both rotation
  cycles (10 reps at n = 25, 20 at n = 50) divide 100;
* every network regular, exactly level-2 and of girth >= 5 on ``R`` -- :mod:`opc.l2reg`'s gates,
  which are the definition and not a sufficient condition;
* ILS, gammas, loci, sequence model and estimation exactly as ``l2reg_n200_v1`` and v1/v2, with
  the **alignments kept** and a tree of blobs per network.

Estimation, the tree of blobs, the audit and the helpers are :mod:`opc.l2reg`'s, called
unchanged: the only thing that differs is ``n``, which its stages read off the spec.

**The estimator is pinned to IQ-TREE 2.4.0**, the build ``alisim_gt_v1/v2``'s ``est_bp1000`` arm
was made with, because this data is mixed with that arm at training time.  The repo default is
2.3.6 (what ``l2reg_n200_v1`` used), so the binary is chosen here and the version is a gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opc import l2reg, l3reg
from opc.grid import chain_seed
from opc.l2reg import BP, CUS, GAMMA_RANGE, MODEL, N_LOCI, REPO, Paths, _write
from opc.simulate import level2 as L2
from opc.simulate import networks as N
from opc.simulate import outgroup as og
from opc.simulate import seqsim

PKG = REPO / "l2reg_train_v1"
PROBE_OUT = REPO / "results" / "validation" / "l2train_probe.json"
#: The seed namespace.  A master of its own, as net-opt's x/y grids take one per draw: it is the
#: leakage fix rather than hygiene, since a shared master plus a shared tag would make this set's
#: replicates reproduce another dataset's cells.
MASTER_SEED = 20260911
TAG = "l2train"
REPS = 900
#: ``rep < SPLITS[0]`` is train, ``< SPLITS[1]`` val, the rest test -- net-opt's ``split_by_rep``.
SPLITS = (700, 800)
PHASES = {"l2t25": 25, "l2t50": 50}
#: ``(level-2 blobs, level-1 blobs)`` per phase, rotated one per replicate pair.  ``k = 2a + b``.
LAYOUTS = {
    25: ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
    50: ((1, 0), (1, 1), (1, 2), (2, 0), (1, 3), (2, 1), (2, 2), (3, 0), (2, 3), (3, 1)),
}
#: Base trees before a build gives up.  l2reg uses 20 at n = 200, where nothing needed a second;
#: at n = 25 a targeted shape took 11 (measured while planning).  Raising it cannot change a
#: network that already builds -- each draw is seeded ``(seed, draw)`` -- only how many are
#: reachable, so it is a cap and not a knob.
MAX_BASE = 100
#: IQ-TREE 2.4.0, ``alisim_gt_v1/v2``'s binary.  Checked, never assumed: `probe` and `estimate`
#: refuse any other version, because the arm this data is mixed with was made with this one.
IQTREE_BINARY = Path("/shared/ns58/conda_envs/mpl/bin/iqtree2")
IQTREE_VERSION = "IQ-TREE multicore version 2.4.0 for Linux x86 64-bit built Feb 12 2025"
#: Measured on this box, 1,000 estimated gene trees per run (n = 200 takes ~3.7 h, hence l2reg's
#: own warning; at these n the stage is affordable).
TOB_NOTE = "24 s each at n = 25 and 243 s at n = 50, measured on this box"


# ------------------------------------------------------------------------------ the design


@dataclass(frozen=True)
class Spec:
    """One network.  ``n_taxa`` is a *field* here and a property on :class:`opc.l2reg.Spec`:
    that is the whole difference, and it is what lets l2reg's stages run this design."""

    phase: str
    n_taxa: int
    rep: int
    n_l2: int
    n_l1: int
    cu: float
    shapes: tuple[str, ...]

    @property
    def key(self) -> str:
        # rep:03d, so 900 keys sort as text.  l2reg's 20 use :02d.
        return f"n{self.n_taxa}_rep{self.rep:03d}_l2x{self.n_l2}_l1x{self.n_l1}_cu{self.cu}"

    @property
    def k(self) -> int:
        return 2 * self.n_l2 + self.n_l1

    @property
    def layout(self) -> L2.Layout:
        return L2.Layout(self.n_l2, self.n_l1, self.shapes)

    @property
    def split(self) -> str:
        train, val = SPLITS
        return "train" if self.rep < train else "val" if self.rep < val else "test"

    @property
    def net_seed(self) -> int:
        # n is in the chain: l2t25 rep0 and l2t50 rep0 must not share a base-tree stream.
        return chain_seed(MASTER_SEED, TAG, "net", self.n_taxa, self.rep)

    @property
    def gt_seed(self) -> int:
        return chain_seed(MASTER_SEED, TAG, "gt", self.key) % (2 ** 31)

    @property
    def est_seeds(self) -> tuple[int, int]:
        """AliSim / IQ-TREE bases, derived exactly as ``seqsim.plan_jobs`` derives them."""
        return (chain_seed(MASTER_SEED, "alisim", self.key, BP) % (2 ** 31),
                chain_seed(MASTER_SEED, "iqtree", self.key, BP) % (2 ** 31))

    def row(self) -> dict:
        return dataclasses.asdict(self) | {
            "key": self.key, "k": self.k, "split": self.split,
            "net_seed": self.net_seed, "gt_seed": self.gt_seed}


def design(phase: str) -> list[Spec]:
    """``l2reg.design``'s loop over this phase's layouts.

    Shapes are rotated *within* each ILS level, which balances them and has a second effect that
    matters at n = 25: a two-blob network draws two *consecutive* shapes and so never asks for the
    same one twice, which is the one case measured to fail there.
    """
    n = PHASES[phase]
    layouts = LAYOUTS[n]
    used = dict.fromkeys(CUS, 0)
    specs = []
    for rep in range(REPS):
        cu = CUS[rep % len(CUS)]
        a, b = layouts[(rep // len(CUS)) % len(layouts)]
        shapes = tuple(L2.SHAPES[(used[cu] + i) % len(L2.SHAPES)] for i in range(a))
        used[cu] += a
        specs.append(Spec(phase, n, rep, a, b, cu, shapes))
    return specs


def all_specs(phases=None) -> list[Spec]:
    return [s for ph in (phases or PHASES) for s in design(ph)]


def paths_of(root: Path, phase: str) -> Paths:
    """Each phase is a self-contained ``l2reg`` layout under the package root."""
    return Paths(root / phase)


# ------------------------------------------------------------------------------ the estimator


def use_pinned_iqtree() -> None:
    """Bind :mod:`seqsim` to the 2.4.0 build, unless ``OPC_IQTREE`` says otherwise.

    ``seqsim.IQTREE`` is read from the environment at *import*, so a per-command environment
    variable is easy to forget and fails silently into 2.3.6.  An explicit ``OPC_IQTREE`` still
    wins; :func:`iqtree_gate` is what actually decides.  Called from ``main`` only, so importing
    this module has no side effect, and workers inherit it across the fork.
    """
    if "OPC_IQTREE" not in os.environ and IQTREE_BINARY.exists():
        seqsim.IQTREE = IQTREE_BINARY


def iqtree_gate() -> dict:
    got = seqsim.iqtree_version() if seqsim.available() else "absent"
    return {"ok": got == IQTREE_VERSION, "want": IQTREE_VERSION, "got": got,
            "binary": str(seqsim.IQTREE)}


# ------------------------------------------------------------------------------ probe


def _probe_one(task):
    n_taxa, a, b, shapes, seed = task
    layout = L2.Layout(a, b, shapes)
    t0 = time.perf_counter()
    try:
        net, placement = L2.build_level2(n_taxa, layout, seed, cu=0.5, gamma_range=GAMMA_RANGE,
                                         cfg=L2._Cfg(max_base=MAX_BASE))  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - a failed build is the measurement
        return {"n_taxa": n_taxa, "cell": f"{a}x{b}:{','.join(shapes) or 'free'}", "seed": seed,
                "ok": False, "error": str(exc)[:300], "seconds": time.perf_counter() - t0}
    report = L2.blob_report(net)
    return {
        "n_taxa": n_taxa, "cell": f"{a}x{b}:{','.join(shapes) or 'free'}", "seed": seed,
        "ok": L2.check(net, layout)["ok"], "seconds": time.perf_counter() - t0,
        "base_draw": placement["base_draw"],
        "shapes": sorted(r["shape"] for r in report if r["n_retics"] == 2),
        "girth_R": [r["girth_R"] for r in report],
        "tree_child": L2.is_tree_child(net),
    }


def probe_tasks(seeds: int) -> list[tuple]:
    """The design's own ``(layout, shapes)`` tuples, plus free placement at each layout.

    Sharper than probing uniform shape modes: what has to build is what ``design`` asks for, and
    at n = 25 a repeated shape (which the design never asks for) is exactly the hard case.
    """
    tasks = []
    for phase, n in PHASES.items():
        cells = sorted({(s.n_l2, s.n_l1, s.shapes) for s in design(phase)})
        cells += sorted({(a, b, ()) for a, b in LAYOUTS[n]})
        for a, b, shapes in cells:
            tasks += [(n, a, b, shapes, 9000 + s) for s in range(seeds)]
    return tasks


def stage_probe(seeds: int, jobs: int, l2pkg: Path) -> int:
    """BLOCKING.  Five gates: the tools are v1/v2's, the estimator is 2.4.0, ``l2reg_n200_v1``
    still re-derives byte for byte, no seed stream is shared, and every design cell places."""
    out: dict = {"phases": PHASES, "seeds_per_cell": seeds, "max_base": MAX_BASE, "gates": {}}
    gates, ok = l3reg._tool_gates()  # noqa: SLF001 - l2reg's first two gates, verbatim
    out["gates"].update(gates)

    out["gates"]["iqtree_is_v1v2s"] = g = iqtree_gate()
    ok &= g["ok"]

    t0 = time.perf_counter()
    try:
        ident = l3reg.l2_identity(l2pkg)
    except FileNotFoundError as exc:  # the gate cannot skip: a missing package is a failure
        ident = {"ok": False, "error": str(exc)}
    ident["seconds"] = time.perf_counter() - t0
    out["gates"]["level2_byte_identity"] = ident
    ok &= ident["ok"]
    print(f"[probe] level-2 byte identity: {'PASS' if ident['ok'] else 'FAIL'} "
          f"({ident.get('differ', ident.get('error'))}) in {ident['seconds']:.0f}s", flush=True)

    mine = all_specs()
    theirs = l2reg.design() + l3reg.design()
    shared = {
        "net_seed": sorted({s.net_seed for s in mine} & {s.net_seed for s in theirs}),
        "gt_seed": sorted({s.gt_seed for s in mine} & {s.gt_seed for s in theirs}),
        "est_seeds": sorted(map(list, {s.est_seeds for s in mine} & {s.est_seeds for s in theirs})),
        "key": sorted({s.key for s in mine} & {s.key for s in theirs}),
    }
    out["gates"]["seed_streams_disjoint"] = {"ok": not any(shared.values()), **shared}
    ok &= not any(shared.values())

    tasks = probe_tasks(seeds)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(max(1, min(jobs, len(tasks)))) as pool:
        rows = pool.map(_probe_one, tasks, chunksize=1)
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault(f"n{r['n_taxa']} {r['cell']}", []).append(r)
    summary = {
        cell: {"built": sum(r["ok"] for r in rs), "of": len(rs),
               "seconds_median": float(np.median([r["seconds"] for r in rs])),
               "seconds_max": float(max(r["seconds"] for r in rs)),
               "base_redraws_max": max(r.get("base_draw", -1) for r in rs)}
        for cell, rs in sorted(by_cell.items())
    }
    out["summary"] = summary
    built = [r for r in rows if r["ok"]]
    free = [r for r in built if r["cell"].endswith("free")]
    out["natural_shape_distribution"] = dict(sorted(Counter(
        s for r in free for s in r["shapes"]).items()))
    out["girth_R_distribution"] = dict(sorted(Counter(g for r in built for g in r["girth_R"]).items()))
    out["tree_child_fraction"] = float(np.mean([r["tree_child"] for r in built])) if built else 0.0
    failed = [c for c, s in summary.items() if s["built"] < s["of"]]
    out["gates"]["every_design_cell_places"] = {"ok": not failed, "failed_cells": failed,
                                                "cells": len(summary), "builds": len(rows)}
    ok &= not failed
    out["seconds"] = time.perf_counter() - t0
    out["ok"] = bool(ok)
    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_OUT.write_text(json.dumps(out, indent=1) + "\n")
    slow = max(summary.items(), key=lambda kv: kv[1]["seconds_max"])
    print(f"[probe] {len(rows)} builds over {len(summary)} cells in {out['seconds']:.0f}s; "
          f"failed: {failed or 'none'}; slowest {slow[0]} at {slow[1]['seconds_max']:.0f}s "
          f"({slow[1]['base_redraws_max']} redraws)")
    print(f"[probe] natural shapes (free): {out['natural_shape_distribution']}; "
          f"girth_R {out['girth_R_distribution']}; tree-child {out['tree_child_fraction']:.2f}")
    for name, g in out["gates"].items():
        print(f"[probe] {name}: {'PASS' if g['ok'] else ('n/a' if g['ok'] is None else 'FAIL')}")
    print(f"[probe] {'PASS' if ok else 'FAIL'} -> {PROBE_OUT}")
    return 0 if ok else 2


# ------------------------------------------------------------------------------ networks


def _network_one(task):
    """Build one network and write its four files.  Returns a status row; never raises.

    Writing happens in the worker, so 900 good networks are on disk whatever the 901st does.  A
    failed build is reported by key and **never reseeded**: a spec's network is a function of its
    seed, and quietly drawing another one would put an un-reproducible network in the dataset.
    """
    spec, root = task
    paths = paths_of(root, spec.phase)
    t0 = time.perf_counter()
    try:
        net, placement = L2.build_level2(
            spec.n_taxa, spec.layout, spec.net_seed, cu=spec.cu, gamma_range=GAMMA_RANGE,
            cfg=L2._Cfg(max_base=MAX_BASE))  # noqa: SLF001
        rec = l2reg.network_record(spec, net, placement)
        if not rec["check"]["ok"]:
            raise RuntimeError(f"failed its own gates: {rec['check']}")
        if abs(rec["median_internal_branch"] - spec.cu) > 1e-9 or rec["ultrametric_spread"] > 1e-9:
            raise RuntimeError(f"not scaled to cu={spec.cu} or not ultrametric")
        _write(paths.net(spec.key), N.to_enewick(net) + "\n")
        _write(paths.regular(spec.key), N.to_enewick(L2.contract(net)) + "\n")
        _write(paths.pickle(spec.key), pickle.dumps(net))
        _write(paths.meta(spec.key), json.dumps(rec, indent=1) + "\n")   # last: the cache marker
        return spec.key, "ok", time.perf_counter() - t0, placement["base_draw"]
    except Exception:  # noqa: BLE001 - reported through the status row, as the other stages do
        return spec.key, "FAILED\n" + traceback.format_exc(), time.perf_counter() - t0, -1


def stage_networks(root: Path, specs: list[Spec], jobs: int, force: bool = False) -> int:
    todo = []
    for s in specs:
        paths = paths_of(root, s.phase)
        if force or not (paths.meta(s.key).exists() and paths.net(s.key).exists()):
            todo.append(s)
    print(f"[networks] {len(todo)} of {len(specs)} to build")
    if not todo:
        return 0
    from multiprocessing import Pool

    fails, draws, t0 = [], Counter(), time.perf_counter()
    with Pool(max(1, min(jobs, len(todo)))) as pool:
        for i, (key, status, secs, draw) in enumerate(
                pool.imap_unordered(_network_one, [(s, root) for s in todo], chunksize=1), 1):
            if status != "ok":
                fails.append(f"--- {key}\n{status}")
            else:
                draws[draw] += 1
            if i % 50 == 0 or i == len(todo):
                print(f"[networks] {i}/{len(todo)} ok={sum(draws.values())} "
                      f"failed={len(fails)} {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print(f"[networks] base redraws: {dict(sorted(draws.items()))}")
    if fails:
        print("\n".join(["[networks] FAILED:"] + fails[:5]))
        print(f"[networks] {len(fails)} of {len(todo)} failed")
        return 1
    return 0


# ------------------------------------------------------------------------------ lengths


_LEN = re.compile(r":[-+0-9.eE]+")


def _strip_lengths(newick: str) -> str:
    """``(t1:0.1,t2:0.2)`` -> ``(t1,t2)``.  Labels are ``t<k>``, so no label holds a colon."""
    return _LEN.sub("", newick)


def _lengths_one(task):
    """``<key>.raw.nwk.gz`` -> ``<key>.len.nwk.gz``: rooted on OUT, OUT pruned, lengths kept.

    net-opt's W13 reads branch lengths off the estimated arm and a length-free tree does not fail
    anywhere -- it degenerates into constant features (``reroot_on_outgroup``'s docstring).  The
    landed ``<key>.nwk.gz`` is topology-only, so the lengths come from the raw file here.

    Deterministic, so it is a separate stage rather than part of estimation: it can be re-run for
    a whole phase without touching an AliSim seed.  Every line is checked two ways -- the taxon
    set, and that **stripping the lengths reproduces the landed topology byte for byte**, which
    is a free cross-check of this rooting against the one that produced the package.
    """
    key, raw_gz, plain_gz, out_gz, taxa = task
    t0 = time.perf_counter()
    try:
        if out_gz.exists() and out_gz.stat().st_mtime > raw_gz.stat().st_mtime:
            return key, "cached", 0.0, 0
        want = set(taxa)
        tmp = out_gz.with_name(out_gz.name + ".tmp")
        out_gz.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with gzip.open(raw_gz, "rt") as raw, gzip.open(plain_gz, "rt") as plain, \
                gzip.open(tmp, "wt") as dst:
            for line, landed in zip(raw, plain, strict=True):
                line, landed = line.strip(), landed.strip()
                if not line:
                    continue
                rooted = og.reroot_on_outgroup(line, keep_lengths=True)
                got = set(l2reg._TAXON.findall(rooted))  # noqa: SLF001
                if got != want:
                    raise RuntimeError(f"tree {n + 1}: {len(got)} taxa, expected {len(want)}")
                if _strip_lengths(rooted) != landed:
                    raise RuntimeError(f"tree {n + 1}: differs from the landed topology")
                dst.write(rooted + "\n")
                n += 1
        tmp.rename(out_gz)
        return key, "ok", time.perf_counter() - t0, n
    except Exception:  # noqa: BLE001
        return key, "FAILED\n" + traceback.format_exc(), time.perf_counter() - t0, 0


def stage_lengths(root: Path, specs: list[Spec], jobs: int) -> int:
    tasks = []
    for s in specs:
        paths = paths_of(root, s.phase)
        raw = paths.est / f"{s.key}.raw.nwk.gz"
        if not raw.exists():
            print(f"[lengths] no estimates yet for {s.key}; run estimate")
            return 1
        tasks.append((s.key, raw, paths.est / f"{s.key}.nwk.gz",
                      paths.est / f"{s.key}.len.nwk.gz", [f"t{i}" for i in range(1, s.n_taxa + 1)]))
    from multiprocessing import Pool

    fails, done, t0 = [], Counter(), time.perf_counter()
    with Pool(max(1, min(jobs, len(tasks)))) as pool:
        for i, (key, status, secs, n) in enumerate(
                pool.imap_unordered(_lengths_one, tasks, chunksize=1), 1):
            done[status.split("\n")[0]] += 1
            if status.startswith("FAILED"):
                fails.append(f"--- {key}\n{status}")
            elif status == "ok" and n != N_LOCI:
                fails.append(f"--- {key}\n{n} trees, expected {N_LOCI}")
            if i % 100 == 0 or i == len(tasks):
                print(f"[lengths] {i}/{len(tasks)} {dict(done)} "
                      f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if fails:
        print("\n".join(["[lengths] FAILED:"] + fails[:5]))
        return 1
    return 0


# ------------------------------------------------------------------------------ package


def _alignments_digest(task):
    """One SHA-256 over a network's alignments, in locus order, and how many there were.

    ``SHA256SUMS`` lists every other file individually, as ``l2reg_n200_v1`` does.  It cannot list
    these: there are 1.8 M of them here against 20,000 there, which would make the checksum file
    ~180 MB and unusable.  One digest per network reads the same bytes and covers the same data.
    """
    key, d = task
    h = hashlib.sha256()
    files = sorted(d.glob("loc*.phy.gz"))
    nbytes = 0
    for p in files:
        nbytes += p.stat().st_size
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return key, h.hexdigest(), len(files), nbytes


def training_file_list(specs: list[Spec]) -> list[str]:
    """Exactly what a training box needs, as paths relative to the package root.

    The per-locus alignments are ~90% of the package and nothing in training reads them; they are
    kept so the estimates can be re-derived or re-estimated with another tool, which is a thing
    one does once, on the machine that has the disk.  Two more omissions are deliberate:

    * ``<key>.raw.nwk.gz`` -- IQ-TREE's output before rooting, ~250 MB a phase.  ``.len.nwk.gz``
      is that tree rooted and pruned with its lengths intact, which is what W13 reads;
    * ``<key>.pkl`` -- a pickle of *this* repo's ``Network`` class.  eNewick plus the ground-truth
      JSON is the handoff that does not depend on the reader having the same class.

    Kept deliberately: ``<key>.nwk.gz``, the topology-only estimates, because they are 21 MB a
    phase and they are what the landed v1/v2 arms look like, so an ingest written against those
    reads this set unchanged.
    """
    rels = ["MANIFEST.tsv", "manifest.json", "README.md", "TRAINING_FILES.txt"]
    for s in specs:
        p = s.phase
        rels += [
            f"{p}/networks/{s.key}.enewick",
            f"{p}/networks/{s.key}.regular.enewick",
            f"{p}/networks/{s.key}.json",
            f"{p}/trees/{s.key}.nwk.gz",
            f"{p}/estimated_bp{BP}/{s.key}.len.nwk.gz",
            f"{p}/estimated_bp{BP}/{s.key}.nwk.gz",
            f"{p}/estimated_bp{BP}/{s.key}.json",
        ]
    return rels


def stage_package(root: Path, specs: list[Spec], jobs: int) -> int:
    rows, recs, sides, tobs = [], {}, {}, {}
    for s in specs:
        paths = paths_of(root, s.phase)
        need = [paths.net(s.key), paths.regular(s.key), paths.meta(s.key), paths.trees(s.key),
                paths.est / f"{s.key}.nwk.gz", paths.est / f"{s.key}.raw.nwk.gz",
                paths.est / f"{s.key}.len.nwk.gz", paths.est / f"{s.key}.json",
                paths.tob / f"{s.key}.raw.tob", paths.tob / f"{s.key}.tob",
                paths.tob / f"{s.key}.pvalues.tob", paths.tob / f"{s.key}.json"]
        absent = [p.name for p in need if not p.exists()]
        if absent:
            print(f"[package] {s.key}: incomplete ({absent})")
            return 1
        recs[s.key] = json.loads(paths.meta(s.key).read_text())
        sides[s.key] = json.loads((paths.est / f"{s.key}.json").read_text())
        tobs[s.key] = json.loads((paths.tob / f"{s.key}.json").read_text())

    from multiprocessing import Pool

    print(f"[package] digesting {len(specs) * N_LOCI:,} alignments", flush=True)
    t0 = time.perf_counter()
    with Pool(max(1, min(jobs, len(specs)))) as pool:
        digests = dict()
        counts = dict()
        aln_bytes = 0
        for key, digest, n, nbytes in pool.imap_unordered(
                _alignments_digest,
                [(s.key, paths_of(root, s.phase).aln / s.key) for s in specs], chunksize=1):
            digests[key], counts[key] = digest, n
            aln_bytes += nbytes
    print(f"[package] digests in {(time.perf_counter() - t0) / 60:.1f} min")
    short = [k for k, n in counts.items() if n != N_LOCI]
    if short:
        print(f"[package] {len(short)} networks do not have {N_LOCI} alignments: {short[:5]}")
        return 1

    for s in specs:
        rec, side, tob = recs[s.key], sides[s.key], tobs[s.key]
        p = s.phase
        rows.append({
            "phase": p, "key": s.key, "rep": s.rep, "split": s.split, "n_taxa": s.n_taxa,
            "n_l2": s.n_l2, "n_l1": s.n_l1, "k": s.k, "cu": s.cu,
            "shapes": ",".join(rec["check"]["shapes"]),
            "girth_R_min": rec["check"]["girth_R_min"],
            "tree_child": rec["check"]["regularity"]["tree_child"],
            "base_draw": rec["placement"]["base_draw"],
            "n_loci": N_LOCI, "bp": BP,
            "network": f"{p}/networks/{s.key}.enewick",
            "regular_network": f"{p}/networks/{s.key}.regular.enewick",
            "truth": f"{p}/networks/{s.key}.json",
            "genetrees": f"{p}/trees/{s.key}.nwk.gz",
            "alignments": f"{p}/alignments_bp{BP}/{s.key}/",
            "estimated": f"{p}/estimated_bp{BP}/{s.key}.nwk.gz",
            "estimated_raw": f"{p}/estimated_bp{BP}/{s.key}.raw.nwk.gz",
            "estimated_len": f"{p}/estimated_bp{BP}/{s.key}.len.nwk.gz",
            "tob": f"{p}/tob_bp{BP}/{s.key}.tob",
            "aln_width_mean": round(side["aln_len_mean"], 1),
            "gap_fraction": round(side["gap_fraction"], 4),
            "rf_norm_mean": round(side["rf_norm_mean"], 4),
            "tob_true_edges": tob["n_true_edges"], "tob_fn": tob["fn"], "tob_fp": tob["fp"],
            "network_sha256": l2reg._sha256(paths_of(root, p).net(s.key)),  # noqa: SLF001
            "genetrees_sha256": l2reg._sha256(paths_of(root, p).trees(s.key)),  # noqa: SLF001
            "alignments_sha256": digests[s.key],
        })
    header = list(rows[0])
    _write(root / "MANIFEST.tsv", "\t".join(header) + "\n" + "".join(
        "\t".join(str(r[h]) for h in header) + "\n" for r in rows))

    audit = {}
    for ph in sorted({s.phase for s in specs}):
        audit[ph] = json.loads((paths_of(root, ph).manifests / "audit_verdict.json").read_text())
    code = l2reg._snapshot_code(root / "code")  # noqa: SLF001
    manifest = {
        "name": root.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code,
        "master_seed": MASTER_SEED, "seed_tag": TAG, "level": 2,
        "phases": {ph: PHASES[ph] for ph in sorted({s.phase for s in specs})},
        "n_networks": len(specs), "reps_per_phase": REPS,
        "splits": {"train": SPLITS[0], "val": SPLITS[1]},
        "layouts": {str(n): [list(c) for c in cs] for n, cs in LAYOUTS.items()},
        "max_base": MAX_BASE,
        "n_loci": N_LOCI, "bp": BP, "gamma_range": list(GAMMA_RANGE), "cu_values": list(CUS),
        "min_girth": L2.MIN_GIRTH,
        "model": MODEL.as_dict(),
        "tools": {"iqtree": seqsim.iqtree_version(), "iqtree_binary": str(seqsim.IQTREE),
                  **l2reg._julia_versions(), "python": sys.version.split()[0],  # noqa: SLF001
                  "numpy": np.__version__, "tree_qmc": l2reg._tob_tool()},  # noqa: SLF001
        "seed_check": [{"cell_key": s.key, "bp": BP, "seed_alisim": s.est_seeds[0],
                        "seed_iqtree": s.est_seeds[1], "gt_seed": s.gt_seed,
                        "net_seed": s.net_seed}
                       for ph in sorted({s.phase for s in specs})
                       for s in design(ph)[:2]],
        "gates": {
            "python_check_ok": all(r["check"]["ok"] for r in recs.values()),
            "phylonetworks_audit_ok": not any(v for ph in audit.values() for v in ph.values()),
            "iqtree_model_used": sorted({v["iqtree_model_used"] for v in sides.values()}),
            "iqtree_version_pinned": iqtree_gate()["ok"],
            "alignments_per_network": sorted({n for n in counts.values()}),
            "tob_exact": sum(t["fn"] == t["fp"] == 0 for t in tobs.values()),
        },
    }
    rels = training_file_list(specs)
    _write(root / "TRAINING_FILES.txt", "".join(r + "\n" for r in rels))
    _write(root / "manifest.json", json.dumps(manifest, indent=1) + "\n")

    # Every file except the per-locus alignments, which MANIFEST.tsv covers one digest per
    # network (see _alignments_digest).  The two checksum files themselves are excluded, and
    # TRAINING_FILES.txt is written above so that it is inside this walk.
    skip = {"SHA256SUMS", "SHA256SUMS.training"}
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.name not in skip
                   and not p.name.endswith((".tmp", ".part"))
                   and f"alignments_bp{BP}" not in p.parts)
    other_bytes = sum(p.stat().st_size for p in files)
    digest_of = {str(p.relative_to(root)): l2reg._sha256(p) for p in files}  # noqa: SLF001
    train_bytes = sum((root / r).stat().st_size for r in rels if (root / r).exists())
    _write(root / "README.md", readme(specs, rows, recs, sides, tobs, manifest,
                                      {"training": train_bytes, "alignments": aln_bytes,
                                       "other": other_bytes}))
    # README.md is in `files`, and it has just been rewritten, so hash it after the fact.
    digest_of["README.md"] = l2reg._sha256(root / "README.md")  # noqa: SLF001
    _write(root / "SHA256SUMS", "".join(
        f"{digest_of[r]}  {r}\n" for r in sorted(digest_of)))
    missing = [r for r in rels if r not in digest_of]
    if missing:
        print(f"[package] training list names files that do not exist: {missing[:5]}")
        return 1
    _write(root / "SHA256SUMS.training", "".join(f"{digest_of[r]}  {r}\n" for r in sorted(rels)))
    print(f"[package] {len(rows)} networks, {len(files)} files checksummed "
          f"(+{len(specs)} alignment digests) -> {root}")
    print(f"[package] training half: {len(rels):,} files, {train_bytes / 1e9:.1f} GB of "
          f"{(train_bytes + aln_bytes + other_bytes - train_bytes) / 1e9 + train_bytes / 1e9:.1f} "
          f"GB -- TRAINING_FILES.txt, SHA256SUMS.training")
    return 0


def readme(specs, rows, recs, sides, tobs, manifest, sizes) -> str:
    by_phase = {}
    for ph in sorted({s.phase for s in specs}):
        mine = [s for s in specs if s.phase == ph]
        srows = [r for r in rows if r["phase"] == ph]
        blobs = [b for s in mine for b in recs[s.key]["blobs"]]
        by_phase[ph] = {
            "n": PHASES[ph], "specs": mine, "rows": srows,
            "k": (min(s.k for s in mine), max(s.k for s in mine)),
            "splits": Counter(s.split for s in mine),
            "shapes": Counter(s for s in mine for s in recs[s.key]["check"]["shapes"]),
            "girth": Counter(b["girth_R"] for b in blobs),
            "tree_child": sum(recs[s.key]["check"]["regularity"]["tree_child"] for s in mine),
            "width": np.mean([sides[s.key]["aln_len_mean"] for s in mine]),
            "gaps": np.mean([sides[s.key]["gap_fraction"] for s in mine]),
            "rf": np.mean([sides[s.key]["rf_norm_mean"] for s in mine]),
            "redraws": Counter(recs[s.key]["placement"]["base_draw"] for s in mine),
            "tob_fn": sum(tobs[s.key]["fn"] for s in mine),
            "tob_fp": sum(tobs[s.key]["fp"] for s in mine),
            "tob_true": sum(tobs[s.key]["n_true_edges"] for s in mine),
            "tob_exact": sum(tobs[s.key]["fn"] == tobs[s.key]["fp"] == 0 for s in mine),
        }
    design_rows = "\n".join(
        f"| `{ph}` | {d['n']} | {len(d['specs'])} | "
        f"{' '.join(f'{a}x{b}' for a, b in LAYOUTS[d['n']])} | {d['k'][0]}-{d['k'][1]} | "
        f"{d['splits']['train']} / {d['splits']['val']} / {d['splits']['test']} |"
        for ph, d in by_phase.items())
    cells = Counter((r["phase"], r["n_l2"], r["n_l1"], r["cu"], r["split"]) for r in rows)
    cell_rows = "\n".join(
        f"| `{ph}` | {a}x{b} | {2 * a + b} | {cu} | {cells[(ph, a, b, cu, 'train')]} / "
        f"{cells[(ph, a, b, cu, 'val')]} / {cells[(ph, a, b, cu, 'test')]} |"
        for ph, d in by_phase.items() for a, b in LAYOUTS[d["n"]] for cu in CUS)
    realised = "\n".join(
        f"| `{ph}` | {dict(sorted(d['shapes'].items()))} | {dict(sorted(d['girth'].items()))} | "
        f"{d['tree_child']}/{len(d['specs'])} | {d['width']:.0f} | {d['gaps']:.0%} | "
        f"{d['rf']:.3f} | {d['tob_exact']}/{len(d['specs'])} | {d['tob_fn']}/{d['tob_true']} | "
        f"{d['tob_fp']} |"
        for ph, d in by_phase.items())
    code = manifest["code"]
    files_table = "\n".join(["| path | what it is |", "|---|---|"] + [
        f"| `{p}` | {d} |" for p, d in _files()])
    name = manifest["name"]
    train_gb = sizes["training"] / 1e9
    full_gb = (sizes["alignments"] + sizes["other"]) / 1e9
    aln_pct = 100 * sizes["alignments"] / max(1, sizes["alignments"] + sizes["other"])
    n_train = len(training_file_list(specs))
    return f"""# {manifest['name']} -- level-2 training data at n = {' and n = '.join(
        str(PHASES[ph]) for ph in by_phase)}

Built {manifest['built_utc']} by `python -m opc.l2train` from rl-net-opt at `{code['commit'][:12]}`
{"(with uncommitted changes -- the exact code is in `code/`)" if code['dirty'] else ""}.

{len(specs):,} phylogenetic networks, each **regular**, **exactly level-2** and of **girth >=
{L2.MIN_GIRTH}** on its regular form. For each network, {N_LOCI:,} gene trees under the network
multispecies coalescent, a {BP} bp AliSim alignment and an IQ-TREE re-estimate per locus, and a
TREE-QMC tree of blobs.

This is the training counterpart of `l2reg_n200_v1`, which is 20 networks at n = 200 and is a
**test** set. The regime it is for is net-opt's mixed true/inferred one, whose training phases
(`x25 x50 y25 y50`, in `alisim_gt_v1/v2`) are level-1 only: no blob any of them carries has two
reticulations.

## Design

Replicates are split by index, as the x/y grids are, so a split is a prefix of the replicates and
each `(layout, cu)` cell lands in every split a whole number of times.

| phase | n | networks | layouts (L2 x L1) | k | train / val / test |
|---|---|---|---|---|---|
{design_rows}

Each replicate pair takes the next layout and each replicate the next ILS level
(`cu = {list(CUS)}`, the median internal branch in coalescent units), so the rotation cycle is
`2 x layouts` and every cell is balanced:

| phase | layout | k | cu | train / val / test |
|---|---|---|---|---|
{cell_rows}

The four level-2 shapes (S0-S3, `l2reg_n200_v1`'s README defines them) rotate within each ILS
level, which balances them and means a two-blob network draws two *consecutive* shapes -- it never
asks for the same shape twice, which is the case that fails to place at n = 25.

One gamma per reticulation, ~U({GAMMA_RANGE[0]}, {GAMMA_RANGE[1]}) on the donor (minor) edge.

## Files

{files_table}

## Realised

| phase | shapes | girth on R | tree-child | aln width | gaps | RF | tob exact | tob missed | tob extra |
|---|---|---|---|---|---|---|---|---|---|
{realised}

Base-tree redraws per phase: {', '.join(f"`{ph}` {dict(sorted(d['redraws'].items()))}"
                                        for ph, d in by_phase.items())} -- the cap is
{MAX_BASE}.

## Sequence model

`{MODEL.sim_model_string()}` + indels `{MODEL.indel_ins:g},{MODEL.indel_del:g}`, size
`{MODEL.indel_size}`, branch scale `{MODEL.branch_scale}` substitutions/site per coalescent unit
(theta = {MODEL.theta:g}), inferred under `{MODEL.inference_model}` with `--fast --keep-ident` --
identical to `alisim_gt_v1/v2` and `l2reg_n200_v1`. `{BP}` is the **root** sequence length:
indels make realised alignments wider. The outgroup is grafted onto each gene tree at
`(1 + {MODEL.outgroup_alpha}) x p99` of that network's root heights, so it contributes no rooting
error.

**IQ-TREE is `{manifest['tools']['iqtree']}`** -- v1/v2's build, not the 2.3.6 `l2reg_n200_v1`
used, because this data is mixed with v1/v2's `est_bp1000` arm at training time. The estimate
stage refuses to run on any other version.

## Reading it

`<key>.nwk.gz` is topology only, as in v1/v2. **`<key>.len.nwk.gz` is the one to train on**: the
same trees, same order, with branch lengths in substitutions per site. net-opt's W13 reads lengths
off the estimated arm, and a length-free tree does not fail anywhere -- it degenerates into
constant features. Stripping the lengths from `.len.nwk.gz` reproduces `.nwk.gz` line for line,
and that is checked on every line of every file when it is written.

Each estimated file is paired locus for locus with `trees/<key>.nwk.gz`, the true trees.

## Copying just the training half

The per-locus alignments are **{aln_pct:.0f}%** of the package and nothing in training reads
them. They are kept so the estimates can be re-derived, or re-estimated with another tool -- which
is done once, on the machine that has the disk. Each phase keeps them in one directory,
`<phase>/alignments_bp{BP}/`, so one exclude leaves every one of them behind:

```bash
rsync -a --exclude 'alignments_bp{BP}/' {name}/ box:/data/{name}/
```

`TRAINING_FILES.txt` is the same thing said exactly rather than by pattern -- {n_train:,} paths:
the truth, the true gene trees, and the estimates with lengths. It also drops `<key>.raw.nwk.gz`
(IQ-TREE's output before rooting, which `.len.nwk.gz` supersedes) and `<key>.pkl` (a pickle of
this repo's `Network` class, which a reader would need the same class for):

```bash
rsync -a --files-from=TRAINING_FILES.txt {name}/ box:/data/{name}/
cd /data/{name} && sha256sum -c SHA256SUMS.training --quiet
```

That is **{train_gb:.1f} GB** against {full_gb:.1f} GB for the whole package.

## Verifying

```bash
sha256sum -c SHA256SUMS --quiet          # everything except the per-locus alignments
sha256sum -c SHA256SUMS.training --quiet # just what TRAINING_FILES.txt names
```

The alignments are covered one digest per network, in `MANIFEST.tsv`'s `alignments_sha256`: the
SHA-256 of that network's `loc*.phy.gz` concatenated in locus order. There are
{len(specs) * N_LOCI:,} of them, so listing each one would make `SHA256SUMS` ~180 MB.

## Reproducibility

Seeds are SHA-256 chains off master seed `{MASTER_SEED}`: networks
`(master, "{TAG}", "net", n, rep)`, gene trees `(master, "{TAG}", "gt", key)`, AliSim / IQ-TREE
`(master, "alisim" | "iqtree", key, bp)` plus the global locus / batch index. The master and tag
are this dataset's own, so no stream is shared with `l2reg_n200_v1`, `l3reg_n200_v1` or the x/y
grids -- checked as a probe gate, not assumed.

`python -m opc.l2train all` rebuilds everything; each stage skips what already exists.
"""


def _files() -> list[tuple[str, str]]:
    """The README's file table.  ``l2reg``'s, with the phase prefix and the length-keeping file."""
    rows = []
    for path, desc in l2reg._files():  # noqa: SLF001
        if path in ("MANIFEST.tsv", "manifest.json", "code/", "SHA256SUMS"):
            rows.append((path, desc))
        elif path.startswith("manifests/"):
            rows.append(("<phase>/" + path, desc))
        else:
            rows.append(("<phase>/" + path, desc))
        if path == f"estimated_bp{BP}/<key>.raw.nwk.gz":
            rows.append((f"<phase>/estimated_bp{BP}/<key>.len.nwk.gz",
                         "the same trees as `<key>.nwk.gz` -- rooted on `OUT`, `OUT` pruned, same "
                         "order -- **with branch lengths** (substitutions/site). What to train "
                         "on: net-opt's W13 reads lengths, and a length-free tree degenerates "
                         "silently"))
    rows.insert(0, ("MANIFEST.tsv", "one row per network -- the index to iterate over, with its "
                                    "`phase`, `split` and `alignments_sha256`"))
    rows.insert(1, ("TRAINING_FILES.txt",
                    "the paths a training box needs, one per line: truth, true gene trees and "
                    "the estimates with lengths -- no sequence data, no `.raw.nwk.gz`, no "
                    "`.pkl`. Feed it to `rsync --files-from`"))
    rows.insert(2, ("SHA256SUMS.training",
                    "checksums for exactly those paths, so a partial copy verifies without "
                    "reporting every file it was never sent"))
    seen = set()
    out = []
    for p, d in rows:
        if p not in seen:
            seen.add(p)
            out.append((p, d))
    return out


# ------------------------------------------------------------------------------ CLI


def _per_phase(fn, root: Path, phases, *args) -> int:
    """Run one stage for each phase in turn, stopping at the first failure."""
    for ph in phases:
        specs = design(ph)
        print(f"=== {ph}: {len(specs)} networks, n = {PHASES[ph]}", flush=True)
        code = fn(paths_of(root, ph), specs, *args)
        if code:
            return code
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opc.l2train", description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("probe", "networks", "genetrees", "pilot", "estimate",
                                      "lengths", "tob", "package", "all"))
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--l2pkg", type=Path, default=l2reg.PKG,
                    help="probe: the level-2 benchmark that must still re-derive byte for byte")
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--jobs", type=int, default=min(48, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()) / "l2train")
    ap.add_argument("--chunks", type=int, default=1, help="estimation tasks per network")
    ap.add_argument("--seeds", type=int, default=25, help="probe: seeds per design cell")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="networks: rebuild even if present")
    args = ap.parse_args(argv)
    if args.jobs >= 100:
        print("refusing --jobs >= 100: a pool that large deadlocks silently under the shared "
              "RLIMIT_NPROC (CLAUDE.md)")
        return 2
    phases = [p for p in args.phases.split(",") if p]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        print(f"unknown phases {unknown}; known: {list(PHASES)}")
        return 2
    use_pinned_iqtree()
    root, specs = args.pkg, all_specs(phases)

    def estimate(paths, sp):
        g = iqtree_gate()
        if not g["ok"]:
            print(f"[estimate] IQ-TREE is {g['got']!r} at {g['binary']}, not the pinned "
                  f"{g['want']!r}. Set OPC_IQTREE to v1/v2's build, or change IQTREE_VERSION.")
            return 2
        return l2reg.stage_estimate(paths, sp, args.jobs, args.scratch, args.chunks, args.dry_run)

    stages = {
        "probe": lambda: stage_probe(args.seeds, args.jobs, args.l2pkg),
        "networks": lambda: stage_networks(root, specs, args.jobs, args.force),
        "genetrees": lambda: _per_phase(
            lambda paths, sp: l2reg.stage_genetrees(paths, sp, args.jobs), root, phases),
        "pilot": lambda: _per_phase(
            lambda paths, sp: l2reg.stage_pilot(paths, sp, args.scratch), root, phases),
        "estimate": lambda: _per_phase(estimate, root, phases),
        "lengths": lambda: stage_lengths(root, specs, args.jobs),
        "tob": lambda: _per_phase(
            lambda paths, sp: l2reg.stage_tob(paths, sp, args.jobs, args.scratch, note=TOB_NOTE),
            root, phases),
        "package": lambda: stage_package(root, specs, args.jobs),
    }
    order = ["probe", "networks", "genetrees", "estimate", "lengths", "tob", "package"] \
        if args.stage == "all" else [args.stage]
    for st in order:
        print(f"##### {st} {datetime.now().isoformat(timespec='seconds')}", flush=True)
        code = stages[st]()
        if code:
            print(f"##### {st} FAILED ({code})")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
