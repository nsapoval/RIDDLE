"""``l2reg_n200_v1``: a benchmark of regular, exactly-level-2 networks at n = 200 (A62).

    python -m opc.l2reg probe                    # BLOCKING gate, before anything is built
    python -m opc.l2reg networks                 # 20 networks: N, R, and their ground truth
    python -m opc.l2reg genetrees                # PhyloNetworks audit (BLOCKING), then the MSNC
    python -m opc.l2reg pilot                    # 38 loci end to end, to measure the budget
    python -m opc.l2reg estimate --jobs 48 --dry-run
    python -m opc.l2reg estimate --jobs 48       # AliSim + IQ-TREE, alignments kept
    python -m opc.l2reg tob                      # TREE-QMC --blob on the estimated gene trees
    python -m opc.l2reg package                  # MANIFEST.tsv, manifest.json, README, SHA256SUMS
    python -m opc.l2reg all --jobs 48

Every stage is idempotent: a network, a gene-tree file or an estimation chunk that exists is
skipped, so an interruption costs only what is missing.

The design was fixed with the user before any code was written:

* 20 networks, 200 taxa, 1,000 gene trees each, 1,000 bp (root length) per locus;
* every network regular, exactly level-2, girth >= 5 on its regular form (:mod:`level2`);
* a varied mix of blobs -- each ``(level-2 blobs, level-1 blobs)`` in ``{1,2,3}^2`` once per ILS
  level and ``(2, 2)`` twice, so ``k = 2a + b`` runs 3..9;
* ILS as in ``alisim_gt_v1/v2``: median internal branch 0.25 or 0.5 coalescent units, 10 each;
* the four level-2 shapes rotated, so each appears 5 times per ILS level;
* one gamma per reticulation, ~U(0.25, 0.5), on the donor (minor) edge, as in v1/v2;
* v1/v2's sequence model verbatim, with the alignments **kept** (v1/v2 deleted them).
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
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opc.grid import chain_seed
from opc.simulate import julia_backend, seqsim
from opc.simulate import level2 as L2
from opc.simulate import networks as N
from opc.simulate import outgroup as og

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "l2reg_n200_v1"
PROBE_OUT = REPO / "results" / "validation" / "l2reg_probe.json"
AUDIT_SCRIPT = REPO / "src" / "opc" / "simulate" / "audit_networks.jl"

N_TAXA, N_NETWORKS, N_LOCI, BP = 200, 20, 1000, 1000
MASTER_SEED = 20260910
CUS = (0.25, 0.5)
COMBOS = ((2, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3))
GAMMA_RANGE = (0.25, 0.5)
#: ``alisim_gt_v1/v2``'s ``model.json``, field for field -- the probe checks it against the file.
MODEL = seqsim.SeqModel()
#: TREE-QMC 4.1.5, built against the system R with its own R library beside it (A63).  ``--blob``
#: exactly as ``tob_qmc_v1`` ran it -- default weighting, since ``--hybrid`` needs branch support
#: and IQ-TREE's trees here carry none, and default seeds.
TREEQMC = Path(os.environ.get("OPC_TREEQMC", "/shared/ns58/software/TREE-QMC/build/tree-qmc"))
TREEQMC_COMMIT = "e577f8ce468d9f7567721ee723074ff686baf703"
TOB_ARGS = ("--blob",)


# ------------------------------------------------------------------------------ the design


@dataclass(frozen=True)
class Spec:
    rep: int
    n_l2: int
    n_l1: int
    cu: float
    shapes: tuple[str, ...]

    @property
    def n_taxa(self) -> int:
        """The taxon count every stage runs this spec at.

        A **property, not a field**: ``dataclasses.asdict`` ignores it, so ``row()`` and the
        ground-truth JSON stay byte-identical and ``l3reg.l2_identity`` keeps passing.  It exists
        so the stages read the spec rather than the module constant, which is what lets a design
        at another ``n`` (:mod:`opc.l2train`, n = 25 and 50) reuse them unchanged.
        """
        return N_TAXA

    @property
    def key(self) -> str:
        return f"n{N_TAXA}_rep{self.rep:02d}_l2x{self.n_l2}_l1x{self.n_l1}_cu{self.cu}"

    @property
    def k(self) -> int:
        return 2 * self.n_l2 + self.n_l1

    @property
    def layout(self) -> L2.Layout:
        return L2.Layout(self.n_l2, self.n_l1, self.shapes)

    @property
    def net_seed(self) -> int:
        return chain_seed(MASTER_SEED, "l2reg", "net", self.rep)

    @property
    def gt_seed(self) -> int:
        return chain_seed(MASTER_SEED, "l2reg", "gt", self.key) % (2 ** 31)

    @property
    def est_seeds(self) -> tuple[int, int]:
        """AliSim / IQ-TREE bases, derived exactly as ``seqsim.plan_jobs`` derives them."""
        return (chain_seed(MASTER_SEED, "alisim", self.key, BP) % (2 ** 31),
                chain_seed(MASTER_SEED, "iqtree", self.key, BP) % (2 ** 31))

    def row(self) -> dict:
        return dataclasses.asdict(self) | {
            "key": self.key, "k": self.k, "net_seed": self.net_seed, "gt_seed": self.gt_seed}


def design() -> list[Spec]:
    """The 20 networks.  Shapes are rotated *within* each ILS level, so both levels see each 5x."""
    used = dict.fromkeys(CUS, 0)
    specs = []
    for rep in range(N_NETWORKS):
        cu = CUS[rep % len(CUS)]
        a, b = COMBOS[(rep // len(CUS)) % len(COMBOS)]
        shapes = tuple(L2.SHAPES[(used[cu] + i) % len(L2.SHAPES)] for i in range(a))
        used[cu] += a
        specs.append(Spec(rep, a, b, cu, shapes))
    return specs


@dataclass(frozen=True)
class Paths:
    root: Path

    def net(self, key: str) -> Path:
        return self.root / "networks" / f"{key}.enewick"

    def regular(self, key: str) -> Path:
        return self.root / "networks" / f"{key}.regular.enewick"

    def meta(self, key: str) -> Path:
        return self.root / "networks" / f"{key}.json"

    def pickle(self, key: str) -> Path:
        return self.root / "networks" / f"{key}.pkl"

    def trees(self, key: str) -> Path:
        return self.root / "trees" / f"{key}.nwk.gz"

    @property
    def est(self) -> Path:
        return self.root / f"estimated_bp{BP}"

    @property
    def aln(self) -> Path:
        return self.root / f"alignments_bp{BP}"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def tob(self) -> Path:
        return self.root / f"tob_bp{BP}"


def _write(path: Path, data: str | bytes) -> None:
    """Sibling ``.tmp`` then rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data) if isinstance(data, bytes) else tmp.write_text(data)
    tmp.rename(path)


def _labels(net: N.Network, taxa) -> list[str]:
    return sorted((net.labels[t] for t in taxa), key=lambda s: int(s[1:]))


def _load(paths: Paths, key: str) -> N.Network:
    with open(paths.pickle(key), "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------------------ probe


def _probe_one(task):
    (a, b), mode, seed = task
    shapes = () if mode == "free" else (mode,) * a
    layout = L2.Layout(a, b, shapes)
    t0 = time.perf_counter()
    try:
        net, placement = L2.build_level2(N_TAXA, layout, seed, cu=0.5, gamma_range=GAMMA_RANGE)
    except Exception as exc:  # noqa: BLE001 - a failed build is the measurement
        return {"combo": [a, b], "mode": mode, "seed": seed, "ok": False, "error": str(exc)[:500],
                "seconds": time.perf_counter() - t0}
    report = L2.blob_report(net)
    return {
        "combo": [a, b], "mode": mode, "seed": seed, "ok": L2.check(net, layout)["ok"],
        "seconds": time.perf_counter() - t0, "base_draw": placement["base_draw"],
        "counters": placement["counters"],
        "shapes": sorted(r["shape"] for r in report if r["n_retics"] == 2),
        "girth_R": [r["girth_R"] for r in report],
        "root_taxa": [r["root_taxa"] for r in report],
        "tree_child": L2.is_tree_child(net),
    }


def stage_probe(seeds: int, jobs: int) -> int:
    """BLOCKING.  The model is v1/v2's, AliSim runs, and every layout places at n = 200."""
    out: dict = {"n_taxa": N_TAXA, "seeds_per_cell": seeds, "gates": {}}
    ok = True

    v1 = REPO / "alisim_gt_v1" / "model.json"
    if v1.exists():
        want = json.loads(v1.read_text())
        got = MODEL.as_dict()
        diff = {k: (want[k], got.get(k)) for k in want if str(want[k]) != str(got.get(k))}
        out["gates"]["model_matches_alisim_gt_v1"] = {"ok": not diff, "diff": diff}
        ok &= not diff
    else:
        out["gates"]["model_matches_alisim_gt_v1"] = {"ok": None, "note": "file absent"}

    with tempfile.TemporaryDirectory(prefix="l2reg-probe-") as td:
        tree = Path(td) / "t.nwk"
        tree.write_text("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);\n")
        try:
            seqsim._run(seqsim.alisim_cmd(tree, Path(td) / "a", 100, MODEL, 1), Path(td))  # noqa: SLF001
            rows = (Path(td) / "a.phy").read_text().split("\n")[0].split()
            alisim_ok = rows[0] == "4"
        except Exception as exc:  # noqa: BLE001
            alisim_ok, rows = False, [str(exc)[:300]]
    out["gates"]["alisim_runs"] = {"ok": alisim_ok, "iqtree": seqsim.iqtree_version(),
                                   "binary": str(seqsim.IQTREE), "header": rows}
    ok &= alisim_ok

    tasks = [(c, m, 7000 + s) for c in COMBOS for m in (*L2.SHAPES, "free") for s in range(seeds)]
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(max(1, min(jobs, len(tasks)))) as pool:
        rows = pool.map(_probe_one, tasks, chunksize=1)
    out["placement"] = rows
    by_cell: dict = {}
    for r in rows:
        by_cell.setdefault(f"{r['combo'][0]}x{r['combo'][1]}:{r['mode']}", []).append(r)
    summary = {
        cell: {"built": sum(r["ok"] for r in rs), "of": len(rs),
               "seconds_median": float(np.median([r["seconds"] for r in rs])),
               "base_redraws_max": max(r.get("base_draw", -1) for r in rs)}
        for cell, rs in sorted(by_cell.items())
    }
    out["summary"] = summary
    free = [s for r in rows if r["mode"] == "free" and r["ok"] for s in r["shapes"]]
    out["natural_shape_distribution"] = dict(sorted(Counter(free).items()))
    girths = [g for r in rows if r["ok"] for g in r["girth_R"]]
    out["girth_R_distribution"] = dict(sorted(Counter(girths).items()))
    out["tree_child_fraction"] = float(np.mean([r["tree_child"] for r in rows if r["ok"]]))
    failed = [c for c, s in summary.items() if s["built"] < s["of"]]
    out["gates"]["every_layout_places"] = {"ok": not failed, "failed_cells": failed}
    ok &= not failed
    out["seconds"] = time.perf_counter() - t0
    out["ok"] = bool(ok)
    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[probe] {len(rows)} builds in {out['seconds']:.0f}s; failed cells: {failed or 'none'}")
    print(f"[probe] natural shapes (free): {out['natural_shape_distribution']}")
    print(f"[probe] girth_R: {out['girth_R_distribution']}  tree-child: "
          f"{out['tree_child_fraction']:.2f}")
    for name, g in out["gates"].items():
        print(f"[probe] {name}: {'PASS' if g['ok'] else ('n/a' if g['ok'] is None else 'FAIL')}")
    print(f"[probe] {'PASS' if ok else 'FAIL'} -> {PROBE_OUT}")
    return 0 if ok else 2


# ------------------------------------------------------------------------------ networks


def network_record(spec: Spec, net: N.Network, placement: dict) -> dict:
    """The ground truth written beside each network."""
    below = N.hardwired_clusters(net)
    tags = {ret.hybrid: f"H{k + 1}" for k, ret in enumerate(net.retics)}
    blocks = L2.nontrivial_blocks(net)
    blob_of = {h: i for i, b in enumerate(blocks) for h in b.hybrids}
    leaf_times = [net.time[v] for v in net.leaf_taxon]
    ib = N.internal_branch_lengths(net)
    return {
        "key": spec.key, "spec": spec.row(), "n_taxa": spec.n_taxa, "labels": net.labels,
        "gamma_range": list(GAMMA_RANGE),
        "check": L2.check(net, spec.layout),
        "blobs": L2.blob_report(net),
        "reticulations": [{
            "tag": tags[r.hybrid], "blob": blob_of[r.hybrid], "gamma": net.gamma_of(r.hybrid),
            # net.time, not r.time: Reticulation.time is the insertion time, before scaling.
            "time": net.time[r.hybrid],
            "hybrid_cluster": _labels(net, below[r.hybrid]),
            "donor_cluster": _labels(net, below[r.donor]),
            "tree_parent_cluster": _labels(net, below[net.tree_parent[r.hybrid]]),
        } for r in net.retics],
        "root_height": max(net.time),
        "ultrametric_spread": max(leaf_times) - min(leaf_times),
        "median_internal_branch": float(np.median(ib)),
        "n_clusters": len(N.full_clusters(net)),
        "regular_network": L2.regular_json(net),
        "placement": placement,
    }


def stage_networks(paths: Paths, specs: list[Spec], force: bool = False) -> int:
    for s in specs:
        if paths.meta(s.key).exists() and paths.net(s.key).exists() and not force:
            print(f"[networks] {s.key}: cached")
            continue
        net, placement = L2.build_level2(s.n_taxa, s.layout, s.net_seed, cu=s.cu,
                                         gamma_range=GAMMA_RANGE)
        rec = network_record(s, net, placement)
        if not rec["check"]["ok"]:
            raise RuntimeError(f"{s.key}: failed its own gates: {rec['check']}")
        if abs(rec["median_internal_branch"] - s.cu) > 1e-9 or rec["ultrametric_spread"] > 1e-9:
            raise RuntimeError(f"{s.key}: not scaled to cu={s.cu} or not ultrametric")
        _write(paths.net(s.key), N.to_enewick(net) + "\n")
        _write(paths.regular(s.key), N.to_enewick(L2.contract(net)) + "\n")
        _write(paths.pickle(s.key), pickle.dumps(net))
        _write(paths.meta(s.key), json.dumps(rec, indent=1) + "\n")   # last: the cache marker
        print(f"[networks] {s.key}: k={net.n_retics} shapes={rec['check']['shapes']} "
              f"girth_R>={rec['check']['girth_R_min']} {placement['seconds']:.1f}s")
    return 0


# ------------------------------------------------------------------------------ gene trees


def _julia(script: Path, *args) -> subprocess.CompletedProcess:
    env = dict(os.environ, JULIA_DEPOT_PATH=str(julia_backend.DEPOT), JULIA_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.run(
        [str(julia_backend.JULIA), f"--project={julia_backend.PROJECT}", str(script),
         *map(str, args)], capture_output=True, text=True, env=env, cwd=str(REPO))


def audit_problems(net: N.Network, row: dict, level: int = 2) -> list[str]:
    """Everything PhyloNetworks read off the file, against what the generator meant.

    ``level`` is the level the network was built at; :mod:`opc.l3reg` passes 3.
    """
    if "error" in row:
        return [f"audit error: {row['error']}"]
    lab = net.labels
    bad = []
    if sorted(row["taxa"]) != sorted(lab):
        bad.append("taxon set")
    if row["numhybrids"] != net.n_retics:
        bad.append(f"numhybrids {row['numhybrids']} != {net.n_retics}")
    py = {frozenset(lab[t] for t in c) for c in N.full_clusters(net)}
    jl = {frozenset(c) for c in row["clusters"]}
    if py != jl:
        bad.append(f"hardwired clusters differ ({len(py ^ jl)} in the symmetric difference)")
    if row["level"] != level:
        bad.append(f"getlevel {row['level']}")
    blobs = L2.nontrivial_blocks(net)
    if row["blob_retics"] != sorted(b.n_retics for b in blobs):
        bad.append(f"per-blob reticulations {row['blob_retics']}")
    if row["ladder"]:
        bad.append("hybrid ladder")
    if row["treechild"] != L2.is_tree_child(net):
        bad.append("tree-child disagrees")
    g_n = min(L2.girth(b) for b in blobs)
    g_r = min(L2.girth(b) for b in L2.nontrivial_blocks(L2.contract(net)))
    if (row["girth_N"], row["girth_R"]) != (g_n, g_r):
        bad.append(f"girth N/R {row['girth_N']}/{row['girth_R']} != {g_n}/{g_r}")
    below = N.hardwired_clusters(net)
    for k, r in enumerate(net.retics):
        h, tag = r.hybrid, f"H{k + 1}"
        got = row["hybrids"].get(tag)
        if got is None:
            bad.append(f"{tag} missing")
            continue
        if frozenset(got["cluster"]) != frozenset(lab[t] for t in below[h]):
            bad.append(f"{tag} cluster")
        # Each parent edge is matched by its parent's cluster -- which is what says the #H copy
        # sits on the donor edge -- and only then is gamma compared, to the file's precision.
        # (Rounding both sides to 8 decimals was tried first and is not a comparison: 0.25719151
        # 50178 and the file's 0.257191515 round to different sides of a boundary.)
        g = net.gamma_of(h)
        want = {frozenset(lab[t] for t in below[r.donor]): (g, g > 0.5),
                frozenset(lab[t] for t in below[net.tree_parent[h]]): (1 - g, g < 0.5)}
        seen = {frozenset(p["cluster"]): (p["gamma"], p["ismajor"]) for p in got["parents"]}
        if len(want) != 2:
            bad.append(f"{tag}: donor and tree parent share a cluster")
        elif set(want) != set(seen) or any(abs(want[c][0] - seen[c][0]) > 1e-9
                                           or want[c][1] != seen[c][1] for c in want):
            bad.append(f"{tag} parent edges (gamma / placement of the #H copy)")
    return bad


def stage_audit(paths: Paths, specs: list[Spec], level: int = 2) -> int:
    """BLOCKING: PhyloNetworks' reading of every eNewick, against Python's intent."""
    inp, out = paths.manifests / "audit_paths.json", paths.manifests / "audit.json"
    _write(inp, json.dumps([str(paths.net(s.key)) for s in specs], indent=1))
    res = _julia(AUDIT_SCRIPT, inp, out)
    if res.returncode != 0:
        raise RuntimeError(f"audit_networks.jl failed:\n{res.stdout}\n{res.stderr}")
    rows = {Path(r["path"]).name[: -len(".enewick")]: r for r in json.loads(out.read_text())}
    verdict = {s.key: audit_problems(_load(paths, s.key), rows[s.key], level) for s in specs}
    _write(paths.manifests / "audit_verdict.json", json.dumps(verdict, indent=1) + "\n")
    bad = {k: v for k, v in verdict.items() if v}
    print(f"[audit] {len(specs) - len(bad)}/{len(specs)} networks agree with PhyloNetworks")
    for k, v in bad.items():
        print(f"  {k}: {v}")
    return 1 if bad else 0


_TAXON = re.compile(r"[(,]([^(),:;]+):")


def _check_trees(path: Path, labels: list[str]) -> None:
    want = set(labels)
    n = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            n += 1
            got = _TAXON.findall(line)
            if len(got) != len(want) or set(got) != want:
                raise RuntimeError(f"{path.name}: tree {n} has the wrong taxon set")
    if n != N_LOCI:
        raise RuntimeError(f"{path.name}: {n} trees, expected {N_LOCI}")


def stage_genetrees(paths: Paths, specs: list[Spec], jobs: int, level: int = 2) -> int:
    if stage_audit(paths, specs, level):
        return 1
    todo = [s for s in specs if not paths.trees(s.key).exists()]
    if not todo:
        print("[genetrees] all cached")
        return 0
    paths.trees(todo[0].key).parent.mkdir(parents=True, exist_ok=True)
    # gene_trees.jl writes straight to `out`, so point it at a .part and rename only after the
    # count and taxon checks: a killed run must not leave a truncated file that reads as done.
    jl = [dict(enewick=str(paths.net(s.key)), out=str(paths.trees(s.key)) + ".part",
               n_loci=N_LOCI, seed=s.gt_seed, n_taxa=s.n_taxa, r=s.k, cu=s.cu, level=level)
          for s in todo]
    t0 = time.perf_counter()
    fails = julia_backend.run_shards(jl, paths.manifests / "genetrees", min(jobs, len(jl)))
    if fails:
        print("\n".join(["[genetrees] FAILED:"] + fails[:10]))
        return 1
    for s in todo:
        part = Path(str(paths.trees(s.key)) + ".part")
        _check_trees(part, _load(paths, s.key).labels)
        part.rename(paths.trees(s.key))
    print(f"[genetrees] {len(todo)} x {N_LOCI} gene trees in {time.perf_counter() - t0:.0f}s")
    return 0


# ------------------------------------------------------------------------------ estimation


def estimation_jobs(paths: Paths, specs: list[Spec], scratch: Path, chunks: int,
                    count: int | None = None, out_dir: Path | None = None,
                    aln_dir: Path | None = None) -> list[seqsim.EstJob]:
    """Chunks are whole IQ-TREE batches, so the estimates do not depend on the chunking."""
    jobs = []
    for s in specs:
        # Per spec, not per call: the batch size is a footprint bound in (n, bp), so a design at
        # another n (:mod:`opc.l2train`) batches differently -- 153 loci at n = 25 against 19 here.
        per_batch = seqsim.batch_loci(s.n_taxa + 1, BP)
        n_batches = -(-N_LOCI // per_batch)
        size = per_batch * -(-n_batches // max(1, chunks))
        n_chunks = -(-N_LOCI // size)
        seed_a, seed_i = s.est_seeds
        for c in range(n_chunks):
            start = c * size
            n = min(size, N_LOCI - start) if count is None else count
            jobs.append(seqsim.EstJob(
                cell_key=s.key, n_taxa=s.n_taxa, cu=s.cu, bp=BP, chunk=c, start=start, count=n,
                cell_m=N_LOCI, per_batch=per_batch, true_gz=paths.trees(s.key),
                out_dir=out_dir or paths.est, seed_alisim=seed_a, seed_iqtree=seed_i,
                model=MODEL, scratch=scratch, n_chunks=1 if count is not None else n_chunks,
                aln_dir=aln_dir or paths.aln,
            ))
            if count is not None:
                break
    return jobs


def _count_lines(gz: Path) -> int:
    with gzip.open(gz, "rt") as fh:
        return sum(1 for ln in fh if ln.strip())


def stage_pilot(paths: Paths, specs: list[Spec], scratch: Path) -> int:
    """Two IQ-TREE batches of the first network, into scratch: seconds and bytes per locus."""
    per_batch = seqsim.batch_loci(specs[0].n_taxa + 1, BP)
    pilot = scratch / "pilot"
    shutil.rmtree(pilot, ignore_errors=True)
    (job,) = estimation_jobs(paths, specs[:1], scratch, 1, count=2 * per_batch,
                             out_dir=pilot / "est", aln_dir=pilot / "aln")
    stem, status, secs = seqsim.run_est_job(job)
    if status != "ok":
        print(status)
        return 1
    side = json.loads(job.sidecar.read_text())
    alns = sorted((pilot / "aln" / job.cell_key).glob("*.phy.gz"))
    per_locus = secs / job.count
    aln_bytes = sum(p.stat().st_size for p in alns) / len(alns)
    total = len(specs) * N_LOCI
    print(f"[pilot] {job.count} loci of {stem} in {secs:.0f}s = {per_locus:.2f} s/locus "
          f"(AliSim {side['alisim_seconds'] / job.count:.2f}, IQ-TREE "
          f"{side['iqtree_seconds'] / job.count:.2f}); model {side['iqtree_model_used']}")
    print(f"[pilot] alignment width {side['aln_len_mean']:.0f}, gaps "
          f"{side['gap_fraction']:.1%}, {aln_bytes / 1e3:.0f} kB/locus gzipped -> "
          f"{aln_bytes * total / 1e9:.1f} GB for the dataset; RF {side['rf_norm_mean']:.3f}")
    print(f"[pilot] single-process: {per_locus * total / 3600:.1f} CPU-h for {total:,} loci")
    return 0


def stage_estimate(paths: Paths, specs: list[Spec], jobs: int, scratch: Path, chunks: int,
                   dry_run: bool) -> int:
    if not seqsim.available():
        print(f"no IQ-TREE at {seqsim.IQTREE}")
        return 2
    missing = [s.key for s in specs if not paths.trees(s.key).exists()]
    if missing:
        print(f"[estimate] no gene trees yet for {len(missing)} networks; run genetrees")
        return 1
    est = estimation_jobs(paths, specs, scratch, chunks)
    serial = sum(j.predicted_seconds() for j in est)
    print(f"[estimate] {len(est)} tasks, {sum(j.count for j in est):,} loci, --jobs {jobs}; "
          f"IQ-TREE {seqsim.iqtree_version()}")
    pen = seqsim.CONCURRENCY_PENALTY
    print(f"[estimate] budget {serial / 3600:.1f} CPU-h nominal, {serial * pen / 3600:.1f} with "
          f"the {pen}x concurrency penalty, ~{serial * pen / 3600 / jobs:.1f} h wall")
    if dry_run:
        return 0
    from multiprocessing import Pool

    paths.est.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    log = paths.est / f"run_bp{BP}.tsv"
    new = not log.exists()
    t0 = time.perf_counter()
    fails: list[str] = []
    done = Counter()
    with open(log, "a") as lf, Pool(processes=jobs) as pool:
        if new:
            lf.write("stem\tstatus\tseconds\n")
        for i, (stem, status, secs) in enumerate(
                pool.imap_unordered(seqsim.run_est_job, est, chunksize=1), 1):
            short = status.split("\n")[0]
            done[short] += 1
            if short.startswith("FAILED"):
                fails.append(f"--- {stem}\n{status}")
            lf.write(f"{stem}\t{short}\t{secs:.1f}\n")
            lf.flush()
            minutes = (time.perf_counter() - t0) / 60
            print(f"[estimate] {i}/{len(est)} {dict(done)} {minutes:.0f} min", flush=True)
    if fails:
        print("\n".join(fails[:3]))
        return 1
    for s in specs:
        seqsim.merge_chunks(s.key, BP, paths.root, "julia", out_dir=paths.est)
        for f in (paths.est / f"{s.key}.nwk.gz", paths.est / f"{s.key}.raw.nwk.gz"):
            if _count_lines(f) != N_LOCI:
                raise RuntimeError(f"{f.name}: expected {N_LOCI} trees")
        n_aln = len(list((paths.aln / s.key).glob("loc*.phy.gz")))
        if n_aln != N_LOCI:
            raise RuntimeError(f"{s.key}: {n_aln} alignments, expected {N_LOCI}")
    print(f"[estimate] all {len(specs)} networks complete")
    return 0


# ------------------------------------------------------------------------------ tree of blobs


def true_tob_clusters(net: N.Network) -> set[frozenset[str]]:
    """The true tree of blobs' clusters: one per bridge of the network, leaves excluded.

    Deleting a bridge leaves exactly its child's descendants on the child's side -- a vertex
    reached any other way would give a path around the bridge -- so each edge's cluster is the
    child's hardwired cluster, and the family is laminar.
    """
    below = N.hardwired_clusters(net)
    out = set()
    for b in N.blocks(net):
        if b.is_bridge:
            ((_, c),) = b.edges
            if 1 < len(below[c]) < net.n_taxa:
                out.add(frozenset(net.labels[t] for t in below[c]))
    return out


def laminar_newick(clusters, taxa: list[str]) -> str:
    """Rooted Newick for a laminar family over ``taxa``: a polytomy wherever the family has one."""
    rank = {x: i for i, x in enumerate(taxa)}
    fam = sorted({frozenset(c) for c in clusters}, key=len)
    top = frozenset(taxa)
    kids: dict[frozenset, list[frozenset]] = {top: []}
    for i, c in enumerate(fam):
        # Ascending size, so the first strict superset is the least one.
        kids.setdefault(next((d for d in fam[i + 1:] if c < d), top), []).append(c)

    def emit(node: frozenset) -> str:
        subs = kids.get(node, [])
        if any(not a.isdisjoint(b) for i, a in enumerate(subs) for b in subs[i + 1:]):
            raise ValueError("not a laminar family")
        items = sorted([(min(rank[x] for x in s), emit(s)) for s in subs]
                       + [(rank[x], x) for x in node.difference(*subs)])
        return "(" + ",".join(s for _, s in items) + ")"

    return emit(top) + ";"


def newick_clusters(newick: str) -> set[frozenset[str]]:
    """Clusters of a rooted Newick's internal nodes, the root excluded."""
    parent, _, lab, children = og.parse_newick(newick)
    leaves: list[set[str]] = [set() for _ in parent]
    for i in range(len(parent) - 1, 0, -1):
        if not children[i]:
            leaves[i] = {lab[i]}
        leaves[parent[i]] |= leaves[i]
    return {frozenset(leaves[i]) for i in range(1, len(parent)) if children[i]}


def tob_problems(newick: str, taxa: set[str]) -> list[str]:
    """``tob_qmc_v1/verify_tob.py``'s checks.  Labels are counted by splitting on structure: a
    non-overlapping findall undercounts adjacent labels in a polytomy."""
    s = newick.strip()
    toks = [t.strip() for t in re.split(r"[(),;:]", s) if t.strip()]
    bad = []
    if not s.endswith(";"):
        bad.append("no semicolon")
    if s.count("(") != s.count(")"):
        bad.append("unbalanced")
    if len(toks) != len(set(toks)):
        bad.append(f"{len(toks) - len(set(toks))} duplicate labels")
    if set(toks) != taxa:
        bad.append(f"label set differs from the input's ({len(set(toks) ^ taxa)} differ)")
    return bad


def _tob_one(task) -> tuple[str, str, float]:
    key, raw_gz, out, log, scratch = task
    t0 = time.perf_counter()
    part = out.with_name(out.name + ".part")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(dir=scratch, prefix="tob-") as td:
        gt = Path(td) / "gt.nwk"
        with gzip.open(raw_gz, "rb") as src, open(gt, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with open(log, "w") as lf:
            res = subprocess.run([str(TREEQMC), *TOB_ARGS, "-i", str(gt), "-o", str(part)],
                                 stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=td)
    secs = time.perf_counter() - t0
    if res.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return key, f"FAILED (exit {res.returncode}; see logs/{log.name})", secs
    part.rename(out)
    return key, "ok", secs


_EXEC = re.compile(r"Execution time: ([\d.]+)\s*secs")
_VERSION = re.compile(r"TREE-QMC version (\S+)")
_ALPHA_BETA = re.compile(r"Contracting branches with alpha = (\S+) and beta = (\S+)")
_PVALUES = "Printing output tree with pvalues:"
_TOKEN = re.compile(r"'[^']*'|[(),;]|[^(),;']+")


def pvalue_tree(log: str) -> str:
    """The annotated tree ``--blob`` prints before contracting: every tested branch's p-values."""
    lines = log.splitlines()
    return lines[lines.index(_PVALUES) + 1].strip()


def annotated_splits(newick: str):
    """``(all leaves, [(leaves, annotation or None), ...])`` for a TREE-QMC p-value tree's
    non-root internal nodes; ``None`` marks a node TREE-QMC never tested.  Quote-aware, since the
    annotations hold ``;``, which :func:`outgroup.parse_newick` would read as the end of the tree.
    """
    stack: list[list[frozenset[str]]] = [[]]
    nodes: list[list] = []
    last = None
    for tok in (t.strip() for t in _TOKEN.findall(newick)):
        if not tok:
            continue
        if tok == "(":
            stack.append([])
            last = None
        elif tok == ")":
            leaves = frozenset().union(*stack.pop())
            nodes.append([leaves, None])
            stack[-1].append(leaves)
            last = len(nodes) - 1
        elif tok == ",":
            last = None
        elif tok == ";":
            break
        elif tok.startswith("'"):
            if last is None:
                raise ValueError("an annotation that follows no internal node")
            nodes[last][1] = dict(kv.split("=", 1) for kv in tok.strip("'[]").split(";"))
        elif last is None:
            stack[-1].append(frozenset({tok.split(":")[0].strip()}))
    *inner, (everything, _) = nodes
    return everything, [(leaves, ann) for leaves, ann in inner]


def contract_tob(newick: str, alpha: float, beta: float, outgroup: str = og.OUTGROUP_LABEL):
    """TREE-QMC's contraction rule, applied to splits rather than to a rooted tree (A63).

    A tested branch is contracted iff ``qtt_p < alpha or qst_p > beta``: the rule in TREE-QMC's
    ``SpeciesTree(Tree *, Dict *, alpha, beta, bool)``.  TREE-QMC applies it by splicing each
    contracted node into its parent, and tests a binary root's branch on the root's *first* child
    only (``get_bipartitions``).  So contracting that branch folds the first child into the root
    and leaves the second child's edge standing: a branch its own rule removed survives.  As a
    set of splits there is no root to get wrong.

    Returns the kept splits as clusters on the side away from ``outgroup`` -- the rooted tree of
    blobs -- and what was tested, contracted, and decided at the root.
    """
    everything, nodes = annotated_splits(newick)
    if outgroup not in everything:
        raise ValueError(f"no {outgroup!r} to root on")
    n_in = len(everything) - 1

    def side(c: frozenset[str]) -> frozenset[str]:
        return c if outgroup not in c else everything - c

    tested: dict[frozenset[str], tuple[float, float]] = {}
    for leaves, ann in nodes:
        if ann is not None:
            if side(leaves) in tested:
                raise ValueError("a branch tested twice")
            tested[side(leaves)] = (float(ann["qtt_p"]), float(ann["qst_p"]))
    untested = {side(leaves) for leaves, ann in nodes if ann is None}
    if not untested <= set(tested):
        raise ValueError("an internal branch TREE-QMC never tested")
    cut = {s for s, (qtt, qst) in tested.items() if qtt < alpha or qst > beta}
    kept = {s for s in tested if s not in cut and 1 < len(s) < n_in}
    return kept, {
        "n_tested": len(tested), "n_contracted": len(cut),
        "n_contracted_alpha": sum(qtt < alpha for qtt, _ in tested.values()),
        "n_contracted_beta": sum(qtt >= alpha and qst > beta for qtt, qst in tested.values()),
        # The branch shared by the root's two children, when TREE-QMC wrote a binary root.
        "root_branch": [{"cluster": sorted(s, key=lambda x: (len(x), x)), "qtt_p": tested[s][0],
                         "qst_p": tested[s][1], "contracted": s in cut} for s in untested],
    }


def finalize_tob(paths: Paths, specs: list[Spec]) -> int:
    """Check each raw tree of blobs, re-derive it from its own p-values, and score both (A63)."""
    bad, tally = [], Counter()
    for s in specs:
        raw = (paths.tob / f"{s.key}.raw.tob").read_text()
        net = _load(paths, s.key)
        problems = tob_problems(raw, set(net.labels) | {og.OUTGROUP_LABEL})
        if problems:
            bad.append(f"{s.key}: {problems}")
            continue
        log = (paths.tob / "logs" / f"{s.key}.log").read_text()
        pv = pvalue_tree(log)
        ab = _ALPHA_BETA.search(log)
        alpha, beta = float(ab.group(1)), float(ab.group(2))
        est, info = contract_tob(pv, alpha, beta)
        got = newick_clusters(og.reroot_on_outgroup(raw))
        extra = got - est
        cut_root = {frozenset(r["cluster"]) for r in info["root_branch"] if r["contracted"]}
        if not est <= got or not extra <= cut_root:
            bad.append(f"{s.key}: TREE-QMC's output is not its own rule on its own p-values")
            continue
        true = true_tob_clusters(net)
        secs, ver = _EXEC.search(log), _VERSION.search(log)
        side = {
            "key": s.key, "input": f"estimated_bp{BP}/{s.key}.raw.nwk.gz",
            "tool": "TREE-QMC", "version": ver.group(1) if ver else None,
            "commit": TREEQMC_COMMIT, "args": list(TOB_ARGS), "alpha": alpha, "beta": beta,
            "seconds": float(secs.group(1)) if secs else None, "rooted_on": og.OUTGROUP_LABEL,
            **info,
            "treeqmc_kept_a_contracted_root_branch": bool(extra),
            "n_blobs": len(L2.nontrivial_blocks(net)),
            "true_tob": laminar_newick(true, net.labels),
            "n_true_edges": len(true), "n_est_edges": len(est),
            "fn": len(true - est), "fp": len(est - true),
            "fn_raw": len(true - got), "fp_raw": len(got - true),
        }
        _write(paths.tob / f"{s.key}.tob", laminar_newick(est, net.labels) + "\n")
        _write(paths.tob / f"{s.key}.pvalues.tob", pv + "\n")
        _write(paths.tob / f"{s.key}.json", json.dumps(side, indent=1) + "\n")
        tally.update(exact=est == true, exact_raw=got == true, fixed=bool(extra),
                     fn=side["fn"], fp=side["fp"], true=len(true))
    if bad:
        print("\n".join(["[tob] FAILED checks:", *bad]))
        return 1
    print(f"[tob] {len(specs)} trees of blobs: {tally['exact']} exactly the true one "
          f"({tally['exact_raw']} as TREE-QMC wrote them; {tally['fixed']} kept a root branch "
          f"its own test contracted); {tally['fn']} of {tally['true']} true edges missed, "
          f"{tally['fp']} extra")
    return 0


def stage_tob(paths: Paths, specs: list[Spec], jobs: int, scratch: Path,
              note: str = "~3 h each at n = 201 (tob_qmc_v1, n = 200: 3.0-3.4 h)") -> int:
    """TREE-QMC ``--blob`` per network, on its estimated gene trees with OUT (A63).

    ``note`` is the per-run time printed before the pool starts.  It is a parameter because the
    cost is entirely in ``n``: measured on this box, 24 s at n = 25 and 243 s at n = 50 against
    ~3.7 h at n = 200, so a driver at another ``n`` (:mod:`opc.l2train`) says its own number.
    """
    if not TREEQMC.exists():
        print(f"no TREE-QMC at {TREEQMC} (set OPC_TREEQMC)")
        return 2
    missing = [s.key for s in specs if not (paths.est / f"{s.key}.raw.nwk.gz").exists()]
    if missing:
        print(f"[tob] no estimated gene trees yet for {len(missing)} networks; run estimate")
        return 1
    (paths.tob / "logs").mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (paths.tob / f"{s.key}.raw.tob").exists()]
    print(f"[tob] {len(todo)} of {len(specs)} networks to run, {min(jobs, max(1, len(todo)))} "
          f"at a time, single-threaded: {note}")
    if todo:
        from multiprocessing import Pool

        tasks = [(s.key, paths.est / f"{s.key}.raw.nwk.gz", paths.tob / f"{s.key}.raw.tob",
                  paths.tob / "logs" / f"{s.key}.log", scratch) for s in todo]
        fails = []
        with Pool(min(jobs, len(tasks))) as pool:
            for i, (key, status, secs) in enumerate(
                    pool.imap_unordered(_tob_one, tasks, chunksize=1), 1):
                print(f"[tob] {i}/{len(tasks)} {key}: {status} ({secs / 60:.0f} min)", flush=True)
                if status != "ok":
                    fails.append(key)
        if fails:
            return 1
    return finalize_tob(paths, specs)


def _tob_tool() -> dict:
    """TREE-QMC and the R it embeds, as installed."""
    root = TREEQMC.parents[1]

    def version(pkg: str) -> str:
        for lib in (root / "Rlib", Path("/usr/lib64/R/library")):
            d = lib / pkg / "DESCRIPTION"
            if d.exists():
                m = re.search(r"^Version:\s*(\S+)", d.read_text(), re.M)
                return m.group(1) if m else "unknown"
        return "absent"

    vtxt = root / "version.txt"
    r = subprocess.run(["R", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    return {"binary": str(TREEQMC), "version": vtxt.read_text().strip() if vtxt.exists() else
            "unknown", "commit": TREEQMC_COMMIT, "args": list(TOB_ARGS), "R": r,
            **{p: version(p) for p in ("Rcpp", "RInside", "MSCquartets", "ape", "phangorn")}}


# ------------------------------------------------------------------------------ package


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout


def _julia_versions() -> dict:
    text = (julia_backend.PROJECT / "Manifest.toml").read_text()
    out = {}
    for pkg in ("PhyloNetworks", "PhyloCoalSimulations"):
        m = re.search(r"\[\[deps\.%s\]\].*?version = \"([^\"]+)\"" % pkg, text, re.S)
        out[pkg] = m.group(1) if m else "unknown"
    res = subprocess.run([str(julia_backend.JULIA), "--version"], capture_output=True, text=True)
    out["julia"] = res.stdout.strip()
    return out


def _snapshot_code(dst: Path) -> dict:
    """The exact code the dataset came from, committed or not."""
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    patch = _git("diff", "HEAD", "--", "src", "tests")
    (dst / "tracked.diff").write_text(patch)
    new = [p for p in _git("ls-files", "--others", "--exclude-standard", "src", "tests").split()
           if p]
    for rel in new:
        (dst / "untracked" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst / "untracked" / rel)
    return {"commit": _git("rev-parse", "HEAD").strip(), "dirty": bool(patch or new),
            "tracked_diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "untracked_files": new}


def stage_package(paths: Paths, specs: list[Spec]) -> int:
    rows, recs, sides, tobs = [], {}, {}, {}
    for s in specs:
        need = [paths.net(s.key), paths.regular(s.key), paths.meta(s.key), paths.trees(s.key),
                paths.est / f"{s.key}.nwk.gz", paths.est / f"{s.key}.raw.nwk.gz",
                paths.est / f"{s.key}.json", paths.tob / f"{s.key}.raw.tob",
                paths.tob / f"{s.key}.tob", paths.tob / f"{s.key}.pvalues.tob",
                paths.tob / f"{s.key}.json"]
        absent = [p.name for p in need if not p.exists()]
        n_aln = len(list((paths.aln / s.key).glob("loc*.phy.gz")))
        if absent or n_aln != N_LOCI:
            print(f"[package] {s.key}: incomplete ({absent}, {n_aln} alignments)")
            return 1
        rec = recs[s.key] = json.loads(paths.meta(s.key).read_text())
        side = sides[s.key] = json.loads((paths.est / f"{s.key}.json").read_text())
        tob = tobs[s.key] = json.loads((paths.tob / f"{s.key}.json").read_text())
        rows.append({
            "key": s.key, "rep": s.rep, "n_taxa": N_TAXA, "n_l2": s.n_l2, "n_l1": s.n_l1,
            "k": s.k, "cu": s.cu, "shapes": ",".join(rec["check"]["shapes"]),
            "girth_R_min": rec["check"]["girth_R_min"],
            "tree_child": rec["check"]["regularity"]["tree_child"],
            "n_loci": N_LOCI, "bp": BP,
            "network": f"networks/{s.key}.enewick",
            "regular_network": f"networks/{s.key}.regular.enewick",
            "truth": f"networks/{s.key}.json",
            "genetrees": f"trees/{s.key}.nwk.gz",
            "alignments": f"alignments_bp{BP}/{s.key}/",
            "estimated": f"estimated_bp{BP}/{s.key}.nwk.gz",
            "estimated_raw": f"estimated_bp{BP}/{s.key}.raw.nwk.gz",
            "tob": f"tob_bp{BP}/{s.key}.tob",
            "tob_raw": f"tob_bp{BP}/{s.key}.raw.tob",
            "tob_pvalues": f"tob_bp{BP}/{s.key}.pvalues.tob",
            "aln_width_mean": round(side["aln_len_mean"], 1),
            "gap_fraction": round(side["gap_fraction"], 4),
            "rf_norm_mean": round(side["rf_norm_mean"], 4),
            "tob_true_edges": tob["n_true_edges"],
            "tob_est_edges": tob["n_est_edges"],
            "tob_fn": tob["fn"],
            "tob_fp": tob["fp"],
            "tob_root_fixed": tob["treeqmc_kept_a_contracted_root_branch"],
            "network_sha256": _sha256(paths.net(s.key)),
            "genetrees_sha256": _sha256(paths.trees(s.key)),
        })
    header = list(rows[0])
    _write(paths.root / "MANIFEST.tsv", "\t".join(header) + "\n" + "".join(
        "\t".join(str(r[h]) for h in header) + "\n" for r in rows))

    audit = json.loads((paths.manifests / "audit_verdict.json").read_text())
    code = _snapshot_code(paths.root / "code")
    manifest = {
        "name": paths.root.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code,
        "master_seed": MASTER_SEED, "n_taxa": N_TAXA, "n_networks": N_NETWORKS,
        "n_loci": N_LOCI, "bp": BP, "gamma_range": list(GAMMA_RANGE),
        "min_girth": L2.MIN_GIRTH,
        "design": [s.row() for s in specs],
        "model": MODEL.as_dict(),
        "tools": {"iqtree": seqsim.iqtree_version(), "iqtree_binary": str(seqsim.IQTREE),
                  **_julia_versions(), "python": sys.version.split()[0],
                  "numpy": np.__version__, "tree_qmc": _tob_tool()},
        "seed_check": [{"cell_key": s.key, "bp": BP, "seed_alisim": s.est_seeds[0],
                        "seed_iqtree": s.est_seeds[1], "gt_seed": s.gt_seed,
                        "net_seed": s.net_seed} for s in specs[:4]],
        "gates": {
            "python_check_ok": all(r["check"]["ok"] for r in recs.values()),
            "phylonetworks_audit_ok": not any(audit.values()),
            "iqtree_model_used": sorted({v["iqtree_model_used"] for v in sides.values()}),
            "tob_exact": sum(t["fn"] == t["fp"] == 0 for t in tobs.values()),
        },
    }
    _write(paths.root / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    _write(paths.root / "README.md", readme(specs, recs, sides, tobs, manifest))

    files = sorted(p for p in paths.root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                   and not p.name.endswith((".tmp", ".part")))
    _write(paths.root / "SHA256SUMS", "".join(
        f"{_sha256(p)}  {p.relative_to(paths.root)}\n" for p in files))
    print(f"[package] {len(rows)} networks, {len(files)} files checksummed -> {paths.root}")
    return 0


def _files() -> list[tuple[str, str]]:
    """The README's file table, kept as data so that no source line has to hold a whole row."""
    return [
        ("networks/<key>.enewick",
         "the network **as simulated**: binary, time-consistent, ultrametric, branch lengths in "
         "coalescent units, one gamma per reticulation (PhyloNetworks `name:length::gamma` "
         "dialect, `#Hk` copy on the minor edge)"),
        ("networks/<key>.regular.enewick",
         "its **regular form** R: every hybrid merged with its only child, so a hybrid carries "
         "its child's children. Topology only -- R's branch lengths are not a model. A hybrid "
         "whose cluster is one taxon is written `(t:0)#Hk`. **Never simulate from R**: "
         "PhyloCoalSimulations sends a hybrid leaf's lineages up its first parent only"),
        ("networks/<key>.json",
         "the ground truth: blobs (shape, girth in both forms, size), reticulations (gamma, "
         "time, hybrid / donor / tree-parent clusters), R as clusters plus Hasse edges "
         "(`regular_network`), every gate, and the generator's rejection counters"),
        ("networks/<key>.pkl", "the `opc.simulate.networks.Network` object itself"),
        ("trees/<key>.nwk.gz",
         f"{N_LOCI:,} rooted true gene trees, one per line, coalescent units"),
        (f"alignments_bp{BP}/<key>/locNNNN.phy.gz",
         f"AliSim's alignment for locus NNNN, sequential PHYLIP, **{N_TAXA} taxa plus the "
         "outgroup `OUT`**"),
        (f"estimated_bp{BP}/<key>.nwk.gz",
         "IQ-TREE's tree for each locus, rooted on `OUT`, `OUT` pruned, **topology only**, in "
         "locus order"),
        (f"estimated_bp{BP}/<key>.raw.nwk.gz",
         "what IQ-TREE returned: unrooted, `OUT` attached, branch lengths in substitutions/site"),
        (f"estimated_bp{BP}/<key>.json",
         "alignment widths, gap fraction, seeds, timings, the model IQ-TREE reported using, RF "
         "against the true trees"),
        (f"tob_bp{BP}/<key>.tob",
         "the estimated **tree of blobs**: TREE-QMC `--blob` on the network's 1,000 estimated "
         "gene trees (the `.raw.nwk.gz`, `OUT` included), rooted on `OUT` and `OUT` pruned, with "
         "TREE-QMC's root-branch defect corrected (see below). Topology only; a polytomy wherever "
         "it found a blob"),
        (f"tob_bp{BP}/<key>.raw.tob",
         "TREE-QMC's output exactly as written: unrooted, `OUT` included, defect and all"),
        (f"tob_bp{BP}/<key>.pvalues.tob",
         "the tree TREE-QMC tested, each branch annotated with its tree- and star-test p-values, "
         "qCFs and minimising quartet -- what `<key>.tob` is derived from"),
        (f"tob_bp{BP}/<key>.json",
         "the command, TREE-QMC version, `alpha` / `beta` and run time; what was contracted, and "
         "at the root; the **true** tree of blobs (`true_tob`, one edge per bridge of the "
         "network); missed (`fn`) / extra (`fp`) edges against it, and the same for the raw "
         "output (`fn_raw`, `fp_raw`)"),
        (f"tob_bp{BP}/logs/<key>.log", "TREE-QMC's log, including every per-branch quartet test"),
        ("MANIFEST.tsv", "one row per network -- the index to iterate over"),
        ("manifest.json", "design, seeds, model, tool versions, code state, gate summary"),
        ("manifests/audit.json",
         "PhyloNetworks' own reading of every network; `audit_verdict.json` compares it with "
         "what the generator meant"),
        ("code/", "the code the dataset was built from: `tracked.diff` against the commit, plus "
         "new files"),
        ("SHA256SUMS", "over every file above: `sha256sum -c SHA256SUMS --quiet`"),
    ]


def readme(specs, recs, sides, tobs, manifest) -> str:
    shapes = Counter(s for r in recs.values() for s in r["check"]["shapes"])
    blobs = [b for r in recs.values() for b in r["blobs"]]
    girths = Counter(b["girth_R"] for b in blobs)
    spans = np.array([b["root_taxa"] for b in blobs])
    tc = sum(r["check"]["regularity"]["tree_child"] for r in recs.values())
    width = np.mean([s["aln_len_mean"] for s in sides.values()])
    gaps = np.mean([s["gap_fraction"] for s in sides.values()])
    rf = np.mean([s["rf_norm_mean"] for s in sides.values()])
    design_rows = "\n".join(
        f"| `{s.key}` | {s.n_l2} | {s.n_l1} | {s.k} | {s.cu} | "
        f"{', '.join(recs[s.key]['check']['shapes'])} | {recs[s.key]['check']['girth_R_min']} |"
        for s in specs)
    code = manifest["code"]
    tq = manifest["tools"]["tree_qmc"]
    t_true = sum(t["n_true_edges"] for t in tobs.values())
    t_fn = sum(t["fn"] for t in tobs.values())
    t_fp = sum(t["fp"] for t in tobs.values())
    t_exact = sum(t["fn"] == t["fp"] == 0 for t in tobs.values())
    t_exact_raw = sum(t["fn_raw"] == t["fp_raw"] == 0 for t in tobs.values())
    t_fixed = sum(t["treeqmc_kept_a_contracted_root_branch"] for t in tobs.values())
    t_hours = [t["seconds"] / 3600 for t in tobs.values() if t["seconds"]]
    tob_rows = "\n".join(
        f"| `{s.key}` | {tobs[s.key]['n_blobs']} | {tobs[s.key]['n_true_edges']} | "
        f"{tobs[s.key]['n_est_edges']} | {tobs[s.key]['fn']} | {tobs[s.key]['fp']} | "
        f"{'yes' if tobs[s.key]['treeqmc_kept_a_contracted_root_branch'] else ''} |"
        for s in specs)
    files_table ="\n".join(["| path | what it is |", "|---|---|"]
                            + [f"| `{p}` | {d} |" for p, d in _files()])
    return f"""# {manifest['name']} -- regular, exactly-level-2 networks at n = {N_TAXA}

Built {manifest['built_utc']} by `python -m opc.l2reg` from rl-net-opt at `{code['commit'][:12]}`
{"(with uncommitted changes -- the exact code is in `code/`)" if code['dirty'] else ""}.

{N_NETWORKS} phylogenetic networks on {N_TAXA} taxa (`t1`..`t{N_TAXA}`). For each network,
{N_LOCI:,} gene trees simulated under the network multispecies coalescent, and for each gene tree
a {BP} bp AliSim alignment and an IQ-TREE re-estimate.

## Every network is

* **regular** -- the Hasse diagram of its hardwired clusters *is* the network, so it is
  reconstructible from its clusters without ambiguity. Checked by definition, not by a
  sufficient condition: the cluster map on the regular form is injective, and its edges, read as
  cluster pairs, are exactly the cover relation of the cluster set.
* **exactly level-2** -- every blob (biconnected component) holds at most two reticulations and
  at least one holds two.
* of **girth >= {L2.MIN_GIRTH}** -- no undirected cycle has fewer than {L2.MIN_GIRTH} edges,
  measured on the regular form, which is the stricter of the two forms below.

All three are checked in Python (`networks/<key>.json`, `check`) and again by an independent
PhyloNetworks parse of every written file (`manifests/audit.json`, `audit_verdict.json`):
clusters, per-hybrid gammas and which edge carries the `#H` copy, level, per-blob reticulation
counts, hybrid ladders and girth all agree.

## Files

{files_table}

## Design

Each (level-2 blobs, level-1 blobs) pair in {{1,2,3}}^2 once per ILS level, and (2, 2) once more.
ILS is the median internal branch, in coalescent units. The four level-2 shapes are rotated
within each ILS level. Every network is also tree-child here ({tc} of {N_NETWORKS}), though only
regularity was required.

| key | level-2 blobs | level-1 blobs | k | cu | shapes | girth |
|---|---|---|---|---|---|---|
{design_rows}

## Level-2 blob shapes

A blob with two reticulations is a subdivided theta graph: two branch vertices P, Q joined by
three paths, one source rho (never P or Q) and one or two sinks. There are exactly four directed
shapes -- the four level-2 generators:

| shape | sinks | the P-Q path through rho | e.g. |
|---|---|---|---|
| S0 | 2 | holds no sink | two reticulations side by side under a common split pair |
| S1 | 1 | holds no sink | one hybrid feeding the other, fed directly from rho |
| S2 | 1 | holds the sink | a nested cycle: the inner hybrid feeds the outer one |
| S3 | 2 | holds a sink | one reticulation's donor path runs through the other's cycle |

Realised: {dict(sorted(shapes.items()))}. Girth on R, over all {len(blobs)} blobs:
{dict(sorted(girths.items()))}. Blob size (taxa below the blob's root): median
{int(np.median(spans))}, range {spans.min()}-{spans.max()} -- no ceiling was imposed, so some blobs
span most of the tree.

## Sequence model

`{MODEL.sim_model_string()}` + indels `{MODEL.indel_ins:g},{MODEL.indel_del:g}`, size
`{MODEL.indel_size}`, branch scale `{MODEL.branch_scale}` substitutions/site per coalescent unit
(theta = {MODEL.theta:g}), inferred under `{MODEL.inference_model}` with `--fast --keep-ident` --
identical to `alisim_gt_v1/v2`. `{BP}` is the **root** sequence length: indels make realised
alignments wider (mean {width:.0f} columns here, {gaps:.0%} gaps). The outgroup is grafted onto
each gene tree at `(1 + {MODEL.outgroup_alpha}) x p99` of that network's root heights, so it
contributes no rooting error. Mean normalised RF of the estimates against the true trees:
{rf:.3f}. IQ-TREE: `{manifest['tools']['iqtree']}` (v1/v2 used 2.4.0).

## Tree of blobs

`tob_bp{BP}/` holds one estimated tree of blobs per network: TREE-QMC {tq['version']}
(`{tq['commit'][:7]}`) run as `tree-qmc {' '.join(tq['args'])}` on the network's {N_LOCI:,}
estimated gene trees, `OUT` included -- default `alpha` / `beta` and seeds, the configuration
`tob_qmc_v1` used. `--hybrid` weighting is not used: it weighs by branch support, and these
IQ-TREE trees carry none. Each run is single-threaded, {min(t_hours):.1f}-{max(t_hours):.1f} h.

**`<key>.tob` is not TREE-QMC's output verbatim.** TREE-QMC tests a binary root's branch on the
root's first child only and contracts by splicing nodes into their parents, so when its own test
rejects that branch the second child's edge survives: one edge its own rule removed stays in the
output. `<key>.tob` applies TREE-QMC's rule -- contract iff `qtt_p < alpha` or `qst_p > beta` --
to the p-values it logged for every branch (`<key>.pvalues.tob`), and is checked to equal
TREE-QMC's output (`<key>.raw.tob`) everywhere except exactly that branch. That branch was
present in {t_fixed} of {len(tobs)} raw outputs.

Against the true tree of blobs -- one edge per bridge of the network, `true_tob` in each JSON --
{t_exact} of {len(tobs)} are exactly right ({t_exact_raw} of TREE-QMC's raw outputs), and over all
20, {t_fn} of {t_true} true edges are missed and {t_fp} are extra. An edge is compared as the
cluster below it, after rooting on `OUT`.

| key | blobs | true edges | estimated edges | missed | extra | root fixed |
|---|---|---|---|---|---|---|
{tob_rows}

## Reproducibility

Seeds are SHA-256 chains off master seed `{MASTER_SEED}`: networks
`(master, "l2reg", "net", rep)`, gene trees `(master, "l2reg", "gt", key)`, AliSim / IQ-TREE
`(master, "alisim" | "iqtree", key, bp)` plus the global locus / batch index.
`python -m opc.l2reg all` rebuilds everything; each stage skips what already exists.
"""


# ------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opc.l2reg", description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("probe", "networks", "genetrees", "pilot", "estimate",
                                      "tob", "package", "all"))
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--jobs", type=int, default=min(48, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()) / "l2reg")
    ap.add_argument("--chunks", type=int, default=5, help="estimation tasks per network")
    ap.add_argument("--seeds", type=int, default=10, help="probe: seeds per (layout, shape)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="networks: rebuild even if present")
    args = ap.parse_args(argv)
    if args.jobs >= 100:
        print("refusing --jobs >= 100: a pool that large deadlocks silently under the shared "
              "RLIMIT_NPROC (CLAUDE.md)")
        return 2
    paths, specs = Paths(args.pkg), design()
    stages = {
        "probe": lambda: stage_probe(args.seeds, args.jobs),
        "networks": lambda: stage_networks(paths, specs, args.force),
        "genetrees": lambda: stage_genetrees(paths, specs, args.jobs),
        "pilot": lambda: stage_pilot(paths, specs, args.scratch),
        "estimate": lambda: stage_estimate(paths, specs, args.jobs, args.scratch, args.chunks,
                                           args.dry_run),
        "tob": lambda: stage_tob(paths, specs, args.jobs, args.scratch),
        "package": lambda: stage_package(paths, specs),
    }
    order = ["probe", "networks", "genetrees", "estimate", "tob", "package"] \
        if args.stage == "all" else [args.stage]
    for st in order:
        code = stages[st]()
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
