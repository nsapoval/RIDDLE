"""``l3reg_n200_v1``: a benchmark of regular, exactly-level-3 networks at n = 200 (A64).

    python -m opc.l3reg probe                    # BLOCKING gate, before anything is built
    python -m opc.l3reg networks                 # 20 networks: N, R, and their ground truth
    python -m opc.l3reg genetrees                # PhyloNetworks audit (BLOCKING), then the MSNC
    python -m opc.l3reg pilot                    # two IQ-TREE batches, to confirm the budget
    python -m opc.l3reg estimate --jobs 48 --dry-run
    python -m opc.l3reg estimate --jobs 48       # AliSim + IQ-TREE, alignments kept
    python -m opc.l3reg tob --jobs 20            # TREE-QMC --blob on the estimated gene trees
    python -m opc.l3reg package                  # MANIFEST.tsv, manifest.json, README, SHA256SUMS
    python -m opc.l3reg all --jobs 48

This is ``l2reg_n200_v1`` (:mod:`opc.l2reg`) at level 3.  The design was fixed with the user on
2026-09-11:

* 20 networks, 200 taxa, 1,000 gene trees each, 1,000 bp (root length) per locus, as l2reg;
* every network regular, exactly level-3, and of girth >= 5 on its regular form (:mod:`level3`);
* each ``(level-3 blobs, level-1 blobs)`` pair in ``{1,2,3}^2`` once per ILS level, and
  ``(2, 2)`` twice, so ``k = 3a + b`` runs 4..12;
* level-3 blobs stratified by their number of sinks, with 1 / 2 / 3 rotated within each ILS
  level.  The generator within a stratum is free, and it is recorded (:mod:`generators`);
* ILS, gammas, the sequence model, IQ-TREE and TREE-QMC all exactly as l2reg.

Only what depends on the level lives here.  Estimation, the tree of blobs and the helpers are
:mod:`opc.l2reg`'s, called unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opc import l2reg
from opc.grid import chain_seed
from opc.l2reg import (
    BP,
    COMBOS,
    CUS,
    GAMMA_RANGE,
    MASTER_SEED,
    MODEL,
    N_LOCI,
    N_NETWORKS,
    N_TAXA,
    REPO,
    Paths,
    _write,
)
from opc.simulate import generators as G
from opc.simulate import level2 as L2
from opc.simulate import level3 as L3
from opc.simulate import networks as N
from opc.simulate import seqsim

PKG = REPO / "l3reg_n200_v1"
PROBE_OUT = REPO / "results" / "validation" / "l3reg_probe.json"
#: The seed namespace.  Keys carry ``l3x`` as well, and the AliSim / IQ-TREE seeds chain off the
#: key alone (``Spec.est_seeds``), so no stream is shared with ``l2reg_n200_v1``.
TAG = "l3reg"
LEVEL = 3


# ------------------------------------------------------------------------------ the design


@dataclass(frozen=True)
class Spec:
    rep: int
    n_l3: int
    n_l1: int
    cu: float
    targets: tuple[str, ...]

    @property
    def n_taxa(self) -> int:
        """As :attr:`opc.l2reg.Spec.n_taxa`: a property, so ``row()`` and the JSON are unchanged."""
        return N_TAXA

    @property
    def key(self) -> str:
        return f"n{N_TAXA}_rep{self.rep:02d}_l3x{self.n_l3}_l1x{self.n_l1}_cu{self.cu}"

    @property
    def k(self) -> int:
        return 3 * self.n_l3 + self.n_l1

    @property
    def layout(self) -> L3.Layout3:
        return L3.Layout3(self.n_l3, self.n_l1, self.targets)

    @property
    def net_seed(self) -> int:
        return chain_seed(MASTER_SEED, TAG, "net", self.rep)

    @property
    def gt_seed(self) -> int:
        return chain_seed(MASTER_SEED, TAG, "gt", self.key) % (2 ** 31)

    @property
    def est_seeds(self) -> tuple[int, int]:
        """AliSim / IQ-TREE bases, derived exactly as ``l2reg.Spec.est_seeds`` derives them."""
        return (chain_seed(MASTER_SEED, "alisim", self.key, BP) % (2 ** 31),
                chain_seed(MASTER_SEED, "iqtree", self.key, BP) % (2 ** 31))

    def row(self) -> dict:
        return dataclasses.asdict(self) | {
            "key": self.key, "k": self.k, "net_seed": self.net_seed, "gt_seed": self.gt_seed}


def design() -> list[Spec]:
    """The 20 networks: ``l2reg.design``'s loop, with sink counts rotated within each ILS level."""
    used = dict.fromkeys(CUS, 0)
    specs = []
    for rep in range(N_NETWORKS):
        cu = CUS[rep % len(CUS)]
        a, b = COMBOS[(rep // len(CUS)) % len(COMBOS)]
        targets = tuple(L3.SINK_TARGETS[(used[cu] + i) % len(L3.SINK_TARGETS)] for i in range(a))
        used[cu] += a
        specs.append(Spec(rep, a, b, cu, targets))
    return specs


# ------------------------------------------------------------------------------ level-2 identity


def l2_identity(pkg: Path = l2reg.PKG) -> dict:
    """Rebuild every published level-2 network with this module loaded, and compare byte for byte.

    The eNewick of ``N`` and of ``R``, the pickle, the ground-truth JSON and the audit verdict are
    all compared.  ``placement.seconds`` is wall clock, so it is copied across before comparing.
    Everything else in the JSON must match, including the rejection counters, which is the RNG
    consumption.
    """
    paths = Paths(pkg)
    audit_rows = {Path(r["path"]).name[: -len(".enewick")]: r
                  for r in json.loads((paths.manifests / "audit.json").read_text())}
    verdict = json.loads((paths.manifests / "audit_verdict.json").read_text())
    specs = l2reg.design()
    differ: dict[str, list[str]] = {}
    for s in specs:
        net, placement = L2.build_level2(N_TAXA, s.layout, s.net_seed, cu=s.cu,
                                         gamma_range=GAMMA_RANGE)
        bad = []
        if (N.to_enewick(net) + "\n").encode() != paths.net(s.key).read_bytes():
            bad.append("enewick")
        if (N.to_enewick(L2.contract(net)) + "\n").encode() != paths.regular(s.key).read_bytes():
            bad.append("regular.enewick")
        if pickle.dumps(net) != paths.pickle(s.key).read_bytes():
            bad.append("pkl")
        published = paths.meta(s.key).read_bytes()
        rec = l2reg.network_record(s, net, placement)
        rec["placement"]["seconds"] = json.loads(published)["placement"]["seconds"]
        if (json.dumps(rec, indent=1) + "\n").encode() != published:
            bad.append("json")
        if l2reg.audit_problems(net, audit_rows[s.key]) != verdict[s.key]:
            bad.append("audit")
        if bad:
            differ[s.key] = bad
    return {"ok": not differ, "n_networks": len(specs), "differ": differ}


# ------------------------------------------------------------------------------ probe


def _probe_one(task):
    (a, b), mode, seed = task
    layout = L3.Layout3(a, b, () if mode == "free" else (mode,) * a)
    t0 = time.perf_counter()
    try:
        net, placement = L3.build_level3(N_TAXA, layout, seed, cu=0.5, gamma_range=GAMMA_RANGE)
    except Exception as exc:  # noqa: BLE001 - a failed build is the measurement
        return {"combo": [a, b], "mode": mode, "seed": seed, "ok": False, "error": str(exc)[:500],
                "seconds": time.perf_counter() - t0}
    report = L3.blob_report3(net)
    l3 = [r for r in report if r["n_retics"] == 3]
    return {
        "combo": [a, b], "mode": mode, "seed": seed, "ok": L3.check3(net, layout)["ok"],
        "seconds": time.perf_counter() - t0, "base_draw": placement["base_draw"],
        "counters": placement["counters"],
        "sinks": sorted(r["n_sinks"] for r in l3),
        "generators": sorted(r["generator"] for r in l3),
        "girth_R": [r["girth_R"] for r in report],
        "root_taxa": [r["root_taxa"] for r in report],
        "tree_child": L2.is_tree_child(net),
    }


def _tool_gates() -> tuple[dict, bool]:
    """``l2reg``'s first two probe gates: the sequence model is v1/v2's, and AliSim runs."""
    gates: dict = {}
    ok = True
    v1 = REPO / "alisim_gt_v1" / "model.json"
    if v1.exists():
        want = json.loads(v1.read_text())
        got = MODEL.as_dict()
        diff = {k: (want[k], got.get(k)) for k in want if str(want[k]) != str(got.get(k))}
        gates["model_matches_alisim_gt_v1"] = {"ok": not diff, "diff": diff}
        ok &= not diff
    else:
        gates["model_matches_alisim_gt_v1"] = {"ok": None, "note": "file absent"}
    with tempfile.TemporaryDirectory(prefix="l3reg-probe-") as td:
        tree = Path(td) / "t.nwk"
        tree.write_text("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);\n")
        try:
            seqsim._run(seqsim.alisim_cmd(tree, Path(td) / "a", 100, MODEL, 1), Path(td))  # noqa: SLF001
            rows = (Path(td) / "a.phy").read_text().split("\n")[0].split()
            alisim_ok = rows[0] == "4"
        except Exception as exc:  # noqa: BLE001
            alisim_ok, rows = False, [str(exc)[:300]]
    gates["alisim_runs"] = {"ok": alisim_ok, "iqtree": seqsim.iqtree_version(),
                            "binary": str(seqsim.IQTREE), "header": rows}
    return gates, bool(ok and alisim_ok)


def stage_probe(seeds: int, jobs: int, l2pkg: Path) -> int:
    """BLOCKING.  Four gates:

    * the tools are v1/v2's;
    * the generator tables are the pinned ones;
    * ``l2reg_n200_v1`` still re-derives byte for byte;
    * every layout places at n = 200.
    """
    out: dict = {"n_taxa": N_TAXA, "seeds_per_cell": seeds, "gates": {}}
    gates, ok = _tool_gates()
    out["gates"].update(gates)

    tables = {str(k): {"n": len(G.generators(k)), "sha256": G.table_sha256(k),
                       "by_sinks": dict(sorted(Counter(g.n_sinks for g in G.generators(k)).items()))}
              for k in (1, 2, 3)}
    tables_ok = ([tables[str(k)]["n"] for k in (1, 2, 3)] == [1, 4, 65]
                 and all(tables[str(k)]["sha256"] == G.PINNED[k] for k in (1, 2, 3)))
    out["gates"]["generator_tables"] = {"ok": tables_ok, **tables}
    ok &= tables_ok

    t0 = time.perf_counter()
    try:
        ident = l2_identity(l2pkg)
    except FileNotFoundError as exc:  # the gate cannot skip: a missing package is a failure
        ident = {"ok": False, "error": str(exc)}
    ident["seconds"] = time.perf_counter() - t0
    out["gates"]["level2_byte_identity"] = ident
    ok &= ident["ok"]
    print(f"[probe] level-2 byte identity: {'PASS' if ident['ok'] else 'FAIL'} "
          f"({ident.get('differ', ident.get('error'))}) in {ident['seconds']:.0f}s", flush=True)

    tasks = [(c, m, 7000 + s) for c in COMBOS for m in (*L3.SINK_TARGETS, "free")
             for s in range(seeds)]
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
               "seconds_max": float(max(r["seconds"] for r in rs)),
               "base_redraws_max": max(r.get("base_draw", -1) for r in rs)}
        for cell, rs in sorted(by_cell.items())
    }
    out["summary"] = summary
    built = [r for r in rows if r["ok"]]
    free = [r for r in built if r["mode"] == "free"]
    out["natural_sinks"] = dict(sorted(Counter(s for r in free for s in r["sinks"]).items()))
    out["natural_generators"] = dict(Counter(g for r in free for g in r["generators"]).most_common())
    out["generators_seen"] = len({g for r in built for g in r["generators"]})
    out["girth_R_distribution"] = dict(sorted(Counter(g for r in built for g in r["girth_R"]).items()))
    out["tree_child_fraction"] = float(np.mean([r["tree_child"] for r in built])) if built else 0.0
    failed = [c for c, s in summary.items() if s["built"] < s["of"]]
    out["gates"]["every_layout_places"] = {"ok": not failed, "failed_cells": failed}
    ok &= not failed
    out["seconds"] = time.perf_counter() - t0
    out["ok"] = bool(ok)
    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_OUT.write_text(json.dumps(out, indent=1) + "\n")
    slow = max(summary.items(), key=lambda kv: kv[1]["seconds_max"])
    print(f"[probe] {len(rows)} builds in {out['seconds']:.0f}s; failed cells: {failed or 'none'}; "
          f"slowest {slow[0]} at {slow[1]['seconds_max']:.0f}s")
    print(f"[probe] natural sinks (free): {out['natural_sinks']}; "
          f"{out['generators_seen']} of 65 generators seen")
    print(f"[probe] girth_R: {out['girth_R_distribution']}  tree-child: "
          f"{out['tree_child_fraction']:.2f}")
    for name, g in out["gates"].items():
        print(f"[probe] {name}: {'PASS' if g['ok'] else ('n/a' if g['ok'] is None else 'FAIL')}")
    print(f"[probe] {'PASS' if ok else 'FAIL'} -> {PROBE_OUT}")
    return 0 if ok else 2


# ------------------------------------------------------------------------------ networks


def network_record(spec: Spec, net: N.Network, placement: dict) -> dict:
    """The ground truth written beside each network: ``l2reg.network_record``'s, at level 3."""
    below = N.hardwired_clusters(net)
    tags = {ret.hybrid: f"H{k + 1}" for k, ret in enumerate(net.retics)}
    blocks = L2.nontrivial_blocks(net)
    blob_of = {h: i for i, b in enumerate(blocks) for h in b.hybrids}
    leaf_times = [net.time[v] for v in net.leaf_taxon]
    ib = N.internal_branch_lengths(net)
    return {
        "key": spec.key, "spec": spec.row(), "n_taxa": N_TAXA, "labels": net.labels,
        "gamma_range": list(GAMMA_RANGE),
        "check": L3.check3(net, spec.layout),
        "blobs": L3.blob_report3(net),
        "reticulations": [{
            "tag": tags[r.hybrid], "blob": blob_of[r.hybrid], "gamma": net.gamma_of(r.hybrid),
            # net.time, not r.time: Reticulation.time is the insertion time, before scaling.
            "time": net.time[r.hybrid],
            "hybrid_cluster": l2reg._labels(net, below[r.hybrid]),  # noqa: SLF001
            "donor_cluster": l2reg._labels(net, below[r.donor]),  # noqa: SLF001
            "tree_parent_cluster": l2reg._labels(net, below[net.tree_parent[r.hybrid]]),  # noqa: SLF001
        } for r in net.retics],
        "root_height": max(net.time),
        "ultrametric_spread": max(leaf_times) - min(leaf_times),
        "median_internal_branch": float(np.median(ib)),
        "n_clusters": len(N.full_clusters(net)),
        "regular_network": L2.regular_json(net),
        "placement": placement,
    }


def _build(spec: Spec):
    return L3.build_level3(N_TAXA, spec.layout, spec.net_seed, cu=spec.cu,
                           gamma_range=GAMMA_RANGE)


def stage_networks(paths: Paths, specs: list[Spec], jobs: int, force: bool = False) -> int:
    """Each network is deterministic in its spec, so they are built in parallel."""
    todo = [s for s in specs
            if force or not (paths.meta(s.key).exists() and paths.net(s.key).exists())]
    for s in specs:
        if s not in todo:
            print(f"[networks] {s.key}: cached")
    if not todo:
        return 0
    from multiprocessing import Pool

    with Pool(max(1, min(jobs, len(todo)))) as pool:
        built = pool.map(_build, todo, chunksize=1)
    for s, (net, placement) in zip(todo, built, strict=True):
        rec = network_record(s, net, placement)
        if not rec["check"]["ok"]:
            raise RuntimeError(f"{s.key}: failed its own gates: {rec['check']}")
        if abs(rec["median_internal_branch"] - s.cu) > 1e-9 or rec["ultrametric_spread"] > 1e-9:
            raise RuntimeError(f"{s.key}: not scaled to cu={s.cu} or not ultrametric")
        _write(paths.net(s.key), N.to_enewick(net) + "\n")
        _write(paths.regular(s.key), N.to_enewick(L2.contract(net)) + "\n")
        _write(paths.pickle(s.key), pickle.dumps(net))
        _write(paths.meta(s.key), json.dumps(rec, indent=1) + "\n")   # last: the cache marker
        print(f"[networks] {s.key}: k={net.n_retics} sinks={rec['check']['sinks']} "
              f"generators={rec['check']['generators']} girth_R>={rec['check']['girth_R_min']} "
              f"{placement['seconds']:.1f}s")
    return 0


# ------------------------------------------------------------------------------ package


def generator_table() -> str:
    """``generators.tsv``: what every generator ID in the ground truth means."""
    head = "id\tk\tn_sinks\tn_vertices\tparallel_arcs\tarcs\n"
    return head + "".join(
        f"{g.id}\t{g.k}\t{g.n_sinks}\t{g.n_vertices}\t{g.parallel_arcs}\t"
        f"{' '.join(f'{u}>{v}' for u, v in g.arcs)}\n"
        for k in (1, 2, 3) for g in G.generators(k))


def _files() -> list[tuple[str, str]]:
    rows = dict(l2reg._files())  # noqa: SLF001
    rows["networks/<key>.json"] = (
        "the ground truth: blobs (generator, sinks, girth in both forms, size), reticulations "
        "(gamma, time, hybrid / donor / tree-parent clusters), R as clusters plus Hasse edges "
        "(`regular_network`), every gate, and the generator's rejection counters")
    out = list(rows.items())
    at = [p for p, _ in out].index("MANIFEST.tsv")
    out.insert(at, ("generators.tsv",
                    "every level-1, -2 and -3 generator: what each ID in `networks/<key>.json` "
                    "means -- its sinks and canonical arcs (vertex 0 the root, then tree "
                    "vertices, reticulations, sinks)"))
    return out


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
            "key": s.key, "rep": s.rep, "n_taxa": N_TAXA, "n_l3": s.n_l3, "n_l1": s.n_l1,
            "k": s.k, "cu": s.cu, "sink_targets": ",".join(t[1:] for t in s.targets),
            "sinks": ",".join(map(str, rec["check"]["sinks"])),
            "generators": ",".join(rec["check"]["generators"]),
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
            "network_sha256": l2reg._sha256(paths.net(s.key)),  # noqa: SLF001
            "genetrees_sha256": l2reg._sha256(paths.trees(s.key)),  # noqa: SLF001
        })
    header = list(rows[0])
    _write(paths.root / "MANIFEST.tsv", "\t".join(header) + "\n" + "".join(
        "\t".join(str(r[h]) for h in header) + "\n" for r in rows))
    _write(paths.root / "generators.tsv", generator_table())

    audit = json.loads((paths.manifests / "audit_verdict.json").read_text())
    code = l2reg._snapshot_code(paths.root / "code")  # noqa: SLF001
    manifest = {
        "name": paths.root.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code,
        "master_seed": MASTER_SEED, "seed_tag": TAG, "level": LEVEL,
        "n_taxa": N_TAXA, "n_networks": N_NETWORKS,
        "n_loci": N_LOCI, "bp": BP, "gamma_range": list(GAMMA_RANGE),
        "min_girth": L3.MIN_GIRTH, "sink_targets": list(L3.SINK_TARGETS),
        "generator_tables": {str(k): {"n": len(G.generators(k)), "sha256": G.table_sha256(k)}
                             for k in (1, 2, 3)},
        "design": [s.row() for s in specs],
        "model": MODEL.as_dict(),
        "tools": {"iqtree": seqsim.iqtree_version(), "iqtree_binary": str(seqsim.IQTREE),
                  **l2reg._julia_versions(), "python": sys.version.split()[0],  # noqa: SLF001
                  "numpy": np.__version__, "tree_qmc": l2reg._tob_tool()},  # noqa: SLF001
        "seed_check": [{"cell_key": s.key, "bp": BP, "seed_alisim": s.est_seeds[0],
                        "seed_iqtree": s.est_seeds[1], "gt_seed": s.gt_seed,
                        "net_seed": s.net_seed} for s in specs[:4]],
        "gates": {
            "python_check_ok": all(r["check"]["ok"] for r in recs.values()),
            "phylonetworks_audit_ok": not any(audit.values()),
            "generator_tables_pinned": all(G.table_sha256(k) == G.PINNED[k] for k in (1, 2, 3)),
            "iqtree_model_used": sorted({v["iqtree_model_used"] for v in sides.values()}),
            "tob_exact": sum(t["fn"] == t["fp"] == 0 for t in tobs.values()),
        },
    }
    _write(paths.root / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    _write(paths.root / "README.md", readme(specs, recs, sides, tobs, manifest))

    files = sorted(p for p in paths.root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"
                   and not p.name.endswith((".tmp", ".part")))
    _write(paths.root / "SHA256SUMS", "".join(
        f"{l2reg._sha256(p)}  {p.relative_to(paths.root)}\n" for p in files))  # noqa: SLF001
    print(f"[package] {len(rows)} networks, {len(files)} files checksummed -> {paths.root}")
    return 0


def readme(specs, recs, sides, tobs, manifest) -> str:
    sinks = Counter(s for r in recs.values() for s in r["check"]["sinks"])
    gens = Counter(g for r in recs.values() for g in r["check"]["generators"])
    blobs = [b for r in recs.values() for b in r["blobs"]]
    n_l3 = sum(b["n_retics"] == 3 for b in blobs)
    girths = Counter(b["girth_R"] for b in blobs)
    spans = np.array([b["root_taxa"] for b in blobs])
    tc = sum(r["check"]["regularity"]["tree_child"] for r in recs.values())
    width = np.mean([s["aln_len_mean"] for s in sides.values()])
    gaps = np.mean([s["gap_fraction"] for s in sides.values()])
    rf = np.mean([s["rf_norm_mean"] for s in sides.values()])
    table3 = G.generators(3)
    by_sinks = Counter(g.n_sinks for g in table3)
    design_rows = "\n".join(
        f"| `{s.key}` | {s.n_l3} | {s.n_l1} | {s.k} | {s.cu} | "
        f"{', '.join(t[1:] for t in s.targets)} | "
        f"{', '.join(recs[s.key]['check']['generators'])} | "
        f"{recs[s.key]['check']['girth_R_min']} |"
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
    files_table = "\n".join(["| path | what it is |", "|---|---|"]
                            + [f"| `{p}` | {d} |" for p, d in _files()])
    return f"""# {manifest['name']} -- regular, exactly-level-3 networks at n = {N_TAXA}

Built {manifest['built_utc']} by `python -m opc.l3reg` from rl-net-opt at `{code['commit'][:12]}`
{"(with uncommitted changes -- the exact code is in `code/`)" if code['dirty'] else ""}.

{N_NETWORKS} phylogenetic networks on {N_TAXA} taxa (`t1`..`t{N_TAXA}`). For each network,
{N_LOCI:,} gene trees simulated under the network multispecies coalescent, and for each gene tree
a {BP} bp AliSim alignment and an IQ-TREE re-estimate. It is `l2reg_n200_v1` at level 3:
everything downstream of the networks is built exactly as there.

## Every network is

* **regular** -- the Hasse diagram of its hardwired clusters *is* the network, so it is
  reconstructible from its clusters without ambiguity. Checked by definition, not by a
  sufficient condition: the cluster map on the regular form is injective, and its edges, read as
  cluster pairs, are exactly the cover relation of the cluster set.
* **exactly level-3** -- every blob (biconnected component) holds at most three reticulations
  and at least one holds three.
* of **girth >= {L3.MIN_GIRTH}** -- no undirected cycle has fewer than {L3.MIN_GIRTH} edges,
  measured on the regular form, which is the stricter of the two forms below.

All three are checked in Python (`networks/<key>.json`, `check`) and again by an independent
PhyloNetworks parse of every written file (`manifests/audit.json`, `audit_verdict.json`):
clusters, per-hybrid gammas and which edge carries the `#H` copy, level, per-blob reticulation
counts, hybrid ladders and girth all agree.

## Files

{files_table}

## Design

Each (level-3 blobs, level-1 blobs) pair in {{1,2,3}}^2 once per ILS level, and (2, 2) once more,
so k = 3a + b runs 4-12. ILS is the median internal branch, in coalescent units. Each level-3
blob was built to a target number of **sinks** -- reticulations whose child leaves the blob --
rotated 1, 2, 3 within each ILS level; which generator it is within that stratum is whatever
placement produced. Every network is also tree-child here ({tc} of {N_NETWORKS}), though only
regularity was required.

| key | level-3 blobs | level-1 blobs | k | cu | sinks | generators | girth |
|---|---|---|---|---|---|---|---|
{design_rows}

## Level-3 blob shapes

A blob with k reticulations, its in-1/out-1 vertices suppressed, is a **level-k generator**: a
biconnected acyclic multigraph with one root, tree vertices of in-degree 1 and out-degree 2, and
k reticulations of in-degree 2 and out-degree at most 1 -- a *sink* when it is 0. There are
{len(table3)} level-3 generators, {by_sinks[1]} with one sink, {by_sinks[2]} with two and
{by_sinks[3]} with three (and 4 at level 2, which are `l2reg_n200_v1`'s S0-S3). `generators.tsv`
lists every one. The IDs `G3s<sinks>-<i>` are this code's canonical order, pinned by SHA-256 in
`manifest.json` -- not a numbering from the literature.

Realised, over the {n_l3} level-3 blobs: sinks {dict(sorted(sinks.items()))}, and
{len(gens)} distinct generators, the most common {', '.join(f'{g} ({n})' for g, n in gens.most_common(5))}.
Girth on R, over all {len(blobs)} blobs: {dict(sorted(girths.items()))}. Blob size (taxa below
the blob's root): median {int(np.median(spans))}, range {spans.min()}-{spans.max()} -- no ceiling
was imposed, so some blobs span most of the tree.

## Sequence model

`{MODEL.sim_model_string()}` + indels `{MODEL.indel_ins:g},{MODEL.indel_del:g}`, size
`{MODEL.indel_size}`, branch scale `{MODEL.branch_scale}` substitutions/site per coalescent unit
(theta = {MODEL.theta:g}), inferred under `{MODEL.inference_model}` with `--fast --keep-ident` --
identical to `alisim_gt_v1/v2` and `l2reg_n200_v1`. `{BP}` is the **root** sequence length:
indels make realised alignments wider (mean {width:.0f} columns here, {gaps:.0%} gaps). The
outgroup is grafted onto each gene tree at `(1 + {MODEL.outgroup_alpha}) x p99` of that network's
root heights, so it contributes no rooting error. Mean normalised RF of the estimates against the
true trees: {rf:.3f}. IQ-TREE: `{manifest['tools']['iqtree']}`.

## Tree of blobs

`tob_bp{BP}/` holds one estimated tree of blobs per network: TREE-QMC {tq['version']}
(`{tq['commit'][:7]}`) run as `tree-qmc {' '.join(tq['args'])}` on the network's {N_LOCI:,}
estimated gene trees, `OUT` included -- default `alpha` / `beta` and seeds, as `l2reg_n200_v1`.
Each run is single-threaded, {min(t_hours):.1f}-{max(t_hours):.1f} h.

**`<key>.tob` is not TREE-QMC's output verbatim.** TREE-QMC tests a binary root's branch on the
root's first child only and contracts by splicing nodes into their parents, so when its own test
rejects that branch the second child's edge survives: one edge its own rule removed stays in the
output. `<key>.tob` applies TREE-QMC's rule -- contract iff `qtt_p < alpha` or `qst_p > beta` --
to the p-values it logged for every branch (`<key>.pvalues.tob`), and is checked to equal
TREE-QMC's output (`<key>.raw.tob`) everywhere except exactly that branch. That branch was
present in {t_fixed} of {len(tobs)} raw outputs.

Against the true tree of blobs -- one edge per bridge of the network, `true_tob` in each JSON --
{t_exact} of {len(tobs)} are exactly right ({t_exact_raw} of TREE-QMC's raw outputs), and over
all {len(tobs)}, {t_fn} of {t_true} true edges are missed and {t_fp} are extra. An edge is
compared as the cluster below it, after rooting on `OUT`.

| key | blobs | true edges | estimated edges | missed | extra | root fixed |
|---|---|---|---|---|---|---|
{tob_rows}

## Reproducibility

Seeds are SHA-256 chains off master seed `{MASTER_SEED}`: networks
`(master, "{TAG}", "net", rep)`, gene trees `(master, "{TAG}", "gt", key)`, AliSim / IQ-TREE
`(master, "alisim" | "iqtree", key, bp)` plus the global locus / batch index.
`python -m opc.l3reg all` rebuilds everything; each stage skips what already exists.
"""


# ------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opc.l3reg", description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("probe", "networks", "genetrees", "pilot", "estimate",
                                      "tob", "package", "all"))
    ap.add_argument("--pkg", type=Path, default=PKG)
    ap.add_argument("--l2pkg", type=Path, default=l2reg.PKG,
                    help="probe: the level-2 dataset that must still re-derive byte for byte")
    ap.add_argument("--jobs", type=int, default=min(48, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()) / "l3reg")
    ap.add_argument("--chunks", type=int, default=5, help="estimation tasks per network")
    ap.add_argument("--seeds", type=int, default=10, help="probe: seeds per (layout, target)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="networks: rebuild even if present")
    args = ap.parse_args(argv)
    if args.jobs >= 100:
        print("refusing --jobs >= 100: a pool that large deadlocks silently under the shared "
              "RLIMIT_NPROC (CLAUDE.md)")
        return 2
    paths, specs = Paths(args.pkg), design()
    stages = {
        "probe": lambda: stage_probe(args.seeds, args.jobs, args.l2pkg),
        "networks": lambda: stage_networks(paths, specs, args.jobs, args.force),
        "genetrees": lambda: l2reg.stage_genetrees(paths, specs, args.jobs, level=LEVEL),
        "pilot": lambda: l2reg.stage_pilot(paths, specs, args.scratch),
        "estimate": lambda: l2reg.stage_estimate(paths, specs, args.jobs, args.scratch,
                                                 args.chunks, args.dry_run),
        "tob": lambda: l2reg.stage_tob(paths, specs, args.jobs, args.scratch),
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
