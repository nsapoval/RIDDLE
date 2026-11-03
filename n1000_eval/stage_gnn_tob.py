#!/usr/bin/env python3
"""Collapse the selF1 GNN network of each cell to its tree of blobs: the resolvers' input.

This dataset has no TREE-QMC tree of blobs at n = 1000. The only inferred TOB is the one our own
network implies. For each cell of the GNN run (``common.GNN_RUN``), take the ``_l2mix_pair`` seed-0
model's network and do three things:

1. Contract every non-bridge edge (``common.tree_of_blobs_general``). Each blob, of any level,
   becomes one node. ``camus_nets.tree_of_blobs`` is level-1 only, so it is used here only as a
   cross-check.
2. Write ``tob/<key>.gnn.rooted.nwk = (OUT,<tob>);``. That is the shape l2reg's resolvers were fed,
   and the gene trees carry OUT.
3. Record ``tob/gnn_tob.json``: the cell's sha256, the TOB's sha256, blob counts and degrees, and
   ``k_pred``. ``run_resolvers.py`` refuses a ``gnn`` task whose GNN cell has changed since.

**The TOB inherits the GNN's budget.** The GNN decoded with ``k_true`` as its budget, so this TOB
indirectly knows k. That is the same caveat ``prefix_tob_eval``'s GNN-TOB arm carries.

**Not every polytomy is a blob.** The GNN's network is not binary: its selection holds fewer
clusters than a binary tree, so its tree part keeps multifurcations. Contracting the blobs leaves
those in place, and a resolver cannot tell them from blobs. Both resolvers turn every eligible
polytomy into one reticulation, so they over-reticulate here by construction. That is
``prefix_tob_eval``'s finding for its GNN-TOB arm, and this keeps its protocol: the TOB is fed as
is. ``gnn_tob.json`` records the two kinds of polytomy separately (``blob_polytomy_degrees``,
``tree_polytomy_degrees``), so a result can be read against how many eligible polytomies are real
blobs.

**Which blobs a resolver can see.** A contracted blob with two outgoing edges, such as a triangle,
is a binary node and is invisible to both resolvers. Of the others:

* BROOQS resolves a node with >= 3 children.
* NetCS needs >= 4 children.

Both are non-root thresholds. The TOB is fed as ``(OUT,<tob>)``, so its root has two children and
is never eligible; see ``pte_common``'s eligibility note.

``n_eligible_*`` records both counts.

Gates (exit 1, and nothing is written for any network if one fails):

    taxa        the TOB's leaves are exactly the truth's taxa
    subset      every TOB cluster is a hardwired cluster of the GNN network
    round-trip  the clusters scanned off the written string equal the bridge clusters
    degrees     the TOB's polytomy degrees equal those computed independently: each blob's
                out-degree, plus each tree-part multifurcation of the network
    level 1     where the GNN network is level 1, the TOB equals ``camus_nets.tree_of_blobs``.
                The count is printed, so a pass with zero such networks shows as one.

Usage:  gnn_cpu_v1/.venv/bin/python n1000_eval/stage_gnn_tob.py [--run DIR] [--keys 0] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import common as C

from opc.rl import camus_nets as CN  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def polytomies(rooted: str) -> list[int]:
    _parent, _len, _lab, children = OG.parse_newick(rooted)
    return sorted((len(c) for c in children if len(c) > 2), reverse=True)


def blob_out_degrees(net) -> tuple[list[int], list[int]]:
    """The TOB's polytomies, predicted independently of the bridge-cluster route.

    There are two kinds, returned separately, each as a descending list of child counts above 2:

    * **blob** polytomies: for each biconnected component with a cycle, the number of edges
      leaving it downward. That is the child count of the node the blob contracts to.
    * **tree** polytomies: multifurcations of the network's own tree part, at a node in no blob.
      A GNN selection holds fewer clusters than a binary tree, so its network is not binary, and
      every such node survives into the tree of blobs looking exactly like a blob.
    """
    import networkx as nx
    G, doubled = C.pte()._undirected(net)                               # noqa: SLF001
    comps = [set(c) for c in nx.biconnected_components(G) if len(c) > 2]
    comps += [set(e) for e in doubled]
    in_blob = set().union(*comps) if comps else set()
    blob = [d for comp in comps
            if (d := sum(1 for v in comp for c in net.net_children(v) if c not in comp)) > 2]
    tree = [d for v in G.nodes if v not in in_blob and (d := len(net.net_children(v))) > 2]
    return sorted(blob, reverse=True), sorted(tree, reverse=True)


def eligible(degrees: list[int], *, threshold: int) -> int:
    """Polytomies a resolver turns into a reticulation (``pte_common``'s eligibility rule).

    The TOB is fed as ``(OUT,<tob>)``. Its root has two children, so it is never eligible, and
    every blob is read as a non-root node: BROOQS needs >= 3 children, NetCS >= 4."""
    return sum(1 for d in degrees if d >= threshold)


def collapse(cell_path: Path, key: str, truth_taxa: set[str]) -> tuple[list[str], dict]:
    raw = cell_path.read_bytes()
    cell = json.loads(raw)
    tag, seed, dec = C.TOB_SOURCE
    if C.cell_decoder(cell) != dec:
        return [f"{cell_path.name}: decoded as {C.cell_decoder(cell)}, need {dec}"], {}
    ms = [m for m in cell["models"] if m["tag"] == tag and int(m["seed"]) == seed]
    if len(ms) != 1:
        return [f"{cell_path.name}: {len(ms)} models with tag {tag} seed {seed}"], {}
    m = ms[0]
    if not m.get("enewick") or int(m["k_pred"]) < 0:
        return [f"{cell_path.name}: {tag} emitted no network"], {}
    net = N.from_enewick(m["enewick"])
    bad = []
    if set(net.labels) != truth_taxa:
        bad.append(f"taxa: {len(set(net.labels) ^ truth_taxa)} labels differ from the truth's")
    clusters = C.pte().bridge_tob_clusters(net)
    tob = C.clusters_to_newick(clusters, net.labels)
    stray = {c for c in clusters if len(c) < len(net.labels)} - CN.label_clusters(net)
    if stray:
        bad.append(f"subset: {len(stray)} TOB clusters are not hardwired clusters of the network")
    scanned, leaves = C.pte().string_clusters(tob)
    if scanned != {c for c in clusters if len(c) < len(net.labels)} or leaves != set(net.labels):
        bad.append("round-trip: the written TOB's clusters differ from the bridge clusters")
    degs = polytomies(tob)
    blob_degs, tree_degs = blob_out_degrees(net)
    if degs != sorted(blob_degs + tree_degs, reverse=True):
        bad.append(f"degrees: TOB polytomies {degs}; blob out-degrees {blob_degs} + tree "
                   f"multifurcations {tree_degs}")
    level = N.level(net)
    l1_checked = False
    if level <= 1:
        ref, _ = C.pte().string_clusters(CN.tree_of_blobs(net))
        if ref != scanned:
            bad.append(f"level 1: tree_of_blobs differs on {len(ref ^ scanned)} clusters")
        l1_checked = True
    rooted = f"({C.OUT},{tob.rstrip(';')});"
    _p, _l, _lab, children = OG.parse_newick(tob)
    root_deg = len(children[0])
    # ``key`` is the dataset's (stage.json); the cell's own ``key`` is CamusCell's ``n1000_repRR``.
    rec = {"key": key, "rep": int(cell["rep"]), "cell": cell_path.name,
           "gnn_cell_sha256": sha(raw), "gnn_enewick_sha256": sha(m["enewick"]),
           "tag": tag, "seed": seed, "decoder": dec, "k_pred": int(m["k_pred"]),
           "gnn_level": level, "level1_cross_checked": l1_checked,
           "n_blobs": len(degs), "n_cycle_components": C.pte().n_cycle_components(net),
           "blob_degrees": degs, "root_degree": root_deg,
           "blob_polytomy_degrees": blob_degs, "tree_polytomy_degrees": tree_degs,
           "n_eligible_brooqs": eligible(degs, threshold=3),
           "n_eligible_netcs": eligible(degs, threshold=4),
           "n_eligible_brooqs_blob": eligible(blob_degs, threshold=3),
           "n_eligible_netcs_blob": eligible(blob_degs, threshold=4),
           "tob_rooted": rooted, "tob_sha256": sha(rooted + "\n")}
    return bad, rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=C.RUNS / C.GNN_RUN, help="the GNN run directory")
    ap.add_argument("--keys", default="", help="reps, e.g. 0-4; default every cell of the run")
    ap.add_argument("--out", type=Path, default=C.TOB, help="where the TOBs and gnn_tob.json go")
    ap.add_argument("--allow-partial", action="store_true",
                    help="stage the cells that exist even if the run is incomplete")
    args = ap.parse_args()

    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run n1000_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    want = C.parse_reps(args.keys)
    cells = [c for c in stage["cells"] if want is None or c["rep"] in want]
    t0 = time.perf_counter()
    bad, recs, absent = [], [], []
    for c in cells:
        f = args.run / "cells" / f"{C.cell_name(C.PHASE, c['rep'], C.ARM)}.json"
        if not f.exists():
            absent.append(f.name)
            continue
        truth = C.truth_of(c["rep"])
        b, rec = collapse(f, c["key"], set(truth.labels))
        bad += [f"{c['key']}: {x}" for x in b]
        if not b:
            recs.append(rec)
            print(f"  {c['key']:30s} k={c['k']} k_pred={rec['k_pred']} level={rec['gnn_level']} "
                  f"blobs={rec['n_blobs']} (cycle comps {rec['n_cycle_components']}) "
                  f"polytomies: blob {rec['blob_polytomy_degrees']} tree "
                  f"{rec['tree_polytomy_degrees']} | eligible brooqs/netcs "
                  f"{rec['n_eligible_brooqs']}/{rec['n_eligible_netcs']} (of which blobs "
                  f"{rec['n_eligible_brooqs_blob']}/{rec['n_eligible_netcs_blob']})", flush=True)
    if absent:
        msg = f"{len(absent)} of {len(cells)} GNN cells missing, e.g. {absent[:2]}"
        if not args.allow_partial:
            raise SystemExit(f"[gnn-tob] {msg}: finish the GNN grid, or --allow-partial")
        print(f"[gnn-tob] {msg} (--allow-partial)")
    if bad:
        print("[gnn-tob] FAILED; nothing written:\n" + "\n".join(f"  {b}" for b in bad))
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    index_p = args.out / "gnn_tob.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {"cells": {}}
    for rec in recs:
        (args.out / f"{rec['key']}.gnn.rooted.nwk").write_text(rec["tob_rooted"] + "\n")
        index["cells"][rec["key"]] = {k: v for k, v in rec.items() if k != "tob_rooted"}
    index.update(built=time.strftime("%Y-%m-%dT%H:%M:%S%z"), run=str(args.run),
                 source={"tag": C.TOB_SOURCE[0], "seed": C.TOB_SOURCE[1],
                         "decoder": C.TOB_SOURCE[2]})
    tmp = index_p.with_name(index_p.name + ".tmp")
    tmp.write_text(json.dumps(index, indent=1) + "\n")
    tmp.replace(index_p)
    n_l1 = sum(r["level1_cross_checked"] for r in recs)
    print(f"[gnn-tob] OK: {len(recs)} TOBs -> {args.out}; level-1 cross-check ran on {n_l1} of "
          f"{len(recs)} (the rest are level 2, where only the bridge route is valid) | "
          f"{time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
