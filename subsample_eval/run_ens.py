#!/usr/bin/env python3
"""Ensembles over the five 500-tree draws of each cell: pooled scores, and assembly votes.

Two phases, both resumable per cell (see README.md, "Ensembles over the draws"):

``score``   For every cell and draw 0-4, re-run the draw exactly as ``run_sub.sh`` did --
            ``run_gnn_cpu.run_cell``'s stages with ``--m-from-trees`` (m = 500) on
            ``inputs/<ds>/d<k>/`` -- then one forward per checkpoint, and every decode the landed
            runs made: both checkpoints and greedy, under level1 and the dataset's relaxed decoder.
            Writes the score cache ``<runs-root>/<ds>/ens/scores/<cell>.{npz,json}``.

            **Gate P1 (exit 3):** each replay reproduces its landed cell under
            ``runs/<ds>/d<k>/{gnn_t1_l2mix_pair,greedy}{,_relaxedL<L>}/`` -- ``n_clusters, m,
            n_loci`` (and ``n_edges`` for the GNN), and per model ``n_selected, tp, fp, fn,
            fn_reachable, k_pred, enewick`` plus ``decode_stats`` when relaxed.  20 GNN and 10
            greedy comparisons per cell.  The forward is ``run_gnn_cpu.forward``'s pair branch with
            the logit kept; its ``p`` is asserted bit-equal to ``forward()``'s on every worker's
            first draw.

``decode``  No model: the cache alone.  For every cell, source (selF1, selHA, greedy) and decoder,
            the five draws are merged (``gnn_cpu_v1/ensemble.py``) and decoded once per
            aggregator -- ``pool``, ``pool_present`` and ``vote`` for the checkpoints, ``vote`` for
            greedy (the greedy-vote control).  Each is written as a ``run_gnn_cpu``-format cell
            JSON with an ``ensemble`` block to
            ``<runs-root>/<ds>/ens/<agg>/{gnn_t1_l2mix_pair,greedy}{,_relaxedL<L>}/cells/``.

            **G-identity (exit 1):** at D = 1, for *each* draw, every aggregator reproduces that
            draw's own selection (as a set of cluster words), on both decoders and for greedy.
            **G-F1 (exit 1):** every ensemble's tp/fp/fn equal ``camus_nets.cluster_counts`` on
            label sets.  **The sweep:** pool and vote over draws 0..D-1 for D = 1-5 (F1, hybrid
            ancestry, k_pred; no jar), whose D = 5 row must be the main decode.  **Reported:**
            vacuity (union larger than every draw; ensemble unlike every draw; the network equal
            to the full-data one) and order sensitivity (``pool`` with the draws reversed).

Usage (``gnn_cpu_v1/.venv/bin/python``, from the repo root)
-----
    python subsample_eval/run_ens.py score  --dataset l2reg --dry-run
    python subsample_eval/run_ens.py score  --dataset l2reg --jobs 5
    python subsample_eval/run_ens.py decode --dataset l2reg --jobs 5
    python subsample_eval/run_ens.py decode --dataset l2reg --summary-only
    python subsample_eval/run_ens.py score  --dataset camus --phases cam50 --reps 0 \\
        --runs-root /tmp/smoke/runs                                 # a smoke; landed runs are read
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import common as C

if str(C.GNN) not in sys.path:           # run_gnn_cpu and ensemble; pylib stays at sys.path[0]
    sys.path.insert(1, str(C.GNN))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ensemble as E  # noqa: E402
import make_sub_csv as MS  # noqa: E402
import run_gnn_cpu as R  # noqa: E402
from opc import bitset  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl.env import OverlapCSR  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.solve import candidates  # noqa: E402
from opc.solve import truth as T  # noqa: E402

#: The checkpoints, as ``run_sub.sh`` ran them: (tag as a cell JSON records it, stem).
MODELS = (("_l2mix_pair", "best"), ("_l2mix_pair:best_ha", "best_ha"))
GREEDY = "greedy"
SOURCES = (*(t for t, _ in MODELS), GREEDY)
#: Safe names for the NPZ keys.
SHORT = {"_l2mix_pair": "selF1", "_l2mix_pair:best_ha": "selHA", GREEDY: "greedy"}
ENS_DATASETS = ("camus", "l2reg", "l3reg")
PARITY_MODEL = ("n_selected", "tp", "fp", "fn", "fn_reachable", "k_pred", "enewick")
SWEEP_AGGS = ("pool", "vote")


def aggs_for(source: str) -> tuple[str, ...]:
    return ("vote",) if source == GREEDY else E.AGGREGATORS


def run_name(source: str, decoder: str) -> str:
    base = C.GREEDY_RUN if source == GREEDY else C.GNN_RUN
    return base + ("" if decoder == "level1" else "_relaxedL" + decoder.rsplit("_L", 1)[1])


def cells_of(ds: C.Dataset, phases: str, reps: str) -> list[dict]:
    want_p = {p for p in phases.split(",") if p}
    want_r = MS.parse_reps(reps)
    return [r for r in C.read_tsv(ds.cells)
            if (not want_p or r["phase"] in want_p) and (want_r is None or int(r["rep"]) in want_r)]


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable))
    tmp.replace(path)


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def decoder_args(tag: str) -> tuple[str, int | None]:
    return ("level1", None) if tag == "level1" else ("relaxed", int(tag.rsplit("_L", 1)[1]))


def scorer_of(words: np.ndarray, truth) -> SimpleNamespace:
    """``run_gnn_cpu.run_cell``'s scorer: ``Instance.cluster_counts`` reads these two fields only."""
    return SimpleNamespace(truth_rows=T.true_cluster_rows(words, truth),
                           n_true=len(NW.nontrivial_clusters(truth)))


# =========================================================================================== score

_W: dict = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    _W["models"] = [m for tag, stem in MODELS
                    for m in R.load_models(C.GNN / "checkpoints", ["_l2mix_pair"], [0], stem)]
    got = [m.tag for m in _W["models"]]
    if got != [t for t, _ in MODELS]:
        raise SystemExit(f"loaded checkpoints {got}, expected {[t for t, _ in MODELS]}")
    _W["checked_forward"] = False


def replay_draw(ds: C.Dataset, row: dict, d: int, work_root: Path) -> dict:
    """``run_gnn_cpu.run_cell``'s shared stages under ``--m-from-trees``, on draw ``d``'s inputs.

    A copy of ``l2mix_rank/scoring.build_cell`` with ``m = min(M, n_loci)`` in its five places;
    gate P1 is what licenses the copy.  Taxon order and ``k`` come from the truth.
    """
    phase, rep, arm = row["phase"], int(row["rep"]), row["arm"]
    cell = CAMUS.CamusCell(phase, rep, arm, ds.draw_inputs(d))
    truth = CAMUS.truth_of(cell)
    taxa = list(truth.labels)
    k = int(truth.n_retics)
    taxon_index = {lab: i for i, lab in enumerate(taxa)}
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{phase}_rep{rep:02d}_d{d}_", dir=work_root))
    gz, table = work / "trees.nwk.gz", work / "clusters.parquet"
    t0 = time.perf_counter()
    try:
        info = CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=R.M)
        m = min(R.M, int(info["n_loci"]))
        counter = extract.count_newick_file(gz, taxon_index, [m])
        masks, support, n_raw = counter.finalize(min_support=1)
        extract.write_cluster_table(table, masks, support, [m], len(taxa), n_raw,
                                    source=str(cell.trees_path), arm=arm)
        words, sup, meta = extract.read_cluster_table(table)
        gs = candidates.build(words, sup[:, meta["checkpoints"].index(m)], len(taxa), m=m, tau=R.TAU)
        graph = OverlapCSR.of(gs)
        xn = I.node_features(gs, graph, I._tree_stats(gz, taxon_index, gs, m))
        xg = I.global_features(gs, graph, k)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"cell": cell, "truth": truth, "taxa": taxa, "k": k, "n_loci": int(info["n_loci"]),
            "m": m, "gs": gs, "graph": graph, "xn": xn, "xg": xg, "prep": time.perf_counter() - t0}


def model_scores(net, xn, xg, graph, gs) -> tuple[np.ndarray, np.ndarray]:
    """``(logits float64, p float32)``: ``run_gnn_cpu.forward``, keeping the logit.

    The pair branch is forward's own (``l2mix_rank.scoring.model_scores`` omits the words, n_taxa
    and m a ``conv: pair`` encoder needs).  ``p`` is forward's exact operations -- float32 sigmoid,
    then ``.float()`` -- and float32 -> float64 is exact, so the logit costs nothing.
    """
    pairs = net.cfg.layers and getattr(P.CONVS.get(net.cfg.conv), "needs_pairs", False)
    with torch.no_grad():
        g = (P.graph_tensors(graph, gs.support, "cpu", words=gs.words, n_taxa=int(gs.n_taxa),
                             m=float(gs.m))
             if pairs else P.graph_tensors(graph, gs.support, "cpu"))
        enc = net.encode(torch.as_tensor(xn), torch.as_tensor(xg), g)
        z = net.cluster_logits(enc)
        p = torch.sigmoid(z).float().numpy()
    if z.dtype != torch.float32:
        raise AssertionError(f"cluster_logits returned {z.dtype}; the exactness argument assumes float32")
    return z.double().numpy(), p


def landed_cell(ds: C.Dataset, d: int, source: str, dec: str, name: str, root: Path):
    kind = "greedy" if source == GREEDY else "gnn"
    f = ds.run_dir(d, kind, dec, root) / "cells" / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def parity(landed: dict | None, got_cell: dict, got_model: dict, tag: str, relaxed: bool) -> list[str]:
    """The fields gate P1 compares; empty when everything reproduces."""
    if landed is None:
        return ["no landed cell"]
    diffs = [f for f in got_cell if landed.get(f) != got_cell[f]]
    ms = [m for m in landed["models"] if m["tag"] == tag]
    if len(ms) != 1:
        return diffs + [f"{len(ms)} landed entries for {tag}"]
    m = ms[0]
    diffs += [f for f in PARITY_MODEL if m.get(f) != got_model[f]]
    if relaxed and m.get("decode_stats") != got_model["decode_stats"]:
        diffs.append("decode_stats")
    return diffs


def score_cell(ds_name: str, row: dict, runs_root: str, landed_root: str) -> dict:
    ds = C.DATASETS[ds_name]
    runs_root, landed_root = Path(runs_root), Path(landed_root)
    if "models" not in _W:
        _init_worker()
    name = C.cell_name(row["phase"], row["rep"], row["arm"])
    out_dir = runs_root / ds.name / "ens" / "scores"
    t_cell = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    meta = {"cell": name, "phase": row["phase"], "rep": int(row["rep"]), "arm": row["arm"],
            "dataset": ds.name, "draws": [], "parity": {}, "problems": []}
    taxa0 = None
    for d in range(ds.draws):
        rd = replay_draw(ds, row, d, runs_root / ds.name / "ens" / "work")
        gs, graph, truth = rd["gs"], rd["graph"], rd["truth"]
        if taxa0 is None:
            taxa0 = rd["taxa"]
            meta.update(key=rd["cell"].key, n=len(taxa0), k=rd["k"], taxa=taxa0,
                        trees=rd["cell"].trees_path.name)
        elif rd["taxa"] != taxa0:
            raise AssertionError(f"{name} d{d}: the truth's taxon order differs between draws")
        scorer = scorer_of(gs.words, truth)
        arrays[f"d{d}_words"] = np.ascontiguousarray(gs.words, dtype=np.uint64)
        arrays[f"d{d}_support"] = np.asarray(gs.support, dtype=np.int64)
        dmeta = {"draw": d, "n_loci": rd["n_loci"], "m": rd["m"], "n_clusters": int(gs.n_clusters),
                 "n_edges": int(graph.nbr.size), "prep": rd["prep"], "forward": {}, "decode": {},
                 "counts": {}}
        decoders = {tag: E.Decoder(len(taxa0), taxa0, rd["k"], *decoder_args(tag), gs=gs,
                                   graph=graph if tag == "level1" else None)
                    for tag in ds.decoders}
        scores = {GREEDY: np.asarray(gs.support, dtype=np.float64)}
        for mdl in _W["models"]:
            t0 = time.perf_counter()
            z, p = model_scores(mdl.net, rd["xn"], rd["xg"], graph, gs)
            dmeta["forward"][mdl.tag] = time.perf_counter() - t0
            if not _W["checked_forward"]:
                ref = R.forward(mdl.net, rd["xn"], rd["xg"], graph, gs)
                if not np.array_equal(ref, p):
                    raise AssertionError(f"{name} d{d} {mdl.tag}: model_scores' p differs from "
                                         f"run_gnn_cpu.forward's")
            arrays[f"d{d}_z_{SHORT[mdl.tag]}"] = z
            arrays[f"d{d}_p_{SHORT[mdl.tag]}"] = p
            scores[mdl.tag] = p
        _W["checked_forward"] = True
        for source in SOURCES:
            for tag, dec in decoders.items():
                t0 = time.perf_counter()
                sel, net, stats = dec.run(scores[source])
                dmeta["decode"][f"{source}|{tag}"] = time.perf_counter() - t0
                arrays[f"d{d}_sel_{SHORT[source]}_{tag}"] = sel
                counts = I.Instance.cluster_counts(scorer, sel)
                got_model = {"n_selected": int(sel.size), **counts, **E.network_fields(net),
                             "decode_stats": stats}
                dmeta["counts"][f"{source}|{tag}"] = {kk: got_model[kk] for kk in (
                    "n_selected", "tp", "fp", "fn", "f1", "k_pred")}
                got_cell = {"n_clusters": int(gs.n_clusters), "m": rd["m"], "n_loci": rd["n_loci"]}
                if source != GREEDY:
                    got_cell["n_edges"] = int(graph.nbr.size)
                diffs = parity(landed_cell(ds, d, source, tag, name, landed_root), got_cell,
                               got_model, source, dec.relaxed)
                meta["parity"][f"d{d}|{source}|{tag}"] = diffs
                if diffs:
                    meta["problems"].append(f"P1 {name} d{d} {source} {tag}: {', '.join(diffs)}")
        meta["draws"].append(dmeta)
    meta["seconds"] = time.perf_counter() - t_cell
    meta["parity_ok"] = not meta["problems"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{name}.npz"
    tmp = out_dir / f"{name}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(npz)
    atomic_json(out_dir / f"{name}.json", meta)     # written last: its presence means "done"
    return meta


# ========================================================================================== decode

def load_draws(ds: C.Dataset, path: Path) -> list[E.Draw]:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    draws = []
    for d in range(ds.draws):
        dr = E.Draw(words=a[f"d{d}_words"], support=a[f"d{d}_support"])
        dr.scores[GREEDY] = a[f"d{d}_support"].astype(np.float64)
        for tag, _ in MODELS:
            dr.scores[tag] = a[f"d{d}_p_{SHORT[tag]}"]
        for source in SOURCES:
            for dec in ds.decoders:
                dr.sel[(source, dec)] = a[f"d{d}_sel_{SHORT[source]}_{dec}"]
        draws.append(dr)
    return draws


def label_sets(words: np.ndarray, rows, taxa: list[str]) -> set[frozenset[str]]:
    n = len(taxa)
    return {frozenset(taxa[i] for i in bitset.to_indices(words[int(r)], n)) for r in rows}


def net_labels(enewick: str) -> tuple[set, frozenset]:
    """``(label clusters, hybrid ancestry)`` of an eNewick, parsed as ``make_sub_csv`` parses one."""
    net = NW.from_enewick(MS.strip_gamma(enewick))
    return CN.label_clusters(net), MS.hybrid_ancestry(net)


def full_networks(ds: C.Dataset, name: str) -> dict:
    """``(source, decoder) -> label clusters`` of the landed full-data network, where one exists."""
    out = {}
    for dec in ds.decoders:
        for kind, sources in (("gnn", [t for t, _ in MODELS]), ("greedy", [GREEDY])):
            f = ds.run_dir("full", kind, dec) / "cells" / f"{name}.json"
            if not f.exists():
                continue
            by_tag = {m["tag"]: m for m in json.loads(f.read_text())["models"]}
            for s in sources:
                m = by_tag.get(s)
                if m and m.get("enewick"):
                    out[(s, dec)] = net_labels(m["enewick"])[0]
    return out


def decode_one(u: E.Union, dec: E.Decoder, agg: str, source: str, truth_scorer, truth_labels,
               truth_ha, taxa) -> dict:
    """One ensemble decode, scored: the model entry a cell JSON carries, plus the checks."""
    t0 = time.perf_counter()
    scores, info = E.aggregate(u, agg, source, dec.tag)
    sel, net, stats = dec.run(scores)
    t_dec = time.perf_counter() - t0
    counts = I.Instance.cluster_counts(truth_scorer, sel)
    alt = CN.cluster_counts(label_sets(u.words, sel, taxa), truth_labels)
    f1_ok = all(alt[f] == counts[f] for f in ("tp", "fp", "fn"))
    fields = E.network_fields(net)
    ha = None
    if fields["enewick"] is not None:
        _, pred_ha = net_labels(fields["enewick"])
        ha = MS.ha_metrics(pred_ha, truth_ha)["ha_balanced_error"]
    return {"sel": sel, "net": net, "info": info, "f1_ok": f1_ok, "ha": ha,
            "entry": {"n_selected": int(sel.size), **counts, **fields, "decode_stats": stats,
                      "wall": {"decode": t_dec},
                      "ensemble": E.ensemble_block(u, agg, source, dec.tag, sel, info)}}


def decode_cell(ds_name: str, row: dict, runs_root: str) -> dict:
    ds = C.DATASETS[ds_name]
    runs_root = Path(runs_root)
    name = C.cell_name(row["phase"], row["rep"], row["arm"])
    base = runs_root / ds.name / "ens"
    meta = json.loads((base / "scores" / f"{name}.json").read_text())
    draws = load_draws(ds, base / "scores" / f"{name}.npz")
    cell = CAMUS.CamusCell(row["phase"], int(row["rep"]), row["arm"], ds.source)
    truth = CAMUS.truth_of(cell)
    taxa = list(truth.labels)
    if taxa != meta["taxa"]:
        raise AssertionError(f"{name}: the source truth's taxon order is not the cached draws'")
    k, n = int(truth.n_retics), len(taxa)
    truth_labels = CN.label_clusters(truth)
    truth_ha = MS.hybrid_ancestry(truth)
    full = full_networks(ds, name)
    problems: list[str] = []
    t_cell = time.perf_counter()

    def instance(u: E.Union) -> tuple[dict, float]:
        t0 = time.perf_counter()
        decs = {tag: E.Decoder(n, taxa, k, *decoder_args(tag), words=u.words, support=u.support)
                for tag in ds.decoders}
        return decs, time.perf_counter() - t0

    # ---- the main decode: all five draws
    t0 = time.perf_counter()
    u = E.Union(draws)
    t_union = time.perf_counter() - t0
    decs, t_inst = instance(u)
    scorer = scorer_of(u.words, truth)
    main: dict[tuple, dict] = {}
    for tag, dec in decs.items():
        for source in SOURCES:
            for agg in aggs_for(source):
                r = decode_one(u, dec, agg, source, scorer, truth_labels, truth_ha, taxa)
                if not r["f1_ok"]:
                    problems.append(f"G-F1 {name} {source} {tag} {agg}: row and label-set counts differ")
                if (source, tag) in full and r["net"] is not None:
                    mine = CN.label_clusters(r["net"])
                    r["entry"]["ensemble"]["net_equals_full"] = mine == full[(source, tag)]
                    r["entry"]["ensemble"]["net_jaccard_full"] = (
                        len(mine & full[(source, tag)]) / max(1, len(mine | full[(source, tag)])))
                main[(source, tag, agg)] = r

    # ---- write one cell JSON per (agg, run directory), in run_gnn_cpu's format
    draw_prep = sum(dm["prep"] for dm in meta["draws"])
    for tag, dec in decs.items():
        max_level = decoder_args(tag)[1]
        for source_group in ([t for t, _ in MODELS], [GREEDY]):
            for agg in aggs_for(source_group[0]):
                models = []
                for source in source_group:
                    e = dict(main[(source, tag, agg)]["entry"])
                    e = {"tag": source, "seed": 0, "decoder": decoder_args(tag)[0],
                         "max_level": max_level, **e}
                    e["draws_wall"] = {
                        "forward": (sum(dm["forward"].get(source, 0.0) for dm in meta["draws"])),
                        "decode": sum(dm["decode"][f"{source}|{tag}"] for dm in meta["draws"])}
                    models.append(e)
                rec = {"phase": row["phase"], "rep": int(row["rep"]), "arm": row["arm"],
                       "key": cell.key, "trees": meta["trees"], "n": n, "k": k,
                       "decoder": decoder_args(tag)[0], "max_level": max_level,
                       "n_loci": meta["draws"][0]["n_loci"], "m": meta["draws"][0]["m"],
                       "n_draws": u.D, "n_loci_per_draw": [dm["n_loci"] for dm in meta["draws"]],
                       "n_clusters": u.n,
                       "n_edges": int(dec.graph.nbr.size) if not dec.relaxed else None,
                       "wall": {"draws_prep": draw_prep, "union": t_union, "instance": t_inst},
                       "models": models}
                atomic_json(base / agg / run_name(source_group[0], tag) / "cells" / f"{name}.json",
                            rec)

    # ---- G-identity: every aggregator over one draw is that draw
    identity = Counter()
    for d, dr in enumerate(draws):
        u1 = E.Union([dr])
        decs1, _ = instance(u1)
        for tag, dec in decs1.items():
            for source in SOURCES:
                want = E.word_set(dr.words, dr.sel[(source, tag)])
                for agg in aggs_for(source):
                    scores, _ = E.aggregate(u1, agg, source, tag)
                    sel, _, _ = dec.run(scores)
                    ok = E.word_set(u1.words, sel) == want
                    identity["ok" if ok else "failed"] += 1
                    if not ok:
                        problems.append(f"G-identity {name} d{d} {source} {tag} {agg}: the one-draw "
                                        f"ensemble is not the draw's own selection")

    # ---- the sweep: draws 0..D-1
    sweep = []
    for D in range(1, ds.draws + 1):
        uD = u if D == ds.draws else E.Union(draws[:D])
        decsD = decs if D == ds.draws else instance(uD)[0]
        scD = scorer if D == ds.draws else scorer_of(uD.words, truth)
        for tag, dec in decsD.items():
            for source in SOURCES:
                for agg in (("vote",) if source == GREEDY else SWEEP_AGGS):
                    r = (main[(source, tag, agg)] if D == ds.draws else
                         decode_one(uD, dec, agg, source, scD, truth_labels, truth_ha, taxa))
                    if D == 1 and E.word_set(uD.words, r["sel"]) != E.word_set(
                            draws[0].words, draws[0].sel[(source, tag)]):
                        problems.append(f"sweep {name} D=1 {source} {tag} {agg}: not draw 0")
                    ent = r["entry"]
                    sweep.append({"D": D, "source": source, "decoder": tag, "agg": agg,
                                  "n_union": uD.n, "n_selected": ent["n_selected"],
                                  "tp": ent["tp"], "fp": ent["fp"], "fn": ent["fn"], "f1": ent["f1"],
                                  "k_pred": ent["k_pred"], "ha_balanced_error": r["ha"]})

    # ---- order sensitivity: pool with the draws reversed (only row-order ties can move it)
    urev = E.Union(draws[::-1])
    decs_rev, _ = instance(urev)
    order = {}
    for tag, dec in decs_rev.items():
        for source, _ in MODELS:
            scores, _ = E.aggregate(urev, "pool", source, tag)
            sel, _, _ = dec.run(scores)
            order[f"{source}|{tag}"] = (E.word_set(urev.words, sel)
                                        == E.word_set(u.words, main[(source, tag, "pool")]["sel"]))

    rec = {"cell": name, "phase": row["phase"], "rep": int(row["rep"]), "arm": row["arm"],
           "n": n, "k": k, "n_union": u.n, "n_per_draw": [int(x.size) for x in u.rowmap],
           "n_full_draw_max": max(int(x.size) for x in u.rowmap),
           "identity": dict(identity), "order_invariant": order, "sweep": sweep,
           "main": {f"{s}|{t}|{a}": {kk: r["entry"][kk] for kk in ("n_selected", "tp", "fp", "fn",
                                                                  "f1", "k_pred")}
                    | {"ha_balanced_error": r["ha"],
                       **{kk: r["entry"]["ensemble"].get(kk) for kk in (
                           "sel_equals_a_draw", "n_majority", "majority_refused",
                           "net_equals_full", "net_jaccard_full", "sel_in_all_draws",
                           "sel_in_one_draw")}}
                    for (s, t, a), r in main.items()},
           "draw_counts": [dm["counts"] for dm in meta["draws"]],
           "problems": problems, "seconds": time.perf_counter() - t_cell}
    atomic_json(base / "decoded" / f"{name}.json", rec)     # last: its presence means "done"
    return rec


# ========================================================================================= summary

def score_summary(ds: C.Dataset, base: Path, cells: list[dict]) -> int:
    metas = [json.loads(p.read_text()) for p in sorted((base / "scores").glob("*.json"))]
    names = {C.cell_name(r["phase"], r["rep"], r["arm"]) for r in cells}
    metas = [m for m in metas if m["cell"] in names]
    n_cmp = sum(len(m["parity"]) for m in metas)
    bad = [p for m in metas for p in m["problems"]]
    print(f"[score] {ds.name}: {len(metas)}/{len(cells)} cells cached | P1/P1g parity: "
          f"{n_cmp - len(bad)}/{n_cmp} replays reproduce their landed cells")
    for p in bad[:30]:
        print(f"  {p}")
    return len(bad)


def decode_summary(ds: C.Dataset, base: Path, cells: list[dict]) -> int:
    names = {C.cell_name(r["phase"], r["rep"], r["arm"]) for r in cells}
    recs = [json.loads(p.read_text()) for p in sorted((base / "decoded").glob("*.json"))]
    recs = [r for r in recs if r["cell"] in names]
    if not recs:
        print(f"[decode] {ds.name}: nothing decoded yet")
        return 0
    problems = [p for r in recs for p in r["problems"]]
    ident = Counter()
    for r in recs:
        ident.update(r["identity"])
    bigger = sum(r["n_union"] > r["n_full_draw_max"] for r in recs)
    order_moved = Counter(k for r in recs for k, ok in r["order_invariant"].items() if not ok)
    lines = [f"[decode] {ds.name}: {len(recs)}/{len(cells)} cells | G-identity {ident['ok']}/"
             f"{ident['ok'] + ident['failed']} | G-F1 and sweep problems "
             f"{sum(not p.startswith('G-identity') for p in problems)} | union > every draw in "
             f"{bigger}/{len(recs)} cells | pool order-sensitive: "
             f"{dict(order_moved) or 'none'}",
             "",
             "cluster F1 medians (no jar): mean of the 5 draws vs the ensemble; 'differs' = ensemble "
             "unlike every draw's selection",
             f"{'source':22s} {'decoder':11s} {'agg':13s} {'cells':>5s} {'draws':>7s} {'ens':>7s} "
             f"{'d md':>7s} {'better':>6s} {'worse':>6s} {'HA ens':>7s} {'differs':>7s} "
             f"{'maj ref':>7s}"]
    keys = sorted({k for r in recs for k in r["main"]},
                  key=lambda s: (SOURCES.index(s.split("|")[0]), s.split("|")[1],
                                 E.AGGREGATORS.index(s.split("|")[2])))
    for key in keys:
        source, tag, agg = key.split("|")
        rows = [r for r in recs if key in r["main"]]
        mean_draw = [statistics.fmean(dc[f"{source}|{tag}"]["f1"] for dc in r["draw_counts"])
                     for r in rows]
        ens = [r["main"][key]["f1"] for r in rows]
        delta = [e - m for e, m in zip(ens, mean_draw)]
        ha = [r["main"][key]["ha_balanced_error"] for r in rows
              if r["main"][key]["ha_balanced_error"] is not None]
        refused = [r["main"][key]["majority_refused"] for r in rows
                   if r["main"][key].get("majority_refused") is not None]
        lines.append(
            f"{source:22s} {tag:11s} {agg:13s} {len(rows):5d} {statistics.median(mean_draw):7.4f} "
            f"{statistics.median(ens):7.4f} {statistics.median(delta):+7.4f} "
            f"{sum(x > 0 for x in delta):6d} {sum(x < 0 for x in delta):6d} "
            f"{(statistics.median(ha) if ha else float('nan')):7.4f} "
            f"{sum(not r['main'][key]['sel_equals_a_draw'] for r in rows):7d} "
            f"{(statistics.fmean(refused) if refused else float('nan')):7.2f}")
    text = "\n".join(lines)
    print(text)
    for p in problems[:30]:
        print(f"  PROBLEM {p}")
    atomic_json(base / "decode_summary.json",
                {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": len(recs),
                 "identity": dict(ident), "union_bigger": bigger,
                 "order_sensitive": dict(order_moved), "problems": problems, "table": lines})
    return len(problems)


# ============================================================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("score", "decode"))
    ap.add_argument("--dataset", required=True, choices=ENS_DATASETS)
    ap.add_argument("--phases", default="", help="comma list; default all")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one thread each)")
    ap.add_argument("--runs-root", type=Path, default=C.RUNS,
                    help="where ens/ is written (a smoke points it at the scratchpad)")
    ap.add_argument("--landed-root", type=Path, default=C.RUNS,
                    help="where the landed d<k> subsample runs are read from (gate P1)")
    ap.add_argument("--force", action="store_true", help="redo cells already done")
    ap.add_argument("--dry-run", action="store_true", help="list the cells still to do")
    ap.add_argument("--summary-only", action="store_true", help="re-tally what is on disk")
    ap.add_argument("--threads", type=int, default=1,
                    help="read by run_gnn_cpu at import; keep 1 (the landed runs used 1)")
    args = ap.parse_args()
    ds = C.DATASETS[args.dataset]
    cells = cells_of(ds, args.phases, args.reps)
    if not cells:
        raise SystemExit("no cells selected")
    base = Path(args.runs_root) / ds.name / "ens"
    if args.phase == "score":
        done = lambda r: (base / "scores" / f"{C.cell_name(r['phase'], r['rep'], r['arm'])}.json").exists()  # noqa: E731
    else:
        missing = [r for r in cells
                   if not (base / "scores" / f"{C.cell_name(r['phase'], r['rep'], r['arm'])}.json").exists()]
        if missing and args.dry_run:
            print(f"[decode] note: {len(missing)} of {len(cells)} cells have no score cache yet; "
                  f"the score phase must run first")
        elif missing and not args.summary_only:
            raise SystemExit(f"{len(missing)} of {len(cells)} cells have no score cache under "
                             f"{base / 'scores'}: run the score phase first")
        done = lambda r: (base / "decoded" / f"{C.cell_name(r['phase'], r['rep'], r['arm'])}.json").exists()  # noqa: E731
    todo = [r for r in cells if args.force or not done(r)]
    print(f"[{args.phase}] {ds.name}: {len(cells)} cells ({len(cells) - len(todo)} done) | jobs "
          f"{args.jobs} | -> {base}", flush=True)
    if args.dry_run:
        for r in todo:
            print(f"  {r['phase']:12s} rep{int(r['rep']):02d} {r['arm']:12s} n={r['n']} k={r['k']}")
        return 0
    summarise = score_summary if args.phase == "score" else decode_summary
    if args.summary_only:
        n_bad = summarise(ds, base, cells)
        return (3 if args.phase == "score" else 1) if n_bad else 0

    fn = score_cell if args.phase == "score" else decode_cell
    fargs = ((str(args.runs_root), str(args.landed_root)) if args.phase == "score"
             else (str(args.runs_root),))
    t_run = time.perf_counter()

    def report(i: int, rec: dict) -> None:
        if args.phase == "score":
            n_ok = sum(not v for v in rec["parity"].values())
            state = f"parity {n_ok}/{len(rec['parity'])}"
        else:
            ident = rec["identity"]
            state = (f"|U|={rec['n_union']} identity {ident.get('ok', 0)}/"
                     f"{ident.get('ok', 0) + ident.get('failed', 0)} | pool F1 "
                     + " ".join(f"{rec['main'][k]['f1']:.3f}" for k in rec["main"]
                                if k.endswith("|pool") and not k.startswith(GREEDY)))
        print(f"[{args.phase}] {i:3d}/{len(todo)} {rec['cell']:28s} {state} | "
              f"{rec['seconds']:5.1f}s | {time.perf_counter() - t_run:5.0f}s"
              + (f" | {len(rec['problems'])} PROBLEMS" if rec["problems"] else ""), flush=True)

    if args.jobs <= 1:
        for i, r in enumerate(todo, 1):
            report(i, fn(ds.name, r, *fargs))
    else:
        init = _init_worker if args.phase == "score" else None
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn"),
                                 initializer=init) as ex:
            futs = [ex.submit(fn, ds.name, r, *fargs) for r in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                report(i, fut.result())
    n_bad = summarise(ds, base, cells)
    if args.phase == "score":
        return 3 if n_bad else 0
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
