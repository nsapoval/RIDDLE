#!/usr/bin/env python3
"""The W10 GNN on several subsamples of one gene-tree file, combined into one network per k.

``infer_gnn_cpu.py`` takes one sample of ``--m`` trees.  This takes ``--draws B`` of them and merges
them (``ensemble.py``): **draw b is exactly ``infer_gnn_cpu.py --m M --seed S+b``** -- the same
``scan``, ``select``, ``prepare``, candidate set, features and forward -- and the B draws' scored
candidate sets are then combined and decoded once:

``--agg pool``          (default) each candidate's mean score over the draws, 0 in a draw where it
                        is below tau, then one decode on the union of the candidate sets;
``--agg vote``          each draw decoded on its own; the clusters a majority of draws selected go
                        first, then the decode re-extends them to a complete system by ``pool``;
``--agg pool_present``  the mean over only the draws that hold the candidate (an ablation).

``subsample_eval/`` benchmarks the three on five 500-of-1,000 draws (``README.md`` there,
"Ensembles over the draws"), and ``subsample_eval/check_ens.py`` holds this script to
``infer_gnn_cpu.py`` (``--draws 1`` is byte-identical) and to that benchmark.

Draws
-----
Each draw is a seeded random sample of ``--m`` of the eligible trees, drawn independently of the
others, so draws overlap: B draws of m from N eligible cover ``1 - (1 - m/N)^B`` of them.  With N
at most ``m`` every draw is the same trees and the ensemble says nothing; that is refused.  N below
``2m`` is allowed, with a warning.  Under ``--midpoint`` / ``--species-tree`` a draw can end up
with fewer than ``m`` trees (rejections the reserve cannot replace); each draw's count is logged.

Output
------
stdout, or ``--out``: one line per ``k``, ``k<TAB>eNewick``, as ``infer_gnn_cpu.py`` writes.
``--draws-out FILE``: each draw's own network, ``k<TAB>draw<TAB>seed<TAB>eNewick``.
``--clusters-out FILE``: per ``k``, every cluster of the union and of the output network --
``taxa`` (comma list), ``size``, ``n_present`` (draws whose candidate set holds it), ``pool`` (mean
score, absent = 0), ``votes`` (draws whose own decode selected it), ``selected`` and ``in_network``.
``votes / B`` is the ensemble's per-cluster support.  ``infer_gnn_ens.md`` is the guide.

Usage
-----
    .venv/bin/python infer_gnn_ens.py --model checkpoints/_l2mix_pair/s0/best.pt \\
        --trees genes.newick --outgroup OUT --k 1,2 --draws 5 --m 1000 --out nets.tsv
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# First, before numpy, torch or opc: it pins threads, puts --opc's library first and checks that
# nothing shadows it or rooting.py.  It reads --threads and --opc from argv, which this accepts too.
import infer_gnn_cpu as IG  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import ensemble as E  # noqa: E402
from opc import bitset  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl.env import OverlapCSR  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.solve import candidates  # noqa: E402

SOURCE = "gnn"
CLUSTER_COLUMNS = ("k", "taxa", "size", "n_present", "pool", "votes", "selected", "in_network")


def build_draw(trees: Path, chosen, reserve, masks, labels, S, taxa_order, *, outgroups=(),
               midpoint=False, species=None, tau=None, tau_frac=None) -> dict:
    """One draw's instance: ``infer_gnn_cpu.infer``'s stage chain, call for call.

    ``infer()`` does these inline and returns only the networks, so this is a copy of its
    ``prepare -> count_clusters -> candidates.build -> OverlapCSR -> node_features`` block;
    ``check_ens.py`` R1 (``--draws 1`` byte-identical to ``infer_gnn_cpu.py``) licenses it.
    """
    taxon_index = {lab: i for i, lab in enumerate(taxa_order)}
    with tempfile.TemporaryDirectory(prefix="infer_ens_") as work:
        gz, table = Path(work) / "trees.nwk.gz", Path(work) / "clusters.parquet"
        prep = IG.prepare(trees, chosen, gz, masks, labels, S, outgroups, midpoint=midpoint,
                          species=species, reserve=reserve)
        mm = prep["n_loci"]
        if not mm:
            raise SystemExit(f"no tree could be rooted ({prep['scheme']}): {len(chosen):,} tried, "
                             f"rejected {prep['rejected']}")
        counter = IG.count_clusters(gz, taxon_index, mm)
        words_raw, support, n_raw = counter.finalize(min_support=1)
        extract.write_cluster_table(table, words_raw, support, [mm], len(taxa_order), n_raw,
                                    source=str(trees))
        words, sup, meta = extract.read_cluster_table(table)
        thr = IG.resolve_tau(mm, tau, tau_frac)
        gs = candidates.build(words, sup[:, meta["checkpoints"].index(mm)], len(taxa_order),
                              m=mm, tau=thr)
        if not gs.n_clusters:
            raise SystemExit(f"|C| = 0: no clade is in {thr:,} of the {mm:,} trees used "
                             f"({IG._tau_text(thr, tau_frac, mm)}); lower --tau or --tau-frac")
        graph = OverlapCSR.of(gs)
        xn = I.node_features(gs, graph, I._tree_stats(gz, taxon_index, gs, mm))
    return {"gs": gs, "graph": graph, "xn": xn, "prep": prep, "m": mm, "tau": thr}


def label_set(words: np.ndarray, row: int, taxa: list[str]) -> frozenset[str]:
    return frozenset(taxa[i] for i in bitset.to_indices(words[row], len(taxa)))


def cluster_rows(k: int, u: E.Union, sel: np.ndarray, net, info: dict, votes: np.ndarray,
                 taxa: list[str]) -> list[dict]:
    """``--clusters-out``: every union row, then every network cluster the union lacks (level1's
    closure can add an intersection that no draw had as a candidate)."""
    in_net = {frozenset(net.labels[i] for i in c) for c in NW.nontrivial_clusters(net)}
    chosen = np.zeros(u.n, dtype=bool)
    chosen[sel] = True
    npres = u.n_present
    rows, seen = [], set()
    for r in range(u.n):
        lab = label_set(u.words, r, taxa)
        seen.add(lab)
        rows.append({"k": k, "taxa": ",".join(sorted(lab)), "size": len(lab),
                     "n_present": int(npres[r]), "pool": float(info["pool"][r]),
                     "votes": int(votes[r]), "selected": int(chosen[r]),
                     "in_network": int(lab in in_net)})
    for lab in sorted(in_net - seen, key=sorted):
        rows.append({"k": k, "taxa": ",".join(sorted(lab)), "size": len(lab), "n_present": 0,
                     "pool": 0.0, "votes": 0, "selected": 0, "in_network": 1})
    rows.sort(key=lambda x: (-x["selected"], -x["in_network"], -x["pool"], -x["votes"]))
    return rows


def infer_ensemble(trees: Path, net, ks, *, outgroups=(), midpoint=False, species_tree=None,
                   taxa=None, n_taxa=0, partial=0, m=1000, draws=5, seed=0, seeds=None, tau=None,
                   tau_frac=None, order=None, decoder="level1", max_level=None, agg="pool",
                   log=sys.stderr) -> list[dict]:
    """One dict per ``k``: ``k, decoder, agg, enewick, n_selected, n_retics, seconds``, plus
    ``draws`` (each draw's own network), ``ensemble`` (``ensemble.ensemble_block``) and
    ``clusters`` (the ``--clusters-out`` rows).

    ``seeds`` overrides ``seed + b``; ``order`` overrides ``sorted(S)`` as the taxon order and must
    be a permutation of S -- both exist so a benchmark cell can be replayed exactly
    (``subsample_eval/check_ens.py`` R2).
    """
    t0 = time.perf_counter()
    tag = RX.decoder_tag(decoder, max_level)
    IG.resolve_tau(1, tau, tau_frac)
    if agg not in E.AGGREGATORS:
        raise ValueError(f"agg {agg!r}: expected one of {', '.join(E.AGGREGATORS)}")
    seeds = [int(seed) + b for b in range(int(draws))] if seeds is None else [int(s) for s in seeds]
    if not seeds:
        raise ValueError("at least one draw is needed")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds {seeds}: a repeated seed is a repeated draw")
    species = species_tree
    if species_tree is not None and not isinstance(species_tree, IG.RT.SpeciesTree):
        try:
            species = IG.RT.SpeciesTree.read(species_tree)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--species-tree {species_tree}: {exc}") from None
    IG.rooting_scheme(outgroups, midpoint, species)

    masks, labels, freq = IG.scan(trees)
    picks, S = [], None
    for s in seeds:
        S_b, chosen, reserve, stats = IG.select(masks, labels, freq, outgroups=outgroups, taxa=taxa,
                                                n_taxa=n_taxa, partial=partial, m=m, seed=s,
                                                species=species)
        if S is not None and S_b != S:
            raise AssertionError("the taxon set changed with the seed")
        S = S_b
        picks.append((s, chosen, reserve, stats))
    eligible = picks[0][3]["eligible"]
    if len(seeds) > 1:
        if not m or eligible <= m:
            raise SystemExit(f"{eligible:,} trees are eligible and --m is {m or 'all'}: every draw "
                             f"would be the same trees, so the ensemble would say nothing.  Lower "
                             f"--m (e.g. --m {max(3, eligible // 2)}) or use infer_gnn_cpu.py.")
        if eligible < 2 * m:
            cover = 1 - (1 - m / eligible) ** len(seeds)
            print(f"[ens] warning: {eligible:,} eligible trees and --m {m:,}: the {len(seeds)} "
                  f"draws overlap heavily (together they cover ~{cover:.0%} of the trees)",
                  file=log, flush=True)
    taxa_order = S if order is None else list(order)
    if sorted(taxa_order) != S:
        raise ValueError("order must be a permutation of the selected taxon set")
    n = len(taxa_order)

    built = []
    for b, (s, chosen, reserve, stats) in enumerate(picks):
        t1 = time.perf_counter()
        d = build_draw(trees, chosen, reserve, masks, labels, S, taxa_order, outgroups=outgroups,
                       midpoint=midpoint, species=species, tau=tau, tau_frac=tau_frac)
        prep = d["prep"]
        rej = sum(prep["rejected"].values())
        print(f"[ens] draw {b} (seed {s}): {d['m']:,} trees used{f', {rej:,} rejected' if rej else ''}"
              f" | |C| = {d['gs'].n_clusters:,} at {IG._tau_text(d['tau'], tau_frac, d['m'])} | "
              f"{time.perf_counter() - t1:.1f}s", file=log, flush=True)
        built.append(d)
    ens_draws = [E.Draw(words=d["gs"].words, support=d["gs"].support) for d in built]
    u = E.Union(ens_draws)
    gs_u, graph_u = E.instance(u.words, u.support, n)
    cover = len(set().union(*(set(c) for _, c, _, _ in picks)))
    print(f"[ens] {stats['n_trees']:,} trees over {stats['n_labels']:,} taxa | S = {n} taxa | "
          f"{eligible:,} eligible | {len(seeds)} draws of {m:,} cover {cover:,} trees | union "
          f"|U| = {u.n:,} (draws {', '.join(f'{r.size:,}' for r in u.rowmap)}) | "
          f"{time.perf_counter() - t0:.1f}s", file=log, flush=True)

    out = []
    for k in ks:
        t1 = time.perf_counter()
        own = []
        for d, dr in zip(built, ens_draws):
            gs, graph = d["gs"], d["graph"]
            p = IG.forward(net, d["xn"], I.global_features(gs, graph, k), graph, gs)
            dec = E.Decoder(n, taxa_order, k, decoder, max_level, gs=gs,
                            graph=None if decoder == "relaxed" else graph)
            sel, dnet, _ = dec.run(p)
            if dnet is None:
                raise RuntimeError(f"k={k}: a draw's {sel.size} selected clusters realise no network")
            dr.scores[SOURCE] = p
            dr.sel[(SOURCE, tag)] = sel
            own.append({"enewick": NW.to_enewick(dnet), "n_selected": int(sel.size),
                        "n_retics": int(dnet.n_retics)})
        dec = E.Decoder(n, taxa_order, k, decoder, max_level, gs=gs_u,
                        graph=None if decoder == "relaxed" else graph_u)
        scores, info = E.aggregate(u, agg, SOURCE, tag)
        sel, enet, _ = dec.run(scores)
        if enet is None:
            raise RuntimeError(f"k={k}: the {sel.size} selected clusters realise no network")
        block = E.ensemble_block(u, agg, SOURCE, tag, sel, info)
        votes = u.votes(SOURCE, tag)
        row = {"k": k, "decoder": tag, "agg": agg, "enewick": NW.to_enewick(enet),
               "n_selected": int(sel.size), "n_retics": int(enet.n_retics),
               "seconds": time.perf_counter() - t1, "seeds": seeds,
               "draws": [{"seed": s, **o} for s, o in zip(seeds, own)], "ensemble": block,
               "clusters": cluster_rows(k, u, sel, enet, info, votes, taxa_order)}
        maj = (f", majority {block['n_majority']} ({block['majority_refused']} refused)"
               if "n_majority" in block else "")
        print(f"[ens] k={k} ({tag}, {agg}): {row['n_selected']} clusters selected{maj}, "
              f"{row['n_retics']} reticulation(s) realised | {block['sel_in_all_draws']} selected "
              f"clusters in every draw's candidate set, {block['sel_in_one_draw']} in one | "
              f"{'equals draw ' + str(block['sel_equals_draw'].index(True)) if block['sel_equals_a_draw'] else 'unlike every draw'}"
              f" | {row['seconds']:.2f}s", file=log, flush=True)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="a checkpoint, e.g. "
                    "checkpoints/_l2mix_pair/s0/best.pt")
    ap.add_argument("--opc", type=Path, default=HERE / "pylib",
                    help="the directory holding opc/, or opc/ itself; read before import")
    ap.add_argument("--trees", type=Path, required=True,
                    help="one Newick tree per line, plain or .gz")
    ap.add_argument("--k", required=True, help="comma list of reticulation budgets, e.g. 1,2,3")
    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--n-taxa", type=int, default=0,
                     help="restrict S to the N most frequent taxa (ties broken by label)")
    sub.add_argument("--taxa", type=Path, default=None,
                     help="restrict S to these labels, one per line")
    root = ap.add_mutually_exclusive_group()
    root.add_argument("--outgroup", default="",
                      help="ranked comma list: each tree is rooted on the first one it carries and "
                           "every one is pruned; a tree carrying none is skipped")
    root.add_argument("--midpoint", action="store_true",
                      help="root each tree at the middle of its longest leaf-to-leaf path")
    root.add_argument("--species-tree", type=Path, default=None, metavar="FILE",
                      help="a rooted species tree to root each gene tree on (infer_gnn_cpu.md 7.5)")
    ap.add_argument("--partial", type=int, default=0, metavar="MIN",
                    help="accept a tree carrying >= MIN taxa of S, pruned to what it has")
    ap.add_argument("--draws", type=int, default=5, help="subsamples to combine (default 5)")
    ap.add_argument("--m", type=int, default=1000,
                    help="trees per draw (default 1000, the m every checkpoint trained at)")
    ap.add_argument("--seed", type=int, default=0,
                    help="draw b uses seed S+b, i.e. it is infer_gnn_cpu.py --m M --seed S+b")
    ap.add_argument("--seeds", default="", help="comma list of seeds, one per draw; replaces "
                                                "--seed and --draws")
    cut = ap.add_mutually_exclusive_group()
    cut.add_argument("--tau", type=int, default=None, metavar="N",
                     help=f"a candidate clade must be in >= N of a draw's trees (default {IG.TAU})")
    cut.add_argument("--tau-frac", default=None, metavar="F",
                     help="a candidate clade must be in >= ceil(F * m) of a draw's m trees")
    ap.add_argument("--decoder", choices=RX.DECODERS, default="level1")
    ap.add_argument("--max-level", type=int, default=None,
                    help="the relaxed decoder's level cap, >= 1; required with --decoder relaxed")
    ap.add_argument("--agg", choices=E.AGGREGATORS, default="pool",
                    help="pool (default), vote, or pool_present -- see the module docstring")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch and BLAS threads; read before import")
    ap.add_argument("--out", type=Path, default=None, help="default stdout")
    ap.add_argument("--draws-out", type=Path, default=None,
                    help="also write each draw's own network: k<TAB>draw<TAB>seed<TAB>eNewick")
    ap.add_argument("--clusters-out", type=Path, default=None,
                    help="also write every union / network cluster's ensemble support (TSV)")
    args = ap.parse_args()

    try:
        ks = [int(k) for k in args.k.split(",") if k.strip()]
    except ValueError:
        ap.error(f"--k {args.k!r}: expected a comma list of integers")
    if not ks or min(ks) < 0:
        ap.error("--k needs one or more non-negative integers")
    if args.partial and args.partial < 3:
        ap.error("--partial MIN must be at least 3: a rooted tree on fewer taxa has no cluster")
    if args.m < 0:
        ap.error("--m must be >= 0")
    if args.draws < 1:
        ap.error("--draws must be at least 1")
    if args.tau is not None and args.tau < 1:
        ap.error("--tau must be at least 1: 1 already keeps every clade any tree has")
    if args.tau_frac is not None:
        try:
            IG.resolve_tau(1, tau_frac=args.tau_frac)
        except ValueError:
            ap.error(f"--tau-frac {args.tau_frac!r}: expected a number in (0, 1]")
    try:
        RX.decoder_tag(args.decoder, args.max_level)
    except ValueError as exc:
        ap.error(f"--decoder {args.decoder}: {exc}")
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or None
    except ValueError:
        ap.error(f"--seeds {args.seeds!r}: expected a comma list of integers")
    outgroups = tuple(dict.fromkeys(o.strip() for o in args.outgroup.split(",") if o.strip()))
    taxa = None if args.taxa is None else [
        s.strip() for s in args.taxa.read_text().splitlines() if s.strip()]

    rows = infer_ensemble(args.trees, IG.load_model(args.model), ks, outgroups=outgroups,
                          midpoint=args.midpoint, species_tree=args.species_tree, taxa=taxa,
                          n_taxa=args.n_taxa, partial=args.partial, m=args.m, draws=args.draws,
                          seed=args.seed, seeds=seeds, tau=args.tau, tau_frac=args.tau_frac,
                          decoder=args.decoder, max_level=args.max_level, agg=args.agg)
    text = "".join(f"{r['k']}\t{r['enewick']}\n" for r in rows)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    if args.draws_out:
        args.draws_out.write_text("".join(f"{r['k']}\t{b}\t{d['seed']}\t{d['enewick']}\n"
                                          for r in rows for b, d in enumerate(r["draws"])))
    if args.clusters_out:
        with open(args.clusters_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, CLUSTER_COLUMNS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in rows:
                for c in r["clusters"]:
                    w.writerow({**c, "pool": f"{c['pool']:.6g}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
