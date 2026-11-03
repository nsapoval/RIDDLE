#!/usr/bin/env python3
"""CPU inference for the W10 segment head on the camus benchmark, and what every stage of it costs.

Self-contained: ``pylib/opc`` beside this file is a byte-for-byte copy of the repository's library,
``checkpoints/`` holds the trained models and ``inputs/`` the gene trees in camus-data's own
layout.  The dependency list is numpy, scipy, pyarrow and a CPU build of torch
(``requirements-cpu.txt``).  No GPU, no Java, no network.  Inside the repository
(``scripts/handoff/run_gnn_cpu.py``) it falls back to ``src/``; ``--checkpoints`` and ``--inputs``
must then point at directories in the package's layout, which ``scripts/pack_gnn_cpu.py`` builds.

What one cell is
----------------

    root each gene tree on OUT and prune it  ->  count clusters  ->  tau = 2 candidate set
    ->  overlap graph  ->  W13 tree statistics  ->  node + global features     [once per cell]
    ->  encoder forward  ->  score-ordered decode  ->  realise the network      [once per model]

That is the chain ``opc.rl.camus.build`` -> ``instances.build_from_trees`` -> ``w10_gnn_eval.py``
ran at home to produce every landed camus row, called function for function, **less the two
things inference never reads**: the rooted-triple tensor (the MPL reward's input, and 37 s of a
~58 s instance build at n = 200, measured) and the truth labels.  A model's end-to-end time on a
cell is the shared stages plus its own three; ``timings.tsv`` carries both halves.

**The true network is read, outside every clock, for two things.**  ``k``: the decode is handed
the true reticulation count as its budget, as it was for every landed row -- BROOQS and NetCS
choose their own, which is a difference of task and not of speed.  And the **taxon order**,
which is not cosmetic: it fixes the row order of the candidate set, and the stable tie-breaks in
``support_rank`` and in the decode inherit it.  Measured at home, truth order against sorted order
moved scores by up to 1e-3 and swapped two clusters on 1 of 63 (cell, model) pairs.  Reproducing
the landed rows needs the landed order.  After the clocks stop, the same network scores the
selection -- ``Instance.cluster_counts``, the landed definition -- which is what parity compares.

Parity
------

``reference.tsv`` is the landed ``rl_gnn_eval_*`` rows for exactly these cells and checkpoints.
A (cell, model) pair matches when ``n_selected``, ``tp`` and ``fp`` all agree.  On a different CPU
a near-tie can resolve the other way -- float32 reductions are not bitwise portable across
instruction sets, and the taxon-order measurement above is the size of effect that produces -- so
a **rare** mismatch that moves F1 by one cluster is that, not a bug.  Many mismatches are a bug.

Decoders
--------

``--decoder level1``, the default, is the formal decoder every landed row used: the (L)/(N3O) mask
of ``opc.rl.env.ClusterEnv``, a budget on distinct intersection classes, and the network read off
the closure.  ``--decoder relaxed --max-level L`` is ``opc.rl.relaxed``: the same scores, rank rule
and budget k, with legality stated on the Hasse diagram of the selection itself -- at most k
reticulations, in-degree at most 2, level at most L, a donor assignment -- and the network read off
that diagram with no closure.  It exists because every level-1 condition fails on level-2 truths;
see that module's docstring.  Parity against ``reference.tsv`` is a statement about level-1 decodes
and is skipped for it.  Every cell JSON records its decoder, and a run directory holds only one.

Prefixes
--------

``--prefix`` adds, to every model entry, ``prefix.truncated`` and ``prefix.rank_laminar``: the
cluster system the decode held before it first spent reticulation budget, cut by two rules
(``opc.rl.prefix``'s docstring).  ``truncated`` is the decoder's own state before its first
budget-spending acceptance; ``rank_laminar`` is the longest laminar prefix of the score ranking,
the same for both decoders.  Each block carries its size, where and on which cluster it stopped,
whether that cluster is true, the landed tp/fp/fn/F1 against the truth, ``contained_hardwired``
(``fp == 0``) and the prefix as a multifurcating Newick with no lengths.  The decode is traced
*after* the timed one, on its own ``prefix`` clock, and must select exactly what the timed one did,
so timings and parity are unchanged.  Without the flag every cell JSON is as it was, and a
``--prefix`` run refuses to resume over cells written without it.

Resolving a tree of blobs
-------------------------

``--tob-manifest TSV --resolve-max-level L`` adds ``tob_resolve`` to every model entry.  Each row
of the manifest (``tob_resolve_eval/stage.py``) gives a cell a tree of blobs and a budget, and the
block runs the GNN as a resolver on it (``opc.rl.tob_resolve``):
- candidates that cross the TOB are cut from the decode;
- every TOB cluster is kept;
- the decode runs once level-1 (``strict_L1``) and once relaxed at level cap L (``relaxed_L<L>``).

The budget is the row's, usually ``k_TOB`` and not ``k_true``.  ``k`` is a global feature, so the
forward is recomputed at each distinct budget, and those forwards are cached per cell.

All of this runs after the timed decode, on its own clock, so timings and parity are unchanged.
It needs ``--decoder level1``.  Without the flag every cell JSON is as it was, and a run with it
refuses to resume over cells written without it.

Trees and threshold
-------------------

Every checkpoint trained on m = 1,000 trees at tau = 2, and so does every landed cell: by default
the first 1,000 trees of a file are read and the rest ignored, **silently**.  ``--m N`` reads N
(``0`` = all, m = the count read) and ``--tau-frac F`` sets tau = ceil(F * m); 0.002 is training's
share, so ``--m 0 --tau-frac 0.002`` on a 1,000-tree file is the default cell exactly.  Off the
defaults each cell JSON gains ``m_arg``, ``tau`` and ``tau_frac``, and a run refuses to resume over
cells built with other values.  ``--partial MIN`` (2026-10-21, ``taxmiss_eval``) admits trees missing
taxa -- each must keep OUT and at least MIN taxa -- and drops each partial tree's own leaf set from
its clusters (``infer_gnn_cpu.count_clusters``' rule); its cells gain ``partial``, ``n_partial``,
``min_taxa`` and ``mean_missing``, and resume is refused across a different MIN.  A tau other than 2 of 1,000 moves every feature normalised by |C|
(``infer_gnn_cpu.md`` §7.6); at 10,000 trees tau = 2 is not possible at all -- the overlap graph
passes ``EDGE_CAP`` (``l2reg_eval/README.md``, gene-tree scaling).

Usage
-----

    python3 run_gnn_cpu.py --self-check
    python3 run_gnn_cpu.py --dry-run
    python3 -u run_gnn_cpu.py --threads 1                        # every cell, every model
    python3 -u run_gnn_cpu.py --phases cam25 --reps 0-1          # a smoke test
    python3 -u run_gnn_cpu.py --tags _mix2L --seeds 0 --threads 4
    python3 -u run_gnn_cpu.py --decoder relaxed --max-level 2    # the relaxed decoder
    python3 -u run_gnn_cpu.py --prefix --run-name <host>_prefix  # + reticulation-free prefixes
    python3 -u run_gnn_cpu.py --tob-manifest TSV --resolve-max-level 2   # + the GNN as a TOB resolver
    python3 -u run_gnn_cpu.py --m 0 --tau-frac 0.002 ...         # every tree, training's share

Resumable per cell: a cell whose JSON is already under ``runs/<run>/cells/`` is skipped unless
``--force``.  ``--threads`` pins torch and every BLAS to that many threads and is in the run name,
so two thread settings never share a directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

# ``zip(strict=True)`` is in the vendored library's import closure (``opc.mpl.msnc3``, reached
# through ``opc.followup.search``), so 3.10 is a hard floor.  Checked here rather than left to
# surface as a TypeError in the middle of the first cell.
if sys.version_info < (3, 10):
    raise SystemExit(f"needs Python 3.10 or newer (running {sys.version.split()[0]}); "
                     f"3.12 is what the package was built and tested with")

HERE = Path(__file__).resolve().parent

# **Threads are pinned before numpy or torch is imported**, because both read these once at load
# and a BLAS that has already sized its pool ignores a later change.  Assigned, not
# ``setdefault``: a stray ``OMP_NUM_THREADS`` in the caller's shell must not silently override the
# flag that names the run directory.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=1)
THREADS = max(1, _pre.parse_known_args()[0].threads)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(THREADS)

PYLIB = HERE / "pylib"
LIB = PYLIB if (PYLIB / "opc").is_dir() else HERE.parents[1] / "src"
sys.path.insert(0, str(LIB))

_T_IMPORT = time.perf_counter()
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:                  # already set in this process; harmless
    pass

import opc  # noqa: E402
from opc import bitset  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import nodecls as NC  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl import prefix as PX  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl import tob_resolve as TR  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.solve import candidates  # noqa: E402
from opc.solve import truth as T  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - _T_IMPORT

#: Every landed camus row was built at 1,000 loci and tau = 2, and every checkpoint trained there.
#: ``--m`` and ``--tau-frac`` move off it (2026-09-21, the l2reg/l3reg gt5k/gt10k runs); their
#: defaults are these, and at the defaults every cell JSON is what it always was.
M, TAU = 1000, 2


def resolve_tau(m: int, tau_frac=None) -> int:
    """``ceil(tau_frac * m)`` over the ``m`` trees used, else ``TAU``.

    A copy of ``infer_gnn_cpu.resolve_tau``'s ``tau_frac`` branch, not an import of it: that module
    parses ``sys.argv`` and edits ``sys.path`` at import.  ``tau_frac`` is read as the decimal it is
    written as, not as a binary float -- ``0.07`` of 100 trees is 7, where ``math.ceil(0.07 * 100)``
    is 8 -- and never goes below 1.  So ``0.002`` of 1,000 trees is training's tau = 2 exactly.
    """
    if tau_frac is None:
        return TAU
    try:
        frac = Fraction(str(tau_frac))
    except ValueError:
        frac = None
    if frac is None or not 0 < frac <= 1:
        raise ValueError(f"tau_frac {tau_frac!r}: expected a number in (0, 1]")
    return max(1, math.ceil(frac * m))
SHARED = ("root", "clusters", "candidates", "overlap", "tree_stats", "features")
PER_MODEL = ("forward", "decode", "network")


class Clock:
    """Wall and process-CPU seconds per named stage.

    CPU time is summed over every thread of the process, so it is the number to compare across
    ``--threads`` settings and across machines with different core counts; wall time is what a
    user waits for.  Both are recorded because they answer different questions.
    """

    def __init__(self):
        self.wall: dict[str, float] = {}
        self.cpu: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.wall[name] = time.perf_counter() - w0
            self.cpu[name] = time.process_time() - c0


def machine() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    cap = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", lambda: "?")()
    return {"host": socket.gethostname(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "cpus_in_affinity": affinity, "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda, "torch_cpu_capability": cap,
            "numpy": np.__version__, "threads": THREADS, "lib": str(LIB),
            "opc": str(Path(opc.__file__).resolve().parent)}


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def parse_reps(spec: str) -> set[int] | None:
    """``"0-19,25"`` -> ``{0..19, 25}``; empty means every rep."""
    if not spec:
        return None
    want: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def load_models(root: Path, tags, seeds, stem: str = "best") -> list:
    """Every ``checkpoints/<tag>/s<seed>/<stem>.pt``, refused if its feature widths are not this
    library's -- the same two checks ``w10_gnn_eval.py`` makes, for the same reason: a width
    mismatch otherwise fails inside the first matmul and reads as a corrupt file.

    ``stem`` other than ``best`` names a second checkpoint of the same run -- ``_l2mixha/s0`` ships
    ``best.pt`` (selected on F1) and ``best_ha.pt`` (on hybrid ancestry) -- and the model is then
    recorded as tag ``<tag>:<stem>``, so every cell JSON, and every CSV keyed on (tag, seed), tells
    the two apart."""
    out = []
    for tag in tags:
        for seed in seeds:
            path = Path(root) / tag / f"s{seed}" / f"{stem}.pt"
            if not path.exists():
                raise SystemExit(f"no checkpoint at {path}")
            t0 = time.perf_counter()
            ck = torch.load(path, map_location="cpu", weights_only=False)
            cfg = P.PolicyConfig(**dict(ck["policy_config"]))
            if (cfg.n_node_features != len(I.NODE_FEATURES)
                    or cfg.n_global_features != len(I.GLOBAL_FEATURES)):
                raise SystemExit(
                    f"{path}: trained with {cfg.n_node_features} node / {cfg.n_global_features} "
                    f"global features; this library has {len(I.NODE_FEATURES)} / "
                    f"{len(I.GLOBAL_FEATURES)}.  The package's pylib/ and checkpoints/ disagree.")
            run_cfg_path = path.parent / "config.json"
            run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}
            if run_cfg.get("decode", "pointwise") != "pointwise":
                raise SystemExit(f"{path}: decode {run_cfg['decode']!r}; this runner implements "
                                 f"the pointwise (W10) decode only")
            net = P.Policy(cfg)
            net.load_state_dict(ck["policy"])
            net.eval()
            out.append(SimpleNamespace(tag=tag if stem == "best" else f"{tag}:{stem}",
                                       seed=int(seed), net=net, epoch=ck.get("epoch"),
                                       load_s=time.perf_counter() - t0))
    return out


def forward(net, xn, xg, graph, gs) -> np.ndarray:
    """``w10_gnn_eval.py``'s scoring, line for line: one encoder pass, the pointwise head.

    A conv that reads pairs (``PairConv``, ``conv: pair``, 2026-09-14) also needs the candidates'
    words, ``n_taxa`` and ``m`` in its graph tensors -- ``policy.graph_tensors_for``'s rule, which is
    what training passed -- and raises without them.  Every other conv gets exactly the call it
    always got, so segment checkpoints reproduce their landed cells.
    """
    pairs = net.cfg.layers and getattr(P.CONVS.get(net.cfg.conv), "needs_pairs", False)
    with torch.no_grad():
        g = (P.graph_tensors(graph, gs.support, "cpu", words=gs.words, n_taxa=int(gs.n_taxa),
                             m=float(gs.m))
             if pairs else P.graph_tensors(graph, gs.support, "cpu"))
        enc = net.encode(torch.as_tensor(xn), torch.as_tensor(xg), g)
        return torch.sigmoid(net.cluster_logits(enc)).float().numpy()


def run_cell(row: dict, models: list, args, warm: dict) -> dict:
    phase, rep, arm = row["phase"], int(row["rep"]), row["arm"]
    cell = CAMUS.CamusCell(phase, rep, arm, Path(args.inputs))
    t0 = time.perf_counter()
    truth = CAMUS.truth_of(cell)                  # untimed: k and the taxon order, nothing else
    truth_s = time.perf_counter() - t0
    taxa = list(truth.labels)
    k = int(truth.n_retics)
    taxon_index = {lab: i for i, lab in enumerate(taxa)}

    Path(args.work).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{phase}_rep{rep:02d}_", dir=args.work))
    gz, table = work / "trees.nwk.gz", work / "clusters.parquet"
    clk = Clock()
    try:
        with clk("root"):
            tree_info = CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=args.m,
                                                 partial=args.partial)
        # ``--m-from-trees``: m is the loci actually present (at most --m), so the node features'
        # ``support / m`` stays a frequency on a file with fewer than --m trees (netcs_bench_corpus
        # ships 911-1000).  Off by default, and a no-op on any file of --m or more, so every landed
        # cell reproduces.  ``--m 0`` reads every tree in the file and makes m the count read.
        n_loci = int(tree_info["n_loci"])
        m = n_loci if args.m == 0 else (min(args.m, n_loci) if args.m_from_trees else args.m)
        # tau stays an absolute count, as it was in training, unless --tau-frac scales it with m.
        tau = resolve_tau(m, args.tau_frac)
        with clk("clusters"):
            # --partial: a tree missing taxa contributes its clusters less its own leaf set
            # (extract.count_newick_file's drop_partial_root; infer_gnn_cpu.count_clusters' rule)
            counter = extract.count_newick_file(gz, taxon_index, [m],
                                                drop_partial_root=bool(args.partial))
            masks, support, n_raw = counter.finalize(min_support=1)
            extract.write_cluster_table(table, masks, support, [m], len(taxa), n_raw,
                                        source=str(cell.trees_path), arm=arm)
            words, sup, meta = extract.read_cluster_table(table)
        with clk("candidates"):
            gs = candidates.build(words, sup[:, meta["checkpoints"].index(m)], len(taxa),
                                  m=m, tau=tau)
        with clk("overlap"):
            graph = OverlapCSR.of(gs)
        with clk("tree_stats"):
            ts = I._tree_stats(gz, taxon_index, gs, m)
        with clk("features"):
            xn = I.node_features(gs, graph, ts)
            xg = I.global_features(gs, graph, k)
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    # Scoring, after every shared clock has stopped.  ``cluster_counts`` is called unbound with a
    # two-field stand-in for the instance so there is one definition of F1, not a copy of it:
    # it reads ``truth_rows`` and ``n_true`` and nothing else.
    scorer = SimpleNamespace(truth_rows=T.true_cluster_rows(gs.words, truth),
                             n_true=len(NW.nontrivial_clusters(truth)))

    results = []
    for mdl in models:
        mkey = f"{mdl.tag}/s{mdl.seed}"
        if mkey not in warm:
            # One discarded forward per model per process: the first call pays torch's lazy
            # initialisation, which is a per-process cost and not a per-cell one.  Recorded in
            # the run header rather than dropped on the floor.
            w0 = time.perf_counter()
            forward(mdl.net, xn, xg, graph, gs)
            warm[mkey] = time.perf_counter() - w0
        mc = Clock()
        with mc("forward"):
            p = forward(mdl.net, xn, xg, graph, gs)
        # The formal decoder is the (L)/(N3O) mask realised through the closure; the relaxed one
        # constrains the Hasse diagram itself and realises it with no closure (opc.rl.relaxed).
        # Same scores, same rank rule, same budget k -- only what counts as legal differs.
        relaxed = args.decoder == "relaxed"
        with mc("decode"):
            if relaxed:
                env = RX.RelaxedDecoder(gs.words, len(taxa), k, args.max_level)
                sel = env.decode(p)
            else:
                env = ClusterEnv(gs, budget=k, graph=graph)
                sel = NC.decode(env, p)
        if not env.verify():
            raise AssertionError(f"{cell.key} {mkey}: the {args.decoder} decode fails its own "
                                 f"check{': ' + env.verify_reason if relaxed else ''}")
        enewick, k_pred = None, -1
        if args.network:
            with mc("network"):
                got = (RX.realise if relaxed else search.realise)(gs.words, sel, len(taxa), taxa)
                enewick = None if got is None else NW.to_enewick(got[0])
            k_pred = -1 if got is None else int(got[0].n_retics)
        counts = I.Instance.cluster_counts(scorer, sel)
        ref = args.reference.get((phase, rep, arm, mdl.tag, mdl.seed))
        match = None if ref is None else bool(
            int(ref["n_selected"]) == int(sel.size) and int(ref["tp"]) == counts["tp"]
            and int(ref["fp"]) == counts["fp"])
        entry = {
            "tag": mdl.tag, "seed": mdl.seed, "epoch": mdl.epoch,
            "decoder": args.decoder, "max_level": args.max_level,
            "wall": mc.wall, "cpu": mc.cpu, "n_selected": int(sel.size), **counts,
            "k_pred": k_pred, "enewick": enewick,
            "ref_f1": None if ref is None else float(ref["f1"]),
            "ref_n_selected": None if ref is None else int(ref["n_selected"]),
            "match": match,
            "decode_stats": env.stats if relaxed else None,
        }
        if args.prefix:
            # After every landed field is fixed, and on its own clock (not in PER_MODEL), so the
            # timings and the parity check are what they would have been without --prefix.
            with mc("prefix"):
                entry["prefix"] = prefix_blocks(env, relaxed, gs, graph, taxa, k, p, sel, scorer,
                                                args.max_level)
        if args.tob_manifest:
            # Same discipline as --prefix: after every landed field, on its own clocks.
            entry["tob_resolve"] = tob_blocks(mdl, args.tob_rows[(phase, rep, arm)], gs, graph,
                                              xn, taxa, k, p, sel, scorer,
                                              args.resolve_max_level, args.network)
        results.append(entry)

    return {"phase": phase, "rep": rep, "arm": arm, "key": cell.key,
            "trees": cell.trees_path.name, "n": len(taxa), "k": k,
            "decoder": args.decoder, "max_level": args.max_level,
            "n_loci": int(tree_info["n_loci"]), "m": m, "n_clusters": int(gs.n_clusters),
            "n_edges": int(graph.nbr.size), "threads": THREADS, "truth_read_s": truth_s,
            **sampling_fields(args, tau),
            **({k: tree_info[k] for k in ("partial", "n_partial", "min_taxa", "mean_missing")}
               if args.partial else {}),
            **({"tob_manifest": str(args.tob_manifest),
                "resolve_max_level": args.resolve_max_level} if args.tob_manifest else {}),
            "wall": clk.wall, "cpu": clk.cpu, "models": results}


def sampling_fields(args, tau: int) -> dict:
    """``m_arg``, ``tau`` and ``tau_frac`` for a cell JSON -- only off the defaults (``--m 1000``,
    no ``--tau-frac``), the ``--prefix`` rule, so a default run's cells are what they always were."""
    if args.m == M and args.tau_frac is None:
        return {}
    return {"m_arg": args.m, "tau": tau, "tau_frac": args.tau_frac}


def prefix_blocks(env, relaxed: bool, gs, graph, taxa, k: int, p, sel, scorer,
                  max_level) -> dict:
    """``--prefix``: both reticulation-free prefixes of this decode (``opc.rl.prefix``), scored.

    The decode is run again, traced, and must select exactly what the timed one did -- the relaxed
    one down to its refusal counts.  ``env`` is reused for level1 (a rollout resets it; ``sel`` is
    already a copy), which saves rebuilding the search state.  ``contained_hardwired`` is
    ``fp == 0`` under the landed ``cluster_counts``: every prefix cluster is a cluster of the true
    network.
    """
    n = len(taxa)
    if relaxed:
        dec, trace, pre = PX.prefixes_relaxed(gs.words, n, k, max_level, p, graph)
        same = bool(np.array_equal(trace.selection, np.sort(sel)) and dec.stats == env.stats)
    else:
        trace, pre = PX.prefixes_level1(env, p, graph)
        same = bool(np.array_equal(trace.selection, np.sort(sel)))
    if not same:
        raise AssertionError(f"the traced {trace.decoder} decode diverged from the timed one")
    true = np.zeros(gs.n_clusters, dtype=bool)
    true[np.asarray(scorer.truth_rows, dtype=np.int64)] = True
    out = {}
    for mode, pf in pre.items():
        counts = I.Instance.cluster_counts(scorer, pf.rows)
        stop = pf.stop_row
        out[mode] = {
            **pf.as_dict(), "full_n_selected": int(sel.size),
            "stop_true": None if stop < 0 else bool(true[stop]),
            "stop_cluster": None if stop < 0 else sorted(
                taxa[i] for i in bitset.to_indices(gs.words[stop], n)),
            **counts, "contained_hardwired": counts["fp"] == 0,
            "newick": PX.tree_newick(gs.words, pf.rows, n, taxa),
        }
    return out


def tob_blocks(mdl, rows: list[dict], gs, graph, xn, taxa, k: int, p, sel, scorer,
               max_level: int, network: bool) -> dict:
    """``--tob-manifest``: the GNN as a resolver of each of this cell's trees of blobs
    (``opc.rl.tob_resolve``), keyed ``<source>@<budget_kind>``.

    The scores at budget ``b`` are a forward with ``k = b`` in the global features.  ``p`` (at the
    landed ``k``) is reused when ``b == k``, so the gate row -- the GNN's own TOB at ``k_true`` --
    decodes the landed scores and must reproduce the landed selection (``same_as_landed``).  The
    ``sel_*`` counts are the landed ``cluster_counts`` of the decoded candidate rows; a TOB cluster
    that is not a candidate is in the network but not in them, so the CSV scores the network."""
    n = len(taxa)
    scores = {int(k): p}
    landed = np.sort(np.asarray(sel, dtype=np.int64))
    relaxed_name = TR.mode_name("relaxed", max_level)
    out = {}
    for r in rows:
        b = int(r["budget"])
        c = Clock()
        cached = b in scores
        with c("forward"):
            if not cached:
                scores[b] = forward(mdl.net, xn, I.global_features(gs, graph, b), graph, gs)
        s = scores[b]
        text = Path(r["path"]).read_text().strip()
        with c("restrict"):
            R = TR.restrict(gs, TR.tob_words(text, taxa))
        with c("strict_L1"):
            a = TR.resolve_level1(gs, graph, s, b, R, taxa, network=network)
        with c(relaxed_name):
            x = TR.resolve_relaxed(gs.words, n, s, b, max_level, R, taxa, network=network)
        block = {"source": r["source"], "budget_kind": r["budget_kind"], "budget": b,
                 "k_tob_brooqs4": int(r["k_tob_brooqs4"]), "k_tob_netcs": int(r["k_tob_netcs"]),
                 "tob_sha256": hashlib.sha256(text.encode()).hexdigest(),
                 "forward_cached": cached, **R.as_dict()}
        for name, got in (("strict_L1", a), (relaxed_name, x)):
            counts = I.Instance.cluster_counts(scorer, got["selection"])
            block[name] = {"n_selected": int(got["selection"].size),
                           "k_pred": got.get("k_pred", -1), "enewick": got.get("enewick"),
                           **{f"sel_{m}": counts[m] for m in ("tp", "fp", "fn", "f1")}}
        block["strict_L1"].update(n_steps=a["n_steps"], n_hybrids=a["n_hybrids"],
                                  same_as_landed=bool(np.array_equal(a["selection"], landed)))
        block[relaxed_name]["decode_stats"] = x["stats"]
        block["wall"], block["cpu"] = c.wall, c.cpu
        out[f"{r['source']}@{r['budget_kind']}"] = block
    return out


def tob_summary(run: Path) -> str:
    """One line per (source, mode): rows, median selection F1, median k_pred - budget, and the gate
    row's reproduction count."""
    got: dict[tuple[str, str], list[dict]] = {}
    for f in sorted((run / "cells").glob("*.json")):
        for mr in json.loads(f.read_text())["models"]:
            for sk, b in (mr.get("tob_resolve") or {}).items():
                for mode in (m for m in b if m == "strict_L1" or m.startswith("relaxed_L")):
                    got.setdefault((sk, mode), []).append({**b[mode], "budget": b["budget"]})
    lines = [f"{'tob@budget':18s} {'mode':11s} {'rows':>5s} {'selF1 med':>9s} {'k-b med':>7s} "
             f"{'=landed':>8s}"]
    for (sk, mode), bs in sorted(got.items()):
        same = [x["same_as_landed"] for x in bs if "same_as_landed" in x]
        lines.append(f"{sk:18s} {mode:11s} {len(bs):5d} "
                     f"{statistics.median(x['sel_f1'] for x in bs):9.4f} "
                     f"{statistics.median(x['k_pred'] - x['budget'] for x in bs):7.1f} "
                     f"{(f'{sum(same)}/{len(same)}' if same else ''):>8s}")
    return "\n".join(lines) if got else "tob_resolve: no cell carries tob_resolve blocks"


def prefix_summary(run: Path) -> str:
    """One line per (tag, mode) over the run's cells that carry prefix blocks."""
    got: dict[tuple[str, str], list[dict]] = {}
    for f in sorted((run / "cells").glob("*.json")):
        for mr in json.loads(f.read_text())["models"]:
            for mode, b in (mr.get("prefix") or {}).items():
                got.setdefault((mr["tag"], mode), []).append(b)
    lines = [f"{'prefix':13s} {'tag':22s} {'rows':>5s} {'contained':>10s} {'stopped':>8s} "
             f"{'stop true':>9s} {'size/full':>9s} {'F1 med':>7s}"]
    for (tag, mode), bs in sorted(got.items()):
        stops = [b["stop_true"] for b in bs if b["stopped"]]
        lines.append(
            f"{mode:13s} {tag:22s} {len(bs):5d} "
            f"{sum(b['contained_hardwired'] for b in bs):4d}/{len(bs):<5d} "
            f"{sum(b['stopped'] for b in bs):8d} {sum(stops):4d}/{len(stops):<4d} "
            f"{statistics.median(b['n_selected'] / max(1, b['full_n_selected']) for b in bs):9.3f} "
            f"{statistics.median(b['f1'] for b in bs):7.4f}")
    return "\n".join(lines) if got else "prefix: no cell carries prefix blocks"


def aggregate(run: Path) -> tuple[list[dict], dict]:
    """Every cell JSON in ``run`` -> one row per (cell, model), and the parity tally."""
    rows = []
    for f in sorted((run / "cells").glob("*.json")):
        c = json.loads(f.read_text())
        prep_w = sum(c["wall"].get(s, 0.0) for s in SHARED)
        prep_c = sum(c["cpu"].get(s, 0.0) for s in SHARED)
        for mr in c["models"]:
            model_w = sum(mr["wall"].get(s, 0.0) for s in PER_MODEL)
            model_c = sum(mr["cpu"].get(s, 0.0) for s in PER_MODEL)
            rows.append({
                "phase": c["phase"], "rep": c["rep"], "arm": c["arm"], "n": c["n"], "k": c["k"],
                "n_clusters": c["n_clusters"], "n_edges": c["n_edges"], "threads": c["threads"],
                "tag": mr["tag"], "seed": mr["seed"],
                **{f"{s}_wall": c["wall"].get(s, float("nan")) for s in SHARED},
                **{f"{s}_wall": mr["wall"].get(s, float("nan")) for s in PER_MODEL},
                "prep_wall": prep_w, "model_wall": model_w, "total_wall": prep_w + model_w,
                "prep_cpu": prep_c, "model_cpu": model_c, "total_cpu": prep_c + model_c,
                "n_selected": mr["n_selected"], "tp": mr["tp"], "fp": mr["fp"], "fn": mr["fn"],
                "f1": mr["f1"], "k_pred": mr["k_pred"], "ref_f1": mr["ref_f1"],
                "match": mr["match"],
            })
    checked = [r for r in rows if r["match"] is not None]
    tally = {"pairs": len(rows), "checked": len(checked),
             "matched": sum(bool(r["match"]) for r in checked),
             "mismatched": [(r["phase"], r["rep"], r["tag"], r["seed"], r["f1"], r["ref_f1"])
                            for r in checked if not r["match"]]}
    return rows, tally


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # per-process name: processes sharing a run directory (taxmiss_eval/run_gnn.sh's phases, netcs's
    # shards) each rewrite this at exit, and one shared ".tmp" let one replace another's file mid-write
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    tmp.replace(path)


def summarise(rows: list[dict], tally: dict) -> str:
    lines = [f"{'phase':7s} {'tag':7s} {'cells':>5s} {'n':>4s} {'|C| med':>8s} "
             f"{'prep s':>7s} {'fwd s':>6s} {'dec s':>6s} {'net s':>6s} {'total s':>8s} "
             f"{'cpu s':>7s} {'F1 med':>7s}"]
    keys = sorted({(r["n"], r["phase"], r["tag"]) for r in rows})
    for n, phase, tag in keys:
        g = [r for r in rows if r["phase"] == phase and r["tag"] == tag]
        med = lambda c: statistics.median(r[c] for r in g)          # noqa: E731
        lines.append(
            f"{phase:7s} {tag:7s} {len({r['rep'] for r in g}):5d} {n:4d} {med('n_clusters'):8.0f} "
            f"{med('prep_wall'):7.2f} {med('forward_wall'):6.2f} {med('decode_wall'):6.2f} "
            f"{med('network_wall'):6.2f} {med('total_wall'):8.2f} {med('total_cpu'):7.2f} "
            f"{med('f1'):7.4f}")
    lines.append(f"parity: {tally['matched']}/{tally['checked']} (cell, model) pairs match the "
                 f"landed rows; {tally['pairs'] - tally['checked']} had no reference")
    for m in tally["mismatched"][:20]:
        lines.append(f"  MISMATCH {m[0]} rep{m[1]:02d} {m[2]} s{m[3]}: F1 {m[4]:.4f} "
                     f"vs landed {m[5]:.4f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch and BLAS threads; read before import, and part of the run name")
    ap.add_argument("--inputs", type=Path, default=HERE / "inputs",
                    help="camus-data layout: <inputs>/n<N>/<RR>/{true_net.nwk, g_500.nwk, ...}")
    ap.add_argument("--cells", type=Path, default=HERE / "CELLS.tsv")
    ap.add_argument("--checkpoints", type=Path, default=HERE / "checkpoints")
    ap.add_argument("--reference", type=Path, default=HERE / "reference.tsv")
    ap.add_argument("--tags", default="", help="comma list, e.g. _mix2L; default every family")
    ap.add_argument("--seeds", default="", help="comma list; default every seed shipped")
    ap.add_argument("--models", default="",
                    help="comma list of tag/sN or tag/sN/<stem>, e.g. _l2mix/s1,_l2mixha/s0/best_ha "
                         "(<stem>.pt instead of best.pt, recorded as tag '<tag>:<stem>'); replaces "
                         "--tags and --seeds, whose product cannot name checkpoints that share no "
                         "seed set")
    ap.add_argument("--phases", default="", help="comma list, e.g. cam25,cam200; default all")
    ap.add_argument("--reps", default="", help="comma list or a-b ranges; default all")
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    ap.add_argument("--run-name", default="", help="default <host>_t<threads>")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch for the rooted trees and the cluster table; default <run>/work")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--no-network", dest="network", action="store_false",
                    help="skip realising the network and writing its eNewick")
    ap.add_argument("--decoder", choices=RX.DECODERS, default="level1",
                    help="level1: the formal (L)/(N3O) decoder every landed row used (default); "
                         "relaxed: Hasse-diagram constraints (opc.rl.relaxed), needs --max-level")
    ap.add_argument("--max-level", type=int, default=None,
                    help="the relaxed decoder's level cap, >= 1; required with --decoder relaxed "
                         "and refused without it")
    ap.add_argument("--prefix", action="store_true",
                    help="also record both reticulation-free prefixes of every decode "
                         "(opc.rl.prefix: 'truncated' and 'rank_laminar'), each scored against the "
                         "truth and written as a multifurcating Newick; off leaves every cell JSON "
                         "as it was")
    ap.add_argument("--tob-manifest", type=Path, default=None, metavar="TSV",
                    help="also resolve each cell's trees of blobs listed here "
                         "(tob_resolve_eval/stage.py's tobs.tsv), strict level-1 and relaxed; "
                         "needs --decoder level1 and --resolve-max-level")
    ap.add_argument("--resolve-max-level", type=int, default=None, metavar="L",
                    help="the relaxed resolve's level cap, >= 1; required with --tob-manifest")
    ap.add_argument("--m-from-trees", action="store_true",
                    help="use the file's own locus count (at most --m) as m, so support / m stays "
                         "a frequency on files with fewer trees; off reproduces every landed cell")
    ap.add_argument("--m", type=int, default=M, metavar="N",
                    help=f"read the first N trees of each file (default {M}, as every checkpoint "
                         f"trained); 0 = every tree, with m the count read")
    ap.add_argument("--tau-frac", default=None, metavar="F",
                    help=f"a candidate must be in >= ceil(F * m) of the m trees, F in (0, 1]; "
                         f"training's {TAU} of {M:,} is 0.002.  Default: tau = {TAU}, a count")
    ap.add_argument("--partial", type=int, default=0, metavar="MIN",
                    help="accept gene trees missing taxa (each must keep OUT and >= MIN taxa; one "
                         "below MIN raises, so m is unchanged), dropping each partial tree's own "
                         "leaf set from the clusters it contributes; off (0) reproduces every "
                         "landed cell.  taxmiss_eval runs it at 4")
    ap.add_argument("--force", action="store_true", help="re-run cells that already have a JSON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="resolve the library, load every checkpoint, print versions, exit")
    args = ap.parse_args()
    try:
        decoder = RX.decoder_tag(args.decoder, args.max_level)
    except ValueError as exc:
        raise SystemExit(f"--decoder {args.decoder}: {exc}")
    if args.m < 0:
        raise SystemExit("--m must be >= 0 (0 = every tree in the file)")
    if args.partial and args.partial < 3:
        raise SystemExit("--partial MIN: a rooted cluster needs at least 3 taxa")
    if args.tob_manifest is not None:
        if decoder != "level1":
            raise SystemExit("--tob-manifest runs both resolve modes itself; use --decoder level1")
        if args.resolve_max_level is None or args.resolve_max_level < 1:
            raise SystemExit("--tob-manifest needs --resolve-max-level >= 1")
    elif args.resolve_max_level is not None:
        raise SystemExit("--resolve-max-level means nothing without --tob-manifest")
    if args.tau_frac is not None:
        try:
            resolve_tau(1, args.tau_frac)
        except ValueError as exc:
            raise SystemExit(f"--tau-frac: {exc}")

    # ---- the library must be the one this package ships, not some other ``opc`` on the path
    here_opc = Path(opc.__file__).resolve().parent
    if not here_opc.is_relative_to(LIB.resolve()):
        raise SystemExit(f"opc resolved to {here_opc}, not under {LIB}: another copy of the "
                         f"library is shadowing the one this runner was built against")

    ck_root = Path(args.checkpoints)
    if args.models:
        # Explicit (tag, seed) pairs.  ``--tags a,b`` takes the *union* of seeds across tags and
        # exits on any pair that does not exist, so two new checkpoints such as _l2mix/s1 and
        # _l2mix75/s0 could otherwise only share a cell's preparation by also re-running _l2mix/s0.
        if args.tags or args.seeds:
            raise SystemExit("--models replaces --tags and --seeds; pass one or the other")
        # ``tag/sN`` loads best.pt; ``tag/sN/<stem>`` loads <stem>.pt and records tag ``tag:<stem>``.
        specs = []
        for spec in filter(None, (x.strip() for x in args.models.split(","))):
            tag, _, rest = spec.partition("/")
            s, _, stem = rest.partition("/")
            if (not tag or ":" in tag or not s[:1] == "s" or not s[1:].isdigit()
                    or "/" in stem):
                raise SystemExit(f"--models: {spec!r} is not tag/sN or tag/sN/<checkpoint stem>")
            specs.append((tag, int(s[1:]), stem or "best"))
        models = [m for tag, seed, stem in specs
                  for m in load_models(ck_root, [tag], [seed], stem)]
        names = [m.tag for m in models]
        if len(set(zip(names, (m.seed for m in models)))) != len(models):
            raise SystemExit(f"--models names one checkpoint twice: {args.models}")
    else:
        tags = [t for t in args.tags.split(",") if t] or sorted(
            p.name for p in ck_root.iterdir() if p.is_dir())
        seeds = [int(s) for s in args.seeds.split(",") if s] or sorted(
            {int(p.name[1:]) for t in tags for p in (ck_root / t).iterdir() if p.name[:1] == "s"})
        models = load_models(ck_root, tags, seeds)
    info = machine()

    if args.self_check:
        print(json.dumps(info, indent=1))
        print(f"[gnn-cpu] import {IMPORT_SECONDS:.2f}s | {len(models)} checkpoints load in "
              f"{sum(m.load_s for m in models):.2f}s: "
              f"{', '.join(f'{m.tag}/s{m.seed}@{m.epoch}' for m in models)}")
        print("[gnn-cpu] self-check OK")
        return 0

    cells = read_tsv(args.cells)
    want_phases = {p for p in args.phases.split(",") if p}
    want_reps = parse_reps(args.reps)
    cells = [c for c in cells if (not want_phases or c["phase"] in want_phases)
             and (want_reps is None or int(c["rep"]) in want_reps)]
    if not cells:
        raise SystemExit("no cells selected")
    args.tob_rows = {}
    if args.tob_manifest:
        for r in read_tsv(args.tob_manifest):
            args.tob_rows.setdefault((r["phase"], int(r["rep"]), r["arm"]), []).append(r)
        # A cell with no manifest rows would be run for nothing; a k_true that disagrees means the
        # manifest was staged for other cells.
        bad = [c for c in cells if (c["phase"], int(c["rep"]), c["arm"]) not in args.tob_rows]
        if bad:
            raise SystemExit(f"{len(bad)} selected cells have no row in {args.tob_manifest}, e.g. "
                             f"{[(c['phase'], c['rep'], c['arm']) for c in bad[:3]]}")
        for c in cells:
            for r in args.tob_rows[(c["phase"], int(c["rep"]), c["arm"])]:
                if int(r["k_true"]) != int(c["k"]) or not Path(r["path"]).exists():
                    raise SystemExit(f"manifest row {r['key']} {r['source']}: k_true "
                                     f"{r['k_true']} vs cell k {c['k']}, or no file {r['path']}")
    # The landed rows are level-1 decodes, so a relaxed run has nothing to be at parity with.
    args.reference = ({(r["phase"], int(r["rep"]), r["arm"], r["tag"], int(r["seed"])): r
                       for r in read_tsv(args.reference)}
                      if decoder == "level1" and Path(args.reference).exists() else {})

    suffix = "" if decoder == "level1" else f"_relaxedL{args.max_level}"
    run = Path(args.out) / (args.run_name or f"{socket.gethostname()}_t{THREADS}{suffix}")
    args.work = args.work or run / "work"
    (run / "cells").mkdir(parents=True, exist_ok=True)
    cell_json = lambda c: run / "cells" / f"{c['phase']}_rep{int(c['rep']):02d}_{c['arm']}.json"  # noqa: E731
    if not args.force:
        # One decoder per run directory: a relaxed run resumed over level-1 cells (or the
        # reverse) would skip them as done and report their numbers under the wrong decoder.
        for c in cells:
            if cell_json(c).exists():
                prior = json.loads(cell_json(c).read_text())
                have = RX.decoder_tag(prior.get("decoder", "level1"), prior.get("max_level"))
                if have != decoder:
                    raise SystemExit(f"{cell_json(c)} was decoded as {have}, this run is "
                                     f"{decoder}: pick another --run-name, or --force to overwrite")
                # Likewise --prefix: a cell written without it would be skipped as done and
                # silently lack the blocks this run exists to produce.
                if args.prefix and not all("prefix" in mr for mr in prior["models"]):
                    raise SystemExit(f"{cell_json(c)} has no prefix blocks and this run is "
                                     f"--prefix: pick another --run-name, or --force to overwrite")
                if args.tob_manifest and not all("tob_resolve" in mr for mr in prior["models"]):
                    raise SystemExit(f"{cell_json(c)} has no tob_resolve blocks and this run has "
                                     f"--tob-manifest: pick another --run-name, or --force")
                # Likewise --m / --tau-frac: resume otherwise checks nothing about which trees or
                # which threshold a cell was built on, so a 1,000-tree cell would pass as a
                # 10,000-tree one.  Cells written before the flags existed are (1000, None).
                have_s = (prior.get("m_arg", M), prior.get("tau_frac"))
                if have_s != (args.m, args.tau_frac):
                    raise SystemExit(f"{cell_json(c)} was built with (--m, --tau-frac) = {have_s}, "
                                     f"this run is {(args.m, args.tau_frac)}: pick another "
                                     f"--run-name, or --force to overwrite")
                # And --partial: a strict cell and a partial one read different clusters off the
                # same file (the partial root), so neither may pass for the other.
                if int(prior.get("partial", 0)) != args.partial:
                    raise SystemExit(f"{cell_json(c)} was built with --partial "
                                     f"{prior.get('partial', 0)}, this run is {args.partial}: pick "
                                     f"another --run-name, or --force to overwrite")
    todo = [c for c in cells if args.force or not cell_json(c).exists()]
    print(f"[gnn-cpu] {len(cells)} cells ({len(cells) - len(todo)} done) x {len(models)} models "
          f"| decoder {decoder} | threads {THREADS} | {info['cpu']} "
          f"({info['torch_cpu_capability']}) | -> {run}", flush=True)
    if args.dry_run:
        for c in todo:
            extra = ""
            if args.tob_manifest:
                rs = args.tob_rows[(c["phase"], int(c["rep"]), c["arm"])]
                extra = (f" | {len(rs)} TOBs, budgets {sorted({int(r['budget']) for r in rs})}, "
                         f"{len({int(r['budget']) for r in rs} - {int(c['k'])})} extra forwards")
            print(f"  {c['phase']:7s} rep{int(c['rep']):02d} {c['arm']:10s} n={c['n']} "
                  f"k={c['k']}{extra}")
        if args.tob_manifest:
            extra = sum(len({int(r["budget"]) for r in args.tob_rows[(c["phase"], int(c["rep"]),
                                                                      c["arm"])]} - {int(c["k"])})
                        for c in todo)
            print(f"[gnn-cpu] tob-resolve: {extra} extra forwards per model over {len(todo)} cells "
                  f"(x{len(models)} models)")
        return 0

    warm: dict[str, float] = {}
    t_run = time.perf_counter()
    for i, row in enumerate(todo, 1):
        rec = run_cell(row, models, args, warm)
        dst = run / "cells" / f"{rec['phase']}_rep{rec['rep']:02d}_{rec['arm']}.json"
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(dst)
        prep = sum(rec["wall"].values())
        fwd = statistics.median(m["wall"]["forward"] for m in rec["models"])
        ok = [m["match"] for m in rec["models"] if m["match"] is not None]
        print(f"[gnn-cpu] {i:3d}/{len(todo)} {rec['phase']:7s} rep{rec['rep']:02d} n={rec['n']} "
              f"k={rec['k']} |C|={rec['n_clusters']:6d} | prep {prep:6.2f}s | fwd med "
              f"{fwd:5.2f}s | F1 med "
              f"{statistics.median(m['f1'] for m in rec['models']):.4f} | parity "
              f"{sum(ok)}/{len(ok)}", flush=True)

    header = {**info, "import_seconds": IMPORT_SECONDS,
              "model_load_seconds": {f"{m.tag}/s{m.seed}": m.load_s for m in models},
              "warmup_forward_seconds": warm, "cells_run_this_invocation": len(todo),
              "wall_this_invocation": time.perf_counter() - t_run,
              "args": {k: str(v) for k, v in vars(args).items()
                       if k not in ("reference", "tob_rows") and not (k == "prefix" and not v)
                       and not (k in ("tob_manifest", "resolve_max_level") and v is None)
                       and not (k == "m" and v == M) and not (k == "tau_frac" and v is None)
                       and not (k == "partial" and not v)}}
    (run / f"header_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(header, indent=1))
    rows, tally = aggregate(run)
    write_tsv(run / "timings.tsv", rows)
    text = f"decoder: {decoder}\n" + summarise(rows, tally)
    if args.prefix:
        text += "\n" + prefix_summary(run)
    if args.tob_manifest:
        text += "\n" + tob_summary(run)
    (run / "summary.txt").write_text(text + "\n")
    print(text)
    return 0 if not tally["mismatched"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
