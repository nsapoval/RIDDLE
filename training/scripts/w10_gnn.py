#!/usr/bin/env python
"""W10 -- the supervised node-classification sidecar: train.  Gates G-N1, G-N4.

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/w10_gnn.py --arch gcn --seed 0

Drop the MDP and ask the supervised question directly: for each candidate cluster ``A``, predict
``P(A in C(N_true))``, then decode the per-node scores into a cluster system the way
greedy-by-support decodes support.  No MPL, no reward, no rollout group -- which is what makes it
cheap enough to evaluate at n = 250, where the swap benchmark costs ~17 minutes a cell.

**Four arms, one variable.**  ``mlp`` is ``layers = 0``, a strict features-only path that never
touches an edge; ``gcn``, ``gin`` and ``segment`` stack three layers that differ in **the aggregator
alone** -- residual, ``LayerNorm``, MLP width all held fixed (:data:`opc.rl.policy.CONVS`).  The
``mlp`` arm is not optional: support alone separates ``C(N_true)`` at AUC 0.9918, so a good AUC says
nothing about whether the overlap graph was used.  ``segment`` is the encoder the RL policy already
runs, so it is the architecture baseline rather than a fourth idea.

**Selection is on decoded cluster F1, not on AUC.**  AUC starts the run at 0.99 and moves in the
fourth decimal; the decoded system is what every downstream number is about.  AUC, average
precision, error reduction and precision@``n_true`` are logged every epoch as diagnostics and none
of them picks the checkpoint.

The checkpoint is written with the three keys ``w4_bc.py`` writes -- ``{"policy", "policy_config",
"epoch"}`` -- so the encoder is dual-use: ``w5_train.py --init`` and ``w8_swap.py --init-encoder``
read it unchanged.
"""

from __future__ import annotations

import os

# Before numpy.  This driver runs no solver, so the only thing that could take every core per
# process is the BLAS behind numpy -- ``CLAUDE.md``'s second route into the silent pool hang, and
# the reason ``f3_search.py`` pins the same five variables.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from opc import paths  # noqa: E402
from opc.followup import search as S  # noqa: E402
from opc.rl import bc as BC  # noqa: E402
from opc.rl import hybrid as HY  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import nodecls as NC  # noqa: E402
from opc.rl import partner as PT  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl import train as TR  # noqa: E402
from opc.rl.env import ClusterEnv  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

#: ``--arch`` -> ``(layers, conv)``.  ``mlp`` is the edge-free control; the rest are three layers
#: differing only in how a neighbourhood is summarised.
ARCHS = {"mlp": (0, "segment"), "gcn": (P.DEFAULT_LAYERS, "gcn"),
         "gin": (P.DEFAULT_LAYERS, "gin"), "segment": (P.DEFAULT_LAYERS, "segment"),
         "pair": (P.DEFAULT_LAYERS, "pair")}

#: G-N1: median decoded cluster F1 against greedy-by-support, in-range (b25 + b50).
G_N1_GREEN, G_N1_AMBER = 0.03, 0.0


def cached_keys(results: Path, phase: str, tau: int, arm: str = "") -> list[str]:
    return sorted(p.stem for p in I.cache_dir(results, phase, tau, arm).glob("*.pkl"))


def x_global_of(entry, zero_scale: bool):
    """The instance's global feature vector, optionally with the three size features zeroed.

    Applied here rather than inside :class:`~opc.rl.train.InstancePool` so the cache stays the one
    every other driver shares -- an ablation that silently rewrote a shared cache would change what
    a *concurrent* run computes.
    """
    if not zero_scale:
        return entry["x_global"]
    xg = NC.zero_scale_globals(entry["inst"].x_global)
    return torch.as_tensor(xg, device=entry["x_global"].device)


def x_nodes_of(entry, blocks):
    """The instance's node matrix, optionally with whole W13 blocks zeroed -- W13's ablation.

    Same rule as :func:`x_global_of`: applied here rather than inside the pool, so an ablation can
    never rewrite the cache a concurrent run is reading.  Zeroing rather than narrowing is what
    A96 established as equivalent to removal-and-retrain (A97 measured the two indistinguishable at
    p 0.32-0.73), and it costs no GPU time, which is the whole appeal.
    """
    if not blocks:
        return entry["x_nodes"]
    return NC.zero_node_blocks(entry["x_nodes"], blocks)


@torch.no_grad()
def scores_for(policy, entry, device, zero_scale: bool, blocks=()) -> np.ndarray:
    policy.eval()
    enc = policy.encode(x_nodes_of(entry, blocks), x_global_of(entry, zero_scale), entry["graph"])
    return torch.sigmoid(policy.cluster_logits(enc)).float().cpu().numpy()


#: Above this many training cells the parquet fast path is not a convenience, it is the only
#: viable path: the fallback materialises every one of them before epoch 0, and A89 records that a
#: pool OOM here **hangs at 0% CPU with nothing in the log** rather than failing.  A89's measured
#: cliff.
_POS_WEIGHT_FALLBACK_MAX = 400


def pos_weight_of(results: Path, tau: int, train_cells, pool) -> float:
    """The pooled ``BCEWithLogitsLoss`` positive weight, from the instance parquets if they exist.

    Falls back to :func:`opc.rl.nodecls.pos_weight` over materialised instances -- which is what
    every landed run used -- so the two must agree; a smoke run on a phase with no parquet takes
    the slow path rather than failing.

    **The fallback is loud, and above ``_POS_WEIGHT_FALLBACK_MAX`` cells it is refused.**  It used
    to be silent, and its trigger was a filename it built itself -- which already omitted
    ``--tag``, so any tagged instance build lost the fast path without saying so.  The name now
    comes from :func:`opc.rl.instances.summary_name`, the same function that wrote it.
    """
    pos = tot = 0
    want = {}
    for phase, key, arm in train_cells:
        want.setdefault((phase, arm), set()).add(key)

    def _slow(why: str) -> float:
        if len(train_cells) > _POS_WEIGHT_FALLBACK_MAX:
            raise SystemExit(
                f"[w10] {why}, and the fallback would materialise all {len(train_cells)} training "
                f"instances before epoch 0 -- which A89 records as a silent hang, not a failure. "
                f"Build the summary parquet first:\n"
                + "\n".join(f"  scripts/rl_instances.py --phase {ph} --tau {tau}"
                             + (f" --arm {a}" if a else "")
                             for ph, a in sorted(want)))
        print(f"[w10] {why}; materialising {len(train_cells)} instances for pos_weight")
        return NC.pos_weight([pool.get(*c)["inst"] for c in train_cells])

    for (phase, arm), keys in want.items():
        name = I.summary_name(phase, tau, arm)
        try:
            df = pd.read_parquet(paths.locate(results, name))
        except (FileNotFoundError, OSError):
            return _slow(f"no {name}")
        d = df[df["key"].isin(keys)]
        if len(d) != len(keys):
            return _slow(f"{name} covers {len(d)} of {len(keys)} cells")
        pos += int(d["n_true_in_ctau"].sum())
        tot += int(d["n_clusters"].sum())
    return NC.pos_weight_from_counts(pos, tot)


def decode_term(logits, inst, args, pos_w, rng):
    """The decode-derived loss term for one instance, and whether it was non-zero.

    **A transient ``ClusterEnv``, deliberately not ``pool.env_of``.**  The accessor memoises, and a
    memoised env on 1,400 train cells is the ~24 GiB that ``--lazy-env`` exists to avoid.  Built
    and discarded costs a measured 19 ms at n = 25 and 46 ms at n = 50, which is the price of the
    arm.  It also keeps the env out of the pool entry, so nothing here can reset a shared
    :class:`~opc.rl.swap.SwapEnv` out from under a walk.

    The trace is taken from ``logits.detach()``: the decode is a constant, and only the *values*
    at the traced indices carry gradient.
    """
    y = NC.labels(inst)
    if args.decode_loss == "band":
        return NC.band_loss_terms(logits, y, rng=rng), True
    env = ClusterEnv(inst.gs, budget=int(inst.k), graph=inst.graph)
    cap = int(np.asarray(inst.truth_rows).size) if args.decode_cap == "n_true" else None
    scores = torch.sigmoid(logits.detach()).float().cpu().numpy().astype(np.float64)
    sel, trace = NC.decode_trace(env, scores, y, cap=cap)
    term = NC.decode_loss_terms(logits, trace, y, args.decode_loss,
                                pos_weight=pos_w, selection=sel)
    return term, bool(float(term.detach()) != 0.0)


#: ``--val-decoder auto`` uses the relaxed decoder on a cell whose truth is above this level.
_LEVEL1 = 1


def _truth_level(inst) -> int:
    """``networks.level`` of the instance's truth, cached on the instance.

    Cheap (microseconds on these networks) but called once per validation cell per epoch, and the
    attribute is set on the live object rather than in the pickle so no cache is invalidated --
    ``check_features`` is about array *widths* and must stay that way.
    """
    lvl = getattr(inst, "_truth_level", None)
    if lvl is None:
        lvl = int(N.level(inst.truth))
        object.__setattr__(inst, "_truth_level", lvl)
    return lvl


def relaxed_decode(inst, scores, max_level: int) -> np.ndarray:
    """:class:`opc.rl.relaxed.RelaxedDecoder` on one instance's candidate set.

    A fresh decoder per call: it holds an ``|C| x cap`` intersection matrix and a ``cap x cap``
    cover matrix that ``decode`` resets anyway, and keeping one per validation cell resident would
    be the same mistake ``--lazy-env`` exists to avoid.  Measured at 0.04-0.10 s a cell at
    n = 25/50, against ~800 s an epoch.
    """
    dec = RX.RelaxedDecoder(inst.gs.words, int(inst.n), int(inst.k), int(max_level))
    return dec.decode(np.asarray(scores, dtype=np.float64))


def truth_ha_of(memo, ident, inst) -> HY.TruthHA:
    """:func:`opc.rl.hybrid.truth_ha`, memoised per ``(phase, key, arm)`` -- a few hundred bytes a
    cell, so like ``greedy_memo`` it lives in the driver and not in the LRU'd pool entry."""
    if memo is None:
        return HY.truth_ha(inst.truth, inst.gs.words)
    if ident not in memo:
        memo[ident] = HY.truth_ha(inst.truth, inst.gs.words)
    return memo[ident]


def pred_ancestry(inst, selection, relaxed: bool):
    """``P`` for a decoded selection, read off the network that decoder's selections stand for.

    Relaxed: ``H(S + trivial)``, no closure.  Formal: :func:`opc.followup.search.realise`'s closed
    system, and ``None`` where it builds no network -- make_csv's ``no_network`` rows, which carry
    no hybrid ancestry either.
    """
    n = int(inst.n)
    if relaxed:
        return HY.ancestry(HY.selection_full(inst.gs.words, selection, n), n)
    got = S.realise(inst.gs.words, selection, n)
    return None if got is None else HY.ancestry(got[1], n)


def ha_chunk(policy, pool, chunk, device, args, truth_memo, ha_pool):
    """Relaxed-decode every instance of a chunk at the parameters its gradient is taken at (A108).

    **Exact on-policy, and that is why it is a chunk at a time.**  ``opt.step`` runs once per chunk,
    so a ``no_grad`` forward here gives the same logits the grad forward below will, up to GPU
    nondeterminism in the last bits (dropout is 0).  The decodes then run in parallel: sequentially
    they cost 0.03-0.14 s a cell, ~800 s on a 700 s epoch.

    **``.get(timeout=)`` is not decoration.**  A89: a pool whose worker dies waits for the dead task
    for ever, at 0% CPU, with nothing in the log.  A timeout turns that into a traceback.
    """
    # Submitted the moment each forward is done, so the decodes run WHILE the remaining forwards
    # do.  Collecting the scores first and mapping afterwards measured +0.09 s a cell at n = 50 --
    # most of a 700 s epoch -- because the chunk then waited on the forwards and the decodes in
    # series.
    handles = []
    for ident in chunk:
        e = pool.get(*ident)
        inst = e["inst"]
        p = scores_for(policy, e, device, args.zero_scale_globals, args.zero_node_block)
        task = (inst.gs.words, int(inst.n), int(inst.k), int(args.ha_max_level), p,
                inst.truth_rows, truth_ha_of(truth_memo, ident, inst))
        handles.append(ha_pool.apply_async(HY.decode_and_target, (task,)))
    policy.train()                                       # scores_for leaves the policy in eval()
    return [h.get(timeout=args.ha_timeout) for h in handles]


#: Per-cell training statistics averaged onto the epoch row as ``train_<name>``.
_HA_TRAIN_STATS = ("ha_fnr", "ha_fpr", "ha_ber", "ha_fn_dropped", "ha_fp_dropped", "ha_n_up",
                   "ha_n_down")


def evaluate(policy, pool, cells, device, zero_scale: bool, blocks=(), *,
             val_decoder: str = "level1", max_level: int = 2, greedy_memo=None,
             truth_memo=None, row_hook=None) -> pd.DataFrame:
    """Per validation cell: the decoded system, greedy's, and the ranking diagnostics.

    ``env.reset()`` is *not* called here -- :func:`opc.rl.nodecls.decode` and
    :func:`opc.rl.env.greedy_support_rollout` both go through :func:`opc.rl.env.rollout`, which
    resets.  The two share one ``ClusterEnv`` per cell for exactly the reason the pool builds one:
    a second env is a second chance for the two selections to be about different objects.

    **``val_decoder="auto"`` picks the decoder per cell, on the truth's level.**  The formal
    decoder masks by (L)/(N3O), which characterise *level-1* clustering systems: on the 20
    exactly-level-2 networks of ``l2reg_n200_v1`` (L) fails on the true cluster set of 20 of 20 and
    the closure breaks the network on 10 of 20, so ``f1 = 1`` is unreachable there by construction
    and a validation curve on such a cell is measuring the mask.  A level-2 cell therefore goes
    through :mod:`opc.rl.relaxed` at ``max_level``; a level-1 cell goes through the formal decoder
    **unchanged**, so every x/y/b phase reproduces its landed numbers bit for bit.

    **Greedy goes through the same decoder as the model on the same cell.**  ``f1_vs_greedy`` built
    from two different decoders is not a contrast, it is a decoder comparison wearing a model's
    name.  This is what ``l2reg_eval/run_greedy.py --decoder relaxed`` does on the far side.

    ``greedy_memo`` caches the relaxed greedy selection per ``(phase, key, arm)``: it depends only
    on ``gs.support``, so it is constant across epochs.  It is **not** put in the pool entry, which
    the LRU evicts.

    ``row_hook``, when given, is called once per cell with the objects this loop already holds --
    ``cell``, ``inst``, ``pool``, ``p``, ``sel``, ``greedy``, ``relaxed``, ``max_level`` -- and the
    dict it returns is merged into that cell's row.  It exists so an eval-only driver
    (``scripts/w10_gnn_valeval.py``) can add slices without a second forward pass or a second copy
    of this loop.  ``None`` -- every training call -- adds nothing.
    """
    rows = []
    for phase, key, arm in cells:
        e = pool.get(phase, key, arm)
        inst = e["inst"]
        p = scores_for(policy, e, device, zero_scale, blocks)
        y = NC.labels(inst)
        lvl = _truth_level(inst)
        use_relaxed = (val_decoder == "auto" and lvl > _LEVEL1)
        if use_relaxed:
            # No ``pool.env_of`` on this path: under ``--lazy-env`` that accessor is the only thing
            # that would build a ClusterEnv for the cell, and a resident env costs more than the
            # instance it came from.
            sel = relaxed_decode(inst, p, max_level)
            memo = greedy_memo if greedy_memo is not None else {}
            gk = (phase, key, arm)
            if gk not in memo:
                memo[gk] = relaxed_decode(
                    inst, np.asarray(inst.gs.support, dtype=np.float64), max_level)
            greedy = memo[gk]
        else:
            # Through the accessors, not ``e["env"]``: under ``--lazy-env`` the entry
            # holds ``None`` until something asks, and the validation cells are the
            # only things that ever ask.
            env = pool.env_of(phase, key, arm)
            greedy = pool.greedy_of(phase, key, arm)
            sel = NC.decode(env, p)
        got, base = inst.cluster_counts(sel), inst.cluster_counts(greedy)
        # Hybrid ancestry (A108), each selection read the way its decoder's networks are built.
        T = truth_ha_of(truth_memo, (phase, key, arm), inst).ancestry
        ha = HY.rates(pred_ancestry(inst, sel, use_relaxed), T)
        ha_g = HY.rates(pred_ancestry(inst, greedy, use_relaxed), T)
        a_head, a_sup = BC.auc(p, y), BC.support_auc(inst)
        rows.append({
            "phase": phase, "key": key, "arm": arm or "true",
            "n": int(inst.n), "k": int(inst.k),
            "truth_level": lvl,
            "decoder": RX.decoder_tag("relaxed", max_level) if use_relaxed else "level1",
            "truth_coverage": float(inst.truth_coverage),
            "f1": got["f1"], "f1_greedy": base["f1"], "f1_vs_greedy": got["f1"] - base["f1"],
            "fp": got["fp"], "fn": got["fn"], "fp_greedy": base["fp"], "fn_greedy": base["fn"],
            "exact": bool(got["fp"] == 0 and got["fn"] == 0),
            "ha_fnr": ha["ha_fnr"], "ha_fpr": ha["ha_fpr"], "ha_ber": ha["ha_ber"],
            "ha_fnr_greedy": ha_g["ha_fnr"], "ha_fpr_greedy": ha_g["ha_fpr"],
            "ha_ber_greedy": ha_g["ha_ber"],
            "auc_head": a_head, "auc_support": a_sup,
            "auc_gain": a_head - a_sup,
            "auc_error_reduction": NC.error_reduction(a_head, a_sup),
            "average_precision": NC.average_precision(p, y),
            "ap_support": NC.average_precision(np.asarray(inst.gs.support, float), y),
            "precision_at_n_true": NC.precision_at_k(p, y, int(np.asarray(inst.truth_rows).size)),
        })
        if row_hook is not None:
            rows[-1].update(row_hook(cell=(phase, key, arm), inst=inst, pool=pool, p=p, sel=sel,
                                     greedy=greedy, relaxed=use_relaxed, max_level=max_level))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCHS), default="gcn")
    ap.add_argument("--phases", default="b25,b50", help="comma-separated training phases")
    ap.add_argument("--arms", default="true",
                    help="comma-separated gene-tree arms, CROSSED with --phases.  'true' is the "
                         "simulated trees; 'est_bp1000' the IQ-TREE re-estimates.  Naming both is "
                         "2x the data -- every training cell contributes its instance under each, "
                         "and they are different objects: a different candidate set, a different "
                         "overlap graph and a different node-feature matrix for the same truth")
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="instances per optimiser step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=P.DEFAULT_HIDDEN)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-cells", type=int, default=0, help="0 = every train cell; smoke only")
    # **Inert since A97**, which removed those three features from `GLOBAL_FEATURES` outright
    # rather than leaving a run-time switch. Kept so the flag is a loud refusal rather than an
    # unrecognised argument, which is what a queue script passing it would otherwise hit.
    ap.add_argument("--zero-scale-globals", action="store_true",
                    help="A96's ablation; REMOVED in A97 and now refuses -- it is the default")
    # W13's per-block ablation.  Repeatable, so `--zero-node-block blen --zero-node-block brank`
    # prices the two length blocks together.  The indices come from `nodecls.block_idx`, which
    # derives them by *name* -- A48's rule, and A97's near-miss where a literal stayed in range.
    ap.add_argument("--zero-node-block", action="append", default=[],
                    choices=sorted(I.NODE_BLOCKS),
                    help="zero a whole W13 node-feature block; repeatable. Zeroing at run time "
                         "reproduces removal-and-retrain (A96/A97) at no GPU cost")
    # **Not optional on a b-grid.**  ``split_by_rep``'s defaults are the S-grids' twenty
    # replicates; on a hundred they put 15 cells in train and 82 in test, and nothing fails.
    ap.add_argument("--split-train", type=int, default=70)
    ap.add_argument("--split-val", type=int, default=80)
    ap.add_argument("--pool-size", type=int, default=192,
                    help="instances held resident; each keeps its graph on the GPU")
    # **Not optional above ~400 cells.**  A resident ``ClusterEnv`` costs more than the instance it
    # is built from (10.5 MiB against 8.7 at n = 25; 20.5 against 18.7 at n = 50), because
    # ``heuristics._csr`` keeps a second, int64 copy of the overlap graph.  Eager construction at
    # the 1,600 cells the x-grids give is ~45.6 GiB against ~46 GB free, and A89 is the record of
    # what an OOM does to a pool here: it hangs, at 0% CPU, with nothing in the log.  A pointwise
    # BCE never touches a train cell's env -- only ``evaluate()`` does, on the val cells.
    ap.add_argument("--lazy-env", action="store_true",
                    help="build ClusterEnv/greedy on first access (val cells only under BCE)")
    # ---- W10 v2: a loss term computed from the decode's own selection ------------------------
    # A91 located the barrier to exact recovery in the *ordering*: 90-98% of wrong picks were
    # avoidable, with the right cluster a median 2 ranks below.  A92 then showed that cloning that
    # ordering with a sequential decoder fails.  These terms keep the static head and re-weight it
    # from what the decode did.  The selection is a CONSTANT inside the term -- a decode-conditioned
    # reweighting, not a policy gradient.
    ap.add_argument("--decode-loss", choices=("none", "step", "set", "band", "ha"), default="none",
                    help="step: listwise CE at avoidable errors | set: BCE on the decode's "
                         "mistakes | band: pairwise margin in the confusable band, no decode | "
                         "ha: balanced hybrid-ancestry pressure on a relaxed decode (A108)")
    # ---- A108: `--decode-loss ha` ----------------------------------------------------------------
    # The training decode is the RELAXED decoder on every cell, level-1 truths included: that is
    # what deployment on l2reg runs, so it is the selection whose hybrids the benchmark scores.
    # Validation keeps `--val-decoder`, so selection stays comparable with `_l2mix`.
    ap.add_argument("--ha-max-level", type=int, default=2,
                    help="the relaxed training decode's level cap under --decode-loss ha")
    # One per instance of the default --batch 8, so a chunk's decodes never queue behind each other.
    ap.add_argument("--ha-jobs", type=int, default=8,
                    help="forked decode workers under --decode-loss ha")
    ap.add_argument("--ha-timeout", type=float, default=300.0,
                    help="seconds a chunk's decodes may take before the run fails loudly (A89)")
    ap.add_argument("--decode-weight", type=float, default=0.0, help="lambda on the term")
    ap.add_argument("--decode-warmup", type=int, default=5,
                    help="epochs of pure BCE first; a decode from a random init teaches nothing")
    # Measured INERT on this grid and kept anyway: the decode already runs 24.3 steps at n = 25
    # against a median n_true of 24, because greedy is maximal at the truth's cardinality (A71).
    ap.add_argument("--decode-cap", choices=("n_true", "none"), default="n_true",
                    help="stop the traced decode after n_true_in_ctau accepts")
    # **The lever-2 arms fine-tune from a converged lever-1 checkpoint rather than train from
    # scratch**, for two reasons.  A decode-derived term needs a model whose decode means
    # something -- that is what ``--decode-warmup`` exists for, and starting from a trained
    # checkpoint is the same idea done properly.  And it isolates the term: the control is the
    # *same* checkpoint given the *same* extra epochs with ``--decode-loss none``, so the
    # comparison cannot be confounded with more training.
    # **The default is `level1`, which is today's behaviour exactly**, so every landed run and
    # every phase whose truths are level-1 reproduces bit for bit.  `auto` is the only other
    # setting on purpose: "relaxed everywhere" would silently change the x/y/b validation numbers
    # that `_mixL` and `_mix2L` were selected on, and the relaxed decoder at max_level 1 is NOT the
    # formal level-1 decoder (no closure, and a budget counted on the diagram rather than on
    # intersection classes), so the two are not interchangeable even where both are legal.
    ap.add_argument("--val-decoder", choices=("level1", "auto"), default="level1",
                    help="auto: relaxed decoder on validation cells whose truth is above level 1, "
                         "the formal one everywhere else")
    ap.add_argument("--val-max-level", type=int, default=2,
                    help="the relaxed decoder's level cap under --val-decoder auto")
    # ---- S1-S3: `--arch pair` (A108 addendum) ----------------------------------------------------
    ap.add_argument("--no-containment", action="store_true",
                    help="pair only: aggregate over the crossing relation alone (the S1 ablation)")
    ap.add_argument("--edge-weight", type=float, default=0.25,
                    help="pair only: lambda on the co-parent BCE that supervises every layer's "
                         "attention gates; 0 leaves the attention unsupervised")
    ap.add_argument("--edge-hard", type=int, default=16,
                    help="pair only: best-supported non-partners per positive query")
    ap.add_argument("--edge-random", type=int, default=2048,
                    help="pair only: uniformly drawn entries per instance per step")
    ap.add_argument("--init", type=Path, default=None,
                    help="warm-start the policy from a checkpoint (w4_bc.py's three-key format)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ha_pool = None
    if args.decode_loss == "ha":
        # A term that can never fire is a run that looks like the arm and is the control.
        if args.decode_weight <= 0:
            print("[w10] --decode-loss ha needs --decode-weight > 0; at 0 it never fires")
            return 2
        # **fork, created before anything puts a tensor on the GPU.**  The workers are pure numpy
        # (``opc.rl.hybrid`` imports no torch), so they never touch the CUDA context they inherit;
        # ``spawn`` would instead re-import this script -- torch, pandas and all -- in each of them.
        import multiprocessing as mp
        ha_pool = mp.get_context("fork").Pool(max(1, args.ha_jobs), initializer=HY.worker_init)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.zero_scale_globals and not NC.SCALE_GLOBAL_IDX:
        print(f"[w10] --zero-scale-globals: A97 removed {', '.join(I.LEGACY_SIZE_FEATURES)} from "
              f"GLOBAL_FEATURES, so there is nothing to zero and the ablation is now the default. "
              f"Re-run without the flag.")
        return 2
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    # "true" is the CLI spelling of the empty arm id; `--arms ,est_bp1000` is unreadable.
    arms = [("" if a.strip() == "true" else a.strip())
            for a in args.arms.split(",") if a.strip()]
    if not arms:
        print("[w10] --arms is empty")
        return 2

    keys_by_phase, train_cells, val_cells = {}, [], []
    for phase in phases:
        for arm in arms:
            keys = cached_keys(args.out, phase, args.tau, arm)
            if not keys:
                print(f"[w10] no instances cached for {phase} tau={args.tau} "
                      f"arm={arm or 'true'}; run scripts/rl_instances.py --phase {phase}"
                      + (f" --arm {arm}" if arm else ""))
                return 2
            keys_by_phase[phase] = keys
            sp = TR.split_by_rep(keys, args.split_train, args.split_val)
            train_cells += [(phase, k, arm) for k in sp["train"]]
            val_cells += [(phase, k, arm) for k in sp["val"]]

    # **The split is arm-blind, which is required and therefore worth checking.**  Both arms of a
    # cell must land on the same side or a validation cell's twin -- same truth network, same
    # topology, the same clusters to find -- is in train.  ``split_by_rep`` keys on the replicate
    # in the cell key and cannot see the arm, so this holds by construction; an assert is what
    # keeps it holding when someone gives an arm its own split.
    for phase in phases:
        tr = {k for ph, k, _ in train_cells if ph == phase}
        va = {k for ph, k, _ in val_cells if ph == phase}
        if tr & va:
            print(f"[w10] {phase}: {len(tr & va)} cell keys are in BOTH train and val across "
                  f"arms -- the same truth network on both sides of the split")
            return 2
    if args.max_cells:
        train_cells, val_cells = train_cells[:args.max_cells], val_cells[:max(2, args.max_cells)]
    # **A silent failure if it is not checked.**  With no val cell, ``evaluate`` returns an empty
    # frame, ``val_f1`` is NaN, ``NaN > -1.0`` is False at every epoch, so ``best.pt`` is never
    # written and the verdict reports ``.iloc[-1]`` as the best epoch.  The run looks finished.
    if not train_cells or not val_cells:
        print(f"[w10] empty split: {len(train_cells)} train / {len(val_cells)} val cells for "
              f"--split-train {args.split_train} --split-val {args.split_val}.  Are the instances "
              f"built for every replicate?  (rl_instances.py --reps is not optional)")
        return 2

    layers, conv = ARCHS[args.arch]
    run_id = f"{args.arch}_s{args.seed}{args.tag}"
    print(f"[w10] arch={args.arch} (layers={layers} conv={conv}) seed={args.seed} "
          f"phases={'+'.join(phases)} arms={'+'.join(a or 'true' for a in arms)} "
          f"tau={args.tau}: {len(train_cells)} train / "
          f"{len(val_cells)} val cells, device={device}"
          f"{' [zero-scale-globals]' if args.zero_scale_globals else ''}"
          f"{' [zero ' + ','.join(args.zero_node_block) + ']' if args.zero_node_block else ''}",
          flush=True)

    pool = TR.InstancePool(keys_by_phase, args.out, args.tau, device,
                           size=max(args.pool_size, args.batch * 2), lazy_env=args.lazy_env,
                           pair_graph=(conv == "pair" and layers > 0))
    probe = pool.get(*train_cells[0])["inst"]
    cfg = P.PolicyConfig(n_node_features=probe.x_nodes.shape[1],
                         n_global_features=probe.x_global.shape[0],
                         hidden=args.hidden, layers=layers, conv=conv, dropout=args.dropout,
                         containment=not args.no_containment)
    policy = P.Policy(cfg).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        # **Refuse a mismatch rather than let ``strict=False`` quietly drop layers.**  The arms
        # differ in the aggregator, and a `gcn` checkpoint loaded into a `segment` policy would
        # share every parameter name it has -- the run would train, converge, and be about the
        # wrong architecture.
        got = P.PolicyConfig(**ck["policy_config"])
        for field in ("conv", "layers", "hidden", "n_node_features", "n_global_features"):
            if getattr(got, field) != getattr(cfg, field):
                print(f"[w10] --init mismatch on {field}: checkpoint {getattr(got, field)!r} "
                      f"vs this run {getattr(cfg, field)!r}")
                return 2
        policy.load_state_dict(ck["policy"])
        print(f"[w10] warm-started from {args.init} (its epoch {ck.get('epoch')})", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Pooled over the training set, not per instance: the rate is 2.3-2.4% at every taxon count,
    # so a per-instance weight would make the loss scale differ between graphs for no reason.
    # **Read from the instance parquets, not by materialising the pool.**  The two counts are
    # already columns there, and forcing 1,400 cells through ``pool.get`` for two scalars sets the
    # run's peak-memory watermark before epoch 0 and builds an env per cell that nothing will use.
    w_pos = pos_weight_of(args.out, args.tau, train_cells, pool)
    pos_w = torch.tensor(w_pos, device=device)
    print(f"[w10] policy: {sum(q.numel() for q in policy.parameters()):,} parameters | "
          f"pos_weight {w_pos:.1f} (positive rate {1 / (1 + w_pos):.3%})", flush=True)

    run_dir = paths.artifact(args.out, f"rl_run_gnn_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"arch": args.arch, "tag": args.tag, "policy_config": vars(cfg),
         # **Recorded, not inferred from the tag.**  A94: ``arch`` was written as ``args.arch``
         # with no record of ``--tag``, so a retrained arm merged with the landed one and only the
         # row count would have shown it.  What a checkpoint was trained on is that same class of
         # fact; ``w10_gnn_eval.py`` copies this onto every evaluation row as ``train_arms``.
         "arms": [a or "true" for a in arms],
         "phases": phases, "tau": args.tau,
         "seed": args.seed, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
         "weight_decay": args.weight_decay, "pos_weight": w_pos,
         "zero_scale_globals": bool(args.zero_scale_globals),
         "zero_node_block": list(args.zero_node_block),
         "decode_loss": args.decode_loss, "decode_weight": args.decode_weight,
         "containment": (not args.no_containment) if conv == "pair" else None,
         "edge_weight": float(args.edge_weight) if conv == "pair" else None,
         "edge_hard": int(args.edge_hard) if conv == "pair" else None,
         "edge_random": int(args.edge_random) if conv == "pair" else None,
         "decode_warmup": args.decode_warmup, "decode_cap": args.decode_cap,
         "ha_train_decoder": (RX.decoder_tag("relaxed", args.ha_max_level)
                              if args.decode_loss == "ha" else None),
         "ha_jobs": int(args.ha_jobs) if args.decode_loss == "ha" else None,
         "val_decoder": args.val_decoder, "val_max_level": int(args.val_max_level),
         "init": str(args.init) if args.init else None,
         "split": [args.split_train, args.split_val],
         "n_train": len(train_cells), "n_val": len(val_cells)}, indent=1))

    # Keyed on (phase, key, arm) and built once: the relaxed greedy selection is a function of
    # `gs.support` alone, so it is the same at every epoch.  Deliberately not in the pool entry --
    # the LRU evicts those, and this is a few hundred int64 per cell.
    greedy_memo: dict = {}
    truth_memo: dict = {}
    rng = np.random.default_rng(args.seed)
    # A **separate** stream for the loss term's own sampling.  Sharing ``rng`` would make the
    # epoch's shuffle depend on how many pairs ``band`` drew, so two arms differing only in
    # ``--decode-loss`` would also differ in the order they saw the data -- a confound that costs
    # nothing to remove and would be invisible in the logs.
    loss_rng = np.random.default_rng(args.seed + 10_000)
    # The co-parent BCE's own stream, for the same reason: no other arm's data order may depend on
    # how many entries this one drew.
    edge_rng = np.random.default_rng(args.seed + 20_000)
    use_edge = conv == "pair" and layers > 0 and args.edge_weight > 0
    edge_memo: dict = {}
    log, best = [], (-1.0, -1)
    best_ha = (float("inf"), -1)
    ha_sel_col = f"val_ha_ber_{RX.decoder_tag('relaxed', args.val_max_level)}"
    for epoch in range(args.epochs):
        policy.train()
        order = list(train_cells)
        rng.shuffle(order)
        t0, losses = time.perf_counter(), []
        use_decode = (args.decode_loss != "none" and args.decode_weight > 0
                      and epoch >= args.decode_warmup)
        use_ha = use_decode and args.decode_loss == "ha"
        d_terms, d_fired, ha_stats = [], 0, []
        e_terms, e_missing = [], 0
        for i in range(0, len(order), args.batch):
            chunk = order[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            ha_out = (ha_chunk(policy, pool, chunk, device, args, truth_memo, ha_pool)
                      if use_ha else None)
            # **Backward per instance; ``--batch`` is gradient accumulation.**  ``w4_bc.py`` records
            # why: the retained activations are small but the ``(2|E|, hidden)`` gather is not, and
            # several instances' peaks overlap where several sequential backwards' do not.
            for j, (phase, key, arm) in enumerate(chunk):
                e = pool.get(phase, key, arm)
                enc = policy.encode(x_nodes_of(e, args.zero_node_block),
                                    x_global_of(e, args.zero_scale_globals), e["graph"],
                                    keep_layer_inputs=use_edge)
                y = torch.zeros(e["inst"].n_clusters, device=device)
                y[torch.as_tensor(np.asarray(e["inst"].truth_rows, dtype=np.int64),
                                  device=device)] = 1
                logits = policy.cluster_logits(enc)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_w)
                losses.append(float(loss.detach()))
                if use_edge:
                    ident = (phase, key, arm)
                    if ident not in edge_memo:
                        edge_memo[ident] = PT.edge_targets(
                            e["inst"], truth_ha_of(truth_memo, ident, e["inst"]), args.edge_hard)
                    tg = edge_memo[ident]
                    e_missing += tg.n_missing
                    idx, lab = PT.sample_edges(tg, int(e["graph"].src.numel()), args.edge_random,
                                               edge_rng)
                    if idx.size:
                        n_pos = float(lab.sum())
                        pw_e = torch.tensor(max(1.0, (lab.size - n_pos) / n_pos) if n_pos else 1.0,
                                            device=device)
                        lab_t = torch.as_tensor(lab, device=device)
                        el = torch.stack([
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                z, lab_t, pos_weight=pw_e)
                            for z in policy.edge_logits(enc, e["graph"],
                                                        torch.as_tensor(idx, device=device))
                        ]).mean()
                        loss = loss + args.edge_weight * el
                        e_terms.append(float(el.detach()))
                if use_ha:
                    tg = ha_out[j][1]
                    extra, fired = HY.ha_loss_terms(logits, tg), bool(tg.rows.size)
                    ha_stats.append(tg.stats)
                    loss = loss + args.decode_weight * extra
                    d_terms.append(float(extra.detach()))
                    d_fired += int(fired)
                elif use_decode:
                    extra, fired = decode_term(logits, e["inst"], args, pos_w, loss_rng)
                    loss = loss + args.decode_weight * extra
                    d_terms.append(float(extra.detach()))
                    d_fired += int(fired)
                (loss / len(chunk)).backward()
                del enc, loss, logits
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        df = evaluate(policy, pool, val_cells, device, args.zero_scale_globals,
                      args.zero_node_block, val_decoder=args.val_decoder,
                      max_level=args.val_max_level, greedy_memo=greedy_memo,
                      truth_memo=truth_memo)
        row = {"epoch": epoch, "arch": args.arch, "seed": args.seed, "conv": conv,
               "layers": layers, "loss": float(np.mean(losses)),
               "decode_loss": args.decode_loss, "decode_weight": float(args.decode_weight),
               "decode_term": float(np.mean(d_terms)) if d_terms else float("nan"),
               "decode_fired": int(d_fired),
               "edge_weight": float(args.edge_weight) if conv == "pair" else float("nan"),
               "edge_loss": float(np.mean(e_terms)) if e_terms else float("nan"),
               "edge_missing": int(e_missing),
               **({f"train_{c}": float(pd.Series([s[c] for s in ha_stats], dtype=float).mean())
                   for c in _HA_TRAIN_STATS} if ha_stats else {}),
               # Hybrid ancestry on the validation decodes (A108): diagnostics, and `best_ha.pt`'s
               # criterion -- never `best.pt`'s.  NaN cells (empty P, or a k = 0 truth) are skipped.
               "val_ha_ber": float(df["ha_ber"].mean()),
               "val_ha_ber_greedy": float(df["ha_ber_greedy"].mean()),
               "val_ha_fnr": float(df["ha_fnr"].mean()), "val_ha_fpr": float(df["ha_fpr"].mean()),
               **{f"val_ha_ber_{d}": float(df.loc[df["decoder"] == d, "ha_ber"].mean())
                  for d in sorted(df["decoder"].unique())},
               **{f"val_ha_ber_greedy_{d}": float(df.loc[df["decoder"] == d, "ha_ber_greedy"].mean())
                  for d in sorted(df["decoder"].unique())},
               "val_f1": float(df["f1"].mean()), "val_f1_greedy": float(df["f1_greedy"].mean()),
               # **`val_f1` is the selection quantity and stays the mean over every cell.**  These
               # two split it by decoder, which is what makes a mixed-decoder run comparable with
               # one that had no level-2 cells at all: `_mixL` and `_mix2L` are level1-only, so
               # `val_f1_level1` is the column to read across the ladder, not `val_f1`.
               **{f"val_f1_{d}": float(df.loc[df["decoder"] == d, "f1"].mean())
                  for d in sorted(df["decoder"].unique())},
               **{f"val_cells_{d}": int((df["decoder"] == d).sum())
                  for d in sorted(df["decoder"].unique())},
               **{f"val_f1_{a}": float(df.loc[df["arm"] == a, "f1"].mean())
                  for a in sorted(df["arm"].unique())},
               **{f"val_f1_vs_greedy_{a}": float(df.loc[df["arm"] == a, "f1_vs_greedy"].mean())
                  for a in sorted(df["arm"].unique())},
               "val_f1_vs_greedy": float(df["f1_vs_greedy"].mean()),
               "val_f1_vs_greedy_median": float(df["f1_vs_greedy"].median()),
               "val_exact": int(df["exact"].sum()),
               "val_auc_head": float(df["auc_head"].median()),
               "val_auc_support": float(df["auc_support"].median()),
               "val_auc_error_reduction": float(df["auc_error_reduction"].median()),
               "val_average_precision": float(df["average_precision"].median()),
               "val_ap_support": float(df["ap_support"].median()),
               "val_precision_at_n_true": float(df["precision_at_n_true"].median()),
               "val_truth_coverage": float(df["truth_coverage"].median()),
               "seconds": time.perf_counter() - t0}
        log.append(row)
        # **Selection is on the decoded mean, and only on it.**  Every other column here is a
        # diagnostic; AUC in particular is at 0.99 before the first step and cannot rank epochs.
        star = ""
        if row["val_f1"] > best[0]:
            best = (row["val_f1"], epoch)
            torch.save({"policy": policy.state_dict(), "policy_config": vars(cfg),
                        "epoch": epoch}, run_dir / "best.pt")
            star = "  <- best"
        # `best_ha.pt`: the lowest relaxed-decoder validation BER, falling back to all cells when
        # no validation cell is level 2.  An option kept open, NOT the selection checkpoint.
        ha_now = row.get(ha_sel_col, row["val_ha_ber"])
        if np.isfinite(ha_now) and ha_now < best_ha[0]:
            best_ha = (float(ha_now), epoch)
            torch.save({"policy": policy.state_dict(), "policy_config": vars(cfg),
                        "epoch": epoch}, run_dir / "best_ha.pt")
            star += "  <- best HA"
        print(f"[w10] epoch {epoch:3d}: loss {row['loss']:.4f} | val F1 {row['val_f1']:.4f} "
              f"(greedy {row['val_f1_greedy']:.4f}, {row['val_f1_vs_greedy']:+.4f}) | "
              f"exact {row['val_exact']}/{len(df)} | AP {row['val_average_precision']:.4f} "
              f"(support {row['val_ap_support']:.4f}) | err-red "
              f"{row['val_auc_error_reduction']:+.3f} | HA BER {row['val_ha_ber']:.3f} "
              f"(greedy {row['val_ha_ber_greedy']:.3f})"
              + (f" | {args.decode_loss} {row['decode_term']:.4f} on {d_fired}/{len(order)}"
                 if use_decode else "")
              + (f" | edge {row['edge_loss']:.4f}" if e_terms else "")
              + f" | {row['seconds']:.0f}s{star}", flush=True)

    ldf = pd.DataFrame(log)
    out = paths.artifact(args.out, f"rl_gnn_train_{run_id}_tau{args.tau}.parquet")
    ldf.to_parquet(out, index=False)

    gain = float(ldf["val_f1_vs_greedy"].iloc[best[1]])
    verdict = {"arch": args.arch, "conv": conv, "layers": layers, "seed": args.seed,
               "containment": (not args.no_containment) if conv == "pair" else None,
               "edge_weight": float(args.edge_weight) if conv == "pair" else None,
               "phases": phases, "best_epoch": best[1], "best_val_f1": best[0],
               "best_val_f1_greedy": float(ldf["val_f1_greedy"].iloc[best[1]]),
               "G_N1_val_f1_vs_greedy": gain,
               "G_N1": "GREEN" if gain >= G_N1_GREEN else
                       "AMBER" if gain > G_N1_AMBER else "RED",
               "val_auc_support": float(ldf["val_auc_support"].iloc[best[1]]),
               "val_auc_error_reduction": float(ldf["val_auc_error_reduction"].iloc[best[1]]),
               "val_truth_coverage": float(ldf["val_truth_coverage"].iloc[best[1]]),
               # The per-decoder split at the selected epoch.  `best_val_f1` above is the mean over
               # a validation set whose composition differs between conditions on this ladder, so
               # it does NOT compare across them; `val_f1_level1` does, because it is the same
               # population `_mixL` and `_mix2L` were selected on.
               "val_decoder": args.val_decoder, "val_max_level": int(args.val_max_level),
               **{c: float(ldf[c].iloc[best[1]]) for c in ldf.columns
                  if c.startswith("val_f1_level1") or c.startswith("val_f1_relaxed")
                  or c.startswith("val_ha_")},
               "best_ha_epoch": int(best_ha[1]),
               "best_ha_criterion": ha_sel_col if ha_sel_col in ldf.columns else "val_ha_ber",
               "best_val_ha_ber": best_ha[0] if np.isfinite(best_ha[0]) else None,
               **{c: int(ldf[c].iloc[best[1]]) for c in ldf.columns
                  if c.startswith("val_cells_")},
               "note": "G-N1 here is the VALIDATION reading and is not the gate; the gate is the "
                       "test split, on w10_gnn_eval.py's rows."}
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print(f"[w10] best epoch {best[1]} val F1 {best[0]:.4f} ({gain:+.4f} vs greedy) "
          f"-> {run_dir / 'best.pt'}")
    print(f"[w10] wrote {out}")
    if ha_pool is not None:
        ha_pool.close()
        ha_pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
