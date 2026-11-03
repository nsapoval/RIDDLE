"""W10 -- the supervised node-classification arm: labels, metrics, and the decode.

The question this module serves is deliberately *not* the MDP's.  Drop the sequence entirely and
ask, per candidate cluster, ``P(A in C(N_true))``; then turn the resulting per-node scores into a
cluster system the same way greedy-by-support turns support into one.  There is no MPL anywhere in
this path -- no reward, no fit, no writeback -- which is what makes it cheap enough to run at
n = 250, where the swap benchmark's per-cell MPL cost is ~17 minutes.

**Nothing here is a second copy of something that exists.**  The label is
:attr:`~opc.rl.instances.Instance.truth_rows` scattered into a mask, the model is
:class:`opc.rl.policy.Policy` with its already-present ``cluster_logits`` head, the legality is
:class:`opc.rl.env.ClusterEnv`, and the cluster metrics are ``Instance.cluster_counts``.  What is
here is the three things that had no home: the tie-aware ranking metrics beside :func:`opc.rl.bc.auc`,
the score-ordered decode, and the diagnostic that asks whether a thresholded prediction is even a
cluster system.

**The decode is one variable changed against the incumbent.**  :func:`decode` is
:func:`opc.rl.env.greedy_support_rollout` with the model's probability substituted for support and
nothing else -- same env, same mask, same ``step``, same stable rank tie-break.  That is what makes
a cluster-F1 delta a statement about the *model*: if the two decodes differed when handed the same
scores, the delta would be measuring the decoder.  ``tests/test_rl_nodecls.py`` pins the parity on
real instances.

It also means a decoded selection is feasible, maximal and within budget ``k`` **by construction** --
``ClusterEnv.step`` enforces (L), (N3O) and the budget -- so there is no repair step and no
threshold.  A threshold would only lose recall: the env terminates when nothing is legal, and greedy
is maximal at exactly the truth's cardinality on 192 of 192 cells (A71).
"""

from __future__ import annotations

import numpy as np

from opc.rl import env as env_mod
from opc.rl import instances as I
from opc.solve import truth as T

__all__ = [
    "labels",
    "positive_rate",
    "pos_weight",
    "pos_weight_from_counts",
    "decode",
    "decode_trace",
    "decode_loss_terms",
    "band_loss_terms",
    "average_precision",
    "precision_at_k",
    "error_reduction",
    "raw_set_diagnostics",
    "SCALE_GLOBAL_IDX",
    "zero_scale_globals",
    "block_idx",
    "zero_node_blocks",
]

#: Indices into :data:`opc.rl.instances.GLOBAL_FEATURES` whose value is a statement about *problem
#: size* rather than about the instance -- ``n_over_100``, ``log_n_clusters``,
#: ``log_n_components``.  **Empty since A97 removed them from the feature set.**
#:
#: **Derived by name, never written as literal indices, and that is the whole point.**  It used to
#: be ``(2, 3, 4)``.  A97 narrowed ``GLOBAL_FEATURES`` from 8 entries to 5, and those three indices
#: now address ``density``, ``mean_degree_frac`` and ``kernel_frac`` -- so a literal would have gone
#: on zeroing three features, silently, and every ablation run against it would have been measuring
#: something else entirely. A name lookup goes empty instead, and :func:`zero_scale_globals`
#: refuses rather than becoming a no-op that looks like it worked.
#:
#: The history this records: training is at n in {25, 50} and evaluation reaches n = 250, so these
#: three were the only features the model had to **extrapolate** on -- ``n_over_100`` ran 0.25/0.5
#: in training against 2.5 at b250, while every *node* feature is instance-normalised and has no
#: such problem (A66's design).  A96 measured zeroing them as **+0.0403** ΔF1 at b250 and positive
#: in range as well, so A97 removed them outright.
#:
#: ``k`` (index 0) was **not** in this set even though it also extrapolates (trained on
#: {0,1,2,3,5}, tested to 16), because it is not a size proxy -- it is the budget the action mask
#: enforces, so dropping it would remove real information rather than an artefact.  It survives
#: A97 and remains a stated risk.
SCALE_GLOBAL_IDX: tuple[int, ...] = tuple(
    i for i, name in enumerate(I.GLOBAL_FEATURES) if name in I.LEGACY_SIZE_FEATURES
)


def labels(inst) -> np.ndarray:
    """``bool[|C|]``: is row ``i`` of ``inst.gs.words`` a cluster of the generating network?

    The positives are ``truth_rows``, which is ``C(N_true) & C_tau`` -- **not** ``C(N_true)``.  A
    cluster the tau filter removed is not a row of the ground set and therefore cannot be a label;
    it is counted in ``inst.n_true`` and shows up as the coverage gap.  So a classifier's ceiling is
    ``inst.truth_coverage``, and it is below 1 on 35% of the b250 test split.
    """
    y = np.zeros(int(inst.n_clusters), dtype=bool)
    y[np.asarray(inst.truth_rows, dtype=np.int64)] = True
    return y


def positive_rate(inst) -> float:
    """``|truth_rows| / |C|``.  Measured at **2.3-2.4% at every one of n = 25/50/100/250**."""
    return float(np.asarray(inst.truth_rows).size / max(1, int(inst.n_clusters)))


def pos_weight_from_counts(pos: int, tot: int) -> float:
    """``(tot - pos) / pos`` -- the ``BCEWithLogitsLoss`` weight, from counts alone.

    Split out from :func:`pos_weight` because the two counts it needs are already columns of
    ``rl_instances_<phase>_tau<tau>.parquet`` (``n_true_in_ctau`` and ``n_clusters``).  Computing
    the weight by materialising every training instance is what a driver does when it does not
    know that: at 1,400 cells it forces a full pass through the pool *before* the first epoch,
    which sets the peak-memory watermark for the whole run and pays a `ClusterEnv` construction
    per cell for two scalars.
    """
    pos, tot = int(pos), int(tot)
    if pos == 0:
        raise ValueError("no positive labels in the pool -- the fixture is vacuous")
    return float((tot - pos) / pos)


def pos_weight(insts) -> float:
    """``(1 - p) / p`` pooled over ``insts`` -- the ``BCEWithLogitsLoss`` weight on the positives.

    Pooled over the training set rather than computed per instance, because the rate is nearly
    constant across taxon count and a per-instance weight would make the loss scale differ between
    graphs for no reason.
    """
    return pos_weight_from_counts(
        sum(int(np.asarray(i.truth_rows).size) for i in insts),
        sum(int(i.n_clusters) for i in insts),
    )


def zero_scale_globals(x_global: np.ndarray) -> np.ndarray:
    """A copy of ``x_global`` with :data:`SCALE_GLOBAL_IDX` zeroed -- A96's transfer ablation.

    **Raises since A97.** The three features it zeroed are no longer in the feature set, so there
    is nothing to zero and a silent identity return would make ``--zero-scale-globals`` report an
    ablation it did not perform. Reproducing A96 means checking out the commit before A97 and
    rebuilding the instance cache; see :data:`opc.rl.instances.LEGACY_SIZE_FEATURES`.
    """
    if not SCALE_GLOBAL_IDX:
        raise SystemExit(
            "zero_scale_globals: the size features it zeroes were removed from GLOBAL_FEATURES in "
            f"A97, so there is nothing to zero (current features: {', '.join(I.GLOBAL_FEATURES)}). "
            "Drop --zero-scale-globals -- it is the default now. To reproduce A96's ablation, check "
            "out the commit before A97 and rebuild instances_tau<t>/.")
    out = np.array(x_global, dtype=np.float32, copy=True)
    out[list(SCALE_GLOBAL_IDX)] = 0.0
    return out


def block_idx(name: str) -> tuple[int, ...]:
    """Columns of :data:`opc.rl.instances.NODE_FEATURES` belonging to W13 block ``name``.

    **Derived by name, and an unknown name raises.**  This is :data:`SCALE_GLOBAL_IDX`'s rule in a
    second place, for the reason A97 records: the literal ``(2, 3, 4)`` stayed *in range* after
    ``GLOBAL_FEATURES`` was narrowed and would have zeroed three useful features while reporting a
    different ablation.  Node blocks are appended in order today, so a literal slice would work
    today -- which is exactly when it is cheapest to not write one.
    """
    if name not in I.NODE_BLOCKS:
        raise SystemExit(f"unknown node block {name!r}; the blocks are "
                         f"{', '.join(sorted(I.NODE_BLOCKS))}.")
    block = I.NODE_BLOCKS[name]
    idx = tuple(i for i, nm in enumerate(I.NODE_FEATURES) if nm in block)
    if len(idx) != len(block):
        raise SystemExit(f"node block {name!r} names {len(block)} features but only {len(idx)} "
                         f"are in NODE_FEATURES; the two have drifted apart.")
    return idx


def zero_node_blocks(x_nodes, names) -> np.ndarray:
    """A copy of ``x_nodes`` with every column of the named W13 blocks set to 0.

    A96 established that zeroing a block at run time on a model *trained* with it reproduces
    removing it and retraining -- A97 measured the two as statistically indistinguishable at
    b50/b100/b250 (p 0.32-0.73) -- so this is a legitimate cheap first cut at pricing one block
    against another, and it costs no GPU time at all.  It is a first cut and not a verdict: only a
    retrain prices a block properly, and A97's b25 exception (-0.0080, p 0.054) is on the record.

    Accepts a torch tensor or a numpy array and returns the same kind, because the training loop
    holds device tensors and the probes hold numpy.
    """
    names = [names] if isinstance(names, str) else list(names)
    if not names:
        return x_nodes
    cols = sorted({i for nm in names for i in block_idx(nm)})
    if hasattr(x_nodes, "clone"):                      # torch.Tensor
        out = x_nodes.clone()
        out[:, cols] = 0.0
        return out
    out = np.array(x_nodes, dtype=np.float32, copy=True)
    out[:, cols] = 0.0
    return out


# ---------------------------------------------------------------------------- the decode


def decode(env, scores: np.ndarray) -> np.ndarray:
    """Accept in descending ``scores`` order under the env's own mask, to termination.

    Structurally identical to :func:`opc.rl.env.greedy_support_rollout` -- the rank array, the
    ``argmin`` over the legal set, the same :func:`opc.rl.env.rollout` -- with ``scores`` where it
    has ``env.support``.  Written that way on purpose: the two must agree exactly when handed the
    same scores, or a cluster-F1 delta is not attributable to the model.

    Ranking through a stable ``argsort`` rather than taking ``argmax(scores[legal])`` directly is
    not cosmetic.  It fixes the tie-break to ascending cluster index for *both* callers, which is
    what makes the parity exact rather than approximate on a tied score vector -- and support is
    heavily tied (51% of the tau = 0 ground set has support exactly 1, A29).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    order = np.argsort(-s, kind="stable")
    rank = np.empty(env.n, dtype=np.int64)
    rank[order] = np.arange(env.n)

    def pick(mask):
        cand = np.flatnonzero(mask)
        return cand[np.argmin(rank[cand])]

    return env_mod.rollout(env, pick)


def decode_trace(env, scores: np.ndarray, y: np.ndarray, cap: int | None = None):
    """:func:`decode`, plus what it *considered* at every step.

    Returns ``(selection, trace)`` where ``trace`` is one ``(picked, legal_true)`` pair per step:
    the index accepted, and the indices of the labelled-true clusters that were **legal at that
    moment**.  A step where ``picked`` is not itself true and ``legal_true`` is non-empty is an
    *avoidable* error in A91's sense -- the model had a true cluster available and passed it over,
    at a median of 2 ranks.

    Recorded inside the ``pick`` closure rather than by re-implementing the loop, so this and
    :func:`decode` and :func:`opc.rl.env.greedy_support_rollout` remain one rollout with three
    scores.  ``legal_mask`` returns a fresh array per call, so there is no aliasing hazard; only
    the true indices are kept, which is ~200 B a step rather than a ``|C|`` mask.

    **This mutates ``env``** -- every rollout does -- so read ``env.selection`` only immediately
    after your own call, and never hand this a pool entry whose env is shared with a
    :class:`~opc.rl.swap.SwapEnv`.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    y = np.asarray(y, dtype=bool)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(env.n, dtype=np.int64)
    rank[order] = np.arange(env.n)
    trace: list[tuple[int, np.ndarray]] = []

    def pick(mask):
        cand = np.flatnonzero(mask)
        a = int(cand[np.argmin(rank[cand])])
        trace.append((a, cand[y[cand]]))
        return a

    return env_mod.rollout(env, pick, max_steps=cap), trace


def decode_loss_terms(logits, trace, y, kind: str, *, pos_weight=None, selection=None):
    """A differentiable error term computed from what the decode actually did.

    ``logits`` is the head's ``float[|C|]`` **with** its graph; the trace is built from a detached
    copy, so the selection is a *constant* here.  This is a decode-conditioned reweighting of a
    static score, **not** a policy gradient -- which is the difference between it and the
    terminal-set GRPO objective ``HANDOFF.md`` names as untried.

    Two kinds, and they are not variations on one idea:

    ``"step"`` -- at each step whose pick ``p`` is not true while some true cluster ``q`` was still
    legal, ``-log [ sum_q e^{z_q} / (e^{z_p} + sum_q e^{z_q}) ]``.  A listwise constraint
    ``z_q > z_p`` on exactly the pairs the decode got wrong: since both were legal and the decode
    ranks by score, that inequality is precisely the condition under which the error disappears.
    It is a **comparison**, which a per-node BCE cannot express.

    It escapes the mechanism A92 measured for ``bc.set_clone_loss`` on four counts: that one
    averaged over *every* step (so ~50 legal true clusters early made the term nearly free and the
    easy steps swamped the late ones), took the full ~2,000-candidate denominator, trained
    ``Policy.logits`` (corr(epoch, val F1) -0.36/-0.15/-0.34), and was teacher-forced on the oracle
    sequence.  This fires only where the decode errs (measured 12-13% of steps), over a denominator
    of ~6-11, on ``cluster_logits`` (corr **+0.92**), stepping along the model's *own* decode.
    The price is on-policy non-stationarity -- the loss is not comparable across epochs, so
    ``corr(epoch, decoded val F1)`` **per seed** is the diagnostic that has to be read instead.

    ``"set"`` -- BCE over the decode's mistakes ``(S minus T) u (T minus S)``.  **Be clear what this is:**
    those terms are a strict subset of the terms already in the full BCE, so adding it is BCE with
    weight ``(1 + lambda)`` on 3-6 nodes and 1 on the other ~2,000 -- per-epoch hard-example
    mining.  It cannot express a preference *between* two nodes; it moves two independent point
    targets and hopes the order follows.  That is exactly why it is the control worth running: if
    ``step`` beats it, the gain belongs to the ordering rather than to the reweighting.

    Returns a 0-d tensor, **exactly zero when the decode made no avoidable error**, which is what
    makes the term vanish on a cell that is already right rather than adding a constant.
    """
    import torch

    y_t = torch.as_tensor(np.asarray(y, dtype=bool), device=logits.device)
    if kind == "step":
        terms = []
        for picked, legal_true in trace:
            if y_t[picked] or legal_true.size == 0:
                continue                       # right pick, or nothing better was available
            q = torch.as_tensor(np.asarray(legal_true, dtype=np.int64), device=logits.device)
            good = torch.logsumexp(logits[q], dim=0)
            terms.append(torch.logaddexp(good, logits[picked]) - good)
        if not terms:
            return logits.sum() * 0.0          # keeps the graph, so `.backward()` is always legal
        return torch.stack(terms).mean()
    if kind == "set":
        if selection is None:
            raise ValueError("kind='set' needs the decoded selection")
        sel = torch.zeros_like(y_t)
        sel[torch.as_tensor(np.asarray(selection, dtype=np.int64), device=logits.device)] = True
        wrong = torch.nonzero(sel ^ y_t, as_tuple=True)[0]
        if wrong.numel() == 0:
            return logits.sum() * 0.0
        import torch.nn.functional as F
        return F.binary_cross_entropy_with_logits(
            logits[wrong], y_t[wrong].to(logits.dtype), pos_weight=pos_weight)
    raise ValueError(f"unknown kind {kind!r}")


def band_loss_terms(logits, y, *, width: int = 3, max_pairs: int = 256, rng=None):
    """A pairwise ranking margin inside the **confusable band**, with no decode at all.

    A91 chose the band -- the top ``width * n_true`` by score -- because it is where the headroom
    is: held-out band AUC 0.9521 against a full-set AUC of 0.99+.  This samples ``(true, false)``
    pairs from it and applies ``softplus(z_p - z_q)``, the same ``z_q > z_p`` constraint
    :func:`decode_loss_terms` imposes at an erring step, but over ~75-150 pairs a cell instead of
    the decode's 3-6 -- and with no env, no rollout and no trace.

    It is the control that prices the decode machinery: if this matches ``"step"``, the trace is
    not earning its cost; if ``"step"`` wins, the decode-conditioning is the thing that mattered.
    """
    import torch

    y_b = np.asarray(y, dtype=bool)
    n_true = int(y_b.sum())
    if n_true == 0:
        return logits.sum() * 0.0
    with torch.no_grad():
        order = torch.argsort(logits, descending=True, stable=True)
    band = order[:min(logits.numel(), width * n_true)].detach().cpu().numpy()
    pos = band[y_b[band]]
    neg = band[~y_b[band]]
    if pos.size == 0 or neg.size == 0:
        return logits.sum() * 0.0
    rng = np.random.default_rng() if rng is None else rng
    k = min(max_pairs, pos.size * neg.size)
    ip = torch.as_tensor(rng.choice(pos, size=k), device=logits.device, dtype=torch.long)
    ineg = torch.as_tensor(rng.choice(neg, size=k), device=logits.device, dtype=torch.long)
    return torch.nn.functional.softplus(logits[ineg] - logits[ip]).mean()


def _system_of(inst, keep: np.ndarray, y: np.ndarray, prefix: str) -> dict:
    words = np.ascontiguousarray(np.asarray(inst.gs.words)[keep])
    ok, j = T.feasible_and_cost(words, int(inst.n))
    return {f"{prefix}_size": int(keep.size), f"{prefix}_feasible": bool(ok),
            f"{prefix}_k_needed": int(j), f"{prefix}_tp": int(y[keep].sum())}


def raw_set_diagnostics(inst, scores: np.ndarray, theta: float = 0.5) -> dict:
    """Does the classifier's *unconstrained* output stand for a network?  Two readings.

    A pure diagnostic and never the headline -- :func:`decode` produces the reported selection and
    it cannot be infeasible.  This one can, and that is the point: it separates "the model learned
    the combinatorial structure" from "the model learned the marginal ranking and the env supplied
    the structure".  Feasibility is checked by :func:`opc.solve.truth.feasible_and_cost`, the
    *independent* validator, which shares no state with :meth:`opc.rl.env.ClusterEnv.legal_mask`.

    **``top`` is the reading to trust, and ``raw`` is the one with a caveat.**

    * ``top_*`` takes the highest-scoring ``|truth_rows|`` clusters.  Threshold-free, and its
      cardinality is the one the decode ends at anyway (greedy is maximal at the truth's
      cardinality on 192 of 192 cells, A71), so it is the honest "same size, legality removed"
      counterfactual.
    * ``raw_*`` thresholds at ``theta``.  **A threshold is not comparable across training runs
      here**, because the head is trained under a ``pos_weight`` of about 42: the optimal logit is
      ``log(w * odds)``, so ``sigmoid > 0.5`` is ``p_true > 1/(1 + w)``, i.e. a true-posterior cut
      near **0.023**, not 0.5.  That is why the thresholded set is large and reliably infeasible;
      it is a property of the loss, not a discovery about the model.  Kept because its *size* is
      informative and because a model that became feasible here would be saying something real.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = labels(inst)
    n_top = int(np.asarray(inst.truth_rows).size)
    top = np.argsort(-s, kind="stable")[:n_top]
    out = _system_of(inst, np.flatnonzero(s > float(theta)), y, "raw")
    out["raw_theta"] = float(theta)
    out.update(_system_of(inst, top, y, "top"))
    return out


# ---------------------------------------------------------------------------- ranking metrics


def _tied_blocks(s: np.ndarray):
    """``(order, starts)`` -- ``s`` sorted descending, and the start index of each tied block."""
    order = np.argsort(-s, kind="stable")
    sv = s[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    return order, starts


def average_precision(scores: np.ndarray, lbls: np.ndarray) -> float:
    """Average precision, **exact in expectation over random orderings within a tied block**.

    AUC is a saturated instrument here -- support alone separates ``C(N_true)`` at 0.9918 -- so AP
    is the metric with room left in it, and at a 2.3% positive rate it is also the one that reflects
    what the decode actually consumes: the top of the ranking.

    **Ties are handled, and not handling them would decide the comparison.**  A plain descending sort
    credits a tied block in whatever order ``argsort`` happens to produce, which is a coin flip per
    block; a model with float scores has no ties while the support baseline is riddled with them, so
    an arbitrary convention would systematically flatter or punish the baseline.  Crediting each
    positive the precision at its block's *end* is deterministic but pessimistic, which favours the
    tie-free model for the same reason.

    So the block is integrated instead.  For a tied block of ``n_b`` items holding ``p_b`` positives,
    beginning after ``s`` items of which ``P_s`` are positive, a positive at uniform rank ``J`` in
    the block has ``E[precision | J = j] = (P_s + 1 + (j-1)(p_b-1)/(n_b-1)) / (s + j)`` -- the inner
    term being the hypergeometric mean of how many of the block's *other* positives precede it.
    Averaging over ``j`` and multiplying by ``p_b`` gives the block's exact contribution.
    ``tests/test_rl_nodecls.py`` checks it against brute-force enumeration of every permutation.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(lbls).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order, starts = _tied_blocks(s)
    yo = y[order]
    total, seen_pos = 0.0, 0
    for b, lo in enumerate(starts):
        hi = int(starts[b + 1]) if b + 1 < starts.size else y.size
        n_b, p_b = hi - lo, int(yo[lo:hi].sum())
        if p_b:
            j = np.arange(1, n_b + 1, dtype=np.float64)
            share = (j - 1.0) * ((p_b - 1) / (n_b - 1)) if n_b > 1 else np.zeros(n_b)
            total += p_b * float(np.mean((seen_pos + 1.0 + share) / (lo + j)))
        seen_pos += p_b
    return float(total / n_pos)


def precision_at_k(scores: np.ndarray, lbls: np.ndarray, k: int) -> float:
    """Fraction of the top ``k`` by score that are positive, ties averaged.

    Read at ``k = |truth_rows|``, where it is the natural companion to the decode: the decode takes
    about that many clusters (greedy is maximal at the truth's cardinality), so this is "how many
    would be right if legality were free".  The gap between it and the decoded precision is what the
    (L)/(N3O)/budget constraints cost.

    A tied block straddling the cut contributes its positives in proportion, which is the expected
    count under a random ordering of the block -- the same convention as :func:`average_precision`.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(lbls).astype(bool)
    k = int(k)
    if k <= 0 or k > y.size:
        return float("nan")
    order, starts = _tied_blocks(s)
    yo = y[order]
    hits = 0.0
    for b, lo in enumerate(starts):
        hi = int(starts[b + 1]) if b + 1 < starts.size else y.size
        if lo >= k:
            break
        p_b = int(yo[lo:hi].sum())
        take = min(hi, k) - lo
        hits += p_b * (take / (hi - lo))
    return float(hits / k)


def error_reduction(auc_head: float, auc_base: float) -> float:
    """``(auc_head - auc_base) / (1 - auc_base)`` -- the statistic to read instead of the gain.

    ``RL_PLAN`` §8 pre-registered "AUC minus support-ranking AUC >= +0.03" for **G-R3** and then
    measured support alone at 0.9918, which makes the entire remaining headroom 0.0082 and the band
    arithmetically unreachable.  Error reduction is what stays meaningful when a baseline is near
    perfect; the landed W4.2 head sits at **0.378** of it.  Report both, substitute neither.
    """
    if not np.isfinite(auc_base) or auc_base >= 1.0:
        return float("nan")
    return float((auc_head - auc_base) / (1.0 - auc_base))
