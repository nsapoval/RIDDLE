"""Hybrid ancestry read off a cluster system, and the loss term that puts pressure on it (A108).

The metric
----------
``l2reg_eval/make_csv.py``'s, after Dai & Molloy (bioRxiv 10.64898/2026.06.06.730607, Fig. 3):
``P`` and ``T`` are the taxa below at least one hybrid of the prediction and of the truth,
``fpr = |P - T| / |P|``, ``fnr = |T - P| / |T|``, and the balanced error is their mean.  make_csv
reads it off a parsed eNewick.  Here it is read off the **Hasse diagram of the cluster system**,
which is the same object: a regular network *is* the Hasse diagram of its clusters
(:mod:`opc.rl.relaxed`), and a hybrid is a node with two covering parents.  Measured equal on 24 of
24 real instances (x25, x50, l2t50 on both arms) against ``hardwired_clusters`` + ``is_hybrid`` on
the ``Network`` -- truths, formal realisations and relaxed realisations alike -- and pinned against
make_csv's own string-scan derivation in ``tests/test_rl_hybrid.py``.

Why a targeted term, not the error itself
-----------------------------------------
The error is a step function of a discrete selection and has no gradient.  So, like
:func:`opc.rl.nodecls.decode_loss_terms`, the selection is a **constant** -- a decode of the
detached scores -- and the term re-weights the logits it implicates:

* **FN side.**  A true hybrid ``h`` is *missed* when some taxon below it has no predicted hybrid
  ancestry.  Its covering parents are what make it a hybrid, and the failure this exists for is that
  one parent scores high and the other low, so the decode keeps the tree.  Each missed hybrid gets
  mass ``fnr * |taxa(h) & FN| / sum_h' |taxa(h') & FN|`` (the masses sum to ``fnr``), split evenly
  over its parents that are rows of ``C_tau`` and **unselected**, each contributing
  ``w * softplus(-z)``.
* **FP side.**  The same over the predicted hybrids with a taxon in ``FP``, mass ``fpr``, split over
  their **selected** parents that are **not** true clusters, each contributing ``w * softplus(z)``.
* ``L = (FN side + FP side) / 2``: balanced as the metric is, and **exactly zero** when both rates
  are, which is what makes it vanish on a cell that is already right.

Mass with nowhere to land is dropped, not redistributed, and counted: a missed hybrid whose parents
are all selected or absent from ``C_tau`` (``ha_fn_dropped``), and a spurious hybrid whose selected
parents are all true clusters (``ha_fp_dropped``) -- the latter is a *missing* intermediate cluster,
which no push on a selected row can express.

Nothing here imports torch at module level: :func:`decode_and_target` runs in pool workers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc import bitset
from opc.rl import relaxed as RX
from opc.simulate import networks as N
from opc.solve import closure

__all__ = ["TruthHA", "HATargets", "hybrid_structure", "ancestry", "selection_full", "truth_ha",
           "rates", "ha_targets", "ha_loss_terms", "decode_and_target", "worker_init"]

_NAN = float("nan")


def hybrid_structure(full: np.ndarray, n_taxa: int):
    """``(taxa, parents)`` for every node of ``H(full)`` with two or more covering parents.

    ``taxa`` is ``bool[h, n]``, each hybrid's cluster; ``parents[i]`` holds the int64 rows of
    ``full`` that cover hybrid ``i``.  ``full`` must carry the trivial clusters: without the
    singletons a leaf hybrid is not a node, and without the root a hybrid under two maximal clusters
    has a parent too few.  In-degree above 2 is kept rather than refused -- no realisable network
    has it, so it never comes out of the relaxed decoder, and a metric should not raise.
    """
    full = np.ascontiguousarray(full)
    k = int(full.shape[0])
    if k == 0:
        return np.zeros((0, n_taxa), bool), []
    parent, child = closure.hasse_covers_words(full, n_taxa)
    indeg = np.bincount(child, minlength=k)
    hyb = np.flatnonzero(indeg >= 2)
    if hyb.size == 0:
        return np.zeros((0, n_taxa), bool), []
    taxa = bitset.to_dense(full[hyb], n_taxa).astype(bool)
    return taxa, [parent[child == h] for h in hyb]


def ancestry(full: np.ndarray, n_taxa: int) -> np.ndarray:
    """``bool[n]``: the taxa below at least one hybrid of ``H(full)``."""
    taxa, _ = hybrid_structure(full, n_taxa)
    return taxa.any(axis=0) if taxa.shape[0] else np.zeros(n_taxa, bool)


def selection_full(words: np.ndarray, selection, n_taxa: int) -> np.ndarray:
    """``F = S + trivial clusters`` with **no closure** -- :func:`opc.rl.relaxed.realise`'s object.

    ``with_trivial_clusters`` appends after the selected rows, so row ``j < len(selection)`` of the
    result is ``words[selection[j]]``; :func:`ha_targets` relies on that.
    """
    sel = np.asarray(selection, dtype=np.int64)
    chosen = (np.ascontiguousarray(np.asarray(words)[sel]) if sel.size
              else np.zeros((0, bitset.n_words(n_taxa)), np.uint64))
    return closure.with_trivial_clusters(chosen, n_taxa)


@dataclass(frozen=True)
class TruthHA:
    """The truth side, built once per cell: ``T``, each true hybrid's taxa, and its parents' rows."""

    ancestry: np.ndarray      # bool[n]
    taxa: np.ndarray          # bool[h, n]
    witness: tuple            # per hybrid, int64 rows of gs.words; -1 = not in C_tau


def truth_ha(truth, words: np.ndarray) -> TruthHA:
    """:class:`TruthHA` for a truth network over a candidate set ``words``.

    Read off ``H(C(N_true) + trivial)`` rather than the ``Network``'s own hybrids, so the truth and
    the prediction go through one definition.  The two agree on every instance measured (module
    docstring).  A parent that is the root is not a candidate and is not a witness.
    """
    n = int(truth.n_taxa)
    cl = sorted(sorted(c) for c in N.nontrivial_clusters(truth))
    tw = (np.stack([bitset.from_indices(c, n) for c in cl]) if cl
          else np.zeros((0, bitset.n_words(n)), np.uint64))
    full = closure.with_trivial_clusters(tw, n)
    taxa, parents = hybrid_structure(full, n)
    sizes = bitset.popcounts(full)
    row_of = {bytes(r): i for i, r in enumerate(np.ascontiguousarray(words))}
    witness = tuple(
        np.array([row_of.get(bytes(full[p]), -1) for p in ps.tolist() if 1 < int(sizes[p]) < n],
                 dtype=np.int64)
        for ps in parents)
    return TruthHA(ancestry=taxa.any(axis=0) if taxa.shape[0] else np.zeros(n, bool),
                   taxa=taxa, witness=witness)


def rates(pred, true) -> dict:
    """make_csv's ``ha_*`` numbers, NaN where a rate is 0/0; ``pred=None`` (no network) is all NaN.

    ``ha_ber`` is NaN when ``P`` is empty, as make_csv leaves it empty; ``T`` is never empty there
    (k >= 3), but is on a k = 0 training cell, which makes ``ha_fnr`` NaN too.
    """
    if pred is None:
        return {"ha_n_pred": -1, "ha_n_true": int(np.asarray(true, bool).sum()), "ha_fp": -1,
                "ha_fn": -1, "ha_fpr": _NAN, "ha_fnr": _NAN, "ha_ber": _NAN}
    P, T = np.asarray(pred, bool), np.asarray(true, bool)
    p, t = int(P.sum()), int(T.sum())
    fp, fn = int((P & ~T).sum()), int((T & ~P).sum())
    fpr = fp / p if p else _NAN
    fnr = fn / t if t else _NAN
    return {"ha_n_pred": p, "ha_n_true": t, "ha_fp": fp, "ha_fn": fn,
            "ha_fpr": fpr, "ha_fnr": fnr, "ha_ber": (fpr + fnr) / 2}


@dataclass(frozen=True)
class HATargets:
    """What the loss pushes: ``sign`` +1 raises a missed hybrid's unselected parent, -1 lowers a
    spurious hybrid's false selected parent.  ``stats`` carries the cell's rates and dropped mass."""

    rows: np.ndarray          # int64
    sign: np.ndarray          # int8
    weight: np.ndarray        # float32
    stats: dict


def ha_targets(words: np.ndarray, n_taxa: int, selection, y, truth: TruthHA) -> HATargets:
    """The module docstring's two sides, for one relaxed selection."""
    sel = np.asarray(selection, dtype=np.int64)
    full = selection_full(words, sel, n_taxa)
    taxa, parents = hybrid_structure(full, n_taxa)
    P = taxa.any(axis=0) if taxa.shape[0] else np.zeros(n_taxa, bool)
    T = truth.ancestry
    st = rates(P, T)
    fnr = st["ha_fn"] / st["ha_n_true"] if st["ha_n_true"] else 0.0
    fpr = st["ha_fp"] / st["ha_n_pred"] if st["ha_n_pred"] else 0.0
    y = np.asarray(y, dtype=bool)
    selected = np.zeros(y.size, dtype=bool)
    selected[sel] = True
    up: dict[int, float] = {}
    down: dict[int, float] = {}
    fn_drop = fp_drop = 0.0

    fn_mask = T & ~P
    if fn_mask.any():
        hits = truth.taxa[:, fn_mask].sum(axis=1)
        tot = float(hits.sum())
        for h in np.flatnonzero(hits).tolist():
            mass = fnr * float(hits[h]) / tot
            wit = truth.witness[h]
            u = wit[wit >= 0]
            u = u[~selected[u]]
            if u.size == 0:
                fn_drop += mass
                continue
            for r in u.tolist():
                up[r] = up.get(r, 0.0) + mass / u.size

    fp_mask = P & ~T
    if fp_mask.any():
        hits = taxa[:, fp_mask].sum(axis=1)
        tot = float(hits.sum())
        for g in np.flatnonzero(hits).tolist():
            mass = fpr * float(hits[g]) / tot
            ps = parents[g]
            v = sel[ps[ps < sel.size]]               # rows of `full` below sel.size are selections
            v = v[~y[v]]
            if v.size == 0:
                fp_drop += mass
                continue
            for r in v.tolist():
                down[r] = down.get(r, 0.0) + mass / v.size

    st.update(ha_fn_dropped=fn_drop, ha_fp_dropped=fp_drop, ha_n_up=len(up), ha_n_down=len(down))
    return HATargets(
        rows=np.array(list(up) + list(down), dtype=np.int64),
        sign=np.array([1] * len(up) + [-1] * len(down), dtype=np.int8),
        weight=np.array(list(up.values()) + list(down.values()), dtype=np.float32),
        stats=st)


def ha_loss_terms(logits, targets: HATargets):
    """``0.5 * sum_r w_r softplus(-sign_r z_r)``: a 0-d tensor that keeps the graph when empty."""
    import torch
    import torch.nn.functional as F

    if targets.rows.size == 0:
        return logits.sum() * 0.0
    dev = logits.device
    r = torch.as_tensor(targets.rows, device=dev)
    s = torch.as_tensor(targets.sign, device=dev).to(logits.dtype)
    w = torch.as_tensor(targets.weight, device=dev).to(logits.dtype)
    return 0.5 * (w * F.softplus(-s * logits[r])).sum()


# ------------------------------------------------------------------------------ pool workers

_LIMITS = None


def worker_init() -> None:
    """Pin a forked worker's BLAS to one thread, if threadpoolctl is there to do it.

    The worker is forked from a trainer whose numpy already sized its pool at the driver's
    ``OMP_NUM_THREADS=2``, so an environment variable set now would change nothing.  The decode's
    BLAS use is a ``|C| x n`` matvec per acceptance, so this is about oversubscription (A44), not
    speed.
    """
    global _LIMITS
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _LIMITS = threadpool_limits(limits=1)


def decode_and_target(task):
    """``(selection, HATargets)`` for one ``(words, n, k, max_level, scores, truth_rows, truth)``."""
    words, n, k, max_level, scores, truth_rows, truth = task
    sel = RX.RelaxedDecoder(words, int(n), int(k), int(max_level)).decode(
        np.asarray(scores, dtype=np.float64))
    y = np.zeros(int(words.shape[0]), dtype=bool)
    y[np.asarray(truth_rows, dtype=np.int64)] = True
    return sel, ha_targets(words, int(n), sel, y, truth)
