"""Co-parent edge targets for :class:`opc.rl.policy.PairConv`'s supervised attention (S3).

A true hybrid's covering parents that are rows of ``C_tau`` (``hybrid.truth_ha`` witnesses) are the
positives: every ordered pair ``(a, b)`` of them is the CSR entry ``a <- b`` of the crossing graph.
The A108 addendum measured every such pair crossing (1,926 of 1,926), so a pair that is not an
entry is counted, not silently dropped.

Negatives are what the gate has to learn to ignore: each positive query's best-supported other
neighbours (the proxy S0 measured as wrong for most minority sides), plus uniformly drawn entries
from every row, so a row with no partner learns to send its attention to the sink.

Numpy only -- no torch -- like :mod:`opc.rl.hybrid`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc.rl import hybrid as HY

__all__ = ["EdgeTargets", "coparent_entries", "hard_negatives", "edge_targets", "sample_edges"]


@dataclass(frozen=True)
class EdgeTargets:
    pos: np.ndarray        # int64 CSR entries a <- b, a and b co-parents of one true hybrid
    hard: np.ndarray       # int64 CSR entries: each positive query's best-supported non-partners
    n_missing: int         # co-parent pairs that are not a crossing entry


def coparent_entries(graph, truth: HY.TruthHA) -> tuple[np.ndarray, int]:
    pos, missing, seen = [], 0, set()
    indptr, nbr = graph.indptr, graph.nbr
    for wit in truth.witness:
        rows = [int(r) for r in wit if r >= 0]
        for a in rows:
            for b in rows:
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                lo, hi = int(indptr[a]), int(indptr[a + 1])
                j = lo + int(np.searchsorted(nbr[lo:hi], b))     # OverlapCSR rows are col-sorted
                if j < hi and int(nbr[j]) == b:
                    pos.append(j)
                else:
                    missing += 1
    return np.asarray(sorted(pos), dtype=np.int64), missing


def hard_negatives(graph, support: np.ndarray, pos: np.ndarray, per_query: int) -> np.ndarray:
    if pos.size == 0 or per_query <= 0:
        return np.zeros(0, dtype=np.int64)
    indptr, nbr = graph.indptr, graph.nbr
    rows = np.searchsorted(indptr, pos, side="right") - 1
    out = []
    for q in np.unique(rows).tolist():
        idx = np.arange(int(indptr[q]), int(indptr[q + 1]), dtype=np.int64)
        idx = idx[~np.isin(idx, pos)]
        out.append(idx[np.argsort(-support[nbr[idx]], kind="stable")[:per_query]])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def edge_targets(inst, truth: HY.TruthHA | None = None, per_query: int = 16) -> EdgeTargets:
    th = truth if truth is not None else HY.truth_ha(inst.truth, inst.gs.words)
    pos, missing = coparent_entries(inst.graph, th)
    hard = hard_negatives(inst.graph, np.asarray(inst.gs.support, dtype=np.float64), pos, per_query)
    return EdgeTargets(pos=pos, hard=hard, n_missing=int(missing))


def sample_edges(targets: EdgeTargets, n_entries: int, n_random: int, rng):
    """``(entries, labels)`` for one step: every positive, every hard negative, ``n_random`` more."""
    rnd = (rng.integers(0, n_entries, size=n_random).astype(np.int64)
           if n_entries and n_random else np.zeros(0, dtype=np.int64))
    idx = np.unique(np.concatenate([targets.pos, targets.hard, rnd]))
    return idx, np.isin(idx, targets.pos).astype(np.float32)
