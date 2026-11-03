"""The reticulation-free prefix of a decode: where it stops, under two rules, and the tree it is.

A score-ordered decode (:func:`opc.rl.nodecls.decode`, :class:`opc.rl.relaxed.RelaxedDecoder`)
accepts compatible clusters first and only later starts spending the reticulation budget.  What it
holds just before the first spending acceptance is a laminar cluster system -- a multifurcating
tree -- and that is the object this module extracts, so it can be set beside a tree of blobs and
tested for containment in the true network's clusters.

Two stop rules, always computed together and always labelled
------------------------------------------------------------

``truncated``
    The decoder's own state immediately before its first budget-spending acceptance.  It is a
    prefix of the landed decode, and it depends on the decoder:

    * ``level1`` pre-accepts the kernel (every degree-0 row of the overlap graph) at ``reset``,
      whatever its score, so every kernel row is in the prefix -- including ones ranked below the
      stop;
    * ``level1`` refuses an overlapping candidate whose selected overlap edges carry two
      intersection classes ((L) at the candidate), and keeps accepting lower-ranked compatible
      clusters past it.  Those are in the prefix too.

    ``relaxed`` visits the kernel in rank order like any other row, and before any reticulation it
    refuses an overlapping candidate only when its ``dr`` exceeds the budget.

``rank_laminar``
    Walk the candidates in score order and stop at the first one that overlaps a row already taken,
    legal or not.  It is the longest laminar prefix of the ranking itself: the same for both
    decoders, and dependent on ``k`` only through the global feature the scores were computed with.

**Why "spends budget" can be read as "overlaps" before the first reticulation.**

* **level1.**  ``SearchState.can_accept`` returns class ``-1`` exactly when the candidate touches
  nothing accepted, and ``n_hybrids`` grows only when a class not yet in ``used`` is opened -- which,
  with ``used`` empty, is every overlapping acceptance.
* **relaxed.**  In a laminar diagram a compatible candidate has one minimal superset ``p`` and its
  maximal subsets are all children of ``p``, so ``dr = |P| + |Ch| - covers - 1 = 0``.  An overlapping
  one has a maximal subset inside the cluster it crosses, which ``p`` does not cover, so ``dr >= 1``.

**Two consequences are asserted on every call** (:func:`_finish`): both prefixes are laminar, and
``rank_laminar`` is contained in ``truncated``.  The containment holds because every candidate ranked
above the ``rank_laminar`` stop is compatible with everything above it, hence legal under either
decoder, hence accepted in rank order.  ``prefix_eval/check_prefix.py`` cross-validates the "spends
means overlaps" reading against a dense-bitset reference that shares nothing with ``n_hybrids`` or
``dr``.

**At ``k = 0`` the level1 truncation never fires**: an overlap is never legal, so the decode runs
to a maximal laminar system and is recorded ``stopped = False``.  The relaxed decoder is the same.

Both traces reuse the decoders' own loops -- a ``pick`` closure around :func:`opc.rl.env.rollout`
(``nodecls.decode_trace``'s pattern) and an ``_accept`` override (``l2mix_rank/walk.py``'s) -- so
there is still one rollout per decoder, and the traced selection must equal the untraced one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from opc import bitset
from opc.followup import search
from opc.rl import env as env_mod
from opc.rl import relaxed as RX
from opc.simulate import networks as NW

__all__ = ["MODES", "LABELS", "Trace", "Prefix", "trace_level1", "trace_relaxed", "truncated",
           "rank_laminar", "prefixes_level1", "prefixes_relaxed", "tree_newick", "overlapping_pairs"]

MODES = ("truncated", "rank_laminar")
#: How an output line or a CSV row names each mode.
LABELS = {"truncated": "prefix_truncated", "rank_laminar": "prefix_rank_laminar"}

_LENGTH = re.compile(r":[^,();]*")


def _rank(scores, n: int) -> np.ndarray:
    """``nodecls.decode``'s rank rule: a stable argsort of ``-scores``, ties to the lower index."""
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (n,):
        raise ValueError(f"scores must be ({n},), got {s.shape}")
    order = np.argsort(-s, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


@dataclass
class Trace:
    """One decode, in acceptance order.

    ``order[i]`` is the ``i``-th acceptance and ``spent[i]`` whether it spent budget; ``kernel`` is
    what was accepted before step 0 (level1's kernel, empty for relaxed); ``selection`` is the
    decode's own sorted output.
    """

    decoder: str
    selection: np.ndarray
    order: np.ndarray
    spent: np.ndarray
    kernel: np.ndarray
    rank: np.ndarray


@dataclass
class Prefix:
    mode: str
    rows: np.ndarray                 # sorted int64 rows into the candidate words
    n_kernel: int                    # rows of ``rows`` with no overlap partner anywhere in C_tau
    n_steps: int                     # acceptances (truncated) or rank positions (rank_laminar) taken
    stopped: bool
    stop_row: int                    # the first reticulation-costing / overlapping row, -1 if none
    stop_rank: int
    n_skipped_above_stop: int        # rows ranked above the stop that the prefix does not hold
    n_kernel_below_stop: int         # prefix rows ranked below the stop (level1's kernel)

    def as_dict(self) -> dict:
        return {"n_selected": int(self.rows.size), "n_kernel": self.n_kernel,
                "n_steps": self.n_steps, "stopped": self.stopped, "stop_row": self.stop_row,
                "stop_rank": self.stop_rank, "n_skipped_above_stop": self.n_skipped_above_stop,
                "n_kernel_below_stop": self.n_kernel_below_stop}


# ------------------------------------------------------------------------------------------ traces

def trace_level1(env, scores) -> Trace:
    """:func:`opc.rl.nodecls.decode`, recording each acceptance and whether it opened a class.

    ``pick`` reads ``n_hybrids`` *before* its step is taken, so the difference between consecutive
    reads (and the final state) is exactly what each acceptance spent.  **This mutates ``env``**, as
    every rollout does.
    """
    rank = _rank(scores, env.n)
    picks: list[int] = []
    hyb: list[int] = []

    def pick(mask):
        cand = np.flatnonzero(mask)
        a = int(cand[np.argmin(rank[cand])])
        picks.append(a)
        hyb.append(env.state.n_hybrids)
        return a

    sel = env_mod.rollout(env, pick)
    hyb.append(env.state.n_hybrids)
    kernel = (np.asarray(env.state.kernel, dtype=np.int64) if env.include_kernel
              else np.zeros(0, dtype=np.int64))
    return Trace("level1", np.asarray(sel, dtype=np.int64), np.asarray(picks, dtype=np.int64),
                 np.diff(np.asarray(hyb, dtype=np.int64)) > 0, np.sort(kernel), rank)


class _TracingRelaxed(RX.RelaxedDecoder):
    """``RelaxedDecoder`` recording ``(candidate, dr)`` per acceptance; decisions untouched."""

    def _accept(self, move) -> None:
        self.trace.append((int(move[0]), int(move[7])))
        super()._accept(move)


def trace_relaxed(words, n_taxa: int, budget: int, max_level: int, scores):
    """``(decoder, Trace)`` for :meth:`opc.rl.relaxed.RelaxedDecoder.decode`.  The decoder is returned
    so a caller can read its ``stats`` and call ``verify()``."""
    dec = _TracingRelaxed(words, n_taxa, budget, max_level)
    dec.trace = []
    sel = dec.decode(scores)
    rows = np.asarray([c for c, _ in dec.trace], dtype=np.int64)
    dr = np.asarray([d for _, d in dec.trace], dtype=np.int64)
    return dec, Trace(f"relaxed_L{int(max_level)}", np.asarray(sel, dtype=np.int64), rows, dr > 0,
                      np.zeros(0, dtype=np.int64), _rank(scores, dec.n))


# ---------------------------------------------------------------------------------------- prefixes

def _degree0(graph) -> np.ndarray:
    return np.diff(np.asarray(graph.indptr)) == 0


def _stop_fields(rows: np.ndarray, rank: np.ndarray, stop_row: int) -> tuple[int, int]:
    """``(rows ranked above the stop the prefix lacks, prefix rows ranked below it)``; ``-1`` for
    both when nothing stopped the decode, where neither has a reference point."""
    if stop_row < 0:
        return -1, -1
    bound = int(rank[stop_row])
    above = int(np.count_nonzero(rank[rows] < bound))
    return bound - above, int(rows.size) - above


def truncated(trace: Trace, graph) -> Prefix:
    """The decode's state just before its first budget-spending acceptance."""
    first = np.flatnonzero(trace.spent)
    cut = int(first[0]) if first.size else int(trace.order.size)
    rows = np.union1d(trace.kernel, trace.order[:cut]).astype(np.int64)
    stop_row = int(trace.order[cut]) if first.size else -1
    skipped, below = _stop_fields(rows, trace.rank, stop_row)
    return Prefix("truncated", rows, int(np.count_nonzero(_degree0(graph)[rows])), cut,
                  bool(first.size), stop_row, int(trace.rank[stop_row]) if stop_row >= 0 else -1,
                  skipped, below)


def rank_laminar(graph, scores) -> Prefix:
    """The longest prefix of the score ranking in which no row overlaps a row above it."""
    n = int(graph.n)
    rank = _rank(scores, n)
    order = np.argsort(rank)
    indptr, nbr = np.asarray(graph.indptr), np.asarray(graph.nbr)
    taken = np.zeros(n, dtype=bool)
    stop_row, cut = -1, n
    for i, v in enumerate(order.tolist()):
        if taken[nbr[indptr[v]:indptr[v + 1]]].any():
            stop_row, cut = int(v), i
            break
        taken[v] = True
    rows = np.sort(order[:cut]).astype(np.int64)
    return Prefix("rank_laminar", rows, int(np.count_nonzero(_degree0(graph)[rows])), cut,
                  stop_row >= 0, stop_row, cut if stop_row >= 0 else -1, 0, 0)


def overlapping_pairs(words: np.ndarray, rows, n_taxa: int) -> int:
    """How many pairs of ``words[rows]`` overlap (intersect, neither containing the other).  Dense
    bitsets, independent of the overlap graph."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size < 2:
        return 0
    d = bitset.to_dense(np.ascontiguousarray(np.asarray(words)[rows]), n_taxa).astype(np.float32)
    inter = d @ d.T
    size = d.sum(axis=1)
    over = (inter > 0) & (inter < size[:, None]) & (inter < size[None, :])
    return int(np.count_nonzero(np.triu(over, k=1)))


def _finish(trace: Trace, graph, scores, words, n_taxa: int) -> dict[str, Prefix]:
    out = {"truncated": truncated(trace, graph), "rank_laminar": rank_laminar(graph, scores)}
    for mode, p in out.items():
        bad = overlapping_pairs(words, p.rows, n_taxa)
        if bad:
            raise AssertionError(f"{trace.decoder} {mode} prefix is not laminar: {bad} overlapping "
                                 f"pairs")
    if not np.isin(out["rank_laminar"].rows, out["truncated"].rows).all():
        raise AssertionError(f"{trace.decoder}: the rank_laminar prefix is not inside the truncated "
                             f"one")
    if not np.isin(out["truncated"].rows, trace.selection).all():
        raise AssertionError(f"{trace.decoder}: the truncated prefix is not inside its own decode")
    return out


def prefixes_level1(env, scores, graph, words=None) -> tuple[Trace, dict[str, Prefix]]:
    """Trace the level1 decode on ``env`` and cut both prefixes from it."""
    trace = trace_level1(env, scores)
    words = env.gs.words if words is None else words
    return trace, _finish(trace, graph, scores, words, int(env.gs.n_taxa))


def prefixes_relaxed(words, n_taxa: int, budget: int, max_level: int, scores, graph):
    """``(decoder, Trace, prefixes)`` for the relaxed decode."""
    dec, trace = trace_relaxed(words, n_taxa, budget, max_level, scores)
    return dec, trace, _finish(trace, graph, scores, words, n_taxa)


# ------------------------------------------------------------------------------------------ output

def tree_newick(words: np.ndarray, rows, n_taxa: int, labels) -> str:
    """The prefix as a rooted, multifurcating Newick with no branch lengths.

    Realised through the landed chain -- :func:`opc.followup.search.realise`, then
    :func:`opc.simulate.networks.to_enewick` -- so it is written the way every predicted network
    is.  On a laminar set the closure adds nothing and the diagram is a tree; both are asserted.
    The lengths ``to_enewick`` writes are rank differences, which say nothing a tree of blobs could
    be compared on, so they are stripped.
    """
    rows = np.asarray(rows, dtype=np.int64)
    got = search.realise(words, rows, n_taxa, labels)
    if got is None:
        raise ValueError(f"a {rows.size}-cluster prefix realises no network")
    net, full = got
    if net.n_retics:
        raise AssertionError(f"a prefix realised {net.n_retics} reticulation(s)")
    if full.shape[0] != rows.size + n_taxa + 1:
        raise AssertionError(f"the closure added {full.shape[0] - rows.size - n_taxa - 1} "
                             f"cluster(s) to a prefix that should be laminar")
    return _LENGTH.sub("", NW.to_enewick(net))
