"""Ensembles over gene-tree subsamples: several draws of trees, one network.

A *draw* is one subsample of a gene-tree collection, taken through the ordinary chain -- root,
count clusters, tau = 2 candidate set, features, one forward -- so it carries its own candidate
set ``C_d`` (rows in first-appearance order, as ``extract.SupportCounter`` leaves them), a score
per row, and the selection its own decode made.  This module combines ``D`` draws into **one**
decode.  It imports numpy and ``opc`` only, never a driver: ``run_gnn_cpu`` and ``infer_gnn_cpu``
each parse ``argv`` and move ``sys.path`` at import, and whoever imports this has already chosen.

The union
---------
The draws' candidate sets are merged on the words themselves (the uint64 rows, compared as bytes),
since row indices mean nothing across draws.  Rows are kept in **first-seen order over draws
0..D-1**, each draw's rows in its own order.  That order is the decoders' tie-break (a stable
``argsort``, lower row first), so it is fixed rather than incidental, and with ``D = 1`` the union
*is* that draw's candidate set, row for row -- which is what makes the identity gates exact.

Every draw is a subset of one collection, so a cluster's support in a draw is at most its support
in the whole, and the union of the draws' tau = 2 sets is contained in the whole collection's.

Aggregators
-----------
``p_d(c)`` is draw ``d``'s score for ``c`` (for a GNN, the float32 ``p`` its decode used; for
greedy, its support), and **0 where c is not in C_d**: below tau in a draw is evidence against.

``pool``          ``sum_d p_d(c) / D``.  The draws are sorted per row before summing, so the float
                  sum is independent of draw order and exact at ``D = 1``.
``pool_present``  the same sum over only the draws that contain ``c``.  An ablation: it asks
                  whether presence across draws carries information of its own.
``vote``          ``v(c)``, the number of draws whose own decode (same source, same decoder)
                  **selected** ``c``.  The key is lexicographic -- majority (``2v > D``) first, then
                  ``v``, then ``pool`` -- and the decoder re-extends past the majority to a maximal
                  system under its own mask and budget.  Majority clusters that are jointly
                  infeasible are skipped by the decoder and counted (``majority_refused``).

Both decoders rank by a stable ``argsort(-float64(s))`` and read nothing but the ranks
(``nodecls.decode``, ``RelaxedDecoder.decode``), so a lexicographic key is passed as **exact
ranks** (:func:`rank_scores`).  Packing it into one float -- ``2 + v + pool`` -- would collapse
every ``pool`` below ~1e-16 into row order, and level1's kernel pre-acceptance and both decoders'
maximality fill do reach rows that small.

Votes are on the **selection**, not the realised network.  Level1's hybrid clusters are added by
the closure and are usually not candidates at all, so they could not be re-accepted anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opc.followup import search
from opc.rl import nodecls as NC
from opc.rl import relaxed as RX
from opc.rl.env import ClusterEnv, OverlapCSR
from opc.simulate import networks as NW
from opc.solve import candidates

#: The GNN aggregators; ``vote`` is also what the greedy-vote control runs, on support.
AGGREGATORS = ("pool", "pool_present", "vote")


@dataclass
class Draw:
    """One subsample's candidate set, scores and selections.

    ``scores`` maps a *source* -- a checkpoint label, or ``"greedy"`` for support -- to one score per
    row.  ``sel`` maps ``(source, decoder tag)`` to the rows that decode selected (sorted).
    """

    words: np.ndarray
    support: np.ndarray
    scores: dict = field(default_factory=dict)
    sel: dict = field(default_factory=dict)


def row_keys(words: np.ndarray) -> list[bytes]:
    words = np.ascontiguousarray(words, dtype=np.uint64)
    return [words[i].tobytes() for i in range(words.shape[0])]


def word_set(words: np.ndarray, rows) -> frozenset:
    """A selection as a set of cluster words (bytes): comparable across candidate sets."""
    words = np.ascontiguousarray(words, dtype=np.uint64)
    return frozenset(words[int(i)].tobytes() for i in np.asarray(rows, dtype=np.int64))


class Union:
    """The draws' candidate sets merged on their words.  See the module docstring."""

    def __init__(self, draws: list[Draw]):
        if not draws:
            raise ValueError("a union needs at least one draw")
        width = int(np.asarray(draws[0].words).shape[1])
        index: dict[bytes, int] = {}
        order: list[tuple[int, int]] = []           # (draw, row) of each union row's first sighting
        self.rowmap: list[np.ndarray] = []
        for d, dr in enumerate(draws):
            w = np.asarray(dr.words)
            if w.dtype != np.uint64 or w.ndim != 2 or w.shape[1] != width:
                raise ValueError(f"draw {d}: words {w.dtype} {w.shape}, expected uint64[K, {width}]"
                                 f" -- every draw must share one taxon order")
            if np.asarray(dr.support).shape != (w.shape[0],):
                raise ValueError(f"draw {d}: {np.asarray(dr.support).shape} supports for "
                                 f"{w.shape[0]} rows")
            keys = row_keys(w)
            if len(set(keys)) != len(keys):
                raise ValueError(f"draw {d}: duplicate candidate rows")
            rm = np.empty(len(keys), dtype=np.int64)
            for i, b in enumerate(keys):
                j = index.get(b)
                if j is None:
                    j = index[b] = len(order)
                    order.append((d, i))
                rm[i] = j
            self.rowmap.append(rm)
        self.draws = draws
        self.D = len(draws)
        self.n = len(order)
        self.width = width
        self.words = (np.stack([np.asarray(draws[d].words)[i] for d, i in order])
                      if order else np.zeros((0, width), np.uint64))
        self.present = np.zeros((self.D, self.n), dtype=bool)
        self.support = np.zeros(self.n, dtype=np.int64)
        for d, dr in enumerate(draws):
            self.present[d, self.rowmap[d]] = True
            self.support[self.rowmap[d]] += np.asarray(dr.support, dtype=np.int64)
            if not np.array_equal(self.words[self.rowmap[d]], np.asarray(dr.words)):
                raise AssertionError(f"draw {d}: the union's rows are not the draw's words")

    @property
    def n_present(self) -> np.ndarray:
        return self.present.sum(axis=0)

    def matrix(self, source: str) -> np.ndarray:
        """``float64[D, U]``: each draw's score for ``source``, 0 where the row is absent."""
        P = np.zeros((self.D, self.n), dtype=np.float64)
        for d, dr in enumerate(self.draws):
            s = np.asarray(dr.scores[source])
            if s.shape != (self.rowmap[d].size,):
                raise ValueError(f"draw {d} {source}: {s.shape} scores for {self.rowmap[d].size} rows")
            P[d, self.rowmap[d]] = s.astype(np.float64)
        return P

    def votes(self, source: str, decoder: str) -> np.ndarray:
        """``int[U]``: how many draws' own ``decoder`` decode under ``source`` selected each row."""
        v = np.zeros(self.n, dtype=np.int64)
        for d, dr in enumerate(self.draws):
            v[self.rowmap[d][np.asarray(dr.sel[(source, decoder)], dtype=np.int64)]] += 1
        return v

    def draw_selections(self, source: str, decoder: str) -> list[frozenset]:
        return [word_set(dr.words, dr.sel[(source, decoder)]) for dr in self.draws]


def rank_scores(*keys) -> np.ndarray:
    """Exact ranks for a lexicographic key, most significant first, higher is better.

    ``np.lexsort`` is stable, so rows tied on every key stay in row order -- the decoders' own
    tie-break -- and the distinct float64 ranks this returns reproduce that order exactly under
    their ``argsort(-s, kind="stable")``.
    """
    cols = [np.asarray(k, dtype=np.float64) for k in keys]
    order = np.lexsort(tuple(-c for c in reversed(cols)))
    out = np.empty(order.size, dtype=np.float64)
    out[order] = order.size - np.arange(order.size, dtype=np.float64)
    return out


def pooled(u: Union, source: str) -> np.ndarray:
    """``pool``: the mean over draws, absent = 0, summed in sorted order (draw-order free)."""
    return np.sort(u.matrix(source), axis=0).sum(axis=0) / u.D


def aggregate(u: Union, agg: str, source: str, decoder: str | None = None) -> tuple[np.ndarray, dict]:
    """``(scores float64[U], info)`` for one aggregator.  ``decoder`` is the tag whose per-draw
    selections ``vote`` counts; the others ignore it.  ``info`` carries what the ensemble block
    records: the pooled score, and for ``vote`` the votes and the majority mask."""
    base = pooled(u, source)
    info = {"pool": base}
    if agg == "pool":
        return base, info
    if agg == "pool_present":
        P = u.matrix(source)
        return np.sort(P, axis=0).sum(axis=0) / np.maximum(u.n_present, 1), info
    if agg == "vote":
        if decoder is None:
            raise ValueError("vote needs the decoder whose selections it counts")
        v = u.votes(source, decoder)
        major = 2 * v > u.D
        info.update(votes=v, majority=major)
        return rank_scores(major, np.where(major, v, 0), base), info
    raise ValueError(f"unknown aggregator {agg!r}; known: {', '.join(AGGREGATORS)}")


def instance(words: np.ndarray, support: np.ndarray, n_taxa: int):
    """``(gs, graph)`` for a union: the ground set at tau = 1 and its crossing graph, built once and
    shared by every decoder, budget and aggregator on that union.  Every union row has support >= 1,
    so tau = 1 keeps them all -- asserted, since a dropped row would shift every index after it."""
    words = np.ascontiguousarray(words, dtype=np.uint64)
    gs = candidates.build(words, np.asarray(support), int(n_taxa), m=0, tau=1)
    if gs.n_clusters != words.shape[0]:
        raise AssertionError(f"the ground set kept {gs.n_clusters} of {words.shape[0]} rows: a row "
                             f"had support 0")
    return gs, OverlapCSR.of(gs)


class Decoder:
    """``run_gnn_cpu.run_cell``'s decode branch on a fixed candidate set, for many score vectors.

    ``level1``: ``ClusterEnv``'s (L)/(N3O)/budget mask, ``nodecls.decode``, the closure's network
    (``search.realise``).  ``relaxed``: ``RelaxedDecoder`` on the Hasse diagram, ``RX.realise``.
    Both are verified after every decode, as the drivers do.  ``gs`` / ``graph`` may be handed in
    (a draw's own, or :func:`instance`'s); otherwise level1 builds them from ``words`` and
    ``support`` with :func:`instance`, and relaxed needs only ``words``.  Neither decoder reads
    ``support`` or ``m`` when scores are given.
    """

    def __init__(self, n_taxa: int, taxa, k: int, decoder: str, max_level=None, *, words=None,
                 support=None, gs=None, graph=None):
        self.tag = RX.decoder_tag(decoder, max_level)
        self.relaxed = decoder == "relaxed"
        self.n_taxa, self.taxa, self.k, self.max_level = int(n_taxa), list(taxa), int(k), max_level
        if gs is None and not self.relaxed:
            gs, graph = instance(words, support, self.n_taxa)
        self.gs = gs
        self.words = np.ascontiguousarray(gs.words if gs is not None else words, dtype=np.uint64)
        if self.relaxed:
            self.dec = RX.RelaxedDecoder(self.words, self.n_taxa, self.k, max_level)
        else:
            self.graph = graph if graph is not None else OverlapCSR.of(gs)
            self.env = ClusterEnv(gs, budget=self.k, graph=self.graph)

    @property
    def n(self) -> int:
        return int(self.words.shape[0])

    def run(self, scores) -> tuple[np.ndarray, object, dict | None]:
        """``(sorted selection, network or None, decode_stats or None)``."""
        if self.relaxed:
            sel = self.dec.decode(scores)
            if not self.dec.verify():
                raise AssertionError(f"relaxed state fails verify: {self.dec.verify_reason}")
            got = RX.realise(self.words, sel, self.n_taxa, self.taxa)
            stats = dict(self.dec.stats)
        else:
            sel = NC.decode(self.env, scores)
            if not self.env.verify():
                raise AssertionError("the level1 decode is not a feasible cluster system")
            got = search.realise(self.words, sel, self.n_taxa, self.taxa)
            stats = None
        sel = np.sort(np.asarray(sel, dtype=np.int64))
        return sel, (None if got is None else got[0]), stats


def network_fields(net) -> dict:
    """``enewick`` and ``k_pred`` as the drivers record them (-1 and None when nothing realised)."""
    return {"enewick": None if net is None else NW.to_enewick(net),
            "k_pred": -1 if net is None else int(net.n_retics)}


def ensemble_block(u: Union, agg: str, source: str, decoder: str, sel: np.ndarray,
                   info: dict) -> dict:
    """What a cell JSON's ``ensemble`` block records about one ensemble decode."""
    mine = word_set(u.words, sel)
    draws = u.draw_selections(source, decoder)
    npres = u.n_present[sel] if sel.size else np.zeros(0, np.int64)
    out = {"agg": agg, "source": source, "n_draws": u.D, "n_union": u.n,
           "n_per_draw": [int(r.size) for r in u.rowmap],
           "sel_equals_draw": [mine == s for s in draws],
           "sel_equals_a_draw": any(mine == s for s in draws),
           "sel_jaccard_to_draws": [len(mine & s) / max(1, len(mine | s)) for s in draws],
           "sel_in_all_draws": int(np.count_nonzero(npres == u.D)),
           "sel_in_one_draw": int(np.count_nonzero(npres == 1)),
           "sel_mean_present": float(npres.mean()) if npres.size else None}
    if "majority" in info:
        major = np.flatnonzero(info["majority"])
        out.update(n_majority=int(major.size),
                   majority_refused=int(np.setdiff1d(major, sel, assume_unique=True).size),
                   n_voted=int(np.count_nonzero(info["votes"])))
    return out
