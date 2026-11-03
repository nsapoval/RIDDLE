"""The GNN as a tree-of-blobs resolver: the best-scoring cluster system a decode can build inside a TOB.

A tree of blobs (TOB) says where reticulation happened and leaves each blob a polytomy.  BROOQS and
NetCS take one and resolve every eligible polytomy into a reticulation.  This module does the same
job with the GNN's scores: the decode is the landed score-ordered greedy, with its legal set cut down
to what the TOB allows.

The two rules
-------------

**Every candidate that crosses a TOB cluster is removed** (``keep``).  A cluster system that keeps
every TOB cluster cannot contain one that overlaps it, and the TOB's clusters are exactly the
bridges of any network that refines it, so a cycle can only form among clusters nested inside one
polytomy.

**Every TOB cluster is in the output** (``forced``), as with both resolvers, which never remove a
cluster they are given.  A TOB cluster that is already a candidate crosses no kept row, so it is
always legal and a decode run to termination accepts it whatever its score.  One that is *not* a
candidate (``missing`` -- the tau = 2 candidate set lacks it) is appended to the selection before
the network is realised.  It is compatible with every selected row, so it costs no budget and opens
no class.

Scores come from the forward pass over the **whole** candidate set, exactly as the model was trained
and as every landed row was computed.  Only the decode is constrained.

Decoders
--------

``level1``: the landed :class:`~opc.rl.env.ClusterEnv` over the **unrestricted** ground set, with
the mask intersected with ``keep``.  This is exact, not an approximation of a rebuilt sub-instance:
(L), (N3O) and the budget read only *accepted* rows, and a row outside ``keep`` is never accepted,
so the state is the one a ground set of ``keep`` alone would hold.  It is also why the landed
decode is reproduced when the TOB is the GNN's own (``prefix_tob_eval``'s arm b), which
``tob_resolve_eval/check_tob_resolve.py`` gates.  The kernel is pre-accepted as landed, unless a
kernel row crosses the TOB.  The kernel is the degree-0 rows of the *full* overlap graph, so that
can happen only through a TOB cluster that is not a candidate.  The env is then built without the
kernel, and the compatible kernel rows are handed to it as ``given``.

``relaxed``: :class:`~opc.rl.relaxed.RelaxedDecoder` on the kept rows plus the missing TOB rows,
with every TOB row scored above the best candidate so the tree is laid down first.  Its legality is
not hereditary, so a crossing row cannot simply be masked: it could be accepted late, when no
better row is legal.  The words handed to it are therefore the restricted set itself.  All TOB rows
must be accepted, and this is asserted.

The budget is whatever the caller passes.  ``tob_resolve_eval`` passes ``k_TOB``, the number of
polytomies BROOQS would resolve.  The model also reads ``k`` through its global feature, so the
caller recomputes the forward at that ``k`` (``run_gnn_cpu.py --tob-manifest``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc import bitset
from opc.followup import search
from opc.rl import camus_nets as CN
from opc.rl import relaxed as RX
from opc.rl.env import ClusterEnv
from opc.simulate import networks as NW

__all__ = ["Restriction", "tob_words", "restrict", "resolve_level1", "resolve_relaxed",
           "MODES", "mode_name"]

#: The two resolve modes, as cell JSONs and CSVs name them.
MODES = ("strict_L1", "relaxed")


def mode_name(mode: str, max_level: int | None = None) -> str:
    return "strict_L1" if mode == "strict_L1" else f"relaxed_L{int(max_level)}"


def tob_words(text: str, taxa: list[str], outgroup: str = "OUT") -> np.ndarray:
    """The TOB's nontrivial ingroup clusters as ``uint64[T, W]`` in ``taxa``'s index order, sorted
    by their byte keys so the row order is a function of the cluster set alone.

    ``text`` is an OUT-rooted TOB (``(OUT,<body>);``), or a rooted TOB over ``taxa`` exactly.  Its
    ingroup must be ``taxa``; a TOB over another taxon set would map clusters onto the wrong rows,
    so that raises."""
    net = NW.from_enewick(text)
    ingroup = CN.ingroup_of(net, outgroup) if outgroup in net.labels else list(net.labels)
    if set(ingroup) != set(taxa):
        diff = set(ingroup) ^ set(taxa)
        raise ValueError(f"TOB taxa differ from the cell's ({len(diff)} differ, e.g. "
                         f"{sorted(diff)[:3]})")
    if net.n_retics:
        raise ValueError(f"a tree of blobs has no reticulations; this one has {net.n_retics}")
    n = len(taxa)
    index = {t: i for i, t in enumerate(taxa)}
    rows = [bitset.from_indices(sorted(index[x] for x in c), n)
            for c in CN.restrict_clusters(net, ingroup)]
    rows.sort(key=lambda r: r.tobytes())
    return bitset.pack_rows(rows, n)


def crossing_mask(words: np.ndarray, tob: np.ndarray, n_taxa: int) -> np.ndarray:
    """``bool[K]``: row crosses some TOB cluster (they intersect and neither contains the other)."""
    if not tob.shape[0] or not words.shape[0]:
        return np.zeros(words.shape[0], dtype=bool)
    D = bitset.to_dense(words, n_taxa).astype(np.float32)
    T = bitset.to_dense(tob, n_taxa).astype(np.float32)
    inter = D @ T.T
    a, b = D.sum(axis=1)[:, None], T.sum(axis=1)[None, :]
    return ((inter > 0) & (inter < a) & (inter < b)).any(axis=1)


@dataclass
class Restriction:
    """What a TOB allows of one candidate set."""

    keep: np.ndarray            # bool[K]: crosses no TOB cluster
    forced_rows: np.ndarray     # int64: candidate rows that are TOB clusters
    missing: np.ndarray         # uint64[M, W]: TOB clusters that are not candidates
    n_tob: int
    kernel_crossing: int        # kernel rows crossing the TOB (only through a missing cluster)

    def as_dict(self) -> dict:
        return {"n_tob_clusters": self.n_tob, "n_forced_present": int(self.forced_rows.size),
                "n_forced_missing": int(self.missing.shape[0]), "n_cand": int(self.keep.size),
                "n_cand_kept": int(self.keep.sum()), "kernel_crossing": self.kernel_crossing}


def restrict(gs, tob: np.ndarray) -> Restriction:
    n = int(gs.n_taxa)
    keep = ~crossing_mask(gs.words, tob, n)
    where = {bitset.key(w): i for i, w in enumerate(gs.words)}
    present, missing = [], []
    for t in tob:
        i = where.get(bitset.key(t))
        (present if i is not None else missing).append(i if i is not None else t)
    forced = np.asarray(sorted(present), dtype=np.int64)
    if forced.size and not keep[forced].all():
        raise AssertionError("a TOB cluster crosses another: the TOB is not a tree")
    kernel = np.asarray(gs.decomposition.kernel, dtype=np.int64)
    return Restriction(keep=keep, forced_rows=forced,
                       missing=bitset.pack_rows(missing, n),
                       n_tob=int(tob.shape[0]),
                       kernel_crossing=int((~keep[kernel]).sum()) if kernel.size else 0)


def _rank(scores, n: int) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (n,):
        raise ValueError(f"scores must be ({n},), got {s.shape}")
    order = np.argsort(-s, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def _realise(words, sel, missing, n, taxa, realise):
    """Realise ``sel`` plus every missing TOB cluster; ``(enewick | None, k_pred)``."""
    allw = np.ascontiguousarray(np.vstack([words, missing])) if missing.shape[0] else words
    full = np.concatenate([np.asarray(sel, dtype=np.int64),
                           words.shape[0] + np.arange(missing.shape[0], dtype=np.int64)])
    got = realise(allw, full, n, taxa)
    if got is None:
        return None, -1
    return NW.to_enewick(got[0]), int(got[0].n_retics)


def resolve_level1(gs, graph, scores, budget: int, R: Restriction, taxa, *,
                   network: bool = True) -> dict:
    """The landed level-1 decode with its mask cut to ``R.keep`` (module docstring)."""
    rank = _rank(scores, gs.n_clusters)
    if R.kernel_crossing:
        env = ClusterEnv(gs, budget=budget, graph=graph, include_kernel=False)
        kernel = np.asarray(gs.decomposition.kernel, dtype=np.int64)
        obs = env.reset(init="given", given=kernel[R.keep[kernel]])
    else:
        env = ClusterEnv(gs, budget=budget, graph=graph)
        obs = env.reset()
    steps = 0
    while True:
        mask = obs["legal"] & R.keep
        if not mask.any():
            break
        cand = np.flatnonzero(mask)
        obs, _r, _done, _info = env.step(int(cand[np.argmin(rank[cand])]))
        steps += 1
    sel = np.sort(env.selection.astype(np.int64))
    if not env.verify():
        raise AssertionError("the restricted level-1 decode fails its own check")
    if not R.keep[sel].all():
        raise AssertionError("the restricted decode accepted a row that crosses the TOB")
    if not np.isin(R.forced_rows, sel).all():
        raise AssertionError("a TOB cluster that is a candidate was not accepted")
    out = {"selection": sel, "n_steps": steps, "n_hybrids": int(env.state.n_hybrids)}
    if network:
        out["enewick"], out["k_pred"] = _realise(gs.words, sel, R.missing, gs.n_taxa, taxa,
                                                 search.realise)
    return out


def resolve_relaxed(words: np.ndarray, n_taxa: int, scores, budget: int, max_level: int,
                    R: Restriction, taxa, *, network: bool = True) -> dict:
    """``RelaxedDecoder`` on the kept rows plus the missing TOB rows, TOB rows first."""
    s = np.asarray(scores, dtype=np.float64)
    idx = np.flatnonzero(R.keep)
    W = np.ascontiguousarray(np.vstack([words[idx], R.missing])) if R.missing.shape[0] \
        else np.ascontiguousarray(words[idx])
    top = (float(s.max()) if s.size else 0.0) + 1.0
    sub = np.concatenate([s[idx], np.full(R.missing.shape[0], top)])
    pos = np.searchsorted(idx, R.forced_rows)
    sub[pos] = top
    forced = np.concatenate([pos, idx.size + np.arange(R.missing.shape[0])]).astype(np.int64)
    dec = RX.RelaxedDecoder(W, n_taxa, budget, max_level)
    sel = dec.decode(sub)
    if not dec.verify():
        raise AssertionError(f"the restricted relaxed decode fails its own check: "
                             f"{dec.verify_reason}")
    if not np.isin(forced, sel).all():
        raise AssertionError("the relaxed decode refused a TOB cluster")
    rows = idx[sel[sel < idx.size]]
    out = {"selection": np.sort(rows), "stats": dec.stats}
    if network:
        got = RX.realise(W, sel, n_taxa, taxa)
        out["enewick"] = None if got is None else NW.to_enewick(got[0])
        out["k_pred"] = -1 if got is None else int(got[0].n_retics)
    return out
