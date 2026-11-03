"""Per-node GNN scores on an emitted network: which candidate each node is, and its logit.

A realised network's node ids are rows of ``full``, the word array ``realise`` returns beside it
(``opc.followup.search.realise``, ``opc.rl.relaxed.realise``).  That makes the map from a node to
the candidate the decoder scored a lookup by word content -- and it has to be by content, not by
position:

* the **level1** path runs ``closure.closure_words``, which re-sorts the rows by bytes and can add
  intersection clusters that were never selected, so ``full[v]`` and ``sel[v]`` are unrelated;
* the **relaxed** path appends the trivial clusters after the selection, so ``full[v] ==
  words[sel[v]]`` for ``v < len(sel)``.  That identity is asserted by :func:`node_rows`' callers,
  not assumed by it.

``z`` is the raw logit, the scale ``cluster_anatomy``'s ``z < 6`` flag was set on.  ``z_std``
standardises it over the whole candidate pool of one forward pass -- per ``k``, because ``k`` is a
global feature and moves every logit.
"""

from __future__ import annotations

import numpy as np

from opc import bitset
from opc.simulate import networks as N

LABEL_FMT = "{:.2f}"


def node_rows(full: np.ndarray, cand_words: np.ndarray) -> np.ndarray:
    """``rows[v]``: the candidate row whose word is ``full[v]``, or -1 if the pool has none.

    Trivial clusters (singletons, the taxon set) and closure-added intersections are the -1s.
    """
    full = np.ascontiguousarray(full)
    cand = np.ascontiguousarray(cand_words)
    index = {cand[i].tobytes(): i for i in range(cand.shape[0])}
    return np.array([index.get(full[v].tobytes(), -1) for v in range(full.shape[0])],
                    dtype=np.int64)


def pool_stats(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64)
    return {"n": int(z.size), "mean": float(z.mean()), "sd": float(z.std(ddof=0))}


def node_kind(net: N.Network, v: int, row: int) -> str:
    """``hybrid`` / ``tree`` for a scored node, ``closure`` for an internal node the pool lacks."""
    if net.is_hybrid(v):
        return "hybrid"
    return "tree" if row >= 0 else "closure"


def tree_labels(net: N.Network, rows: np.ndarray, z: np.ndarray) -> dict[int, str]:
    """``to_enewick``'s ``node_label``: every internal tree node that is a scored candidate."""
    return {v: LABEL_FMT.format(float(z[rows[v]])) for v in range(net.n_nodes)
            if v != net.root and not net.is_leaf(v) and not net.is_hybrid(v) and rows[v] >= 0}


def cluster_rows(net: N.Network, full: np.ndarray, rows: np.ndarray, *, z: np.ndarray,
                 p: np.ndarray, support: np.ndarray, selected: np.ndarray, n_taxa: int,
                 taxa_order, stats: dict, **const) -> list[dict]:
    """One dict per non-leaf, non-root network node: what it is, and what the GNN said of it.

    ``rank_frac`` is the node's rank in ``p`` over the whole pool, 0 = top -- what the decode
    walked.  ``const`` (``ftau``, ``tau``, ``decoder``, ``k``) is repeated on every row.
    """
    sel = set(int(r) for r in np.asarray(selected).tolist())
    order = np.argsort(-np.asarray(p, dtype=np.float64), kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    sd = stats["sd"] or 1.0
    out = []
    for v in range(net.n_nodes):
        if v == net.root or net.is_leaf(v):
            continue
        r = int(rows[v])
        idx = bitset.to_indices(full[v], n_taxa)
        have = r >= 0
        out.append({
            **const, "node": v, "kind": node_kind(net, v, r), "cand_row": r,
            "taxa": ",".join(sorted(taxa_order[i] for i in idx)), "size": len(idx),
            "support": int(support[r]) if have else -1,
            "z": float(z[r]) if have else float("nan"),
            "z_std": (float(z[r]) - stats["mean"]) / sd if have else float("nan"),
            "p": float(p[r]) if have else float("nan"),
            "rank_frac": float(rank[r] / max(1, rank.size - 1)) if have else float("nan"),
            "selected": int(r in sel),
        })
    return out
