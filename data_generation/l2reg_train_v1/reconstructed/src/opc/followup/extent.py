"""F5: candidate-block extent, measured and persisted.

The interim report's §3.5 turns on one number -- *"a single candidate hybrid block whose cycle
spans 43 of 50 taxa makes 99.8% of all triples reticulate-eligible on its own; at CU = 0.5 the 75th
percentile of candidate block extent is 43, at CU = 1.0 it is 28"* -- and that number had **no
committed code and appeared on no artifact**.  :func:`opc.solve.mpl_ilp.hcand_extents` computes an
extent, but it was private, called from one place inside the model build, and never written out.

So the first job here is not analysis, it is making the claim reproducible.  The second is the
comparison the follow-up plan asks for: *candidate* extents against *true* extents, by CU.  If
candidate extents inflate with ILS while the true ones do not, then the coupling's reach is an
artefact of the candidate set rather than of the networks, and capping it is a free tractability
win rather than a modelling choice.

Extent, on both sides, means the same thing
-------------------------------------------

* **Candidate**: for a candidate hybrid ``h``, the union of both endpoints over every overlapping
  pair that meets in ``h`` -- the taxa the block's cycle would span if ``h`` were real.
* **True**: for a real reticulation, the union of :func:`opc.simulate.networks.block_clusters`,
  the clusters of the nodes on its cycle below the apex.  A block induces a complete bipartite
  overlap graph (``block_sides``), so the union over its edges *is* the union over its clusters --
  the two definitions coincide rather than merely resembling each other.
"""

from __future__ import annotations

import numpy as np

from opc import bitset
from opc.simulate import networks as N
from opc.solve import mpl_ilp

__all__ = [
    "QUANTILES",
    "candidate_extents",
    "true_extents",
    "summarise",
    "eligible_fraction",
    "capped_counts",
    "bitset_span",
]

#: Reported quantiles.  p75 is here because it is the statistic the interim report quoted.
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def candidate_extents(ground_set) -> np.ndarray:
    """``|extent(h)|`` for every candidate hybrid in the ground set, over all components."""
    out = []
    for comp in ground_set.components:
        if not comp.n_hcand:
            continue
        _, extent = mpl_ilp.hcand_extents(comp, ground_set.words, ground_set.n_taxa)
        out.append(extent.sum(axis=1))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)


def true_extents(net) -> np.ndarray:
    """``|extent|`` for every real reticulation of ``net``."""
    out = []
    for ret in net.retics:
        clusters = N.block_clusters(net, ret)
        span: set[int] = set()
        for c in clusters:
            span |= set(c)
        out.append(len(span))
    return np.asarray(out, dtype=int)


def eligible_fraction(extent: int, n_taxa: int) -> float:
    """Share of triples a single block of this extent makes reticulate-eligible.

    ``1 - C(n - E, 3) / C(n, 3)`` -- a triple is eligible for that block exactly when it meets the
    block's extent.  This is §0.3's arithmetic, in code so the table in the report is generated
    rather than typed.
    """
    from math import comb

    if extent <= 0 or n_taxa < 3:
        return 0.0
    rest = max(0, n_taxa - int(extent))
    return 1.0 - (comb(rest, 3) / comb(n_taxa, 3) if rest >= 3 else 0.0)


def summarise(values, prefix: str, n_taxa: int = 0) -> dict:
    """Quantiles of an extent sample, plus what its p75 block would make eligible."""
    v = np.asarray(values, dtype=float)
    row = {f"{prefix}_n": int(v.size)}
    if not v.size:
        return row | {f"{prefix}_q{int(q * 100)}": float("nan") for q in QUANTILES}
    qs = np.quantile(v, QUANTILES)
    row |= {f"{prefix}_q{int(q * 100)}": float(x) for q, x in zip(QUANTILES, qs, strict=True)}
    row[f"{prefix}_mean"] = float(v.mean())
    if n_taxa:
        row[f"{prefix}_eligible_at_q75"] = eligible_fraction(int(round(qs[3])), n_taxa)
    return row


def capped_counts(ground_set, caps) -> dict:
    """How many candidate blocks survive each extent cap, and their coupling share.

    The coupling a block generates is proportional to the triples it makes eligible, so the second
    number is the one that predicts the nonzero count -- ``build_model``'s reticulate block is 93%
    of the matrix.
    """
    ext = candidate_extents(ground_set)
    n = ground_set.n_taxa
    total_elig = sum(eligible_fraction(int(e), n) for e in ext)
    out = {"n_hcand": int(ext.size)}
    for cap in caps:
        keep = ext[ext <= cap] if cap else ext
        out[f"n_hcand_le{cap or 'inf'}"] = int(keep.size)
        share = sum(eligible_fraction(int(e), n) for e in keep)
        out[f"coupling_share_le{cap or 'inf'}"] = float(share / total_elig) if total_elig else 0.0
    return out


def bitset_span(words: np.ndarray, n_taxa: int) -> int:
    """Taxa covered by a set of packed cluster rows -- used for ad-hoc checks."""
    if not words.size:
        return 0
    return int(bitset.to_dense(np.ascontiguousarray(words), n_taxa).astype(bool).any(axis=0).sum())
