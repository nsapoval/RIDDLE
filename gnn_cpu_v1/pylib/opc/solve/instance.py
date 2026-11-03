"""Gene trees -> clusters, support, and MDC weights: the MW-GTCS instance (WS1-1).

The weight of a cluster is Than-Nakhleh's, ``w(A) = shift + 1 - sum_T alpha(A, T)``, where
``alpha(A, T)`` is the number of extra lineages ``A`` forces on gene tree ``T`` under MDC.

Two corrections to the handoff spec, both load-bearing.

**The shift is not the locus count.**  The handoff writes ``w(A) = m + 1 - sum alpha`` and glosses
``m`` as the number of gene trees.  In Than-Nakhleh (Eq. 3) the symbol is ``max_A sum_i alpha``,
which is what makes every weight positive; it is a constant for them only because their species
tree is binary, so ``|S|`` is fixed.  Read as the locus count it puts **65%** of a real ``C_tau``
at ``w <= 0`` -- those clusters become unselectable, ``k=0`` stops returning a binary tree, and
"isolated clusters are accepted unconditionally" becomes false.  So this module stores
``sum_T alpha`` and nothing else; :func:`weights` derives ``w`` under a named convention, and the
choice is recorded rather than baked into the artifact.

**alpha collapses, so no gene tree is ever walked twice.**  ``alpha(A,T) = k(A,T) - 1`` with
``k`` the number of *maximal* nodes ``v`` of ``T`` having ``c(v) <= A``.  Since
``c(v) <= c(parent(v))``, the indicator ``[c(v) <= A]`` is monotone up the tree and the maximal
count telescopes::

    k(A,T)      = |A| - sum_{internal v of T} (children(v) - 1) * [c(v) <= A]
    alpha(A,T)  = |A| - 1 - #{internal v of T : c(v) <= A}                    (rooted binary T)
    sum_T alpha = m * (|A| - 1) - sum_{B observed} support(B) * [B <= A]

and the last line is a **subset-sum over the cluster table that already exists** -- the same
object :class:`opc.clusters.extract.SupportCounter` accumulates.  ``SupportCounter`` drops
singletons and the full taxon set, which is exactly right here: no internal node of a binary tree
is a singleton, and the root's cluster ``X`` is never a subset of a proper ``A``.

Because the derivation counts *nodes*, it needs each locus to be rooted and binary.
``SupportCounter.n_nonbinary_loci`` reports violations and :func:`build_instance` refuses to
produce weights when it is non-zero.

**The weight function is an axis, not a constant.**  A21 measured that MDC weights are what stops
the ILP closing: ``CV(w) = 0.203`` puts the edge-formulation LP at ``sum w / 2`` with 98% of
variables at exactly one half, while raw support (``CV = 1.388``) closes the same 309-vertex
component in 0.2 s.  A *shift* cannot fix that -- every convention above is a translation, so the
ranking is identical and ``LP/IP`` is 3.12 with or without one.  A **scheme** changes the ranking,
and :data:`WEIGHT_SCHEMES` names the three worth comparing:

``mdc``            Than-Nakhleh's, under a shift convention.  The anchor: B1's T1 is a
                   conformance test *of this scheme*, not of the solver in general.
``support``        ``support(A)``.  The number of loci in which ``alpha(A,T) = 0``, so it counts
                   how often ``A`` is exactly right and ignores how badly the rest miss.
``mdc-per-taxon``  ``(shift + 1 - sum alpha) / (|A| - 1)``.  Shift-free MDC is
                   ``N(A) = sum_B support(B)*[B subset-of A]`` up to a constant, i.e. the support
                   *mass nested under* ``A``, which grows with ``|A|`` -- that size scaling is
                   what flattens the contrast among survivors, since a large cluster is nearly
                   always a clade of most loci.  Dividing it out leaves support depth per taxon.

Which one recovers the true clusters is an empirical question, and until it is answered no scheme
is privileged in the outputs: ``run_solver.py`` carries ``scheme`` as a column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc import bitset

__all__ = [
    "SHIFT_CONVENTIONS",
    "WEIGHT_SCHEMES",
    "Instance",
    "subset_support_sums",
    "alpha_sums",
    "weights",
    "weight_vector",
    "contrast",
    "build_instance",
]

#: Largest source/target tile.  The float32 accumulation below is exact while
#: ``tile * m_max < 2**24``; 2048 x 5000 = 1.02e7 clears it, and a 2048x2048 float32 tile is
#: 16 MB, which stays reasonable at ``--jobs 190``.
TILE = 2048

SHIFT_CONVENTIONS = ("than-nakhleh", "loci", "none")

#: Weight families.  Unlike a shift these change the *ranking* of clusters, so they are the axis
#: that can move the ILP's closure boundary -- see the module docstring.
WEIGHT_SCHEMES = ("mdc", "support", "mdc-per-taxon")


def _shift_value(convention: str, alpha_sum: np.ndarray, n_loci: int) -> int:
    """The constant ``w = shift + 1 - sum_alpha`` is measured from.

    ``than-nakhleh``  ``max_A sum_T alpha(A,T)``, the paper's own reading -- every weight >= 1.
    ``loci``          the handoff spec's ``m``.  Kept so the difference can be *measured*, not
                      argued about; it makes most weights negative.
    ``none``          ``w = -sum_alpha``, i.e. plain MDC cost with no per-cluster reward.  Then
                      the objective is "minimise deep coalescences" with no resolution pressure.
    """
    if convention == "than-nakhleh":
        return int(alpha_sum.max()) if alpha_sum.size else 0
    if convention == "loci":
        return int(n_loci)
    if convention == "none":
        return -1  # so that shift + 1 == 0
    raise ValueError(f"unknown shift convention {convention!r}")


@dataclass
class Instance:
    """One cell's MW-GTCS ground set, at every ``m`` checkpoint."""

    n_taxa: int
    checkpoints: tuple[int, ...]
    words: np.ndarray  # uint64[K, W]  -- the stored clusters (support >= 2 at the last m)
    support: np.ndarray  # int32[K, n_bins], cumulative
    alpha_sum: np.ndarray  # int64[K, n_bins]  -- sum_T alpha(A, T)
    n_clusters_raw: np.ndarray
    n_source_clusters: int
    n_nonbinary_loci: int

    @property
    def sizes(self) -> np.ndarray:
        return bitset.popcounts(self.words)

    def at(self, m: int):
        """``(words, support, alpha_sum)`` restricted to checkpoint ``m``."""
        bi = list(self.checkpoints).index(m)
        return self.words, self.support[:, bi], self.alpha_sum[:, bi]


def subset_support_sums(
    source_words: np.ndarray,
    source_support: np.ndarray,
    target_words: np.ndarray,
    n_taxa: int,
    *,
    tile: int = TILE,
) -> np.ndarray:
    """``N[a, bin] = sum_b support[b, bin] * [source_b subset-of target_a]``.

    The zeta transform of the support measure over the containment order, restricted to the
    clusters actually observed.  Computed with the same exact float32 GEMM as the overlap kernel:
    intersection counts are integers well below ``2**24``, so ``ic == |B|`` is an exact subset
    test.  Both axes are tiled -- a full ``|sources| x |targets|`` boolean would be 10 GB at the
    worst cell -- and the per-tile products are bounded by ``tile * m_max < 2**24``, so the
    float32 matmul is exact before it is accumulated into float64.
    """
    n_src, n_tgt = source_words.shape[0], target_words.shape[0]
    n_bins = source_support.shape[1]
    out = np.zeros((n_tgt, n_bins), dtype=np.float64)
    if n_src == 0 or n_tgt == 0:
        return out.astype(np.int64)

    max_support = int(source_support.max()) if source_support.size else 0
    if tile * max(1, max_support) >= 2 ** 24:
        tile = max(1, (2 ** 24 - 1) // max(1, max_support))

    src_sizes = bitset.popcounts(source_words).astype(np.float32)
    for t0 in range(0, n_tgt, tile):
        t1 = min(t0 + tile, n_tgt)
        tgt_dense = bitset.to_dense(target_words[t0:t1], n_taxa)
        acc = np.zeros((n_bins, t1 - t0), dtype=np.float64)
        for s0 in range(0, n_src, tile):
            s1 = min(s0 + tile, n_src)
            src_dense = bitset.to_dense(source_words[s0:s1], n_taxa)
            ic = src_dense @ tgt_dense.T                       # |B & A|, exact
            inside = (ic == src_sizes[s0:s1, None]).astype(np.float32)
            acc += (source_support[s0:s1].T.astype(np.float32) @ inside).astype(np.float64)
        out[t0:t1] = acc.T
    return np.rint(out).astype(np.int64)


def alpha_sums(
    source_words: np.ndarray,
    source_support: np.ndarray,
    target_words: np.ndarray,
    checkpoints,
    n_taxa: int,
    *,
    tile: int = TILE,
) -> np.ndarray:
    """``sum_T alpha(A, T)`` for every target ``A`` and every ``m`` checkpoint.

    ``sum_T alpha = m * (|A| - 1) - N(A)`` -- see the module docstring.  ``m`` varies by
    checkpoint and ``N`` is cumulative in the same nested-prefix sense ``support`` is, so one
    subset-sum serves every checkpoint.
    """
    n_sub = subset_support_sums(
        source_words, source_support, target_words, n_taxa, tile=tile
    )
    sizes = bitset.popcounts(target_words).astype(np.int64)
    m = np.asarray(list(checkpoints), dtype=np.int64)
    return m[None, :] * (sizes[:, None] - 1) - n_sub


def weights(alpha_sum: np.ndarray, n_loci: int, convention: str = "than-nakhleh") -> np.ndarray:
    """``w = shift + 1 - sum_T alpha`` under a named shift convention.

    Deliberately a function of the stored ``alpha_sum`` rather than a column: the shift is a
    modelling choice with a >100x effect on ILP solve time, and it must be taken over a *fixed*
    candidate set -- re-thresholding a stored ``w`` would silently change the objective.
    """
    a = np.asarray(alpha_sum, dtype=np.int64)
    return _shift_value(convention, a, n_loci) + 1 - a


def weight_vector(
    alpha_sum: np.ndarray,
    support: np.ndarray,
    sizes: np.ndarray,
    n_loci: int,
    *,
    scheme: str = "mdc",
    convention: str = "than-nakhleh",
) -> np.ndarray:
    """The objective coefficients under a named :data:`WEIGHT_SCHEMES` family.

    All three arrays are indexed by the same candidate set and must already be restricted to it:
    the shift is ``max_A sum alpha`` *over the candidates*, so taking it before filtering and
    re-using it after would silently change the objective -- the trap :func:`weights` documents.

    Returns float64 even where the scheme is integral, because ``mdc-per-taxon`` is not and the
    solver should not see a dtype that depends on the flag.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme {scheme!r}")
    if scheme == "support":
        return np.asarray(support, dtype=np.float64)
    w = weights(alpha_sum, n_loci, convention).astype(np.float64)
    if scheme == "mdc":
        return w
    size = np.asarray(sizes, dtype=np.float64)
    # |A| >= 2 for every stored cluster -- SupportCounter drops singletons -- so this never
    # divides by zero, and a two-taxon cluster is left alone.
    return w / np.maximum(size - 1.0, 1.0)


def contrast(w: np.ndarray) -> float:
    """``CV = std / mean``, the one statistic that predicts whether the ILP will close.

    A21: 0.203 under MDC (LP pinned at ``sum w / 2``, no closure at 300 s) against 1.388 under
    raw support (0.2 s).  Recorded per component so the closure boundary can be read against it
    rather than against vertex count alone.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return float("nan")
    mean = w.mean()
    return float(w.std() / mean) if mean else float("nan")


def build_instance(
    counter,
    n_taxa: int,
    *,
    require_binary: bool = True,
    tile: int = TILE,
    min_target_support: int = 2,
) -> Instance:
    """Turn a finished :class:`~opc.clusters.extract.SupportCounter` into an :class:`Instance`.

    Sources are **every** observed cluster, support 1 included: it is a real internal node of a
    real gene tree and it contributes to every ``alpha``, so dropping it would bias every weight by
    an amount that varies with ``n``.

    ``min_target_support`` is which of those become *candidates*.  The default of 2 was lossless
    for the percolation study, where ``tau >= 2`` always.  It is **not** lossless at ``tau = 0``,
    the headline setting of ``exp-plan-n50-mpl-v0.1``: 51% of the candidate set has support exactly
    1, and hybrid-signature clusters are over-represented there by construction, since a
    reticulation splits its cluster's support across two parental histories (A29).  Pass 1 for a
    tau = 0 ground set.
    """
    src_masks, src_support, n_raw = counter.finalize(min_support=1)
    n_words = bitset.n_words(n_taxa)
    if len(src_masks):
        source_words = np.stack([
            np.frombuffer(int(m).to_bytes(n_words * 8, "little"), dtype=np.uint64)
            for m in src_masks
        ])
    else:
        source_words = np.zeros((0, n_words), dtype=np.uint64)

    keep = np.flatnonzero(src_support[:, -1] >= min_target_support)
    words = np.ascontiguousarray(source_words[keep])
    support = np.ascontiguousarray(src_support[keep])

    if require_binary and counter.n_nonbinary_loci:
        raise ValueError(
            f"{counter.n_nonbinary_loci} loci are not rooted binary; the MDC weight counts "
            "internal nodes and would be wrong.  Pass require_binary=False to compute anyway."
        )

    alpha = alpha_sums(
        source_words, src_support, words, counter.checkpoints, n_taxa, tile=tile
    )
    return Instance(
        n_taxa=n_taxa,
        checkpoints=tuple(counter.checkpoints),
        words=words,
        support=support,
        alpha_sum=alpha,
        n_clusters_raw=np.asarray(n_raw),
        n_source_clusters=int(source_words.shape[0]),
        n_nonbinary_loci=int(counter.n_nonbinary_loci),
    )
