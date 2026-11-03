"""E3: what the clustering system survives being pushed through a coarser statistic.

Three quotients of the same enumeration, and they answer different questions.  Keeping them apart
is most of the work, because the plan's E3 and the plan's E4 are asking about *different* maps and
the answers turn out opposite.

``R(Psi)`` -- the displayed triple system
    The plan's E3 question, and Yu-Nakhleh's.  Two networks in one fiber display exactly the same
    rooted triples, and the plan cites their Fig. 2 result that this forces MPL to be flat between
    them.  **This is the quotient gate G3's threshold was pre-registered against.**

the **shape system** ``{(tau, shape key)}``
    Every triple's trinet, as :mod:`opc.mpl.trinets` keys it.  This is what a per-triple profiled
    objective can distinguish when it carries the whole I2 catalog: two networks in one fiber get
    the same ``W`` column for every triple and every shape, so the E4 relaxation scores them
    identically **whatever the data are**.

the **class vector** ``{(tau, resolution or RETICULATE)}``
    The same thing after the catalog is collapsed to five classes per triple -- the reduction that
    makes E4's coupling writable at n = 50.  Necessarily coarser than the shape system, so its
    fibers are at least as large, and the difference between the two is exactly what the catalog's
    granularity buys the bound.

Flatness is measured, not inherited
-----------------------------------

``R(Psi) = R(Psi')`` implying MPL flatness is an implication in one direction only, so once the
``R`` fibers turn out to be singletons it says nothing at all -- it does not follow that MPL can
separate anything.  :func:`flatness_kl` therefore asks the question directly: hand ``Psi'`` the
triple probabilities ``Psi`` generates and let its own optimiser try to reproduce them.  The
residual is a KL divergence in nats per locus, zero exactly when MPL cannot tell the two apart at
those parameters.

Its control is :func:`flatness_kl` against ``Psi`` itself, which must return zero.  That is the
same device E2a's ``embedding_deficit`` is -- an invariant that catches an optimiser failure before
it can be read as a finding (A34).
"""

from __future__ import annotations

import numpy as np

from opc.mpl import likelihood, profile

__all__ = [
    "QUOTIENTS",
    "fiber_sizes",
    "shape_system",
    "flatness_kl",
    "verdict",
]

#: The quotient maps, by the attribute of :class:`~opc.theory.galled.GalledInstance` that carries
#: them.  ``shape`` is absent because it needs a ``Network`` and is computed separately.
QUOTIENTS = ("triples", "classes")


def fiber_sizes(keys) -> np.ndarray:
    """Fiber sizes, one entry per fiber, given one quotient key per **distinct** system."""
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    _, counts = np.unique(np.asarray(keys), return_counts=True)
    return counts


def shape_system(net, index=None) -> tuple[str, ...]:
    """``(shape key,)`` per taxon triple, in triple-rank order -- the I2-catalog quotient.

    Uses :func:`opc.mpl.trinets.restrict_to_triple` rather than a second implementation, which
    makes it the slow path: ~30 us per triple against ~45 us for a whole instance in the
    enumeration.  Exhaustive at n <= 6, sampled above; ``scripts/e3_fibers.py`` records which.
    """
    from opc.mpl import trinets
    from opc.mpl.triples import triple_index

    idx = triple_index(net.n_taxa) if index is None else index
    return tuple(trinets.restrict_to_triple(net, tuple(row)).key for row in idx)


def flatness_kl(net_a, net_b, *, params_a=None, scale: float = 1e6, restarts: int = 4,
                maxiter: int = 300, seed: int = 0) -> float:
    """Total KL from ``Psi_a``'s triple distribution to the closest ``Psi_b`` can reach, in nats.

    ``sum_tau KL(P(.|Psi_a, theta_a) || P(.|Psi_b, theta_b*))`` with ``theta_b*`` fitted by
    :func:`opc.mpl.likelihood.optimise`.  Zero means ``Psi_b`` reproduces ``Psi_a`` exactly and MPL
    is flat between them; anything else is the log-likelihood ``Psi_b`` gives up, per locus, at an
    infinite number of loci.

    Computed as a **saturation gap** so it reuses the machinery I3 is already validated on: the
    pseudo-data are ``scale * P(.|Psi_a)``, the saturated value is what a model that fits them
    perfectly would score, and the gap is the shortfall.  ``scale`` cancels out of the returned
    number and only sets the arithmetic's dynamic range.
    """
    model_a = likelihood.TripleModel(net_a)
    p_a = model_a.probabilities(params_a)
    obs = scale * p_a
    saturated = sum(profile.saturated_log_likelihood(row) for row in obs)
    model_b = likelihood.TripleModel(net_b)
    _, fitted = likelihood.optimise(model_b, obs, maxiter=maxiter, restarts=restarts, seed=seed)
    return float((saturated - fitted) / scale)


def verdict(rows) -> tuple[str, dict]:
    """Gate G3, on the quotient it was pre-registered against.

    Pre-registered with Nick **before the enumeration ran**, because the plan states G3 as "E3
    shows workable fibers" and never says what workable means:

    * **GREEN** -- at least 90% of clustering systems sit in singleton fibers of ``C -> R``, *and*
      the fiber-size distribution does not grow with ``n``;
    * **AMBER** -- at least 50%, or non-singleton with median fiber size <= 2 and structurally
      characterised, so results can be reported as equivalence classes;
    * **RED** -- fiber sizes grow with ``n``: the quotient gets worse exactly where the study
      lives, and what the search recovers is not a network.

    ``rows`` is one dict per ``n``, carrying ``n_taxa``, ``n_systems``, ``n_singleton``,
    ``median_fiber`` and ``max_fiber``.
    """
    rows = sorted(rows, key=lambda r: r["n_taxa"])
    if not rows:
        return "NOT_RUN", {}
    total = sum(r["n_systems"] for r in rows)
    singleton = sum(r["n_singleton"] for r in rows)
    frac = singleton / total if total else 0.0
    maxima = [r["max_fiber"] for r in rows]
    medians = [r["median_fiber"] for r in rows]
    # "Grows with n" is read off the largest fiber, which is the tail G3 cares about: a median that
    # stays at 1 while the maximum climbs is still a quotient that gets worse.
    grows = len(rows) > 1 and maxima[-1] > maxima[0] and maxima == sorted(maxima)
    info = {
        "n_systems": total,
        "frac_in_singleton_fiber": frac,
        "max_fiber_by_n": {r["n_taxa"]: r["max_fiber"] for r in rows},
        "median_fiber_by_n": {r["n_taxa"]: r["median_fiber"] for r in rows},
        "fiber_size_grows_with_n": bool(grows),
    }
    if grows:
        return "RED", info
    if frac >= 0.90:
        return "GREEN", info
    if frac >= 0.50 or max(medians) <= 2:
        return "AMBER", info
    return "RED", info
