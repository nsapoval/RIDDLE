"""I3: ``W(tau, s)`` -- the best a shape can do on one triple, and whether that is too good.

``exp-plan-n50-mpl-v0.1`` §2 I3: ``W(tau, s) = max`` over shape-``s`` local parameters of the
``tau``-term of ``log L``.  It is the objective coefficient E4's ILP puts on "assign shape ``s`` to
triple ``tau``", so every claim the ILP makes rests on it.

The tau-term is a multinomial log-likelihood::

    W(tau, s) = max_theta  sum_t  rho(t) * log P(t | s, theta)

**Tree shapes are closed form.**  One free parameter, the internal branch length ``x`` above the
cherry.  With ``a`` the count matching the shape's topology and ``b + c`` the other two::

    d/dx [ a log(1 - (2/3)e^-x) + (b+c) log((1/3)e^-x) ] = 0   =>   e^-x = 3(b+c) / (2(a+b+c))

clipped at 1 (``x >= 0``); past the clip the shape's best is the star, ``(1/3, 1/3, 1/3)``.  This
is the MP-EST profile, and it is what the plan quotes.

**Reticulate shapes are the risk, and it is the plan's largest** (§6.1).  Per triple the data give
two free proportions.  A one-reticulation trinet carries three branch lengths and a gamma.  If
those can reach *any* point of the probability simplex then ``W(tau, reticulate)`` equals the
saturated maximum for **every** triple: the relaxation prefers a reticulate shape everywhere, the
upper bound collapses to the saturated likelihood, and the ILP is vacuous.

So the number that matters is not ``W`` itself but the **saturation gap**,
``W(tau, s) - saturated(tau)``, which is ``<= 0`` always and ``== 0`` exactly when the shape fits
the data perfectly.  :func:`saturation_gap` computes it and ``scripts/i3_profile.py`` sweeps it
over the catalog.  A shape whose gap is zero everywhere contributes nothing to a bound.

What makes this answerable rather than a worry is that a trinet shape's probability vector is
*constrained by its own symmetry*: a cycle that separates one taxon from the other two cannot
distinguish those two, so it produces ``(p, q, q)`` no matter what its parameters are, and cannot
fit an asymmetric ``rho`` however many parameters it has.  Counting parameters therefore
overstates the danger; the honest measurement is numerical, over the shapes that actually occur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opc.mpl.likelihood import MAX_LENGTH, shape_probabilities

__all__ = [
    "ORDINARY",
    "ANOMALOUS",
    "SaturatedFit",
    "saturated_log_likelihood",
    "tree_profile",
    "shape_profile",
    "saturates",
    "saturation_gap",
    "reachable_probabilities",
    "effective_edges",
]

#: rho vectors spanning the simplex, at a fixed locus count.
#:
#: **Split into ordinary and anomalous, and the split decides A36's verdict.**  Under the MSC a
#: species tree always makes its own resolution the modal one, so real triple counts almost always
#: have a strict majority resolution; those are :data:`ORDINARY`.  :data:`ANOMALOUS` puts the mass
#: on the two *mismatch* slots instead.  Scoring a shape against a grid that mixes them lets one
#: unreachable corner rescue a shape that saturates everywhere data actually live -- which is
#: exactly the failure §6.1 is about, and is how I3's first pass returned GREEN in error.
#:
#: Lives here rather than in ``scripts/i3_profile.py`` because E4 classifies shapes against the
#: same grid, and two copies of a threshold drift.
ORDINARY = (
    (700, 150, 150), (500, 250, 250), (400, 300, 300), (340, 330, 330),   # symmetric
    (600, 250, 150), (500, 300, 200), (450, 350, 200), (400, 350, 250),   # asymmetric
    (800, 150,  50), (150, 700, 150), (350, 340, 310), (900, 60, 40),
)
ANOMALOUS = ((200, 400, 400), (250, 400, 350), (100, 450, 450))


def saturated_log_likelihood(rho: np.ndarray) -> float:
    """``sum rho(t) log(rho(t)/m)`` -- the multinomial max, attainable only by a perfect fit."""
    rho = np.asarray(rho, dtype=float)
    total = rho.sum()
    if total <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(rho > 0, rho * np.log(np.where(rho > 0, rho / total, 1.0)), 0.0)
    return float(terms.sum())


def tree_profile(rho: np.ndarray, match: int) -> tuple[float, float]:
    """``(W, x)`` for a tree shape whose resolved topology is slot ``match``.  Closed form."""
    rho = np.asarray(rho, dtype=float)
    a = rho[match]
    bc = rho.sum() - a
    total = rho.sum()
    if total <= 0:
        return 0.0, float("inf")
    u = 3.0 * bc / (2.0 * total)
    u = min(1.0, max(1e-300, u))  # x >= 0; past the clip the best this shape can do is the star
    p_match = 1.0 - 2.0 * u / 3.0
    w = (a * math.log(p_match) if a > 0 else 0.0) + (bc * math.log(u / 3.0) if bc > 0 else 0.0)
    return float(w), float(-math.log(u)) if u > 0 else float("inf")


@dataclass
class SaturatedFit:
    """The result of profiling one shape against one triple's counts."""

    w: float
    saturated: float
    probabilities: np.ndarray
    parameters: np.ndarray

    @property
    def gap(self) -> float:
        """``W - saturated``.  Zero means the shape fits the data exactly."""
        return self.w - self.saturated

    @property
    def relative_gap(self) -> float:
        """The gap as a fraction of the saturated value; scale-free across locus counts."""
        return self.gap / abs(self.saturated) if self.saturated else 0.0


def effective_edges(shape) -> tuple:
    """The trinet edges whose length can actually change ``P``.

    A trinet edge below a *plain* leaf carries exactly one lineage, so the DP never applies a
    coalescent to it and its length is inert.  Excluding it matters for more than speed: §6.1's
    argument is a **degrees-of-freedom count**, and counting inert parameters would overstate the
    very risk this module exists to measure.

    A **hybrid** leaf is not inert: its lineage splits and traverses one of two branches inside a
    cycle, where it can meet another lineage.
    """
    return tuple(
        e for e in sorted(shape.edge_sources)
        if e[0] not in shape.leaf_of or len(shape.parents[e[0]]) == 2
    )


def _pack(shape, edge_keys):
    """Which entries of the parameter vector are lengths and which are gammas."""
    n_edges = len(edge_keys)
    n_gammas = len(shape.gamma_source)
    return n_edges, n_gammas


def reachable_probabilities(shape, edge_keys, theta: np.ndarray) -> np.ndarray:
    """``P`` for a batch of parameter vectors ``theta = [lengths..., gammas...]``."""
    n_edges, n_gammas = _pack(shape, edge_keys)
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    lengths = theta[:, :n_edges]
    gammas = theta[:, n_edges:n_edges + n_gammas]
    return shape_probabilities(shape, edge_keys, lengths, gammas)


def shape_profile(shape, rho: np.ndarray, *, restarts: int = 8, seed: int = 0,
                  maxiter: int = 300) -> SaturatedFit:
    """``W(tau, s)`` by numerical maximisation over the shape's own parameters.

    Multi-start L-BFGS-B, because the surface is not concave in general and a single start from the
    middle of the box would understate what the shape can reach -- which would understate the risk
    this function exists to measure.  Tree shapes are handled here too (and cross-checked against
    :func:`tree_profile` in the tests) rather than special-cased, so the comparison across shapes
    is apples to apples.
    """
    from scipy.optimize import minimize

    edge_keys = effective_edges(shape)
    n_edges, n_gammas = _pack(shape, edge_keys)
    rho = np.asarray(rho, dtype=float)
    bounds = [(0.0, MAX_LENGTH)] * n_edges + [(0.0, 1.0)] * n_gammas
    rng = np.random.default_rng(seed)

    def negative(theta):
        p = reachable_probabilities(shape, edge_keys, theta)[0]
        return -float((rho * np.log(np.maximum(p, 1e-300))).sum())

    best = None
    for attempt in range(max(1, restarts)):
        if attempt == 0:
            x0 = np.r_[np.full(n_edges, 1.0), np.full(n_gammas, 0.5)]
        else:
            x0 = np.r_[rng.lognormal(0.0, 1.0, n_edges), rng.uniform(0.02, 0.98, n_gammas)]
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(negative, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    p = reachable_probabilities(shape, edge_keys, best.x)[0]
    return SaturatedFit(
        w=-float(best.fun), saturated=saturated_log_likelihood(rho),
        probabilities=p, parameters=np.asarray(best.x),
    )


def saturation_gap(shape, rho: np.ndarray, **kwargs) -> float:
    return shape_profile(shape, rho, **kwargs).gap


def saturates(shape, *, grid=ORDINARY, tol: float = 1e-6, **kwargs) -> bool:
    """Does this shape fit **every** ordinary rho exactly?  A36's classification, as a predicate.

    A saturating shape contributes nothing to any bound: whatever the data, ``W(tau, s)`` is the
    multinomial maximum.  E4 uses this to decide, **once per shape rather than once per triple**,
    whether a whole signature's ``W`` column is the closed form -- which is the difference between
    a shape table costing seconds and one costing a day at n = 50.
    """
    for rho in grid:
        if saturation_gap(shape, np.asarray(rho, dtype=float), **kwargs) < -tol:
            return False
    return True
