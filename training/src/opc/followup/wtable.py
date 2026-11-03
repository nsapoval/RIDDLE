"""F1/F2: a profiled ``W`` table that finishes at n = 50, and the DOF penalty.

Why this exists
---------------

``--w-mode profiled`` is the mode ``exp-plan-n50-mpl-v0.1`` §E4 actually specifies, and it has
never been run at n = 50: A43 measured the shape table at **> 2,400 s and unfinished** against
0.3 s for ``saturated``, so every published n = 50 row used ``saturated``.  A46 then measured what
that costs -- under ``saturated`` every triple attains its multinomial maximum, so the ILP's
optimum is ``sum_tau saturated(tau)`` exactly, a closed-form function of ``rho`` with no dependence
on the candidate set, the budget or ``tau``.  The bound is a statistic of the gene trees.

So profiling is not an optional refinement here; it is the only version of the experiment that can
answer H2.  It has to be made to finish.

How
---

:func:`opc.mpl.profile.shape_profile` runs a multi-start L-BFGS-B **per (shape, triple)** --
19,600 optimisations per shape.  But a shape's parameters enter only through the probability vector
they produce, and that vector lives in the 2-simplex however many parameters there are.  So the
reachable set ``R(s) = {P(theta) : theta}`` can be sampled **once per shape**, and then

    W(tau, s) = max_{p in R(s)}  sum_t rho_t log p_t

is a maximum of a *linear* function of ``rho`` over a fixed point set -- one matrix product over
all 19,600 triples at once.  :func:`opc.mpl.likelihood.shape_probabilities` is already batched over
parameter vectors, which is what makes the sampling cheap.

What this costs in rigour, and how it is guarded
------------------------------------------------

A mesh can only *under*-state ``W``, because every mesh point is achievable.  Understating ``W``
is the dangerous direction: the bound is an upper bound on ``sup MPL`` only if ``W(tau, s)`` is at
least the tau-term any network inducing ``s`` can achieve.  So:

* every mesh point is a real parameter vector, never an interpolation -- the maximum returned is
  attained, not estimated;
* :func:`validate` compares against :func:`opc.mpl.profile.shape_profile` on a stratified rho grid
  and returns the worst deficit, which the drivers record as a column rather than assert away;
* the existing ``np.minimum(best, saturated)`` clip and the n <= 8 ``n_upper_bound_violations``
  check both still apply.
"""

from __future__ import annotations

import numpy as np

from opc.mpl import profile
from opc.mpl.likelihood import MAX_LENGTH, shape_probabilities

__all__ = [
    "MESH_POINTS",
    "parameter_mesh",
    "reachable_mesh",
    "mesh_profiler",
    "shape_dof",
    "validate",
]

#: Parameter vectors sampled per shape.  The reachable set is 2-dimensional whatever the parameter
#: count, so this is about covering a surface, not a box -- 8,192 is empirically past the point
#: where :func:`validate`'s worst deficit stops moving.
MESH_POINTS = 8192

#: Lengths are sampled log-uniformly over this range, not uniformly over ``[0, MAX_LENGTH]``.
#: Coalescent-unit branch lengths that matter live near 0 -- the follow-up plan's §0.2 puts the
#: interesting band at 0.02-0.5 -- and a uniform sample over [0, 20] puts 97.5% of its points
#: where ``e^-x`` is already numerically zero and every shape has collapsed to the same limit.
MIN_SAMPLED_LENGTH = 1e-3

#: rho **directions** the mesh is seeded from, as compositions of 8 into 3 parts.  Sampling the
#: parameter box does not converge: 16x the points only halved the worst deficit, because the box
#: is 5-9 dimensional while the set it maps onto is 2-dimensional, so random draws pile up in the
#: interior and miss the frontier.
#:
#: Seeding fixes it for a reason rather than by brute force.  ``W`` maximises a function that is
#: **linear in rho** over the fixed point set ``{log p}``, so the maximum is always attained at a
#: vertex of that set's convex hull, and running the reference optimiser at direction ``rho``
#: returns exactly the hull vertex that direction selects.  45 directions therefore pin 45 hull
#: vertices, and what is left is the angular gap between them -- which is what :func:`validate`
#: measures on a rho grid this one deliberately does not contain.
SEED_STEPS = 8
SEED_GRID = tuple(
    (a / SEED_STEPS, b / SEED_STEPS, (SEED_STEPS - a - b) / SEED_STEPS)
    for a in range(SEED_STEPS + 1)
    for b in range(SEED_STEPS + 1 - a)
)


def shape_dof(shape) -> int:
    """Free parameters of a shape: effective edges plus gammas.

    Thin wrapper over :func:`opc.mpl.profile.effective_edges`, which excludes edges below *plain*
    leaves because they carry one lineage and cannot move ``P``.  A hybrid leaf is not excluded
    (A38).
    """
    return len(profile.effective_edges(shape)) + len(shape.gamma_source)


def parameter_mesh(shape, *, points: int = MESH_POINTS, seed: int = 0) -> np.ndarray:
    """``theta`` samples over the shape's own parameter box, structured anchors first."""
    edge_keys = profile.effective_edges(shape)
    n_edges = len(edge_keys)
    n_gammas = len(shape.gamma_source)
    rng = np.random.default_rng(seed)

    # Anchors: the corners a random sample reaches only by luck, and where the interesting limits
    # are -- zero-length (instant coalescence), MAX_LENGTH (no coalescence), gamma at 0/0.5/1.
    anchors = []
    for lo in (0.0, 0.1, 1.0, MAX_LENGTH):
        for g in (0.0, 0.5, 1.0):
            anchors.append(np.r_[np.full(n_edges, lo), np.full(n_gammas, g)])
    anchor = np.asarray(anchors, dtype=float).reshape(len(anchors), n_edges + n_gammas)

    n = max(points - anchor.shape[0], 1)
    lengths = 10.0 ** rng.uniform(np.log10(MIN_SAMPLED_LENGTH), np.log10(MAX_LENGTH),
                                  size=(n, n_edges))
    gammas = rng.uniform(0.0, 1.0, size=(n, n_gammas))
    return np.vstack([anchor, np.hstack([lengths, gammas])])


def reachable_mesh(shape, *, points: int = MESH_POINTS, seed: int = 0,
                   decimals: int = 6, seed_grid=SEED_GRID,
                   seed_restarts: int = 4) -> np.ndarray:
    """The distinct probability vectors ``R(s)`` reaches on the mesh.  ``float[K, 3]``.

    Two sources, and the second is the one that makes it accurate: a log-uniform sample of the
    parameter box, plus the reference optimiser's own optimum at each of :data:`SEED_GRID`'s rho
    directions.  Every point is a genuinely achievable probability vector either way.

    Deduplication is what keeps the downstream product small: many parameter vectors map to the
    same point once a branch is long enough to have decoupled, and at n = 50 the product is over
    19,600 triples.
    """
    theta = parameter_mesh(shape, points=points, seed=seed)
    edge_keys = profile.effective_edges(shape)
    n_edges = len(edge_keys)
    p = shape_probabilities(shape, edge_keys, theta[:, :n_edges], theta[:, n_edges:])
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    if seed_grid:
        anchored = [profile.shape_profile(shape, np.asarray(r, dtype=float),
                                          restarts=seed_restarts).probabilities
                    for r in seed_grid]
        p = np.vstack([p, np.clip(np.asarray(anchored, dtype=float), 1e-300, 1.0)])
    _, keep = np.unique(np.round(p, decimals), axis=0, return_index=True)
    return p[np.sort(keep)]


def mesh_profiler(shape, obs: np.ndarray, *, restarts: int = 4,
                  points: int = MESH_POINTS, seed: int = 0,
                  chunk: int = 2048, seed_grid=SEED_GRID) -> np.ndarray:
    """``W(tau, s)`` for every triple at once.  Drop-in for ``build_shape_table(profiler=...)``.

    ``restarts`` is accepted and ignored -- the signature matches the reference profiler so the two
    are interchangeable, and there is nothing to restart: the mesh is exhaustive over itself.
    """
    del restarts
    obs = np.asarray(obs, dtype=float)
    logp = np.log(reachable_mesh(shape, points=points, seed=seed,
                                 seed_grid=seed_grid)).T  # (3, K)
    out = np.empty(obs.shape[0], dtype=float)
    for lo in range(0, obs.shape[0], chunk):
        hi = min(lo + chunk, obs.shape[0])
        out[lo:hi] = (obs[lo:hi] @ logp).max(axis=1)
    return out


def validate(shape, rows=None, *, points: int = MESH_POINTS, seed: int = 0,
             restarts: int = 8, seed_grid=SEED_GRID) -> dict:
    """Worst deficit of the mesh against :func:`opc.mpl.profile.shape_profile`.

    Returns ``max_deficit`` (>= 0; how far the mesh falls short of the reference optimiser, in log
    units) and ``max_excess`` (>= 0; how far it *beats* it, which is not an error -- the reference
    is a multi-start local optimiser and the mesh is global over its own sample -- but is worth
    seeing, because a large excess means the reference was under-converging).
    """
    if rows is None:
        rows = list(profile.ORDINARY) + list(profile.ANOMALOUS)
    obs = np.asarray(rows, dtype=float)
    mesh = mesh_profiler(shape, obs, points=points, seed=seed, seed_grid=seed_grid)
    ref = np.array([profile.shape_profile(shape, r, restarts=restarts).w for r in obs])
    d = ref - mesh
    return {
        "max_deficit": float(max(0.0, d.max())),
        "max_excess": float(max(0.0, (-d).max())),
        "n_rows": int(obs.shape[0]),
        "dof": shape_dof(shape),
    }
