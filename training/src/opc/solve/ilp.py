"""MW-GTCS(j) on one overlap component (WS1-3).

Maximise ``sum_A w(A) x[A]`` over accepted sets that satisfy (L) and (N3O) with at most ``j``
reticulations.  Solved with ``scipy.optimize.milp``, which is HiGHS.

The model follows the handoff spec with **one row deleted and one row strengthened**.

*Deleted:* ``x[h] >= y[h]``.  The spec requires a block's hybrid cluster to be *in* the accepted
set.  It is not part of (L) -- ``{ {0,1,2}, {2,3,4} }`` is feasible with ``h = {2}`` absent, and
88% of blocks in feasible systems have ``h`` outside ``S`` -- and worse, ``h`` is nested inside
both clusters it separates, so it overlaps neither and is never a vertex of the component being
solved.  The row would constrain a variable the model does not contain.  ``h`` is recovered
afterwards by the closure ``I(S*)``, which is what the theory says it is for.

*Strengthened:* the (N3O) row is ``x[A] + x[B] + x[C] <= 1 + y[h]`` rather than ``<= 2``.  Valid
by the same argument that makes the assignment rows work -- two accepted members of a triangle
already force ``y[h] = 1`` -- and it is the difference between a 90% and a 74% gap at 60 s on the
grid's largest envelope component.  ``<= 2`` cannot cut the all-halves LP point; this can.

At ``j = 0`` the model collapses to maximum-weight independent set and is built that way: with no
reticulation budget every overlapping pair is forbidden outright, and the ``z``/``y`` variables
would only obstruct HiGHS's clique-table detection.  That case is Than-Nakhleh's, and B1's T1
requires it to reproduce their answer exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from opc.solve.overlap import (
    Component,
    monochromatic_cliques,
    monochromatic_triangles,
)

__all__ = ["Solution", "solve_component", "frontier"]

#: Components at or below this many vertices get every monochromatic triangle up front.  Above
#: it, triangles are added only when an integral solution actually violates one -- HiGHS exposes
#: no lazy-constraint callback, so separation is an outer loop rather than a callback.
EAGER_TRIANGLE_MAX = 500

#: Safety valve on that loop.  It terminates on its own (each round adds a row from a finite
#: family), but a capped solve can cycle through near-ties, and an unbounded loop in a grid run
#: is worse than a recorded non-convergence.
MAX_CUT_ROUNDS = 50

#: How far from ``{0, 1}`` a returned variable may sit before the answer stops being a solution.
#: ``milp`` returns ``res.x`` even when the time limit expires **before any integer incumbent is
#: found**, and that vector is fractional.  Rounding it at 0.5 produces a set that is not
#: feasible and an objective that is not a bound -- measured on the boundary sweep, the greedy
#: beat such a "solution" by 33% on the median affected component.  Small components never showed
#: it because they always find an incumbent; 500+ vertex ones do not.
INTEGRALITY_TOL = 1e-6


@dataclass
class Solution:
    """One ``(component, j)`` point of the frontier."""

    budget: int
    objective: float
    accepted: np.ndarray  # local vertex indices
    hybrids: np.ndarray  # indices into ``component.hcand``
    status: int
    mip_gap: float
    dual_bound: float
    seconds: float
    n_rows: int
    n_binaries: int
    n_triangles: int
    cut_rounds: int
    converged: bool
    optimal: bool = field(default=False)
    has_incumbent: bool = field(default=True)

    def as_row(self) -> dict:
        return {
            "budget": self.budget, "objective": self.objective,
            "n_accepted": int(self.accepted.size), "n_hybrids": int(self.hybrids.size),
            "status": self.status, "mip_gap": self.mip_gap, "dual_bound": self.dual_bound,
            "seconds": self.seconds, "n_rows": self.n_rows, "n_binaries": self.n_binaries,
            "n_triangles": self.n_triangles, "cut_rounds": self.cut_rounds,
            "converged": self.converged, "optimal": self.optimal,
            "has_incumbent": self.has_incumbent,
        }


def _z_lookup(comp: Component):
    """``(vertex, class) -> z column``.  ``assignment_pairs`` returns them lexicographically."""
    key = comp.z_vertex.astype(np.int64) * (comp.n_hcand + 1) + comp.z_class.astype(np.int64)
    order = np.argsort(key, kind="stable")
    return key[order], order


def _rows_mwis(comp: Component, weights: np.ndarray):
    """``j = 0``: maximum-weight independent set, one row per edge.

    The clique reformulation ``sum_{A in Q} x[A] <= 1`` over an edge clique cover
    (:func:`~opc.solve.overlap.edge_clique_cover`) is the textbook strengthening and it does cut
    the row count by 3.7x on the grid's largest envelope component -- 4,910 to 1,325.  Measured,
    it makes the **bound worse** (gap 0.134 -> 0.156 at 60 s): HiGHS derives its own clique table
    from the edge rows and does it better than a greedy cover.  So the edge formulation stays, and
    the cover survives as a utility rather than as the model.
    """
    v = comp.n_vertices
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e in range(comp.n_edges):
        rows += [e, e]
        cols += [int(comp.ii[e]), int(comp.jj[e])]
        data += [1.0, 1.0]
    A = sp.coo_matrix((data, (rows, cols)), shape=(comp.n_edges, v)).tocsr()
    return A, np.full(comp.n_edges, 1.0), v, np.zeros(0, np.int64)


def _rows_full(comp: Component, triangles: np.ndarray, cliques=None):
    """The assignment formulation.  Columns are ``[x (V) | z (Z) | y (D)]``."""
    v, z, d = comp.n_vertices, comp.n_z, comp.n_hcand
    n_cols = v + z + d
    key, _ = _z_lookup(comp)

    def zcol(vertex: int, klass: int) -> int:
        k = vertex * (comp.n_hcand + 1) + klass
        i = int(np.searchsorted(key, k))
        return v + i

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    ub: list[float] = []

    def add_row(entries, bound):
        r = len(ub)
        for col, coef in entries:
            rows.append(r)
            cols.append(col)
            data.append(coef)
        ub.append(bound)

    # sum_h z[A,h] <= x[A] -- an accepted cluster belongs to at most one block, a rejected one
    # to none.
    for a in range(v):
        sel = np.flatnonzero(comp.z_vertex == a)
        if sel.size:
            add_row([(v + int(s), 1.0) for s in sel] + [(a, -1.0)], 0.0)

    # x[A] + x[B] <= 1 + z[A,h], and the same for B: accepting an overlapping pair commits both
    # endpoints to the block their intersection defines.
    for e in range(comp.n_edges):
        a, b, c = int(comp.ii[e]), int(comp.jj[e]), int(comp.edge_class[e])
        for endpoint in (a, b):
            add_row([(a, 1.0), (b, 1.0), (zcol(endpoint, c), -1.0)], 1.0)

    # y[h] >= z[A,h] -- a used block costs a reticulation.
    for s in range(z):
        add_row([(v + s, 1.0), (v + z + int(comp.z_class[s]), -1.0)], 0.0)

    # sum_{A in Q} x[A] <= 1 + y[h]: the clique-budget family.  For |Q| = 3 this is the
    # strengthened (N3O) row; for larger Q one row replaces C(|Q|,3) of them and is tighter.
    if cliques is not None:
        for q, klass in cliques:
            add_row([(int(a), 1.0) for a in q] + [(v + z + int(klass), -1.0)], 1.0)
    else:
        for t in triangles:
            a, b, c = (int(x) for x in t)
            klass = _triangle_class(comp, a, b)
            add_row([(a, 1.0), (b, 1.0), (c, 1.0), (v + z + klass, -1.0)], 1.0)

    A = sp.coo_matrix((data, (rows, cols)), shape=(len(ub), n_cols)).tocsr()
    return A, np.asarray(ub, dtype=float), n_cols, np.arange(v + z, n_cols)


def _triangle_class(comp: Component, a: int, b: int) -> int:
    """The shared intersection value of a monochromatic triangle, via one of its edges."""
    hit = np.flatnonzero(((comp.ii == a) & (comp.jj == b)) | ((comp.ii == b) & (comp.jj == a)))
    return int(comp.edge_class[hit[0]])


def _violated_triangles(comp: Component, accepted: np.ndarray) -> np.ndarray:
    """Monochromatic triangles inside the accepted set -- the exact separation problem.

    ``x*`` is integral, so separation is triangle enumeration on the accepted subgraph, which is
    far smaller than the component.  Exactness is what makes the outer loop a valid decomposition
    rather than a heuristic.
    """
    if accepted.size < 3:
        return np.zeros((0, 3), dtype=np.int32)
    keep = np.isin(comp.ii, accepted) & np.isin(comp.jj, accepted)
    if not keep.any():
        return np.zeros((0, 3), dtype=np.int32)
    sub = Component(
        vertices=comp.vertices, ii=comp.ii[keep], jj=comp.jj[keep],
        edge_class=comp.edge_class[keep], hcand=comp.hcand,
        z_vertex=comp.z_vertex, z_class=comp.z_class,
    )
    return monochromatic_triangles(sub)


def solve_component(
    comp: Component,
    weights: np.ndarray,
    budget: int,
    *,
    time_limit: float = 600.0,
    eager_triangle_max: int = EAGER_TRIANGLE_MAX,
    max_cut_rounds: int = MAX_CUT_ROUNDS,
) -> Solution:
    """Solve MW-GTCS(``budget``) on one component.

    ``weights`` is indexed by *local* vertex, i.e. ``w[comp.vertices]``.
    """
    t0 = time.perf_counter()
    v = comp.n_vertices
    eager = comp.n_vertices <= eager_triangle_max
    triangles = monochromatic_triangles(comp) if eager else np.zeros((0, 3), np.int32)
    # Eagerly we can afford the stronger family; the cutting-plane path accumulates individual
    # violated triangles and has no clique to grow them into yet.
    cliques = monochromatic_cliques(comp) if eager else None

    rounds = 0
    while True:
        rounds += 1
        if budget <= 0:
            A, ub, n_cols, y_cols = _rows_mwis(comp, weights)
        else:
            A, ub, n_cols, y_cols = _rows_full(comp, triangles, cliques)
            budget_row = sp.coo_matrix(
                (np.ones(y_cols.size), (np.zeros(y_cols.size, int), y_cols)),
                shape=(1, n_cols),
            ).tocsr()
            A = sp.vstack([A, budget_row]).tocsr()
            ub = np.r_[ub, float(budget)]

        c = np.zeros(n_cols)
        c[:v] = -np.asarray(weights, dtype=float)  # milp minimises
        res = milp(
            c=c,
            constraints=LinearConstraint(A, -np.inf, ub) if A.shape[0] else None,
            integrality=np.ones(n_cols),
            bounds=Bounds(0, 1),
            options={"time_limit": max(1.0, time_limit - (time.perf_counter() - t0))},
        )
        integral = res.x is not None and bool(
            np.all(np.minimum(np.abs(res.x), np.abs(1.0 - np.asarray(res.x))) <= INTEGRALITY_TOL)
        )
        if not integral:
            return Solution(
                budget=budget, objective=float("nan"), accepted=np.zeros(0, np.int32),
                hybrids=np.zeros(0, np.int64), status=int(res.status),
                mip_gap=float(getattr(res, "mip_gap", float("nan"))),
                dual_bound=float(getattr(res, "mip_dual_bound", float("nan"))),
                seconds=time.perf_counter() - t0, n_rows=int(A.shape[0]), n_binaries=n_cols,
                n_triangles=int(triangles.shape[0]), cut_rounds=rounds, converged=False,
                has_incumbent=False,
            )

        x = np.asarray(res.x)[:v]
        accepted = np.flatnonzero(x > 0.5).astype(np.int32)
        if eager or budget <= 0:
            break
        fresh = _violated_triangles(comp, accepted)
        if fresh.shape[0] == 0:
            break
        triangles = np.unique(np.vstack([triangles, fresh]), axis=0)
        if rounds >= max_cut_rounds or time.perf_counter() - t0 >= time_limit:
            break

    hybrids = np.zeros(0, np.int64)
    if budget > 0:
        y = np.asarray(res.x)[comp.n_vertices + comp.n_z:]
        hybrids = np.flatnonzero(y > 0.5).astype(np.int64)

    gap = float(getattr(res, "mip_gap", float("nan")))
    converged = bool(_violated_triangles(comp, accepted).shape[0] == 0)
    return Solution(
        budget=budget,
        objective=float(np.asarray(weights, dtype=float)[accepted].sum()),
        accepted=accepted, hybrids=hybrids, status=int(res.status), mip_gap=gap,
        dual_bound=float(getattr(res, "mip_dual_bound", float("nan"))),
        seconds=time.perf_counter() - t0, n_rows=int(A.shape[0]), n_binaries=n_cols,
        n_triangles=int(triangles.shape[0]), cut_rounds=rounds, converged=converged,
        optimal=bool(res.status == 0 and gap == 0.0 and converged),
    )


def frontier(comp: Component, weights: np.ndarray, j_max: int, **kwargs) -> list[Solution]:
    """``f_K(j)`` for ``j = 0 .. j_max``.

    ``j_max`` is clipped to ``floor(V / 2)``: a reticulation needs two distinct clusters to
    nominate its hybrid, so no component can use more budget than that, and leaving the extra
    solves in would only buy identical answers.  T6 asserts the result is non-decreasing -- which
    is trivially true of the optima and is therefore a live check on time-cap contamination.
    """
    top = min(int(j_max), comp.n_vertices // 2)
    return [solve_component(comp, weights, j, **kwargs) for j in range(top + 1)]
