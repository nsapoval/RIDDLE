"""Primal search over the same feasible set the ILP optimises (WS4 ablations, and the RL gate).

Two jobs, and the second is why this exists now rather than with WS4.

**The ablations `implementation-plan.md` §WS4 asks for** -- greedy by ``w``, greedy by
``w / drho``, and a swap-based local search -- against which any learned policy has to justify
itself.

**The measurement that decides whether a learned policy is worth building at all.**  A21 shows the
ILP's failure on large components is a *dual* failure: it has an incumbent it cannot certify,
because near-uniform MDC weights pin the edge-formulation LP at ``sum w / 2``.  A policy is a
primal method and cannot certify anything either, so the question "would search help?" is exactly
"is there room between a cheap primal and the ILP's incumbent?".  These functions measure that
room.  If it is nil, the incumbent was never the problem.

**The feasible set, stated in component-local terms.**  ``S`` is feasible with budget ``j`` iff

1. every accepted vertex sees a *single* intersection class on its accepted incident edges -- this
   is (L), which decomposes across components because clusters in different components do not
   overlap at all;
2. the accepted subgraph has no triangle -- (N3O); and
3. the accepted edges carry at most ``j`` distinct classes -- the reticulation budget.

Which is precisely the ILP's row set: (1) is ``sum_h z[A,h] <= x[A]`` with the assignment rows,
(2) the clique-budget family, (3) ``sum_h y[h] <= j``.  :func:`check` re-derives all three from
scratch with its own triangle count, so it is an *independent* validator -- used on the ILP's own
output as well as on these heuristics, per the house rule that nothing is checked against itself.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from opc.solve.overlap import Component

__all__ = ["HeuristicResult", "Feasibility", "check", "greedy", "local_search", "best_primal"]

#: Cap on local-search sweeps.  Each accepted improvement strictly raises the objective, so the
#: loop terminates on its own; the cap bounds wall-clock on the components where every sweep is
#: O(E) and improvements keep arriving one at a time.
MAX_SWEEPS = 20


@dataclass
class HeuristicResult:
    """One primal solution for a ``(component, budget)`` pair."""

    method: str
    budget: int
    objective: float
    accepted: np.ndarray  # local vertex indices
    n_hybrids: int
    seconds: float
    sweeps: int = 0

    def as_row(self, prefix: str = "") -> dict:
        return {
            f"{prefix}objective": self.objective,
            f"{prefix}n_accepted": int(self.accepted.size),
            f"{prefix}n_hybrids": self.n_hybrids,
            f"{prefix}seconds": self.seconds,
            f"{prefix}sweeps": self.sweeps,
        }


@dataclass
class Feasibility:
    """Why an accepted set is or is not feasible.  ``ok`` is the conjunction of the three rules."""

    ok: bool
    n_hybrids: int
    l_violating_vertices: int
    n_triangles: int
    over_budget: bool


def _csr(comp: Component) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric adjacency as ``(indptr, neighbour, class)``, neighbours grouped by vertex."""
    v = comp.n_vertices
    src = np.concatenate([comp.ii, comp.jj]).astype(np.int64)
    dst = np.concatenate([comp.jj, comp.ii]).astype(np.int64)
    cls = np.concatenate([comp.edge_class, comp.edge_class]).astype(np.int64)
    order = np.argsort(src, kind="stable")
    src, dst, cls = src[order], dst[order], cls[order]
    indptr = np.zeros(v + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, dst, cls


def check(comp: Component, accepted, budget: int) -> Feasibility:
    """(L), (N3O) and the budget, re-derived from the accepted set alone.

    Independent of both the ILP rows and :class:`_State`: the class-per-vertex test is a groupby
    over the induced edge list, and triangles are counted as ``nnz((A @ A) . A) / 6`` on the
    induced adjacency rather than by the enumerator in :mod:`opc.solve.overlap`.
    """
    accepted = np.asarray(accepted, dtype=np.int64)
    mark = np.zeros(comp.n_vertices, dtype=bool)
    mark[accepted] = True
    keep = mark[comp.ii] & mark[comp.jj]
    ii, jj, cls = comp.ii[keep], comp.jj[keep], comp.edge_class[keep]

    # (L): a vertex may see only one intersection value across its accepted incident edges.
    bad = 0
    if ii.size:
        ends = np.concatenate([ii, jj]).astype(np.int64)
        vals = np.concatenate([cls, cls]).astype(np.int64)
        order = np.lexsort((vals, ends))
        ends, vals = ends[order], vals[order]
        change = np.r_[True, (ends[1:] != ends[:-1]) | (vals[1:] != vals[:-1])]
        uniq_end = ends[change]
        bad = int(np.count_nonzero(np.bincount(uniq_end, minlength=comp.n_vertices) > 1))

    n_tri = 0
    if ii.size:
        n = comp.n_vertices
        a = sp.coo_matrix((np.ones(ii.size, np.int8), (ii, jj)), shape=(n, n)).tocsr()
        a = a + a.T
        n_tri = int((a @ a).multiply(a).sum() // 6)

    n_hyb = int(np.unique(cls).size)
    over = n_hyb > budget
    return Feasibility(
        ok=bool(bad == 0 and n_tri == 0 and not over),
        n_hybrids=n_hyb, l_violating_vertices=bad, n_triangles=n_tri, over_budget=bool(over),
    )


class _State:
    """Incremental accept with an exact guard.  (L) is hereditary, so no lookahead is needed.

    This is the same masking WS4 would need for action selection; building it here means a policy
    would inherit a checked implementation rather than write a second one.
    """

    def __init__(self, comp: Component, budget: int):
        self.comp = comp
        self.budget = int(budget)
        self.indptr, self.nbr, self.cls = _csr(comp)
        self.in_s = np.zeros(comp.n_vertices, dtype=bool)
        self.commit = np.full(comp.n_vertices, -1, dtype=np.int64)
        self.used: set[int] = set()
        self._mark = np.zeros(comp.n_vertices, dtype=bool)

    def _slice(self, v: int):
        lo, hi = self.indptr[v], self.indptr[v + 1]
        return self.nbr[lo:hi], self.cls[lo:hi]

    def can_accept(self, v: int) -> tuple[bool, int]:
        """``(feasible, class)``; the class is ``-1`` when the vertex touches nothing accepted."""
        if self.in_s[v]:
            return False, -1
        nb, nc = self._slice(v)
        sel = self.in_s[nb]
        if not sel.any():
            return True, -1  # an isolated accept commits nothing and costs no budget
        nb, nc = nb[sel], nc[sel]
        klass = int(nc[0])
        if (nc != klass).any():
            return False, -1  # (L) at v itself
        committed = self.commit[nb]
        if ((committed != -1) & (committed != klass)).any():
            return False, -1  # (L) at an already-committed neighbour
        # (N3O): two accepted neighbours of v that are adjacent to each other close a triangle.
        self._mark[nb] = True
        try:
            for u in nb:
                un, _ = self._slice(int(u))
                if self._mark[un].any():
                    return False, -1
        finally:
            self._mark[nb] = False
        if klass not in self.used and len(self.used) >= self.budget:
            return False, -1
        return True, klass

    def accept(self, v: int, klass: int) -> None:
        self.in_s[v] = True
        if klass < 0:
            return
        nb, nc = self._slice(v)
        sel = self.in_s[nb] & (nc == klass)
        self.commit[nb[sel]] = klass
        self.commit[v] = klass
        self.used.add(int(klass))

    def drop(self, vs) -> None:
        """Remove vertices and rebuild the commitments they were part of."""
        self.in_s[np.asarray(vs, dtype=np.int64)] = False
        self.commit[:] = -1
        self.used.clear()
        acc = np.flatnonzero(self.in_s)
        if acc.size:
            mark = np.zeros(self.comp.n_vertices, dtype=bool)
            mark[acc] = True
            keep = mark[self.comp.ii] & mark[self.comp.jj]
            for a, b, c in zip(self.comp.ii[keep], self.comp.jj[keep], self.comp.edge_class[keep]):
                self.commit[int(a)] = self.commit[int(b)] = int(c)
                self.used.add(int(c))

    @property
    def accepted(self) -> np.ndarray:
        return np.flatnonzero(self.in_s).astype(np.int32)


def greedy(
    comp: Component, weights: np.ndarray, budget: int, *, key: str = "w"
) -> HeuristicResult:
    """Accept in weight order (``key="w"``) or bang-per-reticulation order (``"w-per-retic"``).

    ``drho`` is 0 or 1 -- a vertex costs a reticulation only when it commits to a class nobody has
    used yet -- so ``w / (1 + drho)`` has two tiers and the dynamic greedy needs no full rescan:
    a vertex found to cost budget is deferred at half its weight, and deferral is re-examined when
    it comes back to the top.  Its cost can only *fall* (another vertex may open the class first),
    which is why the deferred key is a valid lower bound on its current score.
    """
    t0 = time.perf_counter()
    w = np.asarray(weights, dtype=np.float64)
    st = _State(comp, budget)

    if key == "w":
        for v in np.argsort(-w, kind="stable"):
            ok, klass = st.can_accept(int(v))
            if ok:
                st.accept(int(v), klass)
    elif key == "w-per-retic":
        fresh = [(-float(w[v]), int(v)) for v in range(comp.n_vertices)]
        heapq.heapify(fresh)
        deferred: list[tuple[float, int]] = []
        while fresh or deferred:
            take_fresh = bool(fresh) and (not deferred or fresh[0][0] <= deferred[0][0])
            _, v = heapq.heappop(fresh) if take_fresh else heapq.heappop(deferred)
            ok, klass = st.can_accept(v)
            if not ok:
                continue
            costs = klass >= 0 and klass not in st.used
            if costs and take_fresh:
                heapq.heappush(deferred, (-float(w[v]) / 2.0, v))
                continue
            st.accept(v, klass)
    else:
        raise ValueError(f"unknown greedy key {key!r}")

    acc = st.accepted
    return HeuristicResult(
        method=f"greedy:{key}", budget=budget, objective=float(w[acc].sum()),
        accepted=acc, n_hybrids=len(st.used), seconds=time.perf_counter() - t0,
    )


#: How many one-tight candidates around an evicted vertex the two-improvement pass will pair up.
#: Sorted by weight, so the truncation only ever loses a pair that was already outscored.
TWO_IMPROVE_FANOUT = 16


def _insert_with_eviction(comp, w, budget, cur, indptr, nbr, cls, deadline=None) -> bool:
    """Admit ``v`` by evicting whatever blocks it, when ``w(v)`` exceeds what it displaces.

    A blocker is a neighbour on the *wrong* intersection class, or one that would close a
    triangle -- not merely any neighbour, which is what makes this wider than the MWIS move.
    """
    for i, v in enumerate(np.argsort(-w, kind="stable")):
        if deadline is not None and not i % 64 and time.perf_counter() > deadline:
            return False
        v = int(v)
        if cur[v]:
            continue
        lo, hi = indptr[v], indptr[v + 1]
        nb, nc = nbr[lo:hi], cls[lo:hi]
        sel = cur[nb]
        nb_in, nc_in = nb[sel], nc[sel]
        classes = np.unique(nc_in) if nb_in.size else np.array([-1], dtype=np.int64)
        for klass in classes:
            blockers = {int(u) for u in nb_in[nc_in != klass]}
            keepers = [int(u) for u in nb_in[nc_in == klass]]
            for a in keepers:
                if a in blockers:
                    continue
                an = nbr[indptr[a]:indptr[a + 1]]
                for b in keepers:
                    if b > a and b not in blockers and (an == b).any():
                        blockers.add(a if w[a] <= w[b] else b)
            if float(w[v]) - float(sum(w[u] for u in blockers)) <= 1e-9:
                continue
            trial = cur.copy()
            trial[list(blockers)] = False
            trial[v] = True
            if check(comp, np.flatnonzero(trial), budget).ok:
                cur[:] = trial
                return True
    return False


def _two_improvement(comp, w, budget, cur, indptr, nbr, cls, deadline=None) -> bool:
    """Evict one accepted vertex to admit two -- the move a weight-ordered greedy cannot make.

    The classical MWIS 2-improvement: around an accepted ``u``, the *one-tight* vertices are those
    whose only accepted neighbour is ``u``, so any non-adjacent pair of them becomes admissible
    the moment ``u`` leaves.  Without this a greedy that takes a heavy vertex over two lighter
    neighbours is a fixed point, which is precisely the case where a learned policy would have
    something to beat.
    """
    order = np.flatnonzero(cur)[np.argsort(w[np.flatnonzero(cur)], kind="stable")]
    for i, u in enumerate(order):
        if deadline is not None and not i % 32 and time.perf_counter() > deadline:
            return False
        u = int(u)
        cand = []
        for v in nbr[indptr[u]:indptr[u + 1]]:
            v = int(v)
            if cur[v]:
                continue
            vn = nbr[indptr[v]:indptr[v + 1]]
            acc_nb = vn[cur[vn]]
            if acc_nb.size == 1 and int(acc_nb[0]) == u:
                cand.append(v)
        if len(cand) < 2:
            continue
        cand.sort(key=lambda x: -w[x])
        cand = cand[:TWO_IMPROVE_FANOUT]
        for i, v1 in enumerate(cand[:-1]):
            if float(w[v1]) + float(w[cand[i + 1]]) <= float(w[u]) + 1e-9:
                break  # weights descend, so no later pair can pay either
            v1n = nbr[indptr[v1]:indptr[v1 + 1]]
            for v2 in cand[i + 1:]:
                if float(w[v1]) + float(w[v2]) <= float(w[u]) + 1e-9:
                    break
                if (v1n == v2).any():
                    continue
                trial = cur.copy()
                trial[u] = False
                trial[v1] = trial[v2] = True
                if check(comp, np.flatnonzero(trial), budget).ok:
                    cur[:] = trial
                    return True
    return False


def local_search(
    comp: Component,
    weights: np.ndarray,
    budget: int,
    accepted: np.ndarray,
    *,
    max_sweeps: int = MAX_SWEEPS,
    time_limit: float | None = None,
) -> HeuristicResult:
    """Sweep the two swap neighbourhoods to a local optimum, or until ``time_limit``.

    Every proposed move is validated by :func:`check` before it is kept, so an error in either
    blocker calculation costs a rejected move rather than an infeasible answer -- the heuristic
    is allowed to be clever, but not to be trusted.

    **The time limit is not optional in practice.** A sweep is ``O(V . deg^2)`` and validates each
    candidate move in ``O(E)``, so on a 1,500-vertex boundary component the polish outruns the
    ILP it is supposed to shadow -- measured at *hours* per cell against the ILP's 300 s cap,
    which is how the boundary sweep was first lost.  Whoever gives the ILP a budget must give the
    primal one too.
    """
    t0 = time.perf_counter()
    w = np.asarray(weights, dtype=np.float64)
    indptr, nbr, cls = _csr(comp)
    cur = np.zeros(comp.n_vertices, dtype=bool)
    cur[np.asarray(accepted, dtype=np.int64)] = True

    deadline = None if time_limit is None else t0 + float(time_limit)
    sweeps = 0
    for _ in range(max_sweeps):
        if deadline is not None and time.perf_counter() > deadline:
            break
        sweeps += 1
        moved = _insert_with_eviction(comp, w, budget, cur, indptr, nbr, cls, deadline)
        moved |= _two_improvement(comp, w, budget, cur, indptr, nbr, cls, deadline)
        if not moved:
            break

    acc = np.flatnonzero(cur).astype(np.int32)
    feas = check(comp, acc, budget)
    return HeuristicResult(
        method="local-search", budget=budget, objective=float(w[acc].sum()),
        accepted=acc, n_hybrids=feas.n_hybrids, seconds=time.perf_counter() - t0, sweeps=sweeps,
    )


def best_primal(
    comp: Component, weights: np.ndarray, budget: int, *, time_limit: float | None = None, **kwargs
) -> HeuristicResult:
    """Both greedies, each polished by local search; the winner, with its provenance kept.

    Reported as one number because the RL gate asks a single question -- how much room is left
    above *any* cheap primal -- but ``method`` records which construction won, since a policy
    that only beats the weaker one has beaten nothing.

    ``time_limit`` is the budget for the *whole* call and is split across the two polishes, so
    the primal's cost stays bounded relative to the solve it shadows.
    """
    out: list[HeuristicResult] = []
    each = None if time_limit is None else max(0.05, float(time_limit) / 2.0)
    for key in ("w", "w-per-retic"):
        g = greedy(comp, weights, budget, key=key)
        out.append(g)
        ls = local_search(comp, weights, budget, g.accepted, time_limit=each, **kwargs)
        ls.method = f"local-search<{key}"
        ls.seconds += g.seconds
        out.append(ls)
    return max(out, key=lambda r: r.objective)
