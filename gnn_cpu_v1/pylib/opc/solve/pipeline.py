"""One ground set, end to end: decompose, solve every component, recombine, close.

The stage that `implementation-plan.md` §WS1 leaves implicit between modules 3 and 5 and that
``scripts/run_solver.py`` needs per cell.  It lives in the library rather than in the script so
the same path is exercised by the tests -- ``tests/test_truth.py`` solves through it -- and so
B1's T4a is asserted in one place instead of at every call site.

**The frontier is probed, not swept.**  ``f_K(j)`` is wanted for ``j = 0..jmax``, but on most
components it is constant: the component has no affordable block, or one block that pays for
itself immediately.  Solving ``j = 0`` and the *unbudgeted* problem bounds the whole frontier --
``f_K`` is non-decreasing and both ends agree -- so a flat component costs two solves instead of
``jmax + 1``.  Note what is **not** valid: stopping when ``f_K(j) == f_K(j-1)``.  A plateau at
``j`` says nothing about ``j+1``, because reaching a higher-budget optimum can require dropping a
block wholesale rather than extending the current one.  Only the two-ended sandwich is sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from opc.solve import budget_dp, heuristics, ilp, overlap
from opc.solve.ilp import Solution

__all__ = ["ComponentReport", "InstanceSolution", "component_frontier", "solve_ground_set"]


@dataclass
class ComponentReport:
    """Everything measured about one component: the ILP side and the primal side."""

    index: int
    vertices: np.ndarray  # ground-set rows, in local order
    n_vertices: int
    n_edges: int
    n_hcand: int
    n_z: int
    cv_weight: float
    frontier: list[Solution]
    flat: bool
    probe_budget: int
    probe: Solution | None = None
    from_primal: list[bool] = field(default_factory=list)
    primal: list[heuristics.HeuristicResult] = field(default_factory=list)
    feasible_ok: bool = True

    @property
    def headroom(self) -> float:
        """``(ILP incumbent - best primal) / ILP incumbent`` at the budget where it is largest.

        The RL gate's statistic.  Negative would mean a heuristic beat the solver's incumbent,
        which is possible only where the solve was capped -- and is worth seeing rather than
        clamping away.

        Points where the ILP found no incumbent are **excluded**: there the frontier holds the
        primal itself, so the difference is zero by construction and averaging it in would report
        "the greedy matched the solver" about a solve that produced nothing to match.
        """
        gaps = []
        subs = self.from_primal or [False] * len(self.frontier)
        for sol, prim, sub in zip(self.frontier, self.primal, subs):
            if sub:
                continue
            if sol.objective and np.isfinite(sol.objective) and sol.objective > 0:
                gaps.append((sol.objective - prim.objective) / sol.objective)
        return max(gaps) if gaps else float("nan")


@dataclass
class InstanceSolution:
    """The recombined answer at every budget ``0..k``, plus the per-component detail."""

    selected: np.ndarray  # rows of the ground set, at budget k
    objective: float
    kernel: np.ndarray
    kernel_weight: float
    allocation: list[int]
    components: list[ComponentReport]
    global_solution: budget_dp.GlobalSolution
    j_used: int

    def rows_at(self, k: int) -> np.ndarray:
        """Accepted ground-set rows at budget ``k`` (the kernel plus each component's choice)."""
        _, alloc = self.global_solution.best(k)
        chosen = list(self.kernel)
        for rep, j in zip(self.components, alloc):
            sol = rep.frontier[min(j, len(rep.frontier) - 1)]
            chosen += [int(rep.vertices[a]) for a in sol.accepted]
        return np.array(sorted(chosen), dtype=np.int64)


def component_frontier(
    comp: overlap.Component,
    weights: np.ndarray,
    jmax: int,
    *,
    time_limit: float = 60.0,
    probe: bool = True,
) -> tuple[list[Solution], bool, int, Solution | None]:
    """``(f_K(0..top), flat, probe_budget, probe)`` with the two-ended sandwich described above.

    The probe is returned rather than discarded even when it does not collapse the frontier: it
    is ``f_K(infinity)``, so it is what says whether ``jmax`` truncated anything, and its cost is
    real and belongs in the record.  Measured on 40 envelope cells it is 38% of the solve time
    and collapsed nothing, so ``probe=False`` exists for runs -- the boundary sweep -- that want
    closure-versus-size at fixed budgets rather than the top of the frontier.
    """
    top = min(int(jmax), comp.n_vertices // 2)
    s0 = ilp.solve_component(comp, weights, 0, time_limit=time_limit)
    if top <= 0:
        return [s0], True, 0, None
    if not probe:
        return (
            [s0] + [ilp.solve_component(comp, weights, j, time_limit=time_limit)
                    for j in range(1, top + 1)],
            False, 0, None,
        )

    probe_budget = min(comp.n_hcand, comp.n_vertices // 2)
    if probe_budget <= 0:
        return [s0] * (top + 1), True, 0, None

    top_sol = ilp.solve_component(comp, weights, probe_budget, time_limit=time_limit)
    if s0.optimal and top_sol.optimal and abs(top_sol.objective - s0.objective) <= 1e-9:
        return [s0] * (top + 1), True, probe_budget, top_sol

    out = [s0]
    for j in range(1, top + 1):
        out.append(
            top_sol if j == probe_budget
            else ilp.solve_component(comp, weights, j, time_limit=time_limit)
        )
    return out, False, probe_budget, top_sol


def primal_frontier(
    comp: overlap.Component,
    weights: np.ndarray,
    jmax: int,
    *,
    time_limit: float | None = None,
) -> tuple[list[Solution], list[heuristics.HeuristicResult]]:
    """``f_K(0..top)`` from the primal alone, with no ILP at all.

    Why this exists.  At ``tau*`` the ILP was affordable; at ``tau = 0`` it is not -- the ground
    set is 10-30x larger and collapses to one component of ~1,400 vertices at ~30% edge density
    (A33), where the edge-formulation LP is half-integral and pinned at ``sum w / 2`` (A21).  No
    MIP solver repairs a relaxation that weak.

    What makes the substitution defensible rather than a retreat is A27's measurement: over 5,136
    solves of components with ``V > 60``, the median gap between the ILP's incumbent and
    greedy-plus-swap is **0.0000**, the mean headroom *shrinks* with instance size, and the primal
    costs 0.09% of the ILP's wall-clock.  So on exactly the components where the ILP now fails, the
    primal was already returning the same answer.

    **The frontier is forced non-decreasing.**  A greedy at budget ``j+1`` can score below its own
    result at ``j``, which the ILP's optimum never does; a budget-``j`` selection is feasible at
    ``j+1``, so carrying it forward is valid and keeps :mod:`opc.solve.budget_dp`'s monotonicity
    assumption (T6) true.  ``method`` records where each point came from.
    """
    top = min(int(jmax), comp.n_vertices // 2)
    out: list[Solution] = []
    primal: list[heuristics.HeuristicResult] = []
    for j in range(max(0, top) + 1):
        p = heuristics.best_primal(comp, weights, j, time_limit=time_limit)
        if out and p.objective < out[-1].objective:
            p = replace(p, objective=out[-1].objective, accepted=out[-1].accepted,
                        method=f"carried<{out[-1].budget}")
        primal.append(p)
        out.append(Solution(
            budget=j, objective=float(p.objective), accepted=p.accepted,
            hybrids=np.zeros(0, np.int64), status=0, mip_gap=float("nan"),
            dual_bound=float("nan"), seconds=float(p.seconds), n_rows=0,
            n_binaries=comp.n_vertices, n_triangles=0, cut_rounds=0,
            converged=True, has_incumbent=True, optimal=False,
        ))
    return out, primal


def solve_ground_set(
    words: np.ndarray,
    n_taxa: int,
    weights: np.ndarray,
    k: int,
    *,
    jmax: int | None = None,
    time_limit: float = 60.0,
    with_primal: bool = True,
    validate: bool = True,
    probe: bool = True,
    solver: str = "ilp",
) -> InstanceSolution:
    """Solve MW-GTCS over a whole filtered ground set at budgets ``0..k``.

    ``validate`` runs the independent feasibility check of :mod:`opc.solve.heuristics` on every
    ILP solution.  It is cheap next to the solve and it is the only thing standing between a
    formulation error and a silently wrong network, so it defaults on.

    ``solver="primal"`` skips the ILP entirely and builds every frontier from
    :func:`primal_frontier`.  Nothing downstream changes -- the budget DP, the closure and the
    feasibility check all run the same -- but ``optimal`` is ``False`` everywhere and callers must
    say so rather than report an "optimum".  See :func:`primal_frontier` for when that is the right
    trade.
    """
    if solver not in ("ilp", "primal"):
        raise ValueError(f"unknown solver {solver!r}")
    weights = np.asarray(weights, dtype=np.float64)
    jmax = k if jmax is None else jmax
    dec = overlap.decompose(words, n_taxa)

    # An isolated cluster is accepted only if it pays.  Under every scheme in
    # ``instance.WEIGHT_SCHEMES`` every weight is positive so this keeps the whole kernel -- but
    # the handoff's ``w = m + 1 - sum alpha`` reading makes most of it negative (A19), and
    # "isolated clusters are accepted unconditionally" is false there.
    kernel = dec.kernel[weights[dec.kernel] > 0] if dec.kernel.size else dec.kernel

    reports: list[ComponentReport] = []
    for i, comp in enumerate(dec.components):
        w = weights[comp.vertices]
        if solver == "primal":
            front, primal = primal_frontier(comp, w, jmax, time_limit=time_limit)
            flat, probe_budget, probe_sol = False, 0, None
            from_primal = [True] * len(front)
            ok = (all(heuristics.check(comp, s.accepted, j).ok for j, s in enumerate(front))
                  if validate else True)
            reports.append(ComponentReport(
                index=i, vertices=comp.vertices, n_vertices=comp.n_vertices,
                n_edges=comp.n_edges, n_hcand=comp.n_hcand, n_z=comp.n_z, cv_weight=_cv(w),
                frontier=front, flat=flat, probe_budget=probe_budget, probe=probe_sol,
                primal=primal, feasible_ok=ok, from_primal=from_primal,
            ))
            continue
        front, flat, probe_budget, probe_sol = component_frontier(
            comp, w, jmax, time_limit=time_limit, probe=probe
        )
        primal: list[heuristics.HeuristicResult] = []
        if with_primal or any(not s.has_incumbent for s in front):
            # The primal shadows the ILP, so it gets a slice of the ILP's budget rather than
            # none: a swap sweep is O(V . deg^2) and on a 1,500-vertex component it otherwise
            # runs for hours next to a 300 s solve.
            budget_s = max(1.0, time_limit / 20.0)
            primal = [
                heuristics.best_primal(comp, w, j, time_limit=budget_s)
                for j in range(len(front))
            ]

        # A solve that ran out of time before finding *any* integer point is not a solution, and
        # ``ilp.solve_component`` now says so rather than rounding the relaxation.  The primal is
        # feasible by construction, so it stands in: the frontier keeps a valid lower bound, the
        # budget DP stays sound, and ``from_primal`` records where the answer came from.
        from_primal = [False] * len(front)
        for j, sol in enumerate(front):
            if sol.has_incumbent or not primal:
                continue
            p = primal[j]
            front[j] = replace(
                sol, objective=p.objective, accepted=p.accepted,
                hybrids=np.zeros(0, np.int64), converged=True, optimal=False,
            )
            from_primal[j] = True

        ok = True
        if validate:
            # ``front[j]`` is the budget-``j`` solution (a flat frontier repeats the ``j=0`` one,
            # which uses no hybrids at all), so ``j`` is the right bound at every index.
            ok = all(heuristics.check(comp, sol.accepted, j).ok for j, sol in enumerate(front))

        reports.append(ComponentReport(
            index=i, vertices=comp.vertices, n_vertices=comp.n_vertices, n_edges=comp.n_edges,
            n_hcand=comp.n_hcand, n_z=comp.n_z, cv_weight=_cv(w), frontier=front, flat=flat,
            probe_budget=probe_budget, probe=probe_sol, primal=primal, feasible_ok=ok,
            from_primal=from_primal,
        ))

    glob = budget_dp.recombine([r.frontier for r in reports], float(weights[kernel].sum()), k)
    obj, alloc = glob.best(k)
    selected = list(kernel)
    j_used = 0
    for rep, comp, j in zip(reports, dec.components, alloc):
        sol = rep.frontier[min(j, len(rep.frontier) - 1)]
        selected += [int(comp.vertices[a]) for a in sol.accepted]
        j_used += heuristics.check(comp, sol.accepted, comp.n_hcand).n_hybrids

    return InstanceSolution(
        selected=np.array(sorted(selected), dtype=np.int64), objective=float(obj),
        kernel=kernel, kernel_weight=float(weights[kernel].sum()), allocation=list(alloc),
        components=reports, global_solution=glob, j_used=int(j_used),
    )


def _cv(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0 or w.mean() == 0:
        return float("nan")
    return float(w.std() / w.mean())
