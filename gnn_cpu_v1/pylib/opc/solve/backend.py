"""One MILP call, three solvers -- so E4 can change its mind without changing its model.

``scipy.optimize.milp`` (HiGHS) is what the previous study used and it is fine for a plain
solve-to-optimality.  It has **no lazy-constraint callback and no solution pool**, and
``exp-plan-n50-mpl-v0.1`` §E4 needs the first (the (L)/(N3O) rows have to be separated, not written
out -- A33 measured 478,400 (N3O) triples for a *single* instance) and §E7 needs the second.  SCIP
has both, is Apache-2.0, and ships a manylinux wheel.  §5.3 left the choice open and A44 settled
it: at n = 50 **Gurobi is the only one of the three that can be run at all**, for two reasons that
have nothing to do with branch-and-bound quality.

*Loading.*  A43's model is 69,788,947 nonzeros.  ``_scip`` below writes it a row at a time through
``quicksum``, which is one Python-level operation per entry; ``_gurobi`` hands the CSR to the
matrix API and it crosses into C once.  That difference is not a constant factor on a model this
shape, it is the difference between minutes and never.

*Returning.*  A43 measured ``scipy.optimize.milp``'s ``time_limit`` binding at 10 s and 60 s and
**not** binding at 300 s or 900 s on that same model -- a solve that never comes back holds its
``multiprocessing`` worker until the whole run is killed.  Gurobi's ``TimeLimit`` is checked
through presolve and the root relaxation as well as the tree.  ``scripts/probe_gurobi.py`` re-runs
A43's ladder against it rather than taking the documentation's word for it.

This module is deliberately the *whole* abstraction: a single function taking the same arrays
``scipy.optimize.milp`` takes, so :mod:`opc.solve.ilp` does not have to know which solver ran.
Lazy separation and pools are **not** wrapped here -- they are not shared vocabulary, and
pretending otherwise would hide exactly the difference the backend choice is about.  E4 will call
the solver directly for those.

``threads`` is the one knob that had to be lifted into the shared signature, and it is not a
performance option.  ``RLIMIT_NPROC`` on this box is 8,192 counting *threads* and is shared across
five users; Gurobi defaults to every core it can see (192 here) and E4 runs a ``Pool``, so a
default-threaded solve at ``--jobs 48`` asks for 9,216 threads and reproduces ``CLAUDE.md``'s
silent hang by a new route.  The caller must be able to say.

The point of having several is not redundancy, it is cross-validation: the repo's standing
convention is that every result is checked against an independent implementation, and two MILP
solvers disagreeing on an optimum is the cheapest possible way to catch a mis-specified model.
**That convention is discharged at n <= 8 and not at n = 50**, where only Gurobi loads the model;
A44 records that as a stated limitation rather than leaving it implicit.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

__all__ = ["BACKENDS", "GUROBI_LICENSE_DEFAULT", "MilpResult", "available", "solve_milp"]

BACKENDS = ("highs", "scip", "gurobi")

#: Where this repo keeps its Gurobi license.  Gurobi looks at ``$GRB_LICENSE_FILE``, then
#: ``~/gurobi.lic``, then ``/opt/gurobi/gurobi.lic``; the middle one is unavailable because
#: ``/home`` is shared and full (``CLAUDE.md``'s first rule) and the last is not ours, so the first
#: is pinned here instead of in a shell profile -- a ``Pool`` worker then inherits it and no caller
#: has to remember an export.  ``.gitignore``'s ``.*/`` keeps the file out of git, which matters
#: because a WLS license holds a credential.
GUROBI_LICENSE_DEFAULT = Path(__file__).resolve().parents[3] / ".gurobi" / "gurobi.lic"


def _use_repo_license() -> None:
    """Point Gurobi at :data:`GUROBI_LICENSE_DEFAULT`, without overriding an explicit choice.

    ``setdefault``, so ``GRB_LICENSE_FILE=... python ...`` still wins.  Must run *before* the first
    ``gurobipy.Env`` is constructed -- the license is resolved at environment start, not at import.
    """
    if GUROBI_LICENSE_DEFAULT.exists():
        os.environ.setdefault("GRB_LICENSE_FILE", str(GUROBI_LICENSE_DEFAULT))


def available(backend: str) -> bool:
    """Is the backend importable?  **Not** whether its license permits the model you have.

    Deliberately cheap and deliberately not a license check: the wheel on PyPI ships a ``TYPE=PIP``
    license capped at 2,000 variables, under which ``import gurobipy`` succeeds and E4's 303,298
    column model does not.  Distinguishing those is ``scripts/probe_gurobi.py``'s job, once, rather
    than every solve's.
    """
    if backend == "highs":
        return True
    if backend == "scip":
        try:
            import pyscipopt  # noqa: F401
        except ImportError:
            return False
        return True
    if backend == "gurobi":
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            return False
        return True
    raise ValueError(f"unknown backend {backend!r}")


@dataclass
class MilpResult:
    """What all three backends agree they can report."""

    x: np.ndarray | None
    objective: float
    status: str  # "optimal" | "feasible" | "infeasible" | "unbounded" | "limit" | "error"
    dual_bound: float
    mip_gap: float
    seconds: float
    backend: str
    #: *Why* the solve stopped, in the solver's own vocabulary -- ``"TIME_LIMIT"``,
    #: ``"MEM_LIMIT"``, ``"OPTIMAL"``.  ``status`` deliberately collapses every limit into
    #: ``"feasible"`` so the three backends mean the same thing by the same word, and that
    #: collapse hid a real defect: a run stopped by its *memory* cap after ten minutes is
    #: recorded identically to one that exhausted a four-hour *time* budget, so a verdict
    #: counting unproven rows cannot tell "the solver could not do this" from "the operator
    #: capped it too low".  Empty where the backend does not report one.
    termination: str = ""

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"


def solve_milp(
    c: np.ndarray,
    a_ub: sp.spmatrix | None,
    b_ub: np.ndarray | None,
    *,
    integrality: np.ndarray | None = None,
    lower: float | np.ndarray = 0.0,
    upper: float | np.ndarray = 1.0,
    time_limit: float = 600.0,
    backend: str = "highs",
    gap_limit: float = 0.0,
    threads: int = 0,
    mem_limit: float = 0.0,
) -> MilpResult:
    """Minimise ``c @ x`` subject to ``a_ub @ x <= b_ub``.

    The signature follows ``scipy.optimize.milp``'s conventions -- **minimisation**, and a single
    upper-bounded constraint block -- because that is what :mod:`opc.solve.ilp` already builds.  A
    caller wanting equalities passes both directions; at the sizes here that costs a row, not a
    solve.

    ``threads`` (0 = the solver's own default) and ``mem_limit`` (in GB, 0 = none) are honoured by
    Gurobi and accepted-and-ignored by the other two, which have no equivalent behind
    ``scipy.optimize.milp``.  Both are guards of the same kind as ``time_limit``: they exist so a
    solve that goes wrong ends as a *row in the output* rather than as a wedged pool worker or an
    OOM kill.  See the module docstring for why ``threads`` in particular is not optional here.
    """
    if backend == "highs":
        return _highs(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit)
    if backend == "scip":
        return _scip(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit)
    if backend == "gurobi":
        return _gurobi(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit,
                       threads, mem_limit)
    raise ValueError(f"unknown backend {backend!r}")


def _number(value, default: float = float("nan")) -> float:
    """Coerce a HiGHS scalar that may be ``None`` rather than absent.

    ``getattr(res, "mip_dual_bound", nan)`` looks like it handles this and does not: when the time
    limit is reached with **no integer incumbent**, ``scipy.optimize.milp`` sets the attribute to
    ``None`` rather than omitting it, so the default never fires and ``float(None)`` raises
    ``TypeError``.  That crashes the worker on precisely the instances a run is trying to record as
    "hit the limit" -- the ones §E4's optimality criterion is about -- and it is invisible until a
    model is big enough to time out before finding anything.

    Third time this repo has been bitten at ``milp``'s time-limit boundary: A21 (a fractional
    ``res.x`` returned as if it were a solution) and A43 (per-call limits plus assembly outside the
    clock) are the other two.
    """
    return default if value is None else float(value)


def _grb_attr(model, name: str, default: float = float("nan")) -> float:
    """Read a Gurobi model attribute that may not exist yet.

    ``_number``'s counterpart, for a library that fails the other way round.  ``scipy`` sets
    ``mip_dual_bound`` to ``None``; Gurobi **raises** -- ``ObjVal`` and ``MIPGap`` need
    ``SolCount > 0``, and ``ObjBound`` needs the root relaxation to have been solved, which a time
    limit reached inside presolve means it was not.  Same consequence either way, and it is the one
    A43 spent a run learning: the worker dies on exactly the instances a run is trying to record as
    "hit the limit", which are the ones §E4's optimality criterion is about.

    Fourth time this repo has been bitten at a solver's time-limit boundary (A21, A43 twice), so
    this is coerced rather than commented, and ``tests/test_backend.py`` pins it.
    """
    try:
        return float(getattr(model, name))
    except Exception:  # noqa: BLE001 -- AttributeError from gurobipy, GurobiError, or None
        return default


def _vtype_chars(integrality: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[str]:
    """``'B'``/``'I'``/``'C'`` per column, shared so two backends cannot drift into two models.

    A binary column is an integral one whose bounds already lie inside ``[0, 1]``; anything else
    integral is a general integer.  Stating it once is the point -- a backend that disagreed here
    would solve a *different* problem and still return a plausible number, which is the failure
    mode the cross-check exists to catch and the one it would be least likely to catch.
    """
    return ["B" if integrality[j] and lo[j] >= 0 and hi[j] <= 1
            else ("I" if integrality[j] else "C")
            for j in range(len(integrality))]


def _highs(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit) -> MilpResult:
    from scipy.optimize import Bounds, LinearConstraint, milp

    t0 = time.perf_counter()
    options = {"time_limit": max(1.0, time_limit)}
    if gap_limit:
        options["mip_rel_gap"] = gap_limit
    res = milp(
        c=np.asarray(c, dtype=float),
        constraints=(LinearConstraint(a_ub, -np.inf, b_ub)
                     if a_ub is not None and a_ub.shape[0] else None),
        integrality=(np.ones(len(c)) if integrality is None else np.asarray(integrality)),
        bounds=Bounds(lower, upper),
        options=options,
    )
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(
        int(res.status), "error")
    if status == "optimal" and _number(getattr(res, "mip_gap", 0.0), 0.0) > max(gap_limit, 1e-9):
        status = "feasible"
    return MilpResult(
        x=None if res.x is None else np.asarray(res.x),
        objective=float(res.fun) if res.x is not None else float("nan"),
        status=status,
        dual_bound=_number(getattr(res, "mip_dual_bound", None)),
        mip_gap=_number(getattr(res, "mip_gap", None)),
        seconds=time.perf_counter() - t0,
        backend="highs",
    )


def _scip(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit) -> MilpResult:
    from pyscipopt import Model, quicksum

    t0 = time.perf_counter()
    c = np.asarray(c, dtype=float)
    n = len(c)
    integrality = np.ones(n) if integrality is None else np.asarray(integrality)
    lo = np.full(n, lower, dtype=float) if np.isscalar(lower) else np.asarray(lower, dtype=float)
    hi = np.full(n, upper, dtype=float) if np.isscalar(upper) else np.asarray(upper, dtype=float)

    model = Model()
    model.hideOutput()
    model.setParam("limits/time", max(1.0, time_limit))
    if gap_limit:
        model.setParam("limits/gap", gap_limit)
    vtypes = _vtype_chars(integrality, lo, hi)
    xs = [
        model.addVar(vtype=vtypes[j], lb=lo[j], ub=hi[j], obj=float(c[j]))
        for j in range(n)
    ]
    if a_ub is not None and a_ub.shape[0]:
        csr = sp.csr_matrix(a_ub)
        for i in range(csr.shape[0]):
            lo_i, hi_i = csr.indptr[i], csr.indptr[i + 1]
            if hi_i > lo_i:
                model.addCons(
                    quicksum(float(csr.data[k]) * xs[csr.indices[k]] for k in range(lo_i, hi_i))
                    <= float(b_ub[i])
                )
    model.setMinimize()
    model.optimize()

    scip_status = model.getStatus()
    n_sols = len(model.getSols())
    if scip_status == "optimal":
        status = "optimal"
    elif scip_status == "infeasible":
        status = "infeasible"
    elif scip_status == "unbounded":
        status = "unbounded"
    elif n_sols:
        status = "feasible"
    else:
        status = "limit"
    x = None
    if n_sols:
        sol = model.getBestSol()
        x = np.array([model.getSolVal(sol, v) for v in xs], dtype=float)
    try:
        gap = float(model.getGap())
    except Exception:  # noqa: BLE001
        gap = float("nan")
    return MilpResult(
        x=x,
        objective=float(model.getObjVal()) if n_sols else float("nan"),
        status=status,
        dual_bound=float(model.getDualbound()),
        mip_gap=gap,
        seconds=time.perf_counter() - t0,
        backend="scip",
    )


#: ``GRB.Status`` code -> its name, built by reflection so a code this build knows and this table
#: does not still reports something readable rather than a bare integer.
_GRB_CODE_NAMES: dict[int, str] = {}


def _grb_code_names() -> dict[int, str]:
    global _GRB_CODE_NAMES
    if not _GRB_CODE_NAMES:
        from gurobipy import GRB

        _GRB_CODE_NAMES = {
            int(getattr(GRB, n)): n
            for n in dir(GRB)
            if n.isupper() and isinstance(getattr(GRB, n, None), int)
            and n in {"LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED", "CUTOFF",
                      "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT", "SOLUTION_LIMIT",
                      "INTERRUPTED", "NUMERIC", "SUBOPTIMAL", "INPROGRESS", "USER_OBJ_LIMIT",
                      "WORK_LIMIT", "MEM_LIMIT"}
        }
    return _GRB_CODE_NAMES


def _gurobi(c, a_ub, b_ub, integrality, lower, upper, time_limit, gap_limit,
            threads, mem_limit) -> MilpResult:
    """The matrix API, not the modelling API, and that is the whole reason this backend exists.

    ``_scip`` above writes constraints one row at a time through ``quicksum``, which is a Python
    operation per *nonzero*.  On A43's 69,788,947-nonzero model that is the run, not a preamble to
    it.  ``model.addConstr(A @ x <= b)`` with a scipy CSR hands the same matrix over in one call
    and Gurobi reads it in C.
    """
    import gurobipy as gp
    from gurobipy import GRB

    _use_repo_license()

    t0 = time.perf_counter()
    c = np.asarray(c, dtype=float)
    n = len(c)
    integrality = np.ones(n) if integrality is None else np.asarray(integrality)
    lo = np.full(n, lower, dtype=float) if np.isscalar(lower) else np.asarray(lower, dtype=float)
    hi = np.full(n, upper, dtype=float) if np.isscalar(upper) else np.asarray(upper, dtype=float)

    env = model = None
    try:
        # ``OutputFlag`` is set on the *environment*, before ``start()``, so the license banner is
        # suppressed too.  Forty pool workers each announcing their license is not a log.
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.Params.TimeLimit = max(1.0, time_limit)
        # Gurobi is deterministic for a fixed thread count, so pinning both is what lets a
        # published bound re-derive.  ``Threads = 0`` means "every core", which on this box is 192
        # per worker -- see the module docstring for why that is a hang and not a speedup.
        model.Params.Seed = 0
        if threads:
            model.Params.Threads = int(threads)
        if gap_limit:
            model.Params.MIPGap = gap_limit
        if mem_limit:
            # Soft, so the solve *ends and reports* instead of being OOM-killed.  A killed
            # worker is indistinguishable from a hung one from outside the pool.
            model.Params.SoftMemLimit = float(mem_limit)

        x = model.addMVar(n, lb=lo, ub=hi, obj=c, vtype=_vtype_chars(integrality, lo, hi))
        if a_ub is not None and a_ub.shape[0]:
            model.addConstr(sp.csr_matrix(a_ub) @ x <= np.asarray(b_ub, dtype=float))
        model.ModelSense = GRB.MINIMIZE
        model.optimize()

        code = model.Status
        n_sols = int(model.SolCount)
        if code == GRB.OPTIMAL:
            status = "optimal"
        elif code == GRB.INFEASIBLE:
            status = "infeasible"
        elif code in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
            status = "unbounded"
        elif code in (GRB.TIME_LIMIT, GRB.MEM_LIMIT, GRB.WORK_LIMIT, GRB.NODE_LIMIT,
                      GRB.ITERATION_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
            status = "feasible" if n_sols else "limit"
        else:
            status = "feasible" if n_sols else "error"

        gap = _grb_attr(model, "MIPGap")
        # The same downgrade ``_highs`` applies, so that "optimal" means the same thing whichever
        # solver produced it.  ``nan > x`` is False, which is the wanted answer for an LP with no
        # gap attribute at all.
        if status == "optimal" and gap > max(gap_limit, 1e-9):
            status = "feasible"
        return MilpResult(
            x=np.asarray(x.X, dtype=float) if n_sols else None,
            objective=_grb_attr(model, "ObjVal"),
            status=status,
            dual_bound=_grb_attr(model, "ObjBound"),
            mip_gap=gap,
            seconds=time.perf_counter() - t0,
            backend="gurobi",
            termination=_grb_code_names().get(code, str(code)),
        )
    finally:
        # Not tidiness: one environment per solve over E4's 160 instances leaks a WLS session each,
        # against a license that caps how many may be open at once.
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()
