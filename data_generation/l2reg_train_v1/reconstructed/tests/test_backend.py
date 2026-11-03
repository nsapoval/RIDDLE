"""Three MILP solvers behind one call, and the cross-check that is the point of having them.

``exp-plan-n50-mpl-v0.1`` §5.3 leaves the backend open.  HiGHS via ``scipy.optimize.milp`` has
neither lazy-constraint callbacks (§E4) nor a solution pool (§E7); SCIP has both; A44 added Gurobi
because at n = 50 it is the only one that both loads the model and returns from it.  What is worth
asserting here is that they agree, because two solvers disagreeing on an optimum is the cheapest
way to catch a mis-specified model -- the repo's standing convention applied to the solver rather
than to a kernel.

These run at sizes every license permits, which is deliberate: the agreement that matters is about
the *model*, and A44 records separately that at n = 50 there is no second opinion to be had.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from opc.clusters import extract
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N
from opc.solve import backend, instance, overlap

requires_scip = pytest.mark.skipif(
    not backend.available("scip"), reason="pyscipopt not installed"
)


#: What Gurobi says when the model is bigger than the license allows.  The wheel on PyPI ships a
#: ``TYPE=PIP`` license capped at 2,000 variables and 2,000 constraints, and the cap is enforced
#: at ``optimize()`` -- the model builds first and then refuses to solve, so this cannot be
#: anticipated from the arrays.
SIZE_LIMITED = "size-limited license"


def solve_across(c, a, b, **kwargs) -> dict[str, backend.MilpResult]:
    """Every installed backend on one model, minus any whose *license* cannot take that model.

    Two exclusions, and they are different things.  A backend that is not installed never ran.  A
    backend that is installed but size-limited ran and was refused, and is dropped **only** for
    that reason -- any other exception propagates, because a solver failing on a model the others
    handle is the exact signal this file exists to surface.

    The ``>= 2`` is the repo's rule about vacuous tests: a "cross-check" over one solver passes
    unconditionally and asserts nothing.  When a full license is installed the count goes to three
    and nothing here changes.
    """
    out: dict[str, backend.MilpResult] = {}
    refused: list[str] = []
    for name in backend.BACKENDS:
        if not backend.available(name):
            continue
        try:
            out[name] = backend.solve_milp(c, a, b, backend=name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if SIZE_LIMITED not in str(exc):
                raise
            refused.append(name)
    assert len(out) >= 2, (
        f"cross-validation needs at least two solvers; ran {sorted(out)}, "
        f"size-limited licenses refused {refused}"
    )
    return out


def _independent_set(w, edges, n):
    rows = np.repeat(np.arange(len(edges)), 2)
    cols = np.asarray(edges).ravel()
    a = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(len(edges), n)).tocsr()
    return -np.asarray(w, dtype=float), a, np.ones(len(edges))


@requires_scip
def test_the_backends_agree_on_a_hand_checkable_optimum():
    c, a, b = _independent_set([3.0, 5.0, 4.0, 2.0], [(0, 1), (1, 2), (2, 3)], 4)
    results = solve_across(c, a, b, time_limit=30, threads=1)
    for name, res in results.items():
        assert res.status == "optimal", f"{name} did not prove optimality: {res.status}"
    # {0,2} weighs 7; {1,3} weighs 7 too, and {1} alone is 5.  7 is the answer.
    assert -results["highs"].objective == pytest.approx(7.0)
    for name, res in results.items():
        assert -res.objective == pytest.approx(7.0), f"{name} disagrees with highs"


@requires_scip
def test_the_backends_agree_on_a_real_mw_gtcs_component():
    """A component built from simulated gene trees under real support weights, not a toy.

    ``j = 0`` -- maximum-weight independent set in the overlap graph -- is the one case where the
    model is small enough to solve to proven optimality on both, which is what makes the comparison
    meaningful rather than a comparison of two time limits.
    """
    n = 14
    net = N.with_gamma(N.scaled(N.with_reticulations(N.build_base(n, 2, seed=1), 2), 0.5), 0.5)
    counter = extract.SupportCounter(n, [150])
    idx = {lab: i for i, lab in enumerate(net.labels)}
    buf: list[int] = []
    for nwk in oracle.simulate_gene_trees(net, 150, seed=2):
        buf.clear()
        counter.add_locus(extract.clusters_from_newick(nwk, idx, buf))
    inst = instance.build_instance(counter, n, min_target_support=2)
    words, support, alpha = inst.at(150)
    dec = overlap.decompose(words, n)
    comp = max(dec.components, key=lambda c: c.n_vertices)
    assert comp.n_vertices >= 8, "component too small for the comparison to mean anything"

    w = instance.weight_vector(alpha, support, inst.sizes, 150, scheme="support")[comp.vertices]
    c, a, b = _independent_set(w, list(zip(comp.ii, comp.jj, strict=True)), comp.n_vertices)

    results = solve_across(c, a, b, time_limit=120, threads=1)
    for name, res in results.items():
        assert res.status == "optimal", f"{name} did not prove optimality: {res.status}"
        assert res.objective == pytest.approx(results["highs"].objective, rel=1e-9), \
            f"{name} disagrees with highs on the optimum"
    # Not only the value: every selection must actually be a feasible independent set.
    for name, res in results.items():
        chosen = set(np.flatnonzero(np.asarray(res.x) > 0.5).tolist())
        for i, j in zip(comp.ii, comp.jj, strict=True):
            assert not ({int(i), int(j)} <= chosen), f"{name} returned an infeasible selection"


def test_highs_is_always_available_and_the_optional_ones_are_reported_honestly():
    assert backend.available("highs")
    assert isinstance(backend.available("scip"), bool)
    assert isinstance(backend.available("gurobi"), bool)
    with pytest.raises(ValueError):
        backend.available("cplex")


def test_available_does_not_pretend_to_be_a_license_check():
    """``available('gurobi')`` answers "importable", and must not be read as "usable".

    The PyPI wheel ships a ``TYPE=PIP`` license capped at 2,000 variables.  Under it the import
    succeeds, this function returns ``True``, and E4's 303,298-column model fails at solve time.
    That gap is intentional -- checking a license costs an environment start, and doing it per call
    would put a network round trip inside every solve -- but it is exactly the kind of thing that
    gets misread later, so it is written down as an assertion rather than a comment.
    ``scripts/probe_gurobi.py`` is where the license is actually established, once.
    """
    if not backend.available("gurobi"):
        pytest.skip("gurobipy not installed")
    import gurobipy  # noqa: F401  -- the only claim available() makes, and it holds


@pytest.mark.skipif(not backend.available("gurobi"), reason="gurobipy not installed")
def test_gurobi_uses_the_repo_license_path_without_overriding_an_explicit_one():
    """``/home`` is full, so ``~/gurobi.lic`` is not available and the env var carries the license.

    Two halves, and the second is the one that would bite: a caller who sets ``GRB_LICENSE_FILE``
    on purpose -- a different license, a token server, a debugging run -- must still win.
    """
    import os

    saved = os.environ.get("GRB_LICENSE_FILE")
    try:
        os.environ["GRB_LICENSE_FILE"] = "/nonexistent/explicit.lic"
        backend._use_repo_license()  # noqa: SLF001
        assert os.environ["GRB_LICENSE_FILE"] == "/nonexistent/explicit.lic"

        os.environ.pop("GRB_LICENSE_FILE", None)
        backend._use_repo_license()  # noqa: SLF001
        if backend.GUROBI_LICENSE_DEFAULT.exists():
            assert os.environ["GRB_LICENSE_FILE"] == str(backend.GUROBI_LICENSE_DEFAULT)
        else:
            assert "GRB_LICENSE_FILE" not in os.environ
    finally:
        os.environ.pop("GRB_LICENSE_FILE", None)
        if saved is not None:
            os.environ["GRB_LICENSE_FILE"] = saved


def test_a_time_limit_with_no_incumbent_is_reported_rather_than_raised():
    """``milp`` sets ``mip_dual_bound`` to ``None`` -- not absent -- when it times out empty.

    ``getattr(res, "mip_dual_bound", float("nan"))`` reads as though it handles that and does not:
    the attribute exists, so the default never fires and ``float(None)`` raises ``TypeError``.  The
    worker then dies on exactly the instances a run is trying to record as "hit the limit", which
    are the ones §E4's optimality criterion is about, and nothing smaller ever reproduces it
    because small models always find an incumbent.

    Faked rather than provoked: producing a genuine no-incumbent timeout needs a model far too
    large for a unit test, which is precisely why this went unnoticed until one appeared.
    """
    from types import SimpleNamespace

    from opc.solve import backend as be

    fake = SimpleNamespace(
        x=None, fun=None, status=1, message="Time limit reached.",
        mip_dual_bound=None, mip_gap=None,
    )
    saved = be.milp if hasattr(be, "milp") else None
    import scipy.optimize as so
    original = so.milp
    try:
        so.milp = lambda *a, **k: fake
        res = be.solve_milp(
            np.array([-1.0, -1.0]),
            sp.csr_matrix(np.array([[1.0, 1.0]])),
            np.array([1.0]),
            time_limit=1.0, backend="highs",
        )
    finally:
        so.milp = original
        if saved is not None:
            be.milp = saved

    assert res.status == "limit"
    assert res.x is None
    assert np.isnan(res.dual_bound)
    assert np.isnan(res.mip_gap)
    assert np.isnan(res.objective)
    assert not res.optimal


def test_number_coerces_none_without_swallowing_real_values():
    from opc.solve.backend import _number

    assert np.isnan(_number(None))
    assert _number(None, 0.0) == 0.0
    assert _number(3) == 3.0
    assert _number(0.0) == 0.0  # a real zero must survive, unlike a missing value


def test_grb_attr_coerces_a_raising_attribute_without_swallowing_real_values():
    """``_number``'s counterpart for a library that fails by raising rather than by ``None``."""
    from opc.solve.backend import _grb_attr

    class Raises:
        @property
        def ObjVal(self):  # noqa: N802 -- gurobipy's own spelling
            raise AttributeError("Unable to retrieve attribute 'ObjVal'")

    class Has:
        ObjVal = 0.0  # noqa: N815

    assert np.isnan(_grb_attr(Raises(), "ObjVal"))
    assert _grb_attr(Raises(), "ObjVal", 0.0) == 0.0
    assert _grb_attr(Has(), "ObjVal") == 0.0  # a real zero must survive, unlike a missing value


@pytest.mark.skipif(not backend.available("gurobi"), reason="gurobipy not installed")
def test_gurobi_with_no_incumbent_is_reported_rather_than_raised():
    """The Gurobi shape of the bug A43 hit twice, and the reason ``_grb_attr`` exists.

    ``scipy.optimize.milp`` sets ``mip_dual_bound`` to ``None`` when it stops with no integer
    solution, so a ``getattr`` default never fires and ``float(None)`` raises.  Gurobi fails the
    other way round -- ``ObjVal`` and ``MIPGap`` *raise* while ``SolCount == 0`` -- and the
    consequence is identical: the worker dies on exactly the instances a run is trying to record as
    "hit the limit", which are the ones §E4's optimality criterion is about.

    Provoked rather than faked, using an infeasible model: it is genuinely no-incumbent, it is
    deterministic, and it is small.  The first half of the test asserts that raw attribute access
    really does raise -- without that, this would pass just as well against a version with no guard
    at all, which is the vacuous-test trap ``CLAUDE.md`` lists.
    """
    import gurobipy as gp

    # x binary and x <= -1: no feasible point, hence no incumbent, hence no ObjVal.
    c = np.array([-1.0])
    a = sp.csr_matrix(np.array([[1.0]]))
    b = np.array([-1.0])

    backend._use_repo_license()  # noqa: SLF001
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        probe = gp.Model(env=env)
        x = probe.addMVar(1, vtype="B", obj=c)
        probe.addConstr(a @ x <= b)
        probe.optimize()
        assert probe.SolCount == 0
        with pytest.raises(Exception):  # noqa: B017, PT011 -- gurobipy's own AttributeError
            _ = probe.ObjVal
        probe.dispose()
    finally:
        env.dispose()

    res = backend.solve_milp(c, a, b, backend="gurobi", time_limit=30, threads=1)
    assert res.status == "infeasible"
    assert res.x is None
    assert np.isnan(res.objective)
    assert not res.optimal


def test_the_reason_a_gurobi_solve_stopped_survives_the_status_collapse():
    """``status`` collapses every limit into "feasible"; ``termination`` must not.

    Measured on E4's tau = 2 grid: 62 of 67 unproven rows had been stopped by a memory cap after
    ~10 minutes, not by their four-hour time budget, and nothing in the artifact could tell the
    two apart -- so a verdict counting unproven rows was reading an operator's parameter as a
    statement about the solver.  Asserted on a real time-limited solve rather than a faked status,
    so the mapping is exercised end to end.
    """
    if not backend.available("gurobi"):
        pytest.skip("gurobi not installed")
    rng = np.random.default_rng(0)
    n = 220
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.32]
    c, a, b = _independent_set(rng.integers(1, 100, n).astype(float), edges, n)
    res = backend.solve_milp(c, a, b, integrality=np.ones(n, dtype=bool),
                             backend="gurobi", time_limit=0.05, threads=1)
    assert res.termination, "gurobi reported no termination code at all"
    assert res.termination in {"OPTIMAL", "TIME_LIMIT", "WORK_LIMIT", "INTERRUPTED",
                               "SUBOPTIMAL", "MEM_LIMIT", "NODE_LIMIT", "ITERATION_LIMIT"}
    # Non-vacuous: a code the table does not know would come back as a bare integer string, and a
    # collapse back onto `status` would make this test unable to fail.
    assert not res.termination.isdigit(), f"unmapped GRB status {res.termination}"
    if res.termination == "OPTIMAL":
        assert res.status == "optimal"


def test_the_other_backends_report_an_empty_termination_rather_than_a_wrong_one():
    """HiGHS and SCIP have no equivalent vocabulary, so they say nothing rather than guess."""
    c, a, b = _independent_set([3.0, 5.0, 4.0, 2.0], [(0, 1), (1, 2), (2, 3)], 4)
    res = backend.solve_milp(c, a, b, integrality=np.ones(4, dtype=bool),
                             backend="highs", time_limit=30.0)
    assert res.termination == ""
