"""Primal heuristics, against the frozenset reference semantics and against the ILP optimum.

Two separate claims, and mixing them up would hide a bug in either:

* **feasibility** is checked by :mod:`opc.clusters.systems` -- the slow, explicit definition of
  (L) and (N3O) over frozensets -- never by :func:`opc.solve.heuristics.check`, which is the
  fast implementation under test;
* **quality** is checked against brute force over all subsets, the same oracle ``test_ilp.py``
  uses.  A heuristic that exceeds the optimum has produced something infeasible, which is the
  failure mode worth catching.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from conftest import pack

from opc.clusters import systems
from opc.solve import heuristics, ilp, overlap


def _brute_best(cs, w, k):
    """Max weight over feasible subsets using at most ``k`` distinct intersections."""
    best = -1.0
    for mask in range(1 << len(cs)):
        subset = [cs[i] for i in range(len(cs)) if mask >> i & 1]
        if not systems.is_feasible(set(subset)):
            continue
        hs = {a & b for a, b in combinations(subset, 2) if systems.overlaps(a, b)}
        if len(hs) > k:
            continue
        best = max(best, float(sum(w[i] for i in range(len(cs)) if mask >> i & 1)))
    return best


def _random_systems(rng, n=7, tries=200):
    """Small systems that decompose into exactly one non-trivial component."""
    for _ in range(tries):
        cs = list(dict.fromkeys(
            frozenset(rng.choice(n, size=int(rng.integers(2, n)), replace=False).tolist())
            for _ in range(int(rng.integers(3, 9)))
        ))
        if len(cs) < 3:
            continue
        dec = overlap.decompose(pack(cs, n), n)
        if len(dec.components) != 1 or dec.components[0].n_vertices < 3:
            continue
        yield cs, dec


# ------------------------------------------------------------------------------- the validator


def test_check_agrees_with_the_frozenset_definition_of_feasibility():
    """``check`` is used to validate the *ILP's* output too, so it gets its own oracle.

    ``systems.is_feasible`` is (L) and (N3O) written out over frozensets; ``check`` re-derives
    them from an edge list with a sparse triangle count.  They must agree on every subset, not
    just on the feasible ones.
    """
    rng = np.random.default_rng(3)
    agreed = infeasible = 0
    for cs, dec in _random_systems(rng):
        comp = dec.components[0]
        for mask in range(1 << comp.n_vertices):
            local = [i for i in range(comp.n_vertices) if mask >> i & 1]
            subset = {cs[int(comp.vertices[i])] for i in local}
            hs = {a & b for a, b in combinations(subset, 2) if systems.overlaps(a, b)}
            got = heuristics.check(comp, np.array(local, dtype=np.int64), budget=len(hs))
            assert got.ok == systems.is_feasible(subset)
            if got.ok:
                assert got.n_hybrids == len(hs), "hybrid count must be the intersection count"
            else:
                infeasible += 1
            agreed += 1
        if agreed > 3000:
            break
    assert infeasible > 100, "fixture produced no infeasible subsets -- the test is vacuous"


def test_the_budget_is_part_of_feasibility():
    """A set can satisfy (L) and (N3O) and still be illegal because it costs too much."""
    n = 6
    cs = [frozenset({0, 1, 2}), frozenset({2, 3, 4})]
    dec = overlap.decompose(pack(cs, n), n)
    (comp,) = dec.components
    both = np.array([0, 1], dtype=np.int64)
    assert heuristics.check(comp, both, budget=1).ok
    over = heuristics.check(comp, both, budget=0)
    assert not over.ok and over.over_budget and over.n_hybrids == 1


# ---------------------------------------------------------------------------------- the search


@pytest.mark.parametrize("key", ["w", "w-per-retic"])
def test_greedy_is_always_feasible_and_never_beats_the_optimum(key):
    rng = np.random.default_rng(19 + len(key))
    tested = nonzero_budget = 0
    for cs, dec in _random_systems(rng):
        comp = dec.components[0]
        w = rng.integers(1, 20, size=len(cs)).astype(float)
        for j in (0, 1, 2):
            got = heuristics.greedy(comp, w[comp.vertices], j, key=key)
            subset = {cs[int(comp.vertices[a])] for a in got.accepted}
            assert systems.is_feasible(subset), "the reference semantics must accept it"
            hs = {a & b for a, b in combinations(subset, 2) if systems.overlaps(a, b)}
            assert len(hs) <= j
            assert got.n_hybrids == len(hs)
            opt = _brute_best(list(cs), w, j) - float(w[dec.kernel].sum())
            assert got.objective <= opt + 1e-9
            tested += 1
            nonzero_budget += got.n_hybrids > 0
        if tested > 120:
            break
    assert tested > 100
    assert nonzero_budget > 5, "fixture never exercised the budget -- (L) and the DP are untested"


def test_local_search_only_improves_and_stays_feasible():
    rng = np.random.default_rng(23)
    improved = tested = 0
    for cs, dec in _random_systems(rng):
        comp = dec.components[0]
        w = rng.integers(1, 30, size=len(cs)).astype(float)
        for j in (0, 1, 2):
            g = heuristics.greedy(comp, w[comp.vertices], j)
            ls = heuristics.local_search(comp, w[comp.vertices], j, g.accepted)
            assert ls.objective >= g.objective - 1e-9
            subset = {cs[int(comp.vertices[a])] for a in ls.accepted}
            assert systems.is_feasible(subset)
            opt = _brute_best(list(cs), w, j) - float(w[dec.kernel].sum())
            assert ls.objective <= opt + 1e-9
            improved += ls.objective > g.objective + 1e-9
            tested += 1
        if tested > 90:
            break
    assert tested > 60
    assert improved > 0, "local search never fired -- the swap neighbourhood is untested"


def test_best_primal_dominates_its_constituents_and_usually_reaches_the_optimum():
    """The RL gate's instrument.  On instances this small the primal should already be optimal.

    That is the point: if a greedy plus a swap already closes the gap at ``n=7``, the interesting
    question is whether it still does at ``V > 60``, which is what ``run_solver.py`` measures.
    """
    rng = np.random.default_rng(29)
    hits = tested = 0
    for cs, dec in _random_systems(rng):
        comp = dec.components[0]
        w = rng.integers(1, 40, size=len(cs)).astype(float)
        for j in (0, 1, 2):
            bp = heuristics.best_primal(comp, w[comp.vertices], j)
            g = heuristics.greedy(comp, w[comp.vertices], j)
            assert bp.objective >= g.objective - 1e-9
            sol = ilp.solve_component(comp, w[comp.vertices], j, time_limit=20)
            assert sol.optimal
            assert bp.objective <= sol.objective + 1e-9
            hits += bp.objective >= sol.objective - 1e-9
            tested += 1
        if tested > 60:
            break
    assert tested > 45
    assert hits / tested > 0.8, f"primal reached the ILP optimum on only {hits}/{tested}"


def test_the_ilp_solution_passes_the_independent_feasibility_check():
    """Cross-validation in the other direction: the solver's own output, checked by ``check``.

    ``run_solver.py`` runs this assertion on every solve; here it is pinned on small instances
    where the ILP is known to be optimal.
    """
    rng = np.random.default_rng(37)
    tested = 0
    for cs, dec in _random_systems(rng):
        comp = dec.components[0]
        w = rng.integers(1, 20, size=len(cs)).astype(float)
        for j in (0, 1, 2):
            sol = ilp.solve_component(comp, w[comp.vertices], j, time_limit=20)
            feas = heuristics.check(comp, sol.accepted, j)
            assert feas.ok, (feas, sol.accepted)
            assert feas.n_hybrids <= j
            tested += 1
        if tested > 60:
            break
    assert tested > 45
