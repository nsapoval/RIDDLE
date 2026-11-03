"""B1 gates for the solver core: the ILP and the budget DP (WS1-3, WS1-4).

Every test here compares against **brute force over all subsets** using
``opc.clusters.systems`` -- the frozenset reference semantics -- rather than against another
piece of the solver.  That is the only way to catch a formulation error, as opposed to an
implementation one, and the formulation is where the handoff spec was wrong.

The fixture in :func:`_two_blocks` deliberately draws clusters from two disjoint taxon ranges.
Uniformly random systems on a small taxon set almost always collapse into a single overlap
component, so a naive fixture leaves the budget DP -- the load-bearing claim of the whole
decomposition -- completely untested while appearing to pass.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from conftest import pack

from opc.clusters import systems
from opc.solve import budget_dp, ilp, overlap, pipeline, truth


def _brute_frontier(cs, w, k_max):
    """``max weight`` over feasible subsets using at most ``k`` distinct intersections."""
    best = [-1.0] * (k_max + 1)
    for mask in range(1 << len(cs)):
        subset = [cs[i] for i in range(len(cs)) if mask >> i & 1]
        if not systems.is_feasible(set(subset)):
            continue
        hs = {a & b for a, b in combinations(subset, 2) if systems.overlaps(a, b)}
        val = float(sum(w[i] for i in range(len(cs)) if mask >> i & 1))
        for k in range(len(hs), k_max + 1):
            best[k] = max(best[k], val)
    return best


def _two_blocks(rng, n=12):
    """A system on two disjoint taxon ranges, so ``O(C)`` reliably has two components."""
    def block(lo, hi, k):
        span = list(range(lo, hi))
        return [
            frozenset(rng.choice(span, size=int(rng.integers(2, len(span))),
                                 replace=False).tolist())
            for _ in range(k)
        ]

    half = n // 2
    return list(dict.fromkeys(
        block(0, half, int(rng.integers(2, 4))) + block(half, n, int(rng.integers(2, 4)))
    ))


# ------------------------------------------------------------------------------------ T1 / T3


def test_the_ilp_optimum_matches_brute_force_on_single_components():
    """The formulation itself: objective, (L), (N3O) and the reticulation count together.

    A wrong constraint shows up here as an objective that is too high (something infeasible was
    accepted) or too low (something feasible was forbidden).  Both directions matter -- the
    handoff's ``x[h] >= y[h]`` row fails the second way, and dropping singleton candidates from
    ``Hcand`` fails it too.
    """
    rng = np.random.default_rng(11)
    n, tested = 7, 0
    for _ in range(150):
        cs = list(dict.fromkeys(
            frozenset(rng.choice(n, size=int(rng.integers(2, n)), replace=False).tolist())
            for _ in range(int(rng.integers(3, 8)))
        ))
        if len(cs) < 3:
            continue
        w = rng.integers(1, 20, size=len(cs)).astype(float)
        dec = overlap.decompose(pack(cs, n), n)
        if len(dec.components) != 1:
            continue
        comp = dec.components[0]
        expected = _brute_frontier(cs, w, 2)
        for j in range(3):
            sol = ilp.solve_component(comp, w[comp.vertices], j, time_limit=30)
            assert sol.objective + float(w[dec.kernel].sum()) == pytest.approx(expected[j])
            assert sol.optimal
            tested += 1
    assert tested > 100


def test_decomposition_and_budget_dp_reproduce_the_global_optimum():
    """B1's T3, on systems that genuinely split.

    ``sum_K j_K`` is the reticulation count of the whole solution -- the claim the handoff worries
    about in T4.  It holds because two components can never share an intersection value, so the
    budget is additive in exactly the way the weight is.
    """
    rng = np.random.default_rng(31)
    multi = tested = 0
    for _ in range(120):
        cs = _two_blocks(rng)
        if not 4 <= len(cs) <= 11:
            continue
        n = 12
        w = rng.integers(1, 25, size=len(cs)).astype(float)
        dec = overlap.decompose(pack(cs, n), n)
        multi += len(dec.components) >= 2
        fronts = [ilp.frontier(c, w[c.vertices], 3, time_limit=20) for c in dec.components]
        got = budget_dp.recombine(fronts, float(w[dec.kernel].sum()), 3)
        expected = _brute_frontier(cs, w, 3)
        for k in range(4):
            assert got.objective[k] == pytest.approx(expected[k])
            tested += 1
    assert multi > 40, "fixture failed to produce multi-component systems -- the DP is untested"
    assert tested > 200


def test_two_components_never_share_a_candidate_hybrid():
    """The theorem the budget DP's exactness rests on, checked directly.

    If two components could nominate the same ``h``, ``sum_K j_K`` would over-count and the DP
    would be inexact.  Under (L) they cannot: the cross pairs are nested, and chasing the
    inclusions contradicts one of the two edges being an overlap.
    """
    rng = np.random.default_rng(77)
    checked = 0
    for _ in range(600):
        cs = _two_blocks(rng)
        if len(cs) < 4 or not systems.is_feasible(set(cs)):
            continue
        dec = overlap.decompose(pack(cs, 12), 12)
        if len(dec.components) < 2:
            continue
        seen: set[bytes] = set()
        for comp in dec.components:
            here = {bytes(h) for h in np.ascontiguousarray(comp.hcand)}
            assert not (here & seen), "two components share an intersection value"
            seen |= here
            checked += 1
    assert checked > 20


# ------------------------------------------------------------------------------------ T4a


def test_sum_of_component_budgets_equals_the_reticulation_number():
    """T4a, the gate that had no home until ``opc.solve.pipeline`` existed.

    ``sum_K j_K`` is what the budget DP spends; ``rho(I(S*)) = |E| - |V| + 1`` is what the
    network the answer stands for actually carries, derived from the cover relation with no
    reference to the budget.  They are two independent computations of the same quantity, and
    ``run_solver.py`` asserts this on every solve of the real grid.
    """
    rng = np.random.default_rng(131)
    multi = tested = 0
    for _ in range(80):
        cs = _two_blocks(rng)
        if not 4 <= len(cs) <= 10:
            continue
        n = 12
        w = rng.integers(1, 25, size=len(cs)).astype(float)
        sol = pipeline.solve_ground_set(pack(cs, n), n, w, 3, time_limit=20)
        multi += len(sol.components) >= 2
        rho = truth.reticulations_of(
            np.ascontiguousarray(pack(cs, n)[sol.selected]), n
        )
        assert sol.j_used == rho, (sol.j_used, rho)
        tested += 1
    assert tested > 30
    assert multi > 10, "fixture failed to produce multi-component systems -- T4a is untested"


def test_the_constructed_over_count_instance_is_charged_once():
    """T4c.  The configuration T4 was written to worry about, built rather than sampled.

    Nine taxa, four clusters that all meet in ``h = {8}`` and in nothing else: a true block that
    has fragmented into four pieces, every one of which "nominates" ``h``.  Six overlapping pairs,
    six nominations, one reticulation.  Two things have to hold and both are load-bearing:

    * the four pieces land in **one** overlap component, so the budget DP charges ``h`` once
      rather than once per fragment -- the over-count T4 feared; and
    * (L) holds while (N3O) fails, so the sunflower is exactly the case that needs the triangle
      family.  At ``j = 1`` at most two of the four can be accepted.

    A second, disjoint block makes the global count 2, so the test also sees that the DP adds
    across components rather than collapsing them.
    """
    n = 12
    petals = [frozenset({0, 1, 8}), frozenset({2, 3, 8}), frozenset({4, 5, 8}),
              frozenset({6, 7, 8})]
    other = [frozenset({9, 10}), frozenset({10, 11})]
    cs = petals + other
    w = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    dec = overlap.decompose(pack(cs, n), n)
    assert len(dec.components) == 2, "the sunflower must not split -- that is the whole claim"
    sun = next(c for c in dec.components if c.n_vertices == 4)
    assert sun.n_edges == 6 and sun.n_hcand == 1, "six nominations of a single intersection"
    assert systems.satisfies_l(set(petals)) and not systems.satisfies_n3o(set(petals))

    sol = pipeline.solve_ground_set(pack(cs, n), n, w, 2, time_limit=30)
    sun_rep = next(r for r in sol.components if r.n_vertices == 4)
    at_one = sun_rep.frontier[1]
    assert at_one.optimal and at_one.accepted.size == 2, "(N3O) caps a monochromatic clique at 2"
    assert at_one.objective == pytest.approx(19.0)

    assert sol.j_used == 2, "one reticulation per block, not one per nominating pair"
    assert sol.j_used == truth.reticulations_of(
        np.ascontiguousarray(pack(cs, n)[sol.selected]), n
    )


def test_the_flat_frontier_probe_agrees_with_the_full_sweep():
    """The two-ended sandwich must never change an answer, only the number of solves.

    ``f_K`` is non-decreasing, so ``f_K(0) == f_K(j_max)`` pins every point between.  The invalid
    shortcut -- stopping at the first plateau -- is not implemented, and this test would not
    catch it if it were, which is why the reasoning is written out in the module docstring.
    """
    rng = np.random.default_rng(211)
    flat = swept = 0
    for _ in range(120):
        cs = list(dict.fromkeys(
            frozenset(rng.choice(8, size=int(rng.integers(2, 8)), replace=False).tolist())
            for _ in range(int(rng.integers(3, 8)))
        ))
        if len(cs) < 3:
            continue
        dec = overlap.decompose(pack(cs, 8), 8)
        if not dec.components:
            continue
        comp = dec.components[0]
        w = rng.integers(1, 20, size=len(cs)).astype(float)[comp.vertices]
        probed, is_flat, _, _ = pipeline.component_frontier(comp, w, 3, time_limit=20)
        full = ilp.frontier(comp, w, 3, time_limit=20)
        assert [s.objective for s in probed] == pytest.approx([s.objective for s in full])
        flat += is_flat
        swept += not is_flat
    assert flat > 5 and swept > 5, f"need both branches exercised (flat={flat}, swept={swept})"


# ------------------------------------------------------------------------------------ T6


def test_the_frontier_is_non_decreasing_in_the_budget():
    """Trivially true of optima -- which is exactly why it is a live check.

    A ``j``-feasible solution is ``(j+1)``-feasible, so a decrease can only mean the points are
    time-capped incumbents rather than optima.  Asserted only where every point closed.
    """
    rng = np.random.default_rng(5)
    seen = 0
    for _ in range(60):
        cs = _two_blocks(rng)
        if not 4 <= len(cs) <= 10:
            continue
        w = rng.integers(1, 25, size=len(cs)).astype(float)
        dec = overlap.decompose(pack(cs, 12), 12)
        for comp in dec.components:
            f = ilp.frontier(comp, w[comp.vertices], 3, time_limit=20)
            if not all(s.optimal for s in f):
                continue
            objs = [s.objective for s in f]
            assert objs == sorted(objs), objs
            seen += 1
    assert seen > 20


def test_j_max_is_clipped_to_what_a_component_can_use():
    """A reticulation needs two clusters to nominate its hybrid, so ``j_max = floor(V/2)``.

    Not clipping wastes solves; clipping *below* this would silently lose optima, which is why
    T3 above runs the DP with the clip in place.
    """
    n = 6
    cs = [frozenset({0, 1, 2}), frozenset({2, 3, 4})]
    dec = overlap.decompose(pack(cs, n), n)
    (comp,) = dec.components
    assert comp.n_vertices == 2
    assert len(ilp.frontier(comp, np.ones(2), 5, time_limit=10)) == 2  # j = 0, 1


def test_j_zero_is_maximum_weight_independent_set():
    """``k=0`` must reduce to Than-Nakhleh's max-weight clique in the compatibility graph.

    Built as a pure MWIS with no ``z``/``y`` columns, both because that is the reduction and
    because the extra columns defeat HiGHS's clique-table detection.
    """
    n = 6
    cs = [frozenset({0, 1, 2}), frozenset({2, 3, 4}), frozenset({0, 1})]
    w = np.array([5.0, 4.0, 3.0])
    dec = overlap.decompose(pack(cs, n), n)
    (comp,) = dec.components
    sol = ilp.solve_component(comp, w[comp.vertices], 0, time_limit=10)
    assert sol.hybrids.size == 0
    accepted = {cs[int(comp.vertices[a])] for a in sol.accepted}
    assert len(accepted) == 1, "the two overlapping clusters cannot both be accepted at k=0"
    assert sol.n_binaries == comp.n_vertices
