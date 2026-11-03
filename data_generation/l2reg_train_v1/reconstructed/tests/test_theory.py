"""§8 theory tests: the claims the whole decomposition rests on.

If anything here fails, stop -- the premise of the project is wrong, not the code.

Coverage map (see SPEC-AMENDMENTS):

* §8 test 1  -> ``tests/test_relations.py::test_l_violation_counter_matches_bruteforce``
* §8 test 3a -> :func:`test_feasibility_decomposes_over_components`  (replaces the spec's test 3)
* §8 test 3b -> ``tests/test_networks.py`` (needs real networks)
* §8 test 4  -> :func:`test_closure_preserves_target_class`
* §8 test 2  -> ``tests/test_networks.py``
"""

from __future__ import annotations

from itertools import combinations

import pytest
from conftest import cluster_systems
from hypothesis import HealthCheck, given, settings

from opc.clusters import systems

SLOW = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------- 3a: decomposition is exact


@given(cluster_systems(max_taxa=7, max_clusters=9))
@SLOW
def test_feasibility_decomposes_over_components(sysdef):
    """A subset is (L)+(N3O)-feasible iff its restriction to every overlap component is.

    This is the load-bearing structural claim: an (N3O) triangle and an (L) bad triple each live
    wholly inside one component of ``O(C)``, so per-component optima recombine exactly.  We test
    the feasibility half by brute force over every subset (the objective being additive over
    clusters is arithmetic, not a claim about the decomposition).
    """
    n, cs = sysdef
    if not 1 <= len(cs) <= 9:
        return
    comps = systems.components(cs)
    for mask in range(1 << len(cs)):
        subset = {cs[i] for i in range(len(cs)) if mask >> i & 1}
        whole = systems.is_feasible(subset)
        piecewise = all(systems.is_feasible(subset & comp) for comp in comps)
        assert whole == piecewise


@given(cluster_systems(max_taxa=8, max_clusters=12))
@SLOW
def test_violations_never_span_components(sysdef):
    """The mechanism behind 3a, asserted directly: no violation has members in two components."""
    n, cs = sysdef
    comps = systems.components(cs)
    where = {c: i for i, comp in enumerate(comps) for c in comp}
    for a, b, c in combinations(cs, 3):
        pairwise = (
            systems.overlaps(a, b) and systems.overlaps(a, c) and systems.overlaps(b, c)
        )
        apex_a = systems.overlaps(a, b) and systems.overlaps(a, c) and (a & b) != (a & c)
        if pairwise or apex_a:
            assert where[a] == where[b] == where[c]


# ----------------------------------------------------- 4: closure preserves the target class


@given(cluster_systems(max_taxa=8, max_clusters=10))
@SLOW
def test_closure_preserves_target_class(sysdef):
    """Cor. 7.21, as the spec states it: if ``S`` satisfies (L)+(N3O) then so does ``I(S)``,
    and every cluster ``I`` adds overlaps nothing.

    Note the hypothesis is (L)+(N3O) on ``S`` -- *not* closedness.  Systems that already violate
    (L) or (N3O) are outside the claim and are skipped, which is why this cannot be pointed at a
    raw ``C_tau``.  How far a real ``C_tau`` sits from the class is measured descriptively in
    :func:`test_closure_of_arbitrary_systems_is_measured`, not asserted.
    """
    n, cs = sysdef
    s = set(cs)
    if not systems.is_feasible(s):
        return
    closed = systems.closure(s)
    assert systems.satisfies_n3o(closed)
    assert systems.satisfies_l(closed)
    added = closed - s
    for extra in added:
        assert not any(systems.overlaps(extra, other) for other in closed if other != extra)


@given(cluster_systems(max_taxa=8, max_clusters=10))
@SLOW
def test_closure_is_idempotent_and_closed(sysdef):
    n, cs = sysdef
    closed = systems.closure(set(cs))
    assert systems.is_closed(closed)
    assert systems.closure(closed) == closed


@given(cluster_systems(max_taxa=8, max_clusters=10))
@SLOW
def test_single_pass_closure_suffices_in_the_target_class(sysdef):
    """§3: "pairwise intersections suffice" -- true inside the target class."""
    n, cs = sysdef
    s = set(cs)
    if not systems.is_feasible(s):
        return
    one_pass = s | {a & b for a, b in combinations(s, 2) if systems.overlaps(a, b)}
    assert one_pass == systems.closure(s)


def test_closure_of_arbitrary_systems_is_measured_not_asserted():
    """Descriptive: outside the target class, ``I`` does add overlapping clusters.

    This is the reason test 4 cannot be run against a raw ``C_tau``, and the quantity itself is a
    result -- it measures how far a filtered ground set sits from level-1 realisability.
    """
    a, b, c = frozenset({0, 1, 2, 3}), frozenset({2, 3, 4, 5}), frozenset({3, 4, 6, 7})
    s = {a, b, c}
    assert not systems.is_feasible(s)  # (L) fails at apex b: b&a={2,3} vs b&c={3,4}
    closed = systems.closure(s)
    added = closed - s
    overlapping_added = [
        x for x in added if any(systems.overlaps(x, y) for y in closed if y != x)
    ]
    assert overlapping_added, "expected the closure of a non-feasible system to add overlaps"


# ------------------------------------------------------------------ N3O / L basic sanity anchors


def test_known_small_examples():
    nested = {frozenset({0, 1}), frozenset({0, 1, 2})}
    assert not systems.overlaps(*nested)
    assert systems.is_feasible(nested) and systems.is_closed(nested)

    disjoint = {frozenset({0, 1}), frozenset({2, 3})}
    assert not systems.overlaps(*disjoint)

    pair = {frozenset({0, 1, 2}), frozenset({2, 3, 4})}
    assert systems.overlaps(*pair)
    assert systems.is_feasible(pair)
    assert systems.closure(pair) == pair | {frozenset({2})}

    triangle = {frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})}
    assert not systems.satisfies_n3o(triangle)
    assert systems.l_violations(triangle) == 3  # each vertex is an apex once

    apex = {frozenset({0, 1, 2, 3}), frozenset({3, 4}), frozenset({2, 5})}
    assert systems.satisfies_n3o(apex)
    assert not systems.satisfies_l(apex)


@pytest.mark.parametrize("n_taxa", [4, 6, 8])
def test_components_partition_the_system(n_taxa):
    import numpy as np

    rng = np.random.default_rng(n_taxa)
    cs = list(dict.fromkeys(
        frozenset(rng.choice(n_taxa, size=int(rng.integers(2, n_taxa)), replace=False).tolist())
        for _ in range(30)
    ))
    comps = systems.components(cs)
    assert sum(len(c) for c in comps) == len(cs)
    assert set().union(*comps) == set(cs)
