"""``O(C_tau)`` decomposition and the ILP's model inputs (WS1-2).

Cross-validated against ``opc.clusters.systems``, the frozenset reference semantics, exactly as
``test_relations.py`` validates the bitset kernels.  The claims worth protecting are structural,
not numerical: which triangles need a constraint, that a candidate hybrid is never a vertex of
the component it labels, and that the kernel plus the components partition the ground set.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from conftest import cluster_systems, pack
from hypothesis import HealthCheck, given, settings

from opc import bitset
from opc.clusters import relations, systems
from opc.solve import overlap

SLOW = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@given(cluster_systems(max_taxa=7, max_clusters=10))
@SLOW
def test_kernel_and_components_partition_the_ground_set(sysdef):
    n_taxa, cs = sysdef
    dec = overlap.decompose(pack(cs, n_taxa), n_taxa)
    seen = list(dec.kernel) + [int(v) for c in dec.components for v in c.vertices]
    assert sorted(seen) == list(range(len(cs)))
    assert all(c.n_vertices >= 2 for c in dec.components), "a component has at least one edge"

    ref = [comp for comp in systems.components(cs) if len(comp) >= 2]
    assert len(dec.components) == len(ref)


@given(cluster_systems(max_taxa=7, max_clusters=9))
@SLOW
def test_monochromatic_triangles_are_exactly_the_ones_L_does_not_already_forbid(sysdef):
    """A triangle needs an (N3O) row **iff** its three intersections agree.

    If they differ the triple is an (L) violation, which the assignment rows cut at integrality:
    the apex would need two different ``z`` values at once.  If they agree, (L) is satisfied and
    only (N3O) rules the triangle out -- the sunflower ``{0,1},{0,2},{0,3}`` is the witness that
    (L) does not imply (N3O), so this family cannot be dropped.
    """
    n_taxa, cs = sysdef
    dec = overlap.decompose(pack(cs, n_taxa), n_taxa)
    got = {
        tuple(sorted(int(c.vertices[x]) for x in t))
        for c in dec.components for t in overlap.monochromatic_triangles(c)
    }
    expected = set()
    for a, b, c in combinations(range(len(cs)), 3):
        A, B, C = cs[a], cs[b], cs[c]
        if systems.overlaps(A, B) and systems.overlaps(A, C) and systems.overlaps(B, C):
            if (A & B) == (A & C) == (B & C):
                expected.add((a, b, c))
    assert got == expected


@given(cluster_systems(max_taxa=7, max_clusters=9))
@SLOW
def test_edge_classes_are_the_intersection_values(sysdef):
    n_taxa, cs = sysdef
    words = pack(cs, n_taxa)
    dec = overlap.decompose(words, n_taxa)
    for comp in dec.components:
        for e in range(comp.n_edges):
            a = cs[int(comp.vertices[comp.ii[e]])]
            b = cs[int(comp.vertices[comp.jj[e]])]
            h = comp.hcand[comp.edge_class[e]]
            assert frozenset(bitset.to_indices(h, n_taxa)) == (a & b)


@given(cluster_systems(max_taxa=7, max_clusters=9))
@SLOW
def test_z_pairs_are_exactly_the_realised_vertex_intersection_combinations(sysdef):
    n_taxa, cs = sysdef
    dec = overlap.decompose(pack(cs, n_taxa), n_taxa)
    for comp in dec.components:
        want = set()
        for e in range(comp.n_edges):
            want.add((int(comp.ii[e]), int(comp.edge_class[e])))
            want.add((int(comp.jj[e]), int(comp.edge_class[e])))
        assert set(zip(map(int, comp.z_vertex), map(int, comp.z_class))) == want


def test_a_candidate_hybrid_is_never_a_vertex_of_the_component_it_labels():
    """Why the handoff's ``x[h] >= y[h]`` row cannot be written.

    ``h = A & B`` is nested inside both ``A`` and ``B``, so it overlaps neither.  For it to reach
    their component it would have to overlap some third member, which under (L) forces a second
    intersection value at that apex.  Checked here on the feasible systems, where the claim holds
    exactly; on a raw ``C_tau`` a colliding ``h`` can arrive through unrelated clusters, which is
    why the fraction is reported rather than assumed.
    """
    rng = np.random.default_rng(19)
    checked = elsewhere = 0
    for _ in range(4000):
        n = 7
        cs = list(dict.fromkeys(
            frozenset(rng.choice(n, size=int(rng.integers(2, n)), replace=False).tolist())
            for _ in range(int(rng.integers(3, 8)))
        ))
        if len(cs) < 3 or not systems.is_feasible(set(cs)):
            continue
        dec = overlap.decompose(pack(cs, n), n)
        member = {bytes(pack([c], n)[0]) for c in cs}
        for comp in dec.components:
            in_comp = {bytes(pack([cs[int(v)]], n)[0]) for v in comp.vertices}
            for h in np.ascontiguousarray(comp.hcand):
                assert bytes(h) not in in_comp
                checked += 1
                # It may still be a member of the system somewhere else -- only co-location with
                # the pair it separates is barred.  That is the case the spec's row assumed.
                elsewhere += bytes(h) in member
    assert checked > 200, "fixture produced too few feasible systems to be meaningful"
    assert elsewhere, "h should sometimes be a member of the system, just never of its component"


def test_singleton_intersections_are_kept():
    """``C_tau`` never contains a singleton, but a block can still be defined by one.

    Dropping singleton candidates -- either branch of the handoff's "Design decision" -- would
    make the overlapping pair ``{0,1,2}, {0,3,4}`` unrepresentable, and pairs like it are the
    signature of every reticulation whose hybrid subtends a single taxon.
    """
    n = 6
    cs = [frozenset({0, 1, 2}), frozenset({0, 3, 4})]
    dec = overlap.decompose(pack(cs, n), n)
    (comp,) = dec.components
    assert comp.n_hcand == 1
    assert bitset.popcounts(comp.hcand)[0] == 1
    assert frozenset(bitset.to_indices(comp.hcand[0], n)) == frozenset({0})
    assert overlap.hcand_in_ground_set(comp, pack(cs, n)).tolist() == [False]


def test_assignment_pairs_and_edge_intersections_agree_with_the_L_counter():
    """``edge_intersections`` regroups what ``l_violations_per_vertex`` already computes.

    A vertex is (L)-clean exactly when all its incident edges carry one intersection value, so
    the number of ``z`` classes at a vertex and its bad-pair count must agree on which vertices
    are violating.
    """
    n = 8
    cs = [frozenset({0, 1, 2, 3}), frozenset({2, 3, 4, 5}), frozenset({3, 4, 6, 7})]
    words = pack(cs, n)
    graph = relations.build_overlap_graph(words, n)
    hcand, edge_class = relations.edge_intersections(graph)
    zv, zc = relations.assignment_pairs(graph.ii, graph.jj, edge_class, graph.n_vertices)
    per_vertex = {int(v): 0 for v in range(len(cs))}
    for v in zv:
        per_vertex[int(v)] += 1
    bad = relations.l_violations_per_vertex(graph)
    for v, n_classes in per_vertex.items():
        assert (n_classes > 1) == (bad[v] > 0)
