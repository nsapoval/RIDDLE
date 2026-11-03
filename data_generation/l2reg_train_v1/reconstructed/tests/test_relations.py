"""Kernel correctness: the fast bitset path must equal explicit set arithmetic.

Everything downstream -- component sizes, the verdict itself -- rests on the overlap predicate and
the ``O(E)`` (L) counter being right, so both are cross-validated against brute force here.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import cluster_systems, pack
from hypothesis import given, settings

from opc import bitset
from opc.clusters import relations, systems

# ---------------------------------------------------------------------------- bitset round-trips


@given(cluster_systems())
@settings(max_examples=200, deadline=None)
def test_bitset_roundtrip(sysdef):
    n, cs = sysdef
    for c in cs:
        bits = bitset.from_indices(sorted(c), n)
        assert bitset.to_indices(bits, n) == sorted(c)
        assert bitset.from_key(bitset.key(bits)).tolist() == bits.tolist()


@given(cluster_systems())
@settings(max_examples=200, deadline=None)
def test_dense_and_popcount_agree_with_python(sysdef):
    n, cs = sysdef
    if not cs:
        return
    words = pack(cs, n)
    dense = bitset.to_dense(words, n)
    sizes = bitset.popcounts(words)
    for i, c in enumerate(cs):
        assert sizes[i] == len(c)
        assert set(np.flatnonzero(dense[i]).tolist()) == set(c)


def _fnv1a_reference(data: bytes) -> int:
    """Textbook FNV-1a 64 over a byte string, in plain Python ints."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def test_fnv1a_reference_matches_published_vectors():
    assert _fnv1a_reference(b"") == 0xCBF29CE484222325
    assert _fnv1a_reference(b"a") == 0xAF63DC4C8601EC8C
    assert _fnv1a_reference(b"foobar") == 0x85944171F73967E8


@given(cluster_systems(max_taxa=8))
@settings(max_examples=100, deadline=None)
def test_fnv1a_matches_reference_and_vectorised_form(sysdef):
    """The content id is a cross-language join key, so it must be the real FNV-1a, not a
    numpy-specific accident."""
    n, cs = sysdef
    if not cs:
        return
    words = pack(cs, n)
    scalar = [bitset.fnv1a(words[i]) for i in range(words.shape[0])]
    assert scalar == [_fnv1a_reference(words[i].tobytes()) for i in range(words.shape[0])]
    assert bitset.fnv1a_many(words).tolist() == scalar


def test_fnv1a_is_pinned_across_word_widths():
    # Regression pins, computed from the reference above; these must not drift.
    assert bitset.fnv1a(bitset.from_indices([0], 64)) == 9929646806074584996
    for n in (64, 130, 200):
        bits = bitset.from_indices([0, 5], n)
        assert bitset.fnv1a(bits) == _fnv1a_reference(bits.tobytes())


# ------------------------------------------------------------------------------- overlap kernels


@given(cluster_systems())
@settings(max_examples=300, deadline=None)
def test_both_kernels_match_bruteforce(sysdef):
    n, cs = sysdef
    if len(cs) < 2:
        return
    words = pack(cs, n)
    expected = relations.overlap_pairs_bruteforce(words, n)
    for kernel in ("gemm", "popcount"):
        g = relations.build_overlap_graph(words, n, block=3, kernel=kernel)
        got = set(zip(g.ii.tolist(), g.jj.tolist()))
        assert got == expected, kernel
        assert g.n_edges == len(expected)
        # Degrees must be consistent with the edge list, including on diagonal tiles.
        deg = np.zeros(len(cs), dtype=np.int64)
        for a, b in expected:
            deg[a] += 1
            deg[b] += 1
        assert g.degrees.tolist() == deg.tolist()


@given(cluster_systems())
@settings(max_examples=200, deadline=None)
def test_overlap_predicate_matches_systems_module(sysdef):
    n, cs = sysdef
    if len(cs) < 2:
        return
    words = pack(cs, n)
    g = relations.build_overlap_graph(words, n, block=4)
    got = set(zip(g.ii.tolist(), g.jj.tolist()))
    expected = {
        (i, j)
        for i in range(len(cs))
        for j in range(i + 1, len(cs))
        if systems.overlaps(cs[i], cs[j])
    }
    assert got == expected


def test_block_size_does_not_change_the_answer():
    rng = np.random.default_rng(11)
    n = 40
    cs = []
    while len(cs) < 300:
        size = int(rng.integers(2, n - 1))
        cs.append(frozenset(rng.choice(n, size=size, replace=False).tolist()))
    cs = list(dict.fromkeys(cs))
    words = pack(cs, n)
    ref = relations.build_overlap_graph(words, n, block=len(cs), kernel="popcount")
    for block in (7, 64, 2048):
        g = relations.build_overlap_graph(words, n, block=block)
        assert g.n_edges == ref.n_edges
        assert set(zip(g.ii.tolist(), g.jj.tolist())) == set(zip(ref.ii.tolist(), ref.jj.tolist()))


def test_edge_cap_triggers_dense_bailout():
    rng = np.random.default_rng(3)
    n = 30
    cs = list(dict.fromkeys(
        frozenset(rng.choice(n, size=15, replace=False).tolist()) for _ in range(200)
    ))
    words = pack(cs, n)
    full = relations.build_overlap_graph(words, n)
    assert full.n_edges > 50
    capped = relations.build_overlap_graph(words, n, block=8, edge_cap=10)
    assert capped.dense_bailout
    assert capped.ii is None
    # The count and degrees are still exact -- only materialisation is skipped.
    assert capped.n_edges == full.n_edges
    assert capped.degrees.tolist() == full.degrees.tolist()
    m = relations.component_metrics(capped)
    assert m.giant_size == capped.n_vertices and m.n_triangles == -1


# ------------------------------------------------------------------------- exact (L) counting


@given(cluster_systems())
@settings(max_examples=300, deadline=None)
def test_l_violation_counter_matches_bruteforce(sysdef):
    """§8 test 1. Licenses computing ``n_bad_triples`` exactly instead of sampling."""
    n, cs = sysdef
    if len(cs) < 2:
        return
    words = pack(cs, n)
    g = relations.build_overlap_graph(words, n, block=5)
    fast = int(relations.l_violations_per_vertex(g).sum())
    assert fast == systems.l_violations(cs)


@given(cluster_systems())
@settings(max_examples=200, deadline=None)
def test_component_metrics_match_bruteforce(sysdef):
    n, cs = sysdef
    if len(cs) < 2:
        return
    words = pack(cs, n)
    g = relations.build_overlap_graph(words, n, block=5)
    m = relations.component_metrics(g)

    expected_comps = systems.components(cs)
    assert m.n_components == len(expected_comps)
    assert sorted(m.comp_sizes.tolist()) == sorted(len(c) for c in expected_comps)

    n_tri = sum(
        1
        for i in range(len(cs))
        for j in range(i + 1, len(cs))
        for k in range(j + 1, len(cs))
        if systems.overlaps(cs[i], cs[j])
        and systems.overlaps(cs[i], cs[k])
        and systems.overlaps(cs[j], cs[k])
    )
    assert m.n_triangles == n_tri
    assert m.n_triangles_exact
    assert (m.n_triangles == 0) == systems.satisfies_n3o(cs)
    assert (m.n_l_violations == 0) == systems.satisfies_l(cs)
    assert g.n_isolated == sum(1 for c in expected_comps if len(c) == 1)


def test_sampled_triangles_are_close_to_exact():
    """Above the dense gate we estimate; the estimate must stay in the right ballpark."""
    rng = np.random.default_rng(5)
    n = 60
    cs = list(dict.fromkeys(
        frozenset(rng.choice(n, size=int(rng.integers(5, 40)), replace=False).tolist())
        for _ in range(900)
    ))
    words = pack(cs, n)
    g = relations.build_overlap_graph(words, n)
    exact = relations.component_metrics(g, dense_triangle_max=10_000)
    est = relations.component_metrics(g, dense_triangle_max=10, triangle_sample=5000, seed=1)
    assert exact.n_triangles_exact and not est.n_triangles_exact
    assert exact.n_triangles > 1000
    assert est.n_triangles == pytest.approx(exact.n_triangles, rel=0.2)


# ------------------------------------------------------------------------------ p99 definition


def test_cluster_weighted_p99_is_not_degenerate():
    """D6: with few components the plain p99 is just the max; the weighted one is not."""
    sizes = np.array([100, 5, 5, 5, 5])
    assert np.percentile(sizes, 99) > 0.95 * sizes.max()  # ~96.2 -- effectively the maximum
    assert relations.cluster_weighted_p99(sizes) == 100.0

    # Weighting by cluster count asks the operational question -- how big is the subproblem a
    # randomly chosen cluster lands in -- and gives a different answer.
    many_small = np.array([50] + [1] * 500)
    assert np.percentile(many_small, 99) == 1.0
    assert relations.cluster_weighted_p99(many_small) == 50.0


def test_no_hash_collisions_over_a_million_clusters():
    """§8 test 5's second half, which had no test until now.

    ``bitset.fnv1a`` is not the dedup key -- SPEC-AMENDMENTS A6 replaced that with the raw bitset
    bytes -- but it is the ``cluster_id`` published in every output row and the join key any
    cross-language consumer would use.  A collision there silently merges two clusters in a join.
    Checked at three widths because the hash runs over the *fixed-width* byte stream, so the
    number of words is part of the input.
    """
    import numpy as np

    from opc import bitset

    for n_taxa in (64, 128, 200):
        rng = np.random.default_rng(n_taxa)
        w = bitset.n_words(n_taxa)
        words = rng.integers(0, 2**63, size=(1_000_000, w), dtype=np.int64).astype(np.uint64)
        # Distinct inputs are what the claim is about; drop the (vanishingly few) duplicates.
        uniq = np.unique(words, axis=0)
        ids = bitset.fnv1a_many(uniq)
        assert np.unique(ids).size == uniq.shape[0], (
            f"FNV-1a collision over {uniq.shape[0]:,} distinct clusters at n_taxa={n_taxa}"
        )
