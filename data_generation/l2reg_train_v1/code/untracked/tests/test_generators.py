"""The level-k generators and the blob classifier (A64).

Three checks, none of which shares code with :mod:`opc.simulate.generators`:

* The enumeration against the published counts (Gambette, Berry & Paul 2009: 1, 4, 65), which it
  never reads, and each entry against the definition, read off its arc list here.
* The classifier against :mod:`level2`'s theta-graph shapes, on every placement that
  ``tests/test_level2.py``'s exhaustive fixture admits.  The four level-2 generators must be the
  four shapes, one to one.
* Every generator realised as a network, by a construction written here that never calls the
  placement code, then classified back.  Hopcroft-Tarjan (:func:`networks.blocks`) is what says
  the realisation is one blob, so this also checks the enumeration's biconnectivity test.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from opc.simulate import generators as G
from opc.simulate import level2 as L2
from opc.simulate import networks as N
from test_level2 import reference_girth, reference_regular, shape_fixture, with_r1

COUNTS = {1: 1, 2: 4, 3: 65}
BY_SINKS = {1: {1: 1}, 2: {1: 2, 2: 2}, 3: {1: 17, 2: 32, 3: 16}}
KIND = {(0, 2): 0, (1, 2): 1, (2, 1): 2, (2, 0): 3}   # root, tree vertex, reticulation, sink


def _all() -> list[G.Generator]:
    return [g for k in COUNTS for g in G.generators(k)]


# ------------------------------------------------------------------------------ the tables


def test_counts_match_the_literature():
    for k, n in COUNTS.items():
        gs = G.generators(k)
        assert len(gs) == n
        assert dict(Counter(g.n_sinks for g in gs)) == BY_SINKS[k]
        assert len({g.id for g in gs}) == len({g.arcs for g in gs}) == n
    assert sum(g.parallel_arcs for g in G.generators(3)) == 16


def test_every_entry_satisfies_the_definition():
    for g in _all():
        indeg: Counter = Counter()
        outdeg: Counter = Counter()
        for u, v in g.arcs:
            outdeg[u] += 1
            indeg[v] += 1
        verts = sorted(set(indeg) | set(outdeg))
        assert verts == list(range(2 * g.k + g.n_sinks - 1)) == list(range(g.n_vertices))
        assert len(g.arcs) == 3 * g.k + g.n_sinks - 2 == g.n_arcs
        kinds = [KIND[(indeg[v], outdeg[v])] for v in verts]   # a KeyError is a bad vertex
        assert kinds == sorted(kinds), "the canonical form lists root, tree, reticulation, sink"
        assert Counter(kinds) == +Counter({0: 1, 1: g.k + g.n_sinks - 2, 2: g.k - g.n_sinks,
                                           3: g.n_sinks})
        left, ready, done = dict(indeg), [0], 0              # acyclic, by Kahn's algorithm
        while ready:
            x = ready.pop()
            done += 1
            for u, v in g.arcs:
                if u == x:
                    left[v] -= 1
                    if left[v] == 0:
                        ready.append(v)
        assert done == g.n_vertices, f"{g.id} has a directed cycle"


def test_canonical_form_ignores_labels_and_arc_order():
    rng = random.Random(20260911)
    for g in _all():
        for _ in range(20):
            ids = rng.sample(range(100, 200), g.n_vertices)
            arcs = [(ids[u], ids[v]) for u, v in g.arcs]
            rng.shuffle(arcs)
            assert G.canonical(arcs) == g.arcs
            assert G.lookup(arcs) is g


def test_the_tables_are_pinned():
    """The IDs are ground truth in a published dataset, so a change here must be deliberate."""
    for k in COUNTS:
        assert G.table_sha256(k) == G.PINNED[k]


def test_what_is_not_a_generator_is_refused():
    with pytest.raises(ValueError, match="not a level-2 generator"):
        G.lookup([(0, 1), (0, 1), (1, 2), (2, 3), (2, 3)])   # two level-1 generators in series
    with pytest.raises(ValueError, match="no generator vertex"):
        G.lookup([(0, 1), (1, 2), (0, 2)])                   # vertex 1 has in 1 / out 1
    with pytest.raises(ValueError):
        G.generators(4)


# ------------------------------------------------------------------------------ level 2: S0-S3


def _level2_placements():
    """``test_every_second_placement_agrees_with_every_reference``'s placements that join r1's
    blob, ladders excluded: ``(shape, generator, theta)`` for each."""
    net, _, _, h1 = with_r1()
    blocks = L2.nontrivial_blocks(net)
    emap = {e: i for i, b in enumerate(blocks) for e in b.edges}
    target = blocks.index(L2.block_of(blocks, h1))
    times = sorted(set(net.time))
    for t in [(a + b) / 2 for a, b in zip(times, times[1:])]:
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        for c1 in alive:
            for c2 in alive:
                if c1 == c2:
                    continue
                z = N._lca(net, c1, c2)  # noqa: SLF001
                s1, s2 = L2._sides(net, c1, c2, z)  # noqa: SLF001
                if L2._touched(net, emap, s1 + s2) != {target}:  # noqa: SLF001
                    continue
                trial = net.copy()
                L2.place(trial, c1, c2, t)
                if L2.ladders(trial):
                    continue
                b = L2.block_of(L2.nontrivial_blocks(trial), h1)
                yield L2.blob_shape(b), G.blob_generator(b), L2.theta(b)


#: The level-2 generators by :mod:`level2`'s names.  S2, the nested cycle, is the one with
#: parallel arcs: the inner hybrid feeds the outer one twice over once its cycle is suppressed.
SHAPE_OF = {"S0": "G2s2-01", "S1": "G2s1-01", "S2": "G2s1-02", "S3": "G2s2-02"}


def test_the_four_level2_generators_are_the_four_shapes():
    seen: Counter = Counter()
    for shape, g, th in _level2_placements():
        seen[shape, g.id] += 1
        assert g.n_sinks == len(th.sinks)
    ids: dict[str, set[str]] = {}
    for shape, gid in seen:
        ids.setdefault(shape, set()).add(gid)
    assert sorted(ids) == sorted(L2.SHAPES), "not every shape occurred -- the test is vacuous"
    assert all(len(v) == 1 for v in ids.values()), ids
    assert {s: v.pop() for s, v in ids.items()} == SHAPE_OF
    assert sorted(SHAPE_OF.values()) == sorted(g.id for g in G.generators(2))   # onto
    assert [g.id for g in G.generators(2) if g.parallel_arcs] == [SHAPE_OF["S2"]]
    for shape in L2.SHAPES:
        net, h1 = shape_fixture(shape)
        assert G.blob_generator(L2.block_of(L2.nontrivial_blocks(net), h1)).id == SHAPE_OF[shape]


# ------------------------------------------------------------------------------ realisations


def realise(g: G.Generator) -> N.Network:
    """``g`` as a binary, time-consistent network, built by hand.

    Every arc is subdivided twice with a pendant leaf on each subdivision vertex, and each sink
    gets a leaf.  Each reticulation's first in-arc carries its donor, on the arc's second
    subdivision vertex, at the reticulation's own time.  Generator cycles triple in length, and
    contracting a hybrid's child edge takes back at most one edge per arc, so girth on R is at
    least 6.  Every vertex has a child whose cluster nothing else reaches, which makes the result
    regular and tree-child.
    """
    depth = [0] * g.n_vertices
    for _ in range(g.n_vertices):                     # longest path from the root, by relaxation
        for u, v in g.arcs:
            depth[v] = max(depth[v], depth[u] + 1)
    indeg = Counter(v for _, v in g.arcs)
    outdeg = Counter(u for u, _ in g.arcs)
    donor_arc: dict[int, int] = {}
    for e, (_, v) in enumerate(g.arcs):
        if indeg[v] == 2:
            donor_arc.setdefault(v, e)

    time: list[float] = []
    parent: list[int] = []

    def new(t: float, p: int = -1) -> int:
        time.append(t)
        parent.append(p)
        return len(time) - 1

    gv = [new(3.0 * depth[v]) for v in range(g.n_vertices)]
    horiz, donor_of, pendant = {}, {}, []
    for e, (u, v) in enumerate(g.arcs):
        tu, tv = time[gv[u]], time[gv[v]]
        if donor_arc.get(v) == e:
            s1 = new(tu + (tv - tu) / 2, gv[u])
            s2 = new(tv, s1)
            horiz[s2], donor_of[gv[v]] = gv[v], s2
        else:
            s1 = new(tu + (tv - tu) / 3, gv[u])
            s2 = new(tu + 2 * (tv - tu) / 3, s1)
            parent[gv[v]] = s2
        pendant += [s1, s2]
    horizon = max(time) + 1.0
    leaf_taxon = {}
    for p in pendant + [gv[v] for v in range(g.n_vertices) if outdeg[v] == 0]:
        leaf_taxon[new(horizon, p)] = len(leaf_taxon)
    children: list[list[int]] = [[] for _ in time]
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
    return N.Network(
        n_taxa=len(leaf_taxon), labels=[f"t{i + 1}" for i in range(len(leaf_taxon))],
        time=time, tree_parent=parent, tree_children=children, leaf_taxon=leaf_taxon,
        horiz_child=horiz, donor_of=donor_of,
        retics=[N.Reticulation(h, d, i, time[h]) for i, (h, d) in enumerate(sorted(donor_of.items()))],
        root=gv[0],
    )


@pytest.mark.parametrize("g", _all(), ids=lambda g: g.id)
def test_every_generator_is_realised_and_classified_back(g):
    net = realise(g)
    blobs = L2.nontrivial_blocks(net)
    assert len(blobs) == 1 and blobs[0].n_retics == g.k == N.level(net) == net.n_retics
    (b,) = blobs
    assert G.blob_generator(b) is g
    assert G.n_sinks(b) == g.n_sinks
    assert L2.regularity(net).ok and reference_regular(net)
    assert L2.is_tree_child(net)
    (r,) = L2.nontrivial_blocks(L2.contract(net))
    assert L2.girth(r) == reference_girth(r) >= L2.MIN_GIRTH
