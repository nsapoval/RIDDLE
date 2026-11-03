"""The level-2 generator and its three gates -- regular, exactly level-2, girth >= 5 (A62).

Fixtures are built by hand on one small time-embedded tree, so every shape and every way a gate
must fire is pinned by a network whose structure can be read off this file.  The references
share no code with :mod:`opc.simulate.level2`:

* regularity -- ``R`` rebuilt here from the hybrid->child contraction, clusters compared as
  frozensets against :func:`opc.solve.closure.hasse_covers`;
* girth -- delete each edge in turn and BFS between its ends;
* shape -- the placement table, derived by hand from where the second reticulation's endpoints
  sit relative to the first one's cycle, and checked on *every* second placement the fixture
  admits.
"""

from __future__ import annotations

import re
from collections import Counter, deque

import pytest

from opc.simulate import level2 as L2
from opc.simulate import networks as N
from opc.solve import closure

# ------------------------------------------------------------------------------ fixture builder


def tree(spec: dict) -> tuple[N.Network, dict[str, int]]:
    """A :class:`Network` from ``{name: (parent, time)}``; leaves: names nothing points at."""
    names = list(spec)
    ids = {nm: i for i, nm in enumerate(names)}
    tree_parent = [-1] * len(names)
    tree_children: list[list[int]] = [[] for _ in names]
    for nm, (par, _t) in spec.items():
        if par is not None:
            tree_parent[ids[nm]] = ids[par]
            tree_children[ids[par]].append(ids[nm])
    leaves = [nm for nm in names if not tree_children[ids[nm]]]
    net = N.Network(
        n_taxa=len(leaves), labels=leaves, time=[spec[nm][1] for nm in names],
        tree_parent=tree_parent, tree_children=tree_children,
        leaf_taxon={ids[nm]: k for k, nm in enumerate(leaves)},
        horiz_child={}, donor_of={}, retics=[],
        root=next(ids[nm] for nm in names if spec[nm][0] is None),
    )
    return net, ids


#: An outgroup clade O, and a clade Z whose two sides carry the first reticulation.
BASE = {
    "R0": (None, 0.0), "Z": ("R0", 1.0), "O": ("R0", 3.0), "o1": ("O", 6.0), "o2": ("O", 6.0),
    "A1": ("Z", 2.0), "a1": ("A1", 6.0), "A2": ("A1", 3.0), "a2": ("A2", 6.0), "a3": ("A2", 6.0),
    "B1": ("Z", 2.0), "b1": ("B1", 6.0), "B2": ("B1", 3.0), "b2": ("B2", 6.0), "b3": ("B2", 6.0),
}


def with_r1(spec=BASE):
    """r1 at t=4: donor u1 on side A (edge into a2), hybrid h1 on side B (edge into b2)."""
    net, ids = tree(spec)
    u1, h1 = L2.place(net, ids["a2"], ids["b2"], 4.0)
    return net, ids, u1, h1


def shape_fixture(shape: str) -> tuple[N.Network, int]:
    """One regular network per level-2 shape, each reached by a row of the placement table."""
    net, ids, u1, h1 = with_r1()
    donor_edge, hybrid_edge, t = {
        "S0": (ids["A2"], ids["b1"], 2.5),   # side A  x  pendant off side B
        "S1": (ids["A2"], ids["B2"], 2.5),   # side A  x  side B
        "S2": (ids["b2"], ids["o1"], 5.0),   # below h1  x  outside the clade
        "S3": (u1, ids["o1"], 3.5),          # side A  x  outside the clade
    }[shape]
    L2.place(net, donor_edge, hybrid_edge, t)
    return net, h1


# ------------------------------------------------------------------------------ references


def reference_regular(net: N.Network) -> bool:
    """R rebuilt without :func:`level2.contract`; then the definition, over frozensets."""
    if any(net.is_hybrid(c) for h in net.donor_of for c in net.tree_children[h]):
        return False
    child_of = {net.tree_children[h][0]: h for h in net.donor_of}
    rep = lambda x: child_of.get(x, x)  # noqa: E731
    below = N.hardwired_clusters(net)
    verts = {rep(v) for v in below}
    edges = [(rep(v), rep(c)) for v in below for c in net.net_children(v)
             if not (v in net.donor_of and child_of.get(c) == v)]
    if len({below[v] for v in verts}) != len(verts):
        return False
    pairs = [(below[a], below[b]) for a, b in edges]
    if len(set(pairs)) != len(pairs):
        return False
    return set(pairs) == closure.hasse_covers({below[v] for v in verts})


def reference_girth(block: N.Block) -> int:
    best = None
    for k, (a, b) in enumerate(block.edges):
        adj: dict[int, list[int]] = {}
        for j, (p, c) in enumerate(block.edges):
            if j != k:
                adj.setdefault(p, []).append(c)
                adj.setdefault(c, []).append(p)
        dist, queue = {a: 0}, deque([a])
        while queue:
            x = queue.popleft()
            for y in adj.get(x, []):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        if b in dist:
            best = dist[b] + 1 if best is None else min(best, dist[b] + 1)
    return best or 0


def old_regular_check(net: N.Network) -> bool:
    """``tests/test_networks.py``'s check: drop equal-cluster edges, compare with the covers."""
    below = N.hardwired_clusters(net)
    got = {(below[v], below[c]) for v in below for c in net.net_children(v)
           if below[v] != below[c]}
    return got == closure.hasse_covers(set(below.values()))


def blob(net: N.Network, h: int) -> N.Block:
    return L2.block_of(L2.nontrivial_blocks(net), h)


def blob_r(net: N.Network, h: int) -> N.Block:
    return L2.block_of(L2.nontrivial_blocks(L2.contract(net)), h)


# ------------------------------------------------------------------------------ the four shapes


@pytest.mark.parametrize("shape", L2.SHAPES)
def test_each_shape_fixture_is_classified_and_passes_every_gate(shape):
    net, h1 = shape_fixture(shape)
    b = blob(net, h1)
    assert b.n_retics == 2 and N.level(net) == 2
    assert L2.blob_shape(b) == shape
    assert L2.regularity(net).ok and reference_regular(net)
    assert L2.girth(blob_r(net, h1)) >= L2.MIN_GIRTH
    verdict = L2.check(net, L2.Layout(1, 0, (shape,)))
    assert verdict["ok"], verdict
    # The classifier's two facts, read off directly: sinks, and where rho sits.
    th = L2.theta(b)
    assert len(th.sinks) == (2 if shape in ("S0", "S3") else 1)
    assert sorted(th.path_lengths) == sorted(len(p) - 1 for p in th.paths)


def test_contraction_shortens_s1_and_s2_only_where_the_hybrid_child_edge_is_on_the_cycle():
    """S1 fixture with girth 5 in N and 4 in R: R is what the girth floor is stated on."""
    spec = dict(BASE, B2=("B1", 3.8))
    net, ids, u1, h1 = with_r1(spec)
    L2.place(net, u1, ids["B2"], 3.5)       # side A (just above u1)  x  side B (above B2)
    assert L2.blob_shape(blob(net, h1)) == "S1"
    assert L2.girth(blob(net, h1)) == reference_girth(blob(net, h1)) == 5
    assert L2.girth(blob_r(net, h1)) == reference_girth(blob_r(net, h1)) == 4
    assert L2.regularity(net).ok                 # regular -- rejected on girth alone
    verdict = L2.check(net, L2.Layout(1, 0, ("S1",)))
    assert verdict["girth_R_min"] == 4 and not verdict["ok"]


# ------------------------------------------------------------------------ gates that must fire


def test_non_regular_level2_network_passes_the_old_check_and_fails_this_one():
    """C(u1) = C(h1) | C(h2) sits inside C(b), so C(z) == C(b).  Level 2, S0, girth 5, no
    shortcut, no ladder -- and not regular.  The old contract-equal-clusters check hides it."""
    net, ids = tree({
        "z": (None, 0.0), "a0": ("z", 4.0), "b": ("z", 1.0), "w": ("b", 2.0), "B2": ("b", 2.0),
        "w1": ("w", 4.0), "w2": ("w", 4.0), "y": ("B2", 4.0), "y2": ("B2", 4.0),
    })
    _, h1 = L2.place(net, ids["a0"], ids["y"], 3.0)
    L2.place(net, ids["w1"], ids["a0"], 3.5)
    below = N.hardwired_clusters(net)
    assert below[ids["z"]] == below[ids["b"]]           # the defect is really there
    assert N.level(net) == 2 and L2.blob_shape(blob(net, h1)) == "S0"
    assert L2.girth(blob_r(net, h1)) == 5
    assert L2.shortcuts(net) == [] and L2.ladders(net) == []
    assert not L2.is_tree_child(net)
    assert old_regular_check(net)                        # the weaker check says "regular"
    reg = L2.regularity(net)
    assert not reg.ok and not reg.injective
    assert not reference_regular(net)


def test_a_later_reticulation_can_make_an_earlier_edge_a_shortcut():
    net, ids, u1, h1 = with_r1()
    # Donor on h1's tree in-edge, hybrid on the side-A edge into u1.  The new hybrid is not
    # under the LCA, so the level-1 rule would pass it -- the *old* hybrid's edge is the shortcut.
    c1, c2 = h1, u1
    assert net.tree_parent[c2] != N._lca(net, c1, c2)  # noqa: SLF001
    L2.place(net, c1, c2, 3.5)
    assert L2.shortcuts(net) == [h1]
    assert not L2.regularity(net).ok and not reference_regular(net)


def test_stacked_hybrids_are_a_ladder_and_R_is_undefined():
    net, ids, u1, h1 = with_r1()
    L2.place(net, ids["o1"], ids["b2"], 5.0)   # hybrid directly below h1
    assert L2.ladders(net) == [h1]
    reg = L2.regularity(net)
    assert not reg.ok and reg.ladders == (h1,)
    with pytest.raises(ValueError, match="in-degree 3"):
        L2.contract(net)


def test_a_regular_four_cycle_fails_only_the_girth_floor():
    net, ids = tree({
        "r": (None, 0.0), "x": ("r", 1.0), "d": ("r", 2.0), "y": ("x", 2.0), "l1": ("x", 5.0),
        "l2": ("y", 5.0), "l3": ("y", 5.0), "l4": ("d", 5.0), "l5": ("d", 5.0),
    })
    _, h = L2.place(net, ids["d"], ids["y"], 1.5)
    assert N.level(net) == 1 and L2.regularity(net).ok
    assert L2.girth(blob(net, h)) == L2.girth(blob_r(net, h)) == reference_girth(blob(net, h)) == 4
    log = L2._Log()  # noqa: SLF001
    assert not L2._complete(net, h, None, log)  # noqa: SLF001
    assert log.counters == {"complete:girth": 1}


def test_three_reticulations_in_one_blob_is_not_level_2():
    net, h1 = shape_fixture("S0")
    ids = tree(BASE)[1]
    L2.place(net, ids["o1"], ids["a3"], 5.0)
    b = blob(net, h1)
    assert b.n_retics == 3 and N.level(net) == 3
    with pytest.raises(ValueError):
        L2.blob_shape(b)
    assert not L2.check(net, L2.Layout(1, 0))["ok"]


# ------------------------------------------------------------------------------ exhaustive


#: (region of the donor edge, region of the hybrid edge) -> shape, for a second reticulation that
#: joins the first one's blob.  Regions are relative to r1 = (u1 on side A, h1 on side B, apex Z):
#: A / B the cycle's sides, A' / B' pendants hanging off an interior side vertex, C below u1's
#: tree child, H below h1, O outside Z's clade.
TABLE = {
    ("A", "B"): "S1", ("A", "A'"): "S3", ("A", "B'"): "S0", ("A", "O"): "S3",
    ("B", "A"): "S1", ("B", "A'"): "S0", ("B", "B'"): "S3", ("B", "O"): "S3",
    ("A'", "A"): "S2", ("A'", "B"): "S1", ("A'", "A'"): "S3", ("A'", "B'"): "S0",
    ("A'", "C"): "S3", ("A'", "H"): "S1", ("A'", "O"): "S3",
    ("B'", "A"): "S1", ("B'", "B"): "S2", ("B'", "A'"): "S0", ("B'", "B'"): "S3",
    ("B'", "C"): "S0", ("B'", "H"): "S1", ("B'", "O"): "S3",
    ("C", "A'"): "S3", ("C", "B'"): "S0", ("C", "H"): "S1", ("C", "O"): "S3",
    ("H", "A'"): "S1", ("H", "B'"): "S1", ("H", "C"): "S1", ("H", "O"): "S2",
    ("O", "A"): "S1", ("O", "B"): "S1", ("O", "A'"): "S3", ("O", "B'"): "S3",
    ("O", "C"): "S3", ("O", "H"): "S2",
}


def test_every_second_placement_agrees_with_every_reference():
    net, ids, u1, h1 = with_r1()
    region = {ids[k]: "A" for k in ("A1", "A2")} | {u1: "A"}
    region |= {ids[k]: "B" for k in ("B1", "B2")} | {h1: "B"}
    region |= {ids["a1"]: "A'", ids["a3"]: "A'", ids["b1"]: "B'", ids["b3"]: "B'"}
    region |= {ids["a2"]: "C", ids["b2"]: "H"}
    region |= {ids[k]: "O" for k in ("Z", "O", "o1", "o2")}
    blocks = L2.nontrivial_blocks(net)
    emap = {e: i for i, b in enumerate(blocks) for e in b.edges}
    target = blocks.index(blob(net, h1))
    times = sorted(set(net.time))
    seen: Counter = Counter()
    for t in [(a + b) / 2 for a, b in zip(times, times[1:])]:
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        for c1 in alive:
            for c2 in alive:
                if c1 == c2:
                    continue
                z = N._lca(net, c1, c2)  # noqa: SLF001
                s1, s2 = L2._sides(net, c1, c2, z)  # noqa: SLF001
                if L2._touched(net, emap, s1 + s2) != {target}:  # noqa: SLF001
                    seen["new_blob"] += 1
                    continue
                trial = net.copy()
                L2.place(trial, c1, c2, t)
                reg = L2.regularity(trial)
                assert reg.ok == reference_regular(trial), (region[c1], region[c2], t)
                if L2.ladders(trial):
                    seen["ladder"] += 1
                    continue
                b = blob(trial, h1)
                assert b.n_retics == 2
                shape = L2.blob_shape(b)
                assert shape == TABLE[(region[c1], region[c2])], (region[c1], region[c2], t)
                assert L2.girth(b) == reference_girth(b)
                assert L2.girth(blob_r(trial, h1)) == reference_girth(blob_r(trial, h1))
                if net.tree_parent[c2] == z:
                    assert L2.shortcuts(trial) and not reg.ok
                    seen["local_shortcut"] += 1
                elif not reg.ok:
                    seen["irregular:" + ("shortcut" if reg.shortcuts else "other")] += 1
                else:
                    seen[shape] += 1
    # Not vacuous: every shape occurs regular, and every way to fail occurs.
    for key in (*L2.SHAPES, "ladder", "local_shortcut", "irregular:shortcut",
                "irregular:other", "new_blob"):
        assert seen[key] > 0, (key, dict(seen))


# ------------------------------------------------------------------------------ the generator


@pytest.mark.parametrize("layout,seed", [
    (L2.Layout(1, 1, ("S0",)), 1), (L2.Layout(1, 1, ("S1",)), 2), (L2.Layout(1, 1, ("S2",)), 3),
    (L2.Layout(1, 1, ("S3",)), 4), (L2.Layout(2, 2, ("S0", "S2")), 5), (L2.Layout(2, 1), 6),
])
def test_generated_networks_pass_every_gate_against_the_references(layout, seed):
    net, placement = L2.build_level2(30, layout, seed=seed, cu=0.5)
    verdict = L2.check(net, layout)
    assert verdict["ok"], verdict
    assert reference_regular(net)
    assert net.n_retics == layout.k and N.level(net) == 2
    for b in L2.nontrivial_blocks(L2.contract(net)):
        assert L2.girth(b) == reference_girth(b) >= L2.MIN_GIRTH
    # Per-reticulation gammas, each on its donor (minor) edge, and in the eNewick.
    tags = {ret.hybrid: k + 1 for k, ret in enumerate(net.retics)}
    enw = N.to_enewick(net)
    found: dict[int, list[float]] = {}
    for tag, g in re.findall(r"#H(\d+):[0-9.eE+-]+::([0-9.eE+-]+)", enw):
        found.setdefault(int(tag), []).append(float(g))
    for h, tag in tags.items():
        g = net.gamma_of(h)
        assert 0.25 <= g <= 0.5
        assert sorted(found[tag]) == pytest.approx(sorted([g, 1.0 - g]), abs=1e-9)
    assert placement["counters"], "the generator rejected nothing -- the gates never ran"
    # Deterministic in the seed.
    again, _ = L2.build_level2(30, layout, seed=seed, cu=0.5)
    assert N.to_enewick(again) == enw


def test_per_reticulation_gammas_are_validated_and_cleared():
    net, _ = L2.build_level2(20, L2.Layout(1, 0), seed=9, cu=0.5)
    hs = list(net.donor_of)
    with pytest.raises(ValueError):
        N.with_gammas(net, {hs[0]: 0.3})
    with pytest.raises(ValueError):
        N.with_gammas(net, {h: 1.0 for h in hs})
    assert net.copy().gammas == net.gammas
    plain = N.with_gamma(net, 0.3)
    assert plain.gammas is None and all(plain.gamma_of(h) == 0.3 for h in hs)
    first = net.retics[0].hybrid
    assert N.with_reticulations(net, 1).gammas == {first: net.gammas[first]}
