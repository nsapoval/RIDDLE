"""The level-3 generator and its gates -- regular, exactly level-3, girth >= 5 (A64).

The references are ``tests/test_level2.py``'s, and they share no code with :mod:`level2` or
:mod:`level3`: regularity checked over frozensets against :func:`closure.hasse_covers`, and girth
by edge deletion.  The true tree of blobs is checked against ``tests/test_tob.py``'s bridge
deletion.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter
from functools import lru_cache

import pytest

from opc import l2reg, l3reg
from opc.simulate import generators as G
from opc.simulate import level2 as L2
from opc.simulate import level3 as L3
from opc.simulate import networks as N
from test_level2 import reference_girth, reference_regular, with_r1
from test_tob import _bridge_clusters_by_deletion

LAYOUTS = [
    (L3.Layout3(1, 1, ("s1",)), 1), (L3.Layout3(1, 1, ("s2",)), 2),
    (L3.Layout3(1, 1, ("s3",)), 3), (L3.Layout3(1, 1), 4),
    (L3.Layout3(2, 1, ("s1", "s3")), 5),
]


@lru_cache(maxsize=None)
def _build(layout: L3.Layout3, seed: int):
    return L3.build_level3(40, layout, seed=seed, cu=0.5)


# ------------------------------------------------------------------------------ the generator


@pytest.mark.parametrize("layout,seed", LAYOUTS)
def test_generated_networks_pass_every_gate_against_the_references(layout, seed):
    net, placement = _build(layout, seed)
    verdict = L3.check3(net, layout)
    assert verdict["ok"], verdict
    assert reference_regular(net)
    assert net.n_retics == layout.k and N.level(net) == 3
    for b in L2.nontrivial_blocks(L2.contract(net)):
        assert L2.girth(b) == reference_girth(b) >= L3.MIN_GIRTH
    blobs = L2.nontrivial_blocks(net)
    l3 = [b for b in blobs if b.n_retics == 3]
    assert len(l3) == layout.n_l3 and len(blobs) == layout.n_l3 + layout.n_l1
    if layout.targets:
        assert sorted(G.blob_generator(b).n_sinks for b in l3) == sorted(
            int(t[1]) for t in layout.targets)
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
    again, _ = L3.build_level3(40, layout, seed=seed, cu=0.5)   # deterministic in the seed
    assert N.to_enewick(again) == enw


def test_true_tob_matches_edge_deletion_at_level_3():
    for layout, seed in LAYOUTS:
        net, _ = _build(layout, seed)
        true = l2reg.true_tob_clusters(net)
        assert true, "no bridges at all -- the comparison would be vacuous"
        assert true == _bridge_clusters_by_deletion(net)


def test_check3_fails_what_it_must():
    net, _ = _build(L3.Layout3(1, 1, ("s1",)), 1)
    assert L3.check3(net, L3.Layout3(1, 1, ("s1",)))["ok"]
    assert not L3.check3(net, L3.Layout3(1, 1, ("s2",)))["ok"]      # the wrong stratum
    assert not L3.check3(net, L3.Layout3(1, 2))["ok"]               # the wrong blob count
    two, _ = L2.build_level2(40, L2.Layout(1, 1), seed=6, cu=0.5)
    assert not L3.check3(two, L3.Layout3(1, 1))["ok"]               # level 2 is not level 3


# ------------------------------------------------------------------------------ why no prune


def test_an_intermediate_shortcut_is_cured_by_the_third_reticulation():
    """Why :func:`level3._join` prunes nothing after the intermediate join.

    ``test_level2``'s shortcut fixture: r2's donor splits r1's tree edge, so r1's new tree parent
    reaches r1's donor and its edge is a shortcut.  A third donor on that edge becomes r1's tree
    parent in turn, reaching the donor only through the new horizontal edge -- and some such
    placements give a regular, shortcut-free level-3 network.
    """
    net, _, u1, h1 = with_r1()
    L2.place(net, h1, u1, 3.5)
    assert L2.shortcuts(net) == [h1] and not L2.regularity(net).ok
    assert L2.block_of(L2.nontrivial_blocks(net), h1).n_retics == 2
    times = sorted(set(net.time))
    tried = cured = 0
    for t in [(a + b) / 2 for a, b in zip(times, times[1:])]:
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        if h1 not in alive:
            continue
        for c2 in alive:
            if c2 == h1:
                continue
            trial = net.copy()
            L2.place(trial, h1, c2, t)            # the donor splits the shortcut edge
            tried += 1
            if (N.level(trial) == 3 and not L2.shortcuts(trial) and L2.regularity(trial).ok
                    and reference_regular(trial)):
                cured += 1
    assert tried and cured, (tried, cured)


# ------------------------------------------------------------------------------ the driver


def test_the_design_is_the_one_fixed_with_the_user():
    specs = l3reg.design()
    assert len(specs) == 20 and len({s.key for s in specs}) == 20
    assert all("_l3x" in s.key for s in specs)
    assert Counter((s.n_l3, s.n_l1) for s in specs) == Counter(
        {c: 2 for c in l2reg.COMBOS} | {(2, 2): 4})
    for cu in l2reg.CUS:
        assert Counter(t for s in specs if s.cu == cu for t in s.targets) == {
            "s1": 7, "s2": 7, "s3": 6}
    assert sum(s.k for s in specs) == 160
    assert not {s.est_seeds for s in specs} & {s.est_seeds for s in l2reg.design()}


def _audit_row(net: N.Network) -> dict:
    """What ``audit_networks.jl`` would report, computed from the network itself: enough to
    exercise ``audit_problems``' level without Julia (the real audit runs in ``genetrees``)."""
    lab = net.labels
    below = N.hardwired_clusters(net)
    blobs = L2.nontrivial_blocks(net)
    hybrids = {}
    for k, r in enumerate(net.retics):
        g = net.gamma_of(r.hybrid)
        hybrids[f"H{k + 1}"] = {
            "cluster": sorted(lab[t] for t in below[r.hybrid]),
            "parents": [
                {"cluster": sorted(lab[t] for t in below[r.donor]), "gamma": g, "ismajor": g > 0.5},
                {"cluster": sorted(lab[t] for t in below[net.tree_parent[r.hybrid]]),
                 "gamma": 1 - g, "ismajor": g < 0.5}],
        }
    return {
        "taxa": sorted(lab), "numhybrids": net.n_retics,
        "clusters": [sorted(lab[t] for t in c) for c in N.full_clusters(net)],
        "level": N.level(net), "blob_retics": sorted(b.n_retics for b in blobs),
        "ladder": False, "treechild": L2.is_tree_child(net),
        "girth_N": min(L2.girth(b) for b in blobs),
        "girth_R": min(L2.girth(b) for b in L2.nontrivial_blocks(L2.contract(net))),
        "hybrids": hybrids,
    }


def test_audit_problems_takes_the_level_and_defaults_to_2():
    for fn in (l2reg.audit_problems, l2reg.stage_audit, l2reg.stage_genetrees):
        assert inspect.signature(fn).parameters["level"].default == 2
    net, _ = _build(*LAYOUTS[3])
    row = _audit_row(net)
    assert l2reg.audit_problems(net, row, level=3) == []
    assert l2reg.audit_problems(net, row) == ["getlevel 3"]


@pytest.mark.skipif(not (l2reg.PKG / "manifests" / "audit.json").exists(),
                    reason="l2reg_n200_v1 is not here (the l3reg probe gate cannot skip)")
def test_the_published_level2_networks_rederive_byte_identically():
    """With :mod:`generators`, :mod:`level3` and :mod:`opc.l3reg` loaded, every published level-2
    network rebuilds to the same eNewick, R, pickle, ground-truth JSON and audit verdict."""
    got = l3reg.l2_identity()
    assert got["n_networks"] == 20
    assert got["ok"], got["differ"]
