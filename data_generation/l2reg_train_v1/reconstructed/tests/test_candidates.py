"""I4: the tau = 0 ground set, and the structure an ILP would be built over.

Thin by design -- every kernel underneath is already cross-validated in ``test_overlap.py`` and
``test_relations.py``.  What is checked here is the reading the new study depends on: that tau = 0
really is "support >= 1", that the decomposition loses nothing, and that violations never cross a
component, which is the whole reason the constraint count is tractable.
"""

from __future__ import annotations

import itertools

import numpy as np

from opc import bitset, grid
from opc.clusters import extract, systems
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N
from opc.solve import candidates


def _ground_set(n=14, r=2, cu=0.5, gamma=0.5, seed=0, m=200):
    net = N.with_gamma(N.scaled(N.with_reticulations(N.build_base(n, 3, seed=seed), r), cu), gamma)
    counter = extract.SupportCounter(n, [m])
    idx = {lab: i for i, lab in enumerate(net.labels)}
    buf: list[int] = []
    for nwk in oracle.simulate_gene_trees(net, m, seed=seed + 1):
        buf.clear()
        counter.add_locus(extract.clusters_from_newick(nwk, idx, buf))
    masks, support, _ = counter.finalize(min_support=1)
    words = np.stack([
        np.frombuffer(int(x).to_bytes(bitset.n_words(n) * 8, "little"), dtype=np.uint64)
        for x in masks
    ])
    return net, candidates.build(words, support[:, 0], n, m=m, tau=0)


def test_tau_zero_keeps_every_cluster_any_gene_tree_induced():
    net, gs = _ground_set()
    assert gs.n_clusters == gs.words.shape[0]
    assert (gs.support >= 1).all()
    # And it is strictly more than tau = 2 would keep -- otherwise A29's point is moot here.
    assert (gs.support == 1).sum() > 0


def test_the_decomposition_partitions_the_ground_set_and_loses_nothing():
    _, gs = _ground_set()
    seen = np.concatenate(
        [gs.decomposition.kernel] + [c.vertices for c in gs.components]
    ) if gs.components else gs.decomposition.kernel
    assert np.array_equal(np.sort(seen), np.arange(gs.n_clusters))


def test_no_overlapping_pair_and_no_violation_crosses_a_component():
    """The property that makes the constraint count viable.

    Clusters in different components do not overlap, so they are automatically compatible.  Checked
    against the slow frozenset semantics rather than against the fast kernel that built it.
    """
    _, gs = _ground_set()
    label = np.full(gs.n_clusters, -1, dtype=int)
    for ci, comp in enumerate(gs.components):
        label[comp.vertices] = ci
    sets = [frozenset(np.flatnonzero(bitset.to_dense(gs.words[i:i + 1], gs.n_taxa)[0]).tolist())
            for i in range(gs.n_clusters)]
    crossings = 0
    for i, j in itertools.combinations(range(gs.n_clusters), 2):
        if systems.overlaps(sets[i], sets[j]):
            assert label[i] == label[j] != -1, "an overlapping pair crossed a component"
            crossings += 1
    assert crossings > 0, "no overlapping pair at all, so this proved nothing"


def test_the_reported_model_size_is_what_the_decomposition_actually_holds():
    _, gs = _ground_set()
    s = gs.sizes()
    assert s["n_clusters"] == gs.n_clusters
    assert s["n_l_pairs"] == sum(c.n_edges for c in gs.components)
    assert s["kernel"] + sum(c.n_vertices for c in gs.components) == gs.n_clusters
    # The decomposition is a constraint-count reduction, not only a solve-time one.
    assert s["n_pairs_if_undecomposed"] > s["n_l_pairs"]


def test_it_reads_the_grids_own_cluster_tables():
    """End to end against a real S50 artifact if one is on disk, else skipped."""
    from pathlib import Path

    import pytest

    repo = Path(__file__).resolve().parents[1]
    cells = grid.enumerate_cells(**grid.S50)
    path = repo / "results" / "s50" / "clusters_julia" / f"{cells[0].key}.parquet"
    if not path.exists():
        pytest.skip("S50 cluster tables not present")
    gs = candidates.from_cluster_table(path, m=250)
    assert gs.m == 250 and gs.n_taxa == 50
    assert gs.n_clusters > 0 and (gs.support >= 1).all()
