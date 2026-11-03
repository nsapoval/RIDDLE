"""Cluster extraction must match an independent parser exactly, and be fast enough.

The reference is dendropy's ``encode_bipartitions().leafset_bitmask`` -- the same idiom used in
``/shared/ns58/split-summary-rs/split-summary.py`` -- which parses Newick with completely
different code, so agreement is real evidence rather than a tautology.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from opc.clusters import extract
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N


def _dendropy_clusters(newick: str, taxon_index: dict[str, int]) -> set[int]:
    import dendropy

    tree = dendropy.Tree.get(data=newick + "\n", schema="newick", rooting="force-rooted")
    tree.encode_bipartitions()
    ns = tree.taxon_namespace
    # dendropy's bitmask uses its own taxon ordering; remap onto ours.
    remap = {1 << i: 1 << taxon_index[t.label] for i, t in enumerate(ns)}
    out = set()
    full = (1 << len(taxon_index)) - 1
    for nd in tree.internal_nodes():
        m = nd.leafset_bitmask
        ours = 0
        for i in range(len(ns)):
            if m >> i & 1:
                ours |= remap[1 << i]
        if ours != full and ours & (ours - 1):
            out.add(ours)
    return out


@pytest.mark.parametrize("n_taxa,r", [(8, 0), (12, 2), (25, 3)])
def test_matches_dendropy(n_taxa, r):
    net = N.with_gamma(N.scaled(N.build_base(n_taxa, r, seed=5), 0.5), 0.3)
    idx = {lab: i for i, lab in enumerate(net.labels)}
    trees = oracle.simulate_gene_trees(net, 60, seed=3)
    for nwk in trees:
        mine = set(extract.clusters_from_newick(nwk, idx, []))
        mine = {m for m in mine if m != (1 << n_taxa) - 1 and m & (m - 1)}
        assert mine == _dendropy_clusters(nwk, idx)


def test_parses_branch_lengths_and_internal_labels():
    """PhyloCoalSimulations writes lengths, and can write internal node labels."""
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    plain = "((A,B),(C,D));"
    lengths = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35);"
    labelled = "((A:0.1,B:0.2)i1:0.3,(C:0.15,D:0.25)i2:0.35)i0;"
    support = "((A:0.1,B:0.2)100:0.3,(C:0.15,D:0.25)95:0.35);"
    expected = {0b0011, 0b1100}
    for s in (plain, lengths, labelled, support):
        got = set(extract.clusters_from_newick(s, idx, []))
        got = {m for m in got if m != 0b1111 and m & (m - 1)}
        assert got == expected, s


def test_rejects_unbalanced():
    with pytest.raises(ValueError):
        extract.clusters_from_newick("((A,B),C;", {"A": 0, "B": 1, "C": 2}, [])


def test_support_checkpoints_are_nested_prefixes():
    """Support at checkpoint m must equal a fresh count over the first m loci."""
    net = N.with_gamma(N.scaled(N.build_base(12, 1, seed=9), 0.5), 0.3)
    idx = {lab: i for i, lab in enumerate(net.labels)}
    trees = oracle.simulate_gene_trees(net, 200, seed=11)

    cps = [50, 100, 200]
    counter = extract.SupportCounter(12, cps)
    for nwk in trees:
        counter.add_locus(extract.clusters_from_newick(nwk, idx, []))
    masks, sup, n_raw = counter.finalize(min_support=1)
    lookup = {int(m): sup[i] for i, m in enumerate(masks)}

    for b, m_cut in enumerate(cps):
        direct: dict[int, int] = {}
        for nwk in trees[:m_cut]:
            for mask in set(extract.clusters_from_newick(nwk, idx, [])):
                if mask != (1 << 12) - 1 and mask & (mask - 1):
                    direct[mask] = direct.get(mask, 0) + 1
        assert {k: int(lookup[k][b]) for k in direct} == direct
        assert int(n_raw[b]) == len(direct)


def test_min_support_pruning_is_lossless_for_tau_ge_2():
    net = N.with_gamma(N.scaled(N.build_base(15, 2, seed=4), 0.5), 0.3)
    idx = {lab: i for i, lab in enumerate(net.labels)}
    counter = extract.SupportCounter(15, [100, 300])
    for nwk in oracle.simulate_gene_trees(net, 300, seed=2):
        counter.add_locus(extract.clusters_from_newick(nwk, idx, []))
    all_masks, all_sup, _ = counter.finalize(min_support=1)
    kept_masks, kept_sup, _ = counter.finalize(min_support=2)
    dropped = set(map(int, all_masks)) - set(map(int, kept_masks))
    lookup = {int(m): all_sup[i] for i, m in enumerate(all_masks)}
    # Everything dropped has support < 2 at *every* checkpoint, so no tau>=2 cell can miss it.
    for m in dropped:
        assert lookup[m].max() < 2


def test_roundtrip_parquet(tmp_path):
    net = N.with_gamma(N.scaled(N.build_base(70, 2, seed=6), 0.5), 0.3)  # > 64 taxa: 2 words
    idx = {lab: i for i, lab in enumerate(net.labels)}
    counter = extract.SupportCounter(70, [50, 150])
    for nwk in oracle.simulate_gene_trees(net, 150, seed=8):
        counter.add_locus(extract.clusters_from_newick(nwk, idx, []))
    masks, sup, n_raw = counter.finalize()
    path = tmp_path / "clusters.parquet"
    extract.write_cluster_table(path, masks, sup, [50, 150], 70, n_raw, cell="test")

    words, sup2, meta = extract.read_cluster_table(path)
    assert meta["n_taxa"] == 70 and meta["checkpoints"] == [50, 150]
    assert words.shape == (len(masks), 2)
    assert np.array_equal(sup2, sup)
    back = [int.from_bytes(words[i].tobytes(), "little") for i in range(words.shape[0])]
    assert back == [int(m) for m in masks]


def test_throughput_is_adequate():
    """~20M trees over the full grid; anything above ~2k trees/s is a rounding error in cost."""
    net = N.with_gamma(N.scaled(N.build_base(100, 5, seed=1), 0.5), 0.3)
    idx = {lab: i for i, lab in enumerate(net.labels)}
    trees = oracle.simulate_gene_trees(net, 300, seed=1)
    buf: list[int] = []
    t0 = time.perf_counter()
    for nwk in trees:
        buf.clear()
        extract.clusters_from_newick(nwk, idx, buf)
    rate = len(trees) / (time.perf_counter() - t0)
    assert rate > 1500, f"only {rate:.0f} trees/s at n=100"
