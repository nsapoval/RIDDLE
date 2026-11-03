"""Multispecies *network* coalescent, in pure Python.

Two jobs:

1. **Validation oracle.**  §4.2's calibration only compares ``r=0`` against msprime, which
   exercises the plain MSC -- not the reticulation handling the study actually depends on.  This
   module simulates the MSNC directly, so PhyloCoalSimulations can be checked at ``r > 0`` against
   an independent implementation (``tests/test_calibration.py``).
2. **Zero-install backend.**  The pipeline runs end to end without Julia, which is what makes the
   smoke grid available before the Julia stage exists.

Model.  Branch lengths are in coalescent units, so inside a population of length ``u`` holding
``k`` lineages the waiting time to the next coalescence is ``Exp(k(k-1)/2)``, truncated at ``u``
(the censored coalescent).  At a hybrid node each entering lineage independently takes the donor
edge with probability ``gamma`` and the tree edge with probability ``1 - gamma`` -- the MSNC with
independent inheritance, i.e. PhyloCoalSimulations' ``inheritancecorrelation=0``.  The population
above the root is infinite, so the remaining lineages are forced to coalesce there.

Clusters are carried as Python ``int`` bitmasks (bit ``t`` = taxon ``t``); scalar draws come from
``random.Random`` rather than numpy, which is roughly an order of magnitude faster per event.
"""

from __future__ import annotations

import random

import numpy as np

from opc.simulate.networks import Network

__all__ = [
    "simulate_gene_trees",
    "simulate_cluster_counts",
    "simulate_topology_counts",
    "postorder_nodes",
]


def postorder_nodes(net: Network) -> list[int]:
    """Reachable nodes with every child before both of its parents.

    A horizontal edge has zero length, so a donor and its hybrid share a time and a time-sort
    would leave their order undefined.  Post-order does not care: the hybrid is a child of the
    donor, so it is always emitted first.
    """
    seen: set[int] = set()
    out: list[int] = []
    stack: list[tuple[int, bool]] = [(net.root, False)]
    while stack:
        v, expanded = stack.pop()
        if expanded:
            out.append(v)
            continue
        if v in seen:
            continue
        seen.add(v)
        stack.append((v, True))
        for c in net.net_children(v):
            if c not in seen:
                stack.append((c, False))
    return out


def _run(lineages: list, avail: float, rng: random.Random, formed: list[int]) -> None:
    """Censored coalescent in one population of length ``avail``.  Mutates ``lineages``.

    A lineage is ``(subtree, mask)``: ``subtree`` is a leaf index or a nested pair, ``mask`` is the
    int bitmask of the taxa below it.
    """
    if avail <= 0.0 or len(lineages) < 2:
        return
    t = 0.0
    while len(lineages) > 1:
        k = len(lineages)
        t += rng.expovariate(k * (k - 1) * 0.5)
        if t > avail:
            return
        i = rng.randrange(k)
        j = rng.randrange(k - 1)
        if j >= i:
            j += 1
        if j < i:
            i, j = j, i
        (sa, ma), (sb, mb) = lineages[i], lineages[j]
        mask = ma | mb
        formed.append(mask)
        lineages[i] = ((sa, sb), mask)
        lineages[j] = lineages[-1]
        lineages.pop()


def _simulate_one(net: Network, order: list[int], rng: random.Random):
    """One gene tree.  Returns ``(root_subtree, formed_masks)``.

    ``formed_masks`` are the leaf sets of the gene tree's internal nodes -- the study's raw
    cluster observations, one list per locus.
    """
    pending: dict[tuple[int, int], list] = {}
    formed: list[int] = []

    for v in order:
        if net.is_leaf(v):
            t = net.leaf_taxon[v]
            lineages = [(t, 1 << t)]
        else:
            lineages = []
            for c in net.net_children(v):
                lineages.extend(pending.pop((c, v)))

        parents = net.net_parents(v)
        if not parents:
            _run(lineages, float("inf"), rng, formed)
            return lineages[0][0], formed

        if len(parents) == 1:
            p = parents[0]
            _run(lineages, net.edge_length(p, v), rng, formed)
            pending[(v, p)] = lineages
        else:
            donor = net.donor_of[v]
            tree_p = net.tree_parent[v]
            gamma = net.gamma_of(v)
            to_donor, to_tree = [], []
            for ln in lineages:
                (to_donor if rng.random() < gamma else to_tree).append(ln)
            _run(to_donor, net.edge_length(donor, v), rng, formed)
            _run(to_tree, net.edge_length(tree_p, v), rng, formed)
            pending[(v, donor)] = to_donor
            pending[(v, tree_p)] = to_tree
    raise RuntimeError("post-order traversal ended without reaching the root")


def _newick(node, labels) -> str:
    if not isinstance(node, tuple):
        return labels[node]
    return f"({_newick(node[0], labels)},{_newick(node[1], labels)})"


def simulate_gene_trees(net: Network, n_loci: int, seed: int) -> list[str]:
    """``n_loci`` Newick gene trees.

    Topology only -- the study consumes clusters, and branch lengths would only be needed for the
    Phase 2 sequence simulation, which uses the Julia backend.
    """
    rng = random.Random(seed)
    order = postorder_nodes(net)
    return [_newick(_simulate_one(net, order, rng)[0], net.labels) + ";" for _ in range(n_loci)]


def simulate_cluster_counts(
    net: Network, n_loci: int, seed: int, checkpoints=None
) -> tuple[dict[int, np.ndarray], list[int], np.ndarray]:
    """Simulate and accumulate cluster support without ever building a Newick string.

    ``checkpoints`` are the nested locus-count prefixes of §5 (simulate the largest ``m`` once and
    subsample prefixes).  Returns ``(counts, checkpoints, n_raw)`` where ``counts`` maps a cluster
    bitmask to its *cumulative* support at each checkpoint and ``n_raw[i]`` is the number of
    distinct clusters seen in the first ``checkpoints[i]`` loci.
    """
    cps = sorted(checkpoints or [n_loci])
    if cps[-1] > n_loci:
        raise ValueError(f"checkpoint {cps[-1]} exceeds n_loci={n_loci}")
    rng = random.Random(seed)
    order = postorder_nodes(net)
    full_mask = (1 << net.n_taxa) - 1
    counts: dict[int, np.ndarray] = {}
    n_raw = np.zeros(len(cps), dtype=np.int64)
    nb = len(cps)

    for locus in range(cps[-1]):
        which = int(np.searchsorted(cps, locus, side="right"))
        _, formed = _simulate_one(net, order, rng)
        for mask in set(formed):
            if mask == full_mask or mask & (mask - 1) == 0:
                continue  # drop the full set and singletons, per spec section 3
            vec = counts.get(mask)
            if vec is None:
                vec = counts[mask] = np.zeros(nb, dtype=np.int32)
                n_raw[which] += 1
            vec[which] += 1

    np.cumsum(n_raw, out=n_raw)
    for vec in counts.values():
        np.cumsum(vec, out=vec)
    return counts, cps, n_raw


def simulate_topology_counts(net: Network, n_loci: int, seed: int) -> dict[str, int]:
    """Frequency of each gene-tree topology.  Used by the calibration tests at small ``n``."""
    from collections import Counter

    rng = random.Random(seed)
    order = postorder_nodes(net)
    return dict(
        Counter(
            _canonical(_simulate_one(net, order, rng)[0], net.labels) for _ in range(n_loci)
        )
    )


def _canonical(node, labels) -> str:
    """Newick with children sorted, so topologically identical trees give identical strings."""
    if not isinstance(node, tuple):
        return labels[node]
    a = _canonical(node[0], labels)
    b = _canonical(node[1], labels)
    return f"({a},{b})" if a < b else f"({b},{a})"
