"""MDC weights (WS1-1), against the definition rather than against themselves.

``opc.solve.instance`` never walks a gene tree.  It computes ``sum_T alpha(A,T)`` as a subset-sum
over the cluster table, using an identity that only holds for rooted binary trees.  So the tests
that matter compare it to a *maximal-node counter* -- the definition, implemented separately here
with its own Newick parser and explicit parent links, so agreement is evidence rather than a
tautology.  This mirrors how ``test_extract.py`` validates cluster extraction against dendropy.
"""

from __future__ import annotations

import random
import re

import numpy as np
import pytest

from opc import bitset
from opc.clusters import extract
from opc.solve import instance

_STRIP = re.compile(r":[^,);]*|(?<=\))[^,);]*")
_SPLIT = re.compile(r"([(),])")


def _nodes(newick: str, taxon_index: dict[str, int]):
    """``(masks, parent)`` for **every** node, leaves included.  Oracle only."""
    masks: list[int] = []
    parent: list[int] = []
    stack: list[list[int]] = []
    cur: list[int] = []
    for tok in _SPLIT.split(_STRIP.sub("", newick)):
        if not tok:
            continue
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            m = 0
            for c in cur:
                m |= masks[c]
            masks.append(m)
            parent.append(-1)
            me = len(masks) - 1
            for c in cur:
                parent[c] = me
            cur = stack.pop()
            cur.append(me)
        elif tok == ",":
            continue
        else:
            name = tok.strip().rstrip(";").strip()
            if name:
                masks.append(1 << taxon_index[name])
                parent.append(-1)
                cur.append(len(masks) - 1)
    return masks, parent


def _alpha_by_definition(a_mask: int, masks, parent) -> int:
    """``k(A,T) - 1`` where ``k`` counts maximal nodes ``v`` with ``c(v) <= A``."""
    k = 0
    for v, m in enumerate(masks):
        if m & ~a_mask:
            continue
        p = parent[v]
        if p == -1 or (masks[p] & ~a_mask):
            k += 1
    return k - 1


def _sample(n_taxa: int, n_loci: int, seed: int):
    from dendropy.simulate import treesim

    rng = np.random.default_rng(seed)
    labels = [f"t{i + 1}" for i in range(n_taxa)]
    idx = {lab: i for i, lab in enumerate(labels)}
    trees = []
    while len(trees) < n_loci:
        t = treesim.birth_death_tree(
            birth_rate=1.0, death_rate=0.0, num_extant_tips=n_taxa,
            rng=random.Random(int(rng.integers(1, 2 ** 31))),
        )
        for i, leaf in enumerate(t.leaf_node_iter()):
            leaf.taxon.label = labels[i]
        trees.append(t.as_string(schema="newick", suppress_rooting=True).strip())
    return labels, idx, trees


def _counter(trees, idx, n_taxa, checkpoints):
    c = extract.SupportCounter(n_taxa, checkpoints)
    buf: list[int] = []
    for nwk in trees:
        buf.clear()
        c.add_locus(extract.clusters_from_newick(nwk, idx, buf))
    return c


# ------------------------------------------------------------------------- alpha is the definition


@pytest.mark.parametrize("n_taxa,n_loci", [(6, 40), (8, 60), (12, 50)])
def test_alpha_sums_match_maximal_node_counting(n_taxa, n_loci):
    """The identity the whole module rests on, at every ``m`` checkpoint.

    ``sum_T alpha = m*(|A|-1) - sum_B support(B)*[B <= A]`` is a telescoping of the maximal-node
    count.  If it is off by even one the weights are wrong everywhere and ``k=0`` stops
    reproducing Than-Nakhleh, which is B1's T1.
    """
    labels, idx, trees = _sample(n_taxa, n_loci, seed=11 + n_taxa)
    checkpoints = (n_loci // 2, n_loci)
    inst = instance.build_instance(_counter(trees, idx, n_taxa, checkpoints), n_taxa)
    parsed = [_nodes(nwk, idx) for nwk in trees]

    for bi, m in enumerate(checkpoints):
        for a in range(inst.words.shape[0]):
            a_mask = int.from_bytes(inst.words[a].tobytes(), "little")
            expected = sum(_alpha_by_definition(a_mask, *parsed[t]) for t in range(m))
            assert int(inst.alpha_sum[a, bi]) == expected


def test_alpha_is_zero_exactly_when_the_cluster_is_a_clade():
    """``alpha(A,T) == 0`` iff ``A`` is a cluster of ``T`` -- so support is *not* the weight.

    Worth pinning because it is the tempting shortcut: ``support(A)`` counts the trees where
    ``alpha`` is zero, but says nothing about how badly the others miss.
    """
    n_taxa, n_loci = 8, 40
    labels, idx, trees = _sample(n_taxa, n_loci, seed=404)
    counter = _counter(trees, idx, n_taxa, (n_loci,))
    inst = instance.build_instance(counter, n_taxa)
    parsed = [_nodes(nwk, idx) for nwk in trees]
    for a in range(inst.words.shape[0]):
        a_mask = int.from_bytes(inst.words[a].tobytes(), "little")
        zeros = sum(_alpha_by_definition(a_mask, *parsed[t]) == 0 for t in range(n_loci))
        assert zeros == int(inst.support[a, -1])


def test_subset_support_sums_matches_bruteforce_and_is_tile_invariant():
    n_taxa = 10
    rng = np.random.default_rng(3)
    src = [frozenset(rng.choice(n_taxa, size=int(rng.integers(2, n_taxa)), replace=False).tolist())
           for _ in range(40)]
    src = list(dict.fromkeys(src))
    words = np.stack([bitset.from_indices(c, n_taxa) for c in src])
    support = rng.integers(1, 50, size=(len(src), 2)).astype(np.int32)

    got = instance.subset_support_sums(words, support, words, n_taxa)
    expected = np.zeros_like(got)
    for a, ca in enumerate(src):
        for b, cb in enumerate(src):
            if cb <= ca:
                expected[a] += support[b]
    assert (got == expected).all()
    # Tiling must not change the answer -- the float32 accumulation is exact per tile.
    assert (instance.subset_support_sums(words, support, words, n_taxa, tile=3) == got).all()


# ----------------------------------------------------------------------------------- the shift


def test_than_nakhleh_shift_keeps_every_weight_positive_and_loci_does_not():
    """SPEC-AMENDMENTS: the handoff reads Than-Nakhleh's ``m`` as the locus count.

    It is ``max_A sum_T alpha``.  Under the locus reading most of the ground set has ``w <= 0``,
    which makes those clusters unselectable and breaks both ``k=0`` and the rule that isolated
    clusters are accepted unconditionally.
    """
    n_taxa, n_loci = 10, 60
    labels, idx, trees = _sample(n_taxa, n_loci, seed=77)
    inst = instance.build_instance(_counter(trees, idx, n_taxa, (n_loci,)), n_taxa)
    a = inst.alpha_sum[:, -1]

    w = instance.weights(a, n_loci, "than-nakhleh")
    assert w.min() >= 1, "the paper's shift is exactly what makes every weight positive"
    assert w.max() == int(a.max()) + 1 - int(a.min())

    w_loci = instance.weights(a, n_loci, "loci")
    assert (w_loci <= 0).any(), "the locus reading should sink part of the ground set"
    assert (w_loci == n_loci + 1 - a).all(), "and it must still be the spec's formula"

    assert (instance.weights(a, n_loci, "none") == -a).all()
    with pytest.raises(ValueError):
        instance.weights(a, n_loci, "nonsense")


def test_weight_ordering_is_shift_invariant():
    """Every convention is a translation, so the *ranking* of clusters never changes.

    Which is why the shift is a solve-time flag and not a stored column: it changes the LP bound
    and the solve time, not the preference order.
    """
    a = np.array([0, 3, 7, 7, 12], dtype=np.int64)
    orders = [np.argsort(instance.weights(a, 10, c), kind="stable")
              for c in instance.SHIFT_CONVENTIONS]
    for o in orders[1:]:
        assert (o == orders[0]).all()


# ---------------------------------------------------------------------------------- the schemes


def test_a_scheme_changes_the_ranking_where_a_shift_cannot():
    """The distinction the whole weight axis rests on.

    Every shift convention is a translation, so no shift can reorder two clusters; a scheme is a
    different function of the same instance and must be able to.  If ``mdc`` and ``support`` ever
    agreed on the order, comparing them would be measuring nothing.
    """
    n_taxa, n_loci = 9, 50
    labels, idx, trees = _sample(n_taxa, n_loci, seed=909)
    inst = instance.build_instance(_counter(trees, idx, n_taxa, (n_loci,)), n_taxa)
    a, s, sz = inst.alpha_sum[:, -1], inst.support[:, -1], inst.sizes

    ranks = {
        scheme: np.argsort(
            instance.weight_vector(a, s, sz, n_loci, scheme=scheme), kind="stable"
        )
        for scheme in instance.WEIGHT_SCHEMES
    }
    assert not (ranks["mdc"] == ranks["support"]).all()
    assert not (ranks["mdc"] == ranks["mdc-per-taxon"]).all()

    with pytest.raises(ValueError):
        instance.weight_vector(a, s, sz, n_loci, scheme="nonsense")


def test_per_taxon_is_plain_mdc_when_every_cluster_has_the_same_size():
    """It divides out a size scaling, so it must be the identity when there is none."""
    a = np.array([5, 11, 2, 30], dtype=np.int64)
    s = np.array([9, 3, 40, 1], dtype=np.int32)
    same = np.array([4, 4, 4, 4], dtype=np.int64)
    mdc = instance.weight_vector(a, s, same, 50, scheme="mdc")
    per = instance.weight_vector(a, s, same, 50, scheme="mdc-per-taxon")
    assert np.allclose(per * 3.0, mdc)

    grown = instance.weight_vector(a, s, np.array([2, 4, 4, 4]), 50, scheme="mdc-per-taxon")
    assert not np.allclose(grown, per), "and not the identity when the sizes differ"


def test_support_carries_more_contrast_than_mdc_on_a_real_ground_set():
    """A21's diagnosis in one assertion: MDC is low-contrast, and that is what stalls the ILP.

    The measured numbers on the grid's largest envelope component are CV 0.203 (MDC) against
    1.388 (support).  The fixture here is far smaller, so only the ordering is asserted.
    """
    n_taxa, n_loci = 10, 60
    labels, idx, trees = _sample(n_taxa, n_loci, seed=1234)
    inst = instance.build_instance(_counter(trees, idx, n_taxa, (n_loci,)), n_taxa)
    a, s, sz = inst.alpha_sum[:, -1], inst.support[:, -1], inst.sizes
    assert a.size > 20, "fixture must have a ground set worth measuring contrast on"

    cv = {
        scheme: instance.contrast(instance.weight_vector(a, s, sz, n_loci, scheme=scheme))
        for scheme in instance.WEIGHT_SCHEMES
    }
    assert cv["support"] > cv["mdc"]
    assert instance.contrast(np.zeros(0)) != instance.contrast(np.zeros(0))  # nan, not a crash


# ------------------------------------------------------------------------------ the binary guard


def test_a_non_binary_locus_is_refused_rather_than_silently_miscounted():
    """The identity counts internal *nodes*; a polytomy has fewer than ``n - 1`` of them.

    ``SupportCounter`` also de-duplicates within a locus, so a unifurcation would be invisible.
    Both are caught by the same node-count guard.
    """
    idx = {f"t{i + 1}": i for i in range(4)}
    counter = extract.SupportCounter(4, (2,))
    buf: list[int] = []
    counter.add_locus(extract.clusters_from_newick("((t1,t2),(t3,t4));", idx, buf))
    assert counter.n_nonbinary_loci == 0
    buf.clear()
    counter.add_locus(extract.clusters_from_newick("(t1,t2,(t3,t4));", idx, buf))
    assert counter.n_nonbinary_loci == 1

    with pytest.raises(ValueError, match="not rooted binary"):
        instance.build_instance(counter, 4)
    inst = instance.build_instance(counter, 4, require_binary=False)
    assert inst.n_nonbinary_loci == 1


def test_support_one_clusters_are_sources_but_never_targets():
    """A support-1 cluster cannot enter any ``C_tau`` (``tau >= 2``) but is a real gene-tree node.

    Excluding it from the sources would bias every weight by an amount that varies with ``n``.
    """
    n_taxa, n_loci = 9, 30
    labels, idx, trees = _sample(n_taxa, n_loci, seed=5)
    counter = _counter(trees, idx, n_taxa, (n_loci,))
    inst = instance.build_instance(counter, n_taxa)
    assert inst.support[:, -1].min() >= 2
    assert inst.n_source_clusters >= inst.words.shape[0]
    all_masks, all_support, _ = counter.finalize(min_support=1)
    assert inst.n_source_clusters == len(all_masks)
    assert (all_support[:, -1] == 1).any(), "fixture should contain singleton-support clusters"
