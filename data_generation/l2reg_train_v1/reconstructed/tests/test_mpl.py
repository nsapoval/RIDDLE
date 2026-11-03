"""I1: the rooted-triple frequency tensor, and P(t | Psi, gamma).

The plan calls I1 "non-negotiable -- this number is the currency of every downstream claim", so it
is validated against **two** independent implementations, per this repo's convention: PhyloNet's
``CalGTProb`` and the MSNC simulator in :mod:`opc.simulate.msnc_oracle`.  A formula error the two
happened to share would otherwise pass unnoticed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opc.mpl import msnc3, phylonet, triples
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N

requires_phylonet = pytest.mark.skipif(
    not phylonet.available(), reason="PhyloNet.jar or java not installed"
)


def _labels(n: int) -> dict[str, int]:
    return {f"t{i + 1}": i for i in range(n)}


# ------------------------------------------------------------------------------ rho(t, G)


def test_a_balanced_quartet_induces_exactly_the_four_triples_it_should():
    acc = triples.TripleCounts.empty(4)
    triples.count_tree("((t1,t2),(t3,t4));", _labels(4), acc)
    c = acc.count
    assert (c[0, 1, 2], c[0, 1, 3], c[2, 3, 0], c[2, 3, 1]) == (1, 1, 1, 1)
    assert c[0, 2, 1] == 0 and c[0, 3, 1] == 0
    assert c[1, 0, 2] == 1, "the tensor must be symmetric in its first two axes"
    _, obs = acc.as_rows()
    assert (obs.sum(axis=1) == 1).all(), "a binary tree resolves every triple exactly once"


def test_a_caterpillar_puts_every_triple_in_the_slot_the_convention_says():
    acc = triples.TripleCounts.empty(4)
    triples.count_tree("(((t1,t2),t3),t4);", _labels(4), acc)
    _, obs = acc.as_rows()
    # (a<b<c) rows are 123, 124, 134, 234 and the caterpillar resolves each as ab|c.
    assert obs.tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_a_multifurcation_leaves_its_triple_unresolved_rather_than_guessing():
    acc = triples.TripleCounts.empty(4)
    triples.count_tree("(t1,t2,t3,t4);", _labels(4), acc)
    _, obs = acc.as_rows()
    assert obs.sum() == 0, "a star tree resolves nothing"
    assert acc.n_nonbinary_loci == 1


def test_every_triple_of_a_simulated_locus_set_is_resolved_by_every_locus():
    net = N.scaled(N.build_base(12, 1, seed=5), 1.0)
    acc = triples.TripleCounts.empty(12)
    for nwk in oracle.simulate_gene_trees(net, 50, seed=3):
        triples.count_tree(nwk, {lab: i for i, lab in enumerate(net.labels)}, acc)
    _, obs = acc.as_rows()
    assert (obs.sum(axis=1) == 50).all()
    assert acc.n_nonbinary_loci == 0


# --------------------------------------------------------------------- P(t | Psi, gamma)


def test_on_a_species_tree_the_exact_probability_is_the_textbook_msc_formula():
    """``P(match) = 1 - (2/3) e^{-x}`` for internal branch ``x``; the other two share the rest."""
    net = N.scaled(N.build_base(3, 0, seed=11), 1.0)
    leaf = {t: v for v, t in net.leaf_taxon.items()}
    x = None
    for a, b in ((0, 1), (0, 2), (1, 2)):
        (c,) = (t for t in (0, 1, 2) if t not in (a, b))
        pair = N._lca(net, leaf[a], leaf[b])  # noqa: SLF001
        top = N._lca(net, pair, leaf[c])  # noqa: SLF001
        if pair != top:
            x = net.time[pair] - net.time[top]
            match = msnc3.TOPOLOGIES.index((a, b))
    assert x is not None
    got = msnc3.triple_probabilities(net, (0, 1, 2))
    want = np.full(3, math.exp(-x) / 3)
    want[match] = 1 - 2 / 3 * math.exp(-x)
    assert np.allclose(got, want, atol=1e-12)


def _two_taxa_under_one_hybrid(gamma: float) -> N.Network:
    """``z -> D`` and ``z -> H``; ``D -> t3`` and ``D ~~> H``; ``H -> (t1, t2)``.

    Built by hand because the generator will not reliably produce it at n = 3, and it is the
    configuration that separates the MSNC from the displayed-tree mixture: **two** of the triple's
    lineages enter one hybrid, so they can fail to coalesce below it and then take different sides
    of the cycle.  It also happens to be the smallest fixture that exercises the zero-length
    horizontal edge and a post-order that interleaves leaves with other subtrees' internal nodes --
    the two places the DP has been wrong.
    """
    return N.Network(
        n_taxa=3, labels=["t1", "t2", "t3"],
        time=[0.0, 0.4, 0.4, 0.9, 1.5, 1.5, 1.5],
        tree_parent=[-1, 0, 0, 2, 3, 3, 1],
        tree_children=[[1, 2], [6], [3], [4, 5], [], [], []],
        leaf_taxon={4: 0, 5: 1, 6: 2},
        horiz_child={1: 2}, donor_of={2: 1},
        retics=[N.Reticulation(hybrid=2, donor=1, order=0, time=0.4)],
        root=0, gamma=gamma,
    )


def test_the_exact_probability_matches_the_msnc_simulator_when_two_lineages_share_a_hybrid():
    net = _two_taxa_under_one_hybrid(0.5)
    m = 400_000
    counts = oracle.simulate_topology_counts(net, m, seed=7)
    acc = triples.TripleCounts.empty(3)
    for nwk, c in counts.items():
        one = triples.TripleCounts.empty(3)
        triples.count_tree(nwk, _labels(3), one)
        acc.count += one.count * c
    _, obs = acc.as_rows()
    empirical = obs[0] / m
    exact = msnc3.triple_probabilities(net, (0, 1, 2))
    assert np.abs(exact - empirical).max() < 4 * math.sqrt(0.25 / m)
    # Not a degenerate fixture: an unresolved DP returns the uniform vector, and a DP that loses
    # the horizontal edge returns the underlying tree's answer.  Neither is this.
    assert np.abs(exact - 1 / 3).max() > 0.2


def test_the_displayed_tree_mixture_is_exact_only_while_one_lineage_reaches_each_hybrid():
    """The plan specifies the mixture; this is the measurement of what that costs.

    With a single taxon below the hybrid the mixture is the MSNC.  With two below it, the mixture
    cannot represent the state where they take different sides of the cycle, so it understates ILS
    for that pair -- worst at ``gamma = 0.5``, which is the stratum every pre-registered threshold
    in the plan is stated at.
    """
    net = N.with_gamma(N.scaled(N.with_reticulations(N.build_base(8, 3, seed=0), 2), 0.5), 0.5)
    below = N.hardwired_clusters(net)
    hybrid_clusters = [below[ret.hybrid] for ret in net.retics]
    single, shared = [], []
    for tri in triples.triple_index(8):
        s = set(tri.tolist())
        occ = max((len(s & set(h)) for h in hybrid_clusters), default=0)
        gap = np.abs(
            msnc3.triple_probabilities(net, tuple(tri))
            - msnc3.triple_probabilities_mixture(net, tuple(tri))
        ).max()
        (shared if occ >= 2 else single).append(gap)
    assert single and shared, "fixture did not produce both cases"
    assert max(single) < 1e-12, "the mixture should be exact when one lineage reaches each hybrid"
    assert max(shared) > 1e-3, "the mixture should be measurably wrong when two do"

    # And the size of it, on the fixture where the effect is cleanest.
    hand = _two_taxa_under_one_hybrid(0.5)
    gap = np.abs(
        msnc3.triple_probabilities(hand, (0, 1, 2))
        - msnc3.triple_probabilities_mixture(hand, (0, 1, 2))
    ).max()
    assert gap > 0.05, f"expected a large gap at gamma=0.5, got {gap:.4f}"


def test_the_donor_and_its_hybrid_share_a_time_so_the_order_they_are_processed_in_matters():
    """A regression test for a bug that produced a *plausible* wrong answer.

    The horizontal edge has length zero, so donor and hybrid are simultaneous.  Process the donor
    first and every lineage that chose the donor side is stranded on an edge whose head is already
    done; the reticulation silently stops contributing and the answer collapses towards the
    underlying tree's.  Nothing raises.  So: the network must not score like its own tree.
    """
    net = _two_taxa_under_one_hybrid(0.5)
    tree = msnc3.resolve_one_reticulation(net, 0, take_donor=False)
    assert np.abs(
        msnc3.triple_probabilities(net, (0, 1, 2))
        - msnc3.triple_probabilities(tree, (0, 1, 2))
    ).max() > 0.02


@requires_phylonet
def test_the_exact_probability_matches_phylonet_on_every_triple(tmp_path):
    """The independent implementation.

    Relies on the measured fact (``scripts/probe_phylonet.py``) that PhyloNet scores a gene tree on
    a *subset* of the network's taxa as the correct marginal, which is what makes this a
    triple-by-triple comparison rather than a whole-likelihood one.
    """
    for n, rmax, r, cu, gamma, seed in ((6, 2, 1, 0.5, 0.5, 0), (8, 3, 2, 1.0, 0.3, 1)):
        net = N.with_gamma(
            N.scaled(N.with_reticulations(N.build_base(n, rmax, seed=seed), r), cu), gamma)
        idx = triples.triple_index(n)
        theirs = phylonet.triple_probabilities_many(
            N.to_enewick(net), net.labels, [tuple(x) for x in idx], workdir=tmp_path / f"n{n}")
        ours = np.stack([msnc3.triple_probabilities(net, tuple(x)) for x in idx])
        assert np.abs(ours - theirs).max() < 1e-8


@requires_phylonet
def test_phylonets_pl_flag_is_a_thread_count_and_not_a_pseudo_likelihood_switch(tmp_path):
    """Recorded as a test because the plan's I1 says to validate against "CalGTProb/MPL", and the
    obvious reading of ``-pl`` is wrong: it takes a value, and the value does not change the
    answer.  MPL comes from ``InferNetwork_MPL``."""
    tree = "((A:1.0,B:1.0):0.7,C:1.7);"
    plain = phylonet.cal_gt_prob(tree, ["((A,B),C)"], workdir=tmp_path / "a")[0]
    text = phylonet.nexus({"net": tree}, {"g": "((A,B),C)"}, ["CalGTProb net (g) -pl 1"])
    threaded = float(
        phylonet._LOGP.search(phylonet.run(text, workdir=tmp_path / "b")).group(1))  # noqa: SLF001
    assert plain == pytest.approx(threaded, abs=1e-12)


def test_a_fitted_model_embeds_exactly_in_the_next_one_at_gamma_zero():
    """E2a's whole argument, as an invariant rather than an inference.

    ``sup MPL`` cannot fall when a reticulation is added, because the k-reticulation model is a
    *point* of the (k+1) parameter space at ``gamma_new = 0``.  Writing that point down exactly is
    what makes a monotonicity violation mean something; getting it wrong manufactures violations,
    and did, three separate ways:

    * embedding from the larger network's *generating* lengths instead of the fitted ones -- those
      differ by whatever re-optimisation buys, thousands of log units;
    * zeroing ``gammas[-1]``, when ``Parameters`` orders hybrids by post-order and the new one is
      not last;
    * a gamma floor of 1e-4 rather than 0, worth 1.85 log units at the embedding.

    So the assertion is equality, not inequality.
    """
    from opc.mpl import likelihood, perturb

    net = N.with_gamma(N.scaled(N.with_reticulations(N.build_base(12, 3, seed=2), 2), 0.5), 0.5)
    acc = triples.TripleCounts.empty(12)
    for nwk in oracle.simulate_gene_trees(net, 200, seed=5):
        triples.count_tree(nwk, {lab: i for i, lab in enumerate(net.labels)}, acc)
    _, obs = acc.as_rows()

    small = likelihood.TripleModel(net)
    fitted, ll_k = likelihood.optimise(small, obs, maxiter=60)

    rng = np.random.default_rng(11)
    bigger = perturb.add_reticulation(net, rng)
    assert bigger is not None and bigger.n_retics == net.n_retics + 1
    big = likelihood.TripleModel(bigger)
    embedded = perturb.embed_parameters(fitted, big, bigger)

    assert big.log_likelihood(obs, embedded) == pytest.approx(ll_k, abs=1e-6)
    # And the thing that made it wrong: the new hybrid is not last in the vector.
    new_slot = list(big.params.hybrids).index(bigger.retics[-1].hybrid)
    assert embedded.gammas[new_slot] == 0.0
    assert likelihood.MIN_GAMMA == 0.0, "gamma's interval must be closed for the embedding"


def _hybrid_leaf_network(gamma: float) -> N.Network:
    """``root -> D`` and ``root -> H``; ``D -> (t2,t3)`` and ``D ~~> H``; **H is the leaf t1**."""
    return N.Network(
        n_taxa=3, labels=["t1", "t2", "t3"],
        time=[0.0, 0.4, 0.4, 0.9, 1.5, 1.5],
        tree_parent=[-1, 0, 0, 1, 3, 3],
        tree_children=[[1, 2], [3], [], [4, 5], [], []],
        leaf_taxon={2: 0, 4: 1, 5: 2},
        horiz_child={1: 2}, donor_of={2: 1},
        retics=[N.Reticulation(hybrid=2, donor=1, order=0, time=0.4)],
        root=0, gamma=gamma,
    )


def test_a_hybrid_may_be_a_leaf_and_gamma_still_decides_which_way_it_goes():
    """A hybrid species with one extant descendant -- the common biological case.

    The generator never builds one (it makes hybrids by splitting an edge, so the hybrid always
    keeps a tree child), but the **regular** version of a network does whenever a hybrid's cluster
    is a singleton, and ``solve.closure.network_from_hasse`` produces exactly that.  Both DPs used
    to skip leaves wholesale, so the lineage took its tree parent with probability 1 and gamma was
    ignored: the network scored identically at every gamma, silently.
    """
    from opc.mpl import likelihood

    probs = [msnc3.triple_probabilities(_hybrid_leaf_network(g), (0, 1, 2))
             for g in (0.2, 0.5, 0.8)]
    spread = max(np.abs(a - b).max() for a in probs for b in probs)
    assert spread > 0.02, "gamma made no difference, so the hybrid leaf is being ignored"

    # And the fast path agrees with the reference on it.
    for g in (0.2, 0.5, 0.8):
        net = _hybrid_leaf_network(g)
        model = likelihood.TripleModel(net)
        assert net.retics[0].hybrid in model.params.hybrids
        assert np.abs(
            model.probabilities()[0] - msnc3.triple_probabilities(net, (0, 1, 2))
        ).max() < 1e-12


def test_a_hybrid_leaf_survives_the_round_trip_through_extended_newick():
    """``()#H1`` is malformed *and* drops the taxon name; PhyloNet also refuses a childless
    reticulation, throwing ArrayIndexOutOfBoundsException from inside GeneTreeProbabilityYF.  The
    form that is both well-formed and accepted is the hybrid carrying a zero-length pendant."""
    enw = N.to_enewick(_hybrid_leaf_network(0.5))
    assert "()" not in enw, f"empty group in {enw}"
    assert "(t1:0)#H1" in enw, enw


@requires_phylonet
def test_the_hybrid_leaf_case_matches_phylonet(tmp_path):
    for gamma in (0.2, 0.5, 0.8):
        net = _hybrid_leaf_network(gamma)
        ours = msnc3.triple_probabilities(net, (0, 1, 2))
        theirs = phylonet.triple_probabilities(
            N.to_enewick(net), net.labels, (0, 1, 2), workdir=tmp_path / f"g{gamma}")
        assert np.abs(ours - theirs).max() < 1e-9
