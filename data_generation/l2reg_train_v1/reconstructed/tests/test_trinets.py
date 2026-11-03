"""I2: restricting a level-1 network to three taxa, and the shapes that result.

What is checked here is the **restriction and grouping** machinery -- that a restriction is a valid
level-<=1 trinet, that it preserves the triple probability exactly, and that the shape set
saturates as more networks are sampled.  The plan's I2 also asks for the catalog to be *proved*
complete and for ``P(t | shape, params)`` in closed form per shape; neither is done yet, and
``HANDOFF.md`` says so.  The saturation check below is the evidence that an enumeration by ``k`` is
the right shape for that work.
"""

from __future__ import annotations

import numpy as np
import pytest

from opc.mpl import likelihood, msnc3, trinets, triples
from opc.simulate import networks as N


def _net(n, rmax, r, cu=0.5, gamma=0.5, seed=0, **kw):
    return N.with_gamma(
        N.scaled(N.with_reticulations(N.build_base(n, rmax, seed=seed, **kw), r), cu), gamma)


def test_a_restriction_is_a_valid_trinet_with_at_most_the_networks_reticulations():
    for n, rmax, r in ((8, 3, 2), (12, 4, 4), (20, 4, 3)):
        net = _net(n, rmax, r, seed=n)
        for row in triples.triple_index(n)[:200]:
            tri = trinets.restrict_to_triple(net, tuple(row))
            assert len(tri.leaf_of) == 3
            assert sorted(tri.leaf_of.values()) == [0, 1, 2]
            roots = [v for v in range(tri.n_nodes) if not tri.parents[v]]
            assert roots == [tri.root]
            for v in range(tri.n_nodes):
                assert len(tri.parents[v]) <= 2
                if len(tri.parents[v]) == 2:
                    assert v in tri.gamma_source
            assert tri.n_hybrids <= r


def test_every_surviving_hybrid_keeps_both_of_its_parents():
    """The property that makes the restriction of a level-1 network level-<=1 rather than a tangle.

    ``ancestors`` follows *all* incoming edges, so a hybrid ancestral to one of the three leaves
    drags both of its parents in with it.  There is therefore no "half a reticulation" to handle.
    """
    net = _net(20, 4, 4, seed=3)
    seen_hybrid = 0
    for row in triples.triple_index(20)[:400]:
        tri = trinets.restrict_to_triple(net, tuple(row))
        seen_hybrid += tri.n_hybrids
        for v, _ in tri.gamma_source.items():
            assert len(tri.parents[v]) == 2
    assert seen_hybrid > 0, "no restriction retained a reticulation, so this proved nothing"


def test_the_restriction_preserves_the_triple_probability_exactly():
    """Dropping subtrees and summing lengths across suppressed nodes has to change nothing.

    Checked through the fast evaluator, which is built entirely out of restrictions, against the
    reference DP, which is not.
    """
    for n, rmax, r, cu, gamma, seed in ((6, 2, 1, 0.5, 0.5, 0), (10, 3, 3, 0.5, 0.5, 2)):
        net = _net(n, rmax, r, cu, gamma, seed)
        model = likelihood.TripleModel(net)
        fast = model.probabilities()
        slow = np.stack([msnc3.triple_probabilities(net, tuple(x)) for x in model.index])
        assert np.abs(fast - slow).max() < 1e-12
        assert np.abs(fast.sum(axis=1) - 1.0).max() < 1e-12


def test_the_shape_set_saturates_as_more_networks_are_sampled():
    """Completeness is checkable rather than assumed -- this is what the check looks like.

    New networks at a fixed ``k`` must stop producing new shapes.  If they never stop, the catalog
    is not finite in the way I2 assumes and the enumeration has to be organised differently.
    """
    nets = [_net(10, 3, 2, seed=s) for s in range(8)]
    sizes = []
    catalog: dict = {}
    for net in nets:
        catalog.update(trinets.catalog([net]))
        sizes.append(len(catalog))
    # The last three networks together must add fewer shapes than the first one did on its own.
    assert sizes[-1] - sizes[-4] < sizes[0], (
        f"shape count still climbing: {sizes} -- the catalog is not saturating"
    )


def test_the_number_of_shapes_stays_small_enough_for_grouping_to_be_worth_it():
    """The measurement that justifies the whole design: 19,600 triples, ~71 shapes."""
    net = _net(50, 4, 4, seed=0, vertex_disjoint=True)
    counts = trinets.shape_counts(net)
    assert sum(counts.values()) == 19600
    assert len(counts) < 200, f"{len(counts)} shapes is too many for grouping to pay"
    reps = {}
    for row in triples.triple_index(50):
        t = trinets.restrict_to_triple(net, tuple(row))
        reps.setdefault(t.key, t)
    tree_triples = sum(counts[k] for k, t in reps.items() if t.is_tree)
    assert tree_triples / 19600 > 0.5, "most triples should be reticulation-free at k=4"
    assert max(t.n_nodes for t in reps.values()) <= 20


def test_the_fast_evaluator_and_the_reference_agree_on_perturbed_parameters():
    """Not just at the network's own lengths: the optimiser will move them a long way.

    ``msnc3`` takes the same overrides, so this compares two independent code paths at the same
    off-nominal point rather than comparing each to itself.
    """
    net = _net(10, 3, 3, seed=4)
    model = likelihood.TripleModel(net)
    rng = np.random.default_rng(0)
    params = model.params
    moved = params.with_vector(np.concatenate([
        np.clip(params.lengths * rng.lognormal(0.0, 0.8, params.lengths.shape), 0.0, 20.0),
        rng.uniform(0.05, 0.95, params.gammas.shape),
    ]))
    lengths = {e: v for e, v in zip(moved.edges, moved.lengths, strict=True)}
    gammas = {h: v for h, v in zip(moved.hybrids, moved.gammas, strict=True)}
    fast = model.probabilities(moved)
    slow = np.stack([
        msnc3.triple_probabilities(net, tuple(x), lengths=lengths, gammas=gammas)
        for x in model.index
    ])
    assert np.abs(fast - slow).max() < 1e-12


def test_pendant_branch_lengths_are_not_parameters_because_they_cannot_matter():
    net = _net(10, 3, 2, seed=1)
    model = likelihood.TripleModel(net)
    for child, _ in model.params.edges:
        assert child not in net.leaf_taxon
    # And the count is what the structure says: one per internal edge, two per hybrid.
    internal = sum(
        1 for v in range(net.n_nodes)
        if net.tree_parent[v] >= 0 and not net.is_leaf(v)
    )
    assert len(model.params.edges) == internal + net.n_retics
    assert len(model.params.hybrids) == net.n_retics


# ------------------------------------------------------------------------- I3: profiled weights


def test_the_tree_profile_closed_form_matches_numerical_maximisation():
    """The MP-EST profile ``e^-x = 3(b+c)/(2(a+b+c))``, clipped at 1, against L-BFGS-B."""
    from opc.mpl import profile

    net = _net(6, 1, 0, seed=1)
    shape = trinets.restrict_to_triple(net, (0, 1, 2))
    assert shape.is_tree
    edges = profile.effective_edges(shape)
    match = int(np.argmax(profile.reachable_probabilities(shape, edges, np.ones(len(edges)))[0]))
    for rho in ([700, 150, 150], [400, 300, 300], [500, 300, 200], [200, 400, 400]):
        rho = np.array(rho, dtype=float)
        closed, _ = profile.tree_profile(rho, match)
        numeric = profile.shape_profile(shape, rho, restarts=6).w
        assert closed == pytest.approx(numeric, abs=1e-8)


def test_a_shape_can_only_fit_data_that_shares_its_symmetry():
    """Why §6.1's parameter count overstates the risk (A36).

    A trinet whose cycle separates one taxon from the other two cannot tell those two apart, so it
    emits ``(p, q, q)`` however many parameters it carries -- and then it cannot fit an asymmetric
    ``rho`` at all.  Both halves are asserted, because the finding is that *most* shapes are locked
    this way while a few are not.
    """
    from opc.mpl import profile

    net = _net(8, 3, 2, seed=0)
    shapes = trinets.catalog([net])
    locked, free = [], []
    rng = np.random.default_rng(0)
    for shape in shapes.values():
        edges = profile.effective_edges(shape)
        n_g = len(shape.gamma_source)
        theta = np.hstack([rng.lognormal(0.0, 1.2, (400, len(edges))),
                           rng.uniform(0.0, 1.0, (400, n_g))])
        p = profile.reachable_probabilities(shape, edges, theta)
        spread = np.abs(np.sort(p, axis=-1)[:, 1] - np.sort(p, axis=-1)[:, 0])
        (free if spread.max() > 1e-6 else locked).append(shape)
    assert locked and free, "expected both symmetry-locked and free shapes in this catalog"

    asymmetric = np.array([500.0, 300.0, 200.0])
    for shape in locked[:5]:
        assert profile.saturation_gap(shape, asymmetric, restarts=6) < -1e-3, (
            "a symmetry-locked shape must not fit an asymmetric rho"
        )


def test_the_saturated_maximum_is_an_upper_bound_on_every_shape():
    """``W(tau, s) <= saturated(tau)`` always; exceeding it means the DP is not a
    distribution."""
    from opc.mpl import profile

    net = _net(8, 3, 2, seed=1)
    rho = np.array([450.0, 350.0, 200.0])
    saturated = profile.saturated_log_likelihood(rho)
    for shape in list(trinets.catalog([net]).values())[:12]:
        assert profile.shape_profile(shape, rho, restarts=4).w <= saturated + 1e-6
