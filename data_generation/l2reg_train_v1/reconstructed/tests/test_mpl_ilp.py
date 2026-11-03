"""E4's ILP: the invariants a wrong model would break, on instances small enough to check.

The load-bearing one is the **upper-bound invariant**.  The profiled optimum bounds ``sup MPL``
from above because the parameters are profiled per triple rather than shared, so a bound below any
attainable MPL means the model is missing a feasible assignment.  That is a defect the objective
value alone would never reveal -- it would just look like a tighter bound, which is the direction
one is inclined to believe.  A34's closing note is the whole reason this file is invariants rather
than expected numbers.

Everything here runs at n = 5 against the **exhaustive** enumeration from :mod:`opc.theory.galled`,
so "every galled tree" means every galled tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from opc.clusters import extract
from opc.mpl import likelihood, trinets, triples
from opc.simulate import msnc_oracle
from opc.simulate import networks as N
from opc.solve import backend, candidates, mpl_ilp
from opc.theory import galled as G

N_TAXA, N_LOCI = 5, 300


def _fixture(tree_i: int = 3, cycle_i: int = 2, seed: int = 5):
    """A true galled tree, gene trees from it, and the tau = 0 ground set they induce."""
    from opc import bitset

    trees = list(G.rooted_binary_trees(N_TAXA))
    tree = trees[tree_i]
    combo = (G.cycles_of(tree)[cycle_i],)
    truth = N.scaled(G.to_network(tree, combo), 1.0)
    truth.gamma = 0.5

    counter = extract.SupportCounter(N_TAXA, [N_LOCI])
    acc = triples.TripleCounts.empty(N_TAXA)
    taxon_index = {lab: i for i, lab in enumerate(truth.labels)}
    for line in msnc_oracle.simulate_gene_trees(truth, N_LOCI, seed=seed):
        line += ";"
        out: list[int] = []
        extract.clusters_from_newick(line, taxon_index, out)
        counter.add_locus(out)
        acc.add_tree(triples.child_leaf_sets(line, taxon_index))
    masks, support, _ = counter.finalize(min_support=1)
    words = np.stack([bitset.from_indices(
        [t for t in range(N_TAXA) if int(m) >> t & 1], N_TAXA) for m in masks])
    gs = candidates.build(words, support[:, 0], N_TAXA, m=N_LOCI, tau=0)
    idx, obs = acc.as_rows()
    catalog = trinets.catalog(
        G.to_network(t, c) for t in trees for k in (0, 1)
        for c in G.galled_placements(t, k, G.cycles_of(t))
    )
    return trees, truth, gs, idx, obs, catalog


@pytest.fixture(scope="module")
def instance():
    return _fixture()


def _solve(gs, obs, idx, catalog, budget, **kw):
    table = mpl_ilp.build_shape_table(obs, catalog, mode=kw.pop("mode", "profiled"))
    model = mpl_ilp.build_model(gs, obs, table, budget, index=idx, triangles=gs.triangles())
    res, bound, primal = mpl_ilp.solve(model, time_limit=300, **kw)
    return table, model, res, bound, primal


def test_the_bound_dominates_every_galled_tree_there_is(instance):
    """``ILP >= sup MPL``, checked against the exhaustive enumeration rather than against itself.

    A violation is not a loose bound, it is a missing feasible assignment.  The vacuity guard
    matters as much as the assertion: if the enumeration produced nothing, or if the bound were
    ``+inf``, this would pass while proving nothing.
    """
    trees, _, gs, idx, obs, catalog = instance
    _, _, _, bound, _ = _solve(gs, obs, idx, catalog, budget=1)
    assert np.isfinite(bound)

    scored = 0
    best = -np.inf
    seen: set[int] = set()
    for tree in trees:
        cycles = G.cycles_of(tree)
        for k in (0, 1):
            for combo in G.galled_placements(tree, k, cycles):
                inst = G.instance(tree, combo)
                if inst.system in seen:
                    continue
                seen.add(inst.system)
                _, ll = likelihood.optimise(
                    likelihood.TripleModel(G.to_network(tree, combo)), obs, maxiter=100)
                assert ll <= bound + 1e-6, (
                    f"a galled tree scores {ll} above the bound {bound}"
                )
                best = max(best, ll)
                scored += 1
    assert scored > 100, "the enumeration produced too little to have tested anything"
    # And the bound has to be a bound on something real rather than merely large: an unattainable
    # +inf would satisfy every assertion above.
    assert np.isfinite(best) and bound >= best


def test_the_assignment_respects_its_own_coupling(instance):
    """Every assigned class must have the witness the model says it needs.

    Checked against the *solution*, not against the rows: a coupling block written into the wrong
    columns satisfies itself and nothing else.
    """
    from opc import bitset

    _, _, gs, idx, obs, catalog = instance
    _, model, res, _, _ = _solve(gs, obs, idx, catalog, budget=1)
    assert res.x is not None
    assigned = model.assignment(res.x)
    selected = set(model.selection(res.x).tolist())
    dense = bitset.to_dense(np.ascontiguousarray(gs.words), N_TAXA).astype(bool)
    y = np.asarray(res.x)[model.y_cols] > 0.5

    n_tree = n_ret = 0
    for t, (a, b, c) in enumerate(idx):
        s = int(assigned[t])
        if s < 3:
            p, q, out = ((a, b, c), (a, c, b), (b, c, a))[s]
            witnesses = [i for i in selected if dense[i, p] and dense[i, q] and not dense[i, out]]
            assert witnesses, f"resolution {s} assigned to triple {(a, b, c)} with no witness"
            n_tree += 1
        elif s >= mpl_ilp.RET_OFFSET:
            assert y.any(), "a reticulate class was assigned with no active candidate block"
            n_ret += 1
    assert n_tree > 0 and n_ret > 0, (
        "the solution used only one kind of class, so half the coupling went untested"
    )


def test_the_shape_variables_come_out_integral_although_they_are_continuous(instance):
    """Why ``u`` is not declared binary: for fixed ``x`` and ``y`` its block is an assignment LP
    whose vertices are integral, so making it binary would only cost branch-and-bound nodes."""
    _, _, gs, idx, obs, catalog = instance
    _, model, res, _, _ = _solve(gs, obs, idx, catalog, budget=1)
    u = np.asarray(res.x)[model.u_offset:]
    assert model.integrality[model.u_offset:].sum() == 0
    assert np.abs(np.minimum(u, 1.0 - u)).max() < 1e-6


def test_the_tiebreak_cannot_move_the_bound(instance):
    """The lexicographic second stage exists to fix a degenerate primal, not to change the answer.

    Without it the ILP returns an arbitrary point of an enormous optimal face -- measured, one with
    *no reticulations* -- and the plan's "bound minus the value of the ILP's own solution" then
    reads as a relaxation gap that is really a primal artefact.
    """
    _, _, gs, idx, obs, catalog = instance
    table = mpl_ilp.build_shape_table(obs, catalog, mode="profiled")
    model = mpl_ilp.build_model(gs, obs, table, 1, index=idx, triangles=gs.triangles())
    untied_res, untied_bound, _ = mpl_ilp.solve(model, time_limit=300)
    tied_res, tied_bound, primal = mpl_ilp.solve(
        model, time_limit=300, tiebreak=np.asarray(gs.support, dtype=float))
    assert tied_bound == pytest.approx(untied_bound, rel=1e-9)
    assert primal is not tied_res or untied_res.x is None
    # The tiebroken primal must be at least as good under the tiebreak's own objective.
    tie = np.asarray(gs.support, dtype=float)
    assert tie[model.selection(primal.x)].sum() >= tie[model.selection(untied_res.x)].sum() - 1e-6


@pytest.mark.skipif(not backend.available("scip"), reason="pyscipopt not installed")
def test_highs_and_scip_agree_on_the_optimum(instance):
    """The cross-validation ``solve/backend.py`` exists for: two solvers, one model."""
    _, _, gs, idx, obs, catalog = instance
    _, _, _, highs, _ = _solve(gs, obs, idx, catalog, budget=1, solver="highs")
    _, _, _, scip, _ = _solve(gs, obs, idx, catalog, budget=1, solver="scip")
    assert highs == pytest.approx(scip, rel=1e-6)


def test_the_batched_tree_profile_matches_the_scalar_closed_form():
    """The fast W column against ``profile.tree_profile``, which the tests already validate against
    numerical maximisation.  Every fast kernel in this repo has a slow reference; this is that."""
    from opc.mpl import profile

    rng = np.random.default_rng(0)
    obs = np.vstack([
        rng.multinomial(500, [0.6, 0.2, 0.2], size=20),
        np.array([[300, 300, 300], [0, 0, 0], [1000, 0, 0], [1, 1, 998]]),
    ]).astype(float)
    fast = mpl_ilp._tree_profile_batch(obs)  # noqa: SLF001
    for i, row in enumerate(obs):
        for r in range(3):
            slow, _ = profile.tree_profile(row, r)
            assert fast[i, r] == pytest.approx(slow, abs=1e-9)


def test_shape_signature_counts_what_it_says_it_counts(instance):
    """``(taxa below the hybrid, taxa elsewhere on the cycle)``, against a direct descent.

    This test used to assert ``a + b == 3`` with the comment *"a trinet has three leaves, and the
    apex reaches all of them"*.  The first half is true and the second is not: a leaf can attach
    **above** the cycle's apex, and then it is on no side of the cycle at all.  The assertion held
    only because :func:`opc.solve.mpl_ilp.shape_signature` was taking ``b`` against the trinet's
    *root*, so it pinned the defect rather than the property (A48).  Both are checked now.
    """
    _, _, _, _, _, catalog = instance
    reticulate = 0
    short = 0
    for shape in catalog.values():
        sig = mpl_ilp.shape_signature(shape)
        if shape.is_tree:
            assert sig is None
            assert mpl_ilp.shape_signature_root(shape) is None
            continue
        reticulate += 1
        a, b = sig
        assert a >= 1
        assert a + b <= 3
        short += a + b < 3
        # The old definition, kept so the difference stays measurable: it always sums to three,
        # which is exactly why it could never reach the (1,0)/(1,1)/(2,0) classes.
        ra, rb = mpl_ilp.shape_signature_root(shape)
        assert ra + rb == 3
        assert ra == a, "both definitions count the taxa below the hybrid the same way"
    assert reticulate > 0, "the catalog held no reticulate shape, so nothing was tested"
    assert short > 0, (
        "no shape had a leaf above its apex, so the repaired signature was never exercised and "
        "this test would pass against the defect it exists to catch"
    )


def test_the_model_builds_when_the_overlap_graph_has_more_than_one_component():
    """The (N3O) rows have to reach the right component, and the obvious way to index them is a
    trap.

    ``Component`` is a dataclass of numpy arrays, so ``comps.index(comp)`` returns arrays from
    ``__eq__`` and raises on the *second* component -- while CPython's identity short-circuit makes
    it work perfectly whenever there is only one.  At tau = 0 there usually is only one (A33: the
    overlap graph is a single giant component by CU = 1.0), so this needs a fixture built for it,
    exactly as A21's budget-DP test needed two disjoint taxon ranges.
    """
    from opc import bitset

    n = 8
    masks = [0b0011, 0b0101, 0b1001,  # a monochromatic triangle on taxa 0-3, meeting in {0}
             0b0011 << 4, 0b0101 << 4, 0b1001 << 4]  # and its copy on 4-7
    words = np.stack([bitset.from_indices(
        [t for t in range(n) if m >> t & 1], n) for m in masks])
    gs = candidates.build(words, np.full(len(masks), 5, dtype=np.int32), n, m=5, tau=0)
    assert len(gs.components) >= 2, "the fixture produced one component; nothing was tested"
    assert any(t.shape[0] for t in gs.triangles()), "no (N3O) row, so the indexing went untested"

    obs = np.full((len(triples.triple_index(n)), 3), 10.0)
    table = mpl_ilp.build_shape_table(obs, None, mode="saturated")
    model = mpl_ilp.build_model(gs, obs, table, 1, triangles=gs.triangles())
    _, bound, _ = mpl_ilp.solve(model, time_limit=120)
    assert np.isfinite(bound)


def test_a_bigger_budget_cannot_lower_the_bound(instance):
    """``sup`` over a larger feasible set, so the bound is non-decreasing in ``k``.

    The same monotonicity E2a checks for the criterion, checked here for the relaxation.  It is
    also a live check on the budget row being wired to the right columns: a budget that constrains
    nothing, or the wrong variables, breaks this in one direction or the other.
    """
    _, _, gs, idx, obs, catalog = instance
    bounds = [_solve(gs, obs, idx, catalog, budget=j)[3] for j in (0, 1, 2)]
    assert bounds[0] <= bounds[1] + 1e-6
    assert bounds[1] <= bounds[2] + 1e-6
    assert bounds[0] < bounds[2] - 1e-6, (
        "the budget changed nothing at all, so the reticulate classes are unreachable"
    )


def test_the_primal_is_not_an_upper_bound_unless_optimality_was_proved():
    """A45.  ``-res.objective`` is the incumbent, and the incumbent is a bound the *wrong way*.

    The model maximises ``sum W`` (written as a minimisation of ``-W``), so a feasible incumbent is
    a **lower** bound on the ILP optimum and therefore not an upper bound on ``sup MPL`` at all.
    Measured on A43's n = 50 instance: at a 10 s limit ``bound`` came back as -21,505,335 against
    a true optimum of -4,182,164 -- five times too low, precise, and confidently wrong.

    Faked rather than provoked, for the reason A43's no-incumbent test gives: reproducing it
    needs a model far too large for a unit test, which is why it went unnoticed until one appeared.
    ``valid_upper_bound`` is checked on all four cases the solver can present.
    """
    from types import SimpleNamespace

    def result(status, objective, dual):
        return SimpleNamespace(status=status, objective=objective, dual_bound=dual,
                               x=np.zeros(1), mip_gap=0.0)

    # Proven optimal: dual and primal coincide, so the valid bound *is* the old one.  This is what
    # keeps A42's published numbers re-deriving unchanged.
    opt = result("optimal", 4_182_163.76, 4_182_163.76)
    assert mpl_ilp.valid_upper_bound(opt) == pytest.approx(-opt.objective)

    # Time limit with a dual bound: the ladder's 300 s rung, real numbers.
    lim = result("feasible", 4_197_320.42, -98_823_923.30)
    valid = mpl_ilp.valid_upper_bound(lim)
    assert valid == pytest.approx(98_823_923.30)
    # Vacuous, but on the correct side of the true optimum -- which the primal is not.
    assert valid > -4_182_163.76
    assert -lim.objective < -4_182_163.76, "the primal must sit BELOW the optimum, or this is moot"

    # Time limit before the root relaxation finished: nothing proved, so no finite bound exists.
    assert mpl_ilp.valid_upper_bound(result("limit", float("nan"), float("nan"))) == np.inf
    assert mpl_ilp.valid_upper_bound(result("limit", float("nan"), None)) == np.inf

    assert mpl_ilp.valid_upper_bound(result("infeasible", float("nan"), float("nan"))) == -np.inf


def test_the_valid_bound_agrees_with_the_primal_on_a_solved_instance(instance):
    """On a real model solved to optimality the two must coincide -- otherwise the sign is wrong.

    The unit test above fakes results, so it cannot catch a sign error in how ``dual_bound`` is
    read back from a solver.  This one runs the actual ILP.
    """
    _, _, gs, idx, obs, catalog = instance
    table = mpl_ilp.build_shape_table(obs, catalog, mode="saturated")
    model = mpl_ilp.build_model(gs, obs, table, 1, index=idx, triangles=gs.triangles())
    res, bound, _ = mpl_ilp.solve(model, time_limit=300)
    assert res.status == "optimal", f"instance did not solve: {res.status}"
    assert mpl_ilp.valid_upper_bound(res) == pytest.approx(bound, rel=1e-9)


# ------------------------------------------------------------ the tau axis and its cache keying


def _e4():
    """E4's shared core, now importable: ``opc.solve.exact``, not a script loaded by path."""
    from opc.solve import exact
    return exact


def test_tau_zero_and_one_are_the_same_ground_set_and_share_the_untagged_cache(instance):
    """``build`` filters ``support >= max(tau, 1)``, so the plan's tau = 0 label costs nothing.

    Both halves are asserted together because they have to agree: if the two thresholds ever
    stopped selecting the same clusters, sharing one cache path would silently serve one run's
    answer to the other.
    """
    _trees, _truth, gs, _idx, _obs, _catalog = instance
    at0 = candidates.build(gs.words, gs.support, N_TAXA, tau=0)
    at1 = candidates.build(gs.words, gs.support, N_TAXA, tau=1)
    assert at0.n_clusters == at1.n_clusters
    assert np.array_equal(at0.words, at1.words)

    e4 = _e4()
    assert e4.tau_tag(0) == e4.tau_tag(1) == ""
    assert e4.tau_tag(2) == "_tau2"
    assert e4.solver_tag("gurobi") + e4.tau_tag(2) == "_gurobi_tau2"


def test_raising_tau_shrinks_the_ground_set_to_a_subset_of_itself(instance):
    """Monotone in tau, and a *subset* -- not merely smaller.

    The sweep at tau = 2 is only interpretable beside tau = 0 if the higher threshold deletes
    candidates rather than exchanging them, which a support filter guarantees and a re-derivation
    from a different support column would not.
    """
    _trees, _truth, gs, _idx, _obs, _catalog = instance
    seen = None
    sizes = []
    for tau in (1, 2, 3):
        here = candidates.build(gs.words, gs.support, N_TAXA, tau=tau)
        masks = {int.from_bytes(w.tobytes(), "little") for w in here.words}
        sizes.append(len(masks))
        if seen is not None:
            assert masks <= seen
        seen = masks
    # Non-vacuous: the fixture has to actually lose candidates, or the subset test proves nothing.
    assert sizes[0] > sizes[-1], f"tau did not bind on this fixture: {sizes}"
