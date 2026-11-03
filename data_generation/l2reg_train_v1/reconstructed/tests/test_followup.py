"""The v0.3 follow-up arm: the invariants F0-F6 rest on.

House rules, as elsewhere in this suite:

* **the oracle is never the code under test.** The mesh profiler is checked against
  :func:`opc.mpl.profile.shape_profile`, the row-by-row multi-start optimiser it replaces; the
  saturated-mode identity is checked against a closed form written out here rather than against
  ``ShapeTable``'s own accessor.
* **non-vacuity is asserted.** Several of these tests are only meaningful if the fixture actually
  produced the case -- a catalog with a leaf above the apex, a ground set with a block big enough
  to cap. Where that is so, the test says so and fails if it did not.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opc.followup import extent as EX
from opc.followup import face, wtable
from opc.followup import search as SE
from opc.mpl import likelihood, msnc3, profile, trinets, triples
from opc.simulate import msnc_oracle
from opc.simulate import networks as N
from opc.solve import backend, candidates, mpl_ilp
from opc.solve import truth as TR
from opc.theory import galled as G

REPO = Path(__file__).resolve().parents[1]
N_TAXA, N_LOCI = 5, 300


@pytest.fixture(scope="module")
def small():
    """A galled tree at n = 5, its gene trees, the tau = 0 ground set, and a full catalog."""
    from opc import bitset
    from opc.clusters import extract

    trees = list(G.rooted_binary_trees(N_TAXA))
    tree = trees[3]
    truth = N.scaled(G.to_network(tree, (G.cycles_of(tree)[2],)), 1.0)
    truth.gamma = 0.5
    counter = extract.SupportCounter(N_TAXA, [N_LOCI])
    acc = triples.TripleCounts.empty(N_TAXA)
    taxon_index = {lab: i for i, lab in enumerate(truth.labels)}
    for line in msnc_oracle.simulate_gene_trees(truth, N_LOCI, seed=5):
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
        for c in G.galled_placements(t, k, G.cycles_of(t)))
    return truth, gs, idx, obs, catalog


# ------------------------------------------------------------------- A46 / A47: what saturated is

def test_saturated_mode_makes_the_bound_a_closed_form_of_the_data(small):
    """``sum_tau max_s W(tau, s) == sum_tau saturated(tau)``, against a hand-written saturated sum.

    This is A46 as an assertion.  Under ``mode='saturated'`` the reticulate columns are set to the
    multinomial maximum outright, so the row maximum is that maximum for **every** triple and the
    table's optimum cannot depend on the candidate set, the budget or tau.
    """
    _, _, _, obs, _ = small
    table = mpl_ilp.build_shape_table(obs, None, mode="saturated")
    # The oracle: the multinomial maximum written out directly, not read off the table.
    total = obs.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(obs > 0, obs * np.log(np.where(obs > 0, obs / total, 1.0)), 0.0)
    expected = float(terms.sum())
    assert table.w.max(axis=1).sum() == pytest.approx(expected, rel=1e-12)
    assert table.attaining().all(), "every triple attains the maximum -- by construction"


def test_frac_saturating_is_frac_assigned_reticulate_in_saturated_mode(small):
    """A47: the published statistic collapses to 'was a reticulate class assigned'.

    ``saturating`` is flagged True on exactly the reticulate columns, so reading it at whichever
    class the solver assigned measures the assignment and not the data.  Checked here over every
    possible assignment rather than over the one a solver happened to return.
    """
    _, _, _, obs, _ = small
    table = mpl_ilp.build_shape_table(obs, None, mode="saturated")
    rng = np.random.default_rng(0)
    for _ in range(20):
        assigned = rng.integers(0, len(mpl_ilp.CLASSES), size=table.n_triples)
        frac_sat = table.saturating[np.arange(table.n_triples), assigned].mean()
        frac_ret = (assigned >= mpl_ilp.RET_OFFSET).mean()
        assert frac_sat == pytest.approx(frac_ret, abs=1e-15)


def test_a_tree_class_attains_the_maximum_exactly_when_the_minority_counts_tie():
    """The mechanism behind a saturated-mode bound that is reached with tree classes.

    The MP-EST profile fits ``(a, (b+c)/2, (b+c)/2)`` and the saturated fit is ``(a, b, c)``, so
    the two coincide iff ``b == c``.  Built by hand rather than taken from a fixture: at n = 5
    there are ten triples and whether any ties is luck, so a fixture would make the test's
    coverage a property of the seed.  Both sides are present here by construction.
    """
    tied_rows = [(700, 150, 150), (1000, 0, 0), (400, 300, 300), (150, 700, 150)]
    untied_rows = [(600, 250, 150), (900, 60, 40), (500, 300, 200)]
    obs = np.array(tied_rows + untied_rows, dtype=float)
    table = mpl_ilp.build_shape_table(obs, None, mode="saturated")
    expected = np.r_[np.ones(len(tied_rows), bool), np.zeros(len(untied_rows), bool)]
    assert np.array_equal(table.tree_attains(), expected)
    # and the corresponding per-triple deficit is zero exactly on the tied rows
    deficit = table.saturated - table.w[:, :3].max(axis=1)
    assert np.allclose(deficit[expected], 0.0, atol=1e-9)
    assert (deficit[~expected] > 1e-6).all()


# --------------------------------------------------------------------------- A48: the signature

def test_the_repaired_signature_reaches_classes_the_root_one_cannot(small):
    """A48.  Non-vacuous: the catalog must contain a shape with a leaf above the apex."""
    _, _, _, _, catalog = small
    apex, root = {}, {}
    for shape in catalog.values():
        for store, fn in ((apex, mpl_ilp.shape_signature),
                          (root, mpl_ilp.shape_signature_root)):
            sig = fn(shape)
            if sig is not None:
                store[sig] = store.get(sig, 0) + 1
    assert root, "no reticulate shape in the catalog, so nothing was tested"
    assert all(a + b == 3 for a, b in root), "the old signature is supposed to always sum to 3"
    short = {s for s in apex if sum(s) < 3}
    assert short, "the repaired signature reached no short class, so the fixture proves nothing"
    assert not (short & set(root)), "a short class was reachable under the old definition too"


def test_the_cycle_apex_is_not_the_root_when_a_leaf_attaches_above_it(small):
    """The structural fact the repair turns on, checked directly on the trinets."""
    _, _, _, _, catalog = small
    seen = 0
    for shape in catalog.values():
        if not shape.gamma_source:
            continue
        hybrid = min(shape.gamma_source)
        apex = mpl_ilp.cycle_apex(shape, hybrid)
        under_apex = mpl_ilp._below(shape, apex)
        # The apex is always at or below the root, so it can never reach more taxa.
        assert under_apex <= mpl_ilp._below(shape, shape.root)
        assert apex in mpl_ilp._ancestors(shape, hybrid)
        if len(under_apex) < 3:
            # a leaf attaches above the apex, which is the case the root definition cannot see
            seen += 1
            assert apex != shape.root
            assert sum(mpl_ilp.shape_signature(shape)) < 3
    assert seen > 0, "no leaf attached above an apex, so the repair was never exercised"


# ------------------------------------------------------------------------ the profiled W table

def test_the_mesh_profiler_never_beats_the_reference_optimiser(small):
    """Every mesh point is an achievable probability vector, so the mesh cannot exceed the max.

    Oracle: :func:`opc.mpl.profile.shape_profile`, the multi-start L-BFGS-B this replaces.  A mesh
    that came out *above* it would mean one of the two is evaluating a probability vector the shape
    cannot produce -- which is the failure that would silently invalidate the upper bound.
    """
    _, _, _, _, catalog = small
    reticulate = [s for s in catalog.values() if s.gamma_source][:6]
    assert reticulate, "no reticulate shape to profile"
    for shape in reticulate:
        v = wtable.validate(shape, points=2048)
        assert v["max_excess"] < 1e-6, "the mesh reached a probability vector the shape cannot"
        assert v["max_deficit"] < 5.0


def test_the_mesh_and_the_reference_agree_on_a_tree_shape_closed_form(small):
    """On a tree shape both must land on the MP-EST closed form -- a third implementation."""
    _, _, _, _, catalog = small
    trees = [s for s in catalog.values() if not s.gamma_source]
    assert trees, "no tree shape in the catalog"
    rho = np.array([180.0, 70.0, 50.0])
    best_closed = max(profile.tree_profile(rho, m)[0] for m in range(3))
    got = max(float(wtable.mesh_profiler(s, rho[None, :], points=2048)[0]) for s in trees)
    assert got <= best_closed + 1e-6
    assert got == pytest.approx(best_closed, abs=1e-3)


def test_the_dof_penalty_shifts_each_class_by_lambda_times_its_own_df(small):
    """F2's arithmetic, against the unpenalised table."""
    _, _, _, obs, catalog = small
    base = mpl_ilp.build_shape_table(obs, catalog, mode="saturated")
    lam = 0.25
    pen = mpl_ilp.build_shape_table(obs, catalog, mode="saturated", dof_lambda=lam)
    assert np.allclose(base.w - lam * base.df[None, :], pen.w)
    assert base.df[3] == 0, "the star has no free parameter"
    assert base.df[mpl_ilp.RET_OFFSET:].min() > base.df[:3].max(), (
        "a reticulate class must carry more parameters than a tree class, or F2 is inert")


def test_the_signature_choice_changes_the_profiled_table_and_nothing_in_saturated_mode(small):
    """Why ``--w-signature`` needs a cache tag in profiled mode and is harmless in saturated."""
    _, _, _, obs, catalog = small
    kw = dict(profiler=wtable.mesh_profiler)
    a = mpl_ilp.build_shape_table(obs, catalog, mode="profiled",
                                  signature=mpl_ilp.shape_signature, **kw)
    b = mpl_ilp.build_shape_table(obs, catalog, mode="profiled",
                                  signature=mpl_ilp.shape_signature_root, **kw)
    assert not np.allclose(a.w, b.w), "the repair changed nothing, so the tag would be pointless"
    c = mpl_ilp.build_shape_table(obs, None, mode="saturated",
                                  signature=mpl_ilp.shape_signature)
    d = mpl_ilp.build_shape_table(obs, None, mode="saturated",
                                  signature=mpl_ilp.shape_signature_root)
    assert np.array_equal(c.w, d.w)


# -------------------------------------------------------------------------------- F0: the face

def test_the_induced_class_vector_never_needs_the_star_fallback(small):
    """Every triple of a galled tree restricts to a tree trinet or a signature class.

    The star is a fallback for a restriction this code cannot read, so a nonzero share of it is a
    bug rather than a property -- it was 40% before ``induced_classes`` learned to descend past a
    unary root.
    """
    truth, _, idx, _, _ = small
    classes = face.induced_classes(truth, idx)
    assert (classes != 3).all(), "a triple fell back to the star"
    assert (classes >= mpl_ilp.RET_OFFSET).any(), "no triple was reticulate, so nothing was tested"


def test_the_truth_cannot_score_above_the_bound(small):
    """F0's invariant: ``W_true_induced <= sum_tau max_s W(tau, s)``, which the bound attains."""
    truth, _, idx, obs, catalog = small
    for mode, cat in (("saturated", None), ("profiled", catalog)):
        table = mpl_ilp.build_shape_table(
            obs, cat, mode=mode, profiler=wtable.mesh_profiler if cat else None)
        induced = face.w_of_classes(table, face.induced_classes(truth, idx))
        assert induced <= table.w.max(axis=1).sum() + 1e-6, mode


# ------------------------------------------------------------------------------ F5: the extent

def test_eligible_fraction_reproduces_the_plans_block_extent_table():
    """§0.3's table, which the report quotes: an aggressive cap still leaves half the triples.

    Against ``1 - C(50-E,3)/C(50,3)`` computed here from :func:`math.comb`, not against the plan's
    printed values -- the plan rounds ``E = 30`` to 0.943 where the arithmetic gives **0.941836**,
    which is a typo in the plan rather than a disagreement about the quantity.  The other three
    agree to their printed precision.
    """
    from math import comb

    for e in (43, 30, 20, 10):
        assert EX.eligible_fraction(e, 50) == pytest.approx(1 - comb(50 - e, 3) / comb(50, 3))
    assert EX.eligible_fraction(43, 50) == pytest.approx(0.998, abs=5e-4)
    assert EX.eligible_fraction(20, 50) == pytest.approx(0.793, abs=5e-4)
    assert EX.eligible_fraction(10, 50) == pytest.approx(0.496, abs=5e-4)
    assert EX.eligible_fraction(30, 50) == pytest.approx(0.9418, abs=5e-4)
    assert EX.eligible_fraction(0, 50) == 0.0


def test_true_extents_are_the_span_of_the_blocks_clusters(small):
    """Oracle: the union of :func:`opc.simulate.networks.block_clusters`, taken here by hand."""
    truth, _, _, _, _ = small
    got = EX.true_extents(truth)
    assert got.size == len(truth.retics) > 0
    for e, ret in zip(got, truth.retics, strict=True):
        span = set().union(*N.block_clusters(truth, ret))
        assert e == len(span)


def test_an_extent_cap_removes_coupling_without_removing_the_variable():
    """F5's lever: a capped block keeps its ``y`` column, so the model stays feasible.

    The ground set is built by hand rather than simulated.  On the n = 5 fixture every candidate
    block turns out to have the *same* extent, so no cap can bite and the test would pass without
    exercising anything -- the assertion below caught exactly that.  Here there are deliberately
    two blocks of different extent, so a cap between them must remove one and keep the other.
    """
    from opc import bitset

    n_taxa = 8
    # {0,1,2} and {2,3,4} overlap in {2}: one block, extent {0,1,2,3,4} = 5 taxa.
    # {5,6}   and {6,7}   overlap in {6}: another block, extent {5,6,7} = 3 taxa.
    system = [(0, 1, 2), (2, 3, 4), (5, 6), (6, 7)]
    words = np.stack([bitset.from_indices(list(c), n_taxa) for c in system])
    support = np.full(len(system), 10, dtype=np.int32)
    gs = candidates.build(words, support, n_taxa, m=10, tau=0)

    ext = EX.candidate_extents(gs)
    assert sorted(ext.tolist()) == [3, 5], f"the fixture did not build the two blocks: {ext}"

    idx = triples.triple_index(n_taxa)
    obs = np.tile(np.array([600.0, 250.0, 150.0]), (idx.shape[0], 1))
    table = mpl_ilp.build_shape_table(obs, None, mode="saturated")
    full = mpl_ilp.build_model(gs, obs, table, 2, index=idx, triangles=gs.triangles())
    capped = mpl_ilp.build_model(gs, obs, table, 2, index=idx, triangles=gs.triangles(),
                                 extent_cap=4)

    assert capped.sizes["n_blocks_capped"] == 1, "the cap at 4 should remove exactly one block"
    assert capped.sizes["n_ret_coupling_entries"] < full.sizes["n_ret_coupling_entries"]
    assert capped.sizes["n_nonzeros"] < full.sizes["n_nonzeros"]
    # the y columns survive: same column count, so nothing was deleted from the model itself
    assert capped.sizes["n_cols"] == full.sizes["n_cols"]
    assert len(capped.y_cols) == len(full.y_cols)
    # and the capped model is still feasible -- the point of keeping the variable
    res = backend.solve_milp(capped.c, capped.a_ub, capped.b_ub,
                             integrality=capped.integrality, time_limit=60.0, backend="highs")
    assert res.x is not None, "capping a block made the model infeasible"


# ------------------------------------------------------------------------------- the cache tags

def _e4():
    from opc.solve import exact
    return exact


def test_the_w_signature_tag_keeps_the_published_artifacts_addressable():
    """``--w-signature root`` must land on the untagged paths, or e4_small.parquet is lost."""
    e4 = _e4()
    assert e4.w_tag("root") == ""
    assert e4.w_tag("apex") == "_wfix"
    assert e4.solver_tag("gurobi") + e4.tau_tag(2) + e4.w_tag("root") == "_gurobi_tau2"
    assert e4.solver_tag("gurobi") + e4.tau_tag(2) + e4.w_tag("apex") == "_gurobi_tau2_wfix"
    assert e4.signature_fn("root") is mpl_ilp.shape_signature_root
    assert e4.signature_fn("apex") is mpl_ilp.shape_signature


# ------------------------------------------------------------------------------- F3: the search

def test_the_refit_agrees_with_the_landed_optimiser(small):
    """``Scorer.refit`` calls scipy directly; ``likelihood.optimise`` stays the oracle.

    It calls ``minimize`` itself only to keep ``res.nit`` / ``res.nfev``, which ``optimise``
    discards and which §F3 names as the quantity that governs the budget.  If the two ever drift,
    every MPL regret in ``f3_search_*.parquet`` is measured against a different optimiser than
    E1b's, so the agreement is pinned rather than assumed.
    """
    truth, gs, idx, obs, _ = small
    rows = TR.true_cluster_rows(gs.words, truth)
    assert rows.size > 0, "the truth is not in the ground set, so nothing was refitted"
    made = SE.realise(gs.words, rows, N_TAXA, truth.labels)
    assert made is not None, "the truth's own clusters stand for no network"
    model = likelihood.TripleModel(made[0], index=idx)
    assert model.params.vector.size > 0, "a parameterless network would make this vacuous"
    _, reference = likelihood.optimise(model, obs, maxiter=150)

    got = SE.Scorer(obs, idx, N_TAXA, truth.labels, maxiter=150).refit(gs, rows)
    assert got is not None
    assert got.log_likelihood == pytest.approx(reference, abs=1e-9)
    assert got.n_calls > got.n_params, "fewer calls than parameters means no gradient was taken"


def test_parameter_transfer_is_exact_on_an_unchanged_topology(small):
    """Every branch length is keyed by its (child cluster, parent cluster) bitmask pair.

    So transferring a fit back onto the topology it came from must reproduce it exactly -- not
    approximately.  That exactness is the whole reason the inner loop can score at fitted
    parameters instead of at ``network_from_hasse``'s placeholders, which its own docstring says
    carry no information and which leave gamma unidentifiable.
    """
    truth, gs, idx, obs, _ = small
    rows = TR.true_cluster_rows(gs.words, truth)
    scorer = SE.Scorer(obs, idx, N_TAXA, truth.labels, maxiter=150)
    fitted = scorer.refit(gs, rows)
    assert fitted is not None

    net, words = SE.realise(gs.words, rows, N_TAXA, truth.labels)
    model = likelihood.TripleModel(net, index=idx)
    x, hits, total = scorer.fit.vector_for(model.params, words)
    assert total > 0, "the network has no free parameter, so the transfer was never exercised"
    assert hits == total, "an unchanged topology must transfer every edge, not most of them"
    assert model.log_likelihood(obs, model.params.with_vector(x)) == pytest.approx(
        fitted.log_likelihood, abs=1e-9)


def test_the_empty_selection_is_the_star_rather_than_an_error(small):
    """Nothing selected is a network -- the star -- and its likelihood is a closed form.

    ``closure_words`` cannot stack an empty family, so the naive reading is "infeasible"; A49 shows
    what that costs, having found ``19,600 x 1000 x ln(1/3)`` recorded in an E4 ``bound`` column as
    though a solver had produced it.  Every add-only searcher here starts from the empty selection
    whenever the kernel is empty, so getting this wrong would make every curve start at ``-inf``.
    """
    truth, gs, idx, obs, _ = small
    made = SE.realise(gs.words, np.zeros(0, dtype=np.int64), N_TAXA, truth.labels)
    assert made is not None, "the empty selection was rejected instead of read as the star"
    model = likelihood.TripleModel(made[0], index=idx)
    n_loci = int(obs[0].sum())
    assert model.log_likelihood(obs) == pytest.approx(
        obs.shape[0] * n_loci * np.log(1.0 / 3.0), rel=1e-12)


def test_realise_returns_none_rather_than_raising_on_a_selection_that_is_not_level_1(small):
    """A random or annealed selection hits ``network_from_hasse``'s ValueErrors routinely."""
    from opc import bitset
    from opc.solve import closure as CL

    n = 6
    sets = [[0, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4]]
    words = np.stack([bitset.from_indices(s, n) for s in sets])
    full = CL.with_trivial_clusters(CL.closure_words(words, n), n)
    with pytest.raises(ValueError):
        CL.network_from_hasse(full, n)  # the fixture really is the case being tested
    assert SE.realise(words, np.arange(len(sets)), n) is None


def test_the_search_state_agrees_with_the_independent_validator(small):
    """Moves go through ``heuristics._State``; feasibility is re-derived by ``heuristics.check``.

    ``check`` recomputes (L) by a lexsort groupby and (N3O) as ``nnz((A@A).A)/6`` from the accepted
    set alone -- it shares no state with the mask that produced it.  ``SearchState.verify`` also
    asserts the global ``in_s`` and each component's own agree, because they are two records of one
    fact and a move that updated only one would be invisible to ``check``.
    """
    _, gs, _, _, _ = small
    state = SE.SearchState(gs, 1)
    rng = np.random.default_rng(3)
    rejected = accepted = 0
    for _ in range(400):
        v = int(rng.integers(state.n))
        ok, klass = state.can_accept(v)
        if ok:
            state.accept(v, klass)
            accepted += 1
            assert state.verify(), f"accepting {v} produced an infeasible selection"
        else:
            rejected += 1
    assert accepted > 0, "no move was ever accepted, so verify() only ever saw the empty set"
    assert rejected > 0, "no move was ever rejected, so the action mask was never exercised"
    assert state.n_hybrids <= 1


def test_the_evaluation_counter_counts_exactly_the_likelihood_calls(small, monkeypatch):
    """``n_evals`` is F3's x-axis, so it is pinned against the calls it claims to count."""
    truth, gs, idx, obs, _ = small
    calls = {"n": 0}
    original = likelihood.TripleModel.log_likelihood

    def counted(self, *a, **kw):
        calls["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(likelihood.TripleModel, "log_likelihood", counted)
    scorer = SE.Scorer(obs, idx, N_TAXA, truth.labels, maxiter=150)
    # No refit in the callback: a checkpoint re-optimisation is thousands of calls.  The *search*
    # still re-fits, because ``refit_every`` keeps the transfer source fresh, so the invariant is
    # that every likelihood call is either a scored candidate or one of those -- never neither.
    records = SE.run("greedy-mpl", gs, scorer, 1, budget=40,
                     on_checkpoint=lambda n, sel: {"n": int(n)},
                     checkpoints=(0, 10, 40), seed=1, refit_every=2)
    assert scorer.n_evals > 0, "the search spent no evaluations, so the counter proves nothing"
    assert scorer.n_refits > 0, "no refresh happened, so refit_every was never exercised"
    assert calls["n"] == scorer.n_evals + scorer.n_refit_calls
    assert [r["checkpoint"] for r in records] == [0, 10, 40]
    assert all(r["n_evals"] <= 40 for r in records)

    # And with refreshing off, the axis is exactly the likelihood calls again.
    calls["n"] = 0
    plain = SE.Scorer(obs, idx, N_TAXA, truth.labels, maxiter=150)
    SE.run("greedy-mpl", gs, plain, 1, budget=40, on_checkpoint=lambda n, sel: {"n": int(n)},
           checkpoints=(0, 40), seed=1, refit_every=0)
    assert plain.n_refits == 0
    assert calls["n"] == plain.n_evals


def test_every_method_returns_a_feasible_selection_and_a_monotone_axis(small):
    """All eight searches, end to end: feasible output, non-decreasing evaluation count.

    ``greedy-support`` must spend **zero** evaluations -- it is the baseline the rest are read
    against, and a baseline that consulted the objective would not be one.
    """
    truth, gs, idx, obs, _ = small
    seen = {}
    for method in SE.METHODS:
        scorer = SE.Scorer(obs, idx, N_TAXA, truth.labels, maxiter=60)
        records = SE.run(method, gs, scorer, 1, budget=60, seed=2, top_b=4,
                         checkpoints=(0, 10, 30, 60),
                         on_checkpoint=lambda n, sel: {"n_evals": int(n), "sel": np.asarray(sel)})
        axis = [r["n_evals"] for r in records]
        assert axis == sorted(axis), f"{method}: the evaluation axis went backwards"
        final = records[-1]["sel"]
        feasible, _ = TR.feasible_and_cost(gs.words[final], N_TAXA) if final.size else (True, 0)
        assert feasible, f"{method} returned an infeasible cluster system"
        seen[method] = scorer.n_evals
    assert seen["greedy-support"] == 0, "the baseline consulted the objective"
    assert max(seen.values()) > 0, "no method spent an evaluation, so nothing was searched"


def test_a_beam_clone_shares_structure_and_diverges_in_state(small):
    """``SearchState.copy`` must not ``deepcopy`` the component: that is megabytes per clone.

    At n = 50 one component carries ~300,000 edges and beam100 clones a hundred of them per level,
    so the CSR and the component are shared by reference and only the mutable state is copied.  The
    guard in ``_clone_state`` is what makes that safe if ``heuristics._State`` ever grows a field;
    here the check is that the sharing is real *and* that the two states still diverge.
    """
    _, gs, _, _, _ = small
    a = SE.SearchState(gs, 1)
    legal = a.legal()
    assert len(legal) >= 2, "the fixture offers fewer than two moves, so divergence is untestable"
    b = a.copy()
    assert b.states[0].comp is a.states[0].comp, "the component was copied, not shared"
    assert b.states[0].nbr is a.states[0].nbr, "the CSR was copied, not shared"
    assert b.states[0].in_s is not a.states[0].in_s, "the mutable state is shared"

    v, klass = legal[0]
    a.accept(v, klass)
    assert a.in_s[v] and not b.in_s[v], "the clone followed the original"
    assert a.verify() and b.verify()
    assert b.selection.size == 0 and a.selection.size == 1


def test_a_search_produced_network_scores_the_same_under_the_exact_reference(small):
    """The terminal score is cross-validated on a network the *search* built, not the generator's.

    ``tests/test_mpl.py`` already pins ``TripleModel`` against ``msnc3`` on the generator's own
    networks.  What F3 scores is different: the closure of a cluster selection, which is the
    network's *regular version* (A20) and can carry shapes the generator never emits -- a hybrid
    whose cluster is a singleton, a node suppressed to a unary root.  So the agreement is pinned
    again on that family, or F3's whole y-axis rests on an untested path.
    """
    truth, gs, idx, obs, _ = small
    state = SE.SearchState(gs, 1)
    for v in np.argsort(-np.asarray(gs.support, dtype=float), kind="stable"):
        ok, klass = state.can_accept(int(v))
        if ok:
            state.accept(int(v), klass)
    made = SE.realise(gs.words, state.selection, N_TAXA, truth.labels)
    assert made is not None, "greedy-by-support stood for no network, so nothing was scored"
    net, _ = made
    assert len(net.retics) > 0, "the fixture produced a tree, so no reticulate shape was checked"

    fast = likelihood.TripleModel(net, index=idx).probabilities()
    slow = np.stack([msnc3.triple_probabilities(net, tuple(row)) for row in idx])
    assert np.abs(fast - slow).max() < 1e-10
    assert np.abs(fast.sum(axis=1) - 1.0).max() < 1e-12
