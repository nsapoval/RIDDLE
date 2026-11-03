"""W1 -- the torch MPL evaluator, against the three references RL_PLAN's G-R1c asks for.

The chain matters more than any single check.  ``opc.mpl.msnc3`` is the readable per-triple DP and
is validated against **PhyloNet** (< 1e-8, A30) and against the MSNC simulator to Monte-Carlo
noise; ``opc.mpl.likelihood`` is the shape-grouped numpy kernel and is validated against ``msnc3``;
this module is validated against both.  A port agreeing to float64 round-off inherits that chain.

RL_PLAN W1 lists a **PhyloNet cross-check** as its third validation.  That jar is on this box but
only Java 11 is (A62), and it is an offline one-time batch job rather than an inner loop (A35), so
it is not a unit test here.  What *is* here is the analytic-reduction ladder W1 also asks for,
which needs no external tool and catches exactly the convention errors a PhyloNet diff would.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opc.mpl import likelihood, msnc3, trinets, triples  # noqa: E402
from opc.rl import mpl_torch as MT  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

TOL = 1e-12


def _net(n, seed, *, r=2, cu=0.5, gamma=0.35):
    return N.with_gamma(N.scaled(N.with_reticulations(N.build_base(n, r_max=r, seed=seed), r), cu),
                        gamma)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("n,seed", [(6, 11), (8, 12), (9, 13), (11, 14)])
def test_matches_the_numpy_kernel(n, seed):
    net = _net(n, seed)
    idx = triples.triple_index(n)
    model = likelihood.TripleModel(net, index=idx)
    bm = MT.BatchedTripleModel([model])
    ell, gam = bm.start_vectors([model])
    got = bm.probabilities(ell, gam).squeeze(0).numpy()
    want = model.probabilities()
    # Vacuity guard: a batch of pure-tree shapes would test none of the reticulate transitions.
    assert any(not g.shape.is_tree for g in model.groups), "fixture induced no reticulate trinet"
    assert np.abs(got - want).max() < TOL
    assert np.abs(got.sum(axis=1) - 1.0).max() < 1e-9


def test_matches_msnc3_the_independent_reference():
    """``msnc3`` shares no code with the shape-grouped kernel; it is the per-triple DP."""
    net = _net(8, 21)
    idx = triples.triple_index(8)
    model = likelihood.TripleModel(net, index=idx)
    bm = MT.BatchedTripleModel([model])
    got = bm.probabilities(*bm.start_vectors([model])).squeeze(0).numpy()
    want = msnc3.all_triple_probabilities(net, index=idx)
    assert np.abs(got - want).max() < 1e-10


def test_batches_networks_and_pools_shared_shapes():
    n = 9
    idx = triples.triple_index(n)
    nets = [_net(n, 100 + i, gamma=0.3 + 0.05 * i) for i in range(5)]
    models = [likelihood.TripleModel(x, index=idx) for x in nets]
    bm = MT.BatchedTripleModel(models)
    got = bm.probabilities(*bm.start_vectors(models)).numpy()
    want = np.stack([m.probabilities() for m in models])
    assert np.abs(got - want).max() < TOL
    # Pooling is the point: the batch must have fewer groups than the sum of the per-network ones,
    # or the shape key is not doing the work the batching relies on.
    assert len(bm.groups) < sum(m.n_shapes for m in models)


def test_log_likelihood_matches_and_obs_broadcasts():
    n = 8
    idx = triples.triple_index(n)
    models = [likelihood.TripleModel(_net(n, 300 + i), index=idx) for i in range(3)]
    bm = MT.BatchedTripleModel(models)
    obs = np.zeros((len(idx), 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 700, 200, 100
    got = bm.log_likelihood(torch.as_tensor(obs), *bm.start_vectors(models)).numpy()
    want = np.array([m.log_likelihood(obs) for m in models])
    assert np.abs(got - want).max() < 1e-8


def test_a_ragged_batch_is_refused():
    a = likelihood.TripleModel(_net(7, 5), index=triples.triple_index(7))
    b = likelihood.TripleModel(_net(8, 5), index=triples.triple_index(8))
    with pytest.raises(ValueError, match="share one triple index"):
        MT.BatchedTripleModel([a, b])


# --------------------------------------------------------------------------- gradients


def test_autograd_matches_a_central_difference():
    """The point of the port: an analytic gradient where scipy took ``n_params + 1`` calls."""
    n = 8
    idx = triples.triple_index(n)
    models = [likelihood.TripleModel(_net(n, 400 + i), index=idx) for i in range(2)]
    bm = MT.BatchedTripleModel(models)
    obs = np.zeros((len(idx), 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 600, 250, 150
    ot = torch.as_tensor(obs)
    ell0, gam0 = bm.start_vectors(models)
    ell, gam = ell0.clone().requires_grad_(True), gam0.clone().requires_grad_(True)
    bm.log_likelihood(ot, ell, gam).sum().backward()

    def fd(kind, i, h=1e-6):
        vals = []
        for s in (+1, -1):
            e, g = ell0.clone(), gam0.clone()
            (e if kind == "l" else g)[i] += s * h
            vals.append(float(bm.log_likelihood(ot, e, g).sum()))
        return (vals[0] - vals[1]) / (2 * h)

    rng = np.random.default_rng(0)
    assert gam.grad.numel() > 0, "fixture has no reticulation, so gamma's gradient is untested"
    for i in rng.choice(ell.grad.numel(), size=6, replace=False):
        a = float(ell.grad[i])
        assert abs(fd("l", int(i)) - a) / max(1.0, abs(a)) < 1e-5
    for i in range(gam.grad.numel()):
        a = float(gam.grad[i])
        assert abs(fd("g", int(i)) - a) / max(1.0, abs(a)) < 1e-5


# --------------------------------------------------------------------------- analytic reductions


def _tree_trinet():
    """A 3-leaf *tree* trinet, and which of its edges is the internal one."""
    net = N.build_base(5, r_max=0, seed=77)
    for row in triples.triple_index(5):
        tri = trinets.restrict_to_triple(net, tuple(row))
        if tri.is_tree and len(tri.leaf_of) == 3:
            internal = [e for e in sorted(tri.edge_sources) if e[0] not in tri.leaf_of]
            if len(internal) == 1:
                return tri, tuple(sorted(tri.edge_sources)), internal[0]
    raise AssertionError("no tree trinet with exactly one internal edge in the fixture")


@pytest.mark.parametrize("b", [0.05, 0.5, 1.0, 3.0])
def test_k0_reduces_to_the_mp_est_triple_probability(b):
    """``P(match) = 1 - (2/3) e^{-b}`` -- W1's first analytic reduction, and the MSC's own."""
    tri, keys, internal = _tree_trinet()
    lengths = torch.zeros(1, len(keys), dtype=torch.float64)
    lengths[0, keys.index(internal)] = b
    p = MT.shape_probabilities_torch(tri, keys, lengths, torch.zeros(1, 0, dtype=torch.float64))
    match = 1.0 - (2.0 / 3.0) * np.exp(-b)
    assert abs(float(p.max()) - match) < 1e-12
    others = sorted(float(x) for x in p[0])[:2]
    assert all(abs(o - np.exp(-b) / 3.0) < 1e-12 for o in others)


def test_lambda_to_zero_gives_one_third_each():
    tri, keys, _ = _tree_trinet()
    p = MT.shape_probabilities_torch(tri, keys, torch.zeros(1, len(keys), dtype=torch.float64),
                                     torch.zeros(1, 0, dtype=torch.float64))
    assert np.abs(p.numpy() - 1.0 / 3.0).max() < 1e-14


def test_lambda_to_infinity_gives_the_tree_like_limit():
    tri, keys, internal = _tree_trinet()
    lengths = torch.zeros(1, len(keys), dtype=torch.float64)
    lengths[0, keys.index(internal)] = likelihood.MAX_LENGTH
    p = MT.shape_probabilities_torch(tri, keys, lengths, torch.zeros(1, 0, dtype=torch.float64))
    assert float(p.max()) > 1.0 - 1e-8


@pytest.mark.parametrize("take_donor,g", [(True, 1.0), (False, 0.0)])
def test_gamma_at_an_endpoint_reduces_to_the_displayed_tree(take_donor, g):
    """gamma in {0, 1} must give the displayed tree *exactly* -- and the interval is closed."""
    net = _net(8, 55, r=1)
    idx = triples.triple_index(8)
    model = likelihood.TripleModel(net, index=idx)
    bm = MT.BatchedTripleModel([model])
    ell, gam = bm.start_vectors([model])
    assert gam.numel() == 1, "fixture must carry exactly one reticulation"
    got = bm.probabilities(ell, torch.full_like(gam, g)).squeeze(0).numpy()
    displayed = msnc3.resolve_one_reticulation(net, 0, take_donor)
    want = msnc3.all_triple_probabilities(displayed, index=idx)
    assert np.abs(got - want).max() < 1e-10


# --------------------------------------------------------------------------- profiling


def test_profile_mpl_reaches_the_scipy_optimum_and_reports_its_restarts():
    """Projected Adam against ``likelihood.optimise``, on a surface with real data behind it."""
    from opc.simulate import msnc_oracle as oracle

    n = 8
    idx = triples.triple_index(n)
    truth = _net(n, 909, r=1, gamma=0.4)
    tc = triples.TripleCounts.empty(n)
    tax = {lab: i for i, lab in enumerate(truth.labels)}
    for nw in oracle.simulate_gene_trees(truth, 400, seed=3):
        triples.count_tree(nw, tax, tc)
    _, obs = tc.as_rows()
    obs = obs.astype(float)

    models = [likelihood.TripleModel(x, index=idx)
              for x in (truth, _net(n, 910, r=1), _net(n, 911, r=1))]
    ref = [likelihood.optimise(m, obs, maxiter=400, restarts=4, seed=0)[1] for m in models]
    got = MT.profile_mpl(models, obs, restarts=3, iters=900, lr=0.05, seed=0)

    v = got.value.numpy()
    # Adam is allowed to be a little behind L-BFGS-B on an absolute scale; what it may not do is
    # reorder the candidates, because the reward is only ever used comparatively.
    assert np.argsort(-v).tolist() == np.argsort(-np.asarray(ref)).tolist()
    assert (v - np.asarray(ref)).max() < 1.0
    assert np.abs(v - np.asarray(ref)).max() < 0.02 * abs(float(np.mean(ref)))
    assert got.restart_gain.min() >= 0.0
    assert got.lengths.numel() == sum(len(m.params.edges) for m in models)
    assert got.gammas.numel() == sum(len(m.params.hybrids) for m in models)


def test_mom_init_is_a_function_of_the_data_alone():
    """RL_PLAN §1.2 wants one start for every rollout in a group, from ``rho`` and nothing else."""
    n = 8
    idx = triples.triple_index(n)
    obs = np.zeros((len(idx), 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 800, 120, 80
    a = MT.mom_init_length([likelihood.TripleModel(_net(n, 1), index=idx)], obs)
    b = MT.mom_init_length([likelihood.TripleModel(_net(n, 2), index=idx)], obs)
    assert a == b > 0.0


def test_tail_movement_is_actually_measured():
    """A vacuity guard, and it caught a real defect.

    On the CUDA-graph path ``val`` is the *same tensor* on every replay, so tracking
    ``|v - prev|`` without copying compares a tensor with itself and reports ``0.0`` -- which reads
    as "converged" and means "never compared". The probe printed exactly 0.0 on both arms before
    this was found. The assertion is therefore that the number is **non-zero** on a schedule short
    enough not to have converged, not that it is small.
    """
    n = 8
    idx = triples.triple_index(n)
    models = [likelihood.TripleModel(_net(n, 600 + i, r=1), index=idx) for i in range(3)]
    obs = np.zeros((len(idx), 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 500, 300, 200
    short = MT.profile_mpl(models, obs, restarts=1, iters=40, lr=0.1, seed=0)
    assert float(short.tail_movement.max()) > 0.0
    assert torch.isfinite(short.tail_movement).all()
    # and it must shrink as the schedule lengthens, or it is measuring noise rather than progress
    long = MT.profile_mpl(models, obs, restarts=1, iters=600, lr=0.1, seed=0)
    assert float(long.tail_movement.max()) < float(short.tail_movement.max())


def test_restart_gain_is_non_negative_and_reported_per_network():
    models = [likelihood.TripleModel(_net(8, 700 + i, r=1), index=triples.triple_index(8))
              for i in range(3)]
    obs = np.zeros((56, 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 400, 350, 250
    got = MT.profile_mpl(models, obs, restarts=3, iters=120, seed=1)
    assert got.restart_gain.shape == (3,)
    assert (got.restart_gain >= 0).all(), "a restart cannot make the best value worse"
    assert got.value.shape == (3,)


def test_batched_lbfgs_is_per_network_exact():
    """Batching L-BFGS is only legitimate if it equals N separate runs; this is that check.

    The parameter blocks are disjoint, so optimising three networks together must give each of them
    exactly what optimising it alone gives. If any inner product in the two-loop recursion were a
    *global* dot rather than a per-network segment sum, one network's curvature would steer
    another's step and this would fail.
    """
    n = 8
    idx = triples.triple_index(n)
    models = [likelihood.TripleModel(_net(n, 500 + i, r=1), index=idx) for i in range(3)]
    obs = np.zeros((len(idx), 3))
    obs[:, 0], obs[:, 1], obs[:, 2] = 650, 200, 150
    together = MT.profile_mpl_lbfgs(models, obs, restarts=1, iters=80, seed=0).value.numpy()
    alone = np.array([float(MT.profile_mpl_lbfgs([m], obs, restarts=1, iters=80, seed=0).value[0])
                      for m in models])
    assert np.abs(together - alone).max() < 1e-8


def test_batched_lbfgs_respects_the_box():
    n = 8
    idx = triples.triple_index(n)
    models = [likelihood.TripleModel(_net(n, 510 + i, r=1), index=idx) for i in range(2)]
    obs = np.zeros((len(idx), 3))
    obs[:, 0] = 1000  # a degenerate table: pushes lengths to a bound
    got = MT.profile_mpl_lbfgs(models, obs, restarts=1, iters=60, seed=0)
    assert float(got.lengths.min()) >= likelihood.MIN_LENGTH - 1e-12
    assert float(got.lengths.max()) <= likelihood.MAX_LENGTH + 1e-12
    assert float(got.gammas.min()) >= likelihood.MIN_GAMMA - 1e-12
    assert float(got.gammas.max()) <= likelihood.MAX_GAMMA + 1e-12


def test_batched_lbfgs_reaches_the_scipy_optimum_in_far_fewer_iterations():
    from opc.simulate import msnc_oracle as oracle

    n = 8
    idx = triples.triple_index(n)
    truth = _net(n, 919, r=1, gamma=0.4)
    tc = triples.TripleCounts.empty(n)
    tax = {lab: i for i, lab in enumerate(truth.labels)}
    for nw in oracle.simulate_gene_trees(truth, 400, seed=5):
        triples.count_tree(nw, tax, tc)
    _, obs = tc.as_rows()
    obs = obs.astype(float)
    models = [likelihood.TripleModel(x, index=idx)
              for x in (truth, _net(n, 920, r=1), _net(n, 921, r=1))]
    ref = np.array([likelihood.optimise(m, obs, maxiter=400, restarts=4, seed=0)[1]
                    for m in models])
    got = MT.profile_mpl_lbfgs(models, obs, restarts=2, iters=120, seed=0).value.numpy()
    assert np.argsort(-got).tolist() == np.argsort(-ref).tolist()
    assert (got - ref).max() < 1e-3, "L-BFGS should not overshoot scipy's optimum"

    # The threshold that matters is **not** an absolute number of log units, it is the shortfall
    # against the gaps the reward has to resolve: a group-normalised advantage is a ranking, so a
    # truncation is harmless exactly while it is small next to the spacing between candidates.
    # Measured here at 120 iterations: two of three match scipy to 4e-8 and the third is 1.98 short
    # on -21,575, against an adjacent-candidate gap of ~2,059 -- a tenth of a percent of it.
    short = float((ref - got).max())
    gaps = np.diff(np.sort(ref))
    assert short < 0.05 * float(gaps.min()), (
        f"shortfall {short:.3f} is not small against the smallest candidate gap {gaps.min():.1f}")


# --------------------------------------------------------------------------- fitted parameters


def test_fit_parameters_gives_each_network_its_own_block():
    """The layout contract, cross-checked against the numpy kernel rather than against itself.

    ``fit_parameters`` slices one concatenated vector back into per-network :class:`Parameters` by
    walking ``models`` in order. An off-by-one there would write network *i*'s branch lengths onto
    network *j*'s topology and the result would still be a valid network with a plausible
    distance -- which is the whole reason A80's writeback needs this pinned. So the check is not
    that the shapes line up: it is that **evaluating each returned block through the independent
    numpy kernel reproduces the value the batched optimiser reported for that network**, which a
    mis-slice cannot survive.
    """
    nets = [_net(8, 21, r=1), _net(8, 22, r=2), _net(9, 23, r=2)]
    idx = triples.triple_index(8)
    # Deliberately unequal block sizes: with equal ones a shifted slice would be undetectable.
    models = [likelihood.TripleModel(net, index=triples.triple_index(net.n_taxa)) for net in nets]
    sizes = {(len(m.params.edges), len(m.params.hybrids)) for m in models}
    assert len(sizes) == len(models), f"blocks are not distinguishable by size: {sizes}"
    del idx

    per_n = {}
    for net, model in zip(nets, models, strict=True):
        per_n.setdefault(net.n_taxa, []).append(model)
    for n_taxa, group in per_n.items():
        if len(group) < 2:
            continue
        obs = np.stack([np.asarray(m.probabilities()) * 100.0 for m in group])
        params, res = MT.fit_parameters(group, obs, restarts=1, iters=40, graph=False)
        assert [p.edges for p in params] == [m.params.edges for m in group]
        assert [p.hybrids for p in params] == [m.params.hybrids for m in group]
        for b, (m, p) in enumerate(zip(group, params, strict=True)):
            got = m.log_likelihood(obs[b], p)
            want = float(res.value[b])
            assert abs(got - want) < 1e-6, (
                f"network {b} of {n_taxa} taxa: numpy scores the returned block at {got:.6f} "
                f"where the batched fit reported {want:.6f} -- the slice is wrong")


def test_fit_parameters_improves_on_the_start_it_is_given():
    """A writeback that returned the start would leave ``normwapd`` measuring the placeholders.

    ``network_from_hasse``'s rank-time lengths are the exact input A80's writeback exists to
    replace, so "the optimiser ran" is not the property worth asserting -- "the optimiser moved
    the value" is.
    """
    net = _net(9, 31, r=2)
    model = likelihood.TripleModel(net, index=triples.triple_index(9))
    # Counts drawn from a **different** parameter vector, or the network's own lengths are already
    # the maximum and "did the optimiser move it" is unanswerable -- the vacuous-fixture trap.
    truth = likelihood.Parameters(model.params.edges, model.params.hybrids,
                                  model.params.lengths * 2.5,
                                  np.full(model.params.gammas.size, 0.2))
    obs = np.asarray(model.probabilities(truth)) * 250.0
    start = model.log_likelihood(obs, model.params)
    assert start < model.log_likelihood(obs, truth), "the fixture start is already optimal"
    params, res = MT.fit_parameters([model], obs, restarts=2, iters=200, graph=False)
    assert float(res.value[0]) > start + 1e-6, "the fit did not improve on its own start"
    assert abs(model.log_likelihood(obs, params[0]) - float(res.value[0])) < 1e-6
