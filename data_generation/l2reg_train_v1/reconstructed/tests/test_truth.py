"""Objective-vs-truth alignment: the metric itself, on instances whose answer is known.

The load-bearing case is the *clean* one -- ground set exactly ``C(N)``, budget exactly ``r``.
There the solver must return the whole truth, so F1 is 1, the regret is 0 and ``rho`` is ``r``.
If any of those is off, the metric is wrong and every number it produces on filtered instances is
uninterpretable.  Then the filtered case, where the point is the opposite: ``S_true`` is a strict
subset of ``C(N)``, still feasible, and still a legal competitor.
"""

from __future__ import annotations

import numpy as np
import pytest

from opc import bitset
from opc.simulate import networks as N
from opc.solve import ilp, overlap, pipeline, truth


def _clean_instance(n_taxa: int, r: int, seed: int):
    """``(net, words)`` where ``words`` is exactly the non-trivial ``C(N)``."""
    base = N.build_base(n_taxa, r, seed=seed)
    net = N.with_gamma(N.scaled(N.with_reticulations(base, r), 1.0), 0.3)
    cs = sorted(N.nontrivial_clusters(net), key=sorted)
    words = np.stack([bitset.from_indices(sorted(c), n_taxa) for c in cs])
    return net, words


# --------------------------------------------------------------------------------- clean input


@pytest.mark.parametrize("n_taxa,r,seed", [(8, 1, 21), (10, 2, 5), (12, 2, 402)])
def test_the_truth_is_the_optimum_when_the_ground_set_is_exactly_the_truth(n_taxa, r, seed):
    """T2 restated as an alignment claim: on clean input the objective *does* point at truth.

    Any weight vector works here -- the whole ground set is feasible, so accepting everything is
    optimal -- which is what makes this a test of the metric rather than of the weights.
    """
    net, words = _clean_instance(n_taxa, r, seed)
    w = np.ones(words.shape[0])
    sol = pipeline.solve_ground_set(words, n_taxa, w, r, time_limit=30)
    got = truth.align(words, sol.selected, w, net, n_taxa)

    assert got.n_true == words.shape[0] == got.n_true_in_ctau
    assert got.f1 == pytest.approx(1.0)
    assert got.regret == pytest.approx(0.0)
    assert got.true_feasible
    assert got.rho_selected == r == got.rho_true
    assert got.cluster_distance == 0
    assert got.level1_ok
    assert all(rep.feasible_ok for rep in sol.components)
    assert sol.j_used == got.rho_selected, "T4a: sum_K j_K must equal rho(I(S*))"


@pytest.mark.parametrize("n_taxa,r,seed", [(10, 2, 5), (12, 3, 77)])
def test_the_true_system_is_feasible_and_costs_exactly_its_reticulations(n_taxa, r, seed):
    """(L) and (N3O) are hereditary, so every subset of ``C(N)`` is a legal competitor.

    Checked rather than assumed, and checked by the validator in ``heuristics`` rather than by
    the theory -- a filtered ``S_true`` that came back infeasible would mean the comparison in
    ``align`` is against something the solver was never allowed to return.
    """
    net, words = _clean_instance(n_taxa, r, seed)
    ok, j = truth.feasible_and_cost(words, n_taxa)
    assert ok
    assert j == truth.reticulations_of(words, n_taxa) == r

    rng = np.random.default_rng(seed)
    subsets = 0
    for _ in range(25):
        keep = np.flatnonzero(rng.random(words.shape[0]) < 0.6)
        if keep.size < 2 or keep.size == words.shape[0]:
            continue
        sub_ok, _ = truth.feasible_and_cost(np.ascontiguousarray(words[keep]), n_taxa)
        assert sub_ok, "a subset of a feasible system must stay feasible"
        subsets += 1
    assert subsets > 10, "fixture produced no proper subsets -- heredity is untested"


# ------------------------------------------------------------------------------ filtered input


def test_a_filtered_ground_set_separates_recall_from_the_recoverable_ceiling():
    """With true clusters missing and false ones present, the two recalls must come apart.

    ``recall`` is against all of ``C(N)`` and is capped by what filtering left behind;
    ``recall_recoverable`` is against ``C(N) & C_tau`` and is the one that says whether the
    *solver* lost anything.  Reporting only the first would blame the objective for the threshold.
    """
    n_taxa, r = 10, 2
    net, true_words = _clean_instance(n_taxa, r, seed=5)
    rng = np.random.default_rng(3)

    keep = np.array([0, 2, 3, 4], dtype=np.int64)
    noise = []
    while len(noise) < 3:
        cand = frozenset(rng.choice(n_taxa, size=int(rng.integers(2, n_taxa - 1)),
                                    replace=False).tolist())
        if cand not in N.nontrivial_clusters(net):
            noise.append(bitset.from_indices(sorted(cand), n_taxa))
    words = np.vstack([true_words[keep], np.stack(noise)])

    rows = truth.true_cluster_rows(words, net)
    assert rows.size == keep.size < len(N.nontrivial_clusters(net)), "fixture must lose truth"

    # Accept the true rows only: perfect against the ceiling, short against the whole truth.
    got = truth.align(words, rows, np.ones(words.shape[0]), net, n_taxa)
    assert got.recall_recoverable == pytest.approx(1.0)
    assert got.recall < 1.0
    assert got.precision == pytest.approx(1.0)
    assert got.n_true_in_ctau == keep.size


def test_regret_is_positive_exactly_when_the_optimum_outscores_the_truth():
    """The misalignment signature, constructed.

    Weight the noise clusters heavily and the truth lightly: the optimum then prefers the noise,
    F1 collapses, and the regret is positive.  That combination is what would say the objective,
    not the solver, is the thing to fix.
    """
    n_taxa = 8
    net, true_words = _clean_instance(n_taxa, 1, seed=21)
    decoy = bitset.from_indices([0, 1], n_taxa)
    have = {bytes(r) for r in np.ascontiguousarray(true_words)}
    assert bytes(decoy) not in have, "fixture decoy must not itself be a true cluster"

    words = np.vstack([true_words, decoy[None, :]])
    w = np.r_[np.ones(true_words.shape[0]), 100.0]
    dec = overlap.decompose(words, n_taxa)
    selected = np.array(sorted(
        list(dec.kernel)
        + [int(c.vertices[a]) for c in dec.components
           for a in ilp.solve_component(c, w[c.vertices], 1, time_limit=20).accepted]
    ), dtype=np.int64)

    got = truth.align(words, selected, w, net, n_taxa)
    assert got.w_selected >= got.w_true_feasible
    assert got.regret > 0, "the decoy should outscore what it displaces"
    assert got.precision < 1.0
