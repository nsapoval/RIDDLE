"""F3 -- unlearned search over cluster-system space, and what one scored candidate really costs.

``followup-plan-v0.3.md`` §F3, as revised by ``results/FOLLOWUP_v0.3.md`` §6.3.  H3 is the last
open branch of the follow-up's decision table: H1 is answered no (F0), H2 no on both routes tried
(F1, F2), and H4 withdrawn with F7.  §4's combination rule makes ``not H1 and not H3`` the verdict
"neither exact nor learned search works on this formulation; the objective or the search space
changes", so this module is what stands between the study and that reading.

**What is measured.**  Cluster F1 and MPL regret against the *number of MPL evaluations*, over the
same feasible set the ILP optimises.  Nothing in :mod:`opc.solve` counts evaluations --
``heuristics.HeuristicResult`` records ``seconds`` and ``sweeps`` -- and that count is F3's whole
x-axis, so :class:`Scorer` owns it and every searcher goes through it rather than reconstructing
it from wall-clock afterwards.

**Three things the plan's cost model got wrong, and what this module does about them.**

*The per-candidate floor is not 15 ms.*  ``likelihood.TripleModel.__init__`` calls
``trinets.restrict_to_triple`` once per triple -- 19,600 times at n = 50, in Python, ~2.0 s -- and
there is no cache anywhere in :mod:`opc.mpl`.  A search proposes a new topology per candidate, so
it pays that per *candidate*, not per evaluation, and §6.3's "10^5 evaluations ~ 25 min per run"
is out by two orders of magnitude.  The lever is ``TripleModel(net, index=...)``, which already
takes a triple subsample; :func:`subsample_rows` builds one and ``scripts/probe_f3_cost.py``
measures how far it can be pushed before the *ranking* degrades.

*The ratio §F3 asks for is not ~10x.*  A fixed-parameter evaluation is 7-9 ms at k in {1, 2}; one
re-optimisation is 5,000-11,000 of those calls, because L-BFGS-B takes numerical gradients over
52-56 bounded parameters.  :meth:`Scorer.refit` reports the iteration and call counts, which is
the quantity that governs an RL budget -- §F3 says so, and it is not the evaluation count.

*Cluster F1 >= 0.9 is not a discriminating threshold.*  Greedy-by-support reaches 0.909 at **zero**
MPL evaluations (E1b) and the converged support tiebreak 0.969 (A51), so a curve can clear §F3's
pre-registered band without the objective doing any work -- A46's failure mode in a new place.
``greedy-support`` is therefore a method here, run on the same cell, and ``mpl_regret`` is
recorded beside ``align_f1``.

**Parameters in the inner loop.**  ``closure.network_from_hasse`` sets node times to a rank and
gamma to 0.5, and its docstring says both are placeholders -- gamma is *not identifiable from
clusters at all*.  Ranking there would rank on a signal with no fitted content.  What makes a
fitted signal available cheaply is that the Hasse diagram's nodes **are** clusters:
``network_from_hasse`` numbers them by row index into the closed word array, and
``Parameters.of`` emits ``edges`` as ``(child_node, parent_node)`` pairs, so every branch length is
keyed by a *(child cluster, parent cluster) bitmask pair* and every gamma by the hybrid's own
cluster.  :class:`Fit` transfers a checkpoint's re-optimised parameters onto a new topology
*exactly* wherever the clusters coincide, and reports the hit rate, so a mostly-defaulted transfer
cannot be mistaken for a fitted one.

**What is deliberately reused rather than rewritten.**  ``heuristics._State`` is the exact action
mask -- its own docstring says it exists so "a policy would inherit a checked implementation
rather than write a second one" -- and every accepted state is re-validated with
``heuristics.check``, which re-derives (L) and (N3O) from scratch.  That is the house rule that
nothing is checked against itself, applied to the move loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from opc.mpl import likelihood
from opc.solve import closure, heuristics

__all__ = [
    "DEFAULT_LENGTH", "DEFAULT_GAMMA", "CHECKPOINTS", "METHODS",
    "subsample_rows", "realise", "Fit", "Scorer", "SearchState", "run",
]

#: What an edge with no counterpart in the previous fit is scored at.  One coalescent unit is the
#: grid's own central ILS level (``grid.CU_VALUES``), so it is the least informative choice that is
#: still on the scale of the data rather than at a bound.
DEFAULT_LENGTH = 1.0

#: ``closure.network_from_hasse``'s own default, and the only defensible one: gamma is not
#: identifiable from a clustering system, so a *new* hybrid edge carries no information about it.
DEFAULT_GAMMA = 0.5

#: Geometric, so the curve is read on a log x-axis -- §F3's bands are 10^4 and 10^6.  Clipped to
#: the run's budget by :func:`run`.
CHECKPOINTS = (0, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)

#: Every search this module implements.  ``greedy-support`` spends no MPL evaluations at all and is
#: the baseline the rest are read against (E1b: cluster F1 0.909; A51: 0.969 converged).
METHODS = (
    "greedy-support",
    "greedy-mpl",
    "greedy-mpl-topb",
    "beam1",
    "beam10",
    "beam100",
    "random",
    "anneal",
)


def subsample_rows(n_triples: int, size: int, seed: int) -> np.ndarray | None:
    """``size`` triple rows without replacement, sorted; ``None`` means keep all of them.

    A subsample makes the inner-loop score a *stochastic* estimate of MPL.  What licenses it is
    rank fidelity, not unbiasedness -- a search only ever compares candidates -- and that is what
    ``scripts/probe_f3_cost.py`` measures before a grid is allowed to run.
    """
    if size <= 0 or size >= n_triples:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_triples, size=int(size), replace=False)).astype(np.int64)


def realise(words: np.ndarray, selection, n_taxa: int, labels=None):
    """``(net, closed_words)`` for a cluster selection, or ``None`` if it stands for no network.

    The closure chain ``closure_words -> with_trivial_clusters -> network_from_hasse``, lifted out
    of ``scripts/e4_exact.py:realise`` and ``scripts/e1b_objective.py`` which both inline it.
    ``network_from_hasse`` raises :class:`ValueError` on three distinct malformations -- more than
    one maximal element, a cluster with more than two covering parents, both covering parents
    already donating -- and a random or annealed selection hits all three routinely, so this
    returns ``None`` rather than making every searcher carry the same ``try``.

    ``closed_words`` is returned because it is what makes parameter transfer possible: the
    network's node ids *are* rows of it (``closure.network_from_hasse``).
    """
    sel = np.asarray(selection, dtype=np.int64)
    chosen = np.ascontiguousarray(np.asarray(words)[sel])
    try:
        # ``closure_words`` cannot stack an empty family, and the empty selection is not an error:
        # it is the star, whose likelihood is ``n_triples * m * ln(1/3)`` -- A49's all-star
        # incumbent, and the honest score of "nothing chosen".  Every add-only searcher starts
        # there when the kernel is empty, which it is at small ``n``.
        closed = closure.closure_words(chosen, n_taxa) if chosen.shape[0] else chosen
        full = closure.with_trivial_clusters(closed, n_taxa)
        net = closure.network_from_hasse(full, n_taxa, labels=labels)
    except ValueError:
        return None
    return net, full


def _length_keys(params: likelihood.Parameters, words: np.ndarray) -> list[tuple[bytes, bytes]]:
    """One key per parameter edge: the (child cluster, parent cluster) bitmask pair."""
    return [(words[c].tobytes(), words[p].tobytes()) for c, p in params.edges]


def _gamma_keys(params: likelihood.Parameters, words: np.ndarray) -> list[bytes]:
    return [words[h].tobytes() for h in params.hybrids]


@dataclass
class Fit:
    """Re-optimised parameters, keyed by cluster rather than by node id.

    Node ids are row indices into whichever closed word array produced the network, so they mean
    nothing across topologies; the clusters they stand for mean the same thing everywhere.  That is
    what makes the transfer exact rather than approximate on the edges it covers.
    """

    lengths: dict[tuple[bytes, bytes], float] = field(default_factory=dict)
    gammas: dict[bytes, float] = field(default_factory=dict)
    log_likelihood: float = float("-inf")

    @classmethod
    def of(cls, params: likelihood.Parameters, words: np.ndarray, log_likelihood: float) -> Fit:
        return cls(
            lengths=dict(zip(_length_keys(params, words), (float(x) for x in params.lengths))),
            gammas=dict(zip(_gamma_keys(params, words), (float(x) for x in params.gammas))),
            log_likelihood=float(log_likelihood),
        )

    def vector_for(self, params: likelihood.Parameters, words: np.ndarray):
        """``(x, hits, total)``: a vector in ``params``' own order, and how much of it was fit."""
        lk, gk = _length_keys(params, words), _gamma_keys(params, words)
        xs = [self.lengths.get(k, DEFAULT_LENGTH) for k in lk]
        xs += [self.gammas.get(k, DEFAULT_GAMMA) for k in gk]
        hits = sum(k in self.lengths for k in lk) + sum(k in self.gammas for k in gk)
        bounds = params.bounds()
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        return np.clip(np.asarray(xs, dtype=float), lo, hi), hits, len(lk) + len(gk)


@dataclass
class Refit:
    """One checkpoint re-optimisation, with the two counts §F3 asks for."""

    log_likelihood: float
    n_calls: int
    n_iters: int
    converged: bool
    seconds: float
    n_params: int
    transfer_hits: int
    transfer_total: int


class Scorer:
    """The objective, and the counters that are F3's x-axis.

    ``score`` is the inner loop: a fixed-parameter log-likelihood on the triple subsample, at
    parameters transferred from the last :meth:`refit`.  ``refit`` is the outer loop: L-BFGS-B over
    every branch length and gamma on *all* triples, which is what "true MPL with branch lengths and
    gamma re-optimised" means in §F3 and what the recorded scores are.

    ``refit`` calls ``scipy.optimize.minimize`` directly rather than ``likelihood.optimise``, for
    one reason: ``optimise`` returns ``(params, value)`` and drops ``res.nit`` / ``res.nfev`` /
    ``res.status``, and the iteration count is the number §F3 says governs the budget.  The
    settings are ``optimise``'s own, and ``tests/test_followup.py`` pins the agreement between the
    two on a fixture -- the landed path stays the oracle.
    """

    def __init__(self, obs, index, n_taxa: int, labels=None, *, search_rows=None,
                 maxiter: int = 150):
        self.obs_full = np.asarray(obs)
        self.index_full = np.asarray(index)
        self.n_taxa = int(n_taxa)
        self.labels = None if labels is None else list(labels)
        self.maxiter = int(maxiter)
        self.rows = None if search_rows is None else np.asarray(search_rows, dtype=np.int64)
        if self.rows is None:
            self.index_search, self.obs_search = self.index_full, self.obs_full
        else:
            self.index_search = self.index_full[self.rows]
            self.obs_search = self.obs_full[self.rows]
        self.fit: Fit | None = None
        self._fit_at: int | None = None   # |selection| when ``fit`` was taken

        # -- counters.  ``n_evals`` is the x-axis and counts *scored* candidates: a selection that
        #    stands for no network never reaches the likelihood, so it is counted separately rather
        #    than inflating the axis with work the objective never saw.
        self.n_evals = 0
        self.n_infeasible = 0
        self.n_build_failures = 0
        self.eval_seconds = 0.0
        self.build_seconds = 0.0
        self.closure_seconds = 0.0
        self.n_refits = 0
        self.refit_seconds = 0.0
        self.n_refit_calls = 0
        self.transfer_hits = 0
        self.transfer_total = 0

    @property
    def n_triples_search(self) -> int:
        return int(self.index_search.shape[0])

    def _model(self, net, index):
        """``TripleModel`` or ``None``.

        ``TripleModel.__init__`` raises ``RuntimeError`` when a restriction puts a pendant length
        where the DP cannot use it.  On the grid's own networks that never fires; on a network read
        off an arbitrary cluster selection it can, and it is a property of the candidate, not a bug
        to crash the run with.
        """
        try:
            return likelihood.TripleModel(net, index=index)
        except RuntimeError:
            return None

    def _transfer(self, model, words):
        """A ``Parameters`` for ``model`` from the last fit; ``model.params`` if there is none."""
        if self.fit is None:
            return model.params, 0, len(model.params.edges) + len(model.params.hybrids)
        x, hits, total = self.fit.vector_for(model.params, words)
        return model.params.with_vector(x), hits, total

    def score(self, gs, selection) -> float:
        """Inner-loop score of a cluster selection.  ``-inf`` when it stands for no network."""
        t0 = time.perf_counter()
        made = realise(gs.words, selection, self.n_taxa, self.labels)
        self.closure_seconds += time.perf_counter() - t0
        if made is None:
            self.n_infeasible += 1
            return float("-inf")
        net, words = made
        t1 = time.perf_counter()
        model = self._model(net, self.index_search)
        self.build_seconds += time.perf_counter() - t1
        if model is None:
            self.n_build_failures += 1
            return float("-inf")
        params, hits, total = self._transfer(model, words)
        self.transfer_hits += hits
        self.transfer_total += total
        t2 = time.perf_counter()
        value = model.log_likelihood(self.obs_search, params)
        self.eval_seconds += time.perf_counter() - t2
        self.n_evals += 1
        return float(value)

    def refit(self, gs, selection) -> Refit | None:
        """Full-triple re-optimisation of lengths and gamma.  Updates the transfer source."""
        made = realise(gs.words, selection, self.n_taxa, self.labels)
        if made is None:
            return None
        out = self._refit_from(*made)
        if out is not None:
            self._fit_at = int(np.asarray(selection).size)
        return out

    def _refit_from(self, net, words) -> Refit | None:
        from scipy.optimize import minimize

        model = self._model(net, self.index_full)
        if model is None:
            return None
        base, hits, total = self._transfer(model, words)
        obs = self.obs_full
        calls = 0

        def negative(x):
            nonlocal calls
            calls += 1
            return -model.log_likelihood(obs, base.with_vector(x))

        t0 = time.perf_counter()
        if base.vector.size == 0:
            # The star has no free parameter at all, so there is nothing to optimise and
            # ``minimize`` would be asked for an argmin over an empty space.
            fitted, value, nit, status = base, -negative(base.vector), 0, 0
        else:
            res = minimize(negative, base.vector.copy(), method="L-BFGS-B",
                           bounds=base.bounds(), options={"maxiter": self.maxiter})
            fitted, value = base.with_vector(res.x), float(-res.fun)
            nit, status = int(getattr(res, "nit", -1)), int(res.status)
        seconds = time.perf_counter() - t0
        self.fit = Fit.of(fitted, words, value)
        self.n_refits += 1
        self.refit_seconds += seconds
        self.n_refit_calls += calls
        return Refit(
            log_likelihood=value, n_calls=int(calls), n_iters=nit,
            converged=bool(status == 0), seconds=seconds,
            n_params=int(base.vector.size), transfer_hits=int(hits), transfer_total=int(total),
        )

    def refresh(self, gs, selection, every: int) -> bool:
        """Re-take the transfer source if the search has moved ``every`` clusters since the last.

        **Without this the inner loop ranks on a surrogate that goes stale exactly when the search
        is moving fastest.**  ``probe_f3_cost.py`` measures the transfer covering 86-96% of a
        candidate's edges -- but it re-fits *at the incumbent* before ranking.  A run that re-fits
        only at geometric checkpoints carries a fit from depth ~15 while ranking at depth 49, and
        the coverage measured in that state is **0.18-0.27**: four edges in five are then scored at
        ``DEFAULT_LENGTH`` and the ranking is mostly a statement about a constant.  The column that
        caught it is ``transfer_coverage_at``, which is why it is reported per row.

        Costs one re-optimisation per ``every`` accepted clusters -- about ten per greedy pass.
        """
        if every <= 0:
            return False
        n = int(np.asarray(selection).size)
        if self._fit_at is not None and abs(n - self._fit_at) < every:
            return False
        return self.refit(gs, selection) is not None

    def score_network(self, net, words, *, full: bool = False) -> float:
        """Score a network directly, with no ground set.

        Used by the cost probe and by ``followup.netsearch``, whose candidates are networks rather
        than cluster selections and whose clusters need not lie in any ground set.  It does **not**
        touch ``n_evals``: the caller owns the axis, because the probe scores candidates that are
        not search steps.  ``netsearch`` increments it explicitly, once per proposal scored.
        """
        model = self._model(net, self.index_full if full else self.index_search)
        if model is None:
            return float("-inf")
        params, _, _ = self._transfer(model, words)
        obs = self.obs_full if full else self.obs_search
        t0 = time.perf_counter()
        value = float(model.log_likelihood(obs, params))
        self.eval_seconds += time.perf_counter() - t0
        return value

    def refit_network(self, net, words) -> Refit | None:
        """:meth:`refit` for a network from no ground set.  Same optimiser, same counters."""
        return self._refit_from(net, words)

    def counters(self) -> dict:
        cover = self.transfer_hits / self.transfer_total if self.transfer_total else float("nan")
        return {
            "n_evals": self.n_evals,
            "n_infeasible": self.n_infeasible,
            "n_build_failures": self.n_build_failures,
            "eval_seconds": self.eval_seconds,
            "build_seconds": self.build_seconds,
            "closure_seconds": self.closure_seconds,
            "n_refits": self.n_refits,
            "refit_seconds": self.refit_seconds,
            "n_refit_calls": self.n_refit_calls,
            "transfer_coverage": cover,
            "n_triples_search": self.n_triples_search,
        }


#: Every attribute ``heuristics._State`` carries.  :func:`_clone_state` shares the first five and
#: copies the rest, so a new one appearing must fail loudly rather than be dropped from a clone.
_STATE_FIELDS = ("comp", "budget", "indptr", "nbr", "cls", "in_s", "commit", "used", "_mark")


def _clone_state(st):
    """Copy a ``heuristics._State``, sharing its immutable structure.

    ``copy.deepcopy`` duplicates the component and its CSR arrays -- at n = 50 one component
    carries 300,000 edges, so that is megabytes per clone, and beam search clones once per beam
    member per level.  Only ``in_s``, ``commit`` and ``used`` are mutable; ``_mark`` is scratch
    that ``can_accept`` clears in a ``finally``, and is remade empty rather than copied.
    """
    unexpected = set(vars(st)) - set(_STATE_FIELDS)
    if unexpected:
        raise AssertionError(
            f"heuristics._State grew {sorted(unexpected)}; the clone would silently drop them")
    clone = object.__new__(type(st))
    clone.comp, clone.budget = st.comp, st.budget
    clone.indptr, clone.nbr, clone.cls = st.indptr, st.nbr, st.cls
    clone.in_s, clone.commit = st.in_s.copy(), st.commit.copy()
    clone.used = set(st.used)
    clone._mark = np.zeros(st.comp.n_vertices, dtype=bool)  # noqa: SLF001
    return clone


class SearchState:
    """A feasible cluster selection over the *whole* ground set, with incremental moves.

    Everything in :mod:`opc.solve.heuristics` is per-:class:`~opc.solve.overlap.Component`.  A
    cluster system is global, so this owns one ``heuristics._State`` per component plus the
    cross-component reticulation budget.  That split is exact rather than a simplification: (L) and
    (N3O) decompose across components because clusters in different components do not overlap at
    all (``heuristics``' module docstring), and the hybrid count is additive, so
    ``sum_components len(used) <= k`` is the whole of the global constraint.

    **The kernel is accepted up front**, matching ``pipeline.solve_ground_set``, which keeps every
    isolated cluster whose weight is positive -- and support is always positive.  A degree-0 vertex
    crosses nothing, so the kernel is pairwise compatible, hence laminar, hence at most ``2n - 1``
    clusters however large ``|C|`` is.  It can still be dropped: :meth:`drop` accepts kernel rows.
    """

    def __init__(self, gs, budget: int, *, include_kernel: bool = True):
        self.gs = gs
        self.budget = int(budget)
        self.n = int(gs.n_clusters)
        self.comp_of = np.full(self.n, -1, dtype=np.int64)
        self.local_of = np.full(self.n, -1, dtype=np.int64)
        for ci, comp in enumerate(gs.components):
            self.comp_of[comp.vertices] = ci
            self.local_of[comp.vertices] = np.arange(comp.n_vertices, dtype=np.int64)
        self.states = [heuristics._State(c, self.budget) for c in gs.components]  # noqa: SLF001
        self.in_s = np.zeros(self.n, dtype=bool)
        self.kernel = np.asarray(gs.decomposition.kernel, dtype=np.int64)
        if include_kernel and self.kernel.size:
            self.in_s[self.kernel] = True

    # ------------------------------------------------------------------ moves

    @property
    def n_hybrids(self) -> int:
        return int(sum(len(st.used) for st in self.states))

    @property
    def selection(self) -> np.ndarray:
        return np.flatnonzero(self.in_s).astype(np.int32)

    def can_accept(self, v: int) -> tuple[bool, int]:
        """``(feasible, class)``, with the *global* budget applied on top of the local mask."""
        v = int(v)
        if self.in_s[v]:
            return False, -1
        ci = int(self.comp_of[v])
        if ci < 0:
            return True, -1  # kernel: no overlap partner anywhere, so no constraint to violate
        st = self.states[ci]
        ok, klass = st.can_accept(int(self.local_of[v]))
        if not ok:
            return False, -1
        if klass >= 0 and klass not in st.used and self.n_hybrids >= self.budget:
            return False, -1
        return True, klass

    def accept(self, v: int, klass: int) -> None:
        v = int(v)
        ci = int(self.comp_of[v])
        if ci >= 0:
            self.states[ci].accept(int(self.local_of[v]), int(klass))
        self.in_s[v] = True

    def drop(self, v: int) -> None:
        v = int(v)
        ci = int(self.comp_of[v])
        self.in_s[v] = False
        if ci >= 0:
            self.states[ci].drop([int(self.local_of[v])])

    def legal(self, order=None) -> list[tuple[int, int]]:
        """Every ``(row, class)`` that may be accepted now, in ``order`` if one is given.

        A full scan.  ``can_accept`` is microseconds and one MPL evaluation is milliseconds to
        seconds, so an incremental legal set would optimise the wrong term -- and (L) being
        hereditary makes the incremental version tempting and easy to get subtly wrong.
        """
        rows = range(self.n) if order is None else order
        out = []
        for v in rows:
            ok, klass = self.can_accept(int(v))
            if ok:
                out.append((int(v), int(klass)))
        return out

    def copy(self) -> SearchState:
        """A copy for beam search: mutable state duplicated, immutable structure shared."""
        clone = object.__new__(SearchState)
        clone.gs = self.gs
        clone.budget = self.budget
        clone.n = self.n
        clone.comp_of = self.comp_of
        clone.local_of = self.local_of
        clone.kernel = self.kernel
        clone.in_s = self.in_s.copy()
        clone.states = [_clone_state(st) for st in self.states]
        return clone

    # ------------------------------------------------------------------ validation

    def verify(self) -> bool:
        """Re-derive feasibility with the independent validator, and check the mirror is in sync.

        ``heuristics.check`` recomputes (L) by a lexsort groupby and (N3O) as ``nnz((A@A).A)/6``
        from the accepted set alone -- it shares no state with ``_State``.  The second half of the
        check is that ``self.in_s`` and each component's own ``in_s`` still agree; they are two
        records of the same fact and a move that updated one and not the other would be invisible
        to the first half.
        """
        for ci, (comp, st) in enumerate(zip(self.gs.components, self.states, strict=True)):
            mine = np.flatnonzero(self.in_s[comp.vertices]).astype(np.int64)
            if not np.array_equal(np.sort(mine), np.sort(st.accepted.astype(np.int64))):
                raise AssertionError(f"component {ci}: the two accepted sets disagree")
            if not heuristics.check(comp, mine, self.budget).ok:
                return False
        return self.n_hybrids <= self.budget


# ---------------------------------------------------------------------------- the searches
#
# Every searcher is a generator over *incumbents*: it yields the selection it would return if
# stopped now, as often as it likes, and :func:`run` decides when an evaluation count has crossed a
# checkpoint.  That inversion is what keeps the x-axis honest -- a searcher cannot decide where the
# checkpoints fall, and a checkpoint always reports a selection the search actually held.


def _greedy_support(state, gs, scorer, *, budget, rng, top_b, width, refit_every):
    """Accept in support order.  Spends **no** MPL evaluations -- this is the baseline.

    The global form of ``heuristics.greedy(key="w")`` under ``instance``'s ``support`` scheme,
    which A26 names as the scheme to run and E1b measures at cluster F1 0.909.  Run here on F3's
    own cells rather than cited, so the comparison is on-grid.
    """
    for v in np.argsort(-np.asarray(gs.support, dtype=float), kind="stable"):
        ok, klass = state.can_accept(int(v))
        if ok:
            state.accept(int(v), klass)
    yield state.selection


def _greedy_mpl(state, gs, scorer, *, budget, rng, top_b, width, refit_every):
    """Add the highest-scoring legal cluster, repeatedly.

    ``top_b = 0`` scores *every* legal candidate; a positive ``top_b`` scores only the best that
    many by support.  The gap between the two is the measurement §2 of the plan asks for -- the
    support prior is exactly the job a learned policy would do, so how much it buys is how much
    room a policy has.
    """
    support = np.asarray(gs.support, dtype=float)
    incumbent = state.selection
    while scorer.n_evals < budget:
        cands = state.legal()
        if not cands:
            break
        if top_b > 0 and len(cands) > top_b:
            cands = sorted(cands, key=lambda t: -support[t[0]])[:top_b]
        base = state.selection
        best = (float("-inf"), -1, -1)
        for v, klass in cands:
            if scorer.n_evals >= budget:
                break
            value = scorer.score(gs, np.append(base, np.int32(v)))
            if value > best[0]:
                best = (value, v, klass)
            yield incumbent
        if best[1] < 0:
            break
        state.accept(best[1], best[2])
        incumbent = state.selection
        scorer.refresh(gs, incumbent, refit_every)
        yield incumbent


def _beam(state, gs, scorer, *, budget, rng, top_b, width, refit_every):
    """Beam search over add-moves, scored by MPL.

    ``beam1`` is greedy-mpl restricted to ``top_b`` candidates per step, which is why both are run:
    the pair separates "the beam is too narrow" from "the candidate pool is pruned wrong".
    """
    support = np.asarray(gs.support, dtype=float)
    beam = [state]
    scorer.score(gs, state.selection)
    incumbent = state.selection
    yield incumbent
    while scorer.n_evals < budget:
        children: list[tuple[float, object, int, int]] = []
        for st in beam:
            cands = st.legal()
            if top_b > 0 and len(cands) > top_b:
                cands = sorted(cands, key=lambda t: -support[t[0]])[:top_b]
            base = st.selection
            for v, klass in cands:
                if scorer.n_evals >= budget:
                    break
                value = scorer.score(gs, np.append(base, np.int32(v)))
                children.append((value, st, v, klass))
                yield incumbent
        if not children:
            break
        children.sort(key=lambda t: -t[0])
        beam = []
        for value, st, v, klass in children[:width]:
            clone = st.copy()
            clone.accept(v, klass)
            beam.append(clone)
        # **The incumbent is the best of the CURRENT beam, not the best seen at any depth.**
        # Those differ, and the difference is not cosmetic: the inner-loop score is taken at
        # parameters transferred from the last checkpoint, and that surrogate is *not* comparable
        # across depths -- a deeper selection carries edges the fit has no counterpart for, which
        # are scored at ``DEFAULT_LENGTH``.  A best-seen rule therefore latches onto a shallow
        # selection and stops descending, and it measured cluster F1 0.543 at 21 clusters where the
        # same budget of greedy reached 0.735 at 49.  ``probe_f3_cost.py`` licenses the surrogate
        # for ranking candidates *at one depth*, which is all a greedy step needs and all it tests.
        incumbent = beam[0].selection
        scorer.refresh(gs, incumbent, refit_every)
        yield incumbent


def _complete(st, rng) -> None:
    """One random maximal feasible completion, in a single pass over a shuffled vertex order.

    A pass, not a rescan-per-accept: rescanning is ``O(|C|)`` per accepted cluster and would spend
    more wall-clock on the mask than on the objective, which would make the wall-clock half of
    §F3's "cost per scored candidate" a measurement of this function instead.
    """
    for v in rng.permutation(st.n):
        ok, klass = st.can_accept(int(v))
        if ok:
            st.accept(int(v), klass)


def _random(state, gs, scorer, *, budget, rng, top_b, width, refit_every):
    """Uniform random feasible completion, restarted.  The uninformed control."""
    best_val, incumbent = float("-inf"), state.selection
    while scorer.n_evals < budget:
        st = state.copy()
        _complete(st, rng)
        value = scorer.score(gs, st.selection)
        if value > best_val:
            best_val, incumbent = value, st.selection
            scorer.refresh(gs, incumbent, refit_every)
        yield incumbent


#: How many proposals the annealer scores before it fixes its temperature.  They are scored, so
#: they count against the budget -- a calibration that was free would be a hidden extra budget.
_ANNEAL_TRIES = 32

#: Consecutive proposals that produce no move before the chain gives up.  A rejected proposal is
#: not scored, so it does not advance the budget: without this the loop has no termination
#: argument at all on a state whose legal set is empty.
_ANNEAL_STALL = 1000


def _anneal(state, gs, scorer, *, budget, rng, top_b, width, refit_every):
    """Metropolis over add / drop / swap, from a random feasible completion.

    **The temperature is calibrated from the data, not tuned.**  An untuned annealer that fails
    tells you about the tuning; ``T0`` is set to the median absolute score change over the first
    :data:`_ANNEAL_TRIES` proposals, and cooled geometrically to ``T0 / 1000``.  The calibration
    proposals are scored, so they are counted -- they are part of the budget, not free.
    """
    cur = state.copy()
    _complete(cur, rng)
    cur_val = scorer.score(gs, cur.selection)
    best_val, incumbent = cur_val, cur.selection
    yield incumbent

    def propose(st):
        """``(kind, added, dropped)`` or ``None``; applies the move to ``st``.

        An add is drawn from ``legal()``, not from all of ``C``.  Rejection sampling looks cheaper
        -- ``legal()`` is a full scan -- but the legal set shrinks to a handful as the selection
        fills, so uniform draws over ``|C|`` would fail almost every time and the chain would drift
        into drops without that ever showing up as an error.  The scan is milliseconds against a
        scored candidate's ~1.7 s.
        """
        kind = rng.integers(3)
        chosen = np.flatnonzero(st.in_s)
        if kind == 1 and chosen.size:  # drop
            v = int(chosen[rng.integers(chosen.size)])
            st.drop(v)
            return "drop", -1, v
        dropped = -1
        if kind == 2 and chosen.size:  # swap: make room first, which may re-open a legal add
            dropped = int(chosen[rng.integers(chosen.size)])
            st.drop(dropped)
        cands = st.legal()
        if cands:
            v, klass = cands[int(rng.integers(len(cands)))]
            st.accept(v, klass)
            return ("swap" if dropped >= 0 else "add"), v, dropped
        return ("drop", -1, dropped) if dropped >= 0 else None

    deltas: list[float] = []
    t0_value = None
    stall = 0
    while scorer.n_evals < budget:
        trial = cur.copy()
        if propose(trial) is None:
            # A rejected proposal costs no evaluation, so the budget does not advance and the loop
            # would spin forever on a state with no legal move.  Bounded rather than assumed away:
            # the guard is what turns "the annealer wedged" into a finite run with a stated reason.
            stall += 1
            if stall >= _ANNEAL_STALL:
                break
            yield incumbent
            continue
        stall = 0
        value = scorer.score(gs, trial.selection)
        if not np.isfinite(value):
            # A proposal that stands for no network is rejected outright.  Reaching the Metropolis
            # test with two infinities would make ``delta`` NaN, and ``min(0.0, nan)`` is 0.0 in
            # Python -- so the test would read ``random() < 1.0`` and accept it almost always.
            yield incumbent
            continue
        delta = value - cur_val if np.isfinite(cur_val) else float("inf")
        if t0_value is None:
            deltas.append(abs(delta))
            take = delta > 0
            if len(deltas) >= _ANNEAL_TRIES:
                # A proposal that stands for no network scores -inf, so its delta is infinite.
                # Filtering has to be explicit: ``float(nan) or 1.0`` is nan, nan makes every
                # Metropolis test False, and the annealer would quietly become a hill climber.
                usable = [d for d in deltas if np.isfinite(d) and d > 0.0]
                t0_value = float(np.median(usable)) if usable else 1.0
        else:
            frac = min(1.0, scorer.n_evals / max(1, budget))
            temp = max(1e-9, t0_value * (0.001 ** frac))
            take = delta > 0 or rng.random() < np.exp(min(0.0, delta / temp))
        if take:
            cur, cur_val = trial, value
            if value > best_val:
                best_val, incumbent = value, trial.selection
                scorer.refresh(gs, incumbent, refit_every)
        yield incumbent


_SEARCHERS = {
    "greedy-support": (_greedy_support, {"top_b": 0, "width": 1}),
    "greedy-mpl": (_greedy_mpl, {"top_b": 0, "width": 1}),
    "greedy-mpl-topb": (_greedy_mpl, {"width": 1}),
    "beam1": (_beam, {"width": 1}),
    "beam10": (_beam, {"width": 10}),
    "beam100": (_beam, {"width": 100}),
    "random": (_random, {"top_b": 0, "width": 1}),
    "anneal": (_anneal, {"top_b": 0, "width": 1}),
}


def run(method: str, gs, scorer: Scorer, k: int, *, budget: int, on_checkpoint,
        checkpoints=CHECKPOINTS, seed: int = 0, top_b: int = 64, refit_every: int = 5,
        include_kernel: bool = True) -> list[dict]:
    """Drive one search and return one record per checkpoint.

    ``on_checkpoint(n_evals, selection) -> dict`` is supplied by the caller so that this module
    never imports a driver: cluster recovery needs the true network and ``e4_exact``'s
    ``recovery_columns``, both of which live at script level.
    """
    if method not in _SEARCHERS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_SEARCHERS)}")
    fn, fixed = _SEARCHERS[method]
    kwargs = {"top_b": int(top_b), "width": 1, "refit_every": int(refit_every), **fixed}
    state = SearchState(gs, k, include_kernel=include_kernel)
    rng = np.random.default_rng(seed)
    targets = [int(c) for c in checkpoints if c <= budget]
    if budget not in targets:
        targets.append(int(budget))

    records: list[dict] = []
    ti = 0
    incumbent = state.selection
    last: tuple[np.ndarray, dict] | None = None

    def emit(at: int, sel):
        """One checkpoint row.  An unchanged incumbent is not re-optimised.

        A re-optimisation is thousands of likelihood calls, and a search that has plateaued -- or
        ``greedy-support``, which never moves at all -- would otherwise pay one per checkpoint for
        an answer it already has.  ``reused_refit`` says which rows are copies, so a reader can
        tell a converged plateau from a repeated measurement.
        """
        nonlocal last
        sel = np.asarray(sel, dtype=np.int32)
        if last is not None and np.array_equal(sel, last[0]):
            row = dict(last[1])
            row["reused_refit"] = True
        else:
            row = {**on_checkpoint(int(scorer.n_evals), sel), "reused_refit": False}
            last = (sel, dict(row))
        row["checkpoint"] = int(at)
        row.update(scorer.counters())
        records.append(row)

    # Checkpoint 0 fires *before* the search starts.  Its re-optimisation is what fills
    # ``scorer.fit``, and without it the first candidate every method scores would be ranked at
    # ``network_from_hasse``'s placeholder lengths and gamma = 0.5 -- which carry no information
    # about the data at all, and for an unpruned greedy step that is a thousand candidates.
    if targets and targets[0] == 0:
        emit(0, incumbent)
        ti = 1

    gen = fn(state, gs, scorer, budget=int(budget), rng=rng, **kwargs)
    for incumbent in gen:
        while ti < len(targets) and scorer.n_evals >= targets[ti]:
            emit(targets[ti], incumbent)
            ti += 1
        if scorer.n_evals >= budget:
            break
    while ti < len(targets):
        emit(targets[ti], incumbent)
        ti += 1
    return records
