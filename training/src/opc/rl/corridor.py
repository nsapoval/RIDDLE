"""The **swap corridor**: the few cluster swaps that separate greedy-by-support from the truth.

A75 measured the MPL surface at four *endpoints* per cell -- ``truth``, ``truth_clos``, ``greedy``,
``rl_best`` -- and read a policy that closes 78% of the greedy-to-truth pseudo-likelihood gap while
moving cluster F1 by -0.0096.  What it never measured is the **path**: at tau = 2 greedy and the
truth differ by only 4-11 clusters and have *identical cardinality*, so "get from greedy to the
truth" is a handful of swaps and the question is whether that handful is uphill, whether the uphill
direction is distinguishable from the thousands of alternatives, and whether the policy's
preference points along it.

**The one construction everything rests on.**  For a cell with ground set ``C``, greedy selection
``G`` and true rows ``T``, write ``D = T \\ G`` (the misses), ``K = T & G`` (the hits) and
``J = G \\ T`` (the false positives).  For ``A`` a subset of ``D``,

    S(A) = accept A, then K in support order, then J in support order, then complete by
           argmax-support until the mask is empty.

Measured on five cells while this module was designed: ``S({}) == G`` and ``S(D) == T``
**exactly**, and every ``S(A)`` is feasible and maximal.  So the corridor is a **hypercube** on
``|D|`` bits whose two poles are the baseline and the answer, every node of which is a legal
:class:`~opc.rl.env.ClusterEnv` terminal state.

**Maximal does not mean equal-sized, and an early draft of this docstring said it did.**  On the
pilot cell the corridor runs at 49 clusters throughout but 52, 51 or 50 free parameters depending
on the subset, and one node of nine sits at 48 clusters -- so A75's "not complexity" control is
*not* obtained by construction here, and ``n_params`` has to be carried per node and read beside
every Delta.  It happens to point the safe way on that cell (the truth pole carries **fewer**
parameters than greedy and scores 120,743 log units higher), but that is a measurement rather than
a property, and the driver records it per node for that reason.

*Why the tiers work, and why it is not a coincidence.*  Feasibility here is **hereditary**: (L) and
(N3O) are downward-closed and a subset of a feasible set opens no more intersection classes, so the
budget is too.  Hence any feasible set can be accepted in any order.  Tier 2 therefore takes all of
``K`` when ``A = D`` (because ``A | K == T`` is feasible), and tier 3 takes none of ``J`` (because
``T`` is maximal).  The same fact is why :class:`~opc.rl.env.ClusterEnv` can be add-only without
losing anything: its reachable terminal set is *exactly* the set of all maximal feasible cluster
systems.

**Two operators, and they are deliberately different.**  :func:`corridor_node` tiers ``K`` ahead of
``J`` -- it is an *oracle* construction and has to be, or its far pole would not be the truth.
:func:`force_move` does not know ``T`` at all: it forces one cluster and re-admits the current
selection in support order.  Every comparison between a truth-ward move and an alternative move
goes through :func:`force_move`, so the two arms get identical tie-breaking; using the tiered
operator for the truth-ward arm would hand it the answer and call the result a signal.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from opc.rl.env import ClusterEnv

__all__ = [
    "Frame", "frame", "complete", "corridor_node", "force_move", "support_chain",
    "enumerate_nodes", "monotone_reachable", "epistasis", "legal_windows", "subset_key",
]


@dataclass
class Frame:
    """One cell's corridor frame: the two poles, their difference, and the guards.

    ``truth`` is ``inst.truth_rows`` -- ``C(N_true) & C_tau``, which is not all of ``C(N_true)``
    when the tau filter dropped one (G-R0c: 94.8% of the full design keeps everything).  The guard
    flags say so rather than the far pole silently becoming something else: a cell that fails
    ``truth_complete`` still has a corridor, its far end is just ``T & C_tau``.
    """

    greedy: np.ndarray
    truth: np.ndarray
    misses: np.ndarray            # D = T \ G, in descending support order
    hits: np.ndarray              # K = T & G
    false_pos: np.ndarray         # J = G \ T
    n_true: int
    #: ``truth_rows.size == n_true`` -- the tau filter kept every true cluster (G-R0c).
    truth_complete: bool
    #: ``T`` is accepted in full by the env's own mask.
    truth_feasible: bool
    #: nothing is legal once ``T`` is accepted -- so the far pole is a terminal state.
    truth_maximal: bool
    #: ``|G| == |T|`` -- A71 measured this on 98.5% of the 960-cell design, not 100%.
    balanced: bool
    guards: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return int(self.misses.size)

    def as_row(self) -> dict:
        return {"n_greedy": int(self.greedy.size), "n_truth": int(self.truth.size),
                "n_true": int(self.n_true), "d": self.d,
                "n_false_pos": int(self.false_pos.size),
                "truth_complete": self.truth_complete, "truth_feasible": self.truth_feasible,
                "truth_maximal": self.truth_maximal, "balanced": self.balanced}


def _by_support(env: ClusterEnv, rows, key=None) -> np.ndarray:
    """``rows`` in descending ``key`` order, stably -- support when ``key`` is None.

    ``key`` exists so W12's ``beam.force_decode`` can run this exact operator over the *model's*
    scores rather than over support.  It has to be the same function: a control that priced
    "beam against force" while also changing the ranking would be pricing two things at once.
    Every landed caller passes nothing and gets support, byte for byte.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return rows
    w = env.support if key is None else np.asarray(key, dtype=np.float64)
    return rows[np.argsort(-w[rows], kind="stable")]


def complete(env: ClusterEnv, tiers, *, verify: bool = False, key=None) -> np.ndarray:
    """Accept ``tiers`` in order, by descending support within each, then complete to maximal.

    ``tiers[0]`` is **forced**: every one of its rows must be acceptable when it is reached, and a
    row that is not raises.  That is the contract the corridor's poles rest on -- ``S(D)`` is the
    truth only because tier 0 takes all of ``D`` -- and a silent skip would turn a wrong pole
    into a plausible number.  Later tiers are best-effort: a row blocked by an earlier tier is
    exactly what a swap *is*.

    The tier phases go through ``state.can_accept`` / ``state.accept`` rather than ``env.step``,
    because ``step`` recomputes the whole ``|C| x |S|`` legal mask on every accept and the tiers do
    not need it -- only the final completion does, and there it is what decides termination.
    ``verify=True`` re-derives feasibility with ``heuristics.check`` through
    ``SearchState.verify()``, which shares no state with the mask.
    """
    env.reset()
    for ti, tier in enumerate(tiers):
        for v in _by_support(env, tier, key):
            ok, klass = env.state.can_accept(int(v))
            if ok:
                env.state.accept(int(v), int(klass))
            elif ti == 0 and not env.state.in_s[int(v)]:
                # ``can_accept`` returns ``(False, -1)`` for a row that is *already* in ``S`` --
                # which every kernel row is, because ``reset`` seeds them.  Reading that as a
                # refusal made ``frame`` report the truth infeasible on a cell where the
                # independent validator says otherwise (kernel size 3, three "rejections"), so
                # the two cases are separated here rather than counted together.
                raise ValueError(f"forced cluster {int(v)} is not acceptable; tier 0 must be a "
                                 "feasible set presented before anything can block it")
    w = env.support if key is None else np.asarray(key, dtype=np.float64)
    order = np.argsort(-w, kind="stable")
    rank = np.empty(env.n, dtype=np.int64)
    rank[order] = np.arange(env.n)
    while True:
        cand = np.flatnonzero(env.legal_mask())
        if cand.size == 0:
            break
        v = int(cand[np.argmin(rank[cand])])
        ok, klass = env.state.can_accept(v)
        env.state.accept(v, int(klass))
    if verify and not env.state.verify():
        raise AssertionError("completed selection fails the independent feasibility check")
    return env.selection.copy()


def frame(inst, env: ClusterEnv | None = None) -> Frame:
    """The corridor frame of one instance, with every guard measured rather than assumed."""
    env = env or ClusterEnv(inst.gs, budget=inst.k, graph=inst.graph)
    greedy = complete(env, [[]], verify=True)
    truth = np.sort(np.asarray(inst.truth_rows, dtype=np.int64))

    # Presented in **support** order and counting an already-seeded kernel row as accepted.  Both
    # matter: ``reset`` seeds the kernel, so ``can_accept`` refuses those rows for the one reason
    # that is not a refusal, and reading it otherwise reported the truth infeasible on a cell where
    # ``truth.feasible_and_cost`` -- the order-free validator -- says feasible at exactly ``k``.
    env.reset()
    feasible = True
    for v in _by_support(env, truth):
        if env.state.in_s[int(v)]:
            continue
        ok, klass = env.state.can_accept(int(v))
        if ok:
            env.state.accept(int(v), int(klass))
        else:
            feasible = False
    feasible = feasible and bool(env.state.verify())
    maximal = not bool(env.legal_mask().any())

    misses = np.setdiff1d(truth, greedy)
    return Frame(
        greedy=np.sort(greedy), truth=truth,
        misses=_by_support(env, misses),
        hits=np.intersect1d(truth, greedy), false_pos=np.setdiff1d(greedy, truth),
        n_true=int(inst.n_true),
        truth_complete=bool(truth.size == int(inst.n_true)),
        truth_feasible=bool(feasible), truth_maximal=bool(maximal),
        balanced=bool(greedy.size == truth.size),
    )


def corridor_node(env: ClusterEnv, fr: Frame, A) -> np.ndarray:
    """``S(A)`` -- the corridor node at subset ``A`` of ``fr.misses``.  **Oracle-tiered.**

    ``K`` ahead of ``J`` is what makes ``S(fr.misses)`` the truth exactly, and it is why this
    operator must not be used to score a *move*: it would give the truth-ward arm a tie-break the
    alternative arm does not get.  :func:`force_move` is the operator for that.
    """
    return complete(env, [A, fr.hits, fr.false_pos])


def force_move(env: ClusterEnv, selection, v: int, *, key=None) -> np.ndarray:
    """Force ``v`` into ``selection`` and re-complete.  **Knows nothing about the truth.**

    The uniform move operator: one cluster is pushed to the top of the priority order, whatever
    of the incumbent still fits is re-admitted in support order, and the state is completed to
    maximal.
    Truth-ward when ``v`` is in ``fr.misses`` and an alternative otherwise, with byte-identical
    treatment either way -- which is the whole point of it being one function.
    """
    return complete(env, [[int(v)], np.asarray(selection, dtype=np.int64)], key=key)


def support_chain(fr: Frame) -> list[tuple[int, ...]]:
    """The ``d + 1`` prefixes of ``fr.misses`` -- the corridor walked in *support* order.

    This is the ordering a support-anchored searcher would take, and it is not the best one: on the
    pilot cell it opens with two swaps worth -3.3 and -8.5 log units before a third worth +99,708.
    Reported beside :func:`monotone_reachable`, which says whether *some* ordering is monotone, so
    an ordering artefact cannot be read as a barrier.
    """
    return [tuple(range(i)) for i in range(fr.d + 1)]


def subset_key(A) -> tuple[int, ...]:
    """Canonical key for a subset of ``fr.misses``, as positions into it."""
    return tuple(sorted(int(i) for i in A))


def enumerate_nodes(d: int, *, budget: int = 256, seed: int = 0,
                    chains: int = 12) -> list[tuple[int, ...]]:
    """Which subsets of ``D`` to score: the whole hypercube when it fits, a stratified sample else.

    The full lattice is what :func:`monotone_reachable` and :func:`epistasis` need, so it is taken
    whenever ``2^d <= budget``.  Above that the sample is deliberately not uniform -- uniform draws
    from a hypercube concentrate at depth ``d/2`` and would measure neither pole's neighbourhood.
    It takes both poles, every singleton and every co-singleton (which is what the marginal and the
    interaction are computed from), ``chains`` random monotone chains (so at least some *paths* are
    complete), and uniform subsets to fill.
    """
    if d <= 0:
        return [()]
    if d <= 30 and 2 ** d <= budget:
        return [tuple(c) for r in range(d + 1) for c in itertools.combinations(range(d), r)]
    rng = np.random.default_rng(seed)
    out = {(), tuple(range(d))}
    for i in range(d):
        out.add((i,))
        out.add(subset_key([j for j in range(d) if j != i]))
    for _ in range(chains):
        order = rng.permutation(d)
        for i in range(1, d):
            out.add(subset_key(order[:i]))
    while len(out) < budget:
        r = int(rng.integers(1, d))
        out.add(subset_key(rng.choice(d, size=r, replace=False)))
    return sorted(out, key=lambda t: (len(t), t))


def monotone_reachable(values: dict, d: int) -> dict:
    """Is there a **strictly MPL-increasing** chain from ``S({})`` to ``S(D)``, and how rare is it?

    The direct form of "all we need to do is swap a few clusters in": a hill-climber that only
    ever accepts an improvement reaches the truth iff such a chain exists, and how *hard* it is
    is the share of the ``d!`` orderings that are increasing at every step.  Both come from one
    ``O(2^d d)`` pass over the lattice, because the number of increasing orderings through a subset
    factorises over its predecessors.

    Returns ``exists = False`` with ``n_orderings = 0`` when the far pole is unreachable, and every
    field as ``None`` when ``values`` is not the complete hypercube -- a sampled corridor cannot
    answer this and must not appear to.
    """
    if d <= 0 or d > 30 or len(values) != 2 ** d:
        return {"monotone_exists": None, "monotone_frac_orderings": None,
                "monotone_n_orderings": None}
    count = {(): 1}
    for r in range(1, d + 1):
        for comb in itertools.combinations(range(d), r):
            here = values[comb]
            total = 0
            for a in comb:
                prev = subset_key([x for x in comb if x != a])
                if count[prev] and here > values[prev]:
                    total += count[prev]
            count[comb] = total
    full = tuple(range(d))
    n = int(count[full])
    return {"monotone_exists": bool(n > 0),
            "monotone_frac_orderings": float(n / math.factorial(d)),
            "monotone_n_orderings": n}


def epistasis(values: dict, d: int) -> dict:
    """Is a swap worth the same alone as it is last?  ``solo[a]`` against ``last[a]``.

    An additive corridor is a separable problem and a one-step searcher should walk it.  Strong
    interaction -- a swap worth nothing alone and everything in company -- is a different kind of
    hardness, and it is the one that would say ``k`` clusters have to move together.  Needs both
    poles' neighbourhoods, which :func:`enumerate_nodes` always samples.
    """
    base, full = values.get(()), values.get(tuple(range(d)))
    if base is None or full is None or d <= 0:
        return {"epistasis_max": None, "epistasis_corr": None,
                "solo_gain_max": None, "solo_gain_n_downhill": None}
    solo, last = [], []
    for a in range(d):
        one, co = values.get((a,)), values.get(subset_key([x for x in range(d) if x != a]))
        if one is None or co is None:
            continue
        solo.append(one - base)
        last.append(full - co)
    if not solo:
        return {"epistasis_max": None, "epistasis_corr": None,
                "solo_gain_max": None, "solo_gain_n_downhill": None}
    solo_a, last_a = np.asarray(solo), np.asarray(last)
    corr = (float(np.corrcoef(solo_a, last_a)[0, 1])
            if solo_a.size > 1 and solo_a.std() > 0 and last_a.std() > 0 else float("nan"))
    return {"epistasis_max": float(np.abs(solo_a - last_a).max()),
            "epistasis_corr": corr,
            "solo_gain_max": float(solo_a.max()),
            "solo_gain_n_downhill": int((solo_a < 0).sum())}


def legal_windows(env: ClusterEnv, rows) -> dict:
    """Replay greedy's construction and record when each of ``rows`` is legal.

    **This is the check that the add-only MDP is not the restriction it looks like.**  There is
    no drop action, so the obvious worry is that a cluster greedy displaces can never be chosen at
    all; feasibility being hereditary says otherwise in principle, and this says so per cell.
    Measured on three cells: 0 of 8, 0 of 11 and 0 of 4 missed truth clusters were *never* legal
    -- every one is on the menu at a state greedy itself visits, with the window closing at step
    21-47 of ~50.  So what separates the truth from greedy is a ranking decision taken deep in an
    episode, not an unreachable state, and the failure to look for is credit assignment.

    Returns the accepted sequence, and per row: the steps at which it was legal, the step its
    window closed, and what greedy accepted there.
    """
    rows = np.asarray(rows, dtype=np.int64)
    env.reset()
    mask = env.legal_mask()
    live: dict[int, list[int]] = {int(v): [] for v in rows}
    accepted: list[int] = []
    while mask.any():
        for v in rows:
            if mask[v]:
                live[int(v)].append(len(accepted))
        # Not ``argmax(support)``: greedy's rule is a *stable* argsort of the whole ground set, so
        # ties go to the lower row.  ``greedy_support_rollout`` breaks them that way and G-R0's
        # parity test is against it, so this has to as well.
        cand = np.flatnonzero(mask)
        best = int(cand[np.lexsort((cand, -env.support[cand]))[0]])
        accepted.append(best)
        obs, _r, _done, _i = env.step(best)
        mask = obs["legal"]
    closes = {v: (steps[-1] if steps else -1) for v, steps in live.items()}
    return {"accepted": np.asarray(accepted, dtype=np.int64), "live": live, "closes": closes,
            # The kernel is seeded by ``reset`` and is therefore in the selection but not in the
            # action sequence -- ``n_steps`` is 49 where ``|selection|`` is 52 on a cell with three
            # kernel rows.  Both are returned so a parity check compares the right pair.
            "selection": env.selection.copy(), "n_steps": len(accepted),
            "n_never_legal": int(sum(1 for s in live.values() if not s)),
            "killer": {v: (int(accepted[c]) if c >= 0 else -1) for v, c in closes.items()}}
