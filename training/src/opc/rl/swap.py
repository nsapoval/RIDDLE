"""W8 -- the **swap MDP**: start at greedy, walk by cluster swaps, keep the best system seen.

``results/rl/RL_V1_RESULTS.md`` §12.5 (A76) is the instruction this implements, and it is a
correction to §7 rather than a new idea.  A75 read the MPL surface at four *endpoints* per cell
and concluded the objective was "valid and badly conditioned for search".  A76 walked the path
instead and overturned that: the same decision is worth **+83,816 log units and ranks first of
136** when posed as a one-step swap with the reward attached, and is a support tie-break deferred
**20-45 steps** behind a single terminal reward when posed as a construction divergence.  Local
``spearman(dMPL, dF1)`` is **+0.671** against A75's cross-cell +0.323, and a **truth-blind** MPL
repair search over exactly this move set beats greedy-by-support on **9 of 10** cells (median
**+0.0385** cluster F1, worse on none) against A58's 0 of 68 for eight *add-only* searches.

**So the fault was the MDP.**  This module is the amended one:

===========  =====================================  ====================================
                v1 (:mod:`opc.rl.env`)                 v2 (here)
===========  =====================================  ====================================
``S_0``      ``{}`` plus the kernel                 **greedy-by-support**
state        a partial feasible selection           a **maximal** selection + tabu
action       ``a in legal(S_t)`` -- an *add*        ``v not in S_t``, via ``force_move``
terminates   ``legal(S_t) = {}``                    ``t = H``.  **No stopping rule.**
return       ``MPL(S_T)``                           ``max_t MPL(S_t)`` -- the best seen
episode      ~50 steps, the first ~40 forced        ``H`` steps, each a real decision
===========  =====================================  ====================================

**Every state here is a terminal state of the old MDP.**  ``corridor.force_move`` lands on a
maximal feasible cluster system (pinned by ``tests/test_rl_corridor.py``), so the walk never
leaves the space v1's *answers* lived in -- it moves around inside it instead of building up to
one point of it.  Nothing here re-implements a move: :meth:`SwapEnv.step` **is**
``corridor.force_move``, the operator every comparison in A76 §12 was scored through.

**The policy is stationary, and that is what makes the horizon a free parameter.**  Under a
best-seen return there is no terminal payoff to time your arrival for, so "which swap is best
from here" cannot depend on how many steps remain.  Nothing in :func:`swap_scalars` or
:func:`swap_features` is a function of ``t`` or of ``H`` -- ``swap_distance`` replaces v1's
``t/n`` and is a function of the *state*.  Three things follow, the first a test not a hope:

1. **Trajectories are prefix-nested**: at one seed the ``H = 24`` run is byte-identical to the
   first 24 steps of the ``H = 96`` run (``tests/test_rl_swap.py::test_prefix_property``).
2. **Best-seen is monotone non-decreasing in ``H``** by construction -- more steps can cost
   compute, never quality.
3. **The candidate pool is ``K x H``, not ``K``**, so ``K`` may shrink as ``H`` grows.

**Tabu, because a stochastic walk can cycle where A76's search could not.**  §12.2's repair
search accepted only strict improvements, so it could not revisit; a sampled policy can force
``v`` (which displaces ``u``) and then force ``u`` (which restores ``v``) forever.  The tabu ring
masks the last ``tenure`` forced clusters.  It is a function of the trajectory *prefix*, so (1)
survives it.

**No torch here**, as in :mod:`opc.rl.env` and :mod:`opc.rl.instances`: this is numpy only, so a
``SwapEnv`` is usable in a pool worker with no GPU and the device appears only in the policy layer.

**There is no legality scan here.**  In v1 the mask was the hot loop; here every ``v not in S_t``
is forceable -- ``complete`` presents it to a cleared state, where the mask is all-true -- so
:meth:`SwapEnv.legal_mask` is a boolean AND and the constraint work happens inside
``force_move``'s own completion.  What this module computes per step is :func:`incumbent_stats`,
one gather over the edges incident to ``S_t``: at most ``n - 1 + O(k)`` accepted vertices, about
fifty at n = 50 whatever ``|C|`` is, which is the observation ``ClusterEnv.legal_mask`` rests on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from opc.rl import corridor as CO
from opc.rl.env import ClusterEnv, greedy_support_rollout

__all__ = [
    "N_SWAP_DYNAMIC",
    "N_SWAP_SCALARS",
    "SwapObs",
    "SwapEnv",
    "incumbent_stats",
    "swap_features",
    "swap_scalars",
    "oracle_swap_actions",
]

#: The decoder's per-candidate features here.  Four, against v1's three -- see
#: :func:`swap_features`.
N_SWAP_DYNAMIC = 4
#: The decoder's context scalars.  **Four, exactly as in v1**, so ``Policy.ctx`` keeps its shape
#: and a v1 checkpoint's encoder / key / context weights transfer unchanged; only ``dyn`` is
#: re-initialised.
N_SWAP_SCALARS = 4


def incumbent_stats(graph, in_s: np.ndarray, support: np.ndarray):
    """``(deg_sel, min_w_nbr)`` over the edges incident to the accepted set.

    ``deg_sel[v]`` is how many accepted clusters ``v`` overlaps -- which is how many of them
    forcing ``v`` might displace, measured at a median of 3-4 against a max of 7.
    ``min_w_nbr[v]`` is the *weakest* accepted cluster ``v`` crosses, which is what feature 3 of
    :func:`swap_features` is a margin against.

    The gather is over ``sum_{s in S} deg(s)`` entries -- ~15 k at n = 50, tau = 2 -- never over
    the full 2\\|E\\| edge list, for the reason ``ClusterEnv.legal_mask`` gives: ``|S|`` is about
    fifty whatever ``|C|`` is.

    ``np.minimum.at`` says the second line directly and is the slow way to say it (the note in
    ``env.legal_mask``: unbuffered ``ufunc.at`` leaves numpy's fast path).  A lexsort plus a
    first-of-each-group take is the same statement over a 15 k array.
    """
    n = int(graph.n)
    sel = np.flatnonzero(in_s).astype(np.int64)
    deg_sel = np.zeros(n, dtype=np.int64)
    min_w = np.full(n, np.inf, dtype=np.float64)
    if sel.size == 0:
        return deg_sel, min_w
    indptr = np.asarray(graph.indptr, dtype=np.int64)
    counts = indptr[sel + 1] - indptr[sel]
    if counts.sum() == 0:
        return deg_sel, min_w
    take = np.concatenate([np.arange(indptr[s], indptr[s + 1]) for s in sel])
    dst = np.asarray(graph.nbr, dtype=np.int64)[take]
    w_src = np.asarray(support, dtype=np.float64)[np.repeat(sel, counts)]
    deg_sel = np.bincount(dst, minlength=n)
    order = np.lexsort((w_src, dst))
    d_o, w_o = dst[order], w_src[order]
    first = np.ones(d_o.size, dtype=bool)
    first[1:] = d_o[1:] != d_o[:-1]
    min_w[d_o[first]] = w_o[first]
    return deg_sel, min_w


def swap_features(support: np.ndarray, in_s: np.ndarray, tabu: np.ndarray,
                  deg_sel: np.ndarray, min_w_nbr: np.ndarray) -> np.ndarray:
    """``float32[|C|, 4]`` -- the swap decoder's per-candidate features.

    1. ``deg_sel / |S_t|`` -- how much of the incumbent ``v`` crosses, i.e. how large a
       displacement forcing it would be.  Free from :func:`incumbent_stats`.
    2. ``w(v) / max_{v not in S_t} w`` -- support renormalised over the *action* set, the
       quantity that changes as the walk proceeds; the instance-level support rank is already a
       static node feature and says nothing about the current state.
    3. **the displacement margin** ``(w(v) - min{w(u) : u in S_t, u ~ v}) / max w``.  The swap
       analogue of v1's greedy-indicator bit, handed over for v1's reason (RL_PLAN §5): support
       ranking alone separates ``C(N_true)`` from the tau = 2 ground set at AUC 0.9918 (A73), so
       a policy made to rediscover "is this stronger than the weakest thing it would push out"
       from embeddings would spend its budget on the part that is already solved.  Every
       ``v not in S_t`` has at least one accepted neighbour -- one with none would be acceptable
       outright and ``S_t`` would not be maximal -- so the ``inf`` branch is a guard, not a case.
    4. the **tabu** indicator.  A feature and not only a mask, because the policy has to be able
       to tell "I may not take this" from "this is a bad idea".
    """
    n = support.shape[0]
    out = np.zeros((n, N_SWAP_DYNAMIC), dtype=np.float32)
    w = np.asarray(support, dtype=np.float64)
    out[:, 0] = deg_sel / max(1.0, float(in_s.sum()))
    free = ~in_s
    if free.any():
        top = float(w[free].max())
        if top > 0:
            out[free, 1] = w[free] / top
            margin = np.where(np.isfinite(min_w_nbr), w - min_w_nbr, top)
            out[free, 2] = np.clip(margin[free] / top, -1.0, 1.0)
    out[:, 3] = tabu.astype(np.float32)
    return out


def swap_scalars(env: SwapEnv) -> np.ndarray:
    """``float32[4]`` -- the decoder context.  **Nothing here is a function of ``t`` or ``H``.**

    v1's fourth scalar was ``t/n``.  Feeding absolute time to a policy trained at one horizon is
    exactly what would break the prefix property: at ``H = 96`` the second half of the episode
    carries ``t/n`` values the policy never saw.  ``swap_distance`` -- how far the incumbent has
    moved from greedy -- answers the same "how far out am I" question as a function of the
    **state**, so it is well defined at any horizon and at any point of a run.
    """
    k = max(1, env.env.budget)
    return np.asarray([
        (env.env.budget - env.env.state.n_hybrids) / k,
        env.env.budget / max(1, env.n_taxa),
        env.selection.size / max(1, env.n_taxa),
        env.swap_distance / max(1.0, float(env.greedy.size)),
    ], dtype=np.float32)


@dataclass
class SwapObs:
    """One state as the decoder consumes it.  ``legal`` is ``~in_S_t & ~tabu``."""

    selection: np.ndarray
    in_s: np.ndarray
    legal: np.ndarray
    dynamic: np.ndarray
    scalars: np.ndarray
    t: int
    last_action: int


class SwapEnv:
    """One instance's swap MDP.  Wraps :class:`~opc.rl.env.ClusterEnv`; does not extend it.

    **Wrapping is deliberate.**  ``ClusterEnv`` is G-R0's parity subject -- argmax-support under
    it must reproduce ``search._greedy_support`` selection for selection, and every terminal
    state must satisfy ``heuristics.check``.  A subclass that redefined ``step`` would leave that
    test asserting something other than what it was written to assert.  Here the construction env
    is used for what it is: the thing that produces ``S_0``, and the thing ``force_move``
    completes through.
    """

    #: Optional hard restriction of the action set to the ``top_m`` unselected clusters by
    #: support.  **Off by default, and the default is a measurement rather than caution.**
    #:
    #: It is tempting: A76 §12.2's repair search drew half its 64 candidates by top support for
    #: exactly the reason ``w7_corridor._draw_alternatives`` states -- a uniform draw over a
    #: tau = 2 ground set is mostly low-support rows no search would look at.  Measured over 200
    #: sampled single swaps from greedy on four n = 50 cells, the share that raise cluster F1 is
    #: **1.0-10.0% over the whole free set and 3.1-18.8% over the top 32**, a 3-19x concentration.
    #:
    #: And it costs coverage that the add-only MDP did not cost.  ``corridor.legal_windows``
    #: measured **0 of 100** missed truth clusters as ever illegal in v1 -- feasibility is
    #: hereditary, so nothing was unreachable.  A support cap introduces unreachability outright.
    #: Over 40 validation cells and 287 missed truth clusters, the support rank of a miss among
    #: the free set is median **26**, p90 **187**, max **2,367**, and a cap retains *all* of a
    #: cell's misses on:
    #:
    #: =========  ==================  =========================
    #: cap        cells (of 40)       individual misses
    #: =========  ==================  =========================
    #: top-32     10                  54.7%
    #: top-128    28                  85.0%
    #: top-256    29                  91.3%
    #: top-1024   38                  99.3%
    #: =========  ==================  =========================
    #:
    #: At top-256 that is 8.7% of the truth made unreachable **by construction** -- the same order
    #: as G-R0c's own coverage loss, and it would cap G-R5 silently.  So the concentration is
    #: handed to the *policy* instead, as the support features it already has (static support rank,
    #: and :func:`swap_features`' renormalised support and displacement margin), and this stays an
    #: ablation.  Anything that turns it on owes the coverage number beside the result.
    DEFAULT_TOP_M = None

    def __init__(self, gs, budget: int, *, graph=None, tabu: int = 8,
                 top_m: int | None = DEFAULT_TOP_M, env: ClusterEnv | None = None):
        self.env = env or ClusterEnv(gs, budget=budget, graph=graph)
        self.tabu_tenure = int(tabu)
        self.top_m = None if top_m is None else int(top_m)
        self.n = int(self.env.n)
        self.n_taxa = int(gs.n_taxa)
        self.support = np.asarray(gs.support, dtype=np.float64)
        #: The whole ground set in descending support order, stably -- greedy's own tie-break, so
        #: "top by support" here means the same thing it means in ``greedy_support_rollout``.
        self._order = np.argsort(-self.support, kind="stable")
        self.greedy = np.sort(greedy_support_rollout(self.env)).astype(np.int64)
        self.reset()

    # ------------------------------------------------------------------ episode

    def reset(self, *, start=None) -> SwapObs:
        """``S_0`` is greedy-by-support, computed once in ``__init__`` and replayed here.

        ``start`` exists for the ablation RL_PLAN §2 already has a hook for
        (``ClusterEnv.reset``'s ``init="given"``): a warm start that is not greedy.  W8's
        headline arm does not use it.

        **The underlying ``ClusterEnv`` is re-established, not just the row set, and the first
        draft of this method did not do it.**  ``force_move`` leaves ``self.env.state`` standing at
        the state it built, so a ``reset`` that only rewrote ``self.selection`` left the *previous
        walk's* commitments and opened classes in place: ``swap_scalars``' ``k_rem / k`` came from
        the wrong state and :meth:`verify` validated a selection that was no longer the one this
        object was reporting.  It was invisible on the unit fixture, where every state saturates
        the budget and the two agree by luck -- ``env.state.selection`` disagreeing with
        ``self.selection`` is what showed it.  Re-completing through ``corridor.complete`` costs
        one ``force_move`` (2-4 ms) and also *validates* ``start``: tier 0 raises on a row that is
        not acceptable, so an infeasible warm start cannot be walked from.
        """
        sel = self.greedy if start is None else np.asarray(start, dtype=np.int64)
        rebuilt = CO.complete(self.env, [sel])
        if start is None and not np.array_equal(np.sort(rebuilt), self.greedy):
            raise AssertionError("re-completing greedy did not reproduce it; greedy is maximal by "
                                 "construction (G-R0) and this would mean the env has drifted")
        self._set(rebuilt)
        self.t = 0
        self.last_action = -1
        self._tabu: deque = deque(maxlen=self.tabu_tenure)
        self._seen = {self._fingerprint(self.selection)}
        self.n_revisits = 0
        return self.observe()

    def step(self, action: int) -> tuple[SwapObs, bool]:
        """Force ``action`` and re-complete.  ``done`` is **never** true: the caller owns ``H``.

        There is no stopping rule and no per-step reward -- the return is ``max_t MPL(S_t)`` and
        the trainer assembles it from the trajectory's values.  A76 §12.3 is why the stopping
        rule went: a strictly-improving rule reaches the truth only along an ordering that exists
        on 5 of 7 decidable cells and is **rare** where it exists (median 3.1% of the ``d!``
        orderings), so it stalls the moment its candidate sample misses one.  "Walk for ``H`` and
        report the best seen" has no such failure mode.
        """
        v = int(action)
        if self.in_s[v]:
            raise ValueError(f"cluster {v} is already selected; forcing it is a no-op")
        if not self.legal_mask()[v]:
            raise ValueError(f"cluster {v} is under tabu or the support cap; the mask and the "
                             "policy have diverged")
        self._set(CO.force_move(self.env, self.selection, v))
        self._tabu.append(v)
        self.t += 1
        self.last_action = v
        fp = self._fingerprint(self.selection)
        if fp in self._seen:
            self.n_revisits += 1
        self._seen.add(fp)
        return self.observe(), False

    def observe(self) -> SwapObs:
        tabu = self.tabu_mask()
        deg_sel, min_w = incumbent_stats(self.env.graph, self.in_s, self.support)
        return SwapObs(
            selection=self.selection, in_s=self.in_s,
            legal=self.legal_mask(),
            dynamic=swap_features(self.support, self.in_s, tabu, deg_sel, min_w),
            scalars=swap_scalars(self), t=self.t, last_action=self.last_action)

    def legal_mask(self) -> np.ndarray:
        """``~in_S_t & ~tabu``, and the support cap when one is set.  See :data:`DEFAULT_TOP_M`."""
        ok = (~self.in_s) & (~self.tabu_mask())
        if self.top_m is None:
            return ok
        capped = np.zeros(self.n, dtype=bool)
        capped[self._order[ok[self._order]][:self.top_m]] = True
        return capped

    # ------------------------------------------------------------------ state

    def _set(self, sel: np.ndarray) -> None:
        self.selection = np.sort(np.asarray(sel, dtype=np.int64))
        self.in_s = np.zeros(self.n, dtype=bool)
        self.in_s[self.selection] = True

    def tabu_mask(self) -> np.ndarray:
        """``bool[|C|]`` -- the clusters under tabu.  **Not** the complement of ``legal``.

        A cluster is in the ring because it was just forced, so it is usually *also selected*;
        ``legal`` is ``~in_s & ~tabu`` and this is one of its two terms.  Column 3 of
        :func:`swap_features` is this array, including on selected rows, because "this is a recent
        arrival" is information about an incumbent too.
        """
        m = np.zeros(self.n, dtype=bool)
        if self._tabu:
            m[np.fromiter(self._tabu, dtype=np.int64, count=len(self._tabu))] = True
        return m

    @staticmethod
    def _fingerprint(sel: np.ndarray) -> bytes:
        return np.ascontiguousarray(np.sort(np.asarray(sel, dtype=np.int64))).tobytes()

    @property
    def swap_distance(self) -> int:
        """``|S_t \\ greedy|`` -- how far the walk has gone.  The stationary stand-in for ``t``."""
        return int(np.setdiff1d(self.selection, self.greedy, assume_unique=True).size)

    def verify(self) -> bool:
        """``heuristics.check`` on the current state, plus maximality.

        ``force_move`` leaves the underlying ``ClusterEnv`` standing at ``S_t`` (``complete``
        resets it and rebuilds), so this validates the state actually reached rather than a
        replay of it -- and ``SearchState.verify`` shares no state with anything in this module.
        """
        return bool(self.env.state.verify()) and not bool(self.env.legal_mask().any())


def oracle_swap_actions(inst, env: SwapEnv, fr, *, max_steps: int | None = None) -> list[int]:
    """W8's behaviour-cloning target: the truth-ward swap that most raises cluster F1, greedily.

    **Free of MPL.**  ``cluster_f1`` is an intersection of two index sets and a candidate costs
    one ``force_move`` at 2-4 ms, so a whole trajectory is a few hundred milliseconds -- which is
    why BC is a prerequisite for the ``mpl`` arm rather than an optimisation of it.  The action
    space here is ~1,500-3,000 wide and **every action is legal**, where v1's mask did most of
    the pruning; a cold start would spend its GPU budget discovering that most swaps are bad.

    The sequence stops when no truth cluster is still missing -- there is nothing left to teach
    -- so it is ``<= d`` actions, not ``H``.  What the policy does at a converged state is left
    to the entropy schedule, which is safe here for a reason v1 did not have: the return is the
    **best** state seen, so wandering after the record cannot lose it.

    A72's first defect is the trap, and it is avoided by construction: the clone target must be
    in the order the moves are *taken*, not in index order, and each action must be legal at the
    state it is replayed from.  Both hold because the target is generated by stepping this env.

    **``forced`` is not the tabu ring, and it is not redundant with it.**  Without it the target
    cycles: forcing 22 displaces 23, whose re-admission is then the F1-argmax, which displaces 22.
    Measured on the unit fixture at ``[22, 23, 22, 23, 22, 23]`` -- a clone target that teaches a
    two-cycle.  The tabu happens to mask this at ``tenure >= 2``, but the target must not depend on
    a *training* hyper-parameter for its correctness, and "a swap is worth taking once" is the rule
    the corridor's own framing already uses (``A`` is a **subset** of ``D``, not a sequence).
    """
    env.reset()
    truth = [int(v) for v in fr.truth]
    out: list[int] = []
    forced: set[int] = set()
    limit = int(max_steps) if max_steps is not None else int(fr.d)
    for _ in range(limit):
        legal = env.legal_mask()
        missing = [v for v in truth if legal[v] and v not in forced]
        if not missing:
            break
        f1 = [inst.cluster_f1(CO.force_move(env.env, env.selection, v)) for v in missing]
        best = int(missing[int(np.argmax(f1))])
        out.append(best)
        forced.add(best)
        env.step(best)
    return out
