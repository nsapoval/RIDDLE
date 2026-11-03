"""W5's terminal reward, in two arms.  ``RL_PLAN.md`` §2 ("Reward") and §0.

**Arm ``mpl`` is the objective the study is about**: realise the cluster selection as a level-1
network, re-optimise ``(lambda, gamma)``, return ``sup MPL``.  It is what a deployed policy would
be trained against, because at deployment there is no ``C(N_true)``.

**Arm ``f1`` is cluster F1 against ``C(N_true)``, and it costs nothing.**  No model build, no fit,
no GPU -- an intersection of two index sets.  It exists because the two arms answer two questions
that the plan has no other way to separate:

* *Can this policy class and this loop beat greedy-by-support at all?*  Recovering the truth given
  the truth is strictly easier than recovering it through MPL, so a failure on ``f1`` is a failure
  of capacity or of the training loop, and settles nothing about the objective.
* *Does maximising MPL move towards the truth?*  A58 measured eight unlearned searches over four
  cells and **no MPL-guided search beat greedy-by-support on either axis, 0 of 68 runs**; §4bis
  reproduced it in network space, where edge moves from greedy improved MPL and made recovery
  **worse on 4 of 4 cells**.  If ``f1`` learns and ``mpl`` does not, that is A58 reproduced under
  learning -- which is the study's real question, and an honest answer to it.

``f1`` is a training-time oracle and is **not** a deployable reward.  Nothing in the evaluation
tier may use it as a *reward*; it is a metric there.

Three exact devices keep the MPL arm affordable.  None of them approximates anything:

1. **Dedup before fitting.**  A69 measured 76 distinct networks in 128 near-greedy rollouts, and a
   behaviour-cloned policy starts exactly there.  Distinct selections are fitted; the values are
   broadcast back.
2. **A disk cache** keyed on everything that changes the answer -- the cell, the *closed* cluster
   system, the triple count, the optimiser, its iteration count, its restart count and its seed.
   That is A52's rule ("a cache key must include anything that changes the answer"), and the
   reason the schedule is in the key is that a truncated fit is a *lower bound* whose tightness is
   the schedule (RL_PLAN §1.2): two schedules give two different numbers for one topology.
3. **Batched fitting with tiled restarts**, which is :func:`opc.rl.mpl_torch.profile_mpl` -- four
   starts cost about what one costs, and A60 measured a single start as worth up to 10,905 log
   units on one pilot cell in four.

**An unrealisable selection is not an error and must not be dropped.**  ``search.realise`` returns
``None`` when the closure is malformed, and a random policy produces those.  They are scored at
:data:`UNREALISABLE`, a finite floor, because a ``nan`` reward poisons a group's mean and a dropped
rollout silently changes ``K`` -- and a group-normalised advantage divides by the group's own
spread.  The rate is logged; A65 measured it at 0 of 128 on both probe arms, so a non-zero rate is
itself information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opc.followup import search

__all__ = ["UNREALISABLE", "f1_rewards", "MPLReward", "topology_key",
           "winsorise_unrealisable", "record_increments", "best_seen", "star_value"]

#: How much longer, and how many more starts, a re-fit of a provably non-converged network gets.
#: Applied to the 3.3% that need it, so the cost is negligible against the batch.
REPAIR_ITERS = 4
REPAIR_RESTARTS = 2

#: The reward of a selection that stands for no level-1 network.  Not ``-inf`` and not ``nan``:
#: both destroy the group statistics that §4's advantage is built from.  Relative to a real MPL
#: (order -10^7 at n = 50) this is far below any attainable value while staying finite.
UNREALISABLE = -1e12


def winsorise_unrealisable(values: np.ndarray) -> np.ndarray:
    """Pull :data:`UNREALISABLE` back to "worst in this group", before the advantage is taken.

    The raw value has to be a *finite constant* so it can be cached and compared across groups, and
    it has to be far below any attainable MPL so it never outranks a real network.  Both are true
    of ``-1e12`` -- and they are exactly what makes it poison for a **group-normalised** advantage:
    one unrealisable rollout among sixteen moves the group mean by 6e10 and inflates its standard
    deviation by the same order, so every real rollout's advantage collapses to the same number and
    the whole group teaches the policy nothing except "not that one".

    Mapping it to ``min(finite) - range(finite) - 1`` keeps the ordering (it is still strictly the
    worst), keeps the scale (the group's spread is unchanged to within a factor of two), and is a
    deterministic function of the group, so nothing about the cache changes.  A group with *no*
    realisable member is left alone: there is no scale to preserve, and the advantage is zero
    either way.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    bad = v <= UNREALISABLE / 2.0
    if not bad.any() or bad.all():
        return v
    lo, hi = float(v[~bad].min()), float(v[~bad].max())
    v[bad] = lo - (hi - lo) - 1.0
    return v


def star_value(obs) -> float:
    """``sum(rho) * ln(1/3)`` -- the star's log-likelihood, and a **provable lower bound on the
    optimum of every level-1 network on the same data**.

    Driving every internal branch length to zero sends every triple probability to 1/3 whatever the
    topology, and that point is inside the box every parameter is optimised in.  So for any network
    ``N``, ``sup_{lambda, gamma} MPL(N) >= MPL(star)``.  A *fitted* value below it is therefore not
    a bad network -- it is a **failed optimisation**, provably, with no modelling judgement
    involved.

    That makes it the cheapest correctness check in the reward path, and W8 needed it: at
    ``100 x 2`` over the topologies an exploring swap policy visits, **277 of 8,434 fitted values
    (3.3%) landed below the star**, the worst by **9.6e8 log units** -- against a star-to-optimum
    range of about 1.6e7.  One of those landing on the *start* state corrupts every record
    increment in that gradient step, and it is invisible in a training curve: it reads as an
    enormous reward improvement.  See A78.

    It is a floor on the optimum, not on any particular parameter vector, so it detects
    catastrophic non-convergence and says nothing about a fit that is merely loose.
    """
    return float(np.asarray(obs, dtype=np.float64).sum() * np.log(1.0 / 3.0))


def best_seen(values) -> np.ndarray:
    """The running maximum of a trajectory's values -- ``M*_t = max_{s <= t} M_s``.

    ``np.maximum.accumulate`` is a real ufunc accumulate and stays on numpy's fast path; this is
    not the ``ufunc.at`` trap recorded in ``env.legal_mask`` and ``policy.graph_tensors``.
    """
    return np.maximum.accumulate(np.asarray(values, dtype=np.float64))


def record_increments(values, deadband: float = 0.0) -> np.ndarray:
    """W8's per-step reward: how much step ``t`` improved on the **record standing before it**.

    ``r_t = max(0, M_{t+1} - M*_t)``, and the identity that makes the two credit arms comparable
    is exact at ``deadband = 0``::

        M*_{t+1} = M*_t + r_t      =>      sum_t r_t == max_t M_t - M_0

    So ``--credit dense`` and ``--credit flat`` optimise the **same** return -- the best system
    seen along the rollout -- and differ only in where the credit lands.  That is the whole reason
    the dense arm is affordable: the best-seen return already needs an MPL value at every visited
    state, so the dense signal is spent value rather than new value.

    **The deadband is a noise floor, not a tuning knob.**  A76 §12.6 measured a network's fitted
    MPL moving **65 log units** with batch composition alone -- ``_initial_vectors`` draws each
    restart's perturbation as one contiguous block over the whole batch, so a network's starting
    points depend on what precedes it, and ``MPLReward``'s cache key does not include the batch.
    Over the same block, §12's *unresolved* steps had a median ``|dMPL|`` of **136** against the
    resolved items' **43,231**.  An increment inside that band is fit noise, and training on it
    teaches the policy to chase the optimiser.

    The record still advances on a sub-band improvement (``M*`` is the true running max), so the
    band costs the identity rather than the accounting: ``sum_t r_t <= max_t M_t - M_0``, with
    equality iff every improvement cleared the band.  The gap is the leak, it is computable, and
    ``opc.rl.train`` logs it as ``deadband_leak`` rather than letting it be assumed away.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return np.zeros(0, dtype=np.float64)
    star = np.maximum.accumulate(v)
    inc = v[1:] - star[:-1]
    return np.where(inc > float(deadband), inc, 0.0)


def topology_key(closed_words: np.ndarray) -> str:
    """A content hash of the **closed** cluster system -- the thing the MPL actually depends on.

    Two different selections can close to the same system (the closure adds implied clusters), and
    they have the same likelihood, so keying on the raw selection would miss cache hits that are
    exact rather than approximate.  Rows are sorted before hashing because the closure's row order
    is an artefact of the order clusters were stacked in.
    """
    a = np.ascontiguousarray(np.asarray(closed_words))
    order = np.lexsort(a.T[::-1])
    return hashlib.blake2b(a[order].tobytes(), digest_size=16).hexdigest()


def f1_rewards(inst, selections) -> np.ndarray:
    """Arm ``f1``: cluster F1 against ``C(N_true)``, one per selection.  No MPL, no GPU."""
    return np.asarray([inst.cluster_f1(s) for s in selections], dtype=np.float64)


@dataclass
class MPLStats:
    n_scored: int
    n_distinct: int
    n_cached: int
    n_unrealisable: int
    seconds_realise: float
    seconds_build: float
    seconds_fit: float
    n_shapes_pooled: int
    restart_gain_median: float
    tail_movement_max: float
    #: Networks whose parameter vector is empty -- the star, A49.  Scored in closed form.
    n_parameter_free: int = 0
    #: Topologies actually sent to the optimiser.  **Not** ``n_distinct - n_cached``: ``n_cached``
    #: counts cache-hit *positions*, of which there can be more than there are topologies.
    n_fitted: int = 0
    #: ``profile_mpl`` calls this batch was split into.  Each one captures its own CUDA graph
    #: (21.5 s, A65), so this is a cost term and not bookkeeping -- see :meth:`MPLReward.batch`.
    n_chunks: int = 0
    #: The largest per-chunk pooled-shape count.  Memory tracks this, not ``n_shapes_pooled``.
    shapes_max_chunk: int = 0
    #: Fits that came back below :func:`star_value` -- provably non-converged -- and were
    #: re-fitted at ``REPAIR_ITERS x REPAIR_RESTARTS`` times the schedule.  **A non-zero rate is
    #: information**, and it is what W8's first training run had no way to see (A78).
    n_nonconverged: int = 0
    #: Of those, the ones still below the star after the repair.  They are clamped to it.
    n_unrepaired: int = 0


class MPLReward:
    """Arm ``mpl``: ``sup_{lambda, gamma} MPL`` of the network a selection stands for.

    ``pool`` is an optional ``multiprocessing.Pool`` used for ``TripleModel.__init__``, which is
    ``C(n,3)`` pure-Python ``restrict_to_triple`` calls -- 1.58 s single-core at n = 50 (A53/A65)
    and the second-largest term in a reward after the fit itself.
    """

    #: Networks per ``profile_mpl`` call.  **This is a memory bound, not a tuning knob.**
    #: ``probe_mpl_fast.py`` at batch 128 reached **38-42 GB on a 48 GB card** (A69), and the swap
    #: MDP presents ``B x K x (H + 1)`` states per gradient step -- ~450 at ``B = 2, K = 4`` -- so
    #: an unchunked call is an out-of-memory rather than a slow step.  ``scripts/w7_corridor.py``
    #: already wrapped this class to get the same effect; the wrapper's reason is now the class's.
    #:
    #: It is not free to make it small -- ``mpl_torch._graphed_step`` captures a CUDA graph **per
    #: call** -- but A65's 21.5 s is for a 1,047-1,288-shape batch and the capture scales with the
    #: DP graph.  G-S1's P3, on a 325-shape near-greedy batch: chunk 64 / 128 / 256 at **0.83 /
    #: 1.66 / 2.27 GB** and **73.3 / 72.1 / 63.8 s** -- ~15% for three calls against one.  Set it
    #: as large as the card allows; ``scripts/probe_swap_cost.py --blocks P3`` measures it.
    DEFAULT_CHUNK = 128

    def __init__(self, *, device: str = "cuda", iters: int = 300, restarts: int = 4,
                 seed: int = 0, optimiser: str = "adam", cache_dir: Path | None = None,
                 pool=None, chunk: int = DEFAULT_CHUNK):
        self.chunk = max(1, int(chunk))
        self.device = device
        self.iters = int(iters)
        self.restarts = int(restarts)
        self.seed = int(seed)
        self.optimiser = optimiser
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.pool = pool
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, float] = {}

    # -- the cache key.  Everything here changes the number; nothing here is cosmetic. ----------
    def _key(self, inst, topo: str) -> str:
        parts = [topo, inst.key, str(inst.n), str(inst.index.shape[0]),
                 self.optimiser, str(self.iters), str(self.restarts), str(self.seed)]
        # **Which triples, not just how many** (A52 in one more place).  The row count alone
        # separates a full index from a subsampled one, but two *different* subsamples of the same
        # size would share a key and serve each other's values.  ``index_tag`` is a content hash
        # set by whatever did the subsampling; it is absent on a full index, and the key is then
        # byte-identical to what it always was -- so no existing cache entry is invalidated.
        tag = getattr(inst, "index_tag", "")
        if tag:
            parts.append(tag)
        # **And which gene trees** (A52 again, A100's arm).  ``inst.key`` carries no arm -- it is
        # the *cell*, and the same cell exists under both the simulated and the estimated trees
        # with the same taxon count and the same number of triples.  Everything in the key above is
        # therefore identical across arms while ``inst.obs`` -- the thing the likelihood is a
        # function of -- is not: measured on ``n25_rep80``, up to 23 counts per triple apart.  An
        # estimated-arm run would have read the simulated arm's likelihoods back and said nothing.
        # Appended only when non-empty, like ``index_tag``, so every landed entry keeps its key.
        arm = getattr(inst, "arm", "")
        if arm:
            parts.append(arm)
        return "|".join(parts)

    def _path(self, key: str) -> Path:
        h = hashlib.blake2b(key.encode(), digest_size=16).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def _read(self, key: str):
        if key in self._mem:
            return self._mem[key]
        if not self.cache_dir:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        row = json.loads(p.read_text())
        if row.get("key") != key:          # hash collision, or a stale layout: recompute.
            return None
        self._mem[key] = float(row["value"])
        return self._mem[key]

    def _write(self, key: str, value: float) -> None:
        self._mem[key] = float(value)
        if not self.cache_dir:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "value": float(value)}))
        tmp.rename(p)

    def __call__(self, inst, selections) -> tuple[np.ndarray, MPLStats]:
        """One group.  Delegates to :meth:`batch`, which is where the work is."""
        (values,), stats = self.batch([(inst, selections)])
        return values, stats

    def batch(self, items) -> tuple[list, MPLStats]:
        """``[(instance, selections), ...]`` -> one array of values per item, in **one** fit.

        **Every group in a gradient step is fitted together, not group by group.**  The evaluator's
        cost tracks the number of *pooled trinet shapes* rather than the number of rows (A65), and
        shapes are shared heavily across networks -- 76 distinct near-greedy networks induce 272
        pooled shapes, a per-network median of 36 (A69).  Pooling eight groups therefore costs far
        less than eight pooled fits, and the CUDA graph is captured once instead of eight times
        (21.5 s to capture, A65).

        Every instance in one call must share a triple index, which means one ``n`` per call -- the
        trainer samples its instance batch at a single ``n`` for exactly this reason.  ``obs`` is
        stacked per network, which ``profile_mpl`` already supports and ``w0_baseline`` already
        relies on.

        **Dedup first, then chunk.**  The cache read and the closed-system dedup run over the whole
        call, so nothing is fitted twice by being split; only the *fitting* is chunked, at
        :data:`DEFAULT_CHUNK` networks per ``profile_mpl``.  That ordering matters for the swap
        MDP, where the ``K`` rollouts of a group all begin at greedy and the depth-1 shell recurs
        across steps: chunking before dedup (which is what wrapping this class from outside did)
        would fit the same topology once per chunk it appeared in.
        """
        import time

        from . import mpl_torch as MT

        t0 = time.perf_counter()
        out = [np.full(len(sels), UNREALISABLE, dtype=np.float64) for _inst, sels in items]
        n_unreal = 0
        n_cached = 0
        distinct: set[tuple[int, str]] = set()
        # (item, position) -> closed system key; and the de-duplicated work list.
        todo: dict[tuple[int, str], tuple[object, object]] = {}
        where: list[tuple[int, int, tuple[int, str]]] = []
        for gi, (inst, sels) in enumerate(items):
            for j, sel in enumerate(sels):
                got = search.realise(inst.gs.words, sel, inst.n, list(inst.labels))
                if got is None:
                    n_unreal += 1
                    continue
                ck = (gi, self._key(inst, topology_key(got[1])))
                distinct.add(ck)
                where.append((gi, j, ck))
                hit = self._read(ck[1])
                if hit is not None:
                    out[gi][j] = hit
                    n_cached += 1
                elif ck not in todo:
                    todo[ck] = (inst, got[0])
        t_real = time.perf_counter() - t0

        n_scored = sum(len(sels) for _inst, sels in items)
        stats = MPLStats(n_scored=n_scored, n_distinct=len(distinct), n_cached=n_cached,
                         n_unrealisable=n_unreal, seconds_realise=t_real, seconds_build=0.0,
                         seconds_fit=0.0, n_shapes_pooled=0, restart_gain_median=0.0,
                         tail_movement_max=0.0)
        stats.n_fitted = len(todo)
        if not todo:
            return out, stats

        order = list(todo.items())
        values = np.empty(len(order), dtype=np.float64)
        gains, tails = [], []
        for lo in range(0, len(order), self.chunk):
            part = order[lo:lo + self.chunk]
            t0 = time.perf_counter()
            payload = [(net, inst.index) for _ck, (inst, net) in part]
            if self.pool is not None and len(payload) > 1:
                models = self.pool.map(_build_model, payload, chunksize=1)
            else:
                models = [_build_model(x) for x in payload]
            stats.seconds_build += time.perf_counter() - t0
            pooled = len({g.shape.key for m in models for g in m.groups})
            stats.n_shapes_pooled += pooled
            stats.shapes_max_chunk = max(stats.shapes_max_chunk, pooled)
            stats.n_chunks += 1

            # **A network with no free parameters is scored in closed form, not optimised.**
            # ``search.realise`` succeeds on the empty selection -- A49: "the empty selection is
            # the star, not an error" -- and the star has no internal branch and no hybrid, so its
            # parameter vector is *empty*.  Handing that to a gradient optimiser is not slow, it
            # raises: ``(-val.sum()).backward()`` has nothing to differentiate.  Measured at n = 6,
            # the star's ``log_likelihood`` reproduces ``sum(rho) * ln(1/3)`` to 1e-13, which is
            # A49's own identity arrived at from the other side.
            free, fixed = [], []
            for j, m in enumerate(models):
                free_params = len(m.params.lengths) + len(m.params.gammas)
                (free if free_params else fixed).append(j)
            vals = np.empty(len(models), dtype=np.float64)
            for j in fixed:
                vals[j] = float(models[j].log_likelihood(part[j][1][0].obs, models[j].params))
            stats.n_parameter_free += len(fixed)

            if free:
                obs = np.stack([part[j][1][0].obs for j in free])
                t0 = time.perf_counter()
                fit = MT.profile_mpl if self.optimiser == "adam" else MT.profile_mpl_lbfgs
                res = fit([models[j] for j in free], obs, device=self.device,
                          restarts=self.restarts, iters=self.iters, seed=self.seed)
                stats.seconds_fit += time.perf_counter() - t0
                gains.append(float(res.restart_gain.median()))
                tails.append(float(res.tail_movement.max()))
                got = res.value.detach().cpu().numpy().astype(float)
                for slot, j in enumerate(free):
                    vals[j] = float(got[slot])
            # **The star check, and it is exact rather than a heuristic.**  ``sup MPL(N)`` is at
            # least the star's value for *every* level-1 network (``star_value``), so anything
            # below it is a failed optimisation.  Re-fit those alone -- it was 3.3% of fits at
            # 100 x 2 (A78) -- and clamp anything that still fails, because a lower bound that is
            # merely loose is survivable and one that is 9.6e8 log units wrong is not.
            stars = np.array([star_value(part[j][1][0].obs) for j in range(len(models))])
            broke = [j for j in free if vals[j] < stars[j]]
            if broke:
                stats.n_nonconverged += len(broke)
                res = fit([models[j] for j in broke],
                          np.stack([part[j][1][0].obs for j in broke]), device=self.device,
                          restarts=self.restarts * REPAIR_RESTARTS,
                          iters=self.iters * REPAIR_ITERS, seed=self.seed + 1)
                got = res.value.detach().cpu().numpy().astype(float)
                for slot, j in enumerate(broke):
                    vals[j] = float(got[slot])
                still = [j for j in broke if vals[j] < stars[j]]
                stats.n_unrepaired += len(still)
                for j in still:
                    vals[j] = float(stars[j])
            values[lo:lo + len(part)] = vals
        stats.restart_gain_median = float(np.median(gains)) if gains else 0.0
        stats.tail_movement_max = float(max(tails)) if tails else 0.0

        for j, (ck, _payload) in enumerate(order):
            self._write(ck[1], float(values[j]))
        for gi, j, ck in where:
            hit = self._read(ck[1])
            if hit is not None:
                out[gi][j] = hit
        return out, stats


def _build_model(payload):
    from opc.mpl import likelihood
    net, index = payload
    return likelihood.TripleModel(net, index=index)
