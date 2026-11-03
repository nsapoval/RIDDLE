"""W12 -- top-K decoding over cluster systems, and the two pool controls.

W10 turns per-cluster scores into *one* cluster system: accept in descending score under
:class:`opc.rl.env.ClusterEnv`'s mask until nothing is legal (:func:`opc.rl.nodecls.decode`).
A91 measured what that costs.  The ground set, the decode and the features are all cleared --
an oracle classifier reaches exact recovery on every coverage-complete cell -- and the barrier is
the **ordering**: per-step accuracy decays across the episode, 90-98% of wrong picks were
*avoidable*, and at an avoidable error the right cluster sits a median **2 ranks** below the wrong
one.  A sequential oracle over those same static scores hits the ceiling exactly.

So the information is in the scores and only the choice at the margin is wrong.  This module keeps
the scores and replaces the argmax decode with a **top-K search over systems**, so that a later,
more expensive criterion (a converged rooted-triple MPL fit) has a small pool to choose from
instead of a single answer.

**The canonical order is what makes it efficient, and it is not a heuristic.**  Sort the ground set
by descending score exactly as :func:`~opc.rl.nodecls.decode` does, and require every path to
accept in *strictly increasing rank*.  Every feasible system has a unique descending-rank ordering,
and every prefix of a feasible system is feasible because legality here is **hereditary** -- (L)
and (N3O) are downward-closed and a subset opens no more intersection classes, so the budget is too
(``opc.rl.corridor``'s module docstring states it; ``heuristics._State`` relies on it).  Paths and
feasible systems are therefore in bijection, which buys three things at once:

* **distinct paths give distinct systems**, so nothing has to be deduplicated by hashing;
* at level ``t`` every live entry holds exactly ``t`` clusters, so comparing partial sums at a
  level is comparing like with like -- a beam over sequences of *varying* length would need the
  length normalisation that NLP beam search needs, and this one does not;
* taking the first legal option at every step **is** ``decode``, so ``width=1`` reproduces the
  incumbent exactly.  ``tests/test_rl_beam.py`` pins that, and pins the chain onward to
  :func:`opc.rl.env.greedy_support_rollout` through ``nodecls``' own parity test.

**A deviation permanently excludes what it passed over, and that is the point.**  At an avoidable
error the decode takes a false positive ``p`` while a true cluster ``q`` is still legal; the harm
is that ``p`` blocks ``q``.  Choosing option ``r`` skips options ``1..r-1``, so "take ``q``
instead of ``p``" is one deviation whatever the rank gap -- against the ``gap`` deviations a
ban-one-and-recomplete operator would need.  With avoidable errors running 1.9 a cell at n = 25 and
3.6 at n = 50, and ``max_dev`` bounding how many can be repaired, that difference is the whole
budget.

**The final recompletion is safe for the deviations that matter.**  Skipping ``1..r-1`` also drops
whatever sat between ``p`` and ``q``, which may have been true.  ``recomplete=True`` re-admits
anything still legal at the end of the forward scan, in rank order.  It cannot undo the deviation
that mattered: if ``p`` and ``q`` conflict then taking ``q`` blocks ``p``, and if they do not
conflict then ``p`` never blocked ``q`` and picking it was not the error.  It *can* make two paths
converge, so emission deduplicates on the selection; with ``recomplete=False`` the bijection holds
and the dedup is provably a no-op (asserted in the tests).

**The set score is the model's own, not greedy's.**  Under the pointwise head the log-posterior of
a *set* is ``sum_{v in S} log p_v + sum_{v not in S} log(1 - p_v)``, i.e. ``const + sum_{v in S}
logit(p_v)``; and the head is trained at ``pos_weight ~ 42``, so the calibrated per-cluster term is
``logit(p_v) - log(pos_weight)`` and a cluster pays for itself only above a *true* posterior of
0.5, which is ``sigmoid(z) > 0.977`` (see :data:`DEFAULT_POS_WEIGHT`).  That
score is rank-equivalent to ``decode``'s at a single step -- so the parity above is unaffected --
but it is **not** what greedy maximises over sets, which is what makes the search a different
object rather than a longer look at the same one.  All three readings (``prob``, ``logit``,
``calibrated``) are recorded on every returned system, so a ranking can be recomputed from a
landed artifact without re-running anything.

**Why there is no exact DP.**  The natural one is per overlap component with
``opc.solve.budget_dp``'s (max,+) recombination over the reticulation budget.  It buys nothing
here: ``n_components`` is median 1 (max 3) on all four benchmark grids and ``giant_share`` is
median 1.0 in ``results/rl/scale_census.json``, so there is one component to recombine -- and
inside a component the exact maximum-weight feasible system is the MW-GTCS ILP, which A33 already
found unaffordable at these sizes.  Top-K is strictly harder.  The efficiency here is the canonical
order and heredity, not a decomposition.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

__all__ = [
    "System",
    "SET_SCORES",
    "TERMINALS",
    "DEFAULT_POS_WEIGHT",
    "logits_of",
    "contributions",
    "set_scores",
    "beam_decode",
    "truth_option_profile",
    "gumbel_decode",
    "force_decode",
    "rank_of",
    "rank_order",
]

#: The pooled positive weight the W10 head is trained under (``nodecls.pos_weight``, ~42 at every
#: taxon count because the positive rate is 2.3-2.4% throughout).  It enters the *set* score and
#: nothing else, and it is where the cut sits.
#:
#: Under a ``pos_weight`` BCE the optimum satisfies ``odds(sigmoid(z)) = w * odds(q)`` for the true
#: posterior ``q``, so ``logit(q) = z - log w``.  A cluster therefore *raises* the set
#: log-posterior exactly when ``q > 0.5``, i.e. when ``sigmoid(z) > w / (1 + w) = 0.977``.  Summing
#: the **uncorrected** logits instead cuts at ``sigmoid(z) > 0.5``, which is a true posterior of
#: ``1 / (1 + w) = 0.023`` -- the same mis-reading ``nodecls.raw_set_diagnostics`` records for a
#: naive threshold -- and at that cut almost every legal cluster looks worth taking, so the score
#: degenerates into a proxy for cardinality.
DEFAULT_POS_WEIGHT = 42.0

#: The three readings of a set's score.  ``calibrated`` is the one to rank on -- it is the
#: pointwise model's own set log-posterior up to an additive constant.  ``prob`` and ``logit`` are
#: recorded beside it because they are the two things a reader would otherwise assume were used.
SET_SCORES = ("prob", "logit", "calibrated")

#: What to do with a path that ran out of *forward*-legal candidates.  ``flashlight`` is the
#: default and the only one whose invariant is testable in both directions -- see :func:`_emit`.
TERMINALS = ("flashlight", "complete", "none")

_EPS = 1e-12


@dataclass(frozen=True)
class System:
    """One decoded cluster system, with every reading of its score.

    ``selection`` is ``ClusterEnv.selection``'s own shape -- ascending ``int32`` rows, **kernel
    included**, carrying no order.  ``steps`` is the number of clusters the search itself accepted,
    which is *not* ``selection.size`` on a cell with a non-empty kernel and is the column to read
    when comparing against ``w10_gnn_diag``'s ``n_steps``.
    """

    selection: np.ndarray
    sum_prob: float
    sum_logit: float
    sum_calibrated: float
    deviations: int
    dev_mass: int
    steps: int
    recompleted: int
    maximal: bool

    def score(self, kind: str = "calibrated") -> float:
        if kind == "prob":
            return self.sum_prob
        if kind == "logit":
            return self.sum_logit
        if kind == "calibrated":
            return self.sum_calibrated
        raise ValueError(f"unknown set score {kind!r}; expected one of {SET_SCORES}")

    def key(self) -> bytes:
        return np.ascontiguousarray(self.selection, dtype=np.int32).tobytes()


def logits_of(scores, logits=None) -> np.ndarray:
    """Pre-sigmoid scores, taken rather than reconstructed when the caller has them.

    ``w10_gnn.scores_for`` returns ``sigmoid(z)`` in float32, and ``log(p) - log1p(-p)`` off a
    float32 sigmoid loses the tails -- ``p`` saturates to 1.0 well before ``z`` stops moving, and
    the saturated clusters are exactly the confident true ones whose contribution the set score is
    summing.  Pass ``logits=`` wherever the pre-sigmoid tensor is in hand; the reconstruction is a
    fallback, not the intended path.
    """
    if logits is not None:
        z = np.asarray(logits, dtype=np.float64)
        if z.shape != np.asarray(scores).shape:
            raise ValueError(f"logits {z.shape} do not match scores {np.asarray(scores).shape}")
        return z
    p = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def contributions(scores, kind: str, *, pos_weight: float = DEFAULT_POS_WEIGHT,
                  logits=None) -> np.ndarray:
    """The per-cluster term the set score of ``kind`` sums over the selection."""
    if kind not in SET_SCORES:
        raise ValueError(f"unknown set score {kind!r}; expected one of {SET_SCORES}")
    if kind == "prob":
        return np.asarray(scores, dtype=np.float64)
    z = logits_of(scores, logits)
    if kind == "logit":
        return z
    if pos_weight <= 0.0:
        raise ValueError(f"pos_weight must be positive, got {pos_weight}")
    return z - math.log(float(pos_weight))


def set_scores(selection, scores, *, pos_weight: float = DEFAULT_POS_WEIGHT,
               logits=None) -> tuple[float, float, float]:
    """``(sum_prob, sum_logit, sum_calibrated)`` for one selection."""
    sel = np.asarray(selection, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    z = logits_of(p, logits)
    shift = math.log(float(pos_weight)) if pos_weight > 0 else 0.0
    if sel.size == 0:
        return 0.0, 0.0, 0.0
    return (float(p[sel].sum()), float(z[sel].sum()), float(z[sel].sum() - shift * sel.size))


def rank_order(scores) -> tuple[np.ndarray, np.ndarray]:
    """``(order, rank)`` -- **exactly** ``nodecls.decode``'s, tie-break included.

    A stable ``argsort(-s)`` breaks ties by ascending cluster index; ``argmax(s[legal])`` would
    break them by position within the legal set, which moves as ``S_t`` fills.  Support is heavily
    tied (51% of the tau = 0 ground set sits at support exactly 1), so the two are not the same
    decoder and the parity test would fail.
    """
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    rank = np.empty(s.size, dtype=np.int64)
    rank[order] = np.arange(s.size)
    return order, rank


def rank_of(systems, selection) -> int | None:
    """1-based position of ``selection`` among ``systems``, or ``None`` if it is absent."""
    want = np.ascontiguousarray(np.sort(np.asarray(selection, dtype=np.int32))).tobytes()
    for i, sysm in enumerate(systems):
        if np.ascontiguousarray(np.sort(sysm.selection).astype(np.int32)).tobytes() == want:
            return i + 1
    return None


@dataclass
class _Entry:
    """One live beam path.  ``cursor`` is the rank *already consumed*; the next accept is above it.

    ``forbidden`` is every candidate the path *passed over while it was legal* -- the deviations'
    cost, and the only thing that can make a terminal non-maximal.  A candidate that was already
    illegal when the cursor went past it can never come back (illegality is monotone), and one
    ranked above the cursor is exactly what termination says there are none of.
    """

    state: object
    cursor: int
    score: float
    dev: int
    dev_mass: int
    steps: int
    forbidden: tuple[int, ...] = ()


def _emit(env, entry, mask, scores, contrib, rank, *, terminal: str) -> System | None:
    """Finish one path under ``terminal``, or drop it.  ``mask`` is the entry's own legal mask.

    A path stops when nothing legal ranks **above** its cursor, and that is not the same as
    maximality: a candidate the path passed over may still be legal, in which case the terminal is
    a feasible system with something addable.  Measured on the a97 checkpoint, 22.5% of terminals
    at b25 and 15.1% at b50 are non-maximal, on 20 of 20 cells at both -- and ``env.verify()``
    cannot see it, because they are perfectly feasible.

    ``flashlight`` is the canonical-parent rule: keep the path only if every candidate it
    passed over is now illegal.  Then every survivor is maximal (a cluster outside ``S`` is
    either above
    the cursor, and termination says it is illegal; or passed over, and the test says so; or was
    illegal when passed, and illegality is monotone) **and distinct paths still give distinct
    systems**, so the bijection survives and nothing has to be deduplicated.  ``complete`` instead
    re-admits what was passed over, which also yields maximal systems but lets two paths converge.
    ``none`` emits the terminal as it stands and exists so a test can show the difference.

    At ``width=1`` nothing is ever passed over, so all three agree and the parity with ``decode``
    is untouched.
    """
    env.state = entry.state
    extra = 0
    if terminal == "flashlight":
        if entry.forbidden and bool(mask[np.asarray(entry.forbidden, dtype=np.int64)].any()):
            return None
    elif terminal == "complete":
        while True:
            cand = np.flatnonzero(env.legal_mask())
            if cand.size == 0:
                break
            v = int(cand[np.argmin(rank[cand])])
            ok, klass = entry.state.can_accept(v)
            if not ok:  # pragma: no cover -- the mask and can_accept are parity-tested
                raise AssertionError(f"legal_mask offered {v} and can_accept refused it")
            entry.state.accept(v, int(klass))
            extra += 1
    elif terminal != "none":
        raise ValueError(f"unknown terminal rule {terminal!r}; expected one of {TERMINALS}")
    maximal = not bool(env.legal_mask().any()) if extra else not bool(mask.any())
    sel = entry.state.selection
    return System(
        selection=np.asarray(sel, dtype=np.int32).copy(),
        sum_prob=float(np.asarray(scores, dtype=np.float64)[sel].sum()),
        sum_logit=float(contrib["logit"][sel].sum()),
        sum_calibrated=float(contrib["calibrated"][sel].sum()),
        deviations=int(entry.dev),
        dev_mass=int(entry.dev_mass),
        steps=int(entry.steps),
        recompleted=int(extra),
        maximal=bool(maximal),
    )


def _prune(children, beam: int, per_class: bool):
    """Keep the best ``beam`` children, per deviation class or globally.

    Per class by default, and the reason is that a global cut degenerates.  An early deviation's
    partial sum falls behind by the rank gap immediately and only recovers if the deviation unlocks
    something *later*, so a globally-pruned beam fills with "greedy plus a late tweak" -- the very
    tail of the episode where A91 says accuracy is already worst but where a single swap costs the
    least score.  Quotas per class force the search to spend some of its width on repairs that pay
    off downstream.
    """
    if not children:
        return []
    if not per_class:
        children.sort(key=lambda c: -c[0])
        return children[:beam]
    by_dev: dict[int, list] = {}
    for c in children:
        by_dev.setdefault(c[5], []).append(c)
    kept = []
    for dev in sorted(by_dev):
        rows = by_dev[dev]
        rows.sort(key=lambda c: -c[0])
        kept.extend(rows[:beam])
    return kept


def beam_decode(env, scores, *, k: int = 32, beam: int | None = None, width: int = 8,
                max_dev: int = 2, per_class: bool = True,
                pos_weight: float = DEFAULT_POS_WEIGHT, score: str = "calibrated",
                logits=None, terminal: str = "flashlight",
                include_incumbent: bool = True) -> list[System]:
    """Top-``k`` cluster systems by set score, over the canonical descending-rank order.

    ``width=1, max_dev=0`` is :func:`opc.rl.nodecls.decode` exactly.  ``width<=0`` and
    ``max_dev<0`` mean "unbounded", which is what the exhaustive bijection test uses; on a real
    ground set they do not terminate in any useful time and are not an operating point.

    ``beam`` is the number of paths kept **per deviation class** per level; ``None`` defaults it to
    ``k`` and ``0`` means *unbounded*, i.e. exhaustive.  With ``beam`` set,
    the live set is therefore at most ``beam * (max_dev + 1)`` and each one costs one
    ``legal_mask`` per level.  That is the whole cost model -- the mask is ~1.1 ms at
    ``|C| = 4,652`` and everything else in the loop is a clone of three small arrays.

    ``include_incumbent`` keeps the zero-deviation path -- ``decode``'s own answer -- in the
    returned list even when ``k`` others outscore it.  It is not a courtesy: without it a pool's
    ``f1_ceiling`` can sit *below* the incumbent's F1, and a ceiling that does not bound the thing
    it is quoted against says nothing.  The zero-deviation lineage is unique and its class quota is
    at least one, so it is always *computed*; this is only about what survives the final cut.

    The env's state is swapped, not stepped: ``ClusterEnv.step`` recomputes the full mask on every
    accept and expansion needs one mask per *parent*, not one per candidate.  ``env.state`` is
    restored on the way out, because an ``InstancePool`` built with ``swap_tabu`` hands the same
    ``ClusterEnv`` to a ``SwapEnv`` and a foreign state left behind would surface as a wrong walk
    rather than as an error.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    if score not in SET_SCORES:
        raise ValueError(f"unknown set score {score!r}; expected one of {SET_SCORES}")
    if terminal not in TERMINALS:
        raise ValueError(f"unknown terminal rule {terminal!r}; expected one of {TERMINALS}")
    if k <= 0:
        return []
    width = env.n if width <= 0 else int(width)
    max_dev = env.n if max_dev < 0 else int(max_dev)
    # ``beam=0`` is unbounded and ``beam=None`` falls back to ``k``.  The distinction matters:
    # the exhaustive search is the only mode with **no free parameter**, and at ``max_dev=1`` it is
    # what the pool should be -- there is nothing to tune and nothing to blame a null result on.
    # Silently reading 0 as "keep k" would have pruned a 9,000-member D=2 pool to the output size.
    keep = int(k) if beam is None else (sys.maxsize if beam <= 0 else int(beam))

    contrib = {kind: contributions(s, kind, pos_weight=pos_weight, logits=logits)
               for kind in SET_SCORES}
    key = contrib[score]
    _order, rank = rank_order(s)

    saved = env.state
    try:
        env.reset()
        live = [_Entry(env.state.copy(), -1, 0.0, 0, 0, 0)]
        done: list[System] = []
        incumbent: System | None = None
        while live:
            children = []
            for e in live:
                env.state = e.state
                mask = env.legal_mask()
                cand = np.flatnonzero(mask)
                if cand.size:
                    cand = cand[rank[cand] > e.cursor]
                if cand.size == 0:
                    sysm = _emit(env, e, mask, s, contrib, rank, terminal=terminal)
                    if sysm is not None:
                        if e.dev == 0:
                            incumbent = sysm
                        done.append(sysm)
                    continue
                cand = cand[np.argsort(rank[cand], kind="stable")]
                lim = min(width, cand.size) if e.dev < max_dev else 1
                for r in range(lim):
                    v = int(cand[r])
                    ok, klass = e.state.can_accept(v)
                    if not ok:  # pragma: no cover -- mask/can_accept parity is tested
                        raise AssertionError(f"legal_mask offered {v} and can_accept refused it")
                    children.append((e.score + float(key[v]), e, v, int(klass), int(rank[v]),
                                     e.dev + (1 if r else 0), r, tuple(int(x) for x in cand[:r])))
            live = []
            for value, parent, v, klass, vrank, dev, r, skipped in _prune(children, keep,
                                                                          per_class):
                st = parent.state.copy()
                st.accept(v, klass)
                live.append(_Entry(st, vrank, value, dev, parent.dev_mass + r,
                                   parent.steps + 1, parent.forbidden + skipped))
    finally:
        env.state = saved

    # Ties are broken toward *fewer* deviations, because recompletion lets a deviating path land
    # back on a set a shallower one also reaches.  The two carry identical scores, so without this
    # the reported ``deviations`` would be whichever path the sort happened to visit first.
    ranked, seen = [], set()
    for sysm in sorted(done, key=lambda d: (-d.score(score), d.deviations)):
        h = sysm.key()
        if h in seen:
            continue
        seen.add(h)
        ranked.append(sysm)
    out = ranked[:k]
    if include_incumbent and incumbent is not None:
        if all(d.key() != incumbent.key() for d in out):
            out = sorted(out[:max(k - 1, 0)] + [incumbent],
                         key=lambda d: (-d.score(score), d.deviations))
    return out


def truth_option_profile(env, scores, truth_rows) -> dict:
    """What the beam would have to be to reach ``truth_rows``: ``(width, deviations)``, per cell.

    Walk the truth in rank-increasing order -- the canonical order of the system it is -- and at
    each step record ``r``, its index among the legal candidates ranked above the cursor.  Then
    ``max(r) + 1`` is the branch width the beam needs and ``#{r > 0}`` is the number of deviations,
    which is ``max_dev``.  One decode per cell, no model beyond the scores, no MPL.

    **This is not A91's ``avoidable`` count and must not be read as one.**  That counts errors
    along the *decode's* trajectory; once a beam deviates it is on a different trajectory and the
    two diverge.  This is the quantity a beam is actually sized by, and nothing else in the
    repository computes it.

    Returns ``feasible=False`` when a truth cluster is not legal at its turn.  That should not
    happen -- feasibility is hereditary, so any order of a feasible set is acceptable -- so a False
    here is a statement about the truth or about the mask, not about the ordering.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    _order, rank = rank_order(s)
    rows = np.asarray(truth_rows, dtype=np.int64)
    rows = rows[np.argsort(rank[rows], kind="stable")]

    saved = env.state
    try:
        env.reset()
        st = env.state
        cursor, opts, feasible = -1, [], True
        for v in rows:
            v = int(v)
            if st.in_s[v]:
                continue  # a kernel row, seeded by ``reset``: accepted, never an accept
            cand = np.flatnonzero(env.legal_mask())
            cand = cand[rank[cand] > cursor]
            hit = np.flatnonzero(cand == v)
            if hit.size == 0:
                feasible = False
                break
            cand = cand[np.argsort(rank[cand], kind="stable")]
            opts.append(int(np.flatnonzero(cand == v)[0]))
            ok, klass = st.can_accept(v)
            if not ok:  # pragma: no cover
                raise AssertionError(f"legal_mask offered {v} and can_accept refused it")
            st.accept(v, int(klass))
            cursor = int(rank[v])
        maximal = feasible and not bool(env.legal_mask().any())
    finally:
        env.state = saved
    return {
        "truth_path_feasible": bool(feasible),
        "truth_maximal": bool(maximal),
        "truth_steps": int(len(opts)),
        "width_needed": int(max(opts) + 1) if opts else 0,
        "dev_needed": int(sum(1 for r in opts if r > 0)),
        "dev_mass_needed": int(sum(opts)),
    }


def gumbel_decode(env, scores, *, k: int = 32, temperature: float = 1.0, rng=None,
                  pos_weight: float = DEFAULT_POS_WEIGHT, score: str = "calibrated",
                  logits=None) -> list[System]:
    """``k`` decodes under Gumbel-perturbed scores.  **The control that prices the beam.**

    Adding i.i.d. Gumbel noise to the logits and decoding in the perturbed order is Plackett-Luce
    sampling over the accept order, so this draws systems from the model's own distribution with no
    search at all.  Draw 0 is unperturbed, i.e. exactly ``decode``, so the incumbent is always in
    the pool and any beam gain is measured against a pool that already contains it.

    It exists for the same reason ``nodecls.band_loss_terms`` exists beside the decode-derived
    terms: a diverse pool is easy to produce, and without a cheap generator beside it a top-K
    result would price the *pool* and be reported as the beam.
    """
    from opc.rl import nodecls as NC

    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    rng = rng or np.random.default_rng(0)
    z = logits_of(s, logits)
    contrib = {kind: contributions(s, kind, pos_weight=pos_weight, logits=logits)
               for kind in SET_SCORES}
    seen, out = set(), []
    for i in range(int(k)):
        draw = z if i == 0 else z / float(temperature) + rng.gumbel(size=z.shape)
        sel = NC.decode(env, draw)
        h = np.ascontiguousarray(sel, dtype=np.int32).tobytes()
        if h in seen:
            continue
        seen.add(h)
        out.append(System(
            selection=np.asarray(sel, dtype=np.int32).copy(),
            sum_prob=float(s[sel].sum()),
            sum_logit=float(contrib["logit"][sel].sum()),
            sum_calibrated=float(contrib["calibrated"][sel].sum()),
            deviations=0 if i == 0 else -1,
            dev_mass=0 if i == 0 else -1,
            steps=int(sel.size),
            recompleted=0,
            maximal=True,
        ))
    if len(out) * 4 < int(k):
        raise ValueError(
            f"the Gumbel pool collapsed: {len(out)} distinct of {k} draws at temperature "
            f"{temperature}. A69 measured 48 of 48 perturbed re-completions coming back "
            "byte-identical to greedy, and a ceiling computed over K copies of one system is a "
            "statement about the sampler. Raise --gumbel-temp or lower k.")
    return sorted(out, key=lambda d: -d.score(score))[:int(k)]


def force_decode(env, scores, *, k: int = 32, width: int = 0,
                 pos_weight: float = DEFAULT_POS_WEIGHT, score: str = "calibrated",
                 logits=None) -> list[System]:
    """The depth-1 force-and-recomplete neighbourhood of the incumbent decode.

    ``corridor.force_move`` with the model's rank in place of support's: push one cluster to the
    top of the priority order, re-admit whatever of the incumbent still fits, complete to maximal.
    This is W7's move operator and W9's G-S3 baseline, so the comparison it supports is one this
    repository has already made in a different guise -- and it is the *forcing-in* neighbourhood,
    where the beam's deviation is a forcing-*out*.  The two cover different repairs, which is why
    both are here.

    ``width`` bounds how many candidates are tried (the highest-scoring ones not already in the
    incumbent); ``0`` means every one of them, which is affordable only at n = 25.
    """
    from opc.rl import corridor as CO
    from opc.rl import nodecls as NC

    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (env.n,):
        raise ValueError(f"scores must be ({env.n},), got {s.shape}")
    _order, rank = rank_order(s)
    contrib = {kind: contributions(s, kind, pos_weight=pos_weight, logits=logits)
               for kind in SET_SCORES}

    base = NC.decode(env, s)
    pool = [base]
    outside = np.setdiff1d(np.arange(env.n, dtype=np.int64), np.asarray(base, dtype=np.int64))
    outside = outside[np.argsort(rank[outside], kind="stable")]
    if width > 0:
        outside = outside[:width]
    for v in outside:
        pool.append(CO.force_move(env, base, int(v), key=s))
    seen, out = set(), []
    for sel in pool:
        h = np.ascontiguousarray(np.sort(np.asarray(sel, dtype=np.int32))).tobytes()
        if h in seen:
            continue
        seen.add(h)
        out.append(System(
            selection=np.asarray(sel, dtype=np.int32).copy(),
            sum_prob=float(s[sel].sum()),
            sum_logit=float(contrib["logit"][sel].sum()),
            sum_calibrated=float(contrib["calibrated"][sel].sum()),
            deviations=-1,
            dev_mass=-1,
            steps=int(np.asarray(sel).size),
            recompleted=0,
            maximal=True,
        ))
    return sorted(out, key=lambda d: -d.score(score))[:int(k)]
