"""W2 -- ``ClusterEnv``: the MDP of ``RL_PLAN.md`` §2, over a tau = 0 ground set.

**Almost all of this already existed.**  ``opc.solve.heuristics._State`` is the exact (L)/(N3O)/
budget action mask and says in its own docstring that it exists so "a policy would inherit a
checked implementation rather than write a second one"; ``opc.followup.search.SearchState`` lifts
it to the whole ground set, carries the cross-component budget, and re-derives feasibility with the
independent validator ``heuristics.check``.  This module adds two things and reuses the rest.

**1. A vectorised legal mask, because the scan is the hot loop now.**  ``SearchState.legal()``
calls ``can_accept`` once per row of the ground set, in Python, and its docstring justifies that
against an MPL evaluation costing milliseconds to seconds.  At tau = 0 the ground set is |C| median
3,112 / p95 14,601 / max 16,567 (A31), an episode is ~50 steps, and a GRPO step runs 128 rollouts
-- so that scan runs up to 10^8 times per gradient step against a reward that now costs ~1 ms per
network (W1).  The justification has inverted.

The vectorisation rests on one observation: **every condition is a statement about edges incident
to the accepted set**, and the accepted set is at most ``n - 1 + O(k)`` clusters -- about fifty at
n = 50, whatever |C| is.  So the mask never touches the full 577,000-edge list.  Let ``S`` be the
accepted vertices and ``B`` the ``|C| x |S|`` incidence "v is adjacent to accepted s".  Then

* **(L) at v** -- all selected incident edges carry one intersection class -- is a min/max over the
  nonzeros of a ``|C| x |S|`` class matrix;
* **(L) at a committed neighbour** is the same min/max over ``commit[s]``, masked to the committed;
* **(N3O)** -- two accepted neighbours of ``v`` that are adjacent to each other -- is
  ``((B @ A_S) * B).sum(1) > 0``, where ``A_S`` is the ``|S| x |S|`` adjacency *within* the
  accepted set.  Two small matrix products; no triangle enumeration anywhere.
* **budget** is a lookup of the candidate's class in the set of classes already opened.

That last point has to be got right for a different reason: eager triangle enumeration is what put
all sixteen CU <= 0.5 cells over A33's 120 s cap, so **nothing here may call
``GroundSet.triangles()``**.  It does not.

**2. The gym-shaped surface RL_PLAN §W2 asks for**, with the init policy configurable as
``empty`` (default) | ``greedy`` | ``given(S)``, which is what keeps warm-started and repair-style
rollouts in one codebase rather than two.

**There is no remove action** (RL_PLAN §2: actions are ``a_t in legal(S_t)``, termination is
``legal(S_t) = {}``, no STOP).  ``drop`` is therefore not on the hot path and is left to
``SearchState``, which rebuilds commitments from scratch when it is used.

**G-R0 is parity, and it is exact by construction only if it is checked.**  ``argmax``-support
under this env must reproduce ``search._greedy_support`` selection for selection, and every
terminal state must satisfy ``heuristics.check`` -- which shares no state with ``_State``,
recomputing (L) by a lexsort groupby and (N3O) as ``nnz((A @ A) . A) / 6``.
:func:`greedy_support_rollout` and :meth:`ClusterEnv.verify` are those two checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc.followup import search

__all__ = ["OverlapCSR", "ClusterEnv", "rollout", "greedy_support_rollout"]

#: Sentinel for "this vertex is in the kernel" -- degree 0, so no overlap partner anywhere and no
#: constraint it could violate.  ``SearchState.can_accept`` returns ``(True, -1)`` for these.
KERNEL = -1


@dataclass
class OverlapCSR:
    """The whole ground set's overlap graph as one CSR, with globally unique intersection classes.

    ``opc.solve.overlap.Component`` stores local endpoints and a *per-component* ``edge_class``
    indexing that component's own ``hcand``.  Two components can therefore carry the same local
    class number for different intersections.  That is harmless where the classes are only ever
    compared among edges incident to one vertex -- which is all (L) does, and incident edges are
    always in one component -- but a global mask compares them in one array, so the classes are
    offset per component here.  The offset is also what makes the *budget* count right: the number
    of reticulations is the number of distinct classes opened anywhere, and
    ``sum_components len(used)`` is exactly the size of the offset union.
    """

    n: int
    indptr: np.ndarray      # int64[n + 1]
    nbr: np.ndarray         # int32[2E]
    #: int32[2E], globally unique.  **int32, not int64**, and it is the dominant term: at n = 50,
    #: tau = 2 a cached instance is ~18 MB of which the CSR is 9, and ``cls`` is half of that.  The
    #: class index cannot overflow by construction -- it counts distinct pairwise intersections
    #: within one instance, measured at 8,278 for |C| = 1,355 at tau = 0 (A43) and bounded by
    #: ``sum_components n_hcand``.  Everything that compares against it (``commit``, ``used``) is
    #: int64 and numpy upcasts, so no call site changes.
    cls: np.ndarray
    comp_of: np.ndarray     # int64[n], KERNEL for degree-0 vertices
    n_classes: int

    @classmethod
    def of(cls, gs) -> OverlapCSR:
        n = int(gs.n_clusters)
        comp_of = np.full(n, KERNEL, dtype=np.int64)
        rows, cols, klass = [], [], []
        offset = 0
        for ci, comp in enumerate(gs.components):
            v = np.asarray(comp.vertices, dtype=np.int64)
            comp_of[v] = ci
            gi, gj = v[np.asarray(comp.ii, np.int64)], v[np.asarray(comp.jj, np.int64)]
            gc = np.asarray(comp.edge_class, dtype=np.int64) + offset
            offset += int(comp.n_hcand)
            rows.append(np.concatenate([gi, gj]))
            cols.append(np.concatenate([gj, gi]))
            klass.append(np.concatenate([gc, gc]))
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            k = np.concatenate(klass)
            order = np.lexsort((c, r))
            r, c, k = r[order], c[order], k[order]
            indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n))]).astype(np.int64)
        else:
            c = np.zeros(0, np.int64)
            k = np.zeros(0, np.int64)
            indptr = np.zeros(n + 1, np.int64)
        if offset > np.iinfo(np.int32).max:
            raise ValueError(f"{offset} intersection classes overflow int32")
        return cls(n=n, indptr=indptr, nbr=c.astype(np.int32), cls=k.astype(np.int32),
                   comp_of=comp_of, n_classes=offset)

    def neighbours(self, v: int):
        lo, hi = int(self.indptr[v]), int(self.indptr[v + 1])
        return self.nbr[lo:hi], self.cls[lo:hi]


class ClusterEnv:
    """One instance's MDP.  ``S_0`` is empty (plus the implicit singletons and ``X``).

    The state is delegated to :class:`opc.followup.search.SearchState` -- the same object every
    unlearned search in the follow-up used -- so a move accepted here is one that apparatus would
    have accepted too.  What this class owns is the *mask*, computed in one vectorised pass and
    checked against ``SearchState.legal()`` row for row in the tests.
    """

    def __init__(self, gs, budget: int, *, graph: OverlapCSR | None = None,
                 include_kernel: bool = True):
        self.gs = gs
        self.budget = int(budget)
        self.n = int(gs.n_clusters)
        self.graph = graph or OverlapCSR.of(gs)
        self.include_kernel = bool(include_kernel)
        self.support = np.asarray(gs.support, dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ episode

    def reset(self, *, init: str = "empty", given=None) -> dict:
        """``empty`` | ``greedy`` | ``given`` -- ten lines, and what keeps warm starts here.

        **The state is cleared, not rebuilt.**  ``heuristics._State.__init__`` calls
        ``_csr(comp)``, and at tau = 0 one component carries 3.8 M edges, so constructing a fresh
        ``SearchState`` per episode costs more than the whole rest of the episode: measured at
        366 ms of a 423 ms rollout at |C| = 4,652, which the in-place clear takes to **57 ms**.
        Only ``in_s``, ``commit`` and ``used`` are mutable -- that list is
        ``search._STATE_FIELDS``, which ``_clone_state`` already relies on and asserts against, so
        clearing exactly those three is the same contract a clone honours.
        """
        self.state = self._cleared_state()
        self.t = 0
        self.last_action = -1
        if init == "greedy":
            for v in np.argsort(-self.support, kind="stable"):
                ok, klass = self.state.can_accept(int(v))
                if ok:
                    self.state.accept(int(v), klass)
        elif init == "given":
            for v in np.asarray(given, dtype=np.int64):
                ok, klass = self.state.can_accept(int(v))
                if not ok:
                    raise ValueError(f"given selection is infeasible at cluster {int(v)}")
                self.state.accept(int(v), klass)
        elif init != "empty":
            raise ValueError(f"unknown init policy {init!r}")
        return self.observe()

    def step(self, action: int):
        """``(obs, reward, done, info)``.  Per-step reward is **0** -- see RL_PLAN §2.

        Not shaped with per-step MPL, deliberately: F3 measured that argmax to be actively
        misleading (``greedy-mpl`` worst on both MPL regret and cluster F1, 0 of 68 recoveries,
        A58).  Shaping with a signal known to point the wrong way is worse than terminal-only.
        """
        a = int(action)
        ok, klass = self.state.can_accept(a)
        if not ok:
            raise ValueError(f"illegal action {a}; the mask and the policy have diverged")
        self.state.accept(a, klass)
        self.t += 1
        self.last_action = a
        mask, deg_sel = self.legal_mask(with_features=True)
        return (self.observe(mask=mask, deg_sel=deg_sel), 0.0, not bool(mask.any()),
                {"class": klass})

    def observe(self, mask=None, deg_sel=None) -> dict:
        if mask is None:
            mask, deg_sel = self.legal_mask(with_features=True)
        return {
            "in_s": self.state.in_s,
            "legal": mask,
            "deg_sel": deg_sel,
            "t": self.t,
            "last_action": self.last_action,
            "n_hybrids": self.state.n_hybrids,
            "k_remaining": self.budget - self.state.n_hybrids,
        }

    def _cleared_state(self):
        st = getattr(self, "_state_template", None)
        if st is None:
            st = search.SearchState(self.gs, self.budget, include_kernel=False)
            self._state_template = st
        for local in st.states:
            unexpected = set(vars(local)) - set(search._STATE_FIELDS)  # noqa: SLF001
            if unexpected:
                raise AssertionError(
                    f"heuristics._State grew {sorted(unexpected)}; clearing in place would leave "
                    "them carrying the previous episode")
            local.in_s[:] = False
            local.commit[:] = -1
            local.used.clear()
            local._mark[:] = False  # noqa: SLF001
        st.in_s[:] = False
        if self.include_kernel and st.kernel.size:
            st.in_s[st.kernel] = True
        return st

    # ------------------------------------------------------------------ the mask

    def legal_mask(self, *, with_features: bool = False):
        """``bool[|C|]`` -- exactly ``{v : SearchState.can_accept(v)[0]}``, in one pass.

        The four conditions, in ``_State.can_accept``'s own order, each as an array operation over
        the ``|C| x |S|`` incidence rather than a loop over ``|C|``.

        ``with_features`` also returns ``deg_sel``: each candidate's number of **accepted**
        neighbours.  That is one of RL_PLAN §3's dynamic decoder features ("overlap-degree within
        ``S_t``"), and this function computes it anyway on its way to the (N3O) test -- it is the
        ``deg_sel >= 2`` prefilter.  Returning it costs nothing and stops the decoder recomputing
        a ``|C| x |S|`` incidence that already exists; recomputing it was measured on the tau = 0
        ground set at 366 ms of a 423 ms rollout in a different guise (``reset``), which is the
        same lesson.

        The single-return signature is unchanged, because G-R0's parity test compares this against
        ``SearchState.legal()`` and a test that had to be edited to accommodate a refactor is a
        weaker test than one that did not.
        """
        g = self.graph
        in_s = self.state.in_s
        sel = np.flatnonzero(in_s).astype(np.int64)
        ok = ~in_s.copy()
        zeros = np.zeros(self.n, dtype=np.int64)
        if sel.size == 0:
            # nothing accepted: every unselected cluster is isolated-legal, and has no accepted
            # neighbour by definition.
            return (ok, zeros) if with_features else ok

        # --- gather the edges incident to S once.  |S| ~ 50, so this is ~ sum_{s in S} deg(s). ---
        counts = (g.indptr[sel + 1] - g.indptr[sel]).astype(np.int64)
        if counts.sum() == 0:
            return (ok, zeros) if with_features else ok
        src = np.repeat(np.arange(sel.size, dtype=np.int64), counts)   # column in B
        take = np.concatenate([np.arange(g.indptr[s], g.indptr[s + 1]) for s in sel])
        dst = g.nbr[take].astype(np.int64)                             # the candidate v
        ecls = g.cls[take]

        # --- (L) at v: all selected incident edges carry one class ------------------------------
        # ``np.minimum.at`` would say this directly and is an order of magnitude slower than the
        # scatter-then-count below: unbuffered ufunc.at falls off numpy's fast path entirely.
        # A plain scatter is last-write-wins, which is enough -- any one of the classes will do as
        # the representative, because the test is whether they are *all* equal to it.
        klass = np.full(self.n, -1, dtype=np.int64)
        klass[dst] = ecls
        touched = np.zeros(self.n, dtype=bool)
        touched[dst] = True
        mismatch = np.bincount(dst, weights=(ecls != klass[dst]), minlength=self.n)
        ok &= ~(touched & (mismatch > 0))

        # --- (L) at an already-committed neighbour ----------------------------------------------
        # ``commit`` lives per component in ``_State``; lift the accepted vertices' commitments to
        # the global class numbering, then count each candidate's disagreeing committed neighbours.
        commit_of_sel = self._commit_global(sel)
        cs = commit_of_sel[src]
        has = cs >= 0
        if has.any():
            wrong = np.bincount(dst[has], weights=(cs[has] != klass[dst[has]]), minlength=self.n)
            ok &= wrong == 0

        # --- (N3O): two accepted neighbours of v that are adjacent to each other -----------------
        # "v has two accepted neighbours that are adjacent" is a set intersection, and the set is
        # indexed by the *accepted* vertices -- at most n - 1 + O(k) of them, about fifty at n = 50
        # whatever |C| is.  So each candidate's accepted-neighbour set is a packed bitmask over |S|
        # bits, exactly the representation ``opc.bitset`` uses for clusters, and the test is
        # ``B[v] & A_S[i] != 0`` for some accepted neighbour ``i`` of ``v``.
        #
        # Written as ``((B @ A_S) * B).sum(1)`` first, which is the same statement in two small
        # dense products -- and measured at **19.5 ms** against 0.3 ms for everything else in this
        # function, because a float32 GEMM hands the work to a threaded BLAS that is competing with
        # every other worker in the pool.  A44's lesson in miniature: the danger is not the flops,
        # it is who else wants the cores.  Bitmasks touch no BLAS at all.
        deg_sel = np.bincount(dst, minlength=self.n)
        cand = np.flatnonzero(ok & (deg_sel >= 2))
        if cand.size:
            m = sel.size
            nw = (m + 63) // 64
            pos = np.full(self.n, -1, dtype=np.int64)
            pos[sel] = np.arange(m)
            bm = np.zeros((self.n, nw), dtype=np.uint64)
            bit = (np.uint64(1) << (src & 63).astype(np.uint64))
            np.bitwise_or.at(bm, (dst, src >> 6), bit)
            am = np.zeros((m, nw), dtype=np.uint64)
            inner = in_s[dst]
            if inner.any():
                j = pos[dst[inner]]
                np.bitwise_or.at(am, (src[inner], j >> 6),
                                 np.uint64(1) << (j & 63).astype(np.uint64))
            sub = bm[cand]
            blocked = np.zeros(cand.size, dtype=bool)
            for i in range(m):
                if not am[i].any():
                    continue
                here = (sub[:, i >> 6] >> np.uint64(i & 63)) & np.uint64(1)
                if not here.any():
                    continue
                meets = np.zeros(cand.size, dtype=bool)
                for w in range(nw):
                    meets |= (sub[:, w] & am[i, w]) != 0
                blocked |= (here != 0) & meets
            ok[cand] &= ~blocked

        # --- the budget, applied globally on top of the local mask -------------------------------
        if self.state.n_hybrids >= self.budget:
            used = self._used_global()
            fresh = touched & ~np.isin(klass, used)
            ok &= ~fresh
        return (ok, deg_sel) if with_features else ok

    def _commit_global(self, sel: np.ndarray) -> np.ndarray:
        """``commit`` of each accepted vertex, in the global class numbering (-1 = uncommitted)."""
        out = np.full(sel.size, -1, dtype=np.int64)
        offsets = self._class_offsets()
        for i, v in enumerate(sel):
            ci = int(self.state.comp_of[v])
            if ci < 0:
                continue
            c = int(self.state.states[ci].commit[int(self.state.local_of[v])])
            if c >= 0:
                out[i] = c + offsets[ci]
        return out

    def _used_global(self) -> np.ndarray:
        offsets = self._class_offsets()
        out = [c + offsets[ci] for ci, st in enumerate(self.state.states) for c in st.used]
        return np.asarray(sorted(out), dtype=np.int64)

    def _class_offsets(self) -> np.ndarray:
        if not hasattr(self, "_offsets"):
            sizes = [int(c.n_hcand) for c in self.gs.components]
            self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        return self._offsets

    # ------------------------------------------------------------------ validation

    @property
    def selection(self) -> np.ndarray:
        return self.state.selection

    def verify(self) -> bool:
        """``heuristics.check`` on every component -- shares no state with the mask."""
        return bool(self.state.verify())


def rollout(env: ClusterEnv, pick, max_steps: int | None = None) -> np.ndarray:
    """Run one episode to termination, choosing with ``pick(mask) -> action``.

    **Uses the mask ``step`` already computed.**  ``step`` evaluates ``legal(S_{t+1})`` anyway to
    decide ``done``, so a caller that calls ``legal_mask()`` itself at the top of the loop pays
    for every mask twice -- measured at 2x the whole rollout cost, which at |C| = 4,652 is the
    difference between 130 ms and 260 ms per episode, before any reward is computed.

    ``max_steps`` stops after that many accepted clusters.  **The default of ``None`` must
    reproduce the uncapped episode byte-identically** -- there is exactly one rollout
    implementation here on purpose, because :func:`greedy_support_rollout` and
    :func:`opc.rl.nodecls.decode` differ only in the score they rank by, and a parity test between
    two *different* loops would be testing nothing.  ``tests/test_rl_env.py`` pins the default.

    A cap is a statement about the caller, not about the env: the episode is maximal at the
    truth's cardinality on 192 of 192 cells (A71), and measured on the landed ``segment`` arm the
    decode runs 24.3 steps at n = 25 against a median ``n_true`` of 24 -- so a cap at ``n_true``
    is inert on this grid and is kept as a flag rather than as a mechanism.
    """
    obs = env.reset()
    taken = 0
    while True:
        mask = obs["legal"]
        if not mask.any() or (max_steps is not None and taken >= max_steps):
            return env.selection
        obs, _r, done, _info = env.step(int(pick(mask)))
        taken += 1
        if done:
            return env.selection


def greedy_support_rollout(env: ClusterEnv) -> np.ndarray:
    """Argmax-support under the env's own mask.  **G-R0's parity subject.**

    Must reproduce ``search._greedy_support`` -- and therefore ``heuristics.greedy(key="w")`` --
    selection for selection.  Written as a rollout rather than as a loop over ``argsort`` so that
    what is compared is the *environment*, mask and step and all, and not a second copy of the
    heuristic: a parity test that bypassed ``step`` would be testing nothing.
    """
    order = np.argsort(-env.support, kind="stable")
    rank = np.empty(env.n, dtype=np.int64)
    rank[order] = np.arange(env.n)

    def pick(mask):
        cand = np.flatnonzero(mask)
        return cand[np.argmin(rank[cand])]

    return rollout(env, pick)
