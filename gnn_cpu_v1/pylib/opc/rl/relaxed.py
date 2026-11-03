"""The relaxed decoder: constrain the network a selection stands for, not its level-1 signature.

Why a second decoder
--------------------
:class:`opc.rl.env.ClusterEnv` -- the formal decoder -- masks by Hellmuth-Schaller-Stadler's
characterisation of **level-1** clustering systems, (L) and (N3O); counts reticulations as the
number of distinct pairwise intersections opened; and :func:`opc.followup.search.realise` closes
the accepted set under intersections before reading its Hasse diagram as a network.  All three are
level-1 theorems.  Measured on the 20 exactly-level-2 networks of ``l2reg_n200_v1`` (2026-09-11):

* (L) fails on the true cluster set of 20 of 20 networks and (N3O) on 13, in exactly the overlap
  components of the level-2 blobs (2-3 distinct intersections each, where a level-1 one has 1);
* closing the *true* cluster set breaks its network on 10 of 20 -- at level 2 the intersection of
  two overlapping clusters need not be a cluster -- and closing it only under intersections the
  gene trees support still breaks 6;
* with oracle ranking the formal mask refuses 71 of the 4,026 true clusters in ``C_tau``.

What this decoder constrains instead
------------------------------------
A regular network *is* the Hasse diagram of its clusters, so the constraints are stated on that
object.  The selection ``S`` stands for ``H(S + singletons + taxon set)``, with **no closure**, and
a candidate ``c`` is legal iff ``H(S + c)``

* has reticulation number ``|E| - |V| + 1 <= budget`` (:func:`opc.solve.closure.reticulation_number`);
* has in-degree at most 2 everywhere -- binary reticulations;
* has level at most ``max_level``: no biconnected component carries more reticulations;
* admits a donor assignment: each hybrid's two parents split into tree parent and donor with no
  node donating twice, which is all :class:`~opc.simulate.networks.Network` can hold (:func:`donors`).

Without the closure the network of the true clusters is rebuilt exactly on 20 of 20 l2reg truths.
``max_level = 1`` gives a level-1 decoder too, but not the formal one: no closure, and a budget
counted on the diagram rather than on intersection classes.

Order semantics: the best legal candidate, every step
-----------------------------------------------------
Each step accepts the highest-scoring legal candidate (ties to the lower index -- ``nodecls.decode``'s
rank rule); the decode ends when none is legal.  That is the formal decoder's rule too, but there
the constraints are hereditary, so a single pass in rank order *is* that rule.  Here they are not:
accepting a cluster can make a refused one legal, so a candidate refused when first reached has to
be reconsidered the moment that happens -- not on a later pass, by which time lower-ranked clusters
may have been accepted and blocked it again.  (A pass-based version of this decoder lost 1 of 205
true clusters on l2reg rep13 under oracle ranking for exactly that reason.)

Rechecking every refused candidate after every acceptance is exact and ~10^4 times too slow, so a
refusal is rechecked only when the acceptance could have lifted it:

* **budget** refusals depend on the reticulation number and on ``P`` (the candidate's minimal
  accepted supersets), ``Ch`` (its maximal accepted subsets) and the covers between them.  An
  accepted ``c*`` that overlaps or is disjoint from ``c`` leaves ``P`` and ``Ch`` alone and can only
  remove covers between them, which makes the refusal worse; one nested with ``c`` changes them
  only if it lands with no accepted cluster in between (it becomes a new member of ``P`` or
  ``Ch``).  So those are rechecked -- plus every budget refusal when an acceptance *lowers* the
  reticulation number;
* **in-degree** refusals depend on the same ``P`` and ``Ch`` and, besides, on the in-degree of each
  node of ``Ch`` -- which an acceptance incomparable with ``c`` *can* lower: inserted between a node
  ``q`` and two of ``q``'s parents, ``c*`` takes ``q`` from two parents to one.  So they are rechecked
  in the nested cases above, and whenever an acceptance changes the in-degree of a node inside the
  candidate (:meth:`RelaxedDecoder._indeg_lifted`);
* **level** and **donor** refusals are global and few (0-10 per decode, measured), so all of them
  are rechecked after every acceptance.

Rechecks run through a priority queue in rank order, so the acceptance they produce is the best
legal one.  ``audit`` (on under ``check_every_accept``) probes every refused candidate once the
decode ends and fails if any is legal -- the certificate above, tested rather than trusted.

The second half of the in-degree clause was missing until 2026-09-12.  No decode reached an illegal
state without it, but a legal candidate could stay refused, or the decode take a different path than
the rule above: on the true cluster sets of ``l2reg_n200_v1`` and ``l3reg_n200_v1``, under 14
rankings and two budgets, 28 of 2,800 decodes differed from an exhaustive-recheck oracle, and none
does now.  No landed row is known to have changed (``l3reg_eval/README.md``);
``l3reg_eval/check_relaxed.py`` G2b holds the fix.

Implementation
--------------
The diagram is kept incrementally.  Inserting ``c`` below ``P`` and above ``Ch`` adds ``|P| + |Ch|``
covers and removes every cover ``p -> q`` with ``p in P, q in Ch``, so
``dr = |P| + |Ch| - removed - 1`` needs no rebuild; ``P``, ``Ch`` and the removed covers are read off
an incrementally maintained ``|C| x |H|`` intersection matrix.  Level and donors are checked on the
modified diagram only for candidates that pass the three cheap tests.  :meth:`RelaxedDecoder.verify`
rebuilds the diagram from scratch and checks the incremental state against it, and
``check_every_accept`` does so after every acceptance.  ``l2reg_eval/check_relaxed.py``
cross-validates the whole decoder against a from-scratch reference that rescans from the top at
every step and measures level with the independent :func:`opc.simulate.networks.level`.
"""

from __future__ import annotations

import heapq

import numpy as np

from opc import bitset
from opc.simulate import networks as N
from opc.solve import closure

__all__ = ["DECODERS", "decoder_tag", "RelaxedDecoder", "realise", "network_from_hasse",
           "hasse_level", "donors"]

#: The decoders a driver can be asked for.  ``level1`` is the formal one and the default.
DECODERS = ("level1", "relaxed")
#: Refusal reasons and their codes in :meth:`RelaxedDecoder.decode`'s bookkeeping (0 = not refused).
_REASONS = ("indeg", "budget", "level", "donor")
_CODE = {r: i + 1 for i, r in enumerate(_REASONS)}


def decoder_tag(decoder: str, max_level: int | None = None) -> str:
    """``level1`` or ``relaxed_L<max_level>`` -- how a run, a cell and a CSV row name their decoder.

    Raises ``ValueError`` on an unknown decoder, on a relaxed decoder without ``max_level >= 1``,
    and on a ``max_level`` handed to the formal decoder, which has no such parameter.
    """
    if decoder == "level1":
        if max_level is not None:
            raise ValueError("max_level applies only to the relaxed decoder")
        return "level1"
    if decoder != "relaxed":
        raise ValueError(f"unknown decoder {decoder!r}; known: {', '.join(DECODERS)}")
    if max_level is None or int(max_level) < 1:
        raise ValueError("the relaxed decoder needs a max_level >= 1")
    return f"relaxed_L{int(max_level)}"


# ------------------------------------------------------------------------------ graph primitives

def hasse_level(parent, child, n_nodes: int) -> int:
    """Max over biconnected components of ``|E| - |V| + 1`` on the undirected cover graph.

    With every in-degree at most 2 a blob's cyclomatic number is the number of reticulations in
    it, so this is the level.  Iterative Hopcroft-Tarjan with an edge stack; bridges give 0.
    """
    ends = list(zip((int(x) for x in parent), (int(x) for x in child)))
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for eid, (a, b) in enumerate(ends):
        adj[a].append((b, eid))
        adj[b].append((a, eid))
    disc = [-1] * n_nodes
    low = [0] * n_nodes
    counter = best = 0
    edge_stack: list[int] = []
    for start in range(n_nodes):
        if disc[start] != -1 or not adj[start]:
            continue
        disc[start] = low[start] = counter
        counter += 1
        work = [[start, -1, 0]]                  # node, edge it was reached by, next adj index
        while work:
            top = work[-1]
            v, pe, i = top
            if i < len(adj[v]):
                top[2] = i + 1
                w, eid = adj[v][i]
                if eid == pe:
                    continue
                if disc[w] == -1:
                    edge_stack.append(eid)
                    disc[w] = low[w] = counter
                    counter += 1
                    work.append([w, eid, 0])
                elif disc[w] < disc[v]:
                    edge_stack.append(eid)
                    low[v] = min(low[v], disc[w])
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    n_edges, verts = 0, set()
                    while True:
                        e = edge_stack.pop()
                        n_edges += 1
                        verts.update(ends[e])
                        if e == pe:
                            break
                    best = max(best, n_edges - len(verts) + 1)
    return best


def _pseudoforest(pairs) -> bool:
    """True iff each component of the multigraph on parents (one edge per hybrid, joining its two
    parents) has no more edges than vertices -- exactly when :func:`donors` succeeds."""
    root: dict[int, int] = {}

    def find(x: int) -> int:
        root.setdefault(x, x)
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    pairs = [(int(a), int(b)) for a, b in pairs]
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            root[ra] = rb
    edges: dict[int, int] = {}
    verts: dict[int, int] = {}
    for a, _b in pairs:
        r = find(a)
        edges[r] = edges.get(r, 0) + 1
    for x in list(root):
        r = find(x)
        verts[r] = verts.get(r, 0) + 1
    return all(e <= verts[r] for r, e in edges.items())


def donors(parent_pairs: dict[int, tuple[int, int]]) -> dict[int, int] | None:
    """``hybrid -> donor``, one of its two parents, with no node donating twice; None if impossible.

    Each hybrid is an edge between its two parents and choosing its donor orients it: "donates at
    most once" is in-degree <= 1, possible iff every component has no more edges than vertices.
    A parent left with one unassigned hybrid must donate to it or to nobody, so those are peeled
    first; what remains is disjoint cycles, each oriented around.  Deterministic.
    ``closure.network_from_hasse`` instead takes the first free parent in cluster order, which can
    fail where an assignment exists.
    """
    inc: dict[int, set[int]] = {}
    for h, (a, b) in parent_pairs.items():
        inc.setdefault(int(a), set()).add(int(h))
        inc.setdefault(int(b), set()).add(int(h))
    left = {int(h): (int(a), int(b)) for h, (a, b) in parent_pairs.items()}
    out: dict[int, int] = {}

    def assign(h: int, d: int) -> None:
        out[h] = d
        a, b = left.pop(h)
        inc[a].discard(h)
        inc[b].discard(h)

    while True:
        leaves = [p for p in sorted(inc) if len(inc[p]) == 1]
        if not leaves:
            break
        (h,) = inc[leaves[0]]
        assign(h, leaves[0])
    if any(len(hs) > 2 for hs in inc.values()):
        return None
    while left:
        h = min(left)
        d = left[h][1]
        while True:
            assign(h, d)
            nxt = inc[d]
            if not nxt:
                break
            (h,) = nxt
            a, b = left[h]
            d = b if a == d else a
    return out


def network_from_hasse(words: np.ndarray, n_taxa: int, labels=None) -> N.Network:
    """``H(F)`` as a :class:`~opc.simulate.networks.Network`, for any binary regular network.

    :func:`opc.solve.closure.network_from_hasse` with one change: donors are assigned by
    :func:`donors` rather than first-free, so a diagram with a valid assignment is never refused.
    ``F`` must carry the trivial clusters; unlike there, it need not be closed.
    """
    parent, child = closure.hasse_covers_words(words, n_taxa)
    k = words.shape[0]
    sizes = bitset.popcounts(words)
    covers: dict[int, list[int]] = {v: [] for v in range(k)}
    for p, c in zip(parent, child):
        covers[int(c)].append(int(p))
    roots = [v for v in range(k) if not covers[v]]
    if len(roots) != 1 or int(sizes[roots[0]]) != n_taxa:
        raise ValueError(f"H(F) has {len(roots)} maximal elements; expected exactly the taxon set")
    heavy = [v for v in range(k) if len(covers[v]) > 2]
    if heavy:
        raise ValueError(f"{len(heavy)} clusters have more than two covering parents")
    donor = donors({v: tuple(ps) for v, ps in covers.items() if len(ps) == 2})
    if donor is None:
        raise ValueError("no donor assignment: some node would have to donate to two hybrids")

    labels = list(labels) if labels is not None else [f"t{i + 1}" for i in range(n_taxa)]
    net = N.Network(
        n_taxa=n_taxa, labels=labels,
        time=[float(n_taxa - int(sizes[v])) for v in range(k)],
        tree_parent=[-1] * k, tree_children=[[] for _ in range(k)],
        leaf_taxon={}, horiz_child={}, donor_of={}, retics=[], root=int(roots[0]),
    )
    for v in range(k):
        if int(sizes[v]) == 1:
            net.leaf_taxon[v] = bitset.to_indices(words[v], n_taxa)[0]
    for v in sorted(range(k), key=lambda v: bitset.to_indices(words[v], n_taxa)):
        ps = covers[v]
        if not ps:
            continue
        if len(ps) == 1:
            net.tree_parent[v] = ps[0]
            net.tree_children[ps[0]].append(v)
            continue
        d = donor[v]
        t = ps[0] if ps[0] != d else ps[1]
        net.tree_parent[v] = t
        net.tree_children[t].append(v)
        net.horiz_child[d] = v
        net.donor_of[v] = d
        net.retics.append(N.Reticulation(v, d, len(net.retics), net.time[v]))
    return net


def realise(words: np.ndarray, selection, n_taxa: int, labels=None):
    """``(net, F)`` for a relaxed selection, or None -- :func:`opc.followup.search.realise` without
    the closure.  ``F`` is the selection plus the trivial clusters, the object ``net`` is the Hasse
    diagram of."""
    sel = np.asarray(selection, dtype=np.int64)
    chosen = (np.ascontiguousarray(np.asarray(words)[sel]) if sel.size
              else np.zeros((0, bitset.n_words(n_taxa)), np.uint64))
    full = closure.with_trivial_clusters(chosen, n_taxa)
    try:
        return network_from_hasse(full, n_taxa, labels), full
    except ValueError:
        return None


# ---------------------------------------------------------------------------------- the decoder

class RelaxedDecoder:
    """Accept clusters under the Hasse-diagram constraints of the module docstring.

    One instance per candidate set; :meth:`decode` may be called once per score vector and resets
    the diagram each time.  ``words`` are the candidates (``GroundSet.words``), which must be
    distinct and non-trivial, as ``C_tau`` is.
    """

    def __init__(self, words: np.ndarray, n_taxa: int, budget: int, max_level: int, *,
                 check_every_accept: bool = False):
        decoder_tag("relaxed", max_level)                 # validates max_level
        if int(budget) < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        self.words = np.ascontiguousarray(words)
        self.n_taxa = int(n_taxa)
        self.budget = int(budget)
        self.max_level = int(max_level)
        self.check_every_accept = bool(check_every_accept)
        self.n = int(self.words.shape[0])
        self.D = (bitset.to_dense(self.words, self.n_taxa).astype(np.float32) if self.n
                  else np.zeros((0, self.n_taxa), np.float32))
        self.csize = self.D.sum(axis=1)
        self.stats: dict = {}
        self.verify_reason = ""
        self._selection = np.zeros(0, dtype=np.int64)
        self._reset()

    # ------------------------------------------------------------------ state

    def _alloc(self, cap: int) -> None:
        self.cap = cap
        self.F = np.zeros((cap, self.n_taxa), np.float32)      # the diagram's clusters, dense
        self.sizes = np.zeros(cap, np.float32)
        self.inside = np.zeros((cap, cap), bool)               # [s, t]: t strictly inside s
        self.cov = np.zeros((cap, cap), bool)                  # [s, t]: s covers t
        self.indeg = np.zeros(cap, np.int64)
        self.inter = np.zeros((self.n, cap), np.float32)       # |candidate & node|

    def _grow(self) -> None:
        m, old = self.m, (self.F, self.sizes, self.inside, self.cov, self.indeg, self.inter)
        self._alloc(2 * self.cap)
        self.F[:m], self.sizes[:m], self.indeg[:m] = old[0][:m], old[1][:m], old[4][:m]
        self.inside[:m, :m], self.cov[:m, :m] = old[2][:m, :m], old[3][:m, :m]
        self.inter[:, :m] = old[5][:, :m]

    def _reset(self) -> None:
        n = self.n_taxa
        self._alloc(2 * n + 64)
        self.F[:n] = np.eye(n, dtype=np.float32)
        self.F[n] = 1.0
        self.sizes[:n], self.sizes[n] = 1.0, float(n)
        self.inside[n, :n] = self.cov[n, :n] = True
        self.indeg[:n] = 1
        self.inter[:, :n] = self.D
        self.inter[:, n] = self.csize
        self.m = n + 1
        self.r = 0
        self.node_cand = [-1] * (n + 1)
        self._indeg_changed = np.zeros(0, dtype=np.int64)     # nodes the last acceptance re-parented

    # ------------------------------------------------------------------ one candidate

    def _probe(self, c: int, stats: dict | None):
        """Whether ``c`` is legal now, without changing anything.

        Returns ``None`` if ``c`` is already in the diagram, ``(False, reason)`` if it is refused,
        and ``(True, move)`` if it is legal, ``move`` being what :meth:`_accept` needs.
        """
        m = self.m
        sizes, inter, sc = self.sizes[:m], self.inter[c, :m], self.csize[c]
        eq = inter == sc
        if (eq & (sizes == sc)).any():
            return None
        supi = np.flatnonzero(eq & (sizes > sc))
        subi = np.flatnonzero((inter == sizes) & (sizes < sc))
        P = supi[~self.inside[np.ix_(supi, supi)].any(axis=1)]
        if P.size > 2:
            return False, "indeg"
        Ch = subi[~self.inside[np.ix_(subi, subi)].any(axis=0)]
        pc = self.cov[np.ix_(P, Ch)]
        new_in = self.indeg[Ch] - pc.sum(axis=0) + 1
        if new_in.size and int(new_in.max()) > 2:
            return False, "indeg"
        dr = int(P.size + Ch.size - int(pc.sum()) - 1)
        if self.r + dr > self.budget:
            return False, "budget"
        if stats is not None:
            stats["global_checks"] += 1
        ok, why = self._global_ok(P, Ch, pc)
        if not ok:
            return False, why
        return True, (c, supi, subi, P, Ch, pc, new_in, dr)

    def _global_ok(self, P, Ch, pc) -> tuple[bool, str]:
        """Donors and level, on the diagram as it would be with the candidate inserted."""
        m = self.m
        ii, jj = np.nonzero(self.cov[:m, :m])
        if pc.any():
            rp, rc = np.nonzero(pc)
            keep = ~np.isin(ii * m + jj, P[rp] * m + Ch[rc])
            ii, jj = ii[keep], jj[keep]
        par = np.concatenate([ii, P, np.full(Ch.size, m, dtype=np.int64)])
        chi = np.concatenate([jj, np.full(P.size, m, dtype=np.int64), Ch])
        hyb = np.flatnonzero(np.bincount(chi, minlength=m + 1) == 2)
        if hyb.size and not _pseudoforest([tuple(par[chi == h]) for h in hyb]):
            return False, "donor"
        if hasse_level(par, chi, m + 1) > self.max_level:
            return False, "level"
        return True, ""

    def _accept(self, move) -> None:
        c, supi, subi, P, Ch, pc, new_in, dr = move
        if self.m == self.cap:
            self._grow()
        j = self.m
        x = self.D[c]
        self.F[j], self.sizes[j] = x, self.csize[c]
        self.inside[j, subi] = True
        self.inside[supi, j] = True
        if pc.any():
            rp, rc = np.nonzero(pc)
            self.cov[P[rp], Ch[rc]] = False
        self.cov[P, j] = True
        self.cov[j, Ch] = True
        self._indeg_changed = Ch[self.indeg[Ch] != new_in]
        self.indeg[Ch] = new_in
        self.indeg[j] = P.size
        self.r += dr
        self.inter[:, j] = self.D @ x
        self.node_cand.append(int(c))
        self.m = j + 1
        if self.check_every_accept:
            ok, why = self._matches_rebuild()
            if not ok:
                raise AssertionError(f"incremental diagram diverged after accepting {c}: {why}")

    def _lifted(self, move, reason: np.ndarray) -> np.ndarray:
        """The refused candidates the acceptance ``move`` could have made legal (module docstring)."""
        _c, supi, subi, _P, _Ch, _pc, _new_in, dr = move
        j = self.m - 1
        col, sj, cs = self.inter[:, j], self.sizes[j], self.csize
        out = [np.flatnonzero(reason >= _CODE["level"])]         # level and donor: always
        if dr < 0:
            out.append(np.flatnonzero(reason == _CODE["budget"]))
        local = (reason == _CODE["budget"]) | (reason == _CODE["indeg"])
        below = np.flatnonzero(local & (col == cs) & (cs < sj))  # c strictly inside c*
        if below.size and subi.size:                             # ... unless some t: c < t < c*
            between = ((self.inter[np.ix_(below, subi)] == cs[below, None])
                       & (self.sizes[subi][None, :] > cs[below, None])).any(axis=1)
            below = below[~between]
        above = np.flatnonzero(local & (col == sj) & (cs > sj))  # c* strictly inside c
        if above.size and supi.size:                             # ... unless some t: c* < t < c
            between = ((self.inter[np.ix_(above, supi)] == self.sizes[supi][None, :])
                       & (cs[above, None] > self.sizes[supi][None, :])).any(axis=1)
            above = above[~between]
        out += [below, above, self._indeg_lifted(reason)]
        return np.unique(np.concatenate(out))

    def _indeg_lifted(self, reason: np.ndarray) -> np.ndarray:
        """The in-degree refusals an acceptance *incomparable* with them can lift.

        A candidate's in-degree test reads ``indeg`` on its ``Ch``, and ``c*`` can lower that without
        containing or being contained in the candidate: inserted between a node ``q`` and two of
        ``q``'s parents, it takes ``q`` from two parents to one.  :meth:`_accept` records the nodes
        whose in-degree it changed; every in-degree refusal of a candidate containing one of them
        is rechecked.  Missing until 2026-09-12 -- ``l3reg_eval/check_relaxed.py`` G2b."""
        t = self._indeg_changed
        if not t.size:
            return np.zeros(0, dtype=np.int64)
        hit = (reason == _CODE["indeg"]) & (self.inter[:, t] == self.sizes[t]).any(axis=1)
        return np.flatnonzero(hit)

    # ------------------------------------------------------------------ decode

    def decode(self, scores) -> np.ndarray:
        """Accept the best legal candidate by ``scores``, repeatedly, until none is legal."""
        s = np.asarray(scores, dtype=np.float64)
        if s.shape != (self.n,):
            raise ValueError(f"scores must be ({self.n},), got {s.shape}")
        order = np.argsort(-s, kind="stable")
        rank = np.empty(self.n, dtype=np.int64)
        rank[order] = np.arange(self.n)
        self._reset()
        reason = np.zeros(self.n, dtype=np.int8)       # why a visited candidate stands refused
        queued = np.zeros(self.n, dtype=bool)
        heap: list[tuple[int, int]] = []
        stats = {"global_checks": 0, "rechecks": 0, "recheck_accepts": 0, "duplicates": 0,
                 **{f"rejected_{r}": 0 for r in _REASONS}}
        accepted: list[int] = []

        def visit(c: int) -> bool:
            got = self._probe(c, stats)
            if got is None:
                reason[c] = 0
                stats["duplicates"] += 1
                return False
            ok, what = got
            if not ok:
                reason[c] = _CODE[what]
                stats[f"rejected_{what}"] += 1
                return False
            self._accept(what)
            reason[c] = 0
            accepted.append(c)
            for x in self._lifted(what, reason).tolist():
                if not queued[x]:
                    heapq.heappush(heap, (int(rank[x]), x))
                    queued[x] = True
            return True

        for c in order.tolist():
            # Every candidate ranked above c has been visited: accepted, or refused and queued the
            # moment an acceptance could have lifted the refusal.  So draining the queue in rank
            # order before moving on accepts, at each step, the best legal candidate there is.
            visit(c)
            while heap:
                _, x = heapq.heappop(heap)
                queued[x] = False
                if reason[x]:
                    stats["rechecks"] += 1
                    stats["recheck_accepts"] += visit(x)
        if self.check_every_accept:
            self._audit(reason)
        stats.update(accepted=len(accepted), reticulations=self.r, level=self.level())
        self.stats = stats
        self._selection = np.sort(np.asarray(accepted, dtype=np.int64))
        return self._selection

    def _audit(self, reason: np.ndarray) -> None:
        """No refused candidate may be legal at the end: the recheck certificate, tested."""
        legal = [int(c) for c in np.flatnonzero(reason)
                 if (got := self._probe(int(c), None)) is not None and got[0]]
        if legal:
            raise AssertionError(f"{len(legal)} refused candidates are legal at the end, e.g. "
                                 f"{legal[:5]}: a refusal was lifted and never rechecked")

    @property
    def selection(self) -> np.ndarray:
        return self._selection

    def level(self) -> int:
        ii, jj = np.nonzero(self.cov[:self.m, :self.m])
        return hasse_level(ii, jj, self.m)

    # ------------------------------------------------------------------ validation

    def _rebuild(self):
        m = self.m
        F = self.F[:m]
        sizes = F.sum(axis=1)
        inter = F @ F.T
        inside = (inter == sizes[None, :]) & (sizes[:, None] > sizes[None, :])
        between = (inside.astype(np.float32) @ inside.astype(np.float32)) > 0
        cov = inside & ~between
        return inside, cov, cov.sum(axis=0), int(cov.sum()) - m + 1

    def _matches_rebuild(self) -> tuple[bool, str]:
        m = self.m
        inside, cov, indeg, r = self._rebuild()
        if not np.array_equal(inside, self.inside[:m, :m]):
            return False, "containment differs"
        if not np.array_equal(cov, self.cov[:m, :m]):
            return False, "covers differ"
        if not np.array_equal(indeg, self.indeg[:m]):
            return False, "in-degrees differ"
        if r != self.r:
            return False, f"reticulations {self.r} incremental, {r} rebuilt"
        return True, ""

    def verify(self) -> bool:
        """Rebuild the diagram from scratch; check the incremental state and every constraint."""
        ok, why = self._matches_rebuild()
        if ok:
            m = self.m
            ii, jj = np.nonzero(self.cov[:m, :m])
            hyb = np.flatnonzero(self.indeg[:m] == 2)
            if self.r > self.budget:
                ok, why = False, f"{self.r} reticulations > budget {self.budget}"
            elif int(self.indeg[:m].max()) > 2:
                ok, why = False, "in-degree above 2"
            elif hasse_level(ii, jj, m) > self.max_level:
                ok, why = False, f"level above {self.max_level}"
            elif hyb.size and not _pseudoforest([tuple(ii[jj == h]) for h in hyb]):
                ok, why = False, "no donor assignment"
        self.verify_reason = why
        return ok
