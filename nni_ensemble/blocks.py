"""The emitted network's uncertain region, cut into independent blocks -- pure Python on int masks.

Everything here is a function of the emitted selection ``E`` (Python-int cluster masks, bit ``t`` =
taxon ``t``), the candidate set ``C_tau`` and the logits.  No torch, no gene trees, no truth except
where a function says so.

The diagram
-----------
``H(E + singletons + X)``, as ``opc.rl.relaxed`` reads a relaxed selection: no closure.  A node's
*parents* are its minimal strict supersets among ``E + {X}`` (``cluster_anatomy/geometry._hasse``'s
definition, singletons included -- a leaf with two parents is a hybrid too).

The questionable set ``Q``
--------------------------
An emitted cluster with ``z < thr`` that crosses no emitted cluster, has exactly one parent, and is
not a *blob top* (its children do not cross each other).  The rest of the low-z clusters sit in or
on a blob and are left alone; :func:`lowz_class` says why.

Blocks and their polytomy
-------------------------
Two members of ``Q`` are coupled when one is the other's parent or they share their parent -- their
NNI edges share a node.  A block is a connected component.  Collapsing it leaves one node ``P_b``,
the first node outside ``b`` on any member's parent chain (unique; asserted), with *atoms*: the
maximal clusters of ``(E \\ b) + singletons`` inside ``P_b``, crossing ones merged into a super-atom.

* Every superset of a member contains ``P_b`` (each member has one parent), so anything crossing
  ``P_b`` is disjoint from every member, and every maximal element of ``(E \\ b) & P_b`` is a child
  of ``P_b`` or of a member.
* A block with a **hybrid atom** (an atom with two parents) can change a hybrid's tree parent or
  its reticulation number when re-resolved.  It is excluded and counted.  For every other block a
  re-resolution with clusters that are unions of atoms and cross nothing in ``E \\ b`` keeps the
  reticulation number and the level -- the lemma behind gates N0 and N1.

Configurations
--------------
``A_b``: the ``C_tau`` candidates that are unions of 2..m-1 atoms and cross no cluster of ``E \\ b``.
``R_b``: the maximal laminar families of ``A_b`` -- the maximal cliques of its compatibility graph,
enumerated by Bron-Kerbosch-Tomita on Python-int bitsets (:func:`bk_count`, :func:`bk_best`).  For a
lone cluster over three atoms this is exactly the 1-NNI ``{C, alt1, alt2} & C_tau``.

A block with more than ``R_CAP`` configurations is listed only within radius ``rho`` of the emitted
configuration ``M`` (:func:`radius_configs`): the maximal families with at most ``rho`` non-emitted
clusters.  Best-first cannot list those blocks -- its bound stays tens of nats above every leaf.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict

import numpy as np

LOWZ_CLASSES = ("Q", "crosses", "multi_parent", "blob_top")
CATEGORIES = ("A", "B", "C", "D")


def popcount(m: int) -> int:
    return m.bit_count()


def crosses(a: int, b: int) -> bool:
    """Properly overlapping: they meet, and neither contains the other."""
    i = a & b
    return bool(i) and i != a and i != b


def bits(x: int):
    """The set bits of ``x``, ascending."""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


# ------------------------------------------------------------------------------------ the diagram

class Diagram:
    """``H(E + singletons + X)``: parents of every cluster and leaf, children of every node, and
    which emitted clusters cross another."""

    def __init__(self, emitted, n: int):
        self.n = n
        self.full = (1 << n) - 1
        self.E = list(dict.fromkeys(int(e) for e in emitted))
        self.Eset = set(self.E)
        if self.full in self.Eset or any(popcount(e) < 2 for e in self.E):
            raise ValueError("the emitted selection must hold non-trivial clusters only")
        self.leaves = [1 << t for t in range(n)]
        uppers = sorted(self.E + [self.full], key=popcount)
        self.parents: dict[int, list[int]] = {}
        for m in self.E + self.leaves:
            sup = [d for d in uppers if d != m and d & m == m]
            self.parents[m] = [d for i, d in enumerate(sup)
                               if not any(e & d == e and e != d for e in sup[:i])]
        self.children: dict[int, list[int]] = defaultdict(list)
        for m, ps in self.parents.items():
            for d in ps:
                self.children[d].append(m)
        self.crossing = {e: any(crosses(e, f) for f in self.E if f != e) for e in self.E}

    def blob_top(self, m: int) -> bool:
        kids = self.children.get(m, [])
        return any(crosses(a, b) for a, b in itertools.combinations(kids, 2))

    def siblings(self, m: int) -> set[int]:
        return {s for d in self.parents[m] for s in self.children[d] if s != m}

    def nni(self, c: int) -> set[int]:
        """``cluster_anatomy/geometry._nni_map``'s formula on this diagram: ``S | (C \\ B)`` for a
        sibling ``S`` and a child ``B`` of ``c``.  The independent side of gate N2(a)."""
        out = set()
        for b in self.children.get(c, []):
            rest = c & ~b
            for s in self.siblings(c):
                f = s | rest
                if f != self.full and popcount(f) >= 2 and f != c:
                    out.add(f)
        return out


def lowz_class(D: Diagram, m: int) -> str:
    """Why a low-z emitted cluster is, or is not, questioned (first reason that applies)."""
    if D.crossing[m]:
        return "crosses"
    if len(D.parents[m]) != 1:
        return "multi_parent"
    if D.blob_top(m):
        return "blob_top"
    return "Q"


# -------------------------------------------------------------------------------------- blocks

def make_blocks(D: Diagram, Q: list[int]) -> list[list[int]]:
    """Connected components of ``Q`` under parent/child and shared-parent adjacency, each sorted
    by (size descending, mask); components ordered by their first member."""
    Qs = set(Q)
    up = {q: q for q in Q}

    def find(x):
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            up[ra] = rb

    by_parent = defaultdict(list)
    for q in Q:
        p = D.parents[q][0]
        if p in Qs:
            join(q, p)
        by_parent[p].append(q)
    for qs in by_parent.values():
        for q in qs[1:]:
            join(q, qs[0])
    comp = defaultdict(list)
    for q in Q:
        comp[find(q)].append(q)
    out = [sorted(c, key=lambda x: (-popcount(x), x)) for c in comp.values()]
    return sorted(out, key=lambda c: (-popcount(c[0]), c[0]))


def polytomy(D: Diagram, block: list[int]) -> dict:
    """``P_b``, the atoms, and what the lemma needs to know about them."""
    bs = set(block)
    tops = {D.parents[q][0] for q in block if D.parents[q][0] not in bs}
    if len(tops) != 1:
        raise AssertionError(f"block of {len(block)} has {len(tops)} polytomy nodes")
    P = tops.pop()
    pre = []
    for x in [P, *block]:
        for c in D.children.get(x, []):
            if c not in bs and c not in pre:
                pre.append(c)
    # merge crossing atoms (a blob below), transitively
    up = list(range(len(pre)))

    def find(i):
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i

    for i, j in itertools.combinations(range(len(pre)), 2):
        if pre[i] & pre[j]:
            up[find(i)] = find(j)
    groups = defaultdict(int)
    for i, x in enumerate(pre):
        groups[find(i)] |= x
    atoms = sorted(groups.values(), key=lambda a: (-popcount(a), a))
    hybrid = [x for x in pre if len(D.parents[x]) >= 2]
    union = 0
    for a in atoms:
        union |= a
    if union != P:
        raise AssertionError("the atoms do not cover the polytomy node")
    return {"P": P, "atoms": atoms, "pre": pre, "n_merged": len(pre) - len(atoms),
            "hybrid_atom": bool(hybrid), "p_is_root": P == D.full,
            "p_crosses": P != D.full and D.crossing.get(P, False)}


def atom_bits(Y: int, atoms: list[int]) -> int | None:
    """``Y`` as a bitmask over atom indices, or None when it splits an atom."""
    out = 0
    for i, a in enumerate(atoms):
        x = Y & a
        if x == a:
            out |= 1 << i
        elif x:
            return None
    return out


def allowed(P: int, atoms: list[int], others: list[int], cand_dense: np.ndarray,
            cand_size: np.ndarray, n: int, others_dense: np.ndarray) -> np.ndarray:
    """Rows of ``C_tau`` in ``A_b``: strictly inside ``P``, a union of 2..m-1 atoms, crossing no
    cluster of ``others`` (= ``E \\ b``).  Vectorised over the dense candidate matrix."""
    m = len(atoms)
    pin = np.array([(P >> t) & 1 for t in range(n)], dtype=bool)
    inside = ~cand_dense[:, ~pin].any(axis=1) & (cand_size < pin.sum())
    rows = np.nonzero(inside)[0]
    if rows.size == 0 or m < 3:
        return rows[:0]
    G = np.zeros((n, m), dtype=np.float32)
    asize = np.zeros(m, dtype=np.float32)
    for i, a in enumerate(atoms):
        for t in bits(a):
            G[t, i] = 1.0
        asize[i] = popcount(a)
    cnt = cand_dense[rows].astype(np.float32) @ G
    whole = cnt == asize[None, :]
    ok = ((cnt == 0) | whole).all(axis=1)
    k = whole.sum(axis=1)
    ok &= (k >= 2) & (k <= m - 1)
    rows = rows[ok]
    if rows.size and others_dense.shape[0]:
        ic = cand_dense[rows].astype(np.float32) @ others_dense.T.astype(np.float32)
        so = others_dense.sum(axis=1)[None, :]
        sy = cand_size[rows][:, None]
        bad = ((ic > 0) & (ic < sy) & (ic < so)).any(axis=1)
        rows = rows[~bad]
    return rows


def compat_graph(abits: list[int]) -> list[int]:
    """``adj[i]``: bitmask of options compatible with ``i`` (nested or disjoint, not ``i``)."""
    a = len(abits)
    adj = [0] * a
    for i in range(a):
        for j in range(i + 1, a):
            if not crosses(abits[i], abits[j]):
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def is_laminar(xs) -> bool:
    return not any(crosses(a, b) for a, b in itertools.combinations(xs, 2))


def is_maximal(family_bits: set[int], abits: list[int]) -> bool:
    """No member of ``abits`` outside the family is compatible with all of it."""
    return not any(y not in family_bits and not any(crosses(y, f) for f in family_bits)
                   for y in abits)


# ------------------------------------------------------------------------- Bron-Kerbosch-Tomita

def _pivot(P: int, X: int, adj: list[int]) -> int:
    best, bu = -1, -1
    for u in bits(P | X):
        c = (P & adj[u]).bit_count()
        if c > best:
            best, bu = c, u
    return bu


def bk_count(adj: list[int], cap: int) -> int:
    """The number of maximal cliques, or ``cap + 1`` once it passes ``cap``."""
    count = 0

    def rec(P: int, X: int) -> bool:
        nonlocal count
        if not P:
            if not X:
                count += 1
            return count > cap
        u = _pivot(P, X, adj)
        for v in list(bits(P & ~adj[u])):
            if rec(P & adj[v], X & adj[v]):
                return True
            P &= ~(1 << v)
            X |= 1 << v
        return False

    rec((1 << len(adj)) - 1, 0)
    return count


def bk_all(adj: list[int]) -> list[int]:
    """Every maximal clique, as a bitmask over options.  Unweighted, exhaustive."""
    out = []

    def rec(R: int, P: int, X: int) -> None:
        if not P:
            if not X:
                out.append(R)
            return
        u = _pivot(P, X, adj)
        for v in list(bits(P & ~adj[u])):
            rec(R | (1 << v), P & adj[v], X & adj[v])
            P &= ~(1 << v)
            X |= 1 << v

    rec(0, (1 << len(adj)) - 1, 0)
    return out


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def bk_best(adj: list[int], w: list[float], limit: int, mass_tail: float = 0.0,
            max_nodes: int = 0) -> dict:
    """Maximal cliques in descending total weight, best-first over the BK recursion tree.

    A search node ``(R, P, X)`` stands for the maximal cliques ``R + S``, ``S`` inside ``P``; its bound
    ``W(R) + sum_{j in P} max(0, w_j)`` never increases down the tree, so leaves pop in exact
    descending order.  A node is dead when some ``x`` in ``X`` is compatible with all of ``P`` (no
    extension can be maximal).  Stops after ``limit`` leaves, or once the frontier's mass bound is
    below ``mass_tail`` times what has been found.

    The frontier is not bounded by ``limit``: on a block with 801 options and 52 atoms it held 10^6
    nodes, and no leaf, after 67k expansions.  Use it on blocks whose tree is known to be small
    (``r_count <= R_CAP``); ``max_nodes`` (0 = none) raises rather than let it grow.

    Returns ``cliques`` (bitmasks), ``logw`` (their weights), ``exact`` (the tree was exhausted) and
    ``log_frontier``: ``log sum_frontier exp(W(R)) prod_{j in P} (1 + e^{w_j})``, an upper bound on
    the unlisted mass (``-inf`` when exact).
    """
    a = len(adj)
    sp = [_softplus(x) for x in w]
    pos = [max(0.0, x) for x in w]
    tick = itertools.count()
    full = (1 << a) - 1
    heap = [(-sum(pos), next(tick), 0, full, 0, 0.0, sum(sp))]
    cliques, logw = [], []
    pops = 0

    def frontier() -> float:
        if not heap:
            return -math.inf
        xs = [h[5] + h[6] for h in heap]
        mx = max(xs)
        return mx + math.log(sum(math.exp(x - mx) for x in xs))

    def lse(xs):
        mx = max(xs)
        return mx + math.log(sum(math.exp(x - mx) for x in xs))

    while heap:
        _, _, R, P, X, WR, _ = heapq.heappop(heap)
        pops += 1
        if not P:
            if not X:
                cliques.append(R)
                logw.append(WR)
                if len(cliques) >= limit:
                    break
                if mass_tail > 0 and len(cliques) % 64 == 0:
                    if frontier() - lse(logw) < math.log(mass_tail):
                        break
            continue
        if any(not (P & ~adj[x]) for x in bits(X)):
            continue
        u = _pivot(P, X, adj)
        for v in list(bits(P & ~adj[u])):
            P2, X2 = P & adj[v], X & adj[v]
            W2 = WR + w[v]
            ub = W2 + sum(pos[j] for j in bits(P2))
            heapq.heappush(heap, (-ub, next(tick), R | (1 << v), P2, X2, W2,
                                  sum(sp[j] for j in bits(P2))))
            P &= ~(1 << v)
            X |= 1 << v
        if max_nodes and len(heap) > max_nodes:
            raise MemoryError(f"bk_best: frontier passed {max_nodes} nodes after {pops} expansions "
                              f"and {len(cliques)} leaves ({a} options)")
    return {"cliques": cliques, "logw": logw, "exact": not heap, "log_frontier": frontier(),
            "pops": pops}


def radius_configs(adj: list[int], M: int, rho: int) -> list[int]:
    """Every maximal clique ``F`` with ``|F \\ M| <= rho``, ``M`` itself a maximal clique.

    ``F`` is fixed by its non-emitted part ``N = F \\ M``: an emitted option compatible with all of
    ``N`` is compatible with all of ``F`` (``M`` is a clique), so maximality forces it in, and
    ``F = N + keep(N)``, ``keep(N) = M & adj[v] for every v in N``.  So enumerate the pairwise
    compatible ``N`` with ``|N| <= rho`` and keep those whose ``F`` has no common neighbour.
    ``O(a^rho)`` candidate sets; ``M`` itself is ``N = {}``.
    """
    a = len(adj)
    full = (1 << a) - 1
    non = [v for v in range(a) if not M >> v & 1]
    out = []

    def rec(N: int, keep: int, cand: int, start: int, k: int) -> None:
        F = keep | N
        cn = full
        for y in bits(F):
            cn &= adj[y]
            if not cn:
                out.append(F)
                break
        if k == rho:
            return
        for idx in range(start, len(non)):
            v = non[idx]
            if cand >> v & 1:
                rec(N | (1 << v), keep & adj[v], cand & adj[v], idx + 1, k + 1)

    rec(0, M, full, 0, 0)
    return out


def brute_maximal(abits: list[int]) -> set[frozenset[int]]:
    """Gate N2(b)'s reference: every maximal laminar subfamily, by subset enumeration."""
    a = len(abits)
    out = set()
    for mask in range(1 << a):
        fam = [abits[i] for i in range(a) if mask >> i & 1]
        if not is_laminar(fam):
            continue
        fs = set(fam)
        if is_maximal(fs, abits):
            out.add(frozenset(i for i in range(a) if mask >> i & 1))
    return out


# ---------------------------------------------------------------------------- joint lazy k-best

def kbest_product(lists: list[list[float]], k: int):
    """The ``k`` best index vectors of a product of descending score lists, best first.

    Successors increment only coordinates ``c >= j``, ``j`` the coordinate last incremented, so each
    vector is generated exactly once and no visited set is needed.  Yields ``(score, idx)``.
    """
    B = len(lists)
    start = tuple([0] * B)
    s0 = sum(L[0] for L in lists)
    heap = [(-s0, 0, start, 0)]
    tick = itertools.count(1)
    got = 0
    while heap and got < k:
        neg, _, idx, j = heapq.heappop(heap)
        yield -neg, idx
        got += 1
        for c in range(j, B):
            i = idx[c]
            if i + 1 < len(lists[c]):
                s = -neg - lists[c][i] + lists[c][i + 1]
                nxt = idx[:c] + (i + 1,) + idx[c + 1:]
                heapq.heappush(heap, (-s, next(tick), nxt, c))
