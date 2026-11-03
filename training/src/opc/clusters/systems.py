"""Exact, small-scale predicates on clustering systems.

Everything here works on ``frozenset[int]`` clusters and is written for clarity over speed -- it
is the *reference* semantics that the fast bitset kernels in :mod:`opc.clusters.relations` are
tested against, and it is what the theory tests in ``tests/test_theory.py`` exercise.

Definitions follow §3 of the study spec, which in turn follows Hellmuth, Schaller & Stadler,
*Clustering Systems of Phylogenetic Networks*:

(L)     no ordered triple ``(A; B, C)`` with ``A`` overlapping both ``B`` and ``C`` and
        ``A & B != A & C``.
(N3O)   no three pairwise overlapping clusters (no triangle in the overlap graph).
closed  every pairwise intersection of two overlapping members is itself a member.
"""

from __future__ import annotations

from itertools import combinations

__all__ = [
    "overlaps",
    "overlap_graph",
    "components",
    "satisfies_n3o",
    "l_violations",
    "satisfies_l",
    "is_closed",
    "closure",
    "is_feasible",
]

Cluster = frozenset


def overlaps(a: frozenset, b: frozenset) -> bool:
    """True iff ``a`` and ``b`` properly overlap: they meet, and neither contains the other."""
    inter = a & b
    return bool(inter) and inter != a and inter != b


def overlap_graph(system) -> dict[frozenset, set[frozenset]]:
    """Adjacency of ``O(system)``.  Isolated members appear with an empty neighbour set."""
    members = list(system)
    adj: dict[frozenset, set[frozenset]] = {c: set() for c in members}
    for a, b in combinations(members, 2):
        if overlaps(a, b):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def components(system) -> list[set[frozenset]]:
    """Connected components of ``O(system)``, including singletons (isolated clusters)."""
    adj = overlap_graph(system)
    seen: set[frozenset] = set()
    out = []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.add(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        out.append(comp)
    return out


def satisfies_n3o(system) -> bool:
    """True iff no three members are pairwise overlapping."""
    for a, b, c in combinations(system, 3):
        if overlaps(a, b) and overlaps(a, c) and overlaps(b, c):
            return False
    return True


def l_violations(system) -> int:
    """Number of ordered ``(L)`` violations ``(A; {B, C})``.

    Brute force over every apex and every unordered pair of its neighbours -- the definition,
    used to validate the ``O(deg)`` group-by-intersection counter in
    :func:`opc.clusters.relations.l_violations_per_vertex`.
    """
    adj = overlap_graph(system)
    total = 0
    for a, nbrs in adj.items():
        for b, c in combinations(sorted(nbrs, key=sorted), 2):
            if (a & b) != (a & c):
                total += 1
    return total


def satisfies_l(system) -> bool:
    return l_violations(system) == 0


def is_closed(system) -> bool:
    """True iff every intersection of two overlapping members is itself a member."""
    s = set(system)
    return all((a & b) in s for a, b in combinations(s, 2) if overlaps(a, b))


def closure(system) -> set[frozenset]:
    """``I(S) = S | {A & B : A, B in S overlapping}``.

    Pairwise intersections suffice: for a system satisfying (N3O) no third member overlaps both
    of an overlapping pair, so iterating adds nothing.  We iterate to a fixpoint anyway, and the
    theory tests assert the single-pass result is already closed.
    """
    out = set(system)
    while True:
        new = {a & b for a, b in combinations(out, 2) if overlaps(a, b)} - out
        if not new:
            return out
        out |= new


def is_feasible(subset) -> bool:
    """True iff ``subset`` satisfies both (L) and (N3O) -- the target class for the solver."""
    return satisfies_n3o(subset) and satisfies_l(subset)
