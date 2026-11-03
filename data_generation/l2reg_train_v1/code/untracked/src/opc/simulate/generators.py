"""Level-k generators, and which one a blob of a network is (A64).

A **level-k generator** (van Iersel et al. 2009; Gambette, Berry & Paul 2009) is a biconnected,
acyclic, directed multigraph with

* one root, of in-degree 0 and out-degree 2;
* tree vertices, of in-degree 1 and out-degree 2;
* exactly ``k`` reticulations, of in-degree 2 and out-degree at most 1 -- one of out-degree 0 is a
  **sink**.

Every blob of a binary network is a generator whose arcs have been subdivided, so suppressing the
blob's in-1/out-1 vertices recovers it (:func:`blob_generator`).  With ``s`` sinks a level-k
generator has ``k + s - 2`` tree vertices, ``2k + s - 1`` vertices and ``3k + s - 2`` arcs.  There
are 1, 4 and 65 of them at k = 1, 2, 3 (Gambette, Berry & Paul, CPM 2009, who count 1,993 at
k = 4).

Enumeration
-----------
:func:`generators` builds them from the definition alone, never from the literature's construction
rules, so the published counts are an independent check on it.  Vertices are added in topological
order, root first.  After the root, the type sequence is every distinct arrangement of the tree
vertices, non-sink reticulations and sinks, and each new vertex takes a *multiset* of in-arcs from
earlier vertices within their remaining out-degree.  It is a multiset because parallel arcs are
allowed: the level-1 generator is two parallel arcs.  A result is kept when it has no cut vertex.
Two mistakes corrupt the count silently. Set-based adjacency drops parallel arcs, and testing
2-edge-connectivity instead of the absence of a cut vertex overcounts.

Canonical form and IDs
----------------------
An isomorphism preserves every vertex's (in, out)-degree, so :func:`canonical` permutes only within
the four classes root / tree / reticulation / sink, and keeps the smallest sorted arc tuple.  That
is at most 4! x 3! = 144 permutations at k = 3.  IDs are ``G{k}s{s}-{i:02d}``, numbered in
(sinks, canonical form) order, so they depend on nothing but the definition.  :data:`PINNED` fixes
the SHA-256 of each table, so a change to either the enumeration or the order fails loudly instead
of silently relabelling a dataset's ground truth.  The four level-2 generators are :mod:`level2`'s
S0-S3 (``tests/test_generators.py``).

Blobs
-----
:func:`blob_generator` works on ``N``, never on ``R``.  In ``R`` a hybrid merged with its child has
in-degree 2 and out-degree 2, which no generator vertex has.

Within the blob, the vertices of in-degree 1 and out-degree 1 are subdivision vertices and are
suppressed: a speciation vertex with a pendant subtree, or a donor whose tree child leaves the
blob.  Chains are followed and parallel arcs kept.  What remains is

* the blob's root (0, 2);
* tree vertices (1, 2), including donors whose tree child stays inside;
* reticulations (2, 1): a hybrid whose child is inside;
* sinks (2, 0): a hybrid whose child edge is a bridge.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from opc.simulate import networks as N

__all__ = [
    "MAX_K",
    "PINNED",
    "Generator",
    "canonical",
    "generators",
    "lookup",
    "table_sha256",
    "suppress",
    "n_sinks",
    "blob_generator",
]

#: Enumeration is exhaustive and pure Python.  k = 4 (1,993 generators) is not needed here.
MAX_K = 3

ROOT, TREE, RETIC, SINK = 0, 1, 2, 3
_KIND = {(0, 2): ROOT, (1, 2): TREE, (2, 1): RETIC, (2, 0): SINK}
_IN = {ROOT: 0, TREE: 1, RETIC: 2, SINK: 2}
_OUT = {ROOT: 2, TREE: 2, RETIC: 1, SINK: 0}

Arcs = tuple[tuple[int, int], ...]

#: SHA-256 of :func:`table_sha256`'s rendering of each table -- the IDs a dataset records.
PINNED: dict[int, str] = {
    1: "238891bc3c06174a99a1a3a6e08b8632659545dc33f94cd2f3241ed9c9ee8b1f",
    2: "057effe31a97bf63a96d6c291736d43e152f124bd315e8eb762f41ff9b3515f4",
    3: "06715dbda5be2c16486784a14f04f80457f21a79d496ce2271ef3389b74cb255",
}


@dataclass(frozen=True)
class Generator:
    """One generator.  ``arcs`` is the canonical form: vertex 0 is the root, then the tree
    vertices, the non-sink reticulations and the sinks, each class in canonical order."""

    id: str
    k: int
    n_sinks: int
    arcs: Arcs

    @property
    def n_vertices(self) -> int:
        return 2 * self.k + self.n_sinks - 1

    @property
    def n_arcs(self) -> int:
        return 3 * self.k + self.n_sinks - 2

    @property
    def parallel_arcs(self) -> bool:
        return len(set(self.arcs)) < len(self.arcs)

    def row(self) -> dict:
        return {"id": self.id, "k": self.k, "n_sinks": self.n_sinks,
                "parallel_arcs": self.parallel_arcs, "arcs": [list(a) for a in self.arcs]}


# ------------------------------------------------------------------------------ canonical form


def _kinds(arcs) -> dict[int, int]:
    indeg: Counter = Counter()
    outdeg: Counter = Counter()
    for u, v in arcs:
        outdeg[u] += 1
        indeg[v] += 1
    kinds = {}
    for v in set(indeg) | set(outdeg):
        kind = _KIND.get((indeg[v], outdeg[v]))
        if kind is None:
            raise ValueError(f"vertex {v} has in-degree {indeg[v]} and out-degree {outdeg[v]}: "
                             "no generator vertex does")
        kinds[v] = kind
    return kinds


def canonical(arcs) -> Arcs:
    """The smallest sorted arc tuple over relabellings that keep root, tree vertices,
    reticulations and sinks in that order -- the only relabellings an isomorphism can induce."""
    kinds = _kinds(arcs)
    groups = [sorted(v for v, kind in kinds.items() if kind == c) for c in (ROOT, TREE, RETIC, SINK)]
    if len(groups[ROOT]) != 1:
        raise ValueError(f"{len(groups[ROOT])} vertices of in-degree 0; a generator has one root")
    best = None
    for perm in itertools.product(*(itertools.permutations(g) for g in groups)):
        rank = {v: i for i, v in enumerate(itertools.chain.from_iterable(perm))}
        key = tuple(sorted((rank[u], rank[v]) for u, v in arcs))
        if best is None or key < best:
            best = key
    return best


# ------------------------------------------------------------------------------ enumeration


def _biconnected(n: int, arcs) -> bool:
    """Connected with no cut vertex: delete each vertex in turn and search what is left."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in arcs:
        adj[u].append(v)
        adj[v].append(u)
    for gone in (None, *range(n)):
        rest = [v for v in range(n) if v != gone]
        seen, stack = {rest[0]}, [rest[0]]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y != gone and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != len(rest):
            return False
    return True


def _extend(kinds, cap, arcs, i):
    if i == len(kinds):
        # In-degrees and out-degrees both sum to the arc count, so every in-arc placed means every
        # unit of out-capacity used; ``not any(cap)`` is that, checked rather than assumed.
        if not any(cap) and _biconnected(len(kinds), arcs):
            yield tuple(arcs)
        return
    need = _IN[kinds[i]]
    for tails in itertools.combinations_with_replacement(range(i), need):
        use = Counter(tails)
        if any(cap[j] < m for j, m in use.items()):
            continue
        for j, m in use.items():
            cap[j] -= m
        arcs.extend((j, i) for j in tails)
        yield from _extend(kinds, cap, arcs, i + 1)
        del arcs[-need:]
        for j, m in use.items():
            cap[j] += m


def _raw(k: int):
    """Every level-k generator as an arc list, in some topological order, isomorphs included."""
    for s in range(1, k + 1):
        pool = [TREE] * (k + s - 2) + [RETIC] * (k - s) + [SINK] * s
        for seq in sorted(set(itertools.permutations(pool))):
            if seq[-1] != SINK:
                continue  # the last vertex has nothing after it to send an arc to
            kinds = (ROOT, *seq)
            yield from _extend(kinds, [_OUT[x] for x in kinds], [], 1)


def _sinks(form: Arcs) -> int:
    return sum(kind == SINK for kind in _kinds(form).values())


@lru_cache(maxsize=None)
def generators(k: int) -> tuple[Generator, ...]:
    """Every level-k generator, up to isomorphism, in ID order."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"generators are enumerated for k in 1..{MAX_K}, not {k}")
    forms = sorted({canonical(a) for a in _raw(k)}, key=lambda f: (_sinks(f), f))
    out, per = [], Counter()
    for f in forms:
        s = _sinks(f)
        per[s] += 1
        out.append(Generator(f"G{k}s{s}-{per[s]:02d}", k, s, f))
    return tuple(out)


@lru_cache(maxsize=None)
def _by_form(k: int) -> dict[Arcs, Generator]:
    return {g.arcs: g for g in generators(k)}


def lookup(arcs) -> Generator:
    """The generator ``arcs`` is isomorphic to; raises if it is none."""
    form = canonical(arcs)
    k = sum(kind in (RETIC, SINK) for kind in _kinds(form).values())
    g = _by_form(k).get(form)
    if g is None:
        raise ValueError(f"not a level-{k} generator: {form}")
    return g


def table_sha256(k: int) -> str:
    text = "".join(f"{g.id} {' '.join(f'{u}>{v}' for u, v in g.arcs)}\n" for g in generators(k))
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------ blobs


def _degrees(block: N.Block) -> tuple[Counter, Counter]:
    indeg: Counter = Counter()
    outdeg: Counter = Counter()
    for p, c in block.edges:
        outdeg[p] += 1
        indeg[c] += 1
    return indeg, outdeg


def suppress(block: N.Block) -> list[tuple[int, int]]:
    """The blob's arcs with every in-1/out-1 vertex suppressed, over the ``N`` ids it keeps."""
    indeg, outdeg = _degrees(block)
    out: dict[int, list[int]] = {}
    for p, c in block.edges:
        out.setdefault(p, []).append(c)
    verts = set(indeg) | set(outdeg)
    passing = {v for v in verts if indeg[v] == 1 and outdeg[v] == 1}
    arcs = []
    for u in sorted(verts - passing):
        for x in out.get(u, ()):
            while x in passing:
                (x,) = out[x]
            arcs.append((u, x))
    return arcs


def n_sinks(block: N.Block) -> int:
    """Hybrids whose child edge leaves the blob -- cheap, so a placement loop can gate on it."""
    indeg, outdeg = _degrees(block)
    return sum(indeg[v] == 2 and outdeg[v] == 0 for v in indeg)


def blob_generator(block: N.Block) -> Generator:
    """Which generator a blob of ``N`` is.  Raises if its reticulation vertices are not exactly
    its hybrids, or if what suppression leaves is not a generator."""
    arcs = suppress(block)
    kinds = _kinds(arcs)
    retics = {v for v, kind in kinds.items() if kind in (RETIC, SINK)}
    if retics != set(block.hybrids):
        raise ValueError(f"reticulation vertices {sorted(retics)} are not the blob's hybrids "
                         f"{sorted(block.hybrids)}")
    return lookup(arcs)
