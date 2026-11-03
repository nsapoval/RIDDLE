"""Regular, exactly-level-2 phylogenetic networks: placement and the three gates (A62).

The level-2 benchmark (``python -m opc.l2reg``) needs networks that are

* **regular** -- the Hasse diagram of the hardwired clusters *is* the network, so a cluster-based
  method can reconstruct it without ambiguity;
* **exactly level-2** -- every blob holds at most two reticulations, and at least one holds two;
* of **girth >= 5** -- no undirected cycle shorter than five edges.

:mod:`opc.simulate.networks` is level-1 by construction (its ``consumed`` edge rule), and importing
this module changes nothing it produces.  Reticulations are placed into the same
:class:`~opc.simulate.networks.Network` representation, under different rules, and this module
owns the gates.

Two forms of one network
------------------------
``N`` is binary and time-consistent: every hybrid has one child, reached by a positive-length
edge.  It is what gets simulated.  ``R`` (:func:`contract`) is ``N`` with each hybrid->child edge
contracted.  A binary network is never literally regular -- a hybrid and its child share a
cluster -- so every regularity and girth claim here is a claim about ``R``, and ``N`` is ``R``'s
unique binary expansion (unique because no hybrid's child is a hybrid, :func:`ladders`).
**Never simulate R**: PhyloCoalSimulations sends a hybrid *leaf*'s lineages up its first parent
edge only (``simulatecoalescent_network.jl:226-231``), and PhyloNetworks renames a hybrid leaf on
read (``readwrite.jl:699``).  Both are silent.

The regularity gate
-------------------
:func:`regularity` is the definition, not a sufficient condition: the cluster map on ``R`` is
injective, and ``R``'s edges, read as cluster pairs, are exactly the cover relation of its
clusters.  The weaker check ``tests/test_networks.py`` uses -- contract every edge whose ends
share a cluster, then compare against the covers -- passes a network that is *not* regular once
blobs hold two reticulations, because the contraction hides the duplicate cluster;
``tests/test_level2.py`` pins the counterexample.  Tree-child and shortcut-freeness are reported,
never gated on: together they imply regularity, but regular networks that are not tree-child
are admitted.

Shortcuts
---------
In a time-consistent network only tree edges into hybrids can be shortcuts: ``tree_parent[h] ->
h`` is one iff ``tree_parent[h]`` reaches ``donor_of[h]``.  The level-1 generator checks only the
new hybrid; once a blob holds two reticulations a *later* placement can turn an *earlier*
hybrid's edge into a shortcut through the new horizontal edge, so :func:`shortcuts` checks every
hybrid.

Blob shapes
-----------
A blob holding two reticulations is a subdivided theta graph (its cyclomatic number is its
reticulation count): two branch vertices ``P``, ``Q`` joined by three internally disjoint paths.
It has one source ``rho`` -- of blob-degree 2, so never ``P`` or ``Q`` -- and one or two sinks,
and exactly four directed shapes exist, the four level-2 generators.  :func:`blob_shape` tells
them apart by two facts:

=====  =====  ================================
shape  sinks  the ``P``--``Q`` path through rho
=====  =====  ================================
S0     2      holds no sink
S1     1      holds no sink
S2     1      holds the sink
S3     2      holds a sink
=====  =====  ================================

Placement
---------
Adding donor ``u`` / hybrid ``h`` merges every block holding an edge of the tree path
``u -> z <- h``: every simple ``u``--``h`` path visits the same sequence of blocks and uses an
edge of each.  A new blob must touch none; the second reticulation of a level-2 blob must touch
that blob and nothing else.  Girth and regularity are **not** monotone while a blob is built --
the second reticulation often subdivides the first cycle -- so they are checked when a blob is
complete, and the whole network once more at the end (:func:`check`).
"""

from __future__ import annotations

import time as _time
from collections import Counter, deque
from dataclasses import dataclass, field, replace

import numpy as np

from opc import bitset
from opc.grid import chain_seed
from opc.simulate import networks as N
from opc.solve import closure

__all__ = [
    "SHAPES",
    "MIN_GIRTH",
    "Layout",
    "Regularity",
    "Theta",
    "place",
    "ladders",
    "shortcuts",
    "is_tree_child",
    "contract",
    "regularity",
    "nontrivial_blocks",
    "block_of",
    "block_root",
    "girth",
    "theta",
    "blob_shape",
    "check",
    "blob_report",
    "regular_json",
    "build_level2",
]

SHAPES = ("S0", "S1", "S2", "S3")
MIN_GIRTH = 5


# ------------------------------------------------------------------------------ primitives


def place(net: N.Network, donor_edge: int, hybrid_edge: int, t: float) -> tuple[int, int]:
    """Insert the horizontal edge ``donor -> hybrid`` at time ``t``, in place.

    ``donor_edge`` / ``hybrid_edge`` name tree edges by their child, as ``networks`` does; both
    must span ``t``.  Returns ``(donor, hybrid)``.
    """
    if donor_edge == hybrid_edge:
        raise ValueError("donor and hybrid edges must differ")
    for x in (donor_edge, hybrid_edge):
        p = net.tree_parent[x]
        if p < 0 or not net.time[p] < t < net.time[x]:
            raise ValueError(f"the tree edge into {x} does not span t={t}")
    donor = N._split_edge(net, donor_edge, t)  # noqa: SLF001
    hybrid = N._split_edge(net, hybrid_edge, t)  # noqa: SLF001
    net.horiz_child[donor] = hybrid
    net.donor_of[hybrid] = donor
    net.retics.append(N.Reticulation(hybrid, donor, len(net.retics), t))
    return donor, hybrid


def ladders(net: N.Network) -> list[int]:
    """Hybrids whose child is a hybrid.

    In ``R`` such a pair merges into one vertex of in-degree 3, and ``N`` stops being ``R``'s
    unique binary expansion -- which parent pair forms the upper hybrid is then a free choice.
    """
    return sorted(h for h in net.donor_of if any(net.is_hybrid(c) for c in net.tree_children[h]))


def _reaches(net: N.Network, src: int, dst: int) -> bool:
    # Time never decreases along an edge, so nothing later than ``dst`` can lead back to it.
    limit = net.time[dst]
    stack, seen = [src], {src}
    while stack:
        v = stack.pop()
        if v == dst:
            return True
        for c in net.net_children(v):
            if c not in seen and net.time[c] <= limit:
                seen.add(c)
                stack.append(c)
    return False


def shortcuts(net: N.Network) -> list[int]:
    """Hybrids whose tree in-edge is a shortcut -- the only kind this representation can have."""
    return sorted(h for h, d in net.donor_of.items() if _reaches(net, net.tree_parent[h], d))


def is_tree_child(net: N.Network) -> bool:
    """Every non-leaf vertex has a non-hybrid child (equivalently: every vertex is visible)."""
    for v in N._postorder(net):  # noqa: SLF001
        if not net.is_leaf(v) and all(net.is_hybrid(c) for c in net.net_children(v)):
            return False
    return True


def contract(net: N.Network) -> N.Network:
    """``R``: every hybrid->child edge contracted.  The hybrid keeps its id and its time.

    The hybrid takes over its child's children, and its horizontal edge if the child was a donor;
    a hybrid whose child is a leaf becomes a hybrid leaf.  The child's out-edges lengthen by the
    contracted length, so ``R``'s branch lengths are for drawing, never for simulating.
    """
    lad = ladders(net)
    if lad:
        raise ValueError(f"hybrids {lad} have a hybrid child; R would have in-degree 3")
    r = net.copy()
    for h in sorted(r.donor_of):
        (c,) = r.tree_children[h]
        r.tree_children[h] = list(r.tree_children[c])
        for k in r.tree_children[h]:
            r.tree_parent[k] = h
        if c in r.horiz_child:
            hc = r.horiz_child.pop(c)
            r.horiz_child[h] = hc
            r.donor_of[hc] = h
        if c in r.leaf_taxon:
            r.leaf_taxon[h] = r.leaf_taxon.pop(c)
        r.tree_children[c] = []
        r.tree_parent[c] = -2  # tombstone, as networks._suppress_degree2 does
    r.retics = [replace(ret, donor=r.donor_of[ret.hybrid]) for ret in r.retics]
    return r


# ------------------------------------------------------------------------------ regularity


@dataclass(frozen=True)
class Regularity:
    """What :func:`regularity` measured.  ``ok`` is the gate; the rest says why it failed."""

    n_vertices: int        # reachable vertices of R
    n_clusters: int        # distinct hardwired clusters among them
    n_edges: int
    n_covers: int
    max_indegree: int
    non_cover_edges: int   # edges of R whose cluster pair is not a cover
    missing_covers: int    # covers that no edge of R realises
    ladders: tuple[int, ...]
    shortcuts: tuple[int, ...]
    tree_child: bool

    @property
    def injective(self) -> bool:
        return self.n_vertices == self.n_clusters

    @property
    def ok(self) -> bool:
        return (not self.ladders and self.injective and self.max_indegree <= 2
                and self.non_cover_edges == 0 and self.missing_covers == 0
                and self.n_edges == self.n_covers)


def _edges(net: N.Network) -> list[tuple[int, int]]:
    return [(v, c) for v in N._postorder(net) for c in net.net_children(v)]  # noqa: SLF001


def regularity(net: N.Network) -> Regularity:
    """Is ``R`` the Hasse diagram of its own hardwired clusters?  The definition, checked.

    Covers come from :func:`opc.solve.closure.hasse_covers_words`, an implementation that never
    sees the network -- only the cluster set.
    """
    lad, sc, tc = tuple(ladders(net)), tuple(shortcuts(net)), is_tree_child(net)
    if lad:
        return Regularity(0, 0, 0, 0, 3, 0, 0, lad, sc, tc)
    r = contract(net)
    below = N.hardwired_clusters(r)
    distinct = sorted(set(below.values()), key=lambda c: (len(c), sorted(c)))
    index = {c: i for i, c in enumerate(distinct)}
    edges = _edges(r)
    indeg = Counter(c for _, c in edges)
    pairs = [(index[below[v]], index[below[c]]) for v, c in edges]
    words = np.stack([bitset.from_indices(sorted(c), net.n_taxa) for c in distinct])
    pi, ci = closure.hasse_covers_words(words, net.n_taxa)
    covers = set(zip(pi.tolist(), ci.tolist()))
    return Regularity(
        n_vertices=len(below), n_clusters=len(distinct), n_edges=len(edges),
        n_covers=len(covers), max_indegree=max(indeg.values(), default=0),
        non_cover_edges=sum(1 for p in pairs if p not in covers),
        missing_covers=len(covers - set(pairs)),
        ladders=lad, shortcuts=sc, tree_child=tc,
    )


# ------------------------------------------------------------------------------ blobs


def nontrivial_blocks(net: N.Network) -> list[N.Block]:
    return [b for b in N.blocks(net) if not b.is_bridge]


def block_of(blocks: list[N.Block], hybrid: int) -> N.Block:
    (b,) = [b for b in blocks if hybrid in b.hybrids]
    return b


def block_root(block: N.Block) -> int:
    """The blob's unique source: the vertex with no in-edge inside it."""
    heads = {c for _, c in block.edges}
    (rho,) = {p for p, _ in block.edges} - heads
    return rho


def girth(block: N.Block) -> int:
    """Shortest undirected cycle in one block: BFS from every vertex, exact.  0 for a bridge."""
    adj: dict[int, list[tuple[int, int]]] = {}
    for eid, (p, c) in enumerate(block.edges):
        adj.setdefault(p, []).append((c, eid))
        adj.setdefault(c, []).append((p, eid))
    best = None
    for s in adj:
        dist, via, queue = {s: 0}, {s: -1}, deque([s])
        while queue:
            x = queue.popleft()
            for y, eid in adj[x]:
                if eid == via[x]:
                    continue
                if y not in dist:
                    dist[y], via[y] = dist[x] + 1, eid
                    queue.append(y)
                else:
                    cyc = dist[x] + dist[y] + 1
                    best = cyc if best is None or cyc < best else best
    return best or 0


@dataclass(frozen=True)
class Theta:
    """A level-2 blob as a theta graph: branch vertices, the three paths, its source and sinks."""

    p: int
    q: int
    paths: tuple[tuple[int, ...], ...]   # vertex sequences p ... q
    rho: int
    sinks: tuple[int, ...]
    rho_path: int                        # index of the path rho is interior to

    @property
    def path_lengths(self) -> tuple[int, ...]:
        return tuple(len(pa) - 1 for pa in self.paths)


def theta(block: N.Block) -> Theta:
    und: dict[int, list[int]] = {}
    outdeg: Counter = Counter()
    for p, c in block.edges:
        und.setdefault(p, []).append(c)
        und.setdefault(c, []).append(p)
        outdeg[p] += 1
    branch = sorted(v for v, nb in und.items() if len(nb) == 3)
    if len(branch) != 2 or any(len(nb) > 3 for nb in und.values()):
        raise ValueError(f"not a subdivided theta graph ({len(branch)} branch vertices)")
    p, q = branch
    paths = []
    for first in und[p]:
        path, prev, cur = [p], p, first
        while cur != q:
            path.append(cur)
            (cur, prev) = (next(w for w in und[cur] if w != prev), cur)
        path.append(q)
        paths.append(tuple(path))
    rho = block_root(block)
    sinks = tuple(sorted(v for v in und if outdeg[v] == 0))
    rho_path = next(i for i, pa in enumerate(paths) if rho in pa[1:-1])
    return Theta(p, q, tuple(paths), rho, sinks, rho_path)


def blob_shape(block: N.Block) -> str:
    """``"L1"`` for a one-reticulation blob, ``"S0"``..``"S3"`` for a level-2 one (module doc)."""
    if block.n_retics == 1:
        return "L1"
    if block.n_retics != 2:
        raise ValueError(f"a blob with {block.n_retics} reticulations has no level-2 shape")
    th = theta(block)
    holds = any(s in th.paths[th.rho_path][1:-1] for s in th.sinks)
    return {(2, False): "S0", (1, False): "S1", (1, True): "S2", (2, True): "S3"}[
        (len(th.sinks), holds)]


# ------------------------------------------------------------------------------ the whole network


@dataclass(frozen=True)
class Layout:
    """How many blobs of each kind, and the target shape of each level-2 blob (``None`` = any)."""

    n_l2: int
    n_l1: int
    shapes: tuple[str | None, ...] = ()

    def __post_init__(self):
        if self.n_l2 < 1:
            raise ValueError("an exactly-level-2 network needs at least one level-2 blob")
        if self.shapes and len(self.shapes) != self.n_l2:
            raise ValueError(f"{len(self.shapes)} target shapes for {self.n_l2} level-2 blobs")
        bad = [s for s in self.shapes if s is not None and s not in SHAPES]
        if bad:
            raise ValueError(f"unknown shapes {bad}")

    @property
    def k(self) -> int:
        return 2 * self.n_l2 + self.n_l1

    def targets(self) -> tuple[str | None, ...]:
        return self.shapes or (None,) * self.n_l2


def _tags(net: N.Network) -> dict[int, str]:
    return {ret.hybrid: f"H{k + 1}" for k, ret in enumerate(net.retics)}


def blob_report(net: N.Network) -> list[dict]:
    """One row per blob of ``N``: its shape, girth in both forms, size and position."""
    tags = _tags(net)
    below = N.hardwired_clusters(net)
    r_blocks = nontrivial_blocks(contract(net))
    rows = []
    for b in nontrivial_blocks(net):
        rho = block_root(b)
        row = {
            "hybrids": [tags[h] for h in b.hybrids],
            "n_retics": b.n_retics,
            # A blob over the level cap still gets a row -- the report describes whatever it is
            # handed, and :func:`check` is what says no.
            "shape": blob_shape(b) if b.n_retics <= 2 else f"L{b.n_retics}",
            "girth_R": girth(block_of(r_blocks, b.hybrids[0])),
            "girth_N": girth(b),
            "n_vertices_N": len(b.vertices),
            "n_edges_N": len(b.edges),
            "root_taxa": len(below[rho]),
            "root_time": net.time[rho],
            "hybrid_cluster_sizes": [len(below[h]) for h in b.hybrids],
        }
        if b.n_retics == 2:
            th = theta(b)
            row["theta_path_lengths_N"] = sorted(th.path_lengths)
        rows.append(row)
    return rows


def check(net: N.Network, layout: Layout) -> dict:
    """Every gate, on a finished network.  ``ok`` is their conjunction."""
    blobs = nontrivial_blocks(net)
    counts = Counter(b.n_retics for b in blobs)
    reg = regularity(net)
    report = blob_report(net) if not reg.ladders else []
    shapes = sorted(row["shape"] for row in report if row["n_retics"] == 2)
    want = sorted(s for s in layout.targets() if s is not None)
    free = sum(s is None for s in layout.targets())
    shapes_ok = (not want) or (len(shapes) == len(want) + free
                               and not (Counter(want) - Counter(shapes)))
    girth_min = min((row["girth_R"] for row in report), default=0)
    out = {
        "level": N.level(net),
        "n_blobs_l2": counts.get(2, 0),
        "n_blobs_l1": counts.get(1, 0),
        "max_blob_retics": max(counts, default=0),
        "regular": reg.ok,
        "regularity": {k: getattr(reg, k) for k in (
            "n_vertices", "n_clusters", "n_edges", "n_covers", "max_indegree",
            "non_cover_edges", "missing_covers")} | {
            "ladders": list(reg.ladders), "shortcuts": list(reg.shortcuts),
            "tree_child": reg.tree_child},
        "girth_R_min": girth_min,
        "shapes": shapes,
        "shapes_ok": shapes_ok,
    }
    out["ok"] = (out["level"] == 2 and out["max_blob_retics"] <= 2
                 and out["n_blobs_l2"] == layout.n_l2 and out["n_blobs_l1"] == layout.n_l1
                 and reg.ok and girth_min >= MIN_GIRTH and shapes_ok)
    return out


def regular_json(net: N.Network) -> dict:
    """``R`` as the object a cluster-based method reconstructs: its clusters and its covers."""
    r = contract(net)
    below = N.hardwired_clusters(r)
    order = sorted(below, key=lambda v: (len(below[v]), sorted(below[v])))
    idx = {v: i for i, v in enumerate(order)}
    tags = _tags(r)
    return {
        "clusters": [sorted((net.labels[t] for t in below[v]),
                            key=lambda s: int(s[1:]) if s[1:].isdigit() else s) for v in order],
        "edges": sorted([idx[v], idx[c]] for v, c in _edges(r)),
        "hybrids": {tags[h]: idx[h] for h in r.donor_of},
    }


# ------------------------------------------------------------------------------ the generator


@dataclass
class _Cfg:
    max_base: int = 20          # base trees before giving up
    max_first: int = 40         # first-reticulation draws per blob before redrawing the tree
    max_second: int = 400       # second-reticulation proposals per first reticulation
    first_tries: int = 400      # time draws when looking for one first reticulation
    pairs_per_time: int = 48


@dataclass
class _Log:
    counters: Counter = field(default_factory=Counter)

    def __call__(self, reason: str) -> None:
        self.counters[reason] += 1


def _touched(net: N.Network, emap: dict, sides: list[int]) -> set[int]:
    return {emap[(net.tree_parent[x], x)] for x in sides if (net.tree_parent[x], x) in emap}


def _orient(net: N.Network, a: int, b: int, rng) -> tuple[int, int, int] | None:
    """Donor edge, hybrid edge, LCA -- or None when both would put the hybrid under the LCA."""
    c1, c2 = (a, b) if rng.random() < 0.5 else (b, a)
    z = N._lca(net, c1, c2)  # noqa: SLF001
    if net.tree_parent[c2] == z:
        if net.tree_parent[c1] == z:
            return None
        c1, c2 = c2, c1
    return c1, c2, z


def _sides(net: N.Network, c1: int, c2: int, z: int) -> tuple[list[int], list[int]]:
    s1 = N._up_edges(net, net.tree_parent[c1], z) + [c1]  # noqa: SLF001
    s2 = N._up_edges(net, net.tree_parent[c2], z) + [c2]  # noqa: SLF001
    return s1, s2


def _first(net: N.Network, rng, log: _Log, cfg: _Cfg, min_tree_edges: int) -> N.Network | None:
    """A reticulation that starts a new blob: its tree path touches no existing blob."""
    emap = {e: i for i, b in enumerate(nontrivial_blocks(net)) for e in b.edges}
    horizon = max(net.time)
    for _ in range(cfg.first_tries):
        t = float(rng.uniform(0.0, horizon))
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        if len(alive) < 2:
            continue
        for _ in range(cfg.pairs_per_time):
            a, b = rng.choice(len(alive), size=2, replace=False)
            o = _orient(net, alive[a], alive[b], rng)
            if o is None:
                log("first:shortcut")
                continue
            c1, c2, z = o
            s1, s2 = _sides(net, c1, c2, z)
            if _touched(net, emap, s1 + s2):
                log("first:touches_blob")
                continue
            if len(s1) + len(s2) < min_tree_edges:
                log("first:cycle_short")
                continue
            trial = net.copy()
            _, h = place(trial, c1, c2, t)
            if ladders(trial):
                log("first:ladder")
                continue
            got = block_of(nontrivial_blocks(trial), h)
            if got.n_retics != 1:  # the merge rule predicted a fresh blob
                raise RuntimeError(f"merge rule: predicted a new blob, got {got.n_retics}")
            return trial
    return None


def _tree_clade(net: N.Network, v: int) -> set[int]:
    out, stack = set(), [v]
    while stack:
        x = stack.pop()
        out.add(x)
        stack.extend(net.tree_children[x])
    return out


def _second(net: N.Network, h1: int, rng, log: _Log, cfg: _Cfg):
    """Proposals that join the blob holding ``h1`` and nothing else.  Yields trial networks.

    Half the proposals take both edges inside the blob root's clade and half take one inside and
    one outside: every row of the placement table reaches every shape, but a proposal that
    mostly reaches outside the clade makes S0 -- both ends inside, on opposite halves -- rare.
    """
    blocks = nontrivial_blocks(net)
    target = block_of(blocks, h1)
    ti = blocks.index(target)
    emap = {e: i for i, b in enumerate(blocks) for e in b.edges}
    rho = block_root(target)
    clade = _tree_clade(net, rho)
    t_lo, horizon = net.time[rho], max(net.time)
    for _ in range(cfg.max_second):
        t = float(rng.uniform(t_lo, horizon))
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        inside = [x for x in alive if x in clade]
        outside = [x for x in alive if x not in clade]
        both_in = rng.random() < 0.5
        if both_in and len(inside) >= 2:
            a, b = (inside[i] for i in rng.choice(len(inside), size=2, replace=False))
        elif inside and outside:
            a, b = inside[rng.integers(len(inside))], outside[rng.integers(len(outside))]
        else:
            log("second:no_pair")
            continue
        o = _orient(net, a, b, rng)
        if o is None:
            log("second:shortcut")
            continue
        c1, c2, z = o
        s1, s2 = _sides(net, c1, c2, z)
        if _touched(net, emap, s1 + s2) != {ti}:
            log("second:wrong_blobs")
            continue
        trial = net.copy()
        _, h2 = place(trial, c1, c2, t)
        if ladders(trial):
            log("second:ladder")
            continue
        got = block_of(nontrivial_blocks(trial), h2)
        if h1 not in got.hybrids or got.n_retics != 2:
            raise RuntimeError(f"merge rule: predicted a 2-blob with {h1}, got {got.hybrids}")
        yield trial


def _complete(net: N.Network, h: int, target: str | None, log: _Log) -> bool:
    """The non-monotone gates, once a blob is finished: shape, girth on ``R``, regularity."""
    b = block_of(nontrivial_blocks(net), h)
    if target is not None and blob_shape(b) != target:
        log("complete:shape")
        return False
    if girth(block_of(nontrivial_blocks(contract(net)), h)) < MIN_GIRTH:
        log("complete:girth")
        return False
    if not regularity(net).ok:
        log("complete:irregular")
        return False
    return True


def _blob(net: N.Network, kind: str, target: str | None, rng, log: _Log, cfg: _Cfg):
    for _ in range(cfg.max_first):
        # A level-1 blob's cycle is its girth, and in R it keeps every edge (the hybrid's child
        # edge is a bridge), so the floor can be applied at placement: >= MIN_GIRTH - 1 tree edges
        # plus the horizontal one.  A level-2 blob's first cycle is subdivided by the second.
        one = _first(net, rng, log, cfg, MIN_GIRTH - 1 if kind == "L1" else 0)
        if one is None:
            log("blob:no_first")
            return None
        h1 = one.retics[-1].hybrid
        if kind == "L1":
            if _complete(one, h1, None, log):
                return one
            continue
        for two in _second(one, h1, rng, log, cfg):
            if _complete(two, h1, target, log):
                return two
        log("blob:first_redrawn")
    return None


def build_level2(n_taxa: int, layout: Layout, seed: int, *, cu: float,
                 gamma_range: tuple[float, float] = (0.25, 0.5),
                 cfg: _Cfg | None = None) -> tuple[N.Network, dict]:
    """A regular, exactly-level-2 network, scaled to median internal branch ``cu``.

    The base tree is :func:`networks.build_base` with no reticulations.  Blobs are built in a
    per-network shuffled order, so no kind of blob always gets the emptiest tree.  Gammas come
    from their own RNG stream, drawn in insertion order, so they do not depend on how many
    proposals were rejected.  Returns ``(net, placement)``: every rejection counter and redraw.
    """
    cfg = cfg or _Cfg()
    lo, hi = gamma_range
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"gamma_range {gamma_range} must lie inside (0, 1)")
    log = _Log()
    t0 = _time.perf_counter()
    for draw in range(cfg.max_base):
        rng = np.random.default_rng([seed, draw, 1])
        net = N.build_base(n_taxa, 0, seed=chain_seed(seed, "base", draw))
        specs = [("L2", s) for s in layout.targets()] + [("L1", None)] * layout.n_l1
        order = [specs[i] for i in rng.permutation(len(specs))]
        for kind, target in order:
            net = _blob(net, kind, target, rng, log, cfg)
            if net is None:
                break
        if net is None:
            log("base:redrawn")
            continue
        verdict = check(net, layout)
        if not verdict["ok"]:  # every blob passed on completion; this is the belt and braces
            log("base:final_check_failed")
            continue
        grng = np.random.default_rng([seed, draw, 2])
        gammas = {ret.hybrid: float(grng.uniform(lo, hi)) for ret in net.retics}
        net = N.with_gammas(N.scaled(net, cu), gammas)
        return net, {
            "seed": seed, "base_draw": draw,
            "blob_order": [f"{k}:{s}" if s else k for k, s in order],
            "counters": dict(sorted(log.counters.items())),
            "seconds": _time.perf_counter() - t0,
            "cfg": vars(cfg),
        }
    raise RuntimeError(f"no {layout} network in {cfg.max_base} base trees: "
                       f"{dict(log.counters)}")
