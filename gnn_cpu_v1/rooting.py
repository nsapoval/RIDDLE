"""Rooting a gene tree without an outgroup: at its midpoint, or where a rooted species tree says.

``infer_gnn_cpu.py``'s ``--midpoint`` and ``--species-tree``.  Torch-free, so ``check_rooting.py``
can gate it against dendropy without loading a model.  Hand-rolled on ``OG.parse_newick`` for the
reason ``opc.simulate.outgroup`` is: dendropy costs milliseconds per tree.

Both schemes read the input as **unrooted**.  Every degree-2 node -- a written bifurcating root, or a
unary node -- is suppressed and its two edges summed, so a tree written rooted and the same tree
written with a trifurcation are one object and root identically.  Both put the root **on an edge**,
never at a node, so the result is binary at the root and every taxon is kept.  Lengths are written
with ``repr``, as ``prune_to`` writes them, which round-trips a float exactly.

Midpoint
--------

The middle of the longest leaf-to-leaf path, found by two farthest-leaf passes -- exact on a tree
whose lengths are non-negative -- with ties to the smaller label.  The root goes on the path edge
``(u, v)`` with ``d(a, u) < D/2 <= d(a, v)``, so a midpoint that lands on a node gives a zero-length
root edge rather than a polytomy.  A tree with ``D = 0`` or a negative length has no midpoint.

Species tree
------------

Restrict the rooted species tree to the gene tree's taxa R.  Its root splits R into clades:

* **two**, ``C | R - C``.  The gene tree is rooted at the midpoint of the edge carrying that split.
  When one side is a single taxon, that taxon is the outgroup, the edge is its pendant edge, and it
  **stays in the tree**.  When no edge carries the split, the gene tree disagrees with the species
  tree about its own root and there is nothing to root on: ``split_absent``.
* **three or more**, a polytomy in the species tree.  Any clade whose split the gene tree carries is
  a rooting the species tree allows; the smallest is taken, ties to its first label, and
  ``polytomy_no_split`` when there is none.

A gene tree with an unresolved node where the split would be is ``split_absent`` too: the split is
compatible but not present, and resolving the polytomy would invent an edge.

**The species tree's root is trusted, and cannot be checked.**  A trifurcating root is refused --
that is how unrooted output (ASTRAL, RAxML, IQ-TREE) is written, and reading it as a polytomy would
root every gene tree wrong without an error anywhere -- but a binary root written in an arbitrary
place looks exactly like a real one.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from opc.simulate import outgroup as OG

__all__ = ["RootingError", "Unrooted", "midpoint_root", "SpeciesTree"]


class RootingError(ValueError):
    """A tree the scheme cannot root.  ``reason`` is the code a caller counts; the tree is skipped."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _check_labels(names, what: str) -> None:
    if None in names:
        raise ValueError(f"{what} has an unlabelled leaf")
    dup = sorted(x for x, c in Counter(names).items() if c > 1)
    if dup:
        raise ValueError(f"{what} carries {len(dup)} taxa more than once, e.g. {dup[:5]}; rooting "
                         f"needs single-copy trees")


class Unrooted:
    """A tree as an undirected graph: ``nbr[v]`` maps each neighbour of ``v`` to the edge length, and
    no node has degree 2.  Node ids are ``OG.parse_newick``'s; a suppressed node's ``nbr`` is empty."""

    def __init__(self, nbr: list[dict[int, float]], label: list, leaves: list[int]):
        self.nbr = nbr
        self.label = label
        self.leaves = leaves

    @classmethod
    def of(cls, newick: str) -> "Unrooted":
        parent, length, label, children = OG.parse_newick(newick)
        n = len(parent)
        nbr: list[dict[int, float]] = [{} for _ in range(n)]
        for i in range(1, n):
            nbr[i][parent[i]] = nbr[parent[i]][i] = length[i]
        leaves = [i for i in range(n) if not children[i]]
        _check_labels([label[i] for i in leaves], f"tree {newick[:40]!r}...")
        # Suppressing ``v`` leaves its neighbours' degrees unchanged -- each loses ``v`` and gains the
        # other -- so one pass removes every degree-2 node, chains included.  A leaf has degree 1.
        for v in range(n):
            if len(nbr[v]) == 2:
                (a, la), (b, lb) = nbr[v].items()
                del nbr[a][v], nbr[b][v]
                nbr[a][b] = nbr[b][a] = la + lb
                nbr[v].clear()
        return cls(nbr, label, leaves)

    def farthest(self, src: int):
        """``(leaf, dist, prev)``: the leaf farthest from ``src`` (ties to the smaller label), each
        reachable node's distance from ``src``, and its predecessor on that path."""
        dist, prev, stack = {src: 0.0}, {src: -1}, [src]
        while stack:
            x = stack.pop()
            for y, ell in self.nbr[x].items():
                if y not in dist:
                    dist[y] = dist[x] + ell
                    prev[y] = x
                    stack.append(y)
        return min(self.leaves, key=lambda i: (-dist[i], self.label[i])), dist, prev

    def splits(self, bit: dict[str, int]) -> tuple[int, dict[int, tuple[int, int]]]:
        """``(R, edge)``: the leaf set as a mask over ``bit``, and every edge ``(x, y)`` keyed by
        **both** sides of its split, so a clade and its complement find the same edge."""
        anchor = self.leaves[0]
        order, up, stack = [anchor], {anchor: -1}, [anchor]
        while stack:
            x = stack.pop()
            for y in self.nbr[x]:
                if y not in up:
                    up[y] = x
                    order.append(y)
                    stack.append(y)
        is_leaf = set(self.leaves)
        mask = dict.fromkeys(order, 0)
        for x in reversed(order):                 # children before parents
            if x in is_leaf:
                mask[x] |= bit[self.label[x]]
            if up[x] >= 0:
                mask[up[x]] |= mask[x]
        R = mask[anchor]
        edge: dict[int, tuple[int, int]] = {}
        for x in order[1:]:
            edge[mask[x]] = edge[R ^ mask[x]] = (x, up[x])
        return R, edge

    def emit_rooted(self, u: int, v: int, t: float) -> str:
        """Newick rooted on edge ``{u, v}`` at distance ``t`` from ``u``."""
        ell = self.nbr[u][v]
        t = min(max(t, 0.0), ell)          # D/2 - d(a, u) can overshoot the edge by one ulp
        return f"({self._below(u, v)}:{t!r},{self._below(v, u)}:{ell - t!r});"

    def _below(self, x0: int, f0: int) -> str:
        """The subtree at ``x0`` directed away from ``f0``.  Iterative: a caterpillar of a few
        hundred leaves is deeper than Python's recursion limit."""
        memo: dict[int, str] = {}
        stack = [(x0, f0, False)]
        while stack:
            x, f, done = stack.pop()
            kids = [y for y in self.nbr[x] if y != f]
            if not kids:
                memo[x] = self.label[x]
            elif done:
                memo[x] = "(" + ",".join(f"{memo[y]}:{self.nbr[x][y]!r}" for y in kids) + ")"
            else:
                stack.append((x, f, True))
                stack.extend((y, x, False) for y in kids)
        return memo[x0]


def midpoint_root(newick: str) -> str:
    """``newick`` rooted at the middle of its longest leaf-to-leaf path, every taxon kept."""
    tree = Unrooted.of(newick)
    if len(tree.leaves) < 2:
        raise RootingError("too_few", "a tree needs two leaves to have a midpoint")
    if any(ell < 0 for nb in tree.nbr for ell in nb.values()):
        raise RootingError("negative_length", "the farthest-leaf passes need lengths >= 0")
    a = tree.farthest(min(tree.leaves, key=lambda i: tree.label[i]))[0]
    b, dist, prev = tree.farthest(a)
    D = dist[b]
    if not D > 0:
        raise RootingError("zero_length", "every leaf-to-leaf distance is 0")
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    half = D / 2
    for u, v in zip(path, path[1:]):
        if dist[v] >= half:
            return tree.emit_rooted(u, v, half - dist[u])
    raise AssertionError("unreachable: the path from a ends at distance D")


class SpeciesTree:
    """A rooted species tree, and where it puts the root of any gene tree over a subset of its taxa."""

    def __init__(self, newick: str):
        parent, _, label, children = OG.parse_newick(newick)
        names = [label[i] for i in range(len(parent)) if not children[i]]
        _check_labels(names, "the species tree")
        root = 0
        while len(children[root]) == 1:
            root = children[root][0]
        if len(children[root]) != 2:
            raise ValueError(
                f"its root has out-degree {len(children[root])}, so it is not rooted -- a written "
                f"trifurcation is how unrooted output looks, and reading one as a polytomy would root "
                f"every gene tree wrong.  Root the species tree first.")
        self.names = names                                    # bit i -> label
        self.bit = {x: 1 << i for i, x in enumerate(names)}
        self.taxa = frozenset(names)
        self.children = children
        self.top = root
        self.mask = OG.node_masks(parent, label, children, {x: i for i, x in enumerate(names)})

    @classmethod
    def read(cls, path) -> "SpeciesTree":
        """The one tree in ``path``, which may span lines."""
        trees = [s for s in Path(path).read_text().split(";") if s.strip()]
        if len(trees) != 1:
            raise ValueError(f"expected exactly one tree, found {len(trees)}")
        return cls(trees[0] + ";")

    def root_clades(self, R: int) -> list[int]:
        """The clades the root of this tree restricted to ``R`` splits ``R`` into: descend while
        exactly one child meets ``R``, then read that node's children."""
        v = self.top
        while True:
            hit = [c for c in self.children[v] if self.mask[c] & R]
            if len(hit) != 1:
                return [self.mask[c] & R for c in hit]
            v = hit[0]

    def first_label(self, clade: int) -> str:
        best = None
        while clade:
            low = clade & -clade
            name = self.names[low.bit_length() - 1]
            if best is None or name < best:
                best = name
            clade ^= low
        return best

    def root(self, newick: str) -> tuple[str, str]:
        """``(rooted newick, kind)``, ``kind`` being ``"taxon"`` when one side of the root is a
        single taxon and ``"clade"`` otherwise.  Raises :class:`RootingError` when the gene tree
        carries no split the species tree roots on, and ``ValueError`` on a taxon it does not know."""
        tree = Unrooted.of(newick)
        unknown = sorted(tree.label[i] for i in tree.leaves if tree.label[i] not in self.bit)
        if unknown:
            raise ValueError(f"{len(unknown)} taxa are not in the species tree, e.g. {unknown[:5]}")
        if len(tree.leaves) < 2:
            raise RootingError("too_few", "a tree needs two leaves to be rooted")
        R, edge = tree.splits(self.bit)
        clades = self.root_clades(R)
        if len(clades) == 2:
            usable, reason = [c for c in clades[:1] if c in edge], "split_absent"
        else:
            usable = sorted((c for c in clades if c in edge),
                            key=lambda c: (c.bit_count(), self.first_label(c)))
            reason = "polytomy_no_split"
        if not usable:
            raise RootingError(reason)
        c = usable[0]
        u, v = edge[c]
        kind = "taxon" if min(c.bit_count(), (R ^ c).bit_count()) == 1 else "clade"
        return tree.emit_rooted(u, v, tree.nbr[u][v] / 2), kind
