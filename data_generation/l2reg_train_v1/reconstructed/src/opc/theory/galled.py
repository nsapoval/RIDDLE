"""E3: every binary galled tree at small ``n``, enumerated exhaustively.

``exp-plan-n50-mpl-v0.1`` §E3 asks for "all binary galled trees with k <= 2 (up to isomorphism)" at
n = 5..8, each one's clustering system and triple system, and the quotient.  §0.1 is explicit that
this is the *only* place in the study where enumeration is genuinely exhaustive, which is why it
carries the identifiability question that n = 50 cannot answer.

Why this is not a graph-isomorphism problem
-------------------------------------------

The obvious reading of "up to isomorphism" is a canonical form on labelled DAGs.  It is not needed:
the Hellmuth-Schaller-Stadler correspondence makes **regular networks and clustering systems the
same object**, and the clustering system is already canonical -- a set of subsets of a labelled
taxon set.  So deduplicating on ``C(Psi)`` is deduplicating up to isomorphism, for free, and it is
also the quotient the study's identifiability story is stated over.  Measured, it collapses the
raw placement count by ~3.3x (3,765 placements -> 960 systems at n = 5).

How a galled tree is built
--------------------------

Start from a labelled rooted binary tree ``T`` and pick an ordered pair of non-root nodes
``(d, r)`` whose clusters are **disjoint**.  Subdivide the edge above each -- ``p_d`` above the
donor, ``p_r`` above the hybrid -- and add ``p_d -> p_r``.

Disjointness is doing three jobs at once, and it is worth being explicit that they coincide here.
In a tree, two nodes have disjoint clusters exactly when neither is an ancestor of the other, so it
is simultaneously (i) the condition for the added edge not to create a directed cycle, (ii) A11's
no-shortcut condition -- if ``r`` were below ``d`` then ``d`` would already reach it and the
reticulation would leave no trace in ``C(Psi)`` at all -- and (iii) what makes the result binary.

For ``k = 2`` the two cycles must be **vertex-disjoint** (galled), which is strictly stronger than
level-1 (A7, A28).  Two placements are compatible when neither their occupied base *nodes* nor
their occupied base *edges* meet: a shared node is a shared network node directly, and a shared
edge means one cycle's subdivision point lies on a segment the other traverses, which is the same
thing one step later.  :func:`to_network` plus ``networks.is_galled`` is the independent check, and
``tests/test_galled.py`` runs it.

Encodings
---------

Everything the enumeration emits is a Python ``int`` used as a bitset, because the whole point is
to hash tens of millions of these cheaply:

* a **cluster** is a bitmask over taxa, ``0 <= A < 2^n``;
* a **clustering system** is a bitmask over the ``2^n`` possible clusters (256 bits at n = 8);
* a **triple system** is a bitmask over ``3 * C(n,3)`` slots, ordered ``(ab|c, ac|b, bc|a)`` with
  ``a < b < c`` -- the same convention :meth:`opc.mpl.triples.TripleCounts.as_rows` uses, so a
  triple system and a rho row index the same way.

:func:`triple_bits` precomputes, for every cluster mask, the triples that mask resolves.  A tree's
triple system is then an ``OR`` over its clusters, and a network's is an ``OR`` over its displayed
trees -- which is what makes the inner loop table lookups rather than tree walks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "BaseTree",
    "Cycle",
    "GalledInstance",
    "triple_bits",
    "triple_slots",
    "rooted_binary_trees",
    "n_rooted_binary_trees",
    "n_trees_available",
    "meets_table",
    "widen",
    "RETICULATE",
    "cycles_of",
    "compatible",
    "galled_placements",
    "instance",
    "enumerate_galled",
    "to_network",
    "decode_system",
]


# --------------------------------------------------------------------------- triple encoding


@lru_cache(maxsize=None)
def triple_slots(n_taxa: int) -> tuple[tuple[int, int, int], ...]:
    """The ``C(n,3)`` ascending taxon triples, lexicographic -- ``triple_index``'s own order."""
    return tuple(itertools.combinations(range(n_taxa), 3))


@lru_cache(maxsize=None)
def triple_bits(n_taxa: int) -> tuple[int, ...]:
    """``table[A]`` = the triple-system bits a single cluster ``A`` resolves.

    ``A`` resolves ``ab|c`` exactly when it holds two of the triple's taxa and not the third.  A
    rooted tree displays ``ab|c`` iff some node's cluster does, so ``OR`` over a cluster set is the
    tree's whole triple system and no tree walk is needed.
    """
    slots = triple_slots(n_taxa)
    table = [0] * (1 << n_taxa)
    for rank, (a, b, c) in enumerate(slots):
        ba, bb, bc = 1 << a, 1 << b, 1 << c
        for mask in range(1 << n_taxa):
            has = (bool(mask & ba), bool(mask & bb), bool(mask & bc))
            if has == (True, True, False):
                table[mask] |= 1 << (3 * rank)
            elif has == (True, False, True):
                table[mask] |= 1 << (3 * rank + 1)
            elif has == (False, True, True):
                table[mask] |= 1 << (3 * rank + 2)
    return tuple(table)


@lru_cache(maxsize=None)
def meets_table(n_taxa: int) -> tuple[int, ...]:
    """``table[A]`` = a bitmask over triple *ranks* of the triples ``A`` meets.

    Used for the shape-class quotient below: a reticulation survives the restriction to a triple
    exactly when its hybrid's cluster meets that triple.
    """
    slots = triple_slots(n_taxa)
    table = [0] * (1 << n_taxa)
    for rank, (a, b, c) in enumerate(slots):
        touch = (1 << a) | (1 << b) | (1 << c)
        for mask in range(1 << n_taxa):
            if mask & touch:
                table[mask] |= 1 << rank
    return tuple(table)


@lru_cache(maxsize=None)
def _expand2() -> tuple[int, ...]:
    """One byte of a triple-rank mask widened to the 2-bit-per-triple layout of a class vector."""
    return tuple(
        sum(0b11 << (2 * i) for i in range(8) if b >> i & 1) for b in range(256)
    )


def widen(mask: int, n_slots: int) -> int:
    """A triple-rank bitmask as a 2-bit-per-triple mask of all-ones."""
    table = _expand2()
    out = shift = 0
    while mask:
        out |= table[mask & 0xFF] << (2 * shift)
        mask >>= 8
        shift += 8
    return out & ((1 << (2 * n_slots)) - 1)


#: Class-vector code for "this triple restricts to a reticulate trinet".  The other three codes are
#: the resolution indices 0, 1, 2, matching the triple-system slot order.
RETICULATE = 3


def decode_system(bits: int, n_taxa: int) -> frozenset[int]:
    """A clustering-system bitmask back to the set of cluster masks it stands for."""
    out = set()
    m = 0
    while bits:
        if bits & 1:
            out.add(m)
        bits >>= 1
        m += 1
    if any(c >= (1 << n_taxa) for c in out):
        raise ValueError("clustering system carries a cluster outside the taxon set")
    return frozenset(out)


# --------------------------------------------------------------------------------- base trees


@dataclass(frozen=True)
class BaseTree:
    """A labelled rooted binary tree.  Leaves are nodes ``0..n-1``, taxon ``i`` at node ``i``."""

    n_taxa: int
    par: tuple[int, ...]
    kids: tuple[tuple[int, ...], ...]
    clus: tuple[int, ...]
    root: int
    depth: tuple[int, ...]
    #: The tree's own triple system, the ``k = 0`` answer and one of the displayed trees of every
    #: ``k >= 1`` network built on it.
    triples: int
    #: The tree's own clustering system, as a bitmask over cluster masks.
    system: int
    #: The resolution of every triple, two bits each, in triple-rank order.  A binary tree resolves
    #: each triple exactly once, so this is a re-encoding of :attr:`triples` and not new
    #: information -- but it is the encoding the class vector is built out of.
    resolutions: int

    @property
    def n_nodes(self) -> int:
        return len(self.par)


@lru_cache(maxsize=None)
def _shapes(leaves: tuple[int, ...]) -> tuple:
    """Every labelled rooted binary tree on ``leaves``, as nested pairs.

    Each tree is produced exactly once: the split is always taken with ``min(leaves)`` on the left,
    which breaks the left/right symmetry that would otherwise double-count.
    """
    if len(leaves) == 1:
        return (leaves[0],)
    first, rest = leaves[0], leaves[1:]
    out = []
    for size in range(len(rest)):
        for sub in itertools.combinations(rest, size):
            chosen = set(sub)
            left = (first,) + sub
            right = tuple(x for x in rest if x not in chosen)
            for left_shape in _shapes(left):
                for right_shape in _shapes(right):
                    out.append((left_shape, right_shape))
    return tuple(out)


def n_rooted_binary_trees(n_taxa: int) -> int:
    """``(2n-3)!!`` -- what :func:`rooted_binary_trees` must yield, as an independent count."""
    total = 1
    for i in range(3, 2 * n_taxa - 2, 2):
        total *= i
    return total


def _build(shape, n_taxa: int) -> BaseTree:
    n_nodes = 2 * n_taxa - 1
    par = [-1] * n_nodes
    kids: list[tuple[int, ...]] = [()] * n_nodes
    clus = [0] * n_nodes
    counter = [n_taxa]

    def rec(x) -> int:
        if not isinstance(x, tuple):
            clus[x] = 1 << x
            return x
        v = counter[0]
        counter[0] += 1
        a, b = rec(x[0]), rec(x[1])
        kids[v] = (a, b)
        par[a] = par[b] = v
        clus[v] = clus[a] | clus[b]
        return v

    root = rec(shape)
    depth = [0] * n_nodes
    order = [root]
    while order:
        v = order.pop()
        for c in kids[v]:
            depth[c] = depth[v] + 1
            order.append(c)
    table = triple_bits(n_taxa)
    triples = 0
    system = 0
    for m in clus:
        triples |= table[m]
        system |= 1 << m
    resolutions = 0
    for rank in range(len(triple_slots(n_taxa))):
        three = (triples >> (3 * rank)) & 0b111
        resolutions |= (0 if three & 1 else 1 if three & 2 else 2) << (2 * rank)
    return BaseTree(
        n_taxa=n_taxa, par=tuple(par), kids=tuple(kids), clus=tuple(clus), root=root,
        depth=tuple(depth), triples=triples, system=system, resolutions=resolutions,
    )


def rooted_binary_trees(n_taxa: int, lo: int = 0, hi: int | None = None):
    """Every labelled rooted binary tree on ``n_taxa`` leaves, as :class:`BaseTree`.

    ``lo``/``hi`` take a contiguous slice of the (deterministic) enumeration order, which is how
    ``scripts/e3_fibers.py`` splits the work across a pool: the shape list is memoised per worker,
    so a slice costs one build of the list and then only the trees asked for.
    """
    shapes = _shapes(tuple(range(n_taxa)))
    for shape in shapes[lo:len(shapes) if hi is None else hi]:
        yield _build(shape, n_taxa)


def n_trees_available(n_taxa: int) -> int:
    """How many base trees :func:`rooted_binary_trees` will yield -- the slicing bound."""
    return len(_shapes(tuple(range(n_taxa))))


# ---------------------------------------------------------------------------------- placements


@dataclass(frozen=True)
class Cycle:
    """One reticulation placed on a base tree: donor ``d``, hybrid ``r``, apex ``z = LCA(d, r)``.

    ``donor_path`` and ``hybrid_path`` are the *strict* ancestors of ``d`` and ``r`` that sit below
    ``z``, bottom-up.  In the finished network the cycle runs ``p_d`` -> ``donor_path`` -> ``z`` ->
    ``hybrid_path`` -> ``p_r``, so :attr:`size` counts the same tree edges
    :func:`opc.simulate.networks.cycle_size` does.
    """

    donor: int
    hybrid: int
    apex: int
    donor_path: tuple[int, ...]
    hybrid_path: tuple[int, ...]
    #: Base nodes the cycle passes through, apex included -- a bitmask over base node ids.
    nodes: int
    #: Base edges the cycle traverses, named by their lower endpoint -- likewise a bitmask.
    #:
    #: Bitmasks rather than sets because :func:`compatible` is the enumeration's hot loop: at
    #: n = 8 there are ~115 placements per tree, so ~6,600 compatibility tests per tree and ~890 M
    #: over the grid.  Two integer ``AND``s is the difference between minutes and hours.
    edges: int

    @property
    def size(self) -> int:
        return 2 + len(self.donor_path) + len(self.hybrid_path)


def _path_to(tree: BaseTree, v: int, stop: int) -> tuple[int, ...]:
    out = []
    v = tree.par[v]
    while v != stop:
        if v < 0:
            raise RuntimeError("apex is not an ancestor")
        out.append(v)
        v = tree.par[v]
    return tuple(out)


def _lca(tree: BaseTree, a: int, b: int) -> int:
    da, db = tree.depth[a], tree.depth[b]
    while da > db:
        a = tree.par[a]
        da -= 1
    while db > da:
        b = tree.par[b]
        db -= 1
    while a != b:
        a, b = tree.par[a], tree.par[b]
    return a


def cycles_of(tree: BaseTree, *, allow_shortcuts: bool = False) -> list[Cycle]:
    """Every legal single-reticulation placement on ``tree``.

    Ordered pairs: ``(d, r)`` and ``(r, d)`` are different networks.  Two conditions, and the
    second is **A11**, the most consequential finding of the previous study.

    ``clus[d] & clus[r] == 0`` -- disjoint clusters.  In a tree that is incomparability, so it is
    also what stops the added edge closing a directed cycle.

    ``tree_parent(r) != z`` -- the hybrid must sit **strictly below** the apex.  Otherwise ``z``
    reaches the hybrid both directly and through the donor, so ``z -> p_r`` is a shortcut edge, the
    network is not regular, and Hellmuth et al.'s correspondence -- the thing E3's whole quotient
    rests on -- does not apply to it.  Concretely the cycle then contributes **no overlapping
    pair**: ``c(p_d) = c(d) | c(r)`` is nested inside ``c(z)`` rather than overlapping it, so the
    reticulation is invisible to every cluster-based method.  In the degenerate sub-case where the
    donor is also a child of ``z`` -- the cycle of size 2 -- the clustering system comes out
    **identical to the base tree's**, which is A11 at its starkest.

    This is exactly the rule ``networks._place_from_pairs`` applies under
    ``allow_shortcuts=False``, arrived at from the other direction: there it was measured (only
    33-53% of unconstrained reticulations left any trace), here it is forced by the enumeration
    having to be an enumeration *of regular networks*.  The flag is kept so the excluded set stays
    countable rather than invisible.
    """
    out = []
    nodes = [v for v in range(tree.n_nodes) if v != tree.root]
    for d in nodes:
        cd = tree.clus[d]
        for r in nodes:
            if d == r or cd & tree.clus[r]:
                continue
            z = _lca(tree, d, r)
            if not allow_shortcuts and tree.par[r] == z:
                continue
            dp = _path_to(tree, d, z)
            hp = _path_to(tree, r, z)
            node_bits = 1 << z
            edge_bits = (1 << d) | (1 << r)
            for v in dp + hp:
                node_bits |= 1 << v
                edge_bits |= 1 << v
            out.append(Cycle(
                donor=d, hybrid=r, apex=z, donor_path=dp, hybrid_path=hp,
                nodes=node_bits, edges=edge_bits,
            ))
    return out


def compatible(a: Cycle, b: Cycle) -> bool:
    """Are two placements vertex-disjoint in the finished network -- i.e. still a galled tree?

    Two conditions, and the second is the one that is easy to forget.  A shared base **node** is a
    shared network node outright.  A shared base **edge** means one cycle's subdivision point sits
    on a segment the other cycle traverses whole, so that subdivision point ends up on both cycles
    -- level-1 (edge-disjointness, A7) would allow it, galled does not.
    """
    return not (a.nodes & b.nodes) and not (a.edges & b.edges)


def _compatible_combos(cycles: list[Cycle], k: int):
    """Indices of every pairwise-compatible ``k``-subset.  ``k <= 2`` takes the fast path."""
    if k == 1:
        yield from ((i,) for i in range(len(cycles)))
        return
    if k == 2:
        for i, a in enumerate(cycles):
            an, ae = a.nodes, a.edges
            for j in range(i + 1, len(cycles)):
                b = cycles[j]
                if not (an & b.nodes) and not (ae & b.edges):
                    yield (i, j)
        return
    for combo in itertools.combinations(range(len(cycles)), k):
        if all(compatible(cycles[x], cycles[y]) for x, y in itertools.combinations(combo, 2)):
            yield combo


def galled_placements(tree: BaseTree, k: int, cycles: list[Cycle] | None = None):
    """Every unordered ``k``-tuple of pairwise compatible placements on ``tree``."""
    if k == 0:
        yield ()
        return
    cycles = cycles_of(tree) if cycles is None else cycles
    for combo in _compatible_combos(cycles, k):
        yield tuple(cycles[i] for i in combo)


# ---------------------------------------------------------------- the network, and what it emits


def _dag(tree: BaseTree, cycles) -> tuple[list[int], list[list[int]], list[tuple[int, int]]]:
    """``(tree_par, tree_kids, [(p_d, p_r)])`` for the network the placements describe.

    Subdivision points are appended after the base nodes, so base ids never move -- which is what
    lets the covariates be read off the base tree afterwards.
    """
    tree_par = list(tree.par)
    tree_kids = [list(c) for c in tree.kids]
    marks = []
    for cy in cycles:
        pd = len(tree_par)
        tree_par.append(tree_par[cy.donor])
        tree_kids.append([cy.donor])
        tree_kids[tree_par[pd]][tree_kids[tree_par[pd]].index(cy.donor)] = pd
        tree_par[cy.donor] = pd

        pr = len(tree_par)
        tree_par.append(tree_par[cy.hybrid])
        tree_kids.append([cy.hybrid])
        tree_kids[tree_par[pr]][tree_kids[tree_par[pr]].index(cy.hybrid)] = pr
        tree_par[cy.hybrid] = pr

        marks.append((pd, pr))
    return tree_par, tree_kids, marks


def _clusters_of_tree(sel_par: list[int], n_nodes: int, root: int, leaf_mask) -> list[int]:
    """Node clusters of the tree ``sel_par`` describes, bottom-up.  ``sel_par[root] == -1``."""
    kids: list[list[int]] = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        p = sel_par[v]
        if p >= 0:
            kids[p].append(v)
    clus = [0] * n_nodes
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            m = leaf_mask(v)
            for c in kids[v]:
                m |= clus[c]
            clus[v] = m
            continue
        stack.append((v, True))
        for c in kids[v]:
            stack.append((c, False))
    return clus


@dataclass(frozen=True)
class GalledInstance:
    """One enumerated galled tree, reduced to what E3 quotients over."""

    n_taxa: int
    k: int
    #: ``C(Psi)`` -- the *full* hardwired clustering system, singletons and taxon set included
    #: (A9: the closedness claim is about the full system), as a bitmask over cluster masks.
    system: int
    #: ``R(Psi)`` -- the union over displayed trees of their rooted triples.
    triples: int
    #: The **shape-class vector**: two bits per triple, holding either the resolution the triple's
    #: trinet forces or :data:`RETICULATE`.  This is the quotient E4's relaxation works over -- its
    #: objective is ``sum_tau W(tau, class)``, so two networks with equal class vectors are scored
    #: identically by the profiled ILP whatever the data are.  Strictly coarser than
    #: :attr:`triples`, and therefore the place to look once ``C -> R`` turns out to be injective.
    classes: int
    #: Structural covariates, for characterising whatever the non-singleton fibers have in common.
    cycle_sizes: tuple[int, ...]
    apex_depths: tuple[int, ...]
    hybrid_sizes: tuple[int, ...]
    #: Reticulations whose hybrid cluster is a single leaf -- A38's case, and the one the *regular*
    #: version of a network produces whenever a hybrid species has one extant descendant.
    singleton_hybrids: int
    #: Pairs of reticulations whose hybrids are siblings in the base tree.  The plan names
    #: "sibling-hybrid configurations" as a candidate explanation for non-singleton fibers; a
    #: donor/hybrid pair of siblings cannot occur at all, because that is the size-2 cycle A11
    #: excludes.
    sibling_hybrids: int


def instance(tree: BaseTree, cycles) -> GalledInstance:
    """Build one galled tree's clustering system, triple system and covariates.

    Both systems come from the same DAG rather than from closed-form path arithmetic on the base
    tree.  The closed form is correct for ``k = 1`` and its ``k = 2`` composition is only correct
    *because* the cycles are vertex-disjoint -- an argument this module would then be resting on
    rather than checking.  ``tests/test_galled.py`` keeps the closed form as the independent
    implementation and asserts the two agree.
    """
    n = tree.n_taxa
    table = triple_bits(n)
    if not cycles:
        return GalledInstance(
            n_taxa=n, k=0, system=tree.system, triples=tree.triples,
            classes=tree.resolutions,
            cycle_sizes=(), apex_depths=(), hybrid_sizes=(),
            singleton_hybrids=0, sibling_hybrids=0,
        )

    tree_par, tree_kids, marks = _dag(tree, cycles)
    n_nodes = len(tree_par)
    clus = list(tree.clus) + [0] * (n_nodes - tree.n_nodes)

    def leaf_mask(v: int) -> int:
        return clus[v] if v < n else 0

    # C(Psi): the hardwired system, so every hybrid contributes through *both* its parents.
    net_par: list[list[int]] = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        if tree_par[v] >= 0:
            net_par[v].append(tree_par[v])
    for pd, pr in marks:
        net_par[pr].append(pd)
    net_kids: list[list[int]] = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        for p in net_par[v]:
            net_kids[p].append(v)
    hard = [0] * n_nodes
    stack = [(tree.root, False)]
    seen = set()
    while stack:
        v, done = stack.pop()
        if done:
            m = leaf_mask(v)
            for c in net_kids[v]:
                m |= hard[c]
            hard[v] = m
            continue
        if v in seen:
            continue
        seen.add(v)
        stack.append((v, True))
        for c in net_kids[v]:
            stack.append((c, False))
    system = 0
    for m in hard:
        system |= 1 << m

    # R(Psi): the union over the 2^k displayed trees.  Taking the *tree* parent at every hybrid
    # deletes every horizontal edge, so that displayed tree is the base tree itself and its
    # triples are already on hand -- one traversal saved of the two at k = 1, one of four at k = 2.
    triples = tree.triples
    for choice in itertools.product((False, True), repeat=len(marks)):
        if not any(choice):
            continue
        sel = list(tree_par)
        for take_donor, (pd, pr) in zip(choice, marks, strict=True):
            if take_donor:
                sel[pr] = pd
        for m in _clusters_of_tree(sel, n_nodes, tree.root, leaf_mask):
            triples |= table[m]

    # The shape-class vector.  ``restrict_to_triple`` keeps every node ancestral to one of the
    # three leaves and follows *all* in-edges, so the hybrid ``p_r`` survives exactly when
    # ``c(r)`` meets the triple -- and a surviving hybrid keeps both parents, so it is never
    # suppressed.  "Reticulate trinet" is therefore precisely "the hybrid's cluster meets the
    # triple", with no traversal needed.
    meets = meets_table(n)
    ret_ranks = 0
    for cy in cycles:
        ret_ranks |= meets[tree.clus[cy.hybrid]]
    wide = widen(ret_ranks, len(triple_slots(n)))
    classes = (tree.resolutions & ~wide) | wide  # RETICULATE is 0b11, which is what ``wide`` holds

    return GalledInstance(
        n_taxa=n, k=len(cycles), system=system, triples=triples, classes=classes,
        cycle_sizes=tuple(cy.size for cy in cycles),
        apex_depths=tuple(tree.depth[cy.apex] for cy in cycles),
        hybrid_sizes=tuple(bin(tree.clus[cy.hybrid]).count("1") for cy in cycles),
        singleton_hybrids=sum(tree.clus[cy.hybrid].bit_count() == 1 for cy in cycles),
        sibling_hybrids=sum(tree.par[a.hybrid] == tree.par[b.hybrid]
                            for a, b in itertools.combinations(cycles, 2)),
    )


def enumerate_galled(n_taxa: int, k_max: int, *, trees=None):
    """Every binary galled tree on ``n_taxa`` labelled leaves with ``k <= k_max``.

    Yields one :class:`GalledInstance` per (tree, placement), i.e. **with** duplicates: distinct
    placements often realise the same clustering system, and how often is itself a number E3
    reports.  Deduplicate on :attr:`GalledInstance.system` downstream.
    """
    source = rooted_binary_trees(n_taxa) if trees is None else trees
    for tree in source:
        cycles = cycles_of(tree)
        for k in range(k_max + 1):
            for combo in galled_placements(tree, k, cycles):
                yield instance(tree, combo)


# ------------------------------------------------------------------- the independent check path


def to_network(tree: BaseTree, cycles, labels=None):
    """The same galled tree as an :class:`opc.simulate.networks.Network`.

    Only the tests and the probe use this -- it is the *independent* path the fast enumeration is
    checked against (clusters, ``is_galled``, ``level``, and the displayed trees via
    ``msnc3.resolve_one_reticulation``), not part of the inner loop.

    Node times are the longest-path depth in the DAG, which is monotone along every edge including
    the horizontal one.  The generator gives the horizontal edge length zero; here it gets a
    positive length instead, deliberately.  Nothing structural depends on it, ``msnc3`` walks a
    post-order rather than a time sweep (A30), and a strictly positive length keeps every branch
    length a usable optimiser start.
    """
    from opc.simulate import networks as N

    tree_par, tree_kids, marks = _dag(tree, cycles)
    n_nodes = len(tree_par)
    donor_of = {pr: pd for pd, pr in marks}
    horiz_child = {pd: pr for pd, pr in marks}

    # Longest path from the root, so every edge -- horizontal included -- has positive length.
    rank = [0] * n_nodes
    order, seen = [], set()
    stack = [(tree.root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        if v in seen:
            continue
        seen.add(v)
        stack.append((v, True))
        for c in tree_kids[v] + ([horiz_child[v]] if v in horiz_child else []):
            stack.append((c, False))
    for v in reversed(order):  # root first
        for c in tree_kids[v] + ([horiz_child[v]] if v in horiz_child else []):
            rank[c] = max(rank[c], rank[v] + 1)

    net = N.Network(
        n_taxa=tree.n_taxa,
        labels=list(labels) if labels is not None else [f"t{i + 1}" for i in range(tree.n_taxa)],
        time=[float(x) for x in rank],
        tree_parent=list(tree_par),
        tree_children=[list(c) for c in tree_kids],
        leaf_taxon={v: v for v in range(tree.n_taxa)},
        horiz_child=horiz_child,
        donor_of=donor_of,
        retics=[],
        root=tree.root,
    )
    for i, (pd, pr) in enumerate(marks):
        net.retics.append(N.Reticulation(hybrid=pr, donor=pd, order=i, time=net.time[pr]))
    N.recompute_consumed(net)
    return net
