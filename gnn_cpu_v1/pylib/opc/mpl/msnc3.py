"""P(t | Psi, gamma) for a rooted triple, computed exactly -- the reference semantics for I1.

``exp-plan-n50-mpl-v0.1`` §2 I1 specifies ``P(t | Psi, gamma)`` as "the gamma-weighted mixture over
displayed trees restricted to the triple".  **That mixture is not the MSNC probability**, and the
discrepancy is not a rounding matter, so both are implemented here and the difference is measured
rather than assumed (:func:`triple_probabilities_mixture`).

Why the mixture is not the MSNC
-------------------------------

The mixture assumes one parental tree is drawn and *all* lineages follow it.  Under the
multispecies network coalescent each lineage arriving at a hybrid node chooses independently.  With
one allele per species the two agree only while at most one lineage ever reaches a given hybrid.
As soon as two of the triple's taxa sit below the same reticulation:

* they may fail to coalesce below the hybrid -- probability ``e^{-x}`` over the branch below it;
* they then choose independently, and with probability ``2*gamma*(1-gamma)`` take *different* sides
  of the cycle;
* on different sides they cannot coalesce until the cycle's apex, so the pair is pushed apart.

The mixture has no state in which that happens.  It therefore **understates ILS for pairs below a
reticulation**, most strongly at ``gamma = 0.5`` -- which is the stratum every pre-registered
threshold in the plan is stated at.

How the exact value is computed
-------------------------------

A post-order pass over the sub-DAG spanned by the three taxa's ancestors -- children before
parents, so every lineage that will ever occupy an edge arrives at its foot at once and then
traverses the whole edge.  That makes the coalescent within an edge a single step of the edge's own
length, and it means the computation **never refers to node times**: branch lengths can be varied
freely by an optimiser without the network staying time-consistent or ultrametric.

The state is the **joint** placement of the surviving lineage blocks: a set of ``(edge, block)``
pairs plus the topology already fixed by the first coalescence.  Carrying the *joint* placement is
the whole point.  The two sides of a reticulation cycle are correlated -- one split produced them
and they rejoin at the apex -- so propagating a marginal per side and multiplying them back
together at the apex is wrong, and wrong in a way that can put one taxon on both sides at once.

Transitions:

*edge*          the ``k``-coalescent over its length, ``k <= 3``.
*tree node*     blocks on the child edges move onto the single parent edge.
*hybrid node*   each block chooses the donor edge with probability ``gamma``, the tree edge with
                ``1 - gamma``, **independently per block**.  This is the step the mixture misses.
*root*          an infinite branch: everything coalesces, three-way ties broken uniformly.

This is Yu, Degnan & Nakhleh's coalescent-history sum, restricted to three lineages and therefore
written out rather than enumerated.

Cost
----

``O(|V| * |states|)`` per triple with ``|states|`` in the tens.  It is the reference: correct, and
far too slow to sit inside an optimiser at ``C(50,3) = 19,600`` triples.  The fast path evaluates
:mod:`opc.mpl.trinets`' shape catalog instead, and is checked against this.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from opc.simulate import networks as N

__all__ = [
    "TOPOLOGIES",
    "coalescent_two",
    "coalescent_three",
    "ancestors",
    "triple_probabilities",
    "triple_probabilities_mixture",
    "all_triple_probabilities",
    "resolve_one_reticulation",
]

#: The three rooted topologies of a triple ``(a, b, c)`` with ``a < b < c``, in the order every
#: module in this package uses: ``ab|c``, ``ac|b``, ``bc|a``.
TOPOLOGIES = ((0, 1), (0, 2), (1, 2))
_TOPO_INDEX = {frozenset(p): i for i, p in enumerate(TOPOLOGIES)}

#: Parent slot a leaf's lineage is seeded on before the post-order moves it onto a real edge.
#: A sentinel rather than the leaf's tree parent, because a leaf may be a *hybrid* and then has two
#: in-edges and a gamma to choose between them.
SEED = "seed"


def coalescent_two(x: float) -> float:
    """P(two lineages coalesce within a branch of ``x`` coalescent units)."""
    return -math.expm1(-x)


def coalescent_three(x: float) -> tuple[float, float, float]:
    """``(p3, p2, p1)`` for three lineages entering a branch of length ``x``.

    ``p3`` none coalesce, ``p2`` exactly one coalescence, ``p1`` both.  Rates are ``C(k,2)``, so 3
    then 1, and ``p2 = (3/2) e^{-x}(1 - e^{-2x})`` is the convolution.  ``p1`` is taken as the
    remainder so the three sum to exactly 1 in floating point.
    """
    e1, e3 = math.exp(-x), math.exp(-3.0 * x)
    p2 = 1.5 * e1 * -math.expm1(-2.0 * x)
    return e3, p2, max(0.0, 1.0 - e3 - p2)


def ancestors(net: N.Network, v: int) -> set[int]:
    seen, stack = set(), [v]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(net.net_parents(u))
    return seen


# ------------------------------------------------------------------------------------- the DP

#: A state: ``(placement, topology)``.  ``placement`` is a frozenset of
#: ``((child, parent), block)``
#: -- the edge each surviving lineage block sits on -- and ``topology`` is the index into
#: :data:`TOPOLOGIES` fixed by the first coalescence, or ``None`` while three lineages survive.


def _coalesce(dist: dict, durations: dict) -> dict:
    """Run the k-coalescent inside each of ``durations``' edges, over that edge's own length."""
    out: dict = {}

    def add(state, p: float) -> None:
        if p > 0.0:
            out[state] = out.get(state, 0.0) + p

    for (placement, topo), w in dist.items():
        by_edge: dict = {}
        for edge, block in placement:
            if edge in durations:
                by_edge.setdefault(edge, []).append(block)
        # With at most three lineages only one edge can hold more than one of them, and only one
        # of ``durations``' edges can be that one.
        busy = [e for e, bs in by_edge.items() if len(bs) >= 2 and durations[e] > 0.0]
        if not busy:
            add((placement, topo), w)
            continue
        (edge,) = busy
        dt = durations[edge]
        blocks = by_edge[edge]
        others = frozenset(x for x in placement if x[0] != edge)
        if len(blocks) == 2:
            pc = coalescent_two(dt)
            add((placement, topo), w * (1.0 - pc))
            merged = frozenset(blocks[0] | blocks[1])
            t = topo if topo is not None else _TOPO_INDEX[merged]
            add((others | {(edge, merged)}, t), w * pc)
            continue
        p3, p2, p1 = coalescent_three(dt)
        add((placement, topo), w * p3)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            merged = frozenset(blocks[i] | blocks[j])
            (rest,) = (blocks[m] for m in range(3) if m not in (i, j))
            t = _TOPO_INDEX[merged]  # three blocks means nothing has merged yet
            add((others | {(edge, merged), (edge, rest)}, t), w * p2 / 3.0)
            add((others | {(edge, frozenset({0, 1, 2}))}, t), w * p1 / 3.0)
    return out


def _move(dist: dict, from_edges: set, to_edge) -> dict:
    """A tree node: everything on the child edges continues up the single parent edge."""
    out: dict = {}
    for (placement, topo), w in dist.items():
        moved = frozenset(
            (to_edge, block) if edge in from_edges else (edge, block)
            for edge, block in placement
        )
        out[(moved, topo)] = out.get((moved, topo), 0.0) + w
    return out


def _split(dist: dict, from_edges: set, donor_edge, tree_edge, gamma: float) -> dict:
    """A hybrid node: each arriving block chooses a parent independently."""
    out: dict = {}
    for (placement, topo), w in dist.items():
        here = [(e, b) for e, b in placement if e in from_edges]
        rest = frozenset(x for x in placement if x[0] not in from_edges)
        if not here:
            out[(rest, topo)] = out.get((rest, topo), 0.0) + w
            continue
        for choice in product((donor_edge, tree_edge), repeat=len(here)):
            p = w
            for target in choice:
                p *= gamma if target is donor_edge else 1.0 - gamma
            if p == 0.0:
                continue
            moved = rest | {
                (target, b) for (_, b), target in zip(here, choice, strict=True)
            }
            out[(moved, topo)] = out.get((moved, topo), 0.0) + p
    return out


def _finish(dist: dict) -> np.ndarray:
    """The root's branch is infinite: everything coalesces, three-way ties uniform."""
    out = np.zeros(3)
    for (placement, topo), w in dist.items():
        blocks = [b for _, b in placement]
        if topo is not None:
            out[topo] += w
        elif len(blocks) == 3:
            out += w / 3.0
        elif len(blocks) == 2:
            out[_TOPO_INDEX[frozenset(blocks[0] | blocks[1])]] += w
        else:
            raise RuntimeError("a lone block reached the root with no topology")
    return out


def triple_probabilities(net: N.Network, taxa, *, lengths=None, gammas=None) -> np.ndarray:
    """Exact ``(P(ab|c), P(ac|b), P(bc|a))`` under the MSNC, for ``a < b < c`` in ``taxa``.

    ``lengths`` overrides individual edge lengths, keyed by ``(child, parent)``; ``gammas``
    overrides the donor-edge inheritance probability per hybrid node.  Both exist because
    :class:`~opc.simulate.networks.Network` stores lengths as differences of node *times* and
    carries a single scalar ``gamma`` for the whole network -- fine for simulation, useless as an
    optimiser's parameter vector, which needs one length per internal edge and one gamma per
    reticulation, unconstrained.
    """
    order = tuple(sorted(taxa))
    leaf_node = {t: v for v, t in net.leaf_taxon.items() if t in order}
    if len(leaf_node) != 3:
        raise ValueError(f"taxa {taxa} are not three distinct leaves of this network")

    keep: set[int] = set()
    for t in order:
        keep |= ancestors(net, leaf_node[t])

    def length(child: int, parent: int) -> float:
        if lengths is not None and (child, parent) in lengths:
            return float(lengths[(child, parent)])
        return net.edge_length(parent, child)

    def gamma_of(hybrid: int) -> float:
        if gammas is not None and hybrid in gammas:
            return float(gammas[hybrid])
        return net.gamma_of(hybrid)

    # All three lineages start *at* their leaf, on a sentinel edge that the loop then moves them
    # off.  Two reasons it is done here and in this form.
    #
    # Seeding here rather than when the loop reaches each leaf is not a style choice: a post-order
    # interleaves leaves with the internal nodes of *other* subtrees, so seeding inside the loop
    # would overwrite a distribution that already had several states -- silently, and with the
    # signature of a network in which nothing ever coalesces.
    #
    # The sentinel rather than the pendant edge is what makes a **hybrid leaf** work.  A hybrid
    # species with a single extant descendant is the common biological case, and it is what the
    # *regular* version of a network produces whenever a hybrid's cluster is a singleton
    # (:mod:`opc.solve.closure`).  Its lineage must still choose a parent -- gamma decides which
    # side of the cycle it travels, and that changes the triple probability -- so a leaf cannot be
    # assumed to have exactly one in-edge.
    dist: dict = {
        (frozenset(((leaf_node[t], SEED), frozenset({i})) for i, t in enumerate(order)), None): 1.0
    }
    # Children before parents, so every lineage that will ever be on an edge is on it before the
    # edge's coalescent runs.  ``_postorder`` emits a hybrid before its donor -- the hybrid is a
    # net child of the donor -- which is what the zero-length horizontal edge needs; a time-sorted
    # order would leave those two tied, and with the donor first the donor-side lineages would be
    # stranded on an edge whose head had already been processed.
    for v in N._postorder(net):  # noqa: SLF001
        if v not in keep:
            continue
        if net.is_leaf(v):
            # A plain leaf just hands its lineage up; a hybrid leaf has to split first, so it
            # cannot be skipped.  Its "children" are the sentinel the seed put the lineage on.
            if not net.is_hybrid(v):
                dist = _move(dist, {(v, SEED)}, (v, net.tree_parent[v]))
                continue
            from_edges = {(v, SEED)}
        else:
            from_edges = {(c, v) for c in net.net_children(v) if c in keep}
        if net.tree_parent[v] < 0:
            return _finish(_gather(dist, from_edges))
        if net.is_hybrid(v):
            donor, tree = (v, net.donor_of[v]), (v, net.tree_parent[v])
            dist = _split(dist, from_edges, donor, tree, gamma_of(v))
            dist = _coalesce(dist, {donor: length(*donor), tree: length(*tree)})
        else:
            up = (v, net.tree_parent[v])
            dist = _move(dist, from_edges, up)
            dist = _coalesce(dist, {up: length(*up)})
    raise RuntimeError("the post-order pass never reached the root")


def _gather(dist: dict, from_edges: set) -> dict:
    """Everything on ``from_edges`` arrives at one node; used only at the root."""
    out: dict = {}
    for (placement, topo), w in dist.items():
        moved = frozenset(
            (None, block) if edge in from_edges else (edge, block) for edge, block in placement
        )
        out[(moved, topo)] = out.get((moved, topo), 0.0) + w
    return out


def all_triple_probabilities(net: N.Network, index=None) -> np.ndarray:
    """``P[i, :]`` for every ascending triple.  The slow, exact reference path."""
    from opc.mpl.triples import triple_index

    idx = triple_index(net.n_taxa) if index is None else index
    return np.stack([triple_probabilities(net, tuple(row)) for row in idx])


# ------------------------------------------------------------------ the mixture, for comparison


def resolve_one_reticulation(net: N.Network, index: int, take_donor: bool) -> N.Network:
    """Resolve a single reticulation, leaving the others in place.

    :func:`opc.simulate.networks.displayed_tree` resolves *every* reticulation the same way, which
    cannot express the ``2^k`` mixed parental trees the mixture sums over.
    """
    out = net.copy()
    ret = out.retics[index]
    h, d = ret.hybrid, ret.donor
    del out.horiz_child[d]
    del out.donor_of[h]
    if take_donor:
        tp = out.tree_parent[h]
        out.tree_children[tp].remove(h)
        out.tree_parent[h] = d
        out.tree_children[d].append(h)
        N._suppress_degree2(out, h)  # noqa: SLF001
        if len(out.tree_children[tp]) == 1:
            if out.tree_parent[tp] >= 0:
                N._suppress_degree2(out, tp)  # noqa: SLF001
            else:
                (new_root,) = out.tree_children[tp]
                out.tree_parent[new_root] = -1
                out.tree_children[tp] = []
                out.root = new_root
    else:
        N._suppress_degree2(out, h)  # noqa: SLF001
        N._suppress_degree2(out, d)  # noqa: SLF001
    out.retics = [r for j, r in enumerate(out.retics) if j != index]
    return out


def triple_probabilities_mixture(net: N.Network, taxa, **kwargs) -> np.ndarray:
    """The plan's specification: the gamma-weighted average over the ``2^k`` parental trees.

    Kept so the error can be measured rather than argued about.  See the module docstring for why
    it is not the MSNC probability.
    """
    out = np.zeros(3)
    for choice in product((True, False), repeat=net.n_retics):
        w = 1.0
        tree = net
        # Resolve from the last reticulation backwards: index i stays valid while earlier ones are
        # untouched, and ``resolve_one_reticulation`` renumbers everything after the one it drops.
        for i in range(net.n_retics - 1, -1, -1):
            g = net.gamma_of(net.retics[i].hybrid)
            w *= g if choice[i] else 1.0 - g
            tree = resolve_one_reticulation(tree, i, choice[i])
        if w > 0.0:
            out += w * triple_probabilities(tree, taxa)
    return out
