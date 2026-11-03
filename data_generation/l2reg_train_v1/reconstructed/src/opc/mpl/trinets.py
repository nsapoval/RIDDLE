"""I2: what a level-1 network looks like on three taxa, and which triples share a shape.

Two jobs, and they are the same computation used twice.

**The catalog** (the plan's I2).  The finite set of shapes a binary level-1 network can induce on
three taxa.  Enumerated by restricting every triple of every network in a family and quotienting,
which makes completeness *checkable* -- exhaustively at small ``n``, by sampling at n = 50 --
rather than a claim.  :func:`catalog` returns it, and ``tests/test_trinets.py`` is the proof
obligation.

**The grouping** (what makes I1 affordable).  ``C(50,3) = 19,600`` triples, and the reference DP in
:mod:`opc.mpl.msnc3` costs ~180 us each, so one likelihood evaluation is 3.5 s and an optimiser
needing a few hundred of them is out of reach.  But the triples fall into a handful of shapes:
**68% of them are pure trees** at n = 50, k = 4, and the whole grid collapses to **71 shapes** of
at most 12 nodes -- a reticulation is invisible to a triple unless two of that triple's lineages
can be inside its cycle at once.  Group by shape, gather each group's parameters into arrays, and
one vectorised pass over 71 groups replaces 19,600 scalar ones.

Restriction
-----------

Keep every node ancestral to one of the three leaves, drop the rest, then suppress nodes left with
one parent and one child by summing their edge lengths.  Two things make this exact and cheap:

* a dropped subtree holds none of the triple's lineages, so nothing that happens in it can matter;
* **every hybrid that survives keeps both of its parents.**  ``ancestors`` follows *all* incoming
  edges, so if a hybrid is ancestral to one of the leaves then so is everything above it, by both
  routes.  There is therefore no "half a reticulation" case to handle, and the restriction of a
  level-1 network is a level-<=1 network on three leaves.

The suppression sums lengths across consecutive edges, which is exact because the same lineage set
occupies all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from opc.mpl.msnc3 import ancestors
from opc.simulate import networks as N

__all__ = ["Trinet", "restrict_to_triple", "catalog", "shape_counts"]


@dataclass(frozen=True)
class Trinet:
    """A network restricted to three taxa.

    Nodes are renumbered ``0..n_nodes-1`` in the order a deterministic post-order visits them, so
    two restrictions that produce the same :attr:`key` also produce the same sequence of DP
    operations.  ``parents[v]`` is ``()`` at the root, one entry at a tree node, and two at a
    hybrid -- **slot 0 is always the donor**, matching
    :meth:`opc.simulate.networks.Network.gamma_on`.

    ``edge_sources[(v, slot)]`` lists the original network edges ``(child, parent)`` whose lengths
    sum to that trinet edge, and ``gamma_source[v]`` is the original hybrid node whose gamma a
    hybrid uses.  Keeping references rather than values is what lets an optimiser vary the
    parameters without redoing the restriction.
    """

    n_nodes: int
    root: int
    leaf_of: dict[int, int]
    parents: dict[int, tuple[int, ...]]
    children: dict[int, tuple[int, ...]]
    postorder: tuple[int, ...]
    edge_sources: dict[tuple[int, int], tuple[tuple[int, int], ...]]
    gamma_source: dict[int, int]
    key: str

    @property
    def n_hybrids(self) -> int:
        return len(self.gamma_source)

    @property
    def is_tree(self) -> bool:
        return not self.gamma_source


def restrict_to_triple(net: N.Network, taxa) -> Trinet:
    """The subnetwork spanned by three leaves, degree-2 nodes suppressed."""
    order = tuple(sorted(taxa))
    leaf_node = {t: v for v, t in net.leaf_taxon.items() if t in order}
    if len(leaf_node) != 3:
        raise ValueError(f"taxa {taxa} are not three distinct leaves of this network")

    keep: set[int] = set()
    for t in order:
        keep |= ancestors(net, leaf_node[t])

    # (child -> [(parent, [original edges])]), donor first for a hybrid.
    up: dict[int, list[list]] = {}
    for v in keep:
        rows = []
        if net.is_hybrid(v):
            for p in (net.donor_of[v], net.tree_parent[v]):
                rows.append([p, [(v, p)]])
        elif net.tree_parent[v] >= 0:
            rows.append([net.tree_parent[v], [(v, net.tree_parent[v])]])
        up[v] = rows
    down: dict[int, list[int]] = {v: [] for v in keep}
    for v, rows in up.items():
        for p, _ in rows:
            down[p].append(v)

    leaves = set(leaf_node.values())
    changed = True
    while changed:
        changed = False
        for v in list(up):
            if v in leaves or not up[v] or len(up[v]) != 1 or len(down[v]) != 1:
                continue
            (p, edges), = up[v]
            (c,) = down[v]
            for row in up[c]:
                if row[0] == v:
                    row[0] = p
                    row[1] = row[1] + edges
            down[p] = [c if u == v else u for u in down[p]]
            del up[v], down[v]
            changed = True

    roots = [v for v, rows in up.items() if not rows]
    if len(roots) != 1:
        raise RuntimeError(f"restriction has {len(roots)} roots, expected 1")

    # Deterministic post-order over the surviving DAG, children before parents.  Children are
    # visited in a fixed order (by original node id) so the numbering -- and therefore the key --
    # depends only on the shape, not on dictionary iteration order.
    order_out: list[int] = []
    seen: set[int] = set()

    def walk(v: int) -> None:
        if v in seen:
            return
        seen.add(v)
        for c in sorted(down.get(v, ())):
            walk(c)
        order_out.append(v)

    walk(roots[0])
    ids = {v: i for i, v in enumerate(order_out)}

    parents, children, edge_sources, gamma_source = {}, {}, {}, {}
    for v in order_out:
        parents[ids[v]] = tuple(ids[p] for p, _ in up[v])
        children[ids[v]] = tuple(ids[c] for c in sorted(down.get(v, ())))
        for slot, (_, edges) in enumerate(up[v]):
            edge_sources[(ids[v], slot)] = tuple(edges)
        if len(up[v]) == 2:
            gamma_source[ids[v]] = v

    leaf_of = {ids[leaf_node[t]]: i for i, t in enumerate(order)}
    key = _key(len(ids), ids[roots[0]], leaf_of, parents, children)
    return Trinet(
        n_nodes=len(ids), root=ids[roots[0]], leaf_of=leaf_of, parents=parents,
        children=children, postorder=tuple(ids[v] for v in order_out),
        edge_sources=edge_sources, gamma_source=gamma_source, key=key,
    )


def _key(n_nodes, root, leaf_of, parents, children) -> str:
    """A shape key: equal keys mean an identical sequence of DP operations.

    Not a canonical form under isomorphism, and it does not need to be.  Two isomorphic trinets
    that happen to get different keys land in different groups, which costs a little time and
    nothing else; two *non*-isomorphic trinets can never collide, because the key records the
    whole labelled structure.  The node numbering comes from a deterministic post-order, so the
    key is a function of the shape rather than of dictionary order.
    """
    rows = [f"n{n_nodes}r{root}"]
    for v in range(n_nodes):
        tag = f"L{leaf_of[v]}" if v in leaf_of else "I"
        rows.append(f"{v}:{tag}:p{','.join(map(str, parents[v]))}:"
                    f"c{','.join(map(str, children[v]))}")
    return "|".join(rows)


def shape_counts(net: N.Network, index=None) -> dict[str, int]:
    """How many triples fall in each shape.  The number that decides whether grouping pays."""
    from opc.mpl.triples import triple_index

    idx = triple_index(net.n_taxa) if index is None else index
    out: dict[str, int] = {}
    for row in idx:
        k = restrict_to_triple(net, tuple(row)).key
        out[k] = out.get(k, 0) + 1
    return out


def catalog(networks, index=None) -> dict[str, Trinet]:
    """The distinct shapes a family of networks induces on three taxa -- I2's deliverable.

    Returns one representative :class:`Trinet` per shape.  Completeness is a property of the family
    passed in, which is the point: run it over an exhaustive family at small ``n`` and the result
    is a proof; run it over a sample at n = 50 and it is a check that nothing new appears.
    """
    from opc.mpl.triples import triple_index

    out: dict[str, Trinet] = {}
    for net in networks:
        idx = triple_index(net.n_taxa) if index is None else index
        for row in idx:
            tri = restrict_to_triple(net, tuple(row))
            out.setdefault(tri.key, tri)
    return out
