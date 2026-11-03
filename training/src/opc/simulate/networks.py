"""Constrained level-1 (galled) network generator.

Drives the main grid (decision D1): exact reticulation count, exact inheritance probability, and
level-1 by construction rather than by rejection.

Design notes
------------

**Time, not lengths.**  A node carries a ``time`` measured down from the root; an edge length is a
difference of times.  Time-consistency and ultrametricity are then true by construction rather
than by assertion, which removes an entire class of bug (and is exactly the class that
SiPhyNetwork issue #8 lives in).  Rescaling the ILS axis is one multiply over ``time``.

**Level-1 == edge-disjoint cycles** (SPEC-AMENDMENTS A7).  Two cycles sharing a single *vertex*
are two biconnected components and the network is still level-1; only a shared *edge* merges them.
So the placement rule is: the two tree paths from the LCA down to the two attachment points must
use no edge consumed by an earlier cycle.  The spec's extra acyclicity check is vacuous -- a
horizontal edge between two branches alive at the same instant cannot create a directed cycle,
because time strictly increases downward.

**Galled tree == vertex-disjoint cycles** (A28).  ``exp-plan-n50-mpl-v0.1`` asks for galled trees,
the strictly stronger condition: no two cycles may share even a cut vertex.  Passing
``vertex_disjoint=True``
enforces it -- and, **measured, it changes nothing about this generator's output**, because the
edge rule already implies the vertex rule here.  Every interior vertex of a placed cycle has its
incoming tree edge consumed and both of the apex's child edges are consumed, so a later cycle can
only reach a shared vertex through a consumed edge.  The argument needs each cycle vertex to have
one incoming edge, which is why it fails for a *hybrid* -- and A20's "a hybrid at another cycle's
apex" is exactly that escape, in the regular version rather than here.  The flag is kept because
the invariant should be enforced rather than inherited from an argument about binarity, and
:func:`is_galled` is the gate the driver asserts.  Off by default so the Phase 0 base topologies,
cached on disk, keep reproducing byte-identically.

**Cycle size** is ``len(side1) + len(side2)``: the number of tree edges the cycle spans, which is
also the number of clusters :func:`block_clusters` returns for it, i.e. exactly the quantity the
identifiability stratification is about.  The floor is 3 -- the hybrid must sit strictly below the
LCA (no shortcuts), so its side contributes at least two edges while the donor's may contribute
one.

**Blocked design** (SPEC-AMENDMENTS A2).  :func:`build_base` produces one canonical network per
``(n, replicate)``, scaled so the base tree's median internal branch is 1.0 coalescent unit, with
``r_max`` reticulations recorded in insertion order.  Every grid cell is then a pure derivation:
:func:`scaled` multiplies times to hit the target CU, :func:`with_gamma` sets inheritance, and
:func:`with_reticulations` drops back to a nested subset of the reticulations.  Topology is
therefore identical across the CU, gamma and r axes, making those contrasts paired.
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "Network",
    "Reticulation",
    "from_enewick",
    "build_base",
    "scaled",
    "with_gamma",
    "with_random_gammas",
    "with_reticulations",
    "derive",
    "hardwired_clusters",
    "full_clusters",
    "nontrivial_clusters",
    "singleton_hybrid_fraction",
    "signature_pair",
    "n_cluster_visible",
    "block_clusters",
    "block_sides",
    "cycle_size",
    "cycle_sizes",
    "cycle_vertices",
    "is_galled",
    "displayed_tree",
    "to_enewick",
    "level",
    "internal_branch_lengths",
    "metadata",
]


@dataclass(frozen=True)
class Reticulation:
    """One reticulation: the horizontal edge ``donor -> hybrid``, inserted at ``time``."""

    hybrid: int
    donor: int
    order: int
    time: float


@dataclass
class Network:
    """A rooted, time-embedded, binary level-1 phylogenetic network.

    Every node has exactly one *tree* parent (except the root).  A hybrid node additionally has a
    *donor* parent reached by a zero-length horizontal edge.  ``gamma`` is the inheritance
    probability on the horizontal edge; the tree edge into the hybrid carries ``1 - gamma``.
    """

    n_taxa: int
    labels: list[str]
    time: list[float]
    tree_parent: list[int]
    tree_children: list[list[int]]
    leaf_taxon: dict[int, int]
    horiz_child: dict[int, int]
    donor_of: dict[int, int]
    retics: list[Reticulation]
    root: int
    gamma: float = 0.5
    #: Per-hybrid **donor-side** inheritance probability, keyed by hybrid node id.  Empty means
    #: "use the scalar :attr:`gamma` everywhere", which is what every artifact written before the
    #: b-phases carries -- so an empty map must reproduce the scalar path exactly, and
    #: ``tests/test_networks_gamma.py`` pins that it does.
    #:
    #: The evaluator has always been per-hybrid (``likelihood.Parameters.gammas`` is a vector and
    #: ``msnc3.triple_probabilities`` takes a per-hybrid dict); only this container was scalar, so
    #: a network could not *state* what the optimiser could already fit.
    gammas: dict[int, float] = field(default_factory=dict)
    consumed: set[int] = field(default_factory=set, repr=False)
    #: Vertices any earlier cycle passes through, LCA included.  Only consulted under
    #: ``vertex_disjoint=True``; a shared edge implies shared endpoints, so this subsumes
    #: ``consumed`` whenever it is enforced.
    consumed_vertices: set[int] = field(default_factory=set, repr=False)
    shortcut_rejections: int = 0
    #: Placements rejected for falling outside the requested cycle-size band, so a low placement
    #: rate can be attributed to the band rather than to the disjointness rule.
    cycle_size_rejections: int = 0

    # ---------------------------------------------------------------- structure

    @property
    def n_nodes(self) -> int:
        return len(self.time)

    @property
    def n_retics(self) -> int:
        return len(self.retics)

    def is_hybrid(self, v: int) -> bool:
        return v in self.donor_of

    def is_leaf(self, v: int) -> bool:
        return v in self.leaf_taxon

    def net_children(self, v: int) -> list[int]:
        kids = list(self.tree_children[v])
        h = self.horiz_child.get(v)
        if h is not None:
            kids.append(h)
        return kids

    def net_parents(self, v: int) -> list[int]:
        out = []
        if self.tree_parent[v] >= 0:
            out.append(self.tree_parent[v])
        d = self.donor_of.get(v)
        if d is not None:
            out.append(d)
        return out

    def edge_length(self, parent: int, child: int) -> float:
        return self.time[child] - self.time[parent]

    def gamma_of(self, hybrid: int) -> float:
        """The donor-side inheritance probability at ``hybrid``.

        ``getattr`` rather than attribute access: a pickle written before per-hybrid gamma has no
        such key, and unpickling a dataclass restores ``__dict__`` without calling ``__init__``.
        Same reason as :attr:`consumed_vertices` in :meth:`copy`.
        """
        return getattr(self, "gammas", {}).get(hybrid, self.gamma)

    def gamma_on(self, parent: int, child: int) -> float | None:
        """Inheritance probability of the edge ``parent -> child``, or None for tree edges."""
        if child not in self.donor_of:
            return None
        g = self.gamma_of(child)
        return g if self.donor_of[child] == parent else 1.0 - g

    def copy(self) -> Network:
        return Network(
            n_taxa=self.n_taxa,
            labels=list(self.labels),
            time=list(self.time),
            tree_parent=list(self.tree_parent),
            tree_children=[list(c) for c in self.tree_children],
            leaf_taxon=dict(self.leaf_taxon),
            horiz_child=dict(self.horiz_child),
            donor_of=dict(self.donor_of),
            retics=list(self.retics),
            root=self.root,
            gamma=self.gamma,
            gammas=dict(getattr(self, "gammas", {})),
            consumed=set(self.consumed),
            # getattr, not attribute access: pickles written before A28 have no such key, and
            # unpickling a dataclass restores __dict__ without calling __init__.
            consumed_vertices=set(getattr(self, "consumed_vertices", ())),
            shortcut_rejections=self.shortcut_rejections,
            cycle_size_rejections=getattr(self, "cycle_size_rejections", 0),
        )


# --------------------------------------------------------------------------------- construction


def _from_dendropy(tree, n_taxa: int) -> Network:
    """Convert a dendropy ultrametric birth-death tree into a :class:`Network` with no hybrids."""
    order = list(tree.preorder_node_iter())
    ids = {nd: i for i, nd in enumerate(order)}
    n = len(order)
    time = [0.0] * n
    tree_parent = [-1] * n
    tree_children: list[list[int]] = [[] for _ in range(n)]
    leaf_taxon: dict[int, int] = {}
    labels: list[str] = []
    for nd in order:
        v = ids[nd]
        time[v] = nd.distance_from_root()
        if nd.parent_node is not None:
            p = ids[nd.parent_node]
            tree_parent[v] = p
            tree_children[p].append(v)
    for nd in tree.leaf_node_iter():
        leaf_taxon[ids[nd]] = len(labels)
        labels.append(f"t{len(labels) + 1}")
    if len(labels) != n_taxa:
        raise RuntimeError(f"expected {n_taxa} leaves, dendropy gave {len(labels)}")
    return Network(
        n_taxa=n_taxa,
        labels=labels,
        time=time,
        tree_parent=tree_parent,
        tree_children=tree_children,
        leaf_taxon=leaf_taxon,
        horiz_child={},
        donor_of={},
        retics=[],
        root=ids[tree.seed_node],
    )


def internal_branch_lengths(net: Network) -> np.ndarray:
    """Lengths of the branches the ILS axis is normalised on (D2).

    An edge counts when its child is neither a leaf nor a hybrid node.  Excluding hybrid in-edges
    is not cosmetic: PhyloNetworks flags *both* parent edges of a hybrid as ``hybrid``, and it has
    no way to tell which one was the horizontal donor edge, so this is the only definition the
    Python generator and the Julia validator can both compute and agree on exactly.  The
    zero-length horizontal edges are excluded either way -- including them would drag the median
    down by construction.
    """
    out = [
        net.time[v] - net.time[net.tree_parent[v]]
        for v in range(net.n_nodes)
        if net.tree_parent[v] >= 0 and not net.is_leaf(v) and not net.is_hybrid(v)
    ]
    return np.asarray(out, dtype=float)


def _rescale(net: Network, factor: float) -> None:
    for v in range(net.n_nodes):
        net.time[v] *= factor


def _depth(net: Network, v: int) -> int:
    d = 0
    while net.tree_parent[v] >= 0:
        v = net.tree_parent[v]
        d += 1
    return d


def _lca(net: Network, a: int, b: int) -> int:
    da, db = _depth(net, a), _depth(net, b)
    while da > db:
        a = net.tree_parent[a]
        da -= 1
    while db > da:
        b = net.tree_parent[b]
        db -= 1
    while a != b:
        a = net.tree_parent[a]
        b = net.tree_parent[b]
    return a


def _up_edges(net: Network, x: int, stop: int) -> list[int]:
    """Tree edges on the path from ``x`` up to but excluding ``stop``, named by their child
    node."""
    out = []
    while x != stop:
        out.append(x)
        x = net.tree_parent[x]
        if x < 0:
            raise RuntimeError("stop node is not an ancestor")
    return out


def _alive_tree_edges(net: Network, t: float) -> list[int]:
    """Child ids of tree edges spanning time ``t``."""
    return [
        v
        for v in range(net.n_nodes)
        if net.tree_parent[v] >= 0 and net.time[net.tree_parent[v]] < t < net.time[v]
    ]


def _split_edge(net: Network, child: int, t: float) -> int:
    """Insert a degree-2 node at time ``t`` on the tree edge ending at ``child``.  Returns it."""
    p = net.tree_parent[child]
    x = net.n_nodes
    net.time.append(t)
    net.tree_parent.append(p)
    net.tree_children.append([child])
    net.tree_children[p][net.tree_children[p].index(child)] = x
    net.tree_parent[child] = x
    return x


def _add_reticulation(
    net: Network,
    rng: np.random.Generator,
    max_tries: int = 400,
    allow_shortcuts: bool = False,
    pairs_per_time: int = 48,
    vertex_disjoint: bool = False,
    cycle_size_range: tuple[int, int | None] = (3, None),
) -> bool:
    """Attempt to add one level-1-safe reticulation.  Returns False if no placement was found.

    ``allow_shortcuts=False`` (the default) rejects placements where the hybrid attaches directly
    at the cycle's LCA ``z``.  That configuration makes the edge ``z -> h`` a *shortcut*: ``z``
    already reaches ``h`` through the donor, so the network is not regular, and Hellmuth et al.'s
    correspondence is between clustering systems and *regular* networks.  Concretely, both hybrid
    parents then carry the same hardwired cluster as an ancestor, the cycle contributes no
    overlapping pair, and the reticulation is invisible to any cluster-based method -- with the
    spec's unrestricted "pick two coexisting branches" rule this happens to roughly half of all
    reticulations, which would cap ``retention_hybrid`` near 0.5 for reasons that have nothing to
    do with ILS or filtering.  The rejection rate is recorded as ``shortcut_rejections``.

    ``vertex_disjoint`` upgrades level-1 to galled -- redundant here, see the module docstring --
    and ``cycle_size_range`` is the inclusive ``(lo, hi)`` band the cycle's tree-edge count must
    land in (``hi=None`` for no ceiling).  The band *does* make placement harder, so
    :func:`build_base` cannot be assumed to succeed at a given ``r_max`` under it; measure it with
    ``scripts/probe_generator.py`` rather than assume it.
    """
    horizon = max(net.time)
    for _ in range(max_tries):
        t = float(rng.uniform(0.0, horizon))
        alive = [v for v in _alive_tree_edges(net, t) if v not in net.consumed]
        if vertex_disjoint:
            # An alive edge whose *parent* is already on a cycle cannot host an attachment: the
            # new cycle would have to climb through that parent to reach its LCA.  Filtering here
            # rather than inside the pair loop keeps the quadratic pair enumeration small.
            alive = [v for v in alive if net.tree_parent[v] not in net.consumed_vertices]
        if len(alive) < 2:
            continue
        # Try several pairs at this instant before resampling.  Drawing a single pair per time
        # wastes the (expensive) alive-edge scan and, once earlier cycles have consumed the
        # edges near the root, fails often enough that r=20 at n=200 could not be placed at all.
        order = rng.permutation(len(alive))
        pairs = [
            (int(alive[order[a]]), int(alive[order[b]]))
            for a in range(len(order))
            for b in range(a + 1, len(order))
        ]
        if len(pairs) > pairs_per_time:
            pairs = [pairs[k] for k in rng.choice(len(pairs), size=pairs_per_time, replace=False)]
        if _place_from_pairs(net, pairs, t, allow_shortcuts, vertex_disjoint, cycle_size_range):
            return True
    return False


def _place_from_pairs(
    net: Network,
    pairs,
    t: float,
    allow_shortcuts: bool,
    vertex_disjoint: bool = False,
    cycle_size_range: tuple[int, int | None] = (3, None),
) -> bool:
    """Try each candidate pair at time ``t``; place the first that keeps the network level-1."""
    lo, hi = cycle_size_range
    for c1, c2 in pairs:
        z = _lca(net, c1, c2)
        if not allow_shortcuts and net.tree_parent[c2] == z:
            # The hybrid side must sit strictly below the LCA.  Swapping roles often fixes it.
            if net.tree_parent[c1] == z:
                net.shortcut_rejections += 1
                continue
            c1, c2 = c2, c1
        p1, p2 = net.tree_parent[c1], net.tree_parent[c2]
        side1 = _up_edges(net, p1, z) + [c1]
        side2 = _up_edges(net, p2, z) + [c2]
        if net.consumed.intersection(side1) or net.consumed.intersection(side2):
            continue
        if set(side1) & set(side2):
            continue  # would share an edge with itself; cannot happen for distinct alive edges
        if len(side1) + len(side2) < lo or (hi is not None and len(side1) + len(side2) > hi):
            net.cycle_size_rejections += 1
            continue
        # The cycle's own vertices, in pre-split ids: the LCA, and everything strictly below it on
        # each side down to the attachment point's parent.  ``c1``/``c2`` themselves stay off the
        # cycle -- the split leaves their lower halves outside it -- so they are not claimed.
        interior = set(side1[:-1]) | set(side2[:-1]) | {z}
        if vertex_disjoint and net.consumed_vertices.intersection(interior):
            continue
        donor = _split_edge(net, c1, t)
        hybrid = _split_edge(net, c2, t)
        net.horiz_child[donor] = hybrid
        net.donor_of[hybrid] = donor
        net.retics.append(Reticulation(hybrid, donor, len(net.retics), t))
        net.consumed.update(side1[:-1] + [donor])
        net.consumed.update(side2[:-1] + [hybrid])
        net.consumed_vertices.update(interior | {donor, hybrid})
        return True
    return False


def build_base(
    n_taxa: int,
    r_max: int,
    seed: int,
    *,
    birth_rate: float = 1.0,
    death_rate: float = 0.2,
    max_topology_tries: int = 40,
    gsa_factor: float = 1.5,
    vertex_disjoint: bool = False,
    cycle_size_range: tuple[int, int | None] = (3, None),
) -> Network:
    """One canonical network per ``(n, replicate)``: BD topology + ``r_max`` nested reticulations.

    Sampling uses the **GSA** (``gsa_ntax``), not simple stop-at-``n``-tips.  Stopping the instant
    the ``n``-th tip appears puts the final speciation exactly at the present, producing a cherry
    with two zero-length pendant branches in every single tree -- a systematic artefact that would
    inject maximal ILS into one taxon pair per replicate.  GSA samples uniformly among the
    intervals during which ``n`` lineages exist and has no such bias.

    Times are normalised so the base tree's median internal branch is 1.0; :func:`scaled` then
    re-normalises each derived network onto the target coalescent-unit value.
    """
    import random

    from dendropy.simulate import treesim

    gsa_ntax = max(n_taxa + 2, int(gsa_factor * n_taxa))
    for attempt in range(max_topology_tries):
        rng = np.random.default_rng([seed, attempt])
        try:
            tree = treesim.birth_death_tree(
                birth_rate=birth_rate,
                death_rate=death_rate,
                num_extant_tips=n_taxa,
                gsa_ntax=gsa_ntax,
                rng=random.Random(int(rng.integers(1, 2**31))),
            )
        except TypeError as exc:
            # dendropy raises "Node has no parent and is implicit root: cannot be pruned" when
            # the extinction pruning reaches the seed node.  Draw a different realisation.
            if "implicit root" not in str(exc):
                raise
            continue
        net = _from_dendropy(tree, n_taxa)
        ib = internal_branch_lengths(net)
        if ib.size == 0 or not np.isfinite(np.median(ib)) or np.median(ib) <= 0:
            continue
        _rescale(net, 1.0 / float(np.median(ib)))
        placed = all(
            _add_reticulation(
                net, rng, vertex_disjoint=vertex_disjoint, cycle_size_range=cycle_size_range
            )
            for _ in range(r_max)
        )
        if placed:
            return net
    kind = "vertex-disjoint" if vertex_disjoint else "edge-disjoint"
    raise RuntimeError(
        f"could not place {r_max} {kind} reticulations with cycle size in {cycle_size_range} "
        f"on an n={n_taxa} tree in {max_topology_tries} topologies"
    )


# ------------------------------------------------------------------------------- derivations


def scaled(net: Network, cu: float) -> Network:
    """Uniformly rescale time so *this* network's median internal branch equals ``cu`` (D2).

    Normalising the derived network rather than the base tree matters: inserting a reticulation
    splits an internal branch into two shorter ones, so a fixed base-tree scale would make
    high-``r`` cells systematically shorter-branched -- i.e. it would confound the ``r`` axis with
    the ILS axis, the two things the study most needs to keep apart.  Uniform scaling preserves
    time-consistency, so the network stays valid.
    """
    out = net.copy()
    ib = internal_branch_lengths(out)
    med = float(np.median(ib))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("network has no positive internal branch to normalise on")
    _rescale(out, cu / med)
    return out


def with_gamma(net: Network, gamma: float) -> Network:
    """Set one inheritance probability for the whole network.

    **This clears any per-hybrid map.**  Setting a scalar gamma must mean what it says, and a
    silently surviving per-hybrid map would make ``with_gamma(net, 0.3).gamma_of(h)`` return
    something other than 0.3 -- the class of defect A52 is about, in a new place.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0,1), got {gamma}")
    return replace(net.copy(), gamma=gamma, gammas={})


def derive(base: Network, *, r: int, cu: float, gamma: float,
           gamma_range=None, gamma_seed: int | None = None) -> Network:
    """The truth network a grid cell stands for: ``base -> reticulations -> scale -> gamma``.

    **The order is not decorative.**  ``with_reticulations`` before ``scaled`` before the gamma
    step is what every driver in the study does and what the cached gene trees were simulated
    under, and it had been written out longhand at six call sites -- so a phase needing a
    different gamma step would have had to be added to all six, and would have been added to
    five.  This is that line, once.

    ``gamma_range`` selects the b-grids' per-reticulation draw; ``None`` is the scalar
    :func:`with_gamma` every earlier phase uses, and the two must not be confusable, which is why
    ``gamma_seed`` is required with a range rather than defaulted.
    """
    net = scaled(with_reticulations(base, r), cu)
    if gamma_range is None:
        return with_gamma(net, gamma)
    if gamma_seed is None:
        raise ValueError("gamma_range needs a gamma_seed; see grid.Cell.gamma_seed")
    return with_random_gammas(net, gamma_range[0], gamma_range[1], gamma_seed)


def with_random_gammas(net: Network, lo: float, hi: float, seed: int) -> Network:
    """One inheritance probability per reticulation, drawn ``U(lo, hi)`` and frozen.

    Drawn **in reticulation insertion order**, so the draw for reticulation ``i`` does not depend
    on how many were placed.  That is what keeps :func:`with_reticulations` nested (A2's blocked
    design): the k-ladder derived from one base topology sees the *same* gamma on the same hybrid
    at every ``k``, and the ``k`` contrast stays paired.

    ``net.gamma`` is set to the mean of the draws so that anything reading the scalar -- a report
    column, ``metadata`` -- gets a summary rather than a stale 0.5.
    """
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"gamma range must satisfy 0 < lo <= hi < 1, got ({lo}, {hi})")
    rng = np.random.default_rng(seed)
    out = net.copy()
    out.gammas = {ret.hybrid: float(rng.uniform(lo, hi)) for ret in out.retics}
    if out.gammas:
        out.gamma = float(np.mean(list(out.gammas.values())))
    return out


def _suppress_degree2(net: Network, v: int) -> None:
    """Remove a node with one tree parent and one tree child, splicing the edge back together."""
    p = net.tree_parent[v]
    (c,) = net.tree_children[v]
    net.tree_children[p][net.tree_children[p].index(v)] = c
    net.tree_parent[c] = p
    net.tree_children[v] = []
    net.tree_parent[v] = -2  # tombstone; ids stay stable, dead nodes are never reachable


def with_reticulations(net: Network, r: int) -> Network:
    """Drop back to the first ``r`` reticulations in insertion order (SPEC-AMENDMENTS A2).

    Removing in reverse order is what makes this safe: a later reticulation may have split an edge
    below an earlier one, and unwinding last-first restores each edge before it is needed again.
    """
    if not 0 <= r <= net.n_retics:
        raise ValueError(f"r={r} outside 0..{net.n_retics}")
    out = net.copy()
    for ret in reversed(out.retics[r:]):
        del out.horiz_child[ret.donor]
        del out.donor_of[ret.hybrid]
        _suppress_degree2(out, ret.hybrid)
        _suppress_degree2(out, ret.donor)
    out.retics = out.retics[:r]
    # Prune to the surviving hybrids.  A dropped hybrid's node id is tombstoned rather than
    # reused, so a stale entry would never be *read* -- but it would be carried into
    # ``metadata`` and into any future consumer that iterates the map instead of the retics.
    if out.gammas:
        keep = {ret.hybrid for ret in out.retics}
        out.gammas = {h: g for h, g in out.gammas.items() if h in keep}
    out.consumed = set()
    out.consumed_vertices = set()
    return out


# --------------------------------------------------------------------------- derived quantities


def _postorder(net: Network) -> list[int]:
    """Nodes of the reachable DAG, children before parents."""
    seen: set[int] = set()
    out: list[int] = []
    stack = [(net.root, False)]
    while stack:
        v, expanded = stack.pop()
        if expanded:
            out.append(v)
            continue
        if v in seen:
            continue
        seen.add(v)
        stack.append((v, True))
        for c in net.net_children(v):
            if c not in seen:
                stack.append((c, False))
    return out


def hardwired_clusters(net: Network) -> dict[int, frozenset[int]]:
    """Leaf set reachable below every node (taxon indices)."""
    below: dict[int, frozenset[int]] = {}
    for v in _postorder(net):
        if net.is_leaf(v):
            below[v] = frozenset({net.leaf_taxon[v]})
        else:
            acc: set[int] = set()
            for c in net.net_children(v):
                acc |= below[c]
            below[v] = frozenset(acc)
    return below


def full_clusters(net: Network) -> set[frozenset[int]]:
    """``C_Nstar`` as §4.1 defines it: the *full* hardwired clustering system.

    Singletons and the whole taxon set are included.  This is the object the theory is about --
    it is what Hellmuth et al. Thm 7.38 / 8.9 assert is closed and satisfies (L)+(N3O), and the
    nontrivial restriction below is *not* closed whenever a hybrid's cluster is a single leaf
    (a hybrid species with one extant descendant -- the common biological case).
    """
    return set(hardwired_clusters(net).values())


def nontrivial_clusters(net: Network) -> set[frozenset[int]]:
    """The retention target of §6.3: ``C_Nstar`` minus singletons and the full taxon set.

    Excluding singletons costs nothing structurally -- a singleton is contained in or disjoint
    from every set, so it overlaps nothing and would sit in the overlap graph as an isolated
    vertex -- but it does mean this system is not closed in general.  Use :func:`full_clusters`
    for any closedness claim.
    """
    full = frozenset(range(net.n_taxa))
    return {c for c in hardwired_clusters(net).values() if 1 < len(c) < len(full)}


def singleton_hybrid_fraction(net: Network) -> float:
    """Fraction of reticulations whose hybrid cluster is a single leaf.

    Recorded because it biases ``retention_hybrid`` optimistically: a singleton hybrid cluster is
    recoverable from any overlapping pair that meets in that leaf, which is far easier than
    recovering a large hybrid cluster exactly.
    """
    if not net.retics:
        return float("nan")
    below = hardwired_clusters(net)
    return sum(len(below[r.hybrid]) == 1 for r in net.retics) / len(net.retics)


def level(net: Network) -> int:
    """Maximum number of reticulations in any biconnected component (Hopcroft-Tarjan, iterative).

    A reticulation is counted in the component holding its **incoming edges**, not in every
    component its hybrid node touches.  The distinction is not academic: a cut vertex belongs to
    every biconnected component incident to it, so a hybrid sitting at the apex of a *second*
    cycle would otherwise be counted twice and a level-1 network reported as level-2.  That
    configuration does not arise in the generator's own output -- the ``consumed`` edge set keeps
    the cycles apart -- but it does arise in the **regular version** of such a network, which is
    exactly the object a cluster-based method reconstructs (:mod:`opc.solve.closure`).
    """
    adj: dict[int, list[tuple[int, int]]] = {}
    edges: list[tuple[int, int]] = []
    in_edge: dict[int, int] = {}
    for v in _postorder(net):
        for c in net.net_children(v):
            eid = len(edges)
            edges.append((v, c))
            if net.is_hybrid(c) and net.tree_parent[c] == v:
                in_edge[c] = eid
            adj.setdefault(v, []).append((c, eid))
            adj.setdefault(c, []).append((v, eid))
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    stack_edges: list[int] = []
    best = 0
    counter = 0
    for start in list(adj):
        if start in disc:
            continue
        work = [(start, -1, iter(adj[start]))]
        disc[start] = low[start] = counter
        counter += 1
        while work:
            v, pe, it = work[-1]
            advanced = False
            for w, eid in it:
                if eid == pe:
                    continue
                if w not in disc:
                    stack_edges.append(eid)
                    disc[w] = low[w] = counter
                    counter += 1
                    work.append((w, eid, iter(adj[w])))
                    advanced = True
                    break
                if disc[w] < disc[v]:
                    stack_edges.append(eid)
                    low[v] = min(low[v], disc[w])
            if advanced:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    comp: list[int] = []
                    while stack_edges and stack_edges[-1] != pe:
                        comp.append(stack_edges.pop())
                    if stack_edges:
                        comp.append(stack_edges.pop())
                    here = set(comp)
                    best = max(best, sum(1 for e in in_edge.values() if e in here))
    return best


# ------------------------------------------------------------------------------ extended Newick


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def to_enewick(net: Network) -> str:
    """Extended Newick in the ``name:length::gamma`` dialect PhyloNetworks reads.

    The ``#Hk`` leaf copy is placed on the *minor* hybrid edge (gamma < 0.5), matching the
    PhyloNetworks convention; ties go to the donor edge so the output is deterministic.
    """
    tag = {ret.hybrid: k + 1 for k, ret in enumerate(net.retics)}
    major: dict[int, int] = {}
    for ret in net.retics:
        donor_gamma = net.gamma_of(ret.hybrid)
        major[ret.hybrid] = ret.donor if donor_gamma > 0.5 else net.tree_parent[ret.hybrid]

    def emit(v: int, parent: int | None) -> str:
        if parent is None:
            kids = ",".join(emit(c, v) for c in net.net_children(v))
            return f"({kids});"
        length = _fmt(net.edge_length(parent, v))
        if net.is_hybrid(v):
            g = _fmt(net.gamma_on(parent, v))
            if parent != major[v]:
                return f"#H{tag[v]}:{length}::{g}"
            # **A hybrid may be a leaf**, and then it has no children to parenthesise.
            # ``()#H1`` would be malformed *and* would drop the taxon name silently; eNewick
            # writes the label carrying the tag instead.  The generator never produces a hybrid
            # leaf -- it makes hybrids by splitting an edge -- but the *regular* version of a
            # network does, whenever a hybrid's cluster is a singleton
            # (:mod:`opc.solve.closure`), which is the common biological case of a hybrid
            # species with one extant descendant.
            if net.is_leaf(v):
                # Written as the hybrid carrying a zero-length pendant child rather than as a
                # bare ``label#Hk``.  Semantically identical -- a lineage at the leaf is a lineage
                # at the hybrid -- but PhyloNet requires a reticulation node to have a child and
                # throws ArrayIndexOutOfBoundsException deep inside GeneTreeProbabilityYF if it
                # does not, which is the kind of failure that is easier to prevent than to read.
                return f"({net.labels[net.leaf_taxon[v]]}:0)#H{tag[v]}:{length}::{g}"
            kids = ",".join(emit(c, v) for c in net.net_children(v))
            return f"({kids})#H{tag[v]}:{length}::{g}"
        if net.is_leaf(v):
            return f"{net.labels[net.leaf_taxon[v]]}:{length}"
        kids = ",".join(emit(c, v) for c in net.net_children(v))
        return f"({kids}):{length}"

    return emit(net.root, None)


# ------------------------------------------------------------------------------------ metadata


def metadata(net: Network, **extra) -> dict:
    """Ground truth recorded alongside every network, per §4.1 of the spec."""
    below = hardwired_clusters(net)
    ib = internal_branch_lengths(net)
    quantiles = np.quantile(ib, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist() if ib.size else []
    retics = []
    for k, ret in enumerate(net.retics):
        tp = net.tree_parent[ret.hybrid]
        retics.append(
            {
                "tag": f"H{k + 1}",
                "gamma": net.gamma_of(ret.hybrid),
                "time": ret.time,
                "cycle_size": cycle_size(net, ret),
                "hybrid_cluster": sorted(below[ret.hybrid]),
                "donor_parent_cluster": sorted(below[ret.donor]),
                "tree_parent_cluster": sorted(below[tp]),
            }
        )
    return {
        "n_taxa": net.n_taxa,
        "labels": net.labels,
        "n_retics": net.n_retics,
        "gamma": net.gamma,
        "level": level(net),
        "root_height": max(net.time),
        "internal_branch_quantiles": quantiles,
        "realized_median_internal_cu": float(np.median(ib)) if ib.size else float("nan"),
        "n_clusters_true": len(nontrivial_clusters(net)),
        "n_clusters_true_full": len(full_clusters(net)),
        "singleton_hybrid_fraction": singleton_hybrid_fraction(net),
        "cycle_sizes": cycle_sizes(net),
        "shortcut_rejections": net.shortcut_rejections,
        "n_retics_cluster_visible": n_cluster_visible(net),
        "reticulations": retics,
        **extra,
    }


def write_network(net: Network, enewick_path, json_path, **extra) -> None:
    """One network per file -- sidesteps SiPhyNetwork issue #10 and keeps caching simple."""
    with open(enewick_path, "w") as fh:
        fh.write(to_enewick(net) + "\n")
    with open(json_path, "w") as fh:
        json.dump(metadata(net, **extra), fh, indent=1)


def signature_pair(net: Network, ret: Reticulation) -> tuple[frozenset[int], frozenset[int]]:
    """The two clusters whose overlap is a reticulation's fingerprint in ``C(N*)``.

    The donor's cluster (``c(c1) | c(h)``) and the cluster of the hybrid's tree parent.  They meet
    exactly in the hybrid's cluster, so an inference method sees the reticulation as this
    overlapping pair.
    """
    below = hardwired_clusters(net)
    return below[ret.donor], below[net.tree_parent[ret.hybrid]]


def n_cluster_visible(net: Network) -> int:
    """Reticulations whose signature pair is a genuine overlapping pair of nontrivial clusters.

    A shortcut reticulation scores 0 here: it is present in the network but leaves no trace in the
    clustering system, so no cluster-based method can recover it.  With ``allow_shortcuts=False``
    every reticulation should be visible; the counter exists to prove that, and to quantify the
    loss if shortcuts are ever re-enabled.
    """
    from opc.clusters.systems import overlaps

    below = hardwired_clusters(net)
    cs = nontrivial_clusters(net)
    total = 0
    for ret in net.retics:
        a, b = below[ret.donor], below[net.tree_parent[ret.hybrid]]
        if a in cs and b in cs and overlaps(a, b):
            total += 1
    return total


def block_clusters(net: Network, ret: Reticulation) -> set[frozenset[int]]:
    """Clusters of the nodes on one reticulation's cycle, excluding the cycle's top node.

    The cycle runs from ``z = LCA(donor, hybrid)`` down both sides to the donor and the hybrid.
    ``z`` itself is excluded: its cluster is the union of both sides and sits *above* the block,
    so it is nested with -- not overlapping -- everything inside.  These are the clusters
    ``retention_block`` asks to stay in one piece.
    """
    below = hardwired_clusters(net)
    z = _lca(net, ret.donor, ret.hybrid)
    out: set[frozenset[int]] = set()
    for start in (ret.donor, ret.hybrid):
        v = start
        while v != z:
            out.add(below[v])
            v = net.tree_parent[v]
    return out


def cycle_size(net: Network, ret: Reticulation) -> int:
    """Tree edges on one reticulation's cycle -- the S50 stratification axis (A28).

    Counted after the fact from the finished network rather than remembered from placement, so it
    stays correct for a network read back from disk or produced by anything but this generator.
    Equals ``len(block_clusters(net, ret))`` whenever the cycle's clusters are distinct, which is
    the sense in which it measures how much of the clustering system the cycle contributes.
    """
    z = _lca(net, ret.donor, ret.hybrid)
    total = 0
    for start in (ret.donor, ret.hybrid):
        v = start
        while v != z:
            total += 1
            v = net.tree_parent[v]
    return total


def cycle_sizes(net: Network) -> list[int]:
    return [cycle_size(net, ret) for ret in net.retics]


def cycle_vertices(net: Network, ret: Reticulation) -> set[int]:
    """Every node on one reticulation's cycle, the LCA and both endpoints included."""
    z = _lca(net, ret.donor, ret.hybrid)
    out = {z}
    for start in (ret.donor, ret.hybrid):
        v = start
        while v != z:
            out.add(v)
            v = net.tree_parent[v]
    return out


def recompute_consumed(net: Network) -> None:
    """Rebuild ``consumed`` / ``consumed_vertices`` from the reticulations actually present.

    :func:`with_reticulations` and :func:`displayed_tree` clear both sets, because during
    *construction* they are scratch state.  Anything that wants to add a reticulation to an
    already-built network -- the near-miss perturbations E0b scores, E2a's "add one edge" -- has to
    restore them first, or the new cycle is placed with no knowledge of the existing ones and the
    result is not level-1.
    """
    net.consumed = set()
    net.consumed_vertices = set()
    for ret in net.retics:
        z = _lca(net, ret.donor, ret.hybrid)
        for start in (ret.donor, ret.hybrid):
            v = start
            while v != z:
                net.consumed.add(v)
                net.consumed_vertices.add(v)
                v = net.tree_parent[v]
        net.consumed_vertices.add(z)


def is_galled(net: Network) -> bool:
    """Are the reticulation cycles pairwise **vertex**-disjoint?

    Strictly stronger than ``level(net) <= 1``, which only forbids a shared *edge*: two cycles
    meeting at a cut vertex are still two biconnected components (A7).  ``exp-plan-n50-mpl-v0.1``
    asks for galled trees, so this is the gate that has to be asserted, not ``level``.
    """
    seen: set[int] = set()
    for ret in net.retics:
        vs = cycle_vertices(net, ret)
        if seen & vs:
            return False
        seen |= vs
    return True


def block_sides(net: Network, ret: Reticulation):
    """``(donor_side, hybrid_side, h)`` -- the bipartition :func:`block_clusters` flattens.

    The two sides are what the block's induced overlap graph actually is.  Every cluster on the
    cycle contains the hybrid's cluster ``h``; two clusters on the *same* side are nested (they
    chain up the tree towards ``z``); two on *opposite* sides meet in exactly ``h``.  So a block
    induces a complete bipartite ``K_{|donor|,|hybrid|}`` -- there are no "intermediate" clusters
    linking it, and it stays connected while either side survives.

    ``h`` itself is returned separately because it is nested inside every other member and is
    therefore **isolated in the block's own subgraph**.  Counting it as a block member makes any
    "do the survivors share one component" test fail whenever ``h`` survives (SPEC-AMENDMENTS
    A17).
    """
    below = hardwired_clusters(net)
    z = _lca(net, ret.donor, ret.hybrid)
    sides = []
    for start in (ret.donor, ret.hybrid):
        chain, v = [], start
        while v != z:
            chain.append(below[v])
            v = net.tree_parent[v]
        sides.append(chain)
    h = below[ret.hybrid]
    # The hybrid chain starts at the hybrid node itself, whose cluster is h.
    return sides[0], [c for c in sides[1] if c != h], h


def displayed_tree(net: Network, take_donor: bool) -> Network:
    """The parental tree obtained by resolving every reticulation the same way.

    ``take_donor=True`` keeps the horizontal (donor) edge and drops the tree edge into each
    hybrid; ``False`` does the reverse.  When exactly one lineage enters each hybrid, the MSNC on
    ``net`` is exactly ``gamma * MSC(displayed_tree(True)) + (1-gamma) *
    MSC(displayed_tree(False))``
    -- the analytic oracle used by the r>0 calibration test.
    """
    out = net.copy()
    for ret in reversed(out.retics):
        h, d = ret.hybrid, ret.donor
        del out.horiz_child[d]
        del out.donor_of[h]
        if take_donor:
            # Re-hang h under the donor.  The donor then has two tree children, so it stays a
            # normal binary node; it is the *old* tree parent that may be left unifurcating.
            tp = out.tree_parent[h]
            out.tree_children[tp].remove(h)
            out.tree_parent[h] = d
            out.tree_children[d].append(h)
            _suppress_degree2(out, h)
            if len(out.tree_children[tp]) == 1:
                if out.tree_parent[tp] >= 0:
                    _suppress_degree2(out, tp)
                else:
                    (new_root,) = out.tree_children[tp]  # the root lost a child: descend
                    out.tree_parent[new_root] = -1
                    out.tree_children[tp] = []
                    out.root = new_root
        else:
            _suppress_degree2(out, h)
            _suppress_degree2(out, d)
    out.retics = []
    out.consumed = set()
    out.consumed_vertices = set()
    return out


# ------------------------------------------------------------------ reading a network back in


_ENEWICK_TAG = re.compile(r"#(?:[A-Za-z]+)?(\d+)")


def from_enewick(text: str, *, gamma: str = "mean") -> Network:
    """Parse extended Newick into a :class:`Network` -- the inverse of :func:`to_enewick`.

    Written to read networks from **outside this study** (``data/N100-2``, simulated under a
    different prior by a different pipeline), so it assumes as little as possible: it does *not*
    require level-1, galled, or any cycle-disjointness property, because a truth network is only
    ever asked for its hardwired clusters and its own string.  What it does require is the class
    :class:`Network` can represent -- every node with exactly one tree parent, every hybrid with
    exactly one additional donor parent -- and it **raises** rather than silently dropping an edge
    when the input is outside it.

    **Which parent becomes the tree parent.**  In eNewick a hybrid appears once carrying its
    subtree and once as a bare ``#Hk`` tag.  The occurrence with the subtree is taken as the tree
    parent and the bare one as the donor, which is the inverse of what :func:`to_enewick` writes.
    That is a *convention*, not a recovered fact: eNewick does not distinguish the two parents
    except by where the subtree was printed, and neither does the MSNC -- ``gamma`` does.

    **Times are derived from tree edges, and a donor edge's length is whatever they imply.**
    :class:`Network` stores node *times*, so the two parent edges of a hybrid cannot both be
    honoured (the same limitation ``apply_parameters`` carries, and for the same reason).  Tree
    edges round-trip exactly; a donor edge's stated length is discarded.  In ``data/N100-2`` both
    hybrid edges are usually written at length 0, so nothing is lost there -- but that is a
    property of that dataset, not a guarantee.

    **``gamma`` is a scalar here and a vector in the file.**  ``Network`` carries one inheritance
    probability for the whole network where eNewick states one per hybrid edge.  ``gamma="mean"``
    stores the mean of the donor-side values and ``gamma="first"`` the first; the per-hybrid values
    are kept on ``net.gammas_in`` so a caller that needs them has them.  This is lossy and it does
    not affect what this function exists for: hardwired clusters do not depend on gamma, and
    CmpNets cannot read gamma at all (:func:`opc.mpl.phylonet.strip_gamma`).
    """
    s = text.strip()
    if s.startswith("[&R]") or s.startswith("[&U]"):
        s = s[4:].strip()
    s = s.rstrip().rstrip(";").strip()
    if not s.startswith("("):
        raise ValueError(f"not a Newick string: {s[:60]!r}")

    # ---- recursive descent.  Each *occurrence* becomes a slot; hybrid slots are merged after.
    slots: list[dict] = []
    pos = 0

    def annotate(idx: int, chunk: str) -> None:
        tag = _ENEWICK_TAG.search(chunk)
        name = chunk
        if tag:
            name = chunk[:tag.start()]
            rest = chunk[tag.end():]
            slots[idx]["hyb"] = int(tag.group(1))
        else:
            rest = ""
            if ":" in chunk:
                name, rest = chunk.split(":", 1)
                rest = ":" + rest
        if ":" in name:                       # ``label:len`` before a ``#`` tag
            name, tail = name.split(":", 1)
            rest = ":" + tail + rest
        fields = rest.split(":")[1:] if rest.startswith(":") else []
        slots[idx]["label"] = name.strip() or None
        slots[idx]["len"] = float(fields[0]) if fields and fields[0].strip() else 0.0
        slots[idx]["gamma"] = (float(fields[-1]) if len(fields) >= 2 and fields[-1].strip()
                               else None)

    def parse() -> int:
        nonlocal pos
        idx = len(slots)
        slots.append({"label": None, "hyb": None, "len": 0.0, "gamma": None, "kids": []})
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                slots[idx]["kids"].append(parse())
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                if pos >= len(s) or s[pos] != ")":
                    raise ValueError(f"unbalanced parentheses near offset {pos}")
                pos += 1
                break
        start = pos
        while pos < len(s) and s[pos] not in "(),":
            pos += 1
        annotate(idx, s[start:pos])
        return idx

    root_slot = parse()
    if pos != len(s):
        raise ValueError(f"trailing text after the root at offset {pos}: {s[pos:pos + 40]!r}")

    # ---- merge the two occurrences of each hybrid.  The one carrying the subtree is the body.
    body: dict[int, int] = {}
    for i, nd in enumerate(slots):
        if nd["hyb"] is None:
            continue
        if nd["hyb"] not in body or nd["kids"]:
            if nd["hyb"] in body and slots[body[nd["hyb"]]]["kids"] and nd["kids"]:
                raise ValueError(f"hybrid #{nd['hyb']} carries a subtree at both occurrences")
            body.setdefault(nd["hyb"], i)
            if nd["kids"]:
                body[nd["hyb"]] = i
    canon = {i: (body[nd["hyb"]] if nd["hyb"] is not None else i) for i, nd in enumerate(slots)}
    live = [i for i in range(len(slots)) if canon[i] == i]
    if canon[root_slot] != root_slot:
        raise ValueError("the root is a hybrid occurrence")

    parents: dict[int, list[int]] = {v: [] for v in live}
    for i in live:
        for c in slots[i]["kids"]:
            parents[canon[c]].append(i)
    counts = collections.Counter(len(parents[v]) for v in live)
    if set(counts) - {0, 1, 2}:
        raise ValueError(f"a node has {max(counts)} parents; Network represents at most 2")

    # ---- taxa, in first-appearance order, so ``labels`` is deterministic
    order = {v: i for i, v in enumerate(live)}
    leaves = [v for v in live if not slots[v]["kids"]]
    labels = [slots[v]["label"] for v in sorted(leaves, key=lambda v: order[v])]
    if any(lab is None for lab in labels):
        raise ValueError("an unlabelled leaf")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate leaf labels")
    taxon = {lab: i for i, lab in enumerate(labels)}

    # ---- renumber to a compact 0..n-1 and pick each hybrid's tree parent
    idx_of = {v: i for i, v in enumerate(live)}
    n_nodes = len(live)
    tree_parent = [-1] * n_nodes
    tree_children: list[list[int]] = [[] for _ in range(n_nodes)]
    donor_of: dict[int, int] = {}
    horiz_child: dict[int, int] = {}
    leaf_taxon: dict[int, int] = {}
    gammas_in: dict[int, float] = {}

    for v in live:
        if not slots[v]["kids"]:
            leaf_taxon[idx_of[v]] = taxon[slots[v]["label"]]

    hybrid_len: dict[int, float] = {}
    for v in live:
        ps = parents[v]
        if not ps:
            continue
        if len(ps) == 1:
            tree_parent[idx_of[v]] = idx_of[ps[0]]
            tree_children[idx_of[ps[0]]].append(idx_of[v])
            hybrid_len[idx_of[v]] = slots[v]["len"]
            continue
        # A hybrid: the occurrence that carried the subtree is under ``v`` itself; the other
        # occurrence is a bare tag whose slot id differs from ``v``.
        occ = {p: [j for j in slots[p]["kids"] if canon[j] == v] for p in ps}
        tree_p = next((p for p in ps for j in occ[p] if j == v), None)
        if tree_p is None:                       # both occurrences are bare tags
            tree_p = ps[0]
        (donor_p,) = [p for p in ps if p is not tree_p] or [ps[1]]
        tree_parent[idx_of[v]] = idx_of[tree_p]
        tree_children[idx_of[tree_p]].append(idx_of[v])
        if idx_of[donor_p] in horiz_child:
            raise ValueError(f"node {donor_p} donates to two hybrids; Network holds one")
        donor_of[idx_of[v]] = idx_of[donor_p]
        horiz_child[idx_of[donor_p]] = idx_of[v]
        for j in occ[tree_p]:
            hybrid_len[idx_of[v]] = slots[j]["len"]
        for j in occ[donor_p]:
            if slots[j]["gamma"] is not None:
                gammas_in[idx_of[v]] = float(slots[j]["gamma"])

    root = idx_of[root_slot]
    net = Network(
        n_taxa=len(labels), labels=list(labels), time=[0.0] * n_nodes,
        tree_parent=tree_parent, tree_children=tree_children, leaf_taxon=leaf_taxon,
        horiz_child=horiz_child, donor_of=donor_of, retics=[], root=root,
        gamma=0.5,
    )
    # ---- times, root-downward along tree edges only
    time = [0.0] * n_nodes
    for v in reversed(_postorder(net)):
        p = net.tree_parent[v]
        if p >= 0:
            time[v] = time[p] + hybrid_len.get(v, 0.0)
    net.time = time
    net.retics = [Reticulation(hybrid=h, donor=d, order=k, time=time[h])
                  for k, (h, d) in enumerate(sorted(donor_of.items()))]
    vals = [gammas_in[r.hybrid] for r in net.retics if r.hybrid in gammas_in]
    if vals:
        net.gamma = float(np.mean(vals)) if gamma == "mean" else float(vals[0])
    #: The per-hybrid donor-side inheritance probabilities the file stated.  ``Network.gammas``
    #: now holds them as a field; ``gammas_in`` is kept as an alias because ``opc.rl.external``
    #: reads it and because it is what A82 recorded.
    net.gammas = dict(gammas_in)
    net.gammas_in = gammas_in
    return net
