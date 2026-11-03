"""I1: ``L(Psi, gamma | G)``, and re-optimising branch lengths and gamma.

    log L = sum_tau sum_t  rho(t, G) * log P(t | Psi, gamma)

``rho`` comes from :mod:`opc.mpl.triples` and is a property of the data alone, so it is computed
once per locus set.  ``P`` is the exact MSNC triple probability of :mod:`opc.mpl.msnc3`.

Why this module exists rather than a loop over ``msnc3``
--------------------------------------------------------

The reference DP costs ~180 us per triple, so one evaluation at ``C(50,3) = 19,600`` triples is
3.5 s and an optimiser over ~60 parameters is out of reach.  But the triples fall into ~71 shapes
(:mod:`opc.mpl.trinets`), and every triple in a shape runs the *same sequence of DP operations* --
only the numbers differ.  So the DP is run once per shape with the weights as arrays over that
shape's members.  Same transitions, same state space, ``numpy`` instead of floats.

The two implementations are kept separate on purpose.  ``msnc3`` is the readable one and is what
PhyloNet and the MSNC simulator are checked against; this one is the fast one and is checked
against ``msnc3``.  That is the repo's standing convention -- every fast kernel has a slow
reference -- and it is how both of ``msnc3``'s own bugs were found.

Parameters
----------

The free parameters are **not** what :class:`~opc.simulate.networks.Network` stores.  It keeps node
*times* (so lengths are constrained to be ultrametric and time-consistent) and a single scalar
``gamma`` for the whole network.  That is right for simulation and useless for optimisation, which
needs one length per internal edge and one gamma per reticulation, unconstrained.  So a
:class:`Parameters` vector is threaded through instead, and the network supplies only the topology.

Pendant edges are excluded: a leaf's branch holds exactly one lineage, so its length cannot affect
any triple probability, and including it would hand the optimiser a flat direction per taxon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from opc.mpl import trinets
from opc.mpl.msnc3 import SEED, TOPOLOGIES
from opc.simulate import networks as N

__all__ = [
    "Parameters",
    "TripleModel",
    "shape_probabilities",
    "log_likelihood",
    "optimise",
]

_TOPO_INDEX = {frozenset(p): i for i, p in enumerate(TOPOLOGIES)}
MIN_LENGTH, MAX_LENGTH = 0.0, 20.0
#: gamma's interval is **closed**, and that is not a detail.  The plan's E2a rests on ``sup MPL``
#: being non-decreasing in ``k`` because ``gamma -> 0`` embeds the k-reticulation model in the
#: (k+1)-parameter space -- an argument that needs the endpoint to be reachable.  Measured on a
#: real instance, a floor of 1e-4 costs **1.85 log units** at the embedding, which is enough to
#: manufacture apparent monotonicity violations out of nothing.
MIN_GAMMA, MAX_GAMMA = 0.0, 1.0


@dataclass
class Parameters:
    """Branch lengths (one per internal edge) and inheritance probabilities (one per hybrid)."""

    edges: tuple[tuple[int, int], ...]  # (child, parent), in vector order
    hybrids: tuple[int, ...]  # hybrid node ids, in vector order
    lengths: np.ndarray
    gammas: np.ndarray

    @classmethod
    def of(cls, net: N.Network) -> Parameters:
        """Read the current parameters off a network.

        The horizontal donor edge is included even though the generator gives it length zero: it is
        a real branch of the model, an optimiser is entitled to lengthen it, and pinning it at zero
        would be a modelling choice made by accident.
        """
        edges, hybrids = [], []
        for v in N._postorder(net):  # noqa: SLF001
            if net.tree_parent[v] < 0:
                continue
            if net.is_hybrid(v):
                # **A hybrid may be a leaf.**  A hybrid species with one extant descendant is the
                # common biological case, and it is exactly what the *regular* version of a network
                # produces when a hybrid's cluster is a singleton (:mod:`opc.solve.closure`).  Its
                # lineage still chooses a parent, and gamma decides which side of the cycle it
                # travels, so it is a parameter like any other -- skipping leaves wholesale drops
                # it and every triple that crosses the cycle is then scored at gamma = 0.
                hybrids.append(v)
                edges += [(v, net.donor_of[v]), (v, net.tree_parent[v])]
            elif not net.is_leaf(v):
                edges.append((v, net.tree_parent[v]))
        lengths = np.array([net.edge_length(p, c) for c, p in edges], dtype=float)
        gammas = np.array([net.gamma_of(h) for h in hybrids], dtype=float)
        return cls(tuple(edges), tuple(hybrids), lengths, gammas)

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.lengths, self.gammas])

    def with_vector(self, x: np.ndarray) -> Parameters:
        n = len(self.edges)
        return Parameters(self.edges, self.hybrids, np.asarray(x[:n], dtype=float),
                          np.asarray(x[n:], dtype=float))

    def bounds(self):
        return [(MIN_LENGTH, MAX_LENGTH)] * len(self.edges) + \
               [(MIN_GAMMA, MAX_GAMMA)] * len(self.hybrids)


@dataclass
class _Group:
    """One trinet shape, and the gather that feeds it every member triple's parameters."""

    shape: trinets.Trinet
    members: np.ndarray  # int32[M] -- rows of the triple index
    edge_keys: tuple  # the trinet edges, in the order ``gather`` produces them
    gather: sp.csr_matrix  # (M * n_edges, n_params_lengths)
    gamma_index: np.ndarray  # int32[M, n_hybrids] into the gamma vector


class TripleModel:
    """Everything about a fixed topology that an optimiser can reuse across evaluations."""

    def __init__(self, net: N.Network, index: np.ndarray | None = None):
        from opc.mpl.triples import triple_index

        self.net = net
        self.index = triple_index(net.n_taxa) if index is None else np.asarray(index)
        self.params = Parameters.of(net)
        edge_pos = {e: i for i, e in enumerate(self.params.edges)}
        hybrid_pos = {h: i for i, h in enumerate(self.params.hybrids)}

        by_shape: dict[str, list] = {}
        shapes: dict[str, trinets.Trinet] = {}
        for row_i, row in enumerate(self.index):
            tri = trinets.restrict_to_triple(net, tuple(row))
            shapes.setdefault(tri.key, tri)
            by_shape.setdefault(tri.key, []).append((row_i, tri))

        self.groups: list[_Group] = []
        for key, rows in by_shape.items():
            shape = shapes[key]
            edge_keys = tuple(sorted(shape.edge_sources))
            hyb_keys = tuple(sorted(shape.gamma_source))
            m = len(rows)
            data, ind, indptr = [], [], [0]
            gamma_index = np.zeros((m, len(hyb_keys)), dtype=np.int32)
            for i, (_, tri) in enumerate(rows):
                for e in edge_keys:
                    # A trinet edge below a *plain* leaf carries exactly one lineage, so its length
                    # can never affect a triple probability -- the DP never applies a coalescent to
                    # it -- and it is the only place a pendant network edge can appear, which is
                    # why ``Parameters`` has no entry for one.  Left as an empty row (length 0).
                    #
                    # A **hybrid** leaf is different: its lineage splits and then traverses one of
                    # two branches inside a cycle, where it can meet another lineage, so both of
                    # its in-edges are real parameters.
                    if e[0] not in shape.leaf_of or len(shape.parents[e[0]]) == 2:
                        for src in tri.edge_sources[e]:
                            # A pendant network edge may only feed a trinet edge that is itself
                            # below a leaf -- unless that leaf is a *hybrid*, whose two in-edges
                            # are real parameters.  Anything else means the restriction has put a
                            # length where the DP cannot use it.
                            if src[0] in net.leaf_taxon and not net.is_hybrid(src[0]):
                                raise RuntimeError(
                                    f"pendant network edge {src} feeds trinet edge {e}"
                                )
                            ind.append(edge_pos[src])
                            data.append(1.0)
                    indptr.append(len(ind))
                for j, h in enumerate(hyb_keys):
                    gamma_index[i, j] = hybrid_pos[tri.gamma_source[h]]
            gather = sp.csr_matrix(
                (np.asarray(data), np.asarray(ind), np.asarray(indptr)),
                shape=(m * len(edge_keys), len(self.params.edges)),
            )
            self.groups.append(_Group(
                shape=shape,
                members=np.asarray([r for r, _ in rows], dtype=np.int32),
                edge_keys=edge_keys, gather=gather, gamma_index=gamma_index,
            ))

    @property
    def n_shapes(self) -> int:
        return len(self.groups)

    def probabilities(self, params: Parameters | None = None) -> np.ndarray:
        """``P[i, :]`` for every triple of :attr:`index`, in ``(ab|c, ac|b, bc|a)`` order."""
        p = params or self.params
        out = np.empty((len(self.index), 3))
        for g in self.groups:
            m = len(g.members)
            lengths = (g.gather @ p.lengths).reshape(m, len(g.edge_keys))
            gammas = p.gammas[g.gamma_index] if g.gamma_index.size else g.gamma_index
            out[g.members] = _run_group(g, lengths, gammas)
        return out

    def log_likelihood(self, obs: np.ndarray, params: Parameters | None = None,
                       floor: float = 1e-300) -> float:
        """``sum rho * log P``.  ``obs`` is the ``(n_triples, 3)`` count matrix."""
        probs = self.probabilities(params)
        return float((np.asarray(obs) * np.log(np.maximum(probs, floor))).sum())


# ------------------------------------------------------------------------ the vectorised DP


def _coal2(dt: np.ndarray) -> np.ndarray:
    return -np.expm1(-dt)


def _coal3(dt: np.ndarray):
    e1, e3 = np.exp(-dt), np.exp(-3.0 * dt)
    p2 = 1.5 * e1 * -np.expm1(-2.0 * dt)
    return e3, p2, np.maximum(0.0, 1.0 - e3 - p2)


def _run_group(g: _Group, lengths: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    return shape_probabilities(g.shape, g.edge_keys, lengths, gammas)


def shape_probabilities(shape, edge_keys, lengths: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """``P[b, :]`` for a batch of parameter settings of **one** trinet shape.

    The msnc3 post-order with the state weights as arrays, mirroring
    :func:`opc.mpl.msnc3.triple_probabilities` transition for transition.  The state space is a
    property of the shape, so it is identical for every member of the batch and the arrays are
    never ragged.

    Exposed rather than kept private because :mod:`opc.mpl.profile` needs exactly this -- a shape's
    probabilities at arbitrary parameters, decoupled from any particular network -- and a third
    implementation of the same transitions is the last thing this package needs.

    ``lengths`` is ``(B, len(edge_keys))`` and ``gammas`` is ``(B, n_hybrids)`` ordered by
    ``sorted(shape.gamma_source)``.
    """
    m = lengths.shape[0]
    col = {e: i for i, e in enumerate(edge_keys)}
    hyb_col = {h: i for i, h in enumerate(sorted(shape.gamma_source))}

    # Seeded on a sentinel slot, not on slot 0, because a leaf may be a **hybrid** and then has
    # two in-edges to choose between -- see :class:`opc.mpl.msnc3`'s ``SEED``.
    seed = frozenset(((v, SEED), frozenset({shape.leaf_of[v]})) for v in shape.leaf_of)
    dist: dict = {(seed, None): np.ones(m)}

    for v in shape.postorder:
        if v in shape.leaf_of and len(shape.parents[v]) < 2:
            dist = _move(dist, {(v, SEED)}, (v, 0))
            continue
        if v in shape.leaf_of:
            from_edges = {(v, SEED)}
        else:
            from_edges = {(c, s) for c in shape.children.get(v, ())
                          for s, p in enumerate(shape.parents[c]) if p == v}
        if not shape.parents[v]:
            return _finish(dist, m)
        if len(shape.parents[v]) == 2:
            donor, tree = (v, 0), (v, 1)
            dist = _split(dist, from_edges, donor, tree, gammas[:, hyb_col[v]])
            dist = _coalesce(dist, {donor: lengths[:, col[donor]],
                                    tree: lengths[:, col[tree]]})
        else:
            up = (v, 0)
            dist = _move(dist, from_edges, up)
            dist = _coalesce(dist, {up: lengths[:, col[up]]})
    raise RuntimeError("the post-order pass never reached the root")


def _add(out: dict, state, p: np.ndarray) -> None:
    prev = out.get(state)
    out[state] = p if prev is None else prev + p


def _coalesce(dist: dict, durations: dict) -> dict:
    out: dict = {}
    for (placement, topo), w in dist.items():
        by_edge: dict = {}
        for edge, block in placement:
            if edge in durations:
                by_edge.setdefault(edge, []).append(block)
        busy = [e for e, bs in by_edge.items() if len(bs) >= 2]
        if not busy:
            _add(out, (placement, topo), w)
            continue
        (edge,) = busy
        dt = durations[edge]
        blocks = by_edge[edge]
        others = frozenset(x for x in placement if x[0] != edge)
        if len(blocks) == 2:
            pc = _coal2(dt)
            _add(out, (placement, topo), w * (1.0 - pc))
            merged = frozenset(blocks[0] | blocks[1])
            t = topo if topo is not None else _TOPO_INDEX[merged]
            _add(out, (others | {(edge, merged)}, t), w * pc)
            continue
        p3, p2, p1 = _coal3(dt)
        _add(out, (placement, topo), w * p3)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            merged = frozenset(blocks[i] | blocks[j])
            (rest,) = (blocks[k] for k in range(3) if k not in (i, j))
            t = _TOPO_INDEX[merged]
            _add(out, (others | {(edge, merged), (edge, rest)}, t), w * p2 / 3.0)
            _add(out, (others | {(edge, frozenset({0, 1, 2}))}, t), w * p1 / 3.0)
    return out


def _move(dist: dict, from_edges: set, to_edge) -> dict:
    out: dict = {}
    for (placement, topo), w in dist.items():
        moved = frozenset(
            (to_edge, block) if edge in from_edges else (edge, block) for edge, block in placement
        )
        _add(out, (moved, topo), w)
    return out


def _split(dist: dict, from_edges: set, donor_edge, tree_edge, gamma: np.ndarray) -> dict:
    from itertools import product

    out: dict = {}
    for (placement, topo), w in dist.items():
        here = [(e, b) for e, b in placement if e in from_edges]
        rest = frozenset(x for x in placement if x[0] not in from_edges)
        if not here:
            _add(out, (rest, topo), w)
            continue
        for choice in product((donor_edge, tree_edge), repeat=len(here)):
            p = w.copy()
            for target in choice:
                p = p * (gamma if target is donor_edge else 1.0 - gamma)
            moved = rest | {(t, b) for (_, b), t in zip(here, choice, strict=True)}
            _add(out, (moved, topo), p)
    return out


def _finish(dist: dict, m: int) -> np.ndarray:
    out = np.zeros((m, 3))
    for (placement, topo), w in dist.items():
        blocks = [b for _, b in placement]
        if topo is not None:
            out[:, topo] += w
        elif len(blocks) == 3:
            out += (w / 3.0)[:, None]
        elif len(blocks) == 2:
            out[:, _TOPO_INDEX[frozenset(blocks[0] | blocks[1])]] += w
        else:
            raise RuntimeError("a lone block reached the root with no topology")
    return out


# ------------------------------------------------------------------------------- optimisation


def log_likelihood(model: TripleModel, obs: np.ndarray, params: Parameters | None = None) -> float:
    return model.log_likelihood(obs, params)


def optimise(model: TripleModel, obs: np.ndarray, *, start: Parameters | None = None,
             maxiter: int = 200, restarts: int = 1, seed: int = 0):
    """Re-optimise branch lengths and gamma.  Returns ``(best_params, best_log_likelihood)``.

    L-BFGS-B with numerical gradients over ~60 bounded parameters.  ``restarts > 1`` perturbs the
    start, which matters for E2a: a *false* violation of k-monotonicity is an optimiser failure,
    not a criterion property, so the optimiser has to be given a fair chance before a violation is
    reported as one.
    """
    from scipy.optimize import minimize

    base = start or model.params
    obs = np.asarray(obs)
    bounds = base.bounds()
    rng = np.random.default_rng(seed)

    def negative(x):
        return -model.log_likelihood(obs, base.with_vector(x))

    best_x, best_f = None, np.inf
    x0 = base.vector.copy()
    for attempt in range(max(1, restarts)):
        start_x = x0 if attempt == 0 else np.clip(
            x0 * rng.lognormal(0.0, 0.4, size=x0.shape),
            [b[0] for b in bounds], [b[1] for b in bounds],
        )
        res = minimize(negative, start_x, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)
    return base.with_vector(best_x), -best_f
