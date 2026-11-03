"""W1 -- the rooted-triple MPL, in torch: differentiable, and batched over networks.

``RL_PLAN.md`` §1.1 needs a terminal reward under **0.5 s** (G-R1), where one reward is "build the
network a cluster system stands for, re-optimise (lambda, gamma), return sup MPL".  Measured on
`haldane` at n = 50 with the landed numpy path: ``TripleModel.__init__`` **1.60 s**, one
``log_likelihood`` **10.1 ms**, and one re-optimisation 1,742-10,927 likelihood calls (A53).  A
GRPO step wants 128 of those at once.

**What this module is, and is not.**  It is *not* a reimplementation of the triple DP.  §1.2
proposes compiling ``P(t | Psi, lambda, gamma)`` to a static sparse contraction; that compilation
already exists as :func:`opc.mpl.likelihood.shape_probabilities`, which runs the ``msnc3``
post-order once per *shape* with the state weights as arrays over that shape's member triples --
41 shapes for 19,600 triples on a k = 2 cell of the S50 grid.  What this module does is take that
same DP and change three things about how it is evaluated:

1. **torch instead of numpy, so the gradient is analytic.**  The optimiser's cost is not the
   likelihood, it is the *numerical* gradient: L-BFGS-B over 52-56 bounded parameters spends
   ``n_params + 1`` evaluations per iteration, which is where A53's 1,742-10,927 calls come from.
   Autograd replaces that with one forward and one backward.  The DP is a fixed sequence of
   elementwise operations on ``(B,)`` vectors -- its control flow is a property of the *shape*,
   never of the data -- so it ports transition for transition and traces cleanly.
2. **The batch dimension carries networks as well as triples.**  ``shape_probabilities`` is already
   batched over the member triples of one shape of one network.  Grouping by shape key *across* a
   batch of networks makes ``B`` the total ``(network, triple)`` count in that shape, and the code
   does not change.  128 networks x 19,600 triples is ~2.5 M rows spread over a few dozen groups.
3. **The parameter vectors are concatenated, not padded.**  Networks in a batch have different
   numbers of internal edges and hybrids, so a rectangular ``(B, P)`` layout would be ragged.  The
   lengths of every network live end to end in one flat tensor and each network's gather is shifted
   by its offset, which is exactly what a block-diagonal sparse product does anyway.

The gather itself is stored as a **segment sum**, not as a sparse matrix.  A trinet edge is the sum
of the handful of original edges the restriction suppressed into it, so ``scipy``'s CSR is a 0/1
matrix with a few nonzeros per row; ``(row, col)`` pairs plus ``index_add`` compute the same thing
with dense indexing only.  That matters twice: ``torch.sparse_csr_tensor`` is beta and warns as
much, and a segment sum has a first-class CUDA kernel and an exact autograd rule where the sparse
path's does not.

**The floor is deliberate and matches numpy.**  ``log_likelihood`` clamps ``P`` at ``floor``
before the log, as :meth:`opc.mpl.likelihood.TripleModel.log_likelihood` does.  ``torch.clamp_min``
propagates zero gradient through a clamped entry, which is the correct subgradient and also the
only stable choice: a triple whose probability underflows must not send an infinite gradient back
into the branch lengths.

**gamma's interval is closed and that is load-bearing** (``likelihood.MIN_GAMMA``): E2a's
``sup MPL`` monotonicity in ``k`` rests on ``gamma -> 0`` embedding the k-reticulation model in the
(k+1)-parameter space, and a floor of 1e-4 was measured to cost 1.85 log units at the embedding.
The projection here therefore clamps to ``[0, 1]`` and not to ``[eps, 1-eps]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from opc.mpl import likelihood as L
from opc.mpl.msnc3 import SEED

__all__ = [
    "shape_probabilities_torch",
    "BatchedTripleModel",
    "log_likelihood_from_probs",
    "ProfileResult",
    "mom_init_length",
    "profile_mpl",
    "profile_mpl_lbfgs",
]


def _coal2(dt: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-dt)


def _coal3(dt: torch.Tensor):
    e1, e3 = torch.exp(-dt), torch.exp(-3.0 * dt)
    p2 = 1.5 * e1 * -torch.expm1(-2.0 * dt)
    # ``np.maximum(0.0, ...)`` in the numpy path.  The clamp is a real subgradient, not a hack:
    # ``1 - e3 - p2`` is non-negative in exact arithmetic and only goes negative on round-off.
    return e3, p2, torch.clamp_min(1.0 - e3 - p2, 0.0)


def _add(out: dict, state, p: torch.Tensor) -> None:
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
            t = topo if topo is not None else L._TOPO_INDEX[merged]  # noqa: SLF001
            _add(out, (others | {(edge, merged)}, t), w * pc)
            continue
        p3, p2, p1 = _coal3(dt)
        _add(out, (placement, topo), w * p3)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            merged = frozenset(blocks[i] | blocks[j])
            (rest,) = (blocks[k] for k in range(3) if k not in (i, j))
            t = L._TOPO_INDEX[merged]  # noqa: SLF001
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


def _split(dist: dict, from_edges: set, donor_edge, tree_edge, gamma: torch.Tensor) -> dict:
    from itertools import product

    out: dict = {}
    for (placement, topo), w in dist.items():
        here = [(e, b) for e, b in placement if e in from_edges]
        rest = frozenset(x for x in placement if x[0] not in from_edges)
        if not here:
            _add(out, (rest, topo), w)
            continue
        for choice in product((donor_edge, tree_edge), repeat=len(here)):
            p = w
            for target in choice:
                p = p * (gamma if target is donor_edge else 1.0 - gamma)
            moved = rest | {(t, b) for (_, b), t in zip(here, choice, strict=True)}
            _add(out, (moved, topo), p)
    return out


def _finish(dist: dict, m: int, ref: torch.Tensor) -> torch.Tensor:
    """The root: read each surviving state off as a topology.  Never accumulates in place.

    The numpy path writes into a preallocated ``(m, 3)`` array.  Doing that under autograd would
    put an in-place write on the tape for no gain, so the three columns are summed as lists and
    stacked -- same arithmetic, same order of accumulation within a column.
    """
    cols: list[list[torch.Tensor]] = [[], [], []]
    for (placement, topo), w in dist.items():
        blocks = [b for _, b in placement]
        if topo is not None:
            cols[topo].append(w)
        elif len(blocks) == 3:
            third = w / 3.0
            for c in cols:
                c.append(third)
        elif len(blocks) == 2:
            cols[L._TOPO_INDEX[frozenset(blocks[0] | blocks[1])]].append(w)  # noqa: SLF001
        else:
            raise RuntimeError("a lone block reached the root with no topology")
    zero = torch.zeros(m, dtype=ref.dtype, device=ref.device)
    return torch.stack([torch.stack(c).sum(0) if c else zero for c in cols], dim=1)


def shape_probabilities_torch(shape, edge_keys, lengths: torch.Tensor,
                              gammas: torch.Tensor) -> torch.Tensor:
    """``P[b, :]`` for a batch of parameter settings of **one** trinet shape.

    Transition for transition, :func:`opc.mpl.likelihood.shape_probabilities`.  ``lengths`` is
    ``(B, len(edge_keys))`` and ``gammas`` is ``(B, n_hybrids)`` ordered by
    ``sorted(shape.gamma_source)`` -- the same contract, so the two can be diffed directly.
    """
    m = lengths.shape[0]
    col = {e: i for i, e in enumerate(edge_keys)}
    hyb_col = {h: i for i, h in enumerate(sorted(shape.gamma_source))}

    seed = frozenset(((v, SEED), frozenset({shape.leaf_of[v]})) for v in shape.leaf_of)
    one = torch.ones(m, dtype=lengths.dtype, device=lengths.device)
    dist: dict = {(seed, None): one}

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
            return _finish(dist, m, lengths)
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


@dataclass
class _TorchGroup:
    """One shape, pooled across every network in the batch that induces it."""

    shape: object
    edge_keys: tuple
    rows: torch.Tensor          # int64[M]  -- destination rows in the flat (N*T, 3) output
    seg_out: torch.Tensor       # int64[nnz] -- destination slot in the flat (M * n_edges,) lengths
    seg_in: torch.Tensor        # int64[nnz] -- source index into the concatenated lengths
    gamma_index: torch.Tensor   # int64[M, H] into the concatenated gammas
    n_members: int


class BatchedTripleModel:
    """``N`` networks' :class:`opc.mpl.likelihood.TripleModel`\\ s, re-grouped by shape.

    The parameter vectors are **concatenated**, not padded: ``lengths`` is
    ``(sum_b n_edges(b),)`` and ``gammas`` is ``(sum_b n_hybrids(b),)``.  :attr:`length_slices` and
    :attr:`gamma_slices` say where each network's block starts, which is what an initialiser or a
    projection needs; nothing else in the class cares, because every gather index is already
    absolute.

    ``probabilities`` returns ``(N, T, 3)`` in the ``(ab|c, ac|b, bc|a)`` order
    :mod:`opc.mpl.msnc3` fixes, with ``T`` the length of the shared triple index.  Every network in
    a batch must therefore share one triple index -- they do, because they share a taxon set.
    """

    def __init__(self, models, *, device=None, dtype=torch.float64):
        if not models:
            raise ValueError("BatchedTripleModel needs at least one TripleModel")
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.n_networks = len(models)
        self.n_triples = int(len(models[0].index))
        for m in models:
            if len(m.index) != self.n_triples:
                raise ValueError(
                    "every network in a batch must share one triple index; got "
                    f"{len(m.index)} against {self.n_triples}")

        n_len = [int(len(m.params.edges)) for m in models]
        n_gam = [int(len(m.params.hybrids)) for m in models]
        loff = np.concatenate([[0], np.cumsum(n_len)]).astype(np.int64)
        goff = np.concatenate([[0], np.cumsum(n_gam)]).astype(np.int64)
        self.length_slices = [(int(loff[b]), int(loff[b + 1])) for b in range(len(models))]
        self.gamma_slices = [(int(goff[b]), int(goff[b + 1])) for b in range(len(models))]
        self.n_lengths, self.n_gammas = int(loff[-1]), int(goff[-1])

        pooled: dict[str, list] = {}
        for b, model in enumerate(models):
            for g in model.groups:
                pooled.setdefault(g.shape.key, []).append((b, g))

        self.groups: list[_TorchGroup] = []
        for _key, entries in pooled.items():
            shape = entries[0][1].shape
            edge_keys = entries[0][1].edge_keys
            n_e = len(edge_keys)
            rows, gidx, seg_out, seg_in = [], [], [], []
            slot_base = 0
            for b, g in entries:
                if g.edge_keys != edge_keys:
                    raise RuntimeError(
                        f"shape {shape.key!r} produced two different edge orders; the group key "
                        "no longer determines the gather layout")
                rows.append(np.asarray(g.members, dtype=np.int64) + b * self.n_triples)
                csr = g.gather.tocsr()
                if csr.data.size and not np.all(csr.data == 1.0):
                    raise RuntimeError(
                        "the gather is no longer a 0/1 incidence matrix, so a segment sum is not "
                        "the same operation as the sparse product it replaces")
                counts = np.diff(csr.indptr)
                seg_out.append(np.repeat(np.arange(csr.shape[0], dtype=np.int64), counts)
                               + slot_base)
                seg_in.append(csr.indices.astype(np.int64) + self.length_slices[b][0])
                slot_base += csr.shape[0]
                gi = (np.asarray(g.gamma_index, dtype=np.int64) + self.gamma_slices[b][0]
                      if g.gamma_index.size else
                      np.zeros((len(g.members), 0), dtype=np.int64))
                gidx.append(gi)
            rows_t = torch.as_tensor(np.concatenate(rows), device=self.device)
            m_total = int(rows_t.numel())
            if slot_base != m_total * n_e:
                raise RuntimeError(f"gather slots {slot_base} != members x edges {m_total * n_e}")
            self.groups.append(_TorchGroup(
                shape=shape, edge_keys=edge_keys, rows=rows_t,
                seg_out=torch.as_tensor(np.concatenate(seg_out), device=self.device),
                seg_in=torch.as_tensor(np.concatenate(seg_in), device=self.device),
                gamma_index=torch.as_tensor(np.concatenate(gidx), device=self.device),
                n_members=m_total,
            ))

    # ------------------------------------------------------------------ evaluation

    def probabilities(self, lengths: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """``(N, T, 3)`` triple probabilities.  Differentiable in both arguments."""
        if lengths.shape[-1] != self.n_lengths:
            raise ValueError(f"lengths has {lengths.shape[-1]}, expected {self.n_lengths}")
        if gammas.shape[-1] != self.n_gammas:
            raise ValueError(f"gammas has {gammas.shape[-1]}, expected {self.n_gammas}")
        flat = torch.zeros(self.n_networks * self.n_triples, 3,
                           dtype=lengths.dtype, device=lengths.device)
        for g in self.groups:
            n_e = len(g.edge_keys)
            ell = torch.zeros(g.n_members * n_e, dtype=lengths.dtype, device=lengths.device)
            ell = ell.index_add(0, g.seg_out, lengths[g.seg_in]).reshape(g.n_members, n_e)
            gam = (gammas[g.gamma_index] if g.gamma_index.numel()
                   else gammas.new_zeros((g.n_members, 0)))
            flat = flat.index_copy(0, g.rows, shape_probabilities_torch(g.shape, g.edge_keys,
                                                                       ell, gam))
        return flat.reshape(self.n_networks, self.n_triples, 3)

    def log_likelihood(self, obs: torch.Tensor, lengths: torch.Tensor, gammas: torch.Tensor,
                       *, floor: float = 1e-300) -> torch.Tensor:
        """``sum_t rho log P`` per network -- ``(N,)``.  ``obs`` is ``(T, 3)`` or ``(N, T, 3)``."""
        probs = self.probabilities(lengths, gammas)
        return log_likelihood_from_probs(probs, obs, floor=floor)

    # ------------------------------------------------------------------ parameters

    def start_vectors(self, models) -> tuple[torch.Tensor, torch.Tensor]:
        """The concatenated ``(lengths, gammas)`` each model's own ``Parameters`` carries.

        A *warm* start, and therefore not what a group-normalised reward should be profiled from --
        RL_PLAN §1.2 asks for a canonical data-driven init precisely so truncation bias is
        homogeneous across a rollout group, and A60 measured this particular start (a
        cluster-derived network's rank times) as the **bad basin**.  Exposed because the parity
        tests need it: it is the one start at which the torch and numpy paths must agree exactly.
        """
        return (
            torch.as_tensor(np.concatenate([m.params.lengths for m in models]),
                            dtype=self.dtype, device=self.device),
            torch.as_tensor(np.concatenate([m.params.gammas for m in models]),
                            dtype=self.dtype, device=self.device),
        )


def log_likelihood_from_probs(probs: torch.Tensor, obs: torch.Tensor,
                              *, floor: float = 1e-300) -> torch.Tensor:
    """``sum rho log P`` over the triple axis, clamped as the numpy path clamps."""
    if obs.dim() == 2:
        obs = obs.unsqueeze(0)
    return (obs * torch.log(torch.clamp_min(probs, floor))).sum(dim=(-2, -1))


# --------------------------------------------------------------------------- profiling


@dataclass
class ProfileResult:
    """``sup MPL`` per network, and the two numbers a truncated optimiser owes."""

    value: torch.Tensor          # (N,) best log-likelihood found
    lengths: torch.Tensor        # concatenated best lengths, (P_total,)
    gammas: torch.Tensor         # concatenated best gammas, (G_total,)
    n_iters: int
    restarts: int
    seconds: float
    #: ``best - first restart``.  A60's quantity, measured per call rather than assumed: on the
    #: search tier one cell in four moved by 10,905 log units on the starting vector alone.
    restart_gain: torch.Tensor   # (N,)
    #: Largest ``|Delta logL|`` over the last tenth of the schedule, per network.  A truncated
    #: optimum is a valid *lower* bound on ``sup MPL`` (RL_PLAN §1.2), so this is how far the
    #: bound might still be from tight -- not an error, but not nothing either.
    tail_movement: torch.Tensor  # (N,)


def mom_init_length(models, obs, groups=None) -> float:
    """A canonical branch length from the data alone -- no warm start, no neighbour.

    RL_PLAN §1.2 asks for a *method-of-moments* initialisation rather than a warm start,
    specifically so that truncation bias is **homogeneous** across a rollout group: a
    group-relative advantage is only meaningful if every member got the same quality of fit.  A
    warm start from a neighbour gives the neighbour's basin to whichever rollout is nearest it.

    :func:`opc.mpl.profile.tree_profile` already carries the closed form for one triple under the
    matching tree shape -- ``e^{-x} = 3(b + c) / (2(a + b + c))`` -- so ``x`` is the internal
    branch of that triple's 3-taxon tree, in coalescent units, straight from ``rho``.  A trinet
    edge is the *sum* of the network edges the restriction suppressed into it, so the per-edge
    initialiser is ``median(x)`` divided by how many network edges a trinet edge typically pools;
    that count is already in the gather and is not guessed.
    """
    r = np.asarray(obs, dtype=float)
    if r.ndim == 3:
        r = r[0]
    total = r.sum(axis=1)
    keep = total > 0
    if not keep.any():
        return 1.0
    a = r[keep].max(axis=1)
    u = np.clip(3.0 * (total[keep] - a) / (2.0 * total[keep]), 1e-12, 1.0)
    x = float(np.median(-np.log(u)))
    per_edge = 1.0
    if groups:
        counts = [np.bincount(g.seg_out.cpu().numpy()) for g in groups if g.seg_out.numel()]
        if counts:
            pooled = np.concatenate(counts)
            per_edge = max(1.0, float(np.median(pooled[pooled > 0])))
    return float(np.clip(x / per_edge, 1e-3, L.MAX_LENGTH))



def _initial_vectors(bm, models, obs, restarts: int, n: int, init: str, seed: int):
    """The starting ``(lengths, gammas)`` for every restart, shared by both optimisers.

    Restart ``r != 0`` perturbs multiplicatively exactly as ``likelihood.optimise`` does --
    lognormal(0, 0.4) clipped to the bounds -- so the spread of starts is the one A60 measured, and
    gamma is redrawn uniformly on its closed interval rather than perturbed, because 0.5 is a
    placeholder rather than an estimate (``closure.network_from_hasse``).
    """
    warm_l, warm_g = bm.start_vectors(list(models) * restarts)
    if init == "warm":
        ell, gam = warm_l.clone(), warm_g.clone()
    else:
        lam = mom_init_length(models, obs, bm.groups) if init == "mom" else 1.0
        ell = torch.full_like(warm_l, lam)
        gam = torch.full_like(warm_g, 0.5)
    if restarts > 1:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        for r in range(1, restarts):
            ls, le = bm.length_slices[r * n][0], bm.length_slices[(r + 1) * n - 1][1]
            gs, ge = bm.gamma_slices[r * n][0], bm.gamma_slices[(r + 1) * n - 1][1]
            noise = torch.exp(0.4 * torch.randn(le - ls, generator=gen)).to(ell.device, ell.dtype)
            ell[ls:le] = torch.clamp(ell[ls:le] * noise, L.MIN_LENGTH, L.MAX_LENGTH)
            if ge > gs:
                gam[gs:ge] = torch.rand(ge - gs, generator=gen).to(gam.device, gam.dtype)
    return ell, gam

def profile_mpl(models, obs, *, device=None, dtype=torch.float64, restarts: int = 4,
                iters: int = 300, lr: float = 0.05, seed: int = 0, init: str = "mom",
                graph: bool = True,
                batched: "BatchedTripleModel | None" = None) -> ProfileResult:
    """``sup_{lambda, gamma} MPL`` for every network in ``models``, jointly, on one device.

    **Projected Adam, not L-BFGS, and the reason is exactness rather than taste.**  The objective
    here is ``sum_b logL_b`` over networks whose parameter blocks are disjoint.  Adam's update is
    elementwise, so descending the sum is *identical* to descending each network separately.
    L-BFGS is not: its curvature estimate is built from the whole gradient vector, so batching N
    networks into one L-BFGS would let one network's curvature steer another's step, and the line
    search would be global.  Correct batched L-BFGS needs a per-network two-loop recursion; Adam
    needs nothing, and the evaluator is launch-bound anyway (measured: a forward+backward at n = 50
    costs the same 130 ms at 8 networks as at 128), so iterations are nearly free and matrix-free
    descent is the cheap side of the trade.

    **Restarts cost almost nothing here, and A60 says they are not optional.**  ``restarts`` is
    implemented by *tiling the batch*, not by looping: the evaluator's cost is dominated by kernel
    launches rather than by rows, so four starts of 128 networks cost roughly what one start of 128
    costs.  On the search tier, where restarts were a serial 85 s each, they were skipped and one
    pilot cell in four paid 10,905 log units for it.

    Bounds are enforced by projection, and ``gamma``'s interval is **closed** -- see
    ``likelihood.MIN_GAMMA``: E2a's monotonicity in ``k`` needs ``gamma = 0`` to be reachable, and
    a 1e-4 floor was measured to cost 1.85 log units at the embedding.

    **``graph=True`` captures the forward and backward as a CUDA graph.**  The DP's operation
    sequence is a property of the *shapes*, not of the data, so it is identical on every iteration
    and capturable.  That matters because the cost of this evaluator tracks the number of pooled
    shape groups rather than the number of rows: 128 *distinct* terminal networks at n = 50 induce
    **1,282** groups where 128 clones of one network induce 41, so a benchmark built from clones
    understates a real batch by a factor of thirty in launch count.  The capture removes that term
    rather than reducing it.  It is skipped silently on CPU and falls back to eager if capture
    raises.
    """
    import time as _time

    if init not in ("mom", "warm", "constant"):
        raise ValueError(f"unknown init {init!r}")
    n = len(models)
    restarts = max(1, int(restarts))
    bm = batched or BatchedTripleModel(list(models) * restarts, device=device, dtype=dtype)
    if bm.n_networks != n * restarts:
        raise ValueError("the supplied BatchedTripleModel does not match models x restarts")
    ot = torch.as_tensor(np.asarray(obs), dtype=dtype, device=bm.device)
    if ot.dim() == 3:
        # Per-network counts.  Networks are tiled ``restarts`` times, so the counts must be too --
        # broadcasting a length-N tensor against N*R rows would silently pair rollout i's topology
        # with rollout (i mod N)'s data on every restart but the first.
        if ot.shape[0] != n:
            raise ValueError(f"obs has {ot.shape[0]} rows, expected one per network ({n})")
        ot = ot.repeat(restarts, 1, 1)

    ell, gam = _initial_vectors(bm, models, obs, restarts, n, init, seed)
    ell.requires_grad_(True)
    gam.requires_grad_(True)
    opt = torch.optim.Adam([ell, gam], lr=lr)
    t0 = _time.perf_counter()
    tail_from = max(1, int(iters * 0.9))
    best = torch.full((bm.n_networks,), -float("inf"), dtype=dtype, device=bm.device)
    best_l, best_g = ell.detach().clone(), gam.detach().clone()
    prev = None
    tail = torch.zeros(bm.n_networks, dtype=dtype, device=bm.device)
    step = _graphed_step(bm, ot, ell, gam) if (graph and bm.device.type == "cuda") else None
    for it in range(iters):
        if step is not None:
            val = step()
        else:
            opt.zero_grad(set_to_none=True)
            val = bm.log_likelihood(ot, ell, gam)
            (-val.sum()).backward()
        # **Record before stepping.**  ``val`` was evaluated at the *current* ``(ell, gam)``, so
        # the incumbent has to be banked here; ``opt.step()`` below moves the parameters, and
        # recording them afterwards pairs each value with the point one Adam step past the one
        # that produced it.  That was invisible for as long as nothing read the parameters back --
        # ``value`` is a genuine achieved log-likelihood either way -- and became a real error the
        # moment ``fit_parameters`` began writing them onto a network (A81).  Caught by
        # ``test_fit_parameters_gives_each_network_its_own_block``, where the numpy kernel scored
        # the returned block 3.77 log units *away* from the value reported beside it.
        with torch.no_grad():
            # ``.clone()``, and it is not decoration.  On the CUDA-graph path ``val`` is the
            # *same tensor* every replay -- the graph writes through its captured output address --
            # so ``prev = v`` without a copy aliases the current values and ``v - prev`` is
            # identically zero.  Measured: ``tail_movement`` came back exactly 0.0 on both arms of
            # the probe, which reads as "fully converged" and is in fact "never compared".
            v = val.detach().clone()
            better = v > best
            best = torch.where(better, v, best)
            for lo, hi in _blocks(bm.length_slices, better, bm.n_networks):
                best_l[lo:hi] = ell.detach()[lo:hi]
            for lo, hi in _blocks(bm.gamma_slices, better, bm.n_networks):
                best_g[lo:hi] = gam.detach()[lo:hi]
            if it >= tail_from and prev is not None:
                tail = torch.maximum(tail, (v - prev).abs())
            prev = v
        opt.step()
        with torch.no_grad():
            ell.clamp_(L.MIN_LENGTH, L.MAX_LENGTH)
            gam.clamp_(L.MIN_GAMMA, L.MAX_GAMMA)
    seconds = _time.perf_counter() - t0

    by_restart = best.reshape(restarts, n)
    value, which = by_restart.max(dim=0)
    gain = value - by_restart[0]
    # Gather the winning restart's parameter block for each network.
    out_l = torch.cat([best_l[slice(*bm.length_slices[int(which[b]) * n + b])] for b in range(n)])
    out_g = torch.cat([best_g[slice(*bm.gamma_slices[int(which[b]) * n + b])] for b in range(n)])
    return ProfileResult(
        value=value, lengths=out_l, gammas=out_g, n_iters=iters, restarts=restarts,
        seconds=seconds, restart_gain=gain,
        tail_movement=tail.reshape(restarts, n).gather(0, which.unsqueeze(0)).squeeze(0),
    )


def fit_parameters(models, obs, *, device=None, dtype=torch.float64, restarts: int = 2,
                   iters: int = 300, lr: float = 0.05, seed: int = 0, init: str = "mom",
                   graph: bool = True) -> tuple[list[L.Parameters], ProfileResult]:
    """Fit every model's ``(lengths, gammas)`` jointly on one device -> numpy :class:`Parameters`.

    The batched, **autograd-gradient** counterpart of :func:`opc.mpl.likelihood.optimise`, for the
    callers that need the fitted *parameters* back rather than only ``sup MPL``.
    :func:`opc.mpl.likelihood.apply_parameters` writing a fit onto a network before a
    length-weighted topological comparison (A80) is the caller that motivated it.

    **The saving is the gradient, not the device.**  ``likelihood.optimise`` is L-BFGS-B with a
    *numerical* gradient, so one descent step costs ``n_params + 1`` forward evaluations -- 52 at
    n = 50 -- and a 120-iteration two-restart fit of a single network was measured at **10,608**
    likelihood calls and 40.7 s, of which **98.3%** was inside the DP and 1.7% was scipy.  Autograd
    gives that gradient for one forward and one backward, and because this evaluator's cost tracks
    pooled *shape groups* rather than rows, fitting a cell's networks together costs little more
    than fitting one of them.

    ``init`` defaults to ``"mom"`` rather than to the warm start ``likelihood.optimise`` uses:
    A60 measured a cluster-derived network's own rank-time lengths as **the bad basin**, and that
    is exactly the start a naive writeback would take.

    Returns the parameters in ``models`` order and the :class:`ProfileResult` beside them, so a
    caller can report ``restart_gain`` and ``tail_movement`` rather than assume them away.
    """
    res = profile_mpl(models, obs, device=device, dtype=dtype, restarts=restarts, iters=iters,
                      lr=lr, seed=seed, init=init, graph=graph)
    ell = res.lengths.detach().cpu().numpy().astype(float)
    gam = res.gammas.detach().cpu().numpy().astype(float)
    out, lo, go = [], 0, 0
    for m in models:
        p = m.params
        nl, ng = len(p.edges), len(p.hybrids)
        out.append(L.Parameters(p.edges, p.hybrids,
                                ell[lo:lo + nl].copy(), gam[go:go + ng].copy()))
        lo += nl
        go += ng
    # The layout contract of :class:`BatchedTripleModel` -- blocks concatenated in ``models`` order
    # -- is what makes the slicing above correct.  Assert it rather than trust it: a silent
    # off-by-one here would write one network's branch lengths onto another's topology, and the
    # result would still be a valid network and a plausible distance.
    if lo != ell.size or go != gam.size:
        raise RuntimeError(f"parameter blocks did not tile the profile result: consumed {lo} of "
                           f"{ell.size} lengths and {go} of {gam.size} gammas")
    return out, res


def _graphed_step(bm, ot, ell, gam):
    """Capture one forward+backward and return a ``replay()`` that leaves grads in place.

    The parameters are captured *by address*, so the optimiser may mutate them in place between
    replays and the graph will read the new values -- which is exactly what Adam plus a clamp does.
    Gradients are zeroed inside the capture, because ``backward`` accumulates.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                if ell.grad is not None:
                    ell.grad = None
                if gam.grad is not None:
                    gam.grad = None
                (-bm.log_likelihood(ot, ell, gam).sum()).backward()
        torch.cuda.current_stream().wait_stream(side)
        ell.grad = torch.zeros_like(ell)
        gam.grad = torch.zeros_like(gam)
        handle = torch.cuda.CUDAGraph()
        with torch.cuda.graph(handle):
            ell.grad.zero_()
            gam.grad.zero_()
            static_val = bm.log_likelihood(ot, ell, gam)
            (-static_val.sum()).backward()
    except Exception:  # noqa: BLE001
        return None

    def replay():
        handle.replay()
        return static_val

    return replay


def _blocks(slices, mask: torch.Tensor, n_networks: int):
    """The ``(lo, hi)`` parameter spans of the networks ``mask`` selects."""
    idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    return [slices[b] for b in idx if b < n_networks]


# --------------------------------------------------------------------- batched L-BFGS


def _seg_dot(a: torch.Tensor, b: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """``(N,)`` inner products, one per network, over disjoint parameter blocks."""
    return torch.zeros(n, dtype=a.dtype, device=a.device).index_add_(0, seg, a * b)


def profile_mpl_lbfgs(models, obs, *, device=None, dtype=torch.float64, restarts: int = 4,
                      iters: int = 200, history: int = 10, seed: int = 0, init: str = "mom",
                      max_backtracks: int = 30, c1: float = 1e-4,
                      batched: "BatchedTripleModel | None" = None) -> ProfileResult:
    """``sup MPL`` per network by a **batched, box-projected L-BFGS**.

    :func:`profile_mpl`'s projected Adam is exact-per-network and matrix-free, and it is not
    enough.  Measured on 48 near-greedy terminal states at n = 50 against a 12,000-iteration Adam
    reference:

    | Adam iterations | Kendall tau | mean gap (log units) |
    |---|---|---|
    | 300 | 0.427 | 48.6 |
    | 600 | 0.437 | 21.7 |

    Doubling the schedule **halves the gap and does not move the ranking**, because what a
    group-normalised advantage needs is not a small error but a *homogeneous* one, and first-order
    methods converge at rates that differ per network. scipy's L-BFGS-B reaches the same optimum in
    **55-150 iterations** (A53) -- it is only slow there because its gradient is numerical, at
    ``n_params + 1`` evaluations apiece. With autograd that objection disappears.

    **Batching L-BFGS is exact here, unlike batching it naively.**  ``profile_mpl``'s docstring
    says a single L-BFGS over the concatenated parameters would let one network's curvature steer
    another's step -- true, and the fix is that every inner product in the two-loop recursion is a
    *segment* sum over that network's own block, and the line search carries a per-network step
    size. The parameter blocks are disjoint, so the result is identical to running N separate
    L-BFGS instances, computed in one batched forward and backward.

    Bounds are handled by projection: the step is clamped to the box and the gradient is zeroed
    where it points out of it. ``gamma``'s interval stays **closed** (``likelihood.MIN_GAMMA``).
    Curvature pairs are skipped per network when ``s . y <= 0``, so a network whose step gained no
    curvature keeps the history it had rather than corrupting it.
    """
    import time as _time

    n = len(models)
    restarts = max(1, int(restarts))
    bm = batched or BatchedTripleModel(list(models) * restarts, device=device, dtype=dtype)
    nb = bm.n_networks
    ot = torch.as_tensor(np.asarray(obs), dtype=dtype, device=bm.device)
    if ot.dim() == 3:
        if ot.shape[0] != n:
            raise ValueError(f"obs has {ot.shape[0]} rows, expected one per network ({n})")
        ot = ot.repeat(restarts, 1, 1)

    ell0, gam0 = _initial_vectors(bm, models, obs, restarts, n, init, seed)
    nl, ng = bm.n_lengths, bm.n_gammas
    x = torch.cat([ell0, gam0])
    lo = torch.cat([torch.full((nl,), L.MIN_LENGTH, dtype=dtype, device=bm.device),
                    torch.full((ng,), L.MIN_GAMMA, dtype=dtype, device=bm.device)])
    hi = torch.cat([torch.full((nl,), L.MAX_LENGTH, dtype=dtype, device=bm.device),
                    torch.full((ng,), L.MAX_GAMMA, dtype=dtype, device=bm.device)])
    seg = torch.empty(nl + ng, dtype=torch.int64, device=bm.device)
    for b, (a, z) in enumerate(bm.length_slices):
        seg[a:z] = b
    for b, (a, z) in enumerate(bm.gamma_slices):
        seg[nl + a:nl + z] = b

    def value_and_grad(xv):
        xr = xv.detach().requires_grad_(True)
        val = bm.log_likelihood(ot, xr[:nl], xr[nl:])
        (grad,) = torch.autograd.grad(val.sum(), xr)
        return val.detach(), -grad.detach()          # minimise -logL

    def value(xv):
        with torch.no_grad():
            return bm.log_likelihood(ot, xv[:nl], xv[nl:])

    t0 = _time.perf_counter()
    val, g = value_and_grad(x)
    best = val.clone()
    best_x = x.clone()
    s_hist: list[torch.Tensor] = []
    y_hist: list[torch.Tensor] = []
    rho_hist: list[torch.Tensor] = []
    have_hist = torch.zeros(nb, dtype=torch.bool, device=bm.device)
    tail = torch.zeros(nb, dtype=dtype, device=bm.device)
    prev = val.clone()
    tail_from = max(1, int(iters * 0.9))

    for it in range(iters):
        # Project the gradient: a component pushing out of the box is inactive.
        gp = torch.where((x <= lo) & (g > 0), torch.zeros_like(g), g)
        gp = torch.where((x >= hi) & (gp < 0), torch.zeros_like(gp), gp)

        q = gp.clone()
        alphas = []
        for s_i, y_i, r_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist),
                                 strict=True):
            a_i = r_i * _seg_dot(s_i, q, seg, nb)
            q = q - a_i[seg] * y_i
            alphas.append(a_i)
        if s_hist:
            sy = _seg_dot(s_hist[-1], y_hist[-1], seg, nb)
            yy = _seg_dot(y_hist[-1], y_hist[-1], seg, nb)
            q = q * torch.where(yy > 0, sy / yy.clamp_min(1e-300), torch.ones_like(yy))[seg]
        for s_i, y_i, r_i, a_i in zip(s_hist, y_hist, rho_hist, reversed(alphas), strict=True):
            beta = r_i * _seg_dot(y_i, q, seg, nb)
            q = q + s_i * (a_i - beta)[seg]
        d = -q

        slope = _seg_dot(gp, d, seg, nb)                 # < 0 for a descent direction
        # A network whose direction is not a descent one falls back to steepest descent.
        bad = slope >= 0
        if bad.any():
            d = torch.where(bad[seg], -gp, d)
            slope = _seg_dot(gp, d, seg, nb)

        # **Normalise the step of any network with no curvature, or its history is poisoned before
        # it exists.**  Without stored pairs the direction is plain steepest descent, and the
        # gradient of a log-likelihood over 19,600 triples x 1,000 loci is of order 1e6, so a
        # unit step lands far outside the box, every backtrack overshoots, and the ``(s, y)`` pair
        # that gets stored is garbage.  Measured without this: the optimiser stalled inside 60
        # iterations and returned *identical* values at 60, 150 and 300, up to **20,916 log units**
        # short of scipy.  A unit-norm first step is the textbook fix and costs one segment sum.
        fresh = ~have_hist
        if bool(fresh.any()):
            nrm = _seg_dot(d, d, seg, nb).clamp_min(1e-300).sqrt()
            factor = torch.where(fresh, 1.0 / nrm, torch.ones_like(nrm))
            d = d * factor[seg]
            slope = _seg_dot(gp, d, seg, nb)

        step = torch.ones(nb, dtype=dtype, device=bm.device)
        done = torch.zeros(nb, dtype=torch.bool, device=bm.device)
        x_new = x
        val_new = val
        for _ in range(max_backtracks):
            trial = torch.clamp(x + torch.where(done, torch.zeros_like(step), step)[seg] * d,
                                lo, hi)
            trial = torch.where(done[seg], x_new, trial)
            v = value(trial)
            # Armijo on the *maximised* objective: logL must rise by c1 * t * |slope|.
            ok = v >= val - c1 * step * slope
            take = ok & ~done
            if take.any():
                x_new = torch.where(take[seg], trial, x_new)
                val_new = torch.where(take, v, val_new)
                done = done | take
            if bool(done.all()):
                break
            step = torch.where(done, step, step * 0.5)
        # **A network that never satisfied Armijo has its history cleared, not merely skipped.**
        # Leaving it in place is what made one network of three stall 18.2 log units short and stay
        # there through 160 iterations: the stored curvature keeps proposing a direction the box
        # will not accept, the line search keeps failing, and nothing new is ever learned. Clearing
        # it drops that network back to a normalised steepest-descent step, which either makes
        # progress or is a genuine stationary point.
        stalled = ~done
        if bool(stalled.any()):
            keep_mask = (~stalled)[seg].to(dtype)
            s_hist = [h * keep_mask for h in s_hist]
            y_hist = [h * keep_mask for h in y_hist]
            rho_hist = [r * (~stalled).to(dtype) for r in rho_hist]
            have_hist = have_hist & ~stalled
        s = x_new - x
        val_new = torch.where(done, val_new, val)
        _, g_new = value_and_grad(x_new)
        y = g_new - g
        sy = _seg_dot(s, y, seg, nb)
        keep = (sy > 1e-12) & done
        if bool(keep.any()):
            mask = keep[seg].to(dtype)
            s_hist.append(s * mask)
            y_hist.append(y * mask)
            rho_hist.append(torch.where(keep, 1.0 / sy.clamp_min(1e-300), torch.zeros_like(sy)))
            have_hist = have_hist | keep
            if len(s_hist) > history:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)

        x, val, g = x_new, val_new, g_new
        better = val > best
        best = torch.where(better, val, best)
        best_x = torch.where(better[seg], x, best_x)
        if it >= tail_from:
            tail = torch.maximum(tail, (val - prev).abs())
        prev = val.clone()

    seconds = _time.perf_counter() - t0
    by_restart = best.reshape(restarts, n)
    value_out, which = by_restart.max(dim=0)
    bl, bg = best_x[:nl], best_x[nl:]
    out_l = torch.cat([bl[slice(*bm.length_slices[int(which[b]) * n + b])] for b in range(n)])
    out_g = torch.cat([bg[slice(*bm.gamma_slices[int(which[b]) * n + b])] for b in range(n)])
    return ProfileResult(
        value=value_out, lengths=out_l, gammas=out_g, n_iters=iters, restarts=restarts,
        seconds=seconds, restart_gain=value_out - by_restart[0],
        tail_movement=tail.reshape(restarts, n).gather(0, which.unsqueeze(0)).squeeze(0),
    )
