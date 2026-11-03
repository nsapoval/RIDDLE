"""W3 -- the encoder and the attention decoder.  ``RL_PLAN.md`` §3.

**Encode once, decode per step.**  The overlap graph is static within an episode; only which
vertices are *accepted* changes.  So the GNN runs once per instance and the decoder runs once per
step over the resulting embeddings.  At |C| ~ 1,800 with mean degree ~360, 50 full GNN forwards per
episode would dominate everything the policy does.

**Aggregation is a segment sum, not attention, and the reason is measured.**  ``RL_PLAN`` §3 says
"5-layer GATv2 or GINE".  Two things about the tau = 2 candidate graph argue against per-edge
attention here, and one of them is a hard constraint:

* **Memory.**  A per-edge attention weight needs a per-edge *hidden-width* tensor.  The graph
  carries 2|E| directed entries -- 594 k at the median cell, 2.2 M at the worst -- so one layer's
  edge tensor at width 128 is 300 MB, and a batch of eight instances with three layers is tens of
  gigabytes on a GPU that is shared with another user's job.  A segment sum aggregates through
  ``index_select`` + ``index_add_``, whose backward needs only the index, so the stored activation
  is ``(|C|, hidden)`` -- under a megabyte -- and the ``(2|E|, hidden)`` gather is transient.
  This is also the choice :mod:`opc.rl.mpl_torch` already made and recorded, for its own reason:
  ``torch.sparse_csr_tensor`` is beta and warns as much, and a segment sum has a first-class CUDA
  kernel and an exact autograd rule where the sparse path's does not.
* **Density.**  W0.2 measured tau = 2 density at 0.12-0.26 and mean degree 147-526 (A66).  One hop
  already reaches a fifth of the graph and two hops reach nearly all of it.  Attention over 360
  neighbours is a soft average whichever weights it learns, and five layers over-smooth outright --
  which is why :data:`DEFAULT_LAYERS` is **3**, not 5.

Neither argument is a proof, so neither is asserted.  ``layers=0`` makes the encoder a
**features-only MLP that never touches an edge**, and that arm is one flag on the oracle-F1 reward,
which costs no MPL at all.  It measures directly whether the overlap graph buys anything over the
node features -- rather than the question being settled by this docstring.

**Two neighbourhood summaries, not one.**  Each layer aggregates an unweighted mean *and* a
support-weighted mean.  The unweighted one says how crowded a candidate's neighbourhood is; the
weighted one says whether its neighbours are the ones greedy will take first.  They are different
questions and the second is the one the baseline is built out of.

**Size-agnostic by construction.**  Every feature is normalised per instance
(:mod:`opc.rl.instances`) and no layer's width depends on ``|C|`` or ``n``.  That is what makes
**G-R6** -- train at n <= 30, evaluate at n = 50 -- a test of transfer rather than of rescaling,
and ``tests/test_rl_policy.py`` pins it by running one set of weights at two ``n``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "N_DYNAMIC",
    "PolicyConfig",
    "GraphTensors",
    "Encoded",
    "Policy",
    "SegmentConv",
    "PairConv",
    "PAIR_EDGE_FEATURES",
    "pair_edge_features",
    "graph_tensors_for",
    "GCNConv",
    "GINConv",
    "CONVS",
    "graph_tensors",
    "load_transferable",
]

DEFAULT_HIDDEN = 128
#: Three, not §3's five.  See the module docstring: at density 0.12-0.26 two hops already reach
#: nearly the whole graph, so depth past that buys receptive field that is already saturated and
#: pays for it in over-smoothing.
DEFAULT_LAYERS = 3
#: The decoder's per-step, per-candidate features.  See :func:`dynamic_features`.
N_DYNAMIC = 3


@dataclass
class PolicyConfig:
    n_node_features: int
    n_global_features: int
    hidden: int = DEFAULT_HIDDEN
    layers: int = DEFAULT_LAYERS
    #: Bounded logits, as in the attention-model line of work: an unbounded dot product makes the
    #: entropy schedule (§4: beta 0.02 -> 0.001) meaningless because the policy can saturate the
    #: softmax without the entropy bonus ever being able to pull it back.
    tanh_clip: float = 10.0
    dropout: float = 0.0
    #: How many per-candidate features the decoder's additive term takes.  **3** is the
    #: construction MDP's (:func:`dynamic_features`); the swap MDP passes **4**
    #: (:data:`opc.rl.swap.N_SWAP_DYNAMIC`).  It is a config field rather than two ``Policy``
    #: classes because everything else is shared -- and because the *context* stays four scalars
    #: in both MDPs, so ``embed``, ``convs``, ``key`` and ``ctx`` all keep their shapes and a v1
    #: checkpoint transfers into a v2 policy with only ``dyn`` re-initialised
    #: (:func:`load_transferable`).
    n_dynamic: int = N_DYNAMIC
    #: Which message-passing layer to stack, by name in :data:`CONVS`.
    #:
    #: **The default is ``"segment"`` and it must stay that way.**  Every landed checkpoint stores
    #: its config as ``vars(cfg)`` and every reader rebuilds it with ``PolicyConfig(**ck[
    #: "policy_config"])``, so a dict written before this field existed has to keep constructing --
    #: and has to come back as the layer it was actually trained with.  A dataclass default is what
    #: makes that true; changing it would silently reinterpret six drivers' checkpoints.
    #:
    #: The three convs differ in **the aggregator alone** -- the residual, the ``LayerNorm`` and the
    #: MLP are identical -- so a difference between two arms is attributable to the aggregation and
    #: not to normalisation or to capacity.  W10 is what that comparison is for.
    conv: str = "segment"
    #: Recompute each message-passing layer in the backward instead of keeping its activations.
    #:
    #: **This is not an optimisation, it is what makes the encoder fit.**  Measured at n = 50,
    #: tau = 2, |C| = 2,067 with 771 k directed edges: the encoder retains **771 MiB per layer**
    #: after the forward -- two per-edge float32 tensors, 376 MiB each -- so three layers is
    #: **2.3 GB for one instance**, and a batch of eight is 18 GB before a single rollout is
    #: sampled.  The retention is autograd state, not a transient: it is *insensitive to the edge
    #: chunk size*, which is how it was told apart from the gather it looks like.
    #:
    #: Recomputing costs one extra encoder forward per instance per step (~30 ms) and takes the
    #: retained state to the ``(|C|, hidden)`` boundaries alone -- about 1 MB a layer.
    checkpoint_encoder: bool = True
    #: ``PairConv`` only: aggregate over the contains / contained-in relations as well as the
    #: crossing one.  Ignored by every other conv, and defaulted so a pre-S1 checkpoint's dict still
    #: constructs.
    containment: bool = True


@dataclass
class GraphTensors:
    """The overlap graph on the device, as the two index vectors a segment sum needs.

    Built straight from :class:`opc.rl.env.OverlapCSR`, which already stores both directions of
    every edge -- so ``src``/``dst`` is its ``nbr`` array read against its ``indptr`` and nothing
    is recomputed or re-sorted.
    """

    src: torch.Tensor        # int64[2E]
    dst: torch.Tensor        # int64[2E]
    deg: torch.Tensor        # float32[|C|]  -- clamped at 1 so a kernel vertex divides safely
    w_norm: torch.Tensor     # float32[2E]   -- support of src, normalised within dst's row
    #: float32[|C|] -- the **raw** degree plus one, i.e. the self-loop-augmented degree.
    #:
    #: Kept beside ``deg`` rather than derived from it because they are different quantities and
    #: only one of them is right for :class:`GCNConv`.  ``deg`` is ``max(deg, 1)``, which is a
    #: divisor guard: it makes a kernel vertex (degree 0) divide safely and is off by one on every
    #: vertex that has neighbours.  GCN's symmetric normalisation is over ``A + I``, so its
    #: denominator is ``deg + 1`` exactly -- 1 at a kernel vertex, which coincides, and ``deg + 1``
    #: elsewhere, which does not.  At mean degree 147-526 the discrepancy is small and would never
    #: have shown up as a failure; it would just have made the arm not-quite-GCN.
    deg1: torch.Tensor
    # ---- present only when built with ``words=`` -- what :class:`PairConv` reads, and nothing
    # else does.  Every other conv, and every caller that does not pass ``words``, sees ``None``
    # and computes exactly what it did before S1.
    words: torch.Tensor | None = None          # int64[|C|, W] -- the clusters' uint64 words
    n_taxa: int = 0
    m: float = 0.0
    support: torch.Tensor | None = None        # float32[|C|] -- raw support, not normalised
    size: torch.Tensor | None = None           # float32[|C|]
    cls: torch.Tensor | None = None            # int64[2E] -- OverlapCSR.cls, the edge's A & B
    n_classes: int = 0
    hash_sorted: torch.Tensor | None = None    # int64[|C|] -- row hashes, sorted
    hash_order: torch.Tensor | None = None     # int64[|C|] -- the row each sorted hash belongs to
    cont_parent: torch.Tensor | None = None    # int64[P] -- parent strictly contains child
    cont_child: torch.Tensor | None = None     # int64[P]
    cont_w_child: torch.Tensor | None = None   # float32[P] -- child support / sum over the parent's
    cont_w_parent: torch.Tensor | None = None  # float32[P] -- parent support / sum over the child's
    n_children: torch.Tensor | None = None     # float32[|C|], clamped at 1
    n_parents: torch.Tensor | None = None      # float32[|C|], clamped at 1


def graph_tensors(graph, support: np.ndarray, device, *, words=None, n_taxa=None,
                  m=None) -> GraphTensors:
    n = int(graph.n)
    indptr = np.asarray(graph.indptr, dtype=np.int64)
    dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    src = np.asarray(graph.nbr, dtype=np.int64)
    deg = np.diff(indptr).astype(np.float32)
    w = np.asarray(support, dtype=np.float64)
    ws = w[src]
    # ``np.add.at`` says this directly and is an order of magnitude slower: unbuffered ``ufunc.at``
    # leaves numpy's fast path entirely.  ``bincount`` with weights is the same statement, and at
    # 2|E| = 594 k entries per instance and eight instances a step the difference is the term.
    # (The same substitution is recorded in ``opc.rl.env.legal_mask``.)
    tot = np.bincount(dst, weights=ws, minlength=n)
    w_norm = (ws / np.maximum(tot[dst], 1e-12)).astype(np.float32)
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    gt = GraphTensors(src=t(src, torch.int64), dst=t(dst, torch.int64),
                      deg=t(np.maximum(deg, 1.0), torch.float32),
                      w_norm=t(w_norm, torch.float32),
                      deg1=t(deg + 1.0, torch.float32))
    if words is not None:
        if n_taxa is None or m is None:
            raise ValueError("graph_tensors(words=...) also needs n_taxa and m")
        _attach_pairs(gt, graph, support, words, int(n_taxa), float(m), device)
    return gt


def graph_tensors_for(cfg: "PolicyConfig", inst, device) -> GraphTensors:
    """:func:`graph_tensors` with the pair extras exactly when ``cfg``'s conv reads them."""
    if cfg.layers and getattr(CONVS.get(cfg.conv), "needs_pairs", False):
        return graph_tensors(inst.graph, inst.gs.support, device, words=inst.gs.words,
                             n_taxa=int(inst.n), m=float(inst.m))
    return graph_tensors(inst.graph, inst.gs.support, device)


# ------------------------------------------------------------------ S1-S3: the pair extras
#
# Everything below is int64 arithmetic on the clusters' packed words, on the device.  Two facts
# make that exact rather than approximate: two's-complement wrap gives the same *bits* as uint64
# arithmetic, and every popcount / hash below uses only xor, and, add, multiply and shifts whose
# results are masked back into range -- so an arithmetic shift of a negative word never leaks sign
# bits into a result.  ``tests/test_rl_pairconv.py`` checks both against Python ints.

_M1, _M2, _M4, _H01 = 0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x0101010101010101
_FNV_OFFSET, _FNV_PRIME = 0x6C62272E07BB0142, 0x100000001B3

#: ``PairConv``'s per-edge inputs, in order.  The S0 probe's ``PAIR_FEATURES`` (A108 addendum),
#: with the support log-ratio divided by ``log(m + 1)`` so every column is O(1) without a
#: normaliser.  For the directed entry ``q <- c`` with ``H = q & c``, ``U = q | c``, ``D = q ^ c``
#: (the twin), ``X = q - c`` and ``W = c - q``; a set "is in" when it is a candidate row, a
#: singleton or the whole taxon set, and the latter two count as support ``m``.
PAIR_EDGE_FEATURES = (
    "sq", "sc", "log_sc_over_sq", "size_q", "size_c", "h_over_q", "h_over_c", "h_over_n",
    "h_in", "sh", "u_in", "su", "d_in", "sd", "twin_asym_q", "twin_asym_c",
    "x_in", "sx", "w_in", "sw", "deg_q", "deg_c", "same_class_q", "same_class_c", "rank_c_in_q",
)


def _popcount64(x: torch.Tensor) -> torch.Tensor:
    x = x - ((x >> 1) & _M1)
    x = (x & _M2) + ((x >> 2) & _M2)
    x = (x + (x >> 4)) & _M4
    return ((x * _H01) >> 56) & 0xFF


#: splitmix64's finaliser constants, as the signed int64 values with the same bits.
_SM1, _SM2 = 0xBF58476D1CE4E5B9 - (1 << 64), 0x94D049BB133111EB - (1 << 64)


def _lshr(x: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift on int64: an arithmetic shift with the sign-extended bits masked off."""
    return (x >> k) & ((1 << (64 - k)) - 1)


def _mix64(x: torch.Tensor) -> torch.Tensor:
    x = (x ^ _lshr(x, 30)) * _SM1
    x = (x ^ _lshr(x, 27)) * _SM2
    return x ^ _lshr(x, 31)


def _hash_rows(w: torch.Tensor) -> torch.Tensor:
    """A content hash per row: the lookup key and the tie-break of ``rank_c_in_q``.

    **One word (n <= 64) keeps ``(offset ^ w) * prime``**, which is a bijection there, so it cannot
    collide -- and it is the order every n = 25 / 50 pair checkpoint was trained on, which is why it
    is not replaced.  **Several words get a splitmix64 chain.**  The xor-multiply chain was the
    first version for every width and **collided on a real n = 200 candidate set** (l2reg rep00,
    8,807 clusters of four words): multiplying by a small odd prime barely moves the high bits, and
    clusters confined to different words share most of their structure.  ``_attach_pairs`` raises
    on any collision, which is how it was caught instead of becoming a wrong lookup.
    """
    h = torch.full((w.shape[0],), _FNV_OFFSET, dtype=torch.int64, device=w.device)
    if w.shape[1] == 1:
        return (h ^ w[:, 0]) * _FNV_PRIME
    for j in range(w.shape[1]):
        h = _mix64(h ^ w[:, j])
    return h


def _dense_bits(w: torch.Tensor, n_taxa: int) -> torch.Tensor:
    shifts = torch.arange(64, device=w.device, dtype=torch.int64)
    bits = (w.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(w.shape[0], -1)[:, :n_taxa].to(torch.float32)


def _attach_pairs(gt: GraphTensors, graph, support, words, n_taxa: int, m: float, device) -> None:
    w = torch.as_tensor(np.ascontiguousarray(words).view(np.int64), device=device)
    n_c = int(w.shape[0])
    sup = torch.as_tensor(np.asarray(support, dtype=np.float32), device=device)
    size = _popcount64(w).sum(1).to(torch.float32)
    hs, order = torch.sort(_hash_rows(w))
    if n_c > 1 and bool((hs[1:] == hs[:-1]).any()):
        raise ValueError("two candidate clusters share a 64-bit row hash; the pair lookup would "
                         "be ambiguous")
    # containment, blocked: parent i strictly contains child j iff |i & j| == |j| < |i|
    dense = _dense_bits(w, n_taxa)
    par, chi = [], []
    for lo in range(0, n_c, 2048):
        inter = dense[lo:lo + 2048] @ dense.T
        hit = (inter == size.unsqueeze(0)) & (size[lo:lo + 2048].unsqueeze(1) > size.unsqueeze(0))
        pi, ci = torch.nonzero(hit, as_tuple=True)
        par.append(pi + lo)
        chi.append(ci)
    par = torch.cat(par) if par else torch.zeros(0, dtype=torch.int64, device=device)
    chi = torch.cat(chi) if chi else torch.zeros(0, dtype=torch.int64, device=device)
    zeros = torch.zeros(n_c, dtype=torch.float32, device=device)
    tot_ch = zeros.index_add(0, par, sup[chi])
    tot_pa = zeros.index_add(0, chi, sup[par])
    gt.words, gt.n_taxa, gt.m, gt.support, gt.size = w, n_taxa, m, sup, size
    gt.cls = torch.as_tensor(np.asarray(graph.cls, dtype=np.int64), device=device)
    gt.n_classes = int(graph.n_classes)
    gt.hash_sorted, gt.hash_order = hs, order
    gt.cont_parent, gt.cont_child = par, chi
    gt.cont_w_child = sup[chi] / tot_ch[par].clamp_min(1e-12)
    gt.cont_w_parent = sup[par] / tot_pa[chi].clamp_min(1e-12)
    gt.n_children = torch.bincount(par, minlength=n_c).to(torch.float32).clamp_min(1.0)
    gt.n_parents = torch.bincount(chi, minlength=n_c).to(torch.float32).clamp_min(1.0)


def _lookup_rows(g: GraphTensors, bits: torch.Tensor) -> torch.Tensor:
    """The candidate row each packed set in ``bits`` is, or -1.  Exact: a hash hit is verified."""
    h = _hash_rows(bits)
    i = torch.searchsorted(g.hash_sorted, h).clamp_max(g.hash_sorted.numel() - 1)
    row = g.hash_order[i]
    ok = (g.hash_sorted[i] == h) & (g.words[row] == bits).all(1)
    return torch.where(ok, row, torch.full_like(row, -1))


@torch.no_grad()
def pair_edge_features(g: GraphTensors) -> torch.Tensor:
    """``float32[2E, len(PAIR_EDGE_FEATURES)]`` for every directed entry ``dst <- src``.

    **Computed per forward, not cached.**  At n = 50 the matrix is ~130 MB an instance, and a
    700-instance pool holding it would be ~90 GB of GPU; the computation is a few gathers, one
    sort and five hashed lookups.
    """
    if g.words is None:
        raise ValueError("pair_edge_features needs graph_tensors(..., words=, n_taxa=, m=)")
    n_e, n_c, dev = int(g.src.numel()), int(g.words.shape[0]), g.src.device
    out = torch.empty((n_e, len(PAIR_EDGE_FEATURES)), dtype=torch.float32, device=dev)
    if n_e == 0:
        return out
    q_all, c_all = g.dst, g.src
    deg = g.deg1 - 1.0
    ncls = int(g.n_classes) + 1
    uniq, inv, cnt = torch.unique(q_all * ncls + g.cls, return_inverse=True, return_counts=True)
    same_q = cnt[inv].to(torch.float32)
    same_c = cnt[torch.searchsorted(uniq, c_all * ncls + g.cls)].to(torch.float32)
    # c's support rank inside q's row, 0 = best supported.  **Ties break on c's row hash, not on
    # entry order**: entry order is a property of the labelling, and breaking ties on it made this
    # the one input that changed when the nodes were relabelled (the conv registry's permutation
    # test caught it).  The hash is a function of the cluster itself and is unique per row.
    hc = torch.empty(n_c, dtype=torch.int64, device=dev)
    hc[g.hash_order] = g.hash_sorted
    o0 = torch.argsort(hc[c_all], stable=True)
    o1 = o0[torch.argsort(-g.support[c_all[o0]], stable=True)]
    o2 = o1[torch.argsort(q_all[o1], stable=True)]
    counts = torch.bincount(q_all, minlength=n_c)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.empty(n_e, dtype=torch.float32, device=dev)
    rank[o2] = (torch.arange(n_e, device=dev) - starts[q_all[o2]]).to(torch.float32)
    m, n = float(g.m), float(g.n_taxa)
    lm = math.log(m + 1.0)

    def look(bits, sz):
        r = _lookup_rows(g, bits)
        full = (sz == 1) | (sz == n)
        s = torch.where(r >= 0, g.support[r.clamp_min(0)], torch.zeros_like(sz))
        return ((r >= 0) | full).to(torch.float32), torch.where(full, torch.full_like(s, m), s)

    for lo in range(0, n_e, EDGE_CHUNK):
        hi = min(lo + EDGE_CHUNK, n_e)
        q, c = q_all[lo:hi], c_all[lo:hi]
        wq, wc = g.words[q], g.words[c]
        szq, szc = g.size[q], g.size[c]
        szh = _popcount64(wq & wc).sum(1).to(torch.float32)
        h_in, sh = look(wq & wc, szh)
        u_in, su = look(wq | wc, szq + szc - szh)
        d_in, sd = look(wq ^ wc, szq + szc - 2 * szh)
        x_in, sx = look(wq & ~wc, szq - szh)
        w_in, sw = look(wc & ~wq, szc - szh)
        sq, sc = g.support[q], g.support[c]
        dq, dc = deg[q].clamp_min(1.0), deg[c].clamp_min(1.0)
        out[lo:hi] = torch.stack([
            sq / m, sc / m, torch.log((sc + 1.0) / (sq + 1.0)) / lm,
            szq / n, szc / n, szh / szq, szh / szc, szh / n,
            h_in, sh / m, u_in, su / m, d_in, sd / m,
            (sq - sd) / (sq + sd).clamp_min(1.0), (sc - sd) / (sc + sd).clamp_min(1.0),
            x_in, sx / m, w_in, sw / m,
            deg[q] / n_c, deg[c] / n_c,
            same_q[lo:hi] / dq, same_c[lo:hi] / dc, rank[lo:hi] / dq,
        ], dim=1)
    return out


#: Directed edges gathered at once.  The only wide tensor in the encoder is ``(chunk, hidden)``,
#: so this caps the transient at ~64 MB at ``hidden = 128`` whatever the instance's ``|E|`` is.
#:
#: **Chunking is not premature.**  Unchunked, the gather is ``(2|E|, hidden)``: 300 MB at the
#: median tau = 2 cell and **1.1 GB** at the worst (2.2 M directed edges).  Encoding a batch of
#: instances up front leaves one such block per instance in torch's caching allocator, and the
#: blocks differ in size so they do not get reused -- measured at **8.6 GB** for a batch of four,
#: which is an out-of-memory on any GPU that is shared with anything else.
EDGE_CHUNK = 1 << 17


class SegmentConv(nn.Module):
    """One message-passing layer: mean and support-weighted mean, then a residual MLP."""

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        n_edges = int(g.src.shape[0])
        mean = torch.zeros_like(h)
        wsum = torch.zeros_like(h)
        for lo in range(0, n_edges, EDGE_CHUNK):
            hi = min(lo + EDGE_CHUNK, n_edges)
            src, dst = g.src[lo:hi], g.dst[lo:hi]
            part = h.index_select(0, src)
            mean = mean.index_add(0, dst, part)
            wsum = wsum.index_add(0, dst, part * g.w_norm[lo:hi].unsqueeze(1))
        mean = mean / g.deg.unsqueeze(1)
        return self.norm(h + self.mlp(torch.cat([h, mean, wsum], dim=1)))


class GCNConv(nn.Module):
    """One Kipf-Welling propagation: ``sum_u h_u / sqrt((d_u + 1)(d_v + 1))``, over ``A + I``.

    **The normalisation coefficient is recomputed per chunk rather than stored.**  A ``float32[2E]``
    coefficient would be 33 MB at the worst b250 cell (8.3 M directed edges) and would have to be
    built inside :func:`graph_tensors` for every run, including the RL runs that never ask for it.
    ``g.deg1.index_select`` twice and one ``rsqrt`` is a few hundred microseconds against a gather
    that is happening anyway, and it keeps the transient at ``(chunk, hidden)`` like every other
    layer here.

    The self term is ``h / (d_v + 1)``, which is the ``u = v`` entry of the same sum -- written out
    rather than folded into the loop because the self-loop is not in ``OverlapCSR`` (an overlap
    graph has no self-edges) and adding one would change every other consumer of that structure.
    """

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        n_edges = int(g.src.shape[0])
        agg = h / g.deg1.unsqueeze(1)                      # the u = v term of A + I
        inv = g.deg1.rsqrt()
        for lo in range(0, n_edges, EDGE_CHUNK):
            hi = min(lo + EDGE_CHUNK, n_edges)
            src, dst = g.src[lo:hi], g.dst[lo:hi]
            coef = inv.index_select(0, src) * inv.index_select(0, dst)
            agg = agg.index_add(0, dst, h.index_select(0, src) * coef.unsqueeze(1))
        return self.norm(h + self.mlp(torch.cat([h, agg], dim=1)))


class GINConv(nn.Module):
    """One GIN aggregation: ``MLP((1 + eps) h_v + sum_u h_u)``, ``eps`` learned.

    Sum, deliberately -- that is what makes GIN injective on neighbourhood *multisets* and it is the
    whole of what distinguishes this arm from :class:`GCNConv`.  It is also the reason this arm is
    **pre-registered as the one expected to transfer worst**: a sum over ``N(v)`` is a
    degree-dependent statistic, and mean degree on the overlap graph grows with ``n`` (147-526 at
    n = 50; b250 reaches 4.15e6 undirected edges on 14,124 nodes).  A model trained at n in {25, 50}
    therefore sees aggregates an order of magnitude smaller than the ones it meets at n = 250.

    The ``LayerNorm`` on the *aggregate* is what keeps that from being a numerical failure rather
    than a representational one -- reference GIN uses a ``BatchNorm`` here, which cannot be used
    when instances are fed one at a time -- but it does not remove the dependence, and it is not
    supposed to.  Measuring it is the point.
    """

    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.agg_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        n_edges = int(g.src.shape[0])
        agg = (1.0 + self.eps) * h
        for lo in range(0, n_edges, EDGE_CHUNK):
            hi = min(lo + EDGE_CHUNK, n_edges)
            agg = agg.index_add(0, g.dst[lo:hi], h.index_select(0, g.src[lo:hi]))
        return self.norm(h + self.mlp(torch.cat([h, self.agg_norm(agg)], dim=1)))


class PairConv(nn.Module):
    """``SegmentConv``'s two means, plus a **supervised attention** over crossing neighbours and,
    optionally, the containment relations.  S1-S3 of the A108 addendum.

    **Why each piece.**  The crossing graph already joins a reticulation's two parents, but the
    partner is 1-3% of a mean over 345-590 neighbours, and the S0 probe measured that pairwise
    inputs name it (minor-side hits@1 0.65 against support's 0.31).  So each entry ``q <- c`` gets
    a scalar gate from :data:`PAIR_EDGE_FEATURES` and small projections of both ends, softmaxed over
    ``q``'s row **with a learned sink**: a row that has no partner -- most of them -- can put its
    mass nowhere instead of being forced to average.  :meth:`Policy.edge_logits` exposes the same
    gate so a co-parent BCE can supervise it; S0's node-level reading is the reason the supervision
    goes *into* the aggregation rather than onto a separate head read out afterwards.

    **Containment** is the relation the network is made of and the one the crossing graph lacks: a
    hybrid ``H = A & B`` has no edge to either parent.  ``containment=True`` adds a mean and a
    support-weighted mean over all strict subsets (children) and all strict supersets (parents).

    It is a **superset of** :class:`SegmentConv` -- the same mean and weighted mean enter the same
    residual MLP -- so a difference against ``segment`` is attributable to what was added.
    """

    needs_pairs = True

    def __init__(self, hidden: int, dropout: float = 0.0, containment: bool = True,
                 gate_dim: int = 16):
        super().__init__()
        self.containment = bool(containment)
        self.proj_q = nn.Linear(hidden, gate_dim)
        self.proj_c = nn.Linear(hidden, gate_dim)
        self.gate = nn.Sequential(nn.Linear(len(PAIR_EDGE_FEATURES) + 2 * gate_dim, 32), nn.GELU(),
                                  nn.Linear(32, 1))
        self.sink = nn.Parameter(torch.zeros(()))
        self.value = nn.Linear(hidden, hidden, bias=False)
        width = (8 if self.containment else 4) * hidden
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def gate_logits(self, h: torch.Tensor, g: GraphTensors, ef: torch.Tensor,
                    pos: torch.Tensor) -> torch.Tensor:
        """The unnormalised attention logit of each directed entry in ``pos`` (``dst <- src``)."""
        q, c = g.dst[pos], g.src[pos]
        x = torch.cat([ef[pos], self.proj_q(h.index_select(0, q)),
                       self.proj_c(h.index_select(0, c))], dim=1)
        return self.gate(x).squeeze(-1)

    def attention(self, h: torch.Tensor, g: GraphTensors, ef: torch.Tensor):
        """``(weight per entry, sink weight per row)``; each row's weights and sink sum to 1."""
        n_e, n_c, dev = int(g.src.numel()), int(h.shape[0]), h.device
        sink = self.sink.to(h.dtype)
        if n_e == 0:
            return h.new_zeros(0), torch.ones(n_c, dtype=h.dtype, device=dev)
        a = torch.cat([self.gate_logits(h, g, ef, torch.arange(lo, min(lo + EDGE_CHUNK, n_e),
                                                                device=dev))
                       for lo in range(0, n_e, EDGE_CHUNK)])
        with torch.no_grad():   # a per-row shift for stability; the softmax is invariant to it
            mx = torch.full((n_c,), float("-inf"), dtype=a.dtype, device=dev)
            mx = mx.scatter_reduce(0, g.dst, a.detach(), reduce="amax")
            mx = torch.maximum(mx, sink.detach())
        ea = torch.exp(a - mx[g.dst])
        es = torch.exp(sink - mx)
        z = es + torch.zeros(n_c, dtype=a.dtype, device=dev).index_add(0, g.dst, ea)
        return ea / z[g.dst], es / z

    def forward(self, h: torch.Tensor, g: GraphTensors, ef: torch.Tensor) -> torch.Tensor:
        if ef is None or g.cont_parent is None:
            raise ValueError("PairConv needs graph_tensors(..., words=) and pair_edge_features(g)")
        n_e = int(g.src.numel())
        mean, wsum, attn = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
        wts, _ = self.attention(h, g, ef)
        v = self.value(h)
        for lo in range(0, n_e, EDGE_CHUNK):
            hi = min(lo + EDGE_CHUNK, n_e)
            src, dst = g.src[lo:hi], g.dst[lo:hi]
            part = h.index_select(0, src)
            mean = mean.index_add(0, dst, part)
            wsum = wsum.index_add(0, dst, part * g.w_norm[lo:hi].unsqueeze(1))
            attn = attn.index_add(0, dst, v.index_select(0, src) * wts[lo:hi].unsqueeze(1))
        parts = [h, mean / g.deg.unsqueeze(1), wsum, attn]
        if self.containment:
            n_p = int(g.cont_parent.numel())
            ch_mean, ch_w = torch.zeros_like(h), torch.zeros_like(h)
            pa_mean, pa_w = torch.zeros_like(h), torch.zeros_like(h)
            for lo in range(0, n_p, EDGE_CHUNK):
                hi = min(lo + EDGE_CHUNK, n_p)
                par, chi = g.cont_parent[lo:hi], g.cont_child[lo:hi]
                hc, hp = h.index_select(0, chi), h.index_select(0, par)
                ch_mean = ch_mean.index_add(0, par, hc)
                ch_w = ch_w.index_add(0, par, hc * g.cont_w_child[lo:hi].unsqueeze(1))
                pa_mean = pa_mean.index_add(0, chi, hp)
                pa_w = pa_w.index_add(0, chi, hp * g.cont_w_parent[lo:hi].unsqueeze(1))
            parts += [ch_mean / g.n_children.unsqueeze(1), ch_w,
                      pa_mean / g.n_parents.unsqueeze(1), pa_w]
        return self.norm(h + self.mlp(torch.cat(parts, dim=1)))


#: ``PolicyConfig.conv`` -> the layer it names.  ``"segment"`` is the incumbent and the default.
#:
#: All three take ``(hidden, dropout)`` and return ``LayerNorm(h + MLP([h, <aggregate>]))``, so the
#: residual, the normalisation and the MLP width are held fixed and the **aggregator is the only
#: variable**.  That is what makes a W10 arm comparison attributable.
CONVS = {"segment": SegmentConv, "gcn": GCNConv, "gin": GINConv, "pair": PairConv}


@dataclass
class Encoded:
    """What one instance's encoder pass produces, reused by every step of every rollout on it."""

    h: torch.Tensor          # (|C|, hidden) node embeddings
    keys: torch.Tensor       # (|C|, hidden) decoder keys -- projected once, not per step
    n_clusters: int
    #: Each conv's *input*, kept only under ``encode(keep_layer_inputs=True)`` -- what
    #: :meth:`Policy.edge_logits` re-reads the gates from.  ``(|C|, hidden)`` a layer.
    layer_inputs: list | None = None
    #: :func:`pair_edge_features` of the graph, when the conv reads pairs.
    edge_feat: torch.Tensor | None = None


class Policy(nn.Module):
    """Encoder + attention decoder + the W4.2 classification head, sharing one set of weights.

    §5 is explicit that the supervised arm is not a parallel branch: the pointwise head trains
    *the encoder the RL policy uses*, so that G-R3 ("beat support-ranking AUC") is a statement
    about the features the policy sees rather than about a second model.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.embed = nn.Sequential(
            nn.Linear(cfg.n_node_features + cfg.n_global_features, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
        if cfg.conv not in CONVS:
            raise ValueError(f"unknown conv {cfg.conv!r}; expected one of {sorted(CONVS)}")
        kw = {"containment": cfg.containment} if cfg.conv == "pair" else {}
        self.convs = nn.ModuleList(CONVS[cfg.conv](d, cfg.dropout, **kw)
                                   for _ in range(cfg.layers))
        self.key = nn.Linear(d, d, bias=False)
        # Context: [mean of accepted embeddings ; last action's embedding ; four scalars].
        self.ctx = nn.Sequential(nn.Linear(2 * d + 4, d), nn.GELU(), nn.Linear(d, d))
        #: ``h`` of "no previous action", learned rather than zeroed: ``t = 0`` is a real state the
        #: policy visits once per episode and POMO forces a first action from it.
        self.no_action = nn.Parameter(torch.zeros(d))
        self.dyn = nn.Linear(cfg.n_dynamic, 1)
        self.cls = nn.Linear(d, 1)          # W4.2: P(A in C(N_true)); discarded after use

    # ------------------------------------------------------------------ encoder

    def encode(self, x_nodes: torch.Tensor, x_global: torch.Tensor,
               g: GraphTensors | None, *, keep_layer_inputs: bool = False) -> Encoded:
        """``(|C|, F)`` and ``(G,)`` -> node embeddings.  ``g`` is unused when ``layers == 0``."""
        n = x_nodes.shape[0]
        h = self.embed(torch.cat([x_nodes, x_global.unsqueeze(0).expand(n, -1)], dim=1))
        inputs, ef = [], None
        if self.cfg.layers:
            if g is None:
                raise ValueError("layers > 0 needs the graph; pass graph_tensors(...)")
            pairs = bool(getattr(CONVS[self.cfg.conv], "needs_pairs", False))
            if pairs:
                ef = pair_edge_features(g)
            use_ckpt = self.cfg.checkpoint_encoder and torch.is_grad_enabled()
            for conv in self.convs:
                if keep_layer_inputs:
                    inputs.append(h)
                args = (h, g, ef) if pairs else (h, g)
                if use_ckpt:
                    h = torch.utils.checkpoint.checkpoint(conv, *args, use_reentrant=False)
                else:
                    h = conv(*args)
        return Encoded(h=h, keys=self.key(h), n_clusters=n,
                       layer_inputs=inputs if keep_layer_inputs else None, edge_feat=ef)

    def edge_logits(self, enc: Encoded, g: GraphTensors, pos: torch.Tensor) -> list:
        """One ``(len(pos),)`` gate-logit tensor per ``PairConv`` layer, at that layer's input.

        The co-parent BCE is applied to these, so it supervises the attention the layer actually
        uses rather than a head beside it.
        """
        if not getattr(CONVS[self.cfg.conv], "needs_pairs", False):
            raise ValueError(f"conv {self.cfg.conv!r} has no edge gates")
        if enc.layer_inputs is None:
            raise ValueError("edge_logits needs encode(..., keep_layer_inputs=True)")
        return [conv.gate_logits(hin, g, enc.edge_feat, pos)
                for conv, hin in zip(self.convs, enc.layer_inputs)]

    # ------------------------------------------------------------------ decoder

    def logits(self, enc: Encoded, *, accepted_sum: torch.Tensor, accepted_n: torch.Tensor,
               last_action: torch.Tensor, scalars: torch.Tensor,
               dynamic: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``(R, |C|)`` masked logits for ``R`` rollouts standing at their own states.

        Every rollout of one instance shares ``enc``, so the ``|C| x hidden`` key matrix is
        projected once per instance rather than once per rollout per step -- 50 steps x K rollouts
        of a ``(|C|, hidden)`` projection is the difference between a decoder that costs nothing
        and one that costs as much as the encoder.

        The **dynamic** features enter as an additive per-candidate term rather than by being
        concatenated into the keys, for the same reason: concatenation would make the keys depend
        on the rollout and the step, and the ``(R, |C|, hidden)`` tensor that follows is what a
        50-step episode cannot afford to keep for backward.
        """
        d = self.cfg.hidden
        mean = accepted_sum / accepted_n.clamp_min(1.0).unsqueeze(1)
        prev = torch.where(last_action.unsqueeze(1) >= 0,
                           enc.h.index_select(0, last_action.clamp_min(0)),
                           self.no_action.unsqueeze(0).expand(last_action.shape[0], d))
        q = self.ctx(torch.cat([mean, prev, scalars], dim=1))            # (R, hidden)
        score = q @ enc.keys.transpose(0, 1) / math.sqrt(d)              # (R, |C|)
        score = score + self.dyn(dynamic).squeeze(-1)
        score = self.cfg.tanh_clip * torch.tanh(score)
        return score.masked_fill(~mask, float("-inf"))

    def cluster_logits(self, enc: Encoded) -> torch.Tensor:
        """W4.2's pointwise head: ``(|C|,)`` logits for ``P(A in C(N_true))``."""
        return self.cls(enc.h).squeeze(-1)


def load_transferable(policy: Policy, state_dict: dict) -> dict:
    """Load every parameter whose **shape matches**, and say which ones did not.  No silent skips.

    W8 changes exactly one tensor's shape: ``dyn`` goes from ``(1, 3)`` to ``(1, 4)`` because the
    swap MDP has a fourth per-candidate feature.  Everything the encoder learned -- ``embed``, the
    three ``SegmentConv`` layers, ``key`` -- and the whole decoder context ``ctx`` are unchanged,
    the last of these only because :data:`opc.rl.swap.N_SWAP_SCALARS` was deliberately kept at
    four.  So a W8 policy can start from ``rl_run_bc_oracle/bc_init.pt`` rather than from noise.

    Returns ``{"loaded": [...], "skipped": [...]}``.  ``torch``'s own ``strict=False`` would do
    the loading and report only *missing* keys, not shape *mismatches*, which is the case that
    matters here: a silently unloaded ``dyn`` and a silently unloaded ``key`` look identical from
    the caller and mean completely different things.  Callers are expected to assert on
    ``skipped``.
    """
    own = policy.state_dict()
    loaded, skipped = [], []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    policy.load_state_dict(own)
    return {"loaded": sorted(loaded), "skipped": sorted(skipped)}


def dynamic_features(support: np.ndarray, mask: np.ndarray, deg_sel: np.ndarray,
                     n_accepted: int, extra: np.ndarray | None = None) -> np.ndarray:
    """``float32[|C|, N_DYNAMIC (+ extra)]`` -- RL_PLAN §3's per-step decoder features, in one pass.

    ``extra`` appends ``float32[|C|, e]`` of **static** per-candidate columns, and ``None`` -- the
    default -- reproduces the three-column output byte for byte, which is what keeps every landed
    caller (``w4_bc``, ``w5_train``, ``w6_eval``, ``w8_swap``) computing exactly what it did.
    Its use is W11's pointwise prior: A91 measured that the information needed for exact recovery
    is already in W10's scores and only the tie-break at the margin is wrong, so handing the
    decoder that score as a candidate feature lets it learn the *correction* rather than the
    ranking.  ``PolicyConfig.n_dynamic`` must be widened to match, and ``load_transferable`` then
    reports ``dyn`` as skipped rather than loading a differently-shaped tensor.

    1. ``deg_sel / max(1, |S_t|)`` -- overlap-degree **within the accepted set**, straight out of
       :meth:`opc.rl.env.ClusterEnv.legal_mask`, which computes it anyway.
    2. ``w(A) / max_{legal} w`` -- support renormalised over the *legal* set, which is the quantity
       that changes as the set shrinks; the instance-level support rank is already a static node
       feature and says nothing about the current step.
    3. the **greedy indicator**: 1 at the argmax-support legal candidate, 0 elsewhere.  This is the
       whole of ``greedy-by-support``'s rule handed to the policy as one bit -- deliberate,
       because A67/A54 measure that baseline as selecting exactly the truth's cardinality on 60 of
       60 cells and the exact reticulation count on 60 of 60, and W4.1's behaviour cloning starts
       from it.  A policy that has to *rediscover* argmax-support from embeddings would spend its
       budget on the part that is already solved (RL_PLAN §5).

    A rank over the legal set would be the more informative version of (2) and is not used: it is
    ``O(|C| log |C|)`` per rollout per step, which at 128 rollouts x 50 steps is the step.
    """
    n = support.shape[0]
    width = N_DYNAMIC + (0 if extra is None else int(np.asarray(extra).shape[1]))
    out = np.zeros((n, width), dtype=np.float32)
    out[:, 0] = deg_sel / max(1.0, float(n_accepted))
    idx = np.flatnonzero(mask)
    if idx.size:
        w = support[idx]
        top = float(w.max())
        if top > 0:
            out[idx, 1] = w / top
        out[int(idx[np.argmax(w)]), 2] = 1.0
    if extra is not None:
        out[:, N_DYNAMIC:] = np.asarray(extra, dtype=np.float32)
    return out
