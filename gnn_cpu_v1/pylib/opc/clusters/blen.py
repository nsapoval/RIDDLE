"""Gene-tree branch-length statistics, per candidate cluster.  W13.

``opc.clusters.extract`` deletes ``:length`` before it builds a single bitset (``_STRIP``), and so
does ``opc.mpl.triples``.  Every node feature the supervised arm has ever seen is therefore a
function of *counts* and of the overlap graph, and the branch lengths PhyloCoalSimulations wrote
have been sitting on disk unread.

Under the MSC that is a real omission.  A cluster that is a species-tree clade subtends a **long**
internal branch in the gene trees that display it; a cluster produced by ILS is two coalescences
that happened close together, so its branch is short.  Support cannot express the difference -- a
cluster displayed in 300 loci because it is real and one displayed in 300 loci by deep coalescence
are the same number.

**The unit problem, and why it does not need solving.**  These lengths are in coalescent units and
would be in substitutions per site on real data.  Rather than guess a conversion, every length is
divided by the **maximum branch length in its own tree**, which is exactly invariant to the
per-tree scalar that separates the two systems, and so are ``bdepth_*`` and ``bsib_*`` -- a ratio
does not care what its numerator and denominator are measured in.  ``brank_*`` goes further and is
invariant to **any** monotone per-tree transform, not only a rescale; that is the strongest
statement any of these can make and it is why the block is here beside ``blen_*`` rather than
instead of it.  ``tests/test_blen.py`` pins all four invariances, including the one ``bsib_*``
does **not** have.

**No taxon label enters any feature.**  A cluster is a bitmask and every statistic here is a scalar
attached to that mask; relabelling the taxa permutes the rows and changes nothing else.

**What is measured, and what it is worth.**  Held-out band AUC over the existing ten node features
(A91's protocol -- fit on 8 cells, band = top ``3 * n_true``, test on 6):

===================================== ======== ======== ========
block                                      b25      b50     b250
===================================== ======== ======== ========
``blen_*``   ell / max edge in tree     +0.0050  +0.0038  -0.0002
``brank_*``  within-tree rank           +0.0026  +0.0009  -0.0016
``bdepth_*`` crown height               +0.0113  +0.0076  +0.0075
``bsib_*``   log sibling ratio          +0.0004  +0.0002  +0.0013
all four together                       +0.0154  +0.0119  +0.0089
===================================== ======== ======== ========

against A91's four rejected features at **+0.0011**.  Two things that shape how a result here is
read, both pre-registered: ``bdepth`` extrapolates and ``blen``/``brank`` do **not** -- positive in
range, null to negative at n = 250, which is the A94/A96/A97 signature; and ``bsib`` is near-null
everywhere and is the first block to drop.

``bconf_*``, the fifth block of the W13 feature set, is **not here**: it is built from the overlap
CSR and the support vector, needs no parse at all, and lives in :mod:`opc.rl.instances`.
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TREE_FEATURES",
    "TREE_BLOCKS",
    "GeneTree",
    "parse_with_lengths",
    "accumulate_locus",
    "cluster_tree_stats",
]

#: The same tokeniser :func:`opc.clusters.extract.clusters_from_newick` uses.  What is *not* here
#: is that function's ``_STRIP``, which is the whole point: the lengths survive.
_SPLIT = re.compile(r"([(),])")

#: Guards a ``log(0/0)`` on a zero-length branch beside a zero-length sister.  Both are 0 in
#: exact arithmetic on a degenerate tree, and ``log(eps/eps) == 0`` is the answer that says
#: "these two are the same", which is correct.
_EPS = 1e-12

#: The four tree-derived blocks, in column order.  ``min``/``max`` are order statistics over a
#: sample whose size **is** the support, so they partly re-encode it -- measured, ``blen_min`` has
#: full-set AUC 0.10 (anti-correlated) and band AUC 0.44-0.50, i.e. nearly pure artefact.  They are
#: kept because they were asked for and cost four columns, but ``q25``/``med``/``q75`` are the
#: support-robust members and are where a real effect should show.  ``w10_gnn_diag.py``'s
#: redundancy census reports the correlation without being told to.
TREE_BLOCKS: dict[str, tuple[str, ...]] = {
    # ell / max edge length in the same tree.  Invariant to a per-tree rescale.
    "blen": ("blen_min", "blen_q25", "blen_med", "blen_mean", "blen_q75", "blen_max"),
    # rank of ell among that tree's *internal* branches, scaled to [0, 1].  Uniform by
    # construction, so invariant to any monotone per-tree transform and unable to drift with n.
    "brank": ("brank_q25", "brank_med", "brank_q75"),
    # crown height of the cluster's node above the tips, / root-to-tip height.  *When* the clade
    # coalesced rather than how long it persisted -- a different axis from the subtending branch.
    "bdepth": ("bdepth_min", "bdepth_q25", "bdepth_med", "bdepth_mean",
               "bdepth_q75", "bdepth_max"),
    # tanh(log(ell(A) / max ell(sister))) in the same tree.  A ratio inside one tree, so scale-free
    # with no denominator choice at all, and directly the quantity ILS shrinks.
    #
    # **The tanh is not decoration.**  The raw log-ratio runs -6.4 to +4.8 on a real x25 cell while
    # every other feature in ``NODE_FEATURES`` sits in roughly [0, 1] -- and they all enter one
    # ``nn.Linear`` together (``policy.Policy.embed``), so a column an order of magnitude wider
    # dominates the initial scale of that projection for reasons that have nothing to do with what
    # it carries.  ``tanh`` is monotone, so every invariance claimed for the log-ratio survives it
    # exactly, and the ordering the feature exists to express is untouched.
    "bsib": ("bsib_q25", "bsib_med", "bsib_q75"),
}

TREE_FEATURES: tuple[str, ...] = tuple(n for b in TREE_BLOCKS.values() for n in b)


@dataclass
class GeneTree:
    """One parsed rooted gene tree, lengths retained.

    Node 0 is an **artificial parent** of the real root, which exists so the stack machine never
    has to special-case an empty stack.  It carries the full taxon set and a zero-length edge, and
    every consumer here selects on ``parent > 0`` -- which excludes node 0 *and* the real root
    together, those being exactly the two nodes with no parent edge in the Newick.
    """

    mask: list[int]           # leaf-set bitmask per node
    plen: np.ndarray          # float64[V] -- parent-edge length; 0.0 where there is none
    parent: np.ndarray        # int64[V]   -- -1 for node 0
    children: list[list[int]]
    depth: np.ndarray         # float64[V] -- path length from the real root

    @property
    def n_nodes(self) -> int:
        return len(self.mask)

    def edge_nodes(self) -> list[int]:
        """Every node that has a parent edge in the Newick."""
        return [i for i in range(len(self.mask)) if self.parent[i] > 0]

    def internal_nodes(self) -> list[int]:
        """Every node with children **and** a parent edge -- the clusters this module scores."""
        return [i for i in range(len(self.mask)) if self.children[i] and self.parent[i] > 0]


def parse_with_lengths(newick: str, taxon_index: dict[str, int]) -> GeneTree:
    """One rooted Newick tree -> :class:`GeneTree`, branch lengths retained.

    Structurally this is ``extract.clusters_from_newick``'s stack machine with the ``_STRIP``
    substitution removed and an explicit node array built instead of a mask list.  The masks it
    produces must agree with that function exactly -- ``scripts/probe_blen.py`` checks it on real
    cells of every phase, and ``tests/test_blen.py`` on fixtures -- because a silently different
    parse would give every cluster the wrong branch and nothing would raise.

    Unrooted or multifurcating input is accepted structurally, as there; a missing length reads as
    0.0 rather than raising, because a Newick without lengths is a valid Newick.
    """
    mask: list[int] = []
    plen: list[float] = []
    parent: list[int] = []
    children: list[list[int]] = []

    def new(p: int) -> int:
        mask.append(0)
        plen.append(0.0)
        parent.append(p)
        children.append([])
        if p >= 0:
            children[p].append(len(mask) - 1)
        return len(mask) - 1

    new(-1)                     # the artificial parent of the real root
    stack: list[int] = [0]
    closed: int | None = None

    for tok in _SPLIT.split(newick):
        if not tok:
            continue
        if tok == "(":
            if closed is not None:      # "()(": not Newick, but merge rather than lose the mask
                mask[stack[-1]] |= mask[closed]
                closed = None
            stack.append(new(stack[-1]))
        elif tok == ")":
            if closed is not None:      # "))" -- the inner group carried no ":length"
                mask[stack[-1]] |= mask[closed]
                closed = None
            if len(stack) == 1:
                raise ValueError("unbalanced parentheses in newick")
            closed = stack.pop()
        elif tok == ",":
            if closed is not None:
                mask[stack[-1]] |= mask[closed]
                closed = None
        else:
            body = tok.strip().rstrip(";").strip()
            if not body and closed is None:
                continue
            name, _, rest = body.partition(":")
            length = rest.split(":")[0]           # ":len[:support[:gamma]]"
            value = float(length) if length else 0.0
            if closed is not None:
                # The text directly after a ")" is that internal node's "[label]:length".
                plen[closed] = value
                mask[stack[-1]] |= mask[closed]
                closed = None
            elif name:
                leaf = new(stack[-1])
                mask[leaf] = 1 << taxon_index[name]
                plen[leaf] = value
                mask[stack[-1]] |= mask[leaf]

    if closed is not None:
        mask[stack[-1]] |= mask[closed]
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses in newick")

    par = np.asarray(parent, dtype=np.int64)
    ln = np.asarray(plen, dtype=np.float64)
    dep = np.zeros(len(mask), dtype=np.float64)
    for i in range(1, len(mask)):
        dep[i] = dep[par[i]] + ln[i]
    return GeneTree(mask=mask, plen=ln, parent=par, children=children, depth=dep)


def accumulate_locus(tree: GeneTree, row_of: dict[bytes, int], width: int,
                     acc: list[list[list[float]]], n_obs: np.ndarray) -> bool:
    """Add one locus's four statistics to ``acc``, for the clusters that are rows of the ground set.

    Returns **True when the locus carried no branch length at all**.  That is not a curiosity: a
    Newick without lengths is a valid Newick, so this function cannot raise on one -- it takes the
    ``l_max = 1.0`` / ``height = 1.0`` fallbacks below and emits ``blen_* = 0``, ``bdepth_* = 1``,
    ``bsib_* = 0`` and a ``brank_*`` that is a stable argsort of equal values, i.e. the order the
    nodes happen to appear in the string.  Three constants and one plausible-looking artefact, with
    ``n_obs`` still exactly right, so the caller's support cross-check passes and nothing anywhere
    says the features are empty.  :func:`cluster_tree_stats` counts these and
    ``opc.rl.instances._tree_stats`` refuses on them.

    **A mask appearing twice in one locus is deduplicated by maximum length.**  That mirrors
    ``SupportCounter.add_locus``'s ``set(masks)``, which is what makes ``n_obs`` reproduce
    ``gs.support`` exactly -- and *that* equality is this module's cross-validation against an
    independent implementation, checked in :func:`cluster_tree_stats`'s caller.  It can only fire
    on a unifurcation; ``n_nonbinary_loci`` is 0 on every grid here and the probe asserts it.
    """
    plen, par, ch, dep, mask = tree.plen, tree.parent, tree.children, tree.depth, tree.mask
    edges = tree.edge_nodes()
    if not edges:
        return True
    l_max = max(plen[i] for i in edges)
    degenerate = bool(l_max <= 0.0)   # np.float64 comparison; the annotation says bool
    if degenerate:
        l_max = 1.0
    height = float(dep.max()) if dep.size else 0.0
    if height <= 0.0:
        height = 1.0

    inner = tree.internal_nodes()
    if not inner:
        return degenerate
    ils = np.asarray([plen[i] for i in inner], dtype=np.float64)
    order = np.argsort(np.argsort(ils, kind="stable"), kind="stable")
    denom = max(1, len(inner) - 1)

    best: dict[int, tuple[float, float, float, float]] = {}
    for pos, node in enumerate(inner):
        row = row_of.get(int(mask[node]).to_bytes(width, "little"))
        if row is None:
            continue
        length = float(plen[node])
        prev = best.get(row)
        if prev is not None and prev[0] >= length:
            continue
        sibs = [plen[c] for c in ch[par[node]] if c != node]
        sister = max(sibs) if sibs else 0.0
        best[row] = (
            length,
            float(order[pos]) / denom,
            (height - float(dep[node])) / height,
            math.tanh(math.log((length + _EPS) / (sister + _EPS))),
        )

    for row, (length, rank, crown, sib) in best.items():
        acc[0][row].append(length / l_max)
        acc[1][row].append(rank)
        acc[2][row].append(crown)
        acc[3][row].append(sib)
        n_obs[row] += 1
    return degenerate


def _quantile(sorted_a: np.ndarray, p: float) -> float:
    """numpy's default ('linear') quantile on an already-sorted array.

    Written out rather than calling ``np.quantile`` per row because that re-sorts: at n = 250 this
    runs over ~12,000 rows x 4 blocks a cell.  ``tests/test_blen.py`` pins it against
    ``np.quantile`` on random inputs, which is the only reason it is allowed to exist.
    """
    n = sorted_a.size
    if n == 1:
        return float(sorted_a[0])
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return float(sorted_a[lo] + (h - lo) * (sorted_a[hi] - sorted_a[lo]))


def _six(lists: list[list[float]], n_rows: int) -> np.ndarray:
    out = np.zeros((n_rows, 6), dtype=np.float64)
    for row, values in enumerate(lists):
        if not values:
            continue
        a = np.sort(np.asarray(values, dtype=np.float64))
        out[row] = (a[0], _quantile(a, 0.25), _quantile(a, 0.5),
                    a.mean(), _quantile(a, 0.75), a[-1])
    return out


def _three(lists: list[list[float]], n_rows: int) -> np.ndarray:
    out = np.zeros((n_rows, 3), dtype=np.float64)
    for row, values in enumerate(lists):
        if not values:
            continue
        a = np.sort(np.asarray(values, dtype=np.float64))
        out[row] = (_quantile(a, 0.25), _quantile(a, 0.5), _quantile(a, 0.75))
    return out


def cluster_tree_stats(path, taxon_index: dict[str, int], words: np.ndarray, *,
                       max_loci: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream a Newick file -> ``(float32[K, len(TREE_FEATURES)], int64[K], n_degenerate)``.

    The third return is the number of loci that carried **no branch length at all**, which is the
    one failure this module cannot express as a wrong number -- see :func:`accumulate_locus`.  It
    is returned rather than raised because a probe wants to measure it and a builder wants to
    refuse on it, and those are different jobs.

    The second return is the number of loci each row was observed in.  It must equal the ground
    set's own ``support`` column, and the caller asserts that: the two are produced by different
    parsers over the same file, so the equality is a genuine cross-check of this one rather than a
    restatement of it.

    ``max_loci = 0`` reads the whole file, matching ``opc.mpl.triples.count_newick_file``.

    Memory is not a concern despite the exact quantiles: each locus contributes exactly ``n - 1``
    internal nodes, so the accumulators hold at most ``m * (n - 1)`` values in total -- 249,000 at
    n = 250, m = 1000.  There is no sketching and no approximation anywhere in this module.
    """
    words = np.asarray(words)
    n_rows = int(words.shape[0])
    width = int(words.shape[1]) * 8
    row_of = {words[i].tobytes(): i for i in range(n_rows)}
    acc: list[list[list[float]]] = [[[] for _ in range(n_rows)] for _ in range(4)]
    n_obs = np.zeros(n_rows, dtype=np.int64)

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    seen = n_degenerate = 0
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if max_loci and seen >= max_loci:
                break
            seen += 1
            n_degenerate += accumulate_locus(
                parse_with_lengths(line, taxon_index), row_of, width, acc, n_obs)

    out = np.concatenate([
        _six(acc[0], n_rows), _three(acc[1], n_rows),
        _six(acc[2], n_rows), _three(acc[3], n_rows),
    ], axis=1)
    assert out.shape[1] == len(TREE_FEATURES), (out.shape, TREE_FEATURES)
    return np.ascontiguousarray(out, dtype=np.float32), n_obs, int(n_degenerate)
