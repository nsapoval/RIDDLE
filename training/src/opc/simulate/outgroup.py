"""Outgroup grafting, and rooting an estimated gene tree back on that outgroup.

Cluster extraction reads *rooted* gene trees (§3), but IQ-TREE returns unrooted ones.  Something
in the alignment has to carry the root, so each simulated gene tree gets a synthetic outgroup
leaf grafted above its MRCA before sequences are simulated, and the inferred tree is rooted on
that leaf and the leaf pruned before extraction.

**Grafted onto the gene tree, not added to the species network** (SPEC-AMENDMENTS A15).  The
alternative -- an extra leaf above the network root, re-simulated through the coalescent -- would
invalidate the cached gene trees, and would perturb the ILS axis, because :func:`networks.scaled`
normalises on the median internal branch and an outgroup stem is an internal branch.  Grafting
instead keeps the ingroup taxon set byte-identical to the true-gene-tree arm, so the two arms are
paired locus for locus.  The price is that the outgroup never sorts discordantly and is never
missing, so **it contributes no rooting error of its own**: measured over 28,315 loci whose ingroup
topology was recovered exactly, the root split was right 28,315 times.  ``frac_root_split_correct``
therefore tracks topology error near the root, not rooting error -- real outgroup or midpoint
rooting carries error this design cannot express.

The divergence *time* is fixed per cell -- an outgroup diverges once, not once per locus -- at
``T_out = (1 + alpha) * p99(root heights)``.  The outgroup's branch *length* then varies across
loci exactly as a real one does, and every grafted tree stays ultrametric.

Coalescent units become substitutions/site here rather than via AliSim's ``--branch-scale``,
which is *silently ignored* in partition mode (``results/validation/alisim_probe.json``).  Doing
it in Python keeps one convention, testable in isolation.

Everything here is hand-rolled rather than dendropy, for the same reason
:mod:`opc.clusters.extract` is: dendropy costs milliseconds per tree and this runs over millions.
dendropy appears in the tests instead, as the independent oracle.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

__all__ = [
    "OUTGROUP_LABEL",
    "parse_newick",
    "write_newick",
    "node_masks",
    "root_height",
    "graft_outgroup",
    "reroot_on_outgroup",
    "drop_outgroup",
    "graft_file",
    "root_and_prune_file",
]

OUTGROUP_LABEL = "OUT"

_SPLIT = re.compile(r"([(),;])")


def parse_newick(s: str):
    """``(parent, length, label, children)`` node arrays for one Newick string.

    Node 0 is the root and ``parent[i] < i`` for every other node, so a single reverse pass
    visits children before parents.  Unrooted (multifurcating) input parses fine; the "root" is
    then just the node the string happens to be written from.
    """
    parent: list[int] = [-1]
    length: list[float] = [0.0]
    label: list[str | None] = [None]
    children: list[list[int]] = [[]]
    stack = [0]
    started = False
    current: int | None = None

    def add(p: int) -> int:
        parent.append(p)
        length.append(0.0)
        label.append(None)
        children.append([])
        children[p].append(len(parent) - 1)
        return len(parent) - 1

    for tok in _SPLIT.split(s):
        if not tok or tok.isspace():
            continue
        if tok == "(":
            if started:
                stack.append(add(stack[-1]))
            started = True  # the outermost "(" is node 0, already created
            current = None
        elif tok == ")":
            current = stack.pop()
            if not stack:
                stack.append(current)  # closing the root; keep the stack non-empty
        elif tok == ",":
            current = None
        elif tok == ";":
            break
        else:
            name, _, blen = tok.partition(":")
            name = name.strip()
            if current is None:
                current = 0 if not started else add(stack[-1])
                started = True
            if name:
                label[current] = name
            blen = blen.strip()
            if blen:
                try:
                    length[current] = float(blen)
                except ValueError:
                    length[current] = 0.0
    if len(parent) == 1 and label[0] is None:
        raise ValueError(f"could not parse newick: {s[:80]!r}")
    return parent, length, label, children


def write_newick(parent, length, label, children, scale: float = 1.0, digits: int = 8) -> str:
    """Re-emit a parsed tree, multiplying every branch length by ``scale``.

    Returns the root subtree without the trailing ``;`` so callers can graft onto it.
    """
    parts: list[str] = [""] * len(parent)
    stack = [(0, False)]
    while stack:
        i, done = stack.pop()
        if not children[i]:
            parts[i] = label[i] or ""
        elif done:
            parts[i] = "(" + ",".join(
                f"{parts[c]}:{length[c] * scale:.{digits}f}" for c in children[i]) + ")"
        else:
            stack.append((i, True))
            for c in children[i]:
                stack.append((c, False))
    return parts[0]


def node_masks(parent, label, children, taxon_index: dict[str, int]) -> list[int]:
    """Leaf-set bitmask per node.  One reverse pass, since ``parent[i] < i``."""
    mask = [0] * len(parent)
    for i in range(len(parent) - 1, -1, -1):
        if not children[i]:
            name = label[i]
            if name is None:
                raise ValueError(f"unlabelled leaf at node {i}")
            mask[i] = 1 << taxon_index[name]
        if i:
            mask[parent[i]] |= mask[i]
    return mask


def root_height(parent, length, children) -> float:
    """Max root-to-leaf distance -- the MRCA age, since the gene trees are ultrametric."""
    depth = [0.0] * len(parent)
    best = 0.0
    for i in range(1, len(parent)):
        depth[i] = depth[parent[i]] + length[i]
        if not children[i] and depth[i] > best:
            best = depth[i]
    return best


def graft_outgroup(newick: str, d: float, t_out: float, label: str = OUTGROUP_LABEL,
                   scale: float = 1.0, digits: int = 8) -> str:
    """``(<tree>:d, OUT:t_out);`` with every length multiplied by ``scale``.

    ``d`` and ``t_out`` are given in the tree's own units and scaled alongside it.
    """
    parent, length, lab, children = parse_newick(newick)
    core = write_newick(parent, length, lab, children, scale=scale, digits=digits)
    return (f"({core}:{d * scale:.{digits}f},"
            f"{label}:{t_out * scale:.{digits}f});")


def _resolve(x: int, f: int, adj, elen=None):
    """Descend through degree-2 nodes, so suppressing the old root leaves no unary node.

    Returns ``(x, f, ns, length)``.  ``length`` is the **sum** of the edges walked through and is
    0.0 when ``elen`` is not given.  Summing is the whole reason this is not a one-liner: the
    degree-2 nodes being suppressed take their edges with them, and the single edge that replaces
    the chain has to carry their total or the tree silently shrinks.
    """
    acc = 0.0 if elen is None else elen[(x, f)]
    ns = [y for y in adj[x] if y != f]
    while len(ns) == 1:
        if elen is not None:
            acc += elen[(ns[0], x)]
        x, f = ns[0], x
        ns = [y for y in adj[x] if y != f]
    return x, f, ns, acc


def reroot_on_outgroup(newick: str, label: str = OUTGROUP_LABEL, *,
                       keep_lengths: bool = False, digits: int = 10) -> str:
    """Root on the outgroup's pendant edge and delete the outgroup.

    Rooting an unrooted tree at ``OUT`` and then deleting ``OUT`` is exactly "re-emit the tree
    rooted at ``OUT``'s unique neighbour, excluding ``OUT``" -- no edge splitting needed.

    **Branch lengths are dropped by default, and that default is now a liability rather than a
    saving.**  It was chosen when :func:`opc.clusters.extract.clusters_from_newick` stripped
    ``:length`` anyway and halving the stored file was pure gain.  W13 then built twenty-one node
    features out of those lengths, and a length-free tree does not fail anywhere: it produces
    ``blen_* = 0``, ``bdepth_* = 1``, ``bsib_* = 0`` and a ``brank_*`` that is the **Newick write
    order** -- a column that varies per cluster, looks like a feature and carries nothing.  Pass
    ``keep_lengths=True`` for anything W13 will read.  The default is retained because
    :func:`root_and_prune_file` and :func:`opc.rl.external.prepare_gene_trees` produce landed
    artifacts with it and must keep reproducing them byte for byte.

    Raises if ``OUT`` is absent: a silently unrooted tree would yield wrong clusters without
    erroring anywhere downstream.
    """
    parent, length, lab, children = parse_newick(newick)
    try:
        out = next(i for i, nm in enumerate(lab) if nm == label)
    except StopIteration:
        raise ValueError(f"outgroup {label!r} not found in tree") from None
    if children[out]:
        raise ValueError(f"outgroup {label!r} is not a leaf")

    adj = [list(children[i]) + ([parent[i]] if parent[i] >= 0 else []) for i in range(len(parent))]
    if len(adj[out]) != 1:
        raise ValueError(f"outgroup {label!r} has degree {len(adj[out])}, expected 1")

    # **Keyed on the edge, not on the node.**  Rerooting reverses every edge on the path from the
    # written root to the new one, and ``length[i]`` is the length of the edge above ``i`` *in the
    # written orientation* only.  Reading it by node would give a reversed edge its child's length.
    elen = None
    if keep_lengths:
        elen = {}
        for i in range(1, len(parent)):
            elen[(i, parent[i])] = elen[(parent[i], i)] = length[i]

    memo: dict[tuple[int, int], tuple[str, float]] = {}
    root = adj[out][0]
    stack = [(root, out, False)]
    while stack:
        y0, f0, done = stack.pop()
        x, f, ns, acc = _resolve(y0, f0, adj, elen)
        if not ns:
            memo[(y0, f0)] = (lab[x] or "", acc)
        elif done:
            memo[(y0, f0)] = ("(" + ",".join(
                _emit(memo[(y, x)], keep_lengths, digits) for y in ns) + ")", acc)
        else:
            stack.append((y0, f0, True))
            for y in ns:
                stack.append((y, x, False))
    return memo[(root, out)][0] + ";"


def _emit(entry: tuple[str, float], keep_lengths: bool, digits: int) -> str:
    """One child of a Newick group: ``subtree`` or ``subtree:length``."""
    body, ell = entry
    return f"{body}:{ell:.{digits}f}" if keep_lengths else body


def drop_outgroup(newick: str, label: str = OUTGROUP_LABEL, *,
                  keep_lengths: bool = True, digits: int = 10) -> str:
    """Delete ``label`` from an **already rooted** tree, keeping the root where it is.

    The sibling of :func:`reroot_on_outgroup`, and the difference is the whole point.  That one
    *re-roots* on the outgroup's pendant edge, which is the only thing you can do to an unrooted
    estimated tree; this one keeps the rooting the input already has and merely removes a leaf.

    **Which to use is decided by what the tree is, not by taste.**  On this study's simulated arm
    the gene trees are rooted at the coalescent root and are ultrametric, and W13's ``bdepth_*``
    normalises a node's crown height by the tree's root-to-tip height -- which is the *crown age*
    exactly when the tree is ultrametric, and the longest path from an arbitrary internal node when
    it is not.  Re-rooting a rooted, ultrametric gene tree on a leaf destroys that: measured on
    ``camus-data``, whose ``OUT`` is a simulated lineage rather than a grafted one, deep coalescence
    puts ``OUT`` inside the ingroup on 24-71% of true loci, so outgroup rooting leaves **0 of 1000**
    loci ultrametric where this function leaves 1000 of 1000.  ``bdepth_*`` is also the block the
    trained head leans on hardest (W13: zeroing it costs 0.195 held-out F1 at n = 25), so the choice
    is not cosmetic.

    The candidate set barely notices either way -- ``|C_tau2|`` moves by <= 2 clusters and the
    support vector is *identical* on the cells measured -- which is what makes the two conventions
    a clean one-variable contrast rather than two different experiments.

    Raises on a trifurcating root: "keep the rooting" is not a statement one can make about a tree
    that has none, and silently taking the written trifurcation as a root is A82's failure.
    """
    parent, length, lab, children = parse_newick(newick)
    if len(children[0]) != 2:
        raise ValueError(
            f"drop_outgroup needs a rooted tree; the root has out-degree {len(children[0])}. "
            f"An unrooted tree has no rooting to keep -- use reroot_on_outgroup(keep_lengths=True).")
    try:
        out = next(i for i, nm in enumerate(lab) if nm == label)
    except StopIteration:
        raise ValueError(f"outgroup {label!r} not found in tree") from None
    if children[out]:
        raise ValueError(f"outgroup {label!r} is not a leaf")

    # Detach, then suppress every node the detachment left unary, **summing** the edges walked
    # through.  A chain of suppressions has to carry its total or the tree silently shrinks --
    # ``_resolve``'s rule, and the reason this is not a string edit.
    children[parent[out]].remove(out)

    def emit(i: int) -> tuple[str, float]:
        """``(subtree, length of the edge above it)``, unary nodes already suppressed."""
        acc = length[i]
        while len(children[i]) == 1:
            i = children[i][0]
            acc += length[i]
        if not children[i]:
            return lab[i] or "", acc
        body = "(" + ",".join(
            _emit(emit(c), keep_lengths, digits) for c in children[i]) + ")"
        return body, acc

    # The root's own edge is not written, so its accumulated length is discarded here and only
    # here -- which is exactly right when ``OUT`` was a root child, because then the surviving
    # sibling *becomes* the root and the edge that used to lead to it is above the tree.
    return emit(0)[0] + ";"


def _quantile(values, q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _stream(path, start: int = 0, count: int | None = None):
    """Yield stripped non-blank lines ``[start, start + count)`` by *line* index."""
    stop = None if count is None else start + count
    with _open(path) as fh:
        for i, line in enumerate(fh):
            if stop is not None and i >= stop:
                return
            line = line.strip()
            if not line or i < start:
                continue
            yield line


def graft_file(src_gz, dst, alpha: float = 0.5, branch_scale: float = 1.0, start: int = 0,
               count: int | None = None, cell_m: int | None = None,
               label: str = OUTGROUP_LABEL, floor_frac: float = 0.05, digits: int = 8) -> dict:
    """Graft an outgroup onto trees ``[start, start+count)`` of ``src_gz``, writing to ``dst``.

    Two streaming passes: the first fixes ``T_out`` from the root-height distribution, the second
    grafts and rescales.

    ``T_out`` is always taken over the whole cell -- ``[0, cell_m)``, defaulting to the requested
    slice -- never over the slice alone.  An outgroup diverges once, so its divergence time must
    not depend on how the cell happened to be split into chunks; deriving it per chunk would also
    make a resumed run with a different chunk size silently produce different data.
    """
    src_gz, dst = Path(src_gz), Path(dst)
    if cell_m is None:
        cell_m = (start + count) if count is not None else None
    heights = []
    for line in _stream(src_gz, 0, cell_m):
        parent, length, _, children = parse_newick(line)
        heights.append(root_height(parent, length, children))
    if len(heights) <= start:
        raise ValueError(f"{src_gz}: only {len(heights)} trees, cannot start at {start}")

    p99 = _quantile(heights, 0.99)
    t_out = (1.0 + alpha) * p99
    n_clamped = n_written = 0
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wt") as fh:
        for j, line in enumerate(_stream(src_gz, start, count)):
            h = heights[start + j]
            d = t_out - h
            if d < floor_frac * h:
                d = floor_frac * h
                n_clamped += 1
            fh.write(graft_outgroup(line, d, h + d, label,
                                    scale=branch_scale, digits=digits) + "\n")
            n_written += 1
    tmp.rename(dst)
    return {
        "n_loci": n_written,
        "outgroup_alpha": alpha,
        "t_out": t_out,
        "outgroup_subs_per_site": t_out * branch_scale,
        "root_height_mean": sum(heights) / len(heights),
        "root_height_min": min(heights),
        "root_height_max": max(heights),
        "root_height_p99": p99,
        "n_outgroup_depth_clamped": n_clamped,
    }


def _rf(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a ^ b) / (len(a) + len(b))


def root_and_prune_file(src, dst_gz, taxa, true_gz=None, start: int = 0,
                        label: str = OUTGROUP_LABEL) -> dict:
    """Root every tree of ``src`` on the outgroup, prune it, write one per line to ``dst_gz``.

    ``taxa`` is the ingroup label list; every emitted tree must carry exactly that set.  That
    invariant is what catches a dropped taxon: ``clusters_from_newick`` raises on an *unknown*
    label but is silent on a *missing* one, so a taxon IQ-TREE quietly dropped would corrupt the
    clusters without erroring anywhere.

    With ``true_gz`` the true trees stream in lockstep and the rooted RF and root-split accuracy
    come out for free, since both cluster sets are computed anyway.
    """
    src, dst_gz = Path(src), Path(dst_gz)
    index = {t: i for i, t in enumerate(taxa)}
    expected = frozenset(taxa)
    full = (1 << len(index)) - 1

    truth = _stream(true_gz, start, None) if true_gz is not None else None
    rfs: list[float] = []
    n_root_ok = n_root_scored = n_polytomy = n_written = 0

    tmp = dst_gz.with_name(dst_gz.name + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        for line in _stream(src):
            rooted = reroot_on_outgroup(line, label)
            parent, _, lab, children = parse_newick(rooted)
            got = frozenset(lab[i] for i in range(len(parent)) if not children[i])
            if got != expected:
                raise ValueError(
                    f"{src.name}: estimated tree carries {len(got)} taxa, expected "
                    f"{len(expected)} (missing {sorted(expected - got)[:5]}, "
                    f"extra {sorted(got - expected)[:5]})")
            fh.write(rooted + "\n")
            n_written += 1
            if len(children[0]) > 2:
                n_polytomy += 1
            if truth is None:
                continue
            try:
                tline = next(truth)
            except StopIteration:
                truth = None
                continue
            masks = node_masks(parent, lab, children, index)
            est = {m for m in (masks[i] for i in range(len(parent)) if children[i])
                   if m != full and m & (m - 1)}
            tp, _, tl, tc = parse_newick(tline)
            tmask = node_masks(tp, tl, tc, index)
            tru = {m for m in (tmask[i] for i in range(len(tp)) if tc[i])
                   if m != full and m & (m - 1)}
            rfs.append(_rf(est, tru))
            if len(children[0]) == 2 and len(tc[0]) == 2:
                n_root_scored += 1
                n_root_ok += (min(masks[c] for c in children[0])
                              == min(tmask[c] for c in tc[0]))
    tmp.rename(dst_gz)

    stats = {"n_loci": n_written, "n_root_polytomy": n_polytomy}
    if rfs:
        stats |= {
            "rf_norm_mean": sum(rfs) / len(rfs),
            "rf_norm_median": _quantile(rfs, 0.5),
            "rf_norm_p90": _quantile(rfs, 0.9),
            "frac_rf_zero": sum(v == 0.0 for v in rfs) / len(rfs),
            "frac_root_split_correct": n_root_ok / n_root_scored if n_root_scored else 0.0,
            "n_rf_scored": len(rfs),
        }
    return stats
