"""Gene trees -> cluster bitsets with support counts.

Per §3 of the spec, the clusters of a rooted gene tree are the leaf sets of its internal nodes,
excluding singletons and the whole taxon set.  Support is the number of loci in which a cluster
appears.

Clusters are carried as Python ``int`` bitmasks throughout (arbitrary precision, so ``n > 64``
needs no special case) and converted to fixed-width bytes only when writing parquet.

The ``m`` axis of §5 is handled by *checkpoints*: simulate the largest locus count once, count
into the bin each locus falls in, then take a cumulative sum.  Support at ``m`` is then a column
lookup rather than a re-simulation.  Only clusters reaching support >= 2 at the largest ``m`` are
written -- lossless, because ``tau >= 2`` always and support is non-decreasing in ``m``.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "clusters_from_newick",
    "SupportCounter",
    "count_newick_file",
    "write_cluster_table",
    "read_cluster_table",
]

# Strip ``:length[:support[:gamma]]`` and any label sitting directly after a ``)``.
_STRIP = re.compile(r":[^,);]*|(?<=\))[^,);]*")
_SPLIT = re.compile(r"([(),])")


def clusters_from_newick(newick: str, taxon_index: dict[str, int], out: list[int]) -> list[int]:
    """Append the internal-node leaf sets of one rooted Newick tree, as int bitmasks.

    The root cluster and singletons are excluded.  Unrooted or multifurcating input is accepted
    structurally -- every parenthesised group contributes its leaf set.
    """
    stack: list[int] = [0]
    for tok in _SPLIT.split(_STRIP.sub("", newick)):
        if not tok:
            continue
        if tok == "(":
            stack.append(0)
        elif tok == ")":
            mask = stack.pop()
            out.append(mask)
            stack[-1] |= mask
        elif tok == ",":
            continue
        else:
            name = tok.strip().rstrip(";").strip()
            if name:
                stack[-1] |= 1 << taxon_index[name]
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses in newick")
    return out


class SupportCounter:
    """Accumulates cluster support across loci, with nested-prefix checkpoints."""

    def __init__(self, n_taxa: int, checkpoints):
        self.n_taxa = n_taxa
        self.checkpoints = sorted(checkpoints)
        self.n_bins = len(self.checkpoints)
        self.full_mask = (1 << n_taxa) - 1
        self.counts: dict[int, np.ndarray] = {}
        self.n_raw = np.zeros(self.n_bins, dtype=np.int64)
        self._locus = 0
        #: Loci whose internal-node count was not ``n_taxa - 1``, i.e. not rooted binary.  The
        #: MDC weight in :mod:`opc.solve.instance` reads this counter as a *node* count, and the
        #: ``set(masks)`` dedup below would silently undercount a unifurcation, so a non-binary
        #: locus has to be visible rather than merely tolerated.
        self.n_nonbinary_loci = 0

    def add_locus(self, masks) -> None:
        which = int(np.searchsorted(self.checkpoints, self._locus, side="right"))
        self._locus += 1
        if len(masks) != self.n_taxa - 1:
            self.n_nonbinary_loci += 1
        if which >= self.n_bins:
            return
        for mask in set(masks):
            if mask == self.full_mask or mask & (mask - 1) == 0:
                continue  # full taxon set, singleton, or empty
            vec = self.counts.get(mask)
            if vec is None:
                vec = self.counts[mask] = np.zeros(self.n_bins, dtype=np.int32)
                self.n_raw[which] += 1
            vec[which] += 1

    def finalize(self, min_support: int = 2):
        """Return ``(masks, cumulative_support, n_raw)`` keeping only clusters worth storing.

        ``min_support`` prunes against the *largest* checkpoint.  With the default of 2 this is
        lossless for every ``(m, tau >= 2)`` cell, since support is non-decreasing in ``m``.
        """
        n_raw = np.cumsum(self.n_raw)
        if not self.counts:
            return (
                np.empty(0, dtype=object),
                np.zeros((0, self.n_bins), dtype=np.int32),
                n_raw,
            )
        masks = np.fromiter(self.counts.keys(), dtype=object, count=len(self.counts))
        sup = np.cumsum(np.stack(list(self.counts.values())), axis=1, dtype=np.int32)
        keep = sup[:, -1] >= min_support
        return masks[keep], sup[keep], n_raw


def count_newick_file(path, taxon_index: dict[str, int], checkpoints, *,
                      drop_partial_root: bool = False) -> SupportCounter:
    """Stream a (optionally gzipped) file of one-tree-per-line Newick into a counter.

    ``drop_partial_root`` (local addition, 2026-10-21, for ``taxmiss_eval``): also drop each tree's
    own leaf set when it is not the full set.  ``add_locus`` drops only the full set, which on a
    full-taxon tree is its root cluster; on a tree missing taxa the root cluster is the tree's taxon
    sampling, not evidence of a clade, and would survive.  This is ``infer_gnn_cpu.count_clusters``'
    rule, and it keeps ``instances._tree_stats``' support == n_obs check exact
    (``blen.GeneTree.internal_nodes`` never scores a root).  Off by default; on a full-taxon file it
    changes nothing.
    """
    path = Path(path)
    counter = SupportCounter(len(taxon_index), checkpoints)
    opener = gzip.open if path.suffix == ".gz" else open
    buf: list[int] = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            buf.clear()
            masks = clusters_from_newick(line, taxon_index, buf)
            if drop_partial_root:
                root = 0
                for x in masks:
                    root |= x
                if root != counter.full_mask:
                    masks = [x for x in masks if x != root]
            counter.add_locus(masks)
    return counter


def _masks_to_bytes(masks, n_taxa: int) -> list[bytes]:
    width = ((n_taxa + 63) // 64) * 8
    return [int(m).to_bytes(width, "little") for m in masks]


def write_cluster_table(path, masks, support, checkpoints, n_taxa: int, n_raw, **meta) -> None:
    """Write the cluster/support table.

    ``pa.binary()`` for the bitset column, not a stringified bigint: past 64 taxa a mask does not
    fit an integer column, and the byte form is the same little-endian layout ``opc.bitset`` uses.
    """
    from opc import bitset as _bs

    width = ((n_taxa + 63) // 64) * 8
    cols = {
        "cluster": pa.array(_masks_to_bytes(masks, n_taxa), type=pa.binary(width)),
        "size": pa.array([int(m).bit_count() for m in masks], type=pa.int32()),
    }
    for i, cp in enumerate(checkpoints):
        cols[f"support_m{cp}"] = pa.array(support[:, i], type=pa.int32())
    if len(masks):
        words = np.stack([np.frombuffer(b, dtype=np.uint64) for b in cols["cluster"].to_pylist()])
        cols["cluster_id"] = pa.array(_bs.fnv1a_many(words), type=pa.uint64())
    else:
        cols["cluster_id"] = pa.array([], type=pa.uint64())

    schema_meta = {
        b"n_taxa": str(n_taxa).encode(),
        b"checkpoints": ",".join(str(c) for c in checkpoints).encode(),
        b"n_clusters_raw": ",".join(str(int(x)) for x in n_raw).encode(),
        **{str(k).encode(): str(v).encode() for k, v in meta.items()},
    }
    table = pa.table(cols).replace_schema_metadata(schema_meta)
    pq.write_table(table, str(path), compression="zstd")


def read_cluster_table(path):
    """Inverse of :func:`write_cluster_table`.  Returns ``(words, support, meta)``."""
    table = pq.read_table(str(path))
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    n_taxa = int(meta["n_taxa"])
    checkpoints = [int(c) for c in meta["checkpoints"].split(",")]
    raw = table.column("cluster").to_pylist()
    n_words = (n_taxa + 63) // 64
    if raw:
        words = np.stack([np.frombuffer(b, dtype=np.uint64) for b in raw])
    else:
        words = np.zeros((0, n_words), dtype=np.uint64)
    support = np.stack(
        [table.column(f"support_m{cp}").to_numpy() for cp in checkpoints], axis=1
    ) if raw else np.zeros((0, len(checkpoints)), dtype=np.int32)
    meta["checkpoints"] = checkpoints
    meta["n_taxa"] = n_taxa
    meta["n_clusters_raw"] = [int(x) for x in meta["n_clusters_raw"].split(",")]
    return words, support, meta
