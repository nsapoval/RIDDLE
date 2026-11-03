"""rho(t, G): the rooted-triple frequency tensor (I1, first half).

``exp-plan-n50-mpl-v0.1`` §2 I1.  The Yu-Nakhleh rooted-triple pseudo-likelihood is

    log L(Psi, gamma | G) = sum_{tau} sum_{t in tau} rho(t, G) * log P(t | Psi, gamma)

where ``tau`` runs over the ``C(n,3)`` taxon triples and ``t`` over the three rooted topologies of
each.  ``rho(t, G)`` is a property of the *data* alone, so it is computed once per locus set and
then reused across every network the optimiser or the search proposes.  That factorisation is
what makes the criterion cheap: the inner loop touches ``P``, never ``G``.

Storage
-------

A dense ``int32[n, n, n]`` tensor, ``count[a, b, c]`` = number of loci inducing ``ab|c``, filled
symmetrically in the first two axes so a lookup needs no canonicalisation.  At n = 50 that is
500 KB -- smaller than the sparse alternative and directly indexable, which matters because the
likelihood reads all ``3 * C(50,3)`` entries on every evaluation.

Counting
--------

For a rooted tree, ``ab|c`` is induced exactly when some node has ``a`` and ``b`` under *different*
children and ``c`` outside its cluster.  So one pass over internal nodes suffices: for a node with
children leaf sets ``A``, ``B`` and complement ``C``, every ``(a, b, c)`` in ``A x B x C`` gets
``ab|c``.  Summed over the nodes of a rooted binary tree this is exactly ``C(n,3)`` increments --
each triple counted once, by construction -- so the pass is optimal to within the constant.

**Multifurcations are silently under-counted, and that is the correct behaviour.**  A triple whose
three taxa sit in three different children of one node is unresolved in that locus and contributes
to none of the three topologies, so its three counts sum to less than the locus count.  Callers
that need "all loci resolved this triple" must check :attr:`TripleCounts.n_loci` against the row
sum rather than assume it.  :class:`opc.clusters.extract.SupportCounter` already reports
``n_nonbinary_loci`` for the same reason.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "TripleCounts",
    "child_leaf_sets",
    "count_tree",
    "count_newick_file",
    "triple_index",
]

# Same dialect as opc.clusters.extract: strip ``:length[:...]`` and any label after a ``)``.
_STRIP = re.compile(r":[^,);]*|(?<=\))[^,);]*")
_SPLIT = re.compile(r"([(),])")


def child_leaf_sets(newick: str, taxon_index: dict[str, int]) -> list[list[list[int]]]:
    """Per internal node, the list of its children's leaf-index lists.

    Deliberately *not* reusing :func:`opc.clusters.extract.clusters_from_newick`: that function
    unions a node's children into one bitmask as it pops, which is exactly the information a
    triple needs and the only information it discards.
    """
    stack: list[list[list[int]]] = [[]]
    out: list[list[list[int]]] = []
    for tok in _SPLIT.split(_STRIP.sub("", newick)):
        if not tok:
            continue
        if tok == "(":
            stack.append([])
        elif tok == ")":
            kids = stack.pop()
            out.append(kids)
            merged = [t for k in kids for t in k]
            stack[-1].append(merged)
        elif tok == ",":
            continue
        else:
            name = tok.strip().rstrip(";").strip()
            if name:
                stack[-1].append([taxon_index[name]])
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses in newick")
    return out


@dataclass
class TripleCounts:
    """``count[a, b, c]`` = loci inducing ``ab|c``, symmetric in ``a`` and ``b``."""

    n_taxa: int
    count: np.ndarray  # int32[n, n, n]
    n_loci: int = 0
    n_nonbinary_loci: int = 0

    @classmethod
    def empty(cls, n_taxa: int) -> TripleCounts:
        return cls(n_taxa, np.zeros((n_taxa,) * 3, dtype=np.int32))

    def add_tree(self, children_per_node) -> None:
        universe = np.arange(self.n_taxa)
        n_internal = 0
        for kids in children_per_node:
            n_internal += 1
            if len(kids) < 2:
                continue
            below = np.zeros(self.n_taxa, dtype=bool)
            for k in kids:
                below[k] = True
            outside = universe[~below]
            if outside.size == 0:
                continue  # the root: no third taxon to be an outgroup
            for i in range(len(kids)):
                for j in range(i + 1, len(kids)):
                    a, b = np.asarray(kids[i]), np.asarray(kids[j])
                    self.count[np.ix_(a, b, outside)] += 1
                    self.count[np.ix_(b, a, outside)] += 1
        self.n_loci += 1
        self.n_nonbinary_loci += n_internal != self.n_taxa - 1

    def resolved(self) -> np.ndarray:
        """``r[a, b, c] = count[a,b,c] + count[a,c,b] + count[b,c,a]`` for the triple's row sum."""
        return self.count + self.count.transpose(0, 2, 1) + self.count.transpose(2, 1, 0)

    def frequencies(self) -> np.ndarray:
        """``count / n_loci``.  Yu-Nakhleh's rho, as a proportion of *loci*, not of resolutions."""
        if not self.n_loci:
            raise ValueError("no loci counted")
        return self.count / float(self.n_loci)

    def as_rows(self, order: np.ndarray | None = None):
        """``(idx, obs)`` -- ``idx[i]`` an ascending taxon triple, ``obs[i]`` its three counts.

        The three counts are ordered ``(ab|c, ac|b, bc|a)`` with ``a < b < c``, which is the
        convention every downstream module uses.  This is the form the likelihood, the profiled
        weight table and the PhyloNet comparison all consume, so it exists once here.
        """
        idx = triple_index(self.n_taxa) if order is None else order
        a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
        obs = np.stack(
            [self.count[a, b, c], self.count[a, c, b], self.count[b, c, a]], axis=1
        ).astype(np.int64)
        return idx, obs


def triple_index(n_taxa: int) -> np.ndarray:
    """All ``C(n,3)`` ascending taxon triples, in lexicographic order.  ``int32[C(n,3), 3]``."""
    from itertools import combinations

    return np.fromiter(
        (t for c in combinations(range(n_taxa), 3) for t in c), dtype=np.int32
    ).reshape(-1, 3)


def count_tree(newick: str, taxon_index: dict[str, int], acc: TripleCounts) -> TripleCounts:
    acc.add_tree(child_leaf_sets(newick, taxon_index))
    return acc


def count_newick_file(path, taxon_index: dict[str, int], max_loci: int = 0) -> TripleCounts:
    """Stream a (optionally gzipped) one-tree-per-line Newick file into a counter.

    ``max_loci`` reads only the first ``m`` trees.  The locus files are written in simulation
    order and the ``m`` checkpoints are nested prefixes, so this is the same subsample the cluster
    tables use -- the two views of a cell stay paired locus for locus.
    """
    path = Path(path)
    acc = TripleCounts.empty(len(taxon_index))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            acc.add_tree(child_leaf_sets(line, taxon_index))
            if max_loci and acc.n_loci >= max_loci:
                break
    return acc
