"""I4: one cell's tau = 0 ground set, decomposed, with its feasibility violations.

``exp-plan-n50-mpl-v0.1`` §2 I4: "Distinct clusters at tau = 0; overlap graph; connected
components; (L) violation pairs and (N3O) violation triples restricted to within-component.  Reuse
Phase 0 percolation instrumentation."

Deliberately thin.  Every kernel it needs already exists and is already cross-validated against a
slow frozenset reference in :mod:`opc.clusters.systems`: :func:`opc.solve.overlap.decompose` builds
the overlap graph and splits it, :func:`opc.solve.overlap.monochromatic_triangles` enumerates the
(N3O) rows, and :mod:`opc.clusters.relations` counts (L) violations exactly.  What this module adds
is the *reading* of them the new study needs -- at tau = 0 rather than tau*, and with the
constraint counts E4 has to budget against reported up front.

Two things worth stating rather than rediscovering.

**Violations are within-component by construction.**  Clusters in different components of the
overlap graph do not overlap, so they are automatically compatible: every (L) pair and every (N3O)
triple has all its members in one component.  That is what makes the decomposition a
constraint-count reduction and not only a solve-time one -- writing all ``O(|C|^2)`` pairs at
|C| = 16,567 would be 1.4e8 rows, which is not viable, while the per-component sum is far smaller.

**Only monochromatic triangles are needed** for (N3O).  Under (L) every triangle of a feasible
system has all three pairwise intersections equal, and a triangle whose intersections differ is
already cut at integrality by the assignment rows.  :mod:`opc.solve.overlap`'s docstring measured
this at 10-13% of all triangles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opc.clusters import extract, relations
from opc.solve import overlap

__all__ = ["GroundSet", "build", "from_cluster_table"]


@dataclass
class GroundSet:
    """The candidate clusters of one cell at one ``(m, tau)``, and the structure over them."""

    n_taxa: int
    m: int
    tau: int
    words: np.ndarray  # uint64[K, W]
    support: np.ndarray  # int32[K]
    decomposition: overlap.Decomposition

    @property
    def n_clusters(self) -> int:
        return int(self.words.shape[0])

    @property
    def components(self):
        return self.decomposition.components

    @property
    def kernel_size(self) -> int:
        return int(self.decomposition.kernel.size)

    def triangles(self) -> list[np.ndarray]:
        """The (N3O) rows, per component."""
        return [overlap.monochromatic_triangles(c) for c in self.components]

    def sizes(self) -> dict:
        """What an ILP over this ground set would cost, before it is built.

        The (L) rows are the overlap edges: an overlapping pair whose intersection is not one of
        the two is exactly a pair the selection may not take together unless the intersection is
        also selected, so the edge count *is* the pairwise constraint count.
        """
        tris = self.triangles()
        return {
            "n_clusters": self.n_clusters,
            "kernel": self.kernel_size,
            "n_components": len(self.components),
            "largest_component": max((c.n_vertices for c in self.components), default=0),
            "n_l_pairs": int(sum(c.n_edges for c in self.components)),
            "n_n3o_triples": int(sum(t.shape[0] for t in tris)),
            "n_binaries": int(sum(c.model_size(t.shape[0])["binaries"]
                                  for c, t in zip(self.components, tris, strict=True))),
            "n_rows": int(sum(c.model_size(t.shape[0])["rows"]
                              for c, t in zip(self.components, tris, strict=True))),
            # All O(|C|^2) pairs, for the comparison that justifies decomposing at all.
            "n_pairs_if_undecomposed": self.n_clusters * (self.n_clusters - 1) // 2,
        }

    def l_violations_per_cluster(self) -> np.ndarray:
        """How many (L) violations each cluster is involved in, over the whole ground set."""
        graph = relations.build_overlap_graph(self.words, self.n_taxa)
        return relations.l_violations_per_vertex(graph)


def build(words: np.ndarray, support: np.ndarray, n_taxa: int, *, m: int = 0,
          tau: int = 0, **kwargs) -> GroundSet:
    """Filter to ``support >= max(tau, 1)`` and decompose.

    ``tau = 0`` and ``tau = 1`` mean the same ground set -- "induced by at least one gene tree" --
    and the plan calls it tau = 0, so both are accepted and the label is kept as given.
    """
    keep = np.flatnonzero(np.asarray(support) >= max(int(tau), 1))
    kept = np.ascontiguousarray(np.asarray(words)[keep])
    return GroundSet(
        n_taxa=n_taxa, m=m, tau=tau, words=kept,
        support=np.asarray(support)[keep],
        decomposition=overlap.decompose(kept, n_taxa, **kwargs),
    )


def from_cluster_table(path, *, m: int | None = None, tau: int = 0, **kwargs) -> GroundSet:
    """Load one of ``run_grid.py``'s tau = 0 cluster tables and build the ground set.

    ``m`` selects a checkpoint; the default is the largest.  The support columns are cumulative
    over nested prefixes, so this is a column lookup rather than a re-count.
    """
    words, support, meta = extract.read_cluster_table(path)
    checkpoints = meta["checkpoints"]
    which = len(checkpoints) - 1 if m is None else checkpoints.index(m)
    return build(words, support[:, which], meta["n_taxa"],
                 m=checkpoints[which], tau=tau, **kwargs)
