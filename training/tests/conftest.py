"""Shared fixtures and hypothesis strategies for the opc test suite."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from opc import bitset

MAX_TAXA = 8
MAX_CLUSTERS = 12


@st.composite
def cluster_systems(draw, max_taxa: int = MAX_TAXA, max_clusters: int = MAX_CLUSTERS):
    """A random small clustering system: ``(n_taxa, [frozenset, ...])``.

    Members are non-trivial subsets (size 2 .. n-1), deduplicated, matching the spec's ground-set
    convention of excluding singletons and the full taxon set.
    """
    n = draw(st.integers(min_value=4, max_value=max_taxa))
    universe = list(range(n))
    subsets = draw(
        st.lists(
            st.lists(st.sampled_from(universe), min_size=2, max_size=n - 1, unique=True),
            min_size=1,
            max_size=max_clusters,
        )
    )
    seen, out = set(), []
    for s in subsets:
        f = frozenset(s)
        if 2 <= len(f) <= n - 1 and f not in seen:
            seen.add(f)
            out.append(f)
    return n, out


def pack(system, n_taxa: int) -> np.ndarray:
    """``[frozenset, ...]`` -> ``uint64[N, W]`` in the given order."""
    return bitset.pack_rows([bitset.from_indices(sorted(c), n_taxa) for c in system], n_taxa)
