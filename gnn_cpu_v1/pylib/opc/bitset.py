"""Fixed-width bitset representation of clusters (subsets of the taxon set X).

A cluster over ``n`` taxa is a row of ``W = ceil(n/64)`` ``uint64`` words, little-endian
within and across words: taxon ``t`` is bit ``t % 64`` of word ``t // 64``.

Two derived representations are used downstream:

``words``  ``uint64[N, W]``  -- canonical storage, dedup key, and the reference overlap kernel.
``dense``  ``float32[N, n]`` -- 0/1 membership, used by the GEMM overlap kernel.  Intersection
                               counts come out of a float32 GEMM exactly as long as ``n < 2**24``.

Per SPEC-AMENDMENTS A6, dict keys are the raw ``bytes`` of the word array (exact, no collision
risk).  FNV-1a survives only as :func:`fnv1a` -- a *stable content id* for output rows and
cross-language joins, never for dedup.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "n_words",
    "empty",
    "from_indices",
    "to_indices",
    "pack_rows",
    "to_dense",
    "popcounts",
    "key",
    "from_key",
    "fnv1a",
    "fnv1a_many",
]

_FNV64_OFFSET = np.uint64(0xCBF29CE484222325)
_FNV64_PRIME = np.uint64(0x100000001B3)


def n_words(n_taxa: int) -> int:
    """Number of ``uint64`` words needed to hold a subset of ``n_taxa`` taxa."""
    if n_taxa <= 0:
        raise ValueError(f"n_taxa must be positive, got {n_taxa}")
    return (n_taxa + 63) // 64


def empty(n_rows: int, n_taxa: int) -> np.ndarray:
    """All-zero ``uint64[n_rows, W]`` bitset block."""
    return np.zeros((n_rows, n_words(n_taxa)), dtype=np.uint64)


def from_indices(indices, n_taxa: int) -> np.ndarray:
    """Build a single bitset (``uint64[W]``) from an iterable of taxon indices."""
    out = np.zeros(n_words(n_taxa), dtype=np.uint64)
    for t in indices:
        t = int(t)
        if not 0 <= t < n_taxa:
            raise ValueError(f"taxon index {t} out of range for n_taxa={n_taxa}")
        out[t >> 6] |= np.uint64(1) << np.uint64(t & 63)
    return out


def to_indices(bits: np.ndarray, n_taxa: int) -> list[int]:
    """Sorted taxon indices present in a single bitset row."""
    out = []
    for w in range(n_words(n_taxa)):
        word = int(bits[w])
        while word:
            low = word & -word
            out.append((w << 6) + low.bit_length() - 1)
            word ^= low
    return out


def pack_rows(rows, n_taxa: int) -> np.ndarray:
    """Stack an iterable of ``uint64[W]`` rows (or ``bytes`` keys) into ``uint64[N, W]``."""
    w = n_words(n_taxa)
    rows = list(rows)
    if not rows:
        return np.zeros((0, w), dtype=np.uint64)
    out = np.empty((len(rows), w), dtype=np.uint64)
    for i, r in enumerate(rows):
        out[i] = np.frombuffer(r, dtype=np.uint64) if isinstance(r, (bytes, bytearray)) else r
    return out


def to_dense(words: np.ndarray, n_taxa: int) -> np.ndarray:
    """``uint64[N, W]`` -> ``float32[N, n_taxa]`` 0/1 membership matrix.

    ``np.unpackbits`` works on ``uint8`` with a bit order; viewing the ``uint64`` words as
    little-endian bytes and unpacking ``bitorder="little"`` reproduces the taxon order exactly.
    """
    if words.dtype != np.uint64:
        raise TypeError(f"expected uint64 words, got {words.dtype}")
    n_rows = words.shape[0]
    as_bytes = np.ascontiguousarray(words).view(np.uint8).reshape(n_rows, -1)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return bits[:, :n_taxa].astype(np.float32)


def popcounts(words: np.ndarray) -> np.ndarray:
    """Cardinality of each row, as ``int32``."""
    return np.bitwise_count(words).sum(axis=1, dtype=np.int64).astype(np.int32)


def key(bits: np.ndarray) -> bytes:
    """Canonical dict key for one bitset row: the raw little-endian word bytes."""
    return np.ascontiguousarray(bits, dtype=np.uint64).tobytes()


def from_key(k: bytes) -> np.ndarray:
    """Inverse of :func:`key`."""
    return np.frombuffer(k, dtype=np.uint64)


def fnv1a(bits: np.ndarray) -> int:
    """64-bit FNV-1a over the word array -- a *stable content id*, not a dedup key.

    Hashes the little-endian byte stream so the value is identical in Python, Julia and R.
    """
    h = _FNV64_OFFSET
    with np.errstate(over="ignore"):
        for b in np.ascontiguousarray(bits, dtype=np.uint64).view(np.uint8):
            h = (h ^ np.uint64(b)) * _FNV64_PRIME
    return int(h)


def fnv1a_many(words: np.ndarray) -> np.ndarray:
    """Vectorised :func:`fnv1a` over ``uint64[N, W]`` -> ``uint64[N]``."""
    n_rows = words.shape[0]
    as_bytes = np.ascontiguousarray(words).view(np.uint8).reshape(n_rows, -1)
    h = np.full(n_rows, _FNV64_OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for col in range(as_bytes.shape[1]):
            h = (h ^ as_bytes[:, col].astype(np.uint64)) * _FNV64_PRIME
    return h
