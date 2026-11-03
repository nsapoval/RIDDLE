"""Concatenating per-locus alignments into the superalignment IQ-TREE ``-S`` consumes.

AliSim is invoked once per locus (see :mod:`opc.simulate.seqsim` for why its batched partition
mode is unusable here), so the per-locus PHYLIP files have to be stitched into one alignment plus
a partition file.  That pairing is what lets a single IQ-TREE process infer a separate tree per
locus, and **the line order of the partition file is the contract that pins the order of the
trees in the resulting ``.treefile``** -- probed in ``scripts/probe_alisim.py`` (P2) and frozen as
a test.

Sequential PHYLIP is the container because AliSim writes it natively and, unlike FASTA, a row is
unambiguously one taxon.  Row order varies between AliSim's files (it follows each tree's leaf
order), so rows are keyed by name and emitted in sorted order.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "read_phylip",
    "write_superalignment",
    "write_partition_raxml",
    "gap_fraction",
]

_HEADER = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def read_phylip(src) -> tuple[list[str], list[str]]:
    """Read a sequential-PHYLIP alignment.  Returns ``(names, seqs)`` in file order."""
    text = src if isinstance(src, str) and "\n" in src else Path(src).read_text()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if not m:
            continue
        ntax, nsites = int(m.group(1)), int(m.group(2))
        names, seqs = [], []
        for row in lines[i + 1:]:
            if not row.strip():
                continue
            parts = row.split(None, 1)
            names.append(parts[0])
            seqs.append(parts[1].replace(" ", "") if len(parts) > 1 else "")
            if len(names) == ntax:
                break
        bad = [nm for nm, s in zip(names, seqs) if len(s) != nsites]
        if len(names) != ntax or bad:
            raise ValueError(
                f"{getattr(src, 'name', src)}: expected {ntax}x{nsites}, got {len(names)} rows"
                f"{', ragged at ' + str(bad[:3]) if bad else ''}")
        return names, seqs
    raise ValueError(f"no PHYLIP header in {getattr(src, 'name', src)}")


def write_superalignment(sources, out_path) -> list[int]:
    """Concatenate per-locus alignments side by side.  Returns the per-locus widths.

    Raises if the taxon sets differ between loci -- with ``--keep-ident`` they must not, and a
    silently shrinking taxon set is precisely the corruption
    :func:`opc.simulate.outgroup.root_and_prune_file` exists to catch later.
    """
    out_path = Path(out_path)
    rows: dict[str, list[str]] = {}
    widths: list[int] = []
    expected: frozenset[str] | None = None

    for src in sources:
        names, seqs = read_phylip(src)
        got = frozenset(names)
        if expected is None:
            expected = got
            rows = {nm: [] for nm in names}
        elif got != expected:
            raise ValueError(
                f"{getattr(src, 'name', src)}: taxon set differs from the first locus "
                f"(missing {sorted(expected - got)[:5]}, extra {sorted(got - expected)[:5]})")
        for nm, s in zip(names, seqs):
            rows[nm].append(s)
        widths.append(len(seqs[0]))

    if expected is None:
        raise ValueError("no alignments to concatenate")
    order = sorted(rows)
    total = sum(widths)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wt") as fh:
        fh.write(f" {len(order)} {total}\n")
        for nm in order:
            fh.write(f"{nm}  {''.join(rows[nm])}\n")
    tmp.rename(out_path)
    return widths


def write_partition_raxml(path, widths, prefix: str = "loc", model: str = "DNA") -> list[str]:
    """``<model>, loc000000 = 1-500`` per line, in locus order.  Returns the partition names.

    ``model`` is load-bearing, not cosmetic: in a RAxML-style partition file the first field is
    the substitution model for that partition and it **overrides ``-m`` on the command line**.
    Leaving the conventional ``DNA`` here makes IQ-TREE infer under GTR no matter what ``-m``
    says -- silently, with exit code 0.  Callers pass the model they actually want.
    """
    path = Path(path)
    names, body, start = [], [], 1
    for i, w in enumerate(widths):
        nm = f"{prefix}{i:06d}"
        names.append(nm)
        body.append(f"{model}, {nm} = {start}-{start + w - 1}")
        start += w
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(body) + "\n")
    tmp.rename(path)
    return names


def has_empty_sequence(path) -> bool:
    """True if any row has no observed site at all.

    IQ-TREE drops such rows and does not put them back, so the estimated tree comes back a taxon
    short.  ``--keep-ident`` does not help: it protects *identical* sequences, not empty ones.
    """
    _, seqs = read_phylip(path)
    return any(set(s) <= set("-?nN") for s in seqs)


def gap_counts(path) -> tuple[int, int]:
    """``(gap characters, total characters)``.

    Counts rather than a ratio, so pooling across batches and chunks stays exact instead of
    becoming an average of averages.
    """
    _, seqs = read_phylip(path)
    return sum(s.count("-") for s in seqs), sum(len(s) for s in seqs)


def gap_fraction(path) -> float:
    gaps, total = gap_counts(path)
    return gaps / total if total else 0.0
