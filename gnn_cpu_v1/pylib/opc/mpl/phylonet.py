"""Driving PhyloNet: the independent implementation I1 is validated against.

**Validation only.**  ``InferNetwork_MPL`` -- the incumbent *search* -- is not wrapped here, and
A35 records why: benchmarking it was dropped, so PhyloNet's remaining job is to be a second
opinion on ``P(t | Psi, gamma)`` and nothing else.  The wrapper for it is in git history at
commit b4c4b5c if a later sprint wants it back.

    /home/ns58/software/PhyloNet.jar   (2022 build; this box has Java 11 only -- CLAUDE.md)

PhyloNet has no CLI flags.  It reads a NEXUS file with three blocks -- ``NETWORKS``, ``TREES`` and
``PHYLONET`` -- and executes the commands in the third against the objects named in the first two.
Several commands may share one file, which matters: the JVM takes about a second to start and the
validation asks for thousands of probabilities, so everything here batches.

What this build actually does, measured by ``scripts/probe_phylonet.py``:

* ``CalGTProb net (g1, g2, ...)`` prints ``Total log probability:`` -- the **full** gene-tree
  likelihood, summed over the listed gene trees.
* **``-pl`` is a thread count, not a pseudo-likelihood switch.**  The obvious reading is wrong:
  ``-pl`` without a value is a parse error, and ``-pl 1`` returns the same number as no flag at
  all.  ``CalGTProb`` computes the full likelihood; the pseudo-likelihood lives inside
  ``InferNetwork_MPL``, which this module deliberately does not wrap (A35).
* ``-o`` re-optimises branch lengths and inheritance probabilities before scoring.
* **A gene tree may be on a subset of the network's taxa**, and the number returned is then the
  correct marginal for that subset.  That is what makes triple-by-triple validation possible at
  n = 50 without building trinets by hand: ``P(t | Psi, gamma)`` for one triple is exactly
  ``exp(CalGTProb(Psi, [t]))``.

The last point is the load-bearing one for I1, so :func:`triple_probabilities` is the entry point
the validation uses.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

__all__ = [
    "PHYLONET",
    "JAVA",
    "available",
    "nexus",
    "run",
    "cal_gt_prob",
    "triple_probabilities",
    "CMPNETS_METHODS",
    "strip_gamma",
    "cmpnets_many",
]

PHYLONET = Path(os.environ.get("OPC_PHYLONET", "/home/ns58/software/PhyloNet.jar"))
JAVA = Path(os.environ.get("OPC_JAVA", "/usr/bin/java"))

# The exponent's sign is not optional cosmetics: PhyloNet prints ``-4.34E-5`` for a rare triple,
# and a character class that omits the '-' truncates the number to ``-4.34E`` and raises.
_NUMBER = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
_LOGP = re.compile(r"Total log probability:\s*" + _NUMBER)
_ERROR = re.compile(r"^Error at \[|^Exception|Error: ", re.MULTILINE)

#: ``CMPNETS -m`` methods this module knows how to parse, and the line each one prints.
#:
#: * ``luay`` -- **topology only.**  Two networks with identical topology and five-fold different
#:   branch lengths measure **0.0**, which is what makes it the metric to read beside cluster F1:
#:   it is a statement about the *network* and nothing else.
#: * ``normwapd`` -- topology **and** branch lengths.  The same pair measures **2.546**.  Note the
#:   name oversells it: it is not bounded by 1.
CMPNETS_METHODS = {
    "luay": re.compile(r"Luay's distance between two networks:\s*" + _NUMBER),
    "normwapd": re.compile(r"NormWAPD dissimilarity between two networks:\s*" + _NUMBER),
}

#: ``#H1:1::0.5`` -> ``#H1:1``.  **CmpNets cannot parse an inheritance probability at all.**
#: Measured: both ``#H1:1::0.7`` and ``#H1:1:0:0.7`` raise ``ExNewickException: Unexpected
#: character`` from ``parseNodeDistance``; only ``#H1:<length>`` reads.  Neither method uses gamma
#: -- ``luay`` is topological and ``normwapd`` weights by branch length -- so nothing is lost by
#: dropping it, **but two networks differing only in gamma are at distance 0 under both**, and any
#: claim made from these numbers has to be read with that in mind.
_GAMMA = re.compile(r"::[0-9.eE+-]+")


def available() -> bool:
    return PHYLONET.exists() and (JAVA.exists() or shutil.which(str(JAVA)) is not None)


def nexus(networks: dict[str, str], trees: dict[str, str], commands) -> str:
    """A NEXUS file.  Newick strings may carry a trailing ``;`` or not."""
    def clean(s: str) -> str:
        return s.strip().rstrip(";") + ";"

    out = ["#NEXUS"]
    if networks:
        out.append("BEGIN NETWORKS;")
        out += [f"Network {k} = {clean(v)}" for k, v in networks.items()]
        out.append("END;")
    if trees:
        out.append("BEGIN TREES;")
        out += [f"Tree {k} = {clean(v)}" for k, v in trees.items()]
        out.append("END;")
    out.append("BEGIN PHYLONET;")
    out += [c.strip().rstrip(";") + ";" for c in commands]
    out.append("END;")
    return "\n".join(out) + "\n"


def run(text: str, *, workdir=None, timeout: float = 3600.0, heap: str = "4g") -> str:
    """Execute one NEXUS file and return stdout.  Raises on a PhyloNet-reported error."""
    if not available():
        raise RuntimeError(f"PhyloNet not found at {PHYLONET} (set OPC_PHYLONET)")
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="phylonet-"))
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "run.nex"
    path.write_text(text)
    proc = subprocess.run(
        [str(JAVA), f"-Xmx{heap}", "-jar", str(PHYLONET), str(path)],
        capture_output=True, text=True, timeout=timeout, cwd=str(tmp),
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or _ERROR.search(out):
        raise RuntimeError(f"PhyloNet failed on {path}:\n{out[:4000]}")
    return out


def cal_gt_prob(
    enewick: str,
    gene_trees,
    *,
    optimize: bool = False,
    workdir=None,
    timeout: float = 3600.0,
) -> list[float]:
    """One ``CalGTProb`` per entry of ``gene_trees``; returns the log probabilities in order.

    Each entry is either one Newick string or a sequence of them, in which case the command scores
    that whole set at once -- PhyloNet sums the log probabilities, which is what a locus set means.
    """
    trees: dict[str, str] = {}
    commands: list[str] = []
    for i, item in enumerate(gene_trees):
        names = []
        for j, nwk in enumerate([item] if isinstance(item, str) else list(item)):
            name = f"g{i}_{j}"
            trees[name] = nwk
            names.append(name)
        flag = " -o" if optimize else ""
        commands.append(f"CalGTProb net ({','.join(names)}){flag}")
    out = run(nexus({"net": enewick}, trees, commands), workdir=workdir, timeout=timeout)
    vals = [float(m.group(1)) for m in _LOGP.finditer(out)]
    if len(vals) != len(commands):
        raise RuntimeError(
            f"expected {len(commands)} probabilities, PhyloNet printed {len(vals)}:\n{out[:2000]}"
        )
    return vals


def triple_probabilities(
    enewick: str, labels, taxa, *, workdir=None, timeout: float = 3600.0
) -> np.ndarray:
    """``(P(ab|c), P(ac|b), P(bc|a))`` for one triple, straight out of PhyloNet.

    Relies on the measured fact that a gene tree on a subset of the network's taxa is scored as the
    correct marginal.  The three values are checked to sum to 1, which is a real check: if a future
    build stopped marginalising they would not.
    """
    return triple_probabilities_many(enewick, labels, [taxa], workdir=workdir, timeout=timeout)[0]


def triple_probabilities_many(
    enewick: str, labels, triples, *, workdir=None, timeout: float = 3600.0, tol: float = 1e-6
) -> np.ndarray:
    """:func:`triple_probabilities` for many triples in **one** JVM invocation."""
    gene_trees = []
    for a, b, c in triples:
        la, lb, lc = labels[a], labels[b], labels[c]
        gene_trees += [f"(({la},{lb}),{lc})", f"(({la},{lc}),{lb})", f"(({lb},{lc}),{la})"]
    logs = np.asarray(cal_gt_prob(enewick, gene_trees, workdir=workdir, timeout=timeout))
    p = np.exp(logs).reshape(-1, 3)
    bad = np.abs(p.sum(axis=1) - 1.0) > tol
    if bad.any():
        raise RuntimeError(
            f"{bad.sum()} triples' probabilities do not sum to 1 (worst "
            f"{np.abs(p.sum(1) - 1).max():.2e}); PhyloNet may have stopped marginalising over "
            "the taxa absent from the gene tree"
        )
    return p


def strip_gamma(enewick: str) -> str:
    """Remove ``::<gamma>`` so CmpNets can read the string.  See :data:`_GAMMA`."""
    return _GAMMA.sub("", enewick)


def cmpnets_many(pairs, methods=("luay", "normwapd"), *, workdir=None,
                 timeout: float = 3600.0, heap: str = "4g") -> list[dict]:
    """``[(enewick_a, enewick_b), ...]`` -> one ``{method: distance}`` per pair.

    **One JVM for the whole list.**  PhyloNet costs ~1-2 s to start and CmpNets itself is
    microseconds, so a call per pair would be almost entirely JVM launch; every pair and every
    method goes into a single NEXUS whose ``PHYLONET`` block holds ``len(pairs) x len(methods)``
    ``CMPNETS`` commands, and the outputs are read back in order.  This is the shape
    :func:`triple_probabilities_many` already uses, for the same reason.

    **A35 still binds: this is validation, never an inner loop.**  It is called from
    ``scripts/w9_swap_eval.py``, which is an evaluation driver, and must not appear in a reward or
    a training step.

    Gammas are stripped (:func:`strip_gamma`) because CmpNets cannot parse them.
    """
    bad = [m for m in methods if m not in CMPNETS_METHODS]
    if bad:
        raise ValueError(f"unknown CMPNETS method(s) {bad}; known: {sorted(CMPNETS_METHODS)}")
    pairs = list(pairs)
    if not pairs:
        return []
    nets, cmds, order = {}, [], []
    for i, (a, b) in enumerate(pairs):
        nets[f"a{i}"] = strip_gamma(a)
        nets[f"b{i}"] = strip_gamma(b)
        for m in methods:
            cmds.append(f"CMPNETS a{i} b{i} -m {m}")
            order.append((i, m))
    try:
        out = run(nexus(nets, {}, cmds), workdir=workdir, timeout=timeout, heap=heap)
    except RuntimeError:
        # **One unparseable network must not lose the whole batch.**  Batching is for the JVM's
        # sake, and PhyloNet refuses the *file* rather than the offending command -- so one bad
        # eNewick silently cost every comparison beside it, greedy's included (A80).  Fall
        # back to one call per pair and return ``nan`` only for the pairs that actually fail.
        if len(pairs) == 1:
            raise
        rows: list[dict] = []
        n_bad = 0
        first: str | None = None
        for pair in pairs:
            try:
                rows += cmpnets_many([pair], methods, workdir=None, timeout=timeout, heap=heap)
            except RuntimeError as exc:
                # **Say which, and why, once.**  A silent NaN per pair is how 81 of 300 rows came
                # back empty for exactly the reticulate cells with nothing in the log to chase
                # (A83): the batch-level guard prints, the per-pair one did not, so a systematic
                # failure was indistinguishable from PhyloNet being flaky.
                n_bad += 1
                if first is None:
                    first = f"{type(exc).__name__}: {str(exc)[:300]}"
                rows.append(dict.fromkeys(methods, float("nan")))
        if n_bad:
            print(f"      [cmpnets] {'/'.join(methods)}: batch refused, {n_bad} of {len(pairs)} "
                  f"pairs also failed individually.  First: {first}", flush=True)
        return rows

    # Parsed by consuming each method's matches **in command order**, because every CMPNETS line
    # prints the same sentence and there is nothing in the output tying a number to its pair.
    got: dict[str, list[str]] = {m: CMPNETS_METHODS[m].findall(out) for m in methods}
    for m, vals in got.items():
        want = sum(1 for _i, mm in order if mm == m)
        if len(vals) != want:
            raise RuntimeError(
                f"CMPNETS -m {m}: parsed {len(vals)} distances for {want} commands. The output "
                f"format has changed or a command failed silently:\n{out[:2000]}")
    rows: list[dict] = [{} for _ in pairs]
    seen: dict[str, int] = dict.fromkeys(methods, 0)
    for i, m in order:
        rows[i][m] = float(got[m][seen[m]])
        seen[m] += 1
    return rows
