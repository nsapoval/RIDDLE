#!/usr/bin/env python3
"""The W10 GNN on one file of gene trees: Newick in, eNewick out.

``run_gnn_cpu.py`` is the benchmark.  This is its chain -- ``run_cell``'s calls, in its order -- on
a file nobody has a true network for, so three things the benchmark reads off the truth come from
somewhere else:

* ``k``, the reticulation budget.  It is a global feature *and* the decode's cap, so the model
  cannot run without one, and nothing in the library estimates it.  ``--k`` takes a list: every
  stage up to the node features is shared, and each further ``k`` costs one forward and one decode.
  It is a cap -- the realised network can carry fewer.
* the taxon order: ``sorted(S)``.  It moves stable tie-breaks and nothing else, which the benchmark
  measured at 2 clusters on 1 of 63 (cell, model) pairs.
* ``m``: the trees actually used.  With more than ``--m`` eligible (default 1,000, the ``m`` every
  checkpoint trained at) a seeded sample is taken; with fewer, all of them are and no RNG is
  involved, so a camus file goes through exactly as the benchmark ran it.

``tau``, the support a clade needs to be a candidate, is the benchmark's 2 unless asked otherwise.
``--tau N`` is a count of trees; ``--tau-frac F`` is ``ceil(F * m)`` over the ``m`` trees actually
used, a threshold that scales with the sample.  Every checkpoint trained at 2 of 1,000 trees, so
F = 0.002 is training's, and anything else moves the candidate set and every feature normalised by
it (``I.node_features``, ``I.global_features``) away from what the model saw.

Every checkpoint in ``checkpoints/`` runs.  A ``conv: pair`` model (``_l2mix_pair``) is handed the
candidates' words, ``n_taxa`` and ``m`` its encoder reads, exactly as ``run_gnn_cpu.forward`` hands
them; every other conv gets the call it always got.  ``--decoder level1`` (the default) is the formal
decoder every landed row used, and ``--decoder relaxed --max-level L`` is ``opc.rl.relaxed``'s --
the same scores, rank rule and budget, with legality stated on the selection's Hasse diagram.

What the trees have to be
-------------------------

Rooted, with branch lengths (W13 reads them), and over one taxon set S.  Real input is often none of
those:

* Rooting is one of four schemes, the two new ones in ``rooting.py`` beside this file:

  - ``--outgroup A,B,...`` roots each tree on the first listed taxon it carries and prunes every one
    listed -- ``OG.reroot_on_outgroup(keep_lengths=True)``, which is how the benchmark rooted its
    IQ-TREE and FastTree trees.  A tree carrying none of them is skipped.
  - ``--midpoint`` roots each tree at the middle of its longest leaf-to-leaf path.
  - ``--species-tree FILE`` roots each tree on the split the rooted species tree, restricted to that
    tree's taxa, has at its root: one taxon's pendant edge when a single taxon diverges first, a
    clade's edge otherwise.  Every taxon is kept.  A tree that does not carry the split is rejected,
    not guessed at.
  - With none of them the trees are read as rooted as written, and a non-binary root is refused
    rather than taken for one.

  The two new schemes root on every taxon the tree carries (that the species tree knows) and only
  then prune to S, and they can reject a tree.  A rejected tree is replaced from the eligible trees
  not sampled, in a seeded order, so ``m`` still counts the trees used (see ``prepare``).
* ``--n-taxa N`` sets S to the N most frequent taxa (``--taxa FILE`` names them instead); a tree
  carrying all of S is pruned to it and one lacking any is skipped.  With neither, S is every
  taxon and a tree lacking one is a refusal: ``clusters_from_newick`` is silent on a missing
  taxon, so this is the only place it gets caught.
* ``--partial MIN`` drops "all of S": a tree carrying at least MIN of S is pruned to what it has and
  contributes the clusters it can.  **Its own leaf set is not one of them** -- a 16-taxon tree says
  nothing about whether those 16 are a clade -- so each tree's root cluster is dropped before
  counting (see ``count_clusters``).

Out of distribution, and said here rather than discovered: the checkpoints trained at n = 25-75 on
full-taxon trees; the camus benchmark held cluster F1 ~0.93 to n = 200 and did not go past it;
``--partial`` input was never seen in training; and gene trees split out of one family (DISCO) are
not independent loci.

Usage
-----

    .venv/bin/python infer_gnn_cpu.py --model checkpoints/_mix2L/s0/best.pt \\
        --trees inputs/n50/00/g_500.nwk --outgroup OUT --k 1
    .venv/bin/python infer_gnn_cpu.py --model checkpoints/_mix2L/s0/best.pt --trees genes.newick \\
        --outgroup GCF_037176945.1,GCF_900747795.2 --n-taxa 100 --k 1,2,3 --out nets.tsv
    .venv/bin/python infer_gnn_cpu.py --model checkpoints/_l2mix_pair/s0/best_ha.pt --trees genes.newick \\
        --species-tree species.tre --n-taxa 100 --k 1,2 --decoder relaxed --max-level 2

stdout, or ``--out``, is one line per ``k``: ``k<TAB>eNewick``.  Everything else goes to stderr.
``infer_gnn_cpu.md`` beside this file is the operating guide.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import numbers
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from itertools import islice
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(f"needs Python 3.10 or newer (running {sys.version.split()[0]}); "
                     f"3.12 is what the package was built and tested with")

HERE = Path(__file__).resolve().parent


def _lib_root(path: Path) -> Path:
    """``--opc`` may name the directory holding ``opc/`` or ``opc/`` itself; return the former."""
    path = Path(path).resolve()
    if (path / "opc" / "__init__.py").is_file():
        return path
    if path.name == "opc" and (path / "__init__.py").is_file():
        return path.parent
    raise SystemExit(f"--opc {path}: neither it nor a child 'opc/' is the opc package")


# Threads and the library are both fixed **before numpy, torch or opc is imported** -- threads
# because a BLAS that has sized its pool ignores a later change, the library because whichever
# ``opc`` is first on the path is the one that runs.  ``run_gnn_cpu.py`` does the same for
# ``--threads``, which is also why this does not import it: at import it would put its own
# ``pylib/`` at ``sys.path[0]`` and shadow ``--opc``.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=1)
_pre.add_argument("--opc", type=Path, default=HERE / "pylib")
_PRE = _pre.parse_known_args()[0]
THREADS = max(1, _PRE.threads)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(THREADS)
LIB = _lib_root(_PRE.opc)
sys.path.insert(0, str(LIB))
if str(HERE) not in sys.path:             # ``rooting.py``, when imported from another directory
    sys.path.insert(1, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:                  # already set in this process; harmless
    pass

import opc  # noqa: E402

# Checked before any submodule is imported.  With ``LIB`` at ``sys.path[0]`` the only way another
# copy wins is that this process imported one already, and that copy may not have ``opc.rl`` at all
# -- in which case the next import would fail with a ModuleNotFoundError that hides the real cause.
if not Path(opc.__file__).resolve().parent.is_relative_to(LIB):
    raise SystemExit(f"opc resolved to {Path(opc.__file__).parent}, not under {LIB}: another copy "
                     f"of the library was imported first and is shadowing the one --opc names")

from opc.clusters import extract  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import annotate as AN  # noqa: E402
from opc.rl import instances as I  # noqa: E402
from opc.rl import nodecls as NC  # noqa: E402
from opc.rl import policy as P  # noqa: E402
from opc.rl import prefix as PX  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR  # noqa: E402
from opc.simulate import networks as NW  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402
from opc.solve import candidates  # noqa: E402

import rooting as RT  # noqa: E402

# A sibling file rather than a package, so a stray ``rooting`` imported earlier would win silently --
# the shadowing ``opc`` is checked for above.
if Path(RT.__file__).resolve().parent != HERE:
    raise SystemExit(f"rooting resolved to {RT.__file__}, not {HERE / 'rooting.py'}: another module "
                     f"named rooting was imported first")

#: Every checkpoint trained at tau = 2, on m = 1,000 trees; ``--tau``'s default.
TAU = 2


def resolve_tau(m: int, tau: int | None = None, tau_frac=None) -> int:
    """The support threshold over ``m`` trees: ``tau``, else ``ceil(tau_frac * m)``, else ``TAU``.

    ``tau_frac`` is read as the decimal it is written as, not as a binary float: ``0.07`` of 100
    trees is 7, where ``math.ceil(0.07 * 100)`` is 8.  It never goes below 1, and 1 already keeps
    every cluster any tree has -- ``candidates.build`` reads 0 and 1 as the same ground set, so only
    one spelling is accepted.  Raises ``ValueError`` on both being given or either out of range.
    """
    if tau is not None and tau_frac is not None:
        raise ValueError("tau and tau_frac were both given; pick one")
    if tau_frac is None:
        tau = TAU if tau is None else tau
        if isinstance(tau, bool) or not isinstance(tau, numbers.Integral) or tau < 1:
            raise ValueError(f"tau {tau!r}: expected an integer >= 1")
        return int(tau)
    try:
        frac = Fraction(str(tau_frac))
    except ValueError:
        frac = None
    if frac is None or not 0 < frac <= 1:
        raise ValueError(f"tau_frac {tau_frac!r}: expected a number in (0, 1]")
    return max(1, math.ceil(frac * m))


def _tau_text(tau: int, tau_frac, m: int) -> str:
    """``tau = 2``, or ``tau = 127 (tau-frac 0.002 of 63,430 trees)`` -- the log's spelling."""
    return f"tau = {tau:,}" + ("" if tau_frac is None else f" (tau-frac {tau_frac} of {m:,} trees)")


#: A leaf label is whatever follows ``(`` or ``,`` up to the next delimiter; an internal label
#: follows ``)`` and never matches.  That is enough to take a million-tree file's leaf sets without
#: parsing a tree.  It is checked rather than trusted: :func:`prepare` re-reads every chosen tree's
#: leaves with ``OG.parse_newick`` and refuses on any disagreement.
_LEAF = re.compile(r"[(,]\s*([^\s():,;]+)")


def _lines(path: Path):
    """Every non-blank line, stripped -- the one enumeration both passes index trees by."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _names(mask: int, labels: list[str]) -> set[str]:
    out = set()
    while mask:
        low = mask & -mask
        out.add(labels[low.bit_length() - 1])
        mask ^= low
    return out


def scan(path: Path) -> tuple[list[int], list[str], Counter]:
    """Each tree's leaf set as an int bitmask over ``labels`` (first-seen order), and each taxon's
    tree count.  No tree is parsed."""
    index: dict[str, int] = {}
    freq: Counter = Counter()
    masks = []
    for line in _lines(path):
        names = _LEAF.findall(line)
        freq.update(names)
        m = 0
        for name in names:
            m |= 1 << index.setdefault(name, len(index))
        masks.append(m)
    return masks, list(index), freq


def select(masks, labels, freq, *, outgroups=(), taxa=None, n_taxa=0, partial=0, m=1000,
           seed=0, species=None) -> tuple[list[str], list[int], list[int], dict]:
    """``(S, chosen, reserve, stats)``: the sorted taxon set, the tree indices to use, the eligible
    trees left out in the seeded order a rejected tree is replaced from, and what was skipped.

    Skipping is only ever *on purpose* -- a tree without an outgroup, or lacking what ``--n-taxa``
    / ``--partial`` asked for.  With no restriction asked for, a tree lacking a taxon is refused.
    ``species`` (a :class:`rooting.SpeciesTree`) confines S to its taxa, the way outgroups are kept
    out of it: a taxon the species tree lacks can be pruned, never rooted on or counted.
    """
    bit = {lab: 1 << i for i, lab in enumerate(labels)}
    for og in outgroups:
        if og not in bit:
            raise SystemExit(f"--outgroup {og}: no tree carries it")
    og_mask = sum(bit[og] for og in set(outgroups))
    # A taxon on one side only is often one taxon under two names (an accession version, say), and
    # the refusal below names only the trees' side -- so both are kept, and both are reported.
    not_in_species = [] if species is None else sorted(set(labels) - species.taxa)
    species_only = [] if species is None else sorted(species.taxa - set(labels))
    rename_hint = (f"  The species tree has {len(species_only)} labels no tree carries, e.g. "
                   f"{species_only[:5]}: if one is the same taxon under another name, rename it in "
                   f"the species tree." if species_only else "")
    if taxa is not None:
        unknown = sorted(set(taxa) - set(bit))
        if unknown:
            raise SystemExit(f"--taxa: {len(unknown)} labels no tree carries, e.g. {unknown[:5]}")
        clash = sorted(set(taxa) & set(outgroups))
        if clash:
            raise SystemExit(f"--taxa lists outgroups {clash}; every outgroup is pruned")
        if species is not None:
            absent = sorted(set(taxa) - species.taxa)
            if absent:
                raise SystemExit(f"--taxa: {len(absent)} labels are not in the species tree, "
                                 f"e.g. {absent[:5]}.{rename_hint}")
        S = sorted(set(taxa))
    else:
        if not_in_species and not n_taxa:
            raise SystemExit(
                f"{len(not_in_species)} taxa in the trees are not in the species tree, e.g. "
                f"{not_in_species[:5]}.{rename_hint}  A taxon the species tree lacks cannot be "
                f"placed relative to the root.  Otherwise restrict S with --n-taxa / --taxa, and "
                f"those taxa are pruned instead.")
        ranked = sorted((lab for lab in labels if lab not in outgroups
                         and (species is None or lab in species.taxa)),
                        key=lambda lab: (-freq[lab], lab))
        if n_taxa > len(ranked):
            what = "non-outgroup taxa" if species is None else "taxa the species tree also carries"
            raise SystemExit(f"--n-taxa {n_taxa}: the file has {len(ranked)} {what}")
        S = sorted(ranked[:n_taxa] if n_taxa else ranked)
    if len(S) < 3:
        raise SystemExit(f"S has {len(S)} taxa; a rooted cluster needs at least 3")
    s_mask = sum(bit[t] for t in S)
    need = partial or len(S)

    eligible, no_og, too_few = [], 0, 0
    for i, t in enumerate(masks):
        if og_mask and not t & og_mask:
            no_og += 1
        elif (t & s_mask).bit_count() < need:
            too_few += 1
        else:
            eligible.append(i)
    if too_few and not partial and taxa is None and not n_taxa:
        raise SystemExit(
            f"{too_few:,} of {len(masks) - no_og:,} trees lack some of the {len(S)} taxa.  A "
            f"cluster from a tree missing taxa is not a cluster of the full set, and the extractor "
            f"cannot tell.  Restrict S with --n-taxa / --taxa, or accept partial trees with "
            f"--partial MIN.")
    if not eligible:
        raise SystemExit(f"no tree is eligible: {no_og:,} carry no outgroup, {too_few:,} carry "
                         f"fewer than {need} of the {len(S)} taxa in S")
    reserve: list[int] = []
    if m and len(eligible) > m:
        # The sample is drawn exactly as it always was, so a scheme that never rejects a tree uses
        # the same trees it did before the reserve existed; the reserve has its own generator.
        chosen = sorted(random.Random(seed).sample(eligible, m))
        picked = set(chosen)
        reserve = [i for i in eligible if i not in picked]
        random.Random(seed).shuffle(reserve)
    else:
        chosen = eligible
    return S, chosen, reserve, {"n_trees": len(masks), "n_labels": len(labels),
                                "no_outgroup": no_og, "too_few": too_few,
                                "eligible": len(eligible), "not_in_species": not_in_species,
                                "species_only": species_only}


def prune_to(newick: str, keep) -> str:
    """The tree restricted to the leaves in ``keep``: unary nodes suppressed, internal labels gone.

    ``OG.drop_outgroup``'s ``emit``, generalised from one leaf to any set.  A subtree with no kept
    leaf goes, and a node left with one child is walked through with its edge **added** to the
    child's -- ``OG._resolve``'s rule, and the reason this is not a string edit.  A unary root is
    walked through too, its edge discarded, so a rooted tree comes back rooted at the kept leaves'
    LCA.  An unrooted string keeps its written trifurcation, which is what lets this run *before*
    ``OG.reroot_on_outgroup``.

    Iterative: a caterpillar of a few hundred leaves is deeper than Python's recursion limit.
    Lengths are written with ``repr``, which round-trips a float exactly.
    """
    parent, length, lab, children = OG.parse_newick(newick)
    n = len(parent)
    live = [False] * n
    for i in range(n - 1, -1, -1):          # parent[i] < i: children before parents
        live[i] = any(live[c] for c in children[i]) if children[i] else lab[i] in keep
    if not live[0]:
        raise ValueError(f"prune_to: no leaf of {newick[:60]!r} is kept")
    kids = [[c for c in children[i] if live[c]] for i in range(n)]

    def walk(i: int) -> tuple[int, float]:
        """The first node at or below ``i`` not left with exactly one child, and the edge sum."""
        acc = length[i]
        while len(kids[i]) == 1:
            i = kids[i][0]
            acc += length[i]
        return i, acc

    root = walk(0)[0]
    body: dict[int, str] = {}
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if not kids[v]:
            body[v] = lab[v] or ""
        elif done:
            body[v] = "(" + ",".join(f"{body[w]}:{acc!r}" for w, acc in map(walk, kids[v])) + ")"
        else:
            stack.append((v, True))
            stack.extend((walk(c)[0], False) for c in kids[v])
    return body[root] + ";"


def rooting_scheme(outgroups=(), midpoint=False, species=None) -> str:
    """``outgroup``, ``midpoint``, ``species-tree`` or ``as-written``; refuses more than one."""
    asked = [name for name, on in (("outgroup", bool(outgroups)), ("midpoint", midpoint),
                                   ("species-tree", species is not None)) if on]
    if len(asked) > 1:
        raise SystemExit(f"rooting: {' and '.join(asked)} were both asked for; pick one")
    return asked[0] if asked else "as-written"


def prepare(path: Path, chosen, dst: Path, masks, labels, S, outgroups=(), *, midpoint=False,
            species=None, reserve=()) -> dict:
    """Prune, root and check every chosen tree; write one per line to the gzip ``dst``.

    ``camus.prepare_gene_trees``' checks, for the failures it names: the leaf set is exactly what
    was kept (which also checks the scan's regex against a real parser), no tree lacks branch
    lengths, and root polytomies are counted -- or, with no rooting scheme at all, refused.

    **Outgroup and as-written** are the path this always was.  A tree with nothing to prune reaches
    ``reroot_on_outgroup`` as the untouched input line, the call the benchmark made, so a camus
    file's gzip is the benchmark's byte for byte.

    **Midpoint and species-tree** root first and prune to S second: the root is a property of the
    gene tree, placed on every taxon the scheme can read (for the species tree, every taxon it
    carries -- the others are pruned before rooting, since nothing says where they sit).  Pruning a
    rooted tree walks through a unary root, so the root stays where it was put.

    They can also reject a tree (``rooting.RootingError``).  A rejected tree is counted by reason
    and replaced: each further pass over the file takes as many trees from the front of ``reserve``
    as are still missing, until ``len(chosen)`` are written or the reserve runs out.  Replacements
    arrive out of file order, so while a reserve exists the rooted trees -- at most ``len(chosen)``
    of them -- are held in memory and written in file order at the end.  With no reserve (``--m 0``,
    or no more trees eligible than ``--m``) a rejected tree simply is not used.
    """
    scheme = rooting_scheme(outgroups, midpoint, species)
    new = scheme in ("midpoint", "species-tree")
    want = frozenset(S)
    counts: Counter = Counter()
    rejected: Counter = Counter()

    def one(i: int, line: str) -> str:
        # On the input line, not the rooted tree: ``prune_to`` and every rooting write a length on
        # every edge -- ``0.0`` where there was none -- so after either one the colon is always
        # there and the check would be vacuous.
        if ":" not in line:
            raise ValueError(f"{Path(path).name} tree {i}: no branch lengths. W13's blen_*/"
                             f"brank_*/bdepth_*/bsib_* would degenerate silently.")
        have = _names(masks[i], labels)
        if not new:
            og = next((o for o in outgroups if o in have), None)
            keep = (have & want) | ({og} if og else set())
            if og and keep == have:
                t = line
            else:
                t = prune_to(line, keep)
                counts["n_pruned"] += 1
            if og:
                t = OG.reroot_on_outgroup(t, og, keep_lengths=True)
            expect = keep - {og}
        else:
            expect = have & want
            if scheme == "midpoint":
                t = RT.midpoint_root(line)
            else:
                known = have & species.taxa
                t, kind = species.root(line if known == have else prune_to(line, known))
                counts[f"rooted_{kind}"] += 1
            if expect != have:
                t = prune_to(t, expect)
                counts["n_pruned"] += 1
        parent, _, lab, children = OG.parse_newick(t)
        got = frozenset(lab[j] for j in range(len(parent)) if not children[j])
        if got != expect:
            raise ValueError(f"{Path(path).name} tree {i}: carries {len(got)} taxa after "
                             f"pruning, expected {len(expect)} (missing "
                             f"{sorted(expect - got)[:5]}, extra {sorted(got - expect)[:5]})")
        if len(children[0]) > 2:
            if scheme == "as-written":
                raise SystemExit(
                    f"{Path(path).name} tree {i}: the root has out-degree "
                    f"{len(children[0])}, so the tree is unrooted and has no rooting to keep. "
                    f"Pass --outgroup, --midpoint or --species-tree.")
            counts["n_polytomy"] += 1
        return t

    target = len(chosen)
    hold = new and bool(reserve)
    held: dict[int, str] = {}
    pool = iter(reserve)
    pending, rounds, n_written = list(chosen), 0, 0
    with gzip.open(dst, "wt") as fh:
        while pending:
            pick = iter(sorted(pending))
            nxt = next(pick, None)
            for i, line in enumerate(_lines(path)):
                if nxt is None:
                    break
                if i != nxt:
                    continue
                nxt = next(pick, None)
                if new:
                    try:
                        t = one(i, line)
                    except RT.RootingError as exc:
                        rejected[exc.reason] += 1
                        continue
                else:
                    t = one(i, line)
                if hold:
                    held[i] = t
                else:
                    fh.write(t + "\n")
                n_written += 1
            pending = list(islice(pool, target - n_written)) if hold else []
            rounds += bool(pending)
        for i in sorted(held):
            fh.write(held[i] + "\n")
    return {"n_loci": n_written, "n_pruned": counts["n_pruned"], "n_polytomy": counts["n_polytomy"],
            "scheme": scheme, "rooted_taxon": counts["rooted_taxon"],
            "rooted_clade": counts["rooted_clade"], "rejected": dict(sorted(rejected.items())),
            "topup_rounds": rounds}


def count_clusters(gz: Path, taxon_index: dict[str, int], m: int) -> extract.SupportCounter:
    """``extract.count_newick_file``'s loop, less each tree's own leaf set.

    ``SupportCounter.add_locus`` drops the *full* taxon set, which on a full-taxon tree is its root
    cluster.  On a partial tree the root cluster is the tree's taxon sampling -- not evidence of a
    clade -- and it survives that filter, so it is dropped here.  That is also what keeps
    ``_tree_stats``' support == n_obs cross-check exact: ``blen.GeneTree.internal_nodes`` never
    scores a root (``parent > 0``).  On a full-taxon tree this is the landed loop unchanged.
    """
    counter = extract.SupportCounter(len(taxon_index), [m])
    buf: list[int] = []
    for line in _lines(gz):
        buf.clear()
        extract.clusters_from_newick(line, taxon_index, buf)
        root = 0
        for x in buf:
            root |= x
        counter.add_locus(buf if root == counter.full_mask else [x for x in buf if x != root])
    return counter


def load_model(path: Path, log=sys.stderr):
    """One checkpoint, refused if its feature widths are not this library's -- the checks
    ``run_gnn_cpu.load_models`` makes, since a width mismatch otherwise fails inside the first
    matmul and reads as a corrupt file.  Its conv and epoch go to ``log``: a run directory's
    checkpoints get replaced, and that line is how a network is traced back to the file it came
    from."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = P.PolicyConfig(**dict(ck["policy_config"]))
    if (cfg.n_node_features != len(I.NODE_FEATURES)
            or cfg.n_global_features != len(I.GLOBAL_FEATURES)):
        raise SystemExit(
            f"{path}: trained with {cfg.n_node_features} node / {cfg.n_global_features} global "
            f"features; this library has {len(I.NODE_FEATURES)} / {len(I.GLOBAL_FEATURES)}.  The "
            f"checkpoint and --opc disagree.")
    if cfg.conv not in P.CONVS:
        raise SystemExit(f"{path}: conv {cfg.conv!r}; this library has {sorted(P.CONVS)}.  The "
                         f"checkpoint and --opc disagree.")
    run_cfg_path = path.parent / "config.json"
    run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}
    if run_cfg.get("decode", "pointwise") != "pointwise":
        raise SystemExit(f"{path}: decode {run_cfg['decode']!r}; this implements the pointwise "
                         f"(W10) decode only")
    net = P.Policy(cfg)
    net.load_state_dict(ck["policy"])
    net.eval()
    print(f"[infer] model {path}: conv {cfg.conv}, epoch {ck.get('epoch')}", file=log, flush=True)
    return net


def forward(net, xn, xg, graph, gs):
    """``run_gnn_cpu.forward``: one encoder pass, the pointwise head.

    A conv that reads pairs (``PairConv``, ``conv: pair``) also needs the candidates' words,
    ``n_taxa`` and ``m`` in its graph tensors -- ``policy.graph_tensors_for``'s rule, which is what
    training passed -- and raises without them.  Every other conv gets exactly the call it always
    got, so segment checkpoints reproduce their landed networks.
    """
    pairs = net.cfg.layers and getattr(P.CONVS.get(net.cfg.conv), "needs_pairs", False)
    with torch.no_grad():
        g = (P.graph_tensors(graph, gs.support, "cpu", words=gs.words, n_taxa=int(gs.n_taxa),
                             m=float(gs.m))
             if pairs else P.graph_tensors(graph, gs.support, "cpu"))
        enc = net.encode(torch.as_tensor(xn), torch.as_tensor(xg), g)
        return torch.sigmoid(net.cluster_logits(enc)).float().numpy()


def forward_logits(net, xn, xg, graph, gs):
    """:func:`forward`, also returning the logit: ``(z float64, p)``, ``p`` bit for bit forward's.

    The logit is read before the sigmoid (``cluster_anatomy/scoring.py``'s rule): a float32 sigmoid
    saturates to 1.0 near z = 17 and turns distinct logits into ties.
    """
    pairs = net.cfg.layers and getattr(P.CONVS.get(net.cfg.conv), "needs_pairs", False)
    with torch.no_grad():
        g = (P.graph_tensors(graph, gs.support, "cpu", words=gs.words, n_taxa=int(gs.n_taxa),
                             m=float(gs.m))
             if pairs else P.graph_tensors(graph, gs.support, "cpu"))
        enc = net.encode(torch.as_tensor(xn), torch.as_tensor(xg), g)
        z = net.cluster_logits(enc)
        return z.double().numpy(), torch.sigmoid(z).float().numpy()


def infer(trees: Path, net, ks, *, outgroups=(), midpoint=False, species_tree=None, taxa=None,
          n_taxa=0, partial=0, m=1000, seed=0, tau=None, tau_frac=None, order=None,
          decoder="level1", max_level=None, prefix=False, annotate=False,
          log=sys.stderr) -> list[dict]:
    """One dict per ``k``: ``k``, ``decoder``, ``tau``, ``enewick``, ``n_selected``, ``n_retics``,
    ``seconds``.

    ``tau`` / ``tau_frac`` are :func:`resolve_tau`'s, applied to the trees actually used -- after
    sampling and replacement, so ``tau_frac`` scales with the ``m`` the features divide by.  Neither
    given is ``TAU``, the command line's default.

    ``prefix`` also returns ``prefix_truncated`` and ``prefix_rank_laminar``, each
    :meth:`opc.rl.prefix.Prefix.as_dict` plus ``newick``: the tree the decode held before it first
    spent reticulation budget, under the two stop rules ``opc.rl.prefix`` defines.  The decode is
    then the traced one, which selects exactly what the untraced one does.

    ``species_tree`` is a path or a :class:`rooting.SpeciesTree`.  ``order`` overrides ``sorted(S)``
    as the taxon order and must be a permutation of S.  It exists so a camus cell can be run under
    its truth order, which is what reproducing a landed row takes.

    ``annotate`` also returns, per ``k``, ``z`` (every candidate's logit), ``sel``, ``znet`` (the
    eNewick with each scored internal tree node labelled by its logit), ``clusters``
    (:func:`opc.rl.annotate.cluster_rows`) and ``pool`` (the logit's mean and sd over the pool);
    and the first row carries ``instance``: the pool's words, support, taxon order, ``m``, the
    rooting summary and ``|C|``.  It changes nothing the decode sees.
    """
    t0 = time.perf_counter()
    tag = RX.decoder_tag(decoder, max_level)          # before the scan: a bad flag costs nothing
    resolve_tau(1, tau, tau_frac)                     # likewise
    relaxed = decoder == "relaxed"
    species = species_tree
    if species_tree is not None and not isinstance(species_tree, RT.SpeciesTree):
        try:
            species = RT.SpeciesTree.read(species_tree)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--species-tree {species_tree}: {exc}") from None
    rooting_scheme(outgroups, midpoint, species)

    masks, labels, freq = scan(trees)
    S, chosen, reserve, stats = select(masks, labels, freq, outgroups=outgroups, taxa=taxa,
                                       n_taxa=n_taxa, partial=partial, m=m, seed=seed,
                                       species=species)
    if stats["not_in_species"]:
        # Not a refusal once S is restricted, but never silent: at --n-taxa near the file's taxon
        # count, the missing taxon is exactly one S would otherwise have held.
        only = stats["species_only"]
        print(f"[infer] note: {len(stats['not_in_species'])} taxa the trees carry are not in the "
              f"species tree, so they are pruned and cannot be in S, e.g. "
              f"{stats['not_in_species'][:5]}"
              f"{f'; the species tree has {len(only)} labels no tree carries, e.g. {only[:5]}' if only else ''}",
              file=log, flush=True)
    taxa_order = S if order is None else list(order)
    if sorted(taxa_order) != S:
        raise ValueError("order must be a permutation of the selected taxon set")
    taxon_index = {lab: i for i, lab in enumerate(taxa_order)}

    with tempfile.TemporaryDirectory(prefix="infer_gnn_") as work:
        gz, table = Path(work) / "trees.nwk.gz", Path(work) / "clusters.parquet"
        prep = prepare(trees, chosen, gz, masks, labels, S, outgroups, midpoint=midpoint,
                       species=species, reserve=reserve)
        mm = prep["n_loci"]
        if not mm:
            raise SystemExit(f"no tree could be rooted ({prep['scheme']}): {len(chosen):,} tried, "
                             f"rejected {prep['rejected']}")
        counter = count_clusters(gz, taxon_index, mm)
        words_raw, support, n_raw = counter.finalize(min_support=1)
        extract.write_cluster_table(table, words_raw, support, [mm], len(taxa_order), n_raw,
                                    source=str(trees))
        words, sup, meta = extract.read_cluster_table(table)
        thr = resolve_tau(mm, tau, tau_frac)
        gs = candidates.build(words, sup[:, meta["checkpoints"].index(mm)], len(taxa_order),
                              m=mm, tau=thr)
        if not gs.n_clusters:
            raise SystemExit(f"|C| = 0: no clade is in {thr:,} of the {mm:,} trees used "
                             f"({_tau_text(thr, tau_frac, mm)}); lower --tau or --tau-frac")
        graph = OverlapCSR.of(gs)
        xn = I.node_features(gs, graph, I._tree_stats(gz, taxon_index, gs, mm))

    rooted = f"rooting {prep['scheme']}"
    if prep["scheme"] == "species-tree":
        rooted += f": {prep['rooted_taxon']:,} on one taxon, {prep['rooted_clade']:,} on a clade"
    if prep["scheme"] in ("midpoint", "species-tree"):
        n_rej = sum(prep["rejected"].values())
        why = ", ".join(f"{r} {c:,}" for r, c in prep["rejected"].items())
        rooted += (f", rejected {n_rej:,}{f' ({why})' if n_rej else ''}"
                   f"{f', top-up rounds {prep['topup_rounds']}' if prep['topup_rounds'] else ''}")
    print(f"[infer] {stats['n_trees']:,} trees over {stats['n_labels']:,} taxa | S = {len(S)} taxa"
          f"{f', partial >= {partial}' if partial else ''} | skipped {stats['no_outgroup']:,} "
          f"with no outgroup, {stats['too_few']:,} short of S | {stats['eligible']:,} eligible, "
          f"{mm:,} used{f' (seed {seed})' if mm < stats['eligible'] else ''}, {prep['n_pruned']:,} "
          f"pruned | {rooted} | root polytomies {prep['n_polytomy']} | |C| = {gs.n_clusters:,} "
          f"at {_tau_text(thr, tau_frac, mm)} | "
          f"{time.perf_counter() - t0:.1f}s", file=log, flush=True)

    out = []
    for k in ks:
        t1 = time.perf_counter()
        if annotate:
            z, p = forward_logits(net, xn, I.global_features(gs, graph, k), graph, gs)
        else:
            p = forward(net, xn, I.global_features(gs, graph, k), graph, gs)
        # Same scores, rank rule and budget either way; only what counts as legal differs.
        pre = {}
        if relaxed:
            if prefix:
                env, _trace, pre = PX.prefixes_relaxed(gs.words, len(taxa_order), k, max_level, p,
                                                       graph)
                sel = env.selection
            else:
                env = RX.RelaxedDecoder(gs.words, len(taxa_order), k, max_level)
                sel = env.decode(p)
            if not env.verify():
                raise AssertionError(f"k={k}: the {tag} decode fails its own check: "
                                     f"{env.verify_reason}")
        else:
            env = ClusterEnv(gs, budget=k, graph=graph)
            if prefix:
                trace, pre = PX.prefixes_level1(env, p, graph)
                sel = trace.selection
            else:
                sel = NC.decode(env, p)
            if not env.verify():
                raise AssertionError(f"k={k}: the decode is not a feasible cluster system")
        got = (RX.realise if relaxed else search.realise)(gs.words, sel, len(taxa_order),
                                                          taxa_order)
        if got is None:
            raise RuntimeError(f"k={k}: the {sel.size} selected clusters realise no network")
        row = {"k": k, "decoder": tag, "tau": thr, "enewick": NW.to_enewick(got[0]),
               "n_selected": int(sel.size), "n_retics": int(got[0].n_retics),
               "seconds": time.perf_counter() - t1}
        if annotate:
            row.update(_annotation(got, gs, z, p, sel, relaxed=relaxed, taxa_order=taxa_order,
                                   const={"ftau": tau_frac, "tau": thr, "decoder": tag, "k": k}))
            if not out:
                row["instance"] = {"words": gs.words, "support": np.asarray(gs.support),
                                   "taxa_order": list(taxa_order), "m": mm,
                                   "n_clusters": int(gs.n_clusters), "rooting": rooted,
                                   "scan": {kk: v for kk, v in stats.items()
                                            if isinstance(v, (int, float, str))}}
        for mode, pf in pre.items():
            row[PX.LABELS[mode]] = {**pf.as_dict(), "newick": PX.tree_newick(
                gs.words, pf.rows, len(taxa_order), taxa_order)}
        print(f"[infer] k={k}{f' ({tag})' if relaxed else ''}: {row['n_selected']} clusters "
              f"selected, {row['n_retics']} reticulation(s) realised, {row['seconds']:.2f}s",
              file=log, flush=True)
        for mode in pre:
            b = row[PX.LABELS[mode]]
            print(f"[infer] k={k} {PX.LABELS[mode]}: {b['n_selected']} clusters, "
                  + (f"stopped at step {b['n_steps']} on the cluster ranked {b['stop_rank']}"
                     if b["stopped"] else "never spent budget: the whole decode"),
                  file=log, flush=True)
        out.append(row)
    return out


def _annotation(got, gs, z, p, sel, *, relaxed, taxa_order, const) -> dict:
    """``infer``'s per-``k`` annotation: the labelled network, its node table, the pool's stats."""
    net, full = got
    rows = AN.node_rows(full, gs.words)
    if relaxed and not np.array_equal(rows[:len(sel)], np.asarray(sel, dtype=np.int64)):
        raise AssertionError(f"k={const['k']}: relaxed node ids are not the selection rows")
    pool = AN.pool_stats(z)
    return {"z": z, "sel": np.asarray(sel, dtype=np.int64), "pool": pool,
            "znet": NW.to_enewick(net, node_label=AN.tree_labels(net, rows, z)),
            "clusters": AN.cluster_rows(net, full, rows, z=z, p=p, support=gs.support,
                                        selected=sel, n_taxa=len(taxa_order),
                                        taxa_order=taxa_order, stats=pool, **const)}


CLUSTER_COLS = ("ftau", "tau", "decoder", "k", "node", "kind", "cand_row", "taxa", "size",
                "support", "z", "z_std", "p", "rank_frac", "selected")


def write_annotation(prefix: Path, rows: list[dict], *, model: Path, argv: list[str]) -> None:
    """``--annotate PREFIX``: ``.znet.tsv``, ``.clusters.tsv``, ``.logits.npz``, ``.manifest.json``."""
    import hashlib
    import subprocess
    prefix.parent.mkdir(parents=True, exist_ok=True)
    inst = rows[0]["instance"]
    (prefix.parent / f"{prefix.name}.znet.tsv").write_text(
        "".join(f"{r['k']}\t{r['znet']}\n" for r in rows))
    with open(prefix.parent / f"{prefix.name}.clusters.tsv", "w") as fh:
        fh.write("\t".join(CLUSTER_COLS) + "\n")
        for r in rows:
            for c in r["clusters"]:
                fh.write("\t".join("" if c[col] is None else
                                   (f"{c[col]:.6g}" if isinstance(c[col], float) else str(c[col]))
                                   for col in CLUSTER_COLS) + "\n")
    ks = np.array([r["k"] for r in rows], dtype=np.int64)
    np.savez_compressed(prefix.parent / f"{prefix.name}.logits.npz",
                        words=inst["words"], support=inst["support"],
                        taxa_order=np.array(inst["taxa_order"]), ks=ks,
                        z=np.stack([r["z"] for r in rows]).astype(np.float32),
                        **{f"sel_k{r['k']}": r["sel"] for r in rows})

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    manifest = {
        "argv": argv, "model": str(model.resolve()),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "m": inst["m"], "n_clusters": inst["n_clusters"], "rooting": inst["rooting"],
        "scan": inst["scan"], "n_taxa": len(inst["taxa_order"]),
        "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short", "."),
        "per_k": [{"k": r["k"], "decoder": r["decoder"], "tau": r["tau"],
                   "n_selected": r["n_selected"], "n_retics": r["n_retics"],
                   "seconds": r["seconds"], "pool": r["pool"]} for r in rows],
    }
    (prefix.parent / f"{prefix.name}.manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="a checkpoint, e.g. "
                    "checkpoints/_mix2L/s0/best.pt or checkpoints/_l2mix_pair/s0/best_ha.pt")
    ap.add_argument("--opc", type=Path, default=HERE / "pylib",
                    help="the directory holding opc/, or opc/ itself; read before import")
    ap.add_argument("--trees", type=Path, required=True,
                    help="one Newick tree per line, plain or .gz")
    ap.add_argument("--k", required=True, help="comma list of reticulation budgets, e.g. 1,2,3")
    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--n-taxa", type=int, default=0,
                     help="restrict S to the N most frequent taxa (ties broken by label)")
    sub.add_argument("--taxa", type=Path, default=None,
                     help="restrict S to these labels, one per line")
    root = ap.add_mutually_exclusive_group()
    root.add_argument("--outgroup", default="",
                      help="ranked comma list: each tree is rooted on the first one it carries and "
                           "every one is pruned; a tree carrying none is skipped")
    root.add_argument("--midpoint", action="store_true",
                      help="root each tree at the middle of its longest leaf-to-leaf path")
    root.add_argument("--species-tree", type=Path, default=None, metavar="FILE",
                      help="a rooted species tree: each gene tree is rooted on the split it has at "
                           "the root over that tree's taxa, every taxon kept; a tree without that "
                           "split is rejected and replaced")
    ap.add_argument("--partial", type=int, default=0, metavar="MIN",
                    help="accept a tree carrying >= MIN taxa of S, pruned to what it has "
                         "(default: every tree must carry all of S)")
    ap.add_argument("--m", type=int, default=1000,
                    help="trees to use, a seeded sample when more are eligible; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    cut = ap.add_mutually_exclusive_group()
    cut.add_argument("--tau", type=int, default=None, metavar="N",
                     help=f"a candidate clade must be in >= N of the trees used (default {TAU}, "
                          f"as every checkpoint trained)")
    cut.add_argument("--tau-frac", default=None, metavar="F",
                     help="a candidate clade must be in >= ceil(F * m) of the m trees used, "
                          "F in (0, 1]; training's 2 of 1,000 is 0.002")
    ap.add_argument("--decoder", choices=RX.DECODERS, default="level1",
                    help="level1: the formal (L)/(N3O) decoder every landed row used (default); "
                         "relaxed: Hasse-diagram constraints (opc.rl.relaxed), needs --max-level")
    ap.add_argument("--max-level", type=int, default=None,
                    help="the relaxed decoder's level cap, >= 1; required with --decoder relaxed "
                         "and refused without it")
    ap.add_argument("--stop-at-reticulation", action="store_true",
                    help="write, instead of each network, the tree the decode held before its "
                         "first reticulation, under both stop rules: lines k<TAB>mode<TAB>Newick "
                         "with mode prefix_truncated or prefix_rank_laminar (opc.rl.prefix)")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch and BLAS threads; read before import")
    ap.add_argument("--out", type=Path, default=None, help="default stdout")
    ap.add_argument("--annotate", type=Path, default=None, metavar="PREFIX",
                    help="also write PREFIX.znet.tsv (each network with its internal tree nodes "
                         "labelled by the GNN logit), PREFIX.clusters.tsv (one row per network "
                         "node: z, pool-standardised z, p, support), PREFIX.logits.npz (every "
                         "candidate's logit per k) and PREFIX.manifest.json; the plain output "
                         "is unchanged")
    args = ap.parse_args()

    try:
        ks = [int(k) for k in args.k.split(",") if k.strip()]
    except ValueError:
        ap.error(f"--k {args.k!r}: expected a comma list of integers")
    if not ks or min(ks) < 0:
        ap.error("--k needs one or more non-negative integers")
    if args.partial and args.partial < 3:
        ap.error("--partial MIN must be at least 3: a rooted tree on fewer taxa has no cluster")
    if args.m < 0:
        ap.error("--m must be >= 0")
    if args.tau is not None and args.tau < 1:
        ap.error("--tau must be at least 1: 1 already keeps every clade any tree has")
    if args.tau_frac is not None:
        try:
            resolve_tau(1, tau_frac=args.tau_frac)
        except ValueError:
            ap.error(f"--tau-frac {args.tau_frac!r}: expected a number in (0, 1]")
    try:
        RX.decoder_tag(args.decoder, args.max_level)
    except ValueError as exc:
        ap.error(f"--decoder {args.decoder}: {exc}")
    outgroups = tuple(dict.fromkeys(o.strip() for o in args.outgroup.split(",") if o.strip()))
    taxa = None if args.taxa is None else [
        s.strip() for s in args.taxa.read_text().splitlines() if s.strip()]

    rows = infer(args.trees, load_model(args.model), ks, outgroups=outgroups,
                 midpoint=args.midpoint, species_tree=args.species_tree, taxa=taxa,
                 n_taxa=args.n_taxa, partial=args.partial, m=args.m, seed=args.seed,
                 tau=args.tau, tau_frac=args.tau_frac, decoder=args.decoder,
                 max_level=args.max_level, prefix=args.stop_at_reticulation,
                 annotate=args.annotate is not None)
    if args.annotate is not None:
        write_annotation(args.annotate, rows, model=args.model, argv=sys.argv)
    if args.stop_at_reticulation:
        text = "".join(f"{r['k']}\t{PX.LABELS[mode]}\t{r[PX.LABELS[mode]]['newick']}\n"
                       for r in rows for mode in PX.MODES)
    else:
        text = "".join(f"{r['k']}\t{r['enewick']}\n" for r in rows)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
