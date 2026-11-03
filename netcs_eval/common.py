"""Shared paths, labels, truth normalisation and the true-root oracle for ``netcs_bench_corpus`` --
see README.md.

Every script here runs under ``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**: it
pins every BLAS to one thread before numpy can be imported and puts ``gnn_cpu_v1/pylib`` on the
path by absolute path -- ``l3reg_eval/common.py``'s two rules, for its reasons.

The corpus
----------
``netcs_bench_corpus/LEVEL_{1,2}/TAXA_{50,100,200}/network_<id>/``: 50 networks per (level, n),
each at four ILS levels, so **1,200 conditions on 300 truths**.  Per condition the corpus ships the
true network (``.newick``), IQ-TREE gene trees (``.estimated_gene_trees``, 911-1000 of them) and,
under ``benchmarking/``, TOB-QMC's tree of blobs in its default and fast modes and -- at n <= 100
only -- TINNIK's.  The true tree of blobs (``.blob_tree``) exists once per network, in the
``ils_1.0`` file, because the topology does not depend on ILS (300 of 300 networks, checked by
``stage.py``).

Three things about it drive everything below, all measured on 2026-09-12:

* **There is no outgroup.**  The gene trees are unrooted trifurcations and carry no ``OUT``, and
  both trees of blobs are unrooted.  NetCS and BROOQS need no root, but the GNN and greedy read
  *rooted* clusters -- hence the true-root oracle.
* **The truth's taxon count is n..n+5, not n.**  ``n`` in a phase name is the corpus's nominal
  size; every driver counts taxa off the truth.
* **Two truths are outside ``networks.Network``** (a node donating to two hybrids:
  ``LEVEL_2/TAXA_100/network_1011`` and ``LEVEL_2/TAXA_200/network_26``) and two carry a size-2
  cycle ``to_enewick`` cannot round-trip (``LEVEL_1/TAXA_50/network_1147``, ``1549``).  See
  :func:`normalise_truth`.

The true-root oracle (user decision, 2026-09-12)
------------------------------------------------
camus, l2reg and l3reg rooted every gene tree on a simulated ``OUT``.  This corpus has none, so the
root is supplied from the truth instead, and that is an **oracle**: the GNN and greedy rows here
are handed a root the resolvers never see.

* **Targets.**  :func:`root_targets` takes every tree the truth displays (each switching of its
  reticulations, at most 2^5), suppresses unary nodes, and keeps each one's root bipartition.  A
  hybrid straddling the root makes these differ, which it does on about half of the networks.
* **Cost** of a split ``S``: ``min over targets T of min(|S sym T|, |S sym T^c|)`` -- how many taxa
  are on the wrong side of the nearest displayed root.  It is 0 exactly when ``S`` is a displayed
  root split; under ILS that is **1-59% of loci** depending on the cell, so ``stage.py`` records the
  cost per condition rather than assuming it away.
* **The pick.**  The eligible edge of least cost.  Ties go to the edge whose smaller side is
  smallest, then lexicographically first by sorted labels -- arbitrary but deterministic.  ``OUT``
  is inserted at the edge's **midpoint**, so rooting on it (``outgroup.reroot_on_outgroup``, which
  ``camus.prepare_gene_trees`` already calls for the ``outgroup`` arms) gives a root whose two
  child edges are each half the original edge.  No library change is needed.
* **Used three times, by one function** (:func:`insert_out_at_truth`): the gene trees
  (``stage.py``); the trees of blobs read as trees (``stage.py``); and NetCS/BROOQS output, which
  is re-rooted there only for the rooted metrics -- F1 and luay.  Hybrid ancestry needs no root.
  On a network only a tree edge outside every hybrid's subtree is eligible, so the re-rooting
  never places the root below a hybrid.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from itertools import combinations
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PKG = REPO / "netcs_bench_corpus"     # the corpus: read, never written
GNN = REPO / "gnn_cpu_v1"
BLOB = REPO / "blob_cpu_v1"
PYLIB = GNN / "pylib"
LANDED = REPO / "results" / "rl"

INPUTS = HERE / "inputs"              # camus-data layout: inputs/L<l>_n<n>/<id>/{true_net.nwk, est_ils<x>.nwk}
TOB = HERE / "tob"                    # <key>.tqmc.nwk (resolver input), <key>.<kind>.rooted.nwk (tree rows)
RUNS = HERE / "runs"
CELLS = HERE / "CELLS.tsv"
SHARDS = HERE / "shards"
STAGE = HERE / "stage.json"

if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

OUT = "OUT"
LEVELS = (1, 2)
SIZES = (50, 100, 200)
ILS = ("0.25", "0.5", "1.0", "2.0")
#: The ``gene_trees`` column: the corpus's own IQ-TREE trees, rooted by the true-root oracle.
GENE_TREES = "netcs_iqtree_trueroot"

#: Tree-of-blobs kinds: key -> the tool name in the corpus's ``benchmarking/`` file names.
TOB_KINDS = ("tqmc", "tqmc_fast", "tinnik", "true")
TOB_TOOL = {"tqmc": "TOB-QMC-default", "tqmc_fast": "TOB-QMC-fast", "tinnik": "TINNIK"}
TOB_LABELS = {"tqmc": "TOB-QMC", "tqmc_fast": "TOB-QMC-fast", "tinnik": "TINNIK",
              "true": "true TOB*"}
#: The one tree of blobs NetCS and BROOQS resolve (user decision, 2026-09-12).
RESOLVER_TOB = "tqmc"
RESOLVERS = ("netcs", "brooqs")
RESOLVER_LABELS = {"netcs": "NetCS", "brooqs": "BROOQS"}

#: checkpoint tag -> camus_a104 label.  **Exact match only**: ``_mix2`` is a prefix of ``_mix2L``.
GNN_LABELS = {"_mix": "ours/prior", "_mix2": "ours/2x-data", "_mix2L": "ours/2x-data-full",
              "_mix75": "ours/+n75", "_l2mix": "ours/l2mix", "_l2mix75": "ours/l2mix75",
              # _l2mixha/s0 ships two checkpoints of one run: best.pt (selected on F1) and
              # best_ha.pt (on hybrid ancestry), the latter recorded as tag "_l2mixha:best_ha".
              "_l2mixha": "ours/l2mixHA/selF1", "_l2mixha:best_ha": "ours/l2mixHA/selHA",
              # _l2mix_pair/s0: the same two-checkpoint shape, on the pair conv (conv: pair).
              "_l2mix_pair": "ours/l2mix_pair/selF1", "_l2mix_pair:best_ha": "ours/l2mix_pair/selHA"}


# ------------------------------------------------------------------------------------------ names

def phase_name(level: int, n: int) -> str:
    return f"netcs_L{level}_n{n}"


def phase_dir(level: int, n: int) -> str:
    return f"L{level}_n{n}"


def arm_name(ils: str) -> str:
    return f"netcs_ils{ils}"


def trees_name(ils: str) -> str:
    return f"est_ils{ils}.nwk"


#: phase -> (level, nominal n); arm -> ILS.  The same keys ``opc.rl.camus`` registers, which
#: ``stage.py``'s names gate checks rather than assumes.
PHASES = {phase_name(lv, n): (lv, n) for lv in LEVELS for n in SIZES}
ARMS = {arm_name(i): i for i in ILS}


def key_of(level: int, n: int, netid: int, ils: str) -> str:
    """One condition's key.  ``CamusCell.key`` (``n50_rep1590``) is shared by every level and ILS,
    so it keys nothing here."""
    return f"L{level}_n{n}_net{netid}_ils{ils}"


def cell_name(phase: str, rep: int, arm: str) -> str:
    """The cell-JSON stem ``run_gnn_cpu.py`` writes: ``{phase}_rep{RR}_{arm}``."""
    return f"{phase}_rep{rep:02d}_{arm}"


def corpus_dir(level: int, n: int, netid: int) -> Path:
    return PKG / f"LEVEL_{level}" / f"TAXA_{n}" / f"network_{netid}"


def corpus_network(level: int, n: int, netid: int, ils: str) -> Path:
    return corpus_dir(level, n, netid) / f"network_{netid}_ils_{ils}.newick"


def corpus_trees(level: int, n: int, netid: int, ils: str) -> Path:
    return corpus_dir(level, n, netid) / f"network_{netid}_ils_{ils}.estimated_gene_trees"


def corpus_tob(level: int, n: int, netid: int, ils: str, kind: str) -> Path:
    d = corpus_dir(level, n, netid)
    if kind == "true":
        return d / f"network_{netid}_ils_1.0.blob_tree"      # one per network: ILS-free topology
    return (d / "benchmarking" /
            f"network_{netid}_ils_{ils}.estimated_gene_trees.{TOB_TOOL[kind]}_ALPHA1e-7_BETA0.95.tree")


def network_ids(level: int, n: int) -> list[int]:
    return sorted(int(p.name.split("_", 1)[1])
                  for p in (PKG / f"LEVEL_{level}" / f"TAXA_{n}").glob("network_*") if p.is_dir())


def cell_dir(phase: str, rep: int, inputs: Path = INPUTS) -> Path:
    return Path(inputs) / phase_dir(*PHASES[phase]) / f"{rep:02d}"


def trees_path(phase: str, rep: int, arm: str) -> Path:
    """One condition's staged gene trees: what the GNN and greedy read (the resolvers do not)."""
    return cell_dir(phase, rep) / trees_name(ARMS[arm])


def phylonet_truth_path(phase: str, rep: int) -> Path:
    return cell_dir(phase, rep) / "true_net.phylonet.nwk"


def tob_input_path(key: str) -> Path:
    return TOB / f"{key}.{RESOLVER_TOB}.nwk"


def tob_rooted_path(key: str, kind: str) -> Path:
    return TOB / f"{key}.{kind}.rooted.nwk"


# ---------------------------------------------------------------------------------------- helpers

_GAMMA = re.compile(r"::[0-9.eE+-]+")
#: A hybrid tag.  In this corpus always ``#H<k>``.
HTAG = re.compile(r"#(?:[A-Za-z]+)?\d+")
#: A leaf label: a name right after "(" or ",".  Never a "#H" reference or an internal label.
_LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")


def strip_gamma(enewick: str) -> str:
    """``phylonet.strip_gamma``'s regex, here so the netdist-only paths never import phylonet."""
    return _GAMMA.sub("", enewick)


def leaf_labels(enewick: str) -> frozenset[str]:
    return frozenset(_LEAF.findall(enewick))


def topology(enewick: str) -> str:
    """The string with every ``:length`` and ``::gamma`` removed."""
    return re.sub(r":[^,();]*", "", enewick.strip())


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def parse_reps(spec: str) -> set[int] | None:
    """``"0-9,15"`` -> ``{0..9, 15}``; empty means every rep."""
    if not spec:
        return None
    want: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def cell_decoder(cell: dict) -> str:
    """A cell JSON's decoder tag (``opc.rl.relaxed.decoder_tag``)."""
    from opc.rl.relaxed import decoder_tag
    return decoder_tag(cell.get("decoder", "level1"), cell.get("max_level"))


def decoder_suffix(tag: str) -> str:
    """The CSV label suffix for a decoder tag: nothing for the formal decoder, `` [relaxed L2]``
    for ``relaxed_L2``."""
    return "" if tag == "level1" else " [" + tag.replace("_", " ") + "]"


def truth_of(rep: int, *, phase: str, inputs: Path = INPUTS):
    """The staged truth with OUT pruned: ``camus.truth_of``, the object the GNN rows are scored on.
    The arm does not enter it -- one truth serves every ILS level."""
    from opc.rl import camus
    return camus.truth_of(camus.CamusCell(phase, rep, "", Path(inputs)))


# ---------------------------------------------------------------------------- truth normalisation

def _match(s: str, i: int) -> int:
    """The index of the ``)`` closing the ``(`` at ``s[i]``."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced parentheses")


def _top_commas(group: str) -> int:
    """Commas directly inside the outermost parentheses of ``group``."""
    depth = count = 0
    for ch in group:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 1:
            count += 1
    return count


def unwrap_unary_root(text: str) -> tuple[str, int]:
    """``(<body>:len);`` -> ``<body>``, repeatedly; returns the body (no ``;``) and how many unary
    roots were removed.

    The n = 50 and n = 100 truths are written with a root edge above the first split, and PhyloNet
    sizes that unary node like any other, so a truth with it and a prediction without it would
    differ by a node neither method could have got right.  The n = 200 truths have none.
    """
    s = text.strip().rstrip(";").strip()
    removed = 0
    while s.startswith("("):
        close = _match(s, 0)
        if close != len(s) - 1 or _top_commas(s) != 0:
            break
        m = re.fullmatch(r"(\(.*\))(:[-0-9.eE+]*)?", s[1:close], re.S)
        if not m:
            raise ValueError("a unary root above something other than a parenthesised group")
        s = m.group(1)
        removed += 1
    return s, removed


def bare_tags(body: str) -> list[re.Match]:
    """Each hybrid's bare ``#Hk`` occurrence -- the one with no subtree, written right after
    ``(`` or ``,`` -- with its ``:len::gamma``."""
    return [m for m in re.finditer(r"#(?:[A-Za-z]+)?\d+(?::[^,()]*)?", body)
            if m.start() > 0 and body[m.start() - 1] in "(,"]


def wrap_tags(body: str, matches) -> str:
    """Put each given bare tag inside a unary node: ``,#H1:0::0.3`` -> ``,(#H1:0::0.3)``.

    The unary node becomes the donor in the tag's place.  Its cluster is the hybrid's, so the
    hardwired clusters, k and the level are all unchanged -- which ``stage.py`` checks on every
    truth it is *not* needed for, as a control.
    """
    for m in sorted(matches, key=lambda m: m.start(), reverse=True):
        body = body[:m.start()] + "(" + m.group(0) + ")" + body[m.end():]
    return body


def normalise_truth(text: str) -> dict:
    """The corpus truth as this pipeline stores it.

    * ``body``: unary root removed; and when ``networks.from_enewick`` refuses the string because
      one node donates to two hybrids -- a shape ``Network`` cannot hold -- the fewest bare tags
      wrapped in unary nodes that make it parse, first combination in string order.
    * ``phylonet``: the corpus string with only the unary root removed and gammas stripped.  It is
      the truth every luay compares against: PhyloNet reads the double donor as written, and our
      writer cannot round-trip the size-2-cycle truths.
    """
    from opc.simulate import networks as N

    body, removed = unwrap_unary_root(text)
    out = {"corpus_body": body, "phylonet": strip_gamma(body) + ";", "unary_removed": removed,
           "wrapped": []}
    try:
        N.from_enewick(body + ";")
        return {**out, "body": body}
    except ValueError as exc:
        if "donates to two hybrids" not in str(exc):
            raise
    tags = bare_tags(body)
    for r in range(1, len(tags) + 1):
        for combo in combinations(tags, r):
            cand = wrap_tags(body, combo)
            try:
                N.from_enewick(cand + ";")
            except ValueError:
                continue
            return {**out, "body": cand, "wrapped": [HTAG.match(m.group(0)).group(0) for m in combo]}
    raise ValueError("no wrapping of bare hybrid tags makes this network parse")


# ------------------------------------------------------------------------------ the true-root oracle

def root_targets(net) -> list[int]:
    """The distinct root bipartitions of the trees ``net`` displays, as bitmasks over
    ``net.labels``' order, each written as the side **without** taxon 0 (module docstring)."""
    from opc.simulate import networks as N

    full = (1 << len(net.labels)) - 1
    order = N._postorder(net)                                            # noqa: SLF001
    switch = [(r.hybrid, net.tree_parent[r.hybrid], r.donor) for r in net.retics
              if net.tree_parent[r.hybrid] != r.donor]                   # a shortcut has one parent
    targets: set[int] = set()
    for bits in range(1 << len(switch)):
        cut = {(tp, h) if bits >> j & 1 else (d, h) for j, (h, tp, d) in enumerate(switch)}

        def kids(v, cut=cut):
            return list(dict.fromkeys(c for c in net.net_children(v) if (v, c) not in cut))

        mask: dict[int, int] = {}
        for v in order:
            if net.is_leaf(v):
                mask[v] = 1 << net.leaf_taxon[v]
            else:
                acc = 0
                for c in kids(v):
                    acc |= mask[c]
                mask[v] = acc
        v = net.root
        while True:
            live = [c for c in kids(v) if mask[c]]
            if len(live) != 1:
                break
            v = live[0]
        for c in live:
            t = mask[c]
            if 0 < t < full:
                targets.add(full ^ t if t & 1 else t)
    return sorted(targets)


def split_cost(mask: int, targets, n: int) -> int:
    """Taxa on the wrong side of the nearest target root split (module docstring)."""
    best = n
    for t in targets:
        d = (mask ^ t).bit_count()
        if n - d < d:
            d = n - d
        if d < best:
            best = d
    return best


def _names(mask: int, names: list[str]) -> tuple[str, ...]:
    return tuple(sorted(names[i] for i in range(mask.bit_length()) if mask >> i & 1))


def insert_out_at_truth(newick: str, index: dict[str, int], targets, *, keep_lengths: bool,
                        network: bool = False) -> dict:
    """``newick`` with ``OUT`` inserted at the midpoint of the edge the oracle picks.

    ``index`` is the truth's label -> bit map.  An unknown label raises ``KeyError``.  With
    ``network``, a bare ``#Hk`` leaf stands for its body's leaf set, and only a tree edge outside
    every hybrid's subtree is eligible.  Internal labels other than hybrid tags -- IQ-TREE's support
    values -- are dropped; lengths are kept only with ``keep_lengths``.

    Returns ``newick`` (the string with OUT), ``cost``, ``ties`` (edges sharing the least cost),
    ``side`` (the leaf set below the picked edge), ``root_mask`` and ``n_leaves`` (non-tag leaves),
    so a caller can check the taxon set without parsing twice.
    """
    from opc.simulate import outgroup as OG

    parent, length, lab, children = OG.parse_newick(newick)
    nn, n = len(parent), len(index)
    full = (1 << n) - 1
    is_tag = [bool(x) and HTAG.fullmatch(x) is not None for x in lab]
    mask = [0] * nn
    n_leaves = 0
    if not network:
        if any(is_tag):
            raise ValueError("a hybrid tag in a tree")
        for i in range(nn - 1, -1, -1):
            if not children[i]:
                mask[i] = 1 << index[lab[i]]
                n_leaves += 1
            if i:
                mask[parent[i]] |= mask[i]
        below_h = [False] * nn
    else:
        body_of = {lab[i]: i for i in range(nn) if is_tag[i] and children[i]}
        memo: list[int | None] = [None] * nn

        def m(i: int) -> int:
            if memo[i] is None:
                if not children[i]:
                    memo[i] = m(body_of[lab[i]]) if is_tag[i] else 1 << index[lab[i]]
                else:
                    acc = 0
                    for c in children[i]:
                        acc |= m(c)
                    memo[i] = acc
            return memo[i]

        for i in range(nn):
            mask[i] = m(i)
        n_leaves = sum(1 for i in range(nn) if not children[i] and not is_tag[i])
        below_h = [False] * nn
        for i in range(nn):                               # parent[i] < i: a forward pass inherits
            below_h[i] = (i > 0 and below_h[parent[i]]) or (is_tag[i] and bool(children[i]))
    eligible = [i for i in range(1, nn) if not below_h[i] and not is_tag[i]]
    if not eligible:
        raise ValueError("no eligible edge to root on")
    costs = [split_cost(mask[i], targets, n) for i in eligible]
    best = min(costs)
    tied = [i for i, c in zip(eligible, costs) if c == best]
    pick = tied[0]
    if len(tied) > 1:
        names = [""] * n
        for name, j in index.items():
            names[j] = name

        def key(i: int):
            s, o = mask[i], full ^ mask[i]
            return min((s.bit_count(), _names(s, names)), (o.bit_count(), _names(o, names)))
        pick = min(tied, key=key)

    def emit(i: int) -> str:
        if not children[i]:
            return lab[i] or ""
        out = "(" + ",".join(edge(c) for c in children[i]) + ")"
        return out + lab[i] if network and is_tag[i] else out

    def edge(c: int) -> str:
        s = emit(c)
        if c == pick:
            if keep_lengths:
                half = f"{length[c] / 2:.10g}"
                return f"({s}:{half},{OUT}:0):{half}"
            return f"({s},{OUT})"
        return f"{s}:{length[c]:.10g}" if keep_lengths else s

    return {"newick": emit(0) + ";", "cost": best, "ties": len(tied), "side": mask[pick],
            "root_mask": mask[0], "n_leaves": n_leaves}


def string_clusters(newick: str, index: dict[str, int]) -> set[frozenset[str]]:
    """Nontrivial hardwired clusters read off an eNewick string by ``outgroup.parse_newick`` -- a
    second parser, independent of ``networks.from_enewick`` and with no opinion about which parent
    donates.  ``stage.py`` holds every truth to it."""
    from opc.simulate import outgroup as OG

    parent, _len, lab, children = OG.parse_newick(newick)
    is_tag = [bool(x) and HTAG.fullmatch(x) is not None for x in lab]
    body_of = {lab[i]: i for i in range(len(parent)) if is_tag[i] and children[i]}
    memo: dict[int, int] = {}

    def m(i: int) -> int:
        if i not in memo:
            if not children[i]:
                memo[i] = m(body_of[lab[i]]) if is_tag[i] else 1 << index[lab[i]]
            else:
                acc = 0
                for c in children[i]:
                    acc |= m(c)
                memo[i] = acc
        return memo[i]

    names = [""] * len(index)
    for name, j in index.items():
        names[j] = name
    n = len(index)
    return {frozenset(_names(m(i), names)) for i in range(len(parent))
            if children[i] and 1 < m(i).bit_count() < n}
