"""Shared paths, labels and helpers for the ``l2_n1000_gt_50k_v1`` stress test -- see README.md.

This copies ``l2reg_eval/common.py``, trimmed to one dataset. Every script here runs under
``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**, because it does two things
before anything else loads:

* It pins every BLAS to one thread before numpy can be imported.
* It puts ``gnn_cpu_v1/pylib`` on the path.

**Why this is its own package rather than a third ``REG_GT`` in l2reg_eval.**

* The dataset is n = 1000. l2reg_eval hard-wires n = 200 in ``N_TAXA``, in the ``inputs/n200``
  paths and in ``make_csv``'s luay denominators.
* It has no TREE-QMC tree of blobs, and l2reg's ``stage_tobs`` reads one unconditionally.
* Its MANIFEST ``n_loci`` (50,000) counts the *true* trees. The estimated arm holds
  ``n_est_loci`` = 10,000.

**The resolvers' tree of blobs is our GNN's.** ``stage_gnn_tob.py`` collapses the selF1 GNN network
to its tree of blobs, and BROOQS and NetCS re-resolve that. It is the only inferred TOB this
dataset has.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PKG = REPO / "l2_n1000_gt_50k_v1"     # the dataset: read, never written
GNN = REPO / "gnn_cpu_v1"
BLOB = REPO / "blob_cpu_v1"
PYLIB = GNN / "pylib"
LANDED = REPO / "results" / "rl"

ROOT = HERE
INPUTS = ROOT / "inputs"              # camus-data layout: inputs/n1000/<RR>/{true_net.nwk, est_bp1000.nwk}
TOB = ROOT / "tob"                    # <key>.{gnn,true}.rooted.nwk, gnn_tob.json
RUNS = ROOT / "runs"
CELLS = ROOT / "CELLS.tsv"
STAGE = ROOT / "stage.json"

if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

N_TAXA = 1000
OUT = "OUT"
#: Registered in ``opc.rl.camus``: phase -> ``inputs/n1000``, arm -> ``est_bp1000.nwk`` rooted on OUT.
PHASE, ARM, TREES = "l2_n1000_gt50k", "est_bp1000", "est_bp1000.nwk"
#: The CSV's ``gene_trees``.  It names the tree count, as gt10k's ``est_bp1000_gt10k`` does, and the
#: taxon count, so this table concatenates with l2reg's without colliding.
GENE_TREES = "est_bp1000_gt10k_n1000"
#: Every method reads every estimated tree, at tau = ceil(0.002 m), the gt-scaling rule.  The values
#: are written out rather than computed, so ``make_csv.py`` checks the drivers' own resolution
#: against an independent number.
M, TAU = 10000, 20
TAU_FRAC = "0.002"
LEVEL = 2

#: ``gnn``: the selF1 GNN network collapsed to its tree of blobs (``stage_gnn_tob.py``).
#: ``true``: the oracle, built from the truth's bridges.
VARIANTS = ("gnn", "true")
RESOLVERS = ("netcs", "brooqs")
#: The scope agreed on 2026-10-06: the resolvers on the GNN's TOB.  ``true`` is optional
#: (``run_resolvers.py --variants true``), and ``make_csv`` takes its rows when they exist.
RESOLVER_VARIANTS = ("gnn",)
GNN_REQUIRED = (("_l2mix_pair", "_l2mix_pair:best_ha"), f"relaxed_L{LEVEL}")
GREEDY_REQUIRED = ("level1", f"relaxed_L{LEVEL}")
#: The GNN whose network becomes the resolvers' tree of blobs: tag, seed, decoder, and its run.
TOB_SOURCE = ("_l2mix_pair", 0, f"relaxed_L{LEVEL}")
GNN_RUN = f"gnn_t1_l2mix_pair_ftau0.002_relaxedL{LEVEL}"

GNN_LABELS = {"_l2mix_pair": "ours/l2mix_pair/selF1", "_l2mix_pair:best_ha": "ours/l2mix_pair/selHA"}
TOB_LABELS = {"gnn": "GNN TOB", "true": "true TOB*"}
RESOLVER_LABELS = {
    ("netcs", "gnn"): "NetCS (GNN TOB)", ("brooqs", "gnn"): "BROOQS (GNN TOB)",
    ("netcs", "true"): "NetCS*(true TOB)", ("brooqs", "true"): "BROOQS*(true TOB)",
}

_GAMMA = re.compile(r"::[0-9.eE+-]+")


def strip_gamma(enewick: str) -> str:
    return _GAMMA.sub("", enewick)


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def manifest() -> list[dict]:
    return read_tsv(PKG / "MANIFEST.tsv")


def parse_reps(spec: str) -> set[int] | None:
    """``"0-9,15"`` -> ``{0..9, 15}``; empty means every rep."""
    if not spec:
        return None
    want: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def cell_name(phase: str, rep: int, arm: str) -> str:
    """The cell-JSON stem ``run_gnn_cpu.py`` writes: ``{phase}_rep{RR}_{arm}``."""
    return f"{phase}_rep{rep:02d}_{arm}"


def trees_path(rep: int) -> Path:
    return INPUTS / "n1000" / f"{rep:02d}" / TREES


def cell_decoder(cell: dict) -> str:
    from opc.rl.relaxed import decoder_tag
    return decoder_tag(cell.get("decoder", "level1"), cell.get("max_level"))


def decoder_suffix(tag: str) -> str:
    return "" if tag == "level1" else " [" + tag.replace("_", " ") + "]"


def truth_of(rep: int, *, inputs: Path = INPUTS, phase: str = PHASE):
    """The staged truth with OUT pruned: ``camus.truth_of``, the object the GNN rows are scored on."""
    from opc.rl import camus
    return camus.truth_of(camus.CamusCell(phase, rep, ARM, Path(inputs)))


# ---------------------------------------------------------------------------- the tree of blobs

_PTE = None


def pte():
    """``prefix_tob_eval/pte_common.py``, loaded by path under its own name.

    That module is never named ``common``, so loading it cannot shadow this one. It provides the
    bridge-based tree of blobs (``bridge_tob_clusters``), the string cluster scan
    (``string_clusters``), ``n_cycle_components`` and ``run_timed_to``.
    """
    global _PTE
    if _PTE is None:
        if "pte_common" in sys.modules:
            _PTE = sys.modules["pte_common"]
        else:
            spec = importlib.util.spec_from_file_location(
                "pte_common", REPO / "prefix_tob_eval" / "pte_common.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["pte_common"] = mod
            spec.loader.exec_module(mod)
            _PTE = mod
    return _PTE


def clusters_to_newick(clusters, taxa) -> str:
    """A compatible family of clusters over ``taxa`` -> the Newick of its Hasse diagram.

    Polytomies are kept, and there are no lengths. Children are ordered by their smallest label, so
    the same family always gives the same string. A family that is not laminar raises.
    """
    taxa = frozenset(taxa)
    fam = sorted({frozenset(c) for c in clusters if 1 < len(c) < len(taxa)},
                 key=lambda c: (-len(c), sorted(c)[0]))
    nodes = [taxa] + fam
    parent = [None] * len(nodes)
    for i in range(1, len(nodes)):
        c = nodes[i]
        # The smallest earlier node containing c.  Nodes are sorted by decreasing size, so the last
        # one that contains it is the smallest.
        for j in range(i - 1, -1, -1):
            if c <= nodes[j]:
                parent[i] = j
                break
        for j in range(1, i):
            d = nodes[j]
            if c & d and not (c <= d or d <= c):
                raise ValueError("clusters are not compatible: not a tree")
    kids: dict[int, list] = {i: [] for i in range(len(nodes))}
    covered: dict[int, set] = {i: set() for i in range(len(nodes))}
    for i in range(1, len(nodes)):
        kids[parent[i]].append(i)
        covered[parent[i]] |= nodes[i]

    def key(x):
        return min(map(_label_key, nodes[x])) if isinstance(x, int) else _label_key(x)

    def emit(i: int) -> str:
        items = kids[i] + list(nodes[i] - covered[i])
        items.sort(key=key)
        return "(" + ",".join(emit(x) if isinstance(x, int) else x for x in items) + ")"

    return emit(0) + ";"


def _label_key(lab: str):
    m = re.fullmatch(r"([A-Za-z]*)(\d+)", lab)
    return (m.group(1), int(m.group(2))) if m else (lab, -1)


def tree_of_blobs_general(net) -> str:
    """Any network's tree of blobs: every non-bridge edge contracted, and so every blob of any
    level becomes one polytomy.

    ``camus_nets.tree_of_blobs`` contracts one cycle per reticulation and is wrong at level 2. This
    keeps the cluster below each bridge (``pte_common.bridge_tob_clusters``) and writes their
    Hasse diagram.
    """
    return clusters_to_newick(pte().bridge_tob_clusters(net), net.labels)
