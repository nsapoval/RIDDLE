"""Shared paths, labels and helpers for the l3reg_n200_v1 benchmark and, under
``REG_GT``, its 5,000- and 10,000-tree versions -- see README.md.

Every script here runs under ``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**: it
pins every BLAS to one thread before numpy can be imported (the sibling drivers' rule, for their
reason -- a pool that has already sized itself ignores a later change) and puts
``gnn_cpu_v1/pylib``, the library the checkpoints were trained against, on the path by absolute
path.  ``src/`` has no ``opc.rl`` and must not shadow it.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: **Which dataset** (README, "Gene-tree scaling").  Unset: ``l3reg_n200_v1``, 1,000 trees, and
#: every constant in this module is what it always was.  ``REG_GT=gt5k`` / ``gt10k``: the same 20
#: networks with 5,000 / 10,000 gene trees, staged and run under ``<this dir>/<REG_GT>/``, with
#: their own phase name.  An environment variable because every script reads this module's
#: constants at import, before any argument is parsed; each script's first log line names it.
GT = os.environ.get("REG_GT", "")
#: REG_GT -> (the trees every method reads, tau).  tau is ``ceil(0.002 * m)`` -- training's 2 of
#: 1,000, which is also exactly the v1 row -- written out rather than computed, so ``make_csv.py``
#: checks the drivers' own resolution of ``--tau-frac`` against an independent number.
_GT = {"": (1000, 2), "gt5k": (5000, 10), "gt10k": (10000, 20)}
if GT not in _GT:
    raise SystemExit(f"REG_GT={GT!r}: expected gt5k or gt10k, or unset for the 1,000-tree dataset")
M, TAU = _GT[GT]
TAU_FRAC = "0.002" if GT else None    # None: tau = 2 as a count, the landed rows' rule

PKG = REPO / "l3reg_n200_v1"          # the dataset: read, never written
if GT:
    PKG = REPO / f"l3reg_n200_{GT}_v1"
GNN = REPO / "gnn_cpu_v1"
BLOB = REPO / "blob_cpu_v1"
PYLIB = GNN / "pylib"
LANDED = REPO / "results" / "rl"

ROOT = HERE / GT if GT else HERE      # where this dataset's staging and runs live
INPUTS = ROOT / "inputs"              # camus-data layout: inputs/n200/<RR>/{true_net.nwk, est_bp1000.nwk}
TOB = ROOT / "tob"                    # <key>.{raw,fixed,true}.rooted.nwk
RUNS = ROOT / "runs"
CELLS = ROOT / "CELLS.tsv"
STAGE = ROOT / "stage.json"

if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

N_TAXA = 200
OUT = "OUT"
#: The staged cells' phase and arm, both registered in ``opc.rl.camus`` for this dataset:
#: ``l3reg_n200`` resolves to ``inputs/n200`` and ``est_bp1000`` to ``est_bp1000.nwk`` rooted on
#: OUT.  **The phase name is the whole of the separation.**  ``CamusCell.key`` is ``n200_repRR``
#: for cam200, l2reg and l3reg alike, so nothing but the phase keeps one dataset's cell JSONs, run
#: directories and reference lookups out of another's.  l2reg learned this the hard way while it
#: borrowed camus's ``cam200`` / ``cam_g500``: every file and log line said cam200, a forgotten
#: ``--reference`` keyed the camus cam200 rows onto l2reg cells, and a forgotten ``--out`` found
#: the camus run's ``cam200_*`` files and skipped every l2reg cell as done.
PHASE, ARM, TREES = "l3reg_n200", "est_bp1000", "est_bp1000.nwk"
#: The repo's arm id for AliSim + IQ-TREE re-estimates of its own networks (``opc.arms``), which is
#: what these trees are -- the same model as ``alisim_gt_v1/v2`` and ``l2reg_n200_v1``, at 1000 bp.
GENE_TREES = "est_bp1000"
if GT:
    # Registered in opc.rl.camus beside l3reg_n200.  The CSV's gene_trees says how many trees, as
    # camus's cam_g500 does, so the three datasets' CSVs concatenate without colliding.
    PHASE, GENE_TREES = f"{PHASE}_{GT}", f"{GENE_TREES}_{GT}"
VARIANTS = ("raw", "fixed", "true")
RESOLVERS = ("netcs", "brooqs")
#: The level cap of this dataset's headline relaxed decoder.
LEVEL = 3
#: What the gt5k/gt10k runs were scoped to (2026-09-21): BROOQS and NetCS on the raw and fixed
#: trees of blobs only (all three TOB rows are still emitted: they cost nothing), the
#: ``_l2mix_pair`` checkpoints at the headline decoder only, greedy at level 1 and that decoder.
#: ``GNN_REQUIRED = None`` is v1's rule: every shipped checkpoint at level 1.
RESOLVER_VARIANTS = ("raw", "fixed") if GT else VARIANTS
GNN_REQUIRED = ((("_l2mix_pair", "_l2mix_pair:best_ha"), f"relaxed_L{LEVEL}") if GT else None)
GREEDY_REQUIRED = ("level1", f"relaxed_L{LEVEL}") if GT else ("level1",)

#: checkpoint tag -> camus_a104 label.  **Exact match only**: ``_mix2`` is a prefix of ``_mix2L``.
#: ``_l2mix`` is ``_mix2L``'s four phases plus ``l2t25``/``l2t50``, the level-2 training arms, and
#: ships one seed; camus_a104 holds no rows for it, so its formal-decoder rows are new too.
GNN_LABELS = {"_mix": "ours/prior", "_mix2": "ours/2x-data", "_mix2L": "ours/2x-data-full",
              "_mix75": "ours/+n75", "_l2mix": "ours/l2mix", "_l2mix75": "ours/l2mix75",
              # _l2mixha/s0 ships two checkpoints of one run: best.pt (selected on F1) and
              # best_ha.pt (on hybrid ancestry), the latter recorded as tag "_l2mixha:best_ha".
              "_l2mixha": "ours/l2mixHA/selF1", "_l2mixha:best_ha": "ours/l2mixHA/selHA",
              # _l2mix_pair/s0: the same two-checkpoint shape, on the pair conv (conv: pair).
              "_l2mix_pair": "ours/l2mix_pair/selF1", "_l2mix_pair:best_ha": "ours/l2mix_pair/selHA"}
#: The raw TOB is the camus protocol (``root_tob_on_outgroup`` on TREE-QMC's output), so it keeps
#: the plain camus labels and joins with those rows; the corrected one says so.
TOB_LABELS = {"raw": "TOB-QMC", "fixed": "TOB-QMC (fixed)", "true": "true TOB*"}
RESOLVER_LABELS = {
    ("netcs", "raw"): "NetCS", ("brooqs", "raw"): "BROOQS",
    ("netcs", "fixed"): "NetCS (fixed TOB)", ("brooqs", "fixed"): "BROOQS (fixed TOB)",
    ("netcs", "true"): "NetCS*(true TOB)", ("brooqs", "true"): "BROOQS*(true TOB)",
}

_GAMMA = re.compile(r"::[0-9.eE+-]+")


def strip_gamma(enewick: str) -> str:
    """``phylonet.strip_gamma``'s regex, here so the netdist-only paths never import phylonet."""
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
    """One network's staged gene trees: what the GNN, greedy and both resolvers read."""
    return INPUTS / "n200" / f"{rep:02d}" / TREES


def cell_decoder(cell: dict) -> str:
    """A cell JSON's decoder tag (``opc.rl.relaxed.decoder_tag``).  Cells written before the
    relaxed decoder existed carry no field and were decoded by the formal one."""
    from opc.rl.relaxed import decoder_tag
    return decoder_tag(cell.get("decoder", "level1"), cell.get("max_level"))


def decoder_suffix(tag: str) -> str:
    """The CSV label suffix for a decoder tag: nothing for the formal decoder, `` [relaxed L2]``
    for ``relaxed_L2`` -- so every relaxed row sits beside its level-1 row under its own name."""
    return "" if tag == "level1" else " [" + tag.replace("_", " ") + "]"


def truth_of(rep: int, *, inputs: Path = INPUTS, phase: str = PHASE):
    """The staged truth with OUT pruned: ``camus.truth_of``, the object the GNN rows are scored on."""
    from opc.rl import camus
    return camus.truth_of(camus.CamusCell(phase, rep, ARM, Path(inputs)))
