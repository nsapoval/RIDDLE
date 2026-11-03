"""Paths, the dataset registry and the seeded draw for the gene-tree subsample study -- see README.md.

Every script here runs under ``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**: it
pins every BLAS to one thread before numpy can be imported and puts ``gnn_cpu_v1/pylib`` -- the
library the checkpoints were trained against -- at ``sys.path[0]``.  ``src/`` has no ``opc.rl`` and
must not win.

The benchmarks' own ``common.py`` files are **not imported** (``prefix_eval/common.py``'s rule):
each is a module named ``common``, so loading one would shadow this module.  What is needed from
them -- a condition's key, its stratum, its truth -- is read off their ``CELLS.tsv`` /
``stage.json``, and the truth comes from ``opc.rl.camus.truth_of``, the object ``run_gnn_cpu.py``
scores against.

The draw
--------
A subsample is ``N_SUB`` of a cell's staged gene trees, drawn without replacement and **written in
source order**::

    seed    = chain_seed("sub500", dataset, phase, rep, arm, draw)       # opc.grid, SHA-256
    indices = sorted(random.Random(seed).sample(list(range(n_source)), N_SUB))

The second line is ``gnn_cpu_v1/infer_gnn_cpu.py``'s ``select()`` formula.  On a full-taxon file
every tree is eligible, so a staged subsample holds exactly the trees
``infer_gnn_cpu.py --m 500 --seed <seed>`` would pick from the same file.  ``n_source`` is the
file's own non-blank line count: 1,000 on camus, l2reg and l3reg, 911-1,000 on netcs.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GNN = REPO / "gnn_cpu_v1"
PYLIB = GNN / "pylib"
INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
RESULTS = HERE / "results"
LANDED = REPO / "results" / "rl"
#: Deliberately absent: ``run_gnn_cpu.py`` skips its parity check when ``--reference`` does not
#: exist.  The default ``gnn_cpu_v1/reference.tsv`` would hold a 500-tree camus row against the
#: landed 1,000-tree one under the same (phase, rep, arm) and exit 3.
NO_REFERENCE = HERE / "NO_REFERENCE.tsv"
DEFAULT_JAR = Path.home() / "research" / "stochastic-search" / "tools" / "PhyloNet.jar"
OUT = "OUT"

if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

from opc.grid import chain_seed  # noqa: E402

N_SUB = 500
SALT = "sub500"
#: ``run_gnn_cpu.py --models``: ``_l2mix_pair/s0``'s two checkpoint files.
MODELS = "_l2mix_pair/s0,_l2mix_pair/s0/best_ha"
#: checkpoint tag (as a cell JSON records it) -> the label every per-network CSV uses.
LABELS = {"_l2mix_pair": "ours/l2mix_pair/selF1", "_l2mix_pair:best_ha": "ours/l2mix_pair/selHA",
          "greedy": "greedy"}
METHOD_ORDER = ("ours/l2mix_pair/selF1", "ours/l2mix_pair/selHA", "greedy")
GNN_RUN, GREEDY_RUN = "gnn_t1_l2mix_pair", "greedy"


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, columns, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cell_name(phase: str, rep: int, arm: str) -> str:
    """The cell-JSON stem ``run_gnn_cpu.py`` and ``run_greedy.py`` write: ``{phase}_rep{RR}_{arm}``."""
    return f"{phase}_rep{int(rep):02d}_{arm}"


def decoder_tag(decoder: str, max_level=None) -> str:
    from opc.rl.relaxed import decoder_tag as _tag
    return _tag(decoder, max_level)


def cell_decoder(cell: dict) -> str:
    """A cell JSON's decoder tag.  Cells written before the relaxed decoder carry no field."""
    return decoder_tag(cell.get("decoder", "level1"), cell.get("max_level"))


def decoder_suffix(tag: str) -> str:
    """`` [relaxed L2]`` for ``relaxed_L2``, nothing for ``level1`` -- the landed CSVs' labels."""
    return "" if tag == "level1" else " [" + tag.replace("_", " ") + "]"


def draw_seed(dataset: str, phase: str, rep: int, arm: str, draw: int) -> int:
    return chain_seed(SALT, dataset, phase, int(rep), arm, int(draw))


def draw_indices(seed: int, n_source: int, m: int = N_SUB) -> list[int]:
    """``infer_gnn_cpu.select()``'s draw: ``m`` of ``range(n_source)``, sorted into source order."""
    if n_source < m:
        raise ValueError(f"cannot draw {m} of {n_source} trees")
    return sorted(random.Random(seed).sample(list(range(n_source)), m))


def draw_label(draw) -> str:
    """``d0`` .. ``d4`` for a subsample, ``full`` for the landed 1,000-tree runs."""
    return "full" if draw == "full" else f"d{int(draw)}"


@dataclass(frozen=True)
class Dataset:
    """One benchmark: where its staged inputs are, how many draws it gets, and which landed runs
    hold its full-data rows.

    ``full`` maps ``(kind, decoder tag)`` -> the landed run directory, ``kind`` being ``gnn`` or
    ``greedy``.  camus's relaxed greedy was never run on the full data; its entry is a run under
    ``runs/camus/full/``, made by the command sheet's one extra line.
    """

    name: str
    source: Path            # the benchmark's staged inputs/ (camus-data layout)
    cells: Path             # its CELLS.tsv
    level: int              # the relaxed decoder's level cap
    draws: int
    landed_csv: Path
    full: dict
    stage: Path | None = None           # stage.json: per-cell SHAs and strata
    shards: Path | None = None          # netcs: shards/CELLS_XX.tsv

    @property
    def decoders(self) -> tuple[str, str]:
        return ("level1", f"relaxed_L{self.level}")

    @property
    def inputs(self) -> Path:
        return INPUTS / self.name

    @property
    def manifest(self) -> Path:
        return self.inputs / "MANIFEST.tsv"

    def draw_inputs(self, draw: int) -> Path:
        return self.inputs / draw_label(draw)

    def draw_runs(self, draw, root: Path = RUNS) -> Path:
        return Path(root) / self.name / draw_label(draw)

    def run_dir(self, draw, kind: str, decoder: str, root: Path = RUNS) -> Path:
        """Where ``kind``'s cells for ``decoder`` live: the landed run for ``full``, else ours.
        A run of ours -- including camus's full-data relaxed greedy -- moves with ``root``."""
        if draw == "full":
            p = self.full[(kind, decoder)]
            return Path(root) / p.relative_to(RUNS) if p.is_relative_to(RUNS) else p
        base = GNN_RUN if kind == "gnn" else GREEDY_RUN
        suffix = "" if decoder == "level1" else "_relaxedL" + decoder.rsplit("_L", 1)[1]
        return self.draw_runs(draw, root) / (base + suffix)


def _full(gnn: Path, greedy: Path, level: int) -> dict:
    """``(kind, decoder)`` -> landed run: ``gnn`` and ``greedy`` are the level-1 runs, and each
    relaxed run sits beside its level-1 run with ``_relaxedL<level>`` appended."""
    rel = f"relaxed_L{level}"
    return {("gnn", "level1"): gnn, ("gnn", rel): gnn.with_name(f"{gnn.name}_relaxedL{level}"),
            ("greedy", "level1"): greedy,
            ("greedy", rel): greedy.with_name(f"{greedy.name}_relaxedL{level}")}


_CAMUS_FULL = _full(GNN / "runs" / "C4537CLFY0_t1_l2mix_pair",
                    REPO / "l3reg_eval" / "runs" / "greedy_camus", 2)
_CAMUS_FULL[("greedy", "relaxed_L2")] = RUNS / "camus" / "full" / "greedy_relaxedL2"

DATASETS = {
    "camus": Dataset(
        "camus", GNN / "inputs", GNN / "CELLS.tsv", 2, 5,
        LANDED / "camus_a104_per_network_l2mix.csv", _CAMUS_FULL),
    "l2reg": Dataset(
        "l2reg", REPO / "l2reg_eval" / "inputs", REPO / "l2reg_eval" / "CELLS.tsv", 2, 5,
        LANDED / "l2reg_n200_v1_per_network.csv",
        _full(REPO / "l2reg_eval" / "runs" / GNN_RUN, REPO / "l2reg_eval" / "runs" / GREEDY_RUN, 2),
        stage=REPO / "l2reg_eval" / "stage.json"),
    "l3reg": Dataset(
        "l3reg", REPO / "l3reg_eval" / "inputs", REPO / "l3reg_eval" / "CELLS.tsv", 3, 5,
        LANDED / "l3reg_n200_v1_per_network.csv",
        _full(REPO / "l3reg_eval" / "runs" / GNN_RUN, REPO / "l3reg_eval" / "runs" / GREEDY_RUN, 3),
        stage=REPO / "l3reg_eval" / "stage.json"),
    "netcs": Dataset(
        "netcs", REPO / "netcs_eval" / "inputs", REPO / "netcs_eval" / "CELLS.tsv", 2, 1,
        LANDED / "netcs_bench_per_network.csv",
        _full(REPO / "netcs_eval" / "runs" / GNN_RUN, REPO / "netcs_eval" / "runs" / GREEDY_RUN, 2),
        stage=REPO / "netcs_eval" / "stage.json", shards=REPO / "netcs_eval" / "shards"),
}
#: ``netcs_eval/make_csv.py``'s luay cache: the same ``pair_key`` scheme, so the landed netcs
#: networks' distances are read rather than re-measured.  Read-only; ours is ``runs/phylonet/``.
SEED_CACHES = {"netcs": (REPO / "netcs_eval" / "runs" / "phylonet" / "cache.jsonl",)}
