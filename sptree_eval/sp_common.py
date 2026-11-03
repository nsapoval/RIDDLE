"""Paths and constants shared by ``sptree_eval``'s own scripts -- see README.md.

Named ``sp_common``, not ``common``: ``cluster_anatomy``, ``subsample_eval`` and the benchmarks each
ship a module called ``common``, and the GNN pipeline imports two of them by that name.  Importing
this first pins BLAS to one thread and puts ``gnn_cpu_v1/pylib`` at ``sys.path[0]``, as
``cluster_anatomy/common.py`` does.  Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PYLIB = REPO / "gnn_cpu_v1" / "pylib"
if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

#: The dataset, read only.
PKG = REPO / "sptree_n200_gt10k_v1"
INPUTS = HERE / "inputs"
CELLS = HERE / "CELLS.tsv"
STAGE = HERE / "stage.json"
RUNS = HERE / "runs"
RESULTS = HERE / "results"

#: ``opc.rl.camus``'s names for these cells: the phase is new, the arm is l2reg's (raw IQ-TREE trees,
#: OUT kept, rooted on it by ``prepare_gene_trees``).
PHASE, ARM, TREES = "sptree_n200", "est_bp1000", "est_bp1000.nwk"
OUT = "OUT"
N_TAXA = 200
N_LOCI = 10000
TAXA = frozenset(f"t{i}" for i in range(1, N_TAXA + 1))
#: Non-trivial clusters of a binary rooted tree on 200 taxa: 2n - 1 nodes, less n leaves and the root.
N_TRUE = N_TAXA - 2
KEY_RE = re.compile(r"^n200_rep(\d\d)_cu(0\.1|0\.25|0\.5)$")
#: (tag, m, tau): ``cluster_anatomy/common.SPTREE_SIZES``, restated so nothing here imports torch's side.
SIZES = (("1k", 1000, 2), ("5k", 5000, 10), ("10k", 10000, 20))
SIZE_M = {t: m for t, m, _ in SIZES}
KS = tuple(range(11))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prefix_bytes(path: Path, n: int) -> bytes:
    """The first ``n`` lines of a staged trees file, byte for byte -- what a size-``n`` run reads
    (``camus.prepare_gene_trees(max_loci=n)`` reads the same lines; the file has no blank ones)."""
    out = []
    with open(path, "rb") as fh:
        for line in fh:
            out.append(line)
            if len(out) == n:
                break
    if len(out) != n:
        raise ValueError(f"{path}: {len(out)} lines, wanted a prefix of {n}")
    return b"".join(out)


def stem(rep: int) -> str:
    """The cell stem every GNN-side file uses: ``cluster_anatomy.common.cell_name``."""
    return f"{PHASE}_rep{int(rep):02d}_{ARM}"


def trees_path(rep: int) -> Path:
    return INPUTS / "n200" / f"{int(rep):02d}" / TREES


def truth_path(rep: int) -> Path:
    return INPUTS / "n200" / f"{int(rep):02d}" / "true_net.nwk"


def stage() -> dict:
    if not STAGE.exists():
        raise SystemExit(f"no {STAGE}: run sptree_eval/stage.py first")
    return json.loads(STAGE.read_text())


def parse_reps(spec: str) -> set[int] | None:
    """``"0-9,15"`` -> ``{0..9, 15}``; empty means every rep."""
    if not spec:
        return None
    want: set[int] = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi or lo) + 1))
    return want


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
