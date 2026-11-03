"""Paths, the family table and the MANIFEST readers for the missing-taxa benchmark -- see README.md.

Every script here runs under ``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**.  It
puts ``subsample_eval/`` on ``sys.path`` and imports that package's ``common`` *by that name*, as
``nni_ensemble/ne_common.py`` does and for its reason: ``make_sub_csv`` imports ``common`` by name,
and ``subsample_eval/common.py`` pins every BLAS to one thread and puts ``gnn_cpu_v1/pylib`` at
``sys.path[0]``.  This module is named ``tm_common`` so it cannot shadow it.

The data
--------
``inputs/<fam>/MANIFEST.tsv`` (written on the Linux box) lists every (condition, cell): the published
benchmark's gene trees with ingroup taxa deleted, per tree, by ``missing.tsv``.  A condition is
``p<P>_c<CAP>`` -- ``p`` the per-tree probability of deleting anything, ``cap`` the per-tree cap in
percent of n (``M`` taxa) -- or ``control``, the untouched inputs (``sha256_staged ==
sha256_source``), which is the published benchmark itself and is never staged here.  ``path`` is
relative to ``inputs/``; OUT is in every tree.  The trees of blobs TREE-QMC built on exactly those
files are ``runs/tob/<fam>/<cond>/<cell>.{json,raw.tob,tob,pvalues.tob}``.

``stage.py`` turns each condition whose inputs are present into ``staged/<fam>/<cond>/``: the
CamusCell layout the drivers read (``<phase dir>/<RR>/<arm file>``, a relative symlink into
``inputs/``), the parent's truth copied byte for byte, and a ``CELLS.tsv``.  Everything downstream
is keyed on (family, condition) and finds the conditions by what is staged, so a condition synced
later is picked up by re-running ``stage.py``.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUB = REPO / "subsample_eval"
if str(SUB) not in sys.path:
    sys.path.insert(0, str(SUB))

import common as SC  # noqa: E402  -- subsample_eval's: BLAS pin, gnn_cpu_v1/pylib at sys.path[0]

INPUTS = HERE / "inputs"
STAGED = HERE / "staged"
RUNS = HERE / "runs"
RESULTS = HERE / "results"
TOBS = RUNS / "tob"
OUT = SC.OUT
CONTROL = "control"
#: ``run_gnn_cpu.py --partial``: a tree must keep at least this many taxa.  TREE-QMC ignores trees
#: with fewer than 4 species; no staged tree comes near it (the tightest, n = 25 at cap 36%, keeps
#: 16), so it is a floor that refuses a broken file, not a filter.
PARTIAL = 4
GNN_RUN = "gnn_t1_l2mix_pair"


@dataclass(frozen=True)
class Family:
    """One published benchmark the missing-taxa conditions are drawn from."""

    name: str
    title: str          # the figure prefix the notebooks use
    level: int          # the relaxed decoder's level cap
    parent_inputs: Path  # the published staged inputs (camus-data layout): truths, source trees
    parent_cells: Path   # the published CELLS.tsv

    @property
    def manifest(self) -> Path:
        return INPUTS / self.name / "MANIFEST.tsv"


FAMILIES = {
    "camus": Family("camus", "CAMUS", 2, SC.GNN / "inputs", SC.GNN / "CELLS.tsv"),
    "l2reg": Family("l2reg", "L2reg", 2, REPO / "l2reg_eval" / "inputs", REPO / "l2reg_eval" / "CELLS.tsv"),
    "l3reg": Family("l3reg", "L3reg", 3, REPO / "l3reg_eval" / "inputs", REPO / "l3reg_eval" / "CELLS.tsv"),
    # Staged 2026-10-23 (README, NetCS): OUT re-placed on 4.7% of its trees, T1b in ingroup mode.
    "netcs": Family("netcs", "NetCS", 2, REPO / "netcs_eval" / "inputs", REPO / "netcs_eval" / "CELLS.tsv"),
}


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], columns) -> None:
    SC.write_tsv(path, rows, list(columns))


def write_json(path: Path, obj, indent=1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)


def manifest(fam: str) -> list[dict]:
    """Every MANIFEST row, ``rep`` / ``n`` / ``n_loci`` / ``n_modified`` as ints."""
    rows = read_tsv(FAMILIES[fam].manifest)
    for r in rows:
        for k in ("rep", "n", "n_loci", "n_modified"):
            r[k] = int(r[k])
    return rows


def conditions(fam: str) -> list[str]:
    """The MANIFEST's conditions, control excluded, in file order."""
    out: list[str] = []
    for r in manifest(fam):
        if r["cond"] != CONTROL and r["cond"] not in out:
            out.append(r["cond"])
    return out


def staged_dir(fam: str, cond: str) -> Path:
    return STAGED / fam / cond


def cells_file(fam: str, cond: str) -> Path:
    return staged_dir(fam, cond) / "CELLS.tsv"


def staged_conds(fam: str) -> list[str]:
    """Conditions ``stage.py`` has written, in MANIFEST order."""
    if not FAMILIES[fam].manifest.exists():
        return []
    return [c for c in conditions(fam) if cells_file(fam, c).exists()]


def families_staged() -> list[str]:
    return [f for f in FAMILIES if staged_conds(f)]


def parse_conds(fam: str, spec: str) -> list[str]:
    """``all`` (or empty) -> every staged condition; else a comma list, each one staged."""
    have = staged_conds(fam)
    if not spec or spec == "all":
        return have
    want = [c for c in spec.split(",") if c]
    missing = [c for c in want if c not in have]
    if missing:
        raise SystemExit(f"{fam}: conditions {missing} are not staged (have {have}); run stage.py")
    return want


def gnn_run(fam: str, cond: str, root: Path = RUNS) -> Path:
    """``run_gnn.sh``'s run directory: the relaxed decoder at the family's level."""
    return Path(root) / fam / cond / f"{GNN_RUN}_relaxedL{FAMILIES[fam].level}"


def tob_paths(fam: str, cond: str, cell: str) -> dict[str, Path]:
    base = TOBS / fam / cond
    return {"json": base / f"{cell}.json", "raw": base / f"{cell}.raw.tob",
            "fixed": base / f"{cell}.tob", "pvalues": base / f"{cell}.pvalues.tob"}


def dataset_name(fam: str, cond: str) -> str:
    """The derived dataset ``cluster_anatomy/common.py:_derived()`` registers for this pair (the
    same rule is restated there, which cannot import this module)."""
    return f"{fam}_tm_{cond}"
