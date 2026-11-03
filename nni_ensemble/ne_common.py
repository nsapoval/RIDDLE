"""Paths, constants and shared loaders for the NNI ensemble study -- see README.md.

Every gnn-venv script here imports this module **first**.  The import order is
``ens_anatomy/ea_common.py``'s, for its reason: two sibling packages each ship a module named
``common``.  ``subsample_eval/common.py`` gets the name, because ``make_sub_csv`` imports it by that
name and evaluates ``C.OUT`` when its functions are defined (``make_sub_csv.py:121,284``);
``cluster_anatomy/common.py`` is loaded from its path as ``ca_common``.  Both pin BLAS and put
``gnn_cpu_v1/pylib`` on ``sys.path``.

Nothing here imports torch: every input is a cached cell from ``cluster_anatomy/runs/<ds>/cells/``.

**``run_gnn_cpu`` sniffs ``--threads`` out of ``sys.argv`` at import** (``make_sub_csv`` does not
import it, but ``cluster_anatomy`` code paths do).  No script here may define ``--threads`` or an
option that abbreviates to it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUB = REPO / "subsample_eval"
ANAT = REPO / "cluster_anatomy"
RUNS = HERE / "runs"
RESULTS = HERE / "results"

if str(SUB) not in sys.path:
    sys.path.insert(0, str(SUB))

import common as SC  # noqa: E402  -- subsample_eval's: BLAS, pylib at sys.path[0]

if str(SC.GNN) not in sys.path:
    sys.path.insert(1, str(SC.GNN))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CA = _load("ca_common", ANAT / "common.py")

import numpy as np  # noqa: E402

DATASETS = ("camus", "l2reg", "l3reg", "netcs")
#: Datasets run after the four published ones, kept out of ``DATASETS`` so every default -- the
#: calibration pools, ``bench``'s dataset list, ``check_nni --all`` -- means what it meant when the
#: published tables were made.  n1000: the 20 level-2 networks of ``n1000_eval`` (README, n1000).
EXTRA = ("n1000",)
#: The four published datasets re-run on other gene trees (``ca_common._derived``; README, Derived
#: datasets): 5k / 10k trees, 500-tree draws, and the cells the published tables lack.  Never in a
#: calibration pool; each is scored by its parent network's held-out fold model.
DERIVED = tuple(n for n, d in CA.DATASETS.items() if d.base)
#: The checkpoint's own training/validation networks (``ca_common._train``; README, Train-fit
#: calibration).  Never a benchmark and never in a published pool: ``trainfit.py`` fits on them and
#: ``credible --calib-pool train`` transfers that fit.  Empty until ``train_eval/`` is staged.
TRAIN = tuple(n for n, d in CA.DATASETS.items() if d.kind == "train")
#: ``sptree_eval``'s species-tree datasets, one per (gene-tree prefix, decode budget k)
#: (``ca_common._sptree``).  Never a benchmark pool: their MAP is ``credible --calib-pool train``'s
#: transfer.  Empty until ``sptree_eval/`` is staged.
SPTREE = tuple(n for n, d in CA.DATASETS.items() if d.kind == "sptree")
ALL_DATASETS = DATASETS + EXTRA + DERIVED + TRAIN + SPTREE
#: The emitted-cluster flag ``ens_anatomy/flags.py:69`` and ``cluster_anatomy`` §8 settled on.
Z_THR = 6.0
Z_SWEEP = (5.0, 6.0, 7.0)
#: Nominal credible-set levels.
ALPHAS = (0.5, 0.8, 0.9, 0.95)
#: The level the Hamming balls and the per-block option lists are written at.
ALPHA_BALL = 0.9
#: Per-block enumeration cap.  A block with at most this many configurations (``features.py``'s
#: ``r_count``) is enumerated in full; past it, only the configurations within ``RHO`` swaps of the
#: emitted one are (``blocks.radius_configs``) -- best-first never reaches a leaf on those (README).
R_CAP = 4096
#: The radius of an oversized block's configuration set: at most this many non-emitted clusters.
RHO = 2
#: A full block's configurations are listed best-first to at most ``KEEP_MAX``, stopping early once
#: their mass reaches ``1 - MASS_TAIL``.
KEEP_MAX, MASS_TAIL = 4096, 1e-4
#: Hard ceiling on ``bk_best``'s search-tree nodes (~0.5 GB): past it the cell fails, naming the
#: block, instead of growing the frontier without bound.
NODE_CAP = 1_000_000
#: Joint lazy k-best depth (the F1-oracle curve and gate N5).
K_CAP = 1024
#: Joint members whose networks are realised and get luay.
K_LUAY = (4, 16)
VARIANTS = ("z", "tc", "z+tc", "z+tc+emit")


def family(ds: str) -> str:
    """The published benchmark whose semantics ``ds`` takes: itself, or a derived dataset's base."""
    return CA.DATASETS[ds].family if ds in CA.DATASETS else ds


def kind(ds: str) -> str:
    """``gt``, ``sub``, ``fill`` or ``taxmiss`` for a derived dataset; empty otherwise."""
    return CA.DATASETS[ds].kind if ds in CA.DATASETS else ""


def variant(ds: str) -> tuple[str, str] | None:
    """``(column, value)`` picking a derived dataset's own rows out of a landed CSV shared with its
    siblings, keyed on (phase, rep, arm): a 500-tree draw's ``("draw", "<k>")`` in
    ``<fam>_sub500.csv``, a missing-taxa condition's ``("cond", "<cond>")`` in
    ``taxmiss_eval/results/<fam>_taxmiss.csv``.  ``None`` for every other dataset."""
    k = kind(ds)
    if k == "sub":
        return "draw", str(CA.DATASETS[ds].extra["draw"])
    if k == "taxmiss":
        return "cond", str(CA.DATASETS[ds].extra["cond"])
    return None


def headline(ds: str) -> str:
    """``relaxed_L2``, or ``relaxed_L3`` on l3reg."""
    return CA.DATASETS[ds].headline


def thr_tag(thr: float) -> str:
    """The threshold's spelling in every cache key: ``z6``, ``z5.5``.  The threshold changes the
    blocks, so it is part of the key (SPEC-AMENDMENTS A52's lesson)."""
    return f"z{thr:g}"


def cred_tag(thr: float, rho: int) -> str:
    """``z6_rho2``: the radius changes every oversized block's distribution, so it is in the key."""
    return f"{thr_tag(thr)}_rho{rho}"


def cells(ds: str, phases: str = "", reps: str = "", arms: str = "",
          cells_file: Path | None = None) -> list[dict]:
    """CELLS rows, each with ``name``, ``group`` (the network: a fold never splits one) and the
    stratum fields cluster_anatomy's tables carry."""
    D = CA.DATASETS[ds]
    out = []
    for r in CA.select(D, phases=phases, reps=reps, arms=arms, cells_file=cells_file):
        out.append({**r, "dataset": ds, "name": CA.cell_name(r["phase"], r["rep"], r["arm"]),
                    "group": D.group(r), **D.stratum(r)})
    return out


#: Where the scored cells are read from.  Only for smokes: a scratch ``score_cells --out``.
ANAT_RUNS = Path(os.environ.get("NE_ANAT_RUNS", ANAT / "runs"))


def cell_paths(ds: str, name: str) -> tuple[Path, Path]:
    d = ANAT_RUNS / (CA.DATASETS[ds].anat or ds) / "cells"
    return d / f"{name}.npz", d / f"{name}.json"


#: subsample_eval-shaped records for the EXTRA datasets.  Registering them in
#: ``subsample_eval/common.py`` instead would put them in its stage driver's default list.
SC_EXTRA = {"n1000": SC.Dataset("n1000", REPO / "n1000_eval" / "inputs", REPO / "n1000_eval" / "CELLS.tsv",
                                2, 0, SC.LANDED / "l2_n1000_gt50k_v1_per_network.csv", {},
                                stage=REPO / "n1000_eval" / "stage.json")}


@lru_cache(maxsize=None)
def sc(ds: str):
    """``subsample_eval``'s record for a dataset: its own registry, else :data:`SC_EXTRA`.

    A derived dataset gets a record named after its **base**, because ``make_sub_csv.load_cells`` /
    ``load_truths`` dispatch on ``.name``, with the derived inputs as ``source`` (the truths are
    read from the files ``score_cells`` scored against) and its own landed rows as ``landed_csv``.
    Only ``.name``, ``.source``, ``.cells``, ``.stage`` and ``.landed_csv`` are read here; the
    record's draw paths (``run_dir``, ``draw_inputs``, ``manifest``) are not meant for it."""
    if ds in SC.DATASETS:
        return SC.DATASETS[ds]
    if ds in SC_EXTRA:
        return SC_EXTRA[ds]
    D = CA.DATASETS[ds]
    P = SC.DATASETS[D.family]
    if D.kind == "fill":
        return P
    if D.kind == "sub":
        return dataclasses.replace(P, source=D.inputs, landed_csv=D.extra["landed_csv"])
    if D.kind == "taxmiss":
        return dataclasses.replace(P, source=D.inputs, cells=D.cells, landed_csv=D.extra["landed_csv"])
    return SC.Dataset(D.family, D.inputs, D.cells, P.level, 0, D.extra["landed_csv"], {}, stage=D.stage)


def mask_of(row: np.ndarray) -> int:
    """``cluster_anatomy/scoring.mask_of``: bit ``t`` of the integer is taxon ``t``."""
    return int.from_bytes(np.ascontiguousarray(row).tobytes(), "little")


def masks_of(words: np.ndarray) -> list[int]:
    return [mask_of(words[i]) for i in range(words.shape[0])]


def load_cell(ds: str, name: str) -> dict:
    """One scored cell, no torch: the candidate words, every logit, the headline selection and the
    truth, all as ``cluster_anatomy/score_cells.py`` cached them."""
    npz_path, json_path = cell_paths(ds, name)
    meta = json.loads(json_path.read_text())
    arr = np.load(npz_path)
    hd = headline(ds)
    key = f"gnn__{hd}__sel"
    return {"meta": meta, "words": arr["words"], "z": arr["logits"].astype(np.float64),
            "p": arr["p"], "support": arr["support"].astype(np.int64), "sel": arr[key],
            "true_words": arr["true_words"], "true_row": arr["true_row"],
            "decode": meta["decodes"][f"gnn|{hd}"], "headline": hd}


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, default=_jsonable))
    tmp.replace(path)


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def hexmask(m: int) -> str:
    return format(m, "x")


def unhex(s: str) -> int:
    return int(s, 16)
