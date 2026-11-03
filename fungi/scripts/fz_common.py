"""fungi: paths, constants and loaders shared by every ``fz_*`` script -- see ``fungi/README.md``.

Every script here imports this module **first**.  It puts ``nni_ensemble/`` on ``sys.path`` and imports
``ne_common`` before anything else, which pins BLAS to one thread, puts ``gnn_cpu_v1/pylib`` at
``sys.path[0]`` (with ``gnn_cpu_v1/`` behind it) and raises the recursion limit -- the import order
``nni_ensemble`` itself relies on.  So every module name here carries the ``fz_`` prefix: ``gnn_cpu_v1/``
ships ``ensemble.py``, ``rooting.py`` and ``run_gnn_cpu.py``, and a bare name would collide.

Nothing here imports torch (asserted at the bottom).  ``infer_gnn_cpu`` sniffs ``--threads`` and
``--opc`` out of ``sys.argv`` at import and allows abbreviations, so no script here may define an
option that abbreviates to either; every parser is built with ``allow_abbrev=False``.

The inputs are Stage A's ``infer_gnn_cpu.py --annotate`` outputs under ``analysis/gnn/<scheme>/``:

    fungi_<scheme>_tau<T>_<dec>.{tsv,znet.tsv,clusters.tsv,logits.npz,manifest.json,log}

with ``<dec>`` in ``strictL1`` (``--decoder level1``) / ``L2`` (``--decoder relaxed --max-level 2``).
Stage B (the z+tc ensemble) reads the ``L2`` runs only: nni_ensemble's block lemma is the relaxed
reading of a selection, and the level-1 closure can add clusters the selection does not hold.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUNGI = HERE.parent
REPO = FUNGI.parent
NNI = REPO / "nni_ensemble"
GNN = REPO / "gnn_cpu_v1"

if str(NNI) not in sys.path:
    sys.path.insert(0, str(NNI))

import ne_common as NE  # noqa: E402  -- first: BLAS pin, pylib at sys.path[0], ca_common

import numpy as np  # noqa: E402

import fz_paths as P  # noqa: E402  -- the landed tree, or FZ_VARIANT's (stdlib only)

#: ``fungi/data`` / ``fungi/analysis`` for the landed run; ``fungi/<variant>/{data,analysis}`` under
#: ``FZ_VARIANT`` (``fz_paths``).  Nothing below names a path any other way.
DATA = P.DATA
ANALYSIS = P.ANALYSIS
P.banner()
SCHEMES = ("short", "medium")
TAUS = (2, 4, 8)
KS = tuple(range(1, 30))
#: ``--partial``: 75% of the 646 taxa, bird10k's cont75p rule (273 of 363).
PARTIAL = 485
N_TAXA = 646
DECODERS = {"strictL1": "level1", "L2": "relaxed_L2"}
HEADLINE_DEC = "L2"
LEVEL = 2
MODEL = GNN / "checkpoints" / "_l2mix_pair" / "s0" / "best.pt"
#: The checkpoint every nni_ensemble calibration was fitted on (bird10k v2's manifests carry it too).
MODEL_SHA = "ef5f06b4e8ff6fbab988c157cd0aca573a7ffa70681b1e04eed6730f40662751"
#: The 14 rare taxa every scheme's trees root on when they carry them (155 / 99 of the trees that do).
OUTGROUP = ("GCA_022605425_2", "GCA_024399255_1", "GCA_024399295_1", "GCA_027571585_1",
            "GCA_029215505_1", "GCA_051312775_1", "GCA_051312795_1", "GCA_051312815_1",
            "GCA_056153265_1", "GCF_000091225_2", "GCF_000146465_1", "GCF_000277815_2",
            "GCF_000280035_1", "GCF_036630325_1")
CALIB = NE.RUNS / "calib"
#: ``FZ_BETA=train``: ``nni_ensemble/trainfit.py``'s beta, fitted on the checkpoint's own training networks.
CALIB_TRAIN = NNI / "runs_trainfit" / "calib"
#: ``sim`` (headline) and ``netcs`` for the published betas; ``train`` alone under ``FZ_BETA=train``.
POOLS = P.MAP_POOLS
VARIANTS = ("z+tc", "z")

#: Stage B's hard caps.  Each raises :class:`CapExceeded` naming the cell and block; the cell writes
#: nothing and the driver exits non-zero.  ``A_CAP`` is provisional until ``fz_scan`` has measured
#: the tail; the others bound memory or time of one step.
CAPS = {
    "A_CAP": 8000,            # options in one block (the compat graph is a^2 bits)
    "M_CAP": 256,             # atoms in one block (the atom tensor and shares are m^3 float64)
    "RAD_CAND_CAP": 20_000_000,   # radius_configs' recursion calls
    "RAD_CONF_CAP": 200_000,  # configurations listed within the radius
    "BALL_CAP": 5000,         # realisations of d = 1 ball members per cell and variant
}


class CapExceeded(RuntimeError):
    def __init__(self, cell: str, block, what: str, value, cap):
        super().__init__(f"{cell} block {block}: {what} = {value:,} > cap {cap:,}")
        self.cell, self.block, self.what, self.value, self.cap = cell, block, what, value, cap


# ------------------------------------------------------------------------------------- naming

def run_name(scheme: str, tau: int, dec: str) -> str:
    return f"fungi_{scheme}_tau{tau}_{dec}"


def gnn_prefix(gnn_root: Path, scheme: str, tau: int, dec: str) -> Path:
    return Path(gnn_root) / scheme / run_name(scheme, tau, dec)


def cell_id(scheme: str, tau: int, k: int) -> str:
    return f"{scheme}:tau{tau}:k{k}"


def cell_name(tau: int, k: int) -> str:
    return f"tau{tau}_k{k}"


def parse_cell(cid: str) -> tuple[str, int, int]:
    s, t, k = cid.split(":")
    if s not in SCHEMES or not t.startswith("tau") or not k.startswith("k"):
        raise ValueError(f"cell {cid!r}: expected <scheme>:tau<T>:k<K>")
    return s, int(t[3:]), int(k[1:])


def _ints(spec: str, default) -> tuple[int, ...]:
    if not spec:
        return tuple(default)
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return tuple(out)


def select_cells(args) -> list[tuple[str, int, int]]:
    """``--cells a,b`` / ``--cells @file`` win; otherwise the ``--schemes x --taus x --ks x`` grid.
    Heaviest first is the caller's business (``fz_scan`` knows the costs)."""
    if getattr(args, "cells", ""):
        spec = args.cells
        ids = (Path(spec[1:]).read_text().split() if spec.startswith("@") else spec.split(","))
        out = [parse_cell(c) for c in ids if c]
    else:
        schemes = [s for s in (args.schemes.split(",") if args.schemes else SCHEMES) if s]
        out = [(s, t, k) for s in schemes for t in _ints(args.taus, TAUS) for k in _ints(args.ks, KS)]
    if getattr(args, "limit", 0):
        out = out[:args.limit]
    return out


def add_cell_args(ap, jobs: int = 5) -> None:
    ap.add_argument("--schemes", default="", help="comma list; default short,medium")
    ap.add_argument("--taus", default="", help="e.g. 2,4,8")
    ap.add_argument("--ks", default="", help="e.g. 1-29 or 1,5,29")
    ap.add_argument("--cells", default="", help="short:tau2:k29,... or @file (overrides the grid)")
    ap.add_argument("--limit", type=int, default=0, help="first N cells only (smokes)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=jobs)
    ap.add_argument("--gnn-root", type=Path, default=ANALYSIS / "gnn",
                    help="Stage A's outputs (default fungi/analysis/gnn)")
    ap.add_argument("--out-root", type=Path, default=ANALYSIS,
                    help="where trees/, ens/ go (a scratch dir for dry-runs)")
    ap.add_argument("--zthr", type=float, default=NE.Z_THR)


# ------------------------------------------------------------------------------------ hashing

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj, indent=None) -> None:
    """Atomic, numpy-aware (``ne_common.write_json``)."""
    NE.write_json(Path(path), obj, indent=indent)


def write_tsv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    """Rows as a tab-separated table (gzip on ``.gz``), atomically.  ``cols`` defaults to the union of
    keys in first-seen order (``nni_ensemble/make_csv.py:77-95``, tab instead of comma)."""
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(tmp, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(_cell(r.get(c)) for c in cols) + "\n")
    tmp.replace(path)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if v != v else f"{float(v):.6g}"
    return str(v)


# ------------------------------------------------------------------------------- Stage A inputs

def tsv_lines(path: Path) -> dict[int, str]:
    """``k<TAB>eNewick`` -> {k: eNewick}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            k, enw = line.split("\t", 1)
            out[int(k)] = enw
    return out


_RUNS: dict = {}


def load_run(gnn_root: Path, scheme: str, tau: int, dec: str = HEADLINE_DEC) -> dict:
    """One Stage A job's annotation, cached per process: npz arrays, manifest, the eNewick per k."""
    key = (str(gnn_root), scheme, tau, dec)
    if key in _RUNS:
        return _RUNS[key]
    pre = gnn_prefix(gnn_root, scheme, tau, dec)
    npz_path = pre.with_name(pre.name + ".logits.npz")
    man = read_json(pre.with_name(pre.name + ".manifest.json"))
    arr = np.load(npz_path)
    ks = [int(x) for x in arr["ks"]]
    run = {"scheme": scheme, "tau": tau, "dec": dec, "prefix": pre, "manifest": man,
           "npz_sha": sha256(npz_path), "words": arr["words"],
           "support": arr["support"].astype(np.int64), "taxa": [str(t) for t in arr["taxa_order"]],
           "ks": ks, "z": arr["z"], "sel": {k: arr[f"sel_k{k}"] for k in ks},
           "enewick": tsv_lines(pre.with_name(pre.name + ".tsv")),
           "per_k": {int(r["k"]): r for r in man["per_k"]}}
    _RUNS[key] = run
    return run


def load_cell(gnn_root: Path, scheme: str, tau: int, k: int) -> dict:
    """One relaxed-L2 cell: everything ``nni_ensemble`` read from a sim cell, minus the truth."""
    run = load_run(gnn_root, scheme, tau, HEADLINE_DEC)
    if k not in run["ks"]:
        raise KeyError(f"{cell_id(scheme, tau, k)}: k {k} not in the run's {run['ks']}")
    pk = run["per_k"][k]
    if pk["decoder"] != DECODERS[HEADLINE_DEC]:
        raise SystemExit(f"{run['prefix']}: decoder {pk['decoder']!r}; Stage B reads relaxed_L2 only")
    i = run["ks"].index(k)
    return {"id": cell_id(scheme, tau, k), "scheme": scheme, "tau": tau, "k": k,
            "words": run["words"], "support": run["support"], "taxa": run["taxa"],
            "n": len(run["taxa"]), "m": int(run["manifest"]["m"]),
            "z": run["z"][i].astype(np.float64), "sel": np.asarray(run["sel"][k], dtype=np.int64),
            "enewick": run["enewick"][k], "per_k": pk, "npz_sha": run["npz_sha"],
            "decoder": pk["decoder"]}


def trees_dir(out_root: Path, scheme: str) -> Path:
    return Path(out_root) / "trees" / scheme


def ens_dir(out_root: Path, scheme: str | None = None) -> Path:
    d = Path(out_root) / "ens"
    return d / scheme if scheme else d


def feat_path(out_root: Path, scheme: str, tau: int, k: int, thr: float) -> Path:
    return ens_dir(out_root, scheme) / "feat" / f"{cell_name(tau, k)}__{NE.thr_tag(thr)}.json"


def calib_path(pool: str, thr: float) -> Path:
    return (CALIB_TRAIN if pool == "train" else CALIB) / f"{pool}__{NE.thr_tag(thr)}.json"


def cred_path(out_root: Path, scheme: str, tau: int, k: int, thr: float, rho: int, pool: str) -> Path:
    return (ens_dir(out_root, scheme) / "cred"
            / f"{cell_name(tau, k)}__{NE.cred_tag(thr, rho)}__{pool}.json")


_TRIPLES: dict = {}


def triples_mm(out_root: Path, scheme: str) -> np.ndarray:
    """The scheme's partial-aware triple tensor, memory-mapped once per process (after fork)."""
    key = (str(out_root), scheme)
    if key not in _TRIPLES:
        _TRIPLES[key] = np.load(trees_dir(out_root, scheme) / "triples.u16.npy", mmap_mode="r")
    return _TRIPLES[key]


# ------------------------------------------------------------------------------------- running

def peak_rss_mb() -> float:
    """This process's peak RSS (macOS reports bytes, Linux kilobytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def run_units(units: list, worker, jobs: int, tag: str) -> tuple[list, list]:
    """``worker(unit) -> dict`` over ``units`` in a fork pool; exceptions come back as error dicts
    (a cell's failure never kills the pool), and progress is printed about 20 times."""
    def safe(u):
        try:
            return worker(u)
        except Exception as exc:  # noqa: BLE001 -- reported, and the exit code says so
            return {"unit": str(u), "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc()}

    t0 = time.perf_counter()
    ok, err = [], []
    if jobs > 1 and len(units) > 1:
        global _SAFE
        _SAFE = safe
        pool = get_context("fork").Pool(jobs)
        it = pool.imap_unordered(_call_safe, units)
    else:
        pool, it = None, map(safe, units)
    try:
        for i, r in enumerate(it, 1):
            (err if "error" in r else ok).append(r)
            if i % max(1, len(units) // 20) == 0 or i == len(units):
                print(f"[{tag}]   {i}/{len(units)} | {time.perf_counter() - t0:.0f} s"
                      f"{' | FAILED ' + r['unit'] if 'error' in r else ''}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    for e in err:
        print(f"[{tag}] FAILED {e['unit']}: {e['error']}\n{e['trace']}", file=sys.stderr, flush=True)
    return ok, err


_SAFE = None


def _call_safe(u):
    return _SAFE(u)


if "torch" in sys.modules:
    raise ImportError("fz_common: torch was imported; nothing in fungi/scripts may need it")
