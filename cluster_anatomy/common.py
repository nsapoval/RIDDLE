"""Paths, the dataset registry and the library path for the cluster anatomy study -- see README.md.

Every script here runs under ``gnn_cpu_v1/.venv/bin/python`` and imports this module **first**: it
pins every BLAS to one thread before numpy can be imported, and puts ``gnn_cpu_v1/pylib`` -- the
library ``_l2mix_pair`` was trained against -- at ``sys.path[0]``, with ``gnn_cpu_v1/`` behind it so
``run_gnn_cpu`` (``load_models``, ``forward``, ``M``, ``TAU``) imports as a module.  That is
``l2mix_rank/common.py``'s order, for its reasons.

The benchmarks' own ``common.py`` files are **not imported** (``prefix_eval/common.py``'s rule): each
is a module named ``common``, and loading one would shadow this one.  What is needed from them -- a
condition's key, its stratum, its corpus paths -- is read off their ``CELLS.tsv`` / ``stage.json``,
and the few path rules are restated below with a pointer to the original.

**``run_gnn_cpu`` sniffs ``--threads`` out of ``sys.argv`` when it is imported**
(``run_gnn_cpu.py:114``), with argparse's prefix matching.  No script here may define ``--threads``
or any option it abbreviates (``--t``, ``--th`` ...).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GNN = REPO / "gnn_cpu_v1"
PYLIB = GNN / "pylib"
CHECKPOINTS = GNN / "checkpoints"
RUNS = HERE / "runs"
RESULTS = HERE / "results"
CAMUS_DATASET = REPO / "camus-data" / "camus-data" / "camus-dataset"
NETCS_CORPUS = REPO / "netcs_bench_corpus"

for _p in (GNN, PYLIB):                   # PYLIB last, so it ends up at sys.path[0]
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

#: The headline checkpoint: ``ours/l2mix_pair/selF1`` is ``_l2mix_pair/s0/best.pt``.
TAG, SEED, LABEL = "_l2mix_pair", 0, "ours/l2mix_pair/selF1"
#: The two scorers.  ``gnn`` decodes on the model's float32 ``p``; ``greedy`` on support.
METHODS = ("gnn", "greedy")
METHOD_LABELS = {"gnn": LABEL, "greedy": "greedy"}
#: The tag and seed each method's entry carries in a landed cell JSON.
LANDED_ENTRY = {"gnn": (TAG, SEED), "greedy": ("greedy", 0)}


def pos_weight() -> float:
    """``pos_weight`` from the checkpoint's own training config: every logit is shifted by about
    ``log pos_weight``, so ``sigmoid(z - log pos_weight)`` is the prior-corrected probability."""
    cfg = json.loads((CHECKPOINTS / TAG / f"s{SEED}" / "config.json").read_text())
    return float(cfg["pos_weight"])


# ------------------------------------------------------------------------------------ helpers

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


def cell_name(phase: str, rep, arm: str) -> str:
    """The cell-JSON stem every landed run uses: ``{phase}_rep{RR}_{arm}``."""
    return f"{phase}_rep{int(rep):02d}_{arm}"


def decoder_args(dtag: str) -> tuple[str, int | None]:
    """``level1`` -> ``("level1", None)``; ``relaxed_L2`` -> ``("relaxed", 2)``."""
    if dtag == "level1":
        return "level1", None
    if dtag.startswith("relaxed_L"):
        return "relaxed", int(dtag[len("relaxed_L"):])
    raise ValueError(f"unknown decoder tag {dtag!r}")


def dshort(dtag: str) -> str:
    """The short spelling column names use: ``level1`` -> ``L1f``, ``relaxed_L2`` -> ``rL2``."""
    return "L1f" if dtag == "level1" else "r" + dtag.rsplit("_", 1)[1]


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# ------------------------------------------------------------------------------ the registry

@dataclass(frozen=True)
class Dataset:
    """One staged benchmark.

    ``landed[(method, decoder tag)]`` is the landed run whose cell JSONs a replayed decode must
    reproduce, or ``None`` where none was ever run (camus greedy relaxed L1).  ``prefix_landed`` maps
    a decoder tag to ``prefix_eval``'s run for the same checkpoint -- its truncated-prefix block is a
    decode-*order* target the counts alone cannot give.
    """

    name: str
    title: str                       # the figure prefix the other notebooks use
    inputs: Path
    cells: Path
    decoders: tuple[str, ...]
    headline: str
    landed: dict
    prefix_landed: dict
    m_from_trees: bool = False
    stage: Path | None = None
    shards: Path | None = None
    true_trees_root: Path | None = None
    extra: dict = field(default_factory=dict)
    #: A **derived** dataset (``_derived()``) is one of the four published benchmarks re-run on other
    #: gene trees: ``base`` is that benchmark, whose semantics (stage keying, strata, ILS scale) it
    #: takes; ``kind`` is ``gt`` (more trees), ``sub`` (a 500-tree draw), ``fill`` (the published
    #: inputs, for cells the published tables lack) or ``taxmiss`` (the published trees with taxa
    #: deleted, one condition each); ``parent_phase`` is the base's phase when the
    #: derived CELLS.tsv renames it; ``anat`` reads another dataset's scored cells (fill: the base's).
    base: str | None = None
    kind: str = ""
    parent_phase: str | None = None
    anat: str | None = None

    @property
    def family(self) -> str:
        """The published benchmark whose semantics apply: the dataset itself, or a derived one's base."""
        return self.base or self.name

    def group(self, row: dict) -> str:
        """The network a cell belongs to, as ``nni_ensemble``'s folds group it.  A derived cell takes
        its parent network's group, so it is scored by the fold model that held that network out."""
        return f"{self.family}:{self.parent_phase or row['phase']}:{int(row['rep'])}"

    # -------------------------------------------------------------- cells, strata, scale

    def rows(self, cells_file: Path | None = None) -> list[dict]:
        return read_tsv(Path(cells_file) if cells_file else self.cells)

    @property
    def stage_cells(self) -> dict:
        return _stage_cells(self.family, str(self.stage)) if self.stage else {}

    def stage_cell(self, row: dict) -> dict:
        """This dataset's ``stage.json`` record for a CELLS row (``{}`` on camus, which has none)."""
        if self.family == "netcs":
            return self.stage_cells[(row["phase"], int(row["rep"]), row["arm"])]
        if self.stage:
            return self.stage_cells[int(row["rep"])]
        return {}

    def ils_scale(self, row: dict) -> float:
        """What multiplies the staged truth's lengths for this cell.  ``netcs_eval/stage.py:103`` stages
        the **ils_1.0** network once per truth, and the corpus's ``ils_x`` networks are that one with
        every length multiplied by ``x`` (``check_anatomy`` G6).  Every other dataset: 1."""
        if self.family != "netcs":
            return 1.0
        return float(row["arm"].removeprefix("netcs_ils"))

    def stratum(self, row: dict) -> dict:
        """The per-cell condition columns every table carries, and the ``stratum`` label the
        notebooks group by (``subsample_eval/make_sub_csv.py:131``'s definitions)."""
        n = int(row["n"])
        if self.kind == "train":
            # train_eval's CELLS.tsv carries its own condition columns; there is no stage.json
            return {"stratum": f"{row['phase'].removeprefix('trn_')} {row['split']}",
                    "level": int(row["level"]), "cu": row["cu"], "ils": "", "headline_cell": True,
                    "split": row["split"]}
        if self.kind == "sptree":
            # sptree_eval: species-tree truths (level 0) decoded at a swept budget
            s = self.stage_cell(row)
            return {"stratum": f"cu={s['cu']}", "level": 0, "cu": s["cu"], "ils": "",
                    "headline_cell": True, "size": self.extra["size"],
                    "k_decode": int(self.extra["k_decode"])}
        if self.family == "camus":
            return {"stratum": f"n={row['phase'].removeprefix('cam')}", "level": 1, "cu": "",
                    "ils": "", "headline_cell": row["arm"] == "cam_g500"}
        s = self.stage_cell(row)
        if self.family in ("l2reg", "l3reg", "n1000"):
            level = {"l2reg": 2, "l3reg": 3}.get(self.family, self.extra.get("level"))
            return {"stratum": f"cu={s['cu']}", "level": level,
                    "cu": s["cu"], "ils": "", "headline_cell": True,
                    "n_hi": s.get("n_l2", s.get("n_l3")), "n_l1": s["n_l1"]}
        return {"stratum": f"L{s['level']} ILS {s['ils']}", "level": s["level"], "cu": "",
                "ils": s["ils"], "headline_cell": True, "n_nominal": n,
                "root_exact_frac": s["root_exact_frac"], "root_cost_median": s["root_cost_median"],
                "shortcut_retics": s["shortcut_retics"]}

    # ------------------------------------------------------------------------- truth

    def truth(self, row: dict):
        """The truth ``run_gnn_cpu`` scored against (``camus.truth_of``, OUT pruned, taxa in the
        file's order), with its lengths put on this cell's scale.  Rescaled on a copy, so the
        cached parse is never touched; ``Reticulation.time`` goes stale and nothing here reads it."""
        from opc.rl import camus
        from opc.simulate import networks as NW
        net = camus.truth_of(camus.CamusCell(row["phase"], int(row["rep"]), row["arm"], self.inputs))
        scale = self.ils_scale(row)
        if scale != 1.0:
            net = net.copy()
            NW._rescale(net, scale)  # noqa: SLF001
        return net

    def truth_path(self, row: dict) -> Path:
        from opc.rl import camus
        return camus.CamusCell(row["phase"], int(row["rep"]), row["arm"], self.inputs).base_path

    # ------------------------------------------------------------------- true gene trees

    def true_trees(self, row: dict) -> list[tuple[str, Path, str]]:
        """``[(column, path, rooting)]``: the simulated gene trees behind this cell's estimates.

        ``rooting`` is ``keep_root`` (rooted at the coalescent root; ``OUT`` deleted if present),
        ``outgroup`` (camus only: the same trees re-rooted on ``OUT``, the rooting the estimated arm
        gets) or ``as_written`` (rooted, and no ``OUT`` to delete).
        """
        rep = int(row["rep"])
        if self.base:
            # a derived dataset's true trees are its base's, and check_anatomy's G0 checks those
            return []
        if self.kind == "train":
            # the GNN's own training grids: their true trees are an arm of the cell, not a column
            return []
        if self.kind == "sptree":
            # sptree_eval reads the estimated arm only; its true trees are not staged
            return []
        if self.name == "n1000":
            # l2_n1000_gt_50k_v1 simulated 50,000 true trees per network, and nothing here reads
            # them; listing none keeps check_anatomy's G0 (1,000 trees per file) about the four
            # datasets it was written for.
            return []
        if self.name == "camus":
            from opc.rl import camus
            d = camus.CamusCell(row["phase"], rep, "", CAMUS_DATASET).trees_path
            return [("freq_true", d, "keep_root"), ("freq_true_og", d, "outgroup")]
        if self.name in ("l2reg", "l3reg"):
            key = self.stage_cell(row)["key"]
            return [("freq_true", self.true_trees_root / "trees" / f"{key}.nwk.gz", "as_written")]
        level, n = _netcs_phase(row["phase"])
        ils = row["arm"].removeprefix("netcs_ils")
        d = NETCS_CORPUS / f"LEVEL_{level}" / f"TAXA_{n}" / f"network_{rep}"
        return [("freq_true", d / f"network_{rep}_ils_{ils}.true_gene_trees", "as_written")]


def _netcs_phase(phase: str) -> tuple[int, int]:
    """``netcs_L2_n100`` -> ``(2, 100)`` (``netcs_eval/common.py:PHASES``)."""
    lv, n = phase.removeprefix("netcs_").split("_")
    return int(lv[1:]), int(n[1:])


def netcs_corpus_network(row: dict) -> Path:
    """The corpus's own network for this cell's ILS level (``netcs_eval/common.py:153``)."""
    level, n = _netcs_phase(row["phase"])
    rep, ils = int(row["rep"]), row["arm"].removeprefix("netcs_ils")
    return (NETCS_CORPUS / f"LEVEL_{level}" / f"TAXA_{n}" / f"network_{rep}"
            / f"network_{rep}_ils_{ils}.newick")


@lru_cache(maxsize=None)
def _stage_cells(name: str, path: str) -> dict:
    s = json.loads(Path(path).read_text())
    if name == "netcs":
        return {(c["phase"], int(c["rep"]), c["arm"]): c for c in s["cells"]}
    return {int(c["rep"]): c for c in s["cells"]}


def _gnn_runs(base: Path, dtags) -> dict:
    """``{("gnn", dtag): run}``: the level-1 run is ``base``, each relaxed run sits beside it."""
    out = {}
    for d in dtags:
        out[("gnn", d)] = base if d == "level1" else base.with_name(
            f"{base.name}_relaxedL{decoder_args(d)[1]}")
    return out


def _greedy_runs(base: Path, dtags) -> dict:
    """``{("greedy", dtag): run or None}``, the same naming as :func:`_gnn_runs`."""
    out = {}
    for d in dtags:
        run = base if d == "level1" else base.with_name(f"{base.name}_relaxedL{decoder_args(d)[1]}")
        out[("greedy", d)] = run if run.exists() else None
    return out


def _prefix(ds: str, dtags) -> dict:
    root = REPO / "prefix_eval" / "runs" / ds
    out = {}
    for d in dtags:
        p = root / ("gnn_t1_prefix" if d == "level1" else f"gnn_t1_prefix_relaxedL{decoder_args(d)[1]}")
        if p.exists():
            out[d] = p
    return out


_D3 = ("level1", "relaxed_L1", "relaxed_L2")
_D4 = _D3 + ("relaxed_L3",)

_CAMUS_LANDED = {**_gnn_runs(GNN / "runs" / "C4537CLFY0_t1_l2mix_pair", _D3),
                 # l3reg_eval/run_greedy.py --camus wrote the formal decoder's; subsample_eval's
                 # command sheet the relaxed L2 one (subsample_eval/common.py:189).  Relaxed L1 was
                 # never run on camus, so it has no parity target.
                 ("greedy", "level1"): REPO / "l3reg_eval" / "runs" / "greedy_camus",
                 ("greedy", "relaxed_L1"): None,
                 ("greedy", "relaxed_L2"): REPO / "subsample_eval" / "runs" / "camus" / "full" / "greedy_relaxedL2"}


def _eval(ds: str, dtags) -> dict:
    runs = REPO / f"{ds}_eval" / "runs"
    return {**_gnn_runs(runs / "gnn_t1_l2mix_pair", dtags), **_greedy_runs(runs / "greedy", dtags)}


DATASETS = {
    "camus": Dataset("camus", "CAMUS", GNN / "inputs", GNN / "CELLS.tsv", _D3, "relaxed_L2",
                     _CAMUS_LANDED, _prefix("camus", _D3)),
    "l2reg": Dataset("l2reg", "L2reg", REPO / "l2reg_eval" / "inputs", REPO / "l2reg_eval" / "CELLS.tsv",
                     _D3, "relaxed_L2", _eval("l2reg", _D3), _prefix("l2reg", _D3),
                     stage=REPO / "l2reg_eval" / "stage.json", true_trees_root=REPO / "l2reg_n200_v1"),
    "l3reg": Dataset("l3reg", "L3reg", REPO / "l3reg_eval" / "inputs", REPO / "l3reg_eval" / "CELLS.tsv",
                     _D4, "relaxed_L3", _eval("l3reg", _D4), _prefix("l3reg", _D4),
                     stage=REPO / "l3reg_eval" / "stage.json", true_trees_root=REPO / "l3reg_n200_v1"),
    "netcs": Dataset("netcs", "NetCS", REPO / "netcs_eval" / "inputs", REPO / "netcs_eval" / "CELLS.tsv",
                     _D3, "relaxed_L2", _eval("netcs", _D3), _prefix("netcs", _D3),
                     # netcs_bench_corpus ships 911-1,000 trees per condition, and the landed rows
                     # ran with --m-from-trees (netcs_eval/README.md).
                     m_from_trees=True, stage=REPO / "netcs_eval" / "stage.json",
                     shards=REPO / "netcs_eval" / "shards"),
    # 20 level-2 networks at n = 1000 with 10,000 IQ-TREE trees each (n1000_eval/README.md).  The
    # landed GNN ran with --m 0 --tau-frac 0.002 (m = 10,000, tau = 20) and the relaxed L2 decoder
    # only, so the run directories are spelled out rather than derived by _eval().  ``extra`` carries
    # what the other datasets take from run_gnn_cpu's M and TAU, and nni_ensemble's triple kernel:
    # the dense n^3 tensor is 4 GB here.
    "n1000": Dataset("n1000", "L2n1000", REPO / "n1000_eval" / "inputs", REPO / "n1000_eval" / "CELLS.tsv",
                     ("relaxed_L2",), "relaxed_L2",
                     {("gnn", "relaxed_L2"): REPO / "n1000_eval" / "runs" / "gnn_t1_l2mix_pair_ftau0.002_relaxedL2",
                      ("greedy", "relaxed_L2"): REPO / "n1000_eval" / "runs" / "greedy_relaxedL2"},
                     {}, stage=REPO / "n1000_eval" / "stage.json", true_trees_root=REPO / "l2_n1000_gt_50k_v1",
                     extra={"m": 10000, "tau": 20, "level": 2, "tc_kernel": "block"}),
}

SUB_EVAL = REPO / "subsample_eval"
TAXMISS = REPO / "taxmiss_eval"
#: ``taxmiss_eval/tm_common.PARTIAL``: run_gnn_cpu's ``--partial MIN`` on those datasets.
TAXMISS_PARTIAL = 4
LANDED_RL = REPO / "results" / "rl"
#: The cells the published nni_ensemble tables lack (README, Traps: gate N3's fixture), one CELLS.tsv
#: per base, rows copied verbatim.  camus and l3reg are header-only: their fill entries exist for the
#: parity smokes, which pass the base's own CELLS.tsv as ``--cells``.
FILL_CELLS = REPO / "nni_ensemble" / "fill"


def _derived() -> dict:
    """The four published benchmarks re-run on other gene trees (``nni_ensemble/README.md``, Derived
    datasets).  Each is scored against its own landed GNN run, decodes the headline only, and writes
    under its own name.  Triples are counted with the block kernel (dense's equal, gate G-TC), except
    on netcs, which keeps the dense one its published run used: the block kernel's chunk holds
    ``trees x m_b^3`` and a 54-atom NetCS block at ILS 2.0 took it to 7.2 GB, where dense is n^3
    int32 (33 MB at n = 200) -- measured on netcs_L1_n200 rep21 ils2.0's 500-tree draw, 2026-10-21.

    * ``<fam>_gt5k`` / ``<fam>_gt10k``: l2reg / l3reg at 5,000 / 10,000 estimated trees
      (``l2reg_eval/README.md``, Gene-tree scaling), ``--m 0 --tau-frac 0.002``.
    * ``<fam>_sub500_d<k>``: subsample_eval's 500-tree draw ``k``, ``--m-from-trees``, tau 2.  The
      parent's CELLS.tsv and stage.json: ``subsample_eval/inputs/<fam>/stage.json`` is a staging
      record, not per-cell strata.
    * ``<fam>_fill``: the base itself, on the cells its published tables lack; it reads the base's
      published scored cells, so it has nothing to score.
    * ``<fam>_tm_<cond>``: ``taxmiss_eval``'s missing-taxa condition ``cond`` -- one per
      ``taxmiss_eval/staged/<fam>/<cond>/CELLS.tsv`` that exists, so a condition is registered once
      it is staged (``taxmiss_eval/tm_common.dataset_name``'s rule).  Partial trees
      (``extra["partial"]``), the base's m, its kernel, ``taxmiss_eval/runs/<fam>/<cond>/`` as the
      landed GNN run and ``taxmiss_eval/results/<fam>_taxmiss.csv`` as the landed rows.  No greedy
      run exists, so greedy decodes have no parity target.
    """
    out = {}
    kernel = {"camus": "block", "l2reg": "block", "l3reg": "block", "netcs": "dense"}
    for fam in ("l2reg", "l3reg"):
        P = DATASETS[fam]
        L = decoder_args(P.headline)[1]
        for gt, m, tau in (("gt5k", 5000, 10), ("gt10k", 10000, 20)):
            root, name = REPO / f"{fam}_eval" / gt, f"{fam}_{gt}"
            out[name] = Dataset(
                name, f"{P.title}-{gt}", root / "inputs", root / "CELLS.tsv", (P.headline,), P.headline,
                {("gnn", P.headline): root / "runs" / f"gnn_t1_l2mix_pair_ftau0.002_relaxedL{L}",
                 ("greedy", P.headline): root / "runs" / f"greedy_relaxedL{L}"}, {},
                stage=root / "stage.json", base=fam, kind="gt", parent_phase=f"{fam}_n200",
                extra={"m": m, "tau": tau, "tc_kernel": kernel[fam],
                       "landed_csv": LANDED_RL / f"{fam}_n200_{gt}_v1_per_network.csv"})
    for fam, draws in (("camus", 5), ("l2reg", 5), ("l3reg", 5), ("netcs", 1)):
        P = DATASETS[fam]
        L = decoder_args(P.headline)[1]
        for k in range(draws):
            runs, name = SUB_EVAL / "runs" / fam / f"d{k}", f"{fam}_sub500_d{k}"
            out[name] = Dataset(
                name, f"{P.title}-sub500-d{k}", SUB_EVAL / "inputs" / fam / f"d{k}", P.cells,
                (P.headline,), P.headline,
                {("gnn", P.headline): runs / f"gnn_t1_l2mix_pair_relaxedL{L}",
                 ("greedy", P.headline): runs / f"greedy_relaxedL{L}"}, {},
                m_from_trees=True, stage=P.stage, shards=P.shards, base=fam, kind="sub",
                extra={"tc_kernel": kernel[fam], "draw": k,
                       "landed_csv": SUB_EVAL / "results" / f"{fam}_sub500.csv"})
    for fam in ("camus", "l2reg", "l3reg", "netcs"):
        P = DATASETS[fam]
        L = decoder_args(P.headline)[1]
        for cells in sorted((TAXMISS / "staged" / fam).glob("*/CELLS.tsv")):
            cond = cells.parent.name
            name = f"{fam}_tm_{cond}"
            out[name] = Dataset(
                name, f"{P.title}-taxmiss-{cond}", cells.parent, cells, (P.headline,), P.headline,
                {("gnn", P.headline): TAXMISS / "runs" / fam / cond / f"gnn_t1_l2mix_pair_relaxedL{L}",
                 ("greedy", P.headline): None}, {},
                m_from_trees=P.m_from_trees, stage=P.stage, shards=P.shards, base=fam, kind="taxmiss",
                extra={"tc_kernel": kernel[fam], "cond": cond, "partial": TAXMISS_PARTIAL,
                       "landed_csv": TAXMISS / "results" / f"{fam}_taxmiss.csv"})
    for fam in ("camus", "l2reg", "l3reg", "netcs"):
        P = DATASETS[fam]
        out[f"{fam}_fill"] = replace(
            P, name=f"{fam}_fill", title=f"{P.title}-fill", cells=FILL_CELLS / f"{fam}_CELLS.tsv",
            decoders=(P.headline,), prefix_landed={}, extra={**P.extra, "tc_kernel": kernel[fam]},
            base=fam, kind="fill", anat=fam)
    return out


DATASETS.update(_derived())

TRAIN_EVAL = REPO / "train_eval"


def _train() -> dict:
    """The checkpoint's own training and validation networks (``train_eval/README.md``): phases
    ``trn_{x25,x50,y25,y50,l2t25,l2t50}``, arms ``trn_true`` / ``trn_est_bp1000``, reps 0-799 --
    what ``_l2mix_pair/s0/config.json`` trained (``< 700``) and selected (``< 800``) on.  They exist
    to fit ``nni_ensemble``'s choice model *off* the benchmarks, so ``kind = "train"`` and no
    ``base``: no calibration pool, fold or derived-dataset rule reaches them.  No landed run, so
    parity records ``ok: None``.  ``level1`` is decoded beside the headline only to read the val F1
    against ``verdict.json``, whose ``val_decoder: auto`` is level 1 on x/y.  Registered once
    ``train_eval/CELLS.tsv`` exists, as the taxmiss conditions are (check_anatomy's G0 reads every
    registered dataset's rows)."""
    cells = TRAIN_EVAL / "CELLS.tsv"
    if not cells.exists():
        return {}
    return {"trn": Dataset("trn", "Train", TRAIN_EVAL / "inputs", cells, ("relaxed_L2", "level1"),
                           "relaxed_L2", {}, {}, kind="train",
                           extra={"m": 1000, "tau": 2, "tc_kernel": "dense"})}


DATASETS.update(_train())

SPTREE_EVAL = REPO / "sptree_eval"
#: ``sptree_eval``'s gene-tree prefixes: (tag, m, tau) -- the first m estimated trees, tau = 0.002 m
#: as in the landed gene-tree scaling runs (``l2reg_eval/README.md``).
SPTREE_SIZES = (("1k", 1000, 2), ("5k", 5000, 10), ("10k", 10000, 20))
#: The swept decode budgets.
SPTREE_KS = tuple(range(11))


def sptree_name(size: str, k: int) -> str:
    """``sp1k_k00`` .. ``sp10k_k10``."""
    return f"sp{size}_k{int(k):02d}"


def _sptree() -> dict:
    """``sptree_n200_gt10k_v1`` (``sptree_eval/README.md``): 50 species trees at n = 200 -- truths with
    no reticulation -- each with 10,000 IQ-TREE trees, decoded by the GNN at every budget ``k`` in
    :data:`SPTREE_KS` on the first 1k / 5k / 10k trees.  One dataset per (size, k), so a cell's stem
    stays ``sptree_n200_repRR_est_bp1000`` and every reader keyed on it is unchanged; the k-datasets
    of a size share inputs, CELLS and stage and differ only in ``extra["k_decode"]``, which
    ``scoring.build_cell`` reads in place of the truth's ``k``.  ``relaxed_L1`` is the headline (the
    MAP runs on it); ``level1`` is decoded beside it from the same forward.  No landed run, so parity
    records ``ok: None``; ``kind = "sptree"`` and no ``base``, so no calibration pool, fold or derived
    rule reaches them -- the MAP is ``credible --calib-pool train``'s transfer.  Registered once
    ``sptree_eval/CELLS.tsv`` exists."""
    cells = SPTREE_EVAL / "CELLS.tsv"
    if not cells.exists():
        return {}
    out = {}
    for tag, m, tau in SPTREE_SIZES:
        for k in SPTREE_KS:
            name = sptree_name(tag, k)
            out[name] = Dataset(name, f"SpTree-{tag}-k{k}", SPTREE_EVAL / "inputs", cells,
                                ("relaxed_L1", "level1"), "relaxed_L1", {}, {},
                                stage=SPTREE_EVAL / "stage.json", kind="sptree",
                                extra={"m": m, "tau": tau, "k_decode": k, "tc_kernel": "block",
                                       "size": tag})
    return out


DATASETS.update(_sptree())


def select(ds: Dataset, *, phases: str = "", reps: str = "", cells_file: Path | None = None,
           arms: str = "") -> list[dict]:
    """The CELLS rows a driver runs on: ``--cells`` (a shard file) replaces the dataset's CELLS.tsv,
    and ``--phases`` / ``--reps`` / ``--arms`` filter either."""
    want_p = {p for p in phases.split(",") if p}
    want_a = {a for a in arms.split(",") if a}
    want_r = parse_reps(reps)
    return [c for c in ds.rows(cells_file)
            if (not want_p or c["phase"] in want_p) and (not want_a or c["arm"] in want_a)
            and (want_r is None or int(c["rep"]) in want_r)]
