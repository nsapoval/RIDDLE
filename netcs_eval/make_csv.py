#!/usr/bin/env python3
"""The netcs_bench_corpus per-network CSV, in ``camus_a104_per_network.csv``'s shape -- and its parity gates.

One row per (condition, method[, seed]), a condition being (level, n, network, ILS):

    ours/prior, ours/2x-data-full, ours/+n75   GNN runs (runs/gnn*/), seeds 0-2; ours/2x-data too
                                               when a _mix2 run exists
    ours/l2mix, ours/l2mix75                   GNN _l2mix seeds 0-1, _l2mix75 seed 0
    greedy                                     greedy runs (runs/greedy*/)
    TOB-QMC, TOB-QMC-fast, TINNIK, true TOB*   the corpus's trees of blobs, read as trees (k_pred = 0);
                                               TINNIK exists at n <= 100 only
    NetCS, BROOQS                              resolving TOB-QMC default, native and unrooted

and every GNN and greedy label again with `` [relaxed L1]`` and `` [relaxed L2]``.  A complete build
is 1,200 x (48 GNN + 3 greedy + 3 TOB + 2 resolver) + 800 TINNIK = **68,000 rows** -- 16 checkpoints
x 3 decoders since 2026-09-15, ``_l2mixha``'s and ``_l2mix_pair``'s ``best.pt`` and ``best_ha.pt``
each counted separately (``ours/l2mixHA/selF1``/``selHA``, ``ours/l2mix_pair/selF1``/``selHA``);
the required models are read off the checkpoint *files* under ``gnn_cpu_v1/checkpoints/``.

Columns: camus_a104's eleven in its order -- ``n`` is the corpus's nominal 50/100/200 and
``network_idx`` the corpus network id -- then ``decoder, key, phase, arm, level, ils, n_taxa,
n_loci, shortcut_retics, trees_root_exact_frac, root_cost, n_blobs_tob, tp, fp, fn, precision,
recall, mass_pred, mass_truth, cluster_f1_rooted, status``, then hybrid ancestry's ``ha_n_true,
ha_n_pred, ha_fp, ha_fn, ha_fpr, ha_fnr, ha_balanced_error``.  Empty fields are ``''``.

Rooting -- read this before comparing rows
------------------------------------------
The corpus has no outgroup.  **The GNN and greedy rows are handed the root**: ``stage.py`` put OUT on
each gene tree's edge nearest the true root (``common.py``), and ``trees_root_exact_frac`` says on
what fraction of that condition's loci that edge was a displayed true root exactly.  NetCS and
BROOQS were given nothing: their outputs, and the trees of blobs, are rooted by the same oracle
**only to be scored**, and ``root_cost`` is how many taxa that edge is from a displayed root.  So
``cluster_f1`` and ``luay`` on a resolver or TOB row are an upper envelope of the method's rooted
accuracy, and on a GNN or greedy row include the benefit of a root nobody else had.  Hybrid
ancestry needs no root, and is the one column that compares all rows on equal terms.

Cluster F1
----------
The landed definitions, one per source: the GNN's own ``Instance.cluster_counts`` on its selection,
the same for greedy, and ``camus_nets.cluster_counts`` over the OUT-restricted clusters for
everything else.  A resolver network the oracle's OUT cannot root (``status = out_not_rootable``)
takes ``cluster_f1`` as emitted and leaves ``cluster_f1_rooted`` empty, as camus_a104 did.

Luay: PhyloNet and nothing else
-------------------------------
Every distance and every network size comes from PhyloNet's ``CMPNETS -m luay``; ``opc.mpl.netdist``
is not imported (``l3reg_eval/make_csv.py`` says why).  Sizes by two star probes per network, so
``luay_norm_exact`` is at most 1 by construction; ``luay_norm`` is camus_a104's closed form, on
the condition's actual taxon count.

* **The truth is the corpus's own string** (``inputs/.../true_net.phylonet.nwk``: unary root removed,
  gammas stripped), never our writer's: PhyloNet reads the two double-donor truths as written,
  which ``Network`` cannot hold, and ``to_enewick`` cannot round-trip the two size-2-cycle truths.
  **Gate:** on every other truth, luay(``to_enewick(truth)``, the corpus string) must be 0 -- the
  staged network and the corpus are then one network as the jar sees it.
* **Cached and parallel.**  ~46 k distances and ~90 k size probes would take hours in one JVM.  Each
  (string, string) pair is keyed by its SHA-256 in ``runs/phylonet/cache.jsonl``, pairs PhyloNet
  fails on are never cached, and ``--jobs`` JVMs run ``--chunk`` pairs each.  A rebuild pays only for
  networks it has not seen.

Hybrid ancestry
---------------
Dai & Molloy, bioRxiv 10.64898/2026.06.06.730607, Fig. 3 -- the metric their NetCS paper scores this
corpus with.  Exactly ``l3reg_eval/make_csv.py``'s definition and two-derivation cross-check; the
truth's string derivation reads the corpus string.  A tree of blobs has no hybrids, so its
``ha_fpr`` and ``ha_balanced_error`` are empty and its ``ha_fnr`` is 1.

--parity
--------
``l3reg_eval/make_csv.py --parity``, unchanged: gates (a), (b) and (c) against the landed **camus**
rows, none of which touch this corpus -- they hold this machine's jar and java to the landed luay
path.  (c) borrows ``l3reg_eval/runs/greedy_camus`` rather than re-running it.  ``build`` embeds
``runs/parity.json`` in the summary.

Usage (``gnn_cpu_v1/.venv/bin/python``)
-----
    python netcs_eval/make_csv.py --parity
    python netcs_eval/make_csv.py --jobs 6
    python netcs_eval/make_csv.py --keys L2_n50_net1_ils1.0 --allow-partial --out-csv /tmp/x.csv   # a smoke

``--phylonet`` defaults to ``$OPC_PHYLONET``, else ``~/research/stochastic-search/tools/PhyloNet.jar``;
the system Java 8 runs it (``phylonet.JAVA``, ``$OPC_JAVA``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common as C

import pyarrow.parquet as pq  # noqa: E402

from opc.mpl import phylonet as PN  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl.external import _prune_outgroup  # noqa: E402
from opc.simulate import networks as N  # noqa: E402

CAMUS_COLUMNS = ("method", "gene_trees", "n", "network_idx", "seed", "cluster_f1", "luay",
                 "luay_norm", "luay_norm_exact", "k_pred", "k_true")
#: Hybrid ancestry (module docstring): appended, so every column before it stays where it was.
HA_COLUMNS = ("ha_n_true", "ha_n_pred", "ha_fp", "ha_fn", "ha_fpr", "ha_fnr", "ha_balanced_error")
EXTRA_COLUMNS = ("decoder", "key", "phase", "arm", "level", "ils", "n_taxa", "n_loci",
                 "shortcut_retics", "trees_root_exact_frac", "root_cost", "n_blobs_tob", "tp", "fp",
                 "fn", "precision", "recall", "mass_pred", "mass_truth", "cluster_f1_rooted",
                 "status") + HA_COLUMNS
METHOD_ORDER = ("TOB-QMC", "TOB-QMC-fast", "TINNIK", "true TOB*", "greedy", "ours/prior",
                "ours/2x-data", "ours/2x-data-full", "ours/+n75", "ours/l2mix", "ours/l2mix75",
                "ours/l2mixHA/selF1", "ours/l2mixHA/selHA", "ours/l2mix_pair/selF1",
                "ours/l2mix_pair/selHA", "NetCS", "BROOQS")
#: Checkpoint tag -> the seeds it must have, read off ``gnn_cpu_v1/checkpoints/<tag>/s*/``
#: (``_l2mix`` ships one seed).  A tag with no checkpoint directory falls back to ``SEEDS``.
SEEDS = (0, 1, 2)
def _required_seeds(root: Path) -> dict[str, tuple[int, ...]]:
    """Model tag -> seeds, one entry per checkpoint *file*: ``<tag>/s<N>/best.pt`` is ``<tag>``, and
    any other ``best_<x>.pt`` beside it is ``<tag>:best_<x>`` -- ``run_gnn_cpu.py --models``' naming,
    so ``_l2mixha/s0/best_ha.pt`` (2026-09-14) is required as a model in its own right."""
    out: dict[str, set[int]] = {}
    for d in sorted(root.glob("_*")):
        for s in sorted(d.glob("s*")):
            if s.is_dir() and s.name[1:].isdigit():
                for f in sorted(s.glob("best*.pt")):
                    key = d.name if f.stem == "best" else f"{d.name}:{f.stem}"
                    out.setdefault(key, set()).add(int(s.name[1:]))
    return {k: tuple(sorted(v)) for k, v in sorted(out.items())}


REQUIRED_SEEDS = _required_seeds(C.GNN / "checkpoints")
REQUIRED_TAGS = tuple(sorted(REQUIRED_SEEDS) or ("_mix", "_mix2L", "_mix75"))
#: Every decoder a complete build carries (user decision, 2026-09-12).
REQUIRED_DECODERS = ("level1", "relaxed_L1", "relaxed_L2")
COUNTS = ("tp", "fp", "fn", "precision", "recall")
#: camus cell arm -> the suffix in ``rl_gnn_eval_cam{N}{suffix}_...`` file names.
LANDED_ARM = {"cam_g500": "g500", "cam_iq500": "iq500"}
MAC_GNN_RUN = C.GNN / "runs" / "C4537CLFY0_t1"
MAC_BLOB_RUN = C.BLOB / "runs" / "C4537CLFY0_j8"
L3REG_GREEDY_CAMUS = C.REPO / "l3reg_eval" / "runs" / "greedy_camus"
DEFAULT_JAR = Path.home() / "research" / "stochastic-search" / "tools" / "PhyloNet.jar"
#: A leaf label: a name right after "(" or ",".  Never a "#H" reference or an internal label.
_LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")
#: What closes a hybrid's body: an optional internal label, then the ``#Hk`` tag.
_BODY_TAG = re.compile(r"[^(),:;#\s\[\]]*#[A-Za-z]*\d+")
#: A leaf that is itself a hybrid: ``label#Hk``.
_LEAF_HYBRID = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)#[A-Za-z]*\d+")
_FIELDS = ("luay", "mass_pred", "mass_truth")


def missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pruned(rooted: str) -> str:
    """An OUT-rooted network, OUT pruned and gammas stripped: what the landed luay compared."""
    return C.strip_gamma(N.to_enewick(_prune_outgroup(N.from_enewick(rooted))))


def leaf_set(enewick: str) -> tuple[str, ...]:
    return tuple(sorted(set(_LEAF.findall(enewick))))


def closed_form_max(n: int, k1: int, k2: int) -> int:
    """``n - 1 + floor(3 (k1 + k2) / 2)``: camus_a104's ``luay_norm`` denominator."""
    return max(0, n - 1 + (3 * (k1 + k2)) // 2)


def variant_rank(tag: str) -> int:
    """``level1`` first, then relaxed variants by level cap."""
    return 0 if tag in ("", "level1") else int(tag.rsplit("_L", 1)[1])


def row_order(row: dict, order: dict) -> tuple:
    return (order.get(row["_base"], len(order)), variant_rank(row["decoder"]))


def pair_key(a: str, b: str) -> str:
    return hashlib.sha256(f"{a}\n{b}".encode()).hexdigest()[:40]


class PhyloNetLuay:
    """Every luay and every network size, from PhyloNet alone (module docstring).

    ``cache`` is a JSONL file of ``{"k": pair_key, "luay": int}``; ``None`` measures everything
    afresh, which is what ``--parity`` wants.  ``jobs`` JVMs run ``chunk`` pairs each, every chunk
    in its own work directory, removed once read.
    """

    def __init__(self, jar: Path, cache: Path | None = None, jobs: int = 1, chunk: int = 2000):
        self.jar, self.cache, self.jobs, self.chunk = jar, cache, max(1, jobs), max(1, chunk)
        self.pairs: list[tuple[str, str]] = []

    def add(self, pred: str, truth: str) -> int:
        self.pairs.append((pred, truth))
        return len(self.pairs) - 1

    def _load(self) -> dict[str, int]:
        known: dict[str, int] = {}
        if self.cache is not None and self.cache.exists():
            with open(self.cache) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:          # a torn last line from an interrupted run
                        continue
                    known[r["k"]] = int(r["luay"])
        return known

    def resolve(self, workdir: Path) -> None:
        PN.PHYLONET = Path(self.jar)                  # read at call time by ``available()``
        if not PN.available():
            raise SystemExit(f"PhyloNet not usable: jar {PN.PHYLONET}, java {PN.JAVA}")
        nets = list(dict.fromkeys(s for pair in self.pairs for s in pair))
        self.n_leaves = {s: len(leaf_set(s)) for s in nets}
        cmds, at = list(self.pairs), {}
        for s in nets:
            star = "(" + ",".join(leaf_set(s)) + ")"
            at[s] = len(cmds)
            cmds += [(s, star + ";"), (s, f"({star});")]
        keys = [pair_key(a, b) for a, b in cmds]
        by_key = dict(zip(keys, cmds))
        known = self._load()
        todo = [k for k in dict.fromkeys(keys) if k not in known]
        chunks = [todo[i:i + self.chunk] for i in range(0, len(todo), self.chunk)]
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        print(f"[luay] {len(self.pairs)} distances + {2 * len(nets)} size probes ({len(nets)} "
              f"networks): {len(set(keys)) - len(todo)} cached, {len(todo)} to measure in "
              f"{len(chunks)} chunk(s) x {self.jobs} JVM(s)", flush=True)

        def one(i: int, ks: list[str]):
            wd = workdir / f"chunk{i:05d}"
            got = PN.cmpnets_many([by_key[k] for k in ks], methods=("luay",), workdir=wd,
                                  timeout=6 * 3600.0)
            shutil.rmtree(wd, ignore_errors=True)
            return ks, [r["luay"] for r in got]

        n_failed = 0
        if chunks:
            fh = open(self.cache, "a") if self.cache is not None else None
            try:
                with ThreadPoolExecutor(max_workers=self.jobs) as ex:
                    futs = [ex.submit(one, i, ks) for i, ks in enumerate(chunks)]
                    for done, fut in enumerate(as_completed(futs), 1):
                        ks, vals = fut.result()
                        for k, v in zip(ks, vals):
                            if missing(v):
                                n_failed += 1
                                continue
                            known[k] = int(v)
                            if fh is not None:
                                fh.write(json.dumps({"k": k, "luay": int(v)}) + "\n")
                        if fh is not None:
                            fh.flush()
                        if done % max(1, len(chunks) // 20) == 0 or done == len(chunks):
                            print(f"[luay]   chunk {done}/{len(chunks)} | "
                                  f"{time.perf_counter() - t0:.0f} s", flush=True)
            finally:
                if fh is not None:
                    fh.close()
        got = [known.get(k) for k in keys]
        self.luay = got[:len(self.pairs)]
        self.size = {s: (None if got[i] is None or got[i + 1] is None else got[i] + got[i + 1] - 1)
                     for s, i in at.items()}
        print(f"[luay] PhyloNet done in {time.perf_counter() - t0:.1f} s; {n_failed} commands "
              f"failed (not cached)", flush=True)

    def result(self, i: int) -> dict:
        pred, truth = self.pairs[i]
        xp, xt = self.size[pred], self.size[truth]
        return {"luay": self.luay[i],
                "mass_pred": None if xp is None else xp + self.n_leaves[pred],
                "mass_truth": None if xt is None else xt + self.n_leaves[truth],
                "max_exact": None if xp is None or xt is None else (xp + xt) // 2}


def fmt(x) -> str:
    """camus_a104's formatting: ``%g`` floats, empty for missing."""
    if missing(x):
        return ""
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


# ---------------------------------------------------------------------------------- hybrid ancestry

def hybrid_ancestry(net, drop: str = C.OUT) -> frozenset[str]:
    """Taxa below at least one hybrid: the union of the hybrids' hardwired clusters, as labels."""
    below = N.hardwired_clusters(net)
    return frozenset(net.labels[i] for v in below if net.is_hybrid(v) for i in below[v]) - {drop}


def hybrid_ancestry_text(enewick: str, drop: str = C.OUT) -> frozenset[str]:
    """:func:`hybrid_ancestry` off the string, with no parser -- the independent derivation.

    Every hybrid is written once with its body, ``(...)#Hk`` or a leaf ``label#Hk``, and elsewhere
    as a bare ``#Hk``.  A hybrid inside another's body is only a tag there, but its own body is
    counted in its own right, so the union of the bodies' leaves is the whole set.
    """
    out, opened = set(), []
    for i, ch in enumerate(enewick):
        if ch == "(":
            opened.append(i)
        elif ch == ")":
            j = opened.pop()
            if _BODY_TAG.match(enewick, i + 1):
                out.update(_LEAF.findall(enewick, j, i + 1))
    out.update(_LEAF_HYBRID.findall(enewick))
    return frozenset(out) - {drop}


def ha_metrics(pred: frozenset[str], true: frozenset[str]) -> dict:
    """The ``ha_*`` columns.  An empty ``pred`` has no FPR, so no balanced error either."""
    fp, fn = len(pred - true), len(true - pred)
    fpr = fp / len(pred) if pred else None
    fnr = fn / len(true) if true else None
    return {"ha_n_true": len(true), "ha_n_pred": len(pred), "ha_fp": fp, "ha_fn": fn,
            "ha_fpr": fpr, "ha_fnr": fnr,
            "ha_balanced_error": None if fpr is None or fnr is None else (fpr + fnr) / 2}


# ------------------------------------------------------------------------------------------ rows

def ident(rec: dict) -> tuple[str, int, str]:
    return (rec["phase"], int(rec["rep"]), rec["arm"])


def base_row(rec: dict, method: str, seed: int, decoder: str = "") -> dict:
    return {"method": method + C.decoder_suffix(decoder) if decoder else method, "_base": method,
            "_truth": (rec["phase"], rec["rep"]), "decoder": decoder,
            "gene_trees": C.GENE_TREES, "n": rec["n"], "network_idx": rec["rep"], "seed": seed,
            "k_true": rec["k"], "key": rec["key"], "phase": rec["phase"], "arm": rec["arm"],
            "level": rec["level"], "ils": rec["ils"], "n_taxa": rec["n_taxa"],
            "n_loci": rec["n_loci"], "shortcut_retics": rec["shortcut_retics"],
            "trees_root_exact_frac": rec["root_exact_frac"], "status": "ok"}


def model_row(rec: dict, label: str, seed: int, m: dict, decoder: str) -> dict:
    row = base_row(rec, label, seed, decoder)
    row.update(cluster_f1=m["f1"], cluster_f1_rooted=m["f1"], k_pred=int(m["k_pred"]),
               **{c: m[c] for c in COUNTS})
    if m.get("enewick") is None or int(m["k_pred"]) < 0:
        row["status"] = "no_network"          # realise() refused the selection; no luay exists
    else:
        row["_pred"] = C.strip_gamma(m["enewick"])
    return row


def cell_files(run: Path) -> list[Path]:
    return sorted((run / "cells").glob("netcs_L*_rep*_netcs_ils*.json"))


def gnn_rows(runs: list[Path], by_cell: dict, filtered: bool) -> tuple[list[dict], list[str]]:
    rows, problems, seen = [], [], {}
    for run in runs:
        for f in cell_files(run):
            cell = json.loads(f.read_text())
            rec = by_cell.get(ident(cell))
            if rec is None:
                if not filtered:
                    problems.append(f"{f}: not a staged condition")
                continue
            dec = C.cell_decoder(cell)
            if int(cell["k"]) != rec["k"]:
                problems.append(f"{f}: k = {cell['k']}, staged k = {rec['k']}")
            for m in cell["models"]:
                label = C.GNN_LABELS.get(m["tag"])
                if label is None:
                    problems.append(f"{f}: checkpoint tag {m['tag']!r} has no label")
                    continue
                key = (dec, m["tag"], int(m["seed"]), ident(rec))
                if key in seen:
                    problems.append(f"{key} is in both {seen[key]} and {run}")
                    continue
                seen[key] = str(run)
                rows.append(model_row(rec, label, int(m["seed"]), m, dec))
    present = {(d, t) for d, t, _s, _c in seen}
    need = {(d, t) for d in REQUIRED_DECODERS for t in REQUIRED_TAGS} | present
    for dec, tag in sorted(need, key=lambda x: (variant_rank(x[0]), x[1])):
        seeds = REQUIRED_SEEDS.get(tag, SEEDS)
        gaps = [(s, c) for s in seeds for c in by_cell if (dec, tag, s, c) not in seen]
        if gaps:
            problems.append(f"GNN {tag} ({dec}): {len(gaps)} of {len(seeds) * len(by_cell)} "
                            f"(seed, condition) missing, e.g. {gaps[:2]}")
    return rows, problems


def greedy_rows(runs: list[Path], by_cell: dict, filtered: bool) -> tuple[list[dict], list[str]]:
    rows, got, problems = [], {}, []
    for run in runs:
        for f in cell_files(run):
            cell = json.loads(f.read_text())
            rec = by_cell.get(ident(cell))
            if rec is None:
                if not filtered:
                    problems.append(f"{f}: not a staged condition")
                continue
            dec = C.cell_decoder(cell)
            if (dec, ident(rec)) in got:
                problems.append(f"greedy {dec} {rec['key']} is in both {got[(dec, ident(rec))]} "
                                f"and {run}")
                continue
            got[(dec, ident(rec))] = str(run)
            rows.append(model_row(rec, "greedy", 0, cell["models"][0], dec))
    for dec in sorted(set(REQUIRED_DECODERS) | {d for d, _c in got}, key=variant_rank):
        gaps = sorted(c for c in by_cell if (dec, c) not in got)
        if gaps:
            problems.append(f"greedy ({dec}): {len(gaps)} conditions missing, e.g. {gaps[:3]}")
    return rows, problems


def tob_rows(cells: list[dict], truths: dict) -> list[dict]:
    rows = []
    for rec in cells:
        clusters = truths[(rec["phase"], rec["rep"])]["clusters"]
        for kind in C.TOB_KINDS:
            info = rec["tob"].get(kind)
            if info is None:
                continue                              # TINNIK at n = 200: the corpus has none
            rooted = C.tob_rooted_path(rec["key"], kind).read_text().strip()
            net = N.from_enewick(rooted)
            counts = CN.cluster_counts(CN.restrict_clusters(net, CN.ingroup_of(net)), clusters)
            row = base_row(rec, C.TOB_LABELS[kind], 0)
            row.update(cluster_f1=counts["f1"], cluster_f1_rooted=counts["f1"], k_pred=0,
                       n_blobs_tob=info["n_blobs"], root_cost=info["root_cost"],
                       **{c: counts[c] for c in COUNTS})
            row["_pred"] = pruned(rooted)
            rows.append(row)
    return rows


def resolver_rows(run: Path, cells: list[dict]) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for rec in cells:
        f = run / "cells" / f"{rec['key']}.json"
        if not f.exists():
            problems.append(f"resolvers: no {f.name}")
            continue
        cell = json.loads(f.read_text())
        for name in C.RESOLVERS:
            m = cell["methods"].get(name)
            if m is None:
                problems.append(f"resolvers: {f.name} has no {name}")
                continue
            row = base_row(rec, C.RESOLVER_LABELS[name], 0)
            row["n_blobs_tob"] = cell["tob"]["n_blobs"]
            row["root_cost"] = m.get("root_cost")
            if "error" in m:
                row["status"] = f"error: {m['error']}"
            elif m["out_rootable"]:
                row.update(cluster_f1=m["f1"], cluster_f1_rooted=m["f1"],
                           k_pred=int(m["k_pred"]), **{c: m[c] for c in COUNTS})
                row["_pred"] = pruned(m["enewick_rooted"])
            else:
                row.update(cluster_f1=m["f1_as_emitted"], k_pred=int(m["k_pred"]),
                           status="out_not_rootable")
                if m.get("enewick"):          # hybrid ancestry only: it needs no rooting
                    row["_emitted"] = C.strip_gamma(m["enewick"])
            rows.append(row)
    return rows, problems


def attach_hybrid_ancestry(rows: list[dict], truths: dict) -> tuple[dict, list[str]]:
    """Fill the ``ha_*`` columns (module docstring); return the cross-check counts and problems.

    Runs before :func:`attach_luay`, which pops ``_pred``.  A row is read off ``_pred``, the
    network its luay compares, or for an ``out_not_rootable`` resolver row off ``_emitted``.
    """
    stats = Counter(networks=len(truths), agree=len(truths), rows=0)   # truths: checked in build()
    problems = []
    for row in rows:
        emitted = row.pop("_emitted", None)
        src = row.get("_pred") or emitted
        if src is None:
            continue
        truth = truths[row["_truth"]]
        where = f"hybrid ancestry: {row['method']} {row['key']} seed {row['seed']}"
        try:
            net = N.from_enewick(src)
            got, text = hybrid_ancestry(net), hybrid_ancestry_text(src)
        except (ValueError, IndexError) as e:
            problems.append(f"{where}: {e}")
            continue
        stats["networks"] += 1
        if got != text:
            problems.append(f"{where}: the two derivations differ on {sorted(got ^ text)[:5]}")
            continue
        stats["agree"] += 1
        if set(net.labels) - {C.OUT} != truth["taxa"]:
            problems.append(f"{where}: not the truth's taxon set")
            continue
        row.update(ha_metrics(got, truth["ha"]))
        stats["rows"] += 1
    return dict(stats), problems


def attach_luay(rows: list[dict], truths: dict, luay: PhyloNetLuay) -> list[str]:
    """Every row's luay and masses, plus the truth-string gate (module docstring)."""
    for row in rows:
        pred = row.pop("_pred", None)
        if pred is not None:
            row["_i"] = luay.add(pred, truths[row["_truth"]]["enewick"])
    gate = {}
    for tkey, t in truths.items():
        if t["ours"] is not None:
            gate[tkey] = luay.add(t["ours"], t["enewick"])
    luay.resolve(C.RUNS / "phylonet")
    problems = []
    bad_gate = [(tkey, luay.luay[i]) for tkey, i in gate.items() if luay.luay[i] != 0]
    print(f"[luay] truth-string gate: luay(to_enewick(truth), corpus string) == 0 on "
          f"{len(gate) - len(bad_gate)} of {len(gate)} truths -> {'PASS' if not bad_gate else 'FAIL'}",
          flush=True)
    if bad_gate:
        problems.append(f"truth-string gate: {len(bad_gate)} truths where our writer and the corpus "
                        f"differ as PhyloNet sees them, e.g. {bad_gate[:3]}")
    for row in rows:
        i = row.pop("_i", None)
        if i is None:
            continue
        r = luay.result(i)
        row["mass_pred"], row["mass_truth"] = r["mass_pred"], r["mass_truth"]
        if r["luay"] is None:
            row["status"] = "phylonet_failed"
            continue
        row["luay"] = r["luay"]
        row["luay_norm"] = r["luay"] / closed_form_max(row["n_taxa"], row["k_true"], row["k_pred"])
        if r["max_exact"]:
            row["luay_norm_exact"] = r["luay"] / r["max_exact"]
    return problems


def per_method(rows: list[dict], order: dict) -> dict:
    def med(col, rs):
        xs = [r[col] for r in rs if not missing(r.get(col))]
        return statistics.median(xs) if xs else None
    labels = sorted({r["method"]: r for r in rows}.values(), key=lambda r: row_order(r, order))
    out = {}
    for first in labels:
        rs = [r for r in rows if r["method"] == first["method"]]
        out[first["method"]] = {
            "rows": len(rs), "decoder": first["decoder"],
            "status": dict(Counter(r["status"] for r in rs)),
            **{f"median_{c}": med(c, rs) for c in
               ("cluster_f1", "cluster_f1_rooted", "luay", "luay_norm", "luay_norm_exact",
                "k_pred", "k_true", "ha_fpr", "ha_fnr", "ha_balanced_error", "root_cost")}}
    return out


def build(args, jar: Path) -> int:
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run netcs_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    want = {k for k in args.keys.split(",") if k}
    cells = [c for c in stage["cells"] if not want or c["key"] in want]
    if want and len(cells) != len(want):
        raise SystemExit(f"--keys: {sorted(want - {c['key'] for c in cells})[:5]} are not staged")
    by_cell = {ident(c): c for c in cells}
    networks = {(r["phase"], r["rep"]): r for r in stage["networks"]}
    truths = {}
    for tkey in sorted({(c["phase"], c["rep"]) for c in cells}):
        rec = networks[tkey]
        t = C.truth_of(tkey[1], phase=tkey[0])
        if t.n_retics != rec["k"]:
            raise SystemExit(f"{tkey}: truth n_retics {t.n_retics} != staged k {rec['k']}")
        corpus = C.phylonet_truth_path(*tkey).read_text().strip()
        ha = hybrid_ancestry(t)
        if not ha or ha != hybrid_ancestry_text(corpus):
            raise SystemExit(f"{tkey}: the truth's hybrid ancestry is empty, or its two "
                             f"derivations differ")
        # The writer's string is compared with the corpus's only where the writer can round-trip.
        ours = (None if rec["wrapped_tags"] or rec["shortcut_retics"]
                else C.strip_gamma(N.to_enewick(t)))
        truths[tkey] = {"enewick": corpus, "ours": ours, "clusters": CN.label_clusters(t),
                        "taxa": set(t.labels), "ha": ha}

    def runs_like(pattern: str) -> list[Path]:
        return sorted(p for p in C.RUNS.glob(pattern) if (p / "cells").is_dir())

    gnn_runs = [Path(p) for p in args.gnn_runs] if args.gnn_runs is not None else runs_like("gnn*")
    greedy_runs = ([Path(p) for p in args.greedy_runs] if args.greedy_runs is not None
                   else runs_like("greedy*"))
    rows, problems = gnn_rows(gnn_runs, by_cell, bool(want))
    g_rows, g_prob = greedy_rows(greedy_runs, by_cell, bool(want))
    r_rows, r_prob = resolver_rows(Path(args.resolver_run), cells)
    rows += g_rows + tob_rows(cells, truths) + r_rows
    problems += g_prob + r_prob
    n_models = sum(len(REQUIRED_SEEDS.get(t, SEEDS)) for t in REQUIRED_TAGS)
    expected = sum((n_models + 1) * len(REQUIRED_DECODERS) + len(c["tob"]) + len(C.RESOLVERS)
                   for c in cells)
    if len(rows) != expected:
        problems.append(f"{len(rows)} rows, {expected} expected for {len(cells)} conditions")
    if problems:
        print("[csv] incomplete:\n" + "\n".join(f"  {p}" for p in problems))
        if not args.allow_partial:
            print("[csv] nothing written; --allow-partial writes what exists")
            return 1

    ha_stats, ha_problems = attach_hybrid_ancestry(rows, truths)
    problems += ha_problems
    print(f"[ha] hybrid ancestry: the two derivations agree on {ha_stats['agree']} of "
          f"{ha_stats['networks']} networks; {ha_stats['rows']} rows scored", flush=True)
    luay = PhyloNetLuay(jar, cache=None if args.no_cache else C.RUNS / "phylonet" / "cache.jsonl",
                        jobs=args.jobs, chunk=args.chunk)
    (C.RUNS / "phylonet").mkdir(parents=True, exist_ok=True)
    problems += attach_luay(rows, truths, luay)
    over = [r for r in rows if not missing(r.get("luay_norm_exact")) and r["luay_norm_exact"] > 1]
    if over:                                     # impossible by construction; say so if it happens
        problems.append(f"{len(over)} rows with luay_norm_exact > 1, e.g. "
                        f"{[(r['method'], r['key'], r['seed']) for r in over[:3]]}")
    offsets = dict(Counter(r["mass_truth"] - (2 * r["n_taxa"] + 3 * r["k_true"] - 1)
                           for r in rows if not missing(r.get("mass_truth"))))

    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (row_order(r, order), r["phase"], r["network_idx"], r["arm"], r["seed"]))
    out_csv, out_json = Path(args.out_csv), Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")          # camus_a104's line ending, not csv's \r\n
        w.writerow(CAMUS_COLUMNS + EXTRA_COLUMNS)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in CAMUS_COLUMNS + EXTRA_COLUMNS])

    strata: dict[str, list[dict]] = {}
    for r in rows:
        strata.setdefault(f"L{r['level']}_n{r['n']}_ils{r['ils']:g}", []).append(r)
    parity_p = C.RUNS / "parity.json"
    summary = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(C.PKG),
        "stage": {"built": stage["built"], "dataset_listing_sha256": stage["dataset_listing_sha256"]},
        "conditions": len(cells), "rows": len(rows), "rows_expected": expected,
        "complete": not problems and not want, "problems": problems,
        "decoders": sorted({r["decoder"] for r in rows if r["decoder"]}, key=variant_rank),
        "rooting": {"gene_trees": "true-root oracle: OUT on the edge nearest a displayed true root "
                                  "(netcs_eval/common.py); GNN and greedy rows are handed it",
                    "resolvers_and_tobs": "run unrooted; rooted by the same oracle only to score "
                                          "cluster_f1 and luay",
                    "hybrid_ancestry": "needs no root"},
        "luay": {"engine": "PhyloNet CMPNETS -m luay", "jar": str(jar),
                 "truth": "the corpus string, unary root removed, gammas stripped",
                 "sizes": "two star probes per network",
                 "truth_mass_minus_2n_plus_3k_minus_1": offsets},
        "hybrid_ancestry": {
            "source": "Dai & Molloy, bioRxiv 10.64898/2026.06.06.730607, Fig. 3",
            "definition": "P, T = the ingroup taxa below >= 1 hybrid of the prediction, of the "
                          "truth; ha_fpr = |P - T| / |P|, ha_fnr = |T - P| / |T|, "
                          "ha_balanced_error = their mean",
            "empty_prediction": "ha_fpr and ha_balanced_error empty (0/0)",
            "cross_check": "hardwired clusters of the parsed Network vs an eNewick string scan",
            **ha_stats},
        "runs": {"gnn": [str(p) for p in gnn_runs], "greedy": [str(p) for p in greedy_runs],
                 "resolvers": str(args.resolver_run)},
        "labels": {"gnn": C.GNN_LABELS, "tob": C.TOB_LABELS, "resolvers": C.RESOLVER_LABELS},
        "per_method": per_method(rows, order),
        "per_stratum": {s: per_method(rs, order) for s, rs in sorted(strata.items())},
        "parity": json.loads(parity_p.read_text()) if parity_p.exists() else None,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"[csv] {len(rows)} rows ({expected} expected) -> {out_csv}\n[csv] summary -> {out_json}\n"
          f"[csv] truth mass - (2n + 3k - 1): {offsets}; rows with luay_norm_exact > 1: {len(over)}")
    for label, s in summary["per_method"].items():
        f1, ln, ber = (s[f"median_{c}"] for c in ("cluster_f1", "luay_norm_exact",
                                                   "ha_balanced_error"))
        print(f"  {label:32s} {s['rows']:5d} rows | F1 med {'' if f1 is None else f'{f1:.4f}':6s}"
              f" | luay_norm_exact med {'' if ln is None else f'{ln:.4f}':6s}"
              f" | ha BER med {'' if ber is None else f'{ber:.4f}':6s} | "
              f"k_pred med {s['median_k_pred']} (k {s['median_k_true']}) | {s['status']}")
    return 0 if not problems else 2


# ---------------------------------------------------------------------------------------- parity

def landed(name: str) -> list[dict]:
    return pq.read_table(C.LANDED / name).to_pylist()


def allowed(it: dict) -> set[str]:
    """The fields a known, explained miss may differ in (l3reg_eval/make_csv.py); empty for any other."""
    out = set()
    if (it["phase"], it["rep"]) == ("cam150", 4):
        out.add("mass_truth")                          # the size-2-cycle truth
    if it["what"] == "tob_tqmc" and (it["phase"], it["rep"]) == ("cam200", 15):
        out |= {"luay", "mass_pred"}                   # a different tree of blobs on the Mac
    return out


def parity(args, jar: Path) -> int:
    t0 = time.perf_counter()
    tcache: dict[tuple[str, int], tuple[str, object]] = {}

    def truth(phase: str, rep: int):
        if (phase, rep) not in tcache:
            t = C.truth_of(rep, inputs=C.GNN / "inputs", phase=phase)
            tcache[(phase, rep)] = (C.strip_gamma(N.to_enewick(t)), t)
        return tcache[(phase, rep)]

    luay = PhyloNetLuay(jar, cache=None, jobs=args.jobs, chunk=args.chunk)
    items: list[dict] = []          # one per pair: which gate, and what the landed row says

    def add(gate, what, phase, rep, pred, landed_luay, mass_pred, mass_truth, **extra):
        items.append({"gate": gate, "what": what, "phase": phase, "rep": int(rep),
                      "i": luay.add(pred, truth(phase, int(rep))[0]), "luay": landed_luay,
                      "mass_pred": mass_pred, "mass_truth": mass_truth, **extra})

    # (a) resolver rows: the landed strings themselves
    for what in ("brooqs_tqmctob", "netcs_tqmctob", "brooqs_truetob", "netcs_truetob"):
        for r in landed(f"rl_camus_{what}_tau2.parquet"):
            if missing(r["luay"]):
                continue                    # landed unrootable: nothing to compare
            raw = r["enewick_raw"]
            add("a", what, r["phase"], r["rep"], pruned(raw) if C.OUT in raw else C.strip_gamma(raw),
                r["luay"], r["mass_pred"], r["mass_truth"])
    # (a) TOB-QMC: blob_cpu_v1's Mac trees of blobs against the landed tob_tqmc rows
    tq = {(r["phase"], r["rep"]): r for r in landed("rl_camus_tob_tqmc_tau2.parquet")}
    for p in sorted((MAC_BLOB_RUN / "work").glob("*/tob.rooted.nwk")):
        m = re.match(r"(cam\d+)_rep(\d+)_", p.parent.name)
        r = tq[(m.group(1), int(m.group(2)))]
        add("a", "tob_tqmc", m.group(1), m.group(2), pruned(p.read_text().strip()),
            r["luay"], r["mass_pred"], r["mass_truth"])
    # (a) true TOB: tree_of_blobs on the camus truths, which are level-1, so it is valid here
    for r in landed("rl_camus_tob_true_tau2.parquet"):
        add("a", "tob_true", r["phase"], r["rep"], CN.tree_of_blobs(truth(r["phase"], r["rep"])[1]),
            r["luay"], r["mass_pred"], r["mass_truth"])
    # (b) the Mac GNN networks against the landed GNN rows
    gnn_landed: dict[str, dict] = {}
    for f in sorted((Path(args.gnn_camus_run) / "cells").glob("*.json")):
        cell = json.loads(f.read_text())
        for mdl in cell["models"]:
            if not mdl.get("enewick"):
                continue
            name = (f"rl_gnn_eval_{cell['phase']}{LANDED_ARM[cell['arm']]}_tau2_segment_"
                    f"s{mdl['seed']}{mdl['tag']}.parquet")
            if name not in gnn_landed:
                gnn_landed[name] = ({r["key"]: r for r in landed(name)}
                                    if (C.LANDED / name).exists() else {})
            r = gnn_landed[name].get(cell["key"])
            if r is None or missing(r.get("luay")):
                continue
            add("b", f"{mdl['tag']}/s{mdl['seed']}", cell["phase"], cell["rep"],
                C.strip_gamma(mdl["enewick"]), r["luay"], r["mass_pred"], r["mass_truth"])
    # (c) greedy on the camus cells -- against the landed rows, so only the formal decoder
    greedy_cells = sorted((Path(args.greedy_camus_run) / "cells").glob("*.json"))
    ref = {(r["phase"], int(r["rep"]), r["arm"]): r for r in C.read_tsv(C.GNN / "reference.tsv")
           if r["tag"] == "_mix" and r["seed"] == "0"}
    for f in greedy_cells:
        cell = json.loads(f.read_text())
        if C.cell_decoder(cell) != "level1":
            raise SystemExit(f"{f} was decoded as {C.cell_decoder(cell)}; parity (c) compares "
                             f"against the landed level-1 greedy -- point --greedy-camus-run at a "
                             f"formal-decoder run")
        m = cell["models"][0]
        r = {x["key"]: x for x in landed(f"rl_gnn_eval_{cell['phase']}{LANDED_ARM[cell['arm']]}"
                                         f"_tau2_segment_s0_mix.parquet")}[cell["key"]]
        counts = {k: (m[k], r[f"{k}_greedy"]) for k in ("tp", "fp", "k_pred")}
        counts["f1"] = (round(m["f1"], 12),
                        round(float(ref[(cell["phase"], int(cell["rep"]), cell["arm"])]
                                    ["f1_greedy"]), 12))
        if not m.get("enewick"):
            items.append({"gate": "c", "what": "greedy", "phase": cell["phase"],
                          "rep": int(cell["rep"]), "i": None, "counts": counts})
            continue
        add("c", "greedy", cell["phase"], cell["rep"], C.strip_gamma(m["enewick"]),
            r["luay_greedy"], r["mass_greedy"], r["mass_truth"], counts=counts)

    luay.resolve(C.RUNS / "phylonet_parity")
    report: dict = {"built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jar": str(jar)}
    ok = True

    def diff(it) -> set[str]:
        """The landed fields this pair fails to reproduce."""
        if it["i"] is None:
            return set(_FIELDS)
        res = luay.result(it["i"])
        return {f for f in _FIELDS if missing(it[f]) or res[f] is None or int(res[f]) != int(it[f])}

    def gate(name: str, its: list[dict], label: str) -> bool:
        misses = [(it, d) for it in its if (d := diff(it))]
        known = [(it, d) for it, d in misses if d <= allowed(it)]
        bad = [(it, d) for it, d in misses if not d <= allowed(it)]
        by_field = Counter(f for _it, d in misses for f in d)
        report[name] = {"pairs": len(its), "exact": len(its) - len(misses),
                        "known": [{**{k: it[k] for k in ("what", "phase", "rep")},
                                   "fields": sorted(d)} for it, d in known],
                        "unexplained": [{**{k: it[k] for k in ("what", "phase", "rep")},
                                         "fields": sorted(d)} for it, d in bad[:50]],
                        "misses_by_field": dict(by_field)}
        print(f"[parity] ({name}) {label}: {len(its) - len(misses)} of {len(its)} exact on luay + "
              f"both masses; {len(known)} known misses {sorted({(it['phase'], it['rep']) for it, _d in known})}"
              f", {len(bad)} unexplained -> {'PASS' if not bad else 'FAIL'}")
        for it, d in bad[:10]:
            print(f"      {it['what']} {it['phase']} rep{it['rep']:02d}: {sorted(d)}")
        return not bad

    ok &= gate("a", [it for it in items if it["gate"] == "a"], "resolver/TOB rows")
    g = [it for it in items if it["gate"] == "b"]
    if g:
        ok &= gate("b", g, "Mac GNN networks")
    else:
        report["b"] = None
        print(f"[parity] (b) skipped: no Mac GNN cells under {args.gnn_camus_run}")
    c_items = [it for it in items if it["gate"] == "c"]
    if c_items:
        c_ok = gate("c", c_items, "greedy networks")
        bad_counts = [(it["phase"], it["rep"], {k: v for k, v in it["counts"].items() if v[0] != v[1]})
                      for it in c_items if any(a != b for a, b in it["counts"].values())]
        complete = len(c_items) == 100
        report["c"]["counts_bad"] = bad_counts
        report["c"]["cells"] = len(c_items)
        report["c"]["run"] = str(args.greedy_camus_run)
        c_ok &= not bad_counts and (complete or args.allow_partial)
        ok &= c_ok
        print(f"[parity] (c) greedy tp/fp/k_pred/F1 exact on {len(c_items) - len(bad_counts)} of "
              f"{len(c_items)} camus cells{'' if complete else ' (of 100 expected)'} -> "
              f"{('PASS' if complete else 'PASS (partial)') if c_ok else 'FAIL'}")
    else:
        report["c"] = None
        print(f"[parity] (c) skipped: no {args.greedy_camus_run}/cells -- run "
              f"l3reg_eval/run_greedy.py --camus")

    report["pass"] = bool(ok)
    out = C.RUNS / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"[parity] {'PASS' if ok else 'FAIL'} -> {out}  ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true", help="the gates against landed camus rows")
    ap.add_argument("--phylonet", type=Path,
                    default=Path(os.environ.get("OPC_PHYLONET") or DEFAULT_JAR),
                    help="PhyloNet.jar (Java-8 classes); default $OPC_PHYLONET or %(default)s")
    ap.add_argument("--gnn-runs", nargs="*", default=None,
                    help="GNN run directories; default every runs/gnn*/ with cells")
    ap.add_argument("--greedy-runs", nargs="*", default=None,
                    help="greedy run directories; default every runs/greedy*/ with cells")
    ap.add_argument("--resolver-run", type=Path, default=C.RUNS / "resolvers")
    ap.add_argument("--gnn-camus-run", type=Path, default=MAC_GNN_RUN,
                    help="parity (b): the Mac GNN run on the camus cells")
    ap.add_argument("--greedy-camus-run", type=Path, default=L3REG_GREEDY_CAMUS,
                    help="parity (c): a formal-decoder greedy run on the camus cells")
    ap.add_argument("--keys", default="", help="comma list of condition keys (a smoke); implies partial")
    ap.add_argument("--jobs", type=int, default=4, help="PhyloNet JVMs at once (4 GB heap each)")
    ap.add_argument("--chunk", type=int, default=2000, help="CMPNETS pairs per JVM")
    ap.add_argument("--no-cache", action="store_true", help="measure every luay afresh")
    ap.add_argument("--out-csv", type=Path, default=C.LANDED / "netcs_bench_per_network.csv")
    ap.add_argument("--out-json", type=Path, default=C.LANDED / "netcs_bench.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    jar = args.phylonet.expanduser().resolve()
    if not jar.exists():
        raise SystemExit(f"no PhyloNet jar at {jar}: pass --phylonet or set OPC_PHYLONET")
    return parity(args, jar) if args.parity else build(args, jar)


if __name__ == "__main__":
    raise SystemExit(main())
