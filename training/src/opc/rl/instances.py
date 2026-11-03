"""W2/W3 -- one training instance, built once and cached.  ``RL_PLAN.md`` §2 and §7.

An instance is ``I = (X, C_tau, w, rho, k)``: taxa, the candidate clusters as bitsets, the support
weights, the observed rooted-triple frequency table, and the reticulation budget.  Everything here
already existed somewhere -- ``candidates.build`` for the ground set, ``triples.count_newick_file``
for ``rho``, ``OverlapCSR.of`` for the graph the mask runs on, ``truth.true_cluster_rows`` for the
oracle labels -- and this module is the place that assembles them in a fixed order and writes the
result down.

**Why it is cached rather than rebuilt.**  Building the tau = 2 ground set is 0.4-1.9 s per cell
(measured across CU on the S50 grid) and the locus scan that produces ``rho`` is a few seconds
more.  Against an MPL-reward step at ~200 s that is noise; against the **oracle-F1 arm**, which
spends no MPL at all and runs thousands of steps an hour, it is the entire step.  One arm must
not set the
other arm's design, so the instance is built once.

The comparison that motivated tau = 2 in the first place is the same arithmetic one level up:
the tau = 0 ground set costs **94 s** at CU = 0.25 against tau = 2's 1.9 s, and there are ~10^3
cells (A66, A67).

**No torch here.**  ``opc.rl.env`` is numpy-only and so is this; the policy is where the device
appears.  That keeps an instance picklable, small, and loadable in a pool worker that has no GPU.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opc import arms as armlayout
from opc import bitset
from opc.clusters import blen, extract
from opc.mpl import triples
from opc.rl.env import KERNEL, OverlapCSR
from opc.simulate import networks as N
from opc.solve import candidates
from opc.solve import truth as T

__all__ = [
    "Instance",
    "NODE_FEATURES",
    "GLOBAL_FEATURES",
    "NODE_BLOCKS",
    "CONFLICT_FEATURES",
    "W13_FEATURES",
    "node_features",
    "conflict_features",
    "global_features",
    "cache_dir",
    "check_features",
    "LEGACY_SIZE_FEATURES",
    "build",
    "build_from_trees",
    "load",
    "load_or_build",
]

#: The **graph-derived** node features, in order -- everything the arm had before W13.
#: ``RL_PLAN`` §3 lists "``w(A)`` raw / rank / z-score, ``|A|``,
#: ``|A|/n``, gene-tree occurrence count, overlap-degree, component size", instance-normalised.
#:
#: Two of those collapse and one needs a companion, and both facts are properties of this
#: formulation rather than choices:
#:
#: * **``w(A)`` and the gene-tree occurrence count are the same number.**  ``SupportCounter``
#:   counts the loci whose gene tree induces the cluster, and that count *is* the support weight
#:   (A29).  So ``support / m`` is entered once, as a fraction of loci, not twice under two names.
#: * **``|A|`` needs ``min(|A|, n - |A|)`` beside it.**  A cluster and its complement carry the
#:   same split, and the clustering system is rooted, so size alone is not a scale-free statement
#:   about how much of the taxon set a cluster resolves.
#:
#: Everything is scaled into roughly ``[0, 1]`` or standardised **per instance**, which is what
#: makes the encoder size-agnostic and therefore makes G-R6 (n <= 30 -> n = 50) a real test rather
#: than a rescaling exercise.  W13's blocks hold to the same rule: every one of them is a ratio
#: taken *inside a single gene tree* or *inside a single instance*, so none of them carries a unit
#: and none of them is a statement about problem size -- which is the property A96/A97 removed
#: three global features for lacking.
BASE_NODE_FEATURES: tuple[str, ...] = (
    "support_frac",      # w(A) / m -- the fraction of loci inducing A; also the occurrence count
    "support_rank",      # rank of w(A) among C_tau, ascending, / |C|
    "support_z",         # (w - mean) / sd over C_tau
    "log_support",       # log1p(w) / log1p(max w)
    "size_frac",         # |A| / n
    "size_minor",        # min(|A|, n - |A|) / n
    "degree_frac",       # overlap-degree / (|C| - 1)
    "log_degree",        # log1p(deg) / log1p(max deg)
    "component_frac",    # size of A's overlap component / |C|
    "is_kernel",         # degree 0: crosses nothing, so no constraint can ever refuse it
)

#: W13's fifth block, and the only one that needs no gene-tree parse: it is a function of the
#: overlap CSR and the support vector, both already on the instance.
#:
#: **Why it is not something the encoder already computes.**  ``SegmentConv`` aggregates a
#: support-weighted neighbour *mean*, and a mean cannot compute a max.  ``bconf_max`` is a max over
#: the overlap neighbourhood, which is structurally outside what any of the three convolutions can
#: express and entirely outside what ``mlp`` (``layers = 0``) can see at all.  Screened at
#: **+0.0614 / +0.0441 / +0.0544** held-out band AUC at b25/b50/b250 -- an order of magnitude above
#: every tree-derived block -- but that screen's baseline is a *linear* model with no aggregation,
#: so it overstates the gain for a trained graph arm and not for ``mlp``.  The prediction it makes,
#: recorded before the run: this closes much of the ``segment`` - ``mlp`` gap, and if it does the
#: finding is about **aggregation** rather than about features, and the follow-up is a
#: max-aggregating conv in ``policy.CONVS`` rather than more columns.
CONFLICT_FEATURES: tuple[str, ...] = (
    "bconf_max",         # max w over A's overlap neighbours, / m
    "bconf_mean",        # mean w over A's overlap neighbours, / m
    "bconf_ratio",       # w(A) / (w(A) + max neighbour w) -- 1.0 for an isolated (kernel) cluster
)

#: The five W13 blocks, by name.  Consumed by ``nodecls.block_idx`` for ``--zero-node-block``;
#: **the indices are derived from these names and are never written as literals** -- A48's rule,
#: and A97's near-miss, where the literal ``(2, 3, 4)`` stayed *in range* after a narrowing and
#: would have zeroed three useful features while reporting a different ablation.
NODE_BLOCKS: dict[str, tuple[str, ...]] = {**blen.TREE_BLOCKS, "bconf": CONFLICT_FEATURES}

#: Every column W13 added, in order.  Named so a stale cache and a legacy checkpoint can both be
#: recognised rather than silently misread, exactly as :data:`LEGACY_SIZE_FEATURES` is.
W13_FEATURES: tuple[str, ...] = blen.TREE_FEATURES + CONFLICT_FEATURES

#: Node features, in order: the ten graph-derived ones above, then W13's twenty-one.
NODE_FEATURES: tuple[str, ...] = BASE_NODE_FEATURES + W13_FEATURES

#: Global features, in order.  ``k`` enters the encoder here *and* the decoder context and the
#: budget mask -- ``RL_PLAN`` §2's "k is given at inference", all three routes.
GLOBAL_FEATURES: tuple[str, ...] = (
    "k",                 # the reticulation budget, raw: it is small and its scale is real
    "k_over_n",
    "density",           # 2|E| / (|C|(|C|-1))
    "mean_degree_frac",  # mean degree / (|C| - 1)
    "kernel_frac",       # |kernel| / |C|
)

#: The three features **removed** in A97, kept by name so a stale instance and a legacy checkpoint
#: can both be recognised rather than silently misread.
#:
#: ``n_over_100``, ``log_n_clusters`` and ``log_n_components`` were statements about problem *size*
#: rather than about the instance, and the only inputs the model had to extrapolate on --
#: ``n_over_100`` ran 0.25/0.5 in training against **2.5** at n = 250.  A96 measured zeroing them at
#: **+0.0403** ΔF1 at b250 (p = 0.000, Δ`luay` -18.57), **+0.0225** at b100 and **+0.0182** at b25:
#: they cost accuracy *in* the training range as well as outside it, so they were buying nothing
#: anywhere, and A97 removed them rather than leaving a run-time flag to switch them off.
#:
#: **Recovering the old feature set is a git operation, not a flag.**  Instance caches are live and
#: rebuildable (see :func:`cache_dir`), so reproducing a pre-A97 checkpoint's evaluation means
#: checking out the commit before A97 and rebuilding ``instances_tau<t>/`` -- the same recovery
#: pattern this repository already uses for the archived drivers.
LEGACY_SIZE_FEATURES: tuple[str, ...] = ("n_over_100", "log_n_clusters", "log_n_components")


def _degrees(graph: OverlapCSR) -> np.ndarray:
    return (graph.indptr[1:] - graph.indptr[:-1]).astype(np.float64)


def conflict_features(gs, graph: OverlapCSR) -> np.ndarray:
    """``float32[|C|, 3]`` -- :data:`CONFLICT_FEATURES`.  No gene-tree parse; the CSR is enough.

    A row's overlap neighbours are exactly the clusters it *crosses*: an overlapping pair whose
    intersection is neither of the two, so a selection may not take both unless the intersection is
    taken as well.  "How well supported is the best thing that contradicts me" is therefore a
    statement about contested-ness that neither ``support_frac`` (my own weight) nor
    ``degree_frac`` (how many contradict me, unweighted) can make.

    A kernel row has no neighbours: ``max`` and ``mean`` are 0 and ``ratio`` is 1.0, which reads as
    "nothing contradicts this", which is exactly right and is also what ``is_kernel`` says.
    """
    n_c = int(gs.n_clusters)
    w = np.asarray(gs.support, dtype=np.float64)
    m = max(1.0, float(gs.m))
    nb_max = np.zeros(n_c, dtype=np.float64)
    nb_mean = np.zeros(n_c, dtype=np.float64)
    indptr, nbr = graph.indptr, graph.nbr
    for i in range(n_c):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        if hi > lo:
            block = w[nbr[lo:hi]]
            nb_max[i] = block.max()
            nb_mean[i] = block.mean()
    out = np.stack([
        nb_max / m,
        nb_mean / m,
        w / np.maximum(w + nb_max, 1e-9),
    ], axis=1)
    assert out.shape == (n_c, len(CONFLICT_FEATURES)), (out.shape, CONFLICT_FEATURES)
    return np.ascontiguousarray(out, dtype=np.float32)


def node_features(gs, graph: OverlapCSR, tree_stats: np.ndarray) -> np.ndarray:
    """``float32[|C|, len(NODE_FEATURES)]``.  Static: the graph never changes within an episode.

    ``tree_stats`` is ``blen.cluster_tree_stats``'s ``float32[|C|, len(blen.TREE_FEATURES)]``,
    rows aligned to ``gs.words``.  It is a parameter rather than something computed here because
    this function has no path to the gene trees and :func:`build` already opens them.
    """
    n_c = int(gs.n_clusters)
    w = np.asarray(gs.support, dtype=np.float64)
    deg = _degrees(graph)
    sizes = bitset.popcounts(gs.words).astype(np.float64)
    n_taxa = float(gs.n_taxa)

    comp_size = np.zeros(n_c, dtype=np.float64)
    for comp in gs.components:
        comp_size[np.asarray(comp.vertices, dtype=np.int64)] = float(comp.n_vertices)

    order = np.argsort(np.argsort(w, kind="stable"), kind="stable").astype(np.float64)
    sd = w.std()
    denom = max(1.0, float(n_c - 1))
    out = np.stack([
        w / max(1.0, float(gs.m)),
        order / max(1.0, float(n_c - 1)),
        (w - w.mean()) / sd if sd > 0 else np.zeros(n_c),
        np.log1p(w) / max(1e-9, float(np.log1p(w.max()))),
        sizes / n_taxa,
        np.minimum(sizes, n_taxa - sizes) / n_taxa,
        deg / denom,
        np.log1p(deg) / max(1e-9, float(np.log1p(deg.max()))),
        comp_size / max(1.0, float(n_c)),
        (graph.comp_of == KERNEL).astype(np.float64),
    ], axis=1)
    tree_stats = np.asarray(tree_stats, dtype=np.float64)
    if tree_stats.shape != (n_c, len(blen.TREE_FEATURES)):
        raise ValueError(f"tree_stats must be ({n_c}, {len(blen.TREE_FEATURES)}), "
                         f"got {tree_stats.shape}")
    out = np.concatenate([out, tree_stats, conflict_features(gs, graph)], axis=1)
    # The one thing that must not drift silently: the matrix and its names.  ``global_features``
    # has had this assert since it was written; the node vector has never had one, which is why
    # ``check_features`` could only ever see half the problem.
    assert out.shape == (n_c, len(NODE_FEATURES)), (out.shape, len(NODE_FEATURES))
    return np.ascontiguousarray(out, dtype=np.float32)


def global_features(gs, graph: OverlapCSR, k: int) -> np.ndarray:
    """``float32[len(GLOBAL_FEATURES)]``."""
    n_c = int(gs.n_clusters)
    deg = _degrees(graph)
    pairs = max(1.0, 0.5 * n_c * (n_c - 1))
    out = np.asarray([
        float(k),
        float(k) / float(gs.n_taxa),
        float(0.5 * deg.sum() / pairs),
        float(deg.mean() / max(1.0, n_c - 1)),
        float(gs.kernel_size) / max(1.0, n_c),
    ], dtype=np.float32)
    # The one thing that must not drift silently: the vector and its names.
    assert out.shape == (len(GLOBAL_FEATURES),), (out.shape, GLOBAL_FEATURES)
    return out


@dataclass
class Instance:
    """One cell, at one ``tau``, ready to roll out and to reward.

    ``truth`` and ``truth_rows`` are training-time only: the MPL reward never looks at them, and
    the oracle-F1 arm is *defined* by looking at them.  Keeping both on the instance is what lets
    the two arms share one env, one policy and one loop -- which is the only way the comparison
    between them says anything.
    """

    key: str
    n: int
    k: int
    tau: int
    m: int
    cu: float
    gamma: float
    stratum: str
    rep: int
    gs: object                    # candidates.GroundSet
    graph: OverlapCSR
    obs: np.ndarray               # float64[T, 3] -- rho(t, G), the ab|c / ac|b / bc|a counts
    index: np.ndarray             # int[T, 3]     -- the shared triple index
    truth: object                 # the generating network
    truth_rows: np.ndarray        # int64[]  rows of gs.words that are clusters of `truth`
    n_true: int                   # |C(N_true)| -- may exceed truth_rows.size when tau filtered one
    labels: tuple
    x_nodes: np.ndarray           # float32[|C|, F]
    x_global: np.ndarray          # float32[G]
    #: Which gene trees this was built from -- "" is the simulated ones, "est_bp1000" the IQ-TREE
    #: re-estimates (:mod:`opc.arms`).  **Last, and with an immutable default, deliberately**: a
    #: dataclass default is a real class attribute, so the ~50 GB of pickles written before this
    #: field existed unpickle without it and read back as "", which is exactly what they are.  No
    #: rebuild, and :func:`check_features` must therefore never mention it -- that function is
    #: about *widths*, which have one correct answer per commit; this is about *provenance*, which
    #: legitimately differs between two caches the same commit can both read.
    arm: str = ""

    @property
    def n_clusters(self) -> int:
        return int(self.gs.n_clusters)

    @property
    def truth_coverage(self) -> float:
        return float(self.truth_rows.size / self.n_true) if self.n_true else float("nan")

    def cluster_f1(self, selection) -> float:
        """Cluster F1 of a selection against ``C(N_true)`` -- **the oracle-F1 arm's reward**.

        Byte-for-byte ``opc.solve.truth.align``'s definition, evaluated from ``truth_rows`` instead
        of re-deriving the truth's clusters per call: ``align`` recomputes
        ``nontrivial_clusters(net)`` and a full mask table every time, which at 128 rollouts a step
        is the whole step.  ``tests/test_rl_train.py`` pins the two together, because a reward that
        drifts from the metric it is supposed to be would be invisible in every training curve.

        The denominator is ``n_true``, **not** ``truth_rows.size``: a cluster tau filtered out is
        still a cluster the policy failed to select, and scoring against what survives the filter
        would hand the reward a free point for the ground set's own gaps.
        """
        return self.cluster_counts(selection)["f1"]

    def cluster_counts(self, selection) -> dict:
        """``tp / fp / fn / fn_reachable / precision / recall / f1`` for one selection.

        The same arithmetic :meth:`cluster_f1` always did, with the intermediate counts kept.  F1
        alone cannot say *how* a method is wrong -- a policy that adds a true cluster and a policy
        that drops a false one move it identically -- and the two are different behaviours.

        ``fn`` is denominated on ``n_true`` for the reason in :meth:`cluster_f1`; ``fn_reachable``
        is denominated on what ``C_tau`` actually holds.  Their difference is
        ``n_true - truth_rows.size``, the ground set's own coverage gap, which is zero on most
        n = 50 cells and a fifth of the truth on the worst external n = 100 one.  It belongs beside
        every delta, not in a footnote.
        """
        sel = np.asarray(selection, dtype=np.int64)
        correct = int(np.intersect1d(sel, self.truth_rows, assume_unique=False).size)
        precision = correct / sel.size if sel.size else 0.0
        recall = correct / self.n_true if self.n_true else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if correct else 0.0
        return {
            "tp": correct,
            "fp": int(sel.size) - correct,
            "fn": int(self.n_true) - correct,
            "fn_reachable": int(self.truth_rows.size) - correct,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


def cache_dir(results: Path, phase: str, tau: int, arm: str = "") -> Path:
    """``results/<phase>/instances_tau<t>[_<arm>]/`` -- beside ``genetrees_*`` and ``clusters_*``.

    **The true arm keeps the bare name, and that is the rule rather than backwards compatibility.**
    ``arm = ""`` is the simulated gene trees, which is what every landed pickle and every landed
    checkpoint is; a suffix would move them for nothing.  An estimated arm gets its own *directory*
    rather than its own key inside the shared one because every reader here enumerates the
    directory -- ``w10_gnn.cached_keys``, ``w10_gnn_eval``, ``w9_swap_eval``, ``w12_beam``,
    ``w10_gnn_diag`` and ``probe_blen`` all glob ``*.pkl`` -- so two arms in one directory would
    silently double all of them.  ``w10_gnn_eval`` in particular takes ``test[:max_cells]`` from a
    **lexicographically** sorted list, so mixing arms there would change *which cells are the test
    set* with nothing failing.  A52's rule, and ``CLAUDE.md``'s b-grid replicate hazard, together.

    ``results/n100ext/instances_tau2/`` is the deliberate exception and the precedent to argue
    against: it holds three arms in one directory because ``external.ExternalCell.key`` carries the
    arm and nothing splits that phase by rep.  Do not generalise it.

    A live cache, not a published artifact, so it does **not** go through ``opc.paths`` (which
    files parquets into a study tier).  ``tau`` is in the directory name because it changes every
    array in the file -- A52's rule, and A64's, in the one place where forgetting it would put two
    ground sets under one name.

    **The feature set is *not* in the name, and that is deliberate.**  It is checked instead, by
    :func:`check_features` on every read, because the two policies differ: ``tau`` selects between
    ground sets a run might legitimately want either of, while a feature set is a property of the
    *code* -- there is only ever one correct answer for a given commit, so the right behaviour is to
    refuse a stale pickle rather than to keep both and let a caller pick. A97 relies on this: it
    narrowed ``GLOBAL_FEATURES`` from 8 to 5, and every pre-A97 cache is now a loud failure.
    """
    return Path(results) / phase / (f"instances_tau{tau}_{arm}" if arm else f"instances_tau{tau}")


def summary_name(phase: str, tau: int, arm: str = "", tag: str = "") -> str:
    """The per-run summary parquet's stem+suffix.  **One owner, two readers.**

    ``scripts/rl_instances.py`` writes it and ``scripts/w10_gnn.pos_weight_of`` reads it, and the
    reader's miss is not an error -- it falls back to materialising every training instance before
    epoch 0, which is the peak-memory behaviour ``--lazy-env`` exists to avoid and which A89
    records as a **silent hang** rather than a failure.  So the two spellings must not be able to
    drift, and before this they were two f-strings in two files, one of which already omitted
    ``--tag``.
    """
    return f"rl_instances_{phase}_tau{tau}{'_' + arm if arm else ''}{tag}.parquet"


def _tree_stats(tree_path: Path, taxon_index: dict[str, int], gs, m: int) -> np.ndarray:
    """W13's tree-derived block, with the check that says the second parser read the same file.

    ``blen.cluster_tree_stats`` returns the number of loci each row was seen in, and that number
    must equal the ground set's ``support`` -- which came from ``SupportCounter`` over the same
    Newick through a *different* parser, written months earlier for a different purpose.  So the
    equality is the repository's cross-validation convention discharged for free: an independent
    implementation, not the module checking itself.

    It is worth an assert rather than a test because a silently different parse gives every cluster
    the wrong branch and nothing else in the pipeline would notice -- ``x_nodes`` would simply be
    subtly wrong, and every number downstream with it.

    **The support equality cannot see a length-free tree**, which is the other way this goes wrong
    and the one an estimated arm actually hits: the loci are all there and counted, so ``n_obs`` is
    exactly right while every length-derived column is a constant or an artefact.  That is what
    ``n_degenerate`` is for, and it is checked first because it is the more likely failure.
    """
    stats, n_obs, n_degenerate = blen.cluster_tree_stats(
        tree_path, taxon_index, gs.words, max_loci=m)
    if n_degenerate:
        raise AssertionError(
            f"{tree_path}: {n_degenerate} of {m} loci carry no branch length at all. W13's four "
            f"tree-derived blocks would silently degenerate -- blen_* to 0, bdepth_* to 1, bsib_* "
            f"to 0, and brank_* to the order the nodes appear in the string -- while the support "
            f"cross-check below still passed, because it counts loci and not lengths. If this is "
            f"an estimated arm, its trees were rerooted without "
            f"opc.simulate.outgroup.reroot_on_outgroup(keep_lengths=True); re-run the ingest.")
    support = np.asarray(gs.support, dtype=np.int64)
    if not np.array_equal(n_obs, support):
        bad = int(np.count_nonzero(n_obs != support))
        raise AssertionError(
            f"{tree_path}: blen.cluster_tree_stats saw a different set of loci from "
            f"SupportCounter on {bad} of {support.size} clusters (max |diff| "
            f"{int(np.abs(n_obs - support).max())}). The two parsers disagree; do not trust "
            f"x_nodes until scripts/probe_blen.py says why.")
    return stats


def build(cell, root: Path, *, backend: str = "julia", m: int = 1000, tau: int = 2,
          arm: str = "") -> Instance:
    """Assemble one instance from the grid's caches.

    The load chain is ``scripts/w0_baseline.load_cell``'s, in its fixed order -- base topology,
    reticulations, coalescent-unit scale, gamma, then the gene trees and the cluster table -- and
    the order is not decorative: ``with_reticulations`` before ``scaled`` before ``with_gamma`` is
    what every driver in the study does and what the cached gene trees were simulated under.
    """
    with open(root / "networks" / f"{cell.base_key}.pkl", "rb") as fh:
        base = pickle.load(fh)
    truth = N.derive(base, r=cell.r, cu=cell.cu, gamma=cell.gamma,
                     gamma_range=getattr(cell, "gamma_range", None),
                     gamma_seed=getattr(cell, "gamma_seed", None))
    # Both directories from one place: their names are NOT a single backend suffix apart.
    d = armlayout.arm_dirs(root, backend, arm)
    tree_path = d.trees / f"{cell.key}.nwk.gz"
    taxon_index = {lab: i for i, lab in enumerate(truth.labels)}
    acc = triples.count_newick_file(tree_path, taxon_index, max_loci=m)
    index, obs = acc.as_rows()
    words, support, meta = extract.read_cluster_table(d.clusters / f"{cell.key}.parquet")
    gs = candidates.build(words, support[:, meta["checkpoints"].index(m)], cell.n, m=m, tau=tau)
    graph = OverlapCSR.of(gs)
    rows = T.true_cluster_rows(gs.words, truth)
    tree_stats = _tree_stats(tree_path, taxon_index, gs, m)
    return Instance(
        key=cell.key, n=int(cell.n), k=int(cell.r), tau=int(tau), m=int(m),
        cu=float(cell.cu), gamma=float(cell.gamma), stratum=cell.stratum, rep=int(cell.rep),
        gs=gs, graph=graph, obs=np.ascontiguousarray(obs, dtype=np.float64), index=index,
        truth=truth, truth_rows=rows, n_true=len(N.nontrivial_clusters(truth)),
        labels=tuple(truth.labels),
        x_nodes=node_features(gs, graph, tree_stats),
        x_global=global_features(gs, graph, cell.r), arm=str(arm))


def build_from_trees(key: str, truth, tree_path: Path, table_path: Path, *,
                     m: int = 1000, tau: int = 2, arm: str = "", rep: int = 0,
                     stratum: str = "", cu: float = float("nan"),
                     source: str = "", table_arm: str | None = None) -> Instance:
    """One instance from a **rooted, outgroup-pruned** Newick file and a truth network.

    :func:`build` is the grid's entry point and needs a ``grid.Cell``, a base-topology pickle and
    ``N.derive``; an external dataset has none of those -- it arrives as a network and a file of
    gene trees.  This is the half the two have in common, extracted so there is one definition of
    each quantity rather than two: ``triples`` for ``obs``, ``extract`` for the support table,
    ``candidates.build`` for the ground set, ``true_cluster_rows`` for the target, and
    :func:`_tree_stats` for W13.  ``opc.rl.external.build`` and ``opc.rl.camus.build`` are both
    thin wrappers that say where the files are and what the row's provenance columns should read.

    **The cluster table is written and read straight back** rather than packed in memory.
    ``candidates.build`` wants ``read_cluster_table``'s bitset layout, and packing it by hand would
    be a second encoder of the one object every gate is stated over; round-tripping through the
    study's own writer removes the possibility of the two disagreeing, and the file is a cache.

    ``finalize(min_support=1)``, **not** the module default of 2: A29 measured 51% of ``C`` at
    tau = 0 as having support exactly 1, and that half is not uniform -- hybrid-signature clusters
    are over-represented in it by construction.

    ``table_arm`` defaults to ``arm`` and exists for one caller: ``opc.rl.external`` stamps the
    arm into the cluster parquet's metadata while leaving ``Instance.arm`` empty, because it
    encodes the arm in the *key* instead.  Its parquets are landed, so the two have to stay
    separable rather than be tidied into one.
    """
    taxa = list(truth.labels)
    taxon_index = {lab: i for i, lab in enumerate(taxa)}
    acc = triples.count_newick_file(tree_path, taxon_index, max_loci=m)
    index, obs = acc.as_rows()

    table_path = Path(table_path)
    if not table_path.exists():
        counter = extract.count_newick_file(tree_path, taxon_index, [m])
        masks, support, n_raw = counter.finalize(min_support=1)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        extract.write_cluster_table(table_path, masks, support, [m], len(taxa), n_raw,
                                    source=source or str(tree_path),
                                    arm=arm if table_arm is None else table_arm)
    words, support, meta = extract.read_cluster_table(table_path)
    gs = candidates.build(words, support[:, meta["checkpoints"].index(m)], len(taxa), m=m, tau=tau)
    graph = OverlapCSR.of(gs)
    rows = T.true_cluster_rows(gs.words, truth)
    return Instance(
        key=key, n=len(taxa), k=int(truth.n_retics), tau=int(tau), m=int(m),
        cu=float(cu), gamma=float(truth.gamma), stratum=stratum, rep=int(rep),
        gs=gs, graph=graph, obs=np.ascontiguousarray(obs, dtype=np.float64), index=index,
        truth=truth, truth_rows=rows, n_true=len(N.nontrivial_clusters(truth)),
        labels=tuple(taxa),
        x_nodes=node_features(gs, graph, _tree_stats(tree_path, taxon_index, gs, m)),
        x_global=global_features(gs, graph, int(truth.n_retics)), arm=str(arm))


def check_features(inst, path=None) -> None:
    """Refuse an instance built under a different feature set.

    **The reason this is an assertion and not a comment.**  Every driver takes
    ``n_global_features`` from ``inst.x_global.shape[0]``, so a stale pickle does not fail where it
    is loaded -- it builds a policy of the wrong width and fails inside a matmul several frames
    later, or, for a *checkpoint* of the old width, produces a shape error that reads as a
    corrupted file.  A97 changed the width from 8 to 5, so this is exactly the situation, and the
    cheap check is worth more than the diagnosis.

    **Both halves of the vector are checked.**  Until W13 this looked at ``x_global`` only, which
    made it a guard on one path and therefore a guard on none: W13 took the *node* width from 10 to
    31 and a pre-W13 pickle would have sailed past here and died in ``Policy.embed``.
    """
    where = f"{path}: " if path is not None else ""
    tail = ("The instance cache is live and rebuildable: delete the phase's instances_tau<t>/ and "
            "re-run scripts/rl_instances.py -- scripts/ops/rebuild_instances_w13.sh does every "
            "phase. To read an instance from an older feature set instead, check out the commit "
            "that wrote it and rebuild there.")
    got = int(np.asarray(inst.x_global).shape[0])
    if got != len(GLOBAL_FEATURES):
        raise SystemExit(
            f"{where}instance was built with {got} global features and this code has "
            f"{len(GLOBAL_FEATURES)} ({', '.join(GLOBAL_FEATURES)}). {tail} A97 removed "
            f"{', '.join(LEGACY_SIZE_FEATURES)}.")
    got_nodes = int(np.asarray(inst.x_nodes).shape[1])
    if got_nodes != len(NODE_FEATURES):
        raise SystemExit(
            f"{where}instance was built with {got_nodes} node features and this code has "
            f"{len(NODE_FEATURES)}. {tail} W13 added {len(W13_FEATURES)}: "
            f"{', '.join(W13_FEATURES)}.")


def load(path: Path, *, arm: str | None = None) -> Instance:
    """Read a cached instance, refusing one built under a different feature set.

    The check lives here rather than only in :func:`load_or_build` because ``InstancePool`` and
    ``w10_gnn_eval.py`` read the cache directly; a guard on one path is a guard on none.

    ``arm`` is the *expected* provenance and defaults to not checking, because most callers read a
    path they themselves derived from :func:`cache_dir` and cannot be wrong about it.  The ones
    that can -- a pool holding two arms at once, an evaluator told which arm to test on -- pass it,
    and then a pickle that predates the field reads as ``""`` and matches the true arm exactly.
    """
    with open(path, "rb") as fh:
        inst = pickle.load(fh)
    check_features(inst, path)
    if arm is not None and getattr(inst, "arm", "") != str(arm):
        raise SystemExit(
            f"{path} was built from the {getattr(inst, 'arm', '') or 'true'!r} gene-tree arm and "
            f"this run wants {arm or 'true'!r}. The candidate set, the support table and every "
            f"node feature differ between arms, so this is not a cache that can be shared. "
            f"Rebuild with scripts/rl_instances.py --arm {arm}.")
    return inst


def load_or_build(cell, root: Path, cache: Path, *, backend: str = "julia",
                  m: int = 1000, tau: int = 2, arm: str = "") -> Instance:
    """Read the cached instance, or build and write it.  Atomic rename, as every cache here is.

    **The cached instance's ``tau`` is checked, not trusted.**  ``cache_dir`` puts tau in the path,
    so a mismatch means the caller passed a directory that does not match its argument -- which is
    exactly the confusion A64 records, and it is cheaper to refuse than to train on it.
    """
    path = Path(cache) / f"{cell.key}.pkl"
    if path.exists():
        inst = load(path, arm=arm)
        if inst.tau != int(tau) or inst.m != int(m):
            raise SystemExit(
                f"{path} holds tau={inst.tau} m={inst.m}; this run wants tau={tau} m={m}")
        return inst
    inst = build(cell, root, backend=backend, m=m, tau=tau, arm=arm)
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(inst, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    return inst
