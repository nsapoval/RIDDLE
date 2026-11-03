"""E4's exact-solve core: the parts of the driver that are not the driver.

``scripts/e4_exact.py`` grew into two things at once -- a multiprocessing CLI over the n = 50
grid, and the shared vocabulary four other callers reached into it for: the cache tags, the
ground-set build, the closure realisation, the recovery columns, the thread budget, and the MILP
call.
Those callers loaded it with ``importlib.util.spec_from_file_location`` because a script is not an
importable package, which is a workaround, not a design.

This module is that shared half, promoted.  The driver keeps the pool, the argument parser and the
per-instance workers; everything here is importable, testable and has no ``__main__``.

**The tags are the load-bearing part.**  ``solver_tag``/``tau_tag``/``w_tag`` decide which cache
directory and which parquet a run addresses, and A52 is the amendment that says a key must include
anything that changes the answer.  They are pure functions of their arguments precisely so that a
test can pin them without running a solve -- which is what ``tests/test_mpl_ilp.py`` and
``tests/test_followup.py`` do.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from opc import grid
from opc.clusters import extract
from opc.mpl import triples
from opc.solve import candidates, closure, mpl_ilp
from opc.solve import truth as T

PHASES = {"smoke": grid.SMOKE, "s50": grid.S50}
REDUCED = dict(r={1, 2}, cu={0.5, 1.0}, gamma={0.5})


#: The solver whose outputs keep the unsuffixed paths.
DEFAULT_SOLVER = "highs"


def is_reduced(cell) -> bool:
    return (cell.r in REDUCED["r"] and cell.cu in REDUCED["cu"]
            and cell.gamma in REDUCED["gamma"])


def solver_tag(solver: str) -> str:
    """``""`` for the default solver, ``"_<solver>"`` otherwise -- caches and parquets both.

    **The cache has to be keyed by solver, for exactly the reason the gene-tree cache is keyed by
    backend.**  ``CLAUDE.md``: "do not un-key it, or a julia run silently re-reads the oracle's
    trees and the two stop being independent."  The same sentence holds here with two words
    changed -- a ``--solver gurobi`` run that reads a cache written by HiGHS returns HiGHS's
    numbers under a ``solver: gurobi`` column, which is worse than no cross-check because it looks
    like one.

    The asymmetry -- HiGHS unsuffixed rather than ``_highs`` -- is deliberate and is the weaker of
    two rules giving way to the stronger.  ``results/e4_small/`` already holds eight solved
    instances that ``e4_small.parquet`` was published from, and this repo's artifacts are
    append-only: those files keep their names so that parquet re-derives byte-identically.
    """
    return "" if solver == DEFAULT_SOLVER else f"_{solver}"


def tau_tag(tau: int) -> str:
    """``""`` at the pre-registered tau, ``"_tau<t>"`` above it -- caches and parquets both.

    Keyed for the same reason :func:`solver_tag` is, and against a sharper failure: tau changes the
    *ground set*, so a cached row written at tau = 0 re-read by a tau = 2 run is not a slower
    answer to the same question, it is the answer to a different one, filed under a ``tau: 2``
    column.  Both runs are legitimate and both want the same cell key, which is exactly when a
    shared cache silently wins.

    ``tau = 0`` and ``tau = 1`` select the same clusters (``candidates.build`` filters on
    ``support >= max(tau, 1)``), so they share the untagged path.  The plan pre-registers tau = 0
    and the published artifacts were written there; they keep their names so they re-derive
    byte-identically.
    """
    return "" if int(tau) <= 1 else f"_tau{int(tau)}"


# --------------------------------------------------------------------------------- shared stages


def w_tag(w_signature: str) -> str:
    """``""`` for the signature every published row used, ``"_wfix"`` for the repaired one.

    Keyed for the reason :func:`solver_tag` and :func:`tau_tag` are, and against the same failure.
    A48 corrected ``mpl_ilp.shape_signature`` to index reticulate classes by the cycle apex rather
    than the trinet root, which changes ``W`` in ``profiled`` mode -- and
    ``results/e4_small.parquet`` and ``e4_small_gurobi.parquet`` were both written in that mode.
    Without a tag, re-running
    ``--small`` would overwrite them with different numbers under the same name, which is exactly
    what this repo's append-only rule forbids.  ``--w-signature root`` re-derives them.

    ``saturated`` mode never consults a signature, so the n = 50 grid's numbers are unaffected
    either way; the tag is still applied there so a cache is never ambiguous about what wrote it.

    The full reproduction command for the published profiled artifacts is
    ``--w-signature root --profiler reference``; ``--profiler`` is **not** in the tag because it is
    an implementation of the same quantity rather than a different one, and the two are checked
    against each other in ``tests/test_followup.py`` instead.
    """
    return "" if w_signature == "root" else "_wfix"


def signature_fn(w_signature: str):
    return mpl_ilp.shape_signature_root if w_signature == "root" else mpl_ilp.shape_signature


def profiler_fn(name: str):
    """``None`` is the reference row-by-row optimiser; ``mesh`` is the one that finishes."""
    if name == "reference":
        return None
    from opc.followup import wtable
    return wtable.mesh_profiler


def ground_set_from_loci(newicks, labels, n_taxa, m, tau: int = 0):
    counter = extract.SupportCounter(n_taxa, [m])
    acc = triples.TripleCounts.empty(n_taxa)
    taxon_index = {lab: i for i, lab in enumerate(labels)}
    for line in newicks:
        line = line if line.endswith(";") else line + ";"
        out: list[int] = []
        extract.clusters_from_newick(line, taxon_index, out)
        counter.add_locus(out)
        acc.add_tree(triples.child_leaf_sets(line, taxon_index))
    masks, support, _ = counter.finalize(min_support=1)
    from opc import bitset
    words = np.stack([bitset.from_indices(
        [t for t in range(n_taxa) if int(mask) >> t & 1], n_taxa) for mask in masks])
    return candidates.build(words, support[:, 0], n_taxa, m=m, tau=tau), acc


def realise(model, res, ground_set, labels):
    """The network the ILP's cluster selection stands for: ``I(S*)`` read as a Hasse diagram."""
    chosen = ground_set.words[model.selection(res.x)]
    if chosen.shape[0] == 0:
        return None
    full = closure.with_trivial_clusters(
        closure.closure_words(chosen, ground_set.n_taxa), ground_set.n_taxa)
    return closure.network_from_hasse(full, ground_set.n_taxa, labels=labels)


class TooBig(RuntimeError):
    """The model was built and deliberately not solved.  Recorded, never raised past the driver."""


def recovery_columns(ground_set, selection, net_truth, n_taxa) -> dict:
    """Did the solve recover the *true* network, and not merely a well-scoring one?

    Everything else E4 reports is about the **bound**: how far the relaxation sits above something,
    and how long the solver took to prove it.  None of that answers the question the study is
    actually asking, and until this was added the n = 50 output had no column that could -- `lower`
    versus `logL_truth` says the recovered network *scores* like the truth, which is a different
    claim and a weaker one.

    ``cluster_distance == 0`` is the strong form, and A40 is what licenses reading it that way:
    ``C -> R`` is injective on shortcut-free galled trees, so an identical non-trivial clustering
    system **is** the same network.  ``exact_recovery`` is that test named.

    Never raises.  The ILP that produced ``selection`` may have cost four hours; losing the whole
    row because a closure was malformed would be the expensive kind of tidy.
    """
    try:
        al = T.align(ground_set.words, selection,
                     np.asarray(ground_set.support, dtype=float), net_truth, n_taxa)
        out = {f"align_{k}": v for k, v in al.as_row().items()}
        out["exact_recovery"] = int(al.cluster_distance == 0)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"align_error": f"{type(exc).__name__}: {exc}"}


def thread_budget(jobs: int, threads: int, solver: str) -> tuple[int, int, str]:
    """``(per_worker, total, how)`` -- what this invocation will actually ask the kernel for.

    ``CLAUDE.md`` records that any pool at ``--jobs >= 100`` deadlocks here: the shared
    ``RLIMIT_NPROC`` is 8,192 counting *threads*, a worker fork fails, and the pool then hangs
    silently at 1-8% CPU with no error.  Two runs were lost to it.

    Gurobi reaches the same ceiling from the other direction and it is easier to hit, because
    ``Threads = 0`` does not mean "one", it means **every core** -- 192 on this box.  ``--jobs 48``
    at the default is 9,408 threads requested by one invocation, on a box that already carries
    ~4,700 from four other users.  The failure looks identical and is just as invisible, so it is
    computed and refused up front rather than discovered.

    The ``+ 4`` is the solver's own bookkeeping threads plus the worker process itself; it is an
    estimate, which is why the ceiling it feeds is conservative.
    """
    if solver == "gurobi" and threads <= 0:
        per, how = (os.cpu_count() or 1), "Threads = 0, so Gurobi takes every core"
    elif threads <= 0:
        per, how = 1, "BLAS pinned to 1 thread, solver single-threaded"
    else:
        per, how = threads, "explicit --threads"
    return per, jobs * (per + 4), how


def solve_one(ground_set, obs, idx, catalog, budget, *, time_limit, solver, mode,
              max_tree_coupling, tiebreak=True, max_nonzeros=0, threads=0, mem_limit=0.0,
              w_signature="apex", profiler="reference", dof_lambda=0.0, extent_cap=0):
    table = mpl_ilp.build_shape_table(obs, catalog, mode=mode, dof_lambda=dof_lambda,
                                      profiler=profiler_fn(profiler),
                                      signature=signature_fn(w_signature))
    t0 = time.perf_counter()
    model = mpl_ilp.build_model(ground_set, obs, table, budget, index=idx,
                                triangles=ground_set.triangles(),
                                max_tree_coupling=max_tree_coupling, extent_cap=extent_cap)
    build_seconds = time.perf_counter() - t0
    # A43: `time_limit` stops binding somewhere between 60 s and 300 s on a model this size --
    # measured, 60 s returns in 94 s and 300 s had not returned after 3,270 s.  So the cap below is
    # not an optimisation, it is the only guard that works: a solve that never returns holds its
    # pool worker until the whole run is killed.  Checked *before* the solver is called, because
    # afterwards there is no afterwards.
    if max_nonzeros and model.sizes["n_nonzeros"] > max_nonzeros:
        raise TooBig(
            f"model has {model.sizes['n_nonzeros']:,} nonzeros, above the {max_nonzeros:,} cap; "
            "not solved"
        )
    # The objective never touches the cluster variables, so their optimal face is enormous and an
    # untied solve returns an arbitrary point of it.  Support is the tiebreak because A26 measured
    # it as the scheme to run; any tiebreak would do, and having *none* is what does not.
    weights = np.asarray(ground_set.support, dtype=float)
    res, bound, primal = mpl_ilp.solve(
        model, time_limit=time_limit, solver=solver,
        tiebreak=weights if tiebreak else None, threads=threads, mem_limit=mem_limit)
    return table, model, res, bound, build_seconds, primal


# ------------------------------------------------------------------------------------- verdict


def verdict(df: pd.DataFrame) -> tuple[str, dict]:
    """§E4's pre-registered rule, with A36's saturation column reported beside it.

    **The gap is medianed over proven-optimal rows only, and A45 is why.**  ``relative_gap`` is
    computed against ``bound``, which is the ILP's incumbent; on an instance that hit its time
    limit that number is not an upper bound on ``sup MPL`` at all -- measured five times too low on
    A43's instance at a 10 s limit -- so a median taken over unproven rows is a median over a
    quantity that does not mean what its name says.  Unproven rows already have their own place in
    the verdict: they are what ``frac_proven_optimal`` counts, and §E4's rule reads the two
    together.  ``median_relative_gap_all`` keeps the old, contaminated figure visible so the
    difference between the two is measurable rather than asserted.
    """
    sub = df[df.get("relative_gap").notna()] if "relative_gap" in df else df.iloc[:0]
    if sub.empty:
        return "NOT_RUN", {}
    proven = float((sub.status == "optimal").mean())
    ok = sub[sub.status == "optimal"] if "status" in sub else sub
    gap = float(ok.relative_gap.median()) if len(ok) else float("inf")
    info = {
        "n_instances": int(len(sub)),
        "n_instances_proven": int(len(ok)),
        "median_relative_gap": gap,
        "median_relative_gap_all": float(sub.relative_gap.median()),
        "frac_proven_optimal": proven,
        "median_frac_saturating": float(sub.frac_saturating.median())
        if "frac_saturating" in sub else None,
        "median_frac_assigned_reticulate": float(sub.frac_assigned_reticulate.median())
        if "frac_assigned_reticulate" in sub else None,
    }
    if "relative_gap_vs_optimum" in sub:
        # Available only where the global optimum can be brute-forced.  The pre-registered verdict
        # stays on the plan's own definition; this says how much of that gap is the relaxation and
        # how much is the primal.
        info["median_relative_gap_vs_optimum"] = float(sub.relative_gap_vs_optimum.median())
    if "relative_gap_untied" in sub:
        info["median_relative_gap_untied"] = float(sub.relative_gap_untied.median())
    if "exact_recovery" in sub:
        # Reported, never part of the verdict.  §E4's pre-registered rule is about the *bound*, and
        # bending it after seeing a recovery number would be exactly the thing pre-registration
        # exists to stop.  It is here because E4 had no column that answered "is it the true
        # network", and a bound gap can be excellent while recovery is nil -- A42 saw the MPL
        # argmax miss the truth on 7 of 8 instances, on a set where every bound was optimal.
        info["frac_exact_recovery"] = float(sub.exact_recovery.mean())
        info["median_align_f1"] = float(sub.align_f1.median()) if "align_f1" in sub else None
        info["median_mpl_regret"] = (float(sub.mpl_regret.median())
                                    if "mpl_regret" in sub else None)
    if "relative_gap_valid" in sub:
        # The dual-derived gap: a bound whatever the solver proved, and +inf where it proved
        # nothing.  On an all-optimal set this equals `median_relative_gap` exactly, which is the
        # check that A45's fix did not move A42's published numbers.
        info["median_relative_gap_valid"] = float(sub.relative_gap_valid.median())
    if "n_upper_bound_violations" in sub:
        info["upper_bound_violations"] = int(sub.n_upper_bound_violations.sum())
    if gap <= 0.05 and proven >= 0.80:
        return "GREEN", info
    if gap <= 0.20 or 0.50 <= proven < 0.80:
        return "AMBER", info
    return "RED", info

