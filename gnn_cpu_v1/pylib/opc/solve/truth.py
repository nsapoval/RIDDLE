"""Does maximising ``w`` actually point at the true network?

Every benchmark in `implementation-plan.md` §3 measures the solver against the *objective* --
does it close, how big is the gap, does the frontier behave.  None of them asks the prior
question: on an instance whose answer is known, is the optimum of MW-GTCS the truth?  The study
has 3,960 replicate networks on disk and has never checked.

It matters because the answer changes what is worth building.  The comparison is against
``S_true = C(N) & C_tau`` -- the true clusters that survived filtering, which is feasible because
(L) and (N3O) are hereditary and ``C(N)`` satisfies both -- so ``S_true`` is a *legal competitor*
with its own weight and its own reticulation count.  Three outcomes, three different projects:

* ``w(S*) ~ w(S_true)`` and high F1 -- the objective points at the truth and the exact solver is
  the thing that matters;
* ``w(S*) >> w(S_true)`` and low F1 -- the **objective regret of the truth** is large: the solver
  is right and the objective is wrong, and no amount of solver work (or learned search over the
  same objective) recovers the network;
* ``w(S*) < w(S_true)`` -- only possible when the solve did not close, and then the gap is real
  rather than cosmetic.

Evaluated at the oracle budget ``k = rho(N)`` so that this measures the objective and not a
model-selection rule that does not exist yet (WS3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from opc import bitset
from opc.simulate import networks as N
from opc.solve import closure, heuristics, overlap

__all__ = ["Alignment", "true_cluster_rows", "feasible_and_cost", "reticulations_of", "align"]


@dataclass
class Alignment:
    """One ``(instance, scheme, budget)`` comparison of an accepted set against the truth."""

    n_true: int
    n_true_in_ctau: int
    n_selected: int
    n_correct: int
    precision: float
    recall: float
    f1: float
    recall_recoverable: float
    w_selected: float
    w_true_feasible: float
    regret: float
    true_feasible: bool
    j_true: int
    rho_selected: int
    rho_true: int
    cluster_distance: int
    level1_ok: bool

    #: False positives: selected and not a cluster of the truth.
    #:
    #: **Properties, not fields.**  ``as_row`` is ``asdict``, so a new *field* adds a column to
    #: every artifact ``exact.recovery_columns`` feeds -- and the repo's append-only rule is that
    #: published values re-derive byte-identically, not that new columns are free to appear inside
    #: an old one.  These three are exact functions of fields that are already there, so they cost
    #: nothing to compute and change no file.
    @property
    def fp(self) -> int:
        return self.n_selected - self.n_correct

    #: False negatives against **the truth**, which is what ``recall`` is denominated on: a cluster
    #: filtered out of ``C_tau`` is still a cluster the method failed to select.
    @property
    def fn(self) -> int:
        return self.n_true - self.n_correct

    #: False negatives against **what the ground set actually contains**.  The gap between this and
    #: :attr:`fn` is the candidate set's own coverage, and it is not small at large ``n`` -- the
    #: n = 100 external set's worst cell holds 0.7931 of its truth, so a fifth of ``fn`` there is
    #: unreachable before any method runs.  Report both or the reader cannot tell the two apart.
    @property
    def fn_reachable(self) -> int:
        return self.n_true_in_ctau - self.n_correct

    def as_row(self) -> dict:
        return asdict(self)


def _row_masks(words: np.ndarray) -> list[int]:
    return [int.from_bytes(words[i].tobytes(), "little") for i in range(words.shape[0])]


def true_cluster_rows(words: np.ndarray, net) -> np.ndarray:
    """Indices of ``words`` that are non-trivial clusters of ``net`` -- i.e. ``C(N) & C_tau``.

    ``nontrivial_clusters`` already drops the singletons and the full taxon set, which is the
    same ground-set convention ``SupportCounter`` uses, so the two sides are comparable without
    further filtering.
    """
    truth = {_mask(c) for c in N.nontrivial_clusters(net)}
    have = _row_masks(words)
    return np.array([i for i, m in enumerate(have) if m in truth], dtype=np.int64)


def _mask(cluster) -> int:
    m = 0
    for t in cluster:
        m |= 1 << int(t)
    return m


def feasible_and_cost(words: np.ndarray, n_taxa: int) -> tuple[bool, int]:
    """``(satisfies (L) and (N3O), reticulations it needs)`` for a whole system.

    Runs the component decomposition the solver uses and then the *independent* validator in
    :mod:`opc.solve.heuristics`, so a claim that ``S_true`` is feasible is checked rather than
    inherited from the theory.  The budget is set to each component's full candidate set, so only
    (L) and (N3O) can fail here.
    """
    if words.shape[0] == 0:
        return True, 0
    dec = overlap.decompose(words, n_taxa)
    ok, j = True, 0
    for comp in dec.components:
        feas = heuristics.check(comp, np.arange(comp.n_vertices), budget=comp.n_hcand)
        ok = ok and feas.ok
        j += feas.n_hybrids
    return ok, j


def reticulations_of(words: np.ndarray, n_taxa: int) -> int:
    """``rho(H(I(S)))`` -- the reticulation number of the network the accepted set stands for.

    Derived from the cover relation by ``|E| - |V| + 1``, with no reference to the budget the
    solver spent.  That independence is what makes B1's T4a (``sum_K j_K == rho``) a test.
    """
    if words.shape[0] == 0:
        return 0
    full = closure.with_trivial_clusters(closure.closure_words(words, n_taxa), n_taxa)
    parent, child = closure.hasse_covers_words(full, n_taxa)
    return closure.reticulation_number(parent, child, full.shape[0])


def _nontrivial_masks(words: np.ndarray, n_taxa: int) -> set[int]:
    sizes = bitset.popcounts(words)
    keep = (sizes >= 2) & (sizes < n_taxa)
    return set(_row_masks(np.ascontiguousarray(words[keep])))


def align(
    words: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
    net,
    n_taxa: int,
) -> Alignment:
    """Compare an accepted set against the network that generated the gene trees.

    ``words`` is the filtered ground set ``C_tau``, ``selected`` indexes into it, and ``weights``
    is the objective the selection was made under -- so the regret is measured in the same units
    the solver optimised, which is the only way it says anything about the objective.
    """
    selected = np.asarray(selected, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    true_rows = true_cluster_rows(words, net)
    n_true = len(N.nontrivial_clusters(net))

    chosen = set(selected.tolist())
    correct = len(chosen & set(true_rows.tolist()))
    precision = correct / len(chosen) if chosen else float("nan")
    recall = correct / n_true if n_true else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if chosen and n_true and (precision + recall) > 0
        else 0.0
    )
    recoverable = (
        correct / true_rows.size if true_rows.size else float("nan")
    )

    true_words = np.ascontiguousarray(words[true_rows])
    true_feasible, j_true = feasible_and_cost(true_words, n_taxa)

    sel_words = np.ascontiguousarray(words[selected])
    try:
        rho_sel = reticulations_of(sel_words, n_taxa)
        level1 = True
        if sel_words.shape[0]:
            full = closure.with_trivial_clusters(
                closure.closure_words(sel_words, n_taxa), n_taxa
            )
            got = _nontrivial_masks(full, n_taxa)
            try:
                closure.network_from_hasse(full, n_taxa)
            except ValueError:
                level1 = False
        else:
            got = set()
    except Exception:  # a malformed closure must not lose the rest of the row
        rho_sel, level1, got = -1, False, set()

    want = {_mask(c) for c in N.nontrivial_clusters(net)}
    return Alignment(
        n_true=n_true,
        n_true_in_ctau=int(true_rows.size),
        n_selected=int(selected.size),
        n_correct=int(correct),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        recall_recoverable=float(recoverable),
        w_selected=float(w[selected].sum()),
        w_true_feasible=float(w[true_rows].sum()),
        regret=float(w[selected].sum() - w[true_rows].sum()),
        true_feasible=bool(true_feasible),
        j_true=int(j_true),
        rho_selected=int(rho_sel),
        rho_true=int(net.n_retics),
        cluster_distance=int(len(got ^ want)),
        level1_ok=bool(level1),
    )
