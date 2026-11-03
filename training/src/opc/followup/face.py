"""F0: is the truth on the ILP's optimal face?

``followup-plan-v0.3`` §F0 asks for ``W_true = sum_tau W(tau, s_true(tau))`` against the recorded
bound, and reads the answer as a fork: equal means the relaxation is a certified *generator* and
the remaining problem is ranking within the face; strictly less means the bound points away from
the answer and exact optimisation on this formulation is finished.

Two quantities, because the question has two readings
-----------------------------------------------------

**``W_true_induced``** is the plan's literal request: the class vector the true network itself
induces on each triple, scored under the same ``W``.  It is what the truth *is*.

**``W_true_face``** is the stronger and fairer question: the best the objective can do **subject to
the cluster variables being the truth's**.  The ILP's ``y`` variables are free up to the budget, so
a hand-built assignment understates what ``S_true`` can legally achieve -- the truth is entitled to
pick its own best candidate blocks.  This is computed by fixing ``x`` and re-solving, which is a
bound change rather than an added row (:func:`opc.solve.backend.solve_milp` takes ``lower`` and
``upper`` as arrays; :func:`opc.solve.mpl_ilp.solve` does not expose them, so the backend is called
directly).

``W_true_face <= bound`` always, because it is the same model with more constraints.

**``W_true_induced`` is not bounded by ``W_true_face``, and an earlier version of this module
asserted that it was.**  ``W_true_induced`` is computed *outside* the model: it reads each triple's
class off the true network's own restriction and scores it, without checking that the coupling and
the budget can grant that class.  Two things stop them, both measured:

* **the budget.**  ``sum y <= k`` allows only ``k`` blocks on, and each block hands a triple
  exactly one signature.  The truth's induced signature comes from the *true cycle's* extent; the
  grantable one comes from a *candidate block's* extent, and those differ -- F5 measured true
  extents at a
  median of 19 taxa against a candidate p75 of 41-50.  On ``n50_rep0_r1_cu0.5_g0.5_small`` (k = 1,
  true cycle 13 taxa) the best single candidate block has extent 15 and reproduces the induced
  signature on **5,816 of 6,356** reticulate-induced triples.  The other 540 cannot be served at
  all, and ``W_true_face`` came out 1,744 log units *below* ``W_true_induced``;
* **expressibility.**  Ignoring the budget entirely, some induced classes are still unreachable --
  516 of 19,600 triples on ``n50_rep0_r2_cu1.0_g0.5_small``, all ``ret1_1``.

So ``W_true_induced`` is a **reference score** -- what the truth's own shapes are worth under this
``W`` -- and not an attainable objective value.  The genuine invariant is only
``W_true_induced <= sum_tau max_s W(tau, s)``, which ``tests/test_followup.py`` asserts.

Why this hid.  Under ``saturated`` mode every reticulate class carries the *same* ``W``, so a
triple that induces ``ret1_1`` and is granted ``ret1_2`` scores identically: the mismatch exists
but cannot move the objective.  That is why the two agreed bit-exactly on all 149 saturated
cells -- which is a restatement of A46, not a fact about the truth.  Under the repaired
profiled ``W`` they separate in **both** directions: ``face > induced`` on 3 of 4 cells,
``face < induced`` on the fourth.

A warning about reading this under ``--w-mode saturated``
---------------------------------------------------------

Do not.  A46 measured ``sum_tau max_s W(tau, s) == sum_tau saturated(tau)`` exactly in that mode,
so every feasible point that reaches a reticulate class on every triple attains the optimum and
``W_true_face == bound`` **by arithmetic**, carrying no information about the truth.  F0 is a real
experiment only against a profiled table, which is why F1 runs first.  Both are computed anyway, so
the vacuity is visible beside the answer rather than argued about.
"""

from __future__ import annotations

import numpy as np

from opc.mpl import trinets
from opc.solve import backend, mpl_ilp, truth

__all__ = ["TOPOLOGY_SLOT", "induced_classes", "w_of_classes", "w_true_face", "face_row"]

#: cherry -> resolution column, in the ``(ab|c, ac|b, bc|a)`` order of
#: :meth:`opc.mpl.triples.TripleCounts.as_rows`.  ``restrict_to_triple`` numbers a trinet's leaves
#: by position in the *sorted* triple, and ``triple_index`` emits sorted triples, so the two agree.
TOPOLOGY_SLOT = {frozenset((0, 1)): 0, frozenset((0, 2)): 1, frozenset((1, 2)): 2}


def induced_classes(net, index: np.ndarray) -> np.ndarray:
    """The class each triple falls into under ``net``.  ``int[n_triples]``.

    Tree trinets take their resolution slot; reticulate ones take the signature class computed by
    the **repaired** :func:`opc.solve.mpl_ilp.shape_signature` (A48).  A trinet whose root does not
    split 2--1, or whose signature is outside :data:`opc.solve.mpl_ilp.RET_SIGNATURES`, falls back
    to ``star`` -- which is always available and never optimistic, so the fallback can only make
    ``W_true_induced`` a lower estimate and never an inflated one.
    """
    index = np.asarray(index)
    out = np.full(index.shape[0], 3, dtype=np.int64)  # star
    for t, row in enumerate(index):
        tri = trinets.restrict_to_triple(net, tuple(int(x) for x in row))
        sig = mpl_ilp.shape_signature(tri)
        if sig is None:
            # Descend past a unary root before looking for the cherry.  ``restrict_to_triple``
            # suppresses a degree-2 node only when it has exactly one parent *and* one child, and
            # the root has none -- so a trinet's root is routinely a chain above the real MRCA.
            # Measured on one n = 50 cell: skipping this put 40% of triples on the star fallback,
            # which cost W_true_induced about 885 log units a triple and made the truth look
            # 1.4x worse than the bound for a reason that was entirely an indexing slip.
            node = tri.root
            while len(tri.children.get(node, ())) == 1:
                node = tri.children[node][0]
            for child in tri.children.get(node, ()):
                taxa = mpl_ilp._below(tri, child)
                if len(taxa) == 2:
                    out[t] = TOPOLOGY_SLOT[frozenset(taxa)]
                    break
        elif sig in mpl_ilp.RET_SIGNATURES:
            out[t] = mpl_ilp.RET_OFFSET + mpl_ilp.RET_SIGNATURES.index(sig)
    return out


def w_of_classes(table: mpl_ilp.ShapeTable, classes: np.ndarray) -> float:
    """``sum_tau W(tau, classes[tau])``."""
    classes = np.asarray(classes, dtype=np.int64)
    return float(table.w[np.arange(table.n_triples), classes].sum())


def w_true_face(model: mpl_ilp.Model, ground_set, net, *, solver: str = "gurobi",
                time_limit: float = 3600.0, threads: int = 0,
                mem_limit: float = 0.0, gap_limit: float = 0.0) -> dict:
    """Re-solve with the cluster variables pinned to ``S_true``.

    Returns the objective the truth's selection can reach, the solver status, and how many true
    clusters survived into the ground set -- the last because a truth that is only partly present
    is a different experiment from one that is wholly present, and E0 says which cells are which.

    **Pass ``gap_limit``.**  ``backend._gurobi`` sets ``MIPGap`` only when the limit is truthy, so
    the default ``0.0`` leaves Gurobi at *its* default of 1e-4 -- and this quantity is compared
    against the bound at a relative difference that has been measured as low as 5e-5.  At the
    default tolerance "the truth is off the face" and "the solver stopped early" are the same
    number.  The interim report's §5 records the same defect costing 14 proven optima.
    """
    rows = truth.true_cluster_rows(ground_set.words, net)
    x = np.zeros(model.n_clusters)
    x[rows] = 1.0
    lower = np.zeros(model.c.shape[0])
    upper = np.ones(model.c.shape[0])
    lower[model.x_offset:model.x_offset + model.n_clusters] = x
    upper[model.x_offset:model.x_offset + model.n_clusters] = x
    res = backend.solve_milp(
        model.c, model.a_ub, model.b_ub, integrality=model.integrality,
        lower=lower, upper=upper, time_limit=time_limit, backend=solver,
        gap_limit=gap_limit, threads=threads, mem_limit=mem_limit,
    )
    return {
        "w_true_face": float(-res.objective) if res.x is not None else float("nan"),
        "w_true_face_status": res.status,
        "w_true_face_termination": res.termination,
        "w_true_face_seconds": float(res.seconds),
        "w_true_face_mip_gap": float(res.mip_gap),
        "gap_limit": float(gap_limit),
        "n_true_in_ground_set": int(rows.size),
        "n_true_clusters": int(len(np.unique(rows))),
    }


def face_row(table, model, ground_set, net, index, *, bound_valid: float, **kw) -> dict:
    """One F0 row: both readings, their gaps against the bound, and the invariant."""
    classes = induced_classes(net, index)
    induced = w_of_classes(table, classes)
    row = {"w_true_induced": induced, "w_mode": table.mode}
    row.update(w_true_face(model, ground_set, net, **kw))
    face = row["w_true_face"]
    row["frac_induced_reticulate"] = float((classes >= mpl_ilp.RET_OFFSET).mean())
    row["frac_induced_star"] = float((classes == 3).mean())
    if np.isfinite(bound_valid):
        row["induced_gap"] = float(bound_valid - induced)
        row["face_gap"] = float(bound_valid - face) if np.isfinite(face) else float("nan")
        row["induced_rel_gap"] = float((bound_valid - induced) / abs(bound_valid))
        row["face_rel_gap"] = (float((bound_valid - face) / abs(bound_valid))
                               if np.isfinite(face) else float("nan"))
    # The real invariant: the truth's reference score cannot exceed the table's own row maximum,
    # which is what any assignment is bounded by.  This one does hold.
    row["row_max"] = float(table.w.max(axis=1).sum())
    row["induced_le_row_max"] = bool(induced <= row["row_max"] + 1e-6)
    # A *measurement*, not an invariant: is the truth's induced assignment something the coupling
    # and the budget can actually deliver?  See the module docstring -- it is routinely not.
    row["induced_achievable"] = bool(not np.isfinite(face) or induced <= face + 1e-6)
    # Kept under its old name so rows written before this correction stay joinable; it always was
    # this measurement, it was only mislabelled as an invariant.
    row["invariant_ok"] = row["induced_achievable"]
    return row
