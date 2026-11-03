"""E4: the profiled-triple relaxation as one ILP, and the certificate it comes with.

``exp-plan-n50-mpl-v0.1`` §E4.  Maximise ``sum_tau W(tau, s) y_{tau,s}`` over shape assignments
coupled to a feasible cluster selection, with at most ``k`` reticulations.  The optimum is an
**upper bound** on ``sup MPL`` because the parameters are profiled per triple rather than shared;
re-optimising branch lengths and gamma on the network the solution realises gives a **lower
bound**, and the gap between them is what E4 reports.

Two things this is not, and both are easy to assume it is
---------------------------------------------------------

**It is not MW-GTCS with a different weight vector.**  The cluster variables carry *no objective
coefficient at all*.  Every term of the objective sits on the shape variables; ``x`` appears only
in the coupling and the feasibility rows.  So none of :mod:`opc.solve.instance`'s weight schemes
apply, and A21's diagnosis of why the MDC objective is hard does not transfer either way.

**It does not decompose over overlap components.**  ``sum_tau`` runs over taxon triples, and a
triple whose taxa fall in different components still depends on all of them, so the objective is
not additive and :func:`opc.solve.pipeline.solve_ground_set` and the budget DP are both invalid
here.  (How much a component-wise solve *would* lose is the plan's E6, which is a measurement, not
an assumption.)  The (L) and (N3O) rows still live inside components, so the decomposition survives
as a constraint-*generation* device -- which, after A33, is all it was ever going to be at tau = 0.

Why the shape classes are what they are
---------------------------------------

E3 measured this rather than leaving it to E4 to discover.  Over every galled tree with k <= 2 at
n = 5, 6, 7:

* quotienting clustering systems by the **full I2 shape system** is *injective* -- 456,750 systems,
  456,750 fibers at n = 7;
* quotienting by a **single reticulate class** per triple collapses it -- 19,846 fibers for the
  same 456,750 systems, largest fiber 20,160, and only the k = 0 trees land alone.

So an objective that knows only "this triple is reticulate" cannot distinguish reticulate networks
**whatever the data are**, and the catalog's granularity is not a refinement, it is the mechanism.
What is affordable at n = 50 is in between: index the reticulate classes by the **signature**
``(|tau & h|, |tau & (extent(h) \\ h)|)`` -- how many of the triple's taxa sit below the candidate
hybrid, and how many elsewhere on its cycle.  That is the part of the shape a candidate block
determines without knowing which clusters end up selected, it is what makes the coupling writable,
and it keeps the class count at ten per triple instead of one or seventy-one.

``W`` and the saturation report
-------------------------------

``W(tau, res)`` is the MP-EST closed form (:func:`opc.mpl.profile.tree_profile`).
``W(tau, ret(a,b))`` is the best any catalog shape with that signature can do, which is a maximum
over a superset whenever no catalog shape matches -- looser, never invalid.  A36 measured that 5 of
70 reticulate shapes fit *every* ordinary rho exactly, so for those ``W`` is the saturated
multinomial maximum in closed form and the triple contributes nothing to the bound.  Every table
therefore carries :attr:`ShapeTable.saturating`, and **E4 must report the bound's tightness
alongside how many triples got a saturating class** -- otherwise a vacuous bound is
indistinguishable from a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from opc import bitset
from opc.mpl import profile
from opc.solve import overlap

__all__ = [
    "RET_SIGNATURES",
    "ShapeTable",
    "Model",
    "cycle_apex",
    "shape_signature",
    "shape_signature_root",
    "hcand_extents",
    "class_dof",
    "build_shape_table",
    "build_model",
    "solve",
    "valid_upper_bound",
]

#: ``(taxa below the hybrid, taxa elsewhere on its cycle)`` for a triple.  A reticulation survives
#: the restriction to a triple exactly when its hybrid's cluster meets that triple, so ``a >= 1``;
#: the rest is how the remaining lineages sit relative to the cycle.
RET_SIGNATURES = ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0))

#: Class layout, shared by every triple.  ``star`` is uncoupled and always available, which is what
#: keeps the assignment row feasible on a triple no selected cluster resolves.
TREE_CLASSES = ("res0", "res1", "res2")
STAR_CLASS = "star"
CLASSES = (*TREE_CLASSES, STAR_CLASS, *(f"ret{a}_{b}" for a, b in RET_SIGNATURES))
RET_OFFSET = len(TREE_CLASSES) + 1


def _below(shape, v: int) -> set[int]:
    """The taxa of ``shape`` at or under node ``v``."""
    out: set[int] = set()
    stack = [v]
    seen: set[int] = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u in shape.leaf_of:
            out.add(shape.leaf_of[u])
        stack.extend(shape.children.get(u, ()))
    return out


def _ancestors(shape, v: int) -> set[int]:
    """``v`` and everything above it.  A hybrid has two parents, so this is a set walk."""
    out: set[int] = set()
    stack = [v]
    while stack:
        u = stack.pop()
        if u in out:
            continue
        out.add(u)
        stack.extend(shape.parents.get(u, ()))
    return out


def cycle_apex(shape, hybrid: int) -> int:
    """The lowest node reaching ``hybrid`` by **both** of its parents.

    This is the top of the reticulation cycle.  It is *not* in general the trinet's root: a taxon
    that attaches above the apex is outside the cycle altogether, which is the whole content of
    A48.  Ties are broken on node id so the answer does not depend on set iteration order.
    """
    parents = shape.parents.get(hybrid, ())
    if len(parents) < 2:
        return shape.root
    common = _ancestors(shape, parents[0]) & _ancestors(shape, parents[1])
    if not common:
        return shape.root
    return min(common, key=lambda v: (len(_below(shape, v)), v))


def shape_signature_root(shape) -> tuple[int, int] | None:
    """The signature as E4 computed it before A48.  **Defective; kept so it stays measurable.**

    It takes ``b`` against the trinet's *root* rather than the cycle's apex, on the reasoning that
    "on a trinet with one cycle every common ancestor of the two parents does, so the root always
    works".  The root does not work, because a trinet leaf can attach *above* the apex.  The
    consequence is that ``a + b == 3`` identically -- ``tests/test_mpl_ilp.py`` asserted exactly
    that and so pinned the bug -- and therefore that the three classes with ``a + b < 3``, which
    the *coupling* side generates freely, are backed by no catalog shape at all.  Measured on the
    n = 50 catalog: this function puts 45/8/2 shapes into ``(1,2)/(2,1)/(3,0)`` and **zero** into
    ``(1,0)/(1,1)/(2,0)``, where :func:`shape_signature` puts 23/16/6/6/2/2 into all six.

    Kept beside the corrected version for the reason ``msnc3.triple_probabilities_mixture`` is kept
    beside the exact MSNC: so the difference is a measurement rather than an argument.
    """
    if not shape.gamma_source:
        return None
    hybrid = min(shape.gamma_source)
    under_h = _below(shape, hybrid)
    under_apex = _below(shape, shape.root)
    return len(under_h), len(under_apex - under_h)


def shape_signature(shape) -> tuple[int, int] | None:
    """``(a, b)`` for a catalog trinet, or ``None`` if it carries no reticulation.

    ``a`` counts the shape's leaves below a hybrid and ``b`` the leaves below that hybrid's
    **cycle apex** but not below the hybrid itself.  Leaves attaching above the apex are outside
    the cycle and count in neither, so ``a + b`` ranges over 1..3 -- which is what the coupling at
    :func:`build_model` computes against a candidate block's extent, and what
    :func:`shape_signature_root` got wrong (A48).

    With more than one hybrid the first in post-order is used and the shape is still admitted to
    every matching signature -- a coarser grouping makes ``W`` a maximum over more shapes, which
    loosens the bound and cannot invalidate it.
    """
    if not shape.gamma_source:
        return None
    hybrid = min(shape.gamma_source)
    under_h = _below(shape, hybrid)
    under_apex = _below(shape, cycle_apex(shape, hybrid))
    return len(under_h), len(under_apex - under_h)


#: Free-parameter counts used when no catalog shape backs a class.  The star is the constant
#: ``(1/3, 1/3, 1/3)`` and carries none; a tree class carries the one internal branch length above
#: the cherry (the MP-EST profile); a reticulate class carries, measured over the n = 50 catalog,
#: four to five effective edges plus a gamma.  Only used as a fallback -- with a catalog in hand
#: :func:`class_dof` counts the shapes themselves.
DEFAULT_DOF = {"tree": 1, "star": 0, "ret": 5}


def class_dof(catalog=None, *, signature=None) -> np.ndarray:
    """Free parameters per class, for F2's ``W'(tau, s) = W(tau, s) - lambda * df(s)``.

    ``df`` of a *shape* is ``len(profile.effective_edges(shape)) + len(shape.gamma_source)`` --
    the count :mod:`opc.mpl.profile` already maintains, with edges below plain leaves excluded
    because they cannot move ``P`` and counting them would overstate the very risk the number
    exists to measure.

    ``df`` of a *class* is the **minimum** over the shapes that feed it.  Minimum rather than
    maximum because the penalty has to be the smallest one consistent with the class: a class is a
    max over its shapes, so charging it for the most extravagant member would penalise a triple for
    a shape the bound did not use.
    """
    from opc.mpl import profile as _profile

    sig_of = shape_signature if signature is None else signature
    df = np.zeros(len(CLASSES))
    df[:3] = DEFAULT_DOF["tree"]
    df[3] = DEFAULT_DOF["star"]
    df[RET_OFFSET:] = DEFAULT_DOF["ret"]
    if not catalog:
        return df
    per_sig: dict[tuple[int, int], int] = {}
    tree_dfs: list[int] = []
    for shape in catalog.values():
        d = len(_profile.effective_edges(shape)) + len(shape.gamma_source)
        sig = sig_of(shape)
        if sig is None:
            tree_dfs.append(d)
        else:
            per_sig[sig] = min(per_sig.get(sig, d), d)
    if tree_dfs:
        df[:3] = min(tree_dfs)
    for j, sig in enumerate(RET_SIGNATURES):
        if sig in per_sig:
            df[RET_OFFSET + j] = per_sig[sig]
    return df


@dataclass
class ShapeTable:
    """``W(tau, class)`` for every triple, and which entries are saturated."""

    w: np.ndarray  # float[n_triples, len(CLASSES)]
    saturating: np.ndarray  # bool[n_triples, len(CLASSES)]
    saturated: np.ndarray  # float[n_triples] -- the multinomial maximum, for the report
    mode: str
    #: shape keys feeding each class column -- kept because F2 needs the shape -> column map and
    #: the first version of this dataclass discarded it inside a local.
    shapes_by_class: dict = field(default_factory=dict)
    df: np.ndarray | None = None  # float[len(CLASSES)] -- free parameters, see class_dof
    dof_lambda: float = 0.0

    @property
    def n_triples(self) -> int:
        return int(self.w.shape[0])

    def attaining(self, tol: float = 1e-9) -> np.ndarray:
        """Per triple, does the best class in the **table** reach the multinomial maximum?

        Under ``mode='saturated'`` this is ``True`` everywhere *by construction* -- all six
        reticulate columns are set to ``saturated``, so the row maximum is ``saturated`` whatever
        the data say.  Measured over the whole reduced grid: 160 of 160 cells at exactly 1.0000.
        That is not a finding about the data, it is the definition, and it is the same category
        error the published ``frac_saturating`` makes (A47) one level up.  It is kept, and
        documented, because a number that is 1.0 by construction is worth being able to *show*.

        The two statistics that do carry information are :meth:`tree_attains` -- a property of
        ``rho`` alone, independent of mode -- and the deficit ``sum_tau saturated(tau) - bound``,
        which is what the coupling and the budget are actually worth (A46).
        """
        return self.w.max(axis=1) >= self.saturated - tol

    def tree_attains(self, tol: float = 1e-9) -> np.ndarray:
        """Per triple, does the best **tree** class already reach the multinomial maximum?

        True exactly when the triple's two minority counts are equal, because the MP-EST profile
        fits ``(a, (b+c)/2, (b+c)/2)`` and the saturated fit is ``(a, b, c)``.  This is why a
        saturated-mode bound can equal ``sum_tau saturated(tau)`` while a quarter of the triples
        are assigned a *tree* class: those triples were already at the ceiling.
        """
        return self.saturated - self.w[:, :3].max(axis=1) < tol

    def headroom(self) -> float:
        """``sum_tau saturated(tau) - sum_tau max_r W(tau, res_r)``.

        The total room between the all-tree assignment and the saturated ceiling -- the largest
        amount any reticulate class could ever contribute on this data.  It is the denominator the
        bound's deficit should be read against: a deficit of 600 log units means nothing until it
        is set beside the 7,000 that were available.
        """
        return float(self.saturated.sum() - self.w[:, :3].max(axis=1).sum())


def _tree_profile_batch(obs: np.ndarray) -> np.ndarray:
    """:func:`opc.mpl.profile.tree_profile` over every triple and every resolution at once."""
    obs = np.asarray(obs, dtype=float)
    total = obs.sum(axis=1, keepdims=True)
    bc = total - obs
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.clip(np.where(total > 0, 3.0 * bc / (2.0 * np.maximum(total, 1e-300)), 1.0),
                    1e-300, 1.0)
        p_match = 1.0 - 2.0 * u / 3.0
        w = np.where(obs > 0, obs * np.log(np.maximum(p_match, 1e-300)), 0.0) \
            + np.where(bc > 0, bc * np.log(u / 3.0), 0.0)
    return np.where(total > 0, w, 0.0)


def build_shape_table(obs: np.ndarray, catalog=None, *, mode: str = "profiled",
                      restarts: int = 4, dof_lambda: float = 0.0,
                      profiler=None, signature=None) -> ShapeTable:
    """``W`` for every (triple, class).

    ``mode='saturated'`` sets every reticulate class to the multinomial maximum.  That is the
    crudest valid bound -- §6.1's worst case written down as a knob -- and running both modes is
    how E4 measures what the catalog is worth rather than asserting it.

    **It is also the mode every published n = 50 row was produced under, and under it the ILP's
    optimum is a closed-form function of the data** (A46): every triple attains its multinomial
    maximum, so ``sum_tau max_s W(tau, s) == sum_tau saturated(tau)`` exactly, with no dependence
    on the candidate set, the budget or tau.  Read a saturated-mode bound as a statistic of
    ``rho``, never as a statement about a network.

    ``mode='profiled'`` maximises over the catalog shapes matching each signature.  Shapes that
    saturate are short-circuited to the closed form, which is both faster and the thing A36 says
    has to be counted.

    ``profiler`` overrides how a shape is fitted to every triple at once; it is called as
    ``profiler(shape, obs, restarts=restarts)`` and must return ``float[n_triples]``.  The default
    is :func:`opc.mpl.profile.shape_profile` row by row, which is exact and does not finish at
    n = 50 (A43: > 2,400 s); :func:`opc.followup.wtable.mesh_profiler` is the vectorised one.

    ``dof_lambda`` applies F2's per-triple AIC correction ``W - lambda * df(s)``, **after** the
    saturated short-circuits, because applied before them it would be overwritten.

    ``signature`` selects how a catalog shape is assigned to a class: :func:`shape_signature` (the
    default, repaired at A48) or :func:`shape_signature_root` (what every published row was
    produced under).  It is a parameter rather than a silent fix because
    ``results/e4_small*.parquet`` were written in ``profiled`` mode and this repo's artifacts
    re-derive byte-identically or not at all -- ``e4_exact.py --w-signature root`` reproduces them.
    """
    sig_of = shape_signature if signature is None else signature
    obs = np.asarray(obs, dtype=float)
    n = obs.shape[0]
    w = np.zeros((n, len(CLASSES)))
    sat_flag = np.zeros((n, len(CLASSES)), dtype=bool)

    w[:, :3] = _tree_profile_batch(obs)
    w[:, 3] = obs.sum(axis=1) * np.log(1.0 / 3.0)
    saturated = np.array([profile.saturated_log_likelihood(row) for row in obs])
    df = class_dof(catalog, signature=sig_of)

    def finish(shapes_by_class):
        if dof_lambda:
            w[:] = w - float(dof_lambda) * df[None, :]
        return ShapeTable(w=w, saturating=sat_flag, saturated=saturated, mode=mode,
                          shapes_by_class=shapes_by_class, df=df, dof_lambda=float(dof_lambda))

    if mode == "saturated" or not catalog:
        w[:, RET_OFFSET:] = saturated[:, None]
        sat_flag[:, RET_OFFSET:] = True
        return finish({})

    by_signature: dict[tuple[int, int], list] = {}
    reticulate = []
    # Classify each shape **once**, against A36's ordinary-rho grid, rather than fitting it to
    # every triple.  A saturating shape's W is the multinomial maximum for any data at all, so a
    # signature that owns one needs no optimisation -- which is what takes the n = 50 shape table
    # from a day to seconds.
    saturating_shapes: dict[str, bool] = {}
    for key, shape in catalog.items():
        sig = sig_of(shape)
        if sig is None:
            continue
        reticulate.append((key, shape))
        by_signature.setdefault(sig, []).append((key, shape))
        saturating_shapes[key] = profile.saturates(shape, restarts=restarts)

    fit = profiler if profiler is not None else _profile_rows
    shapes_by_class: dict[int, tuple] = {}
    for j, sig in enumerate(RET_SIGNATURES):
        col = RET_OFFSET + j
        # `or reticulate` is the fallback for a signature no catalog shape carries.  Before A48
        # it fired on three of the six classes every time, and since it pools *every* reticulate
        # shape it reliably picked up one that saturates -- so those three columns were the
        # saturated maximum even in profiled mode.  With the apex signature it fires only on a
        # genuinely unsampled class, and the column records which case it was.
        shapes = by_signature.get(sig) or reticulate
        shapes_by_class[col] = tuple(k for k, _ in shapes)
        if not shapes:
            w[:, col] = saturated
            sat_flag[:, col] = True
            continue
        if any(saturating_shapes[key] for key, _ in shapes):
            w[:, col] = saturated
            sat_flag[:, col] = True
            continue
        best = np.full(n, -np.inf)
        for _, shape in shapes:
            best = np.maximum(best, np.asarray(fit(shape, obs, restarts=restarts), dtype=float))
        w[:, col] = np.minimum(best, saturated)
        sat_flag[:, col] = best >= saturated - 1e-6
    for r in range(3):
        shapes_by_class.setdefault(r, tuple(k for k, s in catalog.items()
                                            if sig_of(s) is None))
    return finish(shapes_by_class)


def _profile_rows(shape, obs: np.ndarray, *, restarts: int = 4) -> np.ndarray:
    """The reference profiler: :func:`opc.mpl.profile.shape_profile`, one triple at a time."""
    return np.array([profile.shape_profile(shape, row, restarts=restarts).w for row in obs])


@dataclass
class Model:
    """The assembled MILP, plus every index a caller needs to read the answer back."""

    c: np.ndarray
    a_ub: sp.csr_matrix
    b_ub: np.ndarray
    integrality: np.ndarray
    n_clusters: int
    x_offset: int
    y_cols: np.ndarray  # global columns of the reticulation indicators
    u_offset: int
    n_triples: int
    hcand_words: np.ndarray
    sizes: dict = field(default_factory=dict)

    def selection(self, x: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.asarray(x)[self.x_offset:self.x_offset + self.n_clusters] > 0.5)

    def assignment(self, x: np.ndarray) -> np.ndarray:
        u = np.asarray(x)[self.u_offset:self.u_offset + self.n_triples * len(CLASSES)]
        return u.reshape(self.n_triples, len(CLASSES)).argmax(axis=1)


def hcand_extents(comp: overlap.Component, words: np.ndarray, n_taxa: int):
    """``(h mask, extent mask)`` per candidate hybrid, as dense taxon-membership rows.

    The *extent* is the union of both endpoints over every overlapping pair that meets in ``h`` --
    the taxa the block's cycle spans.  A triple's signature is taken against it, which is the part
    of the induced trinet a candidate block fixes before any cluster is selected.
    """
    dense = bitset.to_dense(np.ascontiguousarray(words[comp.vertices]), n_taxa).astype(bool)
    h_dense = bitset.to_dense(np.ascontiguousarray(comp.hcand), n_taxa).astype(bool)
    extent = np.zeros_like(h_dense)
    for e in range(comp.n_edges):
        k = int(comp.edge_class[e])
        extent[k] |= dense[int(comp.ii[e])] | dense[int(comp.jj[e])]
    return h_dense, extent


def build_model(ground_set, obs: np.ndarray, table: ShapeTable, budget: int, *,
                index: np.ndarray | None = None, triangles=None,
                max_tree_coupling: int = 200_000_000, extent_cap: int = 0) -> Model:
    """Assemble the MILP.  ``triangles`` supplies (N3O) rows; ``None`` means none yet (lazy).

    ``extent_cap`` (0 = none) is F5's tractability lever.  A candidate block whose cycle spans more
    than ``extent_cap`` taxa **keeps its ``y`` variable** -- so it can still explain an (L)
    violation and the model stays feasible -- but contributes **no reticulate-coupling entries**,
    which is where 93% of the nonzeros live.  It is deliberately not a saturation fix: per §0.3 of
    the follow-up plan, even a cap of 10 leaves half the triples reticulate-eligible.
    """
    from opc.mpl.triples import triple_index

    n_taxa = ground_set.n_taxa
    words = ground_set.words
    n_clusters = int(words.shape[0])
    idx = triple_index(n_taxa) if index is None else np.asarray(index)
    n_triples = int(idx.shape[0])
    n_classes = len(CLASSES)

    comps = list(ground_set.components)
    # Column layout: [x | per-component z and y | u].
    x_offset = 0
    cursor = n_clusters
    comp_cols, y_cols = [], []
    for comp in comps:
        z_lo = cursor
        cursor += comp.n_z
        y_lo = cursor
        cursor += comp.n_hcand
        comp_cols.append((z_lo, y_lo))
        y_cols.extend(range(y_lo, y_lo + comp.n_hcand))
    u_offset = cursor
    n_cols = u_offset + n_triples * n_classes

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    ub: list[float] = []
    # The two coupling blocks are the big ones -- measured at n = 50, one component holds 8,278
    # candidate hybrids and 19,600 triples -- so their entries are accumulated as numpy blocks and
    # concatenated once.  Appending them element by element to the Python lists above costs both
    # the loop and roughly 70 bytes an entry, and at these sizes that is the whole build.
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_row(entries, bound):
        r = len(ub)
        for col, coef in entries:
            rows.append(r)
            cols.append(int(col))
            data.append(float(coef))
        ub.append(float(bound))

    # ---- feasibility, per component: exactly opc.solve.ilp's assignment formulation.
    for ci, (comp, (z_lo, y_lo)) in enumerate(zip(comps, comp_cols, strict=True)):
        gv = comp.vertices.astype(np.int64)  # local -> global cluster column
        key = comp.z_vertex.astype(np.int64) * (comp.n_hcand + 1) + comp.z_class.astype(np.int64)
        order = np.argsort(key, kind="stable")
        skey = key[order]

        def zcol(vertex: int, klass: int, _skey=skey, _z_lo=z_lo, _comp=comp) -> int:
            return _z_lo + int(np.searchsorted(_skey, vertex * (_comp.n_hcand + 1) + klass))

        for a in range(comp.n_vertices):
            sel = np.flatnonzero(comp.z_vertex == a)
            if sel.size:
                add_row([(z_lo + int(s), 1.0) for s in sel] + [(int(gv[a]), -1.0)], 0.0)
        for e in range(comp.n_edges):
            a, b, klass = int(comp.ii[e]), int(comp.jj[e]), int(comp.edge_class[e])
            for endpoint in (a, b):
                add_row([(int(gv[a]), 1.0), (int(gv[b]), 1.0), (zcol(endpoint, klass), -1.0)], 1.0)
        for s in range(comp.n_z):
            add_row([(z_lo + s, 1.0), (y_lo + int(comp.z_class[s]), -1.0)], 0.0)
        # ``comps.index(comp)`` would look natural here and is a trap: ``Component`` is a dataclass
        # of numpy arrays, so its ``__eq__`` returns arrays and ``index`` raises on the *second*
        # component -- invisible on any instance with only one, which at tau = 0 is most of them
        # (A33: the overlap graph is one giant component by CU = 1.0).
        if triangles is not None:
            for tri in (triangles[ci] if ci < len(triangles) else ()):
                a, b, c = (int(v) for v in tri)
                klass = int(comp.edge_class[np.flatnonzero(
                    ((comp.ii == a) & (comp.jj == b)) | ((comp.ii == b) & (comp.jj == a)))[0]])
                add_row([(int(gv[a]), 1.0), (int(gv[b]), 1.0), (int(gv[c]), 1.0),
                         (y_lo + klass, -1.0)], 1.0)

    # ---- budget
    add_row([(col, 1.0) for col in y_cols], budget)

    # ---- assignment: sum_s u = 1.  W <= 0 everywhere, so only the >= side can bind; the <= side
    #      is written anyway because a model whose feasible set depends on the objective's sign is
    #      a trap for the next person to change it.
    for t in range(n_triples):
        base = u_offset + t * n_classes
        add_row([(base + s, -1.0) for s in range(n_classes)], -1.0)
        add_row([(base + s, 1.0) for s in range(n_classes)], 1.0)

    # ---- tree coupling: a resolution needs a selected cluster that witnesses it.
    dense = bitset.to_dense(np.ascontiguousarray(words), n_taxa).astype(bool)
    a_i, b_i, c_i = idx[:, 0], idx[:, 1], idx[:, 2]
    pairs = ((a_i, b_i, c_i), (a_i, c_i, b_i), (b_i, c_i, a_i))
    n_tree_entries = 0
    for r, (p, q, out) in enumerate(pairs):
        witness = dense[:, p] & dense[:, q] & ~dense[:, out]  # [n_clusters, n_triples]
        n_tree_entries += int(witness.sum())
        if n_tree_entries > max_tree_coupling:
            raise RuntimeError(
                f"tree-coupling block needs > {max_tree_coupling:,} nonzeros; the ground set is "
                "too large for this formulation to be written out"
            )
        who, which = np.nonzero(witness)
        add_lo = len(ub)
        for t in range(n_triples):
            add_row([(u_offset + t * n_classes + r, 1.0)], 0.0)
        blocks.append((add_lo + which.astype(np.int64),
                       x_offset + who.astype(np.int64),
                       np.full(who.size, -1.0)))

    # ---- reticulate coupling: a signature class needs a candidate block realising it.
    n_ret_entries = 0
    sig_rows = {}
    for j in range(len(RET_SIGNATURES)):
        add_lo = len(ub)
        for t in range(n_triples):
            add_row([(u_offset + t * n_classes + RET_OFFSET + j, 1.0)], 0.0)
        sig_rows[j] = add_lo
    n_blocks_capped = 0
    for comp, (_, y_lo) in zip(comps, comp_cols, strict=True):
        h_dense, extent = hcand_extents(comp, words, n_taxa)
        for k in range(comp.n_hcand):
            if extent_cap and int(extent[k].sum()) > extent_cap:
                n_blocks_capped += 1
                continue
            in_h = h_dense[k]
            in_cycle = extent[k] & ~in_h
            a = in_h[a_i].astype(np.int8) + in_h[b_i] + in_h[c_i]
            b = in_cycle[a_i].astype(np.int8) + in_cycle[b_i] + in_cycle[c_i]
            for j, (sa, sb) in enumerate(RET_SIGNATURES):
                hit = np.flatnonzero((a == sa) & (b == sb))
                if not hit.size:
                    continue
                n_ret_entries += int(hit.size)
                blocks.append((sig_rows[j] + hit.astype(np.int64),
                               np.full(hit.size, y_lo + k, dtype=np.int64),
                               np.full(hit.size, -1.0)))

    all_rows = np.concatenate([np.asarray(rows, dtype=np.int64), *(r for r, _, _ in blocks)])
    all_cols = np.concatenate([np.asarray(cols, dtype=np.int64), *(c for _, c, _ in blocks)])
    all_data = np.concatenate([np.asarray(data, dtype=float), *(d for _, _, d in blocks)])
    a_ub = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(len(ub), n_cols)).tocsr()
    c = np.zeros(n_cols)
    c[u_offset:] = -table.w.reshape(-1)  # milp minimises
    integrality = np.zeros(n_cols)
    integrality[:u_offset] = 1.0  # x, z and y are binary; u is continuous and provably integral

    return Model(
        c=c, a_ub=a_ub, b_ub=np.asarray(ub, dtype=float), integrality=integrality,
        n_clusters=n_clusters, x_offset=x_offset, y_cols=np.asarray(y_cols, dtype=np.int64),
        u_offset=u_offset, n_triples=n_triples,
        hcand_words=np.vstack([c_.hcand for c_ in comps]) if comps
        else np.zeros((0, bitset.n_words(n_taxa)), np.uint64),
        sizes={
            "n_cols": n_cols, "n_rows": int(a_ub.shape[0]), "n_nonzeros": int(a_ub.nnz),
            "n_clusters": n_clusters, "n_hcand": len(y_cols),
            "n_tree_coupling_entries": n_tree_entries,
            "n_ret_coupling_entries": n_ret_entries,
            "n_components": len(comps),
            "extent_cap": int(extent_cap),
            "n_blocks_capped": int(n_blocks_capped),
        },
    )


def valid_upper_bound(res) -> float:
    """An upper bound on ``sup MPL`` that is a bound whether or not optimality was proved.

    ``solve_milp`` minimises ``c = -W``.  The solver's ``dual_bound`` is a *lower* bound on that
    minimum, so its negation is an *upper* bound on ``max sum W``, which is what §E4 wants.  At
    proven optimality dual and primal coincide and this returns exactly ``-res.objective``.

    Returns ``+inf`` when there is no dual bound -- the solver stopped before finishing the root
    relaxation, so nothing has been proved and the only honest bound is a vacuous one.  A number
    would be worse than infinity here: it would be believed.
    """
    if res.status == "infeasible":
        return -np.inf
    if res.dual_bound is None or not np.isfinite(res.dual_bound):
        return np.inf
    return -float(res.dual_bound)


def solve(model: Model, *, time_limit: float = 3600.0, gap_limit: float = 0.0,
          solver: str = "highs", tiebreak: np.ndarray | None = None,
          threads: int = 0, mem_limit: float = 0.0):
    """Run the MILP and return ``(result, upper_bound, primal_result)``.

    ``upper_bound`` is the objective -- the profiled optimum, which bounds ``sup MPL`` from above.
    Read the sign carefully: ``solve_milp`` minimises, so the bound is ``-res.objective``.

    **That is only a bound when the solve proved optimality**, and A45 measured how badly it fails
    when it did not.  ``-res.objective`` is the *incumbent*: for the maximisation this model
    expresses it is a **lower** bound on the ILP optimum, so at a time limit it can sit far below
    ``sup MPL`` rather than above it.  On A43's instance the 10 s and 60 s rungs returned
    -21,505,335 against a true optimum of -4,182,164 -- a confident, precise number **five times
    too low**, in the one direction that makes it not a bound at all.

    :func:`valid_upper_bound` is the quantity that is always safe: the *dual* bound, which brackets
    the optimum from the correct side whatever the solver had time to prove.  It equals this one at
    optimality, so nothing changes on the instances A42 published, and it is ``+inf`` when the
    solver never finished the root relaxation -- which is the honest answer and is what the 10 s
    rung deserved.

    **The cluster variables are massively degenerate, and ignoring that makes the reported gap
    meaningless.**  Nothing in the objective touches ``x``: every selection that merely *supports*
    the optimal shape assignment scores identically, so the solver returns an arbitrary one among
    a huge optimal face.  Measured at n = 5, the arbitrary selection realised a network with **no
    reticulations at all** and an MPL 450 log units below the optimum, while the bound itself sat
    0.04% above it.  Reported as "bound minus the value of the ILP's own solution" -- which is what
    the plan's §E4 asks for -- that reads as a 29% relaxation gap.  It is not one; it is a primal
    artefact.

    ``tiebreak`` fixes it lexicographically rather than with an epsilon: solve once for the bound,
    pin the objective at its optimum with one extra row, and re-solve maximising
    ``sum tiebreak[A] x[A]``.  Provably cannot move the bound, and it is what the returned
    ``primal_result`` is for.  Passing ``None`` keeps the arbitrary selection, so the artefact
    stays measurable.

    ``threads`` and ``mem_limit`` go to **both** calls.  The tiebreak re-solve is the same model
    with one extra row, so it is the same size and the same hazard; budgeting only the first would
    be budgeting half the run.
    """
    from opc.solve import backend

    res = backend.solve_milp(
        model.c, model.a_ub, model.b_ub, integrality=model.integrality,
        time_limit=time_limit, backend=solver, gap_limit=gap_limit,
        threads=threads, mem_limit=mem_limit,
    )
    bound = -res.objective if res.x is not None else float("nan")
    if tiebreak is None or res.x is None:
        return res, bound, res

    tol = abs(res.objective) * 1e-9 + 1e-6
    a2 = sp.vstack([model.a_ub, sp.csr_matrix(model.c.reshape(1, -1))]).tocsr()
    b2 = np.r_[model.b_ub, res.objective + tol]
    c2 = np.zeros_like(model.c)
    c2[model.x_offset:model.x_offset + model.n_clusters] = -np.asarray(tiebreak, dtype=float)
    second = backend.solve_milp(
        c2, a2, b2, integrality=model.integrality,
        time_limit=time_limit, backend=solver, gap_limit=gap_limit,
        threads=threads, mem_limit=mem_limit,
    )
    return res, bound, (second if second.x is not None else res)
