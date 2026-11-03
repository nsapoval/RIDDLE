"""The Luay (Nakhleh) network distance, its maximum, and the normalisation that follows.

``CMPNETS -m luay`` is the one topological number this repo reports beside cluster F1 (A80), and
it is a **raw unbounded integer**: it grows with the taxon count and with the reticulation count,
so ``luay = 30`` at n = 25 and ``luay = 30`` at n = 250 are not the same statement.  Nothing in
PhyloNet's documentation says what it is bounded by -- the documented page is a navigation shell --
so this module derives the bound from what the jar actually computes, and carries an independent
implementation of the metric so the derivation can be checked rather than believed.

What the metric is
------------------

Recovered from ``PhyloNet.jar`` with ``javap -c``, then confirmed against the jar itself.
``CMPNETS -m luay`` calls ``structs.network.NetworkMetricNakhleh.computeDistanceBetweenTwoNetworks``:

* **No reduction happens.**  ``reduceNetwork`` and ``computeReducedNetwork`` exist on that class
  and are *not* on this path -- ``computeDistanceTupleBetweenTwoNetworks`` calls
  ``computeEquivalentNodeSets`` and then ``computeDistance``, nothing else.  The two networks are
  compared exactly as parsed, degree-2 nodes and all.
* **Node equivalence is bottom-up sub-DAG isomorphism.**  A leaf's signature is ``{(itself, 0)}``
  and leaves are cross-matched between the two networks **by name**; an internal node's signature
  is the multiset ``{(rep(c), multiplicity) : c in children(v)}`` over its children's canonical
  representatives.  So two nodes are the same iff the whole rooted, leaf-labelled sub-DAG below
  them is isomorphic with unordered children.  That is **strictly stronger than cluster/RF
  matching**: sharing a clade's leaf set is not enough, the induced subnetwork has to be identical.
  It is why ``luay`` is large on cells whose cluster F1 is 0.6-0.8.
* **A reticulation node is counted ONCE PER PARENT.**  This is the whole surprise, and it is not
  visible in the arithmetic -- it is in ``Networks.postTraversal``, which walks with a per-node
  visited-child counter and pushes a node once for each edge into it.  The second time a
  reticulation is pushed its counter already equals its out-degree, so it is popped and
  **appended to the list a second time** without its subtree being re-walked.  The metric's
  multiset is therefore over that list, not over the node set: node ``v`` contributes
  ``max(1, indeg(v))`` copies of its class.  Reading it as one-copy-per-node reproduces the jar on
  only **37 of 168** real pairs; reading it this way reproduces it on **168 of 168**.
* **The value is a floored half-sum.**  ``d1 = sum_classes max(0, count1 - count2)``, ``d2``
  symmetrically, returned as ``(d1 + d2) / 2`` in **integer** arithmetic
  (``iadd; iconst_2; idiv; i2d``).  Every raw ``luay`` is an integer; the half-integers in the
  reports are medians over an even number of cells.

Equivalently ``luay = floor(|M1 sym-diff M2| / 2)``, with ``M_i`` the multiset of
sub-DAG-isomorphism classes over ``N_i``'s **traversal**, ``|M_i| = sum_v max(1, indeg(v))``.

The maximum
-----------

Leaves are free.  Both networks carry the same leaf label set, and a leaf's class can contain
nothing but that leaf -- a leaf's signature has child count ``0``, an internal node's has counts
>= 1, and a *unary* node above leaf ``x`` has signature ``("I", ((class(x), 1),))``, not ``x``'s.
So each leaf label contributes ``min(mult1, mult2) >= 1`` to the intersection and every other node
can in principle be unmatched:

    luay(N1, N2) <= floor((|M1| + |M2| - 2 * sum_x min(mult1(x), mult2(x))) / 2)

which is :func:`luay_max_exact`, assumption-free.  For a rooted **binary** network on ``n`` leaves
with ``k`` reticulations and no elementary nodes: tree-internal nodes ``t = n + k - 1``, so
``|E| = 2t + k = 2n + 3k - 2`` and ``|M| = 1 + |E| = 2n + 3k - 1``.  Every leaf has in-degree one,
so the shared term is exactly ``n``.  Substituting both sides:

    luay_max(n, k1, k2) = n - 1 + floor(3 * (k1 + k2) / 2)

**Three per reticulation, not two.**  A reticulation adds two nodes -- itself and the tree node
that donates to it -- and the reticulation node is then counted twice by the traversal.  Reading
it as two per reticulation (``n + k1 + k2 - 1``) is the natural guess and it is **wrong**: it was
exceeded on 58 of 168 measured pairs, by up to 2 at n = 6.  It happens to be right at
``k1 = k2 = 0``, where both formulas give ``n - 1``, which is exactly why a tree-only check cannot
catch the error.

Measured attainment: reached in 13 of 26 ``(n, k1, k2)`` cells on three draws apiece -- including
``n = 6, k = (2,2) -> 11``, ``n = 10, k = (3,3) -> 18`` and ``n = 25, k = (2,3) -> 31`` -- and
never exceeded.

Where the closed form is not exact
----------------------------------

======================================= ====================== ==============================
case                                    ``|M|``                effect
======================================= ====================== ==============================
binary, no elementary nodes             ``2n + 3k - 1``        exact
CAMUS truths after ``_prune_outgroup``  ``2n + 3k``            closed form may be exceeded
``closure.network_from_hasse`` output   ``< 2n + 3k - 1``      closed form loose (polytomies)
different leaf label sets               --                     **formula invalid**; leaves stop
                                                               matching and the shared term
                                                               collapses
``n = 2``                               --                     formula gives 1; only one binary
                                                               tree exists, so the true max is 0
======================================= ====================== ==============================

Outgroup pruning leaves a residual degree-2 node and PhyloNet does **not** suppress it -- pinned
independently by ``tests/test_phylonet_cmpnets.py::test_a_zero_length_donor_edge_is_not_collapsed``.
That is the whole reason two normalised columns exist rather than one: ``luay_norm`` uses the
closed form and is comparable across rows but can exceed 1, while ``luay_norm_exact`` uses the
traversal sizes of the actual pair and cannot.  **Neither is ever clipped** -- a ratio above 1 is
a fact about the input, and clipping it would hide the only signal that says so.

This module must not import :mod:`opc.mpl.phylonet`.  It is the independent implementation that
the JVM's answer is checked against, per the repo's cross-validation convention, and an
implementation that shares code with the thing it validates validates nothing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = [
    "Dag",
    "parse_enewick",
    "as_dag",
    "indegrees",
    "traversal_mass",
    "internal_nodes",
    "leaf_mass",
    "class_counts",
    "nakhleh_terms",
    "nakhleh_distance",
    "luay_max",
    "luay_max_exact",
    "luay_max_from_mass",
    "norm_name",
    "normalised_luay",
    "k_pred_fallback",
    "add_normalised_columns",
    "LUAY_COLUMNS",
]

#: The raw distance columns this module knows how to normalise.  ``luay`` is the arm,
#: ``luay_greedy`` the baseline every delta is taken against, ``luay_ceiling`` W12's pool-best.
#: A normalised name is the raw name with ``_norm`` spliced in after ``luay``, so
#: ``luay_greedy`` -> ``luay_norm_greedy`` -- which is what the drivers write, so a report cannot
#: tell whether a column was computed at evaluation time or derived at report time.
LUAY_COLUMNS = ("luay", "luay_greedy", "luay_ceiling")

_COMMENT = re.compile(r"\[[^\]]*\]")


@dataclass
class Dag:
    """A rooted DAG with leaf labels -- the shape PhyloNet's reader hands the metric.

    Deliberately weaker than :class:`opc.simulate.networks.Network`: any in-degree, any
    out-degree, polytomies, elementary nodes and hybrid leaves are all representable, because
    every one of them occurs in a string the jar accepts and ``Network`` refuses at least three
    of them.  ``children[v]`` is a list rather than a set: a node may have the same child twice,
    and that multiplicity is part of the signature the metric compares.

    Nothing here records a branch length or an inheritance probability, because the metric cannot
    see either (A80, and ``phylonet.strip_gamma`` removes gamma before the JVM is started).
    """

    children: list[list[int]] = field(default_factory=list)
    name: list[str | None] = field(default_factory=list)
    root: int = 0

    def __len__(self) -> int:
        return len(self.children)

    def is_leaf(self, v: int) -> bool:
        return not self.children[v]


def parse_enewick(text: str) -> Dag:
    """Extended Newick -> :class:`Dag`, mirroring PhyloNet's ``ExNewickReader``.

    Deliberately **not** :func:`opc.simulate.networks.from_enewick`, which claims only to be the
    inverse of this repo's own ``to_enewick`` and raises on a node with more than two parents, on
    a donor that feeds two hybrids, on a hybrid carrying a subtree at both occurrences, and on a
    hybrid at the root.  A102 already records three camus truths whose size-2 cycle ``to_enewick``
    writes and ``from_enewick`` cannot read back -- **and PhyloNet accepts the string**.  A
    normaliser that raised where the jar returns a number would put NaN on exactly the rows the
    caveats are about.

    Three things this keeps that a phylogenetics-minded parser would throw away, all of which the
    metric counts:

    * **elementary (out-degree-1) nodes**, because outgroup pruning leaves one behind;
    * **repeated children**, because a node's signature is a multiset;
    * **every ``#Hk`` occurrence merged onto one node**, children taken from whichever occurrence
      carries them -- that is what makes a reticulation one node with two parents rather than two
      nodes, and it is what gives it in-degree 2, which is what the traversal counts twice.
    """
    s = _COMMENT.sub("", text).strip()
    if s.endswith(";"):
        s = s[:-1]
    s = s.strip()
    if not s:
        raise ValueError("empty eNewick")

    dag = Dag()
    hybrids: dict[str, int] = {}
    pos = 0

    def new_node(nm: str | None, kids: list[int]) -> int:
        dag.children.append(kids)
        dag.name.append(nm)
        return len(dag.children) - 1

    def parse_node() -> int:
        nonlocal pos
        kids: list[int] = []
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                kids.append(parse_node())
                if pos >= len(s):
                    raise ValueError("unbalanced '(' in eNewick")
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"unexpected {s[pos]!r} at {pos} in eNewick")
        start = pos
        while pos < len(s) and s[pos] not in ",()":
            pos += 1
        label = s[start:pos].split(":", 1)[0].strip().strip("'\"")
        tag = None
        if "#" in label:
            label, _, rest = label.partition("#")
            tag = "#" + rest
            label = label.strip()
        nm = label or None
        if tag is None:
            return new_node(nm, kids)
        if tag in hybrids:
            v = hybrids[tag]
            if kids:
                dag.children[v].extend(kids)
            if nm and dag.name[v] is None:
                dag.name[v] = nm
            return v
        v = new_node(nm, kids)
        hybrids[tag] = v
        return v

    dag.root = parse_node()
    if pos != len(s):
        raise ValueError(f"trailing {s[pos:]!r} in eNewick")
    # A hybrid written bare at every occurrence ends up childless, i.e. a LEAF -- and leaves are
    # matched across the two networks by name, so it needs one.  Naming it eagerly at parse time
    # would instead put a name on every *internal* hybrid, which is harmless to the distance but
    # makes `leaf_mass` and every name-based assertion report a taxon that does not exist.
    for tag, v in hybrids.items():
        if not dag.children[v] and dag.name[v] is None:
            dag.name[v] = tag
    return dag


def as_dag(x) -> Dag:
    """Accept an eNewick string, a :class:`Dag`, or an :class:`opc.simulate.networks.Network`.

    A ``Network`` is routed **through its own ``to_enewick``** rather than read field by field.
    That is not laziness: ``to_enewick`` is what PhyloNet is actually handed, and it is not a
    transparent encoding -- a hybrid that is a leaf comes out as ``(label:0)#Hk``, which is one
    node in the ``Network`` and two in the string.  Measuring the object would measure a network
    nobody compares.  Measured at ``n25_rep91_r3``: 27 non-leaf nodes in the object, 30 in the
    string, and the difference is exactly its three singleton hybrids.
    """
    if isinstance(x, Dag):
        return x
    if isinstance(x, str):
        return parse_enewick(x)
    from opc.simulate import networks as NW  # local: opc.mpl need not depend on the simulator

    return parse_enewick(NW.to_enewick(x))


def indegrees(dag: Dag) -> list[int]:
    """In-degree per node.  The root's is 0; everything else is at least 1."""
    d = [0] * len(dag)
    for v in range(len(dag)):
        for c in dag.children[v]:
            d[c] += 1
    return d


def traversal_mass(x) -> int:
    """``|M|`` -- how many entries a network contributes to the compared multiset.

    ``sum_v max(1, indeg(v))``, because ``Networks.postTraversal`` appends a node once per edge
    into it (and the root once).  This, not the node count, is what the bound is stated in.
    """
    dag = as_dag(x)
    return sum(max(1, d) for d in indegrees(dag))


def internal_nodes(x) -> int:
    """The number of non-leaf nodes.  Diagnostic only -- the bound is stated in `traversal_mass`.

    Kept because it is the quantity the *node-count* reading of the metric would have used, and
    ``tests/test_netdist.py`` pins the identity ``I == n + 2k - 1`` on a generator network with
    it.  A change to ``to_enewick`` that silently altered the node count would show up there.
    """
    dag = as_dag(x)
    return sum(1 for v in range(len(dag)) if dag.children[v])


def leaf_mass(x) -> dict[str | None, int]:
    """``{leaf label: max(1, indeg)}`` -- what each leaf contributes to its own class.

    Normally 1.  It is 2 for a leaf that is itself a reticulation, which this repo's
    ``to_enewick`` never writes (it expands those to ``(label:0)#Hk``) but a third-party string
    may.
    """
    dag = as_dag(x)
    deg = indegrees(dag)
    return {dag.name[v]: max(1, deg[v]) for v in range(len(dag)) if not dag.children[v]}


def _postorder(dag: Dag) -> list[int]:
    """Children before parents, each node **once**.

    Not a reimplementation of ``postTraversal``: this exists to compute signatures, where visiting
    a node twice is wasted work and nothing more.  The traversal's *multiplicity* enters in
    :func:`class_counts`, as ``max(1, indeg)``, which is where it belongs -- keeping the two apart
    is what made the discrepancy diagnosable at all.  Iterative, because a reconstructed n = 250
    network is deep enough to blow the recursion limit.
    """
    seen = [0] * len(dag)
    order: list[int] = []
    stack = [(dag.root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        if seen[v]:
            continue
        seen[v] = 1
        stack.append((v, True))
        for c in dag.children[v]:
            if not seen[c]:
                stack.append((c, False))
    return order


def class_counts(dags, *, table: dict | None = None) -> list[dict[int, int]]:
    """Per network, the multiset of sub-DAG-isomorphism classes over its **traversal**.

    Hash-consing: a node's canonical form is ``("L", name)`` for a leaf and
    ``("I", sorted (child class, multiplicity) pairs)`` otherwise, and each distinct canonical form
    is interned to an integer once, **in a table shared by both networks** -- which is what makes
    the two counters comparable.  PhyloNet reaches the same equivalence by a different route (it
    interns within network 1, again within network 2, then translates the second's representatives
    into the first's); ``tests/test_netdist.py`` is what says the two routes agree.

    A node contributes ``max(1, indeg(v))`` copies of its class, not one.  See the module
    docstring: one-copy-per-node reproduces the jar on 37 of 168 real pairs, this on 168 of 168.
    """
    table = {} if table is None else table
    out = []
    for dag in dags:
        cls = [-1] * len(dag)
        for v in _postorder(dag):
            if dag.children[v]:
                kids: dict[int, int] = {}
                for c in dag.children[v]:
                    kids[cls[c]] = kids.get(cls[c], 0) + 1
                key = ("I", tuple(sorted(kids.items())))
            else:
                # A leaf matches across networks by NAME, which is what PhyloNet's name-keyed map
                # does.  An unnamed leaf can therefore only ever match another unnamed leaf.
                key = ("L", dag.name[v])
            cid = table.get(key)
            if cid is None:
                cid = len(table)
                table[key] = cid
            cls[v] = cid
        deg = indegrees(dag)
        counts: dict[int, int] = {}
        for v in range(len(dag)):
            counts[cls[v]] = counts.get(cls[v], 0) + max(1, deg[v])
        out.append(counts)
    return out


def nakhleh_terms(a, b) -> tuple[int, int]:
    """``(d1, d2)`` -- the two one-sided counts, **before** the integer halving.

    Exported separately from :func:`nakhleh_distance` because ``(d1 + d2) // 2`` destroys
    information, and when a differential test against the jar disagrees these are what say
    *which side* is wrong.  A test that only ever sees the halved number cannot tell a bad
    traversal from a bad halving.
    """
    table: dict = {}
    c1, c2 = class_counts([as_dag(a), as_dag(b)], table=table)
    d1 = sum(max(0, n - c2.get(k, 0)) for k, n in c1.items())
    d2 = sum(max(0, n - c1.get(k, 0)) for k, n in c2.items())
    return d1, d2


def nakhleh_distance(a, b) -> int:
    """The reference ``luay``: ``floor((d1 + d2) / 2)``.

    Integer division, matching the jar's ``iadd; iconst_2; idiv`` -- an odd symmetric difference
    truncates **down**, so a normalised value of exactly 1.0 needs ``d1 + d2`` in ``{2M, 2M+1}``.
    """
    d1, d2 = nakhleh_terms(a, b)
    return (d1 + d2) // 2


def luay_max(n_taxa: int, k1: int, k2: int) -> int:
    """``n - 1 + floor(3 * (k1 + k2) / 2)`` -- the closed-form maximum.

    For two rooted **binary** networks on a shared leaf label set with no elementary nodes.  Still
    a valid upper bound on a polytomous network, merely loose; **not** valid when either network
    carries an elementary node, which outgroup pruning leaves behind -- use :func:`luay_max_exact`
    there.

    **Three per reticulation, not two.**  A reticulation adds two nodes (itself and the tree node
    that donates to it) and the traversal then counts the reticulation node twice.  The natural
    guess ``n + k1 + k2 - 1`` is wrong and was measured wrong on 58 of 168 pairs; it coincides
    with this at ``k1 = k2 = 0``, which is exactly why a tree-only check cannot catch it.

    At ``n = 2`` it returns 1 where the true maximum is 0, because there is only one binary tree
    on two taxa -- a small-``n`` artifact, not a defect in the bound.
    """
    if n_taxa < 1 or k1 < 0 or k2 < 0:
        raise ValueError(f"n_taxa={n_taxa}, k1={k1}, k2={k2}: need n >= 1 and k >= 0")
    return max(0, n_taxa - 1 + (3 * (k1 + k2)) // 2)


def luay_max_from_mass(mass1: int, mass2: int, shared: int) -> int:
    """``floor((|M1| + |M2| - 2 * shared) / 2)`` -- the bound in its numeric form."""
    if min(mass1, mass2, shared) < 0:
        raise ValueError(f"mass1={mass1}, mass2={mass2}, shared={shared}: all must be >= 0")
    return max(0, (mass1 + mass2 - 2 * shared) // 2)


def luay_max_exact(a, b) -> int:
    """The assumption-free maximum for *this* pair, from the two parsed traversals.

    Holds for any two networks on a shared leaf label set, whatever their degrees, so it is what a
    normalised value that must not exceed 1 has to be divided by.  The shared term is
    ``sum_x min(mult1(x), mult2(x))`` over leaf labels present in both -- exact rather than ``n``,
    because a leaf that is itself a reticulation contributes twice.
    """
    da, db = as_dag(a), as_dag(b)
    la, lb = leaf_mass(da), leaf_mass(db)
    shared = sum(min(m, lb[x]) for x, m in la.items() if x in lb)
    return luay_max_from_mass(traversal_mass(da), traversal_mass(db), shared)


def normalised_luay(luay, *, n_taxa=None, k1=None, k2=None, pair=None):
    """``(luay_norm, luay_norm_exact)``, each ``nan`` when its inputs are absent.

    ``pair`` is the two networks (eNewick strings, ``Dag`` or ``Network`` objects) for the exact
    denominator; omit it and only the closed form comes back.

    **Never clipped.**  A value above 1 means the input broke an assumption of the closed form --
    almost always an elementary node that outgroup pruning left behind -- and that is worth
    seeing rather than hiding.
    """
    nan = float("nan")
    if luay is None or (isinstance(luay, float) and math.isnan(luay)):
        return nan, nan
    closed = nan
    if n_taxa is not None and k1 is not None and k2 is not None:
        d = luay_max(int(n_taxa), int(k1), int(k2))
        closed = float(luay) / d if d > 0 else nan
    exact = nan
    if pair is not None:
        d = luay_max_exact(*pair)
        exact = float(luay) / d if d > 0 else nan
    return closed, exact


def k_pred_fallback(k_pred, k_true):
    """``(k_used, source)`` -- substitute the budget for a missing prediction count.

    ``k_pred`` is absent from essentially every landed RL parquet: 0 of 104 ``rl_gnn_eval_b25``
    files, 0 of 104 at each of b50/b100/b250, 0 of 6 ``rl_swap_eval_*``, 0 of 18
    ``rl_beam_eval_b25``, present only on the camus third-party parquets and the recent
    ``rl_gnn_eval_cam*``.  It cannot be recovered from a row.

    But every eval driver builds ``ClusterEnv(inst.gs, budget=inst.k)``
    (``w10_gnn_eval.py:242``, ``w9_swap_eval.py:367``), so ``k_pred <= k`` always holds for an
    RL-arm prediction.  Substituting ``k`` is therefore a provable **upper** bound on the
    denominator, which makes the ratio a conservative **lower** bound on the true ``luay_norm``.

    ``source`` is ``"measured"`` or ``"budget"`` per row rather than a note in a caption, because
    the two must never be pooled inside one paired test: once new runs start landing with
    ``k_pred``, a pairing that mixes them reports the fallback's bias as a method difference.
    That is A94's defect one level up.
    """
    import numpy as np

    kt = np.asarray(k_true, dtype=float)
    if k_pred is None:
        return kt, np.full(kt.shape, "budget", dtype=object)
    kp = np.asarray(k_pred, dtype=float)
    bad = ~np.isfinite(kp) | (kp < 0)
    used = np.where(bad, kt, kp)
    return used, np.where(bad, "budget", "measured").astype(object)


def norm_name(col: str, *, exact: bool = False) -> str:
    """``luay_greedy`` -> ``luay_norm_greedy`` (or ``luay_norm_exact_greedy``)."""
    stem = "luay_norm_exact" if exact else "luay_norm"
    if col == "luay":
        return stem
    if not col.startswith("luay_"):
        raise ValueError(f"{col!r} is not a luay column")
    return f"{stem}_{col[len('luay_'):]}"


def add_normalised_columns(df, columns=None, *, k_true=None, k_pred=None, inplace=False):
    """Derive ``luay_norm*`` on an evaluation frame, so a landed parquet needs no re-run.

    ``columns`` defaults to whichever of :data:`LUAY_COLUMNS` the frame carries; ``k_true``
    defaults to ``k`` then ``k_true``; ``k_pred`` is a per-column mapping defaulting to
    ``luay -> k_pred``, ``luay_greedy -> k_pred_greedy``, ``luay_ceiling -> k_pred_ceiling``.
    Where a driver also wrote the traversal masses (``mass_truth`` and ``mass_pred``/
    ``mass_greedy``/``mass_ceiling``) the exact column is derived too; on a landed parquet those
    are absent and only the closed form appears.

    Raw columns are never touched -- this only adds.  Returns ``(frame, audit)``, the audit being
    what a report prints in its provenance block rather than leaving to a reader to notice.
    """
    import numpy as np

    out = df if inplace else df.copy()
    cols = [c for c in (columns or LUAY_COLUMNS) if c in out.columns]
    audit = {"rows": int(len(out)), "columns": cols, "budget_fallback": 0,
             "n_norm_gt1": 0, "max_norm": float("nan")}
    kt_col = k_true or next((c for c in ("k", "k_true") if c in out.columns), None)
    if not cols or kt_col is None or "n" not in out.columns:
        return out, audit

    kp_map = dict(k_pred or {c: ("k_pred" if c == "luay" else "k_pred" + c[len("luay"):])
                              for c in cols})
    mass_map = {c: ("mass_pred" if c == "luay" else "mass" + c[len("luay"):]) for c in cols}
    worst = float("nan")
    for c in cols:
        src = kp_map.get(c)
        used, source = k_pred_fallback(
            out[src] if src in out.columns else None, out[kt_col])
        out[f"{norm_name(c)}_k_source"] = source
        audit["budget_fallback"] += int((source == "budget").sum())
        denom = out["n"].to_numpy(float) - 1.0 + np.floor(
            3.0 * (out[kt_col].to_numpy(float) + used) / 2.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(denom > 0, out[c].to_numpy(float) / denom, np.nan)
        out[norm_name(c)] = val
        finite = val[np.isfinite(val)]
        if finite.size:
            audit["n_norm_gt1"] += int((finite > 1.0).sum())
            worst = float(np.nanmax([worst, finite.max()]))
        m1, m2 = "mass_truth", mass_map[c]
        if m1 in out.columns and m2 in out.columns:
            # The exact denominator subtracts the SHARED leaf mass, which is ``n`` whenever no
            # leaf is itself a reticulation -- true of everything ``to_enewick`` writes.  A driver
            # that ever meets a hybrid leaf must write the shared mass instead of assuming ``n``.
            d = (out[m1].to_numpy(float) + out[m2].to_numpy(float)
                 - 2.0 * out["n"].to_numpy(float)) // 2.0
            with np.errstate(invalid="ignore", divide="ignore"):
                out[norm_name(c, exact=True)] = np.where(
                    d > 0, out[c].to_numpy(float) / d, np.nan)
    audit["max_norm"] = worst

    if "luay_greedy" in cols:
        for c in cols:
            if c == "luay_greedy":
                continue
            for exact in (False, True):
                base, here = norm_name("luay_greedy", exact=exact), norm_name(c, exact=exact)
                if base in out.columns and here in out.columns:
                    # **Not proportional to the raw delta**, and that is the point rather than a
                    # wart: each term is divided by its own network's maximum, so this is a
                    # difference of two "fraction of what could have gone wrong" numbers.  It can
                    # disagree in SIGN with ``luay_vs_greedy`` -- a decode two edges further from
                    # the truth on one fewer reticulation improves this and worsens that.
                    out[f"{here}_vs_greedy"] = out[here] - out[base]
    return out, audit
