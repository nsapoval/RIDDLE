"""Gene-tree triplet concordance inside a block's polytomy -- the external signal for a swap.

The trees are the ones the GNN saw: the cell's estimated gene trees, rooted as the landed run rooted
them, cut at the same ``m``.  :func:`estimated_trees` is ``cluster_anatomy/scoring.build_cell``
lines 109-121, **copied** -- ``scoring.py`` imports torch and ``run_gnn_cpu`` (:38, :40).  Gate N0
recounts every candidate's support from the file this writes and requires the cached ``support``.

``tc(Y)`` for an allowed cluster ``Y`` (a union of atoms of ``P_b``) is the mean, over atom triples
``(i, j, k)`` with ``i < j`` atoms of ``Y`` and ``k`` an atom of ``P_b`` outside it, of the share of
resolved gene-tree triples ``a b | c`` (``a in A_i, b in A_j, c in A_k``) among the three resolutions.

* **Per atom triple**, so a large atom does not outvote the rest.
* **Normalised by resolved triples**, because a multifurcating gene tree contributes to none of
  the three (``opc/mpl/triples.py:27-34``).  On a three-atom block the three options' ``tc``
  therefore sum to 1 wherever a triple is resolved at all -- gate N3's second half.
* Under the multispecies coalescent on a tree, a true cluster's ``tc`` is ``1 - 2/3 e^{-t}`` and its
  two NNI rivals' ``1/3 e^{-t}`` each; 1/3 is the no-signal value.

**Partial trees** (a dataset with ``extra["partial"]``: ``taxmiss_eval``'s conditions, 2026-10-21).
Every counter here took *outside a node* to mean *not below it*, so on a tree missing taxa it
counted ``ab|c`` for each absent ``c`` -- at the root too, whose guard never fires because the absent
taxa make ``outside`` non-empty (``fungi/README.md`` measured it at 10-20% of the total, and
:func:`slow_atom_tensor` shared the blind spot, so gate N3 could not see it).  With ``partial=True``
each counter takes outside to mean *in this tree and not below the node*, which is
``fungi/scripts/fz_triples.py``'s definition (ab|c iff |lca(a,b)| < |lca(a,c)| over taxa the tree
carries).  On a full-taxon tree the two are the same thing, bit for bit (``taxmiss_eval/check.py``
P4); ``features.check_tc``'s N3'(c) holds the difference to exactly the absent-``c`` triples.  Off
by default, so every published feat file re-derives unchanged.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from opc.clusters import extract
from opc.mpl import triples
from opc.rl import camus as CAMUS

import ne_common as NE

M, TAU = 1000, 2          # run_gnn_cpu.py:190; a dataset's ``extra`` overrides them (n1000)


def estimated_trees(ds: str, row: dict, work_root: Path, taxa: list[str]) -> tuple[Path, Path, int]:
    """Write the cell's rooted gene trees, as the landed run read them, to a gzip.

    Returns ``(gz, workdir, m)``; the caller removes ``workdir``.  ``taxa`` must be the truth's own
    order (the cell JSON's), which fixes bit ``t``.
    """
    D = NE.CA.DATASETS[ds]
    cell = CAMUS.CamusCell(row["phase"], int(row["rep"]), row["arm"], D.inputs)
    truth = CAMUS.truth_of(cell)
    if list(truth.labels) != list(taxa):
        raise AssertionError(f"{row['phase']} rep {row['rep']}: the truth's taxon order is not the cell's")
    Path(work_root).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{row['phase']}_rep{int(row['rep']):02d}_", dir=work_root))
    gz = work / "trees.nwk.gz"
    Mx = int(D.extra.get("m", M))       # n1000: 10,000 (cluster_anatomy/common.py)
    info = CAMUS.prepare_gene_trees(cell, gz, list(taxa), max_loci=Mx,
                                    partial=int(D.extra.get("partial", 0)))
    m = min(Mx, int(info["n_loci"])) if D.m_from_trees else Mx
    return gz, work, m


def partial_of(ds: str) -> bool:
    """Whether ``ds`` reads trees missing taxa (``cluster_anatomy/common.py``: the taxmiss datasets)."""
    return bool(NE.CA.DATASETS[ds].extra.get("partial", 0))


def support_parity(gz: Path, taxa: list[str], m: int, masks: list[int], support: np.ndarray,
                   partial: bool = False) -> int:
    """How many candidates' support, recounted from ``gz`` at ``m``, differs from the cache.  A
    partial tree's own leaf set is dropped, as ``score_cells`` dropped it (``run_gnn_cpu --partial``)."""
    ti = {t: i for i, t in enumerate(taxa)}
    counter = extract.count_newick_file(gz, ti, [m], drop_partial_root=partial)
    got, sup, _ = counter.finalize(min_support=1)
    table = {int(k): int(s) for k, s in zip(got, sup[:, -1])}
    return sum(1 for mk, s in zip(masks, support) if table.get(mk, 0) != int(s))


def triple_counts(gz: Path, taxa: list[str], m: int, partial: bool = False) -> triples.TripleCounts:
    ti = {t: i for i, t in enumerate(taxa)}
    if partial:
        return partial_triple_counts(gz, ti, m)
    return triples.count_newick_file(gz, ti, max_loci=m)


def partial_triple_counts(gz: Path, ti: dict[str, int], m: int) -> triples.TripleCounts:
    """``triples.count_newick_file`` with ``TripleCounts.add_tree``'s loop restated for trees missing
    taxa: ``outside`` is the taxa **this tree carries** and not below the node (module docstring),
    and a tree is non-binary against its own taxon count.  On a full-taxon tree every line below is
    the published one's."""
    import gzip as _gzip
    n = len(ti)
    acc = triples.TripleCounts.empty(n)
    opener = _gzip.open if Path(gz).suffix == ".gz" else open
    with opener(gz, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            nodes = triples.child_leaf_sets(line, ti)
            present = np.zeros(n, dtype=bool)
            for k in nodes[-1]:                 # post-order: the root closes last
                present[k] = True
            n_internal = 0
            for kids in nodes:
                n_internal += 1
                if len(kids) < 2:
                    continue
                below = np.zeros(n, dtype=bool)
                for k in kids:
                    below[k] = True
                outside = np.flatnonzero(present & ~below)
                if outside.size == 0:
                    continue  # the root: no third taxon to be an outgroup
                for i in range(len(kids)):
                    for j in range(i + 1, len(kids)):
                        a, b = np.asarray(kids[i]), np.asarray(kids[j])
                        acc.count[np.ix_(a, b, outside)] += 1
                        acc.count[np.ix_(b, a, outside)] += 1
            acc.n_loci += 1
            acc.n_nonbinary_loci += n_internal != int(present.sum()) - 1
            if m and acc.n_loci >= m:
                break
    return acc


def atom_tensor(count: np.ndarray, atoms: list[int]) -> np.ndarray:
    """``T[i, j, k] = sum count[A_i, A_j, A_k]``: loci-triples ``ab|c`` summed over three atoms."""
    m = len(atoms)
    idx = []
    owner = []
    for i, a in enumerate(atoms):
        for t in _bits(a):
            idx.append(t)
            owner.append(i)
    idx = np.asarray(idx)
    G = np.zeros((idx.size, m))
    G[np.arange(idx.size), owner] = 1.0
    C = count[np.ix_(idx, idx, idx)].astype(np.float64)
    T = np.tensordot(C, G, axes=([2], [0]))                     # (p, p, m)
    T = np.tensordot(T, G, axes=([1], [0]))                     # (p, m_k, m_j)
    T = np.tensordot(G, T, axes=([0], [0]))                     # (m_i, m_k, m_j)
    return T.transpose(0, 2, 1)                                 # (m_i, m_j, m_k)


def block_atom_tensors(gz: Path, taxa: list[str], m: int, atoms_list: list[list[int]],
                       chunk_entries: int = 20_000_000, _mutant: str = "", partial: bool = False) -> dict:
    """:func:`atom_tensor` of the dense count, for every block in ``atoms_list``, without the dense
    ``int32[n, n, n]`` tensor -- 4 GB at n = 1000, and ``C(n, 3)`` increments per tree.

    ``triples.TripleCounts.add_tree`` adds ``ab|c`` for every node ``v``, every ordered pair ``(a, b)``
    under two different children of ``v``, and every ``c`` outside ``v`` -- outside meaning *not below
    v*, so a taxon missing from the tree counts as outside.  Summed over atoms ``A_i, A_j, A_k``:

    * the ordered pairs under two different children of ``v`` are ``Pair_v = S_v S_v^T - sum_u c_u c_u^T``,
      where ``S_v`` counts each atom's members below ``v`` and ``c_u`` below its child ``u``;
    * the outside members are ``Out_v = |A| - S_v``.

    So ``T = sum_v S_v x S_v x Out_v - sum_u c_u x c_u x Out_parent(u)``.  Only nodes where at least two
    children hold an atom member are needed (``Pair_v`` is 0 at every other node), and that is at most
    ``|P_b| - 1`` nodes per tree.  The cost is ``O(nodes x sum m_b)`` per tree for the counts, plus
    ``O(m_b^3)`` per needed node -- nothing in ``n^3``.

    Trees are parsed with the dense path's own tokenizer, in post-order, so every subtree is the
    contiguous id range ``[first(v), v]`` and ``S`` is one cumulative sum.  Integer-valued float64
    throughout, so the result is exact (well below 2^53) and equals the dense path bit for bit; gate
    G-TC (``check_nni``) holds it to that.

    Returns ``{"T": [m_b x m_b x m_b float64], "n_loci", "n_nonbinary_loci"}``.  ``_mutant`` exists
    for gate G-TC alone (``check_nni.gate_tc``): each value breaks one term, and the fixture must
    catch every one.

    ``partial`` replaces ``|A|`` by each tree's own atom counts, ``S`` at its root: ``Out_v = S_root -
    S_v`` is "carried by this tree and not below v" (module docstring), which is ``|A| - S_v`` again
    on a full-taxon tree.
    """
    import gzip as _gzip
    n = len(taxa)
    ti = {t: i for i, t in enumerate(taxa)}
    sizes = [len(a) for a in atoms_list]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    C = int(off[-1])
    colmap = np.zeros((n, max(C, 1)), dtype=np.int32)
    for b, atoms in enumerate(atoms_list):
        for i, a in enumerate(atoms):
            for t in _bits(a):
                if colmap[t, off[b]:off[b + 1]].any():
                    raise ValueError(f"block {b}: atoms overlap at taxon {t}")
                colmap[t, off[b] + i] = 1
    A = colmap.sum(axis=0).astype(np.float64)
    T = [np.zeros((s, s, s)) for s in sizes]
    n_loci = n_nonbinary = 0

    leaf_tax: list[int] = []
    first: list[int] = []
    parent: list[int] = []
    root_of: list[int] = []         # partial: each node's tree root, as an index into this chunk

    def flush() -> None:
        if not leaf_tax:
            return
        lt = np.asarray(leaf_tax, dtype=np.int64)
        fi = np.asarray(first, dtype=np.int64)
        pa = np.asarray(parent, dtype=np.int64)
        N = lt.size
        L = np.zeros((N, colmap.shape[1]), dtype=np.int32)
        is_leaf = lt >= 0
        L[is_leaf] = colmap[lt[is_leaf]]
        CL = np.zeros((N + 1, colmap.shape[1]), dtype=np.int32)
        np.cumsum(L, axis=0, out=CL[1:])
        S = CL[np.arange(N) + 1] - CL[fi]
        has_par = pa >= 0
        ro = np.asarray(root_of, dtype=np.int64) if partial else None
        for b in range(len(sizes)):
            Sb = S[:, off[b]:off[b + 1]]
            occ = Sb.any(axis=1)
            kids = np.flatnonzero(occ & has_par)
            cnt = np.bincount(pa[kids], minlength=N)
            rel = np.flatnonzero(cnt >= 2)
            if _mutant == "skip_root":
                rel = rel[pa[rel] >= 0]
            if not rel.size:
                continue
            neg = kids[np.isin(pa[kids], rel)]
            mb = sizes[b]
            Ab = A[off[b]:off[b + 1]]
            sp = Sb[rel].astype(np.float64)
            sn = Sb[neg].astype(np.float64)
            if partial:
                Pb = Sb[ro].astype(np.float64)          # each node's tree: its atom members present
                op = Pb[rel] - sp
                on = Pb[neg] - Sb[pa[neg]].astype(np.float64)
            else:
                op = Ab - sp
                on = Ab - Sb[pa[neg]].astype(np.float64)
            Xp = (sp[:, :, None] * sp[:, None, :]).reshape(len(rel), mb * mb)
            Xn = (sn[:, :, None] * sn[:, None, :]).reshape(len(neg), mb * mb)
            add = (Xp.T @ op - (0.0 if _mutant == "no_children" else Xn.T @ on)).reshape(mb, mb, mb)
            if _mutant == "diag_once":
                add[np.arange(mb), np.arange(mb), :] *= 0.5
            T[b] += add
        leaf_tax.clear()
        first.clear()
        parent.clear()
        root_of.clear()

    opener = _gzip.open if Path(gz).suffix == ".gz" else open
    per_tree = 2 * n * max(C, 1)
    with opener(gz, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stack: list[list[int]] = [[]]
            n_internal = n_leaves = 0
            start = len(leaf_tax)
            for tok in triples._SPLIT.split(triples._STRIP.sub("", line)):
                if not tok or tok == ",":
                    continue
                if tok == "(":
                    stack.append([])
                elif tok == ")":
                    kids_ = stack.pop()
                    v = len(leaf_tax)
                    leaf_tax.append(-1)
                    parent.append(-1)
                    first.append(first[kids_[0]] if kids_ else v)
                    for k in kids_:
                        parent[k] = v
                    stack[-1].append(v)
                    n_internal += 1
                else:
                    name = tok.strip().rstrip(";").strip()
                    if name:
                        v = len(leaf_tax)
                        leaf_tax.append(ti[name])
                        parent.append(-1)
                        first.append(v)
                        stack[-1].append(v)
                        n_leaves += 1
            if len(stack) != 1:
                raise ValueError("unbalanced parentheses in newick")
            if partial:
                root_of.extend([len(leaf_tax) - 1] * (len(leaf_tax) - start))
            n_loci += 1
            n_nonbinary += n_internal != (n_leaves if partial else n) - 1
            if len(leaf_tax) * colmap.shape[1] >= chunk_entries - per_tree:
                flush()
            if m and n_loci >= m:
                break
    flush()
    return {"T": T, "n_loci": n_loci, "n_nonbinary_loci": n_nonbinary}


def shares(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``S[i, j, k]``: the share of ``ij|k`` among the three resolutions of atom triple ``{i, j, k}``,
    and the resolved total ``R[i, j, k]`` (NaN share where nothing is resolved)."""
    R = T + T.transpose(0, 2, 1) + T.transpose(2, 1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(R > 0, T / np.where(R > 0, R, 1.0), np.nan)
    return S, R


def tc(S: np.ndarray, ybits: int, m: int) -> float:
    """Mean share of ``ij|k`` over ``i < j`` in ``Y``, ``k`` outside it (atoms of ``P_b``)."""
    inn = [i for i in range(m) if ybits >> i & 1]
    out = [k for k in range(m) if not ybits >> k & 1]
    if len(inn) < 2 or not out:
        return float("nan")
    sub = S[np.ix_(inn, inn, out)]
    iu = np.triu_indices(len(inn), k=1)
    vals = sub[iu[0], iu[1], :].ravel()
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def slow_atom_tensor(gz: Path, taxa: list[str], m: int, atoms: list[int], triples_ijk,
                     partial: bool = False) -> dict:
    """Gate N3's reference: ``T[i, j, k]`` for the listed atom triples, from each gene tree's
    clusters (``extract.clusters_from_newick``) -- ``ab|c`` is induced iff some cluster holds ``a``
    and ``b`` and not ``c``.  No tensor, no ``child_leaf_sets``.  ``partial``: **and** ``c`` is in
    the tree (its root cluster, the union of the rest), ``fz_triples``' N3'(a) reference."""
    import gzip
    ti = {t: i for i, t in enumerate(taxa)}
    out = {t: 0 for t in triples_ijk}
    taxa_of = [list(_bits(a)) for a in atoms]
    with gzip.open(gz, "rt") as fh:
        for n_read, line in enumerate(fh):
            if n_read >= m:
                break
            cl = extract.clusters_from_newick(line.strip(), ti, [])
            present = 0
            for x in cl:
                present |= x
            for (i, j, k) in triples_ijk:
                s = 0
                for a in taxa_of[i]:
                    for b in taxa_of[j]:
                        ab = (1 << a) | (1 << b)
                        for c in taxa_of[k]:
                            cb = 1 << c
                            if partial and not present & cb:
                                continue
                            if any((x & ab) == ab and not (x & cb) for x in cl):
                                s += 1
                out[(i, j, k)] += s
    return out


def _bits(x: int):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)


def absent_c_counts(gz: Path, taxa: list[str], m: int, atoms: list[int], triples_ijk) -> dict:
    """Gate N3'(c)'s prediction: what the full-taxon counters over-count on partial trees.  They
    take every absent ``c`` as outside the node splitting ``a`` from ``b`` -- and each present pair is
    split at exactly one node -- so the excess on atom triple ``(i, j, k)``, ``i != j``, is
    ``sum_t |A_i & t| |A_j & t| |A_k - t|`` over the first ``m`` trees, ``t`` a tree's leaf set."""
    import gzip
    ti = {t: i for i, t in enumerate(taxa)}
    out = {t: 0 for t in triples_ijk}
    with gzip.open(gz, "rt") as fh:
        for n_read, line in enumerate(fh):
            if n_read >= m:
                break
            present = 0
            for x in extract.clusters_from_newick(line.strip(), ti, []):
                present |= x
            for (i, j, k) in triples_ijk:
                out[(i, j, k)] += ((atoms[i] & present).bit_count() * (atoms[j] & present).bit_count()
                                   * (atoms[k] & ~present).bit_count())
    return out
