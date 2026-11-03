"""Stage B's library: ``nni_ensemble``'s per-cell block analysis, truth-free and sized for n = 646.

Imported, unchanged, from ``nni_ensemble``: ``blocks`` (``Diagram``, ``lowz_class``, ``make_blocks``,
``polytomy``, ``atom_bits``, ``allowed``, ``bk_count``, ``bk_all``, ``is_maximal``, ``crosses``, ``bits``)
and ``concord`` (``shares``, ``tc``).

Re-implemented here, each citing its source:

* :func:`analyse_threshold_tf` -- ``features.analyse_threshold`` (``features.py:85-215``) with every
  truth-dependent line removed rather than faked: no ``T``, ``Tset``, ``n_fp``/``n_tp``, ``well_posed``,
  ``category``, ``rstar``, option ``true`` or the low-z ``true``/``nni_true*``/``best_alt_true`` fields.
  ``ctx`` carries no truth at all, so nothing truth-shaped can be computed by accident.
* :func:`atom_tensor_mm` -- ``concord.atom_tensor`` (``concord.py:70-86``), contracted in row slabs
  straight from the memory-mapped ``uint16`` tensor: ~107 MB peak instead of a float64 copy of
  ``count[P, P, P]`` (2.2 GB when ``P_b`` is the root at n = 646).  Exact: every entry is an integer
  far below 2^53.
* :func:`compat_graph_np`, :func:`margins_np`, :func:`tc_batch` -- numpy forms of ``blocks.compat_graph``
  (``:251-260``), ``features._margins`` (``:59-66``) and ``concord.tc`` (``:98-108``), whose Python
  O(a^2) loops do not survive a block with thousands of options.
* :func:`cross_block_gate` -- ``features.py:181-191`` compared only between blocks whose polytomy
  nodes meet (options of disjoint ``P_b`` cannot meet); the original materialises a (sum a)^2 matrix.
* :func:`bk_count_budget` -- ``blocks.bk_count`` (``:284-303``) plus a recursion-node budget, so an
  enumeration that will not finish fails loudly instead of hanging a worker.

Gate N7 (``fz_check.py --parity``) holds every re-implementation to its original on real fungi blocks.
"""

from __future__ import annotations

import math
from collections import Counter

import fz_common as FZ

import numpy as np  # noqa: E402

import blocks as B  # noqa: E402
import concord as K  # noqa: E402
from opc import bitset  # noqa: E402

#: Recursion nodes ``bk_count_budget`` may visit before the cell fails (the scan reports the tail).
BK_NODE_BUDGET = 20_000_000


def dense(words: np.ndarray, n: int) -> np.ndarray:
    """``features._dense``."""
    return bitset.to_dense(words, n) if words.shape[0] else np.zeros((0, n), dtype=bool)


def _bits_dense(xs: list[int], width: int) -> np.ndarray:
    out = np.zeros((len(xs), width), dtype=bool)
    for r, x in enumerate(xs):
        for i in B.bits(x):
            out[r, i] = True
    return out


def _pack_rows(M: np.ndarray) -> list[int]:
    """Each bool row as a Python int, bit j = column j."""
    if M.shape[0] == 0:
        return []
    pk = np.packbits(M, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in pk]


def crossing_matrix(abits: list[int], m: int) -> np.ndarray:
    """``cr[i, j]``: options i and j cross (meet, neither contains the other)."""
    A = _bits_dense(abits, m).astype(np.int32)
    inter = A @ A.T
    sz = A.sum(axis=1)
    return (inter > 0) & (inter < sz[:, None]) & (inter < sz[None, :])


def compat_graph_np(abits: list[int], m: int, cr: np.ndarray | None = None) -> list[int]:
    """``blocks.compat_graph``: ``adj[i]`` = options compatible with ``i`` (nested or disjoint)."""
    if cr is None:
        cr = crossing_matrix(abits, m)
    ok = ~cr
    np.fill_diagonal(ok, False)
    return _pack_rows(ok)


def margins_np(vals, cr: np.ndarray) -> list[float]:
    """``features._margins``: each value minus the best non-NaN value among the options crossing it
    (0 when there is none, or when the value itself is NaN)."""
    v = np.asarray(vals, dtype=float)
    V = np.where(np.isnan(v), -np.inf, v)
    riv = np.where(cr, V[None, :], -np.inf)
    np.fill_diagonal(riv, -np.inf)
    best = riv.max(axis=1) if v.size else v
    ok = np.isfinite(best) & ~np.isnan(v)
    return np.where(ok, v - np.where(ok, best, 0.0), 0.0).tolist()


def tc_batch(S: np.ndarray, abits: list[int], m: int, chunk: int = 128) -> list[float]:
    """``concord.tc`` for every option: the mean of ``S[i, j, k]`` over atoms ``i < j`` in ``Y`` and
    ``k`` outside it, skipping NaN; NaN when ``Y`` has < 2 atoms, nothing is outside, or every share
    is NaN.  ``S`` is symmetric in ``(i, j)``, so the ``i < j`` sum is half the ``i != j`` sum."""
    Y = _bits_dense(abits, m).astype(float)
    good = ~np.isnan(S)
    S0 = np.where(good, S, 0.0)
    G0 = good.astype(float)
    idx = np.arange(m)
    S0[idx, idx, :] = 0.0
    G0[idx, idx, :] = 0.0
    out = np.full(len(abits), np.nan)
    for s in range(0, len(abits), chunk):
        y = Y[s:s + chunk]
        o = 1.0 - y
        US = np.tensordot(S0, o, axes=([2], [1]))            # (m, m, c)
        UG = np.tensordot(G0, o, axes=([2], [1]))
        num = np.einsum("ci,cj,ijc->c", y, y, US) / 2.0
        cnt = np.einsum("ci,cj,ijc->c", y, y, UG) / 2.0
        ny = y.sum(axis=1)
        ok = (ny >= 2) & (ny < m) & (cnt > 0)
        out[s:s + chunk] = np.where(ok, num / np.where(cnt > 0, cnt, 1.0), np.nan)
    return out.tolist()


def atom_tensor_mm(count: np.ndarray, atoms: list[int], chunk: int = 32) -> np.ndarray:
    """``concord.atom_tensor``: ``T[i, j, k] = sum count[A_i, A_j, A_k]``, read from the (memory-mapped)
    tensor in slabs of ``chunk`` rows."""
    m = len(atoms)
    idx, owner = [], []
    for i, a in enumerate(atoms):
        for t in B.bits(a):
            idx.append(t)
            owner.append(i)
    idx = np.asarray(idx)
    G = np.zeros((idx.size, m))
    G[np.arange(idx.size), owner] = 1.0
    T = np.zeros((m, m, m))
    for s in range(0, idx.size, chunk):
        rows = idx[s:s + chunk]
        Cc = np.asarray(count[rows])[:, idx][:, :, idx].astype(np.float64)   # (c, p, p)
        U = np.tensordot(Cc, G, axes=([2], [0]))                             # (c, p, m_k)
        U = np.tensordot(U, G, axes=([1], [0]))                              # (c, m_k, m_j)
        T += np.tensordot(G[s:s + chunk], U, axes=([0], [0]))                # (m_i, m_k, m_j)
    return T.transpose(0, 2, 1)


def bk_count_budget(adj: list[int], cap: int, max_nodes: int = BK_NODE_BUDGET) -> tuple[int, int]:
    """``blocks.bk_count`` (``count``, or ``cap + 1`` once past ``cap``) and the recursion nodes it took.
    Raises ``MemoryError`` past ``max_nodes``."""
    count = 0
    nodes = 0

    def rec(P: int, X: int) -> bool:
        nonlocal count, nodes
        nodes += 1
        if nodes > max_nodes:
            raise MemoryError(f"bk_count: {max_nodes:,} recursion nodes, {count} cliques so far "
                              f"({len(adj)} options)")
        if not P:
            if not X:
                count += 1
            return count > cap
        u = B._pivot(P, X, adj)
        for v in list(B.bits(P & ~adj[u])):
            if rec(P & adj[v], X & adj[v]):
                return True
            P &= ~(1 << v)
            X |= 1 << v
        return False

    rec((1 << len(adj)) - 1, 0)
    return count, nodes


def radius_candidates(adj: list[int], members: int, rho: int) -> int:
    """The number of recursion calls ``blocks.radius_configs(adj, members, rho)`` makes, exactly, for
    ``rho <= 2``: 1 + #non-emitted options + #compatible pairs of them."""
    a = len(adj)
    non = [v for v in range(a) if not members >> v & 1]
    if rho == 0:
        return 1
    tot = 1 + len(non)
    if rho >= 2:
        nonmask = sum(1 << v for v in non)
        for i, v in enumerate(non):
            later = nonmask & ~((1 << (v + 1)) - 1)
            tot += (adj[v] & later).bit_count()
    if rho > 2:
        raise ValueError("radius_candidates is exact for rho <= 2 only")
    return tot


def cross_block_gate(blocks_opts: list[tuple[int, int, list[int]]], n: int, words_w: int) -> None:
    """``features.py:181-191``: options of two different blocks never cross.  ``blocks_opts`` holds
    ``(block id, P_b, option masks)``.  Raises ``AssertionError`` naming the pair."""
    dens = {}
    for bi, P, masks in blocks_opts:
        D = np.stack([np.frombuffer(y.to_bytes(8 * words_w, "little"), np.uint64) for y in masks])
        dens[bi] = dense(D, n).astype(np.float32)
    for x in range(len(blocks_opts)):
        bi, Pi, _ = blocks_opts[x]
        for y in range(x + 1, len(blocks_opts)):
            bj, Pj, _ = blocks_opts[y]
            if not Pi & Pj:
                continue
            Di, Dj = dens[bi], dens[bj]
            ic = Di @ Dj.T
            si, sj = Di.sum(axis=1)[:, None], Dj.sum(axis=1)[None, :]
            if ((ic > 0) & (ic < si) & (ic < sj)).any():
                raise AssertionError(f"options of blocks {bi} and {bj} cross (N0: blocks not independent)")


def analyse_threshold_tf(D: B.Diagram, ctx: dict, thr: float, *, cell: str, with_tc: bool = True,
                         scan: bool = False) -> dict:
    """``features.analyse_threshold`` without the truth.  ``with_tc=False`` skips the triple tensor
    (the scan); ``scan=True`` also records cost drivers and never raises on a cap, recording it."""
    n, W, z, sup, m_trees = ctx["n"], ctx["W"], ctx["z"], ctx["support"], ctx["m"]
    row_of = ctx["row_of"]
    caps = FZ.CAPS
    E = D.E
    zE = {e: float(z[row_of[e]]) for e in E}
    low = [e for e in E if zE[e] < thr]
    cls = {e: B.lowz_class(D, e) for e in low}
    Q = [e for e in low if cls[e] == "Q"]
    Qset = set(Q)
    blocks = B.make_blocks(D, Q)
    block_of = {q: i for i, b in enumerate(blocks) for q in b}
    Eidx = {e: i for i, e in enumerate(E)}
    DE = dense(np.stack([ctx["words"][row_of[e]] for e in E]), n) if E else np.zeros((0, n), bool)

    out_blocks, all_opts, cap_hits = [], [], []
    for bi, b in enumerate(blocks):
        pt = B.polytomy(D, b)
        if pt["P"] in Qset:
            raise AssertionError(f"{cell} block {bi}: its polytomy node is itself questioned")
        mm = len(pt["atoms"])
        rec = {"id": bi, "size": len(b), "m": mm, "n_merged": pt["n_merged"],
               "P": FZ.NE.hexmask(pt["P"]), "p_size": B.popcount(pt["P"]),
               "p_is_root": pt["p_is_root"], "p_crosses": pt["p_crosses"],
               "hybrid_atom": pt["hybrid_atom"], "member_z": [zE[q] for q in b]}
        if pt["hybrid_atom"]:
            rec.update({"excluded": "hybrid_atom", "options": None, "members": None, "r_count": -1})
            out_blocks.append(rec)
            continue
        bs = set(b)
        keep = np.ones(len(E), dtype=bool)
        keep[[Eidx[q] for q in b]] = False
        others = [e for e in E if e not in bs]
        rows = B.allowed(pt["P"], pt["atoms"], others, ctx["dense"], ctx["size"], n, DE[keep])
        masks = [W[r] for r in rows]
        a = len(masks)
        rec["n_allowed"] = a
        if mm > caps["M_CAP"] or a > caps["A_CAP"]:
            what, val, cap = (("atoms", mm, caps["M_CAP"]) if mm > caps["M_CAP"]
                              else ("options", a, caps["A_CAP"]))
            if not scan:
                raise FZ.CapExceeded(cell, bi, what, val, cap)
            cap_hits.append({"block": bi, "what": what, "value": val, "cap": cap})
            rec.update({"excluded": f"cap_{what}", "options": None, "members": None, "r_count": -1})
            out_blocks.append(rec)
            continue
        abits = [B.atom_bits(y, pt["atoms"]) for y in masks]
        if any(x is None for x in abits):
            raise AssertionError(f"{cell} block {bi}: an allowed cluster splits an atom")
        idx_of = {y: i for i, y in enumerate(masks)}
        members = [idx_of.get(q, -1) for q in b]
        if -1 in members:
            raise AssertionError(f"{cell} block {bi}: an emitted member is not in A_b (N0)")
        cr = crossing_matrix(abits, mm)
        adj = compat_graph_np(abits, mm, cr)
        mem_set = set(members)
        memmask = sum(1 << i for i in members)
        # maximality of the emitted family (blocks.is_maximal), on the packed graph: no outside
        # option is compatible with every member
        if any((adj[y] & memmask) == memmask for y in range(a) if y not in mem_set):
            raise AssertionError(f"{cell} block {bi}: the emitted configuration is not maximal in A_b (N0)")
        try:
            r_count, bk_nodes = bk_count_budget(adj, FZ.NE.R_CAP)
        except MemoryError as exc:
            if not scan:
                raise FZ.CapExceeded(cell, bi, "bk_count nodes", BK_NODE_BUDGET + 1, BK_NODE_BUDGET) from exc
            cap_hits.append({"block": bi, "what": "bk_count nodes", "value": BK_NODE_BUDGET + 1,
                             "cap": BK_NODE_BUDGET})
            r_count, bk_nodes = -2, BK_NODE_BUDGET
        rec["bk_nodes"] = bk_nodes
        if r_count > FZ.NE.R_CAP:
            rc = radius_candidates(adj, memmask, FZ.NE.RHO)
            rec["radius_candidates"] = rc
            if rc > caps["RAD_CAND_CAP"]:
                if not scan:
                    raise FZ.CapExceeded(cell, bi, "radius candidates", rc, caps["RAD_CAND_CAP"])
                cap_hits.append({"block": bi, "what": "radius candidates", "value": rc,
                                 "cap": caps["RAD_CAND_CAP"]})

        zs = [float(z[r]) for r in rows]
        tcs = [math.nan] * a
        tc_info = {}
        if with_tc:
            key = (pt["P"], tuple(pt["atoms"]))
            if key not in ctx["S_cache"]:
                S, R = K.shares(atom_tensor_mm(ctx["count"], pt["atoms"]))
                I, J, L = np.ogrid[:mm, :mm, :mm]
                Rv = R[(I != J) & (I != L) & (J != L)]       # resolved locus-triples per atom triple
                ctx["S_cache"][key] = (S, float(Rv.min()) if Rv.size else math.nan,
                                       float(np.median(Rv)) if Rv.size else math.nan)
            S, r_min, r_med = ctx["S_cache"][key]
            tcs = (tc_batch(S, abits, mm) if (mm > 64 or a > 512)
                   else [K.tc(S, x, mm) for x in abits])
            tc_info = {"tc_nan": int(sum(1 for t in tcs if math.isnan(t))), "R_min": r_min,
                       "R_median": r_med}
            if mm == 3 and a == 3:
                tot = sum(tcs)
                if not (math.isnan(tot) or abs(tot - 1.0) < 1e-9):
                    raise AssertionError(f"{cell} block {bi}: three-atom tc sums to {tot} (N3-tc)")

        # N2(a): the lone-cluster NNI identity (features.py:147-155)
        n2a = None
        if len(b) == 1 and mm == 3 and pt["n_merged"] == 0 and not pt["p_crosses"]:
            alts = {x for x in D.nni(b[0]) if x in row_of}
            want = {frozenset([b[0]])} | {frozenset([x]) for x in alts}
            got = {frozenset(masks[i] for i in B.bits(x)) for x in B.bk_all(adj)}
            n2a = want == got
            if not n2a:
                raise AssertionError(f"{cell} block {bi}: R_b is not the 1-NNI set (N2a)")

        opts = {"row": [int(r) for r in rows], "abits": [FZ.NE.hexmask(x) for x in abits],
                "z": zs, "support": [int(sup[r]) for r in rows],
                "freq": [float(sup[r]) / m_trees for r in rows], "tc": tcs,
                "k": [x.bit_count() for x in abits],
                "emitted": [i in mem_set for i in range(a)],
                "margin_z": margins_np(zs, cr), "margin_tc": margins_np(tcs, cr)}
        rec.update({"excluded": "", "options": opts, "members": sorted(members), "r_count": r_count,
                    "n2a": n2a, **tc_info})
        out_blocks.append(rec)
        all_opts.append((bi, pt["P"], masks))

    if not scan and all_opts:
        cross_block_gate(all_opts, n, ctx["words"].shape[1])

    lowz = []
    for e in low:
        nb = set(D.parents[e]) | set(D.children.get(e, [])) | D.siblings(e)
        nb_low = [x for x in nb if x in zE and zE[x] < thr]
        alts = D.nni(e)
        alts_c = [x for x in alts if x in row_of]
        lowz.append({"size": B.popcount(e), "z": zE[e], "cls": cls[e], "block": block_of.get(e, -1),
                     "block_size": len(blocks[block_of[e]]) if e in block_of else 0,
                     "n_adj_low": len(nb_low), "n_nni": len(alts), "n_nni_ctau": len(alts_c)})
    M = len(low)
    return {"thr": thr, "M": M, "n_emitted": len(E), "classes": dict(Counter(cls.values())),
            "log10_3M": M * math.log10(3),
            "log10_nni": sum(math.log10(1 + r["n_nni_ctau"]) for r in lowz),
            "blocks": out_blocks, "lowz": lowz, "cap_hits": cap_hits, "truth_free": True}


def make_ctx(cell: dict, count=None) -> dict:
    """What :func:`analyse_threshold_tf` reads: the cell's candidates, logits and the triple tensor."""
    words = cell["words"]
    W = FZ.NE.masks_of(words)
    dn = _dense_cached(cell, words)
    return {"n": cell["n"], "W": W, "z": cell["z"], "support": cell["support"], "m": cell["m"],
            "row_of": {mk: i for i, mk in enumerate(W)}, "words": words, "dense": dn,
            "size": dn.sum(axis=1), "count": count, "S_cache": {}}


_DENSE: dict = {}


def _dense_cached(cell: dict, words: np.ndarray) -> np.ndarray:
    key = (cell["scheme"], cell["tau"], cell["npz_sha"])
    if key not in _DENSE:
        _DENSE.clear()
        _DENSE[key] = dense(words, cell["n"])
    return _DENSE[key]
