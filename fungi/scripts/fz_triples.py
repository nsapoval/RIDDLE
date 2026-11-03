#!/usr/bin/env python3
"""Stage B2: the partial-aware rooted-triple tensor, once per scheme, and the gates that prove it.

**Why not ``opc.mpl.triples``.**  ``TripleCounts.add_tree`` (``gnn_cpu_v1/pylib/opc/mpl/triples.py:112``)
takes ``outside = universe[~below]``, which on a partial tree includes every taxon the tree does not
carry: it counts ``ab|c`` for each absent ``c`` -- at a partial tree's root too, where the "root:
continue" guard never fires because ``outside`` is not empty.  ``nni_ensemble/concord.slow_atom_tensor``
(gate N3's reference) asks only that ``c`` be outside a cluster, so it shares the blind spot and N3
would pass vacuously.  Every fungi tree the GNN read is partial except a handful.

**The counter.**  For a binary tree with leaf set ``L`` let ``H[a, b]`` be the size of the cluster at
``lca(a, b)``.  For ``a, b, c`` in ``L``, ``lca(a, b)`` and ``lca(a, c)`` both lie on ``a``'s root path,
where cluster sizes strictly decrease downward, so ``ab|c`` iff ``H[a, b] < H[a, c]``.  With absent
taxa and the diagonal set to a large sentinel on the left (``Hb``) and to 0 on the right (``Hc``),
nothing involving them can count:

    count[a, b, c] = sum_t [ Hb_t[a, b] < Hc_t[a, c] ]

symmetric in ``(a, b)``, zero whenever two indices coincide -- ``TripleCounts``' convention.  It is
sharded over the anchor ``a`` and stored as ``uint16`` (counts <= m < 65,535; asserted).

**Polytomies** (the collapsed variant, ``FZ_VARIANT=collapsed``).  A node with d >= 3 children is the
lca of every pair of leaves under two *different* children, so ``H`` is set for exactly those pairs;
three leaves under three different children get equal ``H`` and count for none of their three
resolutions, which is what an unresolved triple is.  The binary branch is the landed code unchanged.

Gates (``--check``; exit 1 on any failure):

* **N3'(a)** -- on 40 trees (the 10 smallest, the 10 largest, 20 seeded at random), the fast counter
  equals a slow reference from each tree's own clusters (``extract.clusters_from_newick``):
  ``ab|c`` iff some cluster holds ``a`` and ``b`` and not ``c``, **and** ``c`` is in the tree -- on
  50,000 sampled triples;
  plus, where trees have polytomies, the 20 trees with the largest unresolved share and triples drawn
  inside each one's largest polytomy -- three children (unresolved, all three rotations) and two
  (resolved) -- so the unresolved case is exercised, not merely allowed;
* **N3'(b)** -- the full tensor: for every distinct ``a, b, c``, the three resolutions sum to **at
  most** the number of trees carrying all three (``presence.npy``), and per anchor ``x`` the shortfall
  summed over ``(b, c)`` is exactly ``2 u_x``, where ``u_x`` counts, over trees and polytomy nodes
  holding ``x``, the leaf pairs under two distinct children other than ``x``'s -- so on binary trees
  (``u = 0``) it is the landed equality.  Plus symmetry in the first two axes, zero diagonals and
  ``count <= m``.  ``--check sample`` does it on 64 anchors, ``full`` on all 646;
* **N3'(d)** -- variant only: every count is at most the landed tensor's (``fungi/analysis/trees``),
  anchor by anchor -- contracting an edge only removes resolutions -- given the same ``chosen_sha256``
  and a landed ``n3.json`` that passed;
* **N3'(c)** -- on 20 of those trees, the published ``triples.count_newick_file`` minus the fast
  counter equals ``#{t : a, b in t, c not in t}`` exactly on the sampled triples: the fix removes the
  blind spot and nothing else.

Writes ``<out-root>/trees/<scheme>/triples.u16.npy`` (n x n x n, sorted label order),
``triples.json`` and ``n3.json``.  Shards are written one by one (``triples.shard<i>of<S>.npy``),
so a killed build resumes.

    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_triples.py --schemes medium --limit-trees 64 --shards 1 --out-root <scratch>
    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_triples.py --schemes medium,short --jobs 5 --check full
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import fz_common as FZ

import numpy as np  # noqa: E402
from numpy.lib.format import open_memmap  # noqa: E402

from opc.clusters import extract  # noqa: E402
from opc.mpl import triples as TR  # noqa: E402

BIG = np.int16(32767)
BATCH = 64


def read_trees(out_root: Path, scheme: str, limit: int = 0) -> tuple[list[str], list[str]]:
    d = FZ.trees_dir(out_root, scheme)
    labels = FZ.read_json(d / "trees.json")["labels"]
    with gzip.open(d / "trees.nwk.gz", "rt") as fh:
        trees = [ln.strip() for ln in fh if ln.strip()]
    return (trees[:limit] if limit else trees), labels


def lca_size(newick: str, ti: dict[str, int], n: int) -> np.ndarray:
    """``H[a, b]`` = size of the cluster at ``lca(a, b)`` for distinct leaves of the tree, else 0.

    A node with d >= 3 children sets ``H`` for the pairs under two different children of it (one
    masked write, not a loop over child pairs); pairs under one child are left to that child."""
    H = np.zeros((n, n), dtype=np.int16)
    for kids in TR.child_leaf_sets(newick, ti):
        if len(kids) == 2:
            a, b = np.asarray(kids[0]), np.asarray(kids[1])
            s = np.int16(a.size + b.size)
            H[np.ix_(a, b)] = s
            H[np.ix_(b, a)] = s
        elif len(kids) > 2:
            X = np.concatenate([np.asarray(k) for k in kids])
            lab = np.repeat(np.arange(len(kids)), [len(k) for k in kids])
            ix = np.ix_(X, X)
            H[ix] = np.where(lab[:, None] != lab[None, :], np.int16(X.size), H[ix])
        else:
            raise ValueError("a unary node: the counter needs no node with one child")
    return H


def unresolved_by_anchor(trees: list[str], labels: list[str]) -> tuple[np.ndarray, list[float]]:
    """``u[x]`` = sum over trees and polytomy nodes v holding x of e2(sizes of v's children other than
    x's) -- the unordered pairs {b, c} with {x, b, c} unresolved -- and each tree's unresolved share."""
    ti = {t: i for i, t in enumerate(labels)}
    u = np.zeros(len(labels), dtype=np.int64)
    share = []
    for t in trees:
        tot_unres, nl = 0, t.count(",") + 1
        for kids in TR.child_leaf_sets(t, ti):
            if len(kids) < 3:
                continue
            sz = np.array([len(k) for k in kids], dtype=np.int64)
            S, Q = int(sz.sum()), int((sz * sz).sum())
            e2 = (S * S - Q) // 2
            for k, sj in zip(kids, sz):
                u[np.asarray(k)] += e2 - int(sj) * (S - int(sj))
            e1 = e2_ = e3 = 0
            for x in sz.tolist():
                e3 += e2_ * x
                e2_ += e1 * x
                e1 += x
            tot_unres += e3
        share.append(tot_unres / max(1, nl * (nl - 1) * (nl - 2) // 6))
    return u, share


def polytomy_triples(trees: list[str], labels: list[str], idx: list[int], k: int, seed: int) -> np.ndarray:
    """Triples inside the largest polytomy of each tree in ``idx``: three children -> all three
    rotations of an unresolved triple; two children -> ``ab|c`` with a, b under one child."""
    ti = {t: i for i, t in enumerate(labels)}
    rng = random.Random(seed)
    out = []
    for i in idx:
        nodes = TR.child_leaf_sets(trees[i], ti)
        kids = max(nodes, key=len)
        if len(kids) < 3:
            continue
        multi = [c for c in kids if len(c) >= 2]
        for _ in range(k):
            x, y, z = rng.sample(kids, 3)
            a, b, c = rng.choice(x), rng.choice(y), rng.choice(z)
            out += [(a, b, c), (a, c, b), (b, c, a)]
            if multi:
                big = rng.choice(multi)
                other = rng.choice([c2 for c2 in kids if c2 is not big])
                a, b = rng.sample(big, 2)
                out.append((a, b, rng.choice(other)))
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


def count_block(trees: list[str], labels: list[str], lo: int, hi: int, batch: int = BATCH) -> np.ndarray:
    """``count[lo:hi]`` over ``trees``: ``uint16[hi - lo, n, n]``."""
    n = len(labels)
    ti = {t: i for i, t in enumerate(labels)}
    acc = np.zeros((hi - lo, n, n), dtype=np.uint16)
    buf = np.empty((batch, n, n), dtype=bool)
    for s in range(0, len(trees), batch):
        chunk = trees[s:s + batch]
        Hc = np.stack([lca_size(t, ti, n) for t in chunk])
        Hb = np.where(Hc > 0, Hc, BIG)
        B = len(chunk)
        for a in range(lo, hi):
            np.less(Hb[:, a, :, None], Hc[:, a, None, :], out=buf[:B])
            acc[a - lo] += np.add.reduce(buf[:B].view(np.uint8), axis=0, dtype=np.uint16)
    return acc


def shard_bounds(n: int, shards: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, shards + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(shards)]


def _shard_job(u):
    out_root, scheme, limit, i, S, lo, hi = u
    d = FZ.trees_dir(out_root, scheme)
    path = d / f"triples.shard{i}of{S}.npy"
    if path.exists():
        return {"unit": f"{scheme}:{i}", "skipped": True}
    t0 = time.perf_counter()
    trees, labels = read_trees(out_root, scheme, limit)
    acc = count_block(trees, labels, lo, hi)
    tmp = d / f"triples.shard{i}of{S}.tmp.npy"
    np.save(tmp, acc)
    tmp.replace(path)
    return {"unit": f"{scheme}:{i}", "seconds": time.perf_counter() - t0, "rss_mb": FZ.peak_rss_mb()}


def build(out_root: Path, scheme: str, jobs: int, shards: int, limit: int, force: bool) -> dict:
    d = FZ.trees_dir(out_root, scheme)
    final = d / "triples.u16.npy"
    trees, labels = read_trees(out_root, scheme, limit)
    n, m = len(labels), len(trees)
    if m >= 65535:
        raise ValueError(f"{m} trees: uint16 counts could overflow")
    if final.exists() and not force:
        meta = FZ.read_json(d / "triples.json")
        if meta["m"] == m and meta["trees_sha256"] == FZ.sha256(d / "trees.nwk.gz"):
            print(f"[triples] {scheme}: {final} is current", flush=True)
            return meta
    if force:
        for p in d.glob("triples.shard*of*.npy"):
            p.unlink()
    t0 = time.perf_counter()
    bounds = shard_bounds(n, shards)
    units = [(out_root, scheme, limit, i, shards, lo, hi) for i, (lo, hi) in enumerate(bounds)]
    ok, err = FZ.run_units(units, _shard_job, jobs, f"triples {scheme}")
    if err:
        raise RuntimeError(f"{len(err)} shards failed")
    tmp = d / "triples.u16.tmp.npy"
    mm = open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n, n, n))
    for i, (lo, hi) in enumerate(bounds):
        mm[lo:hi] = np.load(d / f"triples.shard{i}of{shards}.npy")
    mm.flush()
    del mm
    os.replace(tmp, final)
    for i in range(shards):
        (d / f"triples.shard{i}of{shards}.npy").unlink()
    secs = time.perf_counter() - t0
    meta = {"scheme": scheme, "n": n, "m": m, "limit_trees": limit, "shape": [n, n, n],
            "dtype": "uint16", "shards": shards, "jobs": jobs, "seconds": secs,
            "shard_seconds": [r.get("seconds") for r in ok], "shard_rss_mb": [r.get("rss_mb") for r in ok],
            "trees_sha256": FZ.sha256(d / "trees.nwk.gz"), "sha256": FZ.sha256(final),
            "labels_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest()}
    FZ.write_json(d / "triples.json", meta, indent=1)
    cpu = sum(r.get("seconds") or 0 for r in ok)
    print(f"[triples] {scheme}: n {n}, m {m}, {shards} shards, {secs:.0f} s wall, {cpu:.0f} s in shards "
          f"({1e3 * cpu / max(1, n * m):.3f} ms per (anchor, tree))", flush=True)
    return meta


# ------------------------------------------------------------------------------------------ gates

def gate_trees(trees: list[str], seed: int = 0) -> list[int]:
    """The 10 smallest, the 10 largest and 20 seeded random trees (indices, sorted, distinct)."""
    sizes = [t.count(",") + 1 for t in trees]
    order = sorted(range(len(trees)), key=lambda i: (sizes[i], i))
    pick = set(order[:10]) | set(order[-10:])
    rest = [i for i in range(len(trees)) if i not in pick]
    pick |= set(random.Random(seed).sample(rest, min(20, len(rest))))
    return sorted(pick)


def sample_triples(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, size=(k * 2, 3))
    a = a[(a[:, 0] != a[:, 1]) & (a[:, 0] != a[:, 2]) & (a[:, 1] != a[:, 2])]
    return a[:k]


def slow_counts(trees: list[str], labels: list[str], trip: np.ndarray) -> np.ndarray:
    """N3'(a)'s reference: per tree, ``ab|c`` iff some cluster of the tree holds a, b and not c, and
    c is a leaf of the tree.  Clusters from ``extract.clusters_from_newick`` (root included)."""
    ti = {t: i for i, t in enumerate(labels)}
    n = len(labels)
    out = np.zeros(len(trip), dtype=np.int64)
    a, b, c = trip[:, 0], trip[:, 1], trip[:, 2]
    for t in trees:
        buf: list[int] = []
        extract.clusters_from_newick(t, ti, buf)
        root = 0
        for x in buf:
            root |= x
        Dm = np.array([[(x >> j) & 1 for j in range(n)] for x in buf], dtype=bool)
        present = np.array([(root >> j) & 1 for j in range(n)], dtype=bool)
        hit = (Dm[:, a] & Dm[:, b] & ~Dm[:, c]).any(axis=0) & present[c]
        out += hit
    return out


def check(out_root: Path, scheme: str, mode: str, n3_trees: int = 40) -> dict:
    d = FZ.trees_dir(out_root, scheme)
    meta = FZ.read_json(d / "triples.json")
    trees, labels = read_trees(out_root, scheme, meta["limit_trees"])
    n, m = len(labels), len(trees)
    res = {"scheme": scheme, "mode": mode, "problems": []}
    t0 = time.perf_counter()

    # (a) fast == slow cluster reference, on 40 gate trees
    idx = gate_trees(trees)[:n3_trees]
    sub = [trees[i] for i in idx]
    fast = count_block(sub, labels, 0, n)
    trip = sample_triples(n, 50_000, seed=1)
    f_ab = fast[trip[:, 0], trip[:, 1], trip[:, 2]].astype(np.int64)
    slow = slow_counts(sub, labels, trip)
    res["n3a"] = {"n_trees": len(sub), "n_triples": len(trip), "n_nonzero": int((slow > 0).sum()),
                  "n_mismatch": int((f_ab != slow).sum()),
                  "leaves": [sub[i].count(",") + 1 for i in (0, len(sub) - 1)]}
    if res["n3a"]["n_mismatch"] or not res["n3a"]["n_nonzero"]:
        res["problems"].append(f"N3'(a): {res['n3a']}")
    u, share = unresolved_by_anchor(trees, labels)
    poly = [i for i in sorted(range(m), key=lambda i: (-share[i], i)) if share[i] > 0][:20]
    if poly:                                        # polytomies: the unresolved case, exercised
        ti = {t: i for i, t in enumerate(labels)}
        r3 = {"n_trees": len(poly), "n_triples": 0, "n_unresolved": 0, "n_mismatch": 0}
        allp = []
        for j, i in enumerate(poly):                # per tree: count_block's comparison on one tree's H
            tr3 = polytomy_triples(trees, labels, [i], 100, seed=3 + j)
            allp.append(tr3)
            H = lca_size(trees[i], ti, n)
            Hb = np.where(H > 0, H, BIG)
            got = (Hb[tr3[:, 0], tr3[:, 1]] < H[tr3[:, 0], tr3[:, 2]]).astype(np.int64)
            want = slow_counts([trees[i]], labels, tr3)
            r3["n_triples"] += len(tr3)
            r3["n_unresolved"] += int((want == 0).sum())
            r3["n_mismatch"] += int((got != want).sum())
        psub_a = [trees[i] for i in poly]           # summed over the 20: count_block's accumulation too
        fast_a = count_block(psub_a, labels, 0, n)
        allp = np.concatenate(allp + [sample_triples(n, 50_000, seed=4)])
        got = fast_a[allp[:, 0], allp[:, 1], allp[:, 2]].astype(np.int64)
        del fast_a
        want = slow_counts(psub_a, labels, allp)
        r3["summed_triples"], r3["summed_mismatch"] = len(allp), int((got != want).sum())
        r3["unresolved_share"] = [float(share[i]) for i in (poly[0], poly[-1])]
        res["n3a_poly"] = r3
        if r3["n_mismatch"] or r3["summed_mismatch"] or not r3["n_unresolved"]:
            res["problems"].append(f"N3'(a) polytomies: {r3}")

    # (c) published - fast == #{t: a, b in t, c not in t}, on 20 of them (10 smallest + 10 random)
    pidx = idx[:10] + idx[20:30] if len(idx) >= 30 else idx
    psub = [trees[i] for i in pidx]
    ti = {t: i for i, t in enumerate(labels)}
    with tempfile.TemporaryDirectory() as tmpd:
        gz = Path(tmpd) / "t.nwk.gz"
        with gzip.open(gz, "wt") as fh:
            fh.write("\n".join(psub) + "\n")
        pub = TR.count_newick_file(gz, ti, max_loci=len(psub)).count
    fast_p = count_block(psub, labels, 0, n)
    P = np.zeros((len(psub), n), dtype=np.int64)
    for r, t in enumerate(psub):
        buf: list[int] = []
        extract.clusters_from_newick(t, ti, buf)
        root = 0
        for x in buf:
            root |= x
        P[r] = [(root >> j) & 1 for j in range(n)]
    a, b, c = trip[:, 0], trip[:, 1], trip[:, 2]
    want = (P[:, a] * P[:, b] * (1 - P[:, c])).sum(axis=0)
    got = pub[a, b, c].astype(np.int64) - fast_p[a, b, c].astype(np.int64)
    res["n3c"] = {"n_trees": len(psub), "n_triples": len(trip), "n_mismatch": int((got != want).sum()),
                  "spurious_total": int(want.sum()),
                  "spurious_share_of_published": float(want.sum() / max(1, pub[a, b, c].sum()))}
    del pub
    if res["n3c"]["n_mismatch"] or not res["n3c"]["spurious_total"]:
        res["problems"].append(f"N3'(c): {res['n3c']}")

    # (b) the full tensor: resolutions sum to co-occurrence; symmetry; zero diagonals; <= m
    C = np.load(d / "triples.u16.npy")
    pres = np.load(d / "presence.npy")
    if meta["limit_trees"]:
        pres = pres[:meta["limit_trees"]]
    Pf = pres.astype(np.float32)
    anchors = range(n) if mode == "full" else sorted(random.Random(2).sample(range(n), min(64, n)))
    n_bad, n_checked, n_short, short_total, n_anchor_bad = 0, 0, 0, 0, 0
    eye = np.eye(n, dtype=bool)
    for x in anchors:
        Pa = Pf[pres[:, x]]
        co = (Pa.T @ Pa).astype(np.int64)                    # trees carrying x, b, c
        # the three resolutions of {x, b, c}: xb|c = C[x, b, c], xc|b = C[x, c, b], bc|x = C[b, c, x]
        R = C[x].astype(np.int64) + C[x].T.astype(np.int64) + C[:, :, x].astype(np.int64)
        mask = ~eye
        mask[x, :] = False
        mask[:, x] = False
        gap = co[mask] - R[mask]                             # trees carrying {x, b, c} unresolved
        n_bad += int((gap < 0).sum())
        n_short += int((gap > 0).sum())
        short_total += int(gap.sum())
        if int(gap.sum()) != 2 * int(u[x]):
            n_anchor_bad += 1
        n_checked += int(mask.sum())
        if (C[x, x, :].any() or C[x, :, x].any() or np.diagonal(C[x]).any()):
            res["problems"].append(f"N3'(b): a diagonal entry is non-zero at anchor {x}")
    sym = bool(np.array_equal(C, C.transpose(1, 0, 2))) if mode == "full" else bool(
        all(np.array_equal(C[x], C[:, x, :]) for x in anchors))
    res["n3b"] = {"anchors": len(list(anchors)), "n_checked": n_checked, "n_mismatch": n_bad + n_anchor_bad,
                  "symmetric": sym, "max_count": int(C.max()), "m": m}
    if FZ.P.VARIANT or n_short:
        res["n3b"].update({"n_over": n_bad, "n_anchor_sum_mismatch": n_anchor_bad,
                           "n_unresolved_entries": n_short, "unresolved_total_ordered": short_total})
    if n_bad or n_anchor_bad or not sym or int(C.max()) > m:
        res["problems"].append(f"N3'(b): {res['n3b']}")

    # (d) variant only: no count exceeds the landed tensor's (contraction only removes resolutions)
    if FZ.P.VARIANT and not meta["limit_trees"]:
        ld = FZ.P.LANDED / "trees" / scheme
        li, ln3 = FZ.read_json(ld / "trees.json"), FZ.read_json(ld / "n3.json")
        mine = FZ.read_json(d / "trees.json")
        r4 = {"landed_tensor_sha256": FZ.read_json(ld / "triples.json")["sha256"]}
        if li["chosen_sha256"] != mine["chosen_sha256"] or not ln3.get("ok"):
            res["problems"].append(f"N3'(d): precondition fails (chosen equal "
                                   f"{li['chosen_sha256'] == mine['chosen_sha256']}, landed n3 ok {ln3.get('ok')})")
        else:
            L = np.load(ld / "triples.u16.npy", mmap_mode="r")
            over, removed, lost_anchor = 0, 0, 0
            for x in range(n):
                lx = np.asarray(L[x], dtype=np.int64)
                cx = C[x].astype(np.int64)
                over += int((cx > lx).sum())
                removed += int((lx - cx).sum())
                lost_anchor += int(((lx > 0) & (cx == 0)).sum())
            del L
            r4.update({"n_over": over, "removed_total": removed, "entries_emptied": lost_anchor})
            if over or not removed:
                res["problems"].append(f"N3'(d): {r4}")
        res["n3d"] = r4
    res["seconds"] = time.perf_counter() - t0
    res["ok"] = not res["problems"]
    FZ.write_json(d / "n3.json", res, indent=1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", default=",".join(FZ.SCHEMES))
    ap.add_argument("--out-root", type=Path, default=FZ.ANALYSIS)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--shards", type=int, default=0, help="default: --jobs")
    ap.add_argument("--limit-trees", type=int, default=0, help="first N trees only (smokes)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", choices=("none", "sample", "full"), default="sample")
    args = ap.parse_args()
    problems = []
    for scheme in [s for s in args.schemes.split(",") if s]:
        build(args.out_root, scheme, args.jobs, args.shards or args.jobs, args.limit_trees, args.force)
        if args.check != "none":
            r = check(args.out_root, scheme, args.check)
            print(f"[triples] {scheme} N3': (a) {r['n3a']['n_mismatch']} mismatches of "
                  f"{r['n3a']['n_triples']:,} ({r['n3a']['n_nonzero']:,} non-zero); (b) "
                  f"{r['n3b']['n_mismatch']} of {r['n3b']['n_checked']:,} over {r['n3b']['anchors']} "
                  f"anchors, symmetric {r['n3b']['symmetric']}; (c) {r['n3c']['n_mismatch']} mismatches, "
                  f"published counter's spurious share {r['n3c']['spurious_share_of_published']:.1%} | "
                  f"{r['seconds']:.0f} s", flush=True)
            if "n3a_poly" in r or "n3d" in r:
                print(f"[triples] {scheme} polytomies: (a) {r.get('n3a_poly')}; (b) unresolved "
                      f"{r['n3b'].get('n_unresolved_entries')} entries, anchor-sum mismatches "
                      f"{r['n3b'].get('n_anchor_sum_mismatch')}; (d) {r.get('n3d')}", flush=True)
            problems += [f"{scheme}: {p}" for p in r["problems"]]
    for p in problems:
        print(f"[triples] PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
