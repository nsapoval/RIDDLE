#!/usr/bin/env python3
"""RIDDLE (sweep): one network per k sweep, by majority vote over k = 1..5 -- see README.md.

The rule is the fungi events rule (``fungi/scripts/fz_events.py``), with no taxonomy anchors:

* a reticulation is its **signature** ``(H, {P1 \\ H, P2 \\ H})`` -- the hybrid cluster and the unordered
  pair of sister sets -- read exactly as ``fz_consensus.read_network`` reads it;
* a signature is **kept** when it is in at least ``MIN_VOTES`` of RIDDLE's networks at ``VOTE_KS``;
* the **backbone** is RIDDLE's own k = 0 tree, and each kept signature adds ``H``, ``A u H`` and ``B u H``
  (its hybrid and both parent clusters).  So the sweep is a cluster set, scored like any other.

It also gives every RIDDLE and ASTRAL-IV row the RF that collate leaves empty on networks: ``rf_hw``,
every hardwired cluster read as an unrooted split, against the true tree's; ``rf_hw_norm`` divides by
``2 (n - 3)``.  On a tree that is collate's ``rf`` exactly (gate W2).

Rows (``results/sptree_sweep_rows.csv``), one per (size, rep, method, k):

    RIDDLE         k = 0..10, precision / recall / F1 as collate scored them
    RIDDLE sweep   k empty; ``k_pred`` = ``n_kept``, ``n_cand`` signatures voted on, ``n_added`` clusters
                   the kept ones add to the backbone, ``votes`` every candidate's count
    ASTRAL-IV      k empty, as collate scored it

Gates (``results/sptree_sweep_gates.json``; exit 1 on any):

    W1  every RIDDLE / ASTRAL row's clusters, re-read from its eNewick, give collate's (net_tp, net_n)
    W2  rf_hw == rf wherever collate has an rf (every tree)
    W3  every (size, rep) with RIDDLE at k = 0..5 has one sweep row; its backbone is a tree and the
        sweep contains it
    W4  the signatures read off ``hardwired_clusters`` equal those of an independent recursive walk,
        on every voted network

Usage:  gnn_cpu_v1/.venv/bin/python sptree_eval/sweep.py   (after collate.py)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict

import collate as CO

SP = CO.SP
VOTE_KS = tuple(range(1, 6))
MIN_VOTES = 3
SWEEP = "RIDDLE sweep"
COLS = ("size", "m", "k", "rep", "cu", "key", "method", "tp", "fp", "fn", "n_sel", "precision", "recall",
        "f1", "rf_hw", "rf_hw_norm", "k_pred", "n_cand", "n_kept", "n_added", "votes")


def signatures(net) -> set:
    """``fz_consensus.read_network``'s reticulation signatures, as label sets."""
    lab = net.labels
    hw = {v: frozenset(lab[i] for i in c) for v, c in CO.NW.hardwired_clusters(net).items()}
    out = set()
    for ret in net.retics:
        H = hw[ret.hybrid]
        out.add((H, frozenset(hw[p] - H for p in net.net_parents(ret.hybrid))))
    return out


def signatures_walk(net) -> set:
    """The same, from a recursive walk of ``net_children`` (W4's second opinion)."""
    memo = {}

    def below(v):
        if v not in memo:
            memo[v] = (frozenset([net.labels[net.leaf_taxon[v]]]) if net.is_leaf(v)
                       else frozenset().union(*map(below, net.net_children(v))))
        return memo[v]
    return {(below(r.hybrid), frozenset(below(p) - below(r.hybrid) for p in net.net_parents(r.hybrid)))
            for r in net.retics}


def scored(clusters: set, T: set, TS: set) -> dict:
    rf = len(CO.splits(clusters) ^ TS)
    return {**CO.counts(len(clusters & T), len(clusters)), "rf_hw": rf,
            "rf_hw_norm": rf / (2 * (SP.N_TAXA - 3))}


def main() -> int:
    t0 = time.perf_counter()
    csv.field_size_limit(sys.maxsize)
    with open(SP.RESULTS / "sptree_rows.csv") as fh:
        src = [r for r in csv.DictReader(fh) if r["method"] in (CO.RIDDLE, CO.ASTRAL)]
    truths = {}
    gates = {g: [] for g in ("W1", "W2", "W3", "W4")}
    rows, sweeps = [], defaultdict(dict)
    for r in src:
        rep = int(r["rep"])
        if rep not in truths:
            T = CO.truth_of(rep)[1]
            truths[rep] = (T, CO.splits(T))
        T, TS = truths[rep]
        net = CO.NW.from_enewick(r["enewick"])
        cl = CO.CN.label_clusters(net)
        s = scored(cl, T, TS)
        tag = f"{r['size']} rep{rep:02d} {r['method']} k{r['k']}"
        if (s["tp"], s["n_sel"]) != (int(r["net_tp"]), int(r["net_n"])):
            gates["W1"].append(f"{tag}: {s['tp']}/{s['n_sel']} re-read, {r['net_tp']}/{r['net_n']} collated")
        if r["rf"] != "" and s["rf_hw"] != int(r["rf"]):
            gates["W2"].append(f"{tag}: rf_hw {s['rf_hw']}, rf {r['rf']}")
        base = {c: r[c] for c in ("size", "m", "k", "rep", "cu", "key", "method")}
        # precision / recall / F1 stay collate's: the selection's, which for these methods is the network's
        rows.append({**base, **{c: r[c] for c in ("tp", "fp", "fn", "n_sel", "precision", "recall", "f1")},
                     "rf_hw": s["rf_hw"], "rf_hw_norm": s["rf_hw_norm"], "k_pred": r["k_pred"]})
        if r["method"] == CO.RIDDLE and int(float(r["k"])) <= max(VOTE_KS):
            sweeps[(r["size"], rep)][int(float(r["k"]))] = (base, net, cl)

    n_sweep = 0
    for (size, rep), by_k in sweeps.items():
        if set(by_k) != {0, *VOTE_KS}:
            continue
        T, TS = truths[rep]
        base, net0, c0 = by_k[0]
        votes = Counter()
        for k in VOTE_KS:
            sig = signatures(by_k[k][1])
            if sig != signatures_walk(by_k[k][1]):
                gates["W4"].append(f"{size} rep{rep:02d} k{k}: the two signature readers disagree")
            votes.update(sig)
        kept = [s for s, v in votes.items() if v >= MIN_VOTES]
        add = {c for H, sis in kept for c in (H, *(H | A for A in sis)) if 1 < len(c) < SP.N_TAXA}
        cs = c0 | add
        if net0.n_retics or not c0 <= cs:
            gates["W3"].append(f"{size} rep{rep:02d}: backbone has {net0.n_retics} reticulations")
        rows.append({**base, "k": "", "method": SWEEP, **scored(cs, T, TS), "k_pred": len(kept),
                     "n_cand": len(votes), "n_kept": len(kept), "n_added": len(cs) - len(c0),
                     "votes": " ".join(str(v) for v in sorted(votes.values(), reverse=True))})
        n_sweep += 1
    want = sum(1 for by_k in sweeps.values() if set(by_k) >= {0, *VOTE_KS})
    if n_sweep != want or n_sweep == 0:
        gates["W3"].append(f"{n_sweep} sweep rows, {want} complete sweeps")

    sizes = [t for t, _, _ in SP.SIZES]
    order = {CO.RIDDLE: 0, SWEEP: 1, CO.ASTRAL: 2}
    rows.sort(key=lambda r: (sizes.index(r["size"]), int(r["rep"]), order[r["method"]],
                             -1 if r["k"] == "" else float(r["k"])))
    CO.write_csv(SP.RESULTS / "sptree_sweep_rows.csv", rows, COLS)
    have = Counter((r["size"], r["method"]) for r in rows)
    SP.write_atomic(SP.RESULTS / "sptree_sweep_gates.json", json.dumps({
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "vote_ks": list(VOTE_KS), "min_votes": MIN_VOTES,
        "have": {f"{s} {m}": n for (s, m), n in sorted(have.items())},
        "gates": {g: {"ok": not v, "n": len(v), "first": v[:20]} for g, v in gates.items()},
    }, indent=1) + "\n")
    print(f"[sweep] {len(rows)} rows, {n_sweep} sweeps | gates "
          + " ".join(f"{g}:{'ok' if not v else f'FAIL({len(v)})'}" for g, v in gates.items())
          + f" | {time.perf_counter() - t0:.0f} s -> {SP.RESULTS}")
    for g, v in gates.items():
        for x in v[:5]:
            print(f"[sweep]   {g} {x}")
    return 1 if any(gates.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
