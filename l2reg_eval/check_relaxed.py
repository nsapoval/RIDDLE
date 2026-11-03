#!/usr/bin/env python3
"""Gates for ``opc.rl.relaxed`` -- the relaxed decoder -- before any relaxed row is trusted.

Pure Python: no model, no jar.  Every gate prints PASS or FAIL, and the script exits 1 on any FAIL.

G0  ``donors()``: on pseudoforests an assignment in which every donor is one of its hybrid's
    parents and no node donates twice; on K_{2,3} and on two cycles sharing a vertex, None -- and
    ``_pseudoforest`` agrees with it on every case.
G1  realisation: the true cluster set of each l2reg truth (level 2) and each camus cam200 truth
    (level 1), with no closure, rebuilds the network -- same clusters, ``hasse_level`` equal to the
    independent ``networks.level`` and to the known level, the eNewick round-tripping through
    ``from_enewick`` -- with the truth's k on the (regular) l2reg networks, and on the camus ones
    the very network the formal realiser builds.
G2  cross-validation: the incremental decoder, with ``check_every_accept`` on (a rebuild after
    every acceptance, and an end-of-decode audit that no refused candidate is legal), picks exactly
    what a from-scratch reference picks -- at every step a rescan from the top of the ranking,
    every probe a full Hasse rebuild, level by ``networks.level`` -- at max_level 1 and 2, on the
    200 best-ranked candidates of camus cam25 cells (support at budget k, seeded random scores at
    budget 4) and the 80 best-supported candidates of l2reg cells.
G3  soundness: with oracle ranking (true clusters first, then support) and max_level 2 the decoder
    keeps every true candidate on all 20 l2reg cells; with max_level 1 it keeps fewer; every final
    state passes ``verify()`` and realises to a network of the level it was capped at.
R   report, not gated: support-ranked decodes at max_level 1, 2 and k beside the formal decoder.

Candidate sets are built from the staged gene trees (~7 s per l2reg cell) and cached under
``runs/check_relaxed/``.

Usage:  gnn_cpu_v1/.venv/bin/python l2reg_eval/check_relaxed.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import common as C

import numpy as np  # noqa: E402

from opc import bitset  # noqa: E402
from opc.clusters import extract  # noqa: E402
from opc.followup import search  # noqa: E402
from opc.rl import camus as CAMUS  # noqa: E402
from opc.rl import camus_nets as CN  # noqa: E402
from opc.rl import relaxed as RX  # noqa: E402
from opc.rl.env import ClusterEnv, OverlapCSR, greedy_support_rollout  # noqa: E402
from opc.simulate import networks as N  # noqa: E402
from opc.solve import candidates, closure  # noqa: E402
from opc.solve import truth as T  # noqa: E402

M, TAU = 1000, 2
CACHE = C.RUNS / "check_relaxed"
FAILS: list[str] = []


def gate(name: str, ok: bool, detail: str) -> None:
    print(f"[{name}] {'PASS' if ok else 'FAIL'}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def ground_set(inputs: Path, phase: str, rep: int, arm: str):
    """``(words, support, truth, taxa)`` for one cell: the tau = 2 candidate set, cached."""
    cell = CAMUS.CamusCell(phase, rep, arm, Path(inputs))
    truth = CAMUS.truth_of(cell)
    taxa = list(truth.labels)
    f = CACHE / f"{C.cell_name(phase, rep, arm)}.npz"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        ti = {lab: i for i, lab in enumerate(taxa)}
        work = Path(tempfile.mkdtemp(dir=CACHE))
        try:
            gz, table = work / "t.nwk.gz", work / "c.parquet"
            CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=M)
            counter = extract.count_newick_file(gz, ti, [M])
            masks, sup, n_raw = counter.finalize(min_support=1)
            extract.write_cluster_table(table, masks, sup, [M], len(taxa), n_raw, source="x", arm=arm)
            words, s, meta = extract.read_cluster_table(table)
            gs = candidates.build(words, s[:, meta["checkpoints"].index(M)], len(taxa), m=M, tau=TAU)
            np.savez(f, words=gs.words, support=np.asarray(gs.support, np.float64))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    z = np.load(f)
    return z["words"], z["support"], truth, taxa


def truth_words(truth, taxa) -> np.ndarray:
    ti = {lab: i for i, lab in enumerate(taxa)}
    return np.stack([bitset.from_indices(sorted(ti[x] for x in c), len(taxa))
                     for c in sorted(CN.label_clusters(truth), key=sorted)])


def reference_decode(words, n, scores, budget, max_level) -> np.ndarray:
    """The decoder's semantics with nothing incremental and nothing clever: at every step, scan
    from the top of the ranking and accept the first candidate whose H(S + c), rebuilt from
    scratch, passes every constraint; stop when a full scan finds none."""
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable").tolist()
    sel: list[int] = []

    def legal(c: int) -> bool:
        full = closure.with_trivial_clusters(words[np.asarray(sel + [c], np.int64)], n)
        parent, child = closure.hasse_covers_words(full, n)
        if np.bincount(child, minlength=full.shape[0]).max() > 2:
            return False
        if closure.reticulation_number(parent, child, full.shape[0]) > budget:
            return False
        try:
            net = RX.network_from_hasse(full, n)
        except ValueError:
            return False
        return N.level(net) <= max_level

    while True:
        taken = set(sel)
        nxt = next((c for c in order if c not in taken and legal(c)), None)
        if nxt is None:
            return np.sort(np.asarray(sel, np.int64))
        sel.append(nxt)


def g0() -> None:
    cases = {
        "cycle-3": ({10: (1, 2), 11: (2, 3), 12: (3, 1)}, True),
        "double edge": ({10: (1, 2), 11: (1, 2)}, True),
        "path": ({10: (1, 2), 11: (2, 3), 12: (3, 4)}, True),
        "cycle with tail": ({10: (1, 2), 11: (2, 3), 12: (3, 1), 13: (1, 4), 14: (4, 5)}, True),
        "K_{2,3}": ({10: (1, 2), 11: (1, 2), 12: (1, 2)}, False),
        "two cycles at a vertex": ({10: (1, 2), 11: (2, 1), 12: (1, 3), 13: (3, 1)}, False),
    }
    bad = []
    for name, (pairs, feasible) in cases.items():
        got = RX.donors(pairs)
        if (got is not None) != feasible or RX._pseudoforest(pairs.values()) != feasible:  # noqa: SLF001
            bad.append(name)
        elif got is not None:
            if set(got) != set(pairs) or any(got[h] not in pairs[h] for h in pairs) \
                    or len(set(got.values())) != len(got):
                bad.append(name)
    gate("G0", not bad, f"donor assignment on {len(cases)} cases" + (f"; wrong: {bad}" if bad else ""))


def g1(stage) -> None:
    """A truth's reticulation count is only recoverable from its clusters when every reticulation
    is visible in them.  The l2reg networks are regular by construction, so their k must come
    back; two camus truths (cam200 rep03, rep10) carry a reticulation no cluster witnesses, so for
    level-1 truths the target is the reticulation number of H(C(N)) -- and the formal realiser,
    closure and all, must build the very same network there."""
    bad, invisible, n_ok = [], [], 0
    truths = [(rec["key"], C.truth_of(rec["rep"]), 2) for rec in stage["cells"]]
    truths += [(f"cam200_rep{r:02d}", C.truth_of(r, inputs=C.GNN / "inputs", phase="cam200"), 1)
               for r in range(20)]
    for name, truth, want_level in truths:
        taxa = list(truth.labels)
        words = truth_words(truth, taxa)
        got = RX.realise(words, np.arange(len(words)), len(taxa), taxa)
        if got is None:
            bad.append(f"{name}: not realised")
            continue
        net, full = got
        parent, child = closure.hasse_covers_words(full, len(taxa))
        rho = closure.reticulation_number(parent, child, full.shape[0])
        hl, nl = RX.hasse_level(parent, child, full.shape[0]), N.level(net)
        back = N.from_enewick(N.to_enewick(net))
        checks = [
            ("clusters", CN.label_clusters(net) == CN.label_clusters(truth)),
            ("k == reticulations of H(C(N))", net.n_retics == rho),
            (f"level {hl}/{nl}/{want_level}", hl == nl == want_level),
            ("eNewick round trip", CN.label_clusters(back) == CN.label_clusters(net)
             and back.n_retics == net.n_retics)]
        if want_level == 2:
            checks.append(("k == truth k (regular)", net.n_retics == truth.n_retics))
        else:
            f = search.realise(words, np.arange(len(words)), len(taxa), taxa)
            checks.append(("formal realiser builds the same network",
                           f is not None and f[0].n_retics == net.n_retics
                           and CN.label_clusters(f[0]) == CN.label_clusters(net)))
            if net.n_retics != truth.n_retics:
                invisible.append(f"{name} (k {truth.n_retics} -> {net.n_retics})")
        problems = [p for p, ok in checks if not ok]
        if problems:
            bad.append(f"{name}: {problems}")
        else:
            n_ok += 1
    gate("G1", not bad, f"{n_ok} of {len(truths)} truths rebuilt exactly (20 level-2, 20 level-1); "
         f"level-1 truths with a cluster-invisible reticulation: {invisible or 'none'}"
         + (f"; {bad[:5]}" if bad else ""))


def g2(stage, quick: bool) -> None:
    runs, bad, t0 = 0, [], time.perf_counter()
    rng = np.random.default_rng(20260911)

    def compare(label, words, n, scores, budget, level):
        nonlocal runs
        dec = RX.RelaxedDecoder(words, n, budget, level, check_every_accept=True)
        got, ref = dec.decode(scores), reference_decode(words, n, scores, budget, level)
        runs += 1
        if not np.array_equal(got, ref) or not dec.verify():
            bad.append(f"{label} L{level}: {len(got)} vs {len(ref)} ({dec.verify_reason})")

    for rep in range(2 if quick else 6):
        words, support, truth, taxa = ground_set(C.GNN / "inputs", "cam25", rep, "cam_iq500")
        n, k = len(taxa), int(truth.n_retics)
        for tag, scores, budget in (("support", support, k),
                                    ("random", rng.permutation(len(words)).astype(float), 4)):
            top = np.argsort(-scores, kind="stable")[:200]
            for level in (1, 2):
                compare(f"cam25 rep{rep} {tag}", words[top], n, scores[top], budget, level)
    for rec in stage["cells"][: 1 if quick else 3]:
        words, support, truth, taxa = ground_set(C.INPUTS, C.PHASE, rec["rep"], C.ARM)
        top = np.argsort(-support, kind="stable")[:80]
        for level in (1, 2):
            compare(rec["key"], words[top], len(taxa), support[top], rec["k"], level)
    gate("G2", not bad, f"incremental == from-scratch reference on {runs - len(bad)} of {runs} "
         f"decodes ({time.perf_counter() - t0:.0f} s)" + (f"; {bad[:5]}" if bad else ""))


def g3_and_report(stage, quick: bool) -> None:
    bad, rows = [], []
    for rec in stage["cells"][: 5 if quick else 20]:
        words, support, truth, taxa = ground_set(C.INPUTS, C.PHASE, rec["rep"], C.ARM)
        n, k = len(taxa), rec["k"]
        tr = T.true_cluster_rows(words, truth)
        is_true = np.zeros(len(words), bool)
        is_true[tr] = True
        oracle = is_true * (support.max() + 1) + support
        kept = {}
        for level in (1, 2):
            dec = RX.RelaxedDecoder(words, n, k, level)
            sel = dec.decode(oracle)
            got = RX.realise(words, sel, n, taxa)
            kept[level] = int(is_true[sel].sum())
            if not dec.verify() or got is None or N.level(got[0]) > level:
                bad.append(f"{rec['key']} L{level}: verify {dec.verify_reason!r}, realised "
                           f"{got is not None}")
        if kept[2] != len(tr) or kept[1] >= kept[2]:
            bad.append(f"{rec['key']}: oracle kept L1 {kept[1]}, L2 {kept[2]} of {len(tr)}")
        # report: support ranking, relaxed at L1 / L2 / k, beside the formal decoder
        n_true = len(CN.label_clusters(truth))

        def f1(sel):
            tp = int(is_true[sel].sum())
            p, r = tp / max(len(sel), 1), tp / n_true
            return 2 * p * r / (p + r) if tp else 0.0
        gs = candidates.build(words, support, n, m=M, tau=TAU)
        formal = f1(greedy_support_rollout(ClusterEnv(gs, budget=k, graph=OverlapCSR.of(gs))))
        line = {"key": rec["key"], "k": k, "oracle_L1": kept[1], "oracle_L2": kept[2],
                "true": len(tr), "formal": formal}
        for name, level in (("L1", 1), ("L2", 2), ("Lk", k)):
            t0 = time.perf_counter()
            dec = RX.RelaxedDecoder(words, n, k, level)
            sel = dec.decode(support)
            line[name] = (f1(sel), round(time.perf_counter() - t0, 1),
                          dec.stats["recheck_accepts"])
        rows.append(line)
    gate("G3", not bad, f"oracle keeps every true candidate at max_level 2 on "
         f"{sum(r['oracle_L2'] == r['true'] for r in rows)} of {len(rows)} cells, fewer at "
         f"max_level 1, every state verified" + (f"; {bad[:5]}" if bad else ""))
    print("[R] support ranking, F1 (decode s, recheck accepts): formal level-1 | relaxed L1 | L2 | Lk")
    for r in rows:
        print(f"    {r['key']:30s} k={r['k']} | {r['formal']:.4f} | "
              + " | ".join(f"{v[0]:.4f} ({v[1]}s, {v[2]})" for v in (r["L1"], r["L2"], r["Lk"]))
              + f" | oracle kept L1 {r['oracle_L1']}, L2 {r['oracle_L2']} of {r['true']}")
    med = lambda key: float(np.median([r[key] if key == "formal" else r[key][0] for r in rows]))  # noqa: E731
    print(f"[R] median F1: formal {med('formal'):.4f} | relaxed L1 {med('L1'):.4f} | "
          f"L2 {med('L2'):.4f} | Lk {med('Lk'):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer cells per gate")
    args = ap.parse_args()
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run l2reg_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    t0 = time.perf_counter()
    g0()
    g1(stage)
    g2(stage, args.quick)
    g3_and_report(stage, args.quick)
    print(f"[check_relaxed] {'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)} "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
