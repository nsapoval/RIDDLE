#!/usr/bin/env python3
"""Gates for ``opc.rl.relaxed`` -- the relaxed decoder -- before any relaxed row is trusted.

Pure Python: no model, no jar.  Every gate prints PASS or FAIL, and the script exits 1 on any FAIL.

G0  ``donors()``: on pseudoforests an assignment in which every donor is one of its hybrid's
    parents and no node donates twice; on K_{2,3} and on two cycles sharing a vertex, None -- and
    ``_pseudoforest`` agrees with it on every case.
G1  realisation: the true cluster set of each l3reg truth (level 3) and each camus cam200 truth
    (level 1), with no closure, rebuilds the network -- same clusters, ``hasse_level`` equal to the
    independent ``networks.level`` and to the known level, the eNewick round-tripping through
    ``from_enewick`` -- with the truth's k on the (regular) l3reg networks, and on the camus ones
    the very network the formal realiser builds.
G2  cross-validation: the incremental decoder, with ``check_every_accept`` on (a rebuild after
    every acceptance, and an end-of-decode audit that no refused candidate is legal), picks exactly
    what a from-scratch reference picks -- at every step a rescan from the top of the ranking,
    every probe a full Hasse rebuild, level by ``networks.level`` -- at max_level 1, 2 and 3, on
    the 200 best-ranked candidates of camus cam25 cells (support at budget k, seeded random scores
    at budget 4) and the 80 best-supported candidates of l3reg cells.  An audit failure is a FAIL
    line, not a traceback.
G2b regression: the in-degree recheck (``RelaxedDecoder._indeg_lifted``) on the four full decodes
    that exposed its absence.  Each must still be one the pre-patch rule gets wrong -- otherwise
    the fixture is vacuous and the gate fails -- and the decoder must equal an exhaustive-recheck
    oracle (every refusal rechecked after every acceptance) and pass its own audit.
G3  soundness: with oracle ranking (true clusters first, then support), every final state passes
    ``verify()`` and realises to a network of the level it was capped at.  How many true candidates
    each cap keeps is **reported, not gated** -- see "what G3 does not assert" below.
R   report, not gated: support-ranked decodes at max_level 1, 2, 3 and k beside the formal decoder.

The in-degree recheck (fixed 2026-09-12)
----------------------------------------
``RelaxedDecoder._lifted`` rechecks a refusal only when an acceptance could have lifted it.  Its
in-degree clause assumed only a *nested* acceptance could, and that is false: a cluster that merely
overlaps the candidate, inserted between a node inside it and both of that node's parents, takes the
node from two parents to one and lifts the refusal -- and nothing requeued it.  Every decode still
ended in a legal state, but not always the one the decoder's rule defines: some stopped with a legal
candidate refused, more took a different path to a different maximal selection.

Measured before the fix on the true cluster sets of both datasets, under 14 rankings and two
budgets, against the exhaustive oracle: **4 of 1,120** decodes differed on ``l2reg_n200_v1`` (3 of
them stuck) and **24 of 1,680** on ``l3reg_n200_v1`` (8 stuck).  So it was never a level-3 defect;
l2reg's G2 missed it because its slices never reach a late acceptance that removes a reticulation.
With the fix the decoder equals the oracle on all 2,800 and never fails ``verify()``.

No landed row is known to have changed, so the ``relaxed_L*`` tags stay.  Under support ranking no
acceptance ever removed a reticulation (0 of 20,425 on l2reg and l3reg), so the new clause never
fires, and all 40 landed l2reg greedy relaxed rows re-derive exactly on the fixed library.  The
1,500 landed GNN relaxed selections (400 on l2reg, 1,100 in the camus runs behind
``camus_a104_per_network_l2mix.csv``) are all maximal, and an exact replay of l2reg rep12's nine
models -- the driver unchanged, the scores recomputed -- reproduces its landed rows and equals the
oracle.  For GNN rows that is evidence rather than proof: no cell stores its scores, and a wrong
path can still end maximal.

What G3 does not assert
-----------------------
Two tempting claims about the oracle ladder are **not** theorems, and measuring them here is what
showed it.  The decoder is greedy, so a larger legal set is not a better one: raising the cap can
let it accept a higher-ranked cluster early that blocks more true ones later.

* **"a higher cap keeps at least as many true clusters"** is false.  On rep05 the oracle keeps
  199 at L1, **197** at L2 and 203 at L3.  l2reg's G3 gated the same direction (it failed on
  ``kept[1] >= kept[2]``); that held on its 20 level-2 cells but was never guaranteed.
* **"at its own level the oracle keeps every true candidate"** is false too: rep18 keeps 201 of
  202 at L3.  ``tr`` is the true clusters that survive into ``C_tau``, and H(a subset of a regular
  network's clusters) is not a subnetwork of that network, so there is no theorem to appeal to.

Neither is the in-degree defect: the old certificate held on all six decodes, and G3's numbers are
identical before and after the fix.  They are the greedy rule.  What stays hard-gated is what is
actually guaranteed, and what the rows rest on: every state verifies, and every realised network is
within its cap.

Candidate sets are built from the staged gene trees (~7 s per l3reg cell) and cached under
``runs/check_relaxed/``.  The camus ``cam25_*.npz`` half is dataset-independent, so copying it
from ``l2reg_eval/runs/check_relaxed/`` saves rebuilding it.

Usage:  gnn_cpu_v1/.venv/bin/python l3reg_eval/check_relaxed.py [--quick]
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
#: The level caps the ladder is measured at.  ``LEVELS[-1]`` is this dataset's own level, the rung
#: G3 holds to "keeps every true candidate"; the lower rungs are what the cap costs.
LEVELS = (1, 2, 3)
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
    is visible in them.  The l3reg networks are regular by construction, so their k must come
    back; two camus truths (cam200 rep03, rep10) carry a reticulation no cluster witnesses, so for
    level-1 truths the target is the reticulation number of H(C(N)) -- and the formal realiser,
    closure and all, must build the very same network there."""
    bad, invisible, n_ok = [], [], 0
    truths = [(rec["key"], C.truth_of(rec["rep"]), LEVELS[-1]) for rec in stage["cells"]]
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
        if want_level == LEVELS[-1]:
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
    gate("G1", not bad, f"{n_ok} of {len(truths)} truths rebuilt exactly "
         f"(20 level-{LEVELS[-1]}, 20 level-1); "
         f"level-1 truths with a cluster-invisible reticulation: {invisible or 'none'}"
         + (f"; {bad[:5]}" if bad else ""))


def g2(stage, quick: bool) -> None:
    runs, bad, t0 = 0, [], time.perf_counter()
    rng = np.random.default_rng(20260911)

    def compare(label, words, n, scores, budget, level):
        nonlocal runs
        runs += 1
        dec = RX.RelaxedDecoder(words, n, budget, level, check_every_accept=True)
        try:
            got = dec.decode(scores)
        except AssertionError as exc:       # the audit or the rebuild check: a FAIL line, not a traceback
            bad.append(f"{label} L{level}: {exc}")
            return
        ref = reference_decode(words, n, scores, budget, level)
        if not np.array_equal(got, ref) or not dec.verify():
            bad.append(f"{label} L{level}: {len(got)} vs {len(ref)} ({dec.verify_reason})")

    for rep in range(2 if quick else 6):
        words, support, truth, taxa = ground_set(C.GNN / "inputs", "cam25", rep, "cam_iq500")
        n, k = len(taxa), int(truth.n_retics)
        for tag, scores, budget in (("support", support, k),
                                    ("random", rng.permutation(len(words)).astype(float), 4)):
            top = np.argsort(-scores, kind="stable")[:200]
            for level in LEVELS:
                compare(f"cam25 rep{rep} {tag}", words[top], n, scores[top], budget, level)
    for rec in stage["cells"][: 1 if quick else 3]:
        words, support, truth, taxa = ground_set(C.INPUTS, C.PHASE, rec["rep"], C.ARM)
        top = np.argsort(-support, kind="stable")[:80]
        for level in LEVELS:
            compare(rec["key"], words[top], len(taxa), support[top], rec["k"], level)
    gate("G2", not bad, f"incremental == from-scratch reference on {runs - len(bad)} of {runs} "
         f"decodes ({time.perf_counter() - t0:.0f} s)" + (f"; {bad[:5]}" if bad else ""))


def g2b() -> None:
    """The in-degree recheck (``RelaxedDecoder._indeg_lifted``) on the decodes that exposed its absence.

    G2's slices never reach the case -- it needs a late acceptance that removes a reticulation -- so
    these are full true cluster sets under the rankings that tripped it.  Each fixture must still be
    one the pre-patch rule gets wrong, or the gate is vacuous and says so."""

    class Exhaustive(RX.RelaxedDecoder):
        """The decoder's rule with no cleverness in the recheck: every refusal, after every acceptance."""

        def _lifted(self, move, reason):
            return np.flatnonzero(reason)

    class PrePatch(RX.RelaxedDecoder):
        """The recheck rule as it stood before 2026-09-12: no in-degree clause."""

        def _indeg_lifted(self, reason):
            return np.zeros(0, dtype=np.int64)

    fixtures = [  # dataset, key, ranking, budget, level -- measured 2026-09-12
        ("l3reg_n200_v1", "n200_rep04_l3x1_l1x2_cu0.25", "random 7", "k", 2),    # stuck: 201 of 203
        ("l3reg_n200_v1", "n200_rep04_l3x1_l1x2_cu0.25", "random 7", "k", 3),    # stuck: 201 of 203
        ("l3reg_n200_v1", "n200_rep00_l3x2_l1x2_cu0.25", "random 3", "k/2", 3),  # another path, not stuck
        ("l2reg_n200_v1", "n200_rep12_l2x3_l1x1_cu0.25", "size", "k", 2),        # stuck: 204 of 205
    ]
    bad, done = [], 0
    for ds, key, ranking, budget_rule, level in fixtures:
        manifest = C.REPO / ds / "MANIFEST.tsv"
        if not manifest.exists():
            print(f"[G2b] skipped {ds} {key}: no {manifest}", flush=True)
            continue
        row = next(r for r in C.read_tsv(manifest) if r["key"] == key)
        net = N.from_enewick((C.REPO / ds / row["network"]).read_text())
        taxa = list(net.labels)
        words, n, k = truth_words(net, taxa), len(taxa), int(row["k"])
        budget = k if budget_rule == "k" else max(1, k // 2)
        scores = (bitset.popcounts(words).astype(float) if ranking == "size" else
                  np.random.default_rng(int(ranking.split()[1])).permutation(len(words)).astype(float))
        label = f"{ds.split('_')[0]} {key.split('_')[1]} L{level} budget {budget} {ranking}"
        want = Exhaustive(words, n, budget, level).decode(scores)
        if np.array_equal(PrePatch(words, n, budget, level).decode(scores), want):
            bad.append(f"{label}: the pre-patch rule gets this one right -- the fixture is vacuous")
            continue
        try:
            got = RX.RelaxedDecoder(words, n, budget, level, check_every_accept=True).decode(scores)
        except AssertionError as exc:
            bad.append(f"{label}: {exc}")
            continue
        if not np.array_equal(got, want):
            bad.append(f"{label}: {len(got)} selected, the exhaustive recheck {len(want)}")
            continue
        done += 1
    gate("G2b", not bad and done >= 3, f"on {done} decodes the pre-patch recheck rule gets wrong, the "
         f"decoder now equals an exhaustive-recheck oracle and passes its own audit"
         + (f"; {bad}" if bad else ""))


def g3_and_report(stage, quick: bool) -> None:
    bad, short, nonmono, rows = [], [], [], []
    for rec in stage["cells"][: 5 if quick else 20]:
        words, support, truth, taxa = ground_set(C.INPUTS, C.PHASE, rec["rep"], C.ARM)
        n, k = len(taxa), rec["k"]
        tr = T.true_cluster_rows(words, truth)
        is_true = np.zeros(len(words), bool)
        is_true[tr] = True
        oracle = is_true * (support.max() + 1) + support
        kept = {}
        for level in LEVELS:
            dec = RX.RelaxedDecoder(words, n, k, level)
            sel = dec.decode(oracle)
            got = RX.realise(words, sel, n, taxa)
            kept[level] = int(is_true[sel].sum())
            if not dec.verify() or got is None or N.level(got[0]) > level:
                bad.append(f"{rec['key']} L{level}: verify {dec.verify_reason!r}, realised "
                           f"{got is not None}")
        # Both reported, neither gated -- see "what G3 does not assert" in the module docstring.
        # A greedy decoder is not monotone in its legal set, so a higher cap can keep fewer.
        ladder = [kept[lv] for lv in LEVELS]
        if any(a > b for a, b in zip(ladder, ladder[1:])):
            nonmono.append(f"{rec['key']} {ladder}")
        if kept[LEVELS[-1]] != len(tr):
            short.append(f"{rec['key']} {kept[LEVELS[-1]]}/{len(tr)}")
        # report: support ranking, relaxed at L1 / L2 / k, beside the formal decoder
        n_true = len(CN.label_clusters(truth))

        def f1(sel):
            tp = int(is_true[sel].sum())
            p, r = tp / max(len(sel), 1), tp / n_true
            return 2 * p * r / (p + r) if tp else 0.0
        gs = candidates.build(words, support, n, m=M, tau=TAU)
        formal = f1(greedy_support_rollout(ClusterEnv(gs, budget=k, graph=OverlapCSR.of(gs))))
        line = {"key": rec["key"], "k": k, "true": len(tr), "formal": formal,
                **{f"oracle_L{lv}": kept[lv] for lv in LEVELS}}
        for name, level in [(f"L{lv}", lv) for lv in LEVELS] + [("Lk", k)]:
            t0 = time.perf_counter()
            dec = RX.RelaxedDecoder(words, n, k, level)
            sel = dec.decode(support)
            line[name] = (f1(sel), round(time.perf_counter() - t0, 1),
                          dec.stats["recheck_accepts"])
        rows.append(line)
    top = LEVELS[-1]
    gate("G3", not bad, f"every state verified and realised within its cap on {len(rows)} cells x "
         f"{len(LEVELS)} caps"
         + (f"; {bad[:5]}" if bad else ""))
    print(f"[G3] reported, not gated: oracle keeps every true candidate at max_level {top} on "
          f"{sum(r[f'oracle_L{top}'] == r['true'] for r in rows)} of {len(rows)} cells"
          + (f" (short: {short})" if short else "")
          + f"; the ladder falls with a higher cap on {len(nonmono)}"
          + (f": {nonmono}" if nonmono else ""))
    names = [f"L{lv}" for lv in LEVELS] + ["Lk"]
    print(f"[R] support ranking, F1 (decode s, recheck accepts): formal level-1 | relaxed "
          + " | ".join(names))
    for r in rows:
        print(f"    {r['key']:30s} k={r['k']:2d} | {r['formal']:.4f} | "
              + " | ".join(f"{r[nm][0]:.4f} ({r[nm][1]}s, {r[nm][2]})" for nm in names)
              + " | oracle kept "
              + ", ".join(f"L{lv} {r[f'oracle_L{lv}']}" for lv in LEVELS)
              + f" of {r['true']}")
    med = lambda key: float(np.median([r[key] if key == "formal" else r[key][0] for r in rows]))  # noqa: E731
    print(f"[R] median F1: formal {med('formal'):.4f} | relaxed "
          + " | ".join(f"{nm} {med(nm):.4f}" for nm in names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer cells per gate")
    args = ap.parse_args()
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run l3reg_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    t0 = time.perf_counter()
    g0()
    g1(stage)
    g2(stage, args.quick)
    g2b()
    g3_and_report(stage, args.quick)
    print(f"[check_relaxed] {'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)} "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
