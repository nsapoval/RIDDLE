#!/usr/bin/env python3
"""Gates for ``opc.rl.relaxed`` -- the relaxed decoder -- on netcs_bench_corpus, before any relaxed
row is trusted.

Pure Python: no model, no jar.  Every gate prints PASS or FAIL, and the script exits 1 on any FAIL.
A copy of ``l3reg_eval/check_relaxed.py``; ``diff -u`` shows what changed and why.

G0  ``donors()`` on pseudoforests and non-pseudoforests -- unchanged.
G1  realisation: the true cluster set of **every one of the 300 truths**, with no closure, realises
    to a network with exactly those clusters, whose reticulation count is that of H(C(N)), whose
    ``hasse_level`` equals the independent ``networks.level``, and whose eNewick round-trips --
    or is refused because some cluster has **three or more covering parents** in H(C(N)), which no
    decoder with in-degree <= 2 can represent at any cap.  That is a property of the truth, and is
    reported (2 of 300: ``netcs_L2_n100`` 10105 and 10365, measured 2026-09-12); a refusal for any
    other reason fails the gate.
    **Not gated, reported:** whether k and the level come back as the truth's own.  l3reg's truths
    were regular by construction, so it could gate both; these are not, and a reticulation no
    cluster witnesses -- the two size-2 cycles, for a start -- cannot come back from clusters.
    The formal realiser's agreement on the level-1 truths is reported beside it.
G2  cross-validation: the incremental decoder (``check_every_accept``) picks exactly what a
    from-scratch reference picks, at max_level 1 and 2, on the 200 best-ranked candidates of camus
    cam25 cells (unchanged) and the 80 best-supported candidates of two networks per (level, n)
    at ILS 1.0.
G2b regression: the in-degree recheck on the four decodes that exposed its absence -- unchanged, on
    their l2reg/l3reg fixtures.
G3  soundness: with oracle ranking, every final state passes ``verify()`` and realises within its
    cap, on four networks per (level, n) at ILS 1.0.  How many true candidates each cap keeps is
    **reported, not gated** (l3reg_eval/check_relaxed.py, "what G3 does not assert").
R   report, not gated: support-ranked decodes at max_level 1, 2 and k beside the formal decoder.

Candidate sets are built from the staged gene trees, with m = the file's locus count as the
drivers use it, and cached under ``runs/check_relaxed/``.  The camus ``cam25_*.npz`` half is
dataset-independent and is copied from ``l3reg_eval/runs/check_relaxed/`` when it is there.

Usage:  gnn_cpu_v1/.venv/bin/python netcs_eval/check_relaxed.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import Counter
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
#: The level caps the ladder is measured at: this corpus is level 1 and level 2.
LEVELS = (1, 2)
SAMPLE_ARM = C.arm_name("1.0")
CACHE = C.RUNS / "check_relaxed"
L3REG_CACHE = C.REPO / "l3reg_eval" / "runs" / "check_relaxed"
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
    CACHE.mkdir(parents=True, exist_ok=True)
    if not f.exists() and (L3REG_CACHE / f.name).exists():
        shutil.copyfile(L3REG_CACHE / f.name, f)
    if not f.exists():
        ti = {lab: i for i, lab in enumerate(taxa)}
        work = Path(tempfile.mkdtemp(dir=CACHE))
        try:
            gz, table = work / "t.nwk.gz", work / "c.parquet"
            info = CAMUS.prepare_gene_trees(cell, gz, taxa, max_loci=M)
            m = min(M, int(info["n_loci"]))
            counter = extract.count_newick_file(gz, ti, [m])
            masks, sup, n_raw = counter.finalize(min_support=1)
            extract.write_cluster_table(table, masks, sup, [m], len(taxa), n_raw, source="x", arm=arm)
            words, s, meta = extract.read_cluster_table(table)
            gs = candidates.build(words, s[:, meta["checkpoints"].index(m)], len(taxa), m=m, tau=TAU)
            np.savez(f, words=gs.words, support=np.asarray(gs.support, np.float64))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    z = np.load(f)
    return z["words"], z["support"], truth, taxa


def sample(stage, per_phase: int) -> list[dict]:
    """The first ``per_phase`` networks of every (level, n), at ILS 1.0."""
    out, seen = [], Counter()
    for c in sorted(stage["cells"], key=lambda c: (c["phase"], c["rep"])):
        if c["arm"] == SAMPLE_ARM and seen[c["phase"]] < per_phase:
            out.append(c)
            seen[c["phase"]] += 1
    return out


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


def g1(stage, quick: bool) -> None:
    """Every truth's own cluster set, realised with no closure (module docstring)."""
    nets = stage["networks"]
    if quick:
        seen = Counter()
        nets = [r for r in sorted(nets, key=lambda r: (r["phase"], r["rep"]))
                if (seen.update([r["phase"]]) or seen[r["phase"]] <= 2)]
    bad, unrealisable, k_moved, level_moved, formal = [], [], [], [], Counter()
    t0 = time.perf_counter()
    for rec in nets:
        name = f"{rec['phase']}/{rec['rep']}"
        truth = C.truth_of(rec["rep"], phase=rec["phase"])
        taxa = list(truth.labels)
        words = truth_words(truth, taxa)
        got = RX.realise(words, np.arange(len(words)), len(taxa), taxa)
        if got is None:
            # A cluster with three covering parents cannot be one node of in-degree <= 2, at any
            # cap: a property of the truth, reported.  Any other refusal is a failure.
            full = closure.with_trivial_clusters(words, len(taxa))
            _parent, child = closure.hasse_covers_words(full, len(taxa))
            indeg = int(np.bincount(child, minlength=full.shape[0]).max())
            if indeg > 2:
                unrealisable.append(f"{name} (a cluster with {indeg} covering parents)")
            else:
                bad.append(f"{name}: not realised, yet no cluster has more than two covering parents")
            continue
        net, full = got
        parent, child = closure.hasse_covers_words(full, len(taxa))
        rho = closure.reticulation_number(parent, child, full.shape[0])
        hl, nl = RX.hasse_level(parent, child, full.shape[0]), N.level(net)
        back = N.from_enewick(N.to_enewick(net))
        checks = [
            ("clusters", CN.label_clusters(net) == CN.label_clusters(truth)),
            ("k == reticulations of H(C(N))", net.n_retics == rho),
            (f"hasse_level {hl} == networks.level {nl}", hl == nl),
            ("eNewick round trip", CN.label_clusters(back) == CN.label_clusters(net)
             and back.n_retics == net.n_retics)]
        problems = [p for p, ok in checks if not ok]
        if problems:
            bad.append(f"{name}: {problems}")
            continue
        if net.n_retics != truth.n_retics:
            k_moved.append(f"{name} (k {truth.n_retics} -> {net.n_retics})")
        if nl != rec["level"]:
            level_moved.append(f"{name} (level {rec['level']} -> {nl})")
        if rec["level"] == 1:
            f = search.realise(words, np.arange(len(words)), len(taxa), taxa)
            formal["same" if f is not None and f[0].n_retics == net.n_retics
                   and CN.label_clusters(f[0]) == CN.label_clusters(net) else "differs"] += 1
    n_real = len(nets) - len(unrealisable)
    gate("G1", not bad, f"{n_real - len(bad)} of {n_real} realisable truths realise from their own "
         f"clusters with every hard check ({time.perf_counter() - t0:.0f} s)"
         + (f"; {bad[:5]}" if bad else ""))
    print(f"[G1] reported, not gated: {len(unrealisable)} of {len(nets)} truths are not a Hasse "
          f"diagram of in-degree <= 2 at all: {unrealisable or 'none'}", flush=True)
    print(f"[G1] reported, not gated: k differs from the truth's on {len(k_moved)} "
          f"{k_moved[:12] if k_moved else ''}; the level on {len(level_moved)} "
          f"{level_moved[:12] if level_moved else ''}; the formal realiser builds the same network "
          f"on {formal['same']} of {formal['same'] + formal['differs']} level-1 truths", flush=True)


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
    for rec in sample(stage, 1 if quick else 2):
        words, support, truth, taxa = ground_set(C.INPUTS, rec["phase"], rec["rep"], rec["arm"])
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
    for rec in sample(stage, 1 if quick else 4):
        words, support, truth, taxa = ground_set(C.INPUTS, rec["phase"], rec["rep"], rec["arm"])
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
        # Both reported, neither gated -- see "what G3 does not assert" in l3reg_eval's docstring.
        ladder = [kept[lv] for lv in LEVELS]
        if any(a > b for a, b in zip(ladder, ladder[1:])):
            nonmono.append(f"{rec['key']} {ladder}")
        if kept[rec["level"]] != len(tr):
            short.append(f"{rec['key']} {kept[rec['level']]}/{len(tr)}")
        n_true = len(CN.label_clusters(truth))

        def f1(sel):
            tp = int(is_true[sel].sum())
            p, r = tp / max(len(sel), 1), tp / n_true
            return 2 * p * r / (p + r) if tp else 0.0
        gs = candidates.build(words, support, n, m=M, tau=TAU)
        formal = f1(greedy_support_rollout(ClusterEnv(gs, budget=k, graph=OverlapCSR.of(gs))))
        line = {"key": rec["key"], "k": k, "level": rec["level"], "true": len(tr), "formal": formal,
                **{f"oracle_L{lv}": kept[lv] for lv in LEVELS}}
        for name, level in [(f"L{lv}", lv) for lv in LEVELS] + [("Lk", k)]:
            t0 = time.perf_counter()
            dec = RX.RelaxedDecoder(words, n, k, level)
            sel = dec.decode(support)
            line[name] = (f1(sel), round(time.perf_counter() - t0, 1),
                          dec.stats["recheck_accepts"])
        rows.append(line)
    gate("G3", not bad, f"every state verified and realised within its cap on {len(rows)} cells x "
         f"{len(LEVELS)} caps" + (f"; {bad[:5]}" if bad else ""))
    print(f"[G3] reported, not gated: oracle keeps every true candidate at the cell's own level on "
          f"{sum(r[f'oracle_L{r['level']}'] == r['true'] for r in rows)} of {len(rows)} cells"
          + (f" (short: {short})" if short else "")
          + f"; the ladder falls with a higher cap on {len(nonmono)}"
          + (f": {nonmono}" if nonmono else ""))
    names = [f"L{lv}" for lv in LEVELS] + ["Lk"]
    print("[R] support ranking, F1 (decode s, recheck accepts): formal level-1 | relaxed "
          + " | ".join(names))
    for r in rows:
        print(f"    {r['key']:34s} k={r['k']:2d} | {r['formal']:.4f} | "
              + " | ".join(f"{r[nm][0]:.4f} ({r[nm][1]}s, {r[nm][2]})" for nm in names)
              + " | oracle kept "
              + ", ".join(f"L{lv} {r[f'oracle_L{lv}']}" for lv in LEVELS)
              + f" of {r['true']}")
    for level in LEVELS:
        g = [r for r in rows if r["level"] == level]
        if not g:
            continue
        med = lambda key: float(np.median([r[key] if key == "formal" else r[key][0] for r in g]))  # noqa: E731
        print(f"[R] level-{level} cells, median F1: formal {med('formal'):.4f} | relaxed "
              + " | ".join(f"{nm} {med(nm):.4f}" for nm in names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer truths and cells per gate")
    args = ap.parse_args()
    if not C.STAGE.exists():
        raise SystemExit(f"no {C.STAGE}: run netcs_eval/stage.py first")
    stage = json.loads(C.STAGE.read_text())
    t0 = time.perf_counter()
    g0()
    g1(stage, args.quick)
    g2(stage, args.quick)
    g2b()
    g3_and_report(stage, args.quick)
    print(f"[check_relaxed] {'PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)} "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
