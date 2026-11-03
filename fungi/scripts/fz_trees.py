#!/usr/bin/env python3
"""Stage B1: regenerate, without torch, exactly the gene trees each Stage A job read -- and prove it.

``infer_gnn_cpu.py`` writes the trees it scores into a ``TemporaryDirectory`` that is deleted on exit.
For this grid -- as-written rooting, ``--partial 485``, ``--m 0`` -- its path is short, so it is copied
here rather than imported (``infer_gnn_cpu`` imports torch and sniffs ``sys.argv``):

* ``scan`` (``infer_gnn_cpu.py:232-245``): each line's leaf set by the ``_LEAF`` regex;
* ``select`` (``:247-332``): S is every label in the file, a tree is eligible with >= 485 of them, and
  ``--m 0`` keeps every eligible tree in file order (no RNG);
* ``prepare`` / ``prune_to`` (``:335-377``, ``:427-437``): with no outgroup, ``og and keep == have`` is
  false, so **every** tree goes through ``prune_to`` -- internal labels dropped, lengths ``repr``'d;
  the leaf set must survive intact and the root must be binary (``:449-460``);
* ``count_clusters`` (``:502-520``): ``SupportCounter`` less each tree's own leaf set.

Gates (exit 1 on any failure):

* **N0-trees** -- the count of eligible trees, ``m`` and the label count equal every Stage A manifest
  for the scheme; every regenerated tree is binary with a binary root;
* **N0-support** -- the regenerated counter, filtered at ``support >= tau``, reproduces every present
  Stage A npz's ``words`` and ``support`` **row for row** (set equality is reported separately, so an
  order-only mismatch is named as such).  Also reported: how many partial-tree root clusters (taxon
  samplings, not clades) the published ``extract.count_newick_file`` would have let through at each tau.

Under ``FZ_VARIANT=collapsed`` the input is ``fungi/collapsed/data`` (checked against its stamp), the
trees are multifurcating by design, and N0-trees reads: the root is binary and no node is unary.  Then

* **N0-collapse** -- against the landed ``trees/<scheme>/``: the same labels, ``chosen_sha256`` and
  ``presence.npy`` bytes; per tree (both read by ``blen.parse_with_lengths``, the GNN's own length
  parser), the clusters are a subset of the landed tree's, every landed cluster that is gone had an
  edge < 1e-5 below a non-root parent, no such edge survives, surviving lengths are equal, and the
  number gone is the stamp's ``contracted`` count.

Writes ``<out-root>/trees/<scheme>/``: ``trees.nwk.gz`` (one regenerated tree per line, file order),
``presence.npy`` (bool, trees x taxa in sorted label order), ``trees.json`` and ``parity.json``.

    gnn_cpu_v1/.venv/bin/python fungi/scripts/fz_trees.py --schemes medium,short
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
import time
from pathlib import Path

import fz_common as FZ

import numpy as np  # noqa: E402

from opc.clusters import extract  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

_LEAF = re.compile(r"[(,]\s*([^\s():,;]+)")          # infer_gnn_cpu.py:209


def prune_to(newick: str, keep) -> str:
    """``infer_gnn_cpu.prune_to`` (``:335-377``), verbatim: unary nodes suppressed, internal labels
    gone, lengths ``repr``'d.  Copied because its module imports torch."""
    parent, length, lab, children = OG.parse_newick(newick)
    n = len(parent)
    live = [False] * n
    for i in range(n - 1, -1, -1):
        live[i] = any(live[c] for c in children[i]) if children[i] else lab[i] in keep
    if not live[0]:
        raise ValueError(f"prune_to: no leaf of {newick[:60]!r} is kept")
    kids = [[c for c in children[i] if live[c]] for i in range(n)]

    def walk(i: int) -> tuple[int, float]:
        acc = length[i]
        while len(kids[i]) == 1:
            i = kids[i][0]
            acc += length[i]
        return i, acc

    root = walk(0)[0]
    body: dict[int, str] = {}
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if not kids[v]:
            body[v] = lab[v] or ""
        elif done:
            body[v] = "(" + ",".join(f"{body[w]}:{acc!r}" for w, acc in map(walk, kids[v])) + ")"
        else:
            stack.append((v, True))
            stack.extend((walk(c)[0], False) for c in kids[v])
    return body[root] + ";"


def regenerate(scheme: str, partial: int = FZ.PARTIAL) -> dict:
    src = FZ.DATA / scheme / "genetrees.rooted.tre"
    lines = [ln.strip() for ln in open(src) if ln.strip()]
    leafsets = [set(_LEAF.findall(ln)) for ln in lines]
    labels = sorted(set().union(*leafsets))
    chosen = [i for i, s in enumerate(leafsets) if len(s) >= partial]
    trees, nonbinary, badroot = [], [], []
    for i in chosen:
        line = lines[i]
        if ":" not in line:
            raise ValueError(f"{scheme} tree {i}: no branch lengths")
        t = prune_to(line, leafsets[i])
        parent, _, lab, children = OG.parse_newick(t)
        got = {lab[j] for j in range(len(parent)) if not children[j]}
        if got != leafsets[i]:
            raise ValueError(f"{scheme} tree {i}: the leaf set changed in prune_to")
        if len(children[0]) != 2 or any(len(c) not in (0, 2) for c in children):
            nonbinary.append(i)
        if len(children[0]) != 2 or any(len(c) == 1 for c in children):
            badroot.append(i)
        trees.append(t)
    ti = {t: j for j, t in enumerate(labels)}
    pres = np.zeros((len(chosen), len(labels)), dtype=bool)
    for r, i in enumerate(chosen):
        pres[r, [ti[t] for t in leafsets[i]]] = True
    return {"src": src, "labels": labels, "n_trees": len(lines), "chosen": chosen, "trees": trees,
            "nonbinary": nonbinary, "badroot": badroot, "presence": pres}


def collapse_gate(scheme: str, labels: list[str], trees: list[str], chosen: list[int], pres,
                  problems: list) -> dict:
    """N0-collapse (variant only): these trees are the landed trees with the short edges contracted."""
    from opc.clusters import blen
    ld = FZ.P.LANDED / "trees" / scheme
    li = FZ.read_json(ld / "trees.json")
    st = FZ.P.stamp()
    thr = float(st["threshold"])
    rec = {"landed_trees_sha256": li["trees_sha256"], "threshold": thr}

    def bad(msg):
        problems.append(f"{scheme}: N0-collapse: {msg}")

    if li["labels"] != labels:
        bad("labels differ from the landed run's")
    if li["chosen_sha256"] != hashlib.sha256(np.asarray(chosen, np.int64).tobytes()).hexdigest():
        bad("chosen trees differ from the landed run's")
    lp = np.load(ld / "presence.npy")
    if lp.dtype != pres.dtype or not np.array_equal(lp, pres):
        bad("presence.npy differs from the landed run's")
    raw = [ln.strip() for ln in gzip.open(ld / "trees.nwk.gz", "rt") if ln.strip()]
    if len(raw) != len(trees):
        bad(f"{len(raw)} landed trees, {len(trees)} here")
        return rec
    ti = {t: i for i, t in enumerate(labels)}
    n_gone = n_short_root_child = 0
    not_subset = gone_long = short_survived = len_changed = 0
    for r, c in zip(raw, trees):
        R, C = blen.parse_with_lengths(r, ti), blen.parse_with_lengths(c, ti)
        rmap = {R.mask[i]: (float(R.plen[i]), int(R.parent[i]) == 1) for i in R.internal_nodes()}
        cmap = {C.mask[i]: float(C.plen[i]) for i in C.internal_nodes()}
        if not set(cmap) <= set(rmap):
            not_subset += 1
            continue
        for m, (ln, root_child) in rmap.items():
            if m not in cmap:
                n_gone += 1
                gone_long += not (ln < thr and not root_child)
            elif cmap[m] != ln:
                len_changed += 1
            elif ln < thr:
                n_short_root_child += root_child
                short_survived += not root_child
    want = st["schemes"][scheme]["contracted"]["eligible"]
    for k, v in (("trees whose clusters are not a subset of the landed tree's", not_subset),
                 ("contracted clusters whose edge was >= threshold or under the root", gone_long),
                 ("short non-root-child edges that survived", short_survived),
                 ("surviving edges whose length changed", len_changed)):
        if v:
            bad(f"{v} {k}")
    if n_gone != want:
        bad(f"{n_gone} clusters contracted, the stamp says {want}")
    rec.update({"n_contracted": n_gone, "n_short_root_child_kept": n_short_root_child,
                "stamp_contracted": want, "ok": not (not_subset or gone_long or short_survived
                                                     or len_changed or n_gone != want)})
    return rec


def count_clusters(trees: list[str], taxon_index: dict[str, int], m: int):
    """``infer_gnn_cpu.count_clusters`` (``:502-520``) on the in-memory trees, plus the root clusters it
    drops (for the 'what the published counter would have kept' report)."""
    counter = extract.SupportCounter(len(taxon_index), [m])
    roots: dict[int, int] = {}
    buf: list[int] = []
    for line in trees:
        buf.clear()
        extract.clusters_from_newick(line, taxon_index, buf)
        root = 0
        for x in buf:
            root |= x
        if root == counter.full_mask:
            counter.add_locus(buf)
        else:
            counter.add_locus([x for x in buf if x != root])
            roots[root] = roots.get(root, 0) + 1
    return counter, roots


def words_of(masks, n: int) -> np.ndarray:
    width = ((n + 63) // 64) * 8
    if not len(masks):
        return np.zeros((0, width // 8), dtype=np.uint64)
    return np.stack([np.frombuffer(int(x).to_bytes(width, "little"), dtype=np.uint64) for x in masks])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", default=",".join(FZ.SCHEMES))
    ap.add_argument("--gnn-root", type=Path, default=FZ.ANALYSIS / "gnn")
    ap.add_argument("--out-root", type=Path, default=FZ.ANALYSIS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--require-runs", type=int, default=1,
                    help="fail unless at least this many Stage A npz exist per scheme (0 = trees only)")
    args = ap.parse_args()
    FZ.P.verify_inputs([s for s in args.schemes.split(",") if s])
    problems = []
    for scheme in [s for s in args.schemes.split(",") if s]:
        t0 = time.perf_counter()
        d = FZ.trees_dir(args.out_root, scheme)
        d.mkdir(parents=True, exist_ok=True)
        R = regenerate(scheme)
        labels, trees, n = R["labels"], R["trees"], len(R["labels"])
        if R["nonbinary"] and not FZ.P.VARIANT:
            problems.append(f"{scheme}: {len(R['nonbinary'])} regenerated trees are not binary, e.g. "
                            f"{R['nonbinary'][:5]} (the triple counter assumes binary trees)")
        if R["badroot"] and FZ.P.VARIANT:
            problems.append(f"{scheme}: {len(R['badroot'])} regenerated trees have a non-binary root or a "
                            f"unary node, e.g. {R['badroot'][:5]} (infer_gnn_cpu roots as written)")
        gz = d / "trees.nwk.gz"
        if args.force or not gz.exists():
            tmp = gz.with_name(gz.name + ".tmp")
            # mtime 0 and no file name in the header: the sha256 is a property of the content
            with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as fh:
                fh.write("".join(t + "\n" for t in trees).encode())
            tmp.replace(gz)
            np.save(d / "presence.npy.tmp.npy", R["presence"])
            (d / "presence.npy.tmp.npy").replace(d / "presence.npy")
        else:
            old = [ln.strip() for ln in gzip.open(gz, "rt") if ln.strip()]
            if old != trees:
                problems.append(f"{scheme}: {gz} exists and differs from the regeneration (--force)")
        m = len(trees)
        info = {"scheme": scheme, "src": str(R["src"]), "src_sha256": FZ.sha256(R["src"]),
                "n_trees": R["n_trees"], "n_labels": n, "partial": FZ.PARTIAL, "m": m,
                "chosen_first": R["chosen"][:5],
                "chosen_sha256": hashlib.sha256(np.asarray(R["chosen"], np.int64).tobytes()).hexdigest(),
                "n_nonbinary": len(R["nonbinary"]), "trees_sha256": FZ.sha256(gz),
                "leaves_min": int(R["presence"].sum(1).min()), "leaves_median": float(np.median(R["presence"].sum(1))),
                "taxon_min_trees": int(R["presence"].sum(0).min()), "labels": labels}
        if FZ.P.VARIANT:
            info.update(FZ.P.tag())
            info["n0_collapse"] = collapse_gate(scheme, labels, trees, R["chosen"], R["presence"], problems)
        FZ.write_json(d / "trees.json", info, indent=1)

        # N0-trees + N0-support against every present Stage A job
        counter, roots = count_clusters(trees, {t: i for i, t in enumerate(labels)}, m)
        masks, sup, _ = counter.finalize(min_support=1)
        sup = sup[:, -1].astype(np.int64)
        par = {"scheme": scheme, "m": m, "n_clusters_tau1": int(len(masks)), "runs": {},
               "trees_sha256": info["trees_sha256"]}
        n_runs = 0
        for tau in FZ.TAUS:
            keep = sup >= tau
            W = words_of(masks[keep], n)
            S = sup[keep]
            par[f"partial_root_clusters_tau{tau}"] = int(sum(1 for c in roots.values() if c >= tau))
            for dec in FZ.DECODERS:
                pre = FZ.gnn_prefix(args.gnn_root, scheme, tau, dec)
                npz = pre.with_name(pre.name + ".logits.npz")
                if not npz.exists():
                    continue
                n_runs += 1
                man = FZ.read_json(pre.with_name(pre.name + ".manifest.json"))
                arr = np.load(npz)
                rec = {"npz_sha256": FZ.sha256(npz), "n_clusters": int(len(S))}
                if man["scan"]["eligible"] != m or int(man["m"]) != m or man["scan"]["n_labels"] != n:
                    problems.append(f"{pre.name}: manifest eligible/m/n_labels {man['scan']['eligible']}"
                                    f"/{man['m']}/{man['scan']['n_labels']} != regenerated {m}/{m}/{n}")
                if [str(t) for t in arr["taxa_order"]] != labels:
                    problems.append(f"{pre.name}: taxa_order is not the sorted label set")
                same_rows = (arr["words"].shape == W.shape and np.array_equal(arr["words"], W)
                             and np.array_equal(arr["support"].astype(np.int64), S))
                rec["rows_identical"] = bool(same_rows)
                if not same_rows:
                    a = {(bytes(w.tobytes()), int(s)) for w, s in zip(arr["words"], arr["support"])}
                    b = {(bytes(w.tobytes()), int(s)) for w, s in zip(W, S)}
                    rec["set_equal"] = a == b
                    rec["n_only_npz"], rec["n_only_regen"] = len(a - b), len(b - a)
                    problems.append(f"{pre.name}: N0-support fails ({'order only' if a == b else 'content'}"
                                    f": {len(a - b)} only in npz, {len(b - a)} only regenerated)")
                par["runs"][pre.name] = rec
        if n_runs < args.require_runs:
            problems.append(f"{scheme}: {n_runs} Stage A npz found, --require-runs {args.require_runs}")
        par["n_runs_checked"] = n_runs
        par["ok"] = not [p for p in problems if p.startswith(scheme) or scheme in p]
        FZ.write_json(d / "parity.json", par, indent=1)
        print(f"[trees] {scheme}: {m} trees of {R['n_trees']} ({n} labels), |C_tau1| {len(masks):,}, "
              f"{n_runs} npz checked, {len(R['nonbinary'])} non-binary | "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    for p in problems:
        print(f"[trees] PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
