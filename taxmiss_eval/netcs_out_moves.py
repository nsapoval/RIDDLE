"""Every netcs taxmiss tree against its parent: is the condition a pure deletion, and where it is not,
did OUT move only among the true-root oracle's tied edges?

Per tree i of a cell (staged S[i], parent P[i], both with OUT):
  ingroup   OUT deleted, the unrooted ingroup splits of P[i] pruned to S[i]'s taxa == S[i]'s   (must hold)
  pure      with OUT, the unrooted splits of P[i] pruned == S[i]'s                            (T1b's claim)
  tied      the oracle (netcs_eval.common.insert_out_at_truth's cost) has > 1 least-cost edge on the full tree
  in_tied   where not pure: S[i]'s OUT edge, as a bipartition of the present taxa, is the restriction of one
            of the full tree's least-cost edges                                               (must hold)

Usage: netcs_out_moves.py <jobs> <out.tsv>.  Output: results/netcs_p0.90_c24_out_moves.tsv (README, NetCS).
"""
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO + "/netcs_eval")
import common as C   # noqa: E402  netcs_eval's
import stage as NS   # noqa: E402  netcs_eval's
sys.path.insert(0, REPO + "/gnn_cpu_v1")
import infer_gnn_cpu as IG  # noqa: E402
from opc.simulate import outgroup as OG  # noqa: E402

LEAF = re.compile(r"[(,]\s*([^(),:;#\s\[\]]+)")


def clusters(newick):
    parent, _l, lab, children = OG.parse_newick(newick.strip())
    below = {}
    for i in range(len(parent) - 1, -1, -1):
        below[i] = frozenset(x for c in children[i] for x in below[c]) if children[i] else frozenset([lab[i]])
    allx = below[0]
    return allx, {s for i, s in below.items() if children[i] and 1 < len(s) < len(allx)}


def with_out(newick):
    allx, cl = clusters(newick)
    return {frozenset(s) if "OUT" in s else frozenset(allx - s)
            for c in cl for s in (c, allx - c) if 1 < len(s) < len(allx) - 1}


def ingroup(newick):
    allx, cl = clusters(newick)
    ing = allx - {"OUT"}
    out = set()
    for c in cl:
        s = c - {"OUT"}
        t = ing - s
        if len(s) > 1 and len(t) > 1:
            out.add(min(s, t, key=sorted))
    return out


def out_bipartition(newick):
    """The present ingroup's split at OUT's edge: the two sides once OUT is the root."""
    r = OG.reroot_on_outgroup(newick, keep_lengths=False)
    parent, _l, lab, children = OG.parse_newick(r)
    below = {}
    for i in range(len(parent) - 1, -1, -1):
        below[i] = frozenset(x for c in children[i] for x in below[c]) if children[i] else frozenset([lab[i]])
    sides = [below[c] for c in children[0] if below[c] != frozenset(["OUT"])]
    allin = below[0] - {"OUT"}
    if len(sides) == 1:                      # OUT, (ingroup): one more level down
        node = next(c for c in children[0] if below[c] != frozenset(["OUT"]))
        sides = [below[c] for c in children[node]]
    return frozenset(min(s, allin - s, key=sorted) for s in sides[:1])


def tied_sides(newick, index, targets):
    """Every least-cost edge's leaf set below it, on the full tree (OUT absent)."""
    parent, _l, lab, children = OG.parse_newick(newick)
    nn, n = len(parent), len(index)
    mask = [0] * nn
    for i in range(nn - 1, -1, -1):
        if not children[i]:
            mask[i] = 1 << index[lab[i]]
        if i:
            mask[parent[i]] |= mask[i]
    costs = [(C.split_cost(mask[i], targets, n), mask[i]) for i in range(1, nn)]
    best = min(c for c, _ in costs)
    names = {j: x for x, j in index.items()}
    return best, [(c - best, frozenset(names[j] for j in range(n) if m >> j & 1)) for c, m in costs]


def scan_cell(args):
    level, n, rep, ils = args
    _bad, t = NS.stage_truth(level, n, rep)
    d = f"L{level}_n{n}/{rep:02d}"
    S = [x.strip() for x in open(f"{REPO}/taxmiss_eval/inputs/netcs/p0.90_c24/{d}/est_ils{ils}.nwk") if x.strip()]
    P = [x.strip() for x in open(f"{REPO}/netcs_eval/inputs/{d}/est_ils{ils}.nwk") if x.strip()]
    row = dict(cell=f"netcs_L{level}_n{n}_rep{rep:02d}_netcs_ils{ils}", trees=len(S), ingroup_bad=0, moved=0,
               tied=0, moved_untied=0, moved_not_in_tied=0, excess_max=0)
    taxa = set(t["index"])
    for i in range(len(S)):
        keep = set(LEAF.findall(S[i]))
        pr = IG.prune_to(P[i], keep)
        if ingroup(pr) != ingroup(S[i]):
            row["ingroup_bad"] += 1
            continue
        full = IG.prune_to(P[i], taxa)                       # the parent's tree, OUT deleted
        best, edges = tied_sides(full, t["index"], t["targets"])
        sides = [e for x, e in edges if x == 0]
        row["tied"] += len(sides) > 1
        if with_out(pr) == with_out(S[i]):
            continue
        row["moved"] += 1
        row["moved_untied"] += len(sides) == 1
        present = frozenset(keep - {"OUT"})
        want = out_bipartition(S[i])
        got = {frozenset([min(s & present, present - (s & present), key=sorted)]) for s in sides
               if 0 < len(s & present) < len(present)}
        row["moved_not_in_tied"] += want not in got
        if want not in got:
            ex = [x for x, e in edges if 0 < len(e & present) < len(present)
                  and frozenset([min(e & present, present - (e & present), key=sorted)]) == want]
            row["excess_max"] = max(row["excess_max"], min(ex) if ex else -1)
    return row


def main():
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = sys.argv[2]
    man = list(csv.DictReader(open(f"{REPO}/taxmiss_eval/inputs/netcs/MANIFEST.tsv"), delimiter="\t"))
    work = []
    for r in man:
        lv, nn = r["key"].split("/")[0].split("_")
        work.append((int(lv[1:]), int(nn[1:]), int(r["rep"]), r["arm"].removeprefix("netcs_ils")))
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, row in enumerate(ex.map(scan_cell, work, chunksize=4), 1):
            rows.append(row)
            if k % 100 == 0:
                print(f"{k}/{len(work)}", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    tot = {k: (max if k == "excess_max" else sum)(r[k] for r in rows) for k in rows[0] if k != "cell"}
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
