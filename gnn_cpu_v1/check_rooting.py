#!/usr/bin/env python3
"""The gate for ``rooting.py``: midpoint and species-tree rooting against dendropy.  Exits 1 on any
disagreement, and on any case the fixtures failed to produce.

    .venv/bin/python check_rooting.py              # ~1 min
    .venv/bin/python check_rooting.py --cases 2000 --seed 7

dendropy is the oracle and is deliberately **not** in ``requirements-cpu.txt`` -- inference never
imports it.  It is pure Python: ``.venv/bin/pip install dendropy``.

What is checked, on random trees (lengths continuous, so midpoints never tie):

* **midpoint** -- the root clades equal dendropy's ``reroot_at_midpoint``; the farthest leaf on each
  side of the root is ``D/2`` from it.
* **species tree** -- an independent reference restricts the species tree with dendropy's
  ``extract_tree_with_taxa_labels``, reads every split of the gene tree off dendropy's own
  bipartitions, and says which split the root belongs on (or that none exists).  ``rooting.py``
  must agree on every case: the same root clades and ``kind`` when one exists, the same
  ``RootingError`` reason when none does.  The two root edges are each half the gene tree's edge
  for that split -- **summed** over both written root edges when the tree is written rooted.
* **both** -- rooting changes no unrooted split (RF 0) and no leaf-to-leaf distance (to 1e-9).
* **not vacuous** -- every case below must have occurred at least once, or the gate fails: a one-
  taxon and a clade rooting, ``split_absent``, a species-tree root polytomy both rooted and
  ``polytomy_no_split``, gene trees written with a bifurcating root, and gene-tree polytomies.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pylib"))
sys.path.insert(1, str(HERE))

try:
    import dendropy
    from dendropy.calculate import treecompare
except ImportError:
    raise SystemExit("check_rooting.py needs dendropy as its oracle: .venv/bin/pip install dendropy")

import rooting as RT  # noqa: E402

TOL = 1e-9


def random_tree(rng: random.Random, names, *, polytomy=0.0, root_polytomy=False) -> str:
    """A rooted Newick over ``names`` by random joins; a join takes three with prob ``polytomy``."""
    items = list(names)
    stop = 3 if root_polytomy and len(items) >= 3 else 2
    while len(items) > stop:
        j = 3 if len(items) - 2 >= stop and rng.random() < polytomy else 2
        rng.shuffle(items)
        group, items = items[:j], items[j:]
        items.append("(" + ",".join(f"{s}:{rng.uniform(0.01, 1.0):.6f}" for s in group) + ")")
    return "(" + ",".join(f"{s}:{rng.uniform(0.01, 1.0):.6f}" for s in items) + ");"


def parse(newick: str, ns, rooting: str):
    return dendropy.Tree.get(data=newick, schema="newick", rooting=rooting, taxon_namespace=ns,
                             preserve_underscores=True)


def write(tree) -> str:
    return tree.as_string(schema="newick", suppress_rooting=True,
                          suppress_internal_node_labels=True).strip()


def leafset(node) -> frozenset:
    return frozenset(x.taxon.label for x in node.leaf_nodes())


def root_clades(newick: str, ns) -> set:
    return {leafset(c) for c in parse(newick, ns, "force-rooted").seed_node.child_nodes()}


def as_written(rng: random.Random, newick: str, ns, seen: Counter) -> str:
    """The same tree, written from a random internal node (a trifurcation or wider) 70% of the time
    and with its bifurcating root the rest."""
    t = parse(newick, ns, "force-rooted")
    if rng.random() < 0.7:
        node = rng.choice([x for x in t.preorder_node_iter() if not x.is_leaf()])
        if node is not t.seed_node:
            t.reroot_at_node(node, suppress_unifurcations=True)
    if len(t.seed_node.child_nodes()) == 2:
        seen["gene_written_bifurcating"] += 1
    if any(len(x.child_nodes()) > 2 for x in t.preorder_node_iter() if x is not t.seed_node):
        seen["gene_polytomy"] += 1
    return write(t)


def unrooted_invariants(before: str, after: str, ns) -> str | None:
    """None, or what rooting changed: an unrooted split, or a leaf-to-leaf distance."""
    a, b = parse(before, ns, "force-unrooted"), parse(after, ns, "force-unrooted")
    if treecompare.symmetric_difference(a, b) != 0:
        return "the unrooted topology changed"
    pa, pb = a.phylogenetic_distance_matrix(), b.phylogenetic_distance_matrix()
    taxa = [x.taxon for x in a.leaf_node_iter()]
    for x, y in combinations(taxa, 2):
        da, db = pa.patristic_distance(x, y), pb.patristic_distance(x, y)
        if abs(da - db) > TOL * max(1.0, da):
            return f"d({x.label}, {y.label}) {da!r} -> {db!r}"
    return None


def diameter(newick: str, ns) -> float:
    t = parse(newick, ns, "force-unrooted")
    pdm = t.phylogenetic_distance_matrix()
    return max(pdm.patristic_distance(x, y)
               for x, y in combinations([n.taxon for n in t.leaf_node_iter()], 2))


def check_midpoint(gene: str, ns) -> str | None:
    ours = RT.midpoint_root(gene)
    ref = parse(gene, ns, "force-unrooted")
    ref.reroot_at_midpoint(update_bipartitions=False)
    if root_clades(ours, ns) != {leafset(c) for c in ref.seed_node.child_nodes()}:
        return f"root clades differ from dendropy's: ours {ours} vs {write(ref)}"
    half = diameter(gene, ns) / 2
    rooted = parse(ours, ns, "force-rooted")
    for side in rooted.seed_node.child_nodes():
        far = max(x.distance_from_root() for x in side.leaf_nodes())
        if abs(far - half) > TOL * max(1.0, half):
            return f"the farthest leaf below {sorted(leafset(side))[:3]} is {far!r} from the root, not D/2 = {half!r}"
    return unrooted_invariants(gene, ours, ns)


def reference_species_root(species: str, gene: str, ns):
    """``(clade, edge length)`` the root belongs on, or ``(reason, None)`` -- dendropy only."""
    g = parse(gene, ns, "force-unrooted")
    R = leafset(g.seed_node)
    norm = lambda c: frozenset((c, R - c))                              # noqa: E731
    length: Counter = Counter()
    for node in g.preorder_node_iter():
        if node is not g.seed_node:
            length[norm(leafset(node))] += node.edge.length
    sub = parse(species, ns, "force-rooted").extract_tree_with_taxa_labels(labels=R)
    top = sub.seed_node
    while len(top.child_nodes()) == 1:
        top = top.child_nodes()[0]
    clades = [leafset(c) for c in top.child_nodes()]
    if len(clades) == 2:
        usable, reason = [c for c in clades[:1] if norm(c) in length], "split_absent"
    else:
        usable = sorted((c for c in clades if norm(c) in length), key=lambda c: (len(c), min(c)))
        reason = "polytomy_no_split"
    if not usable:
        return reason, None, len(clades)
    return usable[0], length[norm(usable[0])], len(clades)


def check_species(st: "RT.SpeciesTree", species: str, gene: str, ns, seen: Counter) -> str | None:
    want, ell, n_clades = reference_species_root(species, gene, ns)
    try:
        ours, kind = st.root(gene)
    except RT.RootingError as exc:
        seen[exc.reason] += 1
        return None if exc.reason == want else f"ours rejected ({exc.reason}), the reference roots on {want}"
    if isinstance(want, str):
        return f"ours rooted, the reference rejects ({want})"
    R = leafset(parse(gene, ns, "force-unrooted").seed_node)
    if root_clades(ours, ns) != {want, R - want}:
        return f"ours roots on {root_clades(ours, ns)}, the reference on {want}"
    expect_kind = "taxon" if min(len(want), len(R - want)) == 1 else "clade"
    if kind != expect_kind:
        return f"kind {kind}, expected {expect_kind}"
    seen[f"rooted_{kind}"] += 1
    if n_clades > 2:
        seen["species_polytomy_rooted"] += 1
    halves = [c.edge.length for c in parse(ours, ns, "force-rooted").seed_node.child_nodes()]
    if abs(halves[0] - halves[1]) > TOL * max(1.0, ell) or abs(sum(halves) - ell) > TOL * max(1.0, ell):
        return f"root edges {halves} are not halves of the split's edge {ell!r}"
    return unrooted_invariants(gene, ours, ns)


def refusals() -> list[str]:
    """Each input must be refused the way the docstring says; returns the ones that were not."""
    bad = []

    def expect(exc_type, fn, what, reason=None):
        try:
            fn()
        except exc_type as exc:
            if reason is not None and getattr(exc, "reason", None) != reason:
                bad.append(f"{what}: raised {exc!r}, expected reason {reason}")
            return
        except Exception as exc:                                        # noqa: BLE001
            bad.append(f"{what}: raised {exc!r}, expected {exc_type.__name__}")
            return
        bad.append(f"{what}: accepted")

    expect(ValueError, lambda: RT.SpeciesTree("(A:1,B:1,C:1);"), "trifurcating species root")
    expect(ValueError, lambda: RT.SpeciesTree("((A:1,B:1):1,A:1);"), "species tree with a duplicate")
    st = RT.SpeciesTree("((A:1,B:1):1,(C:1,D:1):1);")
    expect(ValueError, lambda: st.root("(A:1,B:1,X:1);"), "gene taxon not in the species tree")
    expect(RT.RootingError, lambda: RT.midpoint_root("(A:0,B:0,C:0);"), "zero-length tree",
           "zero_length")
    expect(RT.RootingError, lambda: RT.midpoint_root("(A:1,B:-2,C:1);"), "negative length",
           "negative_length")
    try:
        st.root("(A:1,B:1,(C:1,D:1):1);")
    except Exception as exc:                                            # noqa: BLE001
        bad.append(f"a concordant gene tree was refused: {exc!r}")
    with tempfile.NamedTemporaryFile("w", suffix=".tre", delete=False) as fh:
        fh.write("((A:1,B:1):1,C:1);\n((A:1,C:1):1,B:1);\n")
    expect(ValueError, lambda: RT.SpeciesTree.read(fh.name), "a species-tree file with two trees")
    Path(fh.name).unlink()
    # Deeper than Python's recursion limit: nothing here may recurse.
    n = 3000
    cat = "t0:1"
    for i in range(1, n):
        cat = f"({cat},t{i}:1):1"
    cat += ";"
    try:
        RT.midpoint_root(cat)
        _, kind = RT.SpeciesTree(cat).root(cat)
        if kind != "taxon":
            bad.append(f"caterpillar: kind {kind}, expected taxon")
    except RecursionError:
        bad.append(f"a {n}-leaf caterpillar hit the recursion limit")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen: Counter = Counter()
    failures = []
    for case in range(args.cases):
        n = rng.randint(4, 14) if case % 10 else rng.randint(15, 40)
        names = [f"t{i}" for i in range(n)]
        ns = dendropy.TaxonNamespace(names + ["zz"])
        if rng.random() < 0.2:
            # A trifurcating root is refused by design -- it is how an unrooted tree is written -- so
            # a root polytomy is built one level down, beside a taxon ``zz`` no gene tree carries.
            # Restricted to any R, that polytomy is the root.
            poly = random_tree(rng, names, polytomy=0.15, root_polytomy=True)
            species = f"({poly[:-1]}:0.5,zz:1.0);"
        else:
            species = random_tree(rng, names, polytomy=0.15)
        st = RT.SpeciesTree(species)
        R = rng.sample(names, rng.randint(3, n))
        if rng.random() < 0.5:
            concordant = parse(species, ns, "force-rooted").extract_tree_with_taxa_labels(labels=R)
            gene = as_written(rng, write(concordant), ns, seen)
        else:
            gene = as_written(rng, random_tree(rng, R, polytomy=0.1), ns, seen)
        for what, fn in (("midpoint", lambda: check_midpoint(gene, ns)),
                         ("species", lambda: check_species(st, species, gene, ns, seen))):
            try:
                why = fn()
            except Exception as exc:                                    # noqa: BLE001
                why = f"raised {exc!r}"
            if why:
                failures.append(f"case {case} {what}: {why}\n  species {species}\n  gene    {gene}")
        seen["cases"] += 1
    need = ("rooted_taxon", "rooted_clade", "split_absent", "species_polytomy_rooted",
            "polytomy_no_split", "gene_written_bifurcating", "gene_polytomy")
    missing = [x for x in need if not seen[x]]
    bad = refusals()
    print(f"[check_rooting] {seen['cases']} cases: " + ", ".join(f"{k} {seen[k]}" for k in need))
    for f in failures[:10]:
        print("FAIL", f)
    for b in bad:
        print("FAIL refusal:", b)
    if missing:
        print(f"FAIL vacuous: the fixtures never produced {missing}")
    ok = not (failures or bad or missing)
    print(f"[check_rooting] {'OK' if ok else f'{len(failures)} disagreements, {len(bad)} refusal failures'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
