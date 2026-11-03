"""§8 test 6: does PhyloCoalSimulations do what we think it does?

§4.2 as written only compares ``r=0`` against msprime, which exercises the plain MSC -- not the
reticulation handling, which is the only part of the dependency the study actually leans on.
Three tests, in increasing strength:

6a  ``r=0`` vs **msprime**, an independent implementation.  Guarded by an analytic anchor first:
    on a two-taxon tree with divergence 1 coalescent unit, P(coalescence before divergence) is
    ``1 - e^-1 = 0.6321``.  msprime's rate is ``1/(ploidy * N)`` per generation, so this single
    number catches the factor-of-2 ploidy error that is the classic bug in this conversion.
6b  ``r=1`` vs an **exact analytic oracle**.  With one lineage entering the hybrid, the MSNC is
    exactly ``gamma * MSC(donor tree) + (1-gamma) * MSC(tree-parent tree)``, and each term is
    simulated independently in msprime.  Not another simulator of the same model -- a closed-form
    decomposition.
6c  ``r=2``, general (multiple lineages entering hybrids), vs the Python MSNC oracle.

Gates are **effect sizes**, not p-values: at 1e5 loci a chi-square rejects on a 0.3% difference,
which says nothing about whether the two agree for our purposes.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from opc.clusters import extract
from opc.simulate import msnc_oracle as oracle
from opc.simulate import networks as N

REPO = Path(__file__).resolve().parents[1]
JULIA = Path("/shared/ns58/software/julia-1.12.6/bin/julia")
DEPOT = REPO / ".julia"

MAX_ABS_DIFF = 0.005  # largest per-topology frequency difference we accept
MAX_TV = 0.01  # total variation distance

requires_julia = pytest.mark.skipif(
    not JULIA.exists() or not (REPO / "julia" / "Manifest.toml").exists(),
    reason="Julia toolchain not installed",
)


# ------------------------------------------------------------------------------ msprime oracle


def _msprime_topologies(net: N.Network, n_loci: int, seed: int) -> Counter:
    """Gene-tree topology frequencies under the MSC on a species *tree*, via msprime.

    ``ploidy=1`` with ``initial_size=1.0`` makes one generation exactly one coalescent unit, so
    the network's branch lengths transfer with no conversion.  Only leaf populations are sampled:
    sampling an internal population raises "Attempt to sample a lineage from an inactive
    population".
    """
    import msprime

    assert net.n_retics == 0, "msprime oracle handles trees only"
    horizon = max(net.time)
    dem = msprime.Demography()
    for v in range(net.n_nodes):
        if net.tree_parent[v] == -2:
            continue
        dem.add_population(name=f"p{v}", initial_size=1.0)
    for v in range(net.n_nodes):
        kids = net.tree_children[v]
        if not kids:
            continue
        dem.add_population_split(
            time=horizon - net.time[v], derived=[f"p{c}" for c in kids], ancestral=f"p{v}"
        )
    dem.sort_events()  # population splits are emitted in node order, not time order
    samples = {f"p{v}": 1 for v in range(net.n_nodes) if net.is_leaf(v)}
    label_of = {f"p{v}": net.labels[net.leaf_taxon[v]] for v in range(net.n_nodes)
                if net.is_leaf(v)}

    counts: Counter = Counter()
    reps = msprime.sim_ancestry(
        samples=samples, demography=dem, ploidy=1, num_replicates=n_loci,
        random_seed=seed, sequence_length=1, recombination_rate=0,
    )
    for ts in reps:
        tree = ts.first()
        names = {u: label_of[dem.populations[ts.node(u).population].name] for u in ts.samples()}
        counts[_canonical_from_tskit(tree, names)] += 1
    return counts


def _canonical_from_tskit(tree, names) -> str:
    def rec(u):
        kids = tree.children(u)
        if not kids:
            return names[u]
        parts = sorted(rec(c) for c in kids)
        return "(" + ",".join(parts) + ")"

    return rec(tree.root)


def _stats(a: Counter, b: Counter) -> tuple[float, float]:
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    pa = {k: a.get(k, 0) / na for k in keys}
    pb = {k: b.get(k, 0) / nb for k in keys}
    return (
        max(abs(pa[k] - pb[k]) for k in keys),
        0.5 * sum(abs(pa[k] - pb[k]) for k in keys),
    )


def _compare(a: Counter, b: Counter, label: str, null: Counter):
    """Compare two topology distributions against a Monte Carlo noise floor.

    A fixed total-variation gate is meaningless here: TV is summed over every topology, so its
    sampling floor grows with the number of categories.  At n=6 there are 945 rooted topologies
    and two independent draws of the *same* distribution differ by TV ~ 0.06 -- far above any
    absolute threshold worth setting.  ``null`` is a second independent draw from the reference
    simulator, so it measures that floor directly, and the cross-comparison is required to look
    like noise rather than to hit a number chosen in advance.
    """
    md, tv = _stats(a, b)
    md0, tv0 = _stats(a, null)
    _record(label, md, tv, md0, tv0, n_topologies=len(set(a) | set(b)), n_loci=sum(a.values()))
    assert md <= max(1.5 * md0, MAX_ABS_DIFF), (
        f"{label}: max per-topology diff {md:.4f} vs Monte Carlo floor {md0:.4f}"
    )
    assert tv <= max(1.5 * tv0, MAX_TV), (
        f"{label}: total variation {tv:.4f} vs Monte Carlo floor {tv0:.4f}"
    )
    return md, tv, md0, tv0


#: Where the measured calibration evidence is persisted.  Printing it to stdout is not enough:
#: this is the evidence that the PhyloCoalSimulations dependency does what the study assumes, so
#: it has to survive the terminal.  ``scripts/validation_report.py`` renders it.
_RECORD_PATH = REPO / "results" / "validation" / "calibration.json"


def _record(label, md, tv, md0, tv0, **extra) -> None:
    import json
    from datetime import datetime, timezone

    _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(_RECORD_PATH.read_text())
    except (OSError, ValueError):
        data = {}
    data[label] = {
        "max_abs_diff": md, "max_abs_diff_floor": md0,
        "total_variation": tv, "total_variation_floor": tv0,
        "passes": bool(md <= max(1.5 * md0, MAX_ABS_DIFF) and tv <= max(1.5 * tv0, MAX_TV)),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    _RECORD_PATH.write_text(json.dumps(data, indent=1, sort_keys=True))


# ------------------------------------------------------------------------------ Julia backend


def _julia_topologies(net: N.Network, n_loci: int, seed: int, cu: float,
                      level: int | None = None) -> Counter:
    with tempfile.TemporaryDirectory(dir=REPO / "data") as td:
        td = Path(td)
        enw, out = td / "net.enewick", td / "gt.nwk.gz"
        enw.write_text(N.to_enewick(net) + "\n")
        manifest = td / "m.json"
        # ``level`` is omitted unless given, so the manifests 6a-6c write are unchanged.
        extra = "" if level is None else ', "level": %d' % level
        manifest.write_text(
            '[{"enewick": "%s", "out": "%s", "n_loci": %d, "seed": %d, '
            '"n_taxa": %d, "r": %d, "cu": %s%s}]'
            % (enw, out, n_loci, seed, net.n_taxa, net.n_retics, cu, extra)
        )
        env = dict(os.environ, JULIA_DEPOT_PATH=str(DEPOT), JULIA_NUM_THREADS="1")
        res = subprocess.run(
            [str(JULIA), "--project=" + str(REPO / "julia"),
             str(REPO / "src/opc/simulate/gene_trees.jl"), str(manifest)],
            capture_output=True, text=True, env=env, cwd=REPO,
        )
        assert res.returncode == 0, f"julia failed:\n{res.stdout}\n{res.stderr}"

        import gzip

        idx = {lab: i for i, lab in enumerate(net.labels)}
        counts: Counter = Counter()
        with gzip.open(out, "rt") as fh:
            for line in fh:
                if line.strip():
                    masks = extract.clusters_from_newick(line.strip(), idx, [])
                    counts[frozenset(masks)] += 1
        return counts


def _oracle_topologies_as_clustersets(net: N.Network, n_loci: int, seed: int) -> Counter:
    """Same key space as :func:`_julia_topologies`: a rooted topology is its cluster set."""
    import random

    rng = random.Random(seed)
    order = oracle.postorder_nodes(net)
    counts: Counter = Counter()
    for _ in range(n_loci):
        _, formed = oracle._simulate_one(net, order, rng)  # noqa: SLF001
        counts[frozenset(formed)] += 1
    return counts


# ---------------------------------------------------------------------------------- the tests


def test_msprime_coalescent_unit_anchor():
    """The single number that catches a ploidy factor-of-2 in the unit conversion.

    Two lineages sitting in *different* populations cannot coalesce, so the quantity to anchor on
    is coalescence within an ancestral branch: on ``((A,B):x, C)`` with the AB branch ``x=1``
    coalescent unit long, ``P(A and B coalesce inside AB) = 1 - e^-1 = 0.6321``.  With
    ``ploidy=1`` and ``initial_size=1.0`` one generation is one coalescent unit; ``ploidy=2``
    would halve the rate and give 0.3935.
    """
    import msprime

    dem = msprime.Demography()
    for name in ("A", "B", "C", "AB", "root"):
        dem.add_population(name=name, initial_size=1.0)
    dem.add_population_split(time=1.0, derived=["A", "B"], ancestral="AB")
    dem.add_population_split(time=2.0, derived=["AB", "C"], ancestral="root")

    n = 20_000
    coalesced = 0
    for ts in msprime.sim_ancestry(
        samples={"A": 1, "B": 1, "C": 1}, demography=dem, ploidy=1, num_replicates=n,
        random_seed=7, sequence_length=1, recombination_rate=0,
    ):
        tree = ts.first()
        a, b = (u for u in ts.samples() if dem.populations[ts.node(u).population].name in "AB")
        coalesced += tree.tmrca(a, b) < 2.0
    observed = coalesced / n
    assert observed == pytest.approx(1 - math.exp(-1), abs=0.01), (
        f"P(A,B coalesce inside the AB branch)={observed:.4f}, "
        f"expected {1 - math.exp(-1):.4f}; ~0.3935 means ploidy=2 halved the coalescent rate"
    )


def test_oracle_matches_msprime_on_trees():
    """The Python oracle's MSC path, against msprime.  Underpins 6b and 6c."""
    tree = N.scaled(N.build_base(5, 0, seed=13), 0.5)
    got = Counter(oracle.simulate_topology_counts(tree, 60_000, seed=3))
    want = _msprime_topologies(tree, 60_000, seed=99)
    null = _msprime_topologies(tree, 60_000, seed=100)
    md, tv, md0, tv0 = _compare(got, want, "oracle vs msprime (r=0)", null)
    print(f"\n  oracle vs msprime, n=5 r=0: max diff {md:.4f} (floor {md0:.4f}), "
          f"TV {tv:.4f} (floor {tv0:.4f})")


@requires_julia
def test_6a_julia_matches_msprime_r0():
    """PhyloCoalSimulations' MSC path against msprime."""
    tree = N.scaled(N.build_base(5, 0, seed=13), 0.5)
    jl = _julia_topologies(tree, 60_000, seed=21, cu=0.5)
    idx = {lab: i for i, lab in enumerate(tree.labels)}

    def keyed(seed):
        ms = _msprime_topologies(tree, 60_000, seed=seed)
        return Counter(
            {frozenset(extract.clusters_from_newick(k + ";", idx, [])): v for k, v in ms.items()}
        )

    md, tv, md0, tv0 = _compare(jl, keyed(99), "julia vs msprime (r=0)", keyed(100))
    print(f"\n  6a julia vs msprime, n=5 r=0: max diff {md:.4f} (floor {md0:.4f}), "
          f"TV {tv:.4f} (floor {tv0:.4f})")


@requires_julia
@pytest.mark.parametrize("gamma", [0.3, 0.5])
def test_6b_julia_matches_analytic_gamma_mixture(gamma):
    """The important one: reticulation handling against a closed-form decomposition.

    A network whose hybrid has exactly one entering lineage has MSNC distribution
    ``gamma * MSC(T_donor) + (1-gamma) * MSC(T_treeparent)`` exactly.  Both parental trees are
    simulated in msprime, so nothing in the comparison shares code with PhyloCoalSimulations.
    """
    net = None
    for seed in range(60):
        cand = N.with_gamma(N.scaled(N.build_base(5, 1, seed=2000 + seed), 0.6), gamma)
        below = N.hardwired_clusters(cand)
        if len(below[cand.retics[0].hybrid]) == 1:  # single lineage enters the hybrid
            net = cand
            break
    assert net is not None, "no single-lineage-hybrid network found"

    n_loci = 80_000
    jl = _julia_topologies(net, n_loci, seed=31, cu=0.6)
    idx = {lab: i for i, lab in enumerate(net.labels)}

    def mixture(base_seed):
        mix: Counter = Counter()
        for take_donor, weight in ((True, gamma), (False, 1.0 - gamma)):
            t = N.displayed_tree(net, take_donor)
            ms = _msprime_topologies(t, n_loci, seed=base_seed + int(take_donor))
            total = sum(ms.values())
            for k, v in ms.items():
                key = frozenset(extract.clusters_from_newick(k + ";", idx, []))
                mix[key] += weight * v / total * n_loci
        return mix

    md, tv, md0, tv0 = _compare(
        jl, mixture(41), f"julia vs analytic mixture (r=1, gamma={gamma})", mixture(141)
    )
    print(f"\n  6b julia vs analytic mixture, n=5 r=1 gamma={gamma}: "
          f"max diff {md:.4f} (floor {md0:.4f}), TV {tv:.4f} (floor {tv0:.4f})")


@requires_julia
def test_6c_julia_matches_python_oracle_general_network():
    """General case: multiple lineages entering hybrids, r=2, against the independent oracle."""
    net = N.with_gamma(N.scaled(N.build_base(6, 2, seed=77), 0.5), 0.3)
    n_loci = 80_000
    jl = _julia_topologies(net, n_loci, seed=51, cu=0.5)
    ora = _oracle_topologies_as_clustersets(net, n_loci, seed=61)
    null = _oracle_topologies_as_clustersets(net, n_loci, seed=62)
    md, tv, md0, tv0 = _compare(jl, ora, "julia vs python MSNC oracle (r=2)", null)
    print(f"\n  6c julia vs oracle, n=6 r=2: max diff {md:.4f} (floor {md0:.4f}), "
          f"TV {tv:.4f} (floor {tv0:.4f})")


def _marginal_diff(a: Counter, b: Counter) -> float:
    """Largest per-*cluster* frequency difference: sharp where per-topology ones are not."""
    na, nb = sum(a.values()), sum(b.values())
    ma: Counter = Counter()
    mb: Counter = Counter()
    for key, v in a.items():
        for c in key:
            ma[c] += v
    for key, v in b.items():
        for c in key:
            mb[c] += v
    return max(abs(ma[k] / na - mb[k] / nb) for k in set(ma) | set(mb))


@requires_julia
@pytest.mark.parametrize("shape", ["S0", "S1", "S2", "S3"])
def test_6d_julia_matches_oracle_on_level2_blobs(shape):
    """Level-2 blobs, per-reticulation gamma: what the level-2 benchmark simulates (A62).

    Two things differ from 6c on purpose.  The noise floor is two draws of the **reference**
    (``_compare`` measures it as Julia against a second oracle draw, which a systematic Julia
    bias would inflate along with the statistic it is meant to bound).  And the test is shown
    able to fail: the oracle with the two gammas swapped must land well outside that floor on the
    per-cluster marginals, which are the sharper statistic at n=7's 10,395 rooted topologies.
    """
    from opc.simulate import level2 as L2

    layout = L2.Layout(1, 0, (shape,))
    net, _ = L2.build_level2(7, layout, seed=300 + L2.SHAPES.index(shape), cu=0.5)
    assert L2.check(net, layout)["ok"]
    h1, h2 = (r.hybrid for r in net.retics)
    net = N.with_gammas(net, {h1: 0.15, h2: 0.45})

    n_loci = 80_000
    jl = _julia_topologies(net, n_loci, seed=71, cu=0.5, level=2)
    ora = _oracle_topologies_as_clustersets(net, n_loci, seed=81)
    null = _oracle_topologies_as_clustersets(net, n_loci, seed=82)
    md, tv = _stats(jl, ora)
    md0, tv0 = _stats(null, ora)
    _record(f"julia vs oracle (level-2 {shape}, gamma 0.15/0.45)", md, tv, md0, tv0,
            n_topologies=len(set(jl) | set(ora)), n_loci=n_loci)
    assert md <= max(1.5 * md0, MAX_ABS_DIFF), f"{shape}: per-topology {md:.4f} vs {md0:.4f}"
    assert tv <= max(1.5 * tv0, MAX_TV), f"{shape}: TV {tv:.4f} vs {tv0:.4f}"
    cm, cm0 = _marginal_diff(jl, ora), _marginal_diff(null, ora)
    assert cm <= max(1.5 * cm0, MAX_ABS_DIFF), f"{shape}: per-cluster {cm:.4f} vs {cm0:.4f}"

    swapped = _oracle_topologies_as_clustersets(
        N.with_gammas(net, {h1: 0.45, h2: 0.15}), n_loci, seed=83)
    cs = _marginal_diff(jl, swapped)
    assert cs > 3 * max(cm0, MAX_ABS_DIFF), (
        f"{shape}: swapping the gammas moved the marginals by only {cs:.4f} -- the comparison "
        f"cannot tell the two hybrids apart, so its passing says nothing")
    print(f"\n  6d {shape}: per-topology {md:.4f} (floor {md0:.4f}), per-cluster {cm:.4f} "
          f"(floor {cm0:.4f}), gammas swapped {cs:.4f}")


@requires_julia
@pytest.mark.parametrize("sinks", [1, 2, 3])
def test_6e_julia_matches_oracle_on_level3_blobs(sinks):
    """Level-3 blobs, one per sink count: what the level-3 benchmark simulates (A64).

    6d's design, one level up: a noise floor from two draws of the reference, and a gamma swap the
    per-cluster marginals must detect.  n = 7 is the smallest n at which each stratum's blob
    passes every gate, and it is 6d's n.
    """
    from opc.simulate import level3 as L3

    layout = L3.Layout3(1, 0, (f"s{sinks}",))
    net, _ = L3.build_level3(7, layout, seed=400 + sinks, cu=0.5)
    assert L3.check3(net, layout)["ok"]
    h1, h2, h3 = (r.hybrid for r in net.retics)
    net = N.with_gammas(net, {h1: 0.15, h2: 0.45, h3: 0.3})

    n_loci = 80_000
    jl = _julia_topologies(net, n_loci, seed=91, cu=0.5, level=3)
    ora = _oracle_topologies_as_clustersets(net, n_loci, seed=101)
    null = _oracle_topologies_as_clustersets(net, n_loci, seed=102)
    md, tv = _stats(jl, ora)
    md0, tv0 = _stats(null, ora)
    _record(f"julia vs oracle (level-3 s{sinks}, gamma 0.15/0.45/0.30)", md, tv, md0, tv0,
            n_topologies=len(set(jl) | set(ora)), n_loci=n_loci)
    assert md <= max(1.5 * md0, MAX_ABS_DIFF), f"s{sinks}: per-topology {md:.4f} vs {md0:.4f}"
    assert tv <= max(1.5 * tv0, MAX_TV), f"s{sinks}: TV {tv:.4f} vs {tv0:.4f}"
    cm, cm0 = _marginal_diff(jl, ora), _marginal_diff(null, ora)
    assert cm <= max(1.5 * cm0, MAX_ABS_DIFF), f"s{sinks}: per-cluster {cm:.4f} vs {cm0:.4f}"

    swapped = _oracle_topologies_as_clustersets(
        N.with_gammas(net, {h1: 0.45, h2: 0.15, h3: 0.3}), n_loci, seed=103)
    cs = _marginal_diff(jl, swapped)
    assert cs > 3 * max(cm0, MAX_ABS_DIFF), (
        f"s{sinks}: swapping two gammas moved the marginals by only {cs:.4f} -- the comparison "
        f"cannot tell the hybrids apart, so its passing says nothing")
    print(f"\n  6e s{sinks}: per-topology {md:.4f} (floor {md0:.4f}), per-cluster {cm:.4f} "
          f"(floor {cm0:.4f}), gammas swapped {cs:.4f}")


@requires_julia
def test_julia_gates_reject_a_corrupted_network(tmp_path):
    """The validation gates must actually fire.  Strip the gammas and the read must fail."""
    net = N.with_gamma(N.scaled(N.build_base(6, 1, seed=5), 0.5), 0.3)
    import re

    bad = re.sub(r"::[0-9.eE+-]+", "", N.to_enewick(net))
    enw = tmp_path / "bad.enewick"
    enw.write_text(bad + "\n")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        '[{"enewick": "%s", "out": "%s", "n_loci": 10, "seed": 1, '
        '"n_taxa": 6, "r": 1, "cu": 0.5}]' % (enw, tmp_path / "o.nwk.gz")
    )
    env = dict(os.environ, JULIA_DEPOT_PATH=str(DEPOT))
    res = subprocess.run(
        [str(JULIA), "--project=" + str(REPO / "julia"),
         str(REPO / "src/opc/simulate/gene_trees.jl"), str(manifest)],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    assert res.returncode != 0
    assert "gamma" in res.stdout.lower(), res.stdout


