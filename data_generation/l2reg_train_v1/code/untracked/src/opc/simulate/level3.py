"""Regular, exactly-level-3 phylogenetic networks: placement and the gates (A64).

The level-3 benchmark (``python -m opc.l3reg``) needs networks that are

* **regular** -- :func:`level2.regularity`, the definition, unchanged;
* **exactly level-3** -- every blob holds at most three reticulations, and at least one holds
  three;
* of **girth >= 5** on the regular form ``R``.

This is :mod:`level2`'s generator taken one reticulation further.  It imports that module and
changes nothing it produces: level-1 blobs are placed by ``level2._blob`` itself, and a level-3
blob's first reticulation by ``level2._first``.  Everything :mod:`level2`'s docstring says about
the two forms ``N`` and ``R``, the regularity gate, shortcuts and the merge rule holds here
unchanged.

Joining a blob twice
--------------------
``level2._second`` cannot be reused, because it asserts that the blob it joins ends with two
reticulations.  :func:`_join` is its proposal, with three differences, each measured (A64):

* It asserts ``n_before + 1``.  The merge rule holds at any level: a tree path can touch a blob
  only through an edge, so touching exactly one blob adds exactly one reticulation to it.
* When nothing is alive outside the blob root's clade, it takes both edges inside rather than
  giving up.  Once a blob's root is at or near the network's root, 45% of third-join proposals
  otherwise found no pair.
* It rejects ladders only on the final join.

Rho and the clade are recomputed on every call, because the second reticulation can lift the
blob's root.

**Nothing is pruned after the intermediate join.**  A ladder or a shortcut there looks permanent
but is not.  :func:`level2.shortcuts` tests the *current* ``tree_parent[h]``, and a third donor
placed on the edge ``tree_parent[h] -> h`` becomes the new tree parent, which reaches
``donor_of[h]`` only through the new horizontal edge.  ``tests/test_level3.py`` pins a regular
level-3 network made that way from an intermediate that had a shortcut.  Regular does imply
shortcut-free, so the finished blob is still gated on it, by :func:`level2.regularity`.

Shapes
------
A level-3 blob is one of 65 generators (:mod:`generators`).  The design stratifies on the number
of sinks, the hybrids whose child edge leaves the blob.  :func:`_complete3` checks the sink count
first because it is the cheapest gate, then girth on ``R``, then regularity.
"""

from __future__ import annotations

import time as _time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from opc.grid import chain_seed
from opc.simulate import generators as G
from opc.simulate import level2 as L2
from opc.simulate import networks as N

__all__ = [
    "SINK_TARGETS",
    "MIN_GIRTH",
    "Layout3",
    "check3",
    "blob_report3",
    "build_level3",
]

SINK_TARGETS = ("s1", "s2", "s3")
MIN_GIRTH = L2.MIN_GIRTH


@dataclass(frozen=True)
class Layout3:
    """How many blobs of each kind, and each level-3 blob's sink target (``None`` = any)."""

    n_l3: int
    n_l1: int
    targets: tuple[str | None, ...] = ()

    def __post_init__(self):
        if self.n_l3 < 1:
            raise ValueError("an exactly-level-3 network needs at least one level-3 blob")
        if self.n_l1 < 0:
            raise ValueError(f"n_l1 = {self.n_l1}")
        if self.targets and len(self.targets) != self.n_l3:
            raise ValueError(f"{len(self.targets)} sink targets for {self.n_l3} level-3 blobs")
        bad = [s for s in self.targets if s is not None and s not in SINK_TARGETS]
        if bad:
            raise ValueError(f"unknown sink targets {bad}")

    @property
    def k(self) -> int:
        return 3 * self.n_l3 + self.n_l1

    def all_targets(self) -> tuple[str | None, ...]:
        return self.targets or (None,) * self.n_l3


def _sinks_of(target: str) -> int:
    return int(target[1:])


# ------------------------------------------------------------------------------ reports and gates


def blob_report3(net: N.Network) -> list[dict]:
    """:func:`level2.blob_report`'s rows, each with its blob's generator and sink count."""
    rows = L2.blob_report(net)
    tags = {ret.hybrid: f"H{k + 1}" for k, ret in enumerate(net.retics)}
    for row, b in zip(rows, L2.nontrivial_blocks(net), strict=True):
        if row["hybrids"] != [tags[h] for h in b.hybrids]:
            raise RuntimeError("blob_report's rows are not in nontrivial_blocks' order")
        if b.n_retics <= G.MAX_K:
            g = G.blob_generator(b)
            row |= {"generator": g.id, "n_sinks": g.n_sinks}
        else:  # described, never classified: :func:`check3` is what says no
            row |= {"generator": None, "n_sinks": G.n_sinks(b)}
    return rows


def check3(net: N.Network, layout: Layout3) -> dict:
    """Every gate, on a finished network.  ``ok`` is their conjunction."""
    blobs = L2.nontrivial_blocks(net)
    counts = Counter(b.n_retics for b in blobs)
    reg = L2.regularity(net)
    report = blob_report3(net) if not reg.ladders else []
    l3 = [row for row in report if row["n_retics"] == 3]
    sinks = sorted(row["n_sinks"] for row in l3)
    want = Counter(_sinks_of(s) for s in layout.all_targets() if s is not None)
    girth_min = min((row["girth_R"] for row in report), default=0)
    out = {
        "level": N.level(net),
        "n_blobs_l3": counts.get(3, 0),
        "n_blobs_l2": counts.get(2, 0),
        "n_blobs_l1": counts.get(1, 0),
        "max_blob_retics": max(counts, default=0),
        "regular": reg.ok,
        "regularity": {k: getattr(reg, k) for k in (
            "n_vertices", "n_clusters", "n_edges", "n_covers", "max_indegree",
            "non_cover_edges", "missing_covers")} | {
            "ladders": list(reg.ladders), "shortcuts": list(reg.shortcuts),
            "tree_child": reg.tree_child},
        "girth_R_min": girth_min,
        "sinks": sinks,
        "generators": sorted(row["generator"] for row in l3),
        "sinks_ok": bool(report) and not (want - Counter(sinks)),
    }
    out["ok"] = (out["level"] == 3 and out["max_blob_retics"] <= 3
                 and out["n_blobs_l3"] == layout.n_l3 and out["n_blobs_l2"] == 0
                 and out["n_blobs_l1"] == layout.n_l1 and reg.ok
                 and girth_min >= MIN_GIRTH and out["sinks_ok"])
    return out


# ------------------------------------------------------------------------------ the generator


@dataclass
class _Cfg3(L2._Cfg):  # noqa: SLF001
    join_tries: int = 300       # proposals per join
    max_twos: int = 8           # intermediates (two reticulations) tried per first reticulation


def _join(net: N.Network, h: int, n_before: int, rng, log, cfg: _Cfg3, *, final: bool):
    """Proposals that join the blob holding ``h`` and nothing else.  Yields trial networks.

    ``level2._second``'s proposal: half take both edges inside the blob root's clade and half one
    inside and one outside -- both inside when nothing outside is alive (module doc).
    """
    blocks = L2.nontrivial_blocks(net)
    target = L2.block_of(blocks, h)
    if target.n_retics != n_before:
        raise ValueError(f"the blob holds {target.n_retics} reticulations, not {n_before}")
    ti = blocks.index(target)
    emap = {e: i for i, b in enumerate(blocks) for e in b.edges}
    rho = L2.block_root(target)
    clade = L2._tree_clade(net, rho)  # noqa: SLF001
    t_lo, horizon = net.time[rho], max(net.time)
    tag = "L3/join2" if final else "L3/join1"
    for _ in range(cfg.join_tries):
        t = float(rng.uniform(t_lo, horizon))
        alive = N._alive_tree_edges(net, t)  # noqa: SLF001
        inside = [x for x in alive if x in clade]
        outside = [x for x in alive if x not in clade]
        both_in = rng.random() < 0.5
        if (both_in or not outside) and len(inside) >= 2:
            a, b = (inside[i] for i in rng.choice(len(inside), size=2, replace=False))
        elif inside and outside:
            a, b = inside[rng.integers(len(inside))], outside[rng.integers(len(outside))]
        else:
            log(f"{tag}:no_pair")
            continue
        o = L2._orient(net, a, b, rng)  # noqa: SLF001
        if o is None:
            log(f"{tag}:shortcut")
            continue
        c1, c2, z = o
        s1, s2 = L2._sides(net, c1, c2, z)  # noqa: SLF001
        if L2._touched(net, emap, s1 + s2) != {ti}:  # noqa: SLF001
            log(f"{tag}:wrong_blobs")
            continue
        trial = net.copy()
        _, h_new = L2.place(trial, c1, c2, t)
        if final and L2.ladders(trial):
            log(f"{tag}:ladder")
            continue
        got = L2.block_of(L2.nontrivial_blocks(trial), h_new)
        if h not in got.hybrids or got.n_retics != n_before + 1:
            raise RuntimeError(f"merge rule: predicted a {n_before + 1}-blob with {h}, "
                               f"got {got.hybrids}")
        yield trial


def _complete3(net: N.Network, h: int, target: str | None, log) -> bool:
    """The non-monotone gates, once a level-3 blob is finished: sinks, girth on ``R``,
    regularity -- cheapest first."""
    b = L2.block_of(L2.nontrivial_blocks(net), h)
    if target is not None and G.n_sinks(b) != _sinks_of(target):
        log("L3/complete:sinks")
        return False
    if L2.girth(L2.block_of(L2.nontrivial_blocks(L2.contract(net)), h)) < MIN_GIRTH:
        log("L3/complete:girth")
        return False
    if not L2.regularity(net).ok:
        log("L3/complete:irregular")
        return False
    return True


def _blob3(net: N.Network, target: str | None, rng, log, cfg: _Cfg3) -> N.Network | None:
    for _ in range(cfg.max_first):
        one = L2._first(net, rng, log, cfg, 0)  # noqa: SLF001
        if one is None:
            log("L3/no_first")
            return None
        h1 = one.retics[-1].hybrid
        for n_twos, two in enumerate(_join(one, h1, 1, rng, log, cfg, final=False), 1):
            for three in _join(two, h1, 2, rng, log, cfg, final=True):
                if _complete3(three, h1, target, log):
                    return three
            if n_twos >= cfg.max_twos:
                break
        log("L3/first_redrawn")
    return None


def build_level3(n_taxa: int, layout: Layout3, seed: int, *, cu: float,
                 gamma_range: tuple[float, float] = (0.25, 0.5),
                 cfg: _Cfg3 | None = None) -> tuple[N.Network, dict]:
    """A regular, exactly-level-3 network, scaled to median internal branch ``cu``.

    :func:`level2.build_level2`'s procedure: a base tree with no reticulations, blobs built in a
    per-network shuffled order, gammas from their own RNG stream in insertion order.  Returns
    ``(net, placement)``, with every rejection counter and redraw in ``placement``.
    """
    cfg = cfg or _Cfg3()
    lo, hi = gamma_range
    if not 0.0 < lo <= hi < 1.0:
        raise ValueError(f"gamma_range {gamma_range} must lie inside (0, 1)")
    log = L2._Log()  # noqa: SLF001
    t0 = _time.perf_counter()
    for draw in range(cfg.max_base):
        rng = np.random.default_rng([seed, draw, 1])
        net = N.build_base(n_taxa, 0, seed=chain_seed(seed, "base", draw))
        specs = [("L3", s) for s in layout.all_targets()] + [("L1", None)] * layout.n_l1
        order = [specs[i] for i in rng.permutation(len(specs))]
        for kind, target in order:
            if kind == "L3":
                net = _blob3(net, target, rng, log, cfg)
            else:
                net = L2._blob(net, "L1", None, rng, log, cfg)  # noqa: SLF001
            if net is None:
                break
        if net is None:
            log("base:redrawn")
            continue
        verdict = check3(net, layout)
        if not verdict["ok"]:  # every blob passed on completion; this is the belt and braces
            log("base:final_check_failed")
            continue
        grng = np.random.default_rng([seed, draw, 2])
        gammas = {ret.hybrid: float(grng.uniform(lo, hi)) for ret in net.retics}
        net = N.with_gammas(N.scaled(net, cu), gammas)
        return net, {
            "seed": seed, "base_draw": draw,
            "blob_order": [f"{k}:{s}" if s else k for k, s in order],
            "counters": dict(sorted(log.counters.items())),
            "seconds": _time.perf_counter() - t0,
            "cfg": vars(cfg),
        }
    raise RuntimeError(f"no {layout} network in {cfg.max_base} base trees: "
                       f"{dict(log.counters)}")
