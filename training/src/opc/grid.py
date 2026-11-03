"""Grid definition and deterministic seeding.

Grid arithmetic (SPEC-AMENDMENTS A1, extended by A28).  ``gamma`` is meaningless at ``r=0``, and so
is the cycle-size ``stratum``, so the naive full factorial over-counts.  For ``S50`` the plan's
"4 x 4 x 3 x 2 x 2 = 192 cells" deduplicates to **76 configs / 1,520 replicate-runs**.

Blocked design (A2).  One base network per ``(n, replicate, stratum)``; ``r``, ``cu`` and ``gamma``
are all derivations of it, so those three contrasts are within-topology and paired.

Seeding reuses the ``chain_seed`` convention from
``/shared/ns58/MPPhySMC/scripts/lib/seeds.py``: SHA-256 of a colon-joined key, because Python's
built-in ``hash`` is salted per process and would silently destroy reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = [
    "N_VALUES",
    "CU_VALUES",
    "GAMMA_VALUES",
    "M_CHECKPOINTS",
    "REPLICATES",
    "STRATA",
    "r_values",
    "Cell",
    "enumerate_cells",
    "base_networks",
    "chain_seed",
    "config_hash",
    "SMOKE",
    "S50",
    "S20",
    "S30",
    "B25",
    "B50",
    "B100",
    "B250",
    "X25",
    "X50",
    "GAMMA_RANGE",
    "PHASES",
]

N_VALUES = (10, 25, 50, 100, 200)
CU_VALUES = (0.25, 0.5, 1.0, 2.0)
GAMMA_VALUES = (0.5, 0.3, 0.1)
M_CHECKPOINTS = (100, 500, 1000, 5000)
REPLICATES = {10: 20, 25: 20, 50: 20, 100: 20, 200: 10}

#: Cycle-size strata (A28): the inclusive ``(lo, hi)`` band a reticulation cycle's tree-edge count
#: must land in.  ``""`` is "no constraint", used by every pre-S50 phase.
#:
#: **Calibrated, not guessed.**  ``scripts/probe_generator.py`` measures placement rates and writes
#: ``results/validation/generator_probe.json``; these two bands are what it found.  Under no
#: constraint at n=50 the cycle-size distribution is heavily right-skewed -- median 4, q90 = 8 --
#: so a "large" floor cannot be set wherever one likes: four *vertex-disjoint* cycles of size >= 9
#: place on **24%** of 50-tip trees, and >= 8 on 97%, against 100% at >= 7.  Running only the draws
#: that succeed would sample exactly the topologies that admit dense large cycles, which is a shape
#: confound on the very axis being stratified (the failure ``grid.FILL``'s docstring records at
#: n=25, r=10).  ``(3,4)`` and ``(7,None)`` are disjoint, well separated, and both place on 100% of
#: draws at every ``k`` in the grid.
STRATA: dict[str, tuple[int, int | None]] = {
    "": (3, None),
    "small": (3, 4),
    "large": (7, None),
    #: The benchmark band.  Not a third *stratum* in A28's sense -- the b-phases do not stratify on
    #: cycle size, they draw inside one band and **measure** what came out, which is why
    #: there is one band rather than two.  ``(6, 20)`` sits between ``small`` and ``large`` and is
    #: bounded above, which ``large`` is not: at n = 250 an unbounded band admits cycles spanning
    #: most of the tree, and ``k`` vertex-disjoint cycles have to fit in ~``2n-2`` edges.
    "mid": (6, 20),
}

#: The b-phases' inheritance-probability range.  Drawn per reticulation and frozen per base
#: topology (:func:`opc.simulate.networks.with_random_gammas`), rather than taken from
#: :data:`GAMMA_VALUES`'s three fixed levels.  ``0.5`` is the symmetric case every pre-registered
#: n = 50 threshold was stated at and ``0.25`` is well inside the range E0 found recoverable --
#: A31 measured the coverage loss as sitting entirely at ``gamma = 0.1``, the plan's own negative
#: control, which this range therefore excludes.
GAMMA_RANGE = (0.25, 0.5)

_MASK64 = (1 << 64) - 1


def chain_seed(*parts) -> int:
    """Deterministic 64-bit seed from a colon-joined key (SHA-256, stable across processes)."""
    payload = ":".join(str(p) for p in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MASK64


def config_hash(d: dict) -> str:
    """Short stable hash of a config dict, recorded in every output row."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def r_values(n: int) -> list[int]:
    """§5's reticulation counts for ``n``, deduplicated."""
    return sorted({0, 1, 2, -(-n // 20), -(-n // 10)})


@dataclass(frozen=True)
class Cell:
    n: int
    rep: int
    r: int
    cu: float
    gamma: float
    m_max: int
    checkpoints: tuple[int, ...]
    master_seed: int
    #: Suffix on the base-network filename.  Empty for the published grid.  A phase that needs
    #: base topologies drawn under different rules must tag them: on any rejected placement
    #: :func:`opc.simulate.networks.build_base` redraws the *tree*, so a differently-constrained
    #: draw is a different topology and must not overwrite an existing one.
    base_tag: str = ""
    #: Cycle-size stratum, a key of :data:`STRATA`.  Empty outside S50.
    stratum: str = ""
    #: Draw the base topology with vertex-disjoint cycles (a galled tree) rather than merely
    #: edge-disjoint ones (level-1).  S50 only.
    vertex_disjoint: bool = False
    #: ``(lo, hi)`` to draw **one gamma per reticulation** from, uniformly, frozen per base
    #: topology.  ``None`` keeps :attr:`gamma` as a single network-wide value, which is what every
    #: phase before the b-grids does -- so ``None`` must reproduce the old key and the old seed
    #: exactly, and ``tests/test_grid_bench.py`` pins that it does.
    gamma_range: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        s = f"_{self.stratum}" if self.stratum else ""
        if self.gamma_range is not None:
            lo, hi = self.gamma_range
            return f"n{self.n}_rep{self.rep}_r{self.r}_cu{self.cu}_gU{lo}-{hi}{s}"
        return f"n{self.n}_rep{self.rep}_r{self.r}_cu{self.cu}_g{self.gamma}{s}"

    @property
    def base_key(self) -> str:
        s = f"_{self.stratum}" if self.stratum else ""
        return f"n{self.n}_rep{self.rep}{self.base_tag}{s}"

    @property
    def cycle_size_range(self) -> tuple[int, int | None]:
        return STRATA[self.stratum]

    @property
    def seed(self) -> int:
        """The gene-tree seed.

        **The phase name is deliberately not in it**, so two phases that agree on every field agree
        on their gene trees -- reuse when the caches are shared, a silent duplicate when
        they are not.  The b-grids avoid the question with a distinct ``master_seed`` each; see
        their docstring.
        """
        parts = [self.master_seed, self.n, self.rep, self.r, self.cu]
        parts.append(f"U{self.gamma_range[0]}-{self.gamma_range[1]}"
                     if self.gamma_range is not None else self.gamma)
        if self.stratum:
            parts.append(self.stratum)
        return chain_seed(*parts)

    @property
    def gamma_seed(self) -> int:
        """The per-reticulation gamma draw's seed.

        Keyed on the **base topology** -- ``(n, rep, stratum)`` -- and on nothing else, so every
        cell derived from one base sees the same gammas on the same hybrids.  That is what keeps
        A2's blocked design paired: without it, the ``cu`` contrast would move gamma too.
        """
        return chain_seed(self.master_seed, "gamma", self.n, self.rep, self.stratum)

    def as_dict(self) -> dict:
        d = {
            "n": self.n, "rep": self.rep, "r": self.r, "cu": self.cu,
            "gamma": self.gamma, "m_max": self.m_max,
        }
        if self.gamma_range is not None:
            d["gamma_lo"], d["gamma_hi"] = self.gamma_range
        if self.stratum:
            d["stratum"] = self.stratum
        return d


def enumerate_cells(
    n_values=N_VALUES,
    cu_values=CU_VALUES,
    gamma_values=GAMMA_VALUES,
    checkpoints=M_CHECKPOINTS,
    replicates=None,
    master_seed: int = 20260805,
    r_override=None,
    base_tags=None,
    strata=("",),
    vertex_disjoint: bool = False,
    gamma_range=None,
    pairing: str = "cross",
) -> list[Cell]:
    """Every replicate-run in the grid, deduplicated per A1/A28.

    ``r_override`` replaces :func:`r_values` for the given ``n``.  ``base_tags`` marks which of
    those need a freshly drawn base topology rather than the published one.

    ``strata`` crosses the cycle-size axis in.  It is crossed with ``r=0`` as well, even though a
    tree has no cycles: the ``r=0`` cell is the Than-Nakhleh null *for its stratum's topology*, and
    dropping it to a single shared base would break the blocked design (A2) that makes the ``k``
    contrast paired.  The duplication is two cells per (rep, cu), not two grids.

    ``pairing`` selects between the two designs this module now supports.

    ``"cross"`` is the historical one and the default: every ``(r, cu, gamma)`` combination on
    every replicate, so one base topology carries many cells and ``r``/``cu``/``gamma`` are
    *within*-topology contrasts.  That is what makes the S-grids' comparisons paired, and also
    why they give only twenty distinct topologies from 1,600 cells.

    ``"rep"`` gives **one cell per replicate**, with ``(r, cu)`` assigned by a balanced rotation on
    the replicate index -- ``r = r_list[rep % len(r_list)]``, ``cu = cu_values[(rep // len(r_list))
    % len(cu_values)]``.  Every cell is then its own base topology, which is what a held-out split
    needs: :func:`opc.rl.train.split_by_rep` splits on the base topology precisely because cells
    sharing a ``rep`` are re-parameterisations of one tree, so a 70/10/20 split of a ``"cross"``
    grid holds out four *trees*, not twenty.  The price is that the paired within-topology
    contrasts are gone: ``r`` and ``cu`` become between-topology factors, each at
    ``reps / (len(r_list) * len(cu_values))`` replicates.  ``gamma_values`` is unused under
    ``"rep"`` -- those grids pass ``gamma_range`` instead.
    """
    if pairing not in ("cross", "rep"):
        raise ValueError(f"pairing must be 'cross' or 'rep', got {pairing!r}")
    replicates = replicates or REPLICATES
    r_override = r_override or {}
    base_tags = base_tags or {}
    cps = tuple(sorted(checkpoints))
    cells = []
    for n in n_values:
        r_list = list(r_override.get(n, r_values(n)))
        for rep in range(replicates[n]):
            for stratum in strata:
                if pairing == "rep":
                    r = r_list[rep % len(r_list)]
                    cu = cu_values[(rep // len(r_list)) % len(cu_values)]
                    cells.append(
                        Cell(n, rep, r, cu, gamma_values[0], cps[-1], cps, master_seed,
                             base_tags.get(n, ""), stratum, vertex_disjoint, gamma_range)
                    )
                    continue
                for r in r_list:
                    # gamma has no meaning without a reticulation: r=0 carries one, not three.
                    gammas = (gamma_values[0],) if r == 0 else gamma_values
                    for cu in cu_values:
                        for g in gammas:
                            cells.append(
                                Cell(n, rep, r, cu, g, cps[-1], cps, master_seed,
                                     base_tags.get(n, ""), stratum, vertex_disjoint, gamma_range)
                            )
    return cells


@dataclass(frozen=True)
class BaseSpec:
    """What :func:`opc.simulate.networks.build_base` needs for one base topology."""

    n: int
    rep: int
    r_max: int
    stratum: str = ""
    vertex_disjoint: bool = False

    @property
    def cycle_size_range(self) -> tuple[int, int | None]:
        return STRATA[self.stratum]


def base_networks(cells) -> dict[str, BaseSpec]:
    """``base_key -> BaseSpec``: the distinct topologies the cells derive from."""
    out: dict[str, BaseSpec] = {}
    for c in cells:
        prev = out.get(c.base_key)
        r_max = max(c.r, prev.r_max) if prev else c.r
        out[c.base_key] = BaseSpec(c.n, c.rep, r_max, c.stratum, c.vertex_disjoint)
    return out


#: Smoke grid -- end to end in under two minutes.  Kept unchanged (no stratum, no
#: vertex-disjointness) so it stays a wiring test of the pipeline rather than of S50's rules.
SMOKE = dict(
    n_values=(10, 25),
    cu_values=(0.5, 1.0),
    gamma_values=(0.3,),
    checkpoints=(50, 100),
    replicates={10: 2, 25: 2},
)

#: **S50** -- the setting of ``exp-plan-n50-mpl-v0.1`` §1.  n = 50 throughout; galled trees, not
#: merely level-1; the cycle-size axis stratified rather than averaged over.
#:
#: The plan writes the grid as ``4 x 4 x 3 x 2 x 2 = 192`` cells.  Two of those factors collapse
#: (A28): ``gamma`` is meaningless at ``k=0``, and ``|G| in {250, 1000}`` is a pair of nested
#: checkpoints on one simulation rather than two runs.  What is left is **80 configs per
#: replicate** -- ``2 strata x 4 cu`` at ``k=0`` plus ``2 strata x 3 k x 4 cu x 3 gamma`` -- and
#: **1,600 replicate-runs** over **40 base topologies** (20 replicates x 2 strata, each drawn at
#: ``r_max=4``).
S50 = dict(
    n_values=(50,),
    cu_values=(0.25, 0.5, 1.0, 2.0),
    gamma_values=(0.5, 0.3, 0.1),
    checkpoints=(250, 1000),
    replicates={50: 20},
    r_override={50: [0, 1, 2, 4]},
    base_tags={50: "_s50"},
    strata=("small", "large"),
    vertex_disjoint=True,
)

#: **The curriculum grids** -- ``RL_PLAN.md`` §4 trains at ``n in {20, 30}`` and fine-tunes at
#: n = 50, and G-R6 (transfer) is not measurable without them.  Both mirror :data:`S50`'s axes
#: exactly, so a curriculum instance is the *same distribution at a different n* rather than a
#: second design whose contrasts would not be comparable.
#:
#: **The strata need no recalibration and that is measured, not assumed** (A68):
#: ``scripts/probe_generator.py`` at n = 20 and n = 30, 200 draws per (stratum, k), found both
#: ``small = (3, 4)`` and ``large = (7, None)`` placing on **100%** of draws at every ``k`` the
#: grid uses, every drawn network galled and every reticulation cluster-visible.  The unconstrained
#: cycle-size distribution barely moves with ``n`` (median 4 at all three); only the upper tail
#: shortens.  The probes are ``results/validation/generator_probe_n{20,30}.json``.
#:
#: ``r_override`` restates what :func:`r_values` already returns -- ``[0, 1, 2]`` at n = 20 and
#: ``[0, 1, 2, 3]`` at n = 30 -- because those are the ``r_max`` values A68's placement gate was
#: measured at, and a silent change to :func:`r_values` must not silently change what was gated.
#:
#: ``base_tags`` is not optional even though no n = 20 or n = 30 topology exists yet: a base drawn
#: under a stratum + vertex-disjointness constraint is a *different topology* from one drawn
#: without (``build_base`` redraws the tree on a rejected placement), and A28 is the record of what
#: it costs to let two such draws share a filename.
#:
#: 1,120 replicate-runs at n = 20 (2 strata x [4 cu at k=0 + 2 k x 4 cu x 3 gamma] x 20 reps) and
#: 1,600 at n = 30 (2 strata x [4 + 3 x 4 x 3] x 20).
S20 = dict(
    n_values=(20,),
    cu_values=(0.25, 0.5, 1.0, 2.0),
    gamma_values=(0.5, 0.3, 0.1),
    checkpoints=(250, 1000),
    replicates={20: 20},
    r_override={20: [0, 1, 2]},
    base_tags={20: "_s20"},
    strata=("small", "large"),
    vertex_disjoint=True,
)

S30 = dict(
    n_values=(30,),
    cu_values=(0.25, 0.5, 1.0, 2.0),
    gamma_values=(0.5, 0.3, 0.1),
    checkpoints=(250, 1000),
    replicates={30: 20},
    r_override={30: [0, 1, 2, 3]},
    base_tags={30: "_s30"},
    strata=("small", "large"),
    vertex_disjoint=True,
)


#: **The benchmark grids** -- 100 independent networks at each of n = 25, 50, 100, 250, 1,000 true
#: gene trees each, split 70 train / 10 eval / 20 test on the replicate index.  They exist because
#: every held-out set the RL stage has is too small to carry a checkpoint-to-checkpoint claim
#: (A84): the n <= 50 ladders are ten cells with cluster F1 quantised at 1/98, and the only large
#: one, ``data/N100-2``, is another study's simulation whose truths are outside the model class.
#:
#: Four things differ from the S-grids, each for a measured reason.
#:
#: * **``pairing="rep"``** -- one cell per replicate, so 100 cells are 100 *topologies*.  Under
#:   ``"cross"`` a 70/10/20 split would hold out four trees.  The cost is that ``r`` and ``cu``
#:   stop being within-topology contrasts; they are balanced across replicates instead, ten
#:   replicates per ``(k, cu)``.
#: * **``cu`` is ``(0.25, 0.5)``** -- the high-ILS half only.
#: * **``gamma_range``** rather than ``gamma_values``: one inheritance probability per
#:   reticulation, drawn ``U(0.25, 0.5)`` and frozen per base topology.  ``Network`` could not
#:   state that until ``gammas`` was added, though ``likelihood.Parameters`` had been fitting a
#:   vector all along.
#: * **``strata=("mid",)``** -- one band ``(6, 20)``, not two strata.  Cycle size here is measured,
#:   not controlled.
#:
#: ``r_override`` is **probe-derived and not** :func:`r_values`.  Vertex-disjoint cycles of 6-20
#: edges have to fit in the base tree's ~``2n-2``, so ``k`` is bounded near ``(2n-2)/13``; measured
#: on the ``large`` band, placement already fails at k = 10 for n = 100 (2 of 10 draws) and at
#: k = 20 and 25 for n = 250 (0 of 3), and ``(6, 20)`` is the narrower band.  ``B_K`` below is what
#: ``scripts/probe_generator.py`` cleared at >= 95% placement; changing it means re-running that
#: gate, not editing the table.
#:
#: ``master_seed`` differs from the S-grids' deliberately.  Neither the cell seed nor the base seed
#: carries the phase name or ``base_tag`` (see :attr:`Cell.seed`), so a shared ``master_seed``
#: and a shared ``(n, rep, stratum)`` would silently reproduce S50's gene trees under a b-phase
#: name.
#: The ``"mid"`` stratum already separates them; the distinct seed means the separation does not
#: depend on that.
B_MASTER_SEED = 20260829

#: ``n -> the reticulation ladder``.  k = 0 is in every one of them: the pure-tree stratum is the
#: only place A84's "restraint" behaviour is observable, and no S-grid has a k = 0 instance.
#:
#: **Measured, not chosen.**  ``scripts/ops/probe_bench_generator.sh`` ran
#: ``scripts/probe_generator.py`` at 200 draws per (n, k) against the ``mid`` band, gating at
#: >= 95% placement with every draw galled and every reticulation cluster-visible
#: (``results/validation/generator_probe_b*.json``).  Everything here placed on **100%** of draws
#: except one rung, and that rung is why n = 25 has four:
#:
#: * **n = 25, k = 4 places on 30.5%** and is excluded.  k = 1, 2, 3 are all 100%.  It is a
#:   geometric limit rather than a slow draw: four *vertex-disjoint* cycles of >= 6 tree edges need
#:   >= 24 of a 25-tip tree's 48, and the band's own median placed cycle at this n is 6-7.  Taking
#:   the 30.5% that succeed would sample exactly the topologies that admit four dense disjoint
#:   cycles, which is A28's shape confound on the axis the grid varies.
#: * n = 100 places 8 and n = 250 places 16, both at 100%, so the ladder does reach a regime the
#:   trained policy has never seen ({1, 2, 4} at n = 50).
#:
#: Changing a rung means re-running that gate, not editing this table.
B_K: dict[int, list[int]] = {
    25: [0, 1, 2, 3],
    50: [0, 1, 2, 3, 5],
    # n = 75 was gated on 2026-09-08 at 200 draws a rung: k = 1, 2, 3 and **6** all place on
    # **100%**, every draw galled, every reticulation cluster-visible, A28 re-check 0/200
    # (``results/validation/generator_probe_b75.json``).  k = 6 is the point of the rung -- it is
    # the first training data past n = 50's ladder, and ``CAMUS_RESULTS.md`` puts the trained
    # support at k <= 3 with k >= 4 extrapolation.
    75: [0, 1, 2, 3, 6],
    100: [0, 1, 2, 4, 8],
    250: [0, 1, 2, 8, 16],
}


def _bench(n: int) -> dict:
    """One benchmark phase.  100 cells / 100 base topologies / 100,000 gene trees."""
    return dict(
        n_values=(n,),
        cu_values=(0.25, 0.5),
        gamma_values=(0.5,),          # unused under pairing="rep"; gamma_range is what applies
        gamma_range=GAMMA_RANGE,
        checkpoints=(250, 1000),
        replicates={n: 100},
        r_override={n: B_K[n]},
        base_tags={n: "_b"},
        strata=("mid",),
        vertex_disjoint=True,
        master_seed=B_MASTER_SEED,
        pairing="rep",
    )


B25, B50, B100, B250 = (_bench(n) for n in (25, 50, 100, 250))


#: **The extended training grids** -- 900 further draws from the *same* generative process as
#: ``b25``/``b50``, at n = 25 and n = 50 only.  A90 trained W10's classifier on 140 graphs (the
#: b25 + b50 train splits) and left exact recovery at 1 / 3 / 3 / 0 cells of 60; these exist so the
#: data axis can be ablated at 10x before anything is concluded about the objective.
#:
#: **They are training data and nothing else.**  Every reported number stays on the b-grids' 20-cell
#: test splits, so it is paired with A90 and gate G-N0 keeps its benchmark reference.  Raising
#: ``b25``'s own ``replicates`` would have been simpler and is wrong: :func:`opc.rl.train.split_by_rep`
#: puts ``test = rep >= 80`` and every eval driver takes ``test[:max_cells]`` from a
#: **lexicographically** sorted key list, in which ``"n25_rep100_..."`` sorts *before*
#: ``"n25_rep80_..."`` -- so a 900-replicate ``b25`` would silently change which twenty cells are
#: the test set for ``w10_gnn_eval.py``, ``w9_swap_eval.py`` and ``make_bench_report.py`` at once.
#:
#: ``X_MASTER_SEED`` is **the leakage fix, not hygiene**.  The base-topology seed is
#: ``chain_seed(master_seed, "base", n, rep, stratum)`` (``rungrid.build_base_network``) and carries
#: no ``base_tag``; :attr:`Cell.seed` and :attr:`Cell.gamma_seed` likewise.  Under a *shared* master
#: seed, ``x25`` reps 80-99 would reproduce ``b25``'s **test** cells on all three streams -- same
#: base topology, same gene trees, same gammas -- which is training on the test set.  Measured at
#: 20/20 identical before the seed was changed; ``tests/test_grid_bench.py`` pins the disjointness.
#:
#: **900, not 800.**  ``--split-train 700 --split-val 800`` then gives 700 train / 100 val and
#: **100 spare held-out cells per phase**.  Those 200 in-range cells are the powered read: the
#: b-grid test splits are twenty cells with cluster F1 quantised at 0.0435 (n = 25) and 0.0208
#: (n = 50), which A84 is the record of.  The headline stays the 20-cell paired number; the
#: 200-cell one goes beside it.
#:
#: Nothing else differs from :func:`_bench` -- same ``B_K`` ladder, same ``cu``, same
#: ``gamma_range``, same ``mid`` band, same ``vertex_disjoint``, same ``pairing="rep"`` -- so these
#: are more draws from one process rather than a second design.
X_MASTER_SEED = 20260901


def _bench_x(n: int) -> dict:
    """One extended training phase.  900 cells / 900 base topologies / 900,000 gene trees."""
    d = _bench(n)
    d.update(replicates={n: 900}, base_tags={n: "_x"}, master_seed=X_MASTER_SEED)
    return d


X25, X50 = _bench_x(25), _bench_x(50)


#: **The second extended draw at n = 25 and n = 50** -- 900 further cells each, from the same
#: generative process again.  A new *phase* rather than a larger ``replicates`` on ``x25``/``x50``
#: for the reason :func:`_bench_x`'s docstring gives one level up: raising a phase's replicate
#: count silently re-partitions every split that reads it, and every eval driver takes
#: ``test[:max_cells]`` from a **lexicographically** sorted key list.
#:
#: **A third master seed, and for exactly ``X_MASTER_SEED``'s reason.**  None of the three seed
#: streams -- base (``rungrid.build_base_network``), cell (:attr:`Cell.seed`) and gamma
#: (:attr:`Cell.gamma_seed`) -- carries ``base_tag`` or the phase name, so ``y25`` under
#: ``Y = X`` would reproduce ``x25`` cell for cell on all three at once, and under ``Y = B`` it
#: would reproduce ``b25``'s test split.  ``base_tags`` alone cannot separate them; only this
#: constant can.  ``tests/test_grid_bench.py`` pins the disjointness against **both** b and x,
#: with the counterfactual beside it.
Y_MASTER_SEED = 20260908


def _bench_y(n: int) -> dict:
    """A second 900-cell training draw, from the same process as :func:`_bench_x`."""
    d = _bench(n)
    d.update(replicates={n: 900}, base_tags={n: "_y"}, master_seed=Y_MASTER_SEED)
    return d


Y25, Y50 = _bench_y(25), _bench_y(50)


#: **The n = 75 rung** -- the taxon count the training mixture has never had, and the first that
#: reaches ``k = 6``.  ``x75`` is the training half and ``b75`` the held-out benchmark half, which
#: fills the n = 50 -> 100 gap in the benchmark curve.
#:
#: ``x75`` reuses ``X_MASTER_SEED`` and ``b75`` reuses ``B_MASTER_SEED``, which is safe where
#: ``y25`` would not have been: ``n`` is in all three seed streams and no landed phase has
#: ``n = 75``, so the only pair that overlaps is ``x75``/``b75`` on reps 0-99 -- separated by X vs
#: B exactly as ``x25``/``b25`` are.
#:
#: **``x75`` is declared at 900 replicates although only 400 cells get instances.**  That is what
#: makes the x-grids' ``--split-train 700 --split-val 800`` put reps 0-299 in train and 700-799 in
#: validation; the window is chosen at instance-build time with ``rl_instances.py
#: --min-rep/--reps``, not here.  Declaring it at 400 would have put every cell in train and left
#: the rung with no validation cells at all.
X75, B75 = _bench_x(75), _bench(75)


#: ``--phase`` name -> phase dict.  Every driver written from here on resolves through this rather
#: than through its own ``{"smoke": ..., "s50": ...}`` literal: adding a phase should be one edit,
#: and thirteen copies of a two-entry table is how ``s20`` becomes runnable by ``rungrid`` and not
#: by ``w0_baseline`` without anything failing.  The frozen study drivers keep their own
#: literals -- they are on a branch that will never see these phases.
PHASES = {"smoke": SMOKE, "s50": S50, "s20": S20, "s30": S30,
          "b25": B25, "b50": B50, "b100": B100, "b250": B250,
          "x25": X25, "x50": X50,
          "y25": Y25, "y50": Y50, "x75": X75, "b75": B75}
