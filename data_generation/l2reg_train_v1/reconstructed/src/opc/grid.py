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
}

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

    @property
    def key(self) -> str:
        s = f"_{self.stratum}" if self.stratum else ""
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
        return chain_seed(
            self.master_seed, self.n, self.rep, self.r, self.cu, self.gamma, self.stratum
        ) if self.stratum else chain_seed(
            self.master_seed, self.n, self.rep, self.r, self.cu, self.gamma
        )

    def as_dict(self) -> dict:
        d = {
            "n": self.n, "rep": self.rep, "r": self.r, "cu": self.cu,
            "gamma": self.gamma, "m_max": self.m_max,
        }
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
) -> list[Cell]:
    """Every replicate-run in the grid, deduplicated per A1/A28.

    ``r_override`` replaces :func:`r_values` for the given ``n``.  ``base_tags`` marks which of
    those need a freshly drawn base topology rather than the published one.

    ``strata`` crosses the cycle-size axis in.  It is crossed with ``r=0`` as well, even though a
    tree has no cycles: the ``r=0`` cell is the Than-Nakhleh null *for its stratum's topology*, and
    dropping it to a single shared base would break the blocked design (A2) that makes the ``k``
    contrast paired.  The duplication is two cells per (rep, cu), not two grids.
    """
    replicates = replicates or REPLICATES
    r_override = r_override or {}
    base_tags = base_tags or {}
    cps = tuple(sorted(checkpoints))
    cells = []
    for n in n_values:
        for rep in range(replicates[n]):
            for stratum in strata:
                for r in r_override.get(n, r_values(n)):
                    # gamma has no meaning without a reticulation: r=0 carries one, not three.
                    gammas = (gamma_values[0],) if r == 0 else gamma_values
                    for cu in cu_values:
                        for g in gammas:
                            cells.append(
                                Cell(n, rep, r, cu, g, cps[-1], cps, master_seed,
                                     base_tags.get(n, ""), stratum, vertex_disjoint)
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
