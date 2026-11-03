"""Where one gene-tree **arm**'s live caches live under ``results/<phase>/``.

An arm is a choice of gene trees for a phase whose networks are fixed: ``""`` is the simulated
("true") trees the study has always run on, ``"est_bp1000"`` the IQ-TREE re-estimates of 1,000 bp
alignments grown along them.  The truth network is the same object in both; only the evidence about
it changes, and with it the support table, the candidate set, the overlap graph and every node
feature derived from them.

**One module knows the directory names, because they are not symmetric.**  The trees are
``genetrees_est_<backend>_bp<bp>/`` while the cells and cluster tables are
``cells_<backend>_est_bp<bp>/`` and ``clusters_<backend>_est_bp<bp>/`` -- ``est`` moves from before
the backend to after it.  So no single ``backend`` string addresses all three, which is exactly
what :func:`opc.rl.instances.build` assumed when it interpolated one into
``f"genetrees_{backend}"`` and ``f"clusters_{backend}"``.  Before this module the convention was
written down in two places that both had to be right and neither of which said so: those two
format strings, and
``rungrid.run_cell``'s ``arm.cell_dir.parent / f"clusters_{...split('cells_', 1)[-1]}"``.

**Not** :mod:`opc.paths`, which files *published artifacts* into a study tier and creates
directories on write.  These are live caches, rebuildable and untiered -- the distinction
``opc.rl.instances.cache_dir``'s docstring already draws.  And not :mod:`opc.simulate.seqsim`,
which owns AliSim and IQ-TREE: ``opc.rl`` has no business importing that to learn a path.

An unknown arm **raises** rather than inventing a directory, for the reason
``opc.paths.tier_of`` raises on an unmatched stem: a silently invented cache directory is empty,
and an empty cache reads as "nothing has been built yet".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ARM_TRUE", "ArmDirs", "KNOWN_ARMS", "arm_of_bp", "arm_slug", "arm_of_slug",
           "arm_dirs", "is_estimated"]

#: The simulated gene trees.  Empty, so every landed path is the one it has always been.
ARM_TRUE = ""

#: ``est_bp<digits>`` -- the only non-true arm this repo *produces*.  A pattern rather than a
#: list because ``bp`` is a free parameter of the hand-off (A99 measures 500 and 1000).
_EST = re.compile(r"^est_bp(\d+)$")

#: Arms that arrive with an **external dataset** rather than being produced here, and their
#: filename tokens.  Spelled out rather than folded into :data:`_EST`, because ``est_bp<n>`` means
#: "IQ-TREE on an AliSim alignment grown along *our* gene trees" and these are somebody else's
#: pipeline -- the same reason this module exists at all.  They own no ``rungrid`` caches, so
#: :func:`arm_dirs` still refuses them; only the *filename* token and the instance cache directory
#: are theirs, and both go through here.
_EXTRA_SLUGS: dict[str, str] = {
    "cam_trueog": "trueog",
    "cam_iq500": "iq500",
    "cam_g500": "g500",
}
_EXTRA_ARMS = {slug: arm for arm, slug in _EXTRA_SLUGS.items()}

#: Every arm a driver's ``--arm`` may name.  A list, so a typo is a parse error rather than an
#: empty cache directory that reads as "nothing has been built yet".
KNOWN_ARMS: tuple[str, ...] = (ARM_TRUE, "est_bp1000") + tuple(_EXTRA_SLUGS)


@dataclass(frozen=True)
class ArmDirs:
    """The three per-phase cache directories one arm owns.  None of them is created here."""

    trees: Path
    cells: Path
    clusters: Path


def is_estimated(arm: str) -> bool:
    return bool(arm)


def arm_of_bp(bp: int) -> str:
    """``1000 -> "est_bp1000"``.  The one place the arm id is spelled."""
    return f"est_bp{int(bp)}"


def arm_slug(arm: str) -> str:
    """The **underscore-free** token an artifact *filename* carries for ``arm``.

    ``make_gnn_report.ARM_RE`` splits a filename on ``_`` and takes ``(.+)`` greedily up to
    ``_s(\\d+)``, so an arm id with an underscore in it would silently re-partition the name into a
    different phase, arch and tag -- A92's hazard and A94's, which is why the report reads the arm
    from the filename in the first place.  ``est_bp1000`` -> ``est1000``; the true arm -> ``""``.
    """
    if not arm:
        return ""
    if arm in _EXTRA_SLUGS:
        return _EXTRA_SLUGS[arm]
    m = _EST.match(arm)
    if m is None:
        raise KeyError(f"unknown gene-tree arm {arm!r}; expected '', 'est_bp<n>' or one of "
                       f"{sorted(_EXTRA_SLUGS)}")
    return f"est{m.group(1)}"


def arm_of_slug(slug: str) -> str:
    """Inverse of :func:`arm_slug`.  ``"est1000" -> "est_bp1000"``, ``"" -> ""``.

    ``scripts/make_gnn_report.py`` reads the test arm back out of a filename, and doing that with
    a second regex written where it is used is how the two spellings drift.
    """
    if not slug:
        return ARM_TRUE
    if slug in _EXTRA_ARMS:
        return _EXTRA_ARMS[slug]
    m = re.fullmatch(r"est(\d+)", slug)
    if m is None:
        raise KeyError(f"unknown arm slug {slug!r}; expected '', 'est<n>' or one of "
                       f"{sorted(_EXTRA_ARMS)}")
    return f"est_bp{m.group(1)}"


def arm_dirs(root: Path, backend: str, arm: str = ARM_TRUE) -> ArmDirs:
    """The three cache directories for ``arm`` under ``root`` (= ``results/<phase>``).

    ``arm = ""`` returns exactly the three names the study has always used, so nothing that
    predates arms moves.
    """
    root = Path(root)
    if not arm:
        return ArmDirs(trees=root / f"genetrees_{backend}",
                       cells=root / f"cells_{backend}",
                       clusters=root / f"clusters_{backend}")
    m = _EST.match(arm)
    if m is None:
        raise KeyError(f"unknown gene-tree arm {arm!r}; expected '' or 'est_bp<n>'")
    bp = m.group(1)
    # The word order really does differ between the two; see the module docstring.
    return ArmDirs(trees=root / f"genetrees_est_{backend}_bp{bp}",
                   cells=root / f"cells_{backend}_est_bp{bp}",
                   clusters=root / f"clusters_{backend}_est_bp{bp}")
