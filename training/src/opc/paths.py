"""Where a published artifact lives under ``results/``.

``results/`` grew flat: by the end of the v0.3 follow-up it held sixty parquets from three
different studies with nothing but a filename prefix to tell them apart.  The prefix was in fact
always sufficient -- every driver stamps its experiment onto its own output -- so this module makes
that implicit convention explicit and puts one directory level behind it.

**The tier is derived, never passed.**  A driver asks for ``artifact(results, name)`` and gets
``results/<tier>/<name>``; nothing takes a tier argument, so a script cannot file its output under
the wrong study.  :func:`tier_of` is the single rule, and ``scripts/f_rebuild.py`` re-derives every
cache-backed parquet through the same call -- which is what makes the on-disk layout and the code
agree by construction rather than by twenty hand-maintained path constants.

Rules are matched **in order** and the first hit wins, because the tiers are not prefix-disjoint:
``FOLLOWUP_v0.3_search.md`` is the search tier's report and ``FOLLOWUP_v0.3.md`` is the formulation
tier's, and a plain longest-prefix match over an unordered map would put them in the same place.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TIERS", "RULES", "tier_of", "locate", "artifact", "glob"]

#: tier directory -> what it holds, for ``results/README.md`` and for error messages.
TIERS = {
    "study-n50": "exp-plan-n50-mpl-v0.1 -- E0/E0b/E1b/E2a/E3/E4 and the S50 report",
    "followup-v0.3": "followup-plan-v0.3 formulation tier -- F0/F1/F2/F5/F6",
    "search": "followup-plan-v0.3 search tier -- F3, cluster-system and network-space",
    "rl": "RL_PLAN.md -- the W0 re-baseline and census, the policy, and its evaluation",
}

#: (prefix, tier), matched in order.  Search before follow-up: see the module docstring.
RULES: tuple[tuple[str, str], ...] = (
    ("f3_", "search"),
    ("FOLLOWUP_v0.3_search", "search"),
    ("f0_", "followup-v0.3"),
    ("f1_", "followup-v0.3"),
    ("f2_", "followup-v0.3"),
    ("f5_", "followup-v0.3"),
    ("f6_", "followup-v0.3"),
    ("FOLLOWUP_v0.3", "followup-v0.3"),
    # **Before the bare ``clusters_`` rule, and the order is the whole point.**  RULES is matched
    # first-hit-wins, so the curriculum grids' collect parquets would otherwise land in the frozen
    # n = 50 study tier -- silently, because ``clusters_s20.parquet`` starts with ``clusters_``.
    # This is the module docstring's own example (``FOLLOWUP_v0.3_search`` before
    # ``FOLLOWUP_v0.3``) in a second place; ``tests/test_paths.py`` pins it.
    ("clusters_s20", "rl"),
    ("clusters_s30", "rl"),
    # One rule for all four benchmark grids, and one deliberately.  Per-phase rules would be the
    # same trap a level down: ``"clusters_b250".startswith("clusters_b25")`` is **True**, so
    # ``clusters_b25`` listed first would swallow ``clusters_b250``.  Both happen to name the same
    # tier today, which is exactly what would make it invisible.
    ("clusters_b", "rl"),
    # The extended training grids (``x25``/``x50``).  Same argument, and the same *silent* failure
    # if it is omitted: ``clusters_x25.parquet`` does not raise, it matches the bare rule below and
    # lands in the frozen n = 50 study tier.  One rule for both phases, for the reason above.
    ("clusters_x", "rl"),
    # The second extended training draw (``y25``/``y50``), and the n = 75 rung rides on the
    # ``clusters_x``/``clusters_b`` rules above.  Same argument a third time, and it had already
    # gone wrong once here: before this rule ``tier_of("clusters_y25.parquet")`` returned
    # ``study-n50`` -- it does not raise, it matches the bare rule below and files a live training
    # grid into the frozen study tier.  One rule for both phases, because
    # ``"clusters_y250".startswith("clusters_y25")`` is the same shadowing trap as ``clusters_b``.
    ("clusters_y", "rl"),
    ("clusters_", "study-n50"),
    ("e0_", "study-n50"),
    ("e0b_", "study-n50"),
    ("e1b_", "study-n50"),
    ("e2a_", "study-n50"),
    ("e3_", "study-n50"),
    ("e4_", "study-n50"),
    ("S50_", "study-n50"),
    ("INTERIM_", "study-n50"),
    # RL_PLAN.md.  ``w0_`` is the re-baseline and the scale census, ``rl_`` everything the
    # policy itself produces.  Appended rather than inserted: the tiers above are prefix-
    # disjoint from these, so order does not matter here -- but the rule that order *is*
    # significant does, so new prefixes go at the end where they cannot shadow an existing one.
    ("BENCH_V3", "rl"),
    # W10's report.  The parquets need no rule -- ``rl_gnn_*`` already matches ``rl_`` below --
    # but a hand-named report does, and an unmatched stem raises rather than landing at the top.
    ("GNN_PILOT", "rl"),
    # The camus evaluation's report.  Its *parquets* need no rule -- ``rl_gnn_eval_cam25...``
    # already matches ``rl_`` below -- but a hand-named report does, and an unmatched stem raises
    # rather than landing at the top level.  Appended, where a new prefix cannot shadow one above.
    ("CAMUS", "rl"),
    ("w0_", "rl"),
    ("rl_", "rl"),
    ("scale_census", "rl"),
)


def tier_of(name: str) -> str:
    """The tier directory ``name`` belongs to.

    Raises rather than guessing: an unrecognised stem means a new experiment was added without
    saying which study it belongs to, and silently dropping it at the top level is how ``results/``
    became flat the first time.
    """
    for prefix, tier in RULES:
        if name.startswith(prefix):
            return tier
    raise KeyError(
        f"no tier rule matches {name!r}; add one to opc.paths.RULES rather than writing "
        f"to the top level of results/"
    )


def locate(results: Path, name: str) -> Path:
    """``results/<tier>/<name>``, resolved but not created -- the reader's call."""
    return Path(results) / tier_of(name) / name


def artifact(results: Path, name: str) -> Path:
    """``results/<tier>/<name>``, with the tier directory created -- the writer's call.

    Kept distinct from :func:`locate` so that merely *checking whether* an artifact exists cannot
    leave an empty tier directory behind, which makes the layout look populated when it is not.
    """
    out = locate(results, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def glob(results: Path, pattern: str) -> list[Path]:
    """Every artifact matching ``pattern``, searched inside the tier ``pattern`` belongs to.

    A driver that enumerates its own variants -- ``e4_small*.parquet`` picks up the solver- and
    tau-tagged re-runs beside the unsuffixed one -- cannot glob ``results/`` any more, because the
    artifacts are a directory deeper.  Globbing the wrong directory does not raise: it returns an
    empty list, and a report built from it renders a completed grid as "_not run_".  That is the
    failure ``make_s50_report._e4_artifacts`` was written to avoid, so the tier lookup happens here
    rather than at each call site.
    """
    return sorted((Path(results) / tier_of(pattern)).glob(pattern))
