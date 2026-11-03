"""The estimation arm as the driver actually runs it.

Two claims, both about the boundary between the arms: the estimated arm produces usable output,
and turning it on changes nothing whatsoever about the true arm -- the constraint that lets the
existing 15 GB gene-tree cache and every published ``config_hash`` stay valid.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from opc import paths
from opc.simulate import seqsim

REPO = Path(__file__).resolve().parents[1]
JULIA = Path("/shared/ns58/software/julia-1.12.6/bin/julia")
PY = REPO / ".venv" / "bin" / "python"

requires_toolchains = pytest.mark.skipif(
    not (JULIA.exists() and (REPO / "julia" / "Manifest.toml").exists()
         and seqsim.IQTREE.exists() and PY.exists()),
    reason="Julia or IQ-TREE toolchain not installed",
)


def run_grid(out: Path, *extra):
    cmd = [str(PY), "-m", "opc.rungrid", "--phase", "smoke",
           "--backend", "julia", "--jobs", "8", "--out", str(out), *extra]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=3600)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


@requires_toolchains
def test_the_estimation_arm_runs_end_to_end_and_the_true_arm_is_untouched(tmp_path):
    plain = tmp_path / "plain"
    both = tmp_path / "both"
    run_grid(plain)
    out = run_grid(both, "--estimate", "--bp", "200", "--est-m", "50")

    a = pd.read_parquet(paths.locate(plain, "clusters_smoke.parquet"))
    b = pd.read_parquet(paths.locate(both, "clusters_smoke.parquet"))
    assert list(a.columns) == list(b.columns), "--estimate leaked columns into the true arm"
    assert a.equals(b), "--estimate changed the true arm's rows"

    est = pd.read_parquet(paths.locate(both, "clusters_smoke_est.parquet"))
    assert len(est) > 0
    assert set(est["arm"]) == {"estimated"} and set(est["bp"]) == {200}
    # m checkpoints are nested prefixes, so capping at 50 keeps only the checkpoints <= 50.
    assert set(est["m"]) <= {50}
    assert "[estimate]" in out


@requires_toolchains
def test_the_estimated_arm_reports_real_but_not_total_estimation_error(tmp_path):
    out_dir = tmp_path / "run"
    run_grid(out_dir, "--estimate", "--est-only", "--bp", "200,500", "--est-m", "100")
    est = pd.read_parquet(paths.locate(out_dir, "clusters_smoke_est.parquet"))

    per_bp = est.groupby("bp")["est_rf_norm_mean"].mean()
    assert 0.0 < per_bp[200] < 0.9, f"200 bp RF {per_bp[200]} is degenerate"
    assert per_bp[500] < per_bp[200], "longer sequences must estimate better"

    root = est.groupby("bp")["est_frac_root_split_correct"].mean()
    assert 0.0 < root[200] <= 1.0 and root[500] > root[200]

    # Every row must record what made it, so the parquet is self-describing.
    assert set(est["inference_model"]) == {"HKY+F+G4"}
    assert set(est["sim_model"]) == {"HKY{2}+F{0.3,0.2,0.2,0.3}+G4{1}"}
    assert (est["est_iqtree_model_used"].str.startswith("HKY")).all(), \
        "IQ-TREE inferred under a model other than the one requested"
    assert (est["expected_subs_per_internal_branch"] > 0).all()
    assert est["est_hash"].nunique() == est.groupby(["n", "rep", "r", "cu", "gamma", "bp"]).ngroups


@requires_toolchains
def test_a_second_bp_arm_does_not_drop_the_first_from_the_collected_file(tmp_path):
    """Collect reads every arm on disk, not just the one this invocation ran.

    Otherwise running `--bp 200` after `--bp 500` silently replaces 500 bp's rows with 200 bp's
    -- the per-cell parquets survive, but the collected file quietly loses an arm.
    """
    out_dir = tmp_path / "run"
    run_grid(out_dir, "--estimate", "--est-only", "--bp", "500", "--est-m", "50")
    first = pd.read_parquet(paths.locate(out_dir, "clusters_smoke_est.parquet"))
    assert set(first["bp"]) == {500}

    run_grid(out_dir, "--estimate", "--est-only", "--bp", "200", "--est-m", "50")
    both = pd.read_parquet(paths.locate(out_dir, "clusters_smoke_est.parquet"))
    assert set(both["bp"]) == {200, 500}
    assert len(both) == 2 * len(first)


@requires_toolchains
def test_the_estimated_arm_is_cached_on_a_second_run(tmp_path):
    out_dir = tmp_path / "run"
    run_grid(out_dir, "--estimate", "--est-only", "--bp", "200", "--est-m", "50")
    first = pd.read_parquet(paths.locate(out_dir, "clusters_smoke_est.parquet"))
    run_grid(out_dir, "--estimate", "--est-only", "--bp", "200", "--est-m", "50")
    second = pd.read_parquet(paths.locate(out_dir, "clusters_smoke_est.parquet"))
    assert first.equals(second), "a resumed run did not reproduce the first one"


def test_estimate_requires_the_julia_backend():
    res = subprocess.run(
        [sys.executable, "-m", "opc.rungrid", "--phase", "smoke",
         "--estimate", "--dry-run"],
        cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert res.returncode != 0
    assert "requires --backend julia" in res.stderr
