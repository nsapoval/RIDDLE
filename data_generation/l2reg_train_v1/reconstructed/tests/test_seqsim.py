"""Pins the AliSim + IQ-TREE chain, including the two AliSim defects the design routes around.

Gated on the IQ-TREE binary the same way ``tests/test_calibration.py`` gates on Julia.  The
probe answers in ``results/validation/alisim_probe.json`` are re-asserted here rather than merely
recorded, so an IQ-TREE upgrade that changes any of them fails the suite instead of silently
changing the data.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from opc.clusters import extract
from opc.simulate import outgroup as og
from opc.simulate import phylip, seqsim

REPO = Path(__file__).resolve().parents[1]
CACHED = REPO / "results" / "full" / "genetrees_julia" / "n10_rep0_r0_cu0.5_g0.5.nwk.gz"

requires_iqtree = pytest.mark.skipif(not seqsim.IQTREE.exists(),
                                     reason="IQ-TREE not installed")
requires_cache = pytest.mark.skipif(not CACHED.exists(),
                                    reason="no cached gene trees")

THREE = [
    "((t1:0.001,t2:0.001):0.30,(t3:0.30,t4:0.30):0.30);",
    "((t1:0.001,t3:0.001):0.30,(t2:0.30,t4:0.30):0.30);",
    "((t1:0.001,t4:0.001):0.30,(t2:0.30,t3:0.30):0.30);",
]


def run(cmd, cwd):
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True, text=True)


def splits(newick: str, n: int) -> frozenset:
    index = {f"t{i}": i - 1 for i in range(1, n + 1)}
    full = (1 << n) - 1
    return frozenset(
        c for c in (min(m, full ^ m) for m in extract.clusters_from_newick(newick, index, []))
        if 2 <= c.bit_count() <= n - 2)


def make_cell(key="n10_rep0_r0_cu0.5_g0.5", n=10, cu=0.5, m_max=12):
    class C:
        pass

    C.key, C.n, C.cu, C.m_max = key, n, cu, m_max
    C.rep, C.r, C.gamma = 0, 0, 0.5
    return C


# --------------------------------------------------------------------- the model, no binary


def test_branch_scale_and_theta_are_two_names_for_one_number():
    m = seqsim.SeqModel(branch_scale=0.018)
    assert m.theta == pytest.approx(0.036)
    # 500 bp at cu=0.5 carries 4.5 expected substitutions on the median internal branch.
    assert m.expected_subs_per_internal_branch(0.5, 500) == pytest.approx(4.5)


def test_sim_model_string_is_a_real_hky_not_a_k2p():
    # AliSim defaults to equal base frequencies, which would silently make HKY a K2P.
    assert seqsim.SeqModel().sim_model_string() == "HKY{2}+F{0.3,0.2,0.2,0.3}+G4{1}"
    assert seqsim.SeqModel(gamma_shape=0).sim_model_string() == "HKY{2}+F{0.3,0.2,0.2,0.3}"


def test_batch_loci_bounds_the_likelihood_footprint():
    for n, bp in ((11, 500), (101, 500), (201, 1000)):
        k = seqsim.batch_loci(n, bp)
        assert 10 <= k <= 2000
        assert n * bp * k <= 4e6 or k == 10


def test_the_arm_directories_are_keyed_by_backend_and_bp():
    a, _ = seqsim.arm_dirs(Path("/r"), "julia", 500)
    b, _ = seqsim.arm_dirs(Path("/r"), "julia", 200)
    c, _ = seqsim.arm_dirs(Path("/r"), "oracle", 500)
    assert len({a, b, c}) == 3


# ------------------------------------------------------- AliSim behaviours the design assumes


@requires_iqtree
def test_plain_multi_tree_input_silently_uses_only_the_first_tree(tmp_path):
    """The reason AliSim is invoked once per locus rather than once per batch."""
    (tmp_path / "t.nwk").write_text("\n".join(THREE) + "\n")
    res = run([seqsim.IQTREE, "--alisim", tmp_path / "a", "-t", tmp_path / "t.nwk",
               "--length", 200, "-m", "HKY{2.0}+F{0.3,0.2,0.2,0.3}", "-af", "phy",
               "--seed", 1, "-T", 1, "--quiet"], tmp_path)
    assert res.returncode == 0
    assert len(list(tmp_path.glob("*.phy"))) == 1, "AliSim grew multi-tree support; revisit seqsim"


@requires_iqtree
def test_partition_mode_ignores_branch_scale(tmp_path):
    """Why the coalescent-unit rescale happens in Python, not via --branch-scale."""
    lens = {}
    for tag, extra in (("with", ["--branch-scale", 0.018]), ("without", [])):
        d = tmp_path / tag
        d.mkdir()
        (d / "t.nwk").write_text("\n".join(THREE) + "\n")
        (d / "p.nex").write_text("#nexus\nbegin sets;\n" + "\n".join(
            f"  charset loc{i} = {i * 200 + 1}-{(i + 1) * 200};" for i in range(3)) + "\nend;\n")
        res = run([seqsim.IQTREE, "--alisim", d / "a", "-t", d / "t.nwk", "-Q", d / "p.nex",
                   "-m", "HKY{2.0}+F{0.3,0.2,0.2,0.3}", *extra, "-af", "phy",
                   "--seed", 1, "-T", 1], d)
        # The TreeLen column of the model table -- the only place the applied scale shows up.
        lens[tag] = re.findall(r"^ *\d+ {2,}\S+ {2,}([0-9.]+) ", res.stdout, re.M)
    assert lens["with"] and lens["with"] == lens["without"], \
        "AliSim now honours --branch-scale in -Q; revisit seqsim's per-locus rescale"


@requires_iqtree
def test_partition_mode_aborts_when_indels_meet_rate_heterogeneity(tmp_path):
    """The other half of why AliSim runs per locus.  If this ever passes, batching is available."""
    (tmp_path / "t.nwk").write_text("\n".join(THREE) + "\n")
    (tmp_path / "p.nex").write_text("#nexus\nbegin sets;\n" + "\n".join(
        f"  charset loc{i} = {i * 200 + 1}-{(i + 1) * 200};" for i in range(3)) + "\nend;\n")
    res = run([seqsim.IQTREE, "--alisim", tmp_path / "a", "-t", tmp_path / "t.nwk",
               "-Q", tmp_path / "p.nex", "-m", "HKY{2.0}+F{0.3,0.2,0.2,0.3}+G4{1.0}",
               "--indel", "0.03,0.03", "--indel-size", "POW{1.7/100},POW{1.7/100}",
               "--no-unaligned", "-af", "phy", "--seed", 1, "-T", 1], tmp_path)
    assert res.returncode != 0
    assert "updateNewGenomeIndels" in res.stdout + res.stderr


@requires_iqtree
def test_indels_and_rate_heterogeneity_work_one_locus_at_a_time(tmp_path):
    (tmp_path / "t.nwk").write_text(THREE[0] + "\n")
    res = run(seqsim.alisim_cmd(tmp_path / "t.nwk", tmp_path / "a", 300,
                                seqsim.SeqModel(), 1), tmp_path)
    assert res.returncode == 0
    names, seqs = phylip.read_phylip(tmp_path / "a.phy")
    assert sorted(names) == ["t1", "t2", "t3", "t4"]
    assert len(seqs[0]) >= 300  # --length is the ROOT length; insertions widen the alignment


# ---------------------------------------------------------------- the IQ-TREE order contract


@requires_iqtree
def test_the_partition_file_pins_the_order_of_iqtrees_output_trees(tmp_path):
    """Highest-value test here: a permutation would corrupt the m axis with nothing raised."""
    paths = []
    for i, nwk in enumerate(THREE):
        (tmp_path / f"t{i}.nwk").write_text(nwk + "\n")
        assert run(seqsim.alisim_cmd(tmp_path / f"t{i}.nwk", tmp_path / f"l{i}", 3000,
                                     seqsim.SeqModel(gamma_shape=0), 11 + i),
                   tmp_path).returncode == 0
        paths.append(tmp_path / f"l{i}.phy")
    widths = phylip.write_superalignment(paths, tmp_path / "super.phy")

    truth = [splits(t, 4) for t in THREE]
    for tag, names in (("file", ["loc0", "loc1", "loc2"]), ("shuffled", ["c0", "a1", "b2"])):
        parts = tmp_path / f"{tag}.parts"
        start, body = 1, []
        for nm, w in zip(names, widths):
            body.append(f"DNA, {nm} = {start}-{start + w - 1}")
            start += w
        parts.write_text("\n".join(body) + "\n")
        assert run(seqsim.iqtree_cmd(tmp_path / "super.phy", parts, tmp_path / tag,
                                     seqsim.SeqModel(), 3), tmp_path).returncode == 0
        trees = [ln for ln in (tmp_path / f"{tag}.treefile").read_text().splitlines()
                 if ln.strip()]
        assert len(trees) == 3
        assert [splits(t, 4) for t in trees] == truth, f"{tag}: -S permuted the output trees"


# ------------------------------------------------------------------------- the whole chain


@requires_iqtree
@requires_cache
def test_the_true_tree_is_recovered_when_the_alignment_is_long(tmp_path):
    """With enough data the estimate must equal the truth, so any wiring error fails here."""
    with gzip.open(CACHED, "rt") as fh:
        trees = [next(fh).strip() for _ in range(4)]
    true_gz = tmp_path / "true.nwk.gz"
    with gzip.open(true_gz, "wt") as fh:
        fh.write("\n".join(trees) + "\n")

    root = tmp_path / "root"
    (root / "genetrees_julia").mkdir(parents=True)
    shutil.copy(true_gz, root / "genetrees_julia" / "n10_rep0_r0_cu0.5_g0.5.nwk.gz")
    model = seqsim.SeqModel(indel_ins=0.0, indel_del=0.0)  # no indels, so sites are homologous
    jobs = seqsim.plan_jobs([make_cell(m_max=4)], 20000, model, root, "julia", 4,
                            tmp_path, 20260805)
    key, status, _ = seqsim.run_est_job(jobs[0])
    assert status == "ok", status
    side = json.loads(jobs[0].sidecar.read_text())
    assert side["rf_norm_mean"] == 0.0, "20 kbp should recover every true gene tree exactly"
    assert side["frac_root_split_correct"] == 1.0


@requires_iqtree
@requires_cache
def test_the_chain_is_reproducible_and_chunk_invariant(tmp_path):
    """Chunking is an execution detail: same seeds, same T_out, same batch composition."""
    model = seqsim.SeqModel()

    def build(tag, target_seconds):
        root = tmp_path / tag
        (root / "genetrees_julia").mkdir(parents=True)
        shutil.copy(CACHED, root / "genetrees_julia" / "n10_rep0_r0_cu0.5_g0.5.nwk.gz")
        jobs = seqsim.plan_jobs([make_cell(m_max=20)], 500, model, root, "julia", 20,
                                tmp_path, 20260805, target_seconds=target_seconds,
                                iq_target_ns=6e4)
        for j in jobs:
            _, status, _ = seqsim.run_est_job(j)
            assert status == "ok", status
        seqsim.merge_chunks("n10_rep0_r0_cu0.5_g0.5", 500, root, "julia")
        d, _ = seqsim.arm_dirs(root, "julia", 500)
        return len(jobs), gzip.open(d / "n10_rep0_r0_cu0.5_g0.5.nwk.gz", "rb").read(), \
            json.loads((d / "n10_rep0_r0_cu0.5_g0.5.json").read_text())

    n_one, trees_one, side_one = build("one", 1e6)
    n_many, trees_many, side_many = build("many", 0.25)
    assert n_many > n_one == 1, "the chunked build did not actually chunk"
    # Compare decompressed bytes: gzip stores an mtime, so the containers differ.
    assert trees_one == trees_many
    for k in ("graft_t_out", "rf_norm_mean", "frac_root_split_correct", "gap_fraction",
              "aln_len_mean", "n_loci"):
        assert side_one[k] == pytest.approx(side_many[k]), k


@requires_iqtree
@requires_cache
def test_estimated_trees_carry_exactly_the_ingroup_taxon_set(tmp_path):
    root = tmp_path / "root"
    (root / "genetrees_julia").mkdir(parents=True)
    shutil.copy(CACHED, root / "genetrees_julia" / "n10_rep0_r0_cu0.5_g0.5.nwk.gz")
    jobs = seqsim.plan_jobs([make_cell(m_max=8)], 200, seqsim.SeqModel(), root, "julia", 8,
                            tmp_path, 20260805)
    _, status, _ = seqsim.run_est_job(jobs[0])
    assert status == "ok", status
    index = {f"t{i}": i - 1 for i in range(1, 11)}
    with gzip.open(jobs[0].rooted, "rt") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert len(lines) == 8
    for ln in lines:
        _, _, lab, ch = og.parse_newick(ln)
        assert {lab[i] for i in range(len(lab)) if not ch[i]} == set(index)
        extract.clusters_from_newick(ln, index, [])  # must not KeyError


@requires_iqtree
def test_the_model_goes_in_the_partition_file_because_it_overrides_dash_m(tmp_path):
    """A partition file's model field wins over -m, silently and with exit code 0.

    Left at the conventional ``DNA`` the whole arm would infer under GTR while every log line and
    every column said HKY -- matched-model turned misspecified without a single error.
    """
    (tmp_path / "t.nwk").write_text(THREE[0] + "\n")
    assert run(seqsim.alisim_cmd(tmp_path / "t.nwk", tmp_path / "a", 400,
                                 seqsim.SeqModel(indel_ins=0, indel_del=0), 1),
               tmp_path).returncode == 0
    widths = phylip.write_superalignment([tmp_path / "a.phy"], tmp_path / "s.phy")

    model = seqsim.SeqModel().inference_model
    phylip.write_partition_raxml(tmp_path / "dna.parts", widths, model="DNA")
    phylip.write_partition_raxml(tmp_path / "hky.parts", widths, model=model)
    got = {}
    for tag in ("dna", "hky"):
        assert run(seqsim.iqtree_cmd(tmp_path / "s.phy", tmp_path / f"{tag}.parts",
                                     tmp_path / tag, seqsim.SeqModel(), 3),
                   tmp_path).returncode == 0
        rows = re.findall(r"^ *\d+ {2,}(\S+) {2,}[0-9.]+ ",
                          (tmp_path / f"{tag}.iqtree").read_text(), re.M)
        got[tag] = {r.split("+")[0] for r in rows}
    assert got["dna"] == {"GTR"}, "the DNA field no longer forces GTR; revisit the guard"
    assert got["hky"] == {"HKY"}


@requires_iqtree
@requires_cache
def test_a_failed_iqtree_run_is_reported_not_swallowed(tmp_path):
    root = tmp_path / "root"
    (root / "genetrees_julia").mkdir(parents=True)
    shutil.copy(CACHED, root / "genetrees_julia" / "n10_rep0_r0_cu0.5_g0.5.nwk.gz")
    jobs = seqsim.plan_jobs([make_cell(m_max=4)], 200,
                            seqsim.SeqModel(inference_model="NOT_A_MODEL"), root, "julia", 4,
                            tmp_path, 20260805)
    key, status, _ = seqsim.run_est_job(jobs[0])
    assert status.startswith("FAILED")
    assert not jobs[0].rooted.exists()


def test_the_recorded_probe_answers_match_this_build():
    """``results/validation/alisim_probe.json`` is evidence, so it has to stay true."""
    path = REPO / "results" / "validation" / "alisim_probe.json"
    if not path.exists():
        pytest.skip("probe not run")
    p = json.loads(path.read_text())
    assert not p["p1a_multitree_naive"]["one_alignment_per_tree"]
    assert p["p1b_partition_defects"]["branch_scale"]["branch_scale_ignored"]
    assert not p["p1b_partition_defects"]["indel_with_G4"]["ok"]
    for tag in ("file", "shuffled"):
        assert p["p2_partition_order"][tag]["order_preserved"]


# --------------------------------------------------------------------- keeping alignments (A62)

#: 24 Julia gene trees on an n=8 level-2 (S0) network, ``tests/data/l2_n8_s0.enewick``.  The
#: cache-dependent tests above skip on ``main`` since ``results/full`` was pruned; this one does
#: not depend on any cache.
L2_FIXTURE = REPO / "tests" / "data" / "l2_n8_s0_24loci.nwk.gz"


def _l2_job(tmp_path, tag, aln_dir):
    return seqsim.EstJob(
        cell_key="l2_n8_s0", n_taxa=8, cu=0.5, bp=300, chunk=0, start=0, count=24, cell_m=24,
        per_batch=12, true_gz=L2_FIXTURE, out_dir=tmp_path / tag, seed_alisim=101,
        seed_iqtree=202, model=seqsim.SeqModel(), scratch=tmp_path / "scratch", aln_dir=aln_dir)


@requires_iqtree
def test_keeping_alignments_adds_files_and_changes_nothing_else(tmp_path):
    plain = _l2_job(tmp_path, "plain", None)
    kept = _l2_job(tmp_path, "kept", tmp_path / "aln")
    for job in (plain, kept):
        assert seqsim.run_est_job(job)[1] == "ok"
    # Compared decompressed: the tree files' gzip headers carry their write time, to the second,
    # so two identical runs straddling a second boundary differ in bytes and nowhere else.  (The
    # kept alignments are written with mtime=0, so *their* containers are reproducible.)
    for name in ("rooted", "raw"):
        assert gzip.decompress(getattr(plain, name).read_bytes()) == \
            gzip.decompress(getattr(kept, name).read_bytes())
    assert "aln_dir" not in json.loads(plain.sidecar.read_text())

    alns = kept.aln_paths()
    assert len(alns) == 24 and all(p.exists() for p in alns)
    with gzip.open(alns[0], "rt") as fh:
        rows = [r for r in fh.read().split("\n") if r.strip()]
    n, width = map(int, rows[0].split())
    assert n == 9 and {r.split()[0] for r in rows[1:]} == {f"t{i}" for i in range(1, 9)} | {"OUT"}
    side = json.loads(kept.sidecar.read_text())
    assert side["aln_dir"].endswith("l2_n8_s0")
    assert side["aln_len_min"] <= width <= side["aln_len_max"]

    # Cached on a second call; and a cell missing one alignment is not done, and is redone
    # to the same bytes, because the seeds key off the global locus index.
    before = [p.read_bytes() for p in alns]
    assert seqsim.run_est_job(kept)[1] == "cached"
    alns[3].unlink()
    assert seqsim.run_est_job(kept)[1] == "ok"
    assert [p.read_bytes() for p in alns] == before
