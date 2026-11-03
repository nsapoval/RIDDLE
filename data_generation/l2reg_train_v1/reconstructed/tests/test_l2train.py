"""``l2reg_train_v1``'s design, and the two things it changed underneath ``l2reg`` (A65).

The build itself is gated by ``python -m opc.l2train probe``.  What is pinned here is what a
probe cannot see: that the design is the one fixed with the user, that no seed stream is shared
with the published benchmarks, and -- the reason :func:`opc.l3reg.l2_identity` is not enough --
that ``l2reg``'s *own* estimation jobs are byte-for-byte what they were before ``n`` became a
property of the spec.  ``l2_identity`` re-derives networks only, so a regression in
``estimation_jobs`` would sail past it.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from opc import l2reg, l2train as T
from opc import l3reg
from opc.grid import chain_seed
from opc.simulate import level2 as L2
from opc.simulate import outgroup as og


# ------------------------------------------------------------------------------ the design


def test_the_design_is_the_one_fixed_with_the_user():
    for phase, n in T.PHASES.items():
        specs = T.design(phase)
        assert len(specs) == T.REPS == 900
        assert len({s.key for s in specs}) == len(specs)
        # 3-digit reps, so 900 keys sort as text -- 20 keys at :02d do not generalise.
        assert [s.key for s in specs] == sorted(s.key for s in specs)
        assert all(s.n_taxa == n and s.phase == phase for s in specs)
        assert Counter(s.split for s in specs) == {"train": 700, "val": 100, "test": 100}
        # Every (layout, cu) cell is the same size in every split, which is what both rotation
        # cycles dividing 100 buys: a split is a prefix of the replicates.
        cells = Counter((s.n_l2, s.n_l1, s.cu, s.split) for s in specs)
        per_cell = T.REPS // (len(L2_CUS := l2reg.CUS) * len(T.LAYOUTS[n]))
        for a, b in T.LAYOUTS[n]:
            for cu in L2_CUS:
                assert cells[(a, b, cu, "train")] == per_cell * 7 // 9
                assert cells[(a, b, cu, "val")] == cells[(a, b, cu, "test")] == per_cell // 9
        assert {(s.n_l2, s.n_l1) for s in specs} == set(T.LAYOUTS[n])
        assert all(s.k == 2 * s.n_l2 + s.n_l1 for s in specs)


def test_k_starts_at_the_smallest_level_2_network_and_tracks_the_level_1_ladders():
    """k = 2 is one level-2 blob and nothing else; the tops sit near x25's 3 and x50's 5."""
    assert (min(s.k for s in T.design("l2t25")), max(s.k for s in T.design("l2t25"))) == (2, 5)
    assert (min(s.k for s in T.design("l2t50")), max(s.k for s in T.design("l2t50"))) == (2, 7)


def test_shapes_are_balanced_and_never_repeated_inside_one_network():
    """The rotation does two jobs.  Balance is the obvious one.  The other is that a two-blob
    network draws *consecutive* shapes, so it never asks for the same shape twice -- which is
    the one placement case measured to fail at n = 25 (``S2,S2`` built 3 of 4)."""
    for phase in T.PHASES:
        specs = T.design(phase)
        for s in specs:
            assert len(set(s.shapes)) == len(s.shapes) == s.n_l2
            assert set(s.shapes) <= set(L2.SHAPES)
        for cu in l2reg.CUS:
            counts = Counter(sh for s in specs if s.cu == cu for sh in s.shapes)
            assert set(counts) == set(L2.SHAPES)
            assert max(counts.values()) - min(counts.values()) <= 1


def test_no_seed_stream_or_key_is_shared_with_the_published_benchmarks():
    """A dataset's own master seed is the leakage fix, not hygiene: with a shared one this set's
    replicates would reproduce another dataset's cells on every stream."""
    mine = T.all_specs()
    theirs = l2reg.design() + l3reg.design()
    for attr in ("net_seed", "gt_seed", "est_seeds", "key"):
        assert not {getattr(s, attr) for s in mine} & {getattr(s, attr) for s in theirs}, attr
    # n is in the network chain, so the two phases' replicate 0 do not share a base tree either.
    assert T.design("l2t25")[0].net_seed != T.design("l2t50")[0].net_seed


def test_est_seeds_are_derived_exactly_as_seqsim_derives_them():
    for phase in T.PHASES:
        s = T.design(phase)[0]
        assert s.est_seeds == (
            chain_seed(T.MASTER_SEED, "alisim", s.key, l2reg.BP) % (2 ** 31),
            chain_seed(T.MASTER_SEED, "iqtree", s.key, l2reg.BP) % (2 ** 31))


def test_row_carries_the_split_and_survives_a_round_trip():
    s = T.design("l2t25")[750]
    row = s.row()
    assert row["split"] == "val" and row["key"] == s.key and row["k"] == s.k
    assert json.loads(json.dumps(row))["shapes"] == list(s.shapes)


# ------------------------------------------------------------ what l2_identity cannot see


def test_l2regs_own_estimation_jobs_are_unchanged(tmp_path):
    """``n`` moved from a module constant to a property on the spec.  For the published design
    that has to be a no-op, and the networks (which ``l2_identity`` checks) are only half of it:
    these values are the other half."""
    jobs = l2reg.estimation_jobs(l2reg.Paths(l2reg.PKG), l2reg.design(), tmp_path, 5)
    assert len(jobs) == 100                      # 20 networks x 5 chunks
    j = jobs[0]
    assert (j.n_taxa, j.per_batch, j.n_chunks, j.count) == (200, 19, 5, 209)
    assert j.stem == "n200_rep00_l2x2_l1x2_cu0.25.c000"
    assert j.aln_dir == l2reg.Paths(l2reg.PKG).aln   # l2reg keeps its alignments
    assert sum(x.count for x in jobs[:5]) == l2reg.N_LOCI


@pytest.mark.parametrize("phase,per_batch", [("l2t25", 153), ("l2t50", 78)])
def test_training_jobs_batch_at_their_own_n(tmp_path, phase, per_batch):
    """``batch_loci`` is a footprint bound in (n, bp): 19 loci per IQ-TREE process at n = 200,
    153 at n = 25.  Reading it off the module constant would put 19 there too -- correct, but
    eight times more process starts than the batch is worth."""
    specs = T.design(phase)[:3]
    jobs = l2reg.estimation_jobs(T.paths_of(T.PKG, phase), specs, tmp_path, 1)
    assert len(jobs) == 3                        # --chunks 1: one task per network, as alisim_gt
    j = jobs[0]
    assert (j.n_taxa, j.per_batch, j.n_chunks, j.count) == (T.PHASES[phase], per_batch, 1, 1000)
    assert j.stem == specs[0].key                # no .cNNN segment, so no merge step
    assert j.aln_dir == T.paths_of(T.PKG, phase).aln


def test_stage_tob_keeps_its_default_note():
    import inspect

    got = inspect.signature(l2reg.stage_tob).parameters["note"].default
    assert got.startswith("~3 h each at n = 201")


# ------------------------------------------------------------------------------ networks


def test_stage_networks_writes_every_network_and_is_idempotent(tmp_path):
    specs = T.design("l2t25")[:2]
    assert T.stage_networks(tmp_path, specs, jobs=2) == 0
    paths = T.paths_of(tmp_path, "l2t25")
    for s in specs:
        rec = json.loads(paths.meta(s.key).read_text())
        assert rec["check"]["ok"] and rec["n_taxa"] == 25
        assert rec["spec"]["split"] == "train"
        assert abs(rec["median_internal_branch"] - s.cu) < 1e-9
        assert rec["placement"]["cfg"]["max_base"] == T.MAX_BASE
        assert len(rec["check"]["shapes"]) == s.n_l2
        for p in (paths.net(s.key), paths.regular(s.key), paths.pickle(s.key)):
            assert p.stat().st_size > 0
    before = {s.key: paths.meta(s.key).stat().st_mtime_ns for s in specs}
    assert T.stage_networks(tmp_path, specs, jobs=2) == 0
    assert {s.key: paths.meta(s.key).stat().st_mtime_ns for s in specs} == before


def test_one_failed_build_reports_its_key_and_keeps_the_others(tmp_path, monkeypatch):
    """The l3reg pattern this stage is modelled on collects a whole ``pool.map`` before writing
    anything, so one exception discards every good network with it.  Here a build is a function
    of its seed: a failure is reported by key and **never reseeded**, and the rest are on disk."""
    specs = T.design("l2t25")[:2]
    doomed = specs[1].key
    real = L2.build_level2

    def sometimes(n_taxa, layout, seed, **kw):
        if seed == specs[1].net_seed:
            raise RuntimeError(f"no {layout} network in {T.MAX_BASE} base trees")
        return real(n_taxa, layout, seed, **kw)

    # Patched before the pool forks, so the workers inherit it.
    monkeypatch.setattr(L2, "build_level2", sometimes)
    assert T.stage_networks(tmp_path, specs, jobs=2) == 1
    paths = T.paths_of(tmp_path, "l2t25")
    assert paths.meta(specs[0].key).exists()
    assert not paths.meta(doomed).exists() and not paths.net(doomed).exists()


# ------------------------------------------------------------------------------ lengths


#: ``OUT`` nested two levels down, so rerooting reverses a path and suppresses the old root --
#: the case where the length of the replacement edge is a *sum* and a naive port loses one.
NESTED = "((t1:0.1,(OUT:0.7,t2:0.2):0.3):0.05,(t3:0.3,t4:0.15):0.25);"
FLAT = "(OUT:0.4,(t1:0.1,t2:0.2):0.05,(t3:0.3,t4:0.15):0.25);"


def _distances(newick: str) -> dict[frozenset, float]:
    """Patristic distance between every pair of leaves, from the parsed tree."""
    parent, length, lab, children = og.parse_newick(newick)
    adj: list[list[tuple[int, float]]] = [[] for _ in parent]
    for i in range(1, len(parent)):
        adj[i].append((parent[i], length[i]))
        adj[parent[i]].append((i, length[i]))
    leaves = [i for i in range(len(parent)) if not children[i]]
    out = {}
    for src in leaves:
        seen, stack = {src: 0.0}, [src]
        while stack:
            x = stack.pop()
            for y, w in adj[x]:
                if y not in seen:
                    seen[y] = seen[x] + w
                    stack.append(y)
        for dst in leaves:
            if dst != src:
                out[frozenset((lab[src], lab[dst]))] = seen[dst]
    return out


@pytest.mark.parametrize("tree", [FLAT, NESTED])
def test_keep_lengths_preserves_the_topology_and_every_ingroup_distance(tree):
    plain = og.reroot_on_outgroup(tree)
    with_len = og.reroot_on_outgroup(tree, keep_lengths=True)
    # The invariant the lengths stage checks on every line it writes.
    assert T._strip_lengths(with_len) == plain
    before, after = _distances(tree), _distances(with_len)
    ingroup = [p for p in before if og.OUTGROUP_LABEL not in p]
    assert ingroup and all(abs(before[p] - after[p]) < 1e-9 for p in ingroup)


def test_keep_lengths_defaults_to_off_so_landed_artifacts_still_re_derive():
    """``root_and_prune_file`` wrote ``alisim_gt_v1/v2`` and ``l2reg_n200_v1`` with lengths
    dropped; the default has to stay that way or every published estimate moves."""
    import inspect

    assert inspect.signature(og.reroot_on_outgroup).parameters["keep_lengths"].default is False
    assert ":" not in og.reroot_on_outgroup(NESTED)


def test_stage_lengths_writes_paired_trees_and_checks_them(tmp_path):
    """End to end on hand-written 'raw' trees: same count, same order, lengths kept."""
    paths = T.paths_of(tmp_path, "l2t25")
    spec = T.design("l2t25")[0]
    raws = [FLAT, NESTED, FLAT]
    paths.est.mkdir(parents=True, exist_ok=True)
    with gzip.open(paths.est / f"{spec.key}.raw.nwk.gz", "wt") as fh:
        fh.write("\n".join(raws) + "\n")
    with gzip.open(paths.est / f"{spec.key}.nwk.gz", "wt") as fh:
        fh.write("\n".join(og.reroot_on_outgroup(r) for r in raws) + "\n")

    taxa = [f"t{i}" for i in range(1, 5)]
    key, status, _secs, n = T._lengths_one(
        (spec.key, paths.est / f"{spec.key}.raw.nwk.gz", paths.est / f"{spec.key}.nwk.gz",
         paths.est / f"{spec.key}.len.nwk.gz", taxa))
    assert status == "ok" and n == 3
    with gzip.open(paths.est / f"{spec.key}.len.nwk.gz", "rt") as fh:
        got = [ln.strip() for ln in fh if ln.strip()]
    assert len(got) == 3 and all(":" in g for g in got)
    assert [T._strip_lengths(g) for g in got] == [og.reroot_on_outgroup(r) for r in raws]
    # A landed topology that disagrees is a failure, not a silent overwrite.
    with gzip.open(paths.est / f"{spec.key}.nwk.gz", "wt") as fh:
        fh.write("\n".join(["((t1,t2),(t3,t4));"] * 3) + "\n")
    (paths.est / f"{spec.key}.len.nwk.gz").unlink()
    assert T._lengths_one(
        (spec.key, paths.est / f"{spec.key}.raw.nwk.gz", paths.est / f"{spec.key}.nwk.gz",
         paths.est / f"{spec.key}.len.nwk.gz", taxa))[1].startswith("FAILED")


# ------------------------------------------------------------------------------ the estimator


def test_the_pinned_iqtree_is_the_one_v1_v2_used():
    """Not "an IQ-TREE": the arm this data is mixed with at training time was made with 2.4.0,
    and the repo default is 2.3.6."""
    assert "2.4.0" in T.IQTREE_VERSION
    assert T.IQTREE_BINARY != Path("/shared/ns58/conda_envs/mpphysmc-env/bin/iqtree2")


def test_use_pinned_iqtree_yields_to_an_explicit_override(monkeypatch):
    from opc.simulate import seqsim

    monkeypatch.setattr(seqsim, "IQTREE", Path("/somewhere/else/iqtree2"))
    monkeypatch.setenv("OPC_IQTREE", "/somewhere/else/iqtree2")
    T.use_pinned_iqtree()
    assert seqsim.IQTREE == Path("/somewhere/else/iqtree2")


# ------------------------------------------------------------------------------ the package


def test_the_training_list_is_the_light_half_and_names_real_paths():
    """The GPU box is tight on disk, so a copy has to be able to leave the sequences behind.

    Three omissions are deliberate and each one is a file the trainer does not read: the
    alignments (~90% of the package), ``<key>.raw.nwk.gz`` (superseded by ``.len.nwk.gz``, which
    is the same tree rooted and pruned with its lengths), and ``<key>.pkl`` (this repo's
    ``Network`` class, which the reader would have to have).
    """
    specs = T.design("l2t25")[:2] + T.design("l2t50")[:2]
    rels = T.training_file_list(specs)
    assert len(rels) == 4 + 7 * len(specs)
    assert not [r for r in rels if "alignments_bp" in r or r.endswith((".raw.nwk.gz", ".pkl"))]
    for s in specs:
        assert f"{s.phase}/trees/{s.key}.nwk.gz" in rels
        assert f"{s.phase}/estimated_bp{l2reg.BP}/{s.key}.len.nwk.gz" in rels
        assert f"{s.phase}/networks/{s.key}.enewick" in rels
    # rsync --files-from needs relative paths with no leading slash and no "..".
    assert all(not r.startswith("/") and ".." not in r for r in rels)
    assert {"MANIFEST.tsv", "manifest.json", "README.md", "TRAINING_FILES.txt"} <= set(rels)


def _fake_records(specs):
    rows, recs, sides, tobs = [], {}, {}, {}
    for s in specs:
        recs[s.key] = {
            "check": {"ok": True, "shapes": list(s.shapes), "girth_R_min": 5,
                      "regularity": {"tree_child": True}},
            "blobs": [{"girth_R": 5, "root_taxa": 7, "n_retics": 2}],
            "placement": {"base_draw": 0},
        }
        sides[s.key] = {"aln_len_mean": 1125.6, "gap_fraction": 0.113, "rf_norm_mean": 0.087,
                        "iqtree_model_used": "HKY+F+G4"}
        tobs[s.key] = {"n_true_edges": 20, "n_est_edges": 19, "fn": 1, "fp": 0}
        rows.append({"phase": s.phase, "n_l2": s.n_l2, "n_l1": s.n_l1, "cu": s.cu,
                     "split": s.split})
    return rows, recs, sides, tobs


def test_the_readme_renders():
    """It is generated by the last stage of a ten-hour run, so a bad format string there is found
    at the worst possible moment.  This renders it against a plausible record instead."""
    specs = T.design("l2t25")[:2] + T.design("l2t50")[:2]
    rows, recs, sides, tobs = _fake_records(specs)
    manifest = {
        "name": "l2reg_train_v1", "built_utc": "2026-09-11T23:59:59+00:00",
        "code": {"commit": "0123456789abcdef", "dirty": True},
        "tools": {"iqtree": T.IQTREE_VERSION},
    }
    text = T.readme(specs, rows, recs, sides, tobs, manifest,
                    {"training": 1_800_000_000, "alignments": 18_000_000_000,
                     "other": 2_000_000_000})
    assert "# l2reg_train_v1" in text
    assert "rsync -a --files-from=TRAINING_FILES.txt" in text
    assert "--exclude 'alignments_bp1000/'" in text
    assert "1.8 GB" in text and "90%" in text        # the split, stated in the README
    assert "len.nwk.gz" in text and T.IQTREE_VERSION in text
    # A bare "{" is fine here -- the realised table prints dicts and the model string is
    # HKY{2}+F{...} -- so check the names this template actually interpolates.
    assert not [x for x in ("{name}", "{BP}", "{aln_pct", "{train_gb", "{full_gb", "{n_train")
                if x in text]
