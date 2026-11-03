# taxmiss_eval: the benchmark on gene trees with missing taxa

The published CAMUS, L2reg, L3reg and NetCS benchmarks, with ingroup taxa deleted from the gene trees tree by tree
(the Linux box built the inputs and TREE-QMC's trees of blobs). This package runs four methods on them, the ztc
benchmark's:

| method | what runs | label in the tables |
|---|---|---|
| GNN | `gnn_cpu_v1/run_gnn_cpu.py --partial 4 --m-from-trees`, `_l2mix_pair/s0`, relaxed L2 (L3 on L3reg) | `ours/l2mix_pair/selF1 [relaxed L<L>]`, bench `GNN` |
| GNN MAP [z+tc] | `nni_ensemble/run_derived.sh` on the `tm-*` groups: score → features → credible → make_csv → bench (the published, cross-fit beta) | `GNN MAP [z+tc]` in `results/derived/` |
| **RIDDLE** | the same MAP under the **train-fit** beta: `nni_ensemble/run_trainfit.sh derived tm-*` on `rest`'s feats | `GNN MAP [z+tc]` in `results/trainfit/derived/` |
| BROOQS, NetCS | `run_res.py` on the raw TOB rooted on OUT (`--variants raw,fixed` adds the fixed TOB on l2reg/l3reg) | `BROOQS`, `NetCS` |

**RIDDLE is the final call; the vanilla GNN is kept on record.** Every bench table has a `GNN` row next to the MAP's,
and `results/<ds>_taxmiss.csv` has every GNN network with its runtimes. The notebook draws both (RIDDLE = the
train-fit MAP only).

The **control** is the published benchmark itself, `nni_ensemble/results/<ds>_bench.csv` (RIDDLE's:
`results/trainfit/<ds>_bench.csv`). A condition's cells are the published cells, with the same networks, the same
loci and the same estimated trees pruned (stage gate T1b), so every comparison is paired cell by cell. Nothing is
re-run for the control. NetCS is the exception on two counts: OUT is re-placed on ~5% of its trees, and its resolvers
are cross-protocol against the control. See *NetCS*.

Results:
- RIDDLE: `nni_ensemble/results/trainfit/derived/<ds>_tm_<cond>_bench.csv`.
- Cross-fit: `nni_ensemble/results/derived/<ds>_tm_<cond>_bench.csv`.
- Both of those tables hold all four methods.
- Per network: `taxmiss_eval/results/<ds>_taxmiss.csv`, with missingness, TOB error and runtimes.
- Figures: `notebooks/plotting_taxmiss.ipynb` → `plots/{CAMUS,AllDatasets}-taxmiss-*-ours_l2mix_pair_selF1.png`.

## The data

| path | what |
|---|---|
| `inputs/<ds>/MANIFEST.tsv` | every (condition, cell): `cond p cap M phase rep arm cell key n n_loci tier path sha256_source sha256_true sha256_staged n_modified mean_missing` |
| `inputs/<ds>/<cond>/n<N>/<RR>/` | the trees (`g_500.nwk`, `iqtree_500.nwk`, `est_bp1000.raw.nwk`), `missing.tsv` (per locus: what was removed), and the truth or the true TOB |
| `inputs/netcs/<cond>/L<l>_n<N>/<rep>/` | four ILS arms in one directory: `est_ils<x>.nwk` with **its own** `est_ils<x>.missing.tsv` (`true_locus` is the corpus's line), and `true_net.nwk` |
| `runs/tob/<ds>/<cond>/<cell>.{json,raw.tob,tob,pvalues.tob}` | TREE-QMC 4.1.5 `--blob` on exactly those trees (built on Linux) |

- **A condition** `p<P>_c<CAP>`: each tree loses taxa with probability `p`, at most `CAP`% of n (`M` taxa).
  The MANIFEST lists 9 conditions, p ∈ {0.80, 0.90, 0.95} × cap ∈ {12, 24, 36}, plus `control`.
- **On the Mac so far: `p0.90_c24`.** CAMUS has 100 cells (cam25 `cam_iq500`, cam50–200 `cam_g500`); L2reg and
  L3reg have 20 each at n = 200.
  - Every tree carries OUT.
  - About 900 of 1,000 trees per cell lost something.
  - The mean share lost is 0.155 on CAMUS (0.19 at n = 25, falling to 0.12 at n = 200), 0.115 on L2reg and L3reg.
  - The smallest tree keeps 19 taxa.
- **NetCS `p0.90_c24` landed 2026-10-23.** It has 1,200 cells: L1/L2 × n = 50/100/200 × ILS 0.25/0.5/1.0/2.0 × 50
  networks.
  - Each cell holds 911–1,000 loci, the corpus's own count.
  - The mean share lost is 0.148, and the smallest tree keeps 38 taxa.
  - See *NetCS* for what differs from the other three families.
- **More conditions are coming for CAMUS, L2reg and L3reg.** Everything is keyed on (family, condition) and finds
  conditions by what is staged, so `stage.py` then `run_all.sh` picks them up.

## The pieces

| file | role |
|---|---|
| `tm_common.py` | paths, the family table, MANIFEST readers, `PARTIAL = 4`; imports `subsample_eval/common.py` as `common` (BLAS pin, `gnn_cpu_v1/pylib` first) |
| `stage.py` | per (family, condition) present: `staged/<ds>/<cond>/` in camus-data's layout (relative symlinks to the trees, the **parent's** truth copied byte for byte), `CELLS.tsv`, `record.json`; `--scan` writes `scan.tsv` (\|C\| under the partial count). Gates T1–T6 |
| `run_gnn.sh` | `run_gnn_cpu.py --partial 4` per (dataset, condition, phase; rep halves on l2reg/l3reg), MAXJ at once |
| `run_res.py` | BROOQS and NetCS, a task per (condition, cell, variant, resolver), keyed by content id, LPT order, timeouts and invariants; `subsample_eval/run_res.py`'s wrapping, scoring and bookkeeping, loaded by path. Gates S1–S4 |
| `make_csv.py` | `results/<ds>_taxmiss.csv` + `_gates.json` + `_summary.txt`; the jar's luay; the **landed rows** `nni_ensemble` reads for this dataset |
| `check.py` | P1–P4, the partial path held to code that did not write it |
| `run_all.sh` | every step in order (gate → gnn → res → score → csv → rest → **trainfit** → nb), stopping at the first failure |

## What partial trees needed, and what changed elsewhere (every change is off by default)

The GNN chain assumed full-taxon trees in three places, and failed loudly or silently in each:

- **Loudly:** `camus.prepare_gene_trees` raised on any tree that was missing a taxon.
- **Silently, cluster support:** `extract.count_newick_file` kept each partial tree's root cluster. That cluster is the
  tree's taxon sampling, not evidence of a clade.
- **Silently, triple counts:** both of `concord`'s triple kernels counted ab|c whenever c was absent. N3's reference
  shared the same blind spot, so N3 could not catch it.

None of the original files is in git, so the originals were backed up to the session scratchpad before editing.

| file | change |
|---|---|
| `gnn_cpu_v1/pylib/opc/rl/camus.py` | `prepare_gene_trees(partial=MIN)`: a tree's leaf set must be a subset of the truth's, carry OUT and hold at least MIN taxa. A tree that fails raises; nothing is skipped. Returns `n_partial`, `min_taxa`, `mean_missing` |
| `gnn_cpu_v1/pylib/opc/clusters/extract.py` | `count_newick_file(drop_partial_root=True)`: drops a partial tree's own leaf set, which is `infer_gnn_cpu.count_clusters`' rule |
| `gnn_cpu_v1/run_gnn_cpu.py` | `--partial MIN`. The cell JSON gains `partial` and the tree stats; resume is refused across a different MIN. `write_tsv` now uses a per-process temp name, because processes sharing a run directory raced on `timings.tmp` (found by the smoke here) |
| `cluster_anatomy/scoring.py` | `build_cell` reads `ds.extra["partial"]`: the same two calls |
| `cluster_anatomy/common.py` | `_derived()` registers `<ds>_tm_<cond>` (`kind="taxmiss"`) for every staged condition. `base=<ds>`, so the fold is the parent network's |
| `nni_ensemble/concord.py` | the partial-aware kernels: block (`Out = S_root − S_v`), dense (`partial_triple_counts`) and the slow reference (c must be in the tree); plus `absent_c_counts` for N3′(c) |
| `nni_ensemble/features.py` | passes partial through; `head["partial"]`; adds **N3′(c)**: the full-taxon kernel's excess over the partial one is exactly the absent-c triples |
| `nni_ensemble/ne_common.py` | `variant(ds)`: `("draw", k)` for sub, `("cond", cond)` for taxmiss; `sc()` for taxmiss |
| `nni_ensemble/bench.py`, `make_csv.py` | key and filter by `variant(ds)`. A taxmiss dataset keeps the BROOQS and NetCS baselines, read from `results/<ds>_taxmiss.csv` |
| `nni_ensemble/derived.py`, `run_derived.sh` | the groups `tm-camus`, `tm-reg` and `tm-netcs`, with stems `taxmiss_{camus,reg,netcs}`. `TM_CONDS` restricts them, and the bench stem then names those conditions |

**Added when NetCS landed (2026-10-23).** Each change is a no-op on camus/l2reg/l3reg. Their `check.py` cells and
stage gates are unchanged, checked.

| file | change |
|---|---|
| `taxmiss_eval/stage.py` | T1's netcs link: corpus `.estimated_gene_trees` → `netcs_eval/stage.json` (`source_trees`, `trees`) → the parent's `est_ils<x>.nwk`. `missing_path()` reads the per-arm `est_ils<x>.missing.tsv`. **T1b's ingroup mode** for netcs (`T1B_INGROUP`), which writes `t1b.tsv` |
| `taxmiss_eval/run_gnn.sh` | `--m-from-trees`, so m is the cell's own locus count. Without it a netcs cell ran at m = 1000 against 911–1,000 trees, and score parity failed on every smoked cell |
| `taxmiss_eval/check.py` | the lowest rep per (phase, arm), not `rep == 0`: netcs reps are network ids, so the old rule picked 8 cells of 1,200. The temp file name carries the arm |
| `taxmiss_eval/run_res.py` | `SMOKE["netcs"]`: subsample_eval's two cells |
| `taxmiss_eval/run_all.sh` | a `trainfit` step: `run_trainfit.sh derived tm-<group>` after `rest` |
| `nni_ensemble/derived.py` | `TAIL["max_a"]` 1,491 → 1,631 → 2,021 (radius sets 2,042,220), the netcs taxmiss tail cells, each measured alone first |
| `notebooks/plotting_taxmiss.ipynb` | the train-fit MAP is labelled **RIDDLE** (`MAP_NAME`; `"cross"` says cross-fit). NetCS resolver deltas against control are drawn hollow, with `protocol` = `cross` in the table. Adds the re-rooting caveat |

**Verified unchanged** (2026-10-21, into scratch):
- `run_gnn_cpu.py` without `--partial` reproduces the landed cam50 r0 and l2reg r0 cells in every field except the
  clocks.
- `score_cells --check-forward` on published cam50 r0: parity holds on every decoder.
- `features` on published l2reg r0 reproduces its z5, z6 and z7 feat files, apart from `seconds`.
- Default `bench.py` re-derives all six published files byte for byte, and so do the `sub500_camus`, `sub500_reg` and
  `gtscale` benches.
- `make_csv` on `l2reg_sub500_d0` re-derives its eight tables; the `.gz` files differ only in their gzip header
  timestamp.
- `check_anatomy` passes with the taxmiss entries registered.

## Gates

- **`stage.py` (T1–T6).**
  - T1: each staged file hashes to `sha256_staged`. The parent's source hashes to `sha256_source`.
    - On l2reg/l3reg that source is the package's `.raw.nwk.gz`, which decompresses to `l2reg_eval/inputs`' file.
    - On netcs it is the corpus's unrooted `.estimated_gene_trees`. `netcs_eval/stage.json` records it as the source
      of the parent's OUT-inserted file, and both hashes are checked.
  - **T1b: each tree is the parent's tree pruned to its own leaves and nothing else.** Checked on all 140,000
    camus/l2reg/l3reg trees.
    - **On netcs, the ingroup only.** OUT moved on 4.7% of its trees (see *NetCS*), so T1b deletes OUT from both
      trees, compares unrooted splits, and *counts* the trees where only OUT differs (`t1b.tsv`).
    - A full scan of all 1,170,401 netcs trees found no ingroup difference.
  - T2: **every tree's** leaf set is the truth + OUT − `missing.tsv`'s removed, and `n_modified` / `mean_missing`
    reproduce.
  - T3: the TOB JSON names this input, by path and SHA.
  - T4: the truth is the parent's.
  - T5: the partial read accepts each staged cell, and the strict read refuses it.
  - T6: `NO_REFERENCE.tsv` does not exist.
- **`run_res.py --check` (S1–S4):** provenance, leaf sets, wrapping, the A63 pairing of raw and fixed TOBs, and the
  truth equal to the landed one.
- **`check.py` (P1–P4)** on the lowest rep of every (phase, arm), which is rep 0 except on netcs, plus the
  largest-\|C\| cell per condition:
  - P1: the extract partial count equals `infer_gnn_cpu.count_clusters`, and is non-vacuous.
  - P2: `infer_gnn_cpu.infer(partial=4)`, which is the other driver end to end, gives `run_gnn_cpu`'s network.
  - P3: block-partial equals dense-partial equals the slow reference. Full-taxon minus partial equals the absent-c
    count, which is non-zero on every cell.
  - P4: on the published full-taxon trees, every partial counter equals the published one bit for bit.
- **The derived chain:**
  - score parity against `run_gnn.sh`'s cells;
  - N0 (support recounted the partial way);
  - N3 and N3′(c);
  - folds equal the parent's;
  - luay parity against `results/<ds>_taxmiss.csv`;
  - bench B1–B4.
- **`make_csv.py`:**
  - completeness: nothing is written while a GNN cell or a raw-TOB resolver record is missing;
  - m == n_loci == the MANIFEST's, and `partial` == 4;
  - `n_partial` == the MANIFEST's `n_modified`;
  - the two hybrid-ancestry derivations agree;
  - every network gets a luay;
  - the split comparison reproduces each TOB JSON's `fn_raw` / `fp_raw` / `n_true_edges` (140 of 140).
    That comparison is what scores the control TOB's error, `ctrl_tob_*`.

## Measured (2026-10-21, the Mac, smokes on rep 0 and on the largest-|C| cells)

- **|C| at τ = 2 under the partial count** (`stage.py --scan`): a median of 0.88–0.93× the published, at most 1.06×.
  The largest is 10,706 (l3reg rep18). So the GNN's cost is the published one: ≤ 28 s and ≤ 0.93 GB per cell for
  score, ≤ 25 s and ≤ 0.5 GB for `run_gnn_cpu`.
- **MAP tail:** blocks reach 9 atoms (limit 24) and 55 options (limit 1,491). `features` takes ≤ 2 s and < 0.45 GB per
  cell, `credible` < 1 s.
- **Resolvers:** 0.7–1.7× the landed CPU of the same cell. The l3reg rep00 and cam100/cam200 rep00 NetCS outputs are
  `out_not_rootable`, a recorded outcome under the landed rules.
- **Disk:** a scored cell is ~0.13 MB, everything else is smaller; under 100 MB per condition.

| step | jobs | rough wall, one condition × three families |
|---|---|---|
| gate | — | 30 s |
| gnn | 9 processes, 6 at once | ~10 min |
| res | 280 tasks at `--jobs 12`, 3.6 CPU-h projected | ~20–35 min |
| score | tm-camus 5 jobs, tm-reg 4 | ~10–13 min |
| csv | the jar, 4 JVMs | 1–3 min |
| rest | per group | ~3–5 min each |
| nb | — | 10 s |

**Already run, and left in place so the full run resumes over it:**
- GNN on the 12 smoke cells;
- 14 resolver tasks (rep 0 of every camus phase, l2reg rep 0, l3reg rep 0);
- 7 scored cells in `cluster_anatomy/runs/*_tm_p0.90_c24/`;
- 5 camus feat and cred files in `nni_ensemble/runs/camus_tm_p0.90_c24/`.

The partial tables those smokes wrote were deleted again, so nothing under `results/` predates your run.

## Run it (yours)

The whole thing, stopping at the first failure. Logs go to `taxmiss_eval/runs/logs/all_<step>.log`:

```zsh
cd ~/research/rl-net-opt
caffeinate -i bash taxmiss_eval/run_all.sh
```

Or one step at a time, in this order:

```zsh
cd ~/research/rl-net-opt
STEPS=gate bash taxmiss_eval/run_all.sh
caffeinate -i bash taxmiss_eval/run_gnn.sh camus,l2reg,l3reg
caffeinate -i gnn_cpu_v1/.venv/bin/python -u taxmiss_eval/run_res.py --dataset camus,l2reg,l3reg --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200
caffeinate -i bash nni_ensemble/run_derived.sh score tm-camus
caffeinate -i bash nni_ensemble/run_derived.sh score tm-reg
gnn_cpu_v1/.venv/bin/python taxmiss_eval/make_csv.py --dataset camus,l2reg,l3reg --jobs 4
caffeinate -i bash nni_ensemble/run_derived.sh rest tm-camus
caffeinate -i bash nni_ensemble/run_derived.sh rest tm-reg
caffeinate -i bash nni_ensemble/run_trainfit.sh derived tm-camus
caffeinate -i bash nni_ensemble/run_trainfit.sh derived tm-reg
STEPS=nb bash taxmiss_eval/run_all.sh
```

Notes on the step-by-step sheet:
- `run_res.py` exits **3** when some resolver failed or timed out on a cell. That is an outcome: the row keeps its
  status, and `run_all.sh` carries on. Exit 4 means a broken invariant, and that one stops the chain.
- `score` and the GNN runs can overlap `run_res.py` if you drop it to `--jobs 10`. They share nothing.
- `make_csv.py` must come **after** both the GNN and the resolver runs, and `rest` after `make_csv.py`: bench takes the
  GNN, BROOQS and NetCS rows from it.
- `run_trainfit.sh derived` (RIDDLE) must come after `rest`. It symlinks `rest`'s feats and refuses a group until
  `results/derived/bench_<stem>_gates.json` exists.
- Progress at any time: `bash nni_ensemble/run_derived.sh status tm-camus` (or `tm-reg`), and
  `gnn_cpu_v1/.venv/bin/python taxmiss_eval/run_res.py --dataset all --summary-only`.

**When more conditions land** (rsync `inputs/<ds>/<cond>/` and `runs/tob/<ds>/<cond>/` first): stage them, then the
same chain. Every step resumes, so the finished conditions cost seconds. `make_csv.py` always rebuilds a family's
whole table.

```zsh
cd ~/research/rl-net-opt
gnn_cpu_v1/.venv/bin/python taxmiss_eval/stage.py --scan --jobs 6
caffeinate -i bash taxmiss_eval/run_all.sh
```

To launch only the new ones, prefix with `TM_CONDS=p0.80_c12,p0.95_c36`. The bench summary then gets its own stem.

## NetCS (`p0.90_c24`, landed 2026-10-23)

The inputs (`inputs/netcs/`, 3.8 GB) and the TOBs (`runs/tob/netcs/`) are staged in `staged/netcs/p0.90_c24/`, with
`scan.tsv` and `t1b.tsv`. Every gate passes. Three things differ from CAMUS, L2reg and L3reg, and every NetCS number
should be read with them.

**1. OUT is re-placed on 4.7% of the trees; the ingroup is a pure deletion.**
- Linux deleted taxa from the corpus trees and placed OUT on the pruned trees. On some trees it does not sit where
  pruning the parent (`netcs_eval/inputs`, OUT by the true-root oracle on the full tree) puts it.
- A full scan of all 1,170,401 trees:
  - **0** ingroup differences. With OUT deleted, every tree is the parent's tree pruned.
  - OUT sits elsewhere on **54,446 trees (4.65%)**. By phase and ILS the rate is 2–10%, highest at n = 50 with
    ILS 0.25 (L1 9.8%, L2 10.1%) and lowest around n = 200 at ILS 0.5–1.0 (2.2–3.1%). It tracks the share of trees
    whose full-tree oracle root is tied (5–21%).
  - Most moves are among those tied, equally good edges. 4,809 trees (0.4%) land off the tie set, about 3,000 of them
    at n = 50. The largest misplacements are at n = 200, up to 90 taxa from the full-tree optimum.
- The protocol is still "the oracle roots each tree"; the condition simply roots the tree it observes. Even so, **every
  method's NetCS condition − control delta includes this re-rooting**, the GNN's and RIDDLE's too. On the other three
  families it is missing taxa alone.
- Accepted as delivered (2026-10-23): the TOBs were built on these exact files, so OUT cannot move here without
  breaking their provenance. T1b runs in ingroup mode for netcs, and `t1b.tsv` has the sampled per-cell counts
  (5,571 of 119,775 trees).

**2. The resolvers are cross-protocol against the control.**
- The published NetCS control ran BROOQS and NetCS **unrooted, with no OUT** (`netcs_eval/run_resolvers.py`). The
  conditions run them rooted on OUT, labelled `BROOQS*` / `NetCS*`, as the 500-tree subsample did.
- The notebook draws those condition − control deltas hollow (`protocol` = `cross`). Within one condition, RIDDLE
  against `BROOQS*` / `NetCS*` is protocol-matched.
- There is no control TOB error (`ctrl_tob_*` is empty, `control_tobs` 0). The landed NetCS TOB is unrooted and has
  no OUT, so it is a different object (`subsample_eval/run_res.full_tob_splits`).
- `NetCS*` is often `out_not_rootable`: 315 of 1,200 cells in the 500-tree subsample, and both smoke cells here. That
  is a recorded outcome, not a failure.

**3. A cell holds 911–1,000 loci, so the GNN runs with `--m-from-trees`.**
- `run_gnn.sh` lacked the flag, which every published NetCS launcher passes. The first smoke ran at m = 1000 on
  files of fewer than 1,000 trees, and **score parity failed on all 28 cells**: `cell.m` everywhere, and the decoded
  network on some.
- With the flag, m = n_loci on every cell and parity is 28/28. The flag is a no-op on camus/l2reg/l3reg: all 140
  landed cells have m = n_loci = 1000.
- The m = 1000 cells were deleted. Resume checks only `(--m, --tau-frac)`, so it would have reused them silently.
  `make_csv.py`'s `m == n_loci` gate would have caught it too, but only after the whole GNN run.

**Gates (2026-10-23).**
- `stage.py` T1–T6 pass. T1 checks the corpus → `stage.json` → parent link for all 1,200 cells, and T2 checks every
  tree.
- `run_res.py --check` S1–S4 pass: 300 truths, A63 kept/same 871/329.
- `check.py` P1–P4 pass on 25 cells, the lowest rep of each (phase, arm) plus the tail.
  - P2 passes on all 13 that have a GNN cell.
  - P3 shows the **dense partial kernel equal to the block one and the slow reference**, with absent-c non-zero
    everywhere. That closes the one kernel question NetCS had.
- In the smokes: score parity is 28/28, `features --check-tc` 28/28, `credible` reports 0 problems, and `make_csv`
  (`--allow-partial`, tables deleted afterwards) reports 0 problems with 90 luay measured and 0 failed.

**Measured (smokes on the tail and the lowest reps, 2026-10-23).**
- **|C| at τ = 2, partial count:** median 0.85× the published, at most 0.91×. The largest is 10,175
  (`netcs_L1_n200_rep01_netcs_ils0.5`).
- **GNN:** n = 200 cells take ~18 s each (max 24 s), n = 50 cells ~4 s. One n = 200 worker peaked at **4.6 GB**,
  not the 2.35 GB recorded for the published run. `MAXJ=6` still fits, since one `run_gnn.sh` process per phase means
  two n = 200 workers at a time.
- **score:** ~17 s per n = 200 cell, peaking at 4.6 GB.
- **MAP tail:** one smoke cell exceeded the published tail: `netcs_L1_n200_rep01_netcs_ils2.0`, with a 1,631-option
  block (the limit was 1,491).
  - Measured alone, `features` took 15 s / 0.36 GB and `credible` 13 s / 0.08 GB.
  - `derived.TAIL` was raised to it.
  - `features` peaked at 0.53 GB over the 28 cells.
  - **The full run's `rest` then stopped at the gate on 7 more cells** (all n = 200, ILS 2.0; up to 2,021 options and
    2,042,220 radius sets). Block options exist only after `score`, so the smoke could see only its own 28 cells.
    The largest, `netcs_L1_n200_rep48_netcs_ils2.0`, measured alone took `features` 17 s / 0.34 GB and `credible`
    22 s / 0.08 GB, with 0 problems, and `derived.TAIL` was raised to it. The other six run inside `rest`.
- **Resolvers:** 0.65–0.85× the landed CPU, ≤ 0.6 GB.
- **Disk:** ~0.13 MB per scored cell, so ~150 MB for the condition; 165 GB free.

| step | jobs | wall (est.) |
|---|---|---|
| gate | — | 4 min |
| gnn | 6 processes, one per phase, all at once | ~70 min (an n = 200 phase is 200 cells × ~18 s) |
| res | 2,396 tasks at `--jobs 12` | 2.9 h projected at ratio 1.0; ~2–2.5 h at the smoke's 0.65–0.85 |
| score | 24 jobs (phase × ILS arm), 6 at once | ~45–60 min |
| csv | the jar, 4 JVMs | ~5 min |
| rest | probe → tail gate → features → credible → make_csv → bench | ~30 min |
| trainfit | `credible --calib-pool train` → make_csv → bench (RIDDLE) | ~10 min |
| nb | — | 1 min |

**Run it (yours).** The whole chain is about 5–6 h end to end and stops at the first failure:

```zsh
cd ~/research/rl-net-opt
FAMS=netcs caffeinate -i bash taxmiss_eval/run_all.sh
```

Or one step at a time, in this order. This block was paste-tested in `zsh -f -i`, with the dry-run forms on the long
steps:

```zsh
cd ~/research/rl-net-opt
FAMS=netcs STEPS=gate bash taxmiss_eval/run_all.sh
caffeinate -i bash taxmiss_eval/run_gnn.sh netcs
caffeinate -i gnn_cpu_v1/.venv/bin/python -u taxmiss_eval/run_res.py --dataset netcs --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200
caffeinate -i bash nni_ensemble/run_derived.sh score tm-netcs
gnn_cpu_v1/.venv/bin/python taxmiss_eval/make_csv.py --dataset netcs --jobs 4
caffeinate -i bash nni_ensemble/run_derived.sh rest tm-netcs
caffeinate -i bash nni_ensemble/run_trainfit.sh derived tm-netcs
STEPS=nb bash taxmiss_eval/run_all.sh
```

- **Overlap.** `run_res.py` can run in a second terminal at `--jobs 10` beside `run_gnn.sh`.
  - Six GNN workers plus ten resolvers should peak around 24 GB. That is estimated from the smoke's 4.6 GB n = 200 and
    ≤ 0.6 GB resolver peaks; the n = 100 workers were not measured.
  - Watch the free-RAM percentage (not swap) and keep it above ~25%.
  - Don't overlap `run_res.py` with `score`, which can hold four n = 200 workers at once.
- **If `rest` stops at the tail gate,** it names the cells. Run `features.py` on one alone with
  `--phases`/`--reps`/`--arms`, as was done today for `netcs_L1_n200_rep01_netcs_ils2.0` and `…_rep48_…`. If it
  stays small, raise `derived.TAIL` to it (or run once with `FORCE_TAIL=1`). Then resume with
  `FAMS=netcs STEPS="rest trainfit nb" caffeinate -i bash taxmiss_eval/run_all.sh`.
- **Progress:** `bash nni_ensemble/run_derived.sh status tm-netcs` and
  `gnn_cpu_v1/.venv/bin/python taxmiss_eval/run_res.py --dataset netcs --summary-only`.

**Where the results land.**

| what | file |
|---|---|
| **RIDDLE (the final call)** | `nni_ensemble/results/trainfit/derived/netcs_tm_p0.90_c24_bench.csv`, rows `GNN MAP [z+tc]`, and `bench_taxmiss_netcs_summary.txt` beside it; control `nni_ensemble/results/trainfit/netcs_bench.csv` |
| **the vanilla GNN** | the `GNN` rows of the same bench tables; per network with runtimes in `taxmiss_eval/results/netcs_taxmiss.csv` (`ours/l2mix_pair/selF1 [relaxed L2]`) |
| the cross-fit MAP, for the record | `nni_ensemble/results/derived/netcs_tm_p0.90_c24_bench.csv` |
| `BROOQS*` / `NetCS*` | `taxmiss_eval/results/netcs_taxmiss.csv`, and as `BROOQS` / `NetCS` in both bench tables |

**The run (finished 2026-10-23).** Every gate passed with 0 problems: 1,200 cells, B1 1,200/1,200/1,200, luay parity
4,800. `rest` stopped once at the tail gate (see *Measured* above) and resumed. Train-fit credible had 0 warnings.
- **Resolver outcomes:** `BROOQS*` ran OK on all 1,200 cells. `NetCS*` ran OK on 838; 309 were `out_not_rootable` and
  53 were tool errors (47 `rc=-11`, plus 6 output networks with non-taxon or duplicate leaf labels). These are the
  tool-error classes subsample_eval already records as outcomes.
- **Cluster F1, mean of paired condition − control** (n 1,183 GNN cells): GNN −0.034, RIDDLE −0.033, `BROOQS*`
  −0.002 (cross), `NetCS*` −0.029 (cross). On CAMUS (paired) the GNN lost −0.017 and BROOQS −0.003.
  - The GNN's loss grows with ILS: −0.022 / −0.027 / −0.028 / −0.060 at ILS 0.25 / 0.5 / 1 / 2.
  - It is flat in n: −0.026 to −0.040 at every (level, n).
- **RIDDLE against the others, condition (control in brackets):**

  | vs | cluster F1 | luay |
  |---|---|---|
  | GNN | +0.009, 721 better / 121 worse (+0.008) | −0.012 (−0.012) |
  | `BROOQS*` | −0.037, 349 / 842 (−0.005) | +0.019 (+0.002) |
  | `NetCS*` | −0.011, 483 / 658 (−0.006) | +0.016 (+0.009) |

  - The MAP's gain over the GNN survives missing taxa intact. RIDDLE's gap to the resolvers widens.
  - The control column compares RIDDLE with the unrooted resolvers, and the condition column with the rooted ones.
    So that widening is not a paired measurement.
- **The re-rooting does not explain the GNN's loss.** The full OUT-move scan was joined to the paired deltas, with
  both demeaned within phase × ILS. The scan is `results/netcs_p0.90_c24_out_moves.tsv`, written by
  `netcs_out_moves.py` with per-cell moved and off-tie (`moved_not_in_tied`) counts.
  - Cells with more re-placed trees did not lose more. The GNN's F1 slope is +0.04 per unit moved fraction
    (Spearman +0.06, p 0.05), the opposite sign to harm, and worth +0.002 at the mean rate.
  - On the off-tie fraction the GNN shows nothing (p 0.72). `BROOQS*` shows nothing on either.
  - **`NetCS*`** shows a small positive slope (+0.16 per unit, p 0.004), worth about +0.008 at the mean rate.
  - **Caveat:** the moved fraction is set per cell by Linux, not assigned at random, so this is an association.
    Still, a re-rooting harm worth the GNN's −0.034 would have shown up as a negative slope.

## Traps

- **zsh:** an unquoted `$VAR` holding flags is one argument, and a `#` line at the prompt is a command. The blocks
  above have no comments and spell every flag out.
- **`FILTER` with `--reps` on `tm-reg`** (a smoke idiom of `run_derived.sh`) replaces both rep-half jobs' `--reps`.
  Two processes then score the same cell and one dies on the temp file. The cell survives, but the launch fails.
  Smoke on `tm-camus`, which splits by phase.
- **Never run two `make_csv.py`s at once**, ours or `nni_ensemble`'s. Each appends to its jar cache.
- **CAMUS's headline arm is `cam_g500`.** cam25 is `cam_iq500`: it is run, scored and in `<ds>_taxmiss.csv`, but bench
  and the notebook leave it out, as the ztc bench does.
- **The MAP's calibration is fitted on full-taxon trees.** RIDDLE's beta comes from the GNN's own training networks;
  the cross-fit one comes from the parent's held-out fold. `z` and `tc` from partial trees are out of distribution, and
  so is the GNN. That is the point of the benchmark, so nothing is refitted.
- **The MAP's add-on time is not measured** (`time_addon.py` has no partial path). The runtime figure shows the GNN
  and the resolvers only.
