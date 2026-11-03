# subsample_eval: the GNN on 500 of each cell's gene trees

Every landed benchmark row uses a cell's whole gene-tree file: 1,000 trees on camus, l2reg and
l3reg, and 911–1,000 on netcs. This package re-runs our method on **a random 500 of them**, to
answer two questions:
- how far does the method fall when it has half the data?
- how much does its output move from one draw to the next?

| | cells | draws per cell | cell-runs per decoder |
|---|---:|---:|---:|
| camus (`gnn_cpu_v1/CELLS.tsv`) | 100 | 5 | 500 |
| l2reg | 20 | 5 | 100 |
| l3reg | 20 | 5 | 100 |
| netcs | 1,200 | 1 | 1,200 |

**What runs** (decided 2026-09-18):
- **Model:** `_l2mix_pair/s0` only, under both selections: `best.pt` (selF1) and `best_ha.pt`
  (selHA).
- **Decoders:** `level1`, plus `relaxed` at the dataset's level: L2 on camus, l2reg and netcs, L3 on
  l3reg.
- **Baseline:** greedy-by-support on the same subsamples, both decoders. It needs no tree of blobs.
- **Left out:** the TOB resolvers (NetCS, BROOQS). Their trees of blobs would have to be
  re-estimated on each subsample. The subsample files are staged on disk, so a later round can
  build TOBs from exactly the same trees. *(That round is now staged: see "Resolvers on the
  subsample TOBs" at the end.)*

**Nothing outside this directory changed.** The runs drive the existing `gnn_cpu_v1/run_gnn_cpu.py`
and `netcs_eval/run_greedy.py` through their `--inputs/--cells/--out` flags. No module, checkpoint
or `SHA256SUMS` is touched.

| file | what it does |
|---|---|
| `common.py` | Pins BLAS, puts `gnn_cpu_v1/pylib` on the path, and holds the dataset registry: source inputs, `CELLS.tsv`, relaxed level, draws, landed CSV, landed full-data runs. Also defines the seeded draw. |
| `stage.py` | Builds `inputs/<ds>/d<k>/`, `MANIFEST.tsv`, `indices.json` and `stage.json` behind gates G1–G7. `--check` re-derives and compares. |
| `run_sub.sh` | The GNN and greedy runs, one background process per draw (netcs: per shard). |
| `make_sub_csv.py` | One CSV per dataset: the subsample rows and the full-data rows, re-scored alike. Includes the parity gate against the landed CSVs and a summary. |
| `run_ens.py` | The ensembles over a cell's five draws, in two phases: `score` re-runs every draw's forward and caches it, and `decode` combines the draws. See "Ensembles over the draws" below. |
| `make_ens_csv.py` | The ensemble rows, scored by `make_sub_csv.py`'s own functions and read against `results/<ds>_sub500.csv`. |
| `check_ens.py` | Ties `gnn_cpu_v1/infer_gnn_ens.py`, the real-data driver, to `infer_gnn_cpu.py` and to `run_ens.py`. |

## Design

**The subsample is a staged file, not a new flag.** Each cell's *staged* gene-tree file is read
(the one the landed runs read), 500 of its non-blank lines are drawn, and they are written
**unchanged and in source order**. Where they go:
- the path is `inputs/<ds>/d<k>/<phase-dir>/<rep>/`, under the arm's own file name (`g_500.nwk`,
  `iqtree_500.nwk`, `est_bp1000.nwk`, `est_ils<x>.nwk`);
- the truth files beside it are copied byte for byte;
- phase and arm names do not change, so only the directory says that it is a subsample.

On netcs, the staged file carries `netcs_eval/stage.py`'s true-root `OUT`, which is applied line by
line. So drawing staged lines is the same as drawing corpus lines and rooting them.

**The draw:**

```
seed    = opc.grid.chain_seed("sub500", dataset, phase, rep, arm, draw)
indices = sorted(random.Random(seed).sample(list(range(n_source)), 500))
```

That is `gnn_cpu_v1/infer_gnn_cpu.py`'s `select()`, so each staged file holds exactly the trees
`infer_gnn_cpu.py --m 500 --seed <seed>` would pick from the source. Gate G7 checks this against
the function itself, not against a copy of it.
- **camus, l2reg, l3reg:** the draws of one cell share ~250 trees on average, as random halves of
  1,000 should.
- **netcs:** the draw takes 500 of each condition's own 911–1,000 trees.

**What stays the same as the full-data runs:**
- **`--m-from-trees` is on every GNN call.** m = 500, so `support/m`, `bconf_*` and the pair
  conv's `/m` inputs stay frequencies.
- **τ stays 2 as an absolute count**, as in training and on netcs. The relative bar therefore
  doubles, from 0.2% to 0.4%, and |C| shrinks to ~0.6× (column `n_clusters`). That shrinkage is
  part of what the experiment measures.
- **`k` is still `k_true`.**

## Run it, from the repo root, in a terminal

Use your terminal, not a Claude Code background task. Every step resumes per cell.

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar     # the default anyway

$PY subsample_eval/stage.py --check        # ~50 s; ALREADY STAGED AND PASSING (2026-09-18)

# 1. camus: 5 draws at once (5 processes), then the one missing full-data baseline
caffeinate -i bash subsample_eval/run_sub.sh camus                      # ~35 min
caffeinate -i bash subsample_eval/run_sub.sh camus full                 # ~6 min, foreground

# 2. l2reg and l3reg: 5 processes each; both together is fine (10 processes)
caffeinate -i bash subsample_eval/run_sub.sh l2reg &                     # ~15 min
caffeinate -i bash subsample_eval/run_sub.sh l3reg; wait                 # ~15 min

# 3. netcs: draw 0 over the 12 netcs_eval shards, IN WAVES OF 6 (memory: see below)
caffeinate -i bash subsample_eval/run_sub.sh netcs 0 1 2 3 4 5           # ~40 min
caffeinate -i bash subsample_eval/run_sub.sh netcs 6 7 8 9 10 11         # ~40 min

grep exited subsample_eval/runs/*/logs/*.log                            # must print nothing

# 4. the CSVs: seconds each, plus the JVMs for the new networks (minutes on netcs)
for DS in camus l2reg l3reg netcs; do $PY subsample_eval/make_sub_csv.py --dataset $DS --jobs 4; done
```

**Where the timings come from.** They are estimates:
- the landed pair-checkpoint runs, 0.71 / 0.26 / 0.29 / 9.3 CPU-h per decoder (camus / l2reg /
  l3reg / netcs);
- times the 500-tree ratio measured on one smoke cell per dataset: 0.23–0.31 of the full-data CPU
  time for the same cell. The pair model's cost grows faster than |C|, and |C| is ~0.6×;
- plus greedy at ~0.7× its landed time.

**Memory.** `mac-netcs-concurrency`: a netcs worker reached ~2.35 GB at n = 200 on 1,000 trees,
and 12 at once got the run killed. With |C| at ~0.6× the workers will be smaller, but waves of 6
is what is known to be safe.

**Logs.** They go to `subsample_eval/runs/<ds>/logs/{d<k>,shard_XX}.log`, one line per step. A failed
step prints `exited`, and the process moves on to its next step.

## What was run, and what was not (2026-09-18)

- **`stage.py`, every dataset: PASS on G1–G7**, then `stage.py --check`: PASS.
  - 1,900 subsample files were written, 3.6 GB.
  - `infer_gnn_cpu.select` agrees with the staged draw on every G7 cell: 41 of 41, plus 6 more
    checked by hand.
- **The smoke test**, into the session scratchpad: `run_sub.sh` with a one-cell `CELLS` file per
  dataset, every step, draw 0.
  - The cells were camus cam50 rep 0, l2reg rep 0, l3reg rep 0, and netcs L2_n50 net 1 at ILS 1.0.
  - Zero `exited`.
  - Every cell JSON records `m = 500`, `n_loci = 500`, and |C| at 0.57–0.61 of the landed cell.
  - Both models and greedy produced networks, with `k_pred = k`.
- **`make_sub_csv.py` on the smoke:** clean, 0 problems, parity 41 of 41.
- **`make_sub_csv.py --draws full`, every dataset: parity 7,940 of 7,940 full-data rows.** l2reg
  120, l3reg 120, camus 500, netcs 7,200. Every compared field reproduces the landed CSV. This wrote
  only to the scratchpad; its luay measurements are kept in `runs/phylonet/cache.jsonl`.
- **Not run** *(at the time of writing)*: everything under "Run it" except the staging. Those runs
  have since been made -- `runs/<ds>/d<k>/` and `results/*_sub500.*` are them.

## Gates

| gate | where | checks |
|---|---|---|
| G1 | `stage.py` | The source file is the one its benchmark recorded (`gnn_cpu_v1/SHA256SUMS`, or `stage.json`'s SHA-256), and its non-blank line count is `CELLS.tsv`'s `n_loci`. |
| G2 | `stage.py` | Each written file, re-read, is exactly the drawn lines in order, 500 of them. |
| G3 | `stage.py` | Each truth copy is byte-identical to its source. |
| G4 | `stage.py` | Vacuity: a cell's draws are pairwise distinct, and none is the first-500 prefix (what the drivers' `max_loci` cut would take). |
| G5 | `stage.py` | `camus.prepare_gene_trees`, the drivers' first stage, reads one cell per (phase, draw) back with `n_loci = 500` and the truth's exact leaf set. |
| G6 | `stage.py` | `NO_REFERENCE.tsv` does not exist (`run_sub.sh` checks too). |
| G7 | `stage.py` | `infer_gnn_cpu.scan` + `select(m=500, seed)` picks exactly the staged indices. |
| integrity | `make_sub_csv.py` | Every subsample cell has `m = n_loci = 500`, and its staged file still hashes to `MANIFEST.tsv`. Every cell has the expected decoder, `k` and tree file. The hybrid-ancestry derivations agree. `luay_norm_exact ≤ 1`. |
| parity | `make_sub_csv.py` | Every `full` row reproduces the landed CSV field by field, as printed. `ALLOWED` has one entry, camus cam150 rep04 greedy `luay_norm_exact`: the Linux base file sized that size-2-cycle truth differently (0.349693 there, 0.360759 from the jar here). |

## Reading the output

`results/<ds>_sub500.csv` is long-format, one row per (draw, cell, method). The subsample rows and
the full-data rows sit together, with `draw = full` or `0`–`4`.

**Labels.** `method` uses the landed labels: `ours/l2mix_pair/selF1 [relaxed L2]`, `greedy`, and so
on. To plot it with `plotting.ipynb`'s `panel_frame`, filter to one draw or aggregate over draws
first, because that function refuses a duplicate (key, method).

**Scoring.** The scoring is the landed definitions:
- F1 is the driver's own count on the selection.
- luay and the masses come from the PhyloNet jar only.
- netcs is scored against the corpus truth string.
- Hybrid ancestry is derived two ways, and the two must agree.

`n_clusters` is the candidate-set size, and `prep_wall` / `model_wall` are the per-cell costs.

**`results/<ds>_sub500_summary.txt`** gives the following per method, for `cluster_f1` (higher is
better) and for `luay_norm_exact` and `ha_balanced_error` (lower is better):
- the full-data median;
- the pooled subsample median, and each draw's median;
- **delta**: per cell, the mean over its draws minus its full value, reported as median and mean;
- the fractions of cells that got worse and that got better;
- **draw SD**: the mean within-cell SD across draws. This is the draw-to-draw variation, and is
  empty on netcs, which has one draw;
- then the |C| ratio, whether `k_pred = k`, and delta F1 by stratum: camus phase, l2reg/l3reg `cu`,
  netcs level × ILS.

On camus, the main table is the headline 80 `cam_g500` cells; cam25 appears only in its stratum
row.

**`results/<ds>_sub500.json`** has the same, plus the run directories read, every problem, and the
parity counts.

## Traps

- **`--m-from-trees` must be on every GNN call.** Without it, m stays 1,000 and every
  m-normalised input is halved, silently. `run_sub.sh` always passes it, and `make_sub_csv.py`
  refuses a subsample cell whose `m` is not 500.
- **Keep subsample runs out of the landed run directories.** The drivers' resume does not look at
  `m`, the tree file or `n_loci`. A subsample run pointed at a landed run directory would skip every
  cell as done, and one pointed at an empty directory beside the landed runs would be globbed into
  that benchmark's `make_csv.py`. Everything here writes under `subsample_eval/runs/`.
- **`NO_REFERENCE.tsv` must not exist.** camus's default `--reference` would compare 500-tree rows
  with the 1,000-tree `reference.tsv` and exit 3.
- **The staged inputs are re-derivable bit for bit.** Delete `inputs/` to free the 3.6 GB, and
  `stage.py` rebuilds identical files (the manifest SHAs pin them). Only ~30 GB is free on this
  disk.

## Upstream

No existing module changed. Any defect found in these runs needs an amendment number from the
Linux box's register; `SPEC-AMENDMENTS.md` here is a stale prefix. Record it in this README until
then.

---

# Ensembles over the draws (2026-09-19)

Subsampling costs more in luay and hybrid ancestry than in F1 (above). This round asks whether a
cell's **five draws, combined**, can be as good as, or better than, the ordinary 1,000-tree decode.
If they can, the same code is a real-data method. That is `gnn_cpu_v1/infer_gnn_ens.py`, whose
guide is `gnn_cpu_v1/infer_gnn_ens.md`.

## Design

The core is `gnn_cpu_v1/ensemble.py`, which the benchmark and the real-data driver share.

**The pool.** Each draw's τ = 2 candidate set is merged on the cluster words into a union U. Rows
are kept in first-seen order over draws 0..4. That order is the decoders' tie-break, so with one
draw the union *is* that draw's candidate set, row for row. Every draw is a subset of the same
1,000 trees, so U ⊆ C_full.

**Three aggregators** turn the pool into one ranking, and one decode runs on it (same decoder, same
budget `k_true`):
- **`pool`**: a cluster's mean score over the five draws, counting 0 in a draw where it is below
  τ.
- **`pool_present`** (ablation): the same mean over only the draws that hold the cluster.
- **`vote`**: v(c) is the number of draws whose own decode selected c. The majority (2v > 5) goes
  first, most votes first, and the rest follow in `pool` order. The decoder then re-extends the
  majority to a maximal system. Any majority clusters it has to skip are counted
  (`majority_refused`).
- **greedy-vote control**: `vote` over greedy-by-support's own decodes, re-extended by summed
  support. It tells whether a gain is the GNN's or just the ensembling's.

**Two facts make this sound.**
- Both decoders read only the **rank order** of the scores: a stable `argsort`, ties to the lower
  row, no threshold. They never read `support` or `m` once scores are given.
- The lexicographic vote key is therefore passed as **exact ranks** (`ensemble.rank_scores`).
  Packing it into one float would collapse tiny `pool` values into row order.

**Votes count the selection, not the realised network.** Level1's hybrid clusters come from the
closure and are usually not candidates, so they could not be re-accepted anyway.

## Files and outputs

| phase | writes |
|---|---|
| `run_ens.py score` | `runs/<ds>/ens/scores/<cell>.npz`: each draw's words, support, logits, `p` and selections. Beside it, `<cell>.json`: timings, per-draw counts and parity. |
| `run_ens.py decode` | `runs/<ds>/ens/<agg>/{gnn_t1_l2mix_pair,greedy}{,_relaxedL<L>}/cells/<cell>.json`, in `run_gnn_cpu`'s format plus an `ensemble` block. Also `runs/<ds>/ens/decoded/<cell>.json` (gates, sweep, order check) and `runs/<ds>/ens/decode_summary.json`. |
| `make_ens_csv.py` | `results/<ds>_ens.csv`, `_ens_summary.txt`, `_ens.json` and `_ens_sweep.tsv`. |

netcs is out: it has one draw.

## Gates

| gate | where | checks |
|---|---|---|
| P1 / P1g | `run_ens.py score` (exit 3) | Every replayed draw reproduces its landed cell under `runs/<ds>/d<k>/`: `n_clusters, m, n_loci` (and `n_edges` for the GNN), and per model `n_selected, tp, fp, fn, fn_reachable, k_pred, enewick`, plus `decode_stats` when relaxed. That is 20 GNN and 10 greedy comparisons per cell. The forward's `p` is also asserted bit-equal to `run_gnn_cpu.forward`'s on each worker's first draw. |
| G-identity | `run_ens.py decode` (exit 1) | With one draw, every aggregator reproduces that draw's own selection, as a set of cluster words. This is checked for each of the 5 draws, both decoders, both checkpoints and greedy: 70 per cell. |
| G-F1 | `run_ens.py decode` (exit 1) | The ensemble's tp/fp/fn from candidate rows equal `camus_nets.cluster_counts` on label sets. |
| sweep | `run_ens.py decode` | The D = 1 row is draw 0, and the D = 5 row is the main decode. |
| reported | `run_ens.py decode` | Vacuity: the union is larger than every draw, the ensemble is unlike every draw, and the network equals the full-data one. Order sensitivity: `pool` with the draws reversed. |
| R1, R1b, R2 | `check_ens.py` (exit 1) | The real-data driver against `infer_gnn_cpu.py` (byte-identical at `--draws 1`, and draw b = `--seed S+b`) and against `run_ens.py`, end to end. See `gnn_cpu_v1/infer_gnn_ens.md` §7. |

## Run it, from the repo root, in a terminal

Every step resumes per cell. Flags are written out on every line (zsh does not word-split `$VAR`).

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar     # the default anyway

# 0. gates on the real-data driver -- ALREADY RUN AND PASSING (2026-09-19)
$PY subsample_eval/check_ens.py --only r1,r1b                            # ~1 min

# 1. score: every draw's forward again, gated against the landed d0-d4 cells (P1)
caffeinate -i $PY -u subsample_eval/run_ens.py score --dataset camus --jobs 5      # ~20-25 min
caffeinate -i $PY -u subsample_eval/run_ens.py score --dataset l2reg --jobs 5 &    # ~8-10 min
caffeinate -i $PY -u subsample_eval/run_ens.py score --dataset l3reg --jobs 5; wait

# 2. decode: the ensembles, G-identity, G-F1, the sweep. No model, minutes
caffeinate -i $PY -u subsample_eval/run_ens.py decode --dataset camus --jobs 5     # ~10-15 min
caffeinate -i $PY -u subsample_eval/run_ens.py decode --dataset l2reg --jobs 5     # ~5 min
caffeinate -i $PY -u subsample_eval/run_ens.py decode --dataset l3reg --jobs 5     # ~5 min

# 3. R2 against the real run_ens cells (needs step 2 on camus)
$PY subsample_eval/check_ens.py --only r2                                # ~5 min

# 4. the CSVs -- never beside make_sub_csv.py: both append to runs/phylonet/cache.jsonl
for DS in camus l2reg l3reg; do $PY subsample_eval/make_ens_csv.py --dataset $DS --jobs 4; done
```

**Where the timings come from.** They are measured per cell on this Mac in the smokes, single
process:

| cell | score | decode |
|---|---:|---:|
| cam25 | 4 s | 1.5 s |
| cam50 | ~35 s | ~15 s |
| cam100 | ~45 s | ~24 s |
| cam150 | 92 s | 52 s |
| cam200 | 76 s | 50 s |
| l2reg n = 200 | 90 s | 55 s |

The table's figures are then divided by 5 workers, with some allowance for contention.

**Memory.**
- Score workers are the landed subsample runs' size: |C| ≈ 0.6 × full.
- Decode workers hold U, which is ≈ 0.9 × |C_full|. That is below a full-data run, whose netcs
  n = 200 workers measured ~2.35 GB.
- Five workers per dataset, or ten for l2reg and l3reg together, is safe.

**`--dry-run`** lists the cells still to do. `--summary-only` re-tallies what is on disk.
`run_ens.py decode --summary-only` prints a jar-free F1 table: each ensemble against the mean of
its own draws.

## What was run (2026-09-19)

Everything under "Run it" was run, in that order, on this Mac at `--jobs 5`, one dataset at a time.
47 minutes end to end: score 15 / 6 / 6 min (camus / l2reg / l3reg), decode 8 / 3 / 4 min, R2 4 min,
the three CSVs under a minute each. Every step exited 0.

**The gates.**
- **P1/P1g: 4,200 of 4,200 replays reproduce their landed cells** -- 3,000 camus, 600 l2reg, 600
  l3reg -- over both checkpoints, greedy and both decoders.
- **G-identity: 9,800 of 9,800.** **G-F1** and the sweep checks: 0 problems on all 140 cells.
- **The union is larger than every draw in 140 of 140 cells**, and `pool` was order-insensitive in
  every one.
- **`check_ens.py`:** R1 2/2, R1b 5/5 (before the run), R2 24/24 against the real `runs/` cells.
- **The CSVs are clean:** 0 problems, 0 gaps, 1,960 ensemble rows. Both hybrid-ancestry derivations
  agree on 1,960 of 1,960 networks; 3,030 luay commands measured, 0 failures.

**The result: ensembling recovers most of what subsampling costs, and lands at or just below the
full-data decode.** Medians over the main cells, for `pool` (the three aggregators are within
~0.002 of one another):

| | camus selF1 | camus selHA | l2reg selF1 | l3reg selF1 |
|---|---|---|---|---|
| cluster F1: draws -> ens (full) | 0.941 -> 0.951 (0.954) | 0.936 -> 0.952 (0.950) | 0.930 -> 0.933 (0.938) | 0.909 -> 0.920 (0.917) |
| ens - full, median (p) | -0.005 (0.004) | -0.000 (0.21) | -0.002 (0.10) | +0.001 (0.52) |
| ens - draws, cells better | 67 of 80 | 70 of 80 | 17 of 20 | 20 of 20 |

- **ens - draws is positive almost everywhere**: +0.011 to +0.014 F1 on camus (67-72 of 80 cells),
  +0.008 to +0.013 on l2reg and l3reg.
- **ens - full is small and mostly not significant.** It is significantly *below* full for camus
  selF1 (-0.005, p = 0.001-0.007) and for l2reg; for camus selHA, l3reg and every luay column it
  is not distinguishable from the full-data decode.
- **luay recovers nearly all of the loss**: camus selF1 0.274 (draws) -> 0.241 (ens) against 0.228
  (full), and ens - full is not significant (p = 0.33-0.45).
- **Hybrid ancestry, where subsampling hurt most, recovers about half.** l2reg selHA relaxed L2:
  0.049 full, 0.272 draws, 0.119 ens. It is still worse than full (p = 0.06).
- **The gain is the GNN's, not just ensembling's.** The greedy-vote control gains +0.005 F1 on
  camus against the GNN's +0.011 to +0.014, and stays flat on hybrid ancestry.
- **The aggregators are a wash.** `pool`, `pool_present` and `vote` differ by less than the spread
  between checkpoints; `vote` refuses almost no majority clusters (0.00-0.11 per cell), so the
  majority set is nearly always jointly feasible.
- **The ensemble is a genuinely new answer**, not one of its draws: it differs from every draw in
  69-75 of the 80 camus cells, and equals the full-data network in only 1-7.
- **The sweep saturates at 3-4 draws.** camus selF1 `pool`: 0.939 (D=1), 0.946, 0.950, 0.952,
  0.951 (D=5). The fifth draw adds nothing, which matters for what real-data runs should cost.

## Reading the output

- **ens − full is the headline**, but mind what it compares. Five random halves of 1,000 trees
  cover ~97% of them, so it sets two ways of **combining** nearly the same data against each other:
  - one decode on all of it;
  - five half-data decodes merged.

  It does not compare two amounts of data. **ens − draws** mixes more data with the combining, and
  the **draw-count sweep** separates the two.
- **The sweep's vote curve is uneven**, because its majority means 2 of 2, 2 of 3, 3 of 4 and 3 of 5.
- **`unlike draws`** counts cells whose ensemble selection matches no single draw's. **`net = full`**
  counts networks whose cluster set equals the 1,000-tree network's.
- **camus's main table is the 80 `cam_g500` cells**, as in `_sub500_summary.txt`.

## Traps

- **Keep every ensemble run under `runs/<ds>/ens/`.** The `d<k>/` and landed run directories are
  gate P1's reference and must not be written.
- **The score cache is the only record of the per-candidate scores.** The landed cell JSONs keep
  counts and the eNewick only.
- **Selections are compared as sets of cluster words, never as row indices or eNewick strings.**
  Row indices differ between draws and the union, and the eNewick depends on row order.
- **A new aggregator needs its own `run_ens.py` key.** A `decode` resumed with `--force` rewrites
  every aggregator's cells, and one without it skips cells whose `decoded/<cell>.json` exists.

**Follow-up (2026-09-22): the cluster anatomy of these ensembles, and FP flags from them, are in
`ens_anatomy/README.md`.** In short:
- The draws do not flag FPs better than the full-data `z < 6`.
- A learned full + draws flag adds about 3 pp of catch on CAMUS only.
- The FPs that `z < 6` misses are present in all five draws.
- Re-ranking by a learned P(true) does not beat the full-data decode.

---

# Resolvers on the subsample TOBs (2026-09-22)

TREE-QMC trees of blobs (TOBs) now exist for every staged subsample: 1,900 of them, in
`runs/tob/<ds>/d<k>/<cell>.{json,tob,raw.tob,pvalues.tob}`.
- They are TREE-QMC 4.1.5 `--blob`, built on the Linux box by `subsample_eval/tob.py all --jobs 96`.
  That driver is not in this checkout, and neither are the `results/sub500_tob*` files it wrote.
- `runs/tob/partial.*` is a stale mid-run snapshot (842 of 1,900). The final per-stratum TOB table
  exists only at the end of `runs/tob/run.log`, which finishes `1900 of 1900 cells scored ... PASS`.

This round runs NetCS and BROOQS on those TOBs. It scores the networks exactly as the landed
resolver rows were scored, so they sit beside `results/<ds>_sub500.csv` and the landed full-data
rows.

## Design

**Decided with the user (2026-09-22):**
- **netcs is run rooted, the way its TOBs were built.**
  - Its staged gene trees carry the true-root OUT, an oracle, and TREE-QMC was run on them.
  - So the netcs rows use the camus/l2reg/l3reg protocol and are labelled `BROOQS*` / `NetCS*`.
  - They get the same root information the GNN's netcs subsample rows had.
  - They do **not** follow the landed netcs resolver protocol: that ran unrooted, with no OUT
    anywhere and BROOQS without flags. Every netcs sub - full delta therefore crosses protocols,
    and the summary marks each one `X-PROTOCOL`.
- **Two TOB variants.**
  - The raw TOB (TREE-QMC's output as written) runs on all four corpora.
  - The A63-fixed TOB (the contracted root branch removed) runs on l2reg and l3reg only.
  - That is exactly the landed labels: `BROOQS`, `NetCS`, `BROOQS (fixed TOB)`,
    `NetCS (fixed TOB)`.

**The commands are the landed ones.** Every corpus uses the camus / l2reg / l3reg protocol:

| variant | `tob.nwk` fed to the resolver |
|---|---|
| raw | `camus_nets.root_tob_on_outgroup(<cell>.raw.tob)` (`l2reg_eval/stage.py:183`) |
| fixed | `(OUT,<cell>.tob);` (`l2reg_eval/stage.py:184`) |

    netcs    tree-qmc --network -i <staged gene trees> --at tob.nwk -o out.enewick
    brooqs   brooqs -i <staged gene trees> -t tob.nwk -o out.enewick --rooted --dist-var

- `--rooted` means the TOB is correctly rooted; it says nothing about the gene trees.
- `--dist-var` draws its representatives from non-root blobs, so netcs's zero-length oracle OUT
  never enters it.
- BROOQS is seeded (1142), which is why byte-identical parity means something.
- There are 4,200 tasks: (1,900 raw + 200 fixed) × 2 resolvers.

**Scoring** is `run_blob_cpu.score` against C(N_true), with taxon set truth + OUT, on every corpus.
- This is the landed camus/l2reg/l3reg scoring.
- `pte_bench.score` is not used: its netcs branch inserts OUT through the oracle, and these outputs
  already carry one.
- luay, the masses and hybrid ancestry are `make_sub_csv.py`'s, with the landed resolver semantics.
  A network with OUT below one of its own hybrids gets `status = out_not_rootable`, its as-emitted
  cluster F1, no luay, and hybrid ancestry read off the network as emitted.

**Borrowed, not copied.** The runner, content ids, tool fingerprint and eligibility rules come from
`prefix_tob_eval/pte_common.py`; the truths, landed-module loading and invariants come from
`pte_bench.py`.
- Their `pte_` names cannot shadow this package's `common`.
- `run_res.py`'s `run_task` is its own, modelled on `prefix_tob_eval/run_resolvers.py`.
- Nothing outside `subsample_eval/` changed.

| file | what it does |
|---|---|
| `run_res.py` | One task per (draw, cell, variant, resolver), keyed by content id, with a timeout, LPT ordering and invariants on every output. Modes: `--check` (gates S1-S5), `--parity`, `--smoke`, `--dry-run`, `--summary-only`, `--rescore`. |
| `make_res_csv.py` | `results/<ds>_sub500_res.{csv,json,_summary.txt}`: the subsample rows plus the landed full-data resolver rows, re-scored by the same code, with a parity gate against the landed CSVs and a summary. |

## Gates, all exit non-zero on failure. Status 2026-09-22

| gate | checks | status |
|---|---|---|
| S1 | Every MANIFEST row has its TOB files. The JSON's `input` equals the MANIFEST path, and `sha256_input` equals `sha256_sub`, which equals the staged file's hash now. `n_trees = m = 500`. | **PASS** 1,900 / 1,900 |
| S2 | Raw TOB leaves = first and last gene tree = truth + OUT; fixed TOB leaves = truth. Both wrapped trees are `(OUT,<body>)`, and wrapping moves no split. | **PASS** 1,900 |
| S3 | A63: the fixed splits equal the raw splits, minus exactly the contracted root-branch split, precisely where the JSON says TREE-QMC kept one. | **PASS**. Kept: camus 231, l2reg 56, l3reg 52, netcs 856, matching the JSONs |
| S4 | Every cell maps 1:1 to a landed condition. The staged truth equals `pte_bench.truth` (the object the landed rows were scored on) by label cluster, taxa and k. | **PASS** 440 truths |
| S5 | Vacuity, reported only. | camus: 41 of 500 draws' raw TOBs equal the full-data TOB by splits (Mac `blob_cpu_v1` TOB); 3 of 100 cells have five identical draws. l2reg and l3reg: 0 of 100 and 0 of 20. netcs: no full-data TOB under this protocol. |
| `--parity` | Landed full-data TOBs through this driver on landed gene trees: camus cam25 rep01 and cam100 rep09; l2reg and l3reg rep 0, raw and fixed. | **PASS**. Wrapping 6/6 byte-identical to the landed `tob.nwk`. Outputs 12/12: BROOQS eNewick byte-identical with the same score; NetCS the same k and score, and the one l3reg raw NetCS is `out_not_rootable` on both sides. |
| full-row CSV parity | `make_res_csv.py --draws full`: the landed resolver outputs re-scored, field by field against the landed CSVs. | **PASS**: l2reg 80/80, l3reg 80/80, camus 200/200, netcs 2,400/2,400. One allowance, detailed below. |
| smoke | `run_res.py --smoke` into the scratchpad: 8 cells, 16 tasks, then `make_res_csv.py` on them. | 15 ok, 1 known NetCS segfault, 0 invariant violations. The CSVs were clean on all four corpora. |

**The camus full-row allowance.** It is `make_sub_csv.py`'s cam150 rep04 entry, on both resolvers.
- luay is equal: 20.
- `luay_norm_exact` is 0.126582 here against the landed 0.122699, because the jar sizes this
  size-2-cycle truth differently from the Linux base file. It is listed in `ALLOWED`.

**The smoke's one failure** is NetCS `rc=-11` on l3reg d0 rep00 raw. It segfaulted right after
"Perform rerooting on the edge above taxon t176", the known NetCS failure seen in the landed l3reg
runs. Tool errors are outcomes: recorded, never retried.

## Command sheet, from the repo root, in a terminal

Each line below was dry-run as written on 2026-09-22; flags are inline because zsh does not
word-split `$VAR`. Every step resumes per task. A timeout re-runs only when a longer timeout is
asked for.

```bash
PY=gnn_cpu_v1/.venv/bin/python

# 0. ALREADY RUN AND PASSING (2026-09-22)
$PY subsample_eval/run_res.py --check                                  # ~95 s
$PY subsample_eval/run_res.py --parity --jobs 6                        # ~3 min

# 1. the small corpora: 1,800 tasks, ~21 CPU-h, ~1.8 h at --jobs 12
caffeinate -i $PY -u subsample_eval/run_res.py --dataset camus,l2reg,l3reg --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200

# 2. netcs: 2,400 tasks, ~25 CPU-h, ~2.0 h at --jobs 12
caffeinate -i $PY -u subsample_eval/run_res.py --dataset netcs --jobs 12 --timeout-brooqs 7200 --timeout-netcs 7200

# 3. tally: statuses, k against eligible polytomies, hard invariant violations (must be 0)
$PY subsample_eval/run_res.py --dataset all --summary-only

# 4. the CSVs: seconds to a minute each. Never beside make_sub_csv / make_ens_csv (shared luay cache)
for DS in camus l2reg l3reg netcs; do $PY subsample_eval/make_res_csv.py --dataset $DS --jobs 4; done
```

**Where the timings come from.** Each task is projected at the landed 1,000-tree CPU of the same
condition and resolver, times `run_res.RATIO` (BROOQS 0.65, NetCS 0.8).
- The ratios come from the smoke, which measured 500-tree / 1,000-tree CPU of 0.43-0.77 for
  BROOQS and 0.39-1.76 for NetCS. NetCS is the noisy one.
- The longest single task is projected at ~8 min.
- Each run gates S1-S4 before it starts (~95 s; `--no-check` skips them).

**Memory.** In the smoke, BROOQS peaked at 48-91 MB and NetCS at 350-554 MB (n = 200), so
`--jobs 12` is under ~7 GB. That is nothing like the GNN netcs runs, which needed waves of 6.

## Outputs and how to read them

- **`runs/res/<ds>/`**
  - `tasks/<task_id>.json` holds the command, timing, tool note, status, score, invariants and
    eNewick.
  - `work/d<k>/<cell>/<variant>.<resolver>/` holds `tob.nwk`, `out.enewick` and `run.log`.
  - `header_*.json` records the argv, host, tool fingerprint and timeouts; `summary.txt` is written
    after each run.
- **`runs/res/_parity/`** is the parity evidence; **`runs/res/check.json`** is the last `--check`.
- **`results/<ds>_sub500_res.csv`**
  - It has `make_sub_csv.py`'s identity and score columns in its order, so it concatenates with
    `<ds>_sub500.csv`. `n_clusters` and `decoder` are empty, and `tag` is the resolver.
  - The resolver columns follow: `variant, protocol, cond, n_elig, n_blobs_tob, out_rootable,
    f1_as_emitted, cluster_f1_rooted, resolver_cpu, resolver_rss_mb, timed_out, note, root_cost`.
  - Then the TOB's own quality from its JSON: `tob_fn, tob_fp` (A63-fixed), `tob_fn_raw,
    tob_fp_raw`, `tob_a63`, `tob_seconds`.
  - Then `violations` and `task_id`.
- **`_summary.txt`**, per label:
  - the full median, the pooled and per-draw subsample medians, the per-cell delta (mean over
    draws - full), the fractions of cells worse and better, and the draw SD, for `cluster_f1`,
    `luay_norm_exact` and `ha_balanced_error`;
  - statuses and `k_pred` against `k_true`;
  - the cluster-F1 delta by stratum;
  - each resolver minus `ours/l2mix_pair/selF1` (both decoders) on the same (draw, cell), and the
    fraction of pairs the GNN wins.

## Traps

- **netcs `*` rows are not the landed netcs protocol.** Read their deltas against `full` as
  "half the data **and** a root". The landed-protocol full rows keep their plain labels, and the
  `protocol` column says which is which.
- **camus's full rows are the Linux rows of record**, from `results/rl/rl_camus_*_tqmctob_tau2.parquet`.
  The subsample rows ran on this Mac's TREE-QMC build.
  - BROOQS is platform-stable: parity was byte-identical.
  - NetCS agreed Mac-vs-Linux on only 74 of 85 camus networks (`blob_cpu_v1/README.md`), so part of
    a camus NetCS delta can be the platform.
- **The camus full-data TOBs were Linux TOB-QMC; the camus subsample TOBs are too.** S5's
  comparison uses the Mac `blob_cpu_v1` TOBs, the only camus full-data TOB strings on this box. It
  is reporting only.
- **Cell names zero-pad the rep to two digits.** netcs rep 0 is `..._rep00_...`, rep 1000 is
  `..._rep1000_...`. `--cells` and the smoke list now refuse an unknown name. They used to drop it
  silently, which is how the first smoke ran 14 tasks instead of 16.
- **One luay cache** (`runs/phylonet/cache.jsonl`) serves `make_sub_csv`, `make_ens_csv` and
  `make_res_csv`. Run them one at a time.
- **Tool errors are outcomes**, exactly as in the landed runs: NetCS segfaults (`rc=-11`),
  `length_error`, non-taxon labels and `duplicate leaf labels`, and BROOQS "Cannot find a safe
  reroot point". They become `error: ...` rows and are not retried; `run_res.py` exits 3 when any
  exist, and 4 only on a hard invariant violation.

## What was run, and what was not (2026-09-22)

- **Run:**
  - `--check` (PASS);
  - `--parity` (PASS, 12/12 plus 6/6 wrapping);
  - `make_res_csv.py --draws full` on all four corpora, into the scratchpad (parity clean). This
    appended its luay measurements to `runs/phylonet/cache.jsonl`.
  - the smoke and its CSVs, into the scratchpad;
  - `--dry-run` of every command-sheet line.

## What the grid did (2026-09-23)

The user ran steps 1-4. **4,200 of 4,200 tasks have a record, 0 hard invariant violations, 0
timeouts**, and about 40 CPU-h in total (`run_res.py --dataset all --summary-only`).

| | BROOQS | NetCS |
|---|---|---|
| ok | 1,800 of 1,800 | 1,618 of 2,400 |
| `out_not_rootable` | 0 | 433 |
| tool error | 0 | 49 |
| CPU vs the landed 1,000-tree run | 0.49-0.58 | 0.66-0.89 |

- **BROOQS never failed and always produced an OUT-rootable network.** Every one of its outputs has
  `k_pred = n_elig`, as the invariant says it must.
- **NetCS's failures are the landed kinds**, concentrated on netcs (315 unrootable, 45 errors of
  4 kinds) and l3reg. They are outcomes, not bugs, and they are why every median has to be read
  beside Figure 5 of the notebook.
- The CSVs are clean on all four corpora: 0 problems, and full-row parity 200 / 80 / 80 / 2,400.
- `run_res.RATIO` was left at the smoke's (0.65, 0.8); the grid measured 0.55 and 0.80. It only sets
  the projection and the LPT order.

## Plotting

`notebooks/plotting_subsample_res.ipynb` (moved from the repo root on 2026-10-21; run under `phylo-env`
as the other notebooks are, from `notebooks/`) reads `results/<ds>_sub500_res.csv` beside
`results/<ds>_sub500.csv` and writes `plots/<DS>-subsample500-resolvers-*-ours_l2mix_pair_selF1.png`.
GNN MAP [z+tc] on each draw comes from `nni_ensemble`'s derived datasets (its README, Derived datasets):

| figure | what it shows |
|---|---|
| `bench` | all trees vs 500 trees, the four methods side by side |
| `delta` | the per-cell cost of halving the data, per method |
| `vs-gnn` | **the headline**: each resolver minus our GNN, paired on the same (draw, cell) |
| `strata` | where that change concentrates |
| `outcomes` | what each route returns at all, and its k |
| `a63` | the fixed-vs-raw TOB difference (l2reg, l3reg) |
| `tobquality` | the resolver's F1 against the error of the tree of blobs it was handed |

It re-checks the CSV gates on the rows it plots, so a stale CSV cannot pass unnoticed.

## A defect this round found

**`make_res_csv.py`'s `vs_gnn` table counted exact ties as GNN wins** (fixed 2026-09-23). It
differenced a full-precision resolver value against the GNN value as the CSV prints it, which is
`%g`-rounded to six significant digits, so two runs that scored identically came out a hair apart.
Measured on camus BROOQS all-trees cluster F1: 6 of 80 pairs are exact ties, 3 of them were counted
for the GNN, and `frac_gnn_better` read 0.7375 where the true figure is 0.700. Both sides are now
rounded the same way before differencing, and a tie counts for neither. Only that summary table
moved; no CSV value changed, and parity still reproduces the landed CSVs.

This needs an amendment number from the Linux box's register; `SPEC-AMENDMENTS.md` here is a stale
prefix.
