# l2reg_eval: the `l2reg_n200_v1` stress test, staged

`l2reg_n200_v1/` holds 20 **exactly-level-2**, regular networks at n = 200, with k = 3–9 and
1,000 IQ-TREE gene trees each. This directory runs the following on it:
- the W10 GNN checkpoints from `gnn_cpu_v1/`;
- greedy-by-support;
- NetCS and BROOQS on three trees of blobs.

It writes one CSV in the shape of `results/rl/camus_a104_per_network.csv`. There are no CAMUS
rows.

The GNN runs through `gnn_cpu_v1/run_gnn_cpu.py` and the resolvers reuse
`blob_cpu_v1/run_blob_cpu.py`'s own timing and scoring functions. All scoring uses
`gnn_cpu_v1/pylib`. **Every distance and every network size is PhyloNet's**; `opc.mpl.netdist` is
not used. The GNN and greedy drivers offer two decoders (see [Decoders](#decoders)); the formal
level-1 decoder is the default and reproduces the landed camus rows.

| file | what it does |
|---|---|
| `common.py` | paths, the label map, the truth loader; pins BLAS and puts `gnn_cpu_v1/pylib` on the path |
| `stage.py` | builds `inputs/`, `tob/`, `CELLS.tsv` and `stage.json` from the dataset, behind blocking gates |
| `run_greedy.py` | the `greedy` rows (`--decoder`, `--max-level`); `--camus` reruns it on the 100 camus cells for parity |
| `run_resolvers.py` | NetCS and BROOQS on 20 networks × {raw, fixed, true} trees of blobs |
| `check_relaxed.py` | gates for the relaxed decoder; pure Python, ~2 min once its candidate-set cache is built |
| `make_csv.py` | builds the CSV and a summary JSON, with luay from PhyloNet and hybrid ancestry beside it; `--parity` runs the gates against the landed camus rows |
| `camus_expand.py` | appends **one** relaxed camus run's rows to a copy of `camus_a104_per_network.csv`, luay by PhyloNet |
| `camus_expand_multi.py` | the same for **several** runs at once, formal or relaxed, in one wide table |

## Run it, from the repo root, in a terminal

Use your terminal rather than a Claude Code background task: long runs have been killed there
before (`blob_cpu_v1/README.md`). Every step is resumable, so an interrupted step resumes where
it stopped.

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar   # the default anyway

$PY l2reg_eval/stage.py                               # ~20 s; already done once, re-runnable
$PY l2reg_eval/run_greedy.py --camus                  # ~5 min: greedy on the 100 camus cells
$PY l2reg_eval/make_csv.py --parity                   # ~1 min; must print PASS

(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
   --inputs ../l2reg_eval/inputs --cells ../l2reg_eval/CELLS.tsv --tags _mix,_mix2L,_mix75 \
   --reference ../l2reg_eval/NO_REFERENCE.tsv --out ../l2reg_eval/runs --run-name gnn_t1)
                                                       # ~20-25 min: 20 cells x 9 checkpoints

$PY l2reg_eval/run_greedy.py                          # ~3 min
caffeinate -i $PY l2reg_eval/run_resolvers.py --jobs 8   # ~15-25 min: 60 tasks x 2 resolvers
$PY l2reg_eval/make_csv.py                            # ~1 min: the CSV
```

**Optional: the relaxed decoder beside the formal one.** It gives separate rows in the same CSV.
Run the level-cap-2 GNN and greedy runs after the formal ones, before the last `make_csv.py`:

```bash
$PY l2reg_eval/check_relaxed.py                       # ~2 min (built caches), once; must PASS
(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
   --inputs ../l2reg_eval/inputs --cells ../l2reg_eval/CELLS.tsv --tags _mix,_mix2L,_mix75 \
   --reference ../l2reg_eval/NO_REFERENCE.tsv --out ../l2reg_eval/runs --run-name gnn_relaxedL2_t1 \
   --decoder relaxed --max-level 2)                    # ~25-30 min
$PY l2reg_eval/run_greedy.py --decoder relaxed --max-level 2      # ~3 min -> runs/greedy_relaxedL2/
```

The GNN command needs a little explanation:
- `NO_REFERENCE.tsv` does not exist, on purpose. `--reference` holds the landed camus rows the
  driver checks its selections against, keyed by (phase, rep, arm, tag, seed). It has no
  `l2reg_n200` rows, so for these cells it can only ever match nothing; a missing file (read as
  empty) makes that explicit. The relaxed decoder ignores the reference in any case, since the
  landed rows are level-1 decodes.
- The separate `--out`/`--run-name` keeps the l2reg run apart from the camus runs in
  `gnn_cpu_v1/runs/`. While the staged cells borrowed camus's `cam200` / `cam_g500` names, it was
  also what stopped the camus `cam200_*` files there from marking every l2reg cell as done.
- **A run directory holds one decoder.** Every cell JSON records its decoder, and resuming into a
  directory decoded differently is refused. Without `--run-name`, a relaxed run defaults to a
  name ending `_relaxedL<L>`.
- To use more cores, run two shards with distinct run names, for example `--reps 0-9
  --run-name gnn_t1a` and `--reps 10-19 --run-name gnn_t1b`. `make_csv.py` picks up every
  `runs/gnn*/` and every `runs/greedy*/` except `greedy_camus*` by default.
- Use smoke flags first if you like: `run_gnn_cpu.py ... --reps 0` and
  `run_resolvers.py --keys 0 --variants fixed`.

### `_l2mix/s1` and `_l2mix75/s0` (2026-09-13)

Two checkpoints with no common seed set, so they run through `run_gnn_cpu.py --models` (see
`gnn_cpu_v1/README.md`) in their **own** run directories -- resume is per cell, and the existing
`gnn_t1_l2*` directories would skip every cell:

```bash
M=_l2mix/s1,_l2mix75/s0; N=gnn_t1_l2mix_s1_l2mix75_s0
A="--threads 1 --models $M --inputs ../l2reg_eval/inputs --cells ../l2reg_eval/CELLS.tsv --reference ../l2reg_eval/NO_REFERENCE.tsv --out ../l2reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL1 --decoder relaxed --max-level 1)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL2 --decoder relaxed --max-level 2)
$PY l2reg_eval/make_csv.py                            # 960 rows, "complete": true
```

**`make_csv.py` now reads the seeds each tag must have off `gnn_cpu_v1/checkpoints/<tag>/s*/`**
(l3reg_eval's rule), and requires every shipped tag at level 1. The flat `(0, 1, 2)` it used
turned the one-seed `_l2mix` into three spurious "missing" problems, which is why the 2026-09-12
build recorded `"complete": false`. The 2026-09-13 rebuild carries all 840 earlier rows byte for
byte and adds 120: `ours/l2mix` seed 1 and `ours/l2mix75` seed 0, 20 each under the formal
decoder, relaxed L1 and relaxed L2.

### `_l2mixha`: `ours/l2mixHA/selF1` and `ours/l2mixHA/selHA` (2026-09-14)

`gnn_cpu_v1/checkpoints/_l2mixha/s0/` holds two checkpoints of one run:
- `best.pt`, selected on F1, labelled `ours/l2mixHA/selF1`;
- `best_ha.pt`, selected on hybrid-ancestry error, labelled `ours/l2mixHA/selHA`.

`run_gnn_cpu.py --models` names the second as `_l2mixha/s0/best_ha` and records it as tag
`_l2mixha:best_ha` (see `gnn_cpu_v1/README.md`). Both labels are in `common.GNN_LABELS`.
`make_csv.py` now builds its required models from checkpoint **files**, so both are required:

```bash
M=_l2mixha/s0,_l2mixha/s0/best_ha; N=gnn_t1_l2mixha
A="--threads 1 --models $M --inputs ../l2reg_eval/inputs --cells ../l2reg_eval/CELLS.tsv --reference ../l2reg_eval/NO_REFERENCE.tsv --out ../l2reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL1 --decoder relaxed --max-level 1)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL2 --decoder relaxed --max-level 2)
$PY l2reg_eval/make_csv.py                            # 1,080 rows, "complete": true
```

The 2026-09-14 rebuild carries all 960 earlier rows byte for byte and adds 120: each label × 3
decoders × 20 networks, all `ok`.

### `_l2mix_pair`: `ours/l2mix_pair/selF1` and `ours/l2mix_pair/selHA` (2026-09-14)

`gnn_cpu_v1/checkpoints/_l2mix_pair/s0/` is `_l2mix`'s data on the pair conv (`conv: pair`, see
`gnn_cpu_v1/README.md`). It ships two checkpoints: `best.pt` (epoch 24) → `ours/l2mix_pair/selF1`,
and `best_ha.pt` (epoch 17, tag `_l2mix_pair:best_ha`) → `ours/l2mix_pair/selHA`. Both labels are in
`common.GNN_LABELS`, and `make_csv.py` requires both:

```bash
M=_l2mix_pair/s0,_l2mix_pair/s0/best_ha; N=gnn_t1_l2mix_pair
A="--threads 1 --models $M --inputs ../l2reg_eval/inputs --cells ../l2reg_eval/CELLS.tsv --reference ../l2reg_eval/NO_REFERENCE.tsv --out ../l2reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL1 --decoder relaxed --max-level 1)
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL2 --decoder relaxed --max-level 2)
$PY l2reg_eval/make_csv.py                            # 1,200 rows, "complete": true
```

**Superseded deliveries.** Two earlier deliveries were run and replaced. Both sets of runs sit
outside `make_csv.py`'s `runs/gnn*` glob:
- **Epoch 2.** One model twice (tensor-identical). `best.pt` alone was run and landed a partial
  1,140-row CSV. The runs are in `runs_superseded/l2mix_pair_epoch2/`.
- **Epochs 7/6.** A complete 1,200-row CSV. The runs are in `runs_superseded/l2mix_pair_epoch7_6/`,
  and the CSV and JSON in `results/rl/superseded/l2mix_pair_epoch7_6/`.

The 2026-09-15 rebuild on the **epoch-24/17** checkpoints carries all 1,080 pre-pair rows byte for
byte and adds 120: each label × 3 decoders × 20 networks, all `ok`. None of the epoch-7/6 pair rows
survive. Every cell records epoch 24 (`_l2mix_pair`) or 17 (`_l2mix_pair:best_ha`).

### Optional: `ours/2x-data`

That label is the `_mix2` checkpoint, which `gnn_cpu_v1/` does not ship. To add it:
1. Copy `results/rl/rl_run_gnn_segment_s{0,1,2}_mix2/` (`best.pt`, `config.json`,
   `verdict.json`) from the Linux box into `gnn_cpu_v1/checkpoints/_mix2/s{0,1,2}/`. All three
   seeds are needed: the driver takes the union of seeds across tags and exits on a missing one.
2. Run the GNN command with `--tags _mix2 --run-name gnn_mix2_t1` (and again with the relaxed
   flags if you want those rows).
3. Re-run `make_csv.py`.

It can come after everything else.

## Gene-tree scaling: gt5k / gt10k (2026-09-21)

`l2reg_n200_gt5k_v1/` and `l2reg_n200_gt10k_v1/` hold the same 20 networks with 5,000 and 10,000
IQ-TREE gene trees. Each has a TREE-QMC tree of blobs recomputed on its own trees. l3reg has the same
pair (`l3reg_eval/README.md`, same section). The question is how each method changes when it gets
more gene trees, compared with the 1k CSV above.

**Scope:**
- **GNN.** `_l2mix_pair/s0`, both checkpoints (`ours/l2mix_pair/selF1`, `selHA`), at the headline
  decoder only: relaxed L2 here and relaxed L3 on l3reg. It reads **every** tree, with
  `--tau-frac 0.002`, so τ = 10 at 5k and τ = 20 at 10k.
- **Greedy.** Level 1 (the plotted `greedy`) and relaxed L2, on the same candidate set.
- **NetCS and BROOQS.** On the **raw and fixed** trees of blobs. All three TOB rows are still
  emitted.

**Why there is no τ = 2 arm beyond 1k.** Measured on rep00:

| trees | τ | \|C\| | overlap edges |
|---|---|---|---|
| 1k | 2 | 8,789 | 2.0M |
| 5k | 2 | 30,237 | 35M |
| 10k | 2 | 51,924 | 120M, over `EDGE_CAP` (100M), so `candidates.build` raises |
| 10k | 20 | 5,964 | 0.75M |

At 1,000 trees `--tau-frac 0.002` is τ = 2 exactly, so **the landed 1k rows are this arm's
baseline.** Any τ other than 2 of 1,000 is out of distribution for the features normalised by |C|
(`gnn_cpu_v1/infer_gnn_cpu.md` §7.6).

**Pair on networks, not on estimated trees.** gt5k's trees are gt10k's first 5,000. The 1k
dataset's estimated trees are **not** a prefix of either, because the outgroup graft height is taken
over all loci. So the three datasets share networks and true gene trees only.

### How it is wired

- **`REG_GT=gt5k|gt10k` selects the dataset** (`common.py`). Unset, every constant is v1's. Every
  script reads `common`'s constants at import, which is why this is an environment variable. Each
  dataset stages and runs under `l2reg_eval/<REG_GT>/{inputs,tob,runs,CELLS.tsv,stage.json}` with
  phase `l2reg_n200_<REG_GT>` (registered in `opc.rl.camus`). Its CSV is
  `results/rl/l2reg_n200_<REG_GT>_v1_per_network.csv` (+ `.json`) with
  `gene_trees = est_bp1000_<REG_GT>`, so the three CSVs concatenate cleanly.
- **Every driver used to read the first 1,000 trees of any file, silently.** That was
  `run_gnn_cpu.py` and `run_greedy.py`, via `M, TAU = 1000, 2` and `max_loci=M`.
  - `run_gnn_cpu.py` now takes `--m N` (0 = all) and `--tau-frac F`. Off the defaults it records
    `m_arg`, `tau` and `tau_frac` in each cell, and refuses to resume over cells built otherwise.
  - `run_greedy.py` reads `common.M, TAU`.
  - **`make_csv.py` gates every GNN and greedy cell** on `n_loci == m == common.M`,
    `tau == common.TAU` and, for the GNN, `tau_frac == 0.002`. `common.TAU` is written out rather
    than computed, so it checks the driver's ⌈0.002·m⌉. v1 cells are held to (1000, 2), and pass.
- **Completeness follows the scope:** `common.GNN_REQUIRED`, `GREEDY_REQUIRED` and
  `RESOLVER_VARIANTS`. `run_resolvers.py` defaults to raw + fixed there.

### Checked (2026-09-21)

- **Staging.** `stage.py` passed every gate on all four datasets, including tree count = MANIFEST
  `n_loci`. Staged trees: 0.6 GB (5k) and 1.2 GB (10k) per dataset.
- **v1 unchanged.** With `REG_GT` unset, `make_csv.py` rebuilds both landed v1 CSVs byte for byte,
  with the m/τ gate passing on every landed cell.
- **`run_gnn_cpu.py` defaults unchanged.** l2reg 1k rep00, `_l2mix_pair` relaxed L2: the cell JSON
  equals the landed one (timings aside).
- **Equivalence.** `--m 0 --tau-frac 0.002` on the same 1k cell resolves to m = 1000 and τ = 2 and
  selects the same clusters. Only `m_arg` / `tau` / `tau_frac` are added.
- **No truncation.** On gt5k / gt10k rep00:
  - `n_loci = m = 5,000 / 10,000` and τ = 10 / 20;
  - |C| = 6,117 / 5,964, exactly the counts taken independently off the dataset's `.nwk.gz`;
  - greedy's |C| matches;
  - 38 / 49 s per cell for both models, peak RSS 1.9 / 2.0 GB.
- **Guards.**
  - The resume guard refuses a flagless rerun over flagged cells.
  - `make_csv.py` refuses cells hand-edited to 1,000 trees or to no `tau_frac` (the gate is not
    vacuous).
  - Resolver `--self-check` and `--dry-run` pass: 40 tasks per dataset.

**Not run:** the grids below. All smoke output went to the scratchpad; `gt*/runs/` is empty.

### Command sheet (zsh; from the repo root, in a terminal)

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar   # the default anyway

# 0. ALREADY DONE (2026-09-21), every gate passing; re-run only if a dataset changes
for G in gt5k gt10k; do REG_GT=$G $PY l2reg_eval/stage.py && REG_GT=$G $PY l3reg_eval/stage.py; done

# 1. resolvers, the long pole.  One task first: it is also the timing probe for the estimate below,
#    and it is kept by the grid.
REG_GT=gt5k caffeinate -i $PY -u l2reg_eval/run_resolvers.py --keys 0 --variants fixed
REG_GT=gt5k caffeinate -i $PY -u l2reg_eval/run_resolvers.py --jobs 12
REG_GT=gt5k caffeinate -i $PY -u l3reg_eval/run_resolvers.py --jobs 12
REG_GT=gt10k caffeinate -i $PY -u l2reg_eval/run_resolvers.py --jobs 8
REG_GT=gt10k caffeinate -i $PY -u l3reg_eval/run_resolvers.py --jobs 8

# 2. GNN -- a second terminal, alongside step 1: one core and ~2 GB each
for DS in l2reg l3reg; do for G in gt5k gt10k; do
  L=$([ $DS = l2reg ] && echo 2 || echo 3)
  (cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
     --models _l2mix_pair/s0,_l2mix_pair/s0/best_ha --m 0 --tau-frac 0.002 \
     --decoder relaxed --max-level $L --inputs ../${DS}_eval/$G/inputs \
     --cells ../${DS}_eval/$G/CELLS.tsv --reference ../${DS}_eval/NO_REFERENCE.tsv \
     --out ../${DS}_eval/$G/runs --run-name gnn_t1_l2mix_pair_ftau0.002_relaxedL$L)
done; done

# 3. greedy, level 1 and the headline relaxed decoder
for G in gt5k gt10k; do
  REG_GT=$G $PY l2reg_eval/run_greedy.py
  REG_GT=$G $PY l2reg_eval/run_greedy.py --decoder relaxed --max-level 2
  REG_GT=$G $PY l3reg_eval/run_greedy.py
  REG_GT=$G $PY l3reg_eval/run_greedy.py --decoder relaxed --max-level 3
done

# 4. the CSVs; each refuses to write, and lists what is missing, until its dataset is complete
for G in gt5k gt10k; do REG_GT=$G $PY l2reg_eval/make_csv.py; REG_GT=$G $PY l3reg_eval/make_csv.py; done
```

| step | per dataset | all four |
|---|---|---|
| GNN (measured on rep00) | ~13 min (5k), ~16 min (10k) | ~1 h |
| greedy, two decoders | ~7 min (5k), ~14 min (10k) | ~40 min |
| resolvers, raw + fixed (**linear extrapolation**, not measured) | 11–12 CPU-h (5k), 21–24 CPU-h (10k) | ~68 CPU-h, 7–9 h wall |
| `make_csv.py` | ~1 min | ~5 min |

- **The resolver estimate** scales the landed 1k raw + fixed CPU (8,730 s for l2reg, 7,681 s for
  l3reg) by the tree count. BROOQS and NetCS both count quartets tree by tree. The longest
  1k task was 483 s, so expect about 80 min at 10k.
- **Resolver memory** peaked at 700 MB (NetCS) and 145 MB (BROOQS) at 1k; it has not been measured
  at 10k. Hence `--jobs 8` there until the first tasks have been seen.
- `run_resolvers.py` has no timeout. On a hang, Ctrl-C and re-run: finished tasks, and finished
  methods inside a task, are kept.

Each finished dataset gives 220 rows:
- 20 each of `ours/l2mix_pair/selF1 [relaxed L2]` and `selHA [relaxed L2]`;
- 20 each of `greedy` and `greedy [relaxed L2]`;
- 60 TOB rows;
- 80 resolver rows (`NetCS`, `BROOQS`, and the two `(fixed TOB)` labels).

The method labels are v1's. The 1k comparison rows are the same labels in
`l2reg_n200_v1_per_network.csv`, told apart by `gene_trees`.

## Decoders

Both drivers take `--decoder level1` (the default) or `--decoder relaxed --max-level L`. The
scores, the rank rule (descending score, ties to the lower index) and the budget k are the same;
what differs is what counts as a legal selection, and how a selection becomes a network.

**`level1`: the formal decoder.** This is what every landed row used.
- The mask is (L) and (N3O), the Hellmuth–Schaller–Stadler characterisation of *level-1*
  clustering systems.
- The budget counts distinct pairwise intersections.
- The selection is closed under intersections before its Hasse diagram is read as a network.

On level-2 truths all three break:
- (L) fails on 20 of 20 l2reg truths.
- The closure breaks 10 of the 20 true networks.
- With perfect ranking the mask refuses 71 of the 4,026 true clusters in the candidate set.

**`relaxed --max-level L`: `opc.rl.relaxed`.** A candidate is legal if the Hasse diagram of the
selection plus it, with **no closure**, satisfies all four of:
- reticulations (edges − nodes + 1) ≤ k;
- every in-degree ≤ 2;
- no block holds more than L reticulations;
- every hybrid can be given a donor parent without any node donating twice.

With perfect ranking and L = 2 it keeps every true cluster on all 20 networks. `L = 1` is also a
level-1 decoder, but not the formal one (no closure, a different budget rule). Any `L ≥ k` means
no cap. Each step accepts the best legal candidate, as the formal decoder does. Unlike the
formal constraints, these can loosen: an acceptance can make a refused cluster legal. Such a
cluster is rechecked at once, before anything lower-ranked is considered, and each cell's
`decode_stats.recheck_accepts` counts them.

`check_relaxed.py` gates it:
- **G0:** donor-assignment unit cases.
- **G1:** 40 true networks rebuilt from their clusters, with level cross-checked against the
  independent `networks.level`.
- **G2:** the incremental decoder against a from-scratch reference, with a rebuild check after
  every accept.
- **G3:** perfect ranking keeps every true cluster at L = 2.

It also reports support-ranked F1 at L = 1, 2 and k beside the formal decoder.

## Outputs

- `results/rl/l2reg_n200_v1_per_network.csv`: 380 rows for the formal decoder (440 with
  `_mix2`), plus 200 per relaxed variant (180 GNN + 20 greedy).
  - **The camus eleven**, in camus order: `method, gene_trees, n, network_idx, seed,
    cluster_f1, luay, luay_norm, luay_norm_exact, k_pred, k_true`.
  - **Then the extras:** `decoder, key, cu, n_l2, n_l1, n_blobs_tob, tp, fp, fn, precision,
    recall, mass_pred, mass_truth, cluster_f1_rooted, status`.
  - **Then hybrid ancestry:** `ha_n_true, ha_n_pred, ha_fp, ha_fn, ha_fpr, ha_fnr,
    ha_balanced_error` (see [Reading the numbers](#reading-the-numbers)).
  - **Row values:** `gene_trees = est_bp1000`; `network_idx` is the MANIFEST `rep`; `key` is the
    l2reg key. `decoder` is `level1`, `relaxed_L<L>`, or empty for the tree-of-blobs methods.
    Empty fields are `''`, as in camus_a104.
- `results/rl/l2reg_n200_v1.json`: provenance (jar, run directories, decoders, label map),
  per-method medians, and the parity report.
- `l2reg_eval/runs/`: every cell JSON, with every network as eNewick, plus logs and timings.

| label | what |
|---|---|
| `ours/prior`, `ours/2x-data-full`, `ours/+n75` | GNN `_mix`, `_mix2L`, `_mix75`, seeds 0–2, each given `k_true` as its budget, formal decoder |
| `greedy` | greedy-by-support, same candidate set and budget, formal decoder |
| `ours/... [relaxed L2]`, `greedy [relaxed L2]` | the same with `--decoder relaxed --max-level 2` |
| `TOB-QMC` | TREE-QMC's tree of blobs as written, rooted with `root_tob_on_outgroup` (**the camus protocol**), scored as a tree |
| `TOB-QMC (fixed)` | `<key>.tob`: TREE-QMC's own contraction rule applied to the root branch it skips; differs from the raw output on 12 of 20 networks |
| `true TOB*` | the true tree of blobs (oracle) |
| `NetCS`, `BROOQS` | resolving the raw TOB. These keep camus's labels because they are camus's protocol |
| `NetCS (fixed TOB)`, `BROOQS (fixed TOB)` | resolving the fixed TOB |
| `NetCS*(true TOB)`, `BROOQS*(true TOB)` | resolving the true TOB (oracle) |

## Reading the numbers

- **`cluster_f1` = 1 is unreachable for the formal decoder.** Its mask admits only level-1
  cluster systems, and all 20 truths are level-1-infeasible. The relaxed decoder at L ≥ 2 can
  reach every true cluster in the candidate set (98.7% of them are there), so from it the gap to
  1 is the scores, not the decoder. Both get `k_true` = 3–9 as their budget, above anything the
  checkpoints saw at n = 200 in training, and none of them was trained on level-2 labels.
- **BROOQS and NetCS place one reticulation per blob they resolve.** So `k_pred` ≤ that
  row's `n_blobs_tob`. On the true TOB this is `n_l2 + n_l1`, one short of `k` per level-2 blob.
  Neither method is given a budget.
- **Some NetCS networks have no OUT-rooted reading**, because OUT sits below one of their own
  hybrids (`status = out_not_rootable`). As in camus_a104, `cluster_f1` then holds the value
  read off the network as emitted and `luay` is empty. `cluster_f1_rooted` is the principled
  column, and it is empty there.
- **`luay` is PhyloNet `CMPNETS -m luay` and nothing else.** Both networks are OUT-free with
  gammas stripped. The same jar produced every landed luay. With this convention it reproduces
  them on the camus cells (checked 2026-09-11): all 900 GNN rows, 61 of 61 greedy cells, and the
  resolver and TOB rows except one cell whose Mac tree of blobs differs from the Linux one.
- **`mass_pred`/`mass_truth` are measured with the jar too.** CMPNETS against the star tree on the
  network's taxa, and against that star with one unary node above it, gives the network's size
  exactly, as the jar counts it: size = luay(N, star) + luay(N, star′) − 1 + n.
  - On the camus networks this equals the landed masses everywhere but one truth (cam150 rep04,
    which has a cycle of size 2; no l2reg network has one).
  - So `luay_norm_exact` = luay / floor((X_pred + X_truth) / 2) is camus_a104's definition, and is
    at most 1 by construction. `luay_norm` uses the closed form n − 1 + floor(3(k_true + k_pred)/2).
- **Compare `luay_norm_exact` across the two CSVs, not raw `luay`.** The camus truths carried
  one extra unary node (mass 2n + 3k) and these do not (2n + 3k − 1). The JSON reports that
  offset per build. So raw `luay` can differ from the camus convention by up to 1.
- **Cell files are named `l2reg_n200_repRR_est_bp1000.json`**: phase `l2reg_n200` and arm
  `est_bp1000`, both registered in `opc.rl.camus` for this dataset. `make_csv.py` maps rep → key.
  Runs made while the staged cells borrowed camus's `cam200` / `cam_g500` names hold
  `cam200_*.json` files, which `make_csv.py` ignores; move those folders aside and rerun.
- **`ha_*` is hybrid ancestry, the metric Dai & Molloy score NetCS with** (bioRxiv
  10.64898/2026.06.06.730607, Fig. 3). A taxon has hybrid ancestry if it lies below at least one
  hybrid. With P and T those taxa in the prediction and in the truth:
  - `ha_fpr` = |P − T| / |P| and `ha_fnr` = |T − P| / |T|. `ha_balanced_error` is their mean, the
    number the paper reports.
  - A tree of blobs has no hybrids, so its `ha_fpr` is 0/0 and is left empty, as is its
    `ha_balanced_error`. Its `ha_fnr` is 1.
  - The four `out_not_rootable` NetCS rows are read off the network as emitted. Unlike `luay`,
    they have a value, because hybrid ancestry does not depend on where the root is. NetCS put
    OUT below a hybrid in these, and most of the ingroup with it.
  - `make_csv.py` computes it two independent ways and lists any network where they disagree.

## The camus CSV: `camus_expand.py` and `camus_expand_multi.py`

Both score a `gnn_cpu_v1` **camus** run the way camus_a104's GNN rows were scored -- selection F1
from the driver, luay and both masses from PhyloNet -- and write the result beside the landed
rows. Neither touches `results/rl/camus_a104_per_network.csv`; each carries every one of its lines
through byte for byte and appends.

- `camus_expand.py --run <dir>` takes one run and emits the camus eleven, with a sidecar holding
  the extras. It **refuses a level-1 run**, on the assumption that every checkpoint's formal rows
  are already landed.
- `camus_expand_multi.py --run <dir> [--run <dir> ...]` takes any number, formal or relaxed, and
  writes **one wide table**: the camus eleven, then `phase, arm, tag, decoder, n_selected, tp, fp,
  fn, precision, recall, mass_pred, mass_truth, f1_formal_selection, f1_formal_network,
  same_network_as_formal, status, enewick`. Base rows are empty in all seventeen. The refusal is
  per row rather than per run -- a run is rejected only where camus_a104 already holds a row under
  the same (method, gene_trees, n, network_idx, seed) -- which is what lets a **new checkpoint
  family's formal rows** through while still refusing `_mix` / `_mix2L` / `_mix75`'s.

`_l2mix` (2026-09-12) is that case: `gnn_cpu_v1/README.md` has the three runs, and

```bash
PY=gnn_cpu_v1/.venv/bin/python
$PY l2reg_eval/camus_expand_multi.py \
    --run gnn_cpu_v1/runs/C4537CLFY0_t1_relaxedL1 \
    --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2 \
    --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2_relaxedL1 \
    --run gnn_cpu_v1/runs/C4537CLFY0_t1_l2_relaxedL2 \
    --formal-run gnn_cpu_v1/runs/C4537CLFY0_t1 \
    --out results/rl/camus_a104_per_network_l2mix.csv     # ~1 min, PhyloNet is 30 s of it
```

writes 7,836 landed rows + 1,200 new ones (900 `ours/* [relaxed L1]`, 300 `ours/l2mix` across the
three decoders) and `results/rl/camus_a104_per_network_l2mix.json` beside it.

**2026-09-13: `_l2mix/s1` and `_l2mix75/s0`.** Their three camus runs
(`gnn_cpu_v1/runs/C4537CLFY0_t1_l2mix_s1_l2mix75_s0{,_relaxedL1,_relaxedL2}`, via `--models`) join
the list, and the file is rebuilt in place:

```bash
R=gnn_cpu_v1/runs; N=C4537CLFY0_t1_l2mix_s1_l2mix75_s0
$PY l2reg_eval/camus_expand_multi.py \
    --run $R/C4537CLFY0_t1_relaxedL1 --run $R/C4537CLFY0_t1_l2 \
    --run $R/C4537CLFY0_t1_l2_relaxedL1 --run $R/C4537CLFY0_t1_l2_relaxedL2 \
    --run $R/$N --run $R/${N}_relaxedL1 --run $R/${N}_relaxedL2 \
    --formal-run $R/C4537CLFY0_t1 --formal-run $R/C4537CLFY0_t1_l2 \
    --out results/rl/camus_a104_per_network_l2mix.csv     # ~75 s
```

That is 7,836 landed rows + **1,800** new ones. Every one of the previous build's 9,036 lines is
carried through byte for byte (checked), and the 600 added are `ours/l2mix` seed 1 and
`ours/l2mix75` seed 0, 100 each per decoder, all `ok`. In the JSON's medians `ours/l2mix` now
pools its two seeds.

**2026-09-14: `_l2mixha`.** Its three camus runs, `gnn_cpu_v1/runs/C4537CLFY0_t1_l2mixha{,_relaxedL1,
_relaxedL2}`, carry two models each: `best.pt` as tag `_l2mixha` → `ours/l2mixHA/selF1`, and
`best_ha.pt` as tag `_l2mixha:best_ha` → `ours/l2mixHA/selHA`. They join the list:

```bash
H=C4537CLFY0_t1_l2mixha
$PY l2reg_eval/camus_expand_multi.py ...the seven --run lines above... \
    --run $R/$H --run $R/${H}_relaxedL1 --run $R/${H}_relaxedL2 \
    --formal-run $R/C4537CLFY0_t1 --formal-run $R/C4537CLFY0_t1_l2 \
    --out results/rl/camus_a104_per_network_l2mix.csv
```

That is 7,836 landed rows + **2,400** new ones. All 9,636 lines of the previous build are carried
through byte for byte (checked), and the 600 added are each label × 3 decoders × 100 cells, all
`ok`. The `tag` column tells the two checkpoints apart.

**2026-09-15: `_l2mix_pair`.** Its three camus runs, `gnn_cpu_v1/runs/C4537CLFY0_t1_l2mix_pair{,
_relaxedL1,_relaxedL2}`, use the pair conv and carry two models each: `best.pt` at epoch 24 as tag
`_l2mix_pair` → `ours/l2mix_pair/selF1`, and `best_ha.pt` at epoch 17 as tag `_l2mix_pair:best_ha` →
`ours/l2mix_pair/selHA`. They join the list:

```bash
P=C4537CLFY0_t1_l2mix_pair
$PY l2reg_eval/camus_expand_multi.py ...the ten --run lines above... \
    --run $R/$P --run $R/${P}_relaxedL1 --run $R/${P}_relaxedL2 \
    --formal-run $R/C4537CLFY0_t1 --formal-run $R/C4537CLFY0_t1_l2 \
    --out results/rl/camus_a104_per_network_l2mix.csv
```

That is 7,836 landed rows + **3,000** new ones. All 10,236 lines of the previous build are carried
through byte for byte (checked), and the 600 added are each label × 3 decoders × 100 cells, all
`ok`. The superseded runs were never passed here:
- the epoch-2 runs, in `gnn_cpu_v1/runs_superseded/l2mix_pair_epoch2/`;
- the epoch-7/6 runs, in `gnn_cpu_v1/runs_superseded/l2mix_pair_epoch7_6/`, whose 10,836-row build
  is in `results/rl/superseded/l2mix_pair_epoch7_6/`.

The epoch-24/17 build ran relaxed L2 first, and an interim build landed with relaxed L1 at 64 of 100
cells. **Some rows repeat the epoch-7/6 build byte for byte**, because a camus line records the
selection and the network but not the epoch. Wherever both deliveries chose the same clusters, the
two lines are identical, even though every cell in the runs passed here is epoch 24 or 17. The 900 reproduce
`gnn_cpu_v1/runs/C4537CLFY0_t1_relaxedL1/camus_a104_per_network_relaxedL1.csv` on all eleven
columns, which is the check that this path is `camus_expand.py`'s.

- **`--formal-run` is what `f1_formal_network` and `same_network_as_formal` are read from.** A
  `--run` that is itself level-1 is added to the list, so `_l2mix`'s relaxed rows compare against
  `_l2mix`'s own formal network without naming that directory twice.
- **The PhyloNet workdir must be absolute.** `phylonet.run` writes `<workdir>/run.nex` and then
  launches java with `cwd=workdir` while passing that same path as an argument, so a relative one
  resolves twice; the jar prints `No such file 'run.nex'`, exits **zero**, and the only symptom is
  `parsed 0 distances`. The script resolves `--out` and `--work` for this reason.
- **A checkpoint tag with no `common.GNN_LABELS` entry is refused**, rather than silently labelled.

## Parity (`make_csv.py --parity`)

Everything below uses PhyloNet only, and concerns the formal decoder, which is what the landed
rows used. Two misses are known and explained, and each is allowed to differ only in the fields
it explains:
- **cam150 rep04, any row:** `mass_truth` may differ. The truth has a size-2 cycle, which the jar
  sizes differently from the landed number.
- **TOB-QMC cam200 rep15:** `luay` and `mass_pred` may differ. The Mac tree of blobs is not the
  Linux one, so this is a different input, not a different metric.

**(a) Resolver and TOB rows. Gate.** The landed NetCS/BROOQS, TOB-QMC and true-TOB rows must
reproduce their luay and both masses.

**(b) GNN rows. Gate.** The Mac GNN networks in `gnn_cpu_v1/runs/C4537CLFY0_t1` must reproduce the
landed GNN luay and masses: 900 of 900 luay when measured. (netdist, the pure-Python
reimplementation this pipeline no longer uses, matched only 526; it mis-scores networks with a
hybrid of two or more children. That, not a decode or taxon-order effect, is what once made these
rows look irreproducible.)

**(c) Greedy. Gate.** On the 100 camus cells, greedy must reproduce the landed
tp/fp/`k_pred`/F1, its network's size and `luay_greedy` exactly, because it is deterministic.
`make_csv.py` refuses to compare a relaxed-decoder camus run here.

A failure in (b) or (c) would mean the landed realise→luay path differs from this one. Then
`scripts/w10_gnn_eval.py` / `scripts/w9_swap_eval.py` are the files to fetch from the Linux box.

## When something goes wrong

- **`stage.py` fails:** it writes neither `CELLS.tsv` nor `stage.json`, and it names the gate
  that failed.
- **A resolver hangs:** Ctrl-C and re-run. Finished tasks, and finished methods inside a task,
  are kept. `runs/resolvers/work/<key>/<variant>/*.log` holds each tool's full log.
- **A resolver errors:** it becomes a row with `status = error: ...`, not a crash.
  `run_resolvers.py` exits 3 so it is not missed.
- **PhyloNet refuses a network:** that row gets `status = phylonet_failed` and empty luay columns.
  `cmpnets_many` falls back to one call per pair, so one bad string cannot cost the batch.
- **A driver refuses to resume:** the run directory was decoded with another decoder. Pick a new
  `--run-name`, or `--force` to overwrite.
- **`make_csv.py` refuses to write when a cell is missing:** it lists the missing cells, per
  decoder. `--allow-partial` writes what exists.
