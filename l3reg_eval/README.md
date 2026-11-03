# l3reg_eval: the `l3reg_n200_v1` stress test, staged

`l3reg_n200_v1/` holds 20 **exactly-level-3**, regular networks at n = 200, with k = 4–12 and
1,000 IQ-TREE gene trees each. This directory runs the following on it:
- the W10 GNN checkpoints from `gnn_cpu_v1/`, all four families;
- greedy-by-support;
- NetCS and BROOQS on three trees of blobs.

It writes one CSV in the shape of `results/rl/camus_a104_per_network.csv`. There are no CAMUS rows.

It is `l2reg_eval/` at level 3, and deliberately a copy of it rather than a generalisation of it:
`diff -u l2reg_eval/<x>.py l3reg_eval/<x>.py` is the check that it has not drifted, and
`l2reg_eval/` keeps re-deriving its own CSV byte for byte. `camus_expand.py` and
`camus_expand_multi.py` are **not** copied — they write the camus CSV and never read this dataset,
so `l2reg_eval/`'s copies serve both.

The GNN runs through `gnn_cpu_v1/run_gnn_cpu.py` and the resolvers reuse
`blob_cpu_v1/run_blob_cpu.py`'s own timing and scoring functions. All scoring uses
`gnn_cpu_v1/pylib`. **Every distance and every network size is PhyloNet's**; `opc.mpl.netdist` is
not used.

| file | what it does |
|---|---|
| `common.py` | paths, the label map, the truth loader; pins BLAS and puts `gnn_cpu_v1/pylib` on the path |
| `stage.py` | builds `inputs/`, `tob/`, `CELLS.tsv` and `stage.json` from the dataset, behind blocking gates |
| `run_greedy.py` | the `greedy` rows (`--decoder`, `--max-level`); `--camus` reruns it on the 100 camus cells for parity |
| `run_resolvers.py` | NetCS and BROOQS on 20 networks × {raw, fixed, true} trees of blobs |
| `check_relaxed.py` | gates G0–G3 and G2b for the relaxed decoder at L1/L2/L3; pure Python |
| `make_csv.py` | builds the CSV and a summary JSON, with luay from PhyloNet and hybrid ancestry beside it; `--parity` runs the gates against the landed camus rows |

The one edit outside this directory is an additive key in `gnn_cpu_v1/pylib/opc/rl/camus.py`:
`PHASES["l3reg_n200"] = ("n200", 200)`. The `est_bp1000` arm was already there.

## Run it, from the repo root, in a terminal

Use your terminal rather than a Claude Code background task: long runs have been killed there
before (`blob_cpu_v1/README.md`). Every step is resumable, so an interrupted step resumes where it
stopped.

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar   # the default anyway

# --- staging and the gates: ALL FOUR ARE ALREADY DONE AND PASSING (2026-09-12) ---
# Re-runnable, and cheap now that their caches are warm.  Start here only if something changed.
$PY l3reg_eval/stage.py                                  # 17 s   -> OK, 20 networks, every gate
$PY l3reg_eval/run_greedy.py --camus                     # 352 s  -> 100 camus cells
$PY l3reg_eval/make_csv.py --parity                      # 47 s   -> PASS (a), (b), (c)
$PY l3reg_eval/check_relaxed.py                          # 150 s  -> PASS G0-G3, G2b (~225 s cold)

# --- the formal decoder ------------------------------------------------------
(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
   --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --tags _mix,_mix2L,_mix75 \
   --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs --run-name gnn_t1)
(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
   --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --tags _l2mix \
   --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs --run-name gnn_t1_l2)
$PY l3reg_eval/run_greedy.py

# --- the relaxed ladder: L1, L2, L3 ------------------------------------------
for L in 1 2 3; do
  (cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
     --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --tags _mix,_mix2L,_mix75 \
     --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs \
     --run-name gnn_t1_relaxedL$L --decoder relaxed --max-level $L)
  (cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
     --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --tags _l2mix \
     --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs \
     --run-name gnn_t1_l2_relaxedL$L --decoder relaxed --max-level $L)
  $PY l3reg_eval/run_greedy.py --decoder relaxed --max-level $L
done                                                     # ~1 h 40 for all eight GNN runs

# --- the resolvers and the CSV ----------------------------------------------
caffeinate -i $PY l3reg_eval/run_resolvers.py --jobs 8   # 60 tasks x 2 resolvers, ~30-60 min
$PY l3reg_eval/make_csv.py                               # ~2 min
wc -l results/rl/l3reg_n200_v1_per_network.csv           # expect 1061 (header + 1,060 rows)
```

If you prefer the runs spelled out rather than looped, the eight GNN run names are
`gnn_t1`, `gnn_t1_l2`, and `gnn_t1{,_l2}_relaxedL{1,2,3}`; the four greedy ones are `greedy` and
`greedy_relaxedL{1,2,3}`.

The GNN command needs some explanation, and the third point below has bitten this pipeline:

- `NO_REFERENCE.tsv` does not exist, on purpose. `--reference` holds the landed camus rows the
  driver checks its selections against, keyed by (phase, rep, arm, tag, seed). It has no
  `l3reg_n200` rows, so for these cells it can only ever match nothing; a missing file (read as
  empty) makes that explicit. `summary.txt` should say `0/0 pairs match`.
- The separate `--out`/`--run-name` keeps these runs apart from the camus runs in
  `gnn_cpu_v1/runs/`, whose `cam200_*` files would otherwise mark every cell here as done.
- **`_l2mix` must be its own run directory.** Resume is per *cell*, not per (cell, model)
  (`run_gnn_cpu.py:485-489`), so pointing `--tags _l2mix` at a directory that already holds the
  three-family cells skips all 20 and writes **no** `_l2mix` rows, silently and with exit 0.
  Separately, `--tags` defaults to every directory under `checkpoints/` and takes the *union* of
  seeds across tags, so a bare invocation dies looking for `_l2mix/s1/best.pt`.
- **A run directory holds one decoder.** Every cell JSON records its decoder, and resuming into a
  directory decoded differently is refused. Without `--run-name`, a relaxed run defaults to a name
  ending `_relaxedL<L>` — which `make_csv.py`'s `runs/gnn*/` glob does **not** match.
- To use more cores, shard with `--reps 0-9` / `--reps 10-19` and distinct run names.
- Smoke flags first, if you like: `run_gnn_cpu.py ... --reps 0`, `run_greedy.py --dry-run`,
  `run_resolvers.py --self-check` then `--keys 0 --variants fixed`, `check_relaxed.py --quick`.

### `_l2mix/s1` and `_l2mix75/s0` (2026-09-13)

Two checkpoints with no common seed set, run through `run_gnn_cpu.py --models` (see
`gnn_cpu_v1/README.md`) in their **own** run directories, over the same four decoders:

```bash
M=_l2mix/s1,_l2mix75/s0; N=gnn_t1_l2mix_s1_l2mix75_s0
A="--threads 1 --models $M --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
for L in 1 2 3; do
  (cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL$L --decoder relaxed --max-level $L)
done
$PY l3reg_eval/make_csv.py                            # 1,220 rows, "complete": true
```

`make_csv.py` already read each tag's required seeds off `gnn_cpu_v1/checkpoints/`, so it picks
up both. The 2026-09-13 rebuild carries all 1,060 earlier rows byte for byte and adds 160:
`ours/l2mix` seed 1 and `ours/l2mix75` seed 0, 20 each under the formal decoder and relaxed L1, L2
and L3. **The output counts below predate these rows.**

### `_l2mixha`: `ours/l2mixHA/selF1` and `ours/l2mixHA/selHA` (2026-09-14)

`gnn_cpu_v1/checkpoints/_l2mixha/s0/` holds two checkpoints of one run:
- `best.pt`, selected on F1 → `ours/l2mixHA/selF1`;
- `best_ha.pt`, selected on hybrid-ancestry error → `ours/l2mixHA/selHA`, recorded as tag
  `_l2mixha:best_ha` (see `gnn_cpu_v1/README.md`).

Both labels are in `common.GNN_LABELS`. `make_csv.py` now builds its required models from
checkpoint **files**, so both are required, over the same four decoders:

```bash
M=_l2mixha/s0,_l2mixha/s0/best_ha; N=gnn_t1_l2mixha
A="--threads 1 --models $M --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
for L in 1 2 3; do
  (cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL$L --decoder relaxed --max-level $L)
done
$PY l3reg_eval/make_csv.py                            # 1,380 rows, "complete": true
```

The 2026-09-14 rebuild carries all 1,220 earlier rows byte for byte and adds 160: each label × 4
decoders × 20 networks, all `ok`.

### `_l2mix_pair`: `ours/l2mix_pair/selF1` and `ours/l2mix_pair/selHA` (2026-09-14)

`gnn_cpu_v1/checkpoints/_l2mix_pair/s0/` is `_l2mix`'s data on the pair conv (`conv: pair`, see
`gnn_cpu_v1/README.md`). It ships two checkpoints: `best.pt` (epoch 24) → `ours/l2mix_pair/selF1`,
and `best_ha.pt` (epoch 17, tag `_l2mix_pair:best_ha`) → `ours/l2mix_pair/selHA`. Both labels are in
`common.GNN_LABELS`, and `make_csv.py` requires both, over the same four decoders:

```bash
M=_l2mix_pair/s0,_l2mix_pair/s0/best_ha; N=gnn_t1_l2mix_pair
A="--threads 1 --models $M --inputs ../l3reg_eval/inputs --cells ../l3reg_eval/CELLS.tsv --reference ../l3reg_eval/NO_REFERENCE.tsv --out ../l3reg_eval/runs"
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name $N)
for L in 1 2 3; do
  (cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py $A --run-name ${N}_relaxedL$L --decoder relaxed --max-level $L)
done
$PY l3reg_eval/make_csv.py                            # 1,540 rows, "complete": true
```

Two earlier deliveries were run and replaced. Both sets of runs sit outside the `runs/gnn*` glob:
- **Epoch 2** (one model twice) was partly run and stopped. It never reached this CSV. Its runs are in
  `runs_superseded/l2mix_pair_epoch2/`.
- **Epochs 7/6** produced a complete 1,540-row CSV. The runs are in
  `runs_superseded/l2mix_pair_epoch7_6/`, and the CSV and JSON in
  `results/rl/superseded/l2mix_pair_epoch7_6/`.

The 2026-09-15 rebuild on the **epoch-24/17** checkpoints ran relaxed L2 first. A 1,504-row
`--allow-partial` build landed while L3 was still running, then the complete one. That complete build
carries all 1,380 pre-pair rows byte for byte and adds 160: each label × 4 decoders × 20 networks,
all `ok`, every cell at epoch 24 or 17, and none of the epoch-7/6 pair rows survive.

## Gene-tree scaling: gt5k / gt10k (2026-09-21)

`l3reg_n200_gt5k_v1/` and `l3reg_n200_gt10k_v1/` are these 20 networks with 5,000 and 10,000 gene
trees and a recomputed tree of blobs. The run is `l2reg_eval/`'s, at level 3. **`l2reg_eval/README.md`
§ "Gene-tree scaling"** has the rationale, the wiring and the checks, and one command sheet for both
packages. This section covers what is specific to l3reg.

- **`REG_GT=gt5k|gt10k`** selects the dataset (`common.py`, the identical edit, so `diff -u` against
  `l2reg_eval/common.py` stays the drift check). Staging and runs go under
  `l3reg_eval/<REG_GT>/`, with phase `l3reg_n200_<REG_GT>` and CSV
  `results/rl/l3reg_n200_<REG_GT>_v1_per_network.csv`.
- **Scope.**
  - `_l2mix_pair` (selF1 and selHA) at **relaxed L3**, the decoder the plots use here, with
    `--m 0 --tau-frac 0.002`, so τ = 10 at 5k and 20 at 10k.
  - Greedy at level 1 and relaxed L3.
  - NetCS and BROOQS on the raw and fixed trees of blobs.
  - There is no τ = 2 arm beyond 1k: at 10k it passes the overlap graph's `EDGE_CAP`.
- **Staged 2026-09-21, every gate passing.**
  - Staged trees: 0.6 GB (5k) and 1.2 GB (10k).
  - The raw TOB differs from the fixed one on 11 of 20 (5k) and 10 of 20 (10k).
  - At rep00, |C| is 7,238 at 5k and 6,947 at 10k (10,750 at 1k, τ = 2).
- **Not run:** the grids. `gt*/runs/` is empty.

The l3reg lines of the sheet:

```bash
PY=gnn_cpu_v1/.venv/bin/python
REG_GT=gt5k caffeinate -i $PY -u l3reg_eval/run_resolvers.py --jobs 12
REG_GT=gt10k caffeinate -i $PY -u l3reg_eval/run_resolvers.py --jobs 8
for G in gt5k gt10k; do
  (cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 \
     --models _l2mix_pair/s0,_l2mix_pair/s0/best_ha --m 0 --tau-frac 0.002 \
     --decoder relaxed --max-level 3 --inputs ../l3reg_eval/$G/inputs \
     --cells ../l3reg_eval/$G/CELLS.tsv --reference ../l3reg_eval/NO_REFERENCE.tsv \
     --out ../l3reg_eval/$G/runs --run-name gnn_t1_l2mix_pair_ftau0.002_relaxedL3)
  REG_GT=$G $PY l3reg_eval/run_greedy.py
  REG_GT=$G $PY l3reg_eval/run_greedy.py --decoder relaxed --max-level 3
  REG_GT=$G $PY l3reg_eval/make_csv.py
done
```

Each finished dataset gives 220 rows:
- 20 each of `ours/l2mix_pair/selF1 [relaxed L3]` and `selHA [relaxed L3]`;
- 20 each of `greedy` and `greedy [relaxed L3]`;
- 60 TOB rows;
- 80 resolver rows.

**Resolver ceiling.** `k_pred` is still at most `n_l3 + n_l1` on the true TOB, and so is below k by
two per level-3 blob, however many trees the resolvers get.

## Decoders

Both drivers take `--decoder level1` (the default) or `--decoder relaxed --max-level L`. The
scores, the rank rule (descending score, ties to the lower index) and the budget k are the same;
what differs is what counts as a legal selection, and how a selection becomes a network. See
`l2reg_eval/README.md` and `opc.rl.relaxed`'s docstring for the definitions.

**What each decoder costs on these truths**, measured on the 20 true cluster sets (`check_relaxed.py`
re-derives all of it):

| | l2reg (level 2) | l3reg (level 3) |
|---|---|---|
| (L) fails on the true cluster set | 20 of 20 | **20 of 20** |
| (N3O) fails | 13 of 20 | **16 of 20** |
| closing the true set changes the network it stands for | 10 of 20 | **15 of 20** |
| the relaxed realiser rebuilds the truth exactly | 20 of 20 at L2 | **20 of 20 at L3** |

So L3 is the rung with no representational obstacle, and L1 and L2 are **both** inadequate. The
formal realiser is intact only on reps 02, 03, 04, 05 and 09 — mostly the `l3x1` cells with the
smallest k.

**But the cap is not what is costing F1 here.** `check_relaxed.py`'s R report, support-ranked over
all 20 cells, gives median F1 **0.8171** for the formal decoder, 0.8171 at L1, 0.8171 at L2 and
**0.8186** at L3 — a gain of 0.0015. Under *oracle* ranking the ladder is real (rep00 keeps 195
true clusters at L1, 197 at L2, all 201 at L3), so the decoder is not the binding constraint at
these scores: the scores are. That is the comparison the GNN rows exist to make.

### Two things G3 measures rather than asserts

Neither is the in-degree defect below: the old certificate held on every probe, and G3's numbers are
identical before and after the fix. They are the greedy rule:

- **A higher cap can keep *fewer* true clusters.** rep05 keeps 199 at L1, **197** at L2, 203 at L3.
  A larger legal set is not a better one: the decoder can accept a higher-ranked cluster early that
  blocks more later. l2reg's G3 gated this direction and passed only because 20 level-2 cells
  happened to comply.
- **At its own level the oracle need not keep everything.** rep18 keeps 201 of 202 at L3. `tr` is
  the true clusters that survive into `C_tau`, and H(a subset of a regular network's clusters) is
  not a subnetwork of that network, so there is no theorem to appeal to.

So G3 hard-gates only what is guaranteed and what the rows rest on — every state verifies, every
realised network is within its cap — and prints the two counts beside it.

### The in-degree recheck, fixed on 2026-09-12

The relaxed decoder rechecks a refused cluster only when an acceptance could have made it legal. Its
rule assumed an **in-degree** refusal — the cluster would give some node a third parent — could only
be lifted by accepting a cluster *nested* with it. That is false: a cluster that merely overlaps it,
inserted between a node inside it and both of that node's parents, takes the node from two parents
to one, and nothing requeued the refusal. Every decode still ended in a legal state, but not always
the one the decoder's rule defines: some stopped with a legal cluster refused, more took a different
path to a different, still-maximal selection.

`gnn_cpu_v1/pylib/opc/rl/relaxed.py` now also requeues, after each acceptance, the in-degree refusals
of every candidate containing a node whose in-degree changed (`_indeg_lifted`).

| true cluster sets, 20 truths × levels × 2 budgets × 14 rankings | l2reg | l3reg |
|---|---|---|
| decodes | 1,120 | 1,680 |
| **before:** differ from an exhaustive-recheck oracle | 4 (3 stuck) | 24 (8 stuck) |
| **after** | 0 | 0 |

It was never a level-3 defect; l2reg's G2 slices simply never reach a late acceptance that removes a
reticulation. `check_relaxed.py` **G2b** now holds the fix on the four decodes that exposed it, and
fails if any of them stops exposing the old behaviour.

**No landed row is known to have changed, so the `relaxed_L*` tags stay.**
- Under support ranking no acceptance ever removed a reticulation — 0 of 20,425 on l2reg and l3reg —
  so the new clause never fires. All 40 landed l2reg greedy relaxed rows re-derive exactly on the
  fixed library.
- All 1,500 landed GNN relaxed selections (400 on l2reg, 1,100 in the camus runs behind
  `camus_a104_per_network_l2mix.csv`) are maximal, and an exact replay of l2reg rep12's nine models —
  the driver unchanged, the scores recomputed — reproduces the landed rows and matches the oracle.
  For GNN rows that is evidence, not proof: no cell stores its scores, and a wrong path can still end
  maximal. A replay of every cell would settle it.

## Outputs

- `results/rl/l3reg_n200_v1_per_network.csv`: **1,060 rows** — 4 decoders × (3 families × 3 seeds ×
  20 + `_l2mix` × 1 seed × 20 + greedy × 20) = 880, plus 60 tree-of-blobs and 120 resolver rows.
  - **The camus eleven**, in camus order: `method, gene_trees, n, network_idx, seed, cluster_f1,
    luay, luay_norm, luay_norm_exact, k_pred, k_true`.
  - **Then the extras:** `decoder, key, cu, n_l3, n_l1, n_blobs_tob, tp, fp, fn, precision, recall,
    mass_pred, mass_truth, cluster_f1_rooted, status`.
  - **Then hybrid ancestry:** `ha_n_true, ha_n_pred, ha_fp, ha_fn, ha_fpr, ha_fnr,
    ha_balanced_error`.
  - **Row values:** `gene_trees = est_bp1000`; `network_idx` is the MANIFEST `rep`; `key` is the
    l3reg key. `decoder` is `level1`, `relaxed_L<L>`, or empty for the tree-of-blobs methods.
    Empty fields are `''`, as in camus_a104.
  - **The only column-name difference from `l2reg_n200_v1_per_network.csv` is `n_l2` → `n_l3`.**
- `results/rl/l3reg_n200_v1.json`: provenance, per-method medians, and the parity report.
- `l3reg_eval/runs/`: every cell JSON, with every network as eNewick, plus logs and timings.

| label | what |
|---|---|
| `ours/prior`, `ours/2x-data-full`, `ours/+n75` | GNN `_mix`, `_mix2L`, `_mix75`, seeds 0–2, each given `k_true` as its budget |
| `ours/l2mix` | GNN `_l2mix`, **one seed** — the only family trained on any level-2 label, so the only one with a prior reason to transfer here |
| `greedy` | greedy-by-support, same candidate set and budget |
| `... [relaxed L1/L2/L3]` | the same with `--decoder relaxed --max-level L` |
| `TOB-QMC` | TREE-QMC's tree of blobs as written, rooted with `root_tob_on_outgroup` (**the camus protocol**), scored as a tree |
| `TOB-QMC (fixed)` | `<key>.tob`: TREE-QMC's own contraction rule applied to the root branch it skips; differs from the raw output on 10 of 20 networks |
| `true TOB*` | the true tree of blobs (oracle) |
| `NetCS`, `BROOQS` | resolving the raw TOB. These keep camus's labels because they are camus's protocol |
| `NetCS (fixed TOB)`, `BROOQS (fixed TOB)` | resolving the fixed TOB |
| `NetCS*(true TOB)`, `BROOQS*(true TOB)` | resolving the true TOB (oracle) |

## Reading the numbers

- **`cluster_f1` = 1 is unreachable for the formal decoder, and now for relaxed L1 and L2 too** —
  see the table above. Only L3 can represent these truths. Every decoder gets `k_true` = 4–12 as
  its budget, further above anything the checkpoints saw at n = 200 in training than l2reg's 3–9
  was, and none of the four families was trained on level-3 labels.
- **BROOQS and NetCS are two reticulations short per level-3 blob, by construction.** Each places
  one reticulation per blob it resolves, so `k_pred ≤ n_blobs_tob`; on the true TOB that is
  `n_l3 + n_l1` while `k = 3·n_l3 + n_l1`. On rep16 (3 level-3 blobs, 3 level-1, k = 12) the
  ceiling is 6. l2reg lost only one per blob. Read a low `k_pred` here as the method's design, not
  as a failure to find reticulations. Neither method is given a budget.
- **Some NetCS networks have no OUT-rooted reading**, because OUT sits below one of their own
  hybrids (`status = out_not_rootable`). As in camus_a104, `cluster_f1` then holds the value read
  off the network as emitted and `luay` is empty. `cluster_f1_rooted` is the principled column, and
  it is empty there.
- **`luay` is PhyloNet `CMPNETS -m luay` and nothing else.** Both networks are OUT-free with gammas
  stripped. The same jar produced every landed luay, and `--parity` re-checks that on this machine.
- **`mass_pred`/`mass_truth` are measured with the jar too**: size = luay(N, star) + luay(N, star′)
  − 1 + n. The build prints `truth mass - (2n + 3k - 1)`, which must be `[0]` — it is, on all 20,
  so no l3reg truth has the size-2 cycle that made cam150 rep04 an exception (girth on the regular
  form is 5–7 here). `luay_norm_exact` = luay / floor((X_pred + X_truth) / 2) is at most 1 by
  construction; `luay_norm` uses the closed form n − 1 + floor(3(k_true + k_pred)/2).
- **Compare `luay_norm_exact` across CSVs, not raw `luay`.** The camus truths carried one extra
  unary node (mass 2n + 3k) and these do not (2n + 3k − 1).
- **`ha_*` is hybrid ancestry, the metric Dai & Molloy score NetCS with** (bioRxiv
  10.64898/2026.06.06.730607, Fig. 3): the taxa below at least one hybrid, as `ha_fpr` = |P − T| /
  |P|, `ha_fnr` = |T − P| / |T|, and their mean. A tree of blobs has no hybrids, so its `ha_fpr` is
  0/0 and is left empty, as is its `ha_balanced_error`; its `ha_fnr` is 1. `make_csv.py` computes
  it two independent ways and lists any network where they disagree — on this dataset's 20 truths
  they agree 20 of 20.
- **Cell files are named `l3reg_n200_repRR_est_bp1000.json`**: phase `l3reg_n200` and arm
  `est_bp1000`, both registered in `opc.rl.camus`. `make_csv.py` maps rep → key. Note that
  `CamusCell.key` is `n200_repRR` for cam200, l2reg and l3reg alike — **the phase name is the whole
  of the separation** between the three datasets' cells, runs and reference lookups.
- The candidate set is bigger than l2reg's: |C| = 10,724 at rep00 against 8,316, which is the extra
  reticulations showing up in the gene trees.

## Parity (`make_csv.py --parity`)

Everything in it is about the shared `realise` → luay path and the landed **camus** rows; none of
it touches l3reg. **Run it anyway**: `build` embeds `runs/parity.json` in the summary, so skipping
it leaves `"parity": null` and the artifact carries no evidence that its luay path is the landed
one. Two misses are known and explained, and each may differ only in the fields it explains:

- **cam150 rep04, any row:** `mass_truth`. The truth has a size-2 cycle, which the jar sizes
  differently from the landed number.
- **TOB-QMC cam200 rep15:** `luay` and `mass_pred`. The Mac tree of blobs is not the Linux one, so
  this is a different input, not a different metric.

**(a) GATE** the landed NetCS/BROOQS, TOB-QMC and true-TOB rows reproduce their luay and both
masses. **(b) GATE** the Mac GNN networks in `gnn_cpu_v1/runs/C4537CLFY0_t1` reproduce the landed
GNN luay and masses. **(c) GATE** greedy on the 100 camus cells reproduces the landed
tp/fp/`k_pred`/F1, its network's size and `luay_greedy` exactly, because it is deterministic.

**All three PASS here, measured 2026-09-12 in 47 s**, with 0 unexplained misses: (a) 576 of 583
exact, 7 known; (b) 891 of 900 exact, 9 known; (c) 99 of 100 exact on luay and both masses, 1
known, **and 100 of 100 exact on tp/fp/`k_pred`/F1**. `runs/parity.json` holds the record.

**(c) needs `run_greedy.py --camus` to have been run**, and there was nothing to borrow:
`l2reg_eval` kept no `runs/greedy_camus/` — its landed `parity.json` records `"c": null`, i.e.
skipped — so `--greedy-camus-run` has no useful target. Without those cells `make_csv.py --parity`
prints `(c) skipped` and still passes on (a) and (b). It is ~6 min, so it is worth having; this
directory's parity is the stronger of the two.

## When something goes wrong

- **`stage.py` fails:** it writes neither `CELLS.tsv` nor `stage.json`, and it names the gate that
  failed.
- **A resolver hangs:** Ctrl-C and re-run. Finished tasks, and finished methods inside a task, are
  kept. `runs/resolvers/work/<key>/<variant>/*.log` holds each tool's full log.
- **A resolver errors:** it becomes a row with `status = error: ...`, not a crash.
  `run_resolvers.py` exits 3 so it is not missed.
- **PhyloNet refuses a network:** that row gets `status = phylonet_failed` and empty luay columns.
  `cmpnets_many` falls back to one call per pair, so one bad string cannot cost the batch.
- **A driver refuses to resume:** the run directory was decoded with another decoder. Pick a new
  `--run-name`, or `--force` to overwrite.
- **`make_csv.py` refuses to write when a cell is missing:** it lists the missing cells, per
  decoder. `--allow-partial` writes what exists.
- **The CSV is short and nothing complained.** Only `level1` runs are *required*, so a relaxed run
  written to an unmatched directory name loses 200 rows quietly. `wc -l` is the terminal check.
