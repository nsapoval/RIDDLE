# netcs_eval: the NetCS benchmark corpus, staged

`netcs_bench_corpus/` is Dai & Molloy's NetCS benchmark: level-1 and level-2 networks at n = 50,
100 and 200, 50 networks each, at four ILS levels, so **1,200 conditions on 300 truths**. Each
condition has IQ-TREE gene trees and TOB-QMC trees of blobs (default and fast), plus TINNIK's
(n ≤ 100 only). Each network has a true tree of blobs. This directory runs, on every condition:
- the W10 GNN checkpoints from `gnn_cpu_v1/`, all four families;
- greedy-by-support;
- NetCS and BROOQS on TOB-QMC's tree of blobs.

The GNN and greedy each run under the formal decoder and relaxed L1 and L2. The result is one CSV
in the shape of `results/rl/l3reg_n200_v1_per_network.csv`.

It is `l3reg_eval/` for a different corpus, and deliberately a copy of it: `diff -u
l3reg_eval/<x>.py netcs_eval/<x>.py` shows every change.

## Decisions (user, 2026-09-12) — read before comparing rows

- **The corpus has no outgroup, and the GNN and greedy are handed the root.** Gene trees are
  unrooted with no OUT. `stage.py` inserts OUT on each gene tree's edge nearest a displayed true
  root. This is the **true-root oracle** of `common.py`. NetCS and BROOQS get no root.
- **NetCS and BROOQS run native and unrooted:** on the corpus's gene trees and its TOB-QMC
  default tree of blobs, exactly as shipped, and BROOQS runs **without** `--rooted/--dist-var`.
  That differs from every landed camus row. Their outputs, and the trees of blobs, are rooted by
  the same oracle **only to score** `cluster_f1` and `luay`.
- **Only TOB-QMC default is resolved.** TOB-QMC fast, TINNIK and the true tree of blobs appear as
  tree rows only.
- **Decoders:** formal (`level1`), `relaxed_L1` and `relaxed_L2`.

So on a GNN or greedy row `cluster_f1`/`luay` include a root nobody else had. On a resolver or
tree-of-blobs row they are an upper envelope of rooted accuracy. **Hybrid ancestry (`ha_*`)
needs no root** and is the one metric that compares every row on equal terms. It is the metric
Dai & Molloy score NetCS with.

**How good the oracle is, measured by `stage.py`.** Under ILS the true root split is often absent
from a gene tree, so the oracle picks the nearest edge:

| n | fraction of loci rooted at an exact displayed root (median by ILS 0.25 / 0.5 / 1.0 / 2.0) | mean taxa misplaced (median) |
|---|---|---|
| 50 | 0.02 / 0.10–0.12 / 0.23–0.29 / 0.33–0.34 | 2.6–6.2 |
| 100 | 0.01–0.02 / 0.08–0.12 / 0.20–0.28 / 0.20–0.28 | 5.5–12.9 |
| 200 | 0.01 / 0.05–0.07 / 0.09–0.14 / 0.10–0.12 | 12.7–21.0 |

Each row carries its condition's value as `trees_root_exact_frac`. `ils_<x>` is the corpus's
label: branch lengths scale with it (`ils_2.0` is 8 × `ils_0.25`), and ILS is highest at 0.25.

| file | what it does |
|---|---|
| `common.py` | paths and labels; truth normalisation; **the true-root oracle** (`root_targets`, `insert_out_at_truth`); pins BLAS and puts `gnn_cpu_v1/pylib` on the path |
| `stage.py` | builds `inputs/`, `tob/`, `CELLS.tsv`, `shards/` and `stage.json` behind blocking gates |
| `run_shards.sh` | the GNN and greedy runs, 12 shards in parallel, every decoder |
| `run_greedy.py` | the `greedy` rows (`--decoder`, `--max-level`, `--cells` for a shard) |
| `run_resolvers.py` | NetCS and BROOQS on 1,200 conditions, native and unrooted |
| `check_relaxed.py` | gates G0–G3 and G2b for the relaxed decoder at L1/L2 on these truths |
| `make_csv.py` | the CSV and a summary JSON, luay from PhyloNet (cached, parallel JVMs); `--parity` holds the jar to the landed camus rows |

**Two edits outside this directory:**
- `gnn_cpu_v1/pylib/opc/rl/camus.py` gains additive keys: the phases `netcs_L{1,2}_n{50,100,200}`
  and the arms `netcs_ils{0.25,0.5,1.0,2.0}`.
- `gnn_cpu_v1/run_gnn_cpu.py` gains `--m-from-trees`. The corpus ships 911–1,000 trees per
  condition, and without the flag the node features' `support / m` would be divided by 1,000
  regardless. The flag sets m to the file's own locus count, so the quantity stays a frequency.
  τ = 2 stays an absolute count. The flag is off by default and a no-op on any 1,000-tree file.
  **Regression-checked:** l3reg rep02 reproduces its landed `n_selected/tp/fp/k_pred/eNewick`
  exactly with and without it, for `_mix/s0` and `_l2mix/s0`. `run_greedy.py` always uses the
  per-file m; greedy reads no m-normalised feature. `SHA256SUMS` in `gnn_cpu_v1/` no longer
  matches those two files.

## Run it, from the repo root, in a terminal

Use your terminal rather than a Claude Code background task. Every step is resumable.

```bash
PY=gnn_cpu_v1/.venv/bin/python
export OPC_PHYLONET=~/research/stochastic-search/tools/PhyloNet.jar   # the default anyway

# --- staging and gates: ALL DONE AND PASSING (2026-09-12). Re-run only if something changed. ---
$PY netcs_eval/stage.py --jobs 12            # 233 s -> OK, 1,200 conditions, every gate
$PY netcs_eval/check_relaxed.py              # ~3 min -> PASS G0-G3, G2b
$PY netcs_eval/make_csv.py --parity          # 31 s -> PASS (a), (b), (c)

# --- 1. GNN + greedy: 12 shards x (6 GNN runs + 3 greedy runs) ------------------------------
caffeinate -i bash netcs_eval/run_shards.sh  # ~4-5 h wall; logs in netcs_eval/runs/logs/
grep exited netcs_eval/runs/logs/shard_*.log # must print nothing

# --- 2. the resolvers -- after step 1, or both runs compete for cores --------------------------
caffeinate -i $PY netcs_eval/run_resolvers.py --jobs 12   # 1,200 tasks, ~2 h wall

# --- 3. the CSV ------------------------------------------------------------------------------
$PY netcs_eval/make_csv.py --jobs 4          # ~15 min cold; minutes once runs/phylonet/cache.jsonl is warm
wc -l results/rl/netcs_bench_per_network.csv # expect 53601 (header + 53,600 rows) -- see below
```

### `_l2mix/s1` and `_l2mix75/s0` (2026-09-13)

Two checkpoints with no common seed set, run through `run_gnn_cpu.py --models` (see
`gnn_cpu_v1/README.md`) in their **own** run directories, `gnn_t1_l2mix_s1_l2mix75_s0{,_relaxedL1,
_relaxedL2}` -- resume is per cell, so the existing `gnn_t1_l2*` directories would skip every cell.
`run_shards.sh` now runs them as its third model set, and runs `gnn_t1_l2` as `--models
_l2mix/s0`, so a fresh run cannot score `_l2mix/s1` twice. On the landed runs they were launched
as 12 shards at `--threads 1`, one per `shards/CELLS_XX.tsv`, each over the three decoders:

```bash
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py --threads 1 --m-from-trees \
   --models _l2mix/s1,_l2mix75/s0 --inputs ../netcs_eval/inputs --cells ../netcs_eval/shards/CELLS_00.tsv \
   --reference ../netcs_eval/NO_REFERENCE.tsv --out ../netcs_eval/runs --run-name gnn_t1_l2mix_s1_l2mix75_s0)
#   and again with --decoder relaxed --max-level 1 / 2 and --run-name ..._relaxedL1 / ..._relaxedL2
```

`make_csv.py` reads each tag's required seeds off `gnn_cpu_v1/checkpoints/`, so a complete build
is now **53,600 rows**: 1,200 × (36 GNN + 3 greedy + 3 TOB + 2 resolver) + 800 TINNIK. The
46,400-row build of 2026-09-13 16:46 recorded `"complete": false` for exactly these two
checkpoints. The row counts elsewhere in this README predate them.

### `_l2mixha`: `ours/l2mixHA/selF1` and `ours/l2mixHA/selHA` (2026-09-14)

`gnn_cpu_v1/checkpoints/_l2mixha/s0/` holds two checkpoints of one run:
- `best.pt`, selected on F1 → `ours/l2mixHA/selF1`;
- `best_ha.pt`, selected on hybrid-ancestry error → `ours/l2mixHA/selHA`, recorded as tag
  `_l2mixha:best_ha` (`run_gnn_cpu.py --models _l2mixha/s0/best_ha`; see `gnn_cpu_v1/README.md`).

Both labels are in `common.GNN_LABELS`. `make_csv.py` builds its required models from checkpoint
**files**, so both are required. They ran as 12 shards at `--threads 1`, in
`gnn_t1_l2mixha{,_relaxedL1,_relaxedL2}`; `run_shards.sh` runs them as its fourth model set:

```bash
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py --threads 1 --m-from-trees \
   --models _l2mixha/s0,_l2mixha/s0/best_ha --inputs ../netcs_eval/inputs --cells ../netcs_eval/shards/CELLS_00.tsv \
   --reference ../netcs_eval/NO_REFERENCE.tsv --out ../netcs_eval/runs --run-name gnn_t1_l2mixha)
#   and again with --decoder relaxed --max-level 1 / 2 and --run-name ..._relaxedL1 / ..._relaxedL2
```

**A complete build is now 60,800 rows:** 1,200 × (42 GNN + 3 greedy + 3 TOB + 2 resolver) + 800
TINNIK. The 2026-09-14 rebuild carries all 53,600 earlier rows byte for byte and adds 7,200: each
label × 3 decoders × 1,200 conditions, all `ok`.

### `_l2mix_pair`: `ours/l2mix_pair/selF1` and `ours/l2mix_pair/selHA` (2026-09-15)

`gnn_cpu_v1/checkpoints/_l2mix_pair/s0/` is `_l2mix`'s data on the **pair conv** (`conv: pair`,
containment; see `gnn_cpu_v1/README.md`). It ships two checkpoints: `best.pt` (epoch 24) →
`ours/l2mix_pair/selF1`, and `best_ha.pt` (epoch 17, tag `_l2mix_pair:best_ha`) →
`ours/l2mix_pair/selHA`. The run used 12 shards at `--threads 1` into `gnn_t1_l2mix_pair{,
_relaxedL1,_relaxedL2}`, and `run_shards.sh` runs it as its fifth model set:

```bash
(cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py --threads 1 --m-from-trees \
   --models _l2mix_pair/s0,_l2mix_pair/s0/best_ha --inputs ../netcs_eval/inputs --cells ../netcs_eval/shards/CELLS_00.tsv \
   --reference ../netcs_eval/NO_REFERENCE.tsv --out ../netcs_eval/runs --run-name gnn_t1_l2mix_pair)
#   and again with --decoder relaxed --max-level 1 / 2 and --run-name ..._relaxedL1 / ..._relaxedL2
```

The pair forward costs ~2.5× the segment one, so this model set takes ~4 h wall on 12 shards.
Two earlier deliveries were run and replaced. Both sets of runs sit outside the `runs/gnn*` glob:
- **Epoch 2** (one model twice) was partly run and stopped. Its runs are in
  `runs_superseded/l2mix_pair_epoch2/`.
- **Epochs 7/6** produced a complete 68,000-row CSV. The runs are in
  `runs_superseded/l2mix_pair_epoch7_6/`, and the CSV and JSON in
  `results/rl/superseded/l2mix_pair_epoch7_6/`.

The epoch-24/17 run was restarted part way to put **relaxed L2 first** (user, 2026-09-15). Resume is
per cell, so nothing finished was lost. A 66,056-row `--allow-partial` build landed with relaxed L2
complete, then the complete one.

**A complete build is now 68,000 rows:** 1,200 × (48 GNN + 3 greedy + 3 TOB + 2 resolver) + 800
TINNIK. The 2026-09-15 rebuild carries all 60,800 earlier rows byte for byte and adds 7,200: each
label × 3 decoders × 1,200 conditions, all `ok`, every cell at epoch 24 or 17. **100 of the 7,200 lines
repeat an epoch-7/6 line byte for byte.** A CSV line carries no epoch, so wherever both deliveries
chose the same clusters on a condition the two lines are identical.

**Timings** come from the smokes below, taken with other jobs running, so they are on the
pessimistic side:

| stage | per cell, n = 50 / 200 | total CPU |
|---|---|---|
| GNN `gnn_t1` (9 models) | ~20–28 s / ~58 s | ~13 h per decoder |
| GNN `gnn_t1_l2` (1 model) | ~5 s / ~14 s | ~3 h per decoder |
| greedy | ~2–3 s / ~8 s | ~1.4 h per decoder |
| NetCS + BROOQS | ~16 s / ~130 s (BROOQS ~117 s at n = 200) | ~21 h |

**Peak RSS:** ~1.6 GB per GNN process and ~1.2 GB per greedy process, so 12 shards fit in 64 GB
with room.

**If you'd rather not use the script,** one shard, one decoder is:

```bash
(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 --m-from-trees \
   --inputs ../netcs_eval/inputs --cells ../netcs_eval/shards/CELLS_00.tsv --tags _mix,_mix2L,_mix75 \
   --reference ../netcs_eval/NO_REFERENCE.tsv --out ../netcs_eval/runs --run-name gnn_t1)
#   the same with --tags _l2mix --run-name gnn_t1_l2
#   and both again with --decoder relaxed --max-level 1 --run-name gnn_t1_relaxedL1 (and ..._l2_relaxedL1), and L2
$PY netcs_eval/run_greedy.py --cells netcs_eval/shards/CELLS_00.tsv            # and --decoder relaxed --max-level 1 / 2
```

The traps are the ones `l3reg_eval/README.md` spells out:

- **`--m-from-trees` must be on every GNN invocation.** A run without it is a different
  experiment, and nothing downstream notices. Each cell JSON records `m`.
- **`NO_REFERENCE.tsv` does not exist, on purpose.** `summary.txt` says `0/0 pairs match`.
- **`_l2mix` must be its own run directory.** Resume is per cell, not per (cell, model).
- **A run directory holds one decoder.** A relaxed run must be named `gnn*_relaxedL<L>`, which
  `make_csv.py`'s `runs/gnn*/` glob matches.
- **The phase name is the whole of the separation** between datasets, as for l2reg and l3reg.
  `CamusCell.key` (`n50_rep1590`) is shared across levels and ILS levels. Cell files are
  `netcs_L<l>_n<n>_rep<id>_netcs_ils<x>.json`.
- **Shards share run directories.** Cells are atomic, but `timings.tsv`/`summary.txt` are rewritten
  by whichever invocation finishes last. To regenerate them, re-run any one shard's command once
  everything is done: every cell is skipped, and the files are rebuilt from all cells.

**Smoke flags:**
- `run_gnn_cpu.py ... --dry-run`
- `run_greedy.py --dry-run`
- `run_resolvers.py --self-check`, then `--keys L2_n50_net1_ils1.0 --run-name smoke`
- `check_relaxed.py --quick`
- `stage.py --only L2_n50 --limit 2`

## What was run here, and what was not

Run on 2026-09-12:
- `stage.py`, full: every gate passes.
- `check_relaxed.py`: PASS.
- `make_csv.py --parity`: PASS, identical to l3reg's. (a) 576/583 exact, 7 known; (b) 891/900,
  9 known; (c) 99/100 on luay and masses, 100/100 on tp/fp/k_pred/F1. Gate (c) borrows
  `l3reg_eval/runs/greedy_camus`.
- The l3reg regression above.
- Smokes, **into the scratchpad, not `runs/`**, on three edge-case conditions:
  - `L2_n50_net1_ils1.0`, a plain L2;
  - `L1_n50_net1147_ils0.25`, a size-2-cycle truth;
  - `L2_n200_net26_ils1.0`, a double-donor truth.

  On those, GNN `gnn_t1` formal and relaxed L2 and `gnn_t1_l2` formal, greedy at all three
  decoders, and both resolvers all exited 0, and `make_csv.py --keys ... --allow-partial` built 80
  rows from them. All 80 rows are `ok`, every luay is filled, hybrid ancestry's two derivations
  agree 83/83, and the truth-string gate passes.

**Not run:** any full GNN, greedy or resolver run, and the full `make_csv.py`. `runs/` holds only
the staging log, `check_relaxed`'s cache and log, `parity.json`, and a PhyloNet cache warmed by
the smoke (entries are pure functions of the two strings, so they stay valid).

## The corpus, measured by `stage.py`

- **Topology is ILS-free.** All four ILS files of every network have one topology, so there is one
  truth per network. The true tree of blobs ships once per network (`ils_1.0`).
- **Taxon counts are n..n+5**, with labels `t*`/`l*`. `n` in a phase name and the `n` column are
  nominal; `n_taxa` is the count, and `luay_norm` uses it.
- **Unary roots.** The n = 50/100 truths are written with a unary root edge, which staging removes.
  PhyloNet sizes a unary node like any other, so leaving it in would cost every prediction a node.
- **Four truths needed care:**

  | truth | what | how |
  |---|---|---|
  | `L2_n100` 1011, `L2_n200` 26 | one node donates to two hybrids — outside `networks.Network` | the bare `#H181` / `#H305` tag wrapped in a unary node; clusters, k and level unchanged, checked against an independent parser. The wrap is also applied to every other reticulate truth as a control. |
  | `L1_n50` 1147, 1549 | a size-2 cycle; `to_enewick` cannot round-trip it | nothing to fix for clusters (it is invisible to them, A11). luay uses the corpus string. |

- **Truth strings for PhyloNet.** Every luay compares against the corpus's own string: unary root
  removed, gammas stripped (`true_net.phylonet.nwk`). On every other truth,
  luay(`to_enewick(truth)`, corpus string) must be 0; `make_csv.py` gates this.
- **Gene trees:** 911–1,000 per condition. Every locus has branch lengths and exactly the truth's
  taxa. Every staged line roots, through `reroot_on_outgroup`, exactly where the oracle put OUT.
  `prepare_gene_trees` runs clean on all 1,200 files. IQ-TREE's support labels are dropped.
- **Trees of blobs:** all 3,500 parse and carry the truth's taxa. Rooted by the oracle, **the true
  tree of blobs is not always all-true:** median 1 false cluster on L2 truths, 0 on L1. That
  happens when the true root lies inside a blob, so no edge of the tree of blobs is a root.

## What `check_relaxed.py` found

- **Two truths are not Hasse diagrams of in-degree ≤ 2 at all:** `L2_n100` 10105 and 10365. Each
  has one cluster with three covering parents in H(C(N)). No relaxed decode at any cap can
  represent them, and neither can the formal one. G1 reports these; any other refusal fails it.
- **From their own clusters, 38 truths come back with a different k** — always fewer, i.e. a
  reticulation no cluster witnesses; for example the size-2 cycles. **7 come back at a different
  level:** two L1 truths come back as trees (level 0), three L2 truths as level 1, and `L2_n50`
  1284 as level 3, which no L2 cap can represent.
- **The formal realiser builds the same network as the relaxed one on 150 of 150 L1 truths.**
- **Under oracle ranking, the cell's own level keeps every true candidate on 24 of 24 sampled
  cells.** Under support ranking the ladder barely moves: median F1 is formal 0.8298, L1 0.8298,
  L2 0.8298 on L1 cells, and 0.7538 at every rung on L2 cells. As on l3reg, the scores bind, not
  the decoder.

## Outputs

`results/rl/netcs_bench_per_network.csv` has **46,400 rows**. Per condition:
- GNN: (3 families × 3 seeds + `_l2mix` × 1) × 3 decoders = 30;
- greedy: × 3 decoders = 3;
- tree rows: TOB-QMC, TOB-QMC-fast and true TOB = 3, plus TINNIK at n ≤ 100;
- resolvers: NetCS and BROOQS = 2.

That is 1,200 × 38 + 800.

**Columns:**
- **The camus eleven:** `method, gene_trees, n, network_idx, seed, cluster_f1, luay, luay_norm,
  luay_norm_exact, k_pred, k_true`. Here `gene_trees = netcs_iqtree_trueroot`, `n` is nominal,
  and `network_idx` is the corpus network id.
- **Then:** `decoder, key, phase, arm, level, ils, n_taxa, n_loci, shortcut_retics,
  trees_root_exact_frac, root_cost, n_blobs_tob, tp, fp, fn, precision, recall, mass_pred,
  mass_truth, cluster_f1_rooted, status`. `root_cost` is the taxa between the edge a resolver or
  tree-of-blobs row was rooted on and a displayed true root; it is empty for GNN and greedy rows.
- **Then hybrid ancestry:** `ha_n_true, ha_n_pred, ha_fp, ha_fn, ha_fpr, ha_fnr, ha_balanced_error`.
- `key` is `L<l>_n<n>_net<id>_ils<x>`.

`results/rl/netcs_bench.json` holds provenance, the parity report, per-method medians, and the same
medians per (level, n, ILS) stratum. `netcs_eval/runs/` holds every cell JSON, every network as
eNewick, and logs and timings.

| label | what |
|---|---|
| `ours/prior`, `ours/2x-data-full`, `ours/+n75` | GNN `_mix`, `_mix2L`, `_mix75`, seeds 0–2, each given `k_true` as its budget |
| `ours/l2mix` | GNN `_l2mix`, seeds 0–1 — trained on level-2 labels |
| `ours/l2mix75` | GNN `_l2mix75`, seed 0 — `_l2mix`'s phases plus `x75` |
| `greedy` | greedy-by-support, same candidate set and budget |
| `... [relaxed L1/L2]` | the same with `--decoder relaxed --max-level L` |
| `TOB-QMC`, `TOB-QMC-fast`, `TINNIK` | the corpus's inferred trees of blobs, oracle-rooted, scored as trees |
| `true TOB*` | the corpus's true tree of blobs (oracle), oracle-rooted |
| `NetCS`, `BROOQS` | resolving TOB-QMC default, native and unrooted; oracle-rooted to score |

## Reading the numbers

- **Neither resolver is told k,** and each places one reticulation per blob it resolves. The GNN
  and greedy get `k_true` = 1–5.
- **`out_not_rootable`:** a resolver network the oracle's OUT cannot root keeps `cluster_f1` as
  emitted, and leaves `cluster_f1_rooted` and `luay` empty. The re-rooting uses only edges
  outside every hybrid's subtree, so it should be rare.
- **luay is PhyloNet `CMPNETS -m luay` and nothing else.** Masses come from two star probes per
  network. `luay_norm_exact` ≤ 1 is gated. **Compare `luay_norm_exact` across CSVs, not raw luay.**
- **Hybrid ancestry on a tree of blobs:** no hybrids, so `ha_fpr` and `ha_balanced_error` are
  empty and `ha_fnr` is 1. On L1 cells a single hybrid can cover few taxa, so `ha_*` is coarse
  there.

## When something goes wrong

- **`stage.py` fails:** it writes no table, and names the gate.
- **A shard step fails:** its log says `exited`, and the shard moves on. Fix the cause and re-run
  `run_shards.sh <shard>`; finished cells are skipped.
- **A resolver hangs:** Ctrl-C and re-run. `runs/resolvers/work/<key>/*.log` holds each tool's log.
  An error becomes `status = error: ...`, and `run_resolvers.py` exits 3.
- **PhyloNet refuses a network:** `status = phylonet_failed`. Failed pairs are not cached, so the
  next build retries them.
- **`make_csv.py` refuses to write:** it lists what is missing per decoder and tag.
  `--allow-partial` writes what exists. A complete build prints `46400 rows (46400 expected)`.
- **These defects belong in the real `SPEC-AMENDMENTS` register** (the Linux box), not in the local
  prefix: the double-donor truths, the size-2-cycle truths, the two unrealisable truths, and the
  oracle's cost.
