# sptree_eval — RIDDLE across k = 0..10 and ASTRAL-IV on species-tree data (n = 200)

`sptree_n200_gt10k_v1/` is 50 **species trees** on 200 taxa, with **no reticulation**. Each tree comes with 10,000
IQ-TREE gene trees estimated from 1000 bp AliSim alignments. The questions here:

- **RIDDLE across k.** What does RIDDLE do on tree-like data when it is made to spend k = 0..10 reticulations?
- **RIDDLE against ASTRAL-IV.** How does RIDDLE at k = 0 compare with ASTRAL-IV?

Both questions are asked at three gene-tree sizes: the first 1k, the first 5k, and all 10k trees.

Resolvers (BROOQS / NetCS on a tree of blobs) come later, when TOBs exist; see [Later: resolvers](#later-resolvers).

## Design

| | |
|---|---|
| **RIDDLE** | `GNN MAP [z+tc]` under the **train-fit β** (`nni_ensemble/runs_trainfit/calib/train__z6.json`, sha256 `f47199e9…`; `credible --calib-pool train`), on the **relaxed level-1 decoder** (`--decoder relaxed --max-level 1`, tag `relaxed_L1`) |
| side rows | `GNN [relaxed L1]` (the emitted decode RIDDLE starts from), `GNN strict L1` (the formal `level1` decoder, same forward pass, no MAP), `greedy [relaxed L1]` (ranked by support) |
| **k** | the decode budget, swept **0..10**. It is not the truth's k, which is 0 everywhere. |
| gene trees | estimated arm only: `estimated_bp1000/<key>.raw.nwk.gz`, rooted on `OUT` by `camus.prepare_gene_trees` (arm `est_bp1000`, as in every landed benchmark) |
| sizes | the first **1k / 5k / 10k** lines, τ = **2 / 10 / 20** (tau-frac 0.002, as in the landed gene-tree scaling runs). The OUT graft was placed using all 10k loci (`graft_n_loci`), so each prefix is the same trees, not a re-estimate. |
| **ASTRAL-IV** | `~/miniconda3/envs/phylo-env/bin/astral4`, ASTER **v1.22.4.6**. Runs `-t 1 --seed 233 -u 1 -r 4 -s 4 --genelength 1000` (all of these are the defaults, written out) on the same staged prefix bytes (sha256-checked). The tree is rooted on `OUT` in Python with the function that roots the gene trees, and OUT is pruned. |
| checkpoint | `_l2mix_pair/s0/best.pt` (`ours/l2mix_pair/selF1`), as everywhere |

**The data.** The 50 cells are `n200_rep{00..49}_cu{0.1,0.25,0.5}`.
- cu is read from the key. The split is **20 / 15 / 15**, not "by rep mod 3".
- Each truth is binary and ultrametric with **198** non-trivial clusters, so `n_true = 198`, not 199.
- Gene-tree RF to the true gene trees ranges from 0.068 (rep16, cu 0.5) to 0.205 (rep17, cu 0.1).
- Rooting on OUT puts the true root split in 77–90% of gene trees (`frac_root_split_correct`). This is a property of the arm, and the landed benchmarks share it.

The dataset still lacks README, MANIFEST, SHA256SUMS, `code/` and `tob_bp1000/`. `stage.json` records our own sha256 of
every source file. `stage.py --verify-sums` checks the dataset's SHA256SUMS once one ships.

## Layout

```
sptree_eval/
  sp_common.py        paths, constants, prefix bytes (no torch)
  stage.py            D0 integrity + S1-S3 -> inputs/, CELLS.tsv, stage.json
  run_astral.py       ASTRAL-IV, one tree per (size, cell)
  check_sptree.py     cmp (G-PUB, G-SWEEP), sweep (G-K), kernels (tail gate + kernel routing), cred (C2/C3), status
  collate.py          the tables and gates C1-C8
  sweep.py            RIDDLE (sweep): the k = 1..5 majority vote; hardwired RF on every row; gates W1-W4
  runtime.py          seconds per network: RIDDLE (k=0), RIDDLE (sweep), ASTRAL-IV; gates T1-T4
  run_sptree.sh       the launcher: score | probe | rest | astral | collate | status
  astral_queue.sh     ASTRAL-IV as a staged queue (1k -> 5k -> 10k) on the free P-cores
  inputs/n200/RR/     true_net.nwk ((OUT:1,<tree>:0);) and est_bp1000.nwk (the raw trees, decompressed)
  runs/anat/<ds>/cells/        scored cells (score_ksweep)            -- NE_ANAT_RUNS
  runs/ne/<ds>/{feat,cred}/    features / credible                    -- the NE runs root
  runs/ne/probe/, runs/ne/calib/train__z6.json (a symlink, sha-pinned)
  runs/astral/<size>/<stem>/   species.tre, species.rooted.nwk, astral.log, run.json
  runs/logs/                   every step's log
  results/                     sptree_rows.csv, sptree_summary.{csv,txt}, sptree_gates.json;
                               sptree_sweep_rows.csv, sptree_sweep_gates.json (sweep.py);
                               sptree_runtime.csv, sptree_runtime_gates.json (runtime.py)
cluster_anatomy/score_ksweep.py   score_cells over k with one build per (size, rep)
```

## Registry, and what k means

Each (size, k) pair is its own registered dataset, **`sp{1k,5k,10k}_k{00..10}`**: 33 in all, from `cluster_anatomy/common.py:_sptree()`.
They are registered once `CELLS.tsv` exists. Their properties:
- `kind = "sptree"`; no `base`, no landed run (parity `ok: None`), and no calibration pool or fold.
- `extra = {m, tau, k_decode, tc_kernel: block, size}`.
- Cell stems stay `sptree_n200_repRR_est_bp1000`, so every reader keyed on stems is unchanged.
- `nni_ensemble/ne_common.SPTREE` puts them in `ALL_DATASETS`.

**`extra["k_decode"]` replaces the truth's k in `scoring.build_cell`.** That k plays two roles:
- the GNN's global features `(k, k/n)`, so **every k is its own forward pass**;
- the decode's budget.

Everything downstream reads `meta["k"]` from the scored cell: features, credible's legality check (`r ≤ k`) and the MAP.
No step re-derives k from the truth.

**What is shared across k.** The build (prepare, count, `C_tau`, overlap graph, node features) does not depend on k, so
`cluster_anatomy/score_ksweep.py` builds once per (size, rep) and calls `scoring.at_k(cell, k)` for each k.

**On the β.** β's column `k` is a per-option **atom count** (`features.py:178`), not the reticulation number. So
applying β at k = 0..10 raises no range question. The GNN itself trained up to k = 7. k = 8–10 is extrapolation in
the global feature, though inside landed inference (l3reg went to 12).

**Edits to shared code, all off by default:**
- `scoring.py`: `k_decode` and `at_k`;
- `score_cells.score_one`: `cell=` / `build_seconds=`, plus `k_true` written only under `k_decode`;
- `common.py`: `_sptree`, plus the `stratum` and `true_trees` branches;
- `ne_common.py`: `SPTREE`;
- `gnn_cpu_v1/pylib/opc/rl/camus.py`: phase `sptree_n200`.

These directories are untracked in git. The pre-edit copies are snapshotted, and **G-PUB** below shows published
outputs re-derive unchanged.

## Gates

| gate | where | what |
|---|---|---|
| **G-PUB** | `check_sptree.py cmp` | Before the edits, `score_cells` re-scored l2reg_gt10k rep0 (`--check-forward`) and trn_x25 rep700 (a k = 0 cell with a level1 decode). After the edits, it re-scored them again and ran `features` + `credible --calib-pool train` on l2reg_gt10k rep0. Before equals after (JSON and npz). After equals the published cells, except `forward_check` (`True` with `--check-forward`, `None` in the published run). feat and cred equal `nni_ensemble/runs/l2reg_gt10k/feat` and `runs_trainfit/l2reg_gt10k/cred`. **Passed 2026-10-28.** |
| D0 / S1–S3 | `stage.py` | **D0:** integrity, including in-flight-copy detection (`--check-only` polls it). **S1:** the truth, which has 198 clusters and 0 reticulations. **S2:** every staged tree, rooted on OUT, is **string-identical** to the shipped `.len.nwk.gz` line, and its topology is the `.nwk.gz` line's. **S3:** names resolve. **Passed: 50/50 cells, 500,000 trees.** |
| **G-SWEEP** | `check_sptree.py cmp` | A swept cell (built at k = 0, scored at k = 3 via `at_k`) equals a fresh `score_cells --dataset sp1k_k03` cell. Both runs pass `--check-forward`, which also checks G2: p is run_gnn_cpu.forward's, bit for bit. **Passed.** |
| G-K | `check_sptree.py sweep` (end of `score`) | Every scored cell has `k == k_decode`, `k_true == 0`, and `0 ≤ k_pred ≤ k` for all four decodes. |
| tail | `check_sptree.py kernels` (`probe`, `rest`) | Every cell sits inside `derived.TAIL` (`max_a` 2,021, `max_m` 113, `radius_sets` 2,042,220). A cell with a block past `BLOCK_MAX_M = 24` atoms runs `features --tc-kernel dense`; dense equals block (G-TC). |
| N0–N5 | features / credible | as published. N1 at **L = 1** is new territory: relaxed_L1 had never gone through credible before this. |
| C2 / C3 | `check_sptree.py cred` (`rest`) | Every cred is pool `train`, fit `transfer`, with no problems and no warnings, and its emitted F1 equals the scored cell's. |
| C1–C8 | `collate.py` | The gates listed in its docstring: counts, cred hygiene, emitted parity, budgets, unmoved MAP = emitted, k = 0 is a tree, truths, ASTRAL A1. |

## Metrics (`results/sptree_rows.csv`)

There is one row per (size, k, rep, method); ASTRAL-IV rows have an empty `k`.

- `tp fp fn n_sel precision recall f1`: the **selection's** clusters against the 198 true clusters. This is the landed
  definition, `Instance.cluster_counts`. For ASTRAL these are its tree's clusters.
- `net_tp net_n net_f1`: the same counts for the realised network's hardwired clusters. They differ from the selection
  counts only where the strict-L1 closure adds clusters.
- `k_pred`, `level`: of the realised network.
- `rf`, `rf_norm`: unrooted RF on the 200 taxa, divided by `2(n-3)` = 394. Only filled when the prediction is a tree.
- `map_moved`, `p_map`: RIDDLE only.
- `luay`, `luay_norm_exact`: with `LUAY=1`, from the PhyloNet jar, cached in `runs/ne/phylonet/cache.jsonl`.
- **No HAE.** HAE is undefined against a truth with no hybrid.

`sptree_summary.csv` gives mean / median / se by size × k × method × cu (with `all` as one cu value).
`sptree_summary.txt` is the readable table.

## RIDDLE (sweep) (`sweep.py`)

**One network per k sweep, by the fungi events rule** (`fungi/scripts/fz_events.py`), without taxonomy anchors:
- A reticulation is its signature `(H, {P1∖H, P2∖H})`: the hybrid cluster and the unordered pair of sister sets,
  read as `fz_consensus.read_network` reads it.
- A signature is kept when it is in at least `MIN_VOTES = 3` of RIDDLE's networks at `VOTE_KS = 1..5`.
- The kept signatures sit on RIDDLE's own k = 0 tree: each adds `H`, `A ∪ H` and `B ∪ H`. The sweep is that cluster
  set, scored like any other row.

**What the vote keeps.** It keeps 1–4 reticulations per cell and never 0 (1k + 5k: 1 in 4 cells, 2 in 30, 3 in 61,
4 in 5). Against a tree truth every one of them is false. Mean F1 is 0.928 / 0.942 at 1k / 5k, against RIDDLE
k = 0's 0.931 / 0.947.

**nRF on networks (`rf_hw`, `rf_hw_norm`).** Every hardwired cluster is read as an unrooted split, over `2(n-3)`.
On a tree this equals collate's `rf`, which W2 checks.

**Gates:**
- **W1:** clusters re-read from every eNewick give collate's `(net_tp, net_n)`.
- **W2:** `rf_hw == rf` on every tree.
- **W3:** one sweep row per complete sweep, on a tree backbone.
- **W4:** the `hardwired_clusters` signature reader agrees with an independent recursive walk.

## Costs (measured on the Mac, 2026-10-28)

The full grid ran 2026-10-28/29 on the M4 Max (12 P-cores + 4 E-cores, 64 GB). Every worker is single-threaded
(BLAS/torch pinned, `astral4 -t 1`). RIDDLE ran with 6 workers beside up to 5 ASTRAL-IV runs, so at most 11 busy
cores; a 1/min sampler (`runs/logs/full_sampler.tsv`) saw a mean of 7.4 and a maximum of 11.0 cores busy, with
free memory never below 43%. So these timings are uncontended.

| step | wall | CPU | peak RSS / process | notes |
|---|---|---|---|---|
| stage | 75 s (8 jobs) | — | small | `inputs/` 3.0 GB |
| score, 1,650 cells | 62.3 min, `MAXJ=6` | 4.9 h user + 1.2 h sys | 3.15 GB | `runs/anat` 210 MB |
| probe | 32 s | — | small | every cell inside the tail |
| rest, 1,650 cells | 130.8 min, `JOBS=6` | 9.1 h user | 4.2 GB | 84 cells on the dense kernel (13 / 34 / 37 at 1k / 5k / 10k); `runs/ne` 187 MB |
| collate `--luay` | 53 s | 4.1 min | 2.3 GB | luay on 6,600 rows |
| ASTRAL-IV, 1k | 8.5–9.9 min per run | = wall | 0.6 GB | |
| ASTRAL-IV, 5k | 1.86–2.05 h per run | = wall | 2.8 GB | |
| ASTRAL-IV, 10k | **3.82–4.41 h** per run (median 4.04, all 50) | = wall | 4.9 GB | A first rep17 run timed out at 3 h in the subsample stage, after all 4 rounds; hence `--timeout` 12 h. The queue's memory cap allows 8 at once. |

Per cell (`results/sptree_timings.txt`, medians in seconds, 1k / 5k / 10k):
- build, once per rep: 8.8 / 16.0 / 29.8;
- forward, per k: 19.6 / 7.6 / 6.9. 1k is the slowest because it has the largest `C_tau`: τ = 2 keeps more clusters;
- decode: 1.5 / 0.9 / 0.8;
- features, block kernel: 2.8 / 13.3 / 27.3;
- features, dense kernel: 21.5 / 105 / 213;
- credible: about 0.5 everywhere.

All of it comes to 16.4 h of per-cell compute.

ASTRAL-IV is the dominant cost: about 6.4 CPU-h per rep, 318 CPU-h measured for the 150 runs. It is not noise-driven:
the clean and the noisy rep differ by about 10% at 1k and at 5k.

## Command sheet

Run from the repo root in zsh. Each line is resumable: after a stop, rerun the same line and only the missing cells
run. Every step fails loudly, and a step with nothing to do counts as a failure.

```zsh
cd /Users/nsapoval/research/rl-net-opt
caffeinate -i bash sptree_eval/run_sptree.sh score
caffeinate -i bash sptree_eval/run_sptree.sh probe
caffeinate -i bash sptree_eval/run_sptree.sh rest
LUAY=1 bash sptree_eval/run_sptree.sh collate
gnn_cpu_v1/.venv/bin/python sptree_eval/sweep.py
gnn_cpu_v1/.venv/bin/python sptree_eval/runtime.py
```

ASTRAL-IV is independent of the GNN steps and can run in a second terminal at the same time. Use the queue: it runs
one stage per gene-tree count (1k, then 5k, then 10k), takes only the free P-cores, and leaves 6 to RIDDLE while that
is running. The 10k stage waits for the 10k smoke to pass:

```zsh
cd /Users/nsapoval/research/rl-net-opt
caffeinate -i bash sptree_eval/astral_queue.sh
```

To drain it, `touch sptree_eval/runs/logs/astral_queue/STOP`. That launches nothing more and lets the running units
finish. Rerunning the queue resumes it.

What each line does:

| line | does | notes |
|---|---|---|
| `score` | `score_ksweep` with one job per rep, the noisiest gene trees first, `MAXJ=5` at once; then the summaries, `check scored` and G-K | logs in `runs/logs/score_repRR.log` |
| `probe` | `probe_blocks` on all 33 datasets, then the tail gate and kernel routing (`runs/logs/kernels_<ds>.txt`) | If a cell is past `derived.TAIL`, it stops: smoke that cell alone before `FORCE_TAIL=1` |
| `rest` | per dataset: features (block kernel; dense for the routed cells), then credible `--calib-pool train`, then C2/C3 | `JOBS=5` |
| `astral_queue.sh` | 150 ASTRAL-IV units staged 1k → 5k → 10k, at most `CAP=11` busy cores (less `RIDDLE_SLOTS=6` while RIDDLE runs), capped by `MEM_GB=40` / the size's peak RSS | `runs/logs/astral_queue/<size>_repRR.log`, `status.tsv` |
| `astral` | the same 150 runs from one fixed pool, the 10k ones first, `JOBS=5` | `runs/logs/astral.log` |
| `collate` | `results/` and gates C1–C8 | prefix with `LUAY=1` for luay (see Costs) |
| `status` | cells scored / feat / cred per (size, k), and ASTRAL counts | anytime |
| `sweep.py` | `results/sptree_sweep_rows.csv`: RIDDLE (sweep) per (size, rep), and `rf_hw_norm` on every RIDDLE / ASTRAL-IV row; gates W1–W4 | after `collate`, ~5 s; read by `notebooks/plotting_sptree.ipynb` |
| `runtime.py` | `results/sptree_runtime.csv`: wall seconds per (size, rep, method), with RIDDLE's parts; charging rules in its docstring; gates T1–T4 | ~3 s; read by `notebooks/plotting_sptree.ipynb` |

Smokes use `REPS=16,17,34` (any rep list), which makes every count check partial. To check the sheet as pasted:
`zsh -f -i < sheet`.

## Traps

- **`make_csv.py` / `bench.py` do not run here.** Both need a calibration pool, landed luay parity and HAE.
  `collate.py` is the table.
- **`n_true` is 198.** A rooted binary tree on 200 taxa has 199 internal nodes, and the root is trivial.
- **First-N prefixes, not samples.** `infer_gnn_cpu --m N` would draw a seeded *random* sample instead.
  `prepare_gene_trees(max_loci=N)` takes the first N lines, and so does `run_astral.py`, checked by sha256.
- **The block kernel's chunk is `trees × m_b³`.** On noisy cells (cu 0.1) a block can reach 55 atoms, which would take
  GBs per worker. Those cells are routed to the dense kernel; nothing is refused.
- **k > 7 is extrapolation for the GNN's global feature.** k = 0 is in distribution: about 22% of training networks.
- **The GNN spends its whole budget** (`k_pred == k` on every smoke cell). At k > 0 every reticulation is a false one
  against a tree truth, and precision is where it shows.

## Later: resolvers

When TOBs arrive:
- they go in `sptree_eval/tob/<size>/<key>.tob`;
- resolver runs go in `runs/resolvers/{brooqs,netcs}/<size>/<stem>/`;
- each method gets one loader entry in `collate.py` (a row per (size, k, rep), like the GNN rows).

`CELLS.tsv`, `stage.json` and `inputs/` stay as they are.
