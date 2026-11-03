# RIDDLE

**Ultra-scalable regular phylogenetic network inference with graph neural networks.**

RIDDLE turns a file of gene trees into one rooted phylogenetic network per reticulation budget *k*, written as
extended Newick. It runs in two stages:

1. **GNN scoring and decoding (Stage A).**
   - Every clade (cluster) that appears in at least τ gene trees is a candidate.
   - A graph neural network scores each candidate from 31 node features and 5 global features over the candidates'
     overlap graph.
   - The candidates are accepted greedily in score order, while the selection can still be drawn as a network of
     level ≤ 2 with at most *k* reticulations.
2. **MAP re-decision (Stage B).**
   - The emitted clusters the GNN is least sure of (logit z < 6) are grouped into independent blocks.
   - Within each block, the alternative configurations reachable by NNI-type swaps are scored by a calibrated choice
     model on z and gene-tree triplet concordance (tc).
   - The most probable configuration of every block is kept.
   - The choice model is the **train-fit β**: fitted on the networks the GNN was trained on, and selected on the
     networks it was validated on. It never saw a benchmark.

In the code and the result tables RIDDLE is labelled `GNN MAP [z+tc]`, under the train-fit β.

---

## Install

Python ≥ 3.10 (built and tested with 3.12). Inference runs on CPU and needs no GPU, Java or network access.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r gnn_cpu_v1/requirements-cpu.txt        # numpy, scipy, pyarrow, torch 2.11.0
```

On Linux, install the CPU build of torch first so pip does not fetch the multi-GB CUDA wheel:

```bash
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r gnn_cpu_v1/requirements-cpu.txt
```

These steps were checked on macOS (arm64, Python 3.12.9). The Linux line is PyTorch's standard CPU index and was not
tested here.

## Quick start

```bash
.venv/bin/python riddle.py --trees gnn_cpu_v1/inputs/n25/00/iqtree_500.nwk --outgroup OUT --k 1,2 --out example_run
```

The example is one CAMUS benchmark replicate: 1,000 IQ-TREE gene trees on 25 taxa plus the outgroup `OUT`, with its
true network in `gnn_cpu_v1/inputs/n25/00/true_net.nwk`. It takes about 10 seconds.

**Outputs, in `--out`:**

| file | what |
|---|---|
| `riddle.tsv` | **The RIDDLE networks:** one line per k, `k<TAB>eNewick`. |
| `gnn.tsv` | The GNN's own network per k, before the MAP re-decision. |
| `riddle.json` | The arguments; the sha256 of the input, model and β; the taxon set; m and τ; the rooting summary; per k, the number of blocks, how many moved, the MAP's log10 probability, and its reticulation count and level; any k that failed. |
| `work/` | Every intermediate: the rooted trees, Stage A's logits (`stage_a/gnn.logits.npz`), the triple tensor, and the per-k feature and credible-set JSONs. A rerun into the same `--out` resumes, and refuses if the arguments changed. |

To pull one network out as plain eNewick:

```bash
awk -F'\t' '$1 == 2 { print $2 }' example_run/riddle.tsv > net_k2.enewick
```

### Options

| option | meaning |
|---|---|
| `--trees FILE` | One Newick tree per line, plain or `.gz`. Branch lengths are required: node features read them. |
| `--k LIST` | Reticulation budgets, e.g. `1-29` or `1,3,5`. |
| `--outgroup A[,B,…]` / `--midpoint` / `--species-tree FILE` | **Rooting.** The outgroups are ranked: each tree is rooted on the first one it carries, which is then pruned. With a species tree, each gene tree is rooted on the species tree's root split. With none of these, trees are taken as rooted as written. |
| `--partial MIN` | Accept trees that carry at least MIN taxa of S. Without it, every tree must carry every taxon. |
| `--n-taxa N` / `--taxa FILE` | Restrict the taxon set S to the N most frequent taxa, or to a list. |
| `--m M`, `--seed S` | Sample M trees. The default 0 uses all of them. |
| `--tau N` / `--tau-frac F` | Candidate support threshold: τ = N, or τ = ⌈F·m⌉. The default is τ = 2, which is what training used (at m = 1,000). |
| `--jobs J` | Stage B workers, about 0.7–1.4 GB each. |
| `--threads T` | Stage A's torch and BLAS threads. The default is 1, which is what every published number used. |
| `--stage-a PREFIX` | Reuse a finished `infer_gnn_cpu.py … --annotate PREFIX` run. It must be relaxed L2 with the same k list. |

**Fixed settings.** Stage B was built and calibrated for these, so they are not options:
- the checkpoint `_l2mix_pair/s0/best.pt`;
- the relaxed level-2 decoder;
- z < 6 and radius 2.

**The two real-data runs used:**

```bash
# fungi: 646 genomes, rooted gene trees, taxa missing from many trees
.venv/bin/python riddle.py --trees genetrees.rooted.tre --partial 485 --m 0 --seed 0 --tau 2 --k 1-29 --out fungi_run
# birds: 363 species, unrooted IQ-TREE trees rooted on the GOBS species tree
.venv/bin/python riddle.py --trees 63430.gene.trees --species-tree 63430-GOBS.rooted.tre --partial 273 --m 0 --seed 0 --tau-frac 0.0005 --k 1-29 --out birds_run
```

**Cost.** Fungi (n = 646, 1,698 trees, k = 1–29), on an Apple-silicon laptop:
- Stage A took 4.3 min at one thread.
- Stage B took 4.7 min with 5 workers. Peak RSS was 1.4 GB per feature worker and 0.7 GB per credible worker.
- The triple tensor is n³ uint16: 539 MB at n = 646 and 2 GB at n = 1,000.
- m must stay below 65,535 trees.

**Limits.**
- Stage B has caps on block size and enumeration (`fungi/scripts/fz_common.py`, `CAPS`). A k that exceeds one fails,
  is named in `riddle.json`, and the run exits 1. The other k values are still written.
- Stage B forks worker processes, so it runs on POSIX systems only.
- Files under `work/` carry `fungi_` names. Stage B's modules derive their paths from those names (see *How
  `riddle.py` is built*).

### Stage A alone

`gnn_cpu_v1/infer_gnn_cpu.py` is the GNN step on its own, with more options: the level-1 decoder, other checkpoints,
and `--stop-at-reticulation`. Its guide is `gnn_cpu_v1/infer_gnn_cpu.md`, and `gnn_cpu_v1/infer_gnn_ens.md` covers
ensembles over subsamples.

## Model

| | |
|---|---|
| checkpoint | `gnn_cpu_v1/checkpoints/_l2mix_pair/s0/best.pt` |
| sha256 | `ef5f06b4e8ff6fbab988c157cd0aca573a7ffa70681b1e04eed6730f40662751` |
| selected | epoch 30 of 50, on validation cluster F1 (0.9550; greedy-by-support 0.7418) |
| architecture | `pair` convolution with containment aggregators, hidden 128, 3 layers, pointwise head, 1,041,931 parameters |
| features | 31 per candidate: support, size, overlap-graph degree and component, and statistics of the gene-tree branches above the cluster (length, rank, depth, sibling, conflict). 5 global: `k`, `k/n`, density, mean degree fraction, kernel fraction |
| training data | `x25 x50 y25 y50`: level-1 galled networks at n = 25 and 50, two independent draws. `l2t25 l2t50`: regular, exactly level-2 networks at n = 25 and 50, k = 2–7. Each on true gene trees and on IQ-TREE re-estimates from 1,000 bp AliSim alignments; τ = 2, 1,000 loci per network |
| split | replicates 0–699 train (8,400 cells), 700–799 validation (1,200 cells), 800–899 untouched test |
| config / verdict | `gnn_cpu_v1/checkpoints/_l2mix_pair/s0/{config.json,verdict.json}` |

**The train-fit β:**
- **File:** `nni_ensemble/runs_trainfit/calib/train__z6.json`, sha256 `f47199e9746d8468f7c62ee145544f1525de176c94dbe7e4558adadf5c03a610`.
- **Model:** a ridge-penalised multinomial logit over each block's configurations.
- **Features:** z, its margin over crossing rivals, support frequency, the option's atom count, tc, and tc's margin.
- **Fitting data:** the checkpoint's own train replicates (7,601 blocks), fitted on the estimated-gene-tree arm with
  ridge 10³.
- **Selection:** by held-out log-likelihood on the validation replicates' estimated-arm blocks (1,027 validation
  blocks, 553 of them on that arm). The pre-registered grid was ridge 10⁻³–10⁴ × training arm {both, estimated}.
  The selected fit, estimated arm at ridge 10³, is an interior optimum: −503.35 against −507.94 at 10² and −560.81
  at 10⁴. The uniform baseline is −693.97.
- **Code:** `nni_ensemble/trainfit.py` is the fit, and `train_eval/` stages the replicates. The fit reads replicates
  already scored by the benchmark-scoring code, which is not part of this repository, so `trainfit.py` documents the
  fit rather than running on its own.

## Repository layout

| path | what |
|---|---|
| `riddle.py` | The end-to-end inference driver (new). |
| `gnn_cpu_v1/` | **Stage A and the inference library:** `infer_gnn_cpu.py`, `rooting.py`, the benchmark driver `run_gnn_cpu.py` and the library `pylib/opc/`. Also the checkpoint, and the example cell in `inputs/n25/00/`. |
| `fungi/scripts/` | The eight Stage-B modules `riddle.py` calls. The rest of the fungal analysis is not included. |
| `nni_ensemble/` | The z+tc block model that Stage B imports: blocks, calibration, credible sets and triplet concordance. Also the train-fit β (`runs_trainfit/calib/train__z6.json`, at the path Stage B reads) and `trainfit.py`, which fitted it. |
| `cluster_anatomy/common.py` | Imported by `nni_ensemble/ne_common.py`; nothing else from that directory is needed. |
| `train_eval/` | Stages the checkpoint's own train and validation networks for the train-fit β. |
| `l2reg_eval/`, `l3reg_eval/`, `netcs_eval/`, `n1000_eval/` | Staging, GNN runs, resolver runs and per-network tables for the four simulated benchmarks beyond CAMUS. |
| `subsample_eval/` | The 500-tree draws and the TOB resolvers (BROOQS, NetCS) on them. |
| `taxmiss_eval/` | Gene trees with missing taxa. |
| `sptree_eval/` | Species-tree (reticulation-free) data: RIDDLE over k = 0–10 against ASTRAL-IV. |
| `notebooks/` | The plotting notebooks for Section 3.1, the missing-taxa benchmark and the species-tree benchmark. `notebooks/data/results/` holds the aggregated benchmark tables they read. |
| `training/` | The training code, exactly as it trained the checkpoint (see *Training*). |
| `data_generation/` | The generators of the level-2 training grids and the AliSim/IQ-TREE estimation packages (see *Training data*). |

Each `*_eval/` directory and `train_eval/` has a README with its commands, gates and measured costs. Those READMEs
were written for the original machines. Several things they mention are not part of this repository:
- hostnames and absolute paths;
- the `runs/` and `inputs/` directories;
- the benchmark scoring and aggregation code in `cluster_anatomy/` and `nni_ensemble/`.

## Training

`training/` is a verbatim copy of the training checkout, taken from the GPU machine it ran on.

**Environment:**
- Python 3.12.8, with `training/requirements.haldane.lock` (torch 2.11.0+cu128, pandas 3.0.5).
- Julia 1.12.6, with `training/julia/{Project,Manifest}.toml` (PhyloNetworks 1.3.1, PhyloCoalSimulations 1.1.0), for
  simulating gene trees.
- One RTX A6000: about 2,060 s per epoch, about 29 h for the 50 epochs.

**The run.** It is `training/results/rl/rl_run_gnn_pair_s0_l2mix/`: `config.json`, `verdict.json`, the full training
log `rl_run_gnn_pair_s0_l2mix.train.log`, the per-epoch metrics parquet, and the instance summaries that `pos_weight`
is computed from.

```bash
cd training
CUDA_VISIBLE_DEVICES=0 ARCHS=pair SEEDS=0 TAG=_l2mix PHASES=x25,x50,y25,y50,l2t25,l2t50 EPOCHS=50 \
  ARMS=true,est_bp1000 SPLIT_TRAIN=700 SPLIT_VAL=800 POOL=500 LAZY=1 VAL_DECODER=auto VAL_MAX_LEVEL=2 \
  bash scripts/ops/queue_gnn_train.sh
# which runs
.venv/bin/python -u scripts/w10_gnn.py --arch pair --seed 0 --epochs 50 --phases x25,x50,y25,y50,l2t25,l2t50 \
  --arms true,est_bp1000 --split-train 700 --split-val 800 --pool-size 500 --lazy-env --tag _l2mix \
  --val-decoder auto --val-max-level 2
```

Everything else is a `w10_gnn.py` default: τ = 2, batch 8, lr 3·10⁻⁴, weight decay 0.01, hidden 128, containment
on, and edge supervision 0.25 / 16 / 2048.

### Training data

Training reads pickled instances from `training/results/<phase>/instances_tau2{,_est_bp1000}/`. They take 148 GB for
the six phases and both arms, and are **not included**. To rebuild them, from `training/`:

```bash
# 1. level-1 grids: networks and true gene trees (PhyloCoalSimulations)
for p in x25 x50 y25 y50; do .venv/bin/python -u -m opc.rungrid --phase $p --backend julia --jobs 10; done
for p in x25 x50 y25 y50; do .venv/bin/python -u scripts/rl_instances.py --phase $p --tau 2 --reps 900 --include-k0 --jobs 12; done
# 2. their estimated arm: AliSim alignments and IQ-TREE trees, built on a CPU machine.
#    scripts/pack_alisim.py writes the package (handoff/alisim_gt_v1: x25, x50; handoff/alisim_gt_v2: y25, y50);
#    run its run_estimate.py there (data_generation/alisim_gt_v*/), then bring estimated_bp1000/ back
.venv/bin/python -u scripts/unpack_alisim.py --phases x25,x50 --jobs 12
.venv/bin/python -u scripts/unpack_alisim.py --src handoff/alisim_gt_v2/estimated_bp1000 --phases y25,y50 --bp 1000 --jobs 10
for p in x25 x50 y25 y50; do .venv/bin/python -u scripts/rl_instances.py --phase $p --tau 2 --arm est_bp1000 --reps 900 --include-k0 --jobs 12; done
# 3. level-2 grids: data_generation/l2reg_train_v1 (python -m opc.l2train all), copied to data/l2reg_train_v1
for p in l2t25 l2t50; do
  .venv/bin/python -u scripts/rl_l2reg_instances.py --phase $p --tau 2 --reps 800 --jobs 12
  .venv/bin/python -u scripts/rl_l2reg_instances.py --phase $p --tau 2 --arm est_bp1000 --reps 800 --jobs 12
done
```

`alisim_gt_v1` (built 2026-09-04) packaged x25/x50, and `alisim_gt_v2` (2026-09-08) packaged y25/y50. Each also
packaged phases this checkpoint did not use, and each `manifest.json` records the full list. `unpack_alisim.py`
reads `alisim_gt_v1` unless `--src` names another.

## Reproducing the figures

`notebooks/` holds three notebooks. Each is run from `notebooks/` and writes PNGs to `../plots/`:

| notebook | figures | reads |
|---|---|---|
| `plotting_sec3-1.ipynb` | `AllDatasets-sec3-1-{accuracy, runtime-scaling, accuracy-genetrees, strata, retic, paired, runtime-by-corpus}` | `data/results/trainfit/` (bench tables and gates, including `derived/`), `data/results/{camus_own,*_runtime}*`, `../subsample_eval/results/*_sub500_res.csv`, and the `CELLS.tsv` of each benchmark |
| `plotting_taxmiss.ipynb` | `AllDatasets-taxmiss-{accuracy, delta, map-edge, tob, missingness, runtime}`, `CAMUS-taxmiss-by-n` | `../taxmiss_eval/results/`, `../taxmiss_eval/{inputs/*/MANIFEST.tsv, staged/*/p0.90_c24/CELLS.tsv}`, and `data/results/` |
| `plotting_sptree.ipynb` | `sptree-{headline, ksweep, runtime}` | `../sptree_eval/results/` |

Every table they read is included: 83 files, about 9 MB.
- **Aggregated benchmark tables:** in `notebooks/data/results/` (53 files). They were written to `nni_ensemble/results/`
  in the analysis checkout. The notebooks' data-path constants were changed to the new location, and that is the
  only edit to them: `BENCH_DIR` / `PUB_DIR` in `plotting_sec3-1.ipynb` and `RUNTIME_DIR` in
  `plotting_taxmiss.ipynb`. Their markdown and comments still name `nni_ensemble/…` as where each table came from.
- **Every other table** is at the path its pipeline writes it to.

To run them:

```bash
cd notebooks
OMP_NUM_THREADS=1 jupyter nbconvert --to notebook --execute --inplace plotting_sec3-1.ipynb
```

They were run with Python 3.13.2, pandas 2.2.3, numpy 2.2.4, matplotlib 3.10.0, scipy 1.15.2, seaborn 0.13.2 and
jinja2 3.1.6 (`DataFrame.to_latex` needs jinja2).
- Executed top to bottom from a clean copy of this repository, all three run without error, and taxmiss reports
  `MISSING: nothing`. That notebook skips a missing table silently and lists it, so that line is the check.

**Where the tables come from:**

| tables | pipeline |
|---|---|
| `notebooks/data/results/`: `trainfit/*_bench.csv`, `trainfit/derived/*`, `camus_own_*`, `*_runtime*` | The benchmark-aggregation code, which is not included. Per cell, it applied the train-fit β to the GNN logits of the eval drivers' runs below, scored every method against the true network, and timed each stage. CAMUS's own networks come from the published CAMUS corpus. |
| `subsample_eval/results/*_sub500_res.csv` | `subsample_eval/run_res.py`, then `make_res_csv.py` |
| `taxmiss_eval/results/*` | `taxmiss_eval/stage.py`, `run_gnn.sh`, `run_res.py`, then `make_csv.py` |
| `sptree_eval/results/*` | `sptree_eval/run_sptree.sh`, `run_astral.py`, then `collate.py`, `sweep.py` and `runtime.py` |

Re-running those pipelines needs the benchmark gene trees, staged inputs and run directories. They are tens of GB and
are not included. The benchmark networks and gene trees are the published CAMUS and NetCS corpora plus the simulated
L2reg, L3reg, n = 1,000 and species-tree datasets.

## License

GNU Affero General Public License v3.0; see `LICENSE`.

## Citation

*Citation to be added on publication.*
