# AliSim hand-off — gene trees for the `x`- and `b`-grids

Built 2026-09-04 10:15:00 from `net-opt` at `ab6824ae1205`.

Per-network **true** gene tree Newicks, plus everything needed to turn them into alignments and
re-estimated gene trees on a CPU box. No GPU, no network access, no conda env: Python 3.9+ and an
**IQ-TREE 2** binary (AliSim ships inside it) are the whole dependency list.

**Python 3.9 is a hard floor.** The vendored modules use `dict |= dict`, and without a check the
failure lands in the rooting step *after* a cell has been fully simulated and inferred — so the
driver checks `sys.version_info` before doing any work. Ubuntu 20.04's stock `python3` is 3.8.

```
trees/<phase>/<cell_key>.nwk.gz      1,000 rooted true gene trees, one per line, coalescent units
networks/<phase>/<cell_key>.enewick  the true species network the trees were simulated on
MANIFEST.tsv                         one row per network — the index to iterate over
manifest.json                        model, splits, seeds check, per-phase cost
pylib/opc/simulate/                  outgroup.py, phylip.py, seqsim.py — verbatim from the repo
run_estimate.py                      the driver
SHA256SUMS                           over every file above
```

## Inventory

| phase | n | k | cells | requested | loci (requested) | CPU-h @ 500 bp |
|---|---|---|---|---|---|---|
| `x25` | 25 | 0,1,2,3 | 900 | 900 | 900,000 | 58.7 |
| `x50` | 50 | 0,1,2,3,5 | 900 | 900 | 900,000 | 171.4 |
| `b25` | 25 | 0,1,2,3 | 100 | 20 | 20,000 | 1.3 |
| `b50` | 50 | 0,1,2,3,5 | 100 | 20 | 20,000 | 3.8 |
| `b100` | 100 | 0,1,2,4,8 | 100 | 20 | 20,000 | 10.1 |
| `b250` | 250 | 0,1,2,8,16 | 100 | 20 | 20,000 | 33.1 |
| **total** | | | **2200** | **1880** | **1,880,000** | **278.4** |

`requested` marks what this package was built for: the `x`-grids in full (the 10× training data,
700 train / 100 val / 100 spare per phase) and the `b`-grids' 20-cell **test** splits (the sets
every landed benchmark number is on). The `b`-grid train and validation splits are in the package
too, marked `requested=0`, so nothing needs a second round trip — `run_estimate.py --all` reaches
them, at roughly 240 more CPU-h.

The CPU-h column already carries the measured **2.7× concurrency penalty**: the workload is
memory-bandwidth bound, and a full-scale run once spent 1,651 CPU-h where the single-process cost
model predicted 612. Take it as the number to budget against, not the optimistic one.

## Run it

```bash
python3 run_estimate.py --iqtree /path/to/iqtree2 --jobs 32 --dry-run   # budget, then
python3 run_estimate.py --iqtree /path/to/iqtree2 --jobs 32
```

Resumable at cell granularity — a cell whose output is newer than its input is skipped, so an
interruption costs only the cells still missing. Failures never stop the pool; each one leaves
`estimated_bp500/<phase>/<key>.failed.txt` with the tool's own stderr tail.

Useful flags: `--phases b250`, `--splits test`, `--cells keys.txt`, `--max-k 2`, `--limit N`,
`--all`, `--bp 1000`, `--loci 200`, `--scratch /fast/disk`.

### What comes out

Three files per cell under `estimated_bp<BP>/<phase>/`:

| file | what it is |
|---|---|
| `<key>.nwk.gz` | **the deliverable** — rooted, outgroup pruned, ingroup only, one tree per line, in locus order, paired 1:1 with `trees/<phase>/<key>.nwk.gz` |
| `<key>.raw.nwk.gz` | what IQ-TREE actually returned: unrooted, `OUT` still attached |
| `<key>.json` | alignment widths, gap fraction, seeds, timings, the model IQ-TREE reported using, and RF of the estimate against the true tree it came from |

Send back the whole `estimated_bp<BP>/` directory. It is small — trees, not alignments.

### Alignments are transient

Each locus alignment is written to `--scratch`, consumed, and deleted; nothing under
`estimated_bp<BP>/` is sequence data. Retaining them all would be ~45 GB across the requested set.
If you want them kept, say so — it is a change to the driver, not to the package.

`--scratch` needs about `loci × (taxa+1) × bp` bytes **per concurrent job**: ~15 MB at n = 25,
~230 MB at n = 250. On a 32-job run at n = 250 that is ~7 GB, so point it at a real disk rather
than a small tmpfs.

### Driving AliSim yourself

```bash
python3 run_estimate.py --stage graft --jobs 12
```

writes `alisim_ready/<phase>/<key>.nwk.gz` — the same trees with a synthetic outgroup `OUT`
grafted on and branch lengths already converted from coalescent units to substitutions per site.
Feed those to AliSim one line at a time. The step is deterministic (no RNG anywhere in it), so it
does not matter which machine runs it.

## Choosing `--bp` — measured, not assumed

`--bp 500` is this study's default and at these taxon counts it produces gene trees that are
**never exactly right**. Measured here on 2 test cells per phase × 50 loci, one AliSim + IQ-TREE
pass per row:

| n | bp | mean aln width | gap % | RF (norm) mean | median | loci recovered exactly | root split correct | s/locus |
|---|---|---|---|---|---|---|---|---|
| 25 | 500 | 579 | 13.5 | 0.165 | 0.174 | 1% | 76% | 0.077 |
| 25 | 1000 | 1144 | 12.7 | 0.086 | 0.087 | 11% | 87% | 0.109 |
| 25 | 2000 | 2280 | 12.4 | 0.047 | 0.043 | 30% | 89% | 0.167 |
| 50 | 500 | 606 | 17.6 | 0.163 | 0.167 | **0%** | 81% | 0.195 |
| 50 | 1000 | 1217 | 18.2 | 0.093 | 0.089 | 4% | 81% | 0.298 |
| 50 | 2000 | 2426 | 17.4 | 0.052 | 0.042 | 7% | 93% | 0.467 |
| 100 | 500 | 707 | 29.6 | 0.176 | 0.173 | **0%** | 75% | 0.602 |
| 100 | 1000 | 1436 | 30.2 | 0.100 | 0.102 | **0%** | 87% | 0.949 |
| 250 | 500 | 1008 | 50.8 | 0.187 | 0.189 | **0%** | 76% | 2.666 |
| 250 | 1000 | 2024 | 50.5 | 0.104 | 0.101 | **0%** | 88% | 4.616 |

Read it as three things:

* **Doubling `bp` roughly halves RF error and costs about 1.5×** — the cost is `bp**0.6`, not linear,
  so length is unusually cheap here. 500 → 1000 bp is the best trade on this curve.
* **"Recovered exactly" collapses with n whatever the length.** At n ≥ 100 no locus is exactly right
  at either length, so a downstream method reading these trees is working from *systematically*
  perturbed input, not from a noisy version of the truth. That is the point of the arm, but it has
  to be stated rather than discovered.
* **Gap fraction climbs steeply with n** — 13% at n = 25 to 51% at n = 250 — because indel load
  tracks total tree length and `--bp` is the *root* sequence length. Half the n = 250 alignment is
  gap at any length.

The `s/locus` column is single-process and measured on 16 cores; the CPU-h table above applies the
2.7× concurrency penalty to it. The cost model behind that table interpolates to n = 200 and is
**~20% optimistic at n = 250** (2.67 s measured against 2.21 s predicted) — budget `b250`
accordingly.

## The model

`HKY{2}+F{0.3,0.2,0.2,0.3}+G4{1}` + indels `0.03,0.03` size `POW{1.7/100}`, branch scale `0.018` (theta = 0.036), inferred under `HKY+F+G4`

`--bp` is the **root** sequence length, not the alignment width: indels make realised alignments
wider, and gap fraction climbs with `n` because indel load tracks total tree length.

## Why the pipeline is shaped the way it is

Every item here is a measured failure that is **silent** — right exit code, plausible output, wrong
data. They are why `run_estimate.py` exists instead of a shell loop.

* **AliSim is called once per locus.** `-t multi.nwk` simulates from the *first* tree only and says
  nothing. The batched `-Q` partition mode does iterate, but it ignores `--branch-scale` (identical
  TreeLen with and without) and aborts on indels combined with `+G4` — and this design needs both.
* **The coalescent-unit → substitutions rescale happens in Python**, in `outgroup.graft_file`, for
  that same reason.
* **The substitution model goes in the partition file**, where it silently overrides `-m`. Writing
  the conventional `DNA` there makes IQ-TREE infer under GTR whatever `-m` says, exit code 0. IQ-TREE
  also accepts an unknown `-m` and runs anyway, so the driver parses the `.iqtree` report and checks
  what was actually applied.
* **`--keep-ident`**, or IQ-TREE drops identical sequences — common at 200–500 bp — and returns a
  tree short a taxon. Downstream cluster extraction raises on an *unknown* taxon but is silent on a
  *missing* one, so this would corrupt results without erroring anywhere. `root_and_prune_file`
  checks the emitted taxon set against the expected one and raises.
* **The outgroup is grafted onto each gene tree, not added to the species network.** IQ-TREE returns
  unrooted trees and something in the alignment has to carry the root. Grafting keeps the ingroup
  taxon set byte-identical to the true-tree arm, so the two arms stay paired locus for locus. The
  price is that the outgroup never sorts discordantly and never goes missing, so it contributes no
  rooting error of its own — `frac_root_split_correct` tracks topology error near the root, not
  rooting error.
* **`T_out` is fixed per cell**, at `(1 + alpha) · p99(root heights)` over all 1,000 loci — an
  outgroup diverges once, not once per locus. That is why a cell has to travel whole: computing it
  over a subset moves the outgroup.
* **Two indel-driven failures are retried with a fresh seed**, up to 4 attempts: AliSim can delete
  every site of a locus (~1 in 35,000), and a single taxon can lose every observed site (~1 in 792
  in the gappiest cell). Neither is data you could have in hand. `n_alisim_retries` records it.
* **`--jobs >= 100` deadlocks silently** under a shared `RLIMIT_NPROC` — worker forks start failing,
  which kills multiprocessing's `_handle_workers` thread, and the pool then hangs at 1–8% CPU with
  no error, for ever. The driver refuses. `jobs × threads` is the real budget; every IQ-TREE and
  AliSim call here runs at `-T 1` with the BLAS thread env pinned.

## Reproducibility

Seeds are `SHA-256(master_seed : "alisim"|"iqtree" : cell_key : bp)`, derived by `run_estimate.py`
rather than shipped, because they depend on `--bp`. `manifest.json` carries a dozen of them computed
on the source machine and the driver **refuses to start** if its own derivation disagrees.

`MANIFEST.tsv` carries a SHA-256 and a verified tree count per cell; `SHA256SUMS` covers the whole
package. Check it after the transfer:

```bash
sha256sum -c SHA256SUMS --quiet
```
