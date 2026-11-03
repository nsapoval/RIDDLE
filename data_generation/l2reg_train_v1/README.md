# l2reg_train_v1 -- level-2 training data at n = 25 and n = 50

Built 2026-09-12T05:28:55+00:00 by `python -m opc.l2train` from rl-net-opt at `ab6824ae1205`
(with uncommitted changes -- the exact code is in `code/`).

1,800 phylogenetic networks, each **regular**, **exactly level-2** and of **girth >=
5** on its regular form. For each network, 1,000 gene trees under the network
multispecies coalescent, a 1000 bp AliSim alignment and an IQ-TREE re-estimate per locus, and a
TREE-QMC tree of blobs.

This is the training counterpart of `l2reg_n200_v1`, which is 20 networks at n = 200 and is a
**test** set. The regime it is for is net-opt's mixed true/inferred one, whose training phases
(`x25 x50 y25 y50`, in `alisim_gt_v1/v2`) are level-1 only: no blob any of them carries has two
reticulations.

## Design

Replicates are split by index, as the x/y grids are, so a split is a prefix of the replicates and
each `(layout, cu)` cell lands in every split a whole number of times.

| phase | n | networks | layouts (L2 x L1) | k | train / val / test |
|---|---|---|---|---|---|
| `l2t25` | 25 | 900 | 1x0 1x1 1x2 2x0 2x1 | 2-5 | 700 / 100 / 100 |
| `l2t50` | 50 | 900 | 1x0 1x1 1x2 2x0 1x3 2x1 2x2 3x0 2x3 3x1 | 2-7 | 700 / 100 / 100 |

Each replicate pair takes the next layout and each replicate the next ILS level
(`cu = [0.25, 0.5]`, the median internal branch in coalescent units), so the rotation cycle is
`2 x layouts` and every cell is balanced:

| phase | layout | k | cu | train / val / test |
|---|---|---|---|---|
| `l2t25` | 1x0 | 2 | 0.25 | 70 / 10 / 10 |
| `l2t25` | 1x0 | 2 | 0.5 | 70 / 10 / 10 |
| `l2t25` | 1x1 | 3 | 0.25 | 70 / 10 / 10 |
| `l2t25` | 1x1 | 3 | 0.5 | 70 / 10 / 10 |
| `l2t25` | 1x2 | 4 | 0.25 | 70 / 10 / 10 |
| `l2t25` | 1x2 | 4 | 0.5 | 70 / 10 / 10 |
| `l2t25` | 2x0 | 4 | 0.25 | 70 / 10 / 10 |
| `l2t25` | 2x0 | 4 | 0.5 | 70 / 10 / 10 |
| `l2t25` | 2x1 | 5 | 0.25 | 70 / 10 / 10 |
| `l2t25` | 2x1 | 5 | 0.5 | 70 / 10 / 10 |
| `l2t50` | 1x0 | 2 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 1x0 | 2 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 1x1 | 3 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 1x1 | 3 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 1x2 | 4 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 1x2 | 4 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 2x0 | 4 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 2x0 | 4 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 1x3 | 5 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 1x3 | 5 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 2x1 | 5 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 2x1 | 5 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 2x2 | 6 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 2x2 | 6 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 3x0 | 6 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 3x0 | 6 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 2x3 | 7 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 2x3 | 7 | 0.5 | 35 / 5 / 5 |
| `l2t50` | 3x1 | 7 | 0.25 | 35 / 5 / 5 |
| `l2t50` | 3x1 | 7 | 0.5 | 35 / 5 / 5 |

The four level-2 shapes (S0-S3, `l2reg_n200_v1`'s README defines them) rotate within each ILS
level, which balances them and means a two-blob network draws two *consecutive* shapes -- it never
asks for the same shape twice, which is the case that fails to place at n = 25.

One gamma per reticulation, ~U(0.25, 0.5) on the donor (minor) edge.

## Files

| path | what it is |
|---|---|
| `MANIFEST.tsv` | one row per network -- the index to iterate over, with its `phase`, `split` and `alignments_sha256` |
| `TRAINING_FILES.txt` | the paths a training box needs, one per line: truth, true gene trees and the estimates with lengths -- no sequence data, no `.raw.nwk.gz`, no `.pkl`. Feed it to `rsync --files-from` |
| `SHA256SUMS.training` | checksums for exactly those paths, so a partial copy verifies without reporting every file it was never sent |
| `<phase>/networks/<key>.enewick` | the network **as simulated**: binary, time-consistent, ultrametric, branch lengths in coalescent units, one gamma per reticulation (PhyloNetworks `name:length::gamma` dialect, `#Hk` copy on the minor edge) |
| `<phase>/networks/<key>.regular.enewick` | its **regular form** R: every hybrid merged with its only child, so a hybrid carries its child's children. Topology only -- R's branch lengths are not a model. A hybrid whose cluster is one taxon is written `(t:0)#Hk`. **Never simulate from R**: PhyloCoalSimulations sends a hybrid leaf's lineages up its first parent only |
| `<phase>/networks/<key>.json` | the ground truth: blobs (shape, girth in both forms, size), reticulations (gamma, time, hybrid / donor / tree-parent clusters), R as clusters plus Hasse edges (`regular_network`), every gate, and the generator's rejection counters |
| `<phase>/networks/<key>.pkl` | the `opc.simulate.networks.Network` object itself |
| `<phase>/trees/<key>.nwk.gz` | 1,000 rooted true gene trees, one per line, coalescent units |
| `<phase>/alignments_bp1000/<key>/locNNNN.phy.gz` | AliSim's alignment for locus NNNN, sequential PHYLIP, **200 taxa plus the outgroup `OUT`** |
| `<phase>/estimated_bp1000/<key>.nwk.gz` | IQ-TREE's tree for each locus, rooted on `OUT`, `OUT` pruned, **topology only**, in locus order |
| `<phase>/estimated_bp1000/<key>.raw.nwk.gz` | what IQ-TREE returned: unrooted, `OUT` attached, branch lengths in substitutions/site |
| `<phase>/estimated_bp1000/<key>.len.nwk.gz` | the same trees as `<key>.nwk.gz` -- rooted on `OUT`, `OUT` pruned, same order -- **with branch lengths** (substitutions/site). What to train on: net-opt's W13 reads lengths, and a length-free tree degenerates silently |
| `<phase>/estimated_bp1000/<key>.json` | alignment widths, gap fraction, seeds, timings, the model IQ-TREE reported using, RF against the true trees |
| `<phase>/tob_bp1000/<key>.tob` | the estimated **tree of blobs**: TREE-QMC `--blob` on the network's 1,000 estimated gene trees (the `.raw.nwk.gz`, `OUT` included), rooted on `OUT` and `OUT` pruned, with TREE-QMC's root-branch defect corrected (see below). Topology only; a polytomy wherever it found a blob |
| `<phase>/tob_bp1000/<key>.raw.tob` | TREE-QMC's output exactly as written: unrooted, `OUT` included, defect and all |
| `<phase>/tob_bp1000/<key>.pvalues.tob` | the tree TREE-QMC tested, each branch annotated with its tree- and star-test p-values, qCFs and minimising quartet -- what `<key>.tob` is derived from |
| `<phase>/tob_bp1000/<key>.json` | the command, TREE-QMC version, `alpha` / `beta` and run time; what was contracted, and at the root; the **true** tree of blobs (`true_tob`, one edge per bridge of the network); missed (`fn`) / extra (`fp`) edges against it, and the same for the raw output (`fn_raw`, `fp_raw`) |
| `<phase>/tob_bp1000/logs/<key>.log` | TREE-QMC's log, including every per-branch quartet test |
| `manifest.json` | design, seeds, model, tool versions, code state, gate summary |
| `<phase>/manifests/audit.json` | PhyloNetworks' own reading of every network; `audit_verdict.json` compares it with what the generator meant |
| `code/` | the code the dataset was built from: `tracked.diff` against the commit, plus new files |
| `SHA256SUMS` | over every file above: `sha256sum -c SHA256SUMS --quiet` |

## Realised

| phase | shapes | girth on R | tree-child | aln width | gaps | RF | tob exact | tob missed | tob extra |
|---|---|---|---|---|---|---|---|---|---|
| `l2t25` | {'S0': 316, 'S1': 316, 'S2': 314, 'S3': 314} | {5: 1238, 6: 413, 7: 178, 8: 81, 9: 36, 10: 14, 11: 10, 12: 8, 13: 2} | 829/900 | 1193 | 16% | 0.087 | 77/900 | 260/12001 | 2567 |
| `l2t50` | {'S0': 406, 'S1': 406, 'S2': 404, 'S3': 404} | {5: 1459, 6: 624, 7: 311, 8: 173, 9: 81, 10: 58, 11: 34, 12: 19, 13: 10, 14: 6, 15: 9, 16: 5, 17: 1} | 823/900 | 1320 | 24% | 0.085 | 26/900 | 722/29142 | 3956 |

Base-tree redraws per phase: `l2t25` {0: 770, 1: 82, 2: 34, 3: 5, 4: 7, 5: 1, 9: 1}, `l2t50` {0: 859, 1: 36, 2: 2, 3: 3} -- the cap is
100.

## Sequence model

`HKY{2}+F{0.3,0.2,0.2,0.3}+G4{1}` + indels `0.03,0.03`, size
`POW{1.7/100}`, branch scale `0.018` substitutions/site per coalescent unit
(theta = 0.036), inferred under `HKY+F+G4` with `--fast --keep-ident` --
identical to `alisim_gt_v1/v2` and `l2reg_n200_v1`. `1000` is the **root** sequence length:
indels make realised alignments wider. The outgroup is grafted onto each gene tree at
`(1 + 0.5) x p99` of that network's root heights, so it contributes no rooting
error.

**IQ-TREE is `IQ-TREE multicore version 2.4.0 for Linux x86 64-bit built Feb 12 2025`** -- v1/v2's build, not the 2.3.6 `l2reg_n200_v1`
used, because this data is mixed with v1/v2's `est_bp1000` arm at training time. The estimate
stage refuses to run on any other version.

## Reading it

`<key>.nwk.gz` is topology only, as in v1/v2. **`<key>.len.nwk.gz` is the one to train on**: the
same trees, same order, with branch lengths in substitutions per site. net-opt's W13 reads lengths
off the estimated arm, and a length-free tree does not fail anywhere -- it degenerates into
constant features. Stripping the lengths from `.len.nwk.gz` reproduces `.nwk.gz` line for line,
and that is checked on every line of every file when it is written.

Each estimated file is paired locus for locus with `trees/<key>.nwk.gz`, the true trees.

## Copying just the training half

The per-locus alignments are **79%** of the package and nothing in training reads
them. They are kept so the estimates can be re-derived, or re-estimated with another tool -- which
is done once, on the machine that has the disk. Each phase keeps them in one directory,
`<phase>/alignments_bp1000/`, so one exclude leaves every one of them behind:

```bash
rsync -a --exclude 'alignments_bp1000/' l2reg_train_v1/ box:/data/l2reg_train_v1/
```

`TRAINING_FILES.txt` is the same thing said exactly rather than by pattern -- 12,604 paths:
the truth, the true gene trees, and the estimates with lengths. It also drops `<key>.raw.nwk.gz`
(IQ-TREE's output before rooting, which `.len.nwk.gz` supersedes) and `<key>.pkl` (a pickle of
this repo's `Network` class, which a reader would need the same class for):

```bash
rsync -a --files-from=TRAINING_FILES.txt l2reg_train_v1/ box:/data/l2reg_train_v1/
cd /data/l2reg_train_v1 && sha256sum -c SHA256SUMS.training --quiet
```

That is **1.6 GB** against 11.7 GB for the whole package.

## Verifying

```bash
sha256sum -c SHA256SUMS --quiet          # everything except the per-locus alignments
sha256sum -c SHA256SUMS.training --quiet # just what TRAINING_FILES.txt names
```

The alignments are covered one digest per network, in `MANIFEST.tsv`'s `alignments_sha256`: the
SHA-256 of that network's `loc*.phy.gz` concatenated in locus order. There are
1,800,000 of them, so listing each one would make `SHA256SUMS` ~180 MB.

## Reproducibility

Seeds are SHA-256 chains off master seed `20260911`: networks
`(master, "l2train", "net", n, rep)`, gene trees `(master, "l2train", "gt", key)`, AliSim / IQ-TREE
`(master, "alisim" | "iqtree", key, bp)` plus the global locus / batch index. The master and tag
are this dataset's own, so no stream is shared with `l2reg_n200_v1`, `l3reg_n200_v1` or the x/y
grids -- checked as a probe gate, not assumed.

`python -m opc.l2train all` rebuilds everything; each stage skips what already exists.
