# n1000_eval — the n = 1000 stress test (`l2_n1000_gt_50k_v1`)

The dataset `l2_n1000_gt_50k_v1/` (module `opc.n1000`) holds:
- 20 level-2 networks at **n = 1000**, with k = 3–9 and cu 0.25 / 0.5;
- 50,000 true gene trees;
- 10,000 IQ-TREE trees (`estimated_bp1000/<key>.raw.nwk.gz`, OUT kept).

This package runs the GNN, greedy, and BROOQS / NetCS on it. It writes
`results/rl/l2_n1000_gt50k_v1_per_network.csv` (+ `.json`), plotted by `plotting_n1000.ipynb`.

## Design (decided 2026-10-06)

- **Gene trees.** Every method reads the **10,000 estimated trees** at τ = ⌈0.002·m⌉ = 20, the
  gt-scaling rule. The 50k true trees are not used.
- **GNN.** `_l2mix_pair/s0`, `best.pt` (selF1) and `best_ha.pt` (selHA), with the relaxed L2
  decoder.
- **Greedy.** Two decoders: level1 and relaxed L2.
- **Resolvers on the GNN's tree of blobs.** TREE-QMC `--blob` does not reach n = 1000, so there is
  no TOB-QMC row. `stage_gnn_tob.py` collapses the **selF1** network to its tree of blobs, and
  BROOQS (`--rooted --dist-var`) and NetCS re-resolve it. The labels are `BROOQS (GNN TOB)` and
  `NetCS (GNN TOB)`.
- **Optional oracle arm.** The same resolvers on the true TOB (`--variants true`), labelled
  `BROOQS*(true TOB)` and `NetCS*(true TOB)`.

### Why its own package
`l2reg_eval` assumes n = 200 throughout:
- `N_TAXA`;
- the `inputs/n200` paths;
- the luay denominators.

It also reads a TREE-QMC TOB unconditionally, and its tree-count gate reads MANIFEST `n_loci`. Here
`n_loci` is the 50k *true* count; the estimated arm is `n_est_loci` = 10,000.

So this package copies l2reg's drivers, as l3reg_eval copies l2reg_eval. `make_csv.py` goes
further and *imports* l2reg's `make_csv.py` by path, with `common` bound here. Luay, hybrid
ancestry, the m/τ gate and the GNN/greedy row builders are therefore the landed code, not copies.

Two edits outside this package:
- `gnn_cpu_v1/pylib/opc/rl/camus.py` gets `PHASES["l2_n1000_gt50k"] = ("n1000", 1000)`.
- `plotting_n1000.ipynb` is added at the repo root.

## Files

| file | what |
|---|---|
| `common.py` | paths, labels, constants (`M, TAU = 10000, 20`), and `tree_of_blobs_general` (any level) |
| `stage.py` | stages the inputs (see below), with gates; writes `CELLS.tsv` and `stage.json` |
| `run_greedy.py` | l2reg's driver at `C.M, C.TAU`; always records `m` and `tau` |
| `stage_gnn_tob.py` | GNN selF1 network → `tob/<key>.gnn.rooted.nwk`, plus `tob/gnn_tob.json`, with gates |
| `run_resolvers.py` | BROOQS + NetCS per (network, variant): timeouts, the stale-TOB guard, invariants |
| `make_csv.py` | the per-network CSV and its summary JSON |

`stage.py` writes:
- `inputs/n1000/<RR>/{true_net.nwk, est_bp1000.nwk}`, 6.6 GB in all;
- `tob/<key>.true.rooted.nwk`.

## The GNN's tree of blobs

- **`camus_nets.tree_of_blobs` is level-1 only.** It contracts one cycle per reticulation. The
  collapse here instead keeps the cluster below every bridge (`pte_common.bridge_tob_clusters`) and
  writes their Hasse diagram, so a blob of any level becomes one polytomy.
  - Unit-checked: two overlapping cycles, one level-2 blob, become one 4-child polytomy.
  - On level-1 GNN networks, `stage_gnn_tob.py` cross-checks the result against `tree_of_blobs`.
- **Not every polytomy is a blob.** A GNN selection is not a binary tree, so the network's tree
  part keeps multifurcations. These survive the collapse and look exactly like blobs to a resolver.
  - Both resolvers turn every eligible polytomy into one reticulation, so they over-reticulate here
    by construction. `prefix_tob_eval` found the same for its GNN-TOB arm, and this keeps its
    protocol: the TOB is fed as is.
  - `gnn_tob.json` records `blob_polytomy_degrees` and `tree_polytomy_degrees` separately, and
    `n_eligible_*` both in total and blob-only.
  - On rep00: 10 polytomies, 6 blobs + 4 tree multifurcations. 10 are BROOQS-eligible, 5
    NetCS-eligible.
- **The TOB knows k.** The GNN decoded with k_true as its budget.

### Gates

| gate | checks |
|---|---|
| `stage.py` | the names resolve; the truth round-trips (clusters, k, node count, level 2); 10,000 trees per network, each with branch lengths, and exactly t1..t1000 after rooting on OUT; true-TOB polytomies = n_l2 + n_l1 = cycle components; every true-TOB cluster is hardwired in the truth |
| `stage_gnn_tob.py` | the taxa match; every TOB cluster is a hardwired cluster of the GNN network; a string round-trip; the polytomy degrees equal blob out-degrees + tree multifurcations, computed independently; the level-1 cross-check (the count is printed) |
| `run_resolvers.py` | **stale TOB**: the TOB sha and the GNN cell sha equal `gnn_tob.json` and the cell on disk. **Invariants** on every output: k_pred ≤ eligible; BROOQS k_pred = eligible; TOB splits = output bridge splits |
| `make_csv.py` | `stage.json` is this dataset; `n_est_loci == 10000`; every GNN/greedy cell has `(n_loci, m, tau) == (10000, 10000, 20)` and the GNN has `tau_frac == 0.002`; the cells are complete; each resolver cell's TOB sha is current; no violations |

## Checked (2026-10-06, before hand-over)

- **Staging.** All 20 networks pass every gate, in 232 s at `--jobs 8`. The true TOBs have 2–6
  blobs with 4–34 children each.
- **GNN smoke, rep00.**
  - |C| = 33,508 at τ = 20, `n_loci = m = 10000`.
  - 195 s shared preparation, then 52 s forward + 9 s decode per model: **~7 min per cell for both
    models**.
  - **Peak RSS 5.8 GB.**
  - F1: selF1 0.947, selHA 0.952.
- **Greedy smoke, rep00.** 139 s preparation + 5 / 9 s decode (level1 / relaxed), so **~2.5 min
  per cell per decoder**. Peak RSS 4.0 GB. F1 0.892.
- **`stage_gnn_tob.py`** passes on the smoke cell, with the level-1 cross-check run on 1 of 1.
- **`make_csv.py`.**
  - Luay at n = 1000 takes 3 s for 45 networks.
  - Hybrid ancestry: the two derivations agree on 45 of 45.
  - The m/τ gate refuses a hand-edited cell claiming 1,000 trees, so it is not vacuous.
- **Resolver probe, rep00 (gnn and true TOB, capped at 30 min per tool).** **All four runs hit the
  cap.**

  | resolver | progress at 30 min | peak RSS |
  |---|---|---|
  | NetCS | 1 of 5 blobs (GNN TOB), 3 of 4 (true TOB) | 11.8–12.1 GB |
  | BROOQS | still on its first blob | 5.4 GB |

  - Both are single-threaded: CPU ≈ wall.
  - The cap path works: kill, exact CPU and RSS, `status = timeout` in the CSV, and the notebook
    still runs.
  - **Expect hours per network per resolver.** That is a rough extrapolation, not a measurement;
    the first finished tasks will measure it. At n = 200 / 10k the longest task was about 80 min.

## Command sheet (zsh; from the repo root, in a terminal)

```bash
PY=gnn_cpu_v1/.venv/bin/python

# 0. ALREADY DONE (2026-10-06): staging, every gate passing
# $PY n1000_eval/stage.py --jobs 8

# 1. GNN: 4 shards of 5 cells, one terminal each (or & them).  ~7 min/cell, 5.8 GB each -> ~35 min.
(cd gnn_cpu_v1 && caffeinate -i .venv/bin/python -u run_gnn_cpu.py --threads 1 --models _l2mix_pair/s0,_l2mix_pair/s0/best_ha --m 0 --tau-frac 0.002 --decoder relaxed --max-level 2 --inputs ../n1000_eval/inputs --cells ../n1000_eval/CELLS.tsv --reference ../n1000_eval/NO_REFERENCE.tsv --out ../n1000_eval/runs --run-name gnn_t1_l2mix_pair_ftau0.002_relaxedL2 --reps 0-4)
#    ... the same line with --reps 5-9, --reps 10-14 and --reps 15-19

# 2. greedy, alongside step 1: ~2.5 min/cell, 4 GB -> ~50 min per decoder
caffeinate -i $PY -u n1000_eval/run_greedy.py
caffeinate -i $PY -u n1000_eval/run_greedy.py --decoder relaxed --max-level 2

# 3. after step 1 finishes: collapse the 20 selF1 networks to trees of blobs (seconds, gated)
$PY n1000_eval/stage_gnn_tob.py

# 4. resolvers on the GNN TOB: 20 tasks.  ~12 GB per worker (NetCS), so --jobs 4 on 64 GB, with
#    nothing else running.  12 h cap per tool; resumable, and a re-run with a longer cap redoes
#    only the timed-out tools.
caffeinate -i $PY -u n1000_eval/run_resolvers.py --variants gnn --jobs 4 --timeout-brooqs 43200 --timeout-netcs 43200

# 4b. OPTIONAL oracle arm on the true TOB: the same cost again
caffeinate -i $PY -u n1000_eval/run_resolvers.py --variants true --jobs 4 --timeout-brooqs 43200 --timeout-netcs 43200

# 5. then tell me: I run make_csv.py and execute plotting_n1000.ipynb
$PY n1000_eval/make_csv.py
# GNN + greedy (+ TOB rows) only, complete without resolvers -- what was built 2026-10-07:
$PY n1000_eval/make_csv.py --no-resolvers
```

## Traps

- **Killing `run_resolvers.py` does not kill the tools.** `run_timed_to` starts each tool in its own
  session (so a timeout can kill its whole process group), so Ctrl-C on the driver orphans every
  running `tree-qmc` / `brooqs`. They keep running under launchd, holding up to 12 GB each. On
  2026-10-07, 16 such orphans from several launches were found and killed. After stopping a run,
  clean up (the `xargs` matters: in zsh an unquoted multi-line `$pids` is ONE argument):

  ```bash
  for p in $(pgrep -f "blob_cpu_v1/(tools/TREE-QMC/build/tree-qmc|.venv-brooqs)"); do lsof -a -p $p -d cwd -Fn | grep -q n1000_eval/runs && echo $p; done | xargs kill -9
  ```

- **`n_loci` is not the estimated count.** MANIFEST `n_loci` = 50,000 counts the true trees.
  Everything here gates on `n_est_loci` = 10,000.
- **A GNN cell's `key` is CamusCell's (`n1000_repRR`), not the dataset's.** `stage_gnn_tob.py`
  keys by the dataset key from `stage.json`. The first version took the cell's key, and the
  resolver self-check caught it.
- **Re-running the GNN invalidates the GNN TOBs.** After any GNN re-run, run `stage_gnn_tob.py`
  again, then `run_resolvers.py --force` for the affected keys. `run_resolvers.py` and `make_csv.py`
  both refuse stale TOBs.
- **NetCS memory at n = 1000** is about 10× its n = 200 figure; see the probe. Size `--jobs` from
  RAM, not from cores.
- **The PhyloNet cache** lives in `runs/phylonet/`. Do not run two `make_csv.py` builds at once.

## Status (2026-10-07)

- **Done:** GNN (20 cells × 2 models), greedy (20 × 2 decoders), GNN TOBs (20, every gate passing).
- **Built:** `results/rl/l2_n1000_gt50k_v1_per_network.csv` with `--no-resolvers`: 120 rows,
  `complete: true`, scope recorded in the JSON. `plotting_n1000.ipynb` executed and wrote
  `L2n1000-*-ours_l2mix_pair_selF1.png`.
- **Resolvers moved to a bigger box (2026-10-07):** `remote_pkg/` is the hand-off. It holds the 40
  TOBs, a sha256 manifest, a stdlib-only runner and a tool-build script, and its README covers
  send → run → rsync back. `make_remote_pkg.py` rebuilds it and refuses stale TOBs. Results come
  back to `runs/resolvers_remote/` and are scored with `run_resolvers.py --score-only`.
- **Paused locally:** resolvers. 4 of 20 GNN-TOB tasks finished before the run was stopped and are kept
  under `runs/resolvers/cells/`; a re-run resumes from there.
- **2026-10-21: the remote GNN-TOB arm is complete.** All 20 NetCS and all 20 BROOQS runs were rsynced back
  into `runs/resolvers_remote/work/`. Every run has rc 0 and none timed out (7-day cap). NetCS took up to 89.8 h
  wall and 13.8 GB.
  - `run_resolvers.py --run-name resolvers_remote --score-only --jobs 4`: 20/20 ok for both tools. 7 NetCS outputs
    are `out_not_rootable` (rep00, 01, 10, 13, 14, 15, 17). The 13 cells scored on 2026-10-09 re-derive identically
    except `t_start` / `t_end`.
  - `make_csv.py --resolver-run n1000_eval/runs/resolvers_remote` passes every gate and writes
    `l2_n1000_gt50k_v1_per_network.csv`: 160 rows, `complete: true`, scope `gnn, greedy, tob, resolvers`.
    - Its 120 GNN, greedy and TOB rows are identical to the 2026-10-07 `--no-resolvers` build.
    - 153 of its 160 rows are identical to the 2026-10-09 `_partial` build, which is kept as it was. The other 7 are
      the new NetCS rows.
  - `plotting_n1000.ipynb` has not been re-run on the complete file.
