# n1000 resolver hand-off — NetCS and BROOQS on the GNN's trees of blobs

**One job:** run NetCS and BROOQS on 20 networks (n = 1000, 10,000 IQ-TREE gene trees each).
Each network gets the tree of blobs our GNN's selF1 network collapses to. Then send `out/` back.
Scoring, the CSV and the plots all happen on the Mac (`n1000_eval/`).

| file | what |
|---|---|
| `run_remote.py` | the runner: Python ≥ 3.9 **standard library only**, with no repo code and no venv. Run it as `$PY` (the tools env's Python 3.12, set by `env.sh`): the box's system `python3` is 3.6 |
| `MANIFEST.json` | per network: the sha256 of the dataset's `.raw.nwk.gz`, of its decompression, and of each TOB; eligible blob counts; the tool pins the Mac ran |
| `tobs/<key>.gnn.rooted.nwk` | the 20 GNN trees of blobs, as `(OUT,<tob>);`, which is what both tools are fed |
| `tobs/<key>.true.rooted.nwk` | the 20 true trees of blobs, for the optional oracle arm |
| `setup_tools.sh` | builds TREE-QMC `e577f8c` + `treeqmc-e577f8c.patch`, and BROOQS `17b25b2`, into `tools/` |

The gene trees are **not** in the package. The box already has the dataset, and the runner
decompresses `<dataset>/estimated_bp1000/<key>.raw.nwk.gz` itself, checking two sha256s:
- the `.gz` against the manifest;
- the decompressed file against the bytes the Mac's GNN read.

## What it costs (measured on the Mac, 4 workers, under swap pressure)

| tool | wall per network | peak RSS | notes |
|---|---|---|---|
| BROOQS | 1.0–3.3 h (4 of 4 finished) | 5.0–5.4 GB | single-threaded |
| NetCS | 3.4 h on rep02; **> 12 h** on rep00, rep01, rep03 | 6.6–12.1 GB | single-threaded; the long pole |

- Every tool is one process with one thread. Wall time for the grid is therefore roughly the
  slowest single NetCS run, provided all 40 tasks fit in memory at once.
- That takes about 20 × 14 GB + 20 × 6 GB ≈ **400 GB** with `--jobs 40`.
- With less RAM, the runner's memory budget queues tasks: NetCS first and longest-first, with BROOQS
  filling the gaps.
- The default caps are 7 days for NetCS and 2 days for BROOQS. A run that hits its cap is recorded
  as a timeout, never dropped. Re-running with a longer cap redoes only those tasks.

## 1. Send the package (on the Mac)

```bash
REMOTE=<user>@<host>                                  # the Linux box
DEST=/shared/ns58/rl-net-opt/n1000_resolvers_pkg      # anywhere off /home (it is full)
cd ~/research/rl-net-opt
gnn_cpu_v1/.venv/bin/python n1000_eval/make_remote_pkg.py     # refresh; refuses stale TOBs
rsync -av n1000_eval/remote_pkg/ $REMOTE:$DEST/
```

## 2. Tools (on the remote, once)

If the box already has TREE-QMC and BROOQS **at these pins**, skip the build:
- TREE-QMC 4.1.5 @ `e577f8c`, patched with `treeqmc-e577f8c.patch`;
- BROOQS @ `17b25b2`.

Point the runner at them with `export TREE_QMC=... BROOQS=...`, plus that tree-qmc's `R_HOME` and
`LD_LIBRARY_PATH`. Otherwise build them:

```bash
cd $DEST
bash setup_tools.sh        # ~15-20 min; conda env + R + MSCquartets, TREE-QMC, BROOQS -> tools/
source env.sh              # R_HOME, LD_LIBRARY_PATH, TREE_QMC, BROOQS, PY -- every new shell
```

`--check` prints the remote's TREE-QMC commit and patch fingerprint, and its BROOQS environment,
next to the Mac's. They should agree. Even when they do, NetCS output will not be bit-identical to
the Mac's: `rand()` and `unordered_set` order differ between libc++ and glibc, as
`blob_cpu_v1/README.md` measured. BROOQS is seeded and should match.

### Build notes (what went wrong on 2026-10-07/08, and how the script handles it now)

The env is created in **one solve**:
- R 4.4 with Rcpp and RInside, plus `compilers`, Python 3.12 and git;
- **`sysroot_linux-64` pinned up front**: 2.28 if the host glibc has it, else 2.17.

After that, `setup_tools.sh` only **checks** the env and never installs into it:
- the sysroot's `time.h` declares `timespec_get`;
- the sysroot is no newer than the host glibc;
- gxx is at least 14 (RInside needs `CXXABI_1.3.15`), and libstdcxx is no older than gxx. The
  runtime running a major ahead of the compiler is normal: 16.2 with 15.3 was seen here;
- a `<ctime>` probe compiles and runs;
- a probe executable links against RInside.

Any failure stops the script. It prints the relevant package versions and says to remove only
`tools/env` and re-run. The three errors this replaces:

| error | cause |
|---|---|
| `'timespec_get' has not been declared in '::'` | the env resolved an old (2.12) glibc sysroot |
| `undefined reference to __cxa_call_terminate@CXXABI_1.3.15` | GCC 13, pinned by a mistaken fix for the above, against a GCC 14+ libstdcxx |
| a conda conflict dump | patching pins into an existing env, one fix after another |

To reuse a known-good R env instead, for example `tob_qmc_v1`'s `treeqmc-env` if this box still
has it, run `TQMC_ENV=/path/to/env bash setup_tools.sh`.

## 3. Check, then run (on the remote)

```bash
cd $DEST && source env.sh
DATA=/shared/ns58/rl-net-opt/l2_n1000_gt50k_v1        # holds estimated_bp1000/*.raw.nwk.gz
$PY run_remote.py --dataset $DATA --check         # stages 20 x 330 MB trees, verifies every sha256
$PY run_remote.py --dataset $DATA --dry-run --jobs 40

# the grid -- inside tmux (or nohup), so a dropped ssh does not end it
tmux new -s n1000
$PY run_remote.py --dataset $DATA --jobs 40 2>&1 | tee -a run.log
#   optional oracle arm, same cost again:  add  --variants gnn,true  (80 tasks; keep --jobs <= 48)
```

- **`--jobs` is capped by memory, not cores.** The runner reserves 14 GB per NetCS run and 6 GB per
  BROOQS run (`--mem-netcs`, `--mem-brooqs`) inside `--mem-gb`, which defaults to 85% of MemTotal.
  It never starts a task that would exceed that. If the box is shared, pass `--mem-gb` explicitly.
- **Keep `--jobs` ≤ 48 on the shared box.** A per-user thread limit has hung process pools there
  before (repo `CLAUDE.md`). This runner uses no pool, but the limit still counts its tools.
- **Stopping is safe.** Ctrl-C (or a kill of `run_remote.py`) kills every running tool's process
  group before exiting, so nothing is orphaned. Running the same command again resumes, and only
  the tasks that were in flight start over.
- **Disk.** About 6.6 GB of decompressed trees in `inputs/`, plus small outputs.

### Monitor

```bash
$PY run_remote.py --status                        # one line per task, from disk, any time
grep -E "finish|TIMEOUT|FAILED" run.log
ps -u $USER -o pid,etime,rss,args | grep -E "tree-qmc|brooqs" | grep -v grep
```

## 4. Send results back (on the Mac)

```bash
rsync -av $REMOTE:$DEST/out/ n1000_eval/runs/resolvers_remote/work/
```

`out/<key>/<variant>/` already has the layout `n1000_eval/runs/<run>/work/` uses. Each method's
`.json` carries the sha256 of the TOB it ran on, and the remote's tool versions are in
`out/remote_header_*.json`.

## 5. Score, build the CSV, plot (on the Mac)

```bash
PY=gnn_cpu_v1/.venv/bin/python
$PY n1000_eval/run_resolvers.py --run-name resolvers_remote --score-only --jobs 4
#   + --variants gnn,true if the oracle arm ran
$PY n1000_eval/make_csv.py --resolver-run n1000_eval/runs/resolvers_remote
~/miniconda3/envs/phylo-env/bin/jupyter nbconvert --to notebook --execute --inplace plotting_n1000.ipynb
```

- `--score-only` launches no tool. It scores the records that came back, checking for each one:
  - that it ran on exactly this TOB (by sha256);
  - the resolver invariants.

  A network with no record becomes an error row (`not run`), not a missing one.
- `make_csv.py` refuses to write while any GNN-TOB task is missing.
- The Mac's own partial run (`runs/resolvers/`: 4 BROOQS and 1 NetCS finished) is kept apart. It
  can serve as a Mac-vs-Linux parity check: BROOQS should match exactly, and NetCS need not.

**Checked before hand-over (2026-10-07, on the Mac with the system Python 3.9):**
- `--check` staged and verified a network, and the tool fingerprints matched the manifest.
- A 15–20 s cap recorded both tools as timeouts and left no process behind.
- SIGINT mid-run killed both tools (2 → 0 processes) and wrote no records.
- `--score-only` on the Mac's finished rep02 outputs reproduced their k, F1, cluster counts and
  rooted network exactly, and reported a network with no record as `not run`.
