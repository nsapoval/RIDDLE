#!/usr/bin/env bash
# W13 widened NODE_FEATURES from 10 to 31 (the four gene-tree branch-length blocks plus
# `bconf_*`), so every cached instance is now refused by `instances.check_features` -- which since
# W13 checks the *node* width as well as the global one.  Rebuild, smallest phase first so a
# mistake is cheap.
#
# `rebuild_instances_a97.sh`'s shape, and for the same reasons: deleting before rebuilding keeps
# disk flat, and nothing is lost that is not derivable -- the gene trees, networks and cluster
# tables the instances are built from are ~1 GB and untouched.  **No `rungrid` re-run**: W13 reads
# the branch lengths straight out of `genetrees_julia/`, which every phase still has, and touches
# no parquet schema.
#
# The skip check samples the *first* pickle, as A97's did.  That is enough because the cache is
# written by one process per phase and `load_or_build` renames atomically, so a phase is never
# half one feature set and half the other.
#
# **Kill THIS pid first**, then descendants.
set -u
cd "$(dirname "$0")/../.."
run() {  # $1 phase  $2 reps
  local ph="$1" reps="$2" d="results/$1/instances_tau2"
  if [ -d "$d" ] && .venv/bin/python -c "
import sys, glob, pickle
sys.path.insert(0, 'src')
from opc.rl.instances import GLOBAL_FEATURES, NODE_FEATURES
f = sorted(glob.glob('$d/*.pkl'))
if not f:
    sys.exit(1)
inst = pickle.load(open(f[0], 'rb'))
sys.exit(0 if len(inst.x_global) == len(GLOBAL_FEATURES)
              and inst.x_nodes.shape[1] == len(NODE_FEATURES) else 1)
" 2>/dev/null; then
    echo "[w13] $ph already at the current feature set -- skipping"; return
  fi
  echo "[w13] === $ph: removing the stale cache and rebuilding ==="
  rm -rf "$d"
  .venv/bin/python -u scripts/rl_instances.py --phase "$ph" --tau 2 --reps "$reps" \
      --include-k0 --jobs 12 || echo "[w13] FAILED $ph"
  df -h /home | tail -1
}
run b25 100
run b50 100
run b100 100
run x25 900
run x50 900
run b250 100
echo "[w13] done"
