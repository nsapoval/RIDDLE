#!/usr/bin/env bash
# The GNN and greedy runs for netcs_eval, sharded -- see netcs_eval/README.md.  From the repo root:
#
#   caffeinate -i bash netcs_eval/run_shards.sh            # all 12 shards at once
#   caffeinate -i bash netcs_eval/run_shards.sh 0 1 2      # only these shards
#
# Each shard runs its cells (shards/CELLS_XX.tsv) through, one after another:
#   gnn_t1{,_relaxedL1,_relaxedL2}      _mix, _mix2L, _mix75 x seeds 0-2  (run_gnn_cpu.py)
#   gnn_t1_l2{,_relaxedL1,_relaxedL2}   _l2mix/s0
#   gnn_t1_l2mix_s1_l2mix75_s0{,...}    _l2mix/s1 and _l2mix75/s0 (added 2026-09-13; --models)
#   gnn_t1_l2mixha{,...}                _l2mixha/s0's best.pt and best_ha.pt (added 2026-09-14)
#   gnn_t1_l2mix_pair{,...}             _l2mix_pair/s0's best.pt and best_ha.pt, pair conv (2026-09-14)
#   greedy{,_relaxedL1,_relaxedL2}      (run_greedy.py)
# Every driver is resumable per cell, so running this again after an interruption continues where it
# stopped.  All shards share the run directories: each cell JSON is written atomically under its
# own name.  The per-run timings.tsv / summary.txt are rewritten by whichever shard finishes last,
# so regenerate them afterwards if you want them (README).  Logs: netcs_eval/runs/logs/shard_XX.log;
# a failed step prints "exited" there and the shard moves on.
set -u
cd "$(dirname "$0")/.." || exit 1
if [ $# -eq 0 ]; then set -- 0 1 2 3 4 5 6 7 8 9 10 11; fi
mkdir -p netcs_eval/runs/logs

shard() {
  local i cells dec
  i=$(printf %02d "$1")
  cells="netcs_eval/shards/CELLS_$i.tsv"
  for dec in "" relaxedL1 relaxedL2; do
    local flags=() suffix=""
    if [ -n "$dec" ]; then flags=(--decoder relaxed --max-level "${dec#relaxedL}"); suffix="_$dec"; fi
    # <run name> <--tags or --models> <value>.  _l2mix is named by seed: --tags would take both of
    # its seeds, and the third set already scores _l2mix/s1.
    for set in "gnn_t1 --tags _mix,_mix2L,_mix75" "gnn_t1_l2 --models _l2mix/s0" \
               "gnn_t1_l2mix_s1_l2mix75_s0 --models _l2mix/s1,_l2mix75/s0" \
               "gnn_t1_l2mixha --models _l2mixha/s0,_l2mixha/s0/best_ha" \
               "gnn_t1_l2mix_pair --models _l2mix_pair/s0,_l2mix_pair/s0/best_ha"; do
      local name=${set%% *} rest=${set#* }
      local opt=${rest%% *} val=${rest#* }
      echo "[shard $i] $(date '+%F %T') $name$suffix"
      (cd gnn_cpu_v1 && .venv/bin/python -u run_gnn_cpu.py --threads 8 --m-from-trees \
         --inputs ../netcs_eval/inputs --cells "../$cells" "$opt" "$val" \
         --reference ../netcs_eval/NO_REFERENCE.tsv --out ../netcs_eval/runs \
         --run-name "$name$suffix" ${flags[@]+"${flags[@]}"}) || echo "[shard $i] $name$suffix exited $?"
    done
  done
  for dec in "" relaxedL1 relaxedL2; do
    local flags=()
    if [ -n "$dec" ]; then flags=(--decoder relaxed --max-level "${dec#relaxedL}"); fi
    echo "[shard $i] $(date '+%F %T') greedy${dec:+_$dec}"
    gnn_cpu_v1/.venv/bin/python -u netcs_eval/run_greedy.py --cells "$cells" \
      ${flags[@]+"${flags[@]}"} || echo "[shard $i] greedy${dec:+_$dec} exited $?"
  done
  echo "[shard $i] $(date '+%F %T') done"
}

for s in "$@"; do
  shard "$s" > "netcs_eval/runs/logs/shard_$(printf %02d "$s").log" 2>&1 &
done
echo "running shards: $* -- tail -f netcs_eval/runs/logs/shard_*.log"
wait
echo "all shards finished; grep exited netcs_eval/runs/logs/shard_*.log should print nothing"
