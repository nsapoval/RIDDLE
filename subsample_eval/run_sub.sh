#!/usr/bin/env bash
# The subsample runs: _l2mix_pair/s0 (selF1 + selHA) and greedy, level1 and relaxed at the dataset's
# level, on the staged 500-tree inputs -- see subsample_eval/README.md.  From the repo root:
#
#   caffeinate -i bash subsample_eval/run_sub.sh camus 0 1 2 3 4       # one process per draw
#   caffeinate -i bash subsample_eval/run_sub.sh l2reg                 # every draw (0-4) at once
#   caffeinate -i bash subsample_eval/run_sub.sh netcs 0 1 2 3 4 5     # netcs: the args are SHARDS
#   caffeinate -i bash subsample_eval/run_sub.sh netcs 6 7 8 9 10 11   #   (draw 0); waves of 6
#   bash subsample_eval/run_sub.sh camus full                          # the full-data relaxed greedy
#
# Each process runs, one after another, into subsample_eval/runs/<ds>/d<k>/:
#   gnn_t1_l2mix_pair              run_gnn_cpu.py --m-from-trees, level1
#   gnn_t1_l2mix_pair_relaxedL<L>  the same, --decoder relaxed --max-level L (L = 3 on l3reg, else 2)
#   greedy, greedy_relaxedL<L>     netcs_eval/run_greedy.py (per-file m; any CamusCell)
# `camus full` runs only greedy_relaxedL2 on the ORIGINAL 1,000-tree camus inputs, into
# runs/camus/full/: the one full-data baseline no landed run holds.
#
# Environment:
#   STEPS="gnn greedy"         which drivers
#   DECODERS="level1 relaxed"  which decoders
#   MODELS                     run_gnn_cpu.py --models (default _l2mix_pair/s0,_l2mix_pair/s0/best_ha)
#   THREADS=1                  --threads per GNN process
#   OUT_ROOT                   default subsample_eval/runs (a smoke points it at a scratch directory)
#   CELLS                      a cells file to use instead of the dataset's (or the netcs shard's)
#
# Every driver resumes per cell, so rerunning after an interruption continues where it stopped.
# Processes of one dataset share nothing but netcs's run directories, where each cell JSON is
# written atomically under its own name.  Logs: <OUT_ROOT>/<ds>/logs/{d<k>,shard_XX}.log; a failed
# step prints "exited" there and the process moves on.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=$REPO/gnn_cpu_v1/.venv/bin/python
DS=${1:-}
[ $# -gt 0 ] && shift
case $DS in
  camus) DS_CELLS=$REPO/gnn_cpu_v1/CELLS.tsv;  L=2; ALL="0 1 2 3 4" ;;
  l2reg) DS_CELLS=$REPO/l2reg_eval/CELLS.tsv;  L=2; ALL="0 1 2 3 4" ;;
  l3reg) DS_CELLS=$REPO/l3reg_eval/CELLS.tsv;  L=3; ALL="0 1 2 3 4" ;;
  netcs) DS_CELLS=$REPO/netcs_eval/CELLS.tsv;  L=2; ALL="" ;;
  *) echo "usage: $0 {camus|l2reg|l3reg|netcs} [draws | netcs shards | camus full]" >&2; exit 2 ;;
esac
MODELS=${MODELS:-_l2mix_pair/s0,_l2mix_pair/s0/best_ha}
STEPS=${STEPS:-gnn greedy}
DECODERS=${DECODERS:-level1 relaxed}
THREADS=${THREADS:-1}
OUT_ROOT=${OUT_ROOT:-$REPO/subsample_eval/runs}
case $OUT_ROOT in /*) ;; *) OUT_ROOT=$REPO/$OUT_ROOT ;; esac
CELLS=${CELLS:-}
case $CELLS in ""|/*) ;; *) CELLS=$REPO/$CELLS ;; esac

stamp() { date '+%F %T'; }

# <inputs> <cells file> <out dir>
steps() {
  local inputs=$1 cells=$2 out=$3 step dec name flags
  for step in $STEPS; do
    for dec in $DECODERS; do
      if [ "$dec" = level1 ]; then flags=(); sfx=""
      else flags=(--decoder relaxed --max-level "$L"); sfx="_relaxedL$L"; fi
      if [ "$step" = gnn ]; then
        name=gnn_t1_l2mix_pair$sfx
        echo "[$(stamp)] $name"
        (cd "$REPO/gnn_cpu_v1" && "$PY" -u run_gnn_cpu.py --threads "$THREADS" --m-from-trees \
            --models "$MODELS" --inputs "$inputs" --cells "$cells" \
            --reference "$REPO/subsample_eval/NO_REFERENCE.tsv" \
            --out "$out" --run-name "$name" ${flags[@]+"${flags[@]}"}) || echo "[$(stamp)] $name exited $?"
      else
        name=greedy$sfx
        echo "[$(stamp)] $name"
        "$PY" -u "$REPO/netcs_eval/run_greedy.py" --inputs "$inputs" --cells "$cells" \
            --out "$out" --run-name "$name" ${flags[@]+"${flags[@]}"} || echo "[$(stamp)] $name exited $?"
      fi
    done
  done
  echo "[$(stamp)] done"
}

if [ "${1:-}" = full ]; then
  [ "$DS" = camus ] || { echo "'full' is camus-only: every other full-data run is landed" >&2; exit 2; }
  L=2 STEPS=greedy DECODERS=relaxed steps "$REPO/gnn_cpu_v1/inputs" "${CELLS:-$DS_CELLS}" \
      "$OUT_ROOT/camus/full"
  exit 0
fi

[ -f "$REPO/subsample_eval/NO_REFERENCE.tsv" ] && {
  echo "subsample_eval/NO_REFERENCE.tsv must not exist (common.NO_REFERENCE)" >&2; exit 1; }
if [ $# -eq 0 ]; then
  [ -n "$ALL" ] || { echo "netcs: name the shards, in waves of 6 (see README)" >&2; exit 2; }
  set -- $ALL
fi
mkdir -p "$OUT_ROOT/$DS/logs"
for a in "$@"; do
  if [ "$DS" = netcs ]; then
    d=0; log=shard_$(printf %02d "$a")
    cells=${CELLS:-$REPO/netcs_eval/shards/CELLS_$(printf %02d "$a").tsv}
  else
    d=$a; log=d$a; cells=${CELLS:-$DS_CELLS}
  fi
  inputs=$REPO/subsample_eval/inputs/$DS/d$d
  [ -d "$inputs" ] || { echo "no $inputs: run subsample_eval/stage.py" >&2; exit 1; }
  [ -f "$cells" ] || { echo "no $cells" >&2; exit 1; }
  steps "$inputs" "$cells" "$OUT_ROOT/$DS/d$d" > "$OUT_ROOT/$DS/logs/$log.log" 2>&1 &
done
echo "running $DS: $* -- tail -f $OUT_ROOT/$DS/logs/*.log"
wait
echo "all finished; grep exited $OUT_ROOT/$DS/logs/*.log should print nothing"
