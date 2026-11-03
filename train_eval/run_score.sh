#!/bin/bash
# Score every trn cell (reps 0-799, both arms, six phases) with cluster_anatomy/score_cells.py.
#
# 18 jobs, at most MAXJ at once: each n = 50 phase x arm in two rep halves (~40 min each), each
# n = 25 phase x arm whole (~36 min), heaviest first.  Measured on the smoke cells (2026-10-22):
# 2-3 s per n = 25 cell, 4-9 s per n = 50 cell, peak RSS <= 2.5 GB per process, ~45 KB written per
# cell.  Resumable: a cell whose JSON exists is skipped.  Then --summary-only (exit 3 on a parity
# failure; trn has no landed run, so every parity is ok: None).
#
#   caffeinate -i bash train_eval/run_score.sh            # MAXJ=6 by default
set -u
cd "$(dirname "$0")/.."
PY=gnn_cpu_v1/.venv/bin/python
MAXJ=${MAXJ:-6}
LOGS=train_eval/logs
mkdir -p "$LOGS"

jobs=()
for p in trn_l2t50 trn_x50 trn_y50; do
  for a in trn_true trn_est_bp1000; do
    jobs+=("$p $a 0-399" "$p $a 400-799")
  done
done
for p in trn_l2t25 trn_x25 trn_y25; do
  for a in trn_true trn_est_bp1000; do
    jobs+=("$p $a 0-799")
  done
done

run_one() {
  set -- $1
  local log="$LOGS/score_$1_$2_$3.log"
  if gnn_cpu_v1/.venv/bin/python -u cluster_anatomy/score_cells.py --dataset trn --phases "$1" \
       --arms "$2" --reps "$3" > "$log" 2>&1; then
    echo "[run_score] ok   $1 $2 $3 ($(grep -c '^\[score\] *[0-9]*/' "$log") cells)"
  else
    echo "[run_score] FAIL $1 $2 $3 -- see $log"
  fi
}
export -f run_one
export LOGS

echo "[run_score] ${#jobs[@]} jobs, $MAXJ at a time, from $(date '+%H:%M:%S')"
printf '%s\n' "${jobs[@]}" | xargs -P "$MAXJ" -I{} bash -c 'run_one "{}"'
$PY cluster_anatomy/score_cells.py --dataset trn --summary-only > "$LOGS/score_summary.log" 2>&1
rc=$?
tail -20 "$LOGS/score_summary.log"
n=$(ls cluster_anatomy/runs/trn/cells/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[run_score] done $(date '+%H:%M:%S'): $n of 9600 cell JSONs; summary exit $rc"
[ "$rc" = 0 ] && [ "$n" = 9600 ]
