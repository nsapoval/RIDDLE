#!/usr/bin/env bash
# The GNN (ours/l2mix_pair/selF1, the relaxed decoder at the family's level) on the staged
# missing-taxa conditions -- see taxmiss_eval/README.md.  From the repo root:
#
#   caffeinate -i bash taxmiss_eval/run_gnn.sh camus,l2reg,l3reg        # every staged condition
#   caffeinate -i bash taxmiss_eval/run_gnn.sh l2reg p0.90_c24          # named conditions
#
# One run_gnn_cpu.py process per (dataset, condition, phase) -- l2reg / l3reg, which have one phase,
# per rep half -- at most MAXJ at once, each into
# taxmiss_eval/runs/<ds>/<cond>/gnn_t1_l2mix_pair_relaxedL<L>/ with --partial 4 (trees missing taxa;
# each partial tree's own leaf set is dropped from its clusters) and --m-from-trees (m = the loci
# the file holds: netcs's corpus ships 911-1000 per cell, and every published netcs launcher passes
# it; a no-op on camus / l2reg / l3reg, whose files hold 1,000).  --reference points at a file
# that must not exist, so no parity is checked against the full-data reference.tsv.  Every cell
# resumes; a process with nothing left to do exits at once.  At the end every staged cell must
# have its cell JSON, or the script exits 1.
#
# Environment:
#   MAXJ [6]     concurrent processes (one thread each; <= 0.9 GB at n = 200 on camus/l2reg/l3reg --
#                a netcs n = 200 worker peaked at 3.6 GB in the 2026-10-23 smoke, so keep netcs at 6)
#   OUT_ROOT     default taxmiss_eval/runs (a smoke points it at a scratch directory)
#   PHASES       only these phases (comma list), e.g. cam50 for a smoke
#   REPS         run_gnn_cpu.py --reps for every process, e.g. 0 for a smoke (replaces the rep halves)
#   DRY=1        pass --dry-run: list what each process would run, and skip the completeness check
# bash 3.2 (macOS /bin/bash): no wait -n, no associative arrays.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=$REPO/gnn_cpu_v1/.venv/bin/python
DSS=${1:-}
[ $# -gt 0 ] && shift
[ -n "$DSS" ] || { echo "usage: $0 <camus|l2reg|l3reg|netcs>[,...] [cond ...]" >&2; exit 2; }
for ds in $(echo "$DSS" | tr ',' ' '); do
  case $ds in camus|l2reg|l3reg|netcs) ;; *) echo "unknown dataset $ds" >&2; exit 2 ;; esac
done
MAXJ=${MAXJ:-6}
OUT_ROOT=${OUT_ROOT:-$REPO/taxmiss_eval/runs}
case $OUT_ROOT in /*) ;; *) OUT_ROOT=$REPO/$OUT_ROOT ;; esac
PHASES=${PHASES:-}
REPS=${REPS:-}
NOREF=$REPO/subsample_eval/NO_REFERENCE.tsv
[ -e "$NOREF" ] && { echo "$NOREF must not exist (subsample_eval/common.NO_REFERENCE)" >&2; exit 1; }
CONDS_ARG="$*"

level() { [ "$1" = l3reg ] && echo 3 || echo 2; }
conds_of() {   # the named conditions, or every staged one
  if [ -n "$CONDS_ARG" ]; then echo "$CONDS_ARG"; else
    ls -d "$REPO/taxmiss_eval/staged/$1"/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null; fi
}
want_phase() { [ -z "$PHASES" ] && return 0; case ",$PHASES," in *",$1,"*) return 0 ;; esac; return 1; }

LOGS=$OUT_ROOT/logs
mkdir -p "$LOGS"
STATUS=$LOGS/gnn_$(echo "$DSS" | tr ',' '_').status
: > "$STATUS"
stamp() { date '+%F %T'; }
n=0
for ds in $(echo "$DSS" | tr ',' ' '); do
  L=$(level "$ds")
  conds=$(conds_of "$ds")
  [ -n "$conds" ] || { echo "$ds: no staged condition -- run taxmiss_eval/stage.py" >&2; exit 1; }
  for cond in $conds; do
    staged=$REPO/taxmiss_eval/staged/$ds/$cond
    [ -f "$staged/CELLS.tsv" ] || { echo "no $staged/CELLS.tsv -- run taxmiss_eval/stage.py" >&2; exit 1; }
    phases=$(awk -F'\t' 'NR > 1 {print $1}' "$staged/CELLS.tsv" | sort -u)
    nph=$(echo "$phases" | wc -w | tr -d ' ')
    for ph in $phases; do
      want_phase "$ph" || continue
      if [ -n "$REPS" ]; then splits=$REPS
      elif [ "$nph" = 1 ]; then splits="0-9 10-999"
      else splits=all; fi
      for rs in $splits; do
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJ" ]; do sleep 5; done
        n=$((n + 1))
        tag=$ph; [ "$rs" = all ] || tag=${ph}_r${rs}
        log=$LOGS/gnn_${ds}_${cond}_${tag}.log
        ( cd "$REPO/gnn_cpu_v1" && "$PY" -u run_gnn_cpu.py --threads 1 --partial 4 --m-from-trees \
              --models _l2mix_pair/s0 --inputs "$staged" --cells "$staged/CELLS.tsv" \
              --reference "$NOREF" --out "$OUT_ROOT/$ds/$cond" --run-name "gnn_t1_l2mix_pair_relaxedL$L" \
              --decoder relaxed --max-level "$L" --phases "$ph" $([ "$rs" = all ] || echo --reps "$rs") \
              ${DRY:+--dry-run} > "$log" 2>&1
          echo "$? $ds $cond $tag" >> "$STATUS" ) < /dev/null &
        echo "[$(stamp)] started $ds $cond $tag -> $log"
      done
    done
  done
done
[ "$n" -gt 0 ] || { echo "$DSS: no phase selected" >&2; exit 1; }
wait
bad=$(awk '$1 != 0' "$STATUS")
[ -z "$bad" ] || { echo "[$(stamp)] FAILED (exit status, dataset, cond, job):" >&2; echo "$bad" >&2; exit 1; }
[ "$(wc -l < "$STATUS" | tr -d ' ')" = "$n" ] || { echo "only $(wc -l < "$STATUS") of $n processes reported" >&2; exit 1; }
miss=0
if [ -z "$REPS" ] && [ -z "${DRY:-}" ]; then
  for ds in $(echo "$DSS" | tr ',' ' '); do
    L=$(level "$ds")
    for cond in $(conds_of "$ds"); do
      cells=$OUT_ROOT/$ds/$cond/gnn_t1_l2mix_pair_relaxedL$L/cells
      while IFS=$'\t' read -r ph rep arm _rest; do
        [ "$ph" = phase ] && continue
        want_phase "$ph" || continue
        f=$cells/${ph}_rep$(printf %02d "$rep")_${arm}.json
        [ -f "$f" ] || { echo "missing $f" >&2; miss=$((miss + 1)); }
      done < "$REPO/taxmiss_eval/staged/$ds/$cond/CELLS.tsv"
    done
  done
fi
[ "$miss" = 0 ] || { echo "[$(stamp)] $miss cells have no JSON" >&2; exit 1; }
if [ -n "${DRY:-}" ] || [ -n "$REPS" ]; then
  echo "[$(stamp)] $DSS: $n processes done (${DRY:+dry run}${REPS:+--reps $REPS}: completeness not checked) -- logs in $LOGS"
else
  echo "[$(stamp)] $DSS: $n processes done, every cell has its JSON -- logs in $LOGS"
fi
