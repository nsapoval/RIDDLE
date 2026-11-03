#!/usr/bin/env bash
# RIDDLE (GNN MAP [z+tc], train-fit beta, relaxed L1) at k = 0..10, and ASTRAL-IV, on
# sptree_n200_gt10k_v1 at 1k / 5k / 10k gene trees (sptree_eval/README.md).
#
#   caffeinate -i bash sptree_eval/run_sptree.sh score     the GNN: cluster_anatomy/score_ksweep.py, one job per rep
#   caffeinate -i bash sptree_eval/run_sptree.sh probe     probe_blocks + the tail gate + kernel routing, all 33
#   caffeinate -i bash sptree_eval/run_sptree.sh rest      features (block, or dense past 24 atoms) -> credible
#                                                          --calib-pool train -> cred gates
#   caffeinate -i bash sptree_eval/run_sptree.sh astral    ASTRAL-IV, 150 runs
#   bash sptree_eval/run_sptree.sh collate                 the tables and gates C1-C7 (LUAY=1: luay too)
#   bash sptree_eval/run_sptree.sh status
#
# Every step stops the script on failure, and a step with nothing to do is a failure.  Every step is
# resumable: rerun the same command after a stop and only the missing cells run.
#
# Environment (defaults in brackets):
#   MAXJ [5]   concurrent score_ksweep processes (one per rep; a worker peaks at ~2-5 GB)
#   JOBS [5]   probe / features / credible / ASTRAL workers
#   REPS       a rep filter, e.g. "16,17,21" (smokes); every count check then asks for >= 1 cell
#   DRY=1      score: print the job order and the cells still to score, run nothing
#   FORCE_TAIL=1  run rest past the tail gate
#   LUAY=1     collate also computes luay with the PhyloNet jar
#   RUNS [sptree_eval/runs/ne]  ANAT [sptree_eval/runs/anat]  redirect, for smokes
# bash 3.2 (macOS /bin/bash): no wait -n, no associative arrays.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=$REPO/gnn_cpu_v1/.venv/bin/python
MAXJ=${MAXJ:-5}
JOBS=${JOBS:-5}
REPS=${REPS:-}
abs() { case "$1" in /*) echo "$1" ;; *) echo "$REPO/$1" ;; esac; }
RUNS=$(abs "${RUNS:-sptree_eval/runs/ne}")
ANAT=$(abs "${ANAT:-sptree_eval/runs/anat}")
export NE_ANAT_RUNS="$ANAT"
LOGS=$REPO/sptree_eval/runs/logs
mkdir -p "$LOGS" "$RUNS" "$ANAT"
CALIB_SRC=$REPO/nni_ensemble/runs_trainfit/calib/train__z6.json
CALIB_SHA=f47199e9746d8468f7c62ee145544f1525de176c94dbe7e4558adadf5c03a610
H=("$REPO/nni_ensemble/derived.py" --runs-root "$RUNS")
PARTIAL=()
RF=()
[ -n "$REPS" ] && PARTIAL=(--partial) && RF=(--reps "$REPS")

die() { echo "[run_sptree] FAILED: $*" >&2; exit 1; }
say() { echo "[run_sptree] $(date +%H:%M:%S) $*"; }

[ -f "$REPO/sptree_eval/CELLS.tsv" ] || die "not staged: run sptree_eval/stage.py first"
DSS=$($PY -c 'import sys; sys.path.insert(0, "nni_ensemble"); import ne_common as NE; print(" ".join(NE.SPTREE))') \
  || die "cannot list the sptree datasets"
[ "$(echo $DSS | wc -w | tr -d ' ')" = 33 ] || die "expected 33 sptree datasets, got: $DSS"

guard() {
  mkdir -p "$RUNS/calib"
  [ -e "$RUNS/calib/train__z6.json" ] || ln -s "$CALIB_SRC" "$RUNS/calib/train__z6.json"
  got=$(shasum -a 256 "$RUNS/calib/train__z6.json" | cut -d' ' -f1)
  [ "$got" = "$CALIB_SHA" ] || die "$RUNS/calib/train__z6.json is not the train-fit beta (sha $got)"
  for p in sim netcs; do
    [ -e "$RUNS/calib/${p}__z6.json" ] && die "$RUNS/calib/${p}__z6.json must not exist here"
  done
  return 0
}

# the tail gate: every cell inside nni_ensemble/derived.py's TAIL (the dense kernel's measured limits);
# a cell past BLOCK_MAX_M atoms is routed to the dense kernel by rest, not refused
tail_gate() {
  bad=""
  for ds in $DSS; do
    "$PY" sptree_eval/check_sptree.py kernels "$ds" --runs-root "$RUNS" --reps "$REPS" \
      > "$LOGS/kernels_$ds.txt" 2> "$LOGS/tail_$ds.log" || bad="$bad $ds"
  done
  n_dense=$(grep -h '^dense ' "$LOGS"/kernels_sp*.txt | sed 's/^dense //' | tr ',' '\n' | grep -c .)
  say "tail: $(echo $DSS | wc -w | tr -d ' ') datasets, $n_dense cells routed to the dense kernel"
  if [ -n "$bad" ]; then
    cat "$LOGS"/tail_*.log | grep 'past the published tail' | head -5
    [ -n "${FORCE_TAIL:-}" ] || die "past the tail on:$bad (see $LOGS/tail_<ds>.log; smoke those cells alone, or FORCE_TAIL=1)"
    say "FORCE_TAIL set: running past the tail on:$bad"
  fi
  return 0
}

# features on the cells of one kernel: $1 dataset, $2 kernel, $3 comma rep list (nothing to do when empty)
features_on() {
  [ -n "$3" ] || return 0
  "$PY" -u nni_ensemble/features.py --dataset "$1" --thr 6 --check-tc --jobs "$JOBS" --runs-root "$RUNS" \
    --tc-kernel "$2" --reps "$3" > "$LOGS/features_$1_$2.log" 2>&1 \
    || die "$1: features ($2 kernel), see $LOGS/features_$1_$2.log"
}

case "${1:-}" in
  score)
    # one score_ksweep job per rep, the cells with the noisiest gene trees (largest C_tau) first
    ORDER=$("$PY" - "$REPS" <<'EOF'
import json, sys
sys.path.insert(0, "sptree_eval")
import sp_common as SP
want = SP.parse_reps(sys.argv[1])
cells = [c for c in SP.stage()["cells"] if want is None or c["rep"] in want]
print(" ".join(str(c["rep"]) for c in sorted(cells, key=lambda c: -c["rf_norm_mean"])))
EOF
)
    [ -n "$ORDER" ] || die "no reps selected"
    n=$(echo $ORDER | wc -w | tr -d ' ')
    if [ -n "${DRY:-}" ]; then
      say "dry run, job order (reps): $ORDER"
      exec "$PY" cluster_anatomy/score_ksweep.py --reps "$(echo $ORDER | tr ' ' ',')" --out "$ANAT" --dry-run
    fi
    STATUS="$LOGS/score.status"
    : > "$STATUS"
    say "score: $n reps x 3 sizes x 11 k, at most $MAXJ jobs at once; logs in $LOGS/score_repRR.log"
    i=0
    for rep in $ORDER; do
      i=$((i + 1))
      while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJ" ]; do sleep 5; done
      log=$LOGS/score_rep$(printf %02d "$rep").log
      ( "$PY" -u cluster_anatomy/score_ksweep.py --reps "$rep" --out "$ANAT" > "$log" 2>&1
        echo "$? $rep" >> "$STATUS" ) < /dev/null &
      say "  started $i/$n: rep $rep"
    done
    wait
    bad=$(awk '$1 != 0' "$STATUS")
    [ -z "$bad" ] || die "score jobs exited non-zero (exit rep):
$bad"
    [ "$(wc -l < "$STATUS" | tr -d ' ')" = "$n" ] || die "only $(wc -l < "$STATUS") of $n jobs reported"
    for ds in $DSS; do
      "$PY" cluster_anatomy/score_cells.py --dataset "$ds" --out "$ANAT" --summary-only > "$LOGS/summary_$ds.log" 2>&1 \
        || die "$ds: summary failed, see $LOGS/summary_$ds.log"
      "$PY" "${H[@]}" check scored "$ds" ${PARTIAL[@]+"${PARTIAL[@]}"} || die "$ds: not every cell is scored"
    done
    "$PY" sptree_eval/check_sptree.py sweep $DSS || die "gate G-K"
    say "score: done"
    ;;
  probe)
    for ds in $DSS; do
      "$PY" "${H[@]}" check scored "$ds" ${PARTIAL[@]+"${PARTIAL[@]}"} || die "$ds: run '$0 score' first"
      "$PY" -u nni_ensemble/probe_blocks.py --dataset "$ds" --jobs "$JOBS" --out-dir "$RUNS/probe" \
        > "$LOGS/probe_$ds.log" 2>&1 || die "$ds: probe_blocks, see $LOGS/probe_$ds.log"
    done
    tail_gate
    say "probe: done, every cell inside the tail"
    ;;
  rest)
    guard
    tail_gate
    for ds in $DSS; do
      say "rest $ds"
      "$PY" "${H[@]}" check scored "$ds" ${PARTIAL[@]+"${PARTIAL[@]}"} || die "$ds: run '$0 score' first"
      [ -f "$LOGS/kernels_$ds.txt" ] || die "$ds: no kernel routing -- run '$0 probe'"
      features_on "$ds" block "$(sed -n 's/^block //p' "$LOGS/kernels_$ds.txt")"
      features_on "$ds" dense "$(sed -n 's/^dense //p' "$LOGS/kernels_$ds.txt")"
      "$PY" "${H[@]}" check feat "$ds" ${PARTIAL[@]+"${PARTIAL[@]}"} || die "$ds: features incomplete"
      "$PY" -u nni_ensemble/credible.py --dataset "$ds" --calib-pool train --jobs "$JOBS" --runs-root "$RUNS" \
        ${RF[@]+"${RF[@]}"} > "$LOGS/credible_$ds.log" 2>&1 || die "$ds: credible, see $LOGS/credible_$ds.log"
      "$PY" "${H[@]}" check cred "$ds" ${PARTIAL[@]+"${PARTIAL[@]}"} || die "$ds: credible incomplete"
      "$PY" sptree_eval/check_sptree.py cred "$ds" --runs-root "$RUNS" > "$LOGS/credgate_$ds.log" 2>&1 \
        || die "$ds: a cred is not a clean train-fit transfer, see $LOGS/credgate_$ds.log"
    done
    say "rest: done"
    ;;
  astral)
    "$PY" -u sptree_eval/run_astral.py --jobs "$JOBS" ${RF[@]+"${RF[@]}"} > "$LOGS/astral.log" 2>&1
    rc=$?
    tail -1 "$LOGS/astral.log"
    [ $rc = 0 ] || die "astral exit $rc, see $LOGS/astral.log"
    ;;
  collate)
    LU=()
    [ -n "${LUAY:-}" ] && LU=(--luay)
    "$PY" -u sptree_eval/collate.py --runs-root "$RUNS" ${PARTIAL[@]+"${PARTIAL[@]}"} ${LU[@]+"${LU[@]}"} \
      || die "collate gates, see sptree_eval/results/sptree_gates.json"
    ;;
  status)
    exec "$PY" sptree_eval/check_sptree.py status all --runs-root "$RUNS"
    ;;
  *)
    echo "usage: $0 {score|probe|rest|astral|collate|status}" >&2
    exit 2
    ;;
esac
