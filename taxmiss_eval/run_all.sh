#!/usr/bin/env bash
# The whole missing-taxa benchmark, in order, stopping at the first failure -- see taxmiss_eval/README.md.
#
#   caffeinate -i bash taxmiss_eval/run_all.sh                     # every step, every staged condition
#   STEPS="csv rest trainfit nb" bash taxmiss_eval/run_all.sh      # from the per-network table on
#   TM_CONDS=p0.80_c12 caffeinate -i bash taxmiss_eval/run_all.sh  # only a newly staged condition
#
# Steps (STEPS, default all of them):
#   gate   stage.py --check and run_res.py --check                                  seconds
#   gnn    run_gnn.sh <families>              the GNN, --partial 4                  ~10 min
#   res    run_res.py --jobs JOBS             BROOQS and NetCS on the raw TOBs      ~30 min at 12
#   score  run_derived.sh score tm-<group>    the scored cells (z, decodes, parity) ~10 min
#   csv    make_csv.py                        the per-network table, luay by the jar  minutes
#   rest   run_derived.sh rest tm-<group>     features -> credible -> make_csv -> bench   ~5 min each
#   trainfit  run_trainfit.sh derived tm-<group>  RIDDLE: the train-fit beta on rest's feats,
#             credible --calib-pool train -> make_csv -> bench into results/trainfit/derived   minutes
#   nb     notebooks/plotting_taxmiss.ipynb, executed in place (it reads the train fit)  seconds
# Every step resumes.  run_res.py exiting 3 (a resolver failed or timed out on some cell -- an
# outcome, recorded as its row's status) is reported and the chain goes on; any other non-zero exit
# stops it.  Logs: taxmiss_eval/runs/logs/all_<step>.log.
#
# Environment:
#   FAMS      families (comma list); default every one with a staged condition
#   TM_CONDS  only these conditions (comma list) in every step; default every staged one
#   JOBS [12] run_res.py workers         TIMEOUT [7200] each resolver's timeout, seconds
#   MAXJ      run_gnn.sh / run_derived.sh score concurrency (their defaults: 6)
# bash 3.2 (macOS /bin/bash).
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=$REPO/gnn_cpu_v1/.venv/bin/python
STEPS=${STEPS:-gate gnn res score csv rest trainfit nb}
JOBS=${JOBS:-12}
TIMEOUT=${TIMEOUT:-7200}
TM_CONDS=${TM_CONDS:-}
export TM_CONDS
LOGS=$REPO/taxmiss_eval/runs/logs
mkdir -p "$LOGS"
say() { echo "[run_all] $(date '+%F %T') $*"; }
die() { say "FAILED: $*" >&2; exit 1; }

if [ -z "${FAMS:-}" ]; then
  FAMS=$(for f in camus l2reg l3reg netcs; do
           ls "$REPO/taxmiss_eval/staged/$f"/*/CELLS.tsv > /dev/null 2>&1 && printf '%s,' "$f"; done)
  FAMS=${FAMS%,}
fi
[ -n "$FAMS" ] || die "nothing staged -- run taxmiss_eval/stage.py"
CONDS_ARGS=$(echo "$TM_CONDS" | tr ',' ' ')
TMGROUPS=""
case ",$FAMS," in *",camus,"*) TMGROUPS="$TMGROUPS tm-camus" ;; esac
case ",$FAMS," in *",l2reg,"*|*",l3reg,"*) TMGROUPS="$TMGROUPS tm-reg" ;; esac
case ",$FAMS," in *",netcs,"*) TMGROUPS="$TMGROUPS tm-netcs" ;; esac
say "families $FAMS | groups$TMGROUPS | conditions ${TM_CONDS:-every staged one} | steps: $STEPS"

for step in $STEPS; do
  log=$LOGS/all_$step.log
  say "$step -> $log"
  case $step in
    gate)
      { $PY taxmiss_eval/stage.py --check --families "$FAMS" --conds "${TM_CONDS:-all}" \
          && $PY taxmiss_eval/run_res.py --check --dataset "$FAMS" --conds "${TM_CONDS:-all}"; } > "$log" 2>&1 \
        || die "gates, see $log"
      tail -1 "$log" ;;
    gnn)
      # shellcheck disable=SC2086 -- the condition list, word-split on purpose
      bash taxmiss_eval/run_gnn.sh "$FAMS" $CONDS_ARGS > "$log" 2>&1 || die "run_gnn.sh, see $log"
      tail -1 "$log" ;;
    res)
      $PY -u taxmiss_eval/run_res.py --dataset "$FAMS" --conds "${TM_CONDS:-all}" --jobs "$JOBS" \
          --timeout-brooqs "$TIMEOUT" --timeout-netcs "$TIMEOUT" > "$log" 2>&1
      rc=$?
      if [ $rc = 3 ]; then say "  run_res.py: some tasks did not finish ok -- recorded as outcomes (see $log)"
      elif [ $rc != 0 ]; then die "run_res.py exit $rc, see $log"; fi
      grep "did not finish ok\|break an invariant" "$log" | tail -2 ;;
    score)
      for g in $TMGROUPS; do
        bash nni_ensemble/run_derived.sh score "$g" >> "$log" 2>&1 || die "score $g, see $log"
      done
      tail -1 "$log" ;;
    csv)
      $PY taxmiss_eval/make_csv.py --dataset "$FAMS" --jobs 4 > "$log" 2>&1 || die "make_csv.py, see $log"
      grep "===" "$log" ;;
    rest)
      for g in $TMGROUPS; do
        bash nni_ensemble/run_derived.sh rest "$g" >> "$log" 2>&1 || die "rest $g, see $log"
      done
      grep "=== gates" "$log" ;;
    trainfit)
      for g in $TMGROUPS; do
        bash nni_ensemble/run_trainfit.sh derived "$g" >> "$log" 2>&1 || die "trainfit $g, see $log"
      done
      grep "bench tm-" "$log" ;;
    nb)
      ( cd notebooks && OMP_NUM_THREADS=1 "$HOME/miniconda3/envs/phylo-env/bin/jupyter" nbconvert --to notebook \
          --execute --inplace plotting_taxmiss.ipynb ) > "$log" 2>&1 || die "the notebook, see $log"
      tail -1 "$log" ;;
    *) die "unknown step $step (gate gnn res score csv rest trainfit nb)" ;;
  esac
done
say "done: $STEPS"
