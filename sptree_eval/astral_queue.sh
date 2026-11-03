#!/usr/bin/env bash
# ASTRAL-IV's 150 runs as a queue staged by gene-tree count -- every 1k unit, then every 5k, then every
# 10k -- one sptree_eval/run_astral.py unit per (size, rep), at a concurrency that keeps every CPU-bound
# job on its own performance core (sptree_eval/README.md).
#
#   caffeinate -i bash sptree_eval/astral_queue.sh
#   touch sptree_eval/runs/logs/astral_queue/STOP     drain: launch nothing more, let the running ones finish
#
# Slots: CAP CPU-bound jobs on the box (the M4 Max has 12 P-cores; one is left to the system), less
# RIDDLE_SLOTS while the RIDDLE chain (run_sptree.sh / full_riddle_chain.sh) is alive.  Every astral4
# process on the machine counts, the smokes' included.  A size's concurrency is further capped at
# MEM_GB / its measured peak RSS (1k 0.6 GB and 5k 2.8 GB from the smokes; 10k from its own smoke).
# 10k is gated: it waits until the 10k smoke (reps 16 and 17) has finished, and runs only if both passed.
# Units already done (run.json ok) are skipped, so rerunning this resumes.  Exits 1 if any unit failed.
#
# Environment: CAP [11]  RIDDLE_SLOTS [6]  MEM_GB [40]  SIZES [1k,5k,10k]  REPS (a rep filter)
# bash 3.2 (macOS /bin/bash): no wait -n, no associative arrays.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=$REPO/gnn_cpu_v1/.venv/bin/python
CAP=${CAP:-11}
RIDDLE_SLOTS=${RIDDLE_SLOTS:-6}
MEM_GB=${MEM_GB:-40}
SIZES=${SIZES:-1k,5k,10k}
REPS=${REPS:-}
ASTRAL_RUNS=$REPO/sptree_eval/runs/astral
QLOGS=$REPO/sptree_eval/runs/logs/astral_queue
STATUS=$QLOGS/status.tsv
mkdir -p "$QLOGS"
rm -f "$QLOGS/STOP"

say() { echo "[queue] $(date +%H:%M:%S) $*"; }
# anchored on the binary's path: a shell whose command line merely mentions "astral4 -i" (a watcher) is no run
n_astral() { pgrep -f '^[^ ]*/astral4 -i' | grep -c .; }
riddle_alive() { pgrep -f 'run_sptree.sh|full_riddle_chain.sh' > /dev/null; }
mem_free() { memory_pressure | awk '/free percentage/ {sub("%", "", $NF); print $NF}'; }
# peak RSS of a size's finished runs, in GB (rounded up; 0 when none has finished)
peak_gb() {
  "$PY" - "$ASTRAL_RUNS/$1" <<'EOF'
import json, math, sys
from pathlib import Path
r = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob("*/run.json")]
r = [x["rss_mb"] for x in r if x.get("ok")]
print(math.ceil(max(r) / 1024) if r else 0)
EOF
}

# the units still to run, size by size, the noisiest gene trees first within a size
UNITS=$("$PY" - "$SIZES" "$REPS" <<'EOF'
import json, sys
sys.path.insert(0, "sptree_eval")
import sp_common as SP
sizes, want = [s for s in sys.argv[1].split(",") if s], SP.parse_reps(sys.argv[2])
cells = sorted((c for c in SP.stage()["cells"] if want is None or c["rep"] in want), key=lambda c: -c["rf_norm_mean"])
for s in sizes:
    for c in cells:
        p = SP.RUNS / "astral" / s / SP.stem(c["rep"]) / "run.json"
        if not (p.exists() and json.loads(p.read_text()).get("ok")):
            print(f"{s}:{c['rep']}")
EOF
) || { say "FAILED: cannot list the units"; exit 1; }
n=$(echo $UNITS | wc -w | tr -d ' ')
[ "$n" -gt 0 ] || { say "FAILED: nothing to do -- every selected unit has an ok run.json"; exit 1; }
: > "$STATUS"
say "$n units ($(echo $UNITS | tr ' ' '\n' | cut -d: -f1 | uniq -c | awk '{printf "%s %s, ", $1, $2}')) | CAP $CAP, RIDDLE_SLOTS $RIDDLE_SLOTS, MEM_GB $MEM_GB | logs $QLOGS"

gate_10k() {
  while :; do
    s=$("$PY" - "$ASTRAL_RUNS/10k" <<'EOF'
import json, sys
from pathlib import Path
got = {}
for rep in (16, 17):
    p = Path(sys.argv[1]) / f"sptree_n200_rep{rep:02d}_est_bp1000" / "run.json"
    got[rep] = json.loads(p.read_text()).get("ok") if p.exists() else None
print("ok" if all(v is True for v in got.values()) else "wait" if None in got.values() else f"bad {got}")
EOF
)
    # a timed-out first attempt leaves an ok=false run.json while its re-run is still going
    if [ "$s" != ok ] && pgrep -f '^[^ ]*/astral4 -i .*astral/10k/' > /dev/null; then s=wait; fi
    case "$s" in
      ok) return 0 ;;
      wait) sleep 60 ;;
      *) say "10k gate FAILED: the 10k smoke did not pass ($s) -- no 10k unit launched"; return 1 ;;
    esac
  done
}

i=0
cur=""
cap_size=$CAP
failed=0
for u in $UNITS; do
  size=${u%%:*}
  rep=${u##*:}
  [ -e "$QLOGS/STOP" ] && { say "STOP file found: launching nothing more"; break; }
  if [ "$size" != "$cur" ]; then
    if [ "$size" = 10k ]; then
      say "10k: waiting for the 10k smoke to finish and pass"
      gate_10k || { failed=$((failed + 1)); break; }
    fi
    pk=$(peak_gb "$size")
    [ "$pk" -gt 0 ] || { say "FAILED: no finished $size run to size its memory by"; failed=$((failed + 1)); break; }
    cap_size=$((MEM_GB / pk))
    say "stage $size: peak RSS ${pk} GB -> at most $cap_size concurrent $size runs by memory"
    cur=$size
  fi
  while :; do
    [ -e "$QLOGS/STOP" ] && break
    slots=$CAP
    riddle_alive && slots=$((CAP - RIDDLE_SLOTS))
    mine=$(jobs -rp | grep -c .)
    busy=$(n_astral)
    [ "$mine" -gt "$busy" ] && busy=$mine
    [ "$busy" -lt "$slots" ] && [ "$mine" -lt "$cap_size" ] && [ "$(mem_free)" -ge 20 ] && break
    sleep 20
  done
  [ -e "$QLOGS/STOP" ] && { say "STOP file found: launching nothing more"; break; }
  i=$((i + 1))
  before=$(n_astral)
  log=$QLOGS/${size}_rep$(printf %02d "$rep").log
  ( "$PY" -u sptree_eval/run_astral.py --sizes "$size" --reps "$rep" --jobs 1 > "$log" 2>&1
    echo "$? $size $rep $(date +%H:%M:%S)" >> "$STATUS" ) < /dev/null &
  say "  launched $i/$n: $size rep $rep (astral4 running: $before, slots $slots, mem free $(mem_free)%)"
  # let the new astral4 appear before the next slot count
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    [ "$(n_astral)" -gt "$before" ] && break
    sleep 2
  done
  bad=$(awk '$1 != 0' "$STATUS" | grep -c .)
  if [ "$bad" -ge 3 ]; then
    say "3 units failed -- launching nothing more:"
    awk '$1 != 0' "$STATUS"
    break
  fi
done
say "every unit launched or the queue stopped; waiting for $(jobs -rp | grep -c .) running"
wait
ok=$(awk '$1 == 0' "$STATUS" | grep -c .)
bad=$(awk '$1 != 0' "$STATUS" | grep -c .)
say "done: $ok ok, $bad failed, of $n queued ($i launched)"
[ "$bad" = 0 ] && [ "$failed" = 0 ] || { awk '$1 != 0' "$STATUS"; exit 1; }
exit 0
