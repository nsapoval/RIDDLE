#!/usr/bin/env bash
# W10 -- train every arm x every seed, sequentially on ONE GPU.
#
# **Stopping this means killing THIS pid first, not the running python.**  It is a `for` loop, so
# killing the child just makes the loop start the next arm -- ``CLAUDE.md`` records ~4 GPU-hours
# lost to exactly that on ``queue_bench.sh``.  Kill this pid, then its descendants, then check
# ``ps -eo pid,ppid,args`` for a w10 root reparented to pid 1.
set -u
cd "$(dirname "$0")/../.."
# `${VAR-0}`, NOT `${VAR:-0}`: the colon form treats an explicit empty string as unset, so
# `CUDA_VISIBLE_DEVICES="" bash scripts/ops/queue_gnn_eval.sh` -- the documented way to ask for a
# CPU-only run -- silently became GPU 0, the card another user's job is on.  Without the colon an
# empty value is preserved and an unset one still defaults to 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
ARCHS="${ARCHS:-mlp gcn gin segment}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
# `TAG` names the training set, and it is the only thing that keeps a retrained arm from
# overwriting a landed checkpoint.  It must NOT contain `_s<digits>`: `make_gnn_report.ARM_RE`
# splits the arm from the seed on that pattern (A92's glob hazard).
TAG="${TAG:-}"
PHASES="${PHASES:-b25,b50}"
# The gene-tree arms, CROSSED with PHASES.  "true" is the simulated trees; "true,est_bp1000" is
# 2x the data, every cell under both.  It is NOT part of TAG and must not be inferred from it --
# w10_gnn.py records it in config.json and w10_gnn_eval.py copies it onto every row.
ARMS="${ARMS:-true}"
SPLIT_TRAIN="${SPLIT_TRAIN:-70}"
SPLIT_VAL="${SPLIT_VAL:-80}"
POOL="${POOL:-192}"
# Above ~400 cells a resident ClusterEnv costs more than the instance it came from, and an OOM in
# a pool here HANGS rather than fails (A89).  A pointwise BCE never touches a train cell's env.
LAZY="${LAZY:-}"
# The validation decoder.  `level1` is the formal (L)/(N3O) one and every landed run used it;
# `auto` sends a validation cell whose truth is above level 1 through `opc.rl.relaxed` instead and
# leaves every level-1 cell on the formal path, bit for bit.  It is here rather than hard-coded
# because a mixture that contains a level-2 phase and one that does not are the same command
# otherwise, and selecting a checkpoint on a decoder that cannot reach the truth is not a
# validation curve, it is a measurement of the mask.
VAL_DECODER="${VAL_DECODER:-level1}"
VAL_MAX_LEVEL="${VAL_MAX_LEVEL:-2}"
# The decode-derived loss term -- W10 v2's step/set/band, or A108's `ha`.  Each flag is passed ONLY
# when its variable is set, so every landed command line is unchanged.
DECODE_LOSS="${DECODE_LOSS:-}"
DECODE_WEIGHT="${DECODE_WEIGHT:-}"
DECODE_WARMUP="${DECODE_WARMUP:-}"
HA_MAX_LEVEL="${HA_MAX_LEVEL:-}"
HA_JOBS="${HA_JOBS:-}"
for a in $ARCHS; do
  for s in $SEEDS; do
    if [ -f "results/rl/rl_run_gnn_${a}_s${s}${TAG}/best.pt" ]; then
      echo "[queue] skip ${a} s${s}${TAG} (already trained)"; continue
    fi
    echo "[queue] === ${a} seed ${s}${TAG} ==="
    .venv/bin/python -u scripts/w10_gnn.py --arch "$a" --seed "$s" --epochs "$EPOCHS" \
        --phases "$PHASES" --arms "$ARMS" \
        --split-train "$SPLIT_TRAIN" --split-val "$SPLIT_VAL" \
        --pool-size "$POOL" ${LAZY:+--lazy-env} ${TAG:+--tag "$TAG"} \
        --val-decoder "$VAL_DECODER" --val-max-level "$VAL_MAX_LEVEL" \
        ${DECODE_LOSS:+--decode-loss "$DECODE_LOSS"} \
        ${DECODE_WEIGHT:+--decode-weight "$DECODE_WEIGHT"} \
        ${DECODE_WARMUP:+--decode-warmup "$DECODE_WARMUP"} \
        ${HA_MAX_LEVEL:+--ha-max-level "$HA_MAX_LEVEL"} \
        ${HA_JOBS:+--ha-jobs "$HA_JOBS"} || \
      echo "[queue] FAILED ${a} s${s}${TAG}"
  done
done
echo "[queue] done"
