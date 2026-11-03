#!/usr/bin/env bash
# The level-2 data ladder -- four SegmentConv conditions, STRIPED BY SEED.
#
# | tag        | phases                                  | train cells x2 arms | status        |
# |------------|-----------------------------------------|--------------------:|---------------|
# | `_mixL`    | x25,x50                                 |               2,800 | LANDED, reuse |
# | `_mix2L`   | x25,x50,y25,y50                         |               5,600 | LANDED, reuse |
# | `_l2mix`   | + l2t25,l2t50                           |               8,400 | this script   |
# | `_l2mix75` | + x75                                   |               9,000 | this script   |
#
# The first two are `scripts/ops/queue_gnn_a104.sh`'s reference and its `_mix2L` condition, already
# trained at seeds 0/1/2 with exactly these phases, arms, epochs and split.  Retraining them under
# new names would buy nothing but three days of GPU, so this script runs the two new conditions
# only and the ladder is read across all four.
#
# **Seed outer, condition inner** -- the one structural difference from `queue_gnn_a104.sh`, which
# nests condition -> arch -> seed.  Here a first pass with SEEDS="0" delivers seed 0 for *both* new
# conditions before any seed 1 starts, so the ladder can be looked at end to end a day in rather
# than a week in.
#
# **`--val-decoder auto` is not cosmetic.**  The formal decoder masks by (L)/(N3O), which
# characterise *level-1* clustering systems: on the 20 exactly-level-2 networks of
# `l2reg_n200_v1`, (L) fails on the true cluster set of 20 of 20 and the closure breaks the network
# on 10 of 20, so cluster F1 = 1 is unreachable on a level-2 cell by construction.  `auto` sends
# those validation cells through `opc.rl.relaxed` at `--val-max-level 2` and leaves every level-1
# cell on the formal path bit for bit -- measured identical on x25, so `_l2mix`'s x/y/x75
# validation numbers stay comparable with `_mixL` and `_mix2L`.  Read `val_f1_level1` across the
# ladder, **not** `val_f1`: the latter is a mean over validation sets of different composition.
#
# **Stopping this means killing THIS pid first, not the running python.**  It is a `for` loop, so
# killing the child just makes the loop start the next condition -- CLAUDE.md records ~4 GPU-hours
# lost to exactly that.  Kill this pid, then its descendants, then check `ps -eo pid,ppid,args` for
# a w10 root reparented to pid 1.
#
#   CUDA_VISIBLE_DEVICES=0 SEEDS="0"   bash scripts/ops/queue_gnn_l2.sh   # ~24 GPU-h
#   CUDA_VISIBLE_DEVICES=0 SEEDS="1 2" bash scripts/ops/queue_gnn_l2.sh   # ~48 GPU-h
set -u
cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-0 1 2}"
ARCHS="${ARCHS:-segment}"
ARMS="${ARMS:-true,est_bp1000}"
EPOCHS="${EPOCHS:-50}"
# `POOL=500`, not the x-grids' 700.  A89: an OOM in the instance pool HANGS at 0% CPU with nothing
# in the log.  700 sat at 27 GB over 1,400-cell entries; these runs hold up to 9,000.  LAZY=1 is
# not optional either -- and under `--val-decoder auto` a level-2 validation cell never builds a
# ClusterEnv at all, because the relaxed decoder does not take one.
POOL="${POOL:-500}"
CONDITIONS="${CONDITIONS:-_l2mix _l2mix75}"

# A108 -- the `ha` conditions: the same data as their base tag, plus `--decode-loss ha` (balanced
# hybrid-ancestry pressure on a relaxed-L2 decode of every training cell).  Nothing else changes.
# Pivoting between the two data mixtures is `CONDITIONS=_l2mixha` vs `CONDITIONS=_l2mix75ha`.
#
#   CUDA_VISIBLE_DEVICES=0 SEEDS="0 1" CONDITIONS=_l2mixha bash scripts/ops/queue_gnn_l2.sh
#
# HA_WEIGHT (default 1.0), HA_WARMUP (5), HA_JOBS (8) and HA_MAX_LEVEL (2) override the term.
for s in $SEEDS; do
  for tag in $CONDITIONS; do
    loss=""
    case "$tag" in
      _l2mix)     ph="x25,x50,y25,y50,l2t25,l2t50" ;;
      _l2mix75)   ph="x25,x50,y25,y50,l2t25,l2t50,x75" ;;
      _l2mixha)   ph="x25,x50,y25,y50,l2t25,l2t50"; loss=ha ;;
      _l2mix75ha) ph="x25,x50,y25,y50,l2t25,l2t50,x75"; loss=ha ;;
      *) echo "[l2] unknown condition $tag"; exit 2 ;;
    esac
    echo "[l2] === seed $s  $tag  phases=$ph  epochs=$EPOCHS  arms=$ARMS${loss:+  loss=$loss} ==="
    env TAG="$tag" PHASES="$ph" EPOCHS="$EPOCHS" ARMS="$ARMS" \
        SPLIT_TRAIN=700 SPLIT_VAL=800 POOL="$POOL" LAZY=1 \
        VAL_DECODER=auto VAL_MAX_LEVEL=2 \
        DECODE_LOSS="$loss" DECODE_WEIGHT="${loss:+${HA_WEIGHT:-1.0}}" \
        DECODE_WARMUP="${loss:+${HA_WARMUP:-5}}" HA_JOBS="${loss:+${HA_JOBS:-8}}" \
        HA_MAX_LEVEL="${loss:+${HA_MAX_LEVEL:-2}}" \
        ARCHS="$ARCHS" SEEDS="$s" \
        bash scripts/ops/queue_gnn_train.sh || echo "[l2] FAILED $tag s$s"
  done
done
echo "[l2] done"
