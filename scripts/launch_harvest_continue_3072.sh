#!/bin/bash
# Continue stationary harvest on the live 3072/384 profile.
#
# Run this ON the Vast RTX 5090 (or a host with the same paths). It refuses
# the 4096/512 CLI sentinels, does not pass --easy, and does not weld.
# Jupyter on 8080 is not touched. Eval keeps guidance_weight=0 via
# --eval-profile speedrun. This is not field harvest; training_ready stays false.
set -euo pipefail

WORLD=3072
STEPS=64
MB=384
if [ "$WORLD" -eq 4096 ] || [ "$MB" -eq 512 ]; then
  echo 'refusing 4096 worlds or 512 minibatch (CLI sentinels rewrite the job)' >&2
  exit 2
fi

SRC=${ORCHARDBENCH_SOURCE:-/workspace/sources/thekenyos-ikdemo}
PY=${TRAIN_PYTHON:-/workspace/training/envs/flex-gpu/bin/python}
CKPT=${HARVEST_INIT:-/workspace/training/runs/frozen/harvest5-checkpoint-0008.pt}
SCENE=${HARVEST_SCENE:-/workspace/training/scenes/fast-gripper-5}
GAIT=${HARVEST_GAIT:-/workspace/training/gait/g1-cuda-fp32.pt}
OUT=${HARVEST_OUT:-/workspace/training/runs/fast-5090-curriculum-s03-3072w-harvest15}

if [ ! -x "$PY" ]; then
  echo "missing trainer python: $PY" >&2
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  echo "missing harvest5-0008 init: $CKPT" >&2
  exit 1
fi
# Use harvest5-0008, not harvest14 (eval grasp 0.93→0).
case "$CKPT" in
  *harvest14*) echo 'refusing harvest14 init; use harvest5-checkpoint-0008.pt' >&2; exit 2 ;;
esac

export PYTHONPATH=$SRC
export ORCHARDBENCH_SOURCE=$SRC
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$SRC"
exec "$PY" -B scripts/train_fast.py \
  --scene "$SCENE" \
  --gait-checkpoint "$GAIT" \
  --output "$OUT" \
  --initialize-from "$CKPT" \
  --stage stationary_harvest \
  --worlds "$WORLD" --steps "$STEPS" --minibatch-worlds "$MB" \
  --eval-profile speedrun --ik-harvest --demo-updates 0 \
  --video-every 0 --seed 7 \
  --monitor-hub /workspace/training/monitor-live
