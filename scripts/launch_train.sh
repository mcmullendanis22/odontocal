#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.yaml}"
DATA_ROOT="${DATA_ROOT:-./data}"
NPROC="${NPROC:-1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

for SEED in ${SEEDS}; do
  torchrun --nproc-per-node="${NPROC}" -m odontocal.entry.train \
    --experiment "${EXPERIMENT}" \
    --data "${DATA_ROOT}" \
    --seed "${SEED}" \
    --ckpt-dir "checkpoints/seed_${SEED}"
done
