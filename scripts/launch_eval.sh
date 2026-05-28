#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-configs/experiment/main.yaml}"
PREDICTIONS="${PREDICTIONS:-artifacts/predictions.jsonl}"
GROUND_TRUTH="${GROUND_TRUTH:-artifacts/ground_truth.jsonl}"
OUT="${OUT:-artifacts/results.json}"

python -m odontocal.entry.evaluate \
  --experiment "${EXPERIMENT}" \
  --predictions "${PREDICTIONS}" \
  --ground-truth "${GROUND_TRUTH}" \
  --out "${OUT}"
