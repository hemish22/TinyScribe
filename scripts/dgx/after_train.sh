#!/usr/bin/env bash
# Wait for a training job (tjob name train<size>) to finish, then merge its LoRA adapter, convert to
# GGUF and score the test split.
#   GPUS=7 bash scripts/dgx/tjob.sh after2b bash scripts/dgx/after_train.sh 2b "2B fine-tuned, BF16 (DGX)" 4
#   GPUS=4,5,6 bash scripts/dgx/tjob.sh after9b bash scripts/dgx/after_train.sh 9b "9B fine-tuned, BF16 (DGX)" 3
set -euo pipefail
cd "$(dirname "$0")/../.."
SIZE="$1"; LABEL="$2"; ORDER="${3:-50}"
EXIT="runs/logs/train$SIZE.exit"

until [ -f "$EXIT" ]; do sleep 60; done
[ "$(cat "$EXIT")" = 0 ] || { echo "train$SIZE failed (exit $(cat "$EXIT"))"; exit 1; }

first_gpu="${GPUS:-5,6,7}"; first_gpu="${first_gpu%%,*}"
[ -f "models/hf/ft-$SIZE/config.json" ] || \
  CUDA_VISIBLE_DEVICES="$first_gpu" .venv-train/bin/python train/merge.py "runs/train-$SIZE/adapter" "models/hf/ft-$SIZE"
bash scripts/dgx/eval_model.sh "models/hf/ft-$SIZE" "ft-$SIZE" "$LABEL" "$ORDER"
