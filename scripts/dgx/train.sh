#!/usr/bin/env bash
# LoRA fine-tune one model on one free GPU (waits for one within the account's budget).
#   bash scripts/dgx/tjob.sh train9b bash scripts/dgx/train.sh 9b
set -euo pipefail
cd "$(dirname "$0")/../.."
SIZE="$1"
GPU="${GPU:-$(bash scripts/dgx/wait_gpus.sh 1)}"
echo "training $SIZE on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" .venv-train/bin/python train/sft.py --config "train/config_$SIZE.yaml"
