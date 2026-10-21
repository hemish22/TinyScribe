#!/usr/bin/env bash
# Everything after the dataset for one model size, in order; every GPU step waits for a free GPU
# within the account's budget, so this can be queued and left running in tmux.
#   bash scripts/dgx/tjob.sh pipe9b bash scripts/dgx/pipeline.sh 9b
#   bash scripts/dgx/tjob.sh pipe2b bash scripts/dgx/pipeline.sh 2b
# 9b: train -> merge -> score BF16 -> calibration text -> generic + form imatrix -> quant grid ->
#     score every quant (Q8_0 projector).  2b: train -> merge -> score BF16 -> Q8_0 -> score.
# Finished steps are skipped on a rerun (adapter, merged model, scores, imatrices, quants).
set -euo pipefail
cd "$(dirname "$0")/../.."
SIZE="$1"; S=$(echo "$SIZE" | tr a-z A-Z)
HF="models/hf/ft-$SIZE"; DIR="models/gguf/ft-$SIZE"

[ -f "runs/train-$SIZE/adapter/adapter_config.json" ] || bash scripts/dgx/train.sh "$SIZE"
if [ ! -f "$HF/config.json" ]; then
  GPU=$(bash scripts/dgx/wait_gpus.sh 1)
  CUDA_VISIBLE_DEVICES="$GPU" .venv-train/bin/python train/merge.py "runs/train-$SIZE/adapter" "$HF"
fi
order=$([ "$SIZE" = 9b ] && echo 3 || echo 4)
[ -f "runs/ft-$SIZE/scores.json" ] || bash scripts/dgx/eval_model.sh "$HF" "ft-$SIZE" "$S fine-tuned, BF16 (DGX)" "$order"

if [ "$SIZE" = 9b ]; then
  mkdir -p runs/quant
  [ -f runs/quant/calib_form.txt ] || .venv/bin/python quant/calib_text.py form --out runs/quant/calib_form.txt
  [ -f runs/quant/calib_generic.txt ] || .venv/bin/python quant/calib_text.py generic --out runs/quant/calib_generic.txt
  for m in generic form; do
    [ -f "$DIR/imatrix-$m.gguf" ] || bash quant/imatrix.sh "$DIR/model-bf16.gguf" "runs/quant/calib_$m.txt" "$DIR/imatrix-$m.gguf"
  done
  bash quant/quantize.sh "$DIR"
  bash scripts/dgx/eval_quants.sh "$HF" "ft-$SIZE" "$S" 10
else
  TYPES=Q8_0 IMATS=none bash quant/quantize.sh "$DIR"
  bash scripts/dgx/eval_quants.sh "$HF" "ft-$SIZE" "$S" 30
fi
.venv/bin/python eval/results_table.py
