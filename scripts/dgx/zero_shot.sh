#!/usr/bin/env bash
# Zero-shot baseline for one base model: convert HF -> GGUF (same converter as the fine-tuned
# models), serve it on the free GPUs, score the test split, stop the servers.
#   bash scripts/dgx/zero_shot.sh 9B "9B zero-shot, BF16 (DGX)" 1
#   bash scripts/dgx/zero_shot.sh 2B "2B zero-shot, BF16 (DGX)" 2
set -euo pipefail
cd "$(dirname "$0")/../.."
SIZE="$1"; LABEL="$2"; ORDER="${3:-50}"
GPUS="${GPUS:-$(bash scripts/dgx/wait_gpus.sh 4 1)}"; SLOTS="${SLOTS:-2}"   # waits for free GPUs within the account's 4-GPU cap
lower=$(echo "$SIZE" | tr A-Z a-z)
OUT="models/gguf/base-$lower"

[ -f "$OUT/model-bf16.gguf" ] || bash quant/convert.sh "models/hf/Qwen3.5-$SIZE" "$OUT"

trap 'bash scripts/serve_pool.sh stop' EXIT
servers=$(bash scripts/serve_pool.sh start "$OUT/model-bf16.gguf" "$OUT/mmproj-f16.gguf" "$GPUS" "$SLOTS" \
          | sed -n 's/^servers: //p')
echo "servers: $servers"
gb=$(awk "BEGIN{printf \"%.2f GB\", $(stat -c %s "$OUT/model-bf16.gguf")/1e9}")
.venv/bin/python eval/run_eval.py --name "zero-shot-$lower" --label "$LABEL" --servers "$servers" \
  --slots "$SLOTS" --meta "{\"order\": $ORDER, \"size\": \"$gb\"}"
.venv/bin/python eval/results_table.py
