#!/usr/bin/env bash
# Convert any Hugging Face Qwen3.5 checkpoint to GGUF (BF16 + F16 projector), serve it on the free
# GPUs, score the test split, stop the servers. Used for the fine-tuned models; zero_shot.sh does
# the same for the base models.
#   bash scripts/dgx/eval_model.sh models/hf/ft-9b ft-9b "9B fine-tuned, BF16 (DGX)" 3
# Projector test: MMPROJ=q8_0 serves a Q8_0 vision projector (converted on first use) instead of F16.
# GGUF_DIR reuses an existing GGUF directory (e.g. models/gguf/base-9b).
# EVAL_ARGS passes extra run_eval.py options (e.g. "--limit 8" for a quick check).
# MODEL_FILE scores another language-model file from that directory (a quant) instead of model-bf16.gguf.
#   MMPROJ=q8_0 GGUF_DIR=models/gguf/base-9b bash scripts/dgx/eval_model.sh models/hf/Qwen3.5-9B \
#     zero-shot-9b-mmproj-q8 "9B zero-shot, BF16 + Q8_0 projector (DGX)" 10
set -euo pipefail
cd "$(dirname "$0")/../.."
HF="$1"; NAME="$2"; LABEL="$3"; ORDER="${4:-50}"
GPUS="${GPUS:-$(bash scripts/dgx/wait_gpus.sh 4 1)}"; SLOTS="${SLOTS:-2}"   # waits for free GPUs within the account's 4-GPU cap
OUT="${GGUF_DIR:-models/gguf/$NAME}"; MMPROJ="${MMPROJ:-f16}"

[ -f "$OUT/model-bf16.gguf" ] || bash quant/convert.sh "$HF" "$OUT"
MODEL_FILE="${MODEL_FILE:-$OUT/model-bf16.gguf}"
[ -f "$OUT/mmproj-$MMPROJ.gguf" ] || .venv-convert/bin/python third_party/llama.cpp/convert_hf_to_gguf.py \
  "$HF" --mmproj --outtype "$MMPROJ" --outfile "$OUT/mmproj-$MMPROJ.gguf"

export POOL="$NAME"   # own pid file, so evals running side by side stop only their own servers
trap 'bash scripts/serve_pool.sh stop' EXIT
servers=$(bash scripts/serve_pool.sh start "$MODEL_FILE" "$OUT/mmproj-$MMPROJ.gguf" "$GPUS" "$SLOTS" \
          | sed -n 's/^servers: //p')
echo "servers: $servers"
gb=$(awk "BEGIN{printf \"%.2f GB\", $(stat -c %s "$MODEL_FILE")/1e9}")
.venv/bin/python eval/run_eval.py --name "$NAME" --label "$LABEL" --servers "$servers" \
  --slots "$SLOTS" ${EVAL_ARGS:-} --meta "{\"order\": $ORDER, \"size\": \"$gb\", \"mmproj\": \"$MMPROJ\"}"
.venv/bin/python eval/results_table.py
