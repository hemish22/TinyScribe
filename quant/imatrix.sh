#!/usr/bin/env bash
# Build one importance matrix with llama-imatrix on one free GPU.
#   bash quant/imatrix.sh models/gguf/ft-9b/model-bf16.gguf runs/quant/calib_form.txt models/gguf/ft-9b/imatrix-form.gguf
# Every matrix gets the same budget (CHUNKS x CTX tokens), so the generic and the form matrix see
# the same number of tokens. --parse-special reads the chat tokens in the form text as special tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; TEXT="$2"; OUT="$3"
CHUNKS="${CHUNKS:-250}"; CTX="${CTX:-1024}"
BIN=third_party/llama.cpp/build/bin
if [ "$(uname)" = "Linux" ]; then
  GPU="${GPU:-$(bash scripts/dgx/wait_gpus.sh 1)}"
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
"$BIN/llama-imatrix" -m "$MODEL" -f "$TEXT" -o "$OUT" --parse-special -ngl 99 \
  -c "$CTX" --chunks "$CHUNKS" --output-frequency 50
"$BIN/llama-imatrix" -m "$MODEL" --in-file "$OUT" --show-statistics 2>&1 | tail -5
