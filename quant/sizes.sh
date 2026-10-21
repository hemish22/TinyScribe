#!/usr/bin/env bash
# Predicted file size of each quant type, without writing any file (llama-quantize --dry-run).
#   bash quant/sizes.sh models/gguf/ft-9b/model-bf16.gguf [Q4_K_M IQ3_XXS:q4_K ...]
# TYPE:OUT also sets the output tensor type (as in quant/quantize.sh). Sizes in MiB and decimal GB;
# the project's budget is 4.0 GB = 4.0e9 bytes = 3815 MiB.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; shift
TYPES="${*:-Q8_0 Q4_K_M Q3_K_S IQ3_XXS IQ3_XXS:q4_K IQ2_M IQ2_XXS}"
BIN=third_party/llama.cpp/build/bin
# Qwen3.5's multi-token-prediction layer (blk.N.nextn*) never runs in llama.cpp decoding or in
# llama-imatrix, so it has no importance data and the IQ2 types refuse it. Its precision never matters: store it as q2_K.
nextn_args() {
  local n
  n=$("$BIN/llama-quantize" --dry-run "$1" Q8_0 2>&1 | grep -oE "blk\.[0-9]+\.nextn" | head -1 | cut -d. -f2)
  [ -n "$n" ] && echo "--tensor-type blk\.$n\.=q2_K"
  return 0
}
read -r -a NEXTN <<< "$(nextn_args "$MODEL")"
printf "%-14s %s\n" type "size"
for spec in $TYPES; do
  t="${spec%%:*}"; args=(${NEXTN[@]+"${NEXTN[@]}"})
  [ "$spec" != "$t" ] && args+=(--output-tensor-type "${spec#*:}")
  mib=$("$BIN/llama-quantize" --dry-run "${args[@]}" "$MODEL" "$t" 2>&1 | grep "quant size" | tail -1 \
        | sed -E 's/.*quant size *= *([0-9.]+) MiB.*/\1/')
  awk -v s="$spec" -v m="${mib:-0}" 'BEGIN{gb=m*1048576/1e9; printf "%-14s %8.0f MiB  %5.2f GB  %s\n", s, m, gb, (gb<=4.0?"<= 4 GB":"over 4 GB")}'
done
