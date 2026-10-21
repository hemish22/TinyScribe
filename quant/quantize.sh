#!/usr/bin/env bash
# Quant grid for one model directory (CPU only, no GPU).
#   bash quant/quantize.sh models/gguf/ft-9b            # TYPES x {generic, form} imatrix
#   TYPES=Q8_0 IMATS=none bash quant/quantize.sh models/gguf/ft-2b
# Reads DIR/model-bf16.gguf and DIR/imatrix-<name>.gguf; writes DIR/quants/<TYPE>-<name>.gguf
# (<name> = none when no imatrix). Existing files are kept, so the grid can be resumed.
# TYPE:OUT sets the output tensor type too: IQ3_XXS:q4_K -> IQ3_XXS_oq4_K-<name>.gguf. IQ3_XXS's
# default q5_K output tensor (248k vocab) leaves the file at 4.12 GB; q4_K brings it under 4.0 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$1"
TYPES="${TYPES:-Q4_K_M Q3_K_S IQ3_XXS:q4_K IQ2_M IQ2_XXS}"; IMATS="${IMATS:-generic form}"
THREADS="${THREADS:-32}"
BIN=third_party/llama.cpp/build/bin
# Qwen3.5's multi-token-prediction layer (blk.N.nextn*) never runs in llama.cpp decoding or in
# llama-imatrix, so it has no importance data and the IQ2 types refuse it. Its precision never matters: store it as q2_K.
nextn_args() {
  local n
  n=$("$BIN/llama-quantize" --dry-run "$1" Q8_0 2>&1 | grep -oE "blk\.[0-9]+\.nextn" | head -1 | cut -d. -f2)
  [ -n "$n" ] && echo "--tensor-type blk\.$n\.=q2_K"
  return 0
}
mkdir -p "$DIR/quants"
read -r -a NEXTN <<< "$(nextn_args "$DIR/model-bf16.gguf")"
for spec in $TYPES; do
  t="${spec%%:*}"; o=""; name="$t"
  [ "$spec" != "$t" ] && { o="${spec#*:}"; name="${t}_o$o"; }
  for m in $IMATS; do
    out="$DIR/quants/$name-$m.gguf"
    [ -f "$out" ] && { echo "have $out"; continue; }
    args=(${NEXTN[@]+"${NEXTN[@]}"})
    [ "$m" = none ] || args+=(--imatrix "$DIR/imatrix-$m.gguf")
    [ -n "$o" ] && args+=(--output-tensor-type "$o")
    "$BIN/llama-quantize" "${args[@]}" "$DIR/model-bf16.gguf" "$out.tmp" "$t" "$THREADS" > "$out.log" 2>&1
    mv "$out.tmp" "$out"
    echo "$(du -h "$out" | cut -f1)  $out"
  done
done
