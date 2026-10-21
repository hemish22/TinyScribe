#!/usr/bin/env bash
# Score every quant in DIR/quants/ (from quant/quantize.sh) with the Q8_0 vision projector.
#   bash scripts/dgx/eval_quants.sh models/hf/ft-9b ft-9b 9B 20
# Run names: <NAME>-<quant>-<imatrix>, labels like "9B fine-tuned, IQ2_M + form imatrix".
# Runs that already have scores.json are skipped, so the grid can be resumed.
set -euo pipefail
cd "$(dirname "$0")/../.."
HF="$1"; NAME="$2"; SIZE="$3"; ORDER="${4:-20}"
DIR="models/gguf/$NAME"
for f in "$DIR"/quants/*.gguf; do
  base=$(basename "$f" .gguf); quant="${base%-*}"; imat="${base##*-}"
  run="$NAME-$base"
  [ -f "runs/$run/scores.json" ] && { echo "have $run"; ORDER=$((ORDER + 1)); continue; }
  case "$imat" in none) how="";; *) how=" + $imat imatrix";; esac
  label="$SIZE fine-tuned, ${quant/_oq4_K/ (q4_K output)}$how"
  MODEL_FILE="$f" GGUF_DIR="$DIR" MMPROJ=q8_0 bash scripts/dgx/eval_model.sh "$HF" "$run" "$label" "$ORDER"
  ORDER=$((ORDER + 1))
done
