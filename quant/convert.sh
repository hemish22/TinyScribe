#!/usr/bin/env bash
# Convert a Hugging Face Qwen3.5 checkpoint (base or merged fine-tune) to a BF16 GGUF plus its
# F16 vision projector.
#   bash quant/convert.sh models/hf/Qwen3.5-9B models/gguf/base-9b
# -> models/gguf/base-9b/model-bf16.gguf, models/gguf/base-9b/mmproj-f16.gguf
set -euo pipefail
cd "$(dirname "$0")/.."
HF="$1"; OUT="$2"
CONV=third_party/llama.cpp/convert_hf_to_gguf.py
mkdir -p "$OUT"
.venv-convert/bin/python "$CONV" "$HF" --outtype bf16 --outfile "$OUT/model-bf16.gguf"
.venv-convert/bin/python "$CONV" "$HF" --mmproj --outtype f16 --outfile "$OUT/mmproj-f16.gguf"
ls -la "$OUT"
