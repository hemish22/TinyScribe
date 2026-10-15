#!/usr/bin/env bash
# Download the base Hugging Face weights (used for training AND converted to GGUF for the
# zero-shot baseline, so base and fine-tuned models go through the same converter).
#   bash scripts/dgx/download_models.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
for m in Qwen3.5-9B Qwen3.5-2B; do
  .venv/bin/python -m pip show huggingface_hub >/dev/null 2>&1 || uv pip install --python .venv/bin/python huggingface_hub
  .venv/bin/hf download "Qwen/$m" --local-dir "models/hf/$m"
done
du -sh models/hf/*
