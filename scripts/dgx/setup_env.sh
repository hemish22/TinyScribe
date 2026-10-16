#!/usr/bin/env bash
# One-time DGX setup (no sudo). Run from the repo root on the DGX:
#   bash scripts/dgx/setup_env.sh
# Creates .venv (Python 3.11, datagen + eval deps), a CUDA toolchain env for building llama.cpp
# (.cuda, via micromamba: nvcc is not installed system-wide), and links the handwriting banks
# that IndicFormBench already built on this machine.
set -euo pipefail
cd "$(dirname "$0")/../.."

IFB="${IFB:-$HOME/indicformbench}"   # IndicFormBench checkout with tools/hw_datasets

echo "== python env"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[datagen,dev]"
.venv/bin/python -c "from PIL import features; assert features.check('raqm'), 'Pillow without raqm: install libfribidi'; print('raqm ok')"

echo "== CUDA toolchain for llama.cpp (micromamba, conda-forge)"
MM="$(command -v micromamba || ls "$IFB/tools/bin/micromamba" "$HOME/.local/bin/micromamba" 2>/dev/null | head -1 || true)"
if [ -z "$MM" ] || [ ! -x "$MM" ]; then
  echo "micromamba not found; install: curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/.local bin/micromamba"
  exit 1
fi
if [ ! -d .cuda ]; then
  "$MM" create -y -p "$PWD/.cuda" -c conda-forge \
    cuda-version=12.6 cuda-nvcc cuda-cudart-dev libcublas-dev cmake ninja "gxx_linux-64<14"
fi
.cuda/bin/nvcc --version | tail -1

echo "== handwriting banks (EMNIST + IIIT Hindi) from IndicFormBench"
mkdir -p tools
if [ -d "$IFB/tools/hw_datasets" ]; then
  ln -sfn "$IFB/tools/hw_datasets" tools/hw_datasets
  ls tools/hw_datasets
else
  echo "WARNING: $IFB/tools/hw_datasets not found; data will be printed-only (set IFB=...)"
fi
echo "setup done"
