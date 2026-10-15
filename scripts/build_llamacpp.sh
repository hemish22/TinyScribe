#!/usr/bin/env bash
# Build llama.cpp at the commit pinned in third_party/LLAMA_CPP_COMMIT, so the DGX (CUDA) and the
# Mac (Metal) run exactly the same code. First run with no pin: builds master and writes the pin
# (commit that file).
#   bash scripts/build_llamacpp.sh            # auto: CUDA on Linux (uses .cuda env), Metal on macOS
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/llama.cpp
PIN=third_party/LLAMA_CPP_COMMIT

mkdir -p third_party
[ -d "$SRC/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$SRC"
git -C "$SRC" fetch --quiet origin
if [ -s "$PIN" ]; then
  git -C "$SRC" checkout --quiet "$(cat "$PIN")"
else
  git -C "$SRC" checkout --quiet origin/master
  git -C "$SRC" rev-parse HEAD > "$PIN"
  echo "pinned llama.cpp at $(cat "$PIN")"
fi

ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
if [ "$(uname)" = "Linux" ]; then
  CUDA_ENV="$PWD/.cuda"
  export PATH="$CUDA_ENV/bin:$PATH" CUDACXX="$CUDA_ENV/bin/nvcc"
  export CC="$(ls "$CUDA_ENV"/bin/x86_64-conda-linux-gnu-gcc 2>/dev/null || command -v gcc)"
  export CXX="$(ls "$CUDA_ENV"/bin/x86_64-conda-linux-gnu-g++ 2>/dev/null || command -v g++)"
  ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDAToolkit_ROOT="$CUDA_ENV")
fi
cmake -S "$SRC" -B "$SRC/build" "${ARGS[@]}"
cmake --build "$SRC/build" -j "$(getconf _NPROCESSORS_ONLN)" \
  --target llama-server llama-mtmd-cli llama-imatrix llama-quantize llama-cli
ls "$SRC/build/bin"

# converter deps (convert_hf_to_gguf.py) in their own venv, apart from training deps
if [ ! -d .venv-convert ]; then
  uv venv --python 3.11 .venv-convert
  uv pip install --python .venv-convert/bin/python -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"
fi
echo "llama.cpp $(cat "$PIN") built"
