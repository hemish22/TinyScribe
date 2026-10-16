#!/usr/bin/env bash
# Training environment on the DGX: .venv-train with Unsloth (torch + CUDA wheels, transformers v5, trl, peft).
#   bash scripts/dgx/tjob.sh trainenv bash scripts/dgx/setup_train_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv-train ] || uv venv --python 3.11 .venv-train
uv pip install --python .venv-train/bin/python -r train/requirements.txt
uv pip install --python .venv-train/bin/python -e .
.venv-train/bin/python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
import unsloth
print("unsloth", unsloth.__version__)
PY
