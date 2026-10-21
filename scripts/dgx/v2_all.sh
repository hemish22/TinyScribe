#!/usr/bin/env bash
# Data v2 end to end: build the dataset, then queue the per-model pipelines and the zero-shot
# re-runs on the new test split (each in its own tmux window, each waiting for GPU budget).
#   bash scripts/dgx/tjob.sh v2all bash scripts/dgx/v2_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f data/manifest.jsonl ] || .venv/bin/python datagen/build.py --workers "${WORKERS:-48}"
bash scripts/dgx/tjob.sh pipe9b bash scripts/dgx/pipeline.sh 9b
sleep 200   # let the first job claim its GPU before the next one looks
bash scripts/dgx/tjob.sh pipe2b bash scripts/dgx/pipeline.sh 2b
sleep 200
bash scripts/dgx/tjob.sh zs9b bash scripts/dgx/zero_shot.sh 9B "9B zero-shot, BF16 (DGX)" 1
sleep 200
bash scripts/dgx/tjob.sh zs2b bash scripts/dgx/zero_shot.sh 2B "2B zero-shot, BF16 (DGX)" 2
