# TinyScribe: one target per pipeline step. Run `make help`.
SHELL := /bin/bash
PY := .venv/bin/python
# macOS: Pillow finds Homebrew's fribidi/harfbuzz (needed for Indic shaping via raqm) only with this
ifeq ($(shell uname),Darwin)
export DYLD_FALLBACK_LIBRARY_PATH := /opt/homebrew/lib
endif

WORKERS ?= 8
SCALE ?= 1.0
GPUS ?= 4,5,6,7
SLOTS ?= 2
# serve_pool.sh ports: 8100 + GPU index on the DGX, 8081 on the Mac
ifeq ($(shell uname),Darwin)
SERVERS ?= http://127.0.0.1:8081
else
SERVERS ?= $(shell echo $(GPUS) | tr , '\n' | awk '{printf "%shttp://127.0.0.1:%d", (NR>1?",":""), 8100+$$1}')
endif
SIZE ?= 9b

.PHONY: help setup test schemas data data-smoke qa llama serve stop eval table train quantize bench demo

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup: ## Mac: create .venv with datagen + dev deps
	uv venv --python 3.11 && uv pip install -e ".[datagen,dev]" huggingface_hub

test: ## unit tests
	$(PY) -m pytest -q

schemas: ## regenerate schemas/*.schema.json from templates
	$(PY) -m tinyscribe.schema

data: ## build the dataset into data/ (WORKERS=, SCALE=)
	$(PY) datagen/build.py --workers $(WORKERS) --scale $(SCALE)

data-smoke: ## 2% build into data_smoke/ for quick checks
	$(PY) datagen/build.py --workers $(WORKERS) --scale 0.02 --out data_smoke

qa: ## HTML contact sheet of 50 pages: data/qa.html
	$(PY) datagen/qa_sheet.py --n 50

llama: ## build llama.cpp at the pinned commit (CUDA on DGX, Metal on Mac)
	bash scripts/build_llamacpp.sh

serve: ## start servers: make serve MODEL=x.gguf MMPROJ=y.gguf [GPUS=4,5,6,7 SLOTS=2]
	bash scripts/serve_pool.sh start $(MODEL) $(MMPROJ) $(GPUS) $(SLOTS)

stop: ## stop servers started by `make serve`
	bash scripts/serve_pool.sh stop

eval: ## score running servers: make eval NAME=zero-shot-9b LABEL="9B zero-shot, BF16 (DGX)"
	$(PY) eval/run_eval.py --name $(NAME) --label "$(LABEL)" --servers $(SERVERS) --slots $(SLOTS) $(EVAL_ARGS)

table: ## print the results table from runs/
	$(PY) eval/results_table.py

train: ## LoRA fine-tune on one GPU (DGX): make train SIZE=9b|2b [GPU=4]
	CUDA_VISIBLE_DEVICES=$(or $(GPU),4) .venv-train/bin/python train/sft.py --config train/config_$(SIZE).yaml

quantize: ## Week 3: imatrix + quant grid (DGX)
	@echo "Week 3: quant/ grid not written yet"; exit 1

bench: ## Week 3: Mac speed + memory
	@echo "Week 3: bench/bench_mac.py not written yet"; exit 1

demo: ## Week 4: local web demo
	@echo "Week 4: app/ not written yet"; exit 1
