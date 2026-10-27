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

.PHONY: help setup test schemas data data-smoke qa llama serve stop eval table train pipeline quantize bench parse demo

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

train: ## LoRA fine-tune on one GPU (DGX; waits for a GPU within the account's cap): make train SIZE=9b|2b
	bash scripts/dgx/train.sh $(SIZE)

pipeline: ## DGX, all steps after the data for one size, queued in tmux: make pipeline SIZE=9b|2b
	bash scripts/dgx/tjob.sh pipe$(SIZE) bash scripts/dgx/pipeline.sh $(SIZE)

quantize: ## DGX: calibration text, generic + form imatrix, quant grid for models/gguf/ft-9b
	mkdir -p runs/quant
	[ -f runs/quant/calib_form.txt ] || $(PY) quant/calib_text.py form --out runs/quant/calib_form.txt
	[ -f runs/quant/calib_generic.txt ] || $(PY) quant/calib_text.py generic --out runs/quant/calib_generic.txt
	for m in generic form; do [ -f models/gguf/ft-9b/imatrix-$$m.gguf ] || \
	  bash quant/imatrix.sh models/gguf/ft-9b/model-bf16.gguf runs/quant/calib_$$m.txt models/gguf/ft-9b/imatrix-$$m.gguf; done
	bash quant/quantize.sh models/gguf/ft-9b

BENCH_MODEL ?= models/mac/ft-9b/IQ3_XXS_oq4_K-form.gguf
BENCH_MMPROJ ?= models/mac/ft-9b/mmproj-q8_0.gguf
BENCH_RUN ?= ft-9b-IQ3_XXS_oq4_K-form
bench: ## Mac: s/page (Metal, CPU) + peak memory: make bench BENCH_RUN= BENCH_MODEL= BENCH_MMPROJ=
	$(PY) bench/bench_mac.py --run $(BENCH_RUN) --model $(BENCH_MODEL) --mmproj $(BENCH_MMPROJ)

MODEL_NAME ?= 9b
parse: ## read one page: make parse IMAGE=form.jpg SCHEMA=eci_form7_hi_p0 [MODEL_NAME=2b]
	.venv/bin/tinyscribe parse $(IMAGE) --schema $(SCHEMA) --model $(MODEL_NAME)

demo: ## local web demo on http://127.0.0.1:8000 [MODEL_NAME=2b]
	$(PY) app/server.py --model $(MODEL_NAME)
