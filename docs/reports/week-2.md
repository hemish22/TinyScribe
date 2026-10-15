# Week 2 — fine-tune on the DGX

Last updated: Oct 16, 2026 · not started

## 1. Plan

- `train/sft.py`: Unsloth `FastVisionModel`, bf16 LoRA r=16 on language attention + MLP, vision tower frozen. Uses the same prompt (`tinyscribe.prompt`) and thinking-off chat template as inference, with compact JSON targets. 1 epoch over the train split, eval loss on 200 held-out train-type pages, checkpoints every ~500 steps.
- Train the 9B and the 2B at the same time on separate GPUs (in tmux).
- `train/merge.py`: merge LoRA into the base weights, then run `quant/convert.sh` to get a BF16 GGUF and F16 projector.
- Score the fine-tuned 9B and 2B in BF16 through llama.cpp (`make eval`).
- Projector test: F16 vs Q8_0 vision projector on ~50 test pages; keep Q8_0 only if accuracy holds.

## 2. In progress

## 3. Done

## 4. Needs your input
