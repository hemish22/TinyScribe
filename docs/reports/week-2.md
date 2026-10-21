# Week 2 — fine-tune on the DGX

Last updated: Oct 21, 2026 · **complete**

## 1. Plan

- `train/sft.py`: Unsloth `FastVisionModel`, bf16 LoRA r=16 on language attention + MLP, vision tower frozen.
  - Uses the same prompt (`tinyscribe.prompt`, v2) and thinking-off chat template as inference, with compact JSON targets.
  - 1 epoch over the train split. Eval loss on 200 held-out train-split pages. Checkpoints every 200 steps.
- Train the 9B and the 2B at the same time on separate GPUs (in tmux).
- `train/merge.py`: merge LoRA into the base weights. Then `scripts/dgx/eval_model.sh` converts to BF16 GGUF + F16 projector and scores through llama.cpp.
- Score the fine-tuned 9B and 2B in BF16 through llama.cpp.
- Projector test: F16 vs Q8_0 vision projector on ~50 test pages; keep Q8_0 only if accuracy holds.

Baselines to beat (zero-shot, prompt v2, field F1): **9B 84.9**, **2B 69.0**.

## 2. In progress

- Nothing. Week 2 is complete; Week 3 (quantization) is next.

## 3. Done

### Results (test split, 401 pages, BF16 through llama.cpp, grammar on, prompt v2)

| Model | Size | Field F1 | English | Hindi | Tamil | Bengali | Valid JSON | Checkbox acc. |
|---|---|---|---|---|---|---|---|---|
| 9B zero-shot | 18.41 GB | 84.9 | 90.6 | 72.8 | 85.4 | 91.4 | 100 | 89.4 |
| 2B zero-shot | 3.90 GB | 69.0 | 82.2 | 55.4 | 69.4 | 69.4 | 100 | 77.4 |
| **9B fine-tuned** | 18.41 GB | **91.9** | 95.5 | 83.3 | 91.5 | 97.4 | 100 | **99.5** |
| **2B fine-tuned** | 3.90 GB | **87.4** | 93.5 | 77.6 | 85.1 | 93.0 | 100 | 96.5 |
| 9B fine-tuned + Q8_0 projector | 18.41 GB + 624 MB | 91.9 | 95.5 | 83.4 | 91.5 | 97.4 | 100 | 99.4 |

- Fine-tuning: **9B +7.0 points**, **2B +18.4 points**. The fine-tuned 2B beats the zero-shot 9B.
- The test form types (ECI Form 7, synth job application) were never seen in training. On ECI Form 7, the 2B goes from 44.3 → 88.4 and the 9B from 71.8 → 90.7.
- **9B training:** 9,806 pages, 1,226 steps, 2 h 44 min on one A100, peak 21.9 GB. Dev loss 0.0195 (step 200) → 0.0077 (end).
- **Projector decision: Q8_0.** It scores the same on both the base 9B (85.2 vs 84.9) and the fine-tuned 9B (91.9 vs 91.9), and is 294 MB smaller (624 vs 918 MB). Week 3 quants use `mmproj-q8_0.gguf`.

**Breakdown** (`eval/breakdown.py`, field F1 %):

| Group | Pages | 2B zero-shot | 2B fine-tuned | 9B zero-shot | 9B fine-tuned |
|---|---|---|---|---|---|
| english · handwritten | 45 | 74.1 | 86.4 | 82.5 | 90.2 |
| english · printed | 55 | 89.1 | 99.8 | 97.6 | 100.0 |
| hindi · handwritten | 44 | 41.1 | 59.5 | 54.9 | **69.5** |
| hindi · printed | 56 | 66.3 | 91.4 | 86.6 | 93.8 |
| tamil · printed | 101 | 69.4 | 85.1 | 85.4 | 91.5 |
| bengali · printed | 100 | 69.4 | 93.0 | 91.4 | 97.4 |
| clean | 100 | 71.5 | 90.5 | 88.6 | 93.4 |
| light | 99 | 71.5 | 88.2 | 87.0 | 93.6 |
| medium | 112 | 70.7 | 88.1 | 85.5 | 92.8 |
| heavy | 90 | 60.9 | 81.9 | 77.3 | 86.8 |

- **Hindi's low score is handwriting.** Printed Hindi is 93.8; handwritten Hindi is 69.5. The IIIT Hindi word bank has a small vocabulary and test writers are held out, so this is the hardest slice. Goes into LIMITATIONS.md.
- Printed English is essentially solved (100.0). Tamil printed (91.5) is the weakest printed script.
- Heavy degradation costs about 6–7 points against clean.

### Build and fixes

- **`train/sft.py` written.** Each training sample is exactly what llama-server sees at inference:
  - The image plus the prompt in the model's own chat template, with `enable_thinking=False`, so the assistant turn opens `<think>\n\n</think>\n\n`.
  - Target: compact JSON + `<|im_end|>`. Loss is on the target only.
  - A start-up check makes sure the prompt/target token boundary does not merge, which would shift the label mask.
  - Resumes from the last checkpoint if the job is restarted.
- **`train/config_9b.yaml`, `train/config_2b.yaml`:** lr 1e-4 (9B) / 2e-4 (2B), effective batch 8, ~1,230 steps, max 8,192 tokens.
- **Image preprocessing checked:** Qwen3.5's processor (patch 16, merge 2, up to 16.7M pixels) keeps the 1086×1536 pages at about 1,632 vision tokens, the same grid llama.cpp uses.
- **`train/merge.py`** (LoRA → full 16-bit checkpoint, base chat template kept) and **`scripts/dgx/eval_model.sh`** (convert → serve → score) written.

- **`eval/breakdown.py`** (new, tested): field F1 per run split by script × handwritten/printed, by degradation and by form type, as markdown tables.
- **Training env** `.venv-train`: torch 2.14.1+cu130, transformers 5.17.0, trl 1.15.0, peft 0.21.2, unsloth 2026.10.6, plus `flash-linear-attention` 0.5.2. Without fla, the Gated-DeltaNet layers ran a slow torch reference path.
- **Smoke runs passed** (10 steps):
  - 2B: loss 0.47 → 0.18, dev loss 0.22 → 0.17, peak 6.2 GB.
  - 9B: loss 0.33 → 0.18, dev loss 0.25 → 0.20, peak 21.7 GB. It fits one 40 GB A100.
- **Training speed fix: 6.5× faster.** The full run took about 18 s/step while the GPU sat idle about 80% of the time.
  - Not data loading: collate time ≈ 0, and 0 vs 4 workers made no difference.
  - Kernels are compiled per sequence length, and almost every batch had a new length.
  - Fix: pad each batch to a multiple of 128 tokens (`pad_multiple: 128`), about 2% extra compute.
  - A/B on 40 steps, 400 random pages, fresh caches: unpadded 13–46 s/step; padded **≈2.7 s/step**.
- Truncated pages (1 in train) are now dropped from training.
- **Multi-GPU (DDP) tried, not usable.**
  - With Unsloth 2026.10.6 + Qwen3.5, 2-GPU `torchrun` fails in backward: `CheckpointError: A different number of tensors was saved during the original forward and recomputation`. Plain gradient checkpointing fails the same way.
  - Instead each model trains on its own GPU, and the spare GPUs run the projector test.
- **Merge + convert pipeline checked** on the 2B smoke adapter:
  - `train/merge.py` → full checkpoint with the same 632 tensors as the base.
  - All 297 vision tensors are byte-identical, so the vision tower stayed frozen; 186 of 335 language tensors changed (the LoRA targets).
  - `convert_hf_to_gguf.py` gives a 3.90 GB BF16 GGUF (same as the base) and a projector whose tensors match the base. Only the file metadata differs.
- **Projector test (base 9B, all 401 test pages): Q8_0 loses nothing.**

  | Projector | Size | Field F1 | English | Hindi | Tamil | Bengali | Checkbox acc. |
  |---|---|---|---|---|---|---|---|
  | F16 | 918 MB | 84.9 | 90.6 | 72.8 | 85.4 | 91.4 | 89.4 |
  | Q8_0 | 624 MB | 85.2 | 90.9 | 73.1 | 85.8 | 91.2 | 89.5 |

  Differences are noise (≤0.4 points per language). Q8_0 is kept for now; the fine-tuned 9B confirms it.
- Two evals can now run side by side: `serve_pool.sh` uses port 8100 + GPU index on the DGX and one pid file per pool (`POOL`). `make eval` derives `SERVERS` from `GPUS`, and `make train SIZE=… GPU=…` works.
- `scripts/dgx/eval_model.sh` now takes `MMPROJ=q8_0` (projector test) and `GGUF_DIR` (reuse an existing GGUF directory).

## 4. Needs your input

- Nothing for Week 2.
- Optional, still open from Week 1: Bengali/Tamil handwriting (IIIT-INDIC-HW-WORDS download). Today BN/TA test pages are printed only, so the model is untested on BN/TA handwriting.
- Before Week 3's Mac benchmarks: free 10–15 GB of Mac disk (about 24 GB free now; each quant file is 2–5 GB).
