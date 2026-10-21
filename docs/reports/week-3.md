# Week 3 — quantize on the DGX, time on the Mac

Last updated: Oct 21, 2026 · data v2 built; retraining queued (waiting for GPU budget); quant and Mac tooling ready

## 1. Plan

- `quant/calib_text.py`: form-imatrix text from the 800 calibration pages (label text + the fine-tuned model's JSON, chat-templated). Generic imatrix from general text.
- `quant/imatrix.sh`: build both importance matrices with `llama-imatrix`.
- `quant/sizes.sh`: check sizes with `llama-quantize --dry-run`. Then `quant/quantize.sh` builds the grid: Q4_K_M, Q3_K_S, IQ3_XXS, IQ2_M and IQ2_XXS, each with the generic and the form imatrix.
- Score every quant and the 2B Q8_0 on the DGX (4 servers).
- Mac: build llama.cpp at the pinned commit (Metal). `bench/bench_mac.py` measures s/page with Metal (`-ngl 99`) and CPU (`-ngl 0`) and peak memory, for each quant ≤ 4 GB plus the Q4_K_M anchor.
- Optional: simulate an 8 GB Mac with `sudo sysctl iogpu.wired_limit_mb=5300`.
- **Added Oct 21 (your request): Bengali and Tamil handwriting.** Download the data, add it to the generator, rebuild the dataset (v2), retrain both models, and re-score everything on the new test split. The quant grid then runs on the v2 models.

## 2. In progress

All queued in tmux on the DGX. Each GPU step waits until the `srmist18` account is under its 4-GPU cap (`scripts/dgx/wait_gpus.sh`), then starts by itself:

- `pipe9b` (`scripts/dgx/pipeline.sh 9b`):
  1. train
  2. merge
  3. score BF16
  4. calibration text → generic + form imatrix
  5. quant grid
  6. score every quant
- `pipe2b`: train → merge → score BF16 → Q8_0 → score.
- `zs9b`, `zs2b`: zero-shot baselines re-run on the v2 test split (queued after the pipelines).
- **Blocked right now:** other work under the same login holds all 4 GPUs (GPUs 1, 3, 6, 7). No TinyScribe job is running on a GPU.

## 3. Done

### Bengali and Tamil handwriting (data v2)

- **Downloaded** (`scripts/dgx/download_hw.sh`):
  - **Tamil:** IIIT-INDIC-HW-WORDS Tamil, from the Hugging Face mirror `c3rl/IIIT-INDIC-HW-WORDS-Tamil`, 4.9 GB. 13,294 words; 87k train / 16k test images. The test images are from separate writers.
  - **Bengali:** there is no copy of the IIIT Bengali set on Hugging Face (only Hindi and Tamil are mirrored). I used **BanglaWriting** instead:
    - Source: Mendeley Data 10.17632/r43wkvdk4w.1, licence **CC BY-NC 3.0**. 132 MB, hash verified.
    - 260 writers, one page each, every word boxed and labelled by hand: 5,178 words, 14.5k train / 3.7k test crops.
    - Writers with id % 5 = 0 are test.
- **Word banks** (`hw_bank.py tamil-vocab | bengali-vocab | indic-extract`):
  - Train and test samples are numbered so the existing train/test handwriting split keeps them apart.
  - Coverage of the words forms need is thin: IIIT Tamil is news vocabulary, BanglaWriting is free-text essays. Words with both train and test samples: **45 Tamil, 32 Bengali**.
- **Generator changes** (`handwriting.py`, `value_gen.py`):
  - A Bengali or Tamil field is written with **real word images** when every word is in the bank. Otherwise the whole field is written in a **handwriting-style font**, one character group at a time with jitter (Atma for Bengali, Kavivanar for Tamil, both OFL). Comb boxes work the same way, as Hindi already did with Kalam.
  - New handwriting lists use only words that have real samples: Tamil names, cities, relations, countries; Bengali relations, occupations, countries.
  - **Tamil scan clean-up:** IIIT Tamil scans keep frame lines and underline fragments that shrank the words. Rows or columns that are mostly ink are blanked, and samples with a long straight stroke are dropped.
- **Dataset v2** (`datagen/split.yaml` version 2): 10,411 pages, **5,172 handwritten**. v1 data kept as `data_v1`.

  | Handwritten pages | English | Hindi | Bengali | Tamil |
  |---|---|---|---|---|
  | train | 1,265 | 1,234 | 1,253 | 1,238 |
  | test | 45 | 44 | 52 | 41 |
  | fields drawn from real handwriting (test) | 100% | 98% | 59% | 69% |

  - "Real" includes digits and Latin letters drawn from EMNIST, so the share of real **Bengali/Tamil words** is lower.
  - The rest is the handwriting font, which looks close to print for Tamil.
  - Goes into LIMITATIONS.md.
- v1 runs and models are archived (`runs/v1/`, `models/*/v1/`). The v1 results stay in the Week 2 report.

### Quantization tooling (tested on the v1 fine-tuned 9B)

- `quant/calib_text.py`:
  - **Form text:** the 800 calibration pages as the exact chat-templated sequence (checked against the processor's template), with ground-truth compact JSON as the assistant turn. Pages are shuffled, so any prefix mixes all four scripts.
  - **Generic text:** WikiText-2.
- `quant/imatrix.sh`: same budget for both matrices, 250 chunks × 1,024 tokens. About 14 min per matrix on one A100.
- `quant/sizes.sh` / `quant/quantize.sh`, findings:
  - **Budget:** 4 GB means 4.0×10⁹ bytes (3,815 MiB).
  - **Prediction head:** Qwen3.5's multi-token-prediction layer (`blk.32.nextn`) never runs in llama.cpp, so it has no imatrix data, and IQ2 types refuse to quantize without it. It is stored as q2_K; its precision never matters.
  - **IQ3_XXS:** 4.13 GB by default, because its q5_K output tensor (248k vocabulary) is large. With `--output-tensor-type q4_K` it is **3.98 GB**.

  | Quant (9B) | Size | ≤ 4 GB? |
  |---|---|---|
  | Q4_K_M | 5.70 GB | anchor only |
  | Q3_K_S | 4.33 GB | no (kept as reference) |
  | IQ3_XXS, q4_K output | 3.98 GB | yes |
  | IQ2_M | 3.68 GB | yes |
  | IQ2_XXS | 3.17 GB | yes |
  | 2B Q8_0 | 2.03 GB | yes |

  - End-to-end test: a test imatrix gave an IQ2_XXS of 3.0 GB, which quantized fine.
- `scripts/dgx/eval_quants.sh` scores every quant with the Q8_0 projector. `eval_model.sh` gained `MODEL_FILE` and `EVAL_ARGS`.
- **GPU etiquette** (your rule: at most 4 GPUs for the whole `srmist18` account):
  - `scripts/dgx/pick_gpus.sh` counts every srmist18 process on the GPUs.
  - `scripts/dgx/wait_gpus.sh` waits for budget and claims a GPU for 3 minutes, so two of my jobs can't race for it.

### Mac

- llama.cpp built at the pinned commit `d81235049` with Metal (cmake installed into `.venv`, nothing system-wide). 819 MB.
- `bench/bench_mac.py`:
  - **Metal:** `-ngl 99`.
  - **CPU only:** `--device none --no-op-offload --no-mmproj-offload`. With plain `-ngl 0`, llama.cpp still sends large matrix operations to Metal.
  - **Pages:** each page goes through the same client as the DGX scorer; the first page is a warm-up.
  - **Peak memory:** `/usr/bin/time -l` max RSS, which counts the memory-mapped weights; macOS's footprint is kept as well.
  - **Bug found and fixed:** an orphaned server answered later runs. The benchmark now refuses to start if the port is taken, and stops the server process itself.
- Smoke test (2B Q4_K_M, old v1 pages): Metal 13.6 s/page, CPU 51.6 s/page, peak 2.99 GB.

## 4. Needs your input

- **GPUs:** other work under `srmist18` currently holds all 4 GPUs of the account's cap, so TinyScribe is queued. If that work can make room, the 9B retrain takes about 3 h on one GPU and the 2B about 1 h.
- **Mac disk:** 16 GB free. Copying a few quants (3–4 GB each) for timing will need most of it. Freeing 10 GB more would help.
- **Licence note:** BanglaWriting is CC BY-NC 3.0, and IIIT-INDIC-HW-WORDS is for research use. The released model and data must say non-commercial. Fine for a portfolio project; the model card will state it.
