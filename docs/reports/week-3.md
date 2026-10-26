# Week 3 — quantize on the DGX, time on the Mac

Last updated: Oct 26, 2026 · **done**: v2 models retrained, quant grid scored on the DGX, 4 models timed on the Mac

## 1. Plan

- `quant/calib_text.py`: form-imatrix text from the 800 calibration pages (label text + the fine-tuned model's JSON, chat-templated). Generic imatrix from general text.
- `quant/imatrix.sh`: build both importance matrices with `llama-imatrix`.
- `quant/sizes.sh`: check sizes with `llama-quantize --dry-run`. Then `quant/quantize.sh` builds the grid: Q4_K_M, Q3_K_S, IQ3_XXS, IQ2_M and IQ2_XXS, each with the generic and the form imatrix.
- Score every quant and the 2B Q8_0 on the DGX (4 servers).
- Mac: build llama.cpp at the pinned commit (Metal). `bench/bench_mac.py` measures s/page with Metal (`-ngl 99`) and CPU (`-ngl 0`) and peak memory, for each quant ≤ 4 GB plus the Q4_K_M anchor.
- Optional: simulate an 8 GB Mac with `sudo sysctl iogpu.wired_limit_mb=5300`.
- **Added Oct 21 (your request): Bengali and Tamil handwriting.** Download the data, add it to the generator, rebuild the dataset (v2), retrain both models, and re-score everything on the new test split. The quant grid then runs on the v2 models.

## 2. In progress

- Score breakdown (script × handwriting, degradation, form type) on the v2 runs. CPU only, on the DGX; waiting for the DGX to be reachable from the Mac again.
- Optional: 8 GB Mac simulation. The memory numbers below already suggest every model ≤ 4 GB fits.

History:

- **Resumed Oct 22 (08:24), then yielded.** IndicFormBench's `run_v4_pipeline.sh` started 5 GPU jobs in the same seconds, one on the GPU my 9B had just picked. I stopped the 9B to keep the account at 4 GPUs. `wait_gpus.sh` now re-checks after 30 s before claiming. The jobs then waited about 31 h for GPUs and ran Oct 23 15:10 – Oct 24 02:42.
- **Paused at your request (Oct 21, 10:48).**
  - The 2B retrain was stopped at step 641 of 1,226. Checkpoints at steps 200, 400 and 600 are kept in `runs/train-2b`.
  - The queued jobs (`pipe9b`, `zs9b`, `zs2b`) were stopped too. No TinyScribe process is on any GPU.
  - Resume: `bash scripts/dgx/tjob.sh v2resume bash scripts/dgx/v2_all.sh`. It skips the finished dataset, the 2B continues from checkpoint 600, and everything else starts from where it stopped.

Before the pause:

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

### Results (v2 test split: 401 pages, Bengali and Tamil handwriting included)

Accuracy from the DGX (llama.cpp CUDA, same weights and engine as the Mac). Speed and memory from the MacBook Pro M4, 16 GB.

| Model | Size | Field F1 | English | Hindi | Tamil | Bengali | Valid JSON | s / page (Metal) | s / page (CPU) | Peak memory |
|---|---|---|---|---|---|---|---|---|---|---|
| 9B zero-shot, BF16 (DGX) | 18.41 GB | 69.0 | 90.6 | 52.2 | 21.9 | 81.9 | 80.3 | | | |
| 2B zero-shot, BF16 (DGX) | 3.90 GB | 64.9 | 82.2 | 55.5 | 57.2 | 62.5 | 100.0 | | | |
| 9B fine-tuned, BF16 (DGX) | 18.41 GB | 88.6 | 95.5 | 83.4 | 85.4 | 89.0 | 100.0 | | | |
| 2B fine-tuned, BF16 (DGX) | 3.90 GB | 83.8 | 93.8 | 80.2 | 79.1 | 80.5 | 100.0 | | | |
| 9B fine-tuned, Q4_K_M + form imatrix (anchor) | 5.71 GB | 88.5 | 95.6 | 83.3 | 84.8 | 89.2 | 100.0 | 36.5 | 99.3 | 7.99 GB |
| 9B fine-tuned, Q4_K_M + generic imatrix | 5.71 GB | 88.4 | 95.3 | 83.2 | 84.9 | 89.0 | 100.0 | | | |
| 9B fine-tuned, Q3_K_S + form imatrix | 4.34 GB | 84.9 | 95.5 | 81.4 | 75.7 | 84.1 | 100.0 | | | |
| 9B fine-tuned, Q3_K_S + generic imatrix | 4.34 GB | 85.4 | 95.3 | 81.3 | 77.7 | 84.7 | 100.0 | | | |
| **9B fine-tuned, IQ3_XXS (q4_K output) + form imatrix** | **3.89 GB** | **86.5** | 95.1 | 81.6 | 80.6 | 87.0 | 100.0 | **33.2** | 107.2 | **6.29 GB** |
| 9B fine-tuned, IQ3_XXS (q4_K output) + generic imatrix | 3.89 GB | 86.4 | 95.3 | 80.6 | 81.6 | 86.5 | 100.0 | | | |
| 9B fine-tuned, IQ2_M + form imatrix | 3.69 GB | 84.3 | 94.6 | 79.1 | 76.1 | 85.1 | 100.0 | 33.8 | 110.7 | 6.10 GB |
| 9B fine-tuned, IQ2_M + generic imatrix | 3.69 GB | 83.9 | 94.7 | 77.5 | 76.8 | 84.2 | 100.0 | | | |
| 9B fine-tuned, IQ2_XXS + form imatrix | 3.18 GB | 72.9 | 92.1 | 69.5 | 52.6 | 71.7 | 100.0 | | | |
| 9B fine-tuned, IQ2_XXS + generic imatrix | 3.18 GB | 71.8 | 89.6 | 68.4 | 50.6 | 72.3 | 100.0 | | | |
| **2B fine-tuned, Q8_0** | **2.03 GB** | **84.0** | 93.5 | 80.3 | 79.2 | 81.3 | 100.0 | **14.5** | 39.5 | **3.49 GB** |

Findings:

- **Best model ≤ 4 GB: 9B IQ3_XXS (q4_K output) + form imatrix**, 86.5 field F1, 2.1 below BF16. About 33 s per page with Metal, 6.3 GB peak memory.
- **2B Q8_0** is the fast option: 2.3× faster (14.5 s/page), 3.5 GB, 2.5 points lower. Q8_0 loses nothing against the 2B BF16 (84.0 vs 83.8).
- **Form vs generic imatrix:** no real difference. Form wins by 0.1–1.1 on most types, generic wins on Q3_K_S by 0.5. The text-only imatrix cannot see the image tokens; this is the reason for the Week 4 stretch item (image-aware imatrix).
- **IQ2_M** is no faster than IQ3_XXS on Metal and 2.2 points worse. **IQ2_XXS** breaks Tamil (52.6).
- **Q4_K_M** keeps BF16 accuracy (88.5) but is over the 4 GB budget.
- **Fine-tuning** takes the 9B from 69.0 to 88.6. Zero-shot, the 9B writes valid JSON on only 80% of pages and gets 21.9 on Tamil.
- On CPU only, Q4_K_M is faster than the IQ types (IQ dequantization is costlier on CPU); on Metal it is slower (more bytes to read).
- The v2 test split is harder than v1 (Bengali and Tamil handwriting), so these numbers are not comparable with the Week 2 table.

### Mac benchmark (Oct 26)

- `bench/bench_mac.py` on each model: 21 test pages with Metal (`-ngl 99`) and 6 CPU-only, first page of each mode is a warm-up. Pages balanced over the four scripts, same pages for every model. Valid JSON on every timed page.
- About 40% of a 9B page's time is reading the page (vision encoder + about 2,300 prompt tokens at ~135 tokens/s), the rest is writing about 300 tokens of JSON at ~10 tokens/s (2B: ~23 tokens/s).
- **Peak memory, measured again with a better method.** The first numbers (`/usr/bin/time` max RSS) were wrong: the Q4_K_M showed 4.62 GB, less than its own 5.71 GB file.
  - macOS drops clean pages of the memory-mapped weights when memory is short. The Mac had 5–7 GB of swap in use from other apps during the run.
  - llama-server's host prompt cache (default up to 8 GB) kept every past page's prompt and inflated the footprint to 5.9 GB. It never helps here, since each page's image tokens differ, so the benchmark (and the Week 4 CLI and demo) run with `--cache-ram 0`.
  - **Peak memory now = model file + projector file + the server's peak working memory** (macOS `footprint`, sampled every 5 s over 6 pages). Working memory: 1.1 GB for the 2B, 1.7–1.8 GB for the 9B.
  - Timings were re-checked with the cache off: same per-page times.

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

- **IndicFormBench GPU count:** its `run_v4_pipeline.sh` (MAX_GPUS=4) held 5 GPUs at once on Oct 22 and does not count TinyScribe's processes. If both projects share the 4-GPU cap, it should count every `srmist18` process, as `scripts/dgx/pick_gpus.sh` does.
- **Mac disk:** solved (you freed space). The Mac models take 16.3 GB in `models/mac/`.
- **Licence note:** BanglaWriting is CC BY-NC 3.0, and IIIT-INDIC-HW-WORDS is for research use. The released model and data must say non-commercial. Fine for a portfolio project; the model card will state it.
