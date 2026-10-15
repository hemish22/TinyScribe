# Week 3 — quantize on the DGX, time on the Mac

Last updated: Oct 16, 2026 · not started

## 1. Plan

- `quant/calib_text.py`: form-imatrix text from the 800 calibration pages (label text + the fine-tuned model's JSON, chat-templated). Generic imatrix from general text.
- `quant/imatrix.sh`: build both importance matrices with `llama-imatrix`.
- `quant/sizes.sh`: check sizes with `llama-quantize --dry-run`. Then `quant/quantize.sh` builds the grid: Q4_K_M, Q3_K_S, IQ3_XXS, IQ2_M and IQ2_XXS, each with the generic and the form imatrix.
- Score every quant and the 2B Q8_0 on the DGX (4 servers).
- Mac: build llama.cpp at the pinned commit (Metal). `bench/bench_mac.py` measures s/page with Metal (`-ngl 99`) and CPU (`-ngl 0`) and peak memory, for each quant ≤ 4 GB plus the Q4_K_M anchor.
- Optional: simulate an 8 GB Mac with `sudo sysctl iogpu.wired_limit_mb=5300`.

## 2. In progress

## 3. Done

## 4. Needs your input

- Free 10–15 GB of Mac disk before the benchmark.
