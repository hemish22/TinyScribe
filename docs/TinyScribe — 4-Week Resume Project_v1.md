# TinyScribe — A 26B VLM on a 16 GB MacBook

Oct 15, 2026 · @Jhamura

## Goal

Take Gemma 4 26B A4B — a vision-language model that needs a datacenter GPU at full precision — quantize it on the DGX, and run it offline on your 16 GB MacBook Pro M4 to read Indian forms into JSON. The project proves one sentence with numbers: **an A100-class VLM runs on a 16 GB laptop and keeps most of its accuracy.**

- **Why this model.** It is a mixture-of-experts model: [Google notes](https://ai.google.dev/gemma/docs/core) it activates only 4 billion parameters per token but must load all 26 billion. So speed is laptop-friendly; memory is the only problem, and quantization solves memory.
- **The honest catch.** Google lists it at 57.7 GB in BF16 and 14.4 GB even at Q4\_0. On 16 GB it fits only at about 3 bits. How much accuracy survives at 3 bits is the result.
- **Scripts:** English, Hindi, Tamil and Bengali.
- **Fully digital.** Synthetic test forms, no volunteers, no approvals. Fine-tuning is optional.

## Pipeline

The DGX does everything that needs memory; the Mac only runs the files that fit and gets timed.

1. **Generate** synthetic test and calibration forms with exact JSON labels (`datagen/`).
2. **DGX — reference.** Convert the model to GGUF and score it at BF16. This is the number every quant is compared with.
3. **DGX — quantize.** Build importance matrices, produce the quant grid, and score every quant's accuracy there (same GGUF files, so the accuracy carries over to the Mac).
4. **Mac — deploy.** Run the quants that fit in 16 GB with llama.cpp, with grammar-constrained JSON output; measure seconds per page and memory.
5. **Compare** against a small model that fits easily (Gemma 4 E4B at 8 bits) at the same memory budget.

## Memory budget on a 16 GB Mac

macOS lets the GPU use only part of unified memory, so the real budget is about 12–13.5 GB, not 16. Metal caps the working set at roughly 75% of RAM by default ([dev note](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)); `sudo sysctl iogpu.wired_limit_mb=13500` raises it until the next reboot, and `=0` resets it.

| Quant | Bits per weight | Model file (estimate) | Fits with the vision projector and cache? |
| --- | --- | --- | --- |
| BF16 | 16 | \~52 GB | No — DGX only |
| Q4\_K\_M | \~4.8 | \~15.5 GB | No — DGX only |
| Google QAT Q4\_0 | \~4.5 | \~14.5 GB | No — DGX only |
| IQ3\_S | 3.44 | \~11 GB | Only with the raised cap, and tight |
| IQ3\_XXS | 3.06 | \~10 GB | Yes, with the raised cap |
| IQ2\_M | 2.7 | \~9 GB | Yes |

All sizes are estimates from bits × 26B parameters; the vision projector, KV cache and buffers add roughly 1.5 GB more. Confirm each with `llama-quantize --dry-run`, which reports the final size without writing a file ([man page](https://dyn.manpages.debian.org/testing/llama.cpp-tools/llama-quantize.1.en.html)).

## Data

Without fine-tuning you need far less data: a test set to score models and a calibration set to build the importance matrix. The rule that matters is that the two never share templates or handwriting.

| Piece | Plan |
| --- | --- |
| Templates | 12 blank bank or India Post forms, 3 per script: 2 for calibration, 1 for testing. Bank form centres such as [UCO Bank's](https://uco.bank.in/web/guest/download-application-forms) publish forms in Tamil, Bengali and other scripts. Annotate field boxes once. |
| Handwriting | Word crops from [IIIT-INDIC-HW-WORDS](https://ilocr.iiit.ac.in/dataset/22/) (CC BY 4.0, free account) for Hindi, Tamil and Bengali; handwriting-style fonts for English. Train crops go into calibration pages, test crops into test pages. |
| Calibration set | \~200 pages per script (\~800 total) on the calibration templates. |
| Test set | \~100 pages per script (\~400 total) on the held-out templates, with exact JSON labels. |
| Optional realism check | Print \~50 test pages and photograph them on your phone; fill \~15 forms yourself in Hindi and English. |

Leave ID-number fields blank or write `TEST-0000`, so the data contains no realistic-looking ID numbers.

## Models

One big model squeezed hard, one official quantized reference, and one small model at the same memory budget.

| Model | Role | Where it runs |
| --- | --- | --- |
| Gemma 4 26B A4B, BF16 | Accuracy reference | DGX only |
| Gemma 4 26B A4B, Google's official QAT Q4\_0 | How good a 4-bit can get when quantization is trained in | DGX only — too big for 16 GB |
| Gemma 4 26B A4B, your quants | The project | DGX for accuracy, Mac for the ones that fit |
| Gemma 4 E4B, Q8\_0 | Small-model comparison at \~9 GB | Mac |

- **Licence.** Gemma 4 is Apache 2.0, so you can publish your quantized files on Hugging Face.
- **Optional fine-tune.** A LoRA pass on synthetic forms before quantizing, on the DGX. It needs the full BF16 model in memory (\~58 GB), so do it only if an 80 GB A100 is free and time allows.
- **Fallback.** If Gemma 4's vision projector misbehaves in llama.cpp, repeat the same steps with another MoE VLM such as Qwen3-VL-30B-A3B; it is larger, so expect only 2-bit to fit.

## Quantize and deploy

At 2–3 bits, calibration decides quality, so the one experiment beyond the grid is whether calibrating on your own Indic forms beats generic calibration.

**On the DGX**

1. Convert the model to a BF16 GGUF with llama.cpp's converter; export the vision projector (`mmproj`) separately.
2. Build two importance matrices with `llama-imatrix`: one from generic text, one from your calibration forms' labels and JSON outputs in all four scripts.
3. Quantize to IQ2\_M, IQ3\_XXS, IQ3\_S, Q3\_K\_M and Q4\_K\_M, each with both importance matrices. Download Google's QAT Q4\_0 alongside.
4. Score every file on the test set with llama.cpp on GPU. Accuracy depends on the weights, not the machine, so these numbers hold on the Mac.

**On the Mac**

1. Install llama.cpp (Metal is on by default on Apple Silicon).
2. Raise the GPU memory cap: `sudo sysctl iogpu.wired_limit_mb=13500`. Close other apps and watch Memory Pressure in Activity Monitor.
3. Run the quants that fit with everything on the GPU (`-ngl 99`), then once CPU-only (`-ngl 0`) to get a CPU number too.
4. Turn each form's JSON Schema into a llama.cpp grammar, so output is always valid JSON with the right keys.
5. Ship a CLI (`tinyscribe parse form.jpg --schema deposit_slip`) and a small local web demo for the README GIF.

## Evaluation

One table carries the whole project; fill it as results arrive and paste it into the README. Accuracy columns come from the DGX; speed and memory only from the Mac.

- **Field F1 (main metric):** exact match per field after normalising whitespace, numerals and date formats.
- **Per-script F1:** whether 3-bit hurts Tamil or Bengali more than English. Report it either way.
- **Valid JSON:** share of outputs that parse against the schema; should be 100% with the grammar on.
- **Speed and memory:** seconds per page with Metal and CPU-only, and peak memory.

| Model | Field F1 | English | Hindi | Tamil | Bengali | Valid JSON | s / page (Metal) | s / page (CPU) | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 26B, BF16 (DGX) |  |  |  |  |  |  |  |  |  |
| 26B, Google QAT Q4\_0 (DGX) |  |  |  |  |  |  |  |  |  |
| 26B, Q4\_K\_M (DGX) |  |  |  |  |  |  |  |  |  |
| 26B, IQ3\_S + form imatrix |  |  |  |  |  |  |  |  |  |
| 26B, IQ3\_XXS + generic imatrix |  |  |  |  |  |  |  |  |  |
| 26B, IQ3\_XXS + form imatrix |  |  |  |  |  |  |  |  |  |
| 26B, IQ2\_M + form imatrix |  |  |  |  |  |  |  |  |  |
| E4B, Q8\_0 |  |  |  |  |  |  |  |  |  |

## Week-by-week

Four weeks at roughly 10–12 hours a week; tick items off as you go.

### Week 1 — data and reference

- [ ] Download 12 templates (3 per script), annotate field boxes, write one JSON Schema per form type
- [ ] Generate \~800 calibration and \~400 test pages; check 50 by eye
- [ ] On the DGX: download Gemma 4 26B A4B, convert to BF16 GGUF, score the BF16 reference

### Week 2 — quantize on the DGX

- [ ] Build the generic and form-calibrated importance matrices
- [ ] Produce the quant grid and run `--dry-run` sizes for each
- [ ] Score every quant, Google's QAT Q4\_0 and E4B Q8\_0 on the test set; fill the accuracy columns

### Week 3 — run it on the Mac

- [ ] Install llama.cpp, raise the GPU memory cap, load the quants that fit
- [ ] Measure seconds per page (Metal and CPU-only) and peak memory
- [ ] Add grammar-constrained JSON; fill the speed and memory columns

### Week 4 — ship

- [ ] Optional: print and photograph \~50 test pages; fill \~15 forms yourself
- [ ] CLI and local web demo; record the README GIF
- [ ] README with the results table; upload your best quants to Hugging Face with a model card

## Deliverables and resume bullets

The repo is the deliverable; the resume bullets are filled from its results table, never written ahead of the numbers.

**Repo checklist**

- [ ] README that opens with the results table and the demo GIF
- [ ] `make data`, `make quantize`, `make eval`, `make bench` — each runs one step
- [ ] Your best quants and the vision projector on Hugging Face with a short model card
- [ ] `LIMITATIONS.md`: synthetic test forms, 4 scripts, one MacBook for timing

**Resume bullets (placeholders in brackets):**

- Compressed Gemma 4 26B-A4B, a vision-language model needing \~58 GB at full precision, to \[S\] GB and ran it offline on a 16 GB MacBook Pro M4 with llama.cpp, keeping \[X\]% of BF16 field-F1 on Indic form extraction at \[Y\] s/page.
- Built a form-specific importance matrix from synthetic Indic forms that recovered \[Z\] F1 points over generic calibration at 3-bit; \[worst script\] lost \[A\] points against \[B\] for English.
- Showed that at the same \~10 GB budget, a 3-bit 26B MoE model \[beats / trails\] an 8-bit 4B model by \[C\] F1 points.

## Sources

- [Gemma 4 model overview — Google AI for Developers](https://ai.google.dev/gemma/docs/core)
- [Raising the macOS Metal memory cap — dev note](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)
- [llama-quantize man page — Debian](https://dyn.manpages.debian.org/testing/llama.cpp-tools/llama-quantize.1.en.html)
- [UCO Bank forms download centre](https://uco.bank.in/web/guest/download-application-forms)
- [IIIT-INDIC-HW-WORDS (Tamil) — ILOCR dataset page](https://ilocr.iiit.ac.in/dataset/22/)
