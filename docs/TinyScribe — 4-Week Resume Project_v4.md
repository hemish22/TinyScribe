# TinyScribe — A 9B VLM in 4 GB on a MacBook

> **v4 change:** "4 GB" in the title means the language-model file. The full package, with the Q8\_0 vision projector, is about 4.5 GB. The title stays short; the Goal states the exact claim.
> **Reason:** the headline claim must match what ships. A reviewer who adds the vision projector to the model file will otherwise find 4.5 GB, not 4.

Oct 15, 2026 · @Jhamura

Revised Oct 16, 2026 (v4). Everything from v3 is kept word for word; each v4 change sits directly under the text it changes, as a quoted note with its reason.

**What changed in v4**

1. Headline claim: "4 GB" now names the language-model file, with the ~4.5 GB package stated (title and Goal).
2. Motivation: the 4 GB target is justified by 8 GB devices, not by "needs a datacenter GPU" (Goal).
3. Vision projector at Q8\_0 is untested; test it in Week 2, with an F16 fallback (The 4 GB budget).
4. Every accuracy number, including the BF16 reference, is scored through llama.cpp (Models).
5. New repo layout section (Repo layout).
6. Q4\_K\_M also runs on the Mac as a speed anchor (On the Mac).
7. Results table gains a Q3\_K\_S row and the image-aware stretch row (Evaluation).
8. Grammar work and the projector test move to Week 2 to lighten Week 3 (Week-by-week).
9. Repo checklist notes that README and `.gitignore` now exist (Deliverables).

## Goal

Fine-tune Qwen3.5-9B on the DGX to read Indian forms into JSON, then quantize it until the model file is **4 GB or less**, and run it offline on your MacBook Pro M4. The project proves one sentence with numbers: **an 18 GB vision-language model fits in 4 GB and still reads forms well.**

> **v4 change:** the sentence now reads: **an 18 GB vision-language model fits in a 4 GB language-model file (about 4.5 GB with its vision projector) and still reads forms well.**
> **Reason:** the vision projector is a separate file of about 0.5 GB at Q8\_0 (0.9 GB at F16) that must ship with the model. Naming the language-model file and stating the package size keeps the claim exact.

- **Why Qwen3.5-9B.** At BF16 its GGUF is 17.9 GB, so it needs a datacenter GPU; even at Q4\_K\_M it is 5.68 GB. Getting the model under 4 GB takes 3 bits per weight or fewer — hard enough to be interesting, small enough to run comfortably on 16 GB.

  > **v4 change:** the reason for 4 GB is **8 GB devices**, not datacenter hardware. At 17.9 GB the BF16 model already runs on a 24 GB consumer GPU, and Q8\_0 (9.53 GB) already runs on the 16 GB Mac. What it cannot run on is an 8 GB MacBook Air or Mac mini, or a phone, where the GPU gets roughly 5.3 GB. A 4 GB language model plus its projector fits that budget. The DGX is needed for fine-tuning, not for inference. Optional check: imitate an 8 GB Mac on your 16 GB one with `sudo sysctl iogpu.wired_limit_mb=5300` and run the best quant (reset with `=0`).
  > **Reason:** "needs a datacenter GPU" is untrue for inference and easy for a reviewer to challenge. 8 GB devices give the 4 GB target a real reason to exist.
- **The comparison that makes it interesting:** Qwen3.5-2B at 8 bits also fits in 4 GB. Which reads forms better in the same budget — a big model squeezed hard or a small one kept precise?
- **Scripts:** English, Hindi, Tamil and Bengali.
- **Fully digital.** Synthetic forms, no volunteers, no approvals.

## Pipeline

The DGX does the training and the heavy quantization; the Mac only runs the files that make the 4 GB cut and gets timed.

1. **Generate** synthetic filled forms with exact JSON labels (`datagen/`).
2. **DGX — fine-tune** Qwen3.5-9B and Qwen3.5-2B with LoRA on the training pages (`train/`).
3. **DGX — quantize.** Merge, convert to GGUF, build an importance matrix from your own form pages, and produce every quant (`quant/`).
4. **DGX — score** every quant's accuracy on the test set. Accuracy depends on the weights, not the machine, so it carries over to the Mac.
5. **Mac — deploy** the quants that fit in 4 GB with llama.cpp and grammar-constrained JSON; measure speed and memory (`app/`, `bench/`).

## Repo layout

> **v4 change:** new section. The repo is already set up at [github.com/hemish22/TinyScribe](https://github.com/hemish22/TinyScribe); each folder below maps to one pipeline step and one `make` target.
>
> ```
> tinyscribe/
>   datagen/   templates, field boxes, page generator               → make data
>   schemas/   one JSON Schema per form type, plus generated grammars
>   train/     LoRA configs and scripts (DGX)                       → make train
>   quant/     convert, vision projector, imatrix, quantize, sizes  → make quantize
>   eval/      field-F1 scorer, normaliser, results-table writer    → make eval
>   bench/     Mac Metal / CPU timing and peak memory               → make bench
>   app/       CLI (tinyscribe parse) and local web demo
>   docs/      plan versions
>   data/ models/ runs/   large files, kept out of git
> ```
>
> **Reason:** v3 named some folders in the Pipeline but had none for scoring or schemas, so `make eval` had no home and the schemas used by datagen, eval and the grammar had no shared place.

## The 4 GB budget

"4 GB" means the language-model file. The vision projector is a separate 918 MB file at F16; quantize it to Q8\_0 (about half) and count it alongside, so the whole package stays near 4.5 GB.

> **v4 change:** the Q8\_0 vision projector is untested. Quantizing the projector is less well supported in llama.cpp than quantizing the language model, and it can hurt image reading. In Week 2, score the fine-tuned model with the F16 and the Q8\_0 projector on ~50 test pages. If Q8\_0 loses accuracy, ship F16 and state the package as about 4.9 GB in the README.
> **Reason:** the 4.5 GB package size depends on this step, so check it before the quant grid is built on top of it.

| Quant of Qwen3.5-9B | Model file | Within 4 GB? |
| --- | --- | --- |
| BF16 | 17.9 GB | No — DGX reference |
| Q8\_0 | 9.53 GB | No — DGX reference |
| Q4\_K\_M | 5.68 GB | No — DGX reference |
| Q3\_K\_S | 4.32 GB | No, just over |
| IQ3\_XXS | \~3.8–4.0 GB | Borderline — confirm with `--dry-run` |
| IQ2\_M | \~3.6 GB | Yes |
| IQ2\_XXS | \~3.2 GB | Yes |
| Qwen3.5-2B, Q8\_0 | \~2.5 GB (estimate) | Yes — the small-model comparison |

The 9B sizes come from [Unsloth's Qwen3.5-9B GGUF files](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main); their IQ3\_XXS and IQ2 files use a dynamic layout, so your own quants may differ slightly. Confirm each with `llama-quantize --dry-run`, which reports the final size without writing a file ([man page](https://dyn.manpages.debian.org/testing/llama.cpp-tools/llama-quantize.1.en.html)).

At 2–3 bits quality can fall off a cliff, so the real result is which quant is the smallest that still reads forms well. At runtime the package needs roughly 5–6 GB with cache and buffers, comfortably inside the \~75% of unified memory macOS gives the GPU by default ([dev note](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)). No system tweaks needed.

## Data

Everything is generated, so labels are exact and nothing needs collecting. The rule that matters: test forms use templates and handwriting the model never saw in training or calibration.

| Piece | Plan |
| --- | --- |
| Templates | 12 blank bank or India Post forms, 3 per script: 2 for training, 1 held out for testing. Bank form centres such as [UCO Bank's](https://uco.bank.in/web/guest/download-application-forms) publish forms in Tamil, Bengali and other scripts. Annotate field boxes once. |
| Handwriting | Word crops from [IIIT-INDIC-HW-WORDS](https://ilocr.iiit.ac.in/dataset/22/) (CC BY 4.0, free account) for Hindi, Tamil and Bengali; handwriting-style fonts for English. Train crops go into training pages, test crops only into test pages. |
| Training set | \~10k pages (\~2.5k per script) with random names, addresses, dates, amounts and ticked boxes, plus scan noise, blur and slight rotation. |
| Calibration set | \~800 of the training pages, used to build the importance matrix. |
| Test set | \~400 pages (\~100 per script) on the held-out templates. |
| Optional realism check | Print \~50 test pages and photograph them on your phone; fill \~15 forms yourself in Hindi and English. |

Leave ID-number fields blank or write `TEST-0000`, so the data contains no realistic-looking ID numbers.

## Models

One model trained on the DGX and squeezed hard, plus one small model kept precise inside the same 4 GB.

| Model | Role | Where it runs |
| --- | --- | --- |
| Qwen3.5-9B, zero-shot | Baseline: what fine-tuning adds | DGX |
| Qwen3.5-9B, fine-tuned, BF16 | Accuracy reference for every quant | DGX only |
| Qwen3.5-9B, fine-tuned, your ≤ 4 GB quants | The project | DGX for accuracy, Mac for speed |
| Qwen3.5-2B, fine-tuned, Q8\_0 | Small model in the same budget | Mac |

- **Fine-tuning.** LoRA on the language model, vision encoder frozen, compact JSON keys. A 9B model in BF16 is \~18 GB of weights, so it trains comfortably on one 80 GB A100; on a 40 GB card use QLoRA.
- **Compute.** The DGX's default 2-GPU grant is enough. Save checkpoints often.
- **Licence.** Qwen3.5 is Apache 2.0, so you can publish your quantized files on Hugging Face.

> **v4 change:** score every accuracy number through llama.cpp, the reference included. That means the zero-shot baseline, the fine-tuned BF16 reference, every quant and the 2B Q8\_0 all run as GGUF files with the same vision projector, image preprocessing, prompt and grammar. Do not score the reference in Transformers or vLLM. The 2B Q8\_0 accuracy also comes from the DGX; the Mac only times it.
> **Reason:** "keeps X% of full-precision accuracy" is only fair if the reference and the quants run through the same engine. Different image resizing or prompt handling would otherwise show up as quantization loss.

## Quantize and deploy

At 2–3 bits the importance matrix decides quality, so the one experiment beyond the size grid is whether calibrating on your own Indic forms beats generic calibration.

**On the DGX**

1. Merge the LoRA weights and convert to a BF16 GGUF with llama.cpp's converter. Export the vision projector and quantize it to Q8\_0.
2. Build two importance matrices with `llama-imatrix`: one from generic text, one from your calibration forms — their label text plus the fine-tuned model's own JSON outputs in all four scripts, wrapped in its chat template (add `--parse-special`). The tool reads only text, so neither matrix sees image tokens. The form matrix still matches what the decoder actually writes, but expect a smaller gap than true image-aware calibration would give.
3. Run `llama-quantize --dry-run` to check sizes, then quantize to Q4\_K\_M (reference), Q3\_K\_S, IQ3\_XXS, IQ2\_M and IQ2\_XXS, each with both importance matrices.

   > **v4 change:** Q3\_K\_S now has its own row in the results table (see Evaluation).
   > **Reason:** it was in the quant grid and the size table but had nowhere to report its score.
4. Score every file on the test set with llama.cpp on GPU. Accuracy depends on the weights, not the machine, so these numbers hold on the Mac.

**Stretch — an image-aware importance matrix.** `llama-imatrix` accepts only a text file of calibration data ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)). To include image tokens, run the fine-tuned model in Transformers on the calibration form images, record the sum of squared inputs to every linear layer of the language model with forward hooks, and write the totals out in llama.cpp's GGUF imatrix layout, mapping Transformers layer names to GGUF tensor names. Load the result with `llama-imatrix --in-file … --show-statistics` to confirm it parses, then add a third row per quant to the results table. If it works, it is the strongest line on the resume; if it slips, the text-only comparison still stands.

**On the Mac**

1. Install llama.cpp (Metal is on by default on Apple Silicon).
2. Run every quant that is 4 GB or less with everything on the GPU (`-ngl 99`), then once CPU-only (`-ngl 0`) for a CPU number too.

   > **v4 change:** also run Q4\_K\_M (5.68 GB) on the Mac, for speed and memory only. It fits on 16 GB without any system change.
   > **Reason:** it gives the speed numbers an anchor. Without a normal 4-bit to compare with, the table cannot show what going down to 2–3 bits buys in speed and memory.
3. Turn each form's JSON Schema into a llama.cpp grammar, so output is always valid JSON with the right keys.
4. Ship a CLI (`tinyscribe parse form.jpg --schema deposit_slip`) and a small local web demo for the README GIF.

## Evaluation

One table carries the whole project; fill it as results arrive and paste it into the README. Accuracy columns come from the DGX; speed and memory only from the Mac.

- **Field F1 (main metric):** exact match per field after normalising whitespace, numerals and date formats.
- **Per-script F1:** whether 2–3-bit hurts Tamil or Bengali more than English. Report it either way.
- **Valid JSON:** share of outputs that parse against the schema; should be 100% with the grammar on.
- **Speed and memory:** seconds per page with Metal and CPU-only, and peak memory.

| Model | Size | Field F1 | English | Hindi | Tamil | Bengali | Valid JSON | s / page (Metal) | s / page (CPU) | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9B zero-shot, BF16 (DGX) |  |  |  |  |  |  |  |  |  |  |
| 9B fine-tuned, BF16 (DGX) |  |  |  |  |  |  |  |  |  |  |
| 9B fine-tuned, Q4\_K\_M (DGX) |  |  |  |  |  |  |  |  |  |  |
| 9B Q3\_K\_S + form imatrix (v4) |  |  |  |  |  |  |  |  |  |  |
| 9B IQ3\_XXS + generic imatrix |  |  |  |  |  |  |  |  |  |  |
| 9B IQ3\_XXS + form imatrix |  |  |  |  |  |  |  |  |  |  |
| 9B IQ3\_XXS + image-aware imatrix (stretch, v4) |  |  |  |  |  |  |  |  |  |  |
| 9B IQ2\_M + form imatrix |  |  |  |  |  |  |  |  |  |  |
| 9B IQ2\_XXS + form imatrix |  |  |  |  |  |  |  |  |  |  |
| 2B fine-tuned, Q8\_0 |  |  |  |  |  |  |  |  |  |  |

> **v4 change:** two rows added, marked (v4): Q3\_K\_S with the form imatrix, and IQ3\_XXS with the image-aware imatrix from the stretch goal. The Q4\_K\_M row now also gets its Metal, CPU and peak-memory columns from the Mac.
> **Reason:** Q3\_K\_S was quantized but had no row. The v3 stretch goal says to add an image-aware row, but the table had none. The Q4\_K\_M speed numbers are the anchor added under "On the Mac".

## Week-by-week

Four weeks at roughly 10–12 hours a week; tick items off as you go.

### Week 1 — data and baseline

- [ ] Download 12 templates (3 per script), annotate field boxes, write one JSON Schema per form type
- [ ] Generate \~10k training and \~400 test pages; check 50 by eye
- [ ] On the DGX: download Qwen3.5-9B and 2B; run the zero-shot baseline

### Week 2 — fine-tune on the DGX

- [ ] LoRA fine-tune the 9B and the 2B; score both in BF16
- [ ] Merge, convert to GGUF, quantize the vision projector to Q8\_0

> **v4 change:** Week 2 gains two items:
>
> - [ ] Turn each JSON Schema into a llama.cpp grammar and switch on grammar-constrained JSON (moved here from Week 3; do it while the LoRA runs train)
> - [ ] Score the fine-tuned 9B with the F16 and the Q8\_0 vision projector on ~50 test pages; keep Q8\_0 only if it holds up
>
> Week 3 keeps the importance matrices, the quant grid, DGX scoring and Mac timing. Its "add grammar-constrained JSON" item is already done by then.
> **Reason:** Week 3 held five heavy tasks. The grammar work does not depend on any quant, so it fits in Week 2's waiting time. The projector test must come before the quant grid because the package size depends on it.

### Week 3 — quantize and run on the Mac

- [ ] Build the generic and form-calibrated importance matrices
- [ ] Produce the quant grid, check sizes with `--dry-run`, score every quant on the DGX
- [ ] On the Mac: run every quant of 4 GB or less, measure speed and memory, add grammar-constrained JSON

### Week 4 — ship

- [ ] Optional: print and photograph \~50 test pages; fill \~15 forms yourself
- [ ] CLI and local web demo; record the README GIF
- [ ] README with the results table; upload your best quant to Hugging Face with a model card

## Deliverables and resume bullets

The repo is the deliverable; the resume bullets are filled from its results table, never written ahead of the numbers.

**Repo checklist**

- [ ] README that opens with the results table and the demo GIF
- [ ] `make data`, `make train`, `make quantize`, `make eval`, `make bench` — each runs one step
- [ ] Your best quant of 4 GB or less and the Q8\_0 vision projector on Hugging Face, with a short model card
- [ ] `LIMITATIONS.md`: synthetic test forms, 4 scripts, one MacBook for timing, and a text-only importance matrix unless the image-aware stretch is done

> **v4 change:** the repo exists with a `README.md` and `.gitignore`. The `.gitignore` keeps `data/`, `models/`, `runs/`, GGUF, safetensors and imatrix files out of git. The README is a stub that points to this plan until results arrive. Add to `LIMITATIONS.md`: Q8\_0 vision projector (or F16 if the Week 2 test fails), and 8 GB devices simulated by a lower memory cap, not tested on real hardware.
> **Reason:** model and data files are gigabytes and must never be committed. The new limitations follow from the v4 changes above.

**Resume bullets (placeholders in brackets):**

- Fine-tuned Qwen3.5-9B on a DGX A100 to extract JSON from Indian bank forms in 4 scripts, raising field-F1 from \[A\]% zero-shot to \[B\]%.
- Quantized it from 17.9 GB to \[S\] GB and ran it offline on a 16 GB MacBook Pro M4 with llama.cpp, keeping \[X\]% of full-precision accuracy at \[Y\] s/page with 100% schema-valid output.
- Showed that a form-calibrated importance matrix recovers \[Z\] F1 points at \[2/3\]-bit, and that inside the same 4 GB a quantized 9B \[beats / trails\] an 8-bit 2B by \[C\] points.

## Sources

- [Unsloth Qwen3.5-9B GGUF files (sizes)](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main)
- [llama-quantize man page — Debian](https://dyn.manpages.debian.org/testing/llama.cpp-tools/llama-quantize.1.en.html)
- [macOS Metal memory cap — dev note](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)
- [UCO Bank forms download centre](https://uco.bank.in/web/guest/download-application-forms)
- [IIIT-INDIC-HW-WORDS (Tamil) — ILOCR dataset page](https://ilocr.iiit.ac.in/dataset/22/)
