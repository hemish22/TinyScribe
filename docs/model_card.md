---
license: other
license_name: non-commercial-research
license_link: https://github.com/hemish22/TinyScribe/blob/main/LIMITATIONS.md#licences
base_model:
  - Qwen/Qwen3.5-9B
  - Qwen/Qwen3.5-2B
language:
  - en
  - hi
  - bn
  - ta
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - document-understanding
  - form-extraction
  - ocr
  - handwriting
  - indic
---

# TinyScribe GGUF

Qwen3.5-9B and Qwen3.5-2B, fine-tuned to read filled Indian forms (English, Hindi, Bengali, Tamil; printed or handwritten) into JSON, quantized to run offline on a laptop with llama.cpp.

Code, data generator, training and evaluation: [github.com/hemish22/TinyScribe](https://github.com/hemish22/TinyScribe).

## Files

| File | Size | What |
|---|---|---|
| `TinyScribe-9B-IQ3_XXS.gguf` | 3.89 GB | 9B, IQ3_XXS with q4_K output tensor, form-text importance matrix. **Recommended.** |
| `mmproj-TinyScribe-9B-Q8_0.gguf` | 0.62 GB | 9B vision projector, Q8_0 |
| `TinyScribe-2B-Q8_0.gguf` | 2.03 GB | 2B, Q8_0. Faster, 2.5 points lower. |
| `mmproj-TinyScribe-2B-Q8_0.gguf` | 0.36 GB | 2B vision projector, Q8_0 |

## Results

401 held-out test pages: ECI Form 7 and a job application, form types never seen in training, in all four scripts, half handwritten, four degradation levels. Field F1 is exact match after normalization (Unicode NFC, Indic digits, dates). Speed and memory on a MacBook Pro M4, 16 GB, llama.cpp with Metal.

| Model | Field F1 | English | Hindi | Tamil | Bengali | Valid JSON | s / page (Metal) | s / page (CPU) | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B zero-shot, BF16 | 81.2 | 90.6 | 72.7 | 78.5 | 81.9 | 100 | | | |
| TinyScribe 9B, BF16 | 88.6 | 95.5 | 83.4 | 85.4 | 89.0 | 100 | | | |
| **TinyScribe 9B, IQ3_XXS (this repo)** | **86.5** | 95.1 | 81.6 | 80.6 | 87.0 | 100 | **33.2** | 107.2 | **6.29 GB** |
| Qwen3.5-2B zero-shot, BF16 | 64.9 | 82.2 | 55.5 | 57.2 | 62.5 | 100 | | | |
| **TinyScribe 2B, Q8_0 (this repo)** | **84.0** | 93.5 | 80.3 | 79.2 | 81.3 | 100 | **14.5** | 39.5 | **3.49 GB** |

Peak memory = model + projector + llama-server's peak working memory. Weakest case: Hindi handwriting (67.3 for the 9B IQ3_XXS). Full table and breakdowns in the [GitHub README](https://github.com/hemish22/TinyScribe#results).

## Use

The model answers one page at a time and expects the TinyScribe prompt (the page's field ids with their printed labels) and the page's JSON Schema as a grammar. The CLI does both:

```bash
git clone https://github.com/hemish22/TinyScribe && cd TinyScribe
uv venv --python 3.11 && uv pip install -e . && source .venv/bin/activate
make llama                                   # llama.cpp at the pinned commit
hf download hemish22/TinyScribe-GGUF --local-dir models/TinyScribe-GGUF
tinyscribe parse form.jpg --schema eci_form7_hi_p0            # 9B
tinyscribe parse form.jpg --schema eci_form7_hi_p0 --model 2b
make demo                                    # local web demo on http://127.0.0.1:8000
```

53 form pages are supported (`tinyscribe schemas`): Election Commission of India Forms 6, 6A, 6B, 7, 8; Atal Pension Yojana; RTI appeals (Odisha); TN ICDS Anganwadi; WB Bangla Sahayata Kendra; a household survey and a job application. Input photos or scans are resized to a 1536 px long side, as in training.

With llama-server directly (llama.cpp commit `d81235049`):

```bash
llama-server -m TinyScribe-9B-IQ3_XXS.gguf --mmproj mmproj-TinyScribe-9B-Q8_0.gguf \
  --jinja -c 8192 -ngl 99 --cache-ram 0
```

Then POST to `/v1/chat/completions` with the page image, the prompt from `tinyscribe.prompt.build_prompt`, `response_format` = the page's JSON Schema, `temperature: 0` and `chat_template_kwargs: {"enable_thinking": false}` (see `tinyscribe/client.py`).

## Training

- **Data:** 10,411 synthetic pages from 30 form templates (22 real government forms, 8 synthetic) in four scripts, filled with script-correct values, half with handwriting (EMNIST, IIIT-INDIC-HW-WORDS Hindi and Tamil, BanglaWriting crops; a handwriting font where no crop exists), degraded at four levels.
- **Fine-tune:** Unsloth bf16 LoRA, r = 16, language model attention and MLP, vision encoder frozen, thinking off, one epoch (1,226 steps), one A100 40 GB.
- **Quantize:** merged, converted with llama.cpp, importance matrix from the chat-templated calibration pages (text only), IQ3_XXS with a q4_K output tensor to fit under 4.0 GB. The multi-token-prediction layer, unused by llama.cpp, is stored as q2_K.

## Limitations

- Test pages are synthetic; real scans and phone photos will score lower.
- Bengali and Tamil handwriting in training and test is partly a handwriting font, so those scores overstate accuracy on real handwriting.
- Four scripts and 53 known form pages only; a new form needs its template annotated.
- Speed measured on one MacBook Pro M4.
- Details: [LIMITATIONS.md](https://github.com/hemish22/TinyScribe/blob/main/LIMITATIONS.md).

## Licence

**Non-commercial research use.** The training data includes BanglaWriting (CC BY-NC 3.0) and IIIT-INDIC-HW-WORDS (research use only), so these weights may not be used commercially. Base models: Qwen3.5 (Apache-2.0). Code in the GitHub repo: Apache-2.0.
