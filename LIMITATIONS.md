# Limitations

What the numbers in the README do and do not show.

## Test data

- **Synthetic pages.** Every training and test page is a real government or survey form template (ECI Forms 6/6A/6B/7/8, Atal Pension Yojana, RTI appeals, TN ICDS, WB Bangla Sahayata Kendra, plus synthetic household and job-application forms) filled by a generator, then degraded at four levels (clean, light, medium, heavy: JPEG compression, blur, rotation, brightness gradient, slight perspective warp, speckle). No real filled form or phone photo is in the test set. Real scans have lighting, folds, stamps and handwriting styles the generator does not cover, so expect lower accuracy on them.
- **Held out by form type, not by institution.** The test split is ECI Form 7 and the job application in all four scripts; the model never trained on those layouts. Every other form type is in training, so the score measures new layouts of a familiar kind (Indian government forms), not arbitrary documents.
- **Four scripts.** English, Hindi (Devanagari), Bengali and Tamil only. Other Indian scripts (Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Urdu) were never seen.
- **401 test pages** (about 100 per script, 45–60 per script × handwriting group). Differences under about 1 F1 point between models are within noise.

## Handwriting

- **English and Hindi handwriting is real**: word and character crops from EMNIST and IIIT-INDIC-HW-WORDS (Hindi), pasted into the fields (98–100% of handwritten fields on test pages).
- **Bengali and Tamil handwriting is partly a font.** Real crops come from BanglaWriting (Bengali) and IIIT-INDIC-HW-WORDS (Tamil), but those word lists cover few of the words forms need (names, places, occupations). When a field's words have no real sample, the whole field is drawn in a handwriting-style font (Atma for Bengali, Kavivanar for Tamil) with per-character jitter. On test pages 59% of Bengali and 69% of Tamil handwritten fields use real images (counting EMNIST digits and Latin letters); the share of real Bengali/Tamil *words* is lower. The Tamil font looks close to print. **Bengali and Tamil handwritten scores therefore overstate accuracy on real handwriting.**
- Hindi handwriting, the most realistic group, is the weakest (69 F1 for the full-precision 9B).
- Pasted crops are not continuous handwriting: no cursive joins across words, no strike-outs, no writing over the field lines.

## Models and quantization

- **Text-only importance matrix.** `llama-imatrix` takes text only, so both importance matrices (form text and WikiText) see the chat template and JSON but no image tokens. The form-vs-generic comparison found no real difference (≤ 1 point either way), which may be because neither sees the images.
- **Vision projector at Q8_0.** Tested against F16 on the v1 test set (no loss, Week 2) and used for every quant row.
- **Thinking mode off.** Every model, zero-shot included, answers directly; Qwen3.5's reasoning mode was not tried.
- **Same prompt and grammar for every model.** The zero-shot baselines get the fine-tuned models' prompt and JSON Schema grammar. A prompt tuned for the base model might score higher zero-shot.
- **Exact-match field F1** after normalization (Unicode NFC, Indic digits to ASCII, date formats to ISO, whitespace and edge punctuation). A value with one wrong character counts as wrong; there is no partial credit.

## Speed and memory

- **One machine.** Speed and memory come from one MacBook Pro M4 with 16 GB, llama.cpp at commit `d81235049`, Metal. Other Macs (more GPU cores, more memory bandwidth) will differ.
- **20 pages per model with Metal, 5 on CPU** (after one warm-up page each). Per-page time depends on how much text a page has (about 150–450 output tokens).
- **The Mac was not idle.** Other apps were open and macOS had 5–7 GB in swap during the runs, as on a normal working laptop.
- **Peak memory** = model file + vision projector file + llama-server's peak working memory (macOS `footprint`, with the prompt cache off: `--cache-ram 0`). It is what the process needs to run at full speed with all weights in memory; macOS can run it in less by paging weights from disk, more slowly.
- **8 GB Macs are not tested.** The 9B IQ3_XXS needs about 6.3 GB by the measure above, so it should fit, but this was not run on real 8 GB hardware or with a lowered memory cap.

## Licences

- **Non-commercial.** BanglaWriting is CC BY-NC 3.0 and IIIT-INDIC-HW-WORDS is for research use; models trained on pages built from them inherit those terms. Qwen3.5 is Apache-2.0. Form templates are public government documents (GODL-India where stated) or synthetic.
- Fonts: Noto Sans Devanagari/Bengali/Tamil, Kalam, Atma, Kavivanar, all SIL Open Font License (`assets/fonts/`).
