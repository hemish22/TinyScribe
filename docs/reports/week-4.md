# Week 4 — ship

Last updated: Oct 27, 2026 · CLI, web demo, README, LIMITATIONS.md done; README GIF in progress; Hugging Face upload on hold (your call)

## 1. Plan

- `tinyscribe parse form.jpg --schema eci_form7_hi_p0` CLI (uses `tinyscribe.client`).
- `app/`: local web demo (upload a page, pick a schema, see the JSON). Record the README GIF.
- README opens with the results table (`make table`) and the GIF. Write `LIMITATIONS.md`.
- Upload the best quant ≤ 4 GB and its projector to Hugging Face with a model card.
- Optional: print and photograph ~50 test pages; fill ~15 forms by hand.
- Stretch: image-aware importance matrix (Transformers forward hooks → GGUF imatrix).

## 2. In progress

- README GIF: `scripts/record_demo.py` drives the demo in headless Chromium (Playwright) and turns the recording into `docs/demo.gif` with ffmpeg. Waiting on the Playwright download (slow network).

## 3. Done

### Zero-shot baseline corrected (Oct 27)

- A side check flagged that the 9B zero-shot run had valid JSON on only 80% of pages. Cause: one of its four llama-server processes stopped mid-run; 79 pages got "connection refused" (77), a dropped connection or a timeout, and were scored as empty answers.
- Re-ran the 79 pages: **9B zero-shot is 81.2, not 69.0** (valid JSON 100%). Fine-tuning gain: +7.4, not +19.6. All other runs had 0 failed requests.
- `eval/run_eval.py` retries failed requests on resume and exits with an error while any remain; `eval/results_table.py` marks such rows; `eval/breakdown.py` keeps the last record per page. Week 3 report updated.

### CLI (`src/tinyscribe/cli.py`)

```
tinyscribe parse form.jpg --schema eci_form7_hi_p0 [--model 9b|2b|path] [--labels] [-o out.json]
tinyscribe schemas [--language tamil]
```

- Starts llama-server with the benchmark's settings (`src/tinyscribe/server.py`: Metal, `-c 8192`, `--cache-ram 0`), reads each image, stops the server; `--server URL` uses a running one. `--cpu` for CPU only.
- `src/tinyscribe/image.py`: any photo or scan is EXIF-rotated and resized to the 1536 px long side the model trained on; dataset pages pass through byte-identical.
- Unknown schema → suggestions (`eci_form7_hi_p9` → `eci_form7_hi_p0, …`). Exit code 1 if an output does not parse.
- Checked on the Mac: Tamil Form 7 page with the 2B, 13.5 s including model load; Hindi Form 7 page with the 9B, 36 s.

### Web demo (`app/server.py`, `app/index.html`)

- `make demo` (9B) or `MODEL_NAME=2b make demo` → http://127.0.0.1:8000. Python standard library HTTP server + one HTML page, no extra dependencies, nothing leaves the machine.
- Drop or choose an image, pick the form and page (grouped by language), "Read form". Shows a live timer, then each printed label with its value; JSON tab.
- 8 sample buttons (one per script × printed/handwritten, held-out test pages): ✓/✗ against the ground truth, with the true value under each miss. A Bengali handwritten Form 7 sample: 33 of 37 items correct, 31 s.
- Stops its llama-server on Ctrl+C or `kill`.

### README, LIMITATIONS.md, Makefile

- README: pitch, GIF slot, results table (key rows), findings, install, CLI and demo usage with real output, how it was built, one `make` target per step.
- `LIMITATIONS.md`: synthetic test pages, held out by form type, four scripts, 401 pages; Bengali/Tamil handwriting partly a font; text-only imatrix; Q8_0 projector; thinking off; exact-match F1; one Mac, not idle, peak-memory definition; 8 GB untested; non-commercial licences.
- Makefile: `make train` now waits for a GPU within the account cap (was hard-coded to GPU 4); new `pipeline`, `quantize`, `bench`, `parse`, `demo` targets.
- Tests: `tests/test_cli.py` (image prep, EXIF rotation, schema list, label pairs, demo ✓/✗ with Indic digits). 40 passed, 4 skipped (datagen deps not installed on the Mac).

## 4. Needs your input

- **Hugging Face upload:** on hold, as you asked.
- **Licence file:** `pyproject.toml` declares Apache-2.0 for the code but the repo has no `LICENSE` file. Add the standard Apache-2.0 text?
- **Optional items:** photographing ~50 printed test pages (needs you, a printer and a phone) and the image-aware imatrix stretch (needs DGX GPUs for a few hours). Say if you want either.
