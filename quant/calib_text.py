#!/usr/bin/env python3
"""Calibration text for llama-imatrix (text only; the tool cannot take images).

    python quant/calib_text.py form    --out runs/quant/calib_form.txt
    python quant/calib_text.py generic --out runs/quant/calib_generic.txt

form:    the 800 calibration pages (manifest rows with calib=true, train split, 200 per script).
         Each page is the exact chat-templated sequence the fine-tuned model sees and writes:
         user turn with the image placeholder and the page's prompt (field labels in its script),
         then the thinking-off assistant turn with the page's compact JSON. Run llama-imatrix with
         --parse-special so the chat tokens are read as special tokens. The JSON is the page's
         ground truth, which is exactly the format the fine-tuned model was trained to write.
generic: English Wikipedia text (WikiText-2 raw, test split), the usual generic calibration source.
         Downloaded from the Hugging Face mirror Salesforce/wikitext.
Both files get the same --chunks budget in quant/imatrix.sh (250 x 1024 tokens; WikiText-2 test
has about 280k), so the two matrices see the same number of tokens.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinyscribe.prompt import build_prompt  # noqa: E402
from tinyscribe.schema import page_layout  # noqa: E402

# Qwen3.5 chat template output with one image and enable_thinking=False (checked against
# AutoProcessor.apply_chat_template on the DGX, transformers 5.17)
USER = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n{json}<|im_end|>\n"
WIKITEXT = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")


def form_text(manifest: Path) -> str:
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    calib = [r for r in rows if r.get("calib")]
    if not calib:
        raise SystemExit(f"no calib rows in {manifest}")
    random.Random(0).shuffle(calib)   # any prefix (imatrix --chunks) then mixes all four scripts
    parts = []
    for r in calib:
        prompt = build_prompt(page_layout(r["template_id"], r["page"]))
        target = json.dumps(r["target"], ensure_ascii=False, separators=(",", ":"))
        parts.append(USER.format(prompt=prompt) + ASSISTANT.format(json=target))
    print(f"form calibration: {len(calib)} pages", file=sys.stderr)
    return "".join(parts)


def generic_text() -> str:
    import io
    import urllib.request

    import pyarrow.parquet as pq
    data = urllib.request.urlopen(WIKITEXT, timeout=120).read()
    lines = pq.read_table(io.BytesIO(data), columns=["text"]).column("text").to_pylist()
    return "".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["form", "generic"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.jsonl"))
    a = ap.parse_args(argv)
    text = form_text(Path(a.manifest)) if a.kind == "form" else generic_text()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"{a.kind}: {len(text):,} characters -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
