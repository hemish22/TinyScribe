#!/usr/bin/env python3
"""Write an HTML contact sheet of random pages next to their target JSON, for checking by eye
(shaping of Bengali/Tamil/Devanagari, comb boxes, checkbox marks, nothing cut off).

    python datagen/qa_sheet.py --n 50            # -> data/qa.html, open in a browser
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    manifest = Path(a.manifest)
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    rng = random.Random(a.seed)
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["script"], []).append(r)
    pick = [r for rs in by.values() for r in rng.sample(rs, min(len(rs), a.n // len(by)))]
    parts = ["<!doctype html><meta charset=utf-8><title>TinyScribe QA</title><style>"
             "body{font-family:system-ui,'Noto Sans';margin:16px}section{display:flex;gap:16px;"
             "border-top:1px solid #ccc;padding:12px 0}img{width:640px;border:1px solid #999}"
             "pre{white-space:pre-wrap;font-size:13px;max-width:640px}</style>"]
    for r in pick:
        meta = (f"{r['id']} · {r['script']} · {r['split']} · {r['degradation']}"
                f"{' · handwritten' if r['handwriting'] else ''}")
        parts.append(f"<section><img src='{html.escape(r['image'])}' loading=lazy><div><b>{html.escape(meta)}"
                     f"</b><pre>{html.escape(json.dumps(r['target'], ensure_ascii=False, indent=1))}</pre>"
                     "</div></section>")
    out = manifest.parent / "qa.html"
    out.write_text("\n".join(parts))
    print(f"{len(pick)} pages -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
