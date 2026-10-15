#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/hw_bank.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Build the handwriting glyph/word banks used by handwriting.py.

    python src/generate/hw_bank.py emnist            # EMNIST byclass -> per-character glyph bank
    python src/generate/hw_bank.py iiit-vocab        # IIIT-HW-Dev (Hindi) -> vocab.tsv (word, count)
    python src/generate/hw_bank.py iiit-extract      # extract word images for the words value_gen needs

Sources (both research-use; see docs/scope_v3.md):
  EMNIST (NIST SD19 derived), tools/hw_datasets/emnist/gzip.zip
  IIIT-INDIC-HW-WORDS Hindi (CVIT, IIIT Hyderabad), HF c3rl/IIIT-INDIC-HW-WORDS-Hindi parquet
Banks are stored ink-dark-on-white PNGs: bank/<ord(char)>/<n>.png, bank/<word>/<n>.png.
"""
import gzip
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "tools" / "hw_datasets"
EMNIST_BANK = HW / "emnist" / "bank"
IIIT_DIR = HW / "iiit_hw_hindi"
IIIT_BANK = IIIT_DIR / "bank"
BYCLASS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _idx(buf: bytes) -> np.ndarray:
    ndim = buf[3]
    dims = [int.from_bytes(buf[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def build_emnist(per_class: int = 300) -> None:
    with zipfile.ZipFile(HW / "emnist" / "gzip.zip") as z:
        imgs = _idx(gzip.decompress(z.read("gzip/emnist-byclass-train-images-idx3-ubyte.gz")))
        labels = _idx(gzip.decompress(z.read("gzip/emnist-byclass-train-labels-idx1-ubyte.gz")))
    counts = Counter()
    for img, lab in zip(imgs, labels):
        ch = BYCLASS[lab]
        if counts[ch] >= per_class:
            continue
        a = 255 - img.T                      # EMNIST is stored transposed, white ink on black
        d = EMNIST_BANK / str(ord(ch))
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a).save(d / f"{counts[ch]}.png")
        counts[ch] += 1
        if len(counts) == len(BYCLASS) and min(counts.values()) >= per_class:
            break
    print(f"emnist bank: {len(counts)} classes, min {min(counts.values())} per class -> {EMNIST_BANK}")


def _iiit_rows():
    import pyarrow.parquet as pq
    for f in sorted((IIIT_DIR / "data").glob("*.parquet")):
        t = pq.read_table(f, columns=["text", "image"])
        texts = t.column("text").to_pylist()
        images = t.column("image").to_pylist()
        for txt, im in zip(texts, images):
            yield txt, im["bytes"]


def iiit_vocab() -> None:
    c = Counter(txt for txt, _ in _iiit_rows())
    out = IIIT_DIR / "vocab.tsv"
    out.write_text("".join(f"{w}\t{n}\n" for w, n in c.most_common()), encoding="utf-8")
    print(f"iiit vocab: {len(c)} distinct words, {sum(c.values())} images -> {out}")


def iiit_extract(words: set, per_word: int = 25) -> None:
    counts = Counter()
    for txt, b in _iiit_rows():
        if txt not in words or counts[txt] >= per_word:
            continue
        d = IIIT_BANK / txt
        d.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(b)).convert("L").save(d / f"{counts[txt]}.png")
        counts[txt] += 1
    missing = sorted(words - set(counts))
    print(f"iiit bank: {len(counts)}/{len(words)} words extracted -> {IIIT_BANK}")
    if missing:
        print("missing:", " ".join(missing))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "emnist":
        build_emnist()
    elif cmd == "iiit-vocab":
        iiit_vocab()
    elif cmd == "iiit-extract":
        from generate.value_gen import hw_hindi_words
        iiit_extract(hw_hindi_words())
    else:
        sys.exit(__doc__)
