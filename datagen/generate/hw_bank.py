#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/hw_bank.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Build the handwriting glyph/word banks used by handwriting.py.

    python src/generate/hw_bank.py emnist            # EMNIST byclass -> per-character glyph bank
    python src/generate/hw_bank.py iiit-vocab        # IIIT-HW-Dev (Hindi) -> vocab.tsv (word, count)
    python src/generate/hw_bank.py iiit-extract      # extract word images for the words value_gen needs
    python datagen/generate/hw_bank.py tamil-vocab   # TinyScribe: IIIT Tamil -> vocab.tsv (word, train, test)
    python datagen/generate/hw_bank.py bengali-vocab # TinyScribe: BanglaWriting -> vocab.tsv
    python datagen/generate/hw_bank.py indic-extract # TinyScribe: Tamil + Bengali banks for value_gen's words

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
# TinyScribe: Bengali and Tamil handwriting. IIIT-INDIC-HW-WORDS Tamil (HF c3rl/IIIT-INDIC-HW-WORDS-Tamil;
# its test split is used for test pages) and BanglaWriting (Mendeley 10.17632/r43wkvdk4w.1, CC BY-NC 3.0;
# one page per writer, writers with id % 5 == 0 are test). Train and test samples are numbered so that
# handwriting._in_split() keeps them apart: test samples get n % 5 == 0, train samples the other n.
TAMIL_DIR = HW / "iiit_hw_tamil"
BENGALI_DIR = HW / "banglawriting"
INDIC_BANKS = {"tamil": TAMIL_DIR / "bank", "bengali": BENGALI_DIR / "bank"}
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


def _sample_no(i: int, split: str) -> int:
    """i-th sample of a word in a split -> bank file number (test: 0, 5, 10...; train: 1-4, 6-9...)."""
    return 5 * i if split == "test" else i + i // 4 + 1


def _tamil_rows(with_images: bool):
    import pyarrow.parquet as pq
    cols = ["text", "image"] if with_images else ["text"]
    for f in sorted((TAMIL_DIR / "data").glob("*.parquet")):
        split = "test" if f.name.startswith("test") else "train"
        t = pq.read_table(f, columns=cols)
        texts = t.column("text").to_pylist()
        images = t.column("image").to_pylist() if with_images else [None] * len(texts)
        for txt, im in zip(texts, images):
            yield split, txt, (im["bytes"] if im else None)


BW_PUNCT = "।,.;:!?'\"()-–—‘’“”"


def _bengali_rows(with_images: bool):
    """(split, word, grayscale crop) for every clean word box of BanglaWriting. Overwritten or
    struck-out words (label with '*') are skipped; punctuation is trimmed from the label."""
    import json
    import unicodedata
    for jf in sorted((BENGALI_DIR / "converted" / "converted").glob("*.json")):
        writer = int(jf.stem.split("_")[0])
        split = "test" if writer % 5 == 0 else "train"
        d = json.loads(jf.read_text(encoding="utf-8"))
        page = Image.open(jf.with_suffix(".jpg")).convert("L") if with_images else None
        for sh in d["shapes"]:
            lab = sh["label"]
            if "*" in lab:
                continue
            word = unicodedata.normalize("NFC", lab.strip().strip(BW_PUNCT))
            if not word or not any("\u0980" <= c <= "\u09ff" for c in word):
                continue
            crop = None
            if with_images:
                (x0, y0), (x1, y1) = sh["points"][0], sh["points"][-1]
                box = (int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)) + 1, int(max(y0, y1)) + 1)
                crop = page.crop(box)
            yield split, word, crop


def indic_vocab(script: str) -> None:
    rows = _tamil_rows(False) if script == "tamil" else _bengali_rows(False)
    c = {"train": Counter(), "test": Counter()}
    for split, txt, _ in rows:
        c[split][txt] += 1
    words = sorted(set(c["train"]) | set(c["test"]), key=lambda w: -(c["train"][w] + c["test"][w]))
    out = INDIC_BANKS[script].parent / "vocab.tsv"
    out.write_text("".join(f"{w}\t{c['train'][w]}\t{c['test'][w]}\n" for w in words), encoding="utf-8")
    print(f"{script} vocab: {len(words)} distinct words, {sum(c['train'].values())} train / "
          f"{sum(c['test'].values())} test images -> {out}")


def indic_extract(script: str, words: set, per_word: int = 30) -> None:
    bank = INDIC_BANKS[script]
    counts = {"train": Counter(), "test": Counter()}
    rows = _tamil_rows(True) if script == "tamil" else _bengali_rows(True)
    for split, txt, im in rows:
        if txt not in words or counts[split][txt] >= per_word:
            continue
        img = Image.open(io.BytesIO(im)).convert("L") if isinstance(im, bytes) else im
        d = bank / txt
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"{_sample_no(counts[split][txt], split)}.png")
        counts[split][txt] += 1
    have = {w for w in words if counts["train"][w] and counts["test"][w]}
    print(f"{script} bank: {len(have)}/{len(words)} words with train and test samples -> {bank}")
    if words - have:
        print("missing a split:", " ".join(sorted(words - have)))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "datagen"))  # TinyScribe: was src/
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "emnist":
        build_emnist()
    elif cmd == "iiit-vocab":
        iiit_vocab()
    elif cmd == "iiit-extract":
        from generate.value_gen import hw_hindi_words
        iiit_extract(hw_hindi_words())
    elif cmd in ("tamil-vocab", "bengali-vocab"):
        indic_vocab(cmd.split("-")[0])
    elif cmd == "indic-extract":
        from generate.value_gen import hw_indic_words
        for sc in ("tamil", "bengali"):
            indic_extract(sc, hw_indic_words(sc))
    else:
        sys.exit(__doc__)
