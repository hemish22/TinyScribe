#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/handwriting.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Handwritten field values for the EN/HI handwriting split (scope v3, Track C).

English is spliced per character from EMNIST, in block capitals: Indian forms ask
for BLOCK LETTERS, EMNIST size-normalises every glyph (so lowercase cannot keep its
x-height), and scoring is case-insensitive. Hindi uses whole real word images from
IIIT-INDIC-HW-WORDS, so each word keeps a real writer's shaping; value_gen's *_HW
lists only produce words that have samples. Punctuation is drawn as pen strokes.
Banks are built by hw_bank.py.

TinyScribe: Bengali and Tamil too. A field's words come from that script's real word bank
(BanglaWriting / IIIT Tamil) when every word has samples; otherwise the whole field is written
in a handwriting-style font (Atma / Kavivanar, OFL) in the writer's hand, the same way comb
fields use Kalam. STATS counts drawn fields by source ("real" or "font") for the manifest.

Exception: Devanagari in a comb (box-per-letter) field is written one akshara per box,
as people fill these forms, in the Kalam handwriting-style font (OFL, assets/fonts),
not real handwriting: IIIT is word-level (only 96 single-akshara words) and handwritten
words cannot be reliably cut into aksharas. Each glyph still gets the writer's pen,
stroke weight and slant plus per-glyph size/rotation jitter. See docs/RESULTS.md caveats.

A Writer holds the per-form hand: pen colour, stroke weight, slant, letter spacing.
Stroke weight is normalised across both sources, which differ a lot in raw stroke
width. plan_hw()/draw_hw() mirror fill_engine._plan_text()/_draw_plan(): measure
first, plan['ok'] only when the whole value fits, plan['shown'] is exactly the text
on the page.
"""
import functools
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from generate.hw_bank import EMNIST_BANK, IIIT_BANK, INDIC_BANKS, ROOT
from generate.value_gen import aksharas

REF = 160                 # line height (px) tokens are composed at, scaled down when drawn
BASE = 0.78               # baseline / line height
CAP = 0.56                # Latin cap height / line height
BODY = 0.52               # Devanagari body (headline to baseline) / line height
UP_EXT, LOW_EXT = 0.40, 0.35   # matra extents above/below the body, / body
L_MIN, L_MAX = 16, 64     # 64 px line ~ 4.5 mm capitals at 200 DPI
UPPER = set("िीेैोौंँ" "িীেৈোৌংঁ" "ிீ")       # TinyScribe: + Bengali, Tamil signs above the body
LOWER = set("ुूृ" "ুূৃ" "ுூ")
PUNCT = set("-./,@:")
PENS = [(25, 45, 140), (20, 30, 95), (30, 30, 35)]   # blue ballpoint, navy, black
INK_THR = 0.3
KALAM = ROOT / "assets" / "fonts" / "Kalam-Regular.ttf"
# TinyScribe: handwriting-style font per script, and the letter whose body sets its size
HAND_FONTS = {"devanagari": (KALAM, "क"),
              "bengali": (ROOT / "assets" / "fonts" / "Atma-Regular.ttf", "ক"),
              "tamil": (ROOT / "assets" / "fonts" / "Kavivanar-Regular.ttf", "க")}
BLOCKS = {"devanagari": (0x0900, 0x097F), "bengali": (0x0980, 0x09FF), "tamil": (0x0B80, 0x0BFF)}
STATS: Counter = Counter()


def _script_of(s: str) -> str | None:
    """TinyScribe: the Indic script of the first Indic character, or None for Latin/digits."""
    for c in s:
        for name, (lo, hi) in BLOCKS.items():
            if lo <= ord(c) <= hi:
                return name
    return None


def _deva(s: str) -> bool:
    """True for any Indic-script text (TinyScribe: Devanagari, Bengali or Tamil)."""
    return _script_of(s) is not None


def _crop(a: np.ndarray, thr: float = INK_THR):
    rows, cols = np.where(a.max(1) > thr)[0], np.where(a.max(0) > thr)[0]
    if not len(rows) or not len(cols):
        return None
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _from_img(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.float32) / 255


def _resize(a: np.ndarray, w: float, h: float) -> np.ndarray:
    return _from_img(_to_img(a).resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS))


def _shear(a: np.ndarray, deg: float) -> np.ndarray:
    """Slant: shift each row right in proportion to its height above the bottom."""
    k = math.tan(math.radians(deg))
    if abs(k) < 1e-3:
        return a
    h, w = a.shape
    # x_out = x_in + k*(h - y) + off, off keeps everything at x >= 0; PIL wants the inverse map
    c = -k * h if k > 0 else 0.0
    out = _to_img(a).transform((w + int(abs(k) * h) + 2, h), Image.AFFINE, (1, k, c, 0, 1, 0),
                               resample=Image.BICUBIC)
    return _crop(_from_img(out), 0.05)


def _rotate(a: np.ndarray, deg: float) -> np.ndarray:
    return _from_img(_to_img(a).rotate(deg, expand=True, resample=Image.BICUBIC))


def _stroke_width(a: np.ndarray) -> float:
    m = a > 0.5
    area = m.sum()
    if not area:
        return 0.0
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = area - inner.sum()
    return 2 * area / max(1, edge)


def _stroke_norm(a: np.ndarray, target: float) -> np.ndarray:
    """Dilate/erode so the stroke is about `target` px wide (sources differ ~3x)."""
    a = np.pad(a, 4)
    diff = target - _stroke_width(a)
    if abs(diff) >= 1.5:
        k = 2 * int(round(abs(diff) / 2)) + 1
        f = ImageFilter.MaxFilter(k) if diff > 0 else ImageFilter.MinFilter(k)
        a = _from_img(_to_img(a).filter(f))
    a = _from_img(_to_img(a).filter(ImageFilter.GaussianBlur(0.7)))
    return _crop(a, 0.05)


# TinyScribe: train and test pages draw from disjoint handwriting samples. Bank files are
# <n>.png; every 5th sample (n % 5 == 0) is reserved for test pages. Set with set_split().
SPLIT = "train"


def set_split(split: str) -> None:
    global SPLIT
    assert split in ("train", "test"), split
    SPLIT = split


def _in_split(p, split: str) -> bool:
    return (int(p.stem) % 5 == 0) == (split == "test")


def _emnist(ch: str) -> tuple:
    return _emnist_split(ch, SPLIT)


@functools.lru_cache(maxsize=None)
def _emnist_split(ch: str, split: str) -> tuple:
    return tuple(p for p in sorted((EMNIST_BANK / str(ord(ch))).glob("*.png")) if _in_split(p, split))


def _iiit(word: str) -> tuple:
    return _iiit_split(word, SPLIT)


def _bank_words(word: str) -> tuple:
    """TinyScribe: samples of an Indic word from its script's bank (IIIT Hindi, IIIT Tamil, BanglaWriting)."""
    script = _script_of(word)
    if script == "devanagari":
        return _iiit_split(word, SPLIT)
    return _iiit_split(word, SPLIT, str(INDIC_BANKS[script])) if script in INDIC_BANKS else ()


@functools.lru_cache(maxsize=None)
def _iiit_split(word: str, split: str, bank: str = str(IIIT_BANK)) -> tuple:
    """Ink-alpha crops of a word's samples; faint scans (low paper/ink contrast) dropped."""
    out = []
    for p in sorted((Path(bank) / word).glob("*.png")):
        if not _in_split(p, split):
            continue
        g = np.asarray(Image.open(p), dtype=np.float32)
        bg, ink = np.median(g), np.percentile(g, 0.5)
        if bg - ink < 95:
            continue
        a = np.clip((bg - g - 20) / (bg - ink - 20), 0, 1)
        # scans carry edge lines and stray specks that would inflate the crop (and so
        # shrink the word when it is scaled to the line): clear a border band, then
        # trim the outermost 0.4% of ink mass on each side
        by, bx = max(2, a.shape[0] // 40), max(2, a.shape[1] // 40)
        a[:by], a[-by:], a[:, :bx], a[:, -bx:] = 0, 0, 0, 0
        if bank == str(INDIC_BANKS["tamil"]):
            # TinyScribe: IIIT Tamil scans keep parts of their frame lines away from the edge;
            # blank rows / columns that are mostly ink (a Tamil word never spans 60% of them;
            # not for Bengali, whose headline can)
            ink = a > 0.3
            rows_ = np.where(ink.mean(1) > 0.6)[0]
            cols_ = np.where(ink.mean(0) > 0.6)[0]
            for r_ in rows_:
                a[max(0, r_ - 2):r_ + 3] = 0
            for c_ in cols_:
                a[:, max(0, c_ - 2):c_ + 3] = 0
        r, c = np.cumsum(a.sum(1)), np.cumsum(a.sum(0))
        if not r[-1]:
            continue
        r0, r1 = np.searchsorted(r, r[-1] * 0.004), np.searchsorted(r, r[-1] * 0.996)
        c0, c1 = np.searchsorted(c, c[-1] * 0.004), np.searchsorted(c, c[-1] * 0.996)
        a = _crop(a[r0:r1 + 1, c0:c1 + 1])
        if a is not None and bank == str(INDIC_BANKS["tamil"]) and _long_run(a) > 0.9 * a.shape[0]:
            continue   # TinyScribe: leftover guide/underline stroke in the scan; no Tamil letter has one
        if a is not None:
            out.append(a)
    if out:  # drop samples whose shape is far off the word's typical aspect (fragments, mis-crops)
        med = float(np.median([a.shape[1] / a.shape[0] for a in out]))
        out = [a for a in out if 0.65 <= a.shape[1] / a.shape[0] / med <= 1.5]
    return tuple(out)


def _long_run(a: np.ndarray) -> int:
    """TinyScribe: longest horizontal run of ink pixels in any row."""
    best = 0
    for row in a > 0.5:
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        best = max(best, int((np.where(d == -1)[0] - np.where(d == 1)[0]).max()))
    return best


class Writer:
    """One form's hand: kept for every field of the instance."""

    def __init__(self, rng: random.Random):
        self.pen = rng.choice(PENS)
        self.stroke = REF * 0.065 * rng.uniform(0.85, 1.2)
        self.slant = rng.uniform(-10, 6)
        self.gap = rng.uniform(0.06, 0.14)
        self.space = rng.uniform(0.28, 0.40)
        self.density = rng.uniform(0.85, 1.0)


def _punct(ch: str, w: Writer, rng) -> tuple:
    """(alpha, bottom offset below baseline) for a stroke-drawn punctuation mark."""
    cap, pw = REF * CAP, max(2, round(w.stroke))
    if ch == "-":
        W, H = cap * rng.uniform(0.35, 0.5), pw + 2
        im = Image.new("L", (round(W) + pw, H), 0)
        ImageDraw.Draw(im).line([(pw / 2, H / 2), (W, H / 2 + rng.uniform(-1, 1))], fill=255, width=pw)
        return _from_img(im), -cap * 0.42
    if ch in ".:":
        d = pw * 1.4
        im = Image.new("L", (round(d) + 2, round(cap * 0.55) if ch == ":" else round(d) + 2), 0)
        dr = ImageDraw.Draw(im)
        dr.ellipse([1, im.height - d - 1, d + 1, im.height - 1], fill=255)
        if ch == ":":
            dr.ellipse([1, 1, d + 1, d + 1], fill=255)
        return _from_img(im), 0
    if ch == ",":
        H = cap * 0.32
        im = Image.new("L", (round(pw * 2.5), round(H)), 0)
        ImageDraw.Draw(im).line([(im.width - pw, pw), (pw, H - 1)], fill=255, width=pw)
        return _from_img(im), cap * 0.16
    if ch == "/":
        W = cap * rng.uniform(0.45, 0.6)
        im = Image.new("L", (round(W) + pw, round(cap)), 0)
        ImageDraw.Draw(im).line([(pw / 2, cap - pw / 2), (W, pw / 2)], fill=255, width=pw)
        return _from_img(im), 0
    if ch == "@":
        s = round(cap * 1.05)
        im = Image.new("L", (s + pw, s + pw), 0)
        dr = ImageDraw.Draw(im)
        c, r = s / 2, s * 0.22
        dr.ellipse([c - r, c - r, c + r, c + r], outline=255, width=pw)
        dr.line([(c + r, c - r), (c + r, c + r * 1.1)], fill=255, width=pw)
        dr.arc([pw / 2, pw / 2, s, s], start=30, end=345, fill=255, width=pw)
        dr.line([(c + r, c + r * 1.1), (s - pw, c + s * 0.12)], fill=255, width=pw)
        return _from_img(im), cap * 0.12
    raise KeyError(ch)


def _latin_glyph(ch: str, w: Writer, rng) -> tuple:
    if ch in PUNCT:
        return _punct(ch, w, rng)
    paths = _emnist(ch)
    if not paths:
        raise KeyError(ch)
    a = _crop(1 - np.asarray(Image.open(rng.choice(paths)), dtype=np.float32) / 255, 0.15)
    h = REF * CAP * rng.uniform(0.92, 1.06)
    a = _resize(a, a.shape[1] * h / a.shape[0], h)
    a = _stroke_norm(_rotate(_shear(a, w.slant), rng.uniform(-4, 4)), w.stroke)
    return a, rng.uniform(-0.02, 0.02) * REF


@functools.lru_cache(maxsize=None)
def _kalam(script: str = "devanagari") -> ImageFont.FreeTypeFont:
    """Kalam sized so its Devanagari body (headline to baseline of क) is BODY of the line.
    TinyScribe: or the script's own handwriting font (HAND_FONTS), sized the same way."""
    path, probe_ch = HAND_FONTS[script]
    probe = ImageFont.truetype(str(path), 200, layout_engine=ImageFont.Layout.RAQM)
    b = probe.getbbox(probe_ch, anchor="ls")
    return ImageFont.truetype(str(path), round(200 * REF * BODY / -b[1]), layout_engine=ImageFont.Layout.RAQM)


def _akshara_glyph(cl: str, w: Writer, rng) -> tuple:
    """(alpha, ink bottom below baseline) for one akshara drawn in Kalam in the writer's hand."""
    font = _kalam(_script_of(cl) or "devanagari")
    im = Image.new("L", (REF * len(cl) + REF, REF * 2), 0)  # TinyScribe: wide enough for a whole word
    ImageDraw.Draw(im).text((REF // 2, REF * 1.4), cl, font=font, fill=255, anchor="ls")
    a = _from_img(im)
    rows = np.where(a.max(1) > INK_THR)[0]
    off = rows[-1] + 1 - REF * 1.4
    a = _crop(a)
    k = rng.uniform(0.94, 1.06)
    a = _resize(a, a.shape[1] * k * rng.uniform(0.95, 1.05), a.shape[0] * k)
    a = _stroke_norm(_rotate(_shear(a, w.slant), rng.uniform(-4, 4)), w.stroke)
    return a, off * k


def _paste(canvas: np.ndarray, a: np.ndarray, x: int, y: int):
    """Max-composite a onto canvas at (x, y), clipped to the canvas."""
    H, W = canvas.shape
    h, w = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 < x1 and y0 < y1:
        np.maximum(canvas[y0:y1, x0:x1], a[y0 - y:y1 - y, x0 - x:x1 - x], out=canvas[y0:y1, x0:x1])


def _latin_token(text: str, w: Writer, rng) -> np.ndarray:
    glyphs = [_latin_glyph(ch, w, rng) for ch in text]
    base = REF * BASE
    gap = REF * CAP * w.gap
    width = sum(g.shape[1] for g, _ in glyphs) + gap * (len(glyphs) - 1)
    canvas = np.zeros((REF, int(width) + 2), np.float32)
    x = 0.0
    for g, off in glyphs:
        _paste(canvas, g, int(x), int(round(base + off - g.shape[0])))
        x += g.shape[1] + gap * rng.uniform(0.7, 1.3)
    return canvas


def _font_token(word: str, w: Writer, rng) -> np.ndarray:
    """TinyScribe: an Indic word in its script's handwriting font, on a REF-high line. Drawn one
    akshara at a time, each with its own size, rotation and baseline jitter and uneven spacing
    (as in comb fields), so the word looks placed by hand rather than typeset."""
    glyphs = [_akshara_glyph(cl, w, rng) for cl in aksharas(word) if not cl.isspace()]
    gap = REF * BODY * w.gap * 0.6
    width = sum(g.shape[1] for g, _ in glyphs) + gap * 1.3 * len(glyphs) + 4
    canvas = np.zeros((REF, int(width) + 2), np.float32)
    x = 1.0
    for g, off in glyphs:
        y = REF * BASE + off - g.shape[0] + rng.uniform(-0.025, 0.025) * REF
        _paste(canvas, g, int(x), int(round(y)))
        x += g.shape[1] + gap * rng.uniform(0.6, 1.3)
    return canvas[:, :int(x) + 2]


def _deva_token(word: str, w: Writer, rng, font: bool = False) -> np.ndarray:
    if font:
        return _font_token(word, w, rng)
    samples = _bank_words(word)
    if not samples:
        raise KeyError(word)
    a = rng.choice(samples)
    up, low = any(c in UPPER for c in word), any(c in LOWER for c in word)
    body = REF * BODY * rng.uniform(0.95, 1.05)
    h = body * (1 + UP_EXT * up + LOW_EXT * low)
    a = _stroke_norm(_resize(a, a.shape[1] * h / a.shape[0], h), w.stroke)
    canvas = np.zeros((REF, a.shape[1] + 2), np.float32)
    top = REF * BASE - body - (body * UP_EXT if up else 0)
    _paste(canvas, a, 1, int(round(top + rng.uniform(-0.02, 0.02) * REF)))
    return canvas


def _core(tok: str) -> str:
    return tok.strip("".join(PUNCT))


def _token(tok: str, w: Writer, rng, font: bool = False) -> tuple:
    """(alpha at REF line height, text as written). Latin is written in capitals;
    punctuation stuck to a Devanagari word is split off and stroke-drawn."""
    if not _deva(tok):
        up = tok.upper()
        return _latin_token(up, w, rng), up
    core = _core(tok)
    lead, trail = tok[:tok.index(core)], tok[tok.index(core) + len(core):]
    parts = ([_latin_token(lead, w, rng)] if lead else []) + [_deva_token(core, w, rng, font)] \
        + ([_latin_token(trail, w, rng)] if trail else [])
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0], tok


def _ink_rows(a: np.ndarray) -> tuple:
    rows = np.where(a.max(1) > INK_THR)[0]
    return (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, 1)


def plan_hw(value: str, bbox, comb, w: Writer, rng) -> dict:
    """Lay out `value` in handwriting inside value_bbox. `comb` is fill_engine's
    _detect_comb() result on the pristine page (or None). Latin values in a comb
    get one glyph per cell; Devanagari is written across the box (whole-word
    images cannot be split into cells)."""
    x, y, bw, bh = [int(v) for v in bbox]
    if bw <= 4 or bh <= 4 or not value:
        return {"kind": "none", "ok": not value, "shown": ""}
    try:
        if comb and not _deva(value):
            text = value.upper()
            shown = text[:comb["n"]]
            cells = [None if ch == " " else _latin_glyph(ch, w, rng) for ch in shown]
            cap = min(comb["pitch"] * 0.72, (bh - 4) * 0.72)
            if cap < 6:
                return {"kind": "none", "ok": False, "shown": ""}
            return {"kind": "comb", "comb": comb, "cells": cells, "s": cap / (REF * CAP),
                    "ext": (-REF * CAP, 0.0), "shown": shown, "ok": len(shown) == len(text),
                    "source": "real"}
        if comb:
            parts = aksharas(value)
            shown = parts[:comb["n"]]
            cells = [None if c.isspace() else
                     _akshara_glyph(c, w, rng) if _deva(c) else _latin_glyph(c.upper(), w, rng)
                     for c in shown]
            ink = [g for g in cells if g]
            if not ink:
                return {"kind": "none", "ok": False, "shown": ""}
            top = min(off - a.shape[0] for a, off in ink)
            bot = max(off for a, off in ink)
            # largest scale at which every akshara (with its matras) fits its box
            s = min(comb["pitch"] * 0.72 / (REF * CAP), (bh - 4) / (bot - top),
                    (comb["pitch"] - 4) / max(a.shape[1] for a, _ in ink))
            if s * REF * CAP < 6:
                return {"kind": "none", "ok": False, "shown": ""}
            shown_txt = "".join(c if _deva(c) else c.upper() for c in shown)
            return {"kind": "comb", "comb": comb, "cells": cells, "s": s, "ext": (top, bot),
                    "shown": shown_txt, "ok": len(shown) == len(parts), "source": "font"}
        # TinyScribe: Bengali/Tamil fields not fully covered by the word bank use the script font
        words = [t for t in value.split(" ") if t]
        font = _script_of(value) in INDIC_BANKS and not all(
            _bank_words(_core(t)) for t in words if _deva(t))
        toks = [_token(t, w, rng, font) for t in words]
    except KeyError:
        return {"kind": "none", "ok": False, "shown": ""}
    space = REF * w.space
    best = None
    for L in range(min(L_MAX, int(bh * 1.1)), L_MIN - 1, -1):
        s = L / REF
        lines, cur, cur_w = [], [], 0.0
        for a, t in toks:
            tw = a.shape[1] * s
            add = tw + (space * s if cur else 0)
            if cur and cur_w + add > bw - 4:
                lines.append(cur)
                cur, cur_w = [], 0.0
                add = tw
            cur.append((a, t))
            cur_w += add
        if cur:
            lines.append(cur)
        top = min(_ink_rows(a)[0] for ln in lines for a, _ in ln)
        bot = max(_ink_rows(a)[1] for ln in lines for a, _ in ln)
        lh = L
        total = (len(lines) - 1) * lh + (bot - top) * s
        wide = max(sum(a.shape[1] for a, _ in ln) * s + space * s * (len(ln) - 1) for ln in lines)
        best = {"kind": "text", "L": L, "s": s, "lh": lh, "lines": lines, "top": top * s,
                "ink": total, "space": space, "source": "font" if font else "real"}
        if total <= bh - 2 and wide <= bw - 4:
            best["ok"] = True
            break
    else:
        if best is None:
            return {"kind": "none", "ok": False, "shown": ""}
        best["ok"] = False
        keep = int((bh - 2 - (best["ink"] - (len(best["lines"]) - 1) * best["lh"])) // best["lh"]) + 1
        best["lines"] = [ln for ln in best["lines"][:max(0, keep)]
                         if sum(a.shape[1] for a, _ in ln) * best["s"] <= bw - 4]
    best["shown"] = " ".join(t for ln in best["lines"] for _, t in ln)
    return best


def draw_hw(img: Image.Image, plan: dict, bbox, w: Writer, rng):
    """Composite a plan in the writer's pen strictly inside value_bbox."""
    if plan["kind"] == "none" or not plan["shown"]:
        return
    STATS[plan.get("source", "real")] += 1
    x, y, bw, bh = [int(v) for v in bbox]
    layer = np.zeros((bh, bw), np.float32)
    s = plan["s"]
    if plan["kind"] == "comb":
        comb = plan["comb"]
        top, bot = plan["ext"]
        ink_h = (bot - top) * s
        base = (bh - ink_h) / 2 - top * s
        for i, g in enumerate(plan["cells"]):
            if g is None:
                continue
            a, off = g
            a = _resize(a, a.shape[1] * s, a.shape[0] * s)
            cx = comb["centers"][i] - x + rng.uniform(-1.2, 1.2)
            _paste(layer, a, int(round(cx - a.shape[1] / 2)), int(round(base + off * s - a.shape[0])))
        margin = (bh - ink_h) / 2
    else:
        margin = (bh - plan["ink"]) / 2
        jit = max(0.0, min(1.0, margin - 1.5))
        for li, ln in enumerate(plan["lines"]):
            width = sum(a.shape[1] for a, _ in ln) + plan["space"] * (len(ln) - 1)
            line = np.zeros((REF, int(width) + 2), np.float32)
            xc = 0.0
            for a, _ in ln:
                _paste(line, a, int(xc), 0)
                xc += a.shape[1] + plan["space"] * rng.uniform(0.85, 1.15)
            line = _resize(line, line.shape[1] * s, REF * s)
            y0 = margin + li * plan["lh"] - plan["top"] + rng.uniform(-jit, jit)
            _paste(layer, line, int(round(3 + rng.uniform(-1, 1))), int(round(y0)))
    spare = margin - 2
    max_ang = min(2.0, math.degrees(math.atan(spare / (bw / 2)))) if spare > 0 else 0.0
    alpha = _to_img(layer * w.density)
    ang = rng.uniform(-max_ang, max_ang)
    if ang:
        alpha = alpha.rotate(ang, expand=False, resample=Image.BICUBIC)
    rgba = Image.new("RGBA", (bw, bh), w.pen + (0,))
    rgba.putalpha(alpha)
    img.paste(rgba, (x, y), rgba)
