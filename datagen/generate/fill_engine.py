#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/fill_engine.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""IndicFormBench fill engine (Phase 3).

Two modes:
  fill-flat   — template JSON + blank template PDF  → overlaid marks at checkbox
                bboxes (never AcroForm /MK), drawn field values with font/baseline
                jitter, realistic blank fraction; emits one filled PNG per page @
                200 DPI + a filled PDF + ground-truth JSON.
  fill-acro   — native-fillable PDF (AcroForm)     → set text fields, flatten,
                rasterise @ 200 DPI, emit ground-truth JSON  (warm-up path).

bbox convention (frozen schema): pixels @ 200 DPI, origin TOP-LEFT.
  -> PIL: use directly (px, y-down).
  -> PyMuPDF/PDF points: x_pt = x_px*72/200 ; y_pt = y_px*72/200 (fitz y-down).
"""
import argparse
import json
import functools
import math
import random
import sys
from pathlib import Path

import fitz
import jsonschema
from PIL import Image, ImageDraw, ImageFont

# Project root (src/generate/../..) and src/ (for `generate.` imports) on path.
SELF = Path(__file__).resolve()
SRC = SELF.parent.parent
PROJECT_ROOT = SELF.parents[2]
sys.path.insert(0, str(SRC))

from generate.marks import MARK_STYLES, draw_mark
from generate.value_gen import aksharas, generate_value, table_row_values
from generate.handwriting import Writer, draw_hw, plan_hw

DPI = 200
PT_PER_PX = 72 / DPI

# Fonts (macOS system). .ttc needs an index; PIL truetype accepts index arg.
def _first_existing(*paths: str) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    return paths[0]


_FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_LIB = "/usr/share/fonts/truetype/liberation"
# macOS first (the original render environment), then Linux: Liberation Sans is
# metric-compatible with Arial; Noto Sans Devanagari ships in assets/fonts (OFL).
FONT_LATIN = _first_existing("/System/Library/Fonts/Supplemental/Arial.ttf", f"{_LIB}/LiberationSans-Regular.ttf")
FONT_LATIN_ITALIC = _first_existing("/System/Library/Fonts/Supplemental/Arial Italic.ttf", f"{_LIB}/LiberationSans-Italic.ttf")
FONT_LATIN_BOLD = _first_existing("/System/Library/Fonts/Supplemental/Arial Bold.ttf", f"{_LIB}/LiberationSans-Bold.ttf")
FONT_DEVANAGARI = _first_existing("/System/Library/Fonts/Devanagari Sangam MN.ttc", str(_FONTS / "NotoSansDevanagari-Regular.ttf"))
# Indic script -> (font, Unicode block); Noto Sans (OFL) in assets/fonts, shaped with raqm
INDIC = {"devanagari": (FONT_DEVANAGARI, (0x0900, 0x097F)),
         "bengali": (str(_FONTS / "NotoSansBengali-Regular.ttf"), (0x0980, 0x09FF)),
         "tamil": (str(_FONTS / "NotoSansTamil-Regular.ttf"), (0x0B80, 0x0BFF))}
# tall/descending samples per script, so every line of a box shares one ink extent
INK_SAMPLE = {None: "Ágjpqy|É", "devanagari": "कृपूगुझ्ऊिीेैोौँंः",
              "bengali": "কৃপূগুঝ্ঊিীেৈোৌঁংঃ্র", "tamil": "கூழ்ஞெகொக்ஷீஜுஙு"}

# template language -> script of its values (table cells without value_script)
LANG_SCRIPT = {"english": "latin", "hindi": "devanagari", "bilingual": "devanagari",
               "bengali": "bengali", "tamil": "tamil"}

# TinyScribe: template schema lives in datagen/
SCHEMA_PATH = PROJECT_ROOT / "datagen" / "template_schema.json"
CHECKED_PROB = 0.8          # share of groups whose single box gets marked
GROUP_CHECK_PROB = 0.8      # density of filled checkboxes overall
BLANK_FRACTION = 0.25       # fraction of fields intentionally left blank (3.6)
MARK_WEIGHTS = {"pen": 6, "check": 4, "cross": 4, "filled": 2, "dot": 1, "scribble": 2}


class EngineError(RuntimeError):
    pass


def _load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _load_template(tpl_path: Path) -> dict:
    tpl = json.loads(tpl_path.read_text())
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    jsonschema.validate(tpl, schema)
    return tpl


def _jpeg_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _blank_pages(pdf_path: Path, dpi: int = DPI) -> list[Image.Image]:
    doc = fitz.open(str(pdf_path))
    imgs = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return imgs


def _synthetic_blank(tpl: dict, dpi: int = DPI) -> Image.Image:
    """A4 blank page @ dpi with the annotated labels/types as light reference
    guide lines — used when the blank template PDF isn't downloadable yet,
    purely to exercise and eyeball the mark/value pipeline."""
    pt_w, pt_h = 595, 842
    img = Image.new("RGB", (int(pt_w * dpi / 72), int(pt_h * dpi / 72)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    scale = dpi / 72
    for b in tpl["checkboxes"]:
        x, y, w, h = [int(v) for v in b["bbox"]]
        d.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=2)
        d.text((int(x + w + 6 * scale), int(y)), f"[{b['id']} {b['group']}]", fill=(0, 0, 0))
    for f in tpl["fields"]:
        x, y, w, h = [int(v) for v in f["value_bbox"]]
        d.rectangle([x, y, x + w, y + h], outline=(150, 150, 150), width=1)
        d.line([x, y, x + w, y + h], fill=(200, 200, 200), width=1)
        d.text((int(x + 3), int(y - 14)), f"[{f['id']} {f['value_type']}]", fill=(120, 120, 120))
    return img


def _pick_mark_style(rng: random.Random) -> str:
    styles, weights = zip(*MARK_WEIGHTS.items())
    return rng.choices(styles, weights=weights, k=1)[0]


def _page_of(pages: list, element: dict) -> int:
    """Element's 0-based page (schema v0.1: optional, default 0), clamped."""
    return min(max(0, int(element.get("page", 0))), len(pages) - 1)


import statistics


def _hand_font(rng, size, label_script):
    script = "devanagari" if label_script == "bilingual" else label_script
    if script in INDIC:
        return ImageFont.truetype(INDIC[script][0], size, index=0, layout_engine=ImageFont.Layout.RAQM)
    base = rng.choice([FONT_LATIN, FONT_LATIN_ITALIC, FONT_LATIN_BOLD, FONT_LATIN_ITALIC])
    return ImageFont.truetype(base, size, layout_engine=ImageFont.Layout.RAQM)


def _detect_comb(img: Image.Image, bbox) -> dict | None:
    """Detect a one-char-per-cell comb grid inside value_bbox by scanning an
    even pitch of near-vertical fill-strips. Returns {'pitch','centers','n'}
    anchored at the first reliable strip, or None if no comb found.
    MUST be called on the pristine blank page (text/marks pollute strips)."""
    x, y, w, h = [int(v) for v in bbox]
    if w < 20 or h < 8:
        return None
    W, H = img.size
    px = img.convert("L").load()
    top, bot = max(0, y + int(h * 0.25)), min(H, y + int(h * 0.75))
    x1, y1 = min(W, x + w), min(H, y + h)
    cols = [cx for cx in range(max(0, x), x1)
            if sum(1 for yy in range(top, bot) if px[cx, yy] < 110) / max(1, bot - top) > 0.45]
    if len(cols) < 3:
        return None
    strips, run = [], [cols[0]]
    for c in cols[1:]:
        if c - run[-1] <= 2:
            run.append(c)
        else:
            strips.append(int(sum(run) / len(run)))
            run = [c]
    strips.append(int(sum(run) / len(run)))
    if len(strips) < 2:
        return None
    gaps = [strips[i + 1] - strips[i] for i in range(len(strips) - 1)]
    # keep the longest contiguous run whose gaps match the median within 18%
    tol = lambda g, m: abs(g - m) <= m * 0.18
    med = statistics.median(gaps)
    best, cur = [], []
    transitioned = False
    for i, g in enumerate(gaps):
        if tol(g, med):
            if not cur:
                cur = [strips[i]]
            cur.append(strips[i + 1])
        else:
            if len(cur) > len(best):
                best, cur = cur, []
            cur = []
    if len(cur) > len(best):
        best = cur
    if len(best) < 3:
        return None
    med = statistics.median([best[i + 1] - best[i] for i in range(len(best) - 1)])
    pitch = med
    start_x = best[0]
    # a fainter left border (some forms draw the outer line lighter than the cell dividers)
    # exactly one pitch before the first strip, still inside the box, is the comb's real start
    lead = int(round(start_x - pitch))
    if lead >= x and any(sum(1 for yy in range(top, bot) if px[cx, yy] < 160) / max(1, bot - top) > 0.45
                         for cx in range(max(x, lead - 2), min(x1, lead + 3))):
        start_x = start_x - pitch
    cells_vis = int((x + w - start_x) / pitch + 0.2)  # tolerate a box ending just short of its last cell
    if cells_vis < 1:
        return None
    centers = [start_x + pitch * (i + 0.5) for i in range(cells_vis)]
    return {"pitch": pitch, "centers": centers, "n": min(cells_vis, 40)}


def _fit_font(value: str, label_script, max_w: float, max_h: float, rng) -> ImageFont.FreeTypeFont:
    """Largest font size whose widest line fits max_w (and line count fits max_h)."""
    fs = max(8, int(max_h * 0.72))
    while fs > 8:
        font = _hand_font(rng, fs, label_script)
        longest = max(sum(font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in ln)
                      for ln in value.split(" "))
        if longest <= max_w - 4 and (int(max_h / (fs * 1.3)) or 1) >= 1:
            return font
        fs -= 2
    return _hand_font(rng, 8, label_script)


@functools.lru_cache(maxsize=None)
def _covers(path: str, ch: str) -> bool:
    """Whether the font at `path` has a real glyph for `ch` (not the .notdef box)."""
    f = ImageFont.truetype(path, 40, index=0)
    mask = lambda c: (f.getmask(c).size, bytes(f.getmask(c)))
    return ch.isspace() or mask(ch) != mask("\uffff")


class _Mixed:
    """An Indic font with a Latin fallback for the characters it lacks: Noto Sans
    Devanagari/Tamil have no Latin letters and Noto Sans Bengali not even digits or
    punctuation, so a mixed value ("199 Linking Road, इंदौर") would otherwise draw tofu.
    Text is split into runs per font and drawn run by run; a value the Indic font covers
    fully is one run, drawn exactly as before."""

    def __init__(self, script: str, size: int):
        self.path = INDIC[script][0]
        self.main = ImageFont.truetype(self.path, size, index=0, layout_engine=ImageFont.Layout.RAQM)
        self.alt = ImageFont.truetype(FONT_LATIN, size, layout_engine=ImageFont.Layout.RAQM)
        self._meter = ImageDraw.Draw(Image.new("L", (1, 1)))

    def runs(self, text: str) -> list:
        out = []
        for ch in text:
            f = self.main if _covers(self.path, ch) else self.alt
            if out and (out[-1][1] is f or ch.isspace()):
                out[-1][0] += ch
            else:
                out.append([ch, f])
        return out

    def length(self, text: str) -> float:
        return sum(self._meter.textlength(t, font=f) for t, f in self.runs(text))

    def getbbox(self, text: str) -> tuple:
        x, boxes = 0.0, []
        for t, f in self.runs(text):
            b = f.getbbox(t)
            boxes.append((x + b[0], b[1], x + b[2], b[3]))
            x += self._meter.textlength(t, font=f)
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)) if boxes else (0, 0, 0, 0)

    def draw(self, d: ImageDraw.ImageDraw, xy, text: str, fill):
        x, y = xy
        for t, f in self.runs(text):
            d.text((x, y), t, font=f, fill=fill)
            x += self._meter.textlength(t, font=f)


def _indic_script(value: str):
    """The Indic script of `value` (first one found), or None for Latin/digits only."""
    for c in value:
        for name, (_, (lo, hi)) in INDIC.items():
            if lo <= ord(c) <= hi:
                return name
    return None



def _ink_extent(font, texts, script):
    """(top, bottom) of the ink of `texts` plus a tall/descending sample, relative to the draw origin."""
    sample = INK_SAMPLE[script]
    # glyphs are drawn one character at a time, so measure each character on its own
    chars = {ch for ch in "".join(texts) + sample if not ch.isspace()}
    boxes = [font.getbbox(ch) for ch in chars]
    return min(bb[1] for bb in boxes), max(bb[3] for bb in boxes)


def _plan_text(value: str, bbox, blank: Image.Image, rng) -> dict:
    """Decide how `value` is laid out inside value_bbox, measured with the exact
    per-glyph advance and real ink extents the drawer uses (so descenders and
    matras are never clipped). plan['ok'] is False only when the whole value cannot
    be shown (comb too short, or too long even at the 9px minimum); plan['shown']
    is precisely the text that ends up on the page.
    MUST be given the pristine `blank` page (drawn marks/text pollute comb detection)."""
    x, y, w, h = [int(v) for v in bbox]
    if w <= 4 or h <= 4 or not value:
        return {"kind": "none", "ok": bool(value) is False, "shown": ""}
    script = _indic_script(value)
    comb = _detect_comb(blank, bbox)
    if comb and script:
        # one shaped akshara per cell (per-codepoint cells strand matras on dotted circles)
        cells = aksharas(value)
        shown = cells[:comb["n"]]
        size = max(8, int(min(comb["pitch"], h) * 0.82))
        while True:
            font = _Mixed(script, size)
            boxes = [font.getbbox(c) for c in shown if not c.isspace()] or [font.getbbox("0")]
            top, bot = min(b[1] for b in boxes), max(b[3] for b in boxes)
            wide = max(b[2] - b[0] for b in boxes)
            if (bot - top <= h - 2 and wide <= comb["pitch"] - 3) or size <= 8:
                break
            size -= 1
        return {"kind": "comb", "comb": comb, "font": font, "size": size, "top": top,
                "ink": bot - top, "cells": shown, "shown": "".join(shown), "ok": len(shown) == len(cells)}
    if comb:
        size = max(8, int(min(comb["pitch"], h) * 0.82))
        while True:
            font = _hand_font(rng, size, "latin")
            top, bot = _ink_extent(font, [value[:comb["n"]]], None)
            if bot - top <= h - 2 or size <= 8:
                break
            size -= 1
        shown = value[:comb["n"]]
        return {"kind": "comb", "comb": comb, "font": font, "size": size, "top": top,
                "ink": bot - top, "shown": shown, "ok": len(shown) == len(value)}
    base = None if script else rng.choice([FONT_LATIN, FONT_LATIN_ITALIC, FONT_LATIN_BOLD, FONT_LATIN_ITALIC])
    meter = ImageDraw.Draw(Image.new("L", (1, 1)))
    make = (lambda fs: _Mixed(script, fs)) if script \
        else (lambda fs: ImageFont.truetype(base, fs, layout_engine=ImageFont.Layout.RAQM))
    # whole-string measurement, not per-char sum: matras/conjuncts can be
    # narrower or wider than the sum of their component glyphs once shaped
    width = lambda text, font: font.length(text) if isinstance(font, _Mixed) else meter.textlength(text, font=font)
    words = value.split(" ")
    best = None
    for fs in range(max(9, int(h * 0.72)), 8, -1):
        font = make(fs)
        lines, cur = [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if width(trial, font) <= w - 4:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        top, bot = _ink_extent(font, lines, script)
        ink = bot - top
        lh = max(int(fs * 1.25), ink + 1)
        total = (len(lines) - 1) * lh + ink
        fits = total <= h - 2 and all(width(ln, font) <= w - 4 for ln in lines)
        best = {"kind": "text", "font": font, "fs": fs, "lh": lh, "lines": lines,
                "top": top, "ink": total}
        if fits:
            best["ok"] = True
            break
    else:
        best["ok"] = False
        keep = max(1, (h - 2 - best["ink"] + best["lh"] * len(best["lines"])) // best["lh"])
        best["lines"] = [ln for ln in best["lines"][:keep] if width(ln, best["font"]) <= w - 4]
    best["shown"] = " ".join(best["lines"])
    return best


def _draw_plan(img: Image.Image, plan: dict, bbox, rng):
    """Render a plan strictly inside value_bbox (layer is exactly bbox-sized and pasted
    masked). The hand-written tilt is capped so text at the ends of a long box cannot
    swing out of the box."""
    if plan["kind"] == "none" or not plan["shown"]:
        return
    x, y, w, h = [int(v) for v in bbox]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    margin = (h - plan["ink"]) / 2
    jit = max(0.0, min(1.0, margin - 1.5))
    if plan["kind"] == "comb":
        comb, font = plan["comb"], plan["font"]
        y0 = margin - plan["top"]
        if "cells" in plan:  # Devanagari: centred akshara per cell
            for i, cl in enumerate(plan["cells"]):
                b = font.getbbox(cl)
                xc = comb["centers"][i] - (b[0] + b[2]) / 2 + rng.uniform(-1.2, 1.2) - x
                font.draw(ld, (xc, y0 + rng.uniform(-jit, jit)), cl, (30, 30, 40, 255))
        else:
            for i, ch in enumerate(plan["shown"]):
                xc = comb["centers"][i] + rng.uniform(-1.2, 1.2) - x
                ld.text((xc, y0 + rng.uniform(-jit, jit)), ch, font=font, fill=(30, 30, 40, 255))
    else:
        # whole-line draw: per-character drawing bypasses raqm shaping, which
        # Devanagari/Bengali/Tamil (matra reordering, conjunct ligatures) need
        font, lh, lines = plan["font"], plan["lh"], plan["lines"]
        for li, ln in enumerate(lines):
            y0 = margin + li * lh - plan["top"] + rng.uniform(-jit, jit)
            xc = 3 + rng.uniform(-1.0, 1.0)
            if isinstance(font, _Mixed):
                font.draw(ld, (xc, y0), ln, (30, 30, 40, 255))
            else:
                ld.text((xc, y0), ln, font=font, fill=(30, 30, 40, 255))
    spare = margin - jit - 1
    max_ang = min(2.5, math.degrees(math.atan(spare / (w / 2)))) if spare > 0 else 0.0
    ang = rng.uniform(-max_ang, max_ang)
    if ang:
        layer = layer.rotate(ang, expand=False, resample=Image.BICUBIC,
                             fillcolor=(0, 0, 0, 0))
    img.paste(layer, (x, y), layer)


def _emit_instance(tpl, pages, gt, out_dir, instance_id, seed, dpi=DPI, handwriting=False):
    out_dir = Path(out_dir)
    # the handwriting split sits beside the printed one: its own page dir + GT dir,
    # and a __hw page suffix like the degraded splits' __light/__medium/__heavy
    rendered = out_dir / "rendered" / "handwritten" if handwriting else out_dir / "rendered"
    ground_truth = out_dir / ("ground_truth_hw" if handwriting else "ground_truth")
    suffix = "__hw" if handwriting else ""
    rendered.mkdir(parents=True, exist_ok=True)
    ground_truth.mkdir(parents=True, exist_ok=True)
    png_paths = []
    filled_pdf = (rendered if handwriting else out_dir) / f"{instance_id}{suffix}.pdf"
    doc = fitz.open()
    for i, img in enumerate(pages):
        png = rendered / f"{instance_id}_p{i + 1}{suffix}.png"
        img.save(png)
        png_paths.append(str(png))
        pdf_page = doc.new_page(width=img.width * PT_PER_PX, height=img.height * PT_PER_PX)
        pdf_page.insert_image(pdf_page.rect, stream=_jpeg_bytes(img))
    gt["_instance_files"] = {"png_pages": png_paths, "filled_pdf": str(filled_pdf)}
    doc.save(str(filled_pdf))
    doc.close()
    gt_path = ground_truth / f"{instance_id}.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    return gt_path, png_paths, str(filled_pdf)


def fill_flat(tpl_json: Path, pdf_path: Path, out_dir: Path, seed: int = 0,
              blank_frac: float = BLANK_FRACTION, handwriting: bool = False):
    """Core engine: overlay marks + field values on a flat template.
    handwriting=True composites real handwriting (handwriting.py) instead of fonts."""
    pages, gt, truncated = render_flat(tpl_json, pdf_path, seed, blank_frac, handwriting)
    gt_path, pngs, pdf = _emit_instance(None, pages, gt, out_dir, gt["instance_id"], seed, handwriting=handwriting)
    return {"instance_id": gt["instance_id"], "gt": gt_path, "pngs": pngs, "pdf": pdf, "truncated": truncated}


def render_flat(tpl_json: Path, pdf_path: Path, seed: int = 0,
                blank_frac: float = BLANK_FRACTION, handwriting: bool = False):
    """TinyScribe: fill_flat's rendering without the disk writes. Returns (pages, gt, truncated):
    the filled 200 DPI page images, the ground-truth dict and the ids of truncated values."""
    tpl = _load_template(tpl_json)
    rng = random.Random(seed)
    pages = _blank_pages(pdf_path) if pdf_path.exists() else [_synthetic_blank(tpl)]
    blanks = [im.copy() for im in pages]  # pristine copies for comb detection
    instance_id = f"{tpl['template_id']}_s{seed:04d}"

    # ---- checkbox states (one per group; radio semantics) ----
    boxes = list(tpl["checkboxes"])
    groups: dict = {}
    for b in boxes:
        groups.setdefault(b["group"], []).append(b)
    ct = {"checked": {}, "marks": {}}
    for g, members in groups.items():
        chosen = rng.random() < GROUP_CHECK_PROB
        if not chosen:
            for b in members:
                ct["checked"][b["id"]] = False
            continue
        active = rng.choice(members)
        style = _pick_mark_style(rng)
        if active.get("mark") == "circle":
            style = "circle"
        for b in members:
            checked = (b["id"] == active["id"])
            page = _page_of(pages, b)
            ct["checked"][b["id"]] = checked
            if checked:
                ct["marks"][b["id"]] = style
                draw_mark(pages[page], b["bbox"], style, rng)

    # ---- field values ----
    values, truncated = {}, []
    writer = Writer(rng) if handwriting else None
    for f in tpl["fields"]:
        if rng.random() < blank_frac:
            values[f["id"]] = None
            continue
        page = _page_of(pages, f)
        # up to 12 candidate values: keep the first the box can show in full, so the
        # ground truth is never longer than what is actually drawn
        comb = _detect_comb(blanks[page], f["value_bbox"]) if handwriting else None
        for _ in range(12):
            val = generate_value(f["value_type"], f.get("value_script") or f["label_script"], rng,
                                 handwriting=handwriting)
            if handwriting:
                plan = plan_hw(val, f["value_bbox"], comb, writer, rng)
            else:
                plan = _plan_text(val, f["value_bbox"], blanks[page], rng)
            if plan["ok"]:
                break
        if not plan["ok"]:
            truncated.append(f["id"])
        values[f["id"]] = plan["shown"] or None
        if handwriting:
            draw_hw(pages[page], plan, f["value_bbox"], writer, rng)
        else:
            _draw_plan(pages[page], plan, f["value_bbox"], rng)
    # ---- table value cells (scope v3, Track A); after the fields, so adding a table to a
    # template leaves every field value of an existing instance unchanged ----
    gt_tables = []
    for tb in tpl.get("tables", []):
        vcells = [c for c in tb["cells"] if c.get("value_type") and c.get("bbox")]
        if not vcells:
            continue
        page = _page_of(pages, tb)
        rows = sorted({c["row"] for c in vcells})
        policy = tb.get("row_fill", "cells")
        if policy == "prefix":
            filled = set(rows[:rng.randint(1, len(rows))])
        elif policy == "rows":
            filled = {r for r in rows if rng.random() >= blank_frac}
        else:
            filled = set(rows)
        shown, prev = {}, None
        for r in rows:
            rc = sorted((c for c in vcells if c["row"] == r), key=lambda c: c["col"])
            keep = [r in filled and (policy != "cells" or rng.random() >= blank_frac) for c in rc]
            if not any(keep):
                continue
            scripts = [c.get("value_script") or LANG_SCRIPT.get(tpl["language"], "latin") for c in rc]
            combs = [_detect_comb(blanks[page], c["bbox"]) if handwriting else None for c in rc]

            def plan(c, val, comb):
                return plan_hw(val, c["bbox"], comb, writer, rng) if handwriting \
                    else _plan_text(val, c["bbox"], blanks[page], rng)
            # whole-row candidates keep the row consistent (marks/percentage, from/to dates);
            # a cell that never fits then retries on its own
            for _ in range(12):
                vals = table_row_values([c["value_type"] for c in rc], scripts[0], rng, handwriting, prev)
                plans = [plan(c, v, cb) if k else None for c, v, cb, k in zip(rc, vals, combs, keep)]
                if all(p is None or p["ok"] for p in plans):
                    break
            prev = vals
            for i, (c, p) in enumerate(zip(rc, plans)):
                if p is None:
                    continue
                for _ in range(12 if not p["ok"] else 0):
                    p = plan(c, generate_value(c["value_type"], scripts[i], rng, handwriting=handwriting), combs[i])
                    if p["ok"]:
                        break
                if not p["ok"]:
                    truncated.append(f"{tb['id']}[{c['row']},{c['col']}]")
                shown[(c["row"], c["col"])] = p["shown"] or None
                if handwriting:
                    draw_hw(pages[page], p, c["bbox"], writer, rng)
                else:
                    _draw_plan(pages[page], p, c["bbox"], rng)
        gt_tables.append({
            "id": tb["id"], "page": tb.get("page", 0), "bbox": tb["bbox"],
            "cells": [{"row": c["row"], "col": c["col"], "rowspan": c["rowspan"], "colspan": c["colspan"],
                       "text": shown.get((c["row"], c["col"])) if c.get("value_type") else c["text"],
                       **({"value_type": c["value_type"]} if c.get("value_type") else {})}
                      for c in tb["cells"]],
        })
    if truncated:
        print(f"WARN {instance_id}: no value fits fully in {truncated}; ground truth "
              f"holds only the visible part", file=sys.stderr)

    gt = {
        "schema_version": tpl["schema_version"],
        "template_id": tpl["template_id"],
        "instance_id": instance_id,
        "source_url": tpl["source_url"],
        "licence": tpl["licence"],
        "language": tpl["language"],
        "fillable": False,
        "seed": seed,
        "dpi": DPI,
        "generator": "fill_engine.fill_flat",
        "checkboxes": [
            {"id": b["id"], "group": b["group"], "label_text": b["label_text"],
             "label_script": b["label_script"], "page": b.get("page", 0),
             "checked": ct["checked"][b["id"]],
             "mark_style": ct["marks"].get(b["id"])}
            for b in boxes
        ],
        "fields": [
            {"id": f["id"], "label_text": f["label_text"],
             "label_script": f["label_script"], "value_type": f["value_type"],
             "page": f.get("page", 0), "value": values[f["id"]]}
            for f in tpl["fields"]
        ],
        "tables": gt_tables,
    }
    if handwriting:
        gt["split"] = "handwritten"
    return pages, gt, truncated


def fill_acroform(pdf_path: Path, out_dir: Path, seed: int = 0, blank_frac: float = BLANK_FRACTION):
    """Warm-up path: set AcroForm (text/radio) fields, flatten, rasterise @200DPI."""
    rng = random.Random(seed)
    doc = fitz.open(str(pdf_path))
    ticked_groups: set = set()
    gt = {
        "template_id": pdf_path.stem, "instance_id": f"{pdf_path.stem}_s{seed:04d}",
        "seed": seed, "dpi": DPI, "fillable": True,
        "generator": "fill_engine.fill_acroform",
        "fields": [], "checkboxes": [],
    }
    for pno, page in enumerate(doc):
        for w in page.widgets() or []:
            name = w.field_name or f"w_{pno}_{len(gt['fields'])}"
            if w.field_type == 7:  # text
                if rng.random() >= blank_frac:
                    val = generate_value("name", "latin", rng)
                    w.field_value = val
                    w.update()
                    gt["fields"].append({"widget": name, "page": pno, "value": val})
                else:
                    gt["fields"].append({"widget": name, "page": pno, "value": None})
            elif w.field_type == 5:  # radio: tick one option per on/off group
                group = name.split("[")[0]
                if group not in ticked_groups and rng.random() < GROUP_CHECK_PROB:
                    ticked_groups.add(group)
                    w.field_value = w.field_value or "0"
                    w.update()
                    gt["checkboxes"].append({"widget": group, "page": pno, "checked": True})
    doc.bake()
    # rasterise + GT json
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=DPI, colorspace=fitz.csRGB)
        pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    out = Path(out_dir)
    iid = gt["instance_id"]
    rendered = out / "rendered"; rendered.mkdir(parents=True, exist_ok=True)
    gt_dir = out / "ground_truth"; gt_dir.mkdir(parents=True, exist_ok=True)
    pngs = []
    for i, img in enumerate(pages):
        p = rendered / f"{iid}_p{i + 1}.png"
        img.save(p); pngs.append(str(p))
    filled_pdf = out / "rendered" / f"{iid}.pdf"
    doc.save(str(filled_pdf))
    gt["_instance_files"] = {"png_pages": pngs, "filled_pdf": str(filled_pdf)}
    doc.close()
    gt_path = gt_dir / f"{iid}.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2))
    return {"instance_id": iid, "gt": gt_path, "pngs": pngs, "pdf": str(filled_pdf)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    flat = sub.add_parser("fill-flat", help="template JSON + blank PDF")
    flat.add_argument("tpl_json"); flat.add_argument("pdf")
    flat.add_argument("--out", default=str(PROJECT_ROOT / "data"))
    flat.add_argument("--seed", type=int, default=0)
    flat.add_argument("--blank-frac", type=float, default=BLANK_FRACTION)
    flat.add_argument("--handwriting", action="store_true", help="EN/HI handwriting split")
    acro = sub.add_parser("fill-acro", help="native-fillable PDF (warm-up)")
    acro.add_argument("pdf")
    acro.add_argument("--out", default=str(PROJECT_ROOT / "data"))
    acro.add_argument("--seed", type=int, default=0)
    acro.add_argument("--blank-frac", type=float, default=BLANK_FRACTION)
    args = ap.parse_args(argv)
    if args.mode == "fill-flat":
        res = fill_flat(Path(args.tpl_json), Path(args.pdf), Path(args.out),
                        args.seed, args.blank_frac, handwriting=args.handwriting)
    else:
        res = fill_acroform(Path(args.pdf), Path(args.out), args.seed, args.blank_frac)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())