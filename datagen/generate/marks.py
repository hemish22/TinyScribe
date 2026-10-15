# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/marks.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Checkbox mark renderers (Phase 3.4) — drawn OVERLAID at the annotated bbox,
never via AcroForm /MK toggle. All draw in PIXEL space (any DPI); bbox = [x,y,w,h]
top-left origin.

Mark styles: check, cross, filled, dot, scribble, pen (simulated ballpoint:
multi-segment stroke with pressure-style width jitter and overshoot), and circle
(a hand-drawn loop around a printed option word; only for boxes annotated mark=circle).
"""
import math
import random

from PIL import ImageDraw

MARK_STYLES = ["check", "cross", "filled", "dot", "scribble", "pen"]


def _rot(points, cx, cy, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in points]


def _draw_lines(draw, points, fill, width, joint="curve"):
    draw.line(points, fill=fill, width=width, joint=joint)


def draw_check(draw, bbox, rng, fill, width):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = min(w, h) * 0.55
    ang = math.radians(rng.uniform(-4, 4))
    p = _rot([(cx - s * 0.45, cy + s * 0.05), (cx - s * 0.05, cy + s * 0.4), (cx + s * 0.5, cy - s * 0.42)], cx, cy, ang)
    _draw_lines(draw, p, fill, width)


def draw_cross(draw, bbox, rng, fill, width):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = min(w, h) * 0.45
    ang = math.radians(rng.uniform(-6, 6))
    p1 = _rot([(cx - s, cy - s), (cx + s, cy + s)], cx, cy, ang)
    p2 = _rot([(cx - s, cy + s), (cx + s, cy - s)], cx, cy, ang)
    _draw_lines(draw, p1, fill, width)
    _draw_lines(draw, p2, fill, width)


def draw_filled(draw, bbox, rng, fill, width):
    x, y, w, h = bbox
    pad = min(w, h) * rng.uniform(0.10, 0.18)
    cx, cy = x + w / 2, y + h / 2
    side = min(w, h) * rng.uniform(0.5, 0.6)
    ang = math.radians(rng.uniform(-12, 12))
    p = _rot([(cx - side / 2, cy - side / 2), (cx + side / 2, cy - side / 2),
              (cx + side / 2, cy + side / 2), (cx - side / 2, cy + side / 2)], cx, cy, ang)
    draw.polygon([(int(px), int(py)) for px, py in p], fill=fill)


def draw_dot(draw, bbox, rng, fill, width):
    x, y, w, h = bbox
    cx, cy = x + w / 2 + rng.uniform(-w * 0.15, w * 0.15), y + h / 2 + rng.uniform(-h * 0.15, h * 0.15)
    r = min(w, h) * rng.uniform(0.16, 0.24)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def draw_scribble(draw, bbox, rng, fill, width):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    s = min(w, h) * rng.uniform(0.35, 0.5)
    pts = []
    pts.append((cx - s * 0.6 + rng.uniform(-3, 3), cy + s * 0.5 * rng.uniform(-1, 1)))
    for _ in range(rng.randint(2, 4)):
        pts.append((cx + rng.uniform(-s, s), cy + rng.uniform(-s, s)))
    pts.append((cx - s * 0.3 + rng.uniform(-3, 3), cy + s * 0.5))
    _draw_lines(draw, pts, fill, max(1, width), joint="curve")


def draw_pen(draw, bbox, rng, fill, width):
    """Simulated ballpoint: 1-3 strokes, each broken into short segments whose
    width jitters (pressure), plus angular drift and slight overshoot of the box."""
    x, y, w, h = bbox
    cx, cy = x + w / 2 + rng.uniform(-w * 0.1, w * 0.1), y + h / 2 + rng.uniform(-h * 0.1, h * 0.1)
    strokes = rng.randint(1, 3)
    for _ in range(strokes):
        slen = min(w, h) * rng.uniform(0.55, 0.95)
        ang = math.radians(rng.uniform(-70, 70))
        dx, dy = math.cos(ang) * slen, math.sin(ang) * slen
        drift = math.radians(rng.uniform(-8, 8))
        for t0 in [0.0, 0.5]:
            pts = []
            for t in [t0, t0 + 0.5, 1.0]:
                pts.append((cx + t * dx + t * t * 6.0, cy + t * dy + math.cos(t * 4 + drift) * 3.0))
            _draw_lines(draw, pts, fill, max(1, width))


MARK_RENDERERS = {
    "check": draw_check, "cross": draw_cross, "filled": draw_filled,
    "dot": draw_dot, "scribble": draw_scribble, "pen": draw_pen,
}


def draw_circle(draw, bbox, rng, fill, width):
    """A hand-drawn loop around a printed option word: an ellipse a little larger than the
    box, slightly tilted and wobbly, whose end overshoots its start (as a pen loop does)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    rx, ry = w / 2 + max(4, w * 0.08), h / 2 + max(4, h * 0.25)
    ang = math.radians(rng.uniform(-6, 6))
    t0 = rng.uniform(0, 2 * math.pi)
    span = 2 * math.pi + rng.uniform(0.25, 0.6)
    n = 48
    pts = []
    for i in range(n + 1):
        t = t0 + span * i / n
        k = 1 + 0.04 * math.sin(3 * t + t0) + 0.05 * i / n      # wobble + drift: the loop does not close on itself
        pts.append((cx + rx * k * math.cos(t), cy + ry * k * math.sin(t)))
    _draw_lines(draw, _rot(pts, cx, cy, ang), fill, max(2, width))


MARK_RENDERERS["circle"] = draw_circle


def draw_mark(img, bbox, style, rng, fill=(0, 0, 0, 255), base_width=3):
    """Draw a mark of `style` at bbox (px, top-left). Returns nothing."""
    draw = ImageDraw.Draw(img)
    x, y, w, h = [float(v) for v in bbox]
    c = (int(fill[0]), int(fill[1]), int(fill[2]), 255)
    width = max(1, int(base_width * rng.uniform(0.7, 1.4)))
    renderer = MARK_RENDERERS.get(style, draw_check)
    renderer(draw, (x, y, w, h), rng, c, width)