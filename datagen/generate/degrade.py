#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/degrade.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""IndicFormBench Phase 4 — degradation splits (digital realism path).

Each perturbation is a toggleable function applied to a clean rendered page;
three severity levels light/medium/heavy, per-page parameter log. Ground truth
is INHERITED from the clean instance — this script NEVER re-emits GT.

Perturbations: JPEG compression, Gaussian blur, ±5° rotation, brightness
gradient, mild perspective warp, speckle/grain.

Usage:
    python src/generate/degrade.py data/rendered/eci_form6_en_s0005_p*.png --out data/rendered/degraded
    python src/generate/degrade.py x.png --levels medium,heavy --no-rotate --no-speckle
    python src/generate/degrade.py --regen --seed 5
"""
import argparse
import csv
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SELF = Path(__file__).resolve()
PROJECT_ROOT = SELF.parents[2]

NOISE_PERM = 20260909

SEVERITY = {
    "light":  {"jpeg_q": 88, "blur": 0.3, "rotate": 1.2, "gradient": 0.06,
               "warp": 0.006, "speckle": 0.0015},
    "medium": {"jpeg_q": 70, "blur": 0.8, "rotate": 2.5, "gradient": 0.13,
               "warp": 0.012, "speckle": 0.004},
    "heavy":  {"jpeg_q": 45, "blur": 1.6, "rotate": 4.5, "gradient": 0.24,
               "warp": 0.022, "speckle": 0.009},
}
ORDER = ["jpeg", "blur", "rotate", "gradient", "warp", "speckle"]

# --- Perturbation functions ---


def jpeg_compress(img: Image.Image, rng, s: dict):
    q = max(15, min(97, int(s["jpeg_q"] + rng.uniform(-4, 4))))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q, subsampling=0)
    buf.seek(0)
    return Image.open(buf).convert(img.mode), {"q": q}


def gaussian_blur(img: Image.Image, rng, s: dict):
    r = max(0.1, s["blur"] * rng.uniform(0.7, 1.3))
    return img.filter(ImageFilter.GaussianBlur(radius=r)), {"radius": round(r, 3)}


def rotate(img: Image.Image, rng, s: dict):
    ang = round(rng.uniform(-min(5.0, s["rotate"] * 1.4), min(5.0, s["rotate"] * 1.4)), 2)
    return img.rotate(ang, resample=Image.BICUBIC, fillcolor=(250, 249, 247)), {"angle": ang}


def brightness_gradient(img: Image.Image, rng, s: dict):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    amp = round(s["gradient"] * rng.uniform(0.7, 1.3), 4)
    tilt = rng.uniform(-0.25, 0.25)
    xx = np.linspace(0, 1, w)[None, :]
    yy = np.linspace(0, 1, h)[:, None]
    ramp = amp * (yy + tilt * (xx - 0.5))
    arr = arr * np.clip(1.0 - ramp[:, :, None], 0.55, 1.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), {"amp": amp, "tilt": round(tilt, 3)}


def _homography(src, dst):
    """Solve 3x3 H mapping src->dst for 4 corresponding (x,y) points (same order)."""
    A, b = [], []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y]); b.append(xp)
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y]); b.append(yp)
    h = np.linalg.solve(np.asarray(A, float), np.asarray(b, float))
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def perspective_warp(img: Image.Image, rng, s: dict):
    """Mild in-plane perspective: corners shift a few px (~0.3-2% of canvas),
    winding/order PRESERVED (UL, UR, LR, BL), centre fixed -> no flip/transpose."""
    w, h = img.size
    d = int(min(w, h) * s["warp"] * rng.uniform(0.6, 1.4)) or 1
    corners = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
    dst = [(x + rng.uniform(-d, d), y + rng.uniform(-d, d)) for x, y in corners]
    M = np.linalg.inv(_homography(corners, dst))  # output->input mapping for PIL
    M /= M[2, 2]
    coeffs = tuple(M[:2].ravel()) + (M[2, 0], M[2, 1])
    out = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC,
                        fillcolor=(255, 255, 255))
    return out, {"d": d}


def speckle(img: Image.Image, rng, s: dict):
    arr = np.asarray(img.convert("RGB"))
    h, w, ch = arr.shape
    p = s["speckle"]
    if p <= 0:
        return img, {"p": 0}
    nrng = np.random.default_rng((rng.randrange(2 ** 32) + NOISE_PERM) & 0xFFFFFFFF)
    mask = nrng.random((h, w)) < p
    grain = nrng.normal(0, 18 * (p / 0.02) + 4, size=(h, w, ch))
    out = arr.astype(np.float32) + grain
    salt = mask & (arr.sum(axis=-1) < 720) & (nrng.random((h, w)) < 0.5)
    out[salt] = [255, 255, 255]
    out[~salt & mask] = 0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), {"p": p}


PERTURBATIONS = {
    "jpeg": jpeg_compress, "blur": gaussian_blur, "rotate": rotate,
    "gradient": brightness_gradient, "warp": perspective_warp, "speckle": speckle,
}


def degrade_page(img: Image.Image, level: str, seed: int, enabled: list[str], rng=None):
    """Apply enabled perturbations in fixed ORDER; return (degraded, applied_params)."""
    s = SEVERITY[level]
    out = img
    applied = {}
    rng = rng or random.Random((seed * 7919 + NOISE_PERM) & 0xFFFFFFFF)
    for name in ORDER:
        if name in enabled:
            out, desc = PERTURBATIONS[name](out, rng, s)
            applied[name] = desc
    return out, applied


# --- Orientation guard (advisory) ---


def orientation_guard(clean: Image.Image, degraded: Image.Image):
    """No-vision check that a page is NOT rotated ~90deg, flipped or transposed.

    Advisory mode:
      - REJECT  if best is rot90 or rot180 AND margin > 0.05 (true catastrophic rotation)
      - WARN    if identity wins but margin is thin, or blank/constant pages, or near-ties
      - PASS    if identity wins cleanly

    Returns (metrics_dict, verdict) where verdict is pass/warn/reject."""
    if degraded.size != clean.size:
        return {"shape_mismatch": [list(clean.size), list(degraded.size)]}, "reject"

    def _edges(im):
        e = np.asarray(im.convert("L").resize((256, 256)).filter(
            ImageFilter.FIND_EDGES), dtype=float)
        return e - e.mean()

    d = _edges(degraded)
    variants = {
        "identity": _edges(clean),
        "fliplr":   _edges(clean.transpose(Image.FLIP_LEFT_RIGHT)),
        "flipud":   _edges(clean.transpose(Image.FLIP_TOP_BOTTOM)),
        "rot90":    _edges(clean.rotate(90, expand=True)),
        "rot180":   _edges(clean.rotate(180)),
    }

    def _corr(a, b):
        denom = a.std() * b.std() * a.size
        return float((a * b).sum() / denom) if denom > 0 else 0.0

    scores = {k: _corr(d, v) for k, v in variants.items()}
    ident = scores["identity"]
    best = max(scores, key=scores.get)
    best_score = scores[best]
    runner = max(v for k, v in scores.items() if k != best)
    margin = round(best_score - runner, 3)

    # Blank / constant page: edge-map is flat, all correlations ≈ 0 → warn, never reject
    if ident > 0.85 and runner < 0.05:
        verdict = "warn"
    # Best != identity, strong margin → REJECT (true flip or catastrophic rotation)
    elif best != "identity" and margin > 0.05:
        verdict = "reject"
    # Best != identity but thin margin (near-tie) → WARN (false-positive zone)
    elif best != "identity":
        verdict = "warn"
    # Identity wins but signal is weak → WARN
    elif ident <= 0.08:
        verdict = "warn"
    # Identity wins but margin is thin → WARN
    elif margin <= 0.03:
        verdict = "warn"
    else:
        verdict = "pass"

    out = {f"ncc_{k}": round(s, 3) for k, s in scores.items()}
    out.update({"best": best, "margin": margin})
    return out, verdict


# --- Manifest helpers for --regen ---

_INSTANCE_RE = re.compile(r"^(.*)_s(\d+)$")


def _clean_name_from_gt(gt_path: str) -> str:
    """Extract instance_id (clean stem) from a GT file path."""
    return Path(gt_path).stem


def load_manifest_clean_pages(manifest_csv: Path) -> list[tuple[str, Path]]:
    """Return [(instance_id, clean_png_path), ...] from volume_manifest.csv.
    instance_id = template_id_sNNNN, clean_png = each page PNG."""
    rows = []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            inst = r["instance_id"]
            for pp in r["png_paths"].split("|"):
                rows.append((inst, Path(pp)))
    return rows


# --- CLI ---

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="clean rendered PNGs (or glob)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "rendered" / "degraded"))
    ap.add_argument("--levels", default="light,medium,heavy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regen", action="store_true",
                    help="re-degrade ALL pages from volume_manifest.csv (replaces explicit inputs)")
    for name in ORDER:
        ap.add_argument(f"--no-{name}", action="store_true", help=f"disable {name} perturbation")
    args = ap.parse_args(argv)

    # Collect clean pages
    if args.regen:
        manifest = PROJECT_ROOT / "data" / "rendered" / "volume_manifest.csv"
        if not manifest.exists():
            print(f"manifest not found: {manifest}"); return 2
        entries = load_manifest_clean_pages(manifest)
        srcs = [p for _, p in entries]
        instance_map = {str(p): inst for inst, p in entries}
    else:
        srcs = []
        for p in args.inputs:
            pp = Path(p)
            srcs += sorted(pp.parent.glob(pp.name)) if any(ch in pp.name for ch in "*?") else [pp]
        instance_map = {}

    levels = [l for l in args.levels.split(",") if l in SEVERITY]
    enabled = [n for n in ORDER if not getattr(args, f"no_{n}")]
    if not levels:
        print("no valid levels"); return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: degrade all pages ---
    log = {"seed": args.seed, "levels": levels, "enabled": enabled, "pages": []}
    registry = []  # (clean_img, deg_img, src, level, out_png)
    for src in srcs:
        img = Image.open(src).convert("RGB")
        for lv in levels:
            rng = random.Random(args.seed * 7919 + sum(src.name.encode()) * 104729)
            deg, applied = degrade_page(img, lv, args.seed, enabled, rng)
            sub = out_dir / lv
            sub.mkdir(parents=True, exist_ok=True)
            out_png = sub / f"{src.stem}__{lv}.png"
            deg.save(out_png)
            log["pages"].append({
                "source": str(src), "level": lv, "output": str(out_png),
                "size_wh": [deg.size[0], deg.size[1]], "applied": applied,
            })
            registry.append((img, deg, src, lv, out_png))
    (out_dir / "degradation_log.json").write_text(json.dumps(log, indent=2))
    print(f"wrote {len(log['pages'])} degraded pages under {out_dir}")

    # --- Phase 2: advisory orientation guard + guard_report.csv ---
    report_path = out_dir / "guard_report.csv"
    fieldnames = ["template", "seed", "page", "level", "best",
                  "ncc_identity", "ncc_fliplr", "ncc_flipud",
                  "ncc_rot90", "ncc_rot180", "margin", "verdict"]
    counts = {"pass": 0, "warn": 0, "reject": 0}

    with open(report_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for clean, deg, src, lv, out_png in registry:
            m, verdict = orientation_guard(clean, deg)

            # Parse template/seed/page from filename: {instance}__{level}.png
            stem = src.stem  # e.g. eci_form6_en_s0005_p1
            # instance_id: everything before __ or _sNNNN
            inst_id = instance_map.get(str(src), "")
            if not inst_id:
                # try to reconstruct from stem
                m2 = _INSTANCE_RE.match(stem.rsplit("_p", 1)[0] if "_p" in stem else stem)
                inst_id = m2.group(1) if m2 else stem.rsplit("_s", 1)[0] if "_s" in stem else stem

            # seed: extract from instance_id (template_sNNNN)
            seed_match = re.search(r"_s(\d+)$", inst_id)
            seed_val = int(seed_match.group(1)) if seed_match else 0

            # page number from stem
            page_match = re.search(r"_p(\d+)$", stem)
            page_val = int(page_match.group(1)) if page_match else 1

            writer.writerow({
                "template": inst_id.rsplit("_s", 1)[0] if "_s" in inst_id else inst_id,
                "seed": seed_val,
                "page": page_val,
                "level": lv,
                "best": m.get("best", ""),
                "ncc_identity": m.get("ncc_identity", ""),
                "ncc_fliplr": m.get("ncc_fliplr", ""),
                "ncc_flipud": m.get("ncc_flipud", ""),
                "ncc_rot90": m.get("ncc_rot90", ""),
                "ncc_rot180": m.get("ncc_rot180", ""),
                "margin": m.get("margin", ""),
                "verdict": verdict,
            })
            counts[verdict] += 1
            if verdict == "reject":
                print(f"guard REJECT {stem}__{lv}  best={m['best']} margin={m['margin']}")
            elif verdict == "warn":
                print(f"guard WARN   {stem}__{lv}  best={m['best']} ident={m.get('ncc_identity', '?')} margin={m['margin']}")

    # --- Phase 3: negative test (mirrored image MUST be rejected) ---
    if registry:
        clean0 = registry[0][0]
        flipped = clean0.transpose(Image.FLIP_LEFT_RIGHT)
        nm, n_verdict = orientation_guard(clean0, flipped)
        neg_ok = n_verdict == "reject"
        print(f"NEGATIVE TEST {'OK-REJECTED' if neg_ok else 'FAILED'}: mirrored image "
              f"(best={nm.get('best')} ident={nm.get('ncc_identity')} fliplr={nm.get('ncc_fliplr')} "
              f"verdict={n_verdict})")
        if not neg_ok:
            print(f"  UNEXPECTED: mirrored image not rejected — threshold too lax")
            counts["reject"] += 1

    # --- Summary ---
    total = sum(counts.values())
    print(f"\nguard_report: {total} pages | pass={counts['pass']} warn={counts['warn']} reject={counts['reject']}")
    print(f"report written to {report_path}")

    if counts["reject"] > (1 if not neg_ok else 0):
        print(f"REJECT pages found — investigate before shipping")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
