#!/usr/bin/env python3
"""Build the TinyScribe page dataset from the templates and datagen/split.yaml.

    python datagen/build.py                       # full build into data/
    python datagen/build.py --scale 0.01 --out data_smoke   # tiny run for checks
    python datagen/build.py --workers 32

Each template instance is rendered at 200 DPI (fill_engine.render_flat), each content page is
degraded (degrade.degrade_page) at a sampled level, resized to image_long_side and saved as JPEG.
One manifest row per page holds the training target. Resumable: finished instances are skipped.

Output:
    <out>/<split>/images/<sample_id>.jpg
    <out>/<split>/rows/<instance_id>.jsonl     per-instance rows (resume markers)
    <out>/manifest.jsonl                       all rows, written at the end
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image, features

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "datagen"))
sys.path.insert(0, str(REPO / "src"))

from generate import handwriting as hw
from generate.degrade import ORDER, degrade_page
from generate.fill_engine import render_flat
from generate.hw_bank import EMNIST_BANK, IIIT_BANK

from tinyscribe.schema import TEMPLATES_DIR, load_template, page_layouts, target

LANG_SUFFIX = {"en", "hi", "bn", "ta"}
SCRIPTS = ("english", "hindi", "bengali", "tamil")


def form_type(template_id: str) -> str:
    head, _, tail = template_id.rpartition("_")
    return head if tail in LANG_SUFFIX else template_id


def load_recipe(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def plan_jobs(recipe: dict, scale: float) -> list[dict]:
    """One job per template instance, with page budgets spread evenly per script then per template."""
    test_types = set(recipe["test_form_types"])
    by_split: dict[str, dict[str, list[str]]] = {"train": {}, "test": {}}
    for p in sorted(TEMPLATES_DIR.glob("*.json")):
        tpl = load_template(p.stem)
        split = "test" if form_type(p.stem) in test_types else "train"
        by_split[split].setdefault(tpl["language"], []).append(p.stem)
    jobs = []
    for split, langs in by_split.items():
        per_script = recipe["pages"][split] * scale / len(SCRIPTS)
        for ti, (lang, tids) in enumerate(sorted(langs.items())):
            for tj, tid in enumerate(tids):
                n_pages = len(page_layouts(load_template(tid)))
                n_inst = max(1, math.ceil(per_script / len(tids) / n_pages))
                base = recipe["seed_base"][split] + (ti * 100 + tj) * 10000
                jobs += [{"split": split, "template_id": tid, "language": lang, "seed": base + k}
                         for k in range(n_inst)]
    return jobs


def _level(rng: random.Random, mix: dict) -> str:
    return rng.choices(list(mix), weights=list(mix.values()), k=1)[0]


def run_job(job: dict, recipe: dict, out: Path, hw_ok: bool) -> list[dict]:
    split, tid, seed = job["split"], job["template_id"], job["seed"]
    rows_path = out / split / "rows" / f"{tid}_s{seed}.jsonl"
    if rows_path.exists():
        return [json.loads(x) for x in rows_path.read_text().splitlines()]
    rng = random.Random(seed)
    handwriting = (hw_ok and job["language"] in recipe["handwriting_scripts"]
                   and rng.random() < recipe["handwriting_fraction"])
    hw.set_split(split)
    hw.STATS.clear()
    pdf = TEMPLATES_DIR / f"{tid}.pdf"
    pages, gt, truncated = render_flat(TEMPLATES_DIR / f"{tid}.json", pdf, seed=seed, handwriting=handwriting)
    # handwritten fields of this instance by source: real word/glyph images vs handwriting font
    hw_fields = {k: hw.STATS[k] for k in ("real", "font")} if handwriting else None
    img_dir = out / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lay in page_layouts(load_template(tid)):
        sid = f"{tid}_s{seed}_p{lay.page}"
        level = _level(rng, recipe["degradation"][split])
        img = pages[lay.page].convert("RGB")
        if level != "clean":
            img, _ = degrade_page(img, level, seed * 10 + lay.page, ORDER)
        scale = recipe["image_long_side"] / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        rel = f"{split}/images/{sid}.jpg"
        img.save(out / rel, quality=recipe["jpeg_quality"])
        rows.append({"id": sid, "image": rel, "split": split, "template_id": tid,
                     "form_type": form_type(tid), "script": job["language"], "page": lay.page,
                     "schema": lay.schema_name, "seed": seed, "degradation": level,
                     "handwriting": handwriting, "hw_fields": hw_fields, "truncated": bool(truncated),
                     "target": target(lay, gt)})
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def mark_calib(rows: list[dict], n: int, seed: int) -> None:
    rng = random.Random(seed)
    for script in SCRIPTS:
        pool = [r for r in rows if r["split"] == "train" and r["script"] == script]
        for r in rng.sample(pool, min(len(pool), n // len(SCRIPTS))):
            r["calib"] = True
    for r in rows:
        r.setdefault("calib", False)


def check(rows: list[dict], recipe: dict) -> None:
    test_types = set(recipe["test_form_types"])
    leaked = {r["form_type"] for r in rows if r["split"] == "train"} & test_types
    assert not leaked, f"test form types in train: {leaked}"
    assert all(r["form_type"] in test_types for r in rows if r["split"] == "test")
    for split in ("train", "test"):
        c = Counter(r["script"] for r in rows if r["split"] == split)
        print(f"{split}: {sum(c.values())} pages  " + "  ".join(f"{s}={c[s]}" for s in SCRIPTS))
    c = Counter((r["split"], r["degradation"]) for r in rows)
    print("degradation:", dict(sorted(c.items())))
    c = Counter(r["script"] for r in rows if r["handwriting"])
    print("handwritten pages by script:", dict(c))
    tot = Counter()
    for r in rows:
        if r.get("hw_fields"):
            tot.update({f"{r['script']}/{k}": v for k, v in r["hw_fields"].items()})
    print("handwritten fields by source (per instance, counted once per page):", dict(sorted(tot.items())))
    print("handwritten pages:", sum(r["handwriting"] for r in rows),
          " calib pages:", sum(r["calib"] for r in rows),
          " pages from truncated instances:", sum(r["truncated"] for r in rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default=str(REPO / "datagen" / "split.yaml"))
    ap.add_argument("--out", default=str(REPO / "data"))
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the page budget (smoke runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-handwriting", action="store_true")
    a = ap.parse_args(argv)

    if not features.check("raqm"):
        sys.exit("Pillow has no raqm: Indic text would render unshaped. On macOS run with "
                 "DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib (brew install fribidi harfbuzz).")
    hw_ok = not a.no_handwriting and EMNIST_BANK.is_dir() and IIIT_BANK.is_dir()
    if not hw_ok:
        print("handwriting OFF (no banks at tools/hw_datasets or --no-handwriting): all pages printed",
              file=sys.stderr)

    recipe = load_recipe(Path(a.recipe))
    out = Path(a.out)
    jobs = plan_jobs(recipe, a.scale)
    print(f"{len(jobs)} instances to render with {a.workers} workers -> {out}")
    rows: list[dict] = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_job, j, recipe, out, hw_ok): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} instances", flush=True)
    rows.sort(key=lambda r: (r["split"], r["id"]))
    mark_calib(rows, round(recipe["calib_pages"] * a.scale), recipe["seed_base"]["train"])
    check(rows, recipe)
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"manifest: {out / 'manifest.jsonl'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
