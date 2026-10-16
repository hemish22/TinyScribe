#!/usr/bin/env python3
"""Field F1 of one or more scored runs, split by script x handwriting, by degradation and by form type.

    python eval/breakdown.py zero-shot-2b ft-2b
    python eval/breakdown.py smoke-2b-q4km-mac --manifest data_smoke/manifest.jsonl

Reads runs/<name>/preds.jsonl and joins it to the manifest by page id. Prints markdown tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinyscribe.score import Counts, score_page  # noqa: E402

SCRIPTS = ["english", "hindi", "tamil", "bengali"]
DEGRADATIONS = ["clean", "light", "medium", "heavy"]

GROUPINGS = {
    "Script x handwriting": lambda r: f"{r['script']} · {'handwritten' if r.get('handwriting') else 'printed'}",
    "Degradation": lambda r: r.get("degradation", "?"),
    "Form type": lambda r: r.get("form_type", "?"),
}


def order_key(label: str):
    for i, s in enumerate(SCRIPTS):
        if label.startswith(s):
            return (0, i, label)
    if label in DEGRADATIONS:
        return (1, DEGRADATIONS.index(label), label)
    return (2, 0, label)


def group_counts(preds: list[dict], manifest: dict[str, dict]) -> dict[str, dict[str, Counts]]:
    """{grouping title: {group label: Counts}} over the pages present in both preds and manifest."""
    out: dict[str, dict[str, Counts]] = {t: defaultdict(Counts) for t in GROUPINGS}
    for p in preds:
        row = manifest.get(p["id"])
        if row is None:
            continue
        c = score_page(row["target"], p.get("pred"))
        for title, key in GROUPINGS.items():
            out[title][key(row)].add(c)
    return out


def load_preds(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "preds.jsonl").read_text().splitlines() if x.strip()]


def table(title: str, names: list[str], counts: dict[str, dict[str, dict[str, Counts]]]) -> str:
    labels = sorted({g for n in names for g in counts[n][title]}, key=order_key)
    lines = [f"### {title}", "", "| Group | Pages | " + " | ".join(names) + " |",
             "| --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for g in labels:
        pages = max(counts[n][title][g].pages for n in names if g in counts[n][title])
        cells = [f"{100 * counts[n][title][g].f1:.1f}" if g in counts[n][title] else "" for n in names]
        lines.append(f"| {g} | {pages} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names under runs/")
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.jsonl"))
    a = ap.parse_args(argv)
    manifest = {}
    for line in Path(a.manifest).read_text().splitlines():
        r = json.loads(line)
        manifest[r["id"]] = r
    counts = {n: group_counts(load_preds(REPO / "runs" / n), manifest) for n in a.runs}
    print("Field F1 (%) by group.\n")
    print("\n\n".join(table(t, a.runs, counts) for t in GROUPINGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
