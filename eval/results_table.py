#!/usr/bin/env python3
"""Print the README results table from runs/*/scores.json (+ runs/*/bench.json from the Mac).

    python eval/results_table.py            # all runs, ordered by meta.json "order" then name
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COLS = ["Model", "Size", "Field F1", "English", "Hindi", "Tamil", "Bengali", "Valid JSON",
        "s / page (Metal)", "s / page (CPU)", "Peak memory"]


def pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def row(run: Path) -> tuple[float, list[str]] | None:
    sc_path = run / "scores.json"
    if not sc_path.exists():
        return None
    sc = json.loads(sc_path.read_text())
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    bench = json.loads((run / "bench.json").read_text()) if (run / "bench.json").exists() else {}
    by = sc.get("by_script", {})
    get = lambda s: pct(by.get(s, {}).get("field_f1"))
    cells = [meta.get("label", run.name), meta.get("size", ""), pct(sc["overall"]["field_f1"]),
             get("english"), get("hindi"), get("tamil"), get("bengali"), pct(sc["overall"]["valid_json"]),
             bench.get("s_per_page_metal", ""), bench.get("s_per_page_cpu", ""), bench.get("peak_memory", "")]
    return meta.get("order", 999), [str(c) for c in cells]


def main() -> int:
    rows = sorted((r for r in (row(p) for p in sorted((REPO / "runs").glob("*"))) if r),
                  key=lambda t: (t[0], t[1][0]))
    print("| " + " | ".join(COLS) + " |")
    print("| " + " | ".join("---" for _ in COLS) + " |")
    for _, cells in rows:
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
