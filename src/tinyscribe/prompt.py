"""The one prompt every model gets (zero-shot, fine-tuned, every quant). Changing it changes
every number in the results table, so treat it as frozen once the baseline is scored.

v1 -> v2 (Oct 16, 2026, before any training): v1 said only "the filled-in cells left to right",
and the zero-shot 9B copied each table's printed first column (row labels like "10th
(Secondary)" or serial numbers) as a value, shifting every cell by one. v2 names each value
column by its printed header and each row by its printed label, and says labels are not values.
"""
from __future__ import annotations

from tinyscribe.schema import PageLayout

PROMPT_VERSION = 2
LABEL_CHARS = 60

INSTRUCTION = """Read this filled form page and return its contents as JSON.
- fields: each field's value exactly as written on the page, in its original script; "" if blank.
- checkboxes: true if the box is marked, otherwise false.
- tables: for each listed row, only the values written into its blank cells, in the column order listed; "" for a blank cell. Printed headers, row labels and serial numbers are not values.
Keys and the printed labels they belong to:"""


def _label(s: str, n: int = LABEL_CHARS) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _table_lines(tid: str, rows: dict[int, list[dict]], cells: list[dict]) -> list[str]:
    printed = [c for c in cells if not c.get("value_type") and c.get("text")]
    first_value_row = min(rows)
    first_row = rows[first_value_row]

    def header(col: int) -> str:
        hs = [h for h in printed if h["row"] < first_value_row
              and h["col"] <= col < h["col"] + h.get("colspan", 1)]
        return " ".join(_label(h["text"], 30) for h in sorted(hs, key=lambda h: h["row"])) or f"column {col}"

    def row_label(r: int, first_col: int) -> str:
        ls = [c for c in printed if c["row"] <= r < c["row"] + c.get("rowspan", 1) and c["col"] < first_col]
        return " ".join(_label(c["text"], 30) for c in sorted(ls, key=lambda c: c["col"]))

    out = [f"  {tid}: each row = [{' | '.join(header(c['col']) for c in first_row)}]"]
    for r, row in rows.items():
        lab = row_label(r, row[0]["col"])
        out.append(f"    r{r}" + (f" (printed: {lab})" if lab else ""))
    return out


def build_prompt(lay: PageLayout) -> str:
    lines = [INSTRUCTION]
    if lay.fields:
        lines.append("fields:")
        lines += [f"  {f['id']}: {_label(f['label_text'])}" for f in lay.fields]
    if lay.checkboxes:
        lines.append("checkboxes:")
        lines += [f"  {b['id']}: {_label(b['label_text'])}" for b in lay.checkboxes]
    if lay.tables:
        lines.append("tables:")
        for tid, rows in lay.tables.items():
            lines += _table_lines(tid, rows, lay.table_cells.get(tid, []))
    return "\n".join(lines)
