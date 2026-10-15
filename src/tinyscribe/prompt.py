"""The one prompt every model gets (zero-shot, fine-tuned, every quant). Changing it changes
every number in the results table, so treat it as frozen once the baseline is scored."""
from __future__ import annotations

from tinyscribe.schema import PageLayout

PROMPT_VERSION = 1
LABEL_CHARS = 60

INSTRUCTION = """Read this filled form page and return its contents as JSON.
- fields: each field's value exactly as written on the page, in its original script; "" if blank.
- checkboxes: true if the box is marked, otherwise false.
- tables: for each row, the filled-in cells left to right; "" for a blank cell.
Keys and the printed labels they belong to:"""


def build_prompt(lay: PageLayout) -> str:
    lines = [INSTRUCTION]

    def label(s: str) -> str:
        s = " ".join(s.split())
        return s if len(s) <= LABEL_CHARS else s[: LABEL_CHARS - 1] + "…"

    if lay.fields:
        lines.append("fields:")
        lines += [f"  {f['id']}: {label(f['label_text'])}" for f in lay.fields]
    if lay.checkboxes:
        lines.append("checkboxes:")
        lines += [f"  {b['id']}: {label(b['label_text'])}" for b in lay.checkboxes]
    if lay.tables:
        lines.append("tables:")
        lines += [f"  {tid}: rows {', '.join(f'r{r}' for r in rows)}" for tid, rows in lay.tables.items()]
    return "\n".join(lines)
