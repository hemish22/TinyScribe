"""Page layouts, output JSON Schemas and training targets, all derived from template JSON.

One sample is one page image. Its JSON output holds only the items printed on that page:

    {"fields":     {field_id: str},             # "" when the field is blank
     "checkboxes": {checkbox_id: bool},
     "tables":     {table_id: {"r<row>": [str, ...]}}}   # value cells only, in column order

Keys follow template order, which is also the order llama.cpp's grammar enforces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO / "datagen" / "templates"
SCHEMAS_DIR = REPO / "schemas"

# longest value a field may hold; bounds generation so a looping model cannot run on forever
LONG_TYPES = {"address", "text", "other"}
MAX_LEN_LONG, MAX_LEN = 400, 80   # free-text fields hold up to ~40 words


@dataclass
class PageLayout:
    template_id: str
    language: str
    page: int
    fields: list[dict] = field(default_factory=list)
    checkboxes: list[dict] = field(default_factory=list)
    # table_id -> {row: [value-cell dicts in column order]}
    tables: dict[str, dict[int, list[dict]]] = field(default_factory=dict)

    @property
    def schema_name(self) -> str:
        return f"{self.template_id}_p{self.page}"

    def is_empty(self) -> bool:
        return not (self.fields or self.checkboxes or self.tables)


def load_template(template_id: str) -> dict:
    return json.loads((TEMPLATES_DIR / f"{template_id}.json").read_text())


def template_ids() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.json"))


def page_layouts(tpl: dict) -> list[PageLayout]:
    """Every page of a template that carries at least one field, checkbox or table value cell."""
    pages: dict[int, PageLayout] = {}

    def get(p: int) -> PageLayout:
        return pages.setdefault(p, PageLayout(tpl["template_id"], tpl["language"], p))

    for f in tpl["fields"]:
        get(int(f.get("page", 0))).fields.append(f)
    for b in tpl["checkboxes"]:
        get(int(b.get("page", 0))).checkboxes.append(b)
    for tb in tpl.get("tables", []):
        rows: dict[int, list[dict]] = {}
        for c in sorted(tb["cells"], key=lambda c: (c["row"], c["col"])):
            if c.get("value_type") and c.get("bbox"):
                rows.setdefault(c["row"], []).append(c)
        if rows:
            get(int(tb.get("page", 0))).tables[tb["id"]] = rows
    return [pages[p] for p in sorted(pages) if not pages[p].is_empty()]


def page_layout(template_id: str, page: int) -> PageLayout:
    for lay in page_layouts(load_template(template_id)):
        if lay.page == page:
            return lay
    raise KeyError(f"{template_id} has no content on page {page}")


def parse_schema_name(name: str) -> tuple[str, int]:
    """'eci_form7_hi_p0' -> ('eci_form7_hi', 0)."""
    tid, _, p = name.rpartition("_p")
    return tid, int(p)


def json_schema(lay: PageLayout) -> dict:
    def text(max_len: int, label: str) -> dict:
        return {"type": "string", "maxLength": max_len, "description": label}

    props: dict = {}
    if lay.fields:
        props["fields"] = _obj({
            f["id"]: text(MAX_LEN_LONG if f["value_type"] in LONG_TYPES else MAX_LEN, f["label_text"])
            for f in lay.fields})
    if lay.checkboxes:
        props["checkboxes"] = _obj({b["id"]: {"type": "boolean", "description": b["label_text"]}
                                    for b in lay.checkboxes})
    if lay.tables:
        props["tables"] = _obj({
            tid: _obj({f"r{r}": {"type": "array", "minItems": len(cells), "maxItems": len(cells),
                                 "items": {"type": "string", "maxLength": MAX_LEN}}
                       for r, cells in rows.items()})
            for tid, rows in lay.tables.items()})
    schema = _obj(props)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = lay.schema_name
    return schema


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def target(lay: PageLayout, gt: dict) -> dict:
    """The exact JSON a model should produce for this page of a ground-truth instance."""
    values = {f["id"]: f.get("value") or "" for f in gt["fields"]}
    checked = {b["id"]: bool(b.get("checked")) for b in gt["checkboxes"]}
    cells = {(t["id"], c["row"], c["col"]): c.get("text") or ""
             for t in gt.get("tables", []) for c in t["cells"] if c.get("value_type")}
    out: dict = {}
    if lay.fields:
        out["fields"] = {f["id"]: values[f["id"]] for f in lay.fields}
    if lay.checkboxes:
        out["checkboxes"] = {b["id"]: checked[b["id"]] for b in lay.checkboxes}
    if lay.tables:
        out["tables"] = {tid: {f"r{r}": [cells[(tid, r, c["col"])] for c in row] for r, row in rows.items()}
                         for tid, rows in lay.tables.items()}
    return out


def write_all(out_dir: Path = SCHEMAS_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tid in template_ids():
        for lay in page_layouts(load_template(tid)):
            p = out_dir / f"{lay.schema_name}.schema.json"
            p.write_text(json.dumps(json_schema(lay), ensure_ascii=False, indent=2) + "\n")
            written.append(p)
    return written


def load_schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text())


if __name__ == "__main__":
    paths = write_all()
    print(f"wrote {len(paths)} page schemas to {SCHEMAS_DIR}")
