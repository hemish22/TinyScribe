"""Scoring: field F1 (main metric), per-script F1, checkbox accuracy, valid-JSON rate.

Every text value (form fields and table value cells) is one scored item. After normalize():
    TP: gt != "" and pred == gt
    FN: gt != "" and pred != gt
    FP: pred != "" and pred != gt
A wrong value on a filled field counts as both FN and FP; inventing text in a blank field is FP.
A page whose output does not parse or fails its schema scores every key as an empty prediction."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import jsonschema

from tinyscribe.normalize import normalize

SCRIPTS = ("english", "hindi", "bengali", "tamil")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    cb_right: int = 0
    cb_total: int = 0
    pages: int = 0
    valid: int = 0

    def add(self, o: Counts) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else 1.0

    @property
    def checkbox_acc(self) -> float:
        return self.cb_right / self.cb_total if self.cb_total else 1.0

    @property
    def valid_rate(self) -> float:
        return self.valid / self.pages if self.pages else 0.0


def text_items(doc: dict | None) -> dict[str, str]:
    """Flatten fields and table cells into {key: value}."""
    out: dict[str, str] = {}
    if not isinstance(doc, dict):
        return out
    for k, v in (doc.get("fields") or {}).items():
        out[f"f:{k}"] = v if isinstance(v, str) else ""
    for tid, rows in (doc.get("tables") or {}).items():
        for r, cells in (rows or {}).items():
            for i, v in enumerate(cells or []):
                out[f"t:{tid}:{r}:{i}"] = v if isinstance(v, str) else ""
    return out


def score_page(target: dict, pred: dict | None, schema: dict | None = None) -> Counts:
    c = Counts(pages=1)
    ok = isinstance(pred, dict)
    if ok and schema is not None:
        try:
            jsonschema.validate(pred, schema)
        except jsonschema.ValidationError:
            ok = False
    c.valid = int(ok)
    gt_items, pr_items = text_items(target), text_items(pred if ok else None)
    for k, g in gt_items.items():
        g, p = normalize(g), normalize(pr_items.get(k, ""))
        if g and p == g:
            c.tp += 1
        else:
            if g:
                c.fn += 1
            if p:
                c.fp += 1
    gt_cb = target.get("checkboxes") or {}
    pr_cb = (pred.get("checkboxes") or {}) if ok else {}
    for k, g in gt_cb.items():
        c.cb_total += 1
        c.cb_right += int(pr_cb.get(k) is g)
    return c


@dataclass
class Report:
    overall: Counts = field(default_factory=Counts)
    by_script: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))

    def as_dict(self) -> dict:
        def d(c: Counts) -> dict:
            return {"field_f1": round(c.f1, 4), "checkbox_acc": round(c.checkbox_acc, 4),
                    "valid_json": round(c.valid_rate, 4), "pages": c.pages,
                    "tp": c.tp, "fp": c.fp, "fn": c.fn}
        return {"overall": d(self.overall), "by_script": {s: d(self.by_script[s]) for s in SCRIPTS
                                                          if s in self.by_script}}


def score_records(records) -> Report:
    """records: iterable of dicts with 'script', 'target', 'pred' and optionally 'schema'."""
    rep = Report()
    for r in records:
        c = score_page(r["target"], r.get("pred"), r.get("schema"))
        rep.overall.add(c)
        rep.by_script[r["script"]].add(c)
    return rep
