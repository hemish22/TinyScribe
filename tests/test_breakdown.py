import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from breakdown import group_counts, order_key  # noqa: E402

T = {"fields": {"a": "x", "b": "y"}, "checkboxes": {}, "tables": {}}


def row(i, script, hw, deg):
    return {"id": f"p{i}", "script": script, "handwriting": hw, "degradation": deg,
            "form_type": "f", "target": T}


def test_groups_split_by_handwriting_and_degradation():
    manifest = {r["id"]: r for r in [row(1, "hindi", True, "clean"), row(2, "hindi", False, "heavy")]}
    preds = [{"id": "p1", "pred": T},                                   # perfect
             {"id": "p2", "pred": {"fields": {"a": "x", "b": "z"}}},     # one wrong
             {"id": "unknown", "pred": T}]                               # not in manifest: skipped
    g = group_counts(preds, manifest)
    hw = g["Script x handwriting"]
    assert hw["hindi · handwritten"].f1 == 1.0
    assert (hw["hindi · printed"].tp, hw["hindi · printed"].fp, hw["hindi · printed"].fn) == (1, 1, 1)
    assert g["Degradation"]["clean"].pages == 1 and g["Degradation"]["heavy"].pages == 1
    assert g["Form type"]["f"].pages == 2


def test_order_scripts_then_degradations():
    labels = ["heavy", "tamil · printed", "clean", "english · printed", "zzz"]
    assert sorted(labels, key=order_key) == ["english · printed", "tamil · printed", "clean", "heavy", "zzz"]
