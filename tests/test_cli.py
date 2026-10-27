import io
import json

from PIL import Image
from typer.testing import CliRunner

from tinyscribe.cli import app, labeled, schema_names, short
from tinyscribe.image import LONG_SIDE, prepare
from tinyscribe.schema import page_layout


def _jpeg(size, **save) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "JPEG", **save)
    return buf.getvalue()


def test_prepare_resizes_to_training_scale():
    out = Image.open(io.BytesIO(prepare(_jpeg((3000, 2000)))))
    assert max(out.size) == LONG_SIDE and out.format == "JPEG"
    small = Image.open(io.BytesIO(prepare(_jpeg((600, 900)))))
    assert small.size == (1024, LONG_SIDE)


def test_prepare_passes_dataset_pages_through():
    page = _jpeg((1086, LONG_SIDE), quality=95)
    assert prepare(page) == page


def test_prepare_applies_exif_rotation():
    img = Image.new("RGB", (2000, 1000), "white")
    exif = img.getexif()
    exif[0x0112] = 6                     # rotate 90 degrees on display, as phones write it
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    out = Image.open(io.BytesIO(prepare(buf.getvalue())))
    assert out.size == (768, LONG_SIDE)  # portrait after rotation


def test_schema_names_cover_every_page():
    names = schema_names()
    assert len(names) == len(set(names)) == 53
    assert "eci_form7_hi_p0" in names


def test_labeled_pairs_in_page_order():
    lay = page_layout("eci_form7_hi", 0)
    data = {"fields": {lay.fields[0]["id"]: "नेहा"}, "checkboxes": {lay.checkboxes[0]["id"]: True}}
    pairs = labeled(lay, data)
    assert pairs[0] == (lay.fields[0]["label_text"], "नेहा")
    assert (lay.checkboxes[0]["label_text"], "yes") in pairs
    assert len(pairs) == len(lay.fields) + len(lay.checkboxes)


def test_short_label():
    assert short("a  b") == "a b"
    assert len(short("x" * 100)) == 48


def test_unknown_schema_suggests(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(_jpeg((100, 100)))
    r = CliRunner().invoke(app, ["parse", str(img), "--schema", "eci_form7_hi_p9"])
    assert r.exit_code == 2
    assert "eci_form7_hi_p0" in r.output


def test_schemas_command_filters_language():
    r = CliRunner().invoke(app, ["schemas", "--language", "tamil"])
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert lines and all(" tamil " in x for x in lines)


def test_demo_rows_compare_with_truth():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from server import rows_for
    lay = page_layout("eci_form7_hi", 0)
    f = lay.fields[0]["id"]
    target = {"fields": {x["id"]: "" for x in lay.fields} | {f: "१२३"},
              "checkboxes": {b["id"]: False for b in lay.checkboxes}}
    pred = json.loads(json.dumps(target)) | {"fields": target["fields"] | {f: "123"}}
    rows = rows_for(lay, pred, target)
    assert rows[0]["ok"]                 # Devanagari digits normalize to ASCII
    assert all(r["ok"] for r in rows)
    assert "ok" not in rows_for(lay, pred, None)[0]
