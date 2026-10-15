import json
import random
import sys
from pathlib import Path

import jsonschema
import pytest

from tinyscribe.prompt import build_prompt
from tinyscribe.schema import (
    SCHEMAS_DIR,
    json_schema,
    load_template,
    page_layout,
    page_layouts,
    parse_schema_name,
    target,
    template_ids,
)

REPO = Path(__file__).resolve().parents[1]


def test_every_template_has_pages():
    ids = template_ids()
    assert len(ids) == 30
    for tid in ids:
        assert page_layouts(load_template(tid)), tid


def test_committed_schemas_match_templates():
    """schemas/ must be regenerated (make schemas) whenever a template changes."""
    for tid in template_ids():
        for lay in page_layouts(load_template(tid)):
            on_disk = json.loads((SCHEMAS_DIR / f"{lay.schema_name}.schema.json").read_text())
            assert on_disk == json_schema(lay), lay.schema_name


def test_schema_name_roundtrip():
    assert parse_schema_name("eci_form7_hi_p0") == ("eci_form7_hi", 0)
    assert page_layout("eci_form7_hi", 0).schema_name == "eci_form7_hi_p0"


def test_prompt_lists_every_key():
    lay = page_layout("synth_job_application_ta", 0)
    p = build_prompt(lay)
    for f in lay.fields:
        assert f["id"] in p
    for tid in lay.tables:
        assert tid in p


@pytest.mark.parametrize("tid", ["eci_form6_bn", "eci_form7_ta", "synth_household_hi", "pfrda_apy_reg_en"])
def test_rendered_targets_validate(tid):
    """Render a real instance (printed) and check every page target against its schema."""
    from PIL import features
    if not features.check("raqm"):
        pytest.skip("no raqm (macOS: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib)")
    sys.path.insert(0, str(REPO / "datagen"))
    from generate.fill_engine import render_flat
    tpl_dir = REPO / "datagen" / "templates"
    pages, gt, _ = render_flat(tpl_dir / f"{tid}.json", tpl_dir / f"{tid}.pdf", seed=random.Random(tid).randint(0, 99))
    for lay in page_layouts(load_template(tid)):
        t = target(lay, gt)
        jsonschema.validate(t, json_schema(lay))
        assert lay.page < len(pages)
