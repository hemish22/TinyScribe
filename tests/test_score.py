from tinyscribe.score import score_page, score_records

TARGET = {
    "fields": {"name": "मीना शर्मा", "dob": "09/05/2000", "email": ""},
    "checkboxes": {"male": False, "female": True},
    "tables": {"edu": {"r1": ["CBSE", "2015"], "r2": ["", ""]}},
}


def test_perfect():
    c = score_page(TARGET, TARGET)
    assert (c.tp, c.fp, c.fn) == (4, 0, 0)
    assert c.f1 == 1.0 and c.checkbox_acc == 1.0 and c.valid == 1


def test_normalised_match_counts():
    pred = {"fields": {"name": "मीना  शर्मा", "dob": "०९-०५-२०००", "email": ""},
            "checkboxes": {"male": False, "female": True},
            "tables": {"edu": {"r1": ["cbse", "2015"], "r2": ["", ""]}}}
    assert score_page(TARGET, pred).f1 == 1.0


def test_wrong_value_is_fp_and_fn():
    pred = {**TARGET, "fields": {**TARGET["fields"], "name": "मीनी शर्मा"}}
    c = score_page(TARGET, pred)
    assert (c.tp, c.fp, c.fn) == (3, 1, 1)


def test_hallucinated_blank_is_fp():
    pred = {**TARGET, "fields": {**TARGET["fields"], "email": "x@y.com"}}
    c = score_page(TARGET, pred)
    assert (c.tp, c.fp, c.fn) == (4, 1, 0)


def test_missing_key_is_fn():
    pred = {**TARGET, "fields": {"name": "मीना शर्मा", "email": ""}}
    c = score_page(TARGET, pred)
    assert (c.tp, c.fp, c.fn) == (3, 0, 1)


def test_unparsed_output_scores_empty():
    c = score_page(TARGET, None)
    assert (c.tp, c.fp, c.fn, c.valid, c.cb_right) == (0, 0, 4, 0, 0)


def test_schema_failure_scores_empty():
    schema = {"type": "object", "required": ["fields"], "properties": {"fields": {"type": "object"}}}
    c = score_page(TARGET, {"checkboxes": {}}, schema)
    assert c.valid == 0 and c.tp == 0


def test_checkbox_wrong_type_is_wrong():
    pred = {**TARGET, "checkboxes": {"male": "false", "female": 1}}
    assert score_page(TARGET, pred).cb_right == 0


def test_report_by_script():
    rep = score_records([{"script": "hindi", "target": TARGET, "pred": TARGET},
                         {"script": "tamil", "target": TARGET, "pred": None}]).as_dict()
    assert rep["by_script"]["hindi"]["field_f1"] == 1.0
    assert rep["by_script"]["tamil"]["field_f1"] == 0.0
    assert rep["overall"]["valid_json"] == 0.5
