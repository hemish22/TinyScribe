#!/usr/bin/env python3
# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/synth_table_forms.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Synthetic table forms for T2 (scope v3, Track A).

    .venv/bin/python3 src/generate/synth_table_forms.py [--check]

None of the 23 sourced government forms has a record-style table (a header row over
several data rows): their ruled grids are key-value layouts, already scored as fields.
So the T2 forms are built in-house, modelled on two common Indian form tables, in English,
Hindi, Bengali and Tamil:

  synth_household_<lang>        a household information form: applicant fields, two
                                checkbox groups, and a family-members table (S.No. | Name |
                                Relation | Age | Gender | Occupation; rows filled top-down)
  synth_job_application_<lang>  a contractual-job application: applicant fields, gender
                                checkboxes, an education table with a merged two-level
                                "Marks" header and printed row labels, and a work-experience
                                table with a merged "Period" header

Each form is drawn at 200 DPI with the same shaped fonts as the fill engine and saved as
an image PDF (data/templates/<id>.pdf) plus its template JSON, whose boxes come straight
from the drawing (no annotation step). The forms also carry ordinary fields and checkboxes,
so they count toward T1/T3 too. Rows are added to data/template_manifest.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import fitz
import jsonschema
from PIL import Image, ImageDraw, ImageFont

SELF = Path(__file__).resolve()
PROJECT_ROOT = SELF.parents[2]
sys.path.insert(0, str(SELF.parent.parent))
from generate.fill_engine import FONT_LATIN, FONT_LATIN_BOLD, _Mixed  # noqa: E402

W, H = 1654, 2339                 # A4 @ 200 DPI
LM, RM = 110, 1544                # left / right margin
INK = (20, 20, 20)
SOURCE = "https://github.com/hemish22/IndicFormBench/blob/main/src/generate/synth_table_forms.py"
LANGS = {"en": ("english", "latin"), "hi": ("hindi", "devanagari"),
         "bn": ("bengali", "bengali"), "ta": ("tamil", "tamil")}

# ---- printed text, per language ------------------------------------------------------------
HOUSEHOLD = {
    "en": dict(title="HOUSEHOLD INFORMATION FORM", sub="(To be filled by the head of the household)",
               head="Name of Head of Household", address="Address", mobile="Mobile No.",
               ration="Ration Card No.", category="Category",
               cats=["General", "OBC", "SC", "ST"], house="Type of House",
               houses=["Kutcha", "Semi-pucca", "Pucca"], caption="Details of Family Members",
               cols=["S.No.", "Name", "Relation with Head", "Age", "Gender", "Occupation"],
               place="Place", date="Date", sign="Signature of Head of Household"),
    "hi": dict(title="परिवार सूचना प्रपत्र", sub="(परिवार के मुखिया द्वारा भरा जाए)",
               head="परिवार के मुखिया का नाम", address="पता", mobile="मोबाइल नंबर",
               ration="राशन कार्ड संख्या", category="श्रेणी",
               cats=["सामान्य", "अन्य पिछड़ा वर्ग", "अनुसूचित जाति", "अनुसूचित जनजाति"], house="मकान का प्रकार",
               houses=["कच्चा", "अर्ध-पक्का", "पक्का"], caption="परिवार के सदस्यों का विवरण",
               cols=["क्र.सं.", "नाम", "मुखिया से संबंध", "आयु", "लिंग", "व्यवसाय"],
               place="स्थान", date="दिनांक", sign="परिवार के मुखिया के हस्ताक्षर"),
    "bn": dict(title="পরিবারের তথ্য ফর্ম", sub="(পরিবারের প্রধান দ্বারা পূরণ করতে হবে)",
               head="পরিবারের প্রধানের নাম", address="ঠিকানা", mobile="মোবাইল নম্বর",
               ration="রেশন কার্ড নম্বর", category="শ্রেণি",
               cats=["সাধারণ", "ওবিসি", "তফসিলি জাতি", "তফসিলি উপজাতি"], house="বাড়ির ধরন",
               houses=["কাঁচা", "আধা-পাকা", "পাকা"], caption="পরিবারের সদস্যদের বিবরণ",
               cols=["ক্রমিক নং", "নাম", "প্রধানের সাথে সম্পর্ক", "বয়স", "লিঙ্গ", "পেশা"],
               place="স্থান", date="তারিখ", sign="পরিবারের প্রধানের স্বাক্ষর"),
    "ta": dict(title="குடும்பத் தகவல் படிவம்", sub="(குடும்பத் தலைவரால் நிரப்பப்பட வேண்டும்)",
               head="குடும்பத் தலைவரின் பெயர்", address="முகவரி", mobile="கைபேசி எண்",
               ration="குடும்ப அட்டை எண்", category="பிரிவு",
               cats=["பொது", "பிற்படுத்தப்பட்டோர்", "ஆதிதிராவிடர்", "பழங்குடியினர்"], house="வீட்டின் வகை",
               houses=["குடிசை", "ஓட்டு வீடு", "கான்கிரீட் வீடு"], caption="குடும்ப உறுப்பினர்களின் விவரம்",
               cols=["வ.எண்", "பெயர்", "தலைவருடன் உறவு", "வயது", "பாலினம்", "தொழில்"],
               place="இடம்", date="தேதி", sign="குடும்பத் தலைவரின் கையொப்பம்"),
}
JOB = {
    "en": dict(title="APPLICATION FORM FOR CONTRACTUAL APPOINTMENT",
               sub="(Attach self-attested copies of all certificates)",
               name="Name of Applicant", father="Father's / Husband's Name", dob="Date of Birth",
               mobile="Mobile No.", address="Address for Correspondence", gender="Gender",
               genders=["Male", "Female", "Transgender"], edu="Educational Qualifications",
               edu_cols=["Examination", "Board / University", "Year of Passing", "Marks"],
               marks=["Obtained", "Maximum", "Percentage"],
               exams=["10th (Secondary)", "12th (Higher Secondary)", "Graduation", "Post Graduation"],
               exp="Work Experience", exp_cols=["S.No.", "Name of Employer", "Post Held", "Period"],
               period=["From", "To"], place="Place", date="Date", sign="Signature of Applicant"),
    "hi": dict(title="संविदा नियुक्ति हेतु आवेदन पत्र",
               sub="(सभी प्रमाणपत्रों की स्व-सत्यापित प्रतियाँ संलग्न करें)",
               name="आवेदक का नाम", father="पिता / पति का नाम", dob="जन्म तिथि",
               mobile="मोबाइल नंबर", address="पत्राचार का पता", gender="लिंग",
               genders=["पुरुष", "महिला", "ट्रांसजेंडर"], edu="शैक्षणिक योग्यता",
               edu_cols=["परीक्षा", "बोर्ड / विश्वविद्यालय", "उत्तीर्ण वर्ष", "अंक"],
               marks=["प्राप्तांक", "पूर्णांक", "प्रतिशत"],
               exams=["10वीं (हाईस्कूल)", "12वीं (इंटरमीडिएट)", "स्नातक", "स्नातकोत्तर"],
               exp="कार्य अनुभव", exp_cols=["क्र.सं.", "नियोक्ता का नाम", "धारित पद", "अवधि"],
               period=["से", "तक"], place="स्थान", date="दिनांक", sign="आवेदक के हस्ताक्षर"),
    "bn": dict(title="চুক্তিভিত্তিক নিয়োগের জন্য আবেদনপত্র",
               sub="(সমস্ত শংসাপত্রের স্ব-প্রত্যয়িত প্রতিলিপি সংযুক্ত করুন)",
               name="আবেদনকারীর নাম", father="পিতা / স্বামীর নাম", dob="জন্ম তারিখ",
               mobile="মোবাইল নম্বর", address="যোগাযোগের ঠিকানা", gender="লিঙ্গ",
               genders=["পুরুষ", "মহিলা", "রূপান্তরকামী"], edu="শিক্ষাগত যোগ্যতা",
               edu_cols=["পরীক্ষা", "বোর্ড / বিশ্ববিদ্যালয়", "পাশের বছর", "নম্বর"],
               marks=["প্রাপ্ত", "পূর্ণমান", "শতাংশ"],
               exams=["মাধ্যমিক", "উচ্চ মাধ্যমিক", "স্নাতক", "স্নাতকোত্তর"],
               exp="কাজের অভিজ্ঞতা", exp_cols=["ক্রমিক নং", "নিয়োগকর্তার নাম", "পদ", "সময়কাল"],
               period=["থেকে", "পর্যন্ত"], place="স্থান", date="তারিখ", sign="আবেদনকারীর স্বাক্ষর"),
    "ta": dict(title="ஒப்பந்த அடிப்படையிலான பணி நியமனத்திற்கான விண்ணப்பம்",
               sub="(அனைத்துச் சான்றிதழ்களின் சுய சான்றொப்பமிட்ட நகல்களை இணைக்கவும்)",
               name="விண்ணப்பதாரரின் பெயர்", father="தந்தை / கணவர் பெயர்", dob="பிறந்த தேதி",
               mobile="கைபேசி எண்", address="தொடர்பு முகவரி", gender="பாலினம்",
               genders=["ஆண்", "பெண்", "மூன்றாம் பாலினம்"], edu="கல்வித் தகுதி",
               edu_cols=["தேர்வு", "வாரியம் / பல்கலைக்கழகம்", "தேர்ச்சி பெற்ற ஆண்டு", "மதிப்பெண்கள்"],
               marks=["பெற்றவை", "மொத்தம்", "சதவீதம்"],
               exams=["பத்தாம் வகுப்பு", "பன்னிரண்டாம் வகுப்பு", "இளங்கலை", "முதுகலை"],
               exp="பணி அனுபவம்", exp_cols=["வ.எண்", "பணியமர்த்தியவர் பெயர்", "வகித்த பதவி", "காலம்"],
               period=["முதல்", "வரை"], place="இடம்", date="தேதி", sign="விண்ணப்பதாரரின் கையொப்பம்"),
}


class Page:
    """One blank form page plus the template elements drawn on it."""

    def __init__(self, script: str):
        self.script = script
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.checkboxes, self.fields, self.tables = [], [], []
        self._fonts = {}

    def _runs(self, s: str, size: int, bold: bool):
        key = (size, bold)
        if key not in self._fonts:
            latin = ImageFont.truetype(FONT_LATIN_BOLD if bold else FONT_LATIN, size,
                                       layout_engine=ImageFont.Layout.RAQM)
            self._fonts[key] = latin if self.script == "latin" else _Mixed(self.script, size)
        f = self._fonts[key]
        return [(s, f)] if self.script == "latin" else [(t, ff) for t, ff in f.runs(s)]

    def width(self, s: str, size: int, bold=False) -> float:
        return sum(self.d.textlength(t, font=f) for t, f in self._runs(s, size, bold))

    def text(self, x, y, s: str, size: int, bold=False) -> float:
        """Draw s with its top-of-line at y; returns the x after it. Bold for the Indic
        fonts (Regular only) is a 1 px stroke."""
        stroke = 1 if bold and self.script != "latin" else 0
        for t, f in self._runs(s, size, bold):
            self.d.text((x, y), t, font=f, fill=INK, stroke_width=stroke, stroke_fill=INK)
            x += self.d.textlength(t, font=f)
        return x

    def wrap(self, s: str, size: int, maxw: float, bold=False) -> list:
        lines, cur = [], ""
        for word in s.split():
            cand = f"{cur} {word}".strip()
            if cur and self.width(cand, size, bold) > maxw:
                lines.append(cur)
                cur = word
            else:
                cur = cand
        return lines + [cur]

    def centered(self, cx, y, s, size, bold=False):
        self.text(cx - self.width(s, size, bold) / 2, y, s, size, bold)

    # ---- form elements ---------------------------------------------------------------------
    def header(self, title: str, sub: str) -> int:
        size = 44 if self.width(title, 44, True) < RM - LM else 38
        self.centered(W / 2, 120, title, size, bold=True)
        self.centered(W / 2, 192, sub, 28)
        self.d.line([LM, 252, RM, 252], fill=INK, width=4)
        self.d.line([LM, 260, RM, 260], fill=INK, width=1)
        return 300

    def field(self, y, num, label, fid, vtype, lines=1, dotted=False) -> int:
        """`num. label :` then an answer line (or `lines` lines) to the right margin."""
        x = self.text(LM, y, f"{num}.", 30)
        x = self.text(LM + 48, y, label, 30)
        x = self.text(x + 6, y, ":", 30) + 22
        line_ys = [y + 44 + 66 * i for i in range(lines)]
        for ly in line_ys:
            if dotted:
                for dx in range(int(x), RM, 10):
                    self.d.ellipse([dx, ly - 1, dx + 3, ly + 2], fill=INK)
            else:
                self.d.line([x, ly, RM, ly], fill=INK, width=2)
        top = y - 6 if lines == 1 else y - 8
        self.fields.append({"id": fid, "label_text": label, "label_script": self.script,
                            "value_bbox": [int(x), top, int(RM - x), int(line_ys[-1] - 4 - top)],
                            "value_type": vtype, "page": 0})
        return line_ys[-1] + 44

    def comb(self, y, num, label, fid, vtype, n=10, pitch=50, side=52) -> int:
        """`num. label :` then n touching boxes, one character each."""
        self.text(LM, y, f"{num}.", 30)
        x = self.text(LM + 48, y, label, 30)
        x0 = int(self.text(x + 6, y, ":", 30) + 24)
        top = y - 8
        for i in range(n):
            self.d.rectangle([x0 + i * pitch, top, x0 + (i + 1) * pitch, top + side], outline=INK, width=3)
        self.fields.append({"id": fid, "label_text": label, "label_script": self.script,
                            "value_bbox": [x0, top, n * pitch + 3, side + 3], "value_type": vtype, "page": 0})
        return top + side + 40

    def checks(self, y, num, label, group, options, ids) -> int:
        self.text(LM, y, f"{num}.", 30)
        x = self.text(LM + 48, y, label, 30)
        x = self.text(x + 6, y, ":", 30) + 40
        for opt, cid in zip(options, ids):
            w = 36 + 14 + self.width(opt, 30) + 46
            if x + w > RM:                       # wrap the options onto the next line
                y, x = y + 62, LM + 48
            self.d.rectangle([x, y + 2, x + 36, y + 38], outline=INK, width=3)
            self.checkboxes.append({"id": cid, "bbox": [int(x), y + 2, 37, 37], "label_text": opt,
                                    "label_script": self.script, "group": group, "page": 0})
            self.text(x + 50, y, opt, 30)
            x += w
        return y + 76

    def table(self, y, num, caption, tid, widths, header, body, row_h, row_fill, hdr_h=64) -> int:
        """header: list of header rows, each a list of (text, rowspan, colspan) in column order
        (cells covered by a rowspan from above are skipped). body: rows of cells, each either
        ('text', s) printed, ('center', s) printed centred, or ('value', value_type)."""
        self.text(LM, y, f"{num}.", 30)
        self.text(LM + 48, y, caption, 30, bold=True)
        top = y + 56
        xs = [LM]
        for w_ in widths:
            xs.append(xs[-1] + w_)
        nh = len(header)
        ys = [top + hdr_h * i for i in range(nh + 1)]
        for _ in body:
            ys.append(ys[-1] + row_h)
        cells, occupied = [], set()
        for r, row in enumerate(header):
            c = 0
            for text, rs, cs in row:
                while (r, c) in occupied:
                    c += 1
                for rr in range(r, r + rs):
                    for cc in range(c, c + cs):
                        occupied.add((rr, cc))
                x0, x1, y0, y1 = xs[c], xs[c + cs], ys[r], ys[r + rs]
                self.d.rectangle([x0, y0, x1, y1], fill=(236, 236, 236))
                lines = self.wrap(text, 26, x1 - x0 - 16, bold=True)
                lh = 34
                ty = (y0 + y1) / 2 - lh * len(lines) / 2 - 2
                for i, ln in enumerate(lines):
                    self.centered((x0 + x1) / 2, ty + i * lh, ln, 26, bold=True)
                cells.append({"row": r, "col": c, "rowspan": rs, "colspan": cs, "text": text})
                c += cs
        for i, row in enumerate(body):
            r = nh + i
            for c, (kind, val) in enumerate(row):
                x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
                cell = {"row": r, "col": c, "rowspan": 1, "colspan": 1, "text": None}
                if kind == "value":
                    cell.update(bbox=[x0 + 8, y0 + 8, x1 - x0 - 16, y1 - y0 - 16], value_type=val)
                else:
                    cell["text"] = val
                    lines = self.wrap(val, 28, x1 - x0 - 20)
                    ty = (y0 + y1) / 2 - 36 * len(lines) / 2 - 2
                    for k, ln in enumerate(lines):
                        if kind == "center":
                            self.centered((x0 + x1) / 2, ty + k * 36, ln, 28)
                        else:
                            self.text(x0 + 12, ty + k * 36, ln, 28)
                cells.append(cell)
        # ruling: thin inner lines (skipping merged spans), heavier outer frame
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                own = next((k for k in cells if k["row"] <= r < k["row"] + k["rowspan"]
                            and k["col"] <= c < k["col"] + k["colspan"]), None)
                if own["row"] + own["rowspan"] - 1 == r:       # bottom edge of its cell
                    self.d.line([xs[c], ys[r + 1], xs[c + 1], ys[r + 1]], fill=INK, width=2)
                if own["col"] + own["colspan"] - 1 == c:       # right edge of its cell
                    self.d.line([xs[c + 1], ys[r], xs[c + 1], ys[r + 1]], fill=INK, width=2)
        self.d.rectangle([xs[0], ys[0], xs[-1], ys[-1]], outline=INK, width=3)
        self.d.line([xs[0], ys[nh], xs[-1], ys[nh]], fill=INK, width=3)
        self.tables.append({"id": tid, "bbox": [xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0]], "page": 0,
                            "row_fill": row_fill, "cells": cells})
        return ys[-1] + 50

    def footer(self, y, t, place_num, date_num):
        y = self.field(y, place_num, t["place"], "f_place", "city")
        self.field(y, date_num, t["date"], "f_date", "date")
        # the date/place answer lines run to the margin; the signature sits under them
        sw = self.width(t["sign"], 28)
        self.d.line([RM - max(sw, 420), y + 190, RM, y + 190], fill=INK, width=2)
        self.text(RM - sw, y + 202, t["sign"], 28)


def household(lang: str) -> Page:
    t = HOUSEHOLD[lang]
    p = Page(LANGS[lang][1])
    y = p.header(t["title"], t["sub"])
    y = p.field(y, 1, t["head"], "f_head_name", "name", dotted=True)
    y = p.field(y, 2, t["address"], "f_address", "address", dotted=True)
    y = p.comb(y, 3, t["mobile"], "f_mobile", "phone")
    y = p.field(y, 4, t["ration"], "f_ration_card", "serial", dotted=True)
    y = p.checks(y, 5, t["category"], "category", t["cats"],
                 ["cb_category_general", "cb_category_obc", "cb_category_sc", "cb_category_st"])
    y = p.checks(y, 6, t["house"], "house_type", t["houses"],
                 ["cb_house_kutcha", "cb_house_semi_pucca", "cb_house_pucca"])
    body = [[("center", str(i + 1)), ("value", "name"), ("value", "relation"), ("value", "age"),
             ("value", "gender"), ("value", "occupation")] for i in range(8)]
    y = p.table(y + 10, 7, t["caption"], "tbl_family_members", [110, 420, 270, 120, 190, 324],
                [[(c, 1, 1) for c in t["cols"]]], body, row_h=76, row_fill="prefix", hdr_h=84)
    p.footer(y + 10, t, 8, 9)
    return p


def job_application(lang: str) -> Page:
    t = JOB[lang]
    p = Page(LANGS[lang][1])
    y = p.header(t["title"], t["sub"])
    y = p.field(y, 1, t["name"], "f_applicant_name", "name")
    y = p.field(y, 2, t["father"], "f_father_husband_name", "name")
    y = p.field(y, 3, t["dob"], "f_dob", "date")
    y = p.comb(y, 4, t["mobile"], "f_mobile", "phone")
    y = p.field(y, 5, t["address"], "f_address", "address")
    y = p.checks(y, 6, t["gender"], "gender", t["genders"],
                 ["cb_gender_male", "cb_gender_female", "cb_gender_transgender"])
    ec = t["edu_cols"]
    header = [[(ec[0], 2, 1), (ec[1], 2, 1), (ec[2], 2, 1), (ec[3], 1, 3)],
              [(m, 1, 1) for m in t["marks"]]]
    body = [[("text", ex), ("value", "board"), ("value", "year"), ("value", "marks"),
             ("value", "max_marks"), ("value", "percent")] for ex in t["exams"]]
    y = p.table(y + 6, 7, t["edu"], "tbl_education", [290, 420, 170, 180, 170, 204],
                header, body, row_h=84, row_fill="prefix", hdr_h=58)
    xc = t["exp_cols"]
    header = [[(xc[0], 2, 1), (xc[1], 2, 1), (xc[2], 2, 1), (xc[3], 1, 2)],
              [(q, 1, 1) for q in t["period"]]]
    body = [[("center", str(i + 1)), ("value", "employer"), ("value", "designation"), ("value", "date"),
             ("value", "date")] for i in range(3)]
    y = p.table(y, 8, t["exp"], "tbl_experience", [110, 520, 344, 230, 230],
                header, body, row_h=78, row_fill="prefix", hdr_h=58)
    p.footer(y, t, 9, 10)
    return p


FORMS = {"synth_household": household, "synth_job_application": job_application}


def build(check_only=False) -> list:
    schema = json.loads((PROJECT_ROOT / "datagen" / "template_schema.json").read_text())  # TinyScribe
    out = []
    for stem, fn in FORMS.items():
        for lang, (language, script) in LANGS.items():
            tid = f"{stem}_{lang}"
            p = fn(lang)
            tpl = {"schema_version": "0", "template_id": tid, "source_url": SOURCE, "licence": "other",
                   "language": language, "fillable": False,
                   "checkboxes": p.checkboxes, "fields": p.fields, "tables": p.tables}
            jsonschema.validate(tpl, schema)
            bottom = max([f["value_bbox"][1] + f["value_bbox"][3] for f in p.fields] +
                         [t["bbox"][1] + t["bbox"][3] for t in p.tables])
            assert bottom < H - 100, f"{tid}: content runs off the page ({bottom})"
            out.append((tid, language, p, tpl))
            if check_only:
                continue
            pdf_path = PROJECT_ROOT / "datagen" / "templates" / f"{tid}.pdf"  # TinyScribe
            doc = fitz.open()
            page = doc.new_page(width=W * 72 / 200, height=H * 72 / 200)
            import io
            buf = io.BytesIO()
            p.img.save(buf, format="PNG")
            page.insert_image(page.rect, stream=buf.getvalue())
            doc.save(str(pdf_path), deflate=True)
            doc.close()
            lines = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in p.checkboxes)
            flines = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in p.fields)
            tlines = ",\n".join("    " + json.dumps(e, ensure_ascii=False) for e in p.tables)
            head = {k: v for k, v in tpl.items() if k not in ("checkboxes", "fields", "tables")}
            body = json.dumps(head, ensure_ascii=False, indent=2)[:-2]
            text = (body + ',\n  "checkboxes": [\n' + lines + '\n  ],\n  "fields": [\n' + flines
                    + '\n  ],\n  "tables": [\n' + tlines + "\n  ]\n}\n")
            assert json.loads(text) == tpl
            (PROJECT_ROOT / "datagen" / "templates" / f"{tid}.json").write_text(text)  # TinyScribe
            print(f"wrote {tid}: {len(p.fields)} fields, {len(p.checkboxes)} checkboxes, "
                  f"{len(p.tables)} tables ({sum(len(t['cells']) for t in p.tables)} cells)")
    return out


def update_manifest(forms):
    path = PROJECT_ROOT / "data" / "template_manifest.csv"
    rows = list(csv.DictReader(path.open()))
    fields = list(rows[0].keys())
    have = {r["template_id"] for r in rows}
    code = {"english": "en", "hindi": "hi", "bengali": "bn", "tamil": "ta"}
    for tid, language, p, _ in forms:
        if tid in have:
            continue
        rows.append({"template_id": tid, "filename": f"{tid}.pdf", "source_url": SOURCE,
                     "licence": "other (synthetic, made in-house for T2)", "language": code[language],
                     "fillable": "False", "has_checkbox": "True",
                     "notes": "synthetic table form (scope v3 Track A); src/generate/synth_table_forms.py"})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="build and validate only; write nothing")
    a = ap.parse_args(argv)
    forms = build(check_only=a.check)
    # TinyScribe: no template manifest CSV; update_manifest() is not called
    return 0


if __name__ == "__main__":
    sys.exit(main())
