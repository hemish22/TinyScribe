"""Value normalisation applied to both prediction and ground truth before exact-match scoring."""
from __future__ import annotations

import re
import unicodedata

# Devanagari, Bengali and Tamil digits -> ASCII
_DIGITS = {}
for _base in (0x0966, 0x09E6, 0x0BE6):
    for _i in range(10):
        _DIGITS[_base + _i] = ord("0") + _i
# zero-width characters vary between renderers and keyboards; drop them
_DROP = {0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None}
_TABLE = {**_DIGITS, **_DROP}

_DATE_DMY = re.compile(r"^(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})$")
_DATE_YMD = re.compile(r"^(\d{4})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})$")
_SPACE = re.compile(r"\s+")
_TIGHT = re.compile(r"\s*([-/])\s*")       # "Bhopal - 462001" == "Bhopal-462001"
_COMMA = re.compile(r"\s*,\s*")             # "a ,b" == "a, b"
_EDGE = ".,;:"


def normalize(value) -> str:
    """NFC, Indic digits to ASCII, drop zero-width chars, collapse whitespace, casefold,
    canonical spacing around - / and ',', trim edge punctuation, dates to YYYY-MM-DD."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFC", str(value)).translate(_TABLE)
    s = _SPACE.sub(" ", s).strip().casefold()
    m = _DATE_DMY.match(s)
    if m:
        d, mo, y = (int(x) for x in m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _DATE_YMD.match(s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"
    s = _TIGHT.sub(r"\1", s)
    s = _COMMA.sub(", ", s)
    return s.strip(_EDGE + " ")
