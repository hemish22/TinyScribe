"""Bring any input image to the pixels the models were trained and scored on.

Every dataset page was resized once to a 1536 px long side and saved as JPEG quality 95
(datagen/split.yaml: image_long_side, jpeg_quality). The CLI and the demo do the same to a user's
photo or scan, so it reaches the model at the scale it learned.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

LONG_SIDE = 1536
JPEG_QUALITY = 95


def prepare(image: Path | str | bytes) -> bytes:
    """JPEG bytes at LONG_SIDE px long side. A JPEG already at that size (a dataset page) is passed
    through unchanged, so re-encoding never changes the pixels the scorer saw."""
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    img = Image.open(io.BytesIO(data))
    if img.format == "JPEG" and max(img.size) == LONG_SIDE and not _rotated(img):
        return data
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = LONG_SIDE / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def _rotated(img: Image.Image) -> bool:
    return img.getexif().get(0x0112, 1) != 1   # EXIF orientation tag, as phone cameras write it
