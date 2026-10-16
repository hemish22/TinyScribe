#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base model and save a full 16-bit Hugging Face checkpoint,
ready for quant/convert.sh.

    python train/merge.py runs/train-9b/adapter models/hf/ft-9b
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from unsloth import FastVisionModel


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    adapter, out = Path(argv[0]), Path(argv[1])
    model, processor = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=False)
    model.save_pretrained_merged(str(out), processor, save_method="merged_16bit")
    # keep the base model's chat template and preprocessing files byte-identical
    base = Path(model.peft_config["default"].base_model_name_or_path)
    for name in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        if (base / name).exists():
            shutil.copy2(base / name, out / name)
    print("merged ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
