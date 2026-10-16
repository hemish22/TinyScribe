#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3.5 (9B or 2B) on the TinyScribe train split.

    python train/sft.py --config train/config_9b.yaml
    python train/sft.py --config train/config_2b.yaml --smoke     # 40 samples, 10 steps

Under torchrun, gradient accumulation is divided by the GPU count so the effective batch stays
batch_size * grad_accum. Not usable yet: with Unsloth 2026.10.6 + Qwen3.5, data-parallel runs fail
in backward ("A different number of tensors was saved during the original forward and
recomputation"), so both models train on one GPU each.

Each sample is exactly what llama-server sees at inference: the page image plus the frozen prompt
(tinyscribe.prompt) in the model's chat template with thinking switched off, so the assistant turn
opens with an empty <think></think> block. The target is the page's JSON in compact form
(no spaces or newlines), then <|im_end|>. Loss is on the target tokens only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from unsloth import FastVisionModel, is_bf16_supported  # noqa: E402  (unsloth must import first)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

from tinyscribe.prompt import PROMPT_VERSION, build_prompt  # noqa: E402
from tinyscribe.schema import page_layout  # noqa: E402

END = "<|im_end|>"


def compact(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def load_split(manifest: Path, dev_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    # truncated pages (text cut off by the box) have targets that don't match the image
    train = [r for r in rows if r["split"] == "train" and not r.get("truncated")]
    rng = random.Random(seed)
    dev_ids = {r["id"] for r in rng.sample(train, dev_n)}
    return [r for r in train if r["id"] not in dev_ids], [r for r in train if r["id"] in dev_ids]


class Collator:
    """Builds input_ids / attention_mask / pixel values / labels for a batch of manifest rows."""

    def __init__(self, processor, data_root: Path, max_len: int, pad_multiple: int = 1):
        self.p = processor
        self.root = data_root
        self.max_len = max_len
        # pad each batch to a multiple of this many tokens: kernels are compiled per sequence length,
        # so a few length buckets keep the compile caches warm instead of compiling on most steps
        self.pad_multiple = pad_multiple
        self._prompts: dict[tuple[str, int], str] = {}

    def prompt_text(self, r: dict) -> str:
        key = (r["template_id"], r["page"])
        if key not in self._prompts:
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": build_prompt(page_layout(*key))}]}]
            self._prompts[key] = self.p.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self._prompts[key]

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [self.prompt_text(r) for r in batch]
        targets = [compact(r["target"]) + END + "\n" for r in batch]
        images = [Image.open(self.root / r["image"]).convert("RGB") for r in batch]
        enc = self.p(text=[a + b for a, b in zip(prompts, targets)], images=images,
                     return_tensors="pt", padding=True, pad_to_multiple_of=self.pad_multiple)
        labels = torch.full_like(enc["input_ids"], -100)
        tok = self.p.tokenizer
        for i, t in enumerate(targets):
            n = len(tok(t, add_special_tokens=False)["input_ids"])
            # the target is the last n real tokens, whichever side the padding is on
            real = enc["attention_mask"][i].nonzero().squeeze(-1)
            last = real[-n:]
            labels[i, last] = enc["input_ids"][i, last]
        enc["labels"] = labels
        if enc["input_ids"].shape[1] > self.max_len:
            raise ValueError(f"sequence of {enc['input_ids'].shape[1]} tokens > max_len {self.max_len}")
        return enc


def check_boundary(collator: Collator, row: dict) -> None:
    """The target's tokens must be the same alone as at the end of the full sequence."""
    tok = collator.p.tokenizer
    t = compact(row["target"]) + END + "\n"
    alone = tok(t, add_special_tokens=False)["input_ids"]
    full = tok(collator.prompt_text(row) + t, add_special_tokens=False)["input_ids"]
    assert full[-len(alone):] == alone, "prompt/target token boundary merges; label mask would be off"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="40 samples, 10 steps, separate output dir")
    a = ap.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    main_proc = rank == 0
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = REPO / (cfg["output_dir"] + ("-smoke" if a.smoke else ""))
    manifest = REPO / cfg["manifest"]

    train_rows, dev_rows = load_split(manifest, cfg["dev_pages"], cfg["seed"])
    if a.smoke:
        rng = random.Random(cfg["seed"])   # random pages, so the smoke run sees varied layouts and lengths
        train_rows, dev_rows = rng.sample(train_rows, 40), dev_rows[:8]
    if cfg["grad_accum"] % world:
        raise ValueError(f"grad_accum {cfg['grad_accum']} not divisible by {world} GPUs")
    accum = cfg["grad_accum"] // world
    print(f"train {len(train_rows)} pages, dev {len(dev_rows)} pages, prompt v{PROMPT_VERSION}, "
          f"{world} GPU(s) x batch {cfg['batch_size']} x accum {accum}")

    model, processor = FastVisionModel.from_pretrained(
        str(REPO / cfg["base_model"]), load_in_4bit=False,
        # Unsloth's offloaded checkpointing breaks under DDP (recompute saves a different tensor count)
        use_gradient_checkpointing="unsloth" if world == 1 else True,
        device_map={"": rank})
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,          # vision tower frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, bias="none",
        random_state=cfg["seed"])
    FastVisionModel.for_training(model)

    collator = Collator(processor, manifest.parent, cfg["max_len"], cfg.get("pad_multiple", 1))
    check_boundary(collator, train_rows[0])
    sample = collator(train_rows[:1])
    print("example sequence length:", sample["input_ids"].shape[1],
          " target tokens:", int((sample["labels"] != -100).sum()))

    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=accum,
        ddp_find_unused_parameters=False,
        num_train_epochs=cfg["epochs"],
        max_steps=10 if a.smoke else -1,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=0 if a.smoke else cfg["warmup_steps"],
        weight_decay=0.0,
        optim="adamw_8bit",
        bf16=is_bf16_supported(),
        logging_steps=1 if a.smoke else 10,
        eval_strategy="steps",
        eval_steps=5 if a.smoke else cfg["save_steps"],
        save_strategy="steps",
        save_steps=5 if a.smoke else cfg["save_steps"],
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=dev_rows,
                      data_collator=collator)
    resume = any(out.glob("checkpoint-*")) and not a.smoke
    trainer.train(resume_from_checkpoint=resume or None)
    if not main_proc:
        return 0
    model.save_pretrained(str(out / "adapter"))
    processor.save_pretrained(str(out / "adapter"))
    peak_gb = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    (out / "train_meta.json").write_text(json.dumps(
        {"config": cfg, "prompt_version": PROMPT_VERSION, "train_pages": len(train_rows),
         "dev_pages": len(dev_rows), "peak_gpu_gb": peak_gb,
         "log_history": trainer.state.log_history}, indent=2))
    print("peak GPU memory reserved:", peak_gb, "GB")
    print("saved adapter ->", out / "adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
