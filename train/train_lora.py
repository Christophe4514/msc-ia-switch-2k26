#!/usr/bin/env python3
"""Fine-tune NLLB-200 with LoRA (PEFT) on FR→national language splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from datasets import disable_caching
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_moses_pair, preprocess_function  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "fr-ln.yaml",
        help="YAML config (default: configs/fr-ln.yaml)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override config for a quick smoke run",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Override config eval/test sample cap",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg["max_eval_samples"] = args.max_eval_samples

    set_seed(int(cfg.get("seed", 42)))
    disable_caching()

    splits_dir = resolve_path(cfg["dataset_root"]) / cfg["pair"] / "splits"
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pair      : {cfg['pair']} ({cfg['src_lang']} → {cfg['tgt_lang']})")
    print(f"Splits    : {splits_dir}")
    print(f"Model     : {cfg['model_name']}")
    print(f"Output    : {output_dir}")

    raw = load_moses_pair(
        splits_dir,
        tgt_ext=cfg["tgt_file_ext"],
        max_train_samples=cfg.get("max_train_samples"),
        max_eval_samples=cfg.get("max_eval_samples"),
    )
    print(
        f"Sizes     : train={len(raw['train']):,}  "
        f"valid={len(raw['valid']):,}  test={len(raw['test']):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        src_lang=cfg["src_lang"],
        tgt_lang=cfg["tgt_lang"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def _preprocess(batch):
        return preprocess_function(
            batch,
            tokenizer=tokenizer,
            src_lang=cfg["src_lang"],
            tgt_lang=cfg["tgt_lang"],
            max_source_length=int(cfg["max_source_length"]),
            max_target_length=int(cfg["max_target_length"]),
        )

    tokenized = raw.map(
        _preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        logging_steps=int(cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        predict_with_generate=bool(cfg.get("predict_with_generate", True)),
        generation_max_length=int(cfg.get("generation_max_length", 128)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = trainer.evaluate(tokenized["valid"])
    (output_dir / "eval_valid.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("Valid metrics:", metrics)
    print(f"Adapter saved to {output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
