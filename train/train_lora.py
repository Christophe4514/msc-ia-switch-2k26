#!/usr/bin/env python3
"""Fine-tune NLLB-200 with LoRA (PEFT) — 30 epochs, metrics + PNG curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from metrics_utils import compute_translation_metrics, decode_preds_labels  # noqa: E402
from plotting import plot_training_curves  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _is_valid_checkpoint(ckpt_dir: Path) -> bool:
    """Trainer resume needs trainer_state + model/adapter weights."""
    if not (ckpt_dir / "trainer_state.json").exists():
        return False
    for name in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        if (ckpt_dir / name).exists():
            return True
    return False


def find_resume_checkpoint(output_dir: Path, cfg: dict) -> str | None:
    """Resume only if an existing checkpoint matches current LoRA r."""
    if not cfg.get("resume_from_checkpoint", True):
        return None
    run_cfg_path = output_dir / "run_config.json"
    if run_cfg_path.exists():
        old = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        if int(old.get("lora_r", -1)) != int(cfg["lora_r"]):
            print(
                f"Checkpoint LoRA r={old.get('lora_r')} != config r={cfg['lora_r']} "
                "— starting fresh."
            )
            return None
    ckpts = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
        reverse=True,
    )
    for ckpt in ckpts:
        if _is_valid_checkpoint(ckpt):
            if ckpt != ckpts[0]:
                print(
                    f"Skipping incomplete checkpoint {ckpts[0].name} "
                    f"→ resuming from {ckpt.name}"
                )
            return str(ckpt)
        print(f"Skipping invalid checkpoint {ckpt.name} (save interrupted?)")
    return None


def run_train(cfg: dict) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    disable_caching()

    splits_dir = resolve_path(cfg["dataset_root"]) / cfg["pair"] / "splits"
    output_dir = resolve_path(cfg["output_dir"])
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pair      : {cfg['pair']} ({cfg['src_lang']} -> {cfg['tgt_lang']})")
    print(f"Epochs    : {cfg['num_train_epochs']}")
    print(f"LoRA r    : {cfg['lora_r']}  alpha={cfg['lora_alpha']}")
    print(f"Splits    : {splits_dir}")
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
    do_generate = bool(cfg.get("predict_with_generate", True))

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        hyp, ref = decode_preds_labels(tokenizer, preds, labels)
        m = compute_translation_metrics(hyp, ref)
        return {
            "bleu": m.bleu,
            "chrf": m.chrf,
            "wer": m.wer,
            "accuracy": m.accuracy,
        }

    eval_strategy = cfg.get("eval_strategy", "epoch")
    save_strategy = cfg.get("save_strategy", eval_strategy)
    metric_best = cfg.get("metric_for_best_model", "eval_bleu" if do_generate else "eval_loss")
    greater = bool(cfg.get("greater_is_better", do_generate))

    args_kwargs = dict(
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
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=int(cfg["save_total_limit"]),
        predict_with_generate=do_generate,
        generation_max_length=int(cfg.get("generation_max_length", 128)),
        load_best_model_at_end=True,
        metric_for_best_model=metric_best,
        greater_is_better=greater,
        report_to=cfg.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=True,
    )
    if eval_strategy == "steps":
        args_kwargs["eval_steps"] = int(cfg.get("eval_steps", 500))
        args_kwargs["save_steps"] = int(cfg.get("save_steps", 500))

    training_args = Seq2SeqTrainingArguments(**args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["valid"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if do_generate else None,
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resume = find_resume_checkpoint(output_dir, cfg)
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    # Final eval with generation for BLEU even if train used loss-only
    if not do_generate:
        trainer.args.predict_with_generate = True
        trainer.compute_metrics = compute_metrics
    metrics = trainer.evaluate(tokenized["valid"])
    (output_dir / "eval_valid.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("Valid metrics:", metrics)

    history_path = output_dir / "trainer_history.json"
    history_path.write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    plots = plot_training_curves(trainer.state.log_history, plots_dir, cfg["pair"])
    print("Plots:")
    for p in plots:
        print(f"  - {p}")

    print(f"Adapter saved to {output_dir / 'adapter'}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "fr-ln.yaml",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg["max_eval_samples"] = args.max_eval_samples
    run_train(cfg)


if __name__ == "__main__":
    main()
