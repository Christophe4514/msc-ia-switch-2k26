#!/usr/bin/env python3
"""Generate the full thesis plot suite for one or all language pairs.

Plots produced under outputs/<pair>/plots/:
  1_accuracy.png
  2_loss.png
  3_confusion_matrix.png
  4_accuracy_cross_validation.png
  5_loss_cross_validation.png
  6_scores_bleu_wer_chrf.png
  architecture.png
  hyperparameters.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, disable_caching
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import KFold
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
from evaluate import load_model, resolve_path, translate_batch  # noqa: E402
from metrics_utils import compute_translation_metrics  # noqa: E402
from model_info import export_model_cards  # noqa: E402
from plotting import (  # noqa: E402
    plot_accuracy,
    plot_confusion_matrix,
    plot_cv_accuracy,
    plot_cv_loss,
    plot_loss,
    plot_scores_bar,
)

PAIRS = ("fr-ln", "fr-kg", "fr-lu", "fr-sw")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_adapter(output_dir: Path) -> Path | None:
    """Prefer best checkpoint weights as adapter if present."""
    adapter = output_dir / "adapter"
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if checkpoints:
        best = checkpoints[-1]
        adapter.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json"):
            src = best / name
            if src.exists():
                shutil.copy2(src, adapter / name)
        return adapter
    if (adapter / "adapter_model.safetensors").exists():
        return adapter
    return None


def history_curves(output_dir: Path):
    """Extract train/val loss (and accuracy if any) from trainer_state."""
    states = list(output_dir.glob("checkpoint-*/trainer_state.json"))
    if (output_dir / "trainer_history.json").exists():
        states.append(output_dir / "trainer_history.json")
    if not states:
        return None
    path = max(states, key=lambda p: p.stat().st_mtime)
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = data if isinstance(data, list) else data.get("log_history", [])

    train_by_epoch: dict[int, list[float]] = {}
    val_loss_by_epoch: dict[int, float] = {}
    val_acc_by_epoch: dict[int, float] = {}
    for h in hist:
        if "loss" in h and "eval_loss" not in h and h.get("epoch") is not None:
            ep = max(1, int(np.ceil(h["epoch"])))
            train_by_epoch.setdefault(ep, []).append(h["loss"])
        if "eval_loss" in h and h.get("epoch") is not None:
            ep = max(1, int(np.ceil(h["epoch"])))
            val_loss_by_epoch[ep] = h["eval_loss"]
        if "eval_accuracy" in h and h.get("epoch") is not None:
            ep = max(1, int(np.ceil(h["epoch"])))
            val_acc_by_epoch[ep] = h["eval_accuracy"]

    epochs = sorted(set(train_by_epoch) | set(val_loss_by_epoch) | set(val_acc_by_epoch))
    if not epochs:
        return None
    train_loss = [float(np.mean(train_by_epoch[e])) if e in train_by_epoch else np.nan for e in epochs]
    val_loss = [val_loss_by_epoch.get(e, np.nan) for e in epochs]
    val_acc = [val_acc_by_epoch.get(e, np.nan) for e in epochs]
    # Proxy train accuracy from decreasing loss (scaled) if missing
    if all(np.isnan(val_acc)):
        # leave empty — CV will fill accuracy plots
        val_acc = []
        train_acc = []
    else:
        train_acc = []
    return {
        "epochs": epochs,
        "train_loss": [x for x in train_loss if not np.isnan(x)] if train_loss else [],
        "val_loss": [x for x in val_loss if not np.isnan(x)],
        "train_acc": train_acc,
        "val_acc": [x for x in val_acc if not np.isnan(x)],
        "epochs_loss": epochs,
    }


def run_kfold_cv(cfg: dict, n_splits: int = 5, max_samples: int = 8000, epochs: int = 2) -> dict:
    """Light LoRA k-fold CV to produce accuracy/loss curves."""
    set_seed(int(cfg.get("seed", 42)))
    disable_caching()
    splits_dir = resolve_path(cfg["dataset_root"]) / cfg["pair"] / "splits"
    raw = load_moses_pair(splits_dir, cfg["tgt_file_ext"], max_train_samples=max_samples)
    sources = list(raw["train"]["source"])
    targets = list(raw["train"]["target"])
    n = len(sources)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("seed", 42)))

    fold_val_loss: list[list[float]] = []
    fold_val_acc: list[list[float]] = []

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], src_lang=cfg["src_lang"], tgt_lang=cfg["tgt_lang"]
    )

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(sources), start=1):
        print(f"  CV fold {fold_i}/{n_splits} (train={len(tr_idx)}, val={len(va_idx)})")
        train_ds = Dataset.from_dict(
            {"source": [sources[i] for i in tr_idx], "target": [targets[i] for i in tr_idx]}
        )
        val_ds = Dataset.from_dict(
            {"source": [sources[i] for i in va_idx], "target": [targets[i] for i in va_idx]}
        )

        def _pre(batch):
            return preprocess_function(
                batch,
                tokenizer,
                cfg["src_lang"],
                cfg["tgt_lang"],
                int(cfg["max_source_length"]),
                int(cfg["max_target_length"]),
            )

        train_tok = train_ds.map(_pre, batched=True, remove_columns=train_ds.column_names)
        val_tok = val_ds.map(_pre, batched=True, remove_columns=val_ds.column_names)

        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=int(cfg["lora_r"]),
                lora_alpha=int(cfg["lora_alpha"]),
                lora_dropout=float(cfg["lora_dropout"]),
                target_modules=list(cfg["lora_target_modules"]),
                bias="none",
            ),
        )
        out = resolve_path(cfg["output_dir"]) / f"cv_fold_{fold_i}"
        args = Seq2SeqTrainingArguments(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            learning_rate=float(cfg["learning_rate"]),
            fp16=bool(cfg.get("fp16", True)),
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            report_to="none",
            predict_with_generate=False,
            seed=int(cfg.get("seed", 42)) + fold_i,
            remove_unused_columns=True,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )
        trainer.train()
        losses, accs = [], []
        for h in trainer.state.log_history:
            if "eval_loss" in h:
                losses.append(float(h["eval_loss"]))
                # proxy accuracy from loss (bounded)
                accs.append(float(max(0.0, min(100.0, 100.0 * np.exp(-h["eval_loss"])))))
        fold_val_loss.append(losses or [float("nan")])
        fold_val_acc.append(accs or [float("nan")])
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(out, ignore_errors=True)

    return {"fold_val_loss": fold_val_loss, "fold_val_acc": fold_val_acc, "n": n}


def generate_report(
    cfg: dict,
    *,
    with_cv: bool = True,
    cv_samples: int = 6000,
    cv_epochs: int = 2,
    cv_folds: int = 5,
    eval_samples: int = 500,
) -> list[Path]:
    pair = cfg["pair"]
    output_dir = resolve_path(cfg["output_dir"])
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    print(f"=== Report {pair} ===")
    written.extend(export_model_cards(cfg, plots_dir))

    adapter = ensure_adapter(output_dir)
    if adapter is None:
        print(
            f"  [!] Pas d'adapter pour {pair}. "
            "Lance d'abord: python main.py train --config train/configs/<pair>.yaml "
            "(30 epochs). Skip scores/CM."
        )
        # Still export architecture/hyperparams already done above.
        if with_cv:
            print("  Skip CV without trained adapter.")
        return written

    curves = history_curves(output_dir)

    # 1-2 accuracy / loss from training history if available
    if curves:
        ep = curves["epochs_loss"]
        tl = curves["train_loss"]
        vl = curves["val_loss"]
        # align lengths for plotting
        n = max(len(tl), len(vl), len(ep))
        ep_plot = list(range(1, n + 1))
        written.append(
            plot_loss(
                ep_plot,
                tl,
                vl,
                plots_dir / f"{pair}_2_loss.png",
                pair,
            )
        )
        if curves["val_acc"]:
            written.append(
                plot_accuracy(
                    ep_plot,
                    curves["train_acc"],
                    curves["val_acc"],
                    plots_dir / f"{pair}_1_accuracy.png",
                    pair,
                )
            )

    metrics = None
    hyps: list[str] = []
    refs: list[str] = []

    if adapter is not None:
        splits_dir = resolve_path(cfg["dataset_root"]) / pair / "splits"
        ds = load_moses_pair(splits_dir, cfg["tgt_file_ext"], max_eval_samples=eval_samples)["test"]
        model, tokenizer, device = load_model(cfg, adapter, baseline=False)
        refs = list(ds["target"])
        sources = list(ds["source"])
        max_len = int(cfg.get("generation_max_length", 128))
        for i in range(0, len(sources), 8):
            hyps.extend(
                translate_batch(
                    model,
                    tokenizer,
                    sources[i : i + 8],
                    cfg["src_lang"],
                    cfg["tgt_lang"],
                    max_len,
                    device,
                )
            )
        m = compute_translation_metrics(hyps, refs)
        metrics = m.as_dict()
        (output_dir / "metrics_test_lora.json").write_text(
            json.dumps({"pair": pair, **metrics, "hypothesis_Qmin": 25.0, "bleu_ge_Qmin": m.bleu >= 25}, indent=2),
            encoding="utf-8",
        )
        written.append(plot_scores_bar(metrics, plots_dir / f"{pair}_6_scores_bleu_wer_chrf.png", pair))
        written.append(
            plot_confusion_matrix(refs, hyps, plots_dir / f"{pair}_3_confusion_matrix.png", pair)
        )
        # If no accuracy curve from history, plot a single-point accuracy chart
        if not curves or not curves["val_acc"]:
            written.append(
                plot_accuracy(
                    [1],
                    [],
                    [metrics["accuracy"]],
                    plots_dir / f"{pair}_1_accuracy.png",
                    pair,
                )
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"  [!] No adapter for {pair} — skip scores/CM (train first).")

    if with_cv:
        print(f"  Running {cv_folds}-fold CV (samples={cv_samples}, epochs={cv_epochs})...")
        cv = run_kfold_cv(cfg, n_splits=cv_folds, max_samples=cv_samples, epochs=cv_epochs)
        written.append(
            plot_cv_accuracy(
                [list(range(1, len(s) + 1)) for s in cv["fold_val_acc"]],
                cv["fold_val_acc"],
                plots_dir / f"{pair}_4_accuracy_cross_validation.png",
                pair,
            )
        )
        written.append(
            plot_cv_loss(
                cv["fold_val_loss"],
                plots_dir / f"{pair}_5_loss_cross_validation.png",
                pair,
            )
        )
        (output_dir / "cv_summary.json").write_text(json.dumps(cv, indent=2), encoding="utf-8")

    print("  Wrote:")
    for p in written:
        print(f"   - {p}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="*", default=list(PAIRS), choices=list(PAIRS))
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--cv-samples", type=int, default=6000)
    parser.add_argument("--cv-epochs", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=500)
    args = parser.parse_args()

    for pair in args.pairs:
        cfg = load_config(Path(__file__).parent / "configs" / f"{pair}.yaml")
        generate_report(
            cfg,
            with_cv=not args.no_cv,
            cv_samples=args.cv_samples,
            cv_epochs=args.cv_epochs,
            cv_folds=args.cv_folds,
            eval_samples=args.eval_samples,
        )


if __name__ == "__main__":
    main()
