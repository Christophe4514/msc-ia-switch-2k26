"""Model architecture + hyperparameters tables (PNG) for NLLB-200 + LoRA."""

from __future__ import annotations

from pathlib import Path

from plotting import plot_architecture_table, plot_hyperparams_table


def nllb_architecture_rows(cfg: dict) -> list[tuple[str, str, str]]:
    """Summary table in the same spirit as Keras model.summary()."""
    d_model = 1024
    max_len = int(cfg.get("max_source_length", 128))
    vocab = 256_206  # NLLB-200 SentencePiece vocab (approx)
    # Distilled 600M: 12 encoder / 12 decoder layers, FFN 4096, heads 16
    embed = vocab * d_model
    # Rough per-layer transformer block params (self-attn + FFN)
    attn = 4 * d_model * d_model  # q,k,v,o
    ffn = 2 * d_model * 4096
    layer = attn + ffn + 4 * d_model  # norms rough
    enc = 12 * layer
    dec = 12 * (layer + attn)  # + cross-attn
    lm_head = vocab * d_model
    lora_r = int(cfg.get("lora_r", 16))
    # LoRA on q,k,v,o for enc+dec attention projections
    n_proj = 4 * (12 + 12)  # modules targeted per layer-ish
    lora_params = n_proj * (2 * d_model * lora_r)  # A + B

    return [
        ("Input (src tokens)", f"(None, {max_len})", "0"),
        ("Input (tgt tokens)", f"(None, {max_len})", "0"),
        ("shared Embedding", f"(None, {max_len}, {d_model})", f"{embed:,}"),
        ("Encoder ×12 (Transformer)", f"(None, {max_len}, {d_model})", f"{enc:,}"),
        ("Decoder ×12 (Transformer)", f"(None, {max_len}, {d_model})", f"{dec:,}"),
        ("LM Head (Dense)", f"(None, {max_len}, {vocab})", f"{lm_head:,}"),
        (
            f"LoRA adapters (r={lora_r}) on q/k/v/o",
            "delta W = BA",
            f"~{lora_params:,} trainable",
        ),
        ("Total params (base)", "—", "~615,000,000"),
        ("Trainable (LoRA only)", "—", "~4,700,000 (≈0.76%)"),
    ]


def hyperparams_rows(cfg: dict) -> list[tuple[str, str]]:
    return [
        ("Base model", str(cfg.get("model_name"))),
        ("Architecture", "NLLB-200 Transformer (distilled 600M)"),
        ("Fine-tuning", "PEFT LoRA"),
        ("Source lang", str(cfg.get("src_lang"))),
        ("Target lang", str(cfg.get("tgt_lang"))),
        ("Max source length", str(cfg.get("max_source_length"))),
        ("Max target length", str(cfg.get("max_target_length"))),
        ("LoRA r", str(cfg.get("lora_r"))),
        ("LoRA alpha", str(cfg.get("lora_alpha"))),
        ("LoRA dropout", str(cfg.get("lora_dropout"))),
        ("LoRA target modules", ", ".join(cfg.get("lora_target_modules", []))),
        ("Epochs", str(cfg.get("num_train_epochs"))),
        ("Batch size", str(cfg.get("per_device_train_batch_size"))),
        ("Gradient accumulation", str(cfg.get("gradient_accumulation_steps"))),
        ("Effective batch", str(int(cfg.get("per_device_train_batch_size", 1)) * int(cfg.get("gradient_accumulation_steps", 1)))),
        ("Learning rate", str(cfg.get("learning_rate"))),
        ("Weight decay", str(cfg.get("weight_decay"))),
        ("Warmup ratio", str(cfg.get("warmup_ratio"))),
        ("Optimizer", "AdamW"),
        ("Precision", "fp16" if cfg.get("fp16") else ("bf16" if cfg.get("bf16") else "fp32")),
        ("Eval strategy", str(cfg.get("eval_strategy", "epoch"))),
        ("Seed", str(cfg.get("seed", 42))),
        ("Beam size (infer)", "4"),
        ("Metric best model", str(cfg.get("metric_for_best_model", "eval_bleu"))),
    ]


def export_model_cards(cfg: dict, plots_dir: Path) -> list[Path]:
    plots_dir = Path(plots_dir)
    pair = cfg["pair"]
    arch = plot_architecture_table(
        nllb_architecture_rows(cfg),
        plots_dir / f"{pair}_architecture.png",
        f"{pair} — Model Architecture Details (NLLB-200 + LoRA)",
    )
    hyp = plot_hyperparams_table(
        hyperparams_rows(cfg),
        plots_dir / f"{pair}_hyperparameters.png",
        f"{pair} — Hyperparamètres du modèle",
    )
    return [arch, hyp]
