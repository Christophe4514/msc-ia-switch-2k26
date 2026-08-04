#!/usr/bin/env python3
"""Evaluate a LoRA adapter (BLEU, chrF, WER, accuracy) + PNG bar chart."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_moses_pair  # noqa: E402
from metrics_utils import compute_translation_metrics  # noqa: E402
from plotting import plot_final_metrics_bar  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


@torch.inference_mode()
def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    device: torch.device,
) -> list[str]:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_length=max_length,
        num_beams=4,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def load_model(cfg: dict, adapter_dir: Path | None, baseline: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        src_lang=cfg["src_lang"],
        tgt_lang=cfg["tgt_lang"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    if not baseline:
        if adapter_dir is None or not Path(adapter_dir).exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def run_evaluate(
    cfg: dict,
    *,
    split: str = "test",
    adapter: Path | None = None,
    batch_size: int = 8,
    max_samples: int | None = None,
    baseline: bool = False,
) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    splits_dir = resolve_path(cfg["dataset_root"]) / cfg["pair"] / "splits"
    output_dir = resolve_path(cfg["output_dir"])
    adapter_dir = adapter or (output_dir / "adapter")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ds = load_moses_pair(
        splits_dir,
        tgt_ext=cfg["tgt_file_ext"],
        max_eval_samples=max_samples,
    )[split]

    model, tokenizer, device = load_model(cfg, adapter_dir, baseline)

    hyps: list[str] = []
    refs: list[str] = list(ds["target"])
    sources: list[str] = list(ds["source"])
    max_len = int(cfg.get("generation_max_length", 128))

    for i in tqdm(range(0, len(sources), batch_size), desc="Generating"):
        batch = sources[i : i + batch_size]
        hyps.extend(
            translate_batch(
                model,
                tokenizer,
                batch,
                src_lang=cfg["src_lang"],
                tgt_lang=cfg["tgt_lang"],
                max_length=max_len,
                device=device,
            )
        )

    m = compute_translation_metrics(hyps, refs)
    report = {
        "pair": cfg["pair"],
        "split": split,
        "n": m.n,
        "baseline": bool(baseline),
        "adapter": None if baseline else str(adapter_dir),
        "bleu": m.bleu,
        "chrf": m.chrf,
        "wer": m.wer,
        "accuracy": m.accuracy,
        "hypothesis_Qmin": 25.0,
        "bleu_ge_Qmin": m.bleu >= 25.0,
    }

    tag = "baseline" if baseline else "lora"
    out_path = output_dir / f"metrics_{split}_{tag}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    samples = [
        {"source": s, "reference": r, "hypothesis": h}
        for s, r, h in zip(sources[:20], refs[:20], hyps[:20])
    ]
    (output_dir / f"samples_{split}_{tag}.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    chart = plot_final_metrics_bar(
        report,
        plots_dir / f"{cfg['pair']}_{split}_{tag}_metrics.png",
        title=f"{cfg['pair']} — {split} ({tag})",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    print(f"Chart  {chart}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_evaluate(
        cfg,
        split=args.split,
        adapter=args.adapter,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
