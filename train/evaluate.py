#!/usr/bin/env python3
"""Evaluate a LoRA adapter with sacreBLEU + chrF on the test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from sacrebleu import CHRF, corpus_bleu
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_moses_pair  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to LoRA adapter dir (default: <output_dir>/adapter)",
    )
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate base NLLB without LoRA adapter",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    splits_dir = resolve_path(cfg["dataset_root"]) / cfg["pair"] / "splits"
    output_dir = resolve_path(cfg["output_dir"])
    adapter_dir = args.adapter or (output_dir / "adapter")

    ds = load_moses_pair(
        splits_dir,
        tgt_ext=cfg["tgt_file_ext"],
        max_eval_samples=args.max_samples,
    )[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        src_lang=cfg["src_lang"],
        tgt_lang=cfg["tgt_lang"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    if not args.baseline:
        if not Path(adapter_dir).exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()

    hyps: list[str] = []
    refs: list[str] = list(ds["target"])
    sources: list[str] = list(ds["source"])
    max_len = int(cfg.get("generation_max_length", 128))

    for i in tqdm(range(0, len(sources), args.batch_size), desc="Generating"):
        batch = sources[i : i + args.batch_size]
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

    bleu = corpus_bleu(hyps, [refs])
    chrf = CHRF().corpus_score(hyps, [refs])
    report = {
        "pair": cfg["pair"],
        "split": args.split,
        "n": len(hyps),
        "baseline": bool(args.baseline),
        "adapter": None if args.baseline else str(adapter_dir),
        "bleu": bleu.score,
        "chrf": chrf.score,
        "hypothesis_Qmin": 25.0,
        "bleu_ge_Qmin": bleu.score >= 25.0,
    }

    tag = "baseline" if args.baseline else "lora"
    out_path = output_dir / f"metrics_{args.split}_{tag}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Small sample for qualitative inspection
    samples = [
        {"source": s, "reference": r, "hypothesis": h}
        for s, r, h in zip(sources[:10], refs[:10], hyps[:10])
    ]
    (output_dir / f"samples_{args.split}_{tag}.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
