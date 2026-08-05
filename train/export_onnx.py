#!/usr/bin/env python3
"""Export NLLB baseline (or LoRA-merged) to ONNX for Flutter / ONNX Runtime Mobile.

Outputs under exports/<pair>/:
  onnx/           — encoder/decoder ONNX (+ config)
  tokenizer/      — tokenizer.json, sentencepiece, special tokens
  manifest.json   — paths, sizes, language codes for the mobile app
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def export_pair(
    cfg: dict,
    *,
    baseline: bool = True,
    adapter: Path | None = None,
    quantize_int8: bool = False,
    opset: int = 14,
) -> Path:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    pair = cfg["pair"]
    out_root = resolve_path("exports") / pair / ("baseline" if baseline else "lora")
    onnx_dir = out_root / "onnx"
    tok_dir = out_root / "tokenizer"
    out_root.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Export {pair} ({'baseline' if baseline else 'LoRA'}) ===")
    print(f"Model : {cfg['model_name']}")
    print(f"Out   : {out_root}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        src_lang=cfg["src_lang"],
        tgt_lang=cfg["tgt_lang"],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    if not baseline:
        from peft import PeftModel

        adapter_dir = adapter or (resolve_path(cfg["output_dir"]) / "adapter")
        if not (Path(adapter_dir) / "adapter_model.safetensors").exists() and not (
            Path(adapter_dir) / "adapter_model.bin"
        ).exists():
            raise FileNotFoundError(
                f"Adapter LoRA manquant: {adapter_dir}\n"
                "Entraîne d'abord la paire, ou exporte en --baseline."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print("Merging LoRA into base weights...")
        model = model.merge_and_unload()

    model.eval()

    # Prefer Optimum when available (encoder/decoder split ready for ORT)
    used = "optimum"
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        tmp = out_root / "_hf_export"
        tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)

        ort_model = ORTModelForSeq2SeqLM.from_pretrained(tmp, export=True)
        ort_model.save_pretrained(onnx_dir)
        shutil.rmtree(tmp, ignore_errors=True)

        if quantize_int8:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            print("INT8 dynamic quantization...")
            for name in ("encoder_model", "decoder_model", "decoder_with_past_model"):
                onnx_file = onnx_dir / f"{name}.onnx"
                if not onnx_file.exists():
                    continue
                quantizer = ORTQuantizer.from_pretrained(onnx_dir, file_name=f"{name}.onnx")
                qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
                quantizer.quantize(save_dir=onnx_dir / "int8", quantization_config=qconfig)
    except Exception as exc:
        used = "torch.onnx"
        print(f"Optimum export unavailable ({exc}); falling back to torch.onnx encoder-only stub.")
        print("Install for full seq2seq ONNX: pip install 'optimum[onnxruntime]' onnx")
        # Minimal encoder export so the pipeline still produces an artifact + tokenizer
        max_len = int(cfg.get("max_source_length", 128))
        dummy = {
            "input_ids": torch.ones(1, max_len, dtype=torch.long),
            "attention_mask": torch.ones(1, max_len, dtype=torch.long),
        }
        encoder = model.get_encoder()
        torch.onnx.export(
            encoder,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(onnx_dir / "encoder_model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
        )
        model.save_pretrained(onnx_dir / "pytorch_fallback")
        (onnx_dir / "README_FALLBACK.txt").write_text(
            "Full encoder+decoder ONNX requires: pip install 'optimum[onnxruntime]' onnx\n"
            "Then re-run: python main.py export --config ... --baseline\n",
            encoding="utf-8",
        )

    tokenizer.save_pretrained(tok_dir)
    # Also copy sentencepiece if present next to model cache files
    for fname in ("sentencepiece.bpe.model", "source.spm", "target.spm", "spm.model"):
        src = Path(cfg["model_name"])
        # tokenizer files already in tok_dir via save_pretrained

    manifest = {
        "pair": pair,
        "mode": "baseline" if baseline else "lora_merged",
        "base_model": cfg["model_name"],
        "src_lang": cfg["src_lang"],
        "tgt_lang": cfg["tgt_lang"],
        "max_length": int(cfg.get("generation_max_length", 128)),
        "export_backend": used,
        "quantize_int8": quantize_int8,
        "onnx_dir": str(onnx_dir.relative_to(ROOT)).replace("\\", "/"),
        "tokenizer_dir": str(tok_dir.relative_to(ROOT)).replace("\\", "/"),
        "size_mb_onnx": round(_dir_size_mb(onnx_dir), 1),
        "size_mb_tokenizer": round(_dir_size_mb(tok_dir), 1),
        "flutter": {
            "package": "onnxruntime",
            "notes": [
                "Copy exports/<pair>/baseline/onnx and tokenizer into Flutter assets/",
                "Load encoder/decoder ONNX with OnnxRuntime",
                "Force BOS target lang id from tokenizer (e.g. lin_Latn)",
                "NLLB-600M INT8 remains large (~0.5–1 Go): consider OPUS-MT Marian for stricter MEC budgets",
            ],
        },
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Done -> {out_root}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--lora", action="store_true", help="Export merged LoRA instead of baseline")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--int8", action="store_true", help="Dynamic INT8 quantization (Optimum)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    export_pair(
        cfg,
        baseline=not args.lora,
        adapter=args.adapter,
        quantize_int8=args.int8,
    )


if __name__ == "__main__":
    main()
