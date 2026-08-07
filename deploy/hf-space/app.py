#!/usr/bin/env python3
"""Space Gradio gratuit — NLLB (+ LoRA optionnel) pour Flutter.

Hardware recommandé : CPU basic (Settings → Hardware).
Si ZeroGPU : le décorateur @spaces.GPU est activé automatiquement.

API Flutter :
  POST /gradio_api/call/translate
  {"data": ["Bonjour", "fra_Latn", "lin_Latn"]}
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

# ZeroGPU (optionnel) — ignore si Space en CPU basic
try:
    import spaces

    gpu = spaces.GPU
except ImportError:

    def gpu(duration=120):  # noqa: ARG001
        def deco(fn):
            return fn

        return deco


BASE_MODEL = os.environ.get("BASE_MODEL", "facebook/nllb-200-distilled-600M")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "").strip()

TGT_TO_PAIR = {
    "lin_Latn": "fr-ln",
    "swh_Latn": "fr-sw",
    "kon_Latn": "fr-kg",
    "lua_Latn": "fr-lu",
}
SRC_DEFAULT = "fra_Latn"
TARGETS = list(TGT_TO_PAIR.keys())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TOKENIZER = None
_MODEL = None
_LOADED_ADAPTERS: set[str] = set()


def _load_base() -> None:
    """Charge le modèle une seule fois (lazy — évite crash au startup HF)."""
    global _TOKENIZER, _MODEL
    if _MODEL is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {BASE_MODEL} on {DEVICE} …")
    _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_DEFAULT)
    _MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    _MODEL.to(DEVICE)
    _MODEL.eval()
    print("Base model ready.")


def _resolve_adapter_dir(pair: str) -> str | None:
    if not ADAPTER_REPO:
        return None

    local = Path(ADAPTER_REPO)
    if local.exists():
        for cand in (local / pair, local / pair / "adapter"):
            if (cand / "adapter_model.safetensors").exists() or (
                cand / "adapter_model.bin"
            ).exists():
                return str(cand)
        return None

    try:
        from huggingface_hub import list_repo_files, snapshot_download

        files = list_repo_files(ADAPTER_REPO)
        for prefix in (f"{pair}/", f"{pair}/adapter/"):
            if any(
                f.startswith(prefix) and f.endswith((".safetensors", ".bin"))
                for f in files
            ):
                root = snapshot_download(
                    repo_id=ADAPTER_REPO,
                    allow_patterns=[f"{prefix}*"],
                )
                cand = Path(root) / prefix.rstrip("/")
                if (cand / "adapter_model.safetensors").exists() or (
                    cand / "adapter_model.bin"
                ).exists():
                    return str(cand)
    except Exception as exc:
        print(f"[{pair}] adapter introuvable: {exc}")
    return None


def _ensure_adapter(pair: str) -> bool:
    global _MODEL
    _load_base()
    assert _MODEL is not None

    if pair in _LOADED_ADAPTERS:
        if hasattr(_MODEL, "set_adapter"):
            _MODEL.set_adapter(pair)
        return True

    path = _resolve_adapter_dir(pair)
    if not path:
        return False

    from peft import PeftModel

    print(f"[{pair}] load adapter {path}")
    if not _LOADED_ADAPTERS:
        _MODEL = PeftModel.from_pretrained(_MODEL, path, adapter_name=pair)
    else:
        _MODEL.load_adapter(path, adapter_name=pair)
    _MODEL.set_adapter(pair)
    _MODEL.to(DEVICE)
    _MODEL.eval()
    _LOADED_ADAPTERS.add(pair)
    return True


@gpu(duration=120)
@torch.inference_mode()
def translate(text: str, source: str, target: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if target not in TGT_TO_PAIR:
        return f"Erreur: target inconnu ({target}). Choix: {', '.join(TARGETS)}"

    _load_base()
    assert _TOKENIZER is not None and _MODEL is not None

    pair = TGT_TO_PAIR[target]
    has_lora = _ensure_adapter(pair)
    if has_lora and hasattr(_MODEL, "set_adapter"):
        _MODEL.set_adapter(pair)

    src = source if source and source not in ("auto", "") else SRC_DEFAULT
    _TOKENIZER.src_lang = src
    inputs = _TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)
    bos = _TOKENIZER.convert_tokens_to_ids(target)
    out = _MODEL.generate(
        **inputs,
        forced_bos_token_id=bos,
        max_length=128,
        num_beams=4,
    )
    hyp = _TOKENIZER.batch_decode(out, skip_special_tokens=True)[0]
    mode = f"LoRA:{pair}" if has_lora else "baseline"
    print(f"[{mode}] {text[:40]!r} → {hyp[:40]!r}")
    return hyp


with gr.Blocks(title="MSC IA Switch — NLLB RDC") as demo:
    gr.Markdown(
        f"""
        # Traduction FR → langues RDC
        Modèle : `{BASE_MODEL}`  
        Adapters : `{ADAPTER_REPO or "aucun (baseline)"}`  

        Premier appel = chargement du modèle (peut prendre 1–3 min).
        """
    )
    with gr.Row():
        text_in = gr.Textbox(label="Texte français", lines=3)
    with gr.Row():
        src_in = gr.Textbox(value=SRC_DEFAULT, label="Source (FLORES)")
        tgt_in = gr.Dropdown(choices=TARGETS, value="lin_Latn", label="Cible")
    out = gr.Textbox(label="Traduction", lines=3)
    btn = gr.Button("Traduire", variant="primary")
    btn.click(
        fn=translate,
        inputs=[text_in, src_in, tgt_in],
        outputs=out,
        api_name="translate",
    )
    gr.Examples(
        examples=[
            ["Bonjour, comment allez-vous ?", SRC_DEFAULT, "lin_Latn"],
            ["Je vais au marché demain.", SRC_DEFAULT, "swh_Latn"],
            ["Merci beaucoup.", SRC_DEFAULT, "kon_Latn"],
            ["Où est la maison ?", SRC_DEFAULT, "lua_Latn"],
        ],
        inputs=[text_in, src_in, tgt_in],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
