#!/usr/bin/env python3
"""Serveur HTTP local : le modèle entraîné (LoRA) répond aux requêtes de l'app Flutter.

Usage:
  python main.py serve
  python main.py serve --pairs fr-ln fr-sw --port 8765 --baseline

Depuis Flutter (même Wi‑Fi / émulateur Android) :
  POST http://<IP-PC>:8765/translate
  {"text": "Bonjour", "source": "fra_Latn", "target": "lin_Latn"}
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN))

from evaluate import load_config, load_model, resolve_path, translate_batch  # noqa: E402

PAIRS_DEFAULT = ["fr-ln", "fr-kg", "fr-lu", "fr-sw"]

# FLORES code → paire d'export
TGT_TO_PAIR = {
    "lin_Latn": "fr-ln",
    "swh_Latn": "fr-sw",
    "kon_Latn": "fr-kg",
    "lua_Latn": "fr-lu",
}


class ModelRegistry:
    """Charge une fois chaque paire (LoRA si présent, sinon baseline)."""

    def __init__(self, pairs: list[str], *, force_baseline: bool = False) -> None:
        self.force_baseline = force_baseline
        self.engines: dict[str, dict] = {}
        for pair in pairs:
            self._load_pair(pair)

    def _load_pair(self, pair: str) -> None:
        cfg_path = TRAIN / "configs" / f"{pair}.yaml"
        if not cfg_path.exists():
            print(f"[skip] config manquante: {cfg_path}")
            return
        cfg = load_config(cfg_path)
        adapter = resolve_path(cfg["output_dir"]) / "adapter"
        has_lora = (adapter / "adapter_model.safetensors").exists() or (
            adapter / "adapter_model.bin"
        ).exists()
        baseline = self.force_baseline or not has_lora
        mode = "baseline" if baseline else f"LoRA ({adapter})"
        print(f"[load] {pair} → {mode} …")
        try:
            model, tokenizer, device = load_model(
                cfg, None if baseline else adapter, baseline
            )
        except Exception as exc:
            print(f"[error] {pair}: {exc}")
            return
        self.engines[pair] = {
            "cfg": cfg,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "mode": mode,
        }
        print(f"[ok]   {pair} prêt ({cfg['src_lang']} → {cfg['tgt_lang']})")

    def translate(self, text: str, src: str, tgt: str) -> tuple[str, str]:
        pair = TGT_TO_PAIR.get(tgt)
        if pair is None:
            raise ValueError(
                f"Langue cible non supportée: {tgt}. "
                f"Attendues: {list(TGT_TO_PAIR)}"
            )
        engine = self.engines.get(pair)
        if engine is None:
            raise RuntimeError(
                f"Paire {pair} non chargée. Entraînez-la ou lancez avec --baseline."
            )
        cfg = engine["cfg"]
        if src and src != cfg["src_lang"]:
            # On force fra_Latn pour ce mémoire ; tolère 'auto'
            if src not in ("fra_Latn", "auto", ""):
                raise ValueError(f"Source attendue {cfg['src_lang']}, reçu {src}")
        outs = translate_batch(
            engine["model"],
            engine["tokenizer"],
            [text],
            src_lang=cfg["src_lang"],
            tgt_lang=cfg["tgt_lang"],
            max_length=int(cfg.get("generation_max_length", 128)),
            device=engine["device"],
        )
        return outs[0], pair

    def status(self) -> dict:
        return {
            "service": "msc-ia-switch-2k26",
            "pairs": {
                p: {"mode": e["mode"], "src": e["cfg"]["src_lang"], "tgt": e["cfg"]["tgt_lang"]}
                for p, e in self.engines.items()
            },
        }


def make_handler(registry: ModelRegistry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[http] {self.address_string()} - {fmt % args}")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/health", "/status"):
                self._send(200, registry.status())
                return
            # Compat style API publique : /api/v4/translator?text=&source=&target=
            if parsed.path.endswith("/translator") or parsed.path == "/translate":
                qs = parse_qs(parsed.query)
                text = (qs.get("text") or [""])[0]
                src = (qs.get("source") or ["fra_Latn"])[0]
                tgt = (qs.get("target") or [""])[0]
                self._do_translate(text, src, tgt)
                return
            self._send(404, {"error": "Not found. Use GET/POST /translate"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in ("/translate", "/api/v4/translator"):
                self._send(404, {"error": "Not found. Use POST /translate"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "JSON invalide"})
                return
            text = str(data.get("text") or data.get("inputs") or "")
            params = data.get("parameters") or {}
            src = str(
                data.get("source")
                or params.get("src_lang")
                or "fra_Latn"
            )
            tgt = str(
                data.get("target")
                or params.get("tgt_lang")
                or ""
            )
            self._do_translate(text, src, tgt)

        def _do_translate(self, text: str, src: str, tgt: str) -> None:
            if not text.strip():
                self._send(400, {"error": "text vide"})
                return
            if not tgt:
                self._send(400, {"error": "target (FLORES) requis, ex. lin_Latn"})
                return
            try:
                hyp, pair = registry.translate(text.strip(), src, tgt)
                self._send(
                    200,
                    {
                        "translation_text": hyp,
                        "pair": pair,
                        "source": src or "fra_Latn",
                        "target": tgt,
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {"error": str(exc)})

    return Handler


def run_server(
    pairs: list[str],
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    baseline: bool = False,
) -> None:
    registry = ModelRegistry(pairs, force_baseline=baseline)
    if not registry.engines:
        raise SystemExit(
            "Aucun modèle chargé. Entraînez au moins une paire, ou "
            "lancez: python main.py serve --baseline"
        )
    handler = make_handler(registry)
    httpd = ThreadingHTTPServer((host, port), handler)
    print()
    print("=" * 60)
    print("  Serveur prêt pour Flutter")
    print(f"  URL locale : http://127.0.0.1:{port}/translate")
    print(f"  Émulateur Android → http://10.0.2.2:{port}/translate")
    print(f"  Téléphone réel   → http://<IP-de-ce-PC>:{port}/translate")
    print("  Health           → GET /health")
    print("=" * 60)
    print("Ctrl+C pour arrêter.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=PAIRS_DEFAULT,
        choices=PAIRS_DEFAULT,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Forcer NLLB de base (sans LoRA), utile avant fin d'entraînement",
    )
    args = parser.parse_args()
    run_server(args.pairs, host=args.host, port=args.port, baseline=args.baseline)


if __name__ == "__main__":
    main()
