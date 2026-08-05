# Export mobile (Flutter / ONNX Runtime)

Chaîne prévue par le mémoire (Phase 3) : **NLLB (baseline ou LoRA) → ONNX (± INT8) → Flutter hors ligne**.

## Prérequis

```powershell
.\train\.venv\Scripts\Activate.ps1
pip install "optimum[onnxruntime]" onnx onnxruntime
```

## Exporter le baseline (toutes les paires)

```powershell
python main.py export --config train/configs/fr-ln.yaml --baseline
python main.py export --config train/configs/fr-kg.yaml --baseline
python main.py export --config train/configs/fr-lu.yaml --baseline
python main.py export --config train/configs/fr-sw.yaml --baseline

# Option quantification INT8 (PTQ dynamique)
python main.py export --config train/configs/fr-ln.yaml --baseline --int8
```

## Exporter le modèle LoRA (après entraînement)

```powershell
python main.py export --config train/configs/fr-ln.yaml --lora
```

## Sortie

```
exports/<paire>/baseline/
  onnx/            # encoder_model.onnx, decoder_model.onnx, ...
  tokenizer/       # tokenizer.json, etc.
  manifest.json    # codes langue + chemins pour l'app
```

## Intégration Flutter (aperçu)

1. Dépendance : [`onnxruntime`](https://pub.dev/packages/onnxruntime) (ou FFI C++ ORT Mobile).
2. Copier `exports/<paire>/baseline/onnx` + `tokenizer` dans `assets/models/<paire>/`.
3. Au runtime :
   - tokenizer côté Dart/FFI (SentencePiece) ou pré-tokeniser via un petit bridge Python/native ;
   - session ORT sur `encoder_model.onnx` + `decoder_with_past_model.onnx` ;
   - `forced_bos_token_id` = id du code cible (`lin_Latn`, `kon_Latn`, `lua_Latn`, `swh_Latn`).

## Contrainte MEC (important)

| Modèle | Taille indicative | Mobile offline |
|--------|-------------------|----------------|
| NLLB-200 distilled 600M FP32 | ~2.3 Go | trop lourd |
| NLLB-200 600M INT8 | ~0.6–1 Go | limite avec Whisper sous Mlim≈1 Go |
| Helsinki OPUS-MT (`opus-mt-fr-ln`, …) | ~300 Mo | plus réaliste pour Flutter |

Pour la démo Flutter stricte hors-ligne, on pourra aussi exporter les checkpoints **OPUS-MT Marian** (déjà référencés dans `dataset/`) en parallèle du NLLB LoRA (qualité thesis).
