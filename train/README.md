# Phase 2 — Adaptation NLLB-200 par LoRA (PEFT)

Fine-tuning Parameter-Efficient (LoRA) de `facebook/nllb-200-distilled-600M`
sur les splits nettoyés `dataset/<paire>/splits/` (FR → langues nationales RDC).

## Prérequis

- GPU NVIDIA (RTX 4060 8 Go OK)
- Python 3.10+

```powershell
cd train
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entraînement

```powershell
# Lingala (recommandé en premier)
python train_lora.py --config configs/fr-ln.yaml

# Smoke test rapide (~2–5 min)
python train_lora.py --config configs/fr-ln.yaml --max-train-samples 2000 --max-eval-samples 200
```

Autres paires : `configs/fr-kg.yaml`, `fr-lu.yaml`, `fr-sw.yaml`.

## Évaluation (BLEU / chrF)

```powershell
# Après entraînement
python evaluate.py --config configs/fr-ln.yaml --split test

# Baseline NLLB sans LoRA (pour mesurer le gain)
python evaluate.py --config configs/fr-ln.yaml --split test --baseline --max-samples 500
```

Hypothèse 1 du mémoire : **BLEU ≥ Qmin = 25**.

## Codes langue NLLB (FLORES-200)

| Paire | Source | Cible |
|-------|--------|-------|
| fr-ln | `fra_Latn` | `lin_Latn` |
| fr-kg | `fra_Latn` | `kon_Latn` |
| fr-lu | `fra_Latn` | `lua_Latn` |
| fr-sw | `fra_Latn` | `swh_Latn` |

## Sorties

```
outputs/nllb-lora-fr-ln/
  adapter/           # poids LoRA + tokenizer
  run_config.json
  eval_valid.json
  metrics_test_lora.json
```

## Mémoire GPU (indicative, 600M + LoRA, fp16)

| Setting | Valeur typique |
|---------|----------------|
| batch | 4 × accum 8 (= 32 effectif) |
| max length | 128 |
| VRAM | ~6–7 Go |

Si OOM : baisser `per_device_train_batch_size` à 2 ou `max_source_length` à 96.
