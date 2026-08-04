# Phase 2 — Adaptation NLLB-200 par LoRA (PEFT)

Fine-tuning Parameter-Efficient (LoRA) de `facebook/nllb-200-distilled-600M`
sur les splits `dataset/<paire>/splits/`.

## Point d'entrée général (racine du repo)

```powershell
# Activer le venv
.\train\.venv\Scripts\Activate.ps1

# Entraîner une paire (BLEU/WER/accuracy par epoch + PNG)
python main.py train --config train/configs/fr-ln.yaml

# Entraîner toutes les paires
python main.py train-all

# Évaluer sur le test set (JSON + barre PNG)
python main.py evaluate --config train/configs/fr-ln.yaml --split test

# Tester le modèle
python main.py test --config train/configs/fr-ln.yaml --text "Bonjour, comment allez-vous ?"
python main.py test --config train/configs/fr-ln.yaml --interactive
```

## Prérequis

- GPU NVIDIA (RTX 4060 8 Go OK)
- **Python 3.11–3.13**

```powershell
cd train
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Sorties après `train`

```
outputs/nllb-lora-fr-ln/
  adapter/
  trainer_history.json
  eval_valid.json
  plots/
    fr-ln_loss.png
    fr-ln_bleu_chrf.png
    fr-ln_wer_accuracy.png
    fr-ln_dashboard.png
```

## Métriques

| Métrique | Sens |
|----------|------|
| BLEU | Qualité traduction (objectif thesis ≥ 25) |
| chrF | Similarité caractères |
| WER | Erreur au niveau mots (plus bas = mieux) |
| Accuracy | % phrases exactement identiques à la référence |

## Codes langue NLLB (FLORES-200)

| Paire | Source | Cible |
|-------|--------|-------|
| fr-ln | `fra_Latn` | `lin_Latn` |
| fr-kg | `fra_Latn` | `kon_Latn` |
| fr-lu | `fra_Latn` | `lua_Latn` |
| fr-sw | `fra_Latn` | `swh_Latn` |
