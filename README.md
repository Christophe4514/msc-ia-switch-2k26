# msc-ia-switch-2k26

Traduction audio FR → langues nationales RDC (lingala, swahili, tshiluba, kikongo), hors ligne sur mobile.

## Démarrage rapide

```powershell
.\train\.venv\Scripts\Activate.ps1

# Entraînement LoRA (+ graphiques PNG : loss, BLEU, WER, accuracy)
python main.py train --config train/configs/fr-ln.yaml
python main.py train-all

# Évaluation test set
python main.py evaluate --config train/configs/fr-ln.yaml --split test

# Test interactif du modèle
python main.py test --config train/configs/fr-ln.yaml --interactive
```

Voir `dataset/README.md` (corpus) et `train/README.md` (LoRA / métriques).
