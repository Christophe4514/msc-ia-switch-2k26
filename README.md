# msc-ia-switch-2k26

Traduction audio FR → langues nationales RDC (lingala, swahili, tshiluba, kikongo), hors ligne sur mobile.

## Démarrage rapide

```powershell
.\train\.venv\Scripts\Activate.ps1

# Entraînement LoRA (30 epochs)
python main.py train --config train/configs/fr-ln.yaml
python main.py train-all

# Test baseline (sans LoRA) / LoRA
python main.py test --config train/configs/fr-kg.yaml --baseline --interactive
python main.py test --config train/configs/fr-ln.yaml --interactive

# Évaluation
python main.py evaluate --config train/configs/fr-ln.yaml --split test

# Export ONNX pour Flutter
python main.py export --config train/configs/fr-ln.yaml --baseline
python main.py export --config train/configs/fr-ln.yaml --lora --int8
```

Voir `dataset/README.md`, `train/README.md`, `export/README.md`.

## Rapports / graphiques

```powershell
python main.py report --pairs fr-ln fr-kg fr-lu fr-sw
```

Plots dans `outputs/nllb-lora-<paire>/plots/` : accuracy, loss, confusion matrix, CV, scores BLEU/chrF/WER, architecture, hyperparamètres.
