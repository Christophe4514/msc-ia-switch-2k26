# msc-ia-switch-2k26

Traduction audio FR → langues nationales RDC (lingala, swahili, tshiluba, kikongo), hors ligne sur mobile.

## Démarrage rapide

```powershell
.\train\.venv\Scripts\Activate.ps1

# Entraînement LoRA rapide (12 epochs max + early stopping, BLEU en fin)
# Stopper d'abord tout ancien train 30-epochs encore en cours (Ctrl+C)
python main.py train --config train/configs/fr-sw.yaml
python main.py train-all

# Test baseline / LoRA
python main.py test --config train/configs/fr-kg.yaml --baseline --interactive
python main.py test --config train/configs/fr-ln.yaml --interactive

# Évaluation / export ONNX Flutter
python main.py evaluate --config train/configs/fr-ln.yaml --split test
python main.py export --config train/configs/fr-ln.yaml --baseline

# Brancher l'app Flutter sur CE modèle (PC = serveur)
python main.py serve --baseline
# Puis lancer sw_trans_realtime (voir export/README.md)
```

**Flutter** : le téléphone n’importe pas le dossier Python. Lance `serve`, l’app appelle `http://10.0.2.2:8765` (émulateur). Détail : `export/README.md`.

## Config rapide (pourquoi c’est plus court)

| Réglage | Avant | Maintenant |
|---------|-------|------------|
| Epochs | 30 | **12** + early stop (patience 3) |
| BLEU pendant train | chaque epoch (lent) | **seulement à la fin** |
| Batch | 4×8 | **8×4** (même batch effectif, moins de steps) |
| Gros corpus (ln/kg/lu) | tout | **80k** paires max |

Relancer `fr-sw` avec la nouvelle config (Ctrl+C sur l’ancien job d’abord).
