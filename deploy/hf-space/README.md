---
title: MSC IA Switch NLLB
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: false
license: mit
short_description: FR to Lingala Swahili Kikongo Tshiluba NLLB
---

# Space Gradio — traduction FR → langues RDC

## Hardware (important)

Dans **Settings → Hardware** du Space, choisis **CPU basic** (gratuit).

Ne choisis **pas** ZeroGPU sauf si tu as un compte compatible : ZeroGPU exige `@spaces.GPU` (déjà géré dans `app.py`, mais le CPU free est plus simple).

## Variables (Settings → Variables)

| Variable | Exemple |
|----------|---------|
| `BASE_MODEL` | `facebook/nllb-200-distilled-600M` |
| `ADAPTER_REPO` | `TON_USER/msc-ia-switch-lora` |
| `HF_TOKEN` | token read (évite rate-limit Hub) |

Sans `ADAPTER_REPO` → NLLB de base.

## API Flutter

`POST /gradio_api/call/translate` — voir `deploy/README.md`.

Premier appel lent (téléchargement / chargement du modèle ~2.5 Go).
