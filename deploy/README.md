# Déployer le modèle gratuitement (Hugging Face Spaces — Gradio)

Docker Spaces n’est plus gratuit. Utilise le SDK **Gradio** (toujours gratuit).

```
App Flutter  ──Internet──►  Space Gradio (ton NLLB ± LoRA)
```

Le Space peut s’endormir ; le 1er appel le réveille (1–3 min).

---

## 1. Compte + token

https://huggingface.co → token **Write** → https://huggingface.co/settings/tokens

```powershell
pip install huggingface_hub
huggingface-cli login
```

## 2. (Optionnel) Uploader les LoRA après train

```powershell
cd C:\Users\Administrator\Desktop\switch\msc-ia-switch-2k26
huggingface-cli repo create msc-ia-switch-lora --type model

huggingface-cli upload TON_USER/msc-ia-switch-lora outputs/nllb-lora-fr-ln/adapter fr-ln
huggingface-cli upload TON_USER/msc-ia-switch-lora outputs/nllb-lora-fr-sw/adapter fr-sw
huggingface-cli upload TON_USER/msc-ia-switch-lora outputs/nllb-lora-fr-kg/adapter fr-kg
huggingface-cli upload TON_USER/msc-ia-switch-lora outputs/nllb-lora-fr-lu/adapter fr-lu
```

Sans LoRA → le Space tourne en **baseline** NLLB (déjà utile).

## 3. Créer le Space Gradio

1. https://huggingface.co/new-space  
2. Nom : `msc-ia-switch-api`  
3. SDK : **Gradio** (pas Docker)  
4. Hardware : **CPU basic** (gratuit)

Pousse le contenu de `deploy/hf-space/` :

```powershell
cd deploy\hf-space

# Clone le Space vide (remplace TON_USER)
git clone https://huggingface.co/spaces/TON_USER/msc-ia-switch-api
# Git Bash :
cp app.py requirements.txt README.md msc-ia-switch-api/
# PowerShell :
# Copy-Item app.py,requirements.txt,README.md msc-ia-switch-api\

cd msc-ia-switch-api
git add .
git commit -m "Gradio NLLB API for Flutter"
git push
```
**Settings → Variables** du Space :

- `ADAPTER_REPO` = `TON_USER/msc-ia-switch-lora` (si uploadé)

Attends le build. Ouvre l’onglet **App** et teste une phrase.

URL typique : `https://huggingface.co/spaces/TON_USER/msc-ia-switch-api`  
API : `https://TON_USER-msc-ia-switch-api.hf.space`

## 4. Brancher Flutter (PC éteint OK)

Dans `sw_trans_realtime/lib/config/huggingface_config.dart` :

```dart
static const bool useLocalThesisServer = false;

static const String customNllbApiBaseUrl =
    'https://TON_USER-msc-ia-switch-api.hf.space';

/// true = ton Space Gradio ; false = ancienne API type /api/v4/translator
static const bool customSpaceIsGradio = true;
```

Relance l’app.

---

## Limites

| Option | Gratuit ? | Tes LoRA ? |
|--------|-----------|------------|
| **Space Gradio CPU** | Oui | Oui |
| Space public winstxnhdw | Oui | Non |
| `python main.py serve` | Oui | Oui (PC allumé) |
| ONNX sur téléphone | Oui | Oui (lourd) |
| Docker Space / GPU | Payant | Oui |

Hors-ligne total (sans Internet) → export ONNX, voir `export/README.md`.
