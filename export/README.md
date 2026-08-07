# Utiliser le modèle dans Flutter (`sw_trans_realtime`)

## Idée simple

| Où | Rôle |
|----|------|
| **Ce repo (PC)** | Entraîne et fait tourner le vrai modèle NLLB (+ LoRA) |
| **App Flutter** | Envoie le texte FR et affiche la traduction |

Le téléphone **ne charge pas** le dossier `outputs/` Python. Il parle au modèle via **HTTP** (chemin A) ou via des fichiers **ONNX** (chemin B, plus tard).

```
  [Whisper sur téléphone] → texte FR
           │
           ▼
  [App Flutter]  ──HTTP──►  [PC: python main.py serve]  → texte lingala/…
```

---

## Chemin A — recommandé maintenant (même modèle qu’ici)

### 1. Sur le PC (repo ML)

```powershell
cd C:\Users\Administrator\Desktop\switch\msc-ia-switch-2k26
.\train\.venv\Scripts\Activate.ps1

# Si LoRA pas encore prêt pour toutes les paires :
python main.py serve --baseline

# Quand une paire est entraînée, le serveur charge le LoRA automatiquement :
python main.py serve
```

Tu dois voir : `Serveur prêt pour Flutter` et l’URL `http://127.0.0.1:8765`.

### 2. Dans l’app Flutter

Déjà branché dans `HuggingFaceConfig` :

- Émulateur Android → `http://10.0.2.2:8765`
- Windows desktop → `http://127.0.0.1:8765`
- **Téléphone physique** → mets l’IP du PC dans `localThesisHostOverride`  
  (ex. `192.168.1.42`), même Wi‑Fi, firewall autorise le port **8765**.

Fichier :  
`sw_trans_realtime/lib/config/huggingface_config.dart`

### 3. Tester

1. Laisse `python main.py serve` ouvert  
2. Lance l’app  
3. Traduis FR → Lingala / Swahili / Kikongo / Tshiluba  

L’app appelle **exactement** le modèle chargé sur le PC.

---

## Chemin B — hors-ligne sur le téléphone (plus tard)

Quand le train + export sont finis :

```powershell
python main.py export --config train/configs/fr-ln.yaml --lora --int8
# idem fr-sw, fr-kg, fr-lu
```

Puis copie `exports/<paire>/lora/` vers le stockage de l’app  
(`documents/nllb_models/<paire>/`). Le service `OnnxNllbService` s’en charge  
si les fichiers sont présents (sinon fallback serveur / Google).

> NLLB-600M INT8 ≈ 0.6–1 Go **par paire** : lourd pour un APK.  
> Le chemin A reste le plus clair pour la démo mémoire.

---

## Checklist

- [ ] Venv activé  
- [ ] `python main.py serve` (ou `--baseline`) tourne  
- [ ] App et PC sur le même réseau (ou émulateur)  
- [ ] `useLocalThesisServer = true` dans Flutter  
- [ ] (optionnel) IP PC dans `localThesisHostOverride` pour téléphone réel  
