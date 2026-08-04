# Datasets — FR → langues nationales (RDC)

Corpus parallèles français → swahili / lingala / kikongo / tshiluba, alignés sur les paires OPUS-MT Helsinki-NLP.

## Paires

| Dossier | Langue cible | Code OPUS | Modèle OPUS-MT de référence |
|---------|--------------|-----------|------------------------------|
| `fr-sw` | Swahili (Congo `swc` + Swahili standard `sw`) | `swc` / `sw` | [fr-swc](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/fr-swc) |
| `fr-ln` | Lingala | `ln` | [fr-ln](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/fr-ln) |
| `fr-kg` | Kikongo | `kg` | [fr-kg](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/fr-kg) |
| `fr-lu` | Tshiluba (`lua` ; `lu` quasi vide sur OPUS) | `lua` / `lu` | [fr-lu](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/fr-lu), [fr-lua](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/fr-lua) |

## Modèles OPUS-MT (référence)

Ces modèles Marian ont été entraînés sur OPUS (benchmark JW300). Les poids et scores sont sur CSC Object Storage :

| Paire | Archive | BLEU (JW300) |
|-------|---------|--------------|
| fr-swc | https://object.pouta.csc.fi/OPUS-MT-models/fr-swc/opus-2020-01-16.zip | 28.2 |
| fr-ln | https://object.pouta.csc.fi/OPUS-MT-models/fr-ln/opus-2020-01-09.zip | 30.5 |
| fr-kg | https://object.pouta.csc.fi/OPUS-MT-models/fr-kg/opus-2020-01-09.zip | 30.4 |
| fr-lu | https://object.pouta.csc.fi/OPUS-MT-models/fr-lu/opus-2020-01-20.zip | 25.5 |
| fr-lua | https://object.pouta.csc.fi/OPUS-MT-models/fr-lua/opus-2020-01-09.zip | 27.3 |

Aussi sur Hugging Face : `Helsinki-NLP/opus-mt-fr-{swc,ln,kg,lu,lua}`.

> **Note :** JW300 n’est plus distribué publiquement sur OPUS. Les bitexts téléchargés ici viennent surtout de **NLLB** et de corpus plus petits (Tatoeba, Wikimedia, GlobalVoices, TED2020, tico-19, GNOME, …).

## Contenu par dossier

```
dataset/<paire>/
  raw/      # archives .zip OPUS (moses)
  moses/    # bitexts extraits (*.fr / *.<tgt>)
```

Snapshots API OPUS : `dataset/sources/opus_fr-*.json`.

## Volumes (paires approximatives)

| Paire | Corpus principal | ~paires |
|-------|------------------|---------|
| fr-ln | NLLB | 691 883 |
| fr-kg | NLLB | 501 059 |
| fr-lu | NLLB (`fr-lua`) | 592 536 |
| fr-sw | Tatoeba `fr-swc` (172) + GlobalVoices / TED / tico-19 / wikimedia (`fr-sw`) | ~32 k (hors NLLB) |

Pour Swahili, NLLB `fr-sw` (~5.6 M paires, ~344 Mo) est optionnel :

```powershell
.\dataset\download.ps1 -IncludeNllbSw
```

## Téléchargement

```powershell
.\dataset\download.ps1
```

Sources : [OPUS](https://opus.nlpl.eu/), [OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train).
