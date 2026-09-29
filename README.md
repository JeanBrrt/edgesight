# EdgeSight — Détection embarquée et agent conversationnel

Projet personnel, conçu pour démontrer et explorer, de bout en bout, une
chaîne de vision par ordinateur pensée pour l'embarqué : détection de
personnes et de véhicules par un modèle full CNN quantifié INT8
(≤5 Mo), suivi multi-objets, journal d'événements et alertes en temps
réel, le tout interrogeable en langage naturel via un agent
conversationnel qui s'appuie sur un LLM local (tool calling).

Compétences mises en œuvre : préparation de données (COCO), fine-tuning
et évaluation d'un détecteur (NanoDet-Plus), une toolchain de
quantification INT8 maison, sans outil clé en main (export ONNX et
validation numérique contre PyTorch, lecteur de calibration dédié,
quantification statique QDQ par canal, comparaison des méthodes de
calibration, calibration par tranches pour borner la RAM, évaluation
mAP fp32 vs INT8), benchmark de latence, tracking (ByteTrack),
conception d'outils pour un agent LLM et banc de test comparatif de
modèles.

## Sommaire

- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Démo](#démo)
- [Ajouter une scène avec zones de danger](#ajouter-une-scène-avec-zones-de-danger)
- [Rapport technique](#rapport-technique)
- [Aller plus loin](#aller-plus-loin)
- [Contact](#contact)
- [Licence](#licence)

## Structure du dépôt

- `detection/` — préparation des données, entraînement, export ONNX,
  quantification INT8, évaluation
- `agent/` — tracking, journal d'événements, alertes, agent LLM et ses
  outils
- `demo/` — les 3 scripts de démo et leurs assets (`scenes/`,
  `silhouettes/`, `zone_previews/`, `custom/` pour vos vidéos)
- `config/` — réglages en YAML (démo, agent, tracker, zones)
- `setup/` — scripts de vérification de l'installation
- `docs/` — rapport technique et justifications des choix

Déjà inclus dans le clone : les modèles ONNX (fp32 + INT8, 416/512/896px),
le checkpoint pré-entraîné, les configs NanoDet et les vidéos de démo.
Pas besoin de réentraîner pour tester.

## Installation

Testé sous **Windows 11** (Git Bash), Python 3.12, GPU NVIDIA
(RTX 4060 Laptop). Commandes à lancer dans l'ordre, depuis la racine du
projet.

**Prérequis** : [uv](https://github.com/astral-sh/uv), `git`, `curl`,
`unzip`, un GPU NVIDIA (pour le LLM local).

**1. Environnement Python**

```bash
uv venv --python 3.12
source .venv/Scripts/activate
uv pip install -r detection/requirements.txt
uv pip install -r agent/requirements.txt
uv pip install -r demo/requirements.txt
```

**2. Code NanoDet** (figé sur le commit validé, sans écraser les configs
du projet)

```bash
git clone https://github.com/RangiLyu/nanodet.git /tmp/nanodet_upstream
git -C /tmp/nanodet_upstream checkout be9b4a9
mkdir -p detection/third_party/nanodet
cp -rn /tmp/nanodet_upstream/. detection/third_party/nanodet/
rm -rf /tmp/nanodet_upstream
```

**3. torch + dépendances NanoDet** — la version CPU suffit pour la démo ;
la version CUDA ne sert qu'à réentraîner.

```bash
uv pip install -r detection/requirements-torch-cpu.txt     # démo
# ou : uv pip install -r detection/requirements-torch-cu124.txt  (réentraînement)

grep -viE '^torch(>=|<|==)|^torchvision' detection/third_party/nanodet/requirements.txt \
  | uv pip install -r -
uv pip install -e detection/third_party/nanodet
```

**4. Patchs NanoDet** (compatibilité torch 2.x)

```bash
cd detection/third_party/nanodet
sed -i 's/^from torch._six import string_classes$/string_classes = str/' nanodet/data/collate.py
sed -i '/torch.backends.cudnn.benchmark = True/a\    torch.set_float32_matmul_precision("high")' tools/train.py
sed -i 's/if "pytorch-lightning_version" not in ckpt:/if "pytorch-lightning_version" not in ckpt and "state_dict" not in ckpt:/' tools/train.py
cd -
```

**5. llama.cpp** — prendre le build le plus récent sur
[les releases](https://github.com/ggml-org/llama.cpp/releases) (`TAG`),
et la variante CUDA la plus haute qui reste ≤ la « CUDA Version »
affichée par `nvidia-smi` (`CUDA`).

```bash
TAG=b11238
CUDA=12.4
mkdir -p agent/third_party/llama.cpp agent/models
curl -L -o agent/third_party/llama.cpp/llama.zip \
  "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-bin-win-cuda-$CUDA-x64.zip"
curl -L -o agent/third_party/llama.cpp/cudart.zip \
  "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/cudart-llama-bin-win-cuda-$CUDA-x64.zip"
unzip -oq agent/third_party/llama.cpp/llama.zip -d agent/third_party/llama.cpp
unzip -oq agent/third_party/llama.cpp/cudart.zip -d agent/third_party/llama.cpp
```

**6. Modèle LLM** — Granite-4.1-3B (~2,1 Go), depuis le dépôt officiel
[IBM](https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF).

```bash
curl -L -o agent/models/granite-4.1-3b-Q4_K_M.gguf \
  "https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/main/granite-4.1-3b-Q4_K_M.gguf"
ls -l agent/models/granite-4.1-3b-Q4_K_M.gguf   # attendu : 2 099 501 664 octets
```

Si le serveur refuse ensuite de charger le modèle (`model is corrupted
or incomplete`), le téléchargement a été tronqué : le relancer.

**7. Vérifier l'installation**

```bash
uv run python setup/check_demo.py   # doit finir par "Tout est en place..."
```

Le script liste ce qui manque le cas échéant, étape par étape.

## Démo

3 scripts, chacun ajoutant une brique au précédent. À lancer depuis la
racine du projet.

| Script | Contenu |
|---|---|
| `01_detection_demo.py` | Détection seule : les boîtes du modèle INT8 |
| `02_tracking_demo.py` | + suivi multi-objets (identifiants persistants) |
| `03_live_agent_demo.py` | Pipeline complet : suivi, journal, alertes et assistant conversationnel |

```bash
uv run python demo/src/scripts/03_live_agent_demo.py
```

**`03_live_agent_demo.py` est la démonstration complète.** Il lance
lui-même le serveur LLM (quelques secondes au démarrage) et l'arrête en
quittant. Il ouvre la vidéo et une fenêtre **Assistant**, pré-remplie
d'exemples de questions, par exemple :

- « Combien de personnes y a-t-il en ce moment ? »
- « Préviens-moi si une personne reste plus de 10 secondes »
- « Alerte-moi si quelqu'un entre dans la zone centrale » (scène chantier)

Le panneau de droite affiche les outils appelés par l'agent et les
valeurs qu'ils ont renvoyées.

**Commandes dans la fenêtre vidéo** : `c` change de scène (4 vidéos,
puis une scène interactive de chantier où une silhouette suit la
souris), `r` redémarre la vidéo, `q` quitte.

**Réglages** (`config/demo.yaml`) : modèle utilisé (`active_model` :
896px par défaut, 512px ou 416px plus rapides), scènes du cycle. Toute
vidéo déposée dans `demo/assets/custom/` est ajoutée automatiquement au
cycle.

Pour garder le serveur LLM ouvert entre plusieurs lancements (évite de
recharger le modèle), le lancer à part ; la démo le réutilisera :

```bash
agent/third_party/llama.cpp/llama-server.exe \
  -m agent/models/granite-4.1-3b-Q4_K_M.gguf --jinja -ngl 99 -c 8192 --port 8080
```

## Ajouter une scène avec zones de danger

Les zones sont définies par scène dans `config/zones.yaml`.
`agent/src/alerts/define_zones.py` permet de les dessiner à la souris,
puis affiche les blocs YAML à coller dans la config.

```bash
# Sur une vidéo
uv run python agent/src/alerts/define_zones.py demo/assets/custom/ma_video.mp4 \
  --scene-name mon_couloir --zones entree sortie

# Sur une image, avec une silhouette détourée (PNG RGBA) qui suit la souris
uv run python agent/src/alerts/define_zones.py demo/assets/scenes/06_mon_fond.jpg \
  --scene-name mon_site --zones zone_a zone_b \
  --sprite demo/assets/silhouettes/ma_silhouette.png
```

Clic gauche : ajouter un point ; clic droit : annuler ; `n` : fermer la
zone ; `r` : recommencer la zone ; `q` : terminer. Un aperçu est
enregistré dans `demo/assets/zone_previews/<scene-name>.png`.

## Rapport technique

[`docs/rapport.tex`](docs/rapport.tex) documente l'ensemble du projet :
les choix faits à chaque étape, les alternatives écartées, et les
mesures qui les justifient (mAP, latence, taille, taux de réussite de
l'agent). C'est la référence pour toute question sur le *pourquoi*
d'un choix ; le README ne couvre que l'installation et la démo.

| Question | Section du rapport |
|---|---|
| Pourquoi NanoDet-Plus plutôt qu'un autre détecteur ? | Choix d'architecture de détection |
| Comment le jeu de données a-t-il été construit ? | Préparation des données |
| Comment le modèle a-t-il été entraîné, et pourquoi 896px ? | Entraînement — trois générations de modèle |
| Comment fonctionne la quantification INT8, et que coûte-t-elle en précision ? | Quantification INT8 — toolchain maison |
| Quels compromis entre précision et vitesse ? | Performance et compromis précision/vitesse |
| Comment le suivi et le journal d'événements fonctionnent-ils ? | Suivi multi-objets et journalisation des événements |
| Pourquoi Granite-4.1-3B, et comment l'agent a-t-il été évalué ? | Agent conversationnel |
| Comment l'ensemble tourne-t-il en temps réel ? | Intégration système et démonstration live |
| Limites et pistes d'amélioration | Synthèse générale et perspectives |

Le rapport est fourni en source LaTeX (figures dans `docs/figures/`) ;
le compiler avec une distribution LaTeX, par exemple
`latexmk -pdf rapport.tex` depuis `docs/`.
[`docs/justifications.md`](docs/justifications.md) complète le rapport
avec les notes de travail détaillées (commandes, tableaux comparatifs
bruts).

## Aller plus loin

- **Reproduire tout le projet** (données COCO, entraînement,
  quantification) : installer torch CUDA à l'étape 3, puis vérifier avec
  `uv run python setup/check_project.py`. Les scripts sont dans
  `detection/src/`, dans l'ordre (`01_data/` → `04_evaluate/`).

## Contact

Pour toute question technique : jean02800@gmail.com

## Licence

[MIT](LICENSE) pour le code de ce dépôt. Ne couvre ni le code tiers
(`detection/third_party/`, `agent/third_party/`), ni les vidéos de démo
(licence Pixabay), ni les poids de modèles tiers (NanoDet-Plus, Granite),
chacun sous sa propre licence.
