# Viseio — Projet technique (entretien)

Projet de démonstration pour l'entretien Viseio : détection de personnes et
véhicules par un modèle full CNN quantifié INT8 (≤5 Mo), interrogeable via
un agent conversationnel simple (LLM local).

## Ce qui est inclus dans le clone

Les modèles finaux (`detection/models/*.onnx`, fp32 + INT8 pour
416/512/896px, ~40 Mo), le checkpoint pré-entraîné COCO
(`detection/models/pretrained/`, ~10 Mo) et les vidéos/image de démo
(`demo/assets/`, ~97 Mo) sont commités avec le dépôt — pas besoin de
réentraîner ni de retrouver ses propres vidéos pour lancer la démo. Ce
qui reste à installer après un clone (dépendances Python, code NanoDet
vendored, LLM local) est détaillé section par section plus bas ; voir
en particulier [Entraînement](#entraînement-nanodet-plus-vendored-dans-detectionthird_party)
pour le code NanoDet (nécessaire même pour la démo, les scripts important
directement dessus) et [Agent](#agent-llamacpp--granite-41-3b-d2) pour le
LLM local (uniquement pour `03_live_agent_demo.py`). Une fois ces étapes
suivies, `uv run python setup/check_demo.py` vérifie que tout est bien
en place (voir [Vérifier son installation](#vérifier-son-installation)).

## Structure

- `detection/` — préparation des données, entraînement, export, quantification
- `agent/` — définition des tools, journal d'événements, intégration LLM local
- `demo/` — pipeline live (vidéos), interface de requête (3 scripts, voir
  section [Démo](#démo) plus bas)
- `config/` — paramètres ajustables de `agent/` et `demo/` (tracker,
  agent conversationnel, démos), en YAML plutôt qu'en dur dans le code
- `setup/` — scripts de vérification d'installation (voir plus bas)
- `docs/` — plan de travail et notes

## Environnement

Un venv unique à la racine (`.venv/`, Python 3.12, géré via [uv](https://github.com/astral-sh/uv))
sert pour tout le projet pour l'instant. Chaque composant garde son propre
`requirements.txt` — des environnements séparés pourront être introduits plus
tard si `detection/` et `agent/` finissent par tourner sur des machines
différentes.

```bash
uv venv --python 3.12          # création (déjà fait)
source .venv/Scripts/activate
uv pip install -r detection/requirements.txt
uv pip install -r agent/requirements.txt   # une fois les dépendances définies
```

### Entraînement (NanoDet-Plus, vendored dans `detection/third_party/`)

Le code de [NanoDet](https://github.com/RangiLyu/nanodet) est cloné dans
`detection/third_party/nanodet/` (gitignoré, pas versionné). Pour
reconstituer l'environnement d'entraînement après un clone :

```bash
git clone https://github.com/RangiLyu/nanodet.git detection/third_party/nanodet
# figé sur le commit validé avec ce projet -- les patchs ci-dessous
# ciblent ce code précis et pourraient ne plus s'appliquer sur un master
# plus récent
git -C detection/third_party/nanodet checkout be9b4a9

# torch avec CUDA — ignorer le pin `torch>=1.10,<2.0` du repo (obsolète,
# incompatible Python 3.12), installer torch 2.x à la place
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# reste des dépendances, en excluant torch/torchvision déjà installés
grep -viE '^torch(>=|<|==)|^torchvision' detection/third_party/nanodet/requirements.txt \
  | uv pip install -r -

uv pip install -e detection/third_party/nanodet
```

**Patchs nécessaires** (appliqués directement dans le clone local — à
refaire si le dossier est un jour recloné) :

- `nanodet/data/collate.py` importe `from torch._six import string_classes`,
  un module interne supprimé dans torch≥2.0. Remplacer par
  `string_classes = str` (c'était sa valeur en Python 3).
- `tools/train.py` (ligne ~60) : ajout de
  `torch.set_float32_matmul_precision("high")` avant l'entraînement —
  active le mode TF32 sur les Tensor Cores (gain de vitesse, perte de
  précision négligeable), recommandé par PyTorch pour ce GPU.
- `tools/train.py` (ligne ~100) : le chargement de `schedule.load_model`
  route vers `convert_old_model()` dès que `pytorch-lightning_version` est
  absent du checkpoint — mais les poids "Weight" téléchargeables depuis le
  Model Zoo du repo (`{"state_dict": ...}` seul, sans `epoch`/`iter`) ne
  correspondent pas non plus au format que `convert_old_model()` attend,
  d'où un `KeyError: 'epoch'`. Fix : ne déclencher la conversion que si
  `"state_dict"` est *aussi* absent —
  `if "pytorch-lightning_version" not in ckpt and "state_dict" not in ckpt:`

Poids pré-entraînés dans `detection/models/pretrained/` — commités avec
le dépôt (voir plus haut), pas besoin de les retélécharger ; voir
[docs/justifications.md](docs/justifications.md) pour le lien d'origine
du modèle retenu (NanoDet-Plus-m-1.5x, 416) si besoin de le reconstituer
depuis zéro.

### Agent (llama.cpp + Granite-4.1-3B, D2)

**Modèle retenu : Granite-4.1-3B-Q4_K_M** (IBM, Apache 2.0) --- décidé
après un comparatif étendu à 6 candidats et un banc de test réel sur 4
d'entre eux (36 cas x 10 répétitions), voir
[docs/rapport.tex](docs/rapport.tex) section 9 pour le détail complet
(critères, tableaux comparatifs, résultats, justification). Qwen2.5-7B-Instruct
était le modèle initialement déployé pendant le développement de D1/D2 ;
Granite-4.1-3B l'a remplacé sur la base de ce comparatif --- meilleur
taux de réussite mesuré (97,2% contre 90,6--93,9% pour les 3 autres
candidats), ~2x plus rapide, et surtout **un budget VRAM
nettement plus faible** (2,10 Go de poids contre 4,66--4,92 Go pour les
autres) --- décisif car le serveur LLM partage le seul GPU de la
machine avec tout ce qui tourne en parallèle pendant une démonstration
live (section 9.5 du rapport).

Binaire `llama-server` (gitignoré) dans `agent/third_party/llama.cpp/`,
modèle GGUF (gitignoré) dans `agent/models/`. Pour reconstituer après un
clone :

```bash
mkdir -p agent/third_party/llama.cpp agent/models

# binaires Windows CUDA 12.4 (adapter la version CUDA à ton driver —
# `nvidia-smi` indique la version max supportée ; prendre une version
# CUDA du binaire inférieure ou égale à celle-ci)
curl -sL -o agent/third_party/llama.cpp/llama.zip \
  "https://github.com/ggml-org/llama.cpp/releases/download/<tag>/llama-<tag>-bin-win-cuda-12.4-x64.zip"
curl -sL -o agent/third_party/llama.cpp/cudart.zip \
  "https://github.com/ggml-org/llama.cpp/releases/download/<tag>/cudart-llama-bin-win-cuda-12.4-x64.zip"
# <tag> = tag du build "nightly" le plus récent, cf.
# https://github.com/ggml-org/llama.cpp/releases (les binaires ne sont
# plus attachés au tag "latest" mais à un tag nightly séparé, ex. b10809).
# Chaque build publie plusieurs variantes CUDA (ex. 12.4 et 13.4) : garder
# la plus haute qui reste <= la "CUDA Version" affichée par `nvidia-smi`
# -- ex. driver en CUDA 13.0 -> la 13.4 est exclue, prendre la 12.4
# (vérifié avec b11238). Adapter alors le suffixe `cuda-12.4` des deux URLs.
cd agent/third_party/llama.cpp && unzip -o llama.zip && unzip -o cudart.zip

# modèle -- fichier unique (~2,1 Go), à récupérer depuis le dépôt GGUF
# officiel d'IBM (org `ibm-granite` sur Hugging Face) : chercher
# "granite-4.1-3b" + quantification Q4_K_M, et l'enregistrer sous
# agent/models/granite-4.1-3b-Q4_K_M.gguf. Lien exact non recopié ici
# volontairement -- vérifier l'URL sur huggingface.co avant de curl,
# plutôt que de faire confiance à un lien recopié qui peut devenir
# obsolète (déplacement de repo, renommage de fichier).
```

**Lancer le serveur** (depuis la racine du projet) :

```bash
agent/third_party/llama.cpp/llama-server.exe \
  -m agent/models/granite-4.1-3b-Q4_K_M.gguf \
  --jinja -ngl 99 -c 8192 --port 8080
```

`--jinja` est indispensable — active le tool-calling compatible OpenAI.
`-ngl 99` décharge toutes les couches sur GPU (~2,8-3,3 Go de VRAM avec ce
modèle en Q4_K_M, cf. [docs/rapport.tex](docs/rapport.tex) section 9.2
pour le comparatif de modèles candidats). Contrairement à Hermes-3 et
Functionary (deux des candidats écartés), Granite ne nécessite
**aucun** `--chat-template-file` --- son template de tool-calling
survit correctement à la conversion GGUF.

**Point de vigilance vérifié** (rencontré en téléchargeant les modèles
candidats pendant le comparatif) : en téléchargeant plusieurs gros
fichiers en parallèle (binaires + modèle en même temps), un fichier
GGUF s'est retrouvé tronqué silencieusement (`curl` n'a pas remonté
d'erreur). Symptôme : `llama-server` refuse de charger le modèle
(`tensor ... data is not within the file bounds, model is corrupted or
incomplete`). Vérifier la taille du fichier téléchargé contre le header
`Content-Length` de l'URL (`curl -sI <url>`) en cas de doute plutôt que de
supposer que le téléchargement s'est bien passé.

**Modèles alternatifs évalués** (non retenus, détail et justification
dans [docs/rapport.tex](docs/rapport.tex) section 9.2/9.5) :
Qwen2.5-7B-Instruct, Hermes-3-Llama-3.1-8B, Functionary-small-v3.2. Les
commandes de lancement de chacun (avec leur `--chat-template-file`
respectif pour Hermes-3/Functionary) sont documentées dans
[docs/justifications.md](docs/justifications.md).

### Ce qui n'est pas dans le clone

Tout ce que `.gitignore` exclut, et comment l'obtenir — selon que tu
veuilles juste lancer la démo ou reproduire tout le projet.

**Pour la démo uniquement :**

| Ignoré | Nécessaire ? | Comment l'obtenir |
|---|---|---|
| `detection/third_party/` (code NanoDet) | Oui, toujours — les scripts importent directement dessus | [Entraînement](#entraînement-nanodet-plus-vendored-dans-detectionthird_party) : clone + patchs |
| `agent/third_party/llama.cpp/` (binaire) | Oui, pour `03_live_agent_demo.py` seulement | [Agent](#agent-llamacpp--granite-41-3b-d2) |
| `agent/models/*.gguf` (poids du LLM) | Oui, pour `03_live_agent_demo.py` seulement | [Agent](#agent-llamacpp--granite-41-3b-d2) |
| `.venv/` | Oui, toujours | `uv venv` + `uv pip install -r ...` (section [Environnement](#environnement)) |
| `demo/assets/custom/*` | Non, optionnel | Vidéos personnelles, voir `custom_videos_dir` plus bas |

**En plus, pour reproduire tout le projet** (préparation des données,
entraînement, quantification, banc de test agent) :

| Ignoré | Comment l'obtenir |
|---|---|
| `detection/data/01_annotations/` (COCO brut) | `detection/src/01_data/data_download.py` (télécharge depuis les annotations COCO officielles) |
| `detection/data/02_filtered/` à `05_checks/` | Régénérés par le pipeline `detection/src/01_data/` (`data_filter.py` → `data_download.py` → `data_prepare.py`) |
| `workspace/` (checkpoints/logs d'entraînement, ~14 Go) | Régénéré par un réentraînement (`nanodet/tools/train.py`) |

Rien à télécharger pour `detection/models/*.onnx`,
`detection/models/pretrained/*.pth` ni `demo/assets/*.mp4`/`.jpg`/`.webp`
— commités avec le dépôt (voir [Ce qui est inclus dans le
clone](#ce-qui-est-inclus-dans-le-clone) en haut).

### Vérifier son installation

```bash
uv run python setup/check_demo.py      # tout ce qu'il faut pour la démo
uv run python setup/check_project.py   # + tout le pipeline détection/agent (inclut check_demo.py)
```

Chaque script liste ce qui manque — `MANQUANT` bloque, `absent
(optionnel)` dépend de ce que tu comptes faire (GPU CUDA pour
réentraîner, LLM local pour `03_live_agent_demo.py`) — plutôt que de
planter avec une trace Python à la première étape oubliée.

### Démo

3 scripts, une brique de plus à chaque fois — pratique pour présenter le
système en le construisant sous les yeux plutôt que de balancer le
pipeline complet d'un coup. Rangés dans `demo/src/scripts/`, préfixés
par leur ordre (`01_`...`03_`) pour ne jamais avoir à deviner lequel
lancer en premier ; `demo/src/common/` regroupe les 3 fichiers partagés
entre eux (`config.py`, `fps_counter.py`, `source_cycle.py`). Seul le
dernier (`03_live_agent_demo.py`) a besoin de `llama-server` (section
précédente) ; tous importent `agent/src/` (tracker, journal, alerte
selon le script) en plus de la détection ONNX INT8. Lancer depuis la
racine du projet. Touche `c` dans la fenêtre vidéo pour changer de
source (une des vidéos de démo, en cycle), `r` pour redémarrer la vidéo
courante, `q` pour quitter. FPS et source courante affichés en overlay
(coin haut-gauche). Pas de webcam (retirée du projet, non pertinente
pour ce cas d'usage) : uniquement des vidéos, celles fournies par défaut
ou les vôtres (voir `custom_videos_dir` ci-dessous).

Pas de script dédié pour C2 (journal) ou D3 (alerte temps réel)
isolément : les deux se démontrent directement dans
`03_live_agent_demo.py` (durées d'activité consultables via l'agent,
alerte configurable en direct, ex. « préviens-moi si une personne reste
plus de 10 secondes ») plutôt qu'avec un script séparé par brique.

**`config/demo.yaml`** (chargé par `demo/src/common/config.py`) —
configuration partagée par les 3 scripts (un seul fichier à éditer
plutôt que chacun individuellement) :
- `demo_sources` : la séquence cyclée par la touche `c` — plusieurs
  vidéos à densité de circulation croissante (licence Pixabay, libre
  d'usage). Changer de source réinitialise le tracker et vide le journal
  (les `tracker_id` repartent de zéro par classe sur la nouvelle scène).
- `custom_videos_dir` (`demo/assets/custom/` par défaut) : déposer un
  fichier vidéo (`.mp4`/`.avi`/`.mov`/`.mkv`/`.webm`) dans ce dossier
  l'ajoute automatiquement au cycle des sources au prochain lancement,
  à la suite de `demo_sources`, sans éditer le YAML — le libellé affiché
  est dérivé du nom de fichier.
- `active_model` : `"896px"` (mAP INT8=0,466, meilleure précision du
  projet, retenu — voir [docs/rapport.tex](docs/rapport.tex) section 6)
  ou `"512px"`/`"416px"` pour comparer (plus rapides, moins précis).
  `onnx_path` et `input_size` sont dérivés automatiquement du modèle
  choisi (toujours groupés pour éviter un mismatch de résolution).
- `DISPLAY_MAX_WIDTH`, `RAW_SCORE_THRESHOLD`, `PERSON_ALERT_THRESHOLD_SECONDS`
  — réglages d'affichage et d'alerte, voir les commentaires du fichier.

1. **`demo/src/scripts/01_detection_demo.py`** — détection brute
   uniquement, sans tracker/journal/agent : juste les boîtes du modèle
   ONNX INT8, pour juger la sortie du détecteur isolément du reste du
   pipeline.
   ```bash
   uv run python demo/src/scripts/01_detection_demo.py
   ```
2. **`demo/src/scripts/02_tracking_demo.py`** — ajoute C1 (tracking
   multi-classe) : boîtes avec `tracker_id` persistant, sans journal ni
   alerte. Pour juger la stabilité du tracker seul (ID qui tient face à
   une occlusion brève).
   ```bash
   uv run python demo/src/scripts/02_tracking_demo.py
   ```
3. **`demo/src/scripts/03_live_agent_demo.py`** — pipeline complet
   C1+C2+D3+E1/E2 : tracking + journal + alerte (D3, une règle de
   présence posée en dur au démarrage en plus de celles posables en
   direct) + l'agent conversationnel D1/D2, interrogeable en direct — y
   compris pour configurer une alerte à la volée (ex. « préviens-moi si
   une personne reste plus de 10 secondes »), plutôt que de démontrer
   C2/D3 dans des scripts séparés sans LLM. Ouvre trois fenêtres : la
   vidéo (overlay détections/tracking + bannière d'alerte), une fenêtre
   **Logs** (trace complète du logger racine, tools/arguments choisis
   par l'agent) et une fenêtre **Assistant** (historique de conversation
   + champ de saisie, pré-remplie au démarrage d'exemples de questions
   et du rappel des raccourcis clavier). Chaque question lance son
   propre thread pendant que la vidéo continue de tourner sans
   interruption (voir [docs/justifications.md](docs/justifications.md)
   §E1/E2 pour le détail du thread + de la file `queue.Queue`
   utilisés). C'est le script à utiliser pour la démo présentée à
   l'entretien.
   ```bash
   uv run python demo/src/scripts/03_live_agent_demo.py
   # puis, à tout moment, dans la fenêtre Assistant :
   #   combien de personnes maintenant ?
   ```

### Ajouter une scène avec zones de danger

D3 (alerte de zone) surveille des zones de danger définies **par scène**
(`config/zones.yaml`, clé `scenes`) : chaque scène associe un chemin de
source (une vidéo, ou l'image de fond d'une scène interactive) à son
propre jeu de zones nommées, sélectionné automatiquement dans
`03_live_agent_demo.py` selon la source actuellement affichée — une
source sans scène configurée n'a simplement aucune zone surveillée. La
scène fournie par défaut (`chantier`, deux zones `zone_centrale` et
`zone_laterale`) reste inchangée quoi que vous ajoutiez.

`agent/src/alerts/define_zones.py` dessine ces zones à la souris sur une image
fixe ou une vidéo, puis imprime les blocs YAML prêts à coller — il
n'écrit jamais les fichiers de config directement, pour ne jamais écraser
les commentaires déjà présents dedans.

Deux types de scène :

- **Scène interactive** (image + silhouette détourée qui suit la
  souris, comme le chantier fourni) — passer `--sprite` :
  ```bash
  uv run python agent/src/alerts/define_zones.py demo/assets/mon_fond.jpg \
    --scene-name mon_site --zones zone_a zone_b \
    --sprite demo/assets/ma_silhouette.png
  ```
  Colle le premier bloc imprimé sous `scenes:` dans `config/zones.yaml`,
  et le second sous `demo_sources:` dans `config/demo.yaml` (obligatoire
  pour cette scène : c'est la seule façon de la rendre sélectionnable
  par la touche `c`).

- **Vraie vidéo** (zones vérifiées sur les vraies détections
  personne/voiture) — ne pas passer `--sprite` :
  ```bash
  uv run python agent/src/alerts/define_zones.py demo/assets/custom/ma_video.mp4 \
    --scene-name mon_couloir --zones entree sortie
  ```
  Colle le bloc imprimé sous `scenes:` dans `config/zones.yaml`. Si la
  vidéo est déjà reprise par `demo_sources` ou déposée dans
  `demo/assets/custom/` (voir plus haut), rien d'autre à faire — seul le
  chemin compte pour reconnaître la scène, pas son libellé.

Contrôles pendant le dessin : clic gauche = ajoute un point, clic droit
= annule le dernier, `n` = ferme la zone en cours (3 points minimum) et
passe à la suivante, `r` = redémarre la zone en cours, `q` = quitte et
imprime le résultat. `--zones` accepte autant de noms que nécessaire, pas
seulement deux. `--frame-index N` choisit une autre frame qu'un
lancement à froid d'une vidéo. Recalibrer une scène déjà définie
(`--scene-name` déjà utilisé) est un usage normal : le script prévient
juste que cette scène existe déjà, sans bloquer.

## Licence

[MIT](LICENSE) pour le code de ce dépôt (`agent/`, `demo/`,
`detection/src/`, `config/`). Ne couvre ni le code vendored
(`detection/third_party/`, `agent/third_party/`, sous leurs licences
respectives), ni les vidéos de démo (licence Pixabay, voir
`config/demo.yaml`), ni les poids de modèles tiers (NanoDet-Plus,
Granite — voir leurs dépôts d'origine).
