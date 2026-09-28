Non # Justifications — choix d'architecture (B1)

## Comparatif des candidats

| Modèle                         | Full CNN | Graphe exporté propre          | Code source transparent | Poids pré-entraînés indépendants | Taille fp32 | Taille INT8 estimée | mAP COCO 80 classes (val2017, AP 0.5:0.95) |
| ------------------------------ | -------- | ------------------------------ | ----------------------- | -------------------------------- | ----------- | ------------------- | ------------------------------------------ |
| **YOLOX-Nano**                 | Oui      | Oui — décodage/NMS hors graphe | Bon                     | Oui                              | ~3,6 Mo     | ~0,9-1 Mo           | ~25,8%                                     |
| **YOLOX-Tiny**                 | Oui      | Oui (même famille)             | Bon                     | Oui                              | ~20,2 Mo    | ~5 Mo — limite      | ~32,8%                                     |
| **NanoDet-Plus-m (1.0x), 320** | Oui      | Oui — décodage/NMS hors graphe | Très bon                | Oui                              | ~4,7 Mo     | ~1,2 Mo             | ~27,0%                                     |
| **NanoDet-Plus-m (1.0x), 416** | Oui      | Oui (même famille)             | Très bon                | Oui                              | ~4,7 Mo     | ~1,2 Mo             | ~30,4%                                     |
| **NanoDet-Plus-m-1.5x, 320**   | Oui      | Oui (même famille)             | Très bon                | Oui                              | ~9,8 Mo     | ~2,4 Mo             | ~29,9%                                     |
| **NanoDet-Plus-m-1.5x, 416**   | Oui      | Oui (même famille)             | Très bon                | Oui                              | ~9,8 Mo     | ~2,4 Mo             | ~34,1%                                     |
| **YOLOv5n**                    | Oui      | Oui — décodage/NMS hors graphe | Bon (Ultralytics)       | Oui                              | ~14,8 Mo    | ~1,9 Mo             | 28,0%                                       |
| **YOLOv8n**                    | Oui      | Oui — décodage/NMS hors graphe | Bon (Ultralytics)       | Oui                              | ~6,3 Mo     | ~3,2 Mo             | **37,3%**                                   |
| **PP-PicoDet-XS**              | Oui      | Partiel — écosystème PaddlePaddle | Bon, mais hors PyTorch | Oui                             | ~2,8 Mo     | ~0,7 Mo             | >30%¹                                       |
| **MobileNetV2 + SSDLite**      | Oui      | Oui — NMS hors graphe          | Très bon (torchvision)  | Oui                              | ~17,2 Mo    | ~4,3 Mo — peu de marge | ~22,3%                                    |
| **EfficientDet-Lite0**         | Oui      | Non — export natif TFLite      | Bon, mais hors PyTorch/ONNX | Oui                          | ~12,8 Mo    | 4,4 Mo (officiel)   | 25,7%                                       |

¹ PP-PicoDet revendique être "le premier détecteur à dépasser 30% de mAP
sous 1M de paramètres à 416px" (README PaddleDetection) — pas de chiffre
exact par variante trouvé pour PicoDet-XS spécifiquement, formulé avec
cette prudence plutôt que comme une valeur mesurée.

Cette estimation params × 4 octets est une approximation fp32 pure — elle
ignore les buffers de BatchNorm, les métadonnées ONNX, etc. Une fois que tu
auras réellement exporté un modèle, vérifie la taille réelle du fichier
plutôt que de te fier à ce calcul. Chiffres des 4 nouveaux candidats
obtenus par recherche web (pas de mémoire figée), sources en bas de
section.

### Pourquoi ces 4 candidats sont écartés malgré des chiffres compétitifs

Ajoutés après coup (le comparatif initial ne couvrait que 2 familles,
YOLOX et NanoDet-Plus — un vrai angle mort signalé par l'utilisateur,
pas un choix délibéré à l'origine). Les inclure honnêtement plutôt que
prétendre que le tableau initial était exhaustif.

- **YOLOv8n** a un meilleur mAP que notre candidat retenu (37,3% vs
  34,1%) — l'écarter uniquement sur la précision serait malhonnête. Le
  vrai motif est la **licence AGPL-3.0** d'Ultralytics (couvre YOLOv5 et
  YOLOv8 depuis l'unification de leur package) : réutilisation en
  contexte entreprise sans licence commerciale payante oblige à publier
  tout le code source du projet qui l'intègre. Motif professionnel
  concret, pas un prétexte technique construit après coup.
- **YOLOv5n** : même famille/licence, mAP par ailleurs déjà inférieur à
  NanoDet-Plus-m-1.5x (28,0% vs 34,1%) — écarté doublement.
- **PP-PicoDet** : chiffres mAP/taille potentiellement compétitifs
  (peut-être meilleurs que NanoDet à taille égale), mais natif
  **PaddlePaddle**, pas PyTorch. Notre toolchain de quantification
  maison (B3) s'appuie sur `torch.onnx.export` directement depuis les
  poids entraînés — repartir de PaddlePaddle demanderait soit
  `Paddle2ONNX` (une conversion supplémentaire, une source d'erreur et
  d'incertitude sur la fidélité du graphe en plus), soit réentraîner
  dans un framework différent. Coût d'intégration disproportionné pour
  un projet où le levier différenciant est justement la toolchain de
  quantification, pas l'architecture elle-même.
- **MobileNetV2+SSDLite** et **EfficientDet-Lite0** : mAP nettement plus
  faible (~22,3% et 25,7%) pour une taille comparable ou supérieure —
  écartés sur un critère purement quantitatif, pas d'excuse
  supplémentaire nécessaire. `EfficientDet-Lite0` a en plus un export
  natif TFLite (écosystème TensorFlow), pas ONNX/PyTorch — même
  friction d'intégration que PP-PicoDet, en supplément du mAP plus
  faible.

Sources : [YOLOv5/YOLOv8 metrics et tailles](https://docs.ultralytics.com/models/yolov8),
[licence Ultralytics AGPL-3.0](https://medium.com/@bingbai.jp/yolo-model-licenses-a-developers-guide-da722767b6f8),
[PP-PicoDet README](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/configs/picodet/README_en.md),
[SSDLite MobileNetV3 (torchvision)](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html),
[EfficientDet-Lite0 (TF Hub)](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/tensorflow/models/efficientdet/lite0/detection/1.md).

## Décision

**Backbone retenu : NanoDet-Plus-m-1.5x, résolution 416.**

### Pourquoi pas la taille minimale

Les deux finalistes envisagés (NanoDet-Plus-m 1.0x et NanoDet-Plus-m-1.5x,
tous deux à 416) respectent très confortablement la contrainte ≤5 Mo une
fois quantifiés (~1,2 Mo et ~2,4 Mo estimés). La contrainte de taille est un
plafond binaire, pas un objectif à minimiser au-delà de ce qui est demandé
— une fois respectée, gagner encore en légèreté n'apporte plus de signal
différenciant. Le vrai delta entre les deux candidats est la précision
(~30,4% vs ~34,1% de mAP COCO 80 classes), un gain réel et mesurable sur la
qualité du produit final.

Argument retenu : **utiliser la marge disponible sous la contrainte pour
maximiser la précision, plutôt que de minimiser la taille au-delà de ce qui
était demandé** — une lecture fine de la contrainte (5 Mo comme budget à
exploiter, pas comme objectif à sous-utiliser par excès de prudence).

### Sur la transférabilité du mAP 80 classes vers 2 classes (person/car)

Un modèle à ~30-34% de mAP sur COCO 80 classes a de bonnes raisons de
dépasser ce score sur une tâche à 2 classes (person/car) :

- **`person` et `car` sont des classes "faciles" par rapport à la moyenne
  COCO** — le mAP 80-classes est tiré vers le bas par des catégories rares
  ou ambiguës. `person` (classe la mieux représentée dans COCO) obtient
  typiquement un AP par classe ~1,3-1,6× le mAP moyen d'un modèle ; `car`
  se situe généralement autour de ~1,0-1,3×.
- **La classification devient triviale** — trancher entre 2 classes + fond
  au lieu de 80 catégories visuellement proches allège mécaniquement la
  partie classification du score, sans changer la difficulté de
  localisation des boîtes.

Extrapolation très grossière (facteur ~1,2-1,8× le mAP 80-classes, à ne pas
citer telle quelle — aucune mesure contrôlée ne la valide) :
NanoDet-Plus-m-1.5x @416 pourrait atteindre quelque chose comme **~40-55%**
sur person/car. Le chiffre réel viendra uniquement de B2 (entraînement +
calcul du mAP sur notre `test.json`), seule mesure fiable pour cette
tâche précise.

## Hyperparamètres d'entraînement (B2.4)

Le config dupliqué en B2.2 (`config/nanodet-plus-m-1.5x_416-person-car.yml`)
contenait encore les valeurs par défaut du repo, pensées pour un
entraînement **from scratch sur COCO complet** (118k images, 300 epochs,
cluster multi-GPU). Aucune de ces valeurs n'est adaptée à notre cas :
fine-tuning sur 11 732 images, 2 classes, un seul GPU laptop (RTX 4060,
8 Go VRAM, 12 cœurs CPU), délai de projet serré.

| Paramètre                      | Valeur héritée                      | Valeur retenue                                                | Justification                                                                                                                                                                                                                             |
| ------------------------------ | ----------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device.workers_per_gpu`       | 10                                  | **6**                                                         | 12 cœurs dispo, mais 10 workers + processus principal sature une machine perso utilisée en interactif                                                                                                                                     |
| `device.batchsize_per_gpu`     | 96                                  | **24**                                                        | 96 était calibré pour un GPU datacenter (24-80 Go) ; 8 Go de VRAM impose de réduire fortement. Point de départ empirique, à ajuster si OOM ou si de la marge reste disponible                                                             |
| `device.precision`             | 32                                  | **16 (AMP)**                                                  | La RTX 4060 a des tensor cores — l'AMP réduit la VRAM utilisée et accélère l'entraînement, quasi gratuit en qualité                                                                                                                       |
| `schedule.load_model`          | absent (commenté)                   | **`detection/models/pretrained/nanodet-plus-m-1.5x_416.pth`** | Sans ce champ, l'entraînement serait parti from scratch et aurait perdu tout le bénéfice du transfer learning validé en B2.2                                                                                                              |
| `schedule.optimizer.lr`        | 0.001                               | **0.0001**                                                    | Fine-tuning depuis un modèle déjà bon → LR plus prudent que pour un entraînement from scratch, pour ne pas détruire les features pré-entraînées. Cohérent aussi avec la règle de mise à l'échelle linéaire batch/lr (batch divisé par ~4) |
| `schedule.warmup.steps`        | 500                                 | **200**                                                       | Dataset 10× plus petit que COCO complet + départ depuis des poids déjà entraînés (pas une init aléatoire) → besoin de warmup proportionnellement moindre                                                                                  |
| `schedule.total_epochs`        | 300                                 | **40**                                                        | 300 epochs sur 118k images ; sur 11,7k images en fine-tuning, la convergence arrive bien plus tôt                                                                                                                                         |
| `schedule.lr_schedule.T_max`   | 300                                 | **40**                                                        | Doit rester synchronisé avec `total_epochs` pour que le cosine schedule complète un cycle sur toute la durée réelle de l'entraînement                                                                                                     |
| `schedule.lr_schedule.eta_min` | 0.00005                             | **0.000005**                                                  | Réduit proportionnellement au nouveau `lr` de départ                                                                                                                                                                                      |
| `schedule.val_intervals`       | 10                                  | **4**                                                         | Avec seulement 40 epochs au total, valider tous les 10 epochs ne donnait que 4 points de mesure sur toute la run — trop grossier pour repérer un surapprentissage tôt                                                                     |
| `log.interval`                 | 50                                  | **20**                                                        | Epochs plus courtes (moins d'images) → 50 pas donnait trop peu de points de suivi par epoch                                                                                                                                               |
| `save_dir`                     | `workspace/nanodet-plus-m-1.5x_416` | **`workspace/nanodet-plus-m-1.5x_416-person-car`**            | Éviter de mélanger les checkpoints d'entraînement avec le dossier déjà utilisé par les runs de démo B2.1                                                                                                                                  |
| `grad_clip`, `evaluator`       | 35, `CocoDetectionEvaluator`/`mAP`  | **inchangés**                                                 | Pas de raison de les changer a priori ; l'évaluateur est déjà celui qu'on veut, cohérent avec le mAP de référence prévu en B2.6                                                                                                           |

**Non figé à l'avance** : le batch size réellement tenable en VRAM et le
nombre d'epochs optimal restent empiriques — ces valeurs sont un point de
départ, à ajuster (B2.7) selon le comportement observé au premier run
(OOM, courbe de loss).

## Résultat final fp32 (B2.5/B2.6) — référence pour B3

Entraînement complet, 40/40 epochs (LR final 5,15e-06, cohérent avec
`eta_min`). Le mAP sur `val.json` a plafonné à l'**epoch 24** (0,372),
puis légèrement reculé jusqu'à l'epoch 40 (0,370) — `model_best`
correspond donc à l'epoch 24, sauvegardé automatiquement par la logique
"meilleur seulement" du trainer. Les 16 epochs suivantes n'ont rien
apporté, cohérent avec le budget "30-50 epochs, à ajuster" fixé en B2.4.

Évaluation finale sur `test.json` (jamais vu pendant l'entraînement ni la
sélection du checkpoint, script `detection/src/evaluate.py`) :

| Métrique  | val.json (sélection) | test.json (référence finale) |
| --------- | -------------------- | ---------------------------- |
| mAP       | 0,372                | **0,377**                    |
| AP50      | 0,612                | 0,612                        |
| AP75      | 0,375                | 0,379                        |
| AP_small  | 0,184                | 0,175                        |
| AP_medium | 0,515                | 0,542                        |
| AP_large  | 0,661                | 0,687                        |

Résultat quasi identique entre val et test (petits écarts attribuables au
bruit d'échantillonnage, `test.json` ne fait que ~861 images) — **aucun
signe de surapprentissage** de la sélection de checkpoint sur `val.json`.

**mAP = 37,7% devient la référence fp32 officielle pour B3.6** (comparaison
avant/après quantification INT8). Comparé aux références antérieures :
80-classes publié = 34,1% ; extrapolation spéculative pré-entraînement
= 40-55% (jamais validée, résultat réel un peu plus modeste mais toujours
au-dessus de la baseline 80-classes, cohérent avec l'hypothèse
person/car plus faciles que la moyenne COCO).

Écart persistant `person` (mAP ~44%) vs `car` (mAP ~30%) observé pendant
l'entraînement, stable sur les derniers checkpoints — probable
déséquilibre du nombre d'instances hérité de A2 (voir discussion à
l'epoch 8). À mentionner honnêtement en présentation.

## Quantification PTQ — paramètres retenus (B3.4)

| Paramètre          | Valeur retenue                | Pourquoi                                                                                                                                                                                                                                               |
| ------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `quant_format`     | `QDQ`                         | Portable — n'importe quel runtime peut fusionner les paires QuantizeLinear/DequantizeLinear en kernels int8 à sa façon, pas besoin d'implémenter des kernels quantifiés spécifiques (`QOperator`). Cohérent avec l'absence de contrainte hardware (Q1) |
| `per_channel`      | `True` (≠ défaut lib `False`) | Une échelle par filtre plutôt qu'une seule pour toute la couche — évite le compromis résolution/écrêtage entre filtres de plages différentes, gain de précision quasi gratuit (juste quelques scalaires en plus)                                       |
| `calibrate_method` | `MinMax`                      | Le plus simple, aucun hyperparamètre à régler, une seule passe de calibration (cohérent avec notre `CalibrationDataReader` sans `rewind`). Point de départ raisonnable, pas nécessairement optimal                                                     |

**Piste d'amélioration si le temps le permet (rattaché à B3.7)** : `MinMax` est sensible aux valeurs aberrantes dans les données de calibration (une seule activation extrême étire toute la plage). Tester `CalibrationMethod.Entropy` et/ou `Percentile` sur le même pipeline et comparer le mAP obtenu donnerait un **tableau comparatif supplémentaire** pour la présentation (méthode de calibration × mAP post-quantification) — pas indispensable, mais un bon "plus" si le temps du projet le permet après B3.6.

**Piste additionnelle — `quant_format` (`QDQ` vs `QOperator`) — mesurée empiriquement** :

| Format         | Taille fichier            | Remarque                                                                                                                                                                                                                 |
| -------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `QDQ` (retenu) | 3,12 Mo                   | Aucun warning                                                                                                                                                                                                            |
| `QOperator`    | 2,80 Mo (~10% plus petit) | `onnxruntime` émet lui-même un warning : _"Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. Or it will lead to bad performance on x64"_ (source : `onnxruntime/quantization/quantize.py:502`) |

`QOperator` est ~320 Ko plus léger, mais `onnxruntime` déconseille explicitement cette combinaison (`QInt8`/`QInt8` + `QOperator`) pour la performance sur CPU x64 — notre plateforme de dev/démo. Les deux formats respectent très largement la contrainte de 5 Mo (marge 40-60%), donc le gain de taille de `QOperator` ne débloque rien d'utile, alors que sa perte de vitesse documentée serait un vrai coût pour B4 et la démo live. **`QDQ` confirmé comme choix final**, pas seulement pour la portabilité (raisonnement initial) mais aussi validé empiriquement contre la recommandation propre d'`onnxruntime`.

## Décomposition détaillée du modèle QDQ final (3,12 Mo)

Répartition par type de tenseur (`detection/models/nanodet-plus-m-1.5x_416-person-car-int8.onnx`) :

| Type                   | Nb tenseurs | Taille       | %     |
| ---------------------- | ----------- | ------------ | ----- |
| int8                   | 444         | 2,421 Mo     | 89,9% |
| float32                | 445         | 0,136 Mo     | 5,0%  |
| int32                  | 220         | 0,135 Mo     | 5,0%  |
| int64                  | 14          | ~0,000 Mo    | 0,0%  |
| **Total initializers** |             | **2,692 Mo** |       |

**Qui est responsable de quoi**, en creusant les noms exacts des tenseurs :

- **444 tenseurs `*_scale` (float32)** — l'échelle de conversion int8→flottant, une par tenseur quantifié (scalaire pour une activation quantifiée par tenseur, tableau d'une valeur par canal pour un poids quantifié `per_channel`). Reste obligatoirement en float32 : c'est un ratio de conversion précis, le quantifier introduirait une erreur systématique dans toutes les déquantifications en aval. Peu nombreux (444) face aux millions de valeurs réelles quantifiées, donc quasi gratuit à garder en pleine précision (0,136 Mo, ~4% du total).
- **444 tenseurs `*_zero_point`** — **pas en float32**, mais en int8 (334, pour les activations et poids int8) ou int32 (110, pour les biais). Vérifié sur un exemple concret (`onnx::Conv_1533_zero_point`, forme `(24,)`) : valeurs toutes à 0 → confirme que les poids utilisent une quantification **symétrique**.
- **110 tenseurs de poids réels (int8)** + **110 tenseurs de biais réels (int32, jamais compressés — précision nécessaire pour `échelle_entrée × échelle_poids`)** — le contenu appris du modèle à proprement parler.
- **14 tenseurs int64** — constantes de forme/indices pour `Reshape`/`Split`/`Slice`, sans rapport avec la quantification.
- **1 tenseur float32 "parasite"** (`/fpn/upsample/Constant_output_0`) — pas une échelle de quantification, une constante du `Resize` (facteur d'agrandissement de l'upsampling Ghost-PAN), qui se trouve juste être aussi en float32.

Recoupement qui valide les comptes : `334 zero_point(int8) + 110 poids(int8) = 444` ✓, `110 zero_point(int32) + 110 biais(int32) = 220` ✓.

**Écart total initializers (2,692 Mo) vs fichier final (3,122 Mo)** : ~0,43 Mo de structure de graphe (nœuds `QuantizeLinear`/`DequantizeLinear`, noms, métadonnées protobuf) — le "coût" du format QDQ par rapport à `QOperator`.

## Résultat final B3.6 — précision INT8 vs fp32 sur `test.json`

`detection/src/evaluate_int8.py` — décodage réutilise `model.head.post_process()`
(dépend seulement de la config, pas des poids) appliqué à la sortie brute
d'`onnxruntime`, pour éviter de réimplémenter le décodage DFL/NMS à la main.

| Métrique  | fp32 (B2.6) | INT8 (B3.6) | Δ absolu | Δ relatif |
| --------- | ----------- | ----------- | -------- | --------- |
| **mAP**   | 0,377       | 0,346       | -0,031   | -8,2%     |
| AP50      | 0,612       | 0,579       | -0,033   | -5,4%     |
| AP75      | 0,379       | 0,349       | -0,030   | -7,9%     |
| AP_small  | 0,175       | 0,149       | -0,026   | -14,9%    |
| AP_medium | 0,542       | 0,496       | -0,046   | -8,5%     |
| AP_large  | 0,687       | 0,643       | -0,044   | -6,4%     |
| AR@100    | 0,489       | 0,463       | -0,026   | -5,3%     |

**Perte de ~3 points de mAP, dans la norme attendue pour du PTQ** (1-5 points
généralement considérés acceptables dans la littérature).

**Vérification spécifique sur `Resize`** (repéré comme à surveiller en B3.1) :
tracé dans le graphe quantifié, les 2 nœuds `Resize` sont encadrés par
`DequantizeLinear → Resize → QuantizeLinear` — calcul en float32, comportement
standard pour un opérateur sans poids à quantifier, identique au schéma
utilisé à des dizaines d'autres endroits du graphe. Rien d'anormal.

**Attention à la lecture des deltas relatifs vs absolus** : en % relatif,
`AP_small` semble le plus touché (-14,9%) — mais en **delta absolu**, c'est
en fait la catégorie qui perd le _moins_ de points (-0,026, contre -0,044
pour large et -0,046 pour medium). Le pourcentage relatif est gonflé
artificiellement par la base de référence plus faible de `AP_small`
(0,175 vs 0,542-0,687), pas par un mécanisme réellement plus dommageable
pour les petits objets. Ni la structure du graphe ni les deltas absolus ne
soutiennent l'hypothèse d'un problème spécifique à `Resize` ou aux petits
objets — la dégradation est uniforme et cohérente avec une perte de
précision PTQ générale, pas localisée.

**Chiffre clé pour la présentation : ~3,1× plus petit (9,79 Mo → 3,12 Mo)
pour ~3,1 points de mAP en moins.**

## Benchmark de performance (B4) — découverte majeure : `s8s8` vs `u8s8`

`detection/src/benchmark.py` — harnais maison (warmup 15, 150 mesures,
statistiques complètes, mémoire RSS mesurée _avant_ construction du
modèle/session pour capturer le vrai coût de chargement).

| Candidat                       | Médiane     | Écart-type | P95      | FPS       | ΔMém                                              |
| ------------------------------ | ----------- | ---------- | -------- | --------- | ------------------------------------------------- |
| PyTorch fp32 GPU               | 9,01 ms     | 1,82       | 13,55 ms | 111,0     | 379 Mo (⚠️ inclut l'init CUDA, non représentatif) |
| PyTorch fp32 CPU               | 61,68 ms    | 6,46       | 74,40 ms | 16,2      | 48,0 Mo                                           |
| ONNX fp32 CPU                  | 10,95 ms    | 1,71       | 12,20 ms | 91,3      | 60,9 Mo                                           |
| ONNX INT8 QDQ CPU (**s8s8**)   | 21,18 ms    | 1,22       | 23,81 ms | 47,2      | 32,1 Mo                                           |
| ONNX INT8 QOperator CPU (s8s8) | 90,89 ms    | 3,99       | 98,84 ms | 11,0      | 12,7 Mo                                           |
| ONNX INT8 QDQ CPU (**u8s8**)   | **9,99 ms** | 0,59       | 10,75 ms | **100,1** | **12,1 Mo**                                       |

**Découverte** : notre premier choix de quantification (`activation_type=QInt8`,
donc `s8s8` avec des poids eux aussi signés) rendait l'INT8 **2 fois plus
lent que le fp32** — contre-intuitif, mais explicable : les instructions
SIMD rapides pour l'INT8 sur x86/x64 (VNNI) sont câblées pour la
combinaison **activations non-signées (`u8`) × poids signés (`s8`)**, pas
`s8×s8`. C'est exactement ce que le warning `onnxruntime` de B3.4 signalait
(_"Or it will lead to bad performance on x64"_) — confirmé ici de façon
spectaculaire pour `QOperator` (8× plus lent que le fp32) et plus
modérément pour `QDQ` (2× plus lent).

**Fix testé et validé** : `activation_type=QuantType.QUInt8` (poids
toujours `QInt8`) → `s8s8` (21,18 ms) devient `u8s8` (9,99 ms), soit
**~2,1× plus rapide**, et bat même le fp32 (10,95 ms). Idem côté mémoire :
32,1 Mo → 12,1 Mo (~3× moins).

**Vérification de précision sur `u8s8`** (`evaluate_int8.py` pointé vers
le nouveau fichier) : **mAP = 0,348**, quasi identique au `s8s8` (0,346,
+0,002, dans le bruit) — donc **u8s8 gagne sur toute la ligne** : vitesse,
mémoire, et précision équivalente.

**⚠️ Choix final mis à jour** : le modèle réellement recommandé pour le
déploiement est désormais **QDQ + `activation_type=QUInt8` /
`weight_type=QInt8`** (`nanodet-plus-m-1.5x_416-person-car-int8-QDQ-u8s8.onnx`,
3,12 Mo), et non plus la version `s8s8` initialement retenue en B3.4. Bon
exemple pour la présentation : une hypothèse raisonnable au départ (`QInt8`
partout, plus simple à écrire) s'est révélée sous-optimale une fois
mesurée en conditions réelles — la mesure a changé la décision.

**Note sur la mémoire vs la contrainte des 5 Mo** : la contrainte projet
portait sur la taille du **fichier** (stockage), pas sur la RAM au pic
d'inférence — aucun budget RAM n'a été communiqué (Q1/Q3). Le facteur
observé entre poids du modèle et pic mémoire (×4 à ×6 selon les
candidats) est normal pour un CNN (activations intermédiaires jamais
stockées dans le fichier, overhead du runtime) — pas un signe de
problème. À noter honnêtement : sur une vraie cible MCU, la RAM serait un
budget séparé et potentiellement plus contraignant que la flash (les plus
petits MCU ont souvent <1 Mo de RAM), un point à soulever si la question
est posée en entretien.

## Bug découvert lors d'un test webcam préliminaire (avant E1) : double sigmoid

Test rapide (webcam en direct) avant de commencer proprement E1 : des
centaines de boîtes `person`/`car` s'affichaient en pavage sur toute
l'image, scores tous compressés entre 0,50 et 0,68 — sur une image de test
pourtant nette et bien exposée (un visage net en gros plan).

**Diagnostic** : comparaison PyTorch fp32 (poids réels) vs ONNX fp32 vs
ONNX INT8 sur la **même image figée**. PyTorch fp32 donnait un résultat
propre (scores 0,05 à 0,65) ; les deux versions ONNX donnaient le pavage
(scores plancher à ~0,50). La signature (plancher à 0,5, plage compressée)
correspond exactement à un **sigmoid appliqué deux fois** :
`_forward_onnx` (utilisée automatiquement par `torch.onnx.export`, cf.
B3.1) applique déjà `sigmoid()` sur la classification pour produire un
graphe "prêt à l'emploi" — mais `NanoDetPlusHead.get_bboxes` (appelée par
`post_process`, notre méthode de décodage réutilisée depuis B2.6) applique
**aussi** `cls_preds.sigmoid()` inconditionnellement, sans savoir si son
entrée est un logit brut (cas PyTorch normal) ou déjà passée au sigmoid
(cas d'une sortie ONNX). Confirmé en lisant le code source
(`nanodet_plus_head.py:497` et `:552`).

**Fix** : inverser le sigmoid de l'export (`logit(p) = ln(p/(1-p))`) sur la
partie classification de la sortie ONNX, avant de la passer à
`post_process` — pour lui redonner des logits bruts, comme elle l'attend :

```python
def undo_export_sigmoid(raw_output, num_classes):
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)
```

Vérifié visuellement : après correction, PyTorch fp32, ONNX fp32 et ONNX
INT8 donnent tous les trois une détection propre et unique sur la même
image (une boîte `person` à ~64-65%), contre le pavage complet avant fix.

**Impact sur les résultats déjà publiés — aucun.** `evaluate_int8.py`
(B3.6) utilisait aussi `post_process` sur une sortie ONNX, donc était
affecté par ce même bug. Relancé après correction : **mAP = 0,348,
identique au chiffre déjà publié.** Explication : un sigmoid est
strictement croissant, donc le double sigmoid préserve le **classement**
relatif des scores (dont dépend uniquement le calcul du mAP) — seul le
nombre de candidats qui franchissent les seuils absolus (filtrage
pré-NMS, seuil d'affichage) est faussé. Sur le jeu de test COCO (images
nettes, in-distribution), l'effet sur les résultats finaux était
négligeable ; sur une image webcam plus difficile/atypique, l'effet
devenait spectaculairement visible. **Bonne illustration de pourquoi le
mAP seul ne suffit pas à valider un pipeline de bout en bout** — un test
visuel qualitatif (même basique) reste nécessaire avant de considérer le
travail terminé.

Fix propagé dans `detection/src/evaluate_int8.py` (et les scripts de
démo webcam temporaires, hors dépôt).

## D2 — LLM local + function calling

### Runtime retenu : llama.cpp (pas Ollama)

Ollama utilise llama.cpp en interne — pour un usage mono-utilisateur local,
les deux sont proches sous le capot. Argument retenu pour llama.cpp direct :
cohérence avec la philosophie déjà suivie côté vision (ONNX Runtime plutôt
que PyTorch complet) — un moteur d'inférence C++ léger, sans dépendance à
un framework d'entraînement, pour le LLM comme pour la détection.

Précision factuelle vérifiée : le "v" de vLLM signifie "virtual" (mémoire
virtuelle, inspiration de PagedAttention) — aucun rapport avec la vision.

### Comparatif des modèles candidats (VRAM, Q4_K_M)

| Modèle                             | Taille fichier                    | VRAM totale estimée | Marge sur 8 Go                            |
| ---------------------------------- | --------------------------------- | ------------------- | ----------------------------------------- |
| **Qwen2.5-7B-Instruct** _(retenu)_ | ~4,5 Go (confirmé : 4,68 Go réel) | ~5,5-6 Go           | confortable                               |
| Llama-3.1-8B-Instruct              | ~4,7-4,9 Go                       | ~5,5-6,5 Go         | confortable, un peu moins de marge        |
| Qwen3-4B (fine-tune tool-calling)  | ~2,3-2,8 Go (estimé)              | ~3-3,5 Go           | large — à tester si Qwen2.5-7B pose souci |

Qwen2.5-7B-Instruct choisi en premier : présent dans la liste des modèles
avec **handler de tool-calling natif** de llama.cpp (parseur dédié, pas le
fallback générique moins fiable) — critère de sélection prioritaire sur la
taille pure. À garder en tête si on veut tester un autre candidat plus tard.

### Comparatif étendu (6 candidats) — en vue du redesign de l'agent (2026-09)

Le redesign de l'agent (nouveau cahier des charges : 12 tools, alertes de
zone/co-occurrence/rafale, cf. section D1 révisée) a motivé une revue plus
large des candidats LLM que le comparatif à 3 modèles ci-dessus — toujours
avec le même critère décisif que Qwen2.5-7B : présence d'un **handler de
tool-calling natif dans `llama.cpp`** (liste officielle :
`docs/function-calling.md` du dépôt `ggml-org/llama.cpp`), pas seulement
le fallback générique.

| Modèle | Params | Taille Q4_K_M | VRAM totale estimée* | Marge/8 Go | Licence | Remarque |
| ------ | ------ | ------------- | --------------------- | ---------- | ------- | -------- |
| **Qwen2.5-7B-Instruct** *(retenu, déjà en place)* | 7B | 4,68 Go (confirmé) | ~5,5-6 Go | confortable | Apache 2.0 | Référence connue, déjà validée de bout en bout |
| Hermes-3-Llama-3.1-8B (Nous Research) | 8B | 4,92 Go (confirmé) | ~5,7-6,2 Go | confortable | Licence communautaire Llama 3.1 (héritée) | Format de tool-call (Hermes) réputé fiable en JSON structuré |
| Functionary-small-v3.2 (MeetKai) | 8B | ~4,92 Go (quant bartowski) / 4,66 Go en Q4_0 (dépôt officiel, seul quant dispo) | ~5,7-6,2 Go | confortable | Licence communautaire Llama 3.1 (héritée, base Llama-3.1-8B) | Seul modèle conçu exclusivement pour le tool-calling, mais support des appels **parallèles** cassé côté `llama.cpp` (voir note ci-dessous) |
| Command R7B (Cohere) | ~8B | 5,06 Go (confirmé) | ~5,9-6,4 Go | confortable | **CC-BY-NC** (non-commercial) | Écarté — même raisonnement que l'exclusion de YOLOv8n/AGPL en section 2.2 : licence non-commerciale incompatible avec un usage entreprise, quel que soit le mérite technique |
| **Granite-4.1-3B** (IBM) | 3B | ~2,0-2,2 Go (estimé) / 2,10 Go confirmé une fois téléchargé (`granite-4.1-3b-Q4_K_M.gguf`, dépôt officiel `ibm-granite`) | ~2,8-3,3 Go | très large | Apache 2.0 | Beaucoup plus rapide, marge VRAM énorme — capacité de raisonnement sur l'extraction de paramètres fins à valider empiriquement, pas supposer |
| Mistral-Nemo-Instruct-2407 | 12B | 7,48 Go (confirmé) | ~8,3-8,8 Go | **dépasse 8 Go** | Apache 2.0 | Écarté sans même être testé — seul candidat qui dépasse le budget VRAM mesuré, ne laisserait quasi aucune marge pour le KV-cache |

*VRAM totale = poids + KV-cache/overhead runtime, estimée selon le même
ratio que la mesure réelle de Qwen2.5-7B (4,68→~5,5-6 Go) — à confirmer
par mesure directe pour chaque nouveau candidat, pas une valeur garantie.

**Sur l'exclusion de Command R7B** : seul candidat de ce comparatif étendu
écarté sur un critère non technique plutôt que sur un test réel — cohérent
avec la méthode déjà appliquée au choix d'architecture de détection
(section 2.2), où la licence AGPL de YOLOv8n l'avait emporté sur un
meilleur mAP brut.

**4 candidats retenus pour un banc de test réel** (Qwen2.5-7B, Hermes-3-8B,
Functionary-small-v3.2, Granite-4.1-3B) — harnais dédié
(`agent/eval/`, banque de 36 questions couvrant les 12 tools + variantes/
pièges, détail en annexe/README de ce dossier). Deux découvertes
notables en testant, indépendantes du comportement des modèles eux-mêmes :

- **2 modèles sur 3 testés (Hermes-3, Functionary) avaient une conversion
  GGUF qui perdait le template de gestion des tools** — le
  `tokenizer_config.json` d'origine définit bien un template
  `tool_use`/complet avec les balises nécessaires, mais l'outil de
  conversion GGUF n'en a embarqué qu'une version simplifiée sans support
  des tools (`chat_template_caps.supports_tools:false` côté
  `/props` de `llama-server`, vérifiable avant même de lancer un
  benchmark). Corrigé en extrayant le bon template du dépôt non-GGUF
  d'origine et en le forçant via `--chat-template-file`. Un piège
  récurrent, pas un cas isolé — à vérifier systématiquement plutôt que
  supposer qu'un GGUF embarque le bon template.
- **Functionary-small-v3.2 plante de façon déterministe (`llama.cpp`,
  pas notre code) dès qu'une question nécessite 2 appels de tool dans
  un même tour** (`Unexpected empty grammar stack after accepting
  piece: >>>`) — isolé par test direct : un appel séquentiel sur 2
  tours fonctionne, `parallel_tool_calls:false` ne corrige rien (non
  honoré par cette implémentation). Limite dure du moteur de grammaire
  de `llama.cpp` pour ce format, pas un problème de nos schémas.

**Décision finale, après le banc de test complet (4 candidats x 36 cas
x 10 répétitions)** : **Granite-4.1-3B remplace Qwen2.5-7B** comme
modèle déployé. Résultats agrégés (taux de réussite / erreurs serveur /
latence médiane) : Granite 97,2% / 0 / 0,65s — Hermes-3 93,9% / 0 /
1,13s — Qwen2.5-7B 92,8% / 3 / 1,24s — Functionary 90,6% / 9 / 1,03s.
Granite l'emporte sur les trois axes qui comptaient (section 9.1) :
meilleur taux de réussite mesuré (et pas seulement dans la marge de
bruit — le classement a changé entre l'échantillon à 3 répétitions, où
Hermes-3 semblait en tête à 95,4%, et celui à 10, signe que 3
répétitions ne suffisaient pas à trancher), ~2x plus rapide, et surtout
un budget VRAM 2 à 2,5x plus faible (2,10 Go contre 4,66-4,92 Go) —
argument décisif car le serveur LLM tourne en parallèle du reste du
pipeline sur l'unique GPU de la machine pendant la démo live.

Au passage, un bug systémique commun aux 4 modèles a été identifié puis
corrigé : sur une question ciblant une zone qui n'existe pas dans
`config/zones.yaml`, tous hallucinaient une zone réelle plutôt que de
refuser, parce que l'`enum` fermé de `zone_name` ne leur laissait
aucune option légitime pour le signaler. Fix : une valeur d'échappement
`zones.UNKNOWN_ZONE = "zone_inconnue"` ajoutée à l'enum + interceptée
côté `tools.py` comme un résultat normal (pas une erreur). Effet
mesuré, inégal selon le modèle : Qwen 9/10, Hermes-3 et Granite 7/10,
Functionary seulement 1/10 malgré un schéma et un prompt identiques —
le fix ouvre une échappatoire, il ne garantit pas qu'un modèle donné
choisisse de l'emprunter.

Détail complet (critères de sélection, tableau comparatif à 6
candidats, méthodologie du banc de test, analyse des échecs par
modèle, argumentaire de la décision) dans
[docs/rapport.tex](rapport.tex), section 9.

### Schémas de tools : écrits à la main (pas de génération par introspection)

4 tools seulement, `agent/src/tool_schemas.py` — coût de maintenance
manuel jugé plus faible qu'un générateur générique pour ce volume,
cohérent avec le principe déjà suivi de ne pas construire d'abstraction
pour un besoin qui n'existe pas encore.

### Bug trouvé et corrigé en testant D2 : fenêtre de fraîcheur trop stricte

`AgentTools.count_vehicles_now()` appelait `EventStore.active_tracks("car")`
avec son défaut `active_within_seconds=1.0` — pensé pour C1/C2 où
`update()` tourne en continu à chaque frame (donc `last_seen` toujours
très frais). Mais un tool d'agent est appelé après le temps de décision du
LLM (round-trip d'inférence) — **mesuré empiriquement à ~1,6s** pour un
seul appel de tool sur Qwen2.5-7B-Instruct/RTX 4060. Avec le défaut de
1,0s, un véhicule pourtant présent "maintenant" au moment de la détection
n'était plus considéré actif au moment où le tool s'exécutait réellement —
`count_vehicles_now()` répondait 0 au lieu de 1.

Isolé proprement (test à processus unique, délai simulé de 1,6s) :
`active_within_seconds=1.0` → 0 résultat ; `=5.0` → résultat correct.

**Premier fix (pansement)** : élargir `AGENT_FRESHNESS_SECONDS` à 5.0 pour
absorber la latence LLM. Fonctionne, mais reste une valeur arbitraire —
si la latence LLM varie (plusieurs tools enchaînés, modèle plus gros...),
un seuil fixe peut redevenir insuffisant, ou au contraire devenir trop
large et compter comme "présent" un véhicule parti depuis un moment.

**Fix définitif (proposé par l'utilisateur)** : capturer `request_ts =
time.time()` dans `Agent.ask()` **avant** tout appel au LLM, puis
l'injecter explicitement (paramètre `now`, jamais exposé au LLM — absent
de `tool_schemas.py`) au moment d'exécuter le tool. "Maintenant" signifie
alors "au moment où la question a été posée", plus jamais "au moment où
le tool a fini par s'exécuter" — le résultat devient **indépendant de la
latence d'inférence**, quelle qu'elle soit. Revalidé : correct même avec
1,8s de latence mesurée.

Conséquence : `AGENT_FRESHNESS_SECONDS` redevient une vraie question
produit plutôt qu'un fudge-factor de latence — remis à **2.0s** (tolérance
au clignotement occasionnel du tracker, cohérent avec le raisonnement déjà
tenu pour C1/C2), plus le petit tolérance de 5.0 qui ne servait qu'à
masquer le symptôme. Même principe appliqué par cohérence à
`count_vehicles_total` (borne de fin) et `count_vehicles_between`
(date de référence pour parser "HH:MM") — tous acceptent désormais `now`
en paramètre optionnel, injecté par `Agent`, jamais par le LLM.

### Validation de bout en bout

Les 3 requêtes d'exemple du brief testées contre le vrai serveur
`llama-server` + Qwen2.5-7B-Instruct, sur des données synthétiques :
tool correctement choisi et appelé à chaque fois (y compris la
désambiguïsation now/total sans intervention de notre part), paramètres
correctement extraits (horaires, seuil de durée), réponse finale cohérente
en français. `set_person_alert` vérifié écrire réellement dans
`EventStore.get_alerts()`.

## E1/E2 — Pipeline live et interface de requête : le problème du thread

`demo/src/live_demo.py` assemble tous les blocs (C1/C2/D1/D2/D3) dans un
seul script présentable. La seule vraie difficulté technique d'E1/E2
n'était pas l'assemblage en lui-même (chaque brique était déjà validée
séparément) mais **comment interroger l'agent sans interrompre le flux
vidéo** : `cv2.imshow`/`waitKey` doivent tourner en continu, alors qu'une
réponse de l'agent prend ~1,5-2s (latence LLM mesurée en D2) — largement
au-dessus du budget d'une frame.

**Rejeté** : lire la question avec un `input()` directement dans la
boucle vidéo. `input()` est bloquant : la fenêtre vidéo se figerait
pendant toute la saisie de l'utilisateur _et_ pendant l'appel réseau au
LLM — inacceptable pour une démo censée montrer un pipeline "temps réel".

**Retenu** : un thread dédié (`input_worker`) lit la console et appelle
`agent.ask()` de son côté, en parallèle de la boucle vidéo principale.
Communication à sens unique via `queue.Queue` : le thread y dépose
`(question, réponse)` une fois l'appel LLM terminé ; la boucle vidéo
consulte cette file à chaque frame avec `get_nowait()` (non-bloquant —
file vide la plupart du temps, ce qui est le cas normal) et affiche la
réponse dès qu'elle apparaît (bandeau bas d'image, 8s + log console). La
vidéo ne connaît jamais de pause, quelle que soit la latence de l'agent.

Conséquence directe : `EventStore` est maintenant accédé **depuis deux
threads** — la boucle vidéo (C1/C2/D3, à chaque frame) et le thread agent
(D1/D2, via `AgentTools`, à chaque question). `sqlite3.connect()` refuse
par défaut qu'une connexion soit utilisée hors du thread qui l'a créée
(`check_same_thread=True` implicite) — on l'a explicitement désactivé
(`check_same_thread=False`) et ajouté un `threading.Lock()` maison autour
de chaque `self.conn.execute()`/`commit()` dans `event_store.py`, plutôt
que de compter sur le comportement interne (variable selon la
configuration de compilation de SQLite) du module `sqlite3` face à des
accès concurrents. Coût négligeable (les requêtes sont très courtes),
robustesse garantie.

Point d'attention identifié mais non traité (hors scope démo) :
`active_durations()` délègue à `active_tracks()` sans reprendre le
verrou lui-même — nécessaire car `threading.Lock` n'est pas réentrant, et
suffisant puisque tout l'accès SQLite reste dans la méthode déléguée.

## B2.7 — Réentraînement (2026-09-05) : diagnostic et comparatif de config

### Diagnostic ayant motivé ce réentraînement

Le run fp32 de référence (B2.5/B2.6, mAP=0,377) montrait un écart marqué
entre classes : `person` ~44% vs `car` ~30%, et un `AP_small` bas (0,175).
Inspection des pools COCO réels (pas une supposition) :

| Split     | pool `person`-seul | pool `car`-seul | pool `both` |
| --------- | ------------------ | --------------- | ----------- |
| train2017 | 55 596 img         | **3 732 img**   | 8 519 img   |
| val2017   | 2 334 img          | **176 img**     | 359 img     |

`data_filter.py` échantillonnait `N_PER_CLASS=4000` **appliqué au même
titre aux 3 buckets** (person-seul / car-seul / both). Le bucket
`car`-seul était déjà épuisé (3732 < 4000) dans l'ancien train, mais le
bucket `both` — la plus grosse source de voitures — n'était exploité qu'à
4000/8519 (47%). Augmenter `N_PER_CLASS` uniformément (l'approche
initialement envisagée) aurait mécaniquement ajouté surtout du
person-seul (pool quasi illimité) sans corriger l'écart. Côté val2017,
les 3 pools sont déjà plus petits que 4000 : l'ancien `val`/`test`
utilisait donc déjà 100% des images disponibles — l'écart 5,7:1
person:car observé sur `val`/`test` est **inhérent à COCO val2017**, pas
un artefact de notre échantillonnage, et ne peut pas être corrigé par ce
levier (mesure `car` intrinsèquement plus bruitée sur val/test, à
mentionner honnêtement).

### Comparatif de configuration — avant / après

**Échantillonnage (`detection/src/data_filter.py`)**

| Paramètre     | Avant                                     | Après                                 | Pourquoi                                                                                                                                      |
| ------------- | ----------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Stratégie     | `N_PER_CLASS=4000` flat sur les 3 buckets | `BUCKET_TARGETS` explicite par bucket | Un N unique traitait `car`-seul (bottleneck) et `person`-seul (pool quasi illimité) de façon symétrique alors qu'ils n'ont rien de comparable |
| `only_person` | 4000                                      | **5000**                              | Léger volume en plus, mais plafonné bien en dessous du pool réel (55 596) pour ne pas re-diluer `car`                                         |
| `only_car`    | 4000 (déjà plafonné par le pool à 3732)   | **100 000 → tout le pool (3732)**     | Rend explicite qu'on prend tout ce qui existe, plutôt qu'un plafond arbitraire qui se trouve dépasser le pool par coïncidence                 |
| `both`        | 4000 (47% du pool 8519 utilisé)           | **100 000 → tout le pool (8519)**     | Plus grosse source de `car` inexploitée jusqu'ici — le vrai levier                                                                            |

**Volumes résultants**

| Split | Avant (images / anns person / anns car) | Après                        | Δ                                                                                                                                                           |
| ----- | --------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train | 11 732 / 37 266 / 27 582                | **17 251 / 64 628 / 43 867** | +47% images, **+59% instances car** (vs +73% person — ratio person:car passe de 1,35 à 1,47, mais le volume absolu de `car` est ce qui compte pour son mAP) |
| val   | 2 008 / 7 546 / 1 321                   | 2 008 / 7 546 / 1 321        | inchangé (pools val2017 déjà saturés avant le changement, cf. diagnostic)                                                                                   |
| test  | 861 / 3 458 / 611                       | 861 / 3 458 / 611            | inchangé (idem)                                                                                                                                             |

**Entraînement (`config/nanodet-plus-m-1.5x_416-person-car.yml`)**

| Paramètre                            | Avant      | Après          | Pourquoi                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------ | ---------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_size`                         | [416, 416] | **[512, 512]** | `AP_small=0,175` bas ; les voitures COCO sont souvent petites/lointaines dans la scène — plus de résolution aide directement la détection des petits objets. Coût : ~1,51× plus de pixels/image (512²/416²), donc plus de VRAM et de calcul ; oblige à refaire l'export ONNX/quantification/benchmark (B3/B4) une fois ce modèle validé |
| `batchsize_per_gpu`                  | 24         | **16**         | 24 avait été calé empiriquement à 8 Go VRAM pour du 416px ; réduit proportionnellement (24/1,51≈16) pour rester dans le même budget mémoire à 512px. À confirmer par un smoke-test avant la run complète (évite un OOM après plusieurs heures)                                                                                          |
| `optimizer.lr`                       | 0.0001     | **0.00007**    | Règle de mise à l'échelle linéaire batch/lr appliquée au nouveau batch (0.0001 × 16/24)                                                                                                                                                                                                                                                 |
| `lr_schedule.eta_min`                | 0.000005   | **0.0000035**  | Même ratio par rapport au nouveau `lr` que l'ancien (eta_min/lr = 0,05)                                                                                                                                                                                                                                                                 |
| `warmup.steps`                       | 200        | **300**        | Prudence supplémentaire : le réseau doit à la fois reprendre le fine-tuning et adapter ses feature maps à une résolution d'entrée différente de celle du checkpoint pré-entraîné                                                                                                                                                        |
| `total_epochs` / `lr_schedule.T_max` | 40         | **50**         | Le run précédent plafonnait dès l'epoch 24/40, mais le dataset est ~47% plus gros et la résolution change — marge additionnelle pour laisser le temps au réseau de converger sur ces deux nouveautés combinées                                                                                                                          |
| `val_intervals`                      | 4          | 4 (inchangé)   | Toujours ~12-13 points de mesure sur toute la run, suffisant                                                                                                                                                                                                                                                                            |

**Précaution prise avant de lancer** : l'ancien workspace
(`workspace/nanodet-plus-m-1.5x_416-person-car`, contenant le
`model_best` à l'origine des exports fp32/INT8 déjà validés en B3/B4) a
été renommé en `...-416-v1-backup` plutôt qu'écrasé — permet de revenir
en arrière si ce nouveau run à 512px s'avère pire, et garde les exports
déjà quantifiés reproductibles.

**Non résolu ce soir, à traiter demain une fois la run terminée** : si le
mAP à 512px s'avère meilleur, il faudra refaire entièrement B3 (export
ONNX, calibration, quantification) et B4 (benchmark) — un modèle à 512px
sera aussi plus lourd/lent en inférence que celui à 416px, à mesurer
avant de décider lequel déployer réellement (le brief vise du temps réel
embarqué, pas seulement le meilleur mAP).

### Premier point de contrôle (epoch ~4/50, val)

Comparé à l'ancienne référence (416px, epoch 24/40, **convergée**) — donc
une comparaison seulement indicative, ce nouveau checkpoint étant lui
très précoce :

| Métrique | Ancien (convergé) | Nouveau (epoch ~4) | Δ |
|---|---|---|---|
| mAP | 0,372 | 0,386 | +0,014 |
| AP50 | 0,612 | 0,637 | +0,025 |
| AP75 | 0,375 | 0,395 | +0,020 |
| AP_small | 0,184 | 0,214 | **+0,030** |
| AP_medium | 0,515 | 0,526 | +0,011 |
| AP_large | 0,661 | 0,631 | -0,030 |
| person (mAP/AP50) | ~44% / — (non tabulé) | 45,8% / 72,8% | — |
| car (mAP/AP50) | ~30% / — (non tabulé) | 31,4% / 54,6% | — |

Deux observations : (1) ce checkpoint précoce égale déjà le meilleur
checkpoint entièrement convergé de l'ancienne run — encourageant, mais
46 epochs restent à courir, et l'ancienne run avait elle-même régressé
légèrement après son pic (epoch 24→40) ; (2) le couple
AP_small +0,030 / AP_large -0,030 est exactement la signature attendue
d'un passage 416→512px (confirmation empirique de l'hypothèse posée
avant de lancer). En revanche le ratio person:car (45,8/31,4=1,46) est
quasi identique à l'ancien (~44/30=1,47) — **le rééquilibrage des données
n'a pas encore visiblement réduit l'écart à ce stade précoce**, à
réévaluer sur les prochains points de contrôle plutôt que de conclure
maintenant.

### Suivi des points de contrôle suivants

| Epoch | mAP | AP50 | AP_small | AP_large | person (mAP/AP50) | car (mAP/AP50) | ratio p:c |
|---|---|---|---|---|---|---|---|
| ~4 | 0,386 | 0,637 | 0,214 | 0,631 | 45,8 / 72,8 | 31,4 / 54,6 | 1,46 |
| 32 | 0,417 | 0,667 | 0,241 | 0,692 | 47,9 / 74,3 | 35,5 / 59,0 | 1,35 |
| 36 | 0,4175 | 0,666 | 0,241 | 0,694 | 47,8 / 74,3 | 35,7 / 58,9 | **1,34** |

Entre epoch 32 et 36, le mAP global stagne quasiment (+0,0005) — **début
de plateau**, cohérent avec ce qui avait déjà été observé sur l'ancienne
run (plafond atteint vers les 2/3 du budget d'epochs). Le ratio
person:car continue cependant de se resserrer légèrement (1,35→1,34) :
`car` gagne encore un peu (+0,2 mAP) pendant que `person` recule
légèrement (-0,1) — cohérent avec l'effet attendu du rééquilibrage des
données, mais le gain marginal ralentit en même temps que le mAP global.
Reste 14 epochs pour voir si le plateau se confirme ou si le modèle
regagne un peu de terrain sur la fin du cosine schedule (LR encore
décroissant).

### Run terminé (50/50 epochs) et archive du monitoring

Les 50 epochs sont allées à leur terme (confirmé : plus aucun processus
`train.py` actif, `model_last.ckpt` sauvegardé). Le plateau pressenti
s'est confirmé : mAP val a redescendu légèrement après l'epoch 36
(0,4175 → 0,41695 → 0,41619 → 0,41456 aux epochs 40/44/48), donc
**`model_best` reste celui de l'epoch 36** — même dynamique pic-puis-léger-recul
que l'ancienne run (416px), 12 epochs plus tard cette fois.

`workspace/` étant gitignoré (régénéré à chaque run, donc pas un
support fiable pour retracer l'historique plus tard), les courbes de
perte et métriques val de ce run ont été extraites du log texte du run
(`workspace/.../logs-2026-09-05-02-49-01/logs.txt`, propre — contrairement
au `logs.txt` racine qui mélange aussi deux faux départs/smoke-tests
antérieurs) vers des CSV versionnés :

- `detection/src/parse_training_log.py` — parseur regex du format de log
  NanoDet (pas de dépendance tensorboard/protobuf, le texte suffit)
- `detection/results/2026-09-05_512px/train_losses.csv` — 2695 lignes
  (une par itération loggée, `log.interval=20`) : loss_qfl/bbox/dfl +
  versions aux_head, lr, par epoch/itération
- `detection/results/2026-09-05_512px/val_metrics.csv` — 12 lignes (une
  par checkpoint `val_intervals=4`) : mAP/AP50/AP75/AP_small/AP_m/AP_l
  + détail per-classe person/car
- `detection/results/2026-09-05_512px/train_cfg.yml` — copie exacte de la
  config utilisée par ce run (retrouver quels hyperparamètres ont produit
  ces courbes, sans dépendre de l'historique git du `.yml` source qui
  continuera d'évoluer)

But : pouvoir tracer les graphes (courbes de perte, mAP par epoch) pour
la présentation sans dépendre du dossier `workspace/` qui sera écrasé au
prochain entraînement.

### Bug latent découvert (et corrigé) en créant un 3e script de démo

En modifiant en place le `.yml` d'entraînement pour B2.7 (416→512px),
les deux scripts de démo existants (`tracking_demo.py`, `live_demo.py`)
se sont retrouvés silencieusement cassés : ils lisent leur taille
d'entrée via `cfg.data.val.input_size`, càd directement depuis ce même
`.yml` — donc 512px dès ce soir, alors que le modèle ONNX INT8
actuellement déployé (`...-int8-QDQ-u8s8.onnx`) reste figé à 416x416
(résolution de son export). Un mismatch de ce genre ne plante pas
forcément immédiatement (le modèle accepterait une entrée mal
dimensionnée ou la déformerait) mais fausserait silencieusement les
détections — découvert seulement parce qu'un 3e script
(`raw_detection_demo.py`, détection brute sans tracker, seuil 0,5) a été
écrit avec ce même pattern et qu'on a vérifié la cohérence avant de le
lancer.

**Fix** : `INPUT_SIZE` figé en dur dans chacun des 3 scripts de démo
(`(416, 416)`, indépendant du `.yml`), au lieu de le déduire de la config
d'entraînement courante. Découple délibérément le runtime de démo (qui
dépend de la résolution du modèle *déployé*, figée à l'export ONNX) de
la config d'entraînement (qui va continuer à changer à chaque
réentraînement) — évite que ce bug ne se reproduise silencieusement au
prochain changement de résolution d'entraînement.

## B3/B4 rejoués sur le modèle 512px (2026-09-05)

`export_onnx.py`, `validate_onnx.py`, `quantize.py`, `evaluate_int8.py`,
`benchmark.py` : chemins renommés `...416...` → `...512...` (nouveaux
fichiers distincts, l'ancien 416px reste intact pour comparaison),
`INPUT_SHAPE`/dummy shape 416→512. `evaluate.py` n'a nécessité aucune
modification (100% piloté par `cfg`). Variantes `s8s8`/`QOperator` non
régénérées (déjà écartées en B3.4/B4 sur le 416px, cf. plus haut) — les
deux blocs concernés de `benchmark.py` rendus optionnels par cohérence.

**Validation ONNX vs PyTorch** : `max_diff` 1,1-1,7e-05, `allclose=True`
sur les 3 images de test — même ordre de grandeur que le 416px, export
correct.

**Précision, `test.json`** :

| Métrique | 416px fp32 | 512px fp32 | Δ | 416px INT8 u8s8 | 512px INT8 u8s8 | Δ |
|---|---|---|---|---|---|---|
| mAP | 0,377 | **0,421** | +0,044 (+11,7%) | 0,348 | **0,389** | +0,041 (+11,8%) |
| AP50 | 0,612 | 0,668 | +0,056 | — (non tabulé) | 0,624 | — |
| AP75 | 0,379 | 0,438 | +0,059 | — | 0,398 | — |
| AP_small | 0,175 | **0,228** | **+0,053 (+30%)** | — | 0,198 | — |
| AP_medium | 0,542 | 0,583 | +0,041 | — | 0,548 | — |
| AP_large | 0,687 | 0,712 | +0,025 | — | 0,661 | — |
| AR@100 | 0,489 | 0,540 | +0,051 | — | 0,511 | — |
| **Perte due à la quantification (mAP)** | -0,029 (-7,7%) | | | -0,032 (-7,6%) | | quasi identique |

L'hypothèse posée avant de lancer la run (512px pour aider les petits
objets, cf. B2.7) se confirme nettement : **AP_small +30%**, le plus gros
gain relatif de toutes les métriques. La perte due à la quantification
reste stable (~3 points de mAP, ~7,7%) entre les deux résolutions — la
quantification n'interagit pas mal avec le passage à 512px.

**Taille fichier** (confirme empiriquement que la résolution d'entrée
n'affecte pas le nombre de poids, donc pas la taille du fichier) :

| | 416px | 512px |
|---|---|---|
| fp32 | 9,79 Mo | 9,79 Mo (9 789 520 o, quasi identique) |
| INT8 QDQ u8s8 | 3,12 Mo | 3,12 Mo (3 120 752 o, quasi identique) |

**Contrainte ≤5 Mo toujours respectée** pour le modèle 512px (3,12 Mo),
vérifié plutôt que supposé.

**Benchmark latence/mémoire (CPU)** — ici la résolution a un vrai coût,
contrairement à la taille fichier :

| Candidat | 416px médiane | 512px médiane | Δ | 416px FPS | 512px FPS | Δ |
|---|---|---|---|---|---|---|
| PyTorch fp32 GPU | 9,01 ms | 10,09 ms | +12% | 111,0 | 99,1 | -11% |
| PyTorch fp32 CPU | 61,68 ms | 93,96 ms | +52% | 16,2 | 10,6 | -35% |
| ONNX fp32 CPU | 10,95 ms | 20,37 ms | +86% | 91,3 | 49,1 | -46% |
| **ONNX INT8 QDQ u8s8 CPU** | **9,99 ms** | **16,50 ms** | **+65%** | **100,1** | **60,6** | **-39%** |

Mémoire (ΔRSS) du candidat de déploiement : 12,1 Mo (416px) → 20,55 Mo
(512px), +70% — cohérent avec le facteur ~1,51× de pixels en plus
(512²/416²) et un peu de surcoût supplémentaire.

**Arbitrage à trancher (pas fait unilatéralement ici)** : le 512px gagne
nettement en précision (+11,7% mAP, +30% sur les petits objets) mais
perd ~39% de FPS (100,1 → 60,6 sur ce CPU de développement). 60 FPS
reste largement suffisant pour une webcam typique (25-30 FPS), donc le
gain de précision est probablement le bon compromis pour ce projet — mais
c'est un choix produit (précision vs vitesse/consommation), pas une
question purement technique, à trancher explicitement plutôt que
déployer 512px par défaut seulement parce que c'est le plus récent.

## B2.8 (planifié, pas lancé) — recherche du plafond de résolution à ≥30 FPS

Suite à la question du budget mémoire (5 Mo) : **la taille du fichier ne
dépend pas de la résolution** (confirmé ci-dessus, 416/512px quasi
identiques) — ce n'est donc pas le facteur limitant pour pousser la
résolution plus loin. Le vrai levier limitant est la **latence/FPS**
(critère fixé : ≥30 FPS) et, dans une moindre mesure, la RAM d'inférence
(pas de budget officiel, mais à surveiller).

**Méthode** : la latence d'un modèle ne dépend que de sa résolution et de
son architecture, pas des poids entraînés — testé en exportant/quantifiant
le `model_best` 512px actuel à différentes résolutions candidates, SANS
réentraîner, pour trouver le plafond réel avant d'engager des heures de
calcul. Scripts jetables dans le scratchpad (pas versionnés), méthodologie
identique à `benchmark.py` (warmup 15, 150 mesures, RSS baseline avant
construction de session).

| Résolution | Médiane | FPS | P95 | ΔMém |
|---|---|---|---|---|
| 416 | 9,99 ms | 100,1 | 10,75 ms | 12,1 Mo |
| 512 | 16,50 ms | 60,6 | 20,29 ms | 20,55 Mo |
| 640 | 20,18 ms | 49,6 | 29,73 ms | 37,01 Mo |
| **768** | **26,06 ms** | **38,4** | **34,87 ms** | **58,74 Mo** |
| 832 | 33,94 ms | 29,5 | 68,77 ms (bruit) | 62,66 Mo |
| 896 | 36,44 ms | 27,4 | 45,42 ms | 73,31 Mo |

Le plafond réel des 30 FPS se situe entre 768 et 832 (832 déjà sous le
seuil en médiane, P95 très bruité). **768px choisi** : marge confortable
au-dessus de 30 FPS, saut ×2,25 en pixels vs 512px (devrait continuer à
faire progresser l'AP_small comme observé 416→512), net du plafond réel.
Coût annoncé et accepté : temps de calcul total scale avec le nombre de
pixels, donc ~16h estimées (vs ~7h pour la run 512px) — plus d'une nuit.

**Config préparée** (`config/nanodet-plus-m-1.5x_416-person-car.yml`,
même méthode d'ajustement proportionnel qu'en B2.7) :

| Paramètre | 512px (B2.7) | 768px (B2.8) | Ratio appliqué |
|---|---|---|---|
| `input_size` | [512,512] | **[768,768]** | — |
| `batchsize_per_gpu` | 16 | **8** | 16 / (768/512)² ≈ 7,1 → arrondi à 8 |
| `optimizer.lr` | 0,00007 | **0,000035** | ×(8/16) |
| `lr_schedule.eta_min` | 0,0000035 | **0,00000175** | ×(8/16) |
| `warmup.steps` | 300 | **400** | prudence supplémentaire (saut de résolution plus grand) |
| `total_epochs`/`T_max` | 50 | 50 (inchangé) | pas de signal pour en changer |

**Smoke-test validé** avant de préparer cette config : 5,51 Go/8 Go VRAM
à quelques dizaines d'itérations (marge confortable), 2156 itérations/epoch
(cohérent avec 17 251 images / batch 8), pas d'OOM.

**Mécanisme de reprise mis en place** (demandé explicitement) : NanoDet
sauvegarde déjà `model_last.ckpt` (poids + optimiseur + position dans le
cosine schedule + epoch courante) à la fin de **chaque epoch**
(`training_epoch_end`, `nanodet/trainer/task.py`) — pas besoin de
construire un système de checkpoint, juste l'activer. Ligne `# resume: true`
ajoutée en commentaire dans le `.yml` avec instructions : à décommenter
**seulement** pour reprendre après une interruption (relancer ensuite
exactement la même commande), jamais pour un premier lancement (le fichier
`model_last.ckpt` n'existe pas encore, `resume` ferait planter le script
sur un fichier introuvable).

**Granularité honnête** : la reprise se fait par **epoch**, pas par batch
— `model_last.ckpt` n'est écrit qu'à la fin de chaque epoch complet, donc
une interruption en plein milieu d'une epoch perd la progression de cette
epoch (jusqu'à ~29 min estimées à ce rythme : 2156 itérations × ~0,8 s,
mesuré sur le smoke-test). Un système de checkpoint par batch existe dans
l'écosystème PyTorch Lightning mais demanderait un callback personnalisé
plus intrusif — non retenu ici, le compromis epoch-level est jugé
suffisant pour ce projet.

**Ancien workspace 512px sauvegardé** sous
`workspace/nanodet-plus-m-1.5x_416-person-car-512px-v2-backup` avant de
préparer cette config (mêmes précautions qu'en B2.7).

**Non lancé à la demande explicite de l'utilisateur** — commande prête,
à exécuter quand décidé :
```bash
uv run python detection/third_party/nanodet/tools/train.py detection/third_party/nanodet/config/nanodet-plus-m-1.5x_416-person-car.yml 2>&1 | tee logs.txt
```

**Lancé le 2026-09-05 (soirée)** par l'utilisateur lui-même.

### Incident : crash pendant l'epoch 7

Crash système signalé par l'utilisateur pendant l'entraînement, coïncidant
avec un `CUDA error: out of memory` dans `logs.txt` (en pleine epoch 7,
itération 13000/2156 — l'ablation de calibration ci-dessous tournait en
parallèle à ce moment-là, mais **entièrement sur CPU**,
`providers=["CPUExecutionProvider"]` explicite, sans contexte CUDA — donc
pas de cause directe évidente côté VRAM). Cause racine non confirmée avec
certitude (fragmentation VRAM après plusieurs heures de training, ou
incident driver/GPU coïncidant avec le crash système observé) — notée
honnêtement plutôt que supposée.

**Aucune perte au-delà d'une epoch partielle** : `model_last.ckpt`
sauvegardé à la fin de l'epoch 6 (21:17), le mécanisme de reprise
préparé en amont (voir plus haut, `# resume: true`) a servi pour la
première fois en conditions réelles — activé (`resume: true`
décommenté), l'utilisateur peut relancer exactement la même commande
pour reprendre à l'epoch 6.

### Premier point de contrôle (epoch ~4/50, val) — comparé au run 512px

| Métrique | 512px (epoch~4) | 768px (epoch~4) | Δ |
|---|---|---|---|
| mAP | 0,386 | 0,436 | +0,050 |
| AP50 | 0,637 | 0,694 | +0,057 |
| AP75 | 0,395 | 0,462 | +0,067 |
| AP_small | 0,214 | **0,300** | **+0,086** |
| AP_medium | 0,526 | 0,568 | +0,042 |
| AP_large | 0,631 | 0,619 | -0,012 |
| AR@100 | 0,512 | 0,570 | +0,058 |
| AR_small | 0,333 | **0,435** | **+0,102** |
| AR_large | 0,774 | 0,742 | -0,032 |
| person (mAP/AP50) | 45,8 / 72,8 | 49,5 / 77,5 | +3,7 / +4,7 |
| car (mAP/AP50) | 31,4 / 54,6 | **37,7 / 61,4** | **+6,3 / +6,8** |
| ratio person:car (mAP) | 1,46 | **1,31** | resserre nettement |

Le gain sur les petits objets s'accentue par rapport au saut précédent
(AP_small +0,086 ici contre +0,030 pour 416→512 au même point de
contrôle) — la tendance ne s'essouffle pas avec la résolution. Même
signature qu'avant côté `AP_large` (légère baisse, -0,012) : la
résolution redistribue la capacité du modèle vers les petits objets,
un peu au détriment des gros. Plus surprenant : le ratio person:car
(1,31) est déjà meilleur ici, dès l'epoch~4, que le ratio auquel le run
512px avait *terminé* après 36 epochs (1,34) — probable synergie entre
le rééquilibrage des données (inchangé depuis B2.7) et la résolution,
les voitures étant en moyenne plus petites/lointaines que les
personnes dans les scènes COCO.

## Ablation taille du set de calibration PTQ (2026-09-05)

Question posée : maintenant que le train set est ~47% plus gros
(B2.7), vaut-il le coup d'augmenter `N_CALIBRATION` (200, cf. B3.3) ?
Testé empiriquement sur le modèle 512px déjà validé (requantification +
réévaluation `test.json`, 100% CPU — aucun conflit avec l'entraînement
768px en cours sur GPU), fp32/preprocessed réutilisés tels quels (la
calibration ne les affecte pas) :

| N | Temps quantification | mAP | AP50 | AP_small |
|---|---|---|---|---|
| 100 | 24,9 s | **0,3936** | 0,6265 | 0,2020 |
| 200 (retenu, B3.3) | 58,8 s | 0,3890 | 0,6236 | 0,1976 |
| 500 | 137,5 s | 0,3828 | 0,6275 | 0,1953 |
| 1000 | 276,4 s | 0,3798 | 0,6144 | 0,1918 |

**Résultat inattendu, tendance monotone et régulière (pas du bruit)** :
le mAP **se dégrade** quand N augmente, et le temps de quantification
scale linéairement (×11 entre 100 et 1000). Explication la plus
probable : `calibrate_method=MinMax` (retenu en B3.4) fixe l'échelle de
quantification sur les valeurs min/max **observées** des activations —
méthode connue pour sa sensibilité aux outliers. Plus N grandit, plus la
probabilité de tomber sur une image à activation extrême augmente,
étirant artificiellement la plage de quantification au détriment de la
précision sur le cas normal majoritaire.

**Conclusion** : augmenter `N_CALIBRATION` avec la taille du dataset
n'est pas justifié — le facteur limitant n'est pas la représentativité
(200 images suffisaient déjà) mais la sensibilité de la méthode de
calibration elle-même. **200 conservé** (proche de l'optimum observé ;
100 légèrement meilleur mais écart faible, potentiellement du bruit sur
un seul run). Rend **B5.1** (comparaison `Entropy`/`Percentile` vs
`MinMax`, jusqu'ici optionnel) nettement plus motivé : preuve empirique
concrète d'une faiblesse réelle de `MinMax`, pas seulement une réserve
théorique de la littérature.

## Bug latent (3e occurrence) + bug lié : chemins figés à 512px cassés par B2.8

Même classe de bug que celui déjà corrigé dans les scripts de démo (voir
plus haut) : `quantize.py`, `evaluate_int8.py`, `validate_onnx.py`,
`benchmark.py` lisaient `cfg.data.val.input_size` depuis le `.yml`
partagé — passé à 768px dès le lancement de B2.8, alors que les fichiers
ONNX qu'ils manipulent (`...-512-...onnx`) restent figés à 512px depuis
leur export. Déclenché concrètement quand l'utilisateur a testé
`calibrate_method=Percentile` directement dans `quantize.py` :
`INVALID_ARGUMENT: Got invalid dimensions... Got: 768 Expected: 512`.
**Fix identique aux scripts de démo** : `INPUT_SIZE = (512, 512)` en dur
dans chacun des 4 scripts, `cfg.defrost()/freeze()` autour de la
mutation là où la valeur doit être poussée dans un objet `cfg` partagé
(`calibration_ready.py` la lit depuis `cfg`).

**Second bug découvert au passage, plus insidieux** : `validate_onnx.py`
et `benchmark.py` chargeaient le PyTorch depuis le chemin *canonique*
`workspace/nanodet-plus-m-1.5x_416-person-car/model_best/...` — qui
pointe maintenant vers le run 768px (B2.8) **en cours**, pas vers les
poids qui ont produit l'ONNX 512px comparé/benchmarké. Sans le fix de
résolution ci-dessus, l'erreur de shape aurait au moins alerté ; **avec**
seulement le fix de résolution, ces deux scripts auraient tourné sans
erreur mais comparé/mesuré les MAUVAIS poids (768px en cours, souvent
partiels/pas convergés) contre l'ONNX 512px — un résultat silencieusement
faux, pire qu'un crash. Repéré en retraçant manuellement pourquoi
`MODEL_PATH`/`PT_MODEL_PATH` pointaient vers un chemin qui allait
forcément changer de sens à chaque nouveau run.

**Fix** : les deux scripts pointent maintenant explicitement vers
`workspace/nanodet-plus-m-1.5x_416-person-car-512px-v2-backup/model_best/...`
(la sauvegarde faite avant de lancer B2.8), pas le chemin canonique.
Leçon générale : un script qui compare/mesure un artefact déjà exporté
(ONNX figé) ne doit jamais lire un chemin "canonique" qui continue de
bouger avec les runs suivants — toujours pointer vers une sauvegarde
nommée explicitement liée à cet artefact.

## CPU à 100% / RAM sous pression pendant la quantification (2026-09-05)

Signalé par l'utilisateur (gestionnaire des tâches) pendant `quantize.py`
— probable facteur aggravant des freezes système du soir (cf. plus haut),
même si le lien de causalité exact reste incertain. Cause identifiée
dans le code source d'`onnxruntime.quantization.calibrate.py` :
`CalibraterBase.create_inference_session()` construit sa propre
`SessionOptions()` avec seulement `graph_optimization_level` réglé —
**aucune limite de threads posée**, et ce paramètre n'est pas exposé par
l'API publique `quantize_static()` (impossible à corriger en passant un
argument). Par défaut, ONNX Runtime utilise tous les cœurs logiques
disponibles pour une seule session — la boucle de calibration (des
centaines d'inférences d'affilée) sature donc les 12 threads du CPU en
continu pendant toute la durée du calcul.

**Fix** : limitation de l'affinité CPU du *process* au niveau OS
(`psutil.Process().cpu_affinity(...)`, à la moitié des cœurs logiques)
— fonctionne indépendamment de ce que fait la bibliothèque en interne,
puisqu'elle plafonne au niveau du système d'exploitation, pas de
l'application. Appliqué à `quantize.py` (calibration) et
`evaluate_int8.py` (boucle d'évaluation sur 861 images de `test.json`,
même profil de charge soutenue).

**Délibérément PAS appliqué à `benchmark.py`** : ce script mesure la
performance réelle (FPS/latence) — limiter les cœurs changerait la
mesure elle-même, rendant les chiffres incomparables à ceux déjà
enregistrés (60,6 FPS à 512px, etc., mesurés à pleine capacité CPU).
Le compromis protection-matérielle vs fidélité-de-mesure est tranché en
faveur de la fidélité ici ; à relancer avec surveillance manuelle
(température) plutôt qu'un throttling qui fausserait le résultat.
`validate_onnx.py` non modifié non plus (seulement 3 images, charge
négligeable, pas la peine).

## Crash RAM+disque à 100% pendant `quantize.py` (Percentile) — cause probable identifiée

Nouveau crash (`bad allocation`, exception C++ dans `session.run()` sur
un nœud `Conv` de la tête de détection), cette fois avec **RAM et SSD
tous deux à 100% au moment précis du crash** (relevé gestionnaire des
tâches) — un vrai épuisement mémoire, pas de la fragmentation post-reboot
comme l'hypothèse précédente le supposait.

**Mécanisme le plus probable, retrouvé dans le code source** :
`HistogramCalibrater` (base commune `Percentile`/`Entropy`) étend son
histogramme avec la **largeur de bin d'origine** dès qu'une image dépasse
la plage déjà vue :
```python
new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
```
Si une image de calibration produit une activation hors norme sur un des
314 tenseurs calibrés (modèle 512px, compté précisément via analyse
statique du graphe ONNX), le nombre de nouveaux bins créés est
proportionnel à `(nouvelle_plage / largeur_originale)` — potentiellement
des millions d'entrées d'un coup si l'écart est important. RAM qui explose
→ Windows pagine sur le SSD → les deux saturent simultanément, cohérent
avec l'observation.

**Corroboré par une preuve indépendante** : c'est très probablement la
même famille d'outlier qui explique pourquoi `MinMax` se dégradait avec
`N_CALIBRATION` croissant (ablation précédente) — avec `MinMax` l'outlier
fausse juste l'échelle silencieusement, avec `Percentile`/`Entropy` le
même outlier peut faire exploser la mémoire au lieu de juste abîmer un
chiffre. Deux symptômes différents, cause commune.

**Fix appliqué** : `extra_options={"num_bins": 128}` dans `quantize.py`
(défaut `Percentile` = 2048) — réduit le pire cas de 16× (le nombre de
bins ajoutés pour une dérive donnée est proportionnel à `num_bins`), sans
changer la méthode elle-même. Non re-testé empiriquement ce soir (arrêt
volontaire après ce crash, cf. discussion avec l'utilisateur) — le
narratif technique (MinMax sensible aux outliers, confirmé par deux
symptômes indépendants : dégradation mAP + explosion mémoire) est déjà
solide pour la présentation sans avoir besoin de finir la comparaison
empirique au prix d'un 5e crash système le même soir.

**Suite** : `MinMax` classique retesté par l'utilisateur, tourne sans
problème — confirme que le crash est spécifique à `Percentile`/aux
méthodes à histogramme. `num_bins=128` n'a **pas** empêché le crash
(retesté par l'utilisateur) — l'explication par la taille des bins était
incomplète ou l'outlier dépasse largement cette marge ; `Entropy` et
`Distribution` héritant de la même `HistogramCalibrater`, elles ne sont
pas recommandées comme alternatives sans plus de garanties.

**Option retenue à la place** : `MinMax` avec `moving_average=True`
(`extra_options`) — aucun histogramme, même profil mémoire léger que
`MinMax` classique (2 scalaires/tenseur), mais au lieu du min/max brut
jamais vu, une moyenne mobile exponentielle (`averaging_constant=0.01`
par défaut) qui amortit l'influence d'une image à activation extrême
plutôt que de la laisser dicter toute l'échelle d'un coup — répond
directement à la sensibilité aux outliers déjà identifiée, sans le
risque mémoire des méthodes à histogramme. Appliqué dans `quantize.py`,
pas encore testé.

**Testé** : tourne sans problème (confirme l'absence d'histogramme = pas
de risque mémoire). Comparé à `MinMax` classique sur `test.json` :

| Métrique | MinMax classique | MinMax + moving_average | Δ |
|---|---|---|---|
| mAP | 0,389 | 0,389 | = |
| AP50 | 0,624 | 0,624 | = |
| AP75 | 0,398 | 0,398 | = |
| AP_small | 0,198 | 0,198 | = |
| AP_medium | 0,548 | 0,548 | = |
| AP_large | 0,661 | 0,661 | = |
| AR@100 | 0,511 | 0,511 | = |

**Résultat identique, aucune différence mesurable.** Sur ce tirage
précis de 200 images de calibration (seed=42), la moyenne mobile
(`averaging_constant=0,01`) converge vers essentiellement les mêmes
bornes min/max que la version brute — soit qu'il n'y ait pas eu
d'outlier assez dominant dans *ce* tirage pour que l'amortissement
change quelque chose, soit que l'écart résiduel soit trop fin pour
changer les niveaux de quantification INT8 (256 paliers). N'invalide
pas l'intérêt de `moving_average` : le calcul reste structurellement
plus robuste à un futur tirage contenant un vrai outlier dominant (déjà
observé avec `MinMax` classique à N=500/1000) — juste pas de gain
visible sur ce test précis à N=200. **Conservé comme choix par défaut**
pour `quantize.py` : gratuit en coût (même profil mémoire que `MinMax`),
strictement plus robuste en théorie, sans régression mesurée.

## Percentile/Entropy — conclusion finale de l'exploration (B5.1)

Tenté de reproduire la comparaison `Percentile`/`Entropy` sur Google
Colab (`percentile_entropy_colab.ipynb`) pour s'affranchir du risque
matériel local — même pipeline exact (mêmes 200 images de calibration,
même config, mêmes 861 images de test), donc une comparaison valide
malgré l'environnement différent. **La session Colab a aussi planté par
épuisement RAM**, malgré ses ~12-13 Go disponibles (gratuit) contre les
24 Go de la machine locale.

**Conclusion** : le crash n'est pas une particularité de la machine
locale — confirmé sur **deux environnements indépendants**. Preuve
empirique solide que `HistogramCalibrater` (base commune `Percentile`/
`Entropy`) a un vrai défaut d'implémentation sur ce modèle précis
(probable explosion du tableau d'histogramme face à un outlier
d'activation, cf. plus haut, le fix `num_bins=128` n'ayant pas suffi à
la contenir). Exploration arrêtée ici : creuser plus loin (identifier
l'image/le tenseur responsable, patcher `onnxruntime`, payer pour du
Colab Pro à plus de RAM) dépasserait largement le rapport effort/valeur
pour une comparaison optionnelle (B5.1).

**Ce qui reste comme livrable de cette exploration** — un narratif
technique complet et vérifié empiriquement, sans avoir besoin de finir
la comparaison chiffrée :
1. `MinMax` (retenu, B3.4) est mesurablement sensible aux outliers de
   calibration (mAP se dégrade avec `N_CALIBRATION` croissant, ablation
   dédiée)
2. Les méthodes à histogramme censées corriger ce défaut
   (`Percentile`/`Entropy`) ont un défaut d'implémentation encore plus
   sévère sur ce modèle (explosion mémoire, confirmée sur 2
   environnements)
3. `MinMax` + `moving_average=True` est la meilleure réponse pratique
   trouvée : corrige le même problème conceptuel que les méthodes à
   histogramme, sans leur risque — **retenu comme choix final** pour
   `quantize.py`, aucune régression de mAP mesurée (0,389, identique à
   `MinMax` classique sur ce tirage)

## Rebondissement : la vraie cause trouvée, `Percentile` finalement viable

Sur demande explicite de vérifier tous les paramètres de `Percentile`
avant de clore le sujet : la vraie cause du crash n'était pas (seulement)
la croissance du tableau d'histogramme, mais **`HistogramCalibrater.collect_data()`
accumule la totalité des images de calibration en mémoire avant de
calculer le moindre histogramme** (`self.intermediate_outputs.append(...)`
dans une boucle `while True` sur tout le `CalibrationDataReader`, un seul
appel à `self.collector.collect()` à la fin) — contrairement à `MinMax`
qui met à jour ses 2 scalaires par tenseur au fil de l'eau, image par
image. À N=200 images × 314 tenseurs × ~302 Mo/image (calculé plus haut
par analyse statique du graphe), le pic peut atteindre l'ordre de
**~60 Go** — cohérent avec les deux crashs RAM+disque observés (local et
Colab).

**Fix existant dans `onnxruntime`, pas un contournement maison** :
`quantize_static(..., extra_options={"CalibStridedMinMax": N})`
(`quantize.py:848-859` du package) découpe le calibration reader en
tranches de `N` images, appelle `calibrator.collect_data()` plusieurs
fois de suite plutôt qu'une seule fois avec tout — chaque appel traite,
historigramme, puis libère sa tranche avant la suivante. Le nom mentionne
"MinMax" mais le mécanisme est générique (boucle dans `quantize_static`,
pas dans le calibrateur lui-même) : fonctionne aussi avec
`HistogramCalibrater` (`Percentile`/`Entropy`).

**Implémentation** : `NanoDetCalibrationDataReader` (`calibration_ready.py`)
ne supportait pas `__len__`/`set_range` (requis par ce mécanisme, stubs
`NotImplementedError` dans la classe de base) — ajoutés.

**Testé avec `stride=10`** (20 lots de 10 images, pic mémoire estimé
~3 Go au lieu de ~60 Go) : **aucun crash**, quantification propre.

### Comparatif final, mAP sur `test.json`

| Méthode | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR@100 |
|---|---|---|---|---|---|---|---|
| MinMax classique | 0,389 | 0,624 | 0,398 | 0,198 | 0,548 | 0,661 | 0,511 |
| MinMax + moving_average | 0,389 | 0,624 | 0,398 | 0,198 | 0,548 | 0,661 | 0,511 |
| **Percentile + CalibStridedMinMax(10)** | **0,400** | **0,638** | **0,405** | **0,211** | **0,568** | **0,688** | **0,524** |

**`Percentile` gagne réellement, sur toutes les métriques**, une fois le
bug de mémoire contourné — +0,011 de mAP (+2,8%) vs `MinMax`. Ça valide
l'intuition théorique de départ (les méthodes à histogramme sont plus
robustes que le min/max brut) : le problème n'était jamais la méthode de
calibration elle-même, seulement un défaut d'implémentation
d'`onnxruntime` dans la consommation du `CalibrationDataReader` — bon
exemple pour la présentation de la différence entre "l'idée est mauvaise"
et "l'implémentation par défaut a un bug", découverte en creusant au lieu
de s'arrêter au premier échec.

**Décision finale (proposée, à confirmer)** : basculer `quantize.py` sur
`Percentile` + `CalibStridedMinMax=10` comme réglage par défaut,
remplaçant `MinMax`+`moving_average` — mAP réellement meilleur, même
coût mémoire maîtrisé, aucune régression identifiée.

### Confirmé : `Entropy` testé aussi, `Percentile` reste le meilleur

Même fix (`CalibStridedMinMax=10`) appliqué à `Entropy` (l'autre méthode
à histogramme) pour un comparatif complet des 3 candidats avant de
figer un choix :

| Méthode | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR@100 |
|---|---|---|---|---|---|---|---|
| MinMax classique | 0,389 | 0,624 | 0,398 | 0,198 | 0,548 | 0,661 | 0,511 |
| MinMax + moving_average | 0,389 | 0,624 | 0,398 | 0,198 | 0,548 | 0,661 | 0,511 |
| Entropy + CalibStridedMinMax(10) | 0,392 | 0,630 | 0,399 | 0,202 | 0,547 | 0,672 | 0,512 |
| **Percentile + CalibStridedMinMax(10)** | **0,400** | **0,638** | **0,405** | **0,211** | **0,568** | **0,688** | **0,524** |

`Entropy` bat légèrement `MinMax` (+0,003) mais reste net derrière
`Percentile` (+0,011 par rapport à `MinMax`, +0,008 par rapport à
`Entropy`) sur ce modèle. Pas d'exploration plus poussée des
hyperparamètres d'`Entropy` (`num_bins`/`num_quantized_bins`) — l'écart
avec `Percentile` est déjà net et cohérent sur toutes les métriques,
retour sur investissement jugé trop faible pour continuer à ajuster.

### `quantize.py` mis à jour et revalidé — décision actée

`calibrate_method=CalibrationMethod.Percentile`,
`extra_options={"CalibStridedMinMax": 10}`. Modèle officiel
(`nanodet-plus-m-1.5x_512-person-car-int8-QDQ-u8s8.onnx`) régénéré avec
ce réglage et revérifié : **mAP=0,400** (identique au test préliminaire,
reproductible), **3,12 Mo** (taille inchangée, confirme une fois de plus
que la méthode de calibration n'affecte pas la taille du fichier —
seule la répartition fine des poids quantifiés change). Fichiers de test
jetables (`_test_percentile_strided.onnx`, `_test_entropy_strided.onnx`)
supprimés après vérification.

**mAP final du pipeline 512px, INT8, `Percentile`+`CalibStridedMinMax`** :
0,400 sur `test.json` — à comparer au fp32 512px (0,421, perte
quantification -0,021/-5,0%, la plus faible perte mesurée sur tout le
projet) et au premier pipeline INT8 416px (0,348, B3.6) : **+15% de mAP
relatif** entre le tout premier modèle INT8 validé et la version finale
de ce soir, tous facteurs combinés (rééquilibrage données B2.7,
résolution 512px, et calibration `Percentile` corrigée).

## Scripts de démo basculés sur le modèle 512px

Les 5 scripts de démo (`demo/src/*.py`) pointaient encore vers
`nanodet-plus-m-1.5x_416-person-car-int8-QDQ-u8s8.onnx` (mAP=0,348) —
décision restée en suspens depuis le comparatif 416/512px. Basculés vers
`nanodet-plus-m-1.5x_512-person-car-int8-QDQ-u8s8.onnx` (mAP=0,400,
`INPUT_SIZE=(512,512)` mis à jour en conséquence dans chacun) : gain de
précision important (+15%), FPS toujours largement suffisant pour une
démo webcam (60,6 mesuré vs 25-30 typique), aucune raison de garder
l'ancien modèle comme démo par défaut. L'ancien fichier 416px conservé
sur disque (référence historique, comparaisons déjà documentées) mais
plus utilisé par aucun script actif.

## Lecture d'un fichier vidéo en plus de la webcam (2026-09-06)

Constat de l'utilisateur : une démo purement webcam ne peut montrer que
`person` (l'utilisateur devant sa caméra), jamais `car` — ne représente
pas l'étendue réelle du système à 2 classes.

**Vidéo récupérée** : `demo/assets/street_traffic.mp4` (gitignoré comme
tout `demo/assets/`), scène de rue avec voitures et piétons, source
[Pixabay](https://pixabay.com/videos/road-traffic-cars-pedestrians-87444/)
(licence Pixabay Content License — libre d'usage, aucune attribution
requise), 4K, 24 FPS, 186 frames (~7,75 s).

**Code** : les 5 scripts de démo acceptent maintenant `VIDEO_SOURCE`
(webcam via `CAMERA_ID` ou chemin de fichier) au lieu de `CAMERA_ID` en
dur — `cv2.VideoCapture` accepte les deux indifféremment. **Bouclage
automatique** ajouté sur fin de fichier (`cap.set(cv2.CAP_PROP_POS_FRAMES, 0)`
puis `continue`) plutôt que d'arrêter le script — un clip de 7,75 s est
trop court pour laisser le temps à l'alerte D3 (`>10s`) de se déclencher
sans boucler, et un bouclage rend la démo robuste à la durée du clip en
général, pas seulement celui-ci. La détection de fin de flux réelle
(webcam débranchée) reste un arrêt franc — le bouclage ne s'active que
si `VIDEO_SOURCE` est un chemin de fichier (`isinstance(..., str)`), pas
un index de caméra.

**Réglé par défaut sur la vidéo de rue** sur les 5 scripts (cohérence :
toute la progression de démo — brut → tracker → +journal → +alerte →
+agent — montre les deux classes plutôt que de mélanger webcam pour les
premiers et vidéo pour le dernier). Repasser sur la webcam : éditer
`VIDEO_SOURCE = CAMERA_ID` dans le script voulu.

**Bug trouvé juste après** : la vidéo étant en 4K (3840×2160), `cv2.imshow`
ouvrait la fenêtre à la taille réelle de l'image — largement plus grande
que l'écran, d'où l'impression de ne voir que le coin supérieur gauche
(le reste de la fenêtre part hors écran). Fix : `resize_for_display()`
ajouté dans les 5 scripts, réduit l'image à 1280px de large max
**juste avant `cv2.imshow`** (après le dessin des boîtes/bandeaux, donc
aucun recalcul de coordonnées nécessaire — un simple resize global
préserve les positions relatives). Le modèle continue de recevoir les
frames en pleine résolution (`INPUT_SIZE` inchangé) — seul l'affichage
est concerné.

## Vidéo de démo remplacée (2026-09-06)

La première vidéo (`road-traffic-cars-pedestrians-87444`) était en
accéléré avec du flou de mouvement (constaté par l'utilisateur) et trop
courte (7,75 s) pour laisser le temps à l'alerte D3 (>10s) de se
déclencher naturellement. Recherche d'une alternative, avec vérification
empirique plutôt que de se fier à la description de la page :

- Deux candidats initiaux (`street-pedestrians-cars-road-busy-191` 2015,
  `intersection-cars-pedestrian-night-180386` 2023) ont d'abord semblé
  ne pas s'ouvrir avec `cv2.VideoCapture` — **fausse piste** : le
  problème venait d'un chemin `/tmp/...` (chemin Git Bash) passé à un
  process Python natif Windows, pas d'un problème de codec. Une fois
  testés depuis un chemin du projet, les deux s'ouvrent normalement.
- Vitesse de lecture vérifiée par différence inter-frames moyenne (une
  vidéo accélérée montrerait un écart bien plus élevé qu'une vidéo
  normale) : ~4-7 sur 255, cohérent avec une vitesse réelle.
- **Choix final** : [Belgrade, Europe, Street](https://pixabay.com/videos/belgrade-europe-street-serbia-city-47560/)
  (Pixabay, licence libre d'usage, 2020) — 1920x1080, 24 FPS, **16,3 s**
  (>10s, l'alerte peut se déclencher sans boucler prématurément), vitesse
  normale confirmée, luminosité de jour (88,6/255 en moyenne).
- **Départagé empiriquement** contre un candidat de nuit avec le vrai
  modèle plutôt qu'en devinant depuis les vignettes : sur 15 frames
  échantillonnées, Belgrade donne 143 détections `person` + 36 `car`
  (bon équilibre des deux classes) contre seulement 1 `person` + 42
  `car` pour la vidéo de nuit (piétons quasi invisibles dans le noir) —
  écart décisif, confirme l'intérêt de tester avec le modèle réel plutôt
  que de choisir "à l'œil".

## Camouflage des pertes/instabilité du détecteur via le tracker (2026-09-06)

Question posée : peut-on masquer les trous de détection ou le tremblement
des boîtes à l'aide du tracker plutôt que d'améliorer le modèle
lui-même ? Deux techniques implémentées dans `agent/src/tracker.py`,
une troisième documentée sans implémentation.

**1. Extrapolation pendant les pertes ("coasting")** — `ByteTrackTracker`
(package `trackers`) utilise déjà un filtre de Kalman en interne
(`ByteTrackTracklet.predict()`/`get_state_bbox()`) pour retrouver le même
`tracker_id` après une brève occlusion, mais l'API publique `.update()`
ne renvoyait que les boîtes réellement associées à une détection ce
frame-ci (vérifié dans le code source : `result = detections[idx]`,
indexation sur les détections d'entrée). `MultiClassByteTracker.update()`
accède maintenant directement à l'état interne (`self.trackers[cls_idx].tracks`)
pour exposer aussi les pistes confirmées non ré-associées ce frame-ci
mais toujours dans la fenêtre de tolérance (`time_since_update > 0`),
avec leur position extrapolée (`get_state_bbox()`).

**2. Lissage des coordonnées (moyenne mobile exponentielle)** — appliqué
par `(classe, tracker_id)` dans `MultiClassByteTracker._smooth()`
(`SMOOTHING_ALPHA=0.4`), sur les boîtes confirmées ET coasted (même
mécanisme pour les deux). Purge automatique de l'état de lissage pour les
pistes définitivement abandonnées (au-delà de `lost_track_buffer`) —
évite une fuite mémoire sur une démo longue durée.

**Format de sortie étendu** : `[x1,y1,x2,y2,score,tracker_id,is_coasted]`
(7e élément ajouté, pas de rupture — `event_store.py` lit `box[5]` par
index, toujours correct). Les 4 scripts de démo utilisant `draw_tracked`
mis à jour : boîte pointillée (`draw_dashed_rect`, pas de primitive
native OpenCV) + label `"(prédit)"` pour les boîtes coasted, plutôt que
de les afficher comme des détections normales — **distinction visuelle
volontaire, pas de tricherie silencieuse**.

**Barrière imperméable posée entre affichage et journal** : une boîte
`is_coasted=True` ne doit jamais alimenter `EventStore` (C2) — une
extrapolation ferait sinon avancer `last_seen` sans observation réelle,
faussant les durées de présence utilisées par l'alerte D3. Nouvelle
fonction `confirmed_only(tracked)` (filtre `is_coasted`) insérée juste
avant chaque appel à `store.update(...)` dans `tracker_journal_demo.py`,
`tracking_demo.py`, `live_demo.py`.

**Validé** (test manuel, sans interface graphique, 120 frames de
`street_traffic.mp4`) : 4251 boîtes confirmées, 442 coasted, aucune
erreur, les deux classes représentées dans les deux catégories.

**3. Hystérésis de seuil de confiance (expliquée, non implémentée)** —
principe : utiliser un seuil plus élevé pour *créer* une piste qu'un
seuil plus bas pour la *maintenir* une fois confirmée, pour éviter
qu'une piste dont le score oscille autour du seuil ne clignote
(apparaît/disparaît d'un frame à l'autre). **Déjà partiellement en place**
côté association ByteTrack (`high_conf_det_threshold=0.35` pour le
premier stage d'association, un stage à seuil plus bas existe en
interne pour récupérer les détections faibles) — non étendu côté
affichage (`SCORE_THRESHOLD` dans les scripts de démo reste un seuil
unique) faute de bénéfice net mesuré : le coasting (technique 1) couvre
déjà le cas "piste brièvement sous le seuil", l'hystérésis apporterait
un gain marginal supplémentaire pour un risque de complexité/confusion
plus élevé (deux seuils à régler au lieu d'un).

**Finalement implémentée** (demande explicite de l'utilisateur, revient
sur la décision ci-dessus). Placée dans `agent/src/tracker.py`, pas dans
un script de démo — c'est un état par piste (`tracker_id`) qui doit
persister d'un frame à l'autre, même logique que #1/#2.

`MultiClassByteTracker._passes_hysteresis()` — trigger de Schmitt :
`HYSTERESIS_HIGH=0.35` pour qu'une piste **devienne** visible,
`HYSTERESIS_LOW=0.20` seulement pour qu'elle **redevienne** invisible
(pas le même seuil dans les deux sens). État suivi dans `self._visible_ids`
(`(classe, tracker_id)`), purgé comme `_smoothed_boxes` quand une piste
est définitivement abandonnée. Le coasting (#1) ne concerne désormais
que les pistes ayant réellement été visibles avant de se perdre
(`(cls_idx, tracker_id) in self._visible_ids`) — pas une piste jamais
confirmée à l'affichage.

**Conséquence** : le filtrage par score dans les 4 scripts de démo
utilisant le tracker (`if not is_coasted and score < SCORE_THRESHOLD`)
devient redondant — tout ce qui sort de `tracker.update()` est déjà
destiné à être affiché. Filtre retiré, constante `SCORE_THRESHOLD`
supprimée (morte) dans ces 4 fichiers. `raw_detection_demo.py`
(pas de tracker) conserve son propre seuil, logique différente.

**Validé** (test manuel sans interface graphique, 120 frames) : 1985
boîtes confirmées post-hystérésis, 62 coastées, 19 pistes visibles en
fin de run, aucune incohérence détectée (assertion de contrôle :
`score >= HYSTERESIS_LOW` ou piste bien dans `_visible_ids`).

### Bug signalé : aucun pointillé visible dans `tracker_only_demo.py`

L'utilisateur ne voyait jamais de boîte "coastée" (pointillés) à
l'écran, alors que le test sans interface graphique en trouvait
plusieurs dizaines. Diagnostic par instrumentation plutôt que supposition :

`lost_track_buffer` (paramètre de `ByteTrackTracker`) est documenté comme
"nombre de frames à 30 FPS" mais utilisé en interne comme une tolérance
en **secondes d'horloge réelle**
(`maximum_time_without_update = lost_track_buffer / 30.0`), pas un
compte de frames. `lost_track_buffer=30` (valeur d'origine) ⇒ tolérance
réelle = 1,0 s, peu importe le FPS effectif du pipeline.

Test comparatif (mêmes 120 frames, `time.sleep(0.02)` + `cv2.waitKey(1)`
par itération pour simuler le coût d'un vrai rendu GUI, ~10,7 FPS
effectif mesuré) :

| `lost_track_buffer` | Sans délai simulé | Avec délai simulé (~10,7 FPS) |
|---|---|---|
| 30 (avant) | 62 coastées | **20 coastées** |
| 90 (retenu) | — | **89 coastées** |

À 10-15 FPS réels (webcam/vidéo + rendu GUI, plus lent que les 30 FPS
supposés par la librairie), 1 seconde de tolérance ne représente que
10-15 frames de coasting — une fenêtre trop courte pour être visible à
l'œil, cohérent avec le signalement. **`lost_track_buffer` porté à 90**
(~3 s réelles) dans `DEFAULT_TRACKER_KWARGS` — fenêtre de coasting
redevenue nettement visible (89 sur le même test), sans faire
"fantômer" une piste disparue pendant un temps déraisonnable pour une
démo.

## Configuration partagée des scripts de démo (2026-09-06)

Chaque script de démo dupliquait ses propres constantes
(`CONFIG_PATH`, `ONNX_PATH`, `INPUT_SIZE`, `VIDEO_SOURCE`, etc.) —
changer de source vidéo ou de modèle demandait d'éditer les 5 fichiers
un par un. Centralisé dans **`demo/src/config.py`**, importé par les 5
scripts (aucun `sys.path.insert` nécessaire : Python ajoute déjà le
dossier du script exécuté à `sys.path`, `config.py` est un fichier
frère dans `demo/src/`).

**Point de conception notable** : `ONNX_PATH` et `INPUT_SIZE` sont
regroupés dans un dict `MODELS` indexé par nom (`"512px"`/`"416px"`)
plutôt que deux constantes indépendantes — directement motivé par le
bug de mismatch de résolution rencontré plusieurs fois ce soir (un
fichier ONNX figé à une résolution, une config qui en indique une
autre). Changer `ACTIVE_MODEL` change les deux ensemble, structurellement
impossible de les désynchroniser.

`SCORE_THRESHOLD` renommé `RAW_SCORE_THRESHOLD` dans ce fichier central
(clarifie qu'il ne concerne que `raw_detection_demo.py` — les 4 autres
scripts n'en ont plus besoin depuis l'hystérésis côté tracker).

Testé : les 5 scripts démarrent sans erreur d'import après le
refactor (vérifié par exécution réelle avec `timeout`, pas seulement
compilation syntaxique).

## B3/B4 rejoués sur le modèle 768px (B2.8 terminé, 2026-09-07)

Entraînement 768px terminé (50/50 epochs, `model_best`=epoch 40,
mAP val=0,469 — même schéma pic-puis-léger-recul que les runs
précédents, confirmé sur les epochs 44/48 dans le log). Séquence B3/B4
rejouée à l'identique (export ONNX 768px, validation, quantification
`Percentile`+`CalibStridedMinMax=10`, évaluation INT8, benchmark) — mêmes
scripts que pour 512px, chemins/`INPUT_SHAPE` mis à jour à 768.
`evaluate.py` de nouveau sans aucune modification (100% piloté par
`cfg`, qui reflète déjà 768px). `MODEL_PATH`/`PT_MODEL_PATH` dans
`validate_onnx.py`/`benchmark.py` repointés sur le chemin canonique
(B2.8 terminé, plus de run en cours qui le ferait bouger — la
redirection temporaire vers la sauvegarde 512px n'était plus utile).

### Comparatif final, les 3 résolutions

| | 416px | 512px | **768px** |
|---|---|---|---|
| mAP fp32 (test.json) | 0,377 | 0,421 | **0,473** |
| mAP INT8 (test.json) | 0,348 | 0,400 (Percentile+stride) | **0,458** |
| Perte due à la quantification | -7,7% | -4,8% | **-3,2%** (la plus faible du projet) |
| Taille fichier INT8 | 3,12 Mo | 3,12 Mo | 3,12 Mo (confirme une 3e fois que la résolution n'affecte pas la taille) |
| Latence médiane (CPU, INT8) | 9,99 ms | 16,50 ms | **33,65 ms** |
| FPS (CPU, INT8) | 100,1 | 60,6 | **29,7** |

**Point à ne pas passer sous silence** : le 768px repasse **sous le seuil
de 30 FPS** fixé lors de la décision B2.8 (29,7 mesuré ici, contre 38,4
estimé par le test rapide sur poids non entraînés fait avant de lancer
l'entraînement). La latence ne dépend que de la résolution/architecture,
pas des poids — l'écart vient donc d'une différence de conditions de
mesure entre les deux sessions (charge système, etc.), pas d'une erreur
de calcul ; l'estimation préalable au lancement de B2.8 était un peu
optimiste.

**Progression continue** : le 768px confirme la tendance déjà observée
416→512 — la résolution supplémentaire continue d'apporter un vrai gain
de précision (+12,4% de mAP fp32 vs 512px) et réduit encore la perte de
quantification (-3,2%, la meilleure du projet — cohérent avec
l'hypothèse que plus de résolution donne des activations plus stables,
moins sensibles à la quantification). Mais le compromis vitesse devient
plus serré : à 768px, on est maintenant tout juste à la limite (voire
légèrement en dessous) du critère de FPS auto-imposé.

**Décision à trancher (pas prise ici)** : 768px offre la meilleure
précision du projet mais franchit de justesse le seuil de FPS visé.
Choix produit (précision vs respect strict du critère de vitesse), pas
une question purement technique — à décider explicitement avant de
figer le modèle final pour F1/F2 (présentation).

## Anomalie observée sur le run 416px-gen1 (dataset actuel) : val loss > train loss

Le run `2026-09-08_416px-gen1` (416px réentraîné sur le dataset actuel,
17 251 images — cf. l'archive `2026-09-04_416px-archive-ancien-dataset`
pour l'ancien run sur l'ancien dataset, non comparable) montre une val
loss **au-dessus** de la train loss en fin d'entraînement — inverse de
ce qu'on observe sur 512px et 768px.

**Mesure** (somme `loss_qfl+loss_bbox+loss_dfl`, moyenne sur les derniers
points loggés de train vs les points val de l'epoch 48) :

| | train (fin) | val (epoch 48) | écart |
|---|---|---|---|
| 416px (gen1) | ~0,720 | ~0,784 | **val +9 %** |
| 512px | ~0,666 | ~0,708 | val +6 % (déjà limite) |
| 768px | ~0,773 | ~0,744 | val **-4 %** (sens attendu) |

Pas un décalage constant : à l'epoch 4, train et val sont dans la même
fourchette sur les trois runs. L'écart se creuse progressivement pendant
l'entraînement — un vrai croisement tardif (signature d'overfitting),
mais dont l'amplitude dépend fortement de la résolution.

**Hypothèse initiale (invalidée, 2026-09-14)** : le pipeline
d'augmentation (`scale: [0.6,1.4]`, `stretch`, `translate: 0.2`) est
identique en valeurs absolues aux trois résolutions, mais son effet
relatif ne l'est pas ; beaucoup d'objets (`car` en particulier) ont peu
de pixels à 416px, un `scale` à 0.6 les ferait passer sous le plancher
de détectabilité pendant l'entraînement, gonflant artificiellement la
train loss. **Erreur de sens repérée après coup** : une augmentation
qui rend l'entraînement plus dur ferait monter la *train* loss, pas la
val loss — cette piste prédit train > val, l'inverse de ce qui est
observé ici (val > train). Écartée. `AP_small` progresse bien avec la
résolution (0,193→0,240→0,322), mais ce n'est pas une preuve du
mécanisme ci-dessus, juste une observation compatible avec beaucoup
d'autres explications.

**Hypothèse retenue (corrigée)** : la validation et l'entraînement
n'évaluent pas les mêmes poids. `nanodet/trainer/task.py` :
`training_step` utilise `self.model` (poids bruts) ; `validation_step`
utilise `self.avg_model`, une moyenne mobile exponentielle de ces poids
(`ExpMovingAverager`, `decay: 0.9998`, activée sur les 3 configs). Une
EMA d'un modèle qui progresse en continu retarde mécaniquement sur
l'état courant — elle prédit directement val (EMA) > train (poids
bruts), la bonne direction.

Le retard, exprimé en epochs, dépend de la résolution. La formule de
decay (`nanodet/model/weight_averager/ema.py`) :
`decay = 0.9998 * exp(-(1+iteration)/2000) + 0.0002` est fonction du
nombre d'**itérations absolues**, pas des epochs. Le batch diminuant
avec la résolution (24→16→8→6 pour 416/512/768/896), le nombre
d'itérations par epoch augmente dans l'autre sens (718→1078→2156→2875).
Fenêtre de lissage EMA une fois stabilisée, convertie en epochs
(calculée directement depuis la formule, `1/decay_stabilisé /
itérations_par_epoch`) :

| Résolution | Itérations/epoch | Fenêtre EMA (epochs) |
|---|---|---|
| 416px | 718 | **~7,0** |
| 512px | 1078 | ~4,6 |
| 768px | 2156 | ~2,3 |
| 896px (gen4, en cours) | 2875 | ~1,7 |

À 416px, la val loss reflète une moyenne lissée sur ~7 epochs
d'historique — retard important sur un modèle qui progresse encore
epoch après epoch, d'où l'écart visible. Le retard se réduit
monotonement avec la résolution (jusqu'à ~1,7 epoch à 896px),
cohérent avec la disparition progressive de l'anomalie déjà observée
512px→768px→896px. La reproduction à l'identique sur un second run
416px (seed différent) est également cohérente avec ce mécanisme :
il dépend du calendrier batch/itérations, pas d'un aléa d'entraînement.

**Non tranché** : les facteurs secondaires listés dans la version
précédente de cette note (`batchsize_per_gpu` plus grand, LR non
réduit à 416px) restent des contributeurs possibles non isolés, mais
le mécanisme EMA ci-dessus explique déjà la direction et l'échelle de
l'écart sans les invoquer.

**Bug de config trouvé en préparant le réentraînement du soir** :
`resume: true` était resté décommenté dans le `.yml` — reliquat de
l'incident OOM du run 768px (section précédente) où ce mécanisme avait
servi une fois. Aucun `model_last.ckpt` présent dans
`workspace/nanodet-plus-m-1.5x_416-person-car/` pour ce nouveau
lancement à froid : tel quel, `train.py` aurait planté immédiatement sur
un fichier introuvable (probablement la cause du `logs.txt` vide à la
racine, 0 octet). Recommenté avant relance.

## Refonte UX de la démo : suppression de la webcam, fenêtres séparées (2026-09-28)

Constat : la webcam n'apportait rien au cas d'usage (surveillance de
zone fixe, pas de poste utilisateur devant sa caméra) et introduisait ses
propres hoquets driver (déjà source de bugs par le passé, cf. plus haut).
**Retiré entièrement** du projet plutôt que simplement désactivé :
`WEBCAM_KEY`/`toggle_webcam()`/`CAMERA_ID` supprimés de
`demo/src/source_cycle.py`, `demo/src/config.py`, `config/demo.yaml` et
des 4 scripts de démo — plus de branche morte à maintenir.

En contrepartie, `config/demo.yaml` gagne `custom_videos_dir`
(`demo/assets/custom/` par défaut, scanné au démarrage par
`demo/src/config.py`) : déposer un fichier vidéo dans ce dossier suffit
à l'ajouter au cycle des sources, sans éditer le YAML.

`04_live_agent_demo.py` (E1/E2) remplacé la saisie console
(`input_worker`, section précédente) par une interface Tkinter à deux
fenêtres (Logs + Assistant), en plus de la fenêtre vidéo OpenCV
existante — trois fenêtres au total. Raison : présenter la démo à
l'oral avec une saisie dans le même terminal que les logs de lancement
était peu lisible, et rien ne distinguait visuellement une question
utilisateur d'une notification d'alerte D3. La fenêtre Assistant
s'ouvre pré-remplie d'un message d'accueil (exemples de questions +
rappel des raccourcis clavier c/r/q de la fenêtre vidéo) ; la fenêtre
Logs affiche en direct tout ce que le logger racine trace (tools et
arguments choisis par l'agent, alertes D3), sans avoir à rouvrir
`agent/data/04_live_agent_demo.log` pendant la démo.

Contrainte technique : Tkinter doit tourner sur le thread principal
(contrairement à `cv2.HighGUI`, qui tolère en pratique un thread non
principal sous Windows, déjà exploité pour le thread producteur de la
section E1/E2). La boucle vidéo (capture/inférence/tracking/affichage,
ancien contenu de `main()`) est donc passée dans son propre thread,
`root.mainloop()` restant sur le thread principal. Chaque question
posée dans la fenêtre Assistant lance aussi son propre thread court
(`agent.ask()` bloque ~1,5-2s) — jamais dans la boucle vidéo ni dans la
boucle Tkinter. Le champ de saisie est désactivé pendant qu'une question
est en cours, pour garantir qu'un seul thread à la fois mute la
`Session` partagée (même invariant que l'ancien `input_worker`, qui le
garantissait naturellement par le blocage séquentiel de `input()`).

Mise à jour d'un widget Tkinter depuis un thread qui n'est pas le thread
principal (nouvelles réponses/alertes venant de la boucle vidéo, lignes
de log venant de n'importe quel thread) : jamais directement, toujours
via `root.after(0, ...)` (thread-safe) ou, pour les logs, une
`queue.Queue` alimentée par un `logging.Handler` dédié et purgée par un
`root.after` périodique côté fenêtre Logs.
