"""C1 — Intégration tracker : IDs persistants sur les détections.

Enveloppe `ByteTrackTracker` (package `trackers`, successeur de
`supervision.ByteTrack` déprécié depuis 0.28.0), avec une instance
séparée par classe : la librairie n'a aucune notion de classe en
interne, une seule instance partagée entre person/car pourrait donc
associer une détection "car" à une piste "person" par simple
recouvrement géométrique.

Entrée : le format déjà utilisé partout dans le projet depuis B2.6
(sortie de `NanoDetPlusHead.post_process`) :
    {class_idx: [[x1, y1, x2, y2, score], ...], ...}

Sortie : même structure, avec 3 champs ajoutés par boîte :
    {class_idx: [[x1, y1, x2, y2, score, tracker_id, is_coasted, origin], ...], ...}

`is_coasted` (bool) distingue une boîte affichée avec une confiance
pleine (False, trait plein) d'une boîte affichée à titre indicatif
seulement (True, pointillé). `origin` (str) précise laquelle des deux
raisons possibles a produit un `is_coasted=True` -- deux origines
distinctes, avec un mot-clé d'affichage propre à chacune (voir
`draw_tracked` dans les scripts de démo) :
  - `"predicted"` -- position extrapolée par le filtre de Kalman, aucune
    détection du tout ce frame-ci ("en roue libre" au sens strict).
    Plafonné en cumulé par `MAX_CUMULATIVE_PREDICTED_SECONDS` : passé ce
    budget de temps réel total en extrapolation depuis la dernière
    confirmation solide, la piste redevient invisible sur un raté plutôt
    que de continuer à fantômer indéfiniment épisode après épisode.
  - `"weak"` -- une détection réelle a bien eu lieu ce frame-ci (même via
    la passe basse confiance de ByteTrack) mais son score, une fois la
    piste déjà visible, est retombé sous `HYSTERESIS_LOW` -- affichée
    quand même avec sa position réelle plutôt que de disparaître net
    puis réapparaître (bug trouvé en testant 02_tracking_demo.py sur une
    cible à score oscillant).
  - `"confirmed"` -- boîte à pleine confiance (`is_coasted=False`) ;
    valeur portée pour la cohérence du format, non lue par l'affichage.

Dans tous les cas `is_coasted=True`, à l'appelant de ne PAS transmettre
la boîte au journal d'événements (C2) -- ni une extrapolation ni un
match trop faible pour rester confirmé ne doivent compter comme une
observation réelle pour les durées de présence.

Une piste **jamais encore devenue visible** (n'a encore jamais franchi
`HYSTERESIS_HIGH`) n'apparaît toujours pas dans la sortie tant que ce
seuil n'est pas atteint -- le pointillé de secours ci-dessus ne
s'applique qu'à une piste qui *a déjà été* visible au moins une fois
(mémorisé de façon persistante, indépendamment des allers-retours du
trigger de Schmitt lui-même -- une piste qui redescend en "weak" reste
éligible au coasting "predicted" ensuite, jusqu'à ce que le tracker
l'abandonne pour de bon). Un score déjà
filtré côté appelant (ex. `SCORE_THRESHOLD` dans les scripts de démo)
est donc redondant pour les boîtes confirmées : tout ce qui sort d'ici
est déjà destiné à être affiché."""

from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from trackers import ByteTrackTracker

# Tous les paramètres ajustables du tracker (seuils ByteTrack, lissage,
# hystérésis, budget de coasting cumulé) vivent dans config/tracker.yaml
# (racine du projet) plutôt que dans le code -- chaque valeur y est
# justifiée en détail (recalibrées à partir de deux faits mesurés
# empiriquement sur ce projet : scores modestes du modèle et débit réel
# bien sous les 30 FPS supposés par les défauts de la librairie
# `trackers`). Modifier ce YAML seul suffit, aucun besoin de toucher à ce
# fichier pour ajuster un réglage.
_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "tracker.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

DEFAULT_FRAME_RATE: float = _CONFIG["frame_rate"]
DEFAULT_TRACKER_KWARGS: dict = _CONFIG["tracker_kwargs"]
SMOOTHING_ALPHA: float = _CONFIG["smoothing_alpha"]
HYSTERESIS_HIGH: float = _CONFIG["hysteresis_high"]
HYSTERESIS_LOW: float = _CONFIG["hysteresis_low"]
MAX_CUMULATIVE_PREDICTED_SECONDS: float = _CONFIG["max_cumulative_predicted_seconds"]


class MultiClassByteTracker:
    def __init__(self, class_names: list[str], frame_rate: float = DEFAULT_FRAME_RATE, **tracker_kwargs):
        self.class_names = class_names
        kwargs = {**DEFAULT_TRACKER_KWARGS, **tracker_kwargs, "frame_rate": frame_rate}
        self.trackers = {cls_idx: ByteTrackTracker(**kwargs) for cls_idx in range(len(class_names))}
        # Lissage et hystérésis par (classe, tracker_id) -- deux trackers
        # séparés (un par classe) allouent chacun leurs propres IDs à
        # partir de 1, donc un même entier peut désigner deux pistes
        # différentes selon la classe.
        self._smoothed_boxes: dict[tuple[int, int], np.ndarray] = {}
        self._visible_ids: set[tuple[int, int]] = set()
        # Contrairement à `_visible_ids` (bascule à chaque frame selon le
        # trigger de Schmitt), celui-ci ne s'efface JAMAIS tant que la piste
        # sous-jacente est vivante -- purgé uniquement au même moment que
        # `_smoothed_boxes`/`_visible_ids` quand le tracker abandonne
        # définitivement la piste (voir purge en fin de update()). Sert
        # uniquement à décider l'éligibilité au coasting "predicted" : sans
        # cette distinction, une piste qui passe par "weak" (retirée de
        # `_visible_ids` par `_passes_hysteresis`) perdait aussitôt le droit
        # de coaster si le frame suivant tombait à zéro détection -- bug
        # trouvé en testant une occlusion progressive (feuille de papier).
        self._ever_confirmed: set[tuple[int, int]] = set()
        # Cumul, en secondes réelles, de temps passé en coasting "predicted"
        # depuis la dernière confirmation solide -- voir
        # MAX_CUMULATIVE_PREDICTED_SECONDS. Remis à zéro à chaque
        # franchissement de HYSTERESIS_HIGH (_passes_hysteresis), purgé à la
        # mort de la piste comme les ensembles ci-dessus.
        self._predicted_seconds: dict[tuple[int, int], float] = {}
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        """Repart de zéro -- à appeler impérativement en changeant de flux vidéo
        (webcam <-> fichier, ou fichier <-> fichier). Sans ça, les pistes de
        l'ancien flux (positions, filtres de Kalman) survivraient sur les
        premières frames du nouveau flux -- une scène totalement différente --
        et `ByteTrackTracker.reset()` remet ses compteurs d'ID à zéro, ce qui
        ferait entrer en collision un nouveau (classe, tracker_id) avec une
        ancienne ligne déjà présente dans le journal d'événements si celui-ci
        n'est pas nettoyé en même temps (voir EventStore.clear_events(),
        appelé par les scripts de démo au moment du changement de source)."""
        for tracker in self.trackers.values():
            tracker.reset()
        self._smoothed_boxes.clear()
        self._visible_ids.clear()
        self._ever_confirmed.clear()
        self._predicted_seconds.clear()
        self._last_timestamp = None

    def _smooth(self, cls_idx: int, tracker_id: int, box: np.ndarray) -> np.ndarray:
        key = (cls_idx, tracker_id)
        box = np.asarray(box, dtype=np.float32)
        previous = self._smoothed_boxes.get(key)
        smoothed = box if previous is None else SMOOTHING_ALPHA * box + (1 - SMOOTHING_ALPHA) * previous
        self._smoothed_boxes[key] = smoothed
        return smoothed

    def _passes_hysteresis(self, cls_idx: int, tracker_id: int, score: float) -> bool:
        """Trigger de Schmitt : franchir HYSTERESIS_HIGH rend une piste
        visible ; elle ne redevient invisible qu'en repassant sous
        HYSTERESIS_LOW (pas simplement HYSTERESIS_HIGH à nouveau)."""
        key = (cls_idx, tracker_id)
        # Vérifié EN PREMIER, indépendamment de l'état courant : un score qui
        # franchit HYSTERESIS_HIGH doit regagner le budget de coasting même
        # si la piste était déjà "visible" via le bas du trigger (sinon ce
        # reset n'était jamais atteint pour une piste qui oscille entre
        # confirmée-basse et forte sans jamais redescendre sous
        # HYSTERESIS_LOW entre les deux -- bug trouvé en testant le plafond
        # cumulé juste après l'avoir ajouté).
        if score >= HYSTERESIS_HIGH:
            self._visible_ids.add(key)
            self._ever_confirmed.add(key)
            self._predicted_seconds[key] = 0.0  # confirmation solide -- budget de coasting regagné
            return True
        if key in self._visible_ids:
            if score < HYSTERESIS_LOW:
                self._visible_ids.discard(key)
                return False
            return True
        return False

    def update(self, dets: dict, timestamp: float) -> dict:
        """dets : sortie de post_process, {class_idx: [[x1,y1,x2,y2,score], ...]}.
        Renvoie le même format, avec `tracker_id` et `is_coasted` ajoutés
        (voir docstring du module). Les détections pas encore confirmées
        comme piste stable (minimum_consecutive_frames pas encore atteint)
        sont écartées (tracker_id == -1 côté librairie)."""
        # Temps réel écoulé depuis le dernier appel -- utilisé pour incrémenter
        # le budget de coasting cumulé (MAX_CUMULATIVE_PREDICTED_SECONDS),
        # commun à toutes les classes puisqu'elles partagent le même `timestamp`.
        elapsed = 0.0 if self._last_timestamp is None else max(0.0, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        tracked = {}
        for cls_idx in range(len(self.class_names)):
            boxes = dets.get(cls_idx, [])

            if boxes:
                xyxy = np.array([b[:4] for b in boxes], dtype=np.float32)
                confidence = np.array([b[4] for b in boxes], dtype=np.float32)
                detections = sv.Detections(xyxy=xyxy, confidence=confidence)
            else:
                detections = sv.Detections.empty()

            result = self.trackers[cls_idx].update(detections, timestamp=timestamp)

            boxes_out = []
            confirmed_ids = set()
            for xyxy_row, conf, tid in zip(result.xyxy, result.confidence, result.tracker_id):
                if tid == -1:
                    continue
                tid = int(tid)
                confirmed_ids.add(tid)  # compte pour le coasting même si masqué par l'hystérésis
                was_visible = (cls_idx, tid) in self._visible_ids
                if not self._passes_hysteresis(cls_idx, tid, float(conf)):
                    if was_visible:
                        # Piste déjà visible, réellement matchée ce frame-ci
                        # (même faiblement, via la passe basse confiance de
                        # ByteTrack) mais sous HYSTERESIS_LOW -- affichée en
                        # pointillé avec sa position réelle plutôt que de
                        # disparaître net puis réapparaître : bug trouvé en
                        # testant 02_tracking_demo.py (cible à score
                        # oscillant, ex. jouet à angle limite). Sans ce
                        # traitement, ce cas ne passait jamais par la boucle
                        # de coasting ci-dessous (réservée aux pistes non
                        # matchées, `time_since_update > 0`), puisqu'un match
                        # même faible remet `time_since_update` à 0.
                        smoothed = self._smooth(cls_idx, tid, xyxy_row)
                        boxes_out.append([*smoothed.tolist(), float(conf), tid, True, "weak"])
                    continue
                smoothed = self._smooth(cls_idx, tid, xyxy_row)
                boxes_out.append([*smoothed.tolist(), float(conf), tid, False, "confirmed"])

            # Pistes "en roue libre" : confirmées, pas ré-associées à une
            # détection ce frame-ci (time_since_update > 0), mais toujours
            # dans la fenêtre de tolérance interne du tracker (sinon il les
            # aurait déjà retirées de self.tracks). Position lue directement
            # sur l'état du filtre de Kalman -- accès à l'état interne de la
            # librairie (pas exposé par l'API publique .update()), seul
            # moyen d'obtenir cette extrapolation.
            for track in self.trackers[cls_idx].tracks:
                if track.tracker_id == -1 or track.tracker_id in confirmed_ids:
                    continue
                if track.time_since_update <= 0:
                    continue
                if (cls_idx, track.tracker_id) not in self._ever_confirmed:
                    continue  # jamais passée l'hystérésis pendant qu'elle était confirmée
                pred_key = (cls_idx, track.tracker_id)
                if self._predicted_seconds.get(pred_key, 0.0) >= MAX_CUMULATIVE_PREDICTED_SECONDS:
                    continue  # budget de coasting cumulé épuisé -- doit se reconfirmer solidement
                self._predicted_seconds[pred_key] = self._predicted_seconds.get(pred_key, 0.0) + elapsed
                bbox = track.get_state_bbox()
                smoothed = self._smooth(cls_idx, track.tracker_id, bbox)
                boxes_out.append([*smoothed.tolist(), 0.0, int(track.tracker_id), True, "predicted"])

            tracked[cls_idx] = boxes_out

            # Purge le lissage et l'état d'hystérésis des pistes que le
            # tracker a définitivement abandonnées (au-delà de
            # lost_track_buffer) -- évite une fuite mémoire sur une démo
            # qui tourne longtemps.
            alive_ids = {t.tracker_id for t in self.trackers[cls_idx].tracks}
            for key in [k for k in self._smoothed_boxes if k[0] == cls_idx and k[1] not in alive_ids]:
                del self._smoothed_boxes[key]
            self._visible_ids = {k for k in self._visible_ids if k[0] != cls_idx or k[1] in alive_ids}
            self._ever_confirmed = {k for k in self._ever_confirmed if k[0] != cls_idx or k[1] in alive_ids}
            for key in [k for k in self._predicted_seconds if k[0] == cls_idx and k[1] not in alive_ids]:
                del self._predicted_seconds[key]

        return tracked
