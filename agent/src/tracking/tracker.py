"""Suivi multi-objets : un identifiant stable par objet d'une image à
l'autre.

Entrée (sortie de NanoDet) : {class_idx: [[x1, y1, x2, y2, score], ...]}
Sortie : {class_idx: [[x1, y1, x2, y2, score, tracker_id, is_coasted, origin], ...]}

`origin` :
  - "confirmed" : détection sûre (is_coasted=False) ;
  - "weak" : piste déjà visible, détectée mais sous HYSTERESIS_LOW.
    Affichée en pointillés plutôt que de clignoter ;
  - "predicted" : aucune détection, position prédite par le filtre de
    Kalman. Limitée à MAX_CUMULATIVE_PREDICTED_SECONDS cumulées.
Les boîtes is_coasted=True ne doivent pas aller dans le journal : ce ne
sont pas des observations fiables.

Une piste n'apparaît qu'après avoir franchi HYSTERESIS_HIGH une première
fois : tout ce qui sort d'ici est destiné à être affiché."""

from pathlib import Path

import numpy as np
import supervision as sv
import yaml
from trackers import ByteTrackTracker

# Réglages dans config/tracker.yaml.
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
        # Clés (classe, tracker_id) : chaque tracker numérote ses pistes à
        # partir de 1, un même numéro existe donc dans les deux classes.
        self._smoothed_boxes: dict[tuple[int, int], np.ndarray] = {}
        self._visible_ids: set[tuple[int, int]] = set()
        # Pistes déjà visibles au moins une fois, gardées jusqu'à leur
        # abandon (contrairement à _visible_ids). C'est ce qui autorise la
        # position prédite, y compris après un passage en "weak".
        self._ever_confirmed: set[tuple[int, int]] = set()
        # Secondes passées en position prédite depuis la dernière détection
        # solide (voir MAX_CUMULATIVE_PREDICTED_SECONDS).
        self._predicted_seconds: dict[tuple[int, int], float] = {}
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        """Oublie toutes les pistes, à chaque changement de vidéo. Les
        identifiants repartant de 0, vider aussi le journal
        (EventStore.clear_events())."""
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
        """Hystérésis : visible au-dessus de HYSTERESIS_HIGH, invisible
        seulement sous HYSTERESIS_LOW."""
        key = (cls_idx, tracker_id)
        # Testé en premier, même pour une piste déjà visible : une détection
        # solide doit toujours remettre à zéro le temps de position prédite.
        if score >= HYSTERESIS_HIGH:
            self._visible_ids.add(key)
            self._ever_confirmed.add(key)
            self._predicted_seconds[key] = 0.0
            return True
        if key in self._visible_ids:
            if score < HYSTERESIS_LOW:
                self._visible_ids.discard(key)
                return False
            return True
        return False

    def update(self, dets: dict, timestamp: float) -> dict:
        """Associe les détections aux pistes (formats : voir le module). Les
        pistes pas encore établies (tracker_id == -1) sont écartées."""
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
                confirmed_ids.add(tid)  # détectée cette image, même si masquée
                was_visible = (cls_idx, tid) in self._visible_ids
                if not self._passes_hysteresis(cls_idx, tid, float(conf)):
                    if was_visible:
                        # Détection faible d'une piste visible : en pointillés
                        # plutôt que de clignoter.
                        smoothed = self._smooth(cls_idx, tid, xyxy_row)
                        boxes_out.append([*smoothed.tolist(), float(conf), tid, True, "weak"])
                    continue
                smoothed = self._smooth(cls_idx, tid, xyxy_row)
                boxes_out.append([*smoothed.tolist(), float(conf), tid, False, "confirmed"])

            # Pistes sans détection cette image mais pas encore abandonnées :
            # position prédite, lue dans l'état interne de la librairie
            # (non exposée par son API).
            for track in self.trackers[cls_idx].tracks:
                if track.tracker_id == -1 or track.tracker_id in confirmed_ids:
                    continue
                if track.time_since_update <= 0:
                    continue
                if (cls_idx, track.tracker_id) not in self._ever_confirmed:
                    continue  # jamais encore visible
                pred_key = (cls_idx, track.tracker_id)
                if self._predicted_seconds.get(pred_key, 0.0) >= MAX_CUMULATIVE_PREDICTED_SECONDS:
                    continue  # trop longtemps en position prédite
                self._predicted_seconds[pred_key] = self._predicted_seconds.get(pred_key, 0.0) + elapsed
                bbox = track.get_state_bbox()
                smoothed = self._smooth(cls_idx, track.tracker_id, bbox)
                boxes_out.append([*smoothed.tolist(), 0.0, int(track.tracker_id), True, "predicted"])

            tracked[cls_idx] = boxes_out

            # Oublie les pistes abandonnées par le tracker (sinon la mémoire
            # grossit sur une longue démo).
            alive_ids = {t.tracker_id for t in self.trackers[cls_idx].tracks}
            for key in [k for k in self._smoothed_boxes if k[0] == cls_idx and k[1] not in alive_ids]:
                del self._smoothed_boxes[key]
            self._visible_ids = {k for k in self._visible_ids if k[0] != cls_idx or k[1] in alive_ids}
            self._ever_confirmed = {k for k in self._ever_confirmed if k[0] != cls_idx or k[1] in alive_ids}
            for key in [k for k in self._predicted_seconds if k[0] == cls_idx and k[1] not in alive_ids]:
                del self._predicted_seconds[key]

        return tracked
