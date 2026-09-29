"""Zones de danger
"""

import time
from pathlib import Path

import yaml

from ..journal.event_store import EventStore

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "zones.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f) or {}

DEFAULT_MIN_COVERAGE_FRACTION: float = _CONFIG.get("min_coverage_fraction", 0.2)


class Scene:
    """Zones nommées d'une source (vidéo ou image de fond)."""

    def __init__(
        self,
        name: str,
        source_key: str,
        zones: dict[str, list[tuple[float, float]]],
        min_coverage_fraction: float,
    ):
        self.name = name
        self.source_key = source_key
        self.zones = zones
        self.min_coverage_fraction = min_coverage_fraction


def _resolve_source_key(raw_source: str) -> str:
    """Chemin absolu normalisé. Doit donner la même clé que
    SourceCycler.zone_source_key (demo/src/common/source_cycle.py)."""
    return str((_PROJECT_ROOT / raw_source).resolve())


SCENES: dict[str, Scene] = {}
_SCENES_BY_SOURCE: dict[str, Scene] = {}
# Toutes les zones, toutes scènes confondues, pour l'agent (en cas de nom
# en double, la dernière scène l'emporte).
ZONES: dict[str, list[tuple[float, float]]] = {}

for _scene_name, _scene_cfg in (_CONFIG.get("scenes") or {}).items():
    _zones = {
        _zone_name: [tuple(point) for point in polygon]
        for _zone_name, polygon in (_scene_cfg.get("zones") or {}).items()
    }
    _scene = Scene(
        name=_scene_name,
        source_key=_resolve_source_key(_scene_cfg["source"]),
        zones=_zones,
        min_coverage_fraction=_scene_cfg.get("min_coverage_fraction", DEFAULT_MIN_COVERAGE_FRACTION),
    )
    SCENES[_scene_name] = _scene
    _SCENES_BY_SOURCE[_scene.source_key] = _scene
    ZONES.update(_zones)


def scene_for_source(source_key: str) -> "Scene | None":
    """Scène de la source active, ou None si elle n'a pas de zones."""
    return _SCENES_BY_SOURCE.get(source_key)

# Valeur que le LLM peut choisir quand aucune zone ne correspond (voir
# tool_schemas.py) ; pas une vraie zone.
UNKNOWN_ZONE = "zone_inconnue"


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Point dans un polygone simple, par lancer de rayon."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def _box_coverage_fraction(
    x1: float, y1: float, x2: float, y2: float, polygon: list[tuple[float, float]], grid: int = 10
) -> float:
    """Part (0..1) de la boîte couverte par `polygon`, estimée sur une
    grille de grid x grid points."""
    if grid < 1 or x2 <= x1 or y2 <= y1:
        return 0.0
    inside = 0
    for i in range(grid):
        px = x1 + (x2 - x1) * (i + 0.5) / grid
        for j in range(grid):
            py = y1 + (y2 - y1) * (j + 0.5) / grid
            if _point_in_polygon(px, py, polygon):
                inside += 1
    return inside / (grid * grid)


class ZoneMonitor:
    def __init__(self, store: EventStore):
        self.store = store
        self.scene: Scene | None = None
        self.zones: dict[str, list[tuple[float, float]]] = {}
        self.min_coverage_fraction: float = DEFAULT_MIN_COVERAGE_FRACTION
        # Pistes dans chaque zone à l'image précédente.
        self._currently_inside: dict[str, set[tuple[str, int]]] = {}

    def set_scene(self, scene: "Scene | None") -> None:
        """Passe aux zones d'une autre scène (None : aucune zone), à chaque
        changement de source. Oublie les pistes "dedans", dont les
        identifiants vont être réutilisés."""
        self.scene = scene
        self.zones = scene.zones if scene else {}
        self.min_coverage_fraction = scene.min_coverage_fraction if scene else DEFAULT_MIN_COVERAGE_FRACTION
        self._currently_inside = {name: set() for name in self.zones}

    def check(
        self, tracked: dict, class_names: list[str], frame_shape: tuple[int, int], now: float | None = None
    ) -> list[dict]:
        """Journalise chaque nouvelle entrée en zone et renvoie celles qui
        correspondent à une alerte posée (souvent aucune).

        `tracked` : sortie de MultiClassByteTracker.update().
        `frame_shape` : taille de l'image, pour passer les zones en pixels."""
        now = now if now is not None else time.time()
        height, width = frame_shape[:2]
        newly_fired = []
        zone_rules = [a for a in self.store.get_alerts() if a["alert_type"] == "zone"]

        for zone_name, polygon_frac in self.zones.items():
            polygon_px = [(x * width, y * height) for x, y in polygon_frac]
            rules_for_zone = [r for r in zone_rules if r["zone_name"] == zone_name]

            currently_inside_this_frame = set()
            for cls_idx, boxes in tracked.items():
                class_name = class_names[cls_idx]
                for x1, y1, x2, y2, score, tid, is_coasted, origin in boxes:
                    if is_coasted:
                        continue  # position seulement prédite
                    if _box_coverage_fraction(x1, y1, x2, y2, polygon_px) < self.min_coverage_fraction:
                        continue
                    key = (class_name, tid)
                    currently_inside_this_frame.add(key)
                    if key in self._currently_inside[zone_name]:
                        continue  # déjà dedans : pas une nouvelle entrée
                    self.store.log_zone_entry(zone_name, class_name, tid, timestamp=now)
                    for rule in rules_for_zone:
                        if rule["object_class"] in (class_name, "any"):
                            newly_fired.append(
                                {
                                    "alert_type": "zone",
                                    "zone_name": zone_name,
                                    "object_class": class_name,
                                    "track_id": tid,
                                    "detail": f"{class_name} #{tid} est entré(e) dans la zone '{zone_name}'",
                                }
                            )

            self._currently_inside[zone_name] = currently_inside_this_frame

        return newly_fired
