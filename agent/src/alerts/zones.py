"""Zones de danger — détection d'ENTRÉE d'un objet dans une zone
prédéfinie (config/zones.yaml), en coordonnées NORMALISÉES [0,1],
indépendantes de la résolution de la source vidéo (voir ce fichier pour
la justification).

Zones organisées PAR SCÈNE : chaque scène (config/zones.yaml, clé
`scenes`) associe un chemin de source (vidéo réelle ou image de fond
d'une scène interactive) à son propre jeu de zones nommées. `ZoneMonitor`
ne surveille que les zones de la scène active (`set_scene`, appelé par
demo/src/scripts/03_live_agent_demo.py à chaque changement de source) -- une
source sans scène correspondante n'a simplement aucune zone surveillée,
`check()` renvoie alors toujours une liste vide sans cas particulier à
gérer côté appelant. `ZONES` (l'union de toutes les zones, toutes scènes
confondues) reste exposé séparément pour la validation des noms de zone
côté agent (agent/src/agent/tools.py, tool_schemas.py), qui n'a pas
connaissance de la scène actuellement affichée.

Critère d'entrée : au moins `min_coverage_fraction` (par scène,
config/zones.yaml, 20% par défaut) de la SURFACE de la boîte doit
recouvrir le polygone de la zone -- pas un simple point (un ancien test
au point bas-centre de la boîte déclenchait dès le moindre effleurement
du bord de la zone par ce point précis, ce qui produisait des alertes sur
un passage tangent plutôt qu'une vraie entrée). Estimé par
échantillonnage d'une grille de points sur la boîte plutôt qu'un calcul
exact d'aire d'intersection polygone/rectangle : plus simple, et correct
même sur une zone concave (nombre de sommets libre via
define_zones.py), ce qu'un clipping de Sutherland-Hodgman classique ne
garantit pas.

État "dedans/dehors" reconstruit ENTIÈREMENT à chaque frame à partir des
boîtes confirmées actuellement suivies -- pas de purge manuelle
nécessaire : une piste qui disparaît (perdue par le tracker, ou
réellement sortie) sort naturellement de l'ensemble "dedans" au frame
suivant puisqu'elle n'est plus dans `tracked` (ou plus dans la zone), et
une ré-entrée ultérieure (même track_id, après une brève absence par
exemple) redéclenche normalement -- comportement voulu, pas un bug :
contrairement à l'alerte de durée (qui ne notifie qu'une fois par
franchissement, tant que la piste reste vivante), chaque entrée en zone
est un événement distinct qui mérite sa propre notification.

Comme le journal C2 (event_store.py), les boîtes `is_coasted=True`
(extrapolées par le tracker, cf. tracker.py) sont explicitement ignorées
ici -- ne jamais déclencher une alerte de zone sur une position
prédite plutôt qu'observée.
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
    """Une entrée de `scenes:` dans config/zones.yaml -- un jeu de zones
    nommées rattaché à une source précise (vidéo réelle ou image de fond
    d'une scène interactive)."""

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
    """Chemin absolu normalisé -- insensible à relatif/absolu et aux
    séparateurs '/'/'\\' entre les sources par défaut (chemins relatifs,
    '/', cf. config/demo.yaml) et les vidéos personnelles (chemins
    absolus produits par pathlib, cf. demo/src/config.py). Même fonction
    utilisée ici et par `SourceCycler.zone_source_key`
    (demo/src/source_cycle.py) : les deux DOIVENT produire la même clé
    pour qu'une scène soit reconnue comme correspondant à la source
    active."""
    return str((_PROJECT_ROOT / raw_source).resolve())


# {nom_scene: Scene}, et son inverse {source_key: Scene} pour la
# recherche par source active (voir scene_for_source ci-dessous).
SCENES: dict[str, Scene] = {}
_SCENES_BY_SOURCE: dict[str, Scene] = {}
# {nom_zone: [(x_frac, y_frac), ...]} -- union de TOUTES les zones, toutes
# scènes confondues (la dernière scène l'emporte en cas de nom en double,
# cas non rencontré en pratique -- noms choisis distincts entre scènes).
# Utilisé uniquement pour la validation côté agent (tools.py,
# tool_schemas.py), qui n'a pas connaissance de la scène active.
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
    """`source_key` : `SourceCycler.zone_source_key` de la source active
    (demo/src/source_cycle.py) -- même normalisation des deux côtés.
    Renvoie `None` si aucune scène n'est configurée pour cette source
    (comportement normal pour la plupart des vidéos -- pas une erreur)."""
    return _SCENES_BY_SOURCE.get(source_key)

# Valeur d'échappement pour `zone_name` (tool_schemas.py) -- PAS une zone
# réelle de config/zones.yaml. Sans elle, l'enum de l'outil ne contient que
# des zones réelles : la grammaire de tool-calling force alors le LLM à en
# choisir une même quand l'utilisateur en nomme une qui n'existe pas
# ("parking nord"), ce qui produit un appel avec un nom de zone halluciné
# mais syntaxiquement valide (bug identique observé sur les 4 modèles
# testés, cf. docs/rapport.tex section 9.1). Ajoutée à l'enum comme un choix
# explicite et testable plutôt que de compter sur le seul prompt système
# pour dissuader le contournement.
UNKNOWN_ZONE = "zone_inconnue"


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Test d'appartenance par lancer de rayon (règle pair-impair) --
    suffisant pour des polygones simples non auto-intersectants, seul cas
    d'usage ici (zones dessinées à la main dans la config)."""
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
    """Fraction (0..1) de la surface de la boîte [x1,y1,x2,y2] couverte par
    `polygon`, estimée en testant une grille grid x grid de points
    régulièrement espacés sur la boîte (grid=10 -> 100 points, coût
    négligeable même pour plusieurs boîtes/zones par frame)."""
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
        # Pistes actuellement "dedans", par zone -- reconstruit entièrement
        # à chaque appel de check() (voir docstring du module), jamais
        # accumulé ni purgé manuellement.
        self._currently_inside: dict[str, set[tuple[str, int]]] = {}

    def set_scene(self, scene: "Scene | None") -> None:
        """À appeler à chaque changement de source (voir
        demo/src/scripts/03_live_agent_demo.py) -- bascule sur le jeu de zones de
        la nouvelle scène (`scene=None` : aucune scène configurée pour
        cette source, `self.zones` redevient vide et `check()` ne fait
        plus rien). Oublie aussi les pistes actuellement "dedans" : sans
        ça, un track_id réutilisé depuis 0 par la nouvelle scène pourrait
        être vu à tort comme "déjà dedans" et faire manquer sa vraie
        entrée en zone."""
        self.scene = scene
        self.zones = scene.zones if scene else {}
        self.min_coverage_fraction = scene.min_coverage_fraction if scene else DEFAULT_MIN_COVERAGE_FRACTION
        self._currently_inside = {name: set() for name in self.zones}

    def check(
        self, tracked: dict, class_names: list[str], frame_shape: tuple[int, int], now: float | None = None
    ) -> list[dict]:
        """`tracked` : sortie de MultiClassByteTracker.update() (boîtes
        confirmées + coasted -- ces dernières sont ignorées ci-dessous).
        `frame_shape` : (hauteur, largeur, ...) de l'image réelle, pour
        reconvertir les zones normalisées en pixels. Renvoie la liste des
        entrées qui viennent de se produire CE frame-ci (typiquement vide) :
        [{"alert_type": "zone", "zone_name":..., "object_class":...,
          "track_id":..., "detail": "..."}, ...] -- uniquement pour les
        entrées couvertes par une règle configurée (`set_zone_alert`) ;
        l'entrée est journalisée dans tous les cas via `log_zone_entry`,
        qu'une règle existe ou non."""
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
                        continue  # jamais sur une position extrapolée -- même invariant que C2
                    if _box_coverage_fraction(x1, y1, x2, y2, polygon_px) < self.min_coverage_fraction:
                        continue
                    key = (class_name, tid)
                    currently_inside_this_frame.add(key)
                    if key in self._currently_inside[zone_name]:
                        continue  # déjà dedans au frame précédent -- pas une nouvelle entrée
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
