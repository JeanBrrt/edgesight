"""Outil de calibration de zones -- ouvre une image (scène interactive
silhouette+souris) ou une vidéo (zones vérifiées sur les vraies
détections dedans) et laisse cliquer un nombre LIBRE de points (3
minimum) par zone pour définir un polygone. Affiche le résultat en temps
réel et imprime à la fin les blocs YAML prêts à coller dans
config/zones.yaml (et config/demo.yaml si la scène est une nouvelle
scène interactive) -- voir le README, section "Ajouter une scène avec
zones de danger", pour un exemple complet. Un aperçu des zones dessinées
est enregistré dans demo/assets/zone_previews/<scene-name>.png.

Chaque scène définie ici est INDÉPENDANTE des autres (config/zones.yaml,
clé `scenes`) : la scène du chantier déjà fournie n'est jamais touchée
par l'ajout d'une nouvelle scène, `agent/src/alerts/zones.py` sélectionne
automatiquement le bon jeu de zones selon la source affichée
(`SourceCycler.zone_source_key`, demo/src/source_cycle.py).

Deux types de scène :
  - IMAGE (+ --sprite obligatoire) -- une scène interactive comme le
    chantier fourni : fond fixe + silhouette détourée (RGBA) qui suit la
    souris (voir InteractiveOverlaySource, demo/src/source_cycle.py). Les
    zones sont vérifiées contre cette silhouette synthétique, jamais
    contre une vraie personne/voiture.
  - VIDÉO -- les zones sont vérifiées contre les vraies détections de
    cette vidéo (personne/voiture réelles, comme n'importe quelle autre
    source du cycle). `--sprite` n'a pas de sens ici et est ignoré si
    fourni. Une frame de la vidéo (par défaut la première, --frame-index
    pour en choisir une autre) sert de fond pour dessiner les zones.

Usage :
    uv run python agent/src/alerts/define_zones.py <image_ou_video> \
        --scene-name <nom> --zones <nom1> <nom2> ... \
        [--sprite <png_rgba>] [--label <libellé_démo>] \
        [--frame-index N] [--min-coverage-fraction 0.2] \
        [--max-width 1280] [--max-height ...]

Contrôles pendant le dessin :
    clic gauche  -- ajoute un point à la zone en cours
    clic droit   -- annule le dernier point de la zone en cours
    'n'          -- ferme la zone en cours (3 points minimum) et passe
                    à la suivante
    'r'          -- redémarre la zone en cours depuis zéro
    'q'          -- quitte et imprime le résultat (les zones déjà
                    bouclées sont conservées même si tout n'est pas fini)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

MIN_POINTS_PER_ZONE = 3
COLORS = [(66, 135, 245), (66, 245, 111), (245, 66, 197), (245, 173, 66)]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_ZONES_CONFIG_PATH = _PROJECT_ROOT / "config" / "zones.yaml"
_PREVIEWS_DIR = _PROJECT_ROOT / "demo" / "assets" / "zone_previews"


def _detect_screen_size(default: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    """Résolution d'écran réelle via tkinter (stdlib, pas de dépendance
    supplémentaire) -- juste pour dimensionner la fenêtre, jamais affichée.
    Retombe sur `default` si tkinter est indisponible (environnement
    headless) plutôt que de planter l'outil pour un simple confort
    d'affichage."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return default


class ZoneDrawer:
    def __init__(self, display_image: np.ndarray, original_shape: tuple[int, int], zone_names: list[str], scale: float):
        self.display = display_image
        self.orig_h, self.orig_w = original_shape
        self.zone_names = zone_names
        self.scale = scale  # display_px = original_px * scale
        self.completed: dict[str, list[tuple[float, float]]] = {}
        self.zone_idx = 0
        self.current_points_px: list[tuple[int, int]] = []

    @property
    def current_name(self) -> str | None:
        if self.zone_idx >= len(self.zone_names):
            return None
        return self.zone_names[self.zone_idx]

    def add_point(self, x: int, y: int) -> None:
        if self.current_name is None:
            return
        self.current_points_px.append((x, y))

    def undo_point(self) -> None:
        if self.current_points_px:
            self.current_points_px.pop()

    def reset_current(self) -> None:
        self.current_points_px = []

    def close_current_zone(self) -> None:
        """Ferme la zone en cours sur demande explicite ('n') plutôt
        qu'après un nombre fixe de points -- un polygone valide demande
        au moins 3 points, pas de maximum."""
        if self.current_name is None:
            return
        if len(self.current_points_px) < MIN_POINTS_PER_ZONE:
            print(
                f"Zone '{self.current_name}' : {len(self.current_points_px)} point(s), "
                f"il en faut au moins {MIN_POINTS_PER_ZONE} pour fermer."
            )
            return
        frac_points = [
            (round(px / self.scale / self.orig_w, 3), round(py / self.scale / self.orig_h, 3))
            for px, py in self.current_points_px
        ]
        self.completed[self.current_name] = frac_points
        print(f"Zone '{self.current_name}' definie ({len(frac_points)} points) : {frac_points}")
        self.current_points_px = []
        self.zone_idx += 1
        if self.current_name is None:
            print("Toutes les zones sont definies -- 'q' pour quitter et afficher le YAML.")
        else:
            print(f"Zone suivante : '{self.current_name}' -- cliquez vos points, puis 'n' pour la fermer.")

    def _polygon_px(self, name: str) -> np.ndarray:
        pts = [(int(x * self.scale * self.orig_w), int(y * self.scale * self.orig_h)) for x, y in self.completed[name]]
        return np.array(pts, dtype=np.int32)

    def render(self) -> np.ndarray:
        img = self.display.copy()
        for i, name in enumerate(self.zone_names[: self.zone_idx]):
            color = COLORS[i % len(COLORS)]
            poly = self._polygon_px(name)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)
            cv2.putText(img, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.current_name is not None:
            color = COLORS[self.zone_idx % len(COLORS)]
            for px, py in self.current_points_px:
                cv2.circle(img, (px, py), 5, color, -1)
            if len(self.current_points_px) > 1:
                cv2.polylines(
                    img, [np.array(self.current_points_px, dtype=np.int32)], isClosed=False, color=color, thickness=2
                )
            n = len(self.current_points_px)
            ready = " -- 'n' pour fermer" if n >= MIN_POINTS_PER_ZONE else f" -- {MIN_POINTS_PER_ZONE - n} de plus pour pouvoir fermer"
            status = f"Zone '{self.current_name}' : {n} point(s){ready}"
        else:
            status = "Zones terminees -- 'q' pour quitter"

        # Contour noir + texte blanc, pour rester lisible quel que soit le fond.
        cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img


def _load_background(source_path: Path, frame_index: int) -> tuple[np.ndarray, str]:
    """Renvoie (image_bgr, kind) où kind vaut "image" ou "video" --
    déterminé par l'extension (mêmes listes que demo/src/config.py pour
    les vidéos personnelles). Pour une vidéo, extrait la frame
    `frame_index` (0 = première) comme fond de calibration : une image
    fixe suffit, les zones sont des polygones normalisés indépendants du
    reste de la vidéo."""
    suffix = source_path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        image = cv2.imread(str(source_path))
        if image is None:
            sys.exit(f"Impossible de lire l'image : {source_path}")
        return image, "image"
    if suffix in _VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            sys.exit(f"Impossible d'ouvrir la vidéo : {source_path}")
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            sys.exit(f"Impossible de lire la frame {frame_index} de : {source_path}")
        return frame, "video"
    sys.exit(
        f"Extension non reconnue : '{suffix}'. "
        f"Images : {sorted(_IMAGE_EXTENSIONS)} -- Vidéos : {sorted(_VIDEO_EXTENSIONS)}"
    )


def _validate_sprite(sprite_path: Path) -> None:
    """Mêmes vérifications qu'InteractiveOverlaySource (demo/src/source_cycle.py)
    -- échoue ici plutôt qu'au prochain lancement de la démo, une fois la
    scène déjà collée dans les fichiers de config."""
    sprite = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
    if sprite is None:
        sys.exit(f"PNG détouré introuvable ou illisible : {sprite_path}")
    if sprite.ndim != 3 or sprite.shape[2] != 4:
        sys.exit(f"Le PNG détouré doit avoir un canal alpha (RGBA), reçu shape={sprite.shape} : {sprite_path}")


def _existing_scene_names() -> list[str]:
    """Juste pour prévenir si --scene-name recalibre une scène déjà
    définie (recalibrer est un usage légitime, cf. commentaire sur
    'chantier' dans config/zones.yaml -- pas une erreur, seulement une
    confirmation utile)."""
    try:
        with open(_ZONES_CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return list((config.get("scenes") or {}).keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Image (scène interactive) ou vidéo (zones sur détections réelles) servant de fond.")
    parser.add_argument("--scene-name", required=True, help="Nom de la scène (clé sous 'scenes:' dans config/zones.yaml).")
    parser.add_argument("--zones", nargs="+", required=True, help="Noms des zones à définir, dans l'ordre.")
    parser.add_argument(
        "--sprite", default=None,
        help="PNG détouré (RGBA) qui suivra la souris, à ranger dans demo/assets/silhouettes/ "
        "-- obligatoire si 'source' est une image, ignoré si c'est une vidéo.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Libellé affiché dans le cycle de sources (touche 'c') -- par défaut dérivé de --scene-name.",
    )
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="Vidéo uniquement : index de la frame à utiliser comme fond de calibration (0 = première).",
    )
    parser.add_argument(
        "--min-coverage-fraction", type=float, default=None,
        help="Surcharge min_coverage_fraction pour cette scène uniquement (par défaut : valeur globale de config/zones.yaml).",
    )
    parser.add_argument(
        "--max-width", type=int, default=None,
        help="Largeur max d'affichage -- par defaut, deduite automatiquement de la resolution d'ecran detectee.",
    )
    parser.add_argument(
        "--max-height", type=int, default=None,
        help="Hauteur max d'affichage -- par defaut, deduite automatiquement de la resolution d'ecran detectee.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        sys.exit(f"Fichier introuvable : {source_path}")
    original, kind = _load_background(source_path, args.frame_index)

    if kind == "image" and not args.sprite:
        sys.exit("--sprite est obligatoire pour une scène interactive (source image).")
    if kind == "video" and args.sprite:
        print("Note : --sprite ignoré -- une scène vidéo vérifie les vraies détections, pas de silhouette synthétique.")
        args.sprite = None
    sprite_path = None
    if args.sprite:
        sprite_path = Path(args.sprite)
        if not sprite_path.exists():
            sys.exit(f"Sprite introuvable : {sprite_path}")
        _validate_sprite(sprite_path)

    existing = _existing_scene_names()
    if args.scene_name in existing:
        print(f"Note : '{args.scene_name}' existe déjà dans config/zones.yaml -- ce lancement la recalibre.")

    screen_w, screen_h = _detect_screen_size()
    # Marge sous la résolution d'écran détectée -- sinon une image dont le
    # ratio colle à celui de l'écran déborde légèrement (barre de titre,
    # décorations de fenêtre, barre des tâches).
    max_w = args.max_width or int(screen_w * 0.9)
    max_h = args.max_height or int(screen_h * 0.85)

    h, w = original.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    display = cv2.resize(original, (int(w * scale), int(h * scale))) if scale < 1.0 else original.copy()
    print(
        f"Source ({kind}) : {w}x{h} -- ecran detecte : {screen_w}x{screen_h} -- "
        f"affichage a l'echelle {scale:.2f} ({display.shape[1]}x{display.shape[0]})"
    )

    drawer = ZoneDrawer(display, (h, w), args.zones, scale)

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawer.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawer.undo_point()

    window = "Definition de zones -- clic gauche: point / clic droit: annuler / n: fermer / r: reset / q: quitter"
    # WINDOW_NORMAL (plutôt que le défaut WINDOW_AUTOSIZE) rend la fenêtre
    # redimensionnable à la souris -- OpenCV remappe automatiquement les
    # coordonnées de clic vers l'espace de `display` quel que soit le
    # redimensionnement, donc pas d'impact sur la précision des zones.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    cv2.setMouseCallback(window, on_mouse)

    print(
        f"Zone 1/{len(args.zones)} : '{drawer.current_name}' -- cliquez vos points "
        f"({MIN_POINTS_PER_ZONE} minimum), puis 'n' pour fermer."
    )

    while True:
        cv2.imshow(window, drawer.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            drawer.reset_current()
        elif key == ord("n"):
            drawer.close_current_zone()

    cv2.destroyAllWindows()

    if not drawer.completed:
        print("Aucune zone definie.")
        return

    # Chemin tel que fourni sur la ligne de commande, pas résolu en absolu
    # -- attendu relatif à la racine du projet (comme le reste de
    # config/zones.yaml et config/demo.yaml), collé tel quel dans le YAML.
    source_str = args.source.replace("\\", "/")
    label = args.label or args.scene_name.replace("_", " ")

    print("\n--- A coller dans config/zones.yaml (sous la cle 'scenes:') ---\n")
    print(f"  {args.scene_name}:")
    print(f'    source: "{source_str}"')
    if args.min_coverage_fraction is not None:
        print(f"    min_coverage_fraction: {args.min_coverage_fraction}")
    print("    zones:")
    for name, points in drawer.completed.items():
        print(f"      {name}:")
        for x, y in points:
            print(f"        - [{x}, {y}]")

    if kind == "image":
        print("\n--- A coller dans config/demo.yaml (sous la cle 'demo_sources:') ---\n")
        sprite_str = args.sprite.replace("\\", "/")
        print(f'  - ["{label}", "INTERACTIVE:{source_str}|{sprite_str}"]')
    else:
        print(
            "\nVidéo : pas d'entrée à ajouter si le fichier est déjà repris par "
            "config/demo.yaml (demo_sources) ou déposé dans custom_videos_dir "
            "(demo/assets/custom/, repris automatiquement) -- seul le chemin "
            "compte pour que les zones soient reconnues, pas le libellé. Pour "
            "lui donner un libellé personnalisé dans demo_sources :\n"
        )
        print(f'  - ["{label}", "{source_str}"]')

    # Nommé d'après la scène (pas d'après la source) : recalibrer une scène
    # écrase son propre aperçu, et tous les aperçus restent regroupés au
    # même endroit plutôt qu'éparpillés à côté de chaque source.
    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = _PREVIEWS_DIR / f"{args.scene_name}.png"
    cv2.imwrite(str(annotated_path), drawer.render())
    print(f"\nApercu sauvegarde : {annotated_path}")


if __name__ == "__main__":
    main()
