"""Dessine à la souris les zones de danger d'une scène, puis les écrit
dans config/zones.yaml.

Deux types de scène :
  - image + --sprite : scène interactive comme le chantier, où une
    silhouette détourée (PNG RGBA) suit la souris ;
  - vidéo : les zones s'appliquent aux vraies détections. Une seule image
    sert au dessin (la première, ou --frame-index).

Usage :
    uv run python agent/src/alerts/define_zones.py <image_ou_video> \
        --scene-name <nom> --zones <nom1> <nom2> ... \
        [--sprite <png_rgba>] [--label <libellé_démo>] \
        [--frame-index N] [--min-coverage-fraction 0.2] \
        [--max-width 1280] [--max-height ...] [--dry-run]

Contrôles pendant le dessin :
    clic gauche  -- ajoute un point à la zone en cours
    clic droit   -- annule le dernier point de la zone en cours
    'n'          -- ferme la zone en cours (3 points minimum), passe à la suivante
    'r'          -- recommence la zone en cours
    'q'          -- termine et enregistre les zones déjà fermées
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Rendu du texte partagé avec les démos.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "demo" / "src" / "common"))
from text_render import draw_box_label, draw_text, text_color_for  # noqa: E402

MIN_POINTS_PER_ZONE = 3
COLORS = [(66, 135, 245), (66, 245, 111), (245, 66, 197), (245, 173, 66)]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_ZONES_CONFIG_PATH = _PROJECT_ROOT / "config" / "zones.yaml"
_DEMO_CONFIG_PATH = _PROJECT_ROOT / "config" / "demo.yaml"
# Même préfixe que demo/src/common/source_cycle.py
_INTERACTIVE_PREFIX = "INTERACTIVE:"
_PREVIEWS_DIR = _PROJECT_ROOT / "demo" / "assets" / "zone_previews"


def _detect_screen_size(default: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    """Résolution de l'écran via tkinter, `default` si indisponible."""
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
        """Ferme la zone en cours ('n'), s'il y a au moins 3 points."""
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
        print(f"Zone '{self.current_name}' définie ({len(frac_points)} points) : {frac_points}")
        self.current_points_px = []
        self.zone_idx += 1
        if self.current_name is None:
            print("Toutes les zones sont définies -- 'q' pour enregistrer et quitter.")
        else:
            print(f"Zone suivante : '{self.current_name}' -- cliquez vos points, puis 'n' pour la fermer.")

    def _polygon_px(self, name: str) -> np.ndarray:
        pts = [(int(x * self.scale * self.orig_w), int(y * self.scale * self.orig_h)) for x, y in self.completed[name]]
        return np.array(pts, dtype=np.int32)

    def render(self, show_ui: bool = True) -> np.ndarray:
        """Image affichée pendant le dessin. `show_ui=False` : zones seules,
        pour l'aperçu enregistré."""
        img = self.display.copy()
        for i, name in enumerate(self.zone_names[: self.zone_idx]):
            color = COLORS[i % len(COLORS)]
            poly = self._polygon_px(name)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], color, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            top = poly[np.argmin(poly[:, 1])]
            draw_box_label(img, name, int(top[0]), int(top[1]), color, size=15)

        if self.current_name is not None:
            color = COLORS[self.zone_idx % len(COLORS)]
            if len(self.current_points_px) > 1:
                cv2.polylines(
                    img, [np.array(self.current_points_px, dtype=np.int32)], isClosed=False, color=color,
                    thickness=2, lineType=cv2.LINE_AA,
                )
            for px, py in self.current_points_px:
                cv2.circle(img, (px, py), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, (px, py), 4, color, -1, lineType=cv2.LINE_AA)

        if not show_ui:
            return img

        if self.current_name is not None:
            n = len(self.current_points_px)
            step = f"Zone {self.zone_idx + 1}/{len(self.zone_names)}"
            if n >= MIN_POINTS_PER_ZONE:
                hint = f"{n} points — « n » pour fermer la zone"
            else:
                missing = MIN_POINTS_PER_ZONE - n
                hint = f"{n} point{'s' if n > 1 else ''} — encore {missing} avant de pouvoir fermer"
            w_step, _ = draw_text(img, step, (12, 12), size=18, color=(200, 200, 200))
            w_name, _ = draw_text(img, self.current_name, (12 + w_step + 6, 12), size=18, bold=True,
                                  color=text_color_for(color), bg=color, bg_alpha=0.9)
            draw_text(img, hint, (12 + w_step + w_name + 12, 12), size=18)
        else:
            draw_text(img, "Toutes les zones sont dessinées — « q » pour enregistrer et quitter",
                      (12, 12), size=18)

        help_text = "Clic gauche : point · Clic droit : annuler · n : fermer la zone · r : recommencer · q : terminer"
        draw_text(img, help_text, (12, img.shape[0] - 12), size=14, anchor="bottom-left", bg_alpha=0.55)
        return img


def _load_background(source_path: Path, frame_index: int) -> tuple[np.ndarray, str]:
    """Renvoie (image, "image" ou "video") selon l'extension. Pour une
    vidéo, l'image numéro `frame_index`."""
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
    """Mêmes vérifications que la démo, pour échouer avant d'écrire la config."""
    sprite = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
    if sprite is None:
        sys.exit(f"PNG détouré introuvable ou illisible : {sprite_path}")
    if sprite.ndim != 3 or sprite.shape[2] != 4:
        sys.exit(f"Le PNG détouré doit avoir un canal alpha (RGBA), reçu shape={sprite.shape} : {sprite_path}")


def _existing_scene_names() -> list[str]:
    """Pour prévenir quand --scene-name redessine une scène existante."""
    try:
        with open(_ZONES_CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return []
    return list((config.get("scenes") or {}).keys())


def _round_trip_yaml():
    """ruamel.yaml réglé pour réécrire les fichiers de config/ sans perdre
    commentaires, guillemets ni mise en forme."""
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # jamais de retour à la ligne automatique
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _flow_list(values: list):
    """Liste YAML sur une ligne (`[a, b]`)."""
    from ruamel.yaml.comments import CommentedSeq

    seq = CommentedSeq(values)
    seq.fa.set_flow_style()
    return seq


def _write_zones_config(
    scene_name: str, source_str: str, zones: dict, min_coverage_fraction: float | None
) -> str:
    """Ajoute la scène à config/zones.yaml, ou la met à jour sur place.
    Renvoie "ajoutée" ou "mise à jour"."""
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    y = _round_trip_yaml()
    with open(_ZONES_CONFIG_PATH, encoding="utf-8") as f:
        config = y.load(f)
    scenes = config.get("scenes")
    if scenes is None:
        scenes = config["scenes"] = CommentedMap()

    zones_map = CommentedMap()
    for name, points in zones.items():
        zones_map[name] = [_flow_list([x, y_]) for x, y_ in points]

    # zones.py associe une source à UNE scène : si une autre scène utilise
    # déjà cette source, une seule des deux serait active.
    for other_name, other in scenes.items():
        if other_name != scene_name and other.get("source") == source_str:
            print(
                f"Attention : la scène '{other_name}' utilise déjà cette source -- une seule "
                f"des deux sera active. Pour recalibrer, relancer avec --scene-name {other_name}."
            )

    status = "mise à jour" if scene_name in scenes else "ajoutée"
    scene = scenes.get(scene_name)
    if scene is None:
        scene = scenes[scene_name] = CommentedMap()
    scene["source"] = DoubleQuotedScalarString(source_str)
    if min_coverage_fraction is not None:
        scene["min_coverage_fraction"] = min_coverage_fraction
    scene["zones"] = zones_map

    with open(_ZONES_CONFIG_PATH, "w", encoding="utf-8") as f:
        y.dump(config, f)
    return status


def _write_demo_source(label: str, source_entry: str, background_str: str) -> str:
    """Ajoute la scène interactive à demo_sources (config/demo.yaml), ou
    met à jour l'entrée qui a la même image de fond. Renvoie "ajoutée",
    "mise à jour" ou "déjà présente"."""
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    y = _round_trip_yaml()
    with open(_DEMO_CONFIG_PATH, encoding="utf-8") as f:
        config = y.load(f)
    sources = config["demo_sources"]
    entry = _flow_list([DoubleQuotedScalarString(label), DoubleQuotedScalarString(source_entry)])

    status = "ajoutée"
    for i, (_, existing) in enumerate(sources):
        if existing == source_entry:
            return "déjà présente"
        if existing.startswith(_INTERACTIVE_PREFIX) and existing[len(_INTERACTIVE_PREFIX):].split("|")[0] == background_str:
            sources[i] = entry
            status = "mise à jour"
            break
    else:
        sources.append(entry)

    with open(_DEMO_CONFIG_PATH, "w", encoding="utf-8") as f:
        y.dump(config, f)
    return status


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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="N'écrit rien dans config/ : affiche seulement les blocs YAML équivalents.",
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
    # Marge pour la barre de titre et la barre des tâches.
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
    # Fenêtre redimensionnable ; OpenCV recale les clics, sans perte de précision.
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

    # Chemin tel que fourni, relatif à la racine du projet comme dans config/.
    source_str = args.source.replace("\\", "/")
    label = args.label or args.scene_name.replace("_", " ")

    demo_entry = None
    if kind == "image":
        sprite_str = args.sprite.replace("\\", "/")
        demo_entry = f"{_INTERACTIVE_PREFIX}{source_str}|{sprite_str}"

    if args.dry_run:
        print("\n--dry-run : rien n'est écrit. Blocs équivalents :\n")
        print("config/zones.yaml (sous 'scenes:') :")
        print(f"  {args.scene_name}:")
        print(f'    source: "{source_str}"')
        if args.min_coverage_fraction is not None:
            print(f"    min_coverage_fraction: {args.min_coverage_fraction}")
        print("    zones:")
        for name, points in drawer.completed.items():
            print(f"      {name}:")
            for x, y in points:
                print(f"        - [{x}, {y}]")
        if demo_entry:
            print("\nconfig/demo.yaml (sous 'demo_sources:') :")
            print(f'  - ["{label}", "{demo_entry}"]')
    else:
        status = _write_zones_config(args.scene_name, source_str, drawer.completed, args.min_coverage_fraction)
        print(f"\nScène '{args.scene_name}' {status} dans config/zones.yaml.")
        if demo_entry:
            status = _write_demo_source(label, demo_entry, source_str)
            print(f"Scène interactive {status} dans le cycle de config/demo.yaml (touche 'c').")
        else:
            print(
                "Vidéo : ses zones sont actives dès qu'elle est affichée, à condition "
                "qu'elle soit dans le cycle -- listée dans demo_sources "
                "(config/demo.yaml) ou déposée dans demo/assets/custom/."
            )

    # Nommé d'après la scène : la redessiner remplace son aperçu.
    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    annotated_path = _PREVIEWS_DIR / f"{args.scene_name}.png"
    cv2.imwrite(str(annotated_path), drawer.render(show_ui=False))
    print(f"\nAperçu enregistré : {annotated_path}")


if __name__ == "__main__":
    main()
