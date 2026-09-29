"""Sources des démos (vidéos et scènes interactives), changées à la volée
avec 'c', et utilitaires d'affichage (mise à l'échelle, bandeaux).
"""

from pathlib import Path

import cv2
import numpy as np

from config import DEMO_SOURCES, DISPLAY_MAX_WIDTH
from text_render import draw_box_label, draw_hud_line, draw_text, fit_text, text_height

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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


# Hauteur d'affichage max : 85 % de l'écran, pour laisser la place à la
# barre des tâches et à la barre de titre. Sans elle, le bas d'une image
# haute (la scène du chantier) sortait de l'écran.
_MAX_DISPLAY_HEIGHT = int(_detect_screen_size()[1] * 0.85)


def _display_scale(frame) -> float:
    """Facteur de réduction appliqué par resize_for_display (1.0 si
    l'image tient déjà à l'écran)."""
    h, w = frame.shape[:2]
    return min(1.0, DISPLAY_MAX_WIDTH / w, _MAX_DISPLAY_HEIGHT / h)


def ui_scale(frame) -> float:
    """Facteur à passer à text_render : compense la réduction d'affichage,
    pour un texte de même taille à l'écran quelle que soit la source."""
    return 1.0 / _display_scale(frame)


def resize_for_display(frame, max_width: int = DISPLAY_MAX_WIDTH, max_height: int | None = None):
    """Réduit l'image pour qu'elle tienne à l'écran, en largeur et en
    hauteur. INTER_AREA évite de créneler le texte et les traits fins."""
    max_height = _MAX_DISPLAY_HEIGHT if max_height is None else max_height
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


CYCLE_KEY = ord("c")
RESTART_KEY = ord("r")  # sans effet sur une scène interactive

# Scène interactive dans config/demo.yaml :
#   "INTERACTIVE:<image_de_fond>|<png_detoure_RGBA>"
# '|' plutôt que ':', déjà présent dans les chemins Windows (C:\...).
_INTERACTIVE_PREFIX = "INTERACTIVE:"


class InteractiveOverlaySource:
    """Image de fond fixe + silhouette détourée (RGBA) qui suit la souris.
    Même interface que cv2.VideoCapture (read/release) : les démos la
    traitent comme une vidéo.

    Souris : déplace la silhouette. Molette : taille. Clic droit :
    afficher/masquer.
    """

    MIN_SCALE = 0.1
    MAX_SCALE = 3.0
    SCALE_STEP = 0.1

    def __init__(self, background_path: str, sprite_path: str):
        background = cv2.imread(background_path)
        if background is None:
            raise RuntimeError(f"Image de fond introuvable ou illisible : {background_path}")
        self.background = background
        self.h, self.w = background.shape[:2]

        sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if sprite is None:
            raise RuntimeError(f"PNG detoure introuvable ou illisible : {sprite_path}")
        if sprite.ndim != 3 or sprite.shape[2] != 4:
            raise RuntimeError(f"Le PNG detoure doit avoir un canal alpha (RGBA), recu shape={sprite.shape} : {sprite_path}")
        self.sprite_original = sprite  # BGRA

        # Réduction appliquée à l'affichage : sert à ramener la position
        # de la souris dans les coordonnées de l'image d'origine.
        self.display_scale = _display_scale(background)

        self.cursor_native = (self.w // 2, self.h // 2)
        self.scale = 1.0 / 3  # taille de départ, ajustable à la molette
        self.visible = True

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_native = (int(x / self.display_scale), int(y / self.display_scale))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.visible = not self.visible
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Sens de la molette dans les bits 16-31 de `flags` (entier
            # signé). cv2.getMouseWheelDelta() n'existe pas partout.
            wheel = (flags & 0xFFFFFFFF) >> 16
            if wheel >= 0x8000:
                wheel -= 0x10000
            delta = self.SCALE_STEP if wheel > 0 else -self.SCALE_STEP
            self.scale = max(self.MIN_SCALE, min(self.MAX_SCALE, self.scale + delta))

    def _composited_frame(self) -> np.ndarray:
        frame = self.background.copy()
        if not self.visible:
            return frame

        sh, sw = self.sprite_original.shape[:2]
        new_w, new_h = max(1, int(sw * self.scale)), max(1, int(sh * self.scale))
        sprite = cv2.resize(self.sprite_original, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cx, cy = self.cursor_native
        x1, y1 = cx - new_w // 2, cy - new_h // 2
        x2, y2 = x1 + new_w, y1 + new_h

        # Recadrage si la silhouette dépasse du bord de l'image.
        dst_x1, dst_y1 = max(0, x1), max(0, y1)
        dst_x2, dst_y2 = min(self.w, x2), min(self.h, y2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return frame  # sprite entièrement hors cadre
        src_x1, src_y1 = dst_x1 - x1, dst_y1 - y1
        src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)

        roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
        sprite_crop = sprite[src_y1:src_y2, src_x1:src_x2]
        alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (alpha * sprite_crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
        return frame

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self._composited_frame()

    def release(self) -> None:
        pass  # rien à libérer


class SourceCycler:
    def __init__(self):
        self.index = 0
        self.cap = self._open(self.index)

    def _open(self, index: int):
        label, source = DEMO_SOURCES[index]
        if isinstance(source, str) and source.startswith(_INTERACTIVE_PREFIX):
            bg_path, sprite_path = source[len(_INTERACTIVE_PREFIX):].split("|")
            print(f"Source interactive : {label} -- souris: suit / molette: taille / clic droit: afficher-masquer")
            return InteractiveOverlaySource(bg_path, sprite_path)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source « {label} » ({source})")
        print(f"Source vidéo : {label}")
        return cap

    @property
    def label(self) -> str:
        return DEMO_SOURCES[self.index][0]

    def is_file(self) -> bool:
        source = DEMO_SOURCES[self.index][1]
        return isinstance(source, str) and not source.startswith(_INTERACTIVE_PREFIX)

    def is_interactive(self) -> bool:
        return isinstance(self.cap, InteractiveOverlaySource)

    @property
    def zone_source_key(self) -> str:
        """Clé pour retrouver les zones de la source dans config/zones.yaml :
        chemin absolu normalisé, de l'image de fond pour une scène
        interactive."""
        source = DEMO_SOURCES[self.index][1]
        if source.startswith(_INTERACTIVE_PREFIX):
            background_path = source[len(_INTERACTIVE_PREFIX):].split("|")[0]
        else:
            background_path = source
        return str((_PROJECT_ROOT / background_path).resolve())

    def next(self) -> None:
        """Source suivante (revient à la première après la dernière)."""
        self.cap.release()
        self.index = (self.index + 1) % len(DEMO_SOURCES)
        self.cap = self._open(self.index)

    def read(self):
        return self.cap.read()

    def loop_if_file(self) -> None:
        """Revient au début de la vidéo (sans effet sur une scène interactive)."""
        if self.is_file():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        """Transmis à la scène interactive, ignoré pour une vidéo."""
        if isinstance(self.cap, InteractiveOverlaySource):
            self.cap.on_mouse(event, x, y, flags)

    def release(self) -> None:
        self.cap.release()


def draw_zones(frame, zones: dict) -> None:
    """Dessine les zones de danger (coordonnées normalisées -> pixels)."""
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    for name, polygon_frac in zones.items():
        pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)
        cv2.polylines(
            frame, [pts], isClosed=True, color=(0, 0, 255), thickness=max(2, int(round(2 * s))),
            lineType=cv2.LINE_AA,
        )
        # Nom au sommet le plus haut, pour ne pas masquer les boîtes.
        top = pts[np.argmin(pts[:, 1])]
        draw_box_label(frame, name, int(top[0]), int(top[1]), (0, 0, 220), size=14, scale=s)


def draw_source_label(frame, label: str):
    """Affiche la source courante juste sous le FPS (voir fps_counter.py)."""
    draw_hud_line(frame, 1, f"Source : {label}", color=(120, 230, 255), scale=ui_scale(frame))


def draw_controls(frame) -> None:
    """Rappel des touches, sous la source."""
    draw_hud_line(frame, 2, "c : source suivante · r : redémarrer · q : quitter", color=(215, 215, 215),
                  scale=ui_scale(frame))


def draw_banner(
    frame,
    text: str,
    bg_color: tuple,
    at_bottom: bool = False,
    band_height: int = 40,
    size: int = 17,
    bg_alpha: float = 0.85,
) -> None:
    """Bandeau pleine largeur en haut ou en bas, fond semi-transparent et
    texte blanc (tronqué avec « … » s'il est trop long). Sert aux alertes
    et à l'aide de la scène interactive. Tailles en pixels à l'écran."""
    s = ui_scale(frame)
    h, w = frame.shape[:2]
    band_h = int(round(band_height * s))
    y0, y1 = (h - band_h, h) if at_bottom else (0, band_h)
    roi = frame[y0:y1, 0:w]
    tint = np.empty_like(roi)
    tint[:] = bg_color
    cv2.addWeighted(tint, bg_alpha, roi, 1.0 - bg_alpha, 0, dst=roi)
    margin = int(round(14 * s))
    display_text = fit_text(text, w - 2 * margin, size, scale=s)
    text_y = y0 + (band_h - text_height(size, scale=s)) // 2
    draw_text(frame, display_text, (margin, text_y), size=size, bg=None, pad=(0, 0), scale=s)


_INTERACTIVE_HELP_TEXT = (
    "Scène interactive — souris : déplacer la silhouette · molette : redimensionner · "
    "clic droit : afficher / masquer"
)
_INTERACTIVE_HELP_COLOR = (40, 40, 40)
_INTERACTIVE_HELP_BAND_HEIGHT = 44


def draw_interactive_help(frame, is_interactive: bool) -> None:
    """Bandeau bas rappelant les commandes souris, sur une scène
    interactive uniquement.

    Prend un booléen plutôt que le `SourceCycler` : dans
    03_live_agent_demo.py, l'affichage ne doit pas lire `cycler`, qui
    appartient au thread producteur."""
    if not is_interactive:
        return
    draw_banner(
        frame,
        _INTERACTIVE_HELP_TEXT,
        _INTERACTIVE_HELP_COLOR,
        at_bottom=True,
        band_height=_INTERACTIVE_HELP_BAND_HEIGHT,
        size=16,
        bg_alpha=0.7,
    )
