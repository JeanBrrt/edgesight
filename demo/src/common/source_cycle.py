"""Cycle entre vidéos de démo et sources interactives à la volée
(touche 'c'), pour comparer le pipeline sur des sources différentes sans
relancer le script.

Chaque script gère lui-même le nettoyage du tracker/journal au moment du
changement de source (voir `main()` dans chaque démo) : ce module ne fait
que la mécanique de capture, pas la remise à zéro de l'état de tracking.
"""

from pathlib import Path

import cv2
import numpy as np

from config import DEMO_SOURCES, DISPLAY_MAX_WIDTH
from text_render import draw_hud_line, draw_text, fit_text, text_height

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _detect_screen_size(default: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    """Résolution d'écran réelle via tkinter (même technique que
    agent/src/alerts/define_zones.py) -- retombe sur `default` si tkinter est
    indisponible (environnement headless) plutôt que de planter pour un
    simple confort d'affichage."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return default


# Marge sous la hauteur d'écran détectée -- ni winfo_screenheight() ni
# DISPLAY_MAX_WIDTH (config/demo.yaml, ne contraint QUE la largeur) ne
# tiennent compte de la barre des tâches ou de la barre de titre de la
# fenêtre vidéo. Sans ceci, une source dont la hauteur reste importante
# même une fois sa largeur limitée à DISPLAY_MAX_WIDTH (ex. la scène du
# chantier, moins large que 16:9) peut dépasser la hauteur RÉELLEMENT
# visible à l'écran -- son bas (bandeaux d'alerte/aide compris) se
# retrouve alors rendu hors champ, bien qu'il soit correctement dessiné
# sur l'image elle-même (constaté en pratique : bandeau invisible malgré
# resizeWindow, sur un écran dont la hauteur utile est plus petite que
# prévu).
_MAX_DISPLAY_HEIGHT = int(_detect_screen_size()[1] * 0.85)


def _display_scale(frame) -> float:
    """Facteur de réduction appliqué par resize_for_display (1.0 si
    l'image tient déjà à l'écran)."""
    h, w = frame.shape[:2]
    return min(1.0, DISPLAY_MAX_WIDTH / w, _MAX_DISPLAY_HEIGHT / h)


def ui_scale(frame) -> float:
    """Facteur à passer aux fonctions de text_render pour un texte dessiné
    sur l'image d'origine : compense la réduction d'affichage, pour que le
    texte ait la même taille à l'écran quelle que soit la résolution de
    la source (sinon, sur une vidéo 1920px réduite à 1280px, il perdait un
    tiers de sa taille et devenait flou)."""
    return 1.0 / _display_scale(frame)


def resize_for_display(frame, max_width: int = DISPLAY_MAX_WIDTH, max_height: int | None = None):
    """Réduit l'image pour l'affichage en contraignant à la fois la
    largeur (`max_width`, DISPLAY_MAX_WIDTH par défaut) ET la hauteur
    (`max_height`, la hauteur d'écran détectée par défaut -- voir
    _MAX_DISPLAY_HEIGHT ci-dessus) : contraindre uniquement la largeur ne
    suffit pas pour toutes les sources (voir ci-dessus). Les coordonnées
    déjà dessinées sur l'image (boîtes, bandeaux...) n'ont rien à
    recalculer -- un simple resize global préserve leur position
    relative, quel que soit le facteur d'échelle retenu. INTER_AREA :
    l'interpolation adaptée à une réduction (l'interpolation par défaut
    crénelait le texte et les traits fins)."""
    max_height = _MAX_DISPLAY_HEIGHT if max_height is None else max_height
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


CYCLE_KEY = ord("c")
# Revient au début de la vidéo courante (no-op pour une source
# interactive, qui n'a pas de "début" à rejouer -- même garde que
# loop_if_file(), voir SourceCycler ci-dessous). Chaque script de démo se
# charge, comme pour CYCLE_KEY, de remettre à zéro son propre
# tracker/journal en plus de l'appel à loop_if_file().
RESTART_KEY = ord("r")

# Préfixe reconnu dans config/demo.yaml (demo_sources) pour une source
# interactive plutôt qu'un fichier vidéo/webcam -- format :
#   "INTERACTIVE:<image_de_fond>|<png_detoure_RGBA>"
# Le '|' separe les deux chemins (les ':' de drive letter Windows, ex.
# "C:\...", rendent ':' impropre comme séparateur).
_INTERACTIVE_PREFIX = "INTERACTIVE:"


class InteractiveOverlaySource:
    """Source synthétique : une image de fond fixe + un PNG détouré (RGBA)
    qui suit la souris en direct -- pas un fichier vidéo, donc pas
    compatible avec cv2.VideoCapture. Recompose une frame à chaque appel
    de read() à partir de l'état courant (position, échelle, visibilité),
    pour que le pipeline de détection/tracking la traite comme n'importe
    quelle autre frame -- aucun changement necessaire côté scripts de
    démo au-delà de brancher on_mouse() sur la fenêtre cv2.

    Molette = redimensionne le sprite. Clic droit = affiché/masqué.
    Le déplacement suit la souris en continu, aucun clic requis.
    """

    MIN_SCALE = 0.1
    MAX_SCALE = 3.0
    SCALE_STEP = 0.1

    def __init__(self, background_path: str, sprite_path: str, display_max_width: int):
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

        # Même formule que resize_for_display() (voir chaque script de
        # démo) -- reproduite ici pour convertir les coordonnées souris
        # (espace de la fenêtre affichée) vers l'espace natif du fond.
        # Le fond est fixe tant que cette source est active, donc ce
        # facteur ne change jamais -- pas besoin de le recevoir à chaque
        # frame depuis le script appelant.
        self.display_scale = min(1.0, display_max_width / self.w) if self.w > display_max_width else 1.0

        self.cursor_native = (self.w // 2, self.h // 2)
        # Taille de base 3x plus petite que l'echelle "native" du sprite
        # (1.0), sans toucher aux bornes MIN_SCALE/MAX_SCALE -- la molette
        # garde toute son amplitude, seul le point de depart change.
        self.scale = 1.0 / 3
        # Visible par defaut : la silhouette suit la souris des l'arrivee
        # sur cette source, sans clic prealable (clic droit pour la
        # masquer/reafficher).
        self.visible = True

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_native = (int(x / self.display_scale), int(y / self.display_scale))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.visible = not self.visible
        elif event == cv2.EVENT_MOUSEWHEEL:
            # `flags` encode le delta de la molette dans ses bits 16-31,
            # en entier 16 bits SIGNE (positif = molette vers l'avant/haut)
            # -- convention Win32 WM_MOUSEWHEEL reprise telle quelle par
            # HighGUI. `cv2.getMouseWheelDelta()` fait ça nativement mais
            # n'existe pas dans tous les bindings Python d'OpenCV -- extrait
            # à la main pour ne pas dépendre de la version installée.
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

        # Recadrage si le sprite déborde du cadre (souris proche d'un
        # bord) -- sans ça, les slices ci-dessous sortiraient du tableau.
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
        pass  # rien à libérer -- pas de cv2.VideoCapture sous-jacent.


class SourceCycler:
    def __init__(self):
        self.index = 0
        self.cap = self._open(self.index)

    def _open(self, index: int):
        label, source = DEMO_SOURCES[index]
        if isinstance(source, str) and source.startswith(_INTERACTIVE_PREFIX):
            bg_path, sprite_path = source[len(_INTERACTIVE_PREFIX):].split("|")
            print(f"Source interactive : {label} -- souris: suit / molette: taille / clic droit: afficher-masquer")
            return InteractiveOverlaySource(bg_path, sprite_path, DISPLAY_MAX_WIDTH)
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
        """Clé de correspondance avec config/zones.yaml (`source:` de
        chaque scène, voir agent/src/alerts/zones.py) -- chemin absolu normalisé,
        insensible à relatif/absolu et aux séparateurs '/'/'\\' entre les
        sources par défaut (chemins relatifs, '/') et les vidéos
        personnelles (chemins absolus, cf. demo/src/config.py). Pour une
        source interactive, c'est l'image de FOND qui sert de clé (les
        zones sont calibrées dessus, pas sur la silhouette qui suit la
        souris)."""
        source = DEMO_SOURCES[self.index][1]
        if source.startswith(_INTERACTIVE_PREFIX):
            background_path = source[len(_INTERACTIVE_PREFIX):].split("|")[0]
        else:
            background_path = source
        return str((_PROJECT_ROOT / background_path).resolve())

    def next(self) -> None:
        """Passe à la source suivante du cycle (boucle vidéo 1 -> ... ->
        dernière -> vidéo 1)."""
        self.cap.release()
        self.index = (self.index + 1) % len(DEMO_SOURCES)
        self.cap = self._open(self.index)

    def read(self):
        return self.cap.read()

    def loop_if_file(self) -> None:
        """Reboucle au début du fichier vidéo courant (no-op pour une
        source interactive, jamais "terminée")."""
        if self.is_file():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        """Sans effet tant que la source active n'est pas interactive --
        toujours sûr d'appeler inconditionnellement depuis un callback cv2
        enregistré une fois pour toutes au démarrage du script."""
        if isinstance(self.cap, InteractiveOverlaySource):
            self.cap.on_mouse(event, x, y, flags)

    def release(self) -> None:
        self.cap.release()


def draw_source_label(frame, label: str):
    """Affiche la source courante juste sous le FPS (voir fps_counter.py)."""
    draw_hud_line(frame, 1, f"Source : {label}", color=(120, 230, 255), scale=ui_scale(frame))


def draw_banner(
    frame,
    text: str,
    bg_color: tuple,
    at_bottom: bool = False,
    band_height: int = 40,
    size: int = 17,
    bg_alpha: float = 0.85,
) -> None:
    """Bandeau pleine largeur (haut ou bas), fond coloré semi-transparent
    + texte blanc centré verticalement -- partagé par les scripts de démo :
    alerte (bandeau haut, 03_live_agent_demo.py ; la réponse de l'agent,
    elle, s'affiche dans la fenêtre Assistant) et rappel des contrôles
    d'une scène interactive (bandeau bas, draw_interactive_help).
    `band_height` et `size` sont exprimés en pixels à l'écran : ils sont
    mis à l'échelle comme le reste du texte (voir ui_scale). Un texte trop
    long est tronqué avec « … » plutôt que de déborder."""
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
    """Bandeau bas (même style que draw_banner) rappelant les contrôles
    souris, affiché uniquement quand la source active est une scène
    interactive -- sans lui, ses contrôles (déplacement continu, molette,
    clic droit) ne sont découvrables nulle part ailleurs à l'écran. No-op
    sur toute autre source (vidéo classique).

    Prend `is_interactive` en booléen plutôt que le `SourceCycler`
    lui-même : dans 04_live_agent_demo.py, l'affichage tourne dans un
    thread séparé de celui qui possède `cycler` (voir sa docstring) --
    `is_interactive` y est déjà calculé une fois par le thread producteur
    et transmis via la file, pour ne jamais lire l'état de `cycler`
    depuis un autre thread que celui qui le mute."""
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
