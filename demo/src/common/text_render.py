"""Texte lisible sur les images OpenCV, via Pillow.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Première police trouvée, par ordre de préférence : Segoe UI (Windows),
# DejaVu Sans (Linux, et fournie avec matplotlib), Arial.
_FONT_FILES = {
    False: ["segoeui.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"],
    True: ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"],
}


def _font_dirs() -> list[Path]:
    dirs = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental")]
    try:
        import matplotlib

        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except ImportError:
        pass
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=None)
def _font_path(bold: bool) -> str | None:
    for name in _FONT_FILES[bold]:
        for directory in _font_dirs():
            matches = list(directory.rglob(name)) if directory.name == "fonts" else [directory / name]
            for path in matches:
                if path.is_file():
                    return str(path)
    return None


@lru_cache(maxsize=64)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    path = _font_path(bold)
    if path is None:
        # Police intégrée à Pillow, faute de police système.
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size)


def _bgr_to_rgb(color: tuple) -> tuple:
    return (int(color[2]), int(color[1]), int(color[0]))


def text_color_for(bg_bgr: tuple) -> tuple:
    """Blanc ou noir selon la luminosité du fond (BGR)."""
    b, g, r = bg_bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if luminance > 150 else (255, 255, 255)


@lru_cache(maxsize=1024)
def _render(
    text: str, size: int, bold: bool, color: tuple, bg: tuple | None,
    bg_alpha: float, pad_x: int, pad_y: int, radius: int, outline: int,
) -> np.ndarray:
    """Vignette RGBA du texte (fond arrondi optionnel), mise en cache."""
    font = _font(size, bold)
    ascent, descent = font.getmetrics()
    text_w = int(np.ceil(font.getlength(text)))
    w = text_w + 2 * pad_x + 2 * outline
    h = ascent + descent + 2 * pad_y + 2 * outline
    patch = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    if bg is not None:
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius,
                               fill=(*_bgr_to_rgb(bg), int(255 * bg_alpha)))
    draw.text(
        (pad_x + outline, pad_y + outline), text, font=font, fill=(*_bgr_to_rgb(color), 255),
        stroke_width=outline, stroke_fill=(0, 0, 0, 255),
    )
    array = np.asarray(patch)
    array.setflags(write=False)
    return array


def _blend(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Fusionne la vignette dans `frame` (sur place), coupée aux bords."""
    fh, fw = frame.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + pw, fw), min(y + ph, fh)
    if x0 >= x1 or y0 >= y1:
        return
    region = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    rgb_as_bgr = region[:, :, 2::-1].astype(np.float32)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (alpha * rgb_as_bgr + (1.0 - alpha) * roi).astype(np.uint8)


def draw_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    size: int = 16,
    color: tuple = (255, 255, 255),
    bg: tuple | None = (20, 20, 20),
    bg_alpha: float = 0.65,
    bold: bool = False,
    anchor: str = "top-left",
    pad: tuple[int, int] = (8, 4),
    radius: int = 6,
    outline: int = 0,
    scale: float = 1.0,
) -> tuple[int, int]:
    """Dessine `text` dans `frame` (sur place) et renvoie la taille de
    l'étiquette (largeur, hauteur).

    `anchor` : "top-left" (sous `org`) ou "bottom-left" (au-dessus).
    `bg=None` : texte sans fond, `outline` pour un liseré noir. Couleurs
    en BGR."""
    s = max(scale, 0.1)
    patch = _render(
        text, max(int(round(size * s)), 6), bold, tuple(color),
        tuple(bg) if bg is not None else None, bg_alpha,
        int(round(pad[0] * s)), int(round(pad[1] * s)), int(round(radius * s)),
        int(round(outline * s)),
    )
    ph, pw = patch.shape[:2]
    x, y = int(org[0]), int(org[1])
    if anchor == "bottom-left":
        y -= ph
    _blend(frame, patch, x, y)
    return pw, ph


HUD_SIZE = 16
_HUD_MARGIN = 10
_HUD_GAP = 6
_HUD_PAD = (8, 4)


def draw_hud_line(frame: np.ndarray, line: int, text: str, *, color: tuple = (255, 255, 255),
                  bold: bool = False, scale: float = 1.0) -> None:
    """Étiquette du coin haut-gauche, sur la ligne `line` (0 : FPS,
    1 : source)."""
    ascent, descent = _font(max(int(round(HUD_SIZE * scale)), 6), False).getmetrics()
    line_h = ascent + descent + 2 * int(round(_HUD_PAD[1] * scale))
    y = int(round(_HUD_MARGIN * scale)) + line * (line_h + int(round(_HUD_GAP * scale)))
    draw_text(frame, text, (int(round(_HUD_MARGIN * scale)), y), size=HUD_SIZE, color=color,
              bold=bold, pad=_HUD_PAD, scale=scale)


def draw_box_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple, *,
                   size: int = 14, scale: float = 1.0) -> None:
    """Étiquette d'une boîte ou d'une zone, sur fond de sa couleur, au-dessus
    de (x, y), ou en dessous si elle touche le haut de l'image."""
    label_h = text_height(size, scale=scale) + 2 * int(round(3 * scale))
    anchor = "bottom-left" if y - label_h >= 0 else "top-left"
    draw_text(frame, text, (x, y), size=size, color=text_color_for(color), bg=color, bg_alpha=0.9,
              pad=(6, 3), radius=4, anchor=anchor, scale=scale)


def text_width(text: str, size: int, bold: bool = False, scale: float = 1.0) -> int:
    """Largeur en pixels du texte seul (sans marges), à cette taille."""
    return int(np.ceil(_font(max(int(round(size * scale)), 6), bold).getlength(text)))


def text_height(size: int, bold: bool = False, scale: float = 1.0) -> int:
    """Hauteur de ligne en pixels (sans marges), à cette taille."""
    ascent, descent = _font(max(int(round(size * scale)), 6), bold).getmetrics()
    return ascent + descent


def fit_text(text: str, max_width: int, size: int, bold: bool = False, scale: float = 1.0) -> str:
    """Tronque `text` avec « … » pour qu'il tienne dans `max_width` pixels."""
    if text_width(text, size, bold, scale) <= max_width:
        return text
    while text and text_width(text + "…", size, bold, scale) > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


if __name__ == "__main__":  # aperçu rapide : python text_render.py [sortie.png]
    img = np.full((180, 640, 3), 90, np.uint8)
    draw_text(img, "Zone « entrée » : 2 point(s) — encore 1", (12, 12), size=20)
    draw_text(img, "#3 person 0.87", (12, 90), size=16, bg=(245, 135, 66), color=(255, 255, 255))
    draw_text(img, "FPS 24.8", (12, 130), size=16, bold=True, color=(120, 255, 120))
    out = sys.argv[1] if len(sys.argv) > 1 else "text_render_preview.png"
    import cv2

    cv2.imwrite(out, img)
    print("police :", _font_path(False), "->", out)
