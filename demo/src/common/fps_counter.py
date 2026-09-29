"""FPS réel de tout le pipeline (lecture, inférence, suivi, affichage),
moyenné sur les dernières images"""

import time
from collections import deque

from source_cycle import ui_scale
from text_render import draw_hud_line

WINDOW_SIZE = 30  # nombre d'images de la moyenne glissante


class FPSCounter:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """À appeler une fois par image ; renvoie le FPS moyen."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def draw_fps(frame, fps: float):
    """Affiche le FPS en haut à gauche."""
    draw_hud_line(frame, 0, f"FPS  {fps:.1f}", color=(120, 255, 120), bold=True, scale=ui_scale(frame))
