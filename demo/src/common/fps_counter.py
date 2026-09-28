"""Compteur de FPS glissant, affiché en overlay sur les 5 fenêtres de démo.

Mesure le framerate réel de l'ensemble du pipeline (capture + inférence +
tracking + dessin) tel qu'il tourne effectivement, frame après frame --
pas une valeur théorique déduite de benchmark.py (section 6 du rapport),
qui ne mesure que l'inférence seule, hors capture caméra/décodage vidéo et
overhead d'affichage.
"""

import time
from collections import deque

import cv2

WINDOW_SIZE = 30  # nombre de frames sur lesquelles la moyenne glisse


class FPSCounter:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        """À appeler une fois par frame traitée (capture incluse). Renvoie
        le FPS courant, moyenné sur la fenêtre glissante."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def draw_fps(frame, fps: float):
    """Affiche le FPS courant en haut à gauche -- dessiné en dernier (après
    tout le reste, y compris les bandeaux d'alerte) pour ne jamais être
    recouvert, sauf par un bandeau d'alerte plein cadre si l'un est
    affiché ce frame-ci (priorité visuelle assumée à l'alerte, 3s max)."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
