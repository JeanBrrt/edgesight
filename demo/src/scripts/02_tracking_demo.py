"""Détection et tracking 

Usage :
    uv run python demo/src/scripts/02_tracking_demo.py
Touches : 'c' changer de source, 'r' redémarrer la vidéo, 'q' quitter.
"""

import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "demo/src/common")  # modules partagés par les 3 démos
from nanodet_setup import build_postprocessor  # avant nanodet : filtre ses avertissements

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.util import cfg

sys.path.insert(0, ".")  # tracker et zones de danger : agent/src/
from agent.src.tracking.tracker import MultiClassByteTracker
from agent.src.alerts.zones import scene_for_source

from fps_counter import FPSCounter, draw_fps
from source_cycle import SourceCycler, CYCLE_KEY, RESTART_KEY, draw_source_label, draw_controls, draw_interactive_help, draw_zones, resize_for_display, ui_scale
from text_render import draw_box_label
from power import disable_power_throttling
from config import CONFIG_PATH, ONNX_PATH, INPUT_SIZE, DISABLED_CLASSES

# Une couleur fixe par identifiant, pour suivre une piste d'un coup d'œil.
COLORS = [
    (66, 135, 245), (66, 245, 111), (245, 66, 197), (245, 173, 66),
    (66, 245, 233), (197, 66, 245), (245, 66, 66), (144, 245, 66),
]


def undo_export_sigmoid(raw_output, num_classes):
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_length=10):
    """Rectangle en pointillés (absent d'OpenCV), pour les détections
    faibles."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def draw_tracked(frame, tracked: dict, class_names: list[str]):
    s = ui_scale(frame)
    for cls_idx, boxes in tracked.items():
        for x1, y1, x2, y2, score, tid, is_coasted, origin in boxes:
            # Position seulement prédite (aucune détection) : non affichée.
            # Détection réelle mais sous le seuil ("weak") : en pointillés.
            if is_coasted and origin == "predicted":
                continue
            color = COLORS[tid % len(COLORS)]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if is_coasted:
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)), round(10 * s))
                label = f"#{tid} {class_names[cls_idx]} (faible)"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)))
                label = f"#{tid} {class_names[cls_idx]} {score:.2f}"
            draw_box_label(frame, label, x1, y1, color, scale=s)
    return frame


def scene_zones(cycler) -> dict:
    """Zones de danger de la source active ({} si elle n'en a pas)."""
    scene = scene_for_source(cycler.zone_source_key)
    return scene.zones if scene else {}


WINDOW_NAME = "Suivi multi-objets [q pour quitter]"


def main():
    disable_power_throttling()
    model = build_postprocessor(CONFIG_PATH)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    tracker = MultiClassByteTracker(cfg.class_names)
    fps_counter = FPSCounter()
    cycler = SourceCycler()
    zones = scene_zones(cycler)

    # WINDOW_NORMAL + resizeWindow à chaque frame : en mode AUTOSIZE, la
    # fenêtre ne suit pas toujours un changement de taille de source.
    # Le callback souris sert à la scène interactive.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param: cycler.on_mouse(event, x, y, flags))

    print("Tracking seul — 'c' pour changer de source, 'r' pour redémarrer la vidéo, 'q' pour quitter.")

    # Tolère quelques échecs de lecture consécutifs avant d'abandonner.
    camera_fail_streak = 0
    MAX_CAMERA_FAIL_STREAK = 30

    try:
        while True:
            ret, frame = cycler.read()
            if not ret:
                if cycler.is_file():
                    cycler.loop_if_file()
                    continue
                camera_fail_streak += 1
                if camera_fail_streak >= MAX_CAMERA_FAIL_STREAK:
                    print("Lecture de la source échouée, arrêt.")
                    break
                time.sleep(0.03)
                continue
            camera_fail_streak = 0

            img_info = {"id": 0, "file_name": None, "height": frame.shape[0], "width": frame.shape[1]}
            meta = dict(img_info=img_info, raw_img=frame, img=frame)
            meta = pipeline(None, meta, INPUT_SIZE)
            meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
            meta = naive_collate([meta])
            meta["img"] = stack_batch_img(meta["img"], divisible=32)

            onnx_input = meta["img"].numpy().astype(np.float32)
            raw_output = session.run(None, {"input": onnx_input})[0]

            preds = torch.from_numpy(undo_export_sigmoid(raw_output, cfg.model.arch.head.num_classes))
            results = model.head.post_process(preds, meta)
            dets = next(iter(results.values()))
            for cls_idx, cls_name in enumerate(cfg.class_names):
                if cls_name in DISABLED_CLASSES:
                    dets[cls_idx] = []

            tracked = tracker.update(dets, timestamp=time.time())

            result_frame = draw_tracked(frame, tracked, cfg.class_names)
            draw_zones(result_frame, zones)
            draw_fps(result_frame, fps_counter.tick())
            draw_source_label(result_frame, cycler.label)
            draw_controls(result_frame)
            draw_interactive_help(result_frame, cycler.is_interactive())
            result_frame = resize_for_display(result_frame)
            cv2.resizeWindow(WINDOW_NAME, result_frame.shape[1], result_frame.shape[0])
            cv2.imshow(WINDOW_NAME, result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == CYCLE_KEY:
                cycler.next()
                zones = scene_zones(cycler)
                tracker.reset()  # nouvelle scène, anciennes pistes caduques
            if key == RESTART_KEY and cycler.is_file():
                # La vidéo repart de zéro, les pistes aussi.
                cycler.loop_if_file()
                tracker.reset()
    finally:
        cycler.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
