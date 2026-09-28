"""Démo minimale : détection brute (ONNX INT8), sans tracker ni
journal ni agent — juste le modèle affiché tel quel, pour juger sa
sortie indépendamment de tout le reste du pipeline (C1/C2/D3/D1/D2).

Lancer depuis la racine du projet :
    uv run python demo/src/scripts/01_detection_demo.py
Appuyer sur 'q' pour quitter.
"""

import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

sys.path.insert(0, "demo/src/common")  # config.py/fps_counter.py/source_cycle.py
# vivent à part des scripts, partagés par les 3 (voir demo/src/common/)
from fps_counter import FPSCounter, draw_fps
from source_cycle import SourceCycler, CYCLE_KEY, RESTART_KEY, draw_source_label, draw_interactive_help, resize_for_display

from config import (
    CONFIG_PATH,
    ONNX_PATH,
    INPUT_SIZE,
    RAW_SCORE_THRESHOLD,
    DISABLED_CLASSES,
)

COLOR = (66, 135, 245)


def undo_export_sigmoid(raw_output, num_classes):
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)


def draw_detections(frame, dets: dict, class_names: list[str]):
    for cls_idx, boxes in dets.items():
        for x1, y1, x2, y2, score in boxes:
            if score < RAW_SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
            label = f"{class_names[cls_idx]} {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)
    return frame


WINDOW_NAME = "Détection brute (sans tracker) [q pour quitter]"


def main():
    load_config(cfg, CONFIG_PATH)
    model = build_model(cfg.model)  # config seule, pour post_process
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    fps_counter = FPSCounter()
    cycler = SourceCycler()

    # WINDOW_NORMAL + resizeWindow explicite à chaque frame (plus bas) :
    # WINDOW_AUTOSIZE (le défaut) ne redimensionne pas toujours la fenêtre
    # de façon fiable en changeant de source vers une image de dimensions
    # différentes -- un bandeau bas ajouté après coup s'est déjà retrouvé
    # rogné hors de la fenêtre dans ce cas, alors qu'il était bien dessiné
    # sur l'image elle-même.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param: cycler.on_mouse(event, x, y, flags))

    print(f"Détection brute, seuil={RAW_SCORE_THRESHOLD} — 'c' pour changer de source, 'r' pour redémarrer la vidéo, 'q' pour quitter.")

    # Tolère quelques échecs de lecture consécutifs (hoquet transitoire)
    # avant d'abandonner, plutôt que de s'arrêter sur le premier raté.
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
                    print("Lecture caméra échouée, arrêt.")
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

            result_frame = draw_detections(frame, dets, cfg.class_names)
            draw_fps(result_frame, fps_counter.tick())
            draw_source_label(result_frame, cycler.label)
            draw_interactive_help(result_frame, cycler.is_interactive())
            result_frame = resize_for_display(result_frame)
            cv2.resizeWindow(WINDOW_NAME, result_frame.shape[1], result_frame.shape[0])
            cv2.imshow(WINDOW_NAME, result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == CYCLE_KEY:
                cycler.next()  # pas de tracker/journal à nettoyer ici
            if key == RESTART_KEY:
                cycler.loop_if_file()  # no-op pour source interactive
    finally:
        cycler.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
