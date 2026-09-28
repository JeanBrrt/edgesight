"""Vérification visuelle de C1 (tracking) seul, en direct sur les vidéos
de démo — sans journal d'événements (C2) ni alerte (D3).

Affiche les détections avec leur tracker_id persistant : permet de juger
la stabilité du tracker (ID qui ne change pas quand une piste est
momentanément perdue, ré-association après occlusion brève) isolément du
reste du pipeline.

Lancer depuis la racine du projet :
    uv run python demo/src/scripts/02_tracking_demo.py
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

sys.path.insert(0, ".")  # tracker.py vit dans agent/src/ (C1 est un
# composant du système, pas du glue code de démo)
from agent.src.tracking.tracker import MultiClassByteTracker

sys.path.insert(0, "demo/src/common")  # config.py/fps_counter.py/source_cycle.py
# vivent à part des scripts, partagés par les 3 (voir demo/src/common/)
from fps_counter import FPSCounter, draw_fps
from source_cycle import SourceCycler, CYCLE_KEY, RESTART_KEY, draw_source_label, draw_interactive_help, resize_for_display
from config import CONFIG_PATH, ONNX_PATH, INPUT_SIZE, DISABLED_CLASSES

# Une couleur stable par tracker_id (modulo), pour repérer visuellement
# la persistance d'une piste d'un coup d'œil.
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
    """Pas de primitive rectangle pointille native dans OpenCV -- dessine
    quatre bords en segments. Utilise pour distinguer visuellement une
    boite 'coasted' (position extrapolee par le filtre de Kalman du
    tracker, pas une vraie detection ce frame-ci) d'une boite confirmee."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def draw_tracked(frame, tracked: dict, class_names: list[str]):
    for cls_idx, boxes in tracked.items():
        for x1, y1, x2, y2, score, tid, is_coasted, origin in boxes:
            # "predicted" (extrapolation Kalman pure, aucune detection ce
            # frame-ci) n'est jamais affiche : une piste "en roue libre" ne
            # doit pas laisser croire a une detection reelle sur l'ecran de
            # demo. "weak" (detection reelle mais rejetee par l'hysteresis,
            # score trop faible) reste affichee en pointille -- cas
            # different, une detection a bien eu lieu ce frame-ci.
            if is_coasted and origin == "predicted":
                continue
            color = COLORS[tid % len(COLORS)]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if is_coasted:
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {class_names[cls_idx]} (faible)"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {class_names[cls_idx]} {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


WINDOW_NAME = "C1 - tracking seul [q pour quitter]"


def main():
    load_config(cfg, CONFIG_PATH)
    model = build_model(cfg.model)  # config seule, pour post_process
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    tracker = MultiClassByteTracker(cfg.class_names)
    fps_counter = FPSCounter()
    cycler = SourceCycler()

    # Fenêtre créée explicitement (plutôt que par le premier cv2.imshow)
    # pour pouvoir y attacher un callback souris -- sans effet tant que la
    # source active n'est pas interactive (voir SourceCycler.on_mouse).
    # WINDOW_NORMAL + resizeWindow explicite à chaque frame (plus bas) :
    # WINDOW_AUTOSIZE (le défaut) ne redimensionne pas toujours la fenêtre
    # de façon fiable en changeant de source vers une image de dimensions
    # différentes -- un bandeau bas ajouté après coup s'est déjà retrouvé
    # rogné hors de la fenêtre dans ce cas, alors qu'il était bien dessiné
    # sur l'image elle-même.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param: cycler.on_mouse(event, x, y, flags))

    print("Tracking seul — 'c' pour changer de source, 'r' pour redémarrer la vidéo, 'q' pour quitter.")

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

            tracked = tracker.update(dets, timestamp=time.time())

            result_frame = draw_tracked(frame, tracked, cfg.class_names)
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
                cycler.next()
                tracker.reset()  # nouvelle scène : les anciennes pistes n'ont plus de sens
            if key == RESTART_KEY and cycler.is_file():
                # No-op pour source interactive (pas de "début" à
                # rejouer) -- mêmes resets que CYCLE_KEY, la vidéo repart
                # de zéro donc les anciennes pistes aussi.
                cycler.loop_if_file()
                tracker.reset()
    finally:
        cycler.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
