"""Affiche 4 images au hasard de test.json : détections du modèle INT8
(896px) et vérité terrain COCO.

Touches :
    'r' -- 4 nouvelles images
    'g' -- afficher/masquer la vérité terrain
    'q' -- quitter

Lancer depuis la racine du projet :
    uv run python detection/src/04_evaluate/visualize_test_predictions.py

Pour enregistrer une grille en PNG sans ouvrir de fenêtre (images choisies
par leur id COCO, ou au hasard si --ids est omis) :
    uv run python detection/src/04_evaluate/visualize_test_predictions.py \\
        --save docs/figures/predictions_reussites.jpg --ids 160864 377588 17207 78823
"""

import argparse
import random

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
# INPUT_SIZE doit correspondre à la résolution d'export du modèle ONNX,
# quelle que soit celle de CONFIG_PATH.
ONNX_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-u8s8.onnx"
INPUT_SIZE = (896, 896)
TEST_ANN = "detection/data/04_processed/test.json"
TEST_IMG_DIR = "detection/data/03_raw/test"

SAMPLE_SIZE = 4
GRID_COLS = 2
CELL_WIDTH, CELL_HEIGHT = 640, 480
SCORE_THRESHOLD = 0.5

DET_COLOR = (66, 135, 245)  # détections du modèle
GT_COLOR = (66, 245, 111)  # vérité terrain COCO

REROLL_KEY = ord("r")
TOGGLE_GT_KEY = ord("g")
QUIT_KEY = ord("q")

WINDOW_NAME = "Détections vs vérité terrain (test set) -- 'r' reroll, 'g' GT, 'q' quitter"


def undo_export_sigmoid(raw_output, num_classes):
    """Annule le sigmoid de l'export ONNX, que post_process réapplique."""
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)


def run_detection(session, model, pipeline, img_path, num_classes):
    """Renvoie (dets, image), dets en coordonnées de l'image d'origine."""
    img = cv2.imread(img_path)
    img_info = {"id": 0, "file_name": None, "height": img.shape[0], "width": img.shape[1]}
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, INPUT_SIZE)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)

    onnx_input = meta["img"].numpy().astype(np.float32)
    raw_output = session.run(None, {"input": onnx_input})[0]
    preds = torch.from_numpy(undo_export_sigmoid(raw_output, num_classes))
    results = model.head.post_process(preds, meta)
    dets = next(iter(results.values()))
    return dets, img


def draw_detections(frame, dets: dict, class_names: list[str]):
    for cls_idx, boxes in dets.items():
        for x1, y1, x2, y2, score in boxes:
            if score < SCORE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), DET_COLOR, 2)
            label = f"{class_names[cls_idx]} {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DET_COLOR, 2)
    return frame


def draw_ground_truth(frame, coco_gt: COCO, img_id: int, class_names: list[str], cat_id_to_idx: dict):
    # Sans les boîtes de foule (iscrowd), ignorées par l'évaluation COCO.
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id], iscrowd=False)):
        x, y, w, h = map(int, ann["bbox"])
        cls_idx = cat_id_to_idx[ann["category_id"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), GT_COLOR, 2)
        cv2.putText(frame, class_names[cls_idx], (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
    return frame


def fit_cell(img):
    """Redimensionne sans déformer, bandes noires pour compléter."""
    h, w = img.shape[:2]
    scale = min(CELL_WIDTH / w, CELL_HEIGHT / h)
    new_w, new_h = int(w * scale), int(h * scale)
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    x, y = (CELL_WIDTH - new_w) // 2, (CELL_HEIGHT - new_h) // 2
    cell[y : y + new_h, x : x + new_w] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cell


def build_grid(cells: list):
    resized = [fit_cell(cell) for cell in cells]
    rows = [np.hstack(resized[i : i + GRID_COLS]) for i in range(0, len(resized), GRID_COLS)]
    return np.vstack(rows)


def compute_sample(session, model, pipeline, coco_gt: COCO, sample_ids: list, num_classes: int):
    """Inférence sur l'échantillon. Le résultat est gardé en cache : seul
    'r' relance l'inférence, pas 'g'."""
    sample = []
    for img_id in sample_ids:
        file_name = coco_gt.loadImgs([img_id])[0]["file_name"]
        img_path = f"{TEST_IMG_DIR}/{file_name}"
        dets, frame = run_detection(session, model, pipeline, img_path, num_classes)
        sample.append((frame, dets, img_id, file_name))
    return sample


def render(sample: list, coco_gt: COCO, class_names: list[str], cat_id_to_idx: dict, show_gt: bool):
    cells = []
    for frame, dets, img_id, file_name in sample:
        cell = frame.copy()  # ne pas dessiner sur l'image en cache
        cell = draw_detections(cell, dets, class_names)
        if show_gt:
            cell = draw_ground_truth(cell, coco_gt, img_id, class_names, cat_id_to_idx)
        cv2.putText(cell, file_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(cell, file_name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cells.append(cell)
    return build_grid(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="enregistre la grille dans ce PNG au lieu d'ouvrir une fenêtre")
    parser.add_argument("--ids", type=int, nargs="+", help="ids COCO des images (défaut : au hasard)")
    args = parser.parse_args()

    load_config(cfg, CONFIG_PATH)
    cfg.defrost()
    cfg.data.val.input_size = INPUT_SIZE
    cfg.freeze()
    model = build_model(cfg.model)  # sans poids, seul post_process sert
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    num_classes = cfg.model.arch.head.num_classes
    class_names = cfg.class_names

    coco_gt = COCO(TEST_ANN)
    img_ids = sorted(coco_gt.imgs.keys())
    cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(sorted(coco_gt.getCatIds()))}

    show_gt = True
    sample_ids = args.ids or random.sample(img_ids, SAMPLE_SIZE)
    sample = compute_sample(session, model, pipeline, coco_gt, sample_ids, num_classes)

    if args.save:
        cv2.imwrite(args.save, render(sample, coco_gt, class_names, cat_id_to_idx, show_gt))
        print(f"{args.save} écrit (images {sample_ids})")
        return

    print(f"{len(img_ids)} images dans test.json -- 'r' nouvel échantillon, 'g' vérité terrain, 'q' quitter.")

    # Fenêtre redimensionnable à la souris.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            grid = render(sample, coco_gt, class_names, cat_id_to_idx, show_gt)
            cv2.imshow(WINDOW_NAME, grid)

            key = cv2.waitKey(0) & 0xFF
            if key == QUIT_KEY:
                break
            if key == REROLL_KEY:
                sample = compute_sample(
                    session, model, pipeline, coco_gt, random.sample(img_ids, SAMPLE_SIZE), num_classes
                )
            elif key == TOGGLE_GT_KEY:
                show_gt = not show_gt
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
