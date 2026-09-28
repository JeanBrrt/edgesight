import os

import psutil
import torch
import onnxruntime as ort
import numpy as np
import cv2
import json

from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ONNX Runtime sature tous les coeurs logiques par défaut (pas de limite
# de threads posée ici) -- limité au niveau OS pour garder de la marge
# thermique/réactivité (cf. quantize.py pour le détail).
_n_cores = os.cpu_count() or 2
psutil.Process().cpu_affinity(list(range(max(1, _n_cores // 2))))

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
INT8_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-u8s8.onnx"
TEST_ANN = "detection/data/04_processed/test.json"
TEST_IMG_DIR = "detection/data/03_raw/test"
# En dur, cohérent avec la résolution d'export du fichier INT8 ci-dessus.
INPUT_SIZE = (896, 896)

load_config(cfg, CONFIG_PATH)
cfg.defrost()
cfg.data.val.input_size = INPUT_SIZE
cfg.freeze()
model = build_model(
    cfg.model
)  # pas de load_model_weight : on n'utilise que model.head.post_process
pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

session = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])
NUM_CLASSES = cfg.model.arch.head.num_classes


def undo_export_sigmoid(raw_output, num_classes):
    """L'export ONNX (_forward_onnx) applique déjà un sigmoid sur la partie
    classification ; post_process en applique un second en interne. Sans cette
    correction, le sigmoid est appliqué deux fois, ce qui compresse tous les
    scores vers [0.5, 0.7] et laisse passer des centaines de faux candidats
    au NMS (bug découvert et diagnostiqué en marge de E1, cf. justifications.md)."""
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)

coco_gt = COCO(TEST_ANN)
img_ids = sorted(coco_gt.imgs.keys())
img_infos = coco_gt.loadImgs(img_ids)

det_results = {}
for img_info in img_infos:
    img_path = f"{TEST_IMG_DIR}/{img_info['file_name']}"
    img = cv2.imread(img_path)

    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)

    onnx_input = meta["img"].numpy().astype(np.float32)
    raw_output = session.run(None, {"input": onnx_input})[0]

    preds = torch.from_numpy(undo_export_sigmoid(raw_output, NUM_CLASSES))
    results = model.head.post_process(preds, meta)
    det_results.update(results)


def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


cat_ids = sorted(coco_gt.getCatIds())

json_results = []
for image_id, dets in det_results.items():
    for label, bboxes in dets.items():
        category_id = cat_ids[label]
        for bbox in bboxes:
            json_results.append(
                dict(
                    image_id=int(image_id),
                    category_id=int(category_id),
                    bbox=xyxy2xywh(bbox),
                    score=float(bbox[4]),
                )
            )

with open("detection/data/04_processed/test_predictions_int8.json", "w") as f:
    json.dump(json_results, f)

coco_dt = coco_gt.loadRes("detection/data/04_processed/test_predictions_int8.json")
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
