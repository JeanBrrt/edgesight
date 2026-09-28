import torch

from nanodet.util import cfg, load_config, load_model_weight, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import cv2
import json

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
MODEL_PATH = "workspace/nanodet-plus-m-1.5x_896-person-car/model_best/nanodet_model_best.pth"
TEST_ANN = "detection/data/04_processed/test.json"
TEST_IMG_DIR = "detection/data/03_raw/test"

load_config(cfg, CONFIG_PATH)
logger = Logger(0, use_tensorboard=False)
model = build_model(cfg.model)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
load_model_weight(model, ckpt, logger)
model = model.to("cuda:0").eval()
pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

coco_gt = COCO(TEST_ANN)
img_ids = sorted(coco_gt.imgs.keys())
img_infos = coco_gt.loadImgs(img_ids)

det_results = {}
for img_info in img_infos:
    img_path = f"{TEST_IMG_DIR}/{img_info['file_name']}"
    img = cv2.imread(img_path)

    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to("cuda:0")
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)

    with torch.inference_mode():
        results = model.inference(meta)

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

with open("detection/data/04_processed/test_predictions.json", "w") as f:
    json.dump(json_results, f)

coco_dt = coco_gt.loadRes("detection/data/04_processed/test_predictions.json")
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
