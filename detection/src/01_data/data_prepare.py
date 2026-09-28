FILTERED_DIR = "detection/data/02_filtered"
RAW_DIR = "detection/data/03_raw"
PROCESSED_DIR = "detection/data/04_processed"
CLASS_NAMES = ["person", "car"]
TEST_SIZE = 0.3
SEED = 42

import os
import json
from data_filter import load_coco
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import shutil


def build_category_map(coco: COCO, class_names: list[str]) -> dict:
    return {coco.getCatIds(catNms=[name])[0]: i for i, name in enumerate(class_names)}


def filter_downloaded(images_info: list[dict], split_raw_dir: str) -> list[dict]:
    kept = [
        img_info
        for img_info in images_info
        if os.path.exists(f"{split_raw_dir}/{img_info['file_name']}")
    ]
    dropped = len(images_info) - len(kept)
    if dropped:
        print(f"{dropped} images absentes du dique, ignorées")
    return kept


def build_annotations(coco: COCO, images_info: list[dict], cat_map: dict) -> list[dict]:
    img_ids = [img_info["id"] for img_info in images_info]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids, catIds=list(cat_map.keys())))
    for ann in anns:
        ann["category_id"] = cat_map[ann["category_id"]]
    return anns


def save_split(images_info: list[dict], annotations: list[dict], name: str, class_names: list[str]):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    payload = {
        "images": images_info,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }
    with open(f"{PROCESSED_DIR}/{name}.json", "w") as f:
        json.dump(payload, f)
    print(f"{name}: {len(images_info)} images, {len(annotations)} annotations")


def move_test_images(test_images: list[dict]):
    dest_dir = f"{RAW_DIR}/test"
    os.makedirs(dest_dir, exist_ok=True)
    for img_info in test_images:
        src = f"{RAW_DIR}/val/{img_info['file_name']}"
        dst = f"{dest_dir}/{img_info['file_name']}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)


if __name__ == "__main__":
    coco_train = load_coco("train")
    cat_map = build_category_map(coco_train, CLASS_NAMES)

    with open(f"{FILTERED_DIR}/train_selection.json") as f:
        train_images = filter_downloaded(json.load(f), f"{RAW_DIR}/train")
    save_split(
        train_images, build_annotations(coco_train, train_images, cat_map), "train", CLASS_NAMES
    )

    coco_val = load_coco("val")
    with open(f"{FILTERED_DIR}/val_selection.json") as f:
        val_all = filter_downloaded(json.load(f), f"{RAW_DIR}/val")

    val_images, test_images = train_test_split(val_all, test_size=TEST_SIZE, random_state=SEED)
    move_test_images(test_images)
    save_split(val_images, build_annotations(coco_val, val_images, cat_map), "val", CLASS_NAMES)
    save_split(test_images, build_annotations(coco_val, test_images, cat_map), "test", CLASS_NAMES)
