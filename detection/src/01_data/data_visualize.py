PROCESSED_DIR = "detection/data/04_processed"
RAW_DIR = "detection/data/03_raw"
OUTPUT_DIR = "detection/data/05_checks"
N_SAMPLES = 20
SEED = 42

import json
import os
import random as rd
import cv2
import numpy as np


def load_split(name: str) -> dict:
    with open(f"{PROCESSED_DIR}/{name}.json") as f:
        data = json.load(f)

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    return data["images"], anns_by_image, data["categories"]


def draw_annotations(img_path: str, annotations: list[dict], categories: dict) -> "np.array":
    img = cv2.imread(img_path)
    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        label = categories[ann["category_id"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def visualize_split(name: str, n_samples: int):
    images, anns_by_image, categories_list = load_split(name)
    categories = {c["id"]: c["name"] for c in categories_list}

    rd.seed(SEED)
    sample = rd.sample(images, min(n_samples, len(images)))

    out_dir = f"{OUTPUT_DIR}/{name}"
    os.makedirs(out_dir, exist_ok=True)

    for img_info in sample:
        img_path = f"{RAW_DIR}/{name}/{img_info['file_name']}"
        annotated = draw_annotations(img_path, anns_by_image.get(img_info["id"], []), categories)
        cv2.imwrite(f"{out_dir}/{img_info['file_name']}", annotated)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        visualize_split(split, N_SAMPLES)
        print(f"{split}: {N_SAMPLES} images annotées écrites dans {OUTPUT_DIR}/{split}")
