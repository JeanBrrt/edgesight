import os
import json
from pycocotools.coco import COCO
import random as rd

ANNOTATIONS_DIR = "detection/data/01_annotations"
OUTPUT_DIR = "detection/data/02_filtered"
CLASS_NAMES = ["person", "car"]
SEED = 42

# Cibles par bucket (B2.7) : "car" est systématiquement le bucket goulot
# d'étranglement (seulement 3732 images car-seul et 8519 both dans
# train2017, contre 55596 person-seul) — un N_PER_CLASS unique appliqué
# aux 3 buckets ne fait qu'ajouter du person-seul et laisse ~4500 images
# "both" (donc porteuses de voitures) inutilisées. On prend tout ce qui
# existe pour les buckets liés à "car" (min() les plafonne de toute façon
# à la taille réelle du pool, y compris pour les petits pools de val2017)
# et on plafonne modérément person-seul pour ne pas retomber dans le même
# travers.
BUCKET_TARGETS = {
    "only_person": 5000,
    "only_car": 100_000,  # = tout le pool disponible
    "both": 100_000,      # = tout le pool disponible, plus grosse source de "car"
}


def load_coco(split: str) -> COCO:
    path = f"{ANNOTATIONS_DIR}/instances_{split}2017.json"
    return COCO(path)


def get_class_image_ids(coco: COCO, class_names: list[str]) -> dict:
    result = {}
    for name in class_names:
        cat_id = coco.getCatIds(catNms=[name])[0]
        result[name] = set(coco.getImgIds(catIds=[cat_id]))
    return result


def balanced_sample(class_ids: dict, bucket_targets: dict, seed: int) -> set:
    rd.seed(seed)
    person_ids, car_ids = class_ids["person"], class_ids["car"]

    only_person = list(person_ids - car_ids)
    only_car = list(car_ids - person_ids)
    both = list(person_ids & car_ids)

    sampled_person = rd.sample(population=only_person, k=min(bucket_targets["only_person"], len(only_person)))
    sampled_car = rd.sample(population=only_car, k=min(bucket_targets["only_car"], len(only_car)))
    sampled_both = rd.sample(population=both, k=min(bucket_targets["both"], len(both)))

    return set(sampled_person) | set(sampled_car) | set(sampled_both)


def save_selection(coco: COCO, img_ids: set, split: str, output_dir: str):
    imgs_info = coco.loadImgs(list(img_ids))
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{split}_selection.json", "w") as f:
        json.dump(imgs_info, f)


if __name__ == "__main__":
    for split in ["train", "val"]:
        coco = load_coco(split)
        class_ids = get_class_image_ids(coco, CLASS_NAMES)
        selected = balanced_sample(class_ids, BUCKET_TARGETS, SEED)
        save_selection(coco, selected, split, OUTPUT_DIR)
        print(f"{split}: {len(selected)} images sélectionnées")
