import json
import random

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from nanodet.data.transform import Pipeline

TRAIN_ANN = "detection/data/04_processed/train.json"
N_CALIBRATION = 200
SEED = 42


def select_calibration_images(ann_path=TRAIN_ANN, n=N_CALIBRATION, seed=SEED):
    with open(ann_path) as f:
        data = json.load(f)
    random.seed(seed)
    sample = random.sample(data["images"], min(n, len(data["images"])))
    return [f"detection/data/03_raw/train/{img['file_name']}" for img in sample]


class NanoDetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, cfg):
        self.image_paths = image_paths
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self.input_size = cfg.data.val.input_size
        self._active_paths = image_paths
        self.index = 0

    def __len__(self):
        return len(self.image_paths)

    def set_range(self, start_index: int, end_index: int):
        """Requis pour `extra_options={"CalibStridedMinMax": N}` (quantize_static) :
        limite get_next() à une tranche, appelée plusieurs fois de suite avec des
        tranches croissantes -- permet à HistogramCalibrater.collect_data() de
        traiter les images par petits lots plutôt que de tout accumuler en RAM
        avant de calculer le moindre histogramme (cf. justifications.md, cause
        du crash RAM/disque avec `Percentile`)."""
        self._active_paths = self.image_paths[start_index:end_index]
        self.index = 0

    def get_next(self):
        if self.index >= len(self._active_paths):
            return None
        img_path = self._active_paths[self.index]
        self.index += 1

        img = cv2.imread(img_path)
        img_info = {"id": 0, "file_name": None, "height": img.shape[0], "width": img.shape[1]}
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.input_size)

        img_tensor = meta["img"].transpose(2, 0, 1)[None].astype(np.float32)
        return {"input": img_tensor}
