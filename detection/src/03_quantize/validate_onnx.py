import torch
import onnxruntime as ort
import numpy as np
import cv2

from nanodet.util import cfg, load_config, load_model_weight, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
# Génération 4 (896px) toujours EN COURS au moment de ce passage toolchain
# (epoch 37+/50) -- model_best avance donc encore à chaque validation
# (epoch multiple de 4). Chemin canonique quand même : model_best reflète
# le meilleur checkpoint connu à l'instant de l'export, pas un run figé.
MODEL_PATH = "workspace/nanodet-plus-m-1.5x_896-person-car/model_best/nanodet_model_best.pth"
ONNX_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-fp32.onnx"
# En dur, indépendant de cfg.data.val.input_size : ONNX_PATH est figé à
# 896px depuis son export, explicite plutôt qu'implicite.
INPUT_SIZE = (896, 896)
TEST_IMAGES = [
    "detection/data/03_raw/test/000000186873.jpg",
    "detection/data/03_raw/test/000000013177.jpg",
    "detection/data/03_raw/test/000000573258.jpg",
]

load_config(cfg, CONFIG_PATH)
cfg.defrost()
cfg.data.val.input_size = INPUT_SIZE
cfg.freeze()
logger = Logger(0, use_tensorboard=False)
model = build_model(cfg.model)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
load_model_weight(model, ckpt, logger)
model.eval()  # CPU volontairement, pour une comparaison propre sans bruit GPU/CPU

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)


def preprocess(img_path):
    img = cv2.imread(img_path)
    img_info = {"id": 0, "file_name": None, "height": img.shape[0], "width": img.shape[1]}
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)
    return meta["img"]


def pytorch_raw_output(img_tensor):
    with torch.no_grad():
        raw = model(img_tensor)  # chemin normal, PAS _forward_onnx
    cls, reg = raw.split(
        [cfg.model.arch.head.num_classes, raw.shape[-1] - cfg.model.arch.head.num_classes], dim=-1
    )
    cls = cls.sigmoid()  # reproduit manuellement ce que fait _forward_onnx
    return torch.cat([cls, reg], dim=-1).numpy()


def onnx_output(img_tensor):
    return session.run(None, {"input": img_tensor.numpy()})[0]


for path in TEST_IMAGES:
    img_tensor = preprocess(path)
    out_pt = pytorch_raw_output(img_tensor)
    out_onnx = onnx_output(img_tensor)

    max_diff = np.abs(out_pt - out_onnx).max()
    mean_diff = np.abs(out_pt - out_onnx).mean()
    close = np.allclose(out_pt, out_onnx, rtol=1e-3, atol=1e-5)

    print(f"{path}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  allclose={close}")
