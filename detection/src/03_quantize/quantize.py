import os

import psutil
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
import onnxruntime as ort
import numpy as np
from nanodet.util import cfg, load_config
from calibration_ready import NanoDetCalibrationDataReader, select_calibration_images

# quantize_static utilise tous les coeurs et ne se configure pas : on
# limite le processus à la moitié pour garder la machine utilisable.
_n_cores = os.cpu_count() or 2
psutil.Process().cpu_affinity(list(range(max(1, _n_cores // 2))))

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
FP32_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-fp32.onnx"
PREPROCESSED_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-preprocessed.onnx"
INT8_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-u8s8.onnx"
# Doit correspondre à la résolution d'export du modèle fp32.
INPUT_SIZE = (896, 896)

quant_pre_process(FP32_ONNX, PREPROCESSED_ONNX)
print("Pré-traitement (shape inference + optimisation) terminé :", PREPROCESSED_ONNX)

load_config(cfg, CONFIG_PATH)
cfg.defrost()
cfg.data.val.input_size = INPUT_SIZE
cfg.freeze()
image_paths = select_calibration_images()
calibration_reader = NanoDetCalibrationDataReader(image_paths, cfg)

quantize_static(
    model_input=PREPROCESSED_ONNX,
    model_output=INT8_ONNX,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.Percentile,
    # Calibration par tranches de 10 images : sans ça, les 200 images sont
    # gardées en RAM (~60 Go de pic à 512px)
    extra_options={"CalibStridedMinMax": 10},
)
print("Quantification terminée :", INT8_ONNX)

onnx.checker.check_model(INT8_ONNX)
print("Structure valide")

session_int8 = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 896, 896).astype(np.float32)
output = session_int8.run(None, {"input": dummy})[0]
print("Shape de sortie :", output.shape)
