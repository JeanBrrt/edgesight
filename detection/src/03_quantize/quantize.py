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

# La session de calibration créée en interne par onnxruntime.quantization
# ne limite pas ses threads (SessionOptions() par défaut, pas configurable
# via l'API publique quantize_static) -- elle sature donc tous les coeurs
# logiques disponibles. Limité ici au niveau OS (fonctionne quelle que soit
# la bibliothèque), pour garder de la marge thermique/réactivité pendant
# le calcul plutôt que saturer la machine.
_n_cores = os.cpu_count() or 2
psutil.Process().cpu_affinity(list(range(max(1, _n_cores // 2))))

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
FP32_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-fp32.onnx"
PREPROCESSED_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-preprocessed.onnx"
INT8_ONNX = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-u8s8.onnx"
# En dur, cohérent avec la résolution d'export du fp32 ci-dessus (explicite
# plutôt que de dépendre de cfg.data.val.input_size).
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
    # CalibStridedMinMax : traite le CalibrationDataReader par tranches de 10
    # images plutôt que tout accumuler en RAM avant de calculer le moindre
    # histogramme (HistogramCalibrater.collect_data() sinon accumule les 200
    # images d'un coup -- à 512px déjà ~60 Go de pic mesuré ; à 768px
    # (2,25x plus de pixels/image) le pic sans ce fix serait encore pire.
    # Cause du crash RAM/disque observé -- cf. docs/justifications.md.
    # Nécessite que NanoDetCalibrationDataReader implémente
    # __len__/set_range (fait). Percentile+stride mesuré meilleur que
    # MinMax/MinMax+moving_average et Entropy+stride sur le modèle 512px
    # (0,400 vs 0,389/0,389/0,392) -- retenu par défaut ici aussi.
    extra_options={"CalibStridedMinMax": 10},
)
print("Quantification terminée :", INT8_ONNX)

onnx.checker.check_model(INT8_ONNX)
print("Structure valide")

session_int8 = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 896, 896).astype(np.float32)
output = session_int8.run(None, {"input": dummy})[0]
print("Shape de sortie :", output.shape)
