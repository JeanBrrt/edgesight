import torch
from nanodet.util import cfg, load_config, load_model_weight, Logger
from nanodet.model.arch import build_model
import onnx
import onnxsim

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
MODEL_PATH = "workspace/nanodet-plus-m-1.5x_896-person-car/model_best/nanodet_model_best.pth"
OUTPUT_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-fp32.onnx"
INPUT_SHAPE = (896, 896)

load_config(cfg, CONFIG_PATH)
logger = Logger(0, use_tensorboard=False)
model = build_model(cfg.model)
ckpt = torch.load(MODEL_PATH, map_location="cpu")
load_model_weight(model, ckpt, logger)
model.eval()

dummy_input = torch.randn(1, 3, *INPUT_SHAPE)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
)

model_onnx = onnx.load(OUTPUT_PATH)
model_simplified, ok = onnxsim.simplify(model_onnx, check_n=3)
if ok:
    onnx.save(model_simplified, OUTPUT_PATH)
    print("Simplification réussie")
else:
    print("Simplification échouée, on garde le graphe original")

onnx.checker.check_model(OUTPUT_PATH)  # valide la structure du graphe

model_check = onnx.load(OUTPUT_PATH)
op_types = sorted({node.op_type for node in model_check.graph.node})
print("Opérateurs utilisés :", op_types)
