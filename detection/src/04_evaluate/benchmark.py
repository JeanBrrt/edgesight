import gc
import os
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import torch

from nanodet.data.batch_process import stack_batch_img  # noqa: F401 (référence pipeline, non utilisé directement ici)
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

CONFIG_PATH = "detection/third_party/nanodet/config/nanodet-plus-m-1.5x_896-person-car.yml"
# Modèle final retenu (896px, section 4.3) -- chemin canonique model_best.
PT_MODEL_PATH = "workspace/nanodet-plus-m-1.5x_896-person-car/model_best/nanodet_model_best.pth"
ONNX_FP32_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-fp32.onnx"
# En dur, cohérent avec la résolution d'export des fichiers ci-dessus.
INPUT_SIZE = (896, 896)
# s8s8 (QDQ et QOperator) déjà écarté en B3.4/B4 sur le modèle 416px — pas
# régénéré ici, les deux blocs restent optionnels (`if os.path.exists`) au
# cas où on voudrait un jour revalider le choix sur le nouveau modèle.
ONNX_INT8_QDQ_S8S8_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-s8s8.onnx"
ONNX_INT8_QOPERATOR_S8S8_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QOperator-s8s8.onnx"
ONNX_INT8_QDQ_U8S8_PATH = "detection/models/nanodet-plus-m-1.5x_896-person-car-int8-QDQ-u8s8.onnx"
SAMPLE_IMAGE = "detection/data/03_raw/test/000000186873.jpg"

N_WARMUP = 15
N_RUNS = 150
MEMORY_SAMPLE_INTERVAL = 10  # échantillonner la mémoire toutes les 10 itérations

process = psutil.Process()


def prepare_input():
    """Prétraite une image réelle et renvoie le tenseur d'entrée numpy (1,3,H,W) float32 (H=W=INPUT_SIZE)."""
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    img = cv2.imread(SAMPLE_IMAGE)
    img_info = {"id": 0, "file_name": None, "height": img.shape[0], "width": img.shape[1]}
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(None, meta, INPUT_SIZE)
    img_tensor = meta["img"].transpose(2, 0, 1)[None].astype(np.float32)
    return img_tensor


def measure_rss():
    """RSS courant (Mo), après un passage du garbage collector pour un chiffre stable."""
    gc.collect()
    return process.memory_info().rss


def benchmark(name, run_once_fn, baseline_rss, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Chronomètre run_once_fn : warmup, puis n_runs mesures, avec suivi de la mémoire résidente.

    baseline_rss doit être mesuré AVANT la construction du modèle/session
    (pas juste avant la boucle) pour capturer le coût réel de chargement,
    pas seulement la croissance pendant les appels répétés.
    """
    for _ in range(n_warmup):
        run_once_fn()

    times_ms = []
    peak_rss = baseline_rss
    for i in range(n_runs):
        t0 = time.perf_counter()
        run_once_fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

        if i % MEMORY_SAMPLE_INTERVAL == 0:
            peak_rss = max(peak_rss, process.memory_info().rss)

    times_arr = np.array(times_ms)
    median_ms = statistics.median(times_ms)

    return {
        "name": name,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": median_ms,
        "stdev_ms": statistics.stdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "fps": 1000.0 / median_ms,
        "peak_mem_delta_mb": (peak_rss - baseline_rss) / 1e6,
    }


def print_results_table(results):
    headers = ["Candidat", "Moy(ms)", "Méd(ms)", "Écart-type", "Min(ms)", "Max(ms)", "P95(ms)", "FPS", "ΔMém(Mo)"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        print(
            f"| {r['name']} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | "
            f"{r['stdev_ms']:.3f} | {r['min_ms']:.3f} | {r['max_ms']:.3f} | "
            f"{r['p95_ms']:.3f} | {r['fps']:.1f} | {r['peak_mem_delta_mb']:.2f} |"
        )


def main():
    load_config(cfg, CONFIG_PATH)
    img_tensor = prepare_input()
    results = []

    # --- 1. PyTorch fp32, GPU ---
    baseline = measure_rss()
    logger = Logger(0, use_tensorboard=False)
    model_pt = build_model(cfg.model)
    ckpt = torch.load(PT_MODEL_PATH, map_location="cpu")
    load_model_weight(model_pt, ckpt, logger)

    model_pt_gpu = model_pt.to("cuda:0").eval()
    img_tensor_gpu = torch.from_numpy(img_tensor).to("cuda:0")

    def run_pt_gpu():
        with torch.inference_mode():
            model_pt_gpu(img_tensor_gpu)
            torch.cuda.synchronize()

    results.append(benchmark("PyTorch fp32 GPU", run_pt_gpu, baseline))
    torch.cuda.empty_cache()

    # --- 2. PyTorch fp32, CPU (même objet modèle, déplacé) ---
    baseline = measure_rss()
    model_pt_cpu = model_pt.to("cpu").eval()
    img_tensor_cpu = torch.from_numpy(img_tensor).to("cpu")

    def run_pt_cpu():
        with torch.inference_mode():
            model_pt_cpu(img_tensor_cpu)

    results.append(benchmark("PyTorch fp32 CPU", run_pt_cpu, baseline))

    del model_pt_cpu, model_pt_gpu, model_pt
    gc.collect()

    # --- 3. ONNX fp32, CPU ---
    baseline = measure_rss()
    session_fp32 = ort.InferenceSession(ONNX_FP32_PATH, providers=["CPUExecutionProvider"])

    def run_onnx_fp32():
        session_fp32.run(None, {"input": img_tensor})

    results.append(benchmark("ONNX fp32 CPU", run_onnx_fp32, baseline))

    del session_fp32
    gc.collect()

    # --- 4. ONNX INT8 QDQ (s8s8), CPU (optionnel, si le fichier existe) ---
    if os.path.exists(ONNX_INT8_QDQ_S8S8_PATH):
        baseline = measure_rss()
        session_int8_qdq = ort.InferenceSession(ONNX_INT8_QDQ_S8S8_PATH, providers=["CPUExecutionProvider"])

        def run_onnx_int8_qdq():
            session_int8_qdq.run(None, {"input": img_tensor})

        results.append(benchmark("ONNX INT8 QDQ CPU (s8s8)", run_onnx_int8_qdq, baseline))
        del session_int8_qdq
        gc.collect()
    else:
        print(
            f"[info] {ONNX_INT8_QDQ_S8S8_PATH} introuvable — déjà écarté en "
            "B3.4/B4 sur le modèle 416px, pas régénéré ici.\n"
        )

    # --- 5. ONNX INT8 QOperator (s8s8), CPU (optionnel, si le fichier existe) ---
    if os.path.exists(ONNX_INT8_QOPERATOR_S8S8_PATH):
        baseline = measure_rss()
        session_int8_qop = ort.InferenceSession(ONNX_INT8_QOPERATOR_S8S8_PATH, providers=["CPUExecutionProvider"])

        def run_onnx_int8_qop():
            session_int8_qop.run(None, {"input": img_tensor})

        results.append(benchmark("ONNX INT8 QOperator CPU (s8s8)", run_onnx_int8_qop, baseline))
        del session_int8_qop
        gc.collect()
    else:
        print(
            f"[info] {ONNX_INT8_QOPERATOR_S8S8_PATH} introuvable — "
            "régénère-le avec quant_format=QOperator (fichier de sortie différent) pour l'inclure au comparatif.\n"
        )

    # --- 6. ONNX INT8 QDQ (u8s8 activations), CPU (optionnel, si le fichier existe) ---
    if os.path.exists(ONNX_INT8_QDQ_U8S8_PATH):
        baseline = measure_rss()
        session_int8_qdq_u8s8 = ort.InferenceSession(ONNX_INT8_QDQ_U8S8_PATH, providers=["CPUExecutionProvider"])

        def run_onnx_int8_qdq_u8s8():
            session_int8_qdq_u8s8.run(None, {"input": img_tensor})

        results.append(benchmark("ONNX INT8 QDQ CPU (u8s8)", run_onnx_int8_qdq_u8s8, baseline))
        del session_int8_qdq_u8s8
        gc.collect()
    else:
        print(
            f"[info] {ONNX_INT8_QDQ_U8S8_PATH} introuvable — "
            "régénère-le avec activation_type=QUInt8 pour l'inclure au comparatif.\n"
        )

    print()
    print_results_table(results)


if __name__ == "__main__":
    main()
