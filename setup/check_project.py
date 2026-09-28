"""Vérifie que l'environnement est prêt pour reproduire l'ensemble du
projet -- pas seulement la démo (voir check_demo.py pour ça, repris ici
comme premier sous-ensemble) : préparation des données COCO,
entraînement, export ONNX, quantification INT8, évaluation, banc de
test de l'agent.

Usage (depuis la racine du projet) :
    uv run python setup/check_project.py
"""

from pathlib import Path

from _checks import CheckResult, check_file, check_import, run
from check_demo import demo_checks


def training_pipeline_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    for module in ("pycocotools", "onnx", "requests", "tqdm", "sklearn", "psutil"):
        results.append(check_import(module))

    results.append(check_file(
        "detection/data/01_annotations/instances_train2017.json",
        label="annotations COCO train (01_annotations)",
    ))
    results.append(check_file(
        "detection/data/01_annotations/instances_val2017.json",
        label="annotations COCO val (01_annotations)",
    ))
    results.append(CheckResult(
        "Dataset préparé (02_filtered / 03_raw / 04_processed)",
        Path("detection/data/04_processed/train.json").exists(),
        "absent -- normal si le pipeline de données n'a jamais tourné : "
        "uv run python detection/src/01_data/data_filter.py, puis data_download.py, "
        "puis data_prepare.py (voir detection/src/01_data/)",
        optional=True,
    ))
    results.append(check_file(
        "detection/models/pretrained/nanodet-plus-m-1.5x_416.pth",
        label="checkpoint pré-entraîné COCO (fine-tuning + référence zero-shot)",
    ))

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    results.append(CheckResult(
        "GPU CUDA disponible",
        cuda_ok,
        "aucun GPU CUDA détecté -- entraînement/évaluation (detection/src/03_quantize/, "
        "04_evaluate/) en ont besoin ; la démo, elle, tourne en CPU pur via ONNX Runtime",
        optional=True,
    ))

    return results


def agent_eval_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module in ("openai", "supervision", "trackers"):
        results.append(check_import(module))
    results.append(check_file("agent/eval/cases.py"))
    return results


def main() -> int:
    ok_demo = run("Démo (sous-ensemble -- voir check_demo.py)", demo_checks())
    ok_training = run("Pipeline détection (données, entraînement, export, quantification)", training_pipeline_checks())
    ok_agent_eval = run("Banc de test agent (agent/eval/)", agent_eval_checks())

    all_ok = ok_demo and ok_training and ok_agent_eval
    print()
    if all_ok:
        print("Environnement complet prêt.")
    else:
        print("Des éléments bloquants manquent -- voir le détail ci-dessus et le README.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
