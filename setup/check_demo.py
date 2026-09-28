"""Vérifie que tout est en place pour lancer les scripts de démo
(demo/src/scripts/*.py) -- dépendances Python, code NanoDet vendored,
modèles ONNX et vidéos par défaut (tous commités avec le dépôt, voir
README "Ce qui est inclus dans le clone"). Le LLM local (llama-server +
GGUF) n'est nécessaire que pour 03_live_agent_demo.py -- vérifié à part
et jamais bloquant ici, 01_detection_demo.py/02_tracking_demo.py s'en
passent entièrement.

Usage (depuis la racine du projet, après avoir suivi le README pour les
dépendances Python et detection/third_party/) :
    uv run python setup/check_demo.py
"""

from pathlib import Path

from _checks import CheckResult, check_file, check_import, run


def demo_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    for module in ("cv2", "numpy", "onnxruntime", "torch", "yaml"):
        results.append(check_import(module))
    results.append(check_import("nanodet", label="nanodet (detection/third_party/, uv pip install -e ...)"))

    # config/demo.yaml lui-même, puis chaque chemin qu'il référence -- si le
    # fichier est absent ou que PyYAML manque, on le signale et on s'arrête
    # là plutôt que de laisser une exception remonter (le but de ce script
    # est justement de repérer ce genre de trou proprement).
    demo_cfg_path = "config/demo.yaml"
    results.append(check_file(demo_cfg_path))
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    if _yaml is not None and Path(demo_cfg_path).exists():
        try:
            with open(demo_cfg_path, encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
            results.append(check_file(cfg["config_path"], label="config NanoDet (config_path)"))
            for name, model in cfg["models"].items():
                results.append(check_file(model["onnx_path"], label=f"modèle ONNX {name}"))
            for label, source in cfg["demo_sources"]:
                if source.startswith("INTERACTIVE:"):
                    bg_path, sprite_path = source[len("INTERACTIVE:"):].split("|")
                    results.append(check_file(bg_path, label=f"source « {label} » (fond)"))
                    results.append(check_file(sprite_path, label=f"source « {label} » (silhouette)"))
                else:
                    results.append(check_file(source, label=f"source « {label} »"))
        except Exception as exc:  # config malformée -- signaler, ne jamais planter le script
            results.append(CheckResult("Lecture de config/demo.yaml", False, str(exc)))

    # Configs partagées nécessaires même sans LLM (C1 tracking, D3 zones)
    results.append(check_file("config/tracker.yaml"))
    results.append(check_file("config/zones.yaml"))

    # LLM local -- uniquement pour 03_live_agent_demo.py
    results.append(CheckResult(
        "llama-server (binaire)",
        Path("agent/third_party/llama.cpp/llama-server.exe").exists(),
        "agent/third_party/llama.cpp/llama-server.exe -- voir README section Agent",
        optional=True,
    ))
    models_dir = Path("agent/models")
    gguf_found = models_dir.exists() and any(models_dir.glob("*.gguf"))
    results.append(CheckResult(
        "Modèle GGUF (LLM local)",
        gguf_found,
        "agent/models/*.gguf -- voir README section Agent",
        optional=True,
    ))

    return results


def main() -> int:
    ok = run("Démo (demo/src/scripts/*.py)", demo_checks())
    print()
    if ok:
        print("Tout est en place pour 01_detection_demo.py / 02_tracking_demo.py.")
        print("Pour 03_live_agent_demo.py, lance aussi llama-server (voir README) si les")
        print("vérifications optionnelles ci-dessus (llama-server/GGUF) sont en échec.")
    else:
        print("Des éléments bloquants manquent -- voir le détail ci-dessus et le README.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
