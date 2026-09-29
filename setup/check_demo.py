"""Vérifie que tout est en place pour lancer les scripts de démo
(demo/src/scripts/*.py) -- dépendances Python, code NanoDet vendored,
modèles ONNX et vidéos par défaut (tous commités avec le dépôt, voir
README "Structure du dépôt"). Le LLM local (llama-server + GGUF) n'est
nécessaire que pour 03_live_agent_demo.py -- vérifié à part et jamais
bloquant ici, 01_detection_demo.py/02_tracking_demo.py s'en passent
entièrement.

Chaque vérification teste ce dont les démos ont réellement besoin, pas
seulement la présence d'un fichier : imports exacts des scripts (un
`import nanodet` peut réussir alors que `nanodet.model.arch` plante),
modèles ONNX chargés par ONNX Runtime, vidéos décodées par OpenCV, GGUF
contrôlé (en-tête + taille), binaire llama-server exécuté.

Usage (depuis n'importe quel dossier, après avoir suivi le README pour
les dépendances Python et detection/third_party/) :
    uv run python setup/check_demo.py
"""

import subprocess
from pathlib import Path

from _checks import CheckResult, check_call, check_file, check_import, run

# Modules importés par les 3 scripts de démo et par ce qu'ils chargent
# (agent/src/, demo/src/common/) -- pour 03_live_agent_demo.py, tkinter et
# openai en plus.
DEMO_IMPORTS = (
    "cv2", "numpy", "onnxruntime", "torch", "yaml", "PIL",
    "supervision", "trackers", "openai", "tkinter",
)
NANODET_IMPORTS = (
    "nanodet.model.arch", "nanodet.data.transform",
    "nanodet.data.collate", "nanodet.data.batch_process", "nanodet.util",
)
NANODET_HINT = "voir README étapes 2 à 4 (clone, dépendances, patchs)"


def _check_onnx(path: str):
    def fn():
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        return f"{path} (entrée {shape})"
    return fn


def _check_video(path: str):
    def fn():
        import cv2
        if not Path(path).exists():
            raise FileNotFoundError(f"introuvable : {path}")
        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"OpenCV n'arrive pas à décoder la première image de {path}")
        return f"{path} ({frame.shape[1]}x{frame.shape[0]})"
    return fn


def _check_image(path: str, need_alpha: bool = False):
    def fn():
        import cv2
        if not Path(path).exists():
            raise FileNotFoundError(f"introuvable : {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"image illisible par OpenCV : {path}")
        if need_alpha and (img.ndim != 3 or img.shape[2] != 4):
            raise RuntimeError(f"canal alpha (RGBA) attendu, reçu shape={img.shape} : {path}")
        return path
    return fn


def _check_gguf(path: Path, expected_size: int | None):
    def fn():
        if not path.exists():
            raise FileNotFoundError(f"introuvable : {path} -- voir README étape 6")
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            raise RuntimeError(f"en-tête {magic!r} au lieu de b'GGUF' -- fichier corrompu, le retélécharger")
        size = path.stat().st_size
        if expected_size is not None and size != expected_size:
            raise RuntimeError(
                f"{size:,} octets au lieu de {expected_size:,} -- téléchargement tronqué "
                "ou incomplet, le relancer (README étape 6)"
            )
        return f"{path} ({size:,} octets)"
    return fn


def _check_llama_server(binary: Path):
    def fn():
        if not binary.exists():
            raise FileNotFoundError(f"introuvable : {binary} -- voir README étape 5")
        # --version charge les DLL (ggml, CUDA) : un zip cudart oublié ou
        # une archive mal extraite échoue ici, pas seulement au lancement
        # de la démo.
        proc = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            raise RuntimeError(f"code de sortie {proc.returncode} -- DLL manquantes ? (README étape 5) {output[-300:]}")
        version = next((line for line in output.splitlines() if line.startswith("version")), "")
        return f"{binary} {version}".strip()
    return fn


def demo_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    for module in DEMO_IMPORTS:
        results.append(check_import(module))
    for module in NANODET_IMPORTS:
        result = check_import(module)
        if not result.ok:
            result.detail += f" -- {NANODET_HINT}"
        results.append(result)

    try:
        import yaml as _yaml
    except Exception:
        _yaml = None

    # config/demo.yaml lui-même, puis chaque chemin qu'il référence -- si le
    # fichier est absent ou que PyYAML manque, on le signale et on s'arrête
    # là plutôt que de laisser une exception remonter (le but de ce script
    # est justement de repérer ce genre de trou proprement).
    demo_cfg_path = "config/demo.yaml"
    results.append(check_file(demo_cfg_path))
    if _yaml is not None and Path(demo_cfg_path).exists():
        try:
            with open(demo_cfg_path, encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
            results.append(check_file(cfg["config_path"], label="config NanoDet (config_path)"))
            if cfg["active_model"] not in cfg["models"]:
                results.append(CheckResult(
                    "active_model", False,
                    f"« {cfg['active_model']} » absent de models ({', '.join(cfg['models'])})",
                ))
            for name, model in cfg["models"].items():
                results.append(check_call(f"modèle ONNX {name}", _check_onnx(model["onnx_path"])))
            for label, source in cfg["demo_sources"]:
                if source.startswith("INTERACTIVE:"):
                    bg_path, sprite_path = source[len("INTERACTIVE:"):].split("|")
                    results.append(check_call(f"source « {label} » (fond)", _check_image(bg_path)))
                    results.append(check_call(
                        f"source « {label} » (silhouette)", _check_image(sprite_path, need_alpha=True),
                    ))
                else:
                    results.append(check_call(f"source « {label} »", _check_video(source)))
        except Exception as exc:  # config malformée -- signaler, ne jamais planter le script
            results.append(CheckResult("Lecture de config/demo.yaml", False, f"{type(exc).__name__}: {exc}"))

    # Configs partagées nécessaires même sans LLM (C1 tracking, D3 zones)
    results.append(check_file("config/tracker.yaml"))
    results.append(check_file("config/zones.yaml"))

    # LLM local -- uniquement pour 03_live_agent_demo.py. Chemins et taille
    # attendue lus dans config/agent.yaml (même source que la démo).
    launch = {}
    agent_cfg_path = Path("config/agent.yaml")
    if _yaml is not None and agent_cfg_path.exists():
        try:
            with open(agent_cfg_path, encoding="utf-8") as f:
                launch = (_yaml.safe_load(f) or {}).get("llama_server", {}) or {}
        except Exception as exc:
            results.append(CheckResult("Lecture de config/agent.yaml", False, f"{type(exc).__name__}: {exc}", optional=True))
    binary = Path(launch.get("binary", "agent/third_party/llama.cpp/llama-server.exe"))
    gguf = Path(launch.get("model", "agent/models/granite-4.1-3b-Q4_K_M.gguf"))
    results.append(check_call("llama-server (binaire)", _check_llama_server(binary), optional=True))
    results.append(check_call(
        "Modèle GGUF (LLM local)", _check_gguf(gguf, launch.get("model_size_bytes")), optional=True,
    ))

    return results


def main() -> int:
    results = demo_checks()
    ok = run("Démo (demo/src/scripts/*.py)", results)
    llm_ok = all(r.ok for r in results if r.optional)
    print()
    if not ok:
        print("Des éléments bloquants manquent -- voir le détail ci-dessus et le README.")
    elif llm_ok:
        print("Tout est en place pour les 3 démos, y compris 03_live_agent_demo.py (LLM local).")
    else:
        print("Tout est en place pour 01_detection_demo.py / 02_tracking_demo.py.")
        print("Pour 03_live_agent_demo.py, corriger d'abord les vérifications optionnelles")
        print("ci-dessus (llama-server / GGUF, README étapes 5 et 6).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
