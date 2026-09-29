from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "demo.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

CONFIG_PATH = _CONFIG["config_path"]

# input_size : liste dans le YAML, tuple attendu par NanoDet
MODELS = _CONFIG["models"]
for _model in MODELS.values():
    _model["input_size"] = tuple(_model["input_size"])
ACTIVE_MODEL = _CONFIG["active_model"]

ONNX_PATH = MODELS[ACTIVE_MODEL]["onnx_path"]
INPUT_SIZE = MODELS[ACTIVE_MODEL]["input_size"]

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CUSTOM_VIDEOS_DIR = _PROJECT_ROOT / _CONFIG["custom_videos_dir"]
_CUSTOM_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

_CUSTOM_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
_custom_sources = [
    (f"perso - {path.stem.replace('_', ' ').replace('-', ' ')}", str(path))
    for path in sorted(_CUSTOM_VIDEOS_DIR.iterdir())
    if path.suffix.lower() in _CUSTOM_VIDEO_EXTENSIONS
]

DEMO_SOURCES = [tuple(entry) for entry in _CONFIG["demo_sources"]] + _custom_sources

DISPLAY_MAX_WIDTH = _CONFIG["display_max_width"]
DISABLED_CLASSES = _CONFIG["disabled_classes"]
RAW_SCORE_THRESHOLD = _CONFIG["raw_score_threshold"]
PERSON_ALERT_THRESHOLD_SECONDS = _CONFIG["person_alert_threshold_seconds"]
