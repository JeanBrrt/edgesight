"""Lit config/agent.yaml une seule fois pour agent.py et tools.py. Module
à part pour éviter un import circulaire entre les deux."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_PATH = PROJECT_ROOT / "config" / "agent.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

LLAMA_SERVER_URL = _CONFIG["llama_server_url"]

# Sans ce bloc, le serveur est à lancer à la main.
LLAMA_SERVER_LAUNCH = _CONFIG.get("llama_server", {"auto_start": False})
SYSTEM_PROMPT = _CONFIG["system_prompt"]
DEFAULT_MAX_ROUNDS = _CONFIG["max_rounds"]

# Clés facultatives : sans elles, pas de plafond de tokens ni de mémoire.
MAX_TOKENS = _CONFIG.get("max_tokens")

_MEMORY_CONFIG = _CONFIG.get("memory", {})
MEMORY_ENABLED = bool(_MEMORY_CONFIG.get("enabled", False))
MEMORY_MAX_TURNS = int(_MEMORY_CONFIG.get("max_turns", 6))

AGENT_FRESHNESS_SECONDS = _CONFIG["freshness_seconds"]
