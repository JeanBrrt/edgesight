"""Chargement partagé de config/agent.yaml -- lu et parsé UNE SEULE fois
ici, importé ensuite par `agent.py` (D2) et `tools.py` (D1) plutôt que
chacun ouvre indépendamment le même fichier (c'était le cas avant : deux
`yaml.safe_load` du même fichier, un par module). Module séparé plutôt
que centralisé dans `agent.py` directement : `agent.py` importe déjà
`tools.py` (`from .tools import AgentTools`) -- un import en sens
inverse depuis `tools.py` créerait un cycle. Préfixé `_` : usage interne
au package `agent/src/agent/`, pas une API pensée pour d'autres
appelants.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CONFIG_PATH = PROJECT_ROOT / "config" / "agent.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _CONFIG = yaml.safe_load(_f)

LLAMA_SERVER_URL = _CONFIG["llama_server_url"]

# Lancement automatique de llama-server (llm_server.py) -- voir
# config/agent.yaml. `.get(...)` : un config/agent.yaml sans ce bloc
# retombe sur le comportement d'origine (serveur lancé à la main).
LLAMA_SERVER_LAUNCH = _CONFIG.get("llama_server", {"auto_start": False})
SYSTEM_PROMPT = _CONFIG["system_prompt"]
DEFAULT_MAX_ROUNDS = _CONFIG["max_rounds"]

# Plafond de tokens générés par requête -- voir config/agent.yaml.
# `.get(...)` : un config/agent.yaml antérieur à cet ajout garde le
# comportement d'origine (pas de plafond).
MAX_TOKENS = _CONFIG.get("max_tokens")

# Mémoire conversationnelle -- voir config/agent.yaml pour la discussion
# complète. `.get(...)` avec valeurs par défaut : un config/agent.yaml
# antérieur à cet ajout (sans bloc `memory:`) continue de fonctionner
# exactement comme avant (mémoire désactivée), sans erreur de clé absente.
_MEMORY_CONFIG = _CONFIG.get("memory", {})
MEMORY_ENABLED = bool(_MEMORY_CONFIG.get("enabled", False))
MEMORY_MAX_TURNS = int(_MEMORY_CONFIG.get("max_turns", 6))

# Tolérance de fraîcheur pour "actuellement présent" (D1, tools.py) --
# question produit pure (clignotement occasionnel du détecteur, même
# raisonnement que C1/C2), PAS un fudge-factor de latence LLM : `now`
# étant ancré sur l'heure de la question (voir agent.py), le résultat
# est déjà indépendant de la latence d'inférence. Détail complet dans
# config/agent.yaml.
AGENT_FRESHNESS_SECONDS = _CONFIG["freshness_seconds"]
