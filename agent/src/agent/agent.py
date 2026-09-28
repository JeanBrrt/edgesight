"""D2 — Boucle agent : envoie la question au LLM local (llama-server, API
compatible OpenAI), exécute les tools qu'il demande, renvoie le résultat,
récupère la réponse finale en langage naturel.

Suppose que `llama-server` tourne déjà (voir README / commande de lancement
dans docs/justifications.md).
"""

import json
import logging
import time

from openai import OpenAI, OpenAIError

from ..journal.event_store import EventStore
from .tools import AgentTools
from .tool_schemas import TOOL_SCHEMAS
from ._config import (
    LLAMA_SERVER_URL,
    SYSTEM_PROMPT,
    DEFAULT_MAX_ROUNDS,
    MEMORY_ENABLED,
    MEMORY_MAX_TURNS,
)

# Silencieux par défaut (aucun handler configuré ici) -- l'appelant décide
# où va ce log en configurant logging lui-même (cf.
# demo/src/scripts/03_live_agent_demo.py, qui l'écrit dans un fichier pour garder
# la console libre pour la saisie au clavier). Sans configuration
# explicite côté appelant, ces logger.info() n'affichent rien nulle part
# -- aucun risque de polluer un usage qui ne s'y attend pas (ex. le banc
# de test agent/eval/, qui utilise sa propre boucle et jamais cette
# classe directement).
logger = logging.getLogger(__name__)


class Session:
    """Historique conversationnel persistant pour UN appelant donné (D2,
    mémoire -- section correspondante de config/agent.yaml).

    Précaution de conception : chaque appelant crée et garde SA PROPRE
    instance -- `input_worker` (demo/src/scripts/03_live_agent_demo.py) en garde
    une pour toute la durée du script, `alert_notifier` n'en crée jamais
    (chaque notification d'alerte reste un appel isolé, sans historique,
    passé sans `session` à `Agent.ask()`). Deux threads concurrents ne se
    retrouvent donc jamais à muter la même liste `messages` en même temps
    -- contrairement à un historique unique porté par `Agent` lui-même,
    qui aurait exposé une vraie condition de course entre le thread de
    saisie utilisateur et celui des alertes."""

    def __init__(self):
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        """Repart d'un historique vierge (ex. changement de source vidéo,
        où le contexte de la conversation précédente n'a plus de sens)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


class Agent:
    def __init__(self, store: EventStore):
        self.client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
        self.tools = AgentTools(store)
        # Table de dispatch nom_tool -> méthode réelle
        self._dispatch = {
            "count_now": self.tools.count_now,
            "count_total": self.tools.count_total,
            "count_since": self.tools.count_since,
            "count_between": self.tools.count_between,
            "time_since_last_seen": self.tools.time_since_last_seen,
            "average_presence_duration": self.tools.average_presence_duration,
            "peak_concurrent_count": self.tools.peak_concurrent_count,
            "set_presence_duration_alert": self.tools.set_presence_duration_alert,
            "set_co_occurrence_alert": self.tools.set_co_occurrence_alert,
            "set_surge_alert": self.tools.set_surge_alert,
            "set_zone_alert": self.tools.set_zone_alert,
            "count_zone_entries": self.tools.count_zone_entries,
        }

    def _execute_tool(self, name: str, arguments: dict, now: float) -> str:
        """Exécute un tool ; renvoie toujours une chaîne (résultat ou message
        d'erreur lisible) — jamais d'exception qui remonterait jusqu'à l'appelant,
        pour que le LLM puisse réagir à une erreur plutôt que de planter.

        `now` est injecté ici, jamais demandé au LLM (absent de
        `tool_schemas.py`) — c'est l'heure de la question de l'utilisateur,
        pas l'heure d'exécution réelle du tool, pour que le résultat ne
        dépende pas de la latence d'inférence du LLM."""
        if name not in self._dispatch:
            return f"Erreur : tool inconnu '{name}'."
        try:
            result = self._dispatch[name](**arguments, now=now)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as exc:
            return f"Erreur lors de l'exécution de '{name}': {exc}"

    def ask(
        self,
        question: str,
        session: "Session | None" = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        tool_calls_log: list[tuple[str, dict]] | None = None,
    ) -> str:
        """`session` optionnelle (défaut `None`) : sans elle, comportement
        D'ORIGINE inchangé -- une liste [system, user] neuve à chaque
        appel, aucun historique (c'est le cas d'`alert_notifier`, qui ne
        passe jamais de session, volontairement -- voir `Session`).

        Avec une `session`, ET `memory.enabled: true` dans
        config/agent.yaml (les DEUX conditions sont nécessaires), la
        question s'ajoute à l'historique de cette session plutôt que d'en
        repartir de zéro -- pour qu'une question elliptique ("et de
        voitures ?") soit comprise à la lumière de l'échange précédent.
        Repasser `memory.enabled` à `false` désactive ce comportement
        PARTOUT instantanément, même si l'appelant continue de passer une
        `session` -- aucun code à toucher ailleurs pour revenir en arrière.

        `tool_calls_log` optionnelle : si fournie, chaque appel d'outil
        (nom, arguments) de CET échange y est ajouté au fur et à mesure --
        pour un appelant qui veut afficher les tools utilisés sans dépendre
        du fichier de log (cf. le panneau "Outils appelés" de
        demo/src/scripts/03_live_agent_demo.py). Sans elle (défaut), le
        comportement est inchangé : une liste interne jetable, utilisée
        uniquement pour le résumé loggé en fin d'échange."""
        use_memory = MEMORY_ENABLED and session is not None
        if tool_calls_log is None:
            tool_calls_log = []

        # Capturé avant tout appel au LLM — pas au moment où le tool s'exécute
        # réellement (qui dépend de la latence d'inférence, mesurée ~1,6s).
        request_ts = time.time()
        logger.info("Question : %s", question)

        if use_memory:
            messages = session.messages
            # Filet de sécurité : si l'échange en cours échoue (erreur
            # réseau ou abandon après max_rounds), on revient exactement à
            # cette longueur avant de renvoyer une erreur -- une question
            # sans réponse ne doit jamais polluer l'historique des tours
            # suivants avec un message "user" sans "assistant" en réponse.
            #
            # Concurrence (`session.reset()` peut être appelé depuis le
            # thread vidéo principal pendant qu'un `ask()` tourne ici, cf.
            # 03_live_agent_demo.py au changement de source) : `messages`
            # capture la liste actuelle une fois pour toutes ; toute
            # écriture PLUS BAS revérifie `session.messages is messages`
            # avant de committer, pour ne jamais écraser un reset survenu
            # entre-temps avec les données de l'échange en cours. Pas de
            # verrou : le pire cas concret est la perte silencieuse d'un
            # seul échange tombé pile sur un changement de scène (fenêtre
            # de quelques secondes, déclenchement manuel) -- un verrou
            # bloquerait la boucle vidéo pendant l'appel LLM (~1,5-2s),
            # exactement ce que le thread séparé sert à éviter.
            rollback_len = len(messages)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            rollback_len = 0  # jamais utilisé (liste jetable), valeur sans effet
        messages.append({"role": "user", "content": question})

        # `tool_calls_log` accumule CHAQUE appel d'outil (nom, arguments) sur
        # toute la durée de cette question, tous rounds confondus -- sert au
        # résumé loggé ci-dessous (_log_summary) et, si l'appelant a fourni
        # sa propre liste, à son propre affichage (voir docstring ci-dessus).
        # Chaque appel est aussi déjà tracé round par round via les
        # logger.info existants "Tool %s(%s) -> %s", indépendamment de ceci.

        def _log_summary(attempts: int, outcome: str) -> None:
            logger.info(
                "Resume question=%r -- %s -- %d/%d tentative(s) -- outils appeles=%s",
                question, outcome, attempts, max_rounds, tool_calls_log,
            )

        for round_idx in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model="local",
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except OpenAIError as exc:
                # Couvre les erreurs réseau (llama-server pas lancé/injoignable),
                # timeout, réponse malformée, etc. -- sans ce garde-fou,
                # l'exception remonte non gérée jusqu'à l'appelant. Pour
                # input_worker/alert_notifier (demo/src/scripts/03_live_agent_demo.py),
                # ça plantait le thread avec une trace complète imprimée sur
                # la console -- exactement ce qu'on essaie d'éviter en gardant
                # la console propre pour la saisie.
                logger.error("Echec de connexion au serveur LLM (round %d) : %s", round_idx, exc)
                _log_summary(round_idx + 1, "echec de connexion")
                if use_memory and session.messages is messages:
                    del messages[rollback_len:]
                return "Erreur : impossible de contacter le serveur LLM (llama-server est-il lancé ?)."
            message = response.choices[0].message

            if not message.tool_calls:
                logger.info("Reponse finale (round %d) : %s", round_idx, message.content)
                _log_summary(round_idx + 1, "reponse finale")
                if use_memory:
                    messages.append({"role": "assistant", "content": message.content})
                    if session.messages is messages:
                        self._trim_session(session, messages)
                return message.content

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    result = f"Erreur : arguments JSON invalides reçus pour '{call.function.name}'."
                    logger.info("Tool %s -- arguments JSON invalides : %r", call.function.name, call.function.arguments)
                    tool_calls_log.append((call.function.name, {"_json_error": call.function.arguments}))
                else:
                    result = self._execute_tool(call.function.name, arguments, now=request_ts)
                    logger.info("Tool %s(%s) -> %s", call.function.name, arguments, result)
                    tool_calls_log.append((call.function.name, arguments))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        logger.info("Abandon apres %d tours sans reponse finale.", max_rounds)
        _log_summary(max_rounds, "abandon")
        if use_memory and session.messages is messages:
            del messages[rollback_len:]
        return "Désolé, je n'ai pas réussi à répondre après plusieurs tentatives d'appel d'outils."

    @staticmethod
    def _trim_session(session: "Session", messages: list[dict]) -> None:
        """Fenêtre glissante (config/agent.yaml, memory.max_turns) : ne
        garde que les N derniers échanges complets, plus le message
        système en position 0. Coupe systématiquement à une frontière
        d'échange (juste avant un message 'user'), jamais au milieu d'une
        séquence de tool calls -- un historique tronqué en plein milieu
        d'un échange laisserait des messages 'tool' sans le tool_call
        assistant correspondant, que l'API rejetterait.

        `messages` : la liste réellement utilisée par CET appel (pas
        re-dérivée de `session.messages`, cf. note de concurrence dans
        `ask()`) -- l'appelant a déjà vérifié `session.messages is
        messages` avant d'appeler cette méthode."""
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_indices) <= MEMORY_MAX_TURNS:
            return
        cutoff = user_indices[-MEMORY_MAX_TURNS]
        session.messages = [messages[0]] + messages[cutoff:]
