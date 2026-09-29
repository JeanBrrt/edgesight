"""Boucle de l'agent : envoie la question au LLM local (llama-server, API
compatible OpenAI), exécute les outils demandés et renvoie la réponse
finale
"""

import json
import logging
import time

from openai import OpenAI, OpenAIError

from ..journal.event_store import EventStore
from .tools import AgentTools, FinalAnswer
from .tool_schemas import TOOL_SCHEMAS
from ._config import (
    LLAMA_SERVER_URL,
    SYSTEM_PROMPT,
    DEFAULT_MAX_ROUNDS,
    MAX_TOKENS,
    MEMORY_ENABLED,
    MEMORY_MAX_TURNS,
)

# Aucun handler ici : c'est l'appelant qui décide où vont les logs.
logger = logging.getLogger(__name__)


class Session:
    """Historique d'une conversation. Chaque appelant garde la sienne (la
    démo en a une, les alertes n'en ont pas) : deux threads ne modifient
    jamais le même historique."""

    def __init__(self):
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        """Vide l'historique (ex. changement de source vidéo)."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


class Agent:
    def __init__(self, store: EventStore):
        self.client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed")
        self.tools = AgentTools(store)
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
        """Exécute un outil et renvoie toujours une chaîne, erreur comprise,
        pour que le LLM puisse y réagir.

        `now` est l'heure de la question, pas celle de l'exécution : le
        résultat ne dépend pas de la latence du LLM."""
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
        tool_calls_log: list[tuple[str, dict, str]] | None = None,
    ) -> str:
        """Répond à `question`.

        `session` : historique à prolonger, si la mémoire est activée dans
        config/agent.yaml. Sans session, la question est traitée seule.
        `tool_calls_log` : reçoit chaque appel d'outil (nom, arguments,
        résultat), pour l'afficher (panneau "Outils appelés" de la démo)."""
        use_memory = MEMORY_ENABLED and session is not None
        if tool_calls_log is None:
            tool_calls_log = []

        # Heure de la question, avant l'appel au LLM (~1,6 s de latence).
        request_ts = time.time()
        logger.info("Question : %s", question)

        if use_memory:
            messages = session.messages
            # En cas d'échec, l'historique revient à cette longueur : pas de
            # question sans réponse dans la mémoire.
            #
            # La démo peut vider la session pendant cet appel (changement de
            # source). Chaque écriture vérifie donc `session.messages is
            # messages`, pour ne pas écraser ce reset. Pas de verrou : il
            # bloquerait la vidéo pendant l'appel au LLM.
            rollback_len = len(messages)
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            rollback_len = 0
        messages.append({"role": "user", "content": question})

        def _log_summary(attempts: int, outcome: str) -> None:
            logger.info(
                "Resume question=%r -- %s -- %d/%d tentative(s) -- outils appeles=%s",
                question, outcome, attempts, max_rounds, tool_calls_log,
            )

        # Vrai dès qu'un outil renvoie autre chose qu'une FinalAnswer (voir plus bas).
        had_ordinary_result = False

        def _finish(answer: str, round_idx: int, outcome: str) -> str:
            logger.info("Reponse finale (round %d, %s) : %s", round_idx, outcome, answer)
            _log_summary(round_idx + 1, outcome)
            if use_memory:
                messages.append({"role": "assistant", "content": answer})
                if session.messages is messages:
                    self._trim_session(session, messages)
            return answer

        request_options = {"max_tokens": MAX_TOKENS} if MAX_TOKENS else {}

        for round_idx in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model="local",
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    **request_options,
                )
            except OpenAIError as exc:
                # Serveur injoignable, délai dépassé, réponse invalide :
                # message d'erreur plutôt qu'un thread qui plante.
                logger.error("Echec de connexion au serveur LLM (round %d) : %s", round_idx, exc)
                _log_summary(round_idx + 1, "echec de connexion")
                if use_memory and session.messages is messages:
                    del messages[rollback_len:]
                return "Erreur : impossible de contacter le serveur LLM (llama-server est-il lancé ?)."
            choice = response.choices[0]
            message = choice.message

            # Plafond max_tokens atteint : le modèle boucle. Réponse écartée.
            if choice.finish_reason == "length":
                logger.warning("Plafond max_tokens=%s atteint (round %d) -- reponse abandonnee.", MAX_TOKENS, round_idx)
                _log_summary(round_idx + 1, "plafond max_tokens atteint")
                if use_memory and session.messages is messages:
                    del messages[rollback_len:]
                return "Désolé, je n'ai pas réussi à formuler une réponse. Pouvez-vous reformuler la question ?"

            if not message.tool_calls:
                return _finish(message.content, round_idx, "reponse finale")

            messages.append(message.model_dump(exclude_none=True))
            final_answers: list[str] = []
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    result = f"Erreur : arguments JSON invalides reçus pour '{call.function.name}'."
                    logger.info("Tool %s -- arguments JSON invalides : %r", call.function.name, call.function.arguments)
                    tool_calls_log.append((call.function.name, {"_json_error": call.function.arguments}, result))
                else:
                    result = self._execute_tool(call.function.name, arguments, now=request_ts)
                    logger.info("Tool %s(%s) -> %s", call.function.name, arguments, result)
                    tool_calls_log.append((call.function.name, arguments, result))
                if isinstance(result, FinalAnswer):
                    final_answers.append(result)
                else:
                    had_ordinary_result = True
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

            # Si tous les outils ont renvoyé une réponse déjà rédigée
            # (FinalAnswer), on la renvoie telle quelle : le LLM en inversait
            # parfois le sens en la reformulant. Un seul résultat ordinaire,
            # même à un tour précédent, et le LLM reformule normalement.
            if final_answers and not had_ordinary_result:
                return _finish(" ".join(final_answers), round_idx, "reponse finale directe, sans reformulation")

        logger.info("Abandon apres %d tours sans reponse finale.", max_rounds)
        _log_summary(max_rounds, "abandon")
        if use_memory and session.messages is messages:
            del messages[rollback_len:]
        return "Désolé, je n'ai pas réussi à répondre après plusieurs tentatives d'appel d'outils."

    @staticmethod
    def _trim_session(session: "Session", messages: list[dict]) -> None:
        """Ne garde que les memory.max_turns derniers échanges, plus le
        message système. Coupe toujours avant un message "user" : couper au
        milieu d'appels d'outils laisserait des réponses orphelines, que
        l'API refuse."""
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_indices) <= MEMORY_MAX_TURNS:
            return
        cutoff = user_indices[-MEMORY_MAX_TURNS]
        session.messages = [messages[0]] + messages[cutoff:]
