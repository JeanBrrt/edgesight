"""Moteur du banc de test 
"""

import json
import logging
import time

from cases import ANY, Approx, OneOf, TestCase

logger = logging.getLogger(__name__)


class RecordingAgent:
    """Comme Agent, mais les appels d'outils sont enregistrés au lieu
    d'être exécutés, avec un résultat factice pour que le modèle continue."""

    DUMMY_TOOL_RESULT = "3"

    def __init__(
        self,
        client,
        tool_schemas: list[dict],
        system_prompt: str,
        max_rounds: int = 4,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.tool_schemas = tool_schemas
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        # Sans plafond, un modèle qui boucle génère sans fin (vu : plus de
        # 4 000 tokens sur multi__compare_now). 1024 est très large.
        self.max_tokens = max_tokens

    def ask(self, question: str) -> tuple[list[tuple[str, dict]], float, str | None, str | None]:
        """Renvoie (appels enregistrés, latence en s, texte final, erreur).

        `erreur` ne concerne que l'appel à l'API (serveur indisponible,
        réponse invalide), pas un mauvais choix d'outil. Ne lève jamais :
        une question qui plante ne doit pas arrêter le banc (trace dans le
        log)."""
        recorded: list[tuple[str, dict]] = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        start = time.time()
        final_text = None
        error = None
        logger.debug("question: %r", question)

        try:
            for round_idx in range(self.max_rounds):
                response = self.client.chat.completions.create(
                    model="local",
                    messages=messages,
                    tools=self.tool_schemas,
                    max_tokens=self.max_tokens,
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    final_text = message.content
                    logger.debug("round %d: reponse finale -> %r", round_idx, final_text)
                    break

                messages.append(message.model_dump(exclude_none=True))
                for call in message.tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            "round %d: arguments JSON invalides pour '%s': %r",
                            round_idx, call.function.name, call.function.arguments,
                        )
                        arguments = {"_json_error": call.function.arguments}
                    recorded.append((call.function.name, arguments))
                    logger.debug("round %d: tool_call -> %s(%s)", round_idx, call.function.name, arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": self.DUMMY_TOOL_RESULT})
            else:
                logger.warning("max_rounds (%d) atteint sans reponse finale pour %r", self.max_rounds, question)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("echec de l'appel API pour la question %r (round %s)", question, round_idx if "round_idx" in locals() else "?")

        latency = time.time() - start
        logger.debug("latence: %.2fs, %d appel(s) enregistre(s), erreur=%s", latency, len(recorded), error)
        return recorded, latency, final_text, error

    def ask_continuing(
        self, question: str, messages: list[dict]
    ) -> tuple[list[tuple[str, dict]], float, str | None, str | None]:
        """Comme ask(), mais prolonge l'historique `messages` (modifié sur
        place, message système déjà présent), pour tester la mémoire
        (MULTI_TURN_CASES).

        Copie volontaire du corps d'ask() plutôt qu'une factorisation, pour
        ne pas risquer de fausser les 36 cas mono-tour."""
        recorded: list[tuple[str, dict]] = []
        messages.append({"role": "user", "content": question})
        start = time.time()
        final_text = None
        error = None
        logger.debug("question (historique existant, %d messages) : %r", len(messages) - 1, question)

        try:
            for round_idx in range(self.max_rounds):
                response = self.client.chat.completions.create(
                    model="local",
                    messages=messages,
                    tools=self.tool_schemas,
                    max_tokens=self.max_tokens,
                )
                message = response.choices[0].message

                if not message.tool_calls:
                    final_text = message.content
                    messages.append({"role": "assistant", "content": final_text})
                    logger.debug("round %d: reponse finale -> %r", round_idx, final_text)
                    break

                messages.append(message.model_dump(exclude_none=True))
                for call in message.tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            "round %d: arguments JSON invalides pour '%s': %r",
                            round_idx, call.function.name, call.function.arguments,
                        )
                        arguments = {"_json_error": call.function.arguments}
                    recorded.append((call.function.name, arguments))
                    logger.debug("round %d: tool_call -> %s(%s)", round_idx, call.function.name, arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": self.DUMMY_TOOL_RESULT})
            else:
                logger.warning("max_rounds (%d) atteint sans reponse finale pour %r", self.max_rounds, question)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("echec de l'appel API pour la question %r (round %s)", question, round_idx if "round_idx" in locals() else "?")

        latency = time.time() - start
        logger.debug("latence: %.2fs, %d appel(s) enregistre(s), erreur=%s", latency, len(recorded), error)
        return recorded, latency, final_text, error


def _value_matches(expected, actual) -> bool:
    if expected is ANY:
        return True
    if isinstance(expected, Approx):
        try:
            return abs(float(actual) - expected.target) <= expected.tol
        except (TypeError, ValueError):
            return False
    if isinstance(expected, OneOf):
        return actual in expected.values
    return actual == expected


def _call_matches(expected, actual_name: str, actual_args: dict) -> bool:
    if expected.name != actual_name:
        return False
    for key, expected_value in expected.args.items():
        if key not in actual_args or not _value_matches(expected_value, actual_args[key]):
            return False
    return True


def score_calls(expected_calls: tuple, actual_calls: list[tuple[str, dict]], label: str = "") -> dict:
    """Associe chaque appel attendu à un appel réel, sans tenir compte de
    l'ordre. Réussite stricte : tous les appels attendus trouvés, et aucun
    appel en trop (compter les deux classes quand une seule était demandée
    est une erreur)."""
    remaining = list(actual_calls)
    unmatched_expected = []

    for expected in expected_calls:
        match_idx = next(
            (i for i, (name, args) in enumerate(remaining) if _call_matches(expected, name, args)), None
        )
        if match_idx is not None:
            remaining.pop(match_idx)
        else:
            unmatched_expected.append(expected)

    extra_calls = remaining
    passed = not unmatched_expected and not extra_calls
    if not passed:
        logger.debug(
            "score %s ECHEC -- attendus manquants: %s -- appels superflus: %s",
            label, unmatched_expected, extra_calls,
        )
    return {"passed": passed, "unmatched_expected": unmatched_expected, "extra_calls": extra_calls}


def score_case(case: TestCase, actual_calls: list[tuple[str, dict]]) -> dict:
    """score_calls pour un cas mono-tour."""
    return score_calls(case.expected_calls, actual_calls, label=f"'{case.id}'")
