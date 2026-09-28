"""Moteur du banc de test LLM : rejoue la boucle de function-calling
d'`Agent.ask()` (agent/src/agent/agent.py) SANS exécuter réellement les tools,
et note si les appels effectués correspondent à ceux attendus
(cases.py).

Découplage volontaire de l'exécution réelle : ce banc évalue la capacité
du LLM à CHOISIR le bon tool avec les bons arguments à partir d'une
question en langage naturel -- pas le comportement d'EventStore/
AlertMonitor/ZoneMonitor, déjà couverts par un test fonctionnel séparé.
Pas besoin d'un EventStore réel ni de données de scénario pour ça : un
résultat de tool factice suffit à laisser le modèle poursuivre jusqu'à
une réponse finale.
"""

import json
import logging
import time

from cases import ANY, Approx, OneOf, TestCase

logger = logging.getLogger(__name__)


class RecordingAgent:
    """Comme `Agent` (agent/src/agent/agent.py), mais chaque tool_call est
    enregistré (nom + arguments parsés) plutôt qu'exécuté. Renvoie un
    résultat factice constant en réponse à chaque appel -- jamais
    interprété par le scoring, seulement là pour permettre au modèle de
    continuer la conversation jusqu'à un texte final."""

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
        # Sans plafond, `llama-server` génère par défaut sans limite
        # (max_tokens=-1 côté serveur) -- observé en pratique : un modèle
        # qui part en boucle sur une question difficile (ex. le cas
        # multi__compare_now) a continué à générer au-delà de 4000 tokens
        # pour une seule réponse, sans jamais atteindre de token de fin.
        # 1024 est très large pour un tool call + une courte phrase de
        # réponse -- un modèle qui en a besoin de plus est déjà en train
        # de dérailler, pas juste verbeux.
        self.max_tokens = max_tokens

    def ask(self, question: str) -> tuple[list[tuple[str, dict]], float, str | None, str | None]:
        """Renvoie (appels_enregistrés, latence_secondes, texte_final_ou_None,
        erreur_ou_None).

        `erreur` n'est renseignée QUE si l'appel API lui-même a échoué
        (serveur indisponible, requête rejetée, réponse mal formée) --
        distinct d'un mauvais choix de tool par le LLM, qui n'est pas une
        erreur au sens de cette méthode (c'est le rôle de `score_case`).
        Ne lève JAMAIS d'exception : un serveur qui plante sur une seule
        question ne doit pas interrompre tout le banc de test (`traceback`
        complet tout de même écrit dans le log DEBUG via `logger.exception`,
        pour pouvoir diagnostiquer après coup)."""
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
        """Comme `ask()` ci-dessus, mais POURSUIT un historique existant
        (`messages`, modifié EN PLACE) au lieu d'en reconstruire un neuf --
        pour les cas multi-tours (cases.py:MULTI_TURN_CASES), qui testent
        la mémoire conversationnelle (agent/src/agent/agent.py:Session).

        Duplique volontairement le corps de `ask()` plutôt que de le
        factoriser : `ask()` reste intouchée, donc aucun risque de
        régression sur les 36 cas mono-tour existants qui en dépendent
        (run_benchmark.py) -- le prix (un peu de code répété) est jugé
        largement inférieur au risque d'un refactor qui casserait
        silencieusement le banc de test principal.

        `messages` doit déjà contenir au moins le message système (voir
        run_memory_benchmark.py, qui l'initialise une fois par séquence,
        avant le premier appel)."""
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
    """Appariement glouton, indépendant de l'ordre : chaque appel attendu
    cherche un appel réel correspondant parmi ceux encore disponibles.
    `passed` est STRICT : tous les appels attendus doivent être trouvés
    ET aucun appel réel ne doit rester non apparié (un tool superflu,
    ex. compter les deux classes quand une seule était demandée, est une
    vraie erreur de comportement, pas un détail cosmétique).

    Logique partagée par `score_case` (cas mono-tour, ALL_CASES) et le
    scoring par tour des cas multi-tours (MULTI_TURN_CASES,
    run_memory_benchmark.py) -- extraite ici pour ne pas dupliquer
    l'appariement lui-même, seulement les boucles qui l'appellent tour
    par tour restent séparées (voir docstring d'`ask_continuing`)."""
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
    """Cas mono-tour (ALL_CASES) -- fine couche au-dessus de `score_calls`
    pour garder EXACTEMENT la même signature qu'avant cette extraction
    (aucun appelant, ex. run_benchmark.py, n'a besoin d'être modifié)."""
    return score_calls(case.expected_calls, actual_calls, label=f"'{case.id}'")
