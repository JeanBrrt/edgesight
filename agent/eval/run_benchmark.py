"""Banc de test custom : mesure la capacité d'un LLM servi par
llama-server à choisir/paramétrer les tools de l'agent (agent/src/
tool_schemas.py), sur le jeu de questions de cases.py (couvre les 12
tools + variantes/pièges : synonymes, anglais, désambiguïsation
now/total, formulation de temps absolue/relative, appels multiples,
zone inconnue, hors périmètre).

Ne teste PAS l'exécution réelle des tools (EventStore/AlertMonitor/
ZoneMonitor -- déjà couverts par un test fonctionnel séparé), seulement
le comportement du LLM face au langage naturel.

Chaque question est rejouée `--repeats` fois (défaut 10) : la
sortie d'un LLM n'est pas déterministe d'un tick à l'autre (sampling),
un cas qui échoue une fois sur trois n'est pas équivalent à un cas qui
échoue toujours -- distinction perdue par un seul essai.

Usage :
    1. Lancer llama-server avec le modèle à tester (voir README pour la
       commande complète), ex. :
         llama-server.exe -m <chemin_du_modele>.gguf -c 4096 --jinja
    2. Depuis la racine du projet :
         uv run python agent/eval/run_benchmark.py --model-label qwen2.5-7b

Écrit agent/eval/results/<model-label>.json -- à comparer ensuite avec
compare_models.py une fois plusieurs modèles benchmarkés.
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, ".")
from agent.src.agent.tool_schemas import TOOL_SCHEMAS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cases import ALL_CASES  # noqa: E402
from harness import RecordingAgent, score_case  # noqa: E402

logger = logging.getLogger(__name__)

# Même prompt système que la production (config/agent.yaml) -- pas
# recopié en dur, pour ne jamais tester un prompt différent de celui
# réellement déployé.
_AGENT_CONFIG_PATH = Path("config/agent.yaml")
with open(_AGENT_CONFIG_PATH, encoding="utf-8") as _f:
    _AGENT_CONFIG = yaml.safe_load(_f)
SYSTEM_PROMPT = _AGENT_CONFIG["system_prompt"]


def _setup_logging(log_path: Path) -> None:
    """Console à INFO (résumé lisible, ce qui s'affichait déjà avant) +
    fichier à DEBUG (question par question : tool_calls bruts, latence par
    round, raison précise d'un échec -- cf. harness.py) sur le logger
    racine, pour que les deux modules (celui-ci + harness.py) y écrivent
    sans configuration séparée."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(file_handler)


def _build_summary(model_label: str, repeats: int, case_results: list[dict], complete: bool) -> dict:
    all_runs = [run for r in case_results for run in r["runs"]]
    all_latencies = [run["latency_s"] for run in all_runs]
    sorted_latencies = sorted(all_latencies)
    return {
        "model_label": model_label,
        "repeats": repeats,
        "n_cases_total": len(ALL_CASES),
        "n_cases_done": len(case_results),
        "complete": complete,  # False si écrit après une interruption (Ctrl+C) -- résultats partiels
        "overall_pass_rate": sum(r["pass_rate"] for r in case_results) / len(case_results) if case_results else 0.0,
        "n_errored_runs": sum(1 for run in all_runs if run["error"]),
        "latency_mean_s": statistics.mean(all_latencies) if all_latencies else 0.0,
        "latency_median_s": statistics.median(all_latencies) if all_latencies else 0.0,
        "latency_p95_s": sorted_latencies[max(0, int(0.95 * len(sorted_latencies)) - 1)] if sorted_latencies else 0.0,
    }


def _write_results(out_path: Path, model_label: str, repeats: int, case_results: list[dict], complete: bool) -> None:
    summary = _build_summary(model_label, repeats, case_results, complete)
    out_path.write_text(
        json.dumps({"summary": summary, "cases": case_results}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", required=True, help="Nom du modèle testé (nom du fichier résultat).")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="URL du serveur llama-server (compatible OpenAI).")
    parser.add_argument("--repeats", type=int, default=10, help="Répétitions par question (variance d'échantillonnage).")
    parser.add_argument("--out-dir", default="agent/eval/results")
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Plafond de tokens générés par réponse -- garde-fou contre une génération qui part en boucle (défaut serveur souvent illimité).",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Timeout HTTP par requête, en secondes -- un modèle qui ne répond plus doit échouer proprement, pas geler le banc de test.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.model_label}.log"
    out_path = out_dir / f"{args.model_label}.json"
    _setup_logging(log_path)
    logger.info(
        "Modèle : %s -- serveur : %s -- %d cas x %d répétitions (max_tokens=%d, timeout=%.0fs)",
        args.model_label, args.url, len(ALL_CASES), args.repeats, args.max_tokens, args.timeout,
    )
    logger.debug("Prompt système utilisé : %r", SYSTEM_PROMPT)

    client = OpenAI(base_url=args.url, api_key="not-needed", timeout=args.timeout)
    agent = RecordingAgent(client, TOOL_SCHEMAS, SYSTEM_PROMPT, max_tokens=args.max_tokens)

    case_results = []
    interrupted = False
    try:
        for case in ALL_CASES:
            runs = []
            for _ in range(args.repeats):
                recorded, latency, final_text, error = agent.ask(case.question)
                if error is not None:
                    # Un échec de transport/API n'est pas un mauvais choix de
                    # tool -- pas la peine de faire scorer par score_case, qui
                    # comparerait des appels vides aux attendus (toujours faux,
                    # mais pour la mauvaise raison).
                    outcome = {"passed": False, "unmatched_expected": list(case.expected_calls), "extra_calls": []}
                else:
                    outcome = score_case(case, recorded)
                runs.append(
                    {
                        "passed": outcome["passed"],
                        "latency_s": latency,
                        "recorded_calls": recorded,
                        "unmatched_expected": [e.name for e in outcome["unmatched_expected"]],
                        "extra_calls": outcome["extra_calls"],
                        "final_text": final_text,
                        "error": error,
                    }
                )
            n_passed = sum(r["passed"] for r in runs)
            n_errored = sum(1 for r in runs if r["error"])
            pass_rate = n_passed / args.repeats
            case_results.append(
                {"id": case.id, "question": case.question, "notes": case.notes, "runs": runs, "pass_rate": pass_rate}
            )
            if n_errored == args.repeats:
                status = "ERROR"
            elif n_passed == args.repeats:
                status = "OK"
            elif n_passed == 0:
                status = "FAIL"
            else:
                status = "FLAKY"
            log_fn = logger.info if status == "OK" else logger.warning
            suffix = f" -- ERREUR: {next(r['error'] for r in runs if r['error'])}" if n_errored else ""
            log_fn("[%s] %-45s %d/%d  -- %s%s", status, case.id, n_passed, args.repeats, case.question, suffix)

            # Écrit après CHAQUE cas, pas seulement à la fin -- une
            # interruption (Ctrl+C sur une génération qui part en boucle,
            # cf. le cas multi__compare_now) ne doit pas faire perdre tout
            # le run déjà accompli.
            _write_results(out_path, args.model_label, args.repeats, case_results, complete=False)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrompu par l'utilisateur -- %d/%d cas déjà traités, résultats partiels conservés.", len(case_results), len(ALL_CASES))

    _write_results(out_path, args.model_label, args.repeats, case_results, complete=not interrupted)
    summary = _build_summary(args.model_label, args.repeats, case_results, complete=not interrupted)

    logger.info("")
    logger.info("=== %s%s ===", args.model_label, " (PARTIEL)" if interrupted else "")
    logger.info("Taux de réussite global : %.1f%% (%d/%d cas)", summary["overall_pass_rate"] * 100, summary["n_cases_done"], summary["n_cases_total"])
    if summary["n_errored_runs"]:
        logger.warning(
            "%d essai(s) ont échoué au niveau API (pas un mauvais choix de tool) -- voir le log détaillé",
            summary["n_errored_runs"],
        )
    logger.info("Latence médiane : %.2fs (P95 : %.2fs)", summary["latency_median_s"], summary["latency_p95_s"])
    logger.info("Résultats écrits dans %s", out_path)
    logger.info("Log détaillé écrit dans %s", log_path)
    logger.info("Pour analyser les échecs en détail : uv run python agent/eval/show_failures.py %s", out_path)


if __name__ == "__main__":
    main()
