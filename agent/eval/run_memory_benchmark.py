"""Banc de test de la mémoire conversationnelle 

Usage :
    uv run python agent/eval/run_memory_benchmark.py --model-label granite-4.1-3b
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, ".")
from agent.src.agent.tool_schemas import TOOL_SCHEMAS  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cases import MULTI_TURN_CASES  # noqa: E402
from harness import RecordingAgent, score_calls  # noqa: E402

logger = logging.getLogger(__name__)

# Prompt système lu dans config/agent.yaml : le même que la démo.
_AGENT_CONFIG_PATH = Path("config/agent.yaml")
with open(_AGENT_CONFIG_PATH, encoding="utf-8") as _f:
    _AGENT_CONFIG = yaml.safe_load(_f)
SYSTEM_PROMPT = _AGENT_CONFIG["system_prompt"]


def _setup_logging(log_path: Path) -> None:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-label", required=True, help="Nom du modèle testé (nom du fichier résultat).")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="URL du serveur llama-server (compatible OpenAI).")
    parser.add_argument("--out-dir", default="agent/eval/results")
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Plafond de tokens générés par réponse (comme run_benchmark.py).",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout HTTP par requête, en secondes.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.model_label}-memory.log"
    out_path = out_dir / f"{args.model_label}-memory.json"
    _setup_logging(log_path)
    logger.info(
        "Modèle : %s -- serveur : %s -- %d séquence(s) multi-tours (max_tokens=%d, timeout=%.0fs)",
        args.model_label, args.url, len(MULTI_TURN_CASES), args.max_tokens, args.timeout,
    )

    client = OpenAI(base_url=args.url, api_key="not-needed", timeout=args.timeout)
    agent = RecordingAgent(client, TOOL_SCHEMAS, SYSTEM_PROMPT, max_tokens=args.max_tokens)

    case_results = []
    for case in MULTI_TURN_CASES:
        # Historique neuf pour chaque séquence.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        turn_results = []
        sequence_passed = True

        for turn_idx, turn in enumerate(case.turns):
            recorded, latency, final_text, error = agent.ask_continuing(turn.question, messages)
            if error is not None:
                outcome = {"passed": False, "unmatched_expected": list(turn.expected_calls), "extra_calls": []}
            else:
                outcome = score_calls(turn.expected_calls, recorded, label=f"'{case.id}' tour {turn_idx}")
            sequence_passed = sequence_passed and outcome["passed"]
            turn_results.append(
                {
                    "turn": turn_idx,
                    "question": turn.question,
                    "passed": outcome["passed"],
                    "recorded_calls": recorded,
                    "unmatched_expected": [e.name for e in outcome["unmatched_expected"]],
                    "extra_calls": outcome["extra_calls"],
                    "final_text": final_text,
                    "latency_s": latency,
                    "error": error,
                }
            )
            status = "OK" if outcome["passed"] else "FAIL"
            log_fn = logger.info if status == "OK" else logger.warning
            suffix = f" -- ERREUR: {error}" if error else ""
            log_fn("  [%s] tour %d : %-55s%s", status, turn_idx, turn.question, suffix)

            if error is not None:
                # Échec de l'API : la suite de la séquence dépendrait d'un
                # historique incomplet, on s'arrête là.
                break

        case_results.append(
            {"id": case.id, "notes": case.notes, "passed": sequence_passed, "turns": turn_results}
        )
        overall_status = "OK" if sequence_passed else "FAIL"
        logger.info("[%s] %s (%d tours)", overall_status, case.id, len(case.turns))

        out_path.write_text(
            json.dumps({"summary": None, "cases": case_results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "model_label": args.model_label,
        "n_sequences": len(MULTI_TURN_CASES),
        "n_sequences_passed": sum(1 for c in case_results if c["passed"]),
    }
    out_path.write_text(json.dumps({"summary": summary, "cases": case_results}, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("")
    logger.info("=== %s (mémoire) ===", args.model_label)
    logger.info(
        "%d/%d séquences réussies intégralement (tous les tours corrects)",
        summary["n_sequences_passed"], summary["n_sequences"],
    )
    logger.info("Résultats écrits dans %s", out_path)
    logger.info("Log détaillé écrit dans %s", log_path)


if __name__ == "__main__":
    main()
