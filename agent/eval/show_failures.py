"""Affiche les échecs d'un passage de run_benchmark.py : pour chaque cas
pas toujours réussi, ce que le LLM a appelé au lieu de ce qui était
attendu. Distingue les erreurs d'API (serveur) des mauvais choix d'outil.

Usage :
    uv run python agent/eval/show_failures.py agent/eval/results/hermes-3-8b.json
    uv run python agent/eval/show_failures.py agent/eval/results/hermes-3-8b.json --only-errors
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_file", help="Fichier JSON produit par run_benchmark.py")
    parser.add_argument(
        "--only-errors", action="store_true", help="N'afficher que les erreurs API, pas les mauvais choix de tool."
    )
    args = parser.parse_args()

    data = json.loads(Path(args.result_file).read_text(encoding="utf-8"))
    summary = data["summary"]
    print(f"Modèle : {summary['model_label']} -- réussite globale {summary['overall_pass_rate']:.1%}", end="")
    if summary.get("n_errored_runs"):
        print(f" -- {summary['n_errored_runs']} essai(s) en erreur API")
    else:
        print()
    print()

    n_shown = 0
    for case in data["cases"]:
        if case["pass_rate"] == 1.0:
            continue
        has_error = any(run["error"] for run in case["runs"])
        if args.only_errors and not has_error:
            continue

        n_shown += 1
        print(f"--- {case['id']} (réussite {case['pass_rate']:.0%}) ---")
        print(f"Question : {case['question']}")
        if case["notes"]:
            print(f"Note     : {case['notes']}")

        for i, run in enumerate(case["runs"]):
            tag = "OK" if run["passed"] else ("ERREUR API" if run["error"] else "ECHEC")
            print(f"  essai {i + 1} [{tag}]")
            if run["error"]:
                print(f"    -> {run['error']}")
                continue
            if run["passed"]:
                continue
            print(f"    appels reçus       : {run['recorded_calls']}")
            if run["unmatched_expected"]:
                print(f"    attendus manquants : {run['unmatched_expected']}")
            if run["extra_calls"]:
                print(f"    appels superflus   : {run['extra_calls']}")
            print(f"    réponse finale     : {run['final_text']!r}")
        print()

    if n_shown == 0:
        message = "Aucune erreur API trouvée." if args.only_errors else "Aucun échec -- tous les cas passent à 100%."
        print(message)


if __name__ == "__main__":
    main()
