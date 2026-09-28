"""Tableau comparatif de plusieurs runs de run_benchmark.py
(agent/eval/results/*.json, un fichier par modèle) -- réussite globale,
latence, et détail de réussite par tool.

Usage (depuis la racine du projet), une fois au moins 2 fichiers
présents dans agent/eval/results/ :
    uv run python agent/eval/compare_models.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _tool_name_from_case_id(case_id: str) -> str:
    # Convention cases.py : id = "<nom_du_tool>__<variante>" (ou
    # "no_call"/"multi"/"disambiguation" pour les cas transverses).
    return case_id.split("__", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="agent/eval/results")
    args = parser.parse_args()

    files = sorted(Path(args.results_dir).glob("*.json"))
    if not files:
        print(f"Aucun résultat trouvé dans {args.results_dir} -- lancer run_benchmark.py d'abord.")
        return

    summaries = []
    per_tool_by_model: dict[str, dict[str, float]] = {}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        summaries.append(data["summary"])
        per_tool: dict[str, list[float]] = defaultdict(list)
        for case in data["cases"]:
            per_tool[_tool_name_from_case_id(case["id"])].append(case["pass_rate"])
        per_tool_by_model[data["summary"]["model_label"]] = {
            tool: sum(rates) / len(rates) for tool, rates in per_tool.items()
        }

    print(f"{'Modèle':<28}{'Réussite globale':>18}{'Latence médiane':>18}{'P95':>10}")
    print("-" * 74)
    for s in summaries:
        print(
            f"{s['model_label']:<28}{s['overall_pass_rate']:>17.1%} "
            f"{s['latency_median_s']:>16.2f}s {s['latency_p95_s']:>8.2f}s"
        )

    all_tools = sorted({tool for per_tool in per_tool_by_model.values() for tool in per_tool})
    model_labels = [s["model_label"] for s in summaries]

    print("\nDétail de réussite par tool :")
    header = f"{'Tool':<28}" + "".join(f"{label:>18}" for label in model_labels)
    print(header)
    print("-" * len(header))
    for tool in all_tools:
        line = f"{tool:<28}"
        for label in model_labels:
            rate = per_tool_by_model[label].get(tool)
            line += f"{rate:>17.0%} " if rate is not None else f"{'--':>18}"
        print(line)


if __name__ == "__main__":
    main()
