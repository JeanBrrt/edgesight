"""Utilitaires partagés par check_demo.py et check_project.py -- un
format de vérification commun (nom, statut, détail) et un affichage
uniforme, pour ne pas dupliquer la logique d'affichage entre les deux
scripts. Jamais importé en dehors de setup/ (résolu via le sys.path
auto-ajouté par Python pour le script lancé directement, comme
detection/src/ -- pas besoin de sys.path.insert ici)."""

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    # Un échec `optional=True` s'affiche (pour rester visible) mais ne fait
    # jamais échouer le script dans son ensemble -- utilisé pour tout ce
    # qui dépend d'un choix de l'utilisateur (GPU disponible, LLM local
    # installé) plutôt que d'une étape de setup oubliée.
    optional: bool = False


def check_file(path: str, label: str | None = None) -> CheckResult:
    label = label or path
    ok = Path(path).exists()
    return CheckResult(label, ok, path if ok else f"introuvable : {path}")


def check_import(module_name: str, label: str | None = None, optional: bool = False) -> CheckResult:
    label = label or module_name
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        return CheckResult(label, False, f"import {module_name} échoue : {exc}", optional=optional)
    return CheckResult(label, True, f"import {module_name} OK", optional=optional)


def run(title: str, results: list[CheckResult]) -> bool:
    """Affiche le rapport d'une section ; renvoie False seulement si au
    moins un échec NON optionnel est présent (les échecs `optional=True`
    s'affichent comme avertissement, jamais comme blocage)."""
    print(f"\n=== {title} ===")
    blocking_failed = False
    for r in results:
        if r.ok:
            status = "OK"
        elif r.optional:
            status = "absent (optionnel)"
        else:
            status = "MANQUANT"
            blocking_failed = True
        print(f"  [{status}] {r.name}")
        if not r.ok:
            print(f"        -> {r.detail}")
    return not blocking_failed
