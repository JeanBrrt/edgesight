"""D1 — Définition des tools de l'agent.

Fait le pont entre les requêtes en langage naturel de l'utilisateur et
`EventStore`/`ZoneMonitor`. Volontairement agnostique du framework de
function-calling (D2) : de simples fonctions Python typées avec
docstring, regroupées comme méthodes d'une classe pour partager une
connexion `EventStore` unique via `self`.

Tous les tools de comptage/statistiques sont généralisés par
`object_class` ("person" ou "car") plutôt que dupliqués par classe comme
dans la première version (qui ne couvrait que "car", sous le nom
"vehicles") -- une seule implémentation, un paramètre en plus.

Paramètre `now` (D2) : CHAQUE méthode l'accepte en optionnel, jamais
exposé au LLM (absent de `tool_schemas.py`) -- c'est `Agent.ask()` qui
l'injecte, avec l'heure de la question de l'utilisateur plutôt que
l'heure d'exécution réelle du tool (voir doc originale du bug de latence
D2). Présent même sur les méthodes qui ne s'en servent pas
fonctionnellement (ex. `average_presence_duration`) : la table de
dispatch d'`Agent._execute_tool` l'injecte inconditionnellement à
chaque appel, donc chaque méthode doit l'accepter pour rester
compatible.
"""

import time
from datetime import date, datetime

from ..journal.event_store import EventStore
from ..alerts.zones import UNKNOWN_ZONE, ZONES
from ._config import AGENT_FRESHNESS_SECONDS

# Libellés des classes (enum `_CLASSES` de tool_schemas.py) pour les
# réponses rédigées en phrase (time_since_last_seen).
_CLASS_LABELS_FR = {"person": "personne", "car": "voiture"}


class FinalAnswer(str):
    """Résultat de tool déjà rédigé comme réponse définitive à
    l'utilisateur : si TOUS les tools d'un tour en renvoient un,
    `Agent.ask()` le renvoie tel quel, sans le faire reformuler par le LLM
    (voir agent.py). Réservé aux cas où la reformulation s'est montrée
    peu fiable -- un str ordinaire reste la règle pour tous les autres
    tools."""


def _format_duration(seconds: float) -> str:
    """Durée lisible pour une FinalAnswer (le LLM ne passe plus derrière
    pour transformer "125 secondes" en "environ 2 minutes")."""
    total = round(seconds)
    if total < 60:
        return f"{total} seconde{'s' if total > 1 else ''}"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s" if secs else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def _parse_time_today(time_str: str, reference_date: date | None = None) -> float:
    """Convertit une heure "HH:MM" en timestamp epoch, ancré sur une date de
    référence (par défaut aujourd'hui) — jamais un autre jour, les
    événements journalisés ne couvrent qu'une session live. `reference_date`
    vient de `now` (l'heure de la question) plutôt que de la date réelle
    d'exécution, pour rester cohérent en cas de question posée juste avant
    minuit.

    Volontairement strict plutôt que permissif : c'est au LLM (D2) de
    normaliser la formulation de l'utilisateur ("midi et quart", "12h15")
    vers ce format avant d'appeler le tool, pas à cette fonction de
    deviner. Piste d'évolution si le besoin de flexibilité grandit :
    remplacer par `dateparser.parse(time_str)`, qui gère nativement les
    expressions relatives/floues sans imposer de format strict au LLM.
    """
    parsed = datetime.strptime(time_str, "%H:%M")
    ref_date = reference_date if reference_date is not None else datetime.now().date()
    return datetime.combine(ref_date, parsed.time()).timestamp()


class AgentTools:
    def __init__(self, store: EventStore):
        self.store = store

    # ------------------------------------------------------------------
    # Comptage (généralisé par classe)
    # ------------------------------------------------------------------

    def count_now(self, object_class: str, now: float | None = None) -> int:
        """Nombre d'objets de cette classe actuellement présents dans le champ de la caméra."""
        return len(self.store.active_tracks(object_class, active_within_seconds=AGENT_FRESHNESS_SECONDS, now=now))

    def count_total(self, object_class: str, now: float | None = None) -> int:
        """Nombre total d'objets distincts de cette classe vus depuis le début de la session."""
        end_ts = now if now is not None else time.time()
        return self.store.count_between(object_class, 0, end_ts)

    def count_since(
        self,
        object_class: str,
        start_time: str | None = None,
        minutes_ago: float | None = None,
        now: float | None = None,
    ) -> int:
        """Nombre d'objets distincts de cette classe vus depuis un point de
        départ jusqu'à maintenant. Fournir EXACTEMENT un des deux :
        `start_time` (heure absolue "HH:MM", ex. "depuis 12h15") ou
        `minutes_ago` (durée relative, ex. "depuis 10 minutes").

        Le calcul de la borne de fin ("maintenant") se fait ici, côté code,
        jamais côté LLM -- lui faire convertir une heure absolue en délai
        (ou l'inverse) l'obligerait à connaître l'heure actuelle, alors que
        `now` n'est justement jamais exposé dans tool_schemas.py (cf.
        docstring du module).
        """
        if (start_time is None) == (minutes_ago is None):
            raise ValueError("Fournir exactement un des deux : start_time ou minutes_ago.")
        end_ts = now if now is not None else time.time()
        if start_time is not None:
            reference_date = datetime.fromtimestamp(end_ts).date()
            start_ts = _parse_time_today(start_time, reference_date)
        else:
            start_ts = end_ts - minutes_ago * 60
        return self.store.count_between(object_class, start_ts, end_ts)

    def count_between(self, object_class: str, start_time: str, end_time: str, now: float | None = None) -> int:
        """Nombre d'objets distincts de cette classe vus entre deux horaires
        de la journée en cours.

        Args:
            start_time: heure de début, format "HH:MM" (ex. "12:15")
            end_time: heure de fin, format "HH:MM" (ex. "12:30")
        """
        reference_date = datetime.fromtimestamp(now).date() if now is not None else None
        start_ts = _parse_time_today(start_time, reference_date)
        end_ts = _parse_time_today(end_time, reference_date)
        return self.store.count_between(object_class, start_ts, end_ts)

    # ------------------------------------------------------------------
    # Statistiques dérivées
    # ------------------------------------------------------------------

    def time_since_last_seen(self, object_class: str, now: float | None = None) -> str:
        """Depuis combien de temps aucun objet de cette classe n'a été
        détecté -- basé sur la DERNIÈRE détection (last_seen), avec la même
        tolérance de fraîcheur que count_now, pour que les deux tools ne se
        contredisent jamais ("1 voiture maintenant" mais "pas vue depuis
        3s").

        Renvoie une `FinalAnswer` (phrase définitive, jamais reformulée par
        le LLM) : testé avec Granite-4.1-3B sur "depuis combien de temps on
        n'a pas vu de voiture ?" alors qu'une voiture est visible, la
        reformulation reprenait la négation de la question et inversait le
        sens du résultat ("aucune voiture n'est actuellement visible") dans
        ~1 cas sur 3 -- quelle que soit la forme du résultat (nombre,
        booléen, phrase) ou une consigne ajoutée au prompt système."""
        label = _CLASS_LABELS_FR.get(object_class, object_class)
        last_seen = self.store.most_recent_last_seen(object_class)
        if last_seen is None:
            return FinalAnswer(f"Aucune {label} n'a été détectée depuis le début de la session.")
        ref_now = now if now is not None else time.time()
        elapsed = max(0.0, ref_now - last_seen)
        if elapsed <= AGENT_FRESHNESS_SECONDS:
            return FinalAnswer(f"Une {label} est visible en ce moment : la dernière détection date d'à l'instant.")
        return FinalAnswer(
            f"Aucune {label} n'est visible en ce moment : la dernière a été détectée il y a {_format_duration(elapsed)}."
        )

    def average_presence_duration(self, object_class: str, now: float | None = None) -> float | None:
        """Durée moyenne de présence (secondes) des objets de cette classe,
        depuis le début de la session (None si aucun objet jamais vu)."""
        return self.store.average_presence_duration(object_class)

    def peak_concurrent_count(self, object_class: str, now: float | None = None) -> int:
        """Nombre maximal d'objets de cette classe présents simultanément à
        un instant quelconque depuis le début de la session."""
        return self.store.peak_concurrent_count(object_class)

    # ------------------------------------------------------------------
    # Alertes (configuration uniquement -- la surveillance elle-même est
    # `AlertMonitor`/`ZoneMonitor`, hors périmètre agent)
    # ------------------------------------------------------------------

    def set_presence_duration_alert(self, object_class: str, threshold_seconds: float, now: float | None = None) -> str:
        """Configure une alerte si un objet de cette classe reste présent en
        continu plus longtemps que `threshold_seconds` secondes."""
        self.store.set_duration_alert(object_class, threshold_seconds, timestamp=now)
        return f"Alerte configurée : notification si un(e) {object_class} reste plus de {threshold_seconds:.0f}s dans le champ."

    def set_co_occurrence_alert(self, person_threshold: int, car_threshold: int, now: float | None = None) -> str:
        """Configure une alerte si au moins `person_threshold` personnes ET
        au moins `car_threshold` voitures sont présentes simultanément."""
        self.store.set_co_occurrence_alert(person_threshold, car_threshold, timestamp=now)
        return (
            f"Alerte configurée : notification si au moins {person_threshold} personne(s) et "
            f"{car_threshold} véhicule(s) sont présents simultanément."
        )

    def set_surge_alert(self, object_class: str, count_threshold: int, window_minutes: float, now: float | None = None) -> str:
        """Configure une alerte si `count_threshold` objets de cette classe
        ou plus apparaissent en moins de `window_minutes` minutes."""
        self.store.set_surge_alert(object_class, count_threshold, window_minutes, timestamp=now)
        return (
            f"Alerte configurée : notification si {count_threshold} {object_class}(s) ou plus "
            f"apparaissent en moins de {window_minutes:.0f} minutes."
        )

    def set_zone_alert(self, zone_name: str, object_class: str, now: float | None = None) -> str:
        """Configure une alerte si un objet de cette classe (ou 'any' pour
        les deux) entre dans la zone de danger nommée."""
        if zone_name == UNKNOWN_ZONE:
            return self._unknown_zone_message()
        if zone_name not in ZONES:
            # Défensif : ne devrait jamais se produire, `zone_name` est
            # contraint par l'enum du schéma (tool_schemas.py) aux zones
            # réelles + UNKNOWN_ZONE, déjà géré ci-dessus.
            raise ValueError(f"Zone inconnue : '{zone_name}'. Zones disponibles : {', '.join(ZONES)}.")
        self.store.set_zone_alert(zone_name, object_class, timestamp=now)
        return f"Alerte configurée : notification si un(e) {object_class} entre dans la zone '{zone_name}'."

    def count_zone_entries(self, zone_name: str, object_class: str, now: float | None = None) -> int | str:
        """Nombre d'entrées d'objets de cette classe (ou 'any') dans la zone
        nommée, depuis le début de la session."""
        if zone_name == UNKNOWN_ZONE:
            return self._unknown_zone_message()
        if zone_name not in ZONES:
            # Défensif : voir set_zone_alert ci-dessus.
            raise ValueError(f"Zone inconnue : '{zone_name}'. Zones disponibles : {', '.join(ZONES)}.")
        return self.store.count_zone_entries(zone_name, object_class)

    @staticmethod
    def _unknown_zone_message() -> str:
        """Réponse pour `zone_name=UNKNOWN_ZONE` (tool_schemas.py) -- PAS une
        erreur (pas de raise) : c'est le comportement voulu quand le LLM
        signale correctement qu'aucune zone connue ne correspond à la
        demande de l'utilisateur, plutôt que d'halluciner la zone réelle la
        plus proche. Réinjectée telle quelle dans l'historique de la
        conversation (Agent._execute_tool), pour que le LLM la reformule en
        clarification adressée à l'utilisateur."""
        return (
            "Aucune zone connue ne correspond à cette description. "
            f"Zones disponibles : {', '.join(ZONES)}."
        )
