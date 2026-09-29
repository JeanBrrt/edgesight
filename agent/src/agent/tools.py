"""Outils de l'agent
"""

import time
from datetime import date, datetime

from ..journal.event_store import EventStore
from ..alerts.zones import UNKNOWN_ZONE, ZONES
from ._config import AGENT_FRESHNESS_SECONDS

_CLASS_LABELS_FR = {"person": "personne", "car": "voiture"}


class FinalAnswer(str):
    """Réponse déjà rédigée, renvoyée telle quelle à l'utilisateur sans
    reformulation par le LLM (voir agent.py). Réservée aux cas où la
    reformulation s'est montrée peu fiable."""


def _parse_time_today(time_str: str, reference_date: date | None = None) -> float:
    """Heure "HH:MM" -> timestamp, le jour de `reference_date` (par défaut
    aujourd'hui). Format strict : c'est au LLM de convertir "midi et
    quart" en "12:15"."""
    parsed = datetime.strptime(time_str, "%H:%M")
    ref_date = reference_date if reference_date is not None else datetime.now().date()
    return datetime.combine(ref_date, parsed.time()).timestamp()


class AgentTools:
    def __init__(self, store: EventStore):
        self.store = store

    # ------------------------------------------------------------------
    # Comptage
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
        """Nombre d'objets distincts vus depuis `start_time` ("HH:MM") ou
        depuis `minutes_ago` minutes (un seul des deux). "Maintenant" est
        calculé ici : le LLM ne connaît pas l'heure."""
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
        """Nombre d'objets distincts vus entre deux heures "HH:MM" du jour."""
        reference_date = datetime.fromtimestamp(now).date() if now is not None else None
        start_ts = _parse_time_today(start_time, reference_date)
        end_ts = _parse_time_today(end_time, reference_date)
        return self.store.count_between(object_class, start_ts, end_ts)

    # ------------------------------------------------------------------
    # Statistiques dérivées
    # ------------------------------------------------------------------

    def time_since_last_seen(self, object_class: str, now: float | None = None) -> int | str:
        """Secondes écoulées depuis la dernière détection de cette classe,
        avec la même tolérance que count_now pour ne pas le contredire.

        Si un objet est visible (ou jamais vu), renvoie une FinalAnswer : à
        partir d'un 0, Granite-4.1-3B répondait "aucune voiture visible"
        10 fois sur 10. Le nombre de secondes, lui, est bien reformulé."""
        label = _CLASS_LABELS_FR.get(object_class, object_class)
        last_seen = self.store.most_recent_last_seen(object_class)
        if last_seen is None:
            return FinalAnswer(f"Aucune {label} n'a été détectée depuis le début de la session.")
        ref_now = now if now is not None else time.time()
        elapsed = max(0.0, ref_now - last_seen)
        if elapsed <= AGENT_FRESHNESS_SECONDS:
            return FinalAnswer(f"Une {label} est visible en ce moment.")
        return round(elapsed)

    def average_presence_duration(self, object_class: str, now: float | None = None) -> float | None:
        """Durée moyenne de présence en secondes (None si rien n'a été vu)."""
        return self.store.average_presence_duration(object_class)

    def peak_concurrent_count(self, object_class: str, now: float | None = None) -> int:
        """Nombre maximal d'objets présents en même temps depuis le début."""
        return self.store.peak_concurrent_count(object_class)

    # ------------------------------------------------------------------
    # Alertes (pose seulement ; la surveillance est dans agent/src/alerts/)
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
            # Normalement impossible : le schéma limite zone_name aux zones connues.
            raise ValueError(f"Zone inconnue : '{zone_name}'. Zones disponibles : {', '.join(ZONES)}.")
        self.store.set_zone_alert(zone_name, object_class, timestamp=now)
        return f"Alerte configurée : notification si un(e) {object_class} entre dans la zone '{zone_name}'."

    def count_zone_entries(self, zone_name: str, object_class: str, now: float | None = None) -> int | str:
        """Nombre d'entrées d'objets de cette classe (ou 'any') dans la zone
        nommée, depuis le début de la session."""
        if zone_name == UNKNOWN_ZONE:
            return self._unknown_zone_message()
        if zone_name not in ZONES:
            # Normalement impossible (voir set_zone_alert).
            raise ValueError(f"Zone inconnue : '{zone_name}'. Zones disponibles : {', '.join(ZONES)}.")
        return self.store.count_zone_entries(zone_name, object_class)

    @staticmethod
    def _unknown_zone_message() -> str:
        """Réponse quand le LLM signale qu'aucune zone ne correspond (plutôt
        que d'inventer la plus proche) : pas une erreur, le LLM la
        reformule en demande de précision."""
        return (
            "Aucune zone connue ne correspond à cette description. "
            f"Zones disponibles : {', '.join(ZONES)}."
        )
