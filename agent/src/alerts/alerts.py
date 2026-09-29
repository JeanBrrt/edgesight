"""Surveillance des alertes posées par l'agent
"""

import time

from ..journal.event_store import EventStore


class AlertMonitor:
    def __init__(self, store: EventStore):
        self.store = store
        self._notified_duration: set[tuple[str, int]] = set()
        self._co_occurrence_active = False
        self._surge_active: set[str] = set()

    def reset(self):
        """À appeler au changement de vidéo : les identifiants de piste
        repartent de 0 et ne doivent pas hériter de "déjà notifié"."""
        self._notified_duration.clear()
        self._co_occurrence_active = False
        self._surge_active.clear()

    def check(self, now: float | None = None) -> list[dict]:
        """À appeler à chaque image. Renvoie les alertes qui viennent de se
        déclencher (souvent aucune), chacune avec `alert_type` et `detail`."""
        now = now if now is not None else time.time()
        newly_fired = []
        alerts = self.store.get_alerts()

        # --- duration ---------------------------------------------------
        still_active_keys = set()
        for rule in (a for a in alerts if a["alert_type"] == "duration"):
            class_name, threshold = rule["object_class"], rule["threshold_seconds"]
            for track_id, duration in self.store.active_durations(class_name, now=now):
                key = (class_name, track_id)
                still_active_keys.add(key)  # active, franchisse le seuil ou non
                if duration >= threshold and key not in self._notified_duration:
                    self._notified_duration.add(key)
                    newly_fired.append(
                        {
                            "alert_type": "duration",
                            "object_class": class_name,
                            "track_id": track_id,
                            "detail": f"{class_name} #{track_id} présent(e) depuis {duration:.1f}s",
                        }
                    )
        # Oublie les pistes disparues.
        self._notified_duration &= still_active_keys

        # --- co_occurrence (au plus une règle active, cf. event_store.py) ---
        co_rule = next((a for a in alerts if a["alert_type"] == "co_occurrence"), None)
        if co_rule is not None:
            n_person = len(self.store.active_tracks("person", now=now))
            n_car = len(self.store.active_tracks("car", now=now))
            condition = n_person >= co_rule["person_threshold"] and n_car >= co_rule["car_threshold"]
            if condition and not self._co_occurrence_active:
                newly_fired.append(
                    {
                        "alert_type": "co_occurrence",
                        "detail": f"{n_person} personne(s) et {n_car} véhicule(s) présents simultanément",
                    }
                )
            self._co_occurrence_active = condition
        else:
            self._co_occurrence_active = False

        # --- surge (une règle par classe) --------------------------------
        surge_rules = {a["object_class"]: a for a in alerts if a["alert_type"] == "surge"}
        still_surging = set()
        for class_name, rule in surge_rules.items():
            window_start = now - rule["window_minutes"] * 60
            count = self.store.count_between(class_name, window_start, now)
            if count >= rule["count_threshold"]:
                still_surging.add(class_name)
                if class_name not in self._surge_active:
                    newly_fired.append(
                        {
                            "alert_type": "surge",
                            "object_class": class_name,
                            "detail": f"{count} {class_name}(s) en moins de {rule['window_minutes']:.0f} min",
                        }
                    )
        self._surge_active = still_surging

        return newly_fired
