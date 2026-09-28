"""D3 — Mécanisme d'alerte temps réel, 3 des 4 types de règles (voir
event_store.py) : présence continue (`duration`), co-occurrence de deux
classes (`co_occurrence`), rafale sur fenêtre glissante (`surge`). Le 4e
type (`zone`) est géré séparément par `ZoneMonitor` (zones.py) : il a
besoin des boîtes suivies du frame courant (position), pas seulement du
journal `EventStore` -- une dépendance différente qui justifie de ne pas
tout regrouper dans une seule classe.

Boucle de surveillance continue (pas un tool d'agent) : consomme
`EventStore.get_alerts()` (les règles configurées par D1/D2) et l'état
vivant du tracking (C1/C2). Conçu pour tourner dans la même boucle live
que C1/C2 (donc avec la tolérance de fraîcheur *serrée* d'`EventStore`
par défaut, 1,0s -- volontairement plus stricte que celle de D1,
agent/src/agent/tools.py : réagir vite à une vraie disparition importe plus
ici qu'en conversationnel. Pas une question de latence LLM à compenser
dans un sens ou l'autre : `now` est de toute façon ancré sur l'heure de
l'appel des deux côtés).

Chaque type a sa propre logique de "ne pas re-notifier en boucle" :
- `duration` : une fois par (classe, track_id) tant que la piste reste
  active -- état gardé dans `self._notified_duration` (mémoire du
  processus, pas persisté, comme avant).
- `co_occurrence`/`surge` : condition globale (ou par classe pour
  `surge`) à FRONT MONTANT -- notifiée au moment où la condition devient
  vraie, pas à chaque tick tant qu'elle le reste, puis réarmée dès
  qu'elle redevient fausse (`self._co_occurrence_active`/
  `self._surge_active`).
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
        """Réinitialise tout l'état de dédup interne (mémoire du processus,
        jamais persistée -- voir docstring du module). À appeler à un vrai
        changement de vidéo/démo, en plus de `EventStore.clear_alerts()` :
        sans ça, un track_id réutilisé depuis 0 par la nouvelle scène
        pourrait hériter à tort du statut "déjà notifié" d'une piste sans
        rapport de l'ancienne."""
        self._notified_duration.clear()
        self._co_occurrence_active = False
        self._surge_active.clear()

    def check(self, now: float | None = None) -> list[dict]:
        """À appeler à chaque frame/tick de la boucle live. Renvoie la
        liste des alertes qui viennent de se déclencher CE tick-ci
        (typiquement vide) : chaque élément est un dict avec au moins
        `alert_type` et `detail` (message lisible prêt à afficher/logguer)."""
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
        # Oublie les pistes qui ne sont plus actives du tout -- si un futur
        # track_id venait à être réutilisé, il repart sans mémoire d'un
        # ancien déclenchement.
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
