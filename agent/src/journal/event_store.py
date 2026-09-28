"""C2 — Store d'événements : log horodaté des pistes, interrogeable par
plage de temps, plus les règles d'alerte (4 types) et le journal des
entrées en zone.

Entrée de `update()` : la sortie de `MultiClassByteTracker.update()`
(tracker.py) — {class_idx: [[x1, y1, x2, y2, score, track_id, ...], ...]}

Une ligne par (classe, track_id) dans `events`, avec `first_seen`/
`last_seen` mis à jour à chaque frame où la piste est vue. Pas de champ
"actif" stocké : une piste est considérée active si `last_seen` est
récent (voir `active_within_seconds`) — évite d'avoir à décider frame
par frame si une piste doit être "close" (le tracker peut la faire
réapparaître avec le même ID après une brève absence grâce à son propre
buffer).

`alerts` couvre 4 types de règles hétérogènes (durée de présence,
co-occurrence de deux classes, rafale sur fenêtre glissante, entrée en
zone) dans une seule table à colonnes creuses plutôt que 4 tables ou un
blob JSON — chaque règle a un type + les seuls champs qui la concernent,
les autres restant NULL. La clé naturelle est composite
(alert_type, object_class, zone_name) : `object_class`/`zone_name`
utilisent une chaîne vide '' (jamais NULL) comme sentinelle pour les
types qui ne s'en servent pas, parce que SQLite ne fait JAMAIS
collisionner deux NULL dans une contrainte UNIQUE — avec NULL comme
sentinelle, plusieurs lignes "co_occurrence" auraient pu s'accumuler
silencieusement au lieu de se remplacer (ON CONFLICT ne se déclenche
jamais entre deux NULL).

`zone_events` journalise chaque ENTRÉE (transition dehors -> dedans,
pas une ligne par frame passé dans la zone) détectée par `ZoneMonitor`
(zones.py) — sert à la fois à `count_zone_entries` et, en amont, au
déclenchement des alertes de type "zone".

Accédé depuis deux threads à partir de E1/E2 (la boucle vidéo d'un côté,
le thread agent de l'autre) : `check_same_thread=False` + un verrou
autour de chaque accès à `self.conn` pour sérialiser nous-mêmes les
écritures/lectures plutôt que de compter sur le comportement par défaut
du module sqlite3.
"""

import sqlite3
import threading
import time

DEFAULT_DB_PATH = "agent/data/events.db"

# Colonnes de la table `alerts`, dans l'ordre utilisé par _upsert_alert
# et get_alerts() -- un seul endroit à mettre à jour si on ajoute un type.
_ALERT_VALUE_COLUMNS = (
    "threshold_seconds", "person_threshold", "car_threshold",
    "count_threshold", "window_minutes",
)


class EventStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    track_id   INTEGER NOT NULL,
                    class      TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen  REAL NOT NULL,
                    PRIMARY KEY (class, track_id)
                )
                """
            )
            # Ancien schéma (une ligne par classe, alerte de durée
            # uniquement, pas de colonne alert_type) : reconstruit plutôt
            # que migré -- c'est un état de configuration jetable,
            # recréé par les set_*_alert au prochain lancement de la
            # démo, pas une donnée à préserver.
            cur = self.conn.execute("PRAGMA table_info(alerts)")
            existing_columns = {row[1] for row in cur.fetchall()}
            if existing_columns and "alert_type" not in existing_columns:
                self.conn.execute("DROP TABLE alerts")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_type         TEXT NOT NULL,
                    object_class       TEXT NOT NULL DEFAULT '',
                    zone_name          TEXT NOT NULL DEFAULT '',
                    threshold_seconds  REAL,
                    person_threshold   INTEGER,
                    car_threshold      INTEGER,
                    count_threshold    INTEGER,
                    window_minutes     REAL,
                    created_at         REAL NOT NULL,
                    UNIQUE (alert_type, object_class, zone_name)
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS zone_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone_name  TEXT NOT NULL,
                    class      TEXT NOT NULL,
                    track_id   INTEGER NOT NULL,
                    entered_at REAL NOT NULL
                )
                """
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Alertes -- une méthode publique par type de règle, toutes appuyées
    # sur le même upsert générique (clé composite décrite dans le
    # docstring du module).
    # ------------------------------------------------------------------

    def _upsert_alert(
        self,
        alert_type: str,
        object_class: str = "",
        zone_name: str = "",
        timestamp: float | None = None,
        **value_fields,
    ):
        ts = timestamp if timestamp is not None else time.time()
        row = {col: value_fields.get(col) for col in _ALERT_VALUE_COLUMNS}
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO alerts (
                    alert_type, object_class, zone_name, threshold_seconds,
                    person_threshold, car_threshold, count_threshold,
                    window_minutes, created_at
                )
                VALUES (
                    :alert_type, :object_class, :zone_name, :threshold_seconds,
                    :person_threshold, :car_threshold, :count_threshold,
                    :window_minutes, :created_at
                )
                ON CONFLICT(alert_type, object_class, zone_name) DO UPDATE SET
                    threshold_seconds = excluded.threshold_seconds,
                    person_threshold  = excluded.person_threshold,
                    car_threshold     = excluded.car_threshold,
                    count_threshold   = excluded.count_threshold,
                    window_minutes    = excluded.window_minutes,
                    created_at        = excluded.created_at
                """,
                {
                    "alert_type": alert_type,
                    "object_class": object_class,
                    "zone_name": zone_name,
                    "created_at": ts,
                    **row,
                },
            )
            self.conn.commit()

    def set_duration_alert(self, object_class: str, threshold_seconds: float, timestamp: float | None = None):
        """Alerte si un objet de cette classe reste présent en continu plus
        longtemps que `threshold_seconds` (généralise l'ancien set_alert,
        limité à `person`)."""
        self._upsert_alert(
            "duration", object_class=object_class, threshold_seconds=threshold_seconds, timestamp=timestamp
        )

    def set_co_occurrence_alert(self, person_threshold: int, car_threshold: int, timestamp: float | None = None):
        """Alerte si au moins `person_threshold` personnes ET au moins
        `car_threshold` voitures sont présentes simultanément. Une seule
        règle active à la fois (object_class fixé à la sentinelle 'both',
        une reconfiguration remplace la précédente)."""
        self._upsert_alert(
            "co_occurrence",
            object_class="both",
            person_threshold=person_threshold,
            car_threshold=car_threshold,
            timestamp=timestamp,
        )

    def set_surge_alert(self, object_class: str, count_threshold: int, window_minutes: float, timestamp: float | None = None):
        """Alerte si `count_threshold` objets de cette classe ou plus
        apparaissent en moins de `window_minutes` (une règle active par
        classe)."""
        self._upsert_alert(
            "surge",
            object_class=object_class,
            count_threshold=count_threshold,
            window_minutes=window_minutes,
            timestamp=timestamp,
        )

    def set_zone_alert(self, zone_name: str, object_class: str, timestamp: float | None = None):
        """Alerte si un objet de cette classe (ou 'any') entre dans la zone
        nommée (une règle active par couple zone/classe -- plusieurs zones
        distinctes coexistent normalement)."""
        self._upsert_alert("zone", object_class=object_class, zone_name=zone_name, timestamp=timestamp)

    def get_alerts(self) -> list[dict]:
        """Toutes les règles configurées, quel que soit leur type -- à
        `AlertMonitor`/`ZoneMonitor` de filtrer par `alert_type` et de lire
        les champs qui les concernent (les autres valent None)."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT alert_type, object_class, zone_name, threshold_seconds, "
                "person_threshold, car_threshold, count_threshold, window_minutes FROM alerts"
            )
            rows = cur.fetchall()
        columns = (
            "alert_type", "object_class", "zone_name", "threshold_seconds",
            "person_threshold", "car_threshold", "count_threshold", "window_minutes",
        )
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Journal des pistes (C2 d'origine)
    # ------------------------------------------------------------------

    def update(self, tracked: dict, class_names: list[str], timestamp: float | None = None):
        """À appeler une fois par frame avec la sortie du tracker."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            for cls_idx, boxes in tracked.items():
                cls_name = class_names[cls_idx]
                for box in boxes:
                    track_id = int(box[5])
                    self.conn.execute(
                        """
                        INSERT INTO events (track_id, class, first_seen, last_seen)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(class, track_id) DO UPDATE SET last_seen = excluded.last_seen
                        """,
                        (track_id, cls_name, ts, ts),
                    )
            self.conn.commit()

    def count_between(self, class_name: str, start_ts: float, end_ts: float) -> int:
        """Nombre de pistes distinctes de cette classe apparues entre deux horodatages.
        Répond à : "combien de voitures sont passées entre 12h15 et 12h30"."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE class = ? AND first_seen BETWEEN ? AND ?",
                (class_name, start_ts, end_ts),
            )
            return cur.fetchone()[0]

    def active_tracks(
        self, class_name: str | None = None, active_within_seconds: float = 1.0, now: float | None = None
    ) -> list[tuple[int, str, float, float]]:
        """Pistes considérées actives = vues il y a moins de `active_within_seconds`.
        Répond à : "combien de véhicules sont dans le champ maintenant"."""
        now = now if now is not None else time.time()
        cutoff = now - active_within_seconds
        with self._lock:
            if class_name is not None:
                cur = self.conn.execute(
                    "SELECT track_id, class, first_seen, last_seen FROM events WHERE class = ? AND last_seen >= ?",
                    (class_name, cutoff),
                )
            else:
                cur = self.conn.execute(
                    "SELECT track_id, class, first_seen, last_seen FROM events WHERE last_seen >= ?",
                    (cutoff,),
                )
            return cur.fetchall()

    def active_durations(
        self, class_name: str, active_within_seconds: float = 1.0, now: float | None = None
    ) -> list[tuple[int, float]]:
        """Pour chaque piste active de cette classe : (track_id, durée en secondes
        depuis sa première apparition). Répond à : "personne présente depuis plus
        de 10 secondes"."""
        now = now if now is not None else time.time()
        # Ne prend pas self._lock ici : délègue entièrement à active_tracks (qui
        # verrouille déjà) -- un Lock() classique n'est pas réentrant.
        return [(tid, now - first_seen) for tid, _, first_seen, _ in self.active_tracks(class_name, active_within_seconds, now)]

    def most_recent_last_seen(self, class_name: str) -> float | None:
        """Timestamp de la dernière détection, toutes pistes de cette classe
        confondues (None si aucune piste jamais vue). Répond à : "depuis
        combien de temps n'a-t-on pas vu de voiture". Volontairement
        `last_seen` et non `first_seen` : sur un flux continu, la piste la
        plus récente peut être apparue il y a longtemps tout en étant encore
        visible -- `MAX(first_seen)` répondait alors "il y a 1 minute" alors
        qu'une voiture était à l'écran."""
        with self._lock:
            cur = self.conn.execute("SELECT MAX(last_seen) FROM events WHERE class = ?", (class_name,))
            return cur.fetchone()[0]

    def average_presence_duration(self, class_name: str) -> float | None:
        """Durée moyenne (last_seen - first_seen) des pistes de cette classe
        vues depuis le début de la session (None si aucune piste). Une piste
        encore active est comptée avec sa durée "jusqu'ici", pas sa durée
        finale -- légère sous-estimation tant qu'elle reste présente,
        approximation jugée suffisante plutôt qu'une notion de piste
        "terminée" qui n'existe pas ailleurs dans ce store (cf. absence de
        champ "actif", même raisonnement)."""
        with self._lock:
            cur = self.conn.execute("SELECT first_seen, last_seen FROM events WHERE class = ?", (class_name,))
            rows = cur.fetchall()
        if not rows:
            return None
        return sum(last_seen - first_seen for first_seen, last_seen in rows) / len(rows)

    def peak_concurrent_count(self, class_name: str) -> int:
        """Nombre maximal de pistes de cette classe actives SIMULTANÉMENT à
        un instant quelconque depuis le début de la session -- balayage des
        intervalles (first_seen, last_seen), pas un simple COUNT. À
        égalité de timestamp, une arrivée est comptée avant un départ
        (deux pistes vues au même instant exact ont bien coexisté à ce
        frame-là)."""
        with self._lock:
            cur = self.conn.execute("SELECT first_seen, last_seen FROM events WHERE class = ?", (class_name,))
            rows = cur.fetchall()
        if not rows:
            return 0
        events = []
        for first_seen, last_seen in rows:
            events.append((first_seen, 1))
            events.append((last_seen, -1))
        events.sort(key=lambda e: (e[0], -e[1]))
        concurrent = 0
        peak = 0
        for _, delta in events:
            concurrent += delta
            peak = max(peak, concurrent)
        return peak

    # ------------------------------------------------------------------
    # Zones (ZoneMonitor, zones.py)
    # ------------------------------------------------------------------

    def log_zone_entry(self, zone_name: str, class_name: str, track_id: int, timestamp: float | None = None):
        """Journalise une ENTRÉE en zone (transition dehors -> dedans,
        détectée en amont par ZoneMonitor -- pas une ligne par frame passé
        dans la zone)."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO zone_events (zone_name, class, track_id, entered_at) VALUES (?, ?, ?, ?)",
                (zone_name, class_name, track_id, ts),
            )
            self.conn.commit()

    def count_zone_entries(self, zone_name: str, class_name: str | None = None) -> int:
        """Nombre d'entrées journalisées dans cette zone depuis le début de
        la session. `class_name=None` ou `'any'` : toutes classes
        confondues."""
        with self._lock:
            if class_name is not None and class_name != "any":
                cur = self.conn.execute(
                    "SELECT COUNT(*) FROM zone_events WHERE zone_name = ? AND class = ?",
                    (zone_name, class_name),
                )
            else:
                cur = self.conn.execute("SELECT COUNT(*) FROM zone_events WHERE zone_name = ?", (zone_name,))
            return cur.fetchone()[0]

    # ------------------------------------------------------------------

    def clear_events(self):
        """Vide le journal des pistes ET des entrées en zone (garde les
        règles d'alerte). À appeler quand le tracker est réinitialisé
        (changement de source vidéo, MultiClassByteTracker.reset()) : les
        tracker_id qu'il va réémettre repartent de 0 par classe, ce qui
        entrerait en collision avec d'anciennes lignes (class, track_id)
        déjà présentes sinon -- une piste de la nouvelle source se
        retrouverait fusionnée avec une piste sans rapport de l'ancienne,
        faussant sa durée de présence (et, de la même façon, ses éventuelles
        entrées en zone passées). Garde volontairement les règles d'alerte
        (table `alerts`) -- voir `clear_alerts()` pour les effacer, séparément,
        à un vrai changement de vidéo/démo."""
        with self._lock:
            self.conn.execute("DELETE FROM events")
            self.conn.execute("DELETE FROM zone_events")
            self.conn.commit()

    def clear_alerts(self):
        """Supprime toutes les règles d'alerte configurées (set_*_alert).
        À appeler en plus de `clear_events()`, mais seulement à un vrai
        changement de vidéo/démo (pas un simple redémarrage de la même
        vidéo, RESTART_KEY) -- une alerte posée par l'agent sur une scène
        ("préviens-moi si une voiture entre dans le quai de chargement")
        n'a plus de sens sur une scène sans rapport, et ne doit jamais
        se reporter automatiquement dessus."""
        with self._lock:
            self.conn.execute("DELETE FROM alerts")
            self.conn.commit()

    def close(self):
        self.conn.close()
