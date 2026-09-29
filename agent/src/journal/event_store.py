"""Journal d'événements (SQLite)

- `events` : une ligne par piste, avec sa première et sa dernière
  apparition. Une piste est "active" si elle a été vue récemment : pas
  besoin de décider quand elle se termine.
- `alerts` : les 4 types de règles dans une seule table, chaque règle ne
  remplissant que ses colonnes. Les colonnes inutilisées de la clé
  valent '' et non NULL : SQLite ne considère jamais deux NULL comme
  égaux, et une règle reposée se serait ajoutée au lieu de remplacer
  l'ancienne.
- `zone_events` : une ligne par entrée dans une zone.
"""

import sqlite3
import threading
import time

DEFAULT_DB_PATH = "agent/data/events.db"

# Colonnes de valeurs de la table `alerts`.
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
            # Table d'une ancienne version (sans alert_type) : recréée, les
            # règles n'étant pas des données à conserver.
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
    # Règles d'alerte (une méthode par type)
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
        """Alerte si un objet reste plus de `threshold_seconds` secondes."""
        self._upsert_alert(
            "duration", object_class=object_class, threshold_seconds=threshold_seconds, timestamp=timestamp
        )

    def set_co_occurrence_alert(self, person_threshold: int, car_threshold: int, timestamp: float | None = None):
        """Alerte si au moins `person_threshold` personnes et
        `car_threshold` voitures sont présentes en même temps. Une seule
        règle de ce type : la reposer remplace l'ancienne."""
        self._upsert_alert(
            "co_occurrence",
            object_class="both",
            person_threshold=person_threshold,
            car_threshold=car_threshold,
            timestamp=timestamp,
        )

    def set_surge_alert(self, object_class: str, count_threshold: int, window_minutes: float, timestamp: float | None = None):
        """Alerte si `count_threshold` objets ou plus apparaissent en moins
        de `window_minutes` (une règle par classe)."""
        self._upsert_alert(
            "surge",
            object_class=object_class,
            count_threshold=count_threshold,
            window_minutes=window_minutes,
            timestamp=timestamp,
        )

    def set_zone_alert(self, zone_name: str, object_class: str, timestamp: float | None = None):
        """Alerte si un objet de cette classe (ou 'any') entre dans la zone
        (une règle par couple zone/classe)."""
        self._upsert_alert("zone", object_class=object_class, zone_name=zone_name, timestamp=timestamp)

    def get_alerts(self) -> list[dict]:
        """Toutes les règles, tous types confondus (champs inutilisés à None)."""
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
    # Journal des pistes
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
        """Nombre de pistes apparues entre deux instants."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE class = ? AND first_seen BETWEEN ? AND ?",
                (class_name, start_ts, end_ts),
            )
            return cur.fetchone()[0]

    def active_tracks(
        self, class_name: str | None = None, active_within_seconds: float = 1.0, now: float | None = None
    ) -> list[tuple[int, str, float, float]]:
        """Pistes vues il y a moins de `active_within_seconds`."""
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
        """(track_id, secondes depuis sa première apparition) des pistes actives."""
        now = now if now is not None else time.time()
        # Pas de verrou ici : active_tracks le prend déjà, et il n'est pas réentrant.
        return [(tid, now - first_seen) for tid, _, first_seen, _ in self.active_tracks(class_name, active_within_seconds, now)]

    def most_recent_last_seen(self, class_name: str) -> float | None:
        """Instant de la dernière détection de cette classe (None si jamais
        vue). Sur last_seen et non first_seen : une voiture présente depuis
        une minute est toujours visible."""
        with self._lock:
            cur = self.conn.execute("SELECT MAX(last_seen) FROM events WHERE class = ?", (class_name,))
            return cur.fetchone()[0]

    def average_presence_duration(self, class_name: str) -> float | None:
        """Durée moyenne de présence (None si aucune piste). Une piste encore
        présente compte pour sa durée jusqu'ici : légère sous-estimation."""
        with self._lock:
            cur = self.conn.execute("SELECT first_seen, last_seen FROM events WHERE class = ?", (class_name,))
            rows = cur.fetchall()
        if not rows:
            return None
        return sum(last_seen - first_seen for first_seen, last_seen in rows) / len(rows)

    def peak_concurrent_count(self, class_name: str) -> int:
        """Nombre maximal de pistes présentes en même temps, par balayage des
        intervalles de présence. À instant égal, une arrivée compte avant un
        départ : les deux pistes ont coexisté."""
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
        """Enregistre une entrée dans une zone (détectée par ZoneMonitor)."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO zone_events (zone_name, class, track_id, entered_at) VALUES (?, ?, ?, ?)",
                (zone_name, class_name, track_id, ts),
            )
            self.conn.commit()

    def count_zone_entries(self, zone_name: str, class_name: str | None = None) -> int:
        """Nombre d'entrées dans la zone (`None` ou 'any' : toutes classes)."""
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
        """Vide les pistes et les entrées en zone, sans toucher aux règles.
        À appeler avec MultiClassByteTracker.reset() : les identifiants
        repartent de 0 et se mélangeraient aux anciennes pistes."""
        with self._lock:
            self.conn.execute("DELETE FROM events")
            self.conn.execute("DELETE FROM zone_events")
            self.conn.commit()

    def clear_alerts(self):
        """Supprime toutes les règles. Au changement de scène seulement (pas
        au redémarrage d'une vidéo) : une alerte posée sur une scène n'a
        pas de sens sur une autre."""
        with self._lock:
            self.conn.execute("DELETE FROM alerts")
            self.conn.commit()

    def close(self):
        self.conn.close()
