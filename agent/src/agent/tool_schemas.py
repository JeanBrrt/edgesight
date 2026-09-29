"""Schémas des outils envoyés au LLM (format OpenAI, écrits à la main)
"""

from ..alerts.zones import UNKNOWN_ZONE, ZONES

_ZONE_NAMES = list(ZONES)
_CLASSES = ["person", "car"]

_ZONE_NAME_ENUM = _ZONE_NAMES + [UNKNOWN_ZONE]
_ZONE_NAME_DESCRIPTION = (
    "Nom de la zone prédéfinie. Zones connues : " + ", ".join(_ZONE_NAMES) + ". "
    f"Si la zone décrite par l'utilisateur ne correspond CLAIREMENT à aucune "
    f"de ces zones, utiliser '{UNKNOWN_ZONE}' -- ne jamais choisir la zone "
    "réelle la plus proche par approximation."
)

TOOL_SCHEMAS = [
    # --- Comptage ----------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "count_now",
            "description": (
                "Compte le nombre d'objets d'une classe donnée ACTUELLEMENT "
                "présents dans le champ de la caméra. À utiliser pour des "
                "questions comme 'combien de voitures y a-t-il', 'combien de "
                "personnes vois-tu là maintenant'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "enum": _CLASSES, "description": "Classe d'objet à compter."},
                },
                "required": ["object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_total",
            "description": (
                "Compte le nombre TOTAL d'objets distincts d'une classe donnée "
                "vus depuis le début de la session (présents ou déjà repartis). "
                "À utiliser pour 'combien de voitures sont passées au total'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"object_class": {"type": "string", "enum": _CLASSES}},
                "required": ["object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_since",
            "description": (
                "Compte le nombre d'objets distincts d'une classe donnée vus "
                "depuis un point de départ jusqu'à maintenant. Fournir "
                "EXACTEMENT un des deux : 'start_time' pour une heure absolue "
                "('depuis 12h15') ou 'minutes_ago' pour une durée relative "
                "('depuis 10 minutes')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "enum": _CLASSES},
                    "start_time": {
                        "type": "string",
                        "description": "Heure absolue de départ, HH:MM (24h), ex. '12:15'. Omettre si minutes_ago est fourni.",
                    },
                    "minutes_ago": {
                        "type": "number",
                        "description": "Durée relative en minutes, ex. 10. Omettre si start_time est fourni.",
                    },
                },
                "required": ["object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_between",
            "description": (
                "Compte le nombre d'objets distincts d'une classe donnée vus "
                "entre deux horaires précis de la journée en cours. À "
                "utiliser pour 'entre 12h15 et 12h30'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "enum": _CLASSES},
                    "start_time": {"type": "string", "description": "HH:MM (24h), ex. '12:15'."},
                    "end_time": {"type": "string", "description": "HH:MM (24h), ex. '12:30'."},
                },
                "required": ["object_class", "start_time", "end_time"],
            },
        },
    },
    # --- Statistiques dérivées --------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "time_since_last_seen",
            "description": (
                "Renvoie depuis combien de secondes aucun objet d'une classe "
                "donnée n'a été détecté. À utiliser pour 'depuis combien de temps "
                "n'a-t-on pas vu de voiture' ou 'quand la dernière voiture "
                "est-elle passée'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"object_class": {"type": "string", "enum": _CLASSES}},
                "required": ["object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "average_presence_duration",
            "description": (
                "Durée moyenne de présence (en secondes) des objets d'une "
                "classe donnée, depuis le début de la session. À utiliser "
                "pour 'en moyenne combien de temps reste une personne'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"object_class": {"type": "string", "enum": _CLASSES}},
                "required": ["object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peak_concurrent_count",
            "description": (
                "Nombre maximal d'objets d'une classe donnée présents "
                "SIMULTANÉMENT à un instant quelconque depuis le début de la "
                "session. À utiliser pour 'quel a été le pic de véhicules'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"object_class": {"type": "string", "enum": _CLASSES}},
                "required": ["object_class"],
            },
        },
    },
    # --- Alertes -----------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "set_presence_duration_alert",
            "description": (
                "Configure une alerte si un objet d'une classe donnée reste "
                "présent en continu plus longtemps que le seuil indiqué."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "enum": _CLASSES},
                    "threshold_seconds": {"type": "number", "description": "Seuil en secondes."},
                },
                "required": ["object_class", "threshold_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_co_occurrence_alert",
            "description": (
                "Configure une alerte si au moins N personnes ET au moins M "
                "voitures sont présentes simultanément dans le champ. Ex. "
                "'préviens-moi si 2 personnes et 1 voiture sont là en même "
                "temps'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_threshold": {"type": "integer", "description": "Nombre minimum de personnes."},
                    "car_threshold": {"type": "integer", "description": "Nombre minimum de voitures."},
                },
                "required": ["person_threshold", "car_threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_surge_alert",
            "description": (
                "Configure une alerte si plus de N objets d'une classe "
                "donnée apparaissent en moins de M minutes (afflux/rafale)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_class": {"type": "string", "enum": _CLASSES},
                    "count_threshold": {"type": "integer", "description": "Nombre d'objets déclenchant l'alerte."},
                    "window_minutes": {"type": "number", "description": "Fenêtre glissante en minutes."},
                },
                "required": ["object_class", "count_threshold", "window_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_zone_alert",
            "description": (
                "Configure une alerte si un objet d'une classe donnée "
                "pénètre dans une zone de danger prédéfinie."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {
                        "type": "string",
                        "enum": _ZONE_NAME_ENUM,
                        "description": _ZONE_NAME_DESCRIPTION,
                    },
                    "object_class": {
                        "type": "string",
                        "enum": _CLASSES + ["any"],
                        "description": "Classe surveillée dans cette zone ('any' = les deux).",
                    },
                },
                "required": ["zone_name", "object_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_zone_entries",
            "description": (
                "Compte le nombre d'entrées d'objets d'une classe donnée "
                "dans une zone prédéfinie, depuis le début de la session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {"type": "string", "enum": _ZONE_NAME_ENUM, "description": _ZONE_NAME_DESCRIPTION},
                    "object_class": {"type": "string", "enum": _CLASSES + ["any"]},
                },
                "required": ["zone_name", "object_class"],
            },
        },
    },
]
