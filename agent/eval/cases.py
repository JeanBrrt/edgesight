"""Questions du banc de test (run_benchmark.py) et appels d'outils attendus.

Valeur attendue d'un argument :
  - une valeur exacte ;
  - ANY : n'importe quelle valeur ;
  - Approx(cible, tolérance) : pour les conversions ("5 minutes" -> 300 s) ;
  - OneOf((v1, v2)) : plusieurs réponses justes ("plus de 5" : 5 ou 6).
`expected_calls = ()` : aucun appel attendu (question hors sujet).

Ids : "<outil>__<variante>" (ou no_call__, multi__), pour que
compare_models.py regroupe les résultats par outil.

Les cas de zone utilisent les zones de config/zones.yaml (zone_centrale,
zone_laterale) : à mettre à jour si elles changent.
"""

from dataclasses import dataclass, field


class _Any:
    def __repr__(self):
        return "ANY"


ANY = _Any()


@dataclass(frozen=True)
class Approx:
    target: float
    tol: float


@dataclass(frozen=True)
class OneOf:
    values: tuple


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TestCase:
    id: str
    question: str
    expected_calls: tuple = ()
    notes: str = ""


ALL_CASES: list[TestCase] = [
    # ------------------------------------------------------------------
    # count_now -- formulations variées et synonymes
    # ------------------------------------------------------------------
    TestCase(
        "count_now__person_fr",
        "Combien de personnes y a-t-il actuellement ?",
        (ExpectedCall("count_now", {"object_class": "person"}),),
    ),
    TestCase(
        "count_now__car_fr",
        "Combien de voitures vois-tu là maintenant ?",
        (ExpectedCall("count_now", {"object_class": "car"}),),
    ),
    TestCase(
        "count_now__person_en",
        "How many people are currently in frame?",
        (ExpectedCall("count_now", {"object_class": "person"}),),
        notes="Variante anglaise -- la démo est requêtée en anglais.",
    ),
    TestCase(
        "count_now__person_synonym",
        "Combien de piétons sont présents en ce moment ?",
        (ExpectedCall("count_now", {"object_class": "person"}),),
        notes="Synonyme ('piéton' au lieu de 'personne') -- teste la généralisation au-delà du mot exact du schéma.",
    ),
    TestCase(
        "count_now__car_synonym",
        "Combien de véhicules sont garés là ?",
        (ExpectedCall("count_now", {"object_class": "car"}),),
        notes="Synonyme ('véhicule' au lieu de 'voiture').",
    ),
    # ------------------------------------------------------------------
    # count_total
    # ------------------------------------------------------------------
    TestCase(
        "count_total__car",
        "Combien de voitures sont passées au total ?",
        (ExpectedCall("count_total", {"object_class": "car"}),),
    ),
    TestCase(
        "count_total__person",
        "Combien de personnes différentes as-tu vues depuis le début ?",
        (ExpectedCall("count_total", {"object_class": "person"}),),
    ),
    # ------------------------------------------------------------------
    # "maintenant" contre "au total"
    # ------------------------------------------------------------------
    TestCase(
        "disambiguation__now_present_tense",
        "Combien de voitures y a-t-il ?",
        (ExpectedCall("count_now", {"object_class": "car"}),),
        notes="Présent -- devrait pencher vers 'maintenant', pas 'total'.",
    ),
    TestCase(
        "disambiguation__total_past_tense",
        "Combien de voitures sont passées ?",
        (ExpectedCall("count_total", {"object_class": "car"}),),
        notes="Passé composé -- devrait pencher vers 'total', pas 'maintenant'.",
    ),
    # ------------------------------------------------------------------
    # count_since -- relatif (minutes_ago)
    # ------------------------------------------------------------------
    TestCase(
        "count_since__relative_minutes",
        "Combien de personnes depuis 10 minutes ?",
        (ExpectedCall("count_since", {"object_class": "person", "minutes_ago": 10}),),
    ),
    TestCase(
        "count_since__relative_half_hour",
        "Combien de voitures ont été vues cette dernière demi-heure ?",
        (ExpectedCall("count_since", {"object_class": "car", "minutes_ago": 30}),),
    ),
    # ------------------------------------------------------------------
    # count_since -- absolu (start_time)
    # ------------------------------------------------------------------
    TestCase(
        "count_since__absolute_time",
        "Combien de voitures depuis 12h15 ?",
        (ExpectedCall("count_since", {"object_class": "car", "start_time": "12:15"}),),
    ),
    TestCase(
        "count_since__absolute_morning",
        "Depuis 9h du matin, combien de personnes ?",
        (ExpectedCall("count_since", {"object_class": "person", "start_time": "09:00"}),),
    ),
    TestCase(
        "count_since__loose_natural_phrasing",
        "Combien de personnes depuis midi et quart ?",
        (ExpectedCall("count_since", {"object_class": "person", "start_time": "12:15"}),),
        notes="Formulation naturelle non standard ('midi et quart') plutôt que 'HH:MM' -- test adversarial, échec plausible et informatif.",
    ),
    # ------------------------------------------------------------------
    # count_between
    # ------------------------------------------------------------------
    TestCase(
        "count_between__afternoon",
        "Combien de voitures sont passées entre 12h15 et 12h30 ?",
        (ExpectedCall("count_between", {"object_class": "car", "start_time": "12:15", "end_time": "12:30"}),),
    ),
    TestCase(
        "count_between__morning",
        "Combien de personnes entre 8h et midi ?",
        (ExpectedCall("count_between", {"object_class": "person", "start_time": "08:00", "end_time": "12:00"}),),
    ),
    # ------------------------------------------------------------------
    # Statistiques dérivées
    # ------------------------------------------------------------------
    TestCase(
        "time_since_last_seen__car",
        "Quand la dernière voiture est-elle passée ?",
        (ExpectedCall("time_since_last_seen", {"object_class": "car"}),),
    ),
    TestCase(
        "time_since_last_seen__person",
        "Ça fait combien de temps qu'on n'a pas vu de personne ?",
        (ExpectedCall("time_since_last_seen", {"object_class": "person"}),),
    ),
    TestCase(
        "average_presence_duration__person",
        "En moyenne, combien de temps reste une personne ?",
        (ExpectedCall("average_presence_duration", {"object_class": "person"}),),
        notes="À distinguer de time_since_last_seen -- formulation proche, intention différente.",
    ),
    TestCase(
        "average_presence_duration__car",
        "Durée moyenne de présence des voitures ?",
        (ExpectedCall("average_presence_duration", {"object_class": "car"}),),
    ),
    TestCase(
        "peak_concurrent_count__person",
        "Quel a été le pic de personnes présentes en même temps ?",
        (ExpectedCall("peak_concurrent_count", {"object_class": "person"}),),
    ),
    TestCase(
        "peak_concurrent_count__car",
        "Combien de voitures simultanées au maximum aujourd'hui ?",
        (ExpectedCall("peak_concurrent_count", {"object_class": "car"}),),
    ),
    # ------------------------------------------------------------------
    # Alertes
    # ------------------------------------------------------------------
    TestCase(
        "set_presence_duration_alert__person_seconds",
        "Préviens-moi si une personne reste plus de 10 secondes.",
        (ExpectedCall("set_presence_duration_alert", {"object_class": "person", "threshold_seconds": 10}),),
    ),
    TestCase(
        "set_presence_duration_alert__car_minutes_conversion",
        "Alerte si une voiture reste garée plus de 5 minutes.",
        (ExpectedCall("set_presence_duration_alert", {"object_class": "car", "threshold_seconds": Approx(300, 5)}),),
        notes="Conversion minutes -> secondes attendue (5 min = 300s).",
    ),
    TestCase(
        "set_presence_duration_alert__en",
        "Set an alert if a car stays parked for more than 2 minutes.",
        (ExpectedCall("set_presence_duration_alert", {"object_class": "car", "threshold_seconds": Approx(120, 5)}),),
        notes="Variante anglaise, avec conversion minutes -> secondes.",
    ),
    TestCase(
        "set_co_occurrence_alert__basic",
        "Préviens-moi si 2 personnes et 1 voiture sont là en même temps.",
        (ExpectedCall("set_co_occurrence_alert", {"person_threshold": 2, "car_threshold": 1}),),
    ),
    TestCase(
        "set_co_occurrence_alert__at_least",
        "Alerte-moi s'il y a au moins 3 personnes et 2 véhicules simultanément.",
        (ExpectedCall("set_co_occurrence_alert", {"person_threshold": 3, "car_threshold": 2}),),
    ),
    TestCase(
        "set_surge_alert__car",
        "Préviens-moi si plus de 5 voitures arrivent en moins de 2 minutes.",
        (
            ExpectedCall(
                "set_surge_alert",
                {"object_class": "car", "count_threshold": OneOf((5, 6)), "window_minutes": 2},
            ),
        ),
        notes="'Plus de 5' tolère 5 ou 6 selon l'interprétation stricte/large.",
    ),
    TestCase(
        "set_surge_alert__person_or_more",
        "Alerte si 10 personnes ou plus apparaissent en 1 minute.",
        (ExpectedCall("set_surge_alert", {"object_class": "person", "count_threshold": 10, "window_minutes": 1}),),
        notes="'Ou plus' correspond exactement à la sémantique du tool -- pas d'ambiguïté ici, contrairement au cas précédent.",
    ),
    TestCase(
        "set_zone_alert__car_specific_zone",
        "Préviens-moi si une voiture entre dans la zone centrale.",
        (ExpectedCall("set_zone_alert", {"zone_name": "zone_centrale", "object_class": "car"}),),
    ),
    TestCase(
        "set_zone_alert__any_class",
        "Alerte si n'importe qui ou n'importe quoi entre dans la zone latérale.",
        (ExpectedCall("set_zone_alert", {"zone_name": "zone_laterale", "object_class": "any"}),),
    ),
    TestCase(
        "count_zone_entries__car",
        "Combien de fois une voiture est entrée dans la zone centrale ?",
        (ExpectedCall("count_zone_entries", {"zone_name": "zone_centrale", "object_class": "car"}),),
    ),
    TestCase(
        "set_zone_alert__unknown_zone",
        "Préviens-moi si une voiture entre dans le parking nord.",
        (ExpectedCall("set_zone_alert", {"zone_name": "zone_inconnue", "object_class": "car"}),),
        notes="Piège : 'parking nord' n'existe pas dans config/zones.yaml. Attendu : 'zone_inconnue' "
        "plutôt qu'une zone réelle choisie par approximation.",
    ),
    # ------------------------------------------------------------------
    # Hors périmètre / robustesse
    # ------------------------------------------------------------------
    TestCase(
        "no_call__off_topic",
        "Quel temps fait-il aujourd'hui ?",
        (),
        notes="Question hors périmètre -- aucun tool ne répond à ça, le modèle ne devrait rien appeler.",
    ),
    TestCase(
        "no_call__greeting",
        "Bonjour, comment vas-tu ?",
        (),
        notes="Échange social pur, aucune information de détection demandée.",
    ),
    # ------------------------------------------------------------------
    # Appels multiples dans une seule question
    # ------------------------------------------------------------------
    TestCase(
        "multi__compare_now",
        "Y a-t-il plus de voitures que de personnes actuellement ?",
        (
            ExpectedCall("count_now", {"object_class": "car"}),
            ExpectedCall("count_now", {"object_class": "person"}),
        ),
        notes="Nécessite DEUX appels (une classe ne suffit pas à répondre à une comparaison).",
    ),
]


# ==========================================================================
# Cas multi-tours (run_memory_benchmark.py) : les questions d'une séquence
# partagent l'historique, pour tester la mémoire conversationnelle.
# ==========================================================================


@dataclass(frozen=True)
class Turn:
    question: str
    expected_calls: tuple = ()


@dataclass(frozen=True)
class MultiTurnTestCase:
    id: str
    turns: tuple[Turn, ...]
    notes: str = ""


MULTI_TURN_CASES: list[MultiTurnTestCase] = [
    MultiTurnTestCase(
        "memory__followup_class_swap",
        (
            Turn("Combien de personnes y a-t-il actuellement ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
            Turn("Et de voitures ?", (ExpectedCall("count_now", {"object_class": "car"}),)),
        ),
        notes="Cas de référence de la demande initiale -- 'et de voitures ?' n'a de sens "
        "qu'à la lumière du tour précédent (même tool, classe différente).",
    ),
    MultiTurnTestCase(
        "memory__followup_alt_phrasing",
        (
            Turn("Combien de véhicules sont passés au total ?", (ExpectedCall("count_total", {"object_class": "car"}),)),
            Turn("Et côté piétons ?", (ExpectedCall("count_total", {"object_class": "person"}),)),
        ),
        notes="Variante de formulation elliptique ('côté X' plutôt que 'et de X ?'), "
        "sur count_total plutôt que count_now -- vérifie que le suivi de contexte "
        "ne dépend pas d'un tour de phrase unique appris par cœur.",
    ),
    MultiTurnTestCase(
        "memory__no_stale_reuse",
        (
            Turn("Combien de personnes maintenant ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
            Turn("Et maintenant, combien de personnes ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
        ),
        notes="Risque n°2 identifié à la conception (bilan mémoire) : la même question "
        "posée deux fois doit redéclencher le tool à CHAQUE fois, pas réutiliser le "
        "chiffre déjà donné au tour précédent -- 'maintenant' n'est jamais périmé à "
        "l'avance. Couvert par la consigne ajoutée au prompt système "
        "(config/agent.yaml) ; ce cas mesure si elle est réellement suivie.",
    ),
    MultiTurnTestCase(
        "memory__unrelated_topic_no_confusion",
        (
            Turn("Combien de personnes actuellement ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
            Turn(
                "Préviens-moi si une voiture reste garée plus de 30 secondes.",
                (ExpectedCall("set_presence_duration_alert", {"object_class": "car", "threshold_seconds": 30}),),
            ),
        ),
        notes="Un changement de sujet complet en plein milieu de la conversation ne doit "
        "pas faire dériver le 2e tour vers le sujet du 1er (ex. appeler count_now par "
        "réflexe plutôt que le tool d'alerte réellement demandé).",
    ),
    MultiTurnTestCase(
        "memory__three_turn_chain",
        (
            Turn("Combien de personnes maintenant ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
            Turn("Et de voitures ?", (ExpectedCall("count_now", {"object_class": "car"}),)),
            Turn("Et à nouveau les personnes ?", (ExpectedCall("count_now", {"object_class": "person"}),)),
        ),
        notes="Chaîne à 3 tours -- vérifie que le suivi de contexte reste correct au-delà "
        "d'un unique aller-retour (retour à la classe du 1er tour après un détour par "
        "une autre), sans dérive cumulative.",
    ),
]
