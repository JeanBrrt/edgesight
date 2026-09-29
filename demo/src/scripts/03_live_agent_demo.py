"""E1 + E2 — Pipeline live complet : vidéo -> détection -> tracking ->
journal -> alerte -> agent conversationnel, avec trois fenêtres séparées
pour ne jamais mélanger l'affichage vidéo, les logs et la conversation :
  - fenêtre vidéo (OpenCV) : overlay détections/tracking + bannière
    d'alerte (raccourcis clavier c/r/q) -- les réponses de l'agent
    s'affichent uniquement dans la fenêtre Assistant, jamais ici,
    pour ne jamais dupliquer la même information à deux endroits ;
  - fenêtre "Logs" (Tkinter) : trace complète du logger racine (tools et
    arguments choisis par l'agent, alertes D3...), pour un suivi global
    sans avoir à rouvrir agent/data/03_live_agent_demo.log ;
  - fenêtre "Assistant" (Tkinter) : historique de conversation + champ de
    saisie, pré-remplie au démarrage d'un message d'accueil (exemples de
    questions + rappel des raccourcis de la fenêtre vidéo).

D3 (alerte temps réel) tourne en silence sur le plan sonore/console : la
surveillance (AlertMonitor/ZoneMonitor) ne s'imprime jamais sur la
console. Quand une alerte se déclenche, deux choses se produisent : (1)
une bannière apparaît immédiatement en haut de la fenêtre vidéo avec le
détail brut de l'événement -- retour visuel instantané, sans latence LLM ;
(2) un tour de LLM est lancé en arrière-plan (jamais dans la boucle
vidéo) pour transformer cet événement brut en notification en langage
naturel, affichée ensuite dans la fenêtre Assistant comme une réponse
d'agent normale.

Architecture à trois threads/boucles :
  - la boucle Tkinter (root.mainloop()) possède le thread principal --
    c'est une contrainte de Tkinter, contrairement à cv2.HighGUI qui
    tolère en pratique un thread non principal sous Windows ;
  - la boucle vidéo (capture/inférence/tracking/affichage cv2, ancien
    contenu de main()) tourne dans un thread dédié (`video_loop`) ;
  - chaque question posée dans la fenêtre Assistant lance son propre
    thread court (`agent.ask()` bloque ~1,5-2s, latence LLM incluse) --
    jamais dans la boucle vidéo ni dans la boucle Tkinter, sous peine de
    geler l'un ou l'autre pendant tout ce temps. Le champ de saisie est
    désactivé pendant qu'une question est en cours pour ne jamais avoir
    deux threads en train de muter la même `Session` en même temps (cf.
    sa docstring dans agent/src/agent/agent.py).
  - la file `answers` (`queue.Queue`) reste le point de rendez-vous
    unique entre ces threads et la boucle vidéo, qui relaie chaque
    résultat vers la fenêtre Assistant via `root.after(0, ...)` -- seule
    façon thread-safe de toucher un widget Tkinter depuis un autre
    thread. C'est ce qui a nécessité de rendre `EventStore` thread-safe
    (verrou interne, voir event_store.py) : tous ces threads
    lisent/écrivent le même journal.

llama-server est lancé automatiquement s'il ne tourne pas déjà, puis
arrêté à la sortie (agent/src/agent/llm_server.py, bloc `llama_server`
de config/agent.yaml).
Lancer depuis la racine du projet :
    uv run python demo/src/scripts/03_live_agent_demo.py
"""

import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
import numpy as np
import onnxruntime as ort
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config

sys.path.insert(0, ".")  # C1/C2/D1/D2/D3 vivent dans agent/src/
from agent.src.tracking.tracker import MultiClassByteTracker
from agent.src.journal.event_store import EventStore
from agent.src.alerts.alerts import AlertMonitor
from agent.src.alerts.zones import ZoneMonitor, scene_for_source
from agent.src.agent.agent import Agent, Session
from agent.src.agent.llm_server import llama_server

sys.path.insert(0, "demo/src/common")  # config.py/fps_counter.py/source_cycle.py
# vivent à part des scripts, partagés par les 3 (voir demo/src/common/)
from fps_counter import FPSCounter, draw_fps
from source_cycle import (
    SourceCycler,
    CYCLE_KEY,
    RESTART_KEY,
    draw_source_label,
    draw_interactive_help,
    draw_banner,
    resize_for_display,
    ui_scale,
)
from text_render import draw_box_label
from config import (
    CONFIG_PATH,
    ONNX_PATH,
    INPUT_SIZE,
    DISABLED_CLASSES,
)

# Une couleur stable par tracker_id (modulo), pour repérer visuellement
# la persistance d'une piste d'un coup d'œil.
COLORS = [
    (66, 135, 245), (66, 245, 111), (245, 66, 197), (245, 173, 66),
    (66, 245, 233), (197, 66, 245), (245, 66, 66), (144, 245, 66),
]

# Un exemple par famille de tools (agent/src/agent/tool_schemas.py), pour
# montrer d'entrée l'étendue de ce que l'agent sait faire plutôt que
# quatre variantes du même comptage. Regroupés par usage dans le message
# d'accueil (titre de groupe -> questions).
EXAMPLE_QUESTIONS = {
    "Compter": [
        "Combien de personnes y a-t-il en ce moment ?",
        "Combien de voitures sont passées depuis 5 minutes ?",
        "Combien de personnes entre 14h00 et 14h30 ?",
    ],
    "Analyser": [
        "Depuis quand n'a-t-on pas vu de voiture ?",
        "Combien de temps une personne reste-t-elle en moyenne ?",
        "Quel est le maximum de voitures vues en même temps ?",
    ],
    "Surveiller (alertes)": [
        "Préviens-moi si une personne reste plus de 10 secondes",
        "Alerte-moi si plus de 5 voitures passent en 1 minute",
        "Préviens-moi s'il y a au moins 2 personnes et 1 voiture en même temps",
    ],
    "Zones (scène chantier)": [
        "Alerte-moi si quelqu'un entre dans la zone centrale",
        "Combien de personnes sont entrées dans la zone latérale ?",
    ],
}

WELCOME_MESSAGE = (
    "Bienvenue -- Assistant EdgeSight\n"
    "\n"
    "Raccourcis (fenêtre vidéo) :\n"
    "  c = changer de source, r = redémarrer la vidéo, q = quitter\n"
    "\n"
    "Exemples de questions :\n"
    + "".join(
        f"\n  {group}\n" + "".join(f"    - {q}\n" for q in questions)
        for group, questions in EXAMPLE_QUESTIONS.items()
    )
)


def _format_tool_arg(value) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def format_tool_calls(question: str, tool_calls: list[tuple[str, dict, str]]) -> str:
    """Résumé clair et minimaliste des tools D2 appelés par l'agent pour
    UNE question -- affiché dans le panneau "Outils appelés" de la
    fenêtre Assistant (voir main()), en plus de la réponse en langage
    naturel déjà affichée dans la conversation. `tool_calls` : liste
    (nom, arguments, résultat) telle que renvoyée par
    `Agent.ask(tool_calls_log=...)`, agent/src/agent/agent.py -- déjà dans
    l'ordre d'appel, tous rounds confondus. Le résultat est la valeur
    brute renvoyée au LLM : permet de voir d'un coup d'œil si une réponse
    fausse vient du tool ou de sa reformulation par le LLM."""
    if not tool_calls:
        return f"{question}\n  (aucun outil)"
    lines = [question]
    for name, args, result in tool_calls:
        args_str = ", ".join(f"{key}={_format_tool_arg(value)}" for key, value in args.items())
        lines.append(f"  {name}({args_str})")
        lines.append(f"    → {result}")
    return "\n".join(lines)


class QueueLogHandler(logging.Handler):
    """Formate chaque enregistrement puis le dépose dans une `queue.Queue`
    -- thread-safe par construction (n'importe quel thread peut logger),
    contrairement à un widget Tkinter qui ne doit être touché que depuis
    le thread principal (voir `poll_log_queue` dans main(), seul lecteur
    de cette file, qui tourne via `root.after`)."""

    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def undo_export_sigmoid(raw_output, num_classes):
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_length=10):
    """Pas de primitive rectangle pointille native dans OpenCV -- dessine
    quatre bords en segments. Utilise pour distinguer visuellement une
    boite 'coasted' (position extrapolee par le filtre de Kalman du
    tracker, pas une vraie detection ce frame-ci) d'une boite confirmee."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def draw_tracked(frame, tracked: dict, class_names: list[str]):
    s = ui_scale(frame)
    for cls_idx, boxes in tracked.items():
        for x1, y1, x2, y2, score, tid, is_coasted, origin in boxes:
            # "predicted" (extrapolation Kalman pure, aucune detection ce
            # frame-ci) n'est jamais affiche : une piste "en roue libre" ne
            # doit pas laisser croire a une detection reelle sur l'ecran de
            # demo. "weak" (detection reelle mais rejetee par l'hysteresis,
            # score trop faible) reste affichee en pointille -- cas
            # different, une detection a bien eu lieu ce frame-ci.
            if is_coasted and origin == "predicted":
                continue
            color = COLORS[tid % len(COLORS)]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if is_coasted:
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)), round(10 * s))
                label = f"#{tid} {class_names[cls_idx]} (faible)"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)))
                label = f"#{tid} {class_names[cls_idx]} {score:.2f}"
            draw_box_label(frame, label, x1, y1, color, scale=s)
    return frame


def alert_notifier(agent: Agent, alert: dict, answers: "queue.Queue[tuple[str, str, list]]"):
    """D3 -- lance un tour de LLM en arrière-plan pour transformer un
    événement d'alerte brut en notification en langage naturel, sans
    jamais bloquer la boucle vidéo ni la fenêtre Assistant (même
    raisonnement que les questions posées par l'utilisateur : un thread
    daemon de courte durée par alerte, plutôt qu'une file persistante --
    les alertes sont rares/dédupliquées, pas un flux continu). Réutilise
    la même file `answers` que les questions utilisateur : la boucle
    vidéo n'a pas besoin de savoir d'où vient une réponse pour l'afficher.

    Appelle `agent.ask()` SANS session, délibérément : une notification
    d'alerte reste un échange isolé, jamais mêlé à l'historique de la
    conversation utilisateur -- deux threads concurrents (celui-ci et
    une question posée dans la fenêtre Assistant) ne doivent jamais se
    retrouver à muter la même liste de messages en même temps (cf.
    docstring de `Session`)."""
    pseudo_question = (
        f"[Systeme] Une alerte vient de se declencher automatiquement : "
        f"{alert['detail']}. Previens l'utilisateur en une phrase courte."
    )
    tool_calls: list[tuple[str, dict, str]] = []
    answer = agent.ask(pseudo_question, tool_calls_log=tool_calls)
    answers.put((f"[Alerte] {alert['detail']}", answer, tool_calls))


def draw_zones(frame, zones: dict):
    """Dessine le contour des zones de danger configurées (config/zones.yaml)
    -- coordonnées normalisées reconverties en pixels selon la taille réelle
    de la frame courante (voir zones.py)."""
    h, w = frame.shape[:2]
    s = ui_scale(frame)
    for name, polygon_frac in zones.items():
        pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=max(2, int(round(2 * s))),
                      lineType=cv2.LINE_AA)
        # Étiquette au sommet le plus haut du polygone, plutôt qu'au premier
        # point cliqué (souvent en bas, sur les autres boîtes).
        top = pts[np.argmin(pts[:, 1])]
        draw_box_label(frame, name, int(top[0]), int(top[1]), (0, 0, 220), size=14, scale=s)


def confirmed_only(tracked: dict) -> dict:
    """Ecarte les boites 'coasted' (extrapolees par le tracker, pas une
    vraie detection ce frame-ci) avant le journal C2 -- une extrapolation
    ne doit jamais compter comme une observation reelle pour les durees de
    presence (sinon une piste perdue continuerait de 'voir son temps
    tourner' sur une simple prediction de mouvement)."""
    return {
        cls_idx: [box for box in boxes if not box[6]]
        for cls_idx, boxes in tracked.items()
    }


WINDOW_NAME = "Démo EdgeSight"
LOG_PATH = "agent/data/03_live_agent_demo.log"


def main():
    # Deux handlers sur le logger racine : un fichier (trace complète,
    # cf. README) et une file lue par la fenêtre Logs (poll_log_queue,
    # plus bas) -- même enregistrement, deux destinations, aucune des deux
    # ne dépend de l'autre.
    log_queue: "queue.Queue[str]" = queue.Queue()
    log_formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setFormatter(log_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, queue_handler])

    # Repartition explicite des 12 coeurs logiques entre les 3 pools de
    # threads internes (ONNX Runtime, OpenCV, PyTorch) qui se disputaient
    # sinon les memes coeurs des que le thread producteur (lecture+
    # pretraitement, plus bas) tourne en parallele de l'inference -- mesure
    # sur ce poste (12 coeurs, modele 896px) : tout laisser en reglage par
    # defaut degradait le FPS reel au lieu de l'ameliorer (inference seule
    # ralentie de 35,6ms a 56ms sous contention). Ablation menee sur le
    # nombre de threads ONNX (le reste reparti entre OpenCV/PyTorch,
    # section 10.2) : 6 threads ONNX / 2 OpenCV / 4 PyTorch retenu -- 25,5
    # IPS mesures sous cette meme contention (marge ~2% au-dessus du
    # plancher de 25 IPS, section 1.1, plus fine qu'a 9 threads mais FPS
    # reel superieur, 18,2 contre 17,2) et 18,2 FPS reel, contre 12,6-13,4
    # sans repartition.
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    ort_session_options = ort.SessionOptions()
    ort_session_options.intra_op_num_threads = 6
    ort_session_options.inter_op_num_threads = 1

    load_config(cfg, CONFIG_PATH)
    model = build_model(cfg.model)  # config seule, pour post_process
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    ort_session = ort.InferenceSession(ONNX_PATH, sess_options=ort_session_options, providers=["CPUExecutionProvider"])
    tracker = MultiClassByteTracker(cfg.class_names)

    store = EventStore(db_path="agent/data/03_live_agent_demo.db")
    # Le fichier .db persiste entre deux lancements du script, mais les
    # tracker_id repartent TOUJOURS de 0 à chaque démarrage -- sans ce
    # clear, une nouvelle piste #0 hérite du first_seen d'une ligne d'une
    # session précédente (upsert : seul last_seen est mis à jour), et se
    # retrouve avec un âge de plusieurs dizaines de minutes dès la
    # première frame. Même raisonnement que le clear_events() déjà fait
    # au changement de source (CYCLE_KEY, plus bas).
    store.clear_events()
    # Idem pour les règles d'alerte posées en direct par l'agent
    # (set_duration_alert, etc.) : le fichier .db persiste entre deux
    # lancements, donc sans ce clear une alerte posée pendant une démo
    # précédente (ex. "préviens-moi si une voiture entre dans le quai de
    # chargement") réapparaîtrait au prochain lancement, sur une scène qui
    # n'a plus rien à voir. Aucune alerte pré-configurée ici par ailleurs :
    # c'est à l'agent de les poser, seulement si l'utilisateur le demande
    # explicitement -- pas par défaut.
    store.clear_alerts()
    alert_monitor = AlertMonitor(store)
    zone_monitor = ZoneMonitor(store)
    agent = Agent(store)
    fps_counter = FPSCounter()
    cycler = SourceCycler()
    # Scène active dès le départ, pas seulement au premier changement de
    # source (touche 'c') -- sinon la source affichée au tout premier
    # frame n'aurait ses zones surveillées qu'après un premier cycle.
    zone_monitor.set_scene(scene_for_source(cycler.zone_source_key))

    answers: "queue.Queue[tuple[str, str, list]]" = queue.Queue()
    # Une seule Session pour toute la durée du script -- jamais partagée
    # avec alert_notifier (voir sa docstring). L'entrée de la fenêtre
    # Assistant est désactivée pendant qu'une question est en cours
    # (voir send_question) pour garantir qu'un seul thread à la fois la
    # mute, même si l'utilisateur tape vite.
    session = Session()

    # --- Fenêtres Tkinter (Logs + Assistant) -------------------------------
    # Tkinter doit tourner sur le thread principal (contrainte de la
    # bibliothèque) -- la boucle vidéo (cv2) tourne donc dans son propre
    # thread (video_loop, plus bas), et root.mainloop() reste ici.
    root = tk.Tk()
    root.title("EdgeSight — Logs")
    root.geometry("700x300")
    log_widget = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD)
    log_widget.pack(fill=tk.BOTH, expand=True)

    chat_win = tk.Toplevel(root)
    chat_win.title("EdgeSight — Assistant")
    chat_win.geometry("950x500")

    # Conversation (gauche) + outils appelés par l'agent (droite, D2) --
    # deux panneaux indépendants dans la même fenêtre plutôt que deux
    # fenêtres séparées : ils concernent le même échange question/réponse,
    # les garder côte à côte évite d'avoir à recouper deux fenêtres
    # distinctes pour comprendre ce que l'agent a fait. grid (pas pack)
    # pour un partage de largeur explicite (60/40) qui tient sur un
    # redimensionnement de la fenêtre.
    panes = tk.Frame(chat_win)
    panes.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
    panes.columnconfigure(0, weight=3)
    panes.columnconfigure(1, weight=2)
    panes.rowconfigure(1, weight=1)

    tk.Label(panes, text="Conversation", anchor="w").grid(row=0, column=0, sticky="ew")
    chat_widget = scrolledtext.ScrolledText(panes, state=tk.DISABLED, wrap=tk.WORD)
    chat_widget.grid(row=1, column=0, sticky="nsew")

    tk.Label(panes, text="Outils appelés", anchor="w").grid(row=0, column=1, sticky="ew", padx=(6, 0))
    tools_widget = scrolledtext.ScrolledText(panes, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
    tools_widget.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

    entry_frame = tk.Frame(chat_win)
    entry_frame.pack(fill=tk.X, padx=6, pady=6)
    chat_entry = tk.Entry(entry_frame)
    chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    send_button = tk.Button(entry_frame, text="Envoyer")
    send_button.pack(side=tk.LEFT, padx=(6, 0))

    def append_chat(text: str) -> None:
        chat_widget.config(state=tk.NORMAL)
        chat_widget.insert(tk.END, text + "\n\n")
        chat_widget.see(tk.END)
        chat_widget.config(state=tk.DISABLED)

    def append_tools(text: str) -> None:
        tools_widget.config(state=tk.NORMAL)
        tools_widget.insert(tk.END, text + "\n\n")
        tools_widget.see(tk.END)
        tools_widget.config(state=tk.DISABLED)

    append_chat(WELCOME_MESSAGE)

    def ask_and_queue(question: str) -> None:
        tool_calls: list[tuple[str, dict, str]] = []
        answer = agent.ask(question, session=session, tool_calls_log=tool_calls)
        answers.put((question, answer, tool_calls))
        root.after(0, reenable_chat_input)

    def reenable_chat_input() -> None:
        chat_entry.config(state=tk.NORMAL)
        send_button.config(state=tk.NORMAL)
        chat_entry.focus_set()

    def send_question(event=None) -> None:
        question = chat_entry.get().strip()
        if not question:
            return
        chat_entry.delete(0, tk.END)
        chat_entry.config(state=tk.DISABLED)
        send_button.config(state=tk.DISABLED)
        append_chat(f"Vous : {question}")
        threading.Thread(target=ask_and_queue, args=(question,), daemon=True).start()

    chat_entry.bind("<Return>", send_question)
    send_button.config(command=send_question)
    chat_entry.focus_set()

    def poll_log_queue() -> None:
        try:
            while True:
                line = log_queue.get_nowait()
                log_widget.config(state=tk.NORMAL)
                log_widget.insert(tk.END, line + "\n")
                log_widget.see(tk.END)
                log_widget.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        root.after(200, poll_log_queue)

    root.after(200, poll_log_queue)

    # Fermer n'importe laquelle des deux fenêtres Tkinter arrête tout
    # proprement (video_loop détecte shutdown_event et nettoie
    # tracker/journal/capture avant de quitter).
    shutdown_event = threading.Event()

    def on_close() -> None:
        shutdown_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    chat_win.protocol("WM_DELETE_WINDOW", on_close)

    # --- Boucle vidéo (thread dédié) ---------------------------------------
    def video_loop() -> None:
        # WINDOW_NORMAL + resizeWindow explicite à chaque frame (plus
        # bas) : WINDOW_AUTOSIZE (le défaut) ne redimensionne pas
        # toujours la fenêtre de façon fiable en changeant de source vers
        # une image de dimensions différentes -- un bandeau bas ajouté
        # après coup s'est déjà retrouvé rogné hors de la fenêtre dans ce
        # cas, alors qu'il était bien dessiné sur l'image elle-même.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, lambda event, x, y, flags, param: cycler.on_mouse(event, x, y, flags))

        # --- Thread producteur : lecture + pretraitement, en parallele de
        # l'inference (section 10.2 du rapport) ----------------------------
        # Avant ce changement, chaque frame etait lue+pretraitee (~24ms
        # mesures : warp/resize+normalize+conversion tenseur) PUIS inferee
        # (~36ms) de facon strictement sequentielle -- les deux couts
        # s'additionnaient. Ce thread dedie lit et pretraite en continu la
        # frame suivante pendant que ce thread execute l'inference ONNX sur
        # la precedente : les deux couts se chevauchent au lieu de
        # s'additionner. Seul ce thread touche `cycler` (ni son etat --
        # self.cap, self.index -- ni ses methodes) : tout changement de
        # source (touches c/r) passe par la file `source_commands` plutot
        # que d'etre applique directement par ce thread, pour eviter une
        # mutation concurrente de `cycler.cap` depuis deux threads a la
        # fois. `frame_queue` (taille 2) transporte soit une frame
        # pretraitee prete pour l'inference, soit une notification de
        # changement de source pour que ce thread applique les MEMES resets
        # (tracker/journal/alertes/session) qu'avant, juste de facon
        # asynchrone.
        frame_queue: "queue.Queue[tuple[str, object]]" = queue.Queue(maxsize=2)
        source_commands: "queue.Queue[str]" = queue.Queue()
        producer_stop = threading.Event()

        def frame_producer():
            # Compte les echecs de lecture CONSECUTIFS (hoquet transitoire
            # de decodage, pas des fichiers geres separement via
            # loop_if_file) -- un raté isolé ne doit pas arreter tout le
            # script des la premiere frame ratee. Remis a zero des qu'une
            # frame est lue avec succes.
            read_fail_streak = 0
            MAX_READ_FAIL_STREAK = 30
            while not producer_stop.is_set():
                try:
                    while True:
                        cmd = source_commands.get_nowait()
                        if cmd == "cycle":
                            cycler.next()
                            read_fail_streak = 0
                            frame_queue.put(("reset", ("cycle", cycler.zone_source_key)))
                        elif cmd == "restart":
                            if cycler.is_file():
                                cycler.loop_if_file()
                                frame_queue.put(("reset", ("restart", None)))
                except queue.Empty:
                    pass

                ret, frame = cycler.read()
                if not ret:
                    if cycler.is_file():
                        cycler.loop_if_file()
                        continue
                    read_fail_streak += 1
                    if read_fail_streak >= MAX_READ_FAIL_STREAK:
                        frame_queue.put(("source_failed", None))
                        return
                    time.sleep(0.03)
                    continue
                read_fail_streak = 0

                img_info = {"id": 0, "file_name": None, "height": frame.shape[0], "width": frame.shape[1]}
                meta = dict(img_info=img_info, raw_img=frame, img=frame)
                meta = pipeline(None, meta, INPUT_SIZE)
                meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
                meta = naive_collate([meta])
                meta["img"] = stack_batch_img(meta["img"], divisible=32)
                onnx_input = meta["img"].numpy().astype(np.float32)

                payload = (frame, onnx_input, meta, cycler.label, cycler.is_interactive())
                try:
                    frame_queue.put(("frame", payload), timeout=1.0)
                except queue.Full:
                    pass  # thread principal a l'air arrete -- on retente au tour suivant

        producer = threading.Thread(target=frame_producer, daemon=True)
        producer.start()

        print("Pipeline live E1/E2 pret -- voir les fenêtres Logs et Assistant.")
        print(f"Detail des tools/arguments choisis par l'agent + alertes D3 : {LOG_PATH}")

        alert_flash_until = 0.0
        alert_flash_text = ""

        try:
            while not shutdown_event.is_set():
                try:
                    kind, payload = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if kind == "source_failed":
                    print("Lecture de la source échouée, arrêt.")
                    shutdown_event.set()
                    root.after(0, on_close)
                    break
                if kind == "reset":
                    # Memes resets qu'avant (cf. touches CYCLE/RESTART plus
                    # bas), juste declenches ici une fois que le thread
                    # producteur a effectivement applique le changement de
                    # source, plutot que directement au moment de la touche.
                    reset_kind, zone_key = payload
                    tracker.reset()
                    store.clear_events()
                    if reset_kind == "cycle":
                        store.clear_alerts()
                        alert_monitor.reset()
                        zone_monitor.set_scene(scene_for_source(zone_key))
                        session.reset()
                    alert_flash_until = 0.0
                    continue

                frame, onnx_input, meta, source_label, is_interactive = payload

                raw_output = ort_session.run(None, {"input": onnx_input})[0]

                preds = torch.from_numpy(undo_export_sigmoid(raw_output, cfg.model.arch.head.num_classes))
                results = model.head.post_process(preds, meta)
                dets = next(iter(results.values()))
                for cls_idx, cls_name in enumerate(cfg.class_names):
                    if cls_name in DISABLED_CLASSES:
                        dets[cls_idx] = []

                now = time.time()
                tracked = tracker.update(dets, timestamp=now)
                store.update(confirmed_only(tracked), cfg.class_names, timestamp=now)

                # zone_monitor ne surveille que les zones de la scène active
                # (set_scene, appelé à chaque changement de source) -- sur
                # une source sans scène configurée dans config/zones.yaml,
                # check() ne fait rien et renvoie toujours [], pas besoin de
                # condition ici.
                zone_alerts = zone_monitor.check(tracked, cfg.class_names, frame.shape, now=now)
                fired = alert_monitor.check(now=now) + zone_alerts
                for alert in fired:
                    # Silencieux sur la console -- seulement tracé dans
                    # LOG_PATH/la fenêtre Logs, le retour utilisateur passe
                    # par la bannière + le tour de LLM ci-dessous.
                    message = f"[D3] ALERTE ({alert['alert_type']}) : {alert['detail']}"
                    logging.info(message)
                    alert_flash_text = message
                    alert_flash_until = now + 3.0
                    threading.Thread(target=alert_notifier, args=(agent, alert, answers), daemon=True).start()

                # E2 : une réponse de l'agent est-elle arrivée depuis la
                # dernière frame -- question utilisateur (fenêtre Assistant)
                # ou notification d'alerte (alert_notifier ci-dessus), la
                # file ne distingue pas les deux, l'affichage n'a pas besoin
                # de le savoir. Relayée vers la fenêtre Assistant via
                # root.after (seule façon thread-safe de toucher un widget
                # Tkinter depuis ce thread) -- affichée UNIQUEMENT là, plus
                # en bandeau vidéo (redondant : l'écho "Vous : ..." de la
                # question a déjà été affiché par send_question au moment
                # de la saisie, donc seule la réponse est renvoyée ici --
                # sauf pour une alerte, jamais échoée ailleurs). Le panneau
                # "Outils appelés" (droite de la fenêtre Assistant) reçoit
                # le même résumé quelle que soit l'origine (question ou
                # alerte), formaté par format_tool_calls.
                try:
                    question, answer, tool_calls = answers.get_nowait()
                    if question.startswith("[Alerte]"):
                        root.after(0, append_chat, f"{question}\nAgent : {answer}")
                    else:
                        root.after(0, append_chat, f"Agent : {answer}")
                    root.after(0, append_tools, format_tool_calls(question, tool_calls))
                except queue.Empty:
                    pass

                result_frame = draw_tracked(frame, tracked, cfg.class_names)
                # zone_monitor.zones est vide sans scène active pour cette
                # source (voir plus haut) -- draw_zones n'affiche alors
                # simplement rien, pas besoin de condition ici non plus.
                draw_zones(result_frame, zone_monitor.zones)
                draw_fps(result_frame, fps_counter.tick())
                draw_source_label(result_frame, source_label)
                draw_interactive_help(result_frame, is_interactive)

                if now < alert_flash_until:
                    draw_banner(result_frame, alert_flash_text, (0, 0, 200))

                result_frame = resize_for_display(result_frame)

                cv2.resizeWindow(WINDOW_NAME, result_frame.shape[1], result_frame.shape[0])
                cv2.imshow(WINDOW_NAME, result_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    shutdown_event.set()
                    root.after(0, on_close)
                    break
                # Le changement de source lui-meme (cycler.next()/
                # loop_if_file()) est applique par le thread producteur,
                # seul a toucher `cycler` -- ce thread ne fait que deposer
                # la demande ; les resets (tracker/journal/alertes/session)
                # suivront via le message ("reset", ...) traite plus haut,
                # une fois la source effectivement changee.
                if key == CYCLE_KEY:
                    source_commands.put("cycle")
                if key == RESTART_KEY:
                    source_commands.put("restart")  # no-op cote producteur si pas un fichier
        finally:
            producer_stop.set()
            # Draine la file pour ne pas laisser le producteur bloque sur un
            # put() si elle etait pleine au moment de l'arret, puis attend
            # qu'il quitte sa boucle avant de toucher cycler.release() --
            # sinon read()/release() pourraient s'executer concurremment
            # sur le meme cv2.VideoCapture depuis deux threads.
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
            producer.join(timeout=2.0)
            cycler.release()
            cv2.destroyAllWindows()
            store.close()

    video_thread = threading.Thread(target=video_loop, daemon=True)
    video_thread.start()

    root.mainloop()

    # root.mainloop() revient soit parce que 'q' a été pressé dans la
    # fenêtre vidéo (qui a déjà appelé on_close via root.after), soit
    # parce qu'une fenêtre Tkinter a été fermée directement -- dans ce
    # dernier cas, video_loop ne le sait pas encore : shutdown_event le
    # lui signale, puis on attend qu'il ait fini son nettoyage avant de
    # laisser le processus se terminer.
    shutdown_event.set()
    video_thread.join(timeout=3.0)

    # LOG_PATH n'a de sens que pendant CETTE session (trace des tools/
    # alertes du run en cours, cf. docstring du module) -- supprimé
    # plutôt que laissé tronqué sur disque entre deux lancements, pour ne
    # jamais confondre un fichier vide avec "aucune activité loggée cette
    # session-ci". `close()` d'abord : Windows refuse de supprimer un
    # fichier encore ouvert en écriture.
    file_handler.close()
    try:
        os.remove(LOG_PATH)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    # Lance llama-server s'il ne tourne pas déjà, et l'arrête à la
    # sortie (voir agent/src/agent/llm_server.py, config/agent.yaml).
    with llama_server():
        main()
