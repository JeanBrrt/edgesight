"""Démo complète : détection, suivi, journal, alertes et assistant
conversationnel

2 fenêtres :
  - vidéo : détections, zones et bannière d'alerte ;
  - Assistant : conversation, et à droite les outils appelés avec leur
    résultat.

Threads :
  - principal : Tkinter ;
  - boucle vidéo : inférence, suivi, alertes, affichage ;
  - producteur : lecture et prétraitement de l'image suivante, en
    parallèle de l'inférence ;
  - un thread court par question ou par alerte

Usage :
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

sys.path.insert(0, "demo/src/common")  # modules partagés par les 3 démos
from nanodet_setup import build_postprocessor  # avant nanodet : filtre ses avertissements

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.util import cfg

sys.path.insert(0, ".")  # suivi, journal, alertes et agent vivent dans agent/src/
from agent.src.tracking.tracker import MultiClassByteTracker
from agent.src.journal.event_store import EventStore
from agent.src.alerts.alerts import AlertMonitor
from agent.src.alerts.zones import ZoneMonitor, scene_for_source
from agent.src.agent.agent import Agent, Session
from agent.src.agent.llm_server import llama_server

from fps_counter import FPSCounter, draw_fps
from source_cycle import (
    SourceCycler,
    CYCLE_KEY,
    RESTART_KEY,
    draw_source_label,
    draw_controls,
    draw_interactive_help,
    draw_banner,
    draw_zones,
    resize_for_display,
    ui_scale,
)
from text_render import draw_box_label
from power import disable_power_throttling
from config import (
    CONFIG_PATH,
    ONNX_PATH,
    INPUT_SIZE,
    DISABLED_CLASSES,
)

# Une couleur fixe par identifiant, pour suivre une piste d'un coup d'œil.
COLORS = [
    (66, 135, 245),
    (66, 245, 111),
    (245, 66, 197),
    (245, 173, 66),
    (66, 245, 233),
    (197, 66, 245),
    (245, 66, 66),
    (144, 245, 66),
]

# Exemples du message d'accueil, couvrant chaque famille d'outils de
# l'agent (agent/src/agent/tool_schemas.py).
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
    """Texte du panneau "Outils appelés" pour une question : chaque outil,
    ses arguments et la valeur brute renvoyée au LLM. Permet de voir si
    une réponse fausse vient de l'outil ou de sa reformulation."""
    if not tool_calls:
        return f"{question}\n  (aucun outil)"
    lines = [question]
    for name, args, result in tool_calls:
        args_str = ", ".join(f"{key}={_format_tool_arg(value)}" for key, value in args.items())
        lines.append(f"  {name}({args_str})")
        lines.append(f"    → {result}")
    return "\n".join(lines)


def undo_export_sigmoid(raw_output, num_classes):
    cls, reg = np.split(raw_output, [num_classes], axis=-1)
    cls = np.clip(cls, 1e-7, 1 - 1e-7)
    logits = np.log(cls / (1 - cls))
    return np.concatenate([logits, reg], axis=-1)


def draw_dashed_rect(frame, pt1, pt2, color, thickness=2, dash_length=10):
    """Rectangle en pointillés (absent d'OpenCV), pour les détections
    faibles."""
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
            # Position seulement prédite (aucune détection) : non affichée.
            # Détection réelle mais sous le seuil ("weak") : en pointillés.
            if is_coasted and origin == "predicted":
                continue
            color = COLORS[tid % len(COLORS)]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if is_coasted:
                draw_dashed_rect(
                    frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)), round(10 * s)
                )
                label = f"#{tid} {class_names[cls_idx]} (faible)"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(2 * s)))
                label = f"#{tid} {class_names[cls_idx]} {score:.2f}"
            draw_box_label(frame, label, x1, y1, color, scale=s)
    return frame


def alert_notifier(agent: Agent, alert: dict, answers: "queue.Queue[tuple[str, str, list]]"):
    """Fait reformuler une alerte par le LLM, dans son propre thread.

    Sans session : l'alerte ne se mêle pas à la conversation, et ce
    thread ne modifie jamais l'historique en même temps qu'une question."""
    pseudo_question = (
        f"[Systeme] Une alerte vient de se declencher automatiquement : "
        f"{alert['detail']}. Previens l'utilisateur en une phrase courte."
    )
    tool_calls: list[tuple[str, dict, str]] = []
    answer = agent.ask(pseudo_question, tool_calls_log=tool_calls)
    answers.put((f"[Alerte] {alert['detail']}", answer, tool_calls))


def confirmed_only(tracked: dict) -> dict:
    """Ne garde que les vraies détections pour le journal : une position
    prédite ne doit pas prolonger une durée de présence."""
    return {cls_idx: [box for box in boxes if not box[6]] for cls_idx, boxes in tracked.items()}


WINDOW_NAME = "Démo EdgeSight"
LOG_PATH = "agent/data/03_live_agent_demo.log"


def main():
    disable_power_throttling()
    # Outils appelés et alertes, tracés dans LOG_PATH.
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler])

    # Répartition des 12 coeurs entre ONNX Runtime, OpenCV et PyTorch, qui
    # se les disputent sinon (inférence ralentie de 35,6 à 56 ms). 6/2/4
    # donne 18,2 FPS réels contre 12,6-13,4 par défaut (rapport section 10.2).
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    ort_session_options = ort.SessionOptions()
    ort_session_options.intra_op_num_threads = 6
    ort_session_options.inter_op_num_threads = 1

    model = build_postprocessor(CONFIG_PATH)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    ort_session = ort.InferenceSession(
        ONNX_PATH, sess_options=ort_session_options, providers=["CPUExecutionProvider"]
    )
    tracker = MultiClassByteTracker(cfg.class_names)

    store = EventStore(db_path="agent/data/03_live_agent_demo.db")
    # La base survit entre deux lancements : on la vide. Sinon la piste #0
    # hériterait de l'ancienneté d'une ancienne piste #0, et les alertes
    # posées lors d'une démo précédente reviendraient.
    store.clear_events()
    store.clear_alerts()
    alert_monitor = AlertMonitor(store)
    zone_monitor = ZoneMonitor(store)
    agent = Agent(store)
    fps_counter = FPSCounter()
    cycler = SourceCycler()
    # Zones de la première source surveillées dès le départ.
    zone_monitor.set_scene(scene_for_source(cycler.zone_source_key))

    answers: "queue.Queue[tuple[str, str, list]]" = queue.Queue()
    # Historique de la conversation. La saisie est bloquée pendant qu'une
    # question est en cours, pour qu'un seul thread à la fois le modifie.
    session = Session()

    # --- Fenêtre Assistant (Tkinter), sur le thread principal --------------
    root = tk.Tk()
    root.title("EdgeSight — Assistant")
    root.geometry("950x500")

    # Conversation à gauche, outils appelés à droite (largeurs 60/40).
    panes = tk.Frame(root)
    panes.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
    panes.columnconfigure(0, weight=3)
    panes.columnconfigure(1, weight=2)
    panes.rowconfigure(1, weight=1)

    tk.Label(panes, text="Conversation", anchor="w").grid(row=0, column=0, sticky="ew")
    chat_widget = scrolledtext.ScrolledText(panes, state=tk.DISABLED, wrap=tk.WORD)
    chat_widget.grid(row=1, column=0, sticky="nsew")

    tk.Label(panes, text="Outils appelés", anchor="w").grid(
        row=0, column=1, sticky="ew", padx=(6, 0)
    )
    tools_widget = scrolledtext.ScrolledText(
        panes, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9)
    )
    tools_widget.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

    entry_frame = tk.Frame(root)
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

    # Fermer la fenêtre Assistant arrête toute la démo.
    shutdown_event = threading.Event()

    def on_close() -> None:
        shutdown_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # --- Boucle vidéo (thread dédié) ---------------------------------------
    def video_loop() -> None:
        # WINDOW_NORMAL + resizeWindow à chaque frame : en mode AUTOSIZE,
        # la fenêtre ne suit pas toujours un changement de taille de source.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            WINDOW_NAME, lambda event, x, y, flags, param: cycler.on_mouse(event, x, y, flags)
        )

        # --- Thread producteur (rapport section 10.2) -----------------------
        # Il prépare l'image suivante (~24 ms) pendant l'inférence de la
        # courante (~36 ms), au lieu de les enchaîner. Lui seul touche à
        # `cycler` : les touches c/r lui arrivent par `source_commands`.
        # `frame_queue` lui sert à renvoyer les images prêtes, ou un signal
        # de changement de source.
        frame_queue: "queue.Queue[tuple[str, object]]" = queue.Queue(maxsize=2)
        source_commands: "queue.Queue[str]" = queue.Queue()
        producer_stop = threading.Event()

        def frame_producer():
            # Tolère quelques échecs de lecture consécutifs avant d'abandonner.
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

                img_info = {
                    "id": 0,
                    "file_name": None,
                    "height": frame.shape[0],
                    "width": frame.shape[1],
                }
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
                    pass  # boucle vidéo bloquée ou arrêtée : on réessaie au tour suivant

        producer = threading.Thread(target=frame_producer, daemon=True)
        producer.start()

        print("Démo prête -- voir la fenêtre Assistant.")
        print(f"Outils appelés par l'agent et alertes : {LOG_PATH}")

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
                    # Le producteur a changé ou relancé la source : on repart
                    # de zéro (et, sur une nouvelle scène, sans alertes).
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

                preds = torch.from_numpy(
                    undo_export_sigmoid(raw_output, cfg.model.arch.head.num_classes)
                )
                results = model.head.post_process(preds, meta)
                dets = next(iter(results.values()))
                for cls_idx, cls_name in enumerate(cfg.class_names):
                    if cls_name in DISABLED_CLASSES:
                        dets[cls_idx] = []

                now = time.time()
                tracked = tracker.update(dets, timestamp=now)
                store.update(confirmed_only(tracked), cfg.class_names, timestamp=now)

                # Sans zones configurées pour cette source, check() renvoie [].
                zone_alerts = zone_monitor.check(tracked, cfg.class_names, frame.shape, now=now)
                fired = alert_monitor.check(now=now) + zone_alerts
                for alert in fired:
                    # Rien sur la console : logs, bannière, puis message du LLM.
                    message = f"ALERTE ({alert['alert_type']}) : {alert['detail']}"
                    logging.info(message)
                    alert_flash_text = message
                    alert_flash_until = now + 3.0
                    threading.Thread(
                        target=alert_notifier, args=(agent, alert, answers), daemon=True
                    ).start()

                # Réponse arrivée (question ou alerte) : relayée à la fenêtre
                # Assistant via root.after, seul accès sûr à Tkinter. La
                # question est déjà affichée, sauf pour une alerte.
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
                draw_zones(result_frame, zone_monitor.zones)
                draw_fps(result_frame, fps_counter.tick())
                draw_source_label(result_frame, source_label)
                draw_controls(result_frame)
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
                # Transmis au producteur, qui renverra un "reset".
                if key == CYCLE_KEY:
                    source_commands.put("cycle")
                if key == RESTART_KEY:
                    source_commands.put("restart")  # ignoré sur la scène interactive
        finally:
            producer_stop.set()
            # Vide la file pour débloquer le producteur, puis attend qu'il
            # s'arrête avant de libérer la vidéo qu'il lit peut-être encore.
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

    # Fin de mainloop ('q' ou fenêtre fermée) : on prévient la boucle
    # vidéo et on attend qu'elle ait fini de nettoyer.
    shutdown_event.set()
    video_thread.join(timeout=3.0)

    # Le log ne concerne que la session en cours : supprimé à la sortie
    # (fermé d'abord, Windows refuse de supprimer un fichier ouvert).
    file_handler.close()
    try:
        os.remove(LOG_PATH)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    # Lance llama-server s'il ne tourne pas déjà, l'arrête à la sortie.
    with llama_server():
        main()
