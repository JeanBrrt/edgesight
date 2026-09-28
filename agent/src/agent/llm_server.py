"""Lancement automatique de llama-server pour 03_live_agent_demo.py --
évite d'avoir à ouvrir un second terminal avant chaque démo.

`llama_server()` est un context manager :
  - un serveur répond déjà sur llama_server_url -> réutilisé tel quel,
    jamais arrêté à la sortie (il a été lancé par quelqu'un d'autre) ;
  - sinon, si `llama_server.auto_start` est vrai dans config/agent.yaml ->
    lancé en sous-processus, attendu jusqu'à ce que /health réponde, puis
    arrêté à la sortie du `with` (même sur exception) ;
  - dans tous les autres cas (auto_start désactivé, binaire/modèle
    absent, démarrage en échec) -> simple avertissement, la démo continue
    quand même : la vidéo, le tracking et les alertes n'ont pas besoin du
    LLM, seules les questions à l'assistant échoueront.

Stdlib uniquement (subprocess/urllib/ctypes) : pas de dépendance ajoutée
pour un simple appel HTTP de santé.

Sous Windows, le serveur lancé ici est rattaché à un Job Object
"kill on close" : il meurt avec la démo même si celle-ci ne passe pas
par le `finally` (plantage, console fermée, processus tué). Sans ça, un
llama-server orphelin garderait ~3 Go de VRAM, et le lancement suivant
le "réutiliserait" sans jamais l'arrêter.
"""

import contextlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from ._config import LLAMA_SERVER_LAUNCH, LLAMA_SERVER_URL, PROJECT_ROOT

_HEALTH_POLL_SECONDS = 0.5
_STOP_TIMEOUT_SECONDS = 10.0

# Handle du Job Object (Windows) -- gardé en global : c'est sa fermeture,
# à la fin du processus Python quelle qu'en soit la cause, qui tue le
# serveur rattaché.
_job_handle = None


def _log(message: str) -> None:
    # flush : visible immédiatement même si stdout est redirigé vers un
    # fichier (sinon bufferisé, et perdu si le processus est tué).
    print(f"[llama-server] {message}", flush=True)


def _kill_with_parent(process: subprocess.Popen) -> None:
    """Windows uniquement -- rattache `process` à un Job Object configuré
    avec JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Échec silencieux (simple
    avertissement) : l'arrêt normal via `_stop_server` reste en place."""
    global _job_handle
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    ok = job and kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(limits), ctypes.sizeof(limits)
    ) and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle))
    if not ok:
        _log(f"rattachement au Job Object impossible (erreur {ctypes.get_last_error()}) -- "
             "le serveur pourrait survivre à un arrêt brutal de la démo.")
        return
    _job_handle = job


def _health_url() -> str:
    """http://host:port/v1 -> http://host:port/health (/health est à la
    racine du serveur, pas sous le préfixe OpenAI /v1)."""
    parts = urlsplit(LLAMA_SERVER_URL)
    return f"{parts.scheme}://{parts.netloc}/health"


def is_server_ready() -> bool:
    """Vrai seulement quand le modèle est chargé : /health renvoie 503
    pendant le chargement, 200 une fois prêt."""
    try:
        with urllib.request.urlopen(_health_url(), timeout=1.0) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _start_server() -> subprocess.Popen | None:
    binary = _resolve(LLAMA_SERVER_LAUNCH["binary"])
    model = _resolve(LLAMA_SERVER_LAUNCH["model"])
    for label, path in (("binaire llama-server", binary), ("modèle GGUF", model)):
        if not path.exists():
            _log(f"{label} introuvable : {path} -- voir README (étapes 5-6). "
                  "Démo lancée sans LLM.")
            return None

    parts = urlsplit(LLAMA_SERVER_URL)
    command = [
        str(binary), "-m", str(model),
        *[str(arg) for arg in LLAMA_SERVER_LAUNCH.get("args", [])],
        "--host", parts.hostname, "--port", str(parts.port),
    ]
    log_path = _resolve(LLAMA_SERVER_LAUNCH.get("log_path", "agent/data/llama-server.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"démarrage ({model.name}), sortie dans {log_path}")
    # Le fichier reste ouvert côté parent le temps de passer le handle au
    # sous-processus, qui en garde sa propre copie -- fermé aussitôt ici.
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        _kill_with_parent(process)

    timeout = float(LLAMA_SERVER_LAUNCH.get("startup_timeout_seconds", 120))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _log(f"arrêté pendant le démarrage (code {process.returncode}) -- "
                  f"voir {log_path}. Démo lancée sans LLM.")
            return None
        if is_server_ready():
            _log("prêt.")
            return process
        time.sleep(_HEALTH_POLL_SECONDS)

    _log(f"pas prêt après {timeout:.0f}s -- voir {log_path}. Démo lancée sans LLM.")
    _stop_server(process)
    return None


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _log("arrêté.")


@contextlib.contextmanager
def llama_server():
    if is_server_ready():
        _log(f"déjà en cours sur {LLAMA_SERVER_URL} -- réutilisé.")
        yield
        return
    if not LLAMA_SERVER_LAUNCH.get("auto_start", False):
        _log(f"aucun serveur sur {LLAMA_SERVER_URL} et auto_start désactivé "
              "(config/agent.yaml) -- à lancer à la main (README, étape 8).")
        yield
        return

    process = _start_server()
    try:
        yield
    finally:
        if process is not None:
            _stop_server(process)
