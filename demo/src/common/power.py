"""Empêche Windows 11 de brider le processus (EcoQoS).

Windows juge parfois un processus lancé depuis un terminal peu prioritaire
et le bride : cœurs économes (E-cores) et fréquence basse. Sur un
processeur hybride, l'inférence passe alors de ~37 ms à ~500 ms par image.
Sans effet hors de Windows.
"""

import ctypes
import sys

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_EXECUTION_SPEED = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def disable_power_throttling() -> bool:
    """Demande à Windows de ne jamais brider ce processus. Renvoie False si
    ce n'est pas possible (autre système, Windows trop ancien)."""
    if sys.platform != "win32":
        return False
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    # Contrôle la vitesse d'exécution (ControlMask) et la veut non bridée (StateMask = 0).
    state = _PowerThrottlingState(1, _EXECUTION_SPEED, 0)
    try:
        return bool(kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)
        ))
    except (AttributeError, OSError):
        return False
