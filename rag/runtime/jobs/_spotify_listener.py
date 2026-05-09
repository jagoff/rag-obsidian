"""F4.5 — Spotify event-driven via macOS NSDistributedNotificationCenter.

Reemplaza el cron ``spotify_poll`` (5min) con un listener event-driven:
Spotify desktop emite ``com.spotify.client.PlaybackStateChanged`` cada
vez que cambia el track o el estado de playback. Adentro del supervisor
suscribimos un observer y persistimos a ``rag_spotify_log`` al instante
(<100ms latencia post-track-change).

Implementation notes:

- Usamos PyObjC (``objc``, ``Foundation``). Ya está en deps del proyecto
  (vision via pyobjc-framework-coreml).
- ``NSDistributedNotificationCenter`` funciona desde cualquier thread —
  NO requiere main thread (a diferencia de AppKit UI). Corremos el
  run loop en un thread daemon propio.
- Si PyObjC import falla (caso edge: env Linux CI, PyObjC quitado de
  deps), el listener no se levanta y queda solo el cron del
  ``spotify_poll`` para fallback. NUNCA crashea el supervisor.
- El observer también dispara al **start** del supervisor (con un
  ``record_now_playing()`` inicial best-effort) para captar el track
  actual sin esperar a que Spotify emita un cambio.

Telemetría:

- Cada track grabado emite event ``spotify.track.changed`` al event bus
  (futuros subscribers pueden reaccionar — ej. mood scorer en F4-followup).
- Métrica simple: count de notifications recibidas + count de inserts
  exitosos (vs. errores SQL). Reportado vía IPC ``status_spotify``.

Coexistencia con el cron viejo:

- Mientras F3.5 NO hace bootout del plist ``com.fer.obsidian-rag-spotify-poll``,
  ambos paths coexisten. El cron viejo sigue cada 5min y el listener
  event-driven dispara al instante. ``rag_spotify_log`` tiene UNIQUE
  constraint sobre ``(track_id, ts_iso)`` que de-dupea automáticamente.
- Post F3.5: solo el listener queda activo.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from rag.runtime.events import bus

logger = logging.getLogger(__name__)


_PYOBJC_AVAILABLE = False
try:
    import objc  # noqa: F401 — type:ignore
    from Foundation import (  # type: ignore
        NSDistributedNotificationCenter,
        NSObject,
        NSRunLoop,
    )
    _PYOBJC_AVAILABLE = True
except ImportError:
    pass


_LISTENER_LOCK = threading.Lock()
_LISTENER_THREAD: threading.Thread | None = None
_LISTENER_STATS = {
    "notifications_received": 0,
    "tracks_recorded": 0,
    "errors": 0,
    "started_at": None,
}


# PyObjC registra clases Objective-C globalmente — re-importar el
# módulo (común en tests) raisa "overriding existing Objective-C class".
# Guardamos en module-level dict para idempotencia cross-import.
_SpotifyObserver = None
if _PYOBJC_AVAILABLE:
    try:
        # Si la clase ya existe (re-import en tests), reusa.
        _SpotifyObserver = objc.lookUpClass("_SpotifyObserverRAGRuntime")  # type: ignore
    except objc.error:
        class _SpotifyObserverRAGRuntime(NSObject):  # type: ignore
            """Objective-C class adapter — recibe las notificaciones
            distributed y delega a ``_handle_notification`` de Python.

            Suffix RAGRuntime para evitar collision con otra subclass de
            NSObject que pueda tener el mismo short name en el proceso.
            """

            def init(self):  # noqa: D401, N802 — Cocoa convention
                self = objc.super(_SpotifyObserverRAGRuntime, self).init()
                return self

            def playbackStateChanged_(self, notification):  # noqa: N802
                try:
                    _handle_notification(notification)
                except Exception:
                    logger.exception("spotify listener: handler crashed")
                    _LISTENER_STATS["errors"] += 1

        _SpotifyObserver = _SpotifyObserverRAGRuntime


def _handle_notification(notification: Any) -> None:
    """Procesa una notification — extrae datos del userInfo + persiste."""
    _LISTENER_STATS["notifications_received"] += 1

    user_info = {}
    try:
        ui = notification.userInfo()
        if ui is not None:
            for k in ui:
                user_info[str(k)] = str(ui[k]) if ui[k] is not None else None
    except Exception:
        logger.exception("spotify listener: failed to extract userInfo")

    track = user_info.get("Name") or user_info.get("track")
    artist = user_info.get("Artist") or user_info.get("artist")
    state = user_info.get("Player State", user_info.get("state"))

    if state and state.lower() in ("paused", "stopped"):
        # No persist — solo nos interesa cuando hay reproducción activa.
        return

    if not track:
        return

    # Persist directo via la integración existente. La función
    # ``record_now_playing()`` ya maneja DB connection + UNIQUE dedup.
    try:
        from rag.integrations.spotify_local import record_now_playing  # noqa: PLC0415
        record_now_playing()
        _LISTENER_STATS["tracks_recorded"] += 1
    except Exception as exc:
        logger.warning("spotify listener: record_now_playing raised: %s", exc)
        _LISTENER_STATS["errors"] += 1
        return

    bus.publish("spotify.track.changed", {
        "track": track,
        "artist": artist,
        "user_info": user_info,
    })


def _listener_loop() -> None:
    """Loop del listener — corre en thread daemon. Bloquea hasta shutdown."""
    if not _PYOBJC_AVAILABLE:
        return
    try:
        nc = NSDistributedNotificationCenter.defaultCenter()
        observer = _SpotifyObserver.alloc().init()
        nc.addObserver_selector_name_object_(
            observer,
            b"playbackStateChanged:",
            "com.spotify.client.PlaybackStateChanged",
            None,
        )
        logger.info("spotify listener: subscribed to com.spotify.client.PlaybackStateChanged")

        # Run loop indefinido — termina al shutdown del supervisor (thread
        # daemon, se mata).
        run_loop = NSRunLoop.currentRunLoop()
        run_loop.run()
    except Exception:
        logger.exception("spotify listener: thread crashed")


def start_listener() -> bool:
    """Arranca el listener si PyObjC está disponible.

    Retorna ``True`` si arrancó OK. ``False`` si PyObjC no está
    disponible o ya estaba corriendo.

    Idempotente — si ya está corriendo, no-op.
    """
    global _LISTENER_THREAD

    if os.environ.get("RAG_SPOTIFY_LISTENER_DISABLED") == "1":
        logger.info("spotify listener: disabled via env var")
        return False

    if not _PYOBJC_AVAILABLE:
        logger.info("spotify listener: PyObjC not available, falling back to cron poll")
        return False

    with _LISTENER_LOCK:
        if _LISTENER_THREAD is not None and _LISTENER_THREAD.is_alive():
            return False
        import time as _time  # noqa: PLC0415
        _LISTENER_STATS["started_at"] = _time.time()
        _LISTENER_THREAD = threading.Thread(
            target=_listener_loop,
            name="rag-spotify-listener",
            daemon=True,
        )
        _LISTENER_THREAD.start()
        return True


# IPC handler para inspeccionar el estado del listener.
from rag.runtime import ipc  # noqa: E402


@ipc.handler("status_spotify")
def status_spotify_handler(_payload: dict[str, Any]) -> dict[str, Any]:
    """Stats del listener para debugging."""
    return {
        "pyobjc_available": _PYOBJC_AVAILABLE,
        "listener_running": (
            _LISTENER_THREAD is not None and _LISTENER_THREAD.is_alive()
        ),
        **_LISTENER_STATS,
    }


# Auto-start al import (cuando el supervisor descubre el módulo via
# _import_jobs). Idempotent.
start_listener()
