"""F4.5 tests — Spotify event-driven listener.

Tests no requieren Spotify abierto ni notifications reales — solo
verifican que:
- El módulo importa sin error.
- ``start_listener()`` es idempotente.
- El IPC handler ``status_spotify`` retorna stats coherentes.
- ``RAG_SPOTIFY_LISTENER_DISABLED=1`` opt-out funciona.

Tests de notification dispatch real son E2E manual (no automated en CI).
"""
from __future__ import annotations

import sys

import pytest

from rag.runtime import ipc


@pytest.fixture(autouse=True)
def _reset():
    """No hacemos sys.modules.pop del listener — PyObjC class registry
    no permite re-definición de clases Objective-C. Reseteamos solo el
    handler IPC y los stats. El módulo queda cargado entre tests."""
    ipc._reset_handlers()
    if "rag.runtime.jobs._spotify_listener" in sys.modules:
        sl = sys.modules["rag.runtime.jobs._spotify_listener"]
        # Re-register IPC handler post-reset.
        ipc.register_handler("status_spotify", sl.status_spotify_handler)
    yield
    ipc._reset_handlers()


def test_module_imports_without_error():
    import rag.runtime.jobs._spotify_listener as sl
    assert hasattr(sl, "start_listener")
    assert hasattr(sl, "_LISTENER_STATS")


def test_status_spotify_handler_registered():
    import rag.runtime.jobs._spotify_listener  # noqa: F401
    assert "status_spotify" in ipc._registered_handlers()


def test_status_returns_stats():
    import rag.runtime.jobs._spotify_listener as sl
    result = sl.status_spotify_handler({})
    assert "pyobjc_available" in result
    assert "listener_running" in result
    assert "notifications_received" in result
    assert "tracks_recorded" in result


def test_start_listener_idempotent():
    """Llamar start_listener() mientras el thread está vivo NO crea
    duplicados. Si el thread previo terminó, sí se crea uno nuevo —
    eso es comportamiento esperado (recovery)."""
    import rag.runtime.jobs._spotify_listener as sl

    initial_thread = sl._LISTENER_THREAD

    if initial_thread is not None and initial_thread.is_alive():
        # Thread vivo — start_listener debe ser no-op.
        sl.start_listener()
        assert sl._LISTENER_THREAD is initial_thread
    else:
        # Thread no vivo — start_listener arranca uno nuevo.
        if sl._PYOBJC_AVAILABLE:
            started = sl.start_listener()
            assert started is True
            assert sl._LISTENER_THREAD is not initial_thread
            assert sl._LISTENER_THREAD.is_alive()


def test_disabled_via_env_var(monkeypatch):
    """``RAG_SPOTIFY_LISTENER_DISABLED=1`` previene el start."""
    monkeypatch.setenv("RAG_SPOTIFY_LISTENER_DISABLED", "1")
    sys.modules.pop("rag.runtime.jobs._spotify_listener", None)

    import rag.runtime.jobs._spotify_listener as sl
    # Reset thread state — el auto-start del import puede haber corrido
    # antes del monkeypatch en la primera ejecución.
    sl._LISTENER_THREAD = None

    started = sl.start_listener()
    assert started is False
    assert sl._LISTENER_THREAD is None


def test_handle_notification_skips_paused(monkeypatch):
    """Notifications con state=paused NO graban — ahorro DB."""
    import rag.runtime.jobs._spotify_listener as sl

    # Mock notification con state=paused.
    class _UI(dict):
        def __iter__(self):
            return iter(self.keys())

    class _Notif:
        def userInfo(self):
            return _UI({"Player State": "Paused", "Name": "Some Track"})

    n_inserts = {"count": 0}

    def fake_record():
        n_inserts["count"] += 1

    monkeypatch.setattr(
        "rag.integrations.spotify_local.record_now_playing",
        fake_record,
    )

    sl._handle_notification(_Notif())
    assert n_inserts["count"] == 0
    # notification se contó.
    assert sl._LISTENER_STATS["notifications_received"] >= 1


def test_handle_notification_records_playing(monkeypatch):
    import rag.runtime.jobs._spotify_listener as sl

    # Reset stats.
    sl._LISTENER_STATS["tracks_recorded"] = 0

    class _UI(dict):
        def __iter__(self):
            return iter(self.keys())

    class _Notif:
        def userInfo(self):
            return _UI({"Player State": "Playing", "Name": "Track A", "Artist": "X"})

    def fake_record():
        return None  # success

    monkeypatch.setattr(
        "rag.integrations.spotify_local.record_now_playing",
        fake_record,
    )

    sl._handle_notification(_Notif())
    assert sl._LISTENER_STATS["tracks_recorded"] == 1
