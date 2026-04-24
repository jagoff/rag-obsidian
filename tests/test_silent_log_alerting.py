"""Tests para el alerting unificado de silent errors (audit 2026-04-24).

### El bug que estos tests cierran

Pre-fix, `_SILENT_LOG_COUNTER` solo lo incrementaba `rag._silent_log()`.
`rag._log_sql_state_error()` escribía directo a `sql_state_errors.jsonl`
**sin tocar el counter**, así que los alerts a stderr (threshold 20/hora)
no disparaban nunca cuando los fails eran de origen SQL.

Audit midió 1756 errores SQL en 6 días post-semantic-cache wiring
(2026-04-22) — ninguno generó un alert, nadie se enteró de la
degradación hasta que busqué a mano.

### Fix

Extraer el bump+alert a `_bump_silent_log_counter()`; invocar desde
AMBAS funciones. Counter unificado: un spike en cualquier sink dispara
la misma WARNING a stderr.
"""
from __future__ import annotations

import io
import sys
import threading

import rag


def _reset_counter(monkeypatch):
    """Reset del counter + threshold a valores conocidos por test."""
    monkeypatch.setattr(rag, "_SILENT_LOG_COUNTER", {"count": 0, "last_alert_ts": 0.0})
    monkeypatch.setattr(rag, "_SILENT_LOG_ALERT_THRESHOLD", 5)
    monkeypatch.setattr(rag, "_SILENT_LOG_ALERT_INTERVAL", 1000.0)


def test_silent_log_bumps_counter(monkeypatch):
    _reset_counter(monkeypatch)
    rag._silent_log("test_where", ValueError("x"))
    assert rag._SILENT_LOG_COUNTER["count"] == 1


def test_sql_state_error_now_bumps_counter(monkeypatch, tmp_path):
    """Regression guard del bug: pre-fix este era el bug
    — _log_sql_state_error NO actualizaba el counter."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")

    rag._log_sql_state_error("test_event", err="OperationalError('locked')")
    assert rag._SILENT_LOG_COUNTER["count"] == 1, (
        "_log_sql_state_error DEBE bumpear el counter compartido post-fix "
        "audit 2026-04-24 — si este test falla, el alerting quedó parcial"
    )


def test_counter_unified_across_both_sinks(monkeypatch, tmp_path):
    """3 silent + 2 sql errors bumpean el mismo counter. Threshold alto
    para que no se dispare el auto-reset durante el test."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SILENT_LOG_ALERT_THRESHOLD", 100)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")

    for _ in range(3):
        rag._silent_log("sink_a", KeyError("k"))
    for _ in range(2):
        rag._log_sql_state_error("sink_b", err="locked")

    assert rag._SILENT_LOG_COUNTER["count"] == 5


def test_threshold_fires_alert_to_stderr(monkeypatch, tmp_path):
    """5 silent errors con threshold=5 → WARNING a stderr."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)

    for _ in range(5):
        rag._silent_log("test", ValueError())

    out = captured.getvalue()
    assert "[rag/telemetry] WARNING" in out
    assert "5 silent errors" in out
    assert "silent_errors.jsonl" in out
    assert "sql_state_errors.jsonl" in out, (
        "el alert debe mencionar AMBOS sinks post audit 2026-04-24"
    )


def test_threshold_fires_from_sql_errors_too(monkeypatch, tmp_path):
    """Mismo threshold disparado solo desde `_log_sql_state_error` — el
    caso que pre-fix quedaba silencioso."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)

    for _ in range(5):
        rag._log_sql_state_error("queries_sql_write_failed", err="locked")

    out = captured.getvalue()
    assert "[rag/telemetry] WARNING" in out, (
        "SQL errors deben disparar el alert. Pre-audit este test fallaba."
    )


def test_interval_cooldown_suppresses_duplicate_alerts(monkeypatch, tmp_path):
    """Segunda tanda bajo el threshold DESPUÉS del alert no dispara
    otro — sólo un alert por interval window."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)

    # Primera tanda: 5 errors → 1 alert.
    for _ in range(5):
        rag._silent_log("tanda1", ValueError())
    first_alerts = captured.getvalue().count("WARNING")
    assert first_alerts == 1

    # Segunda tanda: 5 errors más → NO alert adicional (interval no pasó).
    for _ in range(5):
        rag._silent_log("tanda2", ValueError())
    second_alerts = captured.getvalue().count("WARNING")
    assert second_alerts == 1, (
        f"expected 1 alert (interval cooldown), got {second_alerts}"
    )


def test_counter_resets_after_alert(monkeypatch, tmp_path):
    """Post-alert, el counter vuelve a 0 — así el próximo ciclo requiere
    un spike fresh, no dispara cada error incremental."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")

    for _ in range(5):
        rag._silent_log("pre", ValueError())

    # Post-alert: counter a 0 aunque el último_alert_ts quedó set.
    assert rag._SILENT_LOG_COUNTER["count"] == 0
    assert rag._SILENT_LOG_COUNTER["last_alert_ts"] > 0


def test_bump_is_thread_safe(monkeypatch, tmp_path):
    """50 threads haciendo bump concurrente → counter = exactly 50 post-
    reset (o 0 si disparó el alert durante la race, pero el threshold
    es 10_000 en este test para evitarlo)."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SILENT_LOG_ALERT_THRESHOLD", 10_000)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")

    ready = threading.Event()

    def worker():
        ready.wait()
        rag._bump_silent_log_counter()

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join()

    assert rag._SILENT_LOG_COUNTER["count"] == 50


def test_bump_never_raises_even_if_stderr_broken(monkeypatch, tmp_path):
    """Si sys.stderr raisea al write, el helper debe tragarse la excepción
    — silent-fail contract de los callers."""
    _reset_counter(monkeypatch)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql.jsonl")

    class _ExplodingStderr:
        def write(self, s): raise IOError("stderr broken")
        def flush(self): raise IOError("stderr broken")

    monkeypatch.setattr(sys, "stderr", _ExplodingStderr())

    # Tirar 5 al threshold — el write al stderr falla.
    # No debe raisear.
    for _ in range(5):
        rag._silent_log("test", ValueError())

    # Contract cumplido: sin excepción + counter se reseteó igual.
    assert rag._SILENT_LOG_COUNTER["count"] == 0
