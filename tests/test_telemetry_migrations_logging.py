"""Tests para que las migrations lazy de _ensure_telemetry_tables loggeen
errores explícitamente en lugar de tragarlos silenciosamente.

Audit del 2026-04-25 detectó que los `try/except: pass` puros alrededor
de `_migrate_cita_detections_add_kind` y `_migrate_audio_transcripts_phase2`
escondían migrations rotas — un siguiente boot las reintentaba y volvía
a fallar idéntico, ad infinitum, sin que nadie se enterara hasta que un
INSERT empezara a quejarse de columnas faltantes. Estos tests garantizan
que el fix:

  - Sí llama `_silent_log` con un evento descriptivo cuando la migration
    raisea.
  - NO re-raises (la rama de fallback del writer todavía funciona con el
    subset de columnas presentes).
  - Sigue corriendo la SEGUNDA migration aunque la primera haya fallado
    (la prevención del effect "una migration rota bloquea las siguientes").
"""
from __future__ import annotations

import sqlite3

import pytest

import rag


@pytest.fixture
def empty_db(tmp_path):
    """In-memory-ish DB conectado a un archivo tmp; safe para mutar."""
    p = tmp_path / "telemetry-test.db"
    conn = sqlite3.connect(str(p))
    yield conn
    conn.close()


def test_migrations_log_when_cita_migration_fails(monkeypatch, empty_db):
    """Si `_migrate_cita_detections_add_kind` raisea, el handler debe
    loggear `migration_cita_detections_failed` via _silent_log y NO
    re-raise. La segunda migration (audio_transcripts) debe correr igual."""
    captured: list[tuple[str, Exception | None]] = []

    def fake_silent_log(where: str, exc: Exception | None) -> None:
        captured.append((where, exc))

    monkeypatch.setattr(rag, "_silent_log", fake_silent_log)

    def boom_cita(conn):
        raise RuntimeError("simulated cita migration failure")

    second_called = {"n": 0}

    def ok_audio(conn):
        second_called["n"] += 1

    monkeypatch.setattr(rag, "_migrate_cita_detections_add_kind", boom_cita)
    monkeypatch.setattr(rag, "_migrate_audio_transcripts_phase2", ok_audio)

    # Reset the per-path cache so _ensure_telemetry_tables actually runs DDL.
    rag._TELEMETRY_DDL_ENSURED_PATHS.discard(":memory:")
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    rag._ensure_telemetry_tables(empty_db)

    # 1) The cita migration's failure WAS logged.
    cita_logs = [c for c in captured if c[0] == "migration_cita_detections_failed"]
    assert len(cita_logs) == 1, f"expected 1 cita-failed log, got {captured!r}"
    assert "simulated cita migration failure" in str(cita_logs[0][1])

    # 2) The audio migration ran anyway (cita's failure didn't break the chain).
    assert second_called["n"] == 1


def test_migrations_log_when_audio_migration_fails(monkeypatch, empty_db):
    """Symmetric: if audio migration fails, log and continue."""
    captured: list[tuple[str, Exception | None]] = []

    def fake_silent_log(where: str, exc: Exception | None) -> None:
        captured.append((where, exc))

    monkeypatch.setattr(rag, "_silent_log", fake_silent_log)

    first_called = {"n": 0}

    def ok_cita(conn):
        first_called["n"] += 1

    def boom_audio(conn):
        raise RuntimeError("simulated audio migration failure")

    monkeypatch.setattr(rag, "_migrate_cita_detections_add_kind", ok_cita)
    monkeypatch.setattr(rag, "_migrate_audio_transcripts_phase2", boom_audio)

    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    rag._ensure_telemetry_tables(empty_db)

    audio_logs = [c for c in captured if c[0] == "migration_audio_transcripts_failed"]
    assert len(audio_logs) == 1, f"expected 1 audio-failed log, got {captured!r}"
    assert "simulated audio migration failure" in str(audio_logs[0][1])

    # First migration ran (no contamination from later failure).
    assert first_called["n"] == 1


def test_migrations_silent_when_both_succeed(monkeypatch, empty_db):
    """Happy path: ambas migrations corren sin raisear → CERO logs.
    Defensa contra un futuro bug donde alguien haga 'log siempre, raisee
    o no' por accidente y empapele el dashboard de noise."""
    captured: list[tuple[str, Exception | None]] = []

    def fake_silent_log(where: str, exc: Exception | None) -> None:
        if where.startswith("migration_"):
            captured.append((where, exc))

    monkeypatch.setattr(rag, "_silent_log", fake_silent_log)
    monkeypatch.setattr(rag, "_migrate_cita_detections_add_kind", lambda conn: None)
    monkeypatch.setattr(rag, "_migrate_audio_transcripts_phase2", lambda conn: None)

    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    rag._ensure_telemetry_tables(empty_db)

    assert captured == [], f"happy path leaked migration logs: {captured!r}"


def test_migration_failure_does_not_block_ensure_telemetry(monkeypatch, empty_db):
    """Aun si AMBAS migrations fallan, _ensure_telemetry_tables debe
    completar normalmente — el DDL principal (CREATE TABLE IF NOT EXISTS)
    ya corrió en la transaction interior, los ALTERs son additive y
    pueden reintentarse en el siguiente boot."""
    monkeypatch.setattr(rag, "_silent_log", lambda *a, **kw: None)
    monkeypatch.setattr(
        rag,
        "_migrate_cita_detections_add_kind",
        lambda conn: (_ for _ in ()).throw(RuntimeError("boom1")),
    )
    monkeypatch.setattr(
        rag,
        "_migrate_audio_transcripts_phase2",
        lambda conn: (_ for _ in ()).throw(RuntimeError("boom2")),
    )

    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    # No raise. Caller (SqliteVecClient init) keeps booting.
    rag._ensure_telemetry_tables(empty_db)
