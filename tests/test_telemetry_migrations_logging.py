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


# ── Per-column / per-index error logging (2026-04-25 bug fix) ─────────────


def test_audio_transcripts_alter_logs_non_duplicate_errors(monkeypatch):
    """Pre-fix las migrations idempotentes tenían un `pass` que tragaba
    cualquier OperationalError NO-duplicate-column (table missing, disk
    full, syntax error). Ahora se logueen via _silent_log."""
    captured: list[tuple[str, Exception | None]] = []
    monkeypatch.setattr(rag, "_silent_log", lambda w, e: captured.append((w, e)))

    # Conn mock: cada ALTER raisea un error que NO es "duplicate column".
    class _BoomConn:
        def execute(self, sql, *args, **kwargs):
            import sqlite3 as _s
            raise _s.OperationalError("table rag_audio_transcripts has no column ts (simulated)")

    rag._migrate_audio_transcripts_phase2(_BoomConn())

    # 7 columnas + 2 índices → 9 errores logueados, todos con el prefix correcto.
    alter_logs = [c for c in captured if c[0] == "migration_audio_transcripts_alter_failed"]
    index_logs = [c for c in captured if c[0] == "migration_audio_transcripts_index_failed"]
    assert len(alter_logs) == 7, (
        f"esperaba 7 alter logs (uno por col), got {len(alter_logs)}: {captured!r}"
    )
    assert len(index_logs) == 2, (
        f"esperaba 2 index logs, got {len(index_logs)}: {captured!r}"
    )


def test_audio_transcripts_alter_silent_on_duplicate_column(monkeypatch):
    """Cuando el error ES "duplicate column" (caso happy idempotent),
    NO se debe loggear nada — esa rama es no-op silencioso."""
    captured: list[tuple[str, Exception | None]] = []
    monkeypatch.setattr(rag, "_silent_log", lambda w, e: captured.append((w, e)))

    class _DupConn:
        def execute(self, sql, *args, **kwargs):
            import sqlite3 as _s
            if "ALTER" in sql:
                raise _s.OperationalError("duplicate column name: audio_hash")
            # CREATE INDEX IF NOT EXISTS — succeeds silently.
            return None

    rag._migrate_audio_transcripts_phase2(_DupConn())

    # CERO alter logs (todas las ALTERs eran no-op por duplicate).
    alter_logs = [c for c in captured if c[0] == "migration_audio_transcripts_alter_failed"]
    assert alter_logs == [], (
        f"happy path duplicate column NO debe loggear: {captured!r}"
    )


def test_cita_detections_alter_logs_non_duplicate_errors(monkeypatch):
    """Mismo bug pre-fix en _migrate_cita_detections_add_kind."""
    captured: list[tuple[str, Exception | None]] = []
    monkeypatch.setattr(rag, "_silent_log", lambda w, e: captured.append((w, e)))

    class _BoomConn:
        def execute(self, sql, *args, **kwargs):
            import sqlite3 as _s
            raise _s.OperationalError("near 'ALTRE': syntax error (simulated)")

    rag._migrate_cita_detections_add_kind(_BoomConn())

    # 2 columnas + 1 índice → 3 errores logueados.
    alter_logs = [c for c in captured if c[0] == "migration_cita_detections_alter_failed"]
    index_logs = [c for c in captured if c[0] == "migration_cita_detections_index_failed"]
    assert len(alter_logs) == 2, f"got {len(alter_logs)}: {captured!r}"
    assert len(index_logs) == 1, f"got {len(index_logs)}: {captured!r}"


def test_cita_detections_alter_silent_on_duplicate_column(monkeypatch):
    """Duplicate column = no-op silencioso (idempotencia)."""
    captured: list[tuple[str, Exception | None]] = []
    monkeypatch.setattr(rag, "_silent_log", lambda w, e: captured.append((w, e)))

    class _DupConn:
        def execute(self, sql, *args, **kwargs):
            import sqlite3 as _s
            if "ALTER" in sql:
                raise _s.OperationalError("duplicate column name: kind")
            return None

    rag._migrate_cita_detections_add_kind(_DupConn())

    alter_logs = [c for c in captured if c[0] == "migration_cita_detections_alter_failed"]
    assert alter_logs == []
