"""Tests de `rag open --nth N` (Fase 2 de Opción B — 2026-04-23).

El flag `--nth N` resuelve el N-ésimo source del último `rag_queries`
row y abre esa nota, auto-rellenando `original_query_id` / `query` /
`rank` en el behavior event. Alternativa sin-setup a registrar el URL
handler de `x-rag-open://`.

Cobertura:
  • _resolve_nth_source_from_last_query: empty DB, session scope, out-of-range,
    paths_json corrupto, happy path.
  • CLI --nth: sin rag_queries previa, out of range, incompatible con path,
    ambos None, evento se emite con original_query_id + rank, --session
    scopes correctamente, Path escape rejected.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import subprocess

import pytest
from click.testing import CliRunner

import rag
from rag import cli


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """DB temp con schemas de rag_queries + rag_behavior."""
    db_path = tmp_path / "telemetry.db"

    @contextlib.contextmanager
    def _conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _conn)

    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                cmd TEXT,
                q TEXT,
                session TEXT,
                paths_json TEXT,
                scores_json TEXT,
                extra_json TEXT
            );
            CREATE TABLE rag_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                event TEXT NOT NULL,
                path TEXT,
                query TEXT,
                rank INTEGER,
                dwell_s REAL,
                extra_json TEXT
            );
            """
        )
        c.commit()
    yield db_path


@pytest.fixture
def tmp_vault(monkeypatch, tmp_path):
    """Vault temp con algunas notas — el comando valida existencia antes de abrir."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# a")
    (vault / "b.md").write_text("# b")
    (vault / "c.md").write_text("# c")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    yield vault


def _insert_query(db_path, q, paths, session=None, ts="2025-01-01T12:00:00"):
    with sqlite3.connect(str(db_path)) as c:
        cur = c.execute(
            "INSERT INTO rag_queries(ts, cmd, q, session, paths_json)"
            " VALUES(?, 'query', ?, ?, ?)",
            (ts, q, session, json.dumps(paths)),
        )
        c.commit()
        return cur.lastrowid


# ── _resolve_nth_source_from_last_query ────────────────────────────────────


def test_resolve_empty_db_returns_none(tmp_db):
    assert rag._resolve_nth_source_from_last_query(nth=1, session=None) is None


def test_resolve_scopes_to_session(tmp_db):
    """Con session=X debe pickear la última query de X, no la global."""
    _insert_query(tmp_db, "global q", ["global.md"],
                  session=None, ts="2025-01-01T14:00:00")
    qid_x = _insert_query(tmp_db, "session x", ["x.md"],
                          session="sess_x", ts="2025-01-01T12:00:00")
    # Otro session más nuevo pero distinto — no debería ganar.
    _insert_query(tmp_db, "session y", ["y.md"],
                  session="sess_y", ts="2025-01-01T15:00:00")
    out = rag._resolve_nth_source_from_last_query(nth=1, session="sess_x")
    assert out is not None
    assert out["original_query_id"] == qid_x
    assert out["path"] == "x.md"
    assert out["query"] == "session x"


def test_resolve_picks_latest_when_no_session(tmp_db):
    """Sin session scope, devuelve la más reciente global."""
    _insert_query(tmp_db, "viejo", ["old.md"], ts="2025-01-01T10:00:00")
    qid_new = _insert_query(tmp_db, "nuevo", ["new.md"],
                            ts="2025-01-01T20:00:00")
    out = rag._resolve_nth_source_from_last_query(nth=1, session=None)
    assert out["original_query_id"] == qid_new
    assert out["path"] == "new.md"


def test_resolve_out_of_range_marks_flag(tmp_db):
    """nth > total_sources → devuelve dict con out_of_range=True."""
    _insert_query(tmp_db, "q", ["a.md", "b.md"])
    out = rag._resolve_nth_source_from_last_query(nth=5, session=None)
    assert out is not None
    assert out["out_of_range"] is True
    assert out["total_sources"] == 2
    assert out["path"] is None


def test_resolve_nth_zero_treated_as_out_of_range(tmp_db):
    """nth=0 no es válido (1-indexed)."""
    _insert_query(tmp_db, "q", ["a.md"])
    out = rag._resolve_nth_source_from_last_query(nth=0, session=None)
    assert out is not None
    assert out["out_of_range"] is True


def test_resolve_negative_nth_treated_as_out_of_range(tmp_db):
    _insert_query(tmp_db, "q", ["a.md"])
    out = rag._resolve_nth_source_from_last_query(nth=-1, session=None)
    assert out is not None
    assert out["out_of_range"] is True


def test_resolve_empty_paths_json_returns_none(tmp_db):
    """paths_json=[] significa query sin sources — no resolvible."""
    _insert_query(tmp_db, "q", [])
    assert rag._resolve_nth_source_from_last_query(nth=1, session=None) is None


def test_resolve_corrupt_paths_json_returns_none(tmp_db):
    """paths_json malformed → None, no crash."""
    with sqlite3.connect(str(tmp_db)) as c:
        c.execute(
            "INSERT INTO rag_queries(ts, cmd, q, paths_json)"
            " VALUES('2025-01-01T12:00:00', 'query', 'q', 'not-json')"
        )
        c.commit()
    assert rag._resolve_nth_source_from_last_query(nth=1, session=None) is None


def test_resolve_missing_session_returns_none(tmp_db):
    """Session con 0 queries → None (aunque haya queries en otras sessions)."""
    _insert_query(tmp_db, "q", ["a.md"], session="other")
    assert rag._resolve_nth_source_from_last_query(
        nth=1, session="missing",
    ) is None


# ── CLI: `rag open --nth N` ─────────────────────────────────────────────────


def test_cli_nth_without_prior_query(tmp_db, tmp_vault):
    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "1"])
    assert result.exit_code == 1
    assert "no hay rag_queries row previa" in result.output


def test_cli_nth_out_of_range(tmp_db, tmp_vault):
    _insert_query(tmp_db, "q", ["a.md", "b.md"])
    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "5"])
    assert result.exit_code == 1
    assert "fuera de rango" in result.output
    assert "2 source(s)" in result.output


def test_cli_nth_and_path_mutually_exclusive(tmp_db, tmp_vault):
    r = CliRunner()
    result = r.invoke(cli, ["open", "a.md", "--nth", "1"])
    assert result.exit_code == 2
    assert "mutuamente excluyentes" in result.output


def test_cli_no_path_no_nth(tmp_db, tmp_vault):
    r = CliRunner()
    result = r.invoke(cli, ["open"])
    assert result.exit_code == 2
    assert "pasá un PATH o --nth N" in result.output


def test_cli_nth_emits_behavior_event_with_original_query_id(
    tmp_db, tmp_vault, monkeypatch,
):
    """Happy path: --nth N resuelve → log_behavior_event emitido con
    original_query_id + rank + query populados."""
    qid = _insert_query(tmp_db, "qué es ikigai", ["a.md", "b.md"],
                        session="sess1")
    # Stub subprocess.run para no tocar el shell real.
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    events: list[dict] = []
    monkeypatch.setattr(rag, "log_behavior_event", events.append)

    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "2"])
    assert result.exit_code == 0, result.output
    assert "✓ opened #2: b.md" in result.output
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "open"
    assert ev["path"] == "b.md"
    assert ev["rank"] == 2
    assert ev["query"] == "qué es ikigai"
    assert ev["original_query_id"] == qid
    assert ev["session"] == "sess1"


def test_cli_nth_respects_session_scope(tmp_db, tmp_vault, monkeypatch):
    """`--session X --nth 1` toma la última query de X, no la global."""
    _insert_query(tmp_db, "global query", ["global.md"],
                  session=None, ts="2025-01-01T20:00:00")
    qid_x = _insert_query(tmp_db, "session query", ["a.md"],
                          session="sess_x", ts="2025-01-01T10:00:00")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    events: list[dict] = []
    monkeypatch.setattr(rag, "log_behavior_event", events.append)

    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "1", "--session", "sess_x"])
    assert result.exit_code == 0
    assert len(events) == 1
    assert events[0]["original_query_id"] == qid_x
    assert events[0]["path"] == "a.md"


def test_cli_explicit_path_preserved(tmp_db, tmp_vault, monkeypatch):
    """Modo path explícito: NO inyecta original_query_id (no hay nth)."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    events: list[dict] = []
    monkeypatch.setattr(rag, "log_behavior_event", events.append)

    r = CliRunner()
    result = r.invoke(cli, ["open", "a.md"])
    assert result.exit_code == 0
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "open"
    assert ev["path"] == "a.md"
    assert "original_query_id" not in ev
    # NOT emite el "✓ opened #N" line (eso es solo para --nth).
    assert "opened #" not in result.output


def test_cli_nth_missing_file_reports_error(tmp_db, tmp_vault, monkeypatch):
    """Si el path resuelto existió en rag_queries pero fue eliminado
    del vault, error claro (no crash silent)."""
    _insert_query(tmp_db, "q", ["ghost.md"])  # archivo no existe
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)

    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "1"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_nth_inherits_query_when_not_given(tmp_db, tmp_vault, monkeypatch):
    """`--nth` sin `--query` hereda `query` del resolved rag_queries row."""
    _insert_query(tmp_db, "búsqueda original", ["a.md"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    events: list[dict] = []
    monkeypatch.setattr(rag, "log_behavior_event", events.append)

    r = CliRunner()
    result = r.invoke(cli, ["open", "--nth", "1"])
    assert result.exit_code == 0
    assert events[0]["query"] == "búsqueda original"


def test_cli_nth_explicit_query_overrides_inherited(tmp_db, tmp_vault, monkeypatch):
    """Si el user pasa `--query Y` junto con `--nth`, Y gana sobre el heredado."""
    _insert_query(tmp_db, "original", ["a.md"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    events: list[dict] = []
    monkeypatch.setattr(rag, "log_behavior_event", events.append)

    r = CliRunner()
    result = r.invoke(
        cli, ["open", "--nth", "1", "--query", "override"],
    )
    assert result.exit_code == 0
    assert events[0]["query"] == "override"
