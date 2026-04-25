"""Tests para `rag behavior backfill` + _health_training_signal (2026-04-23).

El backfill linkea opens huérfanos al rag_queries.id original por
proximidad temporal + session match. Rescata training signal que quedó
desperdigado pre-instrumentación.

Cobertura:
  • _behavior_backfill_candidates: filtra opens con original_query_id NULL
  • _behavior_backfill_find_match: 3 policies en cascada
  • _behavior_delta_seconds: ISO-8601 naive + tz-aware + inválido
  • _behavior_backfill_apply: UPDATE preserva extra fields + agrega provenance
  • behavior_backfill CLI: dry-run no escribe, actual escribe, JSON shape
  • _health_training_signal: CTR, orphans, backfilled count, window-scoped
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

import rag
from rag import cli


# ── Fixture: tmp telemetry.db aislado por test ──────────────────────────────


@pytest.fixture
def tmp_telemetry(monkeypatch, tmp_path):
    """Redirige _ragvec_state_conn a una DB temp con el schema de rag_behavior
    + rag_queries + rag_feedback, así cada test parte de una pizarra limpia.
    """
    db_path = tmp_path / "telemetry.db"

    import contextlib

    @contextlib.contextmanager
    def _conn():
        c = sqlite3.connect(str(db_path))
        c.row_factory = None
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _conn)

    # Create minimal schemas (subset de lo que rag.py expone).
    with _conn() as c:
        c.executescript(
            """
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
            CREATE TABLE rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                cmd TEXT,
                q TEXT,
                session TEXT,
                paths_json TEXT,
                scores_json TEXT,
                top_score REAL,
                extra_json TEXT
            );
            CREATE TABLE rag_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                rating INTEGER,
                q TEXT,
                paths_json TEXT,
                extra_json TEXT
            );
            """
        )
        c.commit()
    yield db_path


def _insert_query(conn_path, ts, q, session=None, paths=None):
    """Helper: insert a rag_queries row. Returns inserted id."""
    with sqlite3.connect(str(conn_path)) as c:
        cur = c.execute(
            "INSERT INTO rag_queries(ts, cmd, q, session, paths_json)"
            " VALUES(?, 'query', ?, ?, ?)",
            (ts, q, session, json.dumps(paths or [])),
        )
        c.commit()
        return cur.lastrowid


def _insert_open(conn_path, ts, path, source="cli", session=None,
                 extra=None):
    """Helper: insert a rag_behavior open event. Returns inserted id."""
    ej = dict(extra or {})
    if session is not None:
        ej["session"] = session
    with sqlite3.connect(str(conn_path)) as c:
        cur = c.execute(
            "INSERT INTO rag_behavior(ts, source, event, path, extra_json)"
            " VALUES(?, ?, 'open', ?, ?)",
            (ts, source, path, json.dumps(ej) if ej else None),
        )
        c.commit()
        return cur.lastrowid


def _now(offset_minutes=0):
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat()


# ── _behavior_backfill_candidates ───────────────────────────────────────────


def test_candidates_filters_out_already_linked(tmp_telemetry):
    """Opens con original_query_id populado NO aparecen como huérfanos."""
    _insert_open(tmp_telemetry, _now(-5), "a.md",
                 extra={"original_query_id": 42})
    _insert_open(tmp_telemetry, _now(-3), "b.md")  # orphan
    out = rag._behavior_backfill_candidates(limit=50)
    assert len(out) == 1
    assert out[0]["path"] == "b.md"


def test_candidates_order_newest_first(tmp_telemetry):
    _insert_open(tmp_telemetry, _now(-30), "old.md")
    _insert_open(tmp_telemetry, _now(-2), "new.md")
    _insert_open(tmp_telemetry, _now(-15), "mid.md")
    out = rag._behavior_backfill_candidates(limit=50)
    assert [o["path"] for o in out] == ["new.md", "mid.md", "old.md"]


def test_candidates_extracts_session_from_extra(tmp_telemetry):
    _insert_open(tmp_telemetry, _now(-5), "a.md",
                 extra={"session": "wa:abc"})
    _insert_open(tmp_telemetry, _now(-4), "b.md",
                 extra={"session_id": "web:xyz"})
    out = rag._behavior_backfill_candidates(limit=10)
    sessions = {o["path"]: o["session"] for o in out}
    assert sessions["a.md"] == "wa:abc"
    assert sessions["b.md"] == "web:xyz"


# ── _behavior_backfill_find_match ─────────────────────────────────────────


def test_match_prefers_same_session(tmp_telemetry):
    """Policy 1: same session within window → strongest signal."""
    orphan_ts = _now(0)
    # Query con same session (mismo minuto) y path unrelated
    q_same_sess = _insert_query(
        tmp_telemetry, _now(-1), "otra cosa", session="sess1",
        paths=["unrelated.md"],
    )
    # Query en otra sesión CON el path específico (más fresh). Policy 1
    # debería ganar aunque policy 2 también matche, porque same_session
    # se chequea primero.
    _insert_query(
        tmp_telemetry, _now(-0.5), "búsqueda exacta",
        session="sess2", paths=["target.md"],
    )
    match = rag._behavior_backfill_find_match(
        orphan_ts=orphan_ts, orphan_path="target.md",
        orphan_session="sess1", window_minutes=10,
    )
    assert match is not None
    assert match["match_policy"] == "same_session"
    assert match["query_id"] == q_same_sess


def test_match_falls_back_to_path_match(tmp_telemetry):
    """Policy 2: no same session, but path appears in a query's paths_json."""
    orphan_ts = _now(0)
    q_path = _insert_query(
        tmp_telemetry, _now(-3), "query with path",
        session=None, paths=["target.md"],
    )
    # Another nearby query WITHOUT target.md in paths — should lose.
    _insert_query(
        tmp_telemetry, _now(-1), "random query",
        session=None, paths=["other.md"],
    )
    match = rag._behavior_backfill_find_match(
        orphan_ts=orphan_ts, orphan_path="target.md",
        orphan_session=None, window_minutes=10,
    )
    assert match is not None
    assert match["match_policy"] == "path_match"
    assert match["query_id"] == q_path


def test_match_time_nearest_last_resort(tmp_telemetry):
    """Policy 3: no session, no path match → closest by time."""
    orphan_ts = _now(0)
    q_close = _insert_query(
        tmp_telemetry, _now(-2), "something",
        session=None, paths=["random.md"],
    )
    _insert_query(
        tmp_telemetry, _now(-8), "older",
        session=None, paths=["random.md"],
    )
    match = rag._behavior_backfill_find_match(
        orphan_ts=orphan_ts, orphan_path="target.md",  # no path_match
        orphan_session=None, window_minutes=10,
    )
    assert match is not None
    assert match["match_policy"] == "time_nearest"
    assert match["query_id"] == q_close


def test_match_none_when_outside_window(tmp_telemetry):
    """No query dentro del window → None."""
    orphan_ts = _now(0)
    _insert_query(
        tmp_telemetry, _now(-30), "way too old",
        session=None, paths=["x.md"],
    )
    match = rag._behavior_backfill_find_match(
        orphan_ts=orphan_ts, orphan_path="target.md",
        orphan_session=None, window_minutes=10,
    )
    assert match is None


# ── _behavior_delta_seconds ─────────────────────────────────────────────────


def test_delta_seconds_tz_aware():
    a = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    b = datetime(2025, 1, 1, 12, 0, 30, tzinfo=timezone.utc).isoformat()
    assert rag._behavior_delta_seconds(a, b) == 30
    assert rag._behavior_delta_seconds(b, a) == 30


def test_delta_seconds_naive():
    a = "2025-01-01T12:00:00"
    b = "2025-01-01T12:05:00"
    assert rag._behavior_delta_seconds(a, b) == 300


def test_delta_seconds_invalid_returns_minus_one():
    assert rag._behavior_delta_seconds("garbage", "2025-01-01") == -1
    assert rag._behavior_delta_seconds("", "") == -1


def test_delta_seconds_mixed_tz_tolerance():
    """Mixed naive+aware: el helper normaliza a naive y devuelve un int."""
    a = "2025-01-01T12:00:00"
    b = "2025-01-01T12:00:15+00:00"
    d = rag._behavior_delta_seconds(a, b)
    assert d >= 0  # No crash + un valor finito


# ── _behavior_backfill_apply ────────────────────────────────────────────────


def test_apply_updates_extra_json(tmp_telemetry):
    rid = _insert_open(tmp_telemetry, _now(-5), "a.md",
                       extra={"rank": 2, "dwell_s": 3.5})
    with sqlite3.connect(str(tmp_telemetry)) as c:
        extra_raw = c.execute(
            "SELECT extra_json FROM rag_behavior WHERE id = ?", (rid,)
        ).fetchone()[0]
    ok = rag._behavior_backfill_apply(
        row_id=rid, extra_raw=extra_raw, query_id=42,
        match_policy="same_session", delta_s=25,
    )
    assert ok is True
    with sqlite3.connect(str(tmp_telemetry)) as c:
        updated_raw = c.execute(
            "SELECT extra_json FROM rag_behavior WHERE id = ?", (rid,)
        ).fetchone()[0]
    updated = json.loads(updated_raw)
    # Preserva fields existentes
    assert updated["rank"] == 2
    assert updated["dwell_s"] == 3.5
    # Agrega provenance nuevos
    assert updated["original_query_id"] == 42
    assert updated["backfilled"] is True
    assert updated["backfill_match_policy"] == "same_session"
    assert updated["backfill_delta_s"] == 25


def test_apply_handles_null_extra_json(tmp_telemetry):
    """Orphan con extra_json=NULL no crashea — arranca dict vacío."""
    with sqlite3.connect(str(tmp_telemetry)) as c:
        cur = c.execute(
            "INSERT INTO rag_behavior(ts, source, event, path)"
            " VALUES(?, 'cli', 'open', 'a.md')",
            (_now(-1),),
        )
        c.commit()
        rid = cur.lastrowid
    ok = rag._behavior_backfill_apply(
        row_id=rid, extra_raw=None, query_id=7,
        match_policy="time_nearest", delta_s=60,
    )
    assert ok is True
    with sqlite3.connect(str(tmp_telemetry)) as c:
        raw = c.execute(
            "SELECT extra_json FROM rag_behavior WHERE id = ?", (rid,)
        ).fetchone()[0]
    data = json.loads(raw)
    assert data["original_query_id"] == 7
    assert data["backfilled"] is True


# ── behavior_backfill CLI ──────────────────────────────────────────────────


def test_cli_dry_run_does_not_write(tmp_telemetry):
    _insert_query(tmp_telemetry, _now(-3), "q", paths=["target.md"])
    _insert_open(tmp_telemetry, _now(-2), "target.md")

    r = CliRunner()
    result = r.invoke(cli, ["behavior", "backfill", "--dry-run"])
    assert result.exit_code == 0
    assert "Dry-run" in result.output
    # Ninguna UPDATE se aplicó — el orphan sigue sin original_query_id.
    orphans = rag._behavior_backfill_candidates(limit=10)
    assert len(orphans) == 1


def test_cli_applies_updates(tmp_telemetry):
    qid = _insert_query(
        tmp_telemetry, _now(-2), "q test", paths=["target.md"],
    )
    oid = _insert_open(tmp_telemetry, _now(-1), "target.md")

    r = CliRunner()
    result = r.invoke(cli, ["behavior", "backfill"])
    assert result.exit_code == 0
    assert "1/1 opens backfilleados" in result.output
    # Orphan desapareció — tiene original_query_id ahora.
    orphans = rag._behavior_backfill_candidates(limit=10)
    assert len(orphans) == 0
    # La fila tiene el link correcto.
    with sqlite3.connect(str(tmp_telemetry)) as c:
        raw = c.execute(
            "SELECT extra_json FROM rag_behavior WHERE id = ?", (oid,)
        ).fetchone()[0]
    assert json.loads(raw)["original_query_id"] == qid


def test_cli_empty_case_reports_consistent(tmp_telemetry):
    """Con 0 opens huérfanos el comando devuelve mensaje verde."""
    r = CliRunner()
    result = r.invoke(cli, ["behavior", "backfill"])
    assert result.exit_code == 0
    assert "No hay opens huérfanos" in result.output


def test_cli_json_output_shape(tmp_telemetry):
    _insert_query(tmp_telemetry, _now(-2), "q", paths=["target.md"])
    _insert_open(tmp_telemetry, _now(-1), "target.md")

    r = CliRunner()
    result = r.invoke(
        cli, ["behavior", "backfill", "--dry-run", "--as-json"],
    )
    assert result.exit_code == 0
    # Output tiene 1+ líneas JSON. Parseamos la primera.
    first_line = result.output.strip().split("\n")[0]
    data = json.loads(first_line)
    assert data["orphans"] == 1
    assert data["matched"] == 1
    assert data["dry_run"] is True


# ── _health_training_signal ────────────────────────────────────────────────


def test_training_signal_empty_db(tmp_telemetry):
    stat = rag._health_training_signal(since_days=7)
    assert stat["window_days"] == 7
    assert stat["impressions"] == 0
    assert stat["opens"] == 0
    assert stat["ctr_pct"] == 0.0
    assert stat["orphan_opens"] == 0


def test_training_signal_ctr_calculation(tmp_telemetry):
    """10 impressions + 2 opens → CTR = 20%."""
    with sqlite3.connect(str(tmp_telemetry)) as c:
        for i in range(10):
            c.execute(
                "INSERT INTO rag_behavior(ts, source, event, path)"
                " VALUES(?, 'cli', 'impression', ?)",
                (_now(-i), f"impression_{i}.md"),
            )
        for i in range(2):
            c.execute(
                "INSERT INTO rag_behavior(ts, source, event, path)"
                " VALUES(?, 'cli', 'open', ?)",
                (_now(-i), f"open_{i}.md"),
            )
        c.commit()
    stat = rag._health_training_signal(since_days=7)
    assert stat["impressions"] == 10
    assert stat["opens"] == 2
    assert stat["ctr_pct"] == 20.0


def test_training_signal_counts_orphans_all_time(tmp_telemetry):
    """Orphans count incluye opens viejos fuera del window — porque
    el backfill puede recogerlos igual."""
    # Open viejo (45 días) sin original_query_id
    with sqlite3.connect(str(tmp_telemetry)) as c:
        c.execute(
            "INSERT INTO rag_behavior(ts, source, event, path)"
            " VALUES(?, 'cli', 'open', 'old.md')",
            ((datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),),
        )
        # Open reciente con link
        c.execute(
            "INSERT INTO rag_behavior(ts, source, event, path, extra_json)"
            " VALUES(?, 'cli', 'open', 'new.md', ?)",
            (_now(-1), json.dumps({"original_query_id": 99})),
        )
        c.commit()
    stat = rag._health_training_signal(since_days=7)
    # Viejo quedó fuera del window (opens=0) pero cuenta como orphan global.
    assert stat["opens"] == 1  # solo el nuevo entra al window
    assert stat["orphan_opens"] == 1  # solo el viejo es huérfano


def test_training_signal_backfilled_count(tmp_telemetry):
    """Opens con backfilled=true se cuentan en backfilled_opens."""
    with sqlite3.connect(str(tmp_telemetry)) as c:
        for i in range(3):
            c.execute(
                "INSERT INTO rag_behavior(ts, source, event, path, extra_json)"
                " VALUES(?, 'cli', 'open', ?, ?)",
                (_now(-i), f"bf_{i}.md",
                 json.dumps({"original_query_id": i, "backfilled": True})),
            )
        c.commit()
    stat = rag._health_training_signal(since_days=7)
    assert stat["backfilled_opens"] == 3


# ── Contract: health payload incluye training_signal ──────────────────────


def test_health_json_payload_has_training_signal(tmp_telemetry):
    r = CliRunner()
    result = r.invoke(cli, ["health", "--as-json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip().split("\n")[-1])
    assert "training_signal" in payload
    ts = payload["training_signal"]
    assert "impressions" in ts
    assert "opens" in ts
    assert "ctr_pct" in ts
    assert "orphan_opens" in ts
    assert "feedback_gate_target" in ts
