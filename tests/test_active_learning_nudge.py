"""Tests del active-learning-nudge command (C.6, 2026-04-29).

Cubre:
- `_count_active_learning_candidates` cuenta queries low-conf en
  ventana sin feedback explícito.
- Skip de queries triviales (test, hola, ping, length<=4).
- `active_learning_nudge` retorna estructura correcta.
- Channel routing: auto / wa / macos / fallback chain.
- Threshold: si N < threshold no dispara nada.
- Dry-run no envía push.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import rag


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(
        """
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            q TEXT NOT NULL,
            session TEXT,
            mode TEXT,
            top_score REAL,
            paths_json TEXT,
            extra_json TEXT
        );
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT
        );
        """
    )
    yield c
    c.close()


def _add_query(
    conn: sqlite3.Connection,
    *,
    q: str,
    top_score: float,
    days_ago: int = 0,
) -> int:
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat(
        timespec="seconds"
    )
    cur = conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, top_score) "
        "VALUES (?, 'chat', ?, ?)",
        (ts, q, top_score),
    )
    return cur.lastrowid


def _add_feedback(
    conn: sqlite3.Connection,
    *,
    q: str,
    days_ago: int = 0,
) -> None:
    ts = (datetime.now() - timedelta(days=days_ago)).isoformat(
        timespec="seconds"
    )
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, scope) "
        "VALUES (?, 1, ?, 'turn')",
        (ts, q),
    )


# ─── _count_active_learning_candidates ────────────────────────────────


def test_count_includes_low_conf_recent(conn):
    _add_query(conn, q="cuanto debe Alex de la macbook", top_score=0.15)
    _add_query(conn, q="cuanto factura Moka este mes", top_score=0.10)
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 2


def test_count_excludes_high_conf(conn):
    _add_query(conn, q="cuanto debe Alex", top_score=0.50)
    _add_query(conn, q="info del proyecto X", top_score=0.85)
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 0


def test_count_excludes_old_queries(conn):
    _add_query(conn, q="query vieja sobre algo", top_score=0.10, days_ago=15)
    _add_query(conn, q="query reciente sobre algo", top_score=0.10, days_ago=1)
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 1


def test_count_excludes_queries_with_feedback(conn):
    """Si la query ya tiene feedback explícito, no es candidate."""
    _add_query(conn, q="query con feedback ya dado", top_score=0.10)
    _add_feedback(conn, q="query con feedback ya dado")
    _add_query(conn, q="query sin feedback aun", top_score=0.10)
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 1


def test_count_excludes_trivial_queries(conn):
    """Queries triviales (test, hola, ping, probando) se excluyen."""
    for trivial in ("test", "hola", "ping", "probando"):
        _add_query(conn, q=trivial, top_score=0.05)
    _add_query(conn, q="real query about something", top_score=0.10)
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 1


def test_count_excludes_short_queries(conn):
    """Length <= 4 chars no cuentan."""
    _add_query(conn, q="abcd", top_score=0.05)  # len 4 → out
    _add_query(conn, q="abcde", top_score=0.05)  # len 5 → in
    n = rag._count_active_learning_candidates(conn, since_days=7)
    assert n == 1


def test_count_handles_missing_table():
    """Si rag_queries no existe, retorna 0 sin raise."""
    c = sqlite3.connect(":memory:")
    n = rag._count_active_learning_candidates(c, since_days=7)
    assert n == 0
    c.close()


# ─── active_learning_nudge ────────────────────────────────────────────


def test_nudge_below_threshold_no_fire(conn, monkeypatch):
    """Con N < threshold, no dispara push."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    # Solo 5 candidates < threshold default 20.
    for i in range(5):
        _add_query(conn, q=f"some low-conf query {i}", top_score=0.10)

    result = rag.active_learning_nudge(threshold=20)
    assert result["n_candidates"] == 5
    assert result["fired"] is False
    assert result["channel_used"] is None


def test_nudge_above_threshold_dry_run(conn, monkeypatch):
    """Con N >= threshold y dry_run=True, NO envía pero reporta."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    result = rag.active_learning_nudge(threshold=20, dry_run=True)
    assert result["n_candidates"] == 25
    assert result["fired"] is False
    assert result["channel_used"] == "(dry-run)"


def test_nudge_channel_macos(conn, monkeypatch):
    """channel='macos' usa osascript directamente."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_macos",
                      return_value=True) as mock_macos, \
         patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=True) as mock_wa:
        result = rag.active_learning_nudge(threshold=20, channel="macos")
    assert result["fired"] is True
    assert result["channel_used"] == "macos"
    mock_macos.assert_called_once()
    mock_wa.assert_not_called()


def test_nudge_channel_wa(conn, monkeypatch):
    """channel='wa' usa solo WA, no fallback."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=True) as mock_wa, \
         patch.object(rag, "_send_active_learning_nudge_macos",
                      return_value=True) as mock_macos:
        result = rag.active_learning_nudge(threshold=20, channel="wa")
    assert result["fired"] is True
    assert result["channel_used"] == "wa"
    mock_wa.assert_called_once()
    mock_macos.assert_not_called()


def test_nudge_channel_wa_failed_no_fallback(conn, monkeypatch):
    """channel='wa' con WA fallando NO cae a macOS — explícito por user."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=False) as mock_wa, \
         patch.object(rag, "_send_active_learning_nudge_macos") as mock_macos:
        result = rag.active_learning_nudge(threshold=20, channel="wa")
    assert result["fired"] is False
    assert result["channel_used"] == "wa-failed"
    mock_macos.assert_not_called()


def test_nudge_channel_auto_wa_first(conn, monkeypatch):
    """channel='auto' (default): WA preferido."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=True) as mock_wa, \
         patch.object(rag, "_send_active_learning_nudge_macos") as mock_macos:
        result = rag.active_learning_nudge(threshold=20, channel="auto")
    assert result["channel_used"] == "wa"
    mock_macos.assert_not_called()


def test_nudge_channel_auto_macos_fallback(conn, monkeypatch):
    """channel='auto': si WA falla, fallback a macOS."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=False) as mock_wa, \
         patch.object(rag, "_send_active_learning_nudge_macos",
                      return_value=True) as mock_macos:
        result = rag.active_learning_nudge(threshold=20, channel="auto")
    assert result["channel_used"] == "macos-fallback"
    mock_wa.assert_called_once()
    mock_macos.assert_called_once()


def test_nudge_channel_auto_all_fail(conn, monkeypatch):
    """Si ambos fallan, channel_used='all-failed', fired=False."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(25):
        _add_query(conn, q=f"low-conf question {i}", top_score=0.10)

    with patch.object(rag, "_send_active_learning_nudge_wa",
                      return_value=False), \
         patch.object(rag, "_send_active_learning_nudge_macos",
                      return_value=False):
        result = rag.active_learning_nudge(threshold=20, channel="auto")
    assert result["fired"] is False
    assert result["channel_used"] == "all-failed"


# ─── source param en _feedback_insert_harvested ────────────────────────


def test_feedback_insert_default_source_harvester(conn, monkeypatch):
    """Default source='harvester' (manual CLI/skill)."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    ok = rag._feedback_insert_harvested(
        q="test query", rating=1, paths=["foo.md"],
    )
    assert ok is True
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback ORDER BY id DESC LIMIT 1"
    ).fetchone()
    import json as _json
    extra = _json.loads(row[0])
    assert extra["source"] == "harvester"


def test_feedback_insert_auto_harvester_source(conn, monkeypatch):
    """source='auto-harvester' tag preserved in extra_json."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    ok = rag._feedback_insert_harvested(
        q="test", rating=1, paths=["foo.md"],
        source="auto-harvester",
    )
    assert ok is True
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback ORDER BY id DESC LIMIT 1"
    ).fetchone()
    import json as _json
    extra = _json.loads(row[0])
    assert extra["source"] == "auto-harvester"


def test_feedback_insert_active_learning_nudge_source(conn, monkeypatch):
    """source='active-learning-nudge' válido para futuros quick-replies."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    ok = rag._feedback_insert_harvested(
        q="test", rating=-1, paths=["foo.md"],
        source="active-learning-nudge",
    )
    assert ok is True
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback ORDER BY id DESC LIMIT 1"
    ).fetchone()
    import json as _json
    extra = _json.loads(row[0])
    assert extra["source"] == "active-learning-nudge"


# ─── Constants sanity ─────────────────────────────────────────────────


def test_constants_in_reasonable_ranges():
    assert 5 <= rag.DEFAULT_ACTIVE_LEARNING_NUDGE_THRESHOLD <= 100
    assert 0.05 <= rag.DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX <= 0.5
    assert 1 <= rag.DEFAULT_ACTIVE_LEARNING_SINCE_DAYS <= 30
