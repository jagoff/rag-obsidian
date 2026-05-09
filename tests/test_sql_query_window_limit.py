"""H-5 fix coverage: `_sql_query_window(..., limit=N)` cap.

Antes del fix, `_sql_query_window` no aceptaba `limit` y los call sites del
dashboard (`web.server._dashboard_compute_sql`) podían arrastrar miles de
rows a Python por cada poll (60s) cuando se combinaba con `ancient_iso`
= epoch o con vaults activos donde `days=30` ya devolvía 6000+ rows.

Estos tests validan tres cosas:

1. Pasar `limit=N` devuelve **exactamente N rows** cuando hay >=N rows en la
   ventana — no N+1, no menos.
2. El `ORDER BY ts` se respeta: los rows devueltos son los **más antiguos**
   de la ventana (combinable con un `since_ts` apretado por el call site
   para acotar el resultado a "los últimos M días").
3. Default `limit=None` preserva el comportamiento histórico (sin cap).

Aislamiento: usamos un sqlite3 in-memory standalone con el schema mínimo
(`ts TEXT, payload TEXT`) — no necesitamos el schema completo de telemetry.db
porque `_sql_query_window` solo construye `SELECT * FROM <table> WHERE ts >= ?
[ORDER BY ts] [LIMIT ?]` y devuelve `list[sqlite3.Row]`.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

import rag


@pytest.fixture
def populated_conn():
    """In-memory sqlite con tabla `t` poblada con 50 rows ordenados por `ts`.

    Rows: ts = "2026-01-01T00:00:00" + i minutos, payload = f"row-{i:02d}".
    Insertamos en orden inverso para asegurarnos que el `ORDER BY ts` del
    helper hace el trabajo, no el orden de inserción.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (ts TEXT NOT NULL, payload TEXT)")
    base = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        ((base + timedelta(minutes=i)).isoformat(timespec="seconds"),
         f"row-{i:02d}")
        for i in range(50)
    ]
    # Insert reversed para validar que el ORDER BY en el helper manda.
    for ts, payload in reversed(rows):
        conn.execute("INSERT INTO t (ts, payload) VALUES (?, ?)", (ts, payload))
    conn.commit()
    return conn


def test_limit_returns_exact_count(populated_conn):
    """`limit=10` sobre 50 rows en ventana → devuelve exactamente 10."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(populated_conn, "t", epoch_ish, limit=10)
    assert len(rows) == 10


def test_limit_respects_order_by_ts_ascending(populated_conn):
    """Con `ORDER BY ts` + `limit=10`, los rows devueltos son los 10 MÁS
    ANTIGUOS de la ventana."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(populated_conn, "t", epoch_ish, limit=10)
    payloads = [r["payload"] for r in rows]
    expected = [f"row-{i:02d}" for i in range(10)]
    assert payloads == expected
    # ts también debe estar en orden creciente.
    timestamps = [r["ts"] for r in rows]
    assert timestamps == sorted(timestamps)


def test_limit_none_returns_all_rows_in_window(populated_conn):
    """Default `limit=None` preserva back-compat: trae todo el set."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(populated_conn, "t", epoch_ish)
    assert len(rows) == 50


def test_limit_larger_than_set_returns_all(populated_conn):
    """`limit` mayor al total → devuelve los rows disponibles, sin error."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(populated_conn, "t", epoch_ish, limit=999)
    assert len(rows) == 50


def test_limit_with_since_ts_window(populated_conn):
    """Combinación `since_ts` + `limit`: la ventana se aplica primero, después
    se acota el resultado por LIMIT.

    Rows en la tabla: ts cada minuto desde 2026-01-01T00:00 (50 total).
    `since_ts = 2026-01-01T00:30:00` deja 20 rows visibles (minuto 30..49).
    `limit=5` → primeros 5 de la ventana → minutos 30..34.
    """
    since = "2026-01-01T00:30:00"
    rows = rag._sql_query_window(populated_conn, "t", since, limit=5)
    assert len(rows) == 5
    payloads = [r["payload"] for r in rows]
    assert payloads == [f"row-{i}" for i in range(30, 35)]


def test_limit_zero_returns_empty(populated_conn):
    """`limit=0` → SQL `LIMIT 0` → 0 rows."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(populated_conn, "t", epoch_ish, limit=0)
    assert rows == []


def test_limit_with_where_clause(populated_conn):
    """`where` + `limit` se combinan correctamente. La cláusula `where` se
    aplica con AND y los params se bindean en orden, después el LIMIT."""
    epoch_ish = "2025-01-01T00:00:00"
    rows = rag._sql_query_window(
        populated_conn,
        "t",
        epoch_ish,
        where="payload LIKE ?",
        params=("row-1%",),  # row-10..row-19
        limit=3,
    )
    assert len(rows) == 3
    payloads = [r["payload"] for r in rows]
    assert payloads == ["row-10", "row-11", "row-12"]
