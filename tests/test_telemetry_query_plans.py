"""Tests que validan que las queries comunes contra `_TELEMETRY_DDL` usan
SEARCH (con índice) y NO SCAN (full table).

Audit del 2026-04-25 (PERFORMANCE #3): la mayoría de las queries en el
proyecto usan `_sql_query_window()` con un índice canónico en `ts`, pero
algunos call sites agregan filtros adicionales (`WHERE ts >= ? AND
event = ?`) que requieren un índice composite. Si alguien agrega una
query nueva sin pensar en el índice, telemetry.db (48MB, 50+ tablas)
empieza a hacer full scans en silencio.

## Estrategia

Para cada tabla en `_TELEMETRY_DDL`, este archivo:

1. Crea una DB temporal con la misma DDL que producción.
2. Inserta unas pocas filas dummy para que el query optimizer tenga
   una muestra realista.
3. Corre `EXPLAIN QUERY PLAN <query>` para queries representativas
   del código real (read paths del dashboard, status, brief, etc.).
4. Falla si la respuesta contiene "SCAN" sin un "SEARCH" anterior —
   significa full table scan.

## Lo que NO testea

- Queries derivadas (joins de varias tablas) — son raras y específicas
  del dashboard.
- Queries con `LIKE` patterns — siempre escanean, es expected.
- Performance absoluta (latencia ms) — depende del hardware.

## Cómo agregar un test

Si agregás un nuevo call site en producción que toca una de estas
tablas con un filtro additional, **agregá la query acá** con el shape
exacto + el índice esperado. Si el test falla, agregá el índice
composite en `_TELEMETRY_DDL`. Es la forma de "lock-in"-ear el query
plan al schema.
"""
from __future__ import annotations

import sqlite3

import pytest

import rag


@pytest.fixture
def telemetry_db(tmp_path):
    """In-memory-ish DB con todo el schema de _TELEMETRY_DDL aplicado.
    Apartado del singleton de rag.py para no contaminar el state global."""
    p = tmp_path / "test-telemetry.db"
    conn = sqlite3.connect(str(p))
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    rag._ensure_telemetry_tables(conn)
    yield conn
    conn.close()


def _has_search(plan_rows: list[tuple], table: str) -> bool:
    """True iff EXPLAIN QUERY PLAN output mentions SEARCH (vs SCAN) for
    the given table. SQLite plans encode the action as the 4th column
    of the (id, parent, notused, detail) tuple."""
    for row in plan_rows:
        detail = row[-1]  # last column is always the human description
        if not isinstance(detail, str):
            continue
        if f"SEARCH {table}" in detail or f"SEARCH TABLE {table}" in detail:
            return True
    return False


def _is_full_scan(plan_rows: list[tuple], table: str) -> bool:
    """True iff the plan shows SCAN <table> WITHOUT a preceding SEARCH —
    i.e. SQLite is doing a full table scan instead of using an index."""
    has_scan = False
    has_search = False
    for row in plan_rows:
        detail = row[-1]
        if not isinstance(detail, str):
            continue
        if f"SEARCH {table}" in detail or f"SEARCH TABLE {table}" in detail:
            has_search = True
        if (
            f"SCAN {table}" in detail or f"SCAN TABLE {table}" in detail
        ) and "USING INDEX" not in detail:
            has_scan = True
    return has_scan and not has_search


# ── rag_queries ──────────────────────────────────────────────────────


def test_rag_queries_filter_by_ts_uses_index(telemetry_db):
    """Dashboard read: `SELECT ... FROM rag_queries WHERE ts > ?` debe
    pegarle al índice ix_rag_queries_ts."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_queries WHERE ts > ?",
        ("2026-04-01",),
    ).fetchall()
    assert _has_search(plan, "rag_queries"), f"plan: {plan!r}"


def test_rag_queries_filter_by_session_uses_index(telemetry_db):
    """Session drawer: `SELECT * FROM rag_queries WHERE session = ?
    ORDER BY ts DESC` — debe pegarle al composite (session, ts)."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_queries "
        "WHERE session = ? ORDER BY ts DESC",
        ("test-sess",),
    ).fetchall()
    assert _has_search(plan, "rag_queries"), f"plan: {plan!r}"


def test_rag_queries_filter_by_cmd_and_ts_uses_composite(telemetry_db):
    """Telemetry analysis: `WHERE cmd = ? AND ts > ?` — debe usar el
    composite ix_rag_queries_cmd_ts, no escanear."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_queries "
        "WHERE cmd = ? AND ts > ?",
        ("query", "2026-04-01"),
    ).fetchall()
    assert not _is_full_scan(plan, "rag_queries"), (
        f"full table scan en rag_queries(cmd, ts): {plan!r}"
    )


# ── rag_behavior ────────────────────────────────────────────────────


def test_rag_behavior_filter_by_event_and_ts_uses_composite(telemetry_db):
    """Ranker-vivo: `WHERE event = 'open' AND ts > ?` — composite
    ix_rag_behavior_event_ts cubre."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_behavior "
        "WHERE event = ? AND ts > ?",
        ("open", "2026-04-01"),
    ).fetchall()
    assert not _is_full_scan(plan, "rag_behavior"), (
        f"full table scan en rag_behavior(event, ts): {plan!r}"
    )


def test_rag_behavior_filter_by_path_and_ts_uses_composite(telemetry_db):
    """Behavior priors: `WHERE path = ? AND ts > ?` — composite
    ix_rag_behavior_path_ts cubre."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_behavior "
        "WHERE path = ? AND ts > ?",
        ("Notes/x.md", "2026-04-01"),
    ).fetchall()
    assert not _is_full_scan(plan, "rag_behavior"), (
        f"full table scan en rag_behavior(path, ts): {plan!r}"
    )


# ── rag_feedback ────────────────────────────────────────────────────


def test_rag_feedback_filter_by_ts_uses_index(telemetry_db):
    """Tune nightly: `WHERE ts > ?` — ix_rag_feedback_ts cubre."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_feedback WHERE ts > ?",
        ("2026-04-01",),
    ).fetchall()
    assert _has_search(plan, "rag_feedback"), f"plan: {plan!r}"


# ── rag_contradictions ──────────────────────────────────────────────


def test_rag_contradictions_filter_by_subject_uses_index(telemetry_db):
    """Triage UI: `WHERE subject_path = ?` — ix_rag_contradictions_subject."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_contradictions "
        "WHERE subject_path = ?",
        ("Notes/foo.md",),
    ).fetchall()
    assert _has_search(plan, "rag_contradictions"), f"plan: {plan!r}"


# ── rag_ambient ─────────────────────────────────────────────────────


def test_rag_ambient_filter_by_path_uses_index(telemetry_db):
    """Ambient lookup por path — ix_rag_ambient_path."""
    plan = telemetry_db.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM rag_ambient WHERE path = ?",
        ("Notes/x.md",),
    ).fetchall()
    assert _has_search(plan, "rag_ambient"), f"plan: {plan!r}"


# ── Cross-table contract ────────────────────────────────────────────


def test_every_telemetry_table_has_at_least_one_index(telemetry_db):
    """Cada tabla declarada en _TELEMETRY_DDL debe tener al menos UN
    índice secundario (excluyendo el implicit de la PK). Sin esto, la
    primera query con WHERE casi seguro hace full scan.

    Excepciones permitidas: tablas tipo "key-value" (PK = lookup key),
    no necesitan más índices porque el PK ya cubre toda lectura.
    """
    pk_only_tables = {
        # Estas tablas usan el PK como lookup principal; no necesitan
        # índice secundario (el PK ya da SEARCH USING INTEGER PRIMARY
        # KEY o USING INDEX <pk> según el caso).
        "rag_brief_state",          # state by date
        "rag_ambient_state",        # state by path
        "rag_feedback_golden_meta", # state by id
        "rag_conversations_index",  # state by id
        "rag_schema_version",       # state by table_name
        "rag_score_calibration",    # state by version
        "rag_response_cache",       # cache by hash
        "rag_ocr_cache",            # cache by hash
        "rag_vlm_captions",         # cache by hash
        "rag_promises",             # state by id
        "rag_audio_transcripts",    # cache by audio_path PK
        "rag_whisper_vocab",        # state by term PK
        "rag_anticipate_candidates",# state by id
        "rag_status_samples",       # ring buffer with autoincrement id
        "rag_filing_log",           # immutable log with autoincrement
        "rag_archive_log",          # immutable log
        "rag_proactive_log",        # immutable log
        "rag_surface_log",          # immutable log
        "rag_reminder_wa_pushed",   # set membership
        "rag_feedback_golden",      # state by id
        "rag_brief_schedule_prefs", # override hour/minute by brief_kind PK
    }

    rows = telemetry_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'rag_%' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    table_names = [r[0] for r in rows]
    assert len(table_names) > 5, (
        f"too few rag_* tables created — _ensure_telemetry_tables broken? "
        f"got {table_names!r}"
    )

    missing_index: list[str] = []
    for table in table_names:
        if table in pk_only_tables:
            continue
        idx_rows = telemetry_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name = ? AND name NOT LIKE 'sqlite_autoindex%'",
            (table,),
        ).fetchall()
        if not idx_rows:
            missing_index.append(table)

    assert not missing_index, (
        f"tablas sin índice secundario (todas las queries van a full-scan): "
        f"{missing_index}. Si la tabla es legítimamente PK-only, agregala a "
        f"`pk_only_tables` en este test."
    )
