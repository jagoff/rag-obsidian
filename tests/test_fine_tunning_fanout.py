"""Tests del fan-out de /api/fine_tunning/rate a las tablas downstream.

El panel evalúa **conversaciones reales** del modelo (decisión 2026-05-01,
documentada en el comentario de diseño en `web/server.py` arriba de
`FineTunningRateRequest`): la lista oficial de streams válidos es

  retrieval_answer  — respuesta del LLM a una query del chat web
  draft / draft_wa  — drafts que el bot redactó para responder a contactos
  style / tool_routing / read_summary — extensible (no populated todavía)

Streams **removidos** del validator (no testear): `retrieval` (input humano,
no respuesta del modelo), `brief` / `anticipate` / `proactive_push`
(pushes sin "respuesta correcta" objetiva), `whatsapp_msg` (input humano).
Los tests que existían sobre esos streams se borraron en 2026-05-04 al
detectarlos rotos durante el audit de quick wins.

Este archivo testea que cada stream válido escribe la row correcta en
su tabla downstream:

  retrieval_answer / draft_wa → rag_feedback (con scope correspondiente)

El fan-out es best-effort: si la escritura downstream falla, el rating
sigue persistiendo en `rag_ft_panel_ratings`. Los tests verifican el
happy path; el silent-fail está cubierto por la cobertura general del
endpoint (retorna 200 incluso si el downstream rompe).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Patrón obligatorio: DB_PATH isolation per-file (mismo que el panel) ──
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag

    snap = _rag.DB_PATH
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    _rag.DB_PATH = db_dir
    from rag import _TELEMETRY_DDL_ENSURED_PATHS, _TELEMETRY_DDL_LOCK

    with _TELEMETRY_DDL_LOCK:
        snapped_paths = set(_TELEMETRY_DDL_ENSURED_PATHS)
        _TELEMETRY_DDL_ENSURED_PATHS.clear()
    try:
        yield db_dir
    finally:
        _rag.DB_PATH = snap
        with _TELEMETRY_DDL_LOCK:
            _TELEMETRY_DDL_ENSURED_PATHS.clear()
            _TELEMETRY_DDL_ENSURED_PATHS.update(snapped_paths)


@pytest.fixture
def client():
    from web.server import app

    return TestClient(app)


@pytest.fixture
def seeded_conn(_isolate_db_path):
    """Conexión a la telemetry.db aislada con todas las tablas creadas."""
    import rag as _rag

    db_file = _rag.DB_PATH / "telemetry.db"
    conn = sqlite3.connect(db_file)
    _rag._ensure_telemetry_tables(conn)
    conn.commit()
    yield conn
    conn.close()


# ── retrieval_answer → rag_feedback ────────────────────────────────────

def test_rate_retrieval_answer_writes_to_rag_feedback(client, seeded_conn):
    seeded_conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, top_score, paths_json, session) "
        "VALUES (datetime('now', '-1 hour'), 'web.chat', "
        "'pregunta con respuesta excelente', 0.85, "
        "'[\"01-Projects/X.md\"]', 'web:xyz')"
    )
    seeded_conn.commit()
    qid = seeded_conn.execute(
        "SELECT id FROM rag_queries ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]

    r = client.post("/api/fine_tunning/rate", json={
        "stream": "retrieval_answer",
        "item_id": str(qid),
        "rating": 1,
        "label": "pregunta con respuesta excelente",
        "comment": None,
    })
    assert r.status_code == 200
    feedback_rows = seeded_conn.execute(
        "SELECT q, rating, scope FROM rag_feedback "
        "WHERE turn_id = ?", (f"fine_tunning:{qid}",),
    ).fetchall()
    assert len(feedback_rows) == 1
    assert feedback_rows[0][1] == 1
    assert feedback_rows[0][2] == "retrieval_answer"


# ── draft_wa → rag_feedback (con scope='draft_wa') ─────────────────────

def test_rate_draft_wa_writes_to_rag_feedback(client, seeded_conn):
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "draft_wa",
        "item_id": "draft-abc123",
        "rating": 1,
        "label": "→ Maria: Listo, gracias.",
        "comment": "el draft estuvo perfecto",
    })
    assert r.status_code == 200

    rows = seeded_conn.execute(
        "SELECT rating, q, scope, json_extract(extra_json, '$.reason') "
        "FROM rag_feedback WHERE turn_id = 'fine_tunning:draft-abc123'"
    ).fetchall()
    assert len(rows) == 1
    rating, q, scope, reason = rows[0]
    assert rating == 1
    assert q == "→ Maria: Listo, gracias."
    assert scope == "draft_wa"
    assert "perfecto" in (reason or "")


# ── Idempotencia: no duplicar rows en rag_feedback ─────────────────────

def test_rate_retrieval_answer_is_idempotent(client, seeded_conn):
    """Si el panel muestra la misma query 2× (porque el user no la rateó
    pero tampoco snoozeó) y vota en ambas, el bridge NO debe duplicar
    la row en rag_feedback."""
    seeded_conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, top_score, paths_json, session) "
        "VALUES (datetime('now'), 'web.chat', 'pregunta repetida', 0.7, '[]', 'web:1')"
    )
    seeded_conn.commit()
    qid = seeded_conn.execute(
        "SELECT id FROM rag_queries ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]

    body = {
        "stream": "retrieval_answer",
        "item_id": str(qid),
        "rating": 1,
        "label": "pregunta repetida",
    }
    client.post("/api/fine_tunning/rate", json=body)
    client.post("/api/fine_tunning/rate", json=body)

    count = seeded_conn.execute(
        "SELECT COUNT(*) FROM rag_feedback WHERE q = 'pregunta repetida'"
    ).fetchone()[0]
    # 2 panel ratings, pero solo 1 feedback row gracias al dedup.
    assert count == 1
