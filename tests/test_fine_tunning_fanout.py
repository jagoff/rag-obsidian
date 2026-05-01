"""Tests del fan-out de /api/fine_tunning/rate a las tablas downstream.

El panel recibe thumbs sobre 7 streams (retrieval, retrieval_answer, brief,
draft_wa, anticipate, proactive_push, whatsapp_msg). Hasta 2026-05-01 los
ratings sólo caían a `rag_ft_panel_ratings` y nadie los leía → el voto
no enseñaba nada.

Este archivo testea que cada stream escribe la row correcta en su tabla
downstream:

  retrieval / retrieval_answer / draft_wa / whatsapp_msg → rag_feedback
  anticipate                                              → rag_anticipate_feedback
  proactive_push                                          → rag_anticipate_feedback (dedup_key sintético)
  brief                                                   → rag_brief_feedback

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
    # rag_anticipate_feedback se crea via _ensure_feedback_table en el helper.
    # rag_brief_feedback ya está en la lista core de _ensure_telemetry_tables.
    conn.commit()
    yield conn
    conn.close()


# ── retrieval (sin _answer) → rag_feedback ───────────────────────────

def test_rate_retrieval_writes_to_rag_feedback(client, seeded_conn):
    """Voto sobre stream='retrieval' debe escribir row en rag_feedback con
    scope='retrieval' para que el ranker-vivo lo levante."""
    seeded_conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, top_score, paths_json, session) "
        "VALUES (datetime('now', '-1 hour'), 'query', "
        "'que pasa con esto', 0.05, '[\"a.md\"]', 'web:abc')"
    )
    seeded_conn.commit()
    qid = seeded_conn.execute(
        "SELECT id FROM rag_queries ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]

    r = client.post("/api/fine_tunning/rate", json={
        "stream": "retrieval",
        "item_id": str(qid),
        "rating": -1,
        "label": "que pasa con esto",
        "comment": "no encontró nada útil",
    })
    assert r.status_code == 200

    rows = seeded_conn.execute(
        "SELECT rating, q, scope FROM rag_feedback WHERE turn_id = ?",
        (f"fine_tunning:{qid}",),
    ).fetchall()
    assert len(rows) == 1
    rating, q, scope = rows[0]
    assert rating == -1
    assert q == "que pasa con esto"
    assert scope == "retrieval"


# ── retrieval_answer → rag_feedback (legacy bridge, sigue válido) ─────

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


# ── anticipate → rag_anticipate_feedback ─────────────────────────────

def test_rate_anticipate_writes_to_anticipate_feedback(client, seeded_conn):
    """Voto sobre stream='anticipate' debe escribir row en rag_anticipate_feedback
    con dedup_key = item_id y rating string ('positive'/'negative')."""
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "anticipate",
        "item_id": "cal:event-uuid-test-1",
        "rating": -1,
        "label": "📅 En 17 min: Entrega llave casa",
        "comment": "ya lo sé, no me hace falta el push",
    })
    assert r.status_code == 200

    # rag_anticipate_feedback table es creada por record_feedback() del
    # módulo rag_anticipate.feedback (lazy via _ensure_feedback_table).
    rows = seeded_conn.execute(
        "SELECT rating, dedup_key, source, reason FROM rag_anticipate_feedback "
        "WHERE dedup_key = ?",
        ("cal:event-uuid-test-1",),
    ).fetchall()
    assert len(rows) == 1
    rating, dedup_key, source, reason = rows[0]
    assert rating == "negative"
    assert dedup_key == "cal:event-uuid-test-1"
    assert source == "panel"
    assert "ya lo sé" in (reason or "")


def test_rate_anticipate_positive_maps_correctly(client, seeded_conn):
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "anticipate",
        "item_id": "gap:test-2",
        "rating": 1,
        "label": "🧭 33 veces preguntaste...",
    })
    assert r.status_code == 200
    row = seeded_conn.execute(
        "SELECT rating FROM rag_anticipate_feedback WHERE dedup_key = ?",
        ("gap:test-2",),
    ).fetchone()
    assert row is not None
    assert row[0] == "positive"


# ── proactive_push → rag_anticipate_feedback con dedup_key sintético ──

def test_rate_proactive_push_uses_synthetic_dedup_key(client, seeded_conn):
    """proactive_push usa dedup_key=`proactive:<item_id>` para distinguir
    de los anticipate genuinos (que usan el dedup_key real del agent).
    Esto permite consumers separados si en el futuro se quiere tunear
    distinto el threshold del proactive scheduler vs el del Anticipatory
    Agent."""
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "proactive_push",
        "item_id": "42",
        "rating": -1,
        "label": "🔗 Nota nueva sin links: [[2026-04-30]]",
        "comment": "no quiero que me alerte de notas chicas",
    })
    assert r.status_code == 200

    rows = seeded_conn.execute(
        "SELECT rating, dedup_key, source FROM rag_anticipate_feedback "
        "WHERE dedup_key = ?",
        ("proactive:42",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "negative"
    assert rows[0][1] == "proactive:42"
    assert rows[0][2] == "panel"


# ── brief → rag_brief_feedback ────────────────────────────────────────

def test_rate_brief_writes_to_brief_feedback(client, seeded_conn):
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "brief",
        "item_id": "Briefs/2026-04-29-morning.md",
        "rating": -1,
        "label": "Briefs/2026-04-29-morning.md",
        "comment": "muy temprano, mover a las 9",
    })
    assert r.status_code == 200

    rows = seeded_conn.execute(
        "SELECT rating, dedup_key, source, reason FROM rag_brief_feedback "
        "WHERE dedup_key = ?",
        ("Briefs/2026-04-29-morning.md",),
    ).fetchall()
    assert len(rows) == 1
    rating, dedup_key, source, reason = rows[0]
    assert rating == "negative"
    assert dedup_key == "Briefs/2026-04-29-morning.md"
    assert source == "panel"
    assert "muy temprano" in (reason or "")


# ── draft_wa → rag_feedback (con scope='draft_wa') ────────────────────

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


# ── whatsapp_msg → rag_feedback (paths = [item_id]) ──────────────────

def test_rate_whatsapp_msg_writes_to_rag_feedback_with_path(client, seeded_conn):
    """El item_id de whatsapp_msg es el path del chunk WA. Lo metemos como
    single-path para que el ranker tenga señal contextual."""
    item_id = "whatsapp://1234@s.whatsapp.net/msg-abc"
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "whatsapp_msg",
        "item_id": item_id,
        "rating": -1,
        "label": "[Juan] Che, viste el artículo sobre...",
        "comment": "irrelevante, no puntuar este chat para queries",
    })
    assert r.status_code == 200

    rows = seeded_conn.execute(
        "SELECT rating, q, scope, paths_json FROM rag_feedback "
        "WHERE turn_id = ?",
        (f"fine_tunning:{item_id}",),
    ).fetchall()
    assert len(rows) == 1
    rating, q, scope, paths_json = rows[0]
    assert rating == -1
    assert scope == "whatsapp_msg"
    assert item_id in (paths_json or "")


# ── Idempotencia: no duplicar rows en rag_feedback ────────────────────

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


# ── Silent-fail si el item_id no existe ────────────────────────────────

def test_rate_retrieval_with_invalid_item_id_does_not_crash(client, seeded_conn):
    """Si el item_id apunta a una query inexistente, el endpoint NO debe
    crashear ni 500. El panel rating queda persistido, el fan-out cae
    silently."""
    r = client.post("/api/fine_tunning/rate", json={
        "stream": "retrieval",
        "item_id": "999999999",  # no existe
        "rating": 1,
        "label": "fallback q",
    })
    assert r.status_code == 200

    # Panel rating sí persistido.
    panel_count = seeded_conn.execute(
        "SELECT COUNT(*) FROM rag_ft_panel_ratings WHERE item_id = ?",
        ("999999999",),
    ).fetchone()[0]
    assert panel_count == 1
