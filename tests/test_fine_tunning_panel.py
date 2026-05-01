"""Tests para el panel /fine_tunning y sus endpoints API.

Casos:
  1.  GET /fine_tunning devuelve HTML con el título correcto.
  2.  GET /api/fine_tunning/queue con tablas vacías devuelve lista vacía.
  3.  GET /api/fine_tunning/queue incluye query de retrieval con low score.
  4.  GET /api/fine_tunning/queue excluye query con score alto (>= 0.15).
  5.  POST /api/fine_tunning/rate persiste la row en rag_ft_panel_ratings.
  6.  POST /api/fine_tunning/rate con rating=0 devuelve 422.
  7.  POST /api/fine_tunning/rate con stream inválido devuelve 422.
  8.  POST /api/fine_tunning/snooze persiste snoozed_until_ts en el futuro.
  9.  GET /api/fine_tunning/queue excluye items snoozeados.
 10.  sw.js incluye la ruta /fine_tunning.
 11.  manifest.webmanifest tiene shortcut a /fine_tunning.
"""

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ── Patrón obligatorio: DB_PATH isolation per-file (snap+restore manual) ──
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag

    snap = _rag.DB_PATH
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    _rag.DB_PATH = db_dir
    # Resetear el set de paths ya inicializados para que _ensure_telemetry_tables
    # corra de nuevo en el tmp dir de este test.
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


# ── Test 1 ─────────────────────────────────────────────────────────────────

def test_get_fine_tunning_returns_html(client):
    r = client.get("/fine_tunning")
    assert r.status_code == 200
    assert "rag · fine_tunning" in r.text
    assert "text/html" in r.headers.get("content-type", "")


# ── Test 2 ─────────────────────────────────────────────────────────────────

def test_queue_empty_returns_empty_list(client, seeded_conn):
    r = client.get("/api/fine_tunning/queue")
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["count"] == 0


# ── Test 3 ─────────────────────────────────────────────────────────────────

def test_queue_includes_retrieval_low_score(client, seeded_conn):
    seeded_conn.execute(
        """
        INSERT INTO rag_queries
            (ts, q, top_score, paths_json, cmd)
        VALUES
            (datetime('now', '-1 hour'), 'qué pasa con esto', 0.05,
             '["a.md"]', 'query')
        """
    )
    seeded_conn.commit()

    r = client.get("/api/fine_tunning/queue")
    assert r.status_code == 200
    items = r.json()["items"]
    retrieval_items = [i for i in items if i["stream"] == "retrieval"]
    assert len(retrieval_items) >= 1
    labels = [i["label"] for i in retrieval_items]
    assert "qué pasa con esto" in labels


# ── Test 4 ─────────────────────────────────────────────────────────────────

def test_queue_excludes_high_score_query(client, seeded_conn):
    seeded_conn.execute(
        """
        INSERT INTO rag_queries
            (ts, q, top_score, paths_json, cmd)
        VALUES
            (datetime('now', '-1 hour'), 'query muy confiable', 0.5,
             '["b.md"]', 'query')
        """
    )
    seeded_conn.commit()

    r = client.get("/api/fine_tunning/queue")
    assert r.status_code == 200
    items = r.json()["items"]
    labels = [i.get("label", "") for i in items]
    assert "query muy confiable" not in labels


# ── Test 5 ─────────────────────────────────────────────────────────────────

def test_post_rate_persists_row(client, seeded_conn):
    body = {
        "stream": "retrieval",
        "item_id": "42",
        "rating": -1,
        "label": "test query",
        "comment": "no encontró nada útil",
    }
    r = client.post("/api/fine_tunning/rate", json=body)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    row = seeded_conn.execute(
        "SELECT stream, item_id, rating, comment FROM rag_ft_panel_ratings"
    ).fetchone()
    assert row is not None
    assert row[0] == "retrieval"
    assert row[1] == "42"
    assert row[2] == -1
    assert row[3] == "no encontró nada útil"


# ── Test 6 ─────────────────────────────────────────────────────────────────

def test_post_rate_invalid_rating_returns_422(client, seeded_conn):
    body = {
        "stream": "retrieval",
        "item_id": "42",
        "rating": 0,
        "label": "x",
    }
    r = client.post("/api/fine_tunning/rate", json=body)
    assert r.status_code == 422


# ── Test 7 ─────────────────────────────────────────────────────────────────

def test_post_rate_invalid_stream_returns_422(client, seeded_conn):
    body = {
        "stream": "invalid_stream_name",
        "item_id": "42",
        "rating": 1,
        "label": "x",
    }
    r = client.post("/api/fine_tunning/rate", json=body)
    assert r.status_code == 422


# ── Test 8 ─────────────────────────────────────────────────────────────────

def test_post_snooze_persists_state(client, seeded_conn):
    body = {"stream": "retrieval", "item_id": "42", "hours": 24}
    r = client.post("/api/fine_tunning/snooze", json=body)
    assert r.status_code == 200

    row = seeded_conn.execute(
        """
        SELECT snoozed_until_ts
        FROM rag_ft_active_queue_state
        WHERE item_id = '42' AND stream = 'retrieval'
        """
    ).fetchone()
    assert row is not None
    snoozed_until = row[0]
    assert snoozed_until is not None
    # Parseable como datetime y en el futuro
    from datetime import datetime

    dt = datetime.fromisoformat(snoozed_until)
    assert dt > datetime.now()


# ── Test 9 ─────────────────────────────────────────────────────────────────

def test_queue_excludes_snoozed_items(client, seeded_conn):
    # Seedear una query low-score
    seeded_conn.execute(
        """
        INSERT INTO rag_queries
            (ts, q, top_score, paths_json, cmd)
        VALUES
            (datetime('now', '-1 hour'), 'pregunta para snooze', 0.04,
             '["c.md"]', 'query')
        """
    )
    seeded_conn.commit()

    # Obtener el id insertado
    row = seeded_conn.execute(
        "SELECT id FROM rag_queries WHERE q = 'pregunta para snooze'"
    ).fetchone()
    assert row is not None
    query_id = str(row[0])

    # Snooze via endpoint
    r = client.post(
        "/api/fine_tunning/snooze",
        json={"stream": "retrieval", "item_id": query_id, "hours": 48},
    )
    assert r.status_code == 200

    # Verificar que no aparece en la cola
    r2 = client.get("/api/fine_tunning/queue")
    assert r2.status_code == 200
    items = r2.json()["items"]
    ids = [i["item_id"] for i in items if i["stream"] == "retrieval"]
    assert query_id not in ids


# ── Test 10 ────────────────────────────────────────────────────────────────

def test_sw_includes_fine_tunning_route():
    sw_content = Path("web/static/sw.js").read_text()
    assert "/fine_tunning" in sw_content


# ── Test 11 ────────────────────────────────────────────────────────────────

def test_manifest_has_fine_tunning_shortcut():
    manifest = json.loads(Path("web/static/manifest.webmanifest").read_text())
    urls = [s["url"] for s in manifest.get("shortcuts", [])]
    assert "/fine_tunning" in urls


# ── Test 12: bridge retrieval_answer → rag_feedback (positive) ─────────────

def test_rate_retrieval_answer_pos_writes_both_tables(client, seeded_conn):
    """Cuando el user puntúa retrieval_answer +1, además de la row en
    rag_ft_panel_ratings se debe persistir un row equivalente en rag_feedback
    para que el ranker-vivo nightly lo levante."""
    seeded_conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, top_score, paths_json, session) "
        "VALUES (datetime('now', '-2 days'), 'web.chat', "
        "'pregunta con respuesta excelente', 0.85, "
        "'[\"01-Projects/X.md\"]', 'web:abc')"
    )
    seeded_conn.commit()
    cur = seeded_conn.execute("SELECT id FROM rag_queries ORDER BY id DESC LIMIT 1")
    qid = cur.fetchone()[0]

    body = {
        "stream": "retrieval_answer",
        "item_id": str(qid),
        "rating": 1,
        "label": "pregunta con respuesta excelente",
        "comment": None,
    }
    r = client.post("/api/fine_tunning/rate", json=body)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    panel_rows = seeded_conn.execute(
        "SELECT stream, rating FROM rag_ft_panel_ratings WHERE item_id = ?",
        (str(qid),),
    ).fetchall()
    assert len(panel_rows) == 1
    assert panel_rows[0] == ("retrieval_answer", 1)

    feedback_rows = seeded_conn.execute(
        "SELECT q, rating FROM rag_feedback WHERE q LIKE 'pregunta con respuesta%'"
    ).fetchall()
    assert len(feedback_rows) == 1
    assert feedback_rows[0][1] == 1


# ── Test 13: bridge retrieval_answer negative + comment carries reason ─────

def test_rate_retrieval_answer_neg_with_comment_persists_reason(client, seeded_conn):
    """Cuando el user puntúa retrieval_answer -1 con un comment, el comment
    debe llegar a rag_feedback como reason para que el ranker-vivo lo lea."""
    seeded_conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, top_score, paths_json, session) "
        "VALUES (datetime('now', '-1 days'), 'web.chat', "
        "'la respuesta del modelo fue incorrecta', 0.7, '[]', 'web:xyz')"
    )
    seeded_conn.commit()
    cur = seeded_conn.execute("SELECT id FROM rag_queries ORDER BY id DESC LIMIT 1")
    qid = cur.fetchone()[0]

    body = {
        "stream": "retrieval_answer",
        "item_id": str(qid),
        "rating": -1,
        "label": "la respuesta del modelo fue incorrecta",
        "comment": "inventó datos que no están en el vault",
    }
    r = client.post("/api/fine_tunning/rate", json=body)
    assert r.status_code == 200

    feedback_rows = seeded_conn.execute(
        "SELECT rating, json_extract(extra_json, '$.reason') "
        "FROM rag_feedback WHERE q LIKE 'la respuesta del modelo%'"
    ).fetchall()
    assert len(feedback_rows) == 1
    rating, reason = feedback_rows[0]
    assert rating == -1
    assert "inventó datos" in (reason or "")


# ── Test 14: otros streams NO tocan rag_feedback ───────────────────────────

def test_rate_other_streams_does_not_touch_rag_feedback(client, seeded_conn):
    """Solo retrieval_answer hace bridge. Otros streams (brief / draft_wa /
    anticipate / proactive_push) deben escribir SOLO a rag_ft_panel_ratings,
    nunca a rag_feedback."""
    initial_count = seeded_conn.execute(
        "SELECT COUNT(*) FROM rag_feedback"
    ).fetchone()[0]

    for stream, item_id in [
        ("brief", "04-Archive/.../2026-04-29-morning.md"),
        ("draft_wa", "draft-abc123"),
        ("anticipate", "cal:event-uuid-1"),
        ("proactive_push", "42"),
    ]:
        body = {
            "stream": stream,
            "item_id": item_id,
            "rating": -1,
            "label": f"{stream} item",
            "comment": "no debería llegar a rag_feedback",
        }
        r = client.post("/api/fine_tunning/rate", json=body)
        assert r.status_code == 200, f"{stream} rate failed: {r.text}"

    final_count = seeded_conn.execute(
        "SELECT COUNT(*) FROM rag_feedback"
    ).fetchone()[0]
    assert final_count == initial_count, (
        f"rag_feedback count cambió de {initial_count} a {final_count} — "
        f"streams != retrieval_answer NO deberían tocar rag_feedback"
    )

    panel_count = seeded_conn.execute(
        "SELECT COUNT(*) FROM rag_ft_panel_ratings WHERE stream != 'retrieval_answer'"
    ).fetchone()[0]
    assert panel_count == 4, "Las 4 rates deberían estar en rag_ft_panel_ratings"
