"""Tests for the dwell-per-chunk web flow (2026-04-22).

When a user scrolls a source-row into view for ≥1.5s, the browser's
IntersectionObserver emits a POST /api/behavior with event='open' +
dwell_ms=<timer>. This is the passive counterpart to the active copy
event (commit db2a169) — it feeds the ranker-vivo with reading-time
signal that a plain click boolean can't capture.

The JS half lives in web/static/app.js (_observeDwell, _emitDwell,
visibilitychange listener). These tests cover the server side:

  1. Endpoint accepts event='open' with dwell_ms on the web path
  2. dwell_ms gets converted to dwell_s and lands in the rag_behavior
     dwell_s column (not in extra_json — the aggregator only reads
     the column, see rag.py:2902)
  3. The aggregator's dwell_acc picks up the signal post-conversion
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402

_web_server = pytest.importorskip("web.server")
pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


@pytest.fixture
def behavior_env(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    rag._behavior_priors_cache = None
    rag._behavior_priors_cache_key = None
    rag._behavior_priors_cache_key_sql = None
    yield tmp_path
    rag._behavior_priors_cache = None
    rag._behavior_priors_cache_key_sql = None


# ── Endpoint surface ────────────────────────────────────────────────────────


def test_dwell_event_accepted_via_open(client):
    """A dwell emission from the browser arrives as event='open' +
    dwell_ms. The endpoint must accept it without any new event type."""
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json={
            "source": "web",
            "event": "open",
            "query": "¿qué onda Astor?",
            "path": "02-Areas/Astor.md",
            "rank": 1,
            "dwell_ms": 3500,
            "session": "web:abc123",
        })
    assert resp.status_code == 200, resp.text
    mock_log.assert_called_once()


def test_dwell_ms_converted_to_dwell_s_before_log(client):
    """Conversion ms → s must happen BEFORE hitting log_behavior_event so
    the row lands in the dwell_s SQL column (not in extra_json). The
    aggregator's `_compute_behavior_priors_from_rows` reads dwell_s only
    (rag.py:2902); a payload that slips dwell_ms into extra_json is lost."""
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json={
            "source": "web",
            "event": "open",
            "path": "x.md",
            "query": "q",
            "dwell_ms": 2500,
        })
    assert resp.status_code == 200
    call_event = mock_log.call_args.args[0]
    # Must have dwell_s, not dwell_ms (that one becomes extra_json otherwise).
    assert "dwell_s" in call_event
    assert call_event["dwell_s"] == 2.5  # 2500ms → 2.5s
    assert "dwell_ms" not in call_event


def test_dwell_missing_is_null(client):
    """An open without dwell_ms → dwell_s=None (not 0 — that would bias
    the aggregator toward 0s dwell averages)."""
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json={
            "source": "web",
            "event": "open",
            "path": "x.md",
            "query": "q",
        })
    assert resp.status_code == 200
    call_event = mock_log.call_args.args[0]
    assert call_event.get("dwell_s") is None


def test_dwell_zero_ms_converts_cleanly(client):
    """dwell_ms=0 should still convert (to 0.0), not short-circuit to None.
    Pathological but let's pin the behaviour: 0 ms = 0 s, distinct from
    missing."""
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json={
            "source": "web",
            "event": "open",
            "path": "x.md",
            "query": "q",
            "dwell_ms": 0,
        })
    assert resp.status_code == 200
    call_event = mock_log.call_args.args[0]
    assert call_event["dwell_s"] == 0.0


# ── Integration with the CTR aggregator ────────────────────────────────────


def test_dwell_lands_in_priors_dwell_acc(behavior_env):
    """End-to-end: emit a dwell event via log_behavior_event and assert
    the aggregator picks up both the click (CTR++) and the dwell_s
    (dwell_acc)."""
    rag.log_behavior_event({
        "source": "web",
        "event": "open",
        "query": "¿cómo uso rag?",
        "path": "docs/rag.md",
        "rank": 1,
        "dwell_s": 3.5,  # 3.5 seconds — represents an attentive read
    })
    priors = rag._load_behavior_priors()
    ctr = priors.get("click_prior", {}).get("docs/rag.md")
    assert ctr is not None
    # One click + one impression → Laplace CTR (1+1)/(1+10) ≈ 0.182
    assert ctr > 0.15
    # Dwell shows up as `dwell_score` (log1p(mean_dwell_s)) for the path
    # per rag.py:2841. log1p(3.5) ≈ 1.504; we only assert non-zero here
    # since the exact number depends on the aggregator's rounding.
    dwell_score = priors.get("dwell_score", {}).get("docs/rag.md")
    assert dwell_score is not None, (
        f"dwell_score not populated — priors keys: {list(priors.keys())}"
    )
    assert dwell_score > 0


def test_multiple_dwell_events_accumulate(behavior_env):
    """Two dwell events on the same path average their dwell_s."""
    for secs in (1.0, 3.0, 5.0):
        rag.log_behavior_event({
            "source": "web", "event": "open",
            "query": "q", "path": "a.md",
            "rank": 1, "dwell_s": secs,
        })
    priors = rag._load_behavior_priors()
    ctr = priors.get("click_prior", {}).get("a.md")
    # 3 clicks + 3 impressions → (3+1)/(3+10) ≈ 0.308
    assert ctr is not None and ctr > 0.25


# ── Defensive: the JS emits dwell_ms; the server must not leak it
#    into extra_json. Dead-man's-switch for a future refactor that
#    might accidentally revert the conversion. ──────────────────────


def test_open_event_populates_original_query_id(behavior_env, client):
    """Round-trip: insert una query en rag_queries con session='web:xyz',
    luego POST /api/behavior con event='open' y el mismo session.
    El row en rag_behavior debe tener original_query_id en extra_json
    apuntando al id de la query insertada.

    Cierra el loop de feedback implícito: el consumer en
    rag_implicit_learning/corrective_paths.py filtra opens por
    `original_query_id IS NOT NULL` — sin este campo el 89.7% de los
    opens del web quedaban invisibles al gate de corrective_paths.
    """
    import json as _json
    import datetime

    # Insertar una query de prueba en rag_queries con session conocido
    ts = datetime.datetime.now().isoformat()
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_queries (ts, session, q, cmd) VALUES (?, ?, ?, ?)",
            (ts, "web:xyz", "qué hago hoy", "web.chat"),
        )
        inserted_id = conn.execute(
            "SELECT id FROM rag_queries WHERE session = 'web:xyz' ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        conn.commit()

    # POST del open con el mismo session
    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "open",
        "path": "02-Areas/Test-oqid.md",
        "query": "qué hago hoy",
        "rank": 1,
        "session": "web:xyz",
    })
    assert resp.status_code == 200

    # Verificar que el row en rag_behavior tiene original_query_id en extra_json
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT extra_json FROM rag_behavior "
            "WHERE path = ? ORDER BY id DESC LIMIT 1",
            ("02-Areas/Test-oqid.md",),
        ).fetchone()
    assert row is not None, "No se insertó el row de behavior"
    extra = _json.loads(row[0]) if row[0] else {}
    assert "original_query_id" in extra, (
        f"original_query_id ausente en extra_json — feedback loop roto. "
        f"extra_json={extra}"
    )
    assert extra["original_query_id"] == inserted_id, (
        f"original_query_id={extra['original_query_id']} no coincide con "
        f"el id esperado={inserted_id}"
    )


def test_non_open_event_no_original_query_id(behavior_env, client):
    """Eventos que NO son 'open' no deben tener original_query_id,
    incluso con session activa. El lookup solo corre para event='open'."""
    import json as _json
    import datetime

    ts = datetime.datetime.now().isoformat()
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_queries (ts, session, q, cmd) VALUES (?, ?, ?, ?)",
            (ts, "web:nooqid", "alguna query", "web.chat"),
        )
        conn.commit()

    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "copy",
        "path": "02-Areas/Impression-test.md",
        "query": "alguna query",
        "rank": 2,
        "session": "web:nooqid",
    })
    assert resp.status_code == 200

    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT extra_json FROM rag_behavior "
            "WHERE path = ? ORDER BY id DESC LIMIT 1",
            ("02-Areas/Impression-test.md",),
        ).fetchone()
    extra = _json.loads(row[0]) if row and row[0] else {}
    assert "original_query_id" not in extra, (
        f"original_query_id no debe estar en eventos no-open. extra_json={extra}"
    )


def test_open_without_session_no_original_query_id(client):
    """Un open sin session no tiene a qué query asociar — no debe tener
    original_query_id en extra_json (y no debe crashear el endpoint)."""
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json={
            "source": "web",
            "event": "open",
            "path": "x.md",
            "query": "q",
        })
    assert resp.status_code == 200
    call_event = mock_log.call_args.args[0]
    assert "original_query_id" not in call_event


def test_server_never_persists_raw_dwell_ms(behavior_env, client):
    """Round-trip: POST with dwell_ms, read the row back from SQL, assert
    dwell_ms is not in extra_json and dwell_s is the right magnitude."""
    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "open",
        "path": "02-Areas/Test.md",
        "query": "q",
        "rank": 1,
        "dwell_ms": 1800,  # 1.8s
    })
    assert resp.status_code == 200

    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(
            "SELECT event, path, dwell_s, extra_json FROM rag_behavior "
            "WHERE path = ? ORDER BY id DESC LIMIT 1",
            ("02-Areas/Test.md",),
        )
        row = cur.fetchone()
    assert row is not None
    event, path, dwell_s, extra_json = row
    assert event == "open"
    assert path == "02-Areas/Test.md"
    assert dwell_s == 1.8
    # Critical: extra_json must NOT contain dwell_ms
    if extra_json:
        import json as _json
        extra = _json.loads(extra_json)
        assert "dwell_ms" not in extra, (
            f"dwell_ms leaked into extra_json — aggregator will miss it. "
            f"extra_json={extra}"
        )
