"""Tests para las extensiones de Mood Mirror:
- `get_weekly_summary()` — stats últimos 7 días.
- `check_outbound_tone(jid, draft)` — pre-send tonal check.
- `/api/wa/mood/weekly` + `/api/wa/thread/check-tone` endpoints.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402
from rag.integrations.whatsapp import mood_mirror as _mm  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path):
    """Reescribe DB_PATH y crea rag_mood_score_daily."""
    import rag as _rag
    orig = _rag.DB_PATH
    try:
        _rag.DB_PATH = Path(tmp_path)
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_mood_score_daily ("
                " date TEXT PRIMARY KEY,"
                " score REAL NOT NULL,"
                " n_signals INTEGER NOT NULL,"
                " sources_used TEXT,"
                " top_evidence TEXT,"
                " updated_at REAL NOT NULL)"
            )
            conn.commit()
        yield tmp_path
    finally:
        _rag.DB_PATH = orig


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _seed_mood(date_offset_days: int, score: float, n_signals: int = 3):
    from rag import _ragvec_state_conn
    import time as _time
    d = (datetime.now().date() - timedelta(days=date_offset_days)).isoformat()
    with _ragvec_state_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO rag_mood_score_daily"
            "(date, score, n_signals, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (d, score, n_signals, _time.time()),
        )
        conn.commit()


def test_weekly_summary_empty(isolated_db):
    """Tabla vacía → has_data=False."""
    s = _mm.get_weekly_summary()
    assert s["has_data"] is False
    assert s["avg_7d"] is None


def test_weekly_summary_with_data(isolated_db):
    """Stats correctas: avg, low_days count, high_days count."""
    _seed_mood(0, 0.30)   # high
    _seed_mood(1, -0.20)  # low
    _seed_mood(2, 0.10)   # neutral
    _seed_mood(3, -0.25)  # low
    _seed_mood(4, 0.00)   # neutral
    s = _mm.get_weekly_summary()
    assert s["has_data"] is True
    assert s["low_days"] == 2
    assert s["high_days"] == 1
    assert s["n_days_with_data"] == 5
    assert s["avg_7d"] is not None


def test_weekly_summary_delta_vs_prev_week(isolated_db):
    """Delta calculado contra los 7 días previos."""
    # Esta semana (días 0-6): avg ~0.10
    for d in range(7):
        _seed_mood(d, 0.10)
    # Semana previa (días 7-13): avg ~0.00
    for d in range(7, 14):
        _seed_mood(d, 0.00)
    s = _mm.get_weekly_summary()
    assert s["delta_vs_prev_week"] is not None
    assert s["delta_vs_prev_week"] > 0.05  # mejoró ~0.10


def test_check_outbound_tone_empty_draft(isolated_db):
    """Draft empty / corto → None."""
    assert _mm.check_outbound_tone("x@y", "") is None
    assert _mm.check_outbound_tone("x@y", "hola") is None


def test_check_outbound_tone_no_tense(isolated_db):
    """Draft sin tense pattern → None."""
    assert _mm.check_outbound_tone("x@y", "hola, te paso el resumen del lunes") is None


def test_check_outbound_tone_tense_neutral_mood(isolated_db):
    """Tense pattern pero mood neutral → severity low o medium (depende
    de si la hora actual cae en late_night)."""
    w = _mm.check_outbound_tone("x@y", "siempre haces lo mismo, en serio???")
    assert w is not None
    # Sin mood data: si NO es late → 'low'; si SÍ es late → 'medium'.
    assert w["severity"] in ("low", "medium")


def test_check_outbound_tone_tense_low_mood(isolated_db):
    """Tense + mood bajo → severity high."""
    _seed_mood(0, -0.30)  # hoy bajo
    w = _mm.check_outbound_tone("x@y", "siempre haces lo mismo, en serio???")
    assert w is not None
    assert w["severity"] == "high"
    assert "ánimo bajo" in w["message"] or "bajón" in w["message"].lower()


def test_endpoint_mood_weekly(client, isolated_db):
    """GET /api/wa/mood/weekly devuelve summary."""
    _seed_mood(0, 0.10)
    r = client.get("/api/wa/mood/weekly")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["has_data"] is True


def test_endpoint_check_tone_no_warning(client, isolated_db):
    """POST /check-tone con draft non-tense → warning=null."""
    r = client.post("/api/wa/thread/check-tone",
                    json={"jid": "x@y", "draft": "hola, cómo va todo?"})
    assert r.status_code == 200
    assert r.json()["warning"] is None


def test_endpoint_check_tone_invalid_jid(client, isolated_db):
    """POST /check-tone con jid sin @ → 400."""
    r = client.post("/api/wa/thread/check-tone",
                    json={"jid": "noatsign", "draft": "siempre haces lo mismo!!"})
    assert r.status_code == 400


def test_endpoint_check_tone_with_warning(client, isolated_db):
    """POST /check-tone con draft tense + mood bajo → warning."""
    _seed_mood(0, -0.30)
    r = client.post("/api/wa/thread/check-tone",
                    json={"jid": "x@y",
                          "draft": "siempre haces lo mismo, en serio???"})
    assert r.status_code == 200
    w = r.json()["warning"]
    assert w is not None
    assert w["severity"] == "high"
