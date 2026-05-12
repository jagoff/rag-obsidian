"""Tests para `GET /api/wa/anticipate/today` — surface del drawer wzp.

El endpoint corre las signals registradas en `_ANTICIPATE_SIGNALS`, filtra
por `min_score`, sortea desc, slicea a `limit`, y serializa al shape UI
(incluyendo `target_jid`, `target_name`, `draft`, `draft_meta` — los 4
campos de enrichment que `_anticipate_candidate_to_dict` omite).

Cache: 120s in-process. Tests resetean el cache entre runs via fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402

from rag.anticipatory import AnticipatoryCandidate  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Reset cache in-process antes de cada test. Sin esto, un test que
    cachea candidates filtra al siguiente."""
    _web_server._WZP_ANTICIPATE_CACHE._cache = {"ts": 0.0, "payload": None}
    # Asegurar que el flag de disabled no contamine
    monkeypatch.delenv("RAG_ANTICIPATE_DISABLED", raising=False)
    yield
    _web_server._WZP_ANTICIPATE_CACHE._cache = {"ts": 0.0, "payload": None}


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _mk(score, *, kind="anticipate-calendar", dedup="k", target_jid=None,
        target_name=None, draft=None, draft_meta=None):
    return AnticipatoryCandidate(
        kind=kind,
        score=score,
        message=f"msg para {kind}",
        dedup_key=dedup,
        snooze_hours=2,
        reason=f"reason {kind}",
        target_jid=target_jid,
        target_name=target_name,
        draft=draft,
        draft_meta=draft_meta,
    )


def test_returns_top_n_sorted_desc(client, monkeypatch):
    """Top-N ordenado por score descendente, respetando `limit`."""
    fake_signals = [
        ("low", lambda now: [_mk(0.40, dedup="low-1")]),
        ("hi", lambda now: [_mk(0.85, dedup="hi-1"), _mk(0.92, dedup="hi-2")]),
        ("mid", lambda now: [_mk(0.55, dedup="mid-1")]),
    ]
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", fake_signals)

    r = client.get("/api/wa/anticipate/today?limit=3")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["disabled"] is False
    assert len(body["candidates"]) == 3
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert body["candidates"][0]["dedup_key"] == "hi-2"  # 0.92
    assert body["candidates"][1]["dedup_key"] == "hi-1"  # 0.85
    assert body["candidates"][2]["dedup_key"] == "mid-1"  # 0.55


def test_respects_min_score(client, monkeypatch):
    """Candidates por debajo de `min_score` quedan filtrados."""
    fake_signals = [
        ("a", lambda now: [_mk(0.10, dedup="a1"), _mk(0.80, dedup="a2")]),
    ]
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", fake_signals)

    r = client.get("/api/wa/anticipate/today?min_score=0.5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["dedup_key"] == "a2"


def test_respects_limit(client, monkeypatch):
    """Slice a `limit` candidates exactos."""
    fake_signals = [
        ("a", lambda now: [_mk(0.9 - i * 0.05, dedup=f"x{i}") for i in range(10)]),
    ]
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", fake_signals)

    r = client.get("/api/wa/anticipate/today?limit=2&min_score=0.0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 2


def test_limit_capped_at_10(client, monkeypatch):
    """Limit > 10 se capa a 10."""
    fake_signals = [
        ("a", lambda now: [_mk(0.9 - i * 0.01, dedup=f"x{i}") for i in range(20)]),
    ]
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", fake_signals)

    r = client.get("/api/wa/anticipate/today?limit=999&min_score=0.0")
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 10


def test_serializes_enrichment_fields(client, monkeypatch):
    """target_jid / target_name / draft / draft_meta llegan al shape UI."""
    cand = _mk(
        0.80,
        kind="anticipate-commitment",
        dedup="c1",
        target_jid="5491155556666@s.whatsapp.net",
        target_name="Mariana",
        draft="Hola Mari, te queria preguntar…",
        draft_meta={"confidence": 0.78, "style_snapshot_hash": "deadbeef"},
    )
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", [("c", lambda now: [cand])])

    r = client.get("/api/wa/anticipate/today")
    assert r.status_code == 200
    body = r.json()
    c = body["candidates"][0]
    assert c["target_jid"] == "5491155556666@s.whatsapp.net"
    assert c["target_name"] == "Mariana"
    assert c["draft"].startswith("Hola Mari")
    assert c["draft_meta"]["confidence"] == 0.78


def test_disabled_env_short_circuits(client, monkeypatch):
    """`RAG_ANTICIPATE_DISABLED=1` devuelve disabled+candidates vacío sin
    correr ninguna signal."""
    monkeypatch.setenv("RAG_ANTICIPATE_DISABLED", "1")
    called = []

    def spy_signal(now):
        called.append(1)
        return [_mk(0.99)]

    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", [("s", spy_signal)])

    r = client.get("/api/wa/anticipate/today")
    assert r.status_code == 200
    body = r.json()
    assert body["disabled"] is True
    assert body["candidates"] == []
    assert called == []  # signal NO se ejecutó


def test_signal_exception_does_not_break_endpoint(client, monkeypatch):
    """Si una signal tira excepción, las otras siguen y el endpoint NO 5xxs."""
    def broken(now):
        raise RuntimeError("signal exploded")

    fake_signals = [
        ("broken", broken),
        ("ok", lambda now: [_mk(0.7, dedup="ok1")]),
    ]
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", fake_signals)

    r = client.get("/api/wa/anticipate/today")
    assert r.status_code == 200
    body = r.json()
    # La signal que funcionó devolvió 1 candidate
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["dedup_key"] == "ok1"


def test_empty_when_no_signals_match(client, monkeypatch):
    """Sin candidates → response shape coherente con candidates vacío."""
    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", [])

    r = client.get("/api/wa/anticipate/today")
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"] == []
    assert body["ok"] is True
    assert body["disabled"] is False


def test_cache_avoids_rerunning_signals(client, monkeypatch):
    """Segunda llamada dentro del TTL no re-ejecuta signals (cache hit)."""
    call_count = [0]

    def signal(now):
        call_count[0] += 1
        return [_mk(0.7, dedup="x")]

    monkeypatch.setattr("rag._ANTICIPATE_SIGNALS", [("s", signal)])

    client.get("/api/wa/anticipate/today")
    assert call_count[0] == 1
    client.get("/api/wa/anticipate/today")
    assert call_count[0] == 1  # cache hit, no re-run
