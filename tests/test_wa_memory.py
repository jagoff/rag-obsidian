"""Tests para `GET /api/wa/memory/<jid>` — drawer 🧠 Recordar de wzp.

El endpoint resuelve nombre del contacto via `_wa_display_name`, corre
`multi_retrieve` para top notas del vault, fetcha últimos N msgs del
bridge, arma summary corto y devuelve todo en un payload.

Cache 300s por JID. Tests resetean el cache entre runs.
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


@pytest.fixture(autouse=True)
def _reset_cache():
    _web_server._WZP_MEMORY_CACHE.clear()
    yield
    _web_server._WZP_MEMORY_CACHE.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _stub_collect(jid, max_notes, max_wa, *, name="Mariana", notes=None, wa=None,
                  last_ts=None):
    notes = notes if notes is not None else [
        {"path": "01-Projects/X/nota.md", "title": "nota", "snippet": "menciona a Mariana",
         "score": 0.42, "mtime": 1746000000}
    ]
    wa = wa if wa is not None else [
        {"id": "m1", "ts": 1746163320, "from_me": False, "content": "Te paso el deck"}
    ]
    summary = f"{len(notes)} nota{'s' if len(notes) != 1 else ''} en el vault"
    return {
        "jid": jid,
        "name": name,
        "summary": summary,
        "notes": notes,
        "wa_recent": wa,
        "stats": {
            "last_wa_ts": last_ts or (wa[0]["ts"] if wa else None),
            "notes_count": len(notes),
            "wa_recent_count": len(wa),
        },
    }


def test_returns_payload_for_jid(client, monkeypatch):
    """Happy path: shape completo con notes + wa_recent + summary + stats."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append((jid, max_notes, max_wa))
        return _stub_collect(jid, max_notes, max_wa)
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r = client.get("/api/wa/memory/5491155556666@s.whatsapp.net")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["cached"] is False
    assert body["name"] == "Mariana"
    assert body["jid"] == "5491155556666@s.whatsapp.net"
    assert body["notes"][0]["title"] == "nota"
    assert body["wa_recent"][0]["content"].startswith("Te paso")
    assert body["stats"]["notes_count"] == 1
    assert calls == [("5491155556666@s.whatsapp.net", 5, 5)]


def test_rejects_invalid_jid(client):
    """JID sin @ → 400."""
    r = client.get("/api/wa/memory/notajid")
    assert r.status_code == 400


def test_respects_query_params(client, monkeypatch):
    """max_notes y max_wa se pasan al collector."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append((jid, max_notes, max_wa))
        return _stub_collect(jid, max_notes, max_wa, notes=[], wa=[])
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r = client.get("/api/wa/memory/x@y?max_notes=10&max_wa=7")
    assert r.status_code == 200
    assert calls == [("x@y", 10, 7)]


def test_cache_key_includes_limits(client, monkeypatch):
    """Mismo JID con límites distintos debe recomputar, no devolver un
    payload cacheado con menos notas/mensajes que los pedidos.
    """
    calls = []

    def fake_collect(jid, max_notes, max_wa):
        calls.append((jid, max_notes, max_wa))
        return _stub_collect(jid, max_notes, max_wa, notes=[], wa=[
            {"id": f"m{max_wa}", "ts": 1746163320, "from_me": False, "content": str(max_wa)}
        ])

    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r1 = client.get("/api/wa/memory/x@y?max_notes=1&max_wa=1")
    r2 = client.get("/api/wa/memory/x@y?max_notes=1&max_wa=7")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is False
    assert calls == [("x@y", 1, 1), ("x@y", 1, 7)]


def test_caps_query_params(client, monkeypatch):
    """max_notes capa 15, max_wa capa 20, ambos floor 1."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append((jid, max_notes, max_wa))
        return _stub_collect(jid, max_notes, max_wa, notes=[], wa=[])
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r = client.get("/api/wa/memory/x@y?max_notes=999&max_wa=999")
    assert r.status_code == 200
    assert calls[0][1] == 15
    assert calls[0][2] == 20

    calls.clear()
    # Otro JID para evitar el cache (keyed por JID).
    r = client.get("/api/wa/memory/p@q?max_notes=0&max_wa=0")
    assert r.status_code == 200
    assert calls[0][1] == 1
    assert calls[0][2] == 1


def test_cache_avoids_recomputing(client, monkeypatch):
    """Segunda llamada al mismo JID dentro del TTL no re-corre collect."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append((jid,))
        return _stub_collect(jid, max_notes, max_wa)
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r1 = client.get("/api/wa/memory/x@y")
    assert r1.json()["cached"] is False
    r2 = client.get("/api/wa/memory/x@y")
    assert r2.json()["cached"] is True
    assert len(calls) == 1


def test_cache_separate_per_jid(client, monkeypatch):
    """JIDs distintos no se pisan en el cache."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append(jid)
        return _stub_collect(jid, max_notes, max_wa, name=jid.split("@")[0])
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    client.get("/api/wa/memory/a@b")
    client.get("/api/wa/memory/c@d")
    assert calls == ["a@b", "c@d"]


def test_invalidate_drops_cache(client, monkeypatch):
    """`_wzp_memory_invalidate(jid)` saca el slot del cache."""
    calls = []
    def fake_collect(jid, max_notes, max_wa):
        calls.append(jid)
        return _stub_collect(jid, max_notes, max_wa)
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    client.get("/api/wa/memory/x@y")
    _web_server._wzp_memory_invalidate("x@y")
    r = client.get("/api/wa/memory/x@y")
    assert r.json()["cached"] is False
    assert len(calls) == 2


def test_empty_notes_does_not_break(client, monkeypatch):
    """Vault sin notas que mencionen al contacto → response shape OK."""
    def fake_collect(jid, max_notes, max_wa):
        return _stub_collect(jid, max_notes, max_wa, notes=[], wa=[])
    monkeypatch.setattr(_web_server, "_wzp_memory_collect", fake_collect)

    r = client.get("/api/wa/memory/x@y")
    assert r.status_code == 200
    body = r.json()
    assert body["notes"] == []
    assert body["wa_recent"] == []


def test_collect_converts_thread_messages(monkeypatch):
    """El collector real debe aceptar el shape de `fetch_thread_for_ui`:
    ISO ts + `is_from_me`. Antes intentaba `int(iso)` y dejaba WA vacío.
    """
    from rag.integrations.whatsapp import fetch as _wa_fetch

    monkeypatch.setattr(_wa_fetch, "_wa_display_name", lambda jid, name: "Mariana")
    monkeypatch.setattr(_web_server, "resolve_vault_paths", lambda _vault: [])
    monkeypatch.setattr(_wa_fetch, "fetch_thread_for_ui", lambda jid, limit, before_ts: {
        "messages": [
            {
                "id": "m1",
                "ts": "2026-05-11T11:00:00-03:00",
                "is_from_me": False,
                "content": "te paso el deck",
            },
            {
                "id": "m2",
                "ts": "2026-05-11T12:00:00-03:00",
                "is_from_me": True,
                "content": "gracias",
            },
        ]
    })

    data = _web_server._wzp_memory_collect("5491155556666@s.whatsapp.net", 5, 5)
    assert data["wa_recent"][0]["id"] == "m2"
    assert data["wa_recent"][0]["from_me"] is True
    assert isinstance(data["wa_recent"][0]["ts"], int)
    assert data["stats"]["wa_recent_count"] == 2
