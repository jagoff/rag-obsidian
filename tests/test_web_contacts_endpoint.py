"""Tests para `GET /api/contacts` — endpoint que alimenta el popover de
`/wzp` y `/mail` en el web chat (2026-04-24).

Mock del cache de contactos en memoria; el endpoint solo es un wrapper
fino sobre `_fuzzy_filter_contacts` con validación de query params.
"""
from __future__ import annotations

import pytest

import rag
from fastapi.testclient import TestClient
import web.server as _server


_client = TestClient(_server.app)


@pytest.fixture
def fake_contacts(monkeypatch):
    """Instala un set fijo de contactos para que el endpoint los devuelva
    determinísticamente, sin tocar Apple Contacts ni el cache de disco.

    El cache v2 chequea ``schema == _CONTACTS_CACHE_SCHEMA`` antes de
    aceptar el tier-1 in-memory (sino cae al disco / osascript real),
    así que el dict debe incluir esa key.
    """
    contacts = [
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": ["grecia@x.com"]},
        {"name": "Gregorio López",  "phones": ["+5491166666666"], "emails": []},
        {"name": "Ágatha García",   "phones": [], "emails": ["agatha@y.com"]},
        {"name": "Mario Pérez",     "phones": ["+5491177777777"], "emails": ["mario@z.com"]},
    ]
    rag._contacts_phone_index = {
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }
    return contacts


def test_endpoint_returns_matching_contacts(fake_contacts):
    resp = _client.get("/api/contacts?q=Gre")
    assert resp.status_code == 200
    data = resp.json()
    names = [c["name"] for c in data["contacts"]]
    assert "Grecia Ferrari" in names
    assert "Gregorio López" in names
    assert data["query"] == "Gre"
    assert data["kind"] == "any"


def test_endpoint_kind_phone_filters_emails_only_contacts(fake_contacts):
    resp = _client.get("/api/contacts?q=&kind=phone&limit=50")
    names = [c["name"] for c in resp.json()["contacts"]]
    # Ágatha tiene email pero no teléfono — excluida del kind=phone.
    assert "Ágatha García" not in names
    assert "Grecia Ferrari" in names


def test_endpoint_kind_email_filters_phone_only_contacts(fake_contacts):
    resp = _client.get("/api/contacts?q=&kind=email&limit=50")
    names = [c["name"] for c in resp.json()["contacts"]]
    # Gregorio tiene teléfono pero no email — excluido del kind=email.
    assert "Gregorio López" not in names
    assert "Ágatha García" in names


def test_endpoint_invalid_kind_falls_back_to_any(fake_contacts):
    resp = _client.get("/api/contacts?q=&kind=potato&limit=50")
    assert resp.status_code == 200
    assert resp.json()["kind"] == "any"


def test_endpoint_empty_query_returns_alphabetical(fake_contacts):
    resp = _client.get("/api/contacts?q=&limit=2")
    data = resp.json()
    assert len(data["contacts"]) == 2
    # _load_contacts_list ordena por `name.lower()` — los nombres de fake_contacts
    # son los únicos en el cache; chequeamos que vinieron del set.
    fake_names = {c["name"] for c in fake_contacts}
    for c in data["contacts"]:
        assert c["name"] in fake_names


def test_endpoint_limit_clamped_min(fake_contacts):
    """`limit < 1` se clampa a 1 en el endpoint para evitar respuestas
    vacías sin sentido."""
    resp_zero = _client.get("/api/contacts?q=&limit=0")
    assert resp_zero.status_code == 200
    # El clamp del endpoint manda limit=1 a `_fuzzy_filter_contacts`.
    assert len(resp_zero.json()["contacts"]) == 1


def test_endpoint_limit_clamped_max(fake_contacts):
    """`limit > 100` se clampa a 100 en el endpoint."""
    resp_huge = _client.get("/api/contacts?q=&limit=10000")
    assert resp_huge.status_code == 200
    # Solo hay 4 fakes — el clamp no se nota con datos chicos, pero el
    # endpoint no debe explotar.
    assert len(resp_huge.json()["contacts"]) == 4


def test_endpoint_no_matches_returns_empty(fake_contacts):
    resp = _client.get("/api/contacts?q=zzzzz")
    assert resp.status_code == 200
    assert resp.json()["contacts"] == []


def test_endpoint_silent_fail_on_internal_error(monkeypatch):
    """Si `_fuzzy_filter_contacts` raisea (ej. cache corrupto), devolvemos
    200 con lista vacía — el popover debe ser silent-fail, no romper el
    chat. El error se logguea via `_silent_log` para debugging."""
    def _boom(*a, **kw):
        raise RuntimeError("simulated cache corruption")
    monkeypatch.setattr(rag, "_fuzzy_filter_contacts", _boom)
    captured_logs = []
    monkeypatch.setattr(rag, "_silent_log", lambda *a, **kw: captured_logs.append(a))

    resp = _client.get("/api/contacts?q=foo")
    assert resp.status_code == 200
    assert resp.json()["contacts"] == []
    # Debug breadcrumb del silent fail.
    assert any("api_contacts" in str(a) for a in captured_logs)


def test_endpoint_handles_empty_cache(monkeypatch):
    """Si Apple Contacts está vacío / osascript no respondió, devolvemos
    lista vacía sin error.

    El loader interno se llama `_load_contacts_full` o `_load_contacts_list`
    según iteración del peer whf6iuag — patcheamos cualquiera que esté
    presente para evitar bind-by-name churn."""
    rag._contacts_phone_index = None
    for name in ("_load_contacts_full", "_load_contacts_list"):
        if hasattr(rag, name):
            monkeypatch.setattr(rag, name, lambda ttl_s=86400: [])
    resp = _client.get("/api/contacts?q=anything")
    assert resp.status_code == 200
    assert resp.json()["contacts"] == []


def test_endpoint_returns_phones_and_emails_arrays(fake_contacts):
    """Sanity check del shape de cada contact retornado — la UI espera
    `phones` y `emails` como arrays para mostrar el primer elemento como
    hint en el popover."""
    resp = _client.get("/api/contacts?q=Grecia")
    data = resp.json()
    assert data["contacts"]
    c = data["contacts"][0]
    assert "name" in c
    assert "phones" in c and isinstance(c["phones"], list)
    assert "emails" in c and isinstance(c["emails"], list)
    assert "score" in c
