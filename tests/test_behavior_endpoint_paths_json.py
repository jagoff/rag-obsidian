"""Tests for the /api/behavior `paths_json` field (Quick Win #2, 2026-04-29).

El listener WhatsApp postea un evento `query_response` después de cada
respuesta del RAG con los `paths` que citó. El endpoint persiste eso al
`extra_json` de `rag_behavior` para que `infer_corrective_paths_from_behavior`
los pueda recuperar como fallback cuando el feedback row no tiene
`paths_json` propio (228/270 = 84% de los feedbacks negativos en WA
estaban skipeando con `n_skip_no_paths` antes de este wiring).

Invariantes acá:
  1. POST con paths_json válido → 200, persiste en extra_json como lista.
  2. POST sin paths_json → flow normal (backwards compat con eventos web).
  3. POST con paths_json malformado (no JSON / no array / entries no string
     / cross-source URI scheme) → 422, NO guardar string roto.
  4. Read-back: el row insertado tiene paths_json en extra_json
     deserializable como `list[str]`.
  5. source="whatsapp" ahora es válido (antes solo "web").
  6. event="query_response" ahora es válido.

Uso `record_feedback`-style mock: parchamos `log_behavior_event` para
capturar el dict que el handler le pasaría al writer SQL, sin tocar
telemetry.db real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skip cleanly si web deps no están instalados (pure-rag test run).
_web_server = pytest.importorskip("web.server")
_fastapi = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


def _base_payload(**overrides) -> dict:
    payload = {
        "source": "whatsapp",
        "event": "query_response",
        "query": "cuánto debe Alex",
        "session": "wa:5491111111111@s.whatsapp.net",
    }
    payload.update(overrides)
    return payload


# ── Happy path ──────────────────────────────────────────────────────────────


def test_paths_json_valid_array_persists_to_extra(client):
    """Caso canónico: WA listener postea con paths_json string,
    el handler lo parsea y lo manda al writer como lista."""
    payload = _base_payload(
        paths_json=json.dumps(["02-Areas/Personal/alex.md", "01-Projects/macbook.md"])
    )
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    mock_log.assert_called_once()
    event = mock_log.call_args.args[0]
    assert event["source"] == "whatsapp"
    assert event["event"] == "query_response"
    assert event["session"] == "wa:5491111111111@s.whatsapp.net"
    # `paths_json` se inyecta como lista parseada (no como string crudo)
    # — el round-trip por `_map_behavior_row` + `_sql_serialise_row` la
    # JSON-encodea como parte del extra_json del row.
    assert event["paths_json"] == [
        "02-Areas/Personal/alex.md",
        "01-Projects/macbook.md",
    ]


def test_paths_json_absent_keeps_backwards_compat(client):
    """Sin paths_json el flow es idéntico a un POST web pre-Quick Win #2."""
    payload = _base_payload()  # no paths_json
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200
    event = mock_log.call_args.args[0]
    # No emitimos la key cuando vino ausente — evita poblar extra_json
    # con un null que después confunde al consumidor.
    assert "paths_json" not in event


def test_paths_json_empty_list_does_not_emit_field(client):
    """Lista vacía == sin señal — no la inyectamos al extra_json."""
    payload = _base_payload(paths_json=json.dumps([]))
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200
    event = mock_log.call_args.args[0]
    assert "paths_json" not in event


# ── Validación: malformed input ─────────────────────────────────────────────


def test_paths_json_invalid_json_returns_422(client):
    """No es JSON parseable → 422, NO se guarda nada."""
    payload = _base_payload(paths_json="not-json-at-all")
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 422
    mock_log.assert_not_called()


def test_paths_json_not_array_returns_422(client):
    """JSON válido pero no es array → 422."""
    payload = _base_payload(paths_json=json.dumps({"foo": "bar"}))
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 422
    mock_log.assert_not_called()


def test_paths_json_array_with_non_string_returns_422(client):
    """Entries no-string → 422."""
    payload = _base_payload(paths_json=json.dumps(["foo.md", 42]))
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 422
    mock_log.assert_not_called()


def test_paths_json_array_with_empty_string_returns_422(client):
    """Entries strings vacías no aportan señal y rompen lookups → 422."""
    payload = _base_payload(paths_json=json.dumps(["foo.md", ""]))
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 422
    mock_log.assert_not_called()


def test_paths_json_with_uri_scheme_returns_422(client):
    """calendar://, whatsapp://, gmail:// no son vault-relative — el
    `_feedback_augmented_cases` SQL no los puede consumir."""
    for bad in ("calendar://EVENT-abc", "whatsapp://chat/123", "gmail://inbox/xyz"):
        payload = _base_payload(paths_json=json.dumps(["foo.md", bad]))
        with patch.object(_web_server, "log_behavior_event") as mock_log:
            resp = client.post("/api/behavior", json=payload)
        assert resp.status_code == 422, f"{bad!r} should be rejected"
        mock_log.assert_not_called()


# ── Source / event expansion ────────────────────────────────────────────────


def test_source_whatsapp_now_accepted(client):
    """Pre-Quick Win #2 el endpoint solo aceptaba source='web'.
    Ahora también 'whatsapp'."""
    payload = _base_payload(source="whatsapp", event="query_response")
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200
    mock_log.assert_called_once()


def test_source_unknown_still_rejected(client):
    """Cualquier source fuera del set conocido se rechaza con 400 —
    así un cliente nuevo se entera por el error y no por silent drop."""
    payload = _base_payload(source="cli")
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 400
    mock_log.assert_not_called()


def test_event_query_response_now_accepted(client):
    """`query_response` es el event nuevo que mete el listener WA."""
    payload = _base_payload(
        event="query_response",
        paths_json=json.dumps(["foo.md"]),
    )
    with patch.object(_web_server, "log_behavior_event") as mock_log:
        resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200
    event = mock_log.call_args.args[0]
    assert event["event"] == "query_response"


# ── Read-back end-to-end (sin mock, con SQL real in-memory) ────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry DB en tmp_path. Mismo patrón que
    `tests/test_brief_feedback_endpoint.py::state_db`.
    `_ragvec_state_conn` crea las tablas on-demand al primer uso.
    """
    import rag as _rag

    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_rag, "DB_PATH", db_path)
    _rag.SqliteVecClient(path=str(db_path))
    with _rag._ragvec_state_conn() as _conn:
        pass
    return db_path


def test_paths_json_round_trips_through_extra_json(state_db, client):
    """Smoke: con DB real (no mock), el row en `rag_behavior` queda con
    `extra_json` que contiene `paths_json` deserializable como list[str].

    Este test ejercita la ruta completa: handler → log_behavior_event →
    _map_behavior_row → _sql_serialise_row → SQLite. Si en algún punto
    se rompe el contract (ej. alguien serialea como string anidado), el
    `_recover_paths_from_behavior` del implicit-feedback inference no va
    a poder recuperar los paths.
    """
    payload = _base_payload(
        paths_json=json.dumps(["02-Areas/X/uno.md", "01-Projects/dos.md"])
    )
    resp = client.post("/api/behavior", json=payload)
    assert resp.status_code == 200, resp.text

    # Leer el row de vuelta del SQL y verificar el extra_json.
    import sqlite3
    import rag as _rag

    conn = sqlite3.connect(str(state_db / _rag._TELEMETRY_DB_FILENAME))
    try:
        row = conn.execute(
            "SELECT extra_json FROM rag_behavior "
            "WHERE event = 'query_response' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "ningún row insertado para query_response"
    extra = json.loads(row[0])
    assert "paths_json" in extra, f"extra_json sin paths_json: {extra}"
    # `paths_json` adentro del extra_json es una lista nativa, no un
    # string anidado. Contract clave para `_recover_paths_from_behavior`.
    assert isinstance(extra["paths_json"], list)
    assert extra["paths_json"] == [
        "02-Areas/X/uno.md",
        "01-Projects/dos.md",
    ]
