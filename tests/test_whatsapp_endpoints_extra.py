"""Tests adicionales para `/api/whatsapp/scheduled` (GET) — audit
2026-04-25 R2-Tests #2.

El sprint anterior (Phase 4c) sólo cubría el happy path con
`status=pending` (`test_endpoint_list_scheduled_pending` en
`tests/test_wa_scheduled.py`). El audit detectó que el endpoint
maneja varias variantes que no tenían cobertura:

- `status=None` (sin query param) → list_scheduled con status=None,
  devuelve todos los estados.
- `status=""` (string vacío) → mismo path que None gracias al
  `(status or "").strip() else None` en el endpoint
  (`web/server.py:3279-3282`).
- `status=sent` / `status=failed` / etc. → se filtran correctamente.
- `limit` clamped a `[1, 1000]` por el `max(1, min(int(limit), 1000))`.
- list_scheduled levanta excepción → 500 con detail "error listando
  scheduled" (DB locked, schema corrupto, ValueError por status
  inválido en `_VALID_STATUSES`).

Patrón de mock: monkeypatch sobre `rag.wa_scheduled.list_scheduled`
con un fake que captura los kwargs (status, limit, ...) y devuelve
un payload sintético. El endpoint hace `from rag import wa_scheduled`
adentro del try y luego `wa_scheduled.list_scheduled(...)` — el
monkeypatch sobre el atributo del módulo se propaga porque la lookup
se resuelve al invocar.

Nota: NO mockeamos la DB porque list_scheduled está mockeado y nunca
llega a tocar `_ragvec_state_conn`. Si en el futuro un test de este
archivo termina ejerciendo el path real, hay que agregar la fixture
`isolated_state_db` de `test_wa_scheduled.py` para no contaminar
la telemetry.db real.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


# ── Helper común: mock que captura kwargs ───────────────────────────────────


def _make_capturing_fake(items: list[dict]):
    """Devuelve `(captured_dict, fake)` — el fake actualiza
    `captured_dict` con los kwargs que recibe y retorna `items`.
    list_scheduled tiene signatura kwarg-only así que el fake usa
    `**kw` para no acoplarse al orden.
    """
    captured: dict = {}

    def _fake(**kw):
        captured.update(kw)
        return items

    return captured, _fake


# ── Tests ───────────────────────────────────────────────────────────────────


def test_scheduled_no_status_returns_all_states(monkeypatch):
    """Sin query param `?status=`, el endpoint debe llamar a
    list_scheduled con `status=None` (no filtro) y devolver TODOS los
    estados (pending, sent, sent_late, cancelled, failed) tal como
    los retorne la capa de datos.

    Audit R2-Tests #2: el único test previo del endpoint sólo
    ejercía `?status=pending`; este path documenta el flujo "dame
    todo" que el dashboard usa cuando el user no filtra.
    """
    items = [
        {"id": 1, "status": "pending", "message_text": "msg1"},
        {"id": 2, "status": "sent", "message_text": "msg2"},
        {"id": 3, "status": "cancelled", "message_text": "msg3"},
        {"id": 4, "status": "failed", "message_text": "msg4"},
    ]
    captured, fake = _make_capturing_fake(items)
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 4
    statuses = {it["status"] for it in data["items"]}
    assert statuses == {"pending", "sent", "cancelled", "failed"}
    # El endpoint normaliza ausencia de status a None antes de delegar.
    assert captured["status"] is None
    # El limit default del endpoint es 200 (clamped a [1, 1000]).
    assert captured["limit"] == 200


def test_scheduled_status_all_alias_for_none(monkeypatch):
    """`?status=` (string vacío) es alias semántico de None — el
    endpoint usa `(status or "").strip() else None` para normalizar.
    Resultado: ambos caminos convergen en list_scheduled con
    status=None y devuelven TODOS los estados.

    Audit R2-Tests #2: testea explícitamente la rama de
    normalización a None vía string vacío. El literal "all" como
    valor NO es alias en el código actual — pasaría tal cual a
    list_scheduled y `_VALID_STATUSES` levantaría ValueError → 500
    (ese caso lo cubre `test_scheduled_returns_500_when_db_fails`
    cuando el fake hace raise ValueError).
    """
    items = [
        {"id": 1, "status": "pending"},
        {"id": 2, "status": "sent"},
        {"id": 3, "status": "sent_late"},
    ]
    captured, fake = _make_capturing_fake(items)
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled?status=")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    # `?status=` (vacío) llega como "" al endpoint; la normalización
    # `(status or "").strip() else None` lo convierte a None antes de
    # delegar — confirma que el alias "no filter" via empty string
    # no rompe.
    assert captured["status"] is None


def test_scheduled_status_sent_filters_correctly(monkeypatch):
    """`?status=sent` se propaga tal cual a list_scheduled (status="sent")
    y el endpoint devuelve sólo lo que la capa de datos haya filtrado.

    Audit R2-Tests #2: el sprint anterior sólo cubría `status=pending`;
    los demás valores válidos del enum (`sent`, `sent_late`, `failed`,
    `cancelled`, `processing`) no tenían smoke test. Cualquier
    regresión que rompa el passthrough del param se detecta acá.
    """
    items = [
        {"id": 10, "status": "sent", "message_text": "ok"},
        {"id": 11, "status": "sent", "message_text": "otro ok"},
    ]
    captured, fake = _make_capturing_fake(items)
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled?status=sent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert all(it["status"] == "sent" for it in data["items"])
    # El endpoint pasa el string sin normalizar (sólo strip), así que
    # "sent" llega como "sent" — el filtrado real lo hace list_scheduled.
    assert captured["status"] == "sent"


def test_scheduled_status_failed_filters_correctly(monkeypatch):
    """Idem `test_scheduled_status_sent_filters_correctly` pero para
    `?status=failed`. Cubre el otro estado terminal y asegura que el
    endpoint no hace ningún case-handling raro por valor.

    Audit R2-Tests #2.
    """
    items = [
        {"id": 99, "status": "failed",
         "last_error": "send_failed",
         "attempt_count": 5,
         "message_text": "feliz cumple"},
    ]
    captured, fake = _make_capturing_fake(items)
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled?status=failed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["items"][0]["status"] == "failed"
    assert data["items"][0]["attempt_count"] == 5
    assert captured["status"] == "failed"


def test_scheduled_limit_capped_at_1000(monkeypatch):
    """`?limit=99999` se clampa a 1000 antes de delegar — el endpoint
    hace `max(1, min(int(limit), 1000))` para evitar que un cliente
    malformado pida 1M de filas y haga estallar el JSON.

    Audit R2-Tests #2: el cap superior no estaba ejercitado.
    """
    captured, fake = _make_capturing_fake([])
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled?limit=99999")
    assert resp.status_code == 200
    # Cap superior = 1000.
    assert captured["limit"] == 1000


def test_scheduled_returns_500_when_db_fails(monkeypatch):
    """Cuando list_scheduled levanta una excepción (DB locked, schema
    corrupto, status inválido por `_VALID_STATUSES`, etc.), el endpoint
    debe traducir a HTTP 500 con detail que mencione "error listando
    scheduled". Sin esto, el frontend del dashboard recibiría una
    excepción no manejada y mostraría una página en blanco.

    Audit R2-Tests #2: el path de error 500 nunca tuvo cobertura.
    """
    def _boom(**_kw):
        raise RuntimeError("sqlite database is locked")

    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", _boom)

    resp = _client.get("/api/whatsapp/scheduled?status=pending")
    assert resp.status_code == 500
    assert "error listando scheduled" in resp.json()["detail"]


def test_scheduled_negative_limit_clamped_to_1(monkeypatch):
    """`?limit=-5` se clampa a 1 (no a 0, no a -5) — el endpoint hace
    `max(1, min(int(limit), 1000))`. Cualquier int <= 0 colapsa al
    piso de 1.

    Audit R2-Tests #2: el cap inferior tampoco estaba ejercitado, y
    es un edge case clásico (frontend bug que serializa -1 como
    "sin limit").
    """
    captured, fake = _make_capturing_fake([])
    monkeypatch.setattr("rag.wa_scheduled.list_scheduled", fake)

    resp = _client.get("/api/whatsapp/scheduled?limit=-5")
    assert resp.status_code == 200
    # Cap inferior = 1 (max con 1).
    assert captured["limit"] == 1
