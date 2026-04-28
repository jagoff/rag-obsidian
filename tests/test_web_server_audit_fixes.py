"""Tests para los fixes del audit 2026-04-25 sobre `web/server.py`.

Cubre cuatro findings cerrados en el mismo commit:

- **R1 #9 + iter** — `_OLLAMA_TOOL_TIMEOUT` y `_OLLAMA_STREAM_TIMEOUT`
  alineados a 90s. R1 #9 original bajó tool 120 → 45s; eval autónomo
  del 2026-04-28 lo subió 45 → 90s al ver que la 2da ronda timeouteaba
  con outputs grandes; repro Playwright el mismo día subió stream 45 →
  90s al ver el mismo problema en la synthesis call post-tools.
- **R2-Performance #1** — `_CHAT_BUCKETS` / `_BEHAVIOR_BUCKETS` ahora
  son `_LRURateBucket` con cap 5000 IPs (antes `defaultdict(deque)`
  crecía sin bound y permitía memory exhaustion bajo rotación de
  proxies).
- **R2-Performance #3** — `/api/dashboard/stream` ahora limita streams
  concurrentes por IP (default 3, env `RAG_SSE_MAX_PER_IP`) para evitar
  N tabs del dashboard acumulando polls SQLite cada 1.5s.
- **R2-OCR #2** — `/api/chat/upload-image` baja el confidence a 0.5 si
  la fecha detectada está más vieja que `RAG_OCR_HISTORICAL_MAX_DAYS`
  (default 30) aunque el detector estuviera muy seguro: foto vieja de
  un ticket no debería crear eventos retroactivos.

Estilo: cada test arma un escenario chico y verifica el invariante
post-fix. Los mocks tocan solo lo imprescindible (OCR, detector,
propose_*) para no depender de Ollama / Apple Vision corriendo.
"""
from __future__ import annotations

import collections
import json
import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


# ── Fixture común: aislar la DB de telemetría ──────────────────────────
# Mismo patrón que `tests/test_chat_image_upload.py` — el endpoint
# `/api/chat/upload-image` toca `rag_cita_detections` via
# `_persist_cita_detection`. Sin aislar, el row del test contamina la
# DB real del user en `~/.local/share/obsidian-rag/ragvec/`.
@pytest.fixture
def _isolated_telemetry(monkeypatch, tmp_path):
    import sqlite3

    db_path = tmp_path / "telemetry.db"

    @contextmanager
    def _fake_conn():
        c = sqlite3.connect(
            str(db_path),
            isolation_level=None,
            check_same_thread=False,
            timeout=30.0,
        )
        try:
            yield c
        finally:
            c.close()

    _CITA_DDL = (
        "CREATE TABLE IF NOT EXISTS rag_cita_detections ("
        " ocr_hash TEXT PRIMARY KEY, image_path TEXT, source TEXT NOT NULL,"
        " decision TEXT NOT NULL, kind TEXT, title TEXT, start_text TEXT,"
        " location TEXT, confidence REAL, event_uid TEXT, reminder_id TEXT,"
        " created_at REAL NOT NULL"
        ")"
    )
    with _fake_conn() as c:
        c.execute(_CITA_DDL)

    monkeypatch.setattr("rag._ragvec_state_conn", _fake_conn)
    yield db_path


def _png_bytes() -> bytes:
    """PNG mínimo válido (1x1) — los tests mockean el OCR así que el
    contenido real no importa, solo necesitamos que FastAPI acepte
    el multipart con content_type=image/png."""
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "8900000000017352474200AECE1CE9000000044741D40000B18F0BFC61050000"
        "000970485973000000010000000174A0BFFC0000000C49444154789C636001"
        "000000000500010A2C0F740000000049454E44AE426082"
    )


# ══════════════════════════════════════════════════════════════════════
# Fix 1 — Ollama client budgets alineados (Tool + Stream = 90s)
#
# Historia:
#   - Audit 2026-04-25 R1 #9: _OLLAMA_TOOL_TIMEOUT bajó 120 → 45s.
#   - Eval autónomo 2026-04-28 (commit b0d140e): subió 45 → 90s tras ver
#     que la 2da ronda (synthesis post-tool-output) consistentemente
#     timeouteaba en 60-80s.
#   - Repro autónomo Playwright 2026-04-28: 3 de 5 queries fallaron con
#     "LLM falló: timed out" en 59-62s — esta vez por el `_OLLAMA_STREAM_
#     TIMEOUT=45s` (que el commit anterior NO había subido). El stream
#     final post-tools comparte el mismo budget conceptual que el tool
#     decision call (cold-load del modelo + prefill + decode), así que
#     debe tener el mismo timeout para no colgarse antes que el otro.
#
# Invariante actual: ambos timeouts == 90s. Si alguien quiere cambiar
# uno, cambiar el otro y ajustar este test, o pensar bien por qué
# divergir.
# ══════════════════════════════════════════════════════════════════════


def test_ollama_tool_timeout_is_90_seconds():
    """El tool-decision client necesita budget de ~90s: la 2da ronda con
    `tools=` schema + tool outputs grandes (whatsapp_search, gmail_recent,
    drive_search) puede tardar 60-80s en MPS warm. Si bajamos a 45s
    volvemos a ver `LLM falló: timed out` en queries multi-tool."""
    assert _server._OLLAMA_TOOL_TIMEOUT == 90.0, (
        f"Esperado _OLLAMA_TOOL_TIMEOUT=90.0 (eval autónomo 2026-04-28, "
        f"commit b0d140e), got {_server._OLLAMA_TOOL_TIMEOUT}"
    )


def test_ollama_stream_timeout_aligned_with_tool_timeout():
    """El stream-final client (synthesis post-tools) puede tardar tanto
    como el tool-decision call: prefill sobre 25-30k chars de contexto +
    decode de la respuesta + cold-load eventual cuando num_ctx adaptive
    cambia respecto del loaded value. Pre-fix (2026-04-28) el stream
    estaba en 45s y disparaba timeouts en ~60s wall time mientras el
    tool client tenía 90s y holgaba — divergencia que invalidaba el
    sentido del fix anterior."""
    assert _server._OLLAMA_STREAM_TIMEOUT == 90.0, (
        f"Esperado _OLLAMA_STREAM_TIMEOUT=90.0 alineado con _OLLAMA_TOOL_"
        f"TIMEOUT (repro Playwright 2026-04-28), "
        f"got {_server._OLLAMA_STREAM_TIMEOUT}"
    )
    # Belt-and-suspenders: si alguien sube uno de los dos sin actualizar
    # el otro, fallamos antes de que llegue al server.
    assert _server._OLLAMA_STREAM_TIMEOUT == _server._OLLAMA_TOOL_TIMEOUT, (
        f"Invariant: stream y tool timeouts deben coincidir. "
        f"stream={_server._OLLAMA_STREAM_TIMEOUT} "
        f"tool={_server._OLLAMA_TOOL_TIMEOUT}"
    )


# ══════════════════════════════════════════════════════════════════════
# Fix 2 — Audit R2-Performance #1: _LRURateBucket con cap 5000 IPs
# ══════════════════════════════════════════════════════════════════════


def test_lru_rate_bucket_evicts_oldest_when_over_max():
    """Insertar 5001 IPs distintas en un bucket con cap 5000 deja
    exactamente 5000 entries, y la primera IP (LRU) ya no está."""
    bucket = _server._LRURateBucket(max_size=5000)
    for i in range(5001):
        # Acceso vía __getitem__ + append, igual que `_check_rate_limit`.
        bucket[f"ip-{i}"].append(123.0)

    assert len(bucket) == 5000
    # ip-0 fue la primera y nunca se accedió de nuevo → debería haber
    # sido eviccionada cuando entró ip-5000.
    assert "ip-0" not in bucket
    # ip-5000 (la última) sí debe estar.
    assert "ip-5000" in bucket


def test_lru_rate_bucket_move_to_end_on_access():
    """Acceder a una key vieja la marca como recently-used: queda
    protegida de la próxima evicción aunque se hayan agregado muchas
    keys nuevas en el medio."""
    bucket = _server._LRURateBucket(max_size=3)
    bucket["a"].append(1.0)
    bucket["b"].append(1.0)
    bucket["c"].append(1.0)

    # Re-acceder a "a" la mueve al final (most recently used).
    _ = bucket["a"]

    # Insertar "d" debería eviccionar "b" (el LRU ahora), no "a".
    bucket["d"].append(1.0)

    assert "a" in bucket
    assert "b" not in bucket
    assert "c" in bucket
    assert "d" in bucket


def test_lru_rate_bucket_returns_deque():
    """`bucket[ip]` siempre devuelve un `collections.deque` para que
    `_check_rate_limit` pueda hacer `popleft()` O(1)."""
    bucket = _server._LRURateBucket()
    events = bucket["test-ip"]
    assert isinstance(events, collections.deque), (
        f"_LRURateBucket[ip] debería devolver deque, "
        f"got {type(events).__name__}"
    )


def test_lru_rate_bucket_setitem_wraps_iterables():
    """Asignar una list (caso legacy de tests viejos como
    test_web_behavior.py:217) tiene que envolverla en deque para
    preservar la API que espera `_check_rate_limit`."""
    bucket = _server._LRURateBucket()
    bucket["ip-1"] = [1.0, 2.0, 3.0]
    assert isinstance(bucket["ip-1"], collections.deque)
    assert list(bucket["ip-1"]) == [1.0, 2.0, 3.0]


def test_lru_rate_bucket_clear_works_for_tests():
    """`.clear()` tiene que vaciar el bucket — los tests existentes
    (`test_web_chat_tools.py`, `test_drive_search_tool.py`, etc.)
    hacen `_CHAT_BUCKETS.clear()` antes de cada test."""
    bucket = _server._LRURateBucket()
    bucket["a"].append(1.0)
    bucket["b"].append(2.0)
    assert len(bucket) == 2
    bucket.clear()
    assert len(bucket) == 0
    assert "a" not in bucket


# ══════════════════════════════════════════════════════════════════════
# Fix 3 — Audit R2-Performance #3: /api/dashboard/stream cap por IP
# ══════════════════════════════════════════════════════════════════════


def test_sse_acquire_release_slot():
    """Smoke test del helper: un IP puede tener hasta `_SSE_MAX_PER_IP`
    slots; el siguiente acquire devuelve False; tras un release vuelve
    a haber un slot disponible."""
    # Aislamos el contador para no chocar con otros tests.
    _server._SSE_CONNECTIONS_PER_IP.clear()

    ip = "1.2.3.4"
    cap = _server._SSE_MAX_PER_IP
    for _ in range(cap):
        assert _server._sse_acquire_slot(ip) is True

    # El (cap+1)-ésimo acquire tiene que rechazarse.
    assert _server._sse_acquire_slot(ip) is False

    # Liberar uno y reintentar — ahora sí entra.
    _server._sse_release_slot(ip)
    assert _server._sse_acquire_slot(ip) is True

    # Cleanup: liberar todos los slots para no contaminar otros tests.
    for _ in range(cap):
        _server._sse_release_slot(ip)
    assert ip not in _server._SSE_CONNECTIONS_PER_IP


def test_sse_endpoint_returns_429_when_over_per_ip_cap(monkeypatch):
    """4ta conexión SSE desde la misma IP recibe 429. Pre-llenamos el
    contador hasta el cap y verificamos que el siguiente GET rebote
    sin abrir el generator (evita el side-effect de quedar pegado en
    el `while True` del SSE)."""
    # Usamos el cap configurado del módulo (default 3) para que el test
    # siga siendo válido si cambia el default.
    cap = _server._SSE_MAX_PER_IP

    # Cliente fijo (TestClient siempre reporta IP "testclient").
    test_ip = "testclient"

    # Reset global counter + pre-fill hasta el cap para simular N
    # streams ya activos.
    _server._SSE_CONNECTIONS_PER_IP.clear()
    _server._SSE_CONNECTIONS_PER_IP[test_ip] = cap

    try:
        resp = _client.get("/api/dashboard/stream")
        assert resp.status_code == 429, (
            f"Esperado 429 con {cap} streams ya abiertos para {test_ip!r}, "
            f"got {resp.status_code}: {resp.text[:200]}"
        )
        assert "too many concurrent streams" in resp.json()["detail"]
    finally:
        _server._SSE_CONNECTIONS_PER_IP.clear()


# ══════════════════════════════════════════════════════════════════════
# Fix 4 — Audit R2-OCR #2: downgrade de confidence si fecha histórica
# ══════════════════════════════════════════════════════════════════════


def test_upload_image_downgrades_when_date_is_historical(
    monkeypatch, _isolated_telemetry,
):
    """Foto con confidence=0.95 PERO fecha de hace 2 años → endpoint
    NO debe devolver action='created'; debe caer al path de
    needs_confirmation aunque el detector estaba muy seguro.

    Caso real del audit: foto de un ticket de cine viejo → "Cine 15/06
    20:00" → auto-creaba un evento retroactivo en el calendar.
    """
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("CINE Viernes 15/06/2024 20:00 hs Hoyts Abasto", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "event",
            "title": "Cine",
            # Fecha explícita en el pasado (2024) — el parser
            # `_parse_natural_datetime` la resuelve a un datetime de
            # ~hace 2 años, lejos del threshold de 30 días.
            "when": "2024-06-15 20:00",
            "location": "Hoyts Abasto",
            "confidence": 0.95,
        },
    )

    # Si el bug estuviera vivo, el endpoint llamaría
    # propose_calendar_event. Lo monkeypatcheamos para que falle el
    # test si lo invoca (sentinela explícito).
    propose_calls = []

    def _fail_if_called(*args, **kwargs):
        propose_calls.append((args, kwargs))
        return json.dumps({"created": True, "event_uid": "BUG-UID"})

    monkeypatch.setattr("rag.propose_calendar_event", _fail_if_called)

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("ticket-viejo.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["action"] == "needs_confirmation", (
        f"Esperado needs_confirmation por fecha histórica, "
        f"got action={data.get('action')!r} (data={data})"
    )
    assert data["action"] != "created"
    # El downgrade reescribe el confidence a 0.5.
    assert data["confidence"] == 0.5
    # propose_calendar_event NO debería haber sido llamado.
    assert propose_calls == [], (
        "El endpoint llamó a propose_calendar_event aunque la fecha "
        "estaba >30 días en el pasado — el guardrail de R2-OCR #2 no "
        "se aplicó."
    )


def test_upload_image_creates_when_date_is_recent_and_confident(
    monkeypatch, _isolated_telemetry,
):
    """Sanity check: confidence=0.95 + fecha en el FUTURO próximo →
    auto-create funciona como antes. El guardrail de R2-OCR #2 NO
    debe romper el flujo normal de eventos agendables."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("TURNO DENTISTA mañana 15hs Cabildo 4567", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "event",
            "title": "Turno dentista",
            # "mañana 15hs" → `_parse_natural_datetime` lo resuelve a
            # mañana relativo a now() → muy lejos de "histórico".
            "when": "mañana 15hs",
            "location": "Cabildo 4567",
            "confidence": 0.95,
        },
    )

    def _fake_propose_calendar(*, title, start, location=None, notes=None, **_):
        return json.dumps({
            "kind": "event",
            "created": True,
            "event_uid": "OK-UID-456",
            "fields": {
                "title": title,
                "start_iso": "2099-04-28T15:00:00",
                "end_iso": "2099-04-28T16:00:00",
                "location": location,
                "notes": notes,
                "all_day": False,
                "calendar": None,
                "recurrence": None,
            },
        })

    monkeypatch.setattr("rag.propose_calendar_event", _fake_propose_calendar)

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("turno.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Fecha cercana → no se gatilla el downgrade → action=created.
    assert data["action"] == "created", (
        f"Sanity check rota: fecha de mañana debería auto-crearse, "
        f"got action={data.get('action')!r}"
    )
    assert data["confidence"] == 0.95


def test_historical_days_constant_default_30():
    """El default debe ser 30 días — documentado en el comentario y
    es el valor consensuado con el user 2026-04-25."""
    # No leemos del módulo directo porque la lectura de env happen al
    # import; chequeamos que el default constant es 30.
    assert _server._CHAT_UPLOAD_HISTORICAL_DAYS_DEFAULT == 30
