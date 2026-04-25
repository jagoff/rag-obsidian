"""Tests for `POST /api/chat/upload-image` — el endpoint que recibe una
imagen subida desde el chat web, le aplica OCR + detector de cita, y
según confidence o auto-crea (≥0.85) el evento/reminder o devuelve
una proposal para confirmación manual.

Decisiones del user (2026-04-25):
- confidence ≥0.85 → auto-create con [↩ deshacer] en el frontend
- confidence <0.85 → card de confirmación, el user clickea [Crear]
- kind="note" o no detect → action="noop" (frontend muestra "no
  detecté nada agendable")

Mocks: el OCR (`_image_text_or_caption`) y el detector
(`_detect_cita_from_ocr`) son stubs para no depender de Apple Vision
ni de Ollama corriendo. Las funciones de creación
(`propose_calendar_event`, `propose_reminder`) también son
monkeypatched para no efectivamente crear eventos en Calendar.app /
Reminders.app durante los tests.
"""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


@pytest.fixture(autouse=True)
def _isolate_cita_detections_table(monkeypatch, tmp_path):
    """Aísla `_ragvec_state_conn` a una DB tmpfile por test.

    Sin esta fixture, los tests que ejercen el endpoint
    `/api/chat/upload-image` persistirían rows en
    `~/.local/share/obsidian-rag/ragvec/telemetry.db` (DB real del user)
    via `_persist_cita_detection`, contaminando entre runs y
    triggereando el dedup en tests subsiguientes que mockean el mismo
    OCR text. Aislándolo por test, cada uno arranca con la tabla vacía.
    """
    import sqlite3
    from contextlib import contextmanager

    db_path = tmp_path / "isolated_telemetry.db"

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

    # Crear schema mínimo de rag_cita_detections (DDL en rag/__init__.py:5437).
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
    """Devuelve un PNG mínimo válido (1x1 transparente) para uploads
    que no necesitan contenido real (los tests mockean el OCR)."""
    # PNG de 1x1 transparente, ~70 bytes. Suficiente para que FastAPI
    # acepte el multipart con content_type=image/png.
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "8900000000017352474200AECE1CE9000000044741D40000B18F0BFC61050000"
        "000970485973000000010000000174A0BFFC0000000C49444154789C636001"
        "000000000500010A2C0F740000000049454E44AE426082"
    )


# ── 1. Validaciones de input ────────────────────────────────────────────


def test_rejects_non_image_content_type():
    """Un upload con content_type='text/plain' debe fallar 400."""
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("noimage.txt", b"hola", "text/plain")},
    )
    assert resp.status_code == 400
    assert "solo imágenes" in resp.json()["detail"]


def test_rejects_empty_image():
    """Un PNG con 0 bytes → 400."""
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400


def test_rejects_oversize_image():
    """Imagen >12MB → 413."""
    big = b"\x00" * (13 * 1024 * 1024)
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("big.png", big, "image/png")},
    )
    assert resp.status_code == 413
    assert "muy grande" in resp.json()["detail"]


# ── 2. Flow noop (OCR vacío, kind=note, no detect) ──────────────────────


def test_noop_when_ocr_returns_empty(monkeypatch):
    """OCR vacío → action='noop' con reason='ocr_empty_or_short'."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption", lambda p: ("", ""),
    )
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert data["reason"] == "ocr_empty_or_short"


def test_noop_when_ocr_too_short(monkeypatch):
    """OCR <20 chars → 'ocr_empty_or_short'."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption", lambda p: ("hola", "ocr"),
    )
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["action"] == "noop"


def test_noop_when_detector_returns_none(monkeypatch):
    """Detector devuelve None (no detect / disabled) → 'no_cita_detected'."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("a" * 100 + " texto largo sin contenido agendable", "ocr"),
    )
    monkeypatch.setattr("rag.ocr._detect_cita_from_ocr", lambda t: None)
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert data["reason"] == "no_cita_detected"


def test_noop_when_kind_is_note(monkeypatch):
    """Detector clasifica como 'note' (info sin acción) → noop con
    reason='kind_not_actionable'. Sin esto, una receta o un meme con
    fecha en metadata podría llenar el calendar."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("texto largo de receta de cocina con instrucciones", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "note",
            "title": "Receta",
            "when": "",
            "location": "",
            "confidence": 0.9,
        },
    )
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert data["reason"] == "kind_not_actionable"
    assert data["kind"] == "note"


# ── 3. Auto-create path (confidence ≥0.85) ──────────────────────────────


def test_auto_creates_event_when_high_confidence(monkeypatch):
    """confidence=0.95 + kind=event → llama propose_calendar_event y
    devuelve action='created' con event_uid."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("TURNO DENTISTA Martes 5 de mayo 15hs Cabildo 4567", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "event",
            "title": "Turno dentista",
            "when": "martes 15hs",
            "location": "Cabildo 4567",
            "confidence": 0.95,
        },
    )
    propose_calls = []

    def _fake_propose_calendar(*, title, start, location=None, notes=None, **_):
        propose_calls.append({"title": title, "start": start, "location": location})
        return json.dumps({
            "kind": "event",
            "created": True,
            "event_uid": "FAKE-UID-123",
            "fields": {
                "title": title,
                "start_iso": "2026-04-28T15:00:00",
                "end_iso": "2026-04-28T16:00:00",
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
    assert data["action"] == "created"
    assert data["kind"] == "event"
    assert data["event_uid"] == "FAKE-UID-123"
    assert data["fields"]["title"] == "Turno dentista"
    assert data["fields"]["location"] == "Cabildo 4567"
    assert data["confidence"] == 0.95
    # Confirma que llamamos al creator con los datos correctos
    assert len(propose_calls) == 1
    assert propose_calls[0]["title"] == "Turno dentista"
    assert propose_calls[0]["start"] == "martes 15hs"


def test_auto_creates_reminder_when_high_confidence(monkeypatch):
    """confidence=0.92 + kind=reminder → llama propose_reminder y
    devuelve action='created' con reminder_id."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("Pagar luz antes del 15 de mayo", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "reminder",
            "title": "Pagar luz",
            "when": "antes del 15 de mayo",
            "location": "",
            "confidence": 0.92,
        },
    )

    def _fake_propose_reminder(*, title, when="", notes=None, **_):
        return json.dumps({
            "kind": "reminder",
            "created": True,
            "reminder_id": "FAKE-REM-456",
            "fields": {
                "title": title,
                "due_iso": "2026-05-15T09:00:00",
                "due_text": when,
                "list": None,
                "priority": None,
                "notes": notes,
                "recurrence": None,
            },
        })

    monkeypatch.setattr("rag.propose_reminder", _fake_propose_reminder)

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("rec.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "created"
    assert data["kind"] == "reminder"
    assert data["reminder_id"] == "FAKE-REM-456"
    assert data["fields"]["title"] == "Pagar luz"


# ── 4. Needs-confirmation path (confidence <0.85) ───────────────────────


def test_needs_confirmation_when_low_confidence_event(monkeypatch):
    """confidence=0.62 + kind=event → action='needs_confirmation', NO
    se crea, devuelve fields normalizados para que el frontend
    renderice una card de proposal estándar."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("Reunion algun dia con cliente", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "event",
            "title": "Reunión",
            "when": "algún día",
            "location": "",
            "confidence": 0.62,
        },
    )

    # Verificar que NO llamamos a propose_calendar_event en este path
    create_called = []
    monkeypatch.setattr(
        "rag.propose_calendar_event",
        lambda **kw: create_called.append(kw) or json.dumps({"created": False}),
    )

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("vago.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "needs_confirmation"
    assert data["kind"] == "event"
    assert data["confidence"] == 0.62
    assert data["fields"]["title"] == "Reunión"
    assert data["fields"]["start_text"] == "algún día"
    # NO se llamó al creator (confidence baja → no auto-crear)
    assert len(create_called) == 0


def test_needs_confirmation_when_low_confidence_reminder(monkeypatch):
    """confidence=0.55 + kind=reminder → needs_confirmation con due_text
    y due_iso (best-effort parsed)."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("comprar leche posiblemente la semana que viene", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "reminder",
            "title": "Comprar leche",
            "when": "la semana que viene",
            "location": "",
            "confidence": 0.55,
        },
    )
    create_called = []
    monkeypatch.setattr(
        "rag.propose_reminder",
        lambda **kw: create_called.append(kw) or json.dumps({"created": False}),
    )

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("leche.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "needs_confirmation"
    assert data["kind"] == "reminder"
    assert data["fields"]["title"] == "Comprar leche"
    assert data["fields"]["due_text"] == "la semana que viene"
    assert len(create_called) == 0


def test_needs_confirmation_when_high_confidence_but_unparseable(monkeypatch):
    """confidence=0.95 pero NL no parseable → propose_calendar_event
    devuelve `created: False` → fallback a needs_confirmation."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("Cumpleaños tía Marta hoy a las dieciochos hs", "ocr"),
    )
    monkeypatch.setattr(
        "rag.ocr._detect_cita_from_ocr",
        lambda t: {
            "kind": "event",
            "title": "Cumple tía Marta",
            "when": "dieciochos hs",  # garbage, no parsea
            "location": "",
            "confidence": 0.95,
        },
    )
    monkeypatch.setattr(
        "rag.propose_calendar_event",
        lambda **kw: json.dumps({
            "kind": "event",
            "created": False,
            "needs_clarification": True,
            "fields": {"title": kw.get("title"), "start_text": kw.get("start"), "start_iso": None},
        }),
    )

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("cumple.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "needs_confirmation"


# ── 5. Robustez: errores no rompen el endpoint ──────────────────────────


def test_ocr_exception_returns_noop(monkeypatch):
    """Si _image_text_or_caption tira excepción → noop con reason
    legible. NO debe propagar 500 al frontend (best-effort)."""
    def _bad_ocr(p):
        raise RuntimeError("ocrmac module crashed")
    monkeypatch.setattr("rag.ocr._image_text_or_caption", _bad_ocr)

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert "ocr_error" in data["reason"]


def test_detector_exception_returns_noop(monkeypatch):
    """Idem detector — si crashea, action=noop con reason='detector_error'."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("a" * 100 + " texto largo agendable", "ocr"),
    )
    def _bad_detect(t):
        raise RuntimeError("ollama down")
    monkeypatch.setattr("rag.ocr._detect_cita_from_ocr", _bad_detect)

    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("img.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "noop"
    assert "detector_error" in data["reason"]


# ── 6. Persistencia: imagen guardada en uploads dir ─────────────────────


def test_image_persists_to_uploads_dir(monkeypatch):
    """Después del upload, la imagen debe quedar en
    ~/.local/share/obsidian-rag/chat-uploads/<sha256>.png. Esto evita
    que tengamos que volver a transferir el binario si el detector se
    re-corre (cache mtime-based ya está en _ocr_image)."""
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption",
        lambda p: ("", ""),  # no importa qué devuelva, solo nos interesa el path
    )
    resp = _client.post(
        "/api/chat/upload-image",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "image_path" in data
    from pathlib import Path
    p = Path(data["image_path"])
    assert p.exists()
    assert p.parent.name == "chat-uploads"
    # Cleanup: borrar el archivo de test
    try:
        p.unlink()
    except Exception:
        pass


# ── Dedup de OCR (issue #6 del audit 2026-04-25) ─────────────────────────


def test_dedup_returns_noop_when_same_image_already_processed(monkeypatch):
    """Si el user sube la misma foto dos veces, la segunda debe
    devolver action='noop' con reason='already_processed_cita' y el
    event_uid original — sin re-llamar al detector ni crear duplicado.

    `_isolate_cita_detections_table` (autouse fixture) garantiza DB tmp
    vacía al inicio del test.
    """
    # Mock OCR + detector → texto idéntico, alta confidence, kind=event.
    OCR_TEXT = "TURNO DENTISTA " + "X" * 100  # > 20 chars (umbral OCR)
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption", lambda p: (OCR_TEXT, "ocr"),
    )

    detect_calls = []
    def _fake_detect(text):
        detect_calls.append(text)
        return {
            "kind": "event",
            "title": "Turno dentista",
            "when": "martes 15hs",
            "location": "Cabildo",
            "confidence": 0.95,
        }
    monkeypatch.setattr("rag.ocr._detect_cita_from_ocr", _fake_detect)

    # Mock propose_calendar_event → simulamos un create exitoso.
    monkeypatch.setattr(
        "rag.propose_calendar_event",
        lambda **kw: json.dumps({
            "kind": "event",
            "created": True,
            "event_uid": "FAKE-EV-DEDUP-1",
            "fields": {"title": kw.get("title")},
        }),
    )

    # 1ra subida: crea el evento.
    resp1 = _client.post(
        "/api/chat/upload-image",
        files={"file": ("turno.png", _png_bytes(), "image/png")},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["action"] == "created"
    assert data1["event_uid"] == "FAKE-EV-DEDUP-1"
    assert len(detect_calls) == 1, "1ra subida llama al detector"

    # 2da subida idéntica: dedup hit → noop con event_uid original.
    resp2 = _client.post(
        "/api/chat/upload-image",
        files={"file": ("turno.png", _png_bytes(), "image/png")},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["action"] == "noop"
    assert data2["reason"] == "already_processed_cita"
    assert data2["event_uid"] == "FAKE-EV-DEDUP-1"
    assert len(detect_calls) == 1, (
        "2da subida NO debe llamar al detector — dedup short-circuit"
    )


def test_dedup_does_not_short_circuit_ambiguous(monkeypatch):
    """Si el primer procesamiento dejó decision='ambiguous' (low confidence
    o NL no parseable), las subidas posteriores SÍ deben re-procesar.
    Razonamiento: el detector LLM es probabilístico — re-ofrecer la card
    de confirmación es lo correcto, no congelar el outcome ambiguo.
    """
    OCR_TEXT = "Reunion algun dia, sin fecha clara " + "Y" * 60
    monkeypatch.setattr(
        "rag.ocr._image_text_or_caption", lambda p: (OCR_TEXT, "ocr"),
    )

    # Detector con confidence baja → router decide needs_confirmation.
    detect_calls = []
    def _fake_detect(text):
        detect_calls.append(text)
        return {
            "kind": "event",
            "title": "Reunión",
            "when": "algún día",
            "location": "",
            "confidence": 0.50,  # < 0.85 autocreate threshold
        }
    monkeypatch.setattr("rag.ocr._detect_cita_from_ocr", _fake_detect)

    # 1ra subida → needs_confirmation, persiste 'ambiguous'.
    resp1 = _client.post(
        "/api/chat/upload-image",
        files={"file": ("vago.png", _png_bytes(), "image/png")},
    )
    assert resp1.json()["action"] == "needs_confirmation"

    # 2da subida → vuelve a procesar (no short-circuit), de nuevo
    # devuelve needs_confirmation. Detector llamado de nuevo.
    resp2 = _client.post(
        "/api/chat/upload-image",
        files={"file": ("vago.png", _png_bytes(), "image/png")},
    )
    assert resp2.json()["action"] == "needs_confirmation"
    assert len(detect_calls) == 2, (
        "ambiguous debe re-procesar — el LLM puede mejorar con más contexto"
    )
