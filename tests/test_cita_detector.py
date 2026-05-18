"""Tests del detector OCR → event / reminder / note.

El detector clasifica texto OCR de una imagen en 3 kinds:
  - `event`: cita / turno / reunión → `propose_calendar_event`
  - `reminder`: tarea / compra / factura → `propose_reminder` (con path
    de la imagen en el `notes` del reminder, ya que Apple Reminders no
    soporta attachments vía AppleScript)
  - `note`: info sin acción → no-op

Cobertura:
  1. Env gate `RAG_CITA_DETECT` (on/off).
  2. `_normalize_ocr_for_hash` + `_ocr_hash_key` — whitespace / case
     insensitive, determinismo, discriminación de contenidos distintos.
  3. `_detect_cita_from_ocr` — helper con mocks: happy path event /
     reminder / note, malformed JSON, exception, bad shape, clamp de
     confidence, normalización de tipos, backward-compat con schema viejo
     `{is_cita, start}` (los LLMs/tests antiguos siguen funcionando).
  4. `_maybe_create_cita_from_ocr` — dedup sidecar, low-confidence
     persistencia, ambiguous (event sin fecha), event happy path,
     reminder happy path (con fecha y sin fecha), note no-op, duplicate
     path, error silent-fail, confidence override.
  5. Integración con `_enrich_body_with_ocr` — body enrichment no se
     rompe cuando el detector fire.

Sin red ni ollama real: todos los LLM + osascript calls están mockeados.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rag


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def _clean_state(monkeypatch, tmp_path):
    """DB temporal limpia para que cada test tenga su propio
    `rag_cita_detections` + `rag_ocr_cache` sin cross-pollination."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    # Reset env por las dudas (tests previos pueden haberlo seteado).
    monkeypatch.delenv("RAG_CITA_DETECT", raising=False)
    yield


class _FakeHelperResponse:
    """Stand-in del retorno de `_helper_client().chat(...)` — solo
    necesita `.message.content`."""
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


def _mock_helper_client(monkeypatch, *, content: str | None = None, exc: Exception | None = None):
    """Reemplaza `rag._helper_client()` por un stub que devuelve `content`
    o lanza `exc`. Retorna la lista de calls para inspección."""
    calls: list[dict] = []

    class _Stub:
        def chat(self, **kwargs):
            calls.append(kwargs)
            if exc is not None:
                raise exc
            return _FakeHelperResponse(content or "")

    stub = _Stub()
    monkeypatch.setattr(rag, "_helper_client", lambda: stub)
    return calls


def _stub_propose_calendar_event(monkeypatch, result: dict):
    import json as _json
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return _json.dumps(result, ensure_ascii=False)

    monkeypatch.setattr(rag, "propose_calendar_event", _fake)
    return calls


def _stub_propose_reminder(monkeypatch, result: dict):
    import json as _json
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return _json.dumps(result, ensure_ascii=False)

    monkeypatch.setattr(rag, "propose_reminder", _fake)
    return calls


# ── _cita_detect_enabled ───────────────────────────────────────────────────


def test_cita_detect_enabled_default_on(monkeypatch):
    """Sin env var → ON."""
    monkeypatch.delenv("RAG_CITA_DETECT", raising=False)
    assert rag._cita_detect_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "FALSE", "No"])
def test_cita_detect_disabled_by_env(monkeypatch, val):
    monkeypatch.setenv("RAG_CITA_DETECT", val)
    assert rag._cita_detect_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", ""])
def test_cita_detect_enabled_for_permissive_values(monkeypatch, val):
    """Cualquier cosa que NO sea 0/false/no → ON (permissive)."""
    monkeypatch.setenv("RAG_CITA_DETECT", val)
    assert rag._cita_detect_enabled() is True


# ── _normalize_ocr_for_hash / _ocr_hash_key ────────────────────────────────


def test_normalize_lowercase_and_collapses_whitespace():
    assert rag._normalize_ocr_for_hash("  TURNO\n\nDr.   García\t10hs  ") == (
        "turno dr. garcía 10hs"
    )


def test_normalize_empty_string_returns_empty():
    assert rag._normalize_ocr_for_hash("") == ""
    assert rag._normalize_ocr_for_hash(None) == ""  # type: ignore[arg-type]


def test_ocr_hash_key_deterministic():
    a = rag._ocr_hash_key("Turno dentista mañana 10hs")
    b = rag._ocr_hash_key("Turno dentista mañana 10hs")
    assert a == b
    assert len(a) == 16  # sha256 truncated to 16 hex chars


def test_ocr_hash_key_whitespace_insensitive():
    """Dos runs de OCR sobre la misma imagen a veces difieren en
    espacios/saltos. El hash debe colapsar esas variaciones."""
    a = rag._ocr_hash_key("Turno dentista\n  mañana 10hs")
    b = rag._ocr_hash_key("TURNO DENTISTA mañana   10hs")
    assert a == b


def test_ocr_hash_key_distinguishes_content():
    a = rag._ocr_hash_key("Turno dentista")
    b = rag._ocr_hash_key("Turno peluquero")
    assert a != b


# ── _detect_cita_from_ocr ──────────────────────────────────────────────────


def test_detect_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_CITA_DETECT", "0")
    # Si llamara al helper con el gate off sería un bug — lo asseramos:
    def _boom(*a, **kw):
        raise AssertionError("no deberíamos llamar al helper con gate off")
    monkeypatch.setattr(rag, "_helper_client", _boom)
    assert rag._detect_cita_from_ocr("turno mañana 10hs consulta completa") is None


def test_detect_returns_none_for_short_text(monkeypatch):
    """< _CITA_MIN_CHARS (20) → skip sin helper call."""
    def _boom(*a, **kw):
        raise AssertionError("no helper call para textos cortos")
    monkeypatch.setattr(rag, "_helper_client", _boom)
    assert rag._detect_cita_from_ocr("corto") is None
    assert rag._detect_cita_from_ocr("") is None


def test_detect_event_returns_normalized_dict(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "event", "title": "Turno dentista", "when": "mañana 10hs", '
        '"location": "Av. Santa Fe 1234", "confidence": 0.92}'
    ))
    out = rag._detect_cita_from_ocr("Turno dentista mañana a las 10hs Av Santa Fe 1234")
    assert out == {
        "kind": "event",
        "title": "Turno dentista",
        "when": "mañana 10hs",
        "location": "Av. Santa Fe 1234",
        "confidence": 0.92,
    }


def test_detect_reminder_returns_normalized_dict(monkeypatch):
    """Tarea sin fecha estricta → kind='reminder', when puede estar vacío."""
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "reminder", "title": "Pagar luz", '
        '"when": "antes del 15", "location": "", "confidence": 0.85}'
    ))
    out = rag._detect_cita_from_ocr("Factura Edenor vencimiento antes del 15")
    assert out is not None
    assert out["kind"] == "reminder"
    assert out["title"] == "Pagar luz"
    assert out["when"] == "antes del 15"
    assert out["confidence"] == 0.85


def test_detect_replaces_hallucinated_relative_with_ocr_date(monkeypatch):
    """El OCR manda: viernes 22 de mayo. Si el helper inventa
    "mañana 15hs", el postproceso usa la fecha real y descarta la hora."""
    ocr = (
        "15/05 Estimadas familias: Les informamos que el día viernes 22 "
        "de mayo se llevará a cabo el acto en conmemoración del 25 de Mayo."
    )
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "reminder", "title": "acto conmemoración", '
        '"when": "mañana 15hs", "location": "", "confidence": 0.95}'
    ))
    out = rag._detect_cita_from_ocr(ocr)
    assert out is not None
    assert out["kind"] == "event"
    assert out["when"] == "viernes 22 de mayo"


def test_detect_preserves_true_relative_date_and_time(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "event", "title": "Turno dentista", '
        '"when": "mañana 15hs", "location": "", "confidence": 0.92}'
    ))
    out = rag._detect_cita_from_ocr("Turno dentista mañana a las 15hs")
    assert out is not None
    assert out["kind"] == "event"
    assert out["when"] == "mañana 15hs"


def test_detect_replaces_relative_but_keeps_ocr_time(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "event", "title": "Acto", '
        '"when": "mañana 15hs", "location": "", "confidence": 0.92}'
    ))
    out = rag._detect_cita_from_ocr(
        "El día viernes 22 de mayo a las 15hs se llevará a cabo el acto."
    )
    assert out is not None
    assert out["kind"] == "event"
    assert out["when"] == "viernes 22 de mayo 15hs"


def test_detect_note_returns_normalized_dict(monkeypatch):
    """Info sin acción → kind='note'."""
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "note", "title": "Receta ibuprofeno", '
        '"when": "", "location": "", "confidence": 0.6}'
    ))
    out = rag._detect_cita_from_ocr("Receta: Ibuprofeno 400mg cada 8hs por 5 días")
    assert out is not None
    assert out["kind"] == "note"


def test_detect_malformed_json_returns_none(monkeypatch):
    _mock_helper_client(monkeypatch, content="{este no es JSON válido}}}")
    assert rag._detect_cita_from_ocr("cualquier texto OCR suficientemente largo") is None


def test_detect_helper_exception_returns_none(monkeypatch):
    _mock_helper_client(monkeypatch, exc=RuntimeError("ollama caído"))
    assert rag._detect_cita_from_ocr("cualquier texto OCR suficientemente largo") is None


def test_detect_non_dict_shape_returns_none(monkeypatch):
    _mock_helper_client(monkeypatch, content='["mal", "shape"]')
    assert rag._detect_cita_from_ocr("cualquier texto OCR suficientemente largo") is None


def test_detect_clamps_confidence_to_unit_interval(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "event", "title": "X", "when": "mañana", '
        '"location": "", "confidence": 9.5}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["confidence"] == 1.0


def test_detect_invalid_kind_falls_back_to_note(monkeypatch):
    """Si el helper inventa un kind desconocido ('xyz'), lo normalizamos a
    'note' (el más seguro — no dispara acción)."""
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "xyz", "title": "raro", "when": "", '
        '"location": "", "confidence": 0.5}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["kind"] == "note"


def test_detect_normalizes_string_confidence_to_float(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"kind": "event", "title": "X", "when": "m", '
        '"location": "", "confidence": "0.75"}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["confidence"] == 0.75


# ── Backward-compat: schema viejo {is_cita, start} ─────────────────────────


def test_detect_backward_compat_is_cita_true_maps_to_event(monkeypatch):
    """Modelos / tests con el schema viejo (`is_cita=True`, `start=...`) se
    mapean transparentemente al schema nuevo (`kind='event'`, `when=...`)."""
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": true, "title": "Turno", "start": "mañana 10hs", '
        '"location": "Palermo", "confidence": 0.9}'
    ))
    out = rag._detect_cita_from_ocr("Turno dentista mañana a las 10hs Palermo")
    assert out is not None
    assert out["kind"] == "event"
    assert out["when"] == "mañana 10hs"


def test_detect_backward_compat_is_cita_false_maps_to_note(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": false, "title": "", "start": "", '
        '"location": "", "confidence": 0.0}'
    ))
    out = rag._detect_cita_from_ocr("foto de un meme sin fecha ni hora")
    assert out is not None
    assert out["kind"] == "note"


# ── _maybe_create_cita_from_ocr: gating ────────────────────────────────────


def test_maybe_create_returns_none_when_disabled(monkeypatch, _clean_state):
    monkeypatch.setenv("RAG_CITA_DETECT", "0")
    img = Path("/tmp/fake.png")
    assert rag._maybe_create_cita_from_ocr(
        "turno mañana 10hs con el dentista", img, source="test",
    ) is None


def test_maybe_create_returns_none_for_short_text(monkeypatch, _clean_state):
    img = Path("/tmp/fake.png")
    assert rag._maybe_create_cita_from_ocr("corto", img, source="test") is None


# ── _maybe_create_cita_from_ocr: dedup sidecar ─────────────────────────────


def test_maybe_create_dedup_short_circuits_on_second_call(
    monkeypatch, _clean_state, tmp_path,
):
    """Llamar dos veces con el mismo texto = 1 sola llamada al detector +
    1 sola al propose_calendar_event. Segunda llamada devuelve cached row."""
    detect_calls = []

    def _fake_detect(text):
        detect_calls.append(text)
        return {
            "kind": "event", "title": "Turno", "when": "mañana 10hs",
            "location": "", "confidence": 0.95,
        }
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", _fake_detect)

    propose_calls = _stub_propose_calendar_event(monkeypatch, {
        "kind": "event", "created": True, "event_uid": "UID-123",
        "fields": {"title": "Turno"},
    })

    img = tmp_path / "cita.png"
    img.write_bytes(b"fake")

    r1 = rag._maybe_create_cita_from_ocr(
        "Turno dentista mañana 10hs consultorio", img, source="test",
    )
    assert r1 is not None
    assert r1["decision"] == "cita"
    assert r1["kind"] == "event"
    assert r1["event_uid"] == "UID-123"
    assert r1.get("cached") is False

    r2 = rag._maybe_create_cita_from_ocr(
        "Turno dentista mañana 10hs consultorio", img, source="test",
    )
    assert r2 is not None
    assert r2["decision"] == "cita"
    assert r2["event_uid"] == "UID-123"
    assert r2.get("cached") is True

    # Segundo call NO debería invocar detector ni propose.
    assert len(detect_calls) == 1, f"Esperaba 1 detect call, hubo {len(detect_calls)}"
    assert len(propose_calls) == 1


# ── _maybe_create_cita_from_ocr: event branch ──────────────────────────────


def test_maybe_create_event_happy_path(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "Turno dentista",
        "when": "martes 15hs", "location": "Palermo", "confidence": 0.9,
    })
    _stub_propose_calendar_event(monkeypatch, {
        "kind": "event", "created": True, "event_uid": "UID-EV", "fields": {},
    })
    img = tmp_path / "ev.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "Turno dentista martes 15hs Palermo", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "cita"
    assert out["kind"] == "event"
    assert out["event_uid"] == "UID-EV"


def test_maybe_create_event_missing_when_is_ambiguous(
    monkeypatch, _clean_state, tmp_path,
):
    """kind='event' pero when='' → no creamos ciegamente."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "reunión", "when": "",
        "location": "oficina", "confidence": 0.9,
    })
    propose_calls = _stub_propose_calendar_event(monkeypatch, {})
    img = tmp_path / "sin_fecha.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo que menciona una reunión sin fecha",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "ambiguous"
    assert out["kind"] == "event"
    assert len(propose_calls) == 0


def test_maybe_create_event_duplicate_path(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "cumple Flor", "when": "26 de mayo",
        "location": "", "confidence": 0.9,
    })
    _stub_propose_calendar_event(monkeypatch, {
        "kind": "event", "created": False, "duplicate": True,
        "existing": {"uid": "UID-EXISTING", "title": "cumple Flor"},
        "fields": {},
    })
    img = tmp_path / "dup.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "mañana es el cumple de Flor 26 de mayo yearly",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "duplicate"
    assert out["event_uid"] == "UID-EXISTING"


def test_maybe_create_event_silent_fail_when_propose_raises(
    monkeypatch, _clean_state, tmp_path,
):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "Turno", "when": "mañana 10hs",
        "location": "", "confidence": 0.9,
    })

    def _boom(**kwargs):
        raise RuntimeError("osascript falló")
    monkeypatch.setattr(rag, "propose_calendar_event", _boom)

    img = tmp_path / "e.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo que describe un turno",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "error"


# ── _maybe_create_cita_from_ocr: reminder branch ───────────────────────────


def test_maybe_create_reminder_with_date(monkeypatch, _clean_state, tmp_path):
    """Reminder con deadline → `propose_reminder` + persist."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "reminder", "title": "Pagar luz",
        "when": "antes del 15", "location": "", "confidence": 0.85,
    })
    calls = _stub_propose_reminder(monkeypatch, {
        "kind": "reminder", "created": True, "reminder_id": "RID-1",
        "fields": {},
    })
    img = tmp_path / "factura.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "Factura Edenor vencimiento antes del 15/05",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "reminder"
    assert out["kind"] == "reminder"
    assert out["reminder_id"] == "RID-1"
    assert out["event_uid"] is None
    # El path de la imagen debe estar en el notes blob del reminder.
    assert len(calls) == 1
    assert str(img) in calls[0]["notes"]
    assert "Imagen:" in calls[0]["notes"]


def test_maybe_create_reminder_without_date(monkeypatch, _clean_state, tmp_path):
    """Reminder sin fecha (lista de compras) → se crea igual."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "reminder", "title": "Compras",
        "when": "", "location": "", "confidence": 0.8,
    })
    calls = _stub_propose_reminder(monkeypatch, {
        "kind": "reminder", "created": True, "reminder_id": "RID-2",
        "fields": {},
    })
    img = tmp_path / "shopping.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "Lista compras: huevos tomates pan aceite papel cocina",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "reminder"
    assert out["reminder_id"] == "RID-2"
    assert len(calls) == 1
    assert calls[0]["when"] == ""  # sin fecha


def test_maybe_create_reminder_silent_fail(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "reminder", "title": "Pagar luz",
        "when": "antes del 15", "location": "", "confidence": 0.9,
    })

    def _boom(**kwargs):
        raise RuntimeError("osascript reminders falló")
    monkeypatch.setattr(rag, "propose_reminder", _boom)
    img = tmp_path / "r.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "Factura a pagar antes del 15 de mayo",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "error"
    assert out["kind"] == "reminder"


# ── _maybe_create_cita_from_ocr: note branch ───────────────────────────────


def test_maybe_create_note_is_noop_for_calendar_and_reminders(
    monkeypatch, _clean_state, tmp_path,
):
    """kind='note' → no llama ni a `propose_calendar_event` ni a
    `propose_reminder`. Solo persist para dedup."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "note", "title": "Receta ibuprofeno",
        "when": "", "location": "", "confidence": 0.8,
    })
    cal_calls = _stub_propose_calendar_event(monkeypatch, {})
    rem_calls = _stub_propose_reminder(monkeypatch, {})

    img = tmp_path / "receta.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "Receta médica: ibuprofeno 400mg cada 8hs por 5 días",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "note"
    assert out["kind"] == "note"
    assert out["event_uid"] is None
    assert out["reminder_id"] is None
    assert len(cal_calls) == 0
    assert len(rem_calls) == 0


# ── _maybe_create_cita_from_ocr: confidence thresholds ─────────────────────


def test_maybe_create_low_confidence_persists_and_does_not_act(
    monkeypatch, _clean_state, tmp_path,
):
    """Confidence bajo el umbral → persist 'low_confidence' sin invocar
    ninguno de los propose_*."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "?", "when": "tal vez",
        "location": "", "confidence": 0.3,
    })
    cal_calls = _stub_propose_calendar_event(monkeypatch, {})
    rem_calls = _stub_propose_reminder(monkeypatch, {})

    img = tmp_path / "borderline.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo para pasar el gate",
        img, source="test",
    )
    assert out is not None
    assert out["decision"] == "low_confidence"
    assert out["event_uid"] is None
    assert out["reminder_id"] is None
    assert len(cal_calls) == 0
    assert len(rem_calls) == 0


def test_maybe_create_min_confidence_override(monkeypatch, _clean_state, tmp_path):
    """`rag scan-citas --min-confidence 0.5` — el override debería permitir
    que una confidence de 0.6 (por debajo del 0.7 default) califique."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "kind": "event", "title": "Turno", "when": "mañana",
        "location": "", "confidence": 0.6,
    })
    propose_calls = _stub_propose_calendar_event(monkeypatch, {
        "kind": "event", "created": True, "event_uid": "UID-OVERRIDE",
        "fields": {},
    })

    img = tmp_path / "borderline-override.png"
    img.write_bytes(b"x")

    # Sin override → low_confidence (0.6 < 0.7 default).
    out_default = rag._maybe_create_cita_from_ocr(
        "turno dentista mañana ab", img, source="test",
    )
    assert out_default is not None
    assert out_default["decision"] == "low_confidence"
    assert len(propose_calls) == 0

    # Con override a 0.5, un HASH DIFERENTE (otra imagen, otro texto) crea.
    img2 = tmp_path / "other.png"
    img2.write_bytes(b"y")
    out_override = rag._maybe_create_cita_from_ocr(
        "otro texto distinto sobre un turno mañana",
        img2, source="test",
        min_confidence=0.5,
    )
    assert out_override is not None
    assert out_override["decision"] == "cita"
    assert len(propose_calls) == 1


# ── Integración con _enrich_body_with_ocr ──────────────────────────────────


def test_enrich_body_still_appends_ocr_when_detector_fires(
    monkeypatch, tmp_path, _clean_state,
):
    """Regresión: el cita detector no debe saltarse el body enrichment.
    Si lo hiciera, el `rag eval` perdería recall sobre notas con
    screenshots (invariante documentado en pm.md)."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "screen.png"
    img.write_bytes(b"x")
    note = vault / "note.md"
    note.write_text("# Cita dentista\n\n![[screen.png]]", encoding="utf-8")

    ocr_text = "Turno dentista martes 15hs Dr García consultorio Palermo"
    monkeypatch.setattr(rag, "_ocr_image", lambda p: ocr_text)

    detect_calls = []

    def _fake_detect(text):
        detect_calls.append(text)
        return {
            "kind": "event", "title": "Turno dentista",
            "when": "martes 15hs", "location": "Palermo", "confidence": 0.95,
        }
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", _fake_detect)
    # Stub propose para no tocar Calendar.
    import json as _json
    monkeypatch.setattr(
        rag, "propose_calendar_event",
        lambda **kw: _json.dumps({"created": True, "event_uid": "UID-X"}),
    )

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    # Crítico: el body enrichment persiste.
    assert "Turno dentista martes 15hs" in out
    assert "<!-- OCR: screen.png -->" in out
    # Y el detector fue llamado una vez.
    assert len(detect_calls) == 1


def test_enrich_body_works_when_detector_crashes(monkeypatch, tmp_path, _clean_state):
    """Si el detector raise, el enrichment NO debe propagar — sigue
    appendeando OCR text al body."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "screen.png"
    img.write_bytes(b"x")
    note = vault / "note.md"
    note.write_text("![[screen.png]]", encoding="utf-8")

    monkeypatch.setattr(rag, "_ocr_image", lambda p: "OCR: algún texto largo")

    def _crash(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(rag, "_maybe_create_cita_from_ocr", _crash)

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert "algún texto largo" in out


def test_enrich_body_skips_detector_when_disabled(monkeypatch, tmp_path, _clean_state):
    """`RAG_CITA_DETECT=0` — el wrapper entra pero no llama al helper.
    Body enrichment sigue igual."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "s.png"; img.write_bytes(b"x")
    note = vault / "n.md"
    note.write_text("![[s.png]]", encoding="utf-8")

    monkeypatch.setattr(rag, "_ocr_image", lambda p: "algún texto suficientemente largo")
    monkeypatch.setenv("RAG_CITA_DETECT", "0")

    def _boom_helper(*a, **kw):
        raise AssertionError("no debería llamar al helper")
    monkeypatch.setattr(rag, "_helper_client", _boom_helper)

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert "algún texto suficientemente largo" in out
