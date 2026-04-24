"""Tests del detector de citas a partir de texto OCR.

Cubre:
  1. Env gate `RAG_CITA_DETECT` (on/off).
  2. `_normalize_ocr_for_hash` — lowercase + whitespace collapse.
  3. `_ocr_hash_key` — determinismo + insensibilidad a espacios.
  4. `_detect_cita_from_ocr` — helper calls con mocks: happy path,
     malformed JSON, exception, shape inválida, clamp de confidence,
     normalización de tipos (is_cita=1, confidence="0.8").
  5. `_maybe_create_cita_from_ocr` — dedup sidecar, low-confidence
     persistencia, ambiguous (sin fecha), create happy path, duplicate
     path, error silent-fail.
  6. Integración con `_enrich_body_with_ocr` — el body OCR enrichment
     se preserva aunque el detector se llame (regresión potencial sobre
     `rag eval` en notas con screenshots).

Sin red ni ollama real: todos los LLM calls están mockeados.
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
    o lanza `exc`. Retorna el mock para inspección."""
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


def test_detect_happy_path_returns_normalized_dict(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": true, "title": "Turno dentista", "start": "mañana 10hs", '
        '"location": "Av. Santa Fe 1234", "confidence": 0.92}'
    ))
    out = rag._detect_cita_from_ocr("Turno dentista mañana a las 10hs Av Santa Fe 1234")
    assert out == {
        "is_cita": True,
        "title": "Turno dentista",
        "start": "mañana 10hs",
        "location": "Av. Santa Fe 1234",
        "confidence": 0.92,
    }


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
        '{"is_cita": true, "title": "X", "start": "mañana", '
        '"location": "", "confidence": 9.5}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["confidence"] == 1.0


def test_detect_normalizes_integer_is_cita_to_bool(monkeypatch):
    """Algunos LLMs devuelven 1/0 en vez de true/false. Lo toleramos."""
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": 1, "title": "X", "start": "mañana", '
        '"location": "", "confidence": 0.8}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["is_cita"] is True


def test_detect_normalizes_string_confidence_to_float(monkeypatch):
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": true, "title": "X", "start": "m", '
        '"location": "", "confidence": "0.75"}'
    ))
    out = rag._detect_cita_from_ocr("texto OCR largo suficiente para pasar el gate")
    assert out is not None
    assert out["confidence"] == 0.75


def test_detect_is_cita_false_returns_dict_not_none(monkeypatch):
    """Respuesta 'no es cita' aún es válida — retorna dict, caller decide
    qué hacer. None solo es para errores."""
    _mock_helper_client(monkeypatch, content=(
        '{"is_cita": false, "title": "", "start": "", '
        '"location": "", "confidence": 0.0}'
    ))
    out = rag._detect_cita_from_ocr("foto de un meme sin fecha ni hora ni nada")
    assert out is not None
    assert out["is_cita"] is False


# ── _maybe_create_cita_from_ocr ────────────────────────────────────────────


def _stub_propose_calendar_event(monkeypatch, result: dict):
    """Reemplaza `rag.propose_calendar_event` para no tocar Calendar.app
    en los tests. Retorna una lista con las llamadas para inspección."""
    import json as _json
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return _json.dumps(result, ensure_ascii=False)

    monkeypatch.setattr(rag, "propose_calendar_event", _fake)
    return calls


def test_maybe_create_returns_none_when_disabled(monkeypatch, _clean_state):
    monkeypatch.setenv("RAG_CITA_DETECT", "0")
    img = Path("/tmp/fake.png")
    assert rag._maybe_create_cita_from_ocr(
        "turno mañana 10hs con el dentista", img, source="test",
    ) is None


def test_maybe_create_returns_none_for_short_text(monkeypatch, _clean_state):
    img = Path("/tmp/fake.png")
    assert rag._maybe_create_cita_from_ocr("corto", img, source="test") is None


def test_maybe_create_dedup_short_circuits_on_second_call(monkeypatch, _clean_state, tmp_path):
    """Llamar dos veces con el mismo texto = 1 sola llamada al detector +
    1 sola al propose_calendar_event. Segunda llamada devuelve cached row."""
    detect_calls = []

    def _fake_detect(text):
        detect_calls.append(text)
        return {
            "is_cita": True, "title": "Turno", "start": "mañana 10hs",
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


def test_maybe_create_low_confidence_persists_and_does_not_create(
    monkeypatch, _clean_state, tmp_path,
):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": True, "title": "?", "start": "tal vez",
        "location": "", "confidence": 0.3,
    })
    propose_calls = _stub_propose_calendar_event(monkeypatch, {})

    img = tmp_path / "borderline.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo para pasar el gate", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "low_confidence"
    assert out["event_uid"] is None
    assert len(propose_calls) == 0  # no debería llamar propose


def test_maybe_create_is_cita_false_persists_as_no(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": False, "title": "", "start": "",
        "location": "", "confidence": 0.0,
    })
    _stub_propose_calendar_event(monkeypatch, {})

    img = tmp_path / "meme.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "foto de un meme con mucho texto pero sin fechas", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "no"


def test_maybe_create_missing_start_is_ambiguous(monkeypatch, _clean_state, tmp_path):
    """Detector dice is_cita=True pero start="" → no creamos evento, persist
    'ambiguous'."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": True, "title": "reunión", "start": "",
        "location": "oficina", "confidence": 0.9,
    })
    propose_calls = _stub_propose_calendar_event(monkeypatch, {})

    img = tmp_path / "sin_fecha.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo que menciona una reunión sin fecha", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "ambiguous"
    assert len(propose_calls) == 0


def test_maybe_create_duplicate_path_persists_duplicate(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": True, "title": "cumple Flor", "start": "26 de mayo",
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
        "mañana es el cumple de Flor 26 de mayo yearly", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "duplicate"
    assert out["event_uid"] == "UID-EXISTING"


def test_maybe_create_silent_fail_when_propose_raises(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": True, "title": "Turno", "start": "mañana 10hs",
        "location": "", "confidence": 0.9,
    })
    def _boom(**kwargs):
        raise RuntimeError("osascript falló")
    monkeypatch.setattr(rag, "propose_calendar_event", _boom)

    img = tmp_path / "e.png"
    img.write_bytes(b"x")
    out = rag._maybe_create_cita_from_ocr(
        "texto suficientemente largo que describe un turno", img, source="test",
    )
    assert out is not None
    assert out["decision"] == "error"


def test_maybe_create_min_confidence_override(monkeypatch, _clean_state, tmp_path):
    """`rag scan-citas --min-confidence 0.5` — el override debería permitir
    que una confidence de 0.6 (por debajo del 0.7 default) califique."""
    monkeypatch.setattr(rag, "_detect_cita_from_ocr", lambda t: {
        "is_cita": True, "title": "Turno", "start": "mañana",
        "location": "", "confidence": 0.6,
    })
    propose_calls = _stub_propose_calendar_event(monkeypatch, {
        "kind": "event", "created": True, "event_uid": "UID-OVERRIDE", "fields": {},
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

    # Con override a 0.5, un HASH DIFERENTE (otra imagen, otro texto) debería crear.
    img2 = tmp_path / "other.png"
    img2.write_bytes(b"y")
    out_override = rag._maybe_create_cita_from_ocr(
        "otro texto distinto sobre un turno mañana", img2, source="test",
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
            "is_cita": True, "title": "Turno dentista",
            "start": "martes 15hs", "location": "Palermo", "confidence": 0.95,
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
