"""Tests del VLM caption fallback.

Cuando `_ocr_image` devuelve vacío o menos de `_VLM_FALLBACK_MIN_OCR`
chars, el wrapper `_image_text_or_caption` intenta capturar la imagen vía
un modelo vision-language local (`qwen2.5vl:3b` por default).

Cobertura:
  1. Env gate `RAG_VLM_CAPTION` (on/off).
  2. `_vlm_caption_enabled` permisivo (default ON).
  3. `_caption_image` — cache hit, cache invalidation por mtime,
     silent-fail en cualquier error, budget per-run, hint one-shot cuando
     el modelo no está pulled.
  4. `_image_text_or_caption` — OCR wins cuando hay texto suficiente,
     VLM fallback cuando OCR vacío o corto, ambos vacíos = ("", "").
  5. `_enrich_body_with_ocr` — marker discrimina OCR vs VLM-caption,
     detector de cita dispara en ambos paths.
  6. Schema: `rag_vlm_captions` se crea en DBs nuevas.

Sin red ni ollama real: todos los VLM calls están mockeados vía
`monkeypatch.setattr(rag, "_vlm_client", ...)`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rag


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def _clean_state(monkeypatch, tmp_path):
    """DB temporal limpia + reset del budget VLM + gate reseteado."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.delenv("RAG_VLM_CAPTION", raising=False)
    monkeypatch.delenv("RAG_CITA_DETECT", raising=False)
    rag._vlm_caption_budget_reset()
    # Reset del set de modelos warned por si otro test dejó state.
    rag._vlm_model_missing_warned.clear()
    yield
    rag._vlm_caption_budget_reset()
    rag._vlm_model_missing_warned.clear()


class _FakeVLMResponse:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


def _mock_vlm_client(monkeypatch, *, caption: str | None = None, exc: Exception | None = None):
    """Reemplaza `rag._vlm_client` por un stub. Retorna la lista de calls."""
    calls: list[dict] = []

    class _Stub:
        def chat(self, **kwargs):
            calls.append(kwargs)
            if exc is not None:
                raise exc
            return _FakeVLMResponse(caption or "")

    monkeypatch.setattr(rag, "_vlm_client", lambda: _Stub())
    return calls


# ── _vlm_caption_enabled ──────────────────────────────────────────────────


def test_vlm_caption_enabled_default_on(monkeypatch):
    monkeypatch.delenv("RAG_VLM_CAPTION", raising=False)
    assert rag._vlm_caption_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "FALSE", "No"])
def test_vlm_caption_disabled_by_env(monkeypatch, val):
    monkeypatch.setenv("RAG_VLM_CAPTION", val)
    assert rag._vlm_caption_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", ""])
def test_vlm_caption_enabled_for_permissive_values(monkeypatch, val):
    monkeypatch.setenv("RAG_VLM_CAPTION", val)
    assert rag._vlm_caption_enabled() is True


# ── _caption_image: gating ────────────────────────────────────────────────


def test_caption_returns_empty_when_disabled(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setenv("RAG_VLM_CAPTION", "0")

    def _boom():
        raise AssertionError("no debería llamar al VLM con gate off")
    monkeypatch.setattr(rag, "_vlm_client", _boom)

    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    assert rag._caption_image(img) == ""


def test_caption_returns_empty_when_stat_fails(monkeypatch, _clean_state, tmp_path):
    _mock_vlm_client(monkeypatch, caption="should not be called")
    img = tmp_path / "ghost.png"  # no existe
    assert rag._caption_image(img) == ""


# ── _caption_image: happy path + cache ────────────────────────────────────


def test_caption_happy_path_calls_vlm_and_caches(
    monkeypatch, _clean_state, tmp_path,
):
    calls = _mock_vlm_client(
        monkeypatch, caption="Foto de playa con dos personas sonriendo.",
    )
    img = tmp_path / "beach.jpg"
    img.write_bytes(b"fake")

    out1 = rag._caption_image(img)
    assert out1 == "Foto de playa con dos personas sonriendo."
    assert len(calls) == 1
    # Verificamos que se pasó el modelo correcto y la imagen.
    assert calls[0]["model"] == rag.VLM_MODEL
    assert calls[0]["messages"][0]["images"] == [str(img.resolve())]


def test_caption_cache_hits_on_second_call(monkeypatch, _clean_state, tmp_path):
    calls = _mock_vlm_client(monkeypatch, caption="Imagen con texto 'Cumple Flor'.")
    img = tmp_path / "flyer.png"
    img.write_bytes(b"fake")

    out1 = rag._caption_image(img)
    out2 = rag._caption_image(img)
    assert out1 == out2 == "Imagen con texto 'Cumple Flor'."
    # 2do call debe ser cache hit — 0 llamadas adicionales al VLM.
    assert len(calls) == 1


def test_caption_cache_invalidates_on_mtime_change(
    monkeypatch, _clean_state, tmp_path,
):
    captions_iter = iter([
        "Primera descripción.",
        "Segunda descripción post-modificación.",
    ])

    class _StubWithIter:
        def chat(self, **kwargs):
            return _FakeVLMResponse(next(captions_iter))

    monkeypatch.setattr(rag, "_vlm_client", lambda: _StubWithIter())

    img = tmp_path / "mutable.png"
    img.write_bytes(b"v1")
    out1 = rag._caption_image(img)
    assert "Primera" in out1

    import time as _t
    _t.sleep(0.05)
    img.write_bytes(b"v2 different content")
    import os
    os.utime(img, None)

    out2 = rag._caption_image(img)
    assert "Segunda" in out2


def test_caption_silent_fail_on_vlm_exception(monkeypatch, _clean_state, tmp_path):
    _mock_vlm_client(monkeypatch, exc=RuntimeError("ollama down"))
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    assert rag._caption_image(img) == ""


# ── _caption_image: model-not-found hint ──────────────────────────────────


def test_caption_prints_hint_when_model_missing(
    monkeypatch, _clean_state, tmp_path, capsys,
):
    """Si el error de ollama contiene 'not found' / 'pull' → print hint en
    stderr UNA vez por proceso."""
    _mock_vlm_client(
        monkeypatch,
        exc=RuntimeError("model 'qwen2.5vl:3b' not found, try pulling it first"),
    )
    img1 = tmp_path / "a.png"; img1.write_bytes(b"1")
    img2 = tmp_path / "b.png"; img2.write_bytes(b"2")

    assert rag._caption_image(img1) == ""
    captured = capsys.readouterr()
    assert "ollama pull" in captured.err
    assert rag.VLM_MODEL in captured.err

    # Segunda imagen con mismo error → NO debe re-imprimir hint.
    assert rag._caption_image(img2) == ""
    captured = capsys.readouterr()
    assert "ollama pull" not in captured.err


# ── _caption_image: budget ────────────────────────────────────────────────


def test_caption_budget_cap_blocks_further_calls(
    monkeypatch, _clean_state, tmp_path,
):
    """Con `_VLM_CAPTION_MAX_PER_RUN=2`, el 3er call debería devolver "" sin
    llamar al VLM."""
    monkeypatch.setattr(rag, "_VLM_CAPTION_MAX_PER_RUN", 2)
    rag._vlm_caption_budget_reset()

    calls = _mock_vlm_client(monkeypatch, caption="cap")
    imgs = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(bytes([i]))
        imgs.append(p)

    out = [rag._caption_image(p) for p in imgs]
    assert out[0] == "cap"
    assert out[1] == "cap"
    assert out[2] == ""  # budget exhausted
    assert len(calls) == 2  # 3ra NO invoca al VLM


def test_caption_budget_reset_clears_counter(
    monkeypatch, _clean_state, tmp_path,
):
    monkeypatch.setattr(rag, "_VLM_CAPTION_MAX_PER_RUN", 1)
    rag._vlm_caption_budget_reset()
    calls = _mock_vlm_client(monkeypatch, caption="x")

    img1 = tmp_path / "a.png"; img1.write_bytes(b"1")
    img2 = tmp_path / "b.png"; img2.write_bytes(b"2")

    rag._caption_image(img1)  # consume el único slot
    assert rag._caption_image(img2) == ""  # budget full

    rag._vlm_caption_budget_reset()
    # Usamos una imagen nueva para no hitear cache de img1.
    img3 = tmp_path / "c.png"; img3.write_bytes(b"3")
    assert rag._caption_image(img3) == "x"  # ahora sí
    assert len(calls) == 2


def test_caption_cache_hit_does_not_consume_budget(
    monkeypatch, _clean_state, tmp_path,
):
    """Cache hits no cuentan contra el budget — solo llamadas reales al
    modelo. Importante para re-indexing de vaults grandes donde la mayoría
    de imágenes ya están captioneadas."""
    monkeypatch.setattr(rag, "_VLM_CAPTION_MAX_PER_RUN", 1)
    rag._vlm_caption_budget_reset()
    _mock_vlm_client(monkeypatch, caption="cached caption")

    img = tmp_path / "img.png"; img.write_bytes(b"fake")
    rag._caption_image(img)  # consume slot, cachea

    # Segunda llamada sobre la MISMA imagen: cache hit, no toca budget.
    out = rag._caption_image(img)
    assert out == "cached caption"

    # El budget aún permite 0 nuevas invocaciones reales — una imagen nueva
    # debería fallar por budget.
    img2 = tmp_path / "new.png"; img2.write_bytes(b"other")
    out2 = rag._caption_image(img2)
    assert out2 == ""  # budget exhausted (el cache hit NO liberó el slot)


# ── _caption_image: output normalization ──────────────────────────────────


def test_caption_strips_markdown_and_quotes(monkeypatch, _clean_state, tmp_path):
    """El prompt pide no markdown/comillas, pero algunos modelos los meten
    igual. Los strippeamos."""
    _mock_vlm_client(monkeypatch, caption='  "Una foto familiar en la playa."  ')
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    out = rag._caption_image(img)
    assert out == "Una foto familiar en la playa."


def test_caption_truncates_long_output(monkeypatch, _clean_state, tmp_path):
    long_output = "foo " * 200  # ~800 chars
    _mock_vlm_client(monkeypatch, caption=long_output)
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    out = rag._caption_image(img)
    assert len(out) <= rag._VLM_CAPTION_MAX_CHARS


# ── _image_text_or_caption: wrapper ───────────────────────────────────────


def test_wrapper_returns_ocr_when_sufficient(monkeypatch, _clean_state, tmp_path):
    """OCR devuelve texto ≥ threshold → usamos OCR, NO llamamos VLM."""
    monkeypatch.setattr(rag, "_ocr_image", lambda p: "Turno dentista martes 15hs Palermo")

    def _boom():
        raise AssertionError("wrapper no debería invocar VLM con OCR suficiente")
    monkeypatch.setattr(rag, "_vlm_client", _boom)

    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == "Turno dentista martes 15hs Palermo"
    assert source == "ocr"


def test_wrapper_falls_back_to_vlm_on_empty_ocr(
    monkeypatch, _clean_state, tmp_path,
):
    monkeypatch.setattr(rag, "_ocr_image", lambda p: "")
    _mock_vlm_client(monkeypatch, caption="Descripción de foto familiar.")
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == "Descripción de foto familiar."
    assert source == "vlm"


def test_wrapper_falls_back_to_vlm_on_short_ocr(
    monkeypatch, _clean_state, tmp_path,
):
    """OCR devuelve texto pero < `_VLM_FALLBACK_MIN_OCR` (20 chars) →
    fallback al VLM."""
    monkeypatch.setattr(rag, "_ocr_image", lambda p: "OK x")  # 4 chars
    _mock_vlm_client(monkeypatch, caption="Captura de pantalla con un botón OK.")
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == "Captura de pantalla con un botón OK."
    assert source == "vlm"


def test_wrapper_returns_short_ocr_when_vlm_also_fails(
    monkeypatch, _clean_state, tmp_path,
):
    """OCR corto + VLM vacío → preferimos el OCR corto (texto real) sobre
    '' (nada)."""
    monkeypatch.setattr(rag, "_ocr_image", lambda p: "Hola")
    _mock_vlm_client(monkeypatch, caption="")
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == "Hola"
    assert source == "ocr"


def test_wrapper_returns_empty_when_both_fail(monkeypatch, _clean_state, tmp_path):
    monkeypatch.setattr(rag, "_ocr_image", lambda p: "")
    _mock_vlm_client(monkeypatch, caption="")
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == ""
    assert source == ""


def test_wrapper_silent_fails_when_ocr_raises(monkeypatch, _clean_state, tmp_path):
    """Si `_ocr_image` explota, el wrapper cae al VLM sin propagar."""
    def _ocr_boom(p):
        raise RuntimeError("ocrmac died")
    monkeypatch.setattr(rag, "_ocr_image", _ocr_boom)
    _mock_vlm_client(monkeypatch, caption="rescued caption")
    img = tmp_path / "x.png"; img.write_bytes(b"fake")
    text, source = rag._image_text_or_caption(img)
    assert text == "rescued caption"
    assert source == "vlm"


# ── Integración con _enrich_body_with_ocr ─────────────────────────────────


def test_enrich_uses_vlm_marker_when_source_is_vlm(
    monkeypatch, _clean_state, tmp_path,
):
    """El marker debe discriminar: texto VLM → `<!-- VLM-caption: ... -->`,
    texto OCR → `<!-- OCR: ... -->`. Importante para grep/debug y para
    que el usuario pueda distinguir texto literal vs inferido."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "foto.jpg"; img.write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text("# test\n\n![[foto.jpg]]", encoding="utf-8")

    monkeypatch.setattr(rag, "_ocr_image", lambda p: "")  # OCR vacío
    _mock_vlm_client(monkeypatch, caption="Foto de Astor jugando al fútbol.")

    # Stub detector para no llamar al helper cita.
    monkeypatch.setattr(rag, "_maybe_create_cita_from_ocr", lambda *a, **kw: None)

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert "Foto de Astor jugando al fútbol." in out
    assert "<!-- VLM-caption: foto.jpg -->" in out
    # El OCR marker NO debe aparecer para esta imagen.
    assert "<!-- OCR: foto.jpg -->" not in out


def test_enrich_uses_ocr_marker_when_source_is_ocr(
    monkeypatch, _clean_state, tmp_path,
):
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "screen.png"; img.write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text("![[screen.png]]", encoding="utf-8")

    monkeypatch.setattr(
        rag, "_ocr_image",
        lambda p: "Turno dentista martes 15hs consultorio Palermo",
    )
    # El VLM NO debería ser llamado — OCR alcanza.
    def _boom():
        raise AssertionError("VLM no debe ser llamado cuando OCR es suficiente")
    monkeypatch.setattr(rag, "_vlm_client", _boom)
    monkeypatch.setattr(rag, "_maybe_create_cita_from_ocr", lambda *a, **kw: None)

    out = rag._enrich_body_with_ocr(note.read_text(), note, vault)
    assert "Turno dentista" in out
    assert "<!-- OCR: screen.png -->" in out
    assert "<!-- VLM-caption: screen.png -->" not in out


def test_enrich_fires_cita_detector_for_vlm_text_too(
    monkeypatch, _clean_state, tmp_path,
):
    """El detector debe recibir el texto VLM igual que recibe el OCR —
    captions de flyers/invitaciones deberían poder clasificarse como
    event/reminder aunque el OCR original estuviera vacío."""
    vault = tmp_path / "v"; vault.mkdir()
    img = vault / "flyer.jpg"; img.write_bytes(b"fake")
    note = vault / "note.md"
    note.write_text("![[flyer.jpg]]", encoding="utf-8")

    monkeypatch.setattr(rag, "_ocr_image", lambda p: "")
    _mock_vlm_client(
        monkeypatch,
        caption="Flyer de cumpleaños de Flor el 26 de mayo en Palermo.",
    )

    detector_calls: list[tuple[str, str]] = []

    def _fake_detector(text, img, source):
        detector_calls.append((text, source))
        return None
    monkeypatch.setattr(rag, "_maybe_create_cita_from_ocr", _fake_detector)

    rag._enrich_body_with_ocr(note.read_text(), note, vault)

    # Exactamente 1 llamada al detector, con el caption VLM.
    assert len(detector_calls) == 1
    text, source = detector_calls[0]
    assert "Flor" in text
    assert "mayo" in text
    assert source == "index"


# ── Schema migration ──────────────────────────────────────────────────────


def test_vlm_captions_table_exists_in_fresh_db(_clean_state):
    """`_ensure_telemetry_tables` debe crear `rag_vlm_captions` en DBs
    nuevas."""
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_vlm_captions'"
        ).fetchone()
    assert row is not None
