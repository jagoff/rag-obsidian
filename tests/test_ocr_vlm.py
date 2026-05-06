"""Tests para el VLM caption fallback (mlx-vlm / granite-vision).

Reemplazó al backend ollama (qwen2.5vl:3b) en Phase 3 de la migración MLX.
Contrato principal: silent-fail total — cualquier error en el VLM devuelve ""
sin propagar excepciones al indexer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rag


# ── _vlm_describe: silent-fail contract ─────────────────────────────────────


def test_vlm_describe_missing_file_returns_empty_string():
    """Path inexistente → silent-fail → ""."""
    result = rag._vlm_describe("/tmp/no-such-file-vlm-test-12345.jpg")
    assert result == ""


def test_vlm_describe_invalid_image_returns_empty_string(tmp_path):
    """Archivo que existe pero no es imagen válida → silent-fail → ""."""
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"this is not a valid image file at all")
    result = rag._vlm_describe(str(bad))
    assert result == ""


def test_vlm_describe_uses_custom_prompt(monkeypatch):
    """Si `_vlm_load` falla, el prompt custom no importa — devuelve "" igual."""
    def _boom(_path, prompt=""):
        raise RuntimeError("load failed")

    monkeypatch.setattr(rag, "_vlm_load", _boom)
    result = rag._vlm_describe("/tmp/whatever.jpg", prompt="Describí esta imagen.")
    assert result == ""


def test_vlm_describe_load_failure_returns_empty_string(monkeypatch):
    """Cuando `_vlm_load` falla (modelo no descargado, OOM, etc.), → ""."""
    def _raise_load():
        raise OSError("model weights missing")

    # `_vlm_describe` llama a `_vlm_load()` internamente
    import rag.ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_vlm_load", _raise_load)
    result = rag._vlm_describe("/tmp/whatever.jpg")
    assert result == ""


def test_vlm_describe_generate_failure_returns_empty_string(monkeypatch, tmp_path):
    """Cuando generate() falla (VRAM OOM, etc.), → ""."""
    img = tmp_path / "real.png"
    img.write_bytes(b"fake png")

    import rag.ocr as ocr_mod

    class _FakeModel:
        config = {}

    class _FakeProcessor:
        pass

    monkeypatch.setattr(ocr_mod, "_vlm_load", lambda: (_FakeModel(), _FakeProcessor()))

    def _bad_generate(*a, **kw):
        raise RuntimeError("VRAM exhausted")

    monkeypatch.setattr("mlx_vlm.generate", _bad_generate)
    result = rag._vlm_describe(str(img))
    assert result == ""


class _FakeTokenizer:
    """Tokenizer falso con `apply_chat_template` que devuelve un string formateado."""
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "formatted_prompt"


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


def test_vlm_describe_returns_string_from_mock(monkeypatch, tmp_path):
    """Con mock que devuelve texto coherente, _vlm_describe propaga ese texto."""
    img = tmp_path / "test.png"
    img.write_bytes(b"fake png")

    import rag.ocr as ocr_mod

    class _FakeModel:
        config = {}

    monkeypatch.setattr(ocr_mod, "_vlm_load", lambda: (_FakeModel(), _FakeProcessor()))

    class _FakeResult:
        text = "Una imagen de prueba con texto OCR visible."

    def _fake_generate(*a, **kw):
        return _FakeResult()

    monkeypatch.setattr("mlx_vlm.generate", _fake_generate)

    result = rag._vlm_describe(str(img))
    assert result == "Una imagen de prueba con texto OCR visible."


def test_vlm_describe_strips_whitespace_from_output(monkeypatch, tmp_path):
    """Output con whitespace extra debe ser stripeado."""
    img = tmp_path / "test.png"
    img.write_bytes(b"fake png")

    import rag.ocr as ocr_mod

    class _FakeModel:
        config = {}

    monkeypatch.setattr(ocr_mod, "_vlm_load", lambda: (_FakeModel(), _FakeProcessor()))

    class _FakeResult:
        text = "  Texto con espacios extra.  \n\n"

    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **kw: _FakeResult())

    result = rag._vlm_describe(str(img))
    assert result == "Texto con espacios extra."


# ── _vlm_idle_unload ─────────────────────────────────────────────────────────


def test_vlm_idle_unload_noop_when_no_model_loaded():
    """Si no hay modelo cargado, idle_unload devuelve False sin crashear."""
    import rag.ocr as ocr_mod
    original_model = ocr_mod._VLM_MODEL_OBJ
    ocr_mod._VLM_MODEL_OBJ = None
    try:
        result = rag._vlm_idle_unload(idle_seconds=0)
        assert result is False
    finally:
        ocr_mod._VLM_MODEL_OBJ = original_model


def test_vlm_idle_unload_noop_when_recently_used():
    """Si el modelo se usó hace menos que `idle_seconds`, no descargar."""
    import time
    import rag.ocr as ocr_mod

    original_model = ocr_mod._VLM_MODEL_OBJ
    original_last = ocr_mod._VLM_LAST_USED

    ocr_mod._VLM_MODEL_OBJ = object()  # simulamos modelo cargado
    ocr_mod._VLM_LAST_USED = time.time()  # usado ahora mismo
    try:
        result = rag._vlm_idle_unload(idle_seconds=3600)
        assert result is False
        assert ocr_mod._VLM_MODEL_OBJ is not None
    finally:
        ocr_mod._VLM_MODEL_OBJ = original_model
        ocr_mod._VLM_LAST_USED = original_last


def test_vlm_idle_unload_evicts_after_timeout():
    """Si el modelo lleva más de `idle_seconds`, se descarga y devuelve True."""
    import time
    import rag.ocr as ocr_mod

    original_model = ocr_mod._VLM_MODEL_OBJ
    original_proc = ocr_mod._VLM_PROCESSOR
    original_last = ocr_mod._VLM_LAST_USED

    ocr_mod._VLM_MODEL_OBJ = object()
    ocr_mod._VLM_PROCESSOR = object()
    ocr_mod._VLM_LAST_USED = time.time() - 1000  # viejo
    try:
        result = rag._vlm_idle_unload(idle_seconds=5)
        assert result is True
        assert ocr_mod._VLM_MODEL_OBJ is None
        assert ocr_mod._VLM_PROCESSOR is None
    finally:
        ocr_mod._VLM_MODEL_OBJ = original_model
        ocr_mod._VLM_PROCESSOR = original_proc
        ocr_mod._VLM_LAST_USED = original_last


# ── VLM_MODEL default ────────────────────────────────────────────────────────


def test_vlm_model_default_is_granite_vision():
    """VLM_MODEL default debe ser el HF ID del granite-vision mlx-community."""
    import os
    # Solo verificamos cuando no hay override de env
    if not os.environ.get("RAG_VLM_MODEL"):
        assert rag.VLM_MODEL == "mlx-community/granite-vision-3.2-2b-4bit"


# ── requires_mlx: smoke real (skipped en CI Linux) ──────────────────────────


@pytest.mark.requires_mlx
def test_vlm_describe_real_image_returns_nonempty(tmp_path):
    """Smoke test real: genera caption de una imagen con texto legible.

    Skipped si mlx_vlm no está instalado, si el modelo no está en cache de
    HuggingFace, o en CI Linux (auto-skip por marker `requires_mlx`).
    Para forzar la descarga del modelo antes de correr: `hf download mlx-community/granite-vision-3.2-2b-4bit`.
    """
    import os
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        pytest.skip("pillow no instalado — necesario para generar imagen de test")

    # Skip si el modelo no está en cache local (evita descarga automática en test suite).
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub_dir = hf_cache / "hub"
    model_slug = "mlx-community/granite-vision-3.2-2b-4bit"
    safe_slug = model_slug.replace("/", "--")
    model_dir = hub_dir / f"models--{safe_slug}"
    if not model_dir.exists():
        pytest.skip(f"Modelo {model_slug} no en cache local — corré: hf download {model_slug}")

    img_path = tmp_path / "smoke_ocr.png"
    img = Image.new("RGB", (512, 256), color="white")
    d = ImageDraw.Draw(img)
    d.text((20, 50), "Hello OCR test", fill="black")
    img.save(str(img_path))

    result = rag._vlm_describe(str(img_path))
    assert isinstance(result, str)
    assert len(result) > 0, "Se esperaba texto no-vacío del VLM real"
