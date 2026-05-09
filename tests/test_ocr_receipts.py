"""Tests for VLM #10 — receipt parser + chart describer + JSON extractor.

NO model loading — todo mockea `_vlm_describe` para verificar lógica
del parser, cache, y silent-fail. El smoke real (con granite-vision)
queda como manual `.venv/bin/python -c '...'`.
"""
from __future__ import annotations

import json

import pytest

import rag
from rag.ocr import (
    _extract_json_object,
    _vlm_parse_receipt,
    _vlm_describe_chart,
)


# ── _extract_json_object ────────────────────────────────────────────────────


def test_extract_json_basic():
    assert _extract_json_object('{"a":1}') == '{"a":1}'


def test_extract_json_with_prose():
    assert _extract_json_object('Acá tenés: {"a":1} listo') == '{"a":1}'


def test_extract_json_markdown_fence():
    raw = '```json\n{"merchant":"X"}\n```'
    assert _extract_json_object(raw) == '{"merchant":"X"}'


def test_extract_json_markdown_fence_no_lang():
    raw = '```\n{"a":1}\n```'
    assert _extract_json_object(raw) == '{"a":1}'


def test_extract_json_nested():
    raw = '{"items":[{"price":10}],"total":10}'
    assert _extract_json_object(raw) == raw


def test_extract_json_curly_inside_string():
    # "}" dentro de string NO debe cerrar el objeto
    raw = '{"note":"with } inside"}'
    assert _extract_json_object(raw) == raw


def test_extract_json_escaped_quote():
    raw = '{"a":"b\\"c"}'
    assert _extract_json_object(raw) == raw


def test_extract_json_empty():
    assert _extract_json_object("") is None
    assert _extract_json_object("no json here") is None


def test_extract_json_unclosed():
    # Objeto sin cierre balanceado → None (no devolvemos JSON inválido)
    assert _extract_json_object('{"a":1') is None


# ── _vlm_parse_receipt ──────────────────────────────────────────────────────


@pytest.fixture
def fake_image(tmp_path):
    p = tmp_path / "ticket.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake jpg")
    return p


def test_parse_receipt_happy_path(monkeypatch, fake_image):
    """VLM emite JSON válido → dict normalizado."""
    fake_json = json.dumps({
        "merchant": "Coto",
        "date": "2026-05-09",
        "total": 12500.50,
        "currency": "ARS",
        "items": [{"description": "leche", "quantity": 2, "price": 3000}],
        "category": "food",
    })
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": fake_json)
    result = _vlm_parse_receipt(fake_image)
    assert isinstance(result, dict)
    assert result["merchant"] == "Coto"
    assert result["total"] == 12500.50
    assert result["category"] == "food"
    assert len(result["items"]) == 1


def test_parse_receipt_with_markdown_fence(monkeypatch, fake_image):
    raw = '```json\n{"merchant":"X","date":"2026-05-09","total":100,"currency":"ARS","items":[],"category":"other"}\n```'
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": raw)
    result = _vlm_parse_receipt(fake_image)
    assert result is not None
    assert result["merchant"] == "X"


def test_parse_receipt_not_a_receipt(monkeypatch, fake_image):
    """VLM detecta que no es recibo → None (no error)."""
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": '{"error":"not_a_receipt"}')
    result = _vlm_parse_receipt(fake_image)
    assert result is None


def test_parse_receipt_invalid_json(monkeypatch, fake_image):
    """VLM devuelve texto sin JSON → None silencioso."""
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": "no es JSON tampoco")
    result = _vlm_parse_receipt(fake_image)
    assert result is None


def test_parse_receipt_empty_response(monkeypatch, fake_image):
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": "")
    result = _vlm_parse_receipt(fake_image)
    assert result is None


def test_parse_receipt_malformed_json(monkeypatch, fake_image):
    """JSON con shape incorrecta (string root) → None."""
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": '"just a string"')
    result = _vlm_parse_receipt(fake_image)
    assert result is None


def test_parse_receipt_image_missing(monkeypatch, tmp_path):
    """Path inexistente → None."""
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": '{"x":1}')
    result = _vlm_parse_receipt(tmp_path / "no-existe.jpg")
    assert result is None


def test_parse_receipt_caption_disabled(monkeypatch, fake_image):
    """RAG_VLM_CAPTION=0 → None sin invocar VLM."""
    monkeypatch.setenv("RAG_VLM_CAPTION", "0")
    called = []
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": called.append(1) or '{"x":1}')
    result = _vlm_parse_receipt(fake_image)
    assert result is None
    assert called == []  # VLM NO se llamó


def test_parse_receipt_cache_hit(monkeypatch, fake_image, tmp_path):
    """Segundo call con misma image+mtime → cache hit, NO re-llama VLM."""
    fake_json = '{"merchant":"X","total":100}'
    calls = []

    def fake_vlm(p, prompt=""):
        calls.append(p)
        return fake_json

    monkeypatch.setattr(rag, "_vlm_describe", fake_vlm)

    # 1ra llamada — VLM se invoca
    r1 = _vlm_parse_receipt(fake_image)
    assert r1 is not None
    assert len(calls) == 1

    # 2da llamada — cache hit, NO invoca VLM
    r2 = _vlm_parse_receipt(fake_image)
    assert r2 is not None
    assert r2["merchant"] == "X"
    assert len(calls) == 1  # mismo count


# ── _vlm_describe_chart ─────────────────────────────────────────────────────


def test_describe_chart_happy_path(monkeypatch, fake_image):
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": "Bar chart con max 25K en marzo")
    result = _vlm_describe_chart(fake_image)
    assert "Bar chart" in result
    assert "25K" in result


def test_describe_chart_not_a_chart(monkeypatch, fake_image):
    """VLM detecta que no es chart → "" silencioso."""
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": "no es un gráfico")
    assert _vlm_describe_chart(fake_image) == ""


def test_describe_chart_truncates(monkeypatch, fake_image):
    long_text = "X" * 1000
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": long_text)
    result = _vlm_describe_chart(fake_image)
    assert len(result) <= 500  # _VLM_CAPTION_MAX_CHARS


def test_describe_chart_strips_quotes_and_newlines(monkeypatch, fake_image):
    monkeypatch.setattr(rag, "_vlm_describe", lambda p, prompt="": '"Caption con\nlínea nueva"')
    result = _vlm_describe_chart(fake_image)
    assert not result.startswith('"')
    assert "\n" not in result
