"""Tests del helper `_split_oversized_lines` (Opción 2 fix embedder hang 2026-05-06).

El helper pre-splittea líneas mayores a `max_chars` antes que el chunker procese
el texto, garantizando que el embedder MPS no aloque attention buffers gigantes
para párrafos de 12k+ chars (root cause del hang `state U` observado).
"""

from __future__ import annotations

import pytest

from rag import _split_oversized_lines


def test_short_line_passthrough():
    text = "linea corta sin nada raro"
    assert _split_oversized_lines(text, max_chars=1500) == text


def test_multiple_short_lines_passthrough():
    text = "uno\ndos\ntres"
    assert _split_oversized_lines(text, max_chars=1500) == text


def test_empty_text():
    assert _split_oversized_lines("", max_chars=1500) == ""


def test_split_by_bullet_separator():
    # Línea de >1500 chars con separator `· ` (típico feeds bullet-style)
    parts = ["evento " + str(i) + " que paso hoy en algun lugar" for i in range(80)]
    line = " · ".join(parts)
    assert len(line) > 1500
    out = _split_oversized_lines(line, max_chars=200)
    out_lines = out.split("\n")
    assert len(out_lines) > 1, "debe haber splitteado en múltiples líneas"
    for ln in out_lines:
        assert len(ln) <= 250, f"linea sigue muy larga: {len(ln)}"


def test_split_by_period_separator():
    # Línea sin `· ` pero con `. ` (oraciones)
    parts = ["esta es la oracion numero " + str(i) + " bastante larga" for i in range(60)]
    line = ". ".join(parts) + "."
    assert len(line) > 1500
    out = _split_oversized_lines(line, max_chars=200)
    out_lines = out.split("\n")
    assert len(out_lines) > 1
    for ln in out_lines:
        assert len(ln) <= 250


def test_split_by_semicolon_separator():
    # Línea con solo `; ` (separador secundario)
    parts = ["item de la lista " + str(i) + " con detalle extra" for i in range(80)]
    line = "; ".join(parts)
    assert len(line) > 1500
    out = _split_oversized_lines(line, max_chars=200)
    out_lines = out.split("\n")
    assert len(out_lines) > 1


def test_whitespace_fallback_no_separators():
    # Línea sin · ni . ni ; — solo palabras separadas por espacios
    word = "palabra"
    line = " ".join([word] * 300)  # ~2400 chars
    assert len(line) > 1500
    out = _split_oversized_lines(line, max_chars=500)
    out_lines = out.split("\n")
    assert len(out_lines) > 1
    for ln in out_lines:
        assert len(ln) <= 600  # algo de slack por last word boundary


def test_preserves_short_lines_in_mixed():
    # Mix: línea corta + línea larga + línea corta
    short_top = "header corto"
    long_mid_parts = ["chunk " + str(i) for i in range(100)]
    long_mid = ". ".join(long_mid_parts)
    short_bot = "footer corto"
    text = f"{short_top}\n{long_mid}\n{short_bot}"
    out = _split_oversized_lines(text, max_chars=200)
    out_lines = out.split("\n")
    assert out_lines[0] == short_top, "primera línea debe quedar igual"
    assert out_lines[-1] == short_bot, "última línea debe quedar igual"


def test_default_max_chars_1500():
    # Sin override, default es 1500. Línea con espacios para que el
    # whitespace fallback pueda partirla.
    word = "palabra"
    line = " ".join([word] * 500)  # ~4000 chars
    out = _split_oversized_lines(line)  # default max_chars=1500
    out_lines = out.split("\n")
    assert len(out_lines) > 1


def test_no_change_at_boundary():
    # Línea exactamente max_chars no debe splittearse
    line = "a" * 1500
    out = _split_oversized_lines(line, max_chars=1500)
    assert out == line


def test_realistic_youtube_transcript_shape():
    # Simula transcript YouTube emitido por youtube_transcript_api
    # como un solo string sin newlines, separator natural ` · `
    segments = [f"segmento {i} con algo de texto" for i in range(200)]
    transcript = " · ".join(segments)  # ~7000+ chars en una línea
    assert len(transcript) > 5000
    out = _split_oversized_lines(transcript, max_chars=1500)
    out_lines = out.split("\n")
    assert len(out_lines) > 3, "transcript debe partirse en varios chunks"
    # Verificar que NO perdemos contenido — concatenando con el separator debe
    # reconstruir el original (módulo posible reordering del separator)
    reconstituted = " · ".join(out_lines)
    # Cada line interna conserva los segments con `· `, las líneas se unen con `\n`
    # No es exact identity por como armamos buf, pero el contenido debe estar
    assert "segmento 0" in out
    assert "segmento 199" in out


def test_unicode_safe():
    # Caracteres unicode (emojis, acentos, CJK) — el splitter cuenta chars
    # no bytes, así que UTF-8 multibyte no rompe length()
    line = ("hola che 👋 esto es un test con acentós " * 100)
    assert len(line) > 1500
    out = _split_oversized_lines(line, max_chars=300)
    out_lines = out.split("\n")
    assert len(out_lines) > 1


@pytest.mark.parametrize("max_chars", [100, 500, 1500, 2000, 5000])
def test_all_output_lines_under_cap_with_slack(max_chars):
    # Para múltiples max_chars, cada línea de salida debe respetar cap+slack
    line = " · ".join([f"item{i}" for i in range(2000)])
    out = _split_oversized_lines(line, max_chars=max_chars)
    out_lines = out.split("\n")
    for ln in out_lines:
        # Slack: el join con sep puede empujar un poco, pero no >2x
        assert len(ln) <= max_chars * 2, (
            f"line {len(ln)} chars exceeds 2*{max_chars} cap"
        )
