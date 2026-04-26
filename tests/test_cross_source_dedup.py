"""Tests para `_cross_source_dedup` (audit 2026-04-25 R2-Cross-source #2).

Cuando el user decide algo en una nota del vault, lo confirma por mail,
lo agenda en Calendar y lo coordina por WhatsApp, el corpus tiene 4
chunks que cubren el MISMO evento. Pre-fix el RAG devolvía los 4 sin
dedup → contexto del LLM lleno de near-duplicates.

Esta dedup pasa después de `_conv_dedup_window` (que dedup-ea WA
intra-chat) y mantiene el chunk de mayor score cuando hay match
cross-source con Jaccard >= 0.7 sobre los primeros 600 chars.
"""
from __future__ import annotations

import pytest

import rag


def _mk_pair(content: str, source: str, score: float):
    """Construye un (candidate, expanded, score) tuple compatible con el
    formato que recibe `_cross_source_dedup`."""
    # `candidate` es (text, meta, id). Usamos `expanded` igual al text
    # porque la función opera sobre el expanded.
    meta = {"source": source, "file": f"{source}/test.md"}
    candidate = (content, meta, f"{source}-id")
    return (candidate, content, score)


def test_dedup_collapses_same_event_across_sources():
    """Mismo texto en vault + gmail + WA → mantiene solo el de mayor score."""
    text = "Reunión con Juan el martes 15 de mayo a las 10am en café Las Violetas para discutir el proyecto de migración"
    pairs = [
        _mk_pair(text + " (vault version)", "vault", 0.92),
        _mk_pair(text + " (gmail version)", "gmail", 0.88),
        _mk_pair(text + " (whatsapp version)", "whatsapp", 0.85),
    ]
    out = rag._cross_source_dedup(pairs, jaccard_threshold=0.7)
    # Solo queda el de mayor score (vault, 0.92)
    assert len(out) == 1
    assert out[0][2] == 0.92
    assert out[0][0][1]["source"] == "vault"


def test_dedup_keeps_distinct_events():
    """Chunks sobre temas distintos NO deben colapsarse aunque sean
    de fuentes distintas."""
    pairs = [
        _mk_pair("Reunión con Juan martes 10am Las Violetas", "vault", 0.92),
        _mk_pair("Comprar yerba y pan en chino esta tarde", "gmail", 0.88),
        _mk_pair("Pagar luz factura mes de abril", "whatsapp", 0.85),
    ]
    out = rag._cross_source_dedup(pairs, jaccard_threshold=0.7)
    # Los 3 quedan porque no comparten tokens
    assert len(out) == 3


def test_dedup_does_not_collapse_same_source():
    """Dedup es ONLY cross-source. 2 chunks del vault con texto similar
    NO se colapsan acá (eso es trabajo de otros mecanismos)."""
    text = "Reunión con Juan el martes 15 de mayo a las 10am en Las Violetas"
    pairs = [
        _mk_pair(text + " primer chunk", "vault", 0.92),
        _mk_pair(text + " segundo chunk", "vault", 0.88),
    ]
    out = rag._cross_source_dedup(pairs, jaccard_threshold=0.7)
    # Mismo source → no se aplica
    assert len(out) == 2


def test_dedup_threshold_configurable_via_env(monkeypatch):
    """Threshold via env var para latency-sensitive surfaces o
    debugging (set 1.0 para deshabilitar)."""
    text = "Reunión con Juan el martes 15 mayo 10am Las Violetas migración proyecto"
    pairs = [
        _mk_pair(text, "vault", 0.92),
        _mk_pair(text, "gmail", 0.88),
    ]

    # Threshold 1.0 deshabilita dedup
    monkeypatch.setenv("RAG_CROSS_SOURCE_DEDUP_THRESHOLD", "1.0")
    out = rag._cross_source_dedup(pairs)
    assert len(out) == 2

    # Threshold 0.5 (más permisivo) sigue dedup-eando
    monkeypatch.setenv("RAG_CROSS_SOURCE_DEDUP_THRESHOLD", "0.5")
    out = rag._cross_source_dedup(pairs)
    assert len(out) == 1


def test_dedup_handles_empty_input():
    """Lista vacía → devuelve lista vacía sin crashear."""
    assert rag._cross_source_dedup([]) == []


def test_dedup_handles_single_pair():
    """1 solo pair → passthrough."""
    pairs = [_mk_pair("solo este", "vault", 0.9)]
    out = rag._cross_source_dedup(pairs)
    assert len(out) == 1


def test_dedup_preserves_score_order():
    """El output mantiene el orden de score descendente del input."""
    pairs = [
        _mk_pair("contenido A único", "vault", 0.95),
        _mk_pair("contenido B único", "gmail", 0.85),
        _mk_pair("contenido C único", "whatsapp", 0.75),
    ]
    out = rag._cross_source_dedup(pairs)
    scores = [p[2] for p in out]
    assert scores == sorted(scores, reverse=True)


def test_dedup_handles_short_text():
    """Tokens demasiado cortos (<3 chars) no cuentan, así que chunks
    muy cortos con palabras pequeñas no se dedup-ean (no hay señal
    suficiente)."""
    pairs = [
        _mk_pair("OK", "vault", 0.9),
        _mk_pair("ok", "gmail", 0.85),
    ]
    # Token sets vacíos → no dedup
    out = rag._cross_source_dedup(pairs)
    assert len(out) == 2
