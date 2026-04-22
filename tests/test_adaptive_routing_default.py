"""Tests del flip de RAG_ADAPTIVE_ROUTING a default ON (2026-04-22).

Contexto:
  - Feature completa desde commit `89ccc0e` (Fase C, 2026-04-22).
  - CLAUDE.md §RAG_ADAPTIVE_ROUTING: "Sin regresión eval: ambas runs ON y
    OFF producen resultados bit-idénticos en `rag eval` (validado
    2026-04-21)".
  - Plan de activación documentado: "flipear a ON por default una vez
    que `rag eval` vuelva a superar el floor de singles 76.19% de forma
    estable".

Pre-requisito ya cubierto en esta sesión: `commit cfac737` hizo que web
+ chat loguen `intent`, así que ahora sí se puede medir el efecto real
del adaptive routing sin invención (pre-flip: 98.4% de queries tenía
intent=NULL en extra_json → imposible medir).

Cambio mínimo: `_adaptive_routing()` cambia default ON, pero respeta
override explícito `RAG_ADAPTIVE_ROUTING=0` para rollback.

Invariantes:
  - Override OFF (=0/false/no) sigue funcionando
  - `RAG_FORCE_FULL_PIPELINE=1` sigue apagando todo (debug escape)
  - `_should_skip_reformulate` sigue gateando en metadata-only intents
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Default ON ───────────────────────────────────────────────────────────────


def test_adaptive_routing_default_is_on(monkeypatch):
    """Post 2026-04-22 el default pasa a ON. Sin env explícito, True."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    assert rag._adaptive_routing() is True, \
        "Post 2026-04-22 RAG_ADAPTIVE_ROUTING debe ser ON por default " \
        "(Fase C completa + intent telemetry poblada)"


# ── Overrides (rollback + debug) ─────────────────────────────────────────────


def test_adaptive_routing_explicit_off_respected(monkeypatch):
    """El operador puede rollback con `RAG_ADAPTIVE_ROUTING=0`."""
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    for val in ("0", "false", "no", "FALSE", "No"):
        monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", val)
        assert rag._adaptive_routing() is False, \
            f"RAG_ADAPTIVE_ROUTING={val!r} debe deshabilitar"


def test_adaptive_routing_explicit_on_respected(monkeypatch):
    """Setear =1 explícito sigue andando (no-op vs default, pero no rompe)."""
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    assert rag._adaptive_routing() is True


def test_adaptive_routing_force_full_pipeline_override(monkeypatch):
    """`RAG_FORCE_FULL_PIPELINE=1` sigue apagando el adaptive — debug
    escape hatch usado para A/B del overhead de dispatch."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)  # default ON
    monkeypatch.setenv("RAG_FORCE_FULL_PIPELINE", "1")
    assert rag._adaptive_routing() is False


def test_adaptive_routing_empty_string_is_on(monkeypatch):
    """Env var vacío (no-op de launchd plists sin el flag) se interpreta
    como "no seteado" → default ON."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "")
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    assert rag._adaptive_routing() is True


# ── Downstream: _should_skip_reformulate sigue gateando correcto ─────────────


def test_should_skip_reformulate_on_metadata_intents(monkeypatch):
    """Con default ON, metadata-only intents saltean reformulate."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    for intent in ("count", "list", "recent", "agenda", "entity_lookup"):
        assert rag._should_skip_reformulate(intent) is True, \
            f"intent={intent!r} debe saltear reformulate con adaptive ON"


def test_should_skip_reformulate_preserves_reformulate_for_semantic(monkeypatch):
    """Semantic / synthesis / comparison / unknown NO saltean — siguen
    pagando reformulate_query (el prompt del LLM necesita el turn history)."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    for intent in ("semantic", "synthesis", "comparison", None, "unknown"):
        assert rag._should_skip_reformulate(intent) is False, \
            f"intent={intent!r} NO debe saltear reformulate (no es metadata-only)"


def test_should_skip_reformulate_off_when_explicit_rollback(monkeypatch):
    """Rollback vía `RAG_ADAPTIVE_ROUTING=0` apaga el skip para TODOS los
    intents — pipeline legacy bit-identical."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "0")
    for intent in ("count", "list", "recent", "semantic", "synthesis"):
        assert rag._should_skip_reformulate(intent) is False
