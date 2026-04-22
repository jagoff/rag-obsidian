"""Tests: context_summary default SKIP (opt-in en vez de opt-out).

Evidencia empírica que motiva el cambio (2026-04-22):

    === context_summaries.json ===
    {}                                    ← 0 entries

    === Corpus actual ===
    meta_obsidian_notes_v11_*:  1259 chunks
    Chunks con "Contexto:" en document:  7 (0.56%)

La feature está efectivamente off en producción: el cache JSON está
vacío y <1% de los chunks del corpus actual la usan. El claim original
del commit que la introdujo ("+11% chain_success") nunca se replicó
contra el queries.yaml actual y CLAUDE.md la marca como "unverified".

Costo operativo documentado que estábamos pagando sin beneficio:
  - ~11 min extra por `rag index --reset` (~0.16s × ~4k chunks)
  - 1-3 s por query cuando el cache estaba cold (aunque el cache estaba
    vacío, el bloque de LLM call ni siquiera corría porque el gate
    `if file_hash in cache` nunca daba miss — solo corría en re-index)

Política nueva: **default SKIP**. Opt-in explícito via `RAG_CONTEXT_SUMMARY=1`
para quien quiera experimentar. El env var legacy
`OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY` se respeta (retrocompat) pero es
redundante con el nuevo default — sirve como marker explícito en
scripts viejos.

Si en el futuro alguien quiere reactivar, debe:
  1. export RAG_CONTEXT_SUMMARY=1
  2. rag index --reset  (reindex con summaries en cada chunk)
  3. rag eval × 3 para validar ganancia real antes de merge
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Default skip ─────────────────────────────────────────────────────────────


def test_context_summary_default_is_skip(monkeypatch):
    """Sin ningún env set, el helper debe retornar "" (skip)."""
    monkeypatch.delenv("RAG_CONTEXT_SUMMARY", raising=False)
    monkeypatch.delenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", raising=False)

    out = rag._generate_context_summary(
        "body text long enough to pass the min-body gate " * 20,
        title="Test", folder="01-Projects",
    )
    assert out == "", (
        "Post 2026-04-22 el default es skip — sin RAG_CONTEXT_SUMMARY=1 "
        "explícito, el helper retorna '' inmediatamente sin LLM call"
    )


# ── Legacy env (opt-out explícito) sigue funcionando ─────────────────────────


def test_legacy_skip_env_still_respected(monkeypatch):
    """`OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1` sigue respetado (retrocompat
    con scripts + plists viejos que lo seteen). Como el default es skip,
    este env es redundante pero no tóxico."""
    monkeypatch.delenv("RAG_CONTEXT_SUMMARY", raising=False)
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", "1")

    out = rag._generate_context_summary(
        "body " * 200, title="Test", folder="test",
    )
    assert out == ""


# ── Opt-in explícito activa el codepath legacy ──────────────────────────────


def test_opt_in_env_triggers_generation(monkeypatch):
    """Con `RAG_CONTEXT_SUMMARY=1` el helper intenta generar (y el caller
    debe caer al LLM). Usamos un mock del client para evitar tocar ollama
    real — lo único que verificamos es que NO short-circuitee en "" antes
    de llamar al client."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("RAG_CONTEXT_SUMMARY", "1")
    monkeypatch.delenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", raising=False)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Una nota sobre testing."
    mock_client.chat.return_value = mock_response

    monkeypatch.setattr(rag, "_summary_client", lambda: mock_client)

    out = rag._generate_context_summary(
        "body " * 200, title="Test", folder="test",
    )
    assert out == "Una nota sobre testing.", (
        "Con RAG_CONTEXT_SUMMARY=1 el helper DEBE correr el LLM call "
        "y devolver el contenido del message. Got: " + repr(out)
    )
    # Sanity: el mock client SÍ se llamó.
    assert mock_client.chat.called


def test_opt_in_truthy_values(monkeypatch):
    """Acepta `1`, `true`, `yes` como truthy."""
    monkeypatch.delenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", raising=False)

    for truthy in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("RAG_CONTEXT_SUMMARY", truthy)
        assert rag._context_summary_enabled() is True, \
            f"RAG_CONTEXT_SUMMARY={truthy!r} debe ser truthy"


def test_opt_in_falsy_values_default_skip(monkeypatch):
    monkeypatch.delenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", raising=False)

    for falsy in ("0", "false", "no", "FALSE", "No", ""):
        monkeypatch.setenv("RAG_CONTEXT_SUMMARY", falsy)
        assert rag._context_summary_enabled() is False, \
            f"RAG_CONTEXT_SUMMARY={falsy!r} debe ser falsy"


# ── Precedencia: legacy skip > opt-in ─────────────────────────────────────────


def test_legacy_skip_overrides_opt_in(monkeypatch):
    """Defensivo: si alguien setea AMBAS, el legacy skip gana (más
    conservative). Raro en práctica, pero un plist heredado + un export
    nuevo en un shell no debe tener comportamiento ambiguo."""
    monkeypatch.setenv("RAG_CONTEXT_SUMMARY", "1")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", "1")
    assert rag._context_summary_enabled() is False, \
        "OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1 debe ganar sobre RAG_CONTEXT_SUMMARY=1"


# ── Helper exportado ────────────────────────────────────────────────────────


def test_context_summary_enabled_helper_exists():
    assert hasattr(rag, "_context_summary_enabled")
    assert callable(rag._context_summary_enabled)
