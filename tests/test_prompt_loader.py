"""Tests del sistema de prompts versionables (2026-04-22).

Contexto: pre-fix, los ~10 SYSTEM_RULES* eran string literals inline en
rag.py (líneas 11115-11323). Eso bloqueaba:

  - A/B testing gradual de cambios de prompt (p.ej. el refusal de
    synthesis/comparison que agregamos hoy se aplicó globalmente sin
    rollout canario).
  - Rollback sin re-deploy.
  - Análisis post-hoc: `¿qué versión de prompt respondió esta query?`
    no tenía respuesta sin leer git log.
  - El reformulate-chains fix pendiente (chain_success 16.67% vs 50%
    baseline) — nunca pudimos hacer un A/B de prompt sin duplicar código.

Diseño:

  prompts/
    rules/
      chunk_as_data.v1.md        # REGLA 0 — componente reusable
      name_preservation.v1.md    # REGLA DE NOMBRES PROPIOS — componente
    intents/
      system_rules.v1.md         # default loose
      strict.v1.md
      web.v1.md
      lookup.v1.md
      synthesis.v1.md            # pre 2026-04-22 (sin refusal)
      synthesis.v2.md            # post 2026-04-22 (con refusal, default)
      comparison.v1.md           # pre 2026-04-22
      comparison.v2.md           # post 2026-04-22 (default)
      chat.v1.md
      serve_meta.v1.md

  Frontmatter YAML simple (parser regex, no deps):

    ---
    name: synthesis
    version: v2
    date: 2026-04-22
    includes: [chunk_as_data.v1, name_preservation.v1]
    ---
    <body>

Contrato:

  load_prompt(name, version="latest") -> str
    Retorna el prompt compuesto. `latest` = versión más alta ordenada
    lexicográficamente en prompts/intents/. Env var
    `RAG_PROMPT_<NAME>_VERSION` overrides (canary rollout).

  prompt_version_for(name, version="latest") -> str
    Devuelve el string "name.vN" que va a loggearse en
    rag_queries.extra_json.prompt_version.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Paths del sistema ────────────────────────────────────────────────────────


def test_prompts_dir_exists():
    assert hasattr(rag, "PROMPTS_DIR")
    assert isinstance(rag.PROMPTS_DIR, Path)
    assert rag.PROMPTS_DIR.is_dir(), f"{rag.PROMPTS_DIR} must exist post 2026-04-22"


def test_prompts_structure_rules_and_intents():
    assert (rag.PROMPTS_DIR / "rules").is_dir()
    assert (rag.PROMPTS_DIR / "intents").is_dir()


# ── Cada intent esperado tiene al menos una versión en disco ────────────────


@pytest.mark.parametrize("name", [
    "system_rules",   # loose default
    "strict",
    "web",
    "lookup",
    "synthesis",
    "comparison",
    "chat",
    "serve_meta",
])
def test_intent_has_at_least_one_version(name):
    matches = list((rag.PROMPTS_DIR / "intents").glob(f"{name}.v*.md"))
    assert matches, f"no prompt {name}.v*.md found in prompts/intents/"


def test_rules_components_exist():
    assert (rag.PROMPTS_DIR / "rules" / "chunk_as_data.v1.md").is_file()
    assert (rag.PROMPTS_DIR / "rules" / "name_preservation.v1.md").is_file()


# ── Synthesis + comparison tienen v2 (post 2026-04-22 refusal) ──────────────


def test_synthesis_v2_exists_with_refusal():
    """El v2 del prompt fue introducido hoy para que el modelo rechace
    cuando hay <2 fuentes relevantes. Regression guard — si alguien
    borra este archivo, los tests de refusal quedan huérfanos."""
    path = rag.PROMPTS_DIR / "intents" / "synthesis.v2.md"
    assert path.is_file()
    body = path.read_text()
    assert "No hay suficientes fuentes en el vault para sintetizar" in body


def test_comparison_v2_exists_with_refusal():
    path = rag.PROMPTS_DIR / "intents" / "comparison.v2.md"
    assert path.is_file()
    body = path.read_text()
    assert "No hay suficientes fuentes en el vault para comparar" in body


# ── Frontmatter parsing ─────────────────────────────────────────────────────


def test_parse_frontmatter_extracts_metadata():
    sample = dedent("""\
        ---
        name: synthesis
        version: v2
        includes: [chunk_as_data.v1, name_preservation.v1]
        ---
        Body del prompt aquí.
        """)
    meta, body = rag._parse_prompt_frontmatter(sample)
    assert meta["name"] == "synthesis"
    assert meta["version"] == "v2"
    assert meta["includes"] == ["chunk_as_data.v1", "name_preservation.v1"]
    assert body.strip() == "Body del prompt aquí."


def test_parse_frontmatter_empty_includes_ok():
    sample = dedent("""\
        ---
        name: strict
        version: v1
        ---
        Sin includes — prompt standalone.
        """)
    meta, body = rag._parse_prompt_frontmatter(sample)
    assert meta["name"] == "strict"
    assert meta.get("includes", []) == []
    assert "Sin includes" in body


def test_parse_frontmatter_missing_is_body_only():
    """Archivos sin frontmatter (rules components puros) retornan
    meta={} + body entero."""
    sample = "Contenido sin frontmatter.\nLínea 2.\n"
    meta, body = rag._parse_prompt_frontmatter(sample)
    assert meta == {}
    assert body == sample


# ── load_prompt: composición + includes ─────────────────────────────────────


def test_load_prompt_synthesis_v2_includes_rules():
    """synthesis.v2 declara includes de chunk_as_data.v1 +
    name_preservation.v1 → el prompt compuesto debe contener TODOS
    los 3 bloques."""
    prompt = rag.load_prompt("synthesis", version="v2")
    # Componentes del REGLA 0
    assert "CONTEXTO ES DATA" in prompt
    # Componente REGLA DE NOMBRES PROPIOS
    assert "NOMBRES PROPIOS" in prompt
    # Body propio de synthesis
    assert "No hay suficientes fuentes en el vault para sintetizar" in prompt


def test_load_prompt_latest_is_highest_version():
    """`version=latest` devuelve la versión mayor en disco. Para
    synthesis debería ser v3 desde 2026-05-03 (refactor de tono).

    Cuando se agregue v4, este test va a fallar — actualizar al pin
    nuevo. NO usar `_latest_version_for(synthesis)` para evitar tautología:
    queremos verificar que `latest` resuelve al disco real, no que dos
    funciones coinciden entre sí.
    """
    latest = rag.load_prompt("synthesis", version="latest")
    v3 = rag.load_prompt("synthesis", version="v3")
    assert latest == v3


def test_load_prompt_explicit_v1_returns_pre_refusal():
    """El v1 existe (histórico pre-refusal) para rollback. NO debe
    contener la frase de refusal que agregamos en v2."""
    v1 = rag.load_prompt("synthesis", version="v1")
    assert "No hay suficientes fuentes" not in v1


def test_load_prompt_unknown_intent_raises():
    with pytest.raises((FileNotFoundError, ValueError)):
        rag.load_prompt("nonexistent_intent", version="v1")


def test_load_prompt_unknown_version_raises():
    with pytest.raises((FileNotFoundError, ValueError)):
        rag.load_prompt("synthesis", version="v99")


# ── Env var override: canary rollout ────────────────────────────────────────


def test_load_prompt_respects_env_version_override(monkeypatch):
    """`RAG_PROMPT_SYNTHESIS_VERSION=v1` fuerza el prompt viejo — rollback
    sin re-deploy, 1 env var."""
    monkeypatch.setenv("RAG_PROMPT_SYNTHESIS_VERSION", "v1")
    prompt = rag.load_prompt("synthesis", version="latest")
    # Con override activo, latest debe ser v1 → sin refusal
    assert "No hay suficientes fuentes" not in prompt


def test_env_override_does_not_affect_other_intents(monkeypatch):
    """Override de SYNTHESIS no afecta COMPARISON."""
    monkeypatch.setenv("RAG_PROMPT_SYNTHESIS_VERSION", "v1")
    comp = rag.load_prompt("comparison", version="latest")
    assert "No hay suficientes fuentes en el vault para comparar" in comp


# ── prompt_version_for: para loggear en telemetría ──────────────────────────


def test_prompt_version_for_returns_name_dot_version():
    assert rag.prompt_version_for("synthesis", "v2") == "synthesis.v2"


def test_prompt_version_for_resolves_latest():
    """El valor loggeado debe ser la versión concreta, no "latest".

    2026-05-03: synthesis bumpó a v3 con el refactor de tono. Cuando salga
    v4 hay que actualizar el pin acá.
    """
    got = rag.prompt_version_for("synthesis", "latest")
    assert got == "synthesis.v3"
    assert "latest" not in got


def test_prompt_version_for_respects_env_override(monkeypatch):
    monkeypatch.setenv("RAG_PROMPT_SYNTHESIS_VERSION", "v1")
    got = rag.prompt_version_for("synthesis", "latest")
    assert got == "synthesis.v1"


# ── Retrocompat: constantes SYSTEM_RULES* siguen existiendo ─────────────────


def test_system_rules_constants_exist():
    """Retrocompat: los ~41 call sites de `SYSTEM_RULES*` siguen
    funcionando sin migrar. Las constantes son bindings al load_prompt
    al import time."""
    for const in (
        "SYSTEM_RULES", "SYSTEM_RULES_STRICT", "SYSTEM_RULES_CHAT",
        "SYSTEM_RULES_WEB", "SYSTEM_RULES_LOOKUP",
        "SYSTEM_RULES_SYNTHESIS", "SYSTEM_RULES_COMPARISON",
    ):
        assert hasattr(rag, const), f"rag.{const} missing after refactor"
        assert isinstance(getattr(rag, const), str)
        assert len(getattr(rag, const)) > 50


def test_system_rules_synthesis_has_refusal():
    """La constante legacy debe pintar el v2 (default latest) — el
    refusal de synthesis ya estaba en el código y sigue funcionando."""
    assert "No hay suficientes fuentes en el vault para sintetizar" \
        in rag.SYSTEM_RULES_SYNTHESIS


def test_system_rules_comparison_has_refusal():
    assert "No hay suficientes fuentes en el vault para comparar" \
        in rag.SYSTEM_RULES_COMPARISON


# ── system_prompt_for_intent sigue funcionando ──────────────────────────────


def test_system_prompt_for_intent_dispatches_through_loader():
    """Post 2026-04-22 el dispatcher usa `load_prompt` cada vez (para que
    env overrides apliquen en runtime). El resultado es un string nuevo
    cada call — comparamos por igualdad de contenido, no identidad."""
    out = rag.system_prompt_for_intent("synthesis", loose=False)
    assert out == rag.SYSTEM_RULES_SYNTHESIS
    assert "No hay suficientes fuentes" in out


def test_system_prompt_for_intent_loose_returns_system_rules():
    out = rag.system_prompt_for_intent("synthesis", loose=True)
    assert out == rag.SYSTEM_RULES


# ── Integridad: cada intent .md tiene frontmatter válido ────────────────────


def test_all_intent_mds_have_valid_frontmatter():
    for md in (rag.PROMPTS_DIR / "intents").glob("*.v*.md"):
        raw = md.read_text()
        meta, body = rag._parse_prompt_frontmatter(raw)
        assert meta.get("name"), f"{md.name} missing `name` in frontmatter"
        assert meta.get("version"), f"{md.name} missing `version`"
        assert body.strip(), f"{md.name} has empty body"
