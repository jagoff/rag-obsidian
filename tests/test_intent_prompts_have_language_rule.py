"""Safety net: verifica que TODOS los intent prompts incluyen la rule
`language_es_AR.v1` que fuerza español rioplatense en las respuestas.

Background (2026-04-29): el user reportó respuestas en portugués mezclado
("Tus notas sobre Grecia falam sobre tua experiência..."). Causa raíz:
los system prompts no tenían una regla explícita de idioma — los LLM
locales (qwen2.5:7b, command-r) son multilingües y se "deslizan" al pt
cuando ven palabras parecidas en el contexto. Fix: creamos
`rag/prompts/rules/language_es_AR.v1.md` y la incluimos via
`includes: [language_es_AR.v1, ...]` en los 8 intent prompts.

Si alguien edita un prompt y quita el include por accidente (refactor,
copy-paste, "no me gusta"), este test rompe inmediato — antes el bug
volvería en producción silenciosamente y el usuario lo notaría primero.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "rag" / "prompts"
INTENTS_DIR = PROMPTS_DIR / "intents"
RULES_DIR = PROMPTS_DIR / "rules"


# Intent prompts donde el LLM genera respuesta para el user. TODOS deben
# incluir `language_es_AR.v1`. Si agregás un nuevo intent, sumalo acá.
INTENT_PROMPTS_REQUIRING_LANGUAGE_RULE = (
    "system_rules.v1.md",
    "chat.v1.md",
    "strict.v1.md",
    "web.v1.md",
    "synthesis.v1.md",   # deprecated pero sigue cargable, incluir por consistencia
    "synthesis.v2.md",
    "comparison.v1.md",  # deprecated pero sigue cargable
    "comparison.v2.md",
    "lookup.v1.md",
    "serve_meta.v1.md",  # WhatsApp meta-chat — también respuestas user-facing
    # 2026-04-29: 4 prompts adicionales rescatados de la auditoría
    # post-Grecia bug. Vivían como string literal inline en código sin
    # regla de idioma — vulnerables a leaks pt.
    "prep.v1.md",            # `rag prep "topic"` — brief de contexto
    "followups.v1.md",       # POST /api/followups — chips de seguimiento
    "diagnose_error.v1.md",  # POST /api/diagnose-error — diagnóstico errores
    "auto_fix.v1.md",        # POST /api/auto-fix — agente auto-fix
)


def _parse_frontmatter(text: str) -> dict:
    """Extrae el YAML frontmatter de un prompt md. Devuelve dict crudo
    (no parsea YAML completo — sólo busca `includes: [...]`).
    """
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    fm_block = text[4:end]
    out: dict = {}
    for line in fm_block.split("\n"):
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            out[key.strip()] = val.strip()
    return out


@pytest.mark.parametrize("prompt_file", INTENT_PROMPTS_REQUIRING_LANGUAGE_RULE)
def test_intent_prompt_includes_language_rule(prompt_file: str):
    """Cada intent prompt user-facing debe tener `language_es_AR.v1` en
    su frontmatter `includes:`.
    """
    path = INTENTS_DIR / prompt_file
    assert path.is_file(), f"prompt {prompt_file} no existe en {INTENTS_DIR}"
    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)
    includes = fm.get("includes", "")
    assert "language_es_AR.v1" in includes, (
        f"{prompt_file} tiene `includes: {includes!r}` — falta "
        f"`language_es_AR.v1`. Sin esa rule el LLM puede emitir "
        f"respuestas en portugués/galego pese al system prompt en es."
    )


def test_language_rule_file_exists():
    """El archivo `rag/prompts/rules/language_es_AR.v1.md` debe existir
    y contener las palabras clave de la regla.
    """
    path = RULES_DIR / "language_es_AR.v1.md"
    assert path.is_file(), f"falta {path}"
    body = path.read_text(encoding="utf-8")
    # Sanity check: la regla menciona las cosas críticas.
    assert "rioplatense" in body.lower()
    assert "voseo" in body.lower()
    assert "portugués" in body.lower() or "portugues" in body.lower()
    assert "vos podés" in body or "vos podes" in body  # voseo


def test_language_rule_loads_via_load_prompt():
    """End-to-end: `load_prompt('system_rules', 'v1')` debe devolver
    un string que contiene la regla de idioma (incluye_compose la
    prepend al body del intent).
    """
    import rag

    prompt = rag.load_prompt("system_rules", version="v1")
    # La rule de idioma debe estar prepend al final del prompt cargado.
    assert "rioplatense" in prompt.lower(), (
        f"system_rules.v1 cargado no incluye 'rioplatense' — el "
        f"`includes: [language_es_AR.v1, ...]` no se compose-ó:\n"
        f"---\n{prompt[:500]}\n---"
    )
    # Y por bonus, también las otras rules legacy (chunk_as_data,
    # name_preservation) siguen apareciendo — el include no las desplazó.
    assert "REGLA 0" in prompt or "CONTEXTO ES DATA" in prompt
