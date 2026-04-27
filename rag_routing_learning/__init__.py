"""Routing learning loop — Bloque B del voice-classifier.

El listener de WhatsApp clasifica audios reenviados con qwen2.5:7b en uno
de 7 buckets (calendar_timed, reminder, inbox, mail, wa_send, rag_query,
none). Cada decisión + la respuesta del usuario se persiste a la tabla
``rag_routing_decisions`` (Phase 0 del rollout: shadow logging via JSONL,
Phase 1+: SQL directo).

Este subpackage analiza ese histórico para que el classifier "aprenda":

- ``patterns`` — extrae n-grams frecuentes con bucket consistente
  (ratio ≥0.9, count ≥5) → candidatos a promover como reglas heurísticas
  inyectadas al sysprompt.
- ``promote`` — UPSERT contra ``rag_routing_rules``, activación/
  desactivación, listado.

Doc: ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes/
04-Archive/99-obsidian-system/99-AI/system/voice-classifier/spec.md

Esquema (en ``rag/__init__.py`` ``_TELEMETRY_DDL``):

- ``rag_routing_decisions`` — una fila por cada decisión del classifier.
  Columnas clave: ``transcript``, ``bucket_llm``, ``bucket_final``,
  ``user_response``, ``embedding`` (BLOB bge-m3 1024d).
- ``rag_routing_rules`` — reglas promovidas desde el extractor. Columnas:
  ``pattern``, ``bucket``, ``evidence_count``, ``evidence_ratio``,
  ``promoted_at``, ``active``.

Sigue el patrón de ``rag_whisper_learning/`` (mismo nivel de abstracción,
mismo estilo de tests, mismo fallback silencioso ante DB ausente).
"""

from rag_routing_learning.patterns import (
    RoutingPattern,
    extract_pivot_phrases,
)
from rag_routing_learning.promote import (
    LearnedRule,
    deactivate_rule,
    list_active_rules,
    list_candidate_patterns,
    upsert_rule,
    render_rules_block,
)

__all__ = [
    "RoutingPattern",
    "LearnedRule",
    "extract_pivot_phrases",
    "upsert_rule",
    "deactivate_rule",
    "list_active_rules",
    "list_candidate_patterns",
    "render_rules_block",
]
