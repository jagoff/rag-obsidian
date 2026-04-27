"""Subpackage del feature **WhatsApp Negotiation Auto-Pilot** — Fase 0
(Foundation).

El feature completo (8 fases, ~10 semanas) lo describe la spec en
`04-Archive/99-obsidian-system/99-AI/system/wa-negotiation-autopilot/
design.md`. Este subpackage entrega **solo la capa de datos** que las
otras fases enchufan encima:

- `state_machine` — grafo de transiciones legales entre los 8 estados
  de una negociación (4 terminales). Puro Python, sin DB.
- `crud` — helpers thin sobre las 5 tablas SQL nuevas
  (`rag_negotiations`, `rag_negotiation_turns`,
  `rag_negotiation_pending_sends`, `rag_style_fingerprints`,
  `rag_behavior_priors_wa`). Usan `rag._ragvec_state_conn()` como
  context manager — el patrón establecido por `rag_routing_learning`,
  `rag_whisper_learning`, etc.

Lo que **NO** está en Fase 0 (los siguientes sprints lo van a montar
encima):

- F1 — Real-time learning loop (watcher fsevents + fingerprint refresh).
- F2 — Confidence classifier (LLM call por turno).
- F3 — Pause simulator + orchestrator daemon.
- F4 — UI lanzamiento PWA (pre-launch screen + endpoint `/api/negotiation/start`).
- F5 — Dashboard panel "Negociaciones" + CLI `rag negotiation trace <id>`.
- F6 — Self-DM + voz como surfaces alternativos.
- F7 — Continuous improvement (threshold tuning automático).
- F8 — Estabilización (per-contact tuning + reportes mensuales).

Cada función pública del subpackage está comentada con qué fase
posterior la consume. El objetivo es que cuando arranquemos F1 ó F3
sepamos exactamente cuál es la API que quedó committed.
"""
from __future__ import annotations

from rag_negotiations.crud import (
    append_turn,
    create_negotiation,
    dequeue_due,
    enqueue_send,
    get_behavior_priors,
    get_negotiation,
    get_style_fingerprint,
    increment_message_count,
    list_negotiations,
    list_turns,
    mark_send,
    update_status,
    upsert_behavior_priors,
    upsert_style_fingerprint,
)
from rag_negotiations.state_machine import (
    CLOSED_OK_STATES,
    InvalidTransitionError,
    STATE_ORDER,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    can_transition,
    is_terminal,
    legal_transitions,
    transition,
)


__all__ = [
    # state_machine
    "CLOSED_OK_STATES",
    "InvalidTransitionError",
    "STATE_ORDER",
    "TERMINAL_STATES",
    "VALID_TRANSITIONS",
    "can_transition",
    "is_terminal",
    "legal_transitions",
    "transition",
    # crud
    "append_turn",
    "create_negotiation",
    "dequeue_due",
    "enqueue_send",
    "get_behavior_priors",
    "get_negotiation",
    "get_style_fingerprint",
    "increment_message_count",
    "list_negotiations",
    "list_turns",
    "mark_send",
    "update_status",
    "upsert_behavior_priors",
    "upsert_style_fingerprint",
]
