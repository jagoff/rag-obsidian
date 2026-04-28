"""Helpers para preparar outputs de tools antes de pasarlos al LLM
para síntesis en prosa.

Bug 2026-04-28: outputs grandes de gmail_recent/whatsapp_search/
drive_search disparaban timeouts (60-80s) en la 2da ronda del LLM.
Fix: truncar a top-N items + agregar summary count para que el LLM
sepa que hay más sin que tenga que procesarlos todos.

Bug 2026-04-28 wave-3 (UI test): `read_note("CLAUDE.md")` devolvió 165k
chars de markdown plano, el LLM con num_ctx=4096 truncó silenciosamente,
y respondió con headings INVENTADOS (alucinación pura). Fix: agregar
shape C "plain text large" — si el output no es JSON y excede 12k chars,
truncar al inicio + final con marker explícito.

API pública:
- `truncate_tool_output_for_synthesis(name, raw_json, max_items=N)` →
  retorna JSON string truncado (o el original si está bajo el cap).

El truncado es POR-TOOL: cada tool tiene una shape distinta.
- gmail_recent retorna `{awaiting_reply: [...], starred: [...],
  inbox_recent: [...]}` → truncar cada lista.
- whatsapp_search retorna `[chunk1, chunk2, ...]` (lista plana) →
  truncar.
- drive_search retorna `{files: [...]}` o lista plana → truncar.
- whatsapp_thread retorna lista de mensajes → truncar.
- whatsapp_search idem.
- read_note retorna markdown plano → truncar a ~12k chars con marker.

El helper es CONSERVADOR: si no reconoce la shape, devuelve el JSON
original sin tocar (no rompe nada).
"""
from __future__ import annotations

import json
from typing import Any


# Caps por tool. Más conservador para tools que devuelven mucho contenido
# textual (whatsapp_search) que para tools de metadata (calendar_ahead).
_DEFAULT_CAPS: dict[str, int] = {
    "gmail_recent": 5,         # antes podía ser 12+
    "whatsapp_search": 8,      # antes podía ser 15+
    "whatsapp_thread": 12,     # antes podía ser 30+
    "whatsapp_pending": 10,    # ya viene cap=10 default, redundante pero explícito
    "drive_search": 5,         # 5 archivos con body es razonable
    # Tools que ya devuelven shapes pequeñas no necesitan cap:
    # "calendar_ahead", "reminders_due", "weather", "finance_summary",
    # "credit_cards_summary".
}


# Cap en chars para outputs de plain-text grande (read_note). 12000 chars
# ≈ 9000 tokens, cabe holgado en num_ctx=4096 dejando margen para system
# prompt + retrieved context + respuesta. Si la nota excede, devolvemos
# inicio (8000) + corte explícito + final (2000) — el LLM ve los
# headings principales del top + la conclusión, evita alucinar.
_PLAIN_TEXT_TOOL_CHAR_CAP = 12000
_PLAIN_TEXT_TOOLS: set[str] = {"read_note"}


def _truncate_list(lst: list[Any], cap: int) -> tuple[list[Any], int]:
    """Recorta `lst` a `cap` items y retorna (lista_recortada, n_dropped)."""
    if not isinstance(lst, list):
        return lst, 0
    if len(lst) <= cap:
        return lst, 0
    return lst[:cap], len(lst) - cap


def truncate_tool_output_for_synthesis(
    tool_name: str,
    raw_output: str,
    *,
    max_items: int | None = None,
) -> str:
    """Truncate a tool's JSON output before passing to the LLM for synthesis.

    Args:
        tool_name: nombre de la tool que produjo el output (ej. "gmail_recent").
        raw_output: el string JSON que la tool devolvió.
        max_items: override del cap default. Si None, usa _DEFAULT_CAPS.

    Returns:
        JSON string. Si el output ya estaba bajo el cap o si la shape no
        es reconocida, devuelve `raw_output` sin tocar.

    El output truncado tiene un campo extra `_truncated_total: N` (donde
    N es el conteo original) si hubo truncado. Esto le da contexto al LLM
    para decir "tenés más mails — acá los 5 más recientes" en lugar de
    asumir que esos son todos.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        return raw_output

    # Shape C: plain-text tool (read_note). Truncar por chars con marker
    # explícito. Mantenemos head + tail para preservar headings (que están
    # arriba) y conclusiones (abajo). Marker informativo para que el LLM
    # sepa que hay contenido faltante.
    if tool_name in _PLAIN_TEXT_TOOLS:
        if len(raw_output) <= _PLAIN_TEXT_TOOL_CHAR_CAP:
            return raw_output
        head_len = int(_PLAIN_TEXT_TOOL_CHAR_CAP * 0.8)  # 9600 chars
        tail_len = _PLAIN_TEXT_TOOL_CHAR_CAP - head_len  # 2400 chars
        head = raw_output[:head_len]
        tail = raw_output[-tail_len:]
        omitted = len(raw_output) - head_len - tail_len
        return (
            f"{head}\n\n"
            f"[…CONTENIDO TRUNCADO: {omitted} caracteres omitidos del medio "
            f"de la nota. Total original: {len(raw_output)} chars. Si necesitás "
            f"el medio, pedí al user un fragmento específico…]\n\n"
            f"{tail}"
        )

    cap = max_items if max_items is not None else _DEFAULT_CAPS.get(tool_name)
    if cap is None:
        return raw_output  # tool no listada, no truncar

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output  # no es JSON válido, no tocar

    # Shape A: lista plana en la raíz.
    if isinstance(parsed, list):
        truncated, n_dropped = _truncate_list(parsed, cap)
        if n_dropped == 0:
            return raw_output
        # Devolver dict con metadata si era lista plana.
        return json.dumps({
            "items": truncated,
            "_truncated_total": len(parsed),
            "_kept": cap,
            "_dropped": n_dropped,
        }, ensure_ascii=False)

    # Shape B: dict con sub-listas (gmail_recent, drive_search, etc).
    if isinstance(parsed, dict):
        any_dropped = False
        result = dict(parsed)
        for key, val in list(parsed.items()):
            if isinstance(val, list):
                truncated, n_dropped = _truncate_list(val, cap)
                if n_dropped > 0:
                    any_dropped = True
                    result[key] = truncated
                    result.setdefault("_truncated", {})[key] = {
                        "total": len(val),
                        "kept": cap,
                        "dropped": n_dropped,
                    }
        if not any_dropped:
            return raw_output
        return json.dumps(result, ensure_ascii=False)

    return raw_output  # primitive shape, nothing to truncate
