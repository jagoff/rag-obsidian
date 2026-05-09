"""WhatsApp tasks/promises LLM extraction.

Pipeline LLM-side del extractor `rag wa-tasks`:

- ``_wa_extract_actions(label, is_group, msgs)`` — qwen2.5:3b extrae
  ``{tasks, questions, commitments}`` de una ventana de chat. Conservative
  prompt: solo flagear items que un humano genuinamente actionaría.
- ``_PROMISE_REGEX_HINTS`` — pre-filter rioplatense conservador (~80%
  rejection rate) antes de gastar una LLM call por chat.
- ``_has_promise_hint(text)`` — bool gate.
- ``_parse_promise_when(when_text, anchor)`` — resuelve "mañana 10am" /
  "en 2hs" a (datetime, confidence). Reusa `_parse_natural_datetime`.
- ``_wa_extract_promises(label, is_group, msgs)`` — LLM-extract de
  promesas con ``direction in/out`` para tracking en `rag_promises`.

Invariantes:
- Silent-fail: cualquier excepción del LLM / JSON inválido → return empty.
  Callers tratan como "nothing to extract", no error.
- Determinismo via ``HELPER_OPTIONS`` (temperature=0, seed=42).
- Anti-loop: drop messages que arrancan con U+200B (son outputs nuestros).
- Cap LLM input a 6000 chars (tail) para acotar cost.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta


# ── Promise regex pre-filter ─────────────────────────────────────────────────
# Patrones rioplatenses comunes para detectar potenciales promesas. Lista
# CONSERVADORA — el LLM filtra después. La idea de este pre-filter es
# descartar el ~80% de msgs que claramente no son promesas (saludos, memes,
# fotos, reacciones), no clasificar con precisión. False-positives están
# bien (cuestan 1 LLM call por chat batch); false-negatives no — significa
# perder una promesa real.
_PROMISE_REGEX_HINTS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bdespu[eé]s\s+(te|le|les|lo|la)\s+(aviso|cuento|paso|escribo|llamo|mando|digo|confirmo)\b",
        r"\bte\s+(llamo|escribo|mando|aviso|paso|confirmo|cuento|digo)\b",
        r"\b(m[aá]s\s+tarde|al\s+rato|en\s+un\s+rato|en\s+un\s+toque)\b",
        r"\b(luego|despu[eé]s)\s+(te|lo|la)\s+(veo|reviso|hago|paso|miro|mando)\b",
        r"\b(ma[ñn]ana|pasado\s+ma[ñn]ana|el\s+lunes|el\s+martes|el\s+mi[eé]rcoles|el\s+jueves|el\s+viernes|el\s+s[aá]bado|el\s+domingo)\s+(te|lo|la)\b",
        r"\b(prometo|prometemos|comprometo)\b",
        r"\b(te\s+confirmo|te\s+digo)\s+(despu[eé]s|m[aá]s\s+tarde|en\s+un\s+rato|ma[ñn]ana)\b",
        r"\b(despu[eé]s|m[aá]s\s+tarde|luego)\s+(lo\s+vemos|arreglamos|charlamos|hablamos)\b",
        r"\ben\s+\d+\s*(min|minutos?|hs?|horas?|d[ií]as?)\b",
        r"\b(ahora|enseguida|en\s+(un\s+momento|nada|breve|seguida))\s+(te|lo|la)\b",
        r"\bvoy\s+a\s+(avisarte|escribirte|mandarte|pasarte|llamarte|confirmarte|decirte|contarte)\b",
        r"\bte\s+lo\s+(paso|mando|escribo|env[ií]o|confirmo)\b",
        r"\besta\s+(tarde|noche|ma[ñn]ana)\b",
        r"\blo\s+(reviso|veo|miro|chequeo)\s+y\s+te\s+(digo|aviso|confirmo|paso)\b",
        r"\bya\s+(te|vuelvo|lo|la)\b",
    )
)


def _wa_extract_actions(chat_label: str, is_group: bool, messages: list[dict]) -> dict:
    """LLM-extract action items from a chat window.

    Conservative prompt: only flag items a human would genuinely action.
    Returns ``{"tasks": [str], "questions": [str], "commitments": [str]}``
    (empty lists on LLM failure — callers treat as "nothing to extract",
    not as an error). Deterministic via HELPER_OPTIONS.

    `commitments` are things the user (yo) promised to do; `tasks` are
    asks directed at the user; `questions` are open questions addressed
    to the user that still need an answer.
    """
    from rag import HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client
    empty = {"tasks": [], "questions": [], "commitments": []}
    if not messages:
        return empty
    convo_lines: list[str] = []
    for m in messages:
        ts = (m["ts"] or "")[:16].replace("T", " ")
        convo_lines.append(f"[{ts}] {m['who']}: {m['text']}")
    convo = "\n".join(convo_lines)
    if len(convo) > 6000:
        convo = convo[-6000:]
    kind = "grupo" if is_group else "chat directo"
    prompt = (
        f"Conversación de WhatsApp ({kind}): {chat_label}\n\n"
        f"{convo}\n\n"
        "Extraé solo items accionables reales para \"yo\" (el usuario). "
        "Sé conservador: si no está claro que sea una acción, omitilo. "
        "Ignorá saludos, small talk, memes, reacciones.\n\n"
        "- tasks: cosas que alguien le pidió a yo (hacer X, mandar Y, revisar Z).\n"
        "- questions: preguntas dirigidas a yo que aún no respondió.\n"
        "- commitments: cosas que yo prometió hacer (\"te mando…\", \"mañana te paso…\").\n\n"
        "Cada item: frase corta en español, 1 línea, sin nombre del chat ni timestamps. "
        "Si no hay nada en una categoría, lista vacía. "
        "Formato estricto JSON: "
        "{\"tasks\": [\"...\"], \"questions\": [\"...\"], \"commitments\": [\"...\"]}"
    )
    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 320, "num_ctx": 4096},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    out = {"tasks": [], "questions": [], "commitments": []}
    for key in out:
        items = data.get(key) or []
        if not isinstance(items, list):
            continue
        seen: set[str] = set()
        for item in items[:10]:
            if not isinstance(item, str):
                continue
            clean = item.strip().strip("-•*").strip()
            if len(clean) < 4 or len(clean) > 240:
                continue
            key_norm = clean.lower()
            if key_norm in seen:
                continue
            seen.add(key_norm)
            out[key].append(clean)
    return out


# --- WhatsApp promise tracking (feat 2026-04-25) ---
#
# Detecta frases como "después te aviso", "te llamo mañana", "en un rato
# lo reviso" en chats de WhatsApp, las persiste en rag_promises, y delega
# el recordatorio al framework anticipate (signal pluggable
# `rag_anticipate/signals/promises.py`).
#
# Pipeline:
#   1. _has_promise_hint(text) → bool          [regex pre-filter, ~80% rejection]
#   2. _wa_extract_promises(...)               [LLM refina y devuelve estructurado]
#   3. _parse_promise_when(when_text)          [resolución a datetime concreto]
#   4. INSERT rag_promises status='pending'
#
# El extractor anti-loop: descarta msgs que empiecen con U+200B (ese marker
# lo agregamos en `_whatsapp_send_to_jid` para los reminders salientes; si
# los re-detectaramos como promesas se haría infinite loop).


def _has_promise_hint(text: str) -> bool:
    """Pre-filter de regex: True si el texto matchea algún patrón de promesa.

    Conservador — false positives están bien (el LLM filtra), pero false
    negatives significan que perdemos una promesa real. Para expandir,
    agregar el patrón a `_PROMISE_REGEX_HINTS` arriba.
    """
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _PROMISE_REGEX_HINTS)


def _parse_promise_when(when_text: str, anchor: datetime | None = None) -> tuple[datetime, float]:
    """Parse del 'cuándo' de una promesa a un (datetime, confidence) concreto.

    Reusa `_parse_natural_datetime` (que ya cubre 'mañana', 'en 2hs',
    'esta tarde', dateparser + LLM fallback) y le suma una capa fina:

    - confidence 0.9: parser nativo devolvió un datetime confiable.
    - confidence 0.3: when_text vacío o el parser falló → default = now+2h.
                      User decision (2026-04-25): preferís accountability
                      rápida sobre tiempo de gracia más largo.
    """
    from rag import _parse_natural_datetime
    base = anchor if anchor is not None else datetime.now()
    when = (when_text or "").strip()
    if not when:
        return (base + timedelta(hours=2), 0.3)
    try:
        parsed = _parse_natural_datetime(when, now=base)
    except Exception:
        parsed = None
    if parsed is None:
        return (base + timedelta(hours=2), 0.3)
    if parsed <= base:
        return (base + timedelta(hours=2), 0.3)
    return (parsed, 0.9)


def _wa_extract_promises(
    chat_label: str,
    is_group: bool,
    messages: list[dict],
) -> list[dict]:
    """LLM-extract promises from a chat window.

    Devuelve lista de promesas estructuradas:
        [{"text": "...", "when_text": "...", "direction": "out|in",
          "msg_id": "...", "msg_ts": "...", "speaker": "..."}, ...]

    Pipeline: anti-loop drop U+200B → regex pre-filter (skip LLM si
    nada matchea) → LLM con prompt determinista format=json → validate
    + dedup por (text, direction). Silent-fail por contrato: empty
    list on cualquier error (LLM, JSON inválido, etc).
    """
    from rag import HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client
    # Re-resolve `_has_promise_hint` via package namespace para que
    # tests con `monkeypatch.setattr(_waint, "_has_promise_hint", ...)`
    # propaguen.
    from rag.integrations.whatsapp import _has_promise_hint
    if not messages:
        return []
    candidates = [m for m in messages if not (m.get("text") or "").startswith("​")]
    if not candidates:
        return []
    if not any(_has_promise_hint(m.get("text") or "") for m in candidates):
        return []
    convo_lines: list[str] = []
    for m in candidates:
        ts = (m.get("ts") or "")[:16].replace("T", " ")
        mid = m.get("msg_id") or "?"
        who = m.get("who") or "?"
        text = m.get("text") or ""
        convo_lines.append(f"[{ts}] [id:{mid}] {who}: {text}")
    convo = "\n".join(convo_lines)
    if len(convo) > 6000:
        convo = convo[-6000:]
    kind = "grupo" if is_group else "chat directo"
    prompt = (
        f"Conversación de WhatsApp ({kind}): {chat_label}\n\n"
        f"{convo}\n\n"
        "Extraé SOLO las PROMESAS — frases donde alguien se compromete a "
        "hacer algo en el futuro pero todavía NO lo hizo.\n\n"
        "Ejemplos:\n"
        "  ✓ \"después te aviso\" → promesa, when_text=\"\"\n"
        "  ✓ \"te llamo mañana 10am\" → promesa, when_text=\"mañana 10am\"\n"
        "  ✓ \"en un rato lo reviso\" → promesa, when_text=\"en un rato\"\n"
        "  ✗ \"te avisé ayer\" → ya pasó, NO es promesa\n"
        "  ✗ \"siempre te aviso\" → general, NO es promesa concreta\n"
        "  ✗ \"buenas\" / \"jaja\" / fotos → small talk, NO\n\n"
        "Para cada promesa devolvé:\n"
        "  - text: la frase exacta (1 línea, sin nombres ni timestamps)\n"
        "  - when_text: el \"cuándo\" si lo dice (\"mañana\", \"en 2hs\", "
        "\"esta tarde\"). Vacío \"\" si no especifica.\n"
        "  - direction: \"out\" si YO (el usuario) prometo a otro;\n"
        "               \"in\" si OTRO promete a mí (el usuario)\n"
        "  - msg_id: el id que aparece en [id:XXX] en la línea original\n\n"
        "Si no hay promesas reales, devolvé lista vacía. Formato JSON estricto:\n"
        '{"promises": [{"text": "...", "when_text": "...", "direction": "out", "msg_id": "..."}]}'
    )
    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 480, "num_ctx": 4096},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    promises = data.get("promises") or []
    if not isinstance(promises, list):
        return []
    msg_by_id: dict[str, dict] = {}
    for m in candidates:
        mid = m.get("msg_id")
        if mid:
            msg_by_id[str(mid)] = m
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for p in promises[:20]:
        if not isinstance(p, dict):
            continue
        text = (p.get("text") or "").strip()
        if not (4 <= len(text) <= 240):
            continue
        direction = (p.get("direction") or "").strip().lower()
        if direction not in ("in", "out"):
            continue
        when_text = (p.get("when_text") or "").strip()
        msg_id = str(p.get("msg_id") or "")
        dedup_key = (text.lower(), direction)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        src = msg_by_id.get(msg_id) or {}
        out.append({
            "text": text,
            "when_text": when_text,
            "direction": direction,
            "msg_id": msg_id,
            "msg_ts": src.get("ts") or "",
            "speaker": src.get("who") or ("yo" if direction == "out" else ""),
        })
    return out


__all__ = [
    "_PROMISE_REGEX_HINTS",
    "_wa_extract_actions",
    "_has_promise_hint",
    "_parse_promise_when",
    "_wa_extract_promises",
]
