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


def _wa_extract_combined(
    chat_label: str,
    is_group: bool,
    messages: list[dict],
) -> dict:
    """LLM-extract en UNA sola call: tasks + questions + commitments + promises.

    Pre-2026-05-09 había 2 calls separadas (`_wa_extract_actions` para
    tasks/questions/commitments + `_wa_extract_promises` para promises). Cada
    chat con activity disparaba 2 LLM calls — 12 chats = 24 calls/tick.
    Fusionado: 1 call/chat, ~50% latencia + tokens.

    Returns:
        ``{"tasks": [str], "questions": [str], "commitments": [str],
            "promises": [{text, when_text, direction, msg_id, msg_ts, speaker}]}``

    Silent-fail: empty dict default si LLM falla.
    """
    from rag import HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client
    # Re-resolve via package namespace para tests con monkeypatch.
    from rag.integrations.whatsapp import _has_promise_hint

    empty = {"tasks": [], "questions": [], "commitments": [], "promises": []}
    if not messages:
        return empty

    # Anti-loop drop U+200B
    candidates = [m for m in messages if not (m.get("text") or "").startswith("​")]
    if not candidates:
        return empty

    # Si NINGÚN msg matchea regex de promesa, igual hacemos la call (porque
    # tasks/questions/commitments no requieren ese gate). El LLM va a devolver
    # `promises: []` y listo. Mantener el gate solo para promesas era una
    # micro-optimización irrelevante post-fusión.
    has_any_promise_hint = any(_has_promise_hint(m.get("text") or "") for m in candidates)

    convo_lines: list[str] = []
    for m in candidates:
        ts = (m.get("ts") or "")[:16].replace("T", " ")
        mid = m.get("id") or m.get("msg_id") or "?"
        who = m.get("who") or "?"
        text = m.get("text") or ""
        # Para que el LLM pueda referenciar el msg_id en promises:
        convo_lines.append(f"[{ts}] [id:{mid}] {who}: {text}")
    convo = "\n".join(convo_lines)
    if len(convo) > 6000:
        convo = convo[-6000:]

    kind = "grupo" if is_group else "chat directo"
    promise_section = (
        "- promises: SOLO promesas — frases donde alguien se compromete a hacer "
        "algo en el futuro pero todavía NO lo hizo (\"te aviso\", \"te llamo "
        "mañana\", \"lo reviso en un rato\"). Para cada una: text, when_text "
        "(\"mañana\", \"en 2hs\", o \"\" si no especifica), direction (\"out\" "
        "si yo prometo a otro; \"in\" si otro promete a mí), msg_id (el id que "
        "aparece entre [id:XXX]). Si no hay promesas reales, lista vacía.\n"
    ) if has_any_promise_hint else (
        "- promises: lista vacía (no hay candidatos en esta ventana).\n"
    )

    prompt = (
        "IDIOMA: respondé SIEMPRE en español rioplatense (voseo argentino). "
        "Nunca uses portugués (\"você\", \"obrigado\", \"essa\"). Nunca uses tuteo "
        "peninsular. Si el chat tiene contexto BR, los items se destilan igual "
        "en español rioplatense.\n\n"
        f"Conversación de WhatsApp ({kind}): {chat_label}\n\n"
        f"{convo}\n\n"
        "Extraé items accionables reales y promesas para \"yo\" (el usuario). "
        "Sé conservador: si no está claro, omitilo. Ignorá saludos, memes, "
        "reacciones.\n\n"
        "- tasks: cosas que alguien le pidió a yo (hacer X, mandar Y).\n"
        "- questions: preguntas dirigidas a yo que aún no respondió.\n"
        "- commitments: cosas que yo prometió hacer en general.\n"
        + promise_section +
        "\nCada item de tasks/questions/commitments: frase corta en español "
        "rioplatense, 1 línea, sin nombre del chat ni timestamps. "
        "Si una categoría está vacía, lista vacía. Formato JSON estricto:\n"
        '{"tasks": ["..."], "questions": ["..."], "commitments": ["..."], '
        '"promises": [{"text": "...", "when_text": "...", "direction": "out", '
        '"msg_id": "..."}]}'
    )

    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # num_predict bumpeado: antes 320 (actions) + 480 (promises) = 800.
            # Ahora 1 call combinada — bump a 720 para no truncar JSON con
            # ambas secciones llenas.
            options={**HELPER_OPTIONS, "num_predict": 720, "num_ctx": 4096},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty

    out = {"tasks": [], "questions": [], "commitments": [], "promises": []}

    # tasks/questions/commitments — same dedup + cap como pre-fusión
    for key in ("tasks", "questions", "commitments"):
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

    # promises — validate + cross-check direction vs is_from_me real (Q10).
    msg_by_id: dict[str, dict] = {}
    for m in candidates:
        mid = m.get("id") or m.get("msg_id")
        if mid:
            msg_by_id[str(mid)] = m

    promises = data.get("promises") or []
    if isinstance(promises, list):
        seen_p: set[tuple[str, str]] = set()
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
            src = msg_by_id.get(msg_id) or {}

            # Q10: cross-check direction. Si el msg cited es is_from_me=True
            # y el LLM dijo direction="in", o vice-versa → el LLM hallucinó.
            # Auto-corregimos basándonos en is_from_me (verdad ground-truth
            # del bridge SQLite) en lugar de descartar la promesa entera.
            #
            # CAVEAT: solo aplicamos el cross-check si `is_from_me` está
            # presente en el src dict. En producción siempre viene del
            # bridge SQL. Tests pre-fusión usan _msg helper que no setea
            # is_from_me — confiamos en el LLM en ese caso para no romper
            # contratos históricos.
            if src and "is_from_me" in src:
                actual_is_from_me = bool(src.get("is_from_me"))
                inferred_direction = "out" if actual_is_from_me else "in"
                if direction != inferred_direction:
                    direction = inferred_direction

            dedup_key = (text.lower(), direction)
            if dedup_key in seen_p:
                continue
            seen_p.add(dedup_key)
            out["promises"].append({
                "text": text,
                "when_text": when_text,
                "direction": direction,
                "msg_id": msg_id,
                "msg_ts": src.get("ts") or "",
                "speaker": src.get("who") or ("yo" if direction == "out" else ""),
            })

    return out


def _wa_extract_actions(chat_label: str, is_group: bool, messages: list[dict]) -> dict:
    """Back-compat wrapper. Devuelve solo {tasks, questions, commitments}.

    A partir de 2026-05-09 el camino interno es ``_wa_extract_combined`` que
    extrae las 4 cosas en una sola LLM call. Este wrapper preserva la firma
    histórica para los call sites que solo querían action items (test code,
    etc.).
    """
    combined = _wa_extract_combined(chat_label, is_group, messages)
    return {
        "tasks": combined.get("tasks", []),
        "questions": combined.get("questions", []),
        "commitments": combined.get("commitments", []),
    }


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
    """Back-compat wrapper. Devuelve solo `promises` desde la combined call.

    A partir de 2026-05-09 el camino interno es ``_wa_extract_combined``.
    Preservado para tests / call sites históricos que solo quieren promesas.

    Para mantener back-compat exact con la versión pre-fusión: respeta el
    promise-hint regex gate. Si NO hay candidato con hint, return [] sin
    invocar el LLM. Esto preserva el contrato "no spendear LLM call si no
    hay nada que extraer" para callers que solo querían promesas.
    """
    if not messages:
        return []
    candidates = [m for m in messages if not (m.get("text") or "").startswith("​")]
    if not candidates:
        return []
    # Re-resolve via package namespace para tests con monkeypatch.
    from rag.integrations.whatsapp import _has_promise_hint
    if not any(_has_promise_hint(m.get("text") or "") for m in candidates):
        return []
    return _wa_extract_combined(chat_label, is_group, messages).get("promises", [])


__all__ = [
    "_PROMISE_REGEX_HINTS",
    "_wa_extract_combined",
    "_wa_extract_actions",
    "_has_promise_hint",
    "_parse_promise_when",
    "_wa_extract_promises",
]
