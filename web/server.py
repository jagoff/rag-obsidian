"""Web UI mínima para `rag chat`.

Espeja el pipeline del CLI (multi_retrieve → command-r streaming → sources)
sobre HTTP + SSE. Sin build step en el frontend; vanilla JS contra este
endpoint. Sesiones persistidas en el mismo store que el CLI — el session_id
de la web es `web:<uuid>` así no colisiona con `tg:<chat_id>` ni con los
ids del chat interactivo.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# rag.py vive en el root del proyecto; lo importamos como módulo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama  # noqa: E402

from rag import (  # noqa: E402
    CHAT_OPTIONS,
    CONTRADICTION_LOG_PATH,
    EVAL_LOG_PATH,
    LOG_PATH,
    _LOG_QUEUE,
    RAG_STATE_SQL,
    _log_sql_state_error,
    _map_cpu_row,
    _map_memory_row,
    _ragvec_state_conn,
    _sql_append_event,
    MORNING_FOLDER,
    OLLAMA_KEEP_ALIVE,
    SESSION_HISTORY_WINDOW,
    _collect_screentime,
    _fmt_hm,
    VAULT_PATH,
    WHATSAPP_BOT_JID,
    WHATSAPP_DB_PATH,
    _apple_enabled,
    _build_tasks_system_rules as _rag_build_tasks_system_rules,
    _collect_scoped_tasks_evidence_multi as _rag_collect_scoped_tasks_evidence_multi,
    _collect_today_evidence,
    _fetch_calendar_ahead,
    _fetch_chrome_bookmarks_used,
    _fetch_drive_evidence,
    _fetch_reminders_due,
    _fetch_vault_activity,
    _fetch_weather_forecast,
    _fetch_whatsapp_unread,
    _format_scoped_tasks_context as _rag_format_scoped_tasks_context,
    _generate_today_narrative,
    _icalbuddy_path,
    _load_corpus,
    _load_vaults_config,
    _path_to_title,
    _pendientes_collect,
    _pendientes_urgent,
    _render_today_prompt,
    _tasks_services_consulted as _rag_tasks_services_consulted,
    append_turn,
    ensure_session,
    find_followup_loops,
    get_db,
    get_pagerank,
    log_behavior_event,
    log_query_event,
    multi_retrieve,
    new_turn_id,
    record_feedback,
    resolve_chat_model,
    resolve_vault_paths,
    save_session,
    session_history,
)

from web.conversation_writer import write_turn, TurnData  # noqa: E402
from web.tools import (  # noqa: E402
    CHAT_TOOLS,
    CHAT_TOOL_OPTIONS,
    PARALLEL_SAFE,
    PROPOSAL_TOOL_NAMES,
    TOOL_FNS,
    _WEB_TOOL_ADDENDUM,
)


# Pre-router: keyword → tool forced execution. Defeats the system prompt's
# "engancháte SIEMPRE con el CONTEXTO" bias which made the LLM decline to
# call tools when retrieval already returned anything (even low-conf vault
# notes). Regex-based, O(len(q)), runs before the LLM sees the query.
#
# Matches are independent — "gastos de este mes en agenda" fires both
# finance_summary and calendar_ahead.
# Patterns covering "planning" queries — "cómo viene mi semana", "qué
# tengo hoy", "qué hay mañana". These are ambiguous enough that we fire
# BOTH reminders_due AND calendar_ahead (via the shared pattern below)
# so the LLM has tasks + events to build a picture of the timeframe.
_PLANNING_PAT = (
    r"\bsemana\b|\bhoy\b|\bma[ñn]ana\b|pasado\s+ma[ñn]ana|\bd[ií]a\b|c[oó]mo\s+viene"
    # "qué tengo / hay / tenés" only counts as planning when followed by a
    # temporal token (hoy, mañana, semana, día, agenda, pendiente) so that
    # "qué tengo sobre coaching" doesn't fire calendar + reminders.
    r"|qu[eé]\s+(tengo|hay|ten[eé]s)\b(?=.{0,40}\b(hoy|ma[ñn]ana|semana|d[ií]a|agenda|pendient|tarea|recordator)\b)"
)

_TOOL_INTENT_RULES: tuple[tuple[str, dict, str], ...] = (
    ("finance_summary", {}, r"gast[oéó]s?|gast[aá][mn]os|gastar|presupuesto|plata|finanz|moze"),
    # NOTE: "recordame" / "agendáme" used to be here as query triggers, but
    # they're CREATE intents now (propose_reminder / propose_calendar_event).
    # Moved to `_PROPOSE_INTENT_RE` below. `recordator` still matches
    # "qué recordatorios tengo" → list.
    ("reminders_due",   {}, r"pendient|tarea|to.?do|recordator|" + _PLANNING_PAT),
    # "inbox" / "bandeja" alone are too generic — Obsidian PARA uses
    # "00-Inbox" heavily. Require an explicit mail signal.
    ("gmail_recent",    {}, r"\b(mail|correo|e.?mail|gmail)\b|bandeja\s+de\s+entrada"),
    ("calendar_ahead",  {}, r"calendari|\bevento\b|\bcita\b|reuni[oó]n|\bagenda\b|pr[oó]xim[ao]s?\s+d[ií]as|" + _PLANNING_PAT),
    ("weather",         {}, r"\bclima\b|\btiempo\b|llov|lluvia|temperatur|pron[oó]stico"),
)
_TOOL_INTENT_COMPILED = tuple(
    (name, args, re.compile(pat, re.IGNORECASE)) for name, args, pat in _TOOL_INTENT_RULES
)


def _detect_tool_intent(q: str) -> list[tuple[str, dict]]:
    """Deterministic keyword → tool routing. Returns (name, args) tuples
    to execute BEFORE the LLM tool-deciding call. Empty list = no forced
    tools (LLM decides freely)."""
    if not q:
        return []
    return [(name, dict(args)) for name, args, rx in _TOOL_INTENT_COMPILED if rx.search(q)]


# Create-intent detection moved to rag.py so both the web chat endpoint
# and the CLI `rag chat` loop can share it without inverting the
# web → rag import direction. See rag.py `_detect_propose_intent` for
# the full regex list + three-branch logic.
from rag import _detect_propose_intent, _detect_metachat_intent  # noqa: E402


# Canned replies for the meta-chat short-circuit. Buckets keyed by the
# class of input; within a bucket we pick one variant by hashing the
# message + the current minute so the same phrase in a tight window is
# stable (no user surprise if they re-send) but repeat visits pick
# different variants (feels alive, not scripted).
_METACHAT_GREETING = (
    "¡Hola! Preguntame lo que quieras sobre tus notas, o decime *recordame …* / *agendá …* si querés crear algo.",
    "Hola 👋 ¿en qué te ayudo? Probá una pregunta sobre tus notas o pedime *recordame …* / *agendá …*.",
    "¡Buenas! Tirame una pregunta, o decime *recordame X* / *el viernes 20hs X* para crear un recordatorio o evento.",
)
_METACHAT_THANKS = (
    "¡De nada!",
    "¡Cuando quieras!",
    "👌",
)
_METACHAT_META = (
    "Puedo responder sobre tus notas, crear recordatorios (*recordame X*) y eventos de calendar (*el viernes 20hs …*). Probá algo.",
    "Consulto tu vault de Obsidian y creo recordatorios / eventos desde texto libre. ¿Qué necesitás?",
    "Leo tus notas y armo recordatorios o eventos cuando se los pedís en lenguaje natural. Tirame algo concreto.",
)
_METACHAT_BYE = (
    "¡Hasta luego!",
    "¡Nos vemos!",
    "👋",
)


def _metachat_bucket(q: str) -> tuple[str, ...]:
    """Classify the meta-chat message into a response bucket."""
    s = q.strip().lower().lstrip("¿ ").lstrip()
    if s.startswith(("gracias", "muchas gracias", "mil gracias",
                     "dale gracias", "thanks", "thx")):
        return _METACHAT_THANKS
    if s.startswith(("chau", "bye", "adiós", "adios", "nos vemos")):
        return _METACHAT_BYE
    if s.startswith(("qué podés", "que podes", "qué sabés", "que sabes",
                     "cómo funcion", "como funcion", "cómo te us",
                     "como te us", "qué comandos", "que comandos",
                     "ayuda", "help", "quién sos", "quien sos",
                     "quién es este", "quien es este")):
        return _METACHAT_META
    return _METACHAT_GREETING


def _pick_metachat_reply(q: str, *, now: float | None = None) -> str:
    """Pick a canned reply for a meta-chat turn.

    Variation seed = hash(q) XOR minute-bucket. Same input within the
    same minute returns the same variant (stable on retry); different
    inputs or different minutes rotate. Tests monkey-patch with fixed
    `now` for determinism.
    """
    import hashlib
    import time as _time
    bucket = _metachat_bucket(q)
    ts = now if now is not None else _time.time()
    minute = int(ts // 60)
    seed = int(hashlib.sha256(f"{q}|{minute}".encode()).hexdigest()[:8], 16)
    return bucket[seed % len(bucket)]


# Post-stream filter enforcing REGLA 0 at the byte level. qwen2.5:7b and
# command-r leak CJK / Cyrillic / Arabic tokens under context pressure; the
# chat path strips them from every streamed delta so the client never sees
# them regardless of what the LLM emits. Whitelists ASCII, Latin (including
# Spanish diacritics and the Latin-Extended blocks), box-drawing, general
# punctuation used in markdown, and everything ≥ U+1F000 (emoji + symbols).
# Kills CJK ideographs + kana + hangul, Cyrillic, Hebrew, Arabic, Syriac,
# and the fullwidth/halfwidth CJK punctuation block.
_FOREIGN_SCRIPT_RE = re.compile(
    "["
    "\u0400-\u052f"     # Cyrillic + Cyrillic Supplement
    "\u0530-\u058f"     # Armenian
    "\u0590-\u05ff"     # Hebrew
    "\u0600-\u06ff"     # Arabic
    "\u0700-\u074f"     # Syriac
    "\u0750-\u077f"     # Arabic Supplement
    "\u0780-\u07bf"     # Thaana
    "\u3000-\u303f"     # CJK Symbols and Punctuation (includes 。、)
    "\u3040-\u309f"     # Hiragana
    "\u30a0-\u30ff"     # Katakana
    "\u3400-\u4dbf"     # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"     # CJK Unified Ideographs
    "\uac00-\ud7af"     # Hangul Syllables
    "\uff00-\uffef"     # Halfwidth and Fullwidth Forms
    "]"
)


def _strip_foreign_scripts(text: str) -> str:
    """Remove characters from non-allowed scripts (CJK, Cyrillic, Hebrew,
    Arabic, …). Preserves Spanish diacritics, ASCII, markdown punctuation,
    and emoji (≥U+1F000). Safe to call per-token on a streaming delta —
    characters are dropped individually (no stateful lookahead)."""
    if not text:
        return text
    return _FOREIGN_SCRIPT_RE.sub("", text)


def _own_conversation_path(session_id: str) -> str | None:
    """Look up the episodic conversation .md for `session_id` in
    rag_conversations_index. Used for the self-citation exclusion guard in
    /api/chat. SQL-only since T10 (the old conversations_index.json path is
    gone)."""
    try:
        from web.conversation_writer import get_conversation_path
        return get_conversation_path(session_id)
    except Exception:
        return None


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="obsidian-rag web", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CORS: same-origin only. The server is bound to 127.0.0.1 by the
# launchd plist, so cross-origin requests would come from a browser
# running an untrusted origin (e.g. a malicious local webpage trying
# to hit our API). FastAPI without CORSMiddleware rejects cross-origin
# by default, but adding the middleware explicitly:
#   1. documents the intent for future maintainers who might otherwise
#      assume CORS wasn't thought about,
#   2. whitelists the exact origins we DO serve (127.0.0.1 / localhost
#      on any port — the plist uses 8765 but dev sometimes uses 8766),
#   3. prevents a future contributor from dropping in
#      `CORSMiddleware(allow_origins=["*"])` thinking "we need CORS",
#      which would open the API to every page in the browser.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402 — must go after `app = FastAPI(...)`

app.add_middleware(
    CORSMiddleware,
    # Regex matches localhost / 127.0.0.1 with any port or no port.
    # Refuses anything else, including file:// and 0.0.0.0.
    allow_origin_regex=r"^http://(127\.0\.0\.1|localhost)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Chat model for /chat. Delegated to `resolve_chat_model()` so it tracks
# whatever rag.py decides is best on this host (command-r > qwen2.5:14b >
# phi4). command-r:35b prefill is slower (~5-10s on a 5k-char context)
# but is the only model in the local set that reliably:
#   - stays in Spanish instead of drifting to Portuguese / Italian
#   - resolves pronouns from conversation history ("ella", "eso") against
#     prior turns instead of treating follow-ups as standalone
#   - follows the "no inline citations / no refusal" rules of this prompt
# qwen2.5:3b was used here for a while as a speed optimization (~600ms
# prefill, ~10× faster) but failed those three constraints consistently.
# Set OBSIDIAN_RAG_WEB_CHAT_MODEL to override if you need speed over quality.
WEB_CHAT_MODEL = os.environ.get("OBSIDIAN_RAG_WEB_CHAT_MODEL") or None

# Runtime model override — persisted to disk so it survives server restarts.
# Takes priority over env / resolve_chat_model() so the user can A/B different
# models from the UI without editing the launchd plist or redeploying.
# Format: {"model": "qwen3.6", "set_at": "2026-04-20T15:00:00"}. Absent or
# malformed → no override active.
_CHAT_MODEL_OVERRIDE_PATH = Path.home() / ".local/share/obsidian-rag" / "chat-model.json"


def _read_chat_model_override() -> str | None:
    """Load the persistent runtime override if any. Silent on any error —
    a corrupt or missing file means "no override, use defaults"."""
    try:
        raw = _CHAT_MODEL_OVERRIDE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        model = (data or {}).get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def _write_chat_model_override(model: str | None) -> None:
    """Persist or clear the runtime override. None → delete the file."""
    _CHAT_MODEL_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        _CHAT_MODEL_OVERRIDE_PATH.unlink(missing_ok=True)
        return
    payload = {"model": model,
               "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = _CHAT_MODEL_OVERRIDE_PATH.with_suffix(_CHAT_MODEL_OVERRIDE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _CHAT_MODEL_OVERRIDE_PATH)


def _resolve_web_chat_model() -> str:
    """Model resolution priority (first match wins):
      1. Runtime override file (_CHAT_MODEL_OVERRIDE_PATH) — set via
         POST /api/chat/model from the UI; persists across restarts.
      2. OBSIDIAN_RAG_WEB_CHAT_MODEL env var — for launchd plist overrides.
      3. rag.resolve_chat_model() — the CHAT_MODEL_PREFERENCE fallback chain.
    Not cached: the override file is cheap to re-read (JSON, <200 bytes)
    and we want runtime flips to take effect on the next /api/chat call
    with no server restart.
    """
    override = _read_chat_model_override()
    if override:
        return override
    return WEB_CHAT_MODEL or resolve_chat_model()


# Module-level system prompt. Kept BYTE-IDENTICAL across all /api/chat
# requests so ollama's prefix cache hits — when the system message tokens
# match a prior request, llama.cpp skips KV recomputation for those ~900
# tokens. Cached prefill is ~50× faster (0.23ms/tok vs 12.7ms/tok measured
# on command-r:35b), saving up to 10s on the first user-facing token.
# Any per-request variation (context, signals, history) goes in the user
# message AFTER this static block.
# v1: versión original 5231 chars (~1307 tok). Preservada como fallback
# documentado vía env var RAG_WEB_PROMPT_VERSION=v1 — útil si al medir
# v2 en producción aparece regresión en algún caso no cubierto por el
# bench. Retirar cuando v2 sume suficiente historia sin incidentes.
_WEB_SYSTEM_PROMPT_V1 = (
    "Eres un asistente de consulta sobre las notas personales de "
    "Obsidian del usuario. NO sos un modelo de conocimiento general.\n\n"
    "REGLA 0 — IDIOMA: respondé SIEMPRE en español rioplatense. "
    "TOTALMENTE PROHIBIDO emitir tokens en chino, japonés, coreano, "
    "árabe, ruso, alemán, portugués, italiano, francés o cualquier "
    "idioma que no sea español. Si el contexto recuperado contiene "
    "fragmentos en otros idiomas (ej. citas en inglés, nombres "
    "propios, código), citalos textualmente entre comillas pero el "
    "resto de tu respuesta TIENE QUE estar en español. Si la "
    "pregunta del usuario está en otro idioma, traducila a español "
    "mentalmente y respondé en español. Esta regla es ABSOLUTA — "
    "ni siquiera caracteres sueltos en otro alfabeto (汉字, "
    "русский, etc.) están permitidos en tu output.\n\n"
    "REGLA 1 — ENGÁNCHATE CON EL CONTEXTO QUE RECIBÍS:\n"
    "  • El CONTEXTO abajo es lo que el retriever consideró más "
    "relacionado con la pregunta. Tu trabajo es leerlo y resumir lo "
    "que aporta, siempre. Incluso si los chunks son breves, parciales "
    "o tangenciales — engancháte con ellos.\n"
    "  • Preguntas tipo '¿tengo algo sobre X?' / '¿qué hay de X?' / "
    "'¿tenés notas para X?' se responden afirmativamente apenas el "
    "CONTEXTO mencione X en título o cuerpo. Listá brevemente lo que "
    "hay.\n"
    "  • Si el CONTEXTO es genuinamente pobre o tangencial para lo "
    "que se pregunta, describí lo que sí aparece y aclaralo en prosa "
    "('las notas que encontré mencionan X pero no detallan Y'). "
    "NUNCA uses fórmulas de refusal del tipo 'no tengo información' "
    "— esa vía de escape está PROHIBIDA en este asistente, siempre "
    "hay que devolver el mejor resumen posible del CONTEXTO.\n"
    "  • Fuera del CONTEXTO no inventes (eso queda para REGLA 3 con "
    "su marcador `<<ext>>`).\n\n"
    "REGLA 2 — NO CITAR NOTAS INLINE: la UI ya muestra la lista de "
    "fuentes recuperadas debajo de cada respuesta (nota, score, "
    "ruta). Por lo tanto TOTALMENTE PROHIBIDO escribir:\n"
    "  • markdown links tipo `[Título](ruta.md)`\n"
    "  • nombres de archivo con extensión (`algo.md`)\n"
    "  • rutas con carpetas PARA (`03-Resources/…`, `02-Areas/…`, etc.)\n"
    "  • el nombre completo de la nota como título dentro de la prosa\n"
    "Podés referirte implícitamente: 'según tus notas', 'en tu nota "
    "sobre X', 'tenés una nota al respecto'. Todo lo demás se ve en "
    "la lista de fuentes — no la dupliques en el cuerpo.\n\n"
    "REGLA 3 — MARCAR EXTERNO: si agregás texto que NO sale "
    "textualmente del contexto (intros, parafraseos, conectores, "
    "opinión, conocimiento general), envolvelo en `<<ext>>...<</ext>>`. "
    "Fuera de esos marcadores TODO debe ser verificable palabra por "
    "palabra en el contexto.\n\n"
    "REGLA 4 — FORMATO: 2-4 oraciones o lista corta de viñetas. Dato "
    "clave en la primera oración; contexto mínimo (qué hace, cómo se "
    "invoca) después. Si la pregunta apunta a un comando, "
    "herramienta, parámetro o valor puntual Y el CONTEXTO incluye "
    "su uso (firma, parámetros, ejemplo, en qué MCP/server vive), "
    "ese uso es OBLIGATORIO en la respuesta — no te quedes en un "
    "token mínimo cuando el chunk lo explica.\n\n"
    "REGLA 4.5 — PRESERVAR LINKS DEL CONTENIDO: si el CONTEXTO "
    "contiene URLs (http://, https://) o wikilinks ([[Nota]]) que "
    "viven DENTRO del cuerpo de una nota (no son la ruta del chunk), "
    "copialos LITERAL en la respuesta — son clickeables y útiles para "
    "el usuario. Aclaración respecto a REGLA 2: REGLA 2 prohíbe "
    "citar las notas-fuente (la lista de abajo ya las muestra). Los "
    "links/URLs que aparecen como contenido de una nota son data, "
    "no citas; tienen que aparecer en la respuesta. Ejemplo: si un "
    "chunk dice 'tutorial: https://x.com/y', escribí 'tutorial: "
    "https://x.com/y', no 'tutorial disponible'.\n\n"
    "REGLA 4.6 — LINK A DOCS OFICIALES (opcional, condicionado): si el "
    "CONTEXTO sobre una herramienta/producto se queda corto Y la "
    "pregunta se beneficiaría de docs oficiales (ej: 'cómo configuro X', "
    "'qué features tiene Y'), podés ofrecer UN link a la documentación "
    "oficial al final, envuelto en `<<ext>>...<</ext>>`. Restricciones "
    "duras:\n"
    "  • Sólo el dominio raíz canónico (ej: https://omnifocus.com, "
    "https://obsidian.md). NUNCA inventes paths profundos "
    "(/docs/v3/foo) — si no estás 100% seguro de la URL exacta, usá "
    "el root.\n"
    "  • Una sola línea, formato: `<<ext>>Más info: <URL></ext>>`.\n"
    "  • NO ofrecer link si el CONTEXTO ya cubre la pregunta, ni para "
    "consultas sobre las notas mismas (vault, tags, búsqueda).\n\n"
    "REGLA 5 — SEGUÍ EL HILO: esta es una conversación, no preguntas "
    "sueltas. Los mensajes previos (user/assistant de arriba) son "
    "contexto vivo. Si la pregunta nueva usa pronombres ('ella', "
    "'eso', 'ahí'), referencias elípticas ('y de X?', 'profundizá'), "
    "o asume un tema del turn anterior, resolvé la referencia usando "
    "el historial y respondé sobre ESE tema. No trates la pregunta "
    "como si empezara de cero.\n\n"
    "REGLA 6 — TRATAMIENTO: le hablás DIRECTAMENTE al usuario en 2da "
    "persona singular, tuteo rioplatense ('vos', 'tu', 'tenés', 'te'). "
    "El usuario ES la persona que hace la pregunta — no es un tercero "
    "del que hablás. TOTALMENTE PROHIBIDO escribir 'el usuario', 'del "
    "usuario', 'le' refiriéndose al usuario, ni describirlo en 3ra "
    "persona. Traducí automáticamente: 'la hija del usuario' → 'tu "
    "hija'; 'las notas del usuario' → 'tus notas'; 'el proyecto X "
    "del usuario' → 'tu proyecto X'. Si una nota dice 'Grecia es la "
    "hija de Fernando', y Fernando es el usuario, escribí 'Grecia es "
    "tu hija'.\n"
)

# v2: comprimido 2898 chars (~724 tok), −44% tokens. Mismas REGLAs
# preservando los anchors visuales que qwen reconoce como patrón
# (`<<ext>>...<</ext>>`, `[Título](ruta.md)`, `03-Resources/`, `[[Nota]]`).
# Medido 2026-04-20: prefill cae 1737ms → ~1100ms en el bench A/B
# gracias al ahorro de ~600 tok del system prompt.
_WEB_SYSTEM_PROMPT_V2 = 'Eres un asistente de consulta sobre las notas personales de Obsidian del usuario. NO sos un modelo de conocimiento general.\n\nREGLA 0 — IDIOMA: respondé SIEMPRE en español rioplatense. PROHIBIDO emitir tokens en otros idiomas o alfabetos (汉字, русский, etc.); caracteres fuera del alfabeto latino sólo se permiten dentro de una cita literal entre comillas. Si la pregunta viene en otro idioma, traducila y respondé en español.\n\nREGLA 1 — ENGANCHÁTE CON EL CONTEXTO: el CONTEXTO de abajo es lo que el retriever consideró más cercano. Resumí SIEMPRE lo que aporta, aun si es breve o tangencial. Preguntas tipo "¿tengo algo sobre X?" se responden afirmativo apenas X aparezca en título o cuerpo — listá brevemente. Si el CONTEXTO es pobre, describí lo que sí aparece ("las notas mencionan X pero no detallan Y"). PROHIBIDO refusal tipo "no tengo información" — siempre devolvé el mejor resumen posible del CONTEXTO. Fuera del CONTEXTO no inventes (ver REGLA 3).\n\nREGLA 2 — NO CITAR NOTAS INLINE: la UI ya muestra la lista de fuentes (nota, score, ruta) debajo. PROHIBIDO markdown links `[Título](ruta.md)`, nombres con extensión (`algo.md`), rutas PARA (`03-Resources/…`, `02-Areas/…`) ni el título completo como header. Referencias implícitas OK: "según tus notas", "en tu nota sobre X".\n\nREGLA 3 — MARCAR EXTERNO: texto que NO salga literal del CONTEXTO (parafraseo, conectores, opinión, conocimiento general) va envuelto en `<<ext>>...<</ext>>`. Fuera de esos marcadores todo debe ser verificable en el CONTEXTO.\n\nREGLA 4 — FORMATO: 2-4 oraciones o lista corta. Dato clave primero, contexto mínimo (qué hace, cómo se invoca) después. Si piden un comando, herramienta o parámetro Y el CONTEXTO tiene su uso (firma, ejemplo, en qué MCP vive), ese uso es OBLIGATORIO en la respuesta.\n\nREGLA 4.5 — PRESERVAR LINKS DEL CONTENIDO: URLs (http://, https://) y wikilinks ([[Nota]]) que vivan DENTRO del cuerpo de una nota son data, no citas-fuente — copialos LITERAL. REGLA 2 sólo prohíbe citar la ruta del chunk; los links internos son clickeables.\n\nREGLA 4.6 — LINK A DOCS OFICIALES (opcional): si el CONTEXTO sobre una herramienta se queda corto, ofrecé UN link al dominio raíz canónico (https://omnifocus.com, NUNCA paths profundos) al final, una sola línea: `<<ext>>Más info: <URL></ext>>`. No lo ofrezcas si el CONTEXTO ya cubre ni en consultas sobre el vault (tags, búsqueda, notas).\n\nREGLA 5 — SEGUÍ EL HILO: es una conversación. Pronombres ("ella", "eso"), referencias elípticas ("y de X?", "profundizá") o temas asumidos se resuelven con los turns previos. No trates la pregunta como si empezara de cero.\n\nREGLA 6 — TRATAMIENTO: hablale DIRECTAMENTE al usuario en 2da persona, tuteo rioplatense ("vos", "tenés", "te"). El usuario ES quien pregunta. PROHIBIDO 3ra persona ("el usuario", "la hija del usuario", "le"). Traducí: "la hija del usuario" → "tu hija"; "las notas del usuario" → "tus notas".'

# Selector con fallback seguro a v1 si el env var toma un valor raro.
_WEB_SYSTEM_PROMPT = (
    _WEB_SYSTEM_PROMPT_V1
    if os.environ.get("RAG_WEB_PROMPT_VERSION", "v2").strip() == "v1"
    else _WEB_SYSTEM_PROMPT_V2
)

# Turn-scoped override appended when `_detect_propose_intent` fires.
# Neutralises REGLA 1 (engancháte con CONTEXTO) + REGLA 2/3/4.5 (citar
# notas, preservar links) which don't apply to create-intent turns —
# there's no CONTEXTO attached, there are no notes to cite, there are
# no sources. Without this override qwen2.5:7b reverts to its "resumí
# lo que el retriever trajo" default and hallucinates a path like
# "04-Archive/Calendar.md" to open instead of calling propose_*.
#
# Critical detail learned the hard way: do NOT give the model a literal
# response example in the override. qwen2.5:7b will copy the example
# verbatim and skip the tool call entirely ("Listo, quedó agendado."
# was the literal example → model responded with exactly that text,
# tool_rounds=0). The override now makes the tool call the PRIMARY
# action and describes the response shape abstractly.
_PROPOSE_CREATE_OVERRIDE = (
    "El usuario te pide REGISTRAR un recordatorio o evento en Apple. "
    "Dos tools disponibles: una para eventos de calendario, otra para "
    "reminders. Ollama te pasa los schemas — invocalas como tool_calls "
    "del protocolo, no como texto ni como prosa.\n\n"
    "Criterio de selección:\n"
    "  - Evento de calendario → visitas, cumpleaños, reuniones, "
    "    turnos médicos, viajes, vacaciones, feriados.\n"
    "  - Reminder → tareas, pagos, llamadas, cosas para acordarse.\n\n"
    "Para el título: extraé el sustantivo + contexto del mensaje "
    "(sin fechas, sin horas). Ej: 'cumpleaños de Astor el viernes' "
    "→ título 'cumpleaños de Astor'.\n\n"
    "Para la fecha/hora: pasala EXACTAMENTE como la dijo el usuario "
    "('el viernes', 'mañana a las 10', 'el miercoles'). La tool la "
    "parsea con anchor de hoy. Si el usuario no dio hora, no pasés "
    "ningún campo de hora — la tool detecta sola que es all-day.\n\n"
    "Después del tool call, el siguiente turn del assistant (sin tool "
    "calls) es UNA oración breve de confirmación. No repitas los "
    "campos (el usuario los ve en el chip inline del chat).\n\n"
    "No cites notas del vault, no inventes paths, no menciones "
    "REGLA 1. El único output válido acá es un tool_call en el "
    "protocolo; imprimir el call como texto es un bug."
)


@app.on_event("startup")
def _warmup() -> None:
    """Hydrate the home cache and kick the bg prewarmer; DO NOT pin the
    chat model or reranker on boot.

    The previous version called `ollama.chat(model=command-r, ...)` on
    startup which pinned ~19 GB of unified memory even when the user was
    only browsing /dashboard or /. On a 36 GB Mac that's enough wired
    memory to starve the kernel and cause host freezes. Warmup of the
    expensive bits (reranker MPS init, chat model load) now happens
    on-demand from the first /api/chat request — 2-3s extra cost on one
    query in exchange for not holding the machine hostage when idle.

    Corpus/BM25/PageRank are cheap (RAM only, no VRAM), so we still
    preload them so the first retrieve is fast.
    """
    import threading

    _load_home_cache()
    _ensure_home_prewarmer()
    _ensure_chat_model_prewarmer()

    def _do_warmup() -> None:
        try:
            from rag import get_db_for
            for _name, path in resolve_vault_paths(None):
                try:
                    col = get_db_for(path)
                    if col.count():
                        _load_corpus(col)
                        get_pagerank(col)
                except Exception:
                    pass
                break
            # End-to-end warmup: load the expensive singletons that the
            # first /api/chat would otherwise pay for (reranker on MPS +
            # bge-m3 SentenceTransformer + one dummy embed pass). Only
            # fires when the operator has opted into pinning them via the
            # same flags that keep them resident.
            if os.environ.get("RAG_RERANKER_NEVER_UNLOAD", "").strip() not in ("", "0", "false", "no"):
                try:
                    from rag import get_reranker as _get_rr
                    _get_rr()
                    print("[warmup] reranker loaded on MPS", flush=True)
                except Exception as _exc:
                    print(f"[warmup] reranker skipped: {_exc}", flush=True)
            if os.environ.get("RAG_LOCAL_EMBED", "").strip() not in ("", "0", "false", "no"):
                try:
                    from rag import _get_local_embedder as _gle
                    _mdl = _gle()
                    if _mdl is not None:
                        _mdl.encode(["warmup"], show_progress_bar=False)
                        print("[warmup] bge-m3 local embedder ready", flush=True)
                except Exception as _exc:
                    print(f"[warmup] local embed skipped: {_exc}", flush=True)
            # Drain any conversation turns that failed to persist on previous
            # runs (transient SQL busy / disk full / etc). Best-effort;
            # survivors stay in the file for the next startup.
            try:
                _retried = _retry_pending_conversation_turns()
                if _retried:
                    print(f"[warmup] retried {_retried} pending conversation turn(s)",
                          flush=True)
            except Exception as _exc:
                print(f"[warmup] conversation-turn retry failed: {_exc}", flush=True)
        except Exception:
            pass

    threading.Thread(target=_do_warmup, daemon=True).start()

    def _idle_sweeper() -> None:
        """Evict the reranker from MPS after `_RERANKER_IDLE_TTL` of no
        activity. Keeps the Mac responsive when the user walks away from
        chat — 2-3 GB of unified memory freed on idle.

        After eviction, schedules a deferred pre-warm 60s later so the
        reranker is hot again before the likely next request (user returns
        to chat after a short break). The pre-warm runs in its own daemon
        thread and MUST NOT block the sweeper loop.

        When `RAG_RERANKER_NEVER_UNLOAD=1` (env var) the sweeper loop
        still runs but skips `maybe_unload_reranker()` entirely — the
        reranker stays pinned in MPS VRAM at all times. Use this on a
        36 GB unified-memory Mac where the ~2-3 GB cost is acceptable and
        eliminating the 9s cold-reload hit after idle eviction is worth it.
        """
        from rag import maybe_unload_reranker, get_reranker
        _never_unload = os.environ.get("RAG_RERANKER_NEVER_UNLOAD", "").strip() not in ("", "0", "false", "no")
        if _never_unload:
            print("[idle-sweep] RAG_RERANKER_NEVER_UNLOAD=1 — reranker pinned, sweeper disabled", flush=True)
        while True:
            try:
                time.sleep(120)
                if _never_unload:
                    continue
                if maybe_unload_reranker():
                    print("[idle-sweep] reranker unloaded from MPS", flush=True)

                    def _deferred_rewarm() -> None:
                        time.sleep(60)
                        try:
                            get_reranker()
                            print("[idle-sweep] reranker pre-warmed (deferred)", flush=True)
                        except Exception:
                            pass

                    threading.Thread(
                        target=_deferred_rewarm, name="reranker-rewarm", daemon=True
                    ).start()
            except Exception:
                pass

    threading.Thread(target=_idle_sweeper, name="idle-sweeper", daemon=True).start()


@app.get("/")
def home_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "home.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


_CHAT_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,80}$")
_TURN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_CHAT_QUESTION_MAX = 16000  # ~4k tokens — bumped from 8000 (2026-04-20) because users sometimes paste long doc excerpts into chat; 16000 still well under CHAT_OPTIONS num_ctx=4096 × 4 chars/token


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    # None → vault activo; "all" → todos los registrados; "name" → ese puntual.
    vault_scope: str | None = None

    @field_validator("question")
    @classmethod
    def _check_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must be non-empty")
        if len(v) > _CHAT_QUESTION_MAX:
            raise ValueError(f"question too long (>{_CHAT_QUESTION_MAX} chars)")
        return v

    @field_validator("session_id")
    @classmethod
    def _check_session(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not _CHAT_SESSION_RE.match(v):
            raise ValueError("invalid session_id format")
        return v

    @field_validator("vault_scope")
    @classmethod
    def _check_vault_scope(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        # Same character class as session_id but shorter — vault names are
        # paths / identifiers registered via `rag vault add`.
        if len(v) > 200 or not re.match(r"^[A-Za-z0-9_./\-]{1,200}$", v):
            raise ValueError("invalid vault_scope format")
        return v


class FeedbackRequest(BaseModel):
    turn_id: str
    rating: int                   # +1 thumbs up / -1 thumbs down
    # Post-hoc parse-time size caps so a multi-MB payload rejects with
    # 422 before we allocate it into memory. The downstream handler
    # trims/caps further as a belt-and-suspenders.
    q: str | None = Field(None, max_length=2000)
    paths: list[str] | None = Field(None, max_length=50)
    session_id: str | None = None
    reason: str | None = Field(None, max_length=500)     # short free-text, optional (≤200 chars after trim)

    @field_validator("turn_id")
    @classmethod
    def _check_turn_id(cls, v: str) -> str:
        if not _TURN_ID_RE.match(v):
            raise ValueError("invalid turn_id format")
        return v

    @field_validator("session_id")
    @classmethod
    def _check_session(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not _CHAT_SESSION_RE.match(v):
            raise ValueError("invalid session_id format")
        return v


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    """Record a +1/-1 rating for a chat turn into feedback.jsonl.

    The rating signal feeds ranker tuning (`rag tune`) and the golden
    cache that biases future retrievals. Client sends the turn_id from
    the `done` SSE event plus the question and the source paths it saw,
    so we don't need to re-scan queries.jsonl on every click.

    Optional `reason` lets the user jot a short note when clicking 👎
    ("genérico", "faltó la nota X", "mezcla temas") — grouped later by
    `rag insights` to surface common failure modes.
    """
    if not req.turn_id or req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="turn_id + rating ±1 requeridos")
    reason = (req.reason or "").strip()[:200] or None
    record_feedback(
        turn_id=req.turn_id,
        rating=req.rating,
        q=(req.q or "").strip(),
        paths=req.paths or [],
        reason=reason,
        session_id=req.session_id,
    )
    return {"ok": True}


# ── /api/behavior ─────────────────────────────────────────────────────────────
# In-memory token bucket for rate limiting: 120 events/min per IP.
# Stored as {ip: [timestamp, ...]} with a 60s sliding window. No new deps.
import collections as _collections
import threading as _threading

_BEHAVIOR_BUCKETS: dict[str, list[float]] = _collections.defaultdict(list)
_BEHAVIOR_RATE_LIMIT = 120
_BEHAVIOR_RATE_WINDOW = 60.0  # seconds

# Chat endpoint bucket — stricter because each hit pins the reranker, chat
# model, and embedder on MPS; 30 chats/min per IP is plenty for human use
# and stops an adversarial loop (e.g. browser extension gone rogue) from
# starving the daemon. Separate from behavior bucket so clicks don't
# consume chat budget.
_CHAT_BUCKETS: dict[str, list[float]] = _collections.defaultdict(list)
_CHAT_RATE_LIMIT = 30
_CHAT_RATE_WINDOW = 60.0

# Single lock protects both buckets — contention is effectively nil
# (list.append under GIL is fast enough that a lock hold measures in µs).
_RATE_LIMIT_LOCK = _threading.Lock()


def _check_rate_limit(bucket: dict[str, list[float]], ip: str,
                      limit: int, window: float) -> None:
    """Sliding-window rate limit per-IP. Raises HTTPException 429 on breach."""
    now = time.time()
    cutoff = now - window
    with _RATE_LIMIT_LOCK:
        events = bucket[ip]
        while events and events[0] < cutoff:
            events.pop(0)
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        events.append(now)

_BEHAVIOR_KNOWN_EVENTS = frozenset({
    "open", "open_external", "positive_implicit",
    "negative_implicit", "kept", "deleted", "save",
})
_BEHAVIOR_SESSION_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,80}$")


class BehaviorRequest(BaseModel):
    source: str
    event: str
    query: str | None = None
    path: str | None = None
    rank: int | None = None
    dwell_ms: int | None = None
    session: str | None = None


@app.post("/api/behavior")
def submit_behavior(req: BehaviorRequest, request: Request) -> dict:
    """Record a user-behavior event from the web dashboard into behavior.jsonl."""
    # Validate source
    if req.source != "web":
        raise HTTPException(status_code=400, detail="source must be 'web'")

    # Validate event
    if req.event not in _BEHAVIOR_KNOWN_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown event '{req.event}'; valid: {sorted(_BEHAVIOR_KNOWN_EVENTS)}",
        )

    # Validate session regex if present
    if req.session is not None and not _BEHAVIOR_SESSION_RE.match(req.session):
        raise HTTPException(status_code=400, detail="session id format invalid")

    # Validate path stays inside vault (no traversal)
    if req.path is not None:
        p = req.path
        if p.startswith("/") or ".." in p.split("/"):
            raise HTTPException(status_code=400, detail="path must be vault-relative")
        try:
            resolved = (VAULT_PATH / p).resolve()
            VAULT_PATH.resolve()  # ensure VAULT_PATH itself resolves
            resolved.relative_to(VAULT_PATH.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path escapes vault root")

    # Rate limit: 120 events/60s per client IP
    client_ip = (request.client.host if request.client else "unknown")
    _check_rate_limit(_BEHAVIOR_BUCKETS, client_ip,
                      _BEHAVIOR_RATE_LIMIT, _BEHAVIOR_RATE_WINDOW)

    try:
        log_behavior_event({
            "source": req.source,
            "event": req.event,
            "query": req.query,
            "path": req.path,
            "rank": req.rank,
            "dwell_ms": req.dwell_ms,
            "session": req.session,
        })
    except Exception as exc:
        # Never 500 on I/O failure — degrade gracefully
        print(f"[behavior] write error: {exc}", flush=True)
        raise HTTPException(status_code=503, detail="event log unavailable")

    return {"ok": True}


def _ollama_alive(timeout: float = 2.0) -> bool:
    """Fast probe: does the ollama daemon answer `/api/tags` within `timeout`s?
    When the daemon hits its stuck-load state the HTTP listener accepts but
    never replies — /api/chat then hangs forever. A short /api/tags probe
    catches that state without waiting for a model load.
    """
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# Dedicated streaming client with a per-chunk read timeout. httpx applies
# `read` to each chunk of a streaming response: if no bytes arrive within
# the budget, ReadTimeout fires and our `except Exception` surfaces an
# error SSE to the frontend. Without this, a stuck-load ollama daemon
# silently wedges the /api/chat stream forever (spinner never clears).
# Budget rationale: qwen2.5:7b prefill on 9k-char ctx measures 5-10s P50,
# so 45s read is >4x the worst observed and well below user patience.
_OLLAMA_STREAM_TIMEOUT = 45.0
_OLLAMA_STREAM_CLIENT = ollama.Client(timeout=_OLLAMA_STREAM_TIMEOUT)

# Shared num_ctx for every call to the chat model from this server — the
# real /api/chat, the preflight probe, and the background prewarmer must
# all pass the same value. ollama re-initialises the KV cache when a
# model call arrives with a different num_ctx than the currently-loaded
# one, and qwen2.5:7b's default (2048) differs from the value the real
# chat path needs (4096 — see _WEB_CHAT_OPTIONS below). A single drift
# between preflight and /api/chat forces a full KV reinit and pushes
# prefill from ~1.5s to ~4.9s (measured 2026-04-20). Defined at module
# scope so there's one source of truth.
_WEB_CHAT_NUM_CTX = 4096


def _ollama_chat_probe(timeout_s: float = 6.0) -> bool:
    """Deep probe: does `/api/chat` actually stream a token within `timeout_s`?

    The shallow `/api/tags` probe is insufficient — we've seen the daemon
    accept /api/tags while /api/chat hangs indefinitely (stuck-load mid-run).
    This probe sends a 1-token chat against the pinned chat model; if the
    daemon is wedged, httpx raises within the budget. Reuses the streaming
    client so the budget applies uniformly.

    CRITICAL: MUST pass num_ctx=_WEB_CHAT_NUM_CTX. Without it, ollama uses
    its default (2048 for qwen2.5:7b) which differs from the real /api/chat
    call (4096), forcing ollama to reinit the KV cache on the next user
    request. Measured impact: prefill 1.5s → 4.9s (3x slower). See
    `num_ctx mismatch` comment in the main chat path for the same trap.

    Also MUST include _WEB_SYSTEM_PROMPT as the system message. The probe
    runs before every /api/chat via _ollama_restart_if_stuck, and if it
    sends a different prompt (even just [user:"."]) ollama overwrites the
    slot's KV cache with that short prompt. The next real request then
    cold-prefills the 1300-token system prompt from scratch. Measured
    2026-04-20: probe w/o system → prefill 4.9s; probe WITH system →
    prefill 3.4s (−1.5s per request). The probe itself pays the cold
    prefill once at startup (~2.5s) and then ~80-100ms on every
    subsequent call since the system cache already exists.
    """
    try:
        _OLLAMA_STREAM_CLIENT.chat(
            model=_resolve_web_chat_model(),
            messages=[
                {"role": "system", "content": _WEB_SYSTEM_PROMPT},
                {"role": "user", "content": "."},
            ],
            options={"num_predict": 1, "num_ctx": _WEB_CHAT_NUM_CTX,
                     "temperature": 0, "seed": 42},
            stream=False,
            think=False,   # thinking-capable models would otherwise emit
                           # <think> blocks as "tokens" with empty content
            keep_alive=-1,
        )
        return True
    except Exception:
        return False


def _ollama_restart_if_stuck() -> bool:
    """Heal the ollama daemon via `brew services restart`. Returns True on
    successful restart. Blocks 3-5s for the bounce.
    """
    try:
        subprocess.run(
            ["/opt/homebrew/bin/brew", "services", "restart", "ollama"],
            check=True, capture_output=True, timeout=30,
        )
        # Wait up to 10s for the daemon to accept traffic again.
        for _ in range(20):
            if _ollama_alive(timeout=1.0):
                return True
            time.sleep(0.5)
    except Exception:
        pass
    return False


@app.post("/api/ollama/restart")
def ollama_restart() -> dict:
    """Panic button #2: brew-services-restart the ollama daemon. Use when
    /api/chat is hanging forever (stuck-load state: daemon accepts HTTP
    but never streams a reply). Blocks ~5-10s.
    """
    ok = _ollama_restart_if_stuck()
    return {"ok": ok, "alive": _ollama_alive()}


@app.post("/api/ollama/unload")
def ollama_unload() -> dict:
    """Panic button: evict every loaded model from ollama + drop the
    reranker from MPS. Frees 15-25 GB of wired unified memory in one
    call when the Mac starts beachballing.

    Drops each loaded model with keep_alive=0 (ollama's immediate-evict
    sentinel) and clears the process-local reranker. Next chat query
    pays 3-8s cold reload but the host stays responsive until then.
    """
    freed = []
    try:
        ps = ollama.ps()
        for m in getattr(ps, "models", []) or []:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if not name:
                continue
            try:
                # A bare generate with keep_alive=0 tells ollama to unload
                # immediately after the (empty) prompt returns.
                ollama.generate(model=name, prompt="", keep_alive=0)
                freed.append(name)
            except Exception as exc:
                freed.append(f"{name} (fail: {exc})")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "freed": freed}
    try:
        from rag import maybe_unload_reranker, _reranker_last_use  # noqa: F401
        import rag as _rag
        _rag._reranker_last_use = 0.0   # force eviction regardless of idle_ttl
        reranker_dropped = maybe_unload_reranker()
    except Exception:
        reranker_dropped = False
    return {"ok": True, "freed_models": freed, "reranker_dropped": reranker_dropped}


class ReminderCreateRequest(BaseModel):
    text: str
    due: str | None = None         # legacy token: "tomorrow" or None
    due_iso: str | None = None     # ISO-8601 datetime; wins over `due`
    list: str | None = None        # Reminders list name; None → default
    priority: int | None = None    # 1 high / 5 medium / 9 low / None
    notes: str | None = None       # body text
    recurrence: dict | None = None  # {freq, interval, byday?}


@app.post("/api/reminders/create")
def create_reminder(req: ReminderCreateRequest) -> dict:
    """Create an Apple Reminder.

    Dual-use:
      - Daily brief's "Para mañana" items POST `{text, due:"tomorrow"}` —
        the legacy path, preserved byte-for-byte.
      - Chat proposal cards POST `{text, due_iso, list, priority, notes,
        recurrence}` after the user clicks ✓ on a `proposal` SSE card.

    If both `due` and `due_iso` are present, `due_iso` wins. `recurrence`
    is best-effort: Reminders.app's AppleScript dictionary inconsistently
    accepts the property across macOS versions — the reminder is created
    regardless (silent-fail on recurrence), caller should advise the user
    to verify in Reminders.app.
    """
    from datetime import datetime as _dt

    from rag import _create_reminder  # noqa: PLC0415

    due_dt: _dt | None = None
    if req.due_iso:
        try:
            due_dt = _dt.fromisoformat(req.due_iso)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"due_iso inválido: {exc}",
            ) from exc

    ok, res = _create_reminder(
        req.text,
        due_token=req.due if not due_dt else None,
        list_name=req.list,
        due_dt=due_dt,
        priority=req.priority,
        notes=req.notes,
        recurrence=req.recurrence,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=res)
    # Bust the home cache so the new reminder shows up in the reminders
    # panel on next load without waiting for SWR.
    try:
        _HOME_STATE["ts"] = 0.0
    except Exception:
        pass
    return {"ok": True, "id": res}


class CalendarCreateRequest(BaseModel):
    title: str
    start_iso: str                 # required
    end_iso: str | None = None     # None → start + 1h
    calendar: str | None = None    # None → first writable
    location: str | None = None
    notes: str | None = None
    all_day: bool = False
    recurrence: dict | None = None  # {freq, interval, byday?}


@app.post("/api/calendar/create")
def create_calendar_event(req: CalendarCreateRequest) -> dict:
    """Create a Calendar.app event. Called from chat proposal cards after
    the user confirms. Returns the event UID so the UI can surface a
    deep-link or deletion path.

    Writes through Calendar.app → iCloud CalDAV calendars are reachable
    (unlike the JXA EventKit read path which can't see them without
    entitlement). If `calendar` is omitted we write to the first writable
    calendar — usually iCloud's default.
    """
    from datetime import datetime as _dt

    from rag import _create_calendar_event  # noqa: PLC0415

    try:
        start_dt = _dt.fromisoformat(req.start_iso)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"start_iso inválido: {exc}",
        ) from exc
    end_dt: _dt | None = None
    if req.end_iso:
        try:
            end_dt = _dt.fromisoformat(req.end_iso)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"end_iso inválido: {exc}",
            ) from exc

    ok, res = _create_calendar_event(
        req.title,
        start_dt,
        end_dt,
        calendar=req.calendar,
        location=req.location,
        notes=req.notes,
        all_day=req.all_day,
        recurrence=req.recurrence,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=res)
    return {"ok": True, "uid": res}


class ReminderCompleteRequest(BaseModel):
    reminder_id: str


@app.post("/api/reminders/complete")
def complete_reminder(req: ReminderCompleteRequest) -> dict:
    """Mark an Apple Reminder as completed by its stable id.

    Called from the home page when the user ticks a reminder. Invalidates
    the home cache so the next /api/home doesn't return the now-completed
    item from SWR.
    """
    from rag import _complete_reminder  # noqa: PLC0415
    ok, msg = _complete_reminder(req.reminder_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    # Bust the home cache so the reminder disappears on next load without
    # waiting for the 60s SWR cycle.
    try:
        _HOME_STATE["ts"] = 0.0
    except Exception:
        pass
    return {"ok": True, "message": msg}


class ReminderDeleteRequest(BaseModel):
    reminder_id: str


@app.post("/api/reminders/delete")
def delete_reminder(req: ReminderDeleteRequest) -> dict:
    """Permanently delete an Apple Reminder. Called from the chat's
    auto-create toast Deshacer button — the user just saw a reminder
    land in Reminders.app and wants to undo it within the 10s window.

    POST (not DELETE verb) because the reminder_id is an
    `x-apple-reminderkit://` URI which is painful to URL-encode.
    """
    from rag import _delete_reminder  # noqa: PLC0415
    ok, msg = _delete_reminder(req.reminder_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    try:
        _HOME_STATE["ts"] = 0.0
    except Exception:
        pass
    return {"ok": True, "message": msg}


class CalendarDeleteRequest(BaseModel):
    event_uid: str


@app.post("/api/calendar/delete")
def delete_calendar_event(req: CalendarDeleteRequest) -> dict:
    """Permanently delete a Calendar.app event by its UID. Companion
    to the chat's auto-create toast Deshacer button for events.
    """
    from rag import _delete_calendar_event  # noqa: PLC0415
    ok, msg = _delete_calendar_event(req.event_uid)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg}


@app.get("/api/model")
def get_chat_model() -> dict:
    """Return the chat model that /api/chat would use right now."""
    return {"model": _resolve_web_chat_model()}


@app.get("/api/session/{sid}")
def get_session_info(sid: str) -> dict:
    """Return session summary for the chat UI `/session` slash command."""
    from rag import load_session  # noqa: PLC0415
    sess = load_session(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session no encontrada")
    turns = sess.get("turns", [])
    return {
        "id": sess.get("id", sid),
        "mode": sess.get("mode", ""),
        "created_at": sess.get("created_at", ""),
        "updated_at": sess.get("updated_at", ""),
        "turns": len(turns),
        "first_q": (turns[0].get("q", "") if turns else "")[:120],
        "last_q": (turns[-1].get("q", "") if turns else "")[:120],
    }


class SaveRequest(BaseModel):
    session_id: str
    title: str | None = None


@app.post("/api/save")
def save_conversation(req: SaveRequest) -> dict:
    """Persist a chat session into the vault as a single Markdown note.

    The CLI's `/save` only saves the last turn; here we flatten the whole
    session so the user can export a back-and-forth thread. Lands in
    00-Inbox with source_metas aggregated across all turns.
    """
    from rag import get_db_for, load_session, save_note  # noqa: PLC0415
    sess = load_session(req.session_id)
    if not sess or not sess.get("turns"):
        raise HTTPException(status_code=404, detail="session vacía o inexistente")
    turns = sess["turns"]
    body_parts: list[str] = []
    for t in turns:
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        if not q and not a:
            continue
        body_parts.append(f"**Pregunta:** {q}\n\n{a}")
    body = "\n\n---\n\n".join(body_parts)
    question = turns[0].get("q", "") or "chat"
    vaults = resolve_vault_paths(None)
    if not vaults:
        raise HTTPException(status_code=400, detail="no hay vault activo")
    col = get_db_for(vaults[0][1])
    seen: set[str] = set()
    source_metas: list[dict] = []
    for t in turns:
        for p in (t.get("paths") or []):
            if p and p not in seen:
                seen.add(p)
                source_metas.append({
                    "file": p,
                    "note": Path(p).stem,
                    "tags": "",
                })
    path = save_note(col, req.title, body, question, source_metas)
    try:
        rel = str(path.relative_to(vaults[0][1]))
    except ValueError:
        rel = str(path)
    return {"ok": True, "path": rel}


@app.post("/api/reindex")
def trigger_reindex() -> dict:
    """Fire-and-forget incremental reindex. `--reset` not exposed via web."""
    try:
        subprocess.Popen(
            ["rag", "index"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="rag CLI no encontrado en PATH")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": "index lanzado en background"}


class FollowupsRequest(BaseModel):
    session_id: str


@app.post("/api/followups")
def followups(req: FollowupsRequest) -> dict:
    """Generate 3 follow-up question chips from the last turn.

    Uses the helper model (qwen2.5:3b) with deterministic temp to keep
    suggestions tight and cheap (~400-800ms on M-series). Returns empty
    list on any error so the UI fails silently.
    """
    from rag import load_session, get_db  # noqa: PLC0415
    sess = load_session(req.session_id)
    if not sess or not sess.get("turns"):
        return {"followups": []}
    last = sess["turns"][-1]
    q = (last.get("q") or "").strip()
    a = (last.get("a") or "").strip()[:800]
    paths = (last.get("paths") or [])[:3]
    snippets: list[str] = []
    if paths:
        try:
            col = get_db()
            for p in paths:
                res = col.get(where={"file": p}, include=["documents", "metadatas"])
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                if not docs:
                    continue
                title = (metas[0].get("note") if metas else None) or Path(p).stem
                body = (docs[0] or "").strip().replace("\n", " ")
                snippets.append(f"- {title}: {body[:280]}")
        except Exception:
            snippets = []
    ctx_bits = [f"Pregunta previa: {q}", f"Respuesta: {a}"]
    if snippets:
        ctx_bits.append("Fragmentos de las notas que aparecieron:\n" + "\n".join(snippets))
    ctx = "\n\n".join(ctx_bits)
    prompt = (
        "Sugerí 3 preguntas de seguimiento concretas que el usuario podría "
        "hacer para profundizar usando su vault de Obsidian. Las preguntas "
        "DEBEN anclarse en hechos, nombres, herramientas o conceptos que "
        "aparezcan literalmente en los fragmentos de arriba — no inventes "
        "ángulos no presentes. Cada pregunta ≤70 caracteres, en español "
        "rioplatense (tuteo). Devolvé SOLO un JSON con la forma "
        '{"followups": ["...", "...", "..."]}. Sin texto extra.\n\n'
        f"{ctx}\n\nJSON:"
    )
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "seed": 42, "num_predict": 220, "num_ctx": 2048},
            format="json",
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = (resp.message.content or "").strip() or "{}"
        data = json.loads(raw)
        if isinstance(data, dict):
            arr = data.get("followups") or data.get("questions") or \
                  next((v for v in data.values() if isinstance(v, list)), [])
        elif isinstance(data, list):
            arr = data
        else:
            arr = []
        out: list[str] = []
        for s in arr:
            if not isinstance(s, str):
                continue
            s = s.strip().strip('"').strip("'")
            if s and len(s) <= 90:
                out.append(s)
            if len(out) >= 3:
                break
        return {"followups": out}
    except Exception:
        return {"followups": []}


# Related-context (Deezer + YouTube) -----------------------------------------
# Deezer's public search API needs no auth (CORS-restricted but server-side
# is fine). Returned every relevant track has a `link` to deezer.com.


def _deezer_search(query: str, limit: int = 2) -> list[dict]:
    import urllib.request
    import urllib.parse
    qs = urllib.parse.urlencode({"q": query, "limit": limit})
    try:
        with urllib.request.urlopen(f"https://api.deezer.com/search?{qs}", timeout=4) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    items: list[dict] = []
    for tr in (data.get("data") or [])[:limit]:
        artist = (tr.get("artist") or {}).get("name") or ""
        album = (tr.get("album") or {}).get("title") or ""
        items.append({
            "source": "deezer",
            "kind": "track",
            "title": tr.get("title") or "",
            "subtitle": f"{artist} · {album}".strip(" ·"),
            "url": tr.get("link") or "",
        })
    return [i for i in items if i.get("url") and i.get("title")]


def _youtube_search(query: str, limit: int = 2) -> list[dict]:
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        return []
    import urllib.request
    import urllib.parse
    qs = urllib.parse.urlencode({
        "part": "snippet", "q": query, "maxResults": limit,
        "type": "video", "key": key, "relevanceLanguage": "es", "safeSearch": "none",
    })
    try:
        with urllib.request.urlopen(f"https://www.googleapis.com/youtube/v3/search?{qs}", timeout=4) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    items: list[dict] = []
    for it in (data.get("items") or [])[:limit]:
        vid = (it.get("id") or {}).get("videoId")
        sn = it.get("snippet") or {}
        if not vid:
            continue
        items.append({
            "source": "youtube",
            "kind": "video",
            "title": sn.get("title") or "",
            "subtitle": sn.get("channelTitle") or "",
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return items


class RelatedRequest(BaseModel):
    query: str
    sources: list[str] | None = None  # subset of ["deezer","youtube"]; None = all


@app.post("/api/related")
def related(req: RelatedRequest) -> dict:
    """External enrichment: Deezer + YouTube. Returns merged items list.
    Empty if no API keys present or query is empty. The frontend decides
    when to call this (low-confidence answers, empty retrieval, etc.) so
    the endpoint stays dumb and side-effect-free.
    """
    q = (req.query or "").strip()
    if not q or len(q) < 3:
        return {"items": []}
    wanted = set(req.sources or ["youtube"])
    items: list[dict] = []
    if "deezer" in wanted:
        items.extend(_deezer_search(q, limit=2))
    if "youtube" in wanted:
        items.extend(_youtube_search(q, limit=2))
    return {"items": items}


class TTSRequest(BaseModel):
    text: str
    voice: str = "Monica"


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Render text with macOS `say` (default voice Mónica) to WAV bytes.

    Local-only: no cloud. WAV chosen over AIFF because browser support is
    universal. Text capped and markdown-stripped so the voice doesn't
    read out backticks and brackets.
    """
    import tempfile  # noqa: PLC0415
    from fastapi.responses import Response  # noqa: PLC0415
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="texto vacío")
    text = text[:1500]
    clean = re.sub(r"```[\s\S]*?```", " ", text)
    clean = re.sub(r"`[^`]*`", " ", clean)
    clean = re.sub(r"[#*_\[\]<>]", "", clean)
    clean = re.sub(r"\(https?://[^)]+\)", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="texto vacío post-limpieza")
    voice = req.voice if re.match(r"^[A-Za-zÀ-ÿ]{1,32}$", req.voice or "") else "Monica"
    fd = None
    out_path = None
    try:
        fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = fh.name
        fh.close()
        subprocess.run(
            ["say", "-v", voice, "--file-format=WAVE",
             "--data-format=LEI16@22050", "-o", out_path, clean],
            check=True, capture_output=True, timeout=45,
        )
        data = Path(out_path).read_bytes()
        return Response(content=data, media_type="audio/wav")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise HTTPException(status_code=500, detail=f"say falló: {detail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="say timeout")
    finally:
        if out_path:
            try:
                Path(out_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/api/vaults")
def list_vaults() -> dict:
    """Expone el registry + vault activo para el picker de la UI."""
    cfg = _load_vaults_config()
    active = resolve_vault_paths(None)
    active_name = active[0][0] if active else None
    registered = sorted(cfg.get("vaults", {}).keys())
    # Si el activo no está en el registry (p.ej. OBSIDIAN_RAG_VAULT apuntando
    # a uno no registrado), lo incluimos igual para que el picker lo muestre.
    if active_name and active_name not in registered:
        registered = [active_name] + registered
    return {
        "active": active_name,
        "registered": registered,
        "current": cfg.get("current"),
    }


# ── Chat model runtime switch ────────────────────────────────────────────────
# Qwen / Llama families ship a new tag every few weeks; hard-coding the
# default in a launchd plist and restarting the daemon for each A/B is
# enough friction to stop users from experimenting. These two endpoints
# let the UI list available local models and pick one at runtime. No
# server restart, no plist edit — the override is persisted to
# ~/.local/share/obsidian-rag/chat-model.json and survives daemon restarts.

# Families we want to surface as chat-capable. Everything else (embeddings,
# rerankers, vision-only, helpers) is filtered out for the picker. This is
# a denylist by prefix/pattern — easier to maintain as new model families
# appear than whitelisting exact tags.
_CHAT_MODEL_FAMILY_DENYLIST = (
    "bge-",          # embedding models
    "nomic-embed",
    "all-minilm",
    "snowflake-arctic-embed",
    "mxbai-embed",
    "jina-embeddings",
)


class ChatModelRequest(BaseModel):
    model: str | None = None  # None / empty → clear override, revert to defaults


@app.get("/api/chat/model")
def get_chat_model() -> dict:
    """Return the currently-active chat model + the local catalog.

    Shape:
      {
        "current": "qwen2.5:7b",      # what _resolve_web_chat_model() returns
        "override": "qwen2.5:7b",     # only set if the runtime file has one
        "env_override": null,         # OBSIDIAN_RAG_WEB_CHAT_MODEL if set
        "default": "qwen2.5:7b",      # resolve_chat_model() baseline
        "available": ["qwen2.5:7b", "qwen3.6", ...]   # local, chat-capable
      }

    The UI uses this to render the selector with the active option marked.
    """
    override = _read_chat_model_override()
    try:
        available = sorted(
            m.model for m in ollama.list().models
            if not any(m.model.startswith(p) for p in _CHAT_MODEL_FAMILY_DENYLIST)
        )
    except Exception:
        available = []
    try:
        default = resolve_chat_model()
    except Exception:
        default = None
    return {
        "current": _resolve_web_chat_model(),
        "override": override,
        "env_override": WEB_CHAT_MODEL,
        "default": default,
        "available": available,
    }


@app.post("/api/chat/model")
def set_chat_model(req: ChatModelRequest) -> dict:
    """Switch the chat model at runtime. Persisted to disk.

    Empty / null `model` clears the override and reverts to the default
    resolution chain (env var → resolve_chat_model). Validates that the
    requested model is actually installed locally; rejects unknown tags
    with 400 so the UI can surface the error instead of the next /api/chat
    crashing on a missing model.

    Side effect: invalidates the response LRU cache so follow-up queries
    don't replay a cached response produced by a different model.
    """
    requested = (req.model or "").strip() or None
    if requested is not None:
        try:
            available = {m.model for m in ollama.list().models}
        except Exception:
            available = set()
        if requested not in available:
            raise HTTPException(
                status_code=400,
                detail=f"model '{requested}' not installed locally "
                       f"(try: ollama pull {requested.split(':')[0]})",
            )
    try:
        _write_chat_model_override(requested)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not persist override: {exc}")
    # Invalidate response cache so the next identical query doesn't replay
    # the previous model's answer.
    try:
        _CHAT_CACHE.clear()
    except Exception:
        pass
    return {
        "ok": True,
        "current": _resolve_web_chat_model(),
        "override": _read_chat_model_override(),
    }



@app.get("/api/history")
def query_history(limit: int = 200) -> dict:
    """Recent chat questions for terminal-style up-arrow nav in the web UI.

    Returns oldest→newest after deduping consecutive identical questions.
    Filters to chat-bound commands so /save, /reindex, internal eval runs,
    etc. don't pollute the history list.

    Source of truth post-cutover 2026-04-19 is the SQL `rag_queries` table
    (JSONL writes are gated off by RAG_STATE_SQL=1). Falls back to
    `queries.jsonl` if SQL path is off or fails, so pre-cutover installs
    keep working.
    """
    limit = max(1, min(int(limit or 200), 1000))
    # `web` = web chat endpoint (default cmd in /api/chat log_query_event);
    # `query`/`chat`/`ask` = CLI paths; older rows may have empty cmd.
    keep_cmds = {"query", "chat", "ask", "web"}
    out: list[str] = []

    if RAG_STATE_SQL:
        try:
            # Pull ≥limit rows and dedup, newest-first, then reverse to
            # oldest→newest for the UI. Over-fetch by 4x so consecutive
            # duplicates don't starve the returned window.
            with _ragvec_state_conn() as conn:
                rows = conn.execute(
                    "SELECT q, cmd FROM rag_queries "
                    "ORDER BY id DESC LIMIT ?",
                    (limit * 4,),
                ).fetchall()
            seen_prev: str | None = None
            picked: list[str] = []
            for q, cmd in rows:  # newest → oldest
                cmd = (cmd or "").strip()
                if cmd and cmd not in keep_cmds:
                    continue
                q = (q or "").strip()
                if not q or q == seen_prev:
                    continue
                picked.append(q)
                seen_prev = q
                if len(picked) >= limit:
                    break
            out = list(reversed(picked))
            if out:
                return {"history": out}
            # SQL returned zero rows — fall through to JSONL for older data.
        except Exception as exc:
            _log_sql_state_error("history_sql_read_failed", err=repr(exc))

    if not LOG_PATH.is_file():
        return {"history": []}
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                cmd = rec.get("cmd") or ""
                if cmd and cmd not in keep_cmds:
                    continue
                q = (rec.get("q") or "").strip()
                if not q:
                    continue
                if out and out[-1] == q:
                    continue
                out.append(q)
    except OSError:
        return {"history": []}
    return {"history": out[-limit:]}


def _resolve_scope(scope: str | None) -> list[tuple[str, "Path"]]:
    if scope is None or scope == "":
        return resolve_vault_paths(None)
    if scope == "all":
        return resolve_vault_paths(["all"])
    return resolve_vault_paths([scope])


class _InlineCitationStripper:
    """Streaming filter that elides `[Title](path.md)` from LLM output.

    Command-r ignores "don't cite inline" instructions about ~15-20% of
    the time even when the prompt is unambiguous. Since the UI renders a
    dedicated sources block below every answer, inline citations show
    the same note twice. This filter is a safety net: it buffers any
    text starting at an open bracket until we know whether the bracket
    opens a `[…](…md)` link or something else. If it does, we drop the
    whole pattern; if it doesn't, we flush the buffered text unchanged.

    Kept mid-module (not inside the chat handler) so tests can import it
    in isolation.
    """

    _MAX_TITLE = 200   # guard against runaway buffers on malformed output
    _MAX_TARGET = 400

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        i = 0
        n = len(self._buf)
        while i < n:
            c = self._buf[i]
            if c != "[":
                out.append(c)
                i += 1
                continue
            # Try to resolve a `[title](target)` pattern starting here.
            close_br = self._buf.find("]", i + 1)
            if close_br == -1:
                if n - i > self._MAX_TITLE:
                    # Not a link — too long to be a title. Flush.
                    out.append(self._buf[i])
                    i += 1
                    continue
                break  # incomplete; wait for more.
            if close_br + 1 >= n:
                break  # don't know yet what follows `]`.
            if self._buf[close_br + 1] != "(":
                # Plain `[…]` text, not a link. Emit and continue.
                out.append(self._buf[i:close_br + 1])
                i = close_br + 1
                continue
            close_pa = self._buf.find(")", close_br + 2)
            if close_pa == -1:
                if n - (close_br + 2) > self._MAX_TARGET:
                    out.append(self._buf[i:close_br + 2])
                    i = close_br + 2
                    continue
                break  # incomplete link tail; wait for more.
            target = self._buf[close_br + 2:close_pa]
            if target.endswith(".md"):
                # Drop the entire `[title](path.md)`.
                i = close_pa + 1
                # Optional: also elide a trailing ", " or " (" that was
                # a separator between consecutive inline citations.
                while i < n and self._buf[i] in (" ", ","):
                    i += 1
            else:
                # External URL or non-note target — keep as-is.
                out.append(self._buf[i:close_pa + 1])
                i = close_pa + 1
        self._buf = self._buf[i:]
        return "".join(out)

    def flush(self) -> str:
        tail = self._buf
        self._buf = ""
        return tail


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _maybe_emit_proposal(name: str, out: str) -> str | None:
    """If `name` is a proposal tool and `out` parses as the expected JSON
    payload, return a pre-encoded SSE frame; else None.

    Routes based on payload kind:
      - `needs_clarification: true` → `proposal` event (UI shows card,
        user confirms/edits).
      - `created: true` → `created` event (UI shows toast with Deshacer).
      - `created: false` → `proposal` event too, but with `error` field
        set so the UI can surface the failure in the card.

    Keeps the gen() body clean — callers just
    `if frame := _maybe_emit_proposal(...): yield frame`. A malformed
    payload is treated as "not a proposal" (the LLM still sees the tool
    output normally via the `role:"tool"` message; the UI just won't
    render anything). No logging here — caller decides what to do.
    """
    if name not in PROPOSAL_TOOL_NAMES:
        return None
    try:
        payload = json.loads(out)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    # Auto-created happy path → toast + undo.
    if payload.get("created") is True:
        return _sse("created", payload)
    # Ambiguous / needs-clarification path → confirmation card.
    if payload.get("needs_clarification") or payload.get("proposal_id"):
        return _sse("proposal", payload)
    # created=false with error set → show proposal card with error state
    # so the user can retry. Emit as proposal event (UI renders card).
    if payload.get("created") is False and payload.get("error"):
        # Inject a synthetic proposal_id so the UI handler treats it as
        # a normal proposal. Keep the error field for visibility.
        payload.setdefault("proposal_id", f"prop-err-{payload.get('kind', 'x')}")
        payload["needs_clarification"] = True
        return _sse("proposal", payload)
    return None


def _source_payload(meta: dict, score: float) -> dict:
    return {
        "file": meta.get("file", ""),
        "note": meta.get("note", ""),
        "folder": meta.get("folder", ""),
        "score": round(float(score), 3),
    }


def _confidence_badge(score: float) -> tuple[str, str]:
    if score >= 3.0:
        return ("🟢", f"alta · {score:.1f}")
    if score >= 0.0:
        return ("🟡", f"media · {score:.1f}")
    return ("🔴", f"baja · {score:.1f}")


def _score_bar(score: float, width: int = 5) -> str:
    clipped = max(-5.0, min(10.0, score))
    normalized = (clipped + 5.0) / 15.0
    filled = int(round(normalized * width))
    return "■" * filled + "□" * (width - filled)


# ── Intent detection: tasks / agenda / pendientes ────────────────────────────
# Short-circuits the normal RAG retrieve when the question is about what the
# user has to do. RAG over vault notes produces hallucinations ("pedido de
# speeds") because the vault doesn't store live calendar/reminders; the
# services layer does.
#
# Tasks-mode is scope-aware: "mañana" queries fetch tomorrow's reminders +
# calendar, "esta semana" fetches 7 days, default fetches 2 weeks. Without
# this, a "qué tengo para mañana" query got today's reminders and the LLM
# correctly concluded "nada para mañana" — the fix is to bring the right
# window, not to patch the LLM.

_TASKS_INTENT_TOKENS = re.compile(
    r"\b(pendientes?|agenda|recordatorios?|to-?dos?|tasks?|reminders?|"
    r"tareas?|reuniones?|meetings?|prioridad(?:es)?|prioriz|"
    r"compromisos?|obligaciones?|deberes?)",
    re.IGNORECASE,
)
_TASKS_INTENT_PHRASES = re.compile(
    # "qué tengo/debo/hay/etc" REQUIERE qualifier de tarea/tiempo después
    # — la versión laxa anterior matcheaba "qué hay sobre X" / "qué tengo
    # en notas sobre Y" y mis-routeaba topic queries al fetcher de agenda.
    r"(qu[eé]\s+(tengo|debo|me\s+falta|me\s+queda|hay)\s+"
    r"(para|que\s+(hacer|recordar|completar|priorizar)|"
    r"hacer|recordar|completar|"
    r"hoy|ma[nñ]ana|esta\s+semana|pendiente|priorizar|gestionar|organizar)|"
    r"what\s+do\s+i\s+(have|need)\s+(to\s+)?(do|complete|finish)|"
    r"(tengo|hay)\s+(algo|cosas?|eventos?|citas?)\s+(pendiente|hoy|ma[nñ]ana|esta\s+semana)|"
    r"organiz(ar|ame)\s+(el\s+d[ií]a|la\s+semana))",
    re.IGNORECASE,
)


def _is_tasks_query(q: str) -> bool:
    if not q:
        return False
    return bool(_TASKS_INTENT_TOKENS.search(q) or _TASKS_INTENT_PHRASES.search(q))


# Anaphora cues that mean "this question depends on the prior turn" — pronouns,
# demonstratives, ellipsis markers. If a follow-up has any of these, route it
# through reformulate_query so retrieval gets the resolved entity. Otherwise
# the original wording is already a self-contained search target.
#
# Deliberately excluded: `el`/`la`/`lo` — they're common articles and would
# fire on virtually every Spanish sentence, causing the reformat call on fully
# self-contained questions (empirically regressed quality by ~0.15 rerank).
_FOLLOWUP_CUES = re.compile(
    r"\b(eso|esa|ese|esto|esta|este|aquello|ella|[eé]l\b(?!\s+[A-ZÁÉÍÓÚ])|"
    r"ahí|ahi|allí|alli|allá|alla|"
    r"it\b|this\b|that\b|he\b|she\b|"
    r"y\s+(de|sobre|con|para|en)|"
    r"profundizá|profundiza|ampliá|amplia|seguí|segui|continuá|continua|"
    r"más\s+(sobre|de|al\s+respecto)|mas\s+(sobre|de|al\s+respecto))\b",
    re.IGNORECASE,
)

# Leading-pronoun pattern: question starts with a pronoun/demonstrative that
# requires prior context to resolve. Only checked at position 0 (after strip).
_LEADING_PRONOUN_RE = re.compile(
    r"^(eso|esa|ese|esto|esta|este|ella|[eé]l|it|this|that|he|she)\b",
    re.IGNORECASE,
)

# Short interrogative without an explicit subject — "qué hay de X?", "cuál
# es?", "cómo funciona?" where X ≤ 1 token. Only fires for ≤5-word questions
# so long complete questions pass through.
_SHORT_INTERROGATIVE_RE = re.compile(
    r"^(qué|que|cuál|cual|cómo|como|por\s+qué|por\s+que|"
    r"what|how|which|why)\b",
    re.IGNORECASE,
)


def _looks_like_followup(q: str) -> bool:
    """Heuristic: should we pay the reformulate_query LLM call (~1s)?

    Fires when the question is contextually incomplete and needs prior-turn
    resolution. Three cases:
      (a) ≤2 words — genuine ellipsis: "y eso?", "más?", "cuál?"
      (b) Starts with a pronoun/demonstrative: "eso que dijiste", "ella cómo?"
      (c) ≤5-word question starting with an interrogative word (incomplete
          subject implied by context): "qué hay de eso?"

    Long standalone questions (≥4 words without leading cues) skip the call —
    the original query is already a good search target. Avoids paying 0.6-2s
    reformulate on "resumime los sprints", "qué dice la nota sobre Grecia", etc.
    """
    if not q:
        return False
    words = q.split()
    # (a) very short — almost certainly anaphoric
    if len(words) <= 2:
        return True
    # (b) starts with a pronoun/demonstrative that needs antecedent resolution
    stripped = q.strip().lstrip("¿").strip()
    if _LEADING_PRONOUN_RE.match(stripped):
        return True
    # (c) short interrogative (≤5 words) — subject likely implicit from context
    if len(words) <= 5 and _SHORT_INTERROGATIVE_RE.match(stripped):
        return True
    # (d) explicit anaphora cue anywhere in the question
    return bool(_FOLLOWUP_CUES.search(q))


# Pronouns / demonstratives that, if still present after reformulation,
# signal the helper model failed to resolve the antecedent. Triggers the
# concat-with-last-user-turn fallback. `el`/`la`/`lo` excluded because
# they're also common articles and would fire on every sentence.
_UNRESOLVED_PRONOUN_RE = re.compile(
    r"\b(ella|él|eso|esa|ese|esto|esta|este|aquello|"
    r"ahí|ahi|allí|alli|allá|alla)\b",
    re.IGNORECASE,
)


_SCOPE_N_DAYS = re.compile(
    r"\bpr[oó]xim[oa]s?\s+(\d+)\s+d[ií]as?\b|\bnext\s+(\d+)\s+days?\b",
    re.IGNORECASE,
)
_SCOPE_WEEK = re.compile(
    r"\besta\s+semana\b|\bpr[oó]xima\s+semana\b|\bsemana\s+que\s+viene\b|"
    r"\bthis\s+week\b|\bnext\s+week\b",
    re.IGNORECASE,
)
_SCOPE_TOMORROW = re.compile(r"\bma(?:ñ|n)ana\b|\btomorrow\b", re.IGNORECASE)
_SCOPE_TODAY = re.compile(
    r"\bhoy\b|\bahora\b|\btoday\b|\beste\s+momento\b", re.IGNORECASE,
)


def _detect_time_scope(q: str) -> dict:
    """Return the time scope carried by the query. Priority: explicit N days
    beats week beats tomorrow beats today; unmarked queries get a 14-day
    reminders window + 7-day calendar window (user asking "qué tengo
    pendiente" wants a broad sweep).
    """
    m = _SCOPE_N_DAYS.search(q)
    if m:
        n = int(m.group(1) or m.group(2) or 0)
        n = max(1, min(30, n))
        return {
            "key": "n_days", "label": f"próximos {n} días",
            "reminders_horizon": n + 1, "calendar_ahead": n,
        }
    if _SCOPE_WEEK.search(q):
        return {
            "key": "week", "label": "esta semana",
            "reminders_horizon": 8, "calendar_ahead": 7,
        }
    if _SCOPE_TOMORROW.search(q):
        return {
            "key": "tomorrow", "label": "mañana",
            "reminders_horizon": 2, "calendar_ahead": 1,
        }
    if _SCOPE_TODAY.search(q):
        return {
            "key": "today", "label": "hoy",
            "reminders_horizon": 1, "calendar_ahead": 0,
        }
    return {
        "key": "general", "label": "próximas 2 semanas",
        "reminders_horizon": 14, "calendar_ahead": 7,
    }


def _fetch_calendar_ahead(days_ahead: int, max_events: int = 40) -> list[dict]:
    """Events from today through today+days_ahead via icalBuddy with relative
    date labels ("today" / "tomorrow" / "day after tomorrow" / YYYY-MM-DD).
    Returns [{date_label, title, time_range}]. Empty list on any failure —
    the brief layer tolerates silent missing calendar.
    """
    if not _apple_enabled() or days_ahead < 0:
        return []
    icb = _icalbuddy_path()
    if not icb:
        return []
    query = "eventsToday" if days_ahead == 0 else f"eventsToday+{days_ahead}"
    try:
        res = subprocess.run(
            [icb, "-nc", "-iep", "title,datetime", "-b", "", query],
            capture_output=True, text=True, timeout=10.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if res.returncode != 0 or not (res.stdout or "").strip():
        return []

    events: list[dict] = []
    current: dict | None = None
    for raw in (res.stdout or "").splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            if current and current.get("title"):
                events.append(current)
            current = {"title": line.strip(), "date_label": "", "time_range": ""}
            continue
        stripped = line.strip()
        if current is None:
            continue
        m = re.search(
            r"(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)\s*-\s*(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)",
            stripped,
        )
        if m:
            current["time_range"] = f"{m.group(1)}–{m.group(2)}"
            # date label is everything before the "at HH:MM" span
            label = stripped.split(" at ", 1)[0].strip()
            if label and label != stripped:
                current["date_label"] = label
        else:
            # date-only line (all-day event)
            current["date_label"] = stripped
    if current and current.get("title"):
        events.append(current)
    return events[:max_events]


def _collect_scoped_evidence(col, now: datetime, scope: dict) -> dict:
    """`_pendientes_collect` under the hood for services that are scope-free
    (gmail, mail, whatsapp, loops, contradictions), then override `reminders`
    and `calendar` with scope-aware fetches. WhatsApp always stays on — user
    wants the recent chat context attached even to non-WA questions.
    """
    ev = _pendientes_collect(col, now, days=14)
    from contextlib import suppress
    with suppress(Exception):
        ev["reminders"] = _fetch_reminders_due(
            now,
            horizon_days=int(scope["reminders_horizon"]),
            max_items=40,
        )
    with suppress(Exception):
        ev["calendar_range"] = _fetch_calendar_ahead(int(scope["calendar_ahead"]))
    # Ensure whatsapp is always populated even if `_pendientes_collect`
    # skipped it silently (bridge transiently down, etc.).
    if not ev.get("whatsapp"):
        with suppress(Exception):
            ev["whatsapp"] = _fetch_whatsapp_unread(hours=24, max_chats=8)
    return ev


def _format_pendientes_context(
    ev: dict, urgent: list[str], now: datetime, scope: dict,
) -> str:
    """Markdown evidence blob for the LLM. Scope label is front-and-centre so
    the model filters to the right window. WhatsApp is always included —
    even a non-WA question benefits from seeing recent chat context.
    """
    lines: list[str] = [
        f"# Evidencia de servicios · {now.strftime('%Y-%m-%d %H:%M')}",
        f"**Ventana temporal pedida:** {scope['label']}",
    ]

    if urgent:
        lines.append(f"\n## 🚨 Urgente / Overdue ({len(urgent)})")
        lines.extend(f"- {u}" for u in urgent)

    cal = ev.get("calendar_range") or ev.get("calendar") or []
    if cal:
        lines.append(f"\n## 📅 Calendar ({len(cal)} eventos)")
        for e in cal[:20]:
            when_bits = [b for b in (e.get("date_label"), e.get("time_range")) if b]
            if not when_bits and e.get("start"):
                when_bits = [e["start"]]
            when = " · ".join(when_bits) or "sin hora"
            lines.append(f"- {when} — {e.get('title','')}")

    rem = ev.get("reminders") or []
    if rem:
        lines.append(f"\n## 📌 Apple Reminders ({len(rem)})")
        for bucket in ("overdue", "today", "upcoming", "undated"):
            bucket_items = [x for x in rem if (x.get("bucket") or "") == bucket]
            for r in bucket_items[:10]:
                due = f" (due {r['due']})" if r.get("due") else ""
                lst = f" [{r['list']}]" if r.get("list") else ""
                lines.append(f"- **{bucket}**: {r.get('name','')}{due}{lst}")

    gm = ev.get("gmail") or {}
    awaiting = gm.get("awaiting_reply") or []
    if awaiting:
        lines.append(f"\n## 📧 Gmail awaiting reply ({len(awaiting)})")
        for m in awaiting[:8]:
            lines.append(
                f"- {m.get('days_old',0)}d · {m.get('subject','')} "
                f"— {m.get('from','')}"
            )

    mail = ev.get("mail_unread") or []
    if mail:
        lines.append(f"\n## 📬 Apple Mail unread 36h ({len(mail)})")
        for m in mail[:5]:
            vip = "VIP · " if m.get("is_vip") else ""
            lines.append(f"- {vip}{m.get('subject','')} — {m.get('sender','')}")

    wa = ev.get("whatsapp") or []
    if wa:
        total = sum(int(w.get("count", 0)) for w in wa)
        lines.append(
            f"\n## 💬 WhatsApp últimas 24h ({len(wa)} chats · {total} msgs)"
        )
        for w in wa[:10]:
            snip = (w.get("last_snippet") or "").strip()[:100]
            snip_part = f' — "{snip}"' if snip else ""
            lines.append(
                f"- {w.get('name','?')} ({w.get('count',0)} msgs){snip_part}"
            )

    stale = ev.get("loops_stale") or []
    activo = ev.get("loops_activo") or []
    if stale or activo:
        lines.append(
            f"\n## 📁 Vault loops ({len(stale)} stale · {len(activo)} activo)"
        )
        for it in stale[:5]:
            src = Path(it["source_note"]).stem
            lines.append(
                f"- stale {it.get('age_days',0)}d: "
                f"{it.get('loop_text','')[:100]} [[{src}]]"
            )
        for it in activo[:5]:
            src = Path(it["source_note"]).stem
            lines.append(
                f"- activo {it.get('age_days',0)}d: "
                f"{it.get('loop_text','')[:100]} [[{src}]]"
            )

    contrad = ev.get("contradictions") or []
    if contrad:
        lines.append(f"\n## ⚠ Contradicciones recientes ({len(contrad)})")
        for c in contrad[:3]:
            tgt = ", ".join(t.get("path", "") for t in c.get("targets", [])[:2])
            lines.append(f"- {c.get('subject_path','')} ↔ {tgt}")

    return "\n".join(lines)


def _pendientes_services_consulted(ev: dict) -> list[str]:
    """Labels of services that returned data — used for UI meta."""
    out: list[str] = []
    if (ev.get("gmail") or {}).get("awaiting_reply"):
        out.append("Gmail")
    if ev.get("mail_unread"):
        out.append("Apple Mail")
    if ev.get("whatsapp"):
        out.append("WhatsApp")
    if ev.get("reminders"):
        out.append("Reminders")
    if ev.get("calendar_range") or ev.get("calendar"):
        out.append("Calendar")
    if ev.get("loops_stale") or ev.get("loops_activo"):
        out.append("Vault loops")
    return out


def _build_tasks_system_rules(scope: dict) -> str:
    """SYSTEM prompt with the concrete time-window embedded. Keeping it per-
    call (instead of a global constant) avoids the 'Sin pendientes' parroting
    failure mode — the model now has to filter by scope, not repeat a fixed
    fallback string.
    """
    return (
        "Sos un asistente personal que organiza la agenda del usuario a "
        "partir de evidencia EN VIVO de sus servicios (Calendar, Reminders, "
        "Gmail, Apple Mail, WhatsApp, loops del vault).\n\n"
        f"VENTANA TEMPORAL: {scope['label']}.\n\n"
        "Reglas estrictas:\n"
        "1. Usá SOLO la evidencia del CONTEXTO. No inventes tareas, "
        "reuniones ni plazos que no estén literalmente listados.\n"
        "2. Filtrá los items a la ventana temporal indicada. "
        "Items de Reminders fuera de la ventana → omitilos.\n"
        "3. Respondé en español, estructurado con encabezados markdown y "
        "bullets. Omití secciones que queden vacías después del filtro.\n"
        "4. Orden: 🚨 Urgente/overdue → 📅 Agenda (Calendar) → "
        "📌 Reminders → 📧 Bandeja (mails) → 💬 WhatsApp → 📁 Vault loops.\n"
        "5. Sé conciso: una línea por item. Para eventos de Calendar incluí "
        "día + hora + título.\n"
        "6. Si tras filtrar por la ventana no queda NADA, respondé "
        "exactamente: 'No tengo nada trackeado para esa ventana.' y parate "
        "ahí — NO copies la frase 'Sin pendientes trackeados' del contexto."
    )


def _gen_tasks_response(sess: dict, question: str, history: list[dict]):
    """SSE generator for the tasks/agenda intent branch. Collects scope-aware
    evidence from ALL registered vaults (home + work) via rag's multi-vault
    collector, builds the structured context, streams the LLM answer.

    Delegates formatting + prompt rules to `rag.py` so web and `rag serve`
    (WhatsApp/bot surface) produce identical output — same sections, same
    `[home]`/`[work]` loop labels, same "SIEMPRE mostrar" prompt semantics.
    """
    now = datetime.now()
    scope = _detect_time_scope(question)
    try:
        all_vaults = resolve_vault_paths(["all"])
    except Exception:
        all_vaults = []
    try:
        if all_vaults:
            ev = _rag_collect_scoped_tasks_evidence_multi(all_vaults, now, scope)
        else:
            ev = _collect_scoped_evidence(get_db(), now, scope)
    except Exception as exc:
        yield _sse("error", {"message": f"servicios fallaron: {exc}"})
        return
    urgent = _pendientes_urgent(ev, now)
    services = (
        _rag_tasks_services_consulted(ev) if all_vaults
        else _pendientes_services_consulted(ev)
    )
    context = (
        _rag_format_scoped_tasks_context(ev, urgent, now, scope) if all_vaults
        else _format_pendientes_context(ev, urgent, now, scope)
    )
    system_rules = _rag_build_tasks_system_rules(scope)

    meta_bits: list[str] = [f"📋 ventana: {scope['label']}"]
    if services:
        meta_bits.append(", ".join(services))
    else:
        meta_bits.append("sin datos de servicios")
    if urgent:
        meta_bits.append(f"🚨 {len(urgent)} urgente(s)")
    yield _sse("meta", {"bits": meta_bits})

    if history:
        messages = (
            [{"role": "system", "content": f"{system_rules}\nCONTEXTO:\n{context}"}]
            + history
            + [{"role": "user", "content": question}]
        )
    else:
        messages = [{"role": "user", "content": (
            f"{system_rules}\nCONTEXTO:\n{context}\n\n"
            f"PREGUNTA: {question}\n\nRESPUESTA:"
        )}]

    parts: list[str] = []
    # Tasks layout with 6+ sections + per-item bullets overflows the default
    # 384 num_predict cap — LLM stops mid-"Vault loops". Bump to 800 so the
    # full brief lands even with both vaults' loops expanded.
    tasks_options = {**CHAT_OPTIONS, "num_predict": 800}
    try:
        for chunk in _OLLAMA_STREAM_CLIENT.chat(
            model=resolve_chat_model(),
            messages=messages,
            options=tasks_options,
            stream=True,
            keep_alive=OLLAMA_KEEP_ALIVE,
        ):
            delta = chunk.message.content or ""
            if delta:
                parts.append(delta)
                yield _sse("token", {"delta": delta})
    except Exception as exc:
        yield _sse("error", {"message": f"LLM falló: {exc}"})
        return

    full = "".join(parts)
    turn_id = new_turn_id()
    append_turn(sess, {
        "q": question,
        "a": full,
        "paths": [],
        "top_score": 0.0,
        "turn_id": turn_id,
        "mode": "tasks",
    })
    save_session(sess)
    log_query_event({
        "cmd": "web.tasks",
        "turn_id": turn_id,
        "session": sess["id"],
        "q": question,
        "scope": scope["key"],
        "scope_label": scope["label"],
        "services": services,
        "n_urgent": len(urgent),
        "n_reminders": len(ev.get("reminders") or []),
        "n_calendar": len(
            ev.get("calendar_range") or ev.get("calendar") or []
        ),
        "n_whatsapp": len(ev.get("whatsapp") or []),
        "n_loops_activo": len(ev.get("loops_activo") or []),
        "n_loops_stale": len(ev.get("loops_stale") or []),
    })

    yield _sse("done", {"turn_id": turn_id, "mode": "tasks"})


_TODAY_FM_SPLIT = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _persist_today_brief(date_label: str, narrative: str) -> None:
    """Write the regenerated brief to `05-Reviews/YYYY-MM-DD-evening.md` so
    subsequent `/api/home` calls hit the cached path. Mirrors the CLI
    `rag today` write — same frontmatter shape so contradiction detection
    and ambient skip rules treat both files the same. Silent-fail.
    """
    from contextlib import suppress

    now_iso = datetime.now().isoformat(timespec="seconds")
    fm_lines = [
        "---",
        f"created: '{now_iso}'",
        "type: evening-brief",
        "tags:",
        "- review",
        "- evening-brief",
        f"date: '{date_label}'",
        "source: web",
        "---",
    ]
    body = "\n".join(fm_lines) + f"\n\n# Evening brief — {date_label}\n\n{narrative}\n"
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    with suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def _today_cached_narrative(date_label: str) -> str | None:
    """Return narrative from `05-Reviews/YYYY-MM-DD-evening.md` if present.
    CLI `rag today` writes this file, so we avoid re-running the LLM when the
    brief was already generated today.
    """
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    body = _TODAY_FM_SPLIT.sub("", raw, count=1).strip()
    # Strip the "# Evening brief — YYYY-MM-DD" header if present
    lines = body.splitlines()
    if lines and lines[0].startswith("# Evening brief"):
        body = "\n".join(lines[1:]).lstrip()
    return body or None


def _fetch_pagerank_top(col, n: int = 5) -> list[dict]:
    """Top-n notes by wikilink PageRank authority. Normalized pr (pr/max_pr).

    Reuses the module-level cache in rag.get_pagerank — cost is a dict sort
    after the first call. Silent-fail via caller suppressor.
    """
    pr_map = get_pagerank(col)
    if not pr_map:
        return []
    corpus = _load_corpus(col)
    ranked = sorted(pr_map.items(), key=lambda kv: -kv[1])[:n]
    # Defensive: `n <= 0` would make `ranked = []` despite a non-empty map,
    # and a hypothetical cache-invalidation race between `get_pagerank` and
    # `sorted` could also narrow it to zero rows. Return early rather than
    # IndexError-ing on `ranked[0][1]`.
    if not ranked:
        return []
    max_pr = ranked[0][1] or 1.0
    out: list[dict] = []
    for rank, (path, pr) in enumerate(ranked, start=1):
        title = _path_to_title(corpus, path) or Path(path).stem
        out.append({
            "title": title,
            "path": path,
            "pr": round(pr / max_pr, 4),
            "rank": rank,
        })
    return out


def _fetch_chrome_top_week(n: int = 5) -> list[dict]:
    """Top-n Chrome pages visited in the last 7 days.

    Chrome's `History` SQLite db is held open (locked) by the running
    browser — copy to a tmp file before reading. Chrome's `visit_time` is
    microseconds since 1601-01-01 UTC; convert to unix epoch for the iso
    stamp. Filters out chrome://, file://, and localhost.

    Known limitations (document, not fix — this endpoint is ambient signal,
    not a source of truth):
    - **Multi-profile**: only reads `Default/`. Chrome stores per-profile
      histories in sibling dirs (`Profile 1/History`, `Profile 2/History`, …).
      Users with multiple profiles get a partial top-n. `Local State` JSON
      would list them if we ever want to aggregate.
    - **WAL mode**: Chrome uses SQLite WAL; recent visits may live in
      `History-wal` until checkpointed. Copying only `History` misses the
      tail of "right now". Acceptable for a 7-day window.
    - **Guest mode / Incognito**: never written to disk by design — invisible
      to this fetcher, correctly so.
    """
    import shutil
    import sqlite3
    import tempfile

    src = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not src.is_file():
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600  # seconds between 1601-01-01 and 1970-01-01
    now_ts = time.time()
    week_ago_chrome = int((now_ts - 7 * 86_400 + CHROME_EPOCH_OFFSET) * 1_000_000)

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        shutil.copyfile(src, tmp.name)
        conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url, u.title AS title,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                  AND u.url NOT LIKE 'chrome://%'
                  AND u.url NOT LIKE 'chrome-extension://%'
                  AND u.url NOT LIKE 'file://%'
                  AND u.url NOT LIKE '%://localhost%'
                  AND u.url NOT LIKE '%://127.0.0.1%'
                GROUP BY u.url
                ORDER BY visit_count DESC
                LIMIT ?
                """,
                (week_ago_chrome, n),
            ).fetchall()
        finally:
            conn.close()

    out: list[dict] = []
    for r in rows:
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "title": r["title"] or "",
            "url": r["url"],
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
    return out


def _fetch_youtube_watched(n: int = 5, hours: int = 168) -> list[dict]:
    """Most-recent YouTube video pages opened in Chrome, last `hours` window.

    Same Chrome History DB + tmp-copy pattern as `_fetch_chrome_top_week`.
    Matches `youtube.com/watch`, `m.youtube.com/watch`, and `youtu.be/<id>`.
    Deduplicates by extracted video id (`v=...` for /watch, path segment
    for youtu.be) so re-opening the same video doesn't clutter the panel.
    Titles in Chrome history are the <title> tag at page load — the ad
    slate title occasionally leaks in instead of the real video title;
    acceptable noise for an ambient signal.
    """
    import shutil
    import sqlite3
    import tempfile
    from urllib.parse import urlparse, parse_qs

    src = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not src.is_file():
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    now_ts = time.time()
    window_start_chrome = int((now_ts - hours * 3_600 + CHROME_EPOCH_OFFSET) * 1_000_000)

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        shutil.copyfile(src, tmp.name)
        conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url, u.title AS title,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                  AND (
                       u.url LIKE '%://www.youtube.com/watch%'
                    OR u.url LIKE '%://youtube.com/watch%'
                    OR u.url LIKE '%://m.youtube.com/watch%'
                    OR u.url LIKE '%://youtu.be/%'
                  )
                GROUP BY u.url
                ORDER BY last_visit DESC
                LIMIT 50
                """,
                (window_start_chrome,),
            ).fetchall()
        finally:
            conn.close()

    seen_ids: set[str] = set()
    out: list[dict] = []
    for r in rows:
        url = r["url"]
        try:
            parsed = urlparse(url)
            if parsed.netloc.endswith("youtu.be"):
                vid = parsed.path.strip("/").split("/")[0] or url
            else:
                vid = (parse_qs(parsed.query).get("v") or [url])[0]
        except Exception:
            vid = url
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        # "Video Title - YouTube" → "Video Title"; Chrome appends " - YouTube"
        # to the page <title> for every watch page.
        raw_title = (r["title"] or "").strip()
        title = raw_title[:-len(" - YouTube")].rstrip() if raw_title.endswith(" - YouTube") else raw_title
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "title": title or url,
            "url": url,
            "video_id": vid,
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
        if len(out) >= n:
            break
    return out


# Baselines from CLAUDE.md (2026-04-16 post-quick-wins floor). Hardcoded
# because they're the reference against which the UI renders drift — changing
# the baseline needs a human decision, not an automatic update from the latest
# `rag eval` entry.
_EVAL_BASELINE = {
    "singles_hit5": 0.9048,
    "chains_hit5": 0.76,
    "chains_mrr": 0.58,
    "chain_success": 0.5556,
}


def _fetch_eval_trend(n: int = 10) -> dict | None:
    """Tail the last `n` entries of eval.jsonl and pair them with the
    hardcoded baseline. Silent-fail (returns None) if the log is missing —
    the CLI writes it only after the first `rag eval` run.
    """
    if not EVAL_LOG_PATH.is_file():
        return None
    history: list[dict] = []
    try:
        with EVAL_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return None
    if not history:
        return None
    history = history[-n:]
    return {
        "latest": history[-1],
        "baseline": dict(_EVAL_BASELINE),
        "history": history,
    }


# Followup aging runs the full `find_followup_loops` pipeline, which costs one
# LLM judge call per open loop. Cheap per-loop but the list scales with vault
# size — 6h cache keeps /api/home under ~2s while still refreshing a few
# times per working day.
_FOLLOWUP_AGING_CACHE: dict = {"ts": 0.0, "payload": None}
_FOLLOWUP_AGING_TTL = 6 * 3600


def _fetch_followup_aging() -> dict | None:
    """Bucketize open-loops-by-age for the /api/home aging widget.

    Buckets: 0-7d, 8-30d, stale_30plus (age≥31 OR status=='stale'). Uses a
    90d window on `find_followup_loops` so the 30+ bucket actually sees aged
    items (the default CLI window is 30d). Resolved loops are dropped —
    they are, by definition, no longer aging.
    """
    now_ts = time.time()
    cached = _FOLLOWUP_AGING_CACHE
    if cached["payload"] is not None and now_ts - cached["ts"] < _FOLLOWUP_AGING_TTL:
        return cached["payload"]

    try:
        col = get_db()
        loops = find_followup_loops(col, VAULT_PATH, days=90)
    except Exception:
        return cached["payload"]  # serve last-known-good on failure

    open_loops = [it for it in loops if it.get("status") != "resolved"]
    buckets = {"0_7": 0, "8_30": 0, "stale_30plus": 0}
    for it in open_loops:
        age = int(it.get("age_days") or 0)
        status = it.get("status") or ""
        if age >= 31 or status == "stale":
            buckets["stale_30plus"] += 1
        elif age >= 8:
            buckets["8_30"] += 1
        else:
            buckets["0_7"] += 1

    sample = sorted(open_loops, key=lambda it: -int(it.get("age_days") or 0))[:5]
    sample_out = [
        {
            "note": it.get("source_note") or "",
            "loop": (it.get("loop_text") or "")[:140],
            "age_days": int(it.get("age_days") or 0),
            "status": it.get("status") or "",
        }
        for it in sample
    ]

    payload = {
        "buckets": buckets,
        "total": len(open_loops),
        "sample": sample_out,
    }
    _FOLLOWUP_AGING_CACHE["ts"] = now_ts
    _FOLLOWUP_AGING_CACHE["payload"] = payload
    return payload


def _fetch_drive_recent(now: datetime, max_items: int = 5) -> list[dict]:
    """Files modified in Google Drive in the last 48h. Thin wrapper over
    rag._fetch_drive_evidence — keeps the home panel's window smaller than
    morning's 5d so "recent" on the dashboard means actionable-today, not
    week-in-review. Silent-fail via the underlying fetcher → [].
    """
    ev = _fetch_drive_evidence(now, days=2, max_items=max_items) or {}
    return ev.get("files") or []


def _fetch_whatsapp_unreplied(hours: int = 48, max_chats: int = 5) -> list[dict]:
    """Chats whose **last** message is inbound and sits without a reply.

    Distinct from `_fetch_whatsapp_unread` (counts all inbound in window):
    this one only surfaces conversations where *you* owe the next move.
    A chat with 20 inbound-then-replied messages does not appear here; a
    chat with 1 inbound-and-ignored message does.

    Uses a window function to pick the latest row per chat, then filters
    to `is_from_me = 0`. Window: `hours` bounds how stale a last-inbound
    is allowed to be — anything older is almost certainly abandoned and
    pollutes the panel.

    Schema note: the bridge stores timestamps in RFC3339-ish text; SQLite's
    `datetime()` handles it but raw string comparison would not — keep the
    `datetime(timestamp)` wrappers.
    """
    if not WHATSAPP_DB_PATH.is_file():
        return []
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            WITH last_msg AS (
              SELECT chat_jid, content, is_from_me, timestamp,
                     ROW_NUMBER() OVER (
                       PARTITION BY chat_jid
                       ORDER BY datetime(timestamp) DESC
                     ) AS rn
              FROM messages
              WHERE chat_jid != ?
                AND chat_jid NOT LIKE '%status@broadcast'
                AND datetime(timestamp) > datetime('now', ?)
            )
            SELECT lm.chat_jid   AS jid,
                   c.name        AS name,
                   lm.content    AS last_content,
                   lm.timestamp  AS last_ts
            FROM last_msg lm
            LEFT JOIN chats c ON c.jid = lm.chat_jid
            WHERE lm.rn = 1 AND lm.is_from_me = 0
            ORDER BY datetime(lm.timestamp) DESC
            LIMIT ?
            """,
            (WHATSAPP_BOT_JID, f"-{int(hours)} hours", int(max_chats) * 3),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    now_ts = time.time()
    out: list[dict] = []
    for r in rows:
        raw_name = (r["name"] or "").strip()
        jid_prefix = (r["jid"] or "").split("@")[0]
        display_name = raw_name or jid_prefix
        # Same "real name" gate as _fetch_whatsapp_unread: digit-only names
        # are unresolved @lid participants, unhelpful to surface.
        if not any(ch.isalpha() for ch in display_name):
            continue
        snippet = (r["last_content"] or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        # Bridge stores naive local timestamps; `fromisoformat` parses either
        # naive or tz-aware and `.timestamp()` does the right thing for both.
        try:
            last_dt = datetime.fromisoformat((r["last_ts"] or "").replace("Z", "+00:00"))
            hours_waiting = max(0.0, (now_ts - last_dt.timestamp()) / 3600.0)
        except Exception:
            hours_waiting = 0.0
        out.append({
            "jid": r["jid"],
            "name": display_name,
            "last_snippet": snippet,
            "hours_waiting": round(hours_waiting, 1),
        })
        if len(out) >= max_chats:
            break
    return out


# MOZE (Money app) export — user drops `MOZE_YYYYMMDD_HHMMSS.csv` into iCloud
# Backup. We pick the newest file and parse locally; no network, no API.
# Dates are MM/DD/YYYY (US format); Price uses ES decimals ("2026,74").
# Expenses are stored as negative numbers — abs() for display.
_FINANCE_BACKUP_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Backup"
_FINANCE_CACHE: dict = {"key": None, "payload": None}


def _fetch_finance(now: datetime | None = None) -> dict | None:
    """Parse the latest MOZE_*.csv export into a home-panel summary.

    Returns None (silent-fail) if: no CSV found, iCloud folder missing,
    or no rows parseable. Cached by (path, mtime) — re-exports invalidate.
    """
    import calendar
    import csv as _csv
    from collections import Counter

    now = now or datetime.now()
    try:
        csvs = sorted(
            _FINANCE_BACKUP_DIR.glob("MOZE_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return None
    if not csvs:
        return None
    src = csvs[0]
    try:
        mtime = src.stat().st_mtime
    except Exception:
        return None
    key = (str(src), mtime)
    if _FINANCE_CACHE["key"] == key:
        return _FINANCE_CACHE["payload"]

    def pnum(s: str) -> float:
        s = (s or "").strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    rows: list[tuple[datetime, dict]] = []
    try:
        with src.open(newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                raw = (r.get("Date") or "").strip()
                if not raw:
                    continue
                try:
                    d = datetime.strptime(raw, "%m/%d/%Y")
                except ValueError:
                    continue
                rows.append((d, r))
    except Exception:
        return None
    if not rows:
        return None

    def ym(d: datetime) -> tuple[int, int]:
        return (d.year, d.month)

    this = ym(now)
    anchor = datetime(now.year, now.month, 1)
    prev = (anchor.year - 1, 12) if anchor.month == 1 else (anchor.year, anchor.month - 1)

    expenses_this: dict[str, float] = {}
    expenses_prev: dict[str, float] = {}
    income_this: dict[str, float] = {}
    by_cat: dict[str, Counter] = {}

    for d, r in rows:
        cur = (r.get("Currency") or "").strip()
        typ = (r.get("Type") or "").strip()
        amt = abs(pnum(r.get("Price")))
        ymk = ym(d)
        if typ == "Expense":
            if ymk == this:
                expenses_this[cur] = expenses_this.get(cur, 0.0) + amt
                cat = (r.get("Main Category") or "—").strip() or "—"
                by_cat.setdefault(cur, Counter())[cat] += amt
            elif ymk == prev:
                expenses_prev[cur] = expenses_prev.get(cur, 0.0) + amt
        elif typ == "Income" and ymk == this:
            income_this[cur] = income_this.get(cur, 0.0) + amt

    rows.sort(key=lambda t: t[0], reverse=True)
    latest: list[dict] = []
    for d, r in rows:
        if len(latest) >= 5:
            break
        if (r.get("Type") or "").strip() not in ("Expense", "Income"):
            continue
        latest.append({
            "date": d.strftime("%Y-%m-%d"),
            "type": (r.get("Type") or "").strip(),
            "category": (r.get("Main Category") or "").strip(),
            "name": (r.get("Name") or "").strip(),
            "store": (r.get("Store") or "").strip(),
            "amount": pnum(r.get("Price")),
            "currency": (r.get("Currency") or "").strip(),
        })

    days_elapsed = now.day
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    ars_this = expenses_this.get("ARS", 0.0)
    ars_prev = expenses_prev.get("ARS", 0.0)
    run_rate = ars_this / days_elapsed if days_elapsed else 0.0
    projected = run_rate * days_in_month
    delta_pct = ((ars_this - ars_prev) / ars_prev * 100.0) if ars_prev else None

    top_cats: list[dict] = []
    for name, amt in (by_cat.get("ARS") or Counter()).most_common(5):
        top_cats.append({
            "name": name,
            "amount": amt,
            "share": (amt / ars_this) if ars_this else 0.0,
        })

    usd_this = expenses_this.get("USD", 0.0) + expenses_this.get("USDB", 0.0)
    usd_prev = expenses_prev.get("USD", 0.0) + expenses_prev.get("USDB", 0.0)

    payload = {
        "source_file": src.name,
        "source_mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        "month_label": now.strftime("%Y-%m"),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "ars": {
            "this_month": ars_this,
            "prev_month": ars_prev,
            "delta_pct": delta_pct,
            "run_rate_daily": run_rate,
            "projected": projected,
            "income": income_this.get("ARS", 0.0),
            "top_categories": top_cats,
        },
        "usd": {
            "this_month": usd_this,
            "prev_month": usd_prev,
        },
        "latest": latest,
    }
    _FINANCE_CACHE["key"] = key
    _FINANCE_CACHE["payload"] = payload
    return payload


# Cold `_home_compute` is 30–40s (12-fetcher fan-out, slowest = pendientes
# at ~25s). The user must NEVER block on that. Strategy:
#   1. Background pre-warmer thread recomputes every BG_INTERVAL, populating
#      _HOME_STATE["payload"] under lock.
#   2. /api/home reads from _HOME_STATE — instant 2ms regardless of TTL.
#   3. Stale-while-revalidate: if payload age > SOFT_TTL, serve stale and
#      kick a one-shot refresh thread (no double-compute thanks to the
#      `computing` flag).
#   4. First request before the pre-warmer finishes blocks once (cold path),
#      then every subsequent request is instant.
import threading
from fastapi.responses import Response

_HOME_LOCK = threading.Lock()
# Cache serialized bytes too — `/api/home` then becomes a raw bytes write
# (no per-request 35KB json.dumps under the GIL).
_HOME_STATE: dict = {
    "payload": None,
    "body": None,
    "ts": 0.0,
    "computing": False,
}
_HOME_SOFT_TTL = 120.0      # serve cached without bg-refresh under this age
# Pre-warmer cadence — bumped 25s → 300s on 2026-04-17 after chat latency
# blew up to 30-160s. Each cycle fans out 12 channel fetchers and can eat
# 6-27s of ollama + MPS + disk, which starves concurrent /api/chat requests
# waiting for the same ollama daemon. 5-minute refresh is plenty for a
# dashboard — the home page SWRs on visit anyway.
_HOME_BG_INTERVAL = 300.0
# Live chat-in-flight counter. While > 0, the prewarmer skips its cycle so
# chat gets exclusive use of ollama + the reranker. Simple counter (not a
# lock) because multiple concurrent chats are fine — they serialise on
# ollama anyway, and we only care about "nobody chatting" vs "someone is".
_CHAT_INFLIGHT = 0
_CHAT_INFLIGHT_LOCK = threading.Lock()
# Persist cache to disk so service restarts don't force a 30-45s cold
# compute on the next user visit. Disk payload is served immediately
# (stale OK — SWR refreshes in bg within seconds).
_HOME_CACHE_PATH = Path.home() / ".local/share/obsidian-rag/home_cache.json"
_HOME_DISK_TTL = 6 * 3600   # ignore disk cache older than this (stale-but-useful window)

_WARMING_BODY = json.dumps({
    "warming": True,
    "today": {"narrative": "", "narrative_source": "warming",
              "brief_path": None,
              "counts": {"recent_notes": 0, "inbox_today": 0, "todos": 0,
                         "new_contradictions": 0, "low_conf_queries": 0,
                         "total": 0},
              "evidence": {}},
    "urgent": [], "signals": {}, "tomorrow_calendar": [],
    "weather_forecast": None,
}).encode()


def _home_refresh(regenerate: bool = False) -> None:
    """Compute + publish under lock; never run two computes in parallel.
    Silent-fail keeps last good payload visible if a fetcher dependency
    (icalBuddy, whisper, ollama) is momentarily down.
    """
    with _HOME_LOCK:
        if _HOME_STATE["computing"]:
            return
        _HOME_STATE["computing"] = True
    t0 = time.time()
    try:
        payload = _home_compute(regenerate)
        body = json.dumps(payload, default=str).encode()
    except Exception:
        return
    finally:
        with _HOME_LOCK:
            _HOME_STATE["computing"] = False
    now = time.time()
    _HOME_STATE["payload"] = payload
    _HOME_STATE["body"] = body
    _HOME_STATE["ts"] = now
    print(f"[home-refresh] compute={(now - t0):.2f}s regen={regenerate}", file=sys.stderr)
    _persist_home_cache(body, now)


def _persist_home_cache(body: bytes, ts: float) -> None:
    """Write the serialized payload + timestamp to disk so a service
    restart can serve it immediately (instead of forcing a 30-45s cold
    compute on the next visit). Atomic rename so a crash mid-write never
    leaves a half-file. Silent-fail — the cache is optional.
    """
    from contextlib import suppress

    with suppress(Exception):
        _HOME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _HOME_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_bytes(b'{"ts":' + str(ts).encode()
                        + b',"body":' + body + b'}')
        tmp.replace(_HOME_CACHE_PATH)


def _load_home_cache() -> None:
    """Hydrate `_HOME_STATE` from disk on server startup. If the file is
    present and younger than `_HOME_DISK_TTL`, the first user visit sees
    stale-but-useful data immediately; SWR refreshes in the background.
    """
    if not _HOME_CACHE_PATH.is_file():
        return
    try:
        raw = _HOME_CACHE_PATH.read_bytes()
        data = json.loads(raw)
        ts = float(data.get("ts", 0))
        body_obj = data.get("body")
        if not ts or body_obj is None:
            return
        age = time.time() - ts
        if age > _HOME_DISK_TTL:
            return
        body_bytes = json.dumps(body_obj, default=str).encode()
        _HOME_STATE["body"] = body_bytes
        _HOME_STATE["payload"] = body_obj
        _HOME_STATE["ts"] = ts
        print(f"[home-cache] hydrated from disk age={age:.0f}s size={len(body_bytes)}B",
              file=sys.stderr)
    except Exception as exc:
        print(f"[home-cache] load failed: {exc}", file=sys.stderr)


@app.get("/api/home")
def home_api(regenerate: bool = False) -> Response:
    _ensure_home_prewarmer()
    # Explicit user action — recompute LLM brief synchronously.
    if regenerate:
        _home_refresh(regenerate=True)
        body = _HOME_STATE["body"] or _WARMING_BODY
        return Response(content=body, media_type="application/json")

    body = _HOME_STATE["body"]
    age = time.time() - _HOME_STATE["ts"] if body else float("inf")

    if body and age < _HOME_SOFT_TTL:
        return Response(content=body, media_type="application/json")
    if body:
        # SWR: return stale immediately, refresh in bg
        threading.Thread(
            target=_home_refresh, name="home-swr", daemon=True,
        ).start()
        return Response(content=body, media_type="application/json")
    # First call ever (pre-warmer still racing). Return placeholder so
    # the frontend skeleton stays up; auto-refresh picks up real payload.
    threading.Thread(
        target=_home_refresh, name="home-cold", daemon=True,
    ).start()
    return Response(content=_WARMING_BODY, media_type="application/json")


_HOME_PREWARMER_STARTED = False

# Opt-in prewarmer. Default OFF because the fan-out (12 channel fetchers
# including ollama-based signals) holds the ollama daemon mid-cycle and
# starves concurrent /api/chat requests — even a 300s interval isn't
# enough because a chat arriving mid-cycle still waits for the 7-27s
# compute to finish before its embed call gets served. The home page
# already uses SWR: first visit pays a cold compute (~10s), subsequent
# visits are instant with a bg refresh. That's the right trade.
# Set OBSIDIAN_RAG_HOME_PREWARM=1 to re-enable for display-only deploys.
_HOME_PREWARM_ENABLED = os.environ.get("OBSIDIAN_RAG_HOME_PREWARM") == "1"

def _ensure_home_prewarmer() -> None:
    """Start the pre-warmer loop iff opt-in. Called from startup and from
    /api/home. Idempotent.
    """
    global _HOME_PREWARMER_STARTED
    if _HOME_PREWARMER_STARTED or not _HOME_PREWARM_ENABLED:
        _HOME_PREWARMER_STARTED = True  # poison so we don't retry the check
        return
    _HOME_PREWARMER_STARTED = True

    def loop() -> None:
        while True:
            try:
                # Skip the cycle while a chat request is in flight — each
                # home-compute fans out 12 channel fetchers (some hitting
                # ollama for helper-model judgments) and would queue behind
                # or evict the chat model, turning a 3s chat into 60s+.
                if _CHAT_INFLIGHT > 0:
                    time.sleep(10)
                    continue
                _home_refresh(regenerate=False)
            except Exception:
                pass
            time.sleep(_HOME_BG_INTERVAL)

    threading.Thread(target=loop, name="home-prewarmer", daemon=True).start()


# Chat-model prewarmer. Root cause: ollama evicts the /chat model from VRAM
# between requests despite keep_alive=-1 when memory pressure builds up
# (observed: qwen2.5:7b dropped after ~5min idle while qwen2.5:3b + bge-m3
# + reranker stayed resident). Each cold reload = 4-7s prefill penalty on
# the next /api/chat, turning warm 2.9s into 10s+.
#
# Fix: every _CHAT_PREWARM_INTERVAL seconds, send a 1-token ping to the
# resolved chat model with keep_alive=-1. This re-pins the model in VRAM
# even if ollama decided to evict during idle. Skipped while a real chat
# is in flight (the chat itself keeps the model warm — no need to duplicate).
#
# Cost: ~100-200ms of ollama compute per cycle, ~5GB VRAM pinned continuously.
# Safe on 36GB unified memory with command-r NOT also resident.
_CHAT_PREWARM_INTERVAL = int(os.environ.get("OBSIDIAN_RAG_CHAT_PREWARM_INTERVAL", "240"))
_CHAT_PREWARMER_STARTED = False


def _ensure_chat_model_prewarmer() -> None:
    """Start the chat-model prewarm loop once. Idempotent."""
    global _CHAT_PREWARMER_STARTED
    if _CHAT_PREWARMER_STARTED:
        return
    _CHAT_PREWARMER_STARTED = True

    def loop() -> None:
        # First cycle runs after a short startup delay so the app finishes
        # booting (corpus cache, home hydration) before we pin the LLM.
        # Subsequent cycles run every _CHAT_PREWARM_INTERVAL.
        time.sleep(15)
        while True:
            try:
                if _CHAT_INFLIGHT > 0:
                    time.sleep(15)
                    continue
                model = _resolve_web_chat_model()
                # Tiny ping — 1 token is enough to re-pin KV cache + model weights.
                # options=keep_alive=-1 instructs ollama to hold the model forever,
                # overriding its internal eviction pressure. Uses the bounded
                # streaming client so a stuck-load daemon can't wedge this
                # thread forever and mask the next real request.
                _OLLAMA_STREAM_CLIENT.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": _WEB_SYSTEM_PROMPT},
                        {"role": "user", "content": "."},
                    ],
                    options={"num_predict": 1, "num_ctx": _WEB_CHAT_NUM_CTX,
                             "temperature": 0, "seed": 42},
                    stream=False,
                    think=False,   # match the probe + main chat path
                    keep_alive=-1,
                )
                print(f"[chat-prewarm] {model} pinned", flush=True)
            except Exception as exc:
                # Silent fail: ollama down, model not loaded, network blip.
                # Next cycle retries. Never crash the daemon thread.
                print(f"[chat-prewarm] skipped: {exc}", flush=True)
            time.sleep(_CHAT_PREWARM_INTERVAL)

    threading.Thread(target=loop, name="chat-model-prewarmer", daemon=True).start()


# ── Response cache ──────────────────────────────────────────────────────
# LRU de respuestas completas de /api/chat para queries exactas repetidas.
# Cuando el mismo user query llega dentro del TTL y el vault no cambió,
# servimos la respuesta cacheada como SSE replay (<100ms wall) en lugar de
# re-correr retrieve + LLM (2-3s).
#
# Cache key: sha256(question|vault_scope|chat_model|vault_chunks_count)[:16].
# Incluir el chunks count efectivamente invalida el cache cuando el vault
# gana/pierde notas (count cambia → key cambia). TTL secundario 5min para
# invalidar respuestas sin cambio de vault pero con información que puede
# haber envejecido (ej. `rag tune` refinó pesos → retrieval diferente).
#
# No cacheamos cuando:
#   - hay history (follow-ups dependen del turno previo, el key no refleja eso)
#   - el response fue vacío/error
#   - _wa_in_query matcheó (WA data change rápidamente, datos frescos importan)
_CHAT_CACHE: "OrderedDict[str, dict]" = __import__("collections").OrderedDict()
_CHAT_CACHE_MAX = 100
_CHAT_CACHE_TTL = 300.0  # 5 min
_CHAT_CACHE_LOCK = threading.Lock()


def _chat_cache_key(question: str, vault_scope: str, model: str, vault_chunks: int) -> str:
    raw = f"{question.strip().lower()}|{vault_scope}|{model}|{vault_chunks}"
    return __import__("hashlib").sha256(raw.encode()).hexdigest()[:16]


def _chat_cache_get(key: str) -> dict | None:
    with _CHAT_CACHE_LOCK:
        entry = _CHAT_CACHE.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > _CHAT_CACHE_TTL:
            del _CHAT_CACHE[key]
            return None
        _CHAT_CACHE.move_to_end(key)
        return dict(entry)


def _chat_cache_put(key: str, payload: dict) -> None:
    with _CHAT_CACHE_LOCK:
        _CHAT_CACHE[key] = {"ts": time.time(), **payload}
        _CHAT_CACHE.move_to_end(key)
        while len(_CHAT_CACHE) > _CHAT_CACHE_MAX:
            _CHAT_CACHE.popitem(last=False)


def _home_compute(regenerate: bool = False) -> dict:
    """Centralizer — aggregates every information channel the user cares
    about into one JSON payload. Powers the home page.

    Channels:
    - today narrative (rag today) + vault evidence for the day
    - live signals (reminders, Apple Mail, Gmail, WhatsApp, calendar, weather)
    - derived signals (open loops, contradictions, low-confidence queries)
    - tomorrow agenda + multi-day weather forecast

    Per-channel silent-fail: if one source errors (e.g. Gmail OAuth expired,
    icalBuddy not installed), that key is missing/empty but the rest renders.
    `regenerate=true` forces a fresh LLM narrative (~10s); default reuses the
    cached brief from `05-Reviews/<date>-evening.md` if present.
    """
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import suppress

    now = datetime.now()
    date_label = now.strftime("%Y-%m-%d")
    col = get_db()

    # ── Fan out: vault evidence + live channels + forecast run in parallel.
    # `_pendientes_collect` internally fans out its 9 fetchers too, so the
    # critical path across the whole page is max(today_evidence, slowest
    # single fetcher, tomorrow_calendar, weather_forecast) instead of the sum.
    # Don't use `with ThreadPoolExecutor(...) as pool:` — its __exit__ calls
    # `shutdown(wait=True)` and blocks the whole endpoint until every future
    # finishes, defeating the per-future timeouts below. We submit, collect
    # with timeouts, and then `shutdown(wait=False)` to return fast while
    # any straggler (e.g. cold `_fetch_followup_aging` doing LLM-judge per
    # loop) keeps running in the background to warm its own cache.
    pool = ThreadPoolExecutor(max_workers=14, thread_name_prefix="home")
    timings: dict[str, float] = {}
    t_submit = time.time()

    def _timed(name: str, fn, *args):
        t0 = time.time()
        try:
            return fn(*args)
        finally:
            timings[name] = time.time() - t0

    try:
        fut_today       = pool.submit(
            _timed, "today", _collect_today_evidence,
            now, VAULT_PATH, LOG_PATH, CONTRADICTION_LOG_PATH,
        )
        fut_signals     = pool.submit(_timed, "signals", _pendientes_collect, col, now, 14)
        fut_tomorrow    = pool.submit(_timed, "tomorrow", _fetch_calendar_ahead, 1, 10)
        fut_forecast    = pool.submit(_timed, "forecast", _fetch_weather_forecast)
        fut_pagerank    = pool.submit(_timed, "pagerank", _fetch_pagerank_top, col, 5)
        fut_chrome      = pool.submit(_timed, "chrome", _fetch_chrome_top_week, 5)
        fut_eval        = pool.submit(_timed, "eval", _fetch_eval_trend)
        fut_followup    = pool.submit(_timed, "followup", _fetch_followup_aging)
        fut_drive       = pool.submit(_timed, "drive", _fetch_drive_recent, now, 5)
        fut_wa_unreplied = pool.submit(_timed, "wa_unreplied", _fetch_whatsapp_unreplied, 48, 5)
        fut_bookmarks   = pool.submit(_timed, "bookmarks", _fetch_chrome_bookmarks_used, 48, 5)
        fut_vaults      = pool.submit(_timed, "vaults", _fetch_vault_activity, 48, 5)
        fut_finance     = pool.submit(_timed, "finance", _fetch_finance, now)
        fut_youtube     = pool.submit(_timed, "youtube", _fetch_youtube_watched, 5, 168)

        try:
            today_ev = fut_today.result(timeout=30)
        except Exception as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            raise HTTPException(status_code=500, detail=str(exc))

        # Pendientes itself fans out 9 fetchers; cold path measured ~25s.
        # Bump cap to 45s — the user never waits on this since pre-warmer
        # eats the cold compute. If it still times out, the next cycle
        # repopulates and SWR keeps stale signals visible meanwhile.
        signals = {}
        with suppress(Exception):
            signals = fut_signals.result(timeout=45)

        tomorrow_calendar = []
        with suppress(Exception):
            tomorrow_calendar = fut_tomorrow.result(timeout=10) or []

        weather_forecast = None
        with suppress(Exception):
            weather_forecast = fut_forecast.result(timeout=10)

        pagerank_top: list[dict] = []
        with suppress(Exception):
            pagerank_top = fut_pagerank.result(timeout=10) or []

        chrome_top_week: list[dict] = []
        with suppress(Exception):
            chrome_top_week = fut_chrome.result(timeout=5) or []

        eval_trend: dict | None = None
        with suppress(Exception):
            eval_trend = fut_eval.result(timeout=5)

        # followup_aging has its own 6h cache + LLM-judge per loop on cold.
        # If not ready within 2s, skip it this cycle — the bg thread keeps
        # computing and the next prewarmer pass (25s later) picks up the
        # warm cache. This shaves ~15s off the cold critical path.
        followup_aging: dict | None = None
        with suppress(Exception):
            followup_aging = fut_followup.result(timeout=2)

        drive_recent: list[dict] = []
        with suppress(Exception):
            drive_recent = fut_drive.result(timeout=10) or []

        whatsapp_unreplied: list[dict] = []
        with suppress(Exception):
            whatsapp_unreplied = fut_wa_unreplied.result(timeout=10) or []

        chrome_bookmarks: list[dict] = []
        with suppress(Exception):
            chrome_bookmarks = fut_bookmarks.result(timeout=5) or []

        vault_activity: dict = {}
        with suppress(Exception):
            vault_activity = fut_vaults.result(timeout=10) or {}

        finance: dict | None = None
        with suppress(Exception):
            finance = fut_finance.result(timeout=5)

        youtube_watched: list[dict] = []
        with suppress(Exception):
            youtube_watched = fut_youtube.result(timeout=5) or []

        signals["pagerank_top"] = pagerank_top
        signals["chrome_top_week"] = chrome_top_week
        signals["eval_trend"] = eval_trend
        signals["followup_aging"] = followup_aging
        signals["drive_recent"] = drive_recent
        signals["whatsapp_unreplied"] = whatsapp_unreplied
        signals["chrome_bookmarks"] = chrome_bookmarks
        signals["vault_activity"] = vault_activity
        signals["finance"] = finance
        signals["youtube_watched"] = youtube_watched
    finally:
        # Detach stragglers; they continue warming caches for the next call.
        pool.shutdown(wait=False)
        if timings:
            ranked = sorted(timings.items(), key=lambda kv: -kv[1])
            summary = " ".join(f"{k}={v:.1f}s" for k, v in ranked if v >= 0.2)
            if summary:
                print(f"[home-compute] {summary} total={time.time() - t_submit:.1f}s",
                      file=sys.stderr)

    today_total = (
        len(today_ev.get("recent_notes") or [])
        + len(today_ev.get("inbox_today") or [])
        + len(today_ev.get("todos") or [])
        + len(today_ev.get("new_contradictions") or [])
        + len(today_ev.get("low_conf_queries") or [])
    )

    narrative = "" if today_total == 0 else (_today_cached_narrative(date_label) or "")
    narrative_source = "cached" if narrative else "none"
    # Only call the LLM when the caller explicitly asks. Default path stays
    # fast — if no cached brief exists yet, the UI shows "pendiente" and
    # offers a button that re-hits with regenerate=true.
    if today_total > 0 and regenerate:
        # Inject the monthly MOZE snapshot so the LLM narrative can cite
        # cuánto gastaste este mes — otherwise it refuses ("no encontré
        # finanzas") even though the home panel renders the numbers.
        finance_snapshot = signals.get("finance") if isinstance(signals, dict) else None
        if finance_snapshot:
            today_ev = {**today_ev, "finance": finance_snapshot}
        prompt = _render_today_prompt(date_label, today_ev)
        narrative = _generate_today_narrative(prompt)
        narrative_source = "generated" if narrative else "error"
        if narrative:
            _persist_today_brief(date_label, narrative)

    brief_rel: str | None = None
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    if path.is_file():
        with suppress(ValueError):
            brief_rel = str(path.relative_to(VAULT_PATH))

    urgent: list[str] = []
    with suppress(Exception):
        urgent = _pendientes_urgent(signals, now)

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "date": date_label,
        "today": {
            "narrative": narrative,
            "narrative_source": narrative_source,
            "brief_path": brief_rel,
            "counts": {
                "recent_notes": len(today_ev.get("recent_notes") or []),
                "inbox_today": len(today_ev.get("inbox_today") or []),
                "todos": len(today_ev.get("todos") or []),
                "new_contradictions": len(today_ev.get("new_contradictions") or []),
                "low_conf_queries": len(today_ev.get("low_conf_queries") or []),
                "total": today_total,
            },
            "evidence": today_ev,
        },
        "urgent": urgent,
        "signals": signals,
        "tomorrow_calendar": tomorrow_calendar,
        "weather_forecast": weather_forecast,
    }


@app.get("/api/pendientes")
def pendientes_api(days: int = 14) -> dict:
    """Raw JSON view of the services layer. Useful for a dedicated pendientes
    card in the UI or for third-party clients; the chat endpoint uses the
    same `_pendientes_collect` internally when intent matches.
    """
    col = get_db()
    now = datetime.now()
    try:
        ev = _pendientes_collect(col, now, days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    urgent = _pendientes_urgent(ev, now)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "days": days,
        "urgent": urgent,
        "services_consulted": _pendientes_services_consulted(ev),
        "evidence": ev,
    }


_CONV_PENDING_PATH = Path.home() / ".local/share/obsidian-rag" / "conversation_turn_pending.jsonl"

# Track in-flight conversation-writer threads so the shutdown hook can
# drain them. We stay with daemon=True (a wedged SQL write must not block
# process exit) but give the event loop a 5s join window — enough for a
# normal SQL upsert + atomic file write. Turns still in flight after the
# timeout fall through to `_append_pending_conversation_turn`'s retry file
# via the exception path in `_persist_conversation_turn`, or were already
# persisted but logged after the deadline (acceptable).
_CONV_WRITERS_LOCK = threading.Lock()
_CONV_WRITERS: "set[threading.Thread]" = set()


def _spawn_conversation_writer(
    target_args: tuple,
    name: str,
) -> threading.Thread:
    """Launch `_persist_conversation_turn` on a daemon thread while
    registering it for the shutdown-drain hook. The wrapper removes the
    thread from the tracker on completion (success or failure) so the set
    doesn't grow unbounded across sessions.
    """
    def _wrapper() -> None:
        try:
            _persist_conversation_turn(*target_args)
        finally:
            with _CONV_WRITERS_LOCK:
                _CONV_WRITERS.discard(threading.current_thread())

    t = threading.Thread(target=_wrapper, name=name, daemon=True)
    with _CONV_WRITERS_LOCK:
        _CONV_WRITERS.add(t)
    t.start()
    return t


@app.on_event("shutdown")
def _drain_conversation_writers() -> None:
    """On server stop, give in-flight conversation writers up to 5s to
    finish. Any that don't make it land in the retry queue on disk via
    `_append_pending_conversation_turn` (the writer's own except path) and
    will be re-applied at next startup by `_retry_pending_conversation_turns`.

    We do NOT block indefinitely — a wedged SQL write should never prevent
    the process from exiting.
    """
    with _CONV_WRITERS_LOCK:
        pending = list(_CONV_WRITERS)
    if not pending:
        return
    import time as _time
    deadline = _time.monotonic() + 5.0
    for t in pending:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break
        t.join(timeout=remaining)
    with _CONV_WRITERS_LOCK:
        stragglers = [t for t in _CONV_WRITERS if t.is_alive()]
    if stragglers:
        try:
            _LOG_QUEUE.put((
                LOG_PATH,
                json.dumps({
                    "kind": "conversation_writer_shutdown_timeout",
                    "count": len(stragglers),
                }) + "\n",
            ))
        except Exception:
            pass


def _append_pending_conversation_turn(rec: dict) -> None:
    """Append a serialisable turn record to the pending-retry file.

    The file is a JSONL ring buffer consumed by _retry_pending_conversation_turns
    at server startup and (best-effort) periodically. Each record is a
    self-contained snapshot — vault_root str, session_id, question, answer,
    sources, confidence, iso timestamp — so re-hydration doesn't need any
    runtime context.
    """
    try:
        _CONV_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CONV_PENDING_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Dead-letter failed too — nothing else we can do without blocking
        # the SSE response. The LOG_QUEUE error record above is the last
        # signal the operator gets.
        pass


def _sanitize_confidence(confidence) -> float:
    """Clamp NaN/±Inf/None/garbage confidences → 0.0 so they never leak
    into persisted state. `retrieve()` returns `float('-inf')` when the
    corpus was empty (meta-chat, zero vec rows); persisting that ended up
    serialised as `-Infinity` inside `conversation_turn_pending.jsonl`
    (valid json via allow_nan=True, but no portable YAML/strict-json
    consumer can round-trip it) and as the literal `-inf` inside the
    conversation frontmatter (`confidence_avg: -inf`), which then broke
    the turn-2 averaging with `float("-inf") * 1 + x` propagating as
    `-inf` forever. Callers that go through `_retry_pending_conversation_turns`
    may also pass `None`/str from malformed pending records — swallow
    both into the 0.0 fallback rather than raising.
    """
    import math
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(c) or math.isinf(c):
        return 0.0
    return c


def _persist_conversation_turn(
    vault_root: Path,
    session_id: str,
    question: str,
    answer: str,
    metas: list[dict],
    scores: list[float],
    confidence: float,
) -> None:
    sources_payload = [
        {"file": m.get("file", ""), "score": float(s)}
        for m, s in zip(metas, scores)
    ]
    clean_conf = _sanitize_confidence(confidence)
    ts = datetime.now(timezone.utc)
    try:
        turn = TurnData(
            question=question,
            answer=answer,
            sources=sources_payload,
            confidence=clean_conf,
            timestamp=ts,
        )
        path = write_turn(vault_root, session_id, turn)
        _LOG_QUEUE.put((
            LOG_PATH,
            json.dumps({
                "kind": "conversation_turn_written",
                "session_id": session_id,
                "path": str(path.relative_to(vault_root)),
            }) + "\n",
        ))
    except Exception as exc:
        # Persist the turn payload to a retry queue on disk so a transient
        # SQL / fs failure doesn't drop it silently. Consumed at startup by
        # _retry_pending_conversation_turns. Errors still surface via
        # LOG_QUEUE for operator visibility, but data is not lost.
        _append_pending_conversation_turn({
            "ts": ts.isoformat(timespec="seconds"),
            "vault_root": str(vault_root),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": sources_payload,
            "confidence": clean_conf,
            "error": repr(exc),
        })
        _LOG_QUEUE.put((
            LOG_PATH,
            json.dumps({
                "kind": "conversation_turn_error",
                "session_id": session_id,
                "error": repr(exc),
                "queued_for_retry": True,
            }) + "\n",
        ))


def _retry_pending_conversation_turns() -> int:
    """Drain _CONV_PENDING_PATH on startup. Returns number of turns retried
    successfully. Best-effort: failures stay in the file for the next run.
    """
    if not _CONV_PENDING_PATH.is_file():
        return 0
    try:
        raw = _CONV_PENDING_PATH.read_text(encoding="utf-8")
    except Exception:
        return 0
    remaining: list[str] = []
    retried = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            ts_str = rec["ts"]
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Sanitize `confidence` on replay too — legacy pending files
            # written before _sanitize_confidence existed may carry
            # -Infinity / NaN, which round-trip through `float(...)` and
            # silently re-poison the restored frontmatter.
            turn = TurnData(
                question=rec["question"],
                answer=rec["answer"],
                sources=rec.get("sources", []),
                confidence=_sanitize_confidence(rec.get("confidence", 0.0)),
                timestamp=ts,
            )
            write_turn(Path(rec["vault_root"]), rec["session_id"], turn)
            retried += 1
        except Exception:
            # Keep the line for the next retry cycle.
            remaining.append(line)
    try:
        if remaining:
            _CONV_PENDING_PATH.write_text("\n".join(remaining) + "\n",
                                           encoding="utf-8")
        else:
            _CONV_PENDING_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    return retried


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    # Rate limit: 30 chat requests / 60s per IP. Each request pins the
    # chat model + reranker + embedder on MPS — a tight loop from a
    # runaway client can starve the daemon for legitimate work.
    client_ip = (request.client.host if request.client else "unknown")
    _check_rate_limit(_CHAT_BUCKETS, client_ip,
                      _CHAT_RATE_LIMIT, _CHAT_RATE_WINDOW)
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    # Detect create-intent EARLY. When true we skip emitting `sources`
    # (vault citations are noise when the user is creating something new,
    # not asking about existing notes) and we bypass the read-intent
    # pre-router further down.
    is_propose_intent = _detect_propose_intent(question)

    # Meta-chat short-circuit — greetings / thanks / "what can you do".
    # 2026-04-21 Playwright probe (Fer F.): "hola" produced "Según tus
    # notas, tenés varias interacciones con diferentes contactos por
    # WhatsApp..." because `_WEB_SYSTEM_PROMPT` REGLA 1 forces the LLM
    # to engage with whatever retrieved context it got. Cheapest fix:
    # detect the bare social / meta turns up front and reply with a
    # canned line, skipping retrieval + tool-calling + LLM entirely.
    # Zero latency (<1ms) + no hallucination possible. See
    # `_detect_metachat_intent` for the matcher shape + rationale.
    is_metachat = (not is_propose_intent) and _detect_metachat_intent(question)

    sid = req.session_id or f"web:{uuid.uuid4().hex[:12]}"
    sess = ensure_session(sid, mode="chat")
    vaults = _resolve_scope(req.vault_scope)
    if not vaults:
        raise HTTPException(status_code=400, detail=f"vault '{req.vault_scope}' no encontrado")

    def _emit_enrich(turn_id: str, q: str, answer: str, top_score: float):
        """Yield an `enrich` SSE event with cross-source signals (WA/Calendar/
        Reminders). 4s wall budget enforced via ThreadPoolExecutor.
        Soft-fail per source — never raises into the stream.
        """
        print(f"[enrich] start turn={turn_id} top_score={top_score} answer_len={len(answer or '')}", flush=True)
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
        try:
            from rag import build_enrich_payload
            with ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(build_enrich_payload, q, answer, top_score)
                _enrich = _fut.result(timeout=4.0)
            if _enrich:
                print(f"[enrich] hit lines={len(_enrich.get('lines', []))}", flush=True)
                return _sse("enrich", {"turn_id": turn_id, **_enrich})
            print("[enrich] no lines (empty payload)", flush=True)
        except _FutTimeout:
            print("[enrich] skipped: 4s budget exceeded", flush=True)
        except Exception as exc:
            print(f"[enrich] skipped: {type(exc).__name__}: {exc}", flush=True)
        return None

    def gen():
        _t0 = time.perf_counter()
        yield _sse("session", {"id": sess["id"]})

        # Meta-chat short-circuit. Canned responses (varied across
        # WA-style variants so repeated "hola" doesn't always say the
        # same thing). Random seeded by minute so the same message in
        # a tight window picks the same variant; tests can monkey-patch
        # `_METACHAT_PICKER` to force a specific variant.
        if is_metachat:
            reply = _pick_metachat_reply(question)
            turn_id = new_turn_id()
            # Stream token-by-token to mimic the shape of a real answer
            # so the UI's token-append animation still plays — keeps UX
            # consistent whether or not retrieval ran.
            yield _sse("sources", {
                "items": [], "confidence": None, "metachat": True,
            })
            yield _sse("status", {"stage": "generating"})
            for i in range(0, len(reply), 40):
                yield _sse("token", {"delta": reply[i:i+40]})
            total_ms = int((time.perf_counter() - _t0) * 1000)
            yield _sse("done", {
                "turn_id": turn_id, "elapsed_ms": total_ms,
                "metachat": True,
            })
            # Log + persist turn so the session + analytics see it.
            try:
                append_turn(sess, {
                    "turn_id": turn_id, "q": question,
                    "a": reply[:500], "metachat": True,
                })
                save_session(sess)
            except Exception:
                pass
            try:
                log_query_event({
                    "cmd": "web.chat.metachat", "q": question[:200],
                    "session": sess["id"], "answered": True,
                    "t_total": round(total_ms / 1000.0, 3),
                })
            except Exception:
                pass
            return

        history = session_history(sess, window=SESSION_HISTORY_WINDOW)

        # Response cache — sirve respuesta cacheada si (question, scope, model,
        # vault_count) matchea + TTL viva + NO es follow-up. Skipped cuando hay
        # history porque la respuesta depende del contexto de la conversación,
        # que el key actual no refleja.
        # Nota: `_cache_key` queda pre-inicializado en None porque el topic-
        # shift gate (~línea 3914) puede reasignar `history = []` DESPUÉS de
        # este bloque; en ese caso, el PUT path abajo entraba con `_cache_key`
        # no-bound y deja un `UnboundLocalError` visible en web.log como
        # `[chat-cache] put failed: cannot access local variable '_cache_key'…`.
        _cache_key: str | None = None
        if not history:
            try:
                from rag import get_db_for
                _vault_chunks = get_db_for(vaults[0][1]).count() if vaults else 0
            except Exception:
                _vault_chunks = 0
            _cache_key = _chat_cache_key(
                question, req.vault_scope or "", _resolve_web_chat_model(), _vault_chunks
            )
            _cached = _chat_cache_get(_cache_key)
            if _cached:
                # Replay completo como SSE. El UI no distingue cached de live.
                yield _sse("status", {"stage": "cached"})
                if not is_propose_intent:
                    yield _sse("sources", {
                        "items": _cached["sources_items"],
                        "confidence": _cached["top_score"],
                    })
                # Stream el texto en chunks chicos para mantener la ilusión
                # de streaming (el cliente ya espera SSE tokens). 40 chars ≈
                # 10 tokens — UI renderea suave sin el feel "pegote instantáneo".
                _text = _cached["text"]
                for i in range(0, len(_text), 40):
                    yield _sse("token", {"delta": _text[i:i+40]})
                _cached_total_ms = int((time.perf_counter() - _t0) * 1000)
                _cached_turn_id = new_turn_id()
                yield _sse("done", {
                    "turn_id": _cached_turn_id,
                    "top_score": _cached["top_score"],
                    "total_ms": _cached_total_ms,
                    "retrieve_ms": 0,
                    "ttft_ms": _cached_total_ms,
                    "llm_ms": 0,
                    "cached": True,
                })
                _enrich_evt = _emit_enrich(
                    _cached_turn_id, question, _text, float(_cached["top_score"]),
                )
                if _enrich_evt:
                    yield _enrich_evt
                print(f"[chat-cache] HIT {_cache_key} total={_cached_total_ms}ms", flush=True)
                # Emit a timing schema-stable line so downstream parsers
                # (tool_rounds/tool_ms/tool_names) don't see an NaN on
                # cache-hit turns. Values are 0/empty — tool loop skipped.
                print(
                    f"[chat-timing] model={_resolve_web_chat_model()} retrieve=0ms "
                    f"reform=0ms reform_outcome=skipped q_words={len(question.split())} "
                    f"wa=0ms llm_prefill=0ms llm_decode=0ms ttft={_cached_total_ms}ms "
                    f"total={_cached_total_ms}ms ctx_chars=0 tokens≈{len(_text.split())} "
                    # Defense-in-depth: el cached top_score pasa por
                    # `round(float(...), 3)` en el PUT (línea ~4476). Post
                    # fix del persist path queda finito, pero saneamos por
                    # las dudas para que el log de cache-hit no cargue un
                    # `-inf` si algún día alguien persiste crudo.
                    f"confidence={_sanitize_confidence(_cached['top_score']):.3f} variants=0 "
                    f"tool_rounds=0 tool_ms=0 tool_names= cached=1",
                    flush=True,
                )
                # Todavía appendeamos al session history — la conversación continúa normal
                append_turn(sess, {
                    "q": question,
                    "a": _text,
                    "paths": [s.get("file", "") for s in _cached["sources_items"]],
                    "top_score": _cached["top_score"],
                    "turn_id": new_turn_id(),
                })
                save_session(sess)
                return

        # Fail fast if ollama is in the "stuck-load" state — accepts HTTP
        # but never responds. Two-layer probe: /api/tags is the cheap
        # "daemon listening?" check; the deep chat probe catches the mode
        # where tags responds but chat hangs (seen 2026-04-19 — tags OK,
        # /api/chat never flushed first chunk). On deep-probe failure we
        # auto-heal via `brew services restart ollama` and re-probe once
        # before giving up, so the user's next /api/chat request doesn't
        # need a manual panic-button press.
        if not _ollama_alive(timeout=2.0) or not _ollama_chat_probe(timeout_s=6.0):
            print("[ollama-preflight] stuck-load detected — auto-restarting", flush=True)
            if not _ollama_restart_if_stuck() or not _ollama_chat_probe(timeout_s=8.0):
                yield _sse("error", {
                    "message": "Ollama no responde (stuck-load). Auto-restart falló. "
                    "Probá: brew services restart ollama",
                })
                return
            print("[ollama-preflight] recovered via restart", flush=True)

        # Kick off WhatsApp fetch in parallel with retrieve so the SQLite
        # round-trip (25-180ms) overlaps with the heavier retrieval work
        # instead of stacking sequentially before the LLM call.
        from concurrent.futures import ThreadPoolExecutor
        _wa_executor = ThreadPoolExecutor(max_workers=1)
        _t_wa_start = time.perf_counter()
        _wa_future = _wa_executor.submit(_fetch_whatsapp_unread, 24, 8)

        # Conversation-aware reformulation is a qwen2.5:3b call that costs
        # 0.6-2s. Only worth paying for short follow-ups that rely on prior
        # context ("y eso?", "más sobre ella"). Standalone questions ≥4 words
        # without anaphora cues skip it — the original query is already a
        # good search target.
        #
        # Two-layer strategy when a follow-up IS detected:
        #   1. reformulate_query rewrites the pronoun using history.
        #   2. If the output STILL contains an unresolved pronoun (helper
        #      got confused by noisy history — e.g. a prior turn cited a
        #      different entity by mistake), fall back to concatenating
        #      the last user turn with the current one. Empirically that
        #      concat anchors retrieval to the right entity without
        #      another LLM call. Example: "tenes algo mas sobre ella?"
        #      after "tenes notas sobre Grecia?" → helper may still emit
        #      "algo mas sobre ella" → concat produces "tenes notas sobre
        #      Grecia? tenes algo mas sobre ella?" which retrieves the
        #      Grecia notes at top (~+0.20 rerank score).
        yield _sse("status", {"stage": "retrieving"})

        _t_reform_start = time.perf_counter()
        # Follow-up resolution: antes llamábamos a reformulate_query (qwen2.5:3b,
        # ~1-2s de LLM call + re-eviction pressure sobre qwen2.5:7b). El comment
        # original admite que el fallback (concat "{last_user_q} {current_q}")
        # empíricamente lograba mismo rerank score con recall igual o mejor
        # (~+0.20 score al anclar con la entidad del turno previo). Saltamos
        # reform directo al concat: 2s → 0ms, y qwen2.5:3b ya no se carga en
        # el path de /chat, aliviando VRAM pressure sobre el 7b pinned.
        search_question = question
        _reform_fired = False
        _reform_used_concat = False
        if history and _looks_like_followup(question):
            _reform_fired = True
            last_user_q = next(
                (m["content"] for m in reversed(history) if m.get("role") == "user"),
                None,
            )
            if last_user_q:
                search_question = f"{last_user_q} {question}"
                _reform_used_concat = True
        _t_reform_end = time.perf_counter()

        try:
            _t_retrieve_start = time.perf_counter()
            # multi_query=False: the qwen2.5:3b paraphrase call costs 1-3s
            # and only marginally improves recall on chat-style questions.
            # k=4 + rerank_pool=5: k=1 + pool=2 starved the LLM context —
            # ambiguous queries (e.g. "dame info sobre Grecia" as proper
            # noun) leaked unrelated top-1 chunks (Contacts/_index), LLM
            # faithfully cited wrong source. Pool=5 matches bge-reranker
            # batch alignment (~210ms on MPS), k=4 gives the LLM real
            # context + room for graph expansion. Reverted if eval regresses
            # below singles hit@5 76.19 (CI lower bound) or chains
            # chain_success < 16.67.
            # Self-citation guard: drop this session's own episodic note +
            # the entire conversations/ folder from candidates. CLAUDE.md
            # marks these as system artifacts; surfacing them as retrieval
            # sources creates a feedback loop where the just-written turn
            # answers the next one (confirmed 2026-04-19).
            _own_conv = _own_conversation_path(sid)
            _exclude = {_own_conv} if _own_conv else None
            result = multi_retrieve(
                vaults, search_question, 4, None, history, None, False,
                multi_query=False, auto_filter=True, date_range=None,
                rerank_pool=5, exclude_paths=_exclude,
                exclude_path_prefixes=("00-Inbox/conversations/",),
            )
            _t_retrieve_end = time.perf_counter()
        except Exception as exc:
            yield _sse("error", {"message": f"retrieve falló: {exc}"})
            return

        if not result["docs"]:
            # Propose-intent turns don't need vault context — the tool
            # loop (propose_reminder / propose_calendar_event) creates
            # things out of thin air. Bailing here with "empty" made the
            # handler return "Sin resultados relevantes." for inputs like
            # "el 26 de Mayo es el cumple de Astor" (2026-04-21 Fer F.
            # Playwright report) which is absurd — the user is declaring
            # a date, not asking about one. Fall through to the tool
            # phase when the detector flagged intent.
            if not is_propose_intent:
                yield _sse("empty", {"message": "Sin resultados relevantes."})
                return

        # Retrieval signals — previously surfaced as a UI meta bar
        # ("🟡 media · 0.8 · N variantes · M nota(s)"). Removed from the
        # UI per user request; instead we pass these signals as context
        # to the LLM so it can frame the answer (e.g. hedge when the
        # confidence is low, or acknowledge when only one note exists).
        _conf = float(result["confidence"])
        _, _conf_label = _confidence_badge(_conf)
        _n_notes = len({m["file"] for m in result["metas"]})
        _n_variants = len(result.get("query_variants", []))
        _filters = result.get("filters_applied") or {}
        retrieval_signals = {
            "confidence_label": _conf_label,
            "confidence_score": round(_conf, 2),
            "n_notes": _n_notes,
            "n_variants": _n_variants,
            "filters": _filters,
        }

        # Mention hits (entity dictionary) → synthetic source rows so the
        # user can see which definitions were injected. `score` set to a
        # sentinel high value (5.0) and bar to all-filled — these aren't
        # ranked by the retriever, they're authoritative user-provided.
        # Uses rag._match_mentions_in_query (same resolver that feeds
        # build_person_context) so the UI reflects exactly which mention
        # notes were folded into the LLM preamble.
        from rag import _match_mentions_in_query as _match_mentions_fn
        from pathlib import Path as _Path
        _mention_paths = _match_mentions_fn(question)
        _mention_sources = [
            {
                "file": rel,
                "note": _Path(rel).stem,
                "folder": str(_Path(rel).parent),
                "score": 5.0,
                "bar": "■■■■■",
            }
            for rel in _mention_paths
        ]

        yield _sse("sources", {
            "items": (
                []
                if is_propose_intent
                else _mention_sources + [
                    {**_source_payload(m, s), "bar": _score_bar(float(s))}
                    for m, s in zip(result["metas"], result["scores"])
                ]
            ),
            "confidence": round(_sanitize_confidence(result["confidence"]), 3),
            "propose_intent": is_propose_intent,
        })

        is_multi = len(vaults) > 1
        # Cap each chunk at 500 chars on the fast path. With 4 chunks that's
        # ~2000 chars ≈ 500 tokens of variable context — small enough that
        # uncached prefill stays under ~6s on command-r:35b (12.7ms/tok),
        # large enough that REGLA 4's "obligatory usage block" still lands.
        _WEB_CHUNK_CAP = 500
        context = "\n\n---\n\n".join(
            (f"[vault: {m.get('_vault', '?')}] " if is_multi else "")
            + f"[nota: {m['note']}] [ruta: {m['file']}]\n{d[:_WEB_CHUNK_CAP]}"
            for d, m in zip(result["docs"], result["metas"])
        )

        # Collect the WA fetch we kicked off before retrieve. By now it's
        # almost always done (retrieve dominates), so .result() is a no-op.
        # Telemetry: log the post-retrieve wait, not wall time — so a 0ms
        # `wa_wait` means the SQLite read overlapped fully with retrieve.
        _t_wa_wait_start = time.perf_counter()
        try:
            wa_recent = _wa_future.result(timeout=2.0)
        except Exception:
            wa_recent = []
        finally:
            _wa_executor.shutdown(wait=False)
        _t_wa_end = time.perf_counter()
        _t_wa_wait_ms = int((_t_wa_end - _t_wa_wait_start) * 1000)
        # WA block gated on query intent — antes se inyectaba en CADA chat,
        # agregando ~500-1000ms de prefill innecesario. Ahora sólo se incluye
        # si la pregunta menciona WhatsApp explícitamente. El fetch paralelo
        # se sigue haciendo (telemetry + /api/home consumption) — sólo el
        # prompt injection es el que filtra.
        _wa_in_query = bool(
            re.search(r"\b(whatsapp|\bwa\b|mensaje|chat de|último[s]? chat)", question, re.IGNORECASE)
        )
        if wa_recent and _wa_in_query:
            total_msgs = sum(int(w.get("count", 0)) for w in wa_recent)
            wa_block_lines = [
                f"[contexto auxiliar: WhatsApp últimas 24h · "
                f"{len(wa_recent)} chats · {total_msgs} msgs]"
            ]
            for w in wa_recent[:8]:
                snip = (w.get("last_snippet") or "").strip()[:100]
                snip_part = f' — "{snip}"' if snip else ""
                wa_block_lines.append(
                    f"- {w.get('name','?')} ({w.get('count',0)} "
                    f"msgs){snip_part}"
                )
            context = context + "\n\n---\n\n" + "\n".join(wa_block_lines)

        # Retrieval signals block removido 2026-04-18: costaba ~300 chars
        # = ~75 tokens = ~150ms de prefill uncached en CADA request. El
        # hedging que calibraba (no-refusal, humildad en confianza baja,
        # engancharse con contexto débil) ya está expresado en REGLA 1 +
        # REGLA 4 del _WEB_SYSTEM_PROMPT (que está byte-identical + cached).
        # Empíricamente el modelo respeta esas reglas sin repetir las
        # señales cada turno — redundante con el system prompt.
        # retrieval_signals dict sigue disponible para telemetría
        # (_n_notes, confidence) pero no se inyecta al prompt.

        # Person-mention enrichment: when the query names someone listed in
        # 99 Mentions @/, prepend that note's body + Apple Contacts data so
        # the LLM knows who the user is talking about before reading the
        # retrieved chunks. Pre-LLM only — never leaked to episodic notes.
        # NOTE: this supersedes the earlier web-local `_mentions_block` —
        # `build_person_context` does the same job with richer output
        # (Apple Contacts metadata + structured format).
        from rag import build_person_context as _build_person_ctx
        from rag import detect_topic_shift as _detect_topic_shift
        _person_ctx = _build_person_ctx(question)
        _person_block = f"{_person_ctx}\n\n---\n\n" if _person_ctx else ""

        # Topic-shift gate: si la current question cambia de tema vs el turno
        # anterior, descartamos history para evitar contaminación cross-tópico
        # del LLM. Caso reportado 2026-04-20: "cual es mi password de avature?"
        # → "busca informacion sobre mi mama" terminaba mezclando ambos temas
        # ("no hay info sobre la contraseña de Avature de tu mamá") porque los
        # 6 últimos mensajes seguían vivos en el prompt. El gate combina
        # (a) regex anafórico protector, (b) person_context fired, (c) cosine
        # bge-m3 < 0.40 vs last user Q. Ver `detect_topic_shift` en rag.py.
        _topic_shifted = False
        _topic_shift_reason = "no-history"
        if history:
            _topic_shifted, _topic_shift_reason = _detect_topic_shift(
                question, history, person_fired=bool(_person_ctx),
            )
            if _topic_shifted:
                history = []


        # Build messages so the system prompt is BYTE-IDENTICAL across all
        # requests — that's the whole game for ollama prefix caching. The
        # context (which changes per query) goes in the user message AFTER
        # any history. With history, this puts the cacheable prefix as:
        #   [system (cached)] + [history turns (partially cached)] + user
        #
        # EXCEPTION: create-intent ("recordame X", "mañana viene Y a casa,
        # calendarizalo"). Vault retrieval is NOISE for those — the LLM
        # otherwise engancha con notas tangenciales and hallucinates
        # "open file 04-Archive/Calendar.md" instead of calling the
        # propose_* tool. Skip the CONTEXTO block entirely in that case.
        if is_propose_intent:
            user_content = f"{_person_block}{question}"
        else:
            user_content = (
                f"{_person_block}CONTEXTO:\n{context}\n\n"
                f"PREGUNTA: {question}\n\nRESPUESTA:"
            )
        # Tool-deciding messages: include _WEB_TOOL_ADDENDUM as a second
        # system message so it's byte-identical across tool-deciding calls
        # (ollama prefix cache stays warm across rounds). The final
        # streaming call below STRIPS the addendum + `tools=` param so its
        # cache state matches the pre-tool-calling era byte-for-byte.
        #
        # On create-intent turns we REPLACE the full stack with just the
        # focused override — the normal _WEB_SYSTEM_PROMPT + addendum pin
        # the model to "summarise the vault" behaviour that competes with
        # the tool call. Prefix cache is cold for that first create-intent
        # turn, acceptable cost given how rare they are compared to reads.
        #
        # History is also SKIPPED on create-intent. If previous turns in
        # the same session show the assistant responding with text (no
        # tool_calls) — which happened constantly during debug before the
        # fix — the model mirrors that pattern and refuses to emit
        # tool_calls this turn either (tool_rounds=0 logged). A create
        # intent is a self-contained action anyway: no narrative continuity
        # with prior turns, no pronouns to resolve against earlier context.
        if is_propose_intent:
            _system_msgs: list[dict] = [
                {"role": "system", "content": _PROPOSE_CREATE_OVERRIDE},
            ]
            _turn_history: list[dict] = []
        else:
            _system_msgs = [
                {"role": "system", "content": _WEB_SYSTEM_PROMPT},
                {"role": "system", "content": _WEB_TOOL_ADDENDUM},
            ]
            _turn_history = history or []
        tool_messages: list[dict] = (
            _system_msgs
            + _turn_history
            + [{"role": "user", "content": user_content}]
        )

        # Log context size for prefill analysis (pre-tool expansion).
        _ctx_chars = sum(len(m.get("content","")) for m in tool_messages)

        # Fast-path LLM options.
        # num_ctx: measured from web.log (75 timing rows, 2026-04-17):
        #   P50 ctx ≈ 4889 chars / 1.35 char-per-token ≈ 3622 tok
        #   P95 ctx ≈ 6065 chars ≈ 4492 tok
        #   P99 ctx ≈ 7751 chars ≈ 5742 tok
        # Budget breakdown per request:
        #   system prompt ~1100 tok + retrieved context ~2500–4500 tok
        #   + history (trimmed) + current query
        # 2560 was silently truncating every request above ~1900 tok of
        # retrieved context. 5120 covers P99 (5742 tok) with ~10% headroom.
        # Don't raise further without evidence — larger ctx has real prefill
        # cost on command-r:35b.
        # num_predict 256: REGLA 5 (web prompt) pide 2-4 oraciones, lo que
        # aterriza en ~80-180 tok; con la nueva REGLA 4 (preservar URLs) las
        # respuestas pueden llegar a ~200 tok. Cap previo de 60 cortaba a
        # media oración. Decode budget: qwen2.5:7b ≈ 30 tok/s → 256 tok ≈ 8s.
        # Prefix caching is automatic in ollama when system prompts match
        # byte-for-byte across requests — see _WEB_SYSTEM_PROMPT for the
        # stable definition that keeps the cache warm. cache_prompt was
        # tested as an explicit option but is not in ollama's accepted set
        # and made some calls hang silently — leave it out.
        _WEB_CHAT_OPTIONS = {
            **CHAT_OPTIONS,
            "num_ctx": _WEB_CHAT_NUM_CTX,
            "num_predict": 256,
        }

        yield _sse("status", {"stage": "generating"})

        # MPS flush sleep removed 2026-04-18: the 20-40s GPU contention
        # it defended against was actually caused by num_ctx mismatch
        # (5120 vs 4096 loaded) forcing ollama KV reinit — fixed at commit
        # 79f6b8e. Without the sleep, warm /api/chat drops by 200ms cleanly.
        # If prefill variance returns to 20s+ range, first suspect num_ctx
        # drift, not MPS contention.

        _web_model = _resolve_web_chat_model()
        print(
            f"[chat-model-keepalive] model={_web_model} keep_alive={OLLAMA_KEEP_ALIVE}"
            f" num_ctx={_WEB_CHAT_OPTIONS['num_ctx']}",
            flush=True,
        )

        # ── ollama-native tool loop ───────────────────────────────────
        # Up to 3 tool-deciding rounds against CHAT_TOOLS. Each round is a
        # non-streaming ollama.chat with tools= set. If the LLM emits
        # tool_calls we execute them (serial bucket first, then parallel
        # bucket via ThreadPoolExecutor) and append `role:"tool"` messages
        # before the next round. When tool_calls comes back empty we break
        # out and fall through to the plain streaming call below.
        #
        # tool_rounds = rounds where ≥1 tool actually executed. When the
        # LLM returns no tool_calls in round 1, tool_rounds stays 0 —
        # that's "no tools needed".
        _TOOL_ROUND_CAP = 3
        tool_rounds = 0
        tool_ms_total = 0
        tool_names_called: list[str] = []

        def _unwrap_args(args):
            """command-r wraps args as {"tool_name":..., "parameters":{...}}.
            Unwrap so we can call the Python function directly (copied from
            rag.py `rag do`)."""
            if isinstance(args, dict) and "parameters" in args and isinstance(args["parameters"], (dict, str)):
                params = args["parameters"]
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                args = params
            if not isinstance(args, dict):
                return {}
            return {k: v for k, v in args.items() if v not in ("", None)}

        try:
            from concurrent.futures import ThreadPoolExecutor as _ToolExecutor
            from concurrent.futures import as_completed as _as_completed

            # ── Pre-router: keyword-forced tools ─────────────────────────
            # Runs BEFORE the LLM tool-deciding call. Guarantees that
            # unambiguous queries (gastos → finance_summary, pendientes →
            # reminders_due, etc.) always fetch fresh data — the LLM's
            # default bias (REGLA 1 "engancháte con CONTEXTO") previously
            # swallowed these queries into stale vault summaries.
            # Results are appended as a single `role:"system"` block with
            # an explicit "DATOS FRESCOS" prefix so the LLM treats them as
            # authoritative over the retrieved vault context.
            #
            # Exception: create-intent queries ("mañana tengo daily a las
            # 10am") match `reminders_due` / `calendar_ahead` via the
            # shared `_PLANNING_PAT` (any "mañana" triggers it). Feeding
            # those read-intent results into context when the user is
            # CREATING something makes the LLM hallucinate that the event
            # already exists. Skip the pre-router entirely when propose
            # intent is detected — let the LLM decide cleanly.
            _propose_intent = is_propose_intent
            _forced_tools = [] if _propose_intent else _detect_tool_intent(question)
            if _forced_tools:
                _f_serial = [(n, a) for n, a in _forced_tools if n not in PARALLEL_SAFE]
                _f_parallel = [(n, a) for n, a in _forced_tools if n in PARALLEL_SAFE]
                # Fan-out: emit all `tool` events upfront so the frontend
                # renders chips before any blocks on execution.
                for _n, _a in _forced_tools:
                    yield _sse("status", {"stage": "tool", "name": _n, "args": _a})
                _forced_results: list[tuple[str, str, int]] = []
                _pre_serial_sum_ms = 0
                _pre_parallel_max_ms = 0
                # Serial bucket (vault-touching tools under GIL/MPS).
                for _n, _a in _f_serial:
                    _t0 = time.perf_counter()
                    try:
                        _res = TOOL_FNS[_n](**_a)
                    except Exception as _exc:
                        _res = f"Error: {_exc}"
                    _ms = int((time.perf_counter() - _t0) * 1000)
                    _pre_serial_sum_ms += _ms
                    _forced_results.append((_n, str(_res), _ms))
                    yield _sse("status", {"stage": "tool_done", "name": _n, "ms": _ms})
                    if _prop := _maybe_emit_proposal(_n, str(_res)):
                        yield _prop
                    tool_names_called.append(_n)
                # Parallel bucket (pure IO fetchers, no shared state).
                if _f_parallel:
                    def _run_tool(name: str, args: dict):
                        _t0 = time.perf_counter()
                        try:
                            _r = TOOL_FNS[name](**args)
                        except Exception as _exc:
                            _r = f"Error: {_exc}"
                        return name, str(_r), int((time.perf_counter() - _t0) * 1000)
                    with _ToolExecutor(max_workers=5) as _pool:
                        _futs = [_pool.submit(_run_tool, n, a) for n, a in _f_parallel]
                        for _fut in _as_completed(_futs):
                            _n, _res, _ms = _fut.result()
                            _pre_parallel_max_ms = max(_pre_parallel_max_ms, _ms)
                            _forced_results.append((_n, _res, _ms))
                            yield _sse("status", {"stage": "tool_done", "name": _n, "ms": _ms})
                            if _prop := _maybe_emit_proposal(_n, _res):
                                yield _prop
                            tool_names_called.append(_n)
                # Replace CONTEXTO entirely with tool output. Prior attempts
                # (append as system msg / prepend to user) failed: REGLA 1
                # in _WEB_SYSTEM_PROMPT ("engancháte SIEMPRE con el
                # CONTEXTO") pinned the LLM on vault retrieval. When the
                # pre-router fires we have fresh authoritative data — the
                # vault context is noise. Replacing the CONTEXTO block (the
                # canonical anchor for REGLA 1) makes the LLM use the tool
                # output as its sole evidence.
                _datos_block = ""
                for _n, _res, _ in _forced_results:
                    _datos_block += f"\n## {_n}\n{_res}\n"
                for _msg in reversed(tool_messages):
                    if _msg.get("role") == "user":
                        _msg["content"] = (
                            f"CONTEXTO (datos en vivo, no del vault):\n{_datos_block}\n\n"
                            f"PREGUNTA: {question}\n\nRESPUESTA:"
                        )
                        break
                tool_rounds += 1
                tool_ms_total += _pre_serial_sum_ms + _pre_parallel_max_ms

            # Gate the LLM tool-deciding loop. Empirically the LLM rarely
            # picks tools beyond what the pre-router already fired (REGLA 1
            # pinning + pre-router-injected data satisfies the query), and
            # the tool-deciding call is a non-streaming cold-prefill
            # (~10-30s on 9k-char ctx) that burns latency for near-zero
            # benefit. Skip it by default — go straight to final streaming.
            # Opt back in with RAG_WEB_TOOL_LLM_DECIDE=1 for queries where
            # chaining multiple tools based on first-round results matters.
            #
            # Exception: propose_reminder / propose_calendar_event can ONLY
            # be invoked via the LLM tool-decide loop (arg extraction is
            # LLM-hard). `_detect_propose_intent` flips the gate on for
            # create-intent queries so the user doesn't have to flip
            # RAG_WEB_TOOL_LLM_DECIDE just to record something.
            _llm_tool_decide = os.environ.get(
                "RAG_WEB_TOOL_LLM_DECIDE", ""
            ).strip() not in ("", "0", "false", "no")
            # `_propose_intent` set earlier (pre-router gate); reuse to avoid
            # re-running the regex.
            _skip_llm_tool_round = not _llm_tool_decide and not _propose_intent

            for _round_idx in range(_TOOL_ROUND_CAP):
                if _skip_llm_tool_round:
                    # Pre-router matched nothing and opt-in not set → skip
                    # the LLM decide round entirely. `break` avoids the
                    # `else:` clause firing (which would nudge the model
                    # with a "cap reached" system message).
                    break
                # On create-intent turns, narrow the tool surface to JUST
                # the 2 propose_* tools. With all 9 tools visible qwen2.5:7b
                # gets decision paralysis and often returns a clarifying
                # question instead of calling the tool. With only 2 tools
                # present and the override telling it to call one of them,
                # the choice is forced.
                _round_tools = CHAT_TOOLS
                if is_propose_intent:
                    _round_tools = [
                        fn for fn in CHAT_TOOLS if fn.__name__ in PROPOSAL_TOOL_NAMES
                    ]
                _tr = _OLLAMA_STREAM_CLIENT.chat(
                    model=_web_model,
                    messages=tool_messages,
                    tools=_round_tools,
                    options=CHAT_TOOL_OPTIONS,
                    stream=False,
                    think=False,   # see _ollama_chat_probe for rationale
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                _tmsg = _tr.message
                _tcalls = list(_tmsg.tool_calls or [])
                if not _tcalls:
                    # Terminal assistant turn — no tools. Don't append it to
                    # tool_messages; if we did, the final streaming call
                    # would see a completed answer and emit zero tokens.
                    # The final call regenerates from the clean prompt.
                    break
                # Persist the assistant turn so the LLM sees its own tool_calls
                # when we come back around (and the final streaming call too).
                tool_messages.append({
                    "role": "assistant",
                    "content": _tmsg.content or "",
                    "tool_calls": [tc.model_dump() for tc in _tcalls],
                })

                # Split into serial (not parallel-safe) and parallel buckets.
                _serial: list[tuple[str, dict]] = []
                _parallel: list[tuple[str, dict]] = []
                for _tc in _tcalls:
                    _name = _tc.function.name
                    _args = _unwrap_args(_tc.function.arguments or {})
                    if _name in PARALLEL_SAFE:
                        _parallel.append((_name, _args))
                    else:
                        _serial.append((_name, _args))

                _round_tool_names: list[str] = []
                _round_parallel_max_ms = 0
                _round_serial_sum_ms = 0

                # Serial bucket first — vault search is the gating signal.
                for _name, _args in _serial:
                    yield _sse("status", {"stage": "tool", "name": _name, "args": _args})
                    _t_tool_start = time.perf_counter()
                    _fn = TOOL_FNS.get(_name)
                    try:
                        if _fn is None:
                            _out = f"Error: tool '{_name}' no existe"
                        else:
                            _ret = _fn(**_args)
                            _out = _ret if isinstance(_ret, str) else json.dumps(_ret, ensure_ascii=False)
                    except Exception as _exc:
                        _out = f"Error: {_exc}"
                    _elapsed_ms = int((time.perf_counter() - _t_tool_start) * 1000)
                    _round_serial_sum_ms += _elapsed_ms
                    _round_tool_names.append(_name)
                    yield _sse("status", {"stage": "tool_done", "name": _name, "ms": _elapsed_ms})
                    if _prop := _maybe_emit_proposal(_name, _out):
                        yield _prop
                    tool_messages.append({
                        "role": "tool",
                        "name": _name,
                        "content": _out,
                    })

                # Parallel bucket: emit all `tool` fan-out events first, then
                # submit + wait. Per spec, tool_done events arrive as futures
                # resolve (interleaved).
                if _parallel:
                    for _name, _args in _parallel:
                        yield _sse("status", {"stage": "tool", "name": _name, "args": _args})

                    def _exec_one(name_args):
                        _name, _args = name_args
                        _t0 = time.perf_counter()
                        _fn = TOOL_FNS.get(_name)
                        try:
                            if _fn is None:
                                _out = f"Error: tool '{_name}' no existe"
                            else:
                                _ret = _fn(**_args)
                                _out = _ret if isinstance(_ret, str) else json.dumps(_ret, ensure_ascii=False)
                        except Exception as _exc:
                            _out = f"Error: {_exc}"
                        return _name, _out, int((time.perf_counter() - _t0) * 1000)

                    _ex = _ToolExecutor(max_workers=5)
                    try:
                        _futures = [_ex.submit(_exec_one, na) for na in _parallel]
                        for _fut in _as_completed(_futures):
                            _name, _out, _elapsed_ms = _fut.result()
                            if _elapsed_ms > _round_parallel_max_ms:
                                _round_parallel_max_ms = _elapsed_ms
                            _round_tool_names.append(_name)
                            yield _sse("status", {"stage": "tool_done", "name": _name, "ms": _elapsed_ms})
                            if _prop := _maybe_emit_proposal(_name, _out):
                                yield _prop
                            tool_messages.append({
                                "role": "tool",
                                "name": _name,
                                "content": _out,
                            })
                    finally:
                        _ex.shutdown(wait=False)

                if _round_tool_names:
                    tool_rounds += 1
                    tool_names_called.extend(_round_tool_names)
                    # Critical-path ms this round = serial sum + parallel max.
                    tool_ms_total += _round_serial_sum_ms + _round_parallel_max_ms
            else:
                # Round cap reached without the model emitting an empty
                # tool_calls turn — nudge it to close out with what it has.
                tool_messages.append({
                    "role": "system",
                    "content": "Alcanzado cap de herramientas; respondé con lo que tenés.",
                })
        except Exception as exc:
            yield _sse("error", {"message": f"LLM falló: {exc}"})
            return

        # Final streaming answer call: strip _WEB_TOOL_ADDENDUM and `tools=`
        # so the prefix cache for plain-text generation matches the pre
        # tool-calling era byte-for-byte. We keep the full tool loop
        # (assistant tool_calls + tool messages) so the LLM sees what it
        # learned; without `tools=` ollama won't re-invoke anything.
        final_messages = [m for m in tool_messages if not (
            m.get("role") == "system" and m.get("content") == _WEB_TOOL_ADDENDUM
        )]

        parts: list[str] = []
        stripper = _InlineCitationStripper()
        _t_llm_start = time.perf_counter()
        _first_token_logged = False
        try:
            for chunk in _OLLAMA_STREAM_CLIENT.chat(
                model=_web_model,
                messages=final_messages,
                options=_WEB_CHAT_OPTIONS,
                stream=True,
                think=False,   # the user-facing stream never includes a
                               # <think> preamble. Thinking-capable models
                               # (qwen3+, deepseek-r1, qwq) otherwise emit
                               # tokens with empty content.delta and the
                               # UI sees 0-token responses (measured on
                               # qwen3.6 2026-04-20).
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                delta = chunk.message.content or ""
                if not delta:
                    continue
                if not _first_token_logged:
                    _t_first_token = time.perf_counter()
                    _first_token_logged = True
                filtered = stripper.feed(delta)
                if filtered:
                    parts.append(filtered)
                    yield _sse("token", {"delta": filtered})
            # Flush any tail that was held back waiting for a close-paren.
            tail = stripper.flush()
            if tail:
                parts.append(tail)
                yield _sse("token", {"delta": tail})
        except Exception as exc:
            yield _sse("error", {"message": f"LLM falló: {exc}"})
            return

        full = "".join(parts)
        _t_done = time.perf_counter()

        # Timing breakdown for diagnostics
        _t_reform_ms = int((_t_reform_end - _t_reform_start) * 1000)
        _t_retrieve_ms = int((_t_retrieve_end - _t_retrieve_start) * 1000)
        _t_wa_ms = _t_wa_wait_ms
        _t_llm_prefill_ms = int((_t_first_token - _t_llm_start) * 1000) if _first_token_logged else -1
        _t_llm_decode_ms = int((_t_done - (_t_first_token if _first_token_logged else _t_llm_start)) * 1000)
        _t_total_ms = int((_t_done - _t0) * 1000)
        _t_ttft_ms = int(((_t_first_token if _first_token_logged else _t_done) - _t0) * 1000)
        _tok_count = len(full.split())
        _q_words = len(question.split())
        _reform_outcome = (
            "concat" if _reform_used_concat
            else "rewritten" if _reform_fired
            else "skipped"
        )
        _tool_names_str = ",".join(tool_names_called)
        print(
            f"[chat-timing] model={_resolve_web_chat_model()} retrieve={_t_retrieve_ms}ms "
            f"reform={_t_reform_ms}ms reform_outcome={_reform_outcome} q_words={_q_words} "
            f"wa={_t_wa_ms}ms llm_prefill={_t_llm_prefill_ms}ms "
            f"llm_decode={_t_llm_decode_ms}ms ttft={_t_ttft_ms}ms "
            f"total={_t_total_ms}ms ctx_chars={_ctx_chars} tokens≈{_tok_count} "
            # `retrieve()` devuelve float('-inf') con corpus vacío;
            # imprimirlo crudo deja `confidence=-inf` en logs que grep
            # + dashboards interpretan como "error sin sanitizar". Sanear
            # al print para que el log sea shape-stable (float finito).
            f"confidence={_sanitize_confidence(result['confidence']):.3f} "
            f"variants={len(result.get('query_variants',[]))} "
            f"tool_rounds={tool_rounds} tool_ms={tool_ms_total} tool_names={_tool_names_str} "
            f"topic_shift={_topic_shift_reason}{'!' if _topic_shifted else ''}",
            flush=True,
        )

        turn_id = new_turn_id()
        append_turn(sess, {
            "q": question,
            "a": full,
            "paths": [m.get("file", "") for m in result["metas"]],
            "top_score": round(_sanitize_confidence(result["confidence"]), 3),
            "turn_id": turn_id,
        })
        save_session(sess)
        log_query_event({
            "cmd": "web",
            "turn_id": turn_id,
            "session": sess["id"],
            "q": question,
            "paths": [m.get("file", "") for m in result["metas"]],
            "scores": [round(float(s), 2) for s in result["scores"]],
            "top_score": round(_sanitize_confidence(result["confidence"]), 2),
            "t_retrieve": round(_t_retrieve_ms / 1000.0, 3),
            "t_gen": round(max(0, _t_total_ms - _t_retrieve_ms) / 1000.0, 3),
            "topic_shifted": _topic_shifted,
            "topic_shift_reason": _topic_shift_reason,
        })

        yield _sse("done", {
            "turn_id": turn_id,
            "top_score": round(_sanitize_confidence(result["confidence"]), 3),
            "total_ms": _t_total_ms,
            "retrieve_ms": _t_retrieve_ms,
            "ttft_ms": _t_ttft_ms,
            "llm_ms": _t_llm_decode_ms,
        })

        _spawn_conversation_writer(
            target_args=(
                VAULT_PATH,
                sess["id"],
                req.question,
                full,
                result["metas"],
                result["scores"],
                result["confidence"],
            ),
            name=f"conv-writer-{turn_id[:8]}",
        )

        _enrich_evt = _emit_enrich(turn_id, question, full, float(result["confidence"]))
        if _enrich_evt:
            yield _enrich_evt

        # Cache store — sólo queries sin history (no follow-ups) y con
        # respuesta no-vacía. Keya con el vault_chunks count computado en el
        # cache-get path arriba (mismo scope/model/vault state → misma key).
        # `_cache_key is not None` cubre el caso donde el turno arrancó con
        # history (no key computado) y el topic-shift gate reasignó
        # `history = []` (línea ~3914), dejando este branch alcanzable sin
        # key válida. Sin el guard, se tiraba `UnboundLocalError` en cada
        # topic-shift turn.
        if not history and full.strip() and _cache_key is not None:
            try:
                _sources_items = [
                    {**_source_payload(m, s), "bar": _score_bar(float(s))}
                    for m, s in zip(result["metas"], result["scores"])
                ]
                _chat_cache_put(_cache_key, {
                    "text": full,
                    "sources_items": _sources_items,
                    "top_score": round(_sanitize_confidence(result["confidence"]), 3),
                })
                print(f"[chat-cache] PUT {_cache_key} len={len(full)}", flush=True)
            except Exception as _cache_err:
                print(f"[chat-cache] put failed: {_cache_err}", flush=True)

    def guarded():
        """Increment/decrement the global chat-in-flight counter around
        the real generator. The home-prewarmer loop checks this counter
        and skips cycles while > 0 so chat gets exclusive ollama time.
        Decrement runs in `finally`, which executes on normal completion
        AND when the client disconnects (generator .close()).
        """
        global _CHAT_INFLIGHT
        with _CHAT_INFLIGHT_LOCK:
            _CHAT_INFLIGHT += 1
        try:
            yield from gen()
        finally:
            with _CHAT_INFLIGHT_LOCK:
                _CHAT_INFLIGHT = max(0, _CHAT_INFLIGHT - 1)

    return StreamingResponse(guarded(), media_type="text/event-stream")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


def _collect_screentime_daily(
    st_start: datetime, st_end: datetime, st_days: int,
) -> list[dict]:
    """Daily screentime buckets via a single SQL scan of knowledgeC.db.

    Replaces N separate `_collect_screentime` calls (one per day). Groups
    /app/usage sessions by local date in SQL so we touch the db once.
    """
    import sqlite3
    from rag import SCREENTIME_DB, _SCREENTIME_COCOA_OFFSET  # noqa: PLC0415

    path = SCREENTIME_DB
    daily: list[dict] = []
    if not path.is_file():
        return daily
    start_ts = st_start.timestamp() - _SCREENTIME_COCOA_OFFSET
    end_ts = st_end.timestamp() - _SCREENTIME_COCOA_OFFSET
    try:
        uri = f"file:{path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                """
                SELECT date(ZSTARTDATE + ?, 'unixepoch', 'localtime') AS day,
                       SUM(ZENDDATE - ZSTARTDATE) AS secs
                FROM ZOBJECT
                WHERE ZSTREAMNAME = '/app/usage'
                  AND ZSTARTDATE >= ?
                  AND ZSTARTDATE < ?
                  AND (ZENDDATE - ZSTARTDATE) >= 5
                GROUP BY day
                ORDER BY day
                """,
                (_SCREENTIME_COCOA_OFFSET, start_ts, end_ts),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return daily
    by_day = {day: int(secs or 0) for day, secs in rows if day}
    for d in range(st_days):
        day_start = (st_end - timedelta(days=st_days - 1 - d)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        key = day_start.strftime("%Y-%m-%d")
        daily.append({"day": key, "secs": by_day.get(key, 0)})
    return daily


# TTL cache for /api/dashboard. Polling clients hit this every minute;
# aggregating 7 JSONL files + 8 sqlite queries per hit is wasteful when
# the underlying logs barely move between polls. SSE stream pushes live
# deltas, so a 30s cache is invisible to the user.
_DASHBOARD_CACHE: dict[int, tuple[float, dict]] = {}
_DASHBOARD_TTL = 30.0


@app.get("/api/dashboard")
def dashboard_api(days: int = 30) -> dict:
    now_ts = time.time()
    hit = _DASHBOARD_CACHE.get(days)
    if hit and now_ts - hit[0] < _DASHBOARD_TTL:
        return hit[1]
    payload = _dashboard_compute(days)
    _DASHBOARD_CACHE[days] = (now_ts, payload)
    return payload


def _dashboard_compute(days: int = 30) -> dict:
    """Aggregate telemetry into dashboard metrics.

    SQL-only since T10. On any exception, logs to sql_state_errors.jsonl and
    returns an empty-shape payload so the dashboard JS can still render
    (the aggregator handles empty lists uniformly).
    """
    try:
        return _dashboard_compute_sql(days)
    except Exception as exc:
        _log_sql_state_error("dashboard_sql_compute_failed", err=repr(exc))
        return _dashboard_aggregate(
            queries=[], all_queries=[], fb_entries=[], ambient_entries=[],
            contra_entries=[], filing_recent=[], tune_entries=[],
            surface_entries=[], days=days,
        )


# ── Dashboard event-row helpers ─────────────────────────────────────────────
# The SQL and JSONL readers converge on a common "event dict" shape (matches
# the JSONL line schema) before the aggregator runs, so only the data source
# differs. `_dashboard_aggregate()` is the single source of truth for every
# field the dashboard JS consumes.


def _dashboard_sql_row_to_event(row, json_cols: tuple[str, ...] = ()) -> dict:
    """Rehydrate a sqlite3.Row into the JSONL-style event dict.

    - `*_json` columns whose base name (`_json` suffix stripped) is listed in
      `json_cols` are decoded and re-keyed to their JSONL equivalent (e.g.
      `paths_json` → `paths`).
    - `extra_json` is decoded and its keys merged into the top level (never
      overwriting an explicit column).
    - Other `*_json` columns are decoded in place if `json_cols` doesn't map
      them — safe default so aggregate callers never see raw JSON strings.
    """
    ev: dict = {}
    try:
        keys = row.keys()
    except Exception:
        return ev
    for k in keys:
        v = row[k]
        if v is None:
            continue
        if k == "extra_json":
            try:
                extra = json.loads(v)
                if isinstance(extra, dict):
                    for ek, ev_val in extra.items():
                        ev.setdefault(ek, ev_val)
            except Exception:
                pass
            continue
        if k.endswith("_json"):
            base = k[:-5]
            try:
                decoded = json.loads(v)
            except Exception:
                decoded = None
            target = base if base in json_cols else k
            if decoded is not None:
                ev[target] = decoded
            continue
        ev[k] = v
    return ev


def _dashboard_compute_sql(days: int = 30) -> dict:
    """SQL-path reader: fetch rag_* rows since cutoff, normalise to JSONL-shape
    event dicts, then call the shared aggregator.

    Tables read: rag_queries, rag_feedback, rag_ambient, rag_contradictions,
    rag_filing_log, rag_tune, rag_surface_log. `all_queries` is the unwindowed
    snapshot of rag_queries (used only for the total_queries_all_time KPI).
    """
    from rag import _ragvec_state_conn, _sql_query_window  # local import keeps module-load light
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_iso = cutoff.isoformat(timespec="seconds")

    # Well-earlier floor — we still need "all time" count + full feedback for
    # corrective-miss/neg-reason surfacing (JSONL path uses the full file).
    ancient_iso = "0000-01-01T00:00:00"

    with _ragvec_state_conn() as conn:
        q_rows_window = _sql_query_window(conn, "rag_queries", cutoff_iso)
        # all_queries: full history — same semantics as JSONL (unbounded scan).
        q_rows_all = _sql_query_window(conn, "rag_queries", ancient_iso)
        fb_rows_all = _sql_query_window(conn, "rag_feedback", ancient_iso)
        amb_rows = _sql_query_window(conn, "rag_ambient", cutoff_iso)
        contra_rows = _sql_query_window(conn, "rag_contradictions", cutoff_iso)
        filing_rows = _sql_query_window(conn, "rag_filing_log", cutoff_iso)
        tune_rows = _sql_query_window(conn, "rag_tune", ancient_iso)
        surface_rows = _sql_query_window(conn, "rag_surface_log", ancient_iso)

    _Q_JSON = ("variants", "paths", "scores", "filters", "bad_citations")
    _F_JSON = ("paths",)
    _C_JSON = ("contradicts",)
    _FIL_JSON = ("neighbors",)
    _T_JSON = ("baseline", "best")

    all_queries = [_dashboard_sql_row_to_event(r, _Q_JSON) for r in q_rows_all]
    queries = [_dashboard_sql_row_to_event(r, _Q_JSON) for r in q_rows_window]
    fb_entries = [_dashboard_sql_row_to_event(r, _F_JSON) for r in fb_rows_all]
    # rag_ambient stores extra fields in `payload_json`, not `extra_json`.
    ambient_entries: list[dict] = []
    for r in amb_rows:
        ev = {}
        try:
            keys = r.keys()
        except Exception:
            keys = []
        for k in keys:
            v = r[k]
            if v is None:
                continue
            if k == "payload_json":
                try:
                    payload = json.loads(v)
                    if isinstance(payload, dict):
                        for pk, pv in payload.items():
                            ev.setdefault(pk, pv)
                except Exception:
                    pass
                continue
            ev[k] = v
        ambient_entries.append(ev)
    contra_entries = [_dashboard_sql_row_to_event(r, _C_JSON) for r in contra_rows]
    # Flatten `singles`/`chains` back onto each tune entry so the JSONL aggregator
    # sees `baseline`/`best` dicts as-is.
    tune_entries = [_dashboard_sql_row_to_event(r, _T_JSON) for r in tune_rows]
    # Filing rows — add an implicit `cmd` marker so the aggregator counts them
    # as filing events (it treats every entry in filing_recent as a filing).
    filing_recent = [_dashboard_sql_row_to_event(r, _FIL_JSON) for r in filing_rows]
    # Surface rows — aggregator counts events by `cmd in {"surface_pair","surface_run"}`,
    # and rag_surface_log preserves `cmd` as a column.
    surface_entries = [_dashboard_sql_row_to_event(r) for r in surface_rows]

    return _dashboard_aggregate(
        queries=queries,
        all_queries=all_queries,
        fb_entries=fb_entries,
        ambient_entries=ambient_entries,
        contra_entries=contra_entries,
        filing_recent=filing_recent,
        tune_entries=tune_entries,
        surface_entries=surface_entries,
        days=days,
    )


def _dashboard_aggregate(
    *,
    queries: list[dict],
    all_queries: list[dict],
    fb_entries: list[dict],
    ambient_entries: list[dict],
    contra_entries: list[dict],
    filing_recent: list[dict],
    tune_entries: list[dict],
    surface_entries: list[dict],
    days: int,
) -> dict:
    """Build the full dashboard payload from normalised event dicts.

    This is the shared body both the JSONL and SQL paths call with identical
    semantics. The dashboard JS key-contract (`queries_per_day`, `hours`,
    `cmds`, `sources`, `score_distribution`, `hot_topics`, `chat_keywords`,
    `feedback`, etc.) is fixed here — don't drop keys without updating
    web/static/dashboard.js.
    """
    import statistics
    from collections import defaultdict

    data_dir = Path.home() / ".local/share/obsidian-rag"
    cutoff = datetime.now() - timedelta(days=days)

    def _ts(e: dict) -> datetime | None:
        raw = e.get("ts") or e.get("timestamp")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = int(len(s) * p / 100)
        return round(s[min(idx, len(s) - 1)], 3)

    # ── Queries ──────────────────────────────────────────────────────
    scores = [float(e["top_score"]) for e in queries if isinstance(e.get("top_score"), (int, float))]
    t_retrieves = [float(e["t_retrieve"]) for e in queries if isinstance(e.get("t_retrieve"), (int, float))]
    t_gens = [float(e["t_gen"]) for e in queries if isinstance(e.get("t_gen"), (int, float)) and e["t_gen"] > 0]
    t_totals = [
        float(e["t_retrieve"]) + float(e["t_gen"])
        for e in queries
        if isinstance(e.get("t_retrieve"), (int, float)) and isinstance(e.get("t_gen"), (int, float)) and e["t_gen"] > 0
    ]

    # Per-day counts
    queries_per_day: dict[str, int] = defaultdict(int)
    latency_per_day: dict[str, list[float]] = defaultdict(list)
    for e in queries:
        t = _ts(e)
        if t:
            day = t.strftime("%Y-%m-%d")
            queries_per_day[day] += 1
            tr = e.get("t_retrieve")
            tg = e.get("t_gen")
            if isinstance(tr, (int, float)) and isinstance(tg, (int, float)) and tg > 0:
                latency_per_day[day].append(float(tr) + float(tg))

    # Hourly heatmap
    hours = defaultdict(int)
    for e in queries:
        t = _ts(e)
        if t:
            hours[t.hour] += 1

    # Command breakdown
    cmds = defaultdict(int)
    for e in queries:
        cmds[e.get("cmd", "?")] += 1

    # Source breakdown. `cmd: serve*` is exclusively the rag-serve fast path
    # (only the WhatsApp listener uses it), so any serve event is whatsapp
    # regardless of session prefix.
    sources = defaultdict(int)
    for e in queries:
        sid = e.get("session") or ""
        cmd = e.get("cmd") or ""
        if sid.startswith("wa:") or cmd.startswith("serve"):
            sources["whatsapp"] += 1
        elif sid.startswith("web:"):
            sources["web"] += 1
        elif sid.startswith("tg:"):
            sources["legacy"] += 1
        else:
            sources["cli"] += 1

    # Score distribution (10 buckets)
    score_buckets = [0] * 10
    for s in scores:
        idx = min(int(max(0.0, min(s, 1.0)) * 10), 9)
        score_buckets[idx] += 1

    # Hot topics
    topics: dict[str, int] = defaultdict(int)
    for e in queries:
        q = e.get("q", "")
        if q:
            words = q.lower().split()[:3]
            topics[" ".join(words)] += 1
    hot_topics = sorted(topics.items(), key=lambda x: -x[1])[:15]

    # Chat keyword cloud — per-word frequency over user-initiated queries
    # (chat/web/whatsapp), not ambient/eval synthetic traffic. Tokenises
    # on word boundaries, strips accents for collation (so "qué" and "que"
    # collapse), drops stopwords + words shorter than 3 chars. Returns the
    # top 60 for the UI cloud; the renderer sizes by log(count).
    _ES_EN_STOPWORDS = {
        "que", "de", "la", "el", "en", "y", "a", "los", "las", "un", "una",
        "del", "se", "por", "con", "para", "mi", "mis", "me", "tu", "tus",
        "su", "sus", "al", "lo", "le", "les", "es", "esta", "este", "esto",
        "esa", "ese", "eso", "como", "cuando", "donde", "quien", "cual",
        "cuales", "sobre", "entre", "desde", "hasta", "sin", "pero", "o",
        "u", "ni", "mas", "muy", "ya", "si", "no", "tambien", "hay", "fue",
        "son", "soy", "ser", "estoy", "estan", "estar", "he", "has", "ha",
        "hemos", "han", "puedo", "puede", "pueden", "podria", "podrian",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "and", "or", "but",
        "if", "then", "else", "when", "where", "what", "which", "who",
        "how", "why", "to", "of", "in", "on", "for", "with", "from",
        "by", "as", "at", "into", "out", "up", "down", "off", "over",
        "this", "that", "these", "those", "it", "its", "i", "you", "he",
        "she", "we", "they", "me", "my", "your", "our", "their",
        "algo", "nada", "todo", "todos", "otra", "otro", "otros", "otras",
        "solo", "sólo", "tan", "tanto", "tener", "tiene", "tienes",
        "hacer", "hace", "haces", "hice", "hicimos", "voy", "vas", "va",
        "vamos", "van", "dame", "darme", "tengo", "tenes", "tenés",
        "estas", "estos", "esas", "esos", "cuantos", "cuantas",
        "hola", "gracias", "porfa", "plis", "oka", "dale", "listo",
        "quiero", "quieres", "quisiera", "necesito", "necesitas",
        "decime", "dime", "decir", "dijo", "dije", "saber", "sabes",
    }
    _KEYWORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñü0-9][A-Za-zÁÉÍÓÚÑáéíóúñü0-9_\-']{2,}")
    def _fold(s: str) -> str:
        import unicodedata
        nfd = unicodedata.normalize("NFD", s.lower())
        return "".join(ch for ch in nfd if not unicodedata.combining(ch))
    keyword_counts: dict[str, int] = defaultdict(int)
    for e in queries:
        sid = e.get("session") or ""
        cmd = e.get("cmd") or ""
        # User-initiated surfaces only — exclude eval/ambient/morning/etc.
        if not (
            sid.startswith(("wa:", "web:", "tg:"))
            or cmd in ("query", "chat", "web", "serve", "serve-web")
        ):
            continue
        q = (e.get("q") or "").strip()
        if not q:
            continue
        for match in _KEYWORD_RE.findall(q):
            folded = _fold(match)
            if len(folded) < 3 or folded in _ES_EN_STOPWORDS:
                continue
            keyword_counts[folded] += 1
    chat_keywords = [
        {"word": w, "count": c}
        for w, c in sorted(keyword_counts.items(), key=lambda x: -x[1])[:5]
        if c >= 2  # skip hapax — noise dominates
    ]

    # Latency time series (daily p50)
    latency_ts = {}
    for day, vals in sorted(latency_per_day.items()):
        latency_ts[day] = {
            "p50": round(statistics.median(vals), 2) if vals else 0,
            "p95": _pct(vals, 95),
            "count": len(vals),
        }

    # ── Health metrics ───────────────────────────────────────────────
    # Only count queries that actually go through retrieval (have scores)
    retrieval_queries = [e for e in queries if isinstance(e.get("top_score"), (int, float)) and e.get("cmd") in ("query", "chat", "web", None)]
    n_retrieval = len(retrieval_queries)

    gated = sum(1 for e in queries if e.get("gated_low_confidence"))
    gate_rate = round(gated / n_retrieval * 100, 1) if n_retrieval else 0

    bad_citation_queries = sum(1 for e in queries if e.get("bad_citations"))
    bad_citation_rate = round(bad_citation_queries / n_retrieval * 100, 1) if n_retrieval else 0
    bad_citation_total = sum(len(e.get("bad_citations", [])) for e in queries)

    # Score quality bands
    retrieval_scores = [float(e["top_score"]) for e in retrieval_queries if isinstance(e.get("top_score"), (int, float))]
    n_scored = len(retrieval_scores)
    score_high = sum(1 for s in retrieval_scores if s >= 0.3)     # strong match
    score_mid = sum(1 for s in retrieval_scores if 0.05 <= s < 0.3)  # acceptable
    score_low = sum(1 for s in retrieval_scores if s < 0.05)      # weak/miss

    # Score trend per day (daily mean)
    score_per_day: dict[str, list[float]] = defaultdict(list)
    for e in retrieval_queries:
        t = _ts(e)
        s = e.get("top_score")
        if t and isinstance(s, (int, float)):
            score_per_day[t.strftime("%Y-%m-%d")].append(float(s))
    score_trend = {
        day: round(statistics.mean(vals), 3)
        for day, vals in sorted(score_per_day.items())
    }

    # Latency ratio (retrieve vs generate dominance)
    avg_ret = statistics.mean(t_retrieves) if t_retrieves else 0
    avg_gen = statistics.mean(t_gens) if t_gens else 0
    retrieve_pct = round(avg_ret / (avg_ret + avg_gen) * 100, 1) if (avg_ret + avg_gen) > 0 else 0

    # Answer length trend (is the LLM being verbose or terse?)
    answer_lens = [e["answer_len"] for e in queries if isinstance(e.get("answer_len"), (int, float)) and e["answer_len"] > 0]
    avg_answer_len = round(statistics.mean(answer_lens)) if answer_lens else 0

    health = {
        "retrieval_queries": n_retrieval,
        "gate_rate": gate_rate,
        "gated_count": gated,
        "bad_citation_rate": bad_citation_rate,
        "bad_citation_queries": bad_citation_queries,
        "bad_citation_total": bad_citation_total,
        "score_high": score_high,
        "score_mid": score_mid,
        "score_low": score_low,
        "score_high_pct": round(score_high / n_scored * 100, 1) if n_scored else 0,
        "score_mid_pct": round(score_mid / n_scored * 100, 1) if n_scored else 0,
        "score_low_pct": round(score_low / n_scored * 100, 1) if n_scored else 0,
        "score_trend": score_trend,
        "retrieve_pct": retrieve_pct,
        "generate_pct": round(100 - retrieve_pct, 1),
        "avg_answer_len": avg_answer_len,
    }

    # ── Feedback ─────────────────────────────────────────────────────
    # fb_entries is the full-history feedback log (SQL or JSONL). Peer
    # work on master added a windowed view for the KPI widget + all-time
    # companions — keep both so the dashboard can show "last N days" and
    # "all time" side-by-side.
    fb_recent = [e for e in fb_entries if (t := _ts(e)) and t >= cutoff]
    fb_pos = sum(1 for e in fb_recent if e.get("rating") == 1)
    fb_neg = sum(1 for e in fb_recent if e.get("rating") == -1)
    fb_pos_all = sum(1 for e in fb_entries if e.get("rating") == 1)
    fb_neg_all = sum(1 for e in fb_entries if e.get("rating") == -1)

    neg_path_counts: dict[str, int] = defaultdict(int)
    pos_path_counts: dict[str, int] = defaultdict(int)
    for e in fb_entries:
        rating = e.get("rating")
        for p in e.get("paths") or []:
            if rating == -1:
                neg_path_counts[p] += 1
            elif rating == 1:
                pos_path_counts[p] += 1

    # Corrective paths: user-supplied "the right note was here" — pure retrieval misses,
    # i.e. golden-eval candidates. Most actionable signal in the whole pipeline.
    corrective_misses: list[dict] = []
    for e in fb_entries:
        cp = e.get("corrective_path")
        if not cp:
            continue
        if cp in (e.get("paths") or []):
            continue
        corrective_misses.append({
            "ts": e.get("ts", ""),
            "q": (e.get("q") or "")[:140],
            "missing_path": cp,
            "retrieved": list((e.get("paths") or [])[:3]),
        })
    corrective_misses.sort(key=lambda x: x["ts"], reverse=True)

    # Recent negative reasons — qualitative why-it-failed signal.
    neg_reasons: list[dict] = []
    for e in fb_entries:
        if e.get("rating") != -1 or not e.get("reason"):
            continue
        neg_reasons.append({
            "ts": e.get("ts", ""),
            "q": (e.get("q") or "")[:140],
            "reason": e["reason"][:200],
            "paths": list((e.get("paths") or [])[:2]),
        })
    neg_reasons.sort(key=lambda x: x["ts"], reverse=True)

    # Calibration mismatch: cross feedback w/ queries by turn_id.
    # - false_confident = retrieved with high score but user said -1
    # - false_gated     = gate refused (low score) but user later said +1 / supplied corrective_path
    queries_by_turn = {e.get("turn_id"): e for e in queries if e.get("turn_id")}
    false_confident = 0
    false_gated = 0
    for fb in fb_entries:
        q_ev = queries_by_turn.get(fb.get("turn_id"))
        if not q_ev:
            continue
        ts = q_ev.get("top_score")
        if not isinstance(ts, (int, float)):
            continue
        if fb.get("rating") == -1 and ts >= 0.20:
            false_confident += 1
        if fb.get("rating") == 1 and ts < 0.05:
            false_gated += 1
        if fb.get("corrective_path") and ts < 0.05:
            false_gated += 1

    # Feedback per day (positive vs negative trend)
    fb_pos_per_day: dict[str, int] = defaultdict(int)
    fb_neg_per_day: dict[str, int] = defaultdict(int)
    for e in fb_recent:
        t = _ts(e)
        if not t:
            continue
        day = t.strftime("%Y-%m-%d")
        if e.get("rating") == 1:
            fb_pos_per_day[day] += 1
        elif e.get("rating") == -1:
            fb_neg_per_day[day] += 1

    feedback_actionable = {
        "total": len(fb_entries),
        "recent_pos": sum(1 for e in fb_recent if e.get("rating") == 1),
        "recent_neg": sum(1 for e in fb_recent if e.get("rating") == -1),
        "net_satisfaction": (
            round((fb_pos - fb_neg) / (fb_pos + fb_neg) * 100, 1)
            if (fb_pos + fb_neg) > 0 else None
        ),
        "top_negative_paths": [
            {"path": p, "count": c, "pos_count": pos_path_counts.get(p, 0)}
            for p, c in sorted(neg_path_counts.items(), key=lambda x: -x[1])[:5]
        ],
        "top_positive_paths": [
            {"path": p, "count": c}
            for p, c in sorted(pos_path_counts.items(), key=lambda x: -x[1])[:5]
        ],
        "corrective_misses": corrective_misses[:5],
        "n_corrective_misses": len(corrective_misses),
        "negative_reasons": neg_reasons[:5],
        "false_confident": false_confident,
        "false_gated": false_gated,
        "per_day_pos": dict(sorted(fb_pos_per_day.items())),
        "per_day_neg": dict(sorted(fb_neg_per_day.items())),
    }

    # ── Ambient ──────────────────────────────────────────────────────
    # ambient_entries is already provided by the caller; filter to window here.
    ambient_recent = [e for e in ambient_entries if (t := _ts(e)) and t >= cutoff]
    ambient_wikilinks = sum(e.get("wikilinks_applied", 0) for e in ambient_recent)
    ambient_per_day: dict[str, int] = defaultdict(int)
    for e in ambient_recent:
        t = _ts(e)
        if t:
            ambient_per_day[t.strftime("%Y-%m-%d")] += 1

    # ── Contradictions ───────────────────────────────────────────────
    contra_recent = [e for e in contra_entries if (t := _ts(e)) and t >= cutoff]
    contra_found = [e for e in contra_recent if e.get("contradicts") and not e.get("skipped")]
    contra_per_day: dict[str, int] = defaultdict(int)
    for e in contra_found:
        t = _ts(e)
        if t:
            contra_per_day[t.strftime("%Y-%m-%d")] += 1

    # ── Surface ──────────────────────────────────────────────────────
    surface_pairs = [e for e in surface_entries if e.get("cmd") == "surface_pair"]
    surface_runs = [e for e in surface_entries if e.get("cmd") == "surface_run"]

    # ── Filing ───────────────────────────────────────────────────────
    # filing_recent is already windowed by the caller.

    # ── Tune ─────────────────────────────────────────────────────────
    tune_history = []
    for e in tune_entries:
        t = _ts(e)
        tune_history.append({
            "ts": e.get("ts"),
            "samples": e.get("samples"),
            "baseline_hit": e.get("baseline", {}).get("hit"),
            "baseline_mrr": e.get("baseline", {}).get("mrr"),
            "best_hit": e.get("best", {}).get("hit"),
            "best_mrr": e.get("best", {}).get("mrr"),
            "delta": e.get("delta"),
        })

    # ── Sessions ─────────────────────────────────────────────────────
    sessions_dir = data_dir / "sessions"
    session_count = 0
    session_turns_total = 0
    wa_sessions = 0
    if sessions_dir.is_dir():
        for sf in sessions_dir.glob("*.json"):
            try:
                sess = json.loads(sf.read_text(encoding="utf-8"))
                session_count += 1
                turns = sess.get("turns", [])
                session_turns_total += len(turns)
                if sess.get("id", "").startswith("wa:"):
                    wa_sessions += 1
            except Exception:
                continue

    # ── Screen Time (macOS knowledgeC.db) ────────────────────────────
    # Cap at 7 days — dashboard day-pickers of 30/90 don't help: knowledge
    # store only keeps ~7-10d of /app/usage before CoreDuet aggregates it
    # away. Querying a wider window returns gaps, not more data.
    st_days = min(max(int(days), 1), 7)
    st_end = datetime.now()
    st_start = st_end - timedelta(days=st_days)
    screentime_agg = _collect_screentime(st_start, st_end)
    screentime_daily = _collect_screentime_daily(st_start, st_end, st_days)
    screentime_total_str = (
        _fmt_hm(int(screentime_agg.get("total_secs") or 0))
        if screentime_agg.get("available") else ""
    )

    # ── Index stats ──────────────────────────────────────────────────
    index_stats = {}
    try:
        col = get_db()
        n_chunks = col.count()
        corpus = _load_corpus(col)
        pr = get_pagerank(col)
        top_pr = sorted(pr.items(), key=lambda x: -x[1])[:5] if pr else []
        index_stats = {
            "chunks": n_chunks,
            "notes_files": len(corpus.get("outlinks", {})),
            "notes_titles": len(corpus.get("title_to_paths", {})),
            "notes": len(corpus.get("outlinks", {})),
            "tags": len(corpus.get("tags", set())),
            "folders": len(corpus.get("folders", set())),
            "top_pagerank": [{"path": p, "score": round(s, 4)} for p, s in top_pr],
        }
    except Exception:
        pass

    return {
        "period_days": days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kpis": {
            "total_queries": len(queries),
            "total_queries_all_time": len(all_queries),
            "avg_latency": round(statistics.mean(t_totals), 2) if t_totals else None,
            "avg_retrieve": round(statistics.mean(t_retrieves), 2) if t_retrieves else None,
            "avg_generate": round(statistics.mean(t_gens), 2) if t_gens else None,
            "feedback_positive": fb_pos,
            "feedback_negative": fb_neg,
            "feedback_positive_all_time": fb_pos_all,
            "feedback_negative_all_time": fb_neg_all,
            "sessions": session_count,
            "wa_sessions": wa_sessions,
            "ambient_hooks": len(ambient_recent),
            "ambient_wikilinks": ambient_wikilinks,
            "contradictions_found": len(contra_found),
            "surface_pairs": len(surface_pairs),
            "filings": len(filing_recent),
        },
        "health": health,
        "index": index_stats,
        "queries_per_day": dict(sorted(queries_per_day.items())),
        "latency_per_day": latency_ts,
        "hours": {str(h): hours.get(h, 0) for h in range(24)},
        "cmds": dict(sorted(cmds.items(), key=lambda x: -x[1])),
        "sources": dict(sources),
        "score_distribution": score_buckets,
        "score_stats": {
            "p50": _pct(scores, 50),
            "p95": _pct(scores, 95),
            "min": round(min(scores), 3) if scores else 0,
            "max": round(max(scores), 3) if scores else 0,
        },
        "latency_stats": {
            "retrieve_p50": _pct(t_retrieves, 50),
            "retrieve_p95": _pct(t_retrieves, 95),
            "generate_p50": _pct(t_gens, 50),
            "generate_p95": _pct(t_gens, 95),
            "total_p50": _pct(t_totals, 50),
            "total_p95": _pct(t_totals, 95),
        },
        "hot_topics": [{"topic": t, "count": c} for t, c in hot_topics],
        "chat_keywords": chat_keywords,
        "ambient_per_day": dict(sorted(ambient_per_day.items())),
        "contradictions_per_day": dict(sorted(contra_per_day.items())),
        "tune_history": tune_history,
        "surface_runs": len(surface_runs),
        "filing_confidence": [round(e.get("confidence", 0), 2) for e in filing_recent],
        "feedback": feedback_actionable,
        "screentime": {
            "available": screentime_agg.get("available", False),
            "window_days": st_days,
            "total_secs": int(screentime_agg.get("total_secs") or 0),
            "total_label": screentime_total_str,
            "top_apps": screentime_agg.get("top_apps") or [],
            "categories": screentime_agg.get("categories") or {},
            "daily": screentime_daily,
        },
    }


# ── Real-time stream ─────────────────────────────────────────────────────────
# Poll rag_* SQL tables for new rows and push them as SSE events. The browser
# polls /api/dashboard for the heavy aggregations; this endpoint streams
# deltas so KPIs and the live feed update without waiting for the next poll.
#
# Post-T10 note: writers no longer touch the `queries.jsonl` / `behavior.jsonl`
# / `ambient.jsonl` / `contradictions.jsonl` files, so the previous tail-based
# implementation emitted zero deltas in production. SQL ID-poll is the only
# viable source now. Each kind maps to a table + a row→event mapper that
# rebuilds the JSONL-equivalent shape `_stream_payload` expects.

# Rows returned per poll per table. Caps runaway growth if a long burst
# happens between polls; anything over this is silently skipped — the
# next poll picks up the newest rows (the UI feed is a live tail, not an
# audit log, so losing intermediate events during a burst is acceptable).
_STREAM_ROW_CAP = 100


def _row_to_query_ev(row: dict) -> dict:
    """Rebuild the JSONL-shaped event `_stream_payload('query', ...)` expects
    from a rag_queries row. `extra_json` carries legacy keys like `gated_
    low_confidence` and `error` — unpack when present, ignore otherwise.
    """
    extra: dict = {}
    raw_extra = row.get("extra_json")
    if raw_extra:
        try:
            extra = json.loads(raw_extra) or {}
        except Exception:
            extra = {}
    paths: list = []
    raw_paths = row.get("paths_json")
    if raw_paths:
        try:
            paths = json.loads(raw_paths) or []
        except Exception:
            paths = []
    bad: list = []
    raw_bad = row.get("bad_citations_json")
    if raw_bad:
        try:
            bad = json.loads(raw_bad) or []
        except Exception:
            bad = []
    return {
        "ts": row.get("ts"),
        "q": row.get("q"),
        "cmd": row.get("cmd"),
        "session": row.get("session"),
        "top_score": row.get("top_score"),
        "t_retrieve": row.get("t_retrieve"),
        "t_gen": row.get("t_gen"),
        "paths": paths,
        "bad_citations": bad,
        "gated_low_confidence": extra.get("gated_low_confidence"),
        "error": extra.get("error"),
    }


def _row_to_feedback_ev(row: dict) -> dict:
    """rag_feedback row → event shape `_stream_payload('feedback', ...)`."""
    extra: dict = {}
    raw_extra = row.get("extra_json")
    if raw_extra:
        try:
            extra = json.loads(raw_extra) or {}
        except Exception:
            extra = {}
    paths: list = []
    raw_paths = row.get("paths_json")
    if raw_paths:
        try:
            paths = json.loads(raw_paths) or []
        except Exception:
            paths = []
    return {
        "ts": row.get("ts"),
        "rating": row.get("rating"),
        "q": row.get("q"),
        "paths": paths,
        "reason": extra.get("reason"),
        "corrective_path": extra.get("corrective_path"),
    }


def _row_to_ambient_ev(row: dict) -> dict:
    """rag_ambient row → event shape `_stream_payload('ambient', ...)`.
    `wikilinks_applied` lives in `payload_json`; absent → 0."""
    payload: dict = {}
    raw_payload = row.get("payload_json")
    if raw_payload:
        try:
            payload = json.loads(raw_payload) or {}
        except Exception:
            payload = {}
    return {
        "ts": row.get("ts"),
        "path": row.get("path"),
        "wikilinks_applied": payload.get("wikilinks_applied", 0),
    }


def _row_to_contradiction_ev(row: dict) -> dict:
    """rag_contradictions row → event shape `_stream_payload('contradiction',
    ...)`. The dashboard expects `path` (subject) and `contradicts` (list)."""
    contradicts: list = []
    raw = row.get("contradicts_json")
    if raw:
        try:
            contradicts = json.loads(raw) or []
        except Exception:
            contradicts = []
    skipped_val = row.get("skipped")
    return {
        "ts": row.get("ts"),
        "path": row.get("subject_path"),
        "contradicts": contradicts,
        "skipped": bool(skipped_val) and skipped_val not in ("", "0", "false", "no"),
    }


# (table, mapper) per stream kind. Mappers rebuild the JSONL-shaped dict that
# `_stream_payload(kind, ev)` consumes — keeps the downstream shaping logic
# (which the dashboard JS already depends on) byte-identical.
_STREAM_SOURCES: dict[str, tuple[str, Callable[[dict], dict]]] = {
    "query":         ("rag_queries",        _row_to_query_ev),
    "feedback":      ("rag_feedback",       _row_to_feedback_ev),
    "ambient":       ("rag_ambient",        _row_to_ambient_ev),
    "contradiction": ("rag_contradictions", _row_to_contradiction_ev),
}


def _stream_max_id(conn, table: str) -> int:
    """Starting cursor for a stream — connection-time high-water mark so the
    client only sees rows written AFTER it connects. Missing table or no rows
    → 0."""
    try:
        row = conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _stream_fetch_since(conn, table: str, last_id: int) -> list[dict]:
    """Return up to `_STREAM_ROW_CAP` new rows with `id > last_id` ordered by
    id ascending. Empty list on any error (connection churn, transient lock)
    — the next poll retries."""
    try:
        cur = conn.execute(
            f"SELECT * FROM {table} WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, _STREAM_ROW_CAP),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        return []


@app.get("/api/dashboard/stream")
async def dashboard_stream() -> StreamingResponse:
    """SSE: poll rag_* SQL tables for new rows and push them as events.

    Each kind tracks its own `last_id` cursor, initialised at connection
    time from `MAX(id)` so the client only receives rows inserted during
    the connection. Per poll, one SQL connection is opened, all 4 tables
    are scanned, then the connection is closed — keeps total connection
    time short (WAL readers don't block writers but we prefer no-op
    polls to not hold the conn).
    """
    # Cursors per kind — populated lazily on first poll so connection-setup
    # errors don't kill the SSE stream (the dashboard uses this endpoint as
    # a heartbeat in addition to a data feed).
    cursors: dict[str, int | None] = {kind: None for kind in _STREAM_SOURCES}

    async def gen():
        last_heartbeat = time.time()
        # Initial hello so the client can flip the indicator immediately.
        yield _sse("hello", {"t": time.time(), "tracking": list(_STREAM_SOURCES)})
        try:
            while True:
                try:
                    with _ragvec_state_conn() as conn:
                        for kind, (table, mapper) in _STREAM_SOURCES.items():
                            if cursors[kind] is None:
                                cursors[kind] = _stream_max_id(conn, table)
                                continue
                            rows = _stream_fetch_since(conn, table, cursors[kind])
                            if not rows:
                                continue
                            cursors[kind] = max(int(r.get("id") or 0) for r in rows)
                            for r in rows:
                                ev = mapper(r)
                                yield _sse(kind, _stream_payload(kind, ev))
                except Exception:
                    # Transient SQL error — skip this cycle, keep the SSE
                    # connection up so the dashboard doesn't reconnect.
                    pass
                now = time.time()
                if now - last_heartbeat >= 15:
                    yield _sse("heartbeat", {"t": now})
                    last_heartbeat = now
                await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_payload(kind: str, ev: dict) -> dict:
    """Trim raw log entries to the fields the dashboard needs."""
    if kind == "query":
        sid = ev.get("session") or ""
        # Whatsapp listener uses session ids like `wa:<jid>` OR custom strings
        # like `<runtime>:<vault>` (rag serve from whatsapp-listener) — treat
        # `serve` cmd as whatsapp by default since that's the only consumer.
        cmd = ev.get("cmd") or ""
        if sid.startswith("wa:") or cmd.startswith("serve"):
            source = "whatsapp"
        elif sid.startswith("web:"):
            source = "web"
        elif sid.startswith("tg:"):
            source = "legacy"
        else:
            source = "cli"
        tr = ev.get("t_retrieve")
        tg = ev.get("t_gen")
        latency = (
            round(float(tr) + float(tg), 2)
            if isinstance(tr, (int, float)) and isinstance(tg, (int, float))
            else None
        )
        # phase: received → in-flight, error → failed, anything else → completed
        if cmd == "serve.received":
            phase = "in_flight"
        elif cmd == "serve.error":
            phase = "error"
        else:
            phase = "done"
        return {
            "ts": ev.get("ts"),
            "q": (ev.get("q") or "")[:160],
            "score": ev.get("top_score"),
            "latency": latency,
            "source": source,
            "cmd": cmd,
            "phase": phase,
            "error": (ev.get("error") or "")[:200] if ev.get("error") else None,
            "gated": bool(ev.get("gated_low_confidence")),
            "bad_citations": len(ev.get("bad_citations") or []),
            "n_paths": len(ev.get("paths") or []),
        }
    if kind == "feedback":
        return {
            "ts": ev.get("ts"),
            "rating": ev.get("rating"),
            "q": (ev.get("q") or "")[:160],
            "reason": (ev.get("reason") or "")[:200],
            "corrective_path": ev.get("corrective_path"),
            "n_paths": len(ev.get("paths") or []),
        }
    if kind == "ambient":
        return {
            "ts": ev.get("ts"),
            "path": ev.get("path"),
            "wikilinks_applied": ev.get("wikilinks_applied", 0),
        }
    if kind == "contradiction":
        return {
            "ts": ev.get("ts"),
            "path": ev.get("path"),
            "contradicts": ev.get("contradicts"),
            "skipped": bool(ev.get("skipped")),
        }
    return ev


# ── RAG-system memory sampler ──────────────────────────────────────────
# Tracks RSS of processes that are part of the rag stack we built:
#   rag        — obsidian-rag python (watch, morning, today, digest, web,
#                mcp, chat, query, launchd-spawned jobs, their children)
#   ollama     — ollama serve + per-model runners (LLM + embeddings)
#   sqlite-vec — sqlite-vec-gui streamlit inspector pointed at ragvec.db
#   whatsapp   — whatsapp-bridge, whatsapp-listener, whatsapp-mcp, vault-sync
#
# System-wide processes (browser, Claude Code, unrelated python) are
# intentionally excluded — this chart answers "how much RAM does OUR
# stack use?", not "what's eating the machine?".
#
# Zero-dep: parses `ps -axo rss=,command=` on the full command line so
# we can match interpreter path + script path (short `comm=` collapses
# everything to `python3`). RSS on macOS double-counts shared memory
# across helper processes, so absolute numbers overstate real usage;
# trends are what the chart is for.
from collections import deque

_MEMORY_STATE_PATH = Path.home() / ".local/share/obsidian-rag/rag_memory.jsonl"
_MEMORY_BUFFER_MAX = 1440  # 24h @ 60s
_MEMORY_SAMPLE_INTERVAL = 60.0
_MEMORY_BUFFER: deque = deque(maxlen=_MEMORY_BUFFER_MAX)
_MEMORY_LOCK = threading.Lock()

_MEMORY_CATEGORIES = ("rag", "ollama", "sqlite-vec", "whatsapp")

# Order matters: first regex to match claims the process. `whatsapp`
# goes before `rag` because whatsapp-mcp lives under a path that also
# matches obsidian-rag venvs in edge cases; `ollama` before `rag`
# likewise for safety.
_RAG_PROC_MATCHERS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("whatsapp",   re.compile(r"whatsapp-(bridge|listener|vault-sync|mcp)")),
    ("ollama",     re.compile(r"(?:^|/)ollama(?:\s|$)")),
    ("sqlite-vec", re.compile(r"sqlite-vec-gui\b")),
    ("rag",        re.compile(r"obsidian-rag")),
)


def _classify_rag_proc(cmd: str) -> str | None:
    for cat, rx in _RAG_PROC_MATCHERS:
        if rx.search(cmd):
            return cat
    return None


def _rag_proc_label(cmd: str, cat: str) -> str:
    """Friendly, aggregatable label for the `top` list."""
    if cat == "rag":
        if "obsidian-rag-mcp" in cmd:
            return "obsidian-rag-mcp"
        if "web/server.py" in cmd:
            return "rag web"
        m = re.search(r"/rag\s+(\w[\w-]*)", cmd)
        if m:
            return f"rag {m.group(1)}"
        if "resource_tracker" in cmd:
            return "rag (resource_tracker)"
        return "rag (python)"
    if cat == "ollama":
        if "ollama runner" in cmd:
            m = re.search(r"sha256-([0-9a-f]{8,})", cmd)
            return f"ollama runner ({m.group(1)[:8]})" if m else "ollama runner"
        if "ollama serve" in cmd:
            return "ollama serve"
        return "ollama"
    if cat == "sqlite-vec":
        return "sqlite-vec-gui"
    if cat == "whatsapp":
        for tag in ("whatsapp-bridge", "whatsapp-listener", "whatsapp-vault-sync", "whatsapp-mcp"):
            if tag in cmd:
                return tag
        return "whatsapp"
    return cmd.split()[0].rsplit("/", 1)[-1]


def _read_vm_stat() -> dict:
    """Parse `vm_stat` to surface pressure + free pages."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2, check=False,
        ).stdout
    except Exception:
        return {}
    page_size = 4096
    stats: dict[str, float] = {}
    for line in out.splitlines():
        if "page size of" in line:
            m = re.search(r"page size of (\d+)", line)
            if m:
                page_size = int(m.group(1))
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip().rstrip(".")
        try:
            stats[key.strip()] = float(val)
        except ValueError:
            continue
    mb = lambda pages: round(pages * page_size / (1024 * 1024), 1)
    return {
        "free_mb": mb(stats.get("Pages free", 0)),
        "active_mb": mb(stats.get("Pages active", 0)),
        "inactive_mb": mb(stats.get("Pages inactive", 0)),
        "wired_mb": mb(stats.get("Pages wired down", 0)),
        "compressed_mb": mb(stats.get("Pages occupied by compressor", 0)),
    }


def _sample_memory() -> dict | None:
    """One sample: RSS totals per rag-stack category + top processes + vm_stat."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "rss=,command="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return None

    by_cat: dict[str, float] = {k: 0.0 for k in _MEMORY_CATEGORIES}
    procs: dict[str, dict] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rss_kb = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        cat = _classify_rag_proc(cmd)
        if cat is None:
            continue
        rss_mb = rss_kb / 1024.0
        by_cat[cat] += rss_mb
        label = _rag_proc_label(cmd, cat)
        slot = procs.get(label)
        if slot is None:
            procs[label] = {"name": label, "mb": rss_mb, "cat": cat}
        else:
            slot["mb"] += rss_mb

    total = round(sum(by_cat.values()), 1)
    by_cat_rounded = {k: round(v, 1) for k, v in by_cat.items()}
    top = sorted(procs.values(), key=lambda d: d["mb"], reverse=True)[:10]
    top_list = [{"name": d["name"], "mb": round(d["mb"], 1), "cat": d["cat"]} for d in top]

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total_mb": total,
        "by_category": by_cat_rounded,
        "top": top_list,
        "vm": _read_vm_stat(),
    }


def _memory_load_history() -> None:
    """Warm the ring buffer from the JSONL sidecar (last N lines)."""
    if not _MEMORY_STATE_PATH.exists():
        return
    try:
        with _MEMORY_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-_MEMORY_BUFFER_MAX:]
    except Exception:
        return
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            _MEMORY_BUFFER.append(json.loads(raw))
        except Exception:
            continue


def _persist_with_sqlite_retry(write_fn, error_tag: str) -> None:
    """Run a one-shot SQL-write closure with transient-lock retry.

    `_memory_persist` + `_cpu_persist` (and historically anything else
    firing from the per-minute samplers) were dropping ~30 samples/day
    to `sql_state_errors.jsonl` with `database is locked`, because the
    bare `_ragvec_state_conn() + _sql_append_event` path has no retry —
    if any other writer holds the WAL write-lock longer than the
    `busy_timeout=30s` window (e.g. during `_write_feedback_golden_sql`
    under embed latency, or a concurrent contradiction worker), the
    sample is lost. Retrying twice with jittered backoff covers the
    transient-contention tail without masking real SQL errors (schema
    drift, disk-full, etc. propagate on the 3rd attempt).
    """
    import random as _r
    import sqlite3 as _sqlite3
    import time as _t
    for attempt in range(3):
        try:
            write_fn()
            return
        except _sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 2:
                _log_sql_state_error(error_tag, err=repr(exc))
                return
            _t.sleep(0.1 + _r.random() * 0.25)
        except Exception as exc:
            _log_sql_state_error(error_tag, err=repr(exc))
            return


def _memory_persist(sample: dict) -> None:
    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_memory_metrics",
                               _map_memory_row(sample))
    _persist_with_sqlite_retry(_do, "memory_sql_write_failed")


def _memory_trim_file() -> None:
    """Keep the JSONL file bounded at ~2x buffer so it doesn't grow forever."""
    try:
        if not _MEMORY_STATE_PATH.exists():
            return
        with _MEMORY_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        cap = _MEMORY_BUFFER_MAX * 2
        if len(lines) <= cap:
            return
        tail = lines[-cap:]
        with _MEMORY_STATE_PATH.open("w", encoding="utf-8") as fh:
            fh.writelines(tail)
    except Exception:
        pass


@app.on_event("startup")
def _start_memory_sampler() -> None:
    _memory_load_history()

    def loop() -> None:
        trim_counter = 0
        while True:
            sample = _sample_memory()
            if sample:
                with _MEMORY_LOCK:
                    _MEMORY_BUFFER.append(sample)
                _memory_persist(sample)
                trim_counter += 1
                if trim_counter >= 60:  # once an hour
                    _memory_trim_file()
                    trim_counter = 0
            time.sleep(_MEMORY_SAMPLE_INTERVAL)

    threading.Thread(target=loop, name="memory-sampler", daemon=True).start()


@app.get("/api/system-memory")
def system_memory_api(minutes: int = 360) -> dict:
    """Return the last `minutes` of samples (default 6h) + a fresh current.

    Categories: python, browser, ollama, node, claude, other (all MB).
    Used for initial backfill; the live stream is `/api/system-memory/stream`.
    """
    minutes = max(5, min(minutes, 1440))
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _MEMORY_LOCK:
        all_samples = list(_MEMORY_BUFFER)

    samples = []
    for s in all_samples:
        ts = s.get("ts") or ""
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                samples.append(s)
        except Exception:
            continue

    current = _sample_memory()
    return {
        "categories": list(_MEMORY_CATEGORIES),
        "samples": samples,
        "current": current,
        "interval_s": _MEMORY_SAMPLE_INTERVAL,
        "live_interval_s": _MEMORY_LIVE_INTERVAL,
        "window_minutes": minutes,
    }


# Live SSE stream. Emits a fresh sample every `_MEMORY_LIVE_INTERVAL` seconds
# so the chart updates in real time instead of waiting for the 60s persistent
# tick. The persistent 60s sampler (for the JSONL sidecar + historical
# buffer) runs independently — live ticks are NOT persisted to avoid
# bloating the JSONL at 2s cadence.
_MEMORY_LIVE_INTERVAL = 2.0


@app.get("/api/system-memory/stream")
async def system_memory_stream() -> StreamingResponse:
    async def gen():
        # Prime with one sample immediately so the client doesn't wait.
        first = await asyncio.to_thread(_sample_memory)
        if first:
            yield _sse("sample", first)
        while True:
            await asyncio.sleep(_MEMORY_LIVE_INTERVAL)
            sample = await asyncio.to_thread(_sample_memory)
            if not sample:
                continue
            yield _sse("sample", sample)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── RAG-stack CPU sampler ──────────────────────────────────────────────
# Same scope as the memory sampler (rag, ollama, sqlite-vec, whatsapp).
# macOS `ps -o %cpu` is a decaying average, which smears transient load
# — instead we compute instantaneous CPU% from deltas of cumulative CPU
# time across two snapshots (`ps -axo pid,cputime,command`). Each caller
# (live SSE stream, persistent sampler, etc.) owns its own prev-state
# dict so concurrent streams don't fight over the baseline.
#
# Returned percentages are in "% of one core" (can exceed 100 on
# multi-threaded processes — ollama runners routinely peg several
# cores during generation).
_CPU_STATE_PATH = Path.home() / ".local/share/obsidian-rag/rag_cpu.jsonl"
_CPU_BUFFER_MAX = 1440  # 24h @ 60s
_CPU_SAMPLE_INTERVAL = 60.0
_CPU_LIVE_INTERVAL = 2.0
_CPU_BUFFER: deque = deque(maxlen=_CPU_BUFFER_MAX)
_CPU_LOCK = threading.Lock()
_CPU_NCORES = os.cpu_count() or 1


def _parse_cputime(s: str) -> float | None:
    """Parse macOS ps cputime — `M:SS.FF` or `H:MM:SS.FF` — into seconds."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        pass
    return None


def _sample_cpu(prev_state: dict) -> dict | None:
    """Instantaneous CPU% per rag-stack category via cputime delta.

    `prev_state` holds the previous snapshot across calls:
        {"ts": monotonic_seconds, "per_pid": {pid: cputime_s}}
    First invocation only primes the state and returns None.
    """
    now_mono = time.monotonic()
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,cputime=,command="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return None

    per_pid_cpu: dict[int, float] = {}
    per_pid_cmd: dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cpu_s = _parse_cputime(parts[1])
        if cpu_s is None:
            continue
        per_pid_cpu[pid] = cpu_s
        per_pid_cmd[pid] = parts[2]

    prev_per_pid = prev_state.get("per_pid") or {}
    prev_ts = prev_state.get("ts")
    prev_state["ts"] = now_mono
    prev_state["per_pid"] = per_pid_cpu

    if prev_ts is None:
        return None
    interval = now_mono - prev_ts
    if interval <= 0:
        return None

    by_cat: dict[str, float] = {k: 0.0 for k in _MEMORY_CATEGORIES}
    procs: dict[str, dict] = {}
    for pid, cpu_s in per_pid_cpu.items():
        prev_cpu = prev_per_pid.get(pid)
        if prev_cpu is None:
            continue
        delta = cpu_s - prev_cpu
        if delta < 0:
            continue  # clock anomaly or pid reuse — skip
        pct = (delta / interval) * 100.0
        if pct < 0.05:
            continue
        cmd = per_pid_cmd[pid]
        cat = _classify_rag_proc(cmd)
        if cat is None:
            continue
        by_cat[cat] += pct
        label = _rag_proc_label(cmd, cat)
        slot = procs.get(label)
        if slot is None:
            procs[label] = {"name": label, "pct": pct, "cat": cat}
        else:
            slot["pct"] += pct

    total = round(sum(by_cat.values()), 1)
    by_cat_rounded = {k: round(v, 1) for k, v in by_cat.items()}
    top = sorted(procs.values(), key=lambda d: d["pct"], reverse=True)[:10]
    top_list = [{"name": d["name"], "pct": round(d["pct"], 1), "cat": d["cat"]} for d in top]

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total_pct": total,
        "by_category": by_cat_rounded,
        "top": top_list,
        "ncores": _CPU_NCORES,
        "interval_s": round(interval, 2),
    }


def _cpu_load_history() -> None:
    if not _CPU_STATE_PATH.exists():
        return
    try:
        with _CPU_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-_CPU_BUFFER_MAX:]
    except Exception:
        return
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            _CPU_BUFFER.append(json.loads(raw))
        except Exception:
            continue


def _cpu_persist(sample: dict) -> None:
    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_cpu_metrics",
                               _map_cpu_row(sample))
    _persist_with_sqlite_retry(_do, "cpu_sql_write_failed")


def _cpu_trim_file() -> None:
    try:
        if not _CPU_STATE_PATH.exists():
            return
        with _CPU_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        cap = _CPU_BUFFER_MAX * 2
        if len(lines) <= cap:
            return
        with _CPU_STATE_PATH.open("w", encoding="utf-8") as fh:
            fh.writelines(lines[-cap:])
    except Exception:
        pass


@app.on_event("startup")
def _start_cpu_sampler() -> None:
    _cpu_load_history()

    def loop() -> None:
        state: dict = {}
        # Prime the baseline, then wait a full interval before the first
        # real sample so the delta is meaningful.
        _sample_cpu(state)
        trim_counter = 0
        while True:
            time.sleep(_CPU_SAMPLE_INTERVAL)
            sample = _sample_cpu(state)
            if sample:
                with _CPU_LOCK:
                    _CPU_BUFFER.append(sample)
                _cpu_persist(sample)
                trim_counter += 1
                if trim_counter >= 60:
                    _cpu_trim_file()
                    trim_counter = 0

    threading.Thread(target=loop, name="cpu-sampler", daemon=True).start()


@app.get("/api/system-cpu")
def system_cpu_api(minutes: int = 360) -> dict:
    """Last `minutes` of rag-stack CPU samples + a freshly computed current.

    Values are `% of one core` per category. Can exceed 100 per category
    (sum across multithreaded procs). `ncores` is included so the client
    can render a secondary scale as % of total CPU if desired.
    """
    minutes = max(5, min(minutes, 1440))
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _CPU_LOCK:
        all_samples = list(_CPU_BUFFER)

    samples = []
    for s in all_samples:
        ts = s.get("ts") or ""
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                samples.append(s)
        except Exception:
            continue

    # Snapshot current via a short-gap sample (sleep in a worker thread
    # would block this sync handler; just use a small synchronous window).
    probe: dict = {}
    _sample_cpu(probe)
    time.sleep(_CPU_LIVE_INTERVAL)
    current = _sample_cpu(probe)

    return {
        "categories": list(_MEMORY_CATEGORIES),
        "samples": samples,
        "current": current,
        "interval_s": _CPU_SAMPLE_INTERVAL,
        "live_interval_s": _CPU_LIVE_INTERVAL,
        "window_minutes": minutes,
        "ncores": _CPU_NCORES,
    }


@app.get("/api/system-cpu/stream")
async def system_cpu_stream() -> StreamingResponse:
    async def gen():
        state: dict = {}
        # Prime baseline, wait one interval, then stream.
        await asyncio.to_thread(_sample_cpu, state)
        while True:
            await asyncio.sleep(_CPU_LIVE_INTERVAL)
            sample = await asyncio.to_thread(_sample_cpu, state)
            if not sample:
                continue
            yield _sse("sample", sample)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
