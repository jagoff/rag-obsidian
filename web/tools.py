"""Chat-scoped tool registry for `POST /api/chat` ollama-native tool-calling.

Exports `CHAT_TOOLS` (ordered list of callables), `TOOL_FNS` (name→callable
dispatch map), `PARALLEL_SAFE` (names safe to run concurrently in a thread
pool), `PROPOSAL_TOOL_NAMES` (names that emit SSE `proposal` events instead
of plain tool output), `CHAT_TOOL_OPTIONS` (ollama options for
tool-deciding call), and `_WEB_TOOL_ADDENDUM` (constant system-prompt
suffix — kept byte-identical for ollama prefix caching).

Glue only — all real logic lives in `rag.py` / `web.server`. Each tool has a
Google-style docstring; ollama derives the JSON schema from signature +
docstring at call time.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import (  # noqa: E402
    _agent_tool_drive_search,
    _agent_tool_read_note,
    _agent_tool_search,
    _agent_tool_weather,
    _agent_tool_whatsapp_search,
    _fetch_gmail_evidence,
    _fetch_reminders_due,
)


_WEB_TOOL_ADDENDUM: str = """Tenés 14 tools para traer datos frescos o registrar acciones. IMPORTANTE: usalas cuando la pregunta las necesita, aunque el CONTEXTO del vault ya tenga algo — el vault puede estar desactualizado o incompleto.

Routing por palabra clave (si aparece → llamá la tool):
- gasto/gasté/gastos/presupuesto/plata/finanza/MOZE → finance_summary
- pendiente/tarea/recordatorio/to-do/checklist → reminders_due
- mail/correo/email/gmail/inbox → gmail_recent
- evento/agenda/calendario/cita/reunión/mañana/próxima semana → calendar_ahead
- clima/tiempo/lluvia/temperatura/pronóstico → weather
- google drive/drive/planilla/spreadsheet/sheet/doc/documento/presentación → drive_search(query='<keywords extraídos>'). Extraé los tokens útiles (nombres propios, sustantivos concretos) y descartá "busca", "decime", "en mi", "drive" — ej. "busca en mi drive qué me adeuda Alexis de la macbook pro" → drive_search(query='alexis macbook pro adeuda').
- whatsapp/wzp/wsp + "chat pendiente"/"respuesta pendiente"/"qué tengo pendiente" → whatsapp_pending. Devuelve la LISTA de chats donde el user debe responder (último inbound sin reply). Sólo lista chats; NO busca en contenido. Usalo también en queries "qué tengo pendiente esta semana/hoy" — los chats pendientes cuentan como tarea pendiente semántica.
- "qué me dijo X / qué me mandó X / cuándo hablamos de Y / el chat donde X mencionó Z / dónde charlamos sobre W" → whatsapp_search(query='<tema o palabras clave>', contact='<nombre opcional>'). Busca DENTRO del contenido de los mensajes WhatsApp (4500+ chunks indexados). Si el user nombra un contacto explícito ("qué me dijo Juan sobre la deuda") pasá `contact='Juan'` para filtrar; si la pregunta es genérica ("dónde charlamos sobre la mudanza") dejá `contact=None`. Es DISTINTO a `whatsapp_pending` — éste busca por CONTENIDO, no lista chats abiertos.
- para profundizar en una nota específica → read_note(path)
- si ninguna aplica y necesitás más contexto del vault → search_vault

Crear cosas nuevas (se agregan automáticamente, el usuario puede deshacer):
- "recordame X" / "acordate X" / "ponete un recordatorio" → propose_reminder(title, when, ...)
- "creá/agendá/bloqueá un evento/reunión/turno" → propose_calendar_event(title, start, ...)
- STATEMENT form implícito: "mañana tengo una daily a las 10am", "el jueves hay standup", "me citaron para entrevista el viernes 3pm" → ESTO TAMBIÉN es create intent. Llamá propose_calendar_event directamente. NO llames calendar_ahead/reminders_due en estos casos — el usuario está AGREGANDO algo, no consultando.
- **CUMPLES Y ANIVERSARIOS** (`cumple`, `cumpleaños`, `aniversario` + fecha): SIEMPRE usá `propose_calendar_event` (no `propose_reminder`) y pasá `recurrence_text="yearly"` para que el evento se repita todos los años. Ej: "el 26 de mayo es el cumple de Astor" → `propose_calendar_event(title="cumpleaños de Astor", start="26 de mayo", all_day=true, recurrence_text="yearly")`. Lo mismo para "aniversario" (casamiento, trabajo, etc.). Sin el `recurrence_text="yearly"` el evento queda de un solo año, inútil para cumples.
- Si la fecha/hora parsea clara, el tool CREA de una (el usuario ve un toast con Deshacer por 10s). Si es ambigua, el tool devuelve una propuesta y el usuario aclara desde una tarjeta.
- No vuelvas a llamar el tool si ya lo hiciste esta ronda.
- En tu respuesta textual después de llamar el tool: SÉ CONCISO (1-2 oraciones). Decí algo tipo "Listo, quedó agendado" o "Ahí te lo sumo, avisame si hay que cambiar algo". No repitas todos los campos — el usuario ya los ve en el toast / tarjeta.

Enviar WhatsApp a terceros (acción destructiva — SIEMPRE pide confirmación):
- "enviale / mandale un mensaje a <Contacto> que diga: <texto>" / "decile a <Contacto>: <texto>" / "escribile a <Contacto>: <texto>" → propose_whatsapp_send(contact_name="<Contacto>", message_text="<texto literal>").
- "respondele a <Contacto> al mensaje del <hint>: <texto>" / "contestale a <Contacto>: <texto>" / "responde el último de <Contacto>: <texto>" → propose_whatsapp_reply(contact_name="<Contacto>", message_text="<texto literal>", when_hint="<hint opcional>"). Es DISTINTO al send — acá el user quiere responder a UN mensaje específico que recibió, no iniciar un thread. Pasale el hint que el user dijo ("el del almuerzo", "el último", "el de las 14:30", "el de ayer") en `when_hint`. Si no especificó hint, dejá `when_hint=None` (agarra el más reciente). El tool resuelve el message_id automáticamente — el user no tiene que copiar ningún ID.
- Ambos tools NO envían: devuelven una proposal card con [Enviar] / [Editar] / [Cancelar]. El user confirma explícitamente con un click. NUNCA prometas que "ya lo mandé" — hasta que el user toque Enviar no sale nada.
- Si el contacto no se resuelve (el tool devuelve `error: not_found` o `no_phone`), avisale al user que no encontraste a la persona en sus Contactos y sugerile que pase el nombre completo o el teléfono.
- En tu respuesta textual: "Te dejo el mensaje armado para <Nombre>, revisalo y tocá Enviar si está ok." (1-2 oraciones, que quede claro que NO se envió todavía). NO repitas el texto del mensaje — la card ya lo muestra.

Regla de citas (CRÍTICA): cita SOLO paths reales del vault devueltos por search_vault/read_note (ej. `[Algo](02-Areas/X/Algo.md)`). NUNCA cites identificadores internos ni nombres de tools: **PROHIBIDO** `[calendar_ahead](...)`, `[reminders_due](...)`, `[gmail_recent](...)`, `[finance_summary](...)`, `[weather](...)`, `[propose_reminder](...)`, `[propose_calendar_event](...)`, thread_id, event_id, proposal_id, ni nada con `.md` que no haya vuelto literalmente de search_vault. Los datos de tools externas (gmail/finance/calendar/reminders/weather) van en PROSA, sin markdown links.

Paralelismo: podés llamar varias tools en el mismo turno si son independientes. Máximo 3 rondas.
"""


CHAT_TOOL_OPTIONS: dict = {
    "num_ctx": 4096,
    "num_predict": 512,
    "temperature": 0.0,
    "seed": 42,
}


def search_vault(query: str, k: int = 5) -> str:
    """Buscar chunks relevantes en el vault (semantic + BM25 + rerank).

    Args:
        query: Pregunta o tema a buscar, en lenguaje natural.
        k: Cantidad de chunks a devolver (1–10, default 5).

    Returns:
        JSON con lista de {note, path, score, content} por score descendente.
    """
    k = max(1, min(10, int(k)))
    return _agent_tool_search(query, k)


def read_note(path: str) -> str:
    """Leer contenido completo de una nota del vault. path debe terminar en .md.

    Args:
        path: Ruta relativa al vault (ej. "02-Areas/Coaching/Ikigai.md").

    Returns:
        Markdown completo de la nota, o mensaje de error.
    """
    return _agent_tool_read_note(path)


def reminders_due(days_ahead: int = 7) -> str:
    """Apple Reminders con fecha pendiente o sin fecha, próximos N días.

    Args:
        days_ahead: Horizonte en días (1–30, default 7).

    Returns:
        JSON `{dated: [...], undated: [...]}`. Errores → lista vacía + "error".
    """
    try:
        horizon = max(1, min(30, int(days_ahead)))
        items = _fetch_reminders_due(datetime.now(), horizon_days=horizon, max_items=40)
        dated = [r for r in items if r.get("bucket") != "undated"]
        undated = [r for r in items if r.get("bucket") == "undated"]
        return json.dumps({"dated": dated, "undated": undated}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"dated": [], "undated": [], "error": str(exc)}, ensure_ascii=False)


def gmail_recent() -> str:
    """Gmail reciente: awaiting-reply, starred, y últimos del inbox (≤12).

    Orden de prioridad en el JSON que sirve al LLM:
      1. `awaiting_reply` — hilos donde el user le debe respuesta a
         alguien (más actionable).
      2. `starred` — hilos flagueados manualmente por el user.
      3. `recent` — últimos N del inbox sin filtros de status. Tapamos el
         gap de iter 5 (user report 2026-04-24): con starred+awaiting
         vacíos, el tool devolvía `_Sin mails pendientes._` aunque el
         inbox tuviera mails perfectamente navegables. "últimos mails"
         para el user = "los más recientes", no "los flagueados".

    Cap a 12 (3*4) en total porque son 3 buckets y queremos margen pero
    sin inundar el CONTEXTO. Dedup implícito: `_fetch_gmail_evidence`
    elimina del bucket `recent` los thread_ids que ya aparecen en
    starred/awaiting, así que iterar los 3 no repite.

    Returns:
        JSON `{unread_count: int, threads: [...]}`. Error → ambos vacíos.
    """
    try:
        now = datetime.now()
        ev = _fetch_gmail_evidence(now) or {}
        unread_count = int(ev.get("unread_count") or 0)
        threads: list[dict] = []

        def _mk_thread(kind: str, item: dict) -> dict:
            ms = int(item.get("internal_date_ms") or 0)
            received_at = (
                datetime.fromtimestamp(ms / 1000.0).isoformat(timespec="minutes")
                if ms else ""
            )
            return {
                "kind": kind,
                "from": item.get("from", ""),
                "subject": item.get("subject", ""),
                "snippet": item.get("snippet", ""),
                "days_old": item.get("days_old"),
                "thread_id": item.get("thread_id", ""),
                "received_at": received_at,
            }

        awaiting = list(ev.get("awaiting_reply") or [])
        starred = list(ev.get("starred") or [])
        recent = list(ev.get("recent") or [])

        CAP = 12
        for item in awaiting:
            if len(threads) >= CAP:
                break
            threads.append(_mk_thread("awaiting_reply", item))
        for item in starred:
            if len(threads) >= CAP:
                break
            threads.append(_mk_thread("starred", item))
        for item in recent:
            if len(threads) >= CAP:
                break
            threads.append(_mk_thread("recent", item))

        return json.dumps({"unread_count": unread_count, "threads": threads}, ensure_ascii=False)
    except Exception:
        return json.dumps({"unread_count": 0, "threads": []}, ensure_ascii=False)


def finance_summary(month: str | None = None) -> str:
    """Resumen de gastos del mes (YYYY-MM) o mes actual desde MOZE.

    Args:
        month: Mes objetivo en formato `YYYY-MM`. Default: mes actual.

    Returns:
        JSON con el summary, o `"{}"` si no hay datos / error.
    """
    try:
        from web.server import _fetch_finance  # lazy: web.server importa este módulo.

        import re as _re
        anchor = datetime.now()
        if month:
            m = _re.match(r"^(\d{4})-(\d{2})$", month.strip())
            if m:
                try:
                    anchor = datetime(int(m.group(1)), int(m.group(2)), 1)
                except ValueError:
                    anchor = datetime.now()
        result = _fetch_finance(anchor)
        return json.dumps(result or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def calendar_ahead(days: int = 3) -> str:
    """Próximos eventos de calendario en los próximos N días.

    Args:
        days: Horizonte en días (1–14, default 3).

    Returns:
        JSON con lista de eventos; `"[]"` si no hay datos / error.
    """
    try:
        from web.server import _fetch_calendar_ahead  # lazy to avoid cycle.
        n = max(1, min(14, int(days)))
        result = _fetch_calendar_ahead(n, max_events=40)
        return json.dumps(result or [], ensure_ascii=False)
    except Exception:
        return "[]"


def weather(location: str | None = None) -> str:
    """Pronóstico: hoy + 2 días.

    Args:
        location: Ciudad a consultar. Default: ubicación configurada.

    Returns:
        JSON con condición actual + 3 días, o mensaje de error.
    """
    return _agent_tool_weather(location)


def drive_search(query: str, max_files: int = 5) -> str:
    """Buscar archivos en Google Drive por contenido y devolver body exportado.

    Usalo cuando el user pide explícitamente buscar en Drive, o cuando la
    pregunta alude a una planilla / doc / presentación que NO está en el
    CONTEXTO del vault (el snapshot diario sólo trae 4 docs recientes).

    Args:
        query: Keywords separadas por espacio (ej. "alexis macbook pro").
            Extraé los tokens útiles del pedido — descartá artículos,
            "busca", "decime", "google drive", etc. Tokens vacíos → error.
        max_files: Cantidad de archivos a devolver con body (1–8, default 5).

    Returns:
        JSON `{tokens, query_used, files: [{name, mime_label, modified,
        link, body}]}`. Body viene exportado a text/csv/plain (capado a
        3500 chars). Error / sin auth → `{files: [], error: "..."}`.
    """
    return _agent_tool_drive_search(query, max_files=max_files)


def whatsapp_search(query: str, contact: str | None = None, k: int = 5) -> str:
    """Buscar en mis conversaciones de WhatsApp por contenido (semantic + BM25 + rerank).

    Distinto a `whatsapp_pending` (que sólo lista chats sin respuesta):
    este tool busca DENTRO del contenido de los mensajes indexados
    (corpus WA local, ~4500 chunks). Usalo cuando el user pregunta
    "qué me dijo X sobre Y", "cuándo hablamos de Z con M", "el chat
    donde N mencionó algo".

    Args:
        query: Tema/palabras clave a buscar en lenguaje natural
            (ej. "deuda macbook", "turno con el doctor", "mudanza").
        contact: Nombre del contacto a filtrar (resolved via Apple
            Contacts → JID). Si no se resuelve, busca sin filtro y
            agrega un `warning` al output.
        k: Cantidad máxima de mensajes a devolver (1–8, default 5).

    Returns:
        JSON `{query, contact_filter, messages: [{jid, contact, ts,
        who: "inbound"|"outbound", text, score}], warning?, error?}`.
        Snippets capados a 400 chars. Si retrieve raisea o el corpus
        está vacío → `{messages: [], error: "..."}`.
    """
    k = max(1, min(8, int(k)))
    return _agent_tool_whatsapp_search(query, contact=contact, k=k)


def whatsapp_pending(hours: int = 48, max_chats: int = 10) -> str:
    """WhatsApp chats esperando respuesta del user — último inbound sin reply.

    A diferencia de "mensajes sin leer" (que cuenta inbound recientes
    aunque ya estén respondidos), esto devuelve sólo chats donde VOS
    debés el próximo mensaje. Un chat con 20 inbound respondidos NO
    aparece; uno con 1 inbound ignorado SÍ. Ventana configurable —
    mensajes más viejos que `hours` se consideran abandonados.

    Args:
        hours: Cuán atrás mirar para decidir "stale". Default 48h
            (2 días; fuera de eso dudoso que sea realmente pendiente).
        max_chats: Máximo de chats a devolver (default 10).

    Returns:
        JSON lista `[{jid, name, last_snippet, hours_waiting}, ...]`
        por orden descendente de timestamp del último mensaje. Error
        o DB no disponible → `"[]"`.
    """
    try:
        from web.server import _fetch_whatsapp_unreplied  # lazy: circular
        h = max(1, min(168, int(hours)))
        n = max(1, min(20, int(max_chats)))
        result = _fetch_whatsapp_unreplied(hours=h, max_chats=n)
        return json.dumps(result or [], ensure_ascii=False)
    except Exception:
        return "[]"


# Chat-exposed tool wrappers for reminder/event creation. Real logic
# lives in rag.py so the CLI rag chat loop can reuse it without the
# web → rag → web circular import. These re-exports keep the ollama
# tool schema extraction pointed at web/tools symbols as before.
from rag import (  # noqa: E402
    propose_reminder,
    propose_calendar_event,
    propose_whatsapp_send,
    propose_whatsapp_reply,
)


CHAT_TOOLS: list[Callable] = [
    search_vault,
    read_note,
    reminders_due,
    gmail_recent,
    finance_summary,
    calendar_ahead,
    weather,
    drive_search,
    whatsapp_pending,
    whatsapp_search,
    propose_reminder,
    propose_calendar_event,
    propose_whatsapp_send,
    propose_whatsapp_reply,
]

TOOL_FNS: dict[str, Callable] = {fn.__name__: fn for fn in CHAT_TOOLS}

PARALLEL_SAFE: set[str] = {
    "weather",
    "finance_summary",
    "calendar_ahead",
    "reminders_due",
    "gmail_recent",
    "drive_search",
    "whatsapp_pending",
    "whatsapp_search",
    "propose_reminder",
    "propose_calendar_event",
    # `propose_whatsapp_send` intencionalmente NO está acá: aunque el tool
    # NO envía (solo drafts), el contact-lookup via osascript es side-
    # effectful (2-3s de latencia) y ejecutarlo en paralelo con otros
    # tools complica el debugging si hay un hang del bridge. Además la
    # semántica de "enviar mensaje" debería correrse aislada — el user
    # espera UN draft por turno, no varios en paralelo.
}

# Tool names whose return value (JSON string) should ALSO be emitted as a
# `proposal` SSE event by the web server — on top of the normal tool-output
# routing. Lets the UI render a confirmation card inline while the LLM's
# final narrative streams normally.
PROPOSAL_TOOL_NAMES: set[str] = {
    "propose_reminder",
    "propose_calendar_event",
    "propose_whatsapp_send",
    "propose_whatsapp_reply",
}
