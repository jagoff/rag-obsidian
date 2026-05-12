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
    _agent_tool_record_contact_observation,
    _agent_tool_search,
    _agent_tool_weather,
    _agent_tool_whatsapp_search,
    _agent_tool_whatsapp_thread,
    _fetch_gmail_evidence,
    _fetch_reminders_due,
)


_WEB_TOOL_ADDENDUM: str = """Tenés 26 tools para traer datos frescos o registrar acciones. IMPORTANTE: usalas cuando la pregunta las necesita, aunque el CONTEXTO del vault ya tenga algo — el vault puede estar desactualizado o incompleto.

Routing por palabra clave (si aparece → llamá la tool):
- gasto/gasté/gastos/presupuesto/plata/finanza/MOZE → finance_summary
- tarjeta/Visa/Mastercard/Amex/saldo a pagar/cierre/vencimiento del resumen → credit_cards_summary
- pendiente/tarea/recordatorio/to-do/checklist → reminders_due
- mail/correo/email/gmail/inbox → gmail_recent
- evento/agenda/calendario/cita/reunión/mañana/próxima semana → calendar_ahead
- clima/tiempo/lluvia/temperatura/pronóstico → weather. **Si el user menciona una ciudad** ("clima en X", "pronóstico en X"), pasá `location='X'`. Sin location devuelve el default (Santa Fe). Ej: "cómo está el clima hoy en Buenos Aires" → weather(location='Buenos Aires').
- google drive/drive/planilla/spreadsheet/sheet/doc/documento/presentación → drive_search(query='<keywords extraídos>'). Extraé los tokens útiles (nombres propios, sustantivos concretos) y descartá "busca", "decime", "en mi", "drive" — ej. "busca en mi drive qué me adeuda Alexis de la macbook pro" → drive_search(query='alexis macbook pro adeuda').
- whatsapp/wzp/wsp + "chat pendiente"/"respuesta pendiente"/"qué tengo pendiente" → whatsapp_pending. Devuelve la LISTA de chats donde el user debe responder (último inbound sin reply). Sólo lista chats; NO busca en contenido. Usalo también en queries "qué tengo pendiente esta semana/hoy" — los chats pendientes cuentan como tarea pendiente semántica.
- "qué me dijo X / qué me mandó X / cuándo hablamos de Y / el chat donde X mencionó Z / dónde charlamos sobre W" → whatsapp_search(query='<tema o palabras clave>', contact='<nombre opcional>'). Busca DENTRO del contenido de los mensajes WhatsApp (4500+ chunks indexados). Si el user nombra un contacto explícito ("qué me dijo Juan sobre la deuda") pasá `contact='Juan'` para filtrar; si la pregunta es genérica ("dónde charlamos sobre la mudanza") dejá `contact=None`. Es DISTINTO a `whatsapp_pending` — éste busca por CONTENIDO, no lista chats abiertos.
- "qué hablamos con X / qué quedamos con X / leéme el chat con X / qué dijimos con X / fijate qué charlamos con X" → whatsapp_thread(contact_name='X', max_messages=15, days=7). Trae los últimos N mensajes LITERALES del chat 1:1 con X, en orden cronológico, leídos directo del bridge SQLite (no del corpus indexado). Distinto a `whatsapp_search` (que matchea por contenido) y `whatsapp_pending` (que lista chats sin responder): acá queremos el hilo tal cual para que el LLM parsee contexto (horas propuestas, decisiones, promesas, confirmaciones). Si después de leer el thread ves una fecha/hora clara propuesta para juntarse ("el jueves a las 4pm", "mañana 10hs"), encadená con `propose_calendar_event(title='Reunión con X', start='<fecha parseada>')` en la MISMA ronda — NO pidas confirmación intermedia al user; el propose_calendar_event ya muestra la card con [Crear].
- "leé/abrí/mostrame/lee la nota <X>" (donde X es nombre o path) → read_note(path='<path>'). Resolvé el path: si el user dice solo el nombre sin ".md", agregalo (ej. "leé CLAUDE" → path='CLAUDE.md'). Esta tool SIEMPRE tiene prioridad cuando el user pide explícitamente leer una nota — NO uses search_vault como fallback.
- para profundizar en una nota específica → read_note(path)
- "cuántas notas tengo / qué tan grande es mi vault / stats del rag / qué tengo indexado / cómo está mi vault" → rag_stats(). Devuelve total de chunks, URLs, breakdown por source (vault/whatsapp/gmail/calendar/etc), modelos activos, feedback counts. Resumí en prosa concisa.
- **NOTA NUEVA EN EL VAULT** ("creá/creame una nota", "anotá/anotame", "guardá/guardame esto en obsidian", "captura/capturame", "agregá una nota") **SIN mencionar contacto/persona ni URL** → `create_note(text='<el texto que el user quiere guardar>', title='<si dijo titulada X>', tags=None)`. Esto NO es WhatsApp ni reminder ni event. Escribe directo a `00-Inbox/` del vault. Confirmá con 1 línea: "Guardado en 00-Inbox/<filename>". **DISAMBIGUACIÓN CRÍTICA**: si el user dice "guardame una nota" / "creá una nota" / "anotame" SIN "a <Persona>" → es `create_note`, NO `propose_whatsapp_send_note`. `propose_whatsapp_send_note` SOLO va cuando hay un destinatario explícito ("mandale a Juan la nota de X"). Sin "a <Contacto>" jamás dispares whatsapp_send_note.
- **URL → NOTA EN EL VAULT** ("creá nota sobre este link <url>", "leéme esta URL", "ingerí <url>", "guardá esta página <url>", o un URL crudo sin verbo) → `fetch_url(url='<url>', save=true)`. Fetchea + summariza + escribe a `00-Inbox/YYYY-MM-DD-HHMM-read-<slug>.md` con frontmatter type:read + tags auto + wikilinks. Confirmá 1 línea: "Guardé '<title>' como 00-Inbox/<filename>". Si el user dice solo "qué dice esta URL" SIN pedir guardar, pasá `save=false`.
- si ninguna aplica y necesitás más contexto del vault → search_vault

**ANTI-PATRÓN CRÍTICO (lookups del vault)**: si la pregunta es de tipo lookup ("cuál es mi X", "dame los datos de Y", "qué es Z", "decime W", "buscá X", "información sobre Y", "password de X", "CBU de Y", "dirección de Z", "teléfono de W"), defaulteala a `search_vault(query=<términos clave>)`. PROHIBIDO llamar `whatsapp_pending` / `reminders_due` / `calendar_ahead` / `gmail_recent` para estas queries — esos tools sirven solo cuando el user pregunta literalmente por "chats pendientes" / "tareas" / "eventos" / "mails". Una query como "cuál es mi iCloud Recovery Password" o "cuál es la password de la wifi" tiene que disparar `search_vault(query='iCloud Recovery Password')` (o el equivalente), NUNCA `whatsapp_pending` ni `reminders_due`. Ante duda entre tool freshness vs lookup → ELEGÍ search_vault, es la fuente canónica para info personal del vault.

Capturar info sobre contactos (PERSISTENTE, escribe al vault sin pedir permiso):
- Cuando el user dice algo relevante sobre una PERSONA (preferencia, update de trabajo/familia, evento importante, tema sensible, cumpleaños, etc.), llamá `record_contact_observation(contact_name='<nombre>', observation='<info corta procesada>', category='<bullet>', source_excerpt='<frase cruda>')`. NO pidas permiso — es una nota interna, el user la puede editar/borrar en Obsidian.
- Triggers típicos: "Seba me trajo un vino" → `record_contact_observation(contact_name='Seba', observation='Le gusta el vino', category='Preferencias', source_excerpt='Seba me trajo un vino')`. "Mi vieja empezó a trabajar en San Pedro" → `(contact_name='Mama', observation='Trabaja en San Pedro', category='Trabajo / contexto', source_excerpt='...')`. "Oscar está sensible con la herencia" → `(contact_name='Oscar', observation='Sensible con tema herencia', category='Notas', ...)`.
- Categorías sugeridas (bullets del template de `99-Contacts/`): "Preferencias", "Trabajo / contexto", "Familia", "Notas", "Eventos importantes". Si no estás seguro, dejá `category=None` (va solo a la sección `## Observaciones` como auditoría).
- Si el contacto no existe en el vault, el tool devuelve `reason: "contact_not_in_vault"`. Avisale al user: "No tengo nota de <X> en el vault — copiá `99-Contacts/_template.md` y completala con su nombre, teléfono, etc., y lo anoto la próxima".
- Confirmá con 1 línea después de la llamada: "Anotado en la nota de <nombre>". NO repitas el texto (el user ya sabe lo que dijo) — solo confirmá que quedó registrado. Si pusiste category, podés agregar "→ **<category>**".
- Podés llamar este tool en PARALELO con otros (search_vault, propose_calendar_event) si el turno tiene múltiples intents — ej. "hoy Astor cumple 3 → observation 'cumple 26 abril' + propose_calendar_event yearly".
- NO usar para queries/lookups ("qué le gusta a X" → usá search_vault / read_note para leer la nota del contacto), ni cuando el user dice explícitamente "no anotes esto" / "no te acuerdes".

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
- **`message_text` es LITERAL hasta el FINAL del mensaje del user**. Cuando el user dice "diciendo: X" / "que diga: X" / "con el texto: X" / ": X", todo lo que viene DESPUÉS de los dos puntos es el cuerpo, hasta el final del prompt — incluyendo signos de exclamación, palabras tipo "programado"/"urgente", emojis, todo. NO interpretes "mensaje programado" / "mensaje urgente" / "mensaje de aviso" como meta-descripciones del envío y las saques del cuerpo: son texto literal que el user quiere mandar. Si dudás, mantené todo lo que viene después de los dos puntos.
- "respondele a <Contacto> al mensaje del <hint>: <texto>" / "contestale a <Contacto>: <texto>" / "responde el último de <Contacto>: <texto>" → propose_whatsapp_reply(contact_name="<Contacto>", message_text="<texto literal>", when_hint="<hint opcional>"). Es DISTINTO al send — acá el user quiere responder a UN mensaje específico que recibió, no iniciar un thread. Pasale el hint que el user dijo ("el del almuerzo", "el último", "el de las 14:30", "el de ayer") en `when_hint`. Si no especificó hint, dejá `when_hint=None` (agarra el más reciente). El tool resuelve el message_id automáticamente — el user no tiene que copiar ningún ID.
- "mandale/pasale/compartile a <Contacto> la (nota|receta|info|pasos|lista|guía) de <Tema>" / "mandale a <Contacto> lo que tengo sobre <Tema>" → propose_whatsapp_send_note(contact_name="<Contacto>", note_query="<Tema>"). DISTINTO a propose_whatsapp_send (texto literal): este busca la nota en el vault, la convierte a formato WhatsApp y arma el draft con el contenido. Si el user dice "la sección X de la nota Y", pasá section="X". Si el user pasa un path explícito (".md"), usalo literal en `note_query`. Si la query es ambigua (low confidence), el tool devuelve `candidates` para que el user elija.
- "mandale/pasale a <Contacto> el (tel|teléfono|email|contacto|dirección) de <Persona/Lugar>" → propose_whatsapp_send_contact_card(recipient_contact="<Contacto>", target_query="<Persona/Lugar>"). Resuelve los datos primero en Apple Contacts y después en notas del vault con frontmatter `phone`/`email`/`address`. Pasá `fields=["phone"]` (o "email"/"address") si el user pidió solo un campo.
- Todos NO envían: devuelven una proposal card con [Enviar] / [Editar] / [Cancelar]. El user confirma explícitamente con un click. NUNCA prometas que "ya lo mandé" — hasta que el user toque Enviar no sale nada.
- Si el contacto no se resuelve (el tool devuelve `error: not_found` o `no_phone`), avisale al user que no encontraste a la persona en sus Contactos y sugerile que pase el nombre completo o el teléfono.
- **GRUPOS de WhatsApp**: el resolver acepta nombres de grupos (ej. "Familia", "RagNet", "Cloud Services"). Si el user dice "mandale al grupo X" o el nombre podría ser ambiguo entre contacto/grupo, **pasá `contact_name="grupo X"`** (con prefijo "grupo " o "group ") para forzar el lookup de grupo y evitar que se confunda con un homónimo en Contactos. Si el user dice solo el nombre (ej. "mandale a Familia"), el resolver intenta Contactos primero y si falla cae automáticamente a grupos. Si el resultado vuelve con `error="ambiguous"` + `candidates`, decile al user los nombres exactos que encontraste y pedí desambiguación. Reply (propose_whatsapp_reply) en grupos NO está soportado — si el user pide "respondele al grupo X", usá propose_whatsapp_send normal (sin quote nativo).
- **PROGRAMAR ENVÍO** (campo opcional `scheduled_for`): si el user mencionó EXPLÍCITAMENTE una fecha/hora futura para mandar el mensaje, pasá `scheduled_for="<ISO8601 con offset -03:00>"` (ej. `"2026-04-26T09:00:00-03:00"`). Patterns que SIEMPRE disparan scheduled_for:
  - "**a las HH:MM**" / "**a las HHhs**" → hoy a esa hora si está en el futuro, mañana si ya pasó. Ej: "mandale a mama a las 11:55" → si ahora son las 10:00 → `scheduled_for="<hoy>T11:55:00-03:00"`. **Este caso se olvida fácil — si ves "a las HH:MM" o variantes, NO ENTRES en el flow de envío inmediato; pasá `scheduled_for` siempre.**
  - "mañana <hora>" / "el <día> <hora>" / "el <fecha> a las <hora>" → resolver la fecha y pasar ISO. Ej: "decile a Oscar el viernes 14:30" → `scheduled_for="<viernes>T14:30:00-03:00"`.
  - "en <N> minutos/horas/días" → now + delta. Ej: "decile a Sole en 2 horas" → `scheduled_for="<now+2h>T...:00-03:00"`.
  - "esta tarde" / "esta noche" / "mañana temprano" → tomalos como hints horarios estándar (16:00 / 20:00 / 08:00) y pasá scheduled_for.

  Resolvé fechas relativas usando la fecha de hoy del contexto del prompt. Si el user NO mencionó NINGUNA fecha/hora futura, OMITÍ el arg — la card sale con [Enviar] inmediato (NO pongas null, NO inventes fecha). Ambiguo de verdad ("mandale mañana eventualmente") → omitir, que decida el humano. Aplica a `propose_whatsapp_send`, `propose_whatsapp_reply`, `propose_whatsapp_send_note`, `propose_whatsapp_send_contact_card`.

  **EJEMPLO CANÓNICO completo** (combinando body literal + scheduled_for): user dice "mandale a mi mamá a las 11:55 diciendo: Hola! mensaje programado" → `propose_whatsapp_send(contact_name="mama", message_text="Hola! mensaje programado", scheduled_for="<hoy>T11:55:00-03:00")`. NUNCA truncar el body a "Hola!" pensando que "mensaje programado" es meta-descripción — es texto literal del cuerpo. NUNCA omitir scheduled_for pensando que "a las 11:55" es información de contexto — es horario explícito.

Gestionar mensajes WhatsApp YA programados (cancelar, reagendar, listar):
- "qué mensajes tengo programados / qué wsps quedan por mandar / lista los mensajes pendientes" → whatsapp_list_scheduled(). Es un QUERY tool (NO emite card) — devuelve JSON con `{items: [{id, contact_name, scheduled_for_local, message_text_preview, ...}]}`. Resumí el listado en prosa concisa: "Tenés 3 mensajes programados: 1) a Grecia mañana 9:00 'feliz cumple', 2) a Oscar el viernes 14:30 ..., 3) ...". Si está vacío decí "No tenés mensajes programados". Por default trae solo `pending`; si el user pide "todos los que mandé este mes" pasá `status="all"` o el filtro específico (`sent`, `cancelled`, `failed`, etc.).
- **CONTRASTE crítico**: `whatsapp_pending` = INCOMING sin respuesta (chats donde tengo que contestar). `whatsapp_list_scheduled` = OUTGOING que YO programé para enviar después. Keywords: "qué chats / a quién contestar" → pending. "qué programé / qué quedan por mandar" → list_scheduled.
- "cancelá el mensaje a <Contacto> que programé" / "borrá el wsp programado a <Contacto>" / "no le mandes a <Contacto> el del <hint>" → propose_whatsapp_cancel_scheduled(contact_name="<Contacto>", when_hint="<hint opcional>"). Emite card con [Cancelar mensaje] / [Volver]. Si hay >1 pending para ese contacto y el user no dio hint claro, el tool devuelve `needs_clarification: true` + `candidates` — decile al user: "Tenés 2 programados para <Contacto>: a) el de mañana 9hs '...', b) el del viernes 14:30 '...'. ¿Cuál cancelo?". El `when_hint` es libre: "el de mañana", "el de la tarde", "el viernes".
- "reagendá / cambiá / movéme el mensaje a <Contacto> para <nueva fecha>" → propose_whatsapp_reschedule_scheduled(contact_name="<Contacto>", new_when="<NL del nuevo horario>", when_hint="<hint opcional>"). El `new_when` lo parseamos con _parse_natural_datetime — pasá lo que el user dijo crudo ("el viernes 18hs", "mañana 14:30"). Si `new_when` está en el pasado o no parsea, la card sale con error y el user reformula.
- En tu respuesta textual: card visible → 1-2 oraciones tipo "Te dejo el cancel armado, dale Cancelar para confirmar" / "Te dejo el reschedule armado para <hora simple>, dale Reagendar". Si vino error en la card, simplemente parafraseá el error ("No encontré ningún mensaje programado para <Contacto>").
- En tu respuesta textual: "Te dejo el mensaje armado para <Nombre>, revisalo y tocá Enviar si está ok." (1-2 oraciones, que quede claro que NO se envió todavía). NO repitas el texto del mensaje — la card ya lo muestra. Si programaste, decí "Te lo dejo programado para <hora simple>, revisalo y dale Programar."

Enviar emails via Gmail (misma lógica — acción destructiva, SIEMPRE pide confirmación):
- "mandale/enviale un mail a <email>: <asunto> — <cuerpo>" / "escribile un mail a <X>: <texto>" / "mandá un correo a <email>: <texto>" → propose_mail_send(to="<email>", subject="<asunto>", body="<cuerpo>").
- El `to` TIENE que ser un email válido (contiene `@`). Si el user menciona un contacto sin email explícito ("mandale mail a Grecia"), pedí aclaración: "¿A qué dirección? No me diste el email." — NO inventes un @gmail.com.
- El tool NO envía: devuelve una proposal card con [Enviar] / [Editar] / [Descartar]. Mismo flow que WhatsApp — el user confirma con click. Hasta que toque Enviar no sale nada.
- Respuesta textual 1-2 oraciones: "Te armé el mail para <to>, revisá y dale Enviar." Sin repetir el cuerpo — la card lo muestra.

Regla de citas (CRÍTICA): cita SOLO paths reales del vault devueltos por search_vault/read_note (ej. `[Algo](02-Areas/X/Algo.md)`). NUNCA cites identificadores internos ni nombres de tools: **PROHIBIDO** `[calendar_ahead](...)`, `[reminders_due](...)`, `[gmail_recent](...)`, `[finance_summary](...)`, `[credit_cards_summary](...)`, `[weather](...)`, `[propose_reminder](...)`, `[propose_calendar_event](...)`, thread_id, event_id, proposal_id, ni nada con `.md` que no haya vuelto literalmente de search_vault. Los datos de tools externas (gmail/finance/cards/calendar/reminders/weather) van en PROSA, sin markdown links.

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
    """Leer contenido completo de una nota del vault.

    Usalo cuando el user pide explícitamente leer una nota por nombre
    ("leé CLAUDE.md", "abrí la nota de Coaching", "mostrame Ikigai",
    "lee la nota X"). Tiene PRIORIDAD sobre search_vault cuando el user
    nombra un archivo concreto. NO uses search_vault como fallback —
    si el user pidió leer una nota específica, usá esta tool.

    Args:
        path: Ruta relativa al vault (ej. "02-Areas/Coaching/Ikigai.md").
            Si el user dice solo el nombre sin ".md", agregalo. Si es
            ambiguo, pasá el nombre tal cual y la tool intentará
            resolver.

    Returns:
        Markdown completo de la nota, o mensaje de error si no existe.
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


def credit_cards_summary() -> str:
    """Resúmenes de tarjeta de crédito (saldo a pagar, fechas de cierre y
    vencimiento, top consumos) parseados de los `.xlsx` que el banco
    emite por ciclo y el user deja en iCloud `/Finances`.

    Returns:
        JSON con lista de tarjetas (`[]` si no hay xlsx / openpyxl falla).
        Cada item: `{brand, last4, holder, closing_date, due_date,
        next_closing_date, next_due_date, total_ars, total_usd,
        minimum_ars, minimum_usd, top_purchases_ars, top_purchases_usd}`.
    """
    try:
        from web.server import _fetch_credit_cards  # lazy: web.server importa este módulo.
        return json.dumps(_fetch_credit_cards() or [], ensure_ascii=False)
    except Exception:
        return "[]"


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
    """Pronóstico del tiempo: hoy + próximos 2 días.

    Usalo cuando el user pregunta por el clima, temperatura, lluvia o
    pronóstico. Si el user menciona una CIUDAD específica ("clima en
    Buenos Aires", "cómo está el tiempo en Mendoza"), SIEMPRE pasá
    `location='<ciudad>'` — sin parámetro devuelve la ubicación
    configurada (default Santa Fe).

    Args:
        location: Ciudad a consultar (ej. "Buenos Aires", "Mendoza",
            "Córdoba"). Default None → usa WEATHER_LOCATION configurado.

    Returns:
        JSON con condición actual + pronóstico 3 días, o mensaje de error.
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


def whatsapp_thread(contact_name: str, max_messages: int = 15, days: int = 7) -> str:
    """Leer los últimos N mensajes con un contacto en WhatsApp.

    Usalo cuando el user te pida ver una conversación específica
    ("qué hablamos con X", "fijate qué quedamos con X", "leéme el chat
    con X") para que el LLM pueda parsear context (horas, decisiones,
    promesas) y eventualmente combinarlo con propose_calendar_event /
    propose_reminder / propose_whatsapp_reply.

    Distinto de whatsapp_search (que busca por contenido) y
    whatsapp_pending (que lista chats sin responder).

    Args:
        contact_name: Nombre del contacto resuelto via Apple Contacts.
        max_messages: Cuántos mensajes traer (1-30, default 15).
        days: Ventana en días (1-30, default 7).

    Returns:
        JSON {contact_name, jid, messages: [{ts, who, text}], count}.
    """
    return _agent_tool_whatsapp_thread(contact_name, max_messages, days)


def record_contact_observation(
    contact_name: str,
    observation: str,
    category: str | None = None,
    source_excerpt: str | None = None,
) -> str:
    """Anotar info relevante sobre un contacto en su nota viva del vault.

    Las notas de `99-Contacts/` son contactos vivos: cada vez que el user
    menciona algo relevante sobre alguien (una preferencia, un update de
    trabajo, un cumpleaños que olvidó, un tema sensible), ESTE tool lo
    graba para que la próxima vez que el LLM lea la nota tenga el contexto.
    Escribe persistente al vault — idempotente si ya existe el texto.

    Usalo **sin pedir permiso** cuando detectes info nueva sobre un
    contacto. Ejemplos de triggers:
    - "Seba me llevó un vino" → observation="Le gusta el vino",
      category="Preferencias", source_excerpt="Seba me llevó un vino"
    - "Mi vieja ahora trabaja en San Pedro" → observation="Trabaja en
      San Pedro", category="Trabajo / contexto"
    - "Oscar anda de mal humor con el tema de la herencia" → observation=
      "Sensible con el tema de la herencia", category="Notas"
    - "El cumple de Astor es el 26 de mayo" → observation="Cumpleaños:
      26 de mayo", category="Preferencias" (además del
      propose_calendar_event yearly si el user pide agendarlo)

    NO usar para queries/lookups ("qué le gusta a X" → usá search_vault /
    read_note), ni para info que el user dice "no te acordes de esto".

    Tras la llamada, mencioná 1 línea que lo anotaste: "Anotado en la nota
    de X" (sin repetir el texto). No esperes confirmación: el user puede
    ver/editar la nota siempre.

    Args:
        contact_name: Nombre tal como el user lo dijo ("Seba", "mi Mama",
            "Sebastián"). Se resuelve por full_name / aliases / filename
            del vault `99-Contacts/`.
        observation: Texto procesado, corto y accionable ("Le gusta el
            vino"). NO copies la frase cruda — eso va en source_excerpt.
        category: OPCIONAL pero recomendado. Bullet del template:
            "Preferencias", "Trabajo / contexto", "Notas", "Familia",
            "Eventos importantes". Si no matchea uno existente, se crea.
            Si lo omitís, va solo a `## Observaciones` (auditoría pura).
        source_excerpt: OPCIONAL pero recomendado. Frase cruda del user
            que motivó la obs ("Seba me llevó un vino"). Auditoría.

    Returns:
        JSON `{ok, file, observation_added, category_updated, reason?}`.
        Si el contacto no existe en el vault, `ok=false` con
        `reason="contact_not_in_vault"` — decile al user que cree la nota
        primero (copiando `99-Contacts/_template.md`).
    """
    return _agent_tool_record_contact_observation(
        contact_name, observation,
        category=category,
        source_excerpt=source_excerpt,
        source_kind="chat",
    )


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


def rag_stats() -> str:
    """Estado del vault y del índice RAG.

    Devuelve totales de chunks/URLs indexados, breakdown por source
    (vault, whatsapp, gmail, calendar, etc.), modelos activos y feedback
    counts. Read-only. Usalo cuando el user pregunta "cuántas notas
    tengo", "qué tan grande es mi vault", "qué tengo indexado", "stats
    del rag", "cómo está mi vault".

    Returns:
        JSON con `{chunks, urls, vault_path, embed_model, chat_model,
        helper_model, breakdown: [{source, chunks, pct, last_indexed}],
        feedback: {pos, neg}}`. Errores → campos vacíos + `error`.
    """
    try:
        from rag import (
            get_db, get_urls_db, _corpus_breakdown_by_source,
            VAULT_PATH, EMBED_MODEL, HELPER_MODEL, CHAT_MODEL_PREFERENCE,
            resolve_chat_model, feedback_counts,
        )
        col = get_db()
        chunks = col.count()
        try:
            urls = get_urls_db().count()
        except Exception:
            urls = 0
        try:
            breakdown_raw = _corpus_breakdown_by_source()
        except Exception:
            breakdown_raw = []
        total = sum(cnt for _, cnt, _ in breakdown_raw) or chunks or 1
        breakdown = [
            {
                "source": src or "(null)",
                "chunks": cnt,
                "pct": round(100 * cnt / total, 1),
                "last_indexed": last_ts,
            }
            for src, cnt, last_ts in breakdown_raw
        ]
        try:
            chat_model = resolve_chat_model()
        except Exception as exc:
            chat_model = f"(error: {exc})"
        try:
            pos, neg = feedback_counts()
        except Exception:
            pos, neg = 0, 0
        return json.dumps(
            {
                "chunks": chunks,
                "urls": urls,
                "vault_path": str(VAULT_PATH),
                "embed_model": EMBED_MODEL,
                "chat_model": chat_model,
                "chat_model_preference": list(CHAT_MODEL_PREFERENCE),
                "helper_model": HELPER_MODEL,
                "breakdown": breakdown,
                "feedback": {"pos": pos, "neg": neg},
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"chunks": 0, "urls": 0, "error": str(exc)}, ensure_ascii=False)


def create_note(text: str, title: str | None = None, tags: list[str] | None = None) -> str:
    """Crear una nota nueva en `00-Inbox/` del vault Obsidian.

    Usá esto cuando el user pide explícitamente capturar algo como nota
    ("anotame X", "creá una nota con Y", "guardá esto en obsidian",
    "captura esto"). Escribe directo, no es una propuesta — el user
    puede borrar/editar la nota en Obsidian si se equivocó. Para
    capturar el contenido de un link específico, usá `fetch_url` con
    `save=True` (que ya escribe la nota con summary + frontmatter
    estructurado).

    Args:
        text: Cuerpo de la nota (markdown). Lo que se escribe debajo
            del frontmatter.
        title: Título opcional. Si se omite, se usa la primera línea de
            `text` como slug del filename. Si se pasa, el filename va a
            ser `YYYY-MM-DD-HHMM-<slug-de-title>.md`.
        tags: Tags extra además del default "capture". Lista de
            strings, kebab-case sin `#` (ej. ["idea", "rag"]).

    Returns:
        JSON `{ok: true, path: "<vault-relative>", filename: "..."}`,
        o `{ok: false, error: "..."}` si el texto está vacío o el write
        falló.
    """
    try:
        from rag import capture_note, VAULT_PATH
        if not (text or "").strip():
            return json.dumps({"ok": False, "error": "texto vacío"}, ensure_ascii=False)
        tag_list = list(tags) if tags else None
        path = capture_note(
            text=text,
            tags=tag_list,
            source="ra-chat",
            title=title,
        )
        rel = str(path.relative_to(VAULT_PATH))
        return json.dumps(
            {"ok": True, "path": rel, "filename": path.name},
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def fetch_url(url: str, save: bool = True) -> str:
    """Fetchear una URL externa, extraer markdown limpio + summary.

    Usá esto cuando el user pide ingerir, leer o crear nota sobre un
    link ("creá una nota sobre este link X", "leéme esta URL", "guardá
    esta página", "fijate qué dice este artículo"). Internamente fetcha
    la página, limpia el HTML (readability/defuddle), resume con LLM
    en 1ra persona, busca notas relacionadas en el vault y arma una
    nota markdown con frontmatter `type: read`, `source: <url>`, tags
    auto-derivados, wikilinks a notas relacionadas.

    Args:
        url: URL completa empezando con http:// o https://.
        save: Si True (default), escribe la nota a
            `00-Inbox/YYYY-MM-DD-HHMM-read-<slug>.md` y la indexa al
            corpus. Si False, fetcha + arma el body pero NO escribe
            (dry-run; útil cuando el user quiere leer el contenido sin
            persistir).

    Returns:
        JSON `{ok: true, title, summary, path?, related: [...],
        tags: [...], text_len, url}`. `path` solo presente si save=true.
        Errores (URL inválida, fetch fail, paywall) → `{ok: false,
        error: "..."}`.
    """
    try:
        from rag import get_db, ingest_read_url, VAULT_PATH
        if not (url or "").strip():
            return json.dumps({"ok": False, "error": "url vacía"}, ensure_ascii=False)
        u = url.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            return json.dumps(
                {"ok": False, "error": "URL inválida: debe empezar con http:// o https://"},
                ensure_ascii=False,
            )
        col = get_db()
        result = ingest_read_url(col, u, save=bool(save))
        out: dict = {
            "ok": True,
            "title": result.get("title", ""),
            "summary": result.get("summary", ""),
            "tags": list(result.get("tags") or []),
            "related": list(result.get("related") or []),
            "text_len": int(result.get("text_len") or 0),
            "url": u,
        }
        if save and result.get("path"):
            try:
                out["path"] = str(result["path"].relative_to(VAULT_PATH))
            except Exception:
                out["path"] = str(result["path"])
        return json.dumps(out, ensure_ascii=False)
    except RuntimeError as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"fetch fail: {exc}"}, ensure_ascii=False)


# Chat-exposed tool wrappers for reminder/event creation. Real logic
# lives in rag.py so the CLI rag chat loop can reuse it without the
# web → rag → web circular import. These re-exports keep the ollama
# tool schema extraction pointed at web/tools symbols as before.
from rag import (  # noqa: E402
    propose_reminder,
    propose_calendar_event,
    propose_whatsapp_send,
    propose_whatsapp_reply,
    propose_whatsapp_send_note,
    propose_whatsapp_send_contact_card,
    propose_whatsapp_cancel_scheduled,
    propose_whatsapp_reschedule_scheduled,
    propose_mail_send,
    whatsapp_list_scheduled,
)


CHAT_TOOLS: list[Callable] = [
    search_vault,
    read_note,
    reminders_due,
    gmail_recent,
    finance_summary,
    credit_cards_summary,
    calendar_ahead,
    weather,
    drive_search,
    whatsapp_pending,
    whatsapp_search,
    whatsapp_thread,
    whatsapp_list_scheduled,
    record_contact_observation,
    rag_stats,
    create_note,
    fetch_url,
    propose_reminder,
    propose_calendar_event,
    propose_whatsapp_send,
    propose_whatsapp_reply,
    propose_whatsapp_send_note,
    propose_whatsapp_send_contact_card,
    propose_whatsapp_cancel_scheduled,
    propose_whatsapp_reschedule_scheduled,
    propose_mail_send,
]

TOOL_FNS: dict[str, Callable] = {fn.__name__: fn for fn in CHAT_TOOLS}

PARALLEL_SAFE: set[str] = {
    "weather",
    "finance_summary",
    "credit_cards_summary",
    "calendar_ahead",
    "reminders_due",
    "gmail_recent",
    "drive_search",
    "whatsapp_pending",
    "whatsapp_search",
    "whatsapp_thread",
    "whatsapp_list_scheduled",  # query-only contra SQLite local — safe.
    "rag_stats",  # read-only contra ragvec/telemetry DBs.
    # create_note + fetch_url NO van acá: hacen file writes (capture_note
    # → 00-Inbox/) o LLM-blocking calls (ingest_read_url → summary +
    # tag-derivation con qwen2.5:7b). Mejor serializarlos para que el
    # debugging sea claro si algo falla.
    "propose_reminder",
    "propose_calendar_event",
    # `propose_whatsapp_send` intencionalmente NO está acá: aunque el tool
    # NO envía (solo drafts), el contact-lookup via osascript es side-
    # effectful (2-3s de latencia) y ejecutarlo en paralelo con otros
    # tools complica el debugging si hay un hang del bridge. Además la
    # semántica de "enviar mensaje" debería correrse aislada — el user
    # espera UN draft por turno, no varios en paralelo.
    # Idem `propose_whatsapp_cancel_scheduled` / `_reschedule_scheduled`:
    # también hacen contact-lookup via osascript y operan sobre estado
    # mutable (rag_whatsapp_scheduled) — better aislado.
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
    "propose_whatsapp_send_note",
    "propose_whatsapp_send_contact_card",
    "propose_whatsapp_cancel_scheduled",
    "propose_whatsapp_reschedule_scheduled",
    "propose_mail_send",
}

# Tool names que CREAN cosas directo (no son propuestas) pero que comparten
# el code path del create-intent narrowing (`is_propose_intent` en
# web/server.py). El narrowing original a `PROPOSAL_TOOL_NAMES` solamente
# excluía estas tools, dejándolas inalcanzables cuando el regex
# `_PROPOSE_INTENT_RE` matcheaba ("creá una nota", URLs crudas, etc.).
# Distintas de las `propose_*` porque NO emiten SSE `proposal` events:
# ejecutan directo, sin tarjeta de confirmación.
CREATE_ACTION_TOOL_NAMES: set[str] = {
    "create_note",
    "fetch_url",
}

# Union — usada por el narrowing en web/server.py:15316-15320.
PROPOSE_OR_CREATE_TOOL_NAMES: set[str] = (
    PROPOSAL_TOOL_NAMES | CREATE_ACTION_TOOL_NAMES
)
