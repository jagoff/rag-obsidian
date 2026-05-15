"""Pre-router intent detection — keyword/regex → forced tool routing.

Extraído de ``web/server.py`` (Phase W1, 2026-05-08) para que los detectors
sean unit-testeables sin tener que cargar el FastAPI app entero. Back-compat:
``web/server.py`` re-importa todos estos nombres así
``from web.server import _detect_tool_intent`` (que es como tests viejos los
importan) sigue andando.

## Surface

Constantes:
- ``_PLANNING_PAT`` — regex parcial usado por reminders/calendar/whatsapp
  rules para matchear "planning words" (hoy/mañana/semana/...).
- ``_TOOL_INTENT_RULES`` / ``_TOOL_INTENT_COMPILED`` — la tabla principal
  (name, args, regex) que `_detect_tool_intent` itera.
- ``_READ_NOTE_TRIGGER_RE``, ``_SEARCH_VAULT_TRIGGER_RE``,
  ``_WEATHER_LOCATION_RE``, ``_WA_LIST_SCHEDULED_TRIGGER_RE``,
  ``_WA_INTENT_RE``, ``_GROUP_PREFIX_RE``, ``_ANAPHORIC_TEMPORAL_RE``,
  ``_ANAPHORIC_REFERENCE_RE`` — regexes singletons pre-compilados para
  evitar recompilar por request en /api/chat (CPython no cachea inline
  flags).

Functions:
- ``_detect_search_vault_intent(q)`` — "busca/buscame X" → search_vault.
- ``_detect_read_note_intent(q)`` — "leé la nota X" / "X.md" → read_note.
- ``_detect_weather_explicit_location_intent(q)`` — "clima en CIUDAD" →
  weather con location explícita (skipea morning-brief noise).
- ``_detect_whatsapp_list_scheduled_intent(q)`` — "qué WA programados" →
  whatsapp_list_scheduled (outgoing) sin caer a whatsapp_pending por
  "mañana".
- ``_detect_tool_intent(q)`` — orchestrator que llama los específicos
  primero, después itera la tabla. Devuelve `[(tool_name, args), ...]`.
- ``_resolve_anaphoric_args(tool_name, question, last_location)`` — args
  para tool re-fire en follow-up anafórico ("y en Barcelona?").

## Por qué módulo separado

- ``web/server.py`` post-WIP es 23.9k LOC. Tests del intent router
  (test_drive_search_tool, test_chat_latency_fixes_2026_05_01) tienen que
  importar el FastAPI app entero solo para llamar `_detect_tool_intent`,
  pagando el cost de prewarm + sentence-transformers + MLX init.
- Mantener constantes regex acá las hace unit-testeables sin pulling del
  resto del web stack.
"""

from __future__ import annotations

import re


# 2026-04-28 BUG-2 / BUG-3: planning verbs sin plural matcheaban "días"
# en queries idiomáticas y disparaban tools spuriamente. Fix: sufijo `s?`
# en cada token flexionado.
_PLANNING_PAT = (
    r"\bsemanas?\b|\bhoy\b|\bma[ñn]ana\b|pasado\s+ma[ñn]ana|\bd[ií]as?\b|c[oó]mo\s+viene"
    # "qué tengo / hay / tenés" only counts as planning when followed by a
    # temporal token (hoy, mañana, semana, día, agenda, pendiente) so that
    # "qué tengo sobre coaching" doesn't fire calendar + reminders.
    r"|qu[eé]\s+(tengo|hay|ten[eé]s)\b(?=.{0,40}\b(hoy|ma[ñn]ana|semanas?|d[ií]as?|agendas?|pendient|tarea|recordator)\b)"
)

_TOOL_INTENT_RULES: tuple[tuple[str, dict, str], ...] = (
    ("finance_summary", {}, r"gast[oéó]s?|gast[aá][mn]os|gastar|presupuesto|plata|finanz|moze|sueldo|haberes|ingresos|neto|recibo.*haberes|recibo.*sueldo|cu[aá]nto.*cobr[aeo]|cu[aá]nto.*gan[aeo]"),
    # Tarjetas: keywords inequívocos del dominio "resumen de tarjeta de
    # crédito" → fuerza credit_cards_summary.
    #
    # Triggers (en orden de especificidad):
    #   1. tarjeta(s) | visa | master(card) | amex | crédito           — marca
    #   2. saldo a pagar | fecha de cierre | fecha de vencimiento      — ciclo
    #   3. resumen de tarjeta | cuánto debo                            — directo
    #   4. último/reciente + gasto/consumo/movimiento/compra/cargo     — recientes
    #   5. consumo/movimiento/cargo + de/del/en/con/mi                 — fraseo
    #
    # 2026-04-26: ampliado tras user report — el LLM alucinaba "Helado en
    # Manalu $15K" cuando la query era "cuál fue mi último gasto" (sin
    # "tarjeta" explícito) porque solo finance_summary disparaba (MOZE).
    # Triggers 4+5 disparan AMBOS tools (finance_summary + credit_cards_
    # summary): el LLM ve los dos contextos y prioriza el más específico
    # via REGLA 1.b del system prompt (datos transaccionales: cita literal,
    # no inventes).
    #
    # No matchean por diseño: "saldo" solo (puede ser cuenta/billetera),
    # "tiempo" (clima), "gasto" solo sin marca (MOZE genérico).
    ("credit_cards_summary", {}, r"\btarjet|\bvisa\b|\bmaster(?:card)?\b|\bamex\b|\bcr[eé]dito\b|saldo.*paga|fecha.*cierre|fecha.*vencim|resumen.*tarjeta|cu[aá]nto.*deb[oe]|\b(?:[uú]ltim[oa]s?|recientes?)\s+(?:gastos?|consumos?|movimientos?|compras?|cargos?|transac)\b|\b(?:consumos?|movimientos?|cargos?)\s+(?:de|del|en|con|mi)"),
    # NOTE: "recordame" / "agendáme" used to be here as query triggers, but
    # they're CREATE intents now (propose_reminder / propose_calendar_event).
    # Moved to `_PROPOSE_INTENT_RE` below. `recordator` still matches
    # "qué recordatorios tengo" → list.
    # `to-do` requiere hyphen literal — `to.?do` (any-char) matcheaba
    # "todos" en queries idiomáticas tipo "mejor serie de todos los tiempos"
    # y disparaba este tool spuriamente. Spanish task vocabulary ya está
    # cubierto por `tarea|pendient|recordator`.
    ("reminders_due",   {}, r"pendient|tarea|\bto-do\b|recordator|" + _PLANNING_PAT),
    # "inbox" / "bandeja" alone are too generic — Obsidian PARA uses
    # "00-Inbox" heavily. Require an explicit mail signal.
    ("gmail_recent",    {}, r"\b(mail|correo|e.?mail|gmail)s?\b|bandeja\s+de\s+entrada"),
    ("calendar_ahead",  {}, r"calendari|\beventos?\b|\bcitas?\b|reuni[oó]n|\bagendas?\b|pr[oó]xim[ao]s?\s+d[ií]as|" + _PLANNING_PAT),
    # `\btiempo\b` SIN plural: "tiempos" en español es casi siempre idiom
    # no-weather ("de todos los tiempos", "tiempos modernos", "los tiempos
    # cambian"). El singular "tiempo" sí es weather común ("qué tiempo
    # hace").
    ("weather",         {}, r"\bclimas?\b|\btiempo\b|llov|lluvia|temperatur|pron[oó]stico"),
    # Drive: "drive" alone es ambiguo (también es una palabra EN genérica,
    # "drive-thru", "hard drive"), pero "drive" precedido por
    # google/en/mi/tu/del/al/a/sobre es señal clara de la intent
    # "buscar en Google Drive". También disparan keywords
    # Drive-específicas (planilla, spreadsheet, sheet, presentación).
    # "drive_search" es el único tool del pre-router que recibe args
    # dinámicos — la query cruda del user se inyecta en
    # `_detect_tool_intent` con key `query`, y el helper de rag filtra
    # stopwords + resuelve aliases internamente. Ver
    # `_agent_tool_drive_search` para la lógica completa.
    ("drive_search",    {}, r"(?:google\s*|(?:en|mi|tu|del|al|a|sobre)\s+)drive\b|\bplanillas?\b|\bspreadsheets?\b|\bsheets?\b|\bpresentaci[oó]n\b|\bpresentaciones\b"),
    # WhatsApp: "qué tengo pendiente" sin contexto WA explícito también
    # dispara el tool — los chats sin respuesta son parte del "pending
    # bucket" semántico del user (reportado 2026-04-24 Fer F.: "me
    # faltan las conversaciones de wzp relativa a la pregunta" cuando
    # preguntó por pendientes de la semana). Keywords explícitos
    # (whatsapp/wzp/wsp/chats) también gatillan. `pendient` plain
    # (sin `\b`) matcha "pendiente(s)/pendient(e) de contestar/etc" —
    # mismo patrón que usa `reminders_due`, así el scope de "pending
    # bucket" está consistente entre tasks/events y chats. `chats?` por
    # sí solo es aceptable porque las otras apps de chat del user
    # (Slack, Messages) no están integradas al RAG todavía — en la
    # práctica "chat" == WhatsApp en este setup.
    ("whatsapp_pending", {}, r"whats.?app|\bwzp\b|\bwsp\b|\bchats?\b|mensajes?\s+sin|pendient|" + _PLANNING_PAT),
    # `whatsapp_search` — buscar DENTRO del contenido de los mensajes
    # (corpus indexado, ~4500 chunks), distinto de `whatsapp_pending`
    # (que sólo lista chats sin respuesta). Triggers son frases de
    # "comunicación pasada" que no tocan los keywords de pending:
    #   - "qué me/te/le dijo Juan" / "qué me dijeron"
    #   - "qué me mandó/escribió María" / "qué me comentó"
    #   - "cuándo quedamos / hablamos / charlamos / acordamos"
    #   - "el chat donde X mencionó Y" / "dónde mencionó / habló de"
    # Importante: NO solapamos con `whatsapp_pending` porque ahí los
    # keywords son `whatsapp/wzp/wsp/chats?/mensajes sin/pendient`. Las
    # frases de acá hablan de pasado (dijo/mandó/quedamos), no de
    # estado (pendiente/sin responder). La cobertura adicional la pone
    # el LLM via el addendum cuando el pre-router no engancha. Pasamos
    # la query cruda al tool — el helper resuelve `contact` desde el
    # texto natural a futuro; por ahora deja `contact=None` y el
    # retrieve unfiltered ya devuelve los matches relevantes.
    ("whatsapp_search", {}, (
        r"qu[eé]\s+(me|te|le|nos)?\s*"
        r"(dij[oe]|dij[oe]ron|mand[oó]|mandaron|escribi[oó]|escribieron|coment[oó]|comentaron|cont[oó]|contaron)"
        r"|cu[aá]ndo\s+(quedam|hablam|charlam|acordam)"
        r"|d[oó]nde\s+(hablam|charlam|menci[oó]n|qued[oó])"
        r"|el\s+chat\s+(donde|en\s+el\s+que)"
    )),
)
_TOOL_INTENT_COMPILED = tuple(
    (name, args, re.compile(pat, re.IGNORECASE)) for name, args, pat in _TOOL_INTENT_RULES
)


# Pre-router heuristic for read_note (eval 2026-04-28 fix). El LLM
# tiende a ignorar read_note y prefiere search_vault aunque el user
# pidió leer un archivo específico. Disparamos el tool a mano cuando
# el patrón es clarísimo.
#
# Conservative match: SOLO dispara si hay (a) qualifier explícito
# "la nota / el archivo / el md / el markdown" + nombre, O (b) un
# nombre que termina en `.md`. Sin esto el regex levantaba "abrí
# gmail" → read_note('gmail.md') (rompía test_web_tool_intent_plurals
# y test_whatsapp_search_tool, eval 2026-04-28).
_READ_NOTE_TRIGGER_RE = re.compile(
    r"\b(?:le[eé]|abr[ií]|mostrame|muestra|mu[eé]strame|"
    r"showme|show\s+me)\s+"
    r"(?:"
    # (a) qualifier-prefixed: "la nota CLAUDE", "el archivo notas",
    # "el md plan", "la nota 02-Areas/Foo".
    r"(?:la\s+nota|el\s+archivo|el\s+md|el\s+markdown)\s+"
    r"([A-Za-z0-9_\-./áéíóúñÁÉÍÓÚÑ][A-Za-z0-9_\-./áéíóúñÁÉÍÓÚÑ]{1,79})"
    r"|"
    # (b) bare path con extensión `.md` explícita.
    r"([A-Za-z0-9_\-./áéíóúñÁÉÍÓÚÑ][A-Za-z0-9_\-./áéíóúñÁÉÍÓÚÑ]{1,79}\.md)"
    r")",
    re.IGNORECASE,
)


_SEARCH_VAULT_TRIGGER_RE = re.compile(
    # "busca|buscá|buscame|encontrame|fijate" + objeto (lo que sigue es
    # query libre, normalmente palabras o nombre de nota). NO requerimos
    # qualifier — "busca <X>" siempre debería disparar search_vault para
    # evitar que el LLM caiga al fallback whatsapp_pending/reminders_due.
    # 2026-05-07: bug "busca Frases positivas para hablar" → LLM
    # respondía con template de pendientes, ignorando search.
    r"\b(?:busca(?:me|r)?|busc[aá](?:me)?|encontrame|encontr[aá]|fijate|"
    r"fij[aá]te|find(?:me)?|search)\b\s+(?P<query>.{2,120}?)(?:[?.!]|$)",
    re.IGNORECASE | re.DOTALL,
)


def _detect_search_vault_intent(q: str) -> tuple[str, dict] | None:
    """Detect 'buscá/buscame/encontrame X' → force search_vault.

    Returns ('search_vault', {'query': <text>}) si matchea, None si no.

    2026-05-07: el LLM (qwen2.5:7b 4-bit) bias-eaba a llamar
    whatsapp_pending + reminders_due cuando la query empezaba con
    'busca' — el verbo no está en el routing keyword del addendum y el
    modelo defaultea a tools de "what's pending" defensivamente.
    Pre-router fuerza search_vault con la query cruda; el LLM nunca
    llega a la decisión de tools.
    """
    if not q or not q.strip():
        return None
    m = _SEARCH_VAULT_TRIGGER_RE.search(q)
    if not m:
        return None
    qtext = (m.group("query") or "").strip()
    if len(qtext) < 2:
        return None
    return ("search_vault", {"query": qtext})


def _detect_read_note_intent(q: str) -> tuple[str, dict] | None:
    """Detect 'leé la nota X / abrí X.md / mostrame el archivo X' patterns.

    Returns ('read_note', {'path': resolved_path}) si matchea, None si no.

    Conservative: solo dispara si el pattern es clarísimo (qualifier
    explícito tipo "la nota / el archivo" o sufijo `.md`). False
    positives son peor que false negatives — si dudoso, el LLM puede
    igual elegir read_note vía el tool addendum.
    """
    if not q or not q.strip():
        return None
    m = _READ_NOTE_TRIGGER_RE.search(q)
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or "").strip()
    if not raw:
        return None
    # Si no termina en .md, agregarlo. Y si el user pasó solo el nombre,
    # asumimos que es relativo al root del vault.
    path = raw if raw.lower().endswith(".md") else f"{raw}.md"
    return ("read_note", {"path": path})


# Eval 2026-04-28 BUG-2b: queries "clima en CIUDAD" disparaban 4 tools
# en paralelo (weather + reminders + calendar + wa_pending) porque
# "hoy" matchea _PLANNING_PAT. Cuando el user es explícito sobre la
# ciudad, queremos SOLO weather con esa location, sin morning-brief
# noise.
#
# 2026-04-28 wave-3: relajado el regex porque la versión inicial era
# demasiado estricta:
#   1. "qué clima hace en BA hoy" no matcheaba — "hace" intercalado
#      entre keyword y prep cortaba el match. Ahora aceptamos hasta 3
#      palabras intermedias.
#   2. Location lowercase ("buenos aires") fallaba porque pedía mayúscula
#      al principio. Ahora aceptamos cualquier letra inicial.
#   3. Lookahead final era demasiado restrictivo. Ahora dejamos que la
#      location sea cualquier cosa hasta `?` / `,` / fin de string /
#      keyword temporal.
_WEATHER_LOCATION_RE = re.compile(
    r"\b(?:clima|tiempos?|pron[oó]stico|temperatur[ao])\b"
    r"(?:\s+\w+){0,3}?"  # palabras intermedias opcionales (hace, hay, está, etc.)
    r"\s+(?:en|de|para|para\s+el?)\s+"
    r"(?P<loc>[A-Za-zÁÉÍÓÚÑáéíóúñ][\wáéíóúñÁÉÍÓÚÑ\s]{1,40}?)"
    r"(?=\s*(?:[?.,!]|$|\bhoy\b|\bmañana\b|\bahora\b|\bpasado\s+mañana\b))",
    re.IGNORECASE,
)


# 2026-04-28 wave-8 (eval Conv 4 T2): patrones de follow-up anafórico
# temporal que indican "lo mismo del turno previo, pero shifteado en
# tiempo o en el slot de location". Repro: "y mañana?" tras "qué hago
# hoy?" — el LLM alucinaba propose_reminder. Usado por el carry-over
# pre-router para re-fire los mismos read-intent tools del turno previo.
_ANAPHORIC_TEMPORAL_RE = re.compile(
    r"^\s*"
    r"(?:y\s+)?"  # opcional "y ..."
    r"(?:"
    # Adverbios temporales puros (con o sin trailing words cortos)
    r"(?:hoy|ma[ñn]ana|ayer|anoche|antier|pasado\s+ma[ñn]ana)"
    # Frases temporales "la/esta/proxima semana/tarde/noche/mañana"
    r"|(?:la|esta|esa|el|este|pr[oó]xim[oa]|la\s+pr[oó]xima|la\s+que\s+viene)"
    r"\s+(?:semana|tarde|noche|ma[ñn]ana|finde|fin\s+de\s+semana|mes|a[ñn]o|d[ií]a)"
    r"(?:\s+que\s+viene)?"
    # Location follow-up: "en X"
    r"|en\s+\w+(?:\s+\w+)?"
    # Weather attribute follow-up: "cuánta lluvia/nieve" (con cualquier
    # trailing — "cuánta lluvia se espera?")
    r"|cu[aá]nta?\s+(?:lluvia|nieve|tormenta|llovizna|fr[ií]o|calor)"
    r")"
    # Trailing chars hasta cap razonable — permite "se espera?", "habrá?",
    # etc. sin saltar a frases largas que ya no son anafóricas.
    r"\b.{0,30}\s*[?!.]?\s*$",
    re.IGNORECASE,
)


# 2026-04-28 wave-8 (P3 #10): patrones meta/referenciales de follow-up.
# Distintos del temporal — acá el user pregunta SOBRE la lista del turno
# previo en vez de cambiar la ventana temporal. Ejemplos del eval Conv 2:
#
#   T1: "qué eventos esta semana?" → calendar_ahead
#   T2: "y de eso qué puedo postergar?" → ❌ off-topic (sin fix)
#
# Match patterns:
#   - "y de eso/esos/esas/esto/estas"
#   - "cuál podría/puedo/podés/sería/es el más X"
#   - "los demás", "el primero", "el último", "el de X"
#   - "el más urgente/importante/...", "qué prioridad...", "qué es lo más X"
#
# Cuando matchea: re-fire el read tool del turno previo Y propagamos su
# output renderizado al CONTEXTO con un header "DATOS DEL TURNO ANTERIOR".
_ANAPHORIC_REFERENCE_RE = re.compile(
    r"^\s*"
    r"(?:y\s+)?"
    r"(?:"
    # "de eso/eso(s)/esto/estas/aquellos" — referencia directa al turno previo
    r"(?:de|sobre|en)\s+(?:eso|esos?|esto|estas?|aquellos?|aquellas?)"
    # "los/las anteriores", "los demás", "el primero", "el último"
    r"|(?:los|las)\s+(?:anteriores?|dem[aá]s|primeros?|[uú]ltimos?)"
    r"|el\s+(?:primero|[uú]ltimo|m[aá]s\s+\w+|de\s+\w+)"
    # "cuál es el más X" — solo anaphoric cuando incluye superlativo/comparativo.
    # Antes esto era `cu[aá]l\s+(?:es|...)` sin guard, lo cual matcheaba
    # "cuál es el descargo del alquiler" (factual, no anafórico) y disparaba
    # carryover del tool del turno previo. Bug repro 2026-05-08.
    r"|cu[aá]l\s+(?:es|son)\s+(?:el|la|los|las|lo)\s+(?:m[aá]s|menos|mejor|peor)\b"
    # "cuál podría/sería/podés/puedo/debería/tendría" — verbos modales sin
    # sustantivo concreto suelen ser anafóricos ("cuál podría posponer?").
    r"|cu[aá]l\s+(?:podr[ií]a|ser[ií]a|pod[eé]s|puedo|deber[ií]a|tendr[ií]a)"
    # "qué tan X", "qué X es lo más Y", "qué es lo más X"
    r"|qu[eé]\s+(?:tan|es\s+lo\s+m[aá]s|es\s+m[aá]s)"
    # "cuáles son los más X", "cuáles puedo X"
    r"|cu[aá]les\s+(?:son|puedo|podr[ií]a|deber[ií]a)"
    r")"
    r"\b.{0,80}\s*[?!.]?\s*$",
    re.IGNORECASE,
)


def _resolve_anaphoric_args(tool_name: str, question: str, last_location: str | None) -> dict:
    """Compute args para el tool re-fire en follow-up anafórico.

    Para `weather` específicamente: si la query menciona una nueva location
    (ej. "y en Barcelona?"), usa esa; sino usa la `last_location` del turno
    previo. Esto preserva el contexto de la ciudad sin re-prompt.

    Para otros tools (calendar_ahead, reminders_due, etc.) los args default
    son fine — el tool ya respeta horizonte de "hoy/esta semana" interno.
    """
    args: dict = {}
    if tool_name == "weather":
        # Detectar si query menciona nueva location.
        m = re.search(r"\ben\s+([A-ZÁÉÍÓÚÑa-záéíóúñ][\wáéíóúñÁÉÍÓÚÑ\s]{1,40}?)(?=\s*[?!.]?\s*$)", question, re.IGNORECASE)
        if m:
            args["location"] = m.group(1).strip().rstrip(",.;:")
        elif last_location:
            args["location"] = last_location
    return args


def _detect_weather_explicit_location_intent(q: str) -> tuple[str, dict] | None:
    """Detect 'clima/tiempo en CIUDAD' patterns. Returns ('weather', {'location': X})
    si matchea, None si no. Cuando matchea, debe forzar SOLO weather sin
    disparar morning-brief tools.

    Conservative pero no demasiado: keyword ('clima'/'tiempo'/'pronóstico')
    + (opcional palabras intermedias) + preposición ('en'/'de'/'para')
    + ciudad. Acepta lowercase ("buenos aires") y mayúscula ("BA").
    """
    if not q or not q.strip():
        return None
    m = _WEATHER_LOCATION_RE.search(q)
    if not m:
        return None
    loc = (m.group("loc") or "").strip().rstrip(",.;:")
    # Strip trailing common time words si quedaron pegados al match
    for tail in (" hoy", " mañana", " ahora", " ahorita"):
        if loc.lower().endswith(tail):
            loc = loc[: -len(tail)].strip()
    # 2026-04-28 wave-4: strip trailing preposiciones que quedaron pegadas
    # cuando el regex non-greedy capturó "Mendoza para" en "clima en Mendoza
    # para mañana". Una prep al final NUNCA es parte del nombre de ciudad.
    for prep in (" para el", " para", " de", " en", " hasta"):
        if loc.lower().endswith(prep):
            loc = loc[: -len(prep)].strip()
    if not loc or len(loc) < 2:
        return None
    return ("weather", {"location": loc})


# Eval 2026-04-28 BUG-3: queries "qué WA programados" se ruteaban a
# whatsapp_pending (incoming) en vez de whatsapp_list_scheduled
# (outgoing). Causa: la palabra "mañana" matcheaba _PLANNING_PAT del
# whatsapp_pending. Fix: helper específico que detecta keywords claros
# de "mensajes que YO programé" y fuerza whatsapp_list_scheduled.
_WA_LIST_SCHEDULED_TRIGGER_RE = re.compile(
    r"\b(?:programad\w*|scheduled|"
    r"(?:quedan|que\s+quedan)\s+por\s+mandar|"
    r"pendientes?\s+de\s+mandar|"
    r"qu[eé]\s+(?:program[eé]|tengo\s+programado\w*))\b",
    re.IGNORECASE,
)

# Singleton pre-compiled — evita re.compile() por request en /api/chat.
# CPython no cachea compilaciones con flags inline, así que la versión
# `re.search(r"...", q, re.IGNORECASE)` recompilaba ~30 veces/min.
_WA_INTENT_RE = re.compile(
    r"\b(whatsapp|\bwa\b|mensaje|chat de|último[s]? chat)",
    re.IGNORECASE,
)


def _detect_whatsapp_list_scheduled_intent(q: str) -> tuple[str, dict] | None:
    """Detect 'qué mensajes WA programados / quedan por mandar' patterns.

    OUTGOING que YO programé (símil a whatsapp_list_scheduled), NO confundir
    con whatsapp_pending (INCOMING sin contestar). Conservative: solo dispara
    con keywords explícitos.

    Returns ('whatsapp_list_scheduled', {}) si matchea, None si no.
    """
    if not q or not q.strip():
        return None
    if not _WA_LIST_SCHEDULED_TRIGGER_RE.search(q):
        return None
    # Sanity: la query debe tocar el dominio WhatsApp. Aceptamos plurales
    # ("wsps", "wzps") y variantes como "whatsapp"/"whatsapps".
    if not re.search(r"\b(?:whats\w*|wzps?|wsps?|mensajes?)\b", q, re.IGNORECASE):
        return None
    return ("whatsapp_list_scheduled", {})


def _detect_tool_intent(q: str) -> list[tuple[str, dict]]:
    """Deterministic keyword → tool routing. Returns (name, args) tuples
    to execute BEFORE the LLM tool-deciding call. Empty list = no forced
    tools (LLM decides freely).

    `drive_search` y `whatsapp_search` reciben la query cruda como
    `query` arg — los tool helpers tokenizan / fan-out internamente.
    All other tools have static args from their rule entry.
    """
    if not q:
        return []
    # read_note tiene prioridad: si el user pidió explícitamente leer
    # una nota específica, devolvemos eso solo y skippeamos las regex
    # genéricas (que podrían disparar reminders_due / calendar_ahead /
    # etc. spuriamente sobre la misma frase).
    _read = _detect_read_note_intent(q)
    if _read is not None:
        return [_read]
    # search_vault explícito ("busca X / buscame X / encontrame X")
    # tiene segunda prioridad: el LLM bias-ea a whatsapp_pending /
    # reminders_due cuando no reconoce el verbo y defaultea a tools de
    # "what's pending". Forzamos search_vault con la query cruda.
    _sv = _detect_search_vault_intent(q)
    if _sv is not None:
        return [_sv]
    # weather con location explícita tiene prioridad: skipea morning-brief
    # spurioso disparado por "hoy"/"mañana" en _PLANNING_PAT.
    _wx = _detect_weather_explicit_location_intent(q)
    if _wx is not None:
        return [_wx]
    # whatsapp_list_scheduled tiene prioridad: keywords claros de "outgoing
    # que YO programé" no deben disparar whatsapp_pending por "mañana".
    _wa_sched = _detect_whatsapp_list_scheduled_intent(q)
    if _wa_sched is not None:
        return [_wa_sched]
    out: list[tuple[str, dict]] = []
    for name, args, rx in _TOOL_INTENT_COMPILED:
        if not rx.search(q):
            continue
        if name == "drive_search":
            out.append((name, {"query": q}))
        elif name == "whatsapp_search":
            out.append((name, {"query": q}))
        else:
            out.append((name, dict(args)))
    return out


__all__ = [
    "_PLANNING_PAT",
    "_TOOL_INTENT_RULES",
    "_TOOL_INTENT_COMPILED",
    "_READ_NOTE_TRIGGER_RE",
    "_SEARCH_VAULT_TRIGGER_RE",
    "_WEATHER_LOCATION_RE",
    "_ANAPHORIC_TEMPORAL_RE",
    "_ANAPHORIC_REFERENCE_RE",
    "_WA_LIST_SCHEDULED_TRIGGER_RE",
    "_WA_INTENT_RE",
    "_detect_search_vault_intent",
    "_detect_read_note_intent",
    "_detect_weather_explicit_location_intent",
    "_detect_whatsapp_list_scheduled_intent",
    "_detect_tool_intent",
    "_resolve_anaphoric_args",
]
