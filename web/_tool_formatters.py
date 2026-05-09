"""Forced-tool output renderers — JSON output → markdown CONTEXTO blocks.

Extraído de ``web/server.py`` (Phase W1, 2026-05-08). Cada formatter recibe
el output crudo (string JSON) de un tool del pre-router y devuelve un
markdown chunk listo para insertar en el CONTEXTO del LLM. Diseñados para
NO leak el tool name (evita citation artifacts tipo ``[[calendar_ahead]]``)
y para empty-state explícito en español rioplatense.

## Surface

Empty-state detector:
- ``_is_empty_tool_output(name, raw)`` — True si el tool devolvió un shape
  que semánticamente significa "no hay nada" (lista vacía, threads=[],
  etc.). Usado por el pre-router para decidir REPLACE vs PRESERVE del
  CONTEXTO retrieve.

Dispatcher:
- ``_format_forced_tool_output(name, raw)`` — routea por `name` al
  formatter específico. Unknown tool → fallback con name leak (único caso).

Per-tool formatters (todos retornan markdown con header ``### <Sección>``,
nunca raisean — malformed JSON cae a passthrough con header):
- ``_format_reminders_block`` — Apple Reminders (con dedup + bucket tags ES).
- ``_format_calendar_block`` — eventos con anclaje temporal + traducción
  de date_label icalBuddy a español.
- ``_format_gmail_block`` — Gmail con bucketing de noise (CI / security /
  marketing / automated / personal) + count desglose en CI por outcome.
- ``_format_finance_block`` — finance summary passthrough JSON pretty.
- ``_format_weather_block`` — weather con summary prominente (ciudad: desc
  + temp).
- ``_format_cards_block`` — credit cards con saldo ARS/USD + vencimiento
  + top consumos.
- ``_format_drive_block`` — Google Drive search results con body cap 2500.
- ``_format_whatsapp_block`` — WA pending chats con humanize wait time.
- ``_format_whatsapp_search_block`` — WA search hits con outbound marker.

Helpers:
- ``_translate_calendar_label`` — icalBuddy "today/tomorrow/..." → tag ES.
- ``_classify_gmail_thread`` — clasifica thread en bucket de noise.

Constantes:
- ``_BUCKET_ES``, ``_CALENDAR_LABEL_ES``, ``_GMAIL_*_RE``.

## Por qué módulo separado

- Tests del renderer (test_drive_search_tool, test_credit_cards_parser, etc.)
  no necesitan el FastAPI app entero — los formatters son funciones puras
  string → string que solo dependen de stdlib (`json`, `re`, `datetime`).
- Cada bug de output rendering (citation leak, empty-state vague, count
  miscount) ahora tiene un punto de inspección único en vez de buscar entre
  23.9k LOC del server.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime


# ── Forced-tool output renderer (2026-04-22, Fer F. report) ──────────
# Before: the pre-router dumped raw JSON under a `## {tool_name}` header
# into the CONTEXTO block. qwen2.5:7b reacted badly — dropped `undated`
# items, invented reminders that weren't in the feed, and occasionally
# seeded citation artifacts like `[[calendar_ahead]]` because the tool
# name leaked as a wikilink-ish token.
#
# The helpers below render each forced-tool result as tidy markdown the
# LLM can cite without inventing: Spanish bucket labels, explicit empty
# states, dedup by (name, due), no tool-name leak, graceful fallback on
# malformed JSON (so a tool exception never crashes the request). Plays
# nicely with REGLA 1 ("engancháte SIEMPRE con el CONTEXTO") because the
# block lives INSIDE the CONTEXTO slot the LLM is pinned on.
#
# Shapes assumed (see web/tools.py):
#   reminders_due  → {"dated": [{name,due,bucket,list}], "undated":[...]}
#   calendar_ahead → [{title, date_label, time_range}]
#   gmail_recent   → {"unread_count": int, "threads": [{kind,from,...}]}
#   finance_summary→ dict with variable fields (passthrough JSON, pretty)
#   weather        → plain string (already friendly)
#   unknown tool   → header + raw content (name allowed as disambiguator)
_BUCKET_ES: dict[str, str] = {
    "overdue": "vencido",
    "today": "hoy",
    "upcoming": "próximo",
}


# Mapeo de date_labels en inglés (formato icalBuddy) → español. icalBuddy
# emite "today" / "tomorrow" / "day after tomorrow" para fechas relativas
# corto plazo, y YYYY-MM-DD o "lunes, abril 28, 2026" para fechas lejanas.
# El LLM (qwen2.5:7b) traducía "tomorrow" a "mañana" pero NO se daba
# cuenta de la inconsistencia con la pregunta del user ("hoy"). Pre-fix
# 2026-04-28 (Playwright autónomo): pregunta "qué tengo agendado para
# hoy específicamente" → respuesta "- Psiquiatra (mañana)" sin
# disclaimer de que NO había nada hoy. Forzar el label en español + tag
# relativo explícito le da al LLM el anclaje temporal que le faltaba.
_CALENDAR_LABEL_ES = {
    "today": "HOY",
    "tomorrow": "MAÑANA",
    "day after tomorrow": "PASADO MAÑANA",
}


# 2026-04-28 wave-8: noise detectors para gmail_recent. Agrupar CI/security/
# marketing reduce ruido en el CONTEXTO y pistas el LLM hacia los mails que
# importan ("tenés 4 mails de CI fallando + 1 de OSDE + 1 personal" en vez
# de listar cada notificación con su SHA).
_GMAIL_GITHUB_BRACKET_RE = re.compile(r"^\s*\[[\w.-]+/[\w.-]+\]")  # [user/repo]
_GMAIL_CI_RE = re.compile(
    r"\b(?:run\s+(?:failed|cancelled|succeeded|skipped)|ci\s+(?:failed|passed)|"
    r"build\s+(?:failed|passed|succeeded)|workflow\s+run|pipeline)\b",
    re.IGNORECASE,
)
_GMAIL_SECURITY_RE = re.compile(
    r"\b(?:security\s+alert|vulnerability|cve-?\d{4}|dependabot|advisory|"
    r"alerta\s+de\s+seguridad)\b",
    re.IGNORECASE,
)
_GMAIL_MARKETING_RE = re.compile(
    r"\b(?:newsletter|unsubscribe|promo(?:ci[oó]n)?|oferta|"
    r"discount|sale|black\s+friday|cyber\s+monday)\b",
    re.IGNORECASE,
)
_GMAIL_NOREPLY_RE = re.compile(
    # Conservative: solo no-reply explícito + alias claramente automatizados.
    # `info@` se quitó porque OSDE/Galicia/etc usan `info@org` para
    # notificaciones legítimas que al user le importan (turnos médicos,
    # avisos bancarios). Mejor un false negative en automated que
    # ocultar un mail real.
    r"(?:no.?reply|noreply|donot.?reply|news(?:letter)?@|marketing@|"
    r"updates?@|notifications?@)",
    re.IGNORECASE,
)


def _is_empty_tool_output(name: str, raw: str) -> bool:
    """True si el tool devolvió un shape que semánticamente significa
    'no hay nada'. Empty-test por tool:

    - ``gmail_recent`` → ``threads == []`` AND ``unread_count == 0``.
    - ``calendar_ahead`` → lista vacía.
    - ``reminders_due`` → ``dated == []`` AND ``undated == []``.
    - ``credit_cards_summary`` → lista vacía (sin xlsx en /Finances).
    - ``finance_summary`` / ``weather`` → siempre ``False`` (tienen output
      útil aun cuando los números son todos cero — un mes sin gastos es
      data válida, no ausencia de data).

    Malformed JSON (no debería pasar — los tools escriben shapes bien
    definidos) devuelve ``False`` conservativamente: si no puedo parsear,
    no asumo que es empty; dejo que el LLM decida con el string crudo.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if name == "gmail_recent":
        if not isinstance(data, dict):
            return False
        threads = data.get("threads") or []
        try:
            unread = int(data.get("unread_count") or 0)
        except (TypeError, ValueError):
            unread = 0
        return not threads and unread == 0
    if name == "calendar_ahead":
        return isinstance(data, list) and not data
    if name == "credit_cards_summary":
        return isinstance(data, list) and not data
    if name == "reminders_due":
        if not isinstance(data, dict):
            return False
        return (
            not (data.get("dated") or [])
            and not (data.get("undated") or [])
        )
    if name == "drive_search":
        # Empty = no files AND an error OR no files without an error (both
        # mean "nothing to show"). Auth errors also count as empty so the
        # fallback CONTEXTO-preserve branch kicks in.
        if not isinstance(data, dict):
            return False
        return not (data.get("files") or [])
    if name == "whatsapp_pending":
        # Shape: list of chat dicts. Empty list = no chats waiting for reply.
        return isinstance(data, list) and not data
    if name == "whatsapp_search":
        # Shape: dict con `messages` list. Empty list = no matches encontrados
        # en el corpus WA. Igual que drive_search: errores también cuentan
        # como "empty" para que el fallback CONTEXTO-preserve kick-in.
        if not isinstance(data, dict):
            return False
        return not (data.get("messages") or [])
    return False


def _format_forced_tool_output(name: str, raw: str) -> str:
    """Render one forced-tool result as markdown for the CONTEXTO block.

    Never raises — malformed / non-JSON input falls through to a raw
    passthrough under the tool's Spanish section header. Tool name is
    NEVER leaked for known tools (prevents `[[calendar_ahead]]` citation
    artifacts observed in production). Unknown tools include the name
    as a disambiguator since we can't invent a friendly label.
    """
    if name == "reminders_due":
        return _format_reminders_block(raw)
    if name == "calendar_ahead":
        return _format_calendar_block(raw)
    if name == "gmail_recent":
        return _format_gmail_block(raw)
    if name == "finance_summary":
        return _format_finance_block(raw)
    if name == "credit_cards_summary":
        return _format_cards_block(raw)
    if name == "weather":
        return _format_weather_block(raw)
    if name == "drive_search":
        return _format_drive_block(raw)
    if name == "whatsapp_pending":
        return _format_whatsapp_block(raw)
    if name == "whatsapp_search":
        return _format_whatsapp_search_block(raw)
    # Unknown tool → keep raw JSON available but wrap in a labeled
    # section so it doesn't merge visually with the next block.
    return f"### Datos ({name})\n{raw}\n"


def _format_reminders_block(raw: str) -> str:
    """Render Apple Reminders JSON as two sub-sections: 'Con fecha' and
    'Sin fecha'. Dedupes by (name, due) — AppleScript occasionally returns
    the same reminder twice across recurring-instance boundaries.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Recordatorios\n{raw}\n"
    if not isinstance(data, dict):
        return f"### Recordatorios\n{raw}\n"

    def _dedup(items):
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            nm = str(it.get("name", "")).strip()
            du = str(it.get("due", "")).strip()
            if not nm:
                continue
            key = (nm, du)
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    dated = _dedup(data.get("dated"))
    undated = _dedup(data.get("undated"))

    if not dated and not undated:
        return "### Recordatorios\n_Sin recordatorios pendientes._\n"

    # Header con counts explícitos. Pre-fix qwen2.5:7b a veces contaba
    # mal ("tres tareas para llamar al dentista" cuando había una sola);
    # emitir N/M literal en el header le da al modelo un ancla numérica
    # para no alucinar totales. Medido 2026-04-23 en scratch_eval.
    header_bits: list[str] = []
    if dated:
        header_bits.append(f"{len(dated)} con fecha")
    if undated:
        header_bits.append(f"{len(undated)} sin fecha")
    header = f"### Recordatorios ({', '.join(header_bits)})"
    lines: list[str] = [header]
    if dated:
        lines.append("**Con fecha:**")
        for it in dated:
            nm = str(it.get("name", "")).strip()
            due = str(it.get("due", "")).strip()
            bucket = str(it.get("bucket", "")).strip().lower()
            date_part, _, time_part = due.partition("T")
            time_str = time_part[:5] if time_part else ""
            stamp = f"{date_part} {time_str}".strip() if date_part else ""
            tag_es = _BUCKET_ES.get(bucket, "")
            # Only prefix a tag for overdue/today (urgency signal).
            # Upcoming is the default forward-looking state; leaving it
            # unmarked reduces noise without losing information.
            prefix = f"[{tag_es}] " if tag_es in ("vencido", "hoy") else ""
            body = f"{stamp} · {nm}" if stamp else nm
            lines.append(f"- {prefix}{body}")
    if undated:
        lines.append("**Sin fecha:**")
        for it in undated:
            nm = str(it.get("name", "")).strip()
            lines.append(f"- {nm}")
    return "\n".join(lines) + "\n"


def _translate_calendar_label(date_label: str, time_range: str = "") -> str:
    """Normaliza un date_label de icalBuddy a español y, cuando es un
    label relativo, ANCLA el tag absoluto entre corchetes para que el LLM
    no confunda "tomorrow" con "hoy". Retorna string vacío si no hay
    label útil.
    """
    if not date_label:
        return ""
    low = date_label.strip().lower()
    if low in _CALENDAR_LABEL_ES:
        # Tag explícito en MAYÚSCULAS — llama atención del LLM y se
        # mantiene incluso si el modelo lo cita literal.
        return _CALENDAR_LABEL_ES[low]
    # icalBuddy también puede devolver "Wednesday, April 30, 2026" o un
    # YYYY-MM-DD plano para fechas más lejanas. No reformateamos —
    # devolvemos como está; el LLM puede leerlo igual.
    return date_label.strip()


def _format_calendar_block(raw: str, now: datetime | None = None) -> str:
    """Render calendar events as a bulleted list. Does NOT dedup — the
    same recurring event across multiple days must show up N times so the
    user sees 'cumpleaños de Astor' twice if it falls on two rendered
    dates.

    Header incluye la fecha de "hoy" (DD/MM/YYYY + día de la semana en
    español) para que el LLM tenga anclaje temporal explícito al
    interpretar items con date_label relativo. Pre-2026-04-28 sólo se
    emitía "### Calendario (N eventos)" sin fecha → el LLM no podía
    distinguir "hoy" en la pregunta del user vs "today/tomorrow" en los
    items, y caía en respuestas crudas tipo "- Psiquiatra (mañana)" para
    "qué tengo hoy".
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Calendario\n{raw}\n"
    if not isinstance(data, list):
        return f"### Calendario\n{raw}\n"

    # Día actual para el header — usamos el now inyectado (testeable) o
    # datetime.now() del huso local. _DAYS_ES traduce el nombre del
    # weekday inglés (datetime.strftime("%A")) a español rioplatense.
    if now is None:
        now = datetime.now()
    _DAYS_ES = {
        "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miércoles",
        "Thursday": "jueves", "Friday": "viernes", "Saturday": "sábado",
        "Sunday": "domingo",
    }
    today_str = f"{_DAYS_ES.get(now.strftime('%A'), now.strftime('%A').lower())} {now.strftime('%d/%m/%Y')}"

    if not data:
        return f"### Calendario (hoy: {today_str})\n_Sin eventos en el horizonte._\n"

    # Count explícito en header (ver comentario en _format_reminders_block).
    valid_events = [ev for ev in data if isinstance(ev, dict) and str(ev.get("title", "")).strip()]
    header = (
        f"### Calendario (hoy: {today_str} · "
        f"{len(valid_events)} evento{'s' if len(valid_events) != 1 else ''})"
    )
    lines: list[str] = [header]
    has_today = False
    for ev in valid_events:
        title = str(ev.get("title", "")).strip()
        date_label = _translate_calendar_label(
            str(ev.get("date_label", "")).strip(),
            str(ev.get("time_range", "")).strip(),
        )
        time_range = str(ev.get("time_range", "")).strip()
        if date_label == "HOY":
            has_today = True
        parts: list[str] = []
        if date_label:
            parts.append(date_label)
        if time_range:
            parts.append(time_range)
        prefix = " ".join(parts)
        if prefix:
            lines.append(f"- {prefix} · {title}")
        else:
            lines.append(f"- {title}")
    # Hint explícito al LLM cuando la lista NO tiene eventos para HOY.
    # Sin esto, el LLM ve "- MAÑANA · Psiquiatra" y lo cita literal sin
    # aclarar que la respuesta a "qué tengo hoy" es "nada hoy".
    if valid_events and not has_today:
        lines.append("_(no hay eventos hoy; los items de arriba son posteriores)_")
    return "\n".join(lines) + "\n"


def _classify_gmail_thread(frm: str, subj: str) -> str:
    """Clasifica un thread de gmail en una bucket de ruido.

    Returns "ci", "security", "marketing", "automated", o "personal".
    Personal es el bucket default — los mails que NO matchean ninguno de
    los patrones de ruido son los que importan al user.
    """
    full = f"{frm} {subj}".strip()
    # Order matters: CI antes que automated porque CI es más específico.
    if _GMAIL_GITHUB_BRACKET_RE.search(subj) or _GMAIL_CI_RE.search(full):
        return "ci"
    if _GMAIL_SECURITY_RE.search(full):
        return "security"
    if _GMAIL_MARKETING_RE.search(full):
        return "marketing"
    if _GMAIL_NOREPLY_RE.search(frm):
        return "automated"
    return "personal"


def _format_gmail_block(raw: str) -> str:
    """Render Gmail evidence as 'Mails' section with awaiting-reply and
    starred threads as bullets. Keeps the tool name out of the output.

    2026-04-28 wave-8: agrupar mails de noise (CI, security alerts, marketing,
    automated notifications) en summary lines. El user pidió "qué mails
    pendientes tengo" y antes recibía 4 líneas separadas con commit SHAs +
    `[jagoff/rag-obsidian]` brackets. Ahora resume "4 mails de CI: 2 fail,
    1 ok, 1 cancelled" + lista solo los mails personales.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Mails\n{raw}\n"
    if not isinstance(data, dict):
        return f"### Mails\n{raw}\n"
    threads = data.get("threads") or []
    unread = int(data.get("unread_count") or 0)
    if not threads and unread == 0:
        return "### Mails\n_Sin mails pendientes._\n"
    # Count explícito en header (ver comentario en _format_reminders_block).
    valid_threads = [t for t in threads if isinstance(t, dict)]
    header_bits: list[str] = []
    if valid_threads:
        header_bits.append(f"{len(valid_threads)} hilo{'s' if len(valid_threads) != 1 else ''}")
    if unread:
        header_bits.append(f"{unread} no leído{'s' if unread != 1 else ''}")
    header = f"### Mails ({', '.join(header_bits)})" if header_bits else "### Mails"
    lines = [header]

    # Agrupar threads por bucket. Personal va listado uno por uno; los
    # demás se resumen en una sola línea con count.
    buckets: dict[str, list[dict]] = {
        "ci": [], "security": [], "marketing": [],
        "automated": [], "personal": [],
    }
    for t in valid_threads:
        frm = str(t.get("from", "")).strip()
        subj = str(t.get("subject", "")).strip()
        bucket = _classify_gmail_thread(frm, subj)
        buckets[bucket].append(t)

    # Resumen de noise (siempre primero — los grupos compactos arriba).
    noise_labels = {
        "ci": "CI/builds",
        "security": "alertas de seguridad",
        "marketing": "marketing/newsletters",
        "automated": "notificaciones automáticas",
    }
    for bucket_name in ("ci", "security", "marketing", "automated"):
        items = buckets[bucket_name]
        if not items:
            continue
        n = len(items)
        label = noise_labels[bucket_name]
        # Para CI agregar el desglose por outcome (failed/passed/cancelled).
        if bucket_name == "ci":
            outcomes = {"failed": 0, "passed": 0, "cancelled": 0, "other": 0}
            for it in items:
                s = str(it.get("subject", "")).lower()
                if "failed" in s or "fail:" in s or " fail" in s:
                    outcomes["failed"] += 1
                elif "passed" in s or "succeeded" in s or "success" in s:
                    outcomes["passed"] += 1
                elif "cancelled" in s or "skipped" in s:
                    outcomes["cancelled"] += 1
                else:
                    outcomes["other"] += 1
            outcome_bits = [f"{v} {k}" for k, v in outcomes.items() if v > 0]
            lines.append(f"- {n} {label} ({', '.join(outcome_bits)})")
        else:
            lines.append(f"- {n} {label}")

    # Mails personales: uno por línea con tag de bucket si corresponde.
    for t in buckets["personal"]:
        frm = str(t.get("from", "")).strip()
        subj = str(t.get("subject", "")).strip()
        kind = str(t.get("kind", "")).strip()
        tag = {
            "awaiting_reply": "esperando respuesta",
            "starred": "starred",
            "recent": "",
        }.get(kind, kind)
        tag_str = f" [{tag}]" if tag else ""
        lines.append(f"- {frm} · {subj}{tag_str}")
    return "\n".join(lines) + "\n"


def _format_finance_block(raw: str) -> str:
    """Finance summary is a dict with month, totals, top categories. Render
    under 'Gastos' section. Passthrough pretty JSON if shape is unexpected
    — the LLM handles a raw dict fine when there's a section header.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Gastos\n{raw}\n"
    if not isinstance(data, dict) or not data:
        return "### Gastos\n_Sin datos financieros._\n"
    try:
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        pretty = raw
    return f"### Gastos\n```json\n{pretty}\n```\n"


def _format_weather_block(raw: str) -> str:
    """Render weather JSON output as human-readable markdown for the LLM.

    Eval 2026-04-28 BUG-2a: el LLM no mencionaba la ciudad porque la
    location quedaba enterrada en el JSON. Fix: parsear y exponer un
    text_summary "Ciudad: descripción (temp°C)" prominente.
    """
    body = (raw or "").strip()
    if not body:
        return "### Clima\n_Sin datos del clima._\n"
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        # No es JSON parseable, dejar el raw como antes.
        return f"### Clima\n{body}\n"

    if not isinstance(data, dict):
        return f"### Clima\n{body}\n"

    # Si el output ya tiene text_summary (post-fix), usarlo como header.
    summary = data.get("text_summary")
    if not summary and data.get("current"):
        loc = data.get("location", "Ubicación no especificada")
        cur = data["current"]
        desc = cur.get("description", "Sin datos")
        temp = cur.get("temp_C") or cur.get("temp")
        summary = f"{loc}: {desc} ({temp}°C)" if temp else f"{loc}: {desc}"

    # Render: summary (prominente) + JSON full por si el LLM quiere los días.
    out = "### Clima\n"
    if summary:
        out += f"{summary}\n"
    out += body  # mantiene el JSON crudo para el LLM si necesita días
    out += "\n"
    return out


def _format_cards_block(raw: str) -> str:
    """Credit cards summary es una lista de dicts (uno por tarjeta). Renderea
    una sub-sección por tarjeta con saldo a pagar (ARS/USD), vencimiento,
    cierre y top 3 consumos. Es passthrough en formato humano para que el
    LLM cite los números directamente sin hacer su propia matemática.
    """
    try:
        cards = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Tarjetas\n{raw}\n"
    if not isinstance(cards, list) or not cards:
        return "### Tarjetas\n_Sin resúmenes de tarjeta disponibles._\n"

    def _fmt_ars(n: float | None) -> str:
        if n is None:
            return "—"
        v = int(round(abs(n)))
        s = f"{v:,}".replace(",", ".")
        return f"${s}"

    def _fmt_usd(n: float | None) -> str:
        if n is None:
            return "—"
        return f"U$S{abs(n):,.2f}"

    lines = ["### Tarjetas"]
    for c in cards:
        if not isinstance(c, dict):
            continue
        brand = c.get("brand") or "Tarjeta"
        last4 = c.get("last4") or "----"
        title = f"{brand} ····{last4}"
        lines.append(f"- **{title}**")
        # Saldo a pagar
        total_bits = []
        if c.get("total_ars"):
            total_bits.append(_fmt_ars(c["total_ars"]))
        if c.get("total_usd"):
            total_bits.append(_fmt_usd(c["total_usd"]))
        if total_bits:
            lines.append(f"  - Total a pagar: {' + '.join(total_bits)}")
        # Mínimo (solo si difiere del total — banks suelen igualarlos)
        min_bits = []
        if c.get("minimum_ars") and c.get("minimum_ars") != c.get("total_ars"):
            min_bits.append(_fmt_ars(c["minimum_ars"]))
        if c.get("minimum_usd") and c.get("minimum_usd") != c.get("total_usd"):
            min_bits.append(_fmt_usd(c["minimum_usd"]))
        if min_bits:
            lines.append(f"  - Mínimo: {' + '.join(min_bits)}")
        # Fechas: vencimiento es la accionable
        if c.get("due_date"):
            lines.append(f"  - Vence: {c['due_date']}")
        if c.get("closing_date"):
            lines.append(f"  - Cierre: {c['closing_date']}")
        if c.get("next_due_date"):
            lines.append(f"  - Próximo venc.: {c['next_due_date']}")
        # Top consumos: ARS + USD (3 + 2 max para no inundar)
        ars_top = c.get("top_purchases_ars") or []
        usd_top = c.get("top_purchases_usd") or []
        if ars_top or usd_top:
            lines.append("  - Top consumos:")
            for p in ars_top[:3]:
                desc = p.get("description") or "—"
                lines.append(f"    - {desc} · {_fmt_ars(p.get('amount'))}")
            for p in usd_top[:2]:
                desc = p.get("description") or "—"
                lines.append(f"    - {desc} · {_fmt_usd(p.get('amount'))}")
    return "\n".join(lines) + "\n"


def _format_drive_block(raw: str) -> str:
    """Render `drive_search` output as a `### Google Drive` section.

    Shape esperado del tool (web/tools.drive_search → rag._agent_tool_drive_search):
        {tokens: [...], query_used: "a b c", files: [{name, mime_label,
         modified, link, body}], error?: "..."}

    Por archivo: header con nombre + tipo + fecha + link (un solo bullet),
    seguido de un bloque de body entre líneas en blanco. Cap defensivo del
    body a 2500 chars acá (aunque el helper ya lo capea a 3500) por si
    algún día elevamos el cap en la tool pero no queremos explotar el
    CONTEXTO acá. Auth / API errors se renderizan como "_Sin resultados
    (motivo: ...)._" para que el LLM lo comunique al user honestamente
    en vez de inventar contenido.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### Google Drive\n{raw}\n"
    if not isinstance(data, dict):
        return f"### Google Drive\n{raw}\n"

    files = data.get("files") or []
    tokens = data.get("tokens") or []
    err = data.get("error") or ""
    query_used = data.get("query_used") or ""

    header = "### Google Drive"
    if query_used:
        header = f"{header} (búsqueda: {query_used})"

    if not files:
        reason_bits: list[str] = []
        if err == "no_google_credentials":
            reason_bits.append("auth de Drive no configurada")
        elif err.startswith("search_failed"):
            reason_bits.append("falló la API de Drive")
        elif err == "query vacía después de filtrar stopwords":
            reason_bits.append("la pregunta no tenía keywords buscables")
        elif err:
            reason_bits.append(err)
        else:
            reason_bits.append(
                f"ningún archivo matchea {tokens!r}" if tokens
                else "ningún archivo matchea la búsqueda"
            )
        return f"{header}\n_Sin resultados ({'; '.join(reason_bits)})._\n"

    # Defensive body cap (ver docstring).
    BODY_CAP = 2500

    lines: list[str] = [header]
    for f in files:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name", "")).strip() or "(sin nombre)"
        mime = str(f.get("mime_label", "")).strip() or "archivo"
        modified = str(f.get("modified", "")).strip()
        link = str(f.get("link", "")).strip()
        date_part = modified.split("T")[0] if modified else ""
        bits: list[str] = [f"**{name}**", f"({mime})"]
        if date_part:
            bits.append(f"· modificado {date_part}")
        if link:
            bits.append(f"· [abrir]({link})")
        lines.append("- " + " ".join(bits))
        body = str(f.get("body", "")).strip()
        if body:
            lines.append("")
            lines.append(body[:BODY_CAP])
            lines.append("")
    return "\n".join(lines) + "\n"


def _format_whatsapp_block(raw: str) -> str:
    """Render `whatsapp_pending` output as a `### WhatsApp` section.

    Shape esperado: list of `{jid, name, last_snippet, hours_waiting}`
    (ver `_fetch_whatsapp_unreplied`). Formatea cada chat como bullet
    con contacto bold, tiempo esperando humanizado ("hace 3h" / "hace
    2d"), y el snippet del último mensaje. Dedupe por (jid, snippet)
    porque el shape del fetcher ya dedupea por jid — redundante pero
    defensivo ante shape drift.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### WhatsApp\n{raw}\n"
    if not isinstance(data, list):
        return f"### WhatsApp\n{raw}\n"
    if not data:
        return "### WhatsApp\n_Sin chats esperando tu respuesta._\n"

    chats = [c for c in data if isinstance(c, dict) and str(c.get("name", "")).strip()]
    header = f"### WhatsApp ({len(chats)} chat{'s' if len(chats) != 1 else ''} esperando respuesta)"
    lines: list[str] = [header]
    for chat in chats:
        name = str(chat.get("name", "")).strip()
        snippet = str(chat.get("last_snippet", "")).strip()
        try:
            hours = float(chat.get("hours_waiting") or 0)
        except (TypeError, ValueError):
            hours = 0.0
        # Humanize wait time: <24h → "Xh", ≥24h → "Xd".
        if hours < 1:
            age = "recién"
        elif hours < 24:
            age = f"hace {int(hours)}h"
        else:
            age = f"hace {int(hours / 24)}d"
        line = f"- **{name}** ({age})"
        if snippet:
            line += f": {snippet}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _format_whatsapp_search_block(raw: str) -> str:
    """Render `whatsapp_search` output as a `### WhatsApp` section with
    one bullet per matched message.

    Shape esperado: ``{query, contact_filter, messages: [{jid, contact,
    ts, who, text, score}], warning?, error?}`` (ver
    `_agent_tool_whatsapp_search`).

    Cada bullet sigue el shape ``- [<contacto> · <fecha>] <snippet>``;
    si el chunk es outbound (sender == "yo") prefija ``yo →`` para que
    el LLM diferencie quién dijo qué. Snippet se renderea capado
    defensivamente a 300 chars (el tool ya lo capa a 400, pero el LLM
    digestivo prefiere bullets más cortos cuando hay 5–8 hits).

    Empty / errored results muestran un mensaje explícito en vez de
    raw JSON — same pattern que `_format_drive_block` para que el LLM
    no invente conversaciones cuando el corpus no devolvió match.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"### WhatsApp\n{raw}\n"
    if not isinstance(data, dict):
        return f"### WhatsApp\n{raw}\n"

    messages = data.get("messages") or []
    contact_filter = data.get("contact_filter")
    err = data.get("error") or ""
    warning = data.get("warning") or ""

    # Empty path — surface why so the LLM can tell the user.
    if not messages:
        reason_bits: list[str] = []
        if err == "query vacía":
            reason_bits.append("la pregunta vino vacía")
        elif err.startswith("retrieve_failed"):
            reason_bits.append("falló el retrieval")
        elif err:
            reason_bits.append(err)
        else:
            if contact_filter:
                reason_bits.append(f"sin matches para {contact_filter!r}")
            else:
                reason_bits.append("sin matches en el corpus de WhatsApp")
        if warning and warning not in reason_bits:
            reason_bits.append(warning)
        return f"### WhatsApp\n_Sin resultados ({'; '.join(reason_bits)})._\n"

    valid = [m for m in messages if isinstance(m, dict) and str(m.get("text", "")).strip()]
    header_bits: list[str] = [f"{len(valid)} mensaje{'s' if len(valid) != 1 else ''}"]
    if contact_filter:
        header_bits.append(f"contacto: {contact_filter}")
    header = f"### WhatsApp ({', '.join(header_bits)})"

    lines: list[str] = [header]
    if warning:
        lines.append(f"_Aviso: {warning}_")

    SNIPPET_CAP = 300  # Defensive — tool capa a 400, pero bullets cortos = LLM más limpio.
    for m in valid:
        contact = str(m.get("contact", "")).strip() or "(sin contacto)"
        ts = str(m.get("ts", "")).strip()
        who = str(m.get("who", "")).strip().lower()
        text = str(m.get("text", "")).strip()
        # Date-only — el LLM se confunde con timestamps largos en
        # bullets; la hora exacta vive en el JSON crudo si lo necesita.
        date_part = ts.split("T")[0] if ts else ""
        prefix = f"[{contact}"
        if date_part:
            prefix += f" · {date_part}"
        prefix += "]"
        # Outbound = lo dijo el user; importante distinguirlo para que
        # el LLM no diga "Juan te dijo X" cuando el "X" lo dijiste vos.
        speaker_marker = "yo → " if who == "outbound" else ""
        snippet = text[:SNIPPET_CAP].replace("\n", " ⏎ ")
        lines.append(f"- {prefix} {speaker_marker}{snippet}")
    return "\n".join(lines) + "\n"


__all__ = [
    "_BUCKET_ES",
    "_CALENDAR_LABEL_ES",
    "_GMAIL_GITHUB_BRACKET_RE",
    "_GMAIL_CI_RE",
    "_GMAIL_SECURITY_RE",
    "_GMAIL_MARKETING_RE",
    "_GMAIL_NOREPLY_RE",
    "_is_empty_tool_output",
    "_format_forced_tool_output",
    "_format_reminders_block",
    "_translate_calendar_label",
    "_format_calendar_block",
    "_classify_gmail_thread",
    "_format_gmail_block",
    "_format_finance_block",
    "_format_weather_block",
    "_format_cards_block",
    "_format_drive_block",
    "_format_whatsapp_block",
    "_format_whatsapp_search_block",
]
