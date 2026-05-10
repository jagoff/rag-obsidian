"""Cross-source aggregations (topics, time overlaps, gaps) — extracted from
rag/today_correlator.py 2026-05-09.

Tres correlators que pre-computan patrones cross-source para el brief LLM:

1. **Topics** — tokens (lowercase ≥4 chars no-stopword) que aparecen en ≥2
   buckets (gmail, whatsapp, youtube, calendar, drive, bookmarks, notas,
   preguntas). Output: list[{topic, sources, sources_count}].

2. **Time overlaps** — items de fuentes distintas que caen en el mismo
   bucket de tiempo (±30min) Y comparten ≥2 tokens. Captura "gmail
   recibido 13:45 + calendar event 14:00" como cross-reference probable.
   Output: list[{time, items, shared_tokens}].

3. **Gaps** — loose ends sin slot agendado. MVP: WhatsApp unreplied ≥24h
   donde la persona NO aparece en tomorrow_calendar. Indica "no le
   agendaste tiempo". Filtra grupos WA (`_looks_like_wa_group`).
   Output: list[{kind, person, hours_waiting, snippet, context}].

## Compartido con todo el archivo

`_STOPWORDS`, `_TOKEN_RE`, `_tokenize` viven acá porque los 3 correlators
los usan. Re-exportados desde `rag.today_correlator` para back-compat.

## Lazy imports para preservar monkeypatches

`_is_self_notification` y `_canonicalize_name` viven en
`rag.today_people_correlator` (split anterior). Los importamos lazy
adentro de las funciones que los necesitan para evitar circular import
y preservar monkeypatches a `rag.today_correlator.X`.
"""
from __future__ import annotations

import re
from collections import defaultdict

__all__ = [
    "_STOPWORDS",
    "_TOKEN_RE",
    "_tokenize",
    "_topic_source_texts",
    "_correlate_topics",
    "_TIME_RE",
    "_parse_time_to_minutes",
    "_correlate_time_overlaps",
    "_correlate_gaps",
    "_WA_GROUP_MARKERS",
    "_looks_like_wa_group",
]


# Stopwords ES + EN comunes que dominan tf y NO son señales de tema.
_STOPWORDS = frozenset({
    # Español
    "de", "la", "el", "y", "en", "del", "al", "para", "con", "que", "los",
    "las", "un", "una", "es", "se", "no", "lo", "te", "tu", "tus", "mi",
    "su", "ese", "este", "esta", "como", "más", "fue", "por", "pero", "le",
    "ya", "esa", "sus", "sin", "todo", "todos", "ser", "han", "ha", "muy",
    "yo", "vos", "te", "qué", "cómo", "dónde", "cuál", "porque", "porqué",
    "hoy", "ayer", "mañana", "tarde", "noche", "ahora", "luego", "después",
    "esto", "eso", "aquí", "allí", "donde", "cuando", "soy", "sos", "está",
    "son", "está", "estuvimos", "estuviste", "estás", "tener", "tener",
    "hace", "hacer", "hizo", "hago", "hace", "voy", "vas", "van", "vamos",
    "puede", "puedo", "podés", "podríamos", "haber", "hay", "tras", "según",
    # Inglés común que aparece en titles
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "were", "have", "has", "had", "will", "would", "could", "should",
    "you", "your", "yours", "their", "they", "them", "what", "when",
    "where", "which", "who", "whom", "why", "how", "than", "then",
    "there", "here", "into", "onto", "over", "under", "about", "after",
    "before", "between", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "same", "too", "very", "can", "just", "now",
    # Tokens muy genéricos que aparecen en múltiples fuentes pero no son tema
    "msg", "msgs", "subject", "from", "title", "snippet",
    "youtube", "video", "watch", "url", "link", "post", "stream",
    "today", "yesterday", "tomorrow",
    "github", "com", "https", "http", "www",
    # CI / GitHub notifications mail noise — los CI failure emails generan
    # tokens repetidos como "failed", "master", "jagoff", "ferrari" que
    # dominan los topics cuando hay 5-10 mails de github en un día. NO son
    # temas de trabajo, son ruido del feed automático.
    "failed", "master", "jagoff", "ferrari", "rag-obsidian", "obsidian-rag",
    "passed", "branch", "commit", "build", "ci", "pull", "push", "actions",
    "workflow", "request", "issue", "pull_request", "released", "deploy",
    "merge", "merged", "closed", "opened", "review", "ready",
    # Apple Mail / Apple Reminders noise
    "apple", "icloud", "mailbox",
    # Genéricos que no aportan tema
    "hola", "saludos", "atentamente", "atte", "cordial", "cordiales",
    "regards", "thanks", "gracias", "buen", "buenos", "buenas",
})


_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ][\w-]{3,}", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    """Lowercase tokens ≥4 chars, sin stopwords. Devuelve set para
    dedup intra-bucket — un token que aparece 5 veces en gmail cuenta
    una sola vez para el cross-source matching.
    """
    if not text:
        return set()
    tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 4}


# Buckets considerados para topic correlation. Cada uno se reduce a un
# "texto representativo" antes de tokenizar.
def _topic_source_texts(today_ev: dict, extras: dict) -> dict[str, list[str]]:
    """Devuelve {source_label: [text_chunks...]} para tokenización. Los
    labels son cortos para que el render del prompt diga "aparece en
    gmail+youtube+notas" en vez de "gmail_today+youtube_today+recent_notes".

    Los mails de self-notification (github bot, etc.) NO contribuyen a
    topics — saturan los tokens con "failed/master/jagoff/ferrari" cuando
    hay 5-10 mails de CI por día.
    """
    from rag.today_people_correlator import _is_self_notification

    return {
        "gmail": [
            (m.get("subject") or "") + " " + (m.get("snippet") or "")
            for m in (extras.get("gmail_today") or [])
            if not _is_self_notification(m.get("from") or m.get("sender") or "")
        ],
        "whatsapp": [
            (w.get("last_snippet") or "")
            for w in (extras.get("whatsapp_today") or [])
        ],
        "youtube": [
            (v.get("title") or "")
            for v in (extras.get("youtube_today") or [])
            + (extras.get("youtube_recent") or [])
            + (extras.get("youtube_watched") or [])
        ],
        "calendar": [
            (c.get("title") or "")
            for c in (extras.get("calendar_today") or [])
            + (extras.get("tomorrow_calendar") or [])
        ],
        "drive": [
            (d.get("name") or d.get("title") or "")
            for d in (extras.get("drive_recent") or [])
        ],
        "bookmarks": [
            (b.get("name") or "")
            for b in (extras.get("chrome_bookmarks") or [])
        ],
        "notas": [
            (n.get("title") or "") + " " + (n.get("snippet") or "")
            for n in (today_ev.get("recent_notes") or [])
        ],
        "preguntas": [
            (q.get("q") or "")
            for q in (today_ev.get("low_conf_queries") or [])
        ],
    }


def _correlate_topics(today_ev: dict, extras: dict) -> list[dict]:
    """Detect tokens (lowercase ≥4 chars no-stopword) that appear in
    ≥2 source buckets. Returns top-N sorted by sources_count.
    """
    source_texts = _topic_source_texts(today_ev, extras)
    source_tokens: dict[str, set[str]] = {}
    for source, texts in source_texts.items():
        bag: set[str] = set()
        for t in texts:
            bag |= _tokenize(t)
        if bag:
            source_tokens[source] = bag

    token_sources: dict[str, set[str]] = defaultdict(set)
    for source, tokens in source_tokens.items():
        for tok in tokens:
            token_sources[tok].add(source)

    out: list[dict] = []
    for tok, sources in token_sources.items():
        if len(sources) < 2:
            continue
        out.append({
            "topic": tok,
            "sources": sorted(sources),
            "sources_count": len(sources),
        })
    out.sort(key=lambda t: (-t["sources_count"], t["topic"]))
    return out[:12]


# ── Time overlap correlation ──────────────────────────────────────────────


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


def _parse_time_to_minutes(time_str: str) -> int | None:
    """Parse "10:00" / "14:30 PM" / "10:00–11:00" → minutes since midnight
    of the FIRST time mentioned. None si no hay hora parseable.
    """
    if not time_str:
        return None
    m = _TIME_RE.search(time_str)
    if not m:
        return None
    h = int(m.group(1))
    mm = int(m.group(2))
    # AM/PM no afecta — calendars en macOS Calendar.app suelen venir
    # ya en formato 24hs. Si está PM y el hour es <12, sumar 12.
    if "PM" in time_str.upper() and h < 12:
        h += 12
    elif "AM" in time_str.upper() and h == 12:
        h = 0
    return h * 60 + mm


def _correlate_time_overlaps(
    today_ev: dict, extras: dict, window_min: int = 30,
) -> list[dict]:
    """Detect events from distinct sources that fall in the same time
    bucket AND share tokens. Useful para detectar:
      - gmail "Reunión 14hs confirmada" recibido 13:45 + calendar event
        "Reunión 14:00–15:00" → match probable
      - calendar "Demo cliente 14:00" + gmail "Re: Demo" recibido 13:30
        → cross-reference
    Cuando 2 items coinciden en time + tienen ≥2 tokens en común
    (post-stopwords, ≥4 chars), es overlap.

    Returns: list of {time, items: [{source, label, snippet, ts_minutes}],
    shared_tokens} sorted by time asc.

    `window_min` (default 30): items dentro de ±N min se consideran
    "mismo bucket". 30 captura "received 13:45 vs event 14:00" pero no
    junta items de horas distintas.
    """
    from rag.today_people_correlator import _is_self_notification

    items: list[dict] = []

    # Gmail today: timestamp del internal_date_ms.
    # `internal_date_ms` viene en epoch UTC desde Gmail API. `fromtimestamp()`
    # sin `tz=` lo convierte a LOCAL time naive. Esa decisión es intencional
    # — los calendar events vienen también en local time (icalBuddy en macOS
    # devuelve la hora local del usuario), entonces compararlos en local
    # naive ES correcto Y consistente. NO usar UTC acá porque entonces un
    # gmail "Reunión 14hs" recibido a las 14:00 ART (17:00 UTC) NO matchearía
    # con un calendar event "14:00–15:00 ART" (que es local).
    # El bug real solo aparece si la Mac está en una timezone que NO es la
    # del user (típicamente CI corriendo en UTC). Para CI tests, los tests
    # usan `_ms_at_local_time()` helper que está en la misma timezone que
    # `fromtimestamp()` → consistente.
    from datetime import datetime as _dt
    for m in (extras.get("gmail_today") or [])[:20]:
        if _is_self_notification(m.get("from") or m.get("sender") or ""):
            continue
        ts_ms = m.get("internal_date_ms")
        if not ts_ms:
            continue
        try:
            t = _dt.fromtimestamp(int(ts_ms) / 1000)
            ts_minutes = t.hour * 60 + t.minute
        except (ValueError, OSError, OverflowError):
            continue
        text = (m.get("subject") or "") + " " + (m.get("snippet") or "")
        items.append({
            "source": "gmail",
            "label": (m.get("subject") or "")[:80],
            "snippet": (m.get("snippet") or "")[:80],
            "ts_minutes": ts_minutes,
            "time_str": f"{t.hour:02d}:{t.minute:02d}",
            "tokens": _tokenize(text),
        })

    # Calendar today: time_range OR start time
    for c in (extras.get("calendar_today") or [])[:20]:
        time_str = c.get("start") or c.get("time_range") or ""
        ts_minutes = _parse_time_to_minutes(time_str)
        if ts_minutes is None:
            continue
        title = c.get("title") or ""
        items.append({
            "source": "calendar",
            "label": title[:80],
            "snippet": (c.get("time_range") or time_str)[:40],
            "ts_minutes": ts_minutes,
            "time_str": time_str.split("–")[0].strip()[:7],
            "tokens": _tokenize(title),
        })

    # Sort by ts_minutes para que el sliding window sea trivial
    items.sort(key=lambda x: x["ts_minutes"])

    # Para cada item, mirar si otro item de DISTINTA fuente está en
    # ±window_min Y comparte ≥2 tokens.
    overlaps: list[dict] = []
    seen_pairs: set[frozenset] = set()
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if b["ts_minutes"] - a["ts_minutes"] > window_min:
                break  # sorted, futuros son más lejanos
            if a["source"] == b["source"]:
                continue
            shared = a["tokens"] & b["tokens"]
            if len(shared) < 2:
                continue
            pair_key = frozenset({
                (a["source"], a["label"]),
                (b["source"], b["label"]),
            })
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            overlaps.append({
                "time": a["time_str"],
                "items": [
                    {"source": a["source"], "label": a["label"],
                     "snippet": a["snippet"]},
                    {"source": b["source"], "label": b["label"],
                     "snippet": b["snippet"]},
                ],
                "shared_tokens": sorted(shared),
            })

    return overlaps[:6]  # top-N para no inflar el prompt


# ── Gaps detection (loose ends sin slot) ─────────────────────────────────


def _correlate_gaps(today_ev: dict, extras: dict) -> list[dict]:
    """Detect 'loose ends' — items que requieren acción del user pero no
    tienen un slot agendado en tomorrow_calendar.

    MVP — focus en WhatsApp unreplied: chats donde el user tarda ≥24h en
    responder Y la persona NO aparece en tomorrow_calendar. Indica que
    'no le agendaste tiempo para responder o resolver'. Es accionable:
    el brief puede sugerir 'bloqueá 30min mañana para responderle a X'.

    Future expansion (NO incluido en MVP, para evitar false positives):
      - Action verbs en gmail_today / wa_today snippets ('preparar
        planilla', 'mandar X') sin slot reservado
      - Reminders overdue + no slot
      - Calendar events de hoy que NO se cerraron (último log de la
        nota matcheante < event end_time)

    Returns: list of {kind, person, hours_waiting, snippet, context}
    sorted by hours_waiting DESC (loose ends más viejos primero).
    """
    from rag.today_people_correlator import _canonicalize_name

    wa_unreplied = extras.get("whatsapp_unreplied") or []
    if not wa_unreplied:
        return []

    cal_tomorrow = extras.get("tomorrow_calendar") or []
    cal_tokens: set[str] = set()
    for c in cal_tomorrow:
        title = c.get("title") or ""
        cal_tokens.update(_tokenize(title))
        # Para nombres cortos que el _tokenize filtra (≥4 chars), agregar
        # tokens lowercase del title raw también.
        for tok in re.split(r"[\s,()/:;]+", title.lower().strip()):
            if tok and tok.isalpha() and len(tok) >= 3:
                cal_tokens.add(tok)

    gaps: list[dict] = []
    for w in wa_unreplied[:10]:
        name = w.get("name") or ""
        hours = w.get("hours_waiting")
        if hours is None:
            continue  # datos incompletos → skip
        if hours < 24:
            continue  # solo cuenta como gap si lleva ≥1 día sin respuesta
        # Filtrar grupos WA — no son personas individuales, no tiene
        # sentido sugerir "agendá tiempo para responderle" a un grupo.
        # Heurística: un chat es grupo si el nombre tiene asteriscos
        # decorativos (`*Humanidades*`), 3+ palabras (típico de
        # "Equipo X de Y"), O contiene marcadores de grupo comunes
        # ("grupo", "team", "group") en cualquier mayúscula.
        if _looks_like_wa_group(name):
            continue
        canonical = _canonicalize_name(name)
        if not canonical:
            continue
        # ¿El nombre aparece en algún título de calendar mañana?
        # Match: cualquier token del canonical name (lowercase, alpha-only)
        # presente en los tokens del calendar.
        name_tokens = set(canonical.split())
        if name_tokens & cal_tokens:
            continue  # tienen slot agendado → no es gap
        gaps.append({
            "kind": "wa_unreplied_no_slot",
            "person": name,
            "hours_waiting": float(hours),
            "snippet": (w.get("last_snippet") or "")[:100],
            "context": "no aparece en calendar de mañana",
        })

    gaps.sort(key=lambda g: -g["hours_waiting"])
    return gaps[:5]


_WA_GROUP_MARKERS = ("grupo", "group", "team", "equipo", "comunidad",
                     "channel", "canal", "broadcast")


def _looks_like_wa_group(name: str) -> bool:
    """Heurística: True si el `name` de un chat WA es un grupo (no
    persona individual). Evita que el correlator de gaps sugiera
    "agendá tiempo para responderle" a `*Humanidades* Cuarto Año`.

    Triggers:
      - Asteriscos decorativos (`*Humanidades*`, `**Equipo**`)
      - ≥4 palabras (típico de nombres de grupo: "Fifteens - Casa
        Santa Fe", "PublicCloudInfrastructure equipo")
      - Markers explícitos: "grupo", "team", "group", "equipo",
        "comunidad", "channel", "canal", "broadcast"

    Falso positivo conocido: una persona con nombre completo de 4+
    tokens (ej. "María José de la Torre Fernández"). Aceptable —
    raro y la peor consecuencia es "no detecto que esa persona es
    un loose end" (false negative tolerable).
    """
    if not name:
        return True
    if "*" in name:
        return True
    name_lower = name.lower()
    for marker in _WA_GROUP_MARKERS:
        if marker in name_lower:
            return True
    # ≥4 palabras: heurística de grupo.
    word_count = sum(
        1 for tok in re.split(r"[\s,()/:;-]+", name) if tok.strip()
    )
    if word_count >= 4:
        return True
    return False
