"""Cross-source correlator for the today brief.

The today brief receives 14+ buckets of data (gmail_today, whatsapp_today,
calendar_today, youtube_today, recent_notes, low_conf_queries, etc.). A 7B LLM
struggles to find cross-source patterns by reading these flat buckets — it
ends up writing tautological "X is related to Y because both are X" insights.

This module pre-computes the patterns BEFORE the LLM call so the prompt
can include a structured ENTIDADES CROSS-SOURCE block. The LLM then narrates
matches that already exist instead of inventing them.

Three correlations + one post-processor:

1. **People** — names that appear in ≥2 sources (gmail-from, wa-name,
   calendar-title). Canonicalized + deduplicated. Output: list of
   {name, appearances: [{source, context, snippet}], sources_count}.

2. **Topics** — keywords that appear in ≥2 source buckets, normalized
   (lowercase, stopwords removed, ≥4 chars). Output: list of
   {topic, sources, sources_count}.

3. **Time overlaps** — events at same hour bucket cross-source (e.g.
   gmail received 14:23 + calendar event 14:00 with overlapping tokens).
   Output: list of {time, items: [{source, label, snippet}], shared_tokens}.

4. **Voice normalization** (post-processing) — `normalize_voice_to_2da_persona`
   replaces 1st-person verbs ("recibí", "trabajé", "me centré") with 2nd
   person singular ("recibiste", "trabajaste", "te centraste") in the LLM
   output. The prompt forbids 1ª persona but the 7B model slips ~10% of
   the time; this is the safety net.

Used by `web/server.py:_home_compute` → passed to the prompt via
`extras["correlations"]` → rendered in `_render_today_prompt`. The voice
normalizer wraps `_generate_today_narrative`'s return value.
"""

from __future__ import annotations

import re
from collections import defaultdict


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


# Self-notification senders — emails de sistema que tienen el `display name`
# del propio user (github notifications usa el nombre del repo owner como
# display). NO son personas reales del cross-source. Filtramos antes de
# extraer name → evita "Fer F" como persona "en gmail+wa".
_SELF_NOTIFICATION_DOMAINS = (
    "@notifications.github.com",
    "@noreply.github.com",
    "@notifications.",
    "@noreply.",
    "@no-reply.",
    "@bot.",
    "@github.com",
    "@google.com",  # Google security alerts
    "@accounts.google.com",
    "@youtube.com",
    "@drive.google.com",
    "@docs.google.com",
    "@calendar.google.com",
    "@stripe.com",
    "@paypal.com",
    "@anthropic.com",  # API + plan emails (auto)
    "@openai.com",
    "@vercel.com",
    "@cloudflare.com",
)


def _is_self_notification(sender_field: str) -> bool:
    """True si el `From:` es de un dominio de notification automatizada
    (github bot, google security alerts, stripe receipts, etc). Esos
    NO son personas reales — el correlator los excluye del bucket de
    personas cross-source.

    Match es CASE-INSENSITIVE y por substring (`@notifications.github.com`
    matchea tanto `notifications@github.com` como `Foo Bar
    <notifications@github.com>`).
    """
    if not sender_field:
        return True  # vacío → tampoco es persona real
    s = sender_field.lower()
    return any(dom in s for dom in _SELF_NOTIFICATION_DOMAINS)


# Regex de email: "Nombre Apellido <email@dom.com>" o "email@dom.com"
_EMAIL_NAME_RE = re.compile(r"^\s*([^<]+?)\s*<.+?>\s*$")
_EMAIL_BARE_RE = re.compile(r"^([\w.+-]+)@[\w.-]+$")


def _extract_name_from_email(sender_field: str) -> str:
    """De "Pablo F <pablo@x.com>" devuelve "Pablo F". De "pablo@x.com"
    devuelve "pablo" (la parte local). Si no parsea, devuelve sender
    tal cual.
    """
    if not sender_field:
        return ""
    sender_field = sender_field.strip().strip('"').strip("'")
    m = _EMAIL_NAME_RE.match(sender_field)
    if m:
        name = m.group(1).strip().strip('"').strip("'")
        # Si el "nombre" es solo email (no había display name), tomar la
        # parte local del email.
        m2 = _EMAIL_BARE_RE.match(name)
        if m2:
            return m2.group(1).replace(".", " ").replace("_", " ").title()
        return name
    m = _EMAIL_BARE_RE.match(sender_field)
    if m:
        return m.group(1).replace(".", " ").replace("_", " ").title()
    return sender_field


# Heurística simple para extraer nombres propios de un title de calendar:
# secuencias de 1-3 palabras Title-Cased ("Pablo", "Pablo Fer", "María
# Fernández"). Excluye palabras comunes capitalizadas (días de la semana,
# meses, palabras como "Sync", "Reunión", "Demo").
_TITLE_TOKEN_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]*$")
_TITLE_NON_NAMES = frozenset({
    "Sync", "Reunión", "Demo", "Meeting", "Call", "Standup", "Daily",
    "Review", "Retro", "Check", "Catch", "Coffee", "Lunch", "Dinner",
    "Work", "Quick", "Brief", "Discussion", "Update", "Followup",
    "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado",
    "Domingo", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
})


def _extract_names_from_title(title: str) -> list[str]:
    """Heurística: extrae secuencias capitalizadas que probablemente son
    nombres propios. Itera token-by-token saltando palabras genéricas
    capitalizadas ("Demo", "Sync", "Reunión", días, meses).

    Ejemplos:
      "Sync con Pablo Fer" → ["Pablo Fer"]
      "Demo Pablo F" → ["Pablo F"]
      "Reunión Marina y Diego" → ["Marina", "Diego"]
      "Demo cliente" → []
      "Standup diario" → []

    No es perfecto — palabras capitalizadas que NO son nombres ("React",
    "Python", "Stripe") pasan como falsos positivos. El correlator
    filtra después por "aparece también en gmail/wa/etc" — un falso
    positivo no genera ruido si no matchea otra fuente.
    """
    if not title:
        return []
    # Tokenizar respetando puntuación común — split por whitespace +
    # caracteres como ":" "(" ")" "," "/".
    tokens = re.split(r"[\s,()/:;]+", title.strip())
    tokens = [t for t in tokens if t]

    def _is_title_token(tok: str) -> bool:
        return bool(_TITLE_TOKEN_RE.match(tok))

    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not _is_title_token(tok) or tok in _TITLE_NON_NAMES:
            i += 1
            continue
        # Encontró inicio de nombre. Extender con hasta 2 tokens más
        # Title-Cased que tampoco sean genéricos.
        end = i + 1
        while end < len(tokens) and end - i < 3:
            nxt = tokens[end]
            # Permitir iniciales solas (1 letra uppercase) como tokens
            # subsiguientes — "Pablo F" tiene "F" como inicial.
            is_initial = re.match(r"^[A-ZÁÉÍÓÚÑ]\.?$", nxt)
            if not (_is_title_token(nxt) or is_initial):
                break
            if nxt in _TITLE_NON_NAMES:
                break
            end += 1
        name = " ".join(tokens[i:end])
        out.append(name)
        i = end
    return out


def _canonicalize_name(name: str) -> str:
    """Normalize for dedup: lowercase, strip whitespace, remove emojis,
    remove diacritics (tildes), sort word tokens. Returns "" si el name
    es vacío o solo digits/símbolos.

    Token-sort permite matchear "Pablo Fernández" vs "Fernández Pablo"
    como la misma persona. Ver `_canonicals_match()` para matching
    subset (e.g. "Pablo" matches "Pablo Fer").

    NFD + diacritic strip: "María" == "Maria" == "MARÍA". Esto es
    importante para hispanohablantes — gmail puede llegar con/sin
    tildes según cómo lo escribió el remitente, y WhatsApp lo mismo
    (autocorrect, teclados móviles). Sin esta normalización, "Marina
    Pérez" en gmail y "Marina Perez" en WhatsApp se detectan como
    personas distintas — false negative del correlator.
    """
    if not name:
        return ""
    # Normalizar Unicode: NFD descompone "é" en "e" + acento combinado,
    # luego filtramos los diacríticos (categoría "Mn" = Mark, Nonspacing).
    import unicodedata
    decomposed = unicodedata.normalize("NFD", name)
    no_diacritics = "".join(
        c for c in decomposed
        if unicodedata.category(c) != "Mn"
    )
    # Strip emojis y caracteres no-alfa (excepto espacios y guiones).
    cleaned = "".join(
        c for c in no_diacritics
        if c.isalpha() or c.isspace() or c in "-'"
    ).strip().lower()
    if not cleaned or len(cleaned) < 2:
        return ""
    tokens = sorted(cleaned.split())
    return " ".join(tokens)


def _canonicals_match(a: str, b: str) -> bool:
    """Dos canonicals son la misma persona si:
      1. Son idénticos, O
      2. Uno es subset del otro Y comparten su token más distintivo.

    Casos cubiertos:
      - "pablo" matches "f pablo" (Pablo + Pablo F, subset)
      - "pablo fer" matches "fer pablo" (token-sorted ya lo hace)
      - "marina pérez" NO matches "marina suárez" (tokens distintos
         del segundo, no son subset)

    El token "más distintivo" en castellano suele ser el último (apellido)
    o el más largo. Como estamos token-sorted, es ambiguo — usamos
    "todos los tokens del más corto deben estar en el más largo".
    """
    if not a or not b:
        return False
    if a == b:
        return True
    a_toks = set(a.split())
    b_toks = set(b.split())
    # Subset: todos los tokens del más chico están en el más grande.
    if a_toks.issubset(b_toks) or b_toks.issubset(a_toks):
        return True
    return False


def _best_display_name(appearances: list[dict]) -> str:
    """De varias apariciones del mismo canonical, elegí el display name
    más informativo: el más largo (probablemente "Marina Pérez" en lugar
    de "Marina"), prefiriendo gmail > calendar > whatsapp por confianza.
    """
    source_priority = {"gmail_today": 0, "calendar": 1, "whatsapp": 2}
    candidates = sorted(
        appearances,
        key=lambda a: (
            source_priority.get(a["source"], 99),
            -len(a.get("display_name", "")),
        ),
    )
    return candidates[0].get("display_name") or "?"


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


def _add_or_merge_appearance(
    groups: list[tuple[str, list[dict]]],
    canonical: str,
    appearance: dict,
) -> None:
    """Append `appearance` to the group whose canonical matches (subset
    rule from `_canonicals_match`). If no group matches, create a new
    one. The first canonical added becomes the group's anchor.

    Using a list-of-(canonical, items) instead of dict because the
    matching is fuzzy (subset) — a strict dict-keyed approach would
    miss "Pablo" vs "Pablo F" since they're different keys.
    """
    for i, (existing_canonical, items) in enumerate(groups):
        if _canonicals_match(canonical, existing_canonical):
            items.append(appearance)
            # Si el canonical actual es MÁS específico (más tokens), lo
            # promovemos a anchor del grupo. Mantiene el "mejor" canonical
            # como referencia para futuros merges.
            if len(canonical.split()) > len(existing_canonical.split()):
                groups[i] = (canonical, items)
            return
    groups.append((canonical, [appearance]))


def _correlate_people(today_ev: dict, extras: dict) -> list[dict]:
    """Detect people appearing in ≥2 sources (gmail, whatsapp, calendar).

    Uses subset-canonical matching so "Pablo" (calendar) and "Pablo Fer"
    (gmail) merge to the same person. See `_canonicals_match()`.

    Returns: list of {name, appearances: [{source, context, snippet}],
    sources_count} sorted by sources_count desc.
    """
    # list of (canonical_key, [appearances]) — uses subset matching, no
    # dict because "pablo" and "f pablo" should merge.
    groups: list[tuple[str, list[dict]]] = []

    for m in (extras.get("gmail_today") or [])[:20]:
        sender = m.get("from") or m.get("sender") or ""
        # Skip self-notifications (github bot, google alerts, stripe
        # receipts) — el `display name` de esos suele matchear con el
        # nombre del user generando falsos cruces "persona en gmail+wa".
        if _is_self_notification(sender):
            continue
        display = _extract_name_from_email(sender)
        canonical = _canonicalize_name(display)
        if not canonical:
            continue
        _add_or_merge_appearance(groups, canonical, {
            "source": "gmail_today",
            "display_name": display,
            "context": (m.get("subject") or "")[:80],
            "snippet": (m.get("snippet") or "")[:100],
        })

    wa_buckets = (extras.get("whatsapp_today") or [])
    wa_buckets = wa_buckets + (extras.get("whatsapp_unreplied") or [])
    seen_wa_canonicals: set[str] = set()
    for w in wa_buckets[:20]:
        name = w.get("name") or ""
        canonical = _canonicalize_name(name)
        if not canonical or canonical in seen_wa_canonicals:
            continue
        seen_wa_canonicals.add(canonical)
        count = w.get("count")
        snippet = (w.get("last_snippet") or "")[:100]
        _add_or_merge_appearance(groups, canonical, {
            "source": "whatsapp",
            "display_name": name,
            "context": f"{count} msgs hoy" if count else "WA pendiente",
            "snippet": snippet,
        })

    cal_today_buckets = extras.get("calendar_today") or []
    cal_tomorrow_buckets = extras.get("tomorrow_calendar") or []
    for c in (cal_today_buckets + cal_tomorrow_buckets)[:20]:
        title = c.get("title") or ""
        time_range = c.get("time_range") or ""
        if not time_range:
            start = c.get("start") or ""
            end = c.get("end") or ""
            time_range = f"{start}–{end}" if start and end else (start or "")
        is_tomorrow = c in cal_tomorrow_buckets
        for name in _extract_names_from_title(title):
            canonical = _canonicalize_name(name)
            if not canonical:
                continue
            _add_or_merge_appearance(groups, canonical, {
                "source": "calendar",
                "display_name": name,
                "context": (
                    f"mañana {time_range}" if is_tomorrow
                    else f"hoy {time_range}"
                ),
                "snippet": title[:100],
            })

    out: list[dict] = []
    for canonical, items in groups:
        sources = {it["source"] for it in items}
        if len(sources) < 2:
            continue
        out.append({
            "name": _best_display_name(items),
            "appearances": items,
            "sources_count": len(sources),
        })
    out.sort(
        key=lambda p: (-p["sources_count"], -len(p["appearances"])),
    )
    return out


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


# ── Voice normalization (post-processing del LLM output) ──────────────────


# Mapeo 1ra persona singular ("yo trabajé") → 2da persona singular ("vos
# trabajaste"). Solo se chequean palabras completas (word-boundary).
# Ordenado por longitud DESC para que "encontré" matchee antes que "encé".
_VOICE_VERB_REPLACEMENTS_1PS = [
    # 1PS pretérito perfecto simple regulares -ar → -aste
    ("trabajé", "trabajaste"), ("revisé", "revisaste"),
    ("centré", "centraste"), ("pasé", "pasaste"), ("noté", "notaste"),
    ("preparé", "preparaste"), ("armé", "armaste"), ("dejé", "dejaste"),
    ("envié", "enviaste"), ("mandé", "mandaste"), ("toqué", "tocaste"),
    ("escribí", "escribiste"), ("entré", "entraste"),
    ("encontré", "encontraste"), ("intenté", "intentaste"),
    ("cerré", "cerraste"), ("llamé", "llamaste"), ("hablé", "hablaste"),
    ("pregunté", "preguntaste"), ("contesté", "contestaste"),
    ("avancé", "avanzaste"), ("logré", "lograste"), ("terminé", "terminaste"),
    ("empecé", "empezaste"), ("agregué", "agregaste"),
    ("investigué", "investigaste"), ("comparé", "comparaste"),
    ("reuní", "reuniste"),
    # 1PS irregulares
    ("recibí", "recibiste"), ("vi", "viste"), ("fui", "fuiste"),
    ("estuve", "estuviste"), ("tuve", "tuviste"), ("hice", "hiciste"),
    ("dije", "dijiste"), ("vine", "viniste"), ("puse", "pusiste"),
    ("supe", "supiste"), ("anduve", "anduviste"), ("traje", "trajiste"),
    ("pude", "pudiste"), ("quise", "quisiste"), ("leí", "leíste"),
    ("oí", "oíste"), ("salí", "saliste"), ("dormí", "dormiste"),
    ("comí", "comiste"), ("subí", "subiste"), ("abrí", "abriste"),
    ("escogí", "escogiste"), ("seguí", "seguiste"), ("conseguí", "conseguiste"),
    # Presente 1PS irregulares
    ("tengo", "tenés"), ("hago", "hacés"), ("digo", "decís"),
    ("pongo", "ponés"), ("salgo", "salís"), ("vengo", "venís"),
    ("conozco", "conocés"), ("sé", "sabés"),
    # 1ra plural ("nosotros") en pretérito perfecto suelen sonar igual
    # (-amos / -imos) — el LLM a veces dice "trabajamos" / "vimos" para
    # incluir al user. Convertir a 2da singular del usuario.
    ("trabajamos", "trabajaste"), ("revisamos", "revisaste"),
    ("vimos", "viste"), ("hicimos", "hiciste"), ("estuvimos", "estuviste"),
    ("tuvimos", "tuviste"), ("fuimos", "fuiste"), ("dijimos", "dijiste"),
    ("tocamos", "tocaste"), ("notamos", "notaste"),
    ("encontramos", "encontraste"), ("dejamos", "dejaste"),
    ("recibimos", "recibiste"), ("preparamos", "preparaste"),
]


# Pronombres / determinantes en 1ra persona → 2da. Cuidado con falsos
# positivos: "mi" puede ser nota musical. La regla pragmática: solo
# reemplazar al inicio de palabra Y dentro de un contexto de prosa
# narrativa. El regex word-boundary es suficiente.
_VOICE_PRONOUN_REPLACEMENTS = [
    # Pronombres sujeto / objeto
    ("yo", "vos"),
    # "me" reflexivo: se queda como "te"
    ("me ", "te "),
    # "mi/mí/mío/míos/mía/mías" → "tu/vos/tuyo..."
    ("mi ", "tu "),
    ("mí ", "vos "),
    ("mío", "tuyo"),
    ("míos", "tuyos"),
    ("mía", "tuya"),
    ("mías", "tuyas"),
    # "nos" reflexivo plural ("nos vimos") → "te"
    ("nos ", "te "),
]


def _make_word_boundary_pattern(words: list[str]) -> "re.Pattern":
    """Compila un único regex que matchea CUALQUIERA de las palabras
    como word completa (word boundary). Más rápido que iterar N regex.
    """
    # Sort longest-first para que "encontré" capture antes que "encé"
    sorted_words = sorted(words, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted_words) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def normalize_voice_to_2da_persona(text: str) -> str:
    """Reemplaza verbos / pronombres en 1ra persona por 2da singular.

    Pragmático: el prompt prohíbe 1ª persona pero el modelo 7B se desliza
    ~10% del tiempo. Este pass es el último guard. NO toca:
      - Texto entre comillas (preserva citas literales del user)
      - Substrings dentro de palabras (ej. "concentré" no matchea
        "centré" porque el word-boundary se aplica)
      - Code fences (no debería haber en briefs, pero por las dudas)

    Estrategia de preservación de citas: split por delimiters de cita
    `"..."` y `'...'`, normalizar SOLO los segmentos fuera de comillas,
    re-juntar.

    Aplica una sola vez (no idempotente para evitar over-replace en
    ediciones repetidas).
    """
    if not text:
        return text

    # Build maps: lowercase verb / pronoun → replacement (preservando
    # case del original donde se pueda — ej. "Recibí" → "Recibiste").
    verb_map = {orig: repl for orig, repl in _VOICE_VERB_REPLACEMENTS_1PS}
    verb_re = _make_word_boundary_pattern(list(verb_map.keys()))

    # Pronouns con espacio trailing — usamos un regex distinto porque
    # ya incluye el espacio en el match.
    # Para "me " etc., el regex es \bme\s — matchea la palabra + 1 ws.
    pronoun_pattern = r"\b(yo|me|mi|mí|mío|míos|mía|mías|nos)\b"
    pronoun_re = re.compile(pronoun_pattern, re.IGNORECASE)
    pronoun_map = {
        "yo": "vos",
        "me": "te",
        "mi": "tu",
        "mí": "vos",
        "mío": "tuyo",
        "míos": "tuyos",
        "mía": "tuya",
        "mías": "tuyas",
        "nos": "te",
    }

    def _preserve_case(orig: str, repl: str) -> str:
        if orig.isupper():
            return repl.upper()
        if orig[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    def _replace_verb(m: re.Match) -> str:
        orig = m.group(0)
        repl = verb_map.get(orig.lower(), orig)
        return _preserve_case(orig, repl)

    def _replace_pronoun(m: re.Match) -> str:
        orig = m.group(0)
        repl = pronoun_map.get(orig.lower(), orig)
        return _preserve_case(orig, repl)

    # Preservar citas: split text por comillas dobles y procesar solo
    # segmentos pares (índices 0, 2, 4...).
    parts = re.split(r'(".*?"|\'.*?\')', text)
    out_parts: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # adentro de comillas
            out_parts.append(part)
            continue
        # Aplicar verb replacements primero, después pronouns.
        normalized = verb_re.sub(_replace_verb, part)
        normalized = pronoun_re.sub(_replace_pronoun, normalized)
        out_parts.append(normalized)
    return "".join(out_parts)


def _correlate_mood(today_ev: dict, extras: dict) -> dict | None:
    """Lee el score diario de hoy + los últimos 7 días desde
    `rag_mood_score_daily` (lo escribe el daemon `mood-poll` cada 30
    min — `rag/mood.py:run_poll_cycle()`). Devuelve un bucket con
    shape:

        {
            "score": float,              # score de hoy (-1..+1)
            "n_signals": int,            # cuántas señales lo soportan
            "sources_used": [str],       # ["spotify", "journal", ...]
            "trend": str,                # "stable" | "improving" | "declining"
            "week_avg": float,           # media móvil 7d
            "drift": {                   # del recent_drift()
                "drifting": bool,
                "n_consecutive": int,
                "avg_score": float,
            },
            "top_evidence": [...],       # top 3 señales del día
        }

    Devuelve `None` si:
      - el feature está off (`RAG_MOOD_ENABLED` no seteado),
      - no hay row para hoy todavía (daemon nunca corrió),
      - hubo error leyendo (silent-fail con None).

    El consumer downstream (today brief prompt) decide si/cómo
    modular tono según este bucket. ESTE módulo NO verbaliza el
    score — solo lo expone como contexto cross-source. La regla
    "no decir 'noté que estás triste'" se aplica en el prompt
    template, no acá.
    """
    try:
        from rag import mood as _mood  # noqa: PLC0415
    except Exception:
        return None
    if not _mood._is_mood_enabled():
        return None

    try:
        today = _mood._today_local()
        score_row = _mood.get_score_for_date(today)
        if score_row is None or score_row.get("n_signals", 0) == 0:
            return None
        recent = _mood.get_recent_scores(days=7)
        drift = _mood.recent_drift(days=7)
    except Exception:
        return None

    # week_avg de los últimos 7 días con n_signals > 0 (excluye hoy si querés
    # comparar con baseline, pero acá lo incluimos: el "trend" mide cuánto
    # se desvía hoy del promedio reciente).
    valid = [r for r in recent if r["n_signals"] > 0]
    if not valid:
        week_avg = score_row["score"]
    else:
        week_avg = sum(r["score"] for r in valid) / len(valid)

    delta = score_row["score"] - week_avg
    if delta > 0.2:
        trend = "improving"
    elif delta < -0.2:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "score": round(score_row["score"], 3),
        "n_signals": score_row["n_signals"],
        "sources_used": score_row.get("sources_used") or [],
        "trend": trend,
        "week_avg": round(week_avg, 3),
        "drift": {
            "drifting": bool(drift.get("drifting", False)),
            "n_consecutive": int(drift.get("n_consecutive", 0)),
            "avg_score": round(float(drift.get("avg_score", 0.0)), 3),
        },
        "top_evidence": (score_row.get("top_evidence") or [])[:3],
    }


def correlate_today_signals(today_ev: dict, extras: dict) -> dict:
    """Pre-correlate cross-source signals. Returns:
        {
            "people": [{name, appearances: [...], sources_count}, ...],
            "topics": [{topic, sources, sources_count}, ...],
            "time_overlaps": [{time, items: [...], shared_tokens}, ...],
            "gaps": [{kind, person, hours_waiting, snippet, context}, ...],
            "mood": {score, trend, drift, ...} | None,  # None si feature off
        }

    Empty buckets are silently skipped — `today_ev` and `extras` can
    have any subset of keys; missing keys default to []/{}.

    `mood` viene poblado solo cuando `RAG_MOOD_ENABLED=1` Y el daemon
    `mood-poll` ya escribió un row para hoy en `rag_mood_score_daily`.
    En cualquier otro caso queda `None` y el prompt downstream lo
    detecta + skipea la modulación.
    """
    return {
        "people": _correlate_people(today_ev or {}, extras or {}),
        "topics": _correlate_topics(today_ev or {}, extras or {}),
        "time_overlaps": _correlate_time_overlaps(today_ev or {}, extras or {}),
        "gaps": _correlate_gaps(today_ev or {}, extras or {}),
        "mood": _correlate_mood(today_ev or {}, extras or {}),
    }
