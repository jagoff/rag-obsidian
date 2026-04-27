"""Cross-source correlator for the today brief.

The today brief receives 14+ buckets of data (gmail_today, whatsapp_today,
calendar_today, youtube_today, recent_notes, low_conf_queries, etc.). A 7B LLM
struggles to find cross-source patterns by reading these flat buckets — it
ends up writing tautological "X is related to Y because both are X" insights.

This module pre-computes the patterns BEFORE the LLM call so the prompt
can include a structured ENTIDADES CROSS-SOURCE block. The LLM then narrates
matches that already exist instead of inventing them.

Three correlations:

1. **People** — names that appear in ≥2 sources (gmail-from, wa-name,
   calendar-title). Canonicalized + deduplicated. Output: list of
   {name, appearances: [{source, context, snippet}], sources_count}.

2. **Topics** — keywords that appear in ≥2 source buckets, normalized
   (lowercase, stopwords removed, ≥4 chars). Output: list of
   {topic, sources, sources_count}.

3. **Time overlaps** — events at same hour bucket cross-source (e.g.
   gmail received 14:23 + calendar event 14:00 with overlapping tokens).

Used by `web/server.py:_home_compute` → passed to the prompt via
`extras["correlations"]` → rendered in `_render_today_prompt`.
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
})


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
    sort word tokens (so "Marina Pérez" == "Pérez Marina"). Returns "".
    si el name es vacío o solo digits/símbolos.

    Token-sort permite matchear "Pablo Fernández" vs "Fernández Pablo"
    como la misma persona. Ver `_canonicals_match()` para matching
    subset (e.g. "Pablo" matches "Pablo Fer").
    """
    if not name:
        return ""
    # Strip emojis y caracteres no-alfa (excepto espacios y guiones).
    cleaned = "".join(
        c for c in name
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
    """
    return {
        "gmail": [
            (m.get("subject") or "") + " " + (m.get("snippet") or "")
            for m in (extras.get("gmail_today") or [])
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


def correlate_today_signals(today_ev: dict, extras: dict) -> dict:
    """Pre-correlate cross-source signals. Returns:
        {
            "people": [{name, appearances: [...], sources_count}, ...],
            "topics": [{topic, sources, sources_count}, ...],
        }

    Empty buckets are silently skipped — `today_ev` and `extras` can
    have any subset of keys; missing keys default to []/{}.
    """
    return {
        "people": _correlate_people(today_ev or {}, extras or {}),
        "topics": _correlate_topics(today_ev or {}, extras or {}),
    }
