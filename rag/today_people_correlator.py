"""People cross-source correlator — extracted from rag/today_correlator.py 2026-05-09.

Detecta personas que aparecen en ≥2 fuentes (gmail, whatsapp, calendar)
para que el brief LLM tenga el match pre-computado en vez de inventarlo.

Pipeline:
  1. Extract display names from each source (gmail-from, wa-name, cal-title).
  2. Filter self-notifications (github bot, google alerts, stripe receipts).
  3. Canonicalize (lowercase + NFD + sin diacríticos + token-sort).
  4. Subset-match (`_canonicals_match`): "Pablo" matches "Pablo Fer" si
     el más corto es subset del más largo.
  5. Group + emit los que tengan ≥2 sources distintas.

Output: list[{name, appearances: [{source, context, snippet}], sources_count}]
sorted by sources_count desc.

Re-exportado desde ``rag.today_correlator`` para preservar
``from rag.today_correlator import _correlate_people`` etc.

NO mueve: ``_STOPWORDS``, ``_TOKEN_RE``, ``_tokenize`` — esos los comparten
los otros 2 correlators (topics + time_overlaps) y se quedan en
``today_correlator.py``.
"""
from __future__ import annotations

import re

__all__ = [
    "_SELF_NOTIFICATION_DOMAINS",
    "_is_self_notification",
    "_EMAIL_NAME_RE",
    "_EMAIL_BARE_RE",
    "_extract_name_from_email",
    "_TITLE_TOKEN_RE",
    "_TITLE_NON_NAMES",
    "_extract_names_from_title",
    "_canonicalize_name",
    "_canonicals_match",
    "_best_display_name",
    "_add_or_merge_appearance",
    "_correlate_people",
]


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
