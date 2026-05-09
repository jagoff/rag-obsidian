"""Person mention enrichment — index + match for 99-Mentions/.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el sub-sistema de mentions parsing + index + query-matching
desde `rag/__init__.py`.

## Qué vive acá

- **Constants**: `_MENTIONS_FOLDER`, `_MENTIONS_BODY_CAP`,
  `_MENTIONS_MAX_PER_QUERY`, `_MENTIONS_MIN_TOKEN_LEN`,
  `_PERSON_CONTEXT_HEADER`, `_RELATION_STEMS`,
  `_MENTION_FIELD_RE`, `_DOSSIER_PHONE_LABELS`,
  `_DOSSIER_EMAIL_LABELS`, `_MENTIONS_EMAIL_RE`,
  `_MENTIONS_PHONE_RE`.
- **Parsing**: `_fold` (NFD strip + lowercase), `_parse_mention_metadata`,
  `_strip_frontmatter`, `_normalise_phone_digits`,
  `_parse_dossier_phones`, `_parse_dossier_emails`.
- **Index + match**: `_load_mentions_index` (cached por max-mtime),
  `_match_mentions_in_query` (word-boundary accent-insensitive).
- **Cache state**: `_mentions_cache`, `_mentions_cache_lock`,
  `_contacts_cache`, `_contacts_cache_lock`,
  `_contacts_permission_warned`.

## NO vive acá (queda en `rag/__init__.py`)

- `_fetch_contact` (AppleScript-based contacts lookup) — depende
  de `_osascript_contact_search` + AppleScript safety guard.
- `build_person_context` — orquesta `_match_mentions_in_query` +
  `_fetch_contact`; lazy-importa ambos.

## Lazy imports

Solo `VAULT_PATH` (en `rag/__init__.py`). Lazy adentro de las
funciones que lo usan (`_load_mentions_index`,
`_match_mentions_in_query`).

## Re-export

`rag/__init__.py` hace `from rag._person_mentions import *  # noqa`.
Preserva 100% compat con `rag._match_mentions_in_query(...)`,
`rag._load_mentions_index(...)`, `rag._fold(...)`,
`rag._parse_mention_metadata(...)`, etc.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from pathlib import Path

__all__ = [
    "_MENTIONS_FOLDER",
    "_MENTIONS_BODY_CAP",
    "_MENTIONS_MAX_PER_QUERY",
    "_MENTIONS_MIN_TOKEN_LEN",
    "_PERSON_CONTEXT_HEADER",
    "_RELATION_STEMS",
    "_MENTION_FIELD_RE",
    "_DOSSIER_PHONE_LABELS",
    "_DOSSIER_EMAIL_LABELS",
    "_MENTIONS_EMAIL_RE",
    "_MENTIONS_PHONE_RE",
    "_mentions_cache",
    "_mentions_cache_lock",
    "_contacts_cache",
    "_contacts_cache_lock",
    "_contacts_permission_warned",
    "_fold",
    "_parse_mention_metadata",
    "_strip_frontmatter",
    "_normalise_phone_digits",
    "_parse_dossier_phones",
    "_parse_dossier_emails",
    "_load_mentions_index",
    "_match_mentions_in_query",
]


_MENTIONS_FOLDER = "99-obsidian/99-Mentions"
_MENTIONS_BODY_CAP = 1500
_MENTIONS_MAX_PER_QUERY = 2
_MENTIONS_MIN_TOKEN_LEN = 3
_PERSON_CONTEXT_HEADER = (
    "IDENTIDAD DEL SUJETO — contexto prioritario\n"
    "Nombraste a una persona cercana tuya (definida en 99-Mentions).\n"
    "NO interpretes el nombre como un país, lugar, evento histórico ni homónimo.\n"
    "Cualquier chunk recuperado que hable de un homónimo (país, ciudad, figura pública) "
    "es ruido — mencionalo solo si pide contexto histórico explícito. "
    "Respondé primero sobre ESTA persona usando los datos que siguen. "
    "Dirigite al usuario en 2da persona (tuteo rioplatense): 'tu hija', 'tu hermano', "
    "'tu socio'. NUNCA 'la hija del usuario', 'el hermano del usuario' ni 3ra persona."
)
# Stems that are kinship/relationship words rather than person names. Searching
# Apple Contacts for these returns substring false-positives (e.g. file stem
# "Mama" matches "Carina (Mama Bianca)"). When the stem is in this set we use
# only the canonical name from the mention body to query Contacts.
_RELATION_STEMS = {
    "mama", "mami", "papa", "papi", "yo", "hijo", "hija",
    "hermano", "hermana", "abuelo", "abuela", "tio", "tia",
    "primo", "prima", "novio", "novia", "esposo", "esposa",
    "pareja", "marido", "mujer",
}
# Frontmatter field labels we parse from mention bodies. Tolerant of bold
# markers, leading dashes, and the YAML-style hyphen list prefix.
_MENTION_FIELD_RE = re.compile(
    r"^[\s\-*]*\*{0,2}([A-Za-zÁÉÍÓÚáéíóúñÑ /]+?)\*{0,2}\s*:\s*(.*)$"
)

_mentions_cache: dict | None = None
# Serialises `_load_mentions_index` — multiple query paths hit it in
# parallel (retrieve()'s mention-boost + ambient hook + followup), and
# concurrent rebuilds would re-scan the folder + redundantly overwrite
# `_mentions_cache`. Lock is cheap: the load is cached by max-mtime so
# steady-state hits never do disk I/O under the lock.
_mentions_cache_lock = threading.Lock()
_contacts_cache: dict[str, dict | None] = {}
_contacts_cache_lock = threading.Lock()
_contacts_permission_warned = False


def _fold(s: str) -> str:
    """Lowercase + strip combining accents (NFD). Matches `_tokenize`."""
    n = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _parse_mention_metadata(body: str) -> dict:
    """Extract canonical name + email from a mention note body. Used to query
    Apple Contacts with the real-name identity rather than a kinship label.

    Recognises field labels: 'Apellido / nombre completo', 'Apellido', 'Nombre
    completo', 'Email'. Skips template placeholders (the `_template.md` ships
    with `+54 9 ...`, `dd/mm/yyyy`, etc.).
    """
    out: dict = {"email": None, "canonical": None}
    for line in body.splitlines():
        m = _MENTION_FIELD_RE.match(line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        value = m.group(2).strip()
        if not value or value in {"...", "—", "-"}:
            continue
        # Strip wikilink wrappers, trailing wikilink fragments
        value = re.sub(r"^\[\[|\]\]$", "", value).split("|")[0].strip()
        if "apellido" in label or "nombre completo" in label:
            if out["canonical"] is None:
                out["canonical"] = value
        elif label == "email" or label == "mail":
            if "@" in value and out["email"] is None:
                out["email"] = re.split(r"[\s,;|]", value)[0].strip()
    return out


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


# Dossier body parsing — centralizado. Antes de 2026-04-21 habían TRES
# regexes divergentes para parsear el bullet `- **Teléfono**: ...`:
#   1. `_MENTIONS_PHONE_RE` (label amplio: tel|phone|cel|whatsapp|wa...)
#   2. `_parse_mention_dossier` (label estrecho: solo "Teléfono")
#   3. `_MENTIONS_EMAIL_RE` (análogo para email)
# Los 2 phone regexes daban resultados distintos: un dossier con `- Cel: X`
# entraba al `_load_mentions_index` pero NO al `_load_phone_index`, dejando
# el WA-sender resolver ciego a ese contacto. Consolidado acá en 2 helpers
# pequeños que cada caller usa — si querés agregar un label nuevo, tocás
# una constante y los 2 índices se sincronizan.
#
# Alt forms aceptadas: `- Email:`, `* Email:`, `- **Email**:`, leading
# whitespace. Case-insensitive. Phones digit-normalised (strip `+`, spaces,
# dashes, parens). Emails lower-cased.

# Etiquetas de phone en dossiers — sincronizado con lo que soporta
# `_parse_dossier_phones`.
_DOSSIER_PHONE_LABELS = (
    "tel[eé]fono|tel[eé]f|tel|phone|celular|cel|m[oó]vil|mobile|whatsapp|wa"
)
_DOSSIER_EMAIL_LABELS = "e-?mail|email|correo"

# Un regex por concepto. `^` + MULTILINE para anclar a línea individual;
# label case-insensitive; opcional bold `**...**`.
_MENTIONS_EMAIL_RE = re.compile(
    rf"^\s*[-*]\s*\*{{0,2}}(?:{_DOSSIER_EMAIL_LABELS})\*{{0,2}}\s*:\s*([^\s<>]+@[^\s<>]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MENTIONS_PHONE_RE = re.compile(
    rf"^\s*[-*]\s*\*{{0,2}}(?:{_DOSSIER_PHONE_LABELS})\*{{0,2}}"
    r"\s*:\s*(\+?[\d\s().\-]{7,})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalise_phone_digits(raw: str) -> str:
    """Strip a phone number to digits only. Returns empty string if the
    remaining digit count is < 8 (too short to be a real phone; drop)."""
    digits = re.sub(r"\D", "", raw or "")
    return digits if len(digits) >= 8 else ""


def _parse_dossier_phones(body: str) -> list[str]:
    """Return all phone numbers in a dossier body as digits-only strings.

    Canonical parser for the `- **Teléfono**: …` / `- Cel: …` / `- WA: …`
    bullets. Reused by `_load_mentions_index` (outbound: query → dossier),
    `_load_phone_index` (inbound: WA sender → dossier), and
    `_parse_mention_dossier` (dossier metadata extraction).

    Filters entries whose digit-normalized form is < 8 digits (too short
    to be a real phone). Preserves order of occurrence. Deduplicates —
    the same phone listed twice returns once.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _MENTIONS_PHONE_RE.finditer(body or ""):
        digits = _normalise_phone_digits(m.group(1))
        if digits and digits not in seen:
            seen.add(digits)
            out.append(digits)
    return out


def _parse_dossier_emails(body: str) -> list[str]:
    """Return all email addresses in a dossier body as lowercased strings.
    Same unification rationale as `_parse_dossier_phones` — one place to
    fix the regex or add labels."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MENTIONS_EMAIL_RE.finditer(body or ""):
        email = (m.group(1) or "").strip().lower()
        if email and "@" in email and email not in seen:
            seen.add(email)
            out.append(email)
    return out


def _load_mentions_index(vault_root: Path | None = None) -> dict[str, str]:
    """Scan 99 Mentions @/ → {folded_token: rel_path}.

    Tokens = filename stem + frontmatter `aliases:` + body-level email +
    phone (last-8 digits). Files prefixed with `_` are scaffolding (see
    `_template.md`). Cached by max-mtime over folder.

    Email / phone enrichment (2026-04-21): dossiers typically carry
    `- **Email**: name@ex.com` and `- **Teléfono**: +54 9 342...` bullets.
    Indexing those lets queries like "mensajes de monicaferrari@gmail.com"
    or "+5493425476623" resolve to the right dossier in
    `build_person_context`, which previously only matched on name/alias.

    Phone is stored under its last 8 digits (accommodates AR "9" prefix
    variance) AND the full digit string; either is sufficient to match
    when the user types the number in different shapes. Email is stored
    lower-cased. Both are folded through `_fold` for alias parity.
    """
    import rag as _rag  # noqa: PLC0415

    global _mentions_cache
    vault_path_default = getattr(_rag, "VAULT_PATH", None)
    root = (vault_root or vault_path_default) / _MENTIONS_FOLDER
    if not root.exists():
        return {}
    files = [p for p in root.glob("*.md") if not p.name.startswith("_")]
    mtime_max = max((p.stat().st_mtime for p in files), default=0.0)
    cache_key = str(root.resolve())
    with _mentions_cache_lock:
        if (
            _mentions_cache
            and _mentions_cache.get("key") == cache_key
            and _mentions_cache.get("mtime") == mtime_max
        ):
            return _mentions_cache["index"]

    idx: dict[str, str] = {}
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(p.relative_to(vault_root or vault_path_default))
        if len(p.stem) >= _MENTIONS_MIN_TOKEN_LEN:
            idx[_fold(p.stem)] = rel
        # Lightweight aliases parser — only the `aliases:` block.
        if txt.startswith("---\n"):
            end = txt.find("\n---", 4)
            if end != -1:
                fm = txt[4:end]
                in_aliases = False
                for line in fm.split("\n"):
                    if line.startswith("aliases:"):
                        rest = line.split(":", 1)[1].strip()
                        if rest.startswith("[") and rest.endswith("]"):
                            for a in rest[1:-1].split(","):
                                a = a.strip().strip("'\"")
                                if len(a) >= _MENTIONS_MIN_TOKEN_LEN:
                                    idx[_fold(a)] = rel
                        else:
                            in_aliases = True
                        continue
                    if in_aliases:
                        if line.startswith("  - "):
                            a = line[4:].strip().strip("'\"")
                            if len(a) >= _MENTIONS_MIN_TOKEN_LEN:
                                idx[_fold(a)] = rel
                        elif line and not line.startswith(" "):
                            in_aliases = False
        # Body-level email + phone enrichment — index the dossier by its
        # contact info so queries with email/phone can resolve to it. Both
        # come from the centralized parsers `_parse_dossier_emails/phones`
        # so this index + `_load_phone_index` stay in sync on label support.
        for email in _parse_dossier_emails(txt):
            idx[_fold(email)] = rel
        for digits in _parse_dossier_phones(txt):
            # Index full digits AND last-8 (AR "9" prefix variance:
            # "+5493425476623" vs "3425476623" both resolve).
            idx[digits] = rel
            if len(digits) > 8:
                idx[digits[-8:]] = rel
    with _mentions_cache_lock:
        _mentions_cache = {"key": cache_key, "mtime": mtime_max, "index": idx}
    return idx


def _match_mentions_in_query(query: str, vault_root: Path | None = None) -> list[str]:
    """Word-boundary, accent-insensitive match over mention tokens (names,
    aliases, body-level emails, phone-digit sequences). Returns rel_paths
    ordered by first occurrence in the query, capped at _MENTIONS_MAX_PER_QUERY.

    Match strategy:
      1. For name/alias/email tokens (contain at least one letter): word-
         boundary `\\b...\\b` over folded query. Emails work because "@" is a
         non-word char so `\\b` aligns at both ends of the local-part and
         domain boundaries — the full email string must appear contiguous
         in the query.
      2. For pure-digit tokens (phone numbers, 8-15 digits): strip all
         non-digits from the query first, then substring-match. This
         accommodates user typing `+54 9 342...` or `(342) 547-...` or
         `3425476623` — all map to the same digit-run and index lookup.
    """
    idx = _load_mentions_index(vault_root)
    if not idx:
        return []
    folded_q = _fold(query)
    # Precompute the digit-only projection of the query once, for phone-matching.
    digits_q = re.sub(r"\D", "", query or "")
    hits: list[tuple[int, str]] = []
    seen_paths: set[str] = set()
    for token, path in idx.items():
        if path in seen_paths:
            continue
        # Pure-digit token → phone. Substring-match against digit-projected query.
        if token.isdigit() and len(token) >= 8:
            pos = digits_q.find(token)
            if pos != -1:
                # Position within digits_q isn't strictly comparable to
                # folded_q positions, but ordering by first digit-match is
                # good enough for the cap-2 case; negative position for
                # phone-matches puts them BEFORE name-matches when both fire.
                hits.append((pos, path))
                seen_paths.add(path)
            continue
        m = re.search(rf"\b{re.escape(token)}\b", folded_q)
        if m:
            hits.append((m.start(), path))
            seen_paths.add(path)
    hits.sort()
    return [p for _, p in hits[:_MENTIONS_MAX_PER_QUERY]]
