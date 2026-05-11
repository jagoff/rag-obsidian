"""Contacts resolution — vault `99-Contacts/` + Apple Contacts My Card.

Las notas en `99-obsidian/99-Contacts/` son la fuente PRIMARIA y autoritativa
de contactos del user — escritas a mano con teléfono real, alias, relación,
apellido completo. Tienen prioridad sobre Apple Contacts y My Card resolver
porque acá NO hay ambigüedad: "Mama.md" es la mamá del user, "Maria.md" es
la esposa, etc. (decisión del user 2026-04-26).

Surface (todas funciones privadas, prefix `_`):

- ``_normalize_hint(s)`` — lowercase + strip accents (NFD).
- ``_strip_emoji_and_symbols(s)`` — quita emojis/símbolos del personName.
- ``_parse_apple_label(raw)`` — `_$!<Mother>!$_` o "Madre" → canonical English.
- ``_load_my_card_relations()`` — Related Names del My Card de Apple Contacts
  (cached 1h). Devuelve `[{label, personName}]`.
- ``_strip_possessive_prefix(s)`` — strippea "mi"/"a mi"/"la"/etc.
- ``_vault_contacts_dir()`` — Path al folder `99-Contacts/` o None.
- ``_parse_vault_contact(path, text)`` — parsea una nota de contact.
- ``_load_vault_contacts()`` — scan del directorio (cached 60s).
- ``_lookup_vault_contact(query)`` — resolver query → contact dict del vault.
- ``_wa_sanitize_applescript_string(value)`` — guard injection osascript.
- ``_exact_contact_lookup(person_name)`` — match exacto en Apple Contacts.
- ``_resolve_via_my_card_relationship(hint)`` — "mama" → personName real
  via My Card.

Invariantes:
- Silent-fail: subprocess error / permission denied / file not found → None
  o lista vacía. Nunca raise.
- Caches con TTL bajo: My Card 1h (cambios raros, dialog de permisos solo
  fires once); vault contacts 60s (user puede editar a mano).
"""

from __future__ import annotations

import re
from datetime import datetime  # noqa: F401  — preservado para compat de re-exports
from pathlib import Path

from rag.integrations.whatsapp._constants import (
    VAULT_CONTACTS_SUBPATH,
    _APPLE_LABEL_ES_TO_EN,
    _RELATIONSHIP_HINT_MAP,
)


# ── Strip de prefijos posesivos ────────────────────────────────────────────
# Cubre: "mi", "a mi", "la", "el" al inicio (sin acento o con). NO toca
# nombres reales que casualmente arrancan con "mi" (ej. "Miguel") porque
# requiere el espacio después. "Miguel" → no strip; "mi Hermana" → strip.
_POSSESSIVE_PREFIX_RE = re.compile(
    r"^\s*(?:a\s+)?(?:mi|m[ií]a|mio|m[ií]o|tu|su|el|la)\s+",
    re.IGNORECASE,
)

# AppleScript injection guard (local — duplicado de
# `_sanitize_applescript_string` en `rag/__init__.py` para evitar import
# circular en load-time). Mantener en sync si la allowlist cambia.
_WA_APPLESCRIPT_SAFE_RE = re.compile(
    r"^[\w \-_'.@#+À-ɏ]{1,200}$"
)


# ── Caches con TTL ──────────────────────────────────────────────────────────
# My Card: cambios raros + Apple Contacts permission dialog fires once.
_MY_CARD_RELATIONS_CACHE: dict | None = None
_MY_CARD_RELATIONS_TTL_S = 3600

# Vault contacts: user puede editar a mano. Cold call ~5-20ms (8 archivos chicos).
_VAULT_CONTACTS_CACHE: dict | None = None
_VAULT_CONTACTS_TTL_S = 60


# ── Kinship inference ─────────────────────────────────────────────────────
# Map de keyword → kinship enum. Aplicado al `relation_label` libre del
# contact note cuando el frontmatter no declara `kinship` explícito. El
# matching es por substring (lowercase, accent-stripped) — chequea palabras
# completas via word-boundary mental, pero acá usamos `in` simple porque las
# keywords son discriminantes (no se solapan con nombres comunes).
#
# Por qué este map: el draft prompt necesita un signal categórico para ajustar
# registro (cariñoso vs formal). El text libre "hija" / "psiquiatra" /
# "amigo de la facu" es bueno para humanos pero el LLM no consistentemente
# infiere el registro correcto. Inyectarle el enum + descripción del registro
# fuerza el alineamiento.
#
# Orden de chequeo importa: el primero que matchee gana. Family-immediate va
# primero porque "mamá" / "papá" / "hijo" son los casos más críticos para
# evitar tono frío.
_KINSHIP_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("family-immediate", (
        "hija", "hijo", "madre", "padre", "mama", "mamá", "papa", "papá",
        "hermana", "hermano", "esposa", "esposo", "marido", "mujer",
    )),
    ("romantic-partner", (
        "pareja", "novia", "novio", "compañera", "compañero",
    )),
    ("family-extended", (
        "tia", "tía", "tio", "tío", "prima", "primo", "abuela", "abuelo",
        "sobrina", "sobrino", "cuñada", "cuñado", "suegra", "suegro",
        "nieta", "nieto", "madrina", "padrino", "ahijada", "ahijado",
    )),
    ("professional-formal", (
        "cliente", "paciente", "psiquiatra", "psicologa", "psicólogo",
        "psicólogа", "medica", "médica", "medico", "médico", "abogada",
        "abogado", "contadora", "contador", "doctor", "doctora",
        "proveedor", "proveedora",
    )),
    ("professional-close", (
        "colega", "socia", "socio", "jefa", "jefe", "ex jefa", "ex jefe",
    )),
    ("friend-close", (
        "amigo cercano", "amiga cercana", "mejor amigo", "mejor amiga",
        "best friend",
    )),
    ("friend-known", (
        "amiga", "amigo", "conocido", "conocida",
    )),
)


def _infer_kinship(relation_label: str) -> str:
    """Devolver enum kinship a partir del label libre.

    Default `unknown` si no matchea nada (incl. label vacío).
    """
    if not relation_label:
        return "unknown"
    norm = _normalize_hint(relation_label)
    for kinship, keywords in _KINSHIP_KEYWORDS:
        for kw in keywords:
            kw_norm = _normalize_hint(kw)
            # Palabra completa o frase compuesta → substring contains.
            if kw_norm in norm:
                return kinship
    return "unknown"


# ── Short-name (apodo familiar) ───────────────────────────────────────────
# Aliases que NO son short_name reales — son labels del template o headers
# wikilink genéricos. Se filtran del fallback "primer alias".
_SHORT_NAME_TEMPLATE_BLOCKLIST: frozenset[str] = frozenset({
    "apodo", "nombre completo", "otra forma de llamarlo", "nombre corto",
    "short name", "shortname",
})


def _infer_short_name(
    explicit: str,
    aliases: list[str],
    full_name: str,
    path_stem: str,
) -> str:
    """Resolver el nombre corto familiar con fallback chain.

    Prioridad:
      1. `explicit` — campo `Apodo` / `Nombre corto` / `Short name` del body.
      2. Primer alias del frontmatter que NO sea template-placeholder NI sea
         igual al full_name (sería redundante).
      3. Primer token del `full_name` si tiene 2+ palabras (ej. "Grecia
         Ferrari" → "Grecia").
      4. `path_stem` (nombre del archivo sin extensión).

    Es heurística — el user puede override agregando `- **Apodo**: <X>` al
    body de la nota.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    full_name_norm = _normalize_hint(full_name) if full_name else ""
    for alias in aliases:
        a = (alias or "").strip()
        if not a:
            continue
        a_norm = _normalize_hint(a)
        if a_norm in _SHORT_NAME_TEMPLATE_BLOCKLIST:
            continue
        if a_norm == full_name_norm:
            continue
        return a
    if full_name and " " in full_name.strip():
        return full_name.strip().split()[0]
    return path_stem or full_name or ""


def _normalize_hint(s: str) -> str:
    """Lowercase + strip accents → match keys in `_RELATIONSHIP_HINT_MAP`."""
    import unicodedata
    fold = unicodedata.normalize("NFD", s.strip().lower())
    return "".join(c for c in fold if unicodedata.category(c) != "Mn")


def _strip_emoji_and_symbols(s: str) -> str:
    """Remove emojis, pictographic symbols, and variation selectors from
    a person name.

    Users often add hearts/decorations to Related Names ("Maria ❤️",
    "Juli 🥰") but the actual Contacts entry is just the plain name —
    so a literal Apple Contacts lookup with the emoji included fails
    and we either get nothing or fuzzy-match a wrong person.

    Caveats:
    - Variation Selectors (U+FE00–U+FE0F) son `Mn` (mark non-spacing) en
      unicodedata — los strippeamos explícitamente. Sin esto, "Maria ❤️"
      → "Maria " con un VS-16 invisible al final que igual rompía el
      lookup downstream.
    - Apple Skin Tone Modifiers (U+1F3FB–U+1F3FF) son `Sk` — strippeados
      por el filter de cat[0] != "L|M|N|Zs".

    We keep letters (L*), accent marks on letters (Mn pero NO los
    Variation Selectors), digits (N*), spaces (Zs), y unas puntuaciones
    comunes en nombres: hyphen, apostrophe, dot, parens.
    """
    import unicodedata
    out_chars = []
    for ch in s:
        # Strip Variation Selectors explícitamente — son `Mn` pero no
        # son acentos sobre letras, son modifiers de emoji.
        if 0xFE00 <= ord(ch) <= 0xFE0F:
            continue
        # Strip Zero-Width Joiner / Non-Joiner (used in emoji sequences).
        if ch in ("‍", "‌"):
            continue
        cat = unicodedata.category(ch)
        # Keep letters (L*), accent marks (M*), numbers (N*), space (Zs),
        # plus common name punctuation.
        if cat[0] in ("L", "M", "N") or cat == "Zs" or ch in "-'.()":
            out_chars.append(ch)
    cleaned = "".join(out_chars).strip()
    # Collapse repeated whitespace (e.g. "Maria  " → "Maria").
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def _parse_apple_label(raw: str) -> str:
    """Convert Apple's `_$!<Mother>!$_` or raw "Madre" to canonical English."""
    if not raw:
        return ""
    m = re.match(r"^_\$!<(.+)>!\$_$", raw)
    core = (m.group(1) if m else raw).lower()
    fold = _normalize_hint(core)
    return _APPLE_LABEL_ES_TO_EN.get(fold, fold)


def _load_my_card_relations() -> list[dict]:
    """Read related names from Apple Contacts My Card.

    Returns list of `{label: 'mother', personName: 'Mamá'}`. Empty list
    if no My Card, no permissions, or osascript fails. Never raises.

    Cached for 1h — Apple Contacts permission dialog only fires once.
    """
    import time as _time
    global _MY_CARD_RELATIONS_CACHE
    now = _time.time()
    if (_MY_CARD_RELATIONS_CACHE
            and (now - _MY_CARD_RELATIONS_CACHE.get("at", 0)) < _MY_CARD_RELATIONS_TTL_S):
        return _MY_CARD_RELATIONS_CACHE["rows"]

    script = '''tell application "Contacts"
  set _out to ""
  try
    set _myCard to my card
    repeat with _rn in (related names of _myCard)
      set _lbl to (label of _rn as string)
      set _val to (value of _rn as string)
      set _out to _out & _lbl & "|||" & _val & linefeed
    end repeat
  end try
  return _out
end tell'''
    import subprocess
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            _MY_CARD_RELATIONS_CACHE = {"at": now, "rows": []}
            return []
    except Exception:
        _MY_CARD_RELATIONS_CACHE = {"at": now, "rows": []}
        return []

    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        parts = line.split("|||")
        if len(parts) < 2:
            continue
        raw_label, person_name = parts[0].strip(), parts[1].strip()
        label = _parse_apple_label(raw_label)
        # Strip emojis y otros símbolos del personName para que el lookup
        # secundario en Apple Contacts no falle. El user típicamente pone
        # "Maria ❤️" como Related Name, pero el contacto real se llama
        # "Maria Apellido" sin emoji — Apple Contacts no encuentra el
        # match con el emoji presente y termina haciendo fuzzy a otro
        # contacto distinto ("Mariano" matchea "Mari").
        cleaned_name = _strip_emoji_and_symbols(person_name)
        if label and cleaned_name:
            rows.append({"label": label, "personName": cleaned_name})

    _MY_CARD_RELATIONS_CACHE = {"at": now, "rows": rows}
    return rows


def _strip_possessive_prefix(s: str) -> str:
    """Remove 'mi '/'a mi '/'tu '/'la ' etc. del inicio para que el alias
    de parentesco quede limpio. Iterativo por si hay cadena ('a mi tu'
    es absurdo pero el LLM puede emitir cualquier cosa)."""
    out = (s or "").strip()
    for _ in range(3):  # max 3 iteraciones — suficiente para cualquier caso real
        new = _POSSESSIVE_PREFIX_RE.sub("", out, count=1)
        if new == out:
            break
        out = new
    return out


def _vault_contacts_dir() -> Path | None:
    """Path al folder de contactos del vault, o None si no se puede
    resolver el VAULT_PATH (no hay vault registrado)."""
    try:
        from rag import VAULT_PATH
    except ImportError:
        return None
    if not VAULT_PATH:
        return None
    target = VAULT_PATH / VAULT_CONTACTS_SUBPATH
    return target if target.is_dir() else None


def _parse_vault_contact(path: Path, text: str | None = None) -> dict:
    """Parsear una nota de contact del vault.

    Devuelve `{full_name, phones, emails, birthday, source: "vault",
    aliases, relation_label, kinship, short_name}`. Campos vacíos default
    a "" o [].

    `kinship` es enum derivado de `relation_label` (ver `_infer_kinship`)
    salvo que el frontmatter declare `kinship: <enum>` explícito.

    `short_name` es el apodo familiar — fallback chain en `_infer_short_name`.
    User puede override agregando `- **Apodo**: <X>` al body.
    """
    if text is None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""

    def _extract_field(label: str) -> str:
        # "- **Label**: value" — el label es regex (puede traer sets como
        # `[eé]` para tolerancia a acentos). NO escapamos corchetes a
        # propósito; el caller pasa labels seguras / hardcodeadas.
        # Uso `[ \t]*` (horizontal whitespace) en vez de `\s*` para NO
        # cruzar newlines. Bug 2026-05-11: `\s*` después del `:` era
        # greedy y consumía `\n` + línea siguiente cuando el bullet
        # estaba vacío (ej. `- **Apodo**:`), capturando como value el
        # contenido del bullet de abajo. Síntoma: `short_name` salía
        # como literal `- **Apellido / nombre completo**: Erica Franzen`.
        pattern = (
            r"^-[ \t]*\*\*[ \t]*"
            + label
            + r"[ \t]*\*\*[ \t]*:[ \t]*(.*)$"
        )
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    raw_phones = _extract_field(r"Tel[eé]fono") or _extract_field(r"Phone")
    phones = [p.strip() for p in raw_phones.split(",") if p.strip()]
    # Filtrar placeholders del template ("+54 9 ...").
    phones = [p for p in phones if not p.endswith("...")]

    raw_emails = _extract_field(r"Email") or _extract_field(r"Correo")
    emails = [e.strip() for e in raw_emails.split(",") if e.strip()]

    full_name = (
        _extract_field(r"Apellido(?:\s*/\s*nombre\s+completo)?")
        or _extract_field(r"Full[\s-]?name")
        or path.stem
    )
    relation = _extract_field(r"Relaci[oó]n") or _extract_field(r"Relation")
    birthday = _extract_field(r"Cumplea[ñn]os") or _extract_field(r"Birthday")

    # Aliases del frontmatter YAML — solo line-by-line, sin dependencia YAML.
    aliases: list[str] = []
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        in_aliases = False
        for line in fm_match.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith("aliases:"):
                in_aliases = True
                # Inline list: "aliases: [a, b, c]"
                inline = stripped[len("aliases:"):].strip()
                if inline.startswith("[") and inline.endswith("]"):
                    in_aliases = False
                    parts = inline[1:-1].split(",")
                    aliases.extend(p.strip().strip('"\'') for p in parts if p.strip())
                continue
            if in_aliases:
                if stripped.startswith("-"):
                    val = stripped[1:].strip().strip('"\'')
                    if val:
                        aliases.append(val)
                elif stripped and not stripped.startswith(" "):
                    in_aliases = False  # otro key del frontmatter

    # También el wikilink del header: `[[X|@Y]]` → captura X y Y como aliases.
    header_match = re.search(
        r"\[\[(?:[^|\]]+/)?([^|\]]+)(?:\|@?([^\]]+))?\]\]", text,
    )
    if header_match:
        wl_target = header_match.group(1).strip()
        wl_alias = (header_match.group(2) or "").strip()
        if wl_target and wl_target not in aliases:
            aliases.append(wl_target)
        if wl_alias and wl_alias not in aliases:
            aliases.append(wl_alias)

    # Filtrar aliases-template ("Otra forma de llamarlo", "Apodo", etc.)
    _TEMPLATE_ALIASES = {
        "otra forma de llamarlo", "apodo", "nombre completo",
    }
    aliases = [
        a for a in aliases
        if _normalize_hint(a) not in _TEMPLATE_ALIASES
    ]

    # Kinship — frontmatter explícito gana sobre infer del relation_label.
    explicit_kinship = ""
    if fm_match:
        for line in fm_match.group(1).splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("kinship:"):
                explicit_kinship = stripped.split(":", 1)[1].strip().strip('"\'')
                break
    kinship = explicit_kinship if explicit_kinship else _infer_kinship(relation)

    # Short name — body field "Apodo" / "Nombre corto" / "Short name" o
    # frontmatter `short_name`. Si nada explícito, fallback chain.
    explicit_short = (
        _extract_field(r"Apodo")
        or _extract_field(r"Nombre[\s-]?corto")
        or _extract_field(r"Short[\s-]?name")
    )
    if not explicit_short and fm_match:
        for line in fm_match.group(1).splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("short_name:", "short-name:", "shortname:")):
                explicit_short = stripped.split(":", 1)[1].strip().strip('"\'')
                break
    short_name = _infer_short_name(
        explicit_short,
        aliases,
        full_name or path.stem,
        path.stem,
    )

    return {
        "full_name": full_name or path.stem,
        "phones": phones,
        "emails": emails,
        "birthday": birthday,
        "source": "vault",
        "aliases": aliases,
        "relation_label": relation,
        "kinship": kinship,
        "short_name": short_name,
    }


def _load_vault_contacts() -> list[dict]:
    """Scan del directorio `99-Contacts/`, parsea cada nota.

    Devuelve `[{stem, path, parsed}]`. Skipea `_template.md` y archivos
    con prefijo `_` (scaffolding/internal). Cached por 60s.
    """
    import time as _time
    global _VAULT_CONTACTS_CACHE
    now = _time.time()
    if (_VAULT_CONTACTS_CACHE
            and (now - _VAULT_CONTACTS_CACHE.get("at", 0)) < _VAULT_CONTACTS_TTL_S):
        return _VAULT_CONTACTS_CACHE["contacts"]

    base = _vault_contacts_dir()
    if not base:
        _VAULT_CONTACTS_CACHE = {"at": now, "contacts": []}
        return []

    out: list[dict] = []
    for p in base.glob("*.md"):
        if p.name.startswith("_") or p.name.startswith("."):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        parsed = _parse_vault_contact(p, text)
        out.append({"stem": p.stem, "path": p, "parsed": parsed})

    _VAULT_CONTACTS_CACHE = {"at": now, "contacts": out}
    return out


def _lookup_vault_contact(query: str) -> dict | None:
    """Resolver un query a un contact del vault (`99-Contacts/`).

    Estrategia (en orden de confianza):
      1. Filename exacto (case + accent insensitive): "Mama" → `Mama.md`.
      2. Match de aliases del frontmatter YAML.
      3. Match contra el campo `**Apellido / nombre completo**`.
      4. Match contra `**Relación**` cuando query es alias de parentesco
         (vía RELATIONSHIP_HINT_MAP). "mama" → busca nota con
         Relación=Mamá/Madre.

    Devuelve dict shape `_fetch_contact` (full_name, phones, emails,
    birthday) + extra fields `source="vault"` y `match_kind` para debug.
    None si no encuentra.
    """
    if not query or not query.strip():
        return None
    q_clean = _strip_possessive_prefix(query).strip()
    q_fold = _normalize_hint(q_clean)
    if not q_fold:
        return None

    contacts = _load_vault_contacts()
    if not contacts:
        return None

    # 1. Filename exact match.
    for c in contacts:
        if _normalize_hint(c["stem"]) == q_fold:
            r = dict(c["parsed"])
            r["match_kind"] = "filename"
            return r

    # 2. Alias match (frontmatter aliases + wikilink header).
    for c in contacts:
        for alias in c["parsed"].get("aliases", []):
            if _normalize_hint(alias) == q_fold:
                r = dict(c["parsed"])
                r["match_kind"] = "alias"
                return r

    # 3. Full name (apellido / nombre completo) — match si el query es
    #    parte del full_name (ej. "Monica" matchea "Monica Ferrari").
    for c in contacts:
        full = _normalize_hint(c["parsed"].get("full_name", ""))
        if full and (full == q_fold or q_fold in full.split()):
            r = dict(c["parsed"])
            r["match_kind"] = "full_name"
            return r

    # 4. Relationship hint ("mama"/"papa"/"hermana") → match contra
    #    el campo **Relación**. Resuelve a English canonical y compara
    #    con la relation del contact (también normalizada).
    canonical = _RELATIONSHIP_HINT_MAP.get(q_fold)
    if canonical:
        for c in contacts:
            rel_raw = c["parsed"].get("relation_label", "")
            if not rel_raw:
                continue
            rel_canonical = _APPLE_LABEL_ES_TO_EN.get(_normalize_hint(rel_raw))
            if rel_canonical == canonical:
                r = dict(c["parsed"])
                r["match_kind"] = "relation"
                return r

    return None


def _wa_sanitize_applescript_string(value: str) -> "str | None":
    """Sanitiza un valor para interpolación segura en osascript (whatsapp.py).

    Retorna la versión escapada si es seguro, o None si contiene chars
    peligrosos para inyección. Misma semántica que _sanitize_applescript_string
    en rag/__init__.py — mantener en sync si se modifica la allowlist.
    """
    if not value:
        return ""
    if not _WA_APPLESCRIPT_SAFE_RE.match(value):
        return None
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _exact_contact_lookup(person_name: str) -> dict | None:
    """Buscar un contacto en Apple Contacts por nombre EXACTO (case-
    insensitive).

    Devuelve `{full_name, phones, emails, birthday}` igual que
    `_fetch_contact`, o `None` si no hay match exacto.

    Se usa como primer intento cuando el lookup viene del Related Names
    resolver — en ese caso el `person_name` es lo que el user puso en
    Contacts, así que un fuzzy/contains lookup puede agarrar el
    contacto equivocado (ej. "Maria" matchea "Mariano" antes que
    "Maria Pérez"). Exact match elimina la ambigüedad.
    """
    if not person_name or not person_name.strip():
        return None
    safe = _wa_sanitize_applescript_string(person_name.strip())
    if safe is None:
        return None  # input peligroso — abortar silenciosamente
    script = f'''tell application "Contacts"
  set _out to ""
  try
    set _people to (every person whose name is "{safe}")
    if (count of _people) > 0 then
      set _p to first item of _people
      set _name to name of _p
      set _phones to ""
      try
        repeat with _ph in (phones of _p)
          set _phones to _phones & (value of _ph as string) & ","
        end repeat
      end try
      set _out to _name & "|||" & _phones
    end if
  end try
  return _out
end tell'''
    import subprocess
    try:
        proc = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip()
        if not line or "|||" not in line:
            return None
        parts = line.split("|||", 1)
        full_name = parts[0].strip()
        phones_csv = parts[1].strip() if len(parts) > 1 else ""
        phones = [p.strip() for p in phones_csv.split(",") if p.strip()]
        return {
            "full_name": full_name,
            "phones": phones,
            "emails": [],
            "birthday": "",
        }
    except Exception:
        return None


def _resolve_via_my_card_relationship(hint: str) -> str | None:
    """Try resolving "mama"/"papa"/etc. → real personName via My Card.

    Acepta también prefijos posesivos rioplatenses ("mi mama", "a mi
    hermana") strippeándolos antes de buscar el alias canónico. Sin este
    paso, el LLM frecuentemente dispara `propose_whatsapp_send` con
    `contact_name="mi Mama"` (en vez de "Mama") y el resolver fallaba
    silenciosamente con `not_found`.

    Returns None if the hint isn't a relationship word, no My Card is set,
    or no related-name match exists for the canonical label.
    """
    cleaned = _strip_possessive_prefix(hint)
    fold = _normalize_hint(cleaned)
    canonical = _RELATIONSHIP_HINT_MAP.get(fold)
    if not canonical:
        return None
    relations = _load_my_card_relations()
    for r in relations:
        if r.get("label") == canonical:
            return r.get("personName")
    return None


__all__ = [
    "_POSSESSIVE_PREFIX_RE",
    "_WA_APPLESCRIPT_SAFE_RE",
    "_normalize_hint",
    "_strip_emoji_and_symbols",
    "_parse_apple_label",
    "_load_my_card_relations",
    "_strip_possessive_prefix",
    "_vault_contacts_dir",
    "_parse_vault_contact",
    "_load_vault_contacts",
    "_lookup_vault_contact",
    "_wa_sanitize_applescript_string",
    "_exact_contact_lookup",
    "_resolve_via_my_card_relationship",
]
