"""WhatsApp integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Sources:

- **Bridge SQLite**: `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db`
  — local sqlite written by [whatsapp-mcp/whatsapp-bridge](https://github.com/lharries/whatsapp-mcp).
  Read-only access; we never mutate the bridge's state. Polled by
  `_fetch_whatsapp_unread`, `_fetch_whatsapp_window`,
  `_whatsapp_resolve_reply_target`.
- **Bridge HTTP** (`http://localhost:8080/api/send`): the same MCP bridge
  exposes a tiny send endpoint. Used by `_whatsapp_send_to_jid` and by the
  `_ambient_whatsapp_send` wrapper (which prefixes U+200B as an anti-loop
  marker so the listener bot doesn't process its own output as a query).
- **Apple Contacts** via `_fetch_contact` (lives in `rag.__init__`): used by
  `_whatsapp_jid_from_contact` to resolve a name like "Grecia" to a JID.

## Surfaces

Read paths:
- `_fetch_whatsapp_unread(hours, max_chats)` — recent inbound messages
  grouped by chat. Used by morning brief.
- `_fetch_whatsapp_window(since, now, processed_ids)` — per-chat conversation
  windows for the wa-tasks extractor. Filters out already-processed message
  IDs (cross-run dedup ring).
- `_whatsapp_resolve_reply_target(contact, when_hint, ...)` — resolve a
  "responder a X" request to a concrete bridge message ID + content.

Send paths:
- `_whatsapp_send_to_jid(jid, text, anti_loop, reply_to)` — low-level POST.
- `_ambient_whatsapp_send(jid, text)` — fire-and-forget wrapper with anti-loop.
- `_whatsapp_jid_from_contact(contact_name)` — name → JID via Apple Contacts.

LLM-on-WA path:
- `_wa_extract_actions(label, is_group, msgs)` — qwen2.5:3b extracts tasks /
  questions / commitments from a chat window. Used by `rag wa-tasks`.
- `_wa_chat_label(name, jid)` — display label (drops digit-only names).
- `_wa_chat_month_link(jid, label, ts)` — wikilink to the vault-sync'd note.
- `_wa_tasks_load_state` / `_wa_tasks_save_state` — high-water mark + dedup ring.
- `_wa_tasks_write_note(vault, run_ts, by_chat, extractions)` — appends a
  timestamped section to `00-Inbox/WA-YYYY-MM-DD.md`.
- `_wa_tasks_plist(rag_bin)` — launchd plist for the 30-min cron.

## Invariants
- Silent-fail: missing bridge DB / locked SQLite / network error / bad JSON
  → return `[]`, `{}`, `None`, or `False`. Never raise out of these helpers.
- The bridge's HTTP send endpoint does NOT support `ContextInfo`/quoted
  messages today. We pass `reply_to` forward-compatibly so that when the
  bridge gains support, no client change is needed.
- `_AMBIENT_ANTILOOP_MARKER` (U+200B) is prefixed only when `anti_loop=True`.
  The listener bot strips this prefix and ignores the message — without it
  we'd loop on our own outputs.

## Why deferred imports
Several helpers (`_silent_log`, `_summary_client`, `HELPER_MODEL`,
`HELPER_OPTIONS`, `OLLAMA_KEEP_ALIVE`, `_AMBIENT_ANTILOOP_MARKER`,
`AMBIENT_WHATSAPP_BRIDGE_URL`, `_RAG_LOG_DIR`, `_fetch_contact`,
`_parse_bridge_timestamp`, `INBOX_FOLDER`) live in `rag.__init__`.
Module-level imports here would deadlock the package load; function-body
imports run after `rag.__init__` finishes loading and respect runtime
monkey-patches (`monkeypatch.setattr(rag, "_X", ...)` works because each
call re-resolves the attribute on the `rag` module).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────
WHATSAPP_NOTE_MAX_CHARS = 4096  # WA hard limit per message
WHATSAPP_BRIDGE_DB_PATH = (
    Path.home()
    / "repositories" / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
)
WHATSAPP_DB_PATH = Path.home() / "repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db"
WHATSAPP_BOT_JID = "120363426178035051@g.us"  # RagNet — bot's own group, skip

WA_TASKS_STATE_PATH = Path.home() / ".local/share/obsidian-rag/wa_tasks_state.json"
WA_TASKS_LOG_PATH = Path.home() / ".local/share/obsidian-rag/wa_tasks.jsonl"
# How many chats × messages per run. Conservative: one LLM call per chat
# so the cap bounds cost. Chats with <2 inbound msgs in the window skip
# (not enough signal for extraction).
WA_TASKS_MAX_CHATS = 12
WA_TASKS_MAX_MSGS_PER_CHAT = 40
WA_TASKS_MIN_INBOUND = 2

WA_CROSS_REF_LIMIT = 3


# ── Send path ────────────────────────────────────────────────────────────────


def _ambient_whatsapp_send(jid: str, text: str) -> bool:
    """Fire-and-forget al bridge local de WhatsApp. Retorna True en 2xx.

    POSTea a `http://localhost:8080/api/send` con body
    `{recipient: <jid>, message: <text>}`. El listener del bot RAG
    filtra mensajes que arrancan con U+200B (anti-loop) — se prefixa
    acá para evitar que nuestro propio output se procese como query.
    """
    return _whatsapp_send_to_jid(jid, text, anti_loop=True)


def _whatsapp_send_to_jid(
    jid: str,
    text: str,
    *,
    anti_loop: bool = True,
    reply_to: dict | None = None,
) -> bool:
    """Low-level POST al bridge local. Dos modos:

    - ``anti_loop=True`` (default, usado por ``_ambient_whatsapp_send``):
      prefixa U+200B para que el listener del bot RAG ignore el mensaje
      como query entrante. Necesario cuando el bot se manda cosas a su
      propio grupo (briefs matutinos, archive pushes, etc.).
    - ``anti_loop=False``: texto literal. Usalo cuando el destinatario
      es un contacto tercero (mensajes iniciados desde el chat del user
      vía ``propose_whatsapp_send``), porque el prefix se vería como un
      char raro en el WhatsApp del contacto.

    ``reply_to`` (optional): cuando el caller quiere responder a un
    mensaje específico con quote nativo de WhatsApp. Shape esperado:
    ``{"message_id": str, "original_text": str, "sender_jid": str?}``.

    Estado actual: el bridge local (whatsapp-mcp/whatsapp-bridge,
    `main.go:707-771`) **NO soporta ``ContextInfo``/``QuotedMessage``**
    out of the box — `SendMessageRequest` solo acepta
    ``{recipient, message, media_path}`` y construye `msg.Conversation`
    plano. Por eso pasamos el ``reply_to`` al payload pero el bridge lo
    ignora silenciosamente; el mensaje sale como reply normal sin la
    cita boxed que ves en la UI nativa de WhatsApp. La info igualmente
    se loguea via el caller (auditoría + traceability) y la UI del
    chat web muestra el contexto del mensaje original al user.

    Cuando el bridge agregue soporte de quote, este helper ya pasa el
    campo — bumpean el bridge y empiezan a salir las citas nativas sin
    cambiar el cliente.

    Retorna True en 2xx del bridge, False en cualquier otra cosa
    (unreachable, 4xx, 5xx, timeout 10s).
    """
    from rag import AMBIENT_WHATSAPP_BRIDGE_URL, _AMBIENT_ANTILOOP_MARKER
    import urllib.request
    payload_text = text
    if anti_loop and not text.startswith(_AMBIENT_ANTILOOP_MARKER):
        payload_text = _AMBIENT_ANTILOOP_MARKER + text
    body: dict = {
        "recipient": jid,
        "message": payload_text,
    }
    if reply_to and isinstance(reply_to, dict):
        # Forward-compatible: el bridge actual ignora estos campos pero
        # cuando agreguen ContextInfo los va a leer sin necesidad de
        # tocar el cliente. Ver docstring arriba.
        rt_id = reply_to.get("message_id") or reply_to.get("id")
        if rt_id:
            body["reply_to"] = {
                "message_id": str(rt_id),
                "original_text": str(reply_to.get("original_text") or reply_to.get("text") or "")[:1024],
                "sender_jid": str(reply_to.get("sender_jid") or reply_to.get("from_jid") or ""),
            }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        AMBIENT_WHATSAPP_BRIDGE_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


_GROUP_PREFIX_RE = re.compile(r"^\s*(?:grupo|group|gpo|gpe?o?)\s*:?\s+", re.IGNORECASE)


# ── My Card relationships → personName resolver ─────────────────────────────
# Cuando el user dice "mandale a Mama" / "decile a papá", el contacto real
# probablemente está bajo otro nombre (ej. "Mamá" con tilde, "Carmen", etc.).
# Apple Contacts permite definir Related Names en tu My Card — ahí seteás
# "Madre → Mamá", "Padre → Carlos", etc. Esta función lee ese mapping vía
# osascript y resuelve "mama" → "Mamá" antes de fallar con `not_found`.
#
# El listener WhatsApp (`whatsapp-listener/listener.ts:4385+`) ya tiene esto
# implementado para el flow de voz. Acá lo replicamos para que el chat web
# (vía `propose_whatsapp_send`) tenga la misma capacidad.

# Spanish/English alias → canonical Apple Contacts label.
# Keys YA están normalizados (lowercase + sin acentos).
_RELATIONSHIP_HINT_MAP: dict[str, str] = {
    # mother
    "mama": "mother", "mami": "mother", "mamita": "mother",
    "madre": "mother", "mom": "mother", "mother": "mother",
    # father
    "papa": "father", "papi": "father", "papito": "father",
    "padre": "father", "dad": "father", "father": "father",
    # siblings
    "hermana": "sister", "herma": "sister", "sis": "sister", "sister": "sister",
    "hermano": "brother", "hermo": "brother", "bro": "brother", "brother": "brother",
    # partner
    "esposa": "wife", "mujer": "wife", "wife": "wife",
    "esposo": "husband", "marido": "husband", "husband": "husband",
    # children
    "hijo": "son", "son": "son",
    "hija": "daughter", "daughter": "daughter",
    # grandparents
    "abuela": "grandmother", "abue": "grandmother",
    "grandmother": "grandmother", "granny": "grandmother",
    "abuelo": "grandfather", "abu": "grandfather", "grandfather": "grandfather",
    # in-laws / extended
    "suegra": "motherInLaw", "suegro": "fatherInLaw",
    "tia": "aunt", "tio": "uncle",
    "prima": "cousin", "primo": "cousin",
}

# Apple's localized Spanish labels also resolve to canonical English.
# `_$!<Mother>!$_` is the "raw" label for the English-locale Contacts; for
# es-AR/es-ES users the same field can be stored as "Madre" plain.
_APPLE_LABEL_ES_TO_EN: dict[str, str] = {
    "madre": "mother", "padre": "father",
    "hermana": "sister", "hermano": "brother",
    "esposa": "wife", "esposo": "husband",
    "marido": "husband", "mujer": "wife",
    "hija": "daughter", "hijo": "son",
    "abuela": "grandmother", "abuelo": "grandfather",
    "suegra": "motherInLaw", "suegro": "fatherInLaw",
    "tia": "aunt", "tio": "uncle",
    "prima": "cousin", "primo": "cousin",
}


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
        if ch in ("\u200d", "\u200c"):
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


# Cache the My Card → relations dump. Contacts changes are rare; refresh
# every hour is more than enough.
_MY_CARD_RELATIONS_CACHE: dict | None = None
_MY_CARD_RELATIONS_TTL_S = 3600


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


# Strip de prefijos posesivos comunes que el LLM deja en `contact_name`
# cuando el user dice "mandale a mi mama" / "decile a mi hermana ..." —
# el LLM a veces preserva "mi" en el arg en vez de pasar solo "mama".
# Sin este strip, "mi Mama" no matchea ningún alias y cae a not_found,
# aunque resolverlo sea trivial.
#
# Cubre: "mi", "a mi", "la", "el" al inicio (sin acento o con). NO toca
# nombres reales que casualmente arrancan con "mi" (ej. "Miguel") porque
# requiere el espacio después. "Miguel" → no strip; "mi Hermana" → strip.
_POSSESSIVE_PREFIX_RE = re.compile(
    r"^\s*(?:a\s+)?(?:mi|m[ií]a|mio|m[ií]o|tu|su|el|la)\s+",
    re.IGNORECASE,
)


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


# ── Vault contacts (`99-Contacts/`) — fuente PRIMARIA ──────────────────────
# Las notas en `04-Archive/99-obsidian-system/99-Contacts/` son la fuente
# autoritativa de contactos del user — escritas a mano con teléfono real,
# alias, relación, apellido completo. Tienen prioridad sobre Apple Contacts
# y My Card resolver porque acá NO hay ambigüedad: "Mama.md" es la mamá del
# user, "Maria.md" es la esposa, etc. (decisión del user 2026-04-26).
#
# Formato esperado (ver `_template.md` en esa carpeta):
#   ---
#   aliases: [Sebastián, Sebastian Serra]   # opcional, en frontmatter YAML
#   ---
#   [[NombreArchivo|@Alias]]
#   - **Relación**: Mamá
#   - **Apellido / nombre completo**: Monica Ferrari
#   - **Teléfono**: +54 9 3425476623

VAULT_CONTACTS_SUBPATH = "04-Archive/99-obsidian-system/99-Contacts"

# Cache del scan completo del directorio. TTL bajo porque el user puede
# editar las notas en cualquier momento — invalidación cada 60s evita
# stale data en sesiones largas. Cold call ~5-20ms (8 archivos chicos).
_VAULT_CONTACTS_CACHE: dict | None = None
_VAULT_CONTACTS_TTL_S = 60


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
    aliases, relation_label}`. Campos vacíos default a "" o [].
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
        pattern = (
            r"^-\s*\*\*\s*"
            + label
            + r"\s*\*\*\s*:\s*(.+)$"
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

    return {
        "full_name": full_name or path.stem,
        "phones": phones,
        "emails": emails,
        "birthday": birthday,
        "source": "vault",
        "aliases": aliases,
        "relation_label": relation,
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
    safe = person_name.replace('"', '\\"')
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
"""Patrón de prefijo explícito para forzar lookup de grupo. Matchea
'grupo X', 'group X', 'gpo X', 'grupo: X' (con o sin dos puntos)."""


def _whatsapp_resolve_group_jid(query: str, *, max_candidates: int = 3) -> dict:
    """Resolve un nombre de grupo a `<grupo_id>@g.us` mirando la tabla
    ``chats`` del bridge SQLite. Apple Contacts NO conoce los grupos
    de WhatsApp — solo los conoce el bridge, que los mantiene
    sincronizados con el cliente WhatsApp del user.

    Estrategia de matching:
      1. Match exacto case-insensitive (`name = ?`) — gana si existe.
      2. Match por substring case-insensitive (`name LIKE '%query%'`).
      3. Si hay >1 match, retorna error con los `max_candidates` más
         recientes (por `last_message_time`) para que el caller le
         pida al user que desambigüe.
      4. Si 0 match → error="not_found".

    Filtramos a JIDs `@g.us` exclusivamente — el resolver de 1:1 ya
    cubre `@s.whatsapp.net`. Skipea status broadcast (`status@broadcast`).

    Returns el mismo shape que `_whatsapp_jid_from_contact()` pero con
    `is_group=True` y `phones=[]`. ``candidates`` viene populado solo
    cuando hay ambigüedad para que el frontend la muestre.
    """
    import rag as _rag
    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        return {"jid": None, "full_name": None, "phones": [], "is_group": True,
                "error": "bridge_db_unavailable"}
    q = (query or "").strip()
    if not q:
        return {"jid": None, "full_name": None, "phones": [], "is_group": True,
                "error": "empty_query"}
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error as e:
        return {"jid": None, "full_name": None, "phones": [], "is_group": True,
                "error": f"bridge_db_open_failed: {str(e)[:80]}"}
    try:
        con.row_factory = sqlite3.Row
        # 1) Exact match — gana siempre.
        rows = con.execute(
            "SELECT jid, name, last_message_time FROM chats "
            "WHERE jid LIKE '%@g.us' AND lower(name) = lower(?) "
            "ORDER BY last_message_time DESC LIMIT 5",
            (q,),
        ).fetchall()
        if not rows:
            # 2) Substring fuzzy.
            rows = con.execute(
                "SELECT jid, name, last_message_time FROM chats "
                "WHERE jid LIKE '%@g.us' AND lower(name) LIKE lower(?) "
                "ORDER BY last_message_time DESC LIMIT 10",
                (f"%{q}%",),
            ).fetchall()
    except sqlite3.Error as e:
        return {"jid": None, "full_name": None, "phones": [], "is_group": True,
                "error": f"bridge_db_query_failed: {str(e)[:80]}"}
    finally:
        con.close()

    if not rows:
        return {"jid": None, "full_name": None, "phones": [], "is_group": True,
                "error": "not_found"}
    if len(rows) == 1:
        r = rows[0]
        return {
            "jid": r["jid"],
            "full_name": r["name"],
            "phones": [],
            "is_group": True,
            "error": None,
        }
    # >1 match → ambigüedad. Devolvemos `candidates` para que el LLM /
    # frontend desambigüe. Cap a `max_candidates`.
    candidates = [
        {"jid": r["jid"], "name": r["name"]}
        for r in rows[:max_candidates]
    ]
    return {
        "jid": None,
        "full_name": None,
        "phones": [],
        "is_group": True,
        "candidates": candidates,
        "error": "ambiguous",
    }


def _whatsapp_jid_from_contact(contact_name: str) -> dict:
    """Resolve a contact name ("Grecia", "Oscar (Tela mosquitera)",
    "grupo Random") to a WhatsApp JID. Tries Apple Contacts first (1:1
    chats `@s.whatsapp.net`), and if that fails, falls back to the
    bridge DB chats table for groups (`@g.us`).

    Forced group lookup: si el query empieza con "grupo X", "group X"
    o variantes (case-insensitive), saltamos directo al group resolver
    sin pasar por Contacts. Útil cuando el user tiene un contacto Y un
    grupo con el mismo nombre y quiere específicamente el grupo.

    Returns para 1:1::

        {"jid": "5491234567890@s.whatsapp.net",
         "full_name": "Grecia Ferrari",
         "phones": ["+54 9 11 ..."],
         "is_group": False,
         "error": None}

    Returns para grupo::

        {"jid": "120363426178035051@g.us",
         "full_name": "RagNet",
         "phones": [],
         "is_group": True,
         "error": None}

    Errores incluyen ``not_found`` (ni Contacts ni grupos), ``no_phone``
    (1:1 sin phone), ``ambiguous`` (>1 grupo matchea, viene con
    ``candidates``), ``empty_query``.
    """
    # Deferred attribute lookup so tests `monkeypatch.setattr(rag, "_fetch_contact", ...)`
    # take effect — patches live on `rag.__init__`, not on this module.
    import rag as _rag
    query = (contact_name or "").strip()
    # Strip leading `@` that the LLM sometimes emits for contact names —
    # habit from Obsidian wikilinks `@Person` and Twitter-style mentions.
    # Apple Contacts doesn't care about the sigil; we do.
    if query.startswith("@"):
        query = query.lstrip("@").strip()
    if not query:
        return {"jid": None, "full_name": None, "phones": [],
                "is_group": False, "error": "empty_query"}

    # Forced group lookup: si el user / LLM puso "grupo X" explícito,
    # saltamos Contacts y vamos directo al bridge. La búsqueda en
    # Contacts no encuentra grupos (no existen ahí) así que el fallback
    # eventualmente igual termina en groups; el prefix solo ahorra el
    # round-trip a osascript.
    forced_group = bool(_GROUP_PREFIX_RE.match(query))
    if forced_group:
        stripped = _GROUP_PREFIX_RE.sub("", query, count=1).strip()
        return _whatsapp_resolve_group_jid(stripped or query)

    # PRIMERA FUENTE: vault `99-Contacts/`. Las notas que el user escribió
    # a mano son la verdad autoritativa — `Mama.md` es la mamá del user,
    # `Maria.md` es la esposa, sin ambigüedad de fuzzy match. Si encuentra
    # acá, ni siquiera consultamos Apple Contacts. Pedido del user
    # 2026-04-26: "en el primer lugar que tiene que buscar el contactos
    # es aca [...] /99-Contacts".
    #
    # Crítico: si vault encuentra match SIN phone (placeholder "+54 9 ..."
    # filtrado, o campo Teléfono vacío), igual usamos vault — el downstream
    # va a devolver `error="no_phone"` con el `full_name` del vault, así el
    # user sabe "Astor está en mi vault pero falta el teléfono" en lugar de
    # mandarle accidentalmente a un contacto distinto de Apple Contacts
    # que casualmente matchea por substring (ej. "Astor" → "Psicopedagoga
    # Astor"). El vault es autoritativo aunque esté incompleto.
    contact = None
    try:
        vault_match = _lookup_vault_contact(query)
        if vault_match:
            contact = {
                "full_name": vault_match.get("full_name") or query,
                "phones": vault_match.get("phones", []),
                "emails": vault_match.get("emails", []),
                "birthday": vault_match.get("birthday", ""),
            }
    except Exception:
        contact = None

    # Reuse the existing osascript-backed contact lookup. Passes `query`
    # as the stem — _fetch_contact will try canonical match, first name,
    # and finally the raw stem against Contacts.app.
    #
    # Importante: `_fetch_contact` tiene un guard que SKIPEA stems de
    # parentesco ("Mama"/"Papa"/"Hermana"/...) porque buscarlos por
    # name-contains genera falsos positivos ("Carmen Mama Bianca" ↩).
    # Eso significa que para "Mama" el lookup retorna None aunque el
    # contacto exista (típicamente bajo "Mamá" con tilde). Resolución
    # correcta: leer Related Names de tu My Card → "Madre → Mamá" → re-
    # llamar al lookup con el personName real.
    if not contact:
        try:
            contact = _rag._fetch_contact(query, email=None, canonical=query)
        except Exception as exc:
            return {"jid": None, "full_name": None, "phones": [],
                    "is_group": False,
                    "error": f"lookup_failed: {str(exc)[:80]}"}

    # Si el primer intento no encontró nada Y el query es un alias de
    # parentesco, probar la resolución vía My Card antes de dar up.
    if not contact:
        resolved_name = _resolve_via_my_card_relationship(query)
        if resolved_name:
            # Intentar EXACT match primero (cuando viene de Related Names,
            # el user tipeó el nombre completo del contacto). Sin esto,
            # `_fetch_contact("Maria")` cae a "name contains" y puede
            # matchear "Mariano Di Maggio" antes que el "Maria <Apellido>"
            # real — bug observado 2026-04-26 con `_fetch_contact("mi Esposa")`.
            contact = _exact_contact_lookup(resolved_name)
            # Si exact falla, fallback a fuzzy estándar.
            if not contact:
                try:
                    contact = _rag._fetch_contact(
                        resolved_name, email=None, canonical=resolved_name,
                    )
                except Exception:
                    contact = None

    if contact:
        phones = list(contact.get("phones") or [])
        if not phones:
            return {"jid": None, "full_name": contact.get("full_name"),
                    "phones": [], "is_group": False, "error": "no_phone"}
        digits = re.sub(r"\D+", "", phones[0])
        if not digits:
            return {"jid": None, "full_name": contact.get("full_name"),
                    "phones": phones, "is_group": False, "error": "no_phone"}
        return {
            "jid": f"{digits}@s.whatsapp.net",
            "full_name": contact.get("full_name") or query,
            "phones": phones,
            "is_group": False,
            "error": None,
        }

    # Contacts miss → fallback a grupos. Mejor "encontré un grupo con
    # ese nombre" que "not_found" — el user puede tener ambos contactos
    # en su cabeza (humanos + grupos) y el LLM no sabe a priori cuál es.
    group = _whatsapp_resolve_group_jid(query)
    if group.get("jid") or group.get("error") == "ambiguous":
        return group

    # Ni Contacts ni grupos → not_found definitivo.
    return {"jid": None, "full_name": None, "phones": [],
            "is_group": False, "error": "not_found"}


def _whatsapp_resolve_reply_target(
    contact_name: str,
    when_hint: str | None = None,
    *,
    db_path: Path | str | None = None,
    keyword: str | None = None,
) -> dict:
    """Resolve a "responder a X" request to a concrete WhatsApp message.

    Pipeline:
      1. ``_whatsapp_jid_from_contact(contact_name)`` → JID candidates.
      2. ``_parse_when_hint(when_hint)`` → (low, high, kind) window.
      3. Scan ``messages.db`` for last inbound (``is_from_me=0``) message
         in the contact's 1:1 chat that fits the window. Optional
         ``keyword`` substring match (case-insensitive) on the content,
         útil cuando el hint trae una palabra clave ("del almuerzo",
         "del médico", "del cumple").
      4. Return ``{"message_id", "text", "ts", "ts_iso", "from_jid",
         "chat_jid", "warning"?}`` o ``{"error": ...}``.

    Returns shape:
      - hit:    ``{"message_id", "text", "ts", "ts_iso", "from_jid",
                 "chat_jid", "when_kind", "candidates_seen"}``
      - miss:   ``{"error": "no_match", "candidates_seen": int,
                 "contact_full_name": str, "when_kind": str}``
      - error:  ``{"error": "<reason>"}``

    Personal 1:1 chats only (chat_jid `<digits>@s.whatsapp.net`). Group
    replies (`@g.us`) intencionalmente NO soportadas — la UX de "respondele
    a Juan" en grupos es ambigua (Juan podría tener varios mensajes en
    chats distintos). Defer hasta que el user lo pida.
    """
    import rag as _rag
    _parse_bridge_timestamp = _rag._parse_bridge_timestamp
    _parse_when_hint = _rag._parse_when_hint
    cn = (contact_name or "").strip()
    if not cn:
        return {"error": "empty_contact"}
    try:
        lookup = _whatsapp_jid_from_contact(cn)
    except Exception as exc:
        return {"error": f"contact_lookup_failed: {str(exc)[:80]}"}
    if not lookup.get("jid"):
        err = lookup.get("error") or "not_found"
        return {"error": f"contact_{err}", "contact_full_name": lookup.get("full_name")}

    # Reply-to en grupos NO está soportado todavía: el resolver de
    # mensajes en grupo requeriría más decisiones UX (quotear a qué
    # miembro? respuesta colectiva?) y el bridge tiene shape distinto
    # (sender_jid != chat_jid). Por ahora retornamos error claro para
    # que el frontend muestre warning. El user puede usar
    # `propose_whatsapp_send` con prefijo "grupo X" para mandar SIN
    # quote, que sí funciona.
    if lookup.get("is_group"):
        return {
            "error": "reply_to_groups_not_supported",
            "contact_full_name": lookup.get("full_name"),
            "is_group": True,
        }

    primary_jid = lookup["jid"]
    full_name = lookup.get("full_name") or cn

    # Build last-10-digit suffix candidates to match `chat_jid` flexibly:
    # Apple Contacts may have "+5491155555555" while bridge stores
    # "5491155555555@s.whatsapp.net" — both end in the same 10 digits.
    suffixes: set[str] = set()
    primary_local = primary_jid.split("@")[0]
    d = re.sub(r"\D+", "", primary_local)
    if len(d) >= 8:
        suffixes.add(d[-10:] if len(d) >= 10 else d)
    for ph in (lookup.get("phones") or []):
        d2 = re.sub(r"\D+", "", ph or "")
        if len(d2) >= 8:
            suffixes.add(d2[-10:] if len(d2) >= 10 else d2)

    low, high, when_kind = _parse_when_hint(when_hint)

    import sqlite3 as _sqlite3
    # Deferred lookup so tests `monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", ...)`
    # take effect — patches live on `rag.__init__`, not on this module.
    db = Path(db_path) if db_path else _rag.WHATSAPP_BRIDGE_DB_PATH
    if not db.exists():
        return {"error": f"bridge_db_missing: {db}"}

    try:
        conn = _sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except _sqlite3.Error as exc:
        return {"error": f"bridge_db_open_failed: {str(exc)[:80]}"}

    try:
        # Pull recent inbound messages from any chat whose JID local-part
        # ends in one of our suffixes. We do this client-side to keep the
        # SQL portable and bounded — practical inbound volume per contact
        # is in the low thousands so a 200-row scan is plenty.
        cur = conn.execute(
            "SELECT id, chat_jid, sender, content, timestamp "
            "FROM messages "
            "WHERE is_from_me = 0 "
            "  AND chat_jid LIKE '%@s.whatsapp.net' "
            "  AND content IS NOT NULL AND content != '' "
            "ORDER BY timestamp DESC "
            "LIMIT 500"
        )
        rows = cur.fetchall()
    except _sqlite3.Error as exc:
        try:
            conn.close()
        except Exception:
            pass
        return {"error": f"bridge_db_query_failed: {str(exc)[:80]}"}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    kw = (keyword or "").strip().lower() or None
    candidates_seen = 0
    best: dict | None = None
    for mid, chat_jid, sender, content, ts_raw in rows:
        local = (chat_jid or "").split("@")[0]
        ldigits = re.sub(r"\D+", "", local)
        if not ldigits:
            continue
        if suffixes and not any(ldigits.endswith(s) for s in suffixes):
            continue
        ts = _parse_bridge_timestamp(ts_raw)
        if ts is None:
            continue
        if low is not None and ts < low:
            continue
        if high is not None and ts >= high:
            continue
        candidates_seen += 1
        if kw and kw not in (content or "").lower():
            continue
        # Rows are ordered by timestamp DESC, so first match in window = newest.
        best = {
            "message_id": mid,
            "text": content or "",
            "ts": ts,
            "ts_iso": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "from_jid": chat_jid,
            "chat_jid": chat_jid,
            "sender": sender or "",
        }
        break

    if best is None:
        return {
            "error": "no_match",
            "candidates_seen": candidates_seen,
            "contact_full_name": full_name,
            "when_kind": when_kind,
        }
    best["when_kind"] = when_kind
    best["candidates_seen"] = candidates_seen
    best["contact_full_name"] = full_name
    return best


# ── Read path: unread + windowed scan ────────────────────────────────────────


def _fetch_whatsapp_unread(hours: int = 24, max_chats: int = 8) -> list[dict]:
    """Inbound WhatsApp messages in the last `hours`, grouped by chat.

    Skips the bot's own group and status broadcasts. Returns a list of
    ``{"name": str, "jid": str, "count": int, "last_snippet": str}``
    sorted by message count desc.

    Entries whose `chats.name` is missing or purely digits (typical of
    `@lid` participants whose profile isn't resolved) are dropped — the
    raw phone-number-like JID pollutes briefs. SQL fetches 3× the needed
    cap so filtered entries don't under-populate the final list.
    """
    # Deferred lookup so tests `monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", ...)`
    # take effect — patches live on `rag.__init__`, not on this module.
    import rag as _rag
    db_path = _rag.WHATSAPP_DB_PATH
    bot_jid = _rag.WHATSAPP_BOT_JID
    if not db_path.is_file():
        return []
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
              m.chat_jid AS jid,
              (SELECT name FROM chats WHERE jid = m.chat_jid) AS name,
              count(*) AS cnt,
              (SELECT content FROM messages
                 WHERE chat_jid = m.chat_jid AND is_from_me = 0
                 ORDER BY datetime(timestamp) DESC LIMIT 1) AS last_content
            FROM messages m
            WHERE m.is_from_me = 0
              AND datetime(m.timestamp) > datetime('now', ?)
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
            GROUP BY m.chat_jid
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (f"-{int(hours)} hours", bot_jid, int(max_chats) * 3),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: list[dict] = []
    for r in rows:
        raw_name = (r["name"] or "").strip()
        jid_prefix = (r["jid"] or "").split("@")[0]
        display_name = raw_name or jid_prefix
        # Drop unnamed contacts (raw phone-number-like JIDs). A "real" name
        # has at least one non-digit character; "Grecia's group" passes,
        # "255804326297735" doesn't.
        if not any(ch.isalpha() for ch in display_name):
            continue
        snippet = (r["last_content"] or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        out.append({
            "jid": r["jid"],
            "name": display_name,
            "count": int(r["cnt"] or 0),
            "last_snippet": snippet,
        })
        if len(out) >= max_chats:
            break
    return out


def _wa_chat_label(raw_name: str, jid: str) -> str:
    """Human-readable chat label. Returns the stored name if it has at least
    one alpha character, else `Contacto …<last4>` from the JID prefix.
    Mirrors the filter in `_fetch_whatsapp_unread` so morning and the
    extractor surface the same set of chats.
    """
    name = (raw_name or "").strip()
    if any(ch.isalpha() for ch in name):
        return name
    prefix = (jid or "").split("@")[0]
    tail = prefix[-4:] if len(prefix) >= 4 else prefix
    return f"Contacto …{tail}" if tail else "Contacto"


def _fetch_whatsapp_window(
    since_ts: datetime | None,
    now_ts: datetime,
    processed_ids: set[str],
) -> list[dict]:
    """Per-chat conversation windows since `since_ts` (or last 24h if None).

    Each entry: ``{"jid", "label", "is_group", "inbound": int,
    "messages": [{"id", "ts", "who", "text", "is_from_me"}]}``. Outbound
    messages are included for LLM context but don't count toward inbound
    threshold. Chats below `WA_TASKS_MIN_INBOUND` are dropped. Skips the
    bot's own group, status broadcasts, and unnamed contacts (same filter
    as `_fetch_whatsapp_unread`).

    `processed_ids` deduplicates across runs: messages already extracted
    are filtered out, but we still fetch them because the LLM may need
    the surrounding context.
    """
    # Deferred lookup so tests `monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", ...)`
    # take effect — patches live on `rag.__init__`, not on this module.
    import rag as _rag
    db_path = _rag.WHATSAPP_DB_PATH
    bot_jid = _rag.WHATSAPP_BOT_JID
    min_inbound = _rag.WA_TASKS_MIN_INBOUND
    max_msgs = _rag.WA_TASKS_MAX_MSGS_PER_CHAT
    max_chats = _rag.WA_TASKS_MAX_CHATS
    if not db_path.is_file():
        return []
    since = since_ts or (now_ts - timedelta(hours=24))
    since_iso = since.strftime("%Y-%m-%d %H:%M:%S")
    import sqlite3
    try:
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5.0,
        )
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
              m.id AS id,
              m.chat_jid AS jid,
              m.sender AS sender,
              m.content AS content,
              m.timestamp AS ts,
              m.is_from_me AS is_from_me,
              m.media_type AS media_type,
              c.name AS chat_name
            FROM messages m
            LEFT JOIN chats c ON c.jid = m.chat_jid
            WHERE datetime(m.timestamp) >= datetime(?)
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
            ORDER BY m.timestamp ASC
            """,
            (since_iso, bot_jid),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    by_chat: dict[str, dict] = {}
    for r in rows:
        jid = r["jid"] or ""
        label = _wa_chat_label(r["chat_name"] or "", jid)
        # Drop unnamed contacts — same policy as morning brief.
        if label.startswith("Contacto …") and not any(ch.isalpha() for ch in (r["chat_name"] or "")):
            continue
        content = (r["content"] or "").strip().replace("\n", " ")
        if not content and r["media_type"]:
            content = f"[{r['media_type']}]"
        if not content:
            continue
        is_from_me = bool(r["is_from_me"])
        who = "yo" if is_from_me else (r["sender"] or "").split("@")[0] or label
        entry = by_chat.setdefault(jid, {
            "jid": jid,
            "label": label,
            "is_group": jid.endswith("@g.us"),
            "inbound": 0,
            "messages": [],
            "new_ids": [],
        })
        msg_id = r["id"] or ""
        new = msg_id and msg_id not in processed_ids
        if not is_from_me:
            entry["inbound"] += 1
        entry["messages"].append({
            "id": msg_id,
            "ts": r["ts"] or "",
            "who": who,
            "text": content[:400],
            "is_from_me": is_from_me,
            "new": new,
        })
        if new:
            entry["new_ids"].append(msg_id)

    out: list[dict] = []
    for entry in by_chat.values():
        if entry["inbound"] < min_inbound:
            continue
        # Skip chats with no *new* inbound messages — purely-read context,
        # nothing to extract. (new_ids includes outbound; re-filter.)
        new_inbound = sum(
            1 for m in entry["messages"] if m["new"] and not m["is_from_me"]
        )
        if new_inbound == 0:
            continue
        # Keep the tail window — extraction cares about recent state.
        entry["messages"] = entry["messages"][-max_msgs:]
        out.append(entry)

    out.sort(key=lambda e: e["inbound"], reverse=True)
    return out[:max_chats]


# ── wa-tasks state + extractor ───────────────────────────────────────────────


def _fetch_whatsapp_recent_with_jid(jid: str, limit: int = 5) -> dict:
    """Últimos ``limit`` mensajes intercambiados con ``jid`` para mostrar
    contexto en el card del chat antes de mandar/programar.

    Distinto de ``_fetch_whatsapp_window``:
      - No filtra por timestamp (devuelve los últimos N independientemente
        de cuándo fueron — útil cuando hace meses no hablan).
      - Filtra por ``chat_jid`` específico (no batch por chat).
      - Devuelve los mensajes en orden cronológico ascendente (más viejo
        arriba, más nuevo abajo) — lectura natural en el thread visual.

    Returns ``{jid, messages_count, last_contact_at, messages: [...]}``
    donde cada mensaje es ``{id, ts (ISO8601 con offset Argentina), who,
    text, is_from_me}``. Si el bridge DB no existe o no hay mensajes
    para el JID, devuelve estructura con ``messages_count=0`` y lista
    vacía (no raisea — es un best-effort de UI).

    Privacidad: no se persiste nada de lo retornado — el endpoint que
    consume esto solo refleja al frontend la data del bridge local.
    """
    import rag as _rag
    db_path = _rag.WHATSAPP_DB_PATH
    bot_jid = _rag.WHATSAPP_BOT_JID
    empty = {"jid": jid, "messages_count": 0, "last_contact_at": None, "messages": []}
    if not jid or "@" not in jid:
        return empty
    if not db_path.is_file():
        return empty
    cap = max(1, min(int(limit or 5), 20))
    import sqlite3
    try:
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5.0,
        )
    except sqlite3.Error:
        return empty
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
              m.id AS id,
              m.sender AS sender,
              m.content AS content,
              m.timestamp AS ts,
              m.is_from_me AS is_from_me,
              m.media_type AS media_type,
              c.name AS chat_name
            FROM messages m
            LEFT JOIN chats c ON c.jid = m.chat_jid
            WHERE m.chat_jid = ?
              AND m.chat_jid != ?
            ORDER BY m.timestamp DESC
            LIMIT ?
            """,
            (jid, bot_jid, cap),
        ).fetchall()
    except sqlite3.Error:
        return empty
    finally:
        con.close()

    if not rows:
        return empty

    chat_label = _wa_chat_label((rows[0]["chat_name"] or ""), jid)
    # Bridge guarda timestamps con offset incluido y space separator
    # ("2024-11-28 20:59:45-03:00"). Solo normalizamos el separador a
    # "T" para que sea ISO8601 estricto y `Date.parse` del browser lo
    # acepte sin caprichos.
    messages = []
    for r in reversed(rows):  # asc (más viejo → más nuevo) para lectura natural
        ts_raw = (r["ts"] or "").strip()
        ts_iso = ts_raw.replace(" ", "T") if ts_raw else ""
        is_from_me = bool(r["is_from_me"])
        content = (r["content"] or "").strip().replace("\n", " ")
        media = (r["media_type"] or "").strip()
        if not content and media:
            content = f"[{media}]"
        if not content:
            continue
        who = "yo" if is_from_me else chat_label
        messages.append({
            "id": r["id"] or "",
            "ts": ts_iso,
            "who": who,
            "text": content[:400],  # cap defensivo: el card no necesita más
            "is_from_me": is_from_me,
        })

    last_ts = messages[-1]["ts"] if messages else None
    return {
        "jid": jid,
        "messages_count": len(messages),
        "last_contact_at": last_ts,
        "messages": messages,
    }


def _wa_tasks_load_state() -> dict:
    """Returns `{last_run_ts: iso|null, processed_ids: [id, ...]}`.

    `processed_ids` is a ring of recent message ids (cap 2000) — cheap dedup
    across overlapping windows. `last_run_ts` is the high-water mark; next
    run fetches strictly after it.
    """
    # Deferred lookup so tests `monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", ...)`
    # are honored — the patch lives on `rag.__init__`, not on this module.
    import rag as _rag
    state_path = _rag.WA_TASKS_STATE_PATH
    if not state_path.is_file():
        return {"last_run_ts": None, "processed_ids": []}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_ts": None, "processed_ids": []}
    if not isinstance(data, dict):
        return {"last_run_ts": None, "processed_ids": []}
    data.setdefault("last_run_ts", None)
    data.setdefault("processed_ids", [])
    if not isinstance(data["processed_ids"], list):
        data["processed_ids"] = []
    return data


def _wa_tasks_save_state(state: dict) -> None:
    import rag as _rag
    state_path = _rag.WA_TASKS_STATE_PATH
    ids = state.get("processed_ids") or []
    if len(ids) > 2000:
        state["processed_ids"] = ids[-2000:]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _wa_extract_actions(chat_label: str, is_group: bool, messages: list[dict]) -> dict:
    """LLM-extract action items from a chat window.

    Conservative prompt: only flag items a human would genuinely action.
    Returns ``{"tasks": [str], "questions": [str], "commitments": [str]}``
    (empty lists on LLM failure — callers treat as "nothing to extract",
    not as an error). Deterministic via HELPER_OPTIONS.

    `commitments` are things the user (yo) promised to do; `tasks` are
    asks directed at the user; `questions` are open questions addressed
    to the user that still need an answer.
    """
    from rag import HELPER_MODEL, HELPER_OPTIONS, OLLAMA_KEEP_ALIVE, _summary_client
    empty = {"tasks": [], "questions": [], "commitments": []}
    if not messages:
        return empty
    convo_lines: list[str] = []
    for m in messages:
        ts = (m["ts"] or "")[:16].replace("T", " ")
        convo_lines.append(f"[{ts}] {m['who']}: {m['text']}")
    convo = "\n".join(convo_lines)
    if len(convo) > 6000:
        convo = convo[-6000:]
    kind = "grupo" if is_group else "chat directo"
    prompt = (
        f"Conversación de WhatsApp ({kind}): {chat_label}\n\n"
        f"{convo}\n\n"
        "Extraé solo items accionables reales para \"yo\" (el usuario). "
        "Sé conservador: si no está claro que sea una acción, omitilo. "
        "Ignorá saludos, small talk, memes, reacciones.\n\n"
        "- tasks: cosas que alguien le pidió a yo (hacer X, mandar Y, revisar Z).\n"
        "- questions: preguntas dirigidas a yo que aún no respondió.\n"
        "- commitments: cosas que yo prometió hacer (\"te mando…\", \"mañana te paso…\").\n\n"
        "Cada item: frase corta en español, 1 línea, sin nombre del chat ni timestamps. "
        "Si no hay nada en una categoría, lista vacía. "
        "Formato estricto JSON: "
        "{\"tasks\": [\"...\"], \"questions\": [\"...\"], \"commitments\": [\"...\"]}"
    )
    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 320, "num_ctx": 4096},
            keep_alive=OLLAMA_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    out = {"tasks": [], "questions": [], "commitments": []}
    for key in out:
        items = data.get(key) or []
        if not isinstance(items, list):
            continue
        seen: set[str] = set()
        for item in items[:10]:
            if not isinstance(item, str):
                continue
            clean = item.strip().strip("-•*").strip()
            if len(clean) < 4 or len(clean) > 240:
                continue
            key_norm = clean.lower()
            if key_norm in seen:
                continue
            seen.add(key_norm)
            out[key].append(clean)
    return out


def _wa_chat_month_link(jid: str, label: str, ts_iso: str) -> str:
    """Wikilink to the vault-sync'd chat note for the message's month.

    Falls back to just the label if the month can't be parsed. The link
    target mirrors `whatsapp-to-vault`'s layout:
    `03-Resources/WhatsApp/<slug>/YYYY-MM.md`.
    """
    slug_src = label if any(ch.isalpha() for ch in label) else (jid.split("@")[0] or "sin-nombre")
    # Same slug rule as vault-sync: strip non-word/dash/dot/space.
    slug = re.sub(r"[^\w\-\. ]+", "", slug_src).strip()
    slug = re.sub(r"\s+", " ", slug)[:80] or "sin-nombre"
    try:
        dt = datetime.fromisoformat(ts_iso[:19].replace(" ", "T"))
        ym = dt.strftime("%Y-%m")
    except Exception:
        return f"[[{label}]]"
    return f"[[03-Resources/WhatsApp/{slug}/{ym}|{label}]]"


def _wa_tasks_write_note(
    vault: Path,
    run_ts: datetime,
    by_chat: list[dict],
    extractions: list[dict],
) -> tuple[Path, bool, int]:
    """Append a timestamped section to `00-Inbox/WA-YYYY-MM-DD.md`.

    Creates the file with frontmatter on first write of the day. Later
    runs append under a new `## HH:MM` heading so the same-day history is
    preserved. Returns ``(path, created, new_items)``. If every extraction
    came back empty, writes nothing and returns `(path, False, 0)`.
    """
    from rag import INBOX_FOLDER
    total_items = sum(
        len(e["tasks"]) + len(e["questions"]) + len(e["commitments"])
        for e in extractions
    )
    date_str = run_ts.strftime("%Y-%m-%d")
    note_path = vault / INBOX_FOLDER / f"WA-{date_str}.md"
    if total_items == 0:
        return note_path, False, 0

    lines: list[str] = []
    section = f"## {run_ts.strftime('%H:%M')} — {sum(1 for e in extractions if any(e[k] for k in ('tasks','questions','commitments')))} chats\n"
    lines.append(section)
    for chat, ext in zip(by_chat, extractions):
        if not any(ext[k] for k in ("tasks", "questions", "commitments")):
            continue
        first_new_ts = next(
            (m["ts"] for m in chat["messages"] if m["new"] and not m["is_from_me"]),
            chat["messages"][-1]["ts"] if chat["messages"] else "",
        )
        link = _wa_chat_month_link(chat["jid"], chat["label"], first_new_ts)
        lines.append(f"### {link}\n")
        for t in ext["tasks"]:
            lines.append(f"- [ ] {t}")
        for q in ext["questions"]:
            lines.append(f"- ❓ {q}")
        for c in ext["commitments"]:
            lines.append(f"- 📌 {c}")
        lines.append("")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    created = not note_path.exists()
    if created:
        header = [
            "---",
            "source: whatsapp",
            "type: wa-tasks",
            f"date: {date_str}",
            "ambient: skip",
            "tags:",
            "- whatsapp",
            "- tasks/wa",
            "---",
            "",
            f"# WhatsApp — tareas {date_str}",
            "",
        ]
        body = "\n".join(header + lines) + "\n"
        note_path.write_text(body, encoding="utf-8")
    else:
        existing = note_path.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        note_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    return note_path, created, total_items


def _wa_tasks_plist(rag_bin: str) -> str:
    """WhatsApp action-item extractor — every 30min.

    Reads delta from the bridge SQLite since last run and distills tasks/
    questions/commitments to `00-Inbox/WA-YYYY-MM-DD.md`. Cheap: one
    qwen2.5:3b call per chat with new inbound messages (capped at 12
    chats). `ambient: skip` in the output frontmatter prevents the
    WhatsApp push loop.
    """
    from rag import _RAG_LOG_DIR
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-wa-tasks</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>wa-tasks</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/wa-tasks.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/wa-tasks.error.log</string>
</dict>
</plist>
"""
