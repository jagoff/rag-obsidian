"""Constantes compartidas del sub-paquete `rag.integrations.whatsapp`.

Centraliza paths, caps, JIDs y mappings que múltiples sub-módulos consumen.
Importable sin side-effects: no toca el filesystem ni inicializa singletons.

NOTA: el `__init__.py` del paquete re-exporta CADA constante de acá para
preservar back-compat con `from rag.integrations.whatsapp import WHATSAPP_NOTE_MAX_CHARS`
(usado por `rag/__init__.py:63261-`, `rag/cross_source_patterns.py`, etc.).
"""

from __future__ import annotations

from pathlib import Path


# ── Paths del bridge SQLite + state files ───────────────────────────────────
WHATSAPP_NOTE_MAX_CHARS = 4096  # WA hard limit per message
# Path corregido 2026-05-09: el repo del bridge vive en ~/repos/, no
# ~/repositories/. El path viejo silent-faileaba todos los fetch (today/
# unread/window/recent_with_jid devolvían [] porque is_file()=False) lo
# que vaciaba la sección WhatsApp del morning brief y dejaba sin extraer
# wa-tasks por días.
WHATSAPP_BRIDGE_DB_PATH = (
    Path.home() / "repos" / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"
)
WHATSAPP_DB_PATH = WHATSAPP_BRIDGE_DB_PATH  # alias back-compat
WHATSAPP_BOT_JID = "120363426178035051@g.us"  # RagNet — bot's own group, skip

WA_TASKS_STATE_PATH = Path.home() / ".local/share/obsidian-rag/wa_tasks_state.json"
WA_TASKS_LOG_PATH = Path.home() / ".local/share/obsidian-rag/wa_tasks.jsonl"

# Caps del extractor de wa-tasks. Conservadores: una LLM call por chat,
# así el cost queda acotado. Chats con <2 inbound msgs en la ventana se skipean
# (no hay señal suficiente para extracción).
WA_TASKS_MAX_CHATS = 12
WA_TASKS_MAX_MSGS_PER_CHAT = 40
WA_TASKS_MIN_INBOUND = 2

WA_CROSS_REF_LIMIT = 3


# ── Vault contacts (`99-Contacts/`) — fuente PRIMARIA ────────────────────────
# Las notas en `99-obsidian/99-Contacts/` son la fuente autoritativa de
# contactos del user — escritas a mano con teléfono real, alias, relación,
# apellido completo. Tienen prioridad sobre Apple Contacts y My Card resolver
# porque acá NO hay ambigüedad: "Mama.md" es la mamá del user, "Maria.md"
# es la esposa, etc. (decisión del user 2026-04-26).
VAULT_CONTACTS_SUBPATH = "99-obsidian/99-Contacts"


# ── Relationship hints (Spanish/English alias → canonical Apple label) ──────
# Keys YA están normalizados (lowercase + sin acentos). Consumido por
# `contacts.py` (vault lookup + my-card resolver) y `resolve.py` (forced
# relationship lookup en `_whatsapp_jid_from_contact`).
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


# ── Contact observations (template del vault `_template.md`) ────────────────
# Marker que abre la sección auto-managed. Único + estable para que el parser
# sepa dónde insertar. Si el user reorganiza la nota a mano, preservamos su
# edición — solo modificamos lo que está bajo este marker.
_OBSERVATIONS_HEADING = "## Observaciones"

# Categorías estándar del template de `99-Contacts/_template.md` (2026-04-26)
# más extensiones semánticas que el user agregó en notas reales
# ("Preferencias", "Eventos importantes"). El LLM elige UNA si la observation
# matchea, o "Notas" como default. Si el LLM rebota → None → va solo a
# `## Observaciones`.
_CONTACT_OBS_STANDARD_CATEGORIES = [
    "Trabajo / contexto",
    "Notas",
    "Preferencias",
    "Eventos importantes",
    "Familia",
    "Cumpleaños",
]


__all__ = [
    "WHATSAPP_NOTE_MAX_CHARS",
    "WHATSAPP_BRIDGE_DB_PATH",
    "WHATSAPP_DB_PATH",
    "WHATSAPP_BOT_JID",
    "WA_TASKS_STATE_PATH",
    "WA_TASKS_LOG_PATH",
    "WA_TASKS_MAX_CHATS",
    "WA_TASKS_MAX_MSGS_PER_CHAT",
    "WA_TASKS_MIN_INBOUND",
    "WA_CROSS_REF_LIMIT",
    "VAULT_CONTACTS_SUBPATH",
    "_RELATIONSHIP_HINT_MAP",
    "_APPLE_LABEL_ES_TO_EN",
    "_OBSERVATIONS_HEADING",
    "_CONTACT_OBS_STANDARD_CATEGORIES",
]
