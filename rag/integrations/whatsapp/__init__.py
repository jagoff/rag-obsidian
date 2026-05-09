"""WhatsApp integration sub-package — split modular 2026-05-08.

Re-exporta el surface PÚBLICO del antiguo `rag/integrations/whatsapp.py`
(2394 LOC, 33 functions, Phase 1b 2026-04-26) ahora distribuido en
sub-módulos por responsabilidad:

| Sub-módulo | Responsabilidad |
|---|---|
| `_constants` | Paths, caps, JIDs, mappings de relaciones |
| `send` | POST al bridge + ambient/draft mode |
| `contacts` | Vault `99-Contacts/` + Apple My Card resolver |
| `observations` | Append observaciones a notas de contact |
| `resolve` | name → JID (1:1 + grupos) + reply-target |
| `fetch` | Read paths del bridge SQLite (today/unread/window/recent) |
| `tasks_state` | High-water mark + dedup ring de wa-tasks |
| `tasks_extract` | LLM action items + promises extraction |
| `tasks_writer` | Escribe `00-Inbox/WA-YYYY-MM-DD.md` + chat-month wikilink |
| `plist` | launchd plist generators |
| `scheduled` | Programar envíos a futuro (ex `rag/wa_scheduled.py`) |
| `cli` | Click command `rag wa-tasks` (ex `rag/wa_tasks.py`) |

## Back-compat invariant

Todo lo que el viejo módulo exportaba (constants `WHATSAPP_*`, `WA_*`,
helpers privados `_wa_*` / `_whatsapp_*` / `_fetch_whatsapp_*` / `_load_*`
/ `_lookup_*` / `_parse_*` / `_strip_*` / `_normalize_*` / `_exact_*` /
`_resolve_*` / `_infer_*` / `_append_*` / `_has_promise_*` / regex
`_PROMISE_REGEX_HINTS` y `_GROUP_PREFIX_RE`) sigue importable como
``from rag.integrations.whatsapp import X`` — re-exportado acá.

Crítico para tests con `monkeypatch.setattr(_waint, "_foo", mock)`:
el patch sobrescribe el atributo del package (`_waint.__dict__`). Para
que el patch propague a internal call sites en sub-módulos, los call
sites usan **deferred re-resolve**:
``from rag.integrations.whatsapp import _foo`` adentro del cuerpo de la
función. Cada call re-importa por package namespace, así el patch del test
gana sobre el binding del sub-módulo.

## Sub-módulos `scheduled` y `cli`

Mismo módulo, ubicación nueva. Los shims `rag/wa_scheduled.py` y
`rag/wa_tasks.py` re-exportan desde acá para preservar imports históricos
(``from rag.wa_scheduled import schedule``, ``from rag import wa_scheduled``).
"""

from __future__ import annotations

# ── Constantes ──────────────────────────────────────────────────────────────
from rag.integrations.whatsapp._constants import (  # noqa: F401
    VAULT_CONTACTS_SUBPATH,
    WA_CROSS_REF_LIMIT,
    WA_TASKS_LOG_PATH,
    WA_TASKS_MAX_CHATS,
    WA_TASKS_MAX_MSGS_PER_CHAT,
    WA_TASKS_MIN_INBOUND,
    WA_TASKS_STATE_PATH,
    WHATSAPP_BOT_JID,
    WHATSAPP_BRIDGE_DB_PATH,
    WHATSAPP_DB_PATH,
    WHATSAPP_NOTE_MAX_CHARS,
    _APPLE_LABEL_ES_TO_EN,
    _CONTACT_OBS_STANDARD_CATEGORIES,
    _OBSERVATIONS_HEADING,
    _RELATIONSHIP_HINT_MAP,
)

# ── Send paths ──────────────────────────────────────────────────────────────
from rag.integrations.whatsapp.send import (  # noqa: F401
    _ambient_whatsapp_send,
    _whatsapp_send_to_jid,
)

# ── Contacts (vault + Apple My Card) ────────────────────────────────────────
from rag.integrations.whatsapp.contacts import (  # noqa: F401
    _POSSESSIVE_PREFIX_RE,
    _WA_APPLESCRIPT_SAFE_RE,
    _exact_contact_lookup,
    _load_my_card_relations,
    _load_vault_contacts,
    _lookup_vault_contact,
    _normalize_hint,
    _parse_apple_label,
    _parse_vault_contact,
    _resolve_via_my_card_relationship,
    _strip_emoji_and_symbols,
    _strip_possessive_prefix,
    _vault_contacts_dir,
    _wa_sanitize_applescript_string,
)

# ── Observations ────────────────────────────────────────────────────────────
from rag.integrations.whatsapp.observations import (  # noqa: F401
    _append_contact_observation,
    _infer_observation_category,
)

# ── JID resolution + reply-target ───────────────────────────────────────────
from rag.integrations.whatsapp.resolve import (  # noqa: F401
    _GROUP_PREFIX_RE,
    _whatsapp_jid_from_contact,
    _whatsapp_resolve_group_jid,
    _whatsapp_resolve_reply_target,
)

# ── Read paths ──────────────────────────────────────────────────────────────
from rag.integrations.whatsapp.fetch import (  # noqa: F401
    _fetch_whatsapp_recent_with_jid,
    _fetch_whatsapp_today,
    _fetch_whatsapp_unread,
    _fetch_whatsapp_window,
    _wa_chat_label,
)

# ── Tasks state ─────────────────────────────────────────────────────────────
from rag.integrations.whatsapp.tasks_state import (  # noqa: F401
    _wa_tasks_load_state,
    _wa_tasks_save_state,
)

# ── Tasks extract (LLM) ─────────────────────────────────────────────────────
from rag.integrations.whatsapp.tasks_extract import (  # noqa: F401
    _PROMISE_REGEX_HINTS,
    _has_promise_hint,
    _parse_promise_when,
    _wa_extract_actions,
    _wa_extract_promises,
)

# ── Tasks writer ────────────────────────────────────────────────────────────
from rag.integrations.whatsapp.tasks_writer import (  # noqa: F401
    _wa_chat_month_link,
    _wa_tasks_write_note,
)

# ── Plist generators ────────────────────────────────────────────────────────
from rag.integrations.whatsapp.plist import _wa_tasks_plist  # noqa: F401


__all__ = [
    # Constants
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
    "_POSSESSIVE_PREFIX_RE",
    "_WA_APPLESCRIPT_SAFE_RE",
    "_GROUP_PREFIX_RE",
    "_PROMISE_REGEX_HINTS",
    # Send
    "_ambient_whatsapp_send",
    "_whatsapp_send_to_jid",
    # Contacts
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
    # Observations
    "_infer_observation_category",
    "_append_contact_observation",
    # Resolve
    "_whatsapp_resolve_group_jid",
    "_whatsapp_jid_from_contact",
    "_whatsapp_resolve_reply_target",
    # Fetch
    "_fetch_whatsapp_today",
    "_fetch_whatsapp_unread",
    "_wa_chat_label",
    "_fetch_whatsapp_window",
    "_fetch_whatsapp_recent_with_jid",
    # Tasks state
    "_wa_tasks_load_state",
    "_wa_tasks_save_state",
    # Tasks extract
    "_wa_extract_actions",
    "_has_promise_hint",
    "_parse_promise_when",
    "_wa_extract_promises",
    # Tasks writer
    "_wa_chat_month_link",
    "_wa_tasks_write_note",
    # Plist
    "_wa_tasks_plist",
]
