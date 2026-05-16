"""WhatsApp JID resolution — name → JID for 1:1 chats and groups.

Tres surfaces:

- ``_whatsapp_resolve_group_jid(query, *, max_candidates)`` — name → `<id>@g.us`
  mirando la tabla ``chats`` del bridge SQLite. Apple Contacts NO conoce los
  grupos; solo el bridge los mantiene sincronizados con el cliente WA del user.
- ``_whatsapp_jid_from_contact(contact_name)`` — orquestador. Vault primero,
  Apple Contacts después, group fallback. Maneja prefix ``grupo X`` para forzar
  group lookup directamente.
- ``_whatsapp_resolve_reply_target(contact_name, when_hint, ...)`` — resolve
  un "responder a X" a un message_id concreto del bridge.

Why deferred imports (`from rag.integrations.whatsapp import _foo` adentro):
``_lookup_vault_contact``, ``_exact_contact_lookup``,
``_resolve_via_my_card_relationship`` viven en ``contacts.py`` (módulo hermano).
Resolverlos a través del package namespace garantiza que monkeypatches de tests
(``monkeypatch.setattr(_waint, "_foo", mock)``) propaguen al call site.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


# Patrón de prefijo explícito para forzar lookup de grupo. Matchea
# 'grupo X', 'group X', 'gpo X', 'grupo: X' (con o sin dos puntos).
_GROUP_PREFIX_RE = re.compile(r"^\s*(?:grupo|group|gpo|gpe?o?)\s*:?\s+", re.IGNORECASE)


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
    # Re-resolve via package namespace para que monkeypatches sobre
    # `rag.integrations.whatsapp._lookup_vault_contact` /
    # `_exact_contact_lookup` / `_resolve_via_my_card_relationship`
    # propaguen.
    from rag.integrations.whatsapp import (
        _exact_contact_lookup,
        _lookup_vault_contact,
        _resolve_via_my_card_relationship,
        _whatsapp_resolve_group_jid as _resolve_group,
    )
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
        return _resolve_group(stripped or query)

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
    group = _resolve_group(query)
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
    # Re-resolve via package para que monkeypatches en
    # `rag.integrations.whatsapp._whatsapp_jid_from_contact` propaguen.
    from rag.integrations.whatsapp import _whatsapp_jid_from_contact
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


__all__ = [
    "_GROUP_PREFIX_RE",
    "_whatsapp_resolve_group_jid",
    "_whatsapp_jid_from_contact",
    "_whatsapp_resolve_reply_target",
]
