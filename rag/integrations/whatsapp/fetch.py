"""WhatsApp read paths — pulls del bridge SQLite (read-only).

Surfaces:

- ``_fetch_whatsapp_today(now, max_chats)`` — inbound messages received
  TODAY (today 00:00 local → now), grouped by chat. Para evening brief.
- ``_fetch_whatsapp_unread(hours, max_chats)`` — rolling window de N hours.
  Para morning brief.
- ``_wa_chat_label(raw_name, jid)`` — label legible. Drop names sin alpha
  chars; fallback a ``Contacto …<last4>``.
- ``_fetch_whatsapp_window(since_ts, now_ts, processed_ids)`` — per-chat
  conversation windows para el extractor de wa-tasks (filtra dedup ring).
- ``_fetch_whatsapp_recent_with_jid(jid, limit)`` — últimos N mensajes con
  un JID específico para el card de UI.

Invariantes:
- All paths read-only (`?mode=ro` URI). Bridge nunca se muta.
- Silent-fail: missing DB / locked SQLite / network → `[]` o `{...empty}`.
  Nunca raise.
- Skip bot's own group (``WHATSAPP_BOT_JID``) y ``status@broadcast``.
- Skip exact-name blacklisted chats such as ``Cloud Services`` so they
  never feed briefs/tasks/context enrichment.
- Skip messages que arrancan con U+200B (anti-loop marker — son nuestros
  propios outputs).
- Drop unnamed contacts (raw phone-number-like JIDs) — un nombre "real"
  tiene al menos un char alpha. Filtra `@lid` participants sin perfil
  resuelto.

Why deferred imports (`import rag as _rag` adentro del cuerpo):
``WHATSAPP_DB_PATH``, ``WHATSAPP_BOT_JID``, ``WA_TASKS_*`` constants viven
en `rag/__init__.py` (re-exportadas desde `_constants.py` de este pkg).
Module-level imports deadlock-ean el package load; function-body imports
respetan monkey-patches de tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path

from ._constants import whatsapp_chat_name_excluded


# Bridge guarda `timestamp` como string `YYYY-MM-DD HH:MM:SS-03:00` (Go default
# `time.Time.String()` con offset local del proceso del bridge). Como el offset
# es siempre el mismo (TZ local del user, hoy -03:00 Argentina), la
# string-comparación lex es correcta SI los bounds se formatean igual.
#
# Antes del 2026-05-09 las queries usaban `WHERE datetime(m.timestamp) >= datetime(?)`,
# lo que ROMPÍA el uso del índice `idx_messages_chat_ts` porque SQLite no puede
# usar índices sobre funciones de la columna. Bench medido: 40.96ms → 0.10ms
# (~400x speedup) al pasar a comparación string pura.
#
# Caveat conocido: si el user viaja a otra TZ, el offset del bridge cambia y la
# lex-compare puede fallar para los rows escritos antes del viaje. Mitigación:
# `_bridge_ts_bound` lee `RAG_TIMEZONE` o asume -03:00 — re-evaluar si soporta
# otras TZ. Para 99% del uso (single-TZ), esto es seguro.
_BRIDGE_TS_FMT = "%Y-%m-%d %H:%M:%S"
_BRIDGE_TZ_OFFSET = "-03:00"  # Argentina local — bridge usa esto al persistir

# Cap defensivo: max OCR/VLM calls per `_fetch_whatsapp_window` invocation.
# El bridge guarda media en `<bridge_repo>/store/<chat_jid>/<filename>`.
# OCR (ocrmac) es ~50-200ms/imagen; VLM fallback (granite) ~1-2s/imagen.
# Sin cap, una ráfaga de 30 fotos en un grupo bloquea el wa-tasks tick.
_WA_OCR_MAX_PER_RUN = 5
_WA_OCR_TEXT_MAX_CHARS = 400  # cap del snippet inyectado al LLM context


def _bridge_ts_bound(dt: datetime) -> str:
    """Formatea un `datetime` para comparar con la columna `messages.timestamp`
    del bridge usando string-lex (no `datetime()` SQL function — usa el índice).
    """
    # Truncar microsegundos: el bridge no los guarda y lex string compara
    # byte-a-byte, así que un microsegundo de más cambia el resultado.
    return dt.strftime(_BRIDGE_TS_FMT) + _BRIDGE_TZ_OFFSET


def _bridge_ts_from_ui(ts: str | None) -> str:
    """Normaliza timestamps recibidos desde `/wa` al formato del bridge.

    La UI consume ISO (`YYYY-MM-DDTHH:MM:SS-03:00`), pero la SQLite del
    bridge guarda `YYYY-MM-DD HH:MM:SS-03:00`. Como varias queries hacen
    comparación lexicográfica para usar índices, guardar o comparar con `T`
    rompe unread/paginación dentro del mismo día.
    """
    s = (ts or "").strip()
    if not s:
        return ""
    if "T" in s:
        s = s.replace("T", " ", 1)
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    return s


import contextlib  # noqa: E402


def _bridge_media_path(chat_jid: str, filename: str) -> Path | None:
    """Resuelve el path local donde el bridge guardó un media file.

    El bridge auto-descarga inbound media a `<repo>/store/<chat_jid>/<filename>`.
    Devuelve `None` si el archivo no existe (puede que el bridge falló al
    bajar, o que el filename no matchea).
    """
    if not chat_jid or not filename:
        return None
    import rag as _rag
    db_path = _rag.WHATSAPP_DB_PATH
    if db_path is None:
        return None
    # `messages.db` vive en `<repo>/store/messages.db`. Media en `<repo>/store/<chat_jid>/<filename>`.
    media = db_path.parent / chat_jid / filename
    return media if media.is_file() else None


def _ocr_image_safe(image_path: Path, *, label: str = "wa") -> str:
    """OCR + VLM caption fallback con timeout suave.

    Wrapper sobre `rag.ocr._image_text_or_caption` con silent-fail completo.
    Si OCR/VLM falla (model no disponible, timeout, etc.) devuelve "" — el
    caller pone solo el placeholder `[image]` como antes del wireup A4.

    El cache (`rag_ocr_cache` + `rag_vlm_captions`) ya está implementado en
    `rag.ocr` así que las imágenes recurrentes (memes reenviados, etc.) NO
    repagan el costo OCR.
    """
    try:
        from rag.ocr import _image_text_or_caption  # noqa: PLC0415
        result = _image_text_or_caption(image_path) or ""
        if isinstance(result, tuple):
            text = result[0] or ""
        else:
            text = result or ""
    except Exception as exc:
        try:
            import rag as _rag
            _rag._silent_log(f"wa_ocr_failed_{label}", exc)
        except Exception:
            pass
        return ""
    if not text:
        return ""
    text = str(text).strip().replace("\n", " ")
    if len(text) > _WA_OCR_TEXT_MAX_CHARS:
        text = text[:_WA_OCR_TEXT_MAX_CHARS - 1] + "…"
    return text


def _enrich_chats_with_lastcontent(
    con,
    grouped_rows: list,
    *,
    bot_jid: str,
    since_bound: str,
) -> list[dict]:
    """Enrich aggregated rows (jid, cnt) with `name` and `last_content`
    in 2 separate queries instead of N+1 correlated subqueries.

    Antes (N+1): `_fetch_whatsapp_unread/today` tenía 2 subqueries escalares
    DENTRO del SELECT — uno para `name` (lookup en chats) y otro para
    `last_content` (lookup ordered en messages). Para 10 chats, eso son 20
    subqueries adicionales que SQLite ejecuta como CORRELATED SCALAR
    SUBQUERIES (verificado con EXPLAIN QUERY PLAN). Bench: ~5-15ms por
    cada subquery par chat → 50-150ms en grupos de 10.

    Ahora: 2 queries totales (uno para names, uno para lastcontent) + merge
    client-side. Para 10 chats: 2 queries totales independientemente del N.
    """
    if not grouped_rows:
        return []

    jids = [r["jid"] for r in grouped_rows]
    placeholders = ",".join("?" * len(jids))

    # Pull names in one shot.
    name_map: dict[str, str] = {}
    try:
        for r in con.execute(
            f"SELECT jid, name FROM chats WHERE jid IN ({placeholders})",
            jids,
        ):
            name_map[r["jid"]] = r["name"] or ""
    except Exception:
        pass

    # Pull last inbound content per chat in one shot using window function.
    # ROW_NUMBER over (partition by chat_jid order by timestamp desc) lets
    # us pick the row with rank=1 = most recent inbound msg per chat.
    last_content_map: dict[str, str] = {}
    try:
        rows = con.execute(
            f"""
            SELECT chat_jid, content FROM (
              SELECT chat_jid, content,
                ROW_NUMBER() OVER (PARTITION BY chat_jid ORDER BY timestamp DESC) AS rn
              FROM messages
              WHERE chat_jid IN ({placeholders})
                AND is_from_me = 0
                AND timestamp >= ?
                AND chat_jid != ?
            )
            WHERE rn = 1
            """,
            (*jids, since_bound, bot_jid),
        ).fetchall()
        for r in rows:
            last_content_map[r["chat_jid"]] = r["content"] or ""
    except Exception:
        pass

    out: list[dict] = []
    for r in grouped_rows:
        jid = r["jid"]
        chat_name = name_map.get(jid, "")
        if whatsapp_chat_name_excluded(chat_name):
            continue
        out.append({
            "jid": jid,
            "name": chat_name,
            "cnt": r["cnt"],
            "last_content": last_content_map.get(jid, ""),
        })
    return out


@contextlib.contextmanager
def _bridge_conn(timeout: float = 5.0):
    """Context manager para abrir una connection read-only al bridge SQLite.

    Centraliza el patrón `sqlite3.connect(file:...?mode=ro, uri=True)` +
    silent-fail si la DB no existe o no se puede abrir. Yieldea la conn ya
    con `row_factory = sqlite3.Row` lista. Si la apertura falla, yieldea None
    (el caller debe checkear). Cierra siempre en `finally`.

    Permite que callers que hacen 2 fetches consecutivos (e.g., morning brief)
    reusen una sola conn en lugar de abrir-cerrar dos veces.
    """
    import rag as _rag
    import sqlite3
    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        yield None
        return
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    except sqlite3.Error:
        yield None
        return
    try:
        con.row_factory = sqlite3.Row
        yield con
    finally:
        try:
            con.close()
        except Exception:
            pass


def _fetch_whatsapp_today(now=None, max_chats: int = 8) -> list[dict]:
    """Inbound WhatsApp messages received TODAY (today 00:00 local → now),
    grouped by chat. Distinto de `_fetch_whatsapp_unread`: ese mira ventana
    rolling de N horas; este corta exactamente al inicio del día local.

    Mirroring `_fetch_whatsapp_unread` shape: list of
    ``{"name": str, "jid": str, "count": int, "last_snippet": str}``
    sorted by message count desc.

    Use case: el evening brief de las 22hs quiere "qué llegó por WA HOY"
    (no "últimas 24hs" que mezclaría parte de ayer). Pasa el corte exacto
    en local time.
    """
    from datetime import datetime as _dt
    if now is None:
        now = _dt.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_bound = _bridge_ts_bound(today_start)
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
        # Pass 1: aggregated counts solamente (sin subqueries escalares).
        # M1 fix 2026-05-09: antes este SELECT tenía 2 subqueries escalares
        # (name + last_content) que SQLite ejecutaba como CORRELATED scan
        # por cada row del GROUP BY → N+1 patron, ~50-150ms en chats con 10+.
        grouped = con.execute(
            """
            SELECT
              m.chat_jid AS jid,
              count(*) AS cnt
            FROM messages m
            WHERE m.is_from_me = 0
              AND m.timestamp >= ?
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
            GROUP BY m.chat_jid
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (today_start_bound, bot_jid, int(max_chats) * 3),
        ).fetchall()
        # Pass 2: enrich names + last_content en 2 queries totales (no N+1).
        rows = _enrich_chats_with_lastcontent(
            con, grouped,
            bot_jid=bot_jid, since_bound=today_start_bound,
        )
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: list[dict] = []
    for r in rows:
        raw_name = (r["name"] or "").strip()
        jid_prefix = (r["jid"] or "").split("@")[0]
        display_name = raw_name or jid_prefix
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
    cutoff_bound = _bridge_ts_bound(datetime.now() - timedelta(hours=int(hours)))
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        # M1 fix 2026-05-09: 2-pass aggregate + enrich en lugar de
        # subqueries escalares N+1. Ver _enrich_chats_with_lastcontent.
        grouped = con.execute(
            """
            SELECT
              m.chat_jid AS jid,
              count(*) AS cnt
            FROM messages m
            WHERE m.is_from_me = 0
              AND m.timestamp > ?
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
            GROUP BY m.chat_jid
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (cutoff_bound, bot_jid, int(max_chats) * 3),
        ).fetchall()
        rows = _enrich_chats_with_lastcontent(
            con, grouped,
            bot_jid=bot_jid, since_bound=cutoff_bound,
        )
        # Anti-loop: drop msgs que arrancan con U+200B post-fetch en Python.
        # Filter en SQL `substr(content,1,1) != char(8203)` rompe el uso del
        # índice. Como los rows aquí ya son `is_from_me=0`, el marker U+200B
        # casi nunca aparece (lo agregamos solo en outbound auto). Si aparece
        # — alguien más manda U+200B legítimo — lo dropeamos acá.
        rows = [r for r in rows if not (r["last_content"] or "").startswith("​")]
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


def _obsidian_rag_config_dir() -> Path:
    explicit = os.environ.get("OBSIDIAN_RAG_CONFIG_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    state_dir = os.environ.get("OBSIDIAN_RAG_STATE_DIR", "").strip()
    if state_dir:
        return Path(state_dir).expanduser() / "config"
    return Path.home() / ".config" / "obsidian-rag"


_SENDER_OVERRIDES_PATH = _obsidian_rag_config_dir() / "wa_sender_overrides.json"
_SENDER_OVERRIDES_CACHE: dict[str, str] = {}
_SENDER_OVERRIDES_MTIME: float = -1.0


def _load_sender_overrides() -> dict[str, str]:
    """Carga `wa_sender_overrides.json` con cache mtime-aware.

    Estructura:
        {"<jid>": "<display name forzado>", ...}

    Casos de uso: cuando bridge.chats.name guarda un push_name erróneo
    o ambiguo (peer puso un nick raro que colisiona con otro contacto),
    el user puede forzar el display name correcto sin tocar código.

    Silent-fail: file no existe / JSON malformado / IO error → {}.
    """
    import json as _json
    global _SENDER_OVERRIDES_CACHE, _SENDER_OVERRIDES_MTIME

    try:
        if not _SENDER_OVERRIDES_PATH.is_file():
            if _SENDER_OVERRIDES_MTIME != -1.0:
                _SENDER_OVERRIDES_CACHE = {}
                _SENDER_OVERRIDES_MTIME = -1.0
            return _SENDER_OVERRIDES_CACHE
        mt = _SENDER_OVERRIDES_PATH.stat().st_mtime
        if mt == _SENDER_OVERRIDES_MTIME:
            return _SENDER_OVERRIDES_CACHE
        raw = _SENDER_OVERRIDES_PATH.read_text(encoding="utf-8")
        data = _json.loads(raw) if raw.strip() else {}
        if isinstance(data, dict):
            _SENDER_OVERRIDES_CACHE = {
                str(k): str(v) for k, v in data.items()
                if isinstance(k, str) and isinstance(v, str) and v.strip()
            }
        else:
            _SENDER_OVERRIDES_CACHE = {}
        _SENDER_OVERRIDES_MTIME = mt
        return _SENDER_OVERRIDES_CACHE
    except Exception:
        return _SENDER_OVERRIDES_CACHE


def _wa_display_name(jid: str, raw_name: str = "") -> str:
    """JID → nombre legible para UI.

    Chain final (con override file que gana sobre TODO):

      (0) `~/.config/obsidian-rag/wa_sender_overrides.json` →
          `{"<jid>": "<display>"}`. Permite al user forzar mapeos
          manuales cuando el bridge guarda un push_name erróneo
          (caso real 2026-05-11: el LID de Maxi en el grupo Recursos
          tenía bridge name "Fer F" — confusión con un push_name de
          otra cuenta).
      (1) bridge name si tiene letras (alpha) → "Grecia 🩷", "Juan P."
      (2) Apple Contacts por phone digits → "Hikari sushi"
      (3) Vault contact note (`99-Contacts/<X>.md` con `wa_jid:`)
      (4) "Contacto …<last4>" fallback final

    Para (3) probamos las 3 variantes del jid (raw, @lid, @s.wa).
    """
    # (0) Override file — manual mapping del user. Mtime-cached.
    # Probamos 3 variantes del jid (raw, @lid, @s.whatsapp.net) porque
    # el bridge guarda senders de grupos a veces bare, a veces full.
    if jid:
        try:
            ov = _load_sender_overrides()
            candidates = [jid]
            if "@" not in jid:
                candidates.extend([f"{jid}@lid", f"{jid}@s.whatsapp.net"])
            for cand in candidates:
                if ov.get(cand):
                    return ov[cand]
        except Exception:
            pass
    name = (raw_name or "").strip()
    if name and any(ch.isalpha() for ch in name):
        return name
    # Apple Contacts lookup por dígitos del JID. Solo aplica a
    # `@s.whatsapp.net` (LIDs no tienen relación con el número real).
    if jid and "@s.whatsapp.net" in jid:
        digits = jid.split("@")[0]
        if digits.isdigit():
            try:
                from rag import _load_contacts_phone_index  # noqa: PLC0415
                idx = _load_contacts_phone_index()
                # Probamos full digits + suffixes progresivos. El index
                # guarda tanto "5493424868405" (full) como "24868405"
                # (local 8 dígitos), así que ambos pegan al cache.
                for k in (digits, digits[-10:], digits[-8:]):
                    apple = idx.get(k)
                    if apple and any(ch.isalpha() for ch in apple):
                        return apple
            except Exception:
                pass
    if jid:
        try:
            from rag.integrations.whatsapp.voice_notes import (  # noqa: PLC0415
                _lookup_contact_name,
            )
            candidates = [jid]
            if "@" not in jid:
                candidates.extend([f"{jid}@lid", f"{jid}@s.whatsapp.net"])
            for cand in candidates:
                vault_name = _lookup_contact_name(cand)
                if vault_name:
                    return vault_name
        except Exception:
            pass
    return _wa_chat_label(name, jid)


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
    since_bound = _bridge_ts_bound(since)
    import sqlite3

    with _bridge_conn() as con:
        if con is None:
            return []
        # Pass 1: identificar qué chats tienen suficiente activity inbound
        # para justificar pull de msgs. Sin esto, en una vault con muchos
        # grupos silenciosos pull-all-then-filter trae miles de rows que se
        # descartan en Python.
        try:
            chat_filter_rows = con.execute(
                """
                SELECT chat_jid, count(*) AS inbound_cnt
                FROM messages
                WHERE timestamp >= ?
                  AND is_from_me = 0
                  AND chat_jid != ?
                  AND chat_jid NOT LIKE '%status@broadcast'
                GROUP BY chat_jid
                HAVING inbound_cnt >= ?
                ORDER BY inbound_cnt DESC
                LIMIT ?
                """,
                (since_bound, bot_jid, int(min_inbound), int(max_chats) * 2),
            ).fetchall()
        except sqlite3.Error:
            return []

        if not chat_filter_rows:
            return []

        eligible_jids = [r["chat_jid"] for r in chat_filter_rows]
        # Pass 2: traer mensajes (inbound + outbound para contexto LLM) SOLO
        # de los chats elegibles. SQL `IN (?,?,...)` con placeholders.
        placeholders = ",".join("?" * len(eligible_jids))
        try:
            rows = con.execute(
                f"""
                SELECT
                  m.id AS id,
                  m.chat_jid AS jid,
                  m.sender AS sender,
                  m.content AS content,
                  m.timestamp AS ts,
                  m.is_from_me AS is_from_me,
                  m.media_type AS media_type,
                  m.filename AS filename,
                  c.name AS chat_name
                FROM messages m
                LEFT JOIN chats c ON c.jid = m.chat_jid
                WHERE m.timestamp >= ?
                  AND m.chat_jid IN ({placeholders})
                ORDER BY m.timestamp ASC
                """,
                (since_bound, *eligible_jids),
            ).fetchall()
        except sqlite3.Error:
            return []
        # Anti-loop: drop U+200B post-fetch.
        rows = [r for r in rows if not (r["content"] or "").startswith("​")]

    by_chat: dict[str, dict] = {}
    ocr_calls_used = 0  # cap defensivo OCR/VLM por invocación (A4)
    for r in rows:
        jid = r["jid"] or ""
        if whatsapp_chat_name_excluded(r["chat_name"] or ""):
            continue
        label = _wa_chat_label(r["chat_name"] or "", jid)
        # Drop unnamed contacts — same policy as morning brief.
        if label.startswith("Contacto …") and not any(ch.isalpha() for ch in (r["chat_name"] or "")):
            continue
        content = (r["content"] or "").strip().replace("\n", " ")
        media_type = (r["media_type"] or "").strip()
        if not content and media_type:
            # A4 (2026-05-09): para imágenes inbound nuevas, intentar OCR/VLM
            # y reemplazar `[image]` con `[image: "<text>"]`. Cap global de
            # `_WA_OCR_MAX_PER_RUN` por invocación. Skipea outbound (`is_from_me=1`)
            # — esos los mandó el user, ya sabe qué dicen.
            content = f"[{media_type}]"
            try:
                msg_id_for_dedup = r["id"] or ""
            except Exception:
                msg_id_for_dedup = ""
            is_image = media_type == "image"
            is_inbound = not bool(r["is_from_me"])
            is_new = bool(msg_id_for_dedup) and msg_id_for_dedup not in processed_ids
            if (is_image and is_inbound and is_new
                    and ocr_calls_used < _WA_OCR_MAX_PER_RUN):
                filename = ""
                try:
                    filename = (r["filename"] or "").strip() if "filename" in r.keys() else ""
                except Exception:
                    filename = ""
                if filename:
                    media_path = _bridge_media_path(jid, filename)
                    if media_path:
                        ocr_calls_used += 1
                        ocr_text = _ocr_image_safe(media_path, label="window")
                        if ocr_text:
                            content = f'[image: "{ocr_text}"]'
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
    if whatsapp_chat_name_excluded(chat_label):
        return empty
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


def _normalize_bridge_ts(ts: str) -> str:
    """Convierte el timestamp del bridge ("YYYY-MM-DD HH:MM:SS-03:00") a
    ISO 8601 estricto ("YYYY-MM-DDTHH:MM:SS-03:00") que `Date.parse` del
    browser acepta sin caprichos. Idempotente para strings ya ISO.
    """
    if not ts:
        return ""
    return ts.replace(" ", "T", 1) if " " in ts else ts


def _avatar_initials(label: str) -> str:
    """Genera iniciales de 1-2 letras para fallback de avatar.

    Toma las primeras letras de las dos primeras palabras del label.
    Si solo hay una palabra usa sus 2 primeras letras.
    """
    parts = [p for p in (label or "").strip().split() if p and any(ch.isalpha() for ch in p)]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def list_chats_for_ui(
    limit: int = 50,
    before_ts: str | None = None,
    q: str | None = None,
    view: str = "default",
) -> list[dict]:
    """Lista paginada de chats para el sidebar de `/wa`.

    Output: lista de dicts con ``{jid, label, is_group, last_ts,
    last_preview, last_from_me, unread_count, avatar_initials}`` ordenada
    por ``last_ts DESC``. ``before_ts`` filtra para paginación hacia
    atrás. ``q`` filtra substring sobre `chats.name` (case-insensitive).

    Unread se computa contra `rag_wa_read_state` (telemetry.db): mensajes
    inbound con `ts > last_seen_ts`. Si nunca se marcó como leído,
    `last_seen_ts` se asume 1970 → todo inbound cuenta.
    """
    import rag as _rag
    import sqlite3

    from . import _db_local

    _db_local.ensure_schema()
    db_path = _rag.WHATSAPP_DB_PATH
    bot_jid = _rag.WHATSAPP_BOT_JID
    if not db_path.is_file():
        return []

    cap = max(1, min(int(limit or 50), 200))
    telemetry_path = _db_local._telemetry_db_path()

    try:
        # `uri=True` permite que ATTACH use `file:...?mode=ro` para
        # mantener la conn bridge en read-only desde el mismo proceso.
        con = sqlite3.connect(f"file:{telemetry_path}", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        con.execute(f"ATTACH DATABASE 'file:{db_path}?mode=ro' AS br")

        # Truth del "último ts" del chat = MAX(messages.timestamp), NO
        # c.last_message_time. Esa columna no se actualiza fiable cuando
        # llegan msgs nuevos a un chat existente (history sync inicial la
        # popula, pero los UPSERTs subsiguientes con ON CONFLICT DO
        # NOTHING dejan la fecha vieja). Síntoma observado: chat de
        # Maria con last_msg de hoy 12:58, pero c.last_message_time
        # devolvía 2026-03-05 → sidebar la ordenaba como vieja. Computar
        # MAX en el SELECT es ~3x más caro que c.last_message_time pero
        # se carga una sola vez al abrir el sidebar y la lectura de
        # 100-300 chats tarda <50ms con el idx_messages_chat_ts existente.
        # Pre-fetch maps de pinned + archived ANTES del SQL para poder
        # filtrar a nivel query — sin esto el LIMIT del SQL excluye
        # archivados viejos que no entran en el top-N por last_ts.
        pinned_map = _db_local.get_pinned_chats()
        archived_map = _db_local.get_archived_chats()
        view_norm = (view or "default").lower()

        where_clauses = [
            "c.jid != ?",
            "c.jid NOT LIKE '%status@broadcast'",
        ]
        params: list = [bot_jid]
        if q:
            where_clauses.append("LOWER(COALESCE(c.name, '')) LIKE ?")
            params.append(f"%{q.lower()}%")
        # View filter at SQL level — garantiza que el LIMIT respete el
        # subset (archivados pueden tener last_ts viejo y caer fuera
        # del top-N si filtráramos solo en Python).
        if archived_map:
            placeholders = ",".join("?" * len(archived_map))
            if view_norm == "archived":
                where_clauses.append(f"c.jid IN ({placeholders})")
            else:
                where_clauses.append(f"c.jid NOT IN ({placeholders})")
            params.extend(archived_map.keys())
        elif view_norm == "archived":
            # No hay archivados — return early sin tocar el bridge.
            return []
        if before_ts:
            before_bound = _bridge_ts_from_ui(before_ts)
            where_clauses.append(
                "(SELECT MAX(m.timestamp) FROM br.messages m WHERE m.chat_jid = c.jid) "
                "IS NOT NULL"
            )
            where_clauses.append(
                "(SELECT MAX(m.timestamp) FROM br.messages m WHERE m.chat_jid = c.jid) < ?"
            )
            params.append(before_bound)
        params.append(cap)

        sql = f"""
            SELECT DISTINCT
              c.jid AS jid,
              COALESCE(c.name, '') AS name,
              (
                SELECT MAX(m.timestamp)
                FROM br.messages m
                WHERE m.chat_jid = c.jid
              ) AS computed_last_ts,
              (
                SELECT m.content
                FROM br.messages m
                WHERE m.chat_jid = c.jid
                ORDER BY m.timestamp DESC
                LIMIT 1
              ) AS last_content,
              (
                SELECT m.media_type
                FROM br.messages m
                WHERE m.chat_jid = c.jid
                ORDER BY m.timestamp DESC
                LIMIT 1
              ) AS last_media,
              (
                SELECT m.is_from_me
                FROM br.messages m
                WHERE m.chat_jid = c.jid
                ORDER BY m.timestamp DESC
                LIMIT 1
              ) AS last_from_me,
              COALESCE(rs.last_seen_ts, '1970-01-01T00:00:00') AS last_seen_ts,
              (
                SELECT COUNT(*)
                FROM br.messages m2
                WHERE m2.chat_jid = c.jid
                  AND m2.is_from_me = 0
                  AND m2.timestamp > COALESCE(rs.last_seen_ts, '1970-01-01T00:00:00')
              ) AS unread_count
            FROM br.chats c
            LEFT JOIN main.rag_wa_read_state rs ON rs.jid = c.jid
            WHERE {' AND '.join(where_clauses)}
            ORDER BY computed_last_ts DESC NULLS LAST
            LIMIT ?
        """
        chat_rows = con.execute(sql, params).fetchall()

        # pinned_map + archived_map ya cargados arriba (pre-SQL).

        out: list[dict] = []
        for r in chat_rows:
            jid = r["jid"] or ""
            name = (r["name"] or "").strip()
            label = _wa_display_name(jid, name)
            if label.startswith("Contacto …") and not any(ch.isalpha() for ch in name):
                continue
            last_content = (r["last_content"] or "").strip().replace("\n", " ")
            last_media = (r["last_media"] or "").strip()
            preview = last_content or (f"[{last_media}]" if last_media else "")
            if len(preview) > 120:
                preview = preview[:117] + "…"
            is_group = jid.endswith("@g.us")
            is_pinned = jid in pinned_map
            is_archived = jid in archived_map
            # Filter SQL ya excluyó archivados según view — no re-filtrar.
            out.append({
                "jid": jid,
                "label": label,
                "name_lower": label.lower(),  # Para ranking por match exacto
                "is_group": is_group,
                "last_ts": _normalize_bridge_ts(r["computed_last_ts"] or ""),
                "last_preview": preview,
                "last_from_me": bool(r["last_from_me"]),
                "unread_count": int(r["unread_count"]),
                "avatar_initials": _avatar_initials(label),
                "pinned": is_pinned,
                "pinned_ts": pinned_map.get(jid, "") if is_pinned else "",
                "archived": is_archived,
            })

        # Dedupe: cuando hay 2+ JIDs (típicamente uno `@lid` + uno
        # `@s.whatsapp.net`) que resuelven al MISMO label (Apple
        # Contacts o vault contact note), colapsamos al de actividad
        # más reciente. Pedido user 2026-05-11 "Maria está duplicada".
        # Grupos quedan exentos — `@g.us` con mismo nombre son
        # legítimamente distintos. Pinned también: si el user pineó
        # una variante, esa gana sobre la otra.
        by_label: dict[str, dict] = {}
        out_dedup: list[dict] = []
        for c in out:
            if c.get("is_group"):
                out_dedup.append(c)
                continue
            key = (c.get("label") or "").strip().lower()
            if not key or key.startswith("contacto "):
                out_dedup.append(c)
                continue
            existing = by_label.get(key)
            if not existing:
                by_label[key] = c
                out_dedup.append(c)
                continue
            # Conflict: keep the one con pinned, sino el de last_ts
            # más reciente, sino el con más unread.
            keep_new = False
            if c.get("pinned") and not existing.get("pinned"):
                keep_new = True
            elif not c.get("pinned") and existing.get("pinned"):
                keep_new = False
            elif _ts_sort_key(c.get("last_ts", "")) > _ts_sort_key(existing.get("last_ts", "")):
                keep_new = True
            elif (c.get("unread_count") or 0) > (existing.get("unread_count") or 0):
                keep_new = True
            if keep_new:
                # Reemplazar in-place en out_dedup.
                try:
                    idx = out_dedup.index(existing)
                    out_dedup[idx] = c
                except ValueError:
                    # Fallback: append si index falla (shouldn't happen)
                    out_dedup.append(c)
                by_label[key] = c

        # Sort: pinned primero (más reciente pin arriba), después el resto
        # por last_ts desc igual que antes. WhatsApp Web hace lo mismo.
        # Si hay query `q`, los contactos cuyo nombre matchea exactamente
        # (case-insensitive) con la query aparecen primero.
        if q:
            q_lower = q.lower().strip()
            out_dedup.sort(key=lambda c: (
                # Exact match = 0, partial match = 1
                0 if (c.get("name_lower") or c.get("label", "").lower()) == q_lower else 1,
                # Pinned siempre arriba
                0 if c.get("pinned") else 1,
                # Luego por pinned_ts o last_ts
                -1 * _ts_sort_key(c.get("pinned_ts", "")) if c.get("pinned")
                    else -1 * _ts_sort_key(c.get("last_ts", "")),
            ))
        else:
            out_dedup.sort(key=lambda c: (
                0 if c.get("pinned") else 1,
                -1 * _ts_sort_key(c.get("pinned_ts", "")) if c.get("pinned")
                    else -1 * _ts_sort_key(c.get("last_ts", "")),
            ))
        return out_dedup
    except sqlite3.Error:
        return []
    finally:
        try:
            con.execute("DETACH DATABASE br")
        except Exception:
            pass
        con.close()


def _ts_sort_key(iso: str) -> float:
    """ISO timestamp → epoch para sort. Empty / malformado → 0."""
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(
            iso.replace(" ", "T", 1).split("+", 1)[0].split("Z", 1)[0]
        ).timestamp()
    except Exception:
        return 0.0


def fetch_thread_for_ui(
    jid: str, limit: int = 50, before_ts: str | None = None
) -> dict:
    """Historial paginado de un chat para el thread view de `/wa`.

    Pagina hacia atrás por ``timestamp DESC``. Cada mensaje viene con
    ``id, ts (ISO), sender, sender_label, content, is_from_me,
    media_type, filename, quoted, reactions, revoked``.

    Reactions se devuelven como lista ``[{emoji, sender_jid, by_me}]``
    leída de `bridge.reactions`. Revoked es un bool — si está en
    `bridge.revokes`, el content se reemplaza por marker para que el
    frontend lo muestre como "este mensaje fue eliminado".
    """
    import rag as _rag
    import sqlite3

    empty = {"jid": jid, "label": "", "is_group": False, "messages": [], "next_before_ts": None}
    if not jid or "@" not in jid:
        return empty

    db_path = _rag.WHATSAPP_DB_PATH
    bot_jid = _rag.WHATSAPP_BOT_JID
    if not db_path.is_file():
        return empty

    cap = max(1, min(int(limit or 50), 200))
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return empty
    try:
        con.row_factory = sqlite3.Row
        # Label del chat (de chats.name).
        chat_row = con.execute(
            "SELECT name FROM chats WHERE jid = ? LIMIT 1", (jid,)
        ).fetchone()
        chat_name = (chat_row["name"] or "") if chat_row else ""
        label = _wa_display_name(jid, chat_name)
        is_group = jid.endswith("@g.us")

        where_clauses = ["m.chat_jid = ?", "m.chat_jid != ?"]
        params: list = [jid, bot_jid]
        if before_ts:
            where_clauses.append("m.timestamp < ?")
            params.append(before_ts)
        params.append(cap)

        rows = con.execute(
            f"""
            SELECT
              m.id AS id,
              m.chat_jid AS chat_jid,
              m.sender AS sender,
              m.content AS content,
              m.timestamp AS ts,
              m.is_from_me AS is_from_me,
              m.media_type AS media_type,
              m.filename AS filename,
              m.quoted_message_id AS quoted_id,
              m.quoted_text AS quoted_text
            FROM messages m
            WHERE {' AND '.join(where_clauses)}
            ORDER BY m.timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        if not rows:
            return {"jid": jid, "label": label, "is_group": is_group, "messages": [], "next_before_ts": None}

        msg_ids = [r["id"] for r in rows if r["id"]]
        placeholders = ",".join("?" * len(msg_ids)) if msg_ids else "''"

        # revoked_by_map: msg_id → quién revocó (peer jid / my jid /
        # "local-only"). Usamos esto para decidir si mostrar tomb
        # ("🚫 Este mensaje fue eliminado") o ocultar el msg entero:
        # - Si is_from_me Y revoked → ocultar (no muestro tomb de mis
        #   propios mensajes, pedido user 2026-05-11).
        # - Si revoked_by == "local-only" → ocultar (hide-for-me local).
        # - Si revoked por peer (peer JID) → tomb.
        revoked_ids: set[str] = set()
        revoked_by_map: dict[str, str] = {}
        if msg_ids:
            try:
                revoke_rows = con.execute(
                    f"SELECT message_id, revoked_by FROM revokes "
                    f"WHERE message_id IN ({placeholders})",
                    msg_ids,
                ).fetchall()
                for r in revoke_rows:
                    revoked_ids.add(r["message_id"])
                    revoked_by_map[r["message_id"]] = r["revoked_by"] or ""
            except sqlite3.Error:
                # Tabla aún sin filas → sin revokes; OK.
                pass

        reactions_by_msg: dict[str, list[dict]] = {}
        if msg_ids:
            try:
                react_rows = con.execute(
                    f"""
                    SELECT message_id, sender_jid, emoji, ts
                    FROM reactions
                    WHERE message_id IN ({placeholders})
                    ORDER BY ts ASC
                    """,
                    msg_ids,
                ).fetchall()
                for rr in react_rows:
                    reactions_by_msg.setdefault(rr["message_id"], []).append({
                        "emoji": rr["emoji"],
                        "sender_jid": rr["sender_jid"],
                    })
            except sqlite3.Error:
                pass

        # Calls del mismo chat dentro de la ventana del thread cargado.
        # Se mergen como "synthetic msgs" con `media_type='call'` para que
        # el render del thread los muestre como bubbles especiales en su
        # posición cronológica. Ventana = ts del msg más viejo a más nuevo.
        call_rows: list = []
        if rows:
            window_min = rows[-1]["ts"] or ""  # más viejo (rows está DESC)
            window_max = rows[0]["ts"] or ""   # más nuevo
            try:
                call_rows = con.execute(
                    """
                    SELECT call_id, chat_jid, from_jid, is_video, is_group,
                           group_jid, offered_ts, accepted_ts, terminated_ts,
                           duration_s, status,
                           COALESCE(terminated_ts, accepted_ts, offered_ts) AS sort_ts
                    FROM calls
                    WHERE chat_jid = ?
                      AND COALESCE(terminated_ts, accepted_ts, offered_ts) BETWEEN ? AND ?
                    ORDER BY sort_ts ASC
                    LIMIT 50
                    """,
                    (jid, window_min, window_max),
                ).fetchall()
            except sqlite3.Error:
                call_rows = []

        # Pre-fetch del bridge label para cada sender único de grupos: si
        # el contact note del vault no matchea (no toda persona tiene
        # nota), igual queremos preferir "Grecia 🩷" que la pone el
        # bridge sobre "Contacto …8025". El bridge guarda esos nombres
        # en `chats` con el jid full (bare + @lid o @s.whatsapp.net) —
        # los buscamos juntos en un solo query.
        bridge_names: dict[str, str] = {}  # bare_local → bridge display name
        unique_senders = {
            (r["sender"] or "").strip() for r in rows if (r["sender"] or "").strip()
        }
        if unique_senders:
            jid_to_bare: dict[str, str] = {}
            for s in unique_senders:
                bare = s.split("@")[0] if "@" in s else s
                jid_to_bare[s] = bare
                jid_to_bare[f"{bare}@lid"] = bare
                jid_to_bare[f"{bare}@s.whatsapp.net"] = bare
            try:
                placeholders = ",".join("?" * len(jid_to_bare))
                for cj, cn in con.execute(
                    f"SELECT jid, name FROM chats WHERE jid IN ({placeholders})",
                    list(jid_to_bare),
                ).fetchall():
                    name = (cn or "").strip()
                    if not name or not any(ch.isalpha() for ch in name):
                        continue
                    bare = jid_to_bare.get(cj)
                    if bare and bare not in bridge_names:
                        bridge_names[bare] = name
            except sqlite3.Error:
                pass

        messages: list[dict] = []
        for r in reversed(rows):  # cronológico asc para lectura natural
            msg_id = r["id"] or ""
            is_from_me = bool(r["is_from_me"])
            sender_raw = (r["sender"] or "").strip()
            if is_from_me:
                sender_label = "yo"
            elif sender_raw:
                # 1) contact note del vault (Mama, Grecia, etc.).
                # 2) bridge.chats.name (Grecia 🩷, push_name del peer).
                # 3) "Contacto …<last4>" fallback de _wa_chat_label.
                bare = sender_raw.split("@")[0] if "@" in sender_raw else sender_raw
                sender_label = _wa_display_name(
                    sender_raw, bridge_names.get(bare, ""),
                )
            else:
                sender_label = label
            content = (r["content"] or "").strip().replace("\n", " ")
            media = (r["media_type"] or "").strip()
            filename = (r["filename"] or "").strip()
            quoted_id = (r["quoted_id"] or "").strip()
            quoted_text = (r["quoted_text"] or "").strip()
            is_revoked = msg_id in revoked_ids
            revoked_by = revoked_by_map.get(msg_id, "")
            if is_revoked:
                # Hide-for-me: nunca renderear (msg propio que el user
                # borró, o msg ajeno que el user ocultó localmente vía
                # /api/wa/hide). Distinto del tomb que sí queremos ver
                # cuando es el peer el que borra su msg.
                if is_from_me or revoked_by == "local-only":
                    continue
                content = ""
                media = ""
                filename = ""

            reactions = reactions_by_msg.get(msg_id, [])
            # Marcar by_me en cada reaction. Si self_jid == sender_jid → by_me.
            # No tenemos self_jid en este conn, lo dejamos None — el frontend
            # decide visualmente (puede chequear contra el JID propio del
            # health endpoint).
            for rx in reactions:
                rx["by_me"] = False  # placeholder, frontend filtra real

            messages.append({
                "id": msg_id,
                "ts": _normalize_bridge_ts(r["ts"] or ""),
                "sender": sender_raw,
                "sender_label": sender_label,
                "content": content,
                "is_from_me": is_from_me,
                "media_type": media or None,
                "filename": filename or None,
                "quoted": {"id": quoted_id, "text": quoted_text} if quoted_id else None,
                "reactions": reactions,
                "revoked": is_revoked,
            })

        # Merge calls como synthetic msgs en el thread. Cada call genera
        # un único bubble basado en su estado final (offered/missed/etc).
        for c in call_rows:
            status = c["status"] or "offered"
            duration_s = c["duration_s"] or 0
            is_video = bool(c["is_video"])
            verb = "Videollamada" if is_video else "Llamada"
            mm = duration_s // 60
            ss = duration_s % 60
            if status == "missed":
                content = f"📵 {verb} perdida"
            elif status == "rejected":
                content = f"❌ {verb} rechazada"
            elif status == "terminated":
                content = f"📞 {verb} · {mm}:{ss:02d}"
            elif status == "accepted":
                content = f"📞 {verb} en curso"
            else:
                content = f"📞 {verb} entrante"
            messages.append({
                "id": f"call:{c['call_id']}",
                "ts": _normalize_bridge_ts(c["sort_ts"] or c["offered_ts"] or ""),
                "sender": c["from_jid"] or "",
                "sender_label": _wa_display_name(c["from_jid"] or "", "") or label,
                "content": content,
                "is_from_me": False,
                "media_type": "call",
                "filename": None,
                "quoted": None,
                "reactions": [],
                "revoked": False,
                # Metadatos extra para el frontend.
                "call_status": status,
                "call_is_video": is_video,
                "call_duration_s": duration_s,
            })

        # Resort cronológico tras el merge.
        messages.sort(key=lambda m: m["ts"] or "")

        # `next_before_ts`: el ts del más viejo de los devueltos para que
        # el cliente pida siguiente página con ese valor.
        next_before_ts = rows[-1]["ts"] if rows else None

        # `last_seen_ts`: el frontend usa este corte para pintar la línea
        # roja de "no leído desde acá" arriba del primer inbound nuevo.
        # `mark_read_for_ui` se llama después del render, así que el corte
        # corresponde al estado PRE-apertura.
        from . import _db_local  # noqa: PLC0415
        last_seen_ts = _db_local.get_last_seen(jid) or ""

        return {
            "jid": jid,
            "label": label,
            "is_group": is_group,
            "messages": messages,
            "next_before_ts": next_before_ts,
            "last_seen_ts": _normalize_bridge_ts(last_seen_ts),
        }
    except sqlite3.Error:
        return empty
    finally:
        con.close()


def mark_read_for_ui(jid: str, last_seen_ts: str | None = None) -> str:
    """Marca el chat como leído hasta ``last_seen_ts`` (default: ahora).

    Devuelve el timestamp efectivamente persistido (útil para que el
    frontend update su estado optimista).
    """
    from datetime import datetime as _dt

    from . import _db_local

    if not jid or "@" not in jid:
        return ""
    ts = _bridge_ts_from_ui(last_seen_ts)
    if not ts:
        ts = _dt.now().strftime(_BRIDGE_TS_FMT) + _BRIDGE_TZ_OFFSET
    _db_local.set_last_seen(jid, ts)
    return ts


__all__ = [
    "_fetch_whatsapp_today",
    "_fetch_whatsapp_unread",
    "_wa_chat_label",
    "_fetch_whatsapp_window",
    "_fetch_whatsapp_recent_with_jid",
    "list_chats_for_ui",
    "fetch_thread_for_ui",
    "mark_read_for_ui",
    "_normalize_bridge_ts",
    "_avatar_initials",
    "_bridge_ts_from_ui",
]
