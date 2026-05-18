"""Search FTS5 cross-chat sobre WhatsApp messages.

Flujo:

1. **Backfill** — `backfill_mirror()` copia rows del bridge `messages.db`
   a `rag_wa_messages_mirror` (en telemetry.db). `INSERT OR IGNORE`
   idempotente; safe correr múltiples veces.

2. **Sync incremental** — `sync_recent(since_ts)` toma todos los
   `messages.timestamp > since` y los upsertea al mirror. El poller
   de `tail.py` lo invoca cada N ticks.

3. **Search** — `search(q, jid=None, limit=50)` corre `MATCH ?
   ORDER BY rank LIMIT N` sobre la virtual table FTS5 + `snippet()`
   highlight. Filtra opcionalmente por `chat_jid`.

El FTS5 + mirror viven en telemetry.db (no en el bridge messages.db,
que es RO desde Python). Los triggers AI/AD/AU de `_db_local.py`
mantienen el índice sync con el mirror.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from ._constants import whatsapp_chat_name_excluded

logger = logging.getLogger("rag.wa.search")

_BACKFILL_BATCH = 1000


def _open_telemetry_rw():
    from . import _db_local

    _db_local.ensure_schema()
    path = _db_local._telemetry_db_path()
    con = sqlite3.connect(f"file:{path}", uri=True, timeout=10.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=10000")
    con.row_factory = sqlite3.Row
    return con


def _attach_bridge(con: sqlite3.Connection) -> bool:
    import rag as _rag

    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        return False
    try:
        con.execute(f"ATTACH DATABASE 'file:{db_path}?mode=ro' AS br")
        return True
    except sqlite3.Error as e:
        logger.warning("attach bridge failed: %s", e)
        return False


def _excluded_chat_jids(con: sqlite3.Connection) -> list[str]:
    """Resolve exact-name blacklisted chats from the attached bridge DB."""
    try:
        rows = con.execute("SELECT jid, COALESCE(name, '') AS name FROM br.chats").fetchall()
    except sqlite3.Error:
        return []
    return [r["jid"] for r in rows if whatsapp_chat_name_excluded(r["name"])]


def _purge_semantic_vec_rowids(con: sqlite3.Connection, rowids: list[int]) -> None:
    """Best-effort delete for the optional WA semantic vec sidecar."""
    if not rowids:
        return
    try:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_wa_messages_vec'"
        ).fetchone()
        if not exists:
            return
        try:
            import sqlite_vec  # type: ignore[import-not-found]  # noqa: PLC0415
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            con.enable_load_extension(False)
        except Exception:
            pass
        placeholders = ",".join("?" * len(rowids))
        con.execute(
            f"DELETE FROM rag_wa_messages_vec WHERE rowid IN ({placeholders})",
            rowids,
        )
    except sqlite3.Error as e:
        logger.warning("semantic vec purge failed: %s", e)


def _purge_excluded_chats_attached(con: sqlite3.Connection) -> int:
    """Delete already-mirrored rows for currently blacklisted chat names."""
    excluded_jids = _excluded_chat_jids(con)
    if not excluded_jids:
        return 0
    placeholders = ",".join("?" * len(excluded_jids))
    try:
        rowids = [
            int(r["rowid"])
            for r in con.execute(
                f"SELECT rowid FROM rag_wa_messages_mirror "
                f"WHERE chat_jid IN ({placeholders})",
                excluded_jids,
            ).fetchall()
        ]
        if not rowids:
            return 0
        _purge_semantic_vec_rowids(con, rowids)
        row_ph = ",".join("?" * len(rowids))
        cur = con.execute(
            f"DELETE FROM rag_wa_messages_mirror WHERE rowid IN ({row_ph})",
            rowids,
        )
        con.commit()
        return int(cur.rowcount or len(rowids))
    except sqlite3.Error as e:
        logger.warning("excluded chat purge failed: %s", e)
        return 0


def purge_excluded_chats() -> int:
    """Public maintenance hook: remove blacklisted chats from WA mirror/FTS."""
    con = _open_telemetry_rw()
    if not _attach_bridge(con):
        con.close()
        return 0
    try:
        return _purge_excluded_chats_attached(con)
    finally:
        try:
            con.execute("DETACH DATABASE br")
        except sqlite3.Error:
            pass
        con.close()


def backfill_mirror() -> int:
    """One-shot backfill: copia TODOS los messages del bridge al mirror.

    Idempotente: `INSERT OR IGNORE` skipea los que ya existen por PK
    `(id, chat_jid)`. Devuelve la cantidad insertada en esta corrida
    (las filas ya existentes no cuentan).
    """
    import rag as _rag

    con = _open_telemetry_rw()
    if not _attach_bridge(con):
        con.close()
        return 0

    bot_jid = _rag.WHATSAPP_BOT_JID
    total_inserted = 0
    try:
        purged = _purge_excluded_chats_attached(con)
        if purged:
            logger.info("backfill: purgué %s rows de chats WA excluidos", purged)

        # Cuántos rows tenemos ya en el mirror
        cnt = con.execute("SELECT count(*) AS n FROM rag_wa_messages_mirror").fetchone()
        had = cnt["n"] if cnt else 0
        logger.info("backfill: mirror tiene %s rows, copiando del bridge", had)

        # Pull todos los messages (batched para no levantar TODO a memoria).
        sql_select = """
            SELECT m.id, m.chat_jid, m.timestamp, m.sender, m.content,
                   COALESCE(c.name, '') AS chat_name
            FROM br.messages m
            LEFT JOIN br.chats c ON c.jid = m.chat_jid
            WHERE m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
              AND (m.content IS NOT NULL AND m.content != '')
            ORDER BY m.timestamp ASC
        """
        cur = con.execute(sql_select, (bot_jid,))
        batch: list[tuple] = []
        while True:
            rows = cur.fetchmany(_BACKFILL_BATCH)
            if not rows:
                break
            batch = [
                (r["id"], r["chat_jid"], r["timestamp"], r["sender"] or "", r["content"] or "")
                for r in rows
                if not whatsapp_chat_name_excluded(r["chat_name"])
            ]
            if not batch:
                continue
            cur2 = con.executemany(
                "INSERT OR IGNORE INTO rag_wa_messages_mirror "
                "(id, chat_jid, ts, sender, content) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            total_inserted += cur2.rowcount or 0
        con.commit()
        logger.info("backfill: inserté %s rows nuevas", total_inserted)
    except sqlite3.Error as e:
        logger.warning("backfill failed: %s", e)
    finally:
        try:
            con.execute("DETACH DATABASE br")
        except sqlite3.Error:
            pass
        con.close()
    return total_inserted


def sync_recent(since_ts: str) -> int:
    """Sync incremental: copia messages con `timestamp > since_ts` al
    mirror. El caller (tail.py) lo invoca periódicamente.
    """
    import rag as _rag

    if not since_ts:
        return 0
    con = _open_telemetry_rw()
    if not _attach_bridge(con):
        con.close()
        return 0
    bot_jid = _rag.WHATSAPP_BOT_JID
    inserted = 0
    try:
        _purge_excluded_chats_attached(con)
        rows = con.execute(
            """
            SELECT m.id, m.chat_jid, m.timestamp, m.sender, m.content,
                   COALESCE(c.name, '') AS chat_name
            FROM br.messages m
            LEFT JOIN br.chats c ON c.jid = m.chat_jid
            WHERE m.timestamp > ?
              AND m.chat_jid != ?
              AND m.chat_jid NOT LIKE '%status@broadcast'
              AND (m.content IS NOT NULL AND m.content != '')
            """,
            (since_ts, bot_jid),
        ).fetchall()
        if not rows:
            return 0
        rows = [r for r in rows if not whatsapp_chat_name_excluded(r["chat_name"])]
        if not rows:
            return 0
        cur = con.executemany(
            "INSERT OR IGNORE INTO rag_wa_messages_mirror "
            "(id, chat_jid, ts, sender, content) VALUES (?, ?, ?, ?, ?)",
            [
                (r["id"], r["chat_jid"], r["timestamp"], r["sender"] or "", r["content"] or "")
                for r in rows
            ],
        )
        inserted = cur.rowcount or 0
        con.commit()
    except sqlite3.Error as e:
        logger.warning("sync_recent failed: %s", e)
    finally:
        try:
            con.execute("DETACH DATABASE br")
        except sqlite3.Error:
            pass
        con.close()
    return inserted


# FTS5 requiere algunas precauciones con strings que pueden romper el
# parser MATCH (caracteres `"` `(` `)` `*` `:` `^`). Sanitizamos
# escapando con quoting estándar.
_FTS_UNSAFE_RE = re.compile(r'[\x00-\x1f"]')


def _sanitize_query(q: str) -> str:
    """Sanitiza un query para FTS5 MATCH. Strip control chars + escape
    quotes. Devuelve la query ya wrappeada en `"..."` para que FTS5 la
    trate como frase completa (más predictable que el syntax full).
    """
    cleaned = _FTS_UNSAFE_RE.sub(" ", q or "").strip()
    cleaned = cleaned.replace('"', '""')
    if not cleaned:
        return ""
    # Si tiene espacios, ponemos cada palabra como término individual
    # (operador implícito = AND). Sino, single term.
    parts = [w for w in cleaned.split() if w]
    if not parts:
        return ""
    return " ".join(f'"{w}"' for w in parts)


def search(q: str, jid: str | None = None, limit: int = 50) -> list[dict]:
    """Search FTS5 sobre el mirror. Returns lista de hits con snippet
    highlighted (HTML `<mark>...</mark>`).
    """
    safe = _sanitize_query(q)
    if not safe:
        return []
    cap = max(1, min(int(limit or 50), 200))

    con = _open_telemetry_rw()
    try:
        where_extra = ""
        params: list[Any] = [safe]
        if jid:
            where_extra = " AND m.chat_jid = ?"
            params.append(jid)
        params.append(cap)

        # JOIN con br.chats para traer el chat name. Necesitamos attach.
        if not _attach_bridge(con):
            # Sin bridge attached, devolvemos hits sin chat name.
            rows = con.execute(
                f"""
                SELECT
                  m.id, m.chat_jid, m.ts, m.sender,
                  snippet(rag_wa_fts, 0, '<mark>', '</mark>', '…', 12) AS snippet
                FROM rag_wa_fts
                JOIN rag_wa_messages_mirror m ON m.rowid = rag_wa_fts.rowid
                WHERE rag_wa_fts MATCH ?{where_extra}
                ORDER BY rank
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "chat_jid": r["chat_jid"],
                    "chat_name": "",
                    "ts": (r["ts"] or "").replace(" ", "T", 1),
                    "sender": r["sender"],
                    "snippet": r["snippet"],
                }
                for r in rows
            ]
        rows = con.execute(
            f"""
            SELECT
              m.id, m.chat_jid, m.ts, m.sender,
              c.name AS chat_name,
              snippet(rag_wa_fts, 0, '<mark>', '</mark>', '…', 12) AS snippet
            FROM rag_wa_fts
            JOIN rag_wa_messages_mirror m ON m.rowid = rag_wa_fts.rowid
            LEFT JOIN br.chats c ON c.jid = m.chat_jid
            WHERE rag_wa_fts MATCH ?{where_extra}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r["id"],
                "chat_jid": r["chat_jid"],
                "chat_name": r["chat_name"] or "",
                "ts": (r["ts"] or "").replace(" ", "T", 1),
                "sender": r["sender"],
                "snippet": r["snippet"],
            }
            for r in rows
            if not whatsapp_chat_name_excluded(r["chat_name"] or "")
        ]
    except sqlite3.Error as e:
        logger.warning("search failed: %s", e)
        return []
    finally:
        try:
            con.execute("DETACH DATABASE br")
        except sqlite3.Error:
            pass
        con.close()
