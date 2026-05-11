"""Tablas SQLite locales que dan soporte a `/wa` (WhatsApp Web propio).

Viven en `telemetry.db` (consistente con el resto de tablas `rag_*`).
Esquemas idempotentes: la migración corre al primer import del módulo y
no rompe nada si ya existen.

Tablas
------
- ``rag_wa_read_state(jid, last_seen_ts)`` — heurística de unread_count
  por chat. La UI llama `mark_read` cuando el user abre/scrollea hasta
  el final del thread; `list_chats` cuenta `messages.timestamp >
  last_seen_ts AND is_from_me=0` por chat.
- ``rag_wa_messages_mirror(id, chat_jid, ts, sender, content)`` —
  mirror append-only del `messages` del bridge para alimentar el
  índice FTS5. `INSERT OR IGNORE` con PK `(id, chat_jid)`.
- ``rag_wa_fts`` — virtual FTS5 sobre el mirror. Usa `content_rowid`
  para evitar duplicar el texto.

El bridge `messages.db` queda intacto — no se le agrega FTS5 ahí porque
necesitaríamos abrirla read-write desde Python y rompería el
constraint "no concurrencia bridge↔rag" sobre la misma WAL.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

# Path imports diferido al uso para que monkeypatching de `DB_PATH` en
# tests (snap+restore manual; ver conftest) gane sobre el binding del
# módulo.
def _telemetry_db_path() -> Path:
    from rag import DB_PATH, _TELEMETRY_DB_FILENAME  # type: ignore

    return Path(DB_PATH) / _TELEMETRY_DB_FILENAME


_MIGRATION_LOCK = threading.Lock()
_MIGRATION_DONE = False


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_wa_read_state (
    jid TEXT PRIMARY KEY,
    last_seen_ts TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rag_wa_messages_mirror (
    id TEXT NOT NULL,
    chat_jid TEXT NOT NULL,
    ts TEXT NOT NULL,
    sender TEXT,
    content TEXT,
    PRIMARY KEY (id, chat_jid)
);
CREATE INDEX IF NOT EXISTS idx_rag_wa_mirror_chat_ts
    ON rag_wa_messages_mirror(chat_jid, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rag_wa_mirror_ts
    ON rag_wa_messages_mirror(ts DESC);

-- FTS5 sobre el mirror. `content=` apunta a la tabla base; SQLite
-- mantiene el índice via triggers que insertamos abajo. Tokenizer
-- `unicode61` con `remove_diacritics=2` para que "cafe" matchee "café".
CREATE VIRTUAL TABLE IF NOT EXISTS rag_wa_fts USING fts5(
    content,
    sender UNINDEXED,
    chat_jid UNINDEXED,
    ts UNINDEXED,
    content='rag_wa_messages_mirror',
    content_rowid='rowid',
    tokenize="unicode61 remove_diacritics 2"
);

-- Triggers para mantener FTS sincronizado. `rowid` virtual de FTS
-- mapea al `rowid` del mirror (SQLite asigna uno automáticamente).
CREATE TRIGGER IF NOT EXISTS rag_wa_fts_ai AFTER INSERT ON rag_wa_messages_mirror BEGIN
    INSERT INTO rag_wa_fts(rowid, content, sender, chat_jid, ts)
    VALUES (new.rowid, new.content, new.sender, new.chat_jid, new.ts);
END;

CREATE TRIGGER IF NOT EXISTS rag_wa_fts_ad AFTER DELETE ON rag_wa_messages_mirror BEGIN
    INSERT INTO rag_wa_fts(rag_wa_fts, rowid, content, sender, chat_jid, ts)
    VALUES ('delete', old.rowid, old.content, old.sender, old.chat_jid, old.ts);
END;

CREATE TRIGGER IF NOT EXISTS rag_wa_fts_au AFTER UPDATE ON rag_wa_messages_mirror BEGIN
    INSERT INTO rag_wa_fts(rag_wa_fts, rowid, content, sender, chat_jid, ts)
    VALUES ('delete', old.rowid, old.content, old.sender, old.chat_jid, old.ts);
    INSERT INTO rag_wa_fts(rowid, content, sender, chat_jid, ts)
    VALUES (new.rowid, new.content, new.sender, new.chat_jid, new.ts);
END;
"""


def _connect() -> sqlite3.Connection:
    """Connect a `telemetry.db` con timeout generoso y WAL implícito.

    Mantiene el mismo patrón del resto del paquete: PRAGMA en cada conn
    porque sqlite3 los aísla per-connection.
    """
    db_path = _telemetry_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    """Corre la migración una sola vez por proceso. Thread-safe."""
    global _MIGRATION_DONE
    if _MIGRATION_DONE:
        return
    with _MIGRATION_LOCK:
        if _MIGRATION_DONE:
            return
        conn = _connect()
        try:
            conn.executescript(_SCHEMA_SQL)
        finally:
            conn.close()
        _MIGRATION_DONE = True


# --- Helpers de lectura/escritura ---


def get_last_seen(jid: str) -> str | None:
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT last_seen_ts FROM rag_wa_read_state WHERE jid = ?", (jid,)
        ).fetchone()
        return row["last_seen_ts"] if row else None
    finally:
        conn.close()


def set_last_seen(jid: str, last_seen_ts: str) -> None:
    ensure_schema()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO rag_wa_read_state (jid, last_seen_ts) VALUES (?, ?) "
            "ON CONFLICT(jid) DO UPDATE SET last_seen_ts = excluded.last_seen_ts, "
            "updated_at = datetime('now')",
            (jid, last_seen_ts),
        )
    finally:
        conn.close()


def upsert_mirror_rows(rows: list[dict[str, Any]]) -> int:
    """Inserta filas en `rag_wa_messages_mirror` ignorando duplicados.

    `rows` esperado: lista de dicts con keys `id`, `chat_jid`, `ts`,
    `sender`, `content`. Devuelve cantidad afectada (rowcount). Los
    triggers FTS5 sincronizan el índice automáticamente.
    """
    if not rows:
        return 0
    ensure_schema()
    conn = _connect()
    try:
        conn.execute("BEGIN")
        cur = conn.executemany(
            "INSERT OR IGNORE INTO rag_wa_messages_mirror (id, chat_jid, ts, sender, content) "
            "VALUES (:id, :chat_jid, :ts, :sender, :content)",
            rows,
        )
        conn.execute("COMMIT")
        return cur.rowcount
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def max_mirror_ts() -> str | None:
    """Devuelve el `ts` más alto del mirror para guiar el backfill incremental."""
    ensure_schema()
    conn = _connect()
    try:
        row = conn.execute("SELECT MAX(ts) AS m FROM rag_wa_messages_mirror").fetchone()
        return row["m"] if row and row["m"] else None
    finally:
        conn.close()
