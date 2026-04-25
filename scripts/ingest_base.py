"""Helpers compartidos para scripts/ingest_*.py.

Extraído 2026-04-25 del saneamiento: los 8 ingest scripts duplicaban
las mismas primitives de state management, upsert/delete y CLI
boilerplate. Esta versión empieza con el patrón HASH-based state
(reminders, contacts, calls, safari), que es el más común (~144 LOC
duplicadas pre-extracción).

Los scripts que usan sync-token (calendar, gmail) o cursor timestamp
(whatsapp, gdrive) tienen sus propias primitives inline — agregarlas
acá requiere más auditoría de divergencias, se hace en otra iteración.

Design:
- Stateless functions (no class, no singletons). Cada call toma la
  sqlite3.Connection del caller — mantiene el control de transacciones
  en el script dueño.
- Parametriza tabla + key column al tope de cada función para que
  múltiples state tables (history + bookmark en safari) compartan el
  mismo código sin hardcoding.
- No hace commit() — el caller decide cuándo comitear (permite
  batching).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime


def make_state_table_ddl(table_name: str, key_col: str, key_type: str = "TEXT") -> str:
    """Genera el `CREATE TABLE IF NOT EXISTS` para una state table hash-based.

    Schema estándar:
      - key_col (primary key)
      - content_hash TEXT NOT NULL
      - last_seen_ts TEXT NOT NULL (ISO-8601 de última vez que vimos el item)
      - updated_at TEXT NOT NULL (ISO-8601 de último update al row)

    `key_type` default TEXT cubre los 4 scripts del patrón (reminder_id str,
    contact_uid str, call_uid str, bookmark_uuid str, history_item_id int).
    Safari usa INTEGER para history_item_id — pasá `key_type="INTEGER"` en ese caso.
    """
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} ("
        f" {key_col} {key_type} PRIMARY KEY,"
        " content_hash TEXT NOT NULL,"
        " last_seen_ts TEXT NOT NULL,"
        " updated_at TEXT NOT NULL"
        ")"
    )


def ensure_state_table(
    conn: sqlite3.Connection,
    table_name: str,
    key_col: str,
    key_type: str = "TEXT",
) -> None:
    """Crea la state table si no existe. Idempotente."""
    conn.execute(make_state_table_ddl(table_name, key_col, key_type))


def load_hashes(
    conn: sqlite3.Connection,
    table_name: str,
    key_col: str,
) -> dict[str, str]:
    """Carga todo el mapping (key → content_hash) de la state table.

    Returns un dict donde las keys son SIEMPRE str (convertidas desde
    INTEGER si hace falta) — permite que el caller compare con hashes
    nuevos sin preocuparse por el tipo del PK.
    """
    cur = conn.execute(f"SELECT {key_col}, content_hash FROM {table_name}")
    return {str(row[0]): row[1] for row in cur.fetchall()}


def upsert_hash(
    conn: sqlite3.Connection,
    table_name: str,
    key_col: str,
    key: str | int,
    content_hash: str,
    now_iso: str | None = None,
) -> None:
    """Inserta/actualiza una fila en la state table.

    `now_iso`: si no se pasa, usa datetime.now().isoformat(timespec="seconds").
    Pasarlo explícitamente es útil para tests (mocked clock) o para mantener
    el mismo timestamp en un batch.
    """
    if now_iso is None:
        now_iso = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        f"INSERT OR REPLACE INTO {table_name} "
        f"({key_col}, content_hash, last_seen_ts, updated_at) "
        f"VALUES (?, ?, ?, ?)",
        (key, content_hash, now_iso, now_iso),
    )


def delete_hash(
    conn: sqlite3.Connection,
    table_name: str,
    key_col: str,
    key: str | int,
) -> None:
    """Borra la fila de la state table por key. Silent si no existe."""
    conn.execute(f"DELETE FROM {table_name} WHERE {key_col} = ?", (key,))


def delete_chunks_by_file_key(col, file_key: str) -> int:
    """Borra todos los chunks con el `file` metadata igual al file_key dado.

    Idempotente — silent si no hay matches. Usado en el pattern de upsert
    de todos los ingest scripts: "delete existing → embed new → add new"
    para garantizar idempotencia del re-ingest. Returns la cantidad de
    chunks borrados (0 si el file key no existía).
    """
    try:
        existing = col.get(where={"file": file_key}, include=[])
        ids = existing.get("ids") or []
        if ids:
            col.delete(ids=ids)
            return len(ids)
    except Exception:
        # Collection I/O errors son silent-fail para no abortar el batch
        # del caller. El próximo re-ingest intentará de nuevo.
        pass
    return 0


def delete_chunks_by_file_keys(col, file_keys: list[str]) -> int:
    """Bulk delete por lista de file_keys. Returns la count total de chunks
    borrados across todos los file keys.

    Los 8 ingest scripts tienen su propio `delete_X()` que itera sobre una
    lista de IDs y arma `file_key = f"{DOC_ID_PREFIX}://{id}"` antes de
    llamar al delete individual. Los callers siguen armando el file_key
    como antes; este helper solo hace el loop + accounting.
    """
    total = 0
    for fk in file_keys:
        total += delete_chunks_by_file_key(col, fk)
    return total
