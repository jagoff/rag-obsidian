"""Explicit, versioned schema migrations para `telemetry.db`.

Mata la clase de bugs `CREATE TABLE IF NOT EXISTS no aplica ALTER en DBs
viejas` (ver bug 2026-04-26 en `rag/__init__.py:5687`). Cada cambio de
schema vive como `migrate_NNN_descripcion(conn)` registrada via decorator
`@migration(version, name)`, se aplica una vez bajo SAVEPOINT, y queda
trackeada en la tabla `rag_schema_migrations(version, name, applied_at,
hash)`.

Diseño:
- **Idempotente**: `apply_pending_migrations` lee `MAX(version)` y aplica
  solo las que estén por arriba. La 2da corrida no toca nada (un solo
  SELECT MAX).
- **Atomico por migration**: cada migration corre dentro de un SAVEPOINT
  propio. Si la migration tira, ROLLBACK TO SAVEPOINT y la version queda
  sin registrar — la próxima corrida la reintenta.
- **Bootstrap heurístico**: en DBs preexistentes que ya tienen las
  columnas/índices que las migrations creanan (porque los `_migrate_*`
  legacy corrieron en boots previos), `bootstrap_existing_db()` registra
  las migrations conocidas como ya aplicadas SIN ejecutarlas — evita
  re-aplicar ALTERs que `OperationalError: duplicate column`.
- **Hash drift**: cada migration tiene un sha1(source_body). Si una
  migration ya aplicada cambia su body, log warning (no error fatal —
  blockea ops).

No-go zones:
- No transaccional ALTER TABLE en SQLite no rompe el SAVEPOINT — solo
  los DDLs pre-3.25 son problemáticos. Para nuestro target (SQLite ≥
  3.35 que viene con Python 3.11+) ALTER TABLE ADD COLUMN sí es
  transaccional.
- No corre downgrades. Si una migration está mal, la próxima migration
  NNN+1 hace el cleanup explícito.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

# Registry: version → (name, callable). OrderedDict garantiza que iteramos
# en orden de inserción, pero igual ordenamos por version cuando aplicamos.
_REGISTRY: "OrderedDict[int, tuple[str, Callable[[sqlite3.Connection], None]]]" = OrderedDict()


# ─── Decorator API ──────────────────────────────────────────────────────────


def migration(version: int, name: str) -> Callable[[Callable[[sqlite3.Connection], None]], Callable[[sqlite3.Connection], None]]:
    """Registra una migration en el módulo.

    Uso:

        @migration(version=1, name="add_trace_id_to_queries")
        def migrate_001_add_trace_id_to_queries(conn: sqlite3.Connection) -> None:
            conn.execute("ALTER TABLE rag_queries ADD COLUMN trace_id TEXT")

    Las migrations se aplican en orden ascendente de `version`. La función
    recibe `conn` con un SAVEPOINT ya abierto — si tira, el SAVEPOINT
    rollbackea y `apply_pending_migrations` propaga el error.
    """
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"migration version debe ser int ≥ 1, got {version!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"migration name debe ser str no vacío, got {name!r}")

    def deco(fn: Callable[[sqlite3.Connection], None]) -> Callable[[sqlite3.Connection], None]:
        if version in _REGISTRY:
            existing_name = _REGISTRY[version][0]
            if existing_name != name:
                raise ValueError(
                    f"migration version={version} ya registrada como "
                    f"{existing_name!r}, no se puede sobrescribir con {name!r}"
                )
        _REGISTRY[version] = (name, fn)
        return fn

    return deco


def _migration_hash(fn: Callable[[sqlite3.Connection], None]) -> str:
    """sha1 del source body de la función. Detecta drift sin necesidad
    de tracking explícito de revisiones — si modificás una migration ya
    aplicada, el hash cambia y log warning."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        # Algunas funciones built-in / lambdas / dynamic no tienen source.
        # Caemos en repr para tener algo determinista.
        src = repr(fn)
    return hashlib.sha1(src.encode("utf-8")).hexdigest()


# ─── Estado de la DB ────────────────────────────────────────────────────────


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    """Crea `rag_schema_migrations` si no existe. Idempotente — el DDL
    en `_TELEMETRY_DDL` ya la crea en boot, pero este helper sirve para
    casos de tests donde abrimos una conn raw sin pasar por el wiring."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_migrations ("
        " version INTEGER PRIMARY KEY,"
        " name TEXT NOT NULL,"
        " applied_at TEXT NOT NULL,"
        " hash TEXT"
        ")"
    )


def current_version(conn: sqlite3.Connection) -> int:
    """Retorna la max(version) aplicada. 0 si ninguna o si la tabla
    no existe todavía."""
    _ensure_migrations_table(conn)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM rag_schema_migrations"
    ).fetchone()
    return int(row[0]) if row else 0


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Set de versions ya registradas como aplicadas."""
    _ensure_migrations_table(conn)
    rows = conn.execute("SELECT version FROM rag_schema_migrations").fetchall()
    return {int(r[0]) for r in rows}


def known_migrations() -> list[tuple[int, str, Callable[[sqlite3.Connection], None]]]:
    """Lista ordenada de (version, name, fn) registradas en el módulo."""
    return [(v, name, fn) for v, (name, fn) in sorted(_REGISTRY.items())]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─── Apply / bootstrap ──────────────────────────────────────────────────────


def apply_pending_migrations(conn: sqlite3.Connection) -> list[int]:
    """Aplica todas las migrations registradas con version > current_version.

    Cada migration corre bajo un SAVEPOINT propio. Si una falla, el
    SAVEPOINT rollbackea esa migration (la version no se registra),
    el error se propaga y las migrations posteriores no corren.

    Retorna la lista de versions aplicadas en esta llamada (vacía si
    estaba up-to-date).

    También chequea drift: si una migration registrada (en _REGISTRY)
    coincide con una version aplicada pero el hash difiere, log warning.
    NO bloquea la app — corregir un typo en un comment de una migration
    no es razón para crashear.
    """
    _ensure_migrations_table(conn)
    cur = current_version(conn)
    applied = applied_versions(conn)
    pending = [(v, name, fn) for v, name, fn in known_migrations() if v > cur]

    # Drift check sobre las ya aplicadas — best-effort, solo log.
    try:
        rows = conn.execute(
            "SELECT version, hash FROM rag_schema_migrations"
        ).fetchall()
        stored_hashes = {int(r[0]): (r[1] or "") for r in rows}
        for v, name, fn in known_migrations():
            if v in applied:
                expected = _migration_hash(fn)
                stored = stored_hashes.get(v, "")
                if stored and stored != expected:
                    logger.warning(
                        "schema_migration_drift version=%d name=%s "
                        "stored_hash=%s expected_hash=%s",
                        v, name, stored[:8], expected[:8],
                    )
    except sqlite3.Error:
        pass

    applied_now: list[int] = []
    for v, name, fn in pending:
        sp_name = f"migration_{v}"
        try:
            conn.execute(f"SAVEPOINT {sp_name}")
            fn(conn)
            conn.execute(
                "INSERT OR REPLACE INTO rag_schema_migrations(version, name, applied_at, hash)"
                " VALUES(?, ?, ?, ?)",
                (v, name, _now_iso(), _migration_hash(fn)),
            )
            conn.execute(f"RELEASE SAVEPOINT {sp_name}")
            applied_now.append(v)
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                conn.execute(f"RELEASE SAVEPOINT {sp_name}")
            raise

    return applied_now


def bootstrap_existing_db(conn: sqlite3.Connection) -> list[int]:
    """Registra todas las migrations conocidas como aplicadas SIN correrlas.

    Pensado para DBs preexistentes que ya tienen las columnas/índices
    que las migrations harían (porque los `_migrate_*` legacy corrieron
    en boots previos antes de este sistema explícito). Heurística: si
    `rag_queries` ya tiene la columna `trace_id`, asumimos que TODOS los
    ALTERs históricos se aplicaron — registramos las migrations 1-N como
    `applied_at=now, hash=current` sin ejecutar el body.

    Solo bootstrappea si la tabla `rag_schema_migrations` está VACÍA.
    Si ya hay rows (apply_pending_migrations corrió antes), no hace nada
    para no pisar history real.

    Retorna la lista de versions registradas (vacía si no se hizo
    bootstrap por que ya había history o porque la heurística no calzó).
    """
    _ensure_migrations_table(conn)
    cur = current_version(conn)
    if cur > 0:
        return []
    if not _heuristic_db_is_post_migrations(conn):
        return []
    versions: list[int] = []
    now = _now_iso()
    for v, name, fn in known_migrations():
        try:
            conn.execute(
                "INSERT OR IGNORE INTO rag_schema_migrations(version, name, applied_at, hash)"
                " VALUES(?, ?, ?, ?)",
                (v, name, now, _migration_hash(fn)),
            )
            versions.append(v)
        except sqlite3.Error:
            # Best-effort; si una falla, las que ya entraron quedan registradas.
            continue
    return versions


def _heuristic_db_is_post_migrations(conn: sqlite3.Connection) -> bool:
    """True si la DB parece ser pre-existente con migrations ya aplicadas
    via los `_migrate_*` legacy. Heurística mínima: existe la tabla
    `rag_queries` y tiene la columna `trace_id`. Cubre las 4 migrations
    iniciales (todas son trace_id-related).

    False si la DB es virgen (ninguna tabla todavía) o si claramente le
    faltan columnas que las migrations agregarían. En ambos casos lo
    correcto es correr `apply_pending_migrations` real.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_queries'"
        ).fetchone()
        if not row:
            return False
        cols = {r[1] for r in conn.execute("PRAGMA table_info(rag_queries)").fetchall()}
        return "trace_id" in cols
    except sqlite3.Error:
        return False


# ─── Migrations registradas ─────────────────────────────────────────────────
#
# Wave inicial: las 4 ALTERs/CREATE INDEX trace_id documentadas en
# `_migrate_trace_id_columns()` en `rag/__init__.py`. Cuando agregues
# nuevas migrations, sumá `@migration(version=N, name="...")` con N
# monotónicamente creciente. NO renombres ni renumeres las anteriores
# (el hash de drift y la version primary key serían inválidos).
#
# Cada migration usa `_alter_add_column_if_missing` o `_create_index_if_missing`
# (helpers idempotentes adentro de la migration) para que sea segura
# correr una segunda vez por accidente sin tirar `duplicate column`.
# El bootstrap evita el caso común; estos helpers son belt-and-suspenders.


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def _alter_add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Idempotent ALTER TABLE ADD COLUMN. Si la tabla no existe (DB virgen
    antes de `_TELEMETRY_DDL`), no-op — el CREATE TABLE inicial ya va a
    crear la columna en su forma final. Si la columna ya existe, no-op."""
    if not _table_exists(conn, table):
        return
    try:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        cols = set()
    if column in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


@migration(version=1, name="add_trace_id_to_queries")
def migrate_001_add_trace_id_to_queries(conn: sqlite3.Connection) -> None:
    """Agrega `trace_id TEXT` a `rag_queries`. Audit 2026-04-25
    R2-Telemetry #1: correlación query → behavior → silent_errors via
    trace_id. En DBs fresh el CREATE TABLE en `_TELEMETRY_DDL` ya tiene
    la columna; este ALTER cubre las DBs pre-2026-04-25."""
    _alter_add_column_if_missing(conn, "rag_queries", "trace_id", "trace_id TEXT")


@migration(version=2, name="add_trace_id_to_behavior")
def migrate_002_add_trace_id_to_behavior(conn: sqlite3.Connection) -> None:
    """Agrega `trace_id TEXT` a `rag_behavior`. Mismo motivo que la 001."""
    _alter_add_column_if_missing(conn, "rag_behavior", "trace_id", "trace_id TEXT")


@migration(version=3, name="add_trace_id_to_ambient")
def migrate_003_add_trace_id_to_ambient(conn: sqlite3.Connection) -> None:
    """Agrega `trace_id TEXT` a `rag_ambient`. Audit 2026-04-27: 29
    `ambient_sql_write_failed: no such column: trace_id` en 14 días."""
    _alter_add_column_if_missing(conn, "rag_ambient", "trace_id", "trace_id TEXT")


@migration(version=4, name="index_rag_queries_trace_id")
def migrate_004_index_rag_queries_trace_id(conn: sqlite3.Connection) -> None:
    """Partial index sobre `rag_queries(trace_id) WHERE trace_id IS NOT
    NULL`. Cubre el bug audited 2026-04-26: `_migrate_trace_id_columns`
    intentaba crear el index en `_TELEMETRY_DDL`, lo que rompía
    `_ensure_telemetry_tables` en DBs pre-trace_id (la columna no
    existía cuando el CREATE INDEX corría). Ahora vive acá, ordenada
    DESPUÉS de las 3 ALTERs."""
    if not _table_exists(conn, "rag_queries"):
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_queries_trace_id "
        "ON rag_queries(trace_id) WHERE trace_id IS NOT NULL"
    )


@migration(version=5, name="create_mood_tables")
def migrate_005_create_mood_tables(conn: sqlite3.Connection) -> None:
    """Mood tracking tables (Fase A del pipeline mood, 2026-04-30).

    Dos tablas:

    - `rag_mood_signals` — append-only, una fila por señal individual
      capturada por los scorers en `rag/mood.py` (Spotify artist-mood,
      compulsive-repeat, late-night-listening, journal keyword-negative,
      journal note-sentiment, wa-outbound tone, queries existential,
      calendar density). Granularidad fina para auditar después qué
      señales movieron el score diario.

    - `rag_mood_score_daily` — una fila por día con el weighted-avg
      computado por `compute_daily_score()`. PRIMARY KEY date para que
      el agregador haga UPSERT idempotente (recalcular un día no
      duplica filas).

    Ambas tablas también están declaradas en `_TELEMETRY_DDL` (CREATE
    TABLE IF NOT EXISTS en boot). Esta migration es belt-and-suspenders
    para DBs ya bootstrapped (current_version=4) que necesitan saltar a 5.

    Behind flag `RAG_MOOD_ENABLED` — las tablas se crean siempre (cheap)
    pero los writers de `rag/mood.py` exit-early si el flag está off.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_mood_signals ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL NOT NULL,"
        " date TEXT NOT NULL,"
        " source TEXT NOT NULL,"
        " signal_kind TEXT NOT NULL,"
        " value REAL NOT NULL,"
        " weight REAL NOT NULL DEFAULT 1.0,"
        " evidence TEXT"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_mood_signals_date "
        "ON rag_mood_signals(date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_mood_signals_ts "
        "ON rag_mood_signals(ts DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_mood_signals_source_date "
        "ON rag_mood_signals(source, date)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_mood_score_daily ("
        " date TEXT PRIMARY KEY,"
        " score REAL NOT NULL,"
        " n_signals INTEGER NOT NULL,"
        " sources_used TEXT,"
        " top_evidence TEXT,"
        " updated_at REAL NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_mood_score_daily_updated "
        "ON rag_mood_score_daily(updated_at DESC)"
    )


__all__ = [
    "migration",
    "current_version",
    "applied_versions",
    "known_migrations",
    "apply_pending_migrations",
    "bootstrap_existing_db",
]
