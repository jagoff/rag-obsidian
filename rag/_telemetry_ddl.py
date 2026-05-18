"""Telemetry DDL — migrations + once-per-(proc, db-path) ensure.

Phase 1c de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer las 3 lazy migrations + `_ensure_telemetry_tables` (que es
el caller) + `generate_trace_id` desde `rag/__init__.py` (que ya bajó
a 63.9k LOC tras Phase 1a + 1b).

## Qué vive acá

- `generate_trace_id()` — 8 chars hex via `secrets.token_hex(4)`.
- `_TELEMETRY_DDL_ENSURED_PATHS` + `_TELEMETRY_DDL_LOCK` — ensure-once
  cache por `(proceso, db_path)`. Pre-fix el DDL corría en cada
  `_ragvec_state_conn()` open (~540/hr); ahora corre exactamente una
  vez por path durante la vida del proceso.
- `_migrate_audio_transcripts_phase2(conn)` — ALTER columnas Phase 2
  whisper-learning + indexes.
- `_migrate_cita_detections_add_kind(conn)` — ALTER cita_detections
  con `kind` + `reminder_id` + index.
- `_migrate_trace_id_columns(conn)` — ALTER 5 tablas con `trace_id`
  + 2 indexes partial.
- `_ensure_telemetry_tables(conn)` — orchestrator: PRAGMA defensive +
  fast-path (skip si todas las tablas existen) + slow-path (BEGIN
  IMMEDIATE + DDL batch + retry transient) + las 3 migrations + las
  versioned migrations en `rag.migrations`.

## Lazy imports

Las funciones acá tienen deps en symbols definidos LATER en
`rag/__init__.py`:
  `_silent_log` — error logger
  `_is_transient_sql_error` — clasificador de OperationalError
  `_TELEMETRY_DDL` — tuple de DDL statements (definido al principio
  del SQL state stack, antes de las migrations).

Top-level imports causarían circular import. Solución: lazy import
dentro de cada función. `rag.migrations` (versioned migrations
infrastructure) ya estaba siendo importado lazy en el path histórico.

## Re-export

`rag/__init__.py` hace `from rag._telemetry_ddl import *  # noqa`.
Preserva 100% compat con call sites históricos.
"""

from __future__ import annotations

import threading
import time

__all__ = [
    "generate_trace_id",
    "_TELEMETRY_DDL_ENSURED_PATHS",
    "_TELEMETRY_DDL_LOCK",
    "_migrate_audio_transcripts_phase2",
    "_migrate_cita_detections_add_kind",
    "_migrate_trace_id_columns",
    "_ensure_telemetry_tables",
]


def generate_trace_id() -> str:
    """Devuelve un trace_id corto (8 chars hex) para correlacionar
    queries / behavior / errores del mismo request.

    8 chars (32 bits) son suficientes para evitar colisiones dentro de
    la ventana de retention 90 días: con 50 queries/día = 4500 IDs por
    ventana, probabilidad de colisión ~1 en 10^6 (birthday). Si llegamos
    a colisionar, no es problema porque las tablas tienen ts también.
    """
    import secrets  # noqa: PLC0415
    return secrets.token_hex(4)  # 8 chars hex


# Per-DB-path once-flag para los DDL idempotentes. Pre-2026-04-24 cada
# `_ragvec_state_conn()` corría los ~28 CREATE TABLE IF NOT EXISTS + ALTER
# en cada conn open, aún cuando el caller no era cold-start (samplers cpu/
# memory abren ~120 conns/hr cada uno; behavior log abre 290/hr; 540+ conn
# opens/hr × 28 statements = 15K+ DDL statements/hr ejecutándose contra
# telemetry.db). Cada DDL es no-op cuando la tabla existe pero igual toma
# el schema lock brevemente — contribuye a la contention WAL que aparece
# como `database is locked` en los writers.
#
# Set keyed por path absoluto: tests que cambian DB_PATH a tmp dirs siguen
# disparando el DDL contra cada DB nueva (el flag es por path). Crash
# recovery: si el proceso muere antes del COMMIT, el flag no se setea
# para ese path, el próximo open reintenta. Multi-proceso: cada proceso
# ensure-once independientemente (las tablas son CREATE IF NOT EXISTS —
# race-safe).
_TELEMETRY_DDL_ENSURED_PATHS: set[str] = set()
_TELEMETRY_DDL_LOCK = threading.Lock()


def _migrate_audio_transcripts_phase2(conn) -> None:
    """Idempotent ALTER para DBs pre-2026-04-25 que ya tienen `rag_audio_transcripts`
    con el schema v1 (audio_path, mtime, text, language, duration_s, model,
    transcribed_at). Phase 2 learning loop (whatsapp-whisper-learning) suma
    7 columnas para soportar logging del listener + correcciones + LLM gating:

    - audio_hash: sha256(bytes), idempotencia + FK soft a rag_audio_corrections.
    - chat_id: para retrieval de chat tail + lookup de "última transcripción"
      por chat (que es lo que `/fix` necesita corregir).
    - avg_logprob: confidence promedio del segmento (whisper-server con
      response_format=verbose_json), usado por el threshold de LLM correction.
    - corrected_text + correction_source: trail de auditoría — original
      siempre queda en `text`, la versión final usada para el reply queda en
      `corrected_text` (cuando hubo correction; NULL en el path happy).
    - note_path + note_initial_hash: para vault_diff watcher detectar si
      el user editó manualmente la nota generada → genera una correction
      implícita.

    Sqlite no soporta `ADD COLUMN IF NOT EXISTS` — cada ALTER tira
    OperationalError ("duplicate column name") si la columna ya existe;
    lo tragamos. Idempotente: correr esto N veces es no-op.

    Llamado desde `_ensure_telemetry_tables` después del CREATE TABLE IF NOT
    EXISTS (que tiene las columnas nuevas en el DDL para installs frescos —
    este migration es para installs preexistentes).
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    new_cols = (
        "audio_hash TEXT",
        "chat_id TEXT",
        "avg_logprob REAL",
        "corrected_text TEXT",
        "correction_source TEXT",
        "note_path TEXT",
        "note_initial_hash TEXT",
    )
    for col_ddl in new_cols:
        try:
            conn.execute(f"ALTER TABLE rag_audio_transcripts ADD COLUMN {col_ddl}")
        except _sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                # Bug fix 2026-04-25: pre-fix esta rama tenía un `pass` que
                # tragaba SILENCIOSAMENTE cualquier error que no fuera
                # "duplicate column" (table not found, disk full, syntax
                # error en el DDL, etc.) sin notificar a nadie. Ahora lo
                # logueamos via _silent_log. El _ensure_telemetry_tables
                # caller también tiene un except más arriba, pero ESE solo
                # se dispara si la migration entera revienta — los errores
                # per-column quedaban invisibles.
                try:
                    _silent_log(
                        "migration_audio_transcripts_alter_failed", exc
                    )
                except Exception:  # pragma: no cover - log path itself failed
                    pass
    for idx_ddl in (
        "CREATE INDEX IF NOT EXISTS ix_audio_transcripts_chat_id ON rag_audio_transcripts(chat_id, transcribed_at)",
        "CREATE INDEX IF NOT EXISTS ix_audio_transcripts_hash ON rag_audio_transcripts(audio_hash)",
    ):
        try:
            conn.execute(idx_ddl)
        except Exception as _idx_exc:
            try:
                _silent_log(
                    "migration_audio_transcripts_index_failed", _idx_exc
                )
            except Exception:  # pragma: no cover
                pass


def _migrate_cita_detections_add_kind(conn) -> None:
    """Idempotent ALTER para DBs pre-2026-04-23 tarde que ya tienen la tabla
    `rag_cita_detections` con el schema inicial (solo `event_uid`, sin
    `kind` ni `reminder_id`).

    Sqlite no soporta `ADD COLUMN IF NOT EXISTS` — intentamos el ALTER y
    tragamos la OperationalError si la columna ya existe. Idempotente:
    correr esto varias veces es no-op.

    Llamado desde `_ensure_telemetry_tables` después del `CREATE TABLE IF
    NOT EXISTS`. Instalaciones nuevas ya tienen las columnas en el DDL;
    instalaciones existentes las suman acá.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    for col_ddl in (
        "ALTER TABLE rag_cita_detections ADD COLUMN kind TEXT",
        "ALTER TABLE rag_cita_detections ADD COLUMN reminder_id TEXT",
    ):
        try:
            conn.execute(col_ddl)
        except _sqlite3.OperationalError as exc:
            # "duplicate column name: kind" → ya migrado, no-op.
            if "duplicate column" not in str(exc).lower():
                # Bug fix 2026-04-25: pre-fix esta rama tenía un `pass`
                # que tragaba CUALQUIER otro error en silencio. El
                # comentario decía "se manifestará ruidosamente en el
                # primer INSERT" — pero los INSERTs son defensivos
                # (subset de columnas), así que en realidad nunca se
                # manifestaba. Ahora lo logueamos para detección.
                try:
                    _silent_log(
                        "migration_cita_detections_alter_failed", exc
                    )
                except Exception:  # pragma: no cover - log path itself failed
                    pass
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cita_detections_kind "
            "ON rag_cita_detections(kind)"
        )
    except Exception as _idx_exc:
        try:
            _silent_log("migration_cita_detections_index_failed", _idx_exc)
        except Exception:  # pragma: no cover
            pass


def _migrate_trace_id_columns(conn) -> None:
    """Idempotent ALTER para agregar ``trace_id`` a rag_queries y rag_behavior
    (audit 2026-04-25 R2-Telemetry #1).

    En DBs pre-2026-04-25 las tablas existen sin la columna. Intentamos
    el ALTER y tragamos OperationalError si la columna ya existe (cuando
    la DB es fresh, el CREATE TABLE arriba ya la creó). Después agregamos
    el índice partial idempotente.

    Sin trace_id, "el chat tardó 30s ayer 4pm" no se puede correlacionar
    contra rag_behavior (qué clicks hizo el user) ni silent_errors (qué
    falló). Con trace_id, un grep cruza las 3 fuentes en O(1).
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    # Audit 2026-04-26: agregamos cpu/memory metrics al ALTER. Antes del fix,
    # algún caller de los samplers `_cpu_persist`/`_memory_persist` inyectaba
    # `trace_id` al payload (correlación request-tracing). Las tablas
    # rag_cpu_metrics + rag_memory_metrics se crearon SIN la columna en el
    # DDL inicial → cada sample dispara `OperationalError: no such column:
    # trace_id` cada 60s. 333 errores en 7d, watchdog de memory-pressure
    # ciego. ALTER idempotente en migration ahora cubre las 4 tablas.
    #
    # Audit 2026-04-27: rag_ambient también faltaba. 29 errores confirmados
    # en sql_state_errors.jsonl del 2026-04-25 (ambient_sql_write_failed:
    # no such column: trace_id). Mismo patrón que cpu/memory metrics: la
    # tabla existía en DBs pre-trace_id sin la columna. ALTER idempotente
    # cubre la brecha para despliegues existentes; el CREATE TABLE en
    # _TELEMETRY_DDL ya tiene trace_id para fresh installs.
    for col_ddl in (
        "ALTER TABLE rag_queries ADD COLUMN trace_id TEXT",
        "ALTER TABLE rag_behavior ADD COLUMN trace_id TEXT",
        "ALTER TABLE rag_cpu_metrics ADD COLUMN trace_id TEXT",
        "ALTER TABLE rag_memory_metrics ADD COLUMN trace_id TEXT",
        "ALTER TABLE rag_ambient ADD COLUMN trace_id TEXT",
    ):
        try:
            conn.execute(col_ddl)
        except _sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                try:
                    _silent_log("migration_trace_id_alter_failed", exc)
                except Exception:
                    pass
    for idx_ddl in (
        "CREATE INDEX IF NOT EXISTS ix_rag_queries_trace_id "
        "ON rag_queries(trace_id) WHERE trace_id IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_rag_behavior_trace_id "
        "ON rag_behavior(trace_id) WHERE trace_id IS NOT NULL",
    ):
        try:
            conn.execute(idx_ddl)
        except Exception as _idx_exc:
            try:
                _silent_log("migration_trace_id_index_failed", _idx_exc)
            except Exception:
                pass


def _migrate_screen_obs_add_image_path(conn) -> None:
    """Idempotent ALTER para agregar `image_path TEXT` a `rag_screen_observations`
    (Peekaboo Fase 3, 2026-05-13).

    DBs pre-Fase-3 tienen la tabla sin la columna. Fresh installs ya la tienen
    via DDL. Tragamos `duplicate column` silenciosamente; cualquier otro error
    se loguea pero no aborta el bootstrap.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    try:
        conn.execute("ALTER TABLE rag_screen_observations ADD COLUMN image_path TEXT")
    except _sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower() and "no such table" not in str(exc).lower():
            try:
                _silent_log("migration_screen_obs_image_path_failed", exc)
            except Exception:
                pass


def _ensure_telemetry_tables(conn) -> None:
    """Create all rag_* telemetry tables + indexes + schema_version rows.

    Idempotent — relies on `CREATE TABLE IF NOT EXISTS` + `INSERT OR IGNORE`.
    Wraps all DDL in a single transaction so a crash mid-setup doesn't leave
    half-built schema. Also sets `synchronous=NORMAL` on this connection so
    telemetry writes survive power loss (vec reads keep their settings).

    Called from SqliteVecClient.__init__ when RAG_STATE_SQL=1, so a process
    que no opt-ineó nunca paga el DDL cost.

    Audit 2026-04-24: el DDL se ejecutaba en cada conn open (~540/hr). Con
    `_TELEMETRY_DDL_ENSURED_PATHS` ahora corre exactamente una vez por
    (proceso, db-path) — la primera invocación. Cada conn subsiguiente
    contra la misma DB sólo aplica el PRAGMA de synchronous (per-conn,
    requerido) y skip el DDL.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _is_transient_sql_error, _silent_log  # noqa: PLC0415
    from rag import _TELEMETRY_DDL  # noqa: PLC0415

    # 2026-04-30: PRAGMAs defensivos — si el conn entró ya en transacción
    # (callers que abren BEGIN antes de invocar), `PRAGMA synchronous` tira
    # "Safety level may not be changed inside a transaction". Skip silent.
    # Audit R2-7 #5: foreign_keys=ON requerido para que los `REFERENCES
    # ... ON DELETE CASCADE` declarados en el schema (ej.
    # rag_entity_mentions → rag_entities) se enforcen. SQLite por
    # default los IGNORA — borrar una entity dejaba rows huérfanas en
    # mentions hasta el prune. Pragma es per-connection.
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except _sqlite3.OperationalError:
        pass
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except _sqlite3.OperationalError:
        pass
    # Identificar la DB por su file path. `database` columna del
    # `database_list` PRAGMA da el path absoluto del main db (vacío para
    # in-memory, en cuyo caso nunca cacheamos — siempre re-ensure).
    try:
        _row = conn.execute("PRAGMA database_list").fetchone()
        db_path = _row[2] if _row else ""
    except Exception:
        db_path = ""
    if db_path and db_path in _TELEMETRY_DDL_ENSURED_PATHS:
        return
    with _TELEMETRY_DDL_LOCK:
        if db_path and db_path in _TELEMETRY_DDL_ENSURED_PATHS:
            return
        # Detect whether rag_schema_version exists. Created by SqliteVecClient on
        # its own, but this helper might run against a bare DB in tests.
        has_schema_version = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_schema_version'"
        ).fetchone())
        # Fast path 2026-04-30 (bug fix `database is locked` recurrente):
        # si TODAS las tablas declaradas en _TELEMETRY_DDL ya existen en
        # sqlite_master, otro proceso las creó. Saltamos el BEGIN write-lock
        # entero — resuelve el race del bootstrap concurrent (cron despierta
        # wa-scheduled-send mientras web server está escribiendo, BEGIN
        # bloquea por busy_timeout=60s y al final tira OperationalError
        # crasheando al worker). Una lectura SELECT FROM sqlite_master no
        # toma write lock, así que es seguro bajo contención.
        declared_tables = {name for name, _ in _TELEMETRY_DDL}
        existing_tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        skip_ddl = declared_tables.issubset(existing_tables)
        try:
            if skip_ddl:
                # Las tablas existen — solo aplicamos los inserts a
                # rag_schema_version (idempotent INSERT OR IGNORE).
                # Fix 2026-04-30: wrappear en BEGIN IMMEDIATE para que los N
                # INSERT OR IGNORE sean una sola transacción y no N
                # autocommit-transactions independientes. En modo autocommit
                # cada INSERT pelea por el write-lock por separado, produciendo
                # `database is locked` bajo contención con el web server.
                # BEGIN IMMEDIATE toma el write-lock UNA vez para todo el batch.
                if has_schema_version:
                    last_exc: Exception | None = None
                    for attempt in range(5):
                        try:
                            conn.execute("BEGIN IMMEDIATE")
                            conn.executemany(
                                "INSERT OR IGNORE INTO rag_schema_version(table_name, version) VALUES(?, 1)",
                                [(name,) for name, _ in _TELEMETRY_DDL],
                            )
                            conn.execute("COMMIT")
                            last_exc = None
                            break
                        except _sqlite3.OperationalError as exc:
                            try:
                                conn.execute("ROLLBACK")
                            except _sqlite3.Error:
                                pass
                            if not _is_transient_sql_error(exc):
                                raise
                            last_exc = exc
                            time.sleep(0.5 * (2 ** attempt))  # 0.5, 1, 2, 4, 8s
                    if last_exc is not None:
                        raise last_exc
            else:
                # Slow path: hacer el bootstrap completo con retry transient.
                # Fix 2026-04-30: BEGIN → BEGIN IMMEDIATE para tomar el
                # write-lock inmediatamente, no diferido hasta el primer write.
                # Con BEGIN DEFERRED la contención se detecta tarde (al primer
                # CREATE TABLE / INSERT) en vez de al BEGIN, generando un burst
                # de OperationalError difícil de diagnosticar.
                last_exc: Exception | None = None
                for attempt in range(5):
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        for _table_name, stmts in _TELEMETRY_DDL:
                            for stmt in stmts:
                                conn.execute(stmt)
                        if has_schema_version:
                            conn.executemany(
                                "INSERT OR IGNORE INTO rag_schema_version(table_name, version) VALUES(?, 1)",
                                [(name,) for name, _ in _TELEMETRY_DDL],
                            )
                        conn.execute("COMMIT")
                        last_exc = None
                        break
                    except _sqlite3.OperationalError as exc:
                        try:
                            conn.execute("ROLLBACK")
                        except _sqlite3.Error:
                            pass
                        if not _is_transient_sql_error(exc):
                            raise
                        last_exc = exc
                        # Antes de reintentar: ¿completó otro proceso el
                        # bootstrap mientras esperábamos? Si sí, salir.
                        existing_after = {
                            r[0] for r in conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table'"
                            ).fetchall()
                        }
                        if declared_tables.issubset(existing_after):
                            last_exc = None
                            break
                        time.sleep(0.5 * (2 ** attempt))  # 0.5, 1, 2, 4, 8s
                if last_exc is not None:
                    raise last_exc
        except _sqlite3.Error:
            try:
                conn.execute("ROLLBACK")
            except _sqlite3.Error:
                pass
            raise
        # Lazy schema migrations — corren FUERA de la transaction principal
        # porque ALTER TABLE no siempre es transactable y no queremos abortar
        # el DDL batch entero si un ALTER de una sola columna falla.
        #
        # Audit follow-up 2026-04-25: pre-fix los except eran `pass` puros, lo
        # que escondía migrations rotas indefinidamente — el siguiente boot
        # las reintentaba y volvía a fallar sin que nadie se enterara. Ahora
        # los logueamos via _silent_log para que aparezcan en el dashboard
        # `/api/status/errors`. Seguimos NO re-raising — el INSERT del writer
        # es defensivo y usa el subset de columnas que sí existen.
        try:
            import rag as _rag  # noqa: PLC0415
            getattr(_rag, "_migrate_cita_detections_add_kind", _migrate_cita_detections_add_kind)(conn)
        except Exception as _migrate_exc:
            try:
                _silent_log("migration_cita_detections_failed", _migrate_exc)
            except Exception:  # pragma: no cover - log path itself failed
                pass
        try:
            import rag as _rag  # noqa: PLC0415
            getattr(_rag, "_migrate_audio_transcripts_phase2", _migrate_audio_transcripts_phase2)(conn)
        except Exception as _migrate_exc:
            try:
                _silent_log("migration_audio_transcripts_failed", _migrate_exc)
            except Exception:  # pragma: no cover - log path itself failed
                pass
        try:
            import rag as _rag  # noqa: PLC0415
            getattr(_rag, "_migrate_trace_id_columns", _migrate_trace_id_columns)(conn)
        except Exception as _migrate_exc:
            try:
                _silent_log("migration_trace_id_failed", _migrate_exc)
            except Exception:  # pragma: no cover
                pass
        try:
            import rag as _rag  # noqa: PLC0415
            getattr(_rag, "_migrate_screen_obs_add_image_path", _migrate_screen_obs_add_image_path)(conn)
        except Exception as _migrate_exc:
            try:
                _silent_log("migration_screen_obs_image_path_failed", _migrate_exc)
            except Exception:  # pragma: no cover
                pass
        # Versioned schema migrations (2026-04-29). Single source of truth para
        # cambios de schema futuros — los `_migrate_*` helpers de arriba son
        # legacy belt-and-suspenders mientras se transicionan. Bootstrap si la
        # DB es preexistente con todas las migrations ya aplicadas via los
        # helpers legacy (heurística: rag_queries.trace_id ya existe). Después
        # apply_pending_migrations es idempotente — un solo SELECT MAX cuando
        # estamos up-to-date. Wrapped en try/except con _silent_log: si una
        # migration tira, app sigue degradada en vez de crashear el boot.
        try:
            from rag import migrations as _migrations  # noqa: PLC0415
            try:
                _migrations.bootstrap_existing_db(conn)
            except Exception as _boot_exc:
                try:
                    _silent_log("schema_migrations_bootstrap_failed", _boot_exc)
                except Exception:  # pragma: no cover
                    pass
            try:
                _migrations.apply_pending_migrations(conn)
            except Exception as _apply_exc:
                try:
                    _silent_log("schema_migrations_apply_failed", _apply_exc)
                except Exception:  # pragma: no cover
                    pass
        except Exception as _mig_import_exc:  # pragma: no cover
            try:
                _silent_log("schema_migrations_import_failed", _mig_import_exc)
            except Exception:
                pass
        if db_path:
            _TELEMETRY_DDL_ENSURED_PATHS.add(db_path)
