"""WhatsApp scheduled messages — programar envíos a futuro (Phase 4c, 2026-04-25).

Wave nueva post split modular: paralela a `rag/wa_tasks.py` (extractor) y
`rag/proactive.py` (push agente), pero acá la primitiva es **schedule + run_due_worker**.
El user (o el chat LLM) le pide "mandale 'feliz cumple' a Grecia mañana 9hs",
el endpoint `/api/whatsapp/schedule` parsea el lenguaje natural a UTC y llama
`schedule()`. El launchd `com.fer.obsidian-rag-wa-scheduled.plist` corre cada
5 min y dispara `run_due_worker()` que pickea pendings con
`scheduled_for_utc <= now` y los manda vía el bridge local.

## Surfaces

- `_ensure_schema(conn)` — CREATE TABLE IF NOT EXISTS (lazy, idempotente).
- `schedule(jid, message_text, scheduled_for_utc, *, contact_name, reply_to,
  proposal_id, source, conn)` — INSERT row pending, retorna `{id, scheduled_for_utc, status}`.
- `cancel(scheduled_id, *, reason, conn)` — UPDATE → 'cancelled' si seguía pending.
- `list_scheduled(*, status, limit, since_iso, conn)` — query con filtros.
- `get_scheduled(scheduled_id, *, conn)` — fetch por id.
- `reschedule(scheduled_id, new_scheduled_for_utc, *, conn)` — UPDATE
  scheduled_for_utc si seguía pending.
- `run_due_worker(now, *, late_threshold_minutes, max_retries, max_per_run,
  dry_run, conn)` — el loop del cron.

## TZ handling

La DB guarda **siempre** UTC. Las funciones aceptan datetime aware (UTC),
naive (asumido UTC), o string ISO8601 (con o sin sufijo `Z`). El parsing de
lenguaje natural ("mañana 9hs", "el lunes que viene a las 14") **no** vive
acá — lo hace el endpoint que llama a `schedule()`. Este módulo solo recibe
instantes UTC concretos.

## Silent-fail contract

`run_due_worker` está envuelto en try/except amplio: cualquier excepción no
esperada se loguea a `rag_ambient` con `cmd='whatsapp_scheduled_worker_error'`
y la función retorna `{ok: False, reason: <repr>}`. Nunca debe hacer crash el
launchd — si truena el plist queda en estado `errored` y deja de correr hasta
que el user lo reactive a mano.

Las funciones interactivas (`schedule`, `cancel`, `reschedule`) **sí** elevan
ValueError ante input malo — el caller (HTTP endpoint) lo traduce a 400.

## Privacy

`_log_ambient(cmd, payload)` recibe metadata operativa (jid, scheduled_id,
delta_minutes, attempt_count, ...) pero **nunca** el body del mensaje. El
texto solo vive en la tabla `rag_whatsapp_scheduled` (que es local, igual
que el resto del telemetry).

## Why deferred imports

Igual que `proactive.py` y `wa_tasks.py`: `_ragvec_state_conn`,
`_ambient_log_event` y `_whatsapp_send_to_jid` viven en
`rag/__init__.py` y `rag.integrations.whatsapp` respectivamente. Para que
`monkeypatch.setattr(rag, "_ragvec_state_conn", ...)` propague desde tests,
los importamos dentro de cada function body.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
_TABLE = "rag_whatsapp_scheduled"
_VALID_STATUSES: tuple[str, ...] = (
    "pending", "processing", "sent", "sent_late", "failed", "cancelled",
)
# Si un row queda en 'processing' por más de este threshold, asumimos
# que el worker que lo agarró crasheó antes de finalizar. Lo reseteamos
# a 'pending' al inicio del siguiente run para que se reintente. Threshold
# debe ser > timeout máximo del bridge (10s) + margen — 5min cubre con
# holgura el caso plist tick + manual run solapados.
_STALE_PROCESSING_MINUTES = 5
_VALID_SOURCES: tuple[str, ...] = ("chat", "dashboard", "nl")
# Margen para clock skew: rechazamos pasado solo si está más de 60s atrás.
_PAST_TOLERANCE_SECONDS = 60
# Anti-acumulación: nadie programa razonablemente >1 año al futuro.
_MAX_FUTURE_DAYS = 365
# Truncado del message_text en respuestas de list_scheduled (no inflar API).
_LIST_TEXT_MAX_CHARS = 500

# Columnas en orden — usado por SELECT y _row_to_dict para no duplicar.
_COLS: tuple[str, ...] = (
    "id", "created_at", "scheduled_for_utc", "jid", "contact_name",
    "message_text", "reply_to_id", "reply_to_text", "reply_to_sender_jid",
    "status", "attempt_count", "last_error", "last_attempt_at",
    "sent_at", "delta_minutes", "proposal_id", "source",
)
_SELECT_COLS_SQL = ", ".join(_COLS)


_SCHEMA_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      scheduled_for_utc TEXT NOT NULL,
      jid TEXT NOT NULL,
      contact_name TEXT,
      message_text TEXT NOT NULL,
      reply_to_id TEXT,
      reply_to_text TEXT,
      reply_to_sender_jid TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      attempt_count INTEGER NOT NULL DEFAULT 0,
      last_error TEXT,
      last_attempt_at TEXT,
      sent_at TEXT,
      delta_minutes INTEGER,
      proposal_id TEXT,
      source TEXT NOT NULL DEFAULT 'chat'
    )""",
    f"CREATE INDEX IF NOT EXISTS ix_rag_whatsapp_scheduled_for "
    f"ON {_TABLE}(scheduled_for_utc)",
    f"CREATE INDEX IF NOT EXISTS ix_rag_whatsapp_scheduled_status "
    f"ON {_TABLE}(status)",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """CREATE TABLE/INDEX IF NOT EXISTS. Idempotente, cheap.

    Llamado lazy desde cada función pública antes de tocar la tabla. No
    hace falta agregarlo al `_ensure_telemetry_tables` global de
    `rag/__init__.py` porque el costo es negligible (sqlite chequea el
    sqlite_master en O(log n)) y nos evita touch del módulo principal en
    esta wave.
    """
    for stmt in _SCHEMA_DDL:
        conn.execute(stmt)
    try:
        conn.commit()
    except Exception:
        # isolation_level=None → autocommit, commit() es no-op pero no falla.
        pass


@contextlib.contextmanager
def _resolve_conn(conn: Optional[sqlite3.Connection]) -> Iterator[sqlite3.Connection]:
    """Reusa `conn` si se pasó, sino abre `_ragvec_state_conn()` (telemetry.db).
    En ambos casos asegura el schema antes de yieldear.

    Si abrimos nosotros, el cierre lo maneja el context manager del padre
    (`_ragvec_state_conn` ya cierra en su `finally`). Si nos pasaron conn,
    NO la cerramos — es del caller.
    """
    if conn is not None:
        _ensure_schema(conn)
        yield conn
        return
    # Lazy import: `_ragvec_state_conn` vive en `rag/__init__.py` y este
    # módulo se carga durante el load del package. Importarlo top-level
    # crea ciclo. La función-body import también respeta monkeypatches.
    from rag import _ragvec_state_conn
    with _ragvec_state_conn() as own:
        _ensure_schema(own)
        yield own


def _now_utc() -> datetime:
    """`datetime.now(timezone.utc)` como helper testeable."""
    return datetime.now(timezone.utc)


def _to_utc_dt(value: Any) -> datetime:
    """Normaliza datetime/string a datetime aware UTC.

    - `datetime` aware → astimezone(UTC).
    - `datetime` naive → asumido UTC (replace(tzinfo=UTC)).
    - `str` ISO8601 → parseado; `Z` final tratado como `+00:00`. Si queda
      naive, asumido UTC.

    Raise `ValueError` si el string no es parseable o el tipo no se reconoce.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("scheduled_for_utc string vacío")
        if s.endswith("Z") or s.endswith("z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as exc:
            raise ValueError(
                f"scheduled_for_utc no parseable: {value!r} ({exc})"
            ) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(
        f"scheduled_for_utc tipo no soportado: {type(value).__name__}"
    )


def _iso_utc(dt: datetime) -> str:
    """ISO8601 segundo-truncated en UTC. Ej: `2026-04-25T13:00:00+00:00`."""
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _validate_schedule_time(dt_utc: datetime, *, now: datetime | None = None) -> None:
    """Eleva ValueError si `dt_utc` está demasiado en el pasado o muy al futuro.

    Margen de 60s al pasado para tolerar clock skew (un mensaje "manda ya"
    puede llegar acá unos segundos después). Cap de 365 días al futuro
    para evitar acumular cosas que el user ya no quiere recordar.
    """
    now = now or _now_utc()
    past_drift = (now - dt_utc).total_seconds()
    if past_drift > _PAST_TOLERANCE_SECONDS:
        raise ValueError(
            f"scheduled_for_utc en el pasado por más de {_PAST_TOLERANCE_SECONDS}s: "
            f"{_iso_utc(dt_utc)} (ahora {_iso_utc(now)}, drift={int(past_drift)}s)"
        )
    if (dt_utc - now) > timedelta(days=_MAX_FUTURE_DAYS):
        raise ValueError(
            f"scheduled_for_utc > now + {_MAX_FUTURE_DAYS} días — rechazado "
            f"(anti-acumulación): {_iso_utc(dt_utc)}"
        )


def _row_to_dict(row: tuple) -> dict:
    """Mapea una row del SELECT al dict canónico (orden definido en `_COLS`)."""
    return dict(zip(_COLS, row))


def _log_ambient(cmd: str, payload: dict | None = None) -> None:
    """Append a `rag_ambient` con metadata operativa. Body del mensaje
    NUNCA va al log (privacy). Silent-fail: si el helper no está disponible
    o la DB está lockeada, swallow — la funcionalidad principal sigue.

    `cmd` es algo como `'whatsapp_scheduled_sent'` /
    `'whatsapp_scheduled_failed'` / `'whatsapp_scheduled_worker_error'`.
    Lo no-conocido del payload se vuelca a `payload_json` por
    `_map_ambient_row` upstream.
    """
    try:
        from rag import _ambient_log_event
        event: dict = {"cmd": cmd}
        if payload:
            event.update(payload)
        _ambient_log_event(event)
    except Exception as exc:
        logger.debug("wa_scheduled_log_ambient_failed cmd=%s err=%r", cmd, exc)


# ── Public API ───────────────────────────────────────────────────────────────


def schedule(
    jid: str,
    message_text: str,
    scheduled_for_utc: Any,
    *,
    contact_name: str | None = None,
    reply_to: dict | None = None,
    proposal_id: str | None = None,
    source: str = "chat",
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """Programa un mensaje de WhatsApp para enviar en `scheduled_for_utc`.

    Validaciones:
      - `jid` no vacío y contiene `'@'`.
      - `message_text` no vacío post-strip.
      - `scheduled_for_utc` parseable a datetime aware UTC; ni más de 60s
        en el pasado ni más de 365 días al futuro.
      - `source` ∈ `{'chat', 'dashboard', 'nl'}`.
      - `reply_to` opcional: dict con keys `message_id|id`, `original_text|text`,
        `sender_jid|from_jid`. None ignora.

    Retorna `{"id": int, "scheduled_for_utc": str_iso, "status": "pending"}`.

    Raise `ValueError` si algo no valida — el endpoint HTTP lo traduce a 400.
    """
    if not isinstance(jid, str) or "@" not in jid or not jid.strip():
        raise ValueError(f"jid inválido (debe ser str no vacío con '@'): {jid!r}")
    if not isinstance(message_text, str):
        raise ValueError(f"message_text debe ser str: {type(message_text).__name__}")
    text = message_text.strip()
    if not text:
        raise ValueError("message_text vacío post-strip")
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"source inválido: {source!r} (válidos: {_VALID_SOURCES})"
        )

    sched_dt = _to_utc_dt(scheduled_for_utc)
    _validate_schedule_time(sched_dt)

    reply_to_id: str | None = None
    reply_to_text: str | None = None
    reply_to_sender: str | None = None
    if reply_to is not None:
        if not isinstance(reply_to, dict):
            raise ValueError(
                f"reply_to debe ser dict o None: {type(reply_to).__name__}"
            )
        rt_id = reply_to.get("message_id") or reply_to.get("id")
        if rt_id:
            reply_to_id = str(rt_id)
            rt_text = reply_to.get("original_text") or reply_to.get("text")
            if rt_text:
                # Cap a 1024 chars — coincide con el cap del send path.
                reply_to_text = str(rt_text)[:1024]
            rt_sender = reply_to.get("sender_jid") or reply_to.get("from_jid")
            if rt_sender:
                reply_to_sender = str(rt_sender)

    now_iso = _iso_utc(_now_utc())
    sched_iso = _iso_utc(sched_dt)

    with _resolve_conn(conn) as c:
        cur = c.execute(
            f"INSERT INTO {_TABLE} ("
            f"created_at, scheduled_for_utc, jid, contact_name, message_text, "
            f"reply_to_id, reply_to_text, reply_to_sender_jid, status, "
            f"attempt_count, proposal_id, source"
            f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)",
            (
                now_iso, sched_iso, jid.strip(),
                (contact_name or None),
                text,
                reply_to_id, reply_to_text, reply_to_sender,
                proposal_id, source,
            ),
        )
        new_id = int(cur.lastrowid or 0)
        try:
            c.commit()
        except Exception:
            pass

    _log_ambient("whatsapp_scheduled_created", {
        "scheduled_id": new_id,
        "jid": jid.strip(),
        "scheduled_for_utc": sched_iso,
        "source": source,
        "proposal_id": proposal_id,
        "has_reply_to": bool(reply_to_id),
    })

    return {
        "id": new_id,
        "scheduled_for_utc": sched_iso,
        "status": "pending",
    }


def cancel(
    scheduled_id: int,
    *,
    reason: str = "user_cancel",
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Cancela un scheduled si seguía en `pending`. Retorna True si cambió
    el estado, False si ya estaba en otro status (sent/sent_late/failed/
    cancelled) o el id no existe.

    `reason` queda guardado en `last_error` como `cancelled:<reason>` para
    que el dashboard pueda mostrar por qué se canceló (ej. 'user_cancel',
    'rescheduled', 'replaced_by_X').
    """
    sid = int(scheduled_id)
    with _resolve_conn(conn) as c:
        cur = c.execute(
            f"UPDATE {_TABLE} SET status='cancelled', last_error=? "
            f"WHERE id=? AND status='pending'",
            (f"cancelled:{reason}", sid),
        )
        try:
            c.commit()
        except Exception:
            pass
        changed = (cur.rowcount or 0) > 0

    if changed:
        _log_ambient("whatsapp_scheduled_cancelled", {
            "scheduled_id": sid,
            "reason": reason,
        })
    return changed


def list_scheduled(
    *,
    status: str | None = None,
    limit: int = 200,
    since_iso: str | None = None,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """Lista scheduled con filtros opcionales.

    - `status` filtra por estado (debe estar en `_VALID_STATUSES`); None = todos.
    - `since_iso` filtra `created_at >= since_iso` (string ISO8601 UTC).
    - `limit` cap de filas (default 200).

    Orden:
      - Si `status='pending'`: por `scheduled_for_utc ASC` (lo más urgente primero).
      - Si no: por `created_at DESC` (lo más reciente primero, pa' un timeline).

    `message_text` se trunca a 500 chars en cada row para no inflar la API
    cuando el dashboard lista 200 entries con mensajes largos.
    """
    where: list[str] = []
    params: list[Any] = []
    if status is not None:
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"status inválido: {status!r} (válidos: {_VALID_STATUSES})"
            )
        where.append("status = ?")
        params.append(status)
    if since_iso is not None:
        if not isinstance(since_iso, str) or not since_iso.strip():
            raise ValueError("since_iso debe ser string ISO8601 no vacío")
        where.append("created_at >= ?")
        params.append(since_iso.strip())

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    order_sql = (
        "scheduled_for_utc ASC"
        if status == "pending"
        else "created_at DESC"
    )
    lim = max(1, min(int(limit), 5000))

    sql = (
        f"SELECT {_SELECT_COLS_SQL} FROM {_TABLE}"
        f"{where_sql} ORDER BY {order_sql} LIMIT ?"
    )
    params.append(lim)

    with _resolve_conn(conn) as c:
        rows = c.execute(sql, params).fetchall()

    out: list[dict] = []
    for row in rows:
        d = _row_to_dict(row)
        msg = d.get("message_text") or ""
        if isinstance(msg, str) and len(msg) > _LIST_TEXT_MAX_CHARS:
            d["message_text"] = msg[:_LIST_TEXT_MAX_CHARS]
        out.append(d)
    return out


def get_scheduled(
    scheduled_id: int,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> dict | None:
    """Fetch por id. Retorna dict canónico o None si no existe.

    No trunca `message_text` (a diferencia de `list_scheduled`) porque acá
    el caller pidió explícitamente uno y probablemente quiere el body
    completo (ej. para mostrar en el detalle del card).
    """
    sid = int(scheduled_id)
    with _resolve_conn(conn) as c:
        row = c.execute(
            f"SELECT {_SELECT_COLS_SQL} FROM {_TABLE} WHERE id = ?",
            (sid,),
        ).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def reschedule(
    scheduled_id: int,
    new_scheduled_for_utc: Any,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Cambia `scheduled_for_utc` si seguía en `pending`. Retorna True si
    cambió, False si ya no estaba pending o el id no existe.

    Valida la nueva fecha igual que `schedule()` (no pasado >60s, no más
    de 365 días al futuro).
    """
    new_dt = _to_utc_dt(new_scheduled_for_utc)
    _validate_schedule_time(new_dt)
    new_iso = _iso_utc(new_dt)
    sid = int(scheduled_id)

    with _resolve_conn(conn) as c:
        cur = c.execute(
            f"UPDATE {_TABLE} SET scheduled_for_utc = ? "
            f"WHERE id = ? AND status = 'pending'",
            (new_iso, sid),
        )
        try:
            c.commit()
        except Exception:
            pass
        changed = (cur.rowcount or 0) > 0

    if changed:
        _log_ambient("whatsapp_scheduled_rescheduled", {
            "scheduled_id": sid,
            "new_scheduled_for_utc": new_iso,
        })
    return changed


def run_due_worker(
    now: Any = None,
    *,
    late_threshold_minutes: int = 5,
    max_retries: int = 5,
    max_per_run: int = 20,
    dry_run: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """Loop principal del cron. Pickea pending con `scheduled_for_utc <= now`
    y los manda vía el bridge local de WhatsApp.

    Args:
      now: datetime aware UTC, naive (asumido UTC), string ISO8601, o None
        (default: `datetime.now(timezone.utc)`). Inyectable para tests.
      late_threshold_minutes: si el envío llega <= N min tarde → status='sent'.
        Si > N → status='sent_late'. En ambos casos se manda igual (decisión
        del user: prefiere late > nada).
      max_retries: tras `attempt_count >= max_retries` fallidos → status='failed'.
        Sin backoff exponencial — el plist ya dispara cada 5min, retry natural.
      max_per_run: cap de filas por invocación (LIMIT del SELECT).
      dry_run: solo computa y loguea, no UPDATE ni envía.
      conn: opcional, reusa la conn del caller.

    Retorna resumen `{ok, processed, sent, sent_late, failed, retried, errors}`.
    Si algo no esperado truena, retorna `{ok: False, reason: <repr>}` y loguea
    a `rag_ambient` con cmd `'whatsapp_scheduled_worker_error'` — silent-fail
    contract para que el launchd nunca quede en estado 'errored'.
    """
    summary: dict = {
        "ok": True,
        "processed": 0,
        "sent": 0,
        "sent_late": 0,
        "failed": 0,
        "retried": 0,
        "errors": [],
    }
    try:
        now_dt = _to_utc_dt(now) if now is not None else _now_utc()
        now_iso = _iso_utc(now_dt)

        with _resolve_conn(conn) as c:
            # ── Step 0: recovery de filas "huérfanas" en 'processing' ──
            # Si otro worker crasheó entre acquire y finalize, sus rows
            # quedan trabadas en 'processing'. Las reseteamos a 'pending'
            # si llevan más de _STALE_PROCESSING_MINUTES (default 5min).
            # Sin esto, los mensajes quedan stuck para siempre. Solo
            # corremos esto en modo no-dry-run (en dry-run no queremos
            # mutaciones a la tabla).
            if not dry_run:
                stale_threshold = _iso_utc(
                    now_dt - timedelta(minutes=_STALE_PROCESSING_MINUTES)
                )
                try:
                    recovered = c.execute(
                        f"UPDATE {_TABLE} SET status = 'pending' "
                        f"WHERE status = 'processing' "
                        f"AND last_attempt_at IS NOT NULL "
                        f"AND last_attempt_at < ?",
                        (stale_threshold,),
                    ).rowcount
                    c.commit()
                    if recovered:
                        _log_ambient("whatsapp_scheduled_processing_recovered", {
                            "count": recovered,
                            "stale_threshold_minutes": _STALE_PROCESSING_MINUTES,
                        })
                except Exception:
                    pass

            rows = c.execute(
                f"SELECT {_SELECT_COLS_SQL} FROM {_TABLE} "
                f"WHERE status = 'pending' AND scheduled_for_utc <= ? "
                f"ORDER BY scheduled_for_utc ASC LIMIT ?",
                (now_iso, max(1, int(max_per_run))),
            ).fetchall()

            # Lazy import del send path: viene de `rag.integrations.whatsapp`
            # que a su vez hace late-import de `rag` — para evitar ciclos al
            # load-time, lo resolvemos acá y solo si hay rows que mandar.
            send_fn = None
            if rows and not dry_run:
                from rag.integrations.whatsapp import _whatsapp_send_to_jid as send_fn  # noqa: F401

            for r in rows:
                try:
                    sched = _row_to_dict(r)
                    sched_dt = _to_utc_dt(sched["scheduled_for_utc"])
                    delta_min = int((now_dt - sched_dt).total_seconds() // 60)
                    target_status = (
                        "sent"
                        if delta_min <= int(late_threshold_minutes)
                        else "sent_late"
                    )

                    reply_to_dict: dict | None = None
                    if sched.get("reply_to_id"):
                        reply_to_dict = {
                            "message_id": sched["reply_to_id"],
                            "original_text": sched.get("reply_to_text") or "",
                            "sender_jid": sched.get("reply_to_sender_jid") or "",
                        }

                    if dry_run:
                        # Dry-run NO acquire ni manda — solo loguea lo que haría.
                        summary["processed"] += 1
                        _log_ambient("whatsapp_scheduled_dry_run", {
                            "scheduled_id": sched["id"],
                            "jid": sched["jid"],
                            "scheduled_for_utc": sched["scheduled_for_utc"],
                            "delta_minutes": delta_min,
                            "would_status": target_status,
                            "attempt_count": sched.get("attempt_count", 0),
                        })
                        continue

                    # ── Atomic acquire: pending → processing ──────────
                    # Esta UPDATE es la barrera de exclusión mutua. Si otro
                    # worker ya agarró este row (status != 'pending'), el
                    # rowcount es 0 y skipeamos sin mandar. Resuelve la
                    # race condition donde 2 workers leen el mismo SELECT
                    # y mandan duplicado al destinatario.
                    acquired = c.execute(
                        f"UPDATE {_TABLE} SET status = 'processing', "
                        f"last_attempt_at = ? "
                        f"WHERE id = ? AND status = 'pending'",
                        (now_iso, sched["id"]),
                    ).rowcount
                    try:
                        c.commit()
                    except Exception:
                        pass
                    if not acquired:
                        # Otro worker se lo llevó entre nuestro SELECT y
                        # este UPDATE. Skip silencioso — él lo va a finalizar.
                        continue

                    summary["processed"] += 1

                    sent_ok = False
                    try:
                        sent_ok = bool(send_fn(  # type: ignore[misc]
                            sched["jid"],
                            sched["message_text"],
                            anti_loop=False,
                            reply_to=reply_to_dict,
                        ))
                    except Exception as send_exc:
                        sent_ok = False
                        summary["errors"].append(
                            f"id={sched['id']} send_exception={send_exc!r}"
                        )
                        logger.exception(
                            "wa_scheduled_send_exception id=%s: %r",
                            sched["id"], send_exc,
                        )

                    new_attempts = int(sched.get("attempt_count") or 0) + 1

                    if sent_ok:
                        # Finalize: processing → sent / sent_late.
                        # Matchear status='processing' en el WHERE garantiza
                        # que solo finalize si seguimos siendo los dueños
                        # (defensivo aunque el acquire ya nos dio exclusión).
                        c.execute(
                            f"UPDATE {_TABLE} SET status = ?, sent_at = ?, "
                            f"delta_minutes = ?, last_attempt_at = ?, "
                            f"attempt_count = ?, last_error = NULL "
                            f"WHERE id = ? AND status = 'processing'",
                            (target_status, now_iso, delta_min, now_iso,
                             new_attempts, sched["id"]),
                        )
                        try:
                            c.commit()
                        except Exception:
                            pass
                        if target_status == "sent":
                            summary["sent"] += 1
                            _log_ambient("whatsapp_scheduled_sent", {
                                "scheduled_id": sched["id"],
                                "jid": sched["jid"],
                                "scheduled_for_utc": sched["scheduled_for_utc"],
                                "delta_minutes": delta_min,
                                "attempt_count": new_attempts,
                            })
                        else:
                            summary["sent_late"] += 1
                            _log_ambient("whatsapp_scheduled_sent_late", {
                                "scheduled_id": sched["id"],
                                "jid": sched["jid"],
                                "scheduled_for_utc": sched["scheduled_for_utc"],
                                "delta_minutes": delta_min,
                                "attempt_count": new_attempts,
                            })
                    else:
                        if new_attempts >= int(max_retries):
                            # Tras N fallos: processing → failed (terminal,
                            # no se reintenta más sin intervención manual).
                            c.execute(
                                f"UPDATE {_TABLE} SET status = 'failed', "
                                f"last_error = ?, last_attempt_at = ?, "
                                f"attempt_count = ? "
                                f"WHERE id = ? AND status = 'processing'",
                                ("send_failed", now_iso, new_attempts,
                                 sched["id"]),
                            )
                            try:
                                c.commit()
                            except Exception:
                                pass
                            summary["failed"] += 1
                            _log_ambient("whatsapp_scheduled_failed", {
                                "scheduled_id": sched["id"],
                                "jid": sched["jid"],
                                "scheduled_for_utc": sched["scheduled_for_utc"],
                                "attempt_count": new_attempts,
                                "last_error": "send_failed",
                            })
                        else:
                            # Failure recoverable: processing → pending para
                            # que el próximo tick lo agarre de nuevo. Sin
                            # esto, el row se queda atrapado en 'processing'
                            # esperando que el recovery loop lo libere
                            # (5min de delay innecesario).
                            c.execute(
                                f"UPDATE {_TABLE} SET status = 'pending', "
                                f"last_error = ?, last_attempt_at = ?, "
                                f"attempt_count = ? "
                                f"WHERE id = ? AND status = 'processing'",
                                ("send_failed", now_iso, new_attempts,
                                 sched["id"]),
                            )
                            try:
                                c.commit()
                            except Exception:
                                pass
                            summary["retried"] += 1
                            _log_ambient("whatsapp_scheduled_retry", {
                                "scheduled_id": sched["id"],
                                "jid": sched["jid"],
                                "scheduled_for_utc": sched["scheduled_for_utc"],
                                "attempt_count": new_attempts,
                                "last_error": "send_failed",
                            })

                except Exception as item_exc:
                    # Una row mala no debe romper toda la batch — log y sigo.
                    summary["errors"].append(
                        f"item_exception={item_exc!r}"
                    )
                    logger.exception(
                        "wa_scheduled_item_failed: %r", item_exc,
                    )

        return summary

    except Exception as exc:
        # Silent-fail wide net: el launchd no debe quedar en estado errored.
        logger.exception("wa_scheduled_worker_error: %r", exc)
        try:
            _log_ambient("whatsapp_scheduled_worker_error", {
                "reason": repr(exc),
            })
        except Exception:
            pass
        return {"ok": False, "reason": str(exc)}
