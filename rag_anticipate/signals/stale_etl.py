"""Signal — Stale ETL / supervisor job watchdog.

Emite un push cuando un job in-process del supervisor (ej. ``auto_harvest``,
``wa_tasks``, ``anticipate``) NO ha corrido con éxito (``exit_code=0``) en
las últimas N horas, asumiendo que sí corría antes (rolling-window
heuristic, no triggers de "primer run jamás").

## Diseño

- Lee ``rag_supervisor_jobs`` (telemetry.db) — schema en
  ``rag/runtime/_telemetry.py``.
- Por cada label configurado: ``MAX(ts_start) WHERE exit_code = 0``.
- Si el último éxito fue hace > ``stale_hours`` (default 48h) **AND** hubo
  algún éxito en los últimos 30 días → stale, push.
- Si NUNCA hubo éxito en los últimos 30 días → asumimos "no usado" o
  feature inactiva (no spamear).
- Si el último éxito fue dentro del threshold → todo OK, skip.

## Threshold por job

Default 48h aplica a todos. Override granular vía env:
- ``RAG_STALE_ETL_HOURS=48`` — global (default).
- ``RAG_STALE_ETL_HOURS_<LABEL>=N`` — per-label override (ej.
  ``RAG_STALE_ETL_HOURS_AUTO_HARVEST=72`` para jobs que naturalmente corren
  menos frecuente). Label uppercase con guiones reemplazados por underscore.

## Anti-spam

- ``snooze_hours=24``: una vez emitido, la misma alerta no se re-pushea
  por 24h. La lógica de snooze global del framework anticipate maneja
  el dedup_key.
- ``dedup_key`` = ``stale_etl:<label>:<YYYY-MM-DD>``: re-emite máximo 1×
  por día por label. Si el job se reactiva (último éxito vuelve a estar
  fresh), la siguiente run de la signal no emite.
- Cap ``_MAX_EMIT=2``: si fallan 5 jobs de golpe, push solo los 2 más
  prioritarios (por antigüedad del último éxito). Evita spam masivo en
  caso de reboot del Mac que pause todo.

## Lista de jobs críticos

Default mínima — los jobs que más impactan al user si se rompen
silenciosamente. Override completa vía ``RAG_STALE_ETL_LABELS``
(comma-separated). Empty string desactiva la signal.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone

from rag_anticipate.signals.base import register_signal


# Default labels de jobs críticos del supervisor in-process. Cada uno
# self-explanatory:
# - ingest_cross_source: backfill Gmail/Calendar/Drive a vault.
# - ingest_whatsapp: WhatsApp bridge → vault.
# - wa_tasks: extractor que llena rag_promises + Inbox WA notes.
# - anticipate: el daemon que ejecuta TODOS los signals (incluyéndome).
# - drift_watcher: detect regresiones en eval (singles/chains hit@5).
_DEFAULT_LABELS = (
    "ingest_cross_source",
    "ingest_whatsapp",
    "wa_tasks",
    "anticipate",
    "drift_watcher",
)

_DEFAULT_STALE_HOURS = 48
_LOOKBACK_DAYS_FOR_BASELINE = 30  # debe haber ≥1 éxito en los últimos 30d
_MAX_EMIT = 2


def _resolve_labels() -> tuple[str, ...]:
    """Lista de labels a monitorear. Env override > default."""
    env = os.environ.get("RAG_STALE_ETL_LABELS")
    if env is None:
        return _DEFAULT_LABELS
    labels = tuple(s.strip() for s in env.split(",") if s.strip())
    return labels


def _resolve_threshold(label: str) -> int:
    """Horas de staleness para un label. Per-label > global > default."""
    safe_label = label.upper().replace("-", "_")
    per_label = os.environ.get(f"RAG_STALE_ETL_HOURS_{safe_label}")
    if per_label:
        try:
            return max(1, int(per_label))
        except ValueError:
            pass
    glob = os.environ.get("RAG_STALE_ETL_HOURS")
    if glob:
        try:
            return max(1, int(glob))
        except ValueError:
            pass
    return _DEFAULT_STALE_HOURS


def _telemetry_db_path():
    """Path al telemetry.db. Reuso del helper del runtime package si está
    disponible; fallback al default."""
    try:
        from rag.runtime._telemetry import supervisor_jobs_db_path
        return supervisor_jobs_db_path()
    except Exception:
        from pathlib import Path
        override = os.environ.get("OBSIDIAN_RAG_DB_PATH")
        if override:
            return Path(override) / "telemetry.db"
        return Path.home() / ".local/share/obsidian-rag/ragvec/telemetry.db"


def _last_success_ts(conn: sqlite3.Connection, label: str) -> datetime | None:
    """MAX(ts_start) para `label` con exit_code=0. None si nunca corrió OK.

    Silent-fail: tabla no existe (rag_supervisor_jobs scaffold sin runs
    aún) → None.
    """
    try:
        row = conn.execute(
            "SELECT MAX(ts_start) FROM rag_supervisor_jobs "
            "WHERE job_label = ? AND exit_code = 0",
            (label,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row or not row[0]:
        return None
    raw = str(row[0])
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_recent_baseline(
    conn: sqlite3.Connection,
    label: str,
    now: datetime,
) -> bool:
    """True si hay ≥1 éxito en los últimos `_LOOKBACK_DAYS_FOR_BASELINE`.

    Razón: si el job NUNCA corrió en los últimos 30d asumimos "feature
    inactiva" y no pusheamos. Si en cambio corrió hace 25d pero no en
    las últimas 48h → eso sí es staleness real.
    """
    threshold = (now - timedelta(days=_LOOKBACK_DAYS_FOR_BASELINE)).isoformat(
        timespec="seconds",
    )
    try:
        row = conn.execute(
            "SELECT 1 FROM rag_supervisor_jobs "
            "WHERE job_label = ? AND exit_code = 0 AND ts_start >= ? "
            "LIMIT 1",
            (label, threshold),
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    return row is not None


def _format_message(label: str, hours_stale: float, last_success_iso: str) -> str:
    """Push WhatsApp-friendly. Compacto, accionable."""
    days = hours_stale / 24.0
    if days >= 2:
        when = f"{days:.0f}d"
    else:
        when = f"{int(hours_stale)}h"
    pretty = label.replace("_", " ")
    return (
        f"⚠️ ETL *{pretty}* sin correr OK hace {when}\n"
        f"  Último éxito: {last_success_iso[:16].replace('T',' ')}\n"
        f"  Probablemente token expirado o crash silencioso. "
        f"Mirar `rag daemons status` o el log del supervisor."
    )


@register_signal(name="stale_etl", snooze_hours=24)
def stale_etl_signal(now: datetime) -> list:
    """Emite hasta 2 candidates: jobs del supervisor que pararon de correr.

    Pasos:
    1. Resolver labels a monitorear (env override or defaults).
    2. Para cada label: last_success + has_recent_baseline.
    3. Si stale (last_success < now - threshold) AND tiene baseline → emit.
    4. Sort por antigüedad descendente, cap a 2.

    Silent-fail completo:
    - DB no existe / locked → [].
    - Tabla rag_supervisor_jobs no existe (instalación nueva pre-runtime) → [].
    - Cualquier excepción → [].

    Disable completo: ``RAG_STALE_ETL_LABELS=`` (empty).
    """
    try:
        labels = _resolve_labels()
        if not labels:
            return []

        from rag import AnticipatoryCandidate

        db_path = _telemetry_db_path()
        if not db_path.exists():
            return []

        # `now` puede venir aware o naive según caller. Normalizamos a UTC
        # para comparar consistente con `ts_start` que se persiste como UTC ISO.
        if now.tzinfo is None:
            now_utc = now.replace(tzinfo=timezone.utc)
        else:
            now_utc = now.astimezone(timezone.utc)

        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=2000")
            stale_jobs: list[tuple[float, str, str]] = []
            for label in labels:
                threshold_hours = _resolve_threshold(label)
                last = _last_success_ts(conn, label)
                if last is None:
                    continue  # job nunca corrió OK → no es "stale", es no-usado
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                hours_stale = (now_utc - last).total_seconds() / 3600.0
                if hours_stale < threshold_hours:
                    continue  # fresh OK
                if not _has_recent_baseline(conn, label, now_utc):
                    continue  # feature inactiva, no spam
                stale_jobs.append((hours_stale, label, last.isoformat(timespec="seconds")))
        finally:
            conn.close()

        if not stale_jobs:
            return []

        # Sort por antigüedad desc (más viejo arriba) — emitimos los más urgentes.
        stale_jobs.sort(key=lambda t: t[0], reverse=True)

        candidates: list = []
        day_str = now.strftime("%Y-%m-%d")
        for hours_stale, label, last_iso in stale_jobs[:_MAX_EMIT]:
            # Score escala con horas: 1.0 a las 7d (168h), clamp 0.5 mínimo
            # para que aún a 48h aparezca con peso decente.
            score = min(1.0, max(0.5, hours_stale / 168.0))
            message = _format_message(label, hours_stale, last_iso)
            dedup_key = f"stale_etl:{label}:{day_str}"
            reason = (
                f"label={label} hours_stale={hours_stale:.1f} "
                f"last_success={last_iso}"
            )
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-stale_etl",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=reason,
            ))
        return candidates
    except Exception:
        return []
