"""Signal — Stale external-ingest source (silent-fail invisible).

Compañero de ``stale_etl`` (monitorea ``rag_supervisor_jobs``). Este signal
mira EL ARCHIVO en el vault — el último ``.md`` por subfolder bajo
``99-obsidian/99-AI/external-ingest/<source>/`` — para detectar fuentes
que silent-failearon a nivel ETL individual (Gmail OAuth expirado,
Calendar sin creds, Drive token revocado) AUNQUE el job paraguas
``ingest_cross_source`` haya corrido exit_code=0.

## Por qué dos signals

- ``stale_etl`` detecta jobs que dejaron de correr (proceso muerto,
  scheduler no dispara, crash).
- ``stale_external_ingest`` detecta que el job corrió pero un ETL
  individual no escribió (silent_fail interno, OAuth expirado, source
  no disponible). El job termina exit_code=0, ``rag_supervisor_jobs``
  no muestra nada raro, pero la carpeta del vault sigue vacía.

## Diseño

- Walks subfolder por subfolder bajo
  ``<vault>/99-obsidian/99-AI/external-ingest/``.
- Para cada subfolder (Gmail, Calendar, Drive, etc.): mtime del archivo
  ``.md`` más reciente.
- Si la fuente tiene baseline (≥1 archivo en los últimos 30d) Y el
  archivo más reciente es > ``stale_days`` (default 7) → push.
- Si NUNCA hubo archivos en los últimos 30d → asumimos source inactivo
  / no configurado, no spam.
- Cap ``_MAX_EMIT=2``: prioriza por días desde el último archivo.

## Threshold

- ``RAG_STALE_EXT_INGEST_DAYS=7`` — global.
- ``RAG_STALE_EXT_INGEST_DAYS_<SOURCE>=N`` — per-source (ej.
  ``RAG_STALE_EXT_INGEST_DAYS_GMAIL=3``).
- ``RAG_STALE_EXT_INGEST_SOURCES`` — comma-separated allowlist (default
  set abajo).

## Anti-spam

- ``snooze_hours=24``.
- ``dedup_key`` = ``stale_external_ingest:<source>:<YYYY-MM-DD>``.

Silent-fail completo: vault no existe / permisos / DB lock → ``[]``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Subfolders bajo external-ingest/ con histórico operacional. Cubre solo
# las fuentes que se actualizan diaria/frecuente — sources que escriben
# 1x al mes (Finanzas/Tarjetas) o on-demand (Drive) NO van acá porque
# son too noisy.
_DEFAULT_SOURCES = (
    "Gmail",
    "Calendar",
    "Chrome",
    "GitHub",
    "Screentime",
    "WhatsApp",
    "Spotify",
)

_DEFAULT_STALE_DAYS = 7
_LOOKBACK_DAYS_FOR_BASELINE = 30
_MAX_EMIT = 2


def _resolve_sources() -> tuple[str, ...]:
    env = os.environ.get("RAG_STALE_EXT_INGEST_SOURCES")
    if env is None:
        return _DEFAULT_SOURCES
    return tuple(s.strip() for s in env.split(",") if s.strip())


def _resolve_threshold_days(source: str) -> int:
    safe = source.upper().replace("-", "_")
    per_source = os.environ.get(f"RAG_STALE_EXT_INGEST_DAYS_{safe}")
    if per_source:
        try:
            return max(1, int(per_source))
        except ValueError:
            pass
    glob = os.environ.get("RAG_STALE_EXT_INGEST_DAYS")
    if glob:
        try:
            return max(1, int(glob))
        except ValueError:
            pass
    return _DEFAULT_STALE_DAYS


def _vault_path() -> Path | None:
    """Resolver vault path. Reusa el constant de `rag` si está disponible."""
    try:
        from rag import VAULT_PATH
        if VAULT_PATH and Path(VAULT_PATH).is_dir():
            return Path(VAULT_PATH)
    except Exception:
        pass
    override = os.environ.get("OBSIDIAN_RAG_VAULT")
    if override:
        p = Path(override)
        if p.is_dir():
            return p
    default = (
        Path.home()
        / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
    )
    return default if default.is_dir() else None


def _latest_mtime_in_dir(d: Path) -> float | None:
    """mtime epoch del archivo .md más reciente bajo `d` recursivo.
    None si vacío / no existe."""
    if not d.is_dir():
        return None
    latest = 0.0
    try:
        for p in d.rglob("*.md"):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > latest:
                latest = m
    except OSError:
        return None
    return latest if latest > 0 else None


def _has_recent_baseline(source_dir: Path, now: datetime) -> bool:
    """True si hay ≥1 archivo .md mtime en los últimos
    `_LOOKBACK_DAYS_FOR_BASELINE`. Razón: si la fuente nunca escribió en
    los últimos 30d, asumimos source inactivo (OAuth no configurado, etc.)
    y NO spameamos.
    """
    cutoff_epoch = (now - timedelta(days=_LOOKBACK_DAYS_FOR_BASELINE)).timestamp()
    try:
        for p in source_dir.rglob("*.md"):
            try:
                if p.stat().st_mtime >= cutoff_epoch:
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


_HINT_BY_SOURCE = {
    "Gmail": "OAuth probablemente expiró — revisar `~/.gmail-mcp/credentials.json` + correr el flow de re-auth.",
    "Calendar": "Calendar OAuth o icalBuddy roto — revisar `~/.calendar-mcp/credentials.json`.",
    "Drive": "Drive OAuth expiró — revisar `~/.gdrive-mcp/credentials.json`.",
    "GoogleDrive": "Drive OAuth expiró — revisar `~/.gdrive-mcp/credentials.json`.",
    "Chrome": "Chrome SQLite probablemente lockeado o profile no resuelto. `pkill -9 'Google Chrome'` + retry, o mirar logs del supervisor.",
    "GitHub": "`gh` CLI no autenticado o repos sin actividad. Correr `gh auth status`.",
    "WhatsApp": "Bridge probablemente desconectado. `launchctl print gui/$(id -u)/com.fer.obsidian-rag-wa-bridge-watchdog`.",
    "Spotify": "Spotify desktop cerrado o token revocado. Revisar `~/.config/obsidian-rag/spotify_client.json`.",
    "Screentime": "macOS knowledgeC.db locked o schema changed (después de macOS upgrade).",
    "Claude": "Sesiones de Claude Code probablemente cero en 7d — feature inactiva si dejaste de usar el CLI.",
    "Claude-Web": "ZIP de claude.ai web no actualizado — dropear nuevo export en `~/.claude-ai-export/`.",
}


def _format_message(source: str, days_stale: float, last_iso: str) -> str:
    pretty = source.replace("-", " ").replace("_", " ")
    when = f"{days_stale:.0f}d" if days_stale >= 1 else f"{int(days_stale * 24)}h"
    hint = _HINT_BY_SOURCE.get(
        source,
        "Probablemente OAuth expiró o source no disponible. "
        "Revisar logs del supervisor + creds del source.",
    )
    return (
        f"⚠️ Ingest *{pretty}* sin archivos nuevos hace {when}\n"
        f"  Último archivo: {last_iso[:16].replace('T', ' ')}\n"
        f"  {hint}"
    )


@register_signal(name="stale_external_ingest", snooze_hours=24)
def stale_external_ingest_signal(now: datetime) -> list:
    """Detecta fuentes external-ingest que silent-failearon a nivel ETL.

    Steps:
    1. Resolver vault path + sources.
    2. Por cada source: mtime del archivo más reciente.
    3. Si stale > threshold AND tiene baseline reciente → emit.
    4. Cap a 2, sort por más viejo.
    """
    try:
        from rag import AnticipatoryCandidate

        vault = _vault_path()
        if vault is None:
            return []
        base = vault / "99-obsidian/99-AI/external-ingest"
        if not base.is_dir():
            return []

        sources = _resolve_sources()
        if not sources:
            return []

        stale: list[tuple[float, str, str]] = []
        for source in sources:
            sdir = base / source
            if not sdir.is_dir():
                continue  # source nunca configurado
            latest_mtime = _latest_mtime_in_dir(sdir)
            if latest_mtime is None:
                continue  # carpeta vacía → asumimos nunca corrió
            threshold_days = _resolve_threshold_days(source)
            age_seconds = (now.timestamp() - latest_mtime)
            days_stale = age_seconds / 86400.0
            if days_stale < threshold_days:
                continue
            if not _has_recent_baseline(sdir, now):
                continue  # source dejó de usarse hace tiempo, no spam
            last_iso = datetime.fromtimestamp(latest_mtime).isoformat(
                timespec="seconds",
            )
            stale.append((days_stale, source, last_iso))

        if not stale:
            return []
        stale.sort(key=lambda t: t[0], reverse=True)

        candidates: list = []
        day_str = now.strftime("%Y-%m-%d")
        for days_stale, source, last_iso in stale[:_MAX_EMIT]:
            # Score [0.5, 1.0] scalado por días (1.0 a las 4 semanas).
            score = min(1.0, max(0.5, days_stale / 28.0))
            message = _format_message(source, days_stale, last_iso)
            dedup_key = f"stale_external_ingest:{source}:{day_str}"
            reason = (
                f"source={source} days_stale={days_stale:.1f} "
                f"last={last_iso}"
            )
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-stale_external_ingest",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=reason,
            ))
        return candidates
    except Exception:
        return []
