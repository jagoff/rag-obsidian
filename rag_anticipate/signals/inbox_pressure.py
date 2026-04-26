"""Signal — Presión de inbox (archivos acumulados sin triar).

Detecta cuando `00-Inbox/` tiene demasiadas notas pendientes de procesar.
El corte es "≥24h sin modificarse" — así no molestamos por capturas que
hicimos hoy (el user va a procesar naturalmente durante el día), sólo por
el pile-up que ya pasó un ciclo y sigue ahí.

Diseño:
- File-system only: cuenta `.md` con `iterdir()` sobre `00-Inbox/` top-level.
  NO recursivo — los sub-dirs del inbox NO representan trabajo pendiente
  del user y se excluyen automáticamente por el `iterdir()` top-level.
  Pre-2026-04-25, episodic memory del chat se escribía a
  `00-Inbox/conversations/` y este check explícitamente la ignoraba; ahora
  vive bajo `04-Archive/99-obsidian-system/99-Claude/conversations/` y ya
  ni siquiera está bajo el inbox, pero dejamos la lógica de "no recursivo"
  por si en el futuro cae otra subcarpeta acá (ej. `00-Inbox/voice/`).
- Ventana de edad: `min_age_hours=24` default. Si bajamos el umbral, cada
  vez que el user captura rápido una idea el signal grita; demasiado alto y
  el pile-up tarda en dispararse.
- Threshold de emisión: 15 notas. Score escala linealmente hasta 35+ → 1.0.
  Con <15 no emite (el umbral "me siento inundado" empírico del user).
- dedup_key por fecha → 1 push por día máx, independiente de cuántas veces
  corra el cron. snooze_hours=48 como doble cinturón (si el user clickea
  "snooze" o el push falla, no lo repetimos por 2 días).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Carpeta del inbox relative al vault root. Obsidian PARA convention.
_INBOX_DIR = "00-Inbox"

# Edad mínima (horas) para que una nota cuente como "stale". Capturas de hoy
# no cuentan — el user las va a procesar durante el día.
_INBOX_MIN_AGE_HOURS = 24

# Threshold para emitir. Con <15 notas stale, no disparamos. 15 es el punto
# donde el user reporta "ya se acumuló demasiado, necesito triar".
_INBOX_EMIT_THRESHOLD = 15

# Score ramp: (count - threshold) / _INBOX_SCORE_RAMP + base.
# Con count=15 → 0.4; con count=35 (threshold + ramp) → 1.0.
_INBOX_SCORE_BASE = 0.4
_INBOX_SCORE_RAMP = 20.0


def _count_stale_inbox(vault: Path, min_age_hours: int = _INBOX_MIN_AGE_HOURS) -> int:
    """Cuenta archivos `.md` en `<vault>/00-Inbox/` top-level con mtime
    más viejo que `min_age_hours` horas.

    Deliberadamente NO recursivo: cualquier subcarpeta bajo el inbox NO
    cuenta — son artefactos que no requieren triage manual. `iterdir()`
    solo ve los entries directos, así que una subfolder con 500 archivos
    queda invisible para este conteo. (Histórico: hasta 2026-04-25, episodic
    memory del chat se escribía a `00-Inbox/conversations/`; tras esa
    fecha vive bajo `04-Archive/99-obsidian-system/99-Claude/conversations/`,
    fuera del inbox enteramente.)

    Silent-fail: si el dir no existe o no se puede leer, devuelve 0 (no
    excepción). El caller ya está envuelto en outer try/except por el
    framework, pero este doble cinturón evita spam en el warning log.
    """
    inbox = vault / _INBOX_DIR
    try:
        if not inbox.exists() or not inbox.is_dir():
            return 0
    except Exception:
        return 0

    cutoff_ts = datetime.now().timestamp() - (min_age_hours * 3600.0)
    count = 0
    try:
        entries = inbox.iterdir()
    except Exception:
        return 0

    for entry in entries:
        try:
            # Solo archivos (no subdirs) con extensión `.md`. El check
            # de `is_file()` también descarta symlinks rotos y sockets
            # por las dudas.
            if not entry.is_file():
                continue
            if entry.suffix.lower() != ".md":
                continue
            mtime = entry.stat().st_mtime
            if mtime <= cutoff_ts:
                count += 1
        except Exception:
            # Archivo individual malformado / permission denied: saltar,
            # seguir contando los otros.
            continue
    return count


@register_signal(name="inbox_pressure", snooze_hours=48)
def inbox_pressure_signal(now: datetime) -> list:
    """Emite MÁXIMO 1 candidate cuando el inbox tiene ≥15 notas stale (>24h).

    Silent-fail total: cualquier error (vault no accesible, permission,
    is_excluded throws, lo que sea) → `[]`. El orchestrator tiene su propio
    outer try/except pero este doble cinturón es el contrato del framework.
    """
    try:
        from rag import (
            AnticipatoryCandidate,
            _resolve_vault_path,
            is_excluded,
        )

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        # Defensive: si el inbox en sí está excluido por config global
        # (improbable pero posible si el user lo overridea), no emitimos.
        try:
            if is_excluded(_INBOX_DIR + "/"):
                return []
        except Exception:
            # is_excluded no debería tirar, pero si lo hace tratamos como
            # "no excluido" y seguimos — safer para evitar silenciar señal.
            pass

        count = _count_stale_inbox(vault, min_age_hours=_INBOX_MIN_AGE_HOURS)

        if count < _INBOX_EMIT_THRESHOLD:
            return []

        # Score: 0.4 en el threshold (15), escala hasta 1.0 en 35+.
        score = min(
            1.0,
            (count - _INBOX_EMIT_THRESHOLD) / _INBOX_SCORE_RAMP + _INBOX_SCORE_BASE,
        )

        message = (
            f"📥 Inbox acumulado: {count} notas sin triar (>24h). "
            f"¿`rag inbox --apply` o triage manual?"
        )

        # dedup_key por fecha — 1 emisión por día máx. El snooze_hours=48
        # agrega un segundo guardrail: si el user no confirma o el push
        # falla, no retry por 2 días (evita que un bug de delivery cause
        # spam en rejoin).
        today_date = now.date().isoformat()
        dedup_key = f"inbox_pressure:{today_date}"

        reason = f"stale_count={count} threshold={_INBOX_EMIT_THRESHOLD}"

        return [AnticipatoryCandidate(
            kind="anticipate-inbox_pressure",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=48,
            reason=reason,
        )]
    except Exception:
        return []
