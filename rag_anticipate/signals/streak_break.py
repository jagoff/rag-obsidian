"""Signal — Racha rota del morning brief.

Detecta cuando el usuario deja de tirar `rag morning` por ≥2 días
consecutivos. Un gap de 1 día es ruido normal (fin de semana, sleep-in,
laptop cerrada). Un gap ≥2 en una ventana de 7 días sugiere que la racha
se rompió y vale la pena un nudge sutil para retomar el hábito.

Fuente: filesystem como primary (más robusto que la tabla SQL
`rag_brief_written` — sobrevive a resets del state DB, drops de tablas y
migrations). Se lista `<vault>/04-Archive/99-obsidian-system/99-Claude/reviews/` buscando archivos cuyo
filename matchea `YYYY-MM-DD.md` EXACTO. El filename es la verdad
autoritativa (no el mtime), porque:

- Si el user re-abre un brief viejo para corregir una línea, el mtime
  salta a "hoy" pero la fecha del brief no cambió → seguir usando
  filename.
- Los briefs de `rag today` se escriben como `YYYY-MM-DD-evening.md` y
  NO matchean el regex (por el "-evening" extra antes de `.md`), así
  que no se cuentan como morning brief.

Ventana fija de 7 días. Si NO hay brief en los últimos 7 días, asumimos
que el usuario pausó voluntariamente (vacaciones, sabbatical, cambio de
rutina) y NO emitimos — mejor quedarse callado que ser el SaaS molesto
que spam-eá todos los días "vuelve a usar nuestro producto".

Score: `min(1.0, gap_days / 5.0)`. Gap=2 → 0.4 (apenas sobre el threshold
default 0.35 del orchestrator). Gap=3 → 0.6. Gap=5 → 1.0 (saturado).

dedup_key: `streak_break:YYYY-MM-DD` — estable por día. Combinado con
`snooze_hours=24`, garantiza máximo un push por día por este signal,
incluso si el cron del anticipatory agent corre 10 veces en una mañana.

Silent-fail total: cualquier excepción interna (vault inaccesible,
permission denied, clock skew que produzca fechas imposibles) termina en
`return []`. El orchestrator tiene su propio outer try/except pero la
signal cumple el contrato igual.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Carpeta canónica donde `rag morning` escribe los briefs.
_REVIEWS_FOLDER = "04-Archive/99-obsidian-system/99-Claude/reviews"

# Matchea "YYYY-MM-DD.md" EXACTO.
# - NO matchea "YYYY-MM-DD-evening.md" (evening brief de `rag today`).
# - NO matchea "YYYY-WNN.md" (digest semanal).
# - NO matchea strings arbitrarios con números (ej. "notes-2025.md").
_MORNING_BRIEF_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.md$")

# Ventana de observación. Más allá, asumimos pausa voluntaria → no emit.
_LOOKBACK_DAYS = 7

# Gap mínimo para emitir. Gap 0-1 → comportamiento normal (weekend/sleep).
_MIN_GAP_DAYS = 2

# Divisor del score: gap=5 → 1.0 (saturado). Gap=2 → 0.4.
_SCORE_DIVISOR = 5.0


def _find_last_morning_brief(
    vault: Path,
    now: datetime,
    within_days: int = _LOOKBACK_DAYS,
) -> date | None:
    """Fecha del morning brief más reciente dentro de la ventana, o None.

    Args:
        vault: root del vault activo.
        now: reloj actual inyectable (tests).
        within_days: cuántos días atrás mirar. Briefs con filename-date
            más viejos que `today - within_days` se ignoran.

    Returns:
        `date` del brief más reciente en `[today - within_days, today]`,
        o `None` si el folder no existe / no hay briefs en la ventana.

    La fecha se parsea del FILENAME (no del mtime) para ser robusto a
    edits manuales posteriores. Si el filename tiene componentes
    inválidos (ej. "2025-13-45.md"), se descarta silenciosamente.
    """
    reviews = vault / _REVIEWS_FOLDER
    if not reviews.is_dir():
        return None

    today = now.date()
    cutoff = today - timedelta(days=within_days)

    best: date | None = None
    try:
        entries = list(reviews.iterdir())
    except Exception:
        return None

    for p in entries:
        try:
            if not p.is_file():
                continue
            m = _MORNING_BRIEF_RE.match(p.name)
            if not m:
                continue
            try:
                file_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                # Fecha inválida (mes 13, día 45, año 0000) → skip.
                continue
            # Fuera de la ventana (demasiado viejo o "futuro" por clock skew).
            if file_date < cutoff or file_date > today:
                continue
            if best is None or file_date > best:
                best = file_date
        except Exception:
            # Entry corrupta individual no tumba el scan.
            continue
    return best


def _count_gap_days(last_brief_date: date, now: datetime) -> int:
    """Días entre la fecha del último brief y `now.date()`.

    - gap=0 → brief escrito HOY.
    - gap=1 → brief escrito AYER, ninguno hoy.
    - gap=N → últimos N días sin brief (último brief hace N días).

    Clamp a 0 si la fecha es "futura" (clock skew), para evitar ints
    negativos leakeando a score.
    """
    today = now.date()
    delta = (today - last_brief_date).days
    return max(0, delta)


@register_signal(name="streak_break", snooze_hours=24)
def streak_break_signal(now: datetime) -> list:
    """Ver docstring del módulo. Retorna MÁXIMO 1 candidate.

    Pasos:
    1. Resuelve vault activo. Si no accesible → [].
    2. Busca último brief en los últimos 7 días. Si no hay → [].
    3. Calcula gap vs `now.date()`. Si gap < 2 → [].
    4. Emite 1 candidate con score proporcional al gap.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        last_brief = _find_last_morning_brief(
            vault, now, within_days=_LOOKBACK_DAYS,
        )
        if last_brief is None:
            # No hay brief en la ventana → pausa voluntaria, silencio.
            return []

        gap = _count_gap_days(last_brief, now)
        if gap < _MIN_GAP_DAYS:
            # Gap 0 o 1 → comportamiento normal (weekend, sleep-in).
            return []

        score = min(1.0, gap / _SCORE_DIVISOR)

        today_iso = now.date().isoformat()
        dedup_key = f"streak_break:{today_iso}"

        message = (
            f"🔥 Racha rota: hace {gap} días que no tirás morning brief. "
            f"¿Retomás hoy con `rag morning`?"
        )

        reason = (
            f"last_brief={last_brief.isoformat()} "
            f"gap_days={gap} "
            f"within_days={_LOOKBACK_DAYS}"
        )

        return [AnticipatoryCandidate(
            kind="anticipate-streak_break",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=24,
            reason=reason,
        )]
    except Exception:
        return []
