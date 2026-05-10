"""Signal — Weekly review due (domingo/lunes sin digest).

Detecta cuando es domingo o lunes y el user NO tiró `rag digest` para
cerrar la semana que terminó. Push: "¿cerramos la semana con un digest?".

Source: filesystem `99-obsidian/99-AI/reviews/YYYY-WNN.md` (donde
`rag digest` escribe los weekly narratives — ver `rag/__init__.py
::digest`). Filename ISO week format (`2026-W19.md`).

Diseño:

- Solo dispara domingo (weekday=6) o lunes (weekday=0). Otros días =
  silencio (no es momento para weekly review).
- Mira el ISO week que termina HOY (domingo) o que terminó ayer (lunes).
- Si existe `<vault>/99-obsidian/99-AI/reviews/<isoweek>.md` → silent
  (ya hay digest, todo bien).
- Si NO existe → push.
- Score fijo 0.6 — claro signal pero no urgente como overdue deadline.
- Snooze 168h (1 semana) — sin sentido pushear todos los días para
  el mismo digest pendiente.
- Dedup_key `weekly_review:<isoweek>` — re-emit cuando empieza nueva
  semana sin digest.

Silent-fail total: vault inaccesible / clock skew / FS permission → [].
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Carpeta canónica donde `rag digest` escribe weekly narratives.
_REVIEWS_FOLDER = "99-obsidian/99-AI/reviews"

# Score fijo — el user "ya sabe" que tendría que tirar digest si es
# domingo/lunes; el push es un nudge, no una urgencia.
_SCORE = 0.6


def _iso_week_label(d: datetime) -> str:
    """Format ISO week label `YYYY-WNN` (matchea `_iso_week_label` en rag.py)."""
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _target_week_for(now: datetime) -> tuple[str, datetime] | None:
    """Devuelve `(week_label, week_end_dt)` para la semana que el user
    debería digest-ear AHORA. None si hoy no es domingo/lunes.

    - Domingo → semana que termina HOY (digest cierra la semana actual).
    - Lunes → semana que terminó AYER (digest de la semana anterior).
    - Otros días → None (silencio).
    """
    weekday = now.weekday()  # Monday=0, Sunday=6
    if weekday == 6:
        # Domingo: el ISO week que termina hoy
        return _iso_week_label(now), now
    if weekday == 0:
        # Lunes: el ISO week de ayer (el digest ideal era ayer/hoy)
        ayer = now - timedelta(days=1)
        return _iso_week_label(ayer), ayer
    return None


@register_signal(name="weekly_review_due", snooze_hours=168)
def weekly_review_due_signal(now: datetime) -> list:
    """Emite máximo 1 candidate los domingos/lunes sin digest semanal.

    Pasos:
    1. Resuelve vault. Si no accesible → [].
    2. Si hoy NO es domingo/lunes → [].
    3. Calcula ISO week target.
    4. Si existe `<reviews>/<week>.md` → [] (ya está hecho).
    5. Emit 1 candidate.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path  # noqa: PLC0415

        target = _target_week_for(now)
        if target is None:
            return []
        week_label, _week_end = target

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        reviews_dir = vault / _REVIEWS_FOLDER
        digest_path = reviews_dir / f"{week_label}.md"

        # Si existe, el digest ya se hizo — silent.
        if digest_path.is_file():
            return []

        message = (
            f"📊 Domingo/Lunes sin weekly review.\n"
            f"  La semana {week_label} no tiene digest.\n"
            f"  ¿Tirás `rag digest --week {week_label}` para cerrarla?"
        )
        dedup_key = f"weekly_review:{week_label}"
        reason = f"week={week_label} weekday={now.weekday()}"

        return [AnticipatoryCandidate(
            kind="anticipate-weekly_review_due",
            score=_SCORE,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=168,
            reason=reason,
        )]
    except Exception:
        return []
