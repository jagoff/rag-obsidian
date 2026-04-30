"""Signal — Mood drift sostenido.

Dispara un push proactivo MUY conservador cuando el user lleva ≥ 3 días
consecutivos con score diario de mood ≤ -0.4 (threshold de
`rag.mood.recent_drift()`). El push NO verbaliza el score ni el
diagnóstico — solo ofrece simplificar el plan del día.

## Diseño

- **Threshold conservador intencional**: `min_consecutive=3` + score
  ≤ -0.4 evita falsos positivos de un solo día random. La racha tiene
  que terminar en hoy o ayer (no avisamos bajones históricos viejos).
  La función `rag.mood.recent_drift()` ya implementa esa lógica — esta
  signal solo la consume.

- **Cooldown 168h (7 días)** — `snooze_hours=168`. El dedup_key incluye
  el `start_date` de la racha actual, así que:
    - Misma racha (sin quebrarse) → mismo dedup_key → snooze ahoga el
      re-push hasta 7 días después.
    - Si la racha se quiebra y vuelve más adelante → dedup_key nuevo
      → push permitido (es una racha distinta, valid signal).

- **Quiet hours**: viene gratis del orchestrator
  (`is_in_quiet_hours(now)` se chequea antes de pushear cualquier
  candidate). Si un drift se detecta a las 23:00, el push se difiere.

- **Sin LLM call**: el mensaje es template fijo. No queremos que un
  modelo improvise tono y termine sonando paternalista. La regla del
  feature: el sistema NUNCA verbaliza mood al user. Solo modula.

- **Score**: lineal con `n_consecutive`. 3 días → 0.55, 5 días → 0.75,
  7+ días → 1.0. Bajo el threshold default 0.35 nada dispara, así que
  3 es el mínimo efectivo.

## Mensaje

Empático pero NO terapéutico. Enfoca en:
  - Reducir scope hoy (1-2 cosas críticas vs todo).
  - Posponer cosas no urgentes.
  - Sugiere `rag mood show` si querés ver vos mismo qué señales
    movieron el score (transparencia + no es paternalismo si vos lo
    pedís explícito).

NO contiene:
  - "Noté que estás triste" / "venís de bajón" / "te ves cansado"
  - "Tomate un día" / "respirá" / "hablalo con alguien"
  - Score literal (-0.5, etc.) ni source de las señales
  - Diagnóstico de cualquier tipo

## Behind flag

`RAG_MOOD_ENABLED` debe estar prendido. Si está off, la signal devuelve
`[]` sin tirar (silent-fail). Si el daemon mood-poll nunca corrió y
no hay rows en `rag_mood_score_daily`, también `[]`.

Aprendido el 2026-04-30 — la fricción de un push paternalista es peor
que la ayuda real. Por eso threshold conservador + cooldown largo +
mensaje template sin LLM.
"""

from __future__ import annotations

from datetime import datetime

from rag_anticipate.signals.base import register_signal


_MIN_CONSECUTIVE = 3
_DRIFT_THRESHOLD = -0.4
_DRIFT_DAYS_WINDOW = 7
_SNOOZE_HOURS = 168  # 7 días: misma racha activa NO re-pushea


def _format_drift_message(n_consecutive: int) -> str:
    """Mensaje template — empatico, factual, sin diagnóstico ni
    therapy-speak. Una sola variante para mantener consistencia."""
    return (
        "Si querés, te ayudo a recortar el plan de hoy a 1-2 cosas que "
        "muevan la aguja real, y el resto lo posponemos sin culpa. "
        "Decime si dale.\n\n"
        "(`rag mood show` te muestra de dónde viene la señal si querés "
        "verla vos.)"
    )


@register_signal(name="mood-drift", snooze_hours=_SNOOZE_HOURS)
def mood_drift_signal(now: datetime) -> list:
    """Detecta racha sostenida de mood bajo y emite a lo sumo 1
    candidate. Silent-fail si el feature está off, si el módulo
    `rag.mood` no está disponible, o si la DB tira."""
    try:
        from rag import AnticipatoryCandidate
    except Exception:
        return []
    try:
        from rag import mood as _mood
    except Exception:
        return []

    # Feature gate explícito.
    try:
        if not _mood._is_mood_enabled():
            return []
    except Exception:
        return []

    try:
        drift = _mood.recent_drift(
            days=_DRIFT_DAYS_WINDOW,
            threshold=_DRIFT_THRESHOLD,
            min_consecutive=_MIN_CONSECUTIVE,
        )
    except Exception:
        return []

    if not drift.get("drifting"):
        return []

    n_consec = int(drift.get("n_consecutive", 0))
    if n_consec < _MIN_CONSECUTIVE:
        return []

    dates = drift.get("dates") or []
    if not dates:
        return []

    # Score: lineal por longitud de racha. 3 días = 0.55, 5 = 0.75, 7+ = 1.0.
    # Todos están por encima del threshold default 0.35.
    score = min(1.0, 0.55 + 0.10 * (n_consec - _MIN_CONSECUTIVE))

    # Dedup_key estable por start_date de la racha. Misma racha, mismo
    # key → snooze de 7d ahoga el re-push. Racha nueva (después de un
    # quiebre) tendría start_date distinto → dedup_key nuevo → push OK.
    start_date = dates[0]
    dedup_key = f"mood-drift:{start_date}"

    message = _format_drift_message(n_consec)

    # `reason` para `--explain` y logs. NO se muestra al user.
    reason = (
        f"n_consecutive={n_consec} threshold={_DRIFT_THRESHOLD} "
        f"avg_score={drift.get('avg_score', 0.0):.2f} "
        f"start={start_date}"
    )

    return [AnticipatoryCandidate(
        kind="anticipate-mood-drift",
        score=score,
        message=message,
        dedup_key=dedup_key,
        snooze_hours=_SNOOZE_HOURS,
        reason=reason,
    )]
