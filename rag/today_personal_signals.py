"""Personal-signal correlators (mood, sleep, cross-patterns) — extracted
from rag/today_correlator.py 2026-05-09.

Tres buckets que el today brief usa para modular tono / mencionar
anomalías / narrar findings sin afirmar causalidad:

1. **Mood** — score diario de hoy + tendencia 7d + drift. Lee de
   `rag_mood_score_daily` (lo escribe el daemon `mood-poll`). Devuelve
   `None` si feature off o sin row para hoy.

2. **Sleep** — última noche + delta vs histórico + anomaly opcional.
   Lee via `rag.integrations.pillow_sleep.last_night/weekly_stats`.
   `anomaly` se llena solo cuando algo cruza thresholds que valen la
   pena narrar (deep% drop, awakenings spike, quality drop, duration
   short).

3. **Cross-patterns** — top findings strong/moderate de Pearson
   cross-source + predicción mood mañana. Lee de
   `rag.cross_source_patterns`. `None` si no hay findings.

Todos pure-readers — no escriben nada, no comparten state. Lazy imports
adentro de cada función (módulos externos pueden no estar instalados).

Re-exportados desde `rag.today_correlator` para preservar
`from rag.today_correlator import _correlate_mood` etc.
"""
from __future__ import annotations

__all__ = [
    "_correlate_mood",
    "_correlate_sleep",
    "_correlate_cross_patterns",
]


def _correlate_mood(today_ev: dict, extras: dict) -> dict | None:
    """Lee el score diario de hoy + los últimos 7 días desde
    `rag_mood_score_daily` (lo escribe el daemon `mood-poll` cada 30
    min — `rag/mood.py:run_poll_cycle()`). Devuelve un bucket con
    shape:

        {
            "score": float,              # score de hoy (-1..+1)
            "n_signals": int,            # cuántas señales lo soportan
            "sources_used": [str],       # ["spotify", "journal", ...]
            "trend": str,                # "stable" | "improving" | "declining"
            "week_avg": float,           # media móvil 7d
            "drift": {                   # del recent_drift()
                "drifting": bool,
                "n_consecutive": int,
                "avg_score": float,
            },
            "top_evidence": [...],       # top 3 señales del día
        }

    Devuelve `None` si:
      - el feature está off (`RAG_MOOD_ENABLED` no seteado),
      - no hay row para hoy todavía (daemon nunca corrió),
      - hubo error leyendo (silent-fail con None).

    El consumer downstream (today brief prompt) decide si/cómo
    modular tono según este bucket. ESTE módulo NO verbaliza el
    score — solo lo expone como contexto cross-source. La regla
    "no decir 'noté que estás triste'" se aplica en el prompt
    template, no acá.
    """
    try:
        from rag import mood as _mood  # noqa: PLC0415
    except Exception:
        return None
    if not _mood._is_mood_enabled():
        return None

    try:
        today = _mood._today_local()
        score_row = _mood.get_score_for_date(today)
        if score_row is None or score_row.get("n_signals", 0) == 0:
            return None
        recent = _mood.get_recent_scores(days=7)
        drift = _mood.recent_drift(days=7)
    except Exception as exc:
        try:
            from rag import _silent_log as _slog  # noqa: PLC0415
            _slog("today_personal_signals.correlate_mood", exc)
        except Exception:
            pass
        return None

    # week_avg de los últimos 7 días con n_signals > 0 (excluye hoy si querés
    # comparar con baseline, pero acá lo incluimos: el "trend" mide cuánto
    # se desvía hoy del promedio reciente).
    valid = [r for r in recent if r["n_signals"] > 0]
    if not valid:
        week_avg = score_row["score"]
    else:
        week_avg = sum(r["score"] for r in valid) / len(valid)

    delta = score_row["score"] - week_avg
    if delta > 0.2:
        trend = "improving"
    elif delta < -0.2:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "score": round(score_row["score"], 3),
        "n_signals": score_row["n_signals"],
        "sources_used": score_row.get("sources_used") or [],
        "trend": trend,
        "week_avg": round(week_avg, 3),
        "drift": {
            "drifting": bool(drift.get("drifting", False)),
            "n_consecutive": int(drift.get("n_consecutive", 0)),
            "avg_score": round(float(drift.get("avg_score", 0.0)), 3),
        },
        "top_evidence": (score_row.get("top_evidence") or [])[:3],
    }


def _correlate_sleep(today_ev: dict, extras: dict) -> dict | None:
    """Lee la última noche + comparación 7d vs histórico desde
    `rag_sleep_sessions` (lo escribe el daemon `obsidian-rag-ingest-pillow`
    1×/día). Devuelve un bucket con shape:

        {
            "date": str,                  # YYYY-MM-DD del end de la sesión
            "duration_h": float,          # horas dormidas anoche
            "quality": float | None,      # 0..1
            "bedtime_local": str,         # "HH:MM"
            "deep_pct": float | None,
            "awakenings": int,
            "delta": {                    # vs histórico
                "duration_h": float | None,
                "quality": float | None,
                "deep_pct": float | None,
                "awakenings": float | None,
            },
            "anomaly": str | None,        # 1 línea cuando hay algo raro
        }

    Devuelve `None` si:
      - el módulo no está disponible (Pillow no instalado),
      - no hay sesiones todavía (daemon nunca corrió),
      - hubo error leyendo (silent-fail con None).

    El consumer downstream (today brief prompt) decide si menciona el
    sueño y cómo. ESTE módulo NO genera narrativa; solo expone los
    datos como contexto cross-source. La regla "solo mencionarlo
    cuando hay anomaly" se aplica en el prompt template, no acá.

    Surface criteria — `anomaly` se llena solo cuando el delta vs
    histórico cruza thresholds que valen la pena narrar (mismo set
    que el `insight` del panel home, sincronizado a propósito):
      • deep% drop ≥ 5pp y week ≤ 18%  → "deep cayó X pts esta semana"
      • awakenings ≥ 3/noche y delta ≥ 1 → "awakenings subieron a X"
      • quality drop ≥ 0.05 → "quality bajó X esta semana"
      • duration drop ≥ 0.5h y week ≤ 6h → "dormiste X h menos"

    Si nada cruza thresholds, `anomaly` queda en None y el prompt
    típicamente ignora el bucket entero (sleep estable = no aporta
    al brief).
    """
    try:
        from rag.integrations.pillow_sleep import last_night, weekly_stats  # noqa: PLC0415
    except Exception:
        return None
    try:
        ln = last_night()
    except Exception as exc:
        try:
            from rag import _silent_log as _slog  # noqa: PLC0415
            _slog("today_personal_signals.correlate_sleep.last_night", exc)
        except Exception:
            pass
        ln = None
    if not ln:
        return None
    try:
        ws = weekly_stats()
    except Exception as exc:
        try:
            from rag import _silent_log as _slog  # noqa: PLC0415
            _slog("today_personal_signals.correlate_sleep.weekly_stats", exc)
        except Exception:
            pass
        ws = {}

    delta = ws.get("delta") or {}
    week = ws.get("week") or {}

    # Anomaly: misma lógica que `_fetch_sleep().insight` en web/server.py.
    # Mantener acá el mismo orden de prioridad para que el panel home
    # y el brief no contradigan: el más severo gana.
    anomaly: str | None = None
    if (delta.get("deep_pct") or 0) <= -5 and (week.get("deep_pct") or 100) <= 18:
        anomaly = (
            f"deep% cayó {abs(delta['deep_pct']):.0f} pts "
            f"({week['deep_pct']:.0f}% esta semana vs hist)"
        )
    elif (delta.get("awakenings") or 0) >= 1 and (week.get("awakenings") or 0) >= 3:
        anomaly = (
            f"awakenings subieron a {week['awakenings']:.1f}/noche "
            f"(+{delta['awakenings']:.1f} vs hist)"
        )
    elif (delta.get("quality") or 0) <= -0.05:
        anomaly = (
            f"quality bajó {abs(delta['quality']):.2f} esta semana "
            f"({week.get('quality', 0):.2f} vs hist)"
        )
    elif (delta.get("duration_h") or 0) <= -0.5 and (week.get("duration_h") or 0) <= 6.0:
        anomaly = (
            f"dormiste {abs(delta['duration_h']):.1f}h menos que el promedio"
        )

    return {
        "date": ln.get("date"),
        "duration_h": round(ln.get("sleep_total_h") or 0, 2),
        "quality": ln.get("quality"),
        "bedtime_local": ln.get("bedtime_local"),
        "deep_pct": (
            round(ln["deep_pct"], 1) if ln.get("deep_pct") is not None else None
        ),
        "awakenings": ln.get("awakenings", 0),
        "delta": {
            "duration_h": round(delta["duration_h"], 2) if "duration_h" in delta else None,
            "quality": round(delta["quality"], 3) if "quality" in delta else None,
            "deep_pct": round(delta["deep_pct"], 1) if "deep_pct" in delta else None,
            "awakenings": round(delta["awakenings"], 1) if "awakenings" in delta else None,
        },
        "anomaly": anomaly,
    }


def _correlate_cross_patterns(today_ev: dict, extras: dict) -> dict | None:
    """Lee findings de Pearson cross-source de
    `rag.cross_source_patterns.patterns_summary` + predicción del mood
    de mañana. Devuelve un bucket con las top correlaciones STRONG y
    la predicción para que el today brief pueda mencionarlas
    factualmente (sin afirmar causalidad).

    Shape:

        {
            "top_findings": [{description, r, severity, lag, ...}, ...],
            "prediction": {prediction, confidence, top_features} | None,
            "n_findings_total": int,
        }

    Devuelve None cuando:
      - El módulo `cross_source_patterns` no carga.
      - No hay findings strong (no exponemos noise al brief).
      - Falla silenciosamente la query.

    Threshold: solo findings con `severity in {strong, moderate}` y r
    significativo. Excluimos `weak` para no contaminar el prompt con
    correlaciones débiles que el LLM podría sobre-interpretar."""
    try:
        from rag.cross_source_patterns import (  # noqa: PLC0415
            patterns_summary, predict_mood_tomorrow,
        )
    except Exception:
        return None

    try:
        summary = patterns_summary(days=30, top=10, lags=(0, 1, 7))
    except Exception as exc:
        try:
            from rag import _silent_log as _slog  # noqa: PLC0415
            _slog("today_personal_signals.cross_patterns.summary", exc)
        except Exception:
            pass
        summary = None

    top_findings: list[dict] = []
    n_findings_total = 0
    if summary:
        n_findings_total = summary.get("n_findings", 0)
        # Filtrar a strong + moderate solamente (descarta weak para no
        # ruido el brief).
        for f in summary.get("top", []):
            if f.get("severity") in ("strong", "moderate"):
                top_findings.append({
                    "description": f.get("description"),
                    "r": f.get("r"),
                    "n": f.get("n"),
                    "lag": f.get("lag"),
                    "severity": f.get("severity"),
                })
            if len(top_findings) >= 5:
                break

    try:
        prediction = predict_mood_tomorrow(days=60)
    except Exception as exc:
        try:
            from rag import _silent_log as _slog  # noqa: PLC0415
            _slog("today_personal_signals.cross_patterns.predict", exc)
        except Exception:
            pass
        prediction = None

    if not top_findings and prediction is None:
        return None

    return {
        "top_findings": top_findings,
        "prediction": prediction,
        "n_findings_total": n_findings_total,
    }
