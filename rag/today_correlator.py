"""Cross-source correlator for the today brief.

The today brief receives 14+ buckets of data (gmail_today, whatsapp_today,
calendar_today, youtube_today, recent_notes, low_conf_queries, etc.). A 7B LLM
struggles to find cross-source patterns by reading these flat buckets — it
ends up writing tautological "X is related to Y because both are X" insights.

This module pre-computes the patterns BEFORE the LLM call so the prompt
can include a structured ENTIDADES CROSS-SOURCE block. The LLM then narrates
matches that already exist instead of inventing them.

Three correlations + one post-processor:

1. **People** — names that appear in ≥2 sources (gmail-from, wa-name,
   calendar-title). Canonicalized + deduplicated. Output: list of
   {name, appearances: [{source, context, snippet}], sources_count}.

2. **Topics** — keywords that appear in ≥2 source buckets, normalized
   (lowercase, stopwords removed, ≥4 chars). Output: list of
   {topic, sources, sources_count}.

3. **Time overlaps** — events at same hour bucket cross-source (e.g.
   gmail received 14:23 + calendar event 14:00 with overlapping tokens).
   Output: list of {time, items: [{source, label, snippet}], shared_tokens}.

4. **Voice normalization** (post-processing) — `normalize_voice_to_2da_persona`
   replaces 1st-person verbs ("recibí", "trabajé", "me centré") with 2nd
   person singular ("recibiste", "trabajaste", "te centraste") in the LLM
   output. The prompt forbids 1ª persona but the 7B model slips ~10% of
   the time; this is the safety net.

Used by `web/server.py:_home_compute` → passed to the prompt via
`extras["correlations"]` → rendered in `_render_today_prompt`. The voice
normalizer wraps `_generate_today_narrative`'s return value.
"""

from __future__ import annotations

import re  # noqa: F401 — usado por imports debajo + por _correlate_mood/sleep
from collections import defaultdict  # noqa: F401 — usado por _correlate_mood/sleep


# ── Aggregations: topics + time overlaps + gaps + tokenizer (re-export) ───
# Movido a `rag/today_aggregations.py` (2026-05-09). Re-exportado para
# preservar `from rag.today_correlator import _correlate_topics` etc. y
# para que el orchestrator + tests sigan accediendo via el namespace
# original.
from rag.today_aggregations import (  # noqa: F401, E402
    _STOPWORDS,
    _TIME_RE,
    _TOKEN_RE,
    _WA_GROUP_MARKERS,
    _correlate_gaps,
    _correlate_time_overlaps,
    _correlate_topics,
    _looks_like_wa_group,
    _parse_time_to_minutes,
    _tokenize,
    _topic_source_texts,
)


# ── People correlation (re-export) ─────────────────────────────────────────
# Movido a `rag/today_people_correlator.py` (2026-05-09). Re-exportado para
# preservar `from rag.today_correlator import _correlate_people` etc.
from rag.today_people_correlator import (  # noqa: F401, E402
    _EMAIL_BARE_RE,
    _EMAIL_NAME_RE,
    _SELF_NOTIFICATION_DOMAINS,
    _TITLE_NON_NAMES,
    _TITLE_TOKEN_RE,
    _add_or_merge_appearance,
    _best_display_name,
    _canonicalize_name,
    _canonicals_match,
    _correlate_people,
    _extract_name_from_email,
    _extract_names_from_title,
    _is_self_notification,
)




# ── Voice normalization (re-export) ────────────────────────────────────────
# Movido a `rag/today_voice_normalizer.py` (2026-05-09). Re-exportado para
# preservar `from rag.today_correlator import normalize_voice_to_2da_persona`.
from rag.today_voice_normalizer import (  # noqa: F401, E402
    _VOICE_PRONOUN_REPLACEMENTS,
    _VOICE_VERB_REPLACEMENTS_1PS,
    _make_word_boundary_pattern,
    normalize_voice_to_2da_persona,
)

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
    except Exception:
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
    except Exception:
        ln = None
    if not ln:
        return None
    try:
        ws = weekly_stats()
    except Exception:
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
    except Exception:
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
    except Exception:
        prediction = None

    if not top_findings and prediction is None:
        return None

    return {
        "top_findings": top_findings,
        "prediction": prediction,
        "n_findings_total": n_findings_total,
    }


def correlate_today_signals(today_ev: dict, extras: dict) -> dict:
    """Pre-correlate cross-source signals. Returns:
        {
            "people": [{name, appearances: [...], sources_count}, ...],
            "topics": [{topic, sources, sources_count}, ...],
            "time_overlaps": [{time, items: [...], shared_tokens}, ...],
            "gaps": [{kind, person, hours_waiting, snippet, context}, ...],
            "mood": {score, trend, drift, ...} | None,  # None si feature off
            "sleep": {date, duration_h, quality, anomaly, ...} | None,
        }

    Empty buckets are silently skipped — `today_ev` and `extras` can
    have any subset of keys; missing keys default to []/{}.

    `mood` viene poblado solo cuando `RAG_MOOD_ENABLED=1` Y el daemon
    `mood-poll` ya escribió un row para hoy en `rag_mood_score_daily`.
    En cualquier otro caso queda `None` y el prompt downstream lo
    detecta + skipea la modulación.

    `sleep` viene poblado cuando hay al menos una sesión en
    `rag_sleep_sessions` (Pillow ingester corrió). Si no, `None`.
    El campo `anomaly` adentro del bucket lo llenamos solo cuando
    hay algo digno de narrar — el prompt del brief debería mencionar
    el sueño solo si `sleep.anomaly` no es None.
    """
    return {
        "people": _correlate_people(today_ev or {}, extras or {}),
        "topics": _correlate_topics(today_ev or {}, extras or {}),
        "time_overlaps": _correlate_time_overlaps(today_ev or {}, extras or {}),
        "gaps": _correlate_gaps(today_ev or {}, extras or {}),
        "mood": _correlate_mood(today_ev or {}, extras or {}),
        "sleep": _correlate_sleep(today_ev or {}, extras or {}),
        # Cross-source statistical patterns (Pearson + lag) +
        # mood prediction for tomorrow. Read-only desde
        # cross_source_patterns. None if engine fails / no findings
        # / no prediction (mismo patrón que `mood`).
        "cross_patterns": _correlate_cross_patterns(today_ev or {}, extras or {}),
    }
