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


# ── Personal signals: mood + sleep + cross_patterns (re-export) ──────────
# Movido a `rag/today_personal_signals.py` (2026-05-09). Re-exportado para
# preservar `from rag.today_correlator import _correlate_mood` etc.
from rag.today_personal_signals import (  # noqa: F401, E402
    _correlate_cross_patterns,
    _correlate_mood,
    _correlate_sleep,
)




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
