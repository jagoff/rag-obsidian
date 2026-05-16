"""Retrieval scoring, source policy, privacy filters, and lightweight MMR."""
from __future__ import annotations

import functools
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

__all__ = [
    'VALID_SOURCES',
    'SOURCE_WEIGHTS',
    'SOURCE_RECENCY_HALFLIFE_DAYS',
    'SOURCE_RETENTION_DAYS',
    'normalize_source',
    '_MEMORY_PATH_PREFIX',
    '_infer_vault_source',
    'source_weight',
    'source_recency_multiplier',
    '_TEMPORAL_INTENT_RECENT_RE',
    '_TEMPORAL_INTENT_HISTORICAL_RE',
    '_INTENT_RECENCY_MULTIPLIERS',
    '_query_temporal_intent_cached',
    '_query_temporal_intent',
    'source_recency_halflife_for_intent',
    '_intent_recency_enabled',
    'source_recency_multiplier_with_intent',
    '_conv_dedup_window',
    '_cross_source_dedup',
    '_CROSS_SOURCE_FILTERS_CACHE',
    '_CROSS_SOURCE_FILTERS_MTIME',
    '_cross_source_filters_path',
    '_load_cross_source_filters',
    '_should_exclude_chunk',
    '_filter_excluded_chunks',
    '_MMR_DIVERSITY_ENABLED',
    '_MMR_SNIPPET_CHARS',
    '_MMR_TOKEN_RE',
    '_mmr_tokens',
    '_jaccard',
    '_compute_adaptive_k',
    '_apply_mmr_reorder',
    'configure_retrieval_scoring',
]


@dataclass(frozen=True)
class _RetrievalScoringDeps:
    db_path: Callable[[], Path]
    silent_log: Callable[[str, BaseException], None]
    get_cross_source_filters_cache: Callable[[], dict | None]
    set_cross_source_filters_cache: Callable[[dict | None], None]
    get_cross_source_filters_mtime: Callable[[], float]
    set_cross_source_filters_mtime: Callable[[float], None]
    mmr_snippet_chars: Callable[[], int]


def _default_db_path() -> Path:
    return Path(
        os.environ.get("OBSIDIAN_RAG_DB_PATH")
        or (Path.home() / ".local/share/obsidian-rag/ragvec")
    )


def _noop_silent_log(where: str, exc: BaseException) -> None:
    del where, exc


def _get_module_filter_cache() -> dict | None:
    return globals().get("_CROSS_SOURCE_FILTERS_CACHE")


def _set_module_filter_cache(value: dict | None) -> None:
    globals()["_CROSS_SOURCE_FILTERS_CACHE"] = value


def _get_module_filter_mtime() -> float:
    return float(globals().get("_CROSS_SOURCE_FILTERS_MTIME", 0.0))


def _set_module_filter_mtime(value: float) -> None:
    globals()["_CROSS_SOURCE_FILTERS_MTIME"] = value


def _module_mmr_snippet_chars() -> int:
    return int(globals().get("_MMR_SNIPPET_CHARS", 600))


_DEPS = _RetrievalScoringDeps(
    db_path=_default_db_path,
    silent_log=_noop_silent_log,
    get_cross_source_filters_cache=_get_module_filter_cache,
    set_cross_source_filters_cache=_set_module_filter_cache,
    get_cross_source_filters_mtime=_get_module_filter_mtime,
    set_cross_source_filters_mtime=_set_module_filter_mtime,
    mmr_snippet_chars=_module_mmr_snippet_chars,
)


def configure_retrieval_scoring(
    *,
    db_path: Callable[[], Path],
    silent_log: Callable[[str, BaseException], None],
    get_cross_source_filters_cache: Callable[[], dict | None],
    set_cross_source_filters_cache: Callable[[dict | None], None],
    get_cross_source_filters_mtime: Callable[[], float],
    set_cross_source_filters_mtime: Callable[[float], None],
    mmr_snippet_chars: Callable[[], int],
) -> None:
    """Wire runtime dependencies from the live ``rag`` facade."""
    global _DEPS
    _DEPS = _RetrievalScoringDeps(
        db_path=db_path,
        silent_log=silent_log,
        get_cross_source_filters_cache=get_cross_source_filters_cache,
        set_cross_source_filters_cache=set_cross_source_filters_cache,
        get_cross_source_filters_mtime=get_cross_source_filters_mtime,
        set_cross_source_filters_mtime=set_cross_source_filters_mtime,
        mmr_snippet_chars=mmr_snippet_chars,
    )


def _deps() -> _RetrievalScoringDeps:
    return _DEPS


# ── Cross-source corpus (Phase 1, 2026-04-20 user decisions §10) ──────────
# The collection stays at v11 (no rename / no re-embed) — source discrimination
# is done purely via a new `source` metadata field with "vault" as the
# default for backward compat. Old chunks without `source` are read as
# "vault" throughout the pipeline (see meta.get("source") or "vault").
#
# The actual ingest paths (WhatsApp, Gmail, Calendar, Reminders) register
# non-vault sources and set `source` explicitly in their metadata. Retrieval
# applies a per-source weight + recency decay in `apply_weighted_scores`
# so non-vault chunks can be surfaced in the same pool without re-calibrating
# the core reranker.
#
# OAuth Google is used for Gmail + Calendar per user override — this breaks
# the "no cloud calls" invariant declared at the top of CLAUDE.md. Tracked
# in docs/design-cross-source-corpus.md §10.6.
VALID_SOURCES: frozenset[str] = frozenset(
    {"vault", "memory", "calendar", "gmail", "whatsapp", "reminders", "messages",
     "contacts", "calls", "safari", "drive", "pillow", "finances", "health"}
)
# `pillow` (iOS sleep tracker) tiene un ingester propio en
# `rag index --source pillow`. Sus datos viven en `rag_sleep_sessions`
# (no en el corpus vectorial), pero la source figura en VALID_SOURCES
# para mantener paridad con `CONFIDENCE_RERANK_MIN_PER_SOURCE` (audit
# de invariantes, ver `test_threshold_helper_all_sources_covered`).

# Per-source weight applied multiplicatively to the final rerank+feature
# score. Vault stays at 1.00 so the default path is a no-op; anything
# non-vault gets softly down-weighted to reflect editorial trust.
SOURCE_WEIGHTS: dict[str, float] = {
    "vault":     1.00,
    "contacts":  0.95,   # editorial trust — user-curated metadata
    "calendar":  0.95,
    "memory":    0.90,   # memo facts/decisions/gotchas — curated by the agent,
                         # softly down-weighted so user-authored vault notes win ties
                         # on queries that match both. See feedback-loop guard in
                         # `is_excluded()` rationale.
    "reminders": 0.90,
    "gmail":     0.85,
    "drive":     0.85,   # Docs/Sheets/Slides: user-authored, high trust like email
    "finances":  0.85,   # PDFs de resúmenes de tarjeta + Excels de movimientos: oficial
    "safari":    0.80,   # browsing signal: rich titles + URLs, same band as calls
    "calls":     0.80,   # log entries: factual but semantically thin
    "whatsapp":  0.75,
    "messages":  0.75,
    # `pillow` (iOS sleep tracker, despachado en `rag index --source pillow`):
    # los datos viven en `rag_sleep_sessions` (no escriben chunks al corpus
    # vectorial), pero la entry queda acá para mantener paridad con
    # `VALID_SOURCES` + `CONFIDENCE_RERANK_MIN_PER_SOURCE`. Weight nominal
    # 0.50 — defensivo si por algún motivo escribiera chunks (no debería).
    "pillow":    0.50,
    # `health` (Apple Health export.xml, despachado en `rag index --source health`):
    # los datos viven en `rag_apple_health_daily` (no escriben chunks al corpus
    # vectorial), pero la entry queda acá para mantener paridad con
    # `VALID_SOURCES`. Weight nominal 0.50 — defensivo.
    "health":    0.50,
}

# Recency half-life per source, in days. None → no decay applied (chunks
# from this source are ranked purely on semantic match + static weight).
# A halflife of H days means a chunk aged H days old gets scored at 0.5×
# its fresh value via recency_boost_for_source(); 2H days → 0.25×, etc.
# Rationale per source (from §4.2 of the design doc):
#   - vault / calendar / contacts: people/events/notes don't age, skip decay
#   - gmail / reminders: mid-term — a 6-month old email is context, not noise
#   - whatsapp / messages / calls: conversational — a 2-month-old trace rarely matters
SOURCE_RECENCY_HALFLIFE_DAYS: dict[str, float | None] = {
    "vault":     None,
    "memory":    None,   # curated knowledge — no temporal decay
    "contacts":  None,
    "calendar":  None,
    "reminders":   90.0,
    "gmail":      180.0,
    "drive":       90.0,   # Google Docs age between email (180d) and chat (30d)
    "finances":   180.0,   # resúmenes de tarjeta relevantes ~6 meses
    "safari":      90.0,   # browsing context ages mid-term
    # WhatsApp/Messages/Calls bumpeados de 30→60 días (audit 2026-04-25
    # R2-Cross-source #5). 30d era muy agresivo: una conversación de
    # hace 31 días caía a 0.5× del peso, perdiendo contra notas viejas
    # del vault aunque fuera más relevante. Caso real del audit:
    # "qué dijo X sobre Y" devolvía nota vault de hace 90d en lugar de
    # un WA de hace 35d (mismo tema, más reciente). 60d half-life
    # mantiene la decadencia conversacional pero da más margen.
    "whatsapp":    60.0,
    "messages":    60.0,
    "calls":       60.0,
    # `pillow`: no aplica recency decay porque no escribe chunks al corpus.
    # Entry presente para coverage del `set(SOURCE_RECENCY_HALFLIFE_DAYS) == VALID_SOURCES`.
    "pillow":     None,
    # `health`: Apple Health daily aggregates viven en tabla SQL dedicada,
    # no como chunks vectoriales. Entry defensiva para mantener invariantes.
    "health":     None,
}

# Retention windows per source, in days. None → keep forever. Used at
# INGEST time (the ingester drops rows older than this) + as a hard
# upper bound for the cleanup path. Notes aren't touched by this
# (vault retention is manual).
SOURCE_RETENTION_DAYS: dict[str, int | None] = {
    "vault":     None,
    "memory":    None,   # never auto-purge memo entries
    "contacts":  None,
    "calendar":  None,
    "reminders": None,
    "gmail":      365,
    "drive":      365,   # user's Drive docs — keep a year like email
    "finances":   None,   # documentos financieros útiles histórico
    "safari":     180,
    "whatsapp":   180,
    "messages":   180,
    "calls":      180,
    # `pillow`: el ingester maneja su propio retention sobre `rag_sleep_sessions`.
    # Acá None para coverage del set check.
    "pillow":     None,
    # `health`: el ingester maneja su propia tabla `rag_apple_health_daily`.
    "health":     None,
}


def normalize_source(value: object, *, default: str = "vault") -> str:
    """Return a valid source string, falling back to `default` on anything
    that isn't in `VALID_SOURCES`. Used wherever we read `meta.get("source")`
    from possibly-legacy metadata."""
    if isinstance(value, str) and value in VALID_SOURCES:
        return value
    return default


# Path-prefix discriminator: chunks whose vault-relative path starts with
# `99-obsidian/99-AI/memory/` belong to the agent-curated
# memo. They get `source="memory"` (weight=0.90) instead of
# `source="vault"` (weight=1.00), so user-authored notes win ties on
# overlapping queries. The carve-out exists because memo entries
# describe THE SYSTEM ITSELF (bug patterns, decisions, gotchas) and were
# previously dominating retrieval at >50% of top-k for technical queries.
# See `is_excluded()` for why memories stay indexed at all.
_MEMORY_PATH_PREFIX = "99-obsidian/99-AI/memory/"


def _infer_vault_source(rel_path: str) -> str:
    """`memory` for memo entries, `vault` otherwise. Called from the
    indexer (`_index_single_file` + the rglob path in `_run_index`) so the
    discriminator is set at write time. Cheap path-prefix check."""
    return "memory" if (rel_path or "").startswith(_MEMORY_PATH_PREFIX) else "vault"


def source_weight(source: str) -> float:
    """Per-source multiplier applied to the final score. Unknown source →
    0.50 (half the vault baseline, same band as messages/whatsapp) — this
    is defensive and should never hit in practice since `normalize_source`
    filters unknowns upstream."""
    return SOURCE_WEIGHTS.get(source, 0.50)


def source_recency_multiplier(
    source: str, created_ts: float | str | None, *, now: float | None = None,
) -> float:
    """Exponential decay multiplier in [0, 1] based on the per-source
    halflife. Returns 1.0 for sources with no halflife configured or when
    `created_ts` is missing/unparseable. `created_ts` accepts epoch seconds
    (float) or ISO-8601 string (matches the format used everywhere else in
    the metadata).
    """
    halflife = SOURCE_RECENCY_HALFLIFE_DAYS.get(source)
    if halflife is None or halflife <= 0:
        return 1.0
    if created_ts is None:
        return 1.0
    try:
        if isinstance(created_ts, (int, float)):
            ts_epoch = float(created_ts)
        else:
            # Accept ISO-8601 with or without timezone; naive → assume local.
            s = str(created_ts)
            dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            ts_epoch = dt.timestamp()
    except (TypeError, ValueError):
        return 1.0
    now_epoch = now if now is not None else time.time()
    age_days = max(0.0, (now_epoch - ts_epoch) / 86400.0)
    # 2 ** -(age / halflife) → 1.0 at age=0, 0.5 at age=halflife, 0.25 at 2×halflife
    import math as _math
    return float(_math.pow(2.0, -(age_days / halflife)))


# ── Quick Win #3: recency boost dinámico por intent (2026-05-04) ────────────
# Capa de modifiers ENCIMA de SOURCE_RECENCY_HALFLIFE_DAYS. La idea: queries
# time-sensitive ("qué hago mañana", "próximo turno", "hoy") quieren un boost
# muchísimo más fuerte de notas recientes que queries históricas ("qué
# hablamos el año pasado", "cuando estaba en Grecia"). En vez de tunear los
# halflifes globales — que afectan al pipeline entero — clasificamos la
# query como recent / historical / neutral y multiplicamos el halflife
# default por un factor:
#
#   recent      → halflife × 0.3  (decay 3.3× más rápido → boost agresivo)
#   historical  → halflife × 3.0  (decay 3× más lento → notas viejas siguen
#                                  jugando)
#   neutral     → halflife × 1.0  (no-op, comportamiento histórico)
#
# Un halflife None (vault, calendar, contacts) queda None siempre — no tiene
# sentido decaer notas que no tienen edad relevante (un evento de calendar
# es atemporal hasta que el filtro temporal lo selecciona).
#
# Detector regex-puro, case-insensitive, español + inglés, determinístico,
# 0% LLM. Patterns ordenados de más específico a más general.

_TEMPORAL_INTENT_RECENT_RE = re.compile(
    r"\b(?:"
    # Español temporal cercano
    r"pr[oó]xim[oa]s?|"                       # próximo / próxima / próximos
    r"cu[aá]ndo|"                              # cuándo (no "cuando estaba/iba")
    r"ma[ñn]ana|"                              # mañana
    r"hoy|"                                    # hoy
    r"ayer|"                                   # ayer (cercano, contextual)
    r"esta\s+semana|"                          # esta semana
    r"este\s+mes|"                             # este mes
    r"esta\s+tarde|esta\s+noche|"             # esta tarde / esta noche
    r"ahora(?:\s+mismo)?|"                     # ahora / ahora mismo
    r"pendiente(?:s)?|"                        # pendientes
    r"sin\s+leer|"                             # sin leer (unread)
    r"reciente(?:s|mente)?|"                   # reciente / recientes / recientemente
    r"[uú]ltim[oa]s?|"                         # último / última / últimos
    r"siguiente(?:s)?|"                        # siguiente / siguientes
    # Inglés temporal cercano
    r"next|"                                   # next (meeting, week, etc.)
    r"upcoming|"                               # upcoming
    r"tomorrow|"                               # tomorrow
    r"today|"                                  # today
    r"yesterday|"                              # yesterday
    r"this\s+week|this\s+month|"              # this week / this month
    r"this\s+afternoon|this\s+evening|"       # this afternoon / evening
    r"right\s+now|"                            # right now
    r"unread|"                                 # unread
    r"latest|recent(?:ly)?|"                   # latest / recent / recently
    r"current(?:ly)?"                          # current / currently
    r")\b",
    re.IGNORECASE,
)

# Para el patrón "próxima reunión / next meeting" pedimos co-ocurrencia con
# un sustantivo conversacional/agenda — eso evita que "el próximo paso" en
# notas de proyecto (sin tinte temporal) caiga en recent. NO obligatorio,
# es un refuerzo: el regex de arriba ya cubre "próxima reunión" via "próxima"
# stand-alone.

_TEMPORAL_INTENT_HISTORICAL_RE = re.compile(
    r"\b(?:"
    # Español pasado lejano
    r"el\s+a[ñn]o\s+pasado|"                   # el año pasado
    r"a[ñn]o\s+pasado|"                        # año pasado
    r"el\s+mes\s+pasado|mes\s+pasado|"        # el mes pasado / mes pasado
    r"la\s+semana\s+pasada|semana\s+pasada|"  # la semana pasada
    r"hace\s+(?:un|una|\d+|mucho|tanto)\s+|"  # hace X tiempo (un mes, 3 años, mucho)
    r"hace\s+a[ñn]os|hace\s+meses|"           # hace años / hace meses
    r"hace\s+rato|hace\s+tiempo|"             # hace rato / hace tiempo
    r"antes\s+de|"                             # antes de
    r"cuando\s+(?:estaba|era|viv[ií]a|trabajaba|estudiaba|estuve|fui|fuimos|tenia|ten[ií]a)|"  # cuando estaba/era/etc
    r"hist[oó]ric[oa]s?|"                     # histórico / histórica
    # Inglés pasado lejano
    r"last\s+year|last\s+month|last\s+week|" # last year/month/week
    r"\d+\s+(?:years?|months?|weeks?)\s+ago|" # X years ago
    r"years?\s+ago|months?\s+ago|"            # years ago / months ago
    r"a\s+long\s+time\s+ago|long\s+ago|"      # a long time ago
    r"back\s+(?:in|when)|"                     # back in / back when
    r"when\s+I\s+(?:was|lived|worked|studied|used\s+to)|"  # when I was/lived/worked
    r"history\s+of|historical(?:ly)?|"        # history of / historical
    r"in\s+the\s+past|"                        # in the past
    r"previously"                              # previously
    r")\b",
    re.IGNORECASE,
)

# Multiplicadores per-intent. Default OFF significa "neutral" → 1.0 (no-op,
# halflife sin cambios). Tuneables si los floors de eval lo justifican; los
# valores actuales son conservadores (0.3 / 3.0) — un boost agresivo pero
# acotado: un halflife de 60d en WA queda en 18d para recent, 180d para
# historical (cabe la conversación de hace 6 meses sin colapsar a 0.5×).
_INTENT_RECENCY_MULTIPLIERS: dict[str, float] = {
    "recent":     0.3,
    "historical": 3.0,
    "neutral":    1.0,
}


@functools.lru_cache(maxsize=256)
def _query_temporal_intent_cached(text: str) -> str:
    """Pure inner — sólo el regex match. Determinístico, LRU 256."""
    if _TEMPORAL_INTENT_HISTORICAL_RE.search(text):
        return "historical"
    if _TEMPORAL_INTENT_RECENT_RE.search(text):
        return "recent"
    return "neutral"


def _query_temporal_intent(query: str) -> str:
    """Clasifica la intent temporal de la query → recent | historical | neutral.

    Regex puro, case-insensitive, español + inglés. Determinístico (misma
    query → mismo intent siempre). Sin LLM, sin estado.

    Reglas de precedencia:
      1. Si matchea historical → "historical". Las pistas de pasado
         distante son menos ambiguas que las de presente (el regex de
         "hace X tiempo" o "el año pasado" no se confunde con prosa
         neutra). Chequeamos primero para que "qué hablamos el año
         pasado" no se confunda con "esta semana".
      2. Si matchea recent → "recent".
      3. Default → "neutral".

    Empty / None / no-string input → "neutral" (defensivo, no rompe a
    callers que olvidan validar).

    Cache: LRU 256 sobre el inner — queries ya vistas vuelven en O(1)
    sin re-evaluar 2 regex (~100-200µs por miss).
    """
    if not query or not isinstance(query, str):
        return "neutral"
    text = query.strip()
    if not text:
        return "neutral"
    return _query_temporal_intent_cached(text)


def source_recency_halflife_for_intent(
    source: str, intent: str,
) -> float | None:
    """Halflife default per source ajustado por la intent temporal.

    `None` halflife (vault, calendar, contacts) queda `None` SIEMPRE —
    fuentes atemporales no tienen sentido decaer aunque la query pida
    "lo más reciente".

    Fallback defensivo:
      - source desconocida → None (defer a `source_recency_multiplier`
        cuyo lookup en SOURCE_RECENCY_HALFLIFE_DAYS también devuelve None).
      - intent desconocido → multiplicador 1.0 (= halflife default).
    """
    base = SOURCE_RECENCY_HALFLIFE_DAYS.get(source)
    if base is None:
        return None
    mult = _INTENT_RECENCY_MULTIPLIERS.get(intent, 1.0)
    return base * mult


def _intent_recency_enabled() -> bool:
    """Gate del feature. Default ON desde 2026-05-04 (Quick Win #3).

    Setear ``RAG_INTENT_RECENCY=0`` desactiva el wiring y `retrieve()`
    cae al `source_recency_multiplier` legacy (sin ajuste por intent).
    Útil para A/B contra el baseline cuando el eval gate se mueva y
    haya que aislar la causa.
    """
    val = os.environ.get("RAG_INTENT_RECENCY", "").strip().lower()
    return val not in ("0", "false", "no", "off")


def source_recency_multiplier_with_intent(
    source: str, created_ts: float | str | None, intent: str,
    *, now: float | None = None,
) -> float:
    """Variante de `source_recency_multiplier` que respeta la intent
    temporal de la query. Cuando el feature está apagado o el intent es
    `neutral`, devuelve EXACTAMENTE lo mismo que `source_recency_multiplier`
    (bit-idéntico — no introduce drift).

    Usa `source_recency_halflife_for_intent` para decidir el halflife
    efectivo y replica el cálculo de decay (epoch parsing + age en días
    + `2 ** -(age/halflife)`) para no depender del `_*_HALFLIFE_DAYS`
    global del módulo en este path.

    Casos triviales (idénticos al legacy):
      - halflife None (vault/calendar/contacts) → 1.0
      - halflife <= 0 → 1.0
      - created_ts None → 1.0
      - parse error → 1.0
    """
    if not _intent_recency_enabled() or intent == "neutral":
        return source_recency_multiplier(source, created_ts, now=now)
    halflife = source_recency_halflife_for_intent(source, intent)
    if halflife is None or halflife <= 0:
        return 1.0
    if created_ts is None:
        return 1.0
    try:
        if isinstance(created_ts, (int, float)):
            ts_epoch = float(created_ts)
        else:
            s = str(created_ts)
            dt = datetime.fromisoformat(s.replace("Z", "+00:00") if s.endswith("Z") else s)
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            ts_epoch = dt.timestamp()
    except (TypeError, ValueError):
        return 1.0
    now_epoch = now if now is not None else time.time()
    age_days = max(0.0, (now_epoch - ts_epoch) / 86400.0)
    import math as _math
    return float(_math.pow(2.0, -(age_days / halflife)))


def _conv_dedup_window(
    scored_pairs: list[tuple], *, window_s: float = 1800.0,
) -> list[tuple]:
    """Collapse WhatsApp (and `messages`) chunks within a time window
    per `chat_jid`. Input: list of (candidate, expanded_text, score) tuples
    sorted by score descending. Returns a filtered list preserving the
    original order, dropping any WA/messages chunk whose `chat_jid`
    already has a higher-scored representative within ±`window_s` of its
    `first_ts`.

    Rationale (design doc §3.3 option A): a single conversation can yield
    5-10 adjacent chunks that all fire on the same query. Without dedup,
    the top-k degenerates into "the same conversation shown 5 different
    ways" and the LLM's context is filled with near-duplicates. Keeping
    the top-scored chunk per conversation-window preserves granularity
    ("the specific message where X said Y") while cutting noise.

    Non-WA sources pass through unchanged — vault, calendar, gmail,
    reminders either don't have this failure mode (vault is already deduped
    by hash per file) or the window semantics don't apply (calendar events
    are points in time, not windows).

    Intentionally simple O(n²) scan per chat — the pool is capped at
    ~RERANK_POOL_MAX so the constant factor is negligible (<1ms for 40
    candidates).
    """
    if not scored_pairs:
        return scored_pairs
    # Per-chat: list of first_ts values already accepted. We compare each
    # incoming WA candidate against these and skip if any is within window.
    kept_by_chat: dict[str, list[float]] = {}
    out: list[tuple] = []
    for pair in scored_pairs:
        candidate, expanded, score = pair
        meta = candidate[1] if isinstance(candidate[1], dict) else {}
        src = meta.get("source") or "vault"
        # Only apply to conversational sources — WhatsApp today, SMS later.
        if src not in ("whatsapp", "messages"):
            out.append(pair)
            continue
        jid = meta.get("chat_jid") or meta.get("file") or ""
        first_ts = meta.get("first_ts")
        try:
            first_ts = float(first_ts) if first_ts is not None else None
        except (TypeError, ValueError):
            first_ts = None
        if not jid or first_ts is None:
            # Missing metadata — can't dedup safely, keep the chunk.
            out.append(pair)
            continue
        # Already accepted something in this chat within window?
        accepted = kept_by_chat.get(jid, [])
        if any(abs(first_ts - prior_ts) <= window_s for prior_ts in accepted):
            continue
        accepted.append(first_ts)
        kept_by_chat[jid] = accepted
        out.append(pair)
    return out


def _cross_source_dedup(
    scored_pairs: list[tuple], *, jaccard_threshold: float = 0.7,
) -> list[tuple]:
    """Colapsa chunks de FUENTES DISTINTAS que cubren el mismo evento/decisión.

    Caso real (audit 2026-04-25 R2-Cross-source #2): el user decide algo
    en una nota del vault, lo confirma por mail, lo agenda en Calendar y
    lo coordina por WhatsApp. Pre-fix el RAG devolvía los 4 chunks como
    si fueran 4 evidencias separadas, llenando el contexto del LLM con
    near-duplicates.

    Algoritmo conservador: para cada par de pairs (i, j) con
    ``source[i] != source[j]``, computa Jaccard de tokens (primeros
    ~600 chars del expanded text, normalizados a lowercase + sin
    puntuación). Si Jaccard >= ``jaccard_threshold``, mantiene SOLO
    el pair de mayor score y descarta el otro.

    Threshold default 0.7 (audit recomienda conservador): 2 chunks que
    comparten 70% de tokens en sus primeros 600 chars son casi
    seguramente la misma cosa. Bajo eso → false positives donde temas
    relacionados pero distintos se colapsan.

    Override via ``RAG_CROSS_SOURCE_DEDUP_THRESHOLD`` env var. Set a
    1.0 para deshabilitar (escape hatch).

    O(n²) en el pool — irrelevante porque el pool está capped a ~40.
    """
    threshold = float(os.environ.get(
        "RAG_CROSS_SOURCE_DEDUP_THRESHOLD", str(jaccard_threshold),
    ))
    if threshold >= 1.0 or len(scored_pairs) < 2:
        return scored_pairs

    # Pre-compute token sets para cada pair. Usamos solo primeros 600
    # chars: si 2 chunks comparten ese prefijo, son casi seguro la misma
    # cosa, y reduce el costo del Jaccard.
    def _tokenize(text: str) -> frozenset[str]:
        # Lowercase + strip puntuación + split. Tokens de longitud >= 3
        # para filtrar stopwords cortas (de, en, la, etc.) que generan
        # match espurio.
        s = (text or "")[:600].lower()
        # Reemplazo simple de puntuación por espacios.
        for p in ".,;:!?¿¡()[]{}\"'`-_/\\|*~":
            s = s.replace(p, " ")
        return frozenset(t for t in s.split() if len(t) >= 3)

    sources: list[str] = []
    tokens_list: list[frozenset[str]] = []
    for cand, expanded, _score in scored_pairs:
        meta = cand[1] if isinstance(cand[1], dict) else {}
        sources.append(meta.get("source") or "vault")
        tokens_list.append(_tokenize(str(expanded)))

    # Greedy: iteramos en orden de score (input ya viene sorted desc).
    # Para cada pair, descartamos pairs posteriores que sean cross-source
    # near-dup. Mantiene el de mayor score por construcción.
    # Audit 2026-04-26 (MEDIUM): docs muy cortos (<50 chars, <8 tokens)
    # generaban falsos positivos. "Reunión Max martes 18hs" (~50 chars)
    # vs un WA message similar producen Jaccard >0.83 con 5 tokens
    # compartidos — uno se borra. Skipear pairs donde AMBOS son cortos.
    # 50 chars es ~8-10 tokens significativos — suficiente para Jaccard
    # robusto sin perder la utilidad de dedup en notas de tamaño normal.
    _DEDUP_MIN_LEN = 50
    pair_lens = []
    for cand, expanded, _ in scored_pairs:
        pair_lens.append(len(str(expanded)))
    drop_indices: set[int] = set()
    for i in range(len(scored_pairs)):
        if i in drop_indices:
            continue
        ti = tokens_list[i]
        if not ti:
            continue
        for j in range(i + 1, len(scored_pairs)):
            if j in drop_indices:
                continue
            if sources[i] == sources[j]:
                continue  # Solo cross-source — dedup intra-source es _conv_dedup_window
            tj = tokens_list[j]
            if not tj:
                continue
            # Skip pairs where both docs are short — Jaccard inflates
            # over small token sets.
            if min(pair_lens[i], pair_lens[j]) < _DEDUP_MIN_LEN:
                continue
            inter = len(ti & tj)
            union = len(ti | tj)
            if union == 0:
                continue
            if (inter / union) >= threshold:
                drop_indices.add(j)

    return [p for k, p in enumerate(scored_pairs) if k not in drop_indices]


# ── Cross-source privacy opt-out (audit 2026-04-25 R2-Cross-source #1) ─────
# Filtro opt-in que excluye del retrieval chunks de fuentes sensibles
# (mails de banking/2FA, chats privados específicos, calendarios marcados,
# etc.). El user crea ``~/.local/share/obsidian-rag/cross-source.yaml``
# con las reglas; sin ese archivo, no hay filtro (default: indexar todo,
# como decidió el user en §10.5 del design doc).
#
# Schema esperado:
#
#   gmail:
#     exclude_labels: [banking, 2fa, otp]
#     exclude_senders: ["*@bank.com", "noreply@2fa.com"]
#   whatsapp:
#     exclude_chats: ["+5491112345678@s.whatsapp.net"]
#   calendar:
#     exclude_calendars: ["Privado"]
#
# El glob simple ``*@bank.com`` matchea fnmatch — no full regex.

_CROSS_SOURCE_FILTERS_CACHE: dict | None = None
_CROSS_SOURCE_FILTERS_MTIME: float = 0.0


def _cross_source_filters_path():
    return _deps().db_path() / "cross-source.yaml"


def _load_cross_source_filters() -> dict:
    """Carga las reglas de exclusión cross-source desde YAML. Cachea
    en memoria con invalidation por mtime — recargas el yaml en hot
    reload sin reiniciar el server.

    Devuelve ``{}`` (no filtra nada) si el archivo no existe, no parsea,
    o contiene un schema inesperado. Silent-fail intencional: la
    privacidad es opt-in, no queremos que un YAML malformado rompa
    todo el retrieval.
    """
    path = _cross_source_filters_path()
    if not path.is_file():
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _deps().get_cross_source_filters_cache() or {}
    cache = _deps().get_cross_source_filters_cache()
    cache_mtime = _deps().get_cross_source_filters_mtime()
    if cache is not None and abs(mtime - cache_mtime) < 0.001:
        return cache
    try:
        import yaml  # noqa: PLC0415
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception as exc:
        # H-7 fix (2026-05-08): NO memoizar `{}` en falla transitoria.
        # Pre-fix: un YAML parse error o read error one-shot deshabilitaba
        # *todas* las reglas cross-source hasta el próximo restart, porque
        # la rama de error caía al cache write final. Ahora: log + return
        # `{}` SIN tocar el cache, así el próximo call reintenta el read.
        _deps().silent_log("cross_source_filters_load", exc)
        return {}
    # Cache solo en el success path — failure transient no envenena el slot.
    _deps().set_cross_source_filters_cache(raw)
    _deps().set_cross_source_filters_mtime(mtime)
    return raw


def _should_exclude_chunk(meta: dict, filters: dict | None = None) -> bool:
    """Determina si un chunk debe ser excluido del retrieval por reglas
    de privacidad cross-source. Devuelve True para excluir.

    Reglas suportadas por source:

    - gmail: ``exclude_labels`` (lista de Gmail labels), ``exclude_senders``
      (lista de glob patterns sobre el campo ``from`` del mail)
    - whatsapp: ``exclude_chats`` (lista de jids exactos)
    - calendar: ``exclude_calendars`` (lista de nombres de calendarios)

    Si no hay filtros para el source del chunk, devuelve False (no excluye).
    """
    if filters is None:
        filters = _load_cross_source_filters()
    if not filters:
        return False
    source = (meta.get("source") or "vault").lower()
    src_filters = filters.get(source)
    if not isinstance(src_filters, dict):
        return False

    if source == "gmail":
        # exclude_labels: matchea contra meta["labels"] (lista) o
        # meta["label"] (singular). Case-insensitive.
        excl_labels = src_filters.get("exclude_labels") or []
        if excl_labels:
            chunk_labels = meta.get("labels") or meta.get("label") or []
            if isinstance(chunk_labels, str):
                chunk_labels = [chunk_labels]
            chunk_labels_lower = {str(l).lower() for l in chunk_labels}
            for excl in excl_labels:
                if str(excl).lower() in chunk_labels_lower:
                    return True
        # exclude_senders: glob patterns contra meta["from"]
        excl_senders = src_filters.get("exclude_senders") or []
        if excl_senders:
            sender = (meta.get("from") or meta.get("sender") or "").lower()
            if sender:
                import fnmatch  # noqa: PLC0415
                for pat in excl_senders:
                    if fnmatch.fnmatch(sender, str(pat).lower()):
                        return True
    elif source in ("whatsapp", "messages"):
        excl_chats = src_filters.get("exclude_chats") or []
        if excl_chats:
            jid = meta.get("chat_jid") or meta.get("jid") or ""
            if jid in excl_chats:
                return True
    elif source == "calendar":
        excl_calendars = src_filters.get("exclude_calendars") or []
        if excl_calendars:
            cal = meta.get("calendar") or meta.get("calendar_name") or ""
            if cal in excl_calendars:
                return True
    return False


def _filter_excluded_chunks(scored_pairs: list[tuple]) -> list[tuple]:
    """Aplica los filtros cross-source de privacidad. Returns una lista
    sin los chunks que matchean alguna regla de exclusión.

    Logguea cuántos chunks se filtraron en silent_errors para
    observabilidad — si el user reporta "no encuentro mi mail bancario",
    el log dice "filtered N gmail chunks por exclude_labels".
    """
    filters = _load_cross_source_filters()
    if not filters:
        return scored_pairs
    out: list[tuple] = []
    excluded_count = 0
    for pair in scored_pairs:
        cand = pair[0]
        meta = cand[1] if isinstance(cand[1], dict) else {}
        if _should_exclude_chunk(meta, filters):
            excluded_count += 1
            continue
        out.append(pair)
    if excluded_count > 0:
        try:
            _deps().silent_log(
                "cross_source_filter_applied",
                Exception(f"excluded {excluded_count} chunks by privacy filters"),
            )
        except Exception:
            pass
    return out


# ── MMR diversity re-ranking (Feature #5 del 2026-04-23) ─────────────────
# Post cross-encoder re-rank, re-order the candidate pool balancing
# relevance (rerank score) vs diversity (token overlap w/ already-selected
# docs). Reduces redundancy in top-k: if three chunks say the same thing
# in slightly different words, MMR promotes a chunk from a different angle.
#
# Algorithm (Carbonell & Goldstein 1998 adapted to our tuple shape):
#   Let D = candidate pool sorted by relevance.
#   Pick S = {top-1}.
#   While |S| < k and D \ S non-empty:
#       for each d in D \ S:
#           mmr(d) = λ · rel(d) - (1-λ) · max_{d' ∈ S} sim(d, d')
#       pick arg-max → append to S.
#
# sim() here is Jaccard over word tokens of the first ~600 chars of each
# doc. Cheap + dependency-free (no extra embeddings needed); captures
# near-duplicates at the word level without fancy semantic similarity.
# λ default 0.7 (bias toward relevance). 1.0 = pure relevance (MMR no-op);
# 0.0 = pure diversity (ignores rerank scores).
#
# Gate:
#   RAG_MMR_DIVERSITY=1 to enable (default OFF — conservative rollout).
#   Operates on the pool of up to `pool_size` candidates. Remaining pool
#   order preserved (not reordered), so the extras_pairs slice stays
#   reranker-ordered.

_MMR_DIVERSITY_ENABLED = os.environ.get(
    "RAG_MMR_DIVERSITY", ""
).strip().lower() in ("1", "true", "yes")

_MMR_SNIPPET_CHARS = 600  # chars of each doc to tokenize for Jaccard
_MMR_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


def _mmr_tokens(text: str) -> frozenset[str]:
    """Tokens used for Jaccard similarity — first ~600 chars, word chars
    only, lowercased. Returns frozenset (hashable, set-op friendly)."""
    if not text:
        return frozenset()
    sample = text[:_deps().mmr_snippet_chars()].lower()
    return frozenset(_MMR_TOKEN_RE.findall(sample))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity — |a ∩ b| / |a ∪ b|. Empty inputs → 0.0."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _compute_adaptive_k(
    scores: list[float], *, k_default: int, min_k: int = 2,
    gap_ratio: float = 0.35,
) -> int:
    """Decide how many top-k results to keep based on score distribution.

    Feature #14 del 2026-04-23. Scans the sorted scores (highest first) and
    finds the first significant drop: if (prev - cur) / |prev| > gap_ratio,
    the cur candidate is "noticeably worse" and we truncate there.

    Reduces prefill token consumption when the top-1 clearly dominates
    (easy queries). Never goes below `min_k` — a rerank that returned
    4 equally-good candidates shouldn't collapse to 1. Never exceeds
    the input length or k_default.

    Conservative: if scores are all negative, noise-level, or NaN, falls
    back to k_default. Never raises.

    Examples:
      [1.2, 0.1, 0.05, 0.02]   gap_ratio=0.35 → k=1 but clamped to min_k=2
      [1.2, 1.1, 0.2, 0.1]     gap_ratio=0.35 → k=2 (drop between #2 and #3)
      [1.0, 0.9, 0.8, 0.7]     gap_ratio=0.35 → k=k_default (no clear drop)
    """
    if not scores:
        return k_default
    n = min(len(scores), k_default)
    if n <= min_k:
        return n
    # Scan for the first significant drop.
    for i in range(1, n):
        prev = scores[i - 1]
        cur = scores[i]
        if prev <= 0:
            # Avoid division surprises on negative scores — just bail to
            # k_default. All-negative scenario = no clear signal.
            continue
        drop = (prev - cur) / abs(prev)
        if drop >= gap_ratio:
            # Found a cliff: keep up to and including index i-1.
            return max(min_k, i)
    return n


def _apply_mmr_reorder(
    scored_pairs: list[tuple],
    *,
    lambda_: float = 0.7,
    pool_size: int | None = None,
) -> list[tuple]:
    """Re-order the first `pool_size` items of `scored_pairs` using MMR.

    Input: list of (candidate, expanded_text, score) sorted by score desc.
    Output: same length; first `pool_size` entries reordered to balance
    relevance with diversity; remainder preserved unchanged.

    O(pool_size²) — fine because pool_size is typically 15-30. The first
    item is always kept (highest relevance), subsequent items selected
    greedily by max MMR score.
    """
    if not scored_pairs:
        return scored_pairs
    lambda_ = max(0.0, min(1.0, lambda_))
    if pool_size is None or pool_size >= len(scored_pairs):
        pool = scored_pairs[:]
        tail: list[tuple] = []
    else:
        pool = scored_pairs[:pool_size]
        tail = scored_pairs[pool_size:]
    if len(pool) <= 1:
        return scored_pairs
    # Precompute token sets for the pool.
    tokens: list[frozenset[str]] = []
    for _, expanded, _ in pool:
        tokens.append(_mmr_tokens(expanded if isinstance(expanded, str) else ""))
    selected_idx: list[int] = [0]  # always keep the highest-relevance first
    remaining: set[int] = set(range(1, len(pool)))
    while remaining:
        best_idx = None
        best_mmr = -1e18
        for i in remaining:
            rel = float(pool[i][2])
            max_sim = 0.0
            for j in selected_idx:
                sim = _jaccard(tokens[i], tokens[j])
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.discard(best_idx)
    reordered = [pool[i] for i in selected_idx]
    return reordered + tail
