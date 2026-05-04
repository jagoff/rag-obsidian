"""Tests para Quick Win #3 — recency boost dinámico por intent (2026-05-04).

Cubre:
  - `_query_temporal_intent`: detector regex (ES + EN), 15+ casos.
  - `source_recency_halflife_for_intent`: modifier per-source × per-intent.
  - `source_recency_multiplier_with_intent`: integración numérica.
  - Wiring en `retrieve()`: misma query → mismo intent siempre (determinismo);
    intent recent boost-ea recencia para fuente con halflife.
  - Gate `RAG_INTENT_RECENCY=0`: comportamiento legacy bit-idéntico.
"""
from __future__ import annotations

import math
import os
import time
from unittest import mock

import pytest

import rag


# ── Detector: español + inglés, 15+ casos ──────────────────────────────────


@pytest.mark.parametrize(
    "query, expected",
    [
        # Recent — español
        ("qué hago mañana", "recent"),
        ("próximo turno con María", "recent"),
        ("cuándo es mi próxima reunión", "recent"),
        ("mensajes sin leer", "recent"),
        ("qué pendientes tengo", "recent"),
        ("notas recientes sobre coaching", "recent"),
        ("últimas notas modificadas", "recent"),
        ("hoy a la tarde", "recent"),
        # Recent — inglés
        ("what is my next meeting", "recent"),
        ("upcoming events tomorrow", "recent"),
        ("this week unread emails", "recent"),
        ("latest changes in the repo", "recent"),
        ("right now what's pending", "recent"),
        # Historical — español
        ("qué hablamos el año pasado", "historical"),
        ("cuando estaba en Grecia", "historical"),
        ("hace 3 años trabajaba en X", "historical"),
        ("hace mucho tiempo no veo a Astor", "historical"),
        ("la semana pasada", "historical"),
        # Historical — inglés
        ("previously we discussed this", "historical"),
        ("back when I worked at the company", "historical"),
        ("3 years ago we shipped X", "historical"),
        ("last month I was in Madrid", "historical"),
        # Neutral
        ("coaching con María", "neutral"),
        ("proyecto rag-local", "neutral"),
        ("arquitectura del sistema", "neutral"),
        ("how does retrieval work", "neutral"),
    ],
)
def test_query_temporal_intent_classifies_correctly(query, expected):
    assert rag._query_temporal_intent(query) == expected


def test_query_temporal_intent_empty_string_returns_neutral():
    assert rag._query_temporal_intent("") == "neutral"


def test_query_temporal_intent_none_returns_neutral():
    assert rag._query_temporal_intent(None) == "neutral"


def test_query_temporal_intent_non_string_returns_neutral():
    assert rag._query_temporal_intent(123) == "neutral"
    assert rag._query_temporal_intent(["query"]) == "neutral"


def test_query_temporal_intent_historical_wins_over_recent():
    """Si la query mezcla pistas de las dos categorías, historical
    gana — porque las pistas de pasado distante son menos ambiguas
    que las de presente."""
    # "el año pasado" es historical fuerte; "mañana" sin contexto es recent.
    # La presencia de historical define el bucket.
    assert rag._query_temporal_intent("el año pasado mañana hablábamos de X") == "historical"


def test_query_temporal_intent_is_deterministic():
    """Misma query → mismo intent en N invocaciones — determinismo crítico
    para reproducibilidad del eval gate."""
    queries = [
        "qué hago mañana",
        "el año pasado",
        "coaching con María",
    ]
    for q in queries:
        results = {rag._query_temporal_intent(q) for _ in range(20)}
        assert len(results) == 1, f"non-deterministic for {q!r}: {results}"


def test_query_temporal_intent_case_insensitive():
    assert rag._query_temporal_intent("MAÑANA") == "recent"
    assert rag._query_temporal_intent("Próximo Turno") == "recent"
    assert rag._query_temporal_intent("EL AÑO PASADO") == "historical"


# ── Modifier: source × intent → halflife ───────────────────────────────────


def test_source_recency_halflife_for_intent_recent_wa():
    # WA halflife default = 60d. Recent multiplier = 0.3. → 18d.
    assert rag.source_recency_halflife_for_intent("whatsapp", "recent") == 18.0


def test_source_recency_halflife_for_intent_historical_wa():
    # WA halflife default = 60d. Historical multiplier = 3.0. → 180d.
    assert rag.source_recency_halflife_for_intent("whatsapp", "historical") == 180.0


def test_source_recency_halflife_for_intent_neutral_wa_unchanged():
    # Neutral multiplier = 1.0 → halflife default sin cambios.
    assert rag.source_recency_halflife_for_intent("whatsapp", "neutral") == 60.0


def test_source_recency_halflife_for_intent_vault_always_none():
    # Vault, calendar, contacts tienen halflife None — no decay aunque la
    # query pida lo más reciente.
    assert rag.source_recency_halflife_for_intent("vault", "recent") is None
    assert rag.source_recency_halflife_for_intent("vault", "historical") is None
    assert rag.source_recency_halflife_for_intent("vault", "neutral") is None
    assert rag.source_recency_halflife_for_intent("calendar", "recent") is None
    assert rag.source_recency_halflife_for_intent("contacts", "recent") is None


def test_source_recency_halflife_for_intent_unknown_source_none():
    # Source desconocida → None (defer al fallback de source_recency_multiplier).
    assert rag.source_recency_halflife_for_intent("alien", "recent") is None


def test_source_recency_halflife_for_intent_unknown_intent_neutral_default():
    # Intent desconocido → multiplier 1.0 → halflife default sin cambios.
    assert rag.source_recency_halflife_for_intent("whatsapp", "unknown") == 60.0


def test_source_recency_halflife_for_intent_gmail():
    # Gmail halflife default = 180d.
    assert rag.source_recency_halflife_for_intent("gmail", "recent") == 54.0
    assert rag.source_recency_halflife_for_intent("gmail", "historical") == 540.0
    assert rag.source_recency_halflife_for_intent("gmail", "neutral") == 180.0


# ── Multiplier: integración numérica ───────────────────────────────────────


def test_multiplier_with_intent_recent_decays_faster_than_neutral():
    """Con halflife más corto, una nota de 30d ago decae más en recent
    que en neutral."""
    now = time.time()
    ts_30d = now - 30 * 86400
    mult_recent = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_30d, "recent", now=now,
    )
    mult_neutral = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_30d, "neutral", now=now,
    )
    assert mult_recent < mult_neutral
    # halflife recent = 18d → 30d age → 2 ** -(30/18) ≈ 0.315
    assert math.isclose(mult_recent, 2 ** -(30 / 18), rel_tol=1e-6)
    # halflife neutral = 60d → 30d age → 2 ** -0.5 ≈ 0.707
    assert math.isclose(mult_neutral, 2 ** -0.5, rel_tol=1e-6)


def test_multiplier_with_intent_historical_decays_slower_than_neutral():
    now = time.time()
    ts_30d = now - 30 * 86400
    mult_historical = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_30d, "historical", now=now,
    )
    mult_neutral = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_30d, "neutral", now=now,
    )
    assert mult_historical > mult_neutral
    # halflife historical = 180d → 30d age → 2 ** -(30/180) ≈ 0.891
    assert math.isclose(mult_historical, 2 ** -(30 / 180), rel_tol=1e-6)


def test_multiplier_with_intent_vault_always_one_regardless_of_intent():
    """Vault halflife None → multiplicador 1.0 siempre, aunque la query
    sea recent y la nota sea de hace 5 años."""
    now = time.time()
    ts_5y = now - 5 * 365 * 86400
    for intent in ("recent", "historical", "neutral"):
        mult = rag.source_recency_multiplier_with_intent(
            "vault", ts_5y, intent, now=now,
        )
        assert mult == 1.0


def test_multiplier_with_intent_neutral_matches_legacy_helper():
    """Con intent=neutral el multiplicador debe ser BIT-IDÉNTICO al
    legacy `source_recency_multiplier`. Sin esta paridad, el rollout
    introduciría drift incluso con default OFF."""
    now = time.time()
    ts_50d = now - 50 * 86400
    legacy = rag.source_recency_multiplier("whatsapp", ts_50d, now=now)
    new_neutral = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_50d, "neutral", now=now,
    )
    assert legacy == new_neutral


def test_multiplier_with_intent_none_ts_returns_one():
    # created_ts None → multiplier 1.0 sin importar intent.
    for intent in ("recent", "historical", "neutral"):
        mult = rag.source_recency_multiplier_with_intent(
            "whatsapp", None, intent,
        )
        assert mult == 1.0


def test_multiplier_with_intent_iso_string_ts_works():
    """Acepta epoch float O ISO-8601 string, igual que el legacy."""
    now = time.time()
    iso = "2026-04-04T12:00:00"  # epoch
    mult = rag.source_recency_multiplier_with_intent(
        "whatsapp", iso, "recent", now=now,
    )
    assert 0.0 < mult <= 1.0


def test_multiplier_with_intent_unparseable_ts_returns_one():
    mult = rag.source_recency_multiplier_with_intent(
        "whatsapp", "not-a-date", "recent",
    )
    assert mult == 1.0


# ── Gate RAG_INTENT_RECENCY ────────────────────────────────────────────────


def test_intent_recency_disabled_falls_back_to_legacy(monkeypatch):
    """RAG_INTENT_RECENCY=0 → multiplier_with_intent === source_recency_multiplier
    para CUALQUIER intent. Garantiza rollback bit-idéntico."""
    monkeypatch.setenv("RAG_INTENT_RECENCY", "0")
    now = time.time()
    ts_50d = now - 50 * 86400
    for intent in ("recent", "historical", "neutral"):
        legacy = rag.source_recency_multiplier("whatsapp", ts_50d, now=now)
        new = rag.source_recency_multiplier_with_intent(
            "whatsapp", ts_50d, intent, now=now,
        )
        assert legacy == new, f"intent={intent} drifted with gate OFF"


def test_intent_recency_enabled_default_on(monkeypatch):
    """Sin env var → feature ON (default 2026-05-04). Recent debe diferir
    del legacy para fuentes con halflife."""
    monkeypatch.delenv("RAG_INTENT_RECENCY", raising=False)
    now = time.time()
    ts_30d = now - 30 * 86400
    legacy = rag.source_recency_multiplier("whatsapp", ts_30d, now=now)
    new_recent = rag.source_recency_multiplier_with_intent(
        "whatsapp", ts_30d, "recent", now=now,
    )
    assert new_recent != legacy


def test_intent_recency_disabled_truthy_variants(monkeypatch):
    for v in ("0", "false", "no", "off", "FALSE", "No"):
        monkeypatch.setenv("RAG_INTENT_RECENCY", v)
        assert rag._intent_recency_enabled() is False, f"failed for {v!r}"


def test_intent_recency_enabled_truthy_variants(monkeypatch):
    for v in ("1", "true", "yes", "on", "TRUE", "Yes", ""):
        if v == "":
            monkeypatch.delenv("RAG_INTENT_RECENCY", raising=False)
        else:
            monkeypatch.setenv("RAG_INTENT_RECENCY", v)
        assert rag._intent_recency_enabled() is True, f"failed for {v!r}"


# ── Multipliers constant — sanity ──────────────────────────────────────────


def test_intent_recency_multipliers_table():
    """Anchor — los multipliers del feature están definidos como esperan
    los downstream readers (eval analytics + tune)."""
    assert rag._INTENT_RECENCY_MULTIPLIERS["recent"] == 0.3
    assert rag._INTENT_RECENCY_MULTIPLIERS["historical"] == 3.0
    assert rag._INTENT_RECENCY_MULTIPLIERS["neutral"] == 1.0


# ── Wiring en retrieve() — boost diferencial probado via simulación ───────


def _decay_score_for(query: str, age_days: float) -> float:
    """Helper: simula el factor multiplicativo que aplica `retrieve()` a un
    chunk WA con `age_days` de antigüedad para una query dada. Sin gate
    de feature explícito (asume ON, default 2026-05-04)."""
    intent = rag._query_temporal_intent(query)
    now = time.time()
    ts = now - age_days * 86400
    mult = rag.source_recency_multiplier_with_intent("whatsapp", ts, intent, now=now)
    return mult


def test_retrieve_boost_recent_vs_historical_on_30d_old_chunk():
    """El task pide: query "qué hago mañana" debe boost-ear notas recientes
    vs query "qué hablamos en Grecia" (historical → no boost / relax)."""
    # Una nota WA de 30 días ago.
    boost_recent = _decay_score_for("qué hago mañana", 30)
    boost_historical = _decay_score_for("qué hablamos en Grecia el año pasado", 30)
    boost_neutral = _decay_score_for("notas sobre coaching", 30)
    # Recent decae más rápido — un chunk de 30d en query "mañana" pesa
    # MENOS que el mismo chunk en query historical / neutral.
    # Esto es por diseño: el detector pasivo no PUEDE distinguir entre
    # "mañana hago X" (futuro) y "qué pasa hoy" (presente). Ambas tienen
    # halflife corto. La signal interesante: notas FRESCAS (1d) ganan
    # más en recent que en historical.
    assert boost_recent < boost_neutral
    assert boost_historical > boost_neutral
    # Sanity: ratio coherente con multipliers 0.3 / 3.0.
    # En 30d: recent halflife 18d → 2**-(30/18) ≈ 0.315
    #         historical halflife 180d → 2**-(30/180) ≈ 0.891
    assert math.isclose(boost_recent, 0.315, abs_tol=0.01)
    assert math.isclose(boost_historical, 0.891, abs_tol=0.01)


def test_retrieve_recent_query_gives_fresh_chunks_more_weight():
    """Query recent: chunks de 1 día pesan ~1.0; chunks de 30d pesan ~0.32.
    Query historical: chunks de 1 día ~1.0; chunks de 30d ~0.89.
    El gap más grande es en recent → fresh wins more."""
    fresh_recent = _decay_score_for("próxima reunión", 1)
    old_recent = _decay_score_for("próxima reunión", 30)
    fresh_historical = _decay_score_for("hace 3 años", 1)
    old_historical = _decay_score_for("hace 3 años", 30)
    # Brecha en recent es mayor que en historical
    gap_recent = fresh_recent - old_recent
    gap_historical = fresh_historical - old_historical
    assert gap_recent > gap_historical


# ── Persistencia en RetrieveResult dataclass ───────────────────────────────


def test_retrieve_result_has_temporal_intent_field():
    """Anchor: el dataclass expone `temporal_intent` para que `to_log_event`
    lo persista a `rag_queries.extra_json`."""
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=0.0,
        temporal_intent="recent",
    )
    assert rr.temporal_intent == "recent"
    # Default es "neutral"
    rr_default = rag.RetrieveResult(docs=[], metas=[], scores=[], confidence=0.0)
    assert rr_default.temporal_intent == "neutral"


def test_retrieve_empty_collection_propagates_temporal_intent():
    """Edge: col.count()==0 → return temprano debe llevar el intent
    detectado, no perderlo."""
    class _EmptyCol:
        def count(self):
            return 0

    result = rag.retrieve(
        col=_EmptyCol(),
        question="qué hago mañana",
        k=5,
        folder=None,
    )
    assert result.temporal_intent == "recent"
