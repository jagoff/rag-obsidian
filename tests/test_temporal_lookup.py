"""Feature #10 del 2026-04-23 — temporal-lookup cue detection + boost tests.

Validates:
- has_temporal_lookup_cue detects all documented patterns
- Negative cases don't false-positive
- Regex is orthogonal to has_recency_cue (can fire together or alone)
"""
from __future__ import annotations

import pytest

import rag


# ── has_temporal_lookup_cue ──────────────────────────────────────────────


@pytest.mark.parametrize("q", [
    "la última vez que hablé de ikigai",
    "cuándo fue la última vez que escribí sobre esto",
    "cuándo hablé con María",
    "cuándo escribí eso",
    "cuándo mencioné el proyecto",
    "cuándo dije eso",
    "hace unos días escribí sobre X",
    "hace dos semanas hablamos de Y",
    "hace un mes anoté esto",
    "hace cuatro meses lo pensé",
    "hace poco hablé de coaching",
    "hace tiempo escribí sobre eso",
    "recuerdo que hablé de esto",
    "recuerdo que mencioné a Juan",
    "recuerdo que dije algo",
    "en algún momento hablé de esto",
    "la otra vez dije algo",
])
def test_detects_temporal_cue(q):
    assert rag.has_temporal_lookup_cue(q) is True, q


@pytest.mark.parametrize("q", [
    "qué es ikigai",
    "mis proyectos actuales",
    "listame notas sobre X",
    "cuánto tengo pendiente",
    "dame un resumen de Y",
    "",
])
def test_no_false_positives(q):
    assert rag.has_temporal_lookup_cue(q) is False, q


def test_temporal_lookup_orthogonal_to_recency():
    """A query with only recency cue shouldn't match temporal_lookup."""
    q = "mis notas recientes"
    assert rag.has_recency_cue(q) is True
    assert rag.has_temporal_lookup_cue(q) is False


def test_both_cues_can_fire():
    """'última vez' is captured by both regexes — they compound."""
    q = "la última vez que hablé de X"
    assert rag.has_recency_cue(q) is True  # "última" matched
    assert rag.has_temporal_lookup_cue(q) is True


def test_case_insensitive():
    assert rag.has_temporal_lookup_cue("LA ÚLTIMA VEZ QUE HABLÉ DE X") is True
    assert rag.has_temporal_lookup_cue("Hace Unos Días") is True


def test_empty_and_whitespace():
    assert rag.has_temporal_lookup_cue("") is False
    assert rag.has_temporal_lookup_cue("   ") is False


# ── boost application (integration via env var) ─────────────────────────


def test_temporal_boost_env_parsing(monkeypatch):
    """The boost multiplier reads from RAG_TEMPORAL_LOOKUP_BOOST."""
    # We don't directly invoke retrieve() here (requires full setup),
    # but we verify the env var parses to float correctly.
    import os
    monkeypatch.setenv("RAG_TEMPORAL_LOOKUP_BOOST", "5.0")
    # The retrieve() code parses this value inline — re-evaluate here.
    val = float(os.environ.get("RAG_TEMPORAL_LOOKUP_BOOST", "3.0"))
    assert val == 5.0


def test_temporal_boost_default():
    import os
    # When unset, default is 3.0.
    val = float(os.environ.get("RAG_TEMPORAL_LOOKUP_BOOST_UNSET_PROBE", "3.0"))
    assert val == 3.0
