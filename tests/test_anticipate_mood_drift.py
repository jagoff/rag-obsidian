"""Tests para `rag_anticipate.signals.mood_drift` — push proactivo
cuando el user lleva ≥3 días con mood bajo sostenido.

Cubre:
1. Feature off (RAG_MOOD_ENABLED no seteado) → []
2. recent_drift devuelve drifting=False → []
3. drifting=True con 3 días consecutivos → 1 candidate con score ≥ 0.55
4. Score escala con n_consecutive (3=0.55, 5=0.75, 7+=1.0).
5. dedup_key estable basado en start_date de la racha — misma racha
   → mismo key, racha distinta → key distinto.
6. Mensaje template no contiene score literal, "mood", ni
   therapy-speak prohibido ("respirá", "tomate un día").
7. snooze_hours = 168 (7 días).
8. Excepción interna en rag.mood → silent-fail [] (no rompe el orchestrator).
"""

from __future__ import annotations

from datetime import datetime

import pytest

import rag  # noqa: F401 — needed para que AnticipatoryCandidate esté disponible
from rag_anticipate.signals.mood_drift import mood_drift_signal


def _drift_off():
    return {"drifting": False, "n_consecutive": 0, "dates": [],
            "avg_score": 0.0, "reason": "no_streak"}


def _drift_on(n: int, start_date: str = "2026-04-28",
              avg_score: float = -0.55):
    dates = [start_date]
    return {"drifting": True, "n_consecutive": n, "dates": dates * n,
            "avg_score": avg_score, "reason": None}


@pytest.fixture
def mood_enabled(monkeypatch):
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")


@pytest.fixture
def fake_drift(monkeypatch):
    """Helper para inyectar un valor de retorno de recent_drift."""
    state = {"value": _drift_off()}

    def _set(value):
        state["value"] = value

    def _fake(*args, **kwargs):
        return state["value"]

    from rag import mood
    monkeypatch.setattr(mood, "recent_drift", _fake)
    return _set


# ── Feature gate ───────────────────────────────────────────────────────────


def test_feature_off_returns_empty(monkeypatch):
    """Sin RAG_MOOD_ENABLED, la signal devuelve [] sin tocar nada."""
    monkeypatch.delenv("RAG_MOOD_ENABLED", raising=False)
    result = mood_drift_signal(datetime.now())
    assert result == []


def test_no_drift_returns_empty(mood_enabled, fake_drift):
    """recent_drift devuelve drifting=False → no signal."""
    fake_drift(_drift_off())
    result = mood_drift_signal(datetime.now())
    assert result == []


# ── Drift activo emite candidate ───────────────────────────────────────────


def test_drift_3_days_emits_candidate(mood_enabled, fake_drift):
    """Racha de 3 días terminando hoy → 1 candidate con score ≥ 0.55."""
    fake_drift(_drift_on(n=3, start_date="2026-04-28", avg_score=-0.55))
    result = mood_drift_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-mood-drift"
    assert c.score >= 0.55
    assert c.snooze_hours == 168


def test_drift_5_days_higher_score(mood_enabled, fake_drift):
    fake_drift(_drift_on(n=5, start_date="2026-04-26", avg_score=-0.6))
    result = mood_drift_signal(datetime.now())
    assert len(result) == 1
    c = result[0]
    # 0.55 + 0.10 * (5 - 3) = 0.75
    assert c.score == pytest.approx(0.75, abs=0.01)


def test_drift_7plus_days_capped_at_1(mood_enabled, fake_drift):
    fake_drift(_drift_on(n=10, start_date="2026-04-21", avg_score=-0.7))
    result = mood_drift_signal(datetime.now())
    assert len(result) == 1
    assert result[0].score == 1.0


# ── dedup_key + reason ─────────────────────────────────────────────────────


def test_dedup_key_stable_within_same_streak(mood_enabled, fake_drift):
    """Misma racha (mismo start_date) en 2 calls → mismo dedup_key."""
    fake_drift(_drift_on(n=3, start_date="2026-04-28", avg_score=-0.5))
    r1 = mood_drift_signal(datetime.now())
    r2 = mood_drift_signal(datetime.now())
    assert r1[0].dedup_key == r2[0].dedup_key
    assert "2026-04-28" in r1[0].dedup_key


def test_dedup_key_different_for_distinct_streaks(mood_enabled, fake_drift):
    """Dos rachas distintas (diferente start_date) → dedup_keys
    distintos. Importante para que después de un quiebre el push
    pueda volver a dispararse en lugar de quedar ahogado por el
    snooze de la racha anterior."""
    fake_drift(_drift_on(n=3, start_date="2026-04-10", avg_score=-0.5))
    r1 = mood_drift_signal(datetime.now())
    fake_drift(_drift_on(n=3, start_date="2026-04-25", avg_score=-0.6))
    r2 = mood_drift_signal(datetime.now())
    assert r1[0].dedup_key != r2[0].dedup_key


def test_reason_includes_n_consecutive(mood_enabled, fake_drift):
    """`reason` (mostrado en --explain) incluye n_consecutive +
    avg_score + start_date — diagnostic interno, NO se muestra al user."""
    fake_drift(_drift_on(n=4, start_date="2026-04-27", avg_score=-0.55))
    result = mood_drift_signal(datetime.now())
    assert "n_consecutive=4" in result[0].reason
    assert "2026-04-27" in result[0].reason


# ── Mensaje no verbaliza mood ──────────────────────────────────────────────


def test_message_does_not_verbalize_mood(mood_enabled, fake_drift):
    """Crítico — el mensaje NO contiene labels de mood en prosa
    (diagnóstico al user), score literal, ni therapy-speak.

    Excepción permitida: `rag mood show` es un command CLI que el user
    puede invocar para ver señales él mismo si quiere transparencia.
    Eso es opt-in del user, no paternalismo. Removemos el backtick-block
    antes de validar prosa."""
    fake_drift(_drift_on(n=4, start_date="2026-04-27", avg_score=-0.65))
    result = mood_drift_signal(datetime.now())
    raw = result[0].message
    # Strip backtick-quoted commands (donde `rag mood show` puede aparecer
    # como sugerencia de transparencia, no como narración).
    import re
    prose = re.sub(r"`[^`]*`", "", raw).lower()

    # Banned terms en la PROSA (verbalización de mood / diagnóstico):
    forbidden = [
        "bajón", "triste", "deprimi", "ánimo", "animo", "mood",
        "noté que", "te ves", "se nota que", "venís cansado",
        "estás mal",
        # Therapy-speak prohibido:
        "tomate un día", "respirá", "respira", "hablalo con",
        "self-care", "auto-cuidado",
    ]
    for term in forbidden:
        assert term not in prose, (
            f"prosa del mensaje contiene '{term}' (prohibido). "
            f"Mensaje completo: {raw}"
        )

    # Tampoco debe filtrar el score numérico, ni siquiera en backticks.
    assert "-0.65" not in raw
    assert "score" not in prose

    # Debe tener tono calmo + offer concreto (recortar plan).
    assert "recortar" in prose or "1-2 cosas" in prose


# ── Silent-fail ────────────────────────────────────────────────────────────


def test_silent_fail_when_recent_drift_raises(mood_enabled, monkeypatch):
    """Si recent_drift tira (DB lock, etc.), signal devuelve [] en
    lugar de propagar — el orchestrator no debería romperse por una
    signal individual."""
    from rag import mood
    def _broken(*args, **kwargs):
        raise RuntimeError("DB locked")
    monkeypatch.setattr(mood, "recent_drift", _broken)
    result = mood_drift_signal(datetime.now())
    assert result == []


# ── Wired into anticipate orchestrator ─────────────────────────────────────


def test_signal_registered_in_anticipate():
    """El signal está auto-registrado en rag_anticipate.SIGNALS via
    el decorator. `anticipate_run_impl` lo va a llamar junto con los
    otros signals."""
    import rag_anticipate
    names = [name for name, _fn in rag_anticipate.SIGNALS]
    assert "mood-drift" in names
