"""Unit tests for `rag.detect_topic_shift`.

Reported bug 2026-04-20 (session `web:b03ec059db32`): turno T-1 "cual es mi
password de avature?" → turno T "busca informacion sobre mi mama" producía
respuesta mezclada ("tu mamá Monica Ferrari... no hay info sobre su
contraseña de Avature") porque los 6 últimos mensajes de history seguían
vivos en el prompt del chat LLM. `detect_topic_shift` decide si descartar
history para el turno actual. Estos tests cubren los tres gates (short,
anaphoric, person, cosine) sin tocar ollama ni el vault — `embed` se
monkeypatchea con vectores deterministas.
"""
from __future__ import annotations

import rag


def _hist(last_user_q: str) -> list[dict]:
    return [
        {"role": "user", "content": last_user_q},
        {"role": "assistant", "content": "prior answer"},
    ]


def test_empty_history_is_not_a_shift():
    shifted, reason, _cos = rag.detect_topic_shift("cualquier cosa", [], person_fired=False)
    assert shifted is False
    assert reason == "no-history"


def test_short_query_keeps_history_even_if_topic_differs(monkeypatch):
    # 1-2 token queries son ellipsis del turno anterior ("y?", "ella?").
    # Aún si la cosine daría shift, el gate (1) corta antes.
    monkeypatch.setattr(rag, "embed", lambda _ts: [[1.0, 0.0], [0.0, 1.0]])
    shifted, reason, _cos = rag.detect_topic_shift(
        "más?", _hist("hablame de Grecia"), person_fired=False,
    )
    assert shifted is False
    assert reason == "short"


def test_anaphoric_cue_keeps_history(monkeypatch):
    # "profundiza sobre eso" tiene marker "eso" + "profundiza" → anaphoric.
    # Cosine gate no se ejecuta (embed nunca se llama).
    called = {"embed": 0}

    def fake_embed(_ts):
        called["embed"] += 1
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(rag, "embed", fake_embed)
    shifted, reason, _cos = rag.detect_topic_shift(
        "profundiza sobre eso",
        _hist("cual es mi password?"),
        person_fired=False,
    )
    assert shifted is False
    assert reason == "anaphoric"
    assert called["embed"] == 0


def test_como_lo_X_is_anaphoric():
    # Edge case del reporte: "y como lo desactivo?" sigue a "como activo X?".
    # Cosine empírica 0.298 (bge-m3 separa activo/desactivo) — el gate de
    # cosine SÍ lo marcaría como shift, pero es claramente un follow-up.
    # El regex "c[oó]mo (lo|la|los|las) \w+" lo protege.
    shifted, reason, _cos = rag.detect_topic_shift(
        "y como lo desactivo?",
        _hist("como activo claude peers?"),
        person_fired=False,
    )
    assert shifted is False
    assert reason == "anaphoric"


def test_person_fired_with_different_prior_topic_drops_history(monkeypatch):
    # El caso reportado. build_person_context matcheó "mama" en la query
    # actual. El turno anterior NO menciona a la misma persona. → shift.
    monkeypatch.setattr(rag, "_match_mentions_in_query", lambda q, *a, **k: (
        ["04-Archive/99-obsidian-system/99-Mentions/Mama.md"] if "mama" in q.lower() else []
    ))
    shifted, reason, _cos = rag.detect_topic_shift(
        "busca informacion sobre mi mama",
        _hist("cual es mi password de avature?"),
        person_fired=True,
    )
    assert shifted is True
    assert reason == "person"


def test_person_fired_same_person_keeps_history(monkeypatch):
    # "contame sobre mi mama" → "busca info sobre mi mama": misma persona
    # matcheada en ambos turnos. Es follow-up natural → keep history.
    monkeypatch.setattr(rag, "_match_mentions_in_query", lambda q, *a, **k: (
        ["04-Archive/99-obsidian-system/99-Mentions/Mama.md"] if "mama" in q.lower() else []
    ))
    shifted, reason, _cos = rag.detect_topic_shift(
        "busca informacion sobre mi mama",
        _hist("contame sobre mi mama"),
        person_fired=True,
    )
    assert shifted is False
    assert reason == "same-person"


def test_cosine_low_drops_history(monkeypatch):
    # No anafórico, no person, current y last ortogonales → shift.
    monkeypatch.setattr(rag, "embed", lambda _ts: [[1.0, 0.0], [0.0, 1.0]])
    shifted, reason, _cos = rag.detect_topic_shift(
        "dame info sobre mis proyectos",
        _hist("como activo claude peers?"),
        person_fired=False,
    )
    assert shifted is True
    assert reason.startswith("cosine=")
    # 1.0·0.0 + 0.0·1.0 = 0 → cosine = 0.000 (< 0.40)
    assert "0.000" in reason


def test_cosine_high_keeps_history(monkeypatch):
    # Vectores casi paralelos → cosine > threshold → no shift.
    monkeypatch.setattr(rag, "embed", lambda _ts: [[1.0, 0.0], [0.95, 0.05]])
    shifted, reason, _cos = rag.detect_topic_shift(
        "que reflexiones tengo sobre coaching?",
        _hist("notas de coaching"),
        person_fired=False,
    )
    assert shifted is False
    assert reason.startswith("cosine=")


def test_embed_failure_is_fail_safe(monkeypatch):
    # Si embed levanta excepción (ollama caído), preferimos mantener
    # history (fail-safe para follow-ups) antes que dropearla.
    def boom(_ts):
        raise ConnectionError("ollama unreachable")

    monkeypatch.setattr(rag, "embed", boom)
    shifted, reason, _cos = rag.detect_topic_shift(
        "algo completamente distinto",
        _hist("otro tema totalmente"),
        person_fired=False,
    )
    assert shifted is False
    assert reason.startswith("embed-failed:ConnectionError")


def test_threshold_boundary(monkeypatch):
    # Justo por encima del threshold → no shift. Uso 0.45 (> 0.40) para evitar
    # el ruido del 1e-8 que `cosine_sim` agrega al denominador y que empuja
    # el equal-case por debajo del strict `<` gate.
    import math
    target = rag.TOPIC_SHIFT_COSINE + 0.05
    v = math.sqrt(1 - target ** 2)
    monkeypatch.setattr(rag, "embed", lambda _ts: [
        [1.0, 0.0], [target, v],
    ])
    shifted, reason, _cos = rag.detect_topic_shift(
        "a query for boundary",
        _hist("another query"),
        person_fired=False,
    )
    assert shifted is False
    assert reason.startswith("cosine=")
