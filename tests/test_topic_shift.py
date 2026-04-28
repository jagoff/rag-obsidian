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

import pytest

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
    # 1.0·0.0 + 0.0·1.0 = 0 → cosine = 0.000 (< TOPIC_SHIFT_COSINE)
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
    # Justo por encima del threshold → no shift. Uso `target = threshold +
    # 0.05` para evitar el ruido del 1e-8 que `cosine_sim` agrega al
    # denominador y que empuja el equal-case por debajo del strict `<` gate.
    # Independent del valor exacto de TOPIC_SHIFT_COSINE (cambió 0.40 → 0.32
    # en P3 — 2026-04-28).
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


# 2026-04-28 P3: extensiones rioplatenses al _TOPIC_SHIFT_FOLLOWUP_RE.
# Son referencias claras al turn anterior que ANTES caían como cosine
# shift porque el vocabulario es vacío en términos del turn previo. Cada
# test verifica que el regex anaphoric agarra la query — sin tener que
# llegar al cosine gate. Si alguno empieza a fallar, alguien tocó el
# regex sin sincronizar — usar este test para regenerar la lista.
@pytest.mark.parametrize(
    "current_q,description",
    [
        # contame / contá: rioplatense, "cuéntame más"
        ("contame más sobre el ranker", "contame más"),
        ("contame otra cosa", "contame otra"),
        ("contá más detalles", "contá más"),
        # ordinal references al turn anterior
        ("del primero que mencionaste", "del primero"),
        ("del último que dijiste", "del último"),
        ("del anterior te decía algo", "del anterior"),
        ("del otro punto", "del otro"),
        ("el primero que mencionaste cuál era", "el primero que"),
        # alguna otra X / algo más
        ("alguna otra opción me podés dar?", "alguna otra opción"),
        ("alguna otra forma de hacerlo?", "alguna otra forma"),
        ("algo más sobre eso?", "algo más sobre"),
        # explicalo / explicame variantes
        ("explicalo otra vez", "explicalo"),
        ("explicame de vuelta", "explicame"),
        ("explicala mejor por favor", "explicala"),
        # dale más / dale otra
        ("dale más detalles", "dale más"),
        ("dale otra vuelta", "dale otra"),
        # y vos qué pensás / dijiste (continuación dialógica)
        ("y vos qué pensás de eso", "y vos qué"),
        ("y tú qué dirías", "y tú qué"),
        # justo eso
        ("justo eso quería preguntar", "justo eso"),
        ("justo por eso te pregunto", "justo por eso"),
        # otra vez X
        ("otra vez eso?", "otra vez eso"),
        ("otra vez el primero?", "otra vez el"),
    ],
)
def test_p3_rioplatense_followups_are_anaphoric(monkeypatch, current_q, description):
    """Cada query nueva del P3 debe ser detectada como anafórica → keep
    history. El cosine gate no debe correr (embed nunca se llama).
    """
    embed_called = {"n": 0}

    def fake_embed(_ts):
        embed_called["n"] += 1
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(rag, "embed", fake_embed)
    shifted, reason, _cos = rag.detect_topic_shift(
        current_q,
        _hist("turno anterior cualquiera con tema distinto"),
        person_fired=False,
    )
    assert shifted is False, f"P3 follow-up '{description}' debería keep history"
    assert reason == "anaphoric", f"P3 '{description}' detectado como {reason!r}"
    assert embed_called["n"] == 0, "anaphoric gate no debería invocar embed"


def test_p3_unrelated_query_is_not_anaphoric(monkeypatch):
    """Sanity: queries autónomas que NO son follow-ups deben caer al
    cosine gate (no quedar atrapadas falsamente por el regex extendido).
    """
    monkeypatch.setattr(rag, "embed", lambda _ts: [[1.0, 0.0], [0.0, 1.0]])
    shifted, reason, _cos = rag.detect_topic_shift(
        "qué tengo agendado para el jueves",
        _hist("contame del primer punto"),
        person_fired=False,
    )
    # Cosine 0 → shift, reason cosine=0.000 (NO anaphoric).
    assert shifted is True
    assert reason.startswith("cosine=")
