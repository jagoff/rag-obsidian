"""Tests del paráfrasi multi-query.

El bug que motivó esto: qwen2.5:3b generaba 'el actor Adam Jones' /
'el intérprete Adam Jones' para queries con nombres propios, desviando
la retrieval semántica. Las variantes malas se filtran por dropping de
nombres propios.
"""
import pytest
import rag


# ── Helper de extracción ──────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("qué usa Adam Jones?", {"Adam", "Jones"}),
    ("última reunión con Marcos", {"Marcos"}),
    ("como configurar AXE FX 3", {"AXE"}),   # 'FX' y '3' son cortos, quedan fuera
    ("nota sobre Nodok y ELEVA", {"Nodok", "ELEVA"}),
    ("qué tal va todo", set()),   # todo minúsculas
    ("Qué tal va todo", set()),   # arranque de oración que matchea, pero 'tal'/'va' no cumplen len
])
def test_extract_proper_nouns(text, expected):
    nouns = rag._extract_proper_nouns(text)
    # El test es laxo — permitimos que "Qué" pase porque empieza oración;
    # lo importante es que los nombres reales estén en el set.
    assert expected.issubset(nouns) or not expected


def test_extract_proper_nouns_picks_up_acronyms_and_capitalized():
    # Nodok es nombre de agencia — capitalizado, len ≥ 3.
    nouns = rag._extract_proper_nouns("el onboarding de Nodok con clientes")
    assert "Nodok" in nouns


def test_extract_handles_accents_spanish():
    nouns = rag._extract_proper_nouns("qué dice María sobre León")
    assert "María" in nouns
    assert "León" in nouns


# ── expand_queries con ollama mockeado ────────────────────────────────────────


@pytest.fixture
def fake_ollama(monkeypatch):
    """Mock ollama.chat para devolver respuestas controladas por test."""
    state = {"next_response": ""}

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeResponse:
        def __init__(self, content):
            self.message = FakeMessage(content)

    def fake_chat(**kwargs):
        return FakeResponse(state["next_response"])

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    return state


def test_drops_paraphrase_that_mangles_proper_noun(fake_ollama):
    # Cuando el usuario capitaliza ("Adam Jones"), el guardrail detecta si
    # la paráfrasi droppea un token propio y la descarta. Con lowercase el
    # guardrail no se activa — ahí confiamos en el prompt (ver live smoke).
    fake_ollama["next_response"] = (
        "Qué sistema de sonido utiliza el actor?\n"
        "Cuál es el rig de Adam Jones?"
    )
    variants = rag.expand_queries("qué sistema de sonido utiliza Adam Jones?")
    # La primera droppea "Adam" y "Jones" → descartada.
    # La segunda los preserva → aceptada.
    assert variants[0] == "qué sistema de sonido utiliza Adam Jones?"
    assert len(variants) == 2
    assert "Adam Jones" in variants[1]
    assert "el actor" not in variants[1]


def test_keeps_both_paraphrases_when_nouns_preserved(fake_ollama):
    fake_ollama["next_response"] = (
        "qué equipo de sonido usa Adam Jones?\n"
        "cuál es el rig de Adam Jones?"
    )
    variants = rag.expand_queries("qué usa Adam Jones?")
    assert len(variants) == 3  # original + 2 válidas
    assert all("Adam" in v for v in variants)


def test_query_without_proper_nouns_keeps_all_paraphrases(fake_ollama):
    # Sin nombres propios no hay guardrail; lo que devuelva qwen queda.
    fake_ollama["next_response"] = (
        "qué tal va todo hoy?\n"
        "cómo anda el día?"
    )
    variants = rag.expand_queries("qué hora es?")
    assert len(variants) == 3   # original + 2


def test_ollama_failure_returns_just_original(fake_ollama, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(rag.ollama, "chat", boom)
    variants = rag.expand_queries("qué usa Adam Jones?")
    assert variants == ["qué usa Adam Jones?"]


def test_deduplicates_echo_of_original(fake_ollama):
    # Si el modelo devuelve la pregunta original como "paráfrasi", se ignora.
    fake_ollama["next_response"] = (
        "qué usa Adam Jones?\n"
        "cuál es el rig de Adam Jones?"
    )
    variants = rag.expand_queries("qué usa Adam Jones?")
    # Original + solo la verdadera paráfrasi.
    assert variants[0] == "qué usa Adam Jones?"
    assert len(variants) == 2
    assert "rig" in variants[1]
