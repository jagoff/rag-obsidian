"""Tests for NLI grounding module (Improvement #1, Fase A standalone).

Scope: split_claims() edge cases + ground_claims_nli() API contract.
NO actual NLI inference yet — that's Fase B.
"""
from __future__ import annotations

import rag


# ────────────────────────────────────────────────────────────────────
# split_claims
# ────────────────────────────────────────────────────────────────────

def test_split_claims_empty():
    assert rag.split_claims("") == []
    assert rag.split_claims("   ") == []


def test_split_claims_single_sentence():
    claims = rag.split_claims("El proyecto comenzó en 2024.")
    assert len(claims) == 1
    assert "proyecto" in claims[0].text
    assert claims[0].is_refusal is False


def test_split_claims_multiple_sentences():
    text = "El proyecto comenzó en 2024. Tiene tres fases. La primera termina en junio."
    claims = rag.split_claims(text)
    assert len(claims) == 3
    assert all(not c.is_refusal for c in claims)


def test_split_claims_refusal_es():
    claims = rag.split_claims("No encontré esto en el vault.")
    assert len(claims) == 1
    assert claims[0].is_refusal is True


def test_split_claims_refusal_en():
    claims = rag.split_claims("I could not find any information.")
    assert len(claims) == 1
    assert claims[0].is_refusal is True


def test_split_claims_not_a_refusal():
    """Normal sentences should NOT match refusal patterns."""
    claims = rag.split_claims("La nota no menciona la fecha exacta del evento.")
    assert len(claims) == 1
    assert claims[0].is_refusal is False


def test_split_claims_preserves_bullet_list():
    """Bullet list = atomic claim."""
    text = "Tareas:\n- Implementar NLI\n- Escribir tests\n- Integrar\n"
    claims = rag.split_claims(text)
    list_claims = [c for c in claims if "Implementar" in c.text and "Escribir" in c.text]
    assert len(list_claims) == 1, f"List should be one atomic claim, got {[c.text for c in claims]}"


def test_split_claims_preserves_numbered_list():
    text = "Pasos:\n1. Primero\n2. Segundo\n3. Tercero\n"
    claims = rag.split_claims(text)
    list_claims = [c for c in claims if "Primero" in c.text and "Segundo" in c.text]
    assert len(list_claims) == 1


def test_split_claims_preserves_markdown_table():
    text = "Eventos:\n| Fecha | Evento |\n|---|---|\n| 2024-05-01 | Kickoff |\n"
    claims = rag.split_claims(text)
    table_claims = [c for c in claims if "Fecha" in c.text and "Evento" in c.text]
    assert len(table_claims) == 1


def test_split_claims_preserves_code_fence():
    text = "Usá este snippet:\n```python\nprint('hello')\n```\nY listo."
    claims = rag.split_claims(text)
    code_claims = [c for c in claims if "print('hello')" in c.text]
    assert len(code_claims) == 1
    assert "```" in code_claims[0].text


def test_split_claims_drops_tiny_fragments():
    """Fragments < 8 chars should be dropped (stray periods, etc.)."""
    text = "Esta es una oración completa. Ok. Y otra larga con contenido."
    claims = rag.split_claims(text)
    assert all(len(c.text) >= 8 for c in claims)


# ────────────────────────────────────────────────────────────────────
# ground_claims_nli (Fase A stub — all neutral)
# ────────────────────────────────────────────────────────────────────

def test_ground_claims_empty_inputs():
    assert rag.ground_claims_nli([], [], []) is None
    assert rag.ground_claims_nli([rag.Claim(text="x")], [], []) is None
    assert rag.ground_claims_nli([], ["doc"], [{"file": "f.md"}]) is None


def test_ground_claims_max_claims_gate():
    """Safety gate: >20 claims → None (skip NLI)."""
    claims = [rag.Claim(text=f"Claim number {i} is long enough") for i in range(25)]
    result = rag.ground_claims_nli(claims, ["doc"], [{"file": "f.md"}], max_claims=20)
    assert result is None


def test_ground_claims_fase_a_returns_neutral():
    """Fase A stub: all claims marked neutral (safe default)."""
    claims = [rag.Claim(text="El proyecto termina en junio.")]
    docs = ["El proyecto termina el 30 de junio de 2024."]
    metas = [{"file": "nota.md", "note": "nota"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result is not None
    assert result.claims_total == 1
    assert result.claims_neutral == 1
    assert result.claims_supported == 0
    assert result.claims_contradicted == 0
    assert result.claims[0].verdict == "neutral"


def test_ground_claims_refusal_marked_neutral():
    """Refusals → neutral (no NLI needed, skipped)."""
    claims = [rag.Claim(text="No encontré esto en el vault.", is_refusal=True)]
    docs = ["random content"]
    metas = [{"file": "f.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result is not None
    assert result.claims[0].verdict == "neutral"


def test_ground_claims_result_shape():
    """GroundingResult has expected fields + aggregate counts."""
    claims = [
        rag.Claim(text="Claim one is long enough"),
        rag.Claim(text="Claim two is also long enough"),
    ]
    docs = ["evidence chunk"]
    metas = [{"file": "f.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result is not None
    assert isinstance(result, rag.GroundingResult)
    assert result.claims_total == 2
    assert result.claims_total == len(result.claims)
    assert (result.claims_supported + result.claims_contradicted
            + result.claims_neutral == result.claims_total)


def test_claim_dataclass_defaults():
    c = rag.Claim(text="hello world")
    assert c.start_char == 0
    assert c.end_char == 0
    assert c.is_refusal is False


def test_claim_grounding_dataclass_defaults():
    cg = rag.ClaimGrounding(text="hello", verdict="neutral")
    assert cg.evidence_chunk_id is None
    assert cg.evidence_span is None
    assert cg.score == 0.0


def test_grounding_result_empty_default():
    gr = rag.GroundingResult()
    assert gr.claims == []
    assert gr.claims_total == 0
    assert gr.nli_ms == 0


def test_is_refusal_variants():
    """Multiple refusal forms recognized."""
    assert rag._is_refusal("No encontré esto")
    assert rag._is_refusal("no encontré nada")
    assert rag._is_refusal("No hay información")
    assert rag._is_refusal("I don't find")
    assert rag._is_refusal("I could not find")
    assert not rag._is_refusal("El documento dice que no")
    assert not rag._is_refusal("No es importante")


def test_is_refusal_recognizes_all_4_canonical_phrases():
    """Audit 2026-04-25 R2-6 #1: las 4 frases de refusal usadas en los
    prompts de `rag/prompts/intents/` deben todas ser detectadas como
    refusal — sino el cache poisoning detector y NLI grounding las
    tratan como contenido factual y se cachean / score-an mal.

    Pre-fix: las frases de comparison/synthesis ('No hay suficientes
    fuentes...') NO matcheaban porque el regex era
    `^no hay\\s+(?:ningún|información)\\b` y faltaba `suficientes`."""
    canonical_refusals = [
        # 1. strict, web, chat, system_rules
        "No tengo esa información en tus notas.",
        # 2. lookup
        "No encontré esto en el vault.",
        # 3. synthesis
        "No hay suficientes fuentes en el vault para sintetizar esto.",
        # 4. comparison
        "No hay suficientes fuentes en el vault para comparar esto.",
    ]
    for phrase in canonical_refusals:
        assert rag._is_refusal(phrase), (
            f"refusal canónico no detectado: {phrase!r} — chequear "
            f"_REFUSAL_PATTERNS en rag/__init__.py"
        )
