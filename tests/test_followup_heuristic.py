"""Unit tests for _looks_like_followup heuristic (web/server.py).

The function is a pure regex classifier with no heavy deps. We re-declare
the minimal regex objects here to avoid importing the full server module
(which pulls in sqlite-vec/ollama). Keep in sync with web/server.py if the
patterns change.
"""
from __future__ import annotations

import re

import pytest

# ── replicated from web/server.py ────────────────────────────────────────────

_FOLLOWUP_CUES = re.compile(
    r"\b(eso|esa|ese|esto|esta|este|aquello|ella|[eé]l\b(?!\s+[A-ZÁÉÍÓÚ])|"
    r"ahí|ahi|allí|alli|allá|alla|"
    r"it\b|this\b|that\b|he\b|she\b|"
    r"y\s+(de|sobre|con|para|en)|"
    r"profundizá|profundiza|ampliá|amplia|seguí|segui|continuá|continua|"
    r"más\s+(sobre|de|al\s+respecto)|mas\s+(sobre|de|al\s+respecto))\b",
    re.IGNORECASE,
)

# ── extended patterns added 2026-04-26 (keep in sync with web/server.py) ────
_FOLLOWUP_CUES_EXT_LEADING_Y = re.compile(
    r"^y\s+(el|la|los|las|qué|que|cómo|como|cu[áa]l|otra|otro|otras|otros)\b",
    re.IGNORECASE,
)
_FOLLOWUP_CUES_EXT_LEADING_VERB_DEFART = re.compile(
    r"^(listame|dame|mostrame|contame|explicame|detallame|decime|enumerame|"
    r"resume|resumi|resumí)\s+(el|la|los|las)\s+\w+",
    re.IGNORECASE,
)
_FOLLOWUP_CUES_EXT_TENGO_ALGUN = re.compile(
    r"^tengo\s+alg[uú]n\b|^qu[ée]\s+\w+\s+(tengo|us[oé]|tenía|tuve)\b",
    re.IGNORECASE,
)

_LEADING_PRONOUN_RE = re.compile(
    r"^(eso|esa|ese|esto|esta|este|ella|[eé]l|it|this|that|he|she)\b",
    re.IGNORECASE,
)

_SHORT_INTERROGATIVE_RE = re.compile(
    r"^(qué|que|cuál|cual|cómo|como|por\s+qué|por\s+que|"
    r"what|how|which|why)\b",
    re.IGNORECASE,
)


def _looks_like_followup(q: str) -> bool:
    if not q:
        return False
    words = q.split()
    if len(words) <= 2:
        return True
    stripped = q.strip().lstrip("¿").strip()
    if _LEADING_PRONOUN_RE.match(stripped):
        return True
    if len(words) <= 5 and _SHORT_INTERROGATIVE_RE.match(stripped):
        return True
    if _FOLLOWUP_CUES.search(q):
        return True
    if _FOLLOWUP_CUES_EXT_LEADING_Y.match(stripped):
        return True
    if _FOLLOWUP_CUES_EXT_LEADING_VERB_DEFART.match(stripped):
        return True
    if _FOLLOWUP_CUES_EXT_TENGO_ALGUN.match(stripped):
        return True
    return False


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("q,expected", [
    # (a) ≤2 words — always a followup
    ("y eso?", True),
    ("más?", True),
    ("cuál?", True),
    # (b) leading pronoun
    ("eso que dijiste antes", True),
    ("ella cómo funciona?", True),
    ("esto tiene relación con el sprint?", True),
    # (c) short interrogative ≤5 words
    ("qué hay de eso?", True),
    ("cómo se relaciona?", True),
    # (d) explicit anaphora cue mid-sentence
    ("contame más sobre eso", True),
    ("profundizá en ese tema", True),
    ("seguí con lo anterior", True),
    # (e) leading "y + article/wh-word" — added 2026-04-26 to close gap on
    # elliptical chain continuations from queries.yaml golden set:
    # "y el icbc" (after "info banco santander"), "y la retrospectiva del
    # barco" (after "objetivos SMART"), "y qué pendientes tengo del sistema
    # RAG", "y cómo las indexás", "y las versiones de la herrumbre".
    ("y el icbc", True),
    ("y la retrospectiva del barco", True),
    ("y qué pendientes tengo del sistema RAG", True),
    ("y cómo las indexás", True),
    ("y las versiones de la herrumbre que tengo", True),
    ("y el otro curso que tenía", True),
    ("y qué equipo de guitarra tengo", True),
    # (f) leading verb + definite article — reproduces the Visa case from
    # session web:cee69e81829c on 2026-04-26: after "Cuanto devo a la visa?"
    # the user asked "listame los gastos en pesos" expecting Visa context.
    ("listame los gastos en pesos", True),
    ("mostrame las cuentas pendientes", True),
    ("detallame los pagos del mes", True),
    ("contame el resumen del proyecto", True),
    # (g) "tengo algún" / "qué <noun> tengo|usé|tenía"
    ("tengo algún workshop agendado sobre RAG", True),
    ("qué herramientas uso para grabar ese tipo de charla", True),
    # should NOT fire — standalone complete questions
    ("qué dice la nota sobre Grecia", False),
    ("resumime los sprints del mes pasado", False),
    ("cuáles son los objetivos del proyecto obsidian-rag", False),
    ("dame un resumen del sprint actual", False),
    ("tenes notas sobre arquitectura hexagonal", False),
    # long question with common article — must NOT fire (no leading "y")
    ("qué notas hay sobre el proyecto de infraestructura", False),
    # mid-sentence "y + article" — must NOT fire (only LEADING fires; this
    # is a self-contained question with a coordinating conjunction, not a
    # follow-up).
    ("notas sobre el proyecto y el otro tema", False),
    # leading verb but NOT followed by definite article + noun — must NOT
    # fire (just generic imperative without ellipsis).
    ("dame un resumen rápido", False),
    ("listame algunos ejemplos", False),
    # empty
    ("", False),
])
def test_looks_like_followup(q, expected):
    assert _looks_like_followup(q) is expected, (
        f"_looks_like_followup({q!r}) = {_looks_like_followup(q)}, expected {expected}"
    )
