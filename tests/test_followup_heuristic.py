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
    return bool(_FOLLOWUP_CUES.search(q))


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
    # should NOT fire — standalone complete questions
    ("qué dice la nota sobre Grecia", False),
    ("resumime los sprints del mes pasado", False),
    ("cuáles son los objetivos del proyecto obsidian-rag", False),
    ("dame un resumen del sprint actual", False),
    ("tenes notas sobre arquitectura hexagonal", False),
    # long question with common article — must NOT fire
    ("qué notas hay sobre el proyecto de infraestructura", False),
    # empty
    ("", False),
])
def test_looks_like_followup(q, expected):
    assert _looks_like_followup(q) is expected, (
        f"_looks_like_followup({q!r}) = {_looks_like_followup(q)}, expected {expected}"
    )
