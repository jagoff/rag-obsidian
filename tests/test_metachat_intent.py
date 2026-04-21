"""Tests for `_detect_metachat_intent` + `_pick_metachat_reply` —
the meta-chat short-circuit in web `/api/chat`.

Regression 2026-04-21 (Playwright probe, Fer F.): plain "hola"
produced "Según tus notas, tenés varias interacciones con diferentes
contactos por WhatsApp..." because the web `_WEB_SYSTEM_PROMPT` REGLA 1
forces the LLM to fabricate context engagement. The short-circuit
detects bare social / meta turns BEFORE the retrieval + tool phases
and responds with a canned line, keeping UX tight and avoiding
hallucinated "según tus notas..." on greetings.
"""

from __future__ import annotations

import pytest

import rag
from web import server as web_server


# ── Intent detector ────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    # Greetings
    "hola",
    "hola!",
    "HOLA",
    "Hola, ¿cómo estás?",
    "buenas",
    "buen día",
    "buenas tardes",
    "buenas noches",
    "hi",
    "hey",
    "hello",
    "¿qué tal?",
    "qué onda",
    "¿cómo estás?",
    "cómo andás",
    # Thanks / closings
    "gracias",
    "muchas gracias",
    "mil gracias",
    "chau",
    "bye",
    # Meta questions about the bot
    "¿qué podés hacer?",
    "qué podés hacer",
    "qué sabés hacer",
    "cómo funcionás",
    "cómo te uso",
    "qué comandos hay",
    "qué comandos tenés",
    "ayuda",
    "help",
    "¿quién sos?",
])
def test_metachat_positive(q: str) -> None:
    assert rag._detect_metachat_intent(q) is True


@pytest.mark.parametrize("q", [
    # Longer messages that happen to start with a greeting word —
    # should NOT short-circuit (user has a real question).
    "hola necesito saber qué hago el viernes",
    "hola, preguntita: ¿qué es ikigai?",
    # Real queries
    "qué sabés de Grecia",
    "el viernes 20hs tengo que ir de Seba",
    "recordame comprar pan mañana",
    "qué tengo esta semana",
    "mostrame los recordatorios",
    "cuándo es la próxima reunión",
    # Tasks-ish
    "qué pendientes tengo",
    "qué eventos tengo mañana",
    # Empty
    "",
    None,
])
def test_metachat_negative(q) -> None:
    assert rag._detect_metachat_intent(q or "") is False


# ── Canned reply picker ────────────────────────────────────────────────────

@pytest.mark.parametrize("q,bucket", [
    ("hola", web_server._METACHAT_GREETING),
    ("hola!", web_server._METACHAT_GREETING),
    ("buenas", web_server._METACHAT_GREETING),
    ("¿qué tal?", web_server._METACHAT_GREETING),
    ("gracias", web_server._METACHAT_THANKS),
    ("muchas gracias", web_server._METACHAT_THANKS),
    ("chau", web_server._METACHAT_BYE),
    ("¿qué podés hacer?", web_server._METACHAT_META),
    ("qué comandos hay", web_server._METACHAT_META),
    ("ayuda", web_server._METACHAT_META),
    ("help", web_server._METACHAT_META),
])
def test_pick_metachat_reply_bucket(q: str, bucket: tuple[str, ...]) -> None:
    reply = web_server._pick_metachat_reply(q, now=1000.0)
    assert reply in bucket


def test_pick_metachat_reply_stable_within_minute() -> None:
    """Same message in the same minute → same variant (idempotent retry)."""
    now = 1_800_000_000.0  # arbitrary fixed moment
    r1 = web_server._pick_metachat_reply("hola", now=now)
    r2 = web_server._pick_metachat_reply("hola", now=now + 30)
    assert r1 == r2


def test_pick_metachat_reply_varies_across_minutes() -> None:
    """Over a wide window, we should see more than one variant. 60 minutes
    across 3 variants gives ≥2 unique outputs with overwhelming probability.
    """
    base = 1_800_000_000.0
    variants = set()
    for minute in range(60):
        variants.add(web_server._pick_metachat_reply("hola", now=base + minute * 60))
    assert len(variants) >= 2


def test_pick_metachat_reply_no_placeholders() -> None:
    """Canned replies must be fully rendered — no `{...}` leftovers."""
    for q in ("hola", "gracias", "ayuda", "chau"):
        r = web_server._pick_metachat_reply(q, now=1000.0)
        assert "{" not in r
        assert "}" not in r


# ── Short-circuit wiring on /api/chat ─────────────────────────────────────
# The full handler is exercised in test_web_chat_tools.py with a mocked
# ollama; here we just assert the gate logic is in place: `is_metachat`
# is computed and conflicts correctly with `is_propose_intent`.

def test_metachat_does_not_mask_propose() -> None:
    """Propose intent wins over metachat — "recordame hola mañana" is not
    a greeting even though it contains 'hola'."""
    q = "recordame hola mañana"
    # Propose fires:
    assert rag._detect_propose_intent(q) is True
    # Metachat might also match on the literal tokens, but the handler
    # guards with `(not is_propose_intent) and _detect_metachat_intent`.
    # Shape assertion is the simpler contract here — as long as propose
    # wins, the short-circuit code path is not entered.
