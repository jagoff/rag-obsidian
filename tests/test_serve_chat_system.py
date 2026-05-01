"""Smoke tests for _SERVE_CHAT_SYSTEM — the system prompt used by
`rag serve`'s `/chat` endpoint (bare-text path from the WhatsApp listener).

Regression: on 2026-04-21 the user asked "Que sabes de Grecia" via WhatsApp.
The listener routed bare text to /chat (LLM-only, no retrieval). The prior
prompt only said "no inventes información sobre las notas del usuario",
which left the door open to world-knowledge prose. command-r produced a
Wikipedia-style paragraph about Greece with zero vault citations —
indistinguishable for the user from a grounded answer.

The fix has two layers:
  1. `detectFactualIntent` in whatsapp-listener/listener.ts forces factual
     questions through /query (RAG) instead of /chat.
  2. This prompt tightened so even if the router misses a case, the LLM
     refuses to produce world-knowledge content and redirects to /search.

These tests assert structural properties of the prompt, not LLM behavior
(which would require a live ollama). They catch accidental deletion /
dilution of the rules during future refactors.
"""

import rag


def test_serve_chat_system_declares_non_general_knowledge():
    p = rag._SERVE_CHAT_SYSTEM
    assert "NO sos" in p and "conocimiento general" in p, (
        "prompt must declare the bot is not a general knowledge model"
    )


def test_serve_chat_system_forbids_wikipedia_mode():
    p = rag._SERVE_CHAT_SYSTEM.lower()
    # Explicit list of content categories the LLM must refuse.
    for kw in ("biograf", "definici", "geograf", "ciencia", "histori"):
        assert kw in p, f"prompt must forbid {kw!r} content"
    assert "wikipedia" in p, "prompt must explicitly say it's not Wikipedia"


def test_serve_chat_system_redirects_factual_to_search():
    p = rag._SERVE_CHAT_SYSTEM
    # The redirect template must be verbatim in the prompt so command-r
    # can echo it rather than improvising.
    assert "/search" in p
    assert "Para eso buscá en tus notas" in p


def test_serve_chat_system_preserves_meta_chat():
    # Greetings and meta questions ("qué podés hacer") should remain
    # answerable — otherwise the bot becomes user-hostile para "hola".
    #
    # 2026-04-27: el prompt se reescribió para imitar el TONO de Fer
    # (rioplatense casual) — se sacó la mención explícita a `/help`
    # porque esa frase pertenece al registro corporate-chatbot que
    # explícitamente prohibimos en REGLA 4 ("¿en qué te puedo ayudar?").
    # En su lugar el prompt enumera saludos como "hola", "gracias", "che",
    # "dale" — el bot responde con esos atómicos en vez de redirigir
    # genéricamente a /help. Ver `serve_meta.v1.md` REGLA 1 + ejemplos.
    p = rag._SERVE_CHAT_SYSTEM.lower()
    # El prompt debe enumerar al menos uno de los saludos comunes para
    # que el bot reconozca que SÍ contesta "hola" / "gracias" en vez de
    # redirigir a `/search` (que es el path de queries factuales).
    assert any(greet in p for greet in ("hola", "gracias", "che", "dale", "qué onda")), (
        "el prompt debe enumerar al menos un saludo común para que "
        "meta-chat NO se redirija a /search"
    )


def test_serve_chat_system_lists_factual_question_shapes():
    # The LLM needs to recognize the shapes that trigger redirect. If any
    # of these is missing we risk the model answering "qué sabés de X"
    # because the prompt only mentioned "qué es X".
    p = rag._SERVE_CHAT_SYSTEM
    for shape in (
        "qué sabés de",
        "qué es",
        "quién es",
        "cómo funciona",
        "cuándo pasó",
    ):
        assert shape in p, f"prompt must enumerate the {shape!r} pattern"
