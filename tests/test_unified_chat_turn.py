"""Tests del pipeline unificado `run_chat_turn(req) -> ChatTurnResult` (#1).

Contexto: pre-fix había dos implementaciones paralelas de chat:
  - CLI `chat()` en rag.py (~3000 líneas del loop interactivo)
  - Web `/api/chat` en web/server.py (~900 líneas del endpoint SSE)

Ambos hacen lo mismo al core del turn (classify → retrieve → LLM →
post-process → log) pero con implementaciones paralelas que divergen.
Cuando hoy tuvimos que agregar `intent` al logueo, tocamos 4 call sites
distintos + 50% chance de olvidar alguno.

Refactor: el pipeline core vive en `run_chat_turn(req: ChatTurnRequest) ->
ChatTurnResult`. CLI y web llaman a esta función y solo manejan su I/O
(terminal con Rich vs SSE). El feature flag `RAG_UNIFIED_CHAT=1` gatea
el rollout — los tests del legacy path siguen pasando aunque el nuevo
pipeline quede en standby.

Scope contenido:
  - `ChatTurnRequest` / `ChatTurnResult` como dataclasses (shape
    contractual).
  - `run_chat_turn()` encapsula: classify_intent, multi_retrieve,
    context build, system_prompt_with_version, LLM streaming accumulated,
    run_parallel_post_process.
  - NO mueve slash commands, NO mueve SSE protocol, NO mueve feedback
    interactivo — eso queda en CLI/web.
  - Fuera de scope: metachat short-circuit, propose-intent, low-conf
    bypass (siguen en el loop del caller, invocados ANTES de
    run_chat_turn).

Tests son de shape + unitarios — no disparan LLM real (mockeado).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── Existencia ──────────────────────────────────────────────────────────────


def test_chat_turn_request_exists():
    assert hasattr(rag, "ChatTurnRequest")


def test_chat_turn_result_exists():
    assert hasattr(rag, "ChatTurnResult")


def test_run_chat_turn_exists():
    assert hasattr(rag, "run_chat_turn")
    assert callable(rag.run_chat_turn)


# ── ChatTurnRequest shape ───────────────────────────────────────────────────


def test_request_minimal_construction():
    """Los campos obligatorios son `question` + `vaults`. Todo lo demás
    tiene default razonable para el caso CLI típico."""
    req = rag.ChatTurnRequest(
        question="qué es ikigai",
        vaults=[],
    )
    assert req.question == "qué es ikigai"
    assert req.vaults == []
    # Defaults razonables
    assert req.history == []
    assert req.loose is False
    assert req.critique is False
    assert req.device == "other"
    assert req.cmd == "chat"
    assert req.k == 4


def test_request_full_construction():
    req = rag.ChatTurnRequest(
        question="q",
        vaults=[("home", Path("/tmp"))],
        history=[{"role": "user", "content": "prev"}],
        loose=True,
        critique=True,
        device="iphone",
        cmd="web",
        k=5,
        folder="01-Projects",
        tag="coaching",
    )
    assert req.device == "iphone"
    assert req.cmd == "web"
    assert req.k == 5
    assert req.folder == "01-Projects"


# ── ChatTurnResult shape ────────────────────────────────────────────────────


def test_result_required_fields():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(rag.ChatTurnResult)}
    # Los de telemetría + respuesta
    required_ish = {
        "answer", "intent", "prompt_version",
        "bad_citations", "bad_citations_count", "citation_repair_attempted",
        "citation_repaired", "critique_fired", "critique_changed",
        "timing",
        "turn_id",
        "retrieve_result",
    }
    assert required_ish.issubset(fields), \
        f"missing fields: {required_ish - fields}"


def test_result_has_reasonable_defaults():
    """ChatTurnResult con sólo `answer` + `retrieve_result` obligatorios."""
    # retrieve_result es obligatorio porque expone las sources/scores/
    # etc. al caller para renderizar. Sin eso, el web endpoint tendría
    # que re-invocar retrieve() — absurdo.
    fake_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-1.0,
    )
    result = rag.ChatTurnResult(
        answer="Hola, no encontré eso en tus notas.",
        retrieve_result=fake_rr,
    )
    assert result.answer.startswith("Hola")
    assert result.intent is None
    assert result.bad_citations_count == 0
    assert result.citation_repaired is False
    assert result.turn_id  # non-empty default (uuid)


# ── run_chat_turn: empty vaults → answer graceful ──────────────────────────


def test_run_chat_turn_with_empty_vaults():
    """Sin vaults, no hay retrieve posible. Debe devolver un
    ChatTurnResult con answer=refusal-template y retrieve_result vacío.
    NO debería invocar al LLM."""
    req = rag.ChatTurnRequest(
        question="qué es ikigai",
        vaults=[],
    )
    with patch("ollama.chat") as mock_chat:
        result = rag.run_chat_turn(req)
    assert isinstance(result, rag.ChatTurnResult)
    assert result.retrieve_result.docs == []
    # Sin LLM call cuando no hay vaults (short-circuit eficiente)
    assert not mock_chat.called
    # Answer debe ser non-empty (mensaje amable) + fast turnaround
    assert result.answer


# ── Feature flag RAG_UNIFIED_CHAT ───────────────────────────────────────────


def test_unified_chat_default_off(monkeypatch):
    """Default OFF — legacy paths siguen funcionando. Flip explícito
    con `RAG_UNIFIED_CHAT=1` cuando el usuario quiere opt-in."""
    monkeypatch.delenv("RAG_UNIFIED_CHAT", raising=False)
    assert rag._unified_chat_enabled() is False


def test_unified_chat_opt_in(monkeypatch):
    monkeypatch.setenv("RAG_UNIFIED_CHAT", "1")
    assert rag._unified_chat_enabled() is True


def test_unified_chat_falsy_values_off(monkeypatch):
    for val in ("0", "false", "no", "", "FALSE", "No"):
        monkeypatch.setenv("RAG_UNIFIED_CHAT", val)
        assert rag._unified_chat_enabled() is False, \
            f"RAG_UNIFIED_CHAT={val!r} should be OFF"


def test_unified_chat_truthy_values_on(monkeypatch):
    for val in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("RAG_UNIFIED_CHAT", val)
        assert rag._unified_chat_enabled() is True


# ── Integration: retrieve_result propagates from multi_retrieve ─────────────


def test_run_chat_turn_propagates_intent_to_retrieve(monkeypatch):
    """El pipeline debe clasificar intent + pasárselo a multi_retrieve
    así se beneficia del adaptive fast-path."""
    captured_kwargs = {}

    class _EmptyCol:
        def count(self): return 0

    def fake_multi_retrieve(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=-1.0,
            intent=kwargs.get("intent"),
            vault_scope=["test"],
        )

    monkeypatch.setattr(rag, "multi_retrieve", fake_multi_retrieve)

    req = rag.ChatTurnRequest(
        question="cuántas notas tengo sobre coaching",  # intent=count
        vaults=[("test", Path("/tmp"))],
    )
    result = rag.run_chat_turn(req)
    # El dispatcher pasó intent al retrieve
    assert captured_kwargs.get("intent") is not None


def test_run_chat_turn_turn_id_is_unique():
    """Cada invocación debe generar un turn_id único (uuid)."""
    req1 = rag.ChatTurnRequest(question="q1", vaults=[])
    req2 = rag.ChatTurnRequest(question="q2", vaults=[])
    result1 = rag.run_chat_turn(req1)
    result2 = rag.run_chat_turn(req2)
    assert result1.turn_id != result2.turn_id
    assert len(result1.turn_id) > 4


# ── to_log_event: dict listo para log_query_event ───────────────────────────


def test_result_to_log_event_shape():
    """El ChatTurnResult debe exponer un método `to_log_event(cmd, session_id)`
    que devuelva el dict shape que `log_query_event()` espera. Así tanto CLI
    como web lo invocan uniformemente y desaparece la duplicación de 4
    call sites con campos que van creciendo."""
    fake_rr = rag.RetrieveResult(
        docs=["d"], metas=[{"file": "a.md"}], scores=[0.8], confidence=0.8,
        intent="synthesis",
    )
    result = rag.ChatTurnResult(
        answer="respuesta",
        retrieve_result=fake_rr,
        intent="synthesis",
        prompt_version="synthesis.v2",
        bad_citations_count=1,
        citation_repair_attempted=True,
        citation_repaired=True,
        turn_id="abc123",
        question="pregunta del usuario",
    )
    event = result.to_log_event(cmd="chat", session_id="sess-1")
    assert event["cmd"] == "chat"
    assert event["session"] == "sess-1"
    assert event["q"]  # non-empty (del request o passthrough)
    assert event["intent"] == "synthesis"
    assert event["prompt_version"] == "synthesis.v2"
    assert event["bad_citations_count"] == 1
    assert event["citation_repaired"] is True
    assert event["turn_id"] == "abc123"
    # Campos standard del shape de rag_queries
    assert "paths" in event
    assert "scores" in event
    assert "top_score" in event
