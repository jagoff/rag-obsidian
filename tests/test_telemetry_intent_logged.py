"""Tests for Game-changer #2.A (2026-04-22) — telemetría honesta: intent
debe loguearse en rag_queries.extra_json en cada log_query_event del flujo
de query().

Pre-fix evidence: 486/500 queries recientes en telemetry.db tenían
`intent=NULL` — el dict pasado a log_query_event nunca incluía `intent`.
Post-fix: cada path de query() loguea `intent` en extra_json.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import rag


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCollection:
    def count(self):
        return 1


def _fake_result(path="01-Projects/nota.md", conf=0.8):
    return {
        "docs": ["chunk"],
        "metas": [{"file": path, "note": "nota", "folder": "01-Projects"}],
        "scores": [conf],
        "confidence": conf,
        "filters_applied": {},
        "query_variants": [],
        "extras": [],
        "graph_docs": [],
        "graph_metas": [],
    }


def _base_patches(monkeypatch, retrieve_result=None, intent="semantic"):
    if retrieve_result is None:
        retrieve_result = _fake_result()
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCollection())
    monkeypatch.setattr(rag, "get_vocabulary", lambda col: ([], []))
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: (intent, {}))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: retrieve_result)
    monkeypatch.setattr(rag, "build_progressive_context", lambda *a, **kw: "CTX")
    monkeypatch.setattr(rag, "user_prompt_block", lambda: "")
    monkeypatch.setattr(rag, "print_sources", lambda r: None)
    monkeypatch.setattr(rag, "find_related", lambda col, metas: [])
    monkeypatch.setattr(rag, "render_related", lambda rel: None)
    monkeypatch.setattr(rag, "new_turn_id", lambda: "tid-1")
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "fake-model")
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])


def test_log_event_includes_intent_in_happy_path(monkeypatch):
    """Normal query path logs intent in the event dict."""
    _base_patches(monkeypatch, intent="semantic")

    def fake_chat(model, messages, options, stream, keep_alive):
        def gen():
            yield _FakeResponse("respuesta")
        return gen()

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--no-multi", "--no-cache", "q"])
    assert result.exit_code == 0, result.output
    assert logged.get("intent") == "semantic"
    assert logged.get("cmd") == "query"


def test_log_event_includes_intent_on_no_docs(monkeypatch):
    """Path 'sin resultados' also logs intent. Uses semantic intent to avoid
    triggering intent-shortcut handlers (count/list/recent/agenda/entity_lookup)
    which short-circuit before reaching the no-docs block."""
    empty_result = {**_fake_result(), "docs": [], "metas": [], "scores": []}
    _base_patches(monkeypatch, retrieve_result=empty_result, intent="semantic")
    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--no-multi", "--no-cache", "q"])
    assert result.exit_code == 0
    assert logged.get("intent") == "semantic"
    assert logged.get("answered") is False


def test_log_event_includes_intent_on_confidence_gate(monkeypatch):
    """Path 'low-confidence refuse' also logs intent."""
    low_result = _fake_result(conf=0.005)  # below CONFIDENCE_RERANK_MIN
    _base_patches(monkeypatch, retrieve_result=low_result, intent="semantic")
    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--no-multi", "--no-cache", "q"])
    assert result.exit_code == 0
    assert logged.get("intent") == "semantic"
    assert logged.get("gated_low_confidence") is True
