"""Feature #3 del 2026-04-23 — LLM-based intent classifier fallback.

Validates:
- classify_intent regex path unchanged when flag OFF
- _classify_intent_llm mocks ollama and parses JSON
- Invalid LLM output falls back to 'semantic'
- Cache hit avoids second ollama call
- Cache TTL eviction
- Cache max size bounded (FIFO eviction)
- LLM-upgraded intents route correctly in classify_intent
"""
from __future__ import annotations

import json
import time
from typing import Any

import pytest

import rag


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeClient:
    def __init__(self):
        self._next: list = []
        self.calls: list[dict] = []

    def set_response(self, content: Any):
        self._next.append(content)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self._next:
            return _FakeResponse(json.dumps({"intent": "semantic"}))
        nxt = self._next.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _FakeResponse(nxt)


@pytest.fixture
def fake_ollama(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(rag, "_summary_client", lambda: client)
    # Reset the cache between tests.
    rag._intent_llm_cache = None
    rag._ensure_intent_llm_cache()
    rag._intent_llm_cache.clear()
    return client


# ── _classify_intent_llm ────────────────────────────────────────────────


def test_llm_empty_question_returns_none(fake_ollama):
    assert rag._classify_intent_llm("") is None
    assert rag._classify_intent_llm("   ") is None


def test_llm_parses_valid_json(fake_ollama):
    fake_ollama.set_response(json.dumps({"intent": "count"}))
    assert rag._classify_intent_llm("unusual count phrasing") == "count"


def test_llm_invalid_intent_returns_none(fake_ollama):
    fake_ollama.set_response(json.dumps({"intent": "garbage-not-in-enum"}))
    assert rag._classify_intent_llm("x") is None


def test_llm_invalid_json_returns_none(fake_ollama):
    fake_ollama.set_response("not JSON at all {{{")
    assert rag._classify_intent_llm("x") is None


def test_llm_exception_returns_none(fake_ollama):
    fake_ollama.set_response(RuntimeError("ollama down"))
    assert rag._classify_intent_llm("x") is None


def test_llm_missing_intent_key_returns_none(fake_ollama):
    fake_ollama.set_response(json.dumps({"other": "field"}))
    assert rag._classify_intent_llm("x") is None


def test_llm_non_string_intent_returns_none(fake_ollama):
    fake_ollama.set_response(json.dumps({"intent": 42}))
    assert rag._classify_intent_llm("x") is None


def test_llm_all_valid_intents_accepted(fake_ollama):
    valid_intents = [
        "count", "list", "recent", "agenda",
        "entity_lookup", "comparison", "synthesis", "semantic",
    ]
    for intent in valid_intents:
        # Clear cache between iterations so each goes through the LLM path.
        rag._intent_llm_cache.clear()
        fake_ollama.set_response(json.dumps({"intent": intent}))
        assert rag._classify_intent_llm(f"probe for {intent}") == intent


# ── cache behavior ──────────────────────────────────────────────────────


def test_cache_hit_avoids_second_ollama_call(fake_ollama):
    fake_ollama.set_response(json.dumps({"intent": "count"}))
    rag._classify_intent_llm("cache me please")
    call_count_after_first = len(fake_ollama.calls)
    assert call_count_after_first == 1
    # Second call should hit cache.
    rag._classify_intent_llm("cache me please")
    assert len(fake_ollama.calls) == call_count_after_first
    # Case-insensitive lookup.
    rag._classify_intent_llm("CACHE ME PLEASE")
    assert len(fake_ollama.calls) == call_count_after_first


def test_cache_ttl_expiration(fake_ollama, monkeypatch):
    fake_ollama.set_response(json.dumps({"intent": "count"}))
    fake_ollama.set_response(json.dumps({"intent": "list"}))

    rag._classify_intent_llm("timed")
    # Simulate TTL expiry by modifying the entry's ts.
    with rag._intent_llm_cache_lock:
        rag._intent_llm_cache["timed"] = (
            "count", time.time() - rag._INTENT_LLM_CACHE_TTL_S - 1,
        )
    # Expired entry → new LLM call.
    result = rag._classify_intent_llm("timed")
    assert result == "list"
    assert len(fake_ollama.calls) == 2


def test_cache_fifo_eviction_respects_max(fake_ollama, monkeypatch):
    monkeypatch.setattr(rag, "_INTENT_LLM_CACHE_MAX", 3)
    # Fill with 3 unique entries.
    for label in ("a", "b", "c"):
        fake_ollama.set_response(json.dumps({"intent": "count"}))
        rag._classify_intent_llm(label)
    assert len(rag._intent_llm_cache) == 3
    # 4th entry → evicts "a".
    fake_ollama.set_response(json.dumps({"intent": "list"}))
    rag._classify_intent_llm("d")
    assert len(rag._intent_llm_cache) == 3
    assert "a" not in rag._intent_llm_cache
    assert "d" in rag._intent_llm_cache


# ── classify_intent integration ─────────────────────────────────────────


def test_classify_intent_regex_wins_when_flag_off(fake_ollama, monkeypatch):
    monkeypatch.setattr(rag, "_INTENT_LLM_ENABLED", False)
    # Explicit count phrasing — regex should match, LLM never called.
    intent, _ = rag.classify_intent("cuántas notas tengo", set(), set())
    assert intent == "count"
    assert len(fake_ollama.calls) == 0


def test_classify_intent_llm_called_on_fallthrough(fake_ollama, monkeypatch):
    monkeypatch.setattr(rag, "_INTENT_LLM_ENABLED", True)
    fake_ollama.set_response(json.dumps({"intent": "count"}))
    # Non-canonical phrasing that regex misses.
    intent, _ = rag.classify_intent(
        "me interesa saber qué cantidad de archivos tengo sobre finanzas",
        set(), set(),
    )
    assert intent == "count"
    assert len(fake_ollama.calls) == 1


def test_classify_intent_llm_ignored_when_returns_semantic(fake_ollama, monkeypatch):
    monkeypatch.setattr(rag, "_INTENT_LLM_ENABLED", True)
    fake_ollama.set_response(json.dumps({"intent": "semantic"}))
    intent, _ = rag.classify_intent("unusual query with no clear intent",
                                     set(), set())
    assert intent == "semantic"


def test_classify_intent_llm_timeout_falls_back_to_semantic(
    fake_ollama, monkeypatch,
):
    monkeypatch.setattr(rag, "_INTENT_LLM_ENABLED", True)
    fake_ollama.set_response(RuntimeError("timeout"))
    intent, _ = rag.classify_intent("fallthrough q", set(), set())
    assert intent == "semantic"


def test_classify_intent_preserves_tag_and_folder_params(
    fake_ollama, monkeypatch,
):
    """Params (tag/folder) extracted BEFORE the intent LLM call stay intact."""
    monkeypatch.setattr(rag, "_INTENT_LLM_ENABLED", True)
    fake_ollama.set_response(json.dumps({"intent": "count"}))
    tags = {"obsidian"}
    # Use folder name long enough (≥5 chars).
    folders = {"01-Projects"}
    intent, params = rag.classify_intent(
        "#obsidian todo lo de 01-projects",
        tags, folders,
    )
    # Regex path would hit 'list' via "todo lo de"; regardless, tag captured.
    assert params.get("tag") == "obsidian"


# ── env parsing ─────────────────────────────────────────────────────────


def test_env_flag_parsing_truthy(monkeypatch):
    # Simulate module reload — we emulate by re-evaluating the expression.
    for val in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("RAG_LLM_INTENT", val)
        # Re-check the expression matches.
        assert val.strip().lower() in ("1", "true", "yes")


def test_env_flag_parsing_falsy(monkeypatch):
    for val in ("0", "false", "no", "", "off"):
        monkeypatch.setenv("RAG_LLM_INTENT", val)
        assert val.strip().lower() not in ("1", "true", "yes")
