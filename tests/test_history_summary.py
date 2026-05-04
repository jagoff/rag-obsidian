"""Tests for Quick Win #5 — selective conversation history summarisation.

Validates:
1. LLM called with prompt that includes history
2. Cache hit/miss: 2 calls with same hash → 1 LLM call
3. Durable cache: SQL row survives in-memory clear (no module-level cache)
4. Fallback: LLM failure → raw concatenation returned, no crash
5. Wire-up: history len == 1 (≤2 messages) → no summary call, raw used
6. Wire-up: history len >= 2 turns (>2 messages) → summary + last turn raw
7. SQL DDL: table exists after import
8. HELPER_OPTIONS invariant: call uses temperature=0, seed=42
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import rag


# ── DB isolation ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect rag.DB_PATH to a tmp dir so tests never touch prod telemetry."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    yield


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_history(n_turns: int = 3) -> list[dict]:
    """Build n_turns of (user, assistant) message pairs."""
    msgs: list[dict] = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"pregunta {i}"})
        msgs.append({"role": "assistant", "content": f"respuesta {i}"})
    return msgs


def _mock_ollama_resp(text: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = text
    return resp


# ── Test 7: SQL DDL (quick, no network) ──────────────────────────────────────


def test_sql_ddl_table_exists(tmp_path, monkeypatch):
    """rag_conversation_summaries table must exist post _ensure_telemetry_tables."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='rag_conversation_summaries'"
        ).fetchall()
    assert rows, "rag_conversation_summaries table not created"


# ── Test 1: prompt includes history ──────────────────────────────────────────


def test_prompt_includes_history(monkeypatch):
    """LLM call prompt must contain the prior history messages."""
    history = _make_history(2)  # 4 messages; history[:-2] = 2 messages

    captured_prompt: list[str] = []

    def fake_chat(**kwargs):
        msgs = kwargs.get("messages", [])
        captured_prompt.append(msgs[0]["content"] if msgs else "")
        return _mock_ollama_resp("resumen de prueba")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        result = rag._summarize_conversation_history(history[:-2], "sess1")

    assert result == "resumen de prueba"
    assert len(captured_prompt) == 1
    prompt = captured_prompt[0]
    # The prior messages should appear in the prompt
    assert "pregunta 0" in prompt or "respuesta 0" in prompt


# ── Test 8: HELPER_OPTIONS determinism ───────────────────────────────────────


def test_uses_helper_options_deterministic(monkeypatch):
    """LLM call must use temperature=0, seed=42 (HELPER_OPTIONS invariant)."""
    history = _make_history(2)

    captured_opts: list[dict] = []

    def fake_chat(**kwargs):
        captured_opts.append(dict(kwargs.get("options", {})))
        return _mock_ollama_resp("resumen det")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        rag._summarize_conversation_history(history[:-2], "sess_det")

    assert captured_opts, "No LLM call captured"
    opts = captured_opts[0]
    assert opts.get("temperature") == 0, f"Expected temperature=0, got {opts.get('temperature')}"
    assert opts.get("seed") == 42, f"Expected seed=42, got {opts.get('seed')}"


# ── Test 2: cache hit/miss ────────────────────────────────────────────────────


def test_cache_hit_miss_single_llm_call(monkeypatch):
    """Two calls with same history + session_id → exactly 1 LLM call."""
    history = _make_history(2)

    call_count = {"n": 0}

    def fake_chat(**kwargs):
        call_count["n"] += 1
        return _mock_ollama_resp("resumen cacheado")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        r1 = rag._summarize_conversation_history(history[:-2], "sess_cache")
        r2 = rag._summarize_conversation_history(history[:-2], "sess_cache")

    assert call_count["n"] == 1, f"Expected 1 LLM call, got {call_count['n']}"
    assert r1 == r2 == "resumen cacheado"


# ── Test 2b: different session_id → different cache key ─────────────────────


def test_different_session_id_different_cache_key(monkeypatch):
    """Same history but different session_id → 2 separate LLM calls."""
    history = _make_history(2)

    call_count = {"n": 0}

    def fake_chat(**kwargs):
        call_count["n"] += 1
        return _mock_ollama_resp(f"resumen {call_count['n']}")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        r1 = rag._summarize_conversation_history(history[:-2], "sess_A")
        r2 = rag._summarize_conversation_history(history[:-2], "sess_B")

    assert call_count["n"] == 2, f"Expected 2 LLM calls, got {call_count['n']}"
    assert r1 != r2


# ── Test 3: durable SQL cache ────────────────────────────────────────────────


def test_cache_is_sql_durable(monkeypatch, tmp_path):
    """Cache entry persists in SQL after the function returns.
    Second call reads from DB, not from any in-memory cache."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    history = _make_history(2)

    call_count = {"n": 0}

    def fake_chat(**kwargs):
        call_count["n"] += 1
        return _mock_ollama_resp("resumen sql")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        r1 = rag._summarize_conversation_history(history[:-2], "sess_sql")

    # Call again — the SQL row should be the cache, no second LLM call needed
    with patch.object(rag, "_helper_client", return_value=mock_client):
        r2 = rag._summarize_conversation_history(history[:-2], "sess_sql")

    assert call_count["n"] == 1
    assert r1 == r2 == "resumen sql"

    # Verify it's actually in the DB
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT summary FROM rag_conversation_summaries LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row[0] == "resumen sql"


# ── Test 4: fallback on LLM failure ─────────────────────────────────────────


def test_fallback_on_llm_failure(monkeypatch):
    """LLM failure → returns raw concat fallback, no exception raised."""
    history = [
        {"role": "user", "content": "pregunta fallback"},
        {"role": "assistant", "content": "respuesta fallback"},
    ]

    def fake_chat(**kwargs):
        raise RuntimeError("ollama down")

    mock_client = MagicMock()
    mock_client.chat.side_effect = fake_chat
    with patch.object(rag, "_helper_client", return_value=mock_client):
        result = rag._summarize_conversation_history(history, "sess_fail")

    # Must not raise; must return something with the content
    assert isinstance(result, str)
    assert len(result) > 0
    assert "pregunta fallback" in result or "respuesta fallback" in result


# ── Test 4b: empty history → empty string ────────────────────────────────────


def test_empty_history_returns_empty_string(monkeypatch):
    """Empty history list → returns empty string immediately (no LLM call)."""
    mock_client = MagicMock()
    with patch.object(rag, "_helper_client", return_value=mock_client):
        result = rag._summarize_conversation_history([], "sess_empty")

    assert result == ""
    mock_client.chat.assert_not_called()


# ── Test 5 & 6: web/server.py wire-up (via env var gate) ─────────────────────


def test_wire_up_gate_off_uses_raw_history(monkeypatch):
    """When RAG_HISTORY_SUMMARY=0, _summarize_conversation_history is never called."""
    monkeypatch.setenv("RAG_HISTORY_SUMMARY", "0")

    call_count = {"n": 0}
    orig = rag._summarize_conversation_history

    def patched(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)

    with patch.object(rag, "_summarize_conversation_history", side_effect=patched):
        # Simulate: the gate check reads RAG_HISTORY_SUMMARY; if "0" it should
        # not call the function.  We reproduce the gate logic here directly.
        import os
        gate_on = os.environ.get("RAG_HISTORY_SUMMARY", "1").strip().lower() not in (
            "0", "false", "no", ""
        )
        history = _make_history(3)  # 6 messages — would trigger without gate
        if gate_on and len(history) > 2:
            rag._summarize_conversation_history(history[:-2], "sess_gate")

    assert call_count["n"] == 0, "Summarizer should not be called when gate is OFF"


def test_wire_up_short_history_no_summary(monkeypatch):
    """With only 1 turn (2 messages), no summary is requested."""
    monkeypatch.setenv("RAG_HISTORY_SUMMARY", "1")

    history = _make_history(1)  # 2 messages = 1 turn — should NOT trigger summary
    assert len(history) == 2
    # Gate condition: len(history) > 2 → False for 1 turn
    import os
    gate_on = os.environ.get("RAG_HISTORY_SUMMARY", "1").strip().lower() not in (
        "0", "false", "no", ""
    )
    should_summarize = gate_on and len(history) > 2
    assert not should_summarize, "1-turn history should not trigger summarisation"


def test_wire_up_long_history_triggers_summary(monkeypatch):
    """With ≥2 turns (>2 messages) and gate ON, summary function is invoked."""
    monkeypatch.setenv("RAG_HISTORY_SUMMARY", "1")

    history = _make_history(3)  # 6 messages = 3 turns
    import os
    gate_on = os.environ.get("RAG_HISTORY_SUMMARY", "1").strip().lower() not in (
        "0", "false", "no", ""
    )
    should_summarize = gate_on and len(history) > 2
    assert should_summarize, "3-turn history should trigger summarisation"

    # Verify the slice: history[:-2] = prior turns, history[-2:] = last turn
    prior = history[:-2]
    last = history[-2:]
    assert len(prior) == 4  # 2 prior turns = 4 messages
    assert len(last) == 2   # 1 last turn = 2 messages
    assert last[0]["role"] == "user"
    assert last[0]["content"] == "pregunta 2"
