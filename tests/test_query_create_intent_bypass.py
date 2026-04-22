"""Tests for the create-intent short-circuit in `rag query` (2026-04-22).

Pre-fix: `rag query "mandarle mensaje a mamá a las 18"` ran the whole RAG
pipeline (retrieve + LLM), taking 60-80s trying to answer from the vault
when the actual user intent was "create an Apple Reminder". See
`rag_queries` telemetry: two real rows measured at 66.7s and 79.4s.

Post-fix: `_detect_propose_intent` fires on the query text, the handler
dispatches a single ollama tool-call round (~3-5s), and we return
without ever touching the retriever or the response LLM.

The regex-based detector can have false positives — in that case
`_handle_chat_create_intent` returns `handled=False` and the flow falls
through to the normal RAG path (no compute wasted beyond what we'd
have paid anyway).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Fixture: isolated DB + disabled warmup so we don't block on whisper/bge ─


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    # Fake col with some content so the "empty index" guard doesn't fire.
    fake_col = MagicMock()
    fake_col.count.return_value = 5
    monkeypatch.setattr(rag, "get_db", lambda: fake_col)
    # warmup_async spawns threads in real life — short-circuit to no-op.
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    yield tmp_path


def _last_query_row() -> dict | None:
    """Return the most recent rag_queries row as a dict (or None)."""
    with rag._ragvec_state_conn() as conn:
        conn.row_factory = None
        cur = conn.execute(
            "SELECT cmd, q, t_retrieve, t_gen, extra_json FROM rag_queries "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
    if row is None:
        return None
    cmd, q, t_retrieve, t_gen, extra_json = row
    import json as _json
    extra = _json.loads(extra_json) if extra_json else {}
    return {
        "cmd": cmd, "q": q, "t_retrieve": t_retrieve,
        "t_gen": t_gen, "extra": extra,
    }


# ── Happy path: create-intent fires, handler returns handled=True ───────────


def test_query_short_circuits_on_create_intent(isolated, monkeypatch):
    """The combo detect=True + handled=True must (a) skip retrieve + LLM,
    (b) log a row with t_retrieve=0 + propose_intent_short_circuit=True,
    (c) return cleanly."""
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: True)
    handler_mock = MagicMock(return_value=(True, {"kind": "reminder",
                                                   "reminder_id": "R-1",
                                                   "title": "mensaje mamá"}))
    monkeypatch.setattr(rag, "_handle_chat_create_intent", handler_mock)
    # Anything past the short-circuit would touch retrieve — guard.
    monkeypatch.setattr(rag, "multi_retrieve",
                        MagicMock(side_effect=AssertionError("should not run")))

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "query", "mandarle mensaje a mamá a las 18",
    ])
    assert result.exit_code == 0
    handler_mock.assert_called_once()
    row = _last_query_row()
    assert row is not None
    assert row["cmd"] == "query"
    assert row["q"] == "mandarle mensaje a mamá a las 18"
    assert row["t_retrieve"] == 0.0
    # t_gen is the handler's wall time — should be small, but >=0
    assert row["t_gen"] >= 0.0
    assert row["extra"].get("propose_intent_short_circuit") is True
    assert row["extra"].get("intent") == "create"


def test_query_short_circuit_skips_retrieve_and_llm(isolated, monkeypatch):
    """Belt + suspenders: make every downstream stage explode if called."""
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: True)
    monkeypatch.setattr(rag, "_handle_chat_create_intent",
                        lambda q: (True, None))
    # Any of these being called would mean the short-circuit leaked.
    monkeypatch.setattr(rag, "multi_retrieve",
                        MagicMock(side_effect=AssertionError("multi_retrieve called")))
    monkeypatch.setattr(rag, "retrieve",
                        MagicMock(side_effect=AssertionError("retrieve called")))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "recordame X mañana"])
    assert result.exit_code == 0


# ── False positive: detect=True but handler returns handled=False ──────────


def test_query_falls_through_when_handler_refuses(isolated, monkeypatch):
    """_handle_chat_create_intent returns handled=False when the LLM
    doesn't emit a tool_call (regex false positive on a phrase like
    "qué dije sobre recordame"). In that case we MUST fall through to
    normal RAG flow — not silently swallow the query."""
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: True)
    monkeypatch.setattr(rag, "_handle_chat_create_intent",
                        lambda q: (False, None))
    # Make the RAG path abort early but observably — we just need to
    # confirm it was reached. An empty index guard fires right after.
    fake_col = MagicMock()
    fake_col.count.return_value = 0  # triggers "Índice vacío" guard
    monkeypatch.setattr(rag, "get_db", lambda: fake_col)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "qué dije sobre recordame"])
    assert result.exit_code == 0
    # The "Índice vacío" message comes from the downstream RAG path —
    # proving the fall-through reached it.
    assert "Índice vacío" in result.output or "ndice vac" in result.output


# ── Not a create-intent at all: detect returns False ────────────────────────


def test_query_ignores_normal_queries(isolated, monkeypatch):
    """For non-create queries, the handler must NOT be invoked — we don't
    want a regex false-positive rate to even approach ollama's tool-decide
    latency cost."""
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: False)
    handler_mock = MagicMock()
    monkeypatch.setattr(rag, "_handle_chat_create_intent", handler_mock)
    fake_col = MagicMock()
    fake_col.count.return_value = 0  # empty index → fast abort
    monkeypatch.setattr(rag, "get_db", lambda: fake_col)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "qué es ikigai"])
    assert result.exit_code == 0
    handler_mock.assert_not_called()


# ── Telemetry parity: short-circuited rows are analytics-visible ────────────


def test_short_circuit_row_is_queryable_as_create_intent(isolated, monkeypatch):
    """The log_query_event row should be filterable as cmd='query' AND
    extra_json.propose_intent_short_circuit=1 — that's what the
    dashboard query uses to tally how often this fires vs full pipeline."""
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: True)
    monkeypatch.setattr(rag, "_handle_chat_create_intent",
                        lambda q: (True, None))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "recordame pagar el gas mañana"])
    assert result.exit_code == 0

    # Mirror the dashboard-style analytics query.
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM rag_queries "
            "WHERE cmd='query' "
            "  AND json_extract(extra_json,'$.propose_intent_short_circuit')=1"
        ).fetchone()
    assert row[0] == 1
