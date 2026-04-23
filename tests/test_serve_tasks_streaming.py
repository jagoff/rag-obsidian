"""
Source-level contract for the streaming + TTFT tracking added to the
`serve.tasks` path (rag.py, inside `_handle_query`).

Pre-fix the tasks branch called `_chat_capped_client().chat(...)` with
stream=False and blocked on the full response. t_gen in serve.tasks
ranged from 11s (simple agenda, no mail evidence) to 68.8s (with Gmail
+ Apple Mail). Impossible to diagnose prefill-vs-decode split without
TTFT.

Post-fix iterates chunks with `stream=True` and captures `ttft_ms` at
the first non-empty content chunk. Surfaced in:
  - log_query_event (rag_queries.extra_json.ttft_ms)
  - response payload (`tasks_payload["ttft_ms"]`)
  - timing dict inside extra_json

This is a source-level assertion — we cannot end-to-end test the serve
endpoint without spinning up ollama + the full tasks collector.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG_PY = (ROOT / "rag.py").read_text(encoding="utf-8")


def _tasks_block() -> str:
    """Locate the serve.tasks block in `_handle_query`. Anchor on the
    `log_query_event` call for `serve.tasks` and walk backwards/forwards."""
    anchor = RAG_PY.find('"cmd": "serve.tasks"')
    assert anchor >= 0, "serve.tasks log_query_event anchor not found"
    # Wide window covers tasks_predict_cap assignment (up-tree) and the
    # `tasks_payload` dict (down-tree).
    start = max(0, anchor - 5000)
    end = min(len(RAG_PY), anchor + 5000)
    return RAG_PY[start:end]


def test_serve_tasks_uses_streaming():
    """The chat call inside serve.tasks must use `stream=True` instead
    of the old blocking form."""
    block = _tasks_block()
    # Must invoke stream=True in the chat call
    assert "stream=True" in block, (
        "serve.tasks LLM call must use stream=True (found non-streaming"
        " call or missing)"
    )
    # Must iterate chunks
    assert "for chunk in _chat_capped_client().chat(" in block, (
        "expected `for chunk in _chat_capped_client().chat(...)` to iterate"
    )


def test_serve_tasks_captures_ttft():
    """TTFT must be initialized None and set at first non-empty chunk."""
    block = _tasks_block()
    assert "ttft_ms: int | None = None" in block, (
        "expected `ttft_ms: int | None = None` initialization"
    )
    # Timing math must use t_gen0 (the start marker for the generation)
    assert "time.perf_counter() - t_gen0" in block, (
        "expected TTFT to be computed relative to t_gen0"
    )
    # The TTFT write must be gated on ttft_ms being None (first-time)
    assert "if ttft_ms is None:" in block, (
        "expected `if ttft_ms is None:` gate on first-token"
    )


def test_serve_tasks_logs_ttft_in_extra_json():
    """`log_query_event` for serve.tasks must include `ttft_ms` as a
    first-class key AND in the `timing` dict."""
    block = _tasks_block()
    # Direct key
    assert '"ttft_ms": ttft_ms' in block, (
        "serve.tasks log_query_event must include `\"ttft_ms\": ttft_ms`"
    )
    # Inside timing dict
    timing_idx = block.find('"timing": _round_timing_ms(')
    assert timing_idx >= 0
    timing_block = block[timing_idx : timing_idx + 500]
    assert "ttft_ms" in timing_block, (
        "`ttft_ms` must be inside the `timing` dict as well"
    )


def test_serve_tasks_response_includes_ttft():
    """The `tasks_payload` returned to the client must include ttft_ms
    so non-WA clients can surface it."""
    block = _tasks_block()
    payload_idx = block.find("tasks_payload = {")
    assert payload_idx >= 0, "tasks_payload dict not found"
    payload_block = block[payload_idx : payload_idx + 1200]
    assert '"ttft_ms": ttft_ms' in payload_block, (
        "tasks_payload must include `ttft_ms` key"
    )


def test_serve_tasks_answer_still_full_concat():
    """Back-compat: the final `answer` must still be the full
    concatenated text (WA listener depends on it). Streaming
    internally must not change the contract with the client."""
    block = _tasks_block()
    # parts list is the streaming accumulator
    assert 'parts: list[str] = []' in block, (
        "expected `parts: list[str] = []` to accumulate streaming content"
    )
    assert 'answer = "".join(parts).strip()' in block, (
        "final answer must be `\"\".join(parts).strip()` for back-compat"
    )


def test_serve_tasks_predict_cap_unchanged():
    """We did NOT touch the 700-token cap — `tasks_predict_cap` still
    max(predict_cap, 700). Truncating mid-agenda is worse than slow."""
    block = _tasks_block()
    assert "tasks_predict_cap = max(predict_cap, 700)" in block, (
        "num_predict cap of 700 must be preserved for agenda completeness"
    )


def test_serve_tasks_exception_handling_preserved():
    """The `except Exception as exc: return {\"error\": ...}` must still
    guard the LLM call. Otherwise a dropped ollama connection returns 500
    instead of a structured error to the client."""
    block = _tasks_block()
    # Find the chat() try/except
    chat_idx = block.find("for chunk in _chat_capped_client().chat(")
    assert chat_idx >= 0
    # Look 800 chars forward for the except + error-return
    window = block[chat_idx : chat_idx + 800]
    assert "except Exception as exc:" in window
    assert 'return {"error": f"LLM falló: {exc}"}' in window, (
        "error-return contract must be preserved"
    )
