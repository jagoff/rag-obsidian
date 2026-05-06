"""Integration tests for MLXBackend.chat(tools=...) — Ola 5 wire.

Verifies that `tools=[...]` flows from the caller through
`tokenizer.apply_chat_template(tools=...)` and that the model output is
parsed by `rag.mlx_tool_calls.parse_tool_calls` into ollama-shape
`Message.ToolCall` objects on the response.

Tests skip when `mlx_lm` is not installed (CI without MLX extras).
`_load` and `_mlx_generate` are patched so the test never loads a real
model — only the wiring is exercised.
"""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

import pytest

mlx_lm = pytest.importorskip("mlx_lm")  # noqa: F841

from rag.llm_backend import ChatOptions, MLXBackend, reset_backend  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    reset_backend()
    yield
    reset_backend()


class _FakeTokenizer:
    """Records `apply_chat_template` invocations and returns a deterministic prompt."""

    def __init__(self, accept_tools: bool = True) -> None:
        self.accept_tools = accept_tools
        self.last_call: dict[str, Any] | None = None

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        tools=None,
    ) -> str:
        if tools is not None and not self.accept_tools:
            raise TypeError("apply_chat_template() got unexpected kwarg 'tools'")
        self.last_call = {
            "messages": list(messages),
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        return "<rendered prompt>"


def _make_backend(monkeypatch, tokenizer: _FakeTokenizer, generated_text: str) -> MLXBackend:
    """Build an MLXBackend with `_load` and `_mlx_generate` patched.

    Avoids any real `mlx_lm.load()` call. The fake tokenizer captures
    `apply_chat_template` so we can assert on `tools=` propagation.
    """
    backend = MLXBackend()
    backend.shutdown_watchdog()  # tests don't need the daemon

    monkeypatch.setattr(
        backend, "_load", lambda model_id: (mock.MagicMock(name="model"), tokenizer),
    )
    monkeypatch.setattr(
        backend,
        "_mlx_generate",
        lambda model, tok, prompt, opts: generated_text,
    )
    return backend


# ---------------------------------------------------------------------------
# 1. tools= propagated to template, parse_tool_calls populates response
# ---------------------------------------------------------------------------


def test_mlx_chat_with_tools_emits_tool_calls(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=True)
    backend = _make_backend(
        monkeypatch,
        tokenizer,
        '<tool_call>\n{"name": "propose_reminder", "arguments": {"title": "llamar a Juan", "due_iso": "2026-05-07T09:00"}}\n</tool_call>',
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "propose_reminder",
                "description": "Crear un recordatorio de Apple Reminders",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "due_iso": {"type": "string"},
                    },
                    "required": ["title"],
                },
            },
        }
    ]
    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "recordame llamar a Juan mañana 9am"}],
        options=ChatOptions(),
        tools=tools,
    )

    assert tokenizer.last_call is not None
    assert tokenizer.last_call["tools"] == tools, "tools must reach apply_chat_template"

    tool_calls = resp.message.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.name == "propose_reminder"
    assert tool_calls[0].function.arguments["title"] == "llamar a Juan"
    assert tool_calls[0].function.arguments["due_iso"] == "2026-05-07T09:00"


# ---------------------------------------------------------------------------
# 2. tokenizer rejects tools= → graceful fallback (no crash, no tool_calls)
# ---------------------------------------------------------------------------


def test_mlx_chat_tokenizer_rejects_tools_falls_back_gracefully(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=False)
    backend = _make_backend(
        monkeypatch,
        tokenizer,
        "Texto plano sin tool calls.",
    )

    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "hola"}],
        options=ChatOptions(),
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )

    # Second call (without tools=) must have succeeded.
    assert tokenizer.last_call is not None
    assert tokenizer.last_call["tools"] is None
    # No tool_calls — caller can route to Ollama-fallback if needed.
    assert resp.message.tool_calls is None
    assert "Texto plano" in resp.message.content


# ---------------------------------------------------------------------------
# 3. tools= present but model output has no <tool_call> → plain text response
# ---------------------------------------------------------------------------


def test_mlx_chat_with_tools_but_no_calls_in_output(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=True)
    backend = _make_backend(
        monkeypatch,
        tokenizer,
        "Lo siento, no puedo crear el recordatorio.",
    )

    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "hola"}],
        options=ChatOptions(),
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )

    assert resp.message.tool_calls is None
    assert resp.message.content == "Lo siento, no puedo crear el recordatorio."


# ---------------------------------------------------------------------------
# 4. multi tool_call blocks — parser returns all of them
# ---------------------------------------------------------------------------


def test_mlx_chat_multiple_tool_calls(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=True)
    text = (
        '<tool_call>\n{"name": "search_vault", "arguments": {"query": "ikigai"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "weather", "arguments": {"location": "Santa Fe"}}\n</tool_call>'
    )
    backend = _make_backend(monkeypatch, tokenizer, text)

    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "?"}],
        options=ChatOptions(),
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )

    calls = resp.message.tool_calls
    assert calls is not None and len(calls) == 2
    assert calls[0].function.name == "search_vault"
    assert calls[1].function.name == "weather"


# ---------------------------------------------------------------------------
# 5. tools=None — backwards-compat: no tools kwarg passed to template
# ---------------------------------------------------------------------------


def test_mlx_chat_without_tools_does_not_pass_tools_kwarg(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=True)
    backend = _make_backend(monkeypatch, tokenizer, "respuesta normal")

    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "hola"}],
        options=ChatOptions(),
    )

    assert tokenizer.last_call is not None
    assert tokenizer.last_call["tools"] is None
    assert resp.message.tool_calls is None
    assert resp.message.content == "respuesta normal"


# ---------------------------------------------------------------------------
# 6. format='json' + tool_calls — tool_calls win, no JSON extraction on content
# ---------------------------------------------------------------------------


def test_mlx_chat_format_json_with_tool_calls(monkeypatch):
    tokenizer = _FakeTokenizer(accept_tools=True)
    backend = _make_backend(
        monkeypatch,
        tokenizer,
        '<tool_call>\n{"name": "x", "arguments": {"k": "v"}}\n</tool_call>',
    )

    resp = backend.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": "?"}],
        options=ChatOptions(),
        format="json",
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )

    assert resp.message.tool_calls is not None
    assert len(resp.message.tool_calls) == 1
    # When tool_calls present, content is the stripped prose (here: empty).
    assert resp.message.content == ""
