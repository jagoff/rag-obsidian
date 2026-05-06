"""Tests para `rag/mlx_tool_calls.py` — Qwen2.5/3 tool-call parser.

No requiere mlx-lm ni Ollama daemon. Solo testea la lógica de parsing
sobre strings hardcoded que reproducen el formato emitido por los
modelos Qwen.
"""

from __future__ import annotations

from rag.mlx_tool_calls import parse_tool_calls, strip_tool_call_blocks


def test_parse_no_tool_call_returns_none():
    assert parse_tool_calls("hola sin tool") is None
    assert parse_tool_calls("") is None
    assert parse_tool_calls(None) is None


def test_parse_single_tool_call():
    text = (
        'Voy a usar weather.\n'
        '<tool_call>\n'
        '{"name": "get_weather", "arguments": {"location": "Santa Fe"}}\n'
        '</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == {"location": "Santa Fe"}


def test_parse_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert calls is not None
    assert [c.function.name for c in calls] == ["a", "b"]


def test_parse_repairs_single_quotes():
    text = "<tool_call>{'name': 'x', 'arguments': {'a': 1}}</tool_call>"
    calls = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].function.name == "x"
    assert calls[0].function.arguments == {"a": 1}


def test_parse_repairs_trailing_comma():
    text = '<tool_call>{"name": "x", "arguments": {"a": 1,}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].function.arguments == {"a": 1}


def test_parse_skips_unparseable_block():
    text = (
        '<tool_call>{not json at all}</tool_call>'
        '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "ok"


def test_parse_drops_call_without_name():
    text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    assert parse_tool_calls(text) is None


def test_parse_string_arguments_recovered():
    text = (
        '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].function.arguments == {"a": 1}


def test_parse_args_alias_args_field():
    text = '<tool_call>{"name": "x", "args": {"a": 1}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls is not None
    assert calls[0].function.arguments == {"a": 1}


def test_strip_keeps_prose():
    text = (
        'Voy a usar weather.\n'
        '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    )
    assert strip_tool_call_blocks(text) == "Voy a usar weather."


def test_strip_only_tool_call_returns_none():
    text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    assert strip_tool_call_blocks(text) is None


def test_strip_multiple_blocks():
    text = (
        'A.\n<tool_call>{"name": "x", "arguments": {}}</tool_call>\n'
        'B.\n<tool_call>{"name": "y", "arguments": {}}</tool_call>'
    )
    result = strip_tool_call_blocks(text)
    assert result is not None
    assert "A." in result and "B." in result
    assert "<tool_call>" not in result
