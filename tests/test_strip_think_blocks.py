"""Tests for strip_think_blocks — Qwen3 <think>...</think> stripping."""

import json
import re

from rag.llm_backend import strip_think_blocks


def test_passthrough_no_think_tag():
    text = '{"contradictions": []}'
    assert strip_think_blocks(text) == text


def test_passthrough_empty_string():
    assert strip_think_blocks("") == ""


def test_single_think_block_before_json():
    text = '<think>Let me reason...</think>\n{"result": 42}'
    result = strip_think_blocks(text)
    assert result == '{"result": 42}'
    assert json.loads(result)["result"] == 42


def test_think_block_with_braces_inside():
    text = '<think>The JSON should be {idx: 1}...</think>\n{"contradictions": []}'
    result = strip_think_blocks(text)
    assert result == '{"contradictions": []}'
    assert json.loads(result)["contradictions"] == []


def test_multiple_think_blocks_all_stripped():
    text = '<think>First...</think>\n<think>Second...</think>\n{"ok": true}'
    result = strip_think_blocks(text)
    assert result == '{"ok": true}'


def test_case_insensitive_think_tag():
    text = '<THINK>uppercase</THINK>\n{"answer": "yes"}'
    result = strip_think_blocks(text)
    assert result == '{"answer": "yes"}'


def test_mixed_case_think_tag():
    text = '<Think>Mixed...</Think>\n{"value": 1}'
    result = strip_think_blocks(text)
    assert result == '{"value": 1}'


def test_unclosed_think_tag_not_modified():
    text = "<think>Incomplete reasoning without closing tag {key: val}"
    result = strip_think_blocks(text)
    assert result == text


def test_think_block_multiline():
    text = (
        "<think>\n"
        "Line 1\n"
        "Line 2\n"
        "</think>\n"
        '{"contradictions": [{"index": 1, "why": "incompatible"}]}'
    )
    result = strip_think_blocks(text)
    assert result.startswith('{"contradictions"')
    data = json.loads(result)
    assert len(data["contradictions"]) == 1


def test_no_think_tag_plain_prose():
    text = 'Here is my analysis: {"result": true}'
    assert strip_think_blocks(text) == text


def test_think_block_then_json_extract_correct():
    raw = (
        "<think>Let me check: {bad_json: no_value}...</think>\n"
        '{"contradictions": [{"index": 2, "why": "says opposite"}]}'
    )
    clean = strip_think_blocks(raw)
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    assert m is not None
    data = json.loads(m.group(0))
    assert data["contradictions"][0]["index"] == 2


def test_strip_returns_str_type():
    result = strip_think_blocks("<think>x</think>\nhello")
    assert isinstance(result, str)
    assert result == "hello"
