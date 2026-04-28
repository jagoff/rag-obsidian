"""Tests para `truncate_tool_output_for_synthesis` (eval 2026-04-28 fix)."""
from __future__ import annotations

import json

import pytest

from rag._tool_output_helpers import (
    _DEFAULT_CAPS,
    _truncate_list,
    truncate_tool_output_for_synthesis,
)


class TestTruncateList:
    def test_under_cap_unchanged(self):
        lst, n = _truncate_list([1, 2, 3], 5)
        assert lst == [1, 2, 3]
        assert n == 0

    def test_at_cap_unchanged(self):
        lst, n = _truncate_list([1, 2, 3], 3)
        assert lst == [1, 2, 3]
        assert n == 0

    def test_over_cap_truncated(self):
        lst, n = _truncate_list([1, 2, 3, 4, 5, 6], 3)
        assert lst == [1, 2, 3]
        assert n == 3

    def test_non_list_passthru(self):
        lst, n = _truncate_list("not a list", 5)  # type: ignore
        assert lst == "not a list"
        assert n == 0


class TestTruncateToolOutput:
    def test_unknown_tool_passthru(self):
        """Si la tool no está en _DEFAULT_CAPS, no truncar."""
        out = truncate_tool_output_for_synthesis("calendar_ahead", '[{"a":1}]')
        assert out == '[{"a":1}]'

    def test_invalid_json_passthru(self):
        out = truncate_tool_output_for_synthesis("gmail_recent", "not json {")
        assert out == "not json {"

    def test_empty_passthru(self):
        assert truncate_tool_output_for_synthesis("gmail_recent", "") == ""

    def test_dict_with_lists_truncated(self):
        """gmail_recent shape: dict con sub-listas. Truncar cada una."""
        items = [{"id": i, "subject": f"mail {i}"} for i in range(20)]
        raw = json.dumps({"awaiting_reply": items, "starred": items[:3]})
        out = truncate_tool_output_for_synthesis("gmail_recent", raw)
        parsed = json.loads(out)
        cap = _DEFAULT_CAPS["gmail_recent"]
        # awaiting_reply truncado a cap, starred sin tocar (under cap)
        assert len(parsed["awaiting_reply"]) == cap
        assert len(parsed["starred"]) == 3
        # Hay metadata
        assert "_truncated" in parsed
        assert parsed["_truncated"]["awaiting_reply"]["total"] == 20
        assert parsed["_truncated"]["awaiting_reply"]["dropped"] == 20 - cap

    def test_flat_list_truncated_with_meta(self):
        """whatsapp_search shape: lista plana. Truncar y wrap en dict."""
        items = [{"text": f"msg {i}"} for i in range(15)]
        raw = json.dumps(items)
        out = truncate_tool_output_for_synthesis("whatsapp_search", raw)
        parsed = json.loads(out)
        cap = _DEFAULT_CAPS["whatsapp_search"]
        assert "items" in parsed
        assert len(parsed["items"]) == cap
        assert parsed["_truncated_total"] == 15
        assert parsed["_kept"] == cap

    def test_under_cap_unchanged(self):
        """Si la lista está bajo el cap, devolver el output original."""
        items = [{"text": "msg"}] * 3
        raw = json.dumps(items)
        out = truncate_tool_output_for_synthesis("whatsapp_search", raw)
        assert out == raw  # passthru, no metadata wrapper

    def test_max_items_override(self):
        items = [{"id": i} for i in range(10)]
        raw = json.dumps(items)
        out = truncate_tool_output_for_synthesis("whatsapp_search", raw, max_items=2)
        parsed = json.loads(out)
        assert parsed["_kept"] == 2
        assert len(parsed["items"]) == 2

    def test_drive_search_dict_shape(self):
        """drive_search puede devolver dict con files inside."""
        files = [{"name": f"file{i}.md", "body": "x" * 1000} for i in range(15)]
        raw = json.dumps({"files": files, "query": "test"})
        out = truncate_tool_output_for_synthesis("drive_search", raw)
        parsed = json.loads(out)
        cap = _DEFAULT_CAPS["drive_search"]
        assert len(parsed["files"]) == cap
        assert parsed["query"] == "test"  # otros campos se preservan
