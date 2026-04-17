"""Tests for log_behavior_event and BEHAVIOR_LOG_PATH (Task 1 symbols)."""
import json
from pathlib import Path

import pytest
import rag


@pytest.fixture
def tmp_behavior_log(tmp_path, monkeypatch):
    bl = tmp_path / "behavior.jsonl"
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", bl)
    return bl


def _flush():
    """Drain the async log queue so writes land synchronously in tests."""
    import queue as _q
    try:
        rag._LOG_QUEUE.join()
    except Exception:
        pass


def test_log_behavior_event_schema(tmp_behavior_log):
    rag.log_behavior_event({"source": "brief", "event": "kept", "path": "02-Areas/Foo.md"})
    _flush()
    lines = [json.loads(l) for l in tmp_behavior_log.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    ev = lines[0]
    assert ev["source"] == "brief"
    assert ev["event"] == "kept"
    assert ev["path"] == "02-Areas/Foo.md"
    assert "ts" in ev


def test_log_behavior_event_no_raise(monkeypatch):
    # Even if the path is bogus, must not raise
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", Path("/no/such/dir/behavior.jsonl"))
    rag.log_behavior_event({"source": "test", "event": "noop"})


def test_behavior_log_path_constant():
    assert str(rag.BEHAVIOR_LOG_PATH).endswith("behavior.jsonl")
    assert "obsidian-rag" in str(rag.BEHAVIOR_LOG_PATH)
