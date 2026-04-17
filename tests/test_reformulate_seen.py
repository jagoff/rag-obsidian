"""Tests for `reformulate_query(seen_titles=...)` and `_titles_from_paths`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_titles_from_paths_basic():
    import rag
    out = rag._titles_from_paths([
        "02-Areas/Coaching/Ikigai.md",
        "03-Resources/CNV.md",
    ])
    assert out == ["Ikigai", "CNV"]


def test_titles_from_paths_dedupes_and_preserves_order():
    import rag
    out = rag._titles_from_paths([
        "02-Areas/Coaching/Ikigai.md",
        "04-Archive/Ikigai.md",  # same stem → dedup
        "03-Resources/CNV.md",
    ])
    assert out == ["Ikigai", "CNV"]


def test_titles_from_paths_handles_empty_and_none():
    import rag
    assert rag._titles_from_paths(None) == []
    assert rag._titles_from_paths([]) == []
    assert rag._titles_from_paths(["", ""]) == []


def test_titles_from_paths_respects_limit():
    import rag
    paths = [f"a/{i}.md" for i in range(20)]
    out = rag._titles_from_paths(paths, limit=3)
    assert len(out) == 3
    assert out == ["0", "1", "2"]


def test_reformulate_no_history_no_summary_returns_input(monkeypatch):
    import rag
    # Should short-circuit without calling the LLM
    called = {"n": 0}

    def _fake_chat(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    out = rag.reformulate_query("qué es X", [], summary=None)
    assert out == "qué es X"
    assert called["n"] == 0


def test_reformulate_accepts_seen_titles_kwarg(monkeypatch):
    """Regression: seen_titles kwarg must be accepted without error.

    Current behavior: kwarg is scaffolding only — NOT injected into the
    prompt (the 2026-04-17 attempt regressed chains by -16pp hit@5 and
    -33pp chain_success; reverted but kwarg kept for future iteration).
    """
    import rag

    class _Msg:
        def __init__(self, c): self.content = c

    class _Resp:
        def __init__(self, c): self.message = _Msg(c)

    captured = {"prompt": None}

    def _fake_chat(model, messages, **kwargs):
        captured["prompt"] = messages[0]["content"]
        return _Resp("out")

    monkeypatch.setattr(rag.ollama, "chat", _fake_chat)
    monkeypatch.setattr(rag, "_postprocess_reformulation",
                        lambda q, raw, hist: raw)
    history = [{"role": "user", "content": "x"}]
    # Must not raise even with seen_titles provided
    rag.reformulate_query("y?", history, seen_titles=["Ikigai", "CNV"])
    # Current contract: seen_titles are NOT injected in the prompt
    assert "Ikigai" not in captured["prompt"]
    assert "CNV" not in captured["prompt"]
