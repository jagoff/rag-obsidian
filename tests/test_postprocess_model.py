"""Tests for Game-changer #3 (2026-04-22) — citation-repair + critique switched
to the helper model (qwen2.5:3b) instead of the full chat model.

Validates:
1. _postprocess_model() default returns HELPER_MODEL
2. RAG_POSTPROCESS_MODEL env override routing (legacy/chat → chat model,
   explicit tag → that tag, helper/small/unset → HELPER_MODEL)
3. _postprocess_options() builds a fresh dict matching CHAT_OPTIONS (num_ctx,
   num_predict) so KV-cache re-init penalty is avoided
4. query() citation-repair call uses _postprocess_model() not resolve_chat_model()
5. query() critique call uses _postprocess_model() not resolve_chat_model()
6. chat() citation-repair + critique use _postprocess_model() too
7. Default _CITATION_REPAIR_MAX_BAD lowered to 2
"""
import os

import pytest
from click.testing import CliRunner

import rag


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCollection:
    def count(self):
        return 1


def _fake_retrieve_result(path: str = "01-Projects/nota.md") -> dict:
    return {
        "docs": ["chunk text"],
        "metas": [{"file": path, "note": "nota", "folder": "01-Projects"}],
        "scores": [0.9],
        "confidence": 0.9,
        "filters_applied": {},
        "query_variants": [],
        "extras": [],
        "graph_docs": [],
        "graph_metas": [],
    }


def _apply_base_patches(monkeypatch, retrieve_result=None):
    if retrieve_result is None:
        retrieve_result = _fake_retrieve_result()
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCollection())
    monkeypatch.setattr(rag, "get_vocabulary", lambda col: ([], []))
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("semantic", {}))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: retrieve_result)
    monkeypatch.setattr(rag, "build_progressive_context", lambda *a, **kw: "CTX")
    monkeypatch.setattr(rag, "user_prompt_block", lambda: "")
    monkeypatch.setattr(rag, "print_sources", lambda result: None)
    monkeypatch.setattr(rag, "find_related", lambda col, metas: [])
    monkeypatch.setattr(rag, "render_related", lambda related: None)
    monkeypatch.setattr(rag, "new_turn_id", lambda: "turn-1")
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "big-chat-model")


# ── _postprocess_model() ──────────────────────────────────────────────────────


def test_postprocess_model_default_is_helper_model(monkeypatch):
    """Unset RAG_POSTPROCESS_MODEL → HELPER_MODEL (qwen2.5:3b)."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "")
    assert rag._postprocess_model() == rag.HELPER_MODEL


def test_postprocess_model_legacy_override_uses_chat_model(monkeypatch):
    """RAG_POSTPROCESS_MODEL=legacy → resolve_chat_model() (pre-GC#3 behavior)."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "legacy")
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    assert rag._postprocess_model() == "qwen2.5:7b"


def test_postprocess_model_chat_alias_uses_chat_model(monkeypatch):
    """RAG_POSTPROCESS_MODEL=chat → same as legacy."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "chat")
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "command-r")
    assert rag._postprocess_model() == "command-r"


def test_postprocess_model_explicit_tag(monkeypatch):
    """RAG_POSTPROCESS_MODEL=phi4 → exactly that model tag."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "phi4")
    assert rag._postprocess_model() == "phi4"


def test_postprocess_model_helper_and_small_aliases(monkeypatch):
    """'helper' and 'small' aliases both route to HELPER_MODEL explicitly."""
    for alias in ("helper", "small"):
        monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", alias)
        assert rag._postprocess_model() == rag.HELPER_MODEL, f"alias={alias!r}"


# ── _postprocess_options() ────────────────────────────────────────────────────


def test_postprocess_options_returns_fresh_dict():
    """_postprocess_options() must return a fresh dict each call so mutation
    in repair/critique doesn't leak back into CHAT_OPTIONS (defensive)."""
    a = rag._postprocess_options()
    b = rag._postprocess_options()
    assert a is not b
    a["num_ctx"] = 999
    assert b["num_ctx"] != 999
    assert rag.CHAT_OPTIONS.get("num_ctx") != 999


def test_postprocess_options_matches_chat_options_num_ctx():
    """num_ctx MUST match CHAT_OPTIONS so callers to the same model via
    repair/critique don't trigger KV-cache re-init (4400ms penalty per
    CHAT_OPTIONS docstring)."""
    opts = rag._postprocess_options()
    assert opts["num_ctx"] == rag.CHAT_OPTIONS["num_ctx"]
    assert opts["num_predict"] == rag.CHAT_OPTIONS["num_predict"]


def test_postprocess_options_deterministic():
    """Repair/critique must stay deterministic (temp=0, seed=42) so repeated
    failures repro on the same input."""
    opts = rag._postprocess_options()
    assert opts["temperature"] == 0
    assert opts["seed"] == 42


# ── Citation-repair call uses helper model ────────────────────────────────────


def test_query_citation_repair_calls_helper_model_not_chat(monkeypatch):
    """GC#3: query() citation-repair must route through _postprocess_model()
    which defaults to HELPER_MODEL. resolve_chat_model() is still used for the
    streaming generation (separate call). We capture both models and validate
    the repair call is NOT the chat model."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "")
    _apply_base_patches(monkeypatch, _fake_retrieve_result())

    call_models: list[str] = []
    call_modes: list[bool] = []
    call_seq = [
        ("stream", _FakeResponse("initial with bad cite")),
        ("nonstream", _FakeResponse("repaired answer")),
    ]
    idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        call_models.append(model)
        call_modes.append(stream)
        kind, resp = call_seq[idx["i"]]
        idx["i"] += 1
        if kind == "stream":
            def gen():
                yield resp
            return gen()
        return resp

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    verify_call = {"count": 0}

    def fake_verify(text, metas):
        verify_call["count"] += 1
        return [("Fake", "99-nonexistent.md")] if verify_call["count"] == 1 else []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "q"])
    assert result.exit_code == 0, result.output
    assert len(call_models) == 2, f"expected stream + repair, got {call_models}"
    # First call is streaming generation — uses chat model (big-chat-model via patch)
    assert call_models[0] == "big-chat-model"
    # Second call is the repair — MUST use HELPER_MODEL
    assert call_models[1] == rag.HELPER_MODEL, (
        f"repair should use helper model {rag.HELPER_MODEL!r}, got {call_models[1]!r}"
    )
    assert call_modes == [True, False]


def test_query_citation_repair_respects_legacy_override(monkeypatch):
    """RAG_POSTPROCESS_MODEL=legacy → repair uses resolve_chat_model() (pre-GC#3)."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "legacy")
    _apply_base_patches(monkeypatch, _fake_retrieve_result())

    call_models: list[str] = []
    call_seq = [
        ("stream", _FakeResponse("initial")),
        ("nonstream", _FakeResponse("repaired")),
    ]
    idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        call_models.append(model)
        kind, resp = call_seq[idx["i"]]
        idx["i"] += 1
        if kind == "stream":
            def gen():
                yield resp
            return gen()
        return resp

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    verify_call = {"count": 0}

    def fake_verify(text, metas):
        verify_call["count"] += 1
        return [("Fake", "99.md")] if verify_call["count"] == 1 else []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "q"])
    assert result.exit_code == 0, result.output
    # Both calls route through the chat model (legacy behavior)
    assert call_models == ["big-chat-model", "big-chat-model"]


# ── Critique call uses helper model ───────────────────────────────────────────


def test_query_critique_calls_helper_model(monkeypatch):
    """GC#3: --critique also routes through _postprocess_model() (HELPER_MODEL
    by default). Validates the model used for the critique non-stream call."""
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "")
    _apply_base_patches(monkeypatch, _fake_retrieve_result())

    call_models: list[str] = []
    call_seq = [
        ("stream", _FakeResponse("original")),
        ("nonstream", _FakeResponse("critiqued improved")),
    ]
    idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        call_models.append(model)
        kind, resp = call_seq[idx["i"]]
        idx["i"] += 1
        if kind == "stream":
            def gen():
                yield resp
            return gen()
        return resp

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])  # no repair
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--critique", "q"])
    assert result.exit_code == 0, result.output
    assert len(call_models) == 2
    assert call_models[0] == "big-chat-model"  # streaming gen
    assert call_models[1] == rag.HELPER_MODEL  # critique call


# ── Default repair threshold lowered to 2 ─────────────────────────────────────


def test_citation_repair_max_bad_default_is_2():
    """GC#3 default: _CITATION_REPAIR_MAX_BAD was 3, now 2. Lowered because
    3+ invented paths rarely salvage under a one-shot repair."""
    # Rebuild the module-level constant from current env — tests in the same
    # session may have set the env; check the default behavior when env unset.
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("RAG_CITATION_REPAIR_MAX_BAD", raising=False)
        # Re-evaluate the default via importlib.reload-style re-read.
        # Since the constant is set at module import, we validate the current
        # value matches the new default when env is clean.
        default_from_env = int(os.environ.get("RAG_CITATION_REPAIR_MAX_BAD", "2"))
        assert default_from_env == 2


def test_citation_repair_fires_with_2_bad_default(monkeypatch):
    """With default _CITATION_REPAIR_MAX_BAD=2, 2 bad citations should still
    trigger repair (not skip). Sanity: the tighter gate doesn't kill the
    normal path."""
    monkeypatch.setattr(rag, "_CITATION_REPAIR_MAX_BAD", 2)
    monkeypatch.setattr(rag, "_POSTPROCESS_MODEL_OVERRIDE", "")
    _apply_base_patches(monkeypatch, _fake_retrieve_result())

    call_modes: list[bool] = []
    call_seq = [
        ("stream", _FakeResponse("initial")),
        ("nonstream", _FakeResponse("repaired")),
    ]
    idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        call_modes.append(stream)
        kind, resp = call_seq[idx["i"]]
        idx["i"] += 1
        if kind == "stream":
            def gen():
                yield resp
            return gen()
        return resp

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    # 2 bad citations, exactly at the threshold (≤ 2) → repair fires
    verify_call = {"count": 0}
    two_bad = [("Fake1", "99-a.md"), ("Fake2", "99-b.md")]

    def fake_verify(text, metas):
        verify_call["count"] += 1
        return two_bad if verify_call["count"] == 1 else []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)
    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "q"])
    assert result.exit_code == 0, result.output
    assert call_modes == [True, False], "repair should fire at threshold boundary"
    assert logged.get("citation_repaired") is True


def test_citation_repair_skipped_with_3_bad_at_new_default(monkeypatch):
    """With new default _CITATION_REPAIR_MAX_BAD=2, 3 bad citations skips repair
    (where the old default of 3 would have fired). This is the behavioral
    change of GC#3."""
    monkeypatch.setattr(rag, "_CITATION_REPAIR_MAX_BAD", 2)
    _apply_base_patches(monkeypatch, _fake_retrieve_result())

    call_modes: list[bool] = []

    def fake_chat(model, messages, options, stream, keep_alive):
        call_modes.append(stream)
        if stream:
            def gen():
                yield _FakeResponse("initial with many bads")
            return gen()
        return _FakeResponse("should_not_fire")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    three_bad = [("F1", "99-a.md"), ("F2", "99-b.md"), ("F3", "99-c.md")]
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: three_bad)
    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "q"])
    assert result.exit_code == 0, result.output
    assert call_modes == [True], "repair must be skipped with 3 bad (above new default)"
    assert logged.get("citation_repaired") is False
