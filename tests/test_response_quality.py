"""Tests for three LLM response-quality features:

1. system_prompt_for_intent — pure routing function
2. Citation-repair loop — fires when verify_citations() returns bad citations;
   replaces `full` only when repair succeeds
3. --critique flag — second non-streaming ollama.chat pass that may replace `full`

Tests for features 2 and 3 invoke the `query` Click command via CliRunner with
the minimal set of monkeypatches needed to reach the repair/critique block:
  - warmup_async (no-op)
  - get_db → fake collection with count() == 1
  - get_vocabulary → ([], [])
  - classify_intent → ("semantic", {})
  - retrieve → synthetic result dict
  - build_progressive_context → fixed context string
  - user_prompt_block → ""
  - ollama.chat → controlled via per-test stub
  - verify_citations → controlled via per-test stub
  - log_query_event → captured via per-test stub
  - print_sources, find_related, render_related, new_turn_id → no-ops
"""
import pytest
from click.testing import CliRunner

import rag


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCollection:
    """Minimal stand-in for SqliteVecCollection — only .count() is needed."""
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
    """Patch everything except ollama.chat / verify_citations / log_query_event."""
    if retrieve_result is None:
        retrieve_result = _fake_retrieve_result()

    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCollection())
    monkeypatch.setattr(rag, "get_vocabulary", lambda col: ([], []))
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("semantic", {}))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: retrieve_result)
    monkeypatch.setattr(rag, "build_progressive_context", lambda *a, **kw: "CONTEXTO_FAKE")
    monkeypatch.setattr(rag, "user_prompt_block", lambda: "")
    monkeypatch.setattr(rag, "print_sources", lambda result: None)
    monkeypatch.setattr(rag, "find_related", lambda col, metas: [])
    monkeypatch.setattr(rag, "render_related", lambda related: None)
    monkeypatch.setattr(rag, "new_turn_id", lambda: "turn-1")
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "fake-model")


# ── 1. system_prompt_for_intent ───────────────────────────────────────────────


@pytest.mark.parametrize("intent", [
    "semantic", "count", "list", "recent", "synthesis", "comparison", "unknown",
])
def test_system_prompt_loose_always_returns_system_rules(intent):
    result = rag.system_prompt_for_intent(intent, loose=True)
    assert result == rag.SYSTEM_RULES


@pytest.mark.parametrize("intent,expected_attr", [
    ("count",      "SYSTEM_RULES_LOOKUP"),
    ("list",       "SYSTEM_RULES_LOOKUP"),
    ("recent",     "SYSTEM_RULES_LOOKUP"),
    ("synthesis",  "SYSTEM_RULES_SYNTHESIS"),
    ("comparison", "SYSTEM_RULES_COMPARISON"),
    ("semantic",   "SYSTEM_RULES_STRICT"),
    ("unknown",    "SYSTEM_RULES_STRICT"),
    ("",           "SYSTEM_RULES_STRICT"),
])
def test_system_prompt_strict_routes_by_intent(intent, expected_attr):
    # Post 2026-04-22: system_prompt_for_intent resuelve via load_prompt
    # cada call (para que env overrides apliquen en runtime). Igualdad
    # de contenido en vez de identidad.
    result = rag.system_prompt_for_intent(intent, loose=False)
    expected = getattr(rag, expected_attr)
    assert result == expected


# ── 2. Citation-repair loop ───────────────────────────────────────────────────


def test_citation_repair_replaces_full_on_success(monkeypatch):
    """When the initial answer has bad citations but the repair returns a valid
    answer with no bad citations, `full` is replaced and citation_repaired=True
    is logged."""
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    # Answers must not contain markdown links — output goes through
    # convert_obsidian_links() which transforms [label](path) into
    # "label: obsidian://..." making raw-link assertions fragile.
    INITIAL_ANSWER = "Respuesta inicial con cita incorrecta."
    REPAIRED_ANSWER = "Respuesta reparada con cita correcta."

    # Use two separate callables to avoid the Python generator issue: a function
    # that contains `yield` is always a generator, even if the yield branch is
    # not reached. We swap the attribute between the two calls.
    call_state = {"count": 0}

    def fake_streaming(model, messages, options, stream, keep_alive):
        call_state["count"] += 1
        yield _FakeResponse(INITIAL_ANSWER)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        call_state["count"] += 1
        return _FakeResponse(REPAIRED_ANSWER)

    # First call is streaming (generation); second is non-streaming (repair).
    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}
    _call_modes: list[bool] = []  # records stream kwarg per call

    def fake_chat(model, messages, options, stream, keep_alive):
        _call_modes.append(stream)
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    # verify_citations: first call (on initial) → bad; second (on repair) → clean.
    verify_call = {"count": 0}
    def fake_verify(text, metas):
        verify_call["count"] += 1
        if verify_call["count"] == 1:
            return [("Fake", "99-nonexistent.md")]
        return []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert REPAIRED_ANSWER in result.output
    assert INITIAL_ANSWER not in result.output
    assert logged.get("citation_repaired") is True
    assert logged.get("bad_citations") == ["99-nonexistent.md"]
    # Repair call MUST be non-streaming — citation-repair relies on the
    # full answer coming back in one shot so we can diff/replace. Regression
    # would silently break the repair path (generator branch returns raw
    # chunks instead of a single response).
    assert len(_call_modes) == 2, f"expected 2 chat calls, got {len(_call_modes)}"
    assert _call_modes[0] is True, "generation call must be streaming"
    assert _call_modes[1] is False, "repair call must be non-streaming"


def test_citation_repair_keeps_original_when_repair_also_bad(monkeypatch):
    """When the repair call itself still has bad citations, original answer is kept
    and citation_repaired remains False."""
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    INITIAL_ANSWER = "Respuesta original con cita incorrecta."
    STILL_BAD_ANSWER = "Respuesta reparada también incorrecta."

    def fake_streaming(model, messages, options, stream, keep_alive):
        yield _FakeResponse(INITIAL_ANSWER)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        return _FakeResponse(STILL_BAD_ANSWER)

    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    # Both verify calls return bad citations — repair doesn't fix it.
    def fake_verify(text, metas):
        return [("Fake", "99-nonexistent.md")]

    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert INITIAL_ANSWER in result.output
    assert STILL_BAD_ANSWER not in result.output
    assert logged.get("citation_repaired") is False


def test_citation_repair_keeps_original_when_repair_returns_empty(monkeypatch):
    """When the repair call returns an empty string, original is kept."""
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    INITIAL_ANSWER = "Respuesta inicial sin citations válidas."

    def fake_streaming(model, messages, options, stream, keep_alive):
        yield _FakeResponse(INITIAL_ANSWER)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        return _FakeResponse("")  # empty repair — should not replace

    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    verify_call = {"count": 0}
    def fake_verify(text, metas):
        verify_call["count"] += 1
        if verify_call["count"] == 1:
            return [("Fake", "99-nonexistent.md")]
        return []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert INITIAL_ANSWER in result.output
    assert logged.get("citation_repaired") is False


def test_citation_repair_skipped_when_too_many_bad_citations(monkeypatch):
    """Perf gate: when the initial answer has > `_CITATION_REPAIR_MAX_BAD`
    bad citations, the repair call is NOT made (the response is likely
    hallucinated throughout and a single-shot repair won't salvage it —
    not worth the 5-8s round-trip). Original answer is kept, repair is
    skipped, and only the streaming generation call happens."""
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    INITIAL_ANSWER = "Respuesta con muchas citas inventadas."

    call_modes: list[bool] = []

    def fake_chat(model, messages, options, stream, keep_alive):
        call_modes.append(stream)
        if stream:
            def gen():
                yield _FakeResponse(INITIAL_ANSWER)
            return gen()
        return _FakeResponse("should_not_be_called")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    # 5 bad citations > default _CITATION_REPAIR_MAX_BAD of 3 → skip repair.
    many_bad = [
        ("Fake1", "99-a.md"),
        ("Fake2", "99-b.md"),
        ("Fake3", "99-c.md"),
        ("Fake4", "99-d.md"),
        ("Fake5", "99-e.md"),
    ]
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: many_bad)

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert INITIAL_ANSWER in result.output
    # Only the streaming generation call — NO repair call was made.
    assert call_modes == [True], f"expected only streaming call, got {call_modes}"
    assert logged.get("citation_repaired") is False


def test_citation_repair_fires_for_few_bad_citations(monkeypatch):
    """Repair still fires when bad citations are ≤ _CITATION_REPAIR_MAX_BAD.
    Sanity regression: the perf gate must not kill the normal repair path."""
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    INITIAL = "Respuesta inicial con cita mala."
    REPAIRED = "Respuesta reparada con cita buena."

    def fake_streaming(model, messages, options, stream, keep_alive):
        yield _FakeResponse(INITIAL)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        return _FakeResponse(REPAIRED)

    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)

    # 2 bad citations ≤ 3 (default threshold) → repair fires normally.
    two_bad = [("Fake1", "99-a.md"), ("Fake2", "99-b.md")]
    verify_call = {"count": 0}
    def fake_verify(text, metas):
        verify_call["count"] += 1
        return two_bad if verify_call["count"] == 1 else []

    monkeypatch.setattr(rag, "verify_citations", fake_verify)

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert REPAIRED in result.output
    assert logged.get("citation_repaired") is True


def test_citation_repair_bad_threshold_env_override(monkeypatch):
    """`RAG_CITATION_REPAIR_MAX_BAD=0` disables repair entirely.

    Confirms the env override path works; does NOT assert a specific count
    beyond "one bad citation is enough to skip when threshold is 0".
    """
    monkeypatch.setattr(rag, "_CITATION_REPAIR_MAX_BAD", 0)
    _apply_base_patches(monkeypatch, _fake_retrieve_result("01-Projects/nota.md"))

    INITIAL = "Respuesta original."

    call_modes: list[bool] = []

    def fake_chat(model, messages, options, stream, keep_alive):
        call_modes.append(stream)
        if stream:
            def gen():
                yield _FakeResponse(INITIAL)
            return gen()
        return _FakeResponse("SHOULD_NOT_FIRE")

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [("Fake", "99-a.md")])

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert INITIAL in result.output
    assert call_modes == [True], f"expected only streaming call, got {call_modes}"
    assert logged.get("citation_repaired") is False


# ── 3. --critique flag ────────────────────────────────────────────────────────


def test_critique_off_by_default_no_second_call(monkeypatch):
    """Without --critique, only the streaming generation call is made.
    critique_fired=False and critique_changed=False are logged."""
    _apply_base_patches(monkeypatch)

    ANSWER = "Respuesta sin critique."
    call_log = []

    def fake_streaming(model, messages, options, stream, keep_alive):
        call_log.append("streaming")
        yield _FakeResponse(ANSWER)

    monkeypatch.setattr(rag.ollama, "chat", fake_streaming)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test question"])

    assert result.exit_code == 0, result.output
    assert call_log == ["streaming"], f"Unexpected extra calls: {call_log}"
    assert ANSWER in result.output
    assert logged.get("critique_fired") is False
    assert logged.get("critique_changed") is False


def test_critique_replaces_answer_when_different(monkeypatch):
    """With --critique and a different critique output, full is replaced,
    critique_changed=True and critique_fired=True are logged."""
    _apply_base_patches(monkeypatch)

    ORIGINAL = "Respuesta original que el critique mejorará."
    CRITIQUE_IMPROVED = "Respuesta mejorada sustancialmente por el critique."

    def fake_streaming(model, messages, options, stream, keep_alive):
        yield _FakeResponse(ORIGINAL)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        return _FakeResponse(CRITIQUE_IMPROVED)

    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--critique", "test question"])

    assert result.exit_code == 0, result.output
    assert CRITIQUE_IMPROVED in result.output
    assert ORIGINAL not in result.output
    assert logged.get("critique_fired") is True
    assert logged.get("critique_changed") is True


def test_critique_keeps_original_when_same(monkeypatch):
    """With --critique, if critique output is the same (modulo whitespace),
    full is unchanged, critique_changed=False, critique_fired=True."""
    _apply_base_patches(monkeypatch)

    ORIGINAL = "Respuesta sin cambios."
    # Same content but with extra surrounding whitespace — normalization should
    # make these equal so critique_changed stays False.
    CRITIQUE_SAME = "  Respuesta sin cambios.  "

    def fake_streaming(model, messages, options, stream, keep_alive):
        yield _FakeResponse(ORIGINAL)

    def fake_nonstreaming(model, messages, options, stream, keep_alive):
        return _FakeResponse(CRITIQUE_SAME)

    _call_seq = [fake_streaming, fake_nonstreaming]
    _call_idx = {"i": 0}

    def fake_chat(model, messages, options, stream, keep_alive):
        fn = _call_seq[_call_idx["i"]]
        _call_idx["i"] += 1
        return fn(model, messages, options, stream, keep_alive)

    monkeypatch.setattr(rag.ollama, "chat", fake_chat)
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])

    logged = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "--critique", "test question"])

    assert result.exit_code == 0, result.output
    assert ORIGINAL in result.output
    assert logged.get("critique_fired") is True
    assert logged.get("critique_changed") is False
