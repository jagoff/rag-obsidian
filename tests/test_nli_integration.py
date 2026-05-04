"""Integration tests for NLI grounding in query() and chat() CLI paths.

All tests mock split_claims + ground_claims_nli — no real mDeBERTa (400 MB
MPS fp32) loaded in CI. Env gate: RAG_NLI_GROUNDING. Pattern follows
test_response_quality.py (CliRunner for CLI surface) + test_nli_model_load.py
(direct module attribute mocking).

Coverage:
  · Query integration — 12 cases
  · Chat integration  — 4 cases
  · Exception safety  — 4 cases
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner

import rag


# ── Shared test data ───────────────────────────────────────────────────────────


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.message = _FakeMsg(content)


class _FakeCollection:
    def count(self) -> int:
        return 1


def _fake_retrieve_result(fast_path: bool = False) -> dict:
    return {
        "docs": ["chunk text del vault"],
        "metas": [{"file": "01-Projects/nota.md", "note": "nota", "folder": "01-Projects"}],
        "scores": [0.9],
        "confidence": 0.9,
        "filters_applied": {},
        "query_variants": [],
        "extras": [],
        "graph_docs": [],
        "graph_metas": [],
        "timing": {},
        "fast_path": fast_path,
    }


def _fake_grounding_result(
    claims_total: int = 3,
    supported: int = 2,
    contradicted: int = 1,
    neutral: int = 0,
) -> rag.GroundingResult:
    claims = [
        *[rag.ClaimGrounding(text=f"Supported {i}", verdict="entails") for i in range(supported)],
        *[rag.ClaimGrounding(text=f"Contradicted {i}", verdict="contradicts") for i in range(contradicted)],
        *[rag.ClaimGrounding(text=f"Neutral {i}", verdict="neutral") for i in range(neutral)],
    ]
    return rag.GroundingResult(
        claims=claims,
        claims_total=claims_total,
        claims_supported=supported,
        claims_contradicted=contradicted,
        claims_neutral=neutral,
        nli_ms=42,
    )


def _fake_claims(n: int = 3) -> list:
    return [rag.Claim(text=f"Claim {i} is long enough text") for i in range(n)]


def _streaming_fn(answer: str):
    """Factory: returns a fake ollama.chat that yields one streaming chunk."""
    def _fake(model, messages, options, stream, keep_alive):
        yield _FakeChunk(answer)
    return _fake


# ── Base patch helpers ─────────────────────────────────────────────────────────


def _apply_base_patches_query(monkeypatch, retrieve_result=None):
    """Patch query() dependencies — excludes split_claims + ground_claims_nli."""
    if retrieve_result is None:
        retrieve_result = _fake_retrieve_result()
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCollection())
    monkeypatch.setattr(rag, "get_vocabulary", lambda col: ([], []))
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("semantic", {}))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: retrieve_result)
    monkeypatch.setattr(rag, "build_progressive_context", lambda *a, **kw: "CONTEXTO_FAKE")
    monkeypatch.setattr(rag, "user_prompt_block", lambda: "")
    monkeypatch.setattr(rag, "build_person_context", lambda q, vault_root=None: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "fake-model")
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])
    monkeypatch.setattr(rag, "print_sources", lambda result: None)
    monkeypatch.setattr(rag, "find_related", lambda col, metas: [])
    monkeypatch.setattr(rag, "render_related", lambda related: None)
    monkeypatch.setattr(rag, "new_turn_id", lambda: "turn-1")
    monkeypatch.setattr(rag.ollama, "chat", _streaming_fn("Respuesta del vault."))


def _apply_base_patches_chat(monkeypatch, tmp_path: Path, retrieve_result=None):
    """Patch chat() dependencies — excludes split_claims + ground_claims_nli."""
    if retrieve_result is None:
        retrieve_result = _fake_retrieve_result()
    fake_sess = {"id": "test-session-id", "turns": [], "mode": "chat"}
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCollection())
    monkeypatch.setattr(rag, "resolve_vault_paths", lambda names: [("test", tmp_path)])
    monkeypatch.setattr(rag, "get_db_for", lambda path: _FakeCollection())
    monkeypatch.setattr(rag, "ensure_session", lambda sid, mode: fake_sess)
    monkeypatch.setattr(rag, "session_history", lambda sess, window=None: [])
    monkeypatch.setattr(rag, "session_summary", lambda sess: "")
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: retrieve_result)
    monkeypatch.setattr(rag, "build_progressive_context", lambda *a, **kw: "CONTEXTO_FAKE")
    monkeypatch.setattr(rag, "user_prompt_block", lambda: "")
    monkeypatch.setattr(rag, "build_person_context", lambda q, vault_root=None: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "fake-model")
    monkeypatch.setattr(rag, "verify_citations", lambda text, metas: [])
    monkeypatch.setattr(rag, "append_turn", lambda sess, turn: None)
    monkeypatch.setattr(rag, "save_session", lambda sess: None)
    monkeypatch.setattr(rag, "print_sources", lambda result: None)
    monkeypatch.setattr(rag, "find_related", lambda col, metas: [])
    monkeypatch.setattr(rag, "render_related", lambda related: None)
    monkeypatch.setattr(rag, "new_turn_id", lambda: "turn-1")
    monkeypatch.setattr(rag, "detect_link_intent", lambda text: (False, None))
    monkeypatch.setattr(rag, "detect_reindex_intent", lambda text: (False, False))
    monkeypatch.setattr(rag, "detect_save_intent", lambda text: (False, None))
    monkeypatch.setattr(rag, "_detect_propose_intent", lambda q: False)
    monkeypatch.setattr(
        rag, "auto_index_vault",
        lambda vpath: {"indexed": 0, "took_ms": 0, "kind": "incremental"},
    )
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)
    monkeypatch.setattr(rag.ollama, "chat", _streaming_fn("Respuesta del vault."))


# ── Query integration: 12 cases ────────────────────────────────────────────────


def test_query_nli_enabled_calls_split_and_ground(monkeypatch):
    """Test 1: RAG_NLI_GROUNDING=1 + semantic intent + 3 claims → both functions called."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    split_calls: list = []
    ground_calls: list = []

    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(3))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda claims, docs, metas, **kw:
                        (ground_calls.append(claims), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "que dice el vault"])

    assert result.exit_code == 0, result.output
    assert len(split_calls) == 1
    assert len(ground_calls) == 1
    assert len(ground_calls[0]) == 3


def test_query_nli_disabled_by_default_skips_grounding(monkeypatch):
    """Test 2: RAG_NLI_GROUNDING unset (default OFF) → neither split_claims nor
    ground_claims_nli called."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    _apply_base_patches_query(monkeypatch)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(2))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "que dice el vault"])

    assert result.exit_code == 0, result.output
    assert split_calls == [], "split_claims must not be called when NLI is OFF"
    assert ground_calls == [], "ground_claims_nli must not be called when NLI is OFF"
    assert logged.get("grounding_summary") is None


def test_query_nli_skips_for_count_intent(monkeypatch):
    """Test 3: RAG_NLI_GROUNDING=1 + intent=count → early return, neither NLI fn called."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("count", {}))
    monkeypatch.setattr(rag, "handle_count", lambda col, params: (0, []))

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(1))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "cuantas notas"])

    assert result.exit_code == 0, result.output
    assert split_calls == [], "split_claims must not run for count intent"
    assert ground_calls == [], "ground_claims_nli must not run for count intent"


def test_query_nli_skips_for_list_intent(monkeypatch):
    """Test 4: RAG_NLI_GROUNDING=1 + intent=list → early return, NLI skipped."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("list", {}))
    monkeypatch.setattr(rag, "handle_list", lambda col, params, limit=50: [])
    monkeypatch.setattr(rag, "render_file_list", lambda title, files: None)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(1))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "listame las notas"])

    assert result.exit_code == 0, result.output
    assert split_calls == []
    assert ground_calls == []


def test_query_nli_skips_for_recent_intent(monkeypatch):
    """Test 5: RAG_NLI_GROUNDING=1 + intent=recent → early return, NLI skipped."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("recent", {}))
    monkeypatch.setattr(rag, "handle_recent", lambda col, params, limit=20: [])
    monkeypatch.setattr(rag, "render_file_list", lambda title, files: None)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(1))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "ultimas notas"])

    assert result.exit_code == 0, result.output
    assert split_calls == []
    assert ground_calls == []


def test_query_nli_skips_for_agenda_intent(monkeypatch):
    """Test 6: RAG_NLI_GROUNDING=1 + intent=agenda → early return, NLI skipped."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag, "classify_intent", lambda q, tags, folders: ("agenda", {}))
    monkeypatch.setattr(rag, "handle_agenda",
                        lambda col, params, limit=20, question=None: [])
    monkeypatch.setattr(rag, "render_file_list", lambda title, files: None)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(1))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])
    monkeypatch.setattr(rag, "log_query_event", lambda ev: None)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "mi agenda"])

    assert result.exit_code == 0, result.output
    assert split_calls == []
    assert ground_calls == []


def test_synthesis_not_in_skip_intents_and_grounding_runs(monkeypatch):
    """Test 7: 'synthesis' is NOT in _nli_skip_intents(); grounding runs for
    semantic-routed queries (synthesis falls through to the semantic LLM pipeline)."""
    assert "synthesis" not in rag._nli_skip_intents(), (
        "synthesis must not be in the NLI skip list"
    )

    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(2))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda claims, docs, metas, **kw:
                        (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "que dice el vault"])

    assert result.exit_code == 0, result.output
    assert len(ground_calls) == 1, "grounding must run for non-skip-intent semantic path"


def test_query_nli_ground_returns_none_no_crash_no_summary(monkeypatch):
    """Test 8: ground_claims_nli returns None → no crash, grounding_summary=None in log."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(2))
    monkeypatch.setattr(rag, "ground_claims_nli", lambda *a, **kw: None)

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test query"])

    assert result.exit_code == 0, result.output
    assert logged.get("grounding_summary") is None
    assert logged.get("nli_ms") == 0


def test_query_nli_grounding_summary_logged_with_contradicted(monkeypatch):
    """Test 9: GroundingResult with contradicted=2 → grounding_summary.contradicted=2 logged."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    gr = _fake_grounding_result(claims_total=4, supported=1, contradicted=2, neutral=1)
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(4))
    monkeypatch.setattr(rag, "ground_claims_nli", lambda *a, **kw: gr)

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test contradictions"])

    assert result.exit_code == 0, result.output
    summary = logged.get("grounding_summary")
    assert summary is not None
    assert summary["contradicted"] == 2
    assert summary["supported"] == 1
    assert summary["neutral"] == 1
    assert summary["claims_total"] == 4
    assert logged.get("nli_ms") == 42


def test_query_nli_empty_response_no_crash(monkeypatch):
    """Test 10: LLM returns empty string → split_claims("") → [] → ground NOT called."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag.ollama, "chat", _streaming_fn(""))

    ground_calls: list = []
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test empty"])

    assert result.exit_code == 0, result.output
    assert ground_calls == [], "ground must not be called when full is empty"
    assert logged.get("grounding_summary") is None


def test_query_nli_refusal_claim_still_calls_ground(monkeypatch):
    """Test 11: split_claims returns single is_refusal=True claim → ground_claims_nli
    IS still called (refusal skip is inside ground_claims_nli, not in query())."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    refusal_claim = rag.Claim(text="No encontre esto en el vault.", is_refusal=True)
    monkeypatch.setattr(rag, "split_claims", lambda text: [refusal_claim])

    ground_calls: list = []
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda claims, docs, metas, **kw:
                        (ground_calls.append(claims), _fake_grounding_result(1, 0, 0, 1))[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test refusal"])

    assert result.exit_code == 0, result.output
    assert len(ground_calls) == 1, "ground_claims_nli must be called even for refusal claims"
    assert ground_calls[0][0].is_refusal is True


def test_query_nli_fast_path_grounding_still_runs(monkeypatch):
    """Test 12: fast_path=True → grounding runs (not gated by fast_path, only flag+intent)."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    rr = _fake_retrieve_result(fast_path=True)
    _apply_base_patches_query(monkeypatch, retrieve_result=rr)

    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(2))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "fast path test"])

    assert result.exit_code == 0, result.output
    assert len(ground_calls) == 1, "grounding must run regardless of fast_path flag"
    assert logged.get("fast_path") is True


# ── Chat integration: 4 cases ─────────────────────────────────────────────────


def test_chat_nli_grounding_runs_for_semantic_turn(monkeypatch, tmp_path):
    """Test 13: RAG_NLI_GROUNDING=1 + chat turn → split_claims + ground_claims_nli called."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_chat(monkeypatch, tmp_path)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(2))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda claims, docs, metas, **kw:
                        (ground_calls.append(True), _fake_grounding_result())[1])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["chat", "--vault", "test", "--no-auto-filter"],
        input="que dice el vault sobre proyectos\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert len(split_calls) == 1, "split_claims must be called once per chat turn"
    assert len(ground_calls) == 1, "ground_claims_nli must be called once per chat turn"


def test_chat_nli_grounding_runs_even_for_count_like_turn(monkeypatch, tmp_path):
    """Test 14: chat() does NOT filter by intent — grounding runs even for count-like questions.

    Decision T4: chat omits the intent-skip check present in query().
    """
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_chat(monkeypatch, tmp_path)

    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(1))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw:
                        (ground_calls.append(True), _fake_grounding_result(1, 1, 0, 0))[1])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["chat", "--vault", "test", "--no-auto-filter"],
        input="cuantas notas tengo en el vault\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert len(ground_calls) == 1, (
        "chat path has no intent-filter — grounding runs unconditionally when ON"
    )


def test_chat_nli_disabled_skips_grounding(monkeypatch, tmp_path):
    """Test 15: RAG_NLI_GROUNDING unset → neither split_claims nor ground called."""
    monkeypatch.delenv("RAG_NLI_GROUNDING", raising=False)
    _apply_base_patches_chat(monkeypatch, tmp_path)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(2))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["chat", "--vault", "test", "--no-auto-filter"],
        input="que hay en el vault\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert split_calls == [], "split_claims must not run when NLI is disabled"
    assert ground_calls == [], "ground_claims_nli must not run when NLI is disabled"


def test_chat_nli_multi_turn_grounding_called_twice(monkeypatch, tmp_path):
    """Test 16: 2-turn chat with NLI ON → ground_claims_nli called exactly twice."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_chat(monkeypatch, tmp_path)

    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(2))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw:
                        (ground_calls.append(True), _fake_grounding_result())[1])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["chat", "--vault", "test", "--no-auto-filter"],
        input="primera pregunta sobre proyectos\nsegunda pregunta sobre notas\n/exit\n",
    )

    assert result.exit_code == 0, result.output
    assert len(ground_calls) == 2, (
        f"expected grounding called 2x (one per turn), got {len(ground_calls)}"
    )


# ── Exception safety: 4 cases ─────────────────────────────────────────────────


def _raise(exc):
    """Helper: raise exc from a lambda-compatible expression."""
    raise exc


def test_query_nli_ground_exception_no_crash_response_delivered(monkeypatch):
    """Test 17: ground_claims_nli raises → no crash, LLM response still delivered to user."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(2))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: _raise(RuntimeError("NLI exploded")))

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test exception"])

    assert result.exit_code == 0, (
        f"query must not crash when ground_claims_nli raises: {result.output}"
    )
    assert "Respuesta del vault." in result.output
    assert logged.get("grounding_summary") is None
    assert logged.get("nli_ms") == 0


def test_query_nli_split_exception_no_crash_ground_not_called(monkeypatch):
    """Test 18: split_claims raises → no crash, ground_claims_nli never called."""
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)

    monkeypatch.setattr(rag, "split_claims",
                        lambda text: _raise(ValueError("split failed")))

    ground_calls: list = []
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test split fail"])

    assert result.exit_code == 0, result.output
    assert "Respuesta del vault." in result.output
    assert ground_calls == [], "ground must not be called when split_claims raises"


def test_query_log_event_sql_failure_no_crash(monkeypatch):
    """Test 19: DB write inside log_query_event fails → no crash (internal try/except handles it).

    Simulates a SQL write failure by making _ragvec_state_conn yield a
    connection whose execute() raises OperationalError. _sql_write_with_retry
    catches non-lock errors and returns silently so query() is unaffected.
    """
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")
    _apply_base_patches_query(monkeypatch)
    monkeypatch.setattr(rag, "split_claims", lambda text: _fake_claims(1))
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: _fake_grounding_result(1, 1, 0, 0))

    @contextmanager
    def _broken_conn():
        class _BrokenC:
            def execute(self, *a, **kw):
                raise sqlite3.OperationalError("disk I/O error")
            def executemany(self, *a, **kw):
                raise sqlite3.OperationalError("disk I/O error")
        yield _BrokenC()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _broken_conn)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test log failure"])

    assert result.exit_code == 0, (
        f"query must not crash when DB write fails: {result.output}"
    )
    assert "Respuesta del vault." in result.output


def test_query_nli_disabled_by_false_string(monkeypatch):
    """Test 20: RAG_NLI_GROUNDING=false → _nli_grounding_enabled() returns False → skipped.

    Edge: env var explicitly set to string "false" must also disable grounding.
    """
    monkeypatch.setenv("RAG_NLI_GROUNDING", "false")

    assert not rag._nli_grounding_enabled(), (
        "RAG_NLI_GROUNDING=false must make _nli_grounding_enabled() return False"
    )

    _apply_base_patches_query(monkeypatch)

    split_calls: list = []
    ground_calls: list = []
    monkeypatch.setattr(rag, "split_claims",
                        lambda text: (split_calls.append(text), _fake_claims(2))[1])
    monkeypatch.setattr(rag, "ground_claims_nli",
                        lambda *a, **kw: (ground_calls.append(True), _fake_grounding_result())[1])

    logged: dict = {}
    monkeypatch.setattr(rag, "log_query_event", lambda ev: logged.update(ev))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["query", "--plain", "test false string"])

    assert result.exit_code == 0, result.output
    assert split_calls == []
    assert ground_calls == []
    assert logged.get("grounding_summary") is None
