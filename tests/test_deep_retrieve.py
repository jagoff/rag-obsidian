"""Tests for deep_retrieve() — the iterative sub-query retrieval that
auto-triggers when the first pass's top rerank score is below
CONFIDENCE_DEEP_THRESHOLD. Previously had zero coverage.

What this covers:
  - _judge_sufficiency parses "SUFICIENTE" vs. sub-query response
  - _judge_sufficiency returns (True, "") on LLM exception (fail-safe)
  - deep_retrieve stops after at most _DEEP_MAX_ITERS passes even if
    the LLM keeps saying the evidence is insufficient
  - deep_retrieve merges new chunks and dedups by (path, first 50 chars)
  - deep_retrieve stops early when sub-query surfaces no new chunks
  - deep_retrieve early-returns when first pass is empty
  - graph neighbours from sub-passes are merged and deduped by path

The real retrieve() is mocked — we care about the orchestration logic,
not the downstream pipeline.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import rag


# ── _judge_sufficiency ───────────────────────────────────────────────────────


def _make_helper_response(content: str):
    resp = MagicMock()
    resp.message.content = content
    return resp


def test_judge_sufficiency_recognises_suficiente(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_helper_response("SUFICIENTE")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is True
    assert sub == ""


def test_judge_sufficiency_returns_sub_query_when_insufficient(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_helper_response("cómo se llama el modelo helper?")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is False
    assert sub == "cómo se llama el modelo helper?"


def test_judge_sufficiency_fails_safe_on_ollama_error(monkeypatch):
    """If the helper raises, deep_retrieve must not loop forever — the
    judge returns (True, '') as a kill-switch."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = RuntimeError("ollama wedged")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is True
    assert sub == ""


# ── deep_retrieve orchestration ──────────────────────────────────────────────


def _res(paths: list[str], scores: list[float], *, graph_paths: list[str] | None = None):
    """Build a retrieve()-shaped result dict for mocking."""
    n = len(paths)
    assert len(scores) == n
    metas = [{"file": p, "note": p.rsplit("/", 1)[-1]} for p in paths]
    docs = [f"content of {p}" for p in paths]
    gm = [{"file": gp, "note": gp} for gp in (graph_paths or [])]
    gd = [f"graph content {gp}" for gp in (graph_paths or [])]
    return {
        "docs": docs,
        "metas": metas,
        "scores": scores,
        "confidence": scores[0] if scores else float("-inf"),
        "graph_docs": gd,
        "graph_metas": gm,
    }


def test_deep_retrieve_returns_first_pass_when_no_docs(monkeypatch):
    """Empty first pass short-circuits — no judge call, no sub-queries."""
    first = {"docs": [], "metas": [], "scores": [], "confidence": float("-inf"),
             "graph_docs": [], "graph_metas": []}
    retrieve_mock = MagicMock(return_value=first)
    judge_mock = MagicMock(return_value=(False, "some sub"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)
    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    assert out is first
    judge_mock.assert_not_called()


def test_deep_retrieve_merges_new_chunks(monkeypatch):
    """Sub-query finds a path the first pass missed → appears in merged result."""
    first = _res(["a.md", "b.md"], [0.05, 0.03])
    second = _res(["b.md", "c.md"], [0.08, 0.04])  # b.md dup, c.md new
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(return_value=(False, "sub query"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)
    # After the first sub-pass, judge again → say sufficient to end the loop.
    judge_mock.side_effect = [(False, "sub"), (True, "")]

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    paths = [m["file"] for m in out["metas"]]
    assert "a.md" in paths
    assert "b.md" in paths
    assert "c.md" in paths
    # b.md should appear exactly once even though both passes returned it.
    assert paths.count("b.md") == 1


def test_deep_retrieve_stops_at_max_iters(monkeypatch):
    """Even if the judge perpetually says insufficient, deep_retrieve must
    stop at _DEEP_MAX_ITERS. We count retrieve() calls: 1 first pass +
    (_DEEP_MAX_ITERS-1) sub-passes max."""
    first = _res(["a.md"], [0.05])
    second = _res(["b.md"], [0.06])
    third = _res(["c.md"], [0.07])
    retrieve_mock = MagicMock(side_effect=[first, second, third, _res(["z.md"], [0.01])])
    judge_mock = MagicMock(return_value=(False, "need more"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    # _DEEP_MAX_ITERS=3 → 1 first + up to 2 sub-queries = 3 calls total.
    assert retrieve_mock.call_count == rag._DEEP_MAX_ITERS


def test_deep_retrieve_stops_early_when_sub_query_adds_nothing(monkeypatch):
    """Sub-query returns only duplicates → break out of the loop
    (prevents wasted iterations when the LLM keeps rephrasing around the
    same cluster)."""
    first = _res(["a.md", "b.md"], [0.05, 0.03])
    duplicate_only = _res(["a.md", "b.md"], [0.08, 0.03])  # both seen
    retrieve_mock = MagicMock(side_effect=[first, duplicate_only])
    judge_mock = MagicMock(return_value=(False, "same-area sub"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    # Exactly 2 calls — first + one sub-query that added nothing.
    assert retrieve_mock.call_count == 2


def test_deep_retrieve_deduplicates_graph_neighbours(monkeypatch):
    """Graph neighbours from the sub-pass merge into the first-pass's
    graph context, deduped by file path."""
    first = _res(["a.md"], [0.05], graph_paths=["g1.md", "g2.md"])
    second = _res(["b.md"], [0.06], graph_paths=["g2.md", "g3.md"])  # g2 dup
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(side_effect=[(False, "sub"), (True, "")])
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    graph_paths = [gm["file"] for gm in out["graph_metas"]]
    assert set(graph_paths) == {"g1.md", "g2.md", "g3.md"}
    assert graph_paths.count("g2.md") == 1


def test_deep_retrieve_sets_confidence_to_new_top(monkeypatch):
    """After merging + re-sorting by score, confidence must reflect the
    best score seen across all passes."""
    first = _res(["a.md"], [0.05])
    second = _res(["b.md"], [0.18])  # higher score than first
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(side_effect=[(False, "sub"), (True, "")])
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    assert out["confidence"] == pytest.approx(0.18)
    assert out["metas"][0]["file"] == "b.md"


# ── Confidence threshold invariant ───────────────────────────────────────────


def test_confidence_deep_threshold_within_expected_range():
    """Guard against someone accidentally dropping the threshold to a
    value that would either never trigger or always trigger."""
    assert 0.01 < rag.CONFIDENCE_DEEP_THRESHOLD < 0.5
    # Must be above CONFIDENCE_RERANK_MIN (the refuse gate) — otherwise
    # deep_retrieve would fire on queries we already refused.
    assert rag.CONFIDENCE_DEEP_THRESHOLD > rag.CONFIDENCE_RERANK_MIN


# ── Wall-time timeout (2026-04-22 fix) ───────────────────────────────────────


def test_deep_retrieve_respects_wall_time_timeout(monkeypatch):
    """Aunque el juez siga diciendo "insuficiente" indefinidamente, el loop
    del deep-retrieve DEBE salir cuando pasa _DEEP_MAX_SECONDS.

    Motivación: el peor query de los últimos 7d tuvo t_retrieve=202.6s — cada
    iteración bajo contención extrema de Ollama puede tardar 60-70s. Sin
    timeout absoluto, 3 iters = ~200s. El guard garantiza que ninguna query
    bloquee más de `_DEEP_MAX_SECONDS` en retrieve.

    Simulamos: `retrieve()` devuelve nuevos chunks rápido pero el juez siempre
    dice insuficiente. Forzamos `_DEEP_MAX_SECONDS` a 0.1s y hacemos que el
    juez consuma 50ms simulados → segunda iteración debería quedar afuera.
    """
    import time as _time

    first = _res(["a.md"], [0.05])
    # Retornamos nuevos chunks cada vez para que `added==0` early-exit NO
    # cortocircuite antes del timeout.
    passes = [first] + [_res([f"new{i}.md"], [0.04 + i * 0.01]) for i in range(10)]
    retrieve_mock = MagicMock(side_effect=passes)

    # Juez siempre insuficiente y lentito — consume tiempo wall-clock real.
    # El sleep del judge debe ser > _DEEP_MAX_SECONDS para que el guard
    # dispare en la segunda iteración del loop (pre-fix corría las 2 iters
    # del `range(1, _DEEP_MAX_ITERS)` = 2 loops + first pass = 3 calls).
    def slow_judge(*args, **kwargs):
        _time.sleep(0.15)
        return (False, "keep going")

    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", slow_judge)
    monkeypatch.setattr(rag, "_DEEP_MAX_SECONDS", 0.1)

    t0 = _time.perf_counter()
    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    elapsed = _time.perf_counter() - t0

    # Expected: first pass (1) + loop iter 1 (judge 150ms > timeout 100ms
    # triggers the guard on iter 2) = 2 calls total, NOT 3 (= _DEEP_MAX_ITERS).
    assert retrieve_mock.call_count < rag._DEEP_MAX_ITERS, (
        f"timeout guard no disparó: {retrieve_mock.call_count} calls "
        f"en {elapsed:.3f}s (esperado <{rag._DEEP_MAX_ITERS})"
    )
    # Sanity: el elapsed debe ser cercano al budget + 1 pase del judge,
    # nunca 2 × 150ms + overhead = ~300ms+.
    assert elapsed < 0.35, f"deep_retrieve tardó {elapsed:.3f}s — timeout no cortó bien"


def test_deep_max_seconds_default_is_30():
    """Default `_DEEP_MAX_SECONDS` = 30s tras el fix 2026-04-22.

    Un default demasiado alto (ej: 120s) deja el bug latente; uno muy bajo
    (ej: 5s) cortaría queries legítimas que necesitan deep retrieval en
    hardware lento. 30s es el sweet spot: >P99 normal (~38s p99 prod), pero
    <runaway range observado (60-200s).

    Override vía `RAG_DEEP_MAX_SECONDS` env var para operadores que quieran
    forzar más holgura en hardware con recursos abundantes.
    """
    # Reload no hace falta; lo leemos directo desde el módulo post-import.
    assert rag._DEEP_MAX_SECONDS == 30.0, (
        f"_DEEP_MAX_SECONDS default debe ser 30s (fue {rag._DEEP_MAX_SECONDS}). "
        "Ver CLAUDE.md § deep_retrieve timeout guard."
    )


def test_deep_max_seconds_env_override(monkeypatch):
    """RAG_DEEP_MAX_SECONDS sobreescribe el default."""
    import importlib
    monkeypatch.setenv("RAG_DEEP_MAX_SECONDS", "60")
    reloaded = importlib.reload(rag)
    try:
        assert reloaded._DEEP_MAX_SECONDS == 60.0
    finally:
        monkeypatch.delenv("RAG_DEEP_MAX_SECONDS", raising=False)
        importlib.reload(rag)


# ── Early-exit por high confidence + iteration tracking (2026-04-25) ─────


def test_deep_retrieve_high_confidence_bypass_skips_loop(monkeypatch):
    """Si el primer top score >= _DEEP_HIGH_CONF_BYPASS, skip el loop —
    no llamar _judge_sufficiency. Ahorra ~1-3s del round-trip al helper."""
    from unittest.mock import MagicMock

    judge_calls = {"n": 0}

    def fake_judge(*args, **kwargs):
        judge_calls["n"] += 1
        return (False, "should not be called")

    high_score_result = {
        "docs": ["doc 1"],
        "metas": [{"file": "n.md", "note": "n"}],
        "scores": [0.95],  # > 0.8 threshold
        "graph_docs": [],
        "graph_metas": [],
    }

    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: high_score_result)
    monkeypatch.setattr(rag, "_judge_sufficiency", fake_judge)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)

    assert judge_calls["n"] == 0, "judge_sufficiency NO debe correr en bypass"
    assert out["deep_retrieve_iterations"] == 1
    assert out["deep_retrieve_exit_reason"] == "high_confidence_bypass"


def test_deep_retrieve_no_bypass_when_top_score_below_threshold(monkeypatch):
    """Si top score < _DEEP_HIGH_CONF_BYPASS, debe entrar al loop."""
    from unittest.mock import MagicMock

    judge_calls = {"n": 0}

    def fake_judge(question, docs, metas):
        judge_calls["n"] += 1
        return (True, "")  # immediately sufficient → exits after 1 judge call

    low_score_result = {
        "docs": ["doc 1"],
        "metas": [{"file": "n.md", "note": "n"}],
        "scores": [0.3],  # well below 0.8
        "graph_docs": [],
        "graph_metas": [],
    }

    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: low_score_result)
    monkeypatch.setattr(rag, "_judge_sufficiency", fake_judge)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)

    assert judge_calls["n"] == 1, "judge_sufficiency debe correr cuando no hay bypass"
    assert out["deep_retrieve_exit_reason"] == "sufficient"


def test_deep_retrieve_bypass_disabled_by_zero_threshold(monkeypatch):
    """RAG_DEEP_HIGH_CONF_BYPASS=0 deshabilita el bypass — todas las
    queries pasan por el sufficiency loop."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(rag, "_DEEP_HIGH_CONF_BYPASS", 0.0)

    judge_calls = {"n": 0}

    def fake_judge(question, docs, metas):
        judge_calls["n"] += 1
        return (True, "")

    high_score_result = {
        "docs": ["doc 1"],
        "metas": [{"file": "n.md", "note": "n"}],
        "scores": [0.99],  # very high, but bypass is disabled
        "graph_docs": [],
        "graph_metas": [],
    }

    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: high_score_result)
    monkeypatch.setattr(rag, "_judge_sufficiency", fake_judge)

    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)

    # Even with top score 0.99, judge runs because bypass disabled.
    assert judge_calls["n"] == 1


def test_deep_retrieve_iterations_reported_when_loop_runs(monkeypatch):
    """Cuando el loop corre, deep_retrieve_iterations debe contar pasadas
    reales (1 = solo first pass; 2 = first + 1 sub-query; etc.)."""
    from unittest.mock import MagicMock

    judge_calls = {"n": 0}

    def fake_judge(question, docs, metas):
        judge_calls["n"] += 1
        # Devolver insufficient en el primer call, sufficient en el segundo
        if judge_calls["n"] == 1:
            return (False, "sub query")
        return (True, "")

    call_idx = {"n": 0}

    def fake_retrieve(*args, **kwargs):
        call_idx["n"] += 1
        return {
            "docs": [f"doc {call_idx['n']}"],
            "metas": [{"file": f"n{call_idx['n']}.md", "note": "n"}],
            "scores": [0.5],  # below bypass threshold
            "graph_docs": [],
            "graph_metas": [],
        }

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "_judge_sufficiency", fake_judge)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)

    # Iteration 1 = first pass; loop-iter 1 found 1 added doc + judge
    # call 2 said "sufficient" → iters_completed=2.
    assert out["deep_retrieve_iterations"] == 2
    assert out["deep_retrieve_exit_reason"] == "sufficient"


def test_deep_retrieve_no_docs_marks_iterations_one(monkeypatch):
    """Si la primera pasada devuelve cero docs, no entramos al loop pero
    igual exponemos los campos de tracking."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: {
        "docs": [], "metas": [], "scores": [],
        "graph_docs": [], "graph_metas": [],
    })

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)

    assert out["deep_retrieve_iterations"] == 1
    assert out["deep_retrieve_exit_reason"] == "no_docs"


def test_deep_high_conf_bypass_default_is_0_8():
    """Default `_DEEP_HIGH_CONF_BYPASS` = 0.8.

    Threshold elegido en el audit 2026-04-25: por debajo de 0.8 el
    sufficiency check del helper sigue produciendo signal útil; por
    encima, casi siempre dice SUFICIENTE → es waste del round-trip al
    LLM helper (~1-3s).
    """
    assert rag._DEEP_HIGH_CONF_BYPASS == 0.8, (
        f"_DEEP_HIGH_CONF_BYPASS default debe ser 0.8 "
        f"(fue {rag._DEEP_HIGH_CONF_BYPASS}). Ver _DEEP_HIGH_CONF_BYPASS docstring."
    )
