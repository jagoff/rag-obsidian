"""Personal Mirror — tests del aggregator + LLM insights.

Cubre:
- ``assemble_mirror()`` paralelo con 8 sources, cache hit/miss,
  cache_hit flag, source failure isolation, timeout per-source.
- ``generate_insights()`` JSON parsing (strict + markdown fence),
  truncation a 5×500 chars, error path cuando LLM falla.
- ``cache_invalidate()`` clear total.
"""
from __future__ import annotations

import json

import pytest

from rag import mirror as mirror_mod
from rag.mirror import (
    _SOURCES,
    assemble_mirror,
    cache_invalidate,
    generate_insights,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Cada test arranca con cache vacío."""
    cache_invalidate()
    yield
    cache_invalidate()


# ── assemble_mirror ────────────────────────────────────────────────────


def _stub_sources(monkeypatch, payloads: dict[str, dict]) -> None:
    """Reemplaza _SOURCES con stubs deterministas."""
    new = {name: (lambda d, p=p: p) for name, p in payloads.items()}
    monkeypatch.setattr(mirror_mod, "_SOURCES", new)


def test_assemble_mirror_returns_all_sources(monkeypatch):
    payloads = {f"src_{i}": {"items": [i]} for i in range(8)}
    _stub_sources(monkeypatch, payloads)

    r = assemble_mirror(date="2026-05-09", use_cache=False)

    assert r["date"] == "2026-05-09"
    assert r["cache_hit"] is False
    assert r["wall_s"] >= 0
    assert set(r["sources"].keys()) == set(payloads.keys())
    for name, payload in payloads.items():
        assert r["sources"][name] == payload


def test_assemble_mirror_cache_hit_on_second_call(monkeypatch):
    payloads = {"a": {"items": [1]}}
    _stub_sources(monkeypatch, payloads)

    r1 = assemble_mirror(date="2026-05-09")
    r2 = assemble_mirror(date="2026-05-09")

    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    # Mismo payload retornado.
    assert r2["sources"] == r1["sources"]
    # No alias bug: r1 sigue siendo cache_hit=False.
    assert r1["cache_hit"] is False


def test_assemble_mirror_use_cache_false_recomputes(monkeypatch):
    payloads = {"a": {"items": [1]}}
    _stub_sources(monkeypatch, payloads)

    assemble_mirror(date="2026-05-09")
    r2 = assemble_mirror(date="2026-05-09", use_cache=False)

    assert r2["cache_hit"] is False


def test_assemble_mirror_cache_invalidate_drops_entries(monkeypatch):
    payloads = {"a": {"items": [1]}}
    _stub_sources(monkeypatch, payloads)

    assemble_mirror(date="2026-05-09")
    cache_invalidate()
    r2 = assemble_mirror(date="2026-05-09")

    assert r2["cache_hit"] is False


def test_assemble_mirror_source_failure_isolated(monkeypatch):
    """Una source que raise no debe romper las otras."""
    def bad(_d):
        raise RuntimeError("source explosion")

    monkeypatch.setattr(
        mirror_mod, "_SOURCES",
        {"good": lambda d: {"items": [1]}, "bad": bad},
    )

    r = assemble_mirror(date="2026-05-09", use_cache=False)

    assert r["sources"]["good"] == {"items": [1]}
    assert "error" in r["sources"]["bad"]
    assert "source explosion" in r["sources"]["bad"]["error"]


def test_assemble_mirror_default_date_is_today(monkeypatch):
    _stub_sources(monkeypatch, {"a": {"items": []}})
    r = assemble_mirror(use_cache=False)
    # YYYY-MM-DD format.
    assert len(r["date"]) == 10 and r["date"][4] == "-" and r["date"][7] == "-"


def test_assemble_mirror_per_date_cache_namespace(monkeypatch):
    counter = {"n": 0}
    def bumper(_d):
        counter["n"] += 1
        return {"n": counter["n"]}
    monkeypatch.setattr(mirror_mod, "_SOURCES", {"a": bumper})

    assemble_mirror(date="2026-05-09")
    assemble_mirror(date="2026-05-10")  # date diferente, cache miss
    assemble_mirror(date="2026-05-09")  # vuelve al primero, hit

    assert counter["n"] == 2  # solo 2 cómputos reales


# ── _SOURCES estructura ────────────────────────────────────────────────


def test_sources_registry_has_expected_blocks():
    """El frontend asume estos nombres exactos. screen_time + screen_context
    agregados post-Fase 2e Peekaboo."""
    expected = {
        "active_projects", "top_entities", "mood_today", "mood_timeline",
        "pendientes", "dormant_notes", "spotify_top",
        "screen_time", "screen_context", "observations",
    }
    assert set(_SOURCES.keys()) == expected


# ── generate_insights ──────────────────────────────────────────────────


def _fake_chat_response(content: str):
    """Mimic ChatResponse with .message.content."""
    class _M:
        def __init__(self, c):
            self.content = c
    class _R:
        def __init__(self, c):
            self.message = _M(c)
    return _R(content)


def test_generate_insights_parses_strict_json(monkeypatch):
    payload = {"insights": ["uno", "dos", "tres"]}
    _patch_backend(monkeypatch, json.dumps(payload))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == ["uno", "dos", "tres"]
    assert "model" in r


def test_generate_insights_strips_markdown_fence(monkeypatch):
    fenced = "```json\n" + json.dumps({"insights": ["a", "b"]}) + "\n```"
    _patch_backend(monkeypatch, fenced)

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == ["a", "b"]


def test_generate_insights_caps_at_5_items(monkeypatch):
    raw = {"insights": [f"i{i}" for i in range(20)]}
    _patch_backend(monkeypatch, json.dumps(raw))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert len(r["insights"]) == 5


def test_generate_insights_truncates_long_items(monkeypatch):
    raw = {"insights": ["x" * 1000]}
    _patch_backend(monkeypatch, json.dumps(raw))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert len(r["insights"][0]) == 500


def test_generate_insights_handles_non_json_response(monkeypatch):
    _patch_backend(monkeypatch, "esto no es json valido")

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == []
    assert "non-JSON" in r.get("error", "")


def test_generate_insights_handles_backend_exception(monkeypatch):
    class _Bad:
        def chat(self, *a, **kw):
            raise RuntimeError("backend down")

    monkeypatch.setattr(
        "rag.llm_backend.get_backend", lambda: _Bad(),
    )
    monkeypatch.setattr(
        "rag.resolve_chat_model", lambda _t: "qwen2.5:3b",
    )

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == []
    assert "backend down" in r.get("error", "")


# ── helpers ─────────────────────────────────────────────────────────────


def _patch_backend(monkeypatch, content: str):
    class _BE:
        def chat(self, **kw):
            return _fake_chat_response(content)

    monkeypatch.setattr("rag.llm_backend.get_backend", lambda: _BE())
    monkeypatch.setattr("rag.resolve_chat_model", lambda _t: "qwen2.5:3b")
