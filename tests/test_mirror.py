"""Personal Mirror — tests del aggregator + LLM insights.

Cubre:
- ``assemble_mirror()`` paralelo con 8 sources, cache hit/miss,
  cache_hit flag, source failure isolation, timeout per-source.
- ``generate_insights()`` JSON parsing (strict + markdown fence),
  truncation a 5×500 chars, error path cuando LLM falla.
- ``cache_invalidate()`` clear total.
- ``_source_screen_time()`` DB inexistente, DB locked (timeout),
  conn leak (conn.close() en excepción).
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rag import mirror as mirror_mod
from rag.mirror import (
    _SOURCES,
    _source_active_projects,
    _source_dormant_notes,
    _source_mood_today,
    _source_whatsapp,
    _summarize_for_llm,
    _source_screen_time,
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
        "pendientes", "whatsapp", "dormant_notes", "spotify_top",
        "screen_time", "screen_context", "observations",
    }
    assert set(_SOURCES.keys()) == expected


def test_source_active_projects_scans_all_registered_vaults(monkeypatch, tmp_path):
    import rag as _rag

    home = tmp_path / "home"
    work = tmp_path / "work"
    (home / "01-Projects" / "Casa").mkdir(parents=True)
    (work / "01-Projects" / "Trabajo").mkdir(parents=True)
    (home / "01-Projects" / "Casa" / "a.md").write_text("home", encoding="utf-8")
    (work / "01-Projects" / "Trabajo" / "b.md").write_text("work", encoding="utf-8")
    monkeypatch.setattr(
        _rag,
        "resolve_vault_paths",
        lambda names: [("home", home), ("work", work)] if names == ["all"] else [("home", home)],
    )

    out = _source_active_projects("2026-05-16")

    by_name = {item["name"]: item for item in out["items"]}
    assert by_name["Casa"]["vault"] == "home"
    assert by_name["Trabajo"]["vault"] == "work"
    assert out["vault_scope"] == ["home", "work"]


def test_source_dormant_notes_scans_all_registered_vaults(monkeypatch, tmp_path):
    import rag as _rag

    home = tmp_path / "home"
    work = tmp_path / "work"
    home_note = home / "02-Areas" / "home-old.md"
    work_note = work / "02-Areas" / "work-old.md"
    home_note.parent.mkdir(parents=True)
    work_note.parent.mkdir(parents=True)
    home_note.write_text("home " * 40, encoding="utf-8")
    work_note.write_text("work " * 40, encoding="utf-8")
    old = time.time() - 45 * 86400
    os.utime(home_note, (old, old))
    os.utime(work_note, (old, old))
    monkeypatch.setattr(
        _rag,
        "resolve_vault_paths",
        lambda names: [("home", home), ("work", work)] if names == ["all"] else [("home", home)],
    )

    out = _source_dormant_notes("2026-05-16")

    by_title = {item["title"]: item for item in out["items"]}
    assert by_title["home-old"]["vault"] == "home"
    assert by_title["work-old"]["vault"] == "work"


def test_source_mood_today_uses_latest_score_when_today_has_no_signal(monkeypatch):
    from rag import mood as _mood

    def fake_get_score(date):
        if date == "2026-05-15":
            return {
                "date": "2026-05-15",
                "score": -0.25,
                "n_signals": 2,
                "sources_used": ["spotify"],
                "top_evidence": [],
                "updated_at": 0,
            }
        return None

    monkeypatch.setattr(_mood, "_is_mood_enabled", lambda: True)
    monkeypatch.setattr(_mood, "is_daemon_enabled", lambda: True)
    monkeypatch.setattr(_mood, "get_score_for_date", fake_get_score)
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [
        {"date": "2026-05-16", "score": 0.0, "n_signals": 0},
        {"date": "2026-05-15", "score": -0.25, "n_signals": 2},
    ])

    out = _source_mood_today("2026-05-16")

    assert out["score"] == -0.25
    assert out["date"] == "2026-05-15"
    assert out["requested_date"] == "2026-05-16"
    assert out["stale"] is True


def test_source_whatsapp_collects_today_recent_and_unreplied(monkeypatch):
    import rag.integrations.whatsapp as _wa

    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_today",
        lambda max_chats=6: [{"name": "Grecia", "count": 2, "last_snippet": "vemos eso"}],
    )
    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_unread",
        lambda hours=24, max_chats=6: [{"name": "Seba", "count": 1, "last_snippet": "ping"}],
    )
    monkeypatch.setattr(
        mirror_mod,
        "_mirror_whatsapp_unreplied",
        lambda hours=48, max_chats=5: [
            {"name": "Joana", "hours_waiting": 7.5, "last_snippet": "me llamás?"}
        ],
    )

    out = _source_whatsapp("2026-05-16")

    assert out["counts"] == {
        "today_chats": 1,
        "recent_inbound_chats": 1,
        "unreplied_chats": 1,
    }
    assert out["unreplied"][0]["name"] == "Joana"


def test_summarize_for_llm_includes_whatsapp_context():
    summary = _summarize_for_llm({
        "date": "2026-05-16",
        "sources": {
            "whatsapp": {
                "counts": {"today_chats": 2, "unreplied_chats": 1},
                "today": [{"name": "Grecia", "count": 2, "last_snippet": "ok"}],
                "unreplied": [
                    {"name": "Joana", "hours_waiting": 7.5, "last_snippet": "me llamás?"}
                ],
            }
        },
    })

    assert "WhatsApp/WZP" in summary
    assert "Joana" in summary
    assert "me llamás?" in summary


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


# ── _source_screen_time ─────────────────────────────────────────────────


def test_source_screen_time_db_not_found(monkeypatch, tmp_path):
    """DB inexistente → devuelve {"apps": []} sin excepción."""
    monkeypatch.setattr(mirror_mod.Path, "home", lambda: tmp_path)
    result = _source_screen_time("2026-05-14")
    assert result == {"apps": []}


def test_source_screen_time_returns_top5(monkeypatch, tmp_path):
    """DB con datos válidos → devuelve top 5 apps ordenadas por uso."""
    db_dir = tmp_path / "Library" / "Application Support" / "ScreenTime"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "MTDatabase.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE ZUSAGE (ZBUNDLEID TEXT, ZTOTALTIMEINSECONDS REAL, ZDAY TEXT)"
    )
    today = "2026-05-14"
    for i, (bundle, secs) in enumerate([
        ("com.apple.Safari", 7200),
        ("com.slack.desktop", 5400),
        ("com.apple.Terminal", 3600),
        ("com.spotify.client", 1800),
        ("com.apple.mail", 900),
        ("com.apple.finder", 300),
    ]):
        conn.execute("INSERT INTO ZUSAGE VALUES (?, ?, ?)", (bundle, secs, today))
    conn.commit()
    conn.close()

    monkeypatch.setattr(mirror_mod.Path, "home", lambda: tmp_path)
    result = _source_screen_time(today)

    assert "apps" in result
    assert len(result["apps"]) == 5
    assert result["apps"][0]["app_name"] == "Safari"
    assert result["apps"][0]["total_hours"] == round(7200 / 3600, 2)


def test_source_screen_time_conn_closed_on_exception(monkeypatch, tmp_path):
    """conn.close() se llama incluso si cursor.execute() falla (no conn leak)."""
    db_dir = tmp_path / "Library" / "Application Support" / "ScreenTime"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "MTDatabase.db"

    # Crear la DB vacía (sin tabla) para que cursor.execute() falle con OperationalError.
    conn_init = sqlite3.connect(str(db_path))
    conn_init.close()

    close_calls = []

    class _TrackedConn:
        """Wrapper que delega a una conexión real y registra close()."""
        def __init__(self, real):
            self._real = real
            self.row_factory = None

        def cursor(self):
            return self._real.cursor()

        def close(self):
            close_calls.append(True)
            self._real.close()

    original_connect = sqlite3.connect

    def patched_connect(path, **kwargs):
        real = original_connect(path, **kwargs)
        return _TrackedConn(real)

    monkeypatch.setattr(mirror_mod.Path, "home", lambda: tmp_path)
    with patch("rag.mirror.sqlite3.connect", side_effect=patched_connect):
        result = _source_screen_time("2026-05-14")

    # La tabla ZUSAGE no existe → OperationalError → conn.close() igual se llama.
    assert "error" in result
    assert len(close_calls) == 1, "conn.close() debe haberse llamado exactamente una vez"
