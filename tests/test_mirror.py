"""Personal Mirror — tests del aggregator + LLM insights.

Cubre:
- ``assemble_mirror()`` paralelo con 8 sources, cache hit/miss,
  cache_hit flag, source failure isolation, timeout per-source.
- ``generate_insights()`` JSON parsing (strict + markdown fence),
  truncation a 5×800 chars, error path cuando LLM falla.
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
    _source_pendientes,
    _source_spotify_top,
    _source_top_entities,
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


@pytest.fixture(autouse=True)
def _isolate_blacklist(monkeypatch, tmp_path):
    import rag.exclusions as _exclusions

    monkeypatch.setattr(_exclusions, "_DB_PATH", tmp_path / "blacklist.db")
    monkeypatch.setattr(_exclusions, "_CONFIG_PATH", tmp_path / "blacklist.json")
    monkeypatch.setattr(_exclusions, "_LEGACY_IGNORED_PATH", tmp_path / "ignored_notes.json")
    monkeypatch.setattr(_exclusions, "_CACHE", None)
    monkeypatch.setattr(_exclusions, "_LEGACY_CACHE", None)
    yield


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


def test_assemble_mirror_cache_hit_refreshes_live_whatsapp(monkeypatch):
    calls = {"stable": 0, "whatsapp": 0}

    def stable(_date):
        calls["stable"] += 1
        return {"n": calls["stable"]}

    def whatsapp(_date):
        calls["whatsapp"] += 1
        return {"counts": {"today_chats": calls["whatsapp"]}}

    monkeypatch.setattr(
        mirror_mod,
        "_SOURCES",
        {"stable": stable, "whatsapp": whatsapp},
    )
    monkeypatch.setattr(mirror_mod, "_is_live_mirror_date", lambda _date: True)

    r1 = assemble_mirror(date="2026-05-09")
    r2 = assemble_mirror(date="2026-05-09")

    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True
    assert calls == {"stable": 1, "whatsapp": 2}
    assert r2["sources"]["stable"]["n"] == 1
    assert r2["sources"]["whatsapp"]["counts"]["today_chats"] == 2
    assert r2["live_refreshed"] == ["whatsapp"]


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


def test_source_top_entities_falls_back_to_whatsapp_when_entities_empty(monkeypatch):
    import rag.integrations.whatsapp as _wa

    monkeypatch.setattr(mirror_mod, "_open_telemetry_ro", lambda: None)
    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_today",
        lambda max_chats=8: [
            {"name": "Cloud Services", "count": 71, "last_snippet": "ceph"},
            {"name": "Maria", "count": 32, "last_snippet": "No entendí"},
        ],
    )

    out = _source_top_entities("2026-05-16")

    assert out["fallback"] == "whatsapp_today"
    assert out["items"][0]["name"] == "Maria"
    assert out["items"][0]["kind"] == "chat"
    assert out["items"][0]["meta"] == "32 msgs hoy"


def test_source_pendientes_uses_fast_vault_scope(monkeypatch, tmp_path):
    import rag as _rag

    monkeypatch.setattr(
        mirror_mod,
        "_source_pendientes_light",
        lambda date: {
            "items": [{
                "category": "reminder",
                "title": "Contratar el pelotero",
                "meta": "undated",
                "when": "",
            }],
            "counts": {"reminders": 1},
            "services_consulted": ["Reminders"],
            "reason": "collector_failed_fallback",
        },
    )
    monkeypatch.setattr(_rag, "resolve_vault_paths", lambda names: [("work", tmp_path)])
    monkeypatch.setattr(
        _rag,
        "_pendientes_extract_loops_fast",
        lambda vault, days, max_items: [
            {
                "source_note": "01-Projects/x.md",
                "loop_text": "cerrar staging urls",
                "age_days": 2,
            },
        ],
    )

    out = _source_pendientes("2026-05-16")

    assert out["counts"]["reminders"] == 1
    assert out["counts"]["loops"] == 1
    assert out["vault_scope"] == ["work"]
    assert out["reason"] == "mirror_fast_collector"
    categories = [item["category"] for item in out["items"]]
    assert "reminder" in categories
    assert "vault loop" in categories


def test_source_spotify_falls_back_to_vault_top_snapshot(monkeypatch, tmp_path):
    import rag as _rag

    vault = tmp_path / "vault"
    spotify_dir = vault / "99-obsidian/99-AI/external-ingest/Spotify"
    spotify_dir.mkdir(parents=True)
    (spotify_dir / "_top.md").write_text(
        """---
source: spotify-top
refreshed_date: 2026-05-12
---

# Spotify Top

## Top tracks (2)

- [Lo siento](https://open.spotify.com/track/1) — Beret
- [Pensando en Ti](https://open.spotify.com/track/2) — Canserbero

## Top artists (1)

- [Canserbero](https://open.spotify.com/artist/1)
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(mirror_mod, "_open_telemetry_ro", lambda: None)
    monkeypatch.setattr(
        _rag,
        "resolve_vault_paths",
        lambda names: [("home", vault)] if names == ["all"] else [("home", vault)],
    )

    out = _source_spotify_top("2026-05-16")

    assert out["mode"] == "top_snapshot"
    assert out["snapshot_date"] == "2026-05-12"
    assert out["items"][0]["track"] == "Lo siento"
    assert out["items"][0]["artist"] == "Beret"


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


def test_source_whatsapp_applies_global_chat_blacklist(monkeypatch):
    import rag.integrations.whatsapp as _wa

    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_today",
        lambda max_chats=6: [
            {"name": "Cloud Services", "count": 10, "last_snippet": "noise"},
            {"name": "Maria", "count": 2, "last_snippet": "ok"},
        ],
    )
    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_unread",
        lambda hours=24, max_chats=6: [
            {"name": "Cloud Services", "count": 10, "last_snippet": "noise"},
            {"name": "Maria", "count": 2, "last_snippet": "ok"},
        ],
    )
    monkeypatch.setattr(
        mirror_mod,
        "_mirror_whatsapp_unreplied",
        lambda hours=48, max_chats=5: [
            {"name": "Cloud Services", "last_snippet": "noise", "hours_waiting": 1.0},
            {"name": "Maria", "last_snippet": "ok", "hours_waiting": 0.1},
        ],
    )

    out = _source_whatsapp("2026-05-17")

    assert [x["name"] for x in out["today"]] == ["Maria"]
    assert [x["name"] for x in out["recent_inbound"]] == ["Maria"]
    assert [x["name"] for x in out["unreplied"]] == ["Maria"]
    assert out["counts"] == {
        "today_chats": 1,
        "recent_inbound_chats": 1,
        "unreplied_chats": 1,
    }


def test_source_whatsapp_enriches_recent_context_and_media(monkeypatch, tmp_path):
    import rag as _rag
    import rag.integrations.whatsapp as _wa

    db = tmp_path / "messages.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT, last_message_time TIMESTAMP)")
    con.execute(
        "CREATE TABLE messages ("
        "id TEXT, chat_jid TEXT, sender TEXT, content TEXT, timestamp TIMESTAMP, "
        "is_from_me BOOLEAN, media_type TEXT, filename TEXT, url TEXT, media_key BLOB, "
        "file_sha256 BLOB, file_enc_sha256 BLOB, file_length INTEGER, "
        "quoted_message_id TEXT, quoted_text TEXT, PRIMARY KEY (id, chat_jid))"
    )
    con.execute("CREATE INDEX idx_messages_chat_ts ON messages(chat_jid, timestamp DESC)")
    jid = "galaxia@lid"
    con.execute("INSERT INTO chats VALUES (?, ?, datetime('now'))", (jid, "Galaxia Kids"))
    rows = [
        ("m1", jid, "them", "Buenísimo! Ahora voy a necesitar los datos del cumple", "-25 minutes", 0, "", ""),
        ("m2", jid, "me", "Astor Ferrari 5 años De blade blade", "-20 minutes", 1, "", ""),
        ("m3", jid, "them", "Me pasas alguna foto de referencia del dibujo?", "-15 minutes", 0, "", ""),
        ("m4", jid, "me", "gracias!", "-10 minutes", 1, "", ""),
        ("m5", jid, "them", "", "-5 minutes", 0, "image", "ref.jpg"),
    ]
    for row in rows:
        con.execute(
            "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename) "
            "VALUES (?, ?, ?, ?, datetime('now', ?), ?, ?, ?)",
            row,
        )
    con.commit()
    con.close()

    monkeypatch.setattr(_rag, "WHATSAPP_DB_PATH", db)
    monkeypatch.setattr(_rag, "WHATSAPP_BOT_JID", "bot@jid")
    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_today",
        lambda max_chats=6: [{"jid": jid, "name": "Galaxia Kids", "count": 3, "last_snippet": ""}],
    )
    monkeypatch.setattr(
        _wa,
        "_fetch_whatsapp_unread",
        lambda hours=24, max_chats=6: [{"jid": jid, "name": "Galaxia Kids", "count": 3, "last_snippet": ""}],
    )

    out = _source_whatsapp("2026-05-17")

    chat = out["unreplied"][0]
    assert chat["last_snippet"].startswith("[imagen]")
    assert "invitaciones y referencia visual" in chat["topic_hint"]
    assert "intercambio de medios" in chat["topic_hint"]
    assert any(msg["snippet"].startswith("[imagen]") for msg in chat["recent_context"])
    assert out["today"][0]["topic_hint"] == chat["topic_hint"]
    assert out["today"][0]["recent_context"] == chat["recent_context"]


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


def test_summarize_for_llm_marks_pendientes_as_detected_not_mentions():
    summary = _summarize_for_llm({
        "date": "2026-05-16",
        "sources": {
            "pendientes": {
                "items": [{
                    "category": "vault loop",
                    "title": "profundizar mis conocimientos en Kubernetes",
                    "meta": "[work] 0d · Guía Práctica - Performance Review",
                    "when": "",
                }],
            },
        },
    })

    assert "Pendientes detectados" in summary
    assert "no asumir que fueron mencionados hoy" in summary
    assert "vault loop; [work] 0d" in summary


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
    payload = {
        "insights": [
            "Insight uno con suficiente detalle para pasar el filtro.",
            "Insight dos con suficiente detalle para pasar el filtro.",
            "Insight tres con suficiente detalle para pasar el filtro.",
        ]
    }
    _patch_backend(monkeypatch, json.dumps(payload))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == payload["insights"]
    assert "model" in r


def test_generate_insights_strips_markdown_fence(monkeypatch):
    raw = {
        "insights": [
            "Insight A con suficiente detalle para pasar el filtro.",
            "Insight B con suficiente detalle para pasar el filtro.",
        ]
    }
    fenced = "```json\n" + json.dumps(raw) + "\n```"
    _patch_backend(monkeypatch, fenced)

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert r["insights"] == raw["insights"]


def test_generate_insights_caps_at_5_items(monkeypatch):
    raw = {"insights": [f"Insight {i} con suficiente detalle para pasar el filtro." for i in range(20)]}
    _patch_backend(monkeypatch, json.dumps(raw))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert len(r["insights"]) == 5


def test_generate_insights_truncates_long_items(monkeypatch):
    raw = {"insights": ["x" * 1000]}
    _patch_backend(monkeypatch, json.dumps(raw))

    r = generate_insights({"date": "2026-05-09", "sources": {}})

    assert len(r["insights"][0]) == 800


def test_generate_insights_filters_unsupported_mention_claims(monkeypatch):
    raw = {
        "insights": [
            "Hoy mencionaste 'profundizar mis conocimientos en Kubernetes' — sigue sin dedicar tiempo a ello.",
            "Cloud Services tiene un mensaje esperando hace 1.3h. Conviene revisarlo antes de que se pierda entre otros chats.",
        ],
    }
    _patch_backend(monkeypatch, json.dumps(raw))

    r = generate_insights({
        "date": "2026-05-09",
        "sources": {
            "whatsapp": {
                "counts": {"today_chats": 1, "unreplied_chats": 1},
                "unreplied": [
                    {
                        "name": "Cloud Services",
                        "hours_waiting": 1.3,
                        "last_snippet": "terminaron las tareas post update",
                    }
                ],
            },
        },
    })

    joined = "\n".join(r["insights"])
    assert "Hoy mencionaste" not in joined
    assert "sigue sin dedicar tiempo" not in joined
    assert "Cloud Services tiene un mensaje esperando hace 1.3h" not in joined
    assert not any("WZP está movido" in item for item in r["insights"])


def test_generate_insights_uses_grounded_rules_when_enough_signal(monkeypatch):
    class _BackendShouldNotRun:
        def chat(self, **kw):
            raise AssertionError("LLM should not run when grounded snapshot is enough")

    monkeypatch.setattr("rag.llm_backend.get_backend", lambda: _BackendShouldNotRun())
    r = generate_insights({
        "date": "2026-05-09",
        "sources": {
            "whatsapp": {
                "counts": {"today_chats": 2, "unreplied_chats": 1},
                "unreplied": [
                    {
                        "name": "Maria",
                        "hours_waiting": 0.4,
                        "last_snippet": "terminaron las tareas post update",
                    }
                ],
            },
            "pendientes": {
                "items": [{
                    "category": "reminder",
                    "title": "Enviar reporte",
                    "meta": "today",
                }],
            },
            "mood_today": {"score": 0.3, "n_signals": 2},
            "active_projects": {
                "items": [{
                    "name": "finops-aws-report",
                    "note_count_30d": 8,
                    "days_ago": 0,
                }],
            },
        },
    })

    assert r["model"] == "grounded-rules"
    assert len(r["insights"]) >= 3
    assert any("WZP está movido" in item for item in r["insights"])
    assert any("Enviar reporte" in item for item in r["insights"])


def test_insights_prompt_requires_grounding():
    prompt = mirror_mod._INSIGHTS_PROMPT

    assert "Usá SOLO datos presentes en el snapshot" in prompt
    assert "vault loop" in prompt
    assert "llamar al dentista" not in prompt


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
