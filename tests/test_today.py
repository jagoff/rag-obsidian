import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest
from click.testing import CliRunner

import rag


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


# Frozen reference moment used across every datetime.now() call in this file.
# Wednesday 2026-04-21 15:00:00 local — chosen to avoid midnight/weekend edge
# cases while still exercising the "today window" logic. Changing this value
# requires revisiting _set_mtime offsets in every test below (grep for
# `_FROZEN_NOW` to find the coupling sites).
_FROZEN_NOW = datetime(2026, 4, 21, 15, 0, 0)


def _install_frozen_now(monkeypatch, frozen: datetime):
    """Monkeypatchea `rag.datetime` con una subclase que fija `.now()` y
    `.today()` a `frozen`. Todo lo demás (constructor, fromisoformat,
    timedelta aritmética, strftime) queda heredado del `datetime` stdlib.
    """
    real_datetime = datetime

    class _FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return frozen.replace(tzinfo=tz)
            return frozen

        @classmethod
        def today(cls):
            return frozen

    monkeypatch.setattr(rag, "datetime", _FrozenDateTime)
    return frozen


@pytest.fixture
def frozen_now(monkeypatch):
    """Pinea `rag.datetime.now()` a _FROZEN_NOW para eliminar flakies de
    timing (midnight boundary, CI clock drift). No dependemos de
    `freezegun` (no está en deps — monkeypatch de datetime subclass alcanza).
    """
    return _install_frozen_now(monkeypatch, _FROZEN_NOW)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed, frozen_now):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    (vault / "04-Archive/99-obsidian-system/99-AI/reviews").mkdir(parents=True)
    (vault / "02-Areas").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="today_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(
        rag, "_index_single_file",
        lambda *a, **kw: "skipped",
    )
    rag._invalidate_corpus_cache()
    return vault, col, tmp_path


def _set_mtime(path: Path, when: datetime):
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _fake_chat(content: str):
    def _chat(*a, **kw):
        return _FakeResponse(content)
    return _chat


NARRATIVE_STUB = (
    "## 🪞 Lo que pasó hoy\ntexto de recap hoy\n\n"
    "## 📥 Sin procesar\nitem sin tags\n\n"
    "## 🔍 Preguntas abiertas\npregunta\n\n"
    "## 🌅 Para mañana\nseed 1\nseed 2\n"
)


# ── _collect_today_evidence ─────────────────────────────────────────────────


def test_today_evidence_empty_vault(tmp_vault):
    vault, _, tmp_path = tmp_vault
    ev = rag._collect_today_evidence(
        _FROZEN_NOW, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    assert ev["recent_notes"] == []
    assert ev["inbox_today"] == []
    assert ev["todos"] == []
    assert ev["new_contradictions"] == []
    assert ev["low_conf_queries"] == []


def test_today_picks_up_modified_today(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "today.md"
    p.write_text("cuerpo de hoy")
    now = _FROZEN_NOW.replace(hour=15, minute=0, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/today.md" in paths


def test_today_excludes_yesterday(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "yesterday.md"
    p.write_text("vieja de ayer")
    yesterday = _FROZEN_NOW.replace(hour=22, minute=0) - timedelta(days=1)
    _set_mtime(p, yesterday)
    ev = rag._collect_today_evidence(
        _FROZEN_NOW, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/yesterday.md" not in paths


def test_today_excludes_reviews_folder(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2026-04-15.md"
    p.write_text("morning brief")
    now = _FROZEN_NOW
    _set_mtime(p, now - timedelta(minutes=10))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-15.md" not in paths


def test_today_inbox_capture_routed_to_inbox_bucket(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "00-Inbox" / "cap.md"
    p.write_text("captura rápida")
    now = _FROZEN_NOW.replace(hour=10, minute=30, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    inbox_paths = {r["path"] for r in ev["inbox_today"]}
    recent_paths = {r["path"] for r in ev["recent_notes"]}
    assert "00-Inbox/cap.md" in inbox_paths
    assert "00-Inbox/cap.md" not in recent_paths


def test_today_inbox_untagged_flag(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p1 = vault / "00-Inbox" / "untagged.md"
    p1.write_text("sin tags")
    p2 = vault / "00-Inbox" / "tagged.md"
    p2.write_text("---\ntags:\n- area/health\n---\nbody")
    now = _FROZEN_NOW.replace(hour=12, minute=0, second=0, microsecond=0)
    _set_mtime(p1, now)
    _set_mtime(p2, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    by_path = {r["path"]: r for r in ev["inbox_today"]}
    assert by_path["00-Inbox/untagged.md"]["tags"] == []
    assert "area/health" in by_path["00-Inbox/tagged.md"]["tags"]


def test_today_midnight_boundary(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p_today = vault / "02-Areas" / "just-after-midnight.md"
    p_today.write_text("justo después")
    p_yest = vault / "02-Areas" / "just-before.md"
    p_yest.write_text("justo antes")
    now = _FROZEN_NOW.replace(hour=0, minute=5, second=0, microsecond=0)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    _set_mtime(p_today, today_start + timedelta(seconds=1))
    _set_mtime(p_yest, today_start - timedelta(seconds=1))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/just-after-midnight.md" in paths
    assert "02-Areas/just-before.md" not in paths


def test_today_low_conf_queries_today_only(tmp_vault):
    vault, _, tmp_path = tmp_vault
    ql = tmp_path / "q.jsonl"
    now = _FROZEN_NOW.replace(hour=14, minute=0, second=0, microsecond=0)
    today_ts = now.replace(hour=10, minute=0).isoformat(timespec="seconds")
    yest_ts = (now - timedelta(days=1)).isoformat(timespec="seconds")
    entries = [
        {"ts": today_ts, "cmd": "query", "q": "q de hoy", "top_score": 0.005},
        {"ts": yest_ts, "cmd": "query", "q": "q de ayer", "top_score": 0.003},
        {"ts": today_ts, "cmd": "query", "q": "q buena", "top_score": 0.4},
    ]
    ql.write_text("\n".join(json.dumps(e) for e in entries))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=ql,
        contradiction_log=tmp_path / "c.jsonl",
    )
    qs = [q["q"] for q in ev["low_conf_queries"]]
    assert "q de hoy" in qs
    assert "q de ayer" not in qs
    assert "q buena" not in qs


def test_today_contradictions_today_only(tmp_vault):
    vault, _, tmp_path = tmp_vault
    contr = tmp_path / "c.jsonl"
    now = _FROZEN_NOW.replace(hour=16, minute=0, second=0, microsecond=0)
    today_entry = {
        "ts": now.replace(hour=11, minute=0).isoformat(timespec="seconds"),
        "cmd": "contradict_index",
        "subject_path": "02-Areas/x.md",
        "contradicts": [{"path": "02-Areas/y.md", "why": "X vs Y"}],
    }
    yest_entry = {
        "ts": (now - timedelta(days=1)).isoformat(timespec="seconds"),
        "cmd": "contradict_index",
        "subject_path": "02-Areas/a.md",
        "contradicts": [{"path": "02-Areas/b.md", "why": "A vs B"}],
    }
    contr.write_text(
        json.dumps(today_entry) + "\n" + json.dumps(yest_entry) + "\n"
    )
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=contr,
    )
    subjects = [c["subject_path"] for c in ev["new_contradictions"]]
    assert subjects == ["02-Areas/x.md"]


def test_today_todo_frontmatter_in_window(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "with-todo.md"
    p.write_text("---\ntodo:\n- algo\ndue: 2026-05-01\n---\nbody")
    now = _FROZEN_NOW.replace(hour=13, minute=0, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=1), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = [t["path"] for t in ev["todos"]]
    assert "02-Areas/with-todo.md" in paths


# ── _render_today_prompt — extras (cross-source signals) ────────────────────


def _ev_minimal():
    """Evidence shell (notas tocadas + 1 inbox) que asegura `total > 0`."""
    return {
        "recent_notes": [
            {"title": "Nota A", "path": "02-Areas/A.md", "snippet": "cuerpo A"}
        ],
        "inbox_today": [],
        "todos": [],
        "new_contradictions": [],
        "low_conf_queries": [],
        "wa_scheduled_today_pending": [],
    }


def test_render_today_prompt_no_extras_backward_compatible():
    """Sin extras, sigue funcionando como antes — 4 secciones obligatorias."""
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal())
    for h in ("Lo que pasó hoy", "Sin procesar", "Preguntas abiertas", "Para mañana"):
        assert f"## " in prompt and h in prompt
    # Sin extras, no aparecen los buckets DATA — chequeamos los headers
    # de bucket específicos (no la palabra suelta "Gmail" que aparece en
    # el preámbulo explicativo).
    assert "## 📧 Gmail" not in prompt
    assert "## 💬 WhatsApp — recibido HOY" not in prompt
    assert "## 💬 WhatsApp — esperando" not in prompt
    assert "## 📺 YouTube" not in prompt


def test_render_today_prompt_includes_gmail_when_provided():
    extras = {
        "gmail_unread": {
            "unread_count": 12,
            "awaiting_reply": [
                {"subject": "Reunión proyecto X", "sender": "fer@ejemplo.com",
                 "days_old": 3},
                {"subject": "Factura abril", "sender": "billing@x.io",
                 "days_old": 1},
            ],
        }
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Gmail" in prompt
    assert "12" in prompt  # unread count
    assert "Reunión proyecto X" in prompt
    assert "fer@ejemplo.com" in prompt


def test_render_today_prompt_includes_wa_unreplied():
    extras = {
        "whatsapp_unreplied": [
            {"name": "Marina", "jid": "549@s.whatsapp.net",
             "last_snippet": "che, te respondo después", "hours_waiting": 26.5},
            {"name": "Equipo X", "jid": "120@g.us",
             "last_snippet": "alguien viene mañana?", "hours_waiting": 8.0},
        ]
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "WhatsApp" in prompt and "respond" in prompt.lower()
    assert "Marina" in prompt
    assert "26" in prompt or "27" in prompt  # hours_waiting redondeado


def test_render_today_prompt_includes_calendar_tomorrow():
    extras = {
        "tomorrow_calendar": [
            {"title": "Sync con Pablo", "date_label": "mañana",
             "time_range": "10:00–11:00"},
            {"title": "Dentista", "date_label": "mañana",
             "time_range": "16:30–17:30"},
        ]
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Sync con Pablo" in prompt
    assert "10:00" in prompt


def test_render_today_prompt_includes_youtube_drive_bookmarks():
    extras = {
        "youtube_watched": [
            {"title": "Video de RAG", "url": "https://yt/abc",
             "video_id": "abc", "visit_count": 1,
             "last_visit_iso": "2026-04-21T10:00"},
        ],
        "drive_recent": [
            {"name": "Spec X.pdf", "last_modified": "2026-04-21T12:00"},
        ],
        "chrome_bookmarks": [
            {"name": "Hacker News", "url": "https://news.ycombinator.com",
             "folder": "tech", "visit_count": 4,
             "last_visit_iso": "2026-04-21T08:00"},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Video de RAG" in prompt
    assert "Spec X.pdf" in prompt
    assert "Hacker News" in prompt


def test_render_today_prompt_includes_gmail_today_bucket():
    """gmail_today (mails recibidos HOY) debe aparecer ANTES que
    gmail_unread (rolling window) en el prompt — es la señal más
    accionable para el evening brief.
    """
    extras = {
        "gmail_today": [
            {"subject": "Reunión 14hs", "from": "fer@x.com",
             "snippet": "confirmás?", "thread_id": "t1",
             "internal_date_ms": 1_700_000_000_000},
            {"subject": "Factura", "from": "billing@svc.com",
             "snippet": "tu factura abril", "thread_id": "t2",
             "internal_date_ms": 1_700_000_001_000},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Gmail — recibido HOY" in prompt
    assert "Reunión 14hs" in prompt
    assert "fer@x.com" in prompt
    assert "confirmás?" in prompt
    # Gmail today aparece antes de los buckets rolling
    idx_today = prompt.find("Gmail — recibido HOY")
    idx_rolling = prompt.find("Gmail — bandeja al cierre")
    if idx_rolling != -1:
        assert idx_today < idx_rolling


def test_render_today_prompt_includes_whatsapp_today_bucket():
    """whatsapp_today (mensajes recibidos HOY) ≠ whatsapp_unreplied (chats
    donde tardás en responder). El primero es lo que llegó hoy, el
    segundo lo que tenés pendiente acumulado.
    """
    extras = {
        "whatsapp_today": [
            {"name": "Marina", "jid": "549@s.whatsapp.net",
             "count": 5, "last_snippet": "che nos vemos mañana?"},
            {"name": "Equipo X", "jid": "120@g.us",
             "count": 12, "last_snippet": "alguien viene?"},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "WhatsApp — recibido HOY" in prompt
    assert "Marina" in prompt
    assert "5 msgs" in prompt
    assert "che nos vemos mañana?" in prompt


def test_render_today_prompt_includes_calendar_today_bucket():
    """calendar_today (eventos del día — pasados + futuros del día) ≠
    tomorrow_calendar (eventos de mañana).
    """
    extras = {
        "calendar_today": [
            {"title": "Standup", "start": "10:00", "end": "10:30"},
            {"title": "Demo cliente", "start": "14:00", "end": "15:00"},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Calendar — eventos de HOY" in prompt
    assert "Standup" in prompt
    assert "10:00–10:30" in prompt
    assert "Demo cliente" in prompt


def test_render_today_prompt_includes_youtube_today_bucket():
    """youtube_today (visto HOY) ≠ youtube_recent (últimos 7d sin hoy)."""
    extras = {
        "youtube_today": [
            {"title": "Video viejo de Ghostty", "url": "https://yt/abc",
             "video_id": "abc"},
        ],
        "youtube_recent": [
            {"title": "Video de la semana", "url": "https://yt/def",
             "video_id": "def"},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "YouTube — visto HOY" in prompt
    assert "Video viejo de Ghostty" in prompt
    assert "YouTube — visto últimos 7 días" in prompt
    assert "Video de la semana" in prompt
    # El de hoy aparece ANTES que el de la semana
    assert prompt.find("YouTube — visto HOY") < prompt.find("YouTube — visto últimos 7")


def test_render_today_prompt_youtube_recent_back_compat_with_watched_key():
    """Tests viejos pasaban `youtube_watched` — la key nueva es
    `youtube_recent` pero el render acepta ambas para no romper.
    """
    extras = {"youtube_watched": [{"title": "Old key still works",
                                    "video_id": "xyz"}]}
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    assert "Old key still works" in prompt


def test_render_today_prompt_renders_correlations_block():
    """Cuando `extras["correlations"]` viene populated por el correlator
    Python, el prompt debe renderear un bloque '🔗 ENTIDADES CROSS-SOURCE'
    al inicio con las personas y temas ya matched.
    """
    extras = {
        "correlations": {
            "people": [
                {"name": "Pablo Fer", "sources_count": 3,
                 "appearances": [
                     {"source": "gmail_today",
                      "context": "Reunión 14hs", "snippet": ""},
                     {"source": "whatsapp",
                      "context": "5 msgs hoy", "snippet": "che"},
                     {"source": "calendar",
                      "context": "mañana 10:00–11:00", "snippet": "Sync"},
                 ]},
            ],
            "topics": [
                {"topic": "ghostty", "sources": ["youtube", "notas"],
                 "sources_count": 2},
            ],
        }
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    # Header del bloque DATA (como H2)
    assert "## 🔗 ENTIDADES CROSS-SOURCE" in prompt
    # Contenido populated
    assert "Pablo Fer" in prompt
    assert "3 fuentes" in prompt
    assert "ghostty" in prompt
    assert "youtube" in prompt and "notas" in prompt
    # Aparece ANTES del bloque "Notas tocadas hoy"
    idx_corr = prompt.find("## 🔗 ENTIDADES CROSS-SOURCE")
    idx_notas = prompt.find("## Notas tocadas hoy")
    assert idx_corr < idx_notas


def test_render_today_prompt_skips_correlations_block_when_empty():
    """Si correlations viene vacío, NO se renderea el bloque (evita ruido
    en el prompt para días sin cross-source detectables).
    """
    extras = {
        "correlations": {"people": [], "topics": []},
        "gmail_today": [
            {"from": "x", "subject": "test", "snippet": "..."},
        ],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    # El HEADER (H2) NO debe aparecer cuando los buckets están vacíos.
    # La frase 'ENTIDADES CROSS-SOURCE' aparece en las instructions de
    # la sección 🔗 Conexiones del día (referencia descripta), pero NO
    # como header del data block.
    assert "## 🔗 ENTIDADES CROSS-SOURCE" not in prompt


def test_render_today_prompt_asks_for_cross_source_matching():
    """El prompt debe instruir explícitamente al LLM a buscar conexiones
    entre fuentes (gmail/wa/calendar/notas) cuando hay extras de varias
    fuentes — esto es lo que diferencia un dump plano de un brief útil.
    """
    extras = {
        "gmail_unread": {"unread_count": 5, "awaiting_reply": []},
        "whatsapp_unreplied": [{"name": "X", "jid": "j", "last_snippet": "hola",
                                "hours_waiting": 5}],
        "tomorrow_calendar": [{"title": "T", "date_label": "mañana",
                               "time_range": "10:00"}],
    }
    prompt = rag._render_today_prompt("2026-04-21", _ev_minimal(), extras=extras)
    # Buscamos cualquier referencia a cross-source / conexiones / agrupar.
    lower = prompt.lower()
    assert any(
        marker in lower
        for marker in ("cross-source", "cross source", "conexion",
                       "conexión", "agrupá", "agrupa ",
                       "entre fuentes", "patrón")
    ), f"prompt no pide cross-source matching:\n{prompt}"


# ── _generate_today_narrative — model selection ─────────────────────────────


def test_generate_today_narrative_uses_qwen7b_by_default(monkeypatch):
    """Default del brief: qwen2.5:7b (mismo del chat) por velocidad.
    qwen2.5:14b se descartó después de medir 187s end-to-end en mac
    (timeout). Override por env si el user tiene paciencia / GPU mejor.
    """
    captured = {}

    def _fake_chat(model, messages, options=None, keep_alive=None):
        captured["model"] = model
        captured["options"] = options
        return _FakeResponse("ok")

    fake_client = type("_C", (), {"chat": staticmethod(_fake_chat)})()
    monkeypatch.setattr(rag, "_today_brief_client", lambda: fake_client)
    monkeypatch.delenv("OBSIDIAN_RAG_TODAY_MODEL", raising=False)

    out = rag._generate_today_narrative("hola")
    assert out == "ok"
    assert captured["model"] == "qwen2.5:7b"
    # Options del brief, NO el CHAT_OPTIONS general (sin temperature=0
    # + sin seed para variedad; same num_ctx 4096 para reusar KV cache
    # del chat).
    assert captured["options"]["temperature"] == 0.4
    assert captured["options"]["num_ctx"] == 4096
    assert "seed" not in captured["options"]  # variedad entre runs


def test_generate_today_narrative_respects_env_override(monkeypatch):
    captured = {}

    def _fake_chat(model, messages, options=None, keep_alive=None):
        captured["model"] = model
        return _FakeResponse("ok")

    fake_client = type("_C", (), {"chat": staticmethod(_fake_chat)})()
    monkeypatch.setattr(rag, "_today_brief_client", lambda: fake_client)
    monkeypatch.setenv("OBSIDIAN_RAG_TODAY_MODEL", "command-r:latest")

    rag._generate_today_narrative("hola")
    assert captured["model"] == "command-r:latest"


# ── CLI `rag today` ──────────────────────────────────────────────────────────


def test_today_cli_silent_no_op_when_empty(tmp_vault, monkeypatch):
    monkeypatch.setattr(rag, "LOG_PATH", tmp_vault[2] / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_vault[2] / "c.jsonl")
    # `today` está en `_CLI_WARMUP_SUBCOMMANDS` (rag/__init__.py:19242) →
    # el group callback dispara `warmup_async()` → spawnea threads,
    # incluyendo `_wu_chat_models()` que llama `ollama.chat` 2 veces para
    # pre-cargar el chat model + helper. Acá nos importa solo el path del
    # CLI `today`, no el warmup global, así que lo desactivamos vía la
    # env var oficial. Ver `warmup_async()` en rag/__init__.py:12574.
    monkeypatch.setenv("RAG_NO_WARMUP", "1")
    called = []

    def _boom(*a, **kw):
        called.append(True)
        return _FakeResponse("must not run")
    monkeypatch.setattr(rag.ollama, "chat", _boom)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    assert "sin actividad hoy" in result.output
    assert called == []


def test_today_cli_dry_run_does_not_write(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    p = vault / "02-Areas" / "activity.md"
    p.write_text("algo hoy")
    _set_mtime(p, _FROZEN_NOW - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--dry-run", "--plain"])
    assert result.exit_code == 0
    assert "type: evening-brief" in result.output
    # 4 expected headers
    for h in ("Lo que pasó hoy", "Sin procesar", "Preguntas abiertas", "Para mañana"):
        assert h in result.output
    # No file written to 04-Archive/99-obsidian-system/99-AI/reviews
    files = list((vault / "04-Archive/99-obsidian-system/99-AI/reviews").glob("*.md"))
    assert files == []


def test_today_cli_writes_evening_suffix_and_frontmatter(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    p = vault / "02-Areas" / "activity.md"
    p.write_text("algo hoy")
    _set_mtime(p, _FROZEN_NOW - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    date_label = _FROZEN_NOW.strftime("%Y-%m-%d")
    expected = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / f"{date_label}-evening.md"
    assert expected.is_file(), result.output
    body = expected.read_text()
    assert "type: evening-brief" in body
    assert f"date: '{date_label}'" in body
    assert "- evening-brief" in body
    for h in ("Lo que pasó hoy", "Sin procesar", "Preguntas abiertas", "Para mañana"):
        assert h in body


def test_today_cli_does_not_collide_with_morning_file(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    date_label = _FROZEN_NOW.strftime("%Y-%m-%d")
    morning_file = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / f"{date_label}.md"
    morning_file.write_text("morning brief existente")

    p = vault / "02-Areas" / "a.md"
    p.write_text("contenido")
    _set_mtime(p, _FROZEN_NOW - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    # Morning file untouched
    assert morning_file.read_text() == "morning brief existente"
    # Evening file present separately
    assert (vault / "04-Archive/99-obsidian-system/99-AI/reviews" / f"{date_label}-evening.md").is_file()


def test_today_plist_registered_in_services(tmp_path):
    spec = rag._services_spec("/tmp/fake-rag")
    labels = [s[0] for s in spec]
    assert "com.fer.obsidian-rag-today" in labels
    today_entry = next(s for s in spec if s[0] == "com.fer.obsidian-rag-today")
    plist_xml = today_entry[2]
    assert "<string>today</string>" in plist_xml
    for wd in (1, 2, 3, 4, 5):
        assert f"<integer>{wd}</integer>" in plist_xml
    assert "<key>Hour</key><integer>22</integer>" in plist_xml
    assert "today.log" in plist_xml
    assert "today.error.log" in plist_xml


# ── _strip_empty_today_sections ─────────────────────────────────────────────


def test_strip_drops_section_with_nada_quedo_suelto():
    raw = (
        "## 🪞 Lo que pasó hoy\n"
        "Trabajaste en el RAG.\n\n"
        "## 📥 Sin procesar\n"
        "Nada quedó suelto que no haya sido ya categorizado.\n\n"
        "## 🌅 Para mañana\n"
        "Llamar al psiquiatra.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "📥 Sin procesar" not in out
    assert "🪞 Lo que pasó hoy" in out
    assert "🌅 Para mañana" in out
    assert "Llamar al psiquiatra" in out


def test_strip_drops_section_with_no_hay_datos_suficientes():
    raw = (
        "## 🔍 Preguntas abiertas\n"
        "No hay datos suficientes para responder.\n\n"
        "## 🌅 Para mañana\n"
        "Revisar el calendario.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "🔍 Preguntas abiertas" not in out
    assert "🌅 Para mañana" in out


def test_strip_drops_section_with_ninguna():
    raw = (
        "## 🔍 Preguntas abiertas\n"
        "Ninguna pregunta abierta hoy.\n\n"
        "## 🪞 Lo que pasó hoy\n"
        "Día tranquilo.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "🔍 Preguntas abiertas" not in out
    assert "🪞 Lo que pasó hoy" in out


def test_strip_drops_section_with_no_hubo():
    raw = (
        "## 📥 Sin procesar\n"
        "No hubo capturas hoy ni mails sin responder.\n\n"
        "## 🪞 Lo que pasó hoy\n"
        "Trabajé en X.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "📥 Sin procesar" not in out
    assert "🪞 Lo que pasó hoy" in out


def test_strip_drops_section_header_only():
    # LLM emite el header sin body — debería caer también
    raw = (
        "## 🪞 Lo que pasó hoy\n"
        "Día activo.\n\n"
        "## 📥 Sin procesar"  # sin newline ni body
    )
    out = rag._strip_empty_today_sections(raw)
    assert "📥 Sin procesar" not in out
    assert "Día activo" in out


def test_strip_keeps_section_with_real_content():
    raw = (
        "## 🪞 Lo que pasó hoy\n"
        "Avanzaste con el RAG y mandaste un mensaje a Grecia.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "🪞 Lo que pasó hoy" in out
    assert "Avanzaste con el RAG" in out


def test_strip_handles_empty_input():
    assert rag._strip_empty_today_sections("") == ""
    assert rag._strip_empty_today_sections(None) is None  # type: ignore[arg-type]


def test_strip_section_mixed_real_and_placeholder_keeps_real():
    raw = (
        "## 🪞 Lo que pasó hoy\n"
        "Trabajaste en X.\n"
        "Charlaste con Y.\n"
        "Nada quedó suelto.\n"  # una línea placeholder mezclada
    )
    out = rag._strip_empty_today_sections(raw)
    # La sección entera tiene 2 líneas reales + 1 placeholder, NO se borra
    assert "🪞 Lo que pasó hoy" in out
    assert "Trabajaste en X" in out


def test_strip_accent_folds_for_match():
    # "no había" con tilde, "ningún" con tilde — el regex está en accent-fold
    raw = (
        "## 📥 Sin procesar\n"
        "No había capturas hoy.\n\n"
        "## 🌅 Para mañana\n"
        "Avanzar con el feature.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "📥 Sin procesar" not in out


def test_strip_drops_multiple_placeholder_sections():
    raw = (
        "## 🪞 Lo que pasó hoy\n"
        "Día activo.\n\n"
        "## 📥 Sin procesar\n"
        "Nada quedó suelto.\n\n"
        "## 🔍 Preguntas abiertas\n"
        "Ninguna pregunta hoy.\n\n"
        "## 🌅 Para mañana\n"
        "Sin novedades.\n"
    )
    out = rag._strip_empty_today_sections(raw)
    assert "🪞 Lo que pasó hoy" in out
    assert "Día activo" in out
    # Las otras 3 caen
    assert "📥 Sin procesar" not in out
    assert "🔍 Preguntas abiertas" not in out
    assert "🌅 Para mañana" not in out
