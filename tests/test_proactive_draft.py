"""Tests para rag/proactive_draft.py — Fase 5."""

from __future__ import annotations

import json

import pytest

import rag
from rag import SqliteVecClient as _TestVecClient
from rag import proactive_draft as pd


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry DB en tmp_path. Mismo patrón que test_anticipate_agent.py."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    col = client.get_or_create_collection(
        name="proactive_draft_test", metadata={"hnsw:space": "cosine"},
    )
    with rag._ragvec_state_conn() as _conn:
        pass
    return col


# ── Pure functions (no LLM, no DB) ──────────────────────────────────────────


def test_disabled_env_var(monkeypatch):
    monkeypatch.setenv("RAG_PROACTIVE_DRAFTS_DISABLE", "1")
    assert pd._disabled() is True
    monkeypatch.setenv("RAG_PROACTIVE_DRAFTS_DISABLE", "")
    assert pd._disabled() is False


def test_when_phrase_thresholds():
    assert pd._when_phrase(-1) == "ya está vencido"
    assert pd._when_phrase(0.5) == "vence hoy"
    assert pd._when_phrase(1.5) == "vence mañana"
    assert pd._when_phrase(5) == "vence en 5 días"


def test_features_summary_empty_returns_fallback():
    assert "voseo argentino estándar" in pd._features_summary({})
    assert "voseo argentino estándar" in pd._features_summary({"insufficient_data": True})


def test_features_summary_full_includes_all_signals():
    out = pd._features_summary({
        "voseo_dominance": 0.95,
        "avg_chars_per_msg": 80,
        "emoji_rate": 0.20,
        "top_openers": ["ma", "che"],
        "top_closers": ["dale"],
        "slang_argentino_hits": {"che": 10, "dale": 5},
    })
    assert "0.95" in out
    assert "80" in out
    assert "usá emojis libremente" in out
    assert "ma" in out and "che" in out


def test_portuguese_leak_detector():
    assert pd._portuguese_leak("você obrigado por isso") is True
    assert pd._portuguese_leak("essa é tua tarea") is True
    assert pd._portuguese_leak("Ma! mando las fotos ahora") is False
    assert pd._portuguese_leak("") is False


def test_parse_draft_response_happy_path():
    raw = '{"draft":"Ma! mando las fotos","confidence":0.85,"reason":"casual"}'
    out = pd._parse_draft_response(raw, max_chars=280)
    assert out == {"draft": "Ma! mando las fotos", "confidence": 0.85, "reason": "casual"}


def test_parse_draft_response_strips_markdown_fence():
    raw = '```json\n{"draft":"hola","confidence":0.5}\n```'
    out = pd._parse_draft_response(raw, max_chars=280)
    assert out is not None and out["draft"] == "hola"


def test_parse_draft_response_truncates_long_draft():
    long_text = "x" * 500
    raw = json.dumps({"draft": long_text, "confidence": 0.5})
    out = pd._parse_draft_response(raw, max_chars=100)
    assert out is not None
    assert len(out["draft"]) <= 100
    assert out["draft"].endswith("…")


def test_parse_draft_response_rejects_portuguese_leak():
    raw = '{"draft":"você é um bom amigo","confidence":0.9}'
    assert pd._parse_draft_response(raw, max_chars=280) is None


def test_parse_draft_response_rejects_garbage():
    assert pd._parse_draft_response("no json here at all", max_chars=280) is None
    assert pd._parse_draft_response("", max_chars=280) is None
    assert pd._parse_draft_response('{"draft":""}', max_chars=280) is None


def test_parse_draft_response_clamps_confidence():
    raw = '{"draft":"x","confidence":2.5}'
    out = pd._parse_draft_response(raw, max_chars=280)
    assert out is not None and out["confidence"] == 1.0
    raw = '{"draft":"x","confidence":-0.3}'
    out = pd._parse_draft_response(raw, max_chars=280)
    assert out is not None and out["confidence"] == 0.0


# ── Allowlist ───────────────────────────────────────────────────────────────


def test_allowlist_empty_jid_denies():
    assert pd._allowlist_check("") is False


def test_allowlist_zero_threshold_allows_anything(monkeypatch):
    monkeypatch.setenv("RAG_PROACTIVE_DRAFTS_MIN_MSGS", "0")
    assert pd._allowlist_check("any@jid") is True


# ── compose_draft (full path with mocked LLM) ───────────────────────────────


def test_compose_draft_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("RAG_PROACTIVE_DRAFTS_DISABLE", "1")
    out = pd.compose_draft(
        target_jid="x@y", target_name="Ma",
        promise_text="prometí mandar fotos", days_until_due=-2,
    )
    assert out is None


def test_compose_draft_returns_none_when_no_jid():
    out = pd.compose_draft(
        target_jid="", target_name="Ma",
        promise_text="x", days_until_due=0,
    )
    assert out is None


def test_compose_draft_returns_none_when_empty_promise():
    out = pd.compose_draft(
        target_jid="x@y", target_name="Ma",
        promise_text="   ", days_until_due=0,
    )
    assert out is None


def test_compose_draft_returns_none_when_allowlist_denies(monkeypatch):
    monkeypatch.setattr(pd, "_allowlist_check", lambda jid, **kw: False)
    out = pd.compose_draft(
        target_jid="x@y", target_name="Ma",
        promise_text="prometí algo", days_until_due=-1,
    )
    assert out is None


def test_compose_draft_happy_path_with_mocked_llm(monkeypatch):
    monkeypatch.setattr(pd, "_allowlist_check", lambda jid, **kw: True)

    class _FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": json.dumps({
                "draft": "Ma! mando las fotos del finde ahora 📸",
                "confidence": 0.78,
                "reason": "menciona la promesa sin sonar acusatorio",
            })}}
    monkeypatch.setattr(rag, "_helper_client", lambda: _FakeClient())

    out = pd.compose_draft(
        target_jid="549110@s.whatsapp.net", target_name="Ma",
        promise_text="le prometí mandar las fotos del finde",
        days_until_due=-1,
    )
    assert out is not None
    assert "fotos" in out["draft"].lower()
    assert 0.0 <= out["confidence"] <= 1.0
    assert "style_snapshot_hash" in out


def test_compose_draft_silent_fail_when_llm_raises(monkeypatch):
    monkeypatch.setattr(pd, "_allowlist_check", lambda jid, **kw: True)

    class _BoomClient:
        def chat(self, **kwargs):
            raise TimeoutError("helper timeout")
    monkeypatch.setattr(rag, "_helper_client", lambda: _BoomClient())

    out = pd.compose_draft(
        target_jid="x@y", target_name="X",
        promise_text="x", days_until_due=0,
    )
    assert out is None


def test_compose_draft_silent_fail_when_llm_returns_garbage(monkeypatch):
    monkeypatch.setattr(pd, "_allowlist_check", lambda jid, **kw: True)

    class _GarbageClient:
        def chat(self, **kwargs):
            return {"message": {"content": "no json output at all"}}
    monkeypatch.setattr(rag, "_helper_client", lambda: _GarbageClient())

    out = pd.compose_draft(
        target_jid="x@y", target_name="X",
        promise_text="x", days_until_due=0,
    )
    assert out is None


# ── push_draft_to_listener (http) ───────────────────────────────────────────


def test_push_draft_silent_fail_when_listener_down(monkeypatch):
    # Apunta a port que nadie escucha — connection refused → silent False.
    monkeypatch.setenv("RAG_LISTENER_PUSH_URL", "http://127.0.0.1:1/push-pending-draft")
    ok = pd.push_draft_to_listener(
        draft_id="abc12345", target_jid="x@y", target_name="X",
        draft_text="hola", signal_kind="anticipate-commitment",
    )
    assert ok is False


def test_push_draft_rejects_empty_inputs():
    assert pd.push_draft_to_listener(
        draft_id="x", target_jid="", target_name="", draft_text="x", signal_kind="k",
    ) is False
    assert pd.push_draft_to_listener(
        draft_id="x", target_jid="x", target_name="", draft_text="", signal_kind="k",
    ) is False


def test_push_draft_succeeds_against_mock_server(monkeypatch):
    import http.server
    import threading

    captured = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            ln = int(self.headers.get("content-length", 0))
            captured.append(json.loads(self.rfile.read(ln) or b"{}"))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a, **k):  # silence stderr
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("RAG_LISTENER_PUSH_URL",
                           f"http://127.0.0.1:{port}/push-pending-draft")
        ok = pd.push_draft_to_listener(
            draft_id="deadbeef", target_jid="549110@s.whatsapp.net",
            target_name="Ma", draft_text="Ma! mando ya 📸",
            signal_kind="anticipate-commitment",
        )
    finally:
        srv.shutdown()
        srv.server_close()

    assert ok is True
    assert len(captured) == 1
    body = captured[0]
    assert body["draft_id"] == "deadbeef"
    assert body["contact_jid"] == "549110@s.whatsapp.net"
    assert body["contact_name"] == "Ma"
    assert body["source"] == "proactive:anticipate-commitment"


# ── log_proactive_draft (sql) ───────────────────────────────────────────────


def _proactive_drafts_count(state_db_conn) -> int:
    cur = state_db_conn.execute("SELECT COUNT(*) FROM rag_proactive_drafts")
    return int(cur.fetchone()[0])


def test_log_proactive_draft_inserts_row(state_db):
    did = pd._new_draft_id()
    pd.log_proactive_draft(
        draft_id=did, signal_kind="anticipate-commitment",
        signal_dedup_key="commit:abc123", target_jid="x@y",
        target_name="X", draft_text="hola", draft_meta={"confidence": 0.7},
        status="pushed",
    )
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT id, status, draft_text, draft_meta_json FROM rag_proactive_drafts WHERE id=?",
            (did,),
        ).fetchone()
    assert row is not None
    assert row[1] == "pushed"
    assert row[2] == "hola"
    meta = json.loads(row[3])
    assert meta["confidence"] == 0.7


def test_log_proactive_draft_rejects_invalid_status(state_db):
    pd.log_proactive_draft(
        draft_id="badstatus", signal_kind="x", signal_dedup_key="x",
        target_jid="x", target_name="x", draft_text="x", draft_meta=None,
        status="invalid_status",
    )
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT id FROM rag_proactive_drafts WHERE id=?", ("badstatus",),
        ).fetchone()
    assert row is None


def test_listener_push_url_default_and_override(monkeypatch):
    monkeypatch.delenv("RAG_LISTENER_PUSH_URL", raising=False)
    assert pd._listener_push_url() == "http://127.0.0.1:8766/push-pending-draft"
    monkeypatch.setenv("RAG_LISTENER_PUSH_URL", "http://10.0.0.1:9999/x")
    assert pd._listener_push_url() == "http://10.0.0.1:9999/x"


def test_new_draft_id_format():
    did = pd._new_draft_id()
    assert len(did) == 8
    int(did, 16)  # raises if not hex


# ── stats() telemetry helper ────────────────────────────────────────────────


def test_stats_empty_db_returns_zero(state_db):
    out = pd.stats(days=7)
    assert out["total_pushed"] == 0
    assert out["total_skipped"] == 0
    assert out["pending"] == 0
    assert out["useful_rate"] == 0.0
    assert out["by_status"] == {}


def test_stats_aggregates_by_status_and_decision(state_db):
    # Insert 3 drafts: 2 pushed, 1 skipped. Plus decisions: 1 approved_si,
    # 1 approved_editar, 0 rejected → useful_rate = 2/2 = 1.0; pending = 0.
    pd.log_proactive_draft(
        draft_id="d_si", signal_kind="anticipate-commitment",
        signal_dedup_key="commit:1", target_jid="x@y", target_name="X",
        draft_text="hola si", draft_meta=None, status="pushed",
    )
    pd.log_proactive_draft(
        draft_id="d_ed", signal_kind="anticipate-commitment",
        signal_dedup_key="commit:2", target_jid="x@y", target_name="X",
        draft_text="hola editar", draft_meta=None, status="pushed",
    )
    pd.log_proactive_draft(
        draft_id="d_skip", signal_kind="anticipate-commitment",
        signal_dedup_key="commit:3", target_jid="x@y", target_name="X",
        draft_text="hola skip", draft_meta=None, status="skipped",
    )
    with rag._ragvec_state_conn() as conn:
        # Insert mock decisions con FK match a draft_id
        from datetime import datetime
        ts = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO rag_draft_decisions"
            " (ts, draft_id, contact_jid, contact_name, bot_draft, decision)"
            " VALUES (?,?,?,?,?,?)",
            (ts, "d_si", "x@y", "X", "hola si", "approved_si"),
        )
        conn.execute(
            "INSERT INTO rag_draft_decisions"
            " (ts, draft_id, contact_jid, contact_name, bot_draft, decision, sent_text)"
            " VALUES (?,?,?,?,?,?,?)",
            (ts, "d_ed", "x@y", "X", "hola editar", "approved_editar", "hola editado"),
        )
        conn.commit()

    out = pd.stats(days=7)
    assert out["total_pushed"] == 2
    assert out["total_skipped"] == 1
    assert out["by_status"] == {"pushed": 2, "skipped": 1}
    assert out["by_decision"] == {"approved_si": 1, "approved_editar": 1}
    assert out["pending"] == 0
    assert out["useful_rate"] == 1.0


def test_stats_useful_rate_with_rejection(state_db):
    pd.log_proactive_draft(
        draft_id="r1", signal_kind="x", signal_dedup_key="k1",
        target_jid="x", target_name="x", draft_text="x",
        draft_meta=None, status="pushed",
    )
    pd.log_proactive_draft(
        draft_id="r2", signal_kind="x", signal_dedup_key="k2",
        target_jid="x", target_name="x", draft_text="x",
        draft_meta=None, status="pushed",
    )
    with rag._ragvec_state_conn() as conn:
        from datetime import datetime
        ts = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO rag_draft_decisions (ts, draft_id, contact_jid, bot_draft, decision)"
            " VALUES (?,?,?,?,?)",
            (ts, "r1", "x", "x", "approved_si"),
        )
        conn.execute(
            "INSERT INTO rag_draft_decisions (ts, draft_id, contact_jid, bot_draft, decision)"
            " VALUES (?,?,?,?,?)",
            (ts, "r2", "x", "x", "rejected"),
        )
        conn.commit()
    out = pd.stats(days=7)
    # 1 si + 0 editar = 1 useful / (1 + 1) decided = 0.5
    assert out["useful_rate"] == 0.5
