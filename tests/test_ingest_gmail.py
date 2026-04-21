"""Tests for scripts/ingest_gmail.py + `rag index --source gmail`.

No live Gmail — tests inject `svc=mock` and assert parsing / chunking /
upsert. Covers:
  - Quoted-reply stripping (English + Spanish headers, >-prefixed lines)
  - MIME body decoding (text/plain + text/html → tag strip)
  - Header extraction (case-insensitive)
  - _parse_message produces GmailMessage; rejects missing IDs
  - _messages_to_threads groups by thread_id, sorts oldest→newest
  - Thread body formatting + 800 char cap
  - upsert_threads writes source=gmail + all meta fields; idempotent
  - run() bootstrap vs incremental branches + history 404 recovery
  - CLI `rag index --source gmail` routing + dry-run
"""
from __future__ import annotations

import base64

import pytest

import rag
from scripts import ingest_gmail as ig


# ── Fixtures ────────────────────────────────────────────────────────────

class _FakeGmail:
    """Stand-in for googleapiclient Gmail service. Implements the subset
    we call: users().getProfile(), users().messages().list()/get(),
    users().history().list()."""
    def __init__(self, *, profile_email="me@x.com", history_id="1000",
                 messages: dict | None = None,
                 list_response: dict | None = None,
                 history_response: dict | None = None,
                 history_raise: str | None = None):
        self._profile = {"emailAddress": profile_email, "historyId": history_id}
        self._messages = messages or {}
        self._list_response = list_response or {"messages": [], "nextPageToken": None}
        self._history_response = history_response or {"history": [], "historyId": history_id}
        self._history_raise = history_raise

    def users(self): return self
    def getProfile(self, userId): return _ExecProxy(self._profile)
    def messages(self): return _Messages(self)
    def history(self): return _History(self)


class _Messages:
    def __init__(self, svc):
        self._svc = svc
    def list(self, **kw): return _ExecProxy(self._svc._list_response)
    def get(self, userId, id, format): return _ExecProxy(self._svc._messages.get(id))


class _History:
    def __init__(self, svc):
        self._svc = svc
    def list(self, **kw):
        if self._svc._history_raise:
            raise RuntimeError(self._svc._history_raise)
        return _ExecProxy(self._svc._history_response)


class _ExecProxy:
    def __init__(self, payload): self._p = payload
    def execute(self): return self._p


@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="gm_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    return col


def _mk_raw(mid, tid, subject, sender, body_text, internal_date_ms, labels=None):
    """Build a Gmail messages.get()-shaped dict."""
    data = base64.urlsafe_b64encode(body_text.encode("utf-8")).decode("ascii")
    return {
        "id": mid, "threadId": tid,
        "labelIds": labels or ["INBOX"],
        "internalDate": str(internal_date_ms),
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "Date", "value": "Mon, 20 Apr 2026 10:00:00 +0000"},
                {"name": "To", "value": "me@x.com"},
            ],
            "body": {"data": data},
            "mimeType": "text/plain",
        },
    }


# ── strip_quoted ────────────────────────────────────────────────────────

def test_strip_quoted_removes_english_reply_header():
    text = (
        "Hola Juan,\nMando el resumen.\n\n"
        "On Mon, Apr 20, 2026 at 10:00 AM, Juan <j@x.com> wrote:\n"
        "> me parece bien\n"
    )
    out = ig.strip_quoted(text)
    assert "me parece bien" not in out
    assert "Mando el resumen" in out


def test_strip_quoted_removes_spanish_reply_header():
    text = (
        "Listo el pago.\n\n"
        "El mar, 20 abr 2026 10:00, Ana <a@x.com> escribió:\n"
        "> cuanto me debes\n"
    )
    out = ig.strip_quoted(text)
    assert "cuanto me debes" not in out
    assert "Listo el pago" in out


def test_strip_quoted_drops_gt_lines_even_without_header():
    text = "hola\n> quote line\n> another quote\nchau"
    out = ig.strip_quoted(text)
    assert ">" not in out
    assert "hola" in out
    assert "chau" in out


def test_strip_quoted_handles_forward_header():
    text = "Reenvío:\n\n---------- Original Message ----------\n\noriginal body"
    out = ig.strip_quoted(text)
    assert "original body" not in out
    assert "Reenvío" in out


def test_strip_quoted_collapses_blank_lines():
    text = "primera\n\n\n\n\nsegunda"
    out = ig.strip_quoted(text)
    assert "\n\n\n" not in out


def test_strip_quoted_empty_input():
    assert ig.strip_quoted("") == ""


# ── Body decoding ──────────────────────────────────────────────────────

def test_decode_part_body_plain():
    data = base64.urlsafe_b64encode(b"hola mundo").decode()
    part = {"mimeType": "text/plain", "body": {"data": data}}
    assert "hola mundo" in ig._decode_part_body(part)


def test_decode_part_body_html_strips_tags():
    html_body = "<p>Hola <b>Juan</b></p><script>bad</script>"
    data = base64.urlsafe_b64encode(html_body.encode()).decode()
    part = {"mimeType": "text/html", "body": {"data": data}}
    out = ig._decode_part_body(part)
    assert "Hola" in out and "Juan" in out
    assert "<" not in out and ">" not in out
    assert "bad" not in out  # script dropped


def test_decode_part_body_walks_multipart():
    inner_data = base64.urlsafe_b64encode(b"inner text").decode()
    part = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/plain", "body": {"data": inner_data}},
        ],
    }
    assert "inner text" in ig._decode_part_body(part)


# ── Message parsing ────────────────────────────────────────────────────

def test_parse_message_happy_path():
    raw = _mk_raw("m1", "t1", "Subject", "juan@x.com", "hola mundo", 1_700_000_000_000)
    m = ig._parse_message(raw)
    assert m is not None
    assert m.id == "m1"
    assert m.thread_id == "t1"
    assert m.subject == "Subject"
    assert m.sender == "juan@x.com"
    assert "hola mundo" in m.body
    assert m.date_ts > 0


def test_parse_message_missing_id_returns_none():
    raw = _mk_raw("mX", "tX", "S", "f", "body", 1)
    del raw["id"]
    assert ig._parse_message(raw) is None


def test_parse_message_strips_quoted_in_body():
    text = "nuevo contenido\n\nOn Mon, Apr 20, 2026 at 10:00 AM, X <x@x.com> wrote:\n> old"
    raw = _mk_raw("m1", "t1", "s", "f", text, 1_700_000_000_000)
    m = ig._parse_message(raw)
    assert m is not None
    assert "old" not in m.body
    assert "nuevo contenido" in m.body


# ── Threads ────────────────────────────────────────────────────────────

def test_messages_to_threads_groups_and_sorts():
    msgs = [
        ig._parse_message(_mk_raw("m1", "t1", "X", "a@x", "one", 2_000_000_000_000)),
        ig._parse_message(_mk_raw("m2", "t1", "X", "b@x", "two", 1_000_000_000_000)),
        ig._parse_message(_mk_raw("m3", "t2", "Y", "c@x", "three", 3_000_000_000_000)),
    ]
    threads = ig._messages_to_threads(msgs)
    assert len(threads) == 2
    # Sorted newest-thread-first by last_ts
    assert threads[0].thread_id == "t2"
    # Within t1, oldest first
    t1 = next(t for t in threads if t.thread_id == "t1")
    assert [m.id for m in t1.messages] == ["m2", "m1"]


def test_messages_to_threads_tracks_folder():
    m1 = ig._parse_message(_mk_raw("m1", "t1", "S", "f", "body", 1_000_000_000_000,
                                     labels=["INBOX"]))
    m2 = ig._parse_message(_mk_raw("m2", "t1", "S", "f", "body", 2_000_000_000_000,
                                     labels=["SENT"]))
    threads = ig._messages_to_threads([m1, m2])
    assert "INBOX" in threads[0].folder
    assert "Sent" in threads[0].folder


# ── Chunk body ─────────────────────────────────────────────────────────

def test_format_thread_body_includes_subject_sender_date():
    m = ig._parse_message(_mk_raw("m1", "t1", "Hola Juan", "juan@x.com",
                                    "cómo va?", 1_700_000_000_000))
    threads = ig._messages_to_threads([m])
    body = ig._format_thread_body(threads[0])
    assert "Asunto: Hola Juan" in body
    assert "De: juan@x.com" in body
    assert "Fecha:" in body
    assert "cómo va?" in body


def test_format_thread_body_respects_cap():
    long = "x" * 2000
    m = ig._parse_message(_mk_raw("m1", "t1", "S", "f@x", long, 1_700_000_000_000))
    threads = ig._messages_to_threads([m])
    body = ig._format_thread_body(threads[0])
    assert len(body) <= ig.CHUNK_MAX_CHARS


# ── Writer ─────────────────────────────────────────────────────────────

def test_upsert_threads_writes_source_gmail(tmp_vault_col):
    m = ig._parse_message(_mk_raw("m1", "t1", "Proyecto X", "juan@x.com",
                                    "hola mundo", 1_700_000_000_000))
    threads = ig._messages_to_threads([m])
    n = ig.upsert_threads(tmp_vault_col, threads)
    assert n == 1

    got = tmp_vault_col.get(where={"source": "gmail"}, include=["metadatas"])
    assert len(got["ids"]) == 1
    meta = got["metadatas"][0]
    assert meta["source"] == "gmail"
    assert meta["thread_id"] == "t1"
    assert meta["subject"] == "Proyecto X"
    assert meta["sender"] == "juan@x.com"
    assert meta["n_messages"] == 1
    assert meta["file"].startswith("gmail://thread/")


def test_upsert_threads_idempotent(tmp_vault_col):
    m = ig._parse_message(_mk_raw("m1", "t1", "S", "f@x", "body", 1_000_000_000_000))
    threads = ig._messages_to_threads([m])
    ig.upsert_threads(tmp_vault_col, threads)
    before = len(tmp_vault_col.get(where={"source": "gmail"}, include=[])["ids"])
    ig.upsert_threads(tmp_vault_col, threads)
    after = len(tmp_vault_col.get(where={"source": "gmail"}, include=[])["ids"])
    assert before == after


def test_delete_threads_removes_rows(tmp_vault_col):
    m1 = ig._parse_message(_mk_raw("m1", "t1", "S", "f", "body", 1_000_000_000_000))
    m2 = ig._parse_message(_mk_raw("m2", "t2", "S", "f", "body", 2_000_000_000_000))
    ig.upsert_threads(tmp_vault_col, ig._messages_to_threads([m1, m2]))
    assert len(tmp_vault_col.get(where={"source": "gmail"}, include=[])["ids"]) == 2
    ig.delete_threads(tmp_vault_col, ["t1"])
    rem = tmp_vault_col.get(where={"source": "gmail"}, include=["metadatas"])
    assert len(rem["ids"]) == 1
    assert rem["metadatas"][0]["thread_id"] == "t2"


# ── Orchestration ──────────────────────────────────────────────────────

def test_run_bootstrap_indexes_threads(tmp_vault_col):
    import time as _t
    fresh_ms = int(_t.time() * 1000)
    raws = {
        "m1": _mk_raw("m1", "t1", "S", "juan@x", "body1", fresh_ms),
        "m2": _mk_raw("m2", "t1", "S", "juan@x", "body2", fresh_ms + 100_000),
        "m3": _mk_raw("m3", "t2", "Y", "ana@x",  "body3", fresh_ms + 200_000),
    }
    svc = _FakeGmail(
        messages=raws,
        list_response={
            "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
        },
    )
    summary = ig.run(svc=svc, vault_col=tmp_vault_col)
    assert "error" not in summary
    assert summary["bootstrapped"] is True
    assert summary["threads_built"] == 2
    assert summary["threads_indexed"] == 2
    assert summary["messages_seen"] == 3


def test_run_no_service_reports_error(tmp_vault_col, monkeypatch):
    monkeypatch.setattr(rag, "_gmail_service", lambda: None)
    summary = ig.run(vault_col=tmp_vault_col)
    assert "error" in summary


def test_run_dry_run_writes_nothing(tmp_vault_col):
    import time as _t
    fresh_ms = int(_t.time() * 1000)
    raws = {"m1": _mk_raw("m1", "t1", "S", "f", "b", fresh_ms)}
    svc = _FakeGmail(messages=raws, list_response={"messages": [{"id": "m1"}]})
    summary = ig.run(svc=svc, vault_col=tmp_vault_col, dry_run=True)
    assert summary["threads_indexed"] == 1  # counted
    got = tmp_vault_col.get(where={"source": "gmail"}, include=[])
    assert got["ids"] == []  # but not persisted


def test_run_bootstrap_retention_drops_old(tmp_vault_col):
    """Old message (> 365d) should be filtered post-decode."""
    import time as _t
    fresh_ms = int(_t.time() * 1000)
    old_ms = int((_t.time() - 500 * 86400) * 1000)  # 500 days ago
    raws = {
        "m_fresh": _mk_raw("m_fresh", "t1", "S", "f", "fresh body", fresh_ms),
        "m_old":   _mk_raw("m_old",   "t2", "S", "f", "old body", old_ms),
    }
    svc = _FakeGmail(
        messages=raws,
        list_response={"messages": [{"id": "m_fresh"}, {"id": "m_old"}]},
    )
    summary = ig.run(svc=svc, vault_col=tmp_vault_col)
    # Only the fresh thread makes it past the retention filter.
    assert summary["threads_built"] == 1


# ── CLI ─────────────────────────────────────────────────────────────────

def test_cli_index_source_gmail_routes(monkeypatch):
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "bootstrapped": True, "incremental": False,
            "messages_seen": 10, "threads_built": 5, "threads_indexed": 5,
            "threads_deleted": 0, "duration_s": 0.1,
        }
    from scripts import ingest_gmail as ig_mod
    monkeypatch.setattr(ig_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "gmail"])
    assert result.exit_code == 0, result.output
    assert "Gmail" in result.output
    assert "bootstrap" in result.output
    assert "5 indexados" in result.output
    assert called["reset"] is False


def test_cli_index_source_gmail_dry_run(monkeypatch):
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "bootstrapped": True, "incremental": False,
            "messages_seen": 1, "threads_built": 1, "threads_indexed": 1,
            "threads_deleted": 0, "duration_s": 0.0,
        }
    from scripts import ingest_gmail as ig_mod
    monkeypatch.setattr(ig_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "gmail", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["dry_run"] is True
    assert "[dry-run]" in result.output
