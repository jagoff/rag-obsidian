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
    assert "gmail" in result.output
    # `bootstrap` renders as extra token; `threads_indexed=5` as `+5`.
    assert "bootstrap" in result.output
    assert "+5" in result.output
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
    assert "dry · " in result.output


# ── 2026-04-24 audit: cursor advancement post-ingest ─────────────────────


class _FakeGmailWithChangingProfile:
    """Variant of `_FakeGmail` que devuelve un historyId DIFERENTE cada
    vez que se llama getProfile. Permite validar que el ingester usa el
    hid fresco post-ingest, no el pre-run.
    """
    def __init__(self, *, initial_hid: str, fresh_hid: str,
                 messages: dict | None = None,
                 list_response: dict | None = None,
                 history_response: dict | None = None,
                 history_raise: str | None = None):
        self._hids = [initial_hid, fresh_hid]  # FIFO
        self._call_count = 0
        self._messages = messages or {}
        self._list_response = list_response or {"messages": [], "nextPageToken": None}
        self._history_response = history_response or {"history": [], "historyId": initial_hid}
        self._history_raise = history_raise

    def users(self): return self

    def getProfile(self, userId):
        # Return next HID from FIFO (first call = initial, second = fresh).
        hid = self._hids[min(self._call_count, len(self._hids) - 1)]
        self._call_count += 1
        return _ExecProxy({"emailAddress": "me@x.com", "historyId": hid})

    def messages(self): return _Messages(self)
    def history(self): return _History(self)


def test_run_bootstrap_advances_cursor_to_fresh_hid(tmp_vault_col):
    """2026-04-24 audit fix: después de un initial bootstrap (stored_hid
    was None), el cursor debe avanzar al `historyId` fresco re-fetcheado
    post-ingest, no al `profile_hid` pre-run. Sin esto, mensajes que
    llegan DURANTE el fetch de 365 días caen en un gap.
    """
    import time as _t
    from scripts import ingest_gmail as ig_mod
    fresh_ms = int(_t.time() * 1000)
    raws = {
        "m1": _mk_raw("m1", "t1", "S", "juan@x", "body1", fresh_ms),
    }
    svc = _FakeGmailWithChangingProfile(
        initial_hid="1000",
        fresh_hid="2000",  # post-ingest historyId (simula mensajes nuevos en Gmail)
        messages=raws,
        list_response={"messages": [{"id": "m1"}]},
    )
    ig_mod.run(svc=svc, vault_col=tmp_vault_col)

    # Leer el cursor persistido. Debería ser "2000" (fresco) no "1000".
    import sqlite3
    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        row = conn.execute(
            "SELECT history_id FROM rag_gmail_state WHERE account_id = ?",
            ("me@x.com",),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "state row no se persistió"
    assert row[0] == "2000", (
        f"cursor debería avanzar al historyId fresco '2000', "
        f"got '{row[0]}' (bug pre-fix: usaba el stale profile_hid)"
    )


def test_run_fallback_bootstrap_does_not_loop_with_stale_hid(tmp_vault_col):
    """2026-04-24 audit: si el path incremental tira 410 Gone
    (history expired), entramos al fallback bootstrap. Pre-fix,
    `new_hid` caía a `profile_hid` (pre-fetch) porque el else branch
    no existía → próximo run: mismo 410 → loop infinito de bootstraps.

    Post-fix: `get_profile(service)` se llama SIEMPRE post-ingest, así
    que el cursor avanza a un hid NUEVO y el próximo run procesa
    incremental desde ahí (si el hid nuevo también expira, al menos
    avanzamos).
    """
    import sqlite3
    import time as _t
    from scripts import ingest_gmail as ig_mod

    # Seed: stored_hid="500" (existe, not None → no initial bootstrap).
    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_gmail_state ("
            " account_id TEXT PRIMARY KEY, history_id TEXT,"
            " last_msg_id TEXT, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO rag_gmail_state VALUES (?, ?, ?, ?)",
            ("me@x.com", "500", "", "2026-04-24"),
        )
        conn.commit()
    finally:
        conn.close()

    fresh_ms = int(_t.time() * 1000)
    raws = {"m1": _mk_raw("m1", "t1", "S", "juan@x", "body1", fresh_ms)}

    # _FakeGmailWithChangingProfile con initial=1000 (irrelevante porque
    # apply_history tira None) y fresh=3000 (el hid post-bootstrap).
    # `history_raise` con mensaje de 404 fuerza el return ([], [], None)
    # en apply_history → triggers fallback bootstrap.
    svc = _FakeGmailWithChangingProfile(
        initial_hid="1000",
        fresh_hid="3000",
        messages=raws,
        list_response={"messages": [{"id": "m1"}]},
        history_raise="404 historyId not found",  # triggers fallback
    )
    summary = ig_mod.run(svc=svc, vault_col=tmp_vault_col)

    # Verificar que entró al path de fallback bootstrap.
    assert summary.get("bootstrapped") is True, (
        f"esperaba fallback bootstrap cuando apply_history devuelve "
        f"latest_hid=None, got summary: {summary}"
    )
    # Y que el cursor avanzó a "3000" (fresh), no "1000" (profile_hid pre-fetch).
    conn = sqlite3.connect(str(state_path))
    try:
        row = conn.execute(
            "SELECT history_id FROM rag_gmail_state WHERE account_id = ?",
            ("me@x.com",),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "3000", (
        f"fallback bootstrap debería avanzar a hid fresco '3000', "
        f"got '{row[0]}' — pre-fix loopeaba con '1000' (stale)"
    )


# ── CAS (compare-and-swap) del cursor — audit 2026-04-25 ────────────────────


def test_save_history_id_cas_bootstrap_succeeds_when_no_row(tmp_vault_col):
    """Bootstrap (expected=None) e INSERT exitoso → True. La tabla
    queda con la row insertada."""
    import sqlite3
    from scripts import ingest_gmail as ig_mod

    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        ig_mod._ensure_state_table(conn)
        won = ig_mod._save_history_id_cas(
            conn, "me@x.com", "100", expected_history_id=None,
        )
        conn.commit()
        assert won is True
        assert ig_mod._load_history_id(conn, "me@x.com") == "100"
    finally:
        conn.close()


def test_save_history_id_cas_bootstrap_loses_race(tmp_vault_col):
    """Bootstrap concurrente: dos workers leen None y ambos intentan
    INSERT. El primero gana (rowcount=1), el segundo pierde
    (rowcount=0 por OR IGNORE). Sin el CAS pre-fix ambos hacían
    INSERT OR REPLACE y el segundo pisaba al primero con un hid
    distinto."""
    import sqlite3
    from scripts import ingest_gmail as ig_mod

    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        ig_mod._ensure_state_table(conn)
        # Worker 1 hace bootstrap con hid="100".
        a = ig_mod._save_history_id_cas(
            conn, "me@x.com", "100", expected_history_id=None,
        )
        # Worker 2 (concurrente, también leyó None) intenta hid="200".
        # OR IGNORE: no toca la row existente.
        b = ig_mod._save_history_id_cas(
            conn, "me@x.com", "200", expected_history_id=None,
        )
        conn.commit()
        assert a is True
        assert b is False
        # El cursor quedó con el hid del primer worker.
        assert ig_mod._load_history_id(conn, "me@x.com") == "100"
    finally:
        conn.close()


def test_save_history_id_cas_incremental_succeeds_when_unchanged(tmp_vault_col):
    """Incremental con expected_hid que matchea: UPDATE gana. Cursor
    avanza al new_hid."""
    import sqlite3
    from scripts import ingest_gmail as ig_mod

    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        ig_mod._ensure_state_table(conn)
        ig_mod._save_history_id(conn, "me@x.com", "500")
        conn.commit()
        # Worker leyó "500" al inicio, ahora intenta avanzar a "600".
        won = ig_mod._save_history_id_cas(
            conn, "me@x.com", "600", expected_history_id="500",
        )
        conn.commit()
        assert won is True
        assert ig_mod._load_history_id(conn, "me@x.com") == "600"
    finally:
        conn.close()


def test_save_history_id_cas_incremental_loses_race(tmp_vault_col):
    """Race incremental: ambos workers leyeron stored_hid='500'; el
    primero avanzó a '600' antes del CAS del segundo. El segundo CAS
    falla (rowcount=0) porque su WHERE history_id='500' ya no matchea.
    El cursor queda en el valor del primer worker — NO se sobrescribe.

    Este es exactamente el bug del audit: pre-fix, INSERT OR REPLACE
    siempre escribía y ambos workers procesaban el mismo rango → corpus
    con duplicados de threads. Post-fix, el segundo worker detecta el
    conflicto y NO retry."""
    import sqlite3
    from scripts import ingest_gmail as ig_mod

    state_path = rag.DB_PATH / "ragvec.db"
    conn = sqlite3.connect(str(state_path))
    try:
        ig_mod._ensure_state_table(conn)
        ig_mod._save_history_id(conn, "me@x.com", "500")
        conn.commit()

        # Worker 1: leyó "500", procesó, avanza a "600". Gana.
        a = ig_mod._save_history_id_cas(
            conn, "me@x.com", "600", expected_history_id="500",
        )
        # Worker 2 (concurrente): leyó "500" antes que worker 1
        # commiteara, procesó el mismo rango, intenta avanzar a "650".
        # Pierde porque su WHERE no matchea.
        b = ig_mod._save_history_id_cas(
            conn, "me@x.com", "650", expected_history_id="500",
        )
        conn.commit()
        assert a is True
        assert b is False
        # El cursor quedó en "600" (worker 1 ganó), no en "650"
        # (worker 2 perdió la carrera).
        assert ig_mod._load_history_id(conn, "me@x.com") == "600"
    finally:
        conn.close()
