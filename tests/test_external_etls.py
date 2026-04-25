import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import rag


@pytest.fixture
def tmp_vault(tmp_path):
    return tmp_path


def test_chrome_timestamp_roundtrip():
    now = time.time()
    chrome = rag._unix_to_chrome_ts(now)
    back = rag._chrome_to_unix_ts(chrome)
    assert abs(back - now) < 0.001


def test_atomic_write_skips_when_unchanged(tmp_path):
    target = tmp_path / "out.md"
    body = "hello"
    assert rag._atomic_write_if_changed(target, body) is True
    assert rag._atomic_write_if_changed(target, body) is False
    assert rag._atomic_write_if_changed(target, "hello!") is True


def _seed_chrome_db(path: Path, rows: list[tuple]):
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT, title TEXT, "
        "visit_count INTEGER, last_visit_time INTEGER)"
    )
    con.executemany(
        "INSERT INTO urls (id, url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


def test_read_chrome_visits_filters_window_and_skips_noise(tmp_path):
    db = tmp_path / "History"
    now = time.time()
    fresh = rag._unix_to_chrome_ts(now - 3600)        # 1h ago
    stale = rag._unix_to_chrome_ts(now - 72 * 3600)   # 72h ago — outside 48h window
    _seed_chrome_db(db, [
        (1, "https://example.com/article", "Article", 3, fresh),
        (2, "chrome://newtab", "New Tab", 1, fresh),
        (3, "https://www.google.com/search?q=foo", "foo - Google", 5, fresh),
        (4, "https://old.example.com", "Old", 1, stale),
        (5, "https://www.youtube.com/watch?v=abc123", "Video", 1, fresh),
    ])
    visits = rag._read_chrome_visits(db, hours=48)
    urls = {v["url"] for v in visits}
    assert "https://example.com/article" in urls
    assert "https://www.youtube.com/watch?v=abc123" in urls
    assert not any(u.startswith("chrome://") for u in urls)
    assert not any("/search?" in u for u in urls)
    assert "https://old.example.com" not in urls


def test_read_chrome_visits_returns_empty_when_db_missing(tmp_path):
    assert rag._read_chrome_visits(tmp_path / "nope.db") == []


def test_sync_chrome_history_writes_chrome_and_youtube(tmp_path):
    db = tmp_path / "History"
    now = time.time()
    fresh = rag._unix_to_chrome_ts(now - 3600)
    _seed_chrome_db(db, [
        (1, "https://example.com/post", "Post", 1, fresh),
        (2, "https://www.youtube.com/watch?v=xyz", "Cool video", 1, fresh),
        (3, "https://m.youtube.com/watch?v=mno&t=42", "Mobile vid", 1, fresh),
    ])
    with patch.object(rag, "_CHROME_HISTORY_PATH", db):
        stats = rag._sync_chrome_history(tmp_path)
    assert stats["ok"] is True
    assert stats["urls"] == 3
    assert stats["youtube_videos"] == 2
    today = time.strftime("%Y-%m-%d")
    chrome_md = (tmp_path / "03-Resources/Chrome" / f"{today}.md").read_text()
    yt_md = (tmp_path / "03-Resources/YouTube" / f"{today}.md").read_text()
    assert "Post" in chrome_md and "Cool video" in chrome_md
    assert "Cool video" in yt_md and "Mobile vid" in yt_md
    assert "example.com/post" not in yt_md  # only watch?v=
    assert "tags:\n- youtube" in yt_md
    assert "tags:\n- chrome-history" in chrome_md


def test_sync_chrome_history_silent_when_no_visits(tmp_path):
    with patch.object(rag, "_CHROME_HISTORY_PATH", tmp_path / "missing"):
        stats = rag._sync_chrome_history(tmp_path)
    assert stats["ok"] is False
    assert stats["reason"] == "no_visits_or_chrome_locked"


def test_sync_reminders_notes_with_mocked_pending(tmp_path):
    fake = [
        {"id": "1", "name": "Pagar luz", "due": "2026-04-25T10:00", "list": "Personal", "bucket": "upcoming"},
        {"id": "2", "name": "Llamar dentista", "due": "", "list": "Personal", "bucket": "undated"},
    ]
    with patch.object(rag, "_apple_enabled", return_value=True), \
         patch.object(rag, "_fetch_reminders_due", return_value=fake):
        stats = rag._sync_reminders_notes(tmp_path)
    assert stats["files_written"] == 1
    assert stats["pending"] == 2
    today = time.strftime("%Y-%m-%d")
    body = (tmp_path / "03-Resources/Reminders" / f"{today}.md").read_text()
    assert "Pagar luz" in body and "Llamar dentista" in body
    assert "## Próximos" in body and "## Sin fecha" in body
    assert "completed_count" not in body  # we dropped completed


def test_sync_reminders_notes_skip_on_unchanged(tmp_path):
    fake = [{"id": "1", "name": "x", "due": "", "list": "L", "bucket": "undated"}]
    with patch.object(rag, "_apple_enabled", return_value=True), \
         patch.object(rag, "_fetch_reminders_due", return_value=fake):
        rag._sync_reminders_notes(tmp_path)
        stats = rag._sync_reminders_notes(tmp_path)
    assert stats["files_written"] == 0  # second call hash-skips


def test_sync_calendar_notes_with_mocked_events(tmp_path):
    fake = [
        {"title": "Standup", "date_label": "today", "time_range": "09:00–09:30"},
        {"title": "Cumple Maria", "date_label": "29 jun 2026", "time_range": ""},
    ]
    with patch.object(rag, "_apple_enabled", return_value=True), \
         patch.object(rag, "_icalbuddy_path", return_value="/fake/icb"), \
         patch.object(rag, "_fetch_calendar_ahead", return_value=fake):
        stats = rag._sync_apple_calendar_notes(tmp_path)
    assert stats["files_written"] == 1
    assert stats["events"] == 2
    files = list((tmp_path / "03-Resources/Calendar").glob("*.md"))
    assert len(files) == 1
    body = files[0].read_text()
    assert "Standup" in body and "Cumple Maria" in body


def test_sync_calendar_silent_without_icalbuddy(tmp_path):
    with patch.object(rag, "_apple_enabled", return_value=True), \
         patch.object(rag, "_icalbuddy_path", return_value=None):
        stats = rag._sync_apple_calendar_notes(tmp_path)
    assert stats == {"ok": False, "reason": "icalbuddy_missing"}


def test_sync_reminders_silent_when_apple_disabled(tmp_path):
    with patch.object(rag, "_apple_enabled", return_value=False):
        stats = rag._sync_reminders_notes(tmp_path)
    assert stats == {"ok": False, "reason": "apple_disabled"}


def test_sync_gmail_silent_without_credentials(tmp_path):
    with patch.object(rag, "_load_google_credentials", return_value=None):
        stats = rag._sync_gmail_notes(tmp_path)
    assert stats == {"ok": False, "reason": "no_google_credentials"}


def test_sync_gdrive_silent_without_credentials(tmp_path):
    with patch.object(rag, "_load_google_credentials", return_value=None):
        stats = rag._sync_gdrive_notes(tmp_path)
    assert stats == {"ok": False, "reason": "no_google_credentials"}


def test_decode_gmail_body_prefers_text_plain():
    import base64
    plain = base64.urlsafe_b64encode(b"Hello plain").decode()
    html = base64.urlsafe_b64encode(b"<p>Hello html</p>").decode()
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": html}},
            {"mimeType": "text/plain", "body": {"data": plain}},
        ],
    }
    assert rag._decode_gmail_body(payload) == "Hello plain"


def test_decode_gmail_body_falls_back_to_html_stripped():
    import base64
    html = base64.urlsafe_b64encode(b"<p><b>Bold</b> body</p>").decode()
    payload = {
        "mimeType": "text/html",
        "body": {"data": html},
    }
    out = rag._decode_gmail_body(payload)
    assert "Bold" in out and "<" not in out


def test_decode_gmail_body_returns_empty_when_no_text():
    payload = {"mimeType": "image/png", "parts": []}
    assert rag._decode_gmail_body(payload) == ""


def _make_proc(returncode=0, stdout="", stderr=""):
    class _P:
        pass
    p = _P()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_sync_github_silent_when_gh_missing(tmp_path):
    with patch("rag.subprocess.run", side_effect=FileNotFoundError):
        stats = rag._sync_github_activity(tmp_path)
    assert stats == {"ok": False, "reason": "gh_unavailable_or_unauth"}


def test_sync_github_writes_activity(tmp_path):
    import json as _json
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = [{
        "type": "PushEvent", "repo": {"name": "fer/repo"}, "created_at": now,
        "payload": {"commits": [{"message": "fix bug"}]},
    }]
    open_prs = {"items": [{
        "title": "feat X", "html_url": "https://github.com/fer/repo/pull/1",
        "repository_url": "https://api.github.com/repos/fer/repo", "number": 1,
    }]}
    calls = []
    def fake_run(args, **kw):
        calls.append(tuple(args))
        if args[:3] == ["gh", "api", "user"]:
            return _make_proc(stdout="fer\n")
        if args[1] == "api" and "events" in args[2]:
            return _make_proc(stdout=_json.dumps(events))
        if args[1] == "api" and args[2] == "search/issues":
            return _make_proc(stdout=_json.dumps(open_prs))
        return _make_proc(returncode=1)
    with patch("rag.subprocess.run", side_effect=fake_run):
        stats = rag._sync_github_activity(tmp_path)
    assert stats["ok"] and stats["files_written"] == 1
    assert stats["events"] == 1 and stats["open_prs"] == 1
    today = datetime.now().strftime("%Y-%m-%d")
    body = (tmp_path / "03-Resources/GitHub" / f"{today}.md").read_text()
    assert "fer/repo" in body and "fix bug" in body
    assert "feat X" in body and "Open PRs" in body


def test_redact_secrets_catches_common_shapes():
    samples = [
        "sk-proj-abcdefghijklmnopqrstuvwxyz12345",
        "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
        'OPENAI_API_KEY="sk-1234567890abcdefghij"',
    ]
    for s in samples:
        out = rag._redact_secrets(s)
        assert "REDACTED" in out, f"missed: {s} → {out}"


def test_claude_extract_turn_skips_non_chat_records():
    assert rag._claude_extract_turn({"type": "summary"}) is None
    assert rag._claude_extract_turn({"type": "user", "message": {"content": ""}}) is None


def test_claude_extract_turn_string_content():
    rec = {"type": "user", "message": {"role": "user", "content": "hola"}, "timestamp": "2026-04-19T12:00:00Z"}
    role, ts, body = rag._claude_extract_turn(rec)
    assert role == "user" and "hola" in body


def test_claude_extract_turn_blocks_content_extracts_text_and_tool_use():
    rec = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": "voy a leer el archivo"},
            {"type": "tool_use", "name": "Read"},
        ]},
        "timestamp": "2026-04-19T12:01:00Z",
    }
    role, _ts, body = rag._claude_extract_turn(rec)
    assert role == "assistant"
    assert "voy a leer" in body and "[tool_use:Read]" in body


def test_claude_extract_turn_redacts_and_caps(monkeypatch):
    monkeypatch.setattr(rag, "_CLAUDE_TURN_BODY_CAP", 30)
    rec = {"type": "user", "message": {"content": "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa more text here"},
           "timestamp": "2026-04-19T12:00:00Z"}
    _r, _t, body = rag._claude_extract_turn(rec)
    assert "REDACTED" in body
    assert "truncado" in body or len(body) <= 60


def test_sync_claude_code_transcripts_writes_per_session(tmp_path, monkeypatch):
    fake_root = tmp_path / "claude-projects"
    proj = fake_root / "-Users-fer-repo"
    proj.mkdir(parents=True)
    session = proj / "abc123.jsonl"
    import json as _json
    session.write_text("\n".join([
        _json.dumps({"type": "user", "message": {"content": "hola"}, "timestamp": "2026-04-19T12:00:00Z"}),
        _json.dumps({"type": "assistant", "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "qué necesitás"}]},
                     "timestamp": "2026-04-19T12:00:05Z"}),
    ]), encoding="utf-8")
    monkeypatch.setattr(rag, "_CLAUDE_PROJECTS_ROOT", fake_root)
    stats = rag._sync_claude_code_transcripts(tmp_path)
    assert stats["files_written"] == 1
    out = (tmp_path / "03-Resources/Claude/-Users-fer-repo/abc123.md").read_text()
    assert "hola" in out and "qué necesitás" in out
    assert "session_id: abc123" in out


def test_sync_claude_code_silent_without_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_CLAUDE_PROJECTS_ROOT", tmp_path / "missing")
    stats = rag._sync_claude_code_transcripts(tmp_path)
    assert stats == {"ok": False, "reason": "no_claude_projects_dir"}


def test_sync_youtube_transcripts_skips_existing(tmp_path, monkeypatch):
    yt_dir = tmp_path / "03-Resources/YouTube"
    yt_dir.mkdir(parents=True)
    (yt_dir / "2026-04-19.md").write_text(
        "- `12:00` [Video](https://www.youtube.com/watch?v=abc123)\n",
        encoding="utf-8",
    )
    transcripts_dir = tmp_path / "03-Resources/YouTube/transcripts"
    transcripts_dir.mkdir(parents=True)
    (transcripts_dir / "abc123.md").write_text("---\n---\nexisting\n", encoding="utf-8")
    called = []
    def fake_fetch(vid):
        called.append(vid)
        return ("es", "should not run")
    monkeypatch.setattr(rag, "_fetch_yt_transcript_for_index", fake_fetch)
    stats = rag._sync_youtube_transcripts(tmp_path)
    assert stats["fetched_this_run"] == 0
    assert called == []


def test_sync_spotify_silent_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_SPOTIFY_TOKEN_PATH", tmp_path / "missing.json")
    stats = rag._sync_spotify_notes(tmp_path)
    assert stats == {"ok": False, "reason": "no_spotify_token"}


def test_spotify_client_returns_none_when_creds_invalid(tmp_path, monkeypatch):
    bad = tmp_path / "spotify_client.json"
    bad.write_text('{"client_id": ""}', encoding="utf-8")
    monkeypatch.setattr(rag, "_SPOTIFY_CREDS_PATH", bad)
    assert rag._spotify_client(allow_interactive=False) is None


def test_sync_spotify_writes_recently_played(tmp_path, monkeypatch):
    fake_recent = {"items": [{
        "played_at": "2026-04-19T10:00:00.000Z",
        "track": {
            "name": "Mi tema",
            "artists": [{"name": "Banda X"}],
            "album": {"name": "Disco Y"},
            "external_urls": {"spotify": "https://open.spotify.com/track/abc"},
        },
    }]}

    class _FakeSp:
        def current_user_recently_played(self, limit=50):
            return fake_recent
        def current_user_top_tracks(self, limit, time_range):
            return {"items": []}
        def current_user_top_artists(self, limit, time_range):
            return {"items": []}

    fake_token = tmp_path / "spotify_token.json"
    fake_token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rag, "_SPOTIFY_TOKEN_PATH", fake_token)
    monkeypatch.setattr(rag, "_spotify_client", lambda allow_interactive=True: _FakeSp())
    stats = rag._sync_spotify_notes(tmp_path)
    assert stats["ok"] and stats["recently_played"] == 1
    today = time.strftime("%Y-%m-%d")
    body = (tmp_path / "03-Resources/Spotify" / f"{today}.md").read_text()
    assert "Mi tema" in body and "Banda X" in body and "Disco Y" in body


def test_sync_spotify_top_refresh_skipped_when_recent(tmp_path, monkeypatch):
    """_top.md exists with recent mtime → should not refresh."""
    sp_dir = tmp_path / "03-Resources/Spotify"
    sp_dir.mkdir(parents=True)
    top = sp_dir / "_top.md"
    top.write_text("---\n---\nstale top\n", encoding="utf-8")
    # Touch mtime to "now" so TTL window does NOT expire.
    import os as _os
    _os.utime(top, None)

    top_calls = []
    class _FakeSp:
        def current_user_recently_played(self, limit=50):
            return {"items": []}
        def current_user_top_tracks(self, limit, time_range):
            top_calls.append("tracks")
            return {"items": []}
        def current_user_top_artists(self, limit, time_range):
            top_calls.append("artists")
            return {"items": []}

    fake_token = tmp_path / "spotify_token.json"
    fake_token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rag, "_SPOTIFY_TOKEN_PATH", fake_token)
    monkeypatch.setattr(rag, "_spotify_client", lambda allow_interactive=True: _FakeSp())
    rag._sync_spotify_notes(tmp_path)
    assert top_calls == []  # mtime fresh → no refresh


def test_sync_youtube_transcripts_fetches_new(tmp_path, monkeypatch):
    yt_dir = tmp_path / "03-Resources/YouTube"
    yt_dir.mkdir(parents=True)
    (yt_dir / "2026-04-19.md").write_text(
        "- `12:00` [Cool video](https://www.youtube.com/watch?v=newvid)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "_fetch_yt_transcript_for_index", lambda v: ("es", "transcript text"))
    stats = rag._sync_youtube_transcripts(tmp_path)
    assert stats["fetched_this_run"] == 1
    assert stats["files_written"] == 1
    body = (tmp_path / "03-Resources/YouTube/transcripts/newvid.md").read_text()
    assert "Cool video" in body and "transcript text" in body
    assert "language: es" in body


def test_run_cross_source_etls_invokes_chrome_bookmarks(tmp_path, monkeypatch):
    """`rag index` (via `_run_cross_source_etls`) debe sincronizar Chrome
    bookmarks junto con el resto de las ETLs (history, reminders, calendar,
    gmail, etc.). Antes del 2026-04-25, `sync_chrome_bookmarks` sólo se
    invocaba con el comando manual `rag bookmarks sync` y los marcadores
    indexados quedaban stale."""
    calls: list[dict] = []

    def fake_sync_chrome_bookmarks(profile=None):
        calls.append({"profile": profile})
        return {"profiles": 1, "total": 5, "per_profile": {"Default": 5}}

    # Bypass guard: en tests tmp_path no es vault canónico.
    monkeypatch.setattr(rag, "_is_cross_source_target", lambda _vp: True)
    # Mock todas las ETLs upstream a no-op para aislar el bloque nuevo.
    monkeypatch.setattr(rag, "_sync_moze_notes", lambda vp: {"ok": False, "reason": "no_csv"})
    monkeypatch.setattr(rag, "_sync_whatsapp_notes", lambda vp: {"ok": False, "reason": "script_missing"})
    monkeypatch.setattr(rag, "_sync_reminders_notes", lambda vp: {"ok": False, "reason": "apple_disabled"})
    monkeypatch.setattr(rag, "_sync_apple_calendar_notes", lambda vp: {"ok": False, "reason": "icalbuddy_missing"})
    monkeypatch.setattr(rag, "_sync_chrome_history", lambda vp: {"ok": False, "reason": "no_visits_or_chrome_locked"})
    monkeypatch.setattr(rag, "_sync_gmail_notes", lambda vp: {"ok": False, "reason": "no_google_credentials"})
    monkeypatch.setattr(rag, "_sync_github_activity", lambda vp: {"ok": False, "reason": "gh_no_login"})
    monkeypatch.setattr(rag, "_sync_claude_code_transcripts", lambda vp: {"ok": False, "reason": "no_claude_projects_dir"})
    monkeypatch.setattr(rag, "_sync_youtube_transcripts", lambda vp: {"ok": False, "reason": "no_videos"})
    monkeypatch.setattr(rag, "_sync_spotify_notes", lambda vp: {"ok": False, "reason": "no_spotify_credentials"})
    monkeypatch.setattr(rag, "sync_chrome_bookmarks", fake_sync_chrome_bookmarks)

    rag._run_cross_source_etls(tmp_path)

    assert len(calls) == 1, "sync_chrome_bookmarks debe invocarse exactamente una vez"
    assert calls[0]["profile"] is None, "se sincronizan TODOS los profiles, no uno específico"


def test_run_cross_source_etls_chrome_bookmarks_silent_when_no_chrome(tmp_path, monkeypatch):
    """Cuando Chrome no está instalado / no hay profiles, el branch debe
    printear 'skip (no Chrome)' sin lanzar excepción."""
    monkeypatch.setattr(rag, "_is_cross_source_target", lambda _vp: True)
    for fn_name in (
        "_sync_moze_notes", "_sync_whatsapp_notes", "_sync_reminders_notes",
        "_sync_apple_calendar_notes", "_sync_chrome_history", "_sync_gmail_notes",
        "_sync_github_activity", "_sync_claude_code_transcripts",
        "_sync_youtube_transcripts", "_sync_spotify_notes",
    ):
        monkeypatch.setattr(rag, fn_name, lambda vp: {"ok": False, "reason": "no_data"})
    monkeypatch.setattr(rag, "chrome_bookmark_files", lambda root=None: [])

    # No raise — la función soporta gracefully el caso "no Chrome".
    rag._run_cross_source_etls(tmp_path)


def test_decode_gmail_body_strips_style_and_script_blocks():
    import base64
    raw = (
        b"<style>body{color:red;} .x{margin:0}</style>"
        b"<p>Hola Mundo</p>"
        b"<script>alert('x')</script>"
    )
    html = base64.urlsafe_b64encode(raw).decode()
    payload = {"mimeType": "text/html", "body": {"data": html}}
    out = rag._decode_gmail_body(payload)
    assert "Hola Mundo" in out
    assert "color:red" not in out
    assert "alert" not in out
