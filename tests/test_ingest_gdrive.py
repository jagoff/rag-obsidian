"""Tests for the Google Drive monthly backfill ingester
(`scripts/ingest_gdrive.py`).

Covers:
  1. `_slugify` — filename safety + unicode.
  2. `_target_path` — date + slug + id8.
  3. `_render_note` — YAML frontmatter correctness.
  4. `_export_body` — decode + cap + error paths.
  5. `_list_recent` — pagination via nextPageToken + max_files cap.
  6. `run` — end-to-end with fake service + idempotency (re-run is no-op).

No network, no real Drive creds. The service is monkey-patched via a
fake shape identical to what googleapiclient returns.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import ingest_gdrive


# ── 1. _slugify ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Alex - Cuotas Macbook", "alex-cuotas-macbook"),
        ("  Leading and trailing spaces  ", "leading-and-trailing-spaces"),
        ("Caracteres/ilegales:*?<>|", "caracteresilegales"),
        ("", "untitled"),
        ("___", "untitled"),
        ("Ñandúbay", "ñandúbay"),
        ("multiple   spaces\t\ttabs", "multiple-spaces-tabs"),
    ],
)
def test_slugify(name: str, expected: str):
    assert ingest_gdrive._slugify(name) == expected


def test_slugify_caps_length():
    long = "a" * 200
    out = ingest_gdrive._slugify(long, max_len=60)
    assert len(out) == 60


# ── 2. _target_path ───────────────────────────────────────────────────────


def test_target_path_uses_modified_date_prefix(tmp_path: Path):
    meta = {
        "id": "1AbCdEfGhIjKlMnOpQrStUvWx",
        "name": "Plan de marketing 2026",
        "modifiedTime": "2026-04-20T13:22:10.000Z",
        "mimeType": "application/vnd.google-apps.document",
    }
    path = ingest_gdrive._target_path(tmp_path, meta)
    assert path.parent == tmp_path / ingest_gdrive.ARCHIVE_SUBPATH
    assert path.name == "2026-04-20__plan-de-marketing-2026__1AbCdEfG.md"


def test_target_path_missing_modified(tmp_path: Path):
    """Drive API occasionally omits modifiedTime on edge cases — we fall
    back to today's date so we still produce a stable path."""
    meta = {
        "id": "XYZ98765ABCDEF",
        "name": "sin modificado",
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }
    path = ingest_gdrive._target_path(tmp_path, meta)
    assert path.suffix == ".md"
    assert "__sin-modificado__XYZ98765.md" in path.name


# ── 3. _render_note ───────────────────────────────────────────────────────


def test_render_note_frontmatter_and_body():
    meta = {
        "id": "docId123",
        "name": "Notas de reunión",
        "mimeType": "application/vnd.google-apps.document",
        "modifiedTime": "2026-04-20T13:22:10.000Z",
        "createdTime": "2026-04-01T09:00:00.000Z",
        "owners": [{"displayName": "Fer F.", "emailAddress": "fer@example.com"}],
        "webViewLink": "https://docs.google.com/document/d/docId123/edit",
    }
    body = "Primer punto\nSegundo punto"
    note = ingest_gdrive._render_note(meta, body, "03-Resources/GoogleDrive/archive/foo.md")

    # frontmatter sanity
    assert note.startswith("---\n")
    assert "source: google-drive" in note
    assert "drive_id: docId123" in note
    assert "mime: doc" in note
    assert "modified: 2026-04-20T13:22:10.000Z" in note
    assert "owner: Fer F." in note
    assert "owner_email: fer@example.com" in note
    assert "link: https://docs.google.com/document/d/docId123/edit" in note
    # tags
    assert "- google-drive" in note
    assert "- gdrive-archive" in note
    assert "- gdrive/doc" in note
    # body appended
    assert "# Notas de reunión" in note
    assert "Primer punto" in note
    assert "Segundo punto" in note


def test_render_note_empty_body_gets_placeholder():
    meta = {
        "id": "x",
        "name": "sin contenido",
        "mimeType": "application/vnd.google-apps.presentation",
        "modifiedTime": "",
        "createdTime": "",
        "owners": [],
        "webViewLink": "",
    }
    note = ingest_gdrive._render_note(meta, "", "foo.md")
    assert "_(sin contenido exportado)_" in note
    assert "mime: slides" in note
    assert "owner: ?" in note


# ── 4. _export_body ───────────────────────────────────────────────────────


class _FakeExportResponse:
    def __init__(self, body):
        self._body = body

    def execute(self):
        return self._body


class _FakeFiles:
    def __init__(self, listing, bodies):
        self._listing = listing
        self._bodies = bodies
        self.list_calls = []
        self.export_calls = []
        self._next_page_tokens: list[str | None] = []

    def set_pagination(self, tokens: list[str | None]):
        self._next_page_tokens = tokens

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        # For pagination tests: pop next token per call; split listing in halves.
        if self._next_page_tokens:
            token = self._next_page_tokens.pop(0)
            # split listing deterministically by call index
            idx = len(self.list_calls) - 1
            half = len(self._listing) // 2
            if idx == 0:
                payload = {"files": self._listing[:half], "nextPageToken": token} if token else {"files": self._listing[:half]}
            else:
                payload = {"files": self._listing[half:], "nextPageToken": token} if token else {"files": self._listing[half:]}
            return _FakeExportResponse(payload)
        return _FakeExportResponse({"files": list(self._listing)})

    def export(self, fileId: str, mimeType: str):
        self.export_calls.append((fileId, mimeType))
        body = self._bodies.get(fileId, "")
        if body is Exception:
            return _ExportRaises()
        return _FakeExportResponse(body)


class _ExportRaises:
    def execute(self):
        raise RuntimeError("503 backend")


class _FakeService:
    def __init__(self, listing, bodies):
        self._files = _FakeFiles(listing, bodies)

    def files(self):
        return self._files


def test_export_body_happy_path():
    svc = _FakeService([], {"id1": b"hola mundo"})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert body == "hola mundo"
    assert err == ""


def test_export_body_caps_at_body_cap():
    svc = _FakeService([], {"id1": "X" * 50_000})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert err == ""
    # cap + truncation marker
    assert len(body) <= 1000 + 100
    assert body.startswith("X" * 1000)
    assert "<!-- truncated at 1000 chars -->" in body


def test_export_body_unknown_mime_returns_err():
    svc = _FakeService([], {})
    body, err = ingest_gdrive._export_body(svc, "id1", "application/pdf", 1000)
    assert body == ""
    assert err.startswith("unsupported_mime:")


def test_export_body_api_error_returns_err():
    svc = _FakeService([], {"id1": Exception})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert body == ""
    assert err.startswith("export_failed:")


# ── 5. _list_recent pagination ────────────────────────────────────────────


def test_list_recent_follows_nextpagetoken():
    listing = [
        {"id": f"f{i}", "name": f"doc{i}", "mimeType": "application/vnd.google-apps.document"}
        for i in range(10)
    ]
    svc = _FakeService(listing, {})
    svc.files().set_pagination(["tok_page2", None])  # page 1 has token, page 2 doesn't
    out = ingest_gdrive._list_recent(
        svc, days=30,
        mime_types=["application/vnd.google-apps.document"],
        max_files=100,
    )
    assert len(out) == 10
    assert len(svc.files().list_calls) == 2
    # page 2 call must carry the token
    assert svc.files().list_calls[1]["pageToken"] == "tok_page2"


def test_list_recent_respects_max_files_cap():
    listing = [
        {"id": f"f{i}", "name": f"doc{i}", "mimeType": "application/vnd.google-apps.document"}
        for i in range(200)
    ]
    svc = _FakeService(listing, {})
    out = ingest_gdrive._list_recent(
        svc, days=30,
        mime_types=["application/vnd.google-apps.document"],
        max_files=5,
    )
    assert len(out) == 5


# ── 6. run() end-to-end ───────────────────────────────────────────────────


def _make_meta(id_, name, mime="application/vnd.google-apps.document", modified="2026-04-20T13:22:10.000Z"):
    return {
        "id": id_,
        "name": name,
        "mimeType": mime,
        "modifiedTime": modified,
        "createdTime": "2026-04-01T00:00:00.000Z",
        "owners": [{"displayName": "Fer F.", "emailAddress": "fer@x.com"}],
        "webViewLink": f"https://docs.google.com/document/d/{id_}/edit",
    }


def test_run_writes_per_doc_markdowns(tmp_path: Path):
    listing = [
        _make_meta("docA", "Plan marketing"),
        _make_meta("docB", "Informe Q1", mime="application/vnd.google-apps.spreadsheet"),
    ]
    bodies = {"docA": "contenido plan", "docB": "cabecera\n1,2,3"}
    svc = _FakeService(listing, bodies)

    summary = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        vault_root=tmp_path, svc=svc,
    )

    assert summary["ok"]
    assert summary["files_seen"] == 2
    assert summary["files_written"] == 2
    assert summary["files_skipped"] == 0
    assert summary["files_failed"] == 0

    archive_dir = tmp_path / ingest_gdrive.ARCHIVE_SUBPATH
    md_files = sorted(archive_dir.glob("*.md"))
    assert len(md_files) == 2
    texts = [p.read_text(encoding="utf-8") for p in md_files]
    assert any("contenido plan" in t for t in texts)
    assert any("cabecera" in t for t in texts)
    assert any("source: google-drive" in t for t in texts)


def test_run_is_idempotent(tmp_path: Path):
    listing = [_make_meta("docA", "Plan marketing")]
    bodies = {"docA": "contenido plan"}
    svc = _FakeService(listing, bodies)

    first = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        vault_root=tmp_path, svc=svc,
    )
    assert first["files_written"] == 1

    second = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        vault_root=tmp_path, svc=_FakeService(listing, bodies),
    )
    # Content unchanged → no re-write
    assert second["files_written"] == 0
    assert second["files_skipped"] == 1


def test_run_dry_run_does_not_export_or_write(tmp_path: Path):
    listing = [_make_meta("docA", "Plan marketing")]
    svc = _FakeService(listing, {"docA": "x"})

    summary = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        vault_root=tmp_path, svc=svc, dry_run=True,
    )
    assert summary["ok"]
    assert summary["files_seen"] == 1
    assert summary["files_written"] == 0
    assert summary["files_skipped"] == 1
    # export() must NOT have been called
    assert svc.files().export_calls == []
    # no file should have been written
    archive_dir = tmp_path / ingest_gdrive.ARCHIVE_SUBPATH
    assert not archive_dir.exists() or not list(archive_dir.glob("*.md"))


def test_run_tolerates_export_failures_and_still_writes_stub(tmp_path: Path):
    """If exporting a specific doc fails, we still emit a stub markdown
    (frontmatter + placeholder body) so the file is visible in Obsidian
    and we can retry the export on a later run. The run is NOT marked
    failed globally — we count per-file."""
    listing = [
        _make_meta("docGOOD", "Plan marketing"),
        _make_meta("docBAD", "Error doc"),
    ]
    bodies = {"docGOOD": "ok", "docBAD": Exception}
    svc = _FakeService(listing, bodies)

    summary = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        vault_root=tmp_path, svc=svc,
    )
    assert summary["ok"]
    assert summary["files_failed"] == 1
    # both files written (stub for the failure)
    archive_dir = tmp_path / ingest_gdrive.ARCHIVE_SUBPATH
    assert len(list(archive_dir.glob("*.md"))) == 2
    bad_note = next(p for p in archive_dir.glob("*.md") if "docBAD"[:8] in p.name)
    assert "_(sin contenido exportado)_" in bad_note.read_text(encoding="utf-8")
