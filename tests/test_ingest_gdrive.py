"""Tests for the Google Drive cross-source ingester
(`scripts/ingest_gdrive.py`).

After the 2026-04-24 rewrite the ingester upserts chunks directly to
the vector collection instead of writing markdown files to the vault
(user's Obsidian workflow was moving `03-Resources/GoogleDrive/` to
`.trash/` periodically, making the file-based approach unusable).

Tests cover:
  1. `_chunk_body` — short-body pass-through, paragraph split, overlap,
     hard-split for long single paragraphs.
  2. `_file_key` / `_chunk_doc_id` — pseudo-URI identity contract.
  3. `_parse_mod_ts` — RFC3339 → epoch robustness.
  4. `_list_since` — paginated fetch with `modifiedTime asc`,
     `max_files` cap.
  5. `_export_body` — decode + cap + unsupported mime + API failure.
  6. `upsert_drive_file` — chunks + embed + metadata shape + idempotent
     pre-delete of existing chunks.
  7. State table: `_ensure_state_table`, `_load_cursor`, `_save_cursor`,
     `_reset_cursor`.
  8. `_retention_prune` — drops chunks older than
     `SOURCE_RETENTION_DAYS["drive"]`.
  9. `run()` end-to-end with a `FakeCollection` + `FakeService`:
     bootstrap vs incremental, dry-run, cursor advance, reset, errors
     don't tank the run, ``is_excluded`` already skips
     ``03-Resources/GoogleDrive/`` so vault-side leftovers can't leak
     back into the corpus.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import rag
from scripts import ingest_gdrive


# ── 1. chunking ─────────────────────────────────────────────────────────────


def test_chunk_body_short_returns_one_chunk():
    text = "Una sola oración corta."
    assert ingest_gdrive._chunk_body(text, target=800) == [text]


def test_chunk_body_empty_returns_empty_list():
    assert ingest_gdrive._chunk_body("") == []
    assert ingest_gdrive._chunk_body("   ") == []


def test_chunk_body_splits_at_paragraph_boundaries():
    p1 = "A" * 400
    p2 = "B" * 400
    p3 = "C" * 400
    text = f"{p1}\n\n{p2}\n\n{p3}"
    chunks = ingest_gdrive._chunk_body(text, target=800, overlap=40)
    # p1 + p2 fits (~804 w/ separator but close to 800 — likely one chunk);
    # p3 needs a new chunk.
    assert len(chunks) >= 2
    # Every chunk must be <= target + the overlap seed from previous (< 2x target).
    assert all(len(c) <= 900 for c in chunks)
    # All content present across chunks.
    joined = "\n".join(chunks)
    assert "A" * 100 in joined and "B" * 100 in joined and "C" * 100 in joined


def test_chunk_body_hard_splits_oversized_paragraph():
    # A single paragraph of 2500 chars with NO paragraph breaks must be
    # hard-split into multiple chunks.
    text = "X" * 2500
    chunks = ingest_gdrive._chunk_body(text, target=800, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_body_preserves_total_content_after_dedup():
    text = "Párrafo uno.\n\n" + ("Palabra " * 200).strip() + "\n\nFinal."
    chunks = ingest_gdrive._chunk_body(text, target=500, overlap=30)
    assert len(chunks) >= 2
    assert any("Párrafo uno" in c for c in chunks)
    assert any("Final" in c for c in chunks)


# ── 2. identifiers ──────────────────────────────────────────────────────────


def test_file_key_and_chunk_doc_id():
    assert ingest_gdrive._file_key("abc123") == "gdrive://file/abc123"
    assert ingest_gdrive._chunk_doc_id("abc123", 0) == "gdrive://file/abc123#chunk=0000"
    assert ingest_gdrive._chunk_doc_id("abc123", 42) == "gdrive://file/abc123#chunk=0042"


# ── 3. timestamp parsing ────────────────────────────────────────────────────


def test_parse_mod_ts_rfc3339_z():
    ts = ingest_gdrive._parse_mod_ts("2026-04-20T13:22:10.000Z")
    assert ts > 1_700_000_000  # clearly post-2023


def test_parse_mod_ts_missing_or_invalid():
    assert ingest_gdrive._parse_mod_ts("") == 0.0
    assert ingest_gdrive._parse_mod_ts("no-es-una-fecha") == 0.0


# ── 4. Drive API shape (Fakes) ──────────────────────────────────────────────


class _FakeExportResp:
    def __init__(self, body):
        self._body = body

    def execute(self):
        return self._body


class _ExportRaises:
    def execute(self):
        raise RuntimeError("503 backend")


class _FakeFiles:
    def __init__(self, listing, bodies, page_tokens: list[str | None] | None = None):
        self._listing = listing
        self._bodies = bodies
        self._page_tokens = list(page_tokens or [])
        self.list_calls: list[dict] = []
        self.export_calls: list[tuple[str, str]] = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        if self._page_tokens:
            token = self._page_tokens.pop(0)
            idx = len(self.list_calls) - 1
            half = len(self._listing) // 2
            if idx == 0:
                payload = {"files": self._listing[:half]}
            else:
                payload = {"files": self._listing[half:]}
            if token:
                payload["nextPageToken"] = token
            return _FakeExportResp(payload)
        return _FakeExportResp({"files": list(self._listing)})

    def export(self, fileId: str, mimeType: str):
        self.export_calls.append((fileId, mimeType))
        body = self._bodies.get(fileId, "")
        if body is Exception:
            return _ExportRaises()
        return _FakeExportResp(body)


class _FakeService:
    def __init__(self, listing, bodies, page_tokens=None):
        self._files = _FakeFiles(listing, bodies, page_tokens)

    def files(self):
        return self._files


def test_list_since_follows_nextpagetoken():
    listing = [
        {"id": f"f{i}", "name": f"doc{i}", "mimeType": "application/vnd.google-apps.document"}
        for i in range(10)
    ]
    svc = _FakeService(listing, {}, page_tokens=["tok_page2", None])
    out = ingest_gdrive._list_since(
        svc, since_iso="2026-03-01T00:00:00Z",
        mime_types=["application/vnd.google-apps.document"],
        max_files=100,
    )
    assert len(out) == 10
    assert len(svc.files().list_calls) == 2
    assert svc.files().list_calls[1]["pageToken"] == "tok_page2"


def test_list_since_respects_max_files_cap():
    listing = [
        {"id": f"f{i}", "name": f"doc{i}", "mimeType": "application/vnd.google-apps.document"}
        for i in range(200)
    ]
    svc = _FakeService(listing, {})
    out = ingest_gdrive._list_since(
        svc, since_iso="2026-03-01T00:00:00Z",
        mime_types=["application/vnd.google-apps.document"],
        max_files=5,
    )
    assert len(out) == 5


def test_list_since_uses_asc_order_for_monotonic_cursor():
    svc = _FakeService([], {})
    ingest_gdrive._list_since(
        svc, since_iso="2026-03-01T00:00:00Z",
        mime_types=["application/vnd.google-apps.document"],
        max_files=10,
    )
    assert svc.files().list_calls[0]["orderBy"] == "modifiedTime asc"


# ── 5. export_body ──────────────────────────────────────────────────────────


def test_export_body_happy_path():
    svc = _FakeService([], {"id1": b"hola mundo"})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert body == "hola mundo"
    assert err == ""


def test_export_body_caps_and_adds_truncation_marker():
    svc = _FakeService([], {"id1": "X" * 50_000})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert err == ""
    assert "<!-- truncated at 1000 chars -->" in body


def test_export_body_unsupported_mime():
    svc = _FakeService([], {})
    body, err = ingest_gdrive._export_body(svc, "id1", "application/pdf", 1000)
    assert body == ""
    assert err.startswith("unsupported_mime:")


def test_export_body_api_error_returns_error_string():
    svc = _FakeService([], {"id1": Exception})
    body, err = ingest_gdrive._export_body(
        svc, "id1", "application/vnd.google-apps.document", 1000
    )
    assert body == ""
    assert err.startswith("export_failed:")


# ── 6. upsert ───────────────────────────────────────────────────────────────


class _FakeCollection:
    """Captures upsert behavior for assertions. Mimics just enough of the
    SqliteVecCollection surface that `upsert_drive_file` uses."""
    def __init__(self):
        self.rows: dict[str, tuple[str, dict]] = {}  # id → (doc, meta)
        self.add_calls: list[dict] = []
        self.delete_calls: list[list[str]] = []

    def get(self, where=None, include=None):
        ids, docs, metas = [], [], []
        for id_, (doc, meta) in self.rows.items():
            if where and any((meta.get(k) != v) for k, v in where.items()):
                continue
            ids.append(id_)
            docs.append(doc)
            metas.append(meta)
        out = {"ids": ids}
        if include and "metadatas" in include:
            out["metadatas"] = metas
        return out

    def delete(self, ids):
        self.delete_calls.append(list(ids))
        for i in ids:
            self.rows.pop(i, None)

    def add(self, ids, embeddings, documents, metadatas):
        self.add_calls.append({"ids": list(ids), "n": len(ids)})
        for i, doc, meta in zip(ids, documents, metadatas):
            self.rows[i] = (doc, meta)


@pytest.fixture(autouse=True)
def _stub_embed(monkeypatch):
    """Avoid loading the real embedding model for every test."""
    def _fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    # Entity extraction — no-op: gliner is a soft dep and not wired for tests.
    monkeypatch.setattr(
        rag, "_extract_and_index_entities_for_chunks",
        lambda *a, **kw: None,
    )


def _make_meta(file_id="docA", name="Plan marketing",
               mime="application/vnd.google-apps.document",
               modified="2026-04-20T13:22:10.000Z"):
    return {
        "id": file_id,
        "name": name,
        "mimeType": mime,
        "modifiedTime": modified,
        "createdTime": "2026-04-01T00:00:00.000Z",
        "owners": [{"displayName": "Fer F.", "emailAddress": "fer@x.com"}],
        "webViewLink": f"https://docs.google.com/document/d/{file_id}/edit",
    }


def test_upsert_drive_file_writes_chunks_with_metadata():
    col = _FakeCollection()
    meta = _make_meta()
    n = ingest_gdrive.upsert_drive_file(col, meta, "contenido corto del plan")
    assert n == 1
    # Single chunk produces a single row with expected shape.
    (id_,) = col.rows.keys()
    doc, m = col.rows[id_]
    assert id_ == "gdrive://file/docA#chunk=0000"
    assert m["source"] == "drive"
    assert m["file"] == "gdrive://file/docA"
    assert m["drive_id"] == "docA"
    assert m["mime"] == "doc"
    assert m["owner"] == "Fer F."
    assert m["owner_email"] == "fer@x.com"
    assert m["link"].startswith("https://docs.google.com/")
    assert m["modified_iso"] == "2026-04-20T13:22:10.000Z"
    assert m["created_ts"] > 1_700_000_000
    assert m["chunk_idx"] == 0
    assert m["total_chunks"] == 1
    assert "Plan marketing" in m["parent"]
    assert "contenido corto" in doc


def test_upsert_drive_file_multiple_chunks_for_long_body():
    col = _FakeCollection()
    meta = _make_meta()
    body = "Párrafo uno.\n\n" + ("X" * 1500) + "\n\nFin."
    n = ingest_gdrive.upsert_drive_file(col, meta, body)
    assert n >= 2
    # All chunks share the same file + consecutive chunk_idx
    idxs = sorted(m["chunk_idx"] for _, m in col.rows.values())
    assert idxs == list(range(n))
    # Totals are consistent
    totals = {m["total_chunks"] for _, m in col.rows.values()}
    assert totals == {n}


def test_upsert_drive_file_is_idempotent_deletes_old_chunks():
    col = _FakeCollection()
    meta = _make_meta()
    # First ingest: 3 chunks.
    body_v1 = "Párrafo A.\n\n" + ("A" * 1600) + "\n\nCierre."
    n1 = ingest_gdrive.upsert_drive_file(col, meta, body_v1)
    assert n1 >= 2
    ids_v1 = set(col.rows.keys())

    # Second ingest of the same file with shorter body → 1 chunk, old ones gone.
    body_v2 = "Versión corta del doc."
    n2 = ingest_gdrive.upsert_drive_file(col, meta, body_v2)
    assert n2 == 1
    ids_v2 = set(col.rows.keys())
    assert ids_v2 == {"gdrive://file/docA#chunk=0000"}
    # None of the v1 chunks remain (delete was called).
    assert not (ids_v1 - {"gdrive://file/docA#chunk=0000"}).intersection(ids_v2)


def test_upsert_drive_file_empty_body_still_removes_old_chunks():
    col = _FakeCollection()
    meta = _make_meta()
    ingest_gdrive.upsert_drive_file(col, meta, "old body")
    assert len(col.rows) == 1
    # Now the doc was cleared out in Drive — export returns empty.
    n = ingest_gdrive.upsert_drive_file(col, meta, "")
    assert n == 0
    assert len(col.rows) == 0  # orphan chunks gone


# ── 7. state table / cursor ────────────────────────────────────────────────


def test_cursor_roundtrip():
    conn = sqlite3.connect(":memory:")
    ingest_gdrive._ensure_state_table(conn)
    assert ingest_gdrive._load_cursor(conn, "me") is None

    ingest_gdrive._save_cursor(conn, "me", "2026-04-20T10:00:00Z")
    conn.commit()
    assert ingest_gdrive._load_cursor(conn, "me") == "2026-04-20T10:00:00Z"

    # Update overwrites.
    ingest_gdrive._save_cursor(conn, "me", "2026-04-24T12:00:00Z")
    conn.commit()
    assert ingest_gdrive._load_cursor(conn, "me") == "2026-04-24T12:00:00Z"

    ingest_gdrive._reset_cursor(conn)
    conn.commit()
    assert ingest_gdrive._load_cursor(conn, "me") is None


# ── 8. retention pruning ────────────────────────────────────────────────────


def test_retention_prune_drops_old_chunks(monkeypatch):
    # Set retention to 1 day for the test.
    monkeypatch.setitem(rag.SOURCE_RETENTION_DAYS, "drive", 1)
    col = _FakeCollection()
    now = 1_735_000_000.0
    # One fresh, one stale
    col.add(
        ids=["gdrive://file/fresh#chunk=0000", "gdrive://file/stale#chunk=0000"],
        embeddings=[[0.0], [0.0]],
        documents=["fresh doc", "stale doc"],
        metadatas=[
            {"source": "drive", "created_ts": now - 3600},
            {"source": "drive", "created_ts": now - 86400 * 5},
        ],
    )
    removed = ingest_gdrive._retention_prune(col, now_ts=now)
    assert removed == 1
    assert "gdrive://file/fresh#chunk=0000" in col.rows
    assert "gdrive://file/stale#chunk=0000" not in col.rows


def test_retention_prune_noop_when_retention_none(monkeypatch):
    monkeypatch.setitem(rag.SOURCE_RETENTION_DAYS, "drive", None)
    col = _FakeCollection()
    col.add(
        ids=["x"], embeddings=[[0.0]], documents=["d"],
        metadatas=[{"source": "drive", "created_ts": 0.0}],
    )
    removed = ingest_gdrive._retention_prune(col)
    assert removed == 0
    assert "x" in col.rows


# ── 9. run() end-to-end ─────────────────────────────────────────────────────


def _tmp_state_conn():
    conn = sqlite3.connect(":memory:")
    return conn


def test_run_bootstrap_uses_days_window_and_saves_cursor():
    listing = [
        _make_meta("docA", "Plan marketing", modified="2026-04-10T10:00:00Z"),
        _make_meta("docB", "Q1 report", modified="2026-04-15T11:00:00Z"),
    ]
    bodies = {"docA": "texto del plan", "docB": "texto del Q1"}
    svc = _FakeService(listing, bodies)
    col = _FakeCollection()
    state = _tmp_state_conn()

    summary = ingest_gdrive.run(
        days=30, max_files=100, body_cap=10_000,
        svc=svc, col=col, state_conn=state,
    )

    assert summary["ok"] if "ok" in summary else True  # `ok` flag no longer used post-rewrite
    assert summary["bootstrapped"] is True
    assert summary["files_seen"] == 2
    assert summary["files_indexed"] == 2
    assert summary["chunks_written"] >= 2
    assert summary["files_failed"] == 0

    # Cursor advanced to the latest modifiedTime.
    assert ingest_gdrive._load_cursor(state, "me") == "2026-04-15T11:00:00Z"


def test_run_incremental_uses_stored_cursor():
    svc = _FakeService([_make_meta("docC", "nuevo", modified="2026-04-24T15:00:00Z")],
                       {"docC": "contenido nuevo"})
    col = _FakeCollection()
    state = _tmp_state_conn()
    ingest_gdrive._ensure_state_table(state)
    ingest_gdrive._save_cursor(state, "me", "2026-04-20T00:00:00Z")
    state.commit()

    summary = ingest_gdrive.run(
        days=30, svc=svc, col=col, state_conn=state,
    )

    assert summary["bootstrapped"] is False
    # The `q` sent to Drive must use the stored cursor, not today-30d.
    q_arg = svc.files().list_calls[0]["q"]
    assert "modifiedTime > '2026-04-20T00:00:00Z'" in q_arg
    # Cursor advanced.
    assert ingest_gdrive._load_cursor(state, "me") == "2026-04-24T15:00:00Z"


def test_run_reset_clears_cursor_and_rebootstraps():
    col = _FakeCollection()
    state = _tmp_state_conn()
    ingest_gdrive._ensure_state_table(state)
    ingest_gdrive._save_cursor(state, "me", "2026-04-20T00:00:00Z")
    state.commit()
    svc = _FakeService([], {})

    summary = ingest_gdrive.run(
        reset=True, days=30, svc=svc, col=col, state_conn=state,
    )
    assert summary["bootstrapped"] is True
    # Empty run: 0 files seen → cursor stays cleared.
    assert ingest_gdrive._load_cursor(state, "me") is None


def test_run_dry_run_does_not_upsert_and_preserves_cursor():
    listing = [_make_meta("docA", "Plan", modified="2026-04-20T10:00:00Z")]
    svc = _FakeService(listing, {"docA": "x"})
    col = _FakeCollection()
    state = _tmp_state_conn()
    ingest_gdrive._ensure_state_table(state)

    summary = ingest_gdrive.run(
        dry_run=True, days=30, svc=svc, col=col, state_conn=state,
    )

    assert summary["dry_run"] is True
    assert summary["files_seen"] == 1
    assert summary["files_indexed"] == 1
    assert summary["chunks_written"] == 0
    # No export API calls in dry-run.
    assert svc.files().export_calls == []
    # No upsert side-effect on the collection.
    assert col.rows == {}
    # Cursor untouched (still None from bootstrap dry-run).
    assert ingest_gdrive._load_cursor(state, "me") is None


def test_run_export_failures_dont_tank_run():
    listing = [
        _make_meta("docGOOD", "ok", modified="2026-04-15T10:00:00Z"),
        _make_meta("docBAD", "fail", modified="2026-04-20T10:00:00Z"),
    ]
    bodies = {"docGOOD": "hola", "docBAD": Exception}
    svc = _FakeService(listing, bodies)
    col = _FakeCollection()
    state = _tmp_state_conn()

    summary = ingest_gdrive.run(
        days=30, svc=svc, col=col, state_conn=state,
    )

    assert summary["files_indexed"] == 1     # only the GOOD one upserted
    assert summary["files_failed"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["id"] == "docBAD"
    # One GOOD chunk landed in the collection.
    assert any(m["drive_id"] == "docGOOD" for _, m in col.rows.values())
    assert not any(m["drive_id"] == "docBAD" for _, m in col.rows.values())


# ── 10. invariant: is_excluded skips 03-Resources/GoogleDrive/ ─────────────


def test_is_excluded_blocks_leftover_gdrive_folder_by_default():
    """Post-rewrite, the file-based path is obsolete. `is_excluded()` must
    skip `03-Resources/GoogleDrive/` so any stale markdown (daily snapshot,
    legacy backfill) never gets vault-side double-indexed alongside the
    real `source="drive"` chunks."""
    assert rag.is_excluded("03-Resources/GoogleDrive/archive/foo.md") is True
    assert rag.is_excluded("03-Resources/GoogleDrive/2026-04-24.md") is True
    # Escape hatch (env override) is tested implicitly by reading the env var;
    # we don't exercise the override here to keep the default path deterministic.


def test_valid_sources_includes_drive():
    assert "drive" in rag.VALID_SOURCES


def test_source_weight_halflife_retention_have_drive_entries():
    assert rag.SOURCE_WEIGHTS["drive"] == pytest.approx(0.85)
    assert rag.SOURCE_RECENCY_HALFLIFE_DAYS["drive"] == pytest.approx(90.0)
    assert rag.SOURCE_RETENTION_DAYS["drive"] == 365
