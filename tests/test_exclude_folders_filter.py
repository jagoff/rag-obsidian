"""Tests para el query param `exclude_folders` en los endpoints
`/api/notes/{related,contradictions,wikilink-suggestions}`.

Aplica a panels del plugin Obsidian: el user configura folders a
excluir en settings (ej. "04-Archive,00-Inbox") y cada call los pasa
como CSV. El filter es server-side para no desperdiciar LLM calls
de contradictions cuando los items se descartarían client-side.

Cubre:
  - `_parse_exclude_folders` parsea CSV correctamente (whitespace,
    slashes, valores vacíos).
  - `_is_in_excluded_folder` matchea por prefijo + separador
    (no false positives con folders vecinos como "04-Archive2").
  - 3 endpoints aplican el filter: related, contradictions, wikilinks.
  - Empty exclude_folders == sin filter (backward compat).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import rag
import web.server as _server


class _FakeCol:
    def __init__(self, count: int = 5):
        self._count = count
        self.id = "test-uuid"

    def count(self) -> int:
        return self._count


@pytest.fixture
def http_client():
    return TestClient(_server.app)


# ── Helpers ───────────────────────────────────────────────────────────────


def test_parse_exclude_folders_empty_returns_empty_list():
    assert _server._parse_exclude_folders(None) == []
    assert _server._parse_exclude_folders("") == []
    assert _server._parse_exclude_folders("   ") == []


def test_parse_exclude_folders_normalizes_with_trailing_slash():
    """Cada folder se normaliza con '/' al final para evitar prefix
    collisions (04-Archive vs 04-Archive2)."""
    assert _server._parse_exclude_folders("04-Archive") == ["04-Archive/"]
    # Trim leading + trailing slashes.
    assert _server._parse_exclude_folders("/04-Archive/") == ["04-Archive/"]


def test_parse_exclude_folders_csv():
    """CSV con whitespace + valores vacíos."""
    result = _server._parse_exclude_folders("04-Archive, 00-Inbox ,, 02-Areas/Old")
    assert result == ["04-Archive/", "00-Inbox/", "02-Areas/Old/"]


def test_is_in_excluded_folder_matches_prefix():
    excluded = ["04-Archive/"]
    assert _server._is_in_excluded_folder("04-Archive/X.md", excluded) is True
    assert _server._is_in_excluded_folder("04-Archive/Sub/Y.md", excluded) is True


def test_is_in_excluded_folder_does_not_match_neighbor_folder():
    """Crítico: '04-Archive2/X.md' NO debe matchear con '04-Archive/'.
    Sin el separador esto sería false positive."""
    excluded = ["04-Archive/"]
    assert _server._is_in_excluded_folder("04-Archive2/X.md", excluded) is False
    assert _server._is_in_excluded_folder("04-Archive-Old/X.md", excluded) is False


def test_is_in_excluded_folder_empty_excluded_returns_false():
    """Sin filter activo, ninguna path se considera excluida."""
    assert _server._is_in_excluded_folder("04-Archive/X.md", []) is False


def test_is_in_excluded_folder_handles_leading_slash():
    """`/04-Archive/X.md` (con slash inicial) debe normalizarse y matchear."""
    assert _server._is_in_excluded_folder(
        "/04-Archive/X.md", ["04-Archive/"]
    ) is True


# ── Endpoint /api/notes/related ───────────────────────────────────────────


def _patch_corpus_for_related(monkeypatch, metas):
    fake_col = _FakeCol(count=len(metas))
    fake_corpus = {
        "metas": metas,
        "outlinks": {},
        "title_to_paths": {},
        "backlinks": {},
        "tags": set(),
    }
    monkeypatch.setattr(_server, "get_db", lambda: fake_col)
    monkeypatch.setattr(rag, "get_db", lambda: fake_col)
    monkeypatch.setattr(_server, "_load_corpus", lambda col: fake_corpus)
    monkeypatch.setattr(rag, "_load_corpus", lambda col: fake_corpus)


def test_related_filters_excluded_folders(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    metas = [
        # Source con tags compartidos.
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        # Vecino activo — debe aparecer.
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud,equilibrio"},
        # Vecino archivado — debe filtrarse.
        {"file": "04-Archive/Old.md", "note": "Old", "folder": "04-Archive",
         "tags": "coaching,salud"},
    ]
    _patch_corpus_for_related(monkeypatch, metas)
    resp = http_client.get("/api/notes/related", params={
        "path": "02-Areas/A.md",
        "limit": 10,
        "exclude_folders": "04-Archive",
    })
    assert resp.status_code == 200
    paths = [it["path"] for it in resp.json()["items"]]
    assert "02-Areas/B.md" in paths
    assert "04-Archive/Old.md" not in paths


def test_related_no_filter_when_param_missing(http_client, monkeypatch, tmp_path):
    """Sin `exclude_folders` (o vacío), todos los items se devuelven —
    backward compat: clientes existentes no rompen."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "04-Archive/Old.md", "note": "Old", "folder": "04-Archive",
         "tags": "coaching,salud"},
    ]
    _patch_corpus_for_related(monkeypatch, metas)
    resp = http_client.get("/api/notes/related", params={
        "path": "02-Areas/A.md", "limit": 10,
    })
    paths = [it["path"] for it in resp.json()["items"]]
    assert "04-Archive/Old.md" in paths


def test_related_filters_multiple_folders_csv(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "x,y"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "x,y"},
        {"file": "04-Archive/Old.md", "note": "Old", "folder": "04-Archive",
         "tags": "x,y"},
        {"file": "00-Inbox/Quick.md", "note": "Quick", "folder": "00-Inbox",
         "tags": "x,y"},
    ]
    _patch_corpus_for_related(monkeypatch, metas)
    resp = http_client.get("/api/notes/related", params={
        "path": "02-Areas/A.md",
        "limit": 10,
        "exclude_folders": "04-Archive,00-Inbox",
    })
    paths = [it["path"] for it in resp.json()["items"]]
    assert "02-Areas/B.md" in paths
    assert "04-Archive/Old.md" not in paths
    assert "00-Inbox/Quick.md" not in paths


# ── Endpoint /api/notes/contradictions ────────────────────────────────────


def test_contradictions_filters_excluded_folders(http_client, monkeypatch, tmp_path):
    """Si una contradicción cae en un folder excluido, NO se devuelve.
    Crítico: NO se desperdicia el LLM call (ya corrió) pero al menos
    el panel queda limpio."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    fake_col = _FakeCol(count=10)
    monkeypatch.setattr(_server, "get_db", lambda: fake_col)
    note_path = tmp_path / "02-Areas"
    note_path.mkdir()
    (note_path / "A.md").write_text("Esta nota dice X. " * 30)  # >200 chars

    def fake_find(col, body, exclude_paths, k):
        return [
            {
                "path": "02-Areas/B.md", "note": "B",
                "snippet": "snippet B", "why": "why B",
            },
            {
                "path": "04-Archive/Old.md", "note": "Old",
                "snippet": "snippet Old", "why": "why Old",
            },
        ]

    monkeypatch.setattr(_server, "find_contradictions_for_note", fake_find)
    resp = http_client.get("/api/notes/contradictions", params={
        "path": "02-Areas/A.md",
        "exclude_folders": "04-Archive",
    })
    body = resp.json()
    paths = [it["path"] for it in body["items"]]
    assert "02-Areas/B.md" in paths
    assert "04-Archive/Old.md" not in paths


# ── Endpoint /api/notes/wikilink-suggestions ──────────────────────────────


def test_wikilinks_filters_excluded_folders(http_client, monkeypatch, tmp_path):
    """Wikilinks tiene `target` (no `path`). El filter aplica a target."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    fake_col = _FakeCol(count=10)
    monkeypatch.setattr(_server, "get_db", lambda: fake_col)
    (tmp_path / "Source.md").write_text("texto con menciones a varias notas")

    def fake_find(col, note_path, **kwargs):
        return [
            {
                "title": "B", "target": "02-Areas/B.md", "line": 1,
                "char_offset": 5, "context": "ctx",
            },
            {
                "title": "Old", "target": "04-Archive/Old.md", "line": 2,
                "char_offset": 50, "context": "ctx old",
            },
        ]

    monkeypatch.setattr(_server, "find_wikilink_suggestions", fake_find)
    resp = http_client.get("/api/notes/wikilink-suggestions", params={
        "path": "Source.md",
        "exclude_folders": "04-Archive",
    })
    body = resp.json()
    targets = [it["target"] for it in body["items"]]
    assert "02-Areas/B.md" in targets
    assert "04-Archive/Old.md" not in targets


def test_wikilinks_filter_preserves_limit(http_client, monkeypatch, tmp_path):
    """Si tras filtrar quedan menos del limit, está bien — no llenamos
    con basura. El plugin entiende lista corta como "no hay más"."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    fake_col = _FakeCol(count=10)
    monkeypatch.setattr(_server, "get_db", lambda: fake_col)
    (tmp_path / "Source.md").write_text("texto")

    def fake_find(col, note_path, **kwargs):
        # Todos en folder excluido.
        return [
            {
                "title": f"X{i}", "target": f"04-Archive/X{i}.md", "line": i,
                "char_offset": i * 10, "context": "ctx",
            }
            for i in range(5)
        ]

    monkeypatch.setattr(_server, "find_wikilink_suggestions", fake_find)
    resp = http_client.get("/api/notes/wikilink-suggestions", params={
        "path": "Source.md",
        "limit": 30,
        "exclude_folders": "04-Archive",
    })
    body = resp.json()
    assert body["items"] == []  # Todos filtrados.
