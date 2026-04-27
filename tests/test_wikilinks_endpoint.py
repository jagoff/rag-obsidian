"""Tests para `GET /api/notes/wikilink-suggestions` + `rag wikilinks suggest --json`.

Foco: validar que el wrap HTTP/CLI sobre `find_wikilink_suggestions` no
cambie el shape ni rompa el contract con el plugin Obsidian. La lógica
de matching (longest-first, ambiguous skip, code-fence skip, etc.) ya
tiene sus tests propios; acá solo cubrimos el endpoint shell.

Cubre:
  - Path validation (sufijo .md, traversal, vault missing).
  - reason=not_found / reason=empty_index.
  - Limit clamping.
  - Item shape (5 keys: title, target, line, char_offset, context).
  - CLI --json paralelo al HTTP.
  - Paridad CLI ↔ HTTP cuando se le pasa --note al CLI.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
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


def _patch_db(monkeypatch, count: int):
    fake = _FakeCol(count=count)
    monkeypatch.setattr(rag, "get_db", lambda: fake)
    monkeypatch.setattr(_server, "get_db", lambda: fake)


# ── HTTP endpoint ─────────────────────────────────────────────────────────


def test_http_rejects_non_md_path(http_client):
    resp = http_client.get("/api/notes/wikilink-suggestions", params={"path": "x"})
    assert resp.status_code == 400


def test_http_path_traversal_blocked(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "../../etc/passwd.md"},
    )
    assert resp.status_code == 400


def test_http_vault_missing_returns_503(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path / "nope")
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "A.md"},
    )
    assert resp.status_code == 503


def test_http_not_found_when_file_missing(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "ghost.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["reason"] == "not_found"


def test_http_empty_index(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=0)
    (tmp_path / "A.md").write_text("# A\n\ncontent")
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "A.md"},
    )
    assert resp.status_code == 200
    assert resp.json()["reason"] == "empty_index"


def test_http_happy_path_returns_items(http_client, monkeypatch, tmp_path):
    """Mockeamos `find_wikilink_suggestions` directamente — la lógica
    real ya se testea en otro lado, acá probamos el wrap."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "Source.md").write_text("# Source\n\ncontenido")

    expected = [
        {
            "title": "Boundaries",
            "target": "02-Areas/Coaching/Boundaries.md",
            "line": 5,
            "char_offset": 142,
            "context": "...habla de Boundaries en el work...",
        },
    ]

    def fake_find(col, note_path, **kwargs):
        return expected

    monkeypatch.setattr(_server, "find_wikilink_suggestions", fake_find)
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "Source.md", "limit": 30},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_path"] == "Source.md"
    assert len(body["items"]) == 1
    item = body["items"][0]
    # Shape exacto.
    assert set(item.keys()) == {"title", "target", "line", "char_offset", "context"}
    assert item["title"] == "Boundaries"
    assert item["char_offset"] == 142


def test_http_limit_clamped(http_client, monkeypatch, tmp_path):
    """limit=999 → clampa a 50 antes de pasar a find_wikilink_suggestions."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("# A\n\nx")

    captured = {"max_per_note": None}

    def fake_find(col, note_path, **kwargs):
        captured["max_per_note"] = kwargs.get("max_per_note")
        return []

    monkeypatch.setattr(_server, "find_wikilink_suggestions", fake_find)
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "A.md", "limit": 999},
    )
    assert resp.status_code == 200
    assert captured["max_per_note"] == 50


def test_http_extracts_real_match(http_client, monkeypatch, tmp_path):
    """Test E2E con find_wikilink_suggestions REAL (no mock).

    Crea una nota source que menciona "Otra nota" pero NO la linkea, y
    asegura que aparece en items.
    """
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)

    # Stub mínimo del corpus — find_wikilink_suggestions usa _load_corpus()
    # → c["title_to_paths"]. Devolvemos una mock simple.
    import types
    fake_corpus = {
        "title_to_paths": {
            "Otra nota": {"02-Areas/Otra nota.md"},
        },
    }

    def fake_load_corpus(col):
        return fake_corpus

    fake_col = _FakeCol(count=10)
    monkeypatch.setattr(_server, "get_db", lambda: fake_col)
    monkeypatch.setattr(rag, "get_db", lambda: fake_col)
    monkeypatch.setattr(rag, "_load_corpus", fake_load_corpus)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)

    (tmp_path / "Source.md").write_text(
        "# Source\n\nEsta nota habla de Otra nota sin linkearla.\n"
    )
    resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "Source.md", "limit": 30},
    )
    body = resp.json()
    titles = [it["title"] for it in body["items"]]
    assert "Otra nota" in titles


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_json_single_note(monkeypatch, tmp_path):
    """`rag wikilinks suggest --note X.md --json` → {items, source_path}."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("# A\n\nx")

    expected = [{
        "title": "X", "target": "X.md", "line": 1,
        "char_offset": 5, "context": "ctx",
    }]
    monkeypatch.setattr(rag, "find_wikilink_suggestions",
                         lambda col, p, **kw: expected)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["wikilinks", "suggest", "--note", "A.md", "--json"],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source_path"] == "A.md"
    assert body["items"] == expected


def test_cli_json_no_note_returns_by_note(monkeypatch, tmp_path):
    """Sin --note, --json devuelve {by_note, total_suggestions, ...}."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("# A\n")
    (tmp_path / "B.md").write_text("# B\n")

    monkeypatch.setattr(
        rag, "_load_corpus",
        lambda col: {
            "metas": [{"file": "A.md"}, {"file": "B.md"}],
            "title_to_paths": {},
        },
    )

    monkeypatch.setattr(
        rag, "find_wikilink_suggestions",
        lambda col, p, **kw: (
            [{"title": "X", "target": "X.md", "line": 1, "char_offset": 0, "context": ""}]
            if p == "A.md" else []
        ),
    )

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["wikilinks", "suggest", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert "by_note" in body
    assert body["total_suggestions"] == 1
    assert body["notes_with_suggestions"] == 1
    assert body["notes_processed"] == 2


def test_cli_json_empty_when_no_paths(monkeypatch, tmp_path):
    """Sin notas en el corpus + --json → output JSON vacío válido."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=0)
    monkeypatch.setattr(
        rag, "_load_corpus",
        lambda col: {"metas": [], "title_to_paths": {}},
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["wikilinks", "suggest", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["items"] == []


# ── Paridad CLI ↔ HTTP ────────────────────────────────────────────────────


def test_cli_and_http_same_shape_for_single_note(http_client, monkeypatch, tmp_path):
    """--note Path --json y GET ?path=Path devuelven el mismo shape."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("# A\n\nbody")

    fake_results = [{
        "title": "Beta", "target": "B.md", "line": 1,
        "char_offset": 5, "context": "c",
    }]
    monkeypatch.setattr(_server, "find_wikilink_suggestions",
                         lambda col, p, **kw: fake_results)
    monkeypatch.setattr(rag, "find_wikilink_suggestions",
                         lambda col, p, **kw: fake_results)

    http_resp = http_client.get(
        "/api/notes/wikilink-suggestions",
        params={"path": "A.md", "limit": 30},
    )
    http_body = http_resp.json()

    runner = CliRunner()
    cli_result = runner.invoke(
        rag.cli, ["wikilinks", "suggest", "--note", "A.md", "--json"],
    )
    cli_body = json.loads(cli_result.output)

    # Mismo source_path + mismas items keys.
    assert http_body["source_path"] == cli_body["source_path"]
    assert len(http_body["items"]) == len(cli_body["items"])
    if http_body["items"] and cli_body["items"]:
        assert set(http_body["items"][0].keys()) == set(cli_body["items"][0].keys())
