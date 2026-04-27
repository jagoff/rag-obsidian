"""Tests para `GET /api/notes/contradictions` + `rag contradictions <path>`.

Ambos endpoints son wraps delgados de `find_contradictions_for_note`
(que es LLM-bound, 5-10s/call). Para los tests mockeamos
`find_contradictions_for_note` directamente y verificamos que:

  - El endpoint valida path (sufijo .md, traversal).
  - Reconoce y devuelve `reason` explícito (empty_index/not_indexed/too_short).
  - El shape del item coincide entre HTTP y CLI (paridad).
  - El body mínimo (<200 chars) corta antes de llamar al LLM.
  - exclude_paths={path} se pasa correctamente — si no, la nota source
    se matchearía contra sí misma.
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


def test_http_rejects_path_without_md_suffix(http_client):
    """Path sin .md → 400 antes de tocar el filesystem."""
    resp = http_client.get("/api/notes/contradictions", params={"path": "foo"})
    assert resp.status_code == 400


def test_http_rejects_missing_path(http_client):
    resp = http_client.get("/api/notes/contradictions")
    assert resp.status_code == 422


def test_http_path_traversal_blocked(http_client, monkeypatch, tmp_path):
    """`../../etc/passwd.md` → 400."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "../../etc/passwd.md"},
    )
    assert resp.status_code == 400


def test_http_vault_missing_returns_503(http_client, monkeypatch, tmp_path):
    fake = tmp_path / "nonexistent"
    monkeypatch.setattr(_server, "VAULT_PATH", fake)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 503


def test_http_not_indexed_when_file_missing(http_client, monkeypatch, tmp_path):
    """El archivo no existe en el vault → reason=not_indexed, no error."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "02-Areas/ghost.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["reason"] == "not_indexed"


def test_http_empty_index(http_client, monkeypatch, tmp_path):
    """count()==0 → reason=empty_index. No se llama al LLM."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=0)
    # Crear el archivo para que pase el is_file() check antes de llegar al
    # empty_index guard.
    f = tmp_path / "02-Areas"
    f.mkdir(parents=True)
    (f / "A.md").write_text("x" * 300)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 200
    assert resp.json()["reason"] == "empty_index"


def test_http_body_too_short(http_client, monkeypatch, tmp_path):
    """body < 200 chars → reason=too_short. No se llama al LLM."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "02-Areas").mkdir()
    (tmp_path / "02-Areas" / "A.md").write_text("Muy corto.")

    called = {"n": 0}

    def fake_find(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(_server, "find_contradictions_for_note", fake_find)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 200
    assert resp.json()["reason"] == "too_short"
    assert called["n"] == 0, "find_contradictions NO debe llamarse con body corto"


def test_http_happy_path_returns_items(http_client, monkeypatch, tmp_path):
    """Body ≥200 chars + corpus non-empty + mock del LLM → items con shape correcto."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "02-Areas").mkdir()
    note_body = "Esta nota dice X. " * 30  # ~600 chars
    (tmp_path / "02-Areas" / "A.md").write_text(note_body)

    def fake_find(col, body, exclude_paths, k):
        # Verificamos que exclude_paths contiene la nota source.
        assert "02-Areas/A.md" in exclude_paths
        return [
            {
                "path": "02-Areas/B.md",
                "note": "B",
                "snippet": "Esta otra nota dice lo contrario.",
                "why": "contradice X con no-X",
            },
        ]

    monkeypatch.setattr(_server, "find_contradictions_for_note", fake_find)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "02-Areas/A.md", "limit": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_path"] == "02-Areas/A.md"
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["path"] == "02-Areas/B.md"
    assert item["note"] == "B"
    assert item["folder"] == "02-Areas"
    assert "contradice" in item["why"]
    assert item["snippet"]


def test_http_limit_clamped(http_client, monkeypatch, tmp_path):
    """limit=999 se clampa a 10 antes de pasar a find_contradictions."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("x" * 300)

    captured = {"k": None}

    def fake_find(col, body, exclude_paths, k):
        captured["k"] = k
        return []

    monkeypatch.setattr(_server, "find_contradictions_for_note", fake_find)
    resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "A.md", "limit": 999},
    )
    assert resp.status_code == 200
    assert captured["k"] == 10


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("x" * 300)
    monkeypatch.setattr(
        rag, "find_contradictions_for_note",
        lambda col, body, exclude_paths, k: [
            {"path": "B.md", "note": "B", "snippet": "x", "why": "test"},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["contradictions", "A.md", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source_path"] == "A.md"
    assert len(body["items"]) == 1
    assert body["items"][0]["why"] == "test"


def test_cli_rejects_non_md_path(monkeypatch):
    _patch_db(monkeypatch, count=10)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["contradictions", "foo", "--json"])
    assert result.exit_code == 2
    body = json.loads(result.output)
    assert "error" in body
    assert ".md" in body["error"]


def test_cli_not_found_returns_reason(monkeypatch, tmp_path):
    """Nota no existe en el filesystem → reason=not_found (CLI; equivalent to
    not_indexed del HTTP — el CLI lee directo del vault, HTTP también pero
    marca distinto para distinguir path-valid-pero-missing vs no-indexed)."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["contradictions", "ghost.md", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["reason"] == "not_found"


def test_cli_too_short_body(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("Corto.")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["contradictions", "A.md", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["reason"] == "too_short"


def test_cli_plain_output(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("x" * 300)
    monkeypatch.setattr(
        rag, "find_contradictions_for_note",
        lambda col, body, exclude_paths, k: [
            {"path": "B.md", "note": "B", "snippet": "x", "why": "test-why"},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["contradictions", "A.md", "--plain"])
    assert result.exit_code == 0
    assert "B.md" in result.output
    assert "test-why" in result.output


# ── Paridad CLI ↔ HTTP ────────────────────────────────────────────────────


def test_cli_and_http_return_same_shape(http_client, monkeypatch, tmp_path):
    """Cualquier divergencia de shape rompe el fallback CLI del plugin —
    este test es el guard contra drift silencioso."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    _patch_db(monkeypatch, count=10)
    (tmp_path / "A.md").write_text("x" * 300)

    mock_results = [
        {"path": "B.md", "note": "B", "snippet": "snippet-b", "why": "why-b"},
    ]
    monkeypatch.setattr(_server, "find_contradictions_for_note",
                         lambda *a, **kw: mock_results)
    monkeypatch.setattr(rag, "find_contradictions_for_note",
                         lambda *a, **kw: mock_results)

    http_resp = http_client.get(
        "/api/notes/contradictions",
        params={"path": "A.md", "limit": 5},
    )
    http_body = http_resp.json()

    runner = CliRunner()
    cli_result = runner.invoke(
        rag.cli, ["contradictions", "A.md", "--json", "--limit", "5"],
    )
    cli_body = json.loads(cli_result.output)

    assert http_body["source_path"] == cli_body["source_path"]
    assert len(http_body["items"]) == len(cli_body["items"])
    if http_body["items"] and cli_body["items"]:
        # Las keys del item deben coincidir byte a byte para que el
        # plugin no diverja entre HTTP y CLI fallback.
        assert set(http_body["items"][0].keys()) == set(cli_body["items"][0].keys())
