"""Tests para `GET /api/notes/related` + `rag related <path>` CLI.

Ambos endpoints son wraps delgados de `find_related()` agregados como
backend para el plugin Obsidian (panel "Notas relacionadas"). El plugin
hace HTTP primero, spawnea el CLI como fallback cuando el web está caído
— ambos paths deben devolver el mismo shape para que la divergencia no
sea silenciosa.

Tests cubren:
  - Path validation (sufijo .md, traversal blocked)
  - Empty index → reason=empty_index
  - Path no en corpus → reason=not_indexed
  - Happy path: vecinos por shared tags
  - Limit clipping (>50 → 50)
  - Vault inexistente → 503 (HTTP) / sin error nuevo (CLI no toca vault)
  - Shape paridad CLI ↔ HTTP

Mocks de `_load_corpus` + `find_related` para no levantar BM25 + sqlite-vec
real — testeamos la capa de wrap, no el algoritmo de ranking (que tiene
sus propios tests).
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import rag
import web.server as _server


# ── Fixtures comunes ──────────────────────────────────────────────────────

class _FakeCol:
    """Mimics SqliteVecCollection.count() + .id."""
    def __init__(self, count: int = 5):
        self._count = count
        self.id = "test-uuid"

    def count(self) -> int:
        return self._count


def _fake_corpus_with_metas(metas: list[dict]) -> dict:
    """Corpus mínimo aceptado por find_related. No usamos outlinks/backlinks
    en estos tests porque exercitamos el path 'tags only'."""
    return {
        "metas": metas,
        "docs": [m.get("note", "") for m in metas],
        "count": len(metas),
        "collection_id": "test",
        "outlinks": {},
        "title_to_paths": {},
        "backlinks": {},
        "tags": set(),
    }


@pytest.fixture
def http_client():
    return TestClient(_server.app)


def _patch_corpus_and_db(monkeypatch, metas: list[dict], count: int | None = None):
    """Aplica los monkeypatches necesarios para que tanto el endpoint web
    como el CLI vean el corpus mockeado.

    Hace falta patchar `_load_corpus` en TRES lugares:
      - `rag._load_corpus`        — usado internamente por `find_related`.
      - `_server._load_corpus`    — usado directo por el endpoint HTTP.
      - `rag.get_db`              — usado por ambos.
      - `_server.get_db`          — usado por el endpoint.
    Si patcheamos solo uno, `find_related` (que llama a `_load_corpus` con
    el binding del módulo `rag`) sigue viendo el original.
    """
    if count is None:
        count = max(1, len(metas))
    fake_db = _FakeCol(count=count)
    fake_corpus = _fake_corpus_with_metas(metas)
    monkeypatch.setattr(rag, "get_db", lambda: fake_db)
    monkeypatch.setattr(_server, "get_db", lambda: fake_db)
    monkeypatch.setattr(rag, "_load_corpus", lambda col: fake_corpus)
    monkeypatch.setattr(_server, "_load_corpus", lambda col: fake_corpus)


# ── HTTP endpoint: GET /api/notes/related ────────────────────────────────


def test_http_rejects_path_without_md_suffix(http_client):
    """Path sin .md → 400 con mensaje claro (no 500, no 422)."""
    resp = http_client.get("/api/notes/related", params={"path": "notes/foo"})
    assert resp.status_code == 400
    assert ".md" in resp.json()["detail"]


def test_http_rejects_missing_path(http_client):
    """Path obligatorio → 422 (FastAPI validation)."""
    resp = http_client.get("/api/notes/related")
    assert resp.status_code == 422


def test_http_path_traversal_blocked(http_client, monkeypatch, tmp_path):
    """Path que escapa el vault (../) → 400, no 200 con datos del filesystem."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "../../etc/passwd.md"},
    )
    assert resp.status_code == 400
    assert "inválido" in resp.json()["detail"] or "invalido" in resp.json()["detail"]


def test_http_vault_missing_returns_503(http_client, monkeypatch, tmp_path):
    """Vault path no existe (iCloud no synced, dev box) → 503 explícito."""
    fake_vault = tmp_path / "nonexistent_vault"
    monkeypatch.setattr(_server, "VAULT_PATH", fake_vault)
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 503
    assert "vault" in resp.json()["detail"].lower()


def test_http_empty_index_returns_empty_with_reason(http_client, monkeypatch, tmp_path):
    """Índice vacío → 200 con items=[] y reason=empty_index (sin error)."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_corpus_and_db(monkeypatch, metas=[], count=0)
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["reason"] == "empty_index"
    assert body["source_path"] == "02-Areas/A.md"


def test_http_path_not_in_corpus_returns_empty_with_reason(
    http_client, monkeypatch, tmp_path,
):
    """Path válido pero no indexado → reason=not_indexed (no error)."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_corpus_and_db(monkeypatch, metas=[
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "x"},
    ])
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["reason"] == "not_indexed"


def test_http_happy_path_returns_neighbors_with_shape(
    http_client, monkeypatch, tmp_path,
):
    """Source con 2 tags, 1 vecino con esos 2 tags → score=2, reason=tags."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud,equilibrio"},
        {"file": "02-Areas/C.md", "note": "C", "folder": "02-Areas",
         "tags": "coaching"},  # solo 1 tag → NO matchea (find_related requiere shared>=2)
    ]
    _patch_corpus_and_db(monkeypatch, metas)
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md", "limit": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_path"] == "02-Areas/A.md"
    paths = [it["path"] for it in body["items"]]
    assert "02-Areas/B.md" in paths
    assert "02-Areas/C.md" not in paths  # shared<2 sin link → excluida
    b = next(it for it in body["items"] if it["path"] == "02-Areas/B.md")
    assert b["score"] == 2
    assert b["reason"] == "tags"
    assert sorted(b["shared_tags"]) == ["coaching", "salud"]
    # tags es lista (parseada del CSV stored)
    assert isinstance(b["tags"], list)
    assert "equilibrio" in b["tags"]


def test_http_limit_clamps_to_max_50(http_client, monkeypatch, tmp_path):
    """limit=999 → clamped a 50. No error 4xx."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_corpus_and_db(monkeypatch, metas=[
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
    ])
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md", "limit": 999},
    )
    assert resp.status_code == 200


def test_http_limit_minimum_clamp(http_client, monkeypatch, tmp_path):
    """limit=0 o negativo → clamped a 1, no division-by-zero."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_corpus_and_db(monkeypatch, metas=[
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
    ])
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md", "limit": 0},
    )
    assert resp.status_code == 200


# ── CLI: rag related <path> ───────────────────────────────────────────────


def test_cli_json_output_shape(monkeypatch):
    """`rag related X.md --json` → JSON parseable con items + source_path."""
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud,equilibrio"},
    ]
    _patch_corpus_and_db(monkeypatch, metas)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source_path"] == "02-Areas/A.md"
    paths = [it["path"] for it in body["items"]]
    assert "02-Areas/B.md" in paths


def test_cli_rejects_path_without_md_suffix(monkeypatch):
    """`rag related foo --json` → exit 2 + error JSON con shape predecible."""
    _patch_corpus_and_db(monkeypatch, metas=[], count=2)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "no-md-suffix", "--json"])
    assert result.exit_code == 2
    body = json.loads(result.output)
    assert "error" in body
    assert ".md" in body["error"]
    assert body["items"] == []


def test_cli_empty_index_json(monkeypatch):
    """Índice vacío + --json → JSON con reason=empty_index."""
    _patch_corpus_and_db(monkeypatch, metas=[], count=0)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["items"] == []
    assert body["reason"] == "empty_index"


def test_cli_path_not_indexed_json(monkeypatch):
    """Path válido pero no en corpus + --json → reason=not_indexed."""
    _patch_corpus_and_db(monkeypatch, metas=[
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "x"},
    ])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md", "--json"])
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["items"] == []
    assert body["reason"] == "not_indexed"


def test_cli_plain_output_tabular(monkeypatch):
    """`rag related X.md --plain` → tabular SCORE\\tREASON\\tPATH (script-friendly)."""
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud"},
    ]
    _patch_corpus_and_db(monkeypatch, metas)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md", "--plain"])
    assert result.exit_code == 0
    # Esperamos al menos una línea con tabs separando 3 columnas.
    lines = [l for l in result.output.strip().split("\n") if l.strip()]
    assert len(lines) >= 1
    parts = lines[0].split("\t")
    assert len(parts) == 3
    score, reason, path = parts
    assert score.isdigit()
    assert reason.strip() in ("tags", "link", "tags+link")
    assert path == "02-Areas/B.md"


def test_cli_rich_output_works(monkeypatch):
    """Default (no --json/--plain) → output con rich, no crashea + tiene path."""
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud"},
    ]
    _patch_corpus_and_db(monkeypatch, metas)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md"])
    assert result.exit_code == 0
    # El output rich incluye el path destino.
    assert "02-Areas/B.md" in result.output


def test_cli_no_results_dim_message(monkeypatch):
    """Source en corpus pero sin vecinos → mensaje 'sin notas relacionadas'."""
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "tag-unique"},
    ]
    _patch_corpus_and_db(monkeypatch, metas)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["related", "02-Areas/A.md", "--plain"])
    assert result.exit_code == 0
    assert "Sin notas relacionadas" in result.output


# ── Paridad CLI ↔ HTTP ───────────────────────────────────────────────────


def test_cli_and_http_produce_same_items(http_client, monkeypatch, tmp_path):
    """El plugin asume que el fallback CLI devuelve el mismo shape que HTTP.
    Si esto se rompe (campo renombrado en uno y no en el otro), el plugin
    diverge silenciosamente al fallback. Este test es el guard."""
    metas = [
        {"file": "02-Areas/A.md", "note": "A", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/B.md", "note": "B", "folder": "02-Areas",
         "tags": "coaching,salud"},
        {"file": "02-Areas/C.md", "note": "C", "folder": "02-Areas",
         "tags": "coaching,salud"},
    ]

    # Aplicamos el patch para ambos paths a la vez (HTTP usa el de _server,
    # find_related interno usa el de rag — el helper cubre los dos).
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    _patch_corpus_and_db(monkeypatch, metas)

    # HTTP path
    resp = http_client.get(
        "/api/notes/related",
        params={"path": "02-Areas/A.md", "limit": 10},
    )
    http_body = resp.json()

    # CLI path
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["related", "02-Areas/A.md", "--json", "--limit", "10"],
    )
    cli_body = json.loads(result.output)

    # Mismo source, mismas paths, mismas keys por item.
    assert http_body["source_path"] == cli_body["source_path"]
    http_paths = sorted(it["path"] for it in http_body["items"])
    cli_paths = sorted(it["path"] for it in cli_body["items"])
    assert http_paths == cli_paths
    if http_body["items"] and cli_body["items"]:
        # Keys del item shape — drift detector.
        assert set(http_body["items"][0].keys()) == set(cli_body["items"][0].keys())
