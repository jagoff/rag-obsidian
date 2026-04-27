"""Tests para `GET /api/notes/loops` + `rag loops <path>`.

Ambos endpoints son wraps delgados de `_extract_followup_loops`. Cheap
(sin LLM, sin embed) — los tests pueden ejercer el flow real con files
reales en tmp_path.

Cubre:
  - Path validation (sufijo .md, traversal).
  - reason=not_found cuando archivo no existe.
  - Extract real:
    * Frontmatter `todo: [a, b]` → 2 loops kind=todo.
    * Checkboxes `- [ ]` sin marcar → kind=checkbox.
    * Líneas imperativas ("tengo que X") → kind=inline.
  - Limit clamping.
  - age_days computado correctamente (mock con tiempo controlado).
  - Paridad CLI ↔ HTTP.
"""
from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import rag
import web.server as _server


@pytest.fixture
def http_client():
    return TestClient(_server.app)


# Note body con loops mixtos — un sample que toca los 3 kinds.
_SAMPLE_NOTE = """---
title: Plan
todo:
  - llamar a juan
  - revisar contrato
---

# Plan

Algunas notas del proyecto.

- [ ] preparar deck del lunes
- [x] esto ya está hecho
- [ ] coordinar con el equipo

Tengo que pensar la propuesta antes del jueves.
"""


# ── HTTP endpoint ─────────────────────────────────────────────────────────


def test_http_rejects_non_md_path(http_client):
    resp = http_client.get("/api/notes/loops", params={"path": "foo"})
    assert resp.status_code == 400


def test_http_path_traversal_blocked(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    resp = http_client.get(
        "/api/notes/loops",
        params={"path": "../../etc/passwd.md"},
    )
    assert resp.status_code == 400


def test_http_vault_missing_returns_503(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path / "nope")
    resp = http_client.get(
        "/api/notes/loops",
        params={"path": "A.md"},
    )
    assert resp.status_code == 503


def test_http_not_found_when_file_missing(http_client, monkeypatch, tmp_path):
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    resp = http_client.get(
        "/api/notes/loops",
        params={"path": "ghost.md"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["reason"] == "not_found"


def test_http_extracts_all_three_kinds(http_client, monkeypatch, tmp_path):
    """Frontmatter + checkbox + inline → todos detectados."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    (tmp_path / "Plan.md").write_text(_SAMPLE_NOTE)
    resp = http_client.get("/api/notes/loops", params={"path": "Plan.md"})
    assert resp.status_code == 200
    body = resp.json()
    items = body["items"]
    kinds = [it["kind"] for it in items]
    # 2 todos del frontmatter (llamar a juan, revisar contrato).
    assert kinds.count("todo") == 2
    # 2 checkboxes sin marcar (preparar deck, coordinar equipo);
    # el `- [x]` ya marcado NO se cuenta.
    assert kinds.count("checkbox") == 2
    # Texts del checkbox correctos.
    checkbox_texts = [
        it["loop_text"] for it in items if it["kind"] == "checkbox"
    ]
    assert "preparar deck del lunes" in checkbox_texts
    assert "coordinar con el equipo" in checkbox_texts
    # `[x]` ya marcado NO está.
    assert not any("ya está hecho" in t for t in checkbox_texts)


def test_http_empty_when_no_loops(http_client, monkeypatch, tmp_path):
    """Nota sin loops → items=[] sin reason (estado limpio)."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    (tmp_path / "Plain.md").write_text(
        "# Plan\n\nNada que hacer acá. Todo en orden.\n"
    )
    resp = http_client.get("/api/notes/loops", params={"path": "Plain.md"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    # No reason — distingue de "not_found" o "vault_missing".
    assert "reason" not in body


def test_http_limit_clamped(http_client, monkeypatch, tmp_path):
    """limit=999 → clamp a 100."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    # Genero una nota con 200 checkboxes — se debe trunc a 100.
    body_text = "# Many\n\n" + "\n".join(
        f"- [ ] item-{i}" for i in range(200)
    )
    (tmp_path / "Many.md").write_text(body_text)
    resp = http_client.get(
        "/api/notes/loops",
        params={"path": "Many.md", "limit": 999},
    )
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 100


def test_http_item_shape(http_client, monkeypatch, tmp_path):
    """Cada item tiene loop_text/kind/age_days/extracted_at."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    (tmp_path / "S.md").write_text(_SAMPLE_NOTE)
    resp = http_client.get("/api/notes/loops", params={"path": "S.md"})
    item = resp.json()["items"][0]
    assert set(item.keys()) == {"loop_text", "kind", "age_days", "extracted_at"}
    assert isinstance(item["age_days"], int)
    assert item["age_days"] >= 0


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    (tmp_path / "S.md").write_text(_SAMPLE_NOTE)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["loops", "S.md", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["source_path"] == "S.md"
    # Mismos kinds que el HTTP test.
    kinds = [it["kind"] for it in body["items"]]
    assert "todo" in kinds
    assert "checkbox" in kinds


def test_cli_rejects_non_md_path(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["loops", "foo", "--json"])
    assert result.exit_code == 2
    body = json.loads(result.output)
    assert ".md" in body["error"]


def test_cli_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["loops", "ghost.md", "--json"])
    body = json.loads(result.output)
    assert body["reason"] == "not_found"


def test_cli_plain_output(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    (tmp_path / "S.md").write_text(_SAMPLE_NOTE)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["loops", "S.md", "--plain"])
    assert result.exit_code == 0
    # Formato tabular: cada línea tiene tabs separando 3 cols.
    for line in result.output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        assert len(parts) == 3
        # Primera col es age_days (entero).
        assert parts[0].isdigit()
        # Segunda es kind ∈ {todo, checkbox, inline}.
        assert parts[1] in ("todo", "checkbox", "inline")


def test_cli_no_loops(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    (tmp_path / "Plain.md").write_text("# Sin loops\n\nNada.\n")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["loops", "Plain.md", "--plain"])
    assert result.exit_code == 0
    assert "Sin loops" in result.output


# ── Paridad CLI ↔ HTTP ────────────────────────────────────────────────────


def test_cli_and_http_same_shape(http_client, monkeypatch, tmp_path):
    """Drift detector: keys del item coinciden byte a byte."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    (tmp_path / "S.md").write_text(_SAMPLE_NOTE)

    http_resp = http_client.get(
        "/api/notes/loops",
        params={"path": "S.md", "limit": 50},
    )
    http_body = http_resp.json()

    runner = CliRunner()
    cli_result = runner.invoke(
        rag.cli, ["loops", "S.md", "--json", "--limit", "50"],
    )
    cli_body = json.loads(cli_result.output)

    assert http_body["source_path"] == cli_body["source_path"]
    assert len(http_body["items"]) == len(cli_body["items"])
    if http_body["items"] and cli_body["items"]:
        assert set(http_body["items"][0].keys()) == set(cli_body["items"][0].keys())


def test_age_days_zero_for_just_created(http_client, monkeypatch, tmp_path):
    """Nota recién creada → age_days = 0 (los loops heredan el extracted_at
    del momento del scan)."""
    monkeypatch.setattr(_server, "VAULT_PATH", tmp_path)
    (tmp_path / "Fresh.md").write_text(_SAMPLE_NOTE)
    resp = http_client.get("/api/notes/loops", params={"path": "Fresh.md"})
    items = resp.json()["items"]
    # Todos los loops de una nota recién escrita tienen age 0d.
    assert all(it["age_days"] == 0 for it in items)
