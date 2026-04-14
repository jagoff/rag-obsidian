"""Tests para `rag file` (filing assistant, fase 1 dry-run).

Cubre: folder-voting vía _suggest_folder_for_note (ya testeado indirecto en
test_inbox.py), inferencia de upward-link (MOC vs vecino), composición en
build_filing_proposal, logging a filing.jsonl, y smoke end-to-end del CLI.
"""
import json
from pathlib import Path

import chromadb
import numpy as np
import pytest

import rag


# ── Fixtures (idénticas en espíritu a test_surface.py) ────────────────────────


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "FILING_LOG_PATH", tmp_path / "filing.jsonl")

    def fake_embed(texts):
        # Single 1-hot embedding per call — tests overridearán por query
        # cuando necesiten diferenciar similarity.
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", fake_embed)

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="filing_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()
    return vault, col


def _add_note(col, vault: Path, rel: str, title: str, embedding,
              body: str | None = None, tags: str = "", outlinks: str = ""):
    full = vault / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body or f"# {title}\n\ncontenido de {title}",
                    encoding="utf-8")
    col.add(
        ids=[f"{rel}::0"],
        embeddings=[list(embedding)],
        documents=[body or f"chunk for {title}"],
        metadatas=[{
            "file": rel, "note": title,
            "folder": str(Path(rel).parent),
            "tags": tags, "outlinks": outlinks, "hash": "x",
        }],
    )
    rag._invalidate_corpus_cache()


# Vectores preparados:
HIGH = [1.0, 0.0, 0.0, 0.0]
MID = [0.8, 0.6, 0.0, 0.0]   # cos(HIGH, MID) = 0.8
LOW = [0.3, 0.0, 0.954, 0.0]  # cos(HIGH, LOW) ≈ 0.3


# ── _top_k_neighbors ──────────────────────────────────────────────────────────


def test_top_k_excludes_inbox(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/new.md", "New", HIGH)
    _add_note(col, vault, "00-Inbox/other-inbox.md", "Other Inbox", HIGH)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)

    # Forzar embed uniform para que TODO parezca max-sim — así el filtro
    # Inbox es lo único que discrimina.
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    neighbors = rag._top_k_neighbors(col, "00-Inbox/new.md", k=5)
    files = [m.get("file") for m, _ in neighbors]
    assert "02-Areas/x.md" in files
    assert all(not f.startswith("00-") for f in files)


def test_top_k_excludes_self(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "02-Areas/a.md", "A", HIGH)
    _add_note(col, vault, "02-Areas/b.md", "B", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    neighbors = rag._top_k_neighbors(col, "02-Areas/a.md", k=5)
    files = [m.get("file") for m, _ in neighbors]
    assert "02-Areas/a.md" not in files


def test_top_k_empty_note_returns_empty(tmp_vault):
    vault, col = tmp_vault
    (vault / "00-Inbox" / "empty.md").write_text("", encoding="utf-8")
    assert rag._top_k_neighbors(col, "00-Inbox/empty.md") == []


# ── _infer_upward_link ────────────────────────────────────────────────────────


def test_upward_link_prefers_moc_by_title():
    neighbors = [
        ({"file": "02-Areas/x.md", "note": "X", "tags": ""}, 0.9),
        ({"file": "02-Areas/moc.md", "note": "MOC Salud", "tags": ""}, 0.7),
    ]
    title, kind = rag._infer_upward_link(neighbors)
    assert title == "MOC Salud"
    assert kind == "moc"


def test_upward_link_prefers_moc_by_tag():
    neighbors = [
        ({"file": "02-Areas/x.md", "note": "X", "tags": "normal"}, 0.9),
        ({"file": "02-Areas/idx.md", "note": "Indice general",
          "tags": "moc,otro"}, 0.6),
    ]
    title, kind = rag._infer_upward_link(neighbors)
    assert title == "Indice general"
    assert kind == "moc"


def test_upward_link_falls_back_to_top_neighbor():
    # Sin MOCs → top-1 como link horizontal.
    neighbors = [
        ({"file": "02-Areas/x.md", "note": "X", "tags": ""}, 0.9),
        ({"file": "02-Areas/y.md", "note": "Y", "tags": ""}, 0.7),
    ]
    title, kind = rag._infer_upward_link(neighbors)
    assert title == "X"
    assert kind == "neighbor"


def test_upward_link_empty_neighbors_returns_empty():
    assert rag._infer_upward_link([]) == ("", "")


# ── build_filing_proposal ─────────────────────────────────────────────────────


def test_build_filing_proposal_happy_path(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    # Target en Inbox.
    _add_note(col, vault, "00-Inbox/postura.md", "postura", HIGH,
              body="Sobre la postura al tocar guitarra.")
    # Vecinos en Salud (mayoría).
    _add_note(col, vault, "02-Areas/Salud/a.md", "Dolor lumbar", HIGH)
    _add_note(col, vault, "02-Areas/Salud/b.md", "Respiración", HIGH)
    _add_note(col, vault, "02-Areas/Salud/MOC.md", "MOC Salud", HIGH)
    # Ruido en otra área.
    _add_note(col, vault, "03-Resources/other.md", "Otro", LOW)

    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    prop = rag.build_filing_proposal(col, "00-Inbox/postura.md", k=5)
    assert prop["path"] == "00-Inbox/postura.md"
    assert prop["note"] == "postura"
    assert prop["folder"] == "02-Areas/Salud"
    assert prop["confidence"] > 0.5
    # Upward-link detecta el MOC.
    assert prop["upward_title"] == "MOC Salud"
    assert prop["upward_kind"] == "moc"
    assert len(prop["neighbors"]) > 0


def test_build_filing_proposal_not_found(tmp_vault):
    vault, col = tmp_vault
    prop = rag.build_filing_proposal(col, "00-Inbox/no-existe.md")
    assert prop == {"path": "00-Inbox/no-existe.md", "error": "not_found"}


# ── Logging ───────────────────────────────────────────────────────────────────


def test_filing_log_writes_jsonl(tmp_vault):
    vault, col = tmp_vault
    prop = {"path": "00-Inbox/x.md", "note": "X", "folder": "02-Areas",
            "confidence": 0.8, "upward_title": "", "upward_kind": "",
            "neighbors": []}
    rag._filing_log_proposal(prop)

    lines = rag.FILING_LOG_PATH.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["cmd"] == "filing_proposal"
    assert entry["path"] == "00-Inbox/x.md"
    assert entry["folder"] == "02-Areas"
    assert "ts" in entry


# ── CLI integration ──────────────────────────────────────────────────────────


def test_cli_file_dry_run_end_to_end(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/note.md", "note", HIGH,
              body="contenido sobre guitarra")
    _add_note(col, vault, "02-Areas/Musica/a.md", "Guitar tone", HIGH)
    _add_note(col, vault, "02-Areas/Musica/b.md", "Acordes", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.file_cmd, ["--plain"])
    assert result.exit_code == 0, result.output
    assert "00-Inbox/note.md" in result.output
    assert "02-Areas/Musica" in result.output
    # Log debe tener al menos una propuesta.
    lines = rag.FILING_LOG_PATH.read_text().splitlines()
    assert len(lines) >= 1


def test_cli_file_single_path_argument(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/target.md", "target", HIGH,
              body="contenido específico")
    _add_note(col, vault, "00-Inbox/other.md", "other", HIGH,
              body="otra cosa")
    _add_note(col, vault, "02-Areas/a.md", "A", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    result = CliRunner().invoke(
        rag.file_cmd, ["--plain", "00-Inbox/target.md"]
    )
    assert result.exit_code == 0, result.output
    assert "00-Inbox/target.md" in result.output
    # Solo una propuesta porque pedimos 1 nota.
    assert "00-Inbox/other.md" not in result.output


def test_cli_file_one_processes_only_oldest(tmp_vault, monkeypatch):
    import time
    vault, col = tmp_vault
    # Crear 3 notas con mtimes distintos — la más vieja primero.
    for i, name in enumerate(["first", "second", "third"]):
        _add_note(col, vault, f"00-Inbox/{name}.md", name, HIGH)
        # mtime ascendente para que sort() ordene first→third.
        p = vault / f"00-Inbox/{name}.md"
        import os
        os.utime(p, (time.time() - 1000 + i*10, time.time() - 1000 + i*10))
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--one"])
    assert result.exit_code == 0, result.output
    assert "first" in result.output
    assert "second" not in result.output
    assert "third" not in result.output


def test_cli_file_empty_inbox_is_graceful(tmp_vault):
    vault, col = tmp_vault
    from click.testing import CliRunner
    result = CliRunner().invoke(rag.file_cmd, ["--plain"])
    assert result.exit_code == 0
    assert "Sin notas" in result.output


def test_cli_file_dry_run_never_moves_files(tmp_vault, monkeypatch):
    """Contrato de fase 1: dry-run puro, no mueve ni reescribe archivos."""
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/note.md", "note", HIGH)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    original_path = vault / "00-Inbox/note.md"
    original_content = original_path.read_text()

    from click.testing import CliRunner
    CliRunner().invoke(rag.file_cmd, ["--plain"])

    # Nota sigue en Inbox, contenido intacto.
    assert original_path.exists()
    assert original_path.read_text() == original_content
    # No aparecieron duplicados en Areas.
    assert not (vault / "02-Areas" / "note.md").exists()
