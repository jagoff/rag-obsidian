"""Tests para `rag file` (filing assistant, fase 1 dry-run).

Cubre: folder-voting vía _suggest_folder_for_note (ya testeado indirecto en
test_inbox.py), inferencia de upward-link (MOC vs vecino), composición en
build_filing_proposal, logging a filing.jsonl, y smoke end-to-end del CLI.
"""
import json
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


# ── Fixtures (idénticas en espíritu a test_surface.py) ────────────────────────


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "FILING_LOG_PATH", tmp_path / "filing.jsonl")
    # Post-T10: filing log writes to rag_filing_log (SQL). Point DB_PATH at
    # the test dir so writes land in a throwaway ragvec.db.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    def fake_embed(texts):
        # Single 1-hot embedding per call — tests overridearán por query
        # cuando necesiten diferenciar similarity.
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", fake_embed)

    client = _TestVecClient(path=str(tmp_path / "chroma"))
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


def test_filing_log_writes_sql(tmp_vault, tmp_path):
    """Post-T10: filing log writes to rag_filing_log in SQL."""
    import sqlite3
    vault, col = tmp_vault
    prop = {"path": "00-Inbox/x.md", "note": "X", "folder": "02-Areas",
            "confidence": 0.8, "upward_title": "", "upward_kind": "",
            "neighbors": []}
    rag._filing_log_proposal(prop)

    conn = sqlite3.connect(str(tmp_path / "ragvec.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT cmd, path, folder, ts FROM rag_filing_log"
        ).fetchall())
    finally:
        conn.close()
    assert len(rows) == 1
    entry = rows[0]
    assert entry["cmd"] == "filing_proposal"
    assert entry["path"] == "00-Inbox/x.md"
    assert entry["folder"] == "02-Areas"
    assert entry["ts"]


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
    # Post-T10: log debe tener al menos una propuesta en rag_filing_log.
    import sqlite3
    conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    try:
        n = conn.execute("SELECT COUNT(*) FROM rag_filing_log").fetchone()[0]
    finally:
        conn.close()
    assert n >= 1


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


# ── Fase 2: apply + undo ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_vault_with_batch(tmp_vault, monkeypatch):
    """Misma fixture + redirige FILING_BATCHES_DIR al tmp_path."""
    vault, col = tmp_vault
    monkeypatch.setattr(
        rag, "FILING_BATCHES_DIR",
        vault.parent / "filing_batches",
    )
    return vault, col


# _append_upward_link / _remove_upward_link


def test_append_upward_link_adds_block(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Título\n\nCuerpo.", encoding="utf-8")
    assert rag._append_upward_link(p, "MOC Salud") is True
    text = p.read_text()
    assert rag.FILING_UPWARD_MARKER in text
    assert "[[MOC Salud]]" in text
    assert "# Título" in text  # contenido intacto


def test_append_upward_link_is_idempotent(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# A\n\nCuerpo.", encoding="utf-8")
    rag._append_upward_link(p, "MOC 1")
    rag._append_upward_link(p, "MOC 2")   # reemplaza, no duplica
    text = p.read_text()
    assert text.count(rag.FILING_UPWARD_MARKER) == 1
    assert "[[MOC 2]]" in text
    assert "[[MOC 1]]" not in text


def test_remove_upward_link_roundtrip(tmp_path):
    p = tmp_path / "note.md"
    original = "# Título\n\nCuerpo.\n"
    p.write_text(original, encoding="utf-8")
    rag._append_upward_link(p, "MOC Salud")
    assert rag._remove_upward_link(p) is True
    # Volver al original (modulo trailing newline normalization).
    assert "MOC Salud" not in p.read_text()
    assert "# Título" in p.read_text()
    assert "Cuerpo." in p.read_text()


def test_remove_upward_link_noop_when_missing(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Sin link\n", encoding="utf-8")
    assert rag._remove_upward_link(p) is False


# _apply_filing_move


def test_apply_filing_move_happy_path(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/target.md", "target", HIGH,
              body="contenido")
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    entry = rag._apply_filing_move(
        col, "00-Inbox/target.md", "02-Areas/Salud", "MOC Salud"
    )

    assert entry["src"] == "00-Inbox/target.md"
    assert entry["dst"] == "02-Areas/Salud/target.md"
    assert entry["upward_title"] == "MOC Salud"
    assert entry["upward_written"] is True

    # File on disk: movido + con upward-link.
    assert not (vault / "00-Inbox/target.md").exists()
    assert (vault / "02-Areas/Salud/target.md").exists()
    content = (vault / "02-Areas/Salud/target.md").read_text()
    assert "[[MOC Salud]]" in content
    assert "contenido" in content


def test_apply_filing_move_creates_missing_target_folder(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/new.md", "new", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    # 02-Areas/Nueva no existe.
    rag._apply_filing_move(col, "00-Inbox/new.md", "02-Areas/Nueva", "")

    assert (vault / "02-Areas/Nueva/new.md").exists()


def test_apply_filing_move_raises_if_dst_exists(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/clash.md", "clash", HIGH)
    (vault / "02-Areas" / "clash.md").parent.mkdir(parents=True, exist_ok=True)
    (vault / "02-Areas" / "clash.md").write_text("existente", encoding="utf-8")
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    with pytest.raises(FileExistsError):
        rag._apply_filing_move(col, "00-Inbox/clash.md", "02-Areas", "")


def test_apply_filing_move_rejects_path_traversal(tmp_vault_with_batch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/x.md", "x", HIGH)

    with pytest.raises(ValueError):
        rag._apply_filing_move(col, "00-Inbox/x.md", "../escape", "")


# Batch persistence


def test_write_batch_log_creates_jsonl(tmp_vault_with_batch):
    vault, col = tmp_vault_with_batch
    entries = [
        {"src": "00-Inbox/a.md", "dst": "02-Areas/a.md",
         "upward_title": "X", "upward_written": True},
        {"src": "00-Inbox/b.md", "dst": "02-Areas/b.md",
         "upward_title": "", "upward_written": False},
    ]
    path = rag._write_filing_batch(entries)
    assert path is not None and path.is_file()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["src"] == "00-Inbox/a.md"


def test_write_batch_log_empty_returns_none(tmp_vault_with_batch):
    assert rag._write_filing_batch([]) is None


def test_last_filing_batch_returns_most_recent(tmp_vault_with_batch):
    import time
    vault, col = tmp_vault_with_batch
    rag.FILING_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    p1 = rag.FILING_BATCHES_DIR / "20260101-000000.jsonl"
    p1.write_text("{}\n")
    time.sleep(0.01)
    p2 = rag.FILING_BATCHES_DIR / "20260102-000000.jsonl"
    p2.write_text("{}\n")

    assert rag._last_filing_batch() == p2


# Rollback


def test_rollback_restores_files_and_removes_upward(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/u.md", "u", HIGH, body="cuerpo u")
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    entry = rag._apply_filing_move(col, "00-Inbox/u.md", "02-Areas", "MOC X")
    batch = rag._write_filing_batch([entry])

    assert (vault / "02-Areas/u.md").exists()
    assert not (vault / "00-Inbox/u.md").exists()

    results = rag._rollback_filing_batch(col, batch)

    assert len(results) == 1
    assert results[0]["ok"] is True
    # Revertido: de vuelta en Inbox, sin upward-link.
    assert (vault / "00-Inbox/u.md").is_file()
    assert not (vault / "02-Areas/u.md").exists()
    content = (vault / "00-Inbox/u.md").read_text()
    assert "MOC X" not in content
    assert "cuerpo u" in content


def test_rollback_reports_error_when_dst_missing(tmp_vault_with_batch):
    vault, col = tmp_vault_with_batch
    # Batch apunta a un dst que nunca existió.
    batch = rag.FILING_BATCHES_DIR
    batch.mkdir(parents=True, exist_ok=True)
    bp = batch / "fake.jsonl"
    bp.write_text(json.dumps({
        "src": "00-Inbox/ghost.md", "dst": "02-Areas/ghost.md",
        "upward_title": "", "upward_written": False,
    }) + "\n")

    results = rag._rollback_filing_batch(col, bp)
    assert results[0]["ok"] is False
    assert "no existe" in results[0]["error"]


# CLI apply (interactive, driven by CliRunner stdin)


def test_cli_apply_moves_on_y(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH, body="c")
    _add_note(col, vault, "02-Areas/Salud/a.md", "A", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    # 'y' acepta la propuesta.
    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--apply"], input="y\n")
    assert result.exit_code == 0, result.output
    assert not (vault / "00-Inbox/n.md").exists()
    assert (vault / "02-Areas/Salud/n.md").exists()


def test_cli_apply_skip_on_s(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH, body="c")
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--apply"], input="s\n")
    assert result.exit_code == 0, result.output
    assert (vault / "00-Inbox/n.md").exists()


def test_cli_apply_edit_uses_new_folder(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH, body="c")
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    # 'e' + new target → debería mover a 03-Resources/Custom.
    result = CliRunner().invoke(
        rag.file_cmd, ["--plain", "--apply"],
        input="e\n03-Resources/Custom\n",
    )
    assert result.exit_code == 0, result.output
    assert (vault / "03-Resources/Custom/n.md").exists()
    assert not (vault / "00-Inbox/n.md").exists()


def test_cli_apply_quit_stops_and_saves_partial_batch(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/a.md", "a", HIGH)
    _add_note(col, vault, "00-Inbox/b.md", "b", HIGH)
    _add_note(col, vault, "00-Inbox/c.md", "c", HIGH)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    # Aceptar primera, quit en la segunda.
    result = CliRunner().invoke(
        rag.file_cmd, ["--plain", "--apply"], input="y\nq\n",
    )
    assert result.exit_code == 0, result.output
    # Primera movida; segunda y tercera se quedan.
    assert (vault / "02-Areas/a.md").exists()
    assert (vault / "00-Inbox/b.md").exists()
    assert (vault / "00-Inbox/c.md").exists()


def test_cli_undo_reverses_last_batch(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH, body="c")
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    # Apply, después undo.
    CliRunner().invoke(rag.file_cmd, ["--plain", "--apply"], input="y\n")
    assert (vault / "02-Areas/n.md").exists()

    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--undo"])
    assert result.exit_code == 0, result.output
    assert (vault / "00-Inbox/n.md").exists()
    assert not (vault / "02-Areas/n.md").exists()
    # El batch se rename a .undone para trazabilidad.
    assert any(
        p.suffix == ".undone"
        for p in rag.FILING_BATCHES_DIR.glob("*")
    )


def test_cli_undo_without_batch_is_graceful(tmp_vault_with_batch):
    from click.testing import CliRunner
    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--undo"])
    assert result.exit_code == 0
    assert "No hay batches" in result.output


def test_cli_apply_and_undo_mutually_exclusive(tmp_vault_with_batch):
    from click.testing import CliRunner
    result = CliRunner().invoke(
        rag.file_cmd, ["--plain", "--apply", "--undo"],
    )
    assert result.exit_code == 0
    assert "No podés combinar" in result.output


# ── Fase 3: personalización ───────────────────────────────────────────────────


def _seed_decision(applied_to: str, decision: str = "accept", src: str = "00-Inbox/x.md"):
    """Post-T10: seed a decision into rag_filing_log (SQL)."""
    event = {
        "ts": "2026-04-14T10:00:00",
        "cmd": "filing_proposal",
        "path": src,
        "note": Path(src).stem,
        "folder": str(Path(applied_to).parent),
        "confidence": 0.8,
        "applied_to": applied_to,
        "decision": decision,
    }
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_filing_log", rag._map_filing_row(event))


def test_load_filing_decisions_empty_when_no_log(tmp_vault):
    assert rag._load_filing_decisions() == []


def test_load_filing_decisions_filters_to_accept_and_edit(tmp_vault):
    _seed_decision("02-Areas/Salud/a.md", decision="accept")
    _seed_decision("02-Areas/Salud/b.md", decision="reject")  # filtered out
    _seed_decision("02-Areas/Salud/c.md", decision="skip")    # filtered out
    _seed_decision("02-Areas/Tech/d.md", decision="edit")     # kept

    decisions = rag._load_filing_decisions()
    paths = [d["applied_to"] for d in decisions]
    assert "02-Areas/Salud/a.md" in paths
    assert "02-Areas/Tech/d.md" in paths
    assert "02-Areas/Salud/b.md" not in paths
    assert "02-Areas/Salud/c.md" not in paths


def test_load_filing_decisions_skips_entries_without_applied_to(tmp_vault):
    # Post-T10: dry-run entries (sin applied_to) viven en rag_filing_log.
    event = {
        "ts": "2026-04-14T10:00:00",
        "cmd": "filing_proposal",
        "path": "00-Inbox/x.md",
        "decision": "accept",
    }
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_filing_log", rag._map_filing_row(event))
    assert rag._load_filing_decisions() == []


def test_load_filing_decisions_respects_limit(tmp_vault):
    for i in range(20):
        _seed_decision(f"02-Areas/n{i}.md")
    out = rag._load_filing_decisions(limit=5)
    assert len(out) == 5


# _personalized_folder_vote


def test_personalized_vote_empty_when_no_history(tmp_vault):
    vault, col = tmp_vault
    votes, ev = rag._personalized_folder_vote(col, [1.0, 0.0, 0.0, 0.0], history=[])
    assert votes == {} and ev == []


def test_personalized_vote_empty_when_history_paths_missing(tmp_vault):
    vault, col = tmp_vault
    history = [{"applied_to": "02-Areas/ghost.md", "target_folder": "02-Areas"}]
    votes, ev = rag._personalized_folder_vote(col, [1.0, 0.0, 0.0, 0.0], history=history)
    assert votes == {} and ev == []


def test_personalized_vote_aggregates_by_folder(tmp_vault):
    vault, col = tmp_vault
    # Tres notas en Salud que el usuario "fileó" en el pasado, todas con
    # embeddings cercanos al query.
    _add_note(col, vault, "02-Areas/Salud/a.md", "A", HIGH)
    _add_note(col, vault, "02-Areas/Salud/b.md", "B", MID)
    _add_note(col, vault, "02-Areas/Tech/c.md", "C", LOW)
    history = [
        {"applied_to": "02-Areas/Salud/a.md", "target_folder": "02-Areas/Salud"},
        {"applied_to": "02-Areas/Salud/b.md", "target_folder": "02-Areas/Salud"},
        {"applied_to": "02-Areas/Tech/c.md", "target_folder": "02-Areas/Tech"},
    ]
    votes, ev = rag._personalized_folder_vote(col, list(HIGH), history=history)
    # Salud debería ganar (suma de sims más alta).
    assert max(votes, key=votes.get) == "02-Areas/Salud"
    assert votes["02-Areas/Salud"] > votes.get("02-Areas/Tech", 0)
    # Tech queda fuera por estar bajo MIN_SIM (0.3 vs sim ~0.3 borderline).
    # Evidence ordenada por similitud desc.
    assert ev[0]["sim"] >= ev[-1]["sim"]


def test_personalized_vote_respects_min_sim_threshold(tmp_vault):
    vault, col = tmp_vault
    # Solo notas en Tech, todas muy disímiles del query.
    _add_note(col, vault, "02-Areas/Tech/a.md", "A", LOW)  # cos ≈ 0.3 (borderline)
    history = [{"applied_to": "02-Areas/Tech/a.md", "target_folder": "02-Areas/Tech"}]
    # Query MUY ortogonal → similitud baja → nada cuenta.
    votes, ev = rag._personalized_folder_vote(col, [0.0, 1.0, 0.0, 0.0], history=history)
    assert votes == {}


# build_filing_proposal con history


def test_build_proposal_baseline_when_history_below_threshold(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    # Solo 2 decisiones — bajo el umbral de FILING_PERSONALIZE_MIN_HISTORY=5.
    _seed_decision("02-Areas/Coaching/a.md")
    _seed_decision("02-Areas/Coaching/b.md")

    prop = rag.build_filing_proposal(col, "00-Inbox/n.md")
    assert prop["source"] == "baseline"
    # Con 2 decisiones no hay personalización — folder viene del baseline puro.
    assert prop["folder"] == "02-Areas"


def test_build_proposal_agreed_boosts_confidence(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH)
    # Baseline: vecinos cercanos en Salud.
    for i in range(5):
        _add_note(col, vault, f"02-Areas/Salud/n{i}.md", f"N{i}", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    # History de 5+ notas también en Salud.
    for i in range(5):
        _seed_decision(f"02-Areas/Salud/n{i}.md")

    prop = rag.build_filing_proposal(col, "00-Inbox/n.md")
    assert prop["folder"] == "02-Areas/Salud"
    assert prop["source"] == "agreed"
    # Confidence boosted.
    assert prop["confidence"] > 0.7   # baseline = 1.0 (todos votan), boost cap a 0.99


def test_build_proposal_personalized_overrides_when_stronger(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    # Inbox note.
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH)
    # Baseline: vecinos en MUCHOS folders distintos → confidence baja.
    _add_note(col, vault, "02-Areas/A/x.md", "Ax", HIGH)
    _add_note(col, vault, "02-Areas/B/x.md", "Bx", HIGH)
    _add_note(col, vault, "02-Areas/C/x.md", "Cx", HIGH)
    _add_note(col, vault, "02-Areas/D/x.md", "Dx", HIGH)
    # Personalized signal: 5 decisiones consistentes en 02-Areas/Personal.
    _add_note(col, vault, "02-Areas/Personal/p1.md", "P1", MID)
    _add_note(col, vault, "02-Areas/Personal/p2.md", "P2", MID)
    _add_note(col, vault, "02-Areas/Personal/p3.md", "P3", MID)
    _add_note(col, vault, "02-Areas/Personal/p4.md", "P4", MID)
    _add_note(col, vault, "02-Areas/Personal/p5.md", "P5", MID)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])
    for i in range(1, 6):
        _seed_decision(f"02-Areas/Personal/p{i}.md")

    prop = rag.build_filing_proposal(col, "00-Inbox/n.md")
    # Baseline ≤ 0.25 (cuatro folders empatados) — personalización debería ganar.
    assert prop["source"] in ("personalized", "agreed")
    if prop["source"] == "personalized":
        assert prop["folder"] == "02-Areas/Personal"


def test_build_proposal_baseline_plus_history_when_baseline_stronger(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH)
    # Baseline FUERTE: 6 vecinos en mismo folder.
    for i in range(6):
        _add_note(col, vault, f"02-Areas/Strong/n{i}.md", f"N{i}", HIGH)
    # Personalized débil: una sola decisión (que sí matchea HIGH) en otro folder.
    _add_note(col, vault, "02-Areas/Other/p1.md", "P1", MID)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])
    # 5 decisiones para superar el umbral, pero solo 1 con sim suficiente.
    _seed_decision("02-Areas/Other/p1.md")
    for i in range(4):
        _seed_decision(f"02-Areas/Strong/n{i}.md")

    prop = rag.build_filing_proposal(col, "00-Inbox/n.md")
    # Baseline manda (confidence 1.0); estado posibles: agreed (la mayoría
    # de history coincide) o baseline+history (no coincide).
    assert prop["folder"] == "02-Areas/Strong"
    assert prop["source"] in ("agreed", "baseline+history", "baseline")


def test_cli_apply_logs_applied_to_for_fase3_history(tmp_vault_with_batch, monkeypatch):
    """Smoke: tras --apply, la línea loggeada debe tener applied_to + decision
    para que la fase 3 pueda usarla como historia."""
    vault, col = tmp_vault_with_batch
    _add_note(col, vault, "00-Inbox/n.md", "n", HIGH)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    CliRunner().invoke(rag.file_cmd, ["--plain", "--apply"], input="y\n")

    # Log debe tener al menos una entry con decision=accept y applied_to.
    decisions = rag._load_filing_decisions()
    assert len(decisions) >= 1
    assert decisions[0]["decision"] == "accept"
    assert decisions[0]["applied_to"].startswith("02-Areas/")


def test_frontmatter_file_skip_is_honored(tmp_vault_with_batch, monkeypatch):
    vault, col = tmp_vault_with_batch
    body_skip = "---\nfile: skip\n---\n\nno tocar"
    _add_note(col, vault, "00-Inbox/protected.md", "protected", HIGH,
              body=body_skip)
    _add_note(col, vault, "02-Areas/x.md", "X", HIGH)
    monkeypatch.setattr(rag, "embed", lambda ts: [list(HIGH) for _ in ts])

    from click.testing import CliRunner
    # Apply — pero la nota está opted-out, así que no hay nada que preguntar.
    result = CliRunner().invoke(rag.file_cmd, ["--plain", "--apply"])
    assert result.exit_code == 0
    # Sigue en Inbox.
    assert (vault / "00-Inbox/protected.md").exists()
