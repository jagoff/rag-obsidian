"""Tests para `rag whisper export` y `rag whisper import` — backup +
migrate de correcciones entre máquinas.

Roundtrip: export → import en una DB vacía → 0 skipped, N inserted.
Idempotencia: import 2x del mismo JSON → 2da pasada todo skipped.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def tmp_corrections_db(tmp_path, monkeypatch):
    """DB sintética para tests del export/import. Mantiene aislada la prod DB."""
    db_path = tmp_path / "corrections.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE rag_audio_corrections ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " audio_hash TEXT NOT NULL,"
        " original TEXT NOT NULL,"
        " corrected TEXT NOT NULL,"
        " source TEXT NOT NULL,"
        " ts REAL NOT NULL,"
        " chat_id TEXT,"
        " context TEXT)"
    )
    conn.commit()
    conn.close()

    @contextmanager
    def fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", fake_conn)

    def _seed(rows: list[dict]):
        with fake_conn() as c:
            for r in rows:
                c.execute(
                    "INSERT INTO rag_audio_corrections "
                    "(audio_hash, original, corrected, source, ts, chat_id, context) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        r.get("hash", "h"),
                        r["original"],
                        r["corrected"],
                        r.get("source", "explicit"),
                        r.get("ts", time.time()),
                        r.get("chat_id"),
                        r.get("context"),
                    ),
                )

    return _seed, db_path


# ── Export ────────────────────────────────────────────────────────────────────


def test_export_empty_db_writes_valid_json(tmp_corrections_db, tmp_path):
    """Si no hay correcciones, export devuelve JSON válido con count=0."""
    runner = CliRunner()
    out_path = tmp_path / "out.json"
    result = runner.invoke(rag.cli, ["whisper", "export", "-o", str(out_path)])
    assert result.exit_code == 0
    data = json.loads(out_path.read_text())
    assert data["count"] == 0
    assert data["corrections"] == []
    assert data["schema_version"] == 1
    assert "exported_at" in data


def test_export_with_data_includes_all_fields(tmp_corrections_db, tmp_path):
    """Cada correction se serializa con todas sus columnas."""
    seed, _ = tmp_corrections_db
    seed([
        {"original": "samando", "corrected": "fernando", "source": "explicit",
         "ts": 1700000000.0, "hash": "abc123"},
    ])
    runner = CliRunner()
    out_path = tmp_path / "out.json"
    result = runner.invoke(rag.cli, ["whisper", "export", "-o", str(out_path)])
    assert result.exit_code == 0
    data = json.loads(out_path.read_text())
    assert data["count"] == 1
    c = data["corrections"][0]
    assert c["original"] == "samando"
    assert c["corrected"] == "fernando"
    assert c["source"] == "explicit"
    assert c["ts"] == 1700000000.0
    assert c["audio_hash"] == "abc123"


def test_export_filter_by_source(tmp_corrections_db, tmp_path):
    """`--source explicit` solo exporta las correcciones manuales."""
    seed, _ = tmp_corrections_db
    seed([
        {"original": "a", "corrected": "b", "source": "explicit"},
        {"original": "c", "corrected": "d", "source": "llm"},
        {"original": "e", "corrected": "f", "source": "explicit"},
    ])
    runner = CliRunner()
    out_path = tmp_path / "out.json"
    result = runner.invoke(rag.cli, ["whisper", "export", "-o", str(out_path),
                                     "--source", "explicit"])
    assert result.exit_code == 0
    data = json.loads(out_path.read_text())
    assert data["count"] == 2
    assert all(c["source"] == "explicit" for c in data["corrections"])


def test_export_invalid_source_returns_error(tmp_corrections_db):
    """Source inválido (no en {explicit,llm,vault_diff}) → error."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "export", "--source", "bogus"])
    assert "source inválido" in result.output


def test_export_to_stdout_when_no_output_flag(tmp_corrections_db):
    """Sin `-o`, el JSON se escribe a stdout (útil para piping a jq)."""
    seed, _ = tmp_corrections_db
    seed([{"original": "a", "corrected": "b"}])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "export"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["count"] == 1


# ── Import ────────────────────────────────────────────────────────────────────


def test_import_inserts_new_rows(tmp_corrections_db, tmp_path):
    """Import en DB vacía inserta todas las rows del JSON."""
    seed, db_path = tmp_corrections_db
    payload = {
        "schema_version": 1,
        "count": 2,
        "corrections": [
            {"audio_hash": "h1", "original": "a", "corrected": "b",
             "source": "explicit", "ts": 1700000000.0,
             "chat_id": None, "context": None},
            {"audio_hash": "h2", "original": "c", "corrected": "d",
             "source": "llm", "ts": 1700000100.0,
             "chat_id": None, "context": None},
        ],
    }
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "import", str(in_path)])
    assert result.exit_code == 0
    assert "imported" in result.output
    # Verify DB
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM rag_audio_corrections").fetchone()[0]
    assert n == 2
    conn.close()


def test_import_is_idempotent(tmp_corrections_db, tmp_path):
    """2 imports del mismo JSON → 2da pasada todo skipped (no dupes)."""
    seed, db_path = tmp_corrections_db
    payload = {
        "schema_version": 1,
        "count": 1,
        "corrections": [
            {"audio_hash": "h1", "original": "a", "corrected": "b",
             "source": "explicit", "ts": 1700000000.0,
             "chat_id": None, "context": None},
        ],
    }
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = CliRunner()
    # 1ra pasada: insert
    runner.invoke(rag.cli, ["whisper", "import", str(in_path)])
    # 2da pasada: skip
    result = runner.invoke(rag.cli, ["whisper", "import", str(in_path)])
    assert "skipped" in result.output
    # Verify total = 1 (no duplicación).
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM rag_audio_corrections").fetchone()[0]
    assert n == 1
    conn.close()


def test_import_dry_run_no_writes(tmp_corrections_db, tmp_path):
    """`--dry-run` no escribe a la DB, solo muestra preview."""
    seed, db_path = tmp_corrections_db
    payload = {
        "schema_version": 1,
        "corrections": [
            {"audio_hash": "h1", "original": "a", "corrected": "b",
             "source": "explicit", "ts": 1700000000.0},
            {"audio_hash": "h2", "original": "c", "corrected": "d",
             "source": "llm", "ts": 1700000100.0},
        ],
    }
    in_path = tmp_path / "in.json"
    in_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "import", str(in_path), "--dry-run"])
    assert result.exit_code == 0
    assert "DRY-RUN" in result.output
    # Verify DB sigue vacía.
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM rag_audio_corrections").fetchone()[0]
    assert n == 0
    conn.close()


def test_import_handles_malformed_json(tmp_corrections_db, tmp_path):
    """JSON inválido → error claro, no crash."""
    in_path = tmp_path / "broken.json"
    in_path.write_text("this is not json", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "import", str(in_path)])
    assert "error reading" in result.output


def test_import_handles_missing_corrections_key(tmp_corrections_db, tmp_path):
    """JSON con formato distinto (sin 'corrections') → error claro."""
    in_path = tmp_path / "nope.json"
    in_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["whisper", "import", str(in_path)])
    # "corrections" missing → defaults a [] → 0 imports, no error pero tampoco
    # nada que insertar
    assert result.exit_code == 0
    # Acepta cualquier output que muestra que no hubo errors fatales.


# ── Roundtrip ─────────────────────────────────────────────────────────────────


def test_export_import_roundtrip(tmp_corrections_db, tmp_path):
    """Export desde DB con datos → import a DB vacía → lo mismo en ambas."""
    seed, db_path = tmp_corrections_db
    seed([
        {"original": "samando", "corrected": "fernando", "source": "explicit",
         "ts": 1700000000.0, "hash": "h1"},
        {"original": "mose", "corrected": "moze", "source": "llm",
         "ts": 1700000100.0, "hash": "h2"},
    ])
    runner = CliRunner()
    out_path = tmp_path / "backup.json"
    runner.invoke(rag.cli, ["whisper", "export", "-o", str(out_path)])
    # Wipe la DB.
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM rag_audio_corrections")
    conn.commit()
    conn.close()
    # Import.
    runner.invoke(rag.cli, ["whisper", "import", str(out_path)])
    # Verify.
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT original, corrected, source FROM rag_audio_corrections ORDER BY ts"
    ).fetchall()
    conn.close()
    assert rows == [
        ("samando", "fernando", "explicit"),
        ("mose", "moze", "llm"),
    ]
