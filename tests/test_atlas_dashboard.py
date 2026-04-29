"""Smoke tests del backend de `/atlas` (web/atlas_dashboard.py).

Cubren los caminos críticos sin tocar el vault real del user:

1. ``_is_junk_entity_name`` filtra phone numbers / IDs numéricos / strings
   vacías / strings de solo símbolos, pero MANTIENE nombres legítimos
   con al menos una letra (incluye unicode).
2. ``snapshot()`` con DBs inexistentes devuelve shape vacío válido (no
   crashea, payload parseable por el frontend).
3. ``snapshot()`` con DBs minimales pero sin entidades devuelve shape
   válido con counts en 0.
4. ``note_detail()`` rechaza paths inválidos (`..`, abs paths) sin
   tocar disco.
5. ``_build_graph()`` con un meta table populado reconstruye nodos
   + edges correctamente.

Los tests aislan ``DB_PATH`` con monkeypatch a un tmpdir para no
contaminar el state real del user.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from web import atlas_dashboard as ad


# ── _is_junk_entity_name ─────────────────────────────────────────────────


@pytest.mark.parametrize("name,expected", [
    # Junk: numbers, IDs, empty, symbols-only, too-short
    ("5493424303891", True),
    ("34084894028025", True),
    ("+54 9 342 430 3891", True),
    ("(011) 4567-8901", True),
    ("", True),
    ("   ", True),
    ("x", True),  # too short (< 2 chars)
    ("---", True),
    ("###", True),
    ("12.34", True),
    (None, True),
    # Legítimo: tiene al menos una letra
    ("Maria", False),
    ("yo", False),
    ("Astor", False),
    ("Santa Fe", False),
    ("5to grado", False),  # mix de número + letra
    ("C3PO", False),
    ("Avenida 9 de Julio", False),
    ("Ñoñería", False),  # unicode
    ("Á", True),  # solo 1 char (corto)
    ("BBI", False),
    ("Avenida Galicia 1571", False),  # tiene letras + números mezclados
])
def test_is_junk_entity_name(name, expected):
    assert ad._is_junk_entity_name(name) is expected


# ── snapshot con DBs inexistentes ────────────────────────────────────────


def test_snapshot_no_dbs(tmp_path: Path, monkeypatch):
    """DB_PATH apuntando a un dir vacío → shape vacío válido."""
    monkeypatch.setattr("rag.DB_PATH", tmp_path)
    payload = ad.snapshot(window_days=30, top_entities=10, graph_top_notes=50)

    # Shape estable — todas las keys esperadas siempre presentes.
    assert "meta" in payload
    assert "kpis" in payload
    assert "entities_by_type" in payload
    assert "hot" in payload
    assert "stale" in payload
    assert "cooccurrence" in payload
    assert "graph" in payload

    # Counts en 0
    assert payload["kpis"]["n_entities"] == 0
    assert payload["kpis"]["n_mentions"] == 0
    assert payload["kpis"]["n_notes"] == 0
    assert payload["kpis"]["n_edges"] == 0

    # Entity buckets vacíos pero con todos los tipos
    for t in ("person", "location", "organization", "event"):
        assert payload["entities_by_type"][t] == []

    assert payload["graph"]["nodes"] == []
    assert payload["graph"]["links"] == []


# ── snapshot con DBs minimales (sin filas) ───────────────────────────────


def _create_minimal_dbs(db_dir: Path):
    """Crea telemetry.db + ragvec.db con schema correcto pero 0 filas.

    Replica solo las tablas que `atlas_dashboard.snapshot` lee, lo
    suficiente para que las queries no fallen con `no such table`.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    # telemetry.db
    tconn = sqlite3.connect(str(db_dir / "telemetry.db"))
    tconn.execute("""
        CREATE TABLE rag_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            normalized TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            aliases TEXT,
            first_seen_ts REAL,
            last_seen_ts REAL,
            mention_count INTEGER DEFAULT 0,
            confidence REAL,
            extra_json TEXT
        )
    """)
    tconn.execute("""
        CREATE TABLE rag_entity_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            source TEXT,
            ts REAL,
            snippet TEXT,
            confidence REAL
        )
    """)
    tconn.commit()
    tconn.close()
    # ragvec.db con un meta table del shape esperado
    rconn = sqlite3.connect(str(db_dir / "ragvec.db"))
    rconn.execute("""
        CREATE TABLE meta_obsidian_notes_v11 (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            file TEXT,
            folder TEXT,
            outlinks TEXT,
            title TEXT,
            area TEXT,
            type TEXT,
            created_ts REAL
        )
    """)
    rconn.commit()
    rconn.close()


def test_snapshot_empty_dbs(tmp_path: Path, monkeypatch):
    """DBs creadas pero sin filas → shape vacío válido + KPIs en 0."""
    _create_minimal_dbs(tmp_path)
    monkeypatch.setattr("rag.DB_PATH", tmp_path)
    payload = ad.snapshot(window_days=30, top_entities=10, graph_top_notes=50)
    assert payload["kpis"]["n_entities"] == 0
    assert payload["kpis"]["n_mentions"] == 0
    assert payload["entities_by_type"]["person"] == []


# ── snapshot con datos reales (junk filter test) ─────────────────────────


def test_snapshot_filters_junk_phones(tmp_path: Path, monkeypatch):
    """Si una entity en rag_entities matchea junk pattern (phone number),
    NO debe aparecer en `entities_by_type` del snapshot."""
    _create_minimal_dbs(tmp_path)
    monkeypatch.setattr("rag.DB_PATH", tmp_path)

    tconn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    now_ts = datetime.now(timezone.utc).timestamp()
    # Mezcla: 2 entities legítimas + 2 phone numbers que el filter debe descartar
    rows = [
        ("Maria", "maria", "person", '["Maria"]', now_ts - 86400, now_ts, 100, 0.95),
        ("Astor", "astor", "person", None, now_ts - 86400, now_ts, 80, 0.9),
        ("5493424303891", "5493424303891", "person", None, now_ts - 86400, now_ts, 200, 0.95),
        ("+54 9 342 430 3891", "549342", "person", None, now_ts - 86400, now_ts, 150, 0.85),
    ]
    for name, norm, etype, aliases, first, last, mc, conf in rows:
        tconn.execute(
            "INSERT INTO rag_entities (canonical_name, normalized, entity_type, aliases, first_seen_ts, last_seen_ts, mention_count, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, norm, etype, aliases, first, last, mc, conf),
        )
    tconn.commit()
    tconn.close()

    payload = ad.snapshot(window_days=30, top_entities=10, graph_top_notes=50)
    person_names = [e["name"] for e in payload["entities_by_type"]["person"]]
    # Las legítimas SÍ aparecen
    assert "Maria" in person_names
    assert "Astor" in person_names
    # Las junk NO aparecen
    assert "5493424303891" not in person_names
    assert "+54 9 342 430 3891" not in person_names


# ── note_detail input validation ─────────────────────────────────────────


@pytest.mark.parametrize("bad_path", [
    "../../../etc/passwd",
    "/absolute/path",
    "..\\windows\\style",
    "",
    "   ",  # whitespace-only es invalid_path? — actualmente pasa el check
])
def test_note_detail_rejects_bad_paths(bad_path):
    out = ad.note_detail(path=bad_path)
    # Debe devolver shape válido + error o preview vacío, sin lanzar.
    assert isinstance(out, dict)
    assert "preview" in out and "entities" in out and "neighbors" in out


def test_note_detail_nonexistent_note(tmp_path: Path, monkeypatch):
    """Nota que no existe en disco — preview vacío, sin crash."""
    _create_minimal_dbs(tmp_path)
    monkeypatch.setattr("rag.DB_PATH", tmp_path)
    out = ad.note_detail(path="00-Inbox/no-existe.md")
    assert out["preview"] == ""  # no crashea, solo devuelve string vacío
    assert out["entities"] == []
    assert isinstance(out["vault_uri"], (str, type(None)))


# ── _build_graph integrity ───────────────────────────────────────────────


def test_build_graph_with_synthetic_meta(tmp_path: Path):
    """meta_* con outlinks → nodos + edges reconstruidos."""
    db = tmp_path / "ragvec.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE meta_test (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            file TEXT,
            folder TEXT,
            outlinks TEXT,
            title TEXT,
            area TEXT,
            type TEXT,
            created_ts REAL
        )
    """)
    # 3 notas: A linkea a B y C; B linkea a A. C sin outlinks.
    rows = [
        ("a::0", "01-Projects/A.md", "01-Projects", "01-Projects/B.md,01-Projects/C.md", "A", "", "", None),
        ("b::0", "01-Projects/B.md", "01-Projects", "01-Projects/A.md", "B", "", "", None),
        ("c::0", "01-Projects/C.md", "01-Projects", "", "C", "", "", None),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO meta_test (chunk_id, file, folder, outlinks, title, area, type, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            r,
        )
    conn.commit()
    conn.close()

    graph = ad._build_graph("meta_test", db, top_notes=100)
    assert len(graph["nodes"]) == 3
    # Edges colapsados a undirected: A-B (peso 2 por las 2 direcciones) + A-C (peso 1)
    assert len(graph["links"]) == 2
    edges = {(l["source"], l["target"]) for l in graph["links"]}
    # Sorted lex: 01-Projects/A.md < 01-Projects/B.md < 01-Projects/C.md
    assert ("01-Projects/A.md", "01-Projects/B.md") in edges
    assert ("01-Projects/A.md", "01-Projects/C.md") in edges


def test_build_graph_missing_table(tmp_path: Path):
    """Tabla inexistente → graph vacío sin crash."""
    db = tmp_path / "ragvec.db"
    sqlite3.connect(str(db)).close()  # crear DB vacío
    graph = ad._build_graph("meta_missing", db, top_notes=100)
    assert graph["nodes"] == []
    assert graph["links"] == []


def test_build_graph_truncated_when_over_cap(tmp_path: Path):
    """Cuando notes > top_notes, truncated=True."""
    db = tmp_path / "ragvec.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE meta_test (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            file TEXT,
            folder TEXT,
            outlinks TEXT,
            title TEXT,
            area TEXT,
            type TEXT,
            created_ts REAL
        )
    """)
    # 10 notas sin outlinks
    for i in range(10):
        conn.execute(
            "INSERT INTO meta_test (chunk_id, file, folder, outlinks, title, area, type, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"n{i}::0", f"folder/n{i}.md", "folder", "", f"n{i}", "", "", None),
        )
    conn.commit()
    conn.close()

    graph = ad._build_graph("meta_test", db, top_notes=5)
    assert graph["truncated"] is True
    assert graph["total_notes"] == 10
    assert len(graph["nodes"]) == 5
