"""Tests del cross-source synthetic generation (Quick Win #4, 2026-04-29).

Cubre:
- Lectura de items desde `meta_obsidian_notes_v11` filtrando por `source`.
- Skip de items con body insuficiente (< MIN_DOC_CHARS_CROSS_SOURCE).
- Idempotencia: re-runs sobre el mismo (file_uri, content_hash) no
  duplican.
- Source mode validation: pasa "vault" → ValueError.
- Source mode validation: pasa "unknown" → ValueError.
- Persistencia con file URI como note_path (gmail://thread/X, etc).
- LLM fallback (silent skip cuando el call falla).
- Stats per-source: by_source distribution incluye cross-source.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_ranker_lgbm.synthetic_queries import (
    CROSS_SOURCE_SOURCES,
    MIN_DOC_CHARS_CROSS_SOURCE,
    _iter_cross_source_items,
    generate_synthetic_queries_for_cross_source,
    get_synthetic_stats,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def telemetry_conn() -> sqlite3.Connection:
    """Connection en memoria para `rag_synthetic_queries`."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    yield c
    c.close()


@pytest.fixture
def state_conn() -> sqlite3.Connection:
    """Connection en memoria con schema de `meta_obsidian_notes_v11` mínimo
    (solo las columnas que usa el lector cross-source)."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute(
        """
        CREATE TABLE meta_obsidian_notes_v11 (
            chunk_id TEXT,
            file TEXT,
            title TEXT,
            document TEXT,
            source TEXT
        )
        """
    )
    yield c
    c.close()


def _insert_item(
    conn: sqlite3.Connection,
    *,
    file_uri: str,
    title: str,
    document: str,
    source: str,
    chunk_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO meta_obsidian_notes_v11 (chunk_id, file, title, document, source) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id or f"{file_uri}#0", file_uri, title, document, source),
    )


def _stub_llm(prompt: str, *, model: str = "test") -> str:
    """Devuelve queries plausibles para que el parser las acepte."""
    import json as _json
    return _json.dumps({
        "queries": [
            {"q": "qué dijo Juan sobre el proyecto", "kind": "factual"},
            {"q": "info del cliente", "kind": "exploratory"},
            {"q": "che, qué onda con esto", "kind": "conversational"},
            {"q": "cuándo es la reunión", "kind": "factual"},
        ]
    })


# ─── _iter_cross_source_items ──────────────────────────────────────────


def test_iter_cross_source_filters_by_source(state_conn):
    _insert_item(
        state_conn,
        file_uri="gmail://thread/abc",
        title="Re: meeting",
        document="hola team, revisamos el proyecto",
        source="gmail",
    )
    _insert_item(
        state_conn,
        file_uri="01-Projects/RAG/note.md",
        title="RAG note",
        document="local rag over obsidian vault",
        source="vault",
    )

    items = _iter_cross_source_items(state_conn, "gmail")
    assert len(items) == 1
    assert items[0][0] == "gmail://thread/abc"

    items_vault = _iter_cross_source_items(state_conn, "vault")
    # Esta función está pensada para cross-source pero soporta vault
    # también (no filtra por scheme — solo por columna `source`).
    assert len(items_vault) == 1


def test_iter_cross_source_dedup_chunks_by_file(state_conn):
    """Múltiples chunks del mismo file → un solo row (el más largo)."""
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="email",
        document="chunk A short",
        source="gmail",
        chunk_id="X#0",
    )
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="email",
        document="chunk B is much longer body content here for testing dedup",
        source="gmail",
        chunk_id="X#1",
    )

    items = _iter_cross_source_items(state_conn, "gmail")
    assert len(items) == 1
    # ORDER BY file ASC, length(document) DESC → primer row tiene el chunk
    # más largo. Dedup keeps that one.
    assert "longer body" in items[0][2]


def test_iter_cross_source_missing_table_returns_empty():
    """Si la tabla no existe, devuelve [] sin raise."""
    c = sqlite3.connect(":memory:")
    items = _iter_cross_source_items(c, "gmail")
    assert items == []
    c.close()


def test_iter_cross_source_ignores_empty_file(state_conn):
    """Rows con file=NULL o file='' se filtran en SQL."""
    state_conn.execute(
        "INSERT INTO meta_obsidian_notes_v11 (chunk_id, file, document, source) "
        "VALUES (?, NULL, ?, ?)",
        ("c0", "should not appear", "gmail"),
    )
    state_conn.execute(
        "INSERT INTO meta_obsidian_notes_v11 (chunk_id, file, document, source) "
        "VALUES (?, '', ?, ?)",
        ("c1", "neither this", "gmail"),
    )
    items = _iter_cross_source_items(state_conn, "gmail")
    assert items == []


# ─── generate_synthetic_queries_for_cross_source — happy path ──────────


def test_cross_source_happy_path_inserts_pairs(telemetry_conn, state_conn):
    _insert_item(
        state_conn,
        file_uri="gmail://thread/abc",
        title="proyecto X status",
        document=(
            "El status del proyecto X es: estamos avanzando con la fase 2, "
            "pendiente reunión con cliente. Próxima entrega 15/05."
        ),
        source="gmail",
    )

    result = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=_stub_llm,
    )

    assert result["source"] == "gmail"
    assert result["n_notes_seen"] == 1
    assert result["n_notes_processed"] == 1
    # 4 queries del stub LLM, default queries_per_note=4 → todas inserted.
    assert result["n_queries_inserted"] == 4
    assert result["n_pairs_total"] == 4

    # Verify persistence: rows en rag_synthetic_queries con file URI.
    rows = telemetry_conn.execute(
        "SELECT note_path, query, query_kind FROM rag_synthetic_queries"
    ).fetchall()
    assert len(rows) == 4
    for note_path, _query, _kind in rows:
        assert note_path == "gmail://thread/abc"


def test_cross_source_skip_short_body(telemetry_conn, state_conn):
    """Items con body < MIN_DOC_CHARS_CROSS_SOURCE skipean sin llamar LLM."""
    _insert_item(
        state_conn,
        file_uri="whatsapp://msg/short",
        title="",
        document="ok",  # 2 chars, way below threshold (50)
        source="whatsapp",
    )

    llm_calls = []

    def tracking_llm(prompt, *, model="test"):
        llm_calls.append(prompt)
        return _stub_llm(prompt, model=model)

    result = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="whatsapp",
        state_conn=state_conn,
        llm_call=tracking_llm,
    )

    assert result["n_notes_skipped_empty"] == 1
    assert result["n_notes_processed"] == 0
    assert len(llm_calls) == 0  # LLM no se invocó


def test_cross_source_idempotent(telemetry_conn, state_conn):
    """Re-run sobre el mismo content_hash skipea con `n_notes_skipped_unchanged`."""
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="status",
        document="A" * 100,  # body suficiente
        source="gmail",
    )

    result1 = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=_stub_llm,
    )
    assert result1["n_notes_processed"] == 1
    assert result1["n_queries_inserted"] >= 1

    # Re-run mismo input → unchanged (mismo hash).
    result2 = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=_stub_llm,
    )
    assert result2["n_notes_skipped_unchanged"] == 1
    assert result2["n_notes_processed"] == 0


def test_cross_source_dry_run_no_persist(telemetry_conn, state_conn):
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="t",
        document="A" * 100,
        source="gmail",
    )

    result = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=_stub_llm,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["n_pairs_total"] > 0  # los reporta como pares "que se hubieran" insertado
    rows = telemetry_conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_queries"
    ).fetchone()
    assert rows[0] == 0  # nada persistido


def test_cross_source_llm_failure_silent_skip(telemetry_conn, state_conn):
    """LLM raising exception → metric n_notes_llm_failed, no insert."""
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="t",
        document="A" * 100,
        source="gmail",
    )

    def broken_llm(prompt, *, model="test"):
        raise RuntimeError("ollama down")

    result = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=broken_llm,
    )
    assert result["n_notes_llm_failed"] == 1
    assert result["n_queries_inserted"] == 0


def test_cross_source_llm_returns_invalid_json(telemetry_conn, state_conn):
    """JSON inválido del LLM → bucket `n_notes_llm_failed`."""
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="t",
        document="A" * 100,
        source="gmail",
    )

    def garbage_llm(prompt, *, model="test"):
        return "not even close to json"

    result = generate_synthetic_queries_for_cross_source(
        telemetry_conn,
        source="gmail",
        state_conn=state_conn,
        llm_call=garbage_llm,
    )
    assert result["n_notes_llm_failed"] == 1


# ─── source validation ────────────────────────────────────────────────


def test_cross_source_rejects_vault(telemetry_conn, state_conn):
    """source='vault' raise — usá la otra función."""
    with pytest.raises(ValueError, match="vault"):
        generate_synthetic_queries_for_cross_source(
            telemetry_conn,
            source="vault",
            state_conn=state_conn,
        )


def test_cross_source_rejects_unknown(telemetry_conn, state_conn):
    with pytest.raises(ValueError, match="cross-source"):
        generate_synthetic_queries_for_cross_source(
            telemetry_conn,
            source="totally_made_up",
            state_conn=state_conn,
        )


def test_cross_source_accepts_all_known_sources():
    """Todos los valores de CROSS_SOURCE_SOURCES son válidos."""
    for src in CROSS_SOURCE_SOURCES:
        # Smoke: con fixtures vacías, debe retornar n_notes_seen=0 sin error.
        c = sqlite3.connect(":memory:")
        sc = sqlite3.connect(":memory:")
        sc.execute(
            "CREATE TABLE meta_obsidian_notes_v11 (file TEXT, title TEXT, "
            "document TEXT, source TEXT)"
        )
        result = generate_synthetic_queries_for_cross_source(
            c, source=src, state_conn=sc, llm_call=_stub_llm,
        )
        assert result["source"] == src
        assert result["n_notes_seen"] == 0
        c.close()
        sc.close()


# ─── stats with by_source ──────────────────────────────────────────────


def test_stats_by_source_distribution(telemetry_conn, state_conn):
    """Después de generar para 2 sources, stats reportan by_source."""
    # Vault note (path sin scheme → "vault" via heuristic).
    _insert_item(
        state_conn,
        file_uri="01-Projects/RAG/note.md",
        title="t",
        document="A" * 100,
        source="vault",
    )
    _insert_item(
        state_conn,
        file_uri="gmail://thread/X",
        title="t",
        document="B" * 100,
        source="gmail",
    )
    _insert_item(
        state_conn,
        file_uri="whatsapp://msg/Y",
        title="t",
        document="C" * 100,
        source="whatsapp",
    )

    # Generar para 2 cross-sources distintas.
    for src in ("gmail", "whatsapp"):
        generate_synthetic_queries_for_cross_source(
            telemetry_conn,
            source=src,
            state_conn=state_conn,
            llm_call=_stub_llm,
        )

    stats = get_synthetic_stats(telemetry_conn)
    by_source = stats["by_source"]
    assert "gmail" in by_source
    assert "whatsapp" in by_source
    assert by_source["gmail"] >= 1
    assert by_source["whatsapp"] >= 1
    # No vault porque no llamamos generate_synthetic_queries (FS-based) acá.
    assert "vault" not in by_source or by_source.get("vault", 0) == 0


def test_stats_by_source_normalizes_aliases(telemetry_conn):
    """`gdrive://` debería contar como 'drive', `wa://` como 'whatsapp'."""
    # Seedeamos directamente el row para cubrir el pathway de stats sin
    # tener que correr la función de generación.
    telemetry_conn.execute(
        """
        CREATE TABLE rag_synthetic_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            note_path TEXT NOT NULL,
            note_hash TEXT NOT NULL,
            query TEXT NOT NULL,
            query_kind TEXT,
            gen_model TEXT,
            gen_meta_json TEXT,
            UNIQUE(note_path, query)
        )
        """
    )
    for note_path in ("gdrive://file/1", "wa://msg/2", "calendar://e/3"):
        telemetry_conn.execute(
            "INSERT INTO rag_synthetic_queries (ts, note_path, note_hash, query, query_kind) "
            "VALUES ('2026-04-29', ?, 'h', ?, 'k')",
            (note_path, f"q-{note_path}"),
        )
    stats = get_synthetic_stats(telemetry_conn)
    by_source = stats["by_source"]
    assert by_source.get("drive") == 1  # gdrive normalized to drive
    assert by_source.get("whatsapp") == 1  # wa normalized to whatsapp
    assert by_source.get("calendar") == 1


# ─── min chars threshold sanity check ──────────────────────────────────


def test_min_doc_chars_constant_sanity():
    """El threshold debe estar en un valor razonable (no 0, no astronómico)."""
    assert 10 <= MIN_DOC_CHARS_CROSS_SOURCE <= 500
