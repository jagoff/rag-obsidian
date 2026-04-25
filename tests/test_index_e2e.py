"""End-to-end indexing test. Builds a tmp vault with 5 Spanish notes and
drives `_index_single_file` against the real ollama+bge-m3 stack (no
embed mock). Verifies:
  - All 5 notes produce chunks (count >= 5).
  - Every chunk has the required metadata keys (file, folder, note).
  - A second pass is a no-op (hash match → 'skipped') on EVERY file.

Gate is `RAG_RUN_E2E=1` — same rationale as test_retrieve_e2e.py.
Run it:

    RAG_RUN_E2E=1 .venv/bin/python -m pytest tests/test_index_e2e.py -q
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAG_MEMORY_PRESSURE_DISABLE", "1")

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RAG_RUN_E2E"),
        reason=(
            "End-to-end index test — requires ollama + bge-m3 pulled. "
            "Opt in with RAG_RUN_E2E=1."
        ),
    ),
    pytest.mark.slow,
]


_NOTES = [
    ("01-projects/alpha.md",
     "# Proyecto Alpha\n\nEste es el proyecto principal de ingeniería.\n"
     "Está en fase de diseño inicial, con tres subtareas mayores."),
    ("01-projects/beta.md",
     "# Proyecto Beta\n\nProyecto secundario, de soporte al cliente.\n"
     "Bloqueado hasta que alpha cierre su primera review."),
    ("02-areas/finanzas.md",
     "# Finanzas\n\nSeguimiento mensual de ingresos y gastos, presupuesto "
     "en curso y categorías de inversión."),
    ("02-areas/salud.md",
     "# Salud\n\nNotas sobre rutinas, suplementos, métricas de sueño "
     "y objetivos trimestrales."),
    ("03-resources/rust-cheatsheet.md",
     "# Rust Cheatsheet\n\nReferencia rápida de sintaxis: ownership, "
     "borrowing, lifetimes, traits, error handling."),
]


@pytest.fixture
def e2e_indexing_env(tmp_path, monkeypatch):
    """Real ollama embed; stub out helper-model enrichments (summary +
    synthetic questions) to keep wall-time bounded. Contradict checker is
    disabled by skip_contradict=True on the indexing call itself."""
    import rag

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="index_e2e_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_check_and_flag_contradictions",
                        lambda *a, **kw: None)
    rag._invalidate_corpus_cache()

    for rel, body in _NOTES:
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return vault, col


def test_index_e2e_all_notes_produce_chunks(e2e_indexing_env):
    """5 notes → >= 5 chunks (chunks might split further on long notes;
    these are short so usually 1 chunk each)."""
    import rag
    vault, col = e2e_indexing_env

    for rel, _body in _NOTES:
        status = rag._index_single_file(col, vault / rel, skip_contradict=True)
        assert status == "indexed", f"{rel}: expected 'indexed', got {status}"

    n = col.count()
    assert n >= len(_NOTES), (
        f"expected >= {len(_NOTES)} chunks (one per note at minimum), got {n}"
    )
    # Upper bound is soft — the chunker merges small pieces. Anything >3×
    # is a sign the chunk boundaries regressed (MIN_CHUNK / MAX_CHUNK drift).
    assert n <= len(_NOTES) * 3, (
        f"got {n} chunks for {len(_NOTES)} short notes — chunk boundaries "
        f"may have regressed (check MIN_CHUNK / MAX_CHUNK in semantic_chunks)."
    )


def test_index_e2e_metadata_shape(e2e_indexing_env):
    """Every chunk must carry file / folder / note. These are the fields
    retrieve() + the LLM prompt header depend on (`_format_chunk_for_llm`
    builds `[nota: {note}] [ruta: {file}]` — drop one and the path-
    extraction + citation-repair rules break silently)."""
    import rag
    vault, col = e2e_indexing_env

    for rel, _body in _NOTES:
        rag._index_single_file(col, vault / rel, skip_contradict=True)

    data = col.get(include=["metadatas"])
    assert data["metadatas"], "collection empty after indexing"
    for meta in data["metadatas"]:
        assert meta.get("file"), f"chunk missing 'file': {meta!r}"
        assert "folder" in meta, f"chunk missing 'folder' key: {meta!r}"
        assert meta.get("note"), f"chunk missing 'note': {meta!r}"


def test_index_e2e_second_pass_is_noop(e2e_indexing_env):
    """Hash-match short-circuit: re-indexing unchanged files returns
    'skipped' for every single one. Regressing this means every `rag
    watch` save re-embeds the whole note (~150ms/note Ollama call),
    killing the incremental indexing contract."""
    import rag
    vault, col = e2e_indexing_env

    # Pass 1
    for rel, _body in _NOTES:
        status = rag._index_single_file(col, vault / rel, skip_contradict=True)
        assert status == "indexed", f"pass 1 {rel}: expected indexed, got {status}"
    count_after_pass1 = col.count()

    # Pass 2 — untouched files
    for rel, _body in _NOTES:
        status = rag._index_single_file(col, vault / rel, skip_contradict=True)
        assert status == "skipped", (
            f"pass 2 {rel}: expected 'skipped' (hash unchanged), got {status!r} "
            "— hash short-circuit broken, re-embeds on every save"
        )
    # Chunk count must not drift.
    assert col.count() == count_after_pass1, (
        f"second-pass re-embedded chunks: {count_after_pass1} → {col.count()}"
    )
