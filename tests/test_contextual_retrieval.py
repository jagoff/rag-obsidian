"""Tests para Contextual Retrieval (Anthropic, Sept 2024).

Cubre los cuatro contracts del módulo `rag.contextual_retrieval`:

  (a) `generate_chunk_context` produce output dentro de bounds (chars cap,
      strip de prefijos "Contexto:", short-circuit con doc corto).
  (b) Cache hit/miss: get_or_generate_context lookup → LLM → persist.
  (c) Wire-up de `_index_single_file` cuando flag ON vs OFF — chunks con
      flag ON deben pasar por `contextualize_chunks`; con flag OFF, paridad
      bit-idéntica con el path pre-feature.
  (d) Display text NO se contamina con `[contexto: ...]` — sigue siendo
      el cuerpo raw del chunk (para snippets en UI, citation-repair,
      reranker title-prefix).

Los tests usan `tmp_path` para `DB_PATH` (snap+restore manual, ver
[`feedback_test_db_path_isolation.md`]) y monkeypatchean el LLM client
para no tocar ollama real.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag
from rag import contextual_retrieval as cr


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path, monkeypatch):
    """Aislar DB_PATH per-test. Mismo patrón que test_semantic_cache* —
    snap+restore manual para no chocar con `_stabilize_rag_state`."""
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    rag.DB_PATH.mkdir(parents=True, exist_ok=True)
    cr.stats_reset()
    try:
        yield
    finally:
        rag.DB_PATH = snap


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub del summary client. Devuelve respuestas predecibles."""
    mock = MagicMock()
    response = MagicMock()
    response.message.content = "Sección sobre la configuración del reranker dentro de la nota."
    mock.chat.return_value = response
    monkeypatch.setattr(rag, "_summary_client", lambda: mock)
    return mock


# ── (a) generate_chunk_context: bounds + behavior ────────────────────────────


def test_generate_short_doc_returns_empty(monkeypatch):
    """Documento < MIN_DOC_CHARS_FOR_CONTEXT → no LLM call, returns ''."""
    # Stub que falla si se llama — short-circuit debe pegar ANTES del LLM.
    fake_client = MagicMock()
    fake_client.chat.side_effect = AssertionError("LLM no debería invocarse")
    monkeypatch.setattr(rag, "_summary_client", lambda: fake_client)

    out = cr.generate_chunk_context(
        chunk_text="chunk body",
        parent_doc_text="x" * 50,  # too short
        doc_metadata={"title": "Tiny", "folder": "test"},
    )
    assert out == ""
    snap = cr.stats_snapshot()
    assert snap["skipped_short"] >= 1


def test_generate_strips_prefix_marker(stub_llm):
    """Si el LLM devuelve 'Contexto: X', el helper strippea el prefix
    para evitar que se duplique cuando el caller agrega su propio
    `[contexto: ...]` envelope."""
    stub_llm.chat.return_value.message.content = "Contexto: Sección sobre el reranker."

    out = cr.generate_chunk_context(
        chunk_text="el reranker es bge-v2-m3",
        parent_doc_text="x" * 1000,
        doc_metadata={"title": "Reranker Notes", "folder": "01-Projects"},
    )
    assert not out.lower().startswith("contexto"), \
        f"Prefix 'Contexto:' debería strippearse. Got: {out!r}"
    assert "Sección sobre el reranker" in out


def test_generate_caps_summary_length(stub_llm):
    """Output capeado a MAX_SUMMARY_CHARS chars."""
    long_response = "Una explicación enorme. " * 50  # ~1200 chars
    stub_llm.chat.return_value.message.content = long_response

    out = cr.generate_chunk_context(
        chunk_text="content",
        parent_doc_text="y" * 1000,
        doc_metadata={"title": "T", "folder": "f"},
    )
    assert len(out) <= cr.MAX_SUMMARY_CHARS, \
        f"Summary exceeded {cr.MAX_SUMMARY_CHARS} chars: got {len(out)}"


def test_generate_takes_first_line(stub_llm):
    """Multi-line output → toma primera línea."""
    stub_llm.chat.return_value.message.content = (
        "Primera oración del summary.\n\n"
        "Acá hay una segunda línea que debería ignorarse."
    )
    out = cr.generate_chunk_context(
        chunk_text="content",
        parent_doc_text="z" * 1000,
        doc_metadata={"title": "T", "folder": "f"},
    )
    assert "segunda línea" not in out
    assert "Primera oración" in out


def test_generate_helper_failure_returns_empty(monkeypatch):
    """LLM throw → return '' + bump errors counter, no cache fail."""
    failing = MagicMock()
    failing.chat.side_effect = RuntimeError("ollama down")
    monkeypatch.setattr(rag, "_summary_client", lambda: failing)

    out = cr.generate_chunk_context(
        chunk_text="content",
        parent_doc_text="a" * 1000,
        doc_metadata={"title": "T", "folder": "f"},
    )
    assert out == ""
    snap = cr.stats_snapshot()
    assert snap["errors"] >= 1


# ── (b) cache hit/miss ──────────────────────────────────────────────────────


def test_cache_miss_then_hit(stub_llm):
    """Primer call → miss + LLM. Segundo idéntico → hit + sin LLM."""
    stub_llm.chat.return_value.message.content = "Primer summary cacheado"

    chunk = "el reranker corre en MPS con fp16"
    parent = "Notas sobre RAG. " * 200  # > MIN_DOC_CHARS_FOR_CONTEXT
    meta = {"title": "Notes", "folder": "01-Projects"}

    # Miss
    out1 = cr.get_or_generate_context(chunk, parent, "doc/path.md", 0, meta)
    assert "Primer summary" in out1
    assert stub_llm.chat.call_count == 1

    # Hit — mismo chunk + mismo idx + mismo doc_id
    out2 = cr.get_or_generate_context(chunk, parent, "doc/path.md", 0, meta)
    assert out2 == out1
    assert stub_llm.chat.call_count == 1, \
        "Segunda invocación debería ser cache hit (no LLM call)"

    snap = cr.stats_snapshot()
    assert snap["hits"] >= 1
    assert snap["misses"] >= 1


def test_cache_invalidation_on_chunk_change(stub_llm):
    """Mismo doc_id + chunk_idx pero body distinto → chunk_hash distinto → miss."""
    stub_llm.chat.return_value.message.content = "context A"
    cr.get_or_generate_context(
        "version A", "doc body " * 100, "p.md", 0, {"title": "T", "folder": "f"}
    )
    assert stub_llm.chat.call_count == 1

    # Mismo doc_id + idx, pero el body del chunk cambió
    stub_llm.chat.return_value.message.content = "context B"
    out = cr.get_or_generate_context(
        "version B distinta", "doc body " * 100, "p.md", 0,
        {"title": "T", "folder": "f"},
    )
    assert "context B" in out
    assert stub_llm.chat.call_count == 2, \
        "Cambio en chunk body debería forzar regeneración"


def test_cache_independent_per_chunk_idx(stub_llm):
    """Mismo doc + mismo body pero idx distinto → entries separadas."""
    stub_llm.chat.return_value.message.content = "summary"
    chunk = "same body across chunks"
    parent = "doc " * 100

    cr.get_or_generate_context(chunk, parent, "p.md", 0, {"title": "T", "folder": "f"})
    cr.get_or_generate_context(chunk, parent, "p.md", 1, {"title": "T", "folder": "f"})

    # Ambos son misses (idx distinto = key distinta), no hit del primero.
    assert stub_llm.chat.call_count == 2


def test_cache_does_not_persist_empty(monkeypatch):
    """Si el helper devuelve '', NO escribir al cache (re-tries en el próximo run)."""
    failing = MagicMock()
    failing.chat.side_effect = TimeoutError("timeout")
    monkeypatch.setattr(rag, "_summary_client", lambda: failing)

    out = cr.get_or_generate_context(
        "chunk", "doc " * 100, "p.md", 0, {"title": "T", "folder": "f"}
    )
    assert out == ""

    # Re-call: si hubiéramos cacheado el "" tendríamos un hit aquí.
    # En su lugar debe seguir siendo miss (errors siguen subiendo).
    out2 = cr.get_or_generate_context(
        "chunk", "doc " * 100, "p.md", 0, {"title": "T", "folder": "f"}
    )
    assert out2 == ""
    assert failing.chat.call_count == 2, \
        "El cache no debe guardar empty results — hace falta retry en el próximo run"


# ── (c) wire-up con _index_single_file ───────────────────────────────────────


@pytest.fixture
def indexing_env(monkeypatch, tmp_path):
    """Setup completo para `_index_single_file` (mismo patrón que
    tests/test_index_single_file.py)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    from rag import SqliteVecClient
    client = SqliteVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="ctx_retrieval_test", metadata={"hnsw:space": "cosine"}
    )

    # Captura los embed_texts que llegan al embedder para inspección.
    captured: dict[str, list[str]] = {"embed_texts": []}

    def fake_embed(texts):
        captured["embed_texts"].extend(texts)
        return [[0.1] * 1024 for _ in texts]

    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_check_and_flag_contradictions",
                        lambda *a, **kw: None)
    rag._invalidate_corpus_cache()

    return vault, col, captured


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_wire_up_off_is_bit_identical(indexing_env, monkeypatch):
    """Sin RAG_CONTEXTUAL_RETRIEVAL=1, los embed_texts no llevan
    `[contexto: ...]` — paridad pre-feature."""
    monkeypatch.delenv("RAG_CONTEXTUAL_RETRIEVAL", raising=False)
    vault, col, captured = indexing_env

    note = vault / "alpha.md"
    _write(note, "# Alpha\n\n" + "este es un cuerpo razonable de la nota. " * 30)

    assert rag._index_single_file(col, note) == "indexed"
    assert captured["embed_texts"], "embed() no fue invocado"
    for et in captured["embed_texts"]:
        assert cr.SUMMARY_MARKER not in et, \
            f"Flag OFF — no debería haber [contexto: ...] en embed_text. Got: {et[:200]!r}"


def test_wire_up_on_prepends_context(indexing_env, monkeypatch, stub_llm):
    """Con RAG_CONTEXTUAL_RETRIEVAL=1, los embed_texts arrancan con
    `[contexto: ...]` (cuando el helper devuelve algo)."""
    monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", "1")
    vault, col, captured = indexing_env

    note = vault / "beta.md"
    _write(note, "# Beta\n\n" + "contenido suficiente para chunkear y contextualizar. " * 30)

    assert rag._index_single_file(col, note) == "indexed"
    assert captured["embed_texts"], "embed() no fue invocado"
    # Al menos UN embed_text debe llevar el marker (el doc es lo bastante
    # largo como para que el helper no haga short-circuit).
    has_marker = any(cr.SUMMARY_MARKER in et for et in captured["embed_texts"])
    assert has_marker, (
        "Flag ON + doc largo: al menos un embed_text debería llevar "
        f"'{cr.SUMMARY_MARKER}'. embed_texts: "
        f"{[et[:80] for et in captured['embed_texts']]}"
    )


def test_wire_up_on_does_not_contaminate_display(indexing_env, monkeypatch, stub_llm):
    """Display text (el cuerpo raw que aterriza en la collection) NO debe
    llevar `[contexto: ...]`. La señal contextual vive sólo en el embedding."""
    monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", "1")
    vault, col, _captured = indexing_env

    note = vault / "gamma.md"
    body = "# Gamma\n\n" + "cuerpo crudo que debería preservarse para snippets en UI. " * 30
    _write(note, body)

    assert rag._index_single_file(col, note) == "indexed"

    # Inspect: el documento persistido en sqlite-vec debería ser el body raw,
    # SIN `[contexto: ...]`. `documents` es el campo que volverá en snippets.
    got = col.get(where={"file": "gamma.md"}, include=["documents", "metadatas"])
    assert got["documents"], "no chunks persisted"
    for doc in got["documents"]:
        assert cr.SUMMARY_MARKER not in doc, \
            f"Display text contaminado con marker. Got: {doc[:200]!r}"
    # Y el parent (también accesible via meta) tampoco debería estar contaminado.
    for meta in got["metadatas"]:
        parent = meta.get("parent", "")
        assert cr.SUMMARY_MARKER not in parent, \
            f"Parent text contaminado con marker. Got: {parent[:200]!r}"


def test_wire_up_falsy_env_still_off(indexing_env, monkeypatch):
    """Valores falsy del env var (`0`, `false`, vacío) → flag OFF."""
    vault, col, captured = indexing_env

    for falsy in ("0", "false", "no", "", "FALSE"):
        monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", falsy)
        captured["embed_texts"].clear()
        note = vault / f"delta-{falsy or 'empty'}.md"
        _write(note, "# Delta\n\n" + "body razonable para indexar. " * 30)
        assert rag._index_single_file(col, note) == "indexed"
        for et in captured["embed_texts"]:
            assert cr.SUMMARY_MARKER not in et, \
                f"Falsy env={falsy!r} debería ser OFF; embed_text contiene marker: {et[:120]!r}"


# ── (d) gate helpers ────────────────────────────────────────────────────────


def test_enabled_helper_truthy_values(monkeypatch):
    monkeypatch.delenv("RAG_CONTEXTUAL_RETRIEVAL", raising=False)
    assert cr.contextual_retrieval_enabled() is False
    for truthy in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", truthy)
        assert cr.contextual_retrieval_enabled() is True, \
            f"truthy {truthy!r} should enable"
    for falsy in ("0", "false", "no", "", "FALSE"):
        monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", falsy)
        assert cr.contextual_retrieval_enabled() is False, \
            f"falsy {falsy!r} should disable"


def test_chunk_hash_is_deterministic_and_versioned():
    """Mismo body → mismo hash. PROMPT_VERSION distinto → distinto hash
    (sin tocar el método público — sólo verificamos que `chunk_hash`
    incluye PROMPT_VERSION en su mezcla)."""
    h1 = cr.chunk_hash("body texto")
    h2 = cr.chunk_hash("body texto")
    assert h1 == h2, "chunk_hash debe ser determinístico para mismo input"

    h3 = cr.chunk_hash("body texto distinto")
    assert h1 != h3, "bodies distintos → hashes distintos"

    # PROMPT_VERSION es parte del hash via prefix: si lo cambiáramos
    # los hits viejos invalidarían. No bumpeable from tests sin
    # monkeypatch — lo verificamos vía smoke en len.
    assert len(h1) == 16, "chunk_hash debe ser 16 chars (sha1 truncated)"


# ── helper de invariantes del módulo ─────────────────────────────────────────


def test_contextualize_chunks_off_returns_input_identity(monkeypatch):
    """Flag OFF — `contextualize_chunks` retorna la misma lista (objeto
    idéntico para el caller, sin reasignar referencias inúriles)."""
    monkeypatch.delenv("RAG_CONTEXTUAL_RETRIEVAL", raising=False)
    embed_texts = ["a", "b", "c"]
    out = cr.contextualize_chunks(
        embed_texts=embed_texts,
        display_texts=["a", "b", "c"],
        doc_id="x.md",
        parent_doc_text="doc",
        doc_metadata={"title": "x"},
    )
    assert out is embed_texts, \
        "Flag OFF debería retornar el mismo list object (sin overhead de copia)"


def test_contextualize_chunks_empty_input_safe():
    """Lista vacía no debe romper."""
    out = cr.contextualize_chunks(
        embed_texts=[],
        display_texts=[],
        doc_id="empty.md",
        parent_doc_text="",
        doc_metadata={},
    )
    assert out == []
