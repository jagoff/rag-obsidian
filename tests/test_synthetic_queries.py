"""Tests para `rag_ranker_lgbm.synthetic_queries` y `hard_negatives`.

LLM y embedding mockeados — no llaman a ollama / sqlite-vec real.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rag_ranker_lgbm.hard_negatives import (
    DEFAULT_NEGATIVES_PER_QUERY,
    get_negatives_stats,
    mine_hard_negatives_for_synthetic,
)
from rag_ranker_lgbm.synthetic_queries import (
    DEFAULT_QUERIES_PER_NOTE,
    _content_hash,
    _parse_generation_response,
    _strip_frontmatter,
    _truncate_body,
    generate_synthetic_queries,
    get_synthetic_stats,
)


# ── Helpers puros ───────────────────────────────────────────────────────────


def test_strip_frontmatter():
    note_with = "---\nfoo: bar\n---\n# Title\nBody text"
    assert _strip_frontmatter(note_with) == "# Title\nBody text"

    note_without = "# Title\nBody"
    assert _strip_frontmatter(note_without) == "# Title\nBody"

    note_unclosed = "---\nfoo: bar\nno closing"
    # Sin cierre, devuelve el raw (no rompe).
    assert _strip_frontmatter(note_unclosed) == note_unclosed


def test_truncate_body_keeps_short_text():
    text = "short content"
    assert _truncate_body(text, max_chars=100) == text


def test_truncate_body_cuts_at_paragraph_break():
    text = "para1\n\npara2\n\npara3 long enough to exceed limit"
    truncated = _truncate_body(text, max_chars=15)
    # Cuts at the second \n\n (which is past the limit but is the last
    # break before max_chars).
    assert truncated.endswith("para1") or truncated.endswith("para2")


def test_content_hash_is_deterministic():
    assert _content_hash("foo") == _content_hash("foo")
    assert _content_hash("foo") != _content_hash("bar")
    assert len(_content_hash("foo")) == 16


# ── Parse LLM response ──────────────────────────────────────────────────────


def test_parse_response_valid_json():
    raw = json.dumps({
        "queries": [
            {"q": "qué es ikigai", "kind": "factual"},
            {"q": "info sobre productividad", "kind": "exploratory"},
        ]
    })
    result = _parse_generation_response(raw)
    assert len(result) == 2
    assert result[0]["q"] == "qué es ikigai"
    assert result[0]["kind"] == "factual"


def test_parse_response_handles_malformed_gracefully():
    assert _parse_generation_response("") == []
    assert _parse_generation_response("not json") == []
    assert _parse_generation_response("[]") == []  # not a dict
    assert _parse_generation_response('{"queries": "not list"}') == []


def test_parse_response_skips_invalid_items():
    raw = json.dumps({
        "queries": [
            {"q": "valid"},
            "not a dict",
            {"no_q_field": "bad"},
            {"q": ""},  # empty string — skipped
            {"q": "another valid"},
        ]
    })
    result = _parse_generation_response(raw)
    assert len(result) == 2
    assert result[0]["q"] == "valid"
    assert result[1]["q"] == "another valid"


def test_parse_response_alternative_field_name():
    """Algunos LLMs devuelven 'query' en lugar de 'q'."""
    raw = json.dumps({
        "queries": [{"query": "alt field", "kind": "factual"}]
    })
    result = _parse_generation_response(raw)
    assert len(result) == 1
    assert result[0]["q"] == "alt field"


# ── End-to-end: generate_synthetic_queries ──────────────────────────────────


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    yield c
    c.close()


@pytest.fixture
def fake_vault(tmp_path: Path) -> Path:
    """Crea un vault fake con 3 notas."""
    (tmp_path / "01-Projects").mkdir()
    (tmp_path / "00-Inbox").mkdir()

    (tmp_path / "01-Projects" / "ikigai.md").write_text(
        "---\ntags: [filosofia]\n---\n# Ikigai\n\nConcepto japonés sobre propósito."
    )
    (tmp_path / "01-Projects" / "productividad.md").write_text(
        "# Productividad\n\nSistema GTD modificado."
    )
    (tmp_path / "00-Inbox" / "captura-rapida.md").write_text(
        "# Captura\n\nNota chica del inbox."
    )
    return tmp_path


def test_generate_synthetic_queries_e2e(fake_vault, conn):
    """Test e2e con LLM mockeado — verifica el flow + persistencia."""
    call_count = {"n": 0}

    def fake_llm(prompt, *, model):
        call_count["n"] += 1
        # Retorna queries genéricas distintas por nota (basado en path).
        return json.dumps({
            "queries": [
                {"q": f"query-{call_count['n']}-A", "kind": "factual"},
                {"q": f"query-{call_count['n']}-B", "kind": "exploratory"},
                {"q": f"query-{call_count['n']}-C", "kind": "conversational"},
            ]
        })

    result = generate_synthetic_queries(
        conn,
        vault=fake_vault,
        queries_per_note=3,
        llm_call=fake_llm,
        is_excluded_fn=lambda p: False,
    )

    assert result["n_notes_seen"] == 3
    assert result["n_notes_processed"] == 3
    assert result["n_notes_skipped_empty"] == 0
    assert result["n_queries_inserted"] == 9  # 3 notas × 3 queries
    assert result["n_pairs_total"] == 9


def test_generate_idempotent_on_unchanged_notes(fake_vault, conn):
    """Re-correr sobre las mismas notas (mismo hash) NO re-genera."""
    def fake_llm(prompt, *, model):
        return json.dumps({"queries": [{"q": "x", "kind": "factual"}]})

    first = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )
    second = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )

    assert first["n_notes_processed"] == 3
    assert second["n_notes_processed"] == 0
    assert second["n_notes_skipped_unchanged"] == 3


def test_generate_re_runs_when_note_content_changes(fake_vault, conn):
    """Si el contenido de una nota cambia, su nuevo hash dispara re-generation."""
    def fake_llm(prompt, *, model):
        return json.dumps({"queries": [{"q": "x", "kind": "factual"}]})

    generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )

    # Modify one note → new hash.
    (fake_vault / "01-Projects" / "ikigai.md").write_text("# CHANGED\nNew content")

    second = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )
    # Solo la modificada se re-procesa; las otras 2 quedan unchanged.
    assert second["n_notes_processed"] == 1
    assert second["n_notes_skipped_unchanged"] == 2


def test_generate_dry_run_does_not_persist(fake_vault, conn):
    def fake_llm(prompt, *, model):
        return json.dumps({"queries": [{"q": "x", "kind": "factual"}]})

    result = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        dry_run=True,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )
    assert result["n_pairs_total"] == 3  # generated
    assert result["n_queries_inserted"] == 0  # NOT persisted

    # DB vacía.
    n = conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_queries"
    ).fetchone()[0]
    assert n == 0


def test_generate_skips_excluded_paths(fake_vault, conn):
    def fake_llm(prompt, *, model):
        return json.dumps({"queries": [{"q": "x", "kind": "factual"}]})

    # Exclude everything in 00-Inbox.
    def custom_exclude(rel_path: str) -> bool:
        return rel_path.startswith("00-Inbox/")

    result = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=fake_llm, is_excluded_fn=custom_exclude,
    )

    assert result["n_notes_seen"] == 2  # 01-Projects, no 00-Inbox
    assert result["n_notes_processed"] == 2


def test_generate_handles_llm_failure(fake_vault, conn):
    """Si el LLM lanza para una nota, la skipea pero sigue con las otras."""
    call_count = {"n": 0}

    def flaky_llm(prompt, *, model):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("ollama timeout")
        return json.dumps({"queries": [{"q": f"q{call_count['n']}", "kind": "factual"}]})

    result = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1,
        llm_call=flaky_llm, is_excluded_fn=lambda p: False,
    )
    assert result["n_notes_processed"] == 2
    assert result["n_notes_llm_failed"] == 1


def test_generate_limit_caps_notes(fake_vault, conn):
    def fake_llm(prompt, *, model):
        return json.dumps({"queries": [{"q": "x", "kind": "factual"}]})

    result = generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=1, limit=2,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )
    assert result["n_notes_seen"] == 2
    assert result["n_notes_processed"] == 2


def test_generate_default_constants():
    assert DEFAULT_QUERIES_PER_NOTE == 4


# ── stats ───────────────────────────────────────────────────────────────────


def test_synthetic_stats(fake_vault, conn):
    def fake_llm(prompt, *, model):
        return json.dumps({
            "queries": [
                {"q": "q1", "kind": "factual"},
                {"q": "q2", "kind": "exploratory"},
            ]
        })

    generate_synthetic_queries(
        conn, vault=fake_vault, queries_per_note=2,
        llm_call=fake_llm, is_excluded_fn=lambda p: False,
    )

    stats = get_synthetic_stats(conn)
    assert stats["n_total"] == 6  # 3 notas × 2 queries
    assert stats["n_unique_notes"] == 3
    assert stats["by_kind"] == {"factual": 3, "exploratory": 3}


# ── Hard negative mining ────────────────────────────────────────────────────


@pytest.fixture
def conn_with_synthetics():
    """Connection con synthetic queries pobladas para mining."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript("""
        CREATE TABLE rag_synthetic_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, note_path TEXT, note_hash TEXT, query TEXT,
            query_kind TEXT, gen_model TEXT, gen_meta_json TEXT,
            UNIQUE(note_path, query)
        );
    """)
    c.execute(
        "INSERT INTO rag_synthetic_queries (ts, note_path, note_hash, query) "
        "VALUES ('2026-04-26', 'a.md', 'h1', 'qué es ikigai')"
    )
    c.execute(
        "INSERT INTO rag_synthetic_queries (ts, note_path, note_hash, query) "
        "VALUES ('2026-04-26', 'b.md', 'h2', 'info productividad')"
    )
    yield c
    c.close()


def test_mine_negatives_e2e(conn_with_synthetics):
    """Test e2e con embed + NN mockeados.

    El fake NN retorna el mismo top-5 para cualquier query. Para la
    primera query (positive=a.md), el top-1 (a.md) se filtra como self.
    Para la segunda (positive=b.md), no hay match con su positive en
    los NN, así que no hay self-filter. Por eso n_filtered_self=1
    (solo la primera).
    """
    def fake_embed(text):
        return [0.1, 0.2, 0.3]

    def fake_nn(emb, k):
        return [
            {"path": "a.md", "cosine": 0.99},  # = positive de query 1, self
            {"path": "almost-dup.md", "cosine": 0.97},  # > 0.95, duplicate
            {"path": "neg-1.md", "cosine": 0.85},
            {"path": "neg-2.md", "cosine": 0.78},
            {"path": "neg-3.md", "cosine": 0.71},
        ]

    result = mine_hard_negatives_for_synthetic(
        conn_with_synthetics,
        embed_fn=fake_embed,
        nearest_neighbors_fn=fake_nn,
        negatives_per_query=3,
    )
    # query 1 (positive=a.md): filtra self + 1 dup → 3 negs (neg-1/2/3)
    # query 2 (positive=b.md): a.md NO es self pero está en NN → entra como neg
    #                          almost-dup.md > 0.95 → filter
    #                          neg-1, neg-2, neg-3 quedan + a.md = 4 → top-3 = 3
    assert result["n_negatives_inserted"] == 6
    assert result["n_filtered_self"] == 1
    assert result["n_filtered_duplicate"] == 2  # 1 por query

    n_in_db = conn_with_synthetics.execute(
        "SELECT COUNT(*) FROM rag_synthetic_negatives"
    ).fetchone()[0]
    assert n_in_db == 6


def test_mine_negatives_idempotent(conn_with_synthetics):
    """Re-correr no duplica negatives."""
    def fake_embed(text):
        return [0.1] * 3

    def fake_nn(emb, k):
        return [{"path": "neg.md", "cosine": 0.5}]

    first = mine_hard_negatives_for_synthetic(
        conn_with_synthetics,
        embed_fn=fake_embed, nearest_neighbors_fn=fake_nn,
        negatives_per_query=1,
    )
    second = mine_hard_negatives_for_synthetic(
        conn_with_synthetics,
        embed_fn=fake_embed, nearest_neighbors_fn=fake_nn,
        negatives_per_query=1,
    )

    assert first["n_negatives_inserted"] == 2
    # Segunda corrida solo procesa queries NUEVAS (que no tengan negatives ya).
    # Como las dos del fixture ya tienen, n_examined=0.
    assert second["n_queries_examined"] == 0


def test_mine_negatives_dry_run(conn_with_synthetics):
    def fake_embed(text):
        return [0.1] * 3

    def fake_nn(emb, k):
        return [{"path": "neg.md", "cosine": 0.5}]

    result = mine_hard_negatives_for_synthetic(
        conn_with_synthetics,
        embed_fn=fake_embed, nearest_neighbors_fn=fake_nn,
        negatives_per_query=1,
        dry_run=True,
    )
    assert result["n_total_pairs"] == 2
    assert result["n_negatives_inserted"] == 0


def test_mine_default_constants():
    assert DEFAULT_NEGATIVES_PER_QUERY == 5
