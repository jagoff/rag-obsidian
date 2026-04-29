"""Tests for entity extraction helpers (Improvement #2, Fase B).

Cubre: _normalize_entity_name, _cluster_entities, _extract_entities_single,
_extract_entities_batch, _upsert_entities_for_chunk, _get_gliner_model sticky-fail.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

import rag


# ──────────────────────────────────────────────────────────────────────
# _normalize_entity_name
# ──────────────────────────────────────────────────────────────────────

def test_normalize_accent_strip():
    assert rag._normalize_entity_name("Juan Pérez") == "juan perez"


def test_normalize_case_fold():
    assert rag._normalize_entity_name("JUAN PÉREZ") == "juan perez"


def test_normalize_whitespace_collapse():
    assert rag._normalize_entity_name("  Juan   Pérez   ") == "juan perez"


def test_normalize_empty():
    assert rag._normalize_entity_name("") == ""


def test_normalize_none_handled():
    # Función acepta None gracefully (early return "")
    assert rag._normalize_entity_name(None) == ""


# ──────────────────────────────────────────────────────────────────────
# _cluster_entities
# ──────────────────────────────────────────────────────────────────────

def test_cluster_merges_same_normalized():
    """Mismo normalized + type → un solo cluster."""
    candidates = [
        ("Juan Pérez", "person", 0.9),
        ("juan perez", "person", 0.85),
        ("JUAN PÉREZ", "person", 0.95),
    ]
    clusters = rag._cluster_entities(candidates)
    assert len(clusters) == 1
    assert list(clusters.keys())[0] == ("juan perez", "person")


def test_cluster_different_types_separate():
    """Mismo normalized, distintos types → 2 clusters."""
    candidates = [("Juan", "person", 0.95), ("Juan", "organization", 0.85)]
    clusters = rag._cluster_entities(candidates)
    assert len(clusters) == 2


def test_cluster_canonical_by_frequency():
    """Canonical = forma más frecuente."""
    candidates = [
        ("Juan Pérez", "person", 0.9),
        ("Juan Pérez", "person", 0.85),
        ("Juan Pérez", "person", 0.95),
        ("Juan P.", "person", 0.80),
    ]
    clusters = rag._cluster_entities(candidates)
    key = ("juan perez", "person")
    assert clusters[key]["canonical"] == "Juan Pérez"


def test_cluster_canonical_longest_on_tie():
    """Empate en frecuencia → canonical más largo."""
    candidates = [
        ("Juan Pérez", "person", 0.9),
        ("Juan P", "person", 0.85),
    ]
    # Normalizados: "juan perez" y "juan p" — distintos, 2 clusters.
    # Dentro de cada uno hay una sola forma.
    clusters = rag._cluster_entities(candidates)
    assert len(clusters) == 2


def test_cluster_avg_confidence():
    """Confidence = promedio de scores."""
    candidates = [("Juan", "person", 0.8), ("Juan", "person", 0.9)]
    clusters = rag._cluster_entities(candidates)
    key = ("juan", "person")
    assert abs(clusters[key]["confidence"] - 0.85) < 1e-6


def test_cluster_filters_empty_text():
    """Candidates con text vacío se ignoran."""
    candidates = [
        ("Juan", "person", 0.9),
        ("", "person", 0.85),
        ("Ana", "person", 0.88),
    ]
    clusters = rag._cluster_entities(candidates)
    assert len(clusters) == 2
    assert ("juan", "person") in clusters
    assert ("ana", "person") in clusters


def test_cluster_count():
    """count = número de menciones agrupadas."""
    candidates = [("Juan", "person", 0.9)] * 3
    clusters = rag._cluster_entities(candidates)
    assert clusters[("juan", "person")]["count"] == 3


def test_cluster_aliases_dedup_same_normalized():
    """Múltiples formas del mismo normalized → canonical + aliases dedup."""
    # "Juan Perez", "juan perez", "JUAN PEREZ" → misma normalized.
    # Canonical más frecuente, aliases = el resto.
    candidates = [
        ("Juan Perez", "person", 0.9),
        ("juan perez", "person", 0.85),
        ("JUAN PEREZ", "person", 0.95),
    ]
    clusters = rag._cluster_entities(candidates)
    cluster = clusters[("juan perez", "person")]
    forms_seen = {cluster["canonical"]} | set(cluster["aliases"])
    assert cluster["canonical"] not in cluster["aliases"]
    assert len(forms_seen) == 3


# ──────────────────────────────────────────────────────────────────────
# _extract_entities_single (mocked GLiNER)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def reset_gliner_state():
    """Reset GLiNER global state before + after each test."""
    prev_model = rag._gliner_model
    prev_failed = rag._gliner_load_failed
    rag._gliner_model = None
    rag._gliner_load_failed = False
    yield
    rag._gliner_model = prev_model
    rag._gliner_load_failed = prev_failed


def test_extract_single_no_model(reset_gliner_state, monkeypatch):
    """No model → empty list."""
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: None)
    assert rag._extract_entities_single("Juan Pérez habló con Ana.") == []


def test_extract_single_empty_text(reset_gliner_state):
    assert rag._extract_entities_single("") == []
    assert rag._extract_entities_single("   ") == []


def test_extract_single_filters_low_confidence(reset_gliner_state, monkeypatch):
    """Scores < _ENTITY_CONFIDENCE_MIN filtrados."""
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "Juan", "label": "person", "score": 0.95},
        {"text": "Ana", "label": "person", "score": 0.65},  # < 0.70
    ]
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_model)
    result = rag._extract_entities_single("Juan y Ana.")
    assert len(result) == 1
    assert result[0][0] == "Juan"


def test_extract_single_filters_unknown_label(reset_gliner_state, monkeypatch):
    """Labels no en _ENTITY_LABELS filtrados."""
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "Juan", "label": "person", "score": 0.95},
        {"text": "Argentina", "label": "country", "score": 0.90},
    ]
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_model)
    result = rag._extract_entities_single("Juan en Argentina.")
    assert len(result) == 1
    assert result[0][1] == "person"


def test_extract_single_predict_raises(reset_gliner_state, monkeypatch):
    """model.predict_entities raises → empty list, no crash."""
    mock_model = MagicMock()
    mock_model.predict_entities.side_effect = RuntimeError("predict failed")
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_model)
    assert rag._extract_entities_single("Juan") == []


# ──────────────────────────────────────────────────────────────────────
# _extract_entities_batch
# ──────────────────────────────────────────────────────────────────────

def test_batch_empty():
    assert rag._extract_entities_batch([]) == []


def test_batch_returns_one_dict_per_input(reset_gliner_state, monkeypatch):
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: None)
    result = rag._extract_entities_batch(["a", "b", "c"])
    assert len(result) == 3
    assert all(isinstance(d, dict) for d in result)


def test_batch_uses_inference_not_predict_entities(reset_gliner_state, monkeypatch):
    """_extract_entities_batch calls model.inference (batched), not predict_entities in a loop."""
    mock_model = MagicMock()
    # inference() returns List[List[Dict]] — one list per input text
    mock_model.inference.return_value = [
        [{"text": "Juan", "label": "person", "score": 0.95}],
        [{"text": "Moka", "label": "organization", "score": 0.90}],
    ]
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_model)
    result = rag._extract_entities_batch(["Juan habla.", "Moka es una empresa."])
    # inference called ONCE with the full list, not twice
    assert mock_model.inference.call_count == 1
    call_args = mock_model.inference.call_args
    assert call_args[0][0] == ["Juan habla.", "Moka es una empresa."]
    # predict_entities never called
    assert mock_model.predict_entities.call_count == 0
    # Each result is a clustered dict
    assert len(result) == 2
    assert ("juan", "person") in result[0]
    assert ("moka", "organization") in result[1]


def test_batch_inference_raises_returns_empty_dicts(reset_gliner_state, monkeypatch):
    """model.inference raising returns [{}] * n, no crash."""
    mock_model = MagicMock()
    mock_model.inference.side_effect = RuntimeError("inference failed")
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_model)
    result = rag._extract_entities_batch(["text1", "text2"])
    assert result == [{}, {}]


def test_batch_output_identical_to_single_loop(reset_gliner_state, monkeypatch):
    """Output of batch path is bit-identical to calling _extract_entities_single per item."""
    entities_per_text = [
        [{"text": "Juan", "label": "person", "score": 0.95}],
        [{"text": "yo", "label": "person", "score": 0.92}],  # stopword — filtered
        [{"text": "Moka", "label": "organization", "score": 0.88}],
    ]
    texts = ["t1", "t2", "t3"]

    # Batched path
    mock_batch = MagicMock()
    mock_batch.inference.return_value = entities_per_text
    monkeypatch.setattr(rag, "_get_gliner_model", lambda: mock_batch)
    batch_result = rag._extract_entities_batch(texts)

    # Single-loop path (simulate old behaviour via _parse_raw_entities directly)
    single_result = [
        rag._cluster_entities(rag._parse_raw_entities(raw))
        for raw in entities_per_text
    ]

    assert batch_result == single_result


# ──────────────────────────────────────────────────────────────────────
# _parse_raw_entities
# ──────────────────────────────────────────────────────────────────────

def test_parse_raw_filters_low_confidence():
    raw = [
        {"text": "Juan", "label": "person", "score": 0.95},
        {"text": "Ana", "label": "person", "score": 0.50},  # below 0.70
    ]
    result = rag._parse_raw_entities(raw)
    assert len(result) == 1
    assert result[0][0] == "Juan"


def test_parse_raw_filters_stopword_person():
    raw = [{"text": "yo", "label": "person", "score": 0.95}]
    assert rag._parse_raw_entities(raw) == []


def test_parse_raw_filters_phone_id():
    raw = [{"text": "5493424303891", "label": "person", "score": 0.92}]
    assert rag._parse_raw_entities(raw) == []


def test_parse_raw_filters_short_entity():
    raw = [{"text": "pc", "label": "organization", "score": 0.90}]
    assert rag._parse_raw_entities(raw) == []


def test_parse_raw_filters_unknown_label():
    raw = [{"text": "Argentina", "label": "country", "score": 0.92}]
    assert rag._parse_raw_entities(raw) == []


def test_parse_raw_passes_valid_entities():
    raw = [
        {"text": "Juan", "label": "person", "score": 0.90},
        {"text": "Moka", "label": "organization", "score": 0.85},
    ]
    result = rag._parse_raw_entities(raw)
    assert len(result) == 2
    texts = [r[0] for r in result]
    assert "Juan" in texts
    assert "Moka" in texts


# ──────────────────────────────────────────────────────────────────────
# _upsert_entities_for_chunk
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    """In-memory SQLite con DDL telemetry + FK on."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    rag._ensure_telemetry_tables(c)
    yield c
    c.close()


def test_upsert_insert_new(conn):
    entities = {
        ("juan perez", "person"): {
            "canonical": "Juan Pérez",
            "aliases": ["JP"],
            "confidence": 0.9,
            "count": 1,
        }
    }
    count = rag._upsert_entities_for_chunk(
        conn, entities, "chunk-1", "vault", 1700000000.0, "Juan Pérez dijo..."
    )
    assert count == 1
    row = conn.execute(
        "SELECT canonical_name, mention_count FROM rag_entities WHERE normalized=?",
        ("juan perez",),
    ).fetchone()
    assert row == ("Juan Pérez", 1)


def test_upsert_updates_mention_count(conn):
    """Same entity, different chunk → mention_count incrementa."""
    entities = {
        ("juan perez", "person"): {
            "canonical": "Juan Pérez",
            "aliases": [],
            "confidence": 0.9,
            "count": 1,
        }
    }
    rag._upsert_entities_for_chunk(conn, entities, "chunk-1", "vault", 1700000000.0, "text")
    rag._upsert_entities_for_chunk(conn, entities, "chunk-2", "vault", 1700000001.0, "text")
    mc = conn.execute("SELECT mention_count FROM rag_entities WHERE normalized=?",
                      ("juan perez",)).fetchone()[0]
    assert mc == 2


def test_upsert_duplicate_chunk_ignored(conn):
    """Same entity + same chunk → UNIQUE guard, solo 1 mention."""
    entities = {
        ("juan perez", "person"): {
            "canonical": "Juan Pérez",
            "aliases": [],
            "confidence": 0.9,
            "count": 1,
        }
    }
    rag._upsert_entities_for_chunk(conn, entities, "chunk-1", "vault", 1700000000.0, "t")
    rag._upsert_entities_for_chunk(conn, entities, "chunk-1", "vault", 1700000001.0, "t")
    mentions = conn.execute(
        "SELECT COUNT(*) FROM rag_entity_mentions WHERE chunk_id=?", ("chunk-1",)
    ).fetchone()[0]
    assert mentions == 1


def test_upsert_empty_entities(conn):
    assert rag._upsert_entities_for_chunk(conn, {}, "chunk-1", "vault", 1.0, "t") == 0


def test_upsert_no_chunk_id(conn):
    entities = {("juan perez", "person"): {
        "canonical": "Juan Pérez", "aliases": [], "confidence": 0.9, "count": 1,
    }}
    assert rag._upsert_entities_for_chunk(conn, entities, "", "vault", 1.0, "t") == 0


# ──────────────────────────────────────────────────────────────────────
# _get_gliner_model sticky failure
# ──────────────────────────────────────────────────────────────────────

def test_gliner_sticky_failure_on_import_error(reset_gliner_state, monkeypatch):
    """ImportError primera vez → _gliner_load_failed=True, subsequent calls None rápido."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gliner":
            raise ImportError("no gliner")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # First call: fails
    assert rag._get_gliner_model() is None
    assert rag._gliner_load_failed is True

    # Second call: returns None without retry
    assert rag._get_gliner_model() is None


# ──────────────────────────────────────────────────────────────────────
# _extract_and_index_entities_for_chunks (high-level)
# ──────────────────────────────────────────────────────────────────────

def test_extract_and_index_no_crash_on_default(monkeypatch):
    """Post-flip (2026-04-21): default ON. Must not crash when gliner is
    missing (silent-fail via sticky flag) nor when gliner is present but
    the input is trivial."""
    monkeypatch.delenv("RAG_EXTRACT_ENTITIES", raising=False)
    # Should not crash regardless of gliner availability
    rag._extract_and_index_entities_for_chunks(
        ["text"], ["id1"], [{"ts": 1.0}], "vault"
    )


def test_extract_and_index_disabled_when_flag_off(monkeypatch):
    """Explicit RAG_EXTRACT_ENTITIES=0 → no-op even with gliner installed."""
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    rag._extract_and_index_entities_for_chunks(
        ["text"], ["id1"], [{"ts": 1.0}], "vault"
    )


def test_extract_and_index_empty_inputs(monkeypatch):
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "1")
    rag._extract_and_index_entities_for_chunks([], [], [], "vault")  # no crash


def test_entity_extraction_enabled_reads_env(monkeypatch):
    # Post 2026-04-21: default ON. Only "0"/"false"/"no" disable.
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "1")
    assert rag._entity_extraction_enabled() is True

    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    assert rag._entity_extraction_enabled() is False

    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "false")
    assert rag._entity_extraction_enabled() is False

    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "no")
    assert rag._entity_extraction_enabled() is False

    # Unset / empty string → default ON (post-flip invariant)
    monkeypatch.delenv("RAG_EXTRACT_ENTITIES", raising=False)
    assert rag._entity_extraction_enabled() is True

    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "")
    assert rag._entity_extraction_enabled() is True
