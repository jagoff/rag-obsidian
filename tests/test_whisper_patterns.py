"""Tests para `rag_whisper_learning.patterns.find_correction_patterns()`.

Pattern detection encuentra single-word swaps repetidos (ej. "samando" →
"fernando" 3 veces) que son signal fuerte de errores sistemáticos del
modelo whisper. Multi-word changes se descartan por noise.
"""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import patch

import pytest

import rag
from rag_whisper_learning.patterns import (
    CorrectionPattern,
    _are_similar,
    find_correction_patterns,
)


@pytest.fixture
def tmp_corrections_db(tmp_path, monkeypatch):
    """DB sintética con la tabla `rag_audio_corrections` poblada para tests.
    Patchea `_ragvec_state_conn` de rag.py para que apunte a esta DB temp.
    """
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

    def _seed(rows: list[dict]):
        for r in rows:
            conn.execute(
                "INSERT INTO rag_audio_corrections "
                "(audio_hash, original, corrected, source, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    r.get("hash", "h"),
                    r["original"],
                    r["corrected"],
                    r.get("source", "explicit"),
                    r.get("ts", time.time()),
                ),
            )
        conn.commit()

    # Patch _ragvec_state_conn para devolver una conn fresca a esta DB.
    from contextlib import contextmanager

    @contextmanager
    def fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", fake_conn)
    yield _seed
    conn.close()


def test_no_corrections_returns_empty(tmp_corrections_db):
    """Sin correcciones acumuladas, devuelve []."""
    result = find_correction_patterns()
    assert result == []


def test_single_word_swap_repeated_detected(tmp_corrections_db):
    """3x el mismo swap → pattern detectado."""
    tmp_corrections_db([
        {"original": "te dije que samando viene mañana", "corrected": "te dije que fernando viene mañana"},
        {"original": "samando ya está", "corrected": "fernando ya está"},
        {"original": "llamale a samando ahora", "corrected": "llamale a fernando ahora"},
    ])
    result = find_correction_patterns(min_count=2)
    assert len(result) == 1
    p = result[0]
    assert p.original == "samando"
    assert p.corrected == "fernando"
    assert p.count == 3
    assert p.sources["explicit"] == 3


def test_min_count_filter(tmp_corrections_db):
    """Solo 1x del swap → no pasa el min_count default."""
    tmp_corrections_db([
        {"original": "samando viene", "corrected": "fernando viene"},
    ])
    assert find_correction_patterns(min_count=2) == []
    # Pero si bajamos el threshold a 1, sí lo encuentra.
    out = find_correction_patterns(min_count=1)
    assert len(out) == 1


def test_multi_word_changes_skipped(tmp_corrections_db):
    """Cambios multi-word se descartan por noise (paráfrasis, reformulación)."""
    tmp_corrections_db([
        # Cambian 2 palabras: paráfrasis, no error sistemático.
        {"original": "te pongo el anotate", "corrected": "te puse la nota"},
        {"original": "te pongo el anotate", "corrected": "te puse la nota"},
        {"original": "te pongo el anotate", "corrected": "te puse la nota"},
    ])
    result = find_correction_patterns(min_count=2)
    assert result == []


def test_typos_in_user_fix_filtered(tmp_corrections_db):
    """Si el "fix" del user es un typo trivial (palabras MUY similares,
    ratio ≥ 0.92), descartar. Ratios reales del stdlib SequenceMatcher:

    - "calendar" vs "calendar." → 0.94 → typo, descartar.
    - "fernando" vs "fernandó" → 0.875 → mantener (signal real).
    """
    tmp_corrections_db([
        {"original": "anotá calendar bien", "corrected": "anotá calendar. bien"},
        {"original": "anotá calendar bien", "corrected": "anotá calendar. bien"},
        {"original": "anotá calendar bien", "corrected": "anotá calendar. bien"},
    ])
    result = find_correction_patterns(min_count=2)
    assert result == []  # _are_similar() los descartó


def test_sources_breakdown(tmp_corrections_db):
    """El breakdown por source (explicit/llm/vault_diff) se devuelve correcto."""
    tmp_corrections_db([
        {"original": "samando", "corrected": "fernando", "source": "explicit"},
        {"original": "samando", "corrected": "fernando", "source": "llm"},
        {"original": "samando", "corrected": "fernando", "source": "explicit"},
        {"original": "samando", "corrected": "fernando", "source": "vault_diff"},
    ])
    result = find_correction_patterns(min_count=2)
    assert len(result) == 1
    p = result[0]
    assert p.count == 4
    assert p.sources == {"explicit": 2, "llm": 1, "vault_diff": 1}


def test_ordered_by_count_desc(tmp_corrections_db):
    """Los patterns más frecuentes aparecen primero."""
    tmp_corrections_db([
        # 4x "samando → fernando"
        *[{"original": "samando", "corrected": "fernando"} for _ in range(4)],
        # 2x "calendar → calendarizá"
        *[{"original": "calendar", "corrected": "calendarizá"} for _ in range(2)],
    ])
    result = find_correction_patterns(min_count=2)
    assert len(result) == 2
    assert result[0].original == "samando"
    assert result[0].count == 4
    assert result[1].original == "calendar"
    assert result[1].count == 2


# ── _are_similar (ratio threshold) ────────────────────────────────────────────


def test_are_similar_typos():
    """Palabras casi-idénticas (ratio ≥ 0.92) son `similar`."""
    assert _are_similar("calendar", "calendar.")  # ratio 0.941


def test_are_similar_distinct():
    """Palabras realmente distintas NO son `similar`."""
    assert not _are_similar("samando", "fernando")  # ratio 0.533
    assert not _are_similar("calendar", "calendarizá")  # ratio 0.842
    assert not _are_similar("hola", "chau")
    # `fernando` vs `fernandó` (acento extra) tiene ratio 0.875 → NO similar
    # con default 0.92. Se mantiene como signal legítimo.
    assert not _are_similar("fernando", "fernandó")


def test_are_similar_handles_empty():
    """Empty strings no rompen."""
    assert not _are_similar("", "fernando")
    assert not _are_similar("fernando", "")
    assert not _are_similar("", "")


def test_are_similar_threshold_configurable():
    """Threshold custom para tests / tuning."""
    # "fernando" vs "fernandó" tiene ratio 0.875.
    assert _are_similar("fernando", "fernandó", threshold=0.85)
    assert not _are_similar("fernando", "fernandó", threshold=0.92)
