"""Tests para rag_routing_learning.patterns.extract_pivot_phrases.

Estrategia: usamos un tmp_path con una DB sqlite real, creamos las tablas
mínimas (`rag_routing_decisions` con el subset de columnas que el extractor
necesita), populamos con datos sintéticos, llamamos al extractor.

El monkeypatch va sobre `rag._ragvec_state_conn` para que apunte a la DB
temporal — exactamente el patrón de los tests existentes de telemetry
(ver test_sql_state_primitives.py).
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_routing_db(tmp_path: Path, monkeypatch):
    """Crea una DB sqlite temporal con la tabla rag_routing_decisions y
    monkeypatchea rag._ragvec_state_conn para apuntarla.

    Devuelve la path para que el test pueda inspectar la DB después.
    """
    import rag

    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    # Sólo las columnas que el extractor de patterns necesita —
    # transcript + bucket_final + ts.
    conn.executescript("""
        CREATE TABLE rag_routing_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            chat_jid TEXT NOT NULL,
            message_id TEXT NOT NULL,
            transcript TEXT NOT NULL,
            transcript_hash TEXT NOT NULL,
            bucket_llm TEXT NOT NULL,
            extracted_json TEXT NOT NULL,
            bucket_final TEXT,
            user_response TEXT,
            UNIQUE(message_id, chat_jid)
        );
        CREATE TABLE rag_routing_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            bucket TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            evidence_ratio REAL NOT NULL,
            promoted_at INTEGER NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            notes TEXT,
            UNIQUE(pattern, bucket)
        );
    """)
    conn.commit()

    @contextlib.contextmanager
    def fake_conn():
        c = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", fake_conn)
    yield db_path
    conn.close()


def _insert_decisions(db_path: Path, rows: list[tuple[str, str, int]]):
    """Insert helper. Cada tupla = (transcript, bucket_final, ts)."""
    conn = sqlite3.connect(str(db_path))
    for i, (transcript, bucket_final, ts) in enumerate(rows):
        conn.execute(
            "INSERT INTO rag_routing_decisions "
            "(ts, chat_jid, message_id, transcript, transcript_hash, "
            " bucket_llm, extracted_json, bucket_final) "
            "VALUES (?, 'test@chat', ?, ?, ?, ?, '{}', ?)",
            (ts, f"msg_{i}", transcript, f"hash_{i}", bucket_final, bucket_final),
        )
    conn.commit()
    conn.close()


# ── extract_pivot_phrases ────────────────────────────────────────────────────


def test_extract_returns_empty_on_empty_db(tmp_routing_db):
    from rag_routing_learning.patterns import extract_pivot_phrases
    assert extract_pivot_phrases() == []


def test_extract_finds_consistent_bigram(tmp_routing_db):
    """Si "tengo que" aparece 5x todas en reminder, la extracción lo agarra."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    _insert_decisions(tmp_routing_db, [
        ("tengo que llamar a juan", "reminder", now - 100),
        ("tengo que comprar pan", "reminder", now - 200),
        ("tengo que ir al banco", "reminder", now - 300),
        ("tengo que pagar la factura", "reminder", now - 400),
        ("tengo que llevar el auto al taller", "reminder", now - 500),
    ])
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    # Debería encontrar "tengo que" como bigram dominante en reminder.
    matching = [p for p in patterns if p.pattern == "tengo que"]
    assert len(matching) == 1
    p = matching[0]
    assert p.bucket == "reminder"
    assert p.count == 5
    assert p.ratio == 1.0
    assert len(p.examples) > 0


def test_extract_filters_below_min_count(tmp_routing_db):
    """Si el bigram aparece <min_count veces, no califica aunque sea 100% un bucket."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    _insert_decisions(tmp_routing_db, [
        ("tengo que A", "reminder", now - 100),
        ("tengo que B", "reminder", now - 200),
    ])
    # Sólo 2 ocurrencias < min_count=5
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    assert all(p.pattern != "tengo que" for p in patterns)


def test_extract_filters_below_min_ratio(tmp_routing_db):
    """Si el bigram va a varios buckets, no califica."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    # 6 ocurrencias: 3 reminder, 3 calendar_timed → ratio 50%
    _insert_decisions(tmp_routing_db, [
        ("tengo que A", "reminder", now - 100),
        ("tengo que B", "reminder", now - 200),
        ("tengo que C", "reminder", now - 300),
        ("tengo que D mañana 3pm", "calendar_timed", now - 400),
        ("tengo que E lunes 9am", "calendar_timed", now - 500),
        ("tengo que F martes 11am", "calendar_timed", now - 600),
    ])
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    # No debería estar — ratio 50% < 90%
    assert all(p.pattern != "tengo que" for p in patterns)


def test_extract_respects_days_window(tmp_routing_db):
    """Filas fuera de la ventana de días se ignoran."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    old_ts = now - 70 * 86400  # 70 días atrás
    _insert_decisions(tmp_routing_db, [
        ("tengo que A", "reminder", old_ts),
        ("tengo que B", "reminder", old_ts - 100),
        ("tengo que C", "reminder", old_ts - 200),
        ("tengo que D", "reminder", old_ts - 300),
        ("tengo que E", "reminder", old_ts - 400),
    ])
    # Default days=60 → todas estas filas son demasiado viejas
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90, days=60)
    assert patterns == []
    # Pero si aumentamos la ventana las agarra
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90, days=100)
    assert any(p.pattern == "tengo que" for p in patterns)


def test_extract_skips_decisions_with_null_bucket_final(tmp_routing_db):
    """Filas donde el user dijo "no" (bucket_final IS NULL) no cuentan."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    conn = sqlite3.connect(str(tmp_routing_db))
    for i in range(5):
        conn.execute(
            "INSERT INTO rag_routing_decisions "
            "(ts, chat_jid, message_id, transcript, transcript_hash, "
            " bucket_llm, extracted_json, bucket_final) "
            "VALUES (?, 'c', ?, 'tengo que algo', 'h', 'reminder', '{}', NULL)",
            (now - i * 100, f"m{i}"),
        )
    conn.commit()
    conn.close()
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    assert patterns == []


def test_extract_counts_each_transcript_once(tmp_routing_db):
    """Si el bigram aparece 2x en un mismo transcript, cuenta UNA vez."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    # Transcript con "tengo que" 3 veces → cuenta como 1 ocurrencia.
    _insert_decisions(tmp_routing_db, [
        ("tengo que tengo que tengo que A", "reminder", now - i * 100)
        for i in range(5)
    ])
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    matching = [p for p in patterns if p.pattern == "tengo que"]
    assert len(matching) == 1
    # 5 transcripts → count=5, no 15.
    assert matching[0].count == 5


def test_extract_finds_trigrams(tmp_routing_db):
    """Trigrams como 'turno con el' deberían capturarse."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    _insert_decisions(tmp_routing_db, [
        ("turno con el psiquiatra el lunes 9am", "calendar_timed", now - 100),
        ("turno con el dentista el martes 10am", "calendar_timed", now - 200),
        ("turno con el médico el jueves 11am", "calendar_timed", now - 300),
        ("turno con el odontólogo el viernes 14h", "calendar_timed", now - 400),
        ("turno con el traumatólogo el lunes 16h", "calendar_timed", now - 500),
    ])
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90, ngram_sizes=(2, 3))
    # Debería capturar tanto "turno con" (bigram) como "turno con el" (trigram)
    bigram = [p for p in patterns if p.pattern == "turno con"]
    trigram = [p for p in patterns if p.pattern == "turno con el"]
    assert len(bigram) == 1
    assert bigram[0].bucket == "calendar_timed"
    assert len(trigram) == 1
    assert trigram[0].bucket == "calendar_timed"


def test_extract_returns_empty_on_db_error(monkeypatch):
    """Sin DB / sin tabla — no crashea, devuelve []."""
    import rag
    from rag_routing_learning.patterns import extract_pivot_phrases

    @contextlib.contextmanager
    def broken():
        # Connection a una DB inexistente que va a fallar al ejecutar.
        c = sqlite3.connect(":memory:")
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", broken)
    assert extract_pivot_phrases() == []


def test_extract_skips_pure_stopword_ngrams(tmp_routing_db):
    """N-grams compuestos enteramente por stopwords no se extraen."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    # "que de" son ambas stopwords — no deberían generar pattern.
    _insert_decisions(tmp_routing_db, [
        ("que de algo importante", "reminder", now - i * 100)
        for i in range(5)
    ])
    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    assert all(p.pattern != "que de" for p in patterns)


def test_extract_orders_by_count_desc(tmp_routing_db):
    """Patrones con más evidencia deberían venir primero."""
    import time
    from rag_routing_learning.patterns import extract_pivot_phrases
    now = int(time.time())
    # "frase frecuente" → 7 veces, "frase rara" → 5 veces. Ambas en reminder.
    rows = []
    for i in range(7):
        rows.append((f"frase frecuente A{i}", "reminder", now - i * 100))
    for i in range(5):
        rows.append((f"frase rara B{i}", "reminder", now - (100 + i) * 100))
    _insert_decisions(tmp_routing_db, rows)

    patterns = extract_pivot_phrases(min_count=5, min_ratio=0.90)
    frec_idx = next(i for i, p in enumerate(patterns) if p.pattern == "frase frecuente")
    rara_idx = next(i for i, p in enumerate(patterns) if p.pattern == "frase rara")
    assert frec_idx < rara_idx
