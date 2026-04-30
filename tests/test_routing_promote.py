"""Tests para rag_routing_learning.promote.

Reusa la fixture `tmp_routing_db` del archivo de patterns — pytest detecta
fixtures por nombre dentro del mismo dir si las exponemos en conftest, pero
acá las redefinimos para mantener los tests independientes.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tmp_routing_db(tmp_path: Path, monkeypatch):
    import rag

    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE rag_routing_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            chat_jid TEXT NOT NULL,
            message_id TEXT NOT NULL,
            transcript TEXT NOT NULL,
            transcript_hash TEXT NOT NULL,
            bucket_llm TEXT NOT NULL,
            confidence_llm TEXT,
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


# ── upsert_rule ──────────────────────────────────────────────────────────────


def test_upsert_inserts_new(tmp_routing_db):
    from rag_routing_learning.promote import upsert_rule
    rid = upsert_rule(
        pattern="tengo que",
        bucket="reminder",
        evidence_count=10,
        evidence_ratio=0.95,
    )
    assert rid is not None
    assert rid > 0
    # Verificá la fila
    conn = sqlite3.connect(str(tmp_routing_db))
    row = conn.execute(
        "SELECT pattern, bucket, evidence_count, evidence_ratio, active "
        "FROM rag_routing_rules WHERE id = ?", (rid,)
    ).fetchone()
    conn.close()
    assert row == ("tengo que", "reminder", 10, 0.95, 1)


def test_upsert_updates_existing(tmp_routing_db):
    from rag_routing_learning.promote import upsert_rule
    rid1 = upsert_rule("tengo que", "reminder", 5, 0.92)
    rid2 = upsert_rule("tengo que", "reminder", 12, 0.97)
    # Mismo id — fue UPDATE no INSERT
    assert rid1 == rid2
    conn = sqlite3.connect(str(tmp_routing_db))
    row = conn.execute(
        "SELECT evidence_count, evidence_ratio FROM rag_routing_rules WHERE id = ?",
        (rid1,),
    ).fetchone()
    conn.close()
    assert row == (12, 0.97)


def test_upsert_does_not_reactivate(tmp_routing_db):
    """Una regla deshabilitada manualmente queda deshabilitada incluso
    si el cron la re-promueve. Solo el user puede reactivarla."""
    from rag_routing_learning.promote import deactivate_rule, upsert_rule
    rid = upsert_rule("foo", "inbox", 5, 0.95)
    assert deactivate_rule(rid) is True
    # Re-upsert con datos nuevos
    rid2 = upsert_rule("foo", "inbox", 20, 0.98)
    assert rid == rid2
    # Active sigue en 0
    conn = sqlite3.connect(str(tmp_routing_db))
    row = conn.execute(
        "SELECT active, evidence_count FROM rag_routing_rules WHERE id = ?",
        (rid,),
    ).fetchone()
    conn.close()
    assert row == (0, 20)


# ── deactivate / reactivate ──────────────────────────────────────────────────


def test_deactivate_rule(tmp_routing_db):
    from rag_routing_learning.promote import deactivate_rule, upsert_rule
    rid = upsert_rule("foo", "inbox", 5, 0.95)
    assert deactivate_rule(rid) is True
    # Idempotencia: deactivar dos veces devuelve False la segunda
    assert deactivate_rule(rid) is False


def test_deactivate_nonexistent_returns_false(tmp_routing_db):
    from rag_routing_learning.promote import deactivate_rule
    assert deactivate_rule(99999) is False


def test_reactivate_rule(tmp_routing_db):
    from rag_routing_learning.promote import (
        deactivate_rule, reactivate_rule, upsert_rule,
    )
    rid = upsert_rule("foo", "inbox", 5, 0.95)
    assert deactivate_rule(rid) is True
    assert reactivate_rule(rid) is True
    # Idempotencia
    assert reactivate_rule(rid) is False


# ── list_active_rules / list_all_rules ───────────────────────────────────────


def test_list_active_rules_returns_only_active(tmp_routing_db):
    from rag_routing_learning.promote import (
        deactivate_rule, list_active_rules, list_all_rules, upsert_rule,
    )
    r1 = upsert_rule("a", "reminder", 5, 0.95)
    r2 = upsert_rule("b", "calendar_timed", 8, 0.96)
    r3 = upsert_rule("c", "inbox", 6, 0.92)
    deactivate_rule(r2)
    active = list_active_rules()
    all_rules = list_all_rules()
    assert len(active) == 2
    assert {r.pattern for r in active} == {"a", "c"}
    assert len(all_rules) == 3
    assert {r.pattern for r in all_rules} == {"a", "b", "c"}


def test_list_active_orders_by_evidence_count_desc(tmp_routing_db):
    from rag_routing_learning.promote import list_active_rules, upsert_rule
    upsert_rule("low", "inbox", 5, 0.91)
    upsert_rule("high", "reminder", 50, 0.99)
    upsert_rule("mid", "calendar_timed", 15, 0.95)
    rules = list_active_rules()
    counts = [r.evidence_count for r in rules]
    assert counts == sorted(counts, reverse=True)


# ── list_candidate_patterns ──────────────────────────────────────────────────


def test_list_candidate_marks_already_promoted(tmp_routing_db):
    """Si un pattern ya está en rag_routing_rules, debe marcarse."""
    import time
    from rag_routing_learning.promote import (
        list_candidate_patterns, upsert_rule,
    )

    now = int(time.time())
    conn = sqlite3.connect(str(tmp_routing_db))
    for i in range(5):
        conn.execute(
            "INSERT INTO rag_routing_decisions "
            "(ts, chat_jid, message_id, transcript, transcript_hash, "
            " bucket_llm, extracted_json, bucket_final) "
            "VALUES (?, 'c', ?, 'tengo que comprar pan A', 'h', 'reminder', '{}', 'reminder')",
            (now - i * 100, f"m{i}"),
        )
    conn.commit()
    conn.close()

    # Sin promote previo, todos los candidates están "no promoted"
    candidates = list_candidate_patterns(min_count=5, min_ratio=0.90)
    assert any(c["pattern"] == "tengo que" for c in candidates)
    tq = next(c for c in candidates if c["pattern"] == "tengo que")
    assert tq["already_promoted"] is False

    # Ahora promovemos
    upsert_rule("tengo que", "reminder", 5, 1.0)
    candidates2 = list_candidate_patterns(min_count=5, min_ratio=0.90)
    tq2 = next(c for c in candidates2 if c["pattern"] == "tengo que")
    assert tq2["already_promoted"] is True
    assert tq2["active"] is True


# ── render_rules_block ───────────────────────────────────────────────────────


def test_render_rules_block_empty():
    from rag_routing_learning.promote import render_rules_block
    assert render_rules_block([]) == ""


def test_render_rules_block_format():
    from rag_routing_learning.promote import LearnedRule, render_rules_block
    rules = [
        LearnedRule(
            id=1, pattern="tengo que", bucket="reminder",
            evidence_count=23, evidence_ratio=0.95, promoted_at=0, active=1,
        ),
        LearnedRule(
            id=2, pattern="turno con", bucket="calendar_timed",
            evidence_count=11, evidence_ratio=1.0, promoted_at=0, active=1,
        ),
    ]
    block = render_rules_block(rules)
    assert "REGLAS APRENDIDAS" in block
    assert '"tengo que"' in block
    assert "reminder" in block
    assert "95%" in block
    assert "23 casos" in block
    assert '"turno con"' in block
    assert "100% de 11 casos" in block
