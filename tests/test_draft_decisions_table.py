"""Tests for the rag_draft_decisions table DDL + helper (2026-04-29).

Cierra el loop de auto-aprendizaje del bot WA: incoming → draft → user
puntúa /si /no /editar en RagNet → reply al contacto + feedback al modelo.
La tabla se popula via /api/draft/decision (web/server.py); estos tests
aseguran que el shape de la tabla + el helper Python `_record_draft_decision`
están correctos antes de que llegue el primer write desde producción.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import rag


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path. _ragvec_state_conn() crea las
    tablas on-demand al primer uso (vía _ensure_telemetry_tables)."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    rag.SqliteVecClient(path=str(db_path))
    with rag._ragvec_state_conn() as _conn:
        pass
    return db_path


# ── DDL: tabla + indices + CHECK constraint ─────────────────────────────────

def test_table_exists(state_db):
    """rag_draft_decisions debe estar registrada en sqlite_master."""
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_draft_decisions'"
        ).fetchone()
    assert row is not None
    assert row[0] == "rag_draft_decisions"


def test_table_schema(state_db):
    """Columnas esperadas + tipos. PRAGMA table_info devuelve
    (cid, name, type, notnull, dflt_value, pk)."""
    with rag._ragvec_state_conn() as conn:
        cols = list(conn.execute(
            "PRAGMA table_info(rag_draft_decisions)"
        ).fetchall())
    by_name = {c[1]: c for c in cols}
    expected_cols = {
        "id", "ts", "draft_id", "contact_jid", "contact_name",
        "original_msgs_json", "bot_draft", "decision", "sent_text",
        "extra_json",
    }
    assert set(by_name) == expected_cols
    # NOT NULL en ts, draft_id, contact_jid, decision (los demás son opcionales).
    assert by_name["ts"][3] == 1
    assert by_name["draft_id"][3] == 1
    assert by_name["contact_jid"][3] == 1
    assert by_name["decision"][3] == 1
    assert by_name["contact_name"][3] == 0
    assert by_name["sent_text"][3] == 0
    # id es la PK.
    assert by_name["id"][5] == 1


def test_indices_present(state_db):
    """Los 3 índices que aceleran las queries del dashboard + dedup."""
    with rag._ragvec_state_conn() as conn:
        idx_rows = list(conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='rag_draft_decisions'"
        ).fetchall())
    names = {r[0] for r in idx_rows}
    assert "ix_rag_draft_decisions_ts" in names
    assert "ix_rag_draft_decisions_decision" in names
    assert "ix_rag_draft_decisions_jid" in names


def test_check_constraint_rejects_invalid_decision(state_db):
    """El CHECK enforcea decision IN ('approved_si', 'approved_editar',
    'rejected', 'expired'). 'banana' debe disparar IntegrityError."""
    with rag._ragvec_state_conn() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO rag_draft_decisions "
                "(ts, draft_id, contact_jid, decision) "
                "VALUES (?, ?, ?, ?)",
                ("2026-04-29T10:00:00", "abc", "x@s.whatsapp.net", "banana"),
            )
            conn.commit()


def test_check_constraint_accepts_all_4_valid(state_db):
    """Las 4 decision types válidas deben pasar el CHECK."""
    valid = ("approved_si", "approved_editar", "rejected", "expired")
    with rag._ragvec_state_conn() as conn:
        for i, dec in enumerate(valid):
            conn.execute(
                "INSERT INTO rag_draft_decisions "
                "(ts, draft_id, contact_jid, decision) "
                "VALUES (?, ?, ?, ?)",
                (f"2026-04-29T10:0{i}:00", f"d{i}", "x@s.whatsapp.net", dec),
            )
        conn.commit()
        cnt = conn.execute(
            "SELECT COUNT(*) FROM rag_draft_decisions"
        ).fetchone()[0]
    assert cnt == 4


# ── Helper _record_draft_decision ────────────────────────────────────────────

def test_record_draft_decision_happy_path(state_db):
    """Helper persiste la row + serializa original_msgs como JSON."""
    rid = rag._record_draft_decision(
        draft_id="abc123",
        contact_jid="5491155555555@s.whatsapp.net",
        contact_name="Juan",
        original_msgs=[
            {"id": "m1", "text": "hola", "ts": "2026-04-29T10:00:00"},
            {"id": "m2", "text": "qué hacés", "ts": "2026-04-29T10:01:00"},
        ],
        bot_draft="todo bien, vos?",
        decision="approved_si",
        sent_text="todo bien, vos?",
    )
    assert isinstance(rid, int)
    assert rid >= 1

    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT draft_id, contact_jid, contact_name, decision, "
            "sent_text, original_msgs_json, bot_draft "
            "FROM rag_draft_decisions WHERE id=?",
            (rid,),
        ).fetchone()
    assert row is not None
    (draft_id, jid, name, decision, sent_text, msgs_json, bot_draft) = row
    assert draft_id == "abc123"
    assert jid == "5491155555555@s.whatsapp.net"
    assert name == "Juan"
    assert decision == "approved_si"
    assert sent_text == "todo bien, vos?"
    assert bot_draft == "todo bien, vos?"
    # original_msgs viene como JSON array. Lo parseamos para chequear shape.
    import json
    msgs = json.loads(msgs_json)
    assert len(msgs) == 2
    assert msgs[0]["text"] == "hola"


def test_record_draft_decision_invalid_decision_returns_none(state_db):
    """Decision inválida → helper devuelve None y no inserta nada."""
    rid = rag._record_draft_decision(
        draft_id="abc",
        contact_jid="x@s.whatsapp.net",
        contact_name=None,
        original_msgs=[],
        bot_draft="",
        decision="banana",  # ← inválido
    )
    assert rid is None
    with rag._ragvec_state_conn() as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM rag_draft_decisions"
        ).fetchone()[0]
    assert cnt == 0


def test_record_draft_decision_minimal_fields(state_db):
    """Optional fields (contact_name, sent_text, extra) → NULL en SQL."""
    rid = rag._record_draft_decision(
        draft_id="d1",
        contact_jid="x@s.whatsapp.net",
        contact_name=None,
        original_msgs=[],
        bot_draft="hola",
        decision="rejected",
    )
    assert rid is not None
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT contact_name, sent_text, extra_json "
            "FROM rag_draft_decisions WHERE id=?",
            (rid,),
        ).fetchone()
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None


def test_record_draft_decision_with_extra(state_db):
    """`extra` se serializa a extra_json."""
    rid = rag._record_draft_decision(
        draft_id="d1",
        contact_jid="x@s.whatsapp.net",
        contact_name=None,
        original_msgs=[],
        bot_draft="",
        decision="expired",
        extra={"draft_score": 0.42, "model": "qwen2.5:7b"},
    )
    assert rid is not None
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT extra_json FROM rag_draft_decisions WHERE id=?", (rid,),
        ).fetchone()
    import json
    extra = json.loads(row[0])
    assert extra == {"draft_score": 0.42, "model": "qwen2.5:7b"}


def test_record_draft_decision_4_decision_types(state_db):
    """Smoke: las 4 decision types enumeradas pasan el helper."""
    valid = ("approved_si", "approved_editar", "rejected", "expired")
    for dec in valid:
        rid = rag._record_draft_decision(
            draft_id=f"d:{dec}",
            contact_jid="x@s.whatsapp.net",
            contact_name=None,
            original_msgs=[],
            bot_draft="x",
            decision=dec,
            sent_text=("x" if dec.startswith("approved") else None),
        )
        assert rid is not None, f"helper rebotó {dec!r}"
    with rag._ragvec_state_conn() as conn:
        rows = list(conn.execute(
            "SELECT decision FROM rag_draft_decisions ORDER BY id"
        ).fetchall())
    assert [r[0] for r in rows] == list(valid)
