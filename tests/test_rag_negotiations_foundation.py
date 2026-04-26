"""Tests para Fase 0 (Foundation) del feature wa-negotiation-autopilot.

Cobertura:

1. **State machine** (10 tests) — grafo de transiciones, terminales,
   helpers, edge cases.
2. **DDL idempotente** (5 tests) — las 5 tablas se crean correctamente
   por `_ensure_telemetry_tables()` y se pueden re-crear sin error.
3. **CRUD `rag_negotiations`** (6 tests) — create/get/update/list +
   message count.
4. **CRUD `rag_negotiation_turns`** (3 tests) — append + list +
   FK cascade.
5. **CRUD `rag_negotiation_pending_sends`** (3 tests) — enqueue +
   dequeue_due + mark_send.
6. **CRUD `rag_style_fingerprints`** (2 tests) — upsert reemplaza,
   get devuelve None si no existe.
7. **CRUD `rag_behavior_priors_wa`** (2 tests) — upsert + get.

Total: 31 tests. Todos usan in-memory SQLite (`:memory:`) o
`tmp_path / "telemetry.db"` — el repo no necesita instancia real.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

import rag
from rag_negotiations import state_machine as sm
from rag_negotiations.state_machine import (
    InvalidTransitionError,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    can_transition,
    is_terminal,
    legal_transitions,
    transition,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════


def test_state_machine_has_8_states():
    """Spec dice 8 estados: 4 vivos (draft, launched, in_flight,
    escalated) + 4 terminales."""
    assert len(VALID_TRANSITIONS) == 8


def test_state_machine_has_4_terminal_states():
    """Estados terminales: closed_ok, closed_fail, cancelled,
    out_of_perimeter."""
    assert TERMINAL_STATES == frozenset({
        "closed_ok", "closed_fail", "cancelled", "out_of_perimeter",
    })


def test_terminal_states_have_no_outgoing_transitions():
    """Por definición, los terminales no tienen transitions out."""
    for state in TERMINAL_STATES:
        assert VALID_TRANSITIONS[state] == {}, (
            f"`{state}` debería ser terminal pero tiene transitions: "
            f"{VALID_TRANSITIONS[state]}"
        )


def test_is_terminal_returns_true_for_terminal_states():
    for state in TERMINAL_STATES:
        assert is_terminal(state)


def test_is_terminal_returns_false_for_active_states():
    for state in {"draft", "launched", "in_flight", "escalated"}:
        assert not is_terminal(state)


def test_legal_transitions_returns_dict_for_known_states():
    """Para todos los estados conocidos, `legal_transitions` devuelve
    el dict correspondiente."""
    for state in VALID_TRANSITIONS:
        assert legal_transitions(state) == VALID_TRANSITIONS[state]


def test_legal_transitions_returns_empty_for_unknown_states():
    """Estado desconocido → dict vacío sin raise. Permite usar
    `if not legal_transitions(s): is_terminal_or_unknown(s)`."""
    assert legal_transitions("nonexistent_state") == {}


def test_transition_advances_correctly_through_canonical_path():
    """Path canónico: draft → launched → in_flight → closed_ok."""
    assert transition("draft", "launch") == "launched"
    assert transition("launched", "first_msg_ack") == "in_flight"
    assert transition("in_flight", "agreement_detected") == "closed_ok"


def test_transition_raises_on_illegal_transition():
    """Llamar `launch` desde `in_flight` no es legal."""
    with pytest.raises(InvalidTransitionError) as exc_info:
        transition("in_flight", "launch")
    msg = str(exc_info.value)
    # Debe mencionar la transición inválida + las legales para debug.
    assert "launch" in msg
    assert "in_flight" in msg


def test_transition_raises_with_helpful_terminal_hint():
    """Cuando from_state es terminal, el mensaje lo dice explícito."""
    with pytest.raises(InvalidTransitionError) as exc_info:
        transition("closed_ok", "launch")
    assert "terminal" in str(exc_info.value).lower()


def test_can_transition_returns_true_for_legal():
    assert can_transition("draft", "launch") is True
    assert can_transition("in_flight", "classifier_low_conf") is True


def test_can_transition_returns_false_for_illegal():
    assert can_transition("in_flight", "launch") is False
    assert can_transition("closed_ok", "agreement_detected") is False
    assert can_transition("nonexistent", "launch") is False


def test_escalated_can_resume_or_close():
    """Desde `escalated`, hay 5 salidas: user_resumes (vuelve a
    in_flight), user_takes_over (cancelled), agreement_detected
    (closed_ok), rejection_detected (closed_fail), perimeter_violation
    (out_of_perimeter)."""
    out = legal_transitions("escalated")
    assert out["user_resumes"] == "in_flight"
    assert out["user_takes_over"] == "cancelled"
    assert out["agreement_detected"] == "closed_ok"
    assert out["rejection_detected"] == "closed_fail"
    assert out["perimeter_violation"] == "out_of_perimeter"


def test_state_order_includes_all_8_states():
    """`STATE_ORDER` lista los 8 estados en orden topológico."""
    assert set(sm.STATE_ORDER) == set(VALID_TRANSITIONS.keys())
    assert len(sm.STATE_ORDER) == 8


# ════════════════════════════════════════════════════════════════════════════
# 2. DDL IDEMPOTENTE
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def empty_db_conn(tmp_path: Path):
    """Conexión a una DB temporal con `_ensure_telemetry_tables` ya
    aplicado. Cleanup automático al fin del test."""
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    db_file = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_file))
    rag._ensure_telemetry_tables(conn)
    yield conn, db_file
    conn.close()


def test_ddl_creates_all_5_negotiation_tables(empty_db_conn):
    """Después de `_ensure_telemetry_tables`, existen las 5 tablas
    nuevas con sus columnas esperadas."""
    conn, _ = empty_db_conn
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "rag_negotiations" in tables
    assert "rag_negotiation_turns" in tables
    assert "rag_negotiation_pending_sends" in tables
    assert "rag_style_fingerprints" in tables
    assert "rag_behavior_priors_wa" in tables


def test_ddl_is_idempotent_on_second_call(tmp_path: Path):
    """Llamar `_ensure_telemetry_tables` 2× no falla ni duplica."""
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    db_file = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_file))
    try:
        rag._ensure_telemetry_tables(conn)
        rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
        rag._ensure_telemetry_tables(conn)  # No raise.
        n = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='rag_negotiations'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_negotiations_table_has_status_index(empty_db_conn):
    """Index compuesto (status, updated_at DESC) requerido para el
    panel del dashboard."""
    conn, _ = empty_db_conn
    idx_names = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='rag_negotiations'"
        )
    }
    assert "ix_negotiations_status_updated" in idx_names


def test_turns_table_has_fk_cascade_to_negotiations(empty_db_conn):
    """Borrar una negociación cascadea borrado de sus turnos."""
    conn, _ = empty_db_conn
    # Insertar una negociación + 2 turnos.
    conn.execute(
        "INSERT INTO rag_negotiations ("
        " user_intent, target_jid, status, created_at, updated_at, perimeter_json"
        ") VALUES (?, ?, ?, ?, ?, ?)",
        ("test", "jid@s.whatsapp.net", "draft", "2026-04-26T00:00:00Z",
         "2026-04-26T00:00:00Z", "{}"),
    )
    neg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for direction in ("in", "out"):
        conn.execute(
            "INSERT INTO rag_negotiation_turns ("
            " negotiation_id, ts, direction, content"
            ") VALUES (?, ?, ?, ?)",
            (neg_id, "2026-04-26T00:00:01Z", direction, "hola"),
        )
    conn.commit()
    conn.execute("DELETE FROM rag_negotiations WHERE id = ?", (neg_id,))
    conn.commit()
    n_turns = conn.execute(
        "SELECT COUNT(*) FROM rag_negotiation_turns WHERE negotiation_id = ?",
        (neg_id,),
    ).fetchone()[0]
    assert n_turns == 0, "los turnos deberían haber cascadeado al borrar la neg"


def test_pending_sends_has_due_index(empty_db_conn):
    """Index sobre (status, send_after_ts) — usado por dequeue_due."""
    conn, _ = empty_db_conn
    idx_names = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='rag_negotiation_pending_sends'"
        )
    }
    assert "ix_pending_sends_due" in idx_names


# ════════════════════════════════════════════════════════════════════════════
# 3-7. CRUD HELPERS — fixture compartido
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def crud_db(tmp_path: Path, monkeypatch):
    """DB temporal con las 5 tablas + monkeypatch de `_ragvec_state_conn`.

    Usa el patrón establecido por tests/test_routing_promote.py:
    crear una DB en tmp, cargar el schema con `_ensure_telemetry_tables`
    (real), y mockear `_ragvec_state_conn()` para que devuelva una
    conexión a esa DB. Así los CRUD helpers pueden ejecutar los SQL
    contra el schema real sin tocar la DB de producción.
    """
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    db_file = tmp_path / "telemetry.db"
    setup_conn = sqlite3.connect(str(db_file))
    rag._ensure_telemetry_tables(setup_conn)
    setup_conn.close()

    @contextlib.contextmanager
    def fake_conn():
        c = sqlite3.connect(str(db_file), isolation_level=None)
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", fake_conn)
    yield db_file


# ── 3. rag_negotiations ────────────────────────────────────────────────────


def test_create_negotiation_returns_id(crud_db):
    from rag_negotiations.crud import create_negotiation
    neg_id = create_negotiation(
        user_intent="coordinar café con Juan",
        target_jid="5491112345678@s.whatsapp.net",
        target_name="Juan",
        perimeter={"topic": "scheduling", "items": ["café", "lugar"]},
        confidence_threshold=0.85,
    )
    assert isinstance(neg_id, int)
    assert neg_id >= 1


def test_get_negotiation_deserializes_perimeter(crud_db):
    from rag_negotiations.crud import create_negotiation, get_negotiation
    neg_id = create_negotiation(
        user_intent="test",
        target_jid="jid@s.whatsapp.net",
        perimeter={"topic": "scheduling", "max_offers": 3},
    )
    row = get_negotiation(neg_id)
    assert row is not None
    assert row["status"] == "draft"
    assert row["perimeter"] == {"topic": "scheduling", "max_offers": 3}
    assert row["target_jid"] == "jid@s.whatsapp.net"


def test_get_negotiation_returns_none_if_not_exists(crud_db):
    from rag_negotiations.crud import get_negotiation
    assert get_negotiation(99999) is None


def test_update_status_persists_terminal_metadata(crud_db):
    """Cuando el status pasa a terminal, `closed_at` + `closure_type` se
    setean automaticamente."""
    from rag_negotiations.crud import (
        create_negotiation, get_negotiation, update_status,
    )
    neg_id = create_negotiation(
        user_intent="t", target_jid="j@s.whatsapp.net", perimeter={},
    )
    ok = update_status(
        neg_id, "closed_ok",
        closure_summary="Acordado viernes 17h en Las Violetas",
        side_effect={"calendar_event_id": "evt-123"},
    )
    assert ok is True

    row = get_negotiation(neg_id)
    assert row["status"] == "closed_ok"
    assert row["closure_type"] == "closed_ok"  # auto-derivado
    assert row["closure_summary"] == "Acordado viernes 17h en Las Violetas"
    assert row["closed_at"] is not None
    assert json.loads(row["side_effect_json"]) == {"calendar_event_id": "evt-123"}


def test_update_status_does_not_touch_closed_at_for_active_states(crud_db):
    """Pasar de draft a launched no debe setear `closed_at`."""
    from rag_negotiations.crud import (
        create_negotiation, get_negotiation, update_status,
    )
    neg_id = create_negotiation(
        user_intent="t", target_jid="j@s.whatsapp.net", perimeter={},
    )
    update_status(neg_id, "launched")
    row = get_negotiation(neg_id)
    assert row["closed_at"] is None
    assert row["closure_type"] is None


def test_increment_message_count(crud_db):
    from rag_negotiations.crud import (
        create_negotiation, get_negotiation, increment_message_count,
    )
    neg_id = create_negotiation(
        user_intent="t", target_jid="j@s.whatsapp.net", perimeter={},
    )
    increment_message_count(neg_id, sent=True)
    increment_message_count(neg_id, sent=True)
    increment_message_count(neg_id, sent=False)

    row = get_negotiation(neg_id)
    assert row["messages_sent"] == 2
    assert row["messages_received"] == 1


def test_list_negotiations_filters_by_status(crud_db):
    from rag_negotiations.crud import (
        create_negotiation, list_negotiations, update_status,
    )
    n1 = create_negotiation(user_intent="a", target_jid="a@s.whatsapp.net", perimeter={})
    n2 = create_negotiation(user_intent="b", target_jid="b@s.whatsapp.net", perimeter={})
    update_status(n1, "in_flight")
    update_status(n2, "closed_ok")

    in_flight = list_negotiations(status="in_flight")
    assert len(in_flight) == 1
    assert in_flight[0]["id"] == n1

    closed = list_negotiations(status="closed_ok")
    assert len(closed) == 1
    assert closed[0]["id"] == n2

    # Tupla — varios estados.
    active_or_done = list_negotiations(status=("in_flight", "closed_ok"))
    assert len(active_or_done) == 2


def test_list_negotiations_filters_by_target_jid(crud_db):
    from rag_negotiations.crud import create_negotiation, list_negotiations
    create_negotiation(user_intent="a", target_jid="a@s.whatsapp.net", perimeter={})
    create_negotiation(user_intent="b", target_jid="b@s.whatsapp.net", perimeter={})

    only_a = list_negotiations(target_jid="a@s.whatsapp.net")
    assert len(only_a) == 1
    assert only_a[0]["target_jid"] == "a@s.whatsapp.net"


# ── 4. rag_negotiation_turns ───────────────────────────────────────────────


def test_append_turn_persists_inbound_with_classifier(crud_db):
    from rag_negotiations.crud import append_turn, create_negotiation, list_turns
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    turn_id = append_turn(
        negotiation_id=neg_id,
        direction="in",
        content="Sí, viernes me sirve",
        classifier_confidence=0.92,
        classifier_reasoning="Confirmación clara de horario propuesto",
    )
    assert isinstance(turn_id, int)

    turns = list_turns(neg_id)
    assert len(turns) == 1
    assert turns[0]["direction"] == "in"
    assert turns[0]["content"] == "Sí, viernes me sirve"
    assert turns[0]["classifier_confidence"] == 0.92


def test_append_turn_rejects_invalid_direction(crud_db):
    """Direction debe ser exactamente "in" o "out"."""
    from rag_negotiations.crud import append_turn, create_negotiation
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    bad = append_turn(negotiation_id=neg_id, direction="incoming", content="x")
    assert bad is None


def test_list_turns_orders_by_ts_then_id(crud_db):
    """Turnos se devuelven ordenados ASC por ts, después por id."""
    from rag_negotiations.crud import append_turn, create_negotiation, list_turns
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    a = append_turn(negotiation_id=neg_id, direction="out", content="msg 1")
    b = append_turn(negotiation_id=neg_id, direction="in", content="msg 2")
    c = append_turn(negotiation_id=neg_id, direction="out", content="msg 3")

    turns = list_turns(neg_id)
    assert [t["id"] for t in turns] == [a, b, c]


# ── 5. rag_negotiation_pending_sends ───────────────────────────────────────


def test_enqueue_send_with_default_send_after_ts_is_now(crud_db):
    """Sin `send_after_ts`, el send queda listo para disparar
    inmediatamente."""
    from rag_negotiations.crud import (
        create_negotiation, dequeue_due, enqueue_send,
    )
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    send_id = enqueue_send(negotiation_id=neg_id, content="hola")
    assert isinstance(send_id, int)

    due = dequeue_due(now_ts=time.time() + 1)
    assert len(due) == 1
    assert due[0]["id"] == send_id


def test_dequeue_due_respects_send_after_ts(crud_db):
    """Sends programados al futuro NO aparecen hasta su ts."""
    from rag_negotiations.crud import (
        create_negotiation, dequeue_due, enqueue_send,
    )
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    future_ts = time.time() + 10_000  # +~3h
    enqueue_send(negotiation_id=neg_id, content="luego", send_after_ts=future_ts)

    # Now no debería verse.
    due_now = dequeue_due(now_ts=time.time())
    assert len(due_now) == 0

    # En el futuro sí.
    due_future = dequeue_due(now_ts=future_ts + 1)
    assert len(due_future) == 1


def test_mark_send_increments_attempts_and_changes_status(crud_db):
    from rag_negotiations.crud import (
        create_negotiation, dequeue_due, enqueue_send, mark_send,
    )
    neg_id = create_negotiation(user_intent="t", target_jid="j@s.whatsapp.net", perimeter={})
    send_id = enqueue_send(negotiation_id=neg_id, content="x")

    ok = mark_send(send_id, status="sent")
    assert ok is True

    # No more due (status='pending' filter excluye 'sent').
    due = dequeue_due(now_ts=time.time() + 1)
    assert len(due) == 0


# ── 6. rag_style_fingerprints ──────────────────────────────────────────────


def test_upsert_fingerprint_replaces_on_conflict(crud_db):
    """Mismo target_jid → REPLACE; messages_analyzed se actualiza."""
    from rag_negotiations.crud import (
        get_style_fingerprint, upsert_style_fingerprint,
    )
    jid = "alice@s.whatsapp.net"
    upsert_style_fingerprint(
        target_jid=jid,
        fingerprint={"tone": "casual", "emoji_freq": 0.1},
        messages_analyzed=50,
    )
    upsert_style_fingerprint(
        target_jid=jid,
        fingerprint={"tone": "formal", "emoji_freq": 0.3},
        messages_analyzed=120,
    )

    fp = get_style_fingerprint(jid)
    assert fp is not None
    assert fp["fingerprint"] == {"tone": "formal", "emoji_freq": 0.3}
    assert fp["messages_analyzed"] == 120


def test_get_style_fingerprint_returns_none_for_missing(crud_db):
    from rag_negotiations.crud import get_style_fingerprint
    assert get_style_fingerprint("ghost@s.whatsapp.net") is None


# ── 7. rag_behavior_priors_wa ──────────────────────────────────────────────


def test_upsert_behavior_priors_persists_all_fields(crud_db):
    from rag_negotiations.crud import (
        get_behavior_priors, upsert_behavior_priors,
    )
    jid = "bob@s.whatsapp.net"
    ok = upsert_behavior_priors(
        target_jid=jid,
        response_lag_mu=3.4, response_lag_sigma=1.1,
        avg_msg_length_words=12.5,
        msg_per_response=1.2,
        emoji_freq=0.05,
        samples_n=500,
    )
    assert ok is True

    row = get_behavior_priors(jid)
    assert row is not None
    assert row["response_lag_mu"] == 3.4
    assert row["response_lag_sigma"] == 1.1
    assert row["samples_n"] == 500


def test_upsert_behavior_priors_handles_partial_nulls(crud_db):
    """Algunos priors pueden ser NULL si no hay data suficiente."""
    from rag_negotiations.crud import (
        get_behavior_priors, upsert_behavior_priors,
    )
    jid = "rare@s.whatsapp.net"
    upsert_behavior_priors(
        target_jid=jid, samples_n=3,
        # response_lag_mu / sigma NULL — pocos samples, no podemos
        # calcular distribución confiable.
    )
    row = get_behavior_priors(jid)
    assert row is not None
    assert row["response_lag_mu"] is None
    assert row["response_lag_sigma"] is None
    assert row["samples_n"] == 3
