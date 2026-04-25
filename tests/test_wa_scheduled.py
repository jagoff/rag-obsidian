"""Tests para `rag.wa_scheduled` + endpoints `/api/whatsapp/scheduled*`
(Phase 4c, 2026-04-25 — programar mensajes de WhatsApp a futuro).

Cubre:

1. `wa_scheduled.schedule()` — happy path + validaciones (jid sin '@',
   message vacío, fecha en el pasado >60s, fecha >365d futuro, margen
   de 60s para clock skew).
2. `wa_scheduled.cancel()` — pending → cancelled, idempotente, no toca
   sent ni cancelled previos.
3. `wa_scheduled.list_scheduled()` — orden por scheduled_for_utc ASC
   cuando filtras pending, por created_at DESC sin filtro, limit cap,
   message_text truncado a 500 chars.
4. `wa_scheduled.reschedule()` — cambia scheduled_for_utc en pending,
   no toca sent, eleva ValueError con fecha en el pasado.
5. `wa_scheduled.run_due_worker()` — happy path (envío puntual),
   sent_late (>5min de delay), retry on failure (incrementa
   attempt_count → failed al llegar a max_retries), no procesa
   cancelled, dry_run no muta state.
6. `wa_scheduled._ensure_schema()` — llamadas múltiples idempotentes.
7. Endpoints HTTP — POST `/api/whatsapp/send` con `scheduled_for`
   (válido, garbage, en pasado), GET `/api/whatsapp/scheduled`,
   POST `.../cancel` y `.../reschedule`, helper
   `_parse_scheduled_for_to_utc`.

Aislamiento de la DB: monkey-patcheamos `rag._ragvec_state_conn` para
que apunte a un sqlite tmpfile per-test. El módulo `wa_scheduled` lo
importa lazy con `from rag import _ragvec_state_conn` adentro de cada
function body — el monkeypatch se propaga porque la lookup se resuelve
en tiempo de invocación. También neutralizamos
`wa_scheduled._log_ambient` para evitar escribir a `rag_ambient`
(otra vez vía la DB de prod).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

import rag
from rag import wa_scheduled


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_state_db(tmp_path, monkeypatch):
    """Sqlite tmpfile en lugar de telemetry.db real.

    Patch sobre `rag._ragvec_state_conn` (lookup lazy en `wa_scheduled`):
    cuando cualquier función del módulo abre una conn, cae acá. El path
    es per-test porque pytest crea un tmp_path único, así que no hay
    cross-talk entre tests aunque corran en paralelo.

    Neutralizamos `_log_ambient` para que las escrituras de auditoría
    no traten de tocar `rag_ambient` (que requeriría
    `_ensure_telemetry_tables` y desviaría el foco del SUT).
    """
    db_path = tmp_path / "state.db"

    @contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path), isolation_level=None,
                            check_same_thread=False, timeout=30.0)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    monkeypatch.setattr(wa_scheduled, "_log_ambient",
                        lambda *a, **kw: None)
    return db_path


def _future_iso(minutes: int = 60) -> str:
    """ISO8601 UTC `now + minutes`."""
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _past_iso(minutes: int = 60) -> str:
    """ISO8601 UTC `now - minutes` (para sembrar rows ya vencidas)."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _seed_row(db_path, *, scheduled_for_utc, status="pending",
              jid="123@s.whatsapp.net", message_text="hola",
              attempt_count=0):
    """INSERT directo en la tabla bypass `schedule()` — útil para
    sembrar status='cancelled'/'sent' o `scheduled_for_utc` en el
    pasado (que `schedule()` rechaza por contract)."""
    # Crear schema si no existe usando schedule() en una row dummy
    # futura, después update directo.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        # Usamos el DDL del módulo para no duplicar.
        for stmt in wa_scheduled._SCHEMA_DDL:
            conn.execute(stmt)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = conn.execute(
            f"INSERT INTO {wa_scheduled._TABLE} ("
            "created_at, scheduled_for_utc, jid, message_text, "
            "status, attempt_count, source"
            ") VALUES (?, ?, ?, ?, ?, ?, 'chat')",
            (now_iso, scheduled_for_utc, jid, message_text, status,
             attempt_count),
        )
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def _read_row(db_path, sid):
    """Read raw row (después de UPDATE/INSERT) — bypass list/get para
    no acoplar el assert al truncado de message_text que hace
    `list_scheduled`."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT * FROM {wa_scheduled._TABLE} WHERE id = ?", (sid,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── 1. schedule() happy path ────────────────────────────────────────────────


def test_schedule_happy_path_returns_dict_with_id(isolated_state_db):
    out = wa_scheduled.schedule(
        jid="5491155555555@s.whatsapp.net",
        message_text="hola grecia",
        scheduled_for_utc=_future_iso(60),
    )
    assert isinstance(out, dict)
    assert isinstance(out["id"], int) and out["id"] > 0
    assert out["status"] == "pending"
    assert "scheduled_for_utc" in out
    # La row se persistió correctamente.
    row = _read_row(isolated_state_db, out["id"])
    assert row is not None
    assert row["jid"] == "5491155555555@s.whatsapp.net"
    assert row["message_text"] == "hola grecia"
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0


def test_schedule_persists_optional_fields(isolated_state_db):
    """contact_name + reply_to + proposal_id + source = chat se
    persisten correctamente."""
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net",
        message_text="reply",
        scheduled_for_utc=_future_iso(120),
        contact_name="Grecia",
        reply_to={
            "message_id": "MID-abc",
            "original_text": "msg original",
            "sender_jid": "456@s.whatsapp.net",
        },
        proposal_id="prop-xyz",
        source="dashboard",
    )
    row = _read_row(isolated_state_db, out["id"])
    assert row["contact_name"] == "Grecia"
    assert row["reply_to_id"] == "MID-abc"
    assert row["reply_to_text"] == "msg original"
    assert row["reply_to_sender_jid"] == "456@s.whatsapp.net"
    assert row["proposal_id"] == "prop-xyz"
    assert row["source"] == "dashboard"


# ── 2. schedule() validaciones ──────────────────────────────────────────────


def test_schedule_rejects_jid_without_at(isolated_state_db):
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="5491155555555",  # sin @
            message_text="hola",
            scheduled_for_utc=_future_iso(60),
        )


def test_schedule_rejects_empty_jid(isolated_state_db):
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="",
            message_text="hola",
            scheduled_for_utc=_future_iso(60),
        )


def test_schedule_rejects_empty_message(isolated_state_db):
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="123@s.whatsapp.net",
            message_text="",
            scheduled_for_utc=_future_iso(60),
        )


def test_schedule_rejects_whitespace_only_message(isolated_state_db):
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="123@s.whatsapp.net",
            message_text="   \n\t  ",
            scheduled_for_utc=_future_iso(60),
        )


def test_schedule_rejects_past_date_beyond_skew(isolated_state_db):
    """Fecha en el pasado >60s rechazada por anti-acumulación."""
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="123@s.whatsapp.net",
            message_text="hola",
            scheduled_for_utc=_past_iso(5),  # 5 min atrás >> 60s
        )


def test_schedule_accepts_within_clock_skew_tolerance(isolated_state_db):
    """Fecha 30s en el pasado → OK (margen de 60s para clock skew)."""
    near_past = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(timespec="seconds")
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net",
        message_text="hola",
        scheduled_for_utc=near_past,
    )
    assert out["status"] == "pending"
    assert out["id"] > 0


def test_schedule_rejects_too_far_future(isolated_state_db):
    """Fecha >365d en el futuro → rechazada."""
    too_far = (datetime.now(timezone.utc) + timedelta(days=400)).isoformat(timespec="seconds")
    with pytest.raises(ValueError):
        wa_scheduled.schedule(
            jid="123@s.whatsapp.net",
            message_text="hola",
            scheduled_for_utc=too_far,
        )


# ── 3. cancel() flow ────────────────────────────────────────────────────────


def test_cancel_pending_returns_true(isolated_state_db):
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net",
        message_text="hola",
        scheduled_for_utc=_future_iso(60),
    )
    sid = out["id"]
    assert wa_scheduled.cancel(sid) is True
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "cancelled"
    assert row["last_error"] and row["last_error"].startswith("cancelled:")


def test_cancel_already_cancelled_is_idempotent(isolated_state_db):
    """Segunda cancel() → False, no doble UPDATE."""
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net",
        message_text="hola",
        scheduled_for_utc=_future_iso(60),
    )
    sid = out["id"]
    assert wa_scheduled.cancel(sid) is True
    assert wa_scheduled.cancel(sid) is False
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "cancelled"  # no cambió


def test_cancel_nonexistent_id_returns_false(isolated_state_db):
    assert wa_scheduled.cancel(999_999) is False


def test_cancel_already_sent_returns_false(isolated_state_db):
    """No se permite cancelar un mensaje ya enviado."""
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(120),
        status="sent",
    )
    assert wa_scheduled.cancel(sid) is False
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "sent"  # no cambió


# ── 4. list_scheduled() ordering + filtering ────────────────────────────────


def test_list_pending_ordered_by_scheduled_for_asc(isolated_state_db):
    """Pending viene ordenado por scheduled_for_utc ASC (urgente primero)."""
    sid_30 = wa_scheduled.schedule(
        jid="a@s.whatsapp.net", message_text="msg30",
        scheduled_for_utc=_future_iso(30),
    )["id"]
    sid_120 = wa_scheduled.schedule(
        jid="b@s.whatsapp.net", message_text="msg120",
        scheduled_for_utc=_future_iso(120),
    )["id"]
    sid_60 = wa_scheduled.schedule(
        jid="c@s.whatsapp.net", message_text="msg60",
        scheduled_for_utc=_future_iso(60),
    )["id"]
    items = wa_scheduled.list_scheduled(status="pending")
    assert [r["id"] for r in items] == [sid_30, sid_60, sid_120]


def test_list_no_filter_returns_all_statuses(isolated_state_db):
    """Sin filtro de status: devuelve todos los rows (pending + sent + cancelled)."""
    # 3 pending
    for m in (30, 60, 120):
        wa_scheduled.schedule(
            jid="x@s.whatsapp.net", message_text=f"msg{m}",
            scheduled_for_utc=_future_iso(m),
        )
    # 1 sent + 1 cancelled (vía seed directo, status custom)
    _seed_row(isolated_state_db, scheduled_for_utc=_past_iso(60), status="sent")
    _seed_row(isolated_state_db, scheduled_for_utc=_future_iso(200), status="cancelled")
    items = wa_scheduled.list_scheduled()
    assert len(items) == 5


def test_list_filter_by_status_sent(isolated_state_db):
    wa_scheduled.schedule(
        jid="a@s.whatsapp.net", message_text="future",
        scheduled_for_utc=_future_iso(60),
    )
    _seed_row(isolated_state_db, scheduled_for_utc=_past_iso(60), status="sent")
    items_sent = wa_scheduled.list_scheduled(status="sent")
    assert len(items_sent) == 1
    assert items_sent[0]["status"] == "sent"


def test_list_limit_caps_results(isolated_state_db):
    """`limit=2` solo devuelve 2."""
    for m in (30, 60, 120, 180):
        wa_scheduled.schedule(
            jid="x@s.whatsapp.net", message_text=f"msg{m}",
            scheduled_for_utc=_future_iso(m),
        )
    items = wa_scheduled.list_scheduled(status="pending", limit=2)
    assert len(items) == 2


def test_list_truncates_message_text_to_500_chars(isolated_state_db):
    """Mensajes >500 chars vienen truncados en list (no inflar API)."""
    long_text = "a" * 1200
    out = wa_scheduled.schedule(
        jid="x@s.whatsapp.net", message_text=long_text,
        scheduled_for_utc=_future_iso(60),
    )
    items = wa_scheduled.list_scheduled(status="pending")
    matching = [r for r in items if r["id"] == out["id"]]
    assert len(matching) == 1
    assert len(matching[0]["message_text"]) == 500
    # `get_scheduled` SÍ devuelve el texto completo.
    full = wa_scheduled.get_scheduled(out["id"])
    assert len(full["message_text"]) == 1200


# ── 5. reschedule() flow ────────────────────────────────────────────────────


def test_reschedule_pending_changes_date(isolated_state_db):
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net", message_text="hola",
        scheduled_for_utc=_future_iso(60),
    )
    new_iso = _future_iso(240)
    assert wa_scheduled.reschedule(out["id"], new_iso) is True
    row = _read_row(isolated_state_db, out["id"])
    # `_iso_utc` reformatea con `+00:00` — comparamos parseando.
    new_dt = datetime.fromisoformat(new_iso)
    db_dt = datetime.fromisoformat(row["scheduled_for_utc"])
    assert abs((db_dt - new_dt).total_seconds()) < 1.0
    assert row["status"] == "pending"


def test_reschedule_sent_returns_false(isolated_state_db):
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(60),
        status="sent",
    )
    assert wa_scheduled.reschedule(sid, _future_iso(60)) is False


def test_reschedule_to_past_raises_value_error(isolated_state_db):
    """Misma validación que schedule() — no aceptamos pasado >60s."""
    out = wa_scheduled.schedule(
        jid="123@s.whatsapp.net", message_text="hola",
        scheduled_for_utc=_future_iso(60),
    )
    with pytest.raises(ValueError):
        wa_scheduled.reschedule(out["id"], _past_iso(10))


# ── 6. run_due_worker() happy path ──────────────────────────────────────────


def test_run_due_worker_sends_pending_due_rows(isolated_state_db, monkeypatch):
    """2 pending vencidos (5min atrás) + 1 pending futuro (10min adelante).
    El worker procesa los 2 vencidos y deja el futuro intacto."""
    sid_a = _seed_row(isolated_state_db,
                      scheduled_for_utc=_past_iso(5),
                      jid="a@s.whatsapp.net", message_text="due A")
    sid_b = _seed_row(isolated_state_db,
                      scheduled_for_utc=_past_iso(3),
                      jid="b@s.whatsapp.net", message_text="due B")
    sid_future = _seed_row(isolated_state_db,
                           scheduled_for_utc=_future_iso(10),
                           jid="c@s.whatsapp.net", message_text="future C")

    sent_calls = []

    def _fake_send(jid, text, *, anti_loop=True, reply_to=None):
        sent_calls.append({"jid": jid, "text": text, "anti_loop": anti_loop,
                           "reply_to": reply_to})
        return True

    import rag.integrations.whatsapp as _waint
    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid", _fake_send)

    summary = wa_scheduled.run_due_worker()
    assert summary["ok"] is True
    assert summary["processed"] == 2
    assert summary["sent"] == 2
    assert summary["sent_late"] == 0
    assert summary["failed"] == 0
    # Los 2 due rows pasaron a 'sent' con sent_at populado.
    for sid in (sid_a, sid_b):
        row = _read_row(isolated_state_db, sid)
        assert row["status"] == "sent"
        assert row["sent_at"] is not None
        assert row["delta_minutes"] is not None
    # El futuro queda intacto.
    row_future = _read_row(isolated_state_db, sid_future)
    assert row_future["status"] == "pending"
    assert row_future["sent_at"] is None
    # Send fue llamado 2 veces.
    assert len(sent_calls) == 2


# ── 7. run_due_worker() late path (Mac dormida) ─────────────────────────────


def test_run_due_worker_late_path(isolated_state_db, monkeypatch):
    """Pending con scheduled_for 60min atrás → status='sent_late'."""
    sid = _seed_row(isolated_state_db,
                    scheduled_for_utc=_past_iso(60),
                    jid="a@s.whatsapp.net", message_text="late msg")

    import rag.integrations.whatsapp as _waint
    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid",
                        lambda *a, **kw: True)

    summary = wa_scheduled.run_due_worker()
    assert summary["processed"] == 1
    assert summary["sent"] == 0
    assert summary["sent_late"] == 1
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "sent_late"
    assert row["delta_minutes"] >= 60


# ── 8. run_due_worker() retry on failure ────────────────────────────────────


def test_run_due_worker_retries_on_failure_until_max(isolated_state_db, monkeypatch):
    """Send falla 5 veces → attempt_count llega a 5 y status='failed'."""
    sid = _seed_row(isolated_state_db,
                    scheduled_for_utc=_past_iso(2),
                    jid="a@s.whatsapp.net", message_text="will fail")

    import rag.integrations.whatsapp as _waint
    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid",
                        lambda *a, **kw: False)

    # Run 1: attempt_count=1, status sigue pending, last_error populado.
    s1 = wa_scheduled.run_due_worker()
    assert s1["retried"] == 1
    assert s1["failed"] == 0
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "pending"
    assert row["attempt_count"] == 1
    assert row["last_error"] == "send_failed"

    # Runs 2-4: attempt_count llega a 4, sigue pending.
    for expected in (2, 3, 4):
        wa_scheduled.run_due_worker()
        row = _read_row(isolated_state_db, sid)
        assert row["status"] == "pending"
        assert row["attempt_count"] == expected

    # Run 5: attempt_count=5, status='failed'.
    s5 = wa_scheduled.run_due_worker()
    assert s5["failed"] == 1
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "failed"
    assert row["attempt_count"] == 5
    assert row["last_error"] == "send_failed"


# ── 9. run_due_worker() no procesa cancelled ────────────────────────────────


def test_run_due_worker_skips_cancelled(isolated_state_db, monkeypatch):
    """Row cancelled vencido → no se procesa ni se manda."""
    sid = _seed_row(isolated_state_db,
                    scheduled_for_utc=_past_iso(10),
                    status="cancelled",
                    jid="a@s.whatsapp.net", message_text="cancelled")

    sent_calls = []

    import rag.integrations.whatsapp as _waint

    def _spy(*a, **kw):
        sent_calls.append((a, kw))
        return True

    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid", _spy)

    summary = wa_scheduled.run_due_worker()
    assert summary["processed"] == 0
    assert summary["sent"] == 0
    assert summary["sent_late"] == 0
    assert sent_calls == []
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "cancelled"  # intacto


# ── 10. run_due_worker() dry_run ────────────────────────────────────────────


def test_run_due_worker_dry_run_no_side_effects(isolated_state_db, monkeypatch):
    sid = _seed_row(isolated_state_db,
                    scheduled_for_utc=_past_iso(2),
                    jid="a@s.whatsapp.net", message_text="dry")

    sent_calls = []

    import rag.integrations.whatsapp as _waint

    def _spy(*a, **kw):
        sent_calls.append((a, kw))
        return True

    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid", _spy)

    summary = wa_scheduled.run_due_worker(dry_run=True)
    assert summary["processed"] == 1
    assert summary["sent"] == 0
    assert summary["sent_late"] == 0
    assert summary["failed"] == 0
    assert summary["retried"] == 0
    # No se llamó al sender.
    assert sent_calls == []
    # La row sigue pending — sin sent_at, attempt_count=0.
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0
    assert row["sent_at"] is None


# ── 11. _ensure_schema() idempotente ────────────────────────────────────────


def test_ensure_schema_is_idempotent(isolated_state_db):
    """5 invocaciones consecutivas → no crash, no duplica indices."""
    conn = sqlite3.connect(str(isolated_state_db), isolation_level=None)
    try:
        for _ in range(5):
            wa_scheduled._ensure_schema(conn)
        # Verificar que existe la tabla y la cantidad esperada de
        # indices (los `CREATE INDEX IF NOT EXISTS` no se duplican).
        idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            f"AND tbl_name='{wa_scheduled._TABLE}' "
            "AND name NOT LIKE 'sqlite_autoindex_%'"
        ).fetchall()
        assert len(idx) == 2
    finally:
        conn.close()


# ── 12-17. Endpoints (FastAPI TestClient) ───────────────────────────────────


@pytest.fixture
def http_client(isolated_state_db):
    """TestClient que comparte la tmp DB con `isolated_state_db`.

    El TestClient se crea por test (no module-level) para que el
    monkeypatch de `_ragvec_state_conn` ya esté activo cuando los
    endpoints ejecuten `from rag import wa_scheduled` y deriven
    rows de la tabla.
    """
    from fastapi.testclient import TestClient
    import web.server as _server
    return TestClient(_server.app)


def test_endpoint_send_with_valid_scheduled_for_returns_200(http_client):
    """POST /api/whatsapp/send con scheduled_for válido → 200,
    response tiene `scheduled: true, id, scheduled_for_utc`."""
    future_ar = (datetime.now(timezone.utc) + timedelta(hours=2)).astimezone(
        timezone(timedelta(hours=-3))
    ).isoformat(timespec="seconds")
    resp = http_client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "msg programado",
        "scheduled_for": future_ar,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["scheduled"] is True
    assert isinstance(body["id"], int)
    assert "scheduled_for_utc" in body


def test_endpoint_send_with_garbage_scheduled_for_returns_400(http_client):
    resp = http_client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola",
        "scheduled_for": "garbage",
    })
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    assert "scheduled_for" in detail and ("inválido" in detail or "invalido" in detail)


def test_endpoint_send_with_past_scheduled_for_returns_400(http_client):
    """scheduled_for de hace 1 día → wa_scheduled.schedule() levanta
    ValueError → endpoint traduce a 400."""
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    resp = http_client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola",
        "scheduled_for": past,
    })
    assert resp.status_code == 400


def test_endpoint_list_scheduled_pending(http_client, isolated_state_db):
    """GET /api/whatsapp/scheduled?status=pending → 200, count=2."""
    sid_a = wa_scheduled.schedule(
        jid="a@s.whatsapp.net", message_text="A",
        scheduled_for_utc=_future_iso(30),
    )["id"]
    sid_b = wa_scheduled.schedule(
        jid="b@s.whatsapp.net", message_text="B",
        scheduled_for_utc=_future_iso(60),
    )["id"]

    resp = http_client.get("/api/whatsapp/scheduled?status=pending")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["items"]) == 2
    # Orden por scheduled_for_utc ASC: A (30min) primero.
    assert [r["id"] for r in body["items"]] == [sid_a, sid_b]


def test_endpoint_cancel_flow(http_client, isolated_state_db):
    """POST /api/whatsapp/scheduled/{id}/cancel — primer call cancela,
    segundo idempotente con `ok: false, reason: ...`."""
    out = wa_scheduled.schedule(
        jid="x@s.whatsapp.net", message_text="msg",
        scheduled_for_utc=_future_iso(60),
    )
    sid = out["id"]
    r1 = http_client.post(f"/api/whatsapp/scheduled/{sid}/cancel")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["ok"] is True
    assert body1["status"] == "cancelled"
    # Segundo intento — idempotente.
    r2 = http_client.post(f"/api/whatsapp/scheduled/{sid}/cancel")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["ok"] is False
    assert body2["reason"] == "not_pending_or_not_found"


def test_endpoint_reschedule_changes_date(http_client, isolated_state_db):
    """POST /api/whatsapp/scheduled/{id}/reschedule con scheduled_for
    válido → 200, scheduled_for_utc nuevo."""
    out = wa_scheduled.schedule(
        jid="x@s.whatsapp.net", message_text="msg",
        scheduled_for_utc=_future_iso(30),
    )
    sid = out["id"]
    new_iso = _future_iso(180)
    resp = http_client.post(
        f"/api/whatsapp/scheduled/{sid}/reschedule",
        json={"scheduled_for": new_iso},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["id"] == sid
    # Verificar persistencia.
    row = _read_row(isolated_state_db, sid)
    assert datetime.fromisoformat(row["scheduled_for_utc"]) > datetime.now(timezone.utc) + timedelta(minutes=120)


# ── 18. _parse_scheduled_for_to_utc helper unit test ────────────────────────


def test_parse_scheduled_for_with_argentina_offset():
    import web.server as _server
    out = _server._parse_scheduled_for_to_utc("2099-01-01T09:00:00-03:00")
    # Argentina 09:00 → UTC 12:00.
    assert out == "2099-01-01T12:00:00+00:00"


def test_parse_scheduled_for_without_offset_assumes_argentina():
    import web.server as _server
    out = _server._parse_scheduled_for_to_utc("2099-01-01T09:00:00")
    # Sin offset → asume AR (-03:00) → UTC 12:00.
    assert out == "2099-01-01T12:00:00+00:00"


def test_parse_scheduled_for_with_z_suffix_kept_as_utc():
    import web.server as _server
    out = _server._parse_scheduled_for_to_utc("2099-01-01T09:00:00Z")
    assert out == "2099-01-01T09:00:00+00:00"


def test_parse_scheduled_for_garbage_raises_value_error():
    import web.server as _server
    with pytest.raises(ValueError):
        _server._parse_scheduled_for_to_utc("garbage")


# ── Plist registration / XML validity ───────────────────────────────────
#
# Tests para evitar el bug del 2026-04-25: el plist de wa-scheduled-send
# se shippeo en commit 9740fa1 pero al user le quedó NO instalado porque
# yo lo dejé como TODO ("corré rag setup"). Estos tests garantizan que
# (1) el plist está registrado en la lista que `rag setup` consume — si
# alguien refactorea la lista y se olvida del nuestro, el test grita;
# (2) el XML generado por `_wa_scheduled_send_plist` es válido (parseable
# como plist real) y tiene los campos esperados — si alguien rompe el
# template (mismatched braces, key fuera de orden, etc.), el test grita
# antes de que un launchctl bootstrap real falle silenciosamente.


def test_wa_scheduled_send_plist_is_registered_in_services_spec():
    """`com.fer.obsidian-rag-wa-scheduled-send` debe estar en la lista
    que `rag setup` consume para instalar los daemons. Sin este registro
    el plist nunca se copia a ~/Library/LaunchAgents/ y el worker no
    arranca.
    """
    import rag as _rag
    spec = _rag._services_spec("/Users/fer/.local/bin/rag")
    labels = [t[0] for t in spec]
    assert "com.fer.obsidian-rag-wa-scheduled-send" in labels, (
        "El plist del worker de mensajes WhatsApp programados no está "
        "registrado en _services_spec — `rag setup` no lo va a instalar. "
        f"Plists registrados: {labels}"
    )
    # Triple sanity-check: la tupla tiene la forma (label, filename, xml).
    entry = next(t for t in spec if t[0] == "com.fer.obsidian-rag-wa-scheduled-send")
    assert len(entry) == 3
    assert entry[1] == "com.fer.obsidian-rag-wa-scheduled-send.plist"
    assert "<?xml" in entry[2]


def test_wa_scheduled_send_plist_xml_is_valid_and_has_expected_fields():
    """El XML generado por `_wa_scheduled_send_plist(rag_bin)` debe ser
    parseable por plistlib (== launchd lo va a aceptar al bootstrap)
    y tener los campos críticos: Label correcto, ProgramArguments con
    el subcomando, StartInterval=300, paths de log apuntando a
    `_RAG_LOG_DIR`.
    """
    import plistlib
    import rag as _rag

    rag_bin = "/Users/fer/.local/bin/rag"
    xml = _rag._wa_scheduled_send_plist(rag_bin)

    # 1) plistlib parsea sin excepción (== XML válido + plist syntax).
    parsed = plistlib.loads(xml.encode("utf-8"))

    # 2) Campos requeridos por launchd:
    assert parsed["Label"] == "com.fer.obsidian-rag-wa-scheduled-send"
    assert parsed["ProgramArguments"] == [rag_bin, "wa-scheduled-send"]

    # 3) Cadencia: 300s = 5min (decisión del user 2026-04-25, igual al
    # cron de reminder-wa-push). Si alguien cambia esto sin pensar, los
    # mensajes programados pueden llegar tarde por minutos.
    assert parsed["StartInterval"] == 300

    # 4) RunAtLoad=False — no queremos que un push de código nuevo
    # spamee mensajes pendientes inesperadamente al daemon recargar.
    assert parsed.get("RunAtLoad") is False

    # 5) Logs apuntan al log dir del proyecto (no a /tmp ni similar).
    # Tolerante: el dir exacto depende de _RAG_LOG_DIR pero el filename
    # debe ser consistente.
    assert parsed["StandardOutPath"].endswith("/wa-scheduled-send.log")
    assert parsed["StandardErrorPath"].endswith("/wa-scheduled-send.error.log")

    # 6) Env vars mínimos para que el subcomando corra fuera de un
    # shell interactivo: HOME, PATH, NO_COLOR, TERM=dumb (sin esto,
    # rich rompería en stderr porque cree que tiene TTY).
    env = parsed["EnvironmentVariables"]
    assert "HOME" in env
    assert "PATH" in env
    assert env.get("NO_COLOR") == "1"
    assert env.get("TERM") == "dumb"


def test_wa_scheduled_send_plist_uses_correct_rag_binary():
    """El plist debe ejecutar EXACTAMENTE el rag_bin que se le pasa.
    Si rag se mueve (ej. de ~/.local/bin a /opt/homebrew/bin tras
    `uv tool install`), el path en el plist debe reflejarlo. Bug típico:
    hardcodear el path en lugar de usar el arg.
    """
    import plistlib
    import rag as _rag
    custom_bin = "/opt/custom/path/to/rag"
    xml = _rag._wa_scheduled_send_plist(custom_bin)
    parsed = plistlib.loads(xml.encode("utf-8"))
    assert parsed["ProgramArguments"][0] == custom_bin
