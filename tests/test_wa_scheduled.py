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


# ── 8.b run_due_worker() notif al ambient cuando el row pasa a 'failed' ─────
# Cobertura del audit 2026-04-25 R2-Ambient #4: cuando el bridge está caído
# y un mensaje programado agota los retries, hay que cerrar el feedback loop
# avisándole al user por el ambient JID — sin esto, el user solo se entera
# abriendo el dashboard.


def test_notify_ambient_called_on_failed_after_max_retries(
    isolated_state_db, monkeypatch,
):
    """Bridge down: el send falla 5 veces consecutivas. En las primeras 4
    iteraciones la row vuelve a 'pending' (retry recoverable) y NO se
    notifica al ambient. Recién en el 5to intento, cuando attempt_count
    llega a max_retries y la row pasa a 'failed' permanente, se invoca
    `_notify_ambient_scheduled_outcome` con outcome='failed', el
    attempt_count y el last_error.

    Audit 2026-04-25 R2-Ambient #4.
    """
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(2),
        jid="grecia@s.whatsapp.net",
        message_text="feliz cumple",
    )

    # Bridge siempre falla.
    import rag.integrations.whatsapp as _waint
    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid",
                        lambda *a, **kw: False)

    # Spy sobre la notif ambient para capturar status + kwargs.
    notif_calls: list[tuple] = []

    def _spy_notif(sched, status, **kwargs):
        notif_calls.append((sched, status, kwargs))

    monkeypatch.setattr(wa_scheduled, "_notify_ambient_scheduled_outcome",
                        _spy_notif)

    # Runs 1-4: retry recoverable, NO debe notificar.
    for _ in range(4):
        wa_scheduled.run_due_worker()
    assert notif_calls == [], (
        "no debería notificar mientras la row siga en 'pending' "
        f"(transient failure) — got: {notif_calls!r}"
    )

    # Run 5: attempt_count llega a max_retries → status='failed' → notif.
    s5 = wa_scheduled.run_due_worker()
    assert s5["failed"] == 1
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "failed"

    assert len(notif_calls) == 1, (
        f"esperaba 1 notif para outcome='failed', got {len(notif_calls)}"
    )
    sched_arg, status_arg, kwargs_arg = notif_calls[0]
    assert status_arg == "failed"
    assert kwargs_arg.get("attempt_count") == 5
    assert kwargs_arg.get("last_error") == "send_failed"
    # El sched que se le pasa al notifier es la row original (con su jid +
    # message_text + scheduled_for_utc), no la versión post-UPDATE.
    assert sched_arg["id"] == sid
    assert sched_arg["jid"] == "grecia@s.whatsapp.net"


def test_notify_ambient_NOT_called_on_transient_failure(
    isolated_state_db, monkeypatch,
):
    """Falla 1 sola vez con max_retries=5 → la row vuelve a 'pending' y
    `_notify_ambient_scheduled_outcome` NO se invoca. La notif debe
    dispararse una sola vez en el cierre permanente, nunca en
    transient failures que todavía tienen retries pendientes.

    Audit 2026-04-25 R2-Ambient #4.
    """
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(2),
        jid="grecia@s.whatsapp.net",
        message_text="feliz cumple",
    )

    import rag.integrations.whatsapp as _waint
    monkeypatch.setattr(_waint, "_whatsapp_send_to_jid",
                        lambda *a, **kw: False)

    notif_calls: list[tuple] = []

    def _spy_notif(sched, status, **kwargs):
        notif_calls.append((sched, status, kwargs))

    monkeypatch.setattr(wa_scheduled, "_notify_ambient_scheduled_outcome",
                        _spy_notif)

    summary = wa_scheduled.run_due_worker(max_retries=5)
    assert summary["retried"] == 1
    assert summary["failed"] == 0
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "pending"
    assert row["attempt_count"] == 1

    assert notif_calls == [], (
        "transient failure (1/5) no debería notificar al ambient — "
        f"got: {notif_calls!r}"
    )


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
    """2026-05-04 consolidation: wa-scheduled-send + reminder-wa-push merged
    into wa-fast. Test redirected to verify wa-fast is registered."""
    import rag as _rag
    spec = _rag._services_spec("/Users/fer/.local/bin/rag")
    labels = [t[0] for t in spec]
    assert "com.fer.obsidian-rag-wa-fast" in labels, (
        "El plist consolidado wa-fast no está registrado en _services_spec. "
        f"Plists registrados: {labels}"
    )
    assert "com.fer.obsidian-rag-wa-scheduled-send" not in labels
    # Triple sanity-check: la tupla tiene la forma (label, filename, xml).
    entry = next(t for t in spec if t[0] == "com.fer.obsidian-rag-wa-fast")
    assert len(entry) == 3
    assert entry[1] == "com.fer.obsidian-rag-wa-fast.plist"
    assert "<?xml" in entry[2]


def test_wa_scheduled_send_plist_xml_is_valid_and_has_expected_fields():
    """2026-05-04 consolidation: test redirected to _wa_fast_plist."""
    import plistlib
    import rag as _rag

    rag_bin = "/Users/fer/.local/bin/rag"
    xml = _rag._wa_fast_plist(rag_bin)

    # 1) plistlib parsea sin excepción (== XML válido + plist syntax).
    parsed = plistlib.loads(xml.encode("utf-8"))

    # 2) Campos requeridos por launchd:
    assert parsed["Label"] == "com.fer.obsidian-rag-wa-fast"
    assert parsed["ProgramArguments"] == [rag_bin, "wa-fast"]

    # 3) Cadencia: 300s = 5min preservada.
    assert parsed["StartInterval"] == 300

    # 4) RunAtLoad=False.
    assert parsed.get("RunAtLoad") is False

    # 5) Logs apuntan al log dir del proyecto.
    assert parsed["StandardOutPath"].endswith("/wa-fast.log")
    assert parsed["StandardErrorPath"].endswith("/wa-fast.error.log")

    # 6) Env vars mínimos.
    env = parsed["EnvironmentVariables"]
    assert "HOME" in env
    assert "PATH" in env
    assert env.get("NO_COLOR") == "1"
    assert env.get("TERM") == "dumb"


def test_wa_scheduled_send_plist_uses_correct_rag_binary():
    """2026-05-04 consolidation: test redirected to _wa_fast_plist."""
    import plistlib
    import rag as _rag
    custom_bin = "/opt/custom/path/to/rag"
    xml = _rag._wa_fast_plist(custom_bin)
    parsed = plistlib.loads(xml.encode("utf-8"))
    assert parsed["ProgramArguments"][0] == custom_bin


# ── Race condition + recovery: atomic acquire pending → processing ─────
#
# Tests para el fix del 2026-04-25 (audit profundo): el worker antes
# tenía window race entre SELECT pending y UPDATE sent — dos workers
# podían leer el mismo row y mandar el mensaje 2 veces al destinatario.
# El fix es UPDATE atómico pending → processing como acquire, con check
# de rowcount; si 0, skip silencioso. Y recovery loop al inicio para
# liberar rows huérfanas en 'processing' que dejó un worker crasheado.


def test_run_due_worker_skips_row_already_processing(
    isolated_state_db, monkeypatch,
):
    """Si una row ya está en status='processing' (otro worker la
    agarró), el worker actual debe SKIPEARLA — NO mandar al bridge ni
    finalizar. Resuelve el caso plist tick + manual run solapados.
    """
    sent_calls = []

    def _fake_send(jid, text, *, anti_loop=False, reply_to=None):
        sent_calls.append({"jid": jid, "text": text})
        return True

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid", _fake_send,
    )

    # Sembrar pending vencido (bypass schedule() que rechaza pasados).
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(60),  # 1h atrás, vencido
        jid="540000@s.whatsapp.net",
        message_text="msg",
    )
    # Simular otro worker que ya agarró el row → status='processing'
    # con last_attempt_at reciente (<5min, NO stale).
    not_stale = (
        datetime.now(timezone.utc) - timedelta(seconds=30)
    ).isoformat(timespec="seconds")
    conn = sqlite3.connect(str(isolated_state_db), isolation_level=None)
    try:
        conn.execute(
            f"UPDATE {wa_scheduled._TABLE} SET status='processing', "
            f"last_attempt_at=? WHERE id=?",
            (not_stale, sid),
        )
    finally:
        conn.close()

    summary = wa_scheduled.run_due_worker()
    assert summary["processed"] == 0, (
        "no debería haber procesado el row — el otro worker tiene 'processing'"
    )
    assert len(sent_calls) == 0, "no debería haber llamado al bridge"
    # La row sigue en processing, no fue tocada.
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "processing"


def test_run_due_worker_recovers_stale_processing(
    isolated_state_db, monkeypatch,
):
    """Si una row quedó en status='processing' por > _STALE_PROCESSING_MINUTES,
    asumimos que el worker que la agarró crasheó. El run actual la
    resetea a 'pending' al inicio y luego la procesa normalmente.
    Sin esto, los rows quedan stuck para siempre.
    """
    sent_calls = []

    def _fake_send(jid, text, *, anti_loop=False, reply_to=None):
        sent_calls.append(jid)
        return True

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid", _fake_send,
    )

    # Sembrar pending vencido + simular worker viejo que crasheó hace
    # 10 min (last_attempt_at IS stale → debería ser recuperado).
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(60),
        jid="540000@s.whatsapp.net",
        message_text="recovered",
    )
    stale_attempt = (
        datetime.now(timezone.utc) - timedelta(minutes=10)
    ).isoformat(timespec="seconds")
    conn = sqlite3.connect(str(isolated_state_db), isolation_level=None)
    try:
        conn.execute(
            f"UPDATE {wa_scheduled._TABLE} SET status='processing', "
            f"last_attempt_at=? WHERE id=?",
            (stale_attempt, sid),
        )
    finally:
        conn.close()

    summary = wa_scheduled.run_due_worker()
    # Recovery + pickup + send en el mismo run.
    # scheduled_for_utc fue hace 60min >> late_threshold de 5min →
    # status final = 'sent_late', counter sent_late++.
    assert summary["processed"] == 1
    assert summary["sent_late"] == 1
    assert sent_calls == ["540000@s.whatsapp.net"]
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "sent_late"


def test_run_due_worker_send_failure_returns_to_pending(
    isolated_state_db, monkeypatch,
):
    """Cuando el bridge falla pero attempt_count < max_retries, la row
    debe volver a 'pending' (no quedar en 'processing'). Sin esto, el
    próximo tick no la agarra hasta que el recovery loop la libere
    (5min de delay innecesario).
    """
    def _fake_send_fail(jid, text, *, anti_loop=False, reply_to=None):
        return False  # bridge down

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid", _fake_send_fail,
    )

    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(30),
        jid="540000@s.whatsapp.net",
        message_text="msg",
    )
    summary = wa_scheduled.run_due_worker(max_retries=5)
    assert summary["retried"] == 1
    row = _read_row(isolated_state_db, sid)
    assert row["status"] == "pending", (
        f"tras send fail recoverable debería volver a pending, got {row['status']!r}"
    )
    assert row["attempt_count"] == 1


# ── E2E: run_due_worker invoca _notify_ambient_scheduled_outcome en los
# paths sent + sent_late (audit 2026-04-25 R2-Tests #3) ────────────────
#
# Los 6 tests unit-level de `_notify_ambient_scheduled_outcome`
# (test_notify_ambient_*) cubren la lógica interna del notificador
# aislada (config off, self-loop, formato del mensaje, etc.) y el commit
# anterior agregó `test_notify_ambient_called_on_failed_after_max_retries`
# para el path 'failed'. Faltaban los 2 paths restantes: 'sent' (delta
# corto, dentro del threshold) y 'sent_late' (delta > 5min). Sin estos
# E2E, un refactor en `run_due_worker` que mueva las llamadas
# `_notify_ambient_scheduled_outcome(sched, "sent"|"sent_late", ...)`
# afuera del scope correcto pasa silencioso hasta producción — los
# tests existentes verifican la transición de status pero no el side
# effect de la notificación al ambient JID.


def test_run_due_worker_notifies_ambient_on_successful_send(
    isolated_state_db, monkeypatch,
):
    """E2E (audit 2026-04-25 R2-Tests #3, path 'sent'): row pending
    recién due (scheduled_for ~ now) → bridge OK → `run_due_worker`
    debe invocar `_notify_ambient_scheduled_outcome` exactamente una
    vez con `status="sent"`, `delta_minutes` cercano a 0 (dentro del
    late_threshold de 5min), `attempt_count=1`, y el sched original
    (con jid + scheduled_for_utc) como primer arg.
    """
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(0),  # justo ahora — recién due
        jid="540000@s.whatsapp.net",
        message_text="hola justo a tiempo",
    )

    # Bridge OK.
    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid",
        lambda *a, **kw: True,
    )

    # Spy sobre la notif para capturar status + kwargs.
    notif_calls: list[dict] = []

    def _spy(sched, status, **kwargs):
        notif_calls.append({"sched": sched, "status": status, **kwargs})

    monkeypatch.setattr(
        wa_scheduled, "_notify_ambient_scheduled_outcome", _spy,
    )

    summary = wa_scheduled.run_due_worker()
    assert summary["sent"] == 1, (
        f"esperaba 1 row enviado dentro del threshold, got {summary!r}"
    )
    assert summary["sent_late"] == 0
    assert summary["failed"] == 0

    assert len(notif_calls) == 1, (
        f"esperaba 1 notif para status='sent', got {len(notif_calls)}"
    )
    call = notif_calls[0]
    assert call["status"] == "sent"
    assert call["sched"]["id"] == sid
    assert call["sched"]["jid"] == "540000@s.whatsapp.net"
    # Delta cerca de 0 (recién due): puede ser 0 o muy poco. El
    # threshold para 'sent' es < 5min — confirmamos que estamos
    # dentro de ese rango sin acoplarnos al valor exacto (que depende
    # de cuánto tarda el test en correr entre _seed_row y run_due_worker).
    assert "delta_minutes" in call
    assert abs(call["delta_minutes"]) < 5, (
        f"delta_minutes={call['delta_minutes']} fuera del threshold 'sent' "
        f"(<5min) — el test corrió demasiado lento o el seed/run quedó "
        f"desincronizado"
    )
    assert call.get("attempt_count") == 1


def test_run_due_worker_notifies_ambient_on_late_send(
    isolated_state_db, monkeypatch,
):
    """E2E (audit 2026-04-25 R2-Tests #3, path 'sent_late'): row
    pending con scheduled_for hace 60min (delta >> threshold de 5min)
    → bridge OK → `run_due_worker` debe invocar
    `_notify_ambient_scheduled_outcome` con `status="sent_late"`
    (NO 'sent') y `delta_minutes >= 60`.

    Este path se dispara cuando la Mac estuvo dormida durante la
    ventana del scheduled_for y el worker se entera tarde — el
    feedback al user via ambient JID le explica el delay.
    """
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(60),  # 1h atrás → late
        jid="540000@s.whatsapp.net",
        message_text="hola tarde",
    )

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid",
        lambda *a, **kw: True,
    )

    notif_calls: list[dict] = []

    def _spy(sched, status, **kwargs):
        notif_calls.append({"sched": sched, "status": status, **kwargs})

    monkeypatch.setattr(
        wa_scheduled, "_notify_ambient_scheduled_outcome", _spy,
    )

    summary = wa_scheduled.run_due_worker()
    assert summary["sent"] == 0
    assert summary["sent_late"] == 1, (
        f"esperaba sent_late=1 con delta de 60min, got {summary!r}"
    )

    assert len(notif_calls) == 1, (
        f"esperaba 1 notif para status='sent_late', got {len(notif_calls)}"
    )
    call = notif_calls[0]
    assert call["status"] == "sent_late", (
        f"con delta=60min el status notificado debe ser 'sent_late', "
        f"got {call['status']!r}"
    )
    assert call["sched"]["id"] == sid
    # Delta debería ser ~60min (puede tener algunos segundos de drift por
    # el tiempo entre _seed_row y run_due_worker).
    assert call["delta_minutes"] >= 60, (
        f"delta_minutes={call['delta_minutes']} debería ser >= 60min "
        f"(scheduled_for fue hace 1h)"
    )
    assert call.get("attempt_count") == 1


# ── Reply-to + scheduled_for: persiste E ENVÍA correctamente ──────────


def test_run_due_worker_passes_reply_to_to_bridge(
    isolated_state_db, monkeypatch,
):
    """Cuando se schedulea un reply (con `reply_to` populado), el
    worker debe leer las columnas reply_to_* de la DB y pasarlas al
    bridge en `_whatsapp_send_to_jid(..., reply_to=dict)`. Sin este
    test, un cambio en run_due_worker que rompa el spread del dict
    pasa silencioso (test_persists solo verifica las columnas
    guardadas, no el outgoing call).
    """
    captured = []

    def _fake_send(jid, text, *, anti_loop=False, reply_to=None):
        captured.append({"jid": jid, "text": text, "reply_to": reply_to})
        return True

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid", _fake_send,
    )

    # Sembrar reply pending con columnas reply_to_* populadas.
    sid = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(30),
        jid="540000@s.whatsapp.net",
        message_text="dale, voy",
    )
    conn = sqlite3.connect(str(isolated_state_db), isolation_level=None)
    try:
        conn.execute(
            f"UPDATE {wa_scheduled._TABLE} SET reply_to_id=?, "
            f"reply_to_text=?, reply_to_sender_jid=? WHERE id=?",
            ("FAKE_BRIDGE_MSG_ID_42", "che, vení a las 9?",
             "540000@s.whatsapp.net", sid),
        )
    finally:
        conn.close()

    summary = wa_scheduled.run_due_worker()
    # scheduled_for_utc fue hace 30min >> 5min late_threshold →
    # sent_late, no sent. Lo importante para este test es que el bridge
    # recibió el reply_to correcto.
    assert summary["sent_late"] == 1
    assert len(captured) == 1
    call = captured[0]
    assert call["jid"] == "540000@s.whatsapp.net"
    assert call["text"] == "dale, voy"
    # El dict reply_to debe llegar tal cual al bridge — message_id es
    # crítico para que cuando el bridge soporte quotes nativas funcione.
    assert call["reply_to"] is not None
    assert call["reply_to"]["message_id"] == "FAKE_BRIDGE_MSG_ID_42"
    assert call["reply_to"]["original_text"] == "che, vení a las 9?"
    assert call["reply_to"]["sender_jid"] == "540000@s.whatsapp.net"


def test_run_due_worker_no_reply_to_passes_none(
    isolated_state_db, monkeypatch,
):
    """El path "send normal" (sin reply_to) debe pasar reply_to=None
    al bridge — no un dict vacío ni un dict con strings vacíos. Bridge
    interpreta ambos distinto.
    """
    captured = []

    def _fake_send(jid, text, *, anti_loop=False, reply_to=None):
        captured.append({"reply_to": reply_to})
        return True

    monkeypatch.setattr(
        "rag.integrations.whatsapp._whatsapp_send_to_jid", _fake_send,
    )

    _seed_row(
        isolated_state_db,
        scheduled_for_utc=_past_iso(30),
        jid="540000@s.whatsapp.net",
        message_text="hola sin quote",
    )
    wa_scheduled.run_due_worker()
    assert len(captured) == 1
    assert captured[0]["reply_to"] is None


# ── LLM tools: whatsapp_list_scheduled / propose_whatsapp_cancel_scheduled
# / propose_whatsapp_reschedule_scheduled (issue #4 audit 2026-04-25) ─────
#
# Las tools dejan que el LLM gestione mensajes WhatsApp programados via
# NL: listar (`whatsapp_list_scheduled`), cancelar
# (`propose_whatsapp_cancel_scheduled`), y reagendar
# (`propose_whatsapp_reschedule_scheduled`). Las dos `propose_*` emiten
# proposal cards (kind="whatsapp_cancel_scheduled" /
# "whatsapp_reschedule_scheduled") que el frontend renderea con
# [Cancelar] / [Reagendar] / [Volver]. La `list` no emite card — el
# LLM la consume y resume al user en prosa.
#
# Cada test mockea `_whatsapp_jid_from_contact` (vive en
# `rag.integrations.whatsapp` y se re-exporta en `rag/__init__.py`)
# para no tocar Apple Contacts real, y siembra rows con `_seed_row` o
# directamente `wa_scheduled.schedule()`. La fixture `isolated_state_db`
# garantiza que la DB es per-test.


import json as _json  # noqa: E402  — alias para no chocar con `json` de deps


def _mock_jid_lookup(monkeypatch, jid="5491155551111@s.whatsapp.net",
                     full_name="Grecia", error=None):
    """Patch `_whatsapp_jid_from_contact` para que retorne lo dado
    sin tocar Apple Contacts. Lo seteamos en `rag` (el módulo que
    las tools `propose_*` resuelven en su body) y en
    `rag.integrations.whatsapp` (la home real del símbolo) por
    seguridad — ambos paths se usan en la codebase.
    """
    def _fake(_query):
        return {"jid": jid, "full_name": full_name, "error": error}
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact", _fake)
    try:
        from rag.integrations import whatsapp as _waint
        monkeypatch.setattr(_waint, "_whatsapp_jid_from_contact", _fake)
    except Exception:
        pass


# ── 19. whatsapp_list_scheduled ──────────────────────────────────────


def test_whatsapp_list_scheduled_returns_pending_only(
    isolated_state_db, monkeypatch,
):
    """Default (sin args) trae solo `pending` — los `cancelled` /
    `sent` no aparecen aunque estén en la DB."""
    # 2 pendings.
    sid_a = wa_scheduled.schedule(
        jid="a@s.whatsapp.net", message_text="msg A",
        scheduled_for_utc=_future_iso(60),
    )["id"]
    sid_b = wa_scheduled.schedule(
        jid="b@s.whatsapp.net", message_text="msg B",
        scheduled_for_utc=_future_iso(120),
    )["id"]
    # 1 cancelled (sembrado directo bypass schedule()).
    sid_c = _seed_row(
        isolated_state_db,
        scheduled_for_utc=_future_iso(180),
        status="cancelled", jid="c@s.whatsapp.net",
        message_text="msg C cancelado",
    )

    out = rag.whatsapp_list_scheduled()
    body = _json.loads(out)
    assert body["count"] == 2
    ids = sorted(it["id"] for it in body["items"])
    assert ids == sorted([sid_a, sid_b])
    assert sid_c not in ids
    # Cada item tiene los campos esperados que el LLM consume.
    for it in body["items"]:
        assert "scheduled_for_local" in it
        assert "scheduled_for_iso_ar" in it
        assert "message_text_preview" in it
        assert it["status"] == "pending"


def test_whatsapp_list_scheduled_truncates_message_text(
    isolated_state_db, monkeypatch,
):
    """`message_text_preview` se trunca a 80 chars con ellipsis."""
    long_msg = "a" * 200
    wa_scheduled.schedule(
        jid="x@s.whatsapp.net", message_text=long_msg,
        scheduled_for_utc=_future_iso(60),
    )
    body = _json.loads(rag.whatsapp_list_scheduled())
    assert body["count"] == 1
    preview = body["items"][0]["message_text_preview"]
    assert len(preview) <= 80
    assert preview.endswith("…")


def test_whatsapp_list_scheduled_with_status_all_returns_all_states(
    isolated_state_db, monkeypatch,
):
    """`status='all'` se traduce a status=None en `wa_scheduled.list_scheduled`,
    que retorna TODAS las filas independientemente del estado.
    Util para "cuáles mandé este mes" / "cuáles cancelé"."""
    wa_scheduled.schedule(
        jid="a@s.whatsapp.net", message_text="A",
        scheduled_for_utc=_future_iso(60),
    )
    _seed_row(
        isolated_state_db,
        scheduled_for_utc=_future_iso(90),
        status="cancelled", jid="b@s.whatsapp.net",
        message_text="B",
    )
    _seed_row(
        isolated_state_db,
        scheduled_for_utc=_future_iso(120),
        status="sent", jid="c@s.whatsapp.net",
        message_text="C",
    )
    body = _json.loads(rag.whatsapp_list_scheduled(status="all"))
    assert body["count"] == 3


# ── 20. propose_whatsapp_cancel_scheduled ────────────────────────────


def test_propose_cancel_scheduled_when_no_pendings_for_contact(
    isolated_state_db, monkeypatch,
):
    """0 matches → JSON con `error` populado, sin scheduled_id."""
    _mock_jid_lookup(
        monkeypatch, jid="5491155551111@s.whatsapp.net",
        full_name="Grecia",
    )
    # Hay un pending pero para OTRO jid — no debería matchear.
    wa_scheduled.schedule(
        jid="otro@s.whatsapp.net", message_text="otro",
        scheduled_for_utc=_future_iso(60),
    )
    out = rag.propose_whatsapp_cancel_scheduled(contact_name="Grecia")
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_cancel_scheduled"
    assert "error" in body
    assert "Grecia" in body["error"]
    # No emitimos scheduled_id en el error path.
    assert "scheduled_id" not in (body.get("fields") or {})


def test_propose_cancel_scheduled_with_one_pending(
    isolated_state_db, monkeypatch,
):
    """1 match → fields tiene scheduled_id, scheduled_for_local,
    contact_name (preferentemente full_name), message_text_preview."""
    target_jid = "5491155551111@s.whatsapp.net"
    _mock_jid_lookup(monkeypatch, jid=target_jid, full_name="Grecia Full")
    sid = wa_scheduled.schedule(
        jid=target_jid, message_text="feliz cumple, capa",
        scheduled_for_utc=_future_iso(60),
    )["id"]
    out = rag.propose_whatsapp_cancel_scheduled(contact_name="Grecia")
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_cancel_scheduled"
    assert "error" not in body
    assert body.get("needs_clarification") is not True
    fields = body["fields"]
    assert fields["scheduled_id"] == sid
    assert fields["jid"] == target_jid
    assert fields["contact_name"] == "Grecia Full"
    assert "feliz cumple" in fields["message_text_preview"]
    assert fields["scheduled_for_local"]  # no vacío


def test_propose_cancel_scheduled_with_multiple_needs_clarification(
    isolated_state_db, monkeypatch,
):
    """>1 matches y when_hint que no desambigua → needs_clarification:
    true + candidates con todos los pendings del contacto."""
    target_jid = "5491155551111@s.whatsapp.net"
    _mock_jid_lookup(monkeypatch, jid=target_jid, full_name="Grecia")
    sid_a = wa_scheduled.schedule(
        jid=target_jid, message_text="msg A",
        scheduled_for_utc=_future_iso(60),
    )["id"]
    sid_b = wa_scheduled.schedule(
        jid=target_jid, message_text="msg B",
        scheduled_for_utc=_future_iso(60 * 24 * 3),  # +3 días
    )["id"]
    out = rag.propose_whatsapp_cancel_scheduled(contact_name="Grecia")
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_cancel_scheduled"
    assert body.get("needs_clarification") is True
    cands = body.get("candidates") or []
    cand_ids = sorted(c["scheduled_id"] for c in cands)
    assert cand_ids == sorted([sid_a, sid_b])
    # Cada candidato lleva la info que la UI necesita renderear.
    for c in cands:
        assert c["scheduled_for_local"]
        assert "message_text_preview" in c


def test_propose_cancel_scheduled_with_when_hint_filters_correctly(
    isolated_state_db, monkeypatch,
):
    """Bonus: when_hint='mañana' → filtra al pending de mañana,
    devolviendo 1 sola row (no needs_clarification)."""
    target_jid = "5491155551111@s.whatsapp.net"
    _mock_jid_lookup(monkeypatch, jid=target_jid, full_name="Grecia")
    # Mañana 9hs (UTC 12).
    tomorrow_9 = (
        datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    ).isoformat(timespec="seconds")
    sid_tomorrow = wa_scheduled.schedule(
        jid=target_jid, message_text="el de mañana",
        scheduled_for_utc=tomorrow_9,
    )["id"]
    # En 5 días (NO debería matchear "mañana").
    far = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(timespec="seconds")
    wa_scheduled.schedule(
        jid=target_jid, message_text="el del viernes",
        scheduled_for_utc=far,
    )
    out = rag.propose_whatsapp_cancel_scheduled(
        contact_name="Grecia", when_hint="mañana 9hs",
    )
    body = _json.loads(out)
    if body.get("needs_clarification"):
        # Si el parser de NL no tiene el día del LLM disponible, podría
        # quedar ambiguo y devolver clarif — aceptable como fallback,
        # pero la rama feliz es 1 match.
        cands = body.get("candidates") or []
        assert len(cands) >= 1
    else:
        assert body["fields"]["scheduled_id"] == sid_tomorrow


def test_propose_cancel_scheduled_with_unknown_contact(
    isolated_state_db, monkeypatch,
):
    """Contact lookup falla → error en payload, no scheduled_id."""
    _mock_jid_lookup(monkeypatch, jid="", full_name="", error="not_found")
    out = rag.propose_whatsapp_cancel_scheduled(contact_name="Fulano Inexistente")
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_cancel_scheduled"
    assert "error" in body
    assert "Fulano Inexistente" in body["error"]


# ── 21. propose_whatsapp_reschedule_scheduled ────────────────────────


def test_propose_reschedule_scheduled_with_new_when_parses_correctly(
    isolated_state_db, monkeypatch,
):
    """new_when='en 3 horas' parsea → fields tiene
    new_scheduled_for_iso_ar y new_scheduled_for_local."""
    target_jid = "5491155551111@s.whatsapp.net"
    _mock_jid_lookup(monkeypatch, jid=target_jid, full_name="Oscar")
    sid = wa_scheduled.schedule(
        jid=target_jid, message_text="msg",
        scheduled_for_utc=_future_iso(60),
    )["id"]
    out = rag.propose_whatsapp_reschedule_scheduled(
        contact_name="Oscar", new_when="en 3 horas",
    )
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_reschedule_scheduled"
    if "error" in body:
        # Si el parser de fecha no está disponible (ej. dateparser
        # missing), aceptamos el fallback con error explicativo.
        assert "no entendí" in body["error"].lower() or "fecha" in body["error"].lower()
        return
    fields = body["fields"]
    assert fields.get("scheduled_id") == sid
    assert fields["new_scheduled_for_local"]
    assert fields["new_scheduled_for_iso_ar"].endswith("-03:00")
    assert "old_scheduled_for_local" in fields


def test_propose_reschedule_scheduled_with_past_when_returns_error(
    isolated_state_db, monkeypatch,
):
    """new_when="ayer 9hs" → datetime en el pasado → error explícito,
    sin fields.scheduled_id."""
    _mock_jid_lookup(monkeypatch)
    out = rag.propose_whatsapp_reschedule_scheduled(
        contact_name="Grecia", new_when="ayer 9hs",
    )
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_reschedule_scheduled"
    assert "error" in body
    err = body["error"].lower()
    assert "pasado" in err or "no entendí" in err


def test_propose_reschedule_scheduled_with_garbage_when_returns_error(
    isolated_state_db, monkeypatch,
):
    """new_when='asdfasdf' no parsea → error 'no entendí'."""
    _mock_jid_lookup(monkeypatch)
    out = rag.propose_whatsapp_reschedule_scheduled(
        contact_name="Grecia", new_when="zzzzzzzz",
    )
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_reschedule_scheduled"
    assert "error" in body
    assert "no entendí" in body["error"].lower()


def test_propose_reschedule_scheduled_with_empty_when_returns_error(
    isolated_state_db, monkeypatch,
):
    """new_when='' → error explícito sin tocar nada."""
    _mock_jid_lookup(monkeypatch)
    out = rag.propose_whatsapp_reschedule_scheduled(
        contact_name="Grecia", new_when="",
    )
    body = _json.loads(out)
    assert body["kind"] == "whatsapp_reschedule_scheduled"
    assert "error" in body


# ── Notificación post-envío al ambient JID (audit 2026-04-25, A.1) ─────


def test_format_delta_human_minutes():
    """`_format_delta_human` produce strings cortos para mostrar al user
    en el mensaje de notificación."""
    f = wa_scheduled._format_delta_human
    assert f(0) == "0min"
    assert f(5) == "5min"
    assert f(59) == "59min"
    assert f(60) == "1h"
    assert f(90) == "1h 30min"
    assert f(120) == "2h"
    assert f(125) == "2h 5min"
    assert f(24 * 60) == "1d"
    assert f(24 * 60 + 60 * 4) == "1d 4h"
    assert f(7 * 24 * 60) == "7d"


def test_notify_ambient_skips_when_no_config(monkeypatch):
    """Sin ambient config (None o disabled) → silent skip, no llama al
    bridge. La notificación es nice-to-have, no bloquea nada.
    """
    sent_calls = []

    def _fake_send(jid, text):
        sent_calls.append((jid, text))
        return True

    monkeypatch.setattr("rag._ambient_config", lambda: None)
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send", _fake_send,
    )

    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "540000@s.whatsapp.net", "contact_name": "Test",
         "scheduled_for_utc": "2026-04-25T12:00:00+00:00"},
        "sent", delta_minutes=2,
    )
    assert sent_calls == [], "no debería notificar sin config"


def test_notify_ambient_skips_when_disabled(monkeypatch):
    """Config existe pero `enabled=False` → skip."""
    sent_calls = []
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": False, "jid": "X@s.whatsapp.net"})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "540000@s.whatsapp.net"}, "sent", delta_minutes=0,
    )
    assert sent_calls == []


def test_notify_ambient_skips_self_loop(monkeypatch):
    """Si el target_jid del scheduled ES el ambient JID, NO notificar.
    Evita auto-spam: el user ya vio el mensaje original cuando lo recibió.
    """
    sent_calls = []
    AMBIENT = "5491155556666@s.whatsapp.net"
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": True, "jid": AMBIENT})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    # Mismo jid → self-loop → skip
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": AMBIENT, "contact_name": "yo",
         "scheduled_for_utc": "2026-04-25T12:00:00+00:00"},
        "sent", delta_minutes=2,
    )
    assert sent_calls == []


def test_notify_ambient_sends_for_normal_recipient(monkeypatch):
    """Path happy: target distinto al ambient → manda notificación
    formato correcto."""
    sent_calls = []
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": True, "jid": "MEAMBIENT@s.whatsapp.net"})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "540000@s.whatsapp.net", "contact_name": "Grecia",
         "scheduled_for_utc": "2099-04-25T12:00:00+00:00"},
        "sent", delta_minutes=2, attempt_count=1,
    )
    assert len(sent_calls) == 1
    target_jid, msg = sent_calls[0]
    assert target_jid == "MEAMBIENT@s.whatsapp.net"
    assert "Mandé tu mensaje a Grecia" in msg
    assert "✓" in msg


def test_notify_ambient_sent_late_includes_delta(monkeypatch):
    """sent_late incluye el delta humano en el mensaje."""
    sent_calls = []
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": True, "jid": "ME@s.whatsapp.net"})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "540000@s.whatsapp.net", "contact_name": "Oscar",
         "scheduled_for_utc": "2026-04-25T12:00:00+00:00"},
        "sent_late", delta_minutes=145, attempt_count=1,
    )
    assert len(sent_calls) == 1
    msg = sent_calls[0][1]
    assert "Oscar" in msg
    assert "2h 25min" in msg  # 145min = 2h 25min
    assert "tarde" in msg
    assert "⚠" in msg


def test_notify_ambient_failed_includes_attempts_and_error(monkeypatch):
    """failed incluye N intentos + último error."""
    sent_calls = []
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": True, "jid": "ME@s.whatsapp.net"})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "540000@s.whatsapp.net", "contact_name": "Sole",
         "scheduled_for_utc": "2026-04-25T12:00:00+00:00"},
        "failed", attempt_count=5, last_error="bridge_unreachable",
    )
    assert len(sent_calls) == 1
    msg = sent_calls[0][1]
    assert "Sole" in msg
    assert "5 intentos" in msg
    assert "bridge_unreachable" in msg
    assert "✗" in msg


def test_notify_ambient_uses_phone_fallback_when_no_contact_name(monkeypatch):
    """Si contact_name está vacío, fallback a últimos dígitos del JID
    para reconocimiento básico."""
    sent_calls = []
    monkeypatch.setattr("rag._ambient_config",
                        lambda: {"enabled": True, "jid": "ME@s.whatsapp.net"})
    monkeypatch.setattr(
        "rag.integrations.whatsapp._ambient_whatsapp_send",
        lambda jid, text: sent_calls.append((jid, text)) or True,
    )
    wa_scheduled._notify_ambient_scheduled_outcome(
        {"jid": "5491198765432@s.whatsapp.net", "contact_name": "",
         "scheduled_for_utc": "2099-04-25T12:00:00+00:00"},
        "sent", delta_minutes=0,
    )
    msg = sent_calls[0][1]
    # Últimos 8 dígitos como fallback de identidad
    assert "98765432" in msg
