"""Tests para `push_due_reminders_to_whatsapp` y los helpers de la tabla
`rag_reminder_wa_pushed`.

El runner lee Apple Reminders via `_fetch_reminders_due`, mandà notifs al
ambient JID via `_ambient_whatsapp_send`, y persiste dedup en SQL para
que el cron cada 5min no pushee dos veces el mismo reminder.

Los tests monkeypatchean ambos collaborators offline y validan:
  - state table roundtrip (mark + check + prune)
  - happy path: 3 reminders due, todos pushed y marcados
  - re-run dentro de la ventana: 0 pushed (todos ya marcados)
  - bridge falla en uno: 2 pushed/marked, 1 NO marked (retry next run)
  - reminder_push_enabled: false → 0 pushed, sin API hit
  - sin ambient config → 0 pushed, summary vacío
  - window filter: due 10 min ago + window=5 → not pushed
  - prune: rows con pushed_at > 30d se borran
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg_path(tmp_path):
    """Redirigir AMBIENT_CONFIG_PATH y DB_PATH a tmp por test.

    Usamos try/finally explícito en vez de `monkeypatch.setattr` porque
    el conftest autouse `_stabilize_rag_state` chequea `rag.DB_PATH`
    contra su snapshot DURANTE su propia teardown — y pytest finaliza
    fixtures independientes en orden FIFO de creación, así que el
    teardown del conftest corría ANTES del teardown del monkeypatch.
    Resultado: warning falso "rag.DB_PATH leaked from test" cada test.

    El fix: restauramos los valores en el propio finally del fixture,
    así para cuando el conftest hace su check, ya están de vuelta.
    """
    p = tmp_path / "ambient.json"
    snap_cfg = rag.AMBIENT_CONFIG_PATH
    snap_db = rag.DB_PATH
    snap_log = rag.AMBIENT_LOG_PATH
    rag.AMBIENT_CONFIG_PATH = p
    rag.DB_PATH = tmp_path
    rag.AMBIENT_LOG_PATH = tmp_path / "ambient.jsonl"
    try:
        yield p
    finally:
        rag.AMBIENT_CONFIG_PATH = snap_cfg
        rag.DB_PATH = snap_db
        rag.AMBIENT_LOG_PATH = snap_log


def _write_cfg(path: Path, *, jid: str = "ragnet@g.us",
               enabled: bool = True, reminder_push_enabled=None) -> None:
    payload: dict = {"jid": jid, "enabled": enabled}
    if reminder_push_enabled is not None:
        payload["reminder_push_enabled"] = reminder_push_enabled
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def fake_send(monkeypatch):
    """Captura los envíos al bridge. ``calls`` es la lista de
    ``{jid, text}`` y ``set_outcomes(seq)`` reconfigura los retornos
    sucesivos del send (default: True para todos)."""
    calls: list[dict] = []
    state = {"outcomes": [True] * 100, "idx": 0}

    def _fake(jid: str, text: str) -> bool:
        calls.append({"jid": jid, "text": text})
        if state["idx"] >= len(state["outcomes"]):
            return True
        out = state["outcomes"][state["idx"]]
        state["idx"] += 1
        return out

    monkeypatch.setattr(rag, "_ambient_whatsapp_send", _fake)

    def set_outcomes(seq: list[bool]) -> None:
        state["outcomes"] = list(seq)
        state["idx"] = 0

    return calls, set_outcomes


@pytest.fixture
def fake_reminders(monkeypatch):
    """Stub `_fetch_reminders_due`. ``set([...])`` define la salida."""
    state: dict = {"items": []}

    def _fake(now: datetime, horizon_days: int = 1, max_items: int = 20) -> list[dict]:
        return list(state["items"])

    monkeypatch.setattr(rag, "_fetch_reminders_due", _fake)

    def setter(items: list[dict]) -> None:
        state["items"] = items

    return setter


def _build_reminder(rid: str, due: datetime, name: str = "Tarea X",
                    list_name: str = "Lista", bucket: str = "today") -> dict:
    return {
        "id": rid,
        "name": name,
        "due": due.isoformat(timespec="minutes"),
        "list": list_name,
        "bucket": bucket,
    }


def _read_pushed_rows(tmp_path: Path) -> list[dict]:
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    if not db.is_file():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT reminder_id, pushed_at, due_iso, title "
            "FROM rag_reminder_wa_pushed ORDER BY reminder_id"
        ).fetchall())
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── State table roundtrip ─────────────────────────────────────────────────────


def test_state_table_mark_check_and_prune(cfg_path, tmp_path):
    """`_reminder_wa_mark_pushed` + `_reminder_wa_was_pushed` + prune."""
    with rag._ragvec_state_conn() as conn:
        # Inicialmente nada está pushed.
        assert rag._reminder_wa_was_pushed(conn, "rid-1") is False

        # Mark + verify.
        rag._reminder_wa_mark_pushed(conn, "rid-1", "2026-04-24T18:00", "Tarea")
        rag._reminder_wa_mark_pushed(conn, "rid-2", "2026-04-24T18:05", "Otra")
        assert rag._reminder_wa_was_pushed(conn, "rid-1") is True
        assert rag._reminder_wa_was_pushed(conn, "rid-2") is True
        assert rag._reminder_wa_was_pushed(conn, "rid-3") is False

        rows = _read_pushed_rows(tmp_path)
        assert len(rows) == 2
        rids = {r["reminder_id"] for r in rows}
        assert rids == {"rid-1", "rid-2"}

        # Prune defaultea a 30 días → rows recientes (just-marked) sobreviven.
        n_pruned = rag._reminder_wa_prune_old(conn, days=30)
        assert n_pruned == 0
        assert len(_read_pushed_rows(tmp_path)) == 2

        # Inyectar un row con pushed_at viejo (45 días atrás) y prune.
        old_ts = (datetime.now() - timedelta(days=45)).isoformat(timespec="seconds")
        conn.execute(
            "UPDATE rag_reminder_wa_pushed SET pushed_at = ? "
            "WHERE reminder_id = 'rid-1'",
            (old_ts,),
        )
        conn.commit()
        n_pruned = rag._reminder_wa_prune_old(conn, days=30)
        assert n_pruned == 1
        rows_after = _read_pushed_rows(tmp_path)
        assert {r["reminder_id"] for r in rows_after} == {"rid-2"}


# ── Happy path ───────────────────────────────────────────────────────────────


def test_happy_path_three_reminders_pushed_and_marked(
    cfg_path, fake_send, fake_reminders, tmp_path,
):
    _write_cfg(cfg_path, jid="ragnet@g.us")
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
        _build_reminder("rid-B", now - timedelta(minutes=2), name="B"),
        _build_reminder("rid-C", now - timedelta(minutes=3), name="C"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert summary["pushed"] == 3
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert len(calls) == 3
    # All went to the configured jid.
    assert {c["jid"] for c in calls} == {"ragnet@g.us"}
    # Body includes ⏰ + title + due_human.
    assert all("⏰" in c["text"] for c in calls)
    titles_in_text = " ".join(c["text"] for c in calls)
    for t in ("A", "B", "C"):
        assert t in titles_in_text

    # Persisted dedup rows.
    rows = _read_pushed_rows(tmp_path)
    assert {r["reminder_id"] for r in rows} == {"rid-A", "rid-B", "rid-C"}


# ── Idempotencia ─────────────────────────────────────────────────────────────


def test_re_run_within_window_is_noop(
    cfg_path, fake_send, fake_reminders,
):
    _write_cfg(cfg_path)
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    items = [
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
        _build_reminder("rid-B", now - timedelta(minutes=2), name="B"),
    ]
    fake_reminders(items)

    s1 = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert s1["pushed"] == 2
    assert len(calls) == 2

    # Same items, second run → 0 pushed (already marked), no new bridge calls.
    s2 = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert s2["pushed"] == 0
    assert s2["skipped"] == 2
    assert s2["failed"] == 0
    assert len(calls) == 2  # no new sends


# ── Falla parcial del bridge ──────────────────────────────────────────────────


def test_bridge_failure_keeps_unmarked_for_retry(
    cfg_path, fake_send, fake_reminders, tmp_path,
):
    _write_cfg(cfg_path)
    calls, set_outcomes = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
        _build_reminder("rid-B", now - timedelta(minutes=2), name="B"),
        _build_reminder("rid-C", now - timedelta(minutes=3), name="C"),
    ])
    # Middle one fails.
    set_outcomes([True, False, True])

    summary = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert summary["pushed"] == 2
    assert summary["failed"] == 1
    assert len(calls) == 3  # all three were attempted

    rows = _read_pushed_rows(tmp_path)
    # Only A and C marked; B retried next run.
    assert {r["reminder_id"] for r in rows} == {"rid-A", "rid-C"}


# ── Opt-out por config ───────────────────────────────────────────────────────


def test_reminder_push_enabled_false_skips_all(
    cfg_path, fake_send, fake_reminders,
):
    _write_cfg(cfg_path, reminder_push_enabled=False)
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert summary["pushed"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary.get("reason") == "reminder_push_disabled"
    assert calls == []


def test_no_ambient_config_returns_empty(
    cfg_path, fake_send, fake_reminders,
):
    # cfg_path not written → file missing.
    assert not cfg_path.is_file()
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert summary["pushed"] == 0
    assert summary["skipped"] == 0
    assert summary["failed"] == 0
    assert summary.get("reason") == "no_ambient_config"
    assert summary["items"] == []
    assert calls == []


# ── Filtro de ventana ─────────────────────────────────────────────────────────


def test_window_filter_excludes_old_due(
    cfg_path, fake_send, fake_reminders,
):
    _write_cfg(cfg_path)
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        # In window — 2 min ago, window=5 → push.
        _build_reminder("rid-recent", now - timedelta(minutes=2),
                        name="Reciente"),
        # Out of window — 10 min ago, window=5 → skip.
        _build_reminder("rid-old", now - timedelta(minutes=10),
                        name="Vieja"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(now=now, window_min=5)
    assert summary["pushed"] == 1
    assert summary["skipped"] == 1
    assert len(calls) == 1
    assert "Reciente" in calls[0]["text"]
    assert "Vieja" not in calls[0]["text"]


def test_max_overdue_excludes_very_old_even_if_window_large(
    cfg_path, fake_send, fake_reminders,
):
    """Window grande pero max_overdue chico → max_overdue gana."""
    _write_cfg(cfg_path)
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        # 30h overdue, max_overdue=1440min (24h) → skip.
        _build_reminder("rid-ancient", now - timedelta(hours=30),
                        name="Ancestral", bucket="overdue"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(
        now=now, window_min=10000, max_overdue_min=1440,
    )
    assert summary["pushed"] == 0
    assert summary["skipped"] == 1
    assert calls == []


# ── Dry-run ──────────────────────────────────────────────────────────────────


def test_dry_run_no_send_no_mark(
    cfg_path, fake_send, fake_reminders, tmp_path,
):
    _write_cfg(cfg_path)
    calls, _ = fake_send
    now = datetime(2026, 4, 24, 18, 0)
    fake_reminders([
        _build_reminder("rid-A", now - timedelta(minutes=1), name="A"),
        _build_reminder("rid-B", now - timedelta(minutes=2), name="B"),
    ])

    summary = rag.push_due_reminders_to_whatsapp(
        now=now, window_min=5, dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["pushed"] == 0
    # items_preview muestra las acciones que harían.
    actions = [it["action"] for it in summary["items"]]
    assert actions.count("dry_run") == 2
    assert calls == []
    assert _read_pushed_rows(tmp_path) == []
