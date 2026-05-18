"""FastAPI routes for the anticipatory inbox and manual actions."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

__all__ = [
    "SnoozeRequest",
    "SilenceRequest",
    "anticipate_inbox",
    "anticipate_send",
    "anticipate_snooze",
    "anticipate_silence",
    "register_anticipate_routes",
]


@dataclass(frozen=True)
class _PushContext:
    ambient_enabled: bool
    silenced: frozenset[str]
    snooze: dict[str, str]
    daily_count: int


class SnoozeRequest(BaseModel):
    hours: int = Field(..., ge=1, le=24 * 365)


class SilenceRequest(BaseModel):
    enabled: bool = True


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def _column_exists(conn, table: str, column: str) -> bool:
    try:
        return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))
    except sqlite3.Error:
        return False


def _candidate_select_columns(conn) -> str:
    message_full = "message_full" if _column_exists(
        conn,
        "rag_anticipate_candidates",
        "message_full",
    ) else "NULL AS message_full"
    return (
        "id, ts, kind, score, dedup_key, selected, sent, reason, "
        f"message_preview, {message_full}"
    )


def _bool_int(value) -> int:
    return 1 if bool(value) else 0


def _load_push_context() -> _PushContext:
    import rag as _rag
    import rag.proactive as _proactive

    try:
        ambient_enabled = bool(_rag._ambient_config())
    except Exception:
        ambient_enabled = False
    try:
        state = _proactive._proactive_load_state()
    except Exception:
        state = {}
    silenced_raw = state.get("silenced") if isinstance(state, dict) else []
    if not isinstance(silenced_raw, (list, tuple, set)):
        silenced_raw = []
    snooze_raw = state.get("snooze") if isinstance(state, dict) else {}
    if not isinstance(snooze_raw, dict):
        snooze_raw = {}
    return _PushContext(
        ambient_enabled=ambient_enabled,
        silenced=frozenset(str(k) for k in (silenced_raw or [])),
        snooze=dict(snooze_raw or {}),
        daily_count=int((state or {}).get("daily_count", 0) or 0),
    )


def _daily_cap_for_kind(kind: str) -> int:
    import rag.proactive as _proactive

    if kind.startswith("anticipate-"):
        return int(getattr(_proactive, "PROACTIVE_ANTICIPATE_DAILY_CAP", 6))
    return int(getattr(_proactive, "PROACTIVE_DAILY_CAP", 3))


def _active_snooze_until(ctx: _PushContext, kind: str, now: datetime) -> str | None:
    raw = ctx.snooze.get(kind)
    if not raw:
        return None
    try:
        until = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if now < until:
        return until.isoformat(timespec="seconds")
    return None


def _gate_skip_reason(
    row: dict,
    *,
    ctx: _PushContext,
    dedup_sent_keys: set[str],
    now: datetime,
) -> str | None:
    kind = str(row.get("kind") or "")
    dedup_key = str(row.get("dedup_key") or "")
    if not ctx.ambient_enabled:
        return "ambient_missing"
    if kind in ctx.silenced:
        return "silenced"
    if _active_snooze_until(ctx, kind, now):
        return "snoozed"
    if ctx.daily_count >= _daily_cap_for_kind(kind):
        return "daily_cap"
    if dedup_key and dedup_key in dedup_sent_keys and not _bool_int(row.get("sent")):
        return "dedup_sent"
    return None


def _status_and_reason(
    row: dict,
    *,
    ctx: _PushContext,
    dedup_sent_keys: set[str],
    now: datetime,
) -> tuple[str, str | None, bool]:
    selected = _bool_int(row.get("selected"))
    sent = _bool_int(row.get("sent"))
    if sent:
        return "sent", None, False

    gate_reason = _gate_skip_reason(
        row,
        ctx=ctx,
        dedup_sent_keys=dedup_sent_keys,
        now=now,
    )
    actionable = gate_reason is None
    if selected:
        return "blocked", gate_reason or "unknown_not_sent", actionable
    return "candidate", gate_reason or "not_selected", actionable


def _dedup_sent_keys(rows: list[dict]) -> set[str]:
    keys = sorted({
        str(row.get("dedup_key") or "")
        for row in rows
        if str(row.get("dedup_key") or "")
    })
    if not keys:
        return set()
    placeholders = ",".join("?" for _ in keys)
    sql = (
        "SELECT DISTINCT dedup_key FROM rag_anticipate_candidates "
        f"WHERE sent = 1 AND dedup_key IN ({placeholders})"
    )
    import rag as _rag

    with _rag._ragvec_state_conn() as conn:
        rows_sql = conn.execute(sql, keys).fetchall()
    return {str(row[0]) for row in rows_sql if row and row[0]}


def _fetch_candidate_rows(limit: int, days: int) -> list[dict]:
    import rag as _rag

    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _rag._ragvec_state_conn() as conn:
        conn.row_factory = sqlite3.Row
        columns = _candidate_select_columns(conn)
        rows = conn.execute(
            f"SELECT {columns} FROM rag_anticipate_candidates "
            "WHERE datetime(ts) >= datetime(?) "
            "ORDER BY ts DESC, id DESC "
            "LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _candidate_by_id(candidate_id: int) -> dict:
    import rag as _rag

    with _rag._ragvec_state_conn() as conn:
        conn.row_factory = sqlite3.Row
        columns = _candidate_select_columns(conn)
        row = conn.execute(
            f"SELECT {columns} FROM rag_anticipate_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="anticipate candidate not found")
    return _row_to_dict(row)


def _mark_candidate_sent(candidate_id: int) -> None:
    import rag as _rag

    with _rag._ragvec_state_conn() as conn:
        conn.execute(
            "UPDATE rag_anticipate_candidates SET sent = 1 WHERE id = ?",
            (candidate_id,),
        )
        conn.commit()


def _normalize_push_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    text = str(reason)
    low = text.lower()
    if "ambient" in low:
        return "ambient_missing"
    if "silenciado" in low or "silenced" in low:
        return "silenced"
    if "snooze" in low:
        return "snoozed"
    if "daily cap" in low or "cap diario" in low:
        return "daily_cap"
    if "bridge send failed" in low:
        return "wa_send_failed"
    return text


def anticipate_inbox(
    limit: int = 50,
    days: int = 7,
    only_actionable: bool = False,
) -> dict:
    limit = max(1, min(int(limit), 200))
    days = max(1, min(int(days), 365))
    now = datetime.now()

    try:
        rows = _fetch_candidate_rows(limit=limit, days=days)
        dedup_sent = _dedup_sent_keys(rows)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="anticipate inbox unavailable") from exc

    ctx = _load_push_context()
    items: list[dict] = []
    for row in rows:
        status, skip_reason, actionable = _status_and_reason(
            row,
            ctx=ctx,
            dedup_sent_keys=dedup_sent,
            now=now,
        )
        item = {
            "id": int(row["id"]),
            "ts": row["ts"],
            "kind": row["kind"],
            "score": float(row["score"]),
            "dedup_key": row["dedup_key"],
            "selected": _bool_int(row["selected"]),
            "sent": _bool_int(row["sent"]),
            "reason": row["reason"],
            "message_preview": row["message_preview"],
            "message_full": row.get("message_full"),
            "message_len": len(str(row.get("message_full") or row.get("message_preview") or "")),
            "status": status,
            "skip_reason": skip_reason,
            "actionable": bool(actionable),
        }
        items.append(item)

    visible = [item for item in items if item["actionable"]] if only_actionable else items
    summary = {
        "limit": limit,
        "days": days,
        "only_actionable": bool(only_actionable),
        "total": len(items),
        "returned": len(visible),
        "sent": sum(1 for item in visible if item["status"] == "sent"),
        "blocked": sum(1 for item in visible if item["status"] == "blocked"),
        "candidate": sum(1 for item in visible if item["status"] == "candidate"),
        "actionable": sum(1 for item in visible if item["actionable"]),
        "hidden_by_only_actionable": len(items) - len(visible),
    }
    return {"ok": True, "items": visible, "summary": summary}


def anticipate_send(candidate_id: int) -> dict:
    import rag as _rag

    try:
        row = _candidate_by_id(candidate_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="anticipate candidate unavailable") from exc

    message = str(row.get("message_full") or row.get("message_preview") or row.get("reason") or "")
    try:
        sent, reason = _rag.proactive_push(
            str(row["kind"]),
            message,
            snooze_hours=None,
            dedup_key=row.get("dedup_key"),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="anticipate send failed") from exc
    sent_bool = bool(sent)
    if sent_bool:
        try:
            _mark_candidate_sent(candidate_id)
        except sqlite3.Error:
            pass
    return {
        "ok": True,
        "sent": sent_bool,
        "skip_reason": _normalize_push_reason(reason),
    }


def anticipate_snooze(candidate_id: int, req: SnoozeRequest) -> dict:
    import rag.proactive as _proactive

    try:
        row = _candidate_by_id(candidate_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="anticipate candidate unavailable") from exc
    kind = str(row["kind"])
    until = datetime.now() + timedelta(hours=int(req.hours))
    state = _proactive._proactive_load_state()
    state.setdefault("snooze", {})[kind] = until.isoformat(timespec="seconds")
    _proactive._proactive_save_state(state)
    return {"ok": True, "kind": kind, "snoozed_until": state["snooze"][kind]}


def anticipate_silence(candidate_id: int, req: SilenceRequest) -> dict:
    import rag.proactive as _proactive

    try:
        row = _candidate_by_id(candidate_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="anticipate candidate unavailable") from exc
    kind = str(row["kind"])
    state = _proactive._proactive_load_state()
    silenced = {str(item) for item in state.get("silenced", [])}
    if req.enabled:
        silenced.add(kind)
    else:
        silenced.discard(kind)
    state["silenced"] = sorted(silenced)
    _proactive._proactive_save_state(state)
    return {"ok": True, "kind": kind, "enabled": bool(req.enabled)}


def register_anticipate_routes(
    app,
    require_admin_token: Callable | None = None,
) -> dict[str, object]:
    admin_dep = [Depends(require_admin_token)] if require_admin_token is not None else []
    app.get("/api/anticipate/inbox")(anticipate_inbox)
    app.post("/api/anticipate/{candidate_id}/send", dependencies=admin_dep)(anticipate_send)
    app.post("/api/anticipate/{candidate_id}/snooze", dependencies=admin_dep)(anticipate_snooze)
    app.post("/api/anticipate/{candidate_id}/silence", dependencies=admin_dep)(anticipate_silence)
    return {name: globals()[name] for name in __all__ if name != "register_anticipate_routes"}
