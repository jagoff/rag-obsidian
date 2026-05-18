"""Mission-control health route.

This endpoint is intentionally read-only at the application level: it only
collects existing telemetry and normalizes it into a compact operator view.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from scripts import audit_telemetry_health as _audit


_STATUS_RANK = {"ok": 0, "degraded": 1, "down": 2}
_AUDIT_STATUS_MAP = {
    "healthy": "ok",
    "ok": "ok",
    "degraded": "degraded",
    "stale": "down",
    "unknown": "degraded",
    "down": "down",
}
_CORE_TABLES = (
    "rag_queries",
    "rag_anticipate_candidates",
    "rag_feedback",
    "rag_response_cache",
)
_TELEMETRY_ERROR_DEGRADED_THRESHOLD = 1
_TELEMETRY_ERROR_ACTION_THRESHOLD = 10
_TELEMETRY_ERROR_DOWN_THRESHOLD = 100


class MissionControlActionRequest(BaseModel):
    action_id: str = Field(..., min_length=1, max_length=120)
    dry_run: bool = True
    days: int = Field(7, ge=1, le=90)
    limit: int = Field(20, ge=1, le=100)
    confidence_below: float = Field(0.35, ge=0.0, le=1.0)
    min_judge_conf: float | None = Field(None, ge=0.0, le=1.0)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _window_days(value: int) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = 7
    return max(1, min(days, 90))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, sqlite3.Row):
        return {key: _jsonable(value[key]) for key in value.keys()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _issues(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def _subsystem(
    subsystem_id: str,
    label: str,
    status: str,
    issues: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_status = status if status in _STATUS_RANK else "degraded"
    return {
        "id": subsystem_id,
        "label": label,
        "status": normalized_status,
        "issues": issues or [],
        "details": _jsonable(details or {}),
    }


def _subsystem_from_audit(
    subsystem_id: str,
    label: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    raw_status = str(raw.get("status") or "unknown")
    status = _AUDIT_STATUS_MAP.get(raw_status, "degraded")
    details = {
        key: value
        for key, value in raw.items()
        if key not in {"status", "issues"}
    }
    if set(details) == {"details"} and isinstance(details.get("details"), dict):
        details = dict(details["details"])
    details["source_status"] = raw_status
    issues = _issues(raw.get("issues"))
    if status != "ok" and not issues:
        issues.append(f"{label} status is {raw_status}")
    return _subsystem(subsystem_id, label, status, issues, details)


def _failed_subsystem(
    subsystem_id: str,
    label: str,
    exc: Exception,
    *,
    status: str = "degraded",
) -> dict[str, Any]:
    return _subsystem(
        subsystem_id,
        label,
        status,
        [f"health check failed: {type(exc).__name__}: {exc}"],
        {"error": repr(exc)},
    )


def _safe_audit_subsystem(
    subsystem_id: str,
    label: str,
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return _subsystem_from_audit(subsystem_id, label, fn())
    except Exception as exc:
        return _failed_subsystem(subsystem_id, label, exc)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _table_stats(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    if not _table_exists(conn, table):
        return {"exists": False, "rows": None, "latest_ts": None}
    rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    latest_ts = None
    if _column_exists(conn, table, "ts"):
        latest_ts = conn.execute(f"SELECT MAX(ts) FROM {table}").fetchone()[0]
    return {"exists": True, "rows": int(rows or 0), "latest_ts": latest_ts}


def _db_stats(
    conn: sqlite3.Connection | None,
    conn_error: Exception | None = None,
) -> dict[str, Any]:
    try:
        import rag

        db_dir = Path(getattr(rag, "DB_PATH"))
        filename = str(getattr(rag, "_TELEMETRY_DB_FILENAME", "telemetry.db"))
        db_path = db_dir / filename
    except Exception:
        db_path = Path("telemetry.db")

    stats: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "accessible": conn is not None and conn_error is None,
        "tables": {},
    }
    if conn_error is not None:
        stats["error"] = repr(conn_error)
    if conn is None:
        return stats

    for table in _CORE_TABLES:
        try:
            stats["tables"][table] = _table_stats(conn, table)
        except Exception as exc:
            stats["tables"][table] = {
                "exists": None,
                "rows": None,
                "latest_ts": None,
                "error": repr(exc),
            }
    return stats


def _safe_conn_audit(
    fn: Callable[[sqlite3.Connection, int], dict[str, Any]],
    conn: sqlite3.Connection,
    days: int,
) -> dict:
    try:
        return _jsonable(fn(conn, days))
    except Exception as exc:
        return {"error": repr(exc)}


def _telemetry_errors_subsystem(
    conn: sqlite3.Connection | None,
    days: int,
    db_stats: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    status = "ok"
    try:
        error_rollup = _jsonable(_audit._audit_sql_errors(days))
    except Exception as exc:
        error_rollup = {"error": repr(exc), "total_errors": None}
        status = "degraded"
        issues.append(f"error rollup failed: {type(exc).__name__}: {exc}")

    total_errors = error_rollup.get("total_errors")
    if isinstance(total_errors, int):
        if total_errors >= _TELEMETRY_ERROR_DOWN_THRESHOLD:
            status = "down"
            issues.append(f"{total_errors} silent/sql telemetry errors in the last {days}d")
        elif total_errors >= _TELEMETRY_ERROR_DEGRADED_THRESHOLD:
            status = "degraded"
            issues.append(f"{total_errors} silent/sql telemetry errors in the last {days}d")
    if int(error_rollup.get("test_pollution_hits") or 0) > 0:
        status = "degraded" if status == "ok" else status
        issues.append("test pollution entries present in telemetry error logs")
    if not db_stats.get("accessible"):
        status = "down"
        issues.append("telemetry database is not accessible")

    details = {
        "error_rollup": error_rollup,
        "db_stats": db_stats,
        "query_latency": None,
        "cache_health": None,
    }
    if conn is not None:
        details["query_latency"] = _safe_conn_audit(_audit._audit_query_latency, conn, days)
        details["cache_health"] = _safe_conn_audit(_audit._audit_cache_health, conn, days)
    return _subsystem("telemetry_errors", "Telemetry errors", status, issues, details)


def _learning_gates_subsystem(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        gap = _jsonable(_audit._audit_feedback_corrective_gap(conn))
    except Exception as exc:
        return _failed_subsystem("learning_gates", "Learning gates", exc)

    issues: list[str] = []
    status = "ok"
    if gap.get("error"):
        status = "degraded"
        issues.append(str(gap["error"]))
    elif not bool(gap.get("gate_open")):
        status = "degraded"
        has_cp = int(gap.get("has_cp") or 0)
        threshold = int(gap.get("gate_threshold") or 0)
        to_close = int(gap.get("rows_to_close_gate") or 0)
        issues.append(
            f"corrective_path gate closed: {has_cp}/{threshold} rows "
            f"({to_close} to close)"
        )
    return _subsystem(
        "learning_gates",
        "Learning gates",
        status,
        issues,
        {"feedback_corrective_gap": gap},
    )


def _db_unavailable_subsystems(exc: Exception) -> list[dict[str, Any]]:
    issue = f"telemetry database unavailable: {type(exc).__name__}: {exc}"
    return [
        _subsystem("anticipate", "Anticipate", "down", [issue], {"error": repr(exc)}),
        _subsystem("retrieval", "Retrieval", "down", [issue], {"error": repr(exc)}),
        _subsystem("chat", "Chat", "down", [issue], {"error": repr(exc)}),
        _subsystem("learning_gates", "Learning gates", "down", [issue], {"error": repr(exc)}),
    ]


def _overall(subsystems: list[dict[str, Any]]) -> str:
    if not subsystems:
        return "degraded"
    return max((str(s.get("status") or "degraded") for s in subsystems), key=_STATUS_RANK.get)


def _add_action(
    actions: list[str],
    seen: set[str],
    action_id: str,
) -> None:
    if action_id in seen:
        return
    seen.add(action_id)
    actions.append(action_id)


def _derive_actions(subsystems: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    by_id = {str(s.get("id")): s for s in subsystems}

    anticipate = by_id.get("anticipate")
    if anticipate and anticipate.get("status") in {"degraded", "down"}:
        _add_action(
            actions,
            seen,
            "open_anticipate_inbox",
        )

    learning = by_id.get("learning_gates")
    if learning:
        gap = learning.get("details", {}).get("feedback_corrective_gap", {})
        gate_closed = "gate_open" in gap and not bool(gap.get("gate_open"))
        if gate_closed or any(
            "corrective_path gate closed" in issue
            for issue in learning.get("issues", [])
        ):
            _add_action(
                actions,
                seen,
                "run_feedback_harvest",
            )

    telemetry = by_id.get("telemetry_errors")
    if telemetry:
        rollup = telemetry.get("details", {}).get("error_rollup", {})
        total_errors = rollup.get("total_errors")
        if isinstance(total_errors, int) and total_errors >= _TELEMETRY_ERROR_ACTION_THRESHOLD:
            _add_action(
                actions,
                seen,
                "inspect_silent_errors",
            )
        db_stats = telemetry.get("details", {}).get("db_stats", {})
        if db_stats and not db_stats.get("accessible", True):
            _add_action(
                actions,
                seen,
                "check_telemetry_db",
            )

    return actions


def mission_control_health_snapshot(days: int = 7) -> dict[str, Any]:
    days = _window_days(days)
    conn_error: Exception | None = None
    conn: sqlite3.Connection | None = None
    subsystems: list[dict[str, Any]]

    try:
        import rag

        with rag._ragvec_state_conn() as live_conn:
            live_conn.row_factory = sqlite3.Row
            conn = live_conn
            stats = _db_stats(conn)
            subsystems = [
                _safe_audit_subsystem(
                    "anticipate",
                    "Anticipate",
                    lambda: _audit.check_anticipate_health(conn, days),
                ),
                _safe_audit_subsystem(
                    "retrieval",
                    "Retrieval",
                    lambda: _audit.check_retrieval_health(conn, days),
                ),
                _safe_audit_subsystem(
                    "chat",
                    "Chat",
                    lambda: _audit.check_chat_health(conn, days),
                ),
                _telemetry_errors_subsystem(conn, days, stats),
                _learning_gates_subsystem(conn),
            ]
    except Exception as exc:
        conn_error = exc
        stats = _db_stats(None, conn_error)
        subsystems = _db_unavailable_subsystems(exc)
        subsystems.insert(3, _telemetry_errors_subsystem(None, days, stats))

    overall = _overall(subsystems)
    return {
        "ok": True,
        "generated_at": _now_utc(),
        "window_days": days,
        "overall": overall,
        "subsystems": subsystems,
        "actions": _derive_actions(subsystems),
    }


def api_mission_control_health(days: int = 7) -> dict[str, Any]:
    return mission_control_health_snapshot(days=days)


def _mission_action_open_anticipate_inbox(req: MissionControlActionRequest) -> dict[str, Any]:
    from web.anticipate_routes import anticipate_inbox  # noqa: PLC0415

    inbox = anticipate_inbox(
        limit=req.limit,
        days=req.days,
        only_actionable=True,
    )
    return {
        "ok": True,
        "action_id": req.action_id,
        "dry_run": True,
        "executed": False,
        "endpoint": "/api/anticipate/inbox?only_actionable=true",
        "result": inbox,
    }


def _mission_action_run_feedback_harvest(req: MissionControlActionRequest) -> dict[str, Any]:
    import rag  # noqa: PLC0415

    days = _window_days(req.days)
    limit = max(1, min(int(req.limit), 100))
    confidence_below = max(0.0, min(float(req.confidence_below), 1.0))
    if req.dry_run:
        with rag._ragvec_state_conn() as conn:
            candidates = _jsonable(_audit._audit_harvest_candidates(conn, days))
        return {
            "ok": True,
            "action_id": req.action_id,
            "dry_run": True,
            "executed": False,
            "result": candidates,
        }

    try:
        stats = rag.auto_harvest(
            since_days=days,
            confidence_below=confidence_below,
            min_judge_conf=req.min_judge_conf,
            limit=limit,
            dry_run=False,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"feedback harvest failed: {exc}") from exc
    return {
        "ok": True,
        "action_id": req.action_id,
        "dry_run": False,
        "executed": True,
        "result": _jsonable(stats),
    }


def _mission_action_inspect_silent_errors(req: MissionControlActionRequest) -> dict[str, Any]:
    days = _window_days(req.days)
    try:
        rollup = _jsonable(_audit._audit_sql_errors(days))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"error inspection failed: {exc}") from exc
    return {
        "ok": True,
        "action_id": req.action_id,
        "dry_run": True,
        "executed": False,
        "result": rollup,
    }


def _mission_action_check_telemetry_db(req: MissionControlActionRequest) -> dict[str, Any]:
    del req
    import rag  # noqa: PLC0415

    try:
        with rag._ragvec_state_conn() as conn:
            rag._ensure_telemetry_tables(conn)
            conn.commit()
            stats = _db_stats(conn)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"telemetry db check failed: {exc}") from exc
    return {
        "ok": True,
        "action_id": "check_telemetry_db",
        "dry_run": False,
        "executed": True,
        "result": stats,
    }


def api_mission_control_action(req: MissionControlActionRequest) -> dict[str, Any]:
    action_id = req.action_id.strip()
    handlers = {
        "open_anticipate_inbox": _mission_action_open_anticipate_inbox,
        "run_feedback_harvest": _mission_action_run_feedback_harvest,
        "inspect_silent_errors": _mission_action_inspect_silent_errors,
        "check_telemetry_db": _mission_action_check_telemetry_db,
        "refresh": lambda request: {
            "ok": True,
            "action_id": "refresh",
            "dry_run": True,
            "executed": False,
            "result": mission_control_health_snapshot(days=request.days),
        },
    }
    handler = handlers.get(action_id)
    if handler is None:
        raise HTTPException(status_code=400, detail=f"unknown mission-control action: {action_id}")
    return handler(req)


def register_mission_control_routes(
    app,
    require_admin_token: Callable | None = None,
) -> dict[str, object]:
    admin_dep = [Depends(require_admin_token)] if require_admin_token is not None else []
    app.get("/api/health/mission-control")(api_mission_control_health)
    app.post(
        "/api/health/mission-control/action",
        dependencies=admin_dep,
    )(api_mission_control_action)
    return {
        "api_mission_control_health": api_mission_control_health,
        "api_mission_control_action": api_mission_control_action,
        "mission_control_health_snapshot": mission_control_health_snapshot,
    }
