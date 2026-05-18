"""Small persistent state helpers for index pre-sync ETLs.

The indexer uses this to make cross-source pre-syncs measurable and to avoid
re-running recently successful, vault-file-producing ETLs during repeated full
rebuilds. It is intentionally JSON-based: losing this file only makes the next
index run do fresh work.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STATE_FILENAME = "index_etl_state.json"
_RECENT_RUNS_LIMIT = 20


@dataclass(frozen=True)
class EtlFreshnessDecision:
    should_skip: bool
    age_s: float | None = None
    reason: str = ""
    previous: dict[str, Any] | None = None


def _state_path(db_dir: Path) -> Path:
    return Path(db_dir) / _STATE_FILENAME


def _load_state(db_dir: Path) -> dict[str, Any]:
    path = _state_path(db_dir)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {"sources": {}, "runs": []}
    if not isinstance(data, dict):
        return {"sources": {}, "runs": []}
    data.setdefault("sources", {})
    data.setdefault("runs", [])
    if not isinstance(data["sources"], dict):
        data["sources"] = {}
    if not isinstance(data["runs"], list):
        data["runs"] = []
    return data


def _save_state(db_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(db_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def freshness_decision(
    db_dir: Path,
    source: str,
    *,
    ttl_s: float,
    now: float | None = None,
) -> EtlFreshnessDecision:
    if ttl_s <= 0:
        return EtlFreshnessDecision(False, reason="disabled")
    state = _load_state(db_dir)
    prev = (state.get("sources") or {}).get(source)
    if not isinstance(prev, dict):
        return EtlFreshnessDecision(False, reason="no_state")
    if not prev.get("ok"):
        return EtlFreshnessDecision(False, reason="previous_not_ok", previous=prev)
    ts = prev.get("finished_ts")
    try:
        age_s = (time.time() if now is None else now) - float(ts)
    except (TypeError, ValueError):
        return EtlFreshnessDecision(False, reason="bad_timestamp", previous=prev)
    if age_s < 0:
        return EtlFreshnessDecision(False, age_s=age_s, reason="clock_skew", previous=prev)
    if age_s <= ttl_s:
        return EtlFreshnessDecision(True, age_s=age_s, reason="fresh", previous=prev)
    return EtlFreshnessDecision(False, age_s=age_s, reason="expired", previous=prev)


def record_source(
    db_dir: Path,
    source: str,
    *,
    stats: dict[str, Any],
    duration_ms: int,
    finished_ts: float | None = None,
) -> None:
    state = _load_state(db_dir)
    finished = time.time() if finished_ts is None else finished_ts
    record = {
        "finished_ts": finished,
        "ok": bool(stats.get("ok")),
        "reason": stats.get("reason"),
        "duration_ms": int(duration_ms),
        "files_written": int(stats.get("files_written") or 0),
        "target": stats.get("target"),
    }
    for key in (
        "urls",
        "youtube_videos",
        "messages",
        "events",
        "open_prs",
        "pending",
        "profiles",
        "total",
    ):
        if key in stats:
            record[key] = stats.get(key)
    state.setdefault("sources", {})[source] = record
    _save_state(db_dir, state)


def record_run(
    db_dir: Path,
    *,
    vault: str,
    reset: bool,
    total_ms: int,
    records: list[dict[str, Any]],
    finished_ts: float | None = None,
) -> None:
    state = _load_state(db_dir)
    run = {
        "finished_ts": time.time() if finished_ts is None else finished_ts,
        "vault": vault,
        "reset": bool(reset),
        "total_ms": int(total_ms),
        "records": records,
    }
    runs = list(state.get("runs") or [])
    runs.append(run)
    state["runs"] = runs[-_RECENT_RUNS_LIMIT:]
    _save_state(db_dir, state)

