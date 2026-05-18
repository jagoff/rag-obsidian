"""Server-side persistence for browser UI layout state.

The browser still keeps a localStorage copy for instant reads, but this
SQLite store is the canonical fallback for local domains such as ra.ai and
survives service restarts.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def _default_state_dir() -> Path:
    return Path(
        os.environ.get("OBSIDIAN_RAG_STATE_DIR")
        or str(Path.home() / ".local/share/obsidian-rag")
    ).expanduser()


_DB_PATH = (
    Path(os.environ["RAG_UI_LAYOUT_DB_PATH"]).expanduser()
    if os.environ.get("RAG_UI_LAYOUT_DB_PATH")
    else _default_state_dir() / "ui-layout.db"
)


def ui_layout_db_path() -> Path:
    return _DB_PATH


def _clean_name(name: str, *, field: str) -> str:
    value = str(name or "").strip()
    if not _NAME_RE.match(value):
        raise ValueError(f"{field} inválido")
    return value


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH, timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ui_layout_state (
          page TEXT NOT NULL,
          key TEXT NOT NULL,
          value_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(page, key)
        )
        """
    )
    return con


def load_layout_with_updated_at(page: str) -> tuple[dict[str, Any], dict[str, str]]:
    clean_page = _clean_name(page, field="page")
    with _connect() as con:
        rows = con.execute(
            """
            SELECT key, value_json, updated_at
            FROM ui_layout_state
            WHERE page = ?
            ORDER BY key
            """,
            (clean_page,),
        ).fetchall()
    out: dict[str, Any] = {}
    updated_at: dict[str, str] = {}
    for key, raw, updated in rows:
        clean_key = str(key)
        try:
            out[clean_key] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        updated_at[clean_key] = str(updated or "")
    return out, updated_at


def load_layout(page: str) -> dict[str, Any]:
    state, _updated_at = load_layout_with_updated_at(page)
    return state
    return out


def save_layout_item(page: str, key: str, value: Any) -> bool:
    clean_page = _clean_name(page, field="page")
    clean_key = _clean_name(key, field="key")
    if value is None:
        return delete_layout_item(clean_page, clean_key)
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        before = con.execute(
            "SELECT value_json FROM ui_layout_state WHERE page = ? AND key = ?",
            (clean_page, clean_key),
        ).fetchone()
        con.execute(
            """
            INSERT INTO ui_layout_state(page, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(page, key) DO UPDATE SET
              value_json = excluded.value_json,
              updated_at = excluded.updated_at
            """,
            (clean_page, clean_key, raw, now),
        )
    return before is None or before[0] != raw


def delete_layout_item(page: str, key: str) -> bool:
    clean_page = _clean_name(page, field="page")
    clean_key = _clean_name(key, field="key")
    with _connect() as con:
        cur = con.execute(
            "DELETE FROM ui_layout_state WHERE page = ? AND key = ?",
            (clean_page, clean_key),
        )
    return cur.rowcount > 0


def replace_layout(page: str, state: dict[str, Any]) -> None:
    clean_page = _clean_name(page, field="page")
    cleaned: dict[str, str] = {}
    for key, value in (state or {}).items():
        clean_key = _clean_name(str(key), field="key")
        if value is None:
            continue
        cleaned[clean_key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as con:
        con.execute("DELETE FROM ui_layout_state WHERE page = ?", (clean_page,))
        con.executemany(
            """
            INSERT INTO ui_layout_state(page, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            [(clean_page, key, raw, now) for key, raw in cleaned.items()],
        )


def clear_layout(page: str) -> bool:
    clean_page = _clean_name(page, field="page")
    with _connect() as con:
        cur = con.execute("DELETE FROM ui_layout_state WHERE page = ?", (clean_page,))
    return cur.rowcount > 0
