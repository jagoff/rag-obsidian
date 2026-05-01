"""Tests para brief_queue() en web/fine_tunning_queries.py."""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from web.fine_tunning_queries import brief_queue


def _setup_schema(conn: sqlite3.Connection) -> None:
    """Create test tables for rag_brief_feedback, rag_ft_panel_ratings, rag_ft_active_queue_state."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rag_brief_feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            dedup_key TEXT NOT NULL,
            rating TEXT NOT NULL CHECK(rating IN ('positive', 'negative', 'mute')),
            reason TEXT,
            source TEXT DEFAULT 'wa'
        );
        CREATE TABLE IF NOT EXISTS rag_ft_panel_ratings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            stream TEXT NOT NULL,
            item_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
            label TEXT,
            comment TEXT,
            session_id TEXT
        );
        CREATE TABLE IF NOT EXISTS rag_ft_active_queue_state(
            item_id TEXT NOT NULL,
            stream TEXT NOT NULL,
            first_seen_ts TEXT NOT NULL,
            last_shown_ts TEXT NOT NULL,
            shown_count INTEGER NOT NULL DEFAULT 0,
            snoozed_until_ts TEXT,
            PRIMARY KEY (item_id, stream)
        );
    """)
    conn.commit()


@pytest.fixture
def tmp_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a temporary test database with schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    _setup_schema(conn)
    yield conn
    conn.close()


def test_brief_queue_empty(tmp_db: sqlite3.Connection) -> None:
    """Empty rag_brief_feedback table returns []."""
    result = brief_queue(tmp_db)
    assert result == []


def test_brief_queue_filters_positive_only(tmp_db: sqlite3.Connection) -> None:
    """A brief with only positive ratings should NOT appear."""
    now = datetime.now().isoformat(timespec="seconds")
    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, "04-Archive/reviews/2026-04-29-morning.md", "positive"),
    )
    tmp_db.commit()

    result = brief_queue(tmp_db)
    assert result == []


def test_brief_queue_includes_negative_dominant(tmp_db: sqlite3.Connection) -> None:
    """A brief with 2 negative and 0 positive ratings should appear."""
    now = datetime.now().isoformat(timespec="seconds")
    dedup_key = "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-29-morning.md"

    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, dedup_key, "negative"),
    )
    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, dedup_key, "negative"),
    )
    tmp_db.commit()

    result = brief_queue(tmp_db)
    assert len(result) == 1
    assert result[0]["item_id"] == dedup_key
    assert result[0]["stream"] == "brief"
    assert result[0]["rating_counts"]["negative"] == 2
    assert result[0]["rating_counts"]["positive"] == 0
    assert result[0]["rating_counts"]["mute"] == 0


def test_brief_queue_excludes_already_rated(tmp_db: sqlite3.Connection) -> None:
    """A brief that already has a row in rag_ft_panel_ratings should be excluded."""
    now = datetime.now().isoformat(timespec="seconds")
    dedup_key = "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-29-morning.md"

    # Add negative feedback
    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, dedup_key, "negative"),
    )

    # Add a panel rating for the same item
    tmp_db.execute(
        "INSERT INTO rag_ft_panel_ratings (ts, stream, item_id, rating) VALUES (?, ?, ?, ?)",
        (now, "brief", dedup_key, -1),
    )
    tmp_db.commit()

    result = brief_queue(tmp_db)
    assert result == []


def test_brief_queue_dedup_by_dedup_key(tmp_db: sqlite3.Connection) -> None:
    """Multiple rows with the same dedup_key should aggregate into one queue item."""
    now = datetime.now().isoformat(timespec="seconds")
    dedup_key = "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-29-morning.md"

    # Add 3 rows: 1 negative, 1 mute, 0 positive
    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, dedup_key, "negative"),
    )
    tmp_db.execute(
        "INSERT INTO rag_brief_feedback (ts, dedup_key, rating) VALUES (?, ?, ?)",
        (now, dedup_key, "mute"),
    )
    tmp_db.commit()

    result = brief_queue(tmp_db)
    assert len(result) == 1
    assert result[0]["item_id"] == dedup_key
    assert result[0]["rating_counts"]["negative"] == 1
    assert result[0]["rating_counts"]["mute"] == 1
    assert result[0]["rating_counts"]["positive"] == 0
