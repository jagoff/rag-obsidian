"""Screen Time helpers — knowledgeC.db read path + deterministic rendering.

Uses a temp sqlite db built with the same schema knowledgeC uses for
`/app/usage`: rows have ZSTREAMNAME, ZVALUESTRING (bundle id), ZSTARTDATE
and ZENDDATE in Cocoa epoch seconds.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag


COCOA = 978307200


def _build_fake_db(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """rows: (bundle, start_cocoa, end_cocoa)"""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE ZOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            ZSTREAMNAME TEXT,
            ZVALUESTRING TEXT,
            ZSTARTDATE DOUBLE,
            ZENDDATE DOUBLE
        )
        """
    )
    for i, (bundle, s, e) in enumerate(rows, 1):
        conn.execute(
            "INSERT INTO ZOBJECT (Z_PK, ZSTREAMNAME, ZVALUESTRING, ZSTARTDATE, ZENDDATE)"
            " VALUES (?, '/app/usage', ?, ?, ?)",
            (i, bundle, s, e),
        )
    conn.commit()
    conn.close()


def _cocoa(dt: datetime) -> float:
    return dt.timestamp() - COCOA


class TestScreentimeCollect:
    def test_missing_db_returns_unavailable(self, tmp_path):
        st = rag._collect_screentime(
            datetime.now() - timedelta(hours=24), datetime.now(),
            db_path=tmp_path / "nope.db",
        )
        assert st["available"] is False
        assert st["total_secs"] == 0
        assert st["top_apps"] == []

    def test_aggregates_by_bundle_and_category(self, tmp_path):
        now = datetime(2026, 4, 16, 18, 0, 0)
        start = now - timedelta(hours=24)
        db = tmp_path / "knowledgeC.db"
        _build_fake_db(db, [
            # Two Windsurf sessions: 30m + 15m
            ("com.exafunction.windsurf", _cocoa(now - timedelta(hours=5)),
             _cocoa(now - timedelta(hours=5) + timedelta(minutes=30))),
            ("com.exafunction.windsurf", _cocoa(now - timedelta(hours=2)),
             _cocoa(now - timedelta(hours=2) + timedelta(minutes=15))),
            # WhatsApp 10m
            ("net.whatsapp.WhatsApp", _cocoa(now - timedelta(hours=3)),
             _cocoa(now - timedelta(hours=3) + timedelta(minutes=10))),
            # Chrome 20m
            ("com.google.Chrome", _cocoa(now - timedelta(hours=1)),
             _cocoa(now - timedelta(hours=1) + timedelta(minutes=20))),
            # Noise: 2s session (filtered by >=5s floor)
            ("com.apple.finder", _cocoa(now - timedelta(minutes=10)),
             _cocoa(now - timedelta(minutes=10) + timedelta(seconds=2))),
            # Outside window (before start) — must be ignored
            ("md.obsidian", _cocoa(start - timedelta(hours=2)),
             _cocoa(start - timedelta(hours=2) + timedelta(minutes=30))),
        ])

        st = rag._collect_screentime(start, now, db_path=db)
        assert st["available"] is True
        # 45 + 10 + 20 = 75 min = 4500s
        assert st["total_secs"] == 4500
        # Top app = Windsurf with 45m
        assert st["top_apps"][0]["label"] == "Windsurf"
        assert st["top_apps"][0]["secs"] == 45 * 60
        # Categories: code=2700, comms=600, browser=1200
        assert st["categories"]["code"] == 2700
        assert st["categories"]["comms"] == 600
        assert st["categories"]["browser"] == 1200
        # Finder noise filtered + Obsidian (outside window) absent
        labels = [a["label"] for a in st["top_apps"]]
        assert "Finder" not in labels
        assert "Obsidian" not in labels

    def test_unknown_bundle_falls_back_to_stem(self, tmp_path):
        now = datetime(2026, 4, 16, 18, 0, 0)
        db = tmp_path / "k.db"
        _build_fake_db(db, [
            ("com.novel.CoolApp", _cocoa(now - timedelta(hours=1)),
             _cocoa(now - timedelta(hours=1) + timedelta(minutes=5))),
        ])
        st = rag._collect_screentime(now - timedelta(hours=2), now, db_path=db)
        assert st["top_apps"][0]["label"] == "CoolApp"
        assert st["categories"].get("otros") == 300


class TestScreentimeRender:
    def test_unavailable_returns_empty(self):
        assert rag._render_screentime_section({}) == ""
        assert rag._render_screentime_section({"available": False}) == ""

    def test_below_floor_returns_empty(self):
        st = {"available": True, "total_secs": 120, "top_apps": [], "categories": {}}
        assert rag._render_screentime_section(st) == ""

    def test_renders_header_top_apps_and_categories(self):
        st = {
            "available": True,
            "total_secs": 4 * 3600 + 30 * 60,
            "top_apps": [
                {"bundle": "com.exafunction.windsurf", "label": "Windsurf", "secs": 3 * 3600},
                {"bundle": "com.googlecode.iterm2", "label": "iTerm", "secs": 45 * 60},
                {"bundle": "net.whatsapp.WhatsApp", "label": "WhatsApp", "secs": 25 * 60},
            ],
            "categories": {
                "code": 3 * 3600 + 45 * 60,
                "comms": 25 * 60,
                "browser": 20 * 60,
            },
        }
        out = rag._render_screentime_section(st)
        assert "## 🖥 Pantalla · 4h 30m activo" in out
        assert "Windsurf 3h" in out
        assert "iTerm 45m" in out
        assert "code 3h 45m" in out
        assert "comms 25m" in out

    def test_skips_subminute_category_totals(self):
        st = {
            "available": True,
            "total_secs": 30 * 60,
            "top_apps": [
                {"bundle": "md.obsidian", "label": "Obsidian", "secs": 30 * 60},
            ],
            "categories": {"notas": 30 * 60, "otros": 45},
        }
        out = rag._render_screentime_section(st)
        assert "otros" not in out
        assert "notas 30m" in out


class TestFmtHm:
    @pytest.mark.parametrize("secs,expected", [
        (0, "0s"),
        (30, "30s"),
        (60, "1m"),
        (59 * 60, "59m"),
        (3600, "1h"),
        (3660, "1h 01m"),
        (2 * 3600 + 5 * 60, "2h 05m"),
    ])
    def test_formats(self, secs, expected):
        assert rag._fmt_hm(secs) == expected
