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


# ── Persistence: _sync_screentime_notes + renderers ─────────────────────────


def _build_multi_day_db(path: Path, days: int = 5) -> None:
    """Build a fake knowledgeC.db with `days` days of activity ending today.

    Each day has 3 fake sessions (Windsurf 30m, Chrome 20m, WhatsApp 10m)
    starting at noon — consistent enough that hash-skip can be tested.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows: list[tuple[str, float, float]] = []
    for d in range(days, -1, -1):
        day = today - timedelta(days=d)
        noon = day.replace(hour=12)
        # Windsurf 30m
        rows.append(("com.exafunction.windsurf",
                     _cocoa(noon),
                     _cocoa(noon + timedelta(minutes=30))))
        # Chrome 20m
        rows.append(("com.google.Chrome",
                     _cocoa(noon + timedelta(minutes=45)),
                     _cocoa(noon + timedelta(minutes=65))))
        # WhatsApp 10m
        rows.append(("net.whatsapp.WhatsApp",
                     _cocoa(noon + timedelta(minutes=80)),
                     _cocoa(noon + timedelta(minutes=90))))
    _build_fake_db(path, rows)


class TestSyncScreentime:
    def test_missing_db_returns_no_data(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        out = rag._sync_screentime_notes(vault, days=3, db_path=tmp_path / "nope.db")
        assert out["ok"] is False
        assert out["reason"] == "no_data"

    def test_writes_daily_and_monthly_notes(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        db = tmp_path / "knowledgeC.db"
        _build_multi_day_db(db, days=3)

        out = rag._sync_screentime_notes(vault, days=5, db_path=db)
        assert out["ok"] is True
        assert out["days_total"] >= 3  # build_multi_day_db creates 4 days
        assert out["files_written"] >= 4  # at least 3 dailies + 1 monthly + 1 index

        target = vault / "03-Resources/Screentime"
        assert target.is_dir()
        # At least one daily YYYY-MM-DD.md
        dailies = list(target.glob("2*-*-*.md"))
        # Filter out monthly (YYYY-MM.md is 7 chars, daily is 10)
        dailies = [p for p in dailies if len(p.stem) == 10]
        assert len(dailies) >= 3
        # Index file
        assert (target / "_index.md").is_file()

    def test_hash_skip_when_unchanged(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        db = tmp_path / "knowledgeC.db"
        _build_multi_day_db(db, days=2)

        out1 = rag._sync_screentime_notes(vault, days=3, db_path=db)
        assert out1["files_written"] >= 3

        # Second run: same data → all skipped
        out2 = rag._sync_screentime_notes(vault, days=3, db_path=db)
        assert out2["files_written"] == 0
        assert out2["files_skipped"] >= 3

    def test_skips_days_without_activity(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        db = tmp_path / "knowledgeC.db"
        # Solo 1 día con < 1min — no debería escribir daily de ese día
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        noon = today.replace(hour=12)
        # 30s session — bajo el floor de 60s → no genera daily note
        _build_fake_db(db, [
            ("com.google.Chrome", _cocoa(noon),
             _cocoa(noon + timedelta(seconds=30))),
        ])
        out = rag._sync_screentime_notes(vault, days=3, db_path=db)
        # ok=True (db is available) pero days_total puede ser 0 si la sesión
        # de 30s fue filtrada por el floor de 5s en _collect_screentime
        # (5s pasa, pero 30s < 60s del floor de día) → days_total=0
        assert out["ok"] is True

    def test_index_lists_all_months(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        db = tmp_path / "knowledgeC.db"
        _build_multi_day_db(db, days=2)

        rag._sync_screentime_notes(vault, days=3, db_path=db)
        idx = (vault / "03-Resources/Screentime/_index.md").read_text()
        assert "type: screentime-index" in idx
        assert "Pantalla — índice mensual" in idx
        # Wikilink format al mes actual
        cur_month = datetime.now().strftime("%Y-%m")
        assert f"[[{cur_month}]]" in idx


class TestRenderScreentimeDaily:
    def test_frontmatter_includes_date_and_total(self):
        st = {
            "available": True,
            "total_secs": 3600,
            "top_apps": [
                {"bundle": "md.obsidian", "label": "Obsidian", "secs": 3600},
            ],
            "categories": {"notas": 3600},
        }
        md = rag._render_screentime_daily_md("2026-04-26", st)
        assert "---" in md
        assert "type: screentime" in md
        assert "date: 2026-04-26" in md
        assert "total_active_secs: 3600" in md
        assert "ambient: skip" in md
        assert "Obsidian · 1h" in md
        assert "notas · 1h" in md

    def test_skips_subminute_categories(self):
        st = {
            "available": True,
            "total_secs": 1800,
            "top_apps": [{"bundle": "x", "label": "X", "secs": 1800}],
            "categories": {"code": 1800, "comms": 30},  # comms < 60s
        }
        md = rag._render_screentime_daily_md("2026-04-26", st)
        assert "code · 30m" in md
        assert "comms" not in md  # filtered (under 60s threshold)


class TestRenderScreentimeMonthly:
    def test_aggregates_apps_across_days(self):
        days = [
            ("2026-04-25", {
                "available": True, "total_secs": 1800,
                "top_apps": [{"bundle": "md.obsidian", "label": "Obsidian", "secs": 1800}],
                "categories": {"notas": 1800},
            }),
            ("2026-04-26", {
                "available": True, "total_secs": 3600,
                "top_apps": [{"bundle": "md.obsidian", "label": "Obsidian", "secs": 3600}],
                "categories": {"notas": 3600},
            }),
        ]
        md = rag._render_screentime_monthly_md("2026-04", days)
        assert "month: 2026-04" in md
        assert "days_active: 2" in md
        assert "total_active_secs: 5400" in md  # 1800 + 3600
        # Per-day table includes wikilinks
        assert "[[2026-04-25]]" in md
        assert "[[2026-04-26]]" in md
        # Aggregated app total
        assert "Obsidian · 1h 30m" in md  # 5400s = 1.5h


class TestRenderScreentimeIndex:
    def test_lists_months_with_totals(self):
        day_data = {
            "2026-04-25": {"total_secs": 3600, "categories": {"code": 3600}},
            "2026-04-26": {"total_secs": 1800, "categories": {"comms": 1800}},
            "2026-03-15": {"total_secs": 7200, "categories": {"code": 7200}},
        }
        months = {
            "2026-04": ["2026-04-25", "2026-04-26"],
            "2026-03": ["2026-03-15"],
        }
        md = rag._render_screentime_index_md(months, day_data)
        assert "Pantalla — índice mensual" in md
        assert "[[2026-04]]" in md
        assert "[[2026-03]]" in md
        # 2026-04 total = 5400s = 1h 30m
        # 2026-03 total = 7200s = 2h
        assert "1h 30m" in md
        assert "2h" in md


class TestFormatEtlDetailScreentime:
    def test_screentime_detail_format(self):
        stats = {"days_total": 5, "months_total": 1, "total_secs": 18000}
        out = rag._format_etl_detail("screentime", stats)
        assert "5 días" in out
        assert "1 meses" in out
        assert "5h" in out

    def test_zero_secs_renders_0min(self):
        stats = {"days_total": 0, "months_total": 0, "total_secs": 0}
        out = rag._format_etl_detail("screentime", stats)
        assert "0min" in out
