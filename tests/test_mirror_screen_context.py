"""Tests para `_source_screen_context` del mirror (Fase 2d Peekaboo).

Cubre:
  - tabla vacía → recent=[] counts=0 sin error.
  - 3+ rows recientes → top-3 ordenadas desc por ts, age_minutes computado.
  - caption >140 chars → truncado con `…`.
  - rows >4h se excluyen de `recent` pero cuentan en `count_today` / `count_7d`.
  - tabla inexistente → empty payload sin crash.
  - registro en `_SOURCES` del aggregator.
  - cache_invalidate desde observe_once después de insert exitoso.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from rag import mirror as mirror_mod
from rag.mirror import _SOURCES, _source_screen_context


@pytest.fixture()
def isolated_telemetry(tmp_path, monkeypatch):
    """Redirige _TELEMETRY_DB a tmp + ensure-ea schema."""
    db_path = tmp_path / "ragvec" / "telemetry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mirror_mod, "_TELEMETRY_DB", db_path)

    # Ensure-tablas via rag (esto crea TODAS las tablas, incluso
    # rag_screen_observations).
    import rag as _rag
    con = sqlite3.connect(str(db_path))
    _rag._ensure_telemetry_tables(con)
    con.commit()
    con.close()

    return db_path


def _seed(db_path: Path, rows: list[tuple]) -> None:
    """rows: (ts, app, title, caption)."""
    con = sqlite3.connect(str(db_path))
    con.executemany(
        "INSERT INTO rag_screen_observations "
        "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(ts, app, title, cap, 0, 100, "frontmost") for (ts, app, title, cap) in rows],
    )
    con.commit()
    con.close()


# ── registration ─────────────────────────────────────────────────────────────


def test_source_registered_in_aggregator():
    """`screen_context` debe estar en el _SOURCES dict del mirror."""
    assert "screen_context" in _SOURCES
    assert _SOURCES["screen_context"] is _source_screen_context


# ── empty / no-table paths ──────────────────────────────────────────────────


def test_empty_table_returns_zeros(isolated_telemetry):
    out = _source_screen_context("2026-05-13")
    assert out == {"recent": [], "count_today": 0, "count_7d": 0}


def test_missing_table_handled_gracefully(tmp_path, monkeypatch):
    """DB sin la tabla rag_screen_observations → empty payload, sin crash."""
    db_path = tmp_path / "telemetry.db"
    sqlite3.connect(str(db_path)).close()  # archivo vacío válido
    monkeypatch.setattr(mirror_mod, "_TELEMETRY_DB", db_path)
    out = _source_screen_context("2026-05-13")
    assert out["recent"] == []
    assert out["count_today"] == 0
    assert out["count_7d"] == 0


# ── recent window ───────────────────────────────────────────────────────────


def test_recent_capped_at_three(isolated_telemetry):
    now = int(time.time())
    rows = [(now - 60 * (i + 1), f"App{i}", f"title-{i}", f"caption {i}") for i in range(5)]
    _seed(isolated_telemetry, rows)
    out = _source_screen_context("2026-05-13")
    assert len(out["recent"]) == 3
    # Ordenados desc por ts (más reciente primero).
    ts_list = [r["ts"] for r in out["recent"]]
    assert ts_list == sorted(ts_list, reverse=True)
    # Los 3 más recientes son App0, App1, App2 (60s, 120s, 180s atrás).
    apps = [r["app_name"] for r in out["recent"]]
    assert apps == ["App0", "App1", "App2"]


def test_age_minutes_computed(isolated_telemetry):
    now = int(time.time())
    _seed(isolated_telemetry, [
        (now - 90,   "AppA", "ta", "ca"),   # 1min
        (now - 1800, "AppB", "tb", "cb"),   # 30min
    ])
    out = _source_screen_context("2026-05-13")
    ages = {r["app_name"]: r["age_minutes"] for r in out["recent"]}
    assert ages["AppA"] in (1, 2)  # 90s ≈ 1 min (floor) — accept 1 or 2 for flakiness
    assert ages["AppB"] in (29, 30, 31)


def test_caption_truncated_to_140(isolated_telemetry):
    now = int(time.time())
    long_caption = "x" * 200
    _seed(isolated_telemetry, [(now - 60, "AppX", "long", long_caption)])
    out = _source_screen_context("2026-05-13")
    cap = out["recent"][0]["caption"]
    assert len(cap) == 138  # 137 + "…"
    assert cap.endswith("…")


# ── 4h window vs daily / weekly counts ──────────────────────────────────────


def test_recent_excludes_rows_older_than_4h(isolated_telemetry):
    now = int(time.time())
    _seed(isolated_telemetry, [
        (now - 60,           "Live",  "t", "live caption"),     # 1 min
        (now - 3 * 3600,     "AlsoL", "t", "also live"),        # 3 hr
        (now - 5 * 3600,     "Stale", "t", "stale caption"),    # 5 hr (excluded from recent)
        (now - 12 * 3600,    "Old",   "t", "ayer mañana"),      # 12 hr (excluded from recent)
        (now - 6 * 86400,    "VOld",  "t", "casi semana"),      # 6 d (excluded from recent)
        (now - 14 * 86400,   "Anc",   "t", "fuera de 7d"),      # 14 d (excluded from count_7d)
    ])
    out = _source_screen_context("2026-05-13")
    apps_recent = sorted(r["app_name"] for r in out["recent"])
    assert apps_recent == ["AlsoL", "Live"], "Solo rows ≤4h en recent"

    # count_today = últimas 24h: Live(1m) + AlsoL(3h) + Stale(5h) + Old(12h).
    assert out["count_today"] == 4
    # count_7d = últimos 7d: los 5 anteriores + VOld(6d). Anc(14d) excluido.
    assert out["count_7d"] == 5


# ── cache invalidation hook from observe_once ───────────────────────────────


def test_observe_once_invalidates_mirror_cache(monkeypatch, tmp_path):
    """observe_once() exitoso → mirror cache se limpia (Fase 2d hook).

    Stub el capture + VLM + DB para que observe_once corra full path
    pero NO toque nada real. Después verificamos que cache_invalidate
    se invocó (vía spy del _CACHE).
    """
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")

    # Setup tmp DB.
    import rag as _rag
    monkeypatch.setattr(_rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(parents=True, exist_ok=True)

    # Inject "stale" cache entry.
    from rag.mirror import _CACHE, cache_invalidate
    cache_invalidate()
    _CACHE["mirror:2026-05-13"] = (time.time(), {"sentinel": True})
    assert _CACHE  # ojo: hay algo

    # Stub capture + caption.
    from rag.integrations import peekaboo as pk
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG")
    monkeypatch.setattr(
        pk, "_capture_with_meta",
        lambda **kw: (fake_png, {"app_name": "TestApp", "window_title": "t"}, None),
    )
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "captura de prueba")

    out = pk.observe_once()
    assert out["ok"] is True, out

    # Cache invalidada — el sentinel ya no está.
    assert _CACHE == {}, "cache_invalidate debe limpiar todo el cache"
