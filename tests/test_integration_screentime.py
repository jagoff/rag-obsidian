"""Tests para `rag.integrations.screentime` — knowledgeC.db read path.

Surfaces cubiertas:
- `_screentime_app_label(bundle)` — bundle ID → human label, fallback al
  segmento dotted.
- `_screentime_category(bundle)` — bundle ID → categoría coarse
  (code/notas/comms/browser/media/otros).
- `_collect_screentime(start, end, db_path)` — agregación per-app y
  per-category con filtros (sesiones <5s, fuera de window).
- `_render_screentime_section(st)` — markdown determinístico para el
  morning brief.

Mocking: construimos una sqlite db real en `tmp_path` con el subset
de schema que `_collect_screentime` consulta (`ZOBJECT` con
`ZSTREAMNAME='/app/usage'`, `ZVALUESTRING`, `ZSTARTDATE`, `ZENDDATE`).
Las fechas están en Cocoa epoch (segundos desde 2001-01-01 UTC).

Importa de `rag.integrations.screentime` directo para que coverage
cuente en el módulo correcto (no via re-export en `rag.<func>`).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag.integrations import screentime as st_mod


_COCOA = st_mod._SCREENTIME_COCOA_OFFSET  # 978307200


def _to_cocoa(dt: datetime) -> float:
    return dt.timestamp() - _COCOA


def _build_db(path: Path, rows: list[tuple[str, float, float]]) -> None:
    """Crea una sqlite db con schema mínimo de knowledgeC para
    `/app/usage`."""
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
            "INSERT INTO ZOBJECT (Z_PK, ZSTREAMNAME, ZVALUESTRING, ZSTARTDATE, ZENDDATE) "
            "VALUES (?, '/app/usage', ?, ?, ?)",
            (i, bundle, s, e),
        )
    conn.commit()
    conn.close()


# ── _screentime_app_label ───────────────────────────────────────────────────


@pytest.mark.parametrize("bundle,label", [
    ("com.exafunction.windsurf", "Windsurf"),
    ("md.obsidian", "Obsidian"),
    ("com.googlecode.iterm2", "iTerm"),
    ("net.whatsapp.WhatsApp", "WhatsApp"),
    ("com.spotify.client", "Spotify"),
])
def test_screentime_app_label_known_bundles(bundle, label):
    assert st_mod._screentime_app_label(bundle) == label


def test_screentime_app_label_unknown_falls_back_to_stem():
    """Bundle no listado → último segmento dotted ("com.foo.NewApp" →
    "NewApp"). Permite que apps nuevas surfaceen sin tocar la tabla."""
    assert st_mod._screentime_app_label("com.foo.bar.NovelApp") == "NovelApp"
    assert st_mod._screentime_app_label("com.basic.app") == "app"
    # Sin dot → pasa verbatim.
    assert st_mod._screentime_app_label("nopointapp") == "nopointapp"


# ── _screentime_category ────────────────────────────────────────────────────


@pytest.mark.parametrize("bundle,category", [
    ("com.exafunction.windsurf", "code"),
    ("com.microsoft.VSCode", "code"),
    ("md.obsidian", "notas"),
    ("notion.id", "notas"),
    ("net.whatsapp.WhatsApp", "comms"),
    ("com.tinyspeck.slackmacgap", "comms"),
    ("com.google.Chrome", "browser"),
    ("com.apple.Safari", "browser"),
    ("com.spotify.client", "media"),
    ("com.apple.podcasts", "media"),
])
def test_screentime_category_classifies_known_bundles(bundle, category):
    assert st_mod._screentime_category(bundle) == category


def test_screentime_category_unknown_returns_otros():
    """Cualquier bundle fuera de las categorías conocidas cae a `otros`
    (Finder, System Settings, apps random)."""
    assert st_mod._screentime_category("com.apple.finder") == "otros"
    assert st_mod._screentime_category("com.foo.bar") == "otros"
    assert st_mod._screentime_category("") == "otros"


# ── _collect_screentime ──────────────────────────────────────────────────────


def test_collect_screentime_missing_db_returns_unavailable(tmp_path):
    """DB inexistente → `available=False` con shape estable. NUNCA
    raisea (silent-fail invariant)."""
    out = st_mod._collect_screentime(
        datetime.now() - timedelta(hours=24), datetime.now(),
        db_path=tmp_path / "no-existe.db",
    )
    assert out == {
        "available": False, "total_secs": 0,
        "top_apps": [], "categories": {},
    }


def test_collect_screentime_aggregates_per_bundle_and_category(tmp_path):
    """Happy path: dos sesiones de Windsurf, una de Chrome, una de
    WhatsApp, una de finder de 2s (filtrada por floor 5s) y una vieja
    fuera de window (excluida). Verificá totales + cats."""
    now = datetime(2026, 4, 29, 18, 0, 0)
    start = now - timedelta(hours=24)
    db = tmp_path / "knowledgeC.db"

    _build_db(db, [
        # Windsurf 30m
        ("com.exafunction.windsurf",
         _to_cocoa(now - timedelta(hours=5)),
         _to_cocoa(now - timedelta(hours=5) + timedelta(minutes=30))),
        # Windsurf 15m (otra sesión, mismo bundle)
        ("com.exafunction.windsurf",
         _to_cocoa(now - timedelta(hours=2)),
         _to_cocoa(now - timedelta(hours=2) + timedelta(minutes=15))),
        # Chrome 20m
        ("com.google.Chrome",
         _to_cocoa(now - timedelta(hours=1)),
         _to_cocoa(now - timedelta(hours=1) + timedelta(minutes=20))),
        # WhatsApp 10m
        ("net.whatsapp.WhatsApp",
         _to_cocoa(now - timedelta(hours=3)),
         _to_cocoa(now - timedelta(hours=3) + timedelta(minutes=10))),
        # Finder 2s (filtered by 5s SQL floor)
        ("com.apple.finder",
         _to_cocoa(now - timedelta(minutes=20)),
         _to_cocoa(now - timedelta(minutes=20) + timedelta(seconds=2))),
        # Obsidian fuera de window (start - 2h)
        ("md.obsidian",
         _to_cocoa(start - timedelta(hours=2)),
         _to_cocoa(start - timedelta(hours=2) + timedelta(minutes=30))),
    ])

    out = st_mod._collect_screentime(start, now, db_path=db)
    assert out["available"] is True
    # 30 + 15 + 20 + 10 = 75min = 4500s
    assert out["total_secs"] == 4500

    # Top: Windsurf (45m) primero.
    labels = [a["label"] for a in out["top_apps"]]
    assert labels[0] == "Windsurf"
    assert out["top_apps"][0]["secs"] == 45 * 60

    # Categorías agregadas.
    assert out["categories"]["code"] == 45 * 60      # Windsurf
    assert out["categories"]["browser"] == 20 * 60    # Chrome
    assert out["categories"]["comms"] == 10 * 60      # WhatsApp

    # Ruido + fuera-window excluidos.
    assert "Finder" not in labels
    assert "Obsidian" not in labels


def test_collect_screentime_unknown_bundle_uses_stem_label(tmp_path):
    """App nueva (no en `_SCREENTIME_APP_LABELS`) debe surfacear con su
    stem como label, en categoría `otros`."""
    now = datetime(2026, 4, 29, 18, 0)
    db = tmp_path / "k.db"
    _build_db(db, [
        ("com.example.NovelApp",
         _to_cocoa(now - timedelta(hours=1)),
         _to_cocoa(now - timedelta(hours=1) + timedelta(minutes=5))),
    ])
    out = st_mod._collect_screentime(
        now - timedelta(hours=2), now, db_path=db,
    )
    assert out["available"] is True
    assert out["top_apps"][0]["label"] == "NovelApp"
    assert out["categories"]["otros"] == 5 * 60


def test_collect_screentime_corrupted_db_silent_fails(tmp_path):
    """Si el archivo existe pero NO es sqlite válido (corrupción /
    truncado), el catch-all → `available=False`."""
    bad_db = tmp_path / "broken.db"
    bad_db.write_bytes(b"not a sqlite file at all, just garbage")
    out = st_mod._collect_screentime(
        datetime.now() - timedelta(hours=1), datetime.now(),
        db_path=bad_db,
    )
    assert out["available"] is False
    assert out["total_secs"] == 0


# ── _render_screentime_section ──────────────────────────────────────────────


def test_render_screentime_section_unavailable_returns_empty():
    """`available=False` o dict vacío → string vacío (no se renderea
    nada en el brief)."""
    assert st_mod._render_screentime_section({}) == ""
    assert st_mod._render_screentime_section({"available": False}) == ""
    assert st_mod._render_screentime_section({"available": False, "total_secs": 999}) == ""


def test_render_screentime_section_below_floor_returns_empty():
    """`total_secs < 300` (5 min) → string vacío. Mac dormida o setup
    fresh → no rendereamos una sección casi vacía."""
    st = {"available": True, "total_secs": 60,
          "top_apps": [], "categories": {}}
    assert st_mod._render_screentime_section(st) == ""


def test_render_screentime_section_renders_header_top_apps_categories():
    """Happy path: header con total formateado, top apps con minutos,
    categorías con cuenta agregada."""
    st = {
        "available": True,
        "total_secs": 4 * 3600 + 30 * 60,
        "top_apps": [
            {"bundle": "com.exafunction.windsurf", "label": "Windsurf",
             "secs": 3 * 3600},
            {"bundle": "com.google.Chrome", "label": "Chrome",
             "secs": 60 * 60},
            {"bundle": "net.whatsapp.WhatsApp", "label": "WhatsApp",
             "secs": 30 * 60},
        ],
        "categories": {
            "code": 3 * 3600,
            "browser": 60 * 60,
            "comms": 30 * 60,
        },
    }
    out = st_mod._render_screentime_section(st)
    assert "## 🖥 Pantalla · 4h 30m activo" in out
    assert "Windsurf 3h" in out
    assert "Chrome 1h" in out
    assert "code 3h" in out
    assert "comms 30m" in out


def test_render_screentime_section_skips_subminute_categories():
    """Categorías con <60s no se rendereran (ruido)."""
    st = {
        "available": True,
        "total_secs": 30 * 60,
        "top_apps": [
            {"bundle": "md.obsidian", "label": "Obsidian", "secs": 30 * 60},
        ],
        "categories": {"notas": 30 * 60, "otros": 30},  # 30s < 60s threshold
    }
    out = st_mod._render_screentime_section(st)
    assert "notas 30m" in out
    assert "otros" not in out


# ── Invariants del módulo ───────────────────────────────────────────────────


def test_cocoa_offset_matches_2001_epoch():
    """Sanity: el offset entre Unix epoch (1970) y Cocoa epoch (2001)
    es exactamente 978,307,200 segundos. Si alguien lo cambia rompe TODO
    el path de fechas."""
    expected = (datetime(2001, 1, 1) - datetime(1970, 1, 1)).total_seconds()
    assert st_mod._SCREENTIME_COCOA_OFFSET == int(expected)
