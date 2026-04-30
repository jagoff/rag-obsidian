"""Tests para `rag.integrations.chrome_bookmarks` — leaf ETL de Chrome.

Surfaces cubiertas:
- `_chrome_to_unix_ts(chrome_us)` — conversión inversa Webkit FILETIME →
  Unix timestamp. Roundtrip-seguro contra `_unix_to_chrome_ts` (que vive
  en `rag.__init__`).
- `_chrome_bookmarks_root()` — resuelve `~/Library/Application Support/
  Google/Chrome`. Usado por callers que listan profiles. Test verifica
  que respete monkeypatch de `Path.home`.
- `_fetch_chrome_bookmarks_used(hours, n)` — top-n bookmarks visitados
  en últimas `hours`. Pipeline: parse JSON tree (recursive con folder
  breadcrumbs) → tmp-copy History sqlite → join por URL → sort por
  `last_visit` desc → cap a n.
- `_fetch_youtube_today(now, n)` — videos de YouTube abiertos HOY.
  Mismo pattern de tmp-copy + filter SQL → dedup por video_id (incl.
  short URLs `youtu.be/...`) → trunc a n.

Mocking strategy:
- `Path.home()` se monkeypatchea para que apunte a `tmp_path`.
  Las funciones leen archivos `Path.home() / "Library/Application Support/
  Google/Chrome/Default/{Bookmarks,History}"`, así que poblamos esa shape
  bajo `tmp_path` y verificamos parsing.
- `History` es un sqlite real con schema mínimo (`urls` + `visits`).
  El módulo lo copia a tmp via `shutil.copyfile` antes de leer; eso
  funciona en sandbox.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from rag.integrations import chrome_bookmarks as cb_mod


_CHROME_EPOCH_OFFSET_S = 11_644_473_600


def _unix_to_chrome_us(unix_ts: float) -> int:
    """Helper: Unix ts (s) → Chrome FILETIME microseconds."""
    return int((unix_ts + _CHROME_EPOCH_OFFSET_S) * 1_000_000)


def _seed_history_db(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    """Crea sqlite con schema Chrome `urls` + `visits`. Cada row de
    `rows` = (url_id, url, title, visit_chrome_us)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE urls ("
        "id INTEGER PRIMARY KEY, url TEXT, title TEXT, visit_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE visits ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, url INTEGER, visit_time INTEGER)"
    )
    for url_id, url, title, visit_us in rows:
        conn.execute(
            "INSERT OR IGNORE INTO urls (id, url, title, visit_count) "
            "VALUES (?, ?, ?, 1)",
            (url_id, url, title),
        )
        conn.execute(
            "INSERT INTO visits (url, visit_time) VALUES (?, ?)",
            (url_id, visit_us),
        )
    conn.commit()
    conn.close()


def _chrome_dir(home: Path) -> Path:
    return home / "Library" / "Application Support" / "Google" / "Chrome" / "Default"


# ── _chrome_to_unix_ts ──────────────────────────────────────────────────


def test_chrome_to_unix_ts_roundtrip():
    """Roundtrip: Unix → Chrome (via `rag._unix_to_chrome_ts`) → Unix
    (via `cb_mod._chrome_to_unix_ts`) debe preservar el timestamp."""
    import rag
    now = time.time()
    chrome = rag._unix_to_chrome_ts(now)
    back = cb_mod._chrome_to_unix_ts(chrome)
    assert abs(back - now) < 0.001


def test_chrome_to_unix_ts_known_constants():
    """Sanity con fixed values: 0us → -11644473600 (epoch Unix de
    Webkit). Una visita a 11644473600000000 (= 1970-01-01) debe
    devolver 0.0."""
    assert cb_mod._chrome_to_unix_ts(0) == -_CHROME_EPOCH_OFFSET_S
    assert cb_mod._chrome_to_unix_ts(_CHROME_EPOCH_OFFSET_S * 1_000_000) == 0.0


# ── _chrome_bookmarks_root ──────────────────────────────────────────────


def test_chrome_bookmarks_root_under_home(monkeypatch, tmp_path):
    """`_chrome_bookmarks_root()` resuelve `Path.home()` a runtime, así
    que el monkeypatch lo respeta."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    root = cb_mod._chrome_bookmarks_root()
    assert root == tmp_path / "Library" / "Application Support" / "Google" / "Chrome"


# ── _fetch_chrome_bookmarks_used: short-circuit paths ───────────────────


def test_fetch_chrome_bookmarks_used_returns_empty_when_files_missing(
    monkeypatch, tmp_path,
):
    """Sin Bookmarks ni History en `~/Library/.../Chrome/Default` → []
    silenciosamente."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    out = cb_mod._fetch_chrome_bookmarks_used(hours=48, n=5)
    assert out == []


def test_fetch_chrome_bookmarks_used_returns_empty_on_corrupt_json(
    monkeypatch, tmp_path,
):
    """JSON malformed en Bookmarks → silent-fail []. History puede
    existir pero NO se toca (return early)."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)
    (chrome / "Bookmarks").write_text("not json at all", encoding="utf-8")
    _seed_history_db(chrome / "History", [(1, "https://x.com", "X", 0)])
    out = cb_mod._fetch_chrome_bookmarks_used()
    assert out == []


def test_fetch_chrome_bookmarks_used_returns_empty_on_empty_bookmarks_tree(
    monkeypatch, tmp_path,
):
    """`Bookmarks` válido pero sin `roots` o todos los roots vacíos →
    `bookmarks={}` → return []."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)
    (chrome / "Bookmarks").write_text(
        json.dumps({"roots": {"bookmark_bar": {"children": []}}}),
        encoding="utf-8",
    )
    _seed_history_db(chrome / "History", [
        (1, "https://x.com", "X", _unix_to_chrome_us(time.time() - 3600)),
    ])
    out = cb_mod._fetch_chrome_bookmarks_used()
    assert out == []


# ── _fetch_chrome_bookmarks_used: happy path with nested folders ────────


def test_fetch_chrome_bookmarks_used_parses_nested_folders_and_joins_history(
    monkeypatch, tmp_path,
):
    """End-to-end: bookmark anidado en folder, visitado en últimas 48h →
    surface con `name`, `folder` (breadcrumb), `visit_count`,
    `last_visit_iso`."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)

    bookmarks_tree = {
        "roots": {
            "bookmark_bar": {
                "name": "Bookmarks bar",
                "type": "folder",
                "children": [
                    {
                        "type": "url",
                        "name": "Anthropic docs",
                        "url": "https://docs.anthropic.com/",
                    },
                    {
                        "type": "folder",
                        "name": "Work",
                        "children": [
                            {
                                "type": "url",
                                "name": "Console",
                                "url": "https://console.anthropic.com/",
                            },
                        ],
                    },
                ],
            },
            "other": {"name": "Other", "type": "folder", "children": []},
        },
    }
    (chrome / "Bookmarks").write_text(
        json.dumps(bookmarks_tree), encoding="utf-8",
    )

    now = time.time()
    fresh_us = _unix_to_chrome_us(now - 3600)
    stale_us = _unix_to_chrome_us(now - 72 * 3600)
    _seed_history_db(chrome / "History", [
        (1, "https://docs.anthropic.com/", "Anthropic docs", fresh_us),
        (2, "https://console.anthropic.com/", "Console", fresh_us),
        (3, "https://random.com/", "Random", fresh_us),
        (4, "https://stale.com/", "Stale", stale_us),
    ])

    out = cb_mod._fetch_chrome_bookmarks_used(hours=48, n=10)
    urls = {item["url"] for item in out}
    assert "https://docs.anthropic.com/" in urls
    assert "https://console.anthropic.com/" in urls
    assert "https://random.com/" not in urls
    assert "https://stale.com/" not in urls

    nested = next(
        i for i in out if i["url"] == "https://console.anthropic.com/"
    )
    assert "Work" in nested["folder"]
    datetime.fromisoformat(nested["last_visit_iso"])


def test_fetch_chrome_bookmarks_used_caps_to_n(
    monkeypatch, tmp_path,
):
    """Más matches que `n` → trunc a n."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)

    children = [
        {
            "type": "url",
            "name": f"Site {i}",
            "url": f"https://site{i}.example/",
        }
        for i in range(10)
    ]
    (chrome / "Bookmarks").write_text(
        json.dumps({"roots": {"bookmark_bar": {
            "name": "Bar", "children": children,
        }}}),
        encoding="utf-8",
    )

    now = time.time()
    rows = [
        (i + 1, f"https://site{i}.example/", f"Site {i}",
         _unix_to_chrome_us(now - (i + 1) * 60))
        for i in range(10)
    ]
    _seed_history_db(chrome / "History", rows)

    out = cb_mod._fetch_chrome_bookmarks_used(hours=48, n=3)
    assert len(out) == 3


# ── _fetch_youtube_today ────────────────────────────────────────────────


def test_fetch_youtube_today_returns_empty_when_history_missing(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    out = cb_mod._fetch_youtube_today(datetime.now(), n=5)
    assert out == []


def test_fetch_youtube_today_dedups_by_video_id_and_orders_recent_first(
    monkeypatch, tmp_path,
):
    """3 visitas a YouTube hoy: dos con `?v=abc` (different params) y una
    a `youtu.be/abc` (mismo video_id). Output debe deduplicar a 1 entry
    con last_visit el más reciente."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)

    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    early_us = _unix_to_chrome_us(now.timestamp() - 3 * 3600)
    late_us = _unix_to_chrome_us(now.timestamp() - 1800)
    short_us = _unix_to_chrome_us(now.timestamp() - 7200)

    rows = [
        (1, "https://www.youtube.com/watch?v=abc&t=10",
         "Cool video - YouTube", early_us),
        (2, "https://www.youtube.com/watch?v=abc",
         "Cool video - YouTube", late_us),
        (3, "https://youtu.be/xyz",
         "Another vid", short_us),
        (4, "https://news.example.com/article",
         "News", late_us),
    ]
    _seed_history_db(chrome / "History", rows)

    out = cb_mod._fetch_youtube_today(now, n=5)
    video_ids = [item["video_id"] for item in out]
    assert "abc" in video_ids
    assert "xyz" in video_ids
    assert video_ids.count("abc") == 1
    for item in out:
        assert "youtube.com" in item["url"] or "youtu.be" in item["url"]
    assert out[0]["video_id"] == "abc"


def test_fetch_youtube_today_strips_youtube_suffix_from_title(
    monkeypatch, tmp_path,
):
    """Chrome guarda títulos como `<video> - YouTube`. La función debe
    quitar el sufijo cuando aparece."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)

    now = datetime.now().replace(hour=12)
    visit_us = _unix_to_chrome_us(now.timestamp() - 3600)
    _seed_history_db(chrome / "History", [
        (1, "https://www.youtube.com/watch?v=demo",
         "El título copado - YouTube", visit_us),
    ])
    out = cb_mod._fetch_youtube_today(now, n=5)
    assert len(out) == 1
    assert out[0]["title"] == "El título copado"


def test_fetch_youtube_today_excludes_pre_today_visits(
    monkeypatch, tmp_path,
):
    """Visita a un video YouTube DE AYER no debe aparecer en `today` —
    el filter usa today_start (00:00 local) como cutoff inferior."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    chrome = _chrome_dir(tmp_path)
    chrome.mkdir(parents=True)

    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_us = _unix_to_chrome_us(today_start.timestamp() - 3600)
    today_us = _unix_to_chrome_us(today_start.timestamp() + 3600)
    _seed_history_db(chrome / "History", [
        (1, "https://www.youtube.com/watch?v=hoy", "Hoy", today_us),
        (2, "https://www.youtube.com/watch?v=ayer", "Ayer", yesterday_us),
    ])
    out = cb_mod._fetch_youtube_today(now, n=10)
    video_ids = [it["video_id"] for it in out]
    assert "hoy" in video_ids
    assert "ayer" not in video_ids
