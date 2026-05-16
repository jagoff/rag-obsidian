"""Screen Time integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Source: macOS Screen Time logs at `~/Library/Application Support/Knowledge/
knowledgeC.db` (CoreDuet's [knowledge store](https://gist.github.com/mac4n6/9d44e3001b1d59d3eb1f49f5e54f4ada)).
Read-only access works without Full Disk Access as long as the file is
readable. Each row in `ZOBJECT` with `ZSTREAMNAME = '/app/usage'` represents
one foreground session — summing `ZENDDATE - ZSTARTDATE` per bundle gives
active-use seconds (NOT wall time — backgrounded apps don't count).

## Surfaces

- `_collect_screentime(start, end, db_path=None)` — per-app + per-category
  foreground usage in `[start, end)`. Returns `{available, total_secs,
  top_apps, categories}`. Sessions <5s ignored (filters spurious re-focuses).
  Unknown bundle IDs surface via their stem so brand-new apps don't hide in
  "otros".
- `_render_screentime_section(st)` — deterministic markdown for the morning
  brief. Empty if `st.available=False` or `total < 5min` (likely a sleeping
  Mac or fresh setup).
- `_screentime_app_label(bundle)` — bundle ID → human label. Falls back to
  the last dotted segment when unknown.
- `_screentime_category(bundle)` — bundle ID → coarse category (`code`,
  `notas`, `comms`, `browser`, `media`, `otros`).

## Invariants
- Silent-fail: missing DB, locked DB, sqlite error → `{available: False, ...}`.
  Never raise.
- The `immutable=1` URI flag lets us read even if macOS holds a write lock
  (read-only mode + WAL ignore — the snapshot may be slightly stale but
  morning briefs don't need second-level accuracy).
- Cocoa epoch offset: 978,307,200 seconds between 1970-01-01 (Unix) and
  2001-01-01 (Cocoa). DON'T forget to subtract it from `datetime.timestamp()`
  before passing to the SQL — the column is in Cocoa-seconds.

## Why deferred imports
`_fmt_hm` lives in `rag.__init__` (also used by `web/server.py` directly).
Module-level `from rag import _fmt_hm` here would deadlock the package load.
Function-body imports run after `rag.__init__` finishes loading.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE


# ── Screen Time (knowledgeC.db) ─────────────────────────────────────────
# macOS logs foreground app usage at `/app/usage` in CoreDuet's knowledge
# store. Read-only access works without Full Disk Access as long as the
# file is readable. Values are per foreground session; summing gives
# active-use seconds (not wall time). Categories are heuristic — bundle
# ID prefix match. Unknown apps render as bundle ID stem so new apps
# surface instead of hiding in "otros".

SCREENTIME_DB = Path.home() / "Library/Application Support/Knowledge/knowledgeC.db"
# 978307200 = seconds between 1970-01-01 and 2001-01-01 (Cocoa epoch).
_SCREENTIME_COCOA_OFFSET = 978307200

_SCREENTIME_APP_LABELS = {
    "com.exafunction.windsurf": "Windsurf",
    "com.googlecode.iterm2": "iTerm",
    "com.apple.Terminal": "Terminal",
    "com.mitchellh.ghostty": "Ghostty",
    "com.microsoft.VSCode": "VS Code",
    "com.sublimetext.4": "Sublime",
    "com.jetbrains.pycharm": "PyCharm",
    "md.obsidian": "Obsidian",
    "com.google.Chrome": "Chrome",
    "com.apple.Safari": "Safari",
    "company.thebrowser.Browser": "Arc",
    "com.brave.Browser": "Brave",
    "net.whatsapp.WhatsApp": "WhatsApp",
    "com.apple.MobileSMS": "Messages",
    "com.tinyspeck.slackmacgap": "Slack",
    "com.hnc.Discord": "Discord",
    "ru.keepcoder.Telegram": "Telegram",
    "com.apple.mail": "Mail",
    "com.apple.iCal": "Calendar",
    "com.flexibits.fantastical2.mac": "Fantastical",
    "com.apple.reminders": "Reminders",
    "com.apple.Notes": "Notes",
    "com.apple.finder": "Finder",
    "com.apple.Photos": "Photos",
    "com.apple.Music": "Music",
    "com.spotify.client": "Spotify",
    "com.apple.QuickTimePlayerX": "QuickTime",
    "com.apple.systempreferences": "System Settings",
    "com.apple.ActivityMonitor": "Activity Monitor",
    "com.figma.Desktop": "Figma",
    "com.linear": "Linear",
    "notion.id": "Notion",
    "com.apple.podcasts": "Podcasts",
}

_SCREENTIME_CATEGORIES = {
    "code": {
        "com.exafunction.windsurf", "com.googlecode.iterm2", "com.apple.Terminal",
        "com.mitchellh.ghostty",
        "com.microsoft.VSCode", "com.sublimetext.4", "com.jetbrains.pycharm",
        "com.apple.dt.Xcode", "com.todesktop.230313mzl4w4u92",  # Cursor
    },
    "notas": {"md.obsidian", "com.apple.Notes", "notion.id"},
    "comms": {
        "net.whatsapp.WhatsApp", "com.apple.MobileSMS", "com.tinyspeck.slackmacgap",
        "com.hnc.Discord", "ru.keepcoder.Telegram", "com.apple.mail", "com.apple.FaceTime",
    },
    "browser": {
        "com.google.Chrome", "com.apple.Safari", "company.thebrowser.Browser",
        "com.brave.Browser", "org.mozilla.firefox",
    },
    "media": {
        "com.apple.Music", "com.spotify.client", "com.apple.QuickTimePlayerX",
        "com.apple.podcasts", "com.apple.TV",
    },
}


def _screentime_app_label(bundle: str) -> str:
    if bundle in _SCREENTIME_APP_LABELS:
        return _SCREENTIME_APP_LABELS[bundle]
    # Fallback: last dotted segment, title-cased ("com.foo.BarApp" → "BarApp")
    return bundle.rsplit(".", 1)[-1] if "." in bundle else bundle


def _screentime_category(bundle: str) -> str:
    for cat, bundles in _SCREENTIME_CATEGORIES.items():
        if bundle in bundles:
            return cat
    return "otros"


def _collect_screentime(
    start: datetime, end: datetime,
    db_path: Path | None = None,
) -> dict:
    """Per-app foreground usage for [start, end). Returns:

    ```
    {
        "available": bool,
        "total_secs": int,
        "top_apps": [{"bundle": str, "label": str, "secs": int}],
        "categories": {"code": int, "comms": int, ...},
    }
    ```

    Silent-degrades to `available=False` if the db is missing or locked.
    Only sessions >= 5s counted (filters spurious re-focuses). Unknown
    bundles surface via their stem so new apps aren't swept into "otros".
    """
    import sqlite3

    path = db_path or SCREENTIME_DB
    empty = {"available": False, "total_secs": 0, "top_apps": [], "categories": {}}
    if not path.is_file():
        return empty

    start_ts = start.timestamp() - _SCREENTIME_COCOA_OFFSET
    end_ts = end.timestamp() - _SCREENTIME_COCOA_OFFSET
    try:
        # immutable=1 lets us read even if macOS holds a write lock.
        uri = f"file:{path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                """
                SELECT ZVALUESTRING, SUM(ZENDDATE - ZSTARTDATE) AS secs
                FROM ZOBJECT
                WHERE ZSTREAMNAME = '/app/usage'
                  AND ZSTARTDATE >= ?
                  AND ZSTARTDATE < ?
                  AND (ZENDDATE - ZSTARTDATE) >= 5
                GROUP BY ZVALUESTRING
                ORDER BY secs DESC
                """,
                (start_ts, end_ts),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        # Bug Hunt 2026-05-08 (M Int screentime): pre-fix swallowing total
        # de exceptions sin traza. Cuando macOS upgradeaba el schema o la
        # DB quedaba locked >2s, el panel screentime devolvía empty sin
        # razón visible en silent_errors_log → operador no podía
        # diferenciar "no hay datos" de "schema cambió". Compat con lazy
        # import del helper para evitar circular import al boot.
        try:
            from rag import _silent_log
            _silent_log("screentime_collect_failed", exc)
        except Exception:
            pass
        return empty

    top: list[dict] = []
    cats: dict[str, int] = {}
    total = 0
    for bundle, secs in rows:
        if not bundle or secs is None:
            continue
        s = int(round(float(secs)))
        if s <= 0:
            continue
        total += s
        top.append({
            "bundle": bundle,
            "label": _screentime_app_label(bundle),
            "secs": s,
        })
        cat = _screentime_category(bundle)
        cats[cat] = cats.get(cat, 0) + s

    return {
        "available": True,
        "total_secs": total,
        "top_apps": top[:10],
        "categories": cats,
    }


def _render_screentime_section(st: dict) -> str:
    """Deterministic "where time went" section. Empty if db unavailable
    or total < 5 min (likely sleeping Mac or brand-new setup).
    """
    from rag import _fmt_hm
    if not st or not st.get("available"):
        return ""
    total = int(st.get("total_secs") or 0)
    if total < 300:
        return ""

    lines = [f"## 🖥 Pantalla · {_fmt_hm(total)} activo"]
    top = (st.get("top_apps") or [])[:5]
    if top:
        parts = [f"{a['label']} {_fmt_hm(a['secs'])}" for a in top]
        lines.append("- " + " · ".join(parts))
    cats = st.get("categories") or {}
    if cats:
        order = ["code", "notas", "comms", "browser", "media", "otros"]
        pieces = []
        for k in order:
            v = cats.get(k, 0)
            if v >= 60:
                pieces.append(f"{k} {_fmt_hm(v)}")
        if pieces:
            lines.append("- " + " · ".join(pieces))
    return "\n".join(lines)


def _fetch_screentime_today(now: datetime, top_n: int = 5) -> dict | None:
    """Today 00:00 → now bucket de screentime para el evening brief.

    Returns the same shape as ``_collect_screentime`` con un slice de los
    top N apps + categorías. Returns ``None`` cuando la DB no está
    disponible o el día acumula < 5 min (Mac dormido / no usado) — el
    prompt builder skipea el bucket en ese caso.

    Solo data factual ("Claude.app 4h, Ghostty 2h") — NO infiere mood ni
    fatiga. El render del prompt es responsable de no moralizar.
    """
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    st = _collect_screentime(start, now)
    if not st.get("available"):
        return None
    total = int(st.get("total_secs") or 0)
    if total < 300:
        return None
    return {
        "available": True,
        "total_secs": total,
        "top_apps": (st.get("top_apps") or [])[:top_n],
        "categories": st.get("categories") or {},
    }


# ── Screen Time persistence (daily + monthly) ─────────────────────────────────
# ETL writers — extraídos de `rag/cross_source_etls.py` (2026-05-09). Mantienen
# el contrato de "silent fail + hash-skip + write only if changed" del resto
# de cross-source ETLs. La lógica está acá (no en cross_source_etls) porque
# comparte estado conceptual con `_collect_screentime` (mismo source DB,
# misma categorización de apps), y aislarlas en un solo módulo simplifica
# debug cuando una de las dos rompe (ej. schema upgrade de macOS).

SCREENTIME_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Screentime"
_SCREENTIME_BACKFILL_DAYS = 30
_SCREENTIME_DAILY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
_SCREENTIME_MONTHLY_RE = re.compile(r"^\d{4}-\d{2}\.md$")


def _sync_screentime_notes(
    vault_root: Path,
    days: int = _SCREENTIME_BACKFILL_DAYS,
    db_path: Path | None = None,
) -> dict:
    """Persist Screen Time per-app foreground usage as vault notes."""
    from collections import defaultdict
    from datetime import datetime, timedelta

    target_dir = vault_root / SCREENTIME_VAULT_SUBPATH
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {"ok": False, "reason": f"mkdir: {exc}"}

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    day_data: dict[str, dict] = {}  # "YYYY-MM-DD" → screentime dict
    months: dict[str, list[str]] = defaultdict(list)  # "YYYY-MM" → ["YYYY-MM-DD", ...]

    db_was_available = False
    for d in range(days, -1, -1):
        day_start = today - timedelta(days=d)
        day_end = day_start + timedelta(days=1)
        st = _collect_screentime(day_start, day_end, db_path=db_path)
        if not st.get("available"):
            if d == days and not (target_dir / "_index.md").is_file():
                return {"ok": False, "reason": "no_data"}
            continue
        db_was_available = True
        if int(st.get("total_secs") or 0) < 60:
            continue
        day_str = day_start.strftime("%Y-%m-%d")
        month_str = day_start.strftime("%Y-%m")
        day_data[day_str] = st
        months[month_str].append(day_str)

    if not db_was_available:
        return {"ok": False, "reason": "no_data"}
    if not day_data:
        return {"ok": True, "files_written": 0, "days_total": 0,
                "target": str(target_dir.relative_to(vault_root))}

    written = 0
    skipped = 0
    current_set: set[str] = set()

    # Daily notes
    for day_str, st in day_data.items():
        body = _render_screentime_daily_md(day_str, st)
        path = target_dir / f"{day_str}.md"
        current_set.add(path.name)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if existing == body:
            skipped += 1
            continue
        path.write_text(body, encoding="utf-8")
        written += 1

    # Monthly aggregates
    for month_str, day_list in sorted(months.items()):
        body = _render_screentime_monthly_md(
            month_str,
            [(d, day_data[d]) for d in sorted(day_list)],
        )
        path = target_dir / f"{month_str}.md"
        current_set.add(path.name)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if existing == body:
            skipped += 1
            continue
        path.write_text(body, encoding="utf-8")
        written += 1

    # Index — tabla mensual rolling
    idx_body = _render_screentime_index_md(months, day_data)
    idx_path = target_dir / "_index.md"
    current_set.add(idx_path.name)
    existing = idx_path.read_text(encoding="utf-8") if idx_path.is_file() else ""
    if existing != idx_body:
        idx_path.write_text(idx_body, encoding="utf-8")
        written += 1
    else:
        skipped += 1

    # Prune días que ya cayeron fuera de la ventana de backfill (>30d).
    for p in target_dir.glob("*.md"):
        if p.name in current_set:
            continue
        if _SCREENTIME_DAILY_RE.match(p.name) or _SCREENTIME_MONTHLY_RE.match(p.name):
            try:
                p.unlink()
            except OSError:
                pass

    total_secs = sum(int(st.get("total_secs") or 0) for st in day_data.values())
    return {
        "ok": True,
        "files_written": written,
        "files_skipped": skipped,
        "days_total": len(day_data),
        "months_total": len(months),
        "total_secs": total_secs,
        "target": str(target_dir.relative_to(vault_root)),
    }


def _render_screentime_daily_md(day_str: str, st: dict) -> str:
    """Daily note: top apps + categorías. Determinístico para que el
    hash-skip funcione."""
    from rag import _fmt_hm  # lazy
    total = int(st.get("total_secs") or 0)
    top_apps = (st.get("top_apps") or [])[:10]
    cats = st.get("categories") or {}

    lines = [
        "---",
        "type: screentime",
        f"date: {day_str}",
        f"total_active_secs: {total}",
        "ambient: skip",
        "tags: [screentime, productividad]",
        "---",
        "",
        f"# Pantalla · {day_str} · {_fmt_hm(total)} activo",
        "",
        "## Top apps",
    ]
    if top_apps:
        for a in top_apps:
            lines.append(f"- {a['label']} · {_fmt_hm(int(a['secs']))}")
    else:
        lines.append("- _sin actividad registrada_")

    if cats:
        lines.append("")
        lines.append("## Por categoría")
        order = ["code", "notas", "comms", "browser", "media", "otros"]
        for k in order:
            v = int(cats.get(k, 0) or 0)
            if v >= 60:
                lines.append(f"- {k} · {_fmt_hm(v)}")

    return "\n".join(lines) + "\n"


def _render_screentime_monthly_md(month_str: str, days: list[tuple[str, dict]]) -> str:
    """Monthly aggregate: top apps del mes + por categoría + tabla diaria."""
    from collections import defaultdict
    from rag import _fmt_hm  # lazy

    total_secs = sum(int(st.get("total_secs") or 0) for _, st in days)
    apps_total: dict[str, dict] = defaultdict(lambda: {"label": "", "secs": 0})
    cats_total: dict[str, int] = defaultdict(int)
    for _day, st in days:
        for a in (st.get("top_apps") or []):
            bundle = a.get("bundle", "")
            apps_total[bundle]["label"] = a.get("label", bundle)
            apps_total[bundle]["secs"] += int(a.get("secs") or 0)
        for k, v in (st.get("categories") or {}).items():
            cats_total[k] += int(v or 0)

    top_apps_sorted = sorted(apps_total.items(), key=lambda kv: -kv[1]["secs"])[:15]

    lines = [
        "---",
        "type: screentime-monthly",
        f"month: {month_str}",
        f"total_active_secs: {total_secs}",
        f"days_active: {len(days)}",
        "ambient: skip",
        "tags: [screentime, productividad]",
        "---",
        "",
        f"# Pantalla · {month_str} · {_fmt_hm(total_secs)} activo ({len(days)} días)",
        "",
        "## Top apps del mes",
    ]
    for _bundle, info in top_apps_sorted:
        lines.append(f"- {info['label']} · {_fmt_hm(info['secs'])}")

    lines.append("")
    lines.append("## Por categoría")
    order = ["code", "notas", "comms", "browser", "media", "otros"]
    for k in order:
        v = int(cats_total.get(k, 0))
        if v >= 60:
            lines.append(f"- {k} · {_fmt_hm(v)}")

    lines.append("")
    lines.append("## Por día")
    lines.append("| Día | Total | Top app | Top categoría |")
    lines.append("|---|---|---|---|")
    for day_str, st in days:
        total = int(st.get("total_secs") or 0)
        top = (st.get("top_apps") or [{}])[0]
        top_label = top.get("label", "—")
        top_secs = int(top.get("secs") or 0)
        cats = st.get("categories") or {}
        top_cat = max(cats.items(), key=lambda kv: kv[1])[0] if cats else "—"
        lines.append(
            f"| [[{day_str}]] | {_fmt_hm(total)} | "
            f"{top_label} ({_fmt_hm(top_secs)}) | {top_cat} |"
        )

    return "\n".join(lines) + "\n"


def _render_screentime_index_md(
    months: dict[str, list[str]], day_data: dict[str, dict]
) -> str:
    """Index note — tabla mensual con totales + top categoría."""
    from collections import defaultdict
    from rag import _fmt_hm  # lazy

    lines = [
        "---",
        "type: screentime-index",
        "ambient: skip",
        "tags: [screentime, indice, productividad]",
        "---",
        "",
        "# Pantalla — índice mensual",
        "",
        "Fuente: `~/Library/Application Support/Knowledge/knowledgeC.db` (CoreDuet).",
        "Ventana: macOS retiene ~30 días — estas notas persisten lo histórico.",
        "",
        "| Mes | Total activo | Días | Top categoría |",
        "|---|---|---:|---|",
    ]
    for month_str in sorted(months.keys()):
        day_list = months[month_str]
        total = sum(int(day_data[d].get("total_secs") or 0) for d in day_list)
        cats_total: dict[str, int] = defaultdict(int)
        for d in day_list:
            for k, v in (day_data[d].get("categories") or {}).items():
                cats_total[k] += int(v or 0)
        top_cat = (
            max(cats_total.items(), key=lambda kv: kv[1])[0] if cats_total else "—"
        )
        lines.append(
            f"| [[{month_str}]] | {_fmt_hm(total)} | {len(day_list)} | {top_cat} |"
        )
    return "\n".join(lines) + "\n"

