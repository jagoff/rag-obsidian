"""MOZE finanzas ETL — extracted from rag/cross_source_etls.py 2026-05-09.

MOZE is the export format of the Tally4 (Money) app. The user's iCloud
container at ``Library/Mobile Documents/iCloud~amoos~Tally4/Documents``
holds periodic ``MOZE_*.csv`` exports (and ``MOZE_*.zip`` backups that
``rag.integrations.tally4_realm.ensure_moze_csv`` extracts on the fly).

This ETL parses the newest CSV, groups rows by ``(year, month)``, and
renders one deterministic markdown note per month under
``99-obsidian/99-AI/external-ingest/Finanzas/MOZE/<YYYY-MM>.md`` plus an
``_index.md`` roll-up. Deterministic output keeps content hashes stable so
``_run_index`` skips unchanged months.

Silent-fail contract: helpers return ``None`` /
``{ok: False, reason: "..."}`` instead of raising. ``_etl_log_swallow`` is
lazy-imported from ``rag.cross_source_etls`` to avoid circular import.

Override paths (env vars, read at module-import time):
  - ``OBSIDIAN_RAG_MOZE_DIR`` → ``MOZE_BACKUP_DIR`` (where Tally4 dumps CSVs).
  - ``OBSIDIAN_RAG_MOZE_FOLDER`` → ``MOZE_VAULT_SUBPATH`` (where notes land
    inside the vault). Default: ``99-obsidian/99-AI/external-ingest/Finanzas/MOZE``.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "MOZE_BACKUP_DIR",
    "MOZE_VAULT_SUBPATH",
    "MOZE_MONTH_ES",
    "_moze_cache_dir",
    "_moze_pnum",
    "_moze_fmt_ars",
    "_moze_parse_latest",
    "_moze_render_month",
    "_sync_moze_notes",
]

# MOZE (Tally4 app) export → vive en su propio container iCloud desde el
# 2026-05-04. Antes compartía dir con los xlsx de tarjetas (CloudDocs/Finances)
# pero el user separó las fuentes. Override: `OBSIDIAN_RAG_MOZE_DIR`.
MOZE_BACKUP_DIR = Path(
    os.environ.get("OBSIDIAN_RAG_MOZE_DIR", "")
    or (Path.home() / "Library/Mobile Documents/iCloud~amoos~Tally4/Documents")
)

MOZE_VAULT_SUBPATH = os.environ.get(
    "OBSIDIAN_RAG_MOZE_FOLDER",
    f"{_EXTERNAL_INGEST_BASE}/Finanzas/MOZE",
)
MOZE_MONTH_ES = [
    "", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _moze_cache_dir() -> Path | None:
    """Cache dir donde `tally4_realm.ensure_moze_csv` deja los CSV
    generados a partir del backup `.zip` de Tally4. Lazy import para no
    cargar el módulo en boot si nunca se necesita."""
    try:
        from rag.integrations.tally4_realm import CACHE_DIR
        return CACHE_DIR
    except Exception:
        return None


def _moze_pnum(s: str) -> float:
    """Parse MOZE numeric column: ES decimals, optional thousand dots."""
    s = (s or "").strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def _moze_fmt_ars(n: float) -> str:
    """$3.470.209 — ES-AR thousands, no decimals."""
    v = int(round(abs(n)))
    s = f"{v:,}".replace(",", ".")
    return f"${s}"


def _moze_parse_latest() -> tuple[Path, list[tuple[datetime, dict]]] | None:
    """Find newest MOZE_*.csv and parse into (date, row) tuples. Dates are
    MM/DD/YYYY; rows with unparseable dates are skipped.

    Post 2026-05-04: si Tally4 dejó un `MOZE_*.zip` más nuevo que el
    último CSV, lo extraemos primero (silent-fail) — ver
    `rag.integrations.tally4_realm`.
    """
    from rag.cross_source_etls import _etl_log_swallow

    try:
        from rag.integrations import tally4_realm
        tally4_realm.ensure_moze_csv(MOZE_BACKUP_DIR)
    except Exception as exc:
        _etl_log_swallow("moze_tally4_ensure_csv", exc)

    try:
        csvs: list[Path] = []
        for d in (MOZE_BACKUP_DIR, _moze_cache_dir()):
            if d and d.exists():
                csvs.extend(d.glob("MOZE_*.csv"))
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as exc:
        _etl_log_swallow("moze_csv_listing", exc)
        return None
    if not csvs:
        return None
    src = csvs[0]
    rows: list[tuple[datetime, dict]] = []
    try:
        import csv
        with src.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                raw = (r.get("Date") or "").strip()
                if not raw:
                    continue
                try:
                    d = datetime.strptime(raw, "%m/%d/%Y")
                except ValueError:
                    continue
                rows.append((d, r))
    except Exception as exc:
        _etl_log_swallow("moze_csv_parse", exc)
        return None
    return (src, rows) if rows else None


def _moze_render_month(year: int, month: int, rows: list[tuple[datetime, dict]],
                       prev_rows: list[tuple[datetime, dict]],
                       source_name: str, source_mtime: float) -> str:
    """Render one monthly note. `rows`/`prev_rows` are pre-filtered to month.
    Deterministic — same inputs → same output — so the file hash is stable
    and the indexer skips unchanged months.
    """
    from collections import Counter

    def _sum_by(rs, cur_only="ARS"):
        tot = 0.0
        for _, r in rs:
            if (r.get("Currency") or "").strip() != cur_only:
                continue
            if (r.get("Type") or "").strip() != "Expense":
                continue
            tot += abs(_moze_pnum(r.get("Price")))
        return tot

    ars_this = _sum_by(rows, "ARS")
    ars_prev = _sum_by(prev_rows, "ARS")
    usd_this = sum(
        abs(_moze_pnum(r.get("Price")))
        for _, r in rows
        if (r.get("Type") or "").strip() == "Expense"
        and (r.get("Currency") or "").strip() in ("USD", "USDB")
    )
    income_ars = sum(
        abs(_moze_pnum(r.get("Price")))
        for _, r in rows
        if (r.get("Type") or "").strip() == "Income"
        and (r.get("Currency") or "").strip() == "ARS"
    )

    cat_tot: Counter = Counter()
    for _, r in rows:
        if (r.get("Currency") or "").strip() != "ARS":
            continue
        if (r.get("Type") or "").strip() != "Expense":
            continue
        cat = (r.get("Main Category") or "—").strip() or "—"
        cat_tot[cat] += abs(_moze_pnum(r.get("Price")))

    delta_pct = ((ars_this - ars_prev) / ars_prev * 100.0) if ars_prev else None
    month_label_es = f"{MOZE_MONTH_ES[month]} {year}"

    lines: list[str] = []
    lines.append("---")
    lines.append("type: finanzas")
    lines.append("source: MOZE")
    lines.append(f"month: {year:04d}-{month:02d}")
    lines.append(f"gasto_ars: {int(round(ars_this))}")
    if usd_this:
        lines.append(f"gasto_usd: {int(round(usd_this))}")
    if income_ars:
        lines.append(f"ingreso_ars: {int(round(income_ars))}")
    lines.append("tags: [finanzas, moze, gastos]")
    lines.append("ambient: skip")
    lines.append(f"source_file: {source_name}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Finanzas — {month_label_es}")
    lines.append("")

    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- Gasto ARS: **{_moze_fmt_ars(ars_this)}**")
    if ars_prev:
        delta_txt = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
        lines.append(f"- Mes anterior: {_moze_fmt_ars(ars_prev)} ({delta_txt})")
    if usd_this:
        lines.append(f"- Gasto USD: ${usd_this:,.2f}")
    if income_ars:
        lines.append(f"- Ingreso ARS: {_moze_fmt_ars(income_ars)}")
    lines.append(f"- Transacciones: {sum(1 for _, r in rows if (r.get('Type') or '') in ('Expense','Income'))}")
    lines.append("")

    if cat_tot:
        lines.append("## Top categorías (ARS)")
        lines.append("")
        for name, amt in cat_tot.most_common(10):
            share = (amt / ars_this * 100.0) if ars_this else 0.0
            lines.append(f"- {name}: {_moze_fmt_ars(amt)} ({share:.0f}%)")
        lines.append("")

    lines.append("## Transacciones")
    lines.append("")
    ordered = sorted(rows, key=lambda t: (t[0], t[1].get("Time") or ""))
    for d, r in ordered:
        typ = (r.get("Type") or "").strip()
        if typ not in ("Expense", "Income", "Receivable"):
            continue
        cur = (r.get("Currency") or "").strip()
        cat = (r.get("Main Category") or "").strip()
        sub = (r.get("Subcategory") or "").strip()
        name = (r.get("Name") or "").strip()
        store = (r.get("Store") or "").strip()
        amt = _moze_pnum(r.get("Price"))
        sign = "+" if typ == "Income" else "-"
        amt_str = _moze_fmt_ars(amt) if cur == "ARS" else f"${abs(amt):,.2f}"
        parts = [
            d.strftime("%Y-%m-%d"),
            typ,
            cat,
        ]
        if sub and sub != cat:
            parts.append(sub)
        if name:
            parts.append(name)
        if store:
            parts.append(store)
        parts.append(f"{sign}{amt_str} {cur}")
        lines.append(f"- {' · '.join(parts)}")
    lines.append("")

    return "\n".join(lines)


def _sync_moze_notes(vault_root: Path) -> dict:
    """Regenerate monthly MOZE notes under `{vault_root}/{MOZE_VAULT_SUBPATH}/`.

    Per-month note: `YYYY-MM.md`. Rewrite only if content changed (compare by
    file text) so the indexer's hash-based skip still works. Also writes an
    `_index.md` at the folder root with a summary table of recent months.

    Returns stats for logging. Silent-fail if no CSV is found.
    """
    from collections import defaultdict

    from rag.cross_source_etls import _etl_log_swallow

    parsed = _moze_parse_latest()
    if not parsed:
        return {"ok": False, "reason": "no_csv"}
    src, rows = parsed

    by_month: dict[tuple[int, int], list[tuple[datetime, dict]]] = defaultdict(list)
    for d, r in rows:
        by_month[(d.year, d.month)].append((d, r))

    target_dir = vault_root / MOZE_VAULT_SUBPATH
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {"ok": False, "reason": f"mkdir: {exc}"}

    months_sorted = sorted(by_month.keys())
    src_mtime = src.stat().st_mtime
    written = 0
    skipped = 0
    current_set = set()

    for i, (year, month) in enumerate(months_sorted):
        fname = f"{year:04d}-{month:02d}.md"
        current_set.add(fname)
        month_rows = by_month[(year, month)]
        prev_rows: list[tuple[datetime, dict]] = []
        if i > 0:
            prev_rows = by_month[months_sorted[i - 1]]
        body = _moze_render_month(year, month, month_rows, prev_rows, src.name, src_mtime)
        path = target_dir / fname
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        # Strip `generated_at`-style volatile fields — we don't emit any, so
        # a direct compare is enough; hash-based index skip follows naturally.
        if existing == body:
            skipped += 1
            continue
        path.write_text(body, encoding="utf-8")
        written += 1

    # Prune stale month files (e.g., a month that got fully deleted in MOZE).
    for p in target_dir.glob("*.md"):
        if p.name == "_index.md":
            continue
        if p.name not in current_set:
            try:
                p.unlink()
            except Exception as exc:
                _etl_log_swallow("moze_prune_stale_month", exc)

    # Roll-up index note — gives chat queries like "cuánto gasté este año"
    # a single surface to land on instead of 12 per-month notes.
    idx_lines = [
        "---",
        "type: finanzas",
        "source: MOZE",
        "tags: [finanzas, moze, indice]",
        "ambient: skip",
        f"source_file: {src.name}",
        "---",
        "",
        "# Finanzas — índice mensual (MOZE)",
        "",
        f"Fuente: `{src.name}` (export Money app).",
        "",
        "| Mes | Gasto ARS | Δ vs mes ant |",
        "|---|---:|---:|",
    ]
    prev_tot = 0.0
    for (year, month) in months_sorted:
        mrows = by_month[(year, month)]
        tot = sum(
            abs(_moze_pnum(r.get("Price")))
            for _, r in mrows
            if (r.get("Type") or "").strip() == "Expense"
            and (r.get("Currency") or "").strip() == "ARS"
        )
        delta = ""
        if prev_tot:
            delta_pct = (tot - prev_tot) / prev_tot * 100.0
            delta = f"{delta_pct:+.1f}%"
        idx_lines.append(
            f"| [[{year:04d}-{month:02d}]] | {_moze_fmt_ars(tot)} | {delta} |"
        )
        prev_tot = tot
    idx_body = "\n".join(idx_lines) + "\n"
    idx_path = target_dir / "_index.md"
    if not idx_path.is_file() or idx_path.read_text(encoding="utf-8") != idx_body:
        idx_path.write_text(idx_body, encoding="utf-8")
        written += 1
    else:
        skipped += 1

    return {
        "ok": True,
        "source": src.name,
        "months_total": len(months_sorted),
        "months_written": written,
        "months_skipped": skipped,
        "target": str(target_dir.relative_to(vault_root)),
    }
