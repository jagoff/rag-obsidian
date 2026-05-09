"""`rag hygiene` — vault hygiene scan.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 125).

Read-only scan: notas sin tags, sin outlinks, huérfanas (sin
backlinks), vacías (body < threshold), stale (mtime > N días), con
TODO/FIXME/WIP markers. NO escribe ni borra. Complementa
`rag dupes` / `rag dead` / `rag wikilinks suggest`.

## API

- `_hygiene_scan(col, *, empty_threshold, stale_days, sample_size)` →
  dict con `total_notes`, `counts`, `samples`.
- `hygiene_cli` (Click command) — registrado al final de
  `rag/__init__.py` via `cli.add_command(hygiene_cli)`.

## Lazy imports

`_load_corpus`, `console`, `get_db` viven en `rag/__init__.py`. Lazy
adentro del cuerpo para evitar circular imports.

## Re-export

`rag/__init__.py` hace `from rag.cli.hygiene import *  # noqa: F401, F403`.
Preserva back-compat con tests que llaman `rag._hygiene_scan(...)`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from rag import SqliteVecCollection

__all__ = ["_hygiene_scan", "hygiene_cli"]


def _hygiene_scan(
    col: "SqliteVecCollection", *,
    empty_threshold: int = 50,
    stale_days: int = 180,
    sample_size: int = 5,
) -> dict:
    """Scan corpus + adjacency for hygiene issues. Returns structured dict
    with counts + sample paths for each category.

    Empty: len(document) < empty_threshold chars.
    Stale: file mtime > stale_days ago.
    Huérfanas: path not in any note's outlinks (= sin backlinks).
    """
    from datetime import datetime as _dt  # noqa: PLC0415

    from rag import _load_corpus  # noqa: PLC0415

    result = {
        "total_notes": 0,
        "sin_tags": [],
        "sin_outlinks": [],
        "huerfanas": [],
        "vacias": [],
        "stale": [],
        "con_wip": [],
    }
    try:
        corpus = _load_corpus(col)
    except Exception as exc:
        result["error"] = repr(exc)
        return result
    metas = corpus.get("metas") or []
    docs = corpus.get("docs") or []
    by_file: dict[str, dict] = {}
    for m, d in zip(metas, docs):
        if not isinstance(m, dict):
            continue
        path = m.get("file", "")
        if not path or "://" in path:
            continue
        entry = by_file.setdefault(path, {
            "doc_chars": 0,
            "tags": m.get("tags", ""),
            "outlinks": m.get("outlinks", ""),
            "modified": m.get("modified", ""),
            "has_wip": False,
        })
        entry["doc_chars"] += len(d or "")
        if d and re.search(r"\b(TODO|FIXME|WIP|XXX)\b", d[:2000]):
            entry["has_wip"] = True
    result["total_notes"] = len(by_file)

    linked_to: set[str] = set()
    for path, entry in by_file.items():
        outlinks_raw = entry.get("outlinks") or ""
        if isinstance(outlinks_raw, str):
            parts = [o.strip() for o in outlinks_raw.split(",") if o.strip()]
        elif isinstance(outlinks_raw, list):
            parts = [str(o).strip() for o in outlinks_raw if o]
        else:
            parts = []
        linked_to.update(parts)

    now = _dt.now()
    stale_cutoff_days = float(stale_days)

    for path, entry in by_file.items():
        tags = entry.get("tags", "") or ""
        if not tags.strip():
            result["sin_tags"].append(path)
        outlinks_raw = entry.get("outlinks", "") or ""
        if not str(outlinks_raw).strip():
            result["sin_outlinks"].append(path)
        note_title = Path(path).stem
        if note_title not in linked_to and path not in linked_to:
            result["huerfanas"].append(path)
        if entry.get("doc_chars", 0) < empty_threshold:
            result["vacias"].append(path)
        mod = entry.get("modified", "")
        if mod:
            try:
                mod_dt = _dt.fromisoformat(str(mod).replace("Z", "+00:00"))
                if mod_dt.tzinfo is not None:
                    mod_dt = mod_dt.replace(tzinfo=None)
                age_days = (now - mod_dt).total_seconds() / 86400.0
                if age_days > stale_cutoff_days:
                    result["stale"].append(path)
            except Exception:
                pass
        if entry.get("has_wip"):
            result["con_wip"].append(path)

    counts = {}
    samples = {}
    for k in ("sin_tags", "sin_outlinks", "huerfanas", "vacias",
              "stale", "con_wip"):
        paths = sorted(result[k])
        counts[k] = len(paths)
        samples[k] = paths[:sample_size]

    return {
        "total_notes": result["total_notes"],
        "counts": counts,
        "samples": samples,
    }


@click.command("hygiene")
@click.option("--empty-threshold", default=50, show_default=True,
              help="Chars por debajo de los cuales una nota se considera vacía.")
@click.option("--stale-days", default=180, show_default=True,
              help="Días sin modificar tras los cuales una nota es stale.")
@click.option("--sample", default=5, show_default=True,
              help="Cuántas paths de sample mostrar por categoría.")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON machine-readable.")
def hygiene_cli(
    empty_threshold: int, stale_days: int, sample: int, as_json: bool,
):
    """Vault hygiene scan — notas sin tags, sin links, huérfanas, vacías.

    Read-only: NO borra ni escribe nada. Solo reporta.

    Categorías:
      • Sin tags       — falta `tags:` en frontmatter.
      • Sin outlinks   — no enlazan a ninguna otra nota ([[...]]).
      • Huérfanas      — nadie las linkea (sin backlinks).
      • Vacías         — body < --empty-threshold chars.
      • Stale          — mtime > --stale-days atrás.
      • Con WIP        — contienen TODO/FIXME/WIP/XXX.

    Para limpiar: `rag dupes` (near-duplicates), `rag dead` (archivado),
    `rag wikilinks suggest` (densificar grafo).
    """
    from rag import console, get_db  # noqa: PLC0415

    try:
        col = get_db()
    except Exception as exc:
        console.print(f"[red]Error abriendo DB: {exc!r}[/red]")
        return
    report = _hygiene_scan(
        col, empty_threshold=empty_threshold,
        stale_days=stale_days, sample_size=sample,
    )
    if as_json:
        click.echo(json.dumps(report))
        return
    console.print()
    console.print(f"[bold]Vault hygiene[/bold]  [dim]· "
                  f"{report['total_notes']:,} notas indexadas[/dim]")
    if "error" in report:
        console.print(f"[red]Error: {report['error']}[/red]")
        return
    console.print()
    categories = [
        ("sin_tags", "Sin tags",
         "agregar frontmatter `tags: [...]` a cada nota"),
        ("sin_outlinks", "Sin outlinks",
         "usar `rag wikilinks suggest` para densificar"),
        ("huerfanas", "Huérfanas (sin backlinks)",
         "revisar relevancia — candidatas a archivar"),
        ("vacias", "Vacías (<threshold chars)",
         "rellenar contenido o archivar"),
        ("stale", f"Stale (>{stale_days}d sin tocar)",
         "revisar si siguen vigentes — `rag dead`"),
        ("con_wip", "Con TODO/WIP markers",
         "tareas pendientes a cerrar"),
    ]
    for key, label, hint in categories:
        n = report["counts"][key]
        color = "green" if n == 0 else ("yellow" if n < 20 else "red")
        console.print(f"  [{color}]{n:4d}[/{color}]  {label}")
        if n > 0:
            for p in report["samples"][key]:
                console.print(f"          [dim]· {p}[/dim]")
            if n > len(report["samples"][key]):
                console.print(f"          [dim]… y {n - len(report['samples'][key])} más[/dim]")
            console.print(f"          [dim italic]→ {hint}[/dim italic]")
    console.print()
