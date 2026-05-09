"""`rag dead` — listar notas candidatas a archivar.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 140).

Read-only: criterio AND (0 outlinks + 0 backlinks + no recuperada en
N días + mtime > N días + fuera de Inbox/Archive/Reviews). No borra
nada — `rag archive` es el que aplica los moves.

## Lazy imports

`find_dead_notes`, `get_db`, `VAULT_PATH`, `LOG_PATH`,
`log_query_event`, `_round_timing_ms`, `console` viven en
`rag/__init__.py`. Lazy adentro del cuerpo del comando para evitar
import circular cuando este módulo se carga en Click registry.

## Re-export

`rag/__init__.py` registra el command via
`cli.add_command(dead_cmd, name="dead")`.
"""

from __future__ import annotations

import time

import click
from rich.rule import Rule
from rich.table import Table

__all__ = ["dead_cmd"]


@click.command(name="dead")
@click.option("--min-age-days", default=365, show_default=True,
              help="Edad mínima por mtime")
@click.option("--query-window-days", default=180, show_default=True,
              help="Ventana para considerar que una nota 'fue usada' en queries")
@click.option("--limit", default=50, show_default=True,
              help="Cap del listado")
@click.option("--folder", default=None, help="Acotar a una subcarpeta")
@click.option("--plain", is_flag=True, help="Salida plana (path por línea)")
def dead_cmd(min_age_days: int, query_window_days: int, limit: int,
             folder: str | None, plain: bool):
    """Listar notas candidatas a archivar (dead code del vault).

    Criterio AND: 0 outlinks + 0 backlinks + no recuperada en N días +
    mtime > N días + fuera de Inbox/Archive/Reviews. No borra nada.
    """
    from rag import (  # noqa: PLC0415
        LOG_PATH,
        VAULT_PATH,
        _round_timing_ms,
        console,
        find_dead_notes,
        get_db,
        log_query_event,
    )

    col = get_db()
    t0 = time.perf_counter()
    items = find_dead_notes(
        col, VAULT_PATH, LOG_PATH,
        min_age_days=min_age_days,
        query_window_days=query_window_days,
    )
    if folder:
        prefix = folder.rstrip("/") + "/"
        items = [
            it for it in items
            if it["path"] == folder or it["path"].startswith(prefix)
        ]
    items = items[:limit]

    log_query_event({
        "cmd": "dead", "min_age_days": min_age_days,
        "query_window_days": query_window_days,
        "folder": folder, "n_candidates": len(items),
        "timing": _round_timing_ms({"total_ms": (time.perf_counter() - t0) * 1000}),
    })

    if not items:
        msg = "Sin candidatos a dead notes."
        click.echo(msg) if plain else console.print(f"[green]{msg}[/green]")
        return
    if plain:
        for it in items:
            click.echo(f"{it['age_days']}d\t{it['path']}")
        return
    console.print()
    console.print(Rule(
        title=f"[bold yellow]🪦 {len(items)} nota(s) candidatas a archivar[/bold yellow]",
        style="yellow",
    ))
    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    tbl.add_column("edad", style="yellow", justify="right")
    tbl.add_column("última modif", style="dim")
    tbl.add_column("path", style="cyan")
    for it in items:
        tbl.add_row(f"{it['age_days']}d", it["mtime"][:10], it["path"])
    console.print(tbl)
    console.print(
        "\n[dim]Criterio AND: 0 outlinks · 0 backlinks · "
        f"no recuperada en {query_window_days}d · mtime > {min_age_days}d.[/dim]"
    )
