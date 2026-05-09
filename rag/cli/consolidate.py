"""`rag consolidate` — episodic memory Phase 2.

Phase 3 cont de modularización (audit perf 2026-05-08).

Promotes recurring conversation clusters from
`99-obsidian/99-AI/conversations/` into PARA and archives the
originals. Heavy lifting en `scripts/consolidate_conversations.py`
— este módulo es el CLI shim.

Corre semanalmente via launchd (Lunes 06:00). Default dry-run; pasá
`--apply` para escribir al vault.

Re-export shim en `rag/__init__.py` registra el command con
`cli.add_command(consolidate)`.
"""

from __future__ import annotations

import json

import click

__all__ = ["consolidate"]


@click.command()
@click.option("--window-days", default=14, show_default=True,
              help="Conversaciones a considerar (por mtime)")
@click.option("--threshold", default=0.75, show_default=True,
              help="Cosine mínimo para agrupar (0.0–1.0)")
@click.option("--min-cluster", default=3, show_default=True,
              help="Tamaño mínimo de cluster para promover")
@click.option("--apply", is_flag=True,
              help="Escribir cambios al vault (default: dry-run)")
@click.option("--json", "as_json", is_flag=True,
              help="Emitir summary como JSON (para servicios / scripts)")
def consolidate(window_days: int, threshold: float, min_cluster: int,
                apply: bool, as_json: bool):
    """Agrupar conversaciones relacionadas y promover a PARA (Phase 2).

    Por default es dry-run — pasá --apply para escribir al vault.

    Barre `99-obsidian/99-AI/conversations/` buscando clusters
    semánticamente similares (embedding cosine ≥ --threshold, tamaño ≥
    --min-cluster) en la ventana de --window-days. Cada cluster se
    sintetiza en una sola nota consolidada (qwen2.5:7b default) y se
    promueve a `01-Projects/` o `03-Resources/` según un clasificador
    de intención. Los originales se archivan en
    `04-Archive/conversations/YYYY-MM/` (excluido del índice).

    Corre semanalmente via launchd (Lunes 06:00) — instalable con
    `rag setup`.
    """
    from rag import console  # noqa: PLC0415
    from scripts.consolidate_conversations import run as _run  # noqa: PLC0415

    dry_run = not apply  # invertimos: --apply activa, default es dry-run
    summary = _run(
        window_days=window_days,
        threshold=threshold,
        min_cluster=min_cluster,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    prefix = "[dry-run] " if dry_run else ""
    console.print(
        f"{prefix}{summary['n_conversations']} conversaciones en ventana · "
        f"{summary['n_clusters']} clusters · "
        f"[green]{summary['n_promoted']}[/green] promovidos · "
        f"{summary['n_archived']} archivados · "
        f"[dim]{summary['duration_s']}s[/dim]"
    )
    for c in summary["clusters"]:
        size = c.get("size", 0)
        target = c.get("target", "?")
        title = c.get("title", "(sin título)")
        if "error" in c:
            console.print(
                f"  [red]✗[/red] [{size}] {target}: {title} — "
                f"[dim]{c['error']}[/dim]"
            )
        elif dry_run:
            console.print(f"  [yellow]→[/yellow] [{size}] {target}: {title}")
        else:
            console.print(f"  [green]✓[/green] [{size}] {target}: {title}")
