"""`rag maintenance` — mantenimiento integral periódico.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 330).

Reindex incremental + cleanup de sesiones expiradas + prune de
context_cache + rotación de logs JSONL + rotación SQL telemetry +
VACUUM oportunístico + .bak cleanup + ignored notes prune + auto-index
state pruning + filing batches GC + tmp orphans + chat-uploads TTL +
URL orphans + feedback orphans + embedder health + dead notes report.

Heavy lifting en `run_maintenance()` (`rag/__init__.py`); este módulo
es el CLI shim + render del summary.

## Sub-modes

- `--validate-cutover`: read-only gate pre-T10 (compara COUNT(*) SQL vs
  líneas en `.bak.<ts>` JSONL). Si delta < 0 → exit 1.
- `--rollback-state-migration`: escape hatch que deshace T2 (renombra
  cada `.bak.<ts>` a su nombre + dropea las 21 rag_*/system_memory_metrics
  tables + VACUUM). Refuses si hay launchd services corriendo (`--force`
  bypassa).

## Lazy imports

Todos los helpers viven en `rag/__init__.py`: `_FILING_BATCH_TTL_DAYS`,
`_ROLLBACK_TABLES`, `_find_bak_files_for_rollback`, `_pgrep_obsidian_rag`,
`_render_cutover_validation`, `_rollback_state_migration`,
`_validate_cutover_state`, `console`, `run_maintenance`. Lazy adentro
del cuerpo del CLI command para evitar circular import.

## Re-export

`rag/__init__.py` registra el command via
`cli.add_command(maintenance_cmd, name="maintenance")`.
"""

from __future__ import annotations

import json

import click
from rich.rule import Rule

__all__ = ["maintenance_cmd"]


@click.command("maintenance")
@click.option("--dry-run", is_flag=True, help="Reportar sin modificar nada")
@click.option("--skip-reindex", is_flag=True, help="Saltear reindex incremental")
@click.option("--skip-logs", is_flag=True, help="Saltear rotación de logs")
@click.option("--verbose", "-v", is_flag=True, help="Mostrar detalles de cada paso")
@click.option("--json", "as_json", is_flag=True, help="Output JSON (para cron/scripts)")
@click.option("--rollback-state-migration", "rollback_state", is_flag=True,
               help="Escape hatch: restaurar .bak.<ts> originales + dropear rag_* tables (T7)")
@click.option("--validate-cutover", "validate_cutover", is_flag=True,
               help="Read-only: comparar COUNT(*) de cada rag_* table vs líneas en su .bak.<ts> (pre-T10 gate)")
@click.option("--force", is_flag=True,
               help="Bypass safety gates en --rollback-state-migration")
def maintenance_cmd(dry_run: bool, skip_reindex: bool, skip_logs: bool, verbose: bool,
                     as_json: bool, rollback_state: bool, validate_cutover: bool,
                     force: bool):
    """Mantenimiento integral: reindex, limpiar sesiones, podar caches, rotar logs, detectar dead notes.

    Seguro para correr periódicamente vía launchd/cron. Con --dry-run solo reporta.

    --rollback-state-migration deshace T2: renombra cada `.bak.<unix_ts>` a su
    nombre original + dropea las 21 rag_*/system_memory_metrics tables +
    VACUUM. Refuses si hay launchd services corriendo (usa --force para
    bypass) o si no hay .bak.<ts> dentro de la ventana de 30 días.

    --validate-cutover compara COUNT(*) de cada rag_* table con el line count
    del `.bak.<ts>` más reciente del JSONL que migró. Read-only, seguro en
    vivo. Propósito: gate pre-T10 — si algún delta es negativo (SQL < JSONL),
    significa que la migración perdió filas y NO se debe strippear el
    fallback. Útil en cron diario durante la ventana de observación.
    """
    from rag import (  # noqa: PLC0415
        _FILING_BATCH_TTL_DAYS,
        _ROLLBACK_TABLES,
        _find_bak_files_for_rollback,
        _pgrep_obsidian_rag,
        _render_cutover_validation,
        _rollback_state_migration,
        _validate_cutover_state,
        console,
        run_maintenance,
    )

    if validate_cutover:
        results = _validate_cutover_state()
        if as_json:
            click.echo(json.dumps(results, ensure_ascii=False, indent=2))
            return
        _render_cutover_validation(results)
        if any(r.get("status") == "fail" for r in results):
            raise SystemExit(1)
        return
    if rollback_state:
        if not as_json:
            mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]ROLLBACK[/red]"
            console.print(Rule(title=f"[bold cyan]State Migration Rollback[/bold cyan] {mode}",
                                style="cyan"))
            console.print()
        if dry_run:
            pids = _pgrep_obsidian_rag()
            bak_map = _find_bak_files_for_rollback()
            summary = {
                "would_refuse": bool(pids and not force) or (not bak_map and not force),
                "running_pids": pids,
                "bak_files_found": len(bak_map),
                "tables_would_drop": [t for t in _ROLLBACK_TABLES],
            }
            if as_json:
                click.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
                return
            if summary["would_refuse"]:
                if pids and not force:
                    console.print(f"  [red]Refused:[/red] {len(pids)} launchd service(s) running — "
                                   "stop services or pass --force.")
                if not bak_map and not force:
                    console.print("  [red]Refused:[/red] no .bak.<unix_ts> files within 30-day window — "
                                   "nothing to restore to (pass --force to DROP anyway).")
            console.print(f"  [bold].bak.<ts> files:[/bold] {len(bak_map)} would be restored")
            console.print(f"  [bold]Tables would drop:[/bold] {len(summary['tables_would_drop'])}")
            return
        result = _rollback_state_migration(force=force)
        if as_json:
            click.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return
        if not result["ok"]:
            console.print(f"  [red]Refused:[/red] {result.get('refused', 'unknown')}")
            raise click.exceptions.Exit(1)
        mb = result.get("bytes_reclaimed", 0) / (1024 * 1024)
        console.print(f"  [bold]Files restored:[/bold] {result['files_restored']}")
        console.print(f"  [bold]Tables dropped:[/bold] {len(result['tables_dropped'])}"
                       f" [dim]({', '.join(result['tables_dropped'][:4])}"
                       f"{', …' if len(result['tables_dropped']) > 4 else ''})[/dim]")
        console.print(f"  [bold]Disk reclaimed:[/bold] {mb:.1f} MB")
        return

    if not as_json:
        mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]"
        console.print(Rule(title=f"[bold cyan]Maintenance[/bold cyan] {mode}", style="cyan"))
        console.print()

    results = run_maintenance(
        dry_run=dry_run,
        skip_reindex=skip_reindex,
        skip_logs=skip_logs,
        verbose=verbose,
    )

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return

    idx = results.get("reindex")
    if idx:
        kind = idx.get("kind", "?")
        color = {"no_changes": "dim", "incremental": "green", "first_time": "yellow"}.get(kind, "white")
        console.print(f"  [bold]Reindex:[/bold] [{color}]{kind}[/{color}]"
                      f" — {idx.get('indexed', 0)} indexed, {idx.get('removed', 0)} orphans removed"
                      f" [dim]({idx.get('scanned', 0)} scanned, {idx.get('took_ms', 0)}ms)[/dim]")
    elif "reindex_error" in results:
        console.print(f"  [bold]Reindex:[/bold] [red]error: {results['reindex_error']}[/red]")
    else:
        console.print("  [bold]Reindex:[/bold] [dim]skipped[/dim]")

    n_sess = results.get("sessions_removed", 0)
    console.print(f"  [bold]Sessions:[/bold] {n_sess} expired removed" if n_sess else
                  "  [bold]Sessions:[/bold] [dim]none expired[/dim]")

    orphans = results.get("orphan_collections", [])
    if orphans:
        action = "would remove" if dry_run else "removed"
        console.print(f"  [bold]Orphan collections:[/bold] [yellow]{action} {len(orphans)}[/yellow]: {', '.join(orphans)}")
    else:
        console.print("  [bold]Orphan collections:[/bold] [dim]none[/dim]")

    seg = results.get("orphan_segments") or {}
    n_seg = seg.get("count", 0)
    if n_seg:
        verb = "would free" if dry_run else "freed"
        mb = seg.get("bytes_freed", 0) / (1024 * 1024)
        console.print(f"  [bold]Orphan segments:[/bold] [yellow]{n_seg} dirs, {verb} {mb:.0f} MB[/yellow]")
    else:
        console.print("  [bold]Orphan segments:[/bold] [dim]none[/dim]")

    wal = results.get("wal_checkpoint") or {}
    if wal.get("ok"):
        b = wal.get("before_bytes", 0) / (1024 * 1024)
        a = wal.get("after_bytes", 0) / (1024 * 1024)
        if dry_run:
            console.print(f"  [bold]WAL:[/bold] [dim]{b:.1f} MB (dry-run)[/dim]")
        else:
            console.print(f"  [bold]WAL:[/bold] checkpoint {b:.1f} → {a:.1f} MB")
    elif "wal_checkpoint_error" in results or wal.get("reason"):
        reason = results.get("wal_checkpoint_error") or wal.get("reason")
        console.print(f"  [bold]WAL:[/bold] [dim red]skipped: {reason}[/dim red]")

    pruned = results.get("context_cache_pruned", 0)
    if pruned:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]Context cache:[/bold] [yellow]{pruned} stale entries {verb}[/yellow]")
    else:
        console.print("  [bold]Context cache:[/bold] [dim]clean[/dim]")

    fg = results.get("feedback_golden")
    if fg:
        console.print(f"  [bold]Feedback golden:[/bold] rebuilt ({fg['positives']}+ / {fg['negatives']}-)")
    elif dry_run:
        console.print("  [bold]Feedback golden:[/bold] [dim]skipped (dry-run)[/dim]")

    rotated = results.get("log_rotation", {})
    if rotated:
        for label, detail in rotated.items():
            console.print(f"  [bold]Log rotation:[/bold] {label} — {detail}")
    else:
        console.print("  [bold]Log rotation:[/bold] [dim]all under threshold[/dim]")

    sql_rot = results.get("sql_rotation") or {}
    if sql_rot:
        rows = sql_rot.get("rows_deleted", {}) or {}
        total = sum(rows.values())
        verb = "would delete" if dry_run else "deleted"
        if total:
            console.print(f"  [bold]SQL rotation:[/bold] [yellow]{verb} {total} rows[/yellow] "
                           f"across {sum(1 for v in rows.values() if v)} tables")
            if verbose:
                for t, n in sorted(rows.items(), key=lambda x: -x[1]):
                    if n:
                        console.print(f"    [dim]{t}: {n}[/dim]")
        else:
            console.print("  [bold]SQL rotation:[/bold] [dim]no rows past retention[/dim]")
        wal = sql_rot.get("wal_checkpoint") or {}
        if wal.get("ok"):
            b = wal.get("before_bytes", 0) / (1024 * 1024)
            a = wal.get("after_bytes", 0) / (1024 * 1024)
            if dry_run:
                console.print(f"  [bold]SQL WAL:[/bold] [dim]{b:.1f} MB (dry-run)[/dim]")
            else:
                console.print(f"  [bold]SQL WAL:[/bold] checkpoint {b:.1f} → {a:.1f} MB")
        if sql_rot.get("vacuum_ran"):
            bv = (sql_rot.get("vacuum_before_bytes") or 0) / (1024 * 1024)
            av = (sql_rot.get("vacuum_after_bytes") or 0) / (1024 * 1024)
            reason = sql_rot.get("vacuum_reason") or ""
            tag = f" [{reason}]" if reason else ""
            console.print(f"  [bold]VACUUM telemetry:[/bold] ran {bv:.1f} → {av:.1f} MB{tag}")
        elif sql_rot.get("vacuum_would_run"):
            console.print("  [bold]VACUUM telemetry:[/bold] [yellow]would run[/yellow]")
        else:
            pct = sql_rot.get("freelist_pct", 0) * 100
            console.print(
                f"  [bold]VACUUM telemetry:[/bold] [dim]skipped (frag {pct:.1f}%)[/dim]"
            )
    elif "sql_rotation_error" in results:
        console.print(f"  [bold]SQL rotation:[/bold] [red]error: {results['sql_rotation_error']}[/red]")

    rv = results.get("ragvec_vacuum") or {}
    if rv:
        if rv.get("ran"):
            bv = (rv.get("before_bytes") or 0) / (1024 * 1024)
            av = (rv.get("after_bytes") or 0) / (1024 * 1024)
            console.print(
                f"  [bold]VACUUM ragvec:[/bold] ran {bv:.1f} → {av:.1f} MB [frag]"
            )
        elif rv.get("would_run"):
            console.print("  [bold]VACUUM ragvec:[/bold] [yellow]would run[/yellow]")
        elif rv.get("error"):
            console.print(f"  [bold]VACUUM ragvec:[/bold] [red]{rv['error']}[/red]")
        else:
            pct = rv.get("freelist_pct", 0) * 100
            console.print(
                f"  [bold]VACUUM ragvec:[/bold] [dim]skipped (frag {pct:.1f}%)[/dim]"
            )

    bak = results.get("bak_cleanup") or {}
    if bak:
        n = bak.get("deleted", 0)
        mb = bak.get("bytes_freed", 0) / (1024 * 1024)
        if n:
            verb = "would purge" if dry_run else "purged"
            console.print(f"  [bold].bak cleanup:[/bold] [yellow]{verb} {n} files ({mb:.1f} MB)[/yellow]")
        else:
            console.print("  [bold].bak cleanup:[/bold] [dim]none >30d[/dim]")

    ign = results.get("ignored_pruned", 0)
    if ign:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]Ignored notes:[/bold] [yellow]{ign} stale entries {verb}[/yellow]")
    else:
        console.print("  [bold]Ignored notes:[/bold] [dim]clean[/dim]")

    idx_st = results.get("index_state_pruned", 0)
    if idx_st:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]Index state:[/bold] [yellow]{idx_st} orphan vault keys {verb}[/yellow]")
    else:
        console.print("  [bold]Index state:[/bold] [dim]clean[/dim]")

    batches = results.get("batches_pruned", 0)
    if batches:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]Filing batches:[/bold] [yellow]{batches} old batches {verb}[/yellow] [dim](>{_FILING_BATCH_TTL_DAYS}d)[/dim]")
    else:
        console.print("  [bold]Filing batches:[/bold] [dim]none expired[/dim]")

    tmps = results.get("tmp_cleaned", 0)
    if tmps:
        verb = "would remove" if dry_run else "removed"
        console.print(f"  [bold]Tmp files:[/bold] [yellow]{tmps} orphans {verb}[/yellow]")
    else:
        console.print("  [bold]Tmp files:[/bold] [dim]none[/dim]")

    cu = results.get("chat_uploads_cleaned") or {}
    if cu:
        n = cu.get("deleted", 0)
        mb = cu.get("bytes_freed", 0) / (1024 * 1024)
        errs = cu.get("errors", []) or []
        if n:
            verb = "would purge" if dry_run else "purged"
            console.print(
                f"  [bold]Chat uploads:[/bold] [yellow]{verb} {n} files ({mb:.1f} MB)[/yellow]"
            )
        else:
            console.print("  [bold]Chat uploads:[/bold] [dim]none past TTL[/dim]")
        if errs and verbose:
            for err in errs:
                console.print(f"    [dim red]{err}[/dim red]")
    elif "chat_uploads_cleaned_error" in results:
        console.print(
            f"  [bold]Chat uploads:[/bold] [red]error: {results['chat_uploads_cleaned_error']}[/red]"
        )

    url_orph = results.get("url_orphans", 0)
    if url_orph:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]URL orphans:[/bold] [yellow]{url_orph} rows {verb}[/yellow]")
    else:
        console.print("  [bold]URL orphans:[/bold] [dim]clean[/dim]")

    fb_orph = results.get("feedback_orphans", 0)
    if fb_orph:
        verb = "would prune" if dry_run else "pruned"
        console.print(f"  [bold]Feedback orphans:[/bold] [yellow]{fb_orph} stale entries {verb}[/yellow]")
    else:
        console.print("  [bold]Feedback orphans:[/bold] [dim]clean[/dim]")

    oll = results.get("embedder") or results.get("ollama", {})
    if "error" in oll:
        console.print(f"  [bold]Embedder:[/bold] [red]unreachable: {oll['error']}[/red]")
    else:
        missing = [m for m, s in oll.items() if s == "missing"]
        if missing:
            console.print(f"  [bold]Embedder:[/bold] [red]missing: {', '.join(missing)}[/red]")
        else:
            console.print(f"  [bold]Embedder:[/bold] [green]{len(oll)} models ok[/green]")

    dead_n = results.get("dead_notes", 0)
    if dead_n:
        console.print(f"  [bold]Dead notes:[/bold] [yellow]{dead_n} candidates[/yellow] [dim](run `rag archive` to act)[/dim]")
    else:
        console.print("  [bold]Dead notes:[/bold] [dim]none[/dim]")

    console.print()
    console.print(f"  [dim]Done in {results.get('took_ms', 0)}ms[/dim]")
