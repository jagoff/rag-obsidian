"""`rag wake-up` — orquestador "todo en uno" para launchd a las 04:00.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 220).

Cuando el user se despierta, todo está fresco — ETLs corridos, vault
reindexado, caches limpios, radares actualizados, brief matutino
pre-renderizado en `00-Inbox/reviews/`, y el chat model cargado
en RAM con `keep_alive=-1` para que el primer `rag chat` responda sin
cold-start.

Cada paso es independiente: si `rag maintenance` falla, `rag morning`
igual corre. Al final imprime un resumen + exit code non-zero si algún
paso falló (para que launchd lo registre en `last exit code` y aparezca
rojo en la `/status` page).

No reemplaza los plists individuales (morning 07:00, maintenance 03:30,
patterns domingo 20:00, emergent viernes 10:00) — solo los amortigua.
Si la Mac estaba en sleep a alguna de esas horas y launchd no pudo
disparar el job, el wake-up lo re-ejecuta a las 04:00.

## Lazy imports

Todos los step commands (`index`, `bookmarks_sync`, `wa_tasks_cmd`,
`maintenance`, `feedback_patterns`, `emergent`, `morning`,
`resolve_chat_model`, `_mlx_chat`, `CHAT_OPTIONS`, `console`) viven en
`rag/__init__.py` o sub-módulos. Lazy adentro del cuerpo para evitar
circular import.

## Re-export

`rag/__init__.py` registra el command via
`cli.add_command(wake_up_cmd, name="wake-up")`.
"""

from __future__ import annotations

import time

import click
from rich.rule import Rule

__all__ = ["wake_up_cmd"]


@click.command(name="wake-up")
@click.option("--dry-run", is_flag=True,
              help="Reportar los pasos sin ejecutar nada")
@click.option("--skip-index", is_flag=True,
              help="Saltear `rag index` (ETLs + reindex)")
@click.option("--skip-bookmarks", is_flag=True,
              help="Saltear `rag bookmarks sync` (Chrome bookmarks). "
                   "Nota: `rag index` ya los sincroniza automáticamente "
                   "desde 2026-04-25 — este paso es redundante por default "
                   "pero queda explícito para que se vea en el log y para "
                   "cubrir el caso `--skip-index`.")
@click.option("--skip-wa-tasks", is_flag=True,
              help="Saltear `rag wa-tasks` (action items de WhatsApp al Inbox)")
@click.option("--skip-maintenance", is_flag=True,
              help="Saltear `rag maintenance`")
@click.option("--skip-radars", is_flag=True,
              help="Saltear `rag patterns` + `rag emergent`")
@click.option("--skip-brief", is_flag=True,
              help="Saltear pre-render de `rag morning`")
@click.option("--skip-warmup", is_flag=True,
              help="Saltear el LLM warmup")
@click.pass_context
def wake_up_cmd(ctx, dry_run: bool, skip_index: bool, skip_bookmarks: bool,
                skip_wa_tasks: bool, skip_maintenance: bool, skip_radars: bool,
                skip_brief: bool, skip_warmup: bool):
    """Wake-up pack: traé todo fresco antes de despertarte.

    Orquesta en este orden (cada paso es independiente — si uno falla,
    los demás siguen):

    \b
      1. `rag index`           — ETLs de todas las fuentes + reindex del vault.
      2. `rag bookmarks sync`  — Chrome bookmarks → sub-índice URLs (rag links).
      3. `rag wa-tasks`        — extractor de action items de WhatsApp al Inbox.
      4. `rag maintenance`     — WAL checkpoint, rotación de logs, cleanup.
      5. `rag patterns`        — radar de feedback dominante.
      6. `rag emergent`        — radar de temas emergentes en queries.
      7. `rag morning`         — brief matutino pre-renderizado a 00-Inbox/reviews/.
      8. LLM warmup            — carga el chat model con keep_alive=-1.

    Nota sobre bookmarks: `rag index` ya los sincroniza automáticamente
    como parte del pre-sync cross-source desde 2026-04-25. El paso 2
    (`rag bookmarks sync`) está acá por dos razones: (a) queda visible
    en el log del wake-up — sino se "esconde" dentro de los logs de
    `rag index`; (b) cubre el caso `--skip-index`, donde igual querés
    refrescar bookmarks sin reindexar todo el vault. El costo es ~100ms
    de duplicación cuando ambos corren (idempotente, segunda corrida
    es no-op por hash-skip).

    Pensado para launchd a las 04:00 — `rag setup` instala el plist.
    Exit code != 0 si algún paso falla (launchd lo marca rojo).

    Flags de skip útiles para debug o cuando un subsistema específico
    está roto y querés correr el resto igual.
    """
    from rag import (  # noqa: PLC0415
        CHAT_OPTIONS,
        _mlx_chat,
        bookmarks_sync,
        console,
        emergent,
        feedback_patterns,
        index,
        maintenance,
        morning,
        resolve_chat_model,
    )
    from rag.wa_tasks import wa_tasks as wa_tasks_cmd  # noqa: PLC0415

    steps: list[tuple[str, object, dict]] = []
    if not skip_index:
        # Click param names — `--full` binds `full_flag`, `--reset` (alias
        # legacy) binds `reset_legacy`. Pasar `reset=False` revienta con
        # `TypeError: index() got an unexpected keyword argument 'reset'`
        # — los 7 cross-source ingesters fallan silent y nada se actualiza
        # post-wake. Mismo patrón que cli/setup.py::catch_up_index().
        steps.append(("rag index", index, dict(
            full_flag=False, reset_legacy=False, no_contradict=False,
            source_opt=None, since_opt=None, dry_run=False, max_chats=None,
            vault_scope=None, contextual=False, fast=False,
        )))
    if not skip_bookmarks:
        steps.append(("rag bookmarks sync", bookmarks_sync, dict(profile=None)))
    if not skip_wa_tasks:
        steps.append(("rag wa-tasks", wa_tasks_cmd, dict(
            dry_run=False, hours=None, force=False,
        )))
    if not skip_maintenance:
        steps.append(("rag maintenance", maintenance, dict(
            dry_run=False, skip_reindex=True, skip_logs=False, verbose=False,
            as_json=False, rollback_state=False, validate_cutover=False,
            force=False,
        )))
    if not skip_radars:
        steps.append(("rag feedback-patterns", feedback_patterns, dict(
            last=500, min_share=0.30, dry_run=False, push=False,
        )))
        steps.append(("rag emergent", emergent, dict(
            days=7, min_size=5, threshold=0.35, dry_run=False, push=False,
        )))
    if not skip_brief:
        steps.append(("rag morning", morning, dict(
            dry_run=False, date_opt=None, lookback_hours=36,
        )))

    total_steps = len(steps) + (0 if skip_warmup else 1)
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    total_start = time.perf_counter()
    console.print()
    console.print(Rule(title=f"[bold cyan]wake-up[/bold cyan] · "
                       f"{total_steps} pasos",
                       style="cyan"))

    for i, (label, fn, kwargs) in enumerate(steps, 1):
        console.print()
        console.print(f"[bold cyan]▶ [{i}/{total_steps}][/bold cyan] {label}")
        if dry_run:
            console.print("  [dim]dry-run: skippeado[/dim]")
            continue
        step_start = time.perf_counter()
        try:
            ctx.invoke(fn, **kwargs)
            ms = int((time.perf_counter() - step_start) * 1000)
            succeeded.append(f"{label} ({ms}ms)")
        except SystemExit as e:
            ms = int((time.perf_counter() - step_start) * 1000)
            code = e.code if e.code is not None else 0
            if code == 0:
                succeeded.append(f"{label} ({ms}ms)")
            else:
                failed.append((label, f"exit {code} ({ms}ms)"))
        except Exception as e:
            ms = int((time.perf_counter() - step_start) * 1000)
            failed.append((label, f"{type(e).__name__}: {e} ({ms}ms)"))
            console.print(f"  [red]✗[/red] {type(e).__name__}: {e}")

    if not skip_warmup:
        idx = len(steps) + 1
        console.print()
        console.print(f"[bold cyan]▶ [{idx}/{total_steps}][/bold cyan] LLM warmup")
        if dry_run:
            console.print("  [dim]dry-run: skippeado[/dim]")
        else:
            step_start = time.perf_counter()
            try:
                model = resolve_chat_model()
                _mlx_chat(
                    model=model,
                    messages=[{"role": "user", "content": "."}],
                    options=CHAT_OPTIONS,
                    keep_alive=-1,
                )
                ms = int((time.perf_counter() - step_start) * 1000)
                succeeded.append(f"LLM warmup · {model} ({ms}ms)")
                console.print(f"  [green]✓[/green] {model} caliente "
                              f"(keep_alive=-1, {ms}ms)")
            except Exception as e:
                ms = int((time.perf_counter() - step_start) * 1000)
                failed.append(("LLM warmup",
                               f"{type(e).__name__}: {e} ({ms}ms)"))
                console.print(f"  [red]✗[/red] {type(e).__name__}: {e}")

    total_ms = int((time.perf_counter() - total_start) * 1000)
    console.print()
    style = "green" if not failed else "yellow"
    verdict = "OK" if not failed else f"{len(failed)} falló(s)"
    console.print(Rule(
        title=f"[bold]wake-up done[/bold] · {verdict} · "
              f"{len(succeeded)} ok · {total_ms}ms",
        style=style,
    ))
    if failed:
        console.print()
        console.print("[yellow]Fallaron:[/yellow]")
        for label, reason in failed:
            console.print(f"  · [bold]{label}[/bold]: {reason}")
        raise SystemExit(1)
