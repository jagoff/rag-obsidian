"""``rag model`` — gestión de modelos por tier (chat / helper / embed / rerank / stt / vlm).

Hace que cambiar de modelo sea trivial:

    rag model list                  # snapshot de los 6 tiers
    rag model list chat             # detalle del tier chat (cached + known)
    rag model show                  # alias de `list`
    rag model set chat qwen2.5:14b  # hot-swap (in-process)
    rag model set helper qwen3:4b --persist   # +rewrite plists +kickstart
    rag model reset chat            # vuelve al default
    rag model current chat          # imprime solo el modelo activo

Source of truth: `rag/models.py` (env vars + defaults + reload hooks).
Hot-swap: muta `os.environ` + dispara los reload hooks registrados en
`rag/_model_hooks.py` (unload del modelo viejo del backend MLX, invalidate
caches in-process).

Persistencia (`--persist`): escribe la env var en los plists relevantes
(`com.fer.obsidian-rag-{web,serve,watch,...}.plist`) y dispara
`launchctl kickstart` así los daemons levantan con el nuevo modelo.
Sin `--persist`, el cambio dura sólo lo que dura la sesión actual del CLI.
"""

from __future__ import annotations

import click

from rag import console
from rag import models as _models


@click.group()
def model():
    """Cambiar el modelo de cada tier sin restart.

    Tiers:
      chat    — respuestas (`qwen2.5:7b` default)
      helper  — paths deterministas, HELPER_OPTIONS (`qwen2.5:3b` default)
      embed   — sentence embedding (`qwen3-embedding:0.6b`, dim 1024 obligatoria)
      rerank  — cross-encoder rerank (`bge-reranker-v2-m3` PT / Qwen3 MLX)
      stt     — Whisper STT (`small` default)
      vlm     — vision-language para OCR fallback (`granite-vision-3.2-2b-4bit`)

    Override por env var: RAG_<TIER>_MODEL. CLI escribe ahí + dispara hooks.
    """


@model.command("list")
@click.argument("tier", required=False, type=click.Choice(_models.TIERS))
def model_list(tier: str | None):
    """Listar modelos por tier.

    Sin TIER → snapshot de los 6 tiers (modelo activo + default si difiere).
    Con TIER → cached locales + known sin cachear (descargables vía huggingface-cli).
    """
    if tier is None:
        active = _models.all_active()
        console.print("[bold]Modelos activos:[/bold]")
        for t in _models.TIERS:
            current = active[t]
            default = _models.DEFAULTS[t]
            marker = "" if current == default else f" [dim](default: {default})[/dim]"
            console.print(f"  [cyan]{t:<7}[/cyan] {current}{marker}")
        console.print(
            "\n[dim]`rag model list <tier>` para ver alternativas. "
            "`rag model set <tier> <name>` para cambiar.[/dim]"
        )
        return

    active = _models.get(tier)
    catalog = _models.list_available(tier)
    console.print(f"[bold]Tier:[/bold] [cyan]{tier}[/cyan]")
    console.print(f"[bold]Activo:[/bold] {active}")
    console.print(f"[bold]Default:[/bold] {_models.DEFAULTS[tier]}")
    console.print(f"[bold]Env var:[/bold] {_models.ENV_VARS[tier]}")
    console.print()

    cached = catalog.get("cached", [])
    known = catalog.get("known", [])
    if cached:
        console.print("[green]Cacheados localmente[/green] (listos para usar):")
        for name in cached:
            marker = " [bold yellow]← activo[/bold yellow]" if name == active else ""
            console.print(f"  • {name}{marker}")
    if known:
        console.print("\n[dim]Conocidos (descargables on-demand):[/dim]")
        for name in known:
            console.print(f"  • [dim]{name}[/dim]")
    if not cached and not known:
        console.print("[yellow]Sin catálogo definido para este tier.[/yellow]")


@model.command("show")
@click.argument("tier", required=False, type=click.Choice(_models.TIERS))
@click.pass_context
def model_show(ctx: click.Context, tier: str | None):
    """Alias de `rag model list`."""
    ctx.invoke(model_list, tier=tier)


@model.command("current")
@click.argument("tier", type=click.Choice(_models.TIERS))
def model_current(tier: str):
    """Imprime sólo el modelo activo (script-friendly, sin formato)."""
    click.echo(_models.get(tier))


@model.command("set")
@click.argument("tier", type=click.Choice(_models.TIERS))
@click.argument("model_name")
@click.option("--persist/--ephemeral", default=False,
              help="--persist: rewrite plists + kickstart daemons. "
                   "--ephemeral (default): solo el proceso CLI actual.")
@click.option("--preload/--no-preload", default=False,
              help="Pre-warm el modelo nuevo después del swap (cold-load ~3-8s).")
@click.option("--unsafe", is_flag=True,
              help="Saltea validación (ej. embed con dim distinta — rompe el index).")
def model_set(tier: str, model_name: str, persist: bool, preload: bool, unsafe: bool):
    """Hot-swap del modelo de un tier."""
    try:
        old = _models.swap(tier, model_name, preload=preload, unsafe=unsafe)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise click.Abort() from exc
    console.print(
        f"[green]✓[/green] [cyan]{tier}[/cyan]: [dim]{old}[/dim] → "
        f"[bold]{model_name}[/bold] [dim](in-process)[/dim]"
    )
    if persist:
        try:
            from rag.cli._model_persist import persist_tier_to_plists
            touched = persist_tier_to_plists(tier, model_name)
        except Exception as exc:
            console.print(f"[yellow]Persist falló:[/yellow] {exc}")
            return
        if touched:
            console.print(
                f"[green]✓[/green] Persistido en {len(touched)} plist(s): "
                f"{', '.join(touched)}"
            )
        else:
            console.print("[dim]No se encontraron plists para actualizar.[/dim]")


@model.command("reset")
@click.argument("tier", required=False, type=click.Choice(_models.TIERS))
@click.option("--persist/--ephemeral", default=False,
              help="--persist: borra la env var de los plists también.")
def model_reset(tier: str | None, persist: bool):
    """Resetear un tier (o todos si no se pasa) al default. Dispara hooks."""
    targets = [tier] if tier else list(_models.TIERS)
    changed: list[tuple[str, str, str]] = []
    for t in targets:
        prev = _models.reset_env(t)
        new = _models.get(t)
        if prev != new:
            changed.append((t, prev, new))
    if not changed:
        console.print("[dim]Nada cambió — todos los tiers ya estaban en default.[/dim]")
        return
    for t, prev, new in changed:
        console.print(
            f"[green]✓[/green] [cyan]{t}[/cyan]: [dim]{prev}[/dim] → "
            f"[bold]{new}[/bold] [dim](default)[/dim]"
        )
    if persist:
        try:
            from rag.cli._model_persist import unset_tier_in_plists
            for t, _, _ in changed:
                touched = unset_tier_in_plists(t)
                if touched:
                    console.print(
                        f"  [dim]plists actualizados ({t}):[/dim] {', '.join(touched)}"
                    )
        except Exception as exc:
            console.print(f"[yellow]Persist falló:[/yellow] {exc}")
