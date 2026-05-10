"""``rag wa-contacts`` — gestionar contact notes desde stats del bridge WA.

Disparador (2026-05-10): el user pidió que el sistema arme automáticamente
las contact notes en `99-Contacts/`, prellenando lo derivable y dejando
campos `<COMPLETAR>` para que el user los llene a su ritmo. Sin esto, el
user creaba las notas a mano para cada contacto nuevo y eso se traducía
en kinship=unknown y registros genéricos en el draft.

Subcomandos:
  - ``backfill`` — escanea el bridge, crea notas faltantes (default
    ``--dry-run`` para que el user inspeccione antes de aplicar).
  - ``list``    — lista las notas existentes con tier + stats.
  - ``stats``   — resumen del scan (cuántas notas faltarían, distribución
    por tier).

El módulo es thin wrapper alrededor de
``rag.integrations.whatsapp.contact_backfill`` — la lógica vive ahí, este
solo expone la CLI Click.
"""

from __future__ import annotations

from pathlib import Path

import click

from rag import _resolve_vault_path, console
from rag.integrations.whatsapp.contact_backfill import (
    BackfillResult,
    backfill_contacts,
    find_promotable_contacts,
)


@click.group(name="wa-contacts")
def wa_contacts():
    """Gestionar contact notes en 99-Contacts/ desde stats del bridge WA.

    Las notas viven en `<vault>/99-obsidian/99-Contacts/<Nombre>.md`. El
    listener TS las lee para inyectar kinship/short_name/dossier al draft
    prompt del bot. Sin nota → kinship=unknown → registro genérico.
    """
    pass


def _format_results_table(results: list[BackfillResult], dry_run: bool) -> None:
    """Imprimir tabla legible de resultados al console (rich)."""
    from rich.table import Table

    if not results:
        console.print("[yellow]Ningún chat candidato encontrado en el bridge.[/yellow]")
        return

    # Counts.
    by_action: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for r in results:
        by_action[r.action] = by_action.get(r.action, 0) + 1
        if r.tier and r.tier != "n/a":
            by_tier[r.tier] = by_tier.get(r.tier, 0) + 1

    # Resumen.
    console.print("\n[bold cyan]Resumen del backfill[/bold cyan]")
    for action, cnt in sorted(by_action.items(), key=lambda x: -x[1]):
        marker = "[green]●[/green]" if action in ("created", "would_create") else "[dim]●[/dim]"
        console.print(f"  {marker} {action}: {cnt}")

    if by_tier:
        console.print("\n[bold cyan]Distribución por tier (creados/would-create)[/bold cyan]")
        for tier, cnt in sorted(by_tier.items(), key=lambda x: -x[1]):
            console.print(f"  · {tier}: {cnt}")

    # Tabla detallada (solo creates + multi_sender; los exists ya están bien).
    interesting = [
        r for r in results
        if r.action in ("created", "would_create", "skipped_multi_sender", "error")
    ]
    if interesting:
        title = (
            "[bold]Plan de notas a crear (dry-run)[/bold]"
            if dry_run
            else "[bold]Notas creadas[/bold]"
        )
        table = Table(title=title, show_lines=False, expand=False)
        table.add_column("Filename", style="cyan")
        table.add_column("Tier", style="magenta")
        table.add_column("Msgs", justify="right")
        table.add_column("Hace días", justify="right")
        table.add_column("JID", style="dim")
        table.add_column("Acción", style="green")
        for r in sorted(
            interesting,
            key=lambda x: (
                {"core": 0, "active": 1, "transient": 2, "n/a": 3}.get(x.tier, 99),
                -x.msg_count,
            ),
        ):
            jid_short = r.chat_jid.split("@")[0][:24]
            table.add_row(
                r.filename or "—",
                r.tier or "—",
                str(r.msg_count),
                f"{r.days_since_last:.0f}",
                jid_short,
                r.action,
            )
        console.print(table)


@wa_contacts.command()
@click.option(
    "--apply",
    "do_apply",
    is_flag=True,
    default=False,
    help="Escribir las notas en el vault. Default: dry-run (solo planifica).",
)
@click.option(
    "--days",
    "days_window",
    type=int,
    default=365,
    show_default=True,
    help="Horizonte en días del scan del bridge.",
)
@click.option(
    "--min-msgs",
    "min_msgs",
    type=int,
    default=1,
    show_default=True,
    help="Filtro mínimo de msg_count para considerar el chat (default 1 — incluye one-shots).",
)
def backfill(do_apply: bool, days_window: int, min_msgs: int) -> None:
    """Crear contact notes faltantes desde stats del bridge WhatsApp.

    Default es dry-run — corré sin --apply primero para ver qué notas se
    crearían, y después agregá --apply para escribirlas. Las notas
    existentes NUNCA se pisan: el user es dueño de su contenido.

    Lo que se decide automático:
      · `tier` (transient/active/core) — basado en msg_count, span,
        recency.
      · `wa_jid` del frontmatter / body — pre-poblado.
      · Footer auto-generado con stats al final de la nota.

    Lo que el user completa después:
      · `kinship` (override del frontmatter)
      · `short_name` / `Apodo`
      · `Relación`, `Apellido`, `Cumpleaños`, `Teléfono`, `Notas`, etc.
    """
    vault_root = _resolve_vault_path()
    if not vault_root or not Path(vault_root).exists():
        console.print("[red]No se pudo resolver vault root.[/red]")
        raise SystemExit(1)

    console.print(
        f"[dim]Vault:[/dim] {vault_root}\n"
        f"[dim]Modo:[/dim] {'APPLY' if do_apply else 'dry-run'}\n"
        f"[dim]Window:[/dim] {days_window}d  [dim]Min msgs:[/dim] {min_msgs}\n"
    )
    results = backfill_contacts(
        vault_root=Path(vault_root),
        dry_run=not do_apply,
        days_window=days_window,
        min_msgs=min_msgs,
    )
    _format_results_table(results, dry_run=not do_apply)
    if not do_apply and any(r.action == "would_create" for r in results):
        console.print(
            "\n[yellow]Dry-run.[/yellow] Para escribir las notas: "
            "[bold]rag wa-contacts backfill --apply[/bold]"
        )


@wa_contacts.command(name="stats")
@click.option(
    "--days",
    "days_window",
    type=int,
    default=365,
    show_default=True,
)
def stats_cmd(days_window: int) -> None:
    """Resumen del scan (sin escribir nada)."""
    vault_root = _resolve_vault_path()
    if not vault_root or not Path(vault_root).exists():
        console.print("[red]No se pudo resolver vault root.[/red]")
        raise SystemExit(1)

    results = backfill_contacts(
        vault_root=Path(vault_root),
        dry_run=True,
        days_window=days_window,
    )
    _format_results_table(results, dry_run=True)


@wa_contacts.command(name="promote-check")
@click.option(
    "--days",
    "days_window",
    type=int,
    default=365,
    show_default=True,
    help="Horizonte en días para recalcular tier desde el bridge.",
)
def promote_check_cmd(days_window: int) -> None:
    """Listar notas existentes que cualifican para subir de tier.

    Recorre las notas con `wa_jid` populado en `99-Contacts/`, recalcula
    el tier actual desde stats del bridge, y lista los casos donde el
    tier nuevo es mayor que el del frontmatter (transient → active, etc).

    No edita las notas — el user decide si actualizar tier + completar
    campos faltantes (ahora que el contacto "merece" un dossier más
    detallado).
    """
    from rich.table import Table

    vault_root = _resolve_vault_path()
    if not vault_root or not Path(vault_root).exists():
        console.print("[red]No se pudo resolver vault root.[/red]")
        raise SystemExit(1)

    candidates = find_promotable_contacts(
        vault_root=Path(vault_root),
        days_window=days_window,
    )

    if not candidates:
        console.print(
            "[green]Ninguna nota cualifica para promote.[/green] "
            "Todos los tiers están alineados con la actividad real."
        )
        return

    console.print(
        f"\n[bold cyan]{len(candidates)} contacto(s) cualifican para promote[/bold cyan]\n"
    )
    table = Table(show_lines=False, expand=False)
    table.add_column("Nota", style="cyan")
    table.add_column("Tier actual", style="dim")
    table.add_column("→ Nuevo", style="green")
    table.add_column("Msgs", justify="right")
    table.add_column("Hace días", justify="right")
    for c in candidates:
        table.add_row(
            c.note_path.name,
            c.current_tier,
            c.new_tier,
            str(c.msg_count),
            f"{c.days_since_last:.0f}",
        )
    console.print(table)
    console.print(
        "\n[yellow]Acción sugerida:[/yellow] abrí cada nota en Obsidian, "
        "actualizá `tier:` en frontmatter y completá los campos "
        "(`kinship`, `Apodo`, `Relación`, `Notas`) que ahora valen la pena."
    )
