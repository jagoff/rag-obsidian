"""``rag style`` — fingerprint del estilo de escritura del user.

Sub-comandos:

- ``rag style refresh`` — re-extrae features desde el bridge WA y
  persiste a SQL + markdown.
- ``rag style show`` — imprime la fingerprint más reciente.

Diseño y rationale en :mod:`rag.style` docstring.
"""
from __future__ import annotations

import json

import click

from rag import console


@click.group(name="style")
def style_cli():
    """Fingerprint del tono / estilo del user.

    Lee outbound de WhatsApp (filtrando bot replies + comandos),
    calcula features agregados (openers, voseo, slang, emojis, etc.)
    y persiste a ``rag_style_fingerprint`` + nota markdown en
    ``99-AI/style/profile.md``.
    """


@style_cli.command("refresh")
@click.option("--window-days", default=90, show_default=True,
              help="Ventana hacia atrás para extraer mensajes")
@click.option("--no-persist", is_flag=True, default=False,
              help="No escribe a telemetry.db (dry-run)")
@click.option("--no-export", is_flag=True, default=False,
              help="No exporta el markdown al vault")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output JSON en vez de tabla")
def style_refresh(window_days: int, no_persist: bool, no_export: bool,
                  as_json: bool):
    """Re-extrae el fingerprint desde el bridge WA."""
    from rag.style import refresh  # noqa: PLC0415
    snap = refresh(
        window_days=window_days,
        persist=not no_persist,
        export_markdown=not no_export,
    )
    if as_json:
        click.echo(json.dumps(snap, indent=2, ensure_ascii=False, default=str))
        return
    if not snap.get("ok"):
        console.print(f"[red]✗[/red] {snap.get('reason', 'falló')}")
        return
    f = snap.get("features", {})
    if f.get("insufficient_data"):
        console.print(
            f"[yellow]Datos insuficientes[/yellow]: solo "
            f"{snap.get('n_messages', 0)} mensajes en {window_days}d"
        )
        return
    console.print(
        f"[green]✓[/green] fingerprint actualizado · "
        f"id={snap.get('id')} · "
        f"{snap.get('n_messages', 0)}/{snap.get('n_raw', 0)} msgs filtrados "
        f"(últimos {window_days}d) · hash {snap.get('content_hash', '?')}"
    )
    if snap.get("exported_to"):
        console.print(f"  markdown → [dim]{snap['exported_to']}[/dim]")
    console.print(
        f"  avg {f.get('avg_chars')} chars · voseo {f.get('voseo_dominance', 0):.0%} · "
        f"slang {f.get('slang_hits', 0)} hits · emoji {f.get('emoji_rate', 0):.0%}"
    )


@style_cli.command("show")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output JSON")
def style_show(as_json: bool):
    """Muestra la fingerprint más reciente persistida."""
    from rag.style import load_latest, render_markdown  # noqa: PLC0415
    snap = load_latest()
    if snap is None:
        console.print(
            "[yellow]Sin fingerprint todavía.[/yellow] Corré "
            "[bold]rag style refresh[/bold] primero."
        )
        return
    if as_json:
        click.echo(json.dumps(snap, indent=2, ensure_ascii=False, default=str))
        return
    console.print(render_markdown(snap))
