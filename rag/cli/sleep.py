"""`rag sleep` — métricas de Pillow (iOS) sincronizadas via iCloud.

Phase 2e de modularización CLI (2026-05-09).

## Subcomandos

- `show [--days N]`   última noche + week vs hist + sparkline 7d
- `patterns`          Pearson r sobre todo el histórico (sleep + mood)
- `ingest`            corre el ingester (alias de `rag index --source pillow`)

## Lazy imports

`console`, `rag.integrations.pillow_sleep.{last_night, weekly_stats,
detect_patterns, ingest}` viven afuera del módulo. Lazy adentro del
cuerpo de cada command para evitar circular import.

## Re-export

`rag/__init__.py` registra el group via
`cli.add_command(sleep_group, name="sleep")`.
"""

from __future__ import annotations

import json

import click

__all__ = [
    "sleep_group",
    "sleep_show_cmd",
    "sleep_patterns_cmd",
    "sleep_ingest_cmd",
]


@click.group("sleep", invoke_without_command=False)
def sleep_group() -> None:
    """Sleep tracking — métricas de Pillow (iOS) sincronizadas via iCloud.

    Subcomandos:

    - rag sleep show [--days N]    última noche + week vs hist
    - rag sleep patterns           Pearson r sobre todo el histórico
    - rag sleep ingest             corre el ingester (alias de `rag index --source pillow`)
    """


@sleep_group.command("show")
@click.option("--days", default=7, type=int,
              help="Días para sparkline + agregados (default 7)")
@click.option("--plain", is_flag=True, help="Output mínimo, sin Rich")
def sleep_show_cmd(days: int, plain: bool) -> None:
    """Resumen de sueño: anoche + comparación con el histórico."""
    from rag import console  # noqa: PLC0415
    from rag.integrations.pillow_sleep import last_night, weekly_stats  # noqa: PLC0415
    ln = last_night()
    if ln is None:
        if plain:
            click.echo("no_data")
        else:
            console.print("[dim]sin sesiones todavía. Corré `rag sleep ingest`.[/dim]")
        return

    ws = weekly_stats()
    week = ws.get("week") or {}
    hist = ws.get("hist") or {}
    delta = ws.get("delta") or {}

    total_h = ln.get("sleep_total_h") or 0
    mins = round(total_h * 60)
    h, m = divmod(mins, 60)
    total_label = f"{h}h{m:02d}m"
    quality = ln.get("quality")
    deep_pct = ln.get("deep_pct")
    rem_pct = ln.get("rem_pct")
    awak = ln.get("awakenings", 0)

    if plain:
        click.echo(f"date\t{ln.get('date')}")
        click.echo(f"duration\t{total_label}\t{total_h:.2f}h")
        click.echo(f"bedtime\t{ln.get('bedtime_local')}\t→\t{ln.get('waketime_local')}")
        if quality is not None:
            click.echo(f"quality\t{quality:.3f}")
        if deep_pct is not None:
            click.echo(f"deep_pct\t{deep_pct:.1f}")
        if rem_pct is not None:
            click.echo(f"rem_pct\t{rem_pct:.1f}")
        click.echo(f"awakenings\t{awak}")
        if week.get("n"):
            click.echo(
                f"week_avg\tn={week['n']}\t"
                f"dur={week.get('duration_h', 0):.2f}h\t"
                f"q={week.get('quality', 0):.2f}\t"
                f"deep={week.get('deep_pct', 0):.1f}%"
            )
        if hist.get("n"):
            click.echo(
                f"hist_avg\tn={hist['n']}\t"
                f"dur={hist.get('duration_h', 0):.2f}h\t"
                f"q={hist.get('quality', 0):.2f}\t"
                f"deep={hist.get('deep_pct', 0):.1f}%"
            )
        return

    # Rich output
    console.print(f"\n[bold]sueño · anoche ({ln.get('date')})[/bold]")
    q_str = f"Q {quality:.2f}" if quality is not None else "—"
    console.print(
        f"  {total_label}  [dim]{q_str}[/dim]  "
        f"[dim]{ln.get('bedtime_local')} → {ln.get('waketime_local')}[/dim]"
    )

    # Stages con warn cues
    deep_color = "yellow" if (deep_pct is not None and deep_pct < 15) else "default"
    rem_color = "yellow" if (rem_pct is not None and rem_pct < 15) else "default"
    awak_color = "red" if awak >= 5 else "yellow" if awak >= 3 else "default"
    deep_str = f"{deep_pct:.0f}%" if deep_pct is not None else "—"
    rem_str = f"{rem_pct:.0f}%" if rem_pct is not None else "—"
    console.print(
        f"  [dim]deep[/dim] [{deep_color}]{deep_str}[/{deep_color}]  "
        f"[dim]rem[/dim] [{rem_color}]{rem_str}[/{rem_color}]  "
        f"[dim]awk[/dim] [{awak_color}]{awak}[/{awak_color}]"
    )

    if week.get("n") and hist.get("n"):
        console.print(f"\n[bold]vs histórico[/bold] (week n={week['n']} · hist n={hist['n']})")

        def fmt_delta(val: float | None, suffix: str, decimals: int = 2) -> str:
            if val is None:
                return "—"
            sign = "+" if val > 0 else ""
            color = "green" if val > 0 else "red" if val < -0.01 else "dim"
            return f"[{color}]{sign}{val:.{decimals}f}{suffix}[/{color}]"

        console.print(
            f"  duración: {week.get('duration_h', 0):.2f}h  "
            f"hist {hist.get('duration_h', 0):.2f}h  "
            f"Δ {fmt_delta(delta.get('duration_h'), 'h', 1)}"
        )
        console.print(
            f"  quality:  {week.get('quality', 0):.2f}  "
            f"hist {hist.get('quality', 0):.2f}  "
            f"Δ {fmt_delta(delta.get('quality'), '', 2)}"
        )
        console.print(
            f"  deep%:    {week.get('deep_pct', 0):.1f}%  "
            f"hist {hist.get('deep_pct', 0):.1f}%  "
            f"Δ {fmt_delta(delta.get('deep_pct'), 'pp', 1)}"
        )
        console.print(
            f"  awk/n:    {week.get('awakenings', 0):.1f}  "
            f"hist {hist.get('awakenings', 0):.1f}  "
            f"Δ {fmt_delta(delta.get('awakenings'), '', 1)}"
        )

    # Sparkline ASCII de quality 7d
    spark = ws.get("spark_quality_7d") or []
    if spark:
        bars = []
        for v in spark:
            if v is None:
                bars.append("·")
            elif v >= 0.85:
                bars.append("[green]█[/green]")
            elif v >= 0.7:
                bars.append("[default]▆[/default]")
            elif v >= 0.5:
                bars.append("[yellow]▄[/yellow]")
            else:
                bars.append("[red]▂[/red]")
        console.print(f"\n[dim]quality 7d:[/dim] {' '.join(bars)}")


@sleep_group.command("patterns")
@click.option("--min-r", default=0.3, type=float,
              help="|r| mínimo para incluir un finding (default 0.3)")
@click.option("--min-n", default=14, type=int,
              help="N mínimo de pares para que el finding sea válido (default 14)")
@click.option("--all", "show_all", is_flag=True,
              help="Mostrar TODOS los candidatos (incluyendo los que no pasan threshold)")
@click.option("--plain", is_flag=True, help="Output mínimo, sin Rich")
def sleep_patterns_cmd(min_r: float, min_n: int, show_all: bool, plain: bool) -> None:
    """Correlaciones Pearson r sobre el histórico de sueño + mood.

    Surface findings ordenados por |r| descendente. Severity tiers:
    weak (<0.4), moderate (<0.5), strong (≥0.5)."""
    from rag import console  # noqa: PLC0415
    from rag.integrations.pillow_sleep import detect_patterns  # noqa: PLC0415
    findings = detect_patterns(
        min_n=min_n,
        min_abs_r=0.0 if show_all else min_r,
    )
    if plain:
        if not findings:
            click.echo("no_findings")
            return
        for f in findings:
            click.echo(
                f"{f['kind']}\tr={f['r']:+.2f}\tn={f['n']}\t{f['severity']}\t{f['description']}"
            )
        return

    if not findings:
        console.print(
            f"[dim]sin findings con |r| ≥ {min_r:.2f} y n ≥ {min_n}. "
            f"Probá `rag sleep patterns --all` o subí los datos primero "
            f"con `rag sleep ingest`.[/dim]"
        )
        return

    console.print(f"\n[bold]sleep patterns · n={findings[0]['n']} sesiones[/bold]")
    for f in findings:
        if f["severity"] == "strong":
            color = "yellow"
        elif f["severity"] == "moderate":
            color = "default"
        else:
            color = "dim"
        sign = "+" if f["r"] > 0 else ""
        console.print(
            f"  [{color}]{f['kind']:30s}[/{color}]  "
            f"r={sign}{f['r']:.2f}  "
            f"[dim]{f['severity']:8s}[/dim]  "
            f"[{color}]{f['description']}[/{color}]"
        )


@sleep_group.command("ingest")
@click.option("--plain", is_flag=True, help="Output mínimo, sin Rich")
def sleep_ingest_cmd(plain: bool) -> None:
    """Corre el ingester de Pillow (alias de `rag index --source pillow`)."""
    from rag import console  # noqa: PLC0415
    from rag.integrations.pillow_sleep import ingest as _ingest_pillow  # noqa: PLC0415
    summary = _ingest_pillow()
    if plain:
        click.echo(json.dumps(summary, ensure_ascii=False, default=str))
        return
    if summary.get("skipped"):
        console.print(
            f"[dim]pillow: {summary.get('reason', 'skipped')} ({summary['file']})[/dim]"
        )
        return
    console.print(
        f"[green]pillow:[/green] {summary['total_parsed']} sesiones · "
        f"{summary['ingested']} ingest · "
        f"{summary['mood_signals']} mood signals · "
        f"{summary['elapsed_ms']:.0f}ms"
    )
