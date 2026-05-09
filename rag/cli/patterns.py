"""`rag patterns` — cross-source pattern detection vía Pearson sobre métricas diarias.

Phase 2e de modularización CLI (2026-05-09).

## Subcomandos

- `show [--days N] [--lags 0,1,7]`     top correlaciones cross-source
- `metrics [--days N]`                  lista métricas registradas + n_días
- `explain --pair A,B [--lag N]`        Pearson explícito sobre un par + lag

## Lazy imports

`console`, `rag.cross_source_patterns.{patterns_summary, metric_label,
collect_daily_metrics, known_metrics, _pearson, _align_series, _severity,
_format_description}` viven afuera del módulo. Lazy adentro del cuerpo
de cada command para evitar circular import.

## Helpers

- `_patterns_severity_color(severity)` — Rich color tier (strong/moderate/weak).
- `_patterns_lag_label(lag)` — label legible (mismo día / +1 día / +1 semana).

Re-exportados desde `rag/__init__.py` por consistencia con `_mood_score_*`
(otros tests legacy podrían empezar a hacer `rag._patterns_*`).

## Re-export

`rag/__init__.py` registra el group via
`cli.add_command(patterns_group, name="patterns")`.
"""

from __future__ import annotations

import click

__all__ = [
    "_patterns_severity_color",
    "_patterns_lag_label",
    "patterns_group",
    "patterns_show_cmd",
    "patterns_metrics_cmd",
    "patterns_explain_cmd",
]


def _patterns_severity_color(severity: str) -> str:
    if severity == "strong":
        return "bright_red"
    if severity == "moderate":
        return "yellow"
    return "white"


def _patterns_lag_label(lag: int) -> str:
    if lag == 0:
        return "mismo día"
    if lag == 1:
        return "+1 día"
    if lag == 7:
        return "+1 semana"
    return f"+{lag}d"


@click.group("patterns", invoke_without_command=False)
def patterns_group() -> None:
    """Cross-source pattern detection — Pearson sobre métricas diarias.

    Subcomandos:
      rag patterns show [--days N] [--lags 0,1,7]   → top correlaciones
      rag patterns metrics                           → lista métricas + n_días
      rag patterns explain --pair A,B [--lag N]     → detalle de un par
    """


@patterns_group.command("show")
@click.option("--days", default=30, type=int, help="Rango histórico (7-90)")
@click.option("--lags", default="0,1,7",
              help="Lags a testear separados por coma (default 0,1,7)")
@click.option("--top", default=10, type=int, help="Cuántas correlaciones mostrar")
@click.option("--plain", is_flag=True, help="Output mínimo, sin Rich")
def patterns_show_cmd(days: int, lags: str, top: int, plain: bool) -> None:
    """Top correlaciones cross-source con severity + lag + p-value."""
    from rag import console  # noqa: PLC0415
    from rag.cross_source_patterns import patterns_summary, metric_label  # noqa: PLC0415
    days_clamped = max(7, min(int(days), 90))
    try:
        lag_list = tuple(sorted(set(
            max(0, min(int(x.strip()), 14))
            for x in lags.split(",") if x.strip()
        ))) or (0,)
    except ValueError:
        lag_list = (0, 1, 7)

    summary = patterns_summary(
        days=days_clamped, top=int(top), lags=lag_list,
    )

    if plain:
        click.echo(f"days={summary['days_range']} lags={summary['lags_tested']} "
                   f"n_findings={summary['n_findings']} "
                   f"strong={summary['by_severity'].get('strong', 0)} "
                   f"moderate={summary['by_severity'].get('moderate', 0)}")
        for f in summary["top"]:
            a, b = f["pair"]
            click.echo(f"  [{f['severity']}] {a} × {b} (lag {f['lag']}d) "
                       f"r={f['r']:+.2f} n={f['n']} p={f['p']:.4f}")
        return

    console.print(f"\n[bold]cross-source patterns[/] · "
                  f"{summary['days_range']} días · lags {summary['lags_tested']}")
    if not summary["top"]:
        console.print("[dim]sin findings — necesitás más data acumulada "
                      "(min 21 días + |r|>=0.4)[/dim]")
        if summary["metrics_with_data"]:
            console.print("\n[dim]métricas con data:[/dim]")
            for name, n in summary["metrics_with_data"]:
                color = "green" if n >= 21 else "yellow" if n >= 7 else "red"
                console.print(f"  [{color}]{name}: {n}d[/{color}]")
        return

    summary_line = (
        f"[bright_red]{summary['by_severity'].get('strong', 0)} strong[/] · "
        f"[yellow]{summary['by_severity'].get('moderate', 0)} moderate[/] · "
        f"{summary['n_findings']} total"
    )
    console.print(summary_line + "\n")

    for f in summary["top"]:
        a, b = f["pair"]
        sev_color = _patterns_severity_color(f["severity"])
        lag_lbl = _patterns_lag_label(f["lag"])
        r_sign = "+" if f["r"] > 0 else ""
        console.print(
            f"  [{sev_color}]●[/] [bold]{metric_label(a)}[/] × "
            f"[bold]{metric_label(b)}[/] [dim]({lag_lbl})[/]"
        )
        console.print(
            f"    [dim]r=[/]{r_sign}{f['r']:.2f} "
            f"[dim]n=[/]{f['n']} [dim]p=[/]{f['p']:.4f} "
            f"[dim]· {f['description']}[/]"
        )


@patterns_group.command("metrics")
@click.option("--days", default=30, type=int)
@click.option("--plain", is_flag=True)
def patterns_metrics_cmd(days: int, plain: bool) -> None:
    """Lista las métricas registradas + cuántos días tienen data."""
    from rag import console  # noqa: PLC0415
    from rag.cross_source_patterns import (  # noqa: PLC0415
        collect_daily_metrics, known_metrics, metric_label,
    )
    days_clamped = max(7, min(int(days), 90))
    metrics = collect_daily_metrics(days=days_clamped)
    rows = sorted(
        ((name, len(metrics.get(name, {}))) for name in known_metrics()),
        key=lambda x: -x[1],
    )
    if plain:
        for name, n in rows:
            click.echo(f"{name}\t{n}")
        return
    console.print(f"\n[bold]métricas registradas[/] ({days_clamped} días):")
    for name, n in rows:
        if n >= 21:
            color, hint = "green", "ok para correlación"
        elif n >= 7:
            color, hint = "yellow", "no alcanza min_n=21"
        else:
            color, hint = "red", "casi sin data"
        console.print(
            f"  [{color}]{name:<28}[/] [dim]{metric_label(name):<32}[/] "
            f"[{color}]{n:>3}d[/] [dim]({hint})[/]"
        )


@patterns_group.command("explain")
@click.option("--pair", required=True,
              help="Pair en formato 'a,b' (ej. mood_score,sleep_quality)")
@click.option("--lag", default=0, type=int, help="Lag en días")
@click.option("--days", default=30, type=int)
@click.option("--plain", is_flag=True)
def patterns_explain_cmd(pair: str, lag: int, days: int, plain: bool) -> None:
    """Calcula Pearson explícito sobre un par + lag específico (debug)."""
    from rag import console  # noqa: PLC0415
    from rag.cross_source_patterns import (  # noqa: PLC0415
        collect_daily_metrics, _pearson, _align_series, metric_label,
        _severity, _format_description,
    )
    parts = [p.strip() for p in pair.split(",")]
    if len(parts) != 2:
        click.echo("error: --pair debe ser 'a,b'", err=True)
        return
    a, b = parts
    days_clamped = max(7, min(int(days), 90))
    metrics = collect_daily_metrics(days=days_clamped, metrics=[a, b])
    if a not in metrics or b not in metrics:
        click.echo("error: métrica desconocida — usá `rag patterns metrics`", err=True)
        return
    xs, ys, dates = _align_series(metrics[a], metrics[b], lag=int(lag))
    r, n, p = _pearson(xs, ys)
    sev = _severity(r)
    desc = _format_description(a, b, r, int(lag))
    if plain:
        click.echo(f"pair={a},{b} lag={lag} r={r:+.3f} n={n} p={p:.4f} "
                   f"severity={sev}")
        click.echo(f"description={desc}")
        for d, x, y in zip(dates, xs, ys):
            click.echo(f"  {d}\t{x:+.3f}\t{y:+.3f}")
        return

    sev_color = _patterns_severity_color(sev)
    console.print(f"\n[bold]{metric_label(a)}[/] × [bold]{metric_label(b)}[/] "
                  f"[dim](lag {lag}d)[/]")
    console.print(f"  [{sev_color}]r={r:+.3f}[/] · n={n} · p={p:.4f} · {sev}")
    console.print(f"  [dim]{desc}[/]\n")
    if dates:
        console.print(f"[dim]pares alineados ({len(dates)}):[/]")
        for d, x, y in zip(dates[:10], xs[:10], ys[:10]):
            console.print(f"  [dim]{d}[/]  {a}={x:+.3f}  {b}={y:+.3f}")
        if len(dates) > 10:
            console.print(f"  [dim]... +{len(dates) - 10} más[/]")
