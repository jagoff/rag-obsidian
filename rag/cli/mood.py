"""`rag mood` — score diario sobre Spotify + journal + WA + queries + calendar.

Phase 2e de modularización CLI (2026-05-09). Behind flag RAG_MOOD_ENABLED=1.

## Subcomandos

- `show [--days N]`     sparkline + score actual + drift status
- `explain [--date D]`  señales del día con evidence
- `compute [--date D]`  re-corre el agregador (UPSERT)
- `enable` / `disable`  activa/desactiva el daemon mood-poll
- `status`              estado del feature: env flag + daemon state
- `poll`                dispara un cycle del daemon manualmente
- `predict`             predicción mood de mañana via LinearRegression

## Lazy imports

`console`, `rag.mood`, `rag.cross_source_patterns.predict_mood_tomorrow`
viven afuera del módulo. Lazy adentro del cuerpo de cada command para
evitar circular import.

## Helpers

- `_mood_score_color(score)` — Rich color tier según signo + magnitud.
- `_mood_score_bar(score)` — glyph Unicode block para sparkline.

Re-exportados desde `rag/__init__.py` para preservar back-compat con
tests legacy (`tests/test_mood.py:875+` hace `rag._mood_score_bar(...)`).

## Re-export

`rag/__init__.py` registra el group via
`cli.add_command(mood_group, name="mood")`.
"""

from __future__ import annotations

import json

import click

__all__ = [
    "_mood_score_color",
    "_mood_score_bar",
    "mood_group",
    "mood_show_cmd",
    "mood_explain_cmd",
    "mood_compute_cmd",
    "mood_enable_cmd",
    "mood_disable_cmd",
    "mood_status_cmd",
    "mood_poll_cmd",
    "mood_predict_cmd",
]


def _mood_score_color(score: float) -> str:
    """Color Rich según signo + magnitud del score."""
    if score <= -0.5:
        return "bright_red"
    if score <= -0.2:
        return "yellow"
    if score >= 0.5:
        return "bright_green"
    if score >= 0.2:
        return "green"
    return "white"


def _mood_score_bar(score: float, *, width: int = 7) -> str:
    """Glyph para sparkline de un día. Score → caracter Unicode block.
    score ≈ 0 → ─, negativo → bloques inferiores, positivo → superiores."""
    # 8-step scale from -1 to +1.
    glyphs = "▁▂▃▄▅▆▇█"
    if score == 0:
        return "─"
    # Map score [-1, 1] → idx [0, 7]. score=-1 → ▁ (más bajo).
    idx = max(0, min(7, int(round((score + 1.0) * 3.5))))
    return glyphs[idx]


@click.group("mood", invoke_without_command=False)
def mood_group() -> None:
    """Mood tracking — score diario sobre Spotify + journal + WA + queries
    + calendar. Behind flag RAG_MOOD_ENABLED=1.

    Subcomandos:

    - rag mood show [--days N]    sparkline + score actual
    - rag mood explain [--date D] señales del día con evidence
    - rag mood compute [--date D] re-corre el agregador (UPSERT)
    """


@mood_group.command("show")
@click.option("--days", default=14, type=int, help="Días para sparkline (default 14)")
@click.option("--plain", is_flag=True, help="Output mínimo, sin Rich")
def mood_show_cmd(days: int, plain: bool) -> None:
    """Muestra sparkline ASCII + score actual + drift status."""
    from rag import console, mood as _mood  # noqa: PLC0415
    if not _mood._is_mood_enabled():
        click.echo("mood feature off (set RAG_MOOD_ENABLED=1)")
        return
    rows = _mood.get_recent_scores(days=days)
    drift = _mood.recent_drift(days=7)
    today_row = _mood.get_score_for_date(_mood._today_local())

    if plain:
        if not rows:
            click.echo("no_data")
            return
        for r in reversed(rows):
            click.echo(f"{r['date']}\t{r['score']:+.2f}\tn={r['n_signals']}")
        if today_row:
            click.echo(f"today\tscore={today_row['score']:+.2f}\tn={today_row['n_signals']}\tsources={today_row['sources_used']}")
        click.echo(f"drift\tdrifting={drift['drifting']}\tconsec={drift['n_consecutive']}\treason={drift.get('reason') or 'streak_active'}")
        return

    if not rows:
        console.print("[dim]no hay scores todavía. Corré `rag mood compute` después de que el daemon junte señales.[/dim]")
        return

    # Sparkline cronológico (oldest → newest).
    spark_chars: list[str] = []
    spark_dates: list[str] = []
    for r in reversed(rows):
        if r["n_signals"] == 0:
            spark_chars.append("·")
        else:
            color = _mood_score_color(r["score"])
            spark_chars.append(f"[{color}]{_mood_score_bar(r['score'])}[/{color}]")
        spark_dates.append(r["date"][-5:])  # MM-DD

    console.print(f"\n[bold]mood — last {days} days[/bold]")
    console.print(" ".join(spark_chars))
    console.print("[dim]" + " ".join(spark_dates) + "[/dim]")

    if today_row and today_row["n_signals"] > 0:
        color = _mood_score_color(today_row["score"])
        console.print(f"\nhoy: [{color}]{today_row['score']:+.2f}[/{color}] "
                      f"({today_row['n_signals']} señales · sources: {', '.join(today_row['sources_used'])})")
        for s in (today_row["top_evidence"] or [])[:3]:
            ev = s.get("evidence", {})
            ev_str = ", ".join(f"{k}={v}" for k, v in list(ev.items())[:3])
            console.print(f"  [dim]· {s['source']}/{s['signal_kind']}[/] "
                          f"[{_mood_score_color(s['value'])}]{s['value']:+.2f}[/] [dim]{ev_str}[/dim]")
    else:
        console.print("\n[dim]hoy: sin score computado todavía (corré `rag mood compute`)[/dim]")

    # Drift status.
    if drift["drifting"]:
        console.print(f"\n[bright_red]⚠ drift detectado:[/bright_red] "
                      f"{drift['n_consecutive']} días consecutivos con score ≤ -0.4 "
                      f"(avg {drift['avg_score']:+.2f})")
        console.print(f"[dim]  fechas: {', '.join(drift['dates'])}[/dim]")
    else:
        if drift["n_consecutive"] > 0:
            console.print(f"\n[dim]racha actual: {drift['n_consecutive']} días bajo threshold "
                          f"(reason: {drift.get('reason') or 'n/a'})[/dim]")
        else:
            console.print("\n[green]sin racha negativa activa[/green]")


@mood_group.command("explain")
@click.option("--date", default=None, help="YYYY-MM-DD (default: hoy)")
@click.option("--plain", is_flag=True)
def mood_explain_cmd(date: str | None, plain: bool) -> None:
    """Lista las señales del día con evidence completa."""
    from rag import console, mood as _mood  # noqa: PLC0415
    if not _mood._is_mood_enabled():
        click.echo("mood feature off (set RAG_MOOD_ENABLED=1)")
        return
    target = date or _mood._today_local()
    signals = _mood._read_signals_for_date(target)
    score_row = _mood.get_score_for_date(target)

    if plain:
        click.echo(f"date={target}")
        if score_row:
            click.echo(f"score={score_row['score']:+.3f} n={score_row['n_signals']} "
                       f"sources={','.join(score_row['sources_used'])}")
        else:
            click.echo("score=none (run `rag mood compute`)")
        for s in signals:
            ev_short = json.dumps(s["evidence"], ensure_ascii=False)[:120]
            click.echo(f"  {s['source']}/{s['signal_kind']}\tval={s['value']:+.2f}\tw={s['weight']:.1f}\tev={ev_short}")
        return

    console.print(f"\n[bold]mood signals — {target}[/bold]")
    if score_row:
        color = _mood_score_color(score_row["score"])
        console.print(f"score: [{color}]{score_row['score']:+.3f}[/] "
                      f"({score_row['n_signals']} señales · "
                      f"sources: {', '.join(score_row['sources_used'])})\n")
    else:
        console.print("[yellow]score no computado todavía. Corré `rag mood compute --date " + target + "`[/yellow]\n")

    if not signals:
        console.print("[dim]no hay señales para esta fecha[/dim]")
        return

    for s in signals:
        color = _mood_score_color(s["value"])
        console.print(f"[bold]{s['source']}/{s['signal_kind']}[/]  "
                      f"[{color}]{s['value']:+.2f}[/]  [dim]w={s['weight']:.1f}[/dim]")
        ev = s.get("evidence") or {}
        for k, v in ev.items():
            v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            if len(v_str) > 100:
                v_str = v_str[:97] + "..."
            console.print(f"  [dim]{k}:[/dim] {v_str}")
        console.print()


@mood_group.command("compute")
@click.option("--date", default=None, help="YYYY-MM-DD (default: hoy)")
@click.option("--plain", is_flag=True)
def mood_compute_cmd(date: str | None, plain: bool) -> None:
    """Re-computa el score diario (UPSERT en rag_mood_score_daily)."""
    from rag import console, mood as _mood  # noqa: PLC0415
    if not _mood._is_mood_enabled():
        click.echo("mood feature off (set RAG_MOOD_ENABLED=1)")
        return
    result = _mood.compute_daily_score(date)
    if plain:
        click.echo(f"date={result['date']} score={result['score']:+.3f} "
                   f"n={result['n_signals']} sources={','.join(result['sources_used'])}")
        return
    color = _mood_score_color(result["score"])
    console.print(f"[bold]{result['date']}[/]  "
                  f"score: [{color}]{result['score']:+.3f}[/]  "
                  f"({result['n_signals']} señales)")
    if result["sources_used"]:
        console.print(f"sources: {', '.join(result['sources_used'])}")


@mood_group.command("enable")
@click.option("--plain", is_flag=True)
def mood_enable_cmd(plain: bool) -> None:
    """Activa el daemon mood-poll (crea el state file).

    Después de esto, el plist `com.fer.obsidian-rag-mood-poll` (cargado
    por `rag setup`) empieza a juntar señales cada 30min. Si el plist
    no está cargado, ejecutar `rag setup` primero."""
    from rag import console, mood as _mood  # noqa: PLC0415
    was_enabled = _mood.is_daemon_enabled()
    _mood.enable_daemon()
    state_path = _mood._daemon_state_file()
    if plain:
        click.echo(f"daemon=enabled state_file={state_path} was_already_enabled={was_enabled}")
        return
    if was_enabled:
        console.print(f"[dim]ya estaba enabled[/dim] · {state_path}")
    else:
        console.print("[green]✓ daemon mood-poll enabled[/green]")
        console.print(f"[dim]state file: {state_path}[/dim]")
        console.print("[dim]el próximo tick (≤30min) va a empezar a juntar señales.[/dim]")
        console.print("[dim]forzar tick ahora: launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-mood-poll[/dim]")


@mood_group.command("disable")
@click.option("--plain", is_flag=True)
def mood_disable_cmd(plain: bool) -> None:
    """Desactiva el daemon mood-poll (borra el state file).

    El plist sigue cargado pero los ticks de 30min hacen exit-early
    (1 stat() syscall + return). Para des-cargar el plist completamente
    usar `launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-mood-poll.plist`."""
    from rag import console, mood as _mood  # noqa: PLC0415
    was_enabled = _mood.is_daemon_enabled()
    _mood.disable_daemon()
    if plain:
        click.echo(f"daemon=disabled was_enabled={was_enabled}")
        return
    if was_enabled:
        console.print("[yellow]daemon mood-poll disabled[/yellow]")
        console.print("[dim]los datos ya recolectados quedan intactos en rag_mood_signals + rag_mood_score_daily.[/dim]")
    else:
        console.print("[dim]ya estaba disabled[/dim]")


@mood_group.command("status")
@click.option("--plain", is_flag=True)
def mood_status_cmd(plain: bool) -> None:
    """Muestra estado del feature: env flag + daemon state."""
    from rag import console, mood as _mood  # noqa: PLC0415
    flag_on = _mood._is_mood_enabled()
    daemon_on = _mood.is_daemon_enabled()
    state_path = _mood._daemon_state_file()
    if plain:
        click.echo(f"env_flag={'on' if flag_on else 'off'} "
                   f"daemon={'enabled' if daemon_on else 'disabled'} "
                   f"state_file={state_path} state_exists={state_path.exists()}")
        return
    flag_label = "[green]on[/green]" if flag_on else "[red]off[/red]"
    daemon_label = "[green]enabled[/green]" if daemon_on else "[yellow]disabled[/yellow]"
    console.print(f"env flag (RAG_MOOD_ENABLED): {flag_label}")
    console.print(f"daemon state:                {daemon_label}")
    console.print(f"[dim]state file: {state_path} (exists={state_path.exists()})[/dim]")
    if not flag_on:
        console.print("[dim]→ exportá RAG_MOOD_ENABLED=1 para usar la CLI[/dim]")
    if not daemon_on:
        console.print("[dim]→ corré `rag mood enable` para que el daemon junte señales[/dim]")


@mood_group.command("poll")
@click.option("--no-llm", is_flag=True, help="Saltea la rama LLM (journal sentiment) — barato")
@click.option("--dry-run", is_flag=True, help="Corre los scorers pero NO persiste señales ni recompute")
@click.option("--plain", is_flag=True)
def mood_poll_cmd(no_llm: bool, dry_run: bool, plain: bool) -> None:
    """Dispara un cycle del daemon manualmente (útil para debugging
    sin esperar 30min).

    Por defecto persiste señales en `rag_mood_signals` y recomputa el
    score diario en `rag_mood_score_daily`. `--dry-run` los inhibe."""
    from rag import console, mood as _mood  # noqa: PLC0415
    if not _mood._is_mood_enabled():
        click.echo("mood feature off (set RAG_MOOD_ENABLED=1)")
        return
    result = _mood.run_poll_cycle(use_llm=not no_llm, persist=not dry_run)
    if plain:
        click.echo(json.dumps(result, ensure_ascii=False))
        return
    if result.get("reason"):
        console.print(f"[yellow]skipped:[/yellow] {result['reason']}")
        return
    console.print(f"[bold]mood poll cycle[/bold] · "
                  f"{result['n_signals_emitted']} señales · "
                  f"elapsed {result.get('elapsed_s', 0):.2f}s")
    for source, n in result.get("scorers", {}).items():
        if n == "error":
            console.print(f"  [red]{source}: error[/red]")
        else:
            color = "green" if n > 0 else "dim"
            console.print(f"  [{color}]{source}: {n} señal(es)[/{color}]")
    score = result.get("score")
    if isinstance(score, dict):
        c = _mood_score_color(score["value"])
        console.print(f"\nscore hoy: [{c}]{score['value']:+.3f}[/] "
                      f"({score['n_signals']} señales · sources: {', '.join(score['sources_used'])})")


@mood_group.command("predict")
@click.option("--days", default=60, type=int, help="Histórico para entrenar (default 60)")
@click.option("--plain", is_flag=True)
def mood_predict_cmd(days: int, plain: bool) -> None:
    """Predicción del mood de mañana via LinearRegression.

    Entrena con features de los últimos N días (lag-1: features[t-1] →
    mood[t]), aplica al estado de hoy, devuelve un valor en [-1, +1] +
    confidence (R²) + top features que más contribuyen.

    Importante: NO es verdad — es estimación basada en patrones
    recientes que pueden cambiar. Si el R² es bajo (<0.3), el modelo
    está adivinando — desconfiá del número y mirá las features."""
    from rag import console, mood as _mood  # noqa: PLC0415
    from rag.cross_source_patterns import predict_mood_tomorrow  # noqa: PLC0415
    if not _mood._is_mood_enabled():
        click.echo("mood feature off (set RAG_MOOD_ENABLED=1)")
        return
    days_clamped = max(21, min(int(days), 90))
    result = predict_mood_tomorrow(days=days_clamped)

    if result is None:
        if plain:
            click.echo(
                f"prediction=none reason=insufficient_data "
                f"min_required=21 days_tried={days_clamped}"
            )
            return
        console.print(
            "[yellow]sin data suficiente para predecir[/] "
            f"[dim](mínimo 21 días con mood + features alineadas; "
            f"intentamos {days_clamped})[/dim]"
        )
        return

    pred = result.get("prediction")
    conf = result.get("confidence", 0.0)
    n = result.get("n_training_days", 0)
    target = result.get("target_date")
    top = result.get("top_features") or []

    if pred is None:
        # Modelo entrenado pero no podemos predecir hoy → mañana por
        # missing features de hoy.
        reason = result.get("reason", "missing_features_today")
        if plain:
            click.echo(f"prediction=none reason={reason} confidence={conf}")
            return
        console.print(f"[yellow]modelo entrenado (R²={conf:+.2f}, n={n}) "
                      f"pero falta data de HOY para predecir mañana[/]")
        console.print(f"[dim]reason: {reason}[/]")
        return

    model_name = result.get("model", "linear")
    alpha = result.get("alpha")
    cv_splits = result.get("cv_n_splits", 0)
    conf_in = result.get("confidence_in_sample", conf)

    if plain:
        alpha_part = f" alpha={alpha:.2f}" if alpha is not None else ""
        click.echo(
            f"prediction={pred:+.3f} confidence={conf:.3f} "
            f"confidence_in_sample={conf_in:.3f} "
            f"model={model_name}{alpha_part} cv_splits={cv_splits} "
            f"n_training={n} target_date={target}"
        )
        for f in top:
            dev = f.get("deviation_contribution", f["contribution"])
            baseline = f.get("value_baseline", 0.0)
            click.echo(
                f"  feature={f['feature']} coef={f['coef']:+.4f} "
                f"value_today={f['value_today']:+.3f} "
                f"baseline={baseline:+.3f} "
                f"deviation_contribution={dev:+.3f} "
                f"contribution={f['contribution']:+.3f}"
            )
        return

    color = _mood_score_color(pred)
    # CV R² puede ser negativo cuando el modelo es peor que predecir
    # la media — eso es signal de "no aprendí nada" y la UI lo flaggea.
    conf_color = (
        "green" if conf >= 0.5 else "yellow" if conf >= 0.2 else "red"
    )
    sign = "+" if pred > 0 else ""
    console.print(f"\n[bold]predicción mood mañana ({target})[/]")
    console.print(f"  [{color}]{sign}{pred:.2f}[/]  "
                  f"confianza CV R² [{conf_color}]{conf:+.2f}[/] "
                  f"[dim](n={n} días, modelo={model_name}"
                  + (f", α={alpha:.2f}" if alpha is not None else "")
                  + f", in-sample R²={conf_in:+.2f})[/dim]")
    if conf < 0.2:
        console.print("[dim]⚠ CV R² bajo — el modelo no está aprendiendo "
                      "del histórico. Mirá los features individuales más "
                      "que el número.[/dim]")
    if top:
        console.print("\n[dim]top features (ordenadas por desviación vs "
                      "tu promedio histórico):[/]")
        for f in top:
            # Usamos deviation_contribution como medida de "qué tan
            # inusual es hoy esta feature, ponderado por su peso en
            # el modelo". Más interpretable que contribution.
            dev = f.get("deviation_contribution", f["contribution"])
            c = "bright_red" if abs(dev) >= 0.3 else "yellow" if abs(dev) >= 0.1 else "white"
            sign_c = "+" if dev > 0 else ""
            baseline = f.get("value_baseline", 0.0)
            console.print(
                f"  [{c}]{f['feature']:<28}[/] "
                f"[dim]hoy={f['value_today']:+.2f} vs base={baseline:+.2f}[/] "
                f"[{c}]→ {sign_c}{dev:.2f}[/]"
            )
