"""`rag synth-queries` — synthetic query generation + hard negative mining.

Phase 3 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el grupo CLI `rag synth-queries` con sus 3 sub-commands
desde `rag/__init__.py` a `rag/cli/synth_queries.py`.

## Pipeline (Sprint 2.A)

Augmentar training data del LightGBM lambdarank desde el corpus,
sin depender del feedback real escaso. Tres comandos:

  generate         Para cada nota del vault, qwen2.5:7b genera 3-5
                   queries (~3-5s/nota, 75-125min para 1500 notas).
  mine-negatives   Para cada synth query, busca top-K NN via embedding
                   y filtra self + duplicates → hard negatives.
  stats            Resumen de las 2 tablas (synthetic_queries +
                   synthetic_negatives).

Toggle con `RAG_LGBM_TRAIN_ENABLED=1` (default OFF — feeder del
LightGBM que NO se usa en retrieve()).

## Patrón CLI sub-package

`@click.group()` standalone (NO `@cli.group(...)`). El group `cli`
de `rag/__init__.py` se registra al final via `cli.add_command(synth_queries)`.
Sub-commands usan `@synth_queries.command(...)` (mismo patrón que
`rag/cli/vault.py` con `@vault.command`).
"""

from __future__ import annotations

import json as _json
import os

import click

__all__ = [
    "synth_queries",
    "synth_queries_generate",
    "synth_queries_mine_negatives",
    "synth_queries_stats",
]


@click.group("synth-queries")
def synth_queries() -> None:
    """Synthetic query generation + hard negative mining (Sprint 2.A).

    Pipeline para augmentar training data del LightGBM lambdarank desde
    el corpus, sin depender del feedback real escaso. Tres comandos:

      generate         Para cada nota del vault, qwen2.5:7b genera 3-5
                       queries (~3-5s/nota, 75-125min para 1500 notas).
      mine-negatives   Para cada synth query, busca top-K NN via embedding
                       y filtra self + duplicates → hard negatives.
      stats            Resumen de las 2 tablas (synthetic_queries +
                       synthetic_negatives).
    """


@synth_queries.command("generate")
@click.option("--limit", default=None, type=int,
              help="Máximo notas a procesar. None = todas.")
@click.option("--queries-per-note", "queries_per_note", default=4,
              show_default=True,
              help="Cuántas queries pedirle al LLM por nota.")
@click.option("--model", default="qwen2.5:7b", show_default=True,
              help="Modelo LLM para generar.")
@click.option("--source", "source", default="vault", show_default=True,
              help="Source a procesar. 'vault' = FS-based (default). "
                   "Cualquier otro (whatsapp/gmail/calendar/drive/"
                   "reminders/safari/contacts/calls) lee items desde "
                   "la collection vectorial filtrando por meta.source "
                   "— Quick Win #4.")
@click.option("--apply", is_flag=True,
              help="Persistir a rag_synthetic_queries (default: dry-run).")
@click.option("--json", "as_json", is_flag=True,
              help="Output JSON (para launchd).")
def synth_queries_generate(
    limit: int | None,
    queries_per_note: int,
    model: str,
    source: str,
    apply: bool,
    as_json: bool,
) -> None:
    """Generar queries sintéticas para el corpus.

    Default: vault (FS-based). Para sources cross-source:
      rag synth-queries generate --source gmail --apply
      rag synth-queries generate --source whatsapp --apply
    """
    from rag import _ragvec_state_conn, console  # noqa: PLC0415

    if os.environ.get("RAG_LGBM_TRAIN_ENABLED", "0").strip().lower() not in (
        "1", "true", "yes",
    ):
        msg = (
            "synth-queries deshabilitado — feeder del LGBM que no se usa en retrieve(). "
            "Setear RAG_LGBM_TRAIN_ENABLED=1 para correr."
        )
        if as_json:
            click.echo(_json.dumps({"skipped": True, "reason": msg}))
        else:
            console.print(f"[dim]{msg}[/dim]")
        return
    from rag_ranker_lgbm import (  # noqa: PLC0415
        CROSS_SOURCE_SOURCES,
        generate_synthetic_queries,
        generate_synthetic_queries_for_cross_source,
    )

    if source != "vault" and source not in CROSS_SOURCE_SOURCES:
        raise click.ClickException(
            f"--source={source!r} inválido. Valores: 'vault' o uno de "
            f"{CROSS_SOURCE_SOURCES}"
        )

    with _ragvec_state_conn() as conn:
        if source == "vault":
            result = generate_synthetic_queries(
                conn,
                limit=limit,
                queries_per_note=queries_per_note,
                model=model,
                dry_run=not apply,
            )
        else:
            result = generate_synthetic_queries_for_cross_source(
                conn,
                source=source,
                limit=limit,
                queries_per_note=queries_per_note,
                model=model,
                dry_run=not apply,
            )

    if as_json:
        out = {k: v for k, v in result.items() if k != "pairs_sample"}
        click.echo(_json.dumps(out))
        return

    console.print()
    console.print(
        f"[bold]rag synth-queries generate[/bold] "
        f"(limit={limit}, qpn={queries_per_note}, model={model}, "
        f"apply={apply})"
    )
    console.print()
    console.print(f"  Notas vistas:           {result['n_notes_seen']}")
    console.print(f"  Notas procesadas:       "
                  f"[green]{result['n_notes_processed']}[/green]")
    console.print(f"  Skip — sin cambios:     "
                  f"{result['n_notes_skipped_unchanged']}")
    console.print(f"  Skip — vacías:          "
                  f"{result['n_notes_skipped_empty']}")
    console.print(f"  Skip — LLM falló:       "
                  f"{result['n_notes_llm_failed']}")
    console.print(f"  Queries inserted:       "
                  f"[bold cyan]{result['n_queries_inserted']}[/bold cyan]")
    console.print(f"  Queries duplicadas:     "
                  f"{result['n_queries_skipped_duplicate']}")
    console.print(f"  Duración:               {result['duration_s']}s")
    console.print()

    if not apply:
        console.print(
            "[yellow]💡 Re-corré con --apply para persistir.[/yellow]"
        )

    if result.get("pairs_sample"):
        console.print("[bold]Sample (primeros 5 pairs)[/bold]")
        for pair in result["pairs_sample"][:5]:
            console.print(
                f"  [dim]{pair['kind']:14}[/dim] "
                f"[cyan]{pair['note_path'][:60]}[/cyan]"
            )
            console.print(f"     → {pair['query']}")


@synth_queries.command("mine-negatives")
@click.option("--limit", default=None, type=int,
              help="Máximo synthetic queries a procesar.")
@click.option("--negatives-per-query", "negatives_per_query", default=5,
              show_default=True)
@click.option("--apply", is_flag=True,
              help="Persistir negatives (default: dry-run).")
@click.option("--json", "as_json", is_flag=True)
def synth_queries_mine_negatives(
    limit: int | None,
    negatives_per_query: int,
    apply: bool,
    as_json: bool,
) -> None:
    """Mining hard negatives sobre las synthetic queries existentes."""
    from rag import _ragvec_state_conn, console  # noqa: PLC0415

    if os.environ.get("RAG_LGBM_TRAIN_ENABLED", "0").strip().lower() not in (
        "1", "true", "yes",
    ):
        msg = (
            "mine-negatives deshabilitado — feeder del LGBM que no se usa en retrieve(). "
            "Setear RAG_LGBM_TRAIN_ENABLED=1 para correr."
        )
        if as_json:
            click.echo(_json.dumps({"skipped": True, "reason": msg}))
        else:
            console.print(f"[dim]{msg}[/dim]")
        return
    from rag_ranker_lgbm import mine_hard_negatives_for_synthetic  # noqa: PLC0415

    with _ragvec_state_conn() as conn:
        result = mine_hard_negatives_for_synthetic(
            conn,
            limit=limit,
            negatives_per_query=negatives_per_query,
            dry_run=not apply,
        )

    if as_json:
        out = {k: v for k, v in result.items() if k != "sample_pairs"}
        click.echo(_json.dumps(out))
        return

    console.print()
    console.print(
        f"[bold]rag synth-queries mine-negatives[/bold] "
        f"(limit={limit}, neg_per_q={negatives_per_query}, apply={apply})"
    )
    console.print()
    console.print(f"  Queries examinadas:     {result['n_queries_examined']}")
    console.print(f"  Queries con negatives:  "
                  f"[green]{result['n_queries_with_negatives']}[/green]")
    console.print(f"  Queries sin neighbors:  "
                  f"{result['n_queries_no_neighbors']}")
    console.print(f"  Negatives inserted:     "
                  f"[bold cyan]{result['n_negatives_inserted']}[/bold cyan]")
    console.print(f"  Filtrados — self:       {result['n_filtered_self']}")
    console.print(f"  Filtrados — duplicate:  {result['n_filtered_duplicate']}")
    console.print()

    if not apply:
        console.print(
            "[yellow]💡 Re-corré con --apply para persistir.[/yellow]"
        )


@synth_queries.command("stats")
@click.option("--json", "as_json", is_flag=True)
def synth_queries_stats(as_json: bool) -> None:
    """Resumen de las tablas synthetic_queries + synthetic_negatives."""
    from rag import _ragvec_state_conn, console  # noqa: PLC0415
    from rag_ranker_lgbm import get_negatives_stats, get_synthetic_stats  # noqa: PLC0415

    with _ragvec_state_conn() as conn:
        synth = get_synthetic_stats(conn)
        neg = get_negatives_stats(conn)

    if as_json:
        click.echo(_json.dumps({"synthetic": synth, "negatives": neg}))
        return

    console.print()
    console.print("[bold]Synthetic queries[/bold]")
    console.print(f"  Total:           {synth['n_total']}")
    console.print(f"  Notas únicas:    {synth['n_unique_notes']}")
    console.print(f"  Por kind:        {synth['by_kind']}")
    console.print(f"  Por modelo:      {synth['by_model']}")
    console.print()
    console.print("[bold]Hard negatives[/bold]")
    console.print(f"  Total:                  {neg['n_total']}")
    console.print(f"  Queries únicas:         {neg['n_unique_queries']}")
    console.print(f"  Avg cosine to query:    {neg['avg_cosine_to_query']}")
    console.print()
