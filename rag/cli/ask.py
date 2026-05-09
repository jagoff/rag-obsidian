"""`rag ask` — quick-query alias minimalista de `rag query`.

Phase 3 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`).

## Diseño

Alias simple de `rag query` con UX más natural para queries rápidas.
`rag query` tiene ~15 flags y bastante superficie para uso avanzado;
`rag ask` es un entrypoint minimalista pensado para el día a día.

Diferencias clave con `rag query`:
- Argumento único: `rag ask "pregunta"`.
- `--quick` flag: skippea deep retrieve + multi-query para <2s TTFT.
- Banner inicial de emoji + pregunta recortada (UX visual).
- Todos los demás flags power-user quedan en `rag query`.

## Implementación

Delega via `ctx.invoke(query, ...)` al CLI command `query` de
`rag/__init__.py`. NO duplica lógica — solo mapea defaults.
"""

from __future__ import annotations

import click

__all__ = ["ask_cli"]


@click.command("ask")
@click.argument("question")
@click.option("--quick", is_flag=True,
              help="Modo rápido: skippea deep retrieve + multi-query (TTFT <2s).")
@click.option("--source", "source_opt", default=None,
              help="Filtrar por fuente (vault,calendar,gmail,whatsapp,...).")
@click.option("--session", "session_id", default=None,
              help="ID de sesión — conectar con historial existente.")
@click.option("--continue", "continue_", is_flag=True,
              help="Reanuda la última sesión.")
@click.option("--plain", is_flag=True,
              help="Salida plana sin UI — pipes / scripts.")
@click.pass_context
def ask_cli(
    ctx: click.Context, question: str, quick: bool,
    source_opt: str | None, session_id: str | None,
    continue_: bool, plain: bool,
):
    """Preguntar al vault — alias minimalista de `rag query`.

    Uso:
        rag ask "qué sé sobre ikigai"
        rag ask --quick "llueve hoy?"
        rag ask --source whatsapp "último mensaje de María"
        rag ask --continue "y el proyecto de coaching?"

    Para modo avanzado con todos los flags: `rag query ...`.
    Para conversación multi-turn: `rag chat`.
    """
    from rag import console, query  # noqa: PLC0415

    if not plain:
        # Minimal banner — just the question echoed with an emoji.
        q_preview = question.strip()
        if len(q_preview) > 80:
            q_preview = q_preview[:77] + "..."
        console.print()
        console.print(f"[cyan]❯[/cyan] [italic]{q_preview}[/italic]")
        console.print()

    # Delegate to the existing `query` command. `--quick` maps to --no-deep;
    # multi stays off (default). Other defaults left untouched so the
    # caller's UX mirrors `rag query` semantics exactly.
    ctx.invoke(
        query,
        question=question,
        k=5, folder=None, tag=None, since=None,
        hyde=False,
        multi=False,
        no_auto_filter=False,
        raw=False, loose=False, force=False,
        session_id=session_id, continue_=continue_,
        plain=plain,
        counter=False,
        no_deep=quick,
        critique=False,
        no_cache=False,
        source_opt=source_opt,
        vault_scope=None,
    )
