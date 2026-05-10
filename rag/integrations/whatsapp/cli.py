"""WhatsApp tasks extractor — `rag wa-tasks` Click command (Phase 4b of monolith split, 2026-04-25).

The bulk of the WA-tasks pipeline (`_fetch_whatsapp_window`, `_wa_extract_actions`,
`_wa_tasks_write_note`, `_wa_tasks_load_state`, `_wa_tasks_save_state`) was
already extracted to `rag/integrations/whatsapp.py` in Phase 1b. What remains
in `rag/__init__.py` is the thin CLI wrapper that orchestrates them — moved
here for consistency with the rest of the modularization.

## Surface

- `wa_tasks` — Click command `rag wa-tasks [--dry-run] [--hours N] [--force]`.
  Lee delta del bridge SQLite (is_from_me=0, ventana incremental vs último
  run). Agrupa por chat; si un chat tiene ≥2 mensajes entrantes nuevos,
  qwen2.5:3b destila action items. Escribe `00-Inbox/WA-YYYY-MM-DD.md` con
  frontmatter `ambient: skip` (evita loop) + wikilinks al archivo del chat/mes.

## Why deferred imports

El handler usa muchísimos helpers del parent package: `VAULT_PATH`,
`_wa_tasks_load_state`, `_wa_tasks_save_state`, `_fetch_whatsapp_window`,
`_wa_extract_actions`, `_wa_tasks_write_note`, `_ragvec_state_conn`,
`_sql_append_event`, `_sql_write_with_retry`, `_map_wa_tasks_row`. Todos
viven en `rag/__init__.py` o en `rag.integrations.whatsapp` (re-exportados
en `rag.<X>` via Phase 1b shim). Resolvemos via `from rag import ...` en
function body para que los monkey-patches de tests propaguen y para evitar
ciclos de import al load-time.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import click

from rag import cli, console


@cli.command("wa-tasks")
@click.option("--dry-run", is_flag=True,
              help="Imprimir acciones extraídas sin escribir el archivo")
@click.option("--hours", type=int, default=None,
              help="Ventana en horas (default: desde último run, o 24h si es la primera corrida)")
@click.option("--force", is_flag=True,
              help="Ignorar state file: procesar toda la ventana como si nunca se hubiera corrido")
def wa_tasks(dry_run: bool, hours: int | None, force: bool):
    """Extrae tareas, preguntas y compromisos de WhatsApp al Inbox.

    Lee delta del bridge SQLite (is_from_me=0, ventana incremental vs último
    run). Agrupa por chat; si un chat tiene ≥2 mensajes entrantes nuevos,
    qwen2.5:3b destila action items. Escribe `00-Inbox/WA-YYYY-MM-DD.md`
    con frontmatter `ambient: skip` (evita loop) + wikilinks al archivo
    del chat/mes. Idempotente: sin items nuevos → no-op silencioso.

    Ideal como launchd cada 30min — el file dispara el watch (00-Inbox
    no está excluido), el morning lo ve como evidencia, y `rag query`
    lo recupera como cualquier otra nota.
    """
    from rag import (
        VAULT_PATH,
        _fetch_whatsapp_window,
        _map_wa_tasks_row,
        _ragvec_state_conn,
        _sql_append_event,
        _sql_write_with_retry,
        _wa_extract_combined,
        _wa_promises_persist,
        _wa_tasks_load_state,
        _wa_tasks_save_state,
        _wa_tasks_write_note,
    )
    now = datetime.now()
    state = _wa_tasks_load_state() if not force else {"last_run_ts": None, "processed_ids": []}
    if hours is not None:
        since = now - timedelta(hours=max(1, int(hours)))
    elif state.get("last_run_ts"):
        try:
            since = datetime.fromisoformat(state["last_run_ts"])
        except Exception:
            since = now - timedelta(hours=24)
    else:
        since = now - timedelta(hours=24)

    processed = set(state.get("processed_ids") or [])
    by_chat = _fetch_whatsapp_window(since, now, processed)
    if not by_chat:
        console.print(f"[dim]sin chats con actividad nueva desde {since:%Y-%m-%d %H:%M}[/dim]")
        if not dry_run:
            state["last_run_ts"] = now.isoformat(timespec="seconds")
            _wa_tasks_save_state(state)
        return

    console.print(
        f"[dim]Ventana:[/dim] {since:%Y-%m-%d %H:%M} → {now:%H:%M} · "
        f"{len(by_chat)} chats con nuevos mensajes entrantes"
    )
    extractions: list[dict] = []
    for chat in by_chat:
        ext = _wa_extract_combined(chat["label"], chat["is_group"], chat["messages"])
        extractions.append(ext)
        n = len(ext["tasks"]) + len(ext["questions"]) + len(ext["commitments"])
        n_p = len(ext.get("promises") or [])
        tag = "[green]✓[/green]" if n or n_p else "[dim]·[/dim]"
        promise_tag = f" + {n_p} promesa(s)" if n_p else ""
        console.print(
            f"  {tag} [cyan]{chat['label']}[/cyan] · {chat['inbound']} inbound → {n} items{promise_tag}"
        )

    total = sum(
        len(e["tasks"]) + len(e["questions"]) + len(e["commitments"])
        for e in extractions
    )
    total_promises = sum(len(e.get("promises") or []) for e in extractions)
    if total == 0 and total_promises == 0:
        console.print("[yellow]sin items accionables extraídos[/yellow]")
        if not dry_run:
            state["last_run_ts"] = now.isoformat(timespec="seconds")
            # Still record the fetched msg ids so we don't re-scan them next run.
            for chat in by_chat:
                for mid in chat["new_ids"]:
                    state.setdefault("processed_ids", []).append(mid)
            _wa_tasks_save_state(state)
        return

    if dry_run:
        header = f"\n[bold]{total} items + {total_promises} promesa(s) extraídas (dry-run — no se escribe)[/bold]"
        console.print(header)
        for chat, ext in zip(by_chat, extractions):
            if not any(ext[k] for k in ("tasks", "questions", "commitments")) and not (ext.get("promises") or []):
                continue
            console.print(f"\n[cyan]{chat['label']}[/cyan]")
            for t in ext["tasks"]:
                console.print(f"  [ ] {t}")
            for q in ext["questions"]:
                console.print(f"  ❓ {q}")
            for c in ext["commitments"]:
                console.print(f"  📌 {c}")
            for p in ext.get("promises") or []:
                arrow = "→" if (p.get("direction") or "").lower() == "out" else "←"
                when = f" [{p['when_text']}]" if p.get("when_text") else ""
                console.print(f"  🤝 {arrow} {p.get('text','')}{when}")
        return

    note_path, created, n_new = _wa_tasks_write_note(
        VAULT_PATH, now, by_chat, extractions,
    )
    # Persistir promesas (status='pending') a `rag_promises`. La signal
    # `commitment_deadline` (rag_anticipate/signals/) las lee y emite push
    # WA cuando overdue / vence pronto. Silent-fail completo — si rompe,
    # el resto del flow (Inbox note + state save) sigue.
    n_promises_persisted = _wa_promises_persist(by_chat, extractions, now)
    # Update state with ALL fetched new ids (including ones from chats that
    # produced no extractions — we don't want to reprocess them).
    for chat in by_chat:
        for mid in chat["new_ids"]:
            state.setdefault("processed_ids", []).append(mid)
    state["last_run_ts"] = now.isoformat(timespec="seconds")
    _wa_tasks_save_state(state)

    # Append to log for analytics / debugging.
    _wa_log_event = {
        "ts": now.isoformat(timespec="seconds"),
        "since": since.isoformat(timespec="seconds"),
        "chats": len(by_chat),
        "items": n_new,
        "path": str(note_path.relative_to(VAULT_PATH)),
    }
    def _do_wa_log() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_wa_tasks",
                               _map_wa_tasks_row(_wa_log_event))
    _sql_write_with_retry(_do_wa_log, "wa_tasks_sql_write_failed")

    promises_tag = (
        f" · [magenta]{n_promises_persisted} promesa(s) → rag_promises[/magenta]"
        if n_promises_persisted else ""
    )
    if n_new > 0:
        rel = note_path.relative_to(VAULT_PATH)
        verb = "creado" if created else "actualizado"
        console.print(
            f"\n[green]✓[/green] {verb}: [cyan]{rel}[/cyan] · {n_new} items{promises_tag}"
        )
    elif n_promises_persisted:
        console.print(
            f"\n[green]✓[/green] sin items para Inbox{promises_tag}"
        )
    else:
        console.print("[yellow]sin items accionables extraídos[/yellow]")
