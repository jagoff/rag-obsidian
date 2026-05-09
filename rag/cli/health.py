"""`rag health` — unified dashboard de salud del sistema.

Phase 3 de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el CLI command `rag health` + sus 5 helpers desde
`rag/__init__.py` a `rag/cli/health.py` (sub-package CLI ya
existente con `vault.py` + `model.py` como precedente).

## Snapshot de 5 segundos

`rag health` agrega signal que estaba spread en:
- `rag stats`     (modelos + index counts)
- `rag log`       (query-level latency)
- `rag feedback status`
- `rag dashboard` (analytics per day)
- `rag eval`      (quality metrics)

Sin reemplazarlos — cada uno tiene deeper features. Pensado para
correr ANTES de cualquier debug: "¿qué está pasando ahora?"

## Secciones

1. **Corpus snapshot**     — chunks, sources breakdown, DB size.
2. **Recent queries**      — count, latency P50/P95, cache hit rate.
3. **Feedback + gate**     — stats, progreso hacia fine-tune gate.
4. **Score calibration**   — fuentes entrenadas, knots, last update.
5. **Training signal**     — CTR + orphan opens (flywheel ranker-vivo).
6. **Features opt-in**     — cuáles de las 6 features están active.

Output: Rich tables agrupadas; `--as-json` para scripts.

## Patrón de registro CLI

`@click.command(...)` (NO `@cli.command(...)`). El grupo `cli` se
bindea después en `rag/__init__.py`:
  from rag.cli.health import health_cli
  cli.add_command(health_cli)
Mismo patrón que `rag/cli/vault.py` y `rag/cli/model.py`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import click

__all__ = [
    "_health_corpus_snapshot",
    "_health_query_stats",
    "_health_calibration_status",
    "_health_training_signal",
    "_health_features_opt_in",
    "health_cli",
]


def _health_corpus_snapshot() -> dict:
    """Fetch corpus stats from the main vec collection."""
    from rag import _ragvec_state_conn, get_db  # noqa: PLC0415

    out = {
        "total_chunks": 0,
        "sources": {},
        "last_index_ts": None,
        "error": None,
    }
    try:
        col = get_db()
        c = col.count() if hasattr(col, "count") else 0
        out["total_chunks"] = int(c)
    except Exception as exc:
        out["error"] = repr(exc)
    # Per-source breakdown from a cheap SQL query.
    try:
        with _ragvec_state_conn():
            # rag_queries doesn't have source — we infer from paths_json.
            # Prefer using the collection's meta if available.
            pass
    except Exception:
        pass
    return out


def _health_query_stats(since_hours: int = 24) -> dict:
    """Aggregate recent query metrics from rag_queries."""
    from rag import _ragvec_state_conn  # noqa: PLC0415

    out = {
        "count": 0,
        "avg_retrieve_ms": 0.0,
        "avg_gen_ms": 0.0,
        "p50_total_ms": 0.0,
        "p95_total_ms": 0.0,
        "cache_hits": 0,
        "by_cmd": {},
    }
    try:
        with _ragvec_state_conn() as conn:
            since_sql = f"datetime('now', '-{int(since_hours)} hours')"
            rows = conn.execute(
                f"SELECT t_retrieve, t_gen, cmd, "
                f"       json_extract(extra_json, '$.cache_hit') AS ch "
                f"FROM rag_queries WHERE ts > {since_sql}"
            ).fetchall()
    except Exception:
        rows = []
    if not rows:
        return out
    out["count"] = len(rows)
    retrieve_ms = [float(r[0]) for r in rows if r[0] is not None]
    gen_ms = [float(r[1]) for r in rows if r[1] is not None]
    totals = [
        (float(r[0] or 0) + float(r[1] or 0))
        for r in rows if r[0] is not None or r[1] is not None
    ]
    if retrieve_ms:
        out["avg_retrieve_ms"] = sum(retrieve_ms) / len(retrieve_ms)
    if gen_ms:
        out["avg_gen_ms"] = sum(gen_ms) / len(gen_ms)
    if totals:
        totals_sorted = sorted(totals)
        p50_idx = int(len(totals_sorted) * 0.50)
        p95_idx = int(len(totals_sorted) * 0.95)
        out["p50_total_ms"] = totals_sorted[min(p50_idx, len(totals_sorted) - 1)]
        out["p95_total_ms"] = totals_sorted[min(p95_idx, len(totals_sorted) - 1)]
    # Cache hit rate.
    out["cache_hits"] = sum(1 for r in rows if r[3])
    # Breakdown by cmd.
    by_cmd: dict[str, int] = {}
    for r in rows:
        cmd = r[2] or "unknown"
        by_cmd[cmd] = by_cmd.get(cmd, 0) + 1
    out["by_cmd"] = by_cmd
    return out


def _health_calibration_status() -> dict:
    """Summarize rag_score_calibration state."""
    from rag import _ragvec_state_conn  # noqa: PLC0415

    out = {"sources_trained": 0, "sources": [], "error": None}
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, n_pos, n_neg, trained_at, "
                "       length(raw_knots_json) "
                "FROM rag_score_calibration ORDER BY trained_at DESC"
            ).fetchall()
        out["sources_trained"] = len(rows)
        for source, n_pos, n_neg, trained_at, knot_bytes in rows:
            out["sources"].append({
                "source": source,
                "n_pos": int(n_pos), "n_neg": int(n_neg),
                "trained_at": trained_at,
                "model_bytes": int(knot_bytes or 0),
            })
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _health_training_signal(since_days: int = 7) -> dict:
    """Signal flywheel para el ranker-vivo fine-tune (2026-04-23).

    Métricas que responden "cuánto dato de training estoy cosechando"
    + "cuánto falta para habilitar el próximo fine-tune":

      • impressions / opens / CTR: el denominador del entrenamiento.
        CTR objetivo para desbloquear un 3er fine-tune con signal real
        ≥ 1% sostenido durante el período (ver docs/finetune-run-*.md).
      • orphan_opens: opens sin `original_query_id` en extra_json.
        `rag behavior backfill` los rescata — visible acá para que
        el operador sepa cuándo correr el comando.
      • feedback_with_cp / feedback_gate: progreso hacia
        ``_FEEDBACK_GATE_TARGET`` (20 corrective_paths). Duplicado
        con la sección Feedback pero acá reportado como window-scoped
        para trending.

    Fail-closed: errores en SQL devuelven zeros + ``error`` key,
    nunca raise.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import (  # noqa: PLC0415
        _FEEDBACK_GATE_TARGET,
        _log_sql_state_error,
        _ragvec_state_conn,
    )

    out = {
        "window_days": int(since_days),
        "impressions": 0,
        "opens": 0,
        "ctr_pct": 0.0,
        "orphan_opens": 0,
        "backfilled_opens": 0,
        "feedback_with_cp": 0,
        "feedback_gate_target": _FEEDBACK_GATE_TARGET,
    }
    since_clause = f"-{int(since_days)} days"

    def _count(conn, key: str, sql: str, params: tuple = ()) -> None:
        """Run one COUNT + populate out[key].

        Degrades gracefully:
          - "no such table" (test env without schema) → silent, key stays 0.
            Loggearlo espameaba `health_training_signal_failed` cada vez que
            un test + integration call invocaba `rag health` contra una DB
            tmp sin todas las tablas. No es un bug en prod.
          - Otros errores SQL → loguear + key stays 0, pero NO raise (otras
            keys igual se computan en llamadas subsecuentes al helper).
        """
        try:
            row = conn.execute(sql, params).fetchone()
            out[key] = int(row[0] or 0) if row else 0
        except _sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "no such table" in msg:
                # Benign in test environments; don't pollute error log.
                return
            _log_sql_state_error(
                "health_training_signal_partial",
                err=f"{key}: {exc!r}",
            )
        except Exception as exc:
            _log_sql_state_error(
                "health_training_signal_partial",
                err=f"{key}: {exc!r}",
            )

    try:
        with _ragvec_state_conn() as conn:
            _count(conn, "impressions",
                   "SELECT COUNT(*) FROM rag_behavior"
                   " WHERE event='impression' AND ts > datetime('now', ?)",
                   (since_clause,))
            _count(conn, "opens",
                   "SELECT COUNT(*) FROM rag_behavior"
                   " WHERE event='open' AND ts > datetime('now', ?)",
                   (since_clause,))
            # Orphan opens are all-time (not window-scoped) porque el
            # backfill command puede recoger opens viejos tambien.
            _count(conn, "orphan_opens",
                   "SELECT COUNT(*) FROM rag_behavior"
                   " WHERE event='open'"
                   " AND (extra_json IS NULL OR"
                   "      json_extract(extra_json, '$.original_query_id') IS NULL)")
            _count(conn, "backfilled_opens",
                   "SELECT COUNT(*) FROM rag_behavior"
                   " WHERE event='open'"
                   " AND json_extract(extra_json, '$.backfilled') = 1")
            # Feedback rows con corrective_path dentro del período.
            _count(conn, "feedback_with_cp",
                   "SELECT COUNT(*) FROM rag_feedback"
                   " WHERE json_extract(extra_json, '$.corrective_path') IS NOT NULL"
                   " AND json_extract(extra_json, '$.corrective_path') != ''"
                   " AND ts > datetime('now', ?)",
                   (since_clause,))
        if out["impressions"] > 0:
            out["ctr_pct"] = round(
                100.0 * out["opens"] / out["impressions"], 3,
            )
    except _sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "unable to open" in msg:
            # Test env / DB missing — silent degrade.
            return out
        out["error"] = repr(exc)
        _log_sql_state_error("health_training_signal_failed", err=repr(exc))
    except Exception as exc:
        out["error"] = repr(exc)
        _log_sql_state_error("health_training_signal_failed", err=repr(exc))
    return out


def _health_features_opt_in() -> dict:
    """Report the on/off state of the 6 feature flags."""
    features = {
        "Feature #1 auto-harvest": {
            "plist": "com.fer.obsidian-rag-auto-harvest",
            "status": "always-on (nightly plist)",
        },
        "Feature #2 score calibration": {
            "env": "RAG_SCORE_CALIBRATION",
            "enabled": bool(
                os.environ.get("RAG_SCORE_CALIBRATION", "").strip().lower()
                in ("1", "true", "yes")
            ),
        },
        "Feature #3 LLM intent": {
            "env": "RAG_LLM_INTENT",
            "enabled": bool(
                os.environ.get("RAG_LLM_INTENT", "").strip().lower()
                in ("1", "true", "yes")
            ),
        },
        "Feature #4 agent loop upgrade": {
            "status": "always-on (prompt + streak detector)",
        },
        "Feature #5 MMR diversity": {
            "env": "RAG_MMR_DIVERSITY",
            "enabled": bool(
                os.environ.get("RAG_MMR_DIVERSITY", "").strip().lower()
                in ("1", "true", "yes")
            ),
        },
        "Feature #6 Personalized PageRank": {
            "env": "RAG_PPR_TOPIC",
            "enabled": bool(
                os.environ.get("RAG_PPR_TOPIC", "").strip().lower()
                in ("1", "true", "yes")
            ),
        },
    }
    return features


@click.command("health")
@click.option("--since", "since_hours", default=24, show_default=True,
              help="Ventana en horas para stats de queries recientes.")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON machine-readable.")
def health_cli(since_hours: int, as_json: bool):
    """Dashboard unificado de salud del sistema.

    Snapshot de 5 segundos pensado para correr antes de cualquier debug:
      - Corpus size + breakdown
      - Queries recientes + latencia P50/P95 + cache hit rate
      - Feedback stats + progreso hacia gate de fine-tune
      - Estado de calibración per-source
      - Qué features opcionales tenés activas

    Para deep-dive usar: `rag stats` (modelos), `rag log` (queries),
    `rag feedback status` (feedback), `rag dashboard` (analytics).
    """
    from rag import _FEEDBACK_GATE_TARGET, _feedback_stats, console  # noqa: PLC0415

    corpus = _health_corpus_snapshot()
    queries = _health_query_stats(since_hours)
    feedback = _feedback_stats()
    calibration = _health_calibration_status()
    features = _health_features_opt_in()
    # Training signal flywheel — CTR + orphan opens + feedback gate.
    # Ventana de 7 días por default; `since_hours` ≤ 48 tiene poco signal
    # así que el widget muestra siempre 7d (orthogonal a since_hours que
    # aplica a la sección Queries).
    training = _health_training_signal(since_days=7)

    payload = {
        "corpus": corpus,
        "queries": {**queries, "since_hours": since_hours},
        "feedback": feedback,
        "training_signal": training,
        "calibration": calibration,
        "features": features,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }

    if as_json:
        click.echo(json.dumps(payload, default=str))
        return

    console.print()
    console.print("[bold cyan]RAG Health Dashboard[/bold cyan]  "
                  f"[dim]({payload['ts']})[/dim]")
    console.print()

    # ── Corpus
    console.print(
        f"[bold]Corpus[/bold]: [cyan]{corpus['total_chunks']:,}[/cyan] chunks  "
        + (f"[red]{corpus['error']}[/red]" if corpus.get("error") else "")
    )

    # ── Queries recent
    q = queries
    if q["count"] > 0:
        hit_pct = (q["cache_hits"] / q["count"] * 100.0) if q["count"] else 0.0
        console.print(
            f"[bold]Queries[/bold] ({since_hours}h): [cyan]{q['count']}[/cyan]  "
            f"|  avg retrieve [cyan]{q['avg_retrieve_ms']:.0f}ms[/cyan]  "
            f"gen [cyan]{q['avg_gen_ms']:.0f}ms[/cyan]  "
            f"|  P50 {q['p50_total_ms']:.0f}ms · P95 {q['p95_total_ms']:.0f}ms  "
            f"|  cache hits [green]{q['cache_hits']}[/green] ({hit_pct:.0f}%)"
        )
        if q["by_cmd"]:
            by_cmd_str = ", ".join(
                f"{c}={n}" for c, n in sorted(
                    q["by_cmd"].items(), key=lambda x: -x[1]
                )[:5]
            )
            console.print(f"  [dim]por cmd: {by_cmd_str}[/dim]")
    else:
        console.print(f"[bold]Queries[/bold] ({since_hours}h): [dim]ninguna[/dim]")

    # ── Feedback
    fb = feedback
    gate_target = _FEEDBACK_GATE_TARGET
    gate_remaining = max(0, gate_target - fb["with_cp"])
    gate_str = (
        "[green](gate open!)[/green]" if gate_remaining == 0
        else f"[yellow]faltan {gate_remaining}[/yellow]"
    )
    console.print(
        f"[bold]Feedback[/bold]: [cyan]{fb['total']}[/cyan] rows  "
        f"| +1 [green]{fb['pos']}[/green]  −1 [red]{fb['neg']}[/red]  "
        f"| corrective [bold green]{fb['with_cp']}[/bold green]/{gate_target}  "
        f"{gate_str}"
    )

    # ── Training signal flywheel
    ts_stat = training
    if "error" not in ts_stat:
        ctr = ts_stat["ctr_pct"]
        # CTR color ladder: <0.5% rojo, 0.5-1% amarillo, ≥1% verde.
        if ctr >= 1.0:
            ctr_color = "green"
        elif ctr >= 0.5:
            ctr_color = "yellow"
        else:
            ctr_color = "red"
        orphan_str = (
            f" | orphans [yellow]{ts_stat['orphan_opens']}[/yellow]"
            f" (run [cyan]rag behavior backfill[/cyan])"
            if ts_stat["orphan_opens"] > 0
            else f" | orphans [green]{ts_stat['orphan_opens']}[/green]"
        )
        console.print(
            f"[bold]Training signal[/bold] ({ts_stat['window_days']}d): "
            f"impressions [cyan]{ts_stat['impressions']:,}[/cyan] / "
            f"opens [cyan]{ts_stat['opens']}[/cyan] / "
            f"CTR [bold {ctr_color}]{ctr:.2f}%[/bold {ctr_color}]"
            f"{orphan_str}"
        )
        if ts_stat["backfilled_opens"] > 0:
            console.print(
                f"  [dim]backfilled: {ts_stat['backfilled_opens']} opens "
                "linked via `rag behavior backfill`[/dim]"
            )
    else:
        console.print(
            "[bold]Training signal[/bold]: "
            f"[red]error: {ts_stat['error']}[/red]"
        )

    # ── Calibration
    cal = calibration
    if cal["sources_trained"] > 0:
        src_list = ", ".join(
            f"{s['source']} ({s['n_pos']}+/{s['n_neg']}−)"
            for s in cal["sources"]
        )
        console.print(
            f"[bold]Calibration[/bold]: [cyan]{cal['sources_trained']}[/cyan] "
            f"sources entrenadas — {src_list}"
        )
    else:
        console.print(
            "[bold]Calibration[/bold]: [dim]ninguna source entrenada "
            "(correr `rag calibrate`)[/dim]"
        )

    # ── Features opt-in
    console.print()
    console.print("[bold]Features opt-in[/bold]")
    for name, info in features.items():
        if "enabled" in info:
            if info["enabled"]:
                state = "[bold green]ON[/bold green]"
            else:
                state = "[dim]off[/dim]"
            env_note = f" (env: [cyan]{info['env']}[/cyan])"
            console.print(f"  {state}  {name}{env_note}")
        else:
            state = info.get("status", "")
            console.print(f"  [green]●[/green]  {name} [dim]— {state}[/dim]")

    console.print()
