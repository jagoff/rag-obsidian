"""`rag feedback auto-harvest` + active learning + implicit feedback inference.

Phase 3 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer auto-harvest + active learning + feedback subgroup commands
desde `rag/__init__.py`.

## Qué vive acá

### Auto-harvest (Feature #1, 2026-04-23)
Harvester autónomo que labelea queries low-confidence sin feedback
explícito, usando un LLM como juez. Desbloquea el gate del fine-tune
del reranker (GC#2.C) sin depender del usuario manual.

- `_auto_harvest_snippets(paths, vault, n_chars)` — read top-5 paths,
  return (path, snippet) tuples.
- `_auto_harvest_judge(q, candidates, model)` — single-judge LLM call
  con prompt estructurado. Returns `{verdict, confidence, reason}`.
- `auto_harvest(...)` — orchestrator que combina single-judge + ensemble
  fallback (`RAG_AUTO_HARVEST_ENSEMBLE=1`).
- `feedback_auto_harvest` — Click command bajo `feedback` group.

### Active learning nudge (C.6, 2026-04-29)
Reemplaza el bash inline del plist legacy por un comando Python:

- `_count_active_learning_candidates(conn)` — count low-conf queries
  sin feedback en window.
- `_get_fine_tunning_retrieval_queue(conn, limit)` — cola para
  `/fine_tunning` UI.
- `_send_active_learning_nudge_wa/macos(n, since_days)` — push channels.
- `active_learning_nudge(...)` — orchestrator (auto/wa/macos).
- `active_learning` Click group con `nudge` sub-command.

### Implicit feedback inference (post-2026-04-29)
- `feedback_infer_implicit` — corrective_path desde behavior post-👎.
- `feedback_detect_requery` — re-queries (paráfrasis <30s) → -1.
- `feedback_classify_sessions` — reward shaping desde session outcomes.

## Patrón CLI sub-package

`@click.command(...)` standalone para cada CLI command (NO
`@feedback.command(...)` que sería circular). Registro al final del
`rag/__init__.py`:
  feedback.add_command(feedback_auto_harvest, name="auto-harvest")
  feedback.add_command(feedback_infer_implicit, name="infer-implicit")
  feedback.add_command(feedback_detect_requery, name="detect-requery")
  feedback.add_command(feedback_classify_sessions, name="classify-sessions")
  cli.add_command(active_learning)

## Lazy imports

Deps en `rag/__init__.py`:
  `_summary_client`, `HELPER_OPTIONS`, `LLM_KEEP_ALIVE`,
  `_resolve_vault_path`, `_harvest_candidates`,
  `_feedback_insert_harvested`, `_feedback_stats`,
  `_FEEDBACK_GATE_TARGET`, `_ragvec_state_conn`, `_silent_log`,
  `console`. Lazy adentro de cada función.

`rag_implicit_learning` (paquete top-level externo): lazy también
para no acoplar import-time.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import click

__all__ = [
    "_AUTO_HARVEST_JUDGE_MODEL",
    "_AUTO_HARVEST_MIN_CONF",
    "_AUTO_HARVEST_SNIPPET_CHARS",
    "_AUTO_HARVEST_ENSEMBLE",
    "_AUTO_HARVEST_ENSEMBLE_ENABLED",
    "DEFAULT_ACTIVE_LEARNING_NUDGE_THRESHOLD",
    "DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX",
    "DEFAULT_ACTIVE_LEARNING_SINCE_DAYS",
    "_ACTIVE_LEARNING_TRIVIAL_QUERIES",
    "_auto_harvest_snippets",
    "_auto_harvest_judge",
    "auto_harvest",
    "feedback_auto_harvest",
    "_count_active_learning_candidates",
    "_get_fine_tunning_retrieval_queue",
    "_send_active_learning_nudge_wa",
    "_send_active_learning_nudge_macos",
    "active_learning_nudge",
    "active_learning",
    "active_learning_nudge_cli",
    "active_learning_suggest_goldens_cli",
    "_run_suggest_goldens",
    "_send_suggest_goldens_wa",
    "feedback_infer_implicit",
    "feedback_detect_requery",
    "feedback_classify_sessions",
]


_AUTO_HARVEST_JUDGE_MODEL = os.environ.get(
    "RAG_AUTO_HARVEST_JUDGE_MODEL", "qwen2.5:7b"
).strip()
_AUTO_HARVEST_MIN_CONF = float(
    os.environ.get("RAG_AUTO_HARVEST_MIN_CONF", "0.8")
)
_AUTO_HARVEST_SNIPPET_CHARS = int(
    os.environ.get("RAG_AUTO_HARVEST_SNIPPET_CHARS", "400")
)
# Sprint 3 hook (2026-04-26): ensemble LLM-judge en auto_harvest.
_AUTO_HARVEST_ENSEMBLE = os.environ.get("RAG_AUTO_HARVEST_ENSEMBLE", "0").strip()
_AUTO_HARVEST_ENSEMBLE_ENABLED = _AUTO_HARVEST_ENSEMBLE not in ("", "0", "false", "no")


def _auto_harvest_snippets(
    paths: list[str], vault: Path | None = None, n_chars: int | None = None,
) -> list[tuple[str, str]]:
    """Read top-5 paths, return (path, snippet) tuples.

    Snippet = first non-empty line (treated as title/heading) + up to
    n_chars of body. Non-existent files get an empty snippet — the
    judge will still see the path but can skip it.
    """
    from rag import _resolve_vault_path  # noqa: PLC0415

    if vault is None:
        vault = _resolve_vault_path()
    if n_chars is None:
        n_chars = _AUTO_HARVEST_SNIPPET_CHARS
    out: list[tuple[str, str]] = []
    for p in paths[:5]:
        full = vault / p
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out.append((p, ""))
            continue
        # Strip YAML frontmatter if present.
        if text.startswith("---\n"):
            end = text.find("\n---", 4)
            if end > 0:
                text = text[end + 4:].lstrip("\n")
        lines = [l.rstrip() for l in text.splitlines() if l.strip()]
        if not lines:
            out.append((p, ""))
            continue
        title = lines[0][:120]
        body = " ".join(lines[1:])[:n_chars]
        snippet = f"{title} — {body}" if body else title
        out.append((p, snippet))
    return out


def _auto_harvest_judge(
    q: str,
    candidates: list[tuple[str, str]],
    *,
    model: str | None = None,
) -> dict | None:
    """Ask the helper LLM which candidate path best answers the query."""
    from rag import HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client  # noqa: PLC0415

    if not candidates:
        return None
    judge_model = (model or _AUTO_HARVEST_JUDGE_MODEL).strip()
    if not judge_model:
        return None
    lines = [
        f"Pregunta del usuario: \"{q}\"",
        "",
        "Candidatos (path + snippet del archivo):",
    ]
    for i, (path, snippet) in enumerate(candidates, 1):
        snippet = snippet or "[archivo vacío o ilegible]"
        lines.append(f"{i}. {path}")
        lines.append(f"   {snippet}")
    lines.extend([
        "",
        "Decidí cuál de los candidatos responde MEJOR la pregunta. "
        "Si ninguno es una respuesta razonable, devolvé verdict=\"none\".",
        "",
        "Respondé JSON estricto sin preámbulo:",
        "{\"verdict\": \"<path exacto de la lista o 'none'>\", "
        "\"confidence\": <0.0 a 1.0>, "
        "\"reason\": \"<explicación corta en una frase>\"}",
    ])
    prompt = "\n".join(lines)
    try:
        resp = _summary_client().chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_ctx": 4096, "num_predict": 200},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = resp.message.content.strip()
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    conf = data.get("confidence")
    reason = data.get("reason", "")
    if verdict is None or isinstance(verdict, str) and verdict.strip().lower() in ("none", "ninguno", "null", ""):
        normalized_verdict = None
    elif isinstance(verdict, str):
        normalized_verdict = verdict.strip()
    else:
        return None
    try:
        conf_f = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        return None
    conf_f = max(0.0, min(1.0, conf_f))
    return {
        "verdict": normalized_verdict,
        "confidence": conf_f,
        "reason": str(reason)[:200] if reason else "",
    }


def auto_harvest(
    *,
    since_days: int = 1,
    confidence_below: float = 0.3,
    min_judge_conf: float | None = None,
    limit: int = 15,
    dry_run: bool = False,
    model: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run one pass of auto-harvest."""
    from rag import (  # noqa: PLC0415
        _feedback_insert_harvested,
        _harvest_candidates,
        _resolve_vault_path,
        console,
    )

    if min_judge_conf is None:
        min_judge_conf = _AUTO_HARVEST_MIN_CONF
    stats = {
        "processed": 0,
        "judged_positive": 0,
        "judged_negative": 0,
        "skipped_low_conf": 0,
        "skipped_invalid_path": 0,
        "skipped_judge_failed": 0,
        "skipped_empty_paths": 0,
        "errors": 0,
    }
    candidates = _harvest_candidates(since_days, confidence_below, limit)
    stats["processed"] = len(candidates)
    if not candidates:
        return stats
    try:
        vault = _resolve_vault_path()
    except Exception:
        vault = None
    for c in candidates:
        paths = c.get("paths") or []
        if not paths:
            stats["skipped_empty_paths"] += 1
            continue
        snippets = _auto_harvest_snippets(paths[:5], vault=vault)
        if _AUTO_HARVEST_ENSEMBLE_ENABLED:
            try:
                from rag_implicit_learning import (  # noqa: PLC0415
                    DEFAULT_ENSEMBLE_MODELS,
                    judge_with_ensemble,
                )
                ens_result = judge_with_ensemble(
                    c["q"], snippets,
                    models=DEFAULT_ENSEMBLE_MODELS,
                    judge_fn=_auto_harvest_judge,
                )
                if ens_result is not None:
                    verdict = {
                        "verdict": ens_result["verdict"],
                        "confidence": ens_result["confidence"],
                        "reason": (
                            f"ensemble agreement={ens_result['agreement']:.2f} "
                            f"({ens_result['n_judges_voted']}/{ens_result['n_judges_total']} judges)"
                        ),
                    }
                else:
                    verdict = None
            except Exception as exc:
                if verbose:
                    console.print(
                        f"  [dim]ensemble error, fallback to single: {exc}[/dim]"
                    )
                verdict = _auto_harvest_judge(c["q"], snippets, model=model)
        else:
            verdict = _auto_harvest_judge(c["q"], snippets, model=model)
        if verdict is None:
            stats["skipped_judge_failed"] += 1
            if verbose:
                console.print(f"  [dim]judge failed: {c['q'][:60]}[/dim]")
            continue
        conf = verdict["confidence"]
        vpath = verdict["verdict"]
        if conf < min_judge_conf:
            stats["skipped_low_conf"] += 1
            if verbose:
                console.print(
                    f"  [dim]low conf ({conf:.2f}): {c['q'][:60]}[/dim]"
                )
            continue
        if vpath is None:
            if not dry_run:
                ok = _feedback_insert_harvested(
                    q=c["q"], rating=-1, paths=paths[:5],
                    original_query_id=c["id"],
                    source="auto-harvester",
                )
                if not ok:
                    stats["errors"] += 1
                    continue
            stats["judged_negative"] += 1
            if verbose:
                console.print(
                    f"  [red]−1[/red] ({conf:.2f}) {c['q'][:60]}"
                )
            continue
        if vpath not in paths:
            stats["skipped_invalid_path"] += 1
            if verbose:
                console.print(
                    f"  [yellow]hallucinated path:[/yellow] {vpath[:80]}"
                )
            continue
        if not dry_run:
            ok = _feedback_insert_harvested(
                q=c["q"], rating=1, paths=[vpath],
                original_query_id=c["id"], corrective_path=vpath,
                source="auto-harvester",
            )
            if not ok:
                stats["errors"] += 1
                continue
        stats["judged_positive"] += 1
        if verbose:
            console.print(
                f"  [green]+1[/green] ({conf:.2f}) {c['q'][:50]} → {vpath}"
            )
    return stats


@click.command("auto-harvest")
@click.option("--limit", default=20, show_default=True,
              help="Cuántos candidatos procesar como máximo.")
@click.option("--since", default=1, show_default=True,
              help="Ventana en días para candidatos (default 1 = últimas 24h).")
@click.option("--confidence-below", "confidence_below", default=0.3,
              show_default=True,
              help="Sólo queries con top_score < este valor.")
@click.option("--min-judge-conf", "min_judge_conf", default=None, type=float,
              help=f"Mínima confidence del judge para insertar (default "
                   f"{_AUTO_HARVEST_MIN_CONF}).")
@click.option("--model", "model", default=None,
              help=f"Override del judge model (default {_AUTO_HARVEST_JUDGE_MODEL}).")
@click.option("--dry-run", is_flag=True,
              help="No insertar filas — sólo mostrar lo que haría.")
@click.option("--verbose", "-v", is_flag=True,
              help="Imprimir el resultado del judge por candidato.")
@click.option("--json", "as_json", is_flag=True,
              help="Output machine-readable JSON (para launchd logs).")
def feedback_auto_harvest(
    limit: int,
    since: int,
    confidence_below: float,
    min_judge_conf: float | None,
    model: str | None,
    dry_run: bool,
    verbose: bool,
    as_json: bool,
):
    """Auto-labelear queries low-confidence usando LLM-as-judge."""
    from rag import _FEEDBACK_GATE_TARGET, _feedback_stats, console  # noqa: PLC0415

    stats = auto_harvest(
        since_days=since,
        confidence_below=confidence_below,
        min_judge_conf=min_judge_conf,
        limit=limit,
        dry_run=dry_run,
        model=model,
        verbose=verbose,
    )
    if as_json:
        click.echo(json.dumps(stats))
        return
    console.print()
    console.print("[bold]Auto-harvest summary[/bold]"
                  + ("  [yellow](dry-run)[/yellow]" if dry_run else ""))
    console.print(f"  Processed: [cyan]{stats['processed']}[/cyan]  "
                  f"| +1: [green]{stats['judged_positive']}[/green]  "
                  f"| −1: [red]{stats['judged_negative']}[/red]")
    console.print(f"  Skipped — low conf: {stats['skipped_low_conf']}, "
                  f"invalid path: {stats['skipped_invalid_path']}, "
                  f"judge failed: {stats['skipped_judge_failed']}, "
                  f"empty paths: {stats['skipped_empty_paths']}")
    if stats["errors"]:
        console.print(f"  [red]SQL errors: {stats['errors']}[/red]")
    if not dry_run and stats["judged_positive"] > 0:
        fb = _feedback_stats()
        remaining = max(0, _FEEDBACK_GATE_TARGET - fb["with_cp"])
        console.print(f"  Gate: [cyan]{fb['with_cp']}[/cyan]"
                      f" / {_FEEDBACK_GATE_TARGET}  "
                      + (f"[yellow](faltan {remaining})[/yellow]"
                         if remaining > 0
                         else "[bold green](gate open!)[/bold green]"))
    console.print()


# ─── C.6 (2026-04-29): active-learning nudge ─────────────────

DEFAULT_ACTIVE_LEARNING_NUDGE_THRESHOLD = 20
DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX = 0.20
DEFAULT_ACTIVE_LEARNING_SINCE_DAYS = 7

_ACTIVE_LEARNING_TRIVIAL_QUERIES = (
    "test", "probando", "hola", "ping",
)


def _count_active_learning_candidates(
    conn,
    *,
    since_days: int = DEFAULT_ACTIVE_LEARNING_SINCE_DAYS,
    top_score_max: float = DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX,
) -> int:
    """Count low-confidence queries last N days sin feedback explicito."""
    placeholders = ",".join(["?"] * len(_ACTIVE_LEARNING_TRIVIAL_QUERIES))
    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*) FROM rag_queries q
            WHERE q.ts > datetime('now', '-' || ? || ' days')
              AND q.top_score IS NOT NULL
              AND q.top_score < ?
              AND NOT EXISTS (
                  SELECT 1 FROM rag_feedback f
                  WHERE f.q = q.q
                    AND ABS(julianday(f.ts) - julianday(q.ts)) < 1
              )
              AND length(q.q) > 4
              AND q.q NOT IN ({placeholders})
            """,
            (since_days, top_score_max, *_ACTIVE_LEARNING_TRIVIAL_QUERIES),
        ).fetchone()
    except Exception:
        return 0
    return int(row[0] or 0) if row else 0


def _get_fine_tunning_retrieval_queue(
    conn,
    *,
    limit: int = 20,
) -> list[dict]:
    """Cola de queries de retrieval candidatas a labeling humano para `/fine_tunning`."""
    from rag import _silent_log  # noqa: PLC0415

    safe_limit = max(1, min(int(limit), 50))
    placeholders = ",".join(["?"] * len(_ACTIVE_LEARNING_TRIVIAL_QUERIES))
    sql = f"""
        SELECT q.id, q.q, q.top_score, q.ts, q.paths_json, q.session
        FROM rag_queries q
        LEFT JOIN rag_ft_panel_ratings ftr
               ON ftr.stream = 'retrieval'
              AND ftr.item_id = CAST(q.id AS TEXT)
        LEFT JOIN rag_ft_active_queue_state fts
               ON fts.stream = 'retrieval'
              AND fts.item_id = CAST(q.id AS TEXT)
        WHERE q.ts > datetime('now', '-14 days')
          AND q.top_score IS NOT NULL
          AND q.top_score < 0.15
          AND length(q.q) > 4
          AND q.q NOT IN ({placeholders})
          AND NOT EXISTS (
              SELECT 1 FROM rag_feedback f
              WHERE f.q = q.q
                AND ABS(julianday(f.ts) - julianday(q.ts)) < 1
          )
          AND ftr.id IS NULL
          AND (fts.snoozed_until_ts IS NULL
               OR fts.snoozed_until_ts < datetime('now'))
        ORDER BY q.ts DESC
        LIMIT ?
    """
    try:
        rows = conn.execute(
            sql,
            (*_ACTIVE_LEARNING_TRIVIAL_QUERIES, safe_limit),
        ).fetchall()
    except Exception as exc:
        try:
            _silent_log("fine_tunning.retrieval_queue", exc)
        except Exception:
            pass
        return []

    out: list[dict] = []
    for row in rows or ():
        try:
            qid, q_text, top_score, ts, paths_json, session = (
                row[0], row[1], row[2], row[3], row[4], row[5]
            )
        except Exception:
            continue
        paths: list[str] = []
        if paths_json:
            try:
                parsed = json.loads(paths_json)
                if isinstance(parsed, list):
                    paths = [str(p) for p in parsed if p is not None]
            except Exception:
                paths = []
        try:
            score_f = float(top_score) if top_score is not None else 0.0
        except Exception:
            score_f = 0.0
        out.append({
            "item_id": str(qid),
            "stream": "retrieval",
            "label": q_text or "",
            "top_score": score_f,
            "ts": ts or "",
            "paths": paths,
            "session_id": session if session else None,
        })
    return out


def _send_active_learning_nudge_wa(n: int, since_days: int) -> bool:
    """Manda un push WA al grupo RagNet con link a /learning."""
    try:
        from rag.integrations.whatsapp import (  # noqa: PLC0415
            WHATSAPP_BOT_JID,
            _ambient_whatsapp_send,
        )
    except Exception:
        return False

    today_iso = datetime.now().strftime("%Y-%m-%d")
    msg = (
        "*Active learning*\n\n"
        f"Tenes *{n} queries* de baja confianza estos ultimos {since_days} "
        "dias sin labels. Cada thumbs-up/down ayuda al ranker a aprender "
        "mas rapido.\n\n"
        "Labelear ahora: http://localhost:8765/learning\n"
        "(o desde terminal: `rag feedback harvest --limit 20`)\n\n"
        f"_active-learning-nudge:{today_iso}_"
    )
    return _ambient_whatsapp_send(WHATSAPP_BOT_JID, msg)


def _send_active_learning_nudge_macos(n: int, since_days: int) -> bool:
    """Fallback: osascript notification de macOS (legacy behavior)."""
    import subprocess  # noqa: PLC0415

    title = "obsidian-rag - active learning"
    subtitle = f"{n} nuevas estos ultimos {since_days} dias"
    msg = (
        f"{n} queries de baja confianza sin labels. Corre "
        "`rag feedback harvest --limit 20` o "
        "http://localhost:8765/learning"
    )
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg}" with title "{title}" '
             f'subtitle "{subtitle}"'],
            timeout=5, check=False,
        )
        return True
    except Exception:
        return False


def active_learning_nudge(
    *,
    threshold: int = DEFAULT_ACTIVE_LEARNING_NUDGE_THRESHOLD,
    since_days: int = DEFAULT_ACTIVE_LEARNING_SINCE_DAYS,
    top_score_max: float = DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX,
    channel: str = "auto",
    dry_run: bool = False,
) -> dict:
    """Run one pass of the active-learning nudge."""
    from rag import _ragvec_state_conn  # noqa: PLC0415

    result = {
        "n_candidates": 0,
        "fired": False,
        "channel_used": None,
        "dry_run": dry_run,
        "threshold": threshold,
        "since_days": since_days,
    }
    try:
        with _ragvec_state_conn() as conn:
            n = _count_active_learning_candidates(
                conn, since_days=since_days, top_score_max=top_score_max,
            )
    except Exception:
        return result
    result["n_candidates"] = n

    if n < threshold:
        return result

    if dry_run:
        result["channel_used"] = "(dry-run)"
        return result

    if channel == "wa":
        ok = _send_active_learning_nudge_wa(n, since_days)
        result["fired"] = ok
        result["channel_used"] = "wa" if ok else "wa-failed"
        return result
    if channel == "macos":
        ok = _send_active_learning_nudge_macos(n, since_days)
        result["fired"] = ok
        result["channel_used"] = "macos"
        return result
    if _send_active_learning_nudge_wa(n, since_days):
        result["fired"] = True
        result["channel_used"] = "wa"
    else:
        ok = _send_active_learning_nudge_macos(n, since_days)
        result["fired"] = ok
        result["channel_used"] = "macos-fallback" if ok else "all-failed"
    return result


@click.group("active-learning")
def active_learning() -> None:
    """Active learning: nudge + (futuro) batch labeling via WA quick reply."""


@active_learning.command("nudge")
@click.option("--threshold", default=DEFAULT_ACTIVE_LEARNING_NUDGE_THRESHOLD,
              show_default=True, type=int,
              help="Minimo de candidates para disparar el push.")
@click.option("--since-days", "since_days",
              default=DEFAULT_ACTIVE_LEARNING_SINCE_DAYS,
              show_default=True, type=int,
              help="Ventana en dias para contar candidates.")
@click.option("--top-score-max", "top_score_max",
              default=DEFAULT_ACTIVE_LEARNING_TOP_SCORE_MAX,
              show_default=True, type=float,
              help="Maximo top_score para clasificar como low-conf.")
@click.option("--channel", type=click.Choice(["auto", "wa", "macos"]),
              default="auto", show_default=True,
              help="Canal del push. 'auto' = WA con macOS fallback.")
@click.option("--dry-run", is_flag=True,
              help="Contar candidates sin disparar push.")
@click.option("--json", "as_json", is_flag=True,
              help="Output JSON (para launchd logs).")
def active_learning_nudge_cli(
    threshold: int, since_days: int, top_score_max: float,
    channel: str, dry_run: bool, as_json: bool,
) -> None:
    """Recordatorio de labelear queries low-confidence."""
    from rag import console  # noqa: PLC0415

    result = active_learning_nudge(
        threshold=threshold,
        since_days=since_days,
        top_score_max=top_score_max,
        channel=channel,
        dry_run=dry_run,
    )
    if as_json:
        click.echo(json.dumps(result))
        return
    console.print()
    console.print("[bold]rag active-learning nudge[/bold]")
    console.print(f"  Candidates ultimos {since_days}d: "
                  f"[cyan]{result['n_candidates']}[/cyan]  "
                  f"(threshold: {threshold})")
    if result["fired"]:
        console.print(f"  [green]Push fired[/green] -> "
                      f"channel={result['channel_used']}")
    elif result["n_candidates"] < threshold:
        console.print("  [dim]Below threshold, no push.[/dim]")
    else:
        console.print(f"  [yellow]Push failed[/yellow] -> "
                      f"channel={result['channel_used']}")
    console.print()


# ─── suggest-goldens (active-learning bootstrap, 2026-05-09) ─────────────


def _run_suggest_goldens(days: int, limit: int) -> dict:
    """Invoca scripts/suggest_goldens.py --json y devuelve el dict parseado.

    Subprocess en lugar de import directo porque el script es del path
    `scripts/` (no parte del paquete `rag/`); evita acoplar el CLI a su
    layout interno y mantiene un solo writer de la lógica de filtrado.
    """
    import subprocess  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    repo = Path(__file__).resolve().parent.parent.parent
    script = repo / "scripts" / "suggest_goldens.py"
    venv_py = repo / ".venv" / "bin" / "python"
    py_bin = str(venv_py) if venv_py.is_file() else "python3"
    if not script.is_file():
        return {"error": "script-missing", "candidates": []}
    try:
        out = subprocess.run(
            [py_bin, str(script), "--days", str(days), "--limit", str(limit),
             "--json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except Exception as exc:
        try:
            _silent_log("active_learning.suggest_goldens.subprocess", exc)
        except Exception:
            pass
        return {"error": "subprocess-failed", "candidates": []}
    if out.returncode != 0:
        return {"error": f"exit-{out.returncode}", "candidates": [],
                "stderr": (out.stderr or "")[:200]}
    try:
        return json.loads(out.stdout)
    except (TypeError, ValueError) as exc:
        try:
            _silent_log("active_learning.suggest_goldens.parse", exc)
        except Exception:
            pass
        return {"error": "parse-failed", "candidates": []}


def _send_suggest_goldens_wa(payload: dict, days: int) -> bool:
    """Push WA al user con count + top 3 candidates como preview."""
    try:
        from rag.integrations.whatsapp import (  # noqa: PLC0415
            WHATSAPP_BOT_JID,
            _ambient_whatsapp_send,
        )
    except Exception:
        return False

    candidates = payload.get("candidates") or []
    n_keep = int(payload.get("n_keep") or len(candidates))
    if not candidates:
        return False

    today_iso = datetime.now().strftime("%Y-%m-%d")
    preview_lines = []
    for c in candidates[:3]:
        q = (c.get("q") or "")[:60]
        path = c.get("expected") or ""
        preview_lines.append(f"  • _{q}_  →  `{path}`")

    msg = (
        "*Active learning · golden set*\n\n"
        f"Tenes *{n_keep} queries* con thumbs-up estos ultimos {days} dias "
        "que podrian sumar al golden set (eval baseline).\n\n"
        + "\n".join(preview_lines)
        + "\n\nReview + paste:\n"
        "`.venv/bin/python scripts/suggest_goldens.py "
        f"--days {days} --limit 10`\n\n"
        f"_active-learning-suggest-goldens:{today_iso}_"
    )
    return _ambient_whatsapp_send(WHATSAPP_BOT_JID, msg)


@active_learning.command("suggest-goldens")
@click.option("--days", default=7, show_default=True, type=int,
              help="Ventana en dias sobre rag_feedback rating=+1.")
@click.option("--limit", default=10, show_default=True, type=int,
              help="Maximo de candidates a emitir.")
@click.option("--threshold", default=3, show_default=True, type=int,
              help="Minimo de candidates accionables para disparar push.")
@click.option("--channel", type=click.Choice(["auto", "wa", "stdout"]),
              default="auto", show_default=True,
              help="Canal del nudge. 'stdout' = solo print (testing).")
@click.option("--dry-run", is_flag=True,
              help="Contar candidates sin disparar push.")
@click.option("--json", "as_json", is_flag=True,
              help="Output JSON (para launchd logs).")
def active_learning_suggest_goldens_cli(
    days: int, limit: int, threshold: int,
    channel: str, dry_run: bool, as_json: bool,
) -> None:
    """Sugerir entries para queries.yaml desde feedback +1 reciente."""
    from rag import console  # noqa: PLC0415

    payload = _run_suggest_goldens(days=days, limit=limit)
    n_keep = int(payload.get("n_keep") or 0)
    candidates = payload.get("candidates") or []
    fired = False
    channel_used: str | None = None

    if n_keep >= threshold and not dry_run and channel != "stdout":
        if channel in ("auto", "wa"):
            if _send_suggest_goldens_wa(payload, days=days):
                fired = True
                channel_used = "wa"
            elif channel == "auto":
                channel_used = "wa-failed"

    result = {
        "days": days,
        "limit": limit,
        "threshold": threshold,
        "n_keep": n_keep,
        "n_candidates_raw": int(payload.get("n_candidates_raw") or 0),
        "n_drop_dup_q": int(payload.get("n_drop_dup_q") or 0),
        "n_drop_path_missing": int(payload.get("n_drop_path_missing") or 0),
        "fired": fired,
        "channel_used": channel_used,
        "dry_run": dry_run,
        "error": payload.get("error"),
    }

    if as_json:
        click.echo(json.dumps(result))
        return

    console.print()
    console.print("[bold]rag active-learning suggest-goldens[/bold]")
    console.print(f"  Window: ultimos {days}d  ·  threshold: {threshold}")
    console.print(f"  Candidates accionables: [cyan]{n_keep}[/cyan] "
                  f"(raw: {result['n_candidates_raw']}, "
                  f"dup_q: {result['n_drop_dup_q']}, "
                  f"path_missing: {result['n_drop_path_missing']})")
    if result.get("error"):
        console.print(f"  [yellow]error: {result['error']}[/yellow]")
        return
    if not candidates:
        console.print("  [dim]Sin candidates nuevos.[/dim]")
        return
    if fired:
        console.print(f"  [green]Push fired[/green] -> {channel_used}")
    elif channel == "stdout" or n_keep < threshold:
        console.print(f"  [dim]Below threshold "
                      f"({n_keep} < {threshold}) o channel=stdout.[/dim]")
    elif dry_run:
        console.print("  [dim]dry-run (no push).[/dim]")
    else:
        console.print(f"  [yellow]Push failed[/yellow] -> {channel_used}")
    console.print()
    console.print("[dim]Top candidates:[/dim]")
    for c in candidates[:5]:
        q = (c.get("q") or "")[:80]
        path = c.get("expected") or ""
        console.print(f"  • {q} → [cyan]{path}[/cyan]")
    console.print()


# ─── implicit feedback inference ─────────────────


@click.command("infer-implicit")
@click.option("--window-seconds", "window_seconds", default=600,
              show_default=True,
              help="Cuántos segundos después de un 👎 considerar como reacción "
                   "del user.")
@click.option("--dry-run", is_flag=True,
              help="No persistir cambios — solo reportar qué se inferiría.")
@click.option("--json", "as_json", is_flag=True,
              help="Emitir summary como JSON (para launchd / scripts).")
def feedback_infer_implicit(
    window_seconds: int, dry_run: bool, as_json: bool
) -> None:
    """Inferir corrective_path implícito desde behavior post-👎."""
    from rag import _ragvec_state_conn, console  # noqa: PLC0415
    from rag_implicit_learning import infer_corrective_paths_from_behavior  # noqa: PLC0415

    with _ragvec_state_conn() as conn:
        result = infer_corrective_paths_from_behavior(
            conn, window_seconds=window_seconds, dry_run=dry_run,
        )

    if as_json:
        summary = {k: v for k, v in result.items() if k != "updates"}
        summary["n_updates_payload_redacted"] = len(result.get("updates", []))
        click.echo(json.dumps(summary))
        return

    console.print()
    console.print("[bold]rag feedback infer-implicit[/bold]")
    console.print(f"  Window: {result['window_seconds']}s · "
                  f"dry-run={'yes' if result['dry_run'] else 'no'}")
    console.print()
    console.print(f"  Candidatos (rating=-1):  {result['n_candidates']}")
    console.print(f"  [bold green]Inferidos:               "
                  f"{result['n_inferred']}[/bold green]")
    via_paraphrase = result.get("n_inferred_via_paraphrase", 0)
    if via_paraphrase:
        console.print(f"    [dim]· vía opens:        "
                      f"{result['n_inferred'] - via_paraphrase}[/dim]")
        console.print(f"    [dim]· vía paráfrasis:   "
                      f"{via_paraphrase}[/dim]")
    console.print(f"  Skip — ya tienen corrective: "
                  f"{result['n_skip_already_corrective']}")
    console.print(f"  Skip — sin session_id:       "
                  f"{result['n_skip_no_session']}")
    console.print(f"  Skip — sin paths_json:       {result['n_skip_no_paths']}")
    console.print(f"  Skip — sin opens ni paráfrasis: "
                  f"{result['n_skip_no_open']}")
    console.print(f"  Skip — abrió el top path:    "
                  f"{result['n_skip_opened_top']}")
    console.print()

    if result["updates"]:
        from rich.table import Table  # noqa: PLC0415
        tbl = Table(show_header=True, header_style="bold cyan")
        tbl.add_column("fb_id", justify="right")
        tbl.add_column("ts", style="dim")
        tbl.add_column("query", overflow="fold", max_width=40)
        tbl.add_column("top_path → corrective", overflow="fold", max_width=60)
        tbl.add_column("in top-k?")
        for u in result["updates"][:20]:
            tbl.add_row(
                str(u["feedback_id"]),
                u["ts"],
                u["query"] or "—",
                f"[red]{u['top_path']}[/red] → [green]{u['corrective_path']}[/green]",
                "✓" if u["in_top_k"] else "[yellow]ext[/yellow]",
            )
        console.print(tbl)
        if len(result["updates"]) > 20:
            console.print(
                f"\n  [dim]... y {len(result['updates']) - 20} más "
                f"(usá --json para todos)[/dim]\n"
            )

    if dry_run and result["n_inferred"] > 0:
        console.print(
            "\n[yellow]💡 Re-corré sin --dry-run para persistir los "
            "corrective_paths.[/yellow]"
        )


@click.command("detect-requery")
@click.option("--window-seconds", "window_seconds", default=30,
              show_default=True,
              help="Gap máximo entre dos queries para considerar re-query.")
@click.option("--similarity-threshold", "similarity_threshold", default=0.5,
              show_default=True,
              help="Threshold de SequenceMatcher para detectar paráfrasis.")
@click.option("--dry-run", is_flag=True,
              help="No persistir cambios — solo reportar detecciones.")
@click.option("--json", "as_json", is_flag=True,
              help="Emitir summary como JSON (para launchd).")
def feedback_detect_requery(
    window_seconds: int,
    similarity_threshold: float,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Detectar re-queries (paráfrasis <30s) como signal negativa implícita."""
    from rag import _ragvec_state_conn, console  # noqa: PLC0415
    from rag_implicit_learning import detect_requery_loss_signal  # noqa: PLC0415

    with _ragvec_state_conn() as conn:
        result = detect_requery_loss_signal(
            conn,
            window_seconds=window_seconds,
            similarity_threshold=similarity_threshold,
            dry_run=dry_run,
        )

    if as_json:
        summary = {k: v for k, v in result.items() if k != "detections"}
        summary["n_detections_payload_redacted"] = len(result.get("detections", []))
        click.echo(json.dumps(summary))
        return

    console.print()
    console.print("[bold]rag feedback detect-requery[/bold]")
    console.print(f"  Window: {result['window_seconds']}s · "
                  f"similarity ≥ {result['similarity_threshold']} · "
                  f"dry-run={'yes' if result['dry_run'] else 'no'}")
    console.print()
    console.print(f"  Turns examinados:      {result['n_turns_examined']}")
    console.print(f"  Pares examinados:      {result['n_pairs_examined']}")
    console.print(f"  [bold green]Re-queries detectadas: "
                  f"{result['n_paraphrases_detected']}[/bold green]")
    console.print(f"  Insertados:            {result['n_inserted']}")
    console.print(f"  Skip — fuera de ventana:  {result['n_skip_outside_window']}")
    console.print(f"  Skip — ya marcado:        {result['n_skip_already_marked']}")
    console.print()


@click.command("classify-sessions")
@click.option("--days", default=7, show_default=True,
              help="Ventana de sessions a clasificar.")
@click.option("--min-confidence", "min_confidence", default=0.7, type=float,
              show_default=True,
              help="Confidence mínima para shapeear reward.")
@click.option("--dry-run", is_flag=True, help="No persistir cambios.")
@click.option("--json", "as_json", is_flag=True, help="Emitir summary como JSON.")
def feedback_classify_sessions(
    days: int, min_confidence: float, dry_run: bool, as_json: bool,
) -> None:
    """Clasificar sessions recientes y aplicar reward shaping a los turns."""
    from rag import _ragvec_state_conn, console  # noqa: PLC0415
    from rag_implicit_learning import (  # noqa: PLC0415
        apply_reward_from_session_outcomes,
        classify_recent_sessions,
        session_outcome_summary,
    )

    with _ragvec_state_conn() as conn:
        analyses = classify_recent_sessions(conn, days=days)
        summary = session_outcome_summary(analyses)
        reward_result = apply_reward_from_session_outcomes(
            conn, days=days, min_confidence=min_confidence, dry_run=dry_run,
        )

    if as_json:
        out = {
            "classification": summary,
            "reward_shaping": {
                k: v for k, v in reward_result.items() if k != "updates"
            },
            "n_updates_payload_redacted": len(reward_result.get("updates", [])),
        }
        click.echo(json.dumps(out))
        return

    console.print()
    console.print(f"[bold]rag feedback classify-sessions[/bold]  "
                  f"(días={days}, min_conf={min_confidence}, "
                  f"dry-run={'yes' if dry_run else 'no'})")
    console.print()
    console.print(f"  Sessions analizadas:    {summary['n_sessions']}")
    console.print(f"    [green]win:[/green]                  "
                  f"{summary['by_outcome']['win']}")
    console.print(f"    [red]loss:[/red]                 "
                  f"{summary['by_outcome']['loss']}")
    console.print(f"    [dim]partial (ambiguo):[/dim]  "
                  f"{summary['by_outcome']['partial']}")
    console.print(f"    [dim]abandon (1 turn):[/dim]   "
                  f"{summary['by_outcome']['abandon']}")
    console.print(f"  Avg confidence:         {summary['avg_confidence']}")
    console.print()
    console.print("[bold]Reward shaping[/bold]")
    console.print(f"  Sessions usadas:        "
                  f"{reward_result['n_sessions_used_for_reward']}")
    console.print(f"  Sessions skip ambiguos: "
                  f"{reward_result['n_skip_ambiguous_outcome']}")
    console.print(f"  Sessions skip lowconf:  "
                  f"{reward_result['n_skip_low_confidence']}")
    console.print(f"  Turns evaluados:        {reward_result['n_turns_total']}")
    console.print(f"  Turns skip explícito:   "
                  f"{reward_result['n_turns_skip_explicit']}")
    console.print(f"  Turns skip ya shaped:   "
                  f"{reward_result['n_turns_skip_already_shaped']}")
    console.print(f"  [bold green]Insertados +1 (win):    "
                  f"{reward_result['n_turns_inserted_pos']}[/bold green]")
    console.print(f"  [bold red]Insertados -1 (loss):   "
                  f"{reward_result['n_turns_inserted_neg']}[/bold red]")
    console.print()
