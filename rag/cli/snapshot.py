"""`rag snapshot` — portable backup/restore de artefactos ML + feedback.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 365).

Bundle de los artefactos ML + feedback data en un JSON portable:
ranker.json + rag_score_calibration + rag_learned_paraphrases +
rag_feedback. Use cases:
  - Backup pre-experimento (activar feature, probar, rollback).
  - Migrar setup entre máquinas.
  - Compartir calibration + feedback entre vaults del mismo user.
  - Disaster recovery cuando ranker.json o la DB se corrompen.

## API

- `_snapshot_create()` → dict serializable.
- `_snapshot_restore(snap, *, apply_*)` → stats dict.
- `_SNAPSHOT_VERSION` — bump cuando cambie el schema del JSON.
- `snapshot_group` (Click group) con sub-commands `create`, `restore`,
  `list` — registrado al final de `rag/__init__.py` via
  `cli.add_command(snapshot_group, name="snapshot")`.

## Lazy imports

`VAULT_PATH`, `RANKER_CONFIG_PATH`, `_ragvec_state_conn`, `console`
viven en `rag/__init__.py`. Lazy adentro de funciones para evitar
circular import.

## Re-export

`rag/__init__.py` re-exporta `_snapshot_create`, `_snapshot_restore`,
`_SNAPSHOT_VERSION` para tests que llamen `rag._snapshot_create(...)`.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path

import click

__all__ = [
    "_SNAPSHOT_VERSION",
    "_snapshot_create",
    "_snapshot_restore",
    "snapshot_group",
]


_SNAPSHOT_VERSION = 1


def _snapshot_create() -> dict:
    """Build a snapshot dict ready to serialize. Never raises — missing
    artifacts are reported as empty lists."""
    from rag import (  # noqa: PLC0415
        RANKER_CONFIG_PATH,
        VAULT_PATH,
        _ragvec_state_conn,
    )

    snap: dict = {
        "version": _SNAPSHOT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "vault_path": str(VAULT_PATH),
        "ranker": None,
        "score_calibration": [],
        "learned_paraphrases": [],
        "feedback": [],
    }
    try:
        if RANKER_CONFIG_PATH.is_file():
            snap["ranker"] = json.loads(
                RANKER_CONFIG_PATH.read_text(encoding="utf-8")
            )
    except Exception:
        pass
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT source, raw_knots_json, cal_knots_json, "
                "       n_pos, n_neg, trained_at, model_version, extra_json "
                "FROM rag_score_calibration"
            ).fetchall()
        for src, raw_k, cal_k, n_pos, n_neg, ts, mv, extra in rows:
            snap["score_calibration"].append({
                "source": src, "raw_knots_json": raw_k,
                "cal_knots_json": cal_k,
                "n_pos": n_pos, "n_neg": n_neg,
                "trained_at": ts, "model_version": mv,
                "extra_json": extra,
            })
    except Exception:
        pass
    try:
        with _ragvec_state_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT q_normalized, paraphrase, hit_count, "
                    "       created_ts, last_used_ts "
                    "FROM rag_learned_paraphrases"
                ).fetchall()
                for q, p, hits, ct, lut in rows:
                    snap["learned_paraphrases"].append({
                        "q_normalized": q, "paraphrase": p,
                        "hit_count": hits,
                        "created_ts": ct, "last_used_ts": lut,
                    })
            except Exception:
                pass
            try:
                rows = conn.execute(
                    "SELECT ts, turn_id, rating, q, scope, paths_json, extra_json "
                    "FROM rag_feedback "
                    "WHERE rating = 1 OR "
                    "      (json_extract(extra_json, '$.corrective_path') IS NOT NULL "
                    "       AND json_extract(extra_json, '$.corrective_path') != '')"
                ).fetchall()
                for ts, tid, r, q, sc, pj, ej in rows:
                    snap["feedback"].append({
                        "ts": ts, "turn_id": tid, "rating": r, "q": q,
                        "scope": sc, "paths_json": pj, "extra_json": ej,
                    })
            except Exception:
                pass
    except Exception:
        pass
    return snap


def _snapshot_restore(snap: dict, *, apply_ranker: bool = True,
                       apply_calibration: bool = True,
                       apply_paraphrases: bool = True,
                       apply_feedback: bool = True) -> dict:
    """Apply snapshot sections selectively. Returns stats dict."""
    from rag import RANKER_CONFIG_PATH, _ragvec_state_conn  # noqa: PLC0415

    stats = {
        "ranker_restored": False,
        "calibration_rows": 0,
        "paraphrases_rows": 0,
        "feedback_rows": 0,
        "errors": [],
    }
    if apply_ranker and snap.get("ranker"):
        try:
            RANKER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            RANKER_CONFIG_PATH.write_text(
                json.dumps(snap["ranker"], indent=2), encoding="utf-8"
            )
            stats["ranker_restored"] = True
        except Exception as exc:
            stats["errors"].append(f"ranker: {exc!r}")
    if apply_calibration and snap.get("score_calibration"):
        try:
            with _ragvec_state_conn() as conn:
                for row in snap["score_calibration"]:
                    conn.execute(
                        "INSERT INTO rag_score_calibration "
                        "(source, raw_knots_json, cal_knots_json, "
                        " n_pos, n_neg, trained_at, model_version, extra_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(source) DO UPDATE SET "
                        " raw_knots_json=excluded.raw_knots_json, "
                        " cal_knots_json=excluded.cal_knots_json, "
                        " n_pos=excluded.n_pos, n_neg=excluded.n_neg, "
                        " trained_at=excluded.trained_at, "
                        " model_version=excluded.model_version, "
                        " extra_json=excluded.extra_json",
                        (
                            row.get("source"), row.get("raw_knots_json"),
                            row.get("cal_knots_json"),
                            row.get("n_pos", 0), row.get("n_neg", 0),
                            row.get("trained_at", ""),
                            row.get("model_version", "isotonic-v1"),
                            row.get("extra_json"),
                        ),
                    )
                    stats["calibration_rows"] += 1
        except Exception as exc:
            stats["errors"].append(f"calibration: {exc!r}")
    if apply_paraphrases and snap.get("learned_paraphrases"):
        try:
            with _ragvec_state_conn() as conn:
                for row in snap["learned_paraphrases"]:
                    conn.execute(
                        "INSERT INTO rag_learned_paraphrases "
                        "(q_normalized, paraphrase, hit_count, created_ts, last_used_ts) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(q_normalized, paraphrase) DO UPDATE SET "
                        "  hit_count = rag_learned_paraphrases.hit_count + excluded.hit_count, "
                        "  last_used_ts = excluded.last_used_ts",
                        (
                            row.get("q_normalized"), row.get("paraphrase"),
                            row.get("hit_count", 1),
                            row.get("created_ts", ""), row.get("last_used_ts", ""),
                        ),
                    )
                    stats["paraphrases_rows"] += 1
        except Exception as exc:
            stats["errors"].append(f"paraphrases: {exc!r}")
    if apply_feedback and snap.get("feedback"):
        try:
            with _ragvec_state_conn() as conn:
                for row in snap["feedback"]:
                    conn.execute(
                        "INSERT OR IGNORE INTO rag_feedback "
                        "(ts, turn_id, rating, q, scope, paths_json, extra_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            row.get("ts"), row.get("turn_id"),
                            row.get("rating"), row.get("q"),
                            row.get("scope"), row.get("paths_json"),
                            row.get("extra_json"),
                        ),
                    )
                    stats["feedback_rows"] += 1
        except Exception as exc:
            stats["errors"].append(f"feedback: {exc!r}")
    return stats


@click.group("snapshot")
def snapshot_group():
    """Portable backup/restore de artifacts ML + feedback.

    Empaqueta ranker.json + rag_score_calibration + rag_learned_paraphrases
    + rag_feedback en JSON. Útil para:
      - Backup antes de experimentos arriesgados.
      - Migrar setup entre máquinas.
      - Disaster recovery.
    """


@snapshot_group.command("create")
@click.option("--output", "output_path", default=None,
              help="Path destino. Default: ~/.local/share/obsidian-rag/snapshots/<ts>.json")
@click.option("--as-json", "as_json", is_flag=True,
              help="Imprimir a stdout en lugar de escribir a archivo.")
def snapshot_create(output_path: str | None, as_json: bool):
    """Crear snapshot del estado actual."""
    from rag import console  # noqa: PLC0415

    snap = _snapshot_create()
    payload = json.dumps(snap, indent=2, default=str)
    if as_json:
        click.echo(payload)
        return
    if output_path:
        out = Path(output_path)
    else:
        snap_dir = Path.home() / ".local/share/obsidian-rag/snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        out = snap_dir / f"snapshot-{ts}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Error writing snapshot: {exc!r}[/red]")
        return
    console.print()
    console.print(f"[green]✓[/green] Snapshot creado: [cyan]{out}[/cyan]")
    console.print(f"  ranker: {'sí' if snap.get('ranker') else 'no'}  "
                  f"| calibration: {len(snap['score_calibration'])} sources  "
                  f"| paraphrases: {len(snap['learned_paraphrases'])} rows  "
                  f"| feedback: {len(snap['feedback'])} rows")
    console.print(f"  size: {len(payload):,} bytes")
    console.print()


@snapshot_group.command("restore")
@click.argument("snap_path", type=click.Path(exists=True))
@click.option("--skip-ranker", is_flag=True, help="No restaurar ranker.json.")
@click.option("--skip-calibration", is_flag=True,
              help="No restaurar score_calibration.")
@click.option("--skip-paraphrases", is_flag=True,
              help="No restaurar learned_paraphrases.")
@click.option("--skip-feedback", is_flag=True,
              help="No restaurar feedback rows.")
@click.option("--yes", is_flag=True, help="No confirmar antes de aplicar.")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON con stats.")
def snapshot_restore(
    snap_path: str, skip_ranker: bool, skip_calibration: bool,
    skip_paraphrases: bool, skip_feedback: bool, yes: bool, as_json: bool,
):
    """Restaurar un snapshot creado con `rag snapshot create`.

    Por default aplica TODAS las secciones; usá los flags --skip-*
    para restore selectivo (ej. solo feedback sin tocar el ranker).

    Las restauraciones son UPSERT (por source / q+paraphrase / turn_id
    en feedback). No borra registros existentes — los mergea.

    Antes de aplicar, se hace un snapshot automático del estado actual
    en `~/.local/share/obsidian-rag/snapshots/pre-restore-<ts>.json`
    para rollback manual si hace falta.
    """
    from rag import console  # noqa: PLC0415

    try:
        snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[red]Error leyendo snapshot: {exc!r}[/red]")
        return
    if snap.get("version") != _SNAPSHOT_VERSION:
        console.print(
            f"[yellow]⚠[/yellow] Version mismatch: "
            f"snapshot={snap.get('version')} vs esperada={_SNAPSHOT_VERSION}. "
            "Puede haber incompatibilidades."
        )

    if not as_json:
        console.print()
        console.print(f"[bold]Snapshot[/bold]: [cyan]{snap_path}[/cyan]")
        console.print(f"  created: {snap.get('created_at', '?')}")
        console.print(f"  host: {snap.get('host', '?')}")
        console.print(f"  vault: {snap.get('vault_path', '?')}")
        console.print(f"  ranker: {'sí' if snap.get('ranker') else 'no'}  "
                      f"| calibration: {len(snap.get('score_calibration', []))} sources  "
                      f"| paraphrases: {len(snap.get('learned_paraphrases', []))} rows  "
                      f"| feedback: {len(snap.get('feedback', []))} rows")
        console.print()

    if not yes and not as_json:
        try:
            ans = click.prompt(
                "¿Aplicar? [y/N]", default="n", show_default=False,
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]cancelled[/dim]")
            return
        if ans not in ("y", "s", "yes", "sí"):
            console.print("[dim]cancelled[/dim]")
            return

    pre_snap = _snapshot_create()
    try:
        pre_dir = Path.home() / ".local/share/obsidian-rag/snapshots"
        pre_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        pre_path = pre_dir / f"pre-restore-{ts}.json"
        pre_path.write_text(
            json.dumps(pre_snap, indent=2, default=str), encoding="utf-8",
        )
        if not as_json:
            console.print(f"  [dim]pre-restore backup: {pre_path}[/dim]")
    except Exception:
        pass

    stats = _snapshot_restore(
        snap,
        apply_ranker=not skip_ranker,
        apply_calibration=not skip_calibration,
        apply_paraphrases=not skip_paraphrases,
        apply_feedback=not skip_feedback,
    )
    if as_json:
        click.echo(json.dumps(stats))
        return
    console.print()
    console.print("[bold]Restore summary[/bold]")
    console.print(f"  ranker: {'[green]✓[/green]' if stats['ranker_restored'] else '[dim]skipped[/dim]'}")
    console.print(f"  calibration rows: [cyan]{stats['calibration_rows']}[/cyan]")
    console.print(f"  paraphrases rows: [cyan]{stats['paraphrases_rows']}[/cyan]")
    console.print(f"  feedback rows:    [cyan]{stats['feedback_rows']}[/cyan]")
    if stats["errors"]:
        console.print(f"  [red]errors: {'; '.join(stats['errors'])}[/red]")
    console.print()


@snapshot_group.command("list")
@click.option("--limit", default=10, show_default=True,
              help="Cuántos snapshots mostrar.")
def snapshot_list(limit: int):
    """Listar snapshots en ~/.local/share/obsidian-rag/snapshots/."""
    from rich.table import Table  # noqa: PLC0415

    from rag import console  # noqa: PLC0415

    snap_dir = Path.home() / ".local/share/obsidian-rag/snapshots"
    if not snap_dir.is_dir():
        console.print("[dim]Sin snapshots. Creá uno con `rag snapshot create`.[/dim]")
        return
    files = sorted(snap_dir.glob("*.json"), reverse=True)[:limit]
    if not files:
        console.print("[dim]Sin snapshots.[/dim]")
        return
    console.print()
    t = Table(show_lines=False, header_style="bold")
    t.add_column("File", style="cyan")
    t.add_column("Size", justify="right")
    t.add_column("Created", style="dim")
    for f in files:
        size_kb = f.stat().st_size / 1024
        ts = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        t.add_row(f.name, f"{size_kb:.1f} KB", ts)
    console.print(t)
    console.print()
