"""`rag daemons` — control plane de los daemons launchd.

Phase 2b de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el grupo CLI `rag daemons` con sus 5 sub-commands +
helpers desde `rag/__init__.py` (que ya bajó a 59.0k LOC tras
Phase 3 feedback_judge) a `rag/cli/daemons_control.py`.

## Sub-commands

  status              — tabla Rich del estado de cada daemon
  reconcile [--apply] — compara real vs spec, converge launchctl state
  doctor              — diagnostica daemons unhealthy + remediación
  retry <label>       — kickstart -k de un daemon específico
  kickstart-overdue   — kickstart de TODOS los daemons overdue (post-wake)

## Helpers

  _all_daemon_labels()              — managed + manual labels
  _plist_on_disk(label)             — Path check
  _compute_reconcile_actions()      — diff state vs spec
  _execute_reconcile_action(action) — launchctl invocation con timeouts
  _doctor_diagnose(row)             — paragraph diagnóstico

## Patrón CLI sub-package

`@click.group("daemons")` standalone (NO `@cli.group(...)`). Los
sub-commands usan `@daemons_group.command(...)` ya que el group y
sus commands viven en el mismo módulo. Wiring en `rag/__init__.py`:
  cli.add_command(daemons_group, name="daemons")

## Lazy imports

Deps en `rag/__init__.py`: `_rag_binary`, `_services_spec`,
`_services_spec_manual`, `_gather_daemon_status`,
`_log_daemon_run_event`, `console`. Lazy adentro de cada función.
`rich.table.Table` se importa lazy adentro de los CLI commands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

__all__ = [
    "daemons_group",
    "daemons_status",
    "daemons_reconcile",
    "daemons_doctor",
    "daemons_retry",
    "daemons_kickstart_overdue",
    "_all_daemon_labels",
    "_plist_on_disk",
    "_compute_reconcile_actions",
    "_execute_reconcile_action",
    "_doctor_diagnose",
]


@click.group("daemons")
def daemons_group():
    """Control plane de los daemons launchd del proyecto."""


@daemons_group.command("status")
@click.option("--json", "as_json", is_flag=True,
              help="Emite JSON en vez de tabla Rich.")
@click.option("--unhealthy-only", is_flag=True,
              help="Solo mostrar daemons con drift (no running, overdue o exit≠0).")
def daemons_status(as_json: bool, unhealthy_only: bool):
    """Estado de cada daemon launchd: state, runs, last_exit, last_tick, overdue."""
    from rich.table import Table  # noqa: PLC0415

    from rag import (  # noqa: PLC0415
        _gather_daemon_status,
        _log_daemon_run_event,
        _rag_binary,
        _services_spec,
        _services_spec_manual,
        console,
    )

    rag_bin = _rag_binary()

    managed_rows = [
        _gather_daemon_status(label, "managed")
        for (label, _fname, _xml) in _services_spec(rag_bin)
    ]
    manual_rows = [
        _gather_daemon_status(spec["label"], spec["category"])
        for spec in _services_spec_manual()
    ]
    all_rows = managed_rows + manual_rows

    def _is_unhealthy(row: dict) -> bool:
        if row["state"] not in ("running", None):
            if row["state"] != "running":
                return True
        last_exit = row["last_exit"]
        if isinstance(last_exit, int) and last_exit != 0:
            return True
        if row["overdue"]:
            return True
        return False

    if unhealthy_only:
        all_rows = [r for r in all_rows if _is_unhealthy(r)]

    unhealthy_count = sum(1 for r in all_rows if _is_unhealthy(r))

    if as_json:
        click.echo(json.dumps(all_rows, indent=2))
    else:
        table = Table(title="Daemons launchd", show_lines=False, header_style="bold")
        table.add_column("Label", style="dim", no_wrap=True)
        table.add_column("Cat", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("Runs", justify="right")
        table.add_column("LastExit", justify="right")
        table.add_column("LastTick", no_wrap=True)
        table.add_column("Overdue")

        for row in all_rows:
            state = row["state"] or "unknown"
            if state == "running":
                state_text = f"[green]{state}[/green]"
            elif state == "missing":
                state_text = f"[red]{state}[/red]"
            else:
                state_text = state

            last_exit = row["last_exit"]
            if isinstance(last_exit, int) and last_exit != 0:
                exit_text = f"[red]{last_exit}[/red]"
            elif last_exit is None:
                exit_text = "-"
            else:
                exit_text = str(last_exit)

            overdue_text = "[yellow]sí[/yellow]" if row["overdue"] else "no"

            slug = row["label"].replace("com.fer.obsidian-rag-", "")
            last_tick = row["last_tick_iso"] or "-"

            table.add_row(
                slug,
                row["category"].replace("manual_keep", "manual").replace("managed", "mgd"),
                state_text,
                str(row["runs"]) if row["runs"] is not None else "-",
                exit_text,
                last_tick,
                overdue_text,
            )

        console.print(table)
        if unhealthy_count:
            console.print(f"[yellow]{unhealthy_count} daemon(s) con drift[/yellow]")

    _log_daemon_run_event(
        label="<status_run>",
        action="status_check",
        reason=f"unhealthy={unhealthy_count}",
    )


def _all_daemon_labels() -> list[tuple[str, str]]:
    """Devuelve lista de (label, category) de todos los daemons conocidos."""
    from rag import _rag_binary, _services_spec, _services_spec_manual  # noqa: PLC0415

    rag_bin = _rag_binary()
    managed = [(label, "managed") for (label, _fname, _xml) in _services_spec(rag_bin)]
    manual = [(spec["label"], spec["category"]) for spec in _services_spec_manual()]
    return managed + manual


def _plist_on_disk(label: str) -> bool:
    """True si ~/Library/LaunchAgents/<label>.plist existe en disco."""
    return (Path.home() / "Library" / "LaunchAgents" / f"{label}.plist").exists()


def _compute_reconcile_actions(
    *, gentle: bool = False, regenerate: bool = False,
) -> list[dict]:
    """Compara estado real vs spec y devuelve lista de acciones requeridas.

    Cada acción es un dict con claves:
        label        — nombre completo del daemon
        kind         — "bootstrap" | "bootout" | "kickstart" | "regenerate"
        reason       — texto libre explicativo
        current_state — dict con state/runs/last_exit/overdue/etc.
        plist_path   — Path al plist en disco (puede ser None)
        factory_xml  — solo para kind="regenerate": XML output de la factory

    Con `gentle=True` solo se generan acciones `kickstart` (no bootout
    huérfanos, no bootstrap, no regenerate). El watchdog corre así.
    """
    from rag import _gather_daemon_status, _rag_binary, _services_spec  # noqa: PLC0415

    actions: list[dict] = []
    _SELF_LABEL = "com.fer.obsidian-rag-daemon-watchdog"

    factory_xml_by_label: dict[str, str] = {}
    if regenerate:
        rag_bin = _rag_binary()
        try:
            for managed_label, _fname, xml_str in _services_spec(rag_bin):
                factory_xml_by_label[managed_label] = xml_str
        except Exception:
            factory_xml_by_label = {}

    for label, category in _all_daemon_labels():
        if label == _SELF_LABEL:
            continue
        row = _gather_daemon_status(label, category)
        state = row.get("state")
        plist_exists = _plist_on_disk(label)
        plist_path = (
            Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
            if plist_exists else None
        )
        last_exit = row.get("last_exit")
        runs = row.get("runs")
        overdue = row.get("overdue", False)

        if (
            regenerate
            and category == "managed"
            and plist_exists
            and plist_path is not None
            and label in factory_xml_by_label
        ):
            try:
                on_disk = plist_path.read_text(encoding="utf-8")
            except OSError:
                on_disk = ""
            factory_xml = factory_xml_by_label[label]
            if on_disk.strip() != factory_xml.strip():
                actions.append({
                    "label": label,
                    "kind": "regenerate",
                    "reason": "on-disk XML drifted vs factory",
                    "current_state": row,
                    "plist_path": plist_path,
                    "factory_xml": factory_xml,
                })
                continue

        if state == "missing" and plist_exists:
            actions.append({
                "label": label,
                "kind": "bootstrap",
                "reason": "plist on disk, not loaded",
                "current_state": row,
                "plist_path": plist_path,
            })
            continue

        if state != "missing" and not plist_exists and not gentle:
            actions.append({
                "label": label,
                "kind": "bootout",
                "reason": "loaded but no plist on disk",
                "current_state": row,
                "plist_path": None,
            })
            continue

        if (
            isinstance(last_exit, int)
            and last_exit != 0
            and runs is not None
            and runs < 3
        ):
            actions.append({
                "label": label,
                "kind": "kickstart",
                "reason": f"last_exit={last_exit} runs={runs}",
                "current_state": row,
                "plist_path": plist_path,
            })
            continue

        if overdue:
            actions.append({
                "label": label,
                "kind": "kickstart",
                "reason": "overdue Nx cadence",
                "current_state": row,
                "plist_path": plist_path,
            })

    return actions


def _execute_reconcile_action(action: dict) -> dict:
    """Ejecuta una acción de reconciliación via launchctl.

    Devuelve {"ok": bool, "exit_code": int, "stderr": str | None}.

    Códigos de salida tratados como éxito:
        bootstrap exit=37 → EALREADY (ya cargado)
        bootout   exit=3  → no existe (ya bootouted)
    """
    import subprocess  # noqa: PLC0415

    label = action["label"]
    kind = action["kind"]
    uid = os.getuid()

    def _safe_run(args: list[str], timeout: int = 10) -> dict:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
            return {"rc": p.returncode, "stderr": p.stderr or None}
        except subprocess.TimeoutExpired:
            return {"rc": -2, "stderr": f"timeout {timeout}s"}
        except OSError as exc:
            return {"rc": -3, "stderr": f"OSError: {exc}"}

    if kind == "bootstrap":
        plist_path = action.get("plist_path")
        if plist_path is None:
            return {"ok": False, "exit_code": -1, "stderr": "plist_path missing"}
        r = _safe_run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)])
        ok = r["rc"] in (0, 37)
        return {"ok": ok, "exit_code": r["rc"], "stderr": r["stderr"]}

    elif kind == "bootout":
        r = _safe_run(["launchctl", "bootout", f"gui/{uid}/{label}"])
        ok = r["rc"] in (0, 3)
        return {"ok": ok, "exit_code": r["rc"], "stderr": r["stderr"]}

    elif kind == "kickstart":
        r = _safe_run(["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"])
        ok = r["rc"] == 0
        return {"ok": ok, "exit_code": r["rc"], "stderr": r["stderr"]}

    elif kind == "regenerate":
        plist_path = action.get("plist_path")
        factory_xml = action.get("factory_xml")
        if plist_path is None or not factory_xml:
            return {"ok": False, "exit_code": -1, "stderr": "missing plist_path or factory_xml"}
        try:
            Path(plist_path).write_text(factory_xml, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "exit_code": -4, "stderr": f"write failed: {exc}"}
        r1 = _safe_run(["launchctl", "bootout", f"gui/{uid}/{label}"])
        r2 = _safe_run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)])
        ok = r2["rc"] in (0, 37)
        stderr = r2["stderr"] or (r1["stderr"] if r1["rc"] not in (0, 3) else None)
        return {"ok": ok, "exit_code": r2["rc"], "stderr": stderr}

    return {"ok": False, "exit_code": -1, "stderr": f"kind desconocido: {kind}"}


def _doctor_diagnose(row: dict) -> str:
    """Devuelve párrafo de diagnóstico + remediación para un daemon unhealthy."""
    label = row["label"]
    slug = label.replace("com.fer.obsidian-rag-", "")
    state = row.get("state", "unknown")
    last_exit = row.get("last_exit")
    overdue = row.get("overdue", False)
    runs = row.get("runs")
    last_tick = row.get("last_tick_iso")

    lines: list[str] = [f"── {slug} ──"]

    if state == "missing":
        lines.append(
            f"  Síntoma: plist no bootstrappeado (state=missing).\n"
            f"  Remediación: `rag daemons reconcile --apply` (si plist en disco) "
            f"o `rag setup` (si plist faltante)."
        )
        return "\n".join(lines)

    if label.endswith("ingest-safari") and isinstance(last_exit, int) and last_exit == 1:
        lines.append(
            f"  Síntoma: exit=1 (database lock conocido de Safari History.db).\n"
            f"  Remediación: `rag daemons retry ingest-safari` para reintentar; "
            f"si persiste, ver `~/.local/share/obsidian-rag/ingest-safari.error.log`."
        )
        return "\n".join(lines)

    if label.endswith("web") and isinstance(runs, int) and runs > 5:
        if last_tick:
            try:
                from datetime import datetime as _dt  # noqa: PLC0415
                age_min = (_dt.now() - _dt.fromisoformat(last_tick)).total_seconds() / 60
                if age_min < 5:
                    lines.append(
                        f"  Síntoma: posible crash loop (runs={runs}, last_tick hace "
                        f"{age_min:.0f}min).\n"
                        f"  Remediación: ver `~/.local/share/obsidian-rag/web.error.log` "
                        f"para la causa raíz."
                    )
                    return "\n".join(lines)
            except Exception:
                pass

    if overdue:
        lines.append(
            f"  Síntoma: daemon overdue (Mac posiblemente dormida durante el tick).\n"
            f"  Remediación: `rag daemons retry {slug}` para forzar tick."
        )
        return "\n".join(lines)

    if isinstance(last_exit, int) and last_exit != 0:
        lines.append(
            f"  Síntoma: último exit_code={last_exit}.\n"
            f"  Remediación: ver log: `tail ~/.local/share/obsidian-rag/{slug}.error.log`."
        )
        return "\n".join(lines)

    lines.append(f"  Síntoma: state={state}, last_exit={last_exit}, overdue={overdue}.")
    return "\n".join(lines)


@daemons_group.command("reconcile")
@click.option("--apply", "do_apply", is_flag=True,
              help="Ejecutar las acciones (default: dry-run).")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Solo mostrar qué se haría.")
@click.option("--gentle", is_flag=True,
              help="Solo kickstart de last_exit≠0 con runs<3 + overdue. "
                   "NO bootout huérfanos ni regenera plists.")
@click.option("--regenerate", is_flag=True,
              help="Detectar drift entre el XML del plist on-disk vs la "
                   "factory en código y regenerar (write + bootout + "
                   "bootstrap). Solo aplica a daemons managed.")
def daemons_reconcile(do_apply: bool, dry_run: bool, gentle: bool, regenerate: bool):
    """Compara estado real vs spec y converge (default: dry-run)."""
    import time  # noqa: PLC0415

    from rich.table import Table  # noqa: PLC0415

    from rag import _gather_daemon_status, _log_daemon_run_event, console  # noqa: PLC0415

    if not do_apply and not dry_run:
        dry_run = True
        console.print("[dim]dry-run implícito — usá --apply para ejecutar acciones.[/dim]")

    if gentle and regenerate:
        console.print("[yellow]warning: --regenerate es ignorado con --gentle[/yellow]")
        regenerate = False

    actions = _compute_reconcile_actions(gentle=gentle, regenerate=regenerate)

    if not actions:
        console.print("✓ sin drift detectado")
        return

    table = Table(title="Acciones reconcile", show_lines=False, header_style="bold")
    table.add_column("Label", style="dim", no_wrap=True)
    table.add_column("Acción", no_wrap=True)
    table.add_column("Motivo")

    for a in actions:
        slug = a["label"].replace("com.fer.obsidian-rag-", "")
        table.add_row(slug, a["kind"], a["reason"])

    console.print(table)

    if dry_run:
        console.print("[dim]dry-run — ejecutá con --apply para aplicar[/dim]")
        return

    ok_count = 0
    fail_count = 0
    for a in actions:
        prev_state = a["current_state"].get("state")
        result = _execute_reconcile_action(a)

        time.sleep(1)
        post_row = _gather_daemon_status(a["label"], a["current_state"].get("category", "managed"))
        new_state = post_row.get("state") if result["ok"] else None

        _log_daemon_run_event(
            label=a["label"],
            action=f"reconcile_{a['kind']}",
            prev_state=prev_state,
            new_state=new_state,
            exit_code=result["exit_code"],
            reason=a["reason"],
        )

        slug = a["label"].replace("com.fer.obsidian-rag-", "")
        if result["ok"]:
            console.print(f"[green]✓[/green] {slug} → {a['kind']} ok")
            ok_count += 1
        else:
            console.print(
                f"[red]✗[/red] {slug} → {a['kind']} exit={result['exit_code']}"
                + (f" ({result['stderr'][:80]})" if result.get("stderr") else "")
            )
            fail_count += 1

    console.print(
        f"\n[green]{ok_count} ok[/green]"
        + (f"  [red]{fail_count} fallidos[/red]" if fail_count else "")
    )


@daemons_group.command("doctor")
def daemons_doctor():
    """Diagnostica daemons unhealthy y sugiere remediación (solo lectura)."""
    from rag import _gather_daemon_status, console  # noqa: PLC0415

    all_labels = _all_daemon_labels()
    rows = [_gather_daemon_status(label, category) for label, category in all_labels]

    def _is_unhealthy_row(row: dict) -> bool:
        state = row["state"]
        last_exit = row["last_exit"]
        if state == "missing":
            return True
        if isinstance(last_exit, int) and last_exit != 0:
            return True
        if row["overdue"]:
            return True
        return False

    unhealthy = [r for r in rows if _is_unhealthy_row(r)]

    if not unhealthy:
        console.print("✓ todos los daemons sanos")
        return

    for row in unhealthy:
        console.print(_doctor_diagnose(row))
        console.print()


@daemons_group.command("retry")
@click.argument("label")
def daemons_retry(label: str):
    """Forzar un kickstart -k del daemon LABEL (slug corto o nombre completo)."""
    import subprocess  # noqa: PLC0415
    import time  # noqa: PLC0415

    from rag import _gather_daemon_status, _log_daemon_run_event, console  # noqa: PLC0415

    full_label = (
        label if label.startswith("com.fer.obsidian-rag-")
        else f"com.fer.obsidian-rag-{label}"
    )

    known_labels = {lbl for lbl, _cat in _all_daemon_labels()}
    if full_label not in known_labels:
        raise click.BadParameter(
            f"'{full_label}' no está en la spec conocida de daemons. "
            f"Usá `rag daemons status` para ver los labels válidos.",
            param_hint="label",
        )

    uid = os.getuid()

    category = next(
        (cat for lbl, cat in _all_daemon_labels() if lbl == full_label), "managed"
    )
    prev_row = _gather_daemon_status(full_label, category)
    prev_state = prev_row.get("state")

    proc = subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{full_label}"],
        capture_output=True, text=True, timeout=10,
    )
    ok = proc.returncode == 0

    time.sleep(2)
    post_row = _gather_daemon_status(full_label, category)
    new_state = post_row.get("state") if ok else None

    _log_daemon_run_event(
        label=full_label,
        action="retry",
        prev_state=prev_state,
        new_state=new_state,
        exit_code=proc.returncode,
        reason="manual retry via CLI",
    )

    slug = full_label.replace("com.fer.obsidian-rag-", "")
    if ok:
        console.print(f"[green]✓[/green] {slug} → kickstart ok (state: {prev_state} → {new_state})")
    else:
        stderr_hint = proc.stderr.strip()[:120] if proc.stderr else ""
        console.print(
            f"[red]✗[/red] {slug} → kickstart exit={proc.returncode}"
            + (f"\n  {stderr_hint}" if stderr_hint else "")
        )


@daemons_group.command("kickstart-overdue")
def daemons_kickstart_overdue():
    """Kickstart de TODOS los daemons en estado overdue.

    Recupera daemons saltados mientras el Mac estaba dormida.
    """
    import subprocess  # noqa: PLC0415
    import time  # noqa: PLC0415

    from rag import _gather_daemon_status, _log_daemon_run_event, console  # noqa: PLC0415

    all_labels = _all_daemon_labels()
    rows = [_gather_daemon_status(label, category) for label, category in all_labels]
    overdue = [r for r in rows if r.get("overdue")]

    if not overdue:
        console.print("✓ ningún daemon overdue")
        return

    uid = os.getuid()
    ok_count = 0
    fail_count = 0

    for row in overdue:
        label = row["label"]
        slug = label.replace("com.fer.obsidian-rag-", "")
        prev_state = row.get("state")

        exit_code = -1
        ok = False
        try:
            proc = subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
                capture_output=True, text=True, timeout=30,
            )
            exit_code = proc.returncode
            ok = exit_code == 0
        except subprocess.TimeoutExpired:
            exit_code = 124
        except OSError as exc:
            exit_code = -2
            console.print(f"[red]✗[/red] {slug} → OSError: {exc}")

        time.sleep(1)
        post_row = _gather_daemon_status(label, row.get("category", "managed"))
        new_state = post_row.get("state") if ok else None

        _log_daemon_run_event(
            label=label,
            action="kickstart",
            prev_state=prev_state,
            new_state=new_state,
            exit_code=exit_code,
            reason="kickstart-overdue batch",
        )

        if ok:
            console.print(f"[green]✓[/green] {slug} → kickstart ok")
            ok_count += 1
        elif exit_code == 124:
            console.print(f"[yellow]⏱[/yellow] {slug} → kickstart timeout (30s) — sigo")
            fail_count += 1
        else:
            console.print(f"[red]✗[/red] {slug} → exit={exit_code}")
            fail_count += 1

    console.print(
        f"\n✓ kickstarteados {ok_count} daemons overdue"
        + (f"  [red]{fail_count} fallidos[/red]" if fail_count else "")
    )
