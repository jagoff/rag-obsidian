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

## Helpers (CLI-side, definidos aca)

  _all_daemon_labels()              — managed + manual labels
  _plist_on_disk(label)             — Path check
  _compute_reconcile_actions()      — diff state vs spec
  _execute_reconcile_action(action) — launchctl invocation con timeouts
  _doctor_diagnose(row)             — paragraph diagnostico

## Helpers stdlib-only (Phase 2c.2, 2026-05-09)

Movidos de rag/__init__.py para eliminar `from rag import X` lazy
indirection en este modulo. Stdlib puro (subprocess + re + pathlib).
Re-export shim en rag/__init__.py preserva `rag._foo` paths usados
por tests (monkeypatch.setattr(rag, "_bootstrap_label", ...)).

  _parse_launchctl_print(stdout)    — regex parser de `launchctl print`
  _daemon_log_path(label)           — heuristica path de log
  _plist_cadence_seconds(label)     — extrae StartInterval del plist
  _gather_daemon_status(label, cat) — orquesta los 3 anteriores
  _loaded_launchd_labels(timeout)   — set de labels cargados via list
  _bootout_label(label)             — wrapper de `launchctl bootout`
  _bootstrap_label(label)           — wrapper de `launchctl bootstrap`

## Patron CLI sub-package

`@click.group("daemons")` standalone (NO `@cli.group(...)`). Los
sub-commands usan `@daemons_group.command(...)` ya que el group y
sus commands viven en el mismo modulo. Wiring en `rag/__init__.py`:
  cli.add_command(daemons_group, name="daemons")

## Lazy imports

Deps en `rag/__init__.py`: `_rag_binary`, `_services_spec`,
`_services_spec_manual`, `_log_daemon_run_event`, `console`. Lazy
adentro de cada funcion. `rich.table.Table` se importa lazy adentro
de los CLI commands. `_gather_daemon_status` ahora local pero los
callers Click commands lo siguen importando lazy via `from rag import
_gather_daemon_status` (re-export shim) para que tests existentes con
`patch.object(rag, "_gather_daemon_status", ...)` sigan funcionando.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import click

from rag.plists import _LAUNCH_AGENTS_DIR, _RAG_LOG_DIR

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
    # Helpers stdlib-only movidos de rag/__init__.py (Phase 2c.2, 2026-05-09):
    "_parse_launchctl_print",
    "_daemon_log_path",
    "_plist_cadence_seconds",
    "_gather_daemon_status",
    "_loaded_launchd_labels",
    "_bootout_label",
    "_bootstrap_label",
]


# ── Helpers launchd (stdlib-only) ───────────────────────────────────
# Movidos de rag/__init__.py el 2026-05-09 (Phase 2c.2 modularización).
# Stdlib puro: `subprocess` / `re` / `pathlib.Path`. Cero deps cross-module
# salvo `_LAUNCH_AGENTS_DIR` y `_RAG_LOG_DIR` que vienen de `rag.plists`.
# Re-export shim en `rag/__init__.py` preserva `rag._foo` paths que usan
# los tests (monkeypatch.setattr(rag, "_bootstrap_label", ...)).


def _parse_launchctl_print(stdout: str) -> dict:
    """Parse output de `launchctl print gui/<uid>/<label>`.

    Extrae state, runs y last_exit. Devuelve un dict con las 3 claves;
    valores no encontrados quedan None.

    Nota: el output de `launchctl print` puede tener sub-secciones con
    `state = active` para procesos hijos. Capturamos solo la PRIMERA
    ocurrencia de cada clave para evitar sobreescribir con sub-estados.
    """
    import re
    result: dict = {"state": None, "runs": None, "last_exit": None}
    for line in stdout.splitlines():
        stripped = line.strip()
        if result["state"] is None:
            m = re.match(r"state\s*=\s*(.+)", stripped)
            if m:
                result["state"] = m.group(1).strip()
                continue
        if result["runs"] is None:
            m = re.match(r"runs\s*=\s*(\d+)", stripped)
            if m:
                result["runs"] = int(m.group(1))
                continue
        if result["last_exit"] is None:
            m = re.match(r"last exit code\s*=\s*(.+)", stripped)
            if m:
                raw = m.group(1).strip()
                try:
                    result["last_exit"] = int(raw)
                except ValueError:
                    result["last_exit"] = raw
        # Early exit when all 3 fields are filled
        if all(v is not None for v in result.values()):
            break
    return result


def _plist_cadence_seconds(label: str) -> int | None:
    """Lee el plist en ~/Library/LaunchAgents/<label>.plist y extrae la
    cadencia esperada en segundos.

    - StartInterval → el valor directamente (segundos).
    - StartCalendarInterval → 86400 (24h como aproximación).
    - KeepAlive sin StartInterval → None (no aplica overdue).
    - Archivo no existe → None.
    """
    import re
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    if not plist_path.exists():
        return None
    try:
        content = plist_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # StartInterval
    m = re.search(
        r"<key>StartInterval</key>\s*<integer>(\d+)</integer>",
        content,
    )
    if m:
        return int(m.group(1))
    # StartCalendarInterval → tratar como 24h
    if "<key>StartCalendarInterval</key>" in content:
        return 86400
    return None


def _daemon_log_path(label: str) -> Path:
    """Heurística: ~/.local/share/obsidian-rag/<slug>.log."""
    slug = label.replace("com.fer.obsidian-rag-", "")
    return _RAG_LOG_DIR / f"{slug}.log"


def _gather_daemon_status(label: str, category: str) -> dict:
    """Recolecta estado de un daemon: launchctl print + mtime log + overdue."""
    import subprocess

    # Deferred re-resolve via `rag` para que los tests existentes
    # (`patch.object(rag, "_daemon_log_path", ...)` /
    # `patch.object(rag, "_plist_cadence_seconds", ...)` /
    # `patch.object(rag, "_parse_launchctl_print", ...)`) sigan ganando
    # sobre el binding del sub-modulo. Patron documentado en CLAUDE.md
    # ("Modular split shim pattern", seccion Architecture invariants).
    from rag import (  # noqa: PLC0415
        _daemon_log_path as _rag_daemon_log_path,
        _parse_launchctl_print as _rag_parse_launchctl_print,
        _plist_cadence_seconds as _rag_plist_cadence_seconds,
    )

    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        proc = None  # type: ignore[assignment]

    if proc is None or (proc.returncode != 0 and (
        "Could not find service" in (proc.stderr or "")
        or proc.returncode == 113
    )):
        parsed = {"state": "missing", "runs": None, "last_exit": None}
    elif proc.returncode != 0:
        parsed = {"state": "unknown", "runs": None, "last_exit": None}
    else:
        parsed = _rag_parse_launchctl_print(proc.stdout)
        if parsed["state"] is None:
            parsed["state"] = "unknown"

    # Last tick via logfile mtime
    log_path = _rag_daemon_log_path(label)
    last_tick_iso: str | None = None
    if log_path.exists():
        try:
            ts = datetime.fromtimestamp(log_path.stat().st_mtime)
            last_tick_iso = ts.isoformat(timespec="seconds")
        except OSError:
            pass

    # Overdue check
    expected_cadence_s = _rag_plist_cadence_seconds(label)
    overdue = False
    if expected_cadence_s and last_tick_iso:
        try:
            last_dt = datetime.fromisoformat(last_tick_iso)
            age_s = (datetime.now() - last_dt).total_seconds()
            overdue = age_s > 2 * expected_cadence_s
        except Exception:
            pass

    return {
        "label": label,
        "category": category,
        "state": parsed["state"],
        "runs": parsed["runs"],
        "last_exit": parsed["last_exit"],
        "last_tick_iso": last_tick_iso,
        "overdue": overdue,
        "expected_cadence_s": expected_cadence_s,
    }


def _loaded_launchd_labels(timeout: int = 5) -> set[str]:
    """Set de labels actualmente cargados en `gui/$UID` via `launchctl list`.

    Una sola llamada para evitar N forks (un `launchctl print` por label es
    lento). Si falla, devuelve set vacío — el caller debe asumir "no sé" y
    no usarlo para gating destructivo.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()
    if proc.returncode != 0:
        return set()
    out: set[str] = set()
    for line in (proc.stdout or "").splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2]:
            out.add(parts[2].strip())
    return out


def _bootout_label(label: str, *, dry_run: bool = False, timeout: int = 15) -> dict:
    """`launchctl bootout gui/$UID/<label>` con timeout + manejo de exit codes.

    Devuelve {"ok": bool, "exit_code": int, "stderr": str, "skipped": bool}.

    Códigos tratados como "OK" (no error, ya estaba parado o no existía):
        exit=3   → service not loaded (ya bootouted)
        exit=113 → "Could not find service" (mismo escenario)
    """
    import subprocess
    if dry_run:
        return {"ok": True, "exit_code": 0, "stderr": "", "skipped": True}
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True, text=True, timeout=timeout,
        )
        rc = proc.returncode
        stderr = (proc.stderr or "").strip()
        # 0 = bootouted; 3 / 113 = ya estaba parado (no es error real).
        ok = rc in (0, 3, 113)
        return {"ok": ok, "exit_code": rc, "stderr": stderr, "skipped": False}
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": 124, "stderr": f"timeout {timeout}s", "skipped": False}
    except OSError as exc:
        return {"ok": False, "exit_code": -1, "stderr": f"OSError: {exc}", "skipped": False}


def _bootstrap_label(label: str, *, dry_run: bool = False, timeout: int = 30) -> dict:
    """`launchctl bootstrap gui/$UID <plist>` con timeout + manejo de exit codes.

    Simétrico a `_bootout_label` — usado por `rag start` para levantar daemons
    EXTERNOS (RagNet whatsapp-*, qdrant) cuyos plists viven en
    `~/Library/LaunchAgents/` pero NO están managed por `_services_spec`.
    Para los managed (`obsidian-rag-*`) seguimos por `setup.callback()` que
    los regenera desde código y los carga vía `launchctl load`.

    Devuelve {"ok": bool, "exit_code": int, "stderr": str, "skipped": bool,
              "missing_plist": bool}.

    Códigos / casos tratados como OK:
        exit=0                                → loaded (success)
        exit=37                               → "Operation already in progress"
        stderr contiene "already loaded" /
            "already bootstrapped"            → ya estaba cargado (no error real)
    """
    import subprocess
    if dry_run:
        return {"ok": True, "exit_code": 0, "stderr": "",
                "skipped": True, "missing_plist": False}
    plist = _LAUNCH_AGENTS_DIR / f"{label}.plist"
    if not plist.is_file():
        return {"ok": False, "exit_code": -1,
                "stderr": "plist no existe en disco",
                "skipped": False, "missing_plist": True}
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist)],
            capture_output=True, text=True, timeout=timeout,
        )
        rc = proc.returncode
        stderr = (proc.stderr or "").strip()
        stderr_lc = stderr.lower()
        # 0 = loaded; 37 = ya en progreso; "already loaded/bootstrapped" en
        # stderr = ya estaba cargado (rc varía por versión de launchctl).
        ok = (
            rc in (0, 37)
            or "already loaded" in stderr_lc
            or "already bootstrapped" in stderr_lc
            or "service is disabled" in stderr_lc  # opt-in necesario, no es error fatal
        )
        return {"ok": ok, "exit_code": rc, "stderr": stderr,
                "skipped": False, "missing_plist": False}
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": 124,
                "stderr": f"timeout {timeout}s",
                "skipped": False, "missing_plist": False}
    except OSError as exc:
        return {"ok": False, "exit_code": -1,
                "stderr": f"OSError: {exc}",
                "skipped": False, "missing_plist": False}


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
