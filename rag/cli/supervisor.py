"""CLI ``rag supervisor`` — gestiona el daemon supervisor in-process.

Subcomandos:

- ``rag supervisor run`` — entrypoint long-running, invocado por launchd.
  En foreground bloquea hasta SIGTERM.
- ``rag supervisor status`` — IPC GET ``/status``, lista jobs registrados
  con stats (runs, fails, last_exit_code, last_duration_s).
- ``rag supervisor trigger <job>`` — IPC POST ``/run/<job>``, dispara un
  job sincrónico y muestra el resultado.
- ``rag supervisor jobs`` — list rápido solo de labels.
- ``rag supervisor logs [--follow]`` — tail del supervisor.log.
- ``rag supervisor ping`` — health check via IPC.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import click

__all__ = ["supervisor_group"]


@click.group("supervisor")
def supervisor_group() -> None:
    """Gestión del supervisor in-process (reemplaza N plists launchd)."""


@supervisor_group.command("run")
def run_cmd() -> None:
    """Entrypoint long-running. Bloquea hasta SIGTERM/SIGINT.

    Invocado por launchd vía ``com.fer.obsidian-rag-supervisor.plist``
    o manualmente para debugging."""
    from rag.runtime.supervisor import main as _main  # noqa: PLC0415
    sys.exit(_main())


def _ipc_call(action: str, **kwargs):
    """Cliente IPC con error handling user-friendly."""
    from rag.runtime import ipc  # noqa: PLC0415
    try:
        return ipc.client_call(action, **kwargs)
    except FileNotFoundError:
        click.secho(
            "✗ supervisor no está corriendo (socket no existe).\n"
            "  arrancalo con: launchctl kickstart -k "
            "gui/$(id -u)/com.fer.obsidian-rag-supervisor",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except (ConnectionRefusedError, OSError) as exc:
        click.secho(f"✗ supervisor unreachable: {exc}", fg="red", err=True)
        sys.exit(2)


@supervisor_group.command("ping")
def ping_cmd() -> None:
    """Health check rápido via IPC."""
    t0 = time.time()
    resp = _ipc_call("ping")
    elapsed_ms = (time.time() - t0) * 1000
    if resp.get("ok"):
        result = resp.get("result", {})
        click.secho(
            f"✓ pong  (latencia {elapsed_ms:.1f}ms · ts={result.get('ts')})",
            fg="green",
        )
    else:
        click.secho(f"✗ {resp.get('error')}", fg="red", err=True)
        sys.exit(1)


@supervisor_group.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
def status_cmd(as_json: bool) -> None:
    """Tabla de jobs registrados con stats."""
    resp = _ipc_call("status")
    if not resp.get("ok"):
        click.secho(f"✗ {resp.get('error')}", fg="red", err=True)
        sys.exit(1)
    result = resp.get("result", {})
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    jobs = result.get("jobs", [])
    uptime = result.get("uptime_s", 0)
    click.echo(f"supervisor uptime: {uptime:.0f}s · jobs registrados: {len(jobs)}")
    click.echo("")
    fmt = "{label:<30} {trigger:<10} {runs:>6} {fails:>5} {last_exit:>4} {last_dur:>8}"
    click.echo(fmt.format(
        label="LABEL", trigger="TRIGGER", runs="RUNS",
        fails="FAILS", last_exit="EXIT", last_dur="DURs",
    ))
    click.echo("─" * 78)
    for j in sorted(jobs, key=lambda j: j["label"]):
        last_dur = j.get("last_duration_s")
        last_dur_s = f"{last_dur:.2f}" if last_dur is not None else "—"
        last_exit = j.get("last_exit_code")
        last_exit_s = str(last_exit) if last_exit is not None else "—"
        click.echo(fmt.format(
            label=j["label"][:30],
            trigger=j["trigger_kind"],
            runs=j["runs_count"],
            fails=j["fails_count"],
            last_exit=last_exit_s,
            last_dur=last_dur_s,
        ))


@supervisor_group.command("jobs")
def jobs_cmd() -> None:
    """Lista nomás los labels registrados."""
    resp = _ipc_call("jobs")
    if not resp.get("ok"):
        click.secho(f"✗ {resp.get('error')}", fg="red", err=True)
        sys.exit(1)
    for label in resp.get("result", {}).get("labels", []):
        click.echo(label)


@supervisor_group.command("trigger")
@click.argument("job")
def trigger_cmd(job: str) -> None:
    """Dispara ``<job>`` sincrónicamente vía IPC. Muestra el result."""
    t0 = time.time()
    resp = _ipc_call("run", job=job)
    elapsed = time.time() - t0
    if not resp.get("ok"):
        click.secho(f"✗ {resp.get('error')}", fg="red", err=True)
        sys.exit(1)
    result = resp.get("result", {})
    if result.get("ok"):
        click.secho(
            f"✓ {job} ok ({result.get('duration_s', 0):.2f}s · "
            f"IPC roundtrip {elapsed:.2f}s)",
            fg="green",
        )
        if result.get("result"):
            click.echo(json.dumps(result["result"], indent=2, default=str))
    else:
        click.secho(f"✗ {job} failed: {result.get('error')}", fg="red", err=True)
        sys.exit(1)


@supervisor_group.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Tail -f del log.")
@click.option("-n", default=50, help="Cantidad de líneas iniciales.")
def logs_cmd(follow: bool, n: int) -> None:
    """Tail del supervisor.log."""
    log_path = Path.home() / ".local/share/obsidian-rag/supervisor.log"
    if not log_path.exists():
        click.secho(f"✗ {log_path} no existe — supervisor nunca corrió",
                    fg="red", err=True)
        sys.exit(1)
    args = ["tail"]
    if follow:
        args.append("-f")
    args += ["-n", str(n), str(log_path)]
    try:
        subprocess.run(args, check=False)
    except KeyboardInterrupt:
        pass
