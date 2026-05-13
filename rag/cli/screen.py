"""`rag screen` — comandos relacionados a Peekaboo (captura + observación).

Grupo Click con 2 subcomandos:

- `rag screen capture` (Fase 1) — captura on-demand + caption granite.
  Wrapper sobre [`rag.integrations.peekaboo.capture_and_caption`](../integrations/peekaboo.py).
  Útil para test manual desde terminal (especialmente mientras TCC se concede).
- `rag screen observe-once` (Fase 2) — tick único del observer pasivo.
  Invoca [`rag.integrations.peekaboo.observe_once`](../integrations/peekaboo.py).
  Sin daemon: pensado para invocación manual o desde un plist launchd
  (`com.fer.obsidian-rag-screen-observer`, no shipped todavía).

`rag screen` sin subcomando = `rag screen capture` por compat. Los flags
de capture viven en `capture` y NO en el grupo raíz — `rag screen --json`
no funciona, hay que usar `rag screen capture --json`.

Gate común para ambos: `RAG_PEEKABOO_ENABLE=1`. Para `observe-once` se
requiere además `RAG_SCREEN_OBSERVE=1` (doble opt-in — Fase 2 design).
"""

from __future__ import annotations

import json
import sys

import click

__all__ = ["screen_cli"]


@click.group("screen", invoke_without_command=True)
@click.pass_context
def screen_cli(ctx: click.Context) -> None:
    """Peekaboo screen capture + observación pasiva.

    Sin subcomando equivale a `rag screen capture`.

    Subcomandos:
        capture       — captura on-demand + caption granite (Fase 1).
        observe-once  — tick único del observer pasivo (Fase 2).
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(screen_capture)


@screen_cli.command("capture")
@click.option(
    "--mode", default="frontmost", show_default=True,
    type=click.Choice(["frontmost", "window", "screen", "multi", "menubar"]),
    help="Modo de captura Peekaboo.",
)
@click.option(
    "--app", "app_name", default=None,
    help="App name para mode=window (ej. 'Safari', 'Code', 'Ghostty').",
)
@click.option(
    "--prompt", default=None,
    help="Override caption prompt. Default: prompt estándar (es, ≤80 palabras).",
)
@click.option(
    "--retina", is_flag=True,
    help="Capturar a 2x density.",
)
@click.option(
    "--keep", is_flag=True,
    help="No borrar la PNG temporal. Imprime el path en stdout para inspección.",
)
@click.option(
    "--json", "as_json", is_flag=True,
    help="Salida machine-readable (dict completo del wrapper).",
)
def screen_capture(
    mode: str,
    app_name: str | None,
    prompt: str | None,
    retina: bool,
    keep: bool,
    as_json: bool,
) -> None:
    """Capturar pantalla y describirla con granite (MLX-VLM).

    Requiere:
        - `RAG_PEEKABOO_ENABLE=1` (default OFF).
        - Peekaboo CLI instalado (`brew install steipete/tap/peekaboo`).
        - Screen Recording permission concedido al terminal en
          System Settings → Privacy & Security → Screen & System Audio Recording.

    Ejemplos:
        rag screen capture
        rag screen capture --app Safari --mode window --retina
        rag screen capture --keep --json
        rag screen capture --prompt "describí sólo el código visible"
    """
    from rag.integrations.peekaboo import capture_and_caption  # noqa: PLC0415

    if mode == "window" and not app_name:
        click.echo("ERROR: --mode window requiere --app NAME", err=True)
        sys.exit(2)

    out = capture_and_caption(
        mode=mode,
        app=app_name,
        prompt=prompt,
        retina=retina,
        keep_image=keep,
    )

    if as_json:
        click.echo(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(0 if out.get("ok") else 1)

    if not out.get("ok"):
        err = out.get("error") or "unknown"
        click.echo(f"[fail] {err}  ({out['took_ms']}ms)", err=True)
        if err.startswith("tcc_denied"):
            click.echo(
                "\nFix: System Settings → Privacy & Security → Screen & System "
                "Audio Recording → enable terminal app (Ghostty/Terminal/iTerm). "
                "Reiniciar terminal después del toggle.",
                err=True,
            )
        elif err == "peekaboo_not_installed":
            click.echo("Fix: brew install steipete/tap/peekaboo", err=True)
        elif err == "peekaboo_disabled":
            click.echo("Fix: export RAG_PEEKABOO_ENABLE=1", err=True)
        sys.exit(1)

    caption = out.get("caption", "").strip() or "(empty caption)"
    click.echo(caption)
    click.echo(f"\n[ok] mode={out['mode']} took={out['took_ms']}ms", err=True)
    if keep and out.get("image_path"):
        click.echo(f"[png] {out['image_path']}", err=True)


@screen_cli.command("observe-once")
@click.option(
    "--mode", default="frontmost", show_default=True,
    type=click.Choice(["frontmost", "window", "screen"]),
    help="Modo Peekaboo (passive observer típicamente frontmost).",
)
@click.option(
    "--dedup-seconds", default=60, show_default=True, type=int,
    help="Ventana de dedup titular (mismo app + título) en segundos.",
)
@click.option(
    "--json", "as_json", is_flag=True,
    help="Salida machine-readable (dict completo de observe_once).",
)
@click.option(
    "--force", is_flag=True,
    help="Bypassa el gate `RAG_SCREEN_OBSERVE` para test manual. NO usar en daemon.",
)
def screen_observe_once(
    mode: str,
    dedup_seconds: int,
    as_json: bool,
    force: bool,
) -> None:
    """Tick único del observer pasivo (Fase 2).

    Hace una pasada de capture + caption + insert con dedup titular y
    skip por quiet hours / app denylist. Pensado para invocación desde
    un plist launchd (cada 10min) o para test manual con `--force`.

    Requiere:
        - `RAG_PEEKABOO_ENABLE=1` — binario activado.
        - `RAG_SCREEN_OBSERVE=1` — daemon activado (o pasar `--force`).
        - Screen Recording TCC concedido al terminal.

    Otros knobs:
        - `RAG_SCREEN_QUIET_HOURS="22:00-07:00"` — ventana sin captura.
        - `RAG_SCREEN_APP_DENY="1Password,Banking"` — apps nunca observadas.

    Ejemplos:
        rag screen observe-once
        rag screen observe-once --force --json
        rag screen observe-once --dedup-seconds 300
    """
    import os  # noqa: PLC0415
    from rag.integrations.peekaboo import observe_once  # noqa: PLC0415

    if force:
        os.environ["RAG_SCREEN_OBSERVE"] = "1"

    out = observe_once(mode=mode, dedup_seconds=max(1, dedup_seconds))

    if as_json:
        click.echo(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(0 if out.get("ok") else 1)

    if out.get("ok"):
        click.echo(
            f"[ok] obs#{out['observation_id']} app={out.get('app_name') or '-'}"
            f" title={out.get('window_title') or '-'!r} took={out['took_ms']}ms"
        )
        caption = (out.get("caption") or "").strip()
        if caption:
            preview = caption if len(caption) <= 200 else caption[:200] + "…"
            click.echo(f"\n{preview}")
        sys.exit(0)

    reason = out.get("skipped_reason")
    err = out.get("error")
    if reason:
        click.echo(f"[skip] {reason}  ({out['took_ms']}ms)", err=True)
        if reason == "observe_disabled":
            click.echo("Fix: export RAG_SCREEN_OBSERVE=1 (o usar --force).", err=True)
        elif reason == "peekaboo_disabled":
            click.echo("Fix: export RAG_PEEKABOO_ENABLE=1.", err=True)
        elif reason == "quiet_hours":
            click.echo(
                f"Quiet hours activos (RAG_SCREEN_QUIET_HOURS="
                f"{os.environ.get('RAG_SCREEN_QUIET_HOURS', '')!r}). Esperá fuera de la ventana.",
                err=True,
            )
        sys.exit(0)  # skip no es error
    if err:
        click.echo(f"[fail] {err}  ({out['took_ms']}ms)", err=True)
        if err.startswith("tcc_denied"):
            click.echo(
                "\nFix: System Settings → Privacy & Security → Screen & System "
                "Audio Recording → enable terminal app. Reiniciar terminal después.",
                err=True,
            )
        sys.exit(1)
    click.echo(f"[unknown] {out}", err=True)
    sys.exit(1)


@screen_cli.command("enable")
@click.option(
    "--no-setup", is_flag=True,
    help="Saltea `rag setup` + kickstart automáticos. Util para test.",
)
@click.option(
    "--no-tcc-helper", is_flag=True,
    help="Saltea apertura automática de System Settings (TCC Screen Recording).",
)
def screen_enable(no_setup: bool, no_tcc_helper: bool) -> None:
    """Activar el observer pasivo del Peekaboo (Fase 2g).

    Todo en un comando:
        1. Toca el state file `screen_observe_enabled`.
        2. Corre `rag setup` para regenerar el plist con las env vars.
        3. Hace `launchctl kickstart -k` del supervisor para recargarlo.
        4. Abre System Settings en Screen & System Audio Recording + copia
           al clipboard el path del Python que el supervisor usa, para que
           lo pegues en el "+ Files..." con un Cmd+V (1 click + 1 paste).

    TCC requirement (macOS SIP-protected, sin API automatable):
        El supervisor launchd invoca `peekaboo` como subprocess de
        `.venv/bin/python`. macOS TCC chequea el responsible-process chain
        y NO hereda el grant de Ghostty. Hay que agregar manualmente el
        binario Python (resolviendo el symlink al cpython real) a Screen
        Recording. Una sola vez — después corre cada 15min sin intervención.

    Equivalente al opt-in de mood: `rag mood enable`.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    from rag.integrations.peekaboo import (  # noqa: PLC0415
        _OBSERVE_STATE_FILE, _observe_state_set,
    )

    _observe_state_set(True)
    click.echo(f"✓ Screen observer activado. State file: {_OBSERVE_STATE_FILE}")

    if no_setup:
        click.echo(
            "\n(--no-setup) Saltea regeneración del plist + kickstart. Correr "
            "después:\n"
            "  rag setup && launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-supervisor\n"
        )
    else:
        click.echo("\n→ Corriendo `rag setup` (regenera plist con env vars Peekaboo)...")
        rc = subprocess.run(["rag", "setup"], check=False).returncode
        if rc != 0:
            click.echo(f"  [warn] rag setup exit={rc} — revisar manualmente.", err=True)
        else:
            click.echo("  ✓ plist regenerado.")

        uid = os.getuid()
        click.echo(f"\n→ Reload del supervisor (launchctl kickstart -k gui/{uid}/com.fer.obsidian-rag-supervisor)...")
        rc = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/com.fer.obsidian-rag-supervisor"],
            check=False, capture_output=True, text=True,
        ).returncode
        if rc != 0:
            click.echo(f"  [warn] kickstart exit={rc} — revisar `launchctl print`.", err=True)
        else:
            click.echo("  ✓ supervisor reloaded.")

    if no_tcc_helper:
        click.echo(
            "\n(--no-tcc-helper) Saltea apertura de System Settings. "
            "Agregar manualmente a Screen Recording:\n"
            f"  {_resolve_supervisor_python()}"
        )
        return

    python_real = _resolve_supervisor_python()
    click.echo(
        "\n=== Último paso — TCC Screen Recording ===\n"
        "macOS no permite agregar binarios al TCC programáticamente (SIP-protected).\n"
        "Te abro System Settings en el pane correcto y copio el path al clipboard.\n"
        "Pegalo con Cmd+V en el diálogo 'Files...' después del '+'.\n"
        f"\nPath del Python del supervisor:\n  {python_real}\n"
    )

    # Copy path to clipboard (pbcopy is macOS-native + no-deps).
    try:
        subprocess.run(["pbcopy"], input=python_real.encode(), check=False)
        click.echo("✓ Path copiado al clipboard (Cmd+V para pegar).")
    except FileNotFoundError:
        click.echo("[warn] pbcopy no disponible — copialo a mano.", err=True)

    # Open System Settings at the Screen Recording pane.
    settings_url = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
    try:
        subprocess.run(["open", settings_url], check=False, timeout=5)
        click.echo("✓ System Settings abierto en Screen Recording.")
        click.echo(
            "\nFlow esperado (1 vez, ~30s):\n"
            "  1. En el pane, click '+' (o 'Add' → Files...).\n"
            "  2. Cmd+Shift+G, pegar (Cmd+V), Enter.\n"
            "  3. Seleccionar python3.13 → click 'Open'.\n"
            "  4. Toggle ON.\n"
            "  5. (Opcional) re-kickstart si te pide: launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-supervisor\n"
            "\nVerificación final (debería retornar ok: true):\n"
            "  echo '{\"action\":\"run\",\"job\":\"screen_observer\"}' | nc -U ~/.local/share/obsidian-rag/supervisor.sock"
        )
    except Exception as exc:
        click.echo(f"[warn] no pude abrir System Settings: {exc}", err=True)


def _resolve_supervisor_python() -> str:
    """Path absoluto del binario Python real que launchd ejecuta para el
    supervisor (resuelve symlinks del venv). Es el path que TCC necesita
    en su allowlist — TCC chequea el inode final, no el symlink."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    venv_python = Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "python"
    try:
        return os.path.realpath(venv_python)
    except OSError:
        return str(venv_python)


@screen_cli.command("disable")
def screen_disable() -> None:
    """Desactivar el observer pasivo (Fase 2g).

    Borra el state file. Próximo `rag setup` regenerá el plist del
    supervisor SIN las env vars Peekaboo — el `screen_observer_job` corre
    pero sale en `observe_disabled` sin tocar Peekaboo ni granite.

    Las rows existentes en `rag_screen_observations` NO se borran
    automáticamente — quedan disponibles para el último `mirror` /
    `today` / `digest` mientras se rotan via `run_maintenance` (retention 7d).
    """
    from rag.integrations.peekaboo import (  # noqa: PLC0415
        _OBSERVE_STATE_FILE, _observe_state_set,
    )
    existed = _OBSERVE_STATE_FILE.is_file()
    _observe_state_set(False)
    if existed:
        click.echo(f"✓ Screen observer desactivado. Borré: {_OBSERVE_STATE_FILE}")
    else:
        click.echo(f"(ya estaba desactivado — {_OBSERVE_STATE_FILE} no existía)")
    click.echo(
        "\nProximos pasos:\n"
        "  1. rag setup     # regenerá el plist del supervisor sin las env vars\n"
        "  2. launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-supervisor\n"
        "                   # recargá el supervisor\n"
    )


@screen_cli.command("status")
def screen_status() -> None:
    """Estado del observer pasivo + smoke checks rápidos.

    Muestra:
        - State file path + existencia.
        - Env vars RAG_PEEKABOO_ENABLE / RAG_SCREEN_OBSERVE / RAG_SCREEN_QUIET_HOURS /
          RAG_SCREEN_APP_DENY en el proceso actual.
        - Última row en rag_screen_observations + age.
        - Count de rows últimos 24h / 7d.
        - Peekaboo binary path (resolved).
    """
    import os  # noqa: PLC0415
    import sqlite3  # noqa: PLC0415
    import time  # noqa: PLC0415
    from rag.integrations.peekaboo import (  # noqa: PLC0415
        _OBSERVE_STATE_FILE, _observe_state_enabled, _resolve_binary,
    )

    click.echo("=== Peekaboo screen observer status ===")
    click.echo(f"  state_file       {_OBSERVE_STATE_FILE}")
    click.echo(f"  state_enabled    {_observe_state_enabled()}")
    click.echo(f"  peekaboo bin     {_resolve_binary() or '(not installed)'}")
    click.echo("  env vars (en este proceso):")
    for k in ("RAG_PEEKABOO_ENABLE", "RAG_SCREEN_OBSERVE",
              "RAG_SCREEN_QUIET_HOURS", "RAG_SCREEN_APP_DENY"):
        click.echo(f"    {k:24} {os.environ.get(k) or '(unset)'}")

    try:
        from rag import DB_PATH  # noqa: PLC0415
        db = DB_PATH / "telemetry.db"
        con = sqlite3.connect(str(db), timeout=5.0)
        try:
            now_ts = int(time.time())
            cnt_24h = con.execute(
                "SELECT COUNT(*) FROM rag_screen_observations WHERE ts >= ?",
                (now_ts - 24 * 3600,),
            ).fetchone()[0]
            cnt_7d = con.execute(
                "SELECT COUNT(*) FROM rag_screen_observations WHERE ts >= ?",
                (now_ts - 7 * 86400,),
            ).fetchone()[0]
            row = con.execute(
                "SELECT id, ts, app_name, window_title, substr(caption,1,80) "
                "FROM rag_screen_observations ORDER BY ts DESC LIMIT 1",
            ).fetchone()
            click.echo(f"  rows last 24h    {cnt_24h}")
            click.echo(f"  rows last 7d     {cnt_7d}")
            if row:
                rid, ts, app, title, cap = row
                age_min = max(0, (now_ts - int(ts)) // 60)
                click.echo(
                    f"  last row         #{rid} app={app or '-'} "
                    f"title={title or '-'!r} age={age_min}min"
                )
                click.echo(f"                   caption: {(cap or '').strip()}")
            else:
                click.echo("  last row         (none)")
        finally:
            con.close()
    except Exception as exc:
        click.echo(f"  (db query failed: {exc})", err=True)
