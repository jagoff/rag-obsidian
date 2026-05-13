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
