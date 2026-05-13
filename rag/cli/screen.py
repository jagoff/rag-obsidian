"""`rag screen` — captura on-demand + caption granite del frontmost.

Wrapper CLI sobre [`rag.integrations.peekaboo.capture_and_caption`](../integrations/peekaboo.py).

Útil para probar la integración Peekaboo desde terminal sin pasar por MCP
(necesario mientras TCC Screen Recording se concede al terminal). Mismo
gate `RAG_PEEKABOO_ENABLE=1`, mismos errores typeados, misma privacy
(PNG temporal en /tmp con chmod 0600, borrado por default).

## UX

- Sin flags: captura el frontmost, captiona, imprime caption + took_ms.
- `--keep`: deja la PNG en disco, imprime el path para inspección.
- `--json`: salida machine-readable (mismo dict que devuelve el wrapper).
- `--app NAME`: usar mode=window + --app NAME.
"""

from __future__ import annotations

import json
import sys

import click

__all__ = ["screen_cli"]


@click.command("screen")
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
def screen_cli(
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
        rag screen
        rag screen --app Safari --mode window --retina
        rag screen --keep --json
        rag screen --prompt "describí sólo el código visible"
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
            click.echo(
                "Fix: brew install steipete/tap/peekaboo",
                err=True,
            )
        elif err == "peekaboo_disabled":
            click.echo(
                "Fix: export RAG_PEEKABOO_ENABLE=1",
                err=True,
            )
        sys.exit(1)

    caption = out.get("caption", "").strip() or "(empty caption)"
    click.echo(caption)
    click.echo(f"\n[ok] mode={out['mode']} took={out['took_ms']}ms", err=True)
    if keep and out.get("image_path"):
        click.echo(f"[png] {out['image_path']}", err=True)
