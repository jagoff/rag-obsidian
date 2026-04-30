"""``rag vault`` — multi-vault management (registry CRUD).

Extraído del monolito ``rag/__init__.py`` el 2026-04-29 como Sub-chunk 1.1
del package-split CLI (ver ``plans/package-split-2026-04-29.md``). Sólo
mueve el grupo y sus 5 subcomandos (`add`, `list`, `use`, `current`,
`remove`) — los helpers compartidos (``_load_vaults_config``,
``_save_vaults_config``, ``_resolve_vault_path``, ``_DEFAULT_VAULT``,
``VAULTS_CONFIG_PATH``) se quedan en ``rag/__init__.py`` porque también
los usan retrieve / index / anticipatory / pendientes. Re-importarlos
acá al estilo de las integraciones evita duplicar lógica.

**Por qué el grupo es standalone (``@click.group()`` y NO ``@cli.group()``):**
Cuando ``rag/__init__.py`` hace ``from rag.cli.vault import ...`` al
final, Python carga el sub-package ``rag.cli`` y rebinda la *attribute*
``rag.cli`` al sub-package (que es ese mismo package), pisando al
``Click.Group`` que vivía como atributo ``rag.cli``. Si en este módulo
intentáramos ``from rag import cli`` dispararíamos
``AttributeError`` (o agarraríamos el sub-package en vez del Group). El
truco consta de dos partes: (1) acá definimos ``vault`` standalone;
(2) ``rag/__init__.py`` guarda el ``cli`` Group antes del import,
re-ata la attribute después, y llama ``cli.add_command(vault)``
explícitamente. Ver el bloque "CLI sub-package re-export shim" del
monolito.

Naming / behavior se preservan **exactos** — los tests existentes en
``tests/test_vaults.py`` deben pasar sin cambios apuntando a
``rag.vault`` (re-exportado al pie de ``rag/__init__.py``).
"""

from __future__ import annotations

import os

import click

# Importamos desde ``rag`` los helpers compartidos + el ``console``
# singleton. Estos atributos están bound MUCHO antes de que el shim al
# pie del monolito dispare el import de este sub-package, así que no hay
# riesgo de circular partial-init. **NO importamos ``cli``**: ver
# docstring del módulo + comentario en ``rag/__init__.py`` (CLI shim).
from rag import (
    _DEFAULT_VAULT,
    _load_vaults_config,
    _save_vaults_config,
    console,
)


@click.group()
def vault():
    """Multi-vault: registrar / cambiar / listar vaults de Obsidian.

    El registry vive en ~/.config/obsidian-rag/vaults.json. Cada vault
    obtiene su propia colección de sqlite-vec automáticamente (namespacing
    por hash del path) — switchear no contamina ni cruza datos.

    Precedencia para resolver el vault activo:
      1. OBSIDIAN_RAG_VAULT env var (override per-invocación, gana siempre).
      2. `vault use <name>` (el "current" del registry, persistente).
      3. Default iCloud Notes (legacy, para usuarios single-vault).
    """


@vault.command("add")
@click.argument("name")
@click.argument("path", type=click.Path(
    exists=True, file_okay=False, dir_okay=True, resolve_path=True,
))
def vault_add(name: str, path: str):
    """Registrar un vault con un nombre. Si es el primero, queda activo."""
    cfg = _load_vaults_config()
    if name in cfg["vaults"] and cfg["vaults"][name] != path:
        console.print(
            f"[yellow]Sobreescribiendo[/yellow] '{name}': "
            f"{cfg['vaults'][name]} → {path}"
        )
    cfg["vaults"][name] = path
    if not cfg["current"]:
        cfg["current"] = name
        marker = " (activo)"
    else:
        marker = ""
    _save_vaults_config(cfg)
    console.print(f"[green]✓[/green] vault [bold]{name}[/bold] → {path}{marker}")


@vault.command("list")
def vault_list():
    """Listar vaults registrados, marcando el activo."""
    cfg = _load_vaults_config()
    if not cfg["vaults"]:
        console.print(
            "[dim]Sin vaults registrados.[/dim] "
            "Usá [bold]rag vault add <name> <path>[/bold] para empezar."
        )
        console.print(f"[dim]Default actual: {_DEFAULT_VAULT}[/dim]")
        return
    cur = cfg["current"]
    env = os.environ.get("OBSIDIAN_RAG_VAULT")
    for name, path in cfg["vaults"].items():
        marker = "[green]→[/green]" if name == cur else "  "
        console.print(f"  {marker} [bold]{name}[/bold]  [dim]{path}[/dim]")
    if env:
        console.print(
            f"\n[yellow]⚠ OBSIDIAN_RAG_VAULT está seteado[/yellow] "
            f"[dim]({env})[/dim] — overridea el registry."
        )


@vault.command("use")
@click.argument("name")
def vault_use(name: str):
    """Cambiar al vault NAME (persistente). Afecta a futuras invocaciones."""
    cfg = _load_vaults_config()
    if name not in cfg["vaults"]:
        registered = ", ".join(cfg["vaults"]) or "(ninguno)"
        console.print(
            f"[red]vault '{name}' no registrado.[/red] "
            f"Registrados: {registered}"
        )
        return
    cfg["current"] = name
    _save_vaults_config(cfg)
    path = cfg["vaults"][name]
    console.print(f"[green]✓[/green] vault activo: [bold]{name}[/bold]  [dim]({path})[/dim]")
    if os.environ.get("OBSIDIAN_RAG_VAULT"):
        console.print(
            "[yellow]⚠[/yellow] OBSIDIAN_RAG_VAULT está seteado — "
            "lo seguirá overrideando hasta que lo desetees."
        )


@vault.command("current")
def vault_current():
    """Mostrar el vault que se va a usar y por qué."""
    env = os.environ.get("OBSIDIAN_RAG_VAULT")
    if env:
        console.print(f"[bold]env[/bold] OBSIDIAN_RAG_VAULT → [cyan]{env}[/cyan]")
        return
    cfg = _load_vaults_config()
    cur = cfg["current"]
    if cur and cur in cfg["vaults"]:
        console.print(
            f"[bold]registry[/bold] [bold]{cur}[/bold] → "
            f"[cyan]{cfg['vaults'][cur]}[/cyan]"
        )
        return
    console.print(f"[bold]default[/bold] → [cyan]{_DEFAULT_VAULT}[/cyan]")


@vault.command("remove")
@click.argument("name")
def vault_remove(name: str):
    """Quitar un vault del registry. NO borra archivos del disco."""
    cfg = _load_vaults_config()
    if name not in cfg["vaults"]:
        console.print(f"[red]vault '{name}' no registrado.[/red]")
        return
    del cfg["vaults"][name]
    if cfg["current"] == name:
        cfg["current"] = next(iter(cfg["vaults"]), None)
    _save_vaults_config(cfg)
    if cfg["current"]:
        console.print(
            f"[green]✓[/green] '{name}' removido. "
            f"Activo ahora: [bold]{cfg['current']}[/bold]"
        )
    else:
        console.print(
            f"[green]✓[/green] '{name}' removido. "
            f"Sin current — caerá al default."
        )


__all__ = [
    "vault",
    "vault_add",
    "vault_list",
    "vault_use",
    "vault_current",
    "vault_remove",
]
