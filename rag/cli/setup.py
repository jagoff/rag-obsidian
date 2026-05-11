"""`rag setup` / `rag stop` / `rag start` — daemon lifecycle commands.

Phase 2d del daemon refactor (audit 2026-05-09: `99-obsidian/99-AI/system/
daemon-refactor-2026-05-09/control-plane-audit.md`). Extraído de
`rag/__init__.py` (lineas ~46059–46916, ~787 LOC) post-Phase 2c (que ya
había movido los 7 helpers stdlib-only a `rag/cli/daemons_control.py`).

## Qué vive acá

- `_setup_install(rag_bin, *, remove, only_labels)` — orquesta plist regen
  + bootout + bootstrap. Llamada por `setup` y `start`.
- `setup` (Click cmd) — instala/desinstala los managed daemons del
  `_services_spec`. Idempotente.
- `stop` (Click cmd) — bootout + plist archiving de TODO el stack
  (managed + manual + RagNet + cloudflared). Default omite qdrant
  (compartido con mem-vault).
- `start` (Click cmd) — arranque mínimo viable (5 daemons core) con
  catch-up index si hay backlog. `--full` instala los 30 del spec.
- `_run_catch_up_index(ctx)` — helper extraído pre-Phase 2d (commit
  `3c67c8d`) para que `start` no atara `ctx.invoke(index)` al closure
  de Click context, destrabando esta extracción.

## Lazy imports

Las funciones leen símbolos de `rag.__init__` via lazy `from rag import
X` adentro del cuerpo (NO module-level). Esto preserva el patrón ya
establecido por los otros sub-módulos extraídos (`dead_notes`,
`maintenance`, `daemons_control`) y mantiene la compat con tests que
hacen `monkeypatch.setattr(rag, "_LAUNCH_AGENTS_DIR", tmp_path)` —
el monkeypatch sobre el binding de `rag.__init__` flow-throughea al
`from rag import _LAUNCH_AGENTS_DIR` adentro del callee.

## Constantes locales

`_RAG_NET_LABELS`, `_QDRANT_LABELS`, `_MINIMAL_MANAGED_LABELS` viven
top-level acá (datos puros, sin refs dinámicos a `rag`). Se re-exportan
en `rag/__init__.py` para que `rag._MINIMAL_MANAGED_LABELS` siga
respondiendo (tests `test_launchd_plists.py:154` lo leen así).

## Re-export

`rag/__init__.py` registra los Click commands via:

    from rag.cli.setup import (
        _MINIMAL_MANAGED_LABELS,
        _QDRANT_LABELS,
        _RAG_NET_LABELS,
        _run_catch_up_index,
        _setup_install,
        setup,
        start,
        stop,
    )
    cli.add_command(setup)
    cli.add_command(start)
    cli.add_command(stop)

Los `rag.setup`, `rag.start`, `rag.stop`, `rag._setup_install`,
`rag._run_catch_up_index` siguen accesibles para los tests que importan
`rag.X` o hacen `monkeypatch.setattr(rag, "_setup_install", mock)`.
"""

from __future__ import annotations

import datetime as _dt
import shutil as _shutil
import subprocess
import time
from pathlib import Path

import click

__all__ = [
    "_MINIMAL_MANAGED_LABELS",
    "_QDRANT_LABELS",
    "_RAG_NET_LABELS",
    "_run_catch_up_index",
    "_setup_install",
    "setup",
    "start",
    "stop",
]


# ── Daemons RagNet (whatsapp-*) y dep externa qdrant ──────────────────────
#
# Estos labels NO están en `_services_spec` (vienen del repo `whatsapp-listener`
# y de instaladores externos), pero `rag stop` necesita poder bootoutearlos
# para parar "todo el sistema completo". Hardcodearlos acá es el approach
# más simple — son estables y bien conocidos. Si en el futuro algún label
# cambia, el bootout falla con exit=3 (no encontrado) y lo reportamos como
# "ya estaba parado", no como error.
#
# El detalle por tier:
#  - rag-net  → la cara de WhatsApp del sistema (RagNet). Drafts, listener,
#               bridge, healthcheck, vault-sync. Default ON en `rag stop`
#               porque sin RagNet la UX del sistema queda a medias (web sí,
#               WhatsApp no).
#  - qdrant   → vector store de mem-vault. Default OFF: compartido con
#               mem-vault y otros agentes locales.
_RAG_NET_LABELS: tuple[str, ...] = (
    "com.fer.whatsapp-bridge",
    "com.fer.whatsapp-listener",
    "com.fer.whatsapp-listener-healthcheck",
    "com.fer.whatsapp-vault-sync",
)
_QDRANT_LABELS: tuple[str, ...] = ("com.fer.qdrant",)

# Set mínimo viable que `rag start --minimal` (default ON desde 2026-05-08)
# instala. Sin briefs, sin learning loops, sin productividad pasiva — solo
# lo necesario para que el chat web funcione + watch para que el corpus
# quede al día + watchdog/wake-hook para self-healing post Mac wake +
# maintenance para WAL checkpoint y rotación de logs. Los managed que
# queden afuera reciben el mismo trato que `_DEPRECATED_LABELS` durante el
# install (bootout + unlink). `rag start --full` instala los 30.
_MINIMAL_MANAGED_LABELS: frozenset[str] = frozenset(
    {
        # Post-supervisor refactor 2026-05-09: minimal === all managed.
        # `daemon-watchdog` + `wake-hook` reemplazados por supervisor
        # internals (APScheduler coalesce + misfire_grace_time). `maintenance`
        # corre adentro del supervisor como cron job (jobs/nightly.py).
        "com.fer.obsidian-rag-supervisor",
        "com.fer.obsidian-rag-watch",
        "com.fer.obsidian-rag-web",
    }
)


# _loaded_launchd_labels, _bootout_label, _bootstrap_label viven en
# rag/cli/daemons_control.py (Phase 2c.2 2026-05-09). Se acceden via
# lazy `from rag import ...` adentro de los cuerpos abajo (preservan
# el shim del namespace `rag` que los expone como `rag._foo`).


import os  # noqa: E402  (local symbol; el código abajo lo usa)


def _find_cloudflare_tunnel_labels() -> tuple[str, ...]:
    """Buscar dinámicamente plists de cloudflare en ~/Library/LaunchAgents/.

    Glob: com.fer.obsidian-rag-cloudflare-*.plist
    Retorna tuple de labels (sin .plist).
    """
    from rag import _LAUNCH_AGENTS_DIR  # noqa: PLC0415

    labels = []
    for plist_path in _LAUNCH_AGENTS_DIR.glob("com.fer.obsidian-rag-cloudflare-*.plist"):
        label = plist_path.stem  # Remove .plist
        labels.append(label)
    return tuple(sorted(labels))


def _cleanup_staled_locks() -> None:
    """Limpiar lock files estaled después de bootout de daemons.

    Verifica si el PID del owner del lock sigue vivo. Si no, borra el lock.
    Archivos: anticipate.lock, reindex.lock, supervisor.sock, supervisor.pid.
    """
    lock_files = [
        Path.home() / ".local" / "share" / "obsidian-rag" / "anticipate.lock",
        Path.home() / ".local" / "share" / "obsidian-rag" / "reindex.lock",
        Path.home() / ".local" / "share" / "obsidian-rag" / "supervisor.sock",
        Path.home() / ".local" / "share" / "obsidian-rag" / "supervisor.pid",
    ]

    for lock_path in lock_files:
        if not lock_path.exists():
            continue
        try:
            # Para .pid, leer el PID
            if lock_path.name.endswith(".pid"):
                pid_str = lock_path.read_text().strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    # Verificar si el proceso sigue vivo (kill 0 = no-op, solo checkea permisos)
                    os.kill(pid, 0)
                    # Si no levantó excepción, el proceso está vivo — no borrar
                    continue
                # Si el contenido no es un PID válido, borrar el lock
            # Para .lock y .sock, simplemente borrar si están viejos o si la info es corrupta
            lock_path.unlink()
        except ProcessLookupError:
            # PID no existe — borrar el lock
            try:
                lock_path.unlink()
            except OSError:
                pass
        except (OSError, ValueError):
            # Error de permisos o parsing — dejar el lock (seguro)
            pass


def _print_access_urls() -> None:
    """Mostrar URLs accesibles, health status y endpoints del web server."""
    from rag import console  # noqa: PLC0415
    from rag.cli._start_helpers import (  # noqa: PLC0415
        get_cloudflared_url,
        get_lan_ip,
        health_probe_web,
        read_plist_env_var,
    )

    console.print("[bold cyan]▸ acceso[/bold cyan]")

    # ── Health probe ─────────────────────────────────────────────────────
    ok, latency_ms = health_probe_web()
    if ok:
        console.print(f"  [green]✓[/green] web respondió en {latency_ms} ms")
    else:
        console.print(
            "[yellow]![/yellow] web no respondió todavía (puede estar en warmup MLX, esperá ~30s)"
        )

    # ── URLs base ────────────────────────────────────────────────────────
    port = 8765
    localhost_base = f"http://127.0.0.1:{port}"

    # Detectar si OBSIDIAN_RAG_BIND_HOST=0.0.0.0 para agregar LAN IP
    lan_ip = None
    bind_host = read_plist_env_var("com.fer.obsidian-rag-web", "OBSIDIAN_RAG_BIND_HOST")
    if bind_host == "0.0.0.0":
        lan_ip = get_lan_ip()

    # ── Endpoints ────────────────────────────────────────────────────────
    endpoints = [
        ("chat", "/chat"),
        ("dashboard", "/dashboard"),
        ("atlas", "/atlas"),
        ("mirror", "/mirror"),
        ("memory", "/memory"),
    ]

    for name, path in endpoints:
        url_localhost = f"{localhost_base}{path}"
        console.print(f"  [dim]·[/dim] [cyan]{name:<12}[/cyan] : {url_localhost}", end="")
        if lan_ip:
            url_lan = f"http://{lan_ip}:{port}{path}"
            console.print(f"  ([dim]LAN[/dim]: {url_lan})")
        else:
            console.print()

    # ── Tunnel URL (si existe) ───────────────────────────────────────────
    tunnel_url = get_cloudflared_url()
    if tunnel_url:
        console.print(f"  [dim]·[/dim] [cyan]{'tunnel':<12}[/cyan] : {tunnel_url}")

    # ── Admin token ──────────────────────────────────────────────────────
    admin_token_path = Path.home() / ".config" / "obsidian-rag" / "admin_token.txt"
    admin_status = (
        "[green]existente[/green]"
        if admin_token_path.exists()
        else "[yellow]se genera al primer hit admin[/yellow]"
    )
    console.print(
        f"  [dim]·[/dim] [cyan]{'admin token':<12}[/cyan] : {admin_token_path} ({admin_status})"
    )

    # ── MCP endpoint ─────────────────────────────────────────────────────
    console.print("[bold cyan]▸ herramientas[/bold cyan]")
    console.print(
        "  [dim]·[/dim] MCP server  : [cyan]obsidian-rag-mcp[/cyan] (entry point para Claude Code / Devin)"
    )

    # ── Verificación recomendada ────────────────────────────────────────
    console.print()
    console.print("[dim]Verificar:[/dim]")
    console.print("  [dim]·[/dim] [cyan]rag daemons status[/cyan]    — health del control plane")
    console.print(
        "  [dim]·[/dim] [cyan]tail -f ~/.local/share/obsidian-rag/web.log[/cyan]  — FastAPI server"
    )
    console.print("  [dim]·[/dim] [cyan]curl -s http://127.0.0.1:8765/health | jq[/cyan]")


def _setup_install(
    rag_bin: str,
    *,
    remove: bool = False,
    only_labels: frozenset[str] | None = None,
) -> None:
    """Lógica interna de install/uninstall de los services managed.

    Llamada por el Click command `setup` y por `rag start` (que pasa
    `only_labels=_MINIMAL_MANAGED_LABELS` cuando el usuario pide `--minimal`).

    Cuando `only_labels` no es None, los labels de `_services_spec()` que
    NO estén en el set reciben el mismo trato que `_DEPRECATED_LABELS`:
    bootout + unlink de disco. Semánticamente: "para este run, lo no
    listado está deprecated".
    """
    from rag import (  # noqa: PLC0415
        _DEPRECATED_LABELS,
        _INSTALL_GATES,
        _LAUNCH_AGENTS_DIR,
        _RAG_LOG_DIR,
        _services_spec,
        console,
    )

    _LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    _RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cleanup de labels deprecated ─────────────────────────────────────
    # Bootout + unlink plists que fueron consolidados / removidos de
    # `_services_spec`. Sin este pass, el plist viejo sigue cargado en
    # paralelo con su reemplazo → doble envío / race conditions.
    # Bug Hunt 2026-05-08 H Tel 3: `launchctl load`/`unload` están
    # deprecated desde macOS Ventura y en macOS Sequoia (25.5.0, OS
    # actual) producen warnings o silently fail. El resto del control-
    # plane (`_execute_reconcile_action`, `_bootstrap_brief_plist`) ya
    # usa `bootstrap`/`bootout` correctamente — sólo `_setup_install`
    # quedó en la API vieja. Migración: bootout acepta exit-code 3
    # (no estaba cargado, OK); bootstrap acepta 37 (ya estaba cargado).
    _uid = os.getuid()
    _domain = f"gui/{_uid}"

    def _bootout(plist_path: Path, label: str) -> None:
        """`launchctl bootout` + wait until el daemon está REALMENTE gone.

        `launchctl bootout` devuelve sync (el subprocess termina) pero el
        teardown interno de launchd queda en flight — el daemon puede seguir
        listado por 1-3s mientras drena. Si bootstrappeamos en esa ventana,
        macOS Sequoia devuelve EIO 5 ("Input/output error") porque ve el
        slot ocupado por el daemon viejo en death throes.

        Fix: poll `launchctl list` hasta que el label DESAPAREZCA (max 3s).
        Eso garantiza que el bootstrap subsiguiente trabaje contra un slot
        limpio. Si el label nunca desaparece (daemon zombie), seguimos
        adelante igual — el bootstrap fallará ruidosamente y el caller lo
        verá.
        """
        from rag.cli.daemons_control import _loaded_launchd_labels  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        try:
            subprocess.run(
                ["launchctl", "bootout", f"{_domain}/{label}"],
                check=False,
                capture_output=True,
            )
        except Exception:
            return
        # Poll hasta 3s o hasta que el label desaparezca.
        deadline = _time.time() + 3.0
        while _time.time() < deadline:
            if label not in _loaded_launchd_labels():
                return
            _time.sleep(0.1)

    for deprecated in _DEPRECATED_LABELS:
        dep_plist = _LAUNCH_AGENTS_DIR / f"{deprecated}.plist"
        if dep_plist.exists():
            _bootout(dep_plist, deprecated)
            dep_plist.unlink()
            console.print(
                f"[dim]·[/dim] deprecated: {deprecated} [dim](bootouted + removido del disco)[/dim]"
            )

    for label, fname, content in _services_spec(rag_bin):
        plist_path = _LAUNCH_AGENTS_DIR / fname
        # Always bootout first so a stale version doesn't linger after reinstall.
        if plist_path.exists():
            _bootout(plist_path, label)
        if remove:
            if plist_path.exists():
                plist_path.unlink()
                console.print(f"[green]✓[/green] removido: {label}")
            else:
                console.print(f"[dim]· no estaba instalado: {label}[/dim]")
            continue
        # Filtro `only_labels`: si está set y este label no está adentro,
        # tratarlo como deprecated (bootout ya hecho arriba + unlink).
        if only_labels is not None and label not in only_labels:
            if plist_path.exists():
                plist_path.unlink()
                console.print(
                    f"[dim]·[/dim] skip {label} "
                    f"[dim](fuera del set mínimo — plist removido del disco)[/dim]"
                )
            continue
        # Install gates: si falta un pre-requisito (credencial / opt-in file),
        # skipear el install con nota. El plist cargado sin su pre-req solo
        # exit-loopea cada cadencia polucionando logs sin hacer nada útil
        # ("deuda silenciosa" — el `daemons status` lo marca ok:false pero
        # el user no lo nota hasta que abre health). Mejor no cargarlo;
        # re-corriendo `rag setup` después del pre-req se activa.
        #
        # Si el plist YA existía en disco de un `rag setup` previo (pre-gate)
        # y el gate ahora no pasa (ej. el user borró el token), además de
        # skipear el re-install, removemos el plist viejo del disco — sino
        # el bootout de arriba ya lo descargó pero macOS lo re-loadea al
        # próximo login. Una limpieza honesta.
        gate = _INSTALL_GATES.get(label)
        if gate is not None:
            check_fn, hint = gate
            if not check_fn():
                if plist_path.exists():
                    plist_path.unlink()
                    console.print(
                        f"[yellow]·[/yellow] skip {label}: {hint} "
                        f"[dim](plist viejo removido del disco)[/dim]"
                    )
                else:
                    console.print(f"[yellow]·[/yellow] skip {label}: {hint}")
                continue
        plist_path.write_text(content, encoding="utf-8")
        try:
            # Bug Hunt H Tel 3: `bootstrap` reemplaza `load`. Exit 37
            # = ya estaba cargado (no es error real, idempotencia OK).
            #
            # Race "bootout-then-bootstrap" (2026-05-10): launchctl bootout
            # devuelve sync, pero el teardown interno de launchd queda en
            # flight. El bootstrap inmediato puede caer con EIO 5
            # ("Input/output error"). Recovery: si exit≠0/37 y el label NO
            # aparece cargado todavía, retry una vez tras 0.5s.
            from rag.cli.daemons_control import _loaded_launchd_labels  # noqa: PLC0415
            import time  # noqa: PLC0415

            def _bootstrap_once() -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["launchctl", "bootstrap", _domain, str(plist_path)],
                    check=False,
                    capture_output=True,
                )

            res = _bootstrap_once()
            if res.returncode not in (0, 37):
                # IMPORTANTE: NO trustear `_loaded_launchd_labels()` acá
                # porque venimos de bootouttear este mismo label hace ~ms.
                # El daemon listado puede ser el viejo en death throes,
                # NO uno nuevo recién bootstrappeado. Bug detectado
                # 2026-05-10: el check de "ya cargado" reportaba ✓ falso
                # cuando el bootstrap realmente había fallado, dejando
                # los daemons sin levantar.
                #
                # Recovery correcto: sleep + retry bootstrap. Si después del
                # retry sigue fallando Y el label aparece cargado, ahí sí
                # confiar (el daemon nuevo respondió al retry, el listado
                # es legítimo). El _bootout() previo ya esperó a que el
                # daemon viejo desaparezca, así que en este punto no debería
                # haber zombies.
                time.sleep(0.5)
                res = _bootstrap_once()
                if res.returncode not in (0, 37):
                    if label in _loaded_launchd_labels():
                        res.returncode = 0

            if res.returncode in (0, 37):
                console.print(f"[green]✓[/green] cargado: [bold]{label}[/bold]")
            else:
                stderr = res.stderr.decode(errors="ignore") if res.stderr else ""
                console.print(
                    f"[red]✗[/red] falló cargar {label} (exit={res.returncode}): {stderr.strip()}"
                )
        except Exception as exc:
            console.print(f"[red]✗[/red] falló cargar {label}: {exc}")


@click.command("setup")
@click.option("--remove", is_flag=True, help="Desinstalar los servicios en lugar de instalarlos")
def setup(remove: bool):
    """Instalar (o desinstalar) los servicios launchd que mantienen el RAG vivo
    sin intervención: `rag watch` (auto-reindex) y `rag digest` (semanal).
    Idempotente — re-correr lo recarga.
    """
    from rag import (  # noqa: PLC0415
        _CONTACTS_PHONE_INDEX_PATH,
        _CONTACTS_PHONE_INDEX_TTL_S,
        _RAG_LOG_DIR,
        _load_contacts_phone_index,
        _rag_binary,
        _silent_log,
        console,
    )

    rag_bin = _rag_binary()
    if not Path(rag_bin).is_file():
        console.print(f"[red]No encuentro el binario `rag`:[/red] {rag_bin}")
        # Incluir el extra `[entities]` por default — sin él, gliner no se
        # instala en el uv tool venv y los ingesters loggean
        # `[feature: entities] dep \`gliner\` not available` cada corrida +
        # la feature de entity-aware retrieval queda desactivada
        # silenciosamente. Aprendido el 2026-04-25 en una sesión donde los
        # 5 *.error.log se llenaron de ese warning durante días sin que
        # nadie lo notara. Si querés MUY mínimo (e.g. CI sin GPU/MPS),
        # corré `uv tool install --reinstall --editable .` (sin extra) y
        # exportá `RAG_EXTRACT_ENTITIES=0` para silenciar.
        console.print(
            "[dim]Instalá primero: uv tool install --reinstall --editable '.[entities]'[/dim]"
        )
        return
    _setup_install(rag_bin, remove=remove, only_labels=None)

    if not remove:
        # Pre-build Apple Contacts disk cache. La primera invocación del
        # resolver `_resolve_sender_to_name` dispara `_load_contacts_phone_index()`
        # que dumpea el address book completo via osascript — ~85s para ~350
        # contactos en el host del autor. Si `rag setup` no lo calienta, ese
        # costo lo paga la primera query cross-source post-install (suele ser
        # `rag serve` handling un WA inbound). Pre-buildearlo acá mueve el
        # hit al setup, donde el user ya está esperando.
        #
        # Silent-fail: si Contacts.app está restringido o el dump falla, el
        # resolver degrada al mask fallback (`…XXXX`) — no bloquea el setup.
        # Solo corre si el cache en disk no existe o expiró (TTL 24h).
        try:
            cache_path = _CONTACTS_PHONE_INDEX_PATH
            cache_stale = (
                not cache_path.is_file()
                or (time.time() - cache_path.stat().st_mtime) > _CONTACTS_PHONE_INDEX_TTL_S
            )
            if cache_stale:
                console.print()
                console.print(
                    "[dim]Pre-building Apple Contacts phone index (primer run, ~1-2min)…[/dim]"
                )
                t0 = time.time()
                idx = _load_contacts_phone_index()
                if idx:
                    console.print(
                        f"[green]✓[/green] contacts cache: "
                        f"{len(idx)} phones indexados en {time.time() - t0:.1f}s"
                    )
                else:
                    console.print(
                        "[yellow]·[/yellow] contacts cache vacío "
                        "(Contacts.app no accesible o sin teléfonos) — "
                        "el resolver usa mask fallback"
                    )
        except Exception as exc:
            _silent_log("setup_contacts_warmup", exc)
            console.print("[yellow]·[/yellow] contacts cache warmup falló — continuá sin ello")

        console.print()
        console.print(
            f"[dim]Logs en {_RAG_LOG_DIR}/{{watch,digest,morning,today,emergent,patterns,archive,"
            f"wa-tasks,online-tune,consolidate,ingest-whatsapp,ingest-gmail,ingest-reminders}}"
            f".{{log,error.log}}[/dim]"
        )


@click.command("stop")
@click.option(
    "--with-rag-net/--without-rag-net",
    default=True,
    help="Incluir daemons de RagNet/WhatsApp (whatsapp-bridge/listener/"
    "healthcheck/vault-sync). Default: sí.",
)
@click.option(
    "--with-qdrant",
    is_flag=True,
    default=False,
    help="Detener también qdrant (com.fer.qdrant). Default OFF: compartido con mem-vault.",
)
@click.option(
    "--all",
    "stop_all",
    is_flag=True,
    default=False,
    help="Atajo para --with-qdrant + --with-rag-net.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="No pedir confirmación.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Mostrar qué pararía pero no ejecutar.",
)
def stop(
    with_rag_net: bool,
    with_qdrant: bool,
    stop_all: bool,
    yes: bool,
    dry_run: bool,
) -> None:
    """Parar TODO el sistema: daemons obsidian-rag-*, RagNet, web server.

    Orden importante para evitar loops de auto-restart:

    1. Primero `daemon-watchdog` y `wake-hook` (los que re-bootstrappean
       a otros cada 5 min / post-wake). Si los pararamos al final, ellos
       ya rebootstrappearon a los anteriores.
    2. Después el resto de daemons managed (`_services_spec`) y manual
       (cloudflare-tunnel, lgbm-train, etc).
    3. Opcional: RagNet (whatsapp-*) si `--with-rag-net` (default ON).
    4. Opcional: qdrant si `--with-qdrant` (default OFF — compartido
       con mem-vault).

    Para volver a levantar todo: `rag start` (regenera plists desde
    código + bootstrap + catch-up index). Qdrant lo volvés a arrancar
    con su propio `launchctl bootstrap` o desde su repo origen — NO se
    archiva aunque esté en los targets porque es compartido con
    mem-vault y otros agentes locales.

    Archivado de plists: después del bootout, los `.plist` de
    `obsidian-rag-*` y `rag-net` se mueven a
    `~/Library/LaunchAgents/.archive-rag-stop-<timestamp>/` para que
    macOS NO los auto-loadee al próximo login. Así "rag off" = "nada
    rag corre, ni ahora ni al rebootear". `rag start` los recrea.
    """
    from rag import (  # noqa: PLC0415
        _LAUNCH_AGENTS_DIR,
        _all_daemon_labels,
        _bootout_label,
        _loaded_launchd_labels,
        _log_daemon_run_event,
        _silent_log,
        console,
    )

    if stop_all:
        with_qdrant = True
        with_rag_net = True

    # ── Detectar cloudflare tunnel plists dinámicamente ──────────────────
    # Si el user arrancó tunnels a mano, bootoutearlos junto con el resto.
    cloudflare_labels = _find_cloudflare_tunnel_labels()

    # ── Construir la lista ordenada de labels a bootoutear ──────────────
    # 1. Watchdog + wake-hook PRIMERO (para que no rebootstrap-een lo demás)
    PRIORITY_FIRST = (
        "com.fer.obsidian-rag-daemon-watchdog",
        "com.fer.obsidian-rag-wake-hook",
    )

    # 2. Resto del set obsidian-rag (managed + manual_keep)
    all_obsidian_rag = [lbl for lbl, _cat in _all_daemon_labels()]
    rest_obsidian_rag = [lbl for lbl in all_obsidian_rag if lbl not in PRIORITY_FIRST]
    # Ordenar para output reproducible (priority first nunca cambia de posición)
    rest_obsidian_rag.sort()

    targets: list[tuple[str, str]] = []  # (label, category)
    for lbl in PRIORITY_FIRST:
        if lbl in all_obsidian_rag:
            targets.append((lbl, "watchdog"))
    targets.extend((lbl, "obsidian-rag") for lbl in rest_obsidian_rag)

    # 3. Cloudflare tunnels (dinámico)
    if cloudflare_labels:
        targets.extend((lbl, "cloudflare") for lbl in cloudflare_labels)

    if with_rag_net:
        targets.extend((lbl, "rag-net") for lbl in _RAG_NET_LABELS)
    if with_qdrant:
        targets.extend((lbl, "qdrant") for lbl in _QDRANT_LABELS)

    if not targets:
        console.print("[yellow]No hay nada que parar.[/yellow]")
        return

    # ── Confirmación ─────────────────────────────────────────────────────
    # Pre-chequear qué labels están realmente cargados para mostrar conteos
    # honestos ("activos vs ya parados") en vez de mentirle al user con el
    # total bruto de plists registrados.
    loaded = _loaded_launchd_labels()
    loaded_known = bool(loaded)  # False = launchctl list falló, no sé

    def _split(cat: str) -> tuple[int, int]:
        """(activos, ya_parados) en la categoría según el set 'loaded'."""
        cat_targets = [lbl for lbl, c in targets if c == cat]
        if not loaded_known:
            return len(cat_targets), 0
        active = sum(1 for lbl in cat_targets if lbl in loaded)
        return active, len(cat_targets) - active

    a_watch, p_watch = _split("watchdog")
    a_obs, p_obs = _split("obsidian-rag")
    a_cf, p_cf = _split("cloudflare")
    a_rn, p_rn = _split("rag-net")
    a_qd, p_qd = _split("qdrant")
    n_active = a_watch + a_obs + a_cf + a_rn + a_qd
    n_stopped = p_watch + p_obs + p_cf + p_rn + p_qd
    n_total = n_active + n_stopped

    def _fmt(active: int, stopped: int) -> str:
        if not loaded_known:
            return f"{active + stopped} [dim](status desconocido)[/dim]"
        if stopped == 0:
            return f"[cyan]{active}[/cyan] activos"
        if active == 0:
            return f"[dim]{stopped} ya parados[/dim]"
        return f"[cyan]{active}[/cyan] activos, [dim]{stopped} ya parados[/dim]"

    console.print()
    if loaded_known:
        console.print(
            f"[bold]rag stop[/bold] — [cyan]{n_active}[/cyan] activos, "
            f"[dim]{n_stopped} ya parados[/dim] (de {n_total} registrados):"
        )
    else:
        console.print(
            f"[bold]rag stop[/bold] — voy a parar [cyan]{n_total}[/cyan] daemons "
            f"[dim](launchctl list falló, no pude verificar status)[/dim]:"
        )
    console.print(f"  [dim]·[/dim] watchdog/wake-hook : {_fmt(a_watch, p_watch)}")
    console.print(f"  [dim]·[/dim] obsidian-rag-*     : {_fmt(a_obs, p_obs)}")
    if cloudflare_labels:
        console.print(f"  [dim]·[/dim] cloudflare tunnels : {_fmt(a_cf, p_cf)}")
    if with_rag_net:
        console.print(f"  [dim]·[/dim] RagNet (whatsapp-*): {_fmt(a_rn, p_rn)}")
    else:
        console.print(
            "  [dim]·[/dim] RagNet (whatsapp-*): [yellow]skip[/yellow] (--without-rag-net)"
        )
    if with_qdrant:
        console.print(f"  [dim]·[/dim] qdrant             : {_fmt(a_qd, p_qd)}")
    else:
        console.print(
            "  [dim]·[/dim] qdrant             : [yellow]skip[/yellow] (default — compartido con mem-vault)"
        )
    console.print()
    if loaded_known and n_active == 0:
        console.print("[green]✓[/green] [dim]nada activo — no hay nada que parar.[/dim]")
        # Igual seguimos para archivar plists residuales si --yes/--dry-run.

    if dry_run:
        console.print("[dim]Dry-run — no ejecuto nada.[/dim]")
        for lbl, cat in targets:
            console.print(f"  [dim]would-bootout[/dim] [{cat}] {lbl}")
        return

    if not yes:
        if not click.confirm("¿Seguir?", default=False):
            console.print("[yellow]Cancelado.[/yellow]")
            return

    # ── Ejecución ────────────────────────────────────────────────────────
    ok_count = 0
    already_count = 0  # ya estaba parado (exit 3 / 113)
    fail_count = 0

    for label, category in targets:
        slug = (
            label.replace("com.fer.obsidian-rag-", "")
            if category in ("watchdog", "obsidian-rag", "cloudflare")
            else label
        )
        result = _bootout_label(label)

        # Distinguir "bootouted ahora" (rc=0) vs "ya estaba parado" (rc=3/113).
        if result["ok"] and result["exit_code"] == 0:
            console.print(f"[green]✓[/green] [{category}] {slug}")
            ok_count += 1
            _log_daemon_run_event(
                label=label,
                action="bootout",
                exit_code=0,
                reason="rag stop",
            )
        elif result["ok"]:
            console.print(f"[dim]·[/dim] [{category}] {slug} [dim](ya estaba parado)[/dim]")
            already_count += 1
        else:
            stderr_hint = result["stderr"][:120] if result["stderr"] else ""
            console.print(
                f"[red]✗[/red] [{category}] {slug} → exit={result['exit_code']}"
                + (f" — {stderr_hint}" if stderr_hint else "")
            )
            fail_count += 1
            _log_daemon_run_event(
                label=label,
                action="bootout",
                exit_code=result["exit_code"],
                reason="rag stop (failed)",
            )

    # ── Limpieza de locks staled ────────────────────────────────────────
    # Después de bootout, algunos daemons pueden dejar locks cuyos PIDs ya
    # no existen. Limpiarlos previene confusiones en el próximo `rag start`.
    try:
        _cleanup_staled_locks()
    except Exception as exc:
        _silent_log("rag_stop_cleanup_locks", exc)

    # ── Verificación final: no debería haber daemons cargados ──────────
    # Post-bootout, hacer un chequeo sanity de que nada quedó cargado.
    final_loaded = _loaded_launchd_labels()
    obsidian_rag_still_loaded = [
        lbl
        for lbl in final_loaded
        if lbl.startswith("com.fer.obsidian-rag-") or lbl in _RAG_NET_LABELS
    ]
    # Cloudflare puede no ser detectable de la misma forma (es dinámico),
    # pero si encontramos labels obsidian-rag vivos, reportar.
    if obsidian_rag_still_loaded and ok_count > 0:
        console.print()
        console.print(
            f"[yellow]![/yellow] post-stop, todavía hay daemons cargados: "
            f"{', '.join(obsidian_rag_still_loaded[:3])}"
            + ("..." if len(obsidian_rag_still_loaded) > 3 else "")
        )
        console.print("[yellow]Hint: probá `rag daemons doctor` para diagnóstico[/yellow]")

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    if fail_count == 0:
        console.print(
            f"[green]✓[/green] sistema detenido — {ok_count} parados, {already_count} ya estaban"
        )
    else:
        console.print(
            f"[yellow]parados {ok_count}, ya estaban {already_count}, "
            f"[red]{fail_count} fallidos[/red][/yellow]"
        )
    # ── Archivar plists bootouted ────────────────────────────────────────
    # Por qué: macOS auto-loadea `~/Library/LaunchAgents/*.plist` al
    # próximo login. Sin este paso, `rag stop` vale solo hasta el próximo
    # reboot/login — ahí los plists vuelven a arrancar aunque el user no
    # haya corrido `rag start`. Moviendolos a `.archive-rag-stop-<ts>/`
    # garantiza "si rag está off, al login nada arranca". `rag start` los
    # regenera desde código via `setup()` — son artefactos, no hay
    # pérdida de estado.
    archive_root = (
        _LAUNCH_AGENTS_DIR / f".archive-rag-stop-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    archived = 0
    archive_skip_categories = {"qdrant"}  # compartidos con mem-vault
    for label, category in targets:
        if category in archive_skip_categories:
            continue
        plist_path = _LAUNCH_AGENTS_DIR / f"{label}.plist"
        if not plist_path.is_file():
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        try:
            _shutil.move(str(plist_path), str(archive_root / f"{label}.plist"))
            archived += 1
        except OSError as exc:
            console.print(f"[yellow]·[/yellow] no pude archivar {label}: {exc}")
    if archived:
        console.print()
        console.print(
            f"[dim]·[/dim] [cyan]{archived}[/cyan] plists movidos a "
            f"[dim]{archive_root}[/dim] "
            "[dim](no van a auto-loadearse al próximo login)[/dim]"
        )

    console.print()
    console.print("[dim]Para volver a arrancar:[/dim]")
    console.print(
        "  [dim]·[/dim] obsidian-rag : [cyan]rag start[/cyan] [dim](o `rag setup` para solo regenerar plists)[/dim]"
    )
    if with_rag_net:
        console.print(
            "  [dim]·[/dim] RagNet       : "
            "[cyan]launchctl bootstrap gui/$(id -u) "
            "~/Library/LaunchAgents/com.fer.whatsapp-listener.plist[/cyan] "
            "(idem bridge / healthcheck / vault-sync)"
        )
    if with_qdrant:
        console.print(
            "  [dim]·[/dim] qdrant       : "
            "[cyan]launchctl bootstrap gui/$(id -u) "
            "~/Library/LaunchAgents/com.fer.qdrant.plist[/cyan]"
        )
    console.print(
        "[dim](Plists quedan en disco; macOS los re-loadea solo al próximo "
        "login. Para borrarlos: `rag setup --remove`.)[/dim]"
    )


def _run_catch_up_index(ctx: click.Context) -> None:
    """Catch-up incremental index del vault, llamado por `rag start`.

    Refactor 2026-05-09 (Phase 2d preliminar): extraido del cuerpo de
    `start` para destrabar la extracción del comando a `rag/cli/setup.py`.
    `start` invocaba `ctx.invoke(index, ...)` directamente, que es legítimo
    pero ata el comando al closure del Click context y no se mueve limpio
    a un módulo externo. Ahora `start` llama esta helper, y la helper
    encapsula el `ctx.invoke` (preserva error handling SystemExit-aware
    + silent_log idéntico al inline original).

    Behavior idéntico al inline pre-refactor:
    - Si otra instancia de `rag index` está corriendo (lock contention
      → SystemExit code != 0), informa al user con un mensaje "skip" y
      sigue (no aborta `start`).
    - Si tira excepción no-Exit, la loggea silent y avisa.
    """
    from rag import _silent_log, console, index  # noqa: PLC0415

    console.print()
    console.print("[bold cyan]▸ catch-up index (incremental)[/bold cyan]")
    try:
        # Click param names — `--full` binds a `full_flag`, `--reset` (alias
        # legacy) a `reset_legacy`. Pasarle `reset=False` revienta con
        # `TypeError: index() got an unexpected keyword argument 'reset'`.
        # `contextual` y `fast` son options nuevas — defaults explícitos.
        ctx.invoke(
            index,
            full_flag=False,
            reset_legacy=False,
            no_contradict=False,
            source_opt=None,
            since_opt=None,
            dry_run=False,
            max_chats=None,
            vault_scope=None,
            contextual=False,
            fast=False,
        )
    except SystemExit as e:
        if e.code != 0:
            console.print(
                "[yellow]·[/yellow] catch-up skip "
                "(otro `rag index` activo, retry manual con `rag index`)"
            )
    except Exception as exc:
        console.print(f"[yellow]·[/yellow] catch-up falló: {exc!r}")
        _silent_log("rag_start_index_failed", exc)


@click.command("start")
@click.option(
    "--minimal/--full",
    "minimal",
    default=True,
    help="Mínimo viable (5 daemons: watch/web/daemon-watchdog/wake-hook/"
    "maintenance) vs full (los 30 de _services_spec, incluyendo briefs, "
    "learning loops y productividad pasiva). Default: mínimo. Lo NO "
    "instalado en modo mínimo se bootouts + removido del disco — "
    "re-correr con `--full` regenera todo.",
)
@click.option(
    "--with-rag-net/--without-rag-net",
    default=True,
    help="Levantar también daemons de RagNet/WhatsApp (whatsapp-bridge/"
    "listener/healthcheck/vault-sync). Default: sí — sin RagNet la UX "
    "del sistema queda a medias (web sí, WhatsApp no).",
)
@click.option(
    "--with-qdrant",
    is_flag=True,
    default=False,
    help="Levantar también qdrant (com.fer.qdrant). Default OFF.",
)
@click.option(
    "--all",
    "start_all",
    is_flag=True,
    default=False,
    help="Atajo para --with-rag-net + --with-qdrant.",
)
@click.option(
    "--no-index",
    is_flag=True,
    default=False,
    help="Saltear el catch-up incremental al final (default: sí lo corre, "
    "para que el corpus quede al último minuto de uso).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="No pedir confirmación.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Mostrar qué levantaría pero no ejecutar.",
)
@click.option(
    "--apply-default",
    is_flag=True,
    default=False,
    help="Antes de levantar daemons, aplicar el profile default de "
    "rag-harness (.devin/mcp-profiles/.default). Si no hay default "
    "seteado, no hace nada. Útil para garantizar un harness consistente "
    "después de un reboot o un wrap mal cerrado.",
)
@click.pass_context
def start(
    ctx: click.Context,
    minimal: bool,
    with_rag_net: bool,
    with_qdrant: bool,
    start_all: bool,
    no_index: bool,
    yes: bool,
    dry_run: bool,
    apply_default: bool,
) -> None:
    """Levantar TODO el sistema y reindexar al último minuto de uso.

    Simétrico a `rag stop`. Idempotente — re-correrlo recarga lo que esté
    caído sin romper lo que ya está vivo. Orden:

    1. `rag setup` — regenera + carga los `obsidian-rag-*` (managed) desde
       código (incluye watch / web / digest / morning / today / ingesters
       cross-source / wa-tasks / online-tune / watchdog / wake-hook / ...).
    2. Externos vía `launchctl bootstrap` desde `~/Library/LaunchAgents/`:
       qdrant (si `--with-qdrant`), RagNet (whatsapp-*) si `--with-rag-net`
       (default ON).
    3. Catch-up index incremental (`rag index`, sin `--reset`) — re-indexa
       todo lo que cambió desde el último run del watcher (típicamente
       minutos; si la Mac estuvo dormida varias horas, captura ese gap).
       Skip con `--no-index`.

    Si un externo no tiene plist en disco, lo reporta como "missing" y
    sigue — instalalo desde su repo origen ([whatsapp-listener](https://github.com/jagoff/whatsapp-listener)
    para RagNet, [mem-vault](https://github.com/jagoff/mem-vault) para
    qdrant) y re-corré `rag start`.
    """
    from rag import (  # noqa: PLC0415
        _bootstrap_label,
        _log_daemon_run_event,
        _rag_binary,
        _services_spec,
        _silent_log,
        console,
    )

    if start_all:
        with_qdrant = True
        with_rag_net = True

    rag_bin = _rag_binary()
    if not Path(rag_bin).is_file():
        console.print(f"[red]No encuentro el binario `rag`:[/red] {rag_bin}")
        console.print(
            "[dim]Instalá primero: uv tool install --reinstall --editable '.[entities]'[/dim]"
        )
        return

    spec_total = len(_services_spec(rag_bin))
    if minimal:
        only_labels: frozenset[str] | None = _MINIMAL_MANAGED_LABELS
        managed_count = len(_MINIMAL_MANAGED_LABELS)
        scope_label = (
            f"[bold]mínimo[/bold] ({managed_count} de {spec_total}: watch/web/"
            "daemon-watchdog/wake-hook/maintenance)"
        )
    else:
        only_labels = None
        managed_count = spec_total
        scope_label = f"[bold]full[/bold] ({spec_total})"

    # ── Preview ──────────────────────────────────────────────────────────
    console.print()
    console.print("[bold]rag start[/bold] — voy a levantar el sistema:")
    console.print(f"  [dim]·[/dim] obsidian-rag-* (managed): {scope_label}")
    if with_rag_net:
        console.print(f"  [dim]·[/dim] RagNet (whatsapp-*)     : {len(_RAG_NET_LABELS)} daemons")
    else:
        console.print(
            "  [dim]·[/dim] RagNet (whatsapp-*)     : [yellow]skip[/yellow] (--without-rag-net)"
        )
    if with_qdrant:
        console.print(f"  [dim]·[/dim] qdrant                  : {len(_QDRANT_LABELS)} daemons")
    else:
        console.print(
            "  [dim]·[/dim] qdrant                  : [yellow]skip[/yellow] (default — usalo si bajaste qdrant)"
        )
    if no_index:
        console.print("  [dim]·[/dim] catch-up index          : [yellow]skip[/yellow] (--no-index)")
    else:
        console.print("  [dim]·[/dim] catch-up index          : [cyan]rag index[/cyan] incremental")

    # ── Hook informativo con rag-harness (no toca configs) ────────────────
    # Si hay un profile activo en `.devin/mcp-profiles/.active`, lo mostramos
    # como contexto. Si hay un default seteado y no coincide con el activo,
    # avisamos. NO aplicamos nada automático — eso lo decide el usuario con
    # `rag-harness use <name>` o `rag-harness default --apply`.
    try:
        # Repo root: rag/cli/setup.py → parent.parent.parent
        _harness_root = Path(__file__).resolve().parent.parent.parent / ".devin" / "mcp-profiles"
        _active_path = _harness_root / ".active"
        _default_path = _harness_root / ".default"
        if _active_path.exists() or _default_path.exists():
            active_p = _active_path.read_text().strip() if _active_path.exists() else None
            default_p = _default_path.read_text().strip() if _default_path.exists() else None
            if active_p:
                console.print(
                    f"  [dim]·[/dim] rag-harness profile     : [cyan]{active_p}[/cyan]"
                    + (
                        " [yellow](≠ default)[/yellow]"
                        if default_p and default_p != active_p
                        else ""
                    )
                )
            elif default_p:
                console.print(
                    f"  [dim]·[/dim] rag-harness default     : [yellow]{default_p}[/yellow] (no aplicado — `rag-harness default --apply`)"
                )
    except Exception:
        # Hook informativo — un error acá no debe romper `rag start`
        pass

    console.print()

    if dry_run:
        console.print("[dim]Dry-run — no ejecuto nada.[/dim]")
        if apply_default:
            try:
                _hd = (
                    Path(__file__).resolve().parent.parent.parent
                    / ".devin"
                    / "mcp-profiles"
                    / ".default"
                )
                if _hd.exists():
                    console.print(
                        f"  [dim]would-apply[/dim] [harness] profile default = {_hd.read_text().strip()}"
                    )
                else:
                    console.print("  [dim]would-skip[/dim] [harness] no hay default seteado")
            except Exception:
                pass
        if minimal:
            installed = sorted(_MINIMAL_MANAGED_LABELS)
            console.print(f"  [dim]would-install[/dim] [managed-minimal] {len(installed)} labels:")
            for lbl in installed:
                console.print(f"    [dim]·[/dim] {lbl}")
            removed = sorted(
                lbl for lbl, _, _ in _services_spec(rag_bin) if lbl not in _MINIMAL_MANAGED_LABELS
            )
            if removed:
                console.print(
                    f"  [dim]would-remove[/dim] [managed-out-of-scope] {len(removed)} labels"
                )
        else:
            console.print(
                f"  [dim]would-install[/dim] [managed-full] {len(_services_spec(rag_bin))} labels"
            )
        if with_qdrant:
            for lbl in _QDRANT_LABELS:
                console.print(f"  [dim]would-bootstrap[/dim] [qdrant] {lbl}")
        if with_rag_net:
            for lbl in _RAG_NET_LABELS:
                console.print(f"  [dim]would-bootstrap[/dim] [rag-net] {lbl}")
        if not no_index:
            console.print("  [dim]would-call[/dim] [index] rag index (incremental)")
        return

    if not yes:
        if not click.confirm("¿Seguir?", default=True):
            console.print("[yellow]Cancelado.[/yellow]")
            return

    # ── Apply rag-harness default (opt-in via --apply-default) ───────────
    # No-op si no hay default seteado. Falla suave si el harness no está
    # disponible — no debe bloquear `rag start` por un problema cosmético.
    if apply_default:
        try:
            _harness_default = (
                Path(__file__).resolve().parent.parent.parent
                / ".devin"
                / "mcp-profiles"
                / ".default"
            )
            if _harness_default.exists():
                _default_name = _harness_default.read_text().strip()
                console.print()
                console.print(
                    f"[bold cyan]▸ rag-harness default[/bold cyan] → aplicando '{_default_name}'"
                )
                _hb = Path.home() / ".local" / "bin" / "rag-harness"
                if _hb.exists():
                    import subprocess as _sp

                    _r = _sp.run([str(_hb), "use", _default_name], capture_output=True, text=True)
                    if _r.returncode == 0:
                        console.print(f"  [green]✓[/green] profile '{_default_name}' aplicado")
                        console.print(
                            "  [dim](las sesiones existentes de Claude Code / Devin no ven el cambio hasta reabrir)[/dim]"
                        )
                    else:
                        console.print(
                            f"  [yellow]![/yellow] rag-harness use falló: {_r.stderr.strip()[:200]}"
                        )
                else:
                    console.print(
                        f"  [yellow]![/yellow] rag-harness no está en {_hb} — instalá el symlink primero"
                    )
            else:
                console.print()
                console.print(
                    "[dim]--apply-default pasado pero no hay default seteado — skip[/dim]"
                )
        except Exception as _exc:
            console.print(f"[dim]--apply-default falló silenciosamente: {_exc!r}[/dim]")

    # ── 1. Catch-up incremental index (ANTES de bootstrap daemons) ───────
    # MLX-aware reorder (2026-05-08): el catch-up corre PRIMERO porque el
    # web/watch bootstrappados cargan MLX en sus propios procesos vía
    # warmup_async, y co-residency MLX × N procesos triggerea
    # `[METAL] Command buffer execution failed` reproducible
    # determinísticamente (memoria 94a2a13e). Antes el orden era:
    # 1) setup_install daemons → 2) externos → 3) catch-up index.
    # El índex era el ÚLTIMO con web/watch ya cargando MLX en paralelo.
    # Ahora corre el index PRIMERO con el proceso `rag start` solo, y
    # DESPUÉS bootstrappa daemons (que pueden ir cargando MLX
    # tranquilamente sin conflicto).
    if not no_index:
        _run_catch_up_index(ctx)

    # ── 2. obsidian-rag-* via setup callback ─────────────────────────────
    # `_setup_install` regenera los plists desde código (captura overrides
    # del schedule auto-tune + cualquier cambio en _services_spec) y los
    # carga vía `launchctl load`. Ya es idempotente. Cuando `minimal=True`,
    # los labels fuera de `_MINIMAL_MANAGED_LABELS` se bootouts + remueven
    # del disco (mismo trato que `_DEPRECATED_LABELS`).
    console.print()
    console.print("[bold cyan]▸ obsidian-rag-* (managed daemons)[/bold cyan]")
    try:
        _setup_install(rag_bin, remove=False, only_labels=only_labels)
    except Exception as exc:
        console.print(f"[red]✗[/red] setup falló: {exc!r}")
        # No abortar — los externos podrían levantarse igual.
        _silent_log("rag_start_setup_failed", exc)

    # ── 2. Externos: qdrant → rag-net ────────────────────────────────────
    # Orden: qdrant primero (si el user lo pidió, quiere que esté up antes
    # de que RagNet arranque a chatear con mem-vault). Después RagNet
    # (consume web).
    ext_targets: list[tuple[str, str]] = []
    if with_qdrant:
        ext_targets.extend((lbl, "qdrant") for lbl in _QDRANT_LABELS)
    if with_rag_net:
        ext_targets.extend((lbl, "rag-net") for lbl in _RAG_NET_LABELS)

    ext_ok = ext_already = ext_fail = ext_missing = 0
    if ext_targets:
        console.print()
        console.print("[bold cyan]▸ daemons externos[/bold cyan]")
        for label, category in ext_targets:
            result = _bootstrap_label(label)
            if result["missing_plist"]:
                console.print(
                    f"[yellow]·[/yellow] [{category}] {label} "
                    f"[dim](sin plist en disco — instalalo desde su repo)[/dim]"
                )
                ext_missing += 1
            elif result["ok"] and result["exit_code"] == 0:
                console.print(f"[green]✓[/green] [{category}] {label}")
                ext_ok += 1
                _log_daemon_run_event(
                    label=label,
                    action="bootstrap",
                    exit_code=0,
                    reason="rag start",
                )
            elif result["ok"]:
                console.print(f"[dim]·[/dim] [{category}] {label} [dim](ya estaba cargado)[/dim]")
                ext_already += 1
            else:
                stderr_hint = result["stderr"][:120] if result["stderr"] else ""
                console.print(
                    f"[red]✗[/red] [{category}] {label} → exit={result['exit_code']}"
                    + (f" — {stderr_hint}" if stderr_hint else "")
                )
                ext_fail += 1
                _log_daemon_run_event(
                    label=label,
                    action="bootstrap",
                    exit_code=result["exit_code"],
                    reason="rag start (failed)",
                )

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    if ext_fail == 0:
        console.print("[bold green]✓ sistema levantado[/bold green]")
    else:
        console.print(
            f"[yellow]sistema levantado parcialmente — "
            f"[red]{ext_fail} externo(s) fallidos[/red][/yellow]"
        )
    if ext_targets:
        parts = []
        if ext_ok:
            parts.append(f"{ext_ok} cargados")
        if ext_already:
            parts.append(f"{ext_already} ya estaban")
        if ext_missing:
            parts.append(f"[yellow]{ext_missing} sin plist[/yellow]")
        if ext_fail:
            parts.append(f"[red]{ext_fail} fallidos[/red]")
        console.print(f"  [dim]externos:[/dim] {', '.join(parts)}")
    console.print()

    # ── URLs y endpoints accesibles ──────────────────────────────────────
    _print_access_urls()
