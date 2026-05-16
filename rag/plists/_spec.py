"""Spec orquestación de daemons launchd + install gates + deprecated labels.

`_services_spec(rag_bin)` retorna la lista canónica de daemons que el
control plane (`rag daemons {status|reconcile|doctor|retry}`) gestiona.
Cada entry: `(label, plist_filename, plist_xml)`.

Post-supervisor refactor (2026-05-09): solo **3 plists managed** —
``supervisor``, ``watch``, ``web``. Los 27 plists de cron/timer
fueron migrados al supervisor in-process (ver
``rag/runtime/jobs/*.py``). Sus labels están listados en
``_DEPRECATED_LABELS`` para que ``rag setup`` los bootouts y borre
de disco automáticamente. Las factories siguen disponibles en los
módulos ``rag.plists.{briefs,control,ingest,learning,maintenance,
poll,proactive,wa}.py`` por si se necesita rollback emergencia
(set ``RAG_USE_LEGACY_PLISTS=1`` para resucitarlos manualmente).

Brief schedule overrides (2026-04-29): morning/today/digest se
manejan ahora en ``rag/runtime/jobs/briefs.py`` con los defaults
históricos (Mon-Fri 7:00 / 22:00, Sun 22:00). Override via
``rag_brief_schedule_prefs`` queda como F3-followup pendiente
(IPC ``/reload-schedules`` del supervisor).
"""
from __future__ import annotations

from pathlib import Path

from rag._constants import _GOOGLE_TOKEN_PATH
from rag.plists.persistent import _supervisor_plist, _watch_plist, _web_plist

__all__ = [
    "_DEPRECATED_LABELS",
    "_INSTALL_GATES",
    "_calendar_creds_exist",
    "_google_token_exists",
    "_mood_daemon_opted_in",
    "_services_spec",
    "_services_spec_manual",
]


def _services_spec(rag_bin: str) -> list[tuple[str, str, str]]:
    """Return [(label, plist_filename, plist_xml), ...].

    Post-refactor 2026-05-09: solo 3 plists. Los 27 daemons cron viejos
    fueron migrados al supervisor in-process (``rag/runtime/``). Sus
    labels están en ``_DEPRECATED_LABELS`` para auto-bootout en
    ``rag setup``.
    """
    return [
        # Persistent supervisor — 1 daemon que orquesta 28 jobs cron
        # in-process via APScheduler + IPC + event bus + 3 SQL watchers
        # + Spotify NSDistributedNotificationCenter listener. Reemplaza
        # 27 plists viejos (ver _DEPRECATED_LABELS abajo).
        ("com.fer.obsidian-rag-supervisor",
         "com.fer.obsidian-rag-supervisor.plist",
         _supervisor_plist(rag_bin)),
        # Persistent watchdog observing ALL registered vaults. Sin
        # cron — file watcher KeepAlive=true.
        ("com.fer.obsidian-rag-watch", "com.fer.obsidian-rag-watch.plist",
         _watch_plist(rag_bin)),
        # FastAPI web UI + SSE chat en :8765. Proceso aparte del
        # supervisor por fault isolation.
        ("com.fer.obsidian-rag-web", "com.fer.obsidian-rag-web.plist",
         _web_plist(rag_bin)),
    ]


# ── Install gates: pre-requisitos por label ─────────────────────────────────
#
# Cada entry es `(check_fn, hint)`. Si `check_fn()` devuelve False, `rag setup`
# NO instala el plist y muestra `hint` para que el user sepa cómo activar.
# Re-correr `rag setup` después del pre-req instala el plist.
#
# Por qué este gate existe: sin él, un plist se carga y falla cada cadencia
# cuando falta la credencial / opt-in. Polución de logs + `daemons status`
# con rows en rojo que el user no mira hasta que rompe algo más visible.
# Aprendido el 2026-05-04 del daemon `ingest-gmail` en exit-loop hace 4 días
# sin alerta, + `mood-poll` cargado sin opt-in consumiendo nothing visible.
def _google_token_exists() -> bool:
    """Gmail + Drive comparten token — un OAuth flow cubre ambos."""
    return _GOOGLE_TOKEN_PATH.is_file()


def _calendar_creds_exist() -> bool:
    return (Path.home() / ".calendar-mcp" / "credentials.json").is_file()


def _mood_daemon_opted_in() -> bool:
    """True si el user hizo `rag mood enable` (crea el state file)."""
    return (Path.home() / ".local/share/obsidian-rag/mood_enabled").is_file()


# ── Labels deprecated ──────────────────────────────────────────────────────
#
# Plists que EXISTIERON en `_services_spec` y fueron consolidados/removidos.
# `rag setup` los bootouts + borra de disco para cerrar el gap de migración
# (sino el plist viejo sigue cargado y corriendo en paralelo con su reemplazo,
# doble notificación / doble envío / race conditions).
#
# Una vez que todos los usuarios hayan corrido `rag setup` al menos una vez
# post-consolidación, el entry correspondiente se puede remover de acá —
# pero mantenerlo no cuesta nada (es un set chico) y protege contra rollback
# parcial (user revertió a versión vieja y después volvió a la nueva).
_DEPRECATED_LABELS: frozenset[str] = frozenset({
    # External RagNet helper removed 2026-05-16: monthly WhatsApp notes are
    # now written by `rag.integrations.whatsapp_etl` during `rag index`.
    "com.fer.whatsapp-vault-sync",
    # Consolidados en `wa-fast` el 2026-05-04:
    "com.fer.obsidian-rag-reminder-wa-push",
    "com.fer.obsidian-rag-wa-scheduled-send",
    # Consolidados en `ingest-cross-source` el 2026-05-04:
    "com.fer.obsidian-rag-ingest-gmail",
    "com.fer.obsidian-rag-ingest-calendar",
    "com.fer.obsidian-rag-ingest-reminders",
    "com.fer.obsidian-rag-ingest-calls",
    "com.fer.obsidian-rag-ingest-safari",
    "com.fer.obsidian-rag-ingest-drive",
    "com.fer.obsidian-rag-ingest-pillow",
    # Migrados al supervisor in-process el 2026-05-09 (ver
    # rag/runtime/jobs/*.py). El supervisor maneja todos los crones via
    # APScheduler + 3 SQL triggers + spotify NSDistributed.
    "com.fer.obsidian-rag-auto-harvest",
    "com.fer.obsidian-rag-whisper-vocab",
    "com.fer.obsidian-rag-implicit-feedback",
    "com.fer.obsidian-rag-online-tune",
    "com.fer.obsidian-rag-maintenance",
    "com.fer.obsidian-rag-calibrate",
    "com.fer.obsidian-rag-anticipate",
    "com.fer.obsidian-rag-routing-rules",
    "com.fer.obsidian-rag-wa-fast",
    "com.fer.obsidian-rag-ingest-whatsapp",
    "com.fer.obsidian-rag-ingest-cross-source",
    "com.fer.obsidian-rag-mood-poll",
    "com.fer.obsidian-rag-spotify-poll",
    "com.fer.obsidian-rag-wa-tasks",
    "com.fer.obsidian-rag-emergent",
    "com.fer.obsidian-rag-patterns",
    "com.fer.obsidian-rag-archive",
    "com.fer.obsidian-rag-distill",
    "com.fer.obsidian-rag-active-learning-nudge",
    "com.fer.obsidian-rag-active-learning-suggest-goldens",
    "com.fer.obsidian-rag-brief-auto-tune",
    "com.fer.obsidian-rag-morning",
    "com.fer.obsidian-rag-today",
    "com.fer.obsidian-rag-digest",
    "com.fer.obsidian-rag-vault-cleanup",
    "com.fer.obsidian-rag-wake-up",
    "com.fer.obsidian-rag-consolidate",
    "com.fer.obsidian-rag-drift-watcher",
    "com.fer.obsidian-rag-daemon-watchdog",
    "com.fer.obsidian-rag-wake-hook",
})


# Post-supervisor refactor 2026-05-09: los install gates ya no aplican
# en _services_spec (que solo tiene supervisor + watch + web, todos
# always-on). El gate de mood-opt-in se manejaba en mood-poll plist
# pre-supervisor; ahora el job in-process ``mood_poll_job`` invoca un
# script que ya tiene ese gate adentro (skip silently si ``mood_enabled``
# state file no existe). El dict queda vacío como contract API por si
# se agrega un nuevo plist gated en el futuro.
_INSTALL_GATES: dict[str, tuple] = {}


def _services_spec_manual() -> list[dict]:
    """Daemons launchd que existen en disco pero NO tienen factory en código.
    Instalados a mano por el usuario. `rag setup` no los toca; el control
    plane (`rag daemons status / reconcile`) los monitorea pero no los
    regenera. Si alguno se rompe, el fix es manual (re-copiar plist desde
    backup o regenerarlo en su repo origen).

    Limpieza 2026-05-04: se removieron 4 entries fantasma (cloudflare-tunnel*,
    lgbm-train, paraphrases-train).

    Limpieza 2026-05-10: se removieron `synth-refresh` y `log-rotate` —
    ya no tienen plist en disco ni logs (los .log archivados últimos en
    2026-04/05 ya fueron rotados/borrados). Eran "drift" permanente en
    `rag daemons status` sin nada que el user pueda hacer al respecto.
    Si en el futuro se vuelven a necesitar, re-añadir acá + regenerar
    el plist desde el repo origen del user.

    Lista actualmente vacía — todos los daemons activos viven en
    `_services_spec()` (los 3 managed) o en `_RAG_NET_LABELS` (externos).
    Mantengo la función como contract API por si en el futuro vuelve a
    haber un manual_keep legítimo.
    """
    return []
