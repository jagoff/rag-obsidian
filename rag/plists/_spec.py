"""Spec orquestación de daemons launchd + install gates + deprecated labels.

`_services_spec(rag_bin)` retorna la lista canónica de daemons que el
control plane (`rag daemons {status|reconcile|doctor|retry}`) gestiona.
Cada entry: `(label, plist_filename, plist_xml)`.

Brief schedule overrides (2026-04-29): morning/today/digest plists
consultan `rag_brief_schedule_prefs` antes de generar XML — si hay
override, usa esa hora; si no, default histórico. Lookup silent-fail
para que `rag setup` en una install brand-new no bloquee nunca.
"""
from __future__ import annotations

from pathlib import Path

from rag._constants import _GOOGLE_TOKEN_PATH
from rag.plists._legacy import (
    _consolidate_plist,
    _daemon_watchdog_plist,
    _ingest_cross_source_plist,
    _ingest_whatsapp_plist,
    _maintenance_plist,
    _mood_poll_plist,
    _spotify_poll_plist,
    _vault_cleanup_plist,
    _wa_fast_plist,
    _wake_hook_plist,
    _wake_up_plist,
)
from rag.plists.briefs import _digest_plist, _morning_plist, _today_plist
from rag.plists.learning import (
    _auto_harvest_plist,
    _calibration_plist,
    _implicit_feedback_plist,
    _online_tune_plist,
    _routing_rules_plist,
    _whisper_vocab_plist,
)
from rag.plists.persistent import _watch_plist, _web_plist
from rag.plists.proactive import (
    _active_learning_nudge_plist,
    _anticipate_plist,
    _archive_plist,
    _brief_auto_tune_plist,
    _distill_plist,
    _emergent_plist,
    _patterns_plist,
)

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

    Brief schedule overrides (2026-04-29): morning/today/digest plists
    consult `rag_brief_schedule_prefs` via `rag.brief_schedule.
    get_brief_schedule_pref()` BEFORE generating their XML. If a row
    exists, the override `(hour, minute)` is substituted; otherwise the
    historical defaults from the plist functions kick in. The lookup
    is silent-fail on any SQL error (fresh DB, locked file, etc.) so
    `rag setup` on a brand-new install never blocks on telemetry.
    """
    from rag.integrations.whatsapp import _wa_tasks_plist  # noqa: PLC0415
    # Lazy import keeps module-load lightweight + avoids a circular
    # import (brief_schedule lazy-imports back into rag for SQL conn).
    try:
        from rag.brief_schedule import get_brief_schedule_pref
    except Exception:
        def get_brief_schedule_pref(_kind):  # type: ignore
            return None

    def _override(kind: str, default_hour: int, default_minute: int) -> tuple[int, int]:
        try:
            pref = get_brief_schedule_pref(kind)
        except Exception:
            pref = None
        if pref is None:
            return (default_hour, default_minute)
        return (int(pref.get("hour", default_hour)), int(pref.get("minute", default_minute)))

    morning_h, morning_m = _override("morning", 7, 0)
    today_h, today_m = _override("today", 22, 0)
    digest_h, digest_m = _override("digest", 22, 0)

    return [
        ("com.fer.obsidian-rag-watch", "com.fer.obsidian-rag-watch.plist",
         _watch_plist(rag_bin)),
        # Nota histórica (2026-05-01): `com.fer.obsidian-rag-serve` +
        # `com.fer.obsidian-rag-serve-watchdog` fueron deprecados — split-brain
        # con FastAPI web + crash-loop bajo memory pressure. FastAPI cubre todos
        # los endpoints reales; WA listener cae a subprocess `rag query` cuando
        # :7832 no responde. Las factories `_serve_plist` + `_serve_watchdog_plist`
        # se borraron en Fase 2a (2026-05-09); los labels siguen en
        # `_DEPRECATED_LABELS` para que `rag setup` los bootouts en disco.
        # Re-activación: agregar `POST /api/query` al FastAPI y apuntar el
        # listener a :8765 (no resucitar el plist viejo).
        # ────────────────────────────────────────────────────────────────────
        # Web UI daemon — previously installed manually outside `rag setup`
        # (Apr 2026), which left HF_HUB_OFFLINE + RAG_MEMORY_PRESSURE_INTERVAL
        # missing in the actual plist and produced 64× [local-embed] falls +
        # 23× MPS Metal OOMs. Now generated from source.
        ("com.fer.obsidian-rag-web", "com.fer.obsidian-rag-web.plist",
         _web_plist(rag_bin)),
        ("com.fer.obsidian-rag-digest", "com.fer.obsidian-rag-digest.plist",
         _digest_plist(rag_bin, hour=digest_h, minute=digest_m)),
        ("com.fer.obsidian-rag-morning", "com.fer.obsidian-rag-morning.plist",
         _morning_plist(rag_bin, hour=morning_h, minute=morning_m)),
        ("com.fer.obsidian-rag-today", "com.fer.obsidian-rag-today.plist",
         _today_plist(rag_bin, hour=today_h, minute=today_m)),
        ("com.fer.obsidian-rag-wake-up", "com.fer.obsidian-rag-wake-up.plist",
         _wake_up_plist(rag_bin)),
        ("com.fer.obsidian-rag-emergent", "com.fer.obsidian-rag-emergent.plist",
         _emergent_plist(rag_bin)),
        ("com.fer.obsidian-rag-patterns", "com.fer.obsidian-rag-patterns.plist",
         _patterns_plist(rag_bin)),
        ("com.fer.obsidian-rag-archive", "com.fer.obsidian-rag-archive.plist",
         _archive_plist(rag_bin)),
        # Weekly conversation distiller — rescata bot answers de
        # conversations con sources missing antes de que el conocimiento
        # quede sólo en logs no-indexados. Defense-in-depth con
        # promote-on-cite del archive (commit `e89c42f`, 2026-05-04).
        ("com.fer.obsidian-rag-distill", "com.fer.obsidian-rag-distill.plist",
         _distill_plist(rag_bin)),
        ("com.fer.obsidian-rag-wa-tasks", "com.fer.obsidian-rag-wa-tasks.plist",
         _wa_tasks_plist(rag_bin)),
        # WA workers time-sensitive (5 min) — worker unificado que corre
        # `remind-wa` + `wa-scheduled-send` en serie. Consolidación
        # 2026-05-04: antes eran 2 plists separados (mismo cron, mismo
        # budget), ahora 1. Ver `_wa_fast_plist` docstring para rationale.
        ("com.fer.obsidian-rag-wa-fast",
         "com.fer.obsidian-rag-wa-fast.plist",
         _wa_fast_plist(rag_bin)),
        ("com.fer.obsidian-rag-auto-harvest", "com.fer.obsidian-rag-auto-harvest.plist",
         _auto_harvest_plist(rag_bin)),
        # Active-learning nudge (C.6, 2026-04-29) — Lunes 10am: cuenta
        # queries low-conf ultimos 7d sin labels y manda push al RagNet
        # con link a /learning si supera 20 candidates. Reemplazo del
        # bash inline que disparaba osascript notification.
        ("com.fer.obsidian-rag-active-learning-nudge",
         "com.fer.obsidian-rag-active-learning-nudge.plist",
         _active_learning_nudge_plist(rag_bin)),
        # Implicit corrective_path inference — corre 03:25, 5 min antes
        # del online-tune. Pre-popula corrective_paths desde behavior
        # post-👎 sin pedir input al user. Sprint 1 del cierre del loop
        # de auto-aprendizaje (2026-04-26).
        ("com.fer.obsidian-rag-implicit-feedback",
         "com.fer.obsidian-rag-implicit-feedback.plist",
         _implicit_feedback_plist(rag_bin)),
        ("com.fer.obsidian-rag-online-tune", "com.fer.obsidian-rag-online-tune.plist",
         _online_tune_plist(rag_bin)),
        ("com.fer.obsidian-rag-calibrate", "com.fer.obsidian-rag-calibrate.plist",
         _calibration_plist(rag_bin)),
        ("com.fer.obsidian-rag-maintenance", "com.fer.obsidian-rag-maintenance.plist",
         _maintenance_plist(rag_bin)),
        ("com.fer.obsidian-rag-consolidate", "com.fer.obsidian-rag-consolidate.plist",
         _consolidate_plist(rag_bin)),
        # Daily vault transient cleanup (2026-04-27) — barre carpetas de
        # "sistema" bajo 99-obsidian/99-AI/ con TTLs por
        # carpeta y mueve archivos viejos a `.trash/` del vault. Whitelist:
        # `memory/` + `skills/`. Lógica en scripts/cleanup_vault_transient.py.
        ("com.fer.obsidian-rag-vault-cleanup",
         "com.fer.obsidian-rag-vault-cleanup.plist",
         _vault_cleanup_plist(rag_bin)),
        # Anticipatory agent (2026-04-24) — game-changer push proactivo cada
        # 10 min. Comparte daily_cap=3 con emergent/patterns. Silenciable
        # per-kind: `rag silence anticipate-calendar`. Kill global:
        # RAG_ANTICIPATE_DISABLED=1.
        ("com.fer.obsidian-rag-anticipate", "com.fer.obsidian-rag-anticipate.plist",
         _anticipate_plist(rag_bin)),
        # Cross-source ingesters (consolidación 2026-05-04) — antes eran 8
        # plists separados; ahora 2:
        #   - `ingest-whatsapp` queda aparte (cadencia 15 min, hot path).
        #   - `ingest-cross-source` (1h) wrappea gmail/calendar/reminders/
        #     calls/safari/drive/pillow con TTL per-source + gates por
        #     credencial. Los sub-ingesters siguen invocables standalone
        #     via `rag index --source X` para debug / manual refresh.
        ("com.fer.obsidian-rag-ingest-whatsapp",
         "com.fer.obsidian-rag-ingest-whatsapp.plist",
         _ingest_whatsapp_plist(rag_bin)),
        ("com.fer.obsidian-rag-ingest-cross-source",
         "com.fer.obsidian-rag-ingest-cross-source.plist",
         _ingest_cross_source_plist(rag_bin)),
        # Mood signal poller (2026-04-30) — cada 30min junta señales de
        # Spotify + journal + WA outbound + queries + calendar y
        # recomputa el score diario. Behind opt-in: el plist se carga
        # siempre pero el script exit-early si el state file
        # ~/.local/share/obsidian-rag/mood_enabled no existe.
        # Toggle con `rag mood enable` / `rag mood disable`.
        ("com.fer.obsidian-rag-mood-poll",
         "com.fer.obsidian-rag-mood-poll.plist",
         _mood_poll_plist(rag_bin)),
        # Routing rules detector (2026-04-30) — daemon long-running
        # que scanea patrones de comportamiento y sugiere new routes.
        ("com.fer.obsidian-rag-routing-rules",
         "com.fer.obsidian-rag-routing-rules.plist",
         _routing_rules_plist(rag_bin)),
        # Whisper vocabulary refresh (2026-04-30) — nightly extractor
        # para mejorar transcripción de audios WhatsApp.
        ("com.fer.obsidian-rag-whisper-vocab",
         "com.fer.obsidian-rag-whisper-vocab.plist",
         _whisper_vocab_plist(rag_bin)),
        ("com.fer.obsidian-rag-spotify-poll",
         "com.fer.obsidian-rag-spotify-poll.plist",
         _spotify_poll_plist(rag_bin)),
        # Brief schedule auto-tune (2026-04-29) — Sunday 03:00 reads
        # rag_brief_feedback last 30d, decides whether to shift any of
        # morning/today/digest plists' StartCalendarInterval forward
        # (within safe bands), and re-bootstraps only the affected kind.
        # Silent-fail end-to-end — no UX disruption from a stuck cron.
        ("com.fer.obsidian-rag-brief-auto-tune",
         "com.fer.obsidian-rag-brief-auto-tune.plist",
         _brief_auto_tune_plist(rag_bin)),
        # Daemon watchdog — T4 del control plane (2026-05-01). Self-healing
        # loop que corre `rag daemons reconcile --apply --gentle` cada 5min
        # para retry-ear daemons fallidos + kickstart-ear overdues sin tocar
        # el registro de plists. RunAtLoad=true + StartInterval=300 → primer
        # tick inmediato post-bootstrap, después cada 5min. Reemplaza el
        # catchup del difunto serve-watchdog, ahora para TODO el stack.
        ("com.fer.obsidian-rag-daemon-watchdog",
         "com.fer.obsidian-rag-daemon-watchdog.plist",
         _daemon_watchdog_plist(rag_bin)),
        # 2026-05-01: wake hook — sidecar Python loop con KeepAlive=true
        # que polea pmset cada 60s y dispara `rag daemons kickstart-overdue`
        # cuando detecta un wake user-visible nuevo. Resuelve el lag post-
        # Mac-wake del watchdog StartInterval=300 (era hasta 5min). Costo:
        # ~10 MB RSS idle + un fork de pmset cada 60s (~50ms).
        ("com.fer.obsidian-rag-wake-hook",
         "com.fer.obsidian-rag-wake-hook.plist",
         _wake_hook_plist(rag_bin)),
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
})


# Post-consolidación 2026-05-04: los gates de gmail/calendar/drive se
# movieron DENTRO del comando `ingest-cross-source` (skip silencioso +
# cursor update por-source si la creds no existe). Este dict queda sólo
# para labels que aún viven como plist individual en `_services_spec`.
_INSTALL_GATES: dict[str, tuple] = {
    "com.fer.obsidian-rag-mood-poll": (
        _mood_daemon_opted_in,
        "mood-poll es opt-in — activar con [cyan]rag mood enable[/cyan] "
        "+ [cyan]rag setup[/cyan]",
    ),
}


def _services_spec_manual() -> list[dict]:
    """Daemons launchd que existen en disco pero NO tienen factory en código.
    Instalados a mano por el usuario. `rag setup` no los toca; el control
    plane (`rag daemons status / reconcile`) los monitorea pero no los
    regenera. Si alguno se rompe, el fix es manual (re-copiar plist desde
    backup o regenerarlo en su repo origen).

    Limpieza 2026-05-04: se removieron 4 entries que llevaban meses como
    "fantasmas" — sin plist en disco, sin log, con tick `-` en cada
    `daemons status`:

      - `cloudflare-tunnel` + `cloudflare-tunnel-watcher`: se instalan
        aparte cuando el user decide exponer web vía `cloudflared`. Si
        están corriendo, aparecen en `daemons status` por launchctl;
        no hace falta el registry para eso.
      - `lgbm-train`, `paraphrases-train`: jobs de fine-tuning que se
        corren a mano (`rag tune …`), no tienen plist automatizado.
        Dejarlos en el registry les asignaba un slot en el dashboard
        que solo decía "missing".

    Los 3 restantes SÍ tienen histórico de ejecución (logs en
    `~/.local/share/obsidian-rag/{log-rotate,spotify-poll,synth-refresh}.log`
    de 2026-04/05) — se mantienen mientras el user decida si los usa
    o los archiva.
    """
    return [
        {"label": "com.fer.obsidian-rag-synth-refresh", "category": "manual_keep"},
        {"label": "com.fer.obsidian-rag-log-rotate", "category": "manual_keep"},
    ]
