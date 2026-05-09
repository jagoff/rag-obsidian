# Daemons del proyecto

> **Nota 2026-05-09**: refactor "supervisor único in-process" en progreso (F0-F4 done, F5+bootouts pending). El supervisor reemplaza 27 cron daemons con 1 plist + 1 process Python long-running. Coexisten en shadow mode hoy. Ver [Supervisor refactor](#supervisor-refactor-en-progreso) abajo + ADR completo en [`99-AI/system/daemon-refactor-2026-05-09/supervisor-refactor-adr.md`](file:///Users/fer/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/Notes/99-obsidian/99-AI/system/daemon-refactor-2026-05-09/supervisor-refactor-adr.md).

Source de verdad: lista de tuplas en [`rag/plists/_spec.py`](../rag/plists/_spec.py) función `_services_spec()`. Factories por dominio en `rag/plists/{briefs,control,ingest,learning,maintenance,persistent,poll,proactive,wa}.py` + `rag/integrations/whatsapp/plist.py`. Renderer XML schema-driven en `rag/plists/_render.py`. Manuales (no en `_services_spec`): `synth-refresh`, `log-rotate` — trackeados por control plane via `_services_spec_manual()`.

## Resource budget defaults (audit 2026-05-09)

Aplicado a TODOS los plists managed:

- **`ProcessType=Background`** — todos salvo `web` (Interactive) + `watch` (Adaptive). Baja prioridad scheduler vs procesos foreground del user.
- **`LowPriorityIO=true`** — solo en batch nocturno (auto-harvest, online-tune, calibrate, implicit-feedback, whisper-vocab, maintenance, vault-cleanup, consolidate, archive, distill, brief-auto-tune, drift-watcher, active-learning-*, emergent, patterns, wake-up, routing-rules, ingest-cross-source). Bajo I/O contention si user vuelve a la app a las 3 AM.
- **`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`** — todos los plists con `RAG_LLM_BACKEND=mlx`. Evita HEAD requests a HuggingFace en cold-start + previene hang post-Mac-wake si la red está caída.
- **`ThrottleInterval`** — frequent workers: routing-rules (5min) + wa-fast (5min) usan 30s; ingest-* y wa-tasks usan 60s. Evita spawn loops bajo backoff.
- **`ExitTimeOut=10s` en `daemon-watchdog`** — preventivo contra hang en SQL locked.
- **Cadencias bajadas** (no aportaban coverage): anticipate 10min→15min, spotify-poll 60s→5min.
- **Stagger nightly**: calibrate 04:30→05:00 (libera ventana de online-tune que dura ~24min en M-chip).

## Listado actual

`launchctl list | grep obsidian-rag`:

| Plist | Cadencia | Comando | Propósito |
|---|---|---|---|
| `watch` | watcher | `rag watch` | Auto-reindex |
| `serve` | KeepAlive | `rag serve` | MCP server |
| `web` | KeepAlive | `web/server.py` | Web UI + chat |
| `morning` / `today` / `digest` | cal 7am / 22hs / weekly | `rag morning|today|digest` | Briefs |
| `wake-up` | calendar | `rag wake-up` | Setup post-sleep |
| `emergent` / `patterns` | weekly | | Detectores |
| `archive` | weekly | `rag archive` | Auto-archivo dead notes |
| `wa-tasks` / `reminder-wa-push` / `wa-scheduled-send` | 30min / 5min / 5min | | WhatsApp loops |
| `anticipate` | 10min | `rag anticipate` | Anticipatory agent |
| `auto-harvest` / `online-tune` / `calibrate` | weekly / nightly 03:30 / nightly | | Ranker-vivo |
| `maintenance` | weekly | | Vacuum + WAL checkpoint + log rotation |
| `consolidate` | nightly | `rag consolidate` | Memory consolidation Phase 2 |
| `ingest-{whatsapp,gmail,calendar,reminders,drive,calls,safari,pillow}` | varias | `rag index --source X` | |
| `implicit-feedback` / `routing-rules` | 15min / 5min | | |
| `cloudflare-tunnel` + `tunnel-watcher` (manual) | KeepAlive | | HTTPS público |
| `serve-watchdog` | daemon | | (deprecated 2026-05-01, reemplazado por daemon-watchdog) |
| `daemon-watchdog` | 5min | `rag daemons reconcile --apply --gentle` | Control plane retry + kickstart-overdue |
| `active-learning-nudge` | lunes 10am | | WA nudge para labeling |
| `brief-auto-tune` | Domingo 03:00 | `rag brief schedule auto-tune --apply` | |
| `lgbm-train` (manual) | Domingo 02:30 | `rag tune-lambdarank --apply` | |
| `paraphrases-train` (manual) | Domingo 04:30 | | `rag_learned_paraphrases` |
| `spotify-poll` (manual) | 60s | | `rag_spotify_log` |
| `synth-refresh` (manual) | Sábado 22:00 | | Feeder lgbm-train |
| `vault-cleanup` | nightly | | |
| `whisper-vocab` | 03:15 | | Vocab WA refresh |
| `mood-poll` | 30min | | UI no cableada (signals NO en home.v2) |

## Control plane `rag daemons`

- `status [--json --unhealthy-only]` — tabla loaded/running/last_exit/overdue/category.
- `reconcile [--apply --dry-run --gentle]` — converge drift entre `_services_spec()` + lo cargado. `--gentle` = retry exit≠0 + kickstart overdues, NO bootea huérfanos ni regenera plists. Para reconciliación agresiva: `--apply` sin `--gentle`.
- `doctor` — diagnóstico humano + remediation sugerida.
- `retry <label>` — kickstart -k puntual.
- `kickstart-overdue` — catchup post-sleep manual.

Las acciones se loggean a `rag_daemon_runs` (telemetry.db, retention 90d).

## Checklist al agregar plist nuevo

1. Factory `_<nombre>_plist(rag_bin)` + tuple en `_services_spec()`.
2. CLI subcommand.
3. Smoke `rag <subcomando> --dry-run`.
4. Generar plist + copiar a `~/Library/LaunchAgents/`.
5. `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist`.
6. Verificar: `launchctl list | grep` + `launchctl print gui/$UID/com.fer.obsidian-rag-<nombre>`.
7. Esperar tick (o `launchctl kickstart -k`) + verificar log generado.
8. Solo entonces marcar feature como completa.

**Anti-patrón**: dejar TODO "corré `rag setup`" en commit msg. Aprendido 2026-04-25 con `wa-scheduled-send` (commit `9740fa1` — plist nunca se copió, user programó msg, no llegó). Regla universal en [`~/.claude/CLAUDE.md`](file:///Users/fer/.claude/CLAUDE.md) sección "Features con daemons / cron / launchd".

## Supervisor refactor (en progreso)

Refactor F0-F5 que colapsa 27 plists en 1 supervisor in-process (`com.fer.obsidian-rag-supervisor`). Estado por fase:

| Fase | Status | Commit |
|------|--------|--------|
| F0 — Foundations (`rag/runtime/{scheduler,ipc,events,_telemetry}.py`) | ✅ done | `3fe7a7a` |
| F1 — Skeleton + drift-watcher shadow mode | ✅ done | `f2ceb02` |
| F2.1+F2.2 — Nightly batch (6 jobs) + MLX shared warmup | ✅ done | `4d32bd3` |
| F2.3 — A/B 3 noches | ⏳ passive | — |
| F2.4 — Bootout 6 plists nightly | ⏳ post-A/B | — |
| F3.1-F3.4 — Hot-path + proactive + briefs + watchdog | ✅ done | `976d466` |
| F3.5 — Bootout 19 plists migrados | ⏳ post-A/B | — |
| F4.4+F4.5 — mood on-demand IPC + spotify event-driven | ✅ done | `143c881` |
| F4.1+F4.2+F4.3 — SQL triggers + WA bridge hook | ✅ done | `0bb78a5` |
| F5 — Eval baseline + métricas + docs final | ⏳ post-bootout | — |

**Game changer Opción A (consolidación)**: 28 jobs registrados in-supervisor:
- 6 nightly (auto-harvest, whisper-vocab, implicit-feedback, online-tune, maintenance, calibrate).
- 8 hot-path frequent (anticipate, routing-rules, wa-fast, ingest-whatsapp, ingest-cross-source, mood-poll, spotify-poll, wa-tasks).
- 7 proactive (emergent, patterns, archive, distill, active-learning-{nudge,suggest-goldens}, brief-auto-tune).
- 3 briefs (morning, today, digest).
- 3 housekeeping (vault-cleanup, wake-up, consolidate).
- 1 quality (drift-watcher).

**Game changer Opción C (event-reactive)**: 5 paths sub-segundo via supervisor:
- `compute_mood` IPC (web `/api/mood` se calienta on-demand, TTL 30min).
- Spotify `NSDistributedNotificationCenter` listener (track change → DB <100ms).
- `sql.feedback.inserted` trigger → routing-rules in-process (<30s post-INSERT).
- `sql.eval_run.completed` trigger → drift-watcher in-process (<60s).
- `wa.message.inbound` trigger → wa-tasks (≤30s post-mensaje, vs 30min cron).

### CLI: `rag supervisor`

```bash
rag supervisor run               # entrypoint long-running (invocado por launchd)
rag supervisor ping              # health check IPC, latencia ms
rag supervisor status [--json]   # tabla de jobs registrados con stats
rag supervisor jobs              # solo labels
rag supervisor trigger <job>     # dispara <job> sincrónico via IPC
rag supervisor logs [-f] [-n N]  # tail supervisor.log
```

IPC handlers built-in: `ping`, `status`, `jobs`, `run`, `compute_mood`, `invalidate_mood_cache`, `status_spotify`, `status_sql_watchers`.

### Activación manual (shadow mode)

El plist supervisor NO se instala automáticamente todavía (F2.3 A/B passive). Para arrancarlo en paralelo a los plists viejos:

```bash
.venv/bin/python -c "
from rag.plists.persistent import _supervisor_plist
open('/Users/fer/Library/LaunchAgents/com.fer.obsidian-rag-supervisor.plist','w').write(
    _supervisor_plist('/Users/fer/.local/bin/rag'))
"
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-supervisor.plist
rag supervisor status
```

### Opt-out granular

Env vars:
- `RAG_SUPERVISOR_MLX_WARMUP=0` — sin warmup MLX async al startup.
- `RAG_SQL_WATCHERS_DISABLED=1` — apaga los 3 SQL watchers.
- `RAG_F41_DISABLED=1` / `RAG_F42_DISABLED=1` / `RAG_F43_DISABLED=1` — granular.
- `RAG_SPOTIFY_LISTENER_DISABLED=1` — apaga listener PyObjC, queda cron fallback.
