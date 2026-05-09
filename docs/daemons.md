# Daemons del proyecto

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
