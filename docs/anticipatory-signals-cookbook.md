# Anticipatory Agent — Signals Cookbook

Doc complementario a [`docs/anticipatory-agent.md`](anticipatory-agent.md)
(design doc del MVP). Este archivo es el **catálogo completo** de señales
registradas + features Phase 2, con ejemplos concretos, env vars de tuning
y hints sobre cuándo silenciar.

> Arquitectura, rationale, MVP y roadmap general → design doc principal.
> "Cómo se construyó el MVP" → [`plans/anticipatory-agent.md`](../plans/anticipatory-agent.md).
> Comandos operacionales del día a día → [`CLAUDE.md`](../CLAUDE.md) sección Anticipatory Agent.

---

## Señales activas

13 señales total: 3 hardcoded en [`rag.py`](../rag.py) (legacy del MVP del
2026-04-24) + 10 en el package `rag_anticipate.signals.*` (auto-discover
via `pkgutil.iter_modules` al importar `rag_anticipate`).

**Core (3 originales en `rag.py`)** — `anticipate-calendar`,
`anticipate-echo`, `anticipate-commitment`.

**Extension (10 en `rag_anticipate/signals/`)** — `anticipate-anniversary`,
`anticipate-gap`, `anticipate-person_reunion`, `anticipate-deadline`,
`anticipate-inbox_pressure`, `anticipate-streak_break`,
`anticipate-orphan_surface`, `anticipate-dupes_pressure`,
`anticipate-question_awaiting`, `anticipate-reading_backlog`.

Todas comparten el contrato `fn(now: datetime) -> list[AnticipatoryCandidate]`,
silent-fail end-to-end, dedup_key estable cross-runs.

---

## 1. `anticipate-calendar`

- **Trigger**: evento de hoy que arranca en `[15min, 90min]` y tiene contexto
  en el vault con `top_score ≥ 0.25`.
- **Score**: `max(0, min(1, 1 - delta_min / 90))`. 15min → 0.83, 45min → 0.50,
  89min → 0.01.
- **dedup_key**: `f"cal:{title[:60]}:{start.strftime('%Y-%m-%dT%H:%M')}"`.
- **snooze_hours**: `2`.
- **Mensaje**: `📅 En 30 min: Standup equipo plataforma\n\nContexto en el
  vault:\n  · [[standup-template]] (..., score 78%)\n  > Hoy review de
  migrations + wins...`
- **Env vars**: `RAG_ANTICIPATE_CALENDAR_MIN_MIN` (default `15`),
  `RAG_ANTICIPATE_CALENDAR_MAX_MIN` (default `90`).
- **Cuándo silenciar**: si todo tu calendario son meetings recurrentes
  obvios. `rag silence anticipate-calendar`.

---

## 2. `anticipate-echo`

- **Trigger**: nota tocada en las últimas 6h (≥500 chars) que matchea
  semánticamente con una nota vieja (≥60d) con cosine `≥ 0.70`.
- **Score**: `score = old_score` (cosine de la vieja, no normalizado).
- **dedup_key**: `f"echo:{today_rel}:{old_meta['file']}"`.
- **snooze_hours**: `72` (3 días — no repetir la misma resonancia).
- **Mensaje**: `🔮 Lo que escribiste hoy resuena con algo de hace ~14
  meses:\n\nHoy: [[2026-04-25-thinking-loud]]\nEntonces:
  [[burnout-signals]] (...)\n\nCosine 78%. ¿Mergear, archivar o solo releer?`
- **Env vars**: `RAG_ANTICIPATE_ECHO_MIN_AGE_DAYS` (default `60`),
  `RAG_ANTICIPATE_ECHO_MIN_COSINE` (default `0.70`).
- **Cuándo silenciar**: vault con mucha redundancia temática que trae los
  mismos echos toda la semana → subir threshold a `0.80` o silenciar.

---

## 3. `anticipate-commitment`

- **Trigger**: open loops detectados por `find_followup_loops()` con
  `status="stale"` y `age_days ≥ 7`.
- **Score**: `max(0, min(1, age_days / 30))`. 7d → 0.23, 15d → 0.50, 30+d → 1.0.
- **dedup_key**: `f"commit:{sha256(loop_text + '|' + source_note)[:12]}"`.
- **snooze_hours**: `168` (1 semana).
- **Mensaje**: `⏰ Hace 12 días dijiste que ibas a hacer algo y no veo
  señal:\n\n  > Le mando los thumbnails a Marcos antes del viernes.\n\n
  Fuente: [[2026-04-13-call-marcos]]\n\n¿Avance? Si ya está hecho, \`rag fix\`...`
- **Env vars**: `RAG_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS` (default `7`).
- **Cuándo silenciar**: muchos loops "informativos" que se cierran sin nota
  explícita → preferí `rag fix` o subir el threshold de age.

---

## 4. `anticipate-anniversary`

- **Trigger**: nota en `01-Projects/`, `02-Areas/`, `03-Resources/` que
  cumple 360-370d (target 365), ≥500 chars. Lee `created:` del frontmatter,
  fallback a `mtime`.
- **Score**: `1 - |365 - age_days| / 10`, clamp `[0, 1]`. Día 365 → 1.0;
  ±5d → 0.5; fuera de la ventana → no emite.
- **dedup_key**: `f"anniv:{file_rel}:{year_created}"`.
- **snooze_hours**: `720` (30 días — no relanzar la misma anniversary 2×).
- **Mensaje**: `🎂 Hace 1 año escribiste: [[2025-04-25-quarterly-retro]]\n
   > Q1 cerró con foco en migration backend...\n\n¿Releer, actualizar o archivar?`
- **Env vars**: ninguna directa (constantes hardcoded en el módulo).
- **Cuándo silenciar**: <1 año usando el vault → silent natural. Ping
  nostalgia molesto → `rag silence anticipate-anniversary`.

---

## 5. `anticipate-gap`

- **Trigger**: cluster de ≥3 queries similares (cosine de embeddings ≥0.75)
  en los últimos 14d donde la mejor cobertura (top-1 retrieval) tiene
  `score < 0.30` → realmente no hay nota cubriendo el tema.
- **Score**: `min(1.0, len(cluster) / 10.0)`. 3 queries → 0.30, 10+ → 1.0.
- **dedup_key**: `f"gap:{sha256(rep)[:12]}"` (`rep` = query más corto del
  cluster, forma canónica).
- **snooze_hours**: `168` (1 semana).
- **Mensaje**: `🧭 4 veces preguntaste algo como 'cómo configuro tunnels
  de Cloudflare en el Mac' en 14d sin nota que lo cubra. ¿Capturar síntesis?`
- **Env vars**: ninguna directa. Reusa la tabla `rag_queries`.
- **Cuándo silenciar**: si usás `rag emergent` regularmente ya tenés esto
  cubierto en modo interactivo → silenciar evita doble-ping.

---

## 6. `anticipate-person_reunion`

- **Trigger**: nota modificada en las últimas 6h (≥300 chars) que menciona
  un wikilink que parece nombre propio (`[[Nombre Apellido]]`, 1-4 palabras
  capitalizadas, sin paths ni dígitos), donde la última mención previa de
  esa persona en el vault fue hace **>30 días**.
- **Score**: `min(1.0, gap_days / 180.0)`. 30d → 0.17, 90d → 0.50, 180+d → 1.0.
- **dedup_key**: `f"reunion:{person}:{today_file_rel}"`.
- **snooze_hours**: `72`.
- **Mensaje**: `👋 Después de 87 días mencionás a [[María Rodríguez]] de
  nuevo en [[2026-04-25-coffee-chat]]. Última nota:
  [[2026-01-29-onboarding-call]] (2026-01-29).`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: vault con muchísimos wikilinks a personas random
  (transcripts de podcasts, CVs, etc.) → falsos positivos. Bajar weight a
  `0.5` o silenciar.

---

## 7. `anticipate-deadline`

- **Trigger**: nota con frontmatter `due:` cuya fecha cae en `[hoy, hoy+3d]`.
  Acepta `2026-04-28`, `"2026-04-28T10:00"`, `"28/04/2026"`, listas YAML.
- **Score**: `1.0 - days_until / 4.0`. Hoy → 1.0, mañana → 0.75, +2d → 0.50,
  +3d → 0.25.
- **dedup_key**: `f"deadline:{rel}:{due_iso}"`.
- **snooze_hours**: `24` (1×/día máx por deadline).
- **Mensaje**: `📌 Deadline en 1 días: [[Q2-roadmap-deck]]\n  Due:
  2026-04-26\n  ¿Avance? Si ya está hecho, marcarlo.`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: si usás otra herramienta para deadlines (Things,
  Reminders, Linear) y no querés double-tracking → silenciar.

---

## 8. `anticipate-inbox_pressure`

- **Trigger**: ≥15 archivos `.md` en `<vault>/00-Inbox/` top-level (no
  recursivo — ignora `00-Inbox/conversations/` que es episodic memory del
  web server) con `mtime` >24h.
- **Score**: `0.4 + (count - 15) / 20.0`, clamp `[0, 1]`. count=15 → 0.4,
  count=25 → 0.9, count=35+ → 1.0.
- **dedup_key**: `f"inbox_pressure:{today_iso_date}"` — máx 1 push/día.
- **snooze_hours**: `48`.
- **Mensaje**: `📥 Inbox acumulado: 22 notas sin triar (>24h). ¿\`rag inbox
  --apply\` o triage manual?`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: workflow donde el inbox es buffer intencional de N
  días y nunca lo procesás bajo presión → silenciar.

---

## 9. `anticipate-streak_break`

- **Trigger**: ≥2 días sin morning brief (filename `YYYY-MM-DD.md` en
  `04-Archive/99-obsidian-system/99-AI/reviews/`) dentro de la ventana de 7d. **Si NO hay ningún brief en
  los últimos 7d** asume "pausa voluntaria" y NO emite (no spam-ear vacaciones).
- **Score**: `min(1.0, gap_days / 5.0)`. 2d → 0.4, 3d → 0.6, 5+d → 1.0.
- **dedup_key**: `f"streak_break:{today_iso_date}"`.
- **snooze_hours**: `24`.
- **Mensaje**: `🔥 Racha rota: hace 3 días que no tirás morning brief.
  ¿Retomás hoy con \`rag morning\`?`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: si moviste el ritual a evening brief exclusivamente
  (`YYYY-MM-DD-evening.md`) → silenciar (sólo cuenta morning briefs).

---

## 10. `anticipate-orphan_surface`

- **Trigger**: nota en `01-Projects/`, `02-Areas/`, `03-Resources/`
  modificada en `[now-24h, now-2h]` (grace de 2h post-save), ≥200 chars,
  con **0 wikilinks outgoing**.
- **Score** (bands fijos por tamaño): `≥3000 chars → 0.9`, `≥1500 → 0.8`,
  `≥500 → 0.6`, `≥200 → 0.4`.
- **dedup_key**: `f"orphan:{file_rel}"`.
- **snooze_hours**: `24`.
- **Mensaje**: `🔗 Nota nueva sin links: [[reading-notes-anti-fragile]]\n
  Tamaño: 1842 chars, 0 wikilinks outgoing.\n  ¿Correr \`rag wikilinks
  suggest --path 03-Resources/...\`?`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: si capturás muchas atomic notes / fleeting que no
  necesitan links salientes → silenciar para evitar ruido sistemático.

---

## 11. `anticipate-dupes_pressure`

- **Trigger**: ≥5 pares de notas con stems de filename muy similares
  (`difflib.SequenceMatcher.ratio() ≥ 0.85` sobre nombres normalizados —
  lowercase + strip de no-alfanuméricos). Cap defensivo de 2000 archivos
  escaneados.
- **Score**: `0.5 + (count - 5) / 15.0`, clamp `[0, 1]`. 5 pares → 0.5,
  12 pares → ~0.97, 20+ → 1.0.
- **dedup_key**: `f"dupes_pressure:{ISO_year}-W{ISO_week:02d}"` — máx 1×/semana.
- **snooze_hours**: `336` (2 semanas).
- **Mensaje**: `👥 7 pares de posibles duplicados acumulados. ¿\`rag dupes
  --threshold 0.85\` para revisar y mergear?`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: sistema de IDs en filenames (`note-001.md`,
  `note-002.md`) que da falsos positivos sistemáticos → silenciar y usar
  `rag dupes` manual.

---

## 12. `anticipate-question_awaiting`

- **Trigger**: row en `rag_wa_tasks` con `kind='question'` en los últimos
  14d, ≥3d de edad, **sin** un mensaje posterior del user (`user='me'`)
  en el mismo `source_chat` dentro de los 3d siguientes.
- **Score**: `min(1.0, days_since_asked / 14.0)`, redondeado a 4 decimales.
  3d → 0.21, 7d → 0.50, 14+d → 1.0.
- **dedup_key**: `f"awaiting:{chat_id}:{question_ts_iso_date}"`.
- **snooze_hours**: `168`.
- **Mensaje**: `💬 Che, ya viste lo del cambio de horario del miércoles? —
  pregunta sin respuesta hace 5 días en WhatsApp. ¿Responder?`
- **Env vars**: ninguna directa. Depende del schema extendido de
  `rag_wa_tasks` (columnas `kind`, `source_chat`, `message_preview`, `user`).
  Si el schema sigue siendo el legacy de invocaciones, la signal silencia
  via `OperationalError`.
- **Cuándo silenciar**: si no querés que el agent te recuerde mensajes
  de WA por privacidad / disciplina → silenciar.

---

## 13. `anticipate-reading_backlog`

- **Trigger**: ≥10 notas con `mtime` >7d que califican como to-read:
  - frontmatter `status: to-read` o `status: unread`
  - frontmatter `tags:` que contiene `to-read` o `unread`
  - inline `#to-read` o `#unread` en el body
  - O viven en `03-Resources/Reading/` (folder convention)
- **Score**: `0.5 + (count - 10) / 30.0`, clamp `[0, 1]`. count=10 → 0.5,
  count=25 → 1.0, count=40+ → 1.0.
- **dedup_key**: `f"reading_backlog:{ISO_year}-W{ISO_week:02d}"` — máx 1×/semana.
- **snooze_hours**: `168`.
- **Mensaje**: `📚 14 notas en backlog de lectura (≥7d). ¿Clear session
  de 1h o archivar?`
- **Env vars**: ninguna directa.
- **Cuándo silenciar**: si usás Pocket / Readwise / Instapaper para reading
  list y el vault es solo para notas post-lectura → silenciar.

---

## Features Phase 2 (módulos en `rag_anticipate/`)

Helpers complementarios al orchestrator MVP. **Construidos pero no todos
integrados** todavía al loop de `anticipate_run_impl()` — el plan es ir
cableándolos progresivamente.

### `feedback` ([`rag_anticipate/feedback.py`](../rag_anticipate/feedback.py))

API pública:

- `record_feedback(dedup_key, rating, *, reason="", source="wa") -> bool`
- `parse_wa_reply(reply_text) -> "positive" | "negative" | "mute" | None`
- `feedback_stats(kind=None, days=30) -> dict`
- `recent_feedback(limit=20) -> list[dict]`

Tabla nueva `rag_anticipate_feedback` con columns `(id, ts, dedup_key,
rating, source, reason)`. Created idempotently via `_ensure_feedback_table()`.

`parse_wa_reply` reconoce:

- `👍` / `👌` / `✅` / `:thumbsup:` / `util` / `si` / `ok` → `positive`
- `👎` / `🚫` / `❌` / `:thumbsdown:` / `no` / `noutil` → `negative`
- `🔇` / `🙅` / `:mute:` / `silenciar` / `basta` → `mute`

Precedencia conservadora: `mute > negative > positive` (si manda "👍 pero
👎", asumimos descontento).

```python
from rag_anticipate.feedback import parse_wa_reply, record_feedback

reply = "👍 sirvió mucho"
rating = parse_wa_reply(reply)  # → "positive"
if rating:
    record_feedback("cal:standup:2026-04-25T10:00", rating, reason=reply)
```

### `quiet_hours` ([`rag_anticipate/quiet_hours.py`](../rag_anticipate/quiet_hours.py))

API pública:

- `is_night(now: datetime) -> bool`
- `is_in_meeting(now: datetime) -> bool`
- `is_focus_state() -> bool`
- `is_quiet_now(now: datetime) -> tuple[bool, str]`

Reglas:

- **Night** — ventana default `22:00 → 08:00` (wrap-around). Override con
  `RAG_ANTICIPATE_QUIET_NIGHT_START` y `RAG_ANTICIPATE_QUIET_NIGHT_END`
  (formato `HH:MM` 24h).
- **Meeting** — hay un evento del calendario en curso ahora. Lee
  `rag._fetch_calendar_today()` y parsea `start`/`end` HH:MM o H:MM AM/PM.
- **Focus** — el state actual contiene keyword `focus`, `deep-work`,
  `concentrado`, o `no molestar` (substring case-insensitive).
- **Bypass total**: `RAG_ANTICIPATE_BYPASS_QUIET=1` (o `"true"`) fuerza
  `(False, "")`.

Precedencia de razones: `night > focus > meeting` (determinístico para tests).

Cómo integrar al orchestrator (futuro): early-return de `anticipate_run_impl()`
si `is_quiet_now(now)[0]` es `True`, logueando `reason`.

### `weights` ([`rag_anticipate/weights.py`](../rag_anticipate/weights.py))

API pública:

- `load_weights() -> dict[str, float]`
- `save_weights(weights) -> bool`
- `set_weight(kind, weight) -> bool`
- `remove_weight(kind) -> bool`
- `apply_weight(kind, score) -> float`
- `list_weights() -> list[tuple[str, float]]`

Persistencia: `~/.local/share/obsidian-rag/anticipate_weights.json` (JSON
flat dict). Range válido `[0.0, 5.0]` por weight (out-of-range silenciosamente
ignorado). Atomic write (tmp + rename).

```python
from rag_anticipate.weights import set_weight, apply_weight

set_weight("anticipate-calendar", 1.5)        # boost calendar 50%
set_weight("anticipate-orphan_surface", 0.5)  # half-mute orphan

final = apply_weight("anticipate-calendar", candidate.score)
# final == min(1.0, score * 1.5)
```

### `dashboard` ([`rag_anticipate/dashboard.py`](../rag_anticipate/dashboard.py))

API pública:

- `fetch_metrics(days=7) -> dict`
- `render_dashboard(days=7) -> str`
- `top_reasons_skipped(days=7, limit=10) -> list[tuple[str, int]]`
- `signal_health(days=7) -> dict`

Útil para diagnóstico: cuál signal genera ruido (`status="noisy"`), cuál
nunca emite (`status="silent"`), cuál se cayó (`status="stale"`), cuál
está OK (`status="healthy"`).

Reglas de `signal_health` (priority en orden, primer match gana):

1. `silent` — `last_emit is None` Y 0 emits en ventana.
2. `noisy` — `≥10` emits en ventana Y `avg_score < 0.3`.
3. `stale` — 0 emits en ventana pero `last_emit` existe.
4. `healthy` — caso normal.

Output ejemplo `render_dashboard(7)`:

```
Anticipate Dashboard (7 days)
=============================
Total evaluated: 142
Total selected:  18
Total sent:      14
Send rate:       9.9%
Selection rate:  12.7%

By signal:
  anticipate-calendar     45 eval / 8 sel / 7 sent  avg_score=0.62
  anticipate-echo         28 eval / 4 sel / 3 sent  avg_score=0.74
  anticipate-orphan_surf  35 eval / 2 sel / 2 sent  avg_score=0.58
```

### `voice` ([`rag_anticipate/voice.py`](../rag_anticipate/voice.py))

API pública:

- `text_to_audio(text, *, voice="Monica", out_dir=None) -> Path | None`
- `is_tts_available() -> bool`
- `cleanup_old_briefs(max_age_days=7, *, out_dir=None) -> int`

Reusa el endpoint `/api/tts` del web server existente
(`http://127.0.0.1:8765/api/tts` por default; override con
`RAG_WEB_BASE_URL`). Voz default `Monica` (español ES).

Cache de briefs en `~/.local/share/obsidian-rag/voice_briefs/brief-<sha256[:12]>.wav`
— mismo texto reusa el archivo. `cleanup_old_briefs(7)` elimina los wav de
más de 7 días.

Phase 2 plan: WhatsApp voice notes en lugar de texto, usando el sha del
texto como key estable (mismo brief = mismo audio = mismo file en cache).

### `lockfile` ([`rag_anticipate/lockfile.py`](../rag_anticipate/lockfile.py))

API pública:

- `with anticipate_lock(*, timeout_seconds=0.0) as acquired: ...`
- `lock_status() -> dict`

Path: `~/.local/share/obsidian-rag/anticipate.lock`. Cooperative via
`fcntl.flock(LOCK_EX | LOCK_NB)`. Default non-blocking (timeout=0); con
`timeout_seconds>0` espera hasta N segundos antes de devolver `False`.

El lockfile contiene `pid=N ts=T` (texto) para diagnostics. `lock_status()`
parsea sin acquirir y reporta `{"held": bool, "pid": int|None, "ts": int|None}`.

Importante para evitar runs concurrent del daemon (launchd doble + un
`rag anticipate run` manual al mismo tiempo, o dos `launchctl kickstart`
en rapid-fire).

```python
from rag_anticipate.lockfile import anticipate_lock, lock_status

with anticipate_lock() as acquired:
    if not acquired:
        return  # otro proceso ya está corriendo
    # ... do work ...

print(lock_status())
# {'held': False, 'pid': 12345, 'ts': 1745532000}
```

---

## Tuning end-to-end

### Si recibís muy pocos pushes

- Bajá `RAG_ANTICIPATE_MIN_SCORE` (default `0.35` → probá `0.25`).
- Boost weights de signals específicas: `set_weight("anticipate-X", 1.5)`.
- Verificá que `daily_cap=3` no se está agotando con `emergent` / `patterns`
  antes que anticipate (budget global compartido).
- Revisá `signal_health(7)`: si todo está `silent`, capaz no hay data
  fuente (calendar empty, vault no accesible).

### Si recibís demasiado ruido

- Subí `RAG_ANTICIPATE_MIN_SCORE` (`0.35 → 0.5`).
- Silenciá signals específicas: `rag silence anticipate-X`.
- Reducí weights con `set_weight("anticipate-X", 0.5)` (no las eliminás —
  siguen evaluándose y logueándose en el dashboard, sólo bajan en el ranking).
- Configurá quiet hours más amplias:
  `RAG_ANTICIPATE_QUIET_NIGHT_START=20:00`,
  `RAG_ANTICIPATE_QUIET_NIGHT_END=10:00`.

### Si una signal es `silent` en el dashboard

- Check si la fuente de data existe (calendar events visibles, frontmatter
  `due:` poblado, vault notes en los buckets esperados).
- Reducí su threshold-específico (ej. `RAG_ANTICIPATE_ECHO_MIN_COSINE`
  `0.70 → 0.60`).
- Ejecutá `rag anticipate run --dry-run --explain` y mirá la columna de
  reason — los candidates eliminados por threshold se loguean igual.

### Si una signal es `noisy` (avg_score <0.3)

- Reducí weight: `set_weight("anticipate-X", 0.5)`.
- O subí el threshold global `RAG_ANTICIPATE_MIN_SCORE`.
- Inspeccioná `top_reasons_skipped(7, 10)` para ver por qué la mayoría no
  llegan al top-1.

### Kill-switch global

`RAG_ANTICIPATE_DISABLED=1` en el plist
(`com.fer.obsidian-rag-anticipate.plist`) → el daemon hace early-return
sin evaluar nada. Útil para vacaciones o debugging del web server sin
ruido del agent.

---

## Ver también

- [Design doc principal](anticipatory-agent.md) — arquitectura, rationale,
  MVP, contrato de signals.
- [Plan de ejecución](../plans/anticipatory-agent.md) — cómo se construyó
  el MVP del 2026-04-24 y el roadmap Phase 2.
- [`CLAUDE.md` sección Anticipatory Agent](../CLAUDE.md) — comandos
  operacionales del día a día (kickstart, silenciar, dry-run, kill switches).
- [`rag_anticipate/__init__.py`](../rag_anticipate/__init__.py) — contrato
  formal que cada signal nueva debe cumplir + cómo agregar una.
