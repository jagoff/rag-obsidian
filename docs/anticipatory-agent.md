# Anticipatory Agent — the vault talks first

**Game-changer 2026-04-24.** El obsidian-rag deja de ser puramente "pull" (vos preguntás, él responde) y pasa a "push" proactivo: cuando tiene algo genuinamente relevante que decirte, te escribe por WhatsApp sin que le preguntes nada.

No es un chatbot que manda cualquier cosa. Es un agente que **espera** a tener una observación timely y la empuja cuando vale la pena — con budget diario, silenciamiento per-kind, dedup, y threshold tuning.

---

## Rationale — por qué push > pull en ciertos contextos

El workflow actual del RAG asume que vos tomás la iniciativa: abrís el chat, preguntás algo, leés la respuesta. Eso funciona bien cuando sabés que hay algo ahí para preguntar.

Pero hay 3 clases de preguntas donde **vos no sabés que deberías preguntar**:

1. **Calendar proximity** — en 30 min tenés una call con alguien. Tu vault tiene contexto sobre esa persona (últimas interactions, proyectos compartidos, temas abiertos). Normalmente no abrís el chat a preguntar "qué sé de X" antes de cada call. Push: "📅 En 30 min: call con X — contexto: [[X - conversaciones]] score 82%".

2. **Temporal echo** — escribiste hoy una nota sobre coaching. Hace 8 meses escribiste algo casi idéntico y te olvidaste. El vector search no lo sugiere porque no preguntaste nada. Push: "🔮 Lo que escribiste hoy resuena con algo de hace ~8 meses: [[Coaching - 2025-08]]".

3. **Stale commitment** — dijiste hace 11 días que ibas a llamar a tu viejo para preguntarle sobre el asado y no aparece en ninguna nota posterior. Push: "⏰ Hace 11 días dijiste que ibas a llamarlo y no veo señal. ¿Avance?".

Las 3 son preguntas que nunca harías explícitamente, pero que el sistema puede detectar con data que ya tiene.

---

## Arquitectura

```
┌───────────────────────────┐
│  launchd (StartInterval)  │   cada 10 min
│  rag anticipate run       │
└──────────┬────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ anticipate_run_impl()            │
│  - orchestrator                  │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 3 signals (parallel-safe)        │
│  · _anticipate_signal_calendar   │
│  · _anticipate_signal_echo       │
│  · _anticipate_signal_commitment │
└──────────┬───────────────────────┘
           │  cada signal → list[AnticipatoryCandidate]
           ▼
┌──────────────────────────────────┐
│ Filter by score ≥ 0.35           │
│ Dedup 24h via SQL lookup         │
│ Sort by score desc               │
│ Pick top-1                       │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ proactive_push(kind, msg, snooze)│
│  - silence list (rag silence)    │
│  - per-kind snooze               │
│  - daily_cap=3 (shared global)   │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ WhatsApp bridge (ambient config) │
└──────────────────────────────────┘
```

Todo en SQL (`rag_anticipate_candidates` table). Todo local. Cero servicios cloud.

---

## Las 3 señales activas

### 1. Calendar proximity (`anticipate-calendar`)

**Trigger**: evento de hoy que arranca en [15, 90] minutos.

**Score**: `1.0 - (delta_min / 90.0)` — más inminente = score más alto.

**Gate**: el retrieve del título del evento contra el vault tiene que devolver score ≥ 0.25 (si no hay contexto, no tiene sentido empujar).

**Snooze**: 2h (mismo evento no re-pushea en 2h).

**Ejemplo de mensaje**:
```
📅 En 30 min: Coaching call con Juan

Contexto en el vault:
  · [[Coaching - Juan.md]] (02-Areas/Coaching - Juan.md, score 87%)
    > Última sesión: hablamos sobre burnout y el plan de 90 días...
```

**Dedup key**: `cal:{title[:60]}:{YYYY-MM-DDTHH:MM}`

---

### 2. Temporal echo (`anticipate-echo`)

**Trigger**: nota modificada en últimas 6h con ≥500 chars → retrieve top-5 sobre el vault filtrado a notas >60 días de antigüedad.

**Score**: cosine directo (si `≥ 0.70`, emit).

**Lógica**: buscás resonancias con tu vos-del-pasado. El cerebro humano ya tiene esto (déjà vu), el vault hasta ahora no.

**Snooze**: 72h (mismo par hoy-vieja no se repushea en 3 días).

**Ejemplo**:
```
🔮 Lo que escribiste hoy resuena con algo de hace ~8 meses:

Hoy: [[2026-04-24 - coaching reflection]]
Entonces: [[2025-08-15 - coaching insights]] (02-Areas/Coaching/2025-08-15 - coaching insights.md)

Cosine 84%. ¿Mergear, archivar o solo releer?
```

**Dedup key**: `echo:{today_rel}:{old_file}`

---

### 3. Stale commitment (`anticipate-commitment`)

**Trigger**: reutiliza `find_followup_loops()` del comando `rag followup`. Filtra a status=`stale` y age_days ≥ 7. Pickea el más viejo.

**Score**: `min(1.0, age_days / 30.0)` — cap a 1.0 cuando tiene ≥30 días.

**Snooze**: 168h (1 semana).

**Ejemplo**:
```
⏰ Hace 11 días dijiste que ibas a hacer algo y no veo señal:

  > llamar al viejo para preguntarle el asado de abuela

Fuente: [[2026-04-13 - family.md]]

¿Avance? Si ya está hecho, `rag fix` con la nota resolutoria.
```

**Dedup key**: `commit:{sha256(quote + source)[:12]}`

---

## Tuning via env vars

| Var | Default | Qué ajusta |
|----|---------|-----------|
| `RAG_ANTICIPATE_MIN_SCORE` | `0.35` | Threshold global para que un candidate pase al top-1 pick |
| `RAG_ANTICIPATE_DEDUP_WINDOW_HOURS` | `24` | Ventana de dedup por `dedup_key` |
| `RAG_ANTICIPATE_CALENDAR_MIN_MIN` | `15` | Minutos mínimos al evento para triggear push |
| `RAG_ANTICIPATE_CALENDAR_MAX_MIN` | `90` | Ventana máxima (evento más lejos → no emit) |
| `RAG_ANTICIPATE_ECHO_MIN_AGE_DAYS` | `60` | Edad mínima de la nota "vieja" para considerar echo |
| `RAG_ANTICIPATE_ECHO_MIN_COSINE` | `0.70` | Cosine threshold entre nota-hoy y nota-vieja |
| `RAG_ANTICIPATE_COMMITMENT_MIN_AGE_DAYS` | `7` | Age mínimo de un loop stale para push |
| `RAG_ANTICIPATE_DISABLED` | — | Si `1/true/yes`: agent hace early-return (kill switch) |

Todos son `os.environ.get()` al import — hay que reiniciar el daemon para que tomen efecto nuevos valores. Para probar en runtime, usar `rag anticipate explain` + `--force`.

---

## CLI

```bash
rag anticipate              # = rag anticipate run
rag anticipate run          # evalúa señales, empuja top-1
rag anticipate run --dry-run   # loguea pero NO pushea
rag anticipate run --explain   # muestra todos los candidates con scores
rag anticipate run --force     # bypassea dedup + daily_cap
rag anticipate explain      # = --dry-run --explain --force combinados (útil para debug)
rag anticipate log          # últimos 20 candidates (sent + skipped)
rag anticipate log -n 50 --only-sent
```

---

## Silenciar / kill switches

### Per-kind (el resto sigue funcionando)
```bash
rag silence anticipate-calendar
rag silence anticipate-echo
rag silence anticipate-commitment
rag silence anticipate-calendar --off   # reactivar
rag silence --list                       # ver qué está silenciado
```

### Global (el daemon sigue corriendo pero early-returns)
```bash
# Añadir al plist:
launchctl setenv RAG_ANTICIPATE_DISABLED 1
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-anticipate
# Re-activar:
launchctl unsetenv RAG_ANTICIPATE_DISABLED
```

### Nuclear (desinstalar el daemon)
```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist
rm ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist
# y sacar la entry de _services_spec en rag.py (o no, el código es idempotente)
```

---

## Cómo se comparte el budget con otros pushes

`anticipate` usa `proactive_push()` — el mismo pipeline que `rag emergent` y `rag patterns` — que tiene:
- `PROACTIVE_DAILY_CAP = 3` pushes totales por día (sumados a través de kinds)
- Per-kind snooze (tras cada push a un kind, se setea `snooze_hours` — no se re-pushea ese kind hasta que expire)
- Silence list persistente en `~/.local/share/obsidian-rag/proactive.json`

Si el día ya tuvo 3 pushes (de cualquier combinación de `emergent`, `patterns`, `anticipate-*`), el próximo candidate no se envía — se loguea con `sent=0` en `rag_anticipate_candidates` pero nada más.

**Priorización natural**: cada signal tiene snooze propio, y dentro de `anticipate-*` el orchestrator agarra el top-1 por score. Calendar cuando tiene evento cerca gana fácil (score hasta 0.83 a 15min); echo y commitment son más modestos (0.7 típicos) — por diseño, priorizamos lo urgent-importante.

---

## Analytics

Tabla `rag_anticipate_candidates`:
```sql
CREATE TABLE rag_anticipate_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,           -- ISO 8601
    kind TEXT NOT NULL,          -- anticipate-calendar / -echo / -commitment
    score REAL NOT NULL,         -- [0, 1]
    dedup_key TEXT NOT NULL,
    selected INTEGER NOT NULL,   -- 0/1 si fue el top-1 elegido
    sent INTEGER NOT NULL,        -- 0/1 si proactive_push devolvió True
    reason TEXT,
    message_preview TEXT
);
```

TODOS los candidates (incluso los filtrados por score o dedup) se loguean — podés tunear thresholds mirando qué descartó y qué mandó:

```bash
rag anticipate log -n 100
# O SQL directo:
sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db "
  SELECT kind, COUNT(*), AVG(score), SUM(sent)
  FROM rag_anticipate_candidates
  WHERE ts >= date('now', '-7 days')
  GROUP BY kind;
"
```

---

## Agregar una señal nueva (recipe)

1. Escribir `_anticipate_signal_XXX(now: datetime) -> list[AnticipatoryCandidate]` en `rag.py` (cerca de las otras signals).
2. Usar un `kind` con prefix `anticipate-XXX` (respeta la convención — permite silenciar per-kind sin tocar otros).
3. Elegir `snooze_hours` según naturaleza: urgente-recurrente (calendar) 2h; resonancia (echo) 72h; accountability (commitment) 168h.
4. Construir `dedup_key` estable entre runs (hash o natural key). Si el key no es estable, el candidate se va a pushear indefinidamente cada 10 min.
5. Agregar el signal a `_ANTICIPATE_SIGNALS`:
   ```python
   _ANTICIPATE_SIGNALS: tuple[...] = (
       ("calendar", _anticipate_signal_calendar),
       ("echo", _anticipate_signal_echo),
       ("commitment", _anticipate_signal_commitment),
       ("XXX", _anticipate_signal_XXX),
   )
   ```
6. Tests en `tests/test_anticipate_agent.py` — mockear los inputs externos (calendar/retrieve/loops) y verificar que emit/skip/score respeten la lógica.

**Regla**: cada signal debe fallar silenciosamente (try/except + `_silent_log`). Una signal rota no puede tumbar al agente entero.

---

## Roadmap (Fase 2, NO implementado todavía)

- **Feedback loop**: replies con 👍/👎/🔇 a los pushes → ajusta score per-kind o silencia auto.
- **Quiet hours contextuales**: 22h→8h default silent except calendar crítico; detectar "en reunión" via calendar API.
- **Voice briefs matinales**: morning via TTS en WhatsApp (audio file en lugar de texto).
- **User-configurable weights**: `rag anticipate weights --kind anticipate-echo --boost 1.5`.
- **Ranker-vivo para pushes**: learn from acceptance rate qué tipo de push te sirve.

---

## Verificación / smoke

```bash
# Smoke directo (sin push)
rag anticipate explain

# Smoke con dry-run
rag anticipate run --dry-run

# Forzar un push real (bypasseando dedup)
rag anticipate run --force

# Ver log
rag anticipate log -n 20

# Activar daemon
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist
tail -f ~/.local/share/obsidian-rag/anticipate.log

# Desactivar daemon
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist
```

**Tests**: `tests/test_anticipate_agent.py` (38 casos) + `tests/test_setup_online_tune.py::test_services_spec_includes_anticipate` + `tests/test_sql_state_primitives.py` (table count drift guard).
