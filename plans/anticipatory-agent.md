# Anticipatory Agent — Implementation Plan

**Fecha**: 2026-04-24
**Feature**: Background daemon que evalúa señales anticipatorias y empuja a WhatsApp el insight más relevante del momento. Convierte el vault de "pull" (yo pregunto) a "push" (el vault me habla cuando es timely).

**Goal**: Game-changer. Que el RAG hable primero cuando tiene algo genuinamente útil que decir, sin spamear.

**Architecture**: Single entrypoint `rag anticipate` que corre N scorers (cada uno produce `AnticipatoryCandidate(kind, score, message, key, snooze_hours)`), merge, dedup por `key` y `rag_proactive_log`, pick top-1 ≥ threshold, envía vía `proactive_push()` existente. launchd cada 10min.

**Tech stack**: Reutiliza toda la infra existente:
- `proactive_push(kind, msg, snooze_hours=N)` → silence + snooze + daily_cap=3 ya manejados
- `_fetch_calendar_today()` + `_fetch_calendar_ahead(days)` → eventos próximos (icalBuddy)
- `followup()` lógica → open loops stale
- `retrieve()` + `embed()` → búsqueda semántica
- `rag_proactive_log` → SQL append-only log
- `rag silence <kind>` → kill-switch per-kind
- launchd + env defaults iguales a `com.fer.obsidian-rag-emergent.plist`

---

## Invariantes (no romper)

- `rag eval` gate sigue pasando (no toca retrieve/rerank).
- 4603 tests en verde post-cambios.
- `proactive_push()` API intacta — solo agregamos callers.
- `PROACTIVE_DAILY_CAP=3` respetado (el agente anticipatorio compite por el mismo cap contra `emergent`/`patterns`).
- Silence + snooze per-kind funcionan igual (`rag silence off anticipate-calendar`).
- Sin JSONL nuevo: todo a SQL vía `_sql_append_event` sobre `rag_proactive_log` + tabla nueva opcional `rag_anticipate_candidates` para analytics.

---

## Scope

**Fase 1 (esta sesión)** — 3 señales mínimas + scheduler + CLI + daemon + tests.

**Fase 2 (documentada, NO ejecutada)** — feedback loop (reply 👍/👎 a los pushes ajusta thresholds), quiet hours contextuales (en reunión + nocturno), TTS morning voice-brief, user-configurable weights.

**Fase 3 (futuro)** — aprender del rate de aceptación (ranker-vivo pero para pushes proactivos).

---

## Phase 1 — MVP runtime

### Task 1 — Data model + CLI skeleton

**Files**:
- Modify: `rag.py` — agregar sección `# ── ANTICIPATORY AGENT ──` (~100 LOC inicial)
- Create: `tests/test_anticipate_agent.py`

**Changes**:

1. Dataclass `AnticipatoryCandidate`:
```python
@dataclass(frozen=True)
class AnticipatoryCandidate:
    kind: str                    # "anticipate-calendar" | "anticipate-echo" | "anticipate-commitment"
    score: float                 # [0, 1]; higher = more urgent/relevant
    message: str                 # WA body, ya formateado con emoji prefix
    dedup_key: str               # ej. event_uid, source_path, loop_id — para idempotencia 24h
    snooze_hours: int            # por-kind default: calendar=2, echo=72, commitment=168
    reason: str                  # debug / explain mode: por qué este candidate
```

2. Nueva tabla `rag_anticipate_candidates` (optional analytics, append-only):
```sql
CREATE TABLE IF NOT EXISTS rag_anticipate_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    score REAL NOT NULL,
    dedup_key TEXT NOT NULL,
    selected INTEGER NOT NULL,   -- 0/1 si fue el top-1 enviado
    sent INTEGER NOT NULL,        -- 0/1 si proactive_push dijo True
    reason TEXT,
    message_preview TEXT
);
CREATE INDEX IF NOT EXISTS ix_rag_anticipate_candidates_ts
    ON rag_anticipate_candidates(ts);
CREATE INDEX IF NOT EXISTS ix_rag_anticipate_candidates_dedup
    ON rag_anticipate_candidates(dedup_key);
```

DDL en `_ensure_telemetry_tables` siguiendo el patrón de `rag_proactive_log`.

3. `_anticipate_dedup_seen(dedup_key, window_hours=24) -> bool`:
```python
def _anticipate_dedup_seen(dedup_key: str, window_hours: int = 24) -> bool:
    """True si ya hubo un candidate con este dedup_key en la ventana."""
    cutoff = (datetime.now() - timedelta(hours=window_hours)).isoformat(timespec="seconds")
    with _ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM rag_anticipate_candidates "
            "WHERE dedup_key = ? AND ts >= ? AND sent = 1 LIMIT 1",
            (dedup_key, cutoff),
        ).fetchone()
    return row is not None
```

4. CLI skeleton:
```python
@cli.group(invoke_without_command=True)
@click.pass_context
def anticipate(ctx):
    """Anticipatory agent — el vault te habla sin que preguntes."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(anticipate_run, dry_run=False, explain=False, force=False)


@anticipate.command("run")
@click.option("--dry-run", is_flag=True, help="Evalúa + loguea pero NO pushea")
@click.option("--explain", is_flag=True, help="Muestra todos los candidates + scores")
@click.option("--force", is_flag=True, help="Bypassea daily_cap + snooze (debug)")
def anticipate_run(dry_run: bool, explain: bool, force: bool):
    """Corre una pasada. Default action del group."""
    ...


@anticipate.command("log")
@click.option("-n", "--limit", type=int, default=20)
def anticipate_log(limit: int):
    """Últimos N candidates de rag_anticipate_candidates."""
    ...


@anticipate.command("explain")
@click.argument("kind", required=False)
def anticipate_explain(kind: str | None):
    """Mostrar el scoring de cada señal para debugging."""
    ...
```

**Tests** (`tests/test_anticipate_agent.py`):
- `test_anticipate_candidate_dataclass_frozen` — kind/score/message/dedup_key/snooze_hours/reason todos required
- `test_anticipate_dedup_seen_returns_false_when_empty`
- `test_anticipate_dedup_seen_returns_true_within_window`
- `test_anticipate_dedup_seen_returns_false_after_window_expires`
- `test_anticipate_cli_dry_run_does_not_push` — mock proactive_push, assert no-op

**Commit**: `feat(anticipate): data model + CLI skeleton (Phase 1.1)`

---

### Task 2 — Señal #1: Calendar proximity

**Files**: `rag.py` (agregar función), `tests/test_anticipate_calendar.py`

**Lógica**:
- Lee `_fetch_calendar_ahead(days_ahead=1)` → eventos en próximas 24h
- Filtra: eventos que empiecen en [15, 90] min desde `now` (sweet spot)
- Para cada evento:
  - Extrae `title`, `participants` (del `notes`/`attendees` del output icalBuddy), `start_at`
  - `score = 1.0 - (minutes_until / 90.0)` → score ∈ [0, 1], más alto = más inminente
  - Retrieve `k=3` con query = `f"{title} {' '.join(participants)}"` sobre el vault
  - Si `retrieve.scores[0] >= 0.25`: armá brief con top-1 path + 1-line snippet
  - `dedup_key = f"cal:{event_uid}"`, `snooze_hours=2` (cada evento no se repushea en 2h — evento próximo ya fue flaggeado)
- Si `retrieve.scores[0] < 0.25`: no emite candidate (evento sin contexto en vault = no vale push)
- Mensaje template:
```
📅 En {minutes_until} min: {title}
{persona_mentions_si_aplica}

Contexto relevante:
· [[{note_title}]] — {snippet_1line}

Score: {confidence_pct}%
```

**Función**:
```python
def _anticipate_signal_calendar(now: datetime) -> list[AnticipatoryCandidate]:
    """Eventos próximos 15-90 min con contexto en vault."""
    events = _fetch_calendar_ahead(days_ahead=1, max_events=40)
    out: list[AnticipatoryCandidate] = []
    for ev in events:
        start = _parse_event_start(ev)  # datetime
        if not start:
            continue
        delta_min = (start - now).total_seconds() / 60.0
        if delta_min < 15 or delta_min > 90:
            continue
        title = ev.get("title", "").strip()
        if not title:
            continue
        # Retrieve contexto
        q = _build_calendar_query(ev)  # title + attendees
        col = get_db()
        result = retrieve(col, q, 3, precise=False, multi_query=False, auto_filter=False)
        if not result["scores"] or result["scores"][0] < 0.25:
            continue
        score = 1.0 - (delta_min / 90.0)
        msg = _format_calendar_brief(ev, result, delta_min)
        dedup_key = f"cal:{ev.get('uid', title[:50])}"
        out.append(AnticipatoryCandidate(
            kind="anticipate-calendar",
            score=score,
            message=msg,
            dedup_key=dedup_key,
            snooze_hours=2,
            reason=f"event in {int(delta_min)}min, top_score={result['scores'][0]:.2f}",
        ))
    return out
```

**Tests** (con mocks de `_fetch_calendar_ahead` y `retrieve`):
- `test_calendar_signal_empty_calendar_returns_empty`
- `test_calendar_signal_event_within_window_emits_candidate`
- `test_calendar_signal_event_too_soon_skipped` (< 15min)
- `test_calendar_signal_event_too_far_skipped` (> 90min)
- `test_calendar_signal_no_vault_context_skipped` (retrieve score < 0.25)
- `test_calendar_signal_score_inversely_proportional_to_time_until` (more imminent = higher)
- `test_calendar_signal_dedup_key_uses_event_uid`

**Commit**: `feat(anticipate): señal calendar proximity + tests`

---

### Task 3 — Señal #2: Temporal echo

**Files**: `rag.py`, `tests/test_anticipate_echo.py`

**Lógica**:
- Lee última nota modificada hoy (file mtime dentro de últimas 6h, en el vault, ≥500 chars para evitar notas triviales)
- Embed del primer párrafo (o primeros 500 chars)
- Retrieve top-5 sobre vault filtrado a `mtime < now - 60d` (solo notas "viejas")
- Si top-1 cosine ≥ 0.70: emite candidate
- `score = top_1_cosine` (directo)
- Mensaje template:
```
🔮 Escribiste hoy algo parecido a lo que pensabas hace {months_ago} meses:

Hoy: [[{today_note_title}]]
Entonces: [[{old_note_title}]] ({old_date})

¿Revisar? Los dos tocan: {shared_theme_keywords}
```

- `dedup_key = f"echo:{today_note_path}:{old_note_path}"`, `snooze_hours=72` (el mismo par no se repushea en 3 días)

**Función**:
```python
def _anticipate_signal_echo(now: datetime) -> list[AnticipatoryCandidate]:
    """Nota de hoy que resuena con una vieja (>60d)."""
    vault = _resolve_vault_path()
    recent = _find_recent_notes(vault, within_hours=6, min_chars=500, limit=3)
    out: list[AnticipatoryCandidate] = []
    for note in recent:
        snippet = _note_first_chars(note, n=500)
        col = get_db()
        result = retrieve(col, snippet, 5, precise=False, multi_query=False, auto_filter=False)
        # Filter to notes older than 60d
        matches = [
            (m, s) for m, s in zip(result["metas"], result["scores"])
            if _note_age_days(m.get("file", "")) > 60
            and m.get("file", "") != str(note.relative_to(vault))
        ]
        if not matches or matches[0][1] < 0.70:
            continue
        old_meta, old_score = matches[0]
        msg = _format_echo_brief(note, old_meta, old_score)
        dedup_key = f"echo:{note.name}:{old_meta.get('file','')}"
        out.append(AnticipatoryCandidate(
            kind="anticipate-echo",
            score=old_score,
            message=msg,
            dedup_key=dedup_key,
            snooze_hours=72,
            reason=f"cosine={old_score:.2f}, age={_note_age_days(old_meta['file'])}d",
        ))
    return out
```

**Helpers**:
```python
def _find_recent_notes(vault: Path, within_hours: int, min_chars: int,
                       limit: int) -> list[Path]:
    """Notas del vault modificadas recientemente. Sorted desc by mtime."""
    cutoff = time.time() - within_hours * 3600
    notes: list[tuple[float, Path]] = []
    for p in vault.rglob("*.md"):
        if is_excluded(str(p.relative_to(vault))):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        if stat.st_mtime < cutoff or stat.st_size < min_chars:
            continue
        notes.append((stat.st_mtime, p))
    notes.sort(reverse=True)
    return [p for _, p in notes[:limit]]


def _note_age_days(file_rel: str) -> float:
    """Edad de una nota en días. Silent-fail → 0."""
    try:
        p = _resolve_vault_path() / file_rel
        return (time.time() - p.stat().st_mtime) / 86400.0
    except Exception:
        return 0.0


def _note_first_chars(p: Path, n: int) -> str:
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        # Strip frontmatter if present
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                text = text[end + 5:]
        return text[:n].strip()
    except Exception:
        return ""
```

**Tests**:
- `test_echo_no_recent_notes_returns_empty`
- `test_echo_recent_note_no_old_matches_returns_empty`
- `test_echo_recent_matches_old_emits_candidate` (mock retrieve para devolver old note)
- `test_echo_excludes_same_note_as_match` (top hit = la misma nota de hoy → skip)
- `test_echo_low_cosine_below_0_70_skipped`
- `test_echo_dedup_key_combines_today_and_old`

**Commit**: `feat(anticipate): señal temporal echo + tests`

---

### Task 4 — Señal #3: Stale commitment push

**Files**: `rag.py`, `tests/test_anticipate_commitment.py`

**Lógica**:
- Reutiliza la lógica de `followup()` (`rag.py:39125+`) — función interna que devuelve `list[OpenLoop]` (ver `_scan_open_loops` / similar)
- Filtra loops con `status == "stale"` y `age_days >= 7`
- Score = `min(1.0, age_days / 30.0)` → capped a 1 cuando el loop tiene ≥30 días
- Toma el más viejo de 1 loop (solo 1 push por run)
- Mensaje:
```
⏰ Hace {days} días dijiste que ibas a hacer algo y no veo señal:

> {loop_quote}

{source_note_link}

¿Avance? Si ya está hecho, `/fix` con la nota resolutoria.
```

- `dedup_key = f"commit:{loop_hash}"` (hash estable del loop text + source)
- `snooze_hours=168` (1 semana — no molestarlo con el mismo loop 2 veces la misma semana)

**Función**:
```python
def _anticipate_signal_commitment(now: datetime) -> list[AnticipatoryCandidate]:
    """Open loops stale ≥7d — push 1 por run."""
    from rag import _scan_all_open_loops  # o el nombre real en followup()
    loops = _scan_all_open_loops()  # devuelve dicts con status/age/quote/source
    stale = [l for l in loops if l.get("status") == "stale" and l.get("age_days", 0) >= 7]
    if not stale:
        return []
    stale.sort(key=lambda l: l["age_days"], reverse=True)
    top = stale[0]
    score = min(1.0, top["age_days"] / 30.0)
    loop_hash = hashlib.sha256(
        (top["quote"] + "|" + top["source"]).encode("utf-8")
    ).hexdigest()[:12]
    msg = _format_commitment_brief(top)
    return [AnticipatoryCandidate(
        kind="anticipate-commitment",
        score=score,
        message=msg,
        dedup_key=f"commit:{loop_hash}",
        snooze_hours=168,
        reason=f"age={top['age_days']}d, quote={top['quote'][:40]}",
    )]
```

**Nota**: hay que identificar la función real del followup que devuelve loops raw. Si `followup()` solo imprime, extraer la lógica a `_scan_all_open_loops()` y dejar que `followup()` lo llame.

**Tests**:
- `test_commitment_no_loops_returns_empty`
- `test_commitment_only_fresh_loops_returns_empty` (age < 7d)
- `test_commitment_stale_loop_emits_candidate`
- `test_commitment_picks_oldest_when_multiple`
- `test_commitment_score_scales_with_age_capped_at_30d`
- `test_commitment_dedup_key_stable_across_runs`

**Commit**: `feat(anticipate): señal stale commitment + tests`

---

### Task 5 — Orchestrator + pick top-1

**Files**: `rag.py` (completar `anticipate_run`), `tests/test_anticipate_orchestrator.py`

**Lógica**:
```python
_ANTICIPATE_MIN_SCORE = float(os.environ.get("RAG_ANTICIPATE_MIN_SCORE", "0.35"))

_ANTICIPATE_SIGNALS = (
    ("anticipate-calendar", _anticipate_signal_calendar),
    ("anticipate-echo",     _anticipate_signal_echo),
    ("anticipate-commitment", _anticipate_signal_commitment),
)


def anticipate_run_impl(
    *, dry_run: bool = False, explain: bool = False, force: bool = False,
) -> dict:
    """Run one pass. Returns {selected: candidate_dict|None, all: [...]}
    para testing + explain mode."""
    now = datetime.now()
    all_candidates: list[AnticipatoryCandidate] = []
    for kind, fn in _ANTICIPATE_SIGNALS:
        try:
            all_candidates.extend(fn(now))
        except Exception as exc:
            _silent_log("anticipate_signal_failed", exc, extra={"kind": kind})
    # Log all candidates (analytics)
    for c in all_candidates:
        _anticipate_log_candidate(c, selected=False, sent=False)

    if not all_candidates:
        return {"selected": None, "all": []}

    # Filter by threshold + dedup
    viable = [
        c for c in all_candidates
        if c.score >= _ANTICIPATE_MIN_SCORE
        and (force or not _anticipate_dedup_seen(c.dedup_key, window_hours=24))
    ]
    viable.sort(key=lambda c: c.score, reverse=True)

    if explain:
        _print_explain(all_candidates, viable)

    if not viable:
        return {"selected": None, "all": [asdict(c) for c in all_candidates]}

    top = viable[0]

    sent = False
    if not dry_run:
        if force:
            # Bypass proactive_can_push (still log it)
            sent = _ambient_whatsapp_send(_ambient_config().get("jid", ""), top.message)
        else:
            sent = proactive_push(top.kind, top.message, snooze_hours=top.snooze_hours)

    _anticipate_log_candidate(top, selected=True, sent=sent)
    return {"selected": asdict(top), "sent": sent, "all": [asdict(c) for c in all_candidates]}
```

**Logging**:
```python
def _anticipate_log_candidate(c: AnticipatoryCandidate, selected: bool, sent: bool) -> None:
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": c.kind,
        "score": c.score,
        "dedup_key": c.dedup_key,
        "selected": 1 if selected else 0,
        "sent": 1 if sent else 0,
        "reason": c.reason,
        "message_preview": c.message[:120],
    }
    def _do():
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_anticipate_candidates", row)
    _sql_write_with_retry(_do, "anticipate_log_failed")
```

**Tests**:
- `test_orchestrator_no_signals_returns_no_selection`
- `test_orchestrator_all_below_threshold_returns_no_selection`
- `test_orchestrator_picks_highest_score_when_multiple`
- `test_orchestrator_skips_dedup_seen` (mock `_anticipate_dedup_seen → True`)
- `test_orchestrator_dry_run_does_not_call_proactive_push`
- `test_orchestrator_force_bypasses_dedup`
- `test_orchestrator_one_signal_error_does_not_block_others`
- `test_orchestrator_logs_all_candidates_to_sql`

**Commit**: `feat(anticipate): orchestrator + top-1 selection + analytics log`

---

### Task 6 — CLI commands complete + explain mode

**Files**: `rag.py` (implementar subcomandos `log`, `explain`), `tests/test_anticipate_cli.py`

**Commands**:

```python
@anticipate.command("log")
@click.option("-n", "--limit", type=int, default=20)
@click.option("--only-sent", is_flag=True)
def anticipate_log(limit: int, only_sent: bool):
    """Últimas N entries de rag_anticipate_candidates."""
    rows = _anticipate_fetch_log(limit=limit, only_sent=only_sent)
    if not rows:
        console.print("[dim]sin registros todavía[/dim]")
        return
    table = Table(title=f"anticipate log (últimos {len(rows)})")
    table.add_column("ts"); table.add_column("kind"); table.add_column("score")
    table.add_column("selected"); table.add_column("sent"); table.add_column("reason")
    for r in rows:
        table.add_row(r["ts"][11:19], r["kind"], f"{r['score']:.2f}",
                      "✓" if r["selected"] else "", "✓" if r["sent"] else "",
                      (r.get("reason") or "")[:60])
    console.print(table)


@anticipate.command("explain")
def anticipate_explain():
    """Muestra scoring de TODAS las señales del momento actual (no pushea)."""
    result = anticipate_run_impl(dry_run=True, explain=True, force=True)
    if not result["all"]:
        console.print("[dim]ninguna señal activa ahora[/dim]")
        return
    # Render en explain format
```

**Tests**:
- `test_anticipate_log_empty_renders_message`
- `test_anticipate_log_renders_rows`
- `test_anticipate_explain_prints_all_candidates` (capture stdout)
- `test_anticipate_run_subcommand_default_invokes_run` (`rag anticipate` sin subcmd = run)

**Commit**: `feat(anticipate): log + explain subcomandos`

---

### Task 7 — launchd daemon + integración en `rag setup`

**Files**:
- Create: `~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist` (dev local)
- Modify: `rag.py` — agregar el plist al `_ALL_SERVICES_SPEC` que maneja `rag setup`
- Create: `tests/test_anticipate_plist.py`

**plist template** (sigue el patrón de `com.fer.obsidian-rag-emergent.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-anticipate</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/fer/.local/bin/rag</string>
    <string>anticipate</string>
    <string>run</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>/Users/fer</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/Users/fer/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>600</integer>  <!-- cada 10 min -->
  <key>StandardOutPath</key><string>/Users/fer/.local/share/obsidian-rag/anticipate.log</string>
  <key>StandardErrorPath</key><string>/Users/fer/.local/share/obsidian-rag/anticipate.error.log</string>
</dict>
</plist>
```

**Integración en `_ALL_SERVICES_SPEC`**:
- Buscar el registry ya existente (`_SERVICES_SPEC` / `_ALL_SERVICES_SPEC` — nombre exacto a confirmar durante ejecución, es lo que `rag setup` usa)
- Agregar entry:
```python
{
    "label": "com.fer.obsidian-rag-anticipate",
    "program_args": ["/Users/fer/.local/bin/rag", "anticipate", "run"],
    "schedule_kind": "interval",
    "schedule_value": 600,  # seconds
    "description": "Anticipatory agent — push timely info sin preguntar.",
},
```

**Tests**:
- `test_anticipate_in_services_spec` (test_services_spec.py drift guard)
- `test_services_spec_total_count` se bumpea de 21 → 22
- Cargar el plist via `rag setup` (dev smoke) y assertir que existe

**Commit**: `feat(anticipate): launchd plist + setup integration`

---

### Task 8 — Docs + CLAUDE.md

**Files**:
- Modify: `CLAUDE.md` — nueva sección sobre Anticipatory Agent
- Modify: `docs/automatizaciones.md` — agregar al catálogo de daemons
- Modify: `docs/como-funciona.md` — agregar al flow overview
- Create: `docs/anticipatory-agent.md` — design doc detallado (señales, scoring, tuning)

**Contenido del design doc** (`docs/anticipatory-agent.md`):
1. Rationale — por qué push > pull para ciertos contextos
2. Arquitectura — scheduler + señales + scoring + dedup
3. Señales activas (3) con ejemplos de mensaje real
4. Tuning: threshold, windows, snooze per-kind
5. Feedback loop (deferred a Fase 2)
6. Cómo agregar una señal nueva (recipe para el futuro)
7. Silenciar: `rag silence off anticipate-calendar` etc.

**Commit**: `docs(anticipate): design doc + CLAUDE.md integration`

---

### Task 9 — Eval gate + manual smoke test

**Files**: solo verificación

**Steps**:

1. Full pytest suite:
```bash
.venv/bin/python -m pytest tests/ -q
```
Todos en verde (4603+ tests).

2. `rag eval` — no regresión:
```bash
rag eval
```
Singles ≥60%, chains ≥73% (floors actuales).

3. Smoke manual:
```bash
# Dry-run (log pero no push)
rag anticipate run --dry-run --explain

# Explain (todas las señales con score)
rag anticipate explain

# Log
rag anticipate log -n 10
```

4. Activar daemon:
```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist
launchctl kickstart -k gui/$(id -u)/com.fer.obsidian-rag-anticipate
tail -f ~/.local/share/obsidian-rag/anticipate.log
```

5. Con un evento real en calendar (mock cuando no hay), verificar push real a WA.

**Commit**: `chore(anticipate): verification + smoke test passed`

---

## Execution order

| Task | Blocking | Est. LoC | Parallel-safe? |
|------|----------|----------|-----------------|
| 1 — data model + CLI skeleton | — | ~80 prod + ~60 test | — |
| 2 — calendar signal | 1 | ~80 prod + ~100 test | con 3 y 4 sí (functions independientes) |
| 3 — echo signal | 1 | ~80 prod + ~100 test | con 2 y 4 sí |
| 4 — commitment signal | 1 | ~60 prod + ~80 test | con 2 y 3 sí |
| 5 — orchestrator | 2, 3, 4 | ~80 prod + ~100 test | — |
| 6 — CLI complete | 5 | ~60 prod + ~60 test | — |
| 7 — launchd | 5 | ~30 prod + ~30 test + plist | con 6 sí |
| 8 — docs | 7 | ~150 prose | con 9 sí |
| 9 — verification | 8 | — | — |

Total estimado: ~560 LOC prod + ~530 LOC tests + ~150 prose docs.

**Checkpoints**: commit + push después de 1, 2+3+4, 5, 6, 7, 8, 9.

---

## Rollback

- Bajar daemon: `launchctl bootout gui/$(id -u)/com.fer.obsidian-rag-anticipate && rm ~/Library/LaunchAgents/com.fer.obsidian-rag-anticipate.plist`
- Silenciar desde CLI sin revertir: `rag silence off anticipate-calendar && rag silence off anticipate-echo && rag silence off anticipate-commitment`
- Kill global: `RAG_ANTICIPATE_DISABLED=1` env var (detectar al inicio de `anticipate_run_impl`, early return)
- Revertir código: `git revert` los commits (cada task es su propio commit).

---

## Fase 2 (NO ejecutada ahora)

### 2.A — Feedback loop
El listener de WhatsApp (whatsapp-listener proyecto separado) detecta replies tipo `👍`/`👎`/`🔇` a mensajes que tengan el prefix de anticipate (ej. emoji 📅🔮⏰ + `dedup_key` en header oculto). Escribe a tabla `rag_anticipate_feedback` que incluye `dedup_key + rating`. Un scorer nuevo penaliza kinds con rating negativo acumulado.

### 2.B — Quiet hours contextuales
- Nocturno: 22h → 8h default → no push excepto calendar crítico (<20min)
- "En reunión": si `_fetch_calendar_today()` muestra evento AHORA → diferir push 15min post-evento
- `rag state focus-code` → silent por TTL del state

### 2.C — Voice brief matinal
- `morning` ya genera brief. Nuevo: reproducirlo vía TTS (`/api/tts` ya existe) + enviar audio file a WA en lugar de texto.
- Hace el brief consumible sin mirar pantalla (manejando, auriculares).

### 2.D — User-configurable weights
- CLI `rag anticipate weights --kind anticipate-echo --weight 0.5` ajusta boost per-kind
- Persist en `~/.local/share/obsidian-rag/anticipate_weights.json`

---

## Self-review

**Spec coverage**:
- ✓ Task 1: data model + CLI skeleton
- ✓ Task 2-4: 3 señales concretas
- ✓ Task 5: orchestrator con dedup + threshold + logging
- ✓ Task 6: CLI (run, log, explain)
- ✓ Task 7: daemon + setup integration
- ✓ Task 8: docs
- ✓ Task 9: verification + eval gate
- ✓ Rollback explícito + kill switches

**Placeholders**: 0 — cada task tiene código concreto, tests con nombres específicos, y commit message.

**Consistency**: `AnticipatoryCandidate` definido en Task 1, usado por Tasks 2-5 con los mismos fields. `_anticipate_dedup_seen` + `_anticipate_log_candidate` mismos nombres en toda la plan.

**Riesgos identificados**:
1. `_scan_all_open_loops` puede no existir como función extraíble — Task 4 puede requerir refactor primero de `followup()`. Mitigación: si es inline-only, crear el helper en Task 4 mismo.
2. `_fetch_calendar_ahead` puede no devolver `uid` en todos los casos — fallback a `title[:50]` para dedup_key.
3. Daily cap de 3 es compartido con emergent + patterns. El anticipate puede ser "silenciado" por otros kinds que consumieron el cap. Mitigación: OK por diseño — el usuario controla vía silence. No aumentar el cap sin su OK explícito.
4. El eval gate NO toca retrieve/rerank, pero el signal de echo usa `retrieve()` — corre full pytest también para catch.
