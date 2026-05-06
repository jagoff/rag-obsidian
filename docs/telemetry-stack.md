# Telemetry stack

Detalle del SQL state store post-T10 (2026-04-19) + post-split (2026-04-21). Resumen + invariantes en [`CLAUDE.md`](../CLAUDE.md).

## Database layout

Dos databases en `~/.local/share/obsidian-rag/ragvec/`:

- **`ragvec.db`** (~104M) — sqlite-vec corpus + 10 state tables (cursors ingesters): `rag_whatsapp_state`, `rag_calendar_state`, `rag_gmail_state`, `rag_reminders_state`, `rag_contacts_state`, `rag_calls_state`, `rag_safari_history_state`, `rag_safari_bookmark_state`, `rag_wa_media_state`, `rag_schema_version`.
- **`telemetry.db`** (~36M) — 45+ tablas operativas: `rag_queries`, `rag_behavior`, `rag_feedback`, `rag_feedback_golden*`, `rag_tune`, `rag_contradictions`, `rag_ambient*`, `rag_brief_*`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`, `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics`, `rag_conversations_index`, `rag_conversation_summaries`, `rag_response_cache`, `rag_entities`, `rag_entity_mentions`, `rag_ocr_cache`, `rag_vlm_captions`, `rag_audio_transcripts`, `rag_learned_paraphrases`, `rag_cita_detections`, `rag_score_calibration`, `rag_chunk_contexts`, `rag_schema_version`.

**Reset total**: `rm ragvec/{ragvec,telemetry}.db && rag index --reset`. Solo telemetría: `rm ragvec/telemetry.db`.

SQL es único storage path (T10 stripped JSONL writers + readers). `RAG_STATE_SQL` removida del código 2026-05-04; los plists la siguen seteando como deployment trail.

## Retention (via `rag maintenance`)

- 90d: `rag_queries`, `rag_behavior`, `rag_cpu_metrics`, `rag_memory_metrics` (30d), `rag_conversation_summaries` (30d), `system_memory_metrics` (30d).
- 60d: `rag_ambient`, `rag_brief_written`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`.
- Keep all forever: `rag_feedback`, `rag_tune`, `rag_contradictions`, `rag_draft_decisions`, `rag_brief_feedback`.

## Primitives (`rag/__init__.py` `# ── SQL state store ──`)

- `_ensure_telemetry_tables(conn)` — idempotent DDL, **ensure-once por (proceso, db_path)** desde commit `09f00bd` (5-8x speedup; agregar entry nueva a `_TELEMETRY_DDL` requiere reiniciar daemons).
- `_ragvec_state_conn()` — short-lived WAL conn `synchronous=NORMAL` + `busy_timeout=10000`.
- `_sql_append_event`, `_sql_upsert`, `_sql_query_window`, `_sql_max_ts`.

**Writer contract**: single-row BEGIN/COMMIT. On exception → log a `sql_state_errors.jsonl` y silently drop. Callers nunca ven exception.

**Reader contract**: SQL-only. Empty snapshots / False / None on error. Retrieval pipeline keeps working.

## Invariantes telemetry stack (audit 2026-04-24 + 2026-04-25)

Cuatro reglas que el código debe respetar:

1. **Todo silent-error sink llama `_bump_silent_log_counter()`** post-write. Sin esto, alerting a stderr (threshold `RAG_SILENT_LOG_ALERT_THRESHOLD=20/h`) queda parcial. Tests: `tests/test_silent_log_alerting.py`.

2. **Async writer = paquete completo de 4 cambios**: (a) helper gate per-writer `_log_X_event_background_default()`, (b) caller con branch sync/async, (c) autouse fixture en conftest setea `RAG_LOG_X_ASYNC=0`, (d) doc del env var. Tests: `tests/test_sql_async_writers.py`.

3. **Readers SQL: retry + stale-cache fallback, nunca empty default que sobrescriba memo**. Modelo: `_load_behavior_priors`, `load_feedback_golden`. Tests: `tests/test_sql_reader_retry.py`.

4. **Tests con TestClient o writers SQL aíslan `DB_PATH` per-file**. NO hay autouse global (intentos conftest-wide reverteados). Pattern obligatorio (snap+restore manual, no `monkeypatch.setattr`):

   ```python
   @pytest.fixture(autouse=True)
   def _isolate_db_path(tmp_path):
       import rag as _rag
       snap = _rag.DB_PATH
       _rag.DB_PATH = tmp_path / "ragvec"
       try: yield
       finally: _rag.DB_PATH = snap
   ```

   Razón snap+restore: `monkeypatch.setattr` revierte en su propio teardown DESPUÉS del teardown de `_stabilize_rag_state` → warning falso. Pollution medida 2026-04-25: 161 test_tag entries en `sql_state_errors.jsonl`, 5 rows `question='test'` en `rag_response_cache`, 57 rows `cmd='web.chat.degenerate'` en `rag_queries`.

**Diagnóstico data-first**: `python scripts/audit_telemetry_health.py --days 7` — primer comando antes de cualquier "auditá el sistema". Agrega los 5 queries que reprodujeron audit 2026-04-24 en 1 segundo.

## Other state (on-disk, no DB)

- `ranker.json` + `ranker.{ts}.json` (3 más recientes) — tuned weights + backups por `rag tune --apply`. Reset: borrar.
- `sessions/*.json` + `last_session` — multi-turn (TTL 30d, cap 50 turns).
- `ambient.json`, `filing_batches/*.jsonl`, `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json`.
- `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink SQL failures.

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden*` en **`telemetry.db`** (post-split). Full re-embed: `rag index --reset`.

## Implicit feedback (reward shaping con negativos débiles)

`rag feedback classify-sessions` backpropaga outcome de session → cada turn. Branches: `win` → `rating=+1`, `loss` → `-1`, `partial` → skip, `abandon` con `top_score < 0.4` → `rating=-1` source `session_outcome_weak_negative`. Treatment training: weight=0.3 (constante `WEAK_NEGATIVE_TRAINING_WEIGHT`) → lambdarank gradient penaliza la mitad. Pre-fix había 542 abandons / 18 losses (asymetría 30:1); post-fix absorbe ~50-100 negativos débiles/semana sin contaminar positivos.

Configs in-code (no env var): `WEAK_NEGATIVE_TOP_SCORE_THRESHOLD=0.4`, `WEAK_NEGATIVE_TRAINING_WEIGHT=0.3`.
