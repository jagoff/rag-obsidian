# Code review — pending items

Tracking file for the `code-review` branch audit (19 commits total).
Everything below was flagged in the initial review but deferred from the
main pass; keep as backlog for future iterations.

## Deferred — non-trivial scope

### 1. `except Exception: pass` audit (~100 sites)
**Finding**: rag.py has ~100 silent exception swallows. Many are intentional
(best-effort logging, cache invalidation, optional enrichment) but some
mask real bugs: contradict detector JSON parse, reranker unload, feedback
golden rebuild, `_rebuild_feedback_golden`.

**Why deferred**: sweeping the 100 call sites risks behavioural regressions
in subsystems I didn't verify. P0/P1 sites that had real data-loss
consequences were already addressed (SQL fallback + dead-letter, retry
queue for conversation turns, synthetic_questions sentinel).

**Suggested approach** for a follow-up: add a `_silent_log(where, exc)`
helper that logs at WARN via `_LOG_QUEUE` instead of `pass`. Migrate the
~20 call sites that can't be clearly categorised as "best-effort
ignore" — leave the rest as documented intent.

### 2. Watch debounce race tests
**Finding**: `rag.py:12579` — `watch()` embeds the `Handler`
class + debounce loop directly in the CLI command body. Exception
recovery in `_index_single_file` is handled (`try/except Exception`
inside the loop), but there are no tests for:
- rapid file saves during debounce window
- observer exception recovery
- observer.stop() cleanup under load

**Why deferred**: testing properly requires extracting the handler +
debounce loop to a unit-testable function, which is a non-trivial refactor
of a stable, production-proven code path. The existing
`_index_single_file` error handling is correct; the gap is test
coverage, not behaviour.

### 3. Marked package upgrade check
**Finding**: `web/static/vendor/marked.min.js` is a vendored copy of an
unknown version. The XSS hardening added in `P2.15` (renderer.html override
+ _sanitizeHtml) makes this a non-blocker, but periodic re-vendoring is
still good hygiene.

## False positives from the subagent review

For the record — these were claimed as P0/P1 issues but verified to be
non-issues:

- **"SQL fallback doesn't fall through to JSONL"** (SQL subagent P0 #3).
  Verified at `rag.py:548-559`: the fallthrough is correct. The only
  residual gap (both SQL + JSONL queue.Full) is now covered by the
  dead-letter sink (`_write_dead_letter`).

- **"Timezone mismatch in retention"** (SQL subagent P1 #8). Verified:
  both write path (`datetime.now()`) and read path (`_dt.fromtimestamp`)
  use naive local time. Consistent end to end.

- **"Double-commit in conversation_writer"** (SQL subagent P0 #1).
  `BEGIN IMMEDIATE` + `conn.commit()` is the idiomatic SQLite pattern
  when `isolation_level=None`. No nested-transaction violation — SQLite
  doesn't support nested transactions through this API.

- **"Tables missing from SQL rotation policy"** (SQL subagent P2 #12).
  The `rag_cpu_metrics`, `rag_memory_metrics`, and `system_memory_metrics`
  tables are already in `_SQL_ROTATION_POLICY` (rag.py:25690-25704).

## What shipped

19 commits on `code-review`:

**P0 (critical, shipped)**
- SQL-first / disk-second in `_write_turn_body` (conversation atomicity)
- PRAGMA stampede fix in both `_open_sql_conn` and `_ragvec_state_conn`
- fsync before rename in `_atomic_write`
- Dead-letter sink for SQL+JSONL double-failure
- 30s helper / 90s chat timeout on 19 ollama.chat call sites
- Thread locks on `_corpus_cache` and `_behavior_priors_cache`
- Retry queue on disk for failed conversation turns

**P1 (high, shipped)**
- Pagerank cache keyed on collection_id (was `id(dict)`)
- Synthetic questions sentinel — transient failures no longer cached
- Rate limit on `/api/chat` (30/60s per IP, same bucket pattern as behavior)
- Pydantic validation on ChatRequest.session_id / FeedbackRequest.turn_id
- 17 new MCP server tests (previously zero coverage)
- 10 new deep_retrieve tests (previously zero coverage)
- CI ruff scope expanded to `web/` + `scripts/` (+17 autofixed findings)

**P2 (defensive, shipped)**
- XSS hardening in frontend markdown render (renderer.html escape +
  _sanitizeHtml DOM walker)
- PID suffix on ranker.json backups (avoids cron+manual collisions)
- Two anti-pattern tests tightened (citation-repair stream assertion,
  concurrent writers turn-order assertion)

**Test suite**: 1155 passed / 1 failed (vault-dependent test unrelated
to any change here). Previous master baseline: 1121 passed / 7 failed.
