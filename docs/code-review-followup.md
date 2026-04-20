# Code review — pending items

**Status: all cleared.** Historical record only — the 3 deferred items
from the initial `code-review` branch audit (19 commits) have now been
addressed. Kept as documentation of the rationale + the false-positive
triage from the subagent review.

## Resolved — previously deferred

### 1. `except Exception: pass` audit  — resolved

**Original finding**: rag.py had ~120 silent exception swallows. Most
were intentional (best-effort logging, cache invalidation, optional
enrichment) but a subset could mask real bugs at P0/P1 sites.

**What shipped** (`observability: _silent_log helper + migrate 21
critical except Exception sites`):

- New `_silent_log(where, exc)` helper near the `_LOG_QUEUE` definition
  — records one JSON line per swallowed exception to
  `~/.local/share/obsidian-rag/silent_errors.jsonl` via the same
  non-blocking daemon thread that serves `queries.jsonl` / `behavior.jsonl`.
  Contract: helper never raises; callers keep their silent-fail semantics.
- Migrated 21 data-loss-adjacent sites: collection ops audit log,
  brief_state read+write, diff_brief_signal, session load/list/last,
  context + synthetic-q caches, **ranker.json load** (silent revert to
  defaults is user-invisible), **feedback.jsonl write** (explicit user
  feedback loss), **feedback golden rebuild** (including the zero-vec
  embed fallback that was silently poisoning the ranker with 1-dim
  vectors on ollama outages), **contradict detector** (both helper-chat
  timeout and JSON parse fail — two sites the review explicitly called
  out), and **reranker unload** (called out by name).
- CLI surface: `rag log --silent-errors [--summary]` — list or aggregate
  by (`where` × `exc_type`). Pattern-matching reveals e.g. 50 identical
  `JSONDecodeError`s under `synthetic_q_cache_load` → cache is corrupt;
  delete + move on.
- Tests: `tests/test_silent_log.py` (5 cases: happy path, long-message
  truncation, serialisation failure, queue-full failure, 10-thread
  concurrent writes — all 100 lines parseable) +
  `tests/test_silent_errors_log.py` (6 cases covering the CLI render).

**Sites left as bare `pass`** (genuinely best-effort — noise to log):
- Optional enrichment (Calendar, Reminders, Mail, Screen Time, weather).
- Cache-invalidation cleanups during shutdown; `atexit` best-effort sink.
- HTTP posts to bridges that may be down (whatsapp-bridge).
- chmod on NFS/SMB/FUSE (already documented intent in
  `_ensure_state_dir_secure`).
- Narrow numeric-parse catches (`except (TypeError, ValueError, IndexError)`)
  on per-event fields like behavior.dwell_ms / ts-hour extract — logging
  each would produce N lines per query.

### 2. Watch debounce race tests — resolved

**Original finding**: `watch()` embedded the `Handler` class + debounce
loop directly in the CLI command body. Three edge cases lacked tests:
rapid file saves during the debounce window, per-file exception recovery,
observer cleanup under load.

**What shipped** (`test: add tests/test_silent_log.py +
tests/test_watch.py`):

- Extracted two pure helpers so the logic is testable without spawning
  a real `watchdog.Observer`:
  - `_watch_filter_path(raw_path, vault_path, exclude_folders)` — the
    `.md` + under-vault + exclude-folder filter that used to live in
    `Handler._queue`. Returns `Path | None`. Never raises on malformed
    paths (matches original `try/except ValueError: return`).
  - `_watch_drain_once(vstate, pending_lock)` — drains one vault's
    pending set, calls `_index_single_file` per file, returns a list
    of `(name, status, rel_path, err)` tuples. **Per-file errors never
    abort the batch** (explicit test).
- `watch()` CLI now delegates to both helpers; the visible behaviour of
  the debounce loop + console rendering is unchanged.
- Tests: `tests/test_watch.py` — 14 cases:
  - 7 filter tests (accept `.md`, reject non-`.md`, reject outside
    vault, reject excluded folder, prefix-only match guard,
    folder-itself match, tolerate nonexistent path for deletes).
  - 7 drain tests: empty pending no-op, status propagation, **per-file
    error continuation**, vault-relative rendering, absolute-path
    fallback, **rapid-save dedup via set** (3 files × 20 adds → 3
    indexes), **thread-safe against concurrent producer** (drain twice
    while a thread keeps queueing — no loss, no duplicates).

### 3. `marked.min.js` re-vendor — resolved

**Original finding**: `web/static/vendor/marked.min.js` was a vendored
copy of v14.1.3. XSS hardening made this non-blocking but periodic
re-vendoring is good hygiene.

**What shipped**: updated to **v18.0.2** (latest, 2026-04 release) from
jsdelivr's UMD build. File grew 36 KB → 42 KB. Renderer API (token-based
`link({ href, title, tokens })` + `html({ text })` overrides) is stable
across v4 through v18 — verified with a direct Node smoke test against
the existing renderer config in `web/static/app.js`. The belt-and-
suspenders `_sanitizeHtml` DOM walker is unchanged.

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

## What shipped — cumulative

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

**Followup items (now also shipped)**
- `_silent_log` helper + 21 migrated sites + `rag log --silent-errors`
  CLI + 11 tests across two files
- `_watch_filter_path` + `_watch_drain_once` extraction + 14 tests
- `marked.min.js` re-vendored from v14.1.3 to v18.0.2

**Security + hygiene (landed alongside the followup)**
- 0o700 state dir + 0o600 OAuth token writes (+ 6 tests)
- TOCTOU fix in reranker idle-unload
- Composite SQL indexes on `rag_queries(session,ts)` + `rag_behavior(path,ts)`
- Dependabot weekly pip + monthly github-actions scans
- Removed dead `main.py` uv-init stub
- Corrected drift in CLAUDE.md (line counts, test counts, pipeline facts)

**Test suite**: 1210 passed / 0 failed. Previous master baseline:
1121 passed / 7 failed; mid-branch baseline (before this pass): 1203
passed / 1 failed (vault-unrelated chain referencing dead notes —
queries.yaml pruned as part of this batch).
