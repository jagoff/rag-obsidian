# Episodic Memory for obsidian-rag — Implementation Plan

**Feature**: `/api/chat` auto-documents every conversation as a markdown note in the Obsidian vault. Notes are picked up by the existing watcher and become retrievable via `rag query` and `/api/chat` itself (self-citation).

**Status**: Phase 0 done. Phase 1 ready to execute.

**Scope**:
- v0 (Phase 1): web `/chat` only. Session-grouped notes in `00-Inbox/conversations/`.
- v1 (Phase 2): weekly consolidation → promote recurring topics to PARA.
- v2 (Phase 3, documented, not executed): WhatsApp listener integration.

**Design decisions (locked)**:
- **a1**: Web `/chat` only for v0.
- **b1**: One note per conversation (multi-turn, appended by `session_id`).
- **c3**: Folder indexed by existing `com.fer.obsidian-rag-watch` — no separate pipeline.

---

## Phase 0 — Documentation Discovery (COMPLETED)

### Allowed APIs (verified, with sources)

**Server integration (`web/server.py`)**
- Chat endpoint: `@app.post("/api/chat")` at `web/server.py:2646`, function `chat(req: ChatRequest) -> StreamingResponse` at line 2647.
- Request schema: `ChatRequest(BaseModel)` at lines 302-306 with fields `question: str`, `session_id: str | None`, `vault_scope: str | None`.
- SSE utility: `_sse(event, data)` at lines 1119-1120.
- Session generation: line 2652 `sid = req.session_id or f"web:{uuid.uuid4().hex[:12]}"`; validated via `ensure_session(sid, mode="chat")` at line 2653.
- `done` event emission: lines 3079-3086 yields `{turn_id, top_score, total_ms, retrieve_ms, ttft_ms, llm_ms}`.
- **Hook point**: immediately after `done` yield (line 3086), before `_emit_enrich()` at 3088. Variables in scope:
  - `sess["id"]` — session id (persisted at line 3068)
  - `turn_id` — unique turn UUID (line 3060)
  - `req.question` — user query (line 2648)
  - `full` — accumulated LLM response (line 3031)
  - `result["metas"]` and `result["scores"]` — source metadata (post-rerank)
  - `result["confidence"]` — top rerank score
- Source payload shape: `_source_payload(meta, score)` at `web/server.py:1123-1129` returns `{file, note, folder, score, bar}`.
- Async pattern to follow: `threading.Thread(target=..., daemon=True).start()` as used at line 245 (simpler than ThreadPoolExecutor for true fire-and-forget).

**Watcher indexing (`rag.py`)**
- Watcher entry: `watch()` at `rag.py:9092-9222`. Uses watchdog `Observer` + 3s debounce loop.
- Vault resolution: `_resolve_vault_path()` at lines 385-395. Precedence: `OBSIDIAN_RAG_VAULT` env → `vaults.json` current → `_DEFAULT_VAULT` (line 348, `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`).
- Recursive scan: `observer.schedule(Handler(vs), str(vs["path"]), recursive=True)` at line 9188 — entire vault tree.
- File filter: `if not raw_path.endswith(".md"): return` at line 9155.
- Exclusion layers:
  1. Dot-folders hardcoded via `is_excluded(rel_path)` at lines 1053-1057.
  2. Env-configurable `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` (default `"03-Resources/WhatsApp"`) at lines 9111-9119.
- **VERIFIED**: `00-Inbox/conversations/` is NOT excluded. Will be indexed automatically.
- Incremental logic: SHA256-hash comparison at lines 8768-8785 (mtime ignored). Writing to an existing file with same content = skipped.
- Vector store: sqlite-vec at `~/.local/share/obsidian-rag/ragvec/` (line 399, `DB_PATH`). Collection `obsidian_notes_v9` (or `obsidian_notes_v9_{sha256[:8]}` for non-default vaults) via `get_db_for(vault_path)` at lines 2581-2595.
- Watcher log: `/Users/fer/.local/share/obsidian-rag/watch.log` (plist stdout).

**Project conventions**
- Python 3.13+, `uv tool install --reinstall --editable .` after pyproject edits.
- Entrypoints at `pyproject.toml:26-28`: `rag = "rag:cli"`, `obsidian-rag-mcp = "mcp_server:main"`.
- No structlog/loguru. Logging via `_LOG_QUEUE.put((path, json_line))` pattern at `rag.py:410-448`. One daemon writer thread consumes queue.
- Test style (`tests/test_ambient.py`, `tests/test_inbox.py`): `monkeypatch.setattr(rag, "VAULT_PATH", tmp_path / "vault")` — never touch real FS. Embeddings mocked via `fake_embed` fixture. `conftest.py` autouse fixture clears `_embed_cache` + `_expand_cache`.
- Eval baselines (must not regress): singles hit@5 88.10%, MRR 0.772, p95 2447ms.
- Config: no `.env`, no pyproject. Vault path via env var or `~/.local/share/obsidian-rag/vaults.json`.

### Anti-patterns (do NOT do)

- ❌ Do NOT add a `python-frontmatter` or YAML dependency. Hand-roll the small, fixed-schema frontmatter.
- ❌ Do NOT modify `rag.py` for indexing — watcher handles it.
- ❌ Do NOT write synchronously inside the SSE generator. Will block the stream.
- ❌ Do NOT import server-side structures into the writer module. Writer takes plain data.
- ❌ Do NOT use `asyncio.create_task` — this generator is sync (`yield from`), not an async handler. Use `threading.Thread`.
- ❌ Do NOT emit a new SSE event from the writer. Silent write; errors logged server-side only.
- ❌ Do NOT hardcode the vault path in the writer. Accept a `vault_root: Path` argument.

---

## Phase 1a — `web/conversation_writer.py`

**Goal**: A self-contained module that, given conversation data, upserts a markdown note for that `session_id`. No server integration yet.

### Public API (to implement)

```python
# web/conversation_writer.py

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class TurnData:
    question: str
    answer: str
    sources: list[dict]  # [{"file": str, "score": float}, ...]
    confidence: float
    timestamp: datetime  # UTC

def write_turn(
    vault_root: Path,
    session_id: str,
    turn: TurnData,
    *,
    subfolder: str = "00-Inbox/conversations",
) -> Path:
    """Append `turn` to the conversation note for `session_id`.

    - Creates the note if it doesn't exist (first turn).
    - Parses + regenerates YAML frontmatter with updated counters.
    - Appends a "## Turn N — HH:MM" block to the body.
    - Atomic: write to temp file in same dir, then os.replace().
    - Thread-safe: acquires an fcntl.flock on the target file during update.

    Returns the absolute path to the written note.
    """
```

### Implementation constraints

- **Filename (first turn only)**: `YYYY-MM-DD-HHMM-<slug>.md`, where `<slug>` is `slugify(question, max_len=50)`. The note is located by a **sidecar index** `~/.local/share/obsidian-rag/conversations_index.json` mapping `session_id → relative_path`. This avoids scanning the folder on every turn.
- **Slugify**: lowercase ASCII, strip accents (`unicodedata.normalize("NFKD")`), replace non-`[a-z0-9]+` with `-`, trim, max 50 chars.
- **Frontmatter schema** (hand-rolled YAML, fixed keys in fixed order):
  ```yaml
  ---
  session_id: web:abc123def456
  created: 2026-04-19T04:12:00Z
  updated: 2026-04-19T04:18:33Z
  turns: 3
  confidence_avg: 0.482
  sources:
    - 02-Areas/Coaching.md
    - 03-Resources/Ikigai.md
  tags:
    - conversation
    - rag-chat
  ---
  ```
  Emit only these keys. No user input reaches YAML (session_id is regex-validated upstream; sources are internal paths).
- **Body append pattern**:
  ```markdown
  ## Turn 3 — 04:18

  > ¿qué era el Ikigai?

  El Ikigai es...

  **Sources**: [[02-Areas/Coaching]] · [[03-Resources/Ikigai]]
  ```
  Obsidian wikilinks (not markdown `[text](url)`) so the backlink graph picks them up.
- **Atomicity**: always read → modify in-memory → write full file to `path.with_suffix(".md.tmp")` → `os.replace()`. Hold `fcntl.flock(fd, LOCK_EX)` on the tmp or the target during the operation.
- **Sources dedup**: the frontmatter `sources` list is the union across all turns, sorted, unique.
- **confidence_avg**: simple running mean, recomputed from scratch each turn (parse existing frontmatter, compute `(old_avg * (turns - 1) + new_confidence) / turns`, or recompute by reading all turn blocks — cleaner to just store `confidence_sum` and `turns`; but the plan requires `confidence_avg` visible, so store that, derive from `turns * avg + new / (turns + 1)`).

### Dependencies

Standard library only: `pathlib`, `datetime`, `dataclasses`, `fcntl`, `os`, `re`, `unicodedata`, `json`.

### Verification (Phase 1a)

Unit tests in `tests/test_conversation_writer.py`:
1. First turn creates note with expected filename + frontmatter.
2. Second turn appends "## Turn 2" block, updates `turns`, `updated`, `sources` union.
3. Slugify handles accents (`¿Qué es el Ikigai?` → `que-es-el-ikigai`).
4. Index file maps session_id → path correctly; second turn finds path via index.
5. Concurrent writes (simulated via two threads with `threading.Barrier`) produce a consistent final file (no lost turns, no corruption). **This is the key correctness test.**
6. Malformed existing frontmatter raises `ValueError` (refuse to corrupt an existing note).

Run: `.venv/bin/python -m pytest tests/test_conversation_writer.py -q`

### Anti-pattern grep (must return 0 matches)

```bash
grep -n "import yaml" web/conversation_writer.py
grep -n "asyncio" web/conversation_writer.py
grep -n "python-frontmatter" web/conversation_writer.py
grep -rn "VAULT_PATH\|DB_PATH\|ollama" web/conversation_writer.py  # writer stays decoupled
```

---

## Phase 1b — server hook

**Goal**: Call `write_turn()` from `/api/chat` after the `done` event, off the SSE thread.

### Changes to `web/server.py`

1. **Import** (near line 24 with other local imports):
   ```python
   from web.conversation_writer import write_turn, TurnData
   ```

2. **Helper** (near the existing `_emit_enrich` at ~line 2660, before the `chat()` function):
   ```python
   def _persist_conversation_turn(
       vault_root: Path,
       session_id: str,
       question: str,
       answer: str,
       metas: list[dict],
       scores: list[float],
       confidence: float,
   ) -> None:
       """Fire-and-forget write of the current turn to the vault. Swallow
       errors; log to _LOG_QUEUE. Never raise — caller is a daemon thread."""
       try:
           turn = TurnData(
               question=question,
               answer=answer,
               sources=[
                   {"file": m.get("file", ""), "score": float(s)}
                   for m, s in zip(metas, scores)
               ],
               confidence=float(confidence),
               timestamp=datetime.now(timezone.utc),
           )
           path = write_turn(vault_root, session_id, turn)
           _LOG_QUEUE.put((
               LOG_PATH,
               json.dumps({
                   "kind": "conversation_turn_written",
                   "session_id": session_id,
                   "path": str(path.relative_to(vault_root)),
               }) + "\n",
           ))
       except Exception as exc:
           _LOG_QUEUE.put((
               LOG_PATH,
               json.dumps({
                   "kind": "conversation_turn_error",
                   "session_id": session_id,
                   "error": repr(exc),
               }) + "\n",
           ))
   ```

3. **Hook** (right after the `done` yield at line 3086, before the enrich block at 3088):
   ```python
   # Persist turn to vault (fire-and-forget; does not block SSE or subsequent enrich).
   threading.Thread(
       target=_persist_conversation_turn,
       args=(
           VAULT_PATH,
           sess["id"],
           req.question,
           full,
           result["metas"],
           result["scores"],
           result["confidence"],
       ),
       daemon=True,
       name=f"conv-writer-{turn_id[:8]}",
   ).start()
   ```

   Verify `VAULT_PATH` is already imported from `rag` at the top of `server.py`; if not, add it to the existing `from rag import (...)` block.

### Verification (Phase 1b)

1. Start server: `uv run --project /Users/fer/repositories/obsidian-rag uvicorn web.server:app --port 8765 --reload`.
2. Send a chat request:
   ```bash
   curl -N -X POST http://localhost:8765/api/chat \
     -H "Content-Type: application/json" \
     -d '{"question":"¿qué es el Ikigai?","session_id":"web:test-episodic-001"}'
   ```
3. Observe SSE stream — should complete normally. `done` event arrives within expected latency (no added blocking).
4. Check file: `ls -la "$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes/00-Inbox/conversations/"` — a new `.md` file should exist.
5. `cat` the file — verify frontmatter + "## Turn 1" block match schema.
6. Repeat curl (same session_id, different question) — file should gain "## Turn 2" block; `turns: 2` in frontmatter.
7. Tail log: `tail -f ~/.local/share/obsidian-rag/chat_timings.log` (or equivalent LOG_PATH) — look for `conversation_turn_written` entries.
8. Wait 5s after step 6; then `rag query "ikigai"` — the conversation note should appear in results (confirms watcher picked it up + embedded it). **This is the "self-citation works" smoke test.**

### Failure modes to verify

- Kill writer mid-write (send SIGSTOP to server): chat stream should still emit `done`, next request should still work.
- Corrupt an existing conv note manually: second turn should log `conversation_turn_error` but not crash the stream.

---

## Phase 1c — tests + CLAUDE.md + eval sanity

### New file: `tests/test_conversation_writer.py`

Follow `tests/test_ambient.py:23-46` fixture pattern:
- `tmp_vault` fixture: `tmp_path` vault, `monkeypatch.setattr(rag, "VAULT_PATH", vault)`.
- Override index path: patch module-level `INDEX_PATH` in `conversation_writer` to `tmp_path / "index.json"`.
- 6 tests per Phase 1a verification list above.

### Update `repositories/obsidian-rag/CLAUDE.md`

Add a new section near the "Carpetas relevantes" area:
```markdown
### `00-Inbox/conversations/` — episodic memory
Cada conversación del web `/chat` se persiste acá como nota markdown (una por
`session_id`, multi-turno). Escritura post-`done` en `web/server.py:~3090` vía
`web/conversation_writer.py`, fire-and-forget. Indexada por el watcher existente
(no requiere código extra). La consolidación semanal (`scripts/consolidate_
conversations.py`) promueve clusters ≥3 a PARA y archiva originales.

**No editar a mano** — son artefactos del sistema. Si querés curar, usá la
consolidación semanal o movelas vos a PARA.
```

### Eval sanity (no regression gate)

Run full suite:
```bash
cd /Users/fer/repositories/obsidian-rag
.venv/bin/python -m pytest tests/ -q
```
All existing tests must still pass. Then:
```bash
rag eval
```
Must not regress `hit@5` below `88.10 - 2pp = 86.10%`. (Conversation notes in the index add noise — confirm it's bounded.)

If regression > 2pp: add `"00-Inbox/conversations"` to the default `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` in the plist (keeps self-citation off but preserves the vault artifact).

---

## Phase 2 — weekly consolidation

**Goal**: `scripts/consolidate_conversations.py`, run weekly by launchd, finds clusters of ≥3 semantically similar conversations within a 14-day window and writes a single synthesized note to `01-Projects/` or `03-Resources/`, archiving the originals.

### Approach (embedding-based clustering, no MCP)

1. Load all notes under `00-Inbox/conversations/` modified in the last 14 days.
2. Embed `title + first-turn-question` for each, using `rag.embed_texts()` (bge-m3, already loaded).
3. Build cosine similarity graph; edges ≥ 0.75. Extract connected components of size ≥ 3.
4. For each component:
   - Decide target folder via a cheap heuristic: if any turn mentions a date/deadline/action verb (`"hacer"`, `"mandar"`, `"agendar"`, `"próximo"`, `"el lunes"`, etc.) → `01-Projects/`. Else → `03-Resources/`.
   - Synthesize via local LLM (`qwen2.5:3b` — cheap, already loaded per memory). Prompt: "Given these N related Q&A turns, write a single consolidated note (frontmatter + body) that captures the common theme, key decisions, and open questions. Preserve wikilinks."
   - Write to target folder with filename `YYYY-MM-DD-<cluster-slug>.md` + frontmatter tag `consolidated-conversation` + wikilinks to the originals.
5. Move originals to `04-Archive/conversations/YYYY-MM/`.
6. Journal: append to `~/.local/share/obsidian-rag/consolidation.log` — JSONL with `{run_at, clusters_found, notes_promoted, originals_archived}`.

### Script template

Use `scripts/bench_chat.py:1-239` as the structural template (argparse, `rag.resolve_vault_paths()`, `print()` + `_LOG_QUEUE.put()`).

### launchd plist

`~/Library/LaunchAgents/com.fer.obsidian-rag-consolidate.plist`:
- `ProgramArguments`: `["/Users/fer/.local/bin/rag", "consolidate"]` — OR add a new CLI subcommand in `rag.py` that calls the script. Prefer the subcommand for consistency with other services (`com.fer.obsidian-rag-*` all use `rag <subcommand>`).
- `StartCalendarInterval`: Mondays 06:00 local (`{Weekday: 1, Hour: 6, Minute: 0}`).
- Logs: `/Users/fer/.local/share/obsidian-rag/consolidate.log` + `.error.log`.
- `KeepAlive: false`, `RunAtLoad: false` (don't run on boot).

### CLI subcommand

Add to `rag.py` under the `cli` click group:
```python
@cli.command()
@click.option("--window-days", default=14)
@click.option("--min-cluster", default=3)
@click.option("--similarity", default=0.75)
@click.option("--dry-run", is_flag=True)
def consolidate(window_days, min_cluster, similarity, dry_run):
    """Promote recurring conversation clusters to PARA."""
    from scripts.consolidate_conversations import run
    run(window_days=window_days, min_cluster=min_cluster,
        similarity=similarity, dry_run=dry_run)
```

### Verification (Phase 2)

1. Seed `00-Inbox/conversations/` with 5 synthetic conv notes (3 about "ikigai", 2 about "n8n").
2. `rag consolidate --dry-run` — prints "would promote: ikigai (3 notes) → 03-Resources/". No FS changes.
3. `rag consolidate` (real) — writes the promoted note, archives the 3 ikigai originals. The 2 n8n notes stay in-place (below min-cluster).
4. `rag query "ikigai"` — the new consolidated note appears (indexed by watcher).
5. `launchctl load ~/Library/LaunchAgents/com.fer.obsidian-rag-consolidate.plist && launchctl start com.fer.obsidian-rag-consolidate` — triggers the job manually; log file should contain one run entry.

---

## Phase 3 — WhatsApp integration (DOCUMENTED, NOT EXECUTED)

When ready:
- `whatsapp-listener/listener.ts` has `/ask` and `/rag` commands with session keys `wa:<jid>`.
- Add a call after RAG answer is computed: POST to local HTTP endpoint `http://localhost:8765/api/chat/persist` (new) with `{session_id, question, answer, sources, confidence}`.
- Server-side: that endpoint calls `write_turn()` directly. Reuses Phase 1a module wholesale.
- No changes needed in `conversation_writer.py`.

This gives full-stack episodic memory (web + WA) without duplicating the writer.

---

## Execution order

| Phase | Owner | Blocking | Est. LoC |
|------|-------|----------|----------|
| 1a | `pm` → writer specialist | — | ~180 (module) + ~200 (tests) |
| 1b | `pm` → server specialist | 1a | ~40 in `web/server.py` |
| 1c | `pm` → generalist | 1b | ~20 docs + 0 code |
| 2  | `pm` → generalist | 1c verified in prod ≥1 week | ~250 (script) + plist |
| 3  | — | 2 stabilized | (deferred) |

Phase 1 can ship end-to-end in one dispatch. Phase 2 should wait until Fer has actually used the feature for a few days — the consolidation heuristics should be tuned against real conversation patterns, not synthetic ones.
