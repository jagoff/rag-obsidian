"""
Contract tests for the parallelized `warmup_async` (2026-04-22).

Pre-fix the 5 warmup targets (reranker, ollama-embed, local-embed, corpus,
chat-models) ran serially inside a single background thread. Total ~20s.
The main thread of `rag query` reached `retrieve()` at ~1-2s → the
`_local_embedder_ready` Event fired only at ~15s → `wait(1.5s)` timed out
→ fallback to ollama `embed()` which was ALSO cold (the ollama warmup
queued behind the reranker). Measured `embed_ms` 10-12s in outliers.

Post-fix the 5 targets spawn as parallel daemon threads. Max individual
~5s, total wall ~5s. The Event fires around 5s — matches the new wait
default of 4000ms (`RAG_LOCAL_EMBED_WAIT_MS`).

These tests are source-level: we can't actually trigger a cold-load in CI
without keeping ~500MB HF cache around.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG_PY = (ROOT / "rag.py").read_text(encoding="utf-8")


def test_warmup_has_parallel_target_helpers():
    """Each of the 5 warmup targets must be factored into its own helper,
    proving the parallelization refactor happened."""
    for helper in [
        "def _wu_reranker",
        "def _wu_ollama_embed",
        "def _wu_local_embed",
        "def _wu_corpus",
        "def _wu_chat_models",
    ]:
        assert helper in RAG_PY, f"missing warmup helper: {helper}"


def test_warmup_run_spawns_threads_not_serial_calls():
    """`_run()` inside `warmup_async` must spawn a list of Threads, not
    call the helpers directly (which would serialize them)."""
    idx = RAG_PY.find("def warmup_async() -> None")
    assert idx >= 0
    block = RAG_PY[idx : idx + 6000]
    # The parallel implementation builds a `threads = [...]` list
    assert "threads = [" in block, (
        "expected `threads = [...]` list-comprehension of warmup targets"
    )
    # And starts each
    assert "for t in threads" in block and "t.start()" in block, (
        "expected `for t in threads: t.start()` to launch them"
    )
    # Do NOT serialize by calling the helpers directly in _run.
    # (They're referenced via the `targets` tuple, not as bare calls.)


def test_warmup_targets_list_has_all_five():
    """The `targets` list in `_run` should name all 5 warmup targets."""
    idx = RAG_PY.find("def warmup_async() -> None")
    block = RAG_PY[idx : idx + 6000]
    targets_idx = block.find("targets = [")
    assert targets_idx >= 0
    targets_block = block[targets_idx : targets_idx + 600]
    for name in ["reranker", "ollama-embed", "local-embed", "corpus", "chat-models"]:
        assert f'"{name}"' in targets_block, f"missing target name: {name}"


def test_warmup_threads_are_daemon():
    """Warmup threads MUST be daemon — otherwise a hung cold-load would
    prevent the main process from exiting on normal shutdown."""
    idx = RAG_PY.find("def warmup_async() -> None")
    block = RAG_PY[idx : idx + 6000]
    # The list-comp that builds threads must set daemon=True
    assert "daemon=True" in block, (
        "warmup threads must be daemon — missing `daemon=True` in Thread(...)"
    )


def test_local_embed_wait_default_bumped_to_4s():
    """The default `RAG_LOCAL_EMBED_WAIT_MS` in retrieve() must be 4000ms
    (4.0s) to match the parallel warmup completion time. The legacy 1500ms
    timed out before the Event fired."""
    # Find the retrieve() wait ternary — search from the `_wait_s = (`
    # assignment which is unique to this block.
    idx = RAG_PY.find("_wait_ms_str = os.environ.get(\"RAG_LOCAL_EMBED_WAIT_MS\")")
    assert idx >= 0, "retrieve() wait block not found"
    block = RAG_PY[idx : idx + 500]
    # Expect `else 4.0)` as the default branch of the ternary
    assert "else 4.0)" in block, (
        f"expected `else 4.0)` as the new default in the wait ternary;"
        f" block was: {block!r}"
    )
    # The except-ValueError fallback should also be 4.0, not 1.5
    assert "_wait_s = 4.0" in block, (
        f"except-ValueError fallback needs to use 4.0s; block: {block!r}"
    )


def test_warmup_async_still_idempotent():
    """The `_warmup_started` global + `_warmup_lock` must still guard
    against double-start. Paralelization shouldn't have touched that."""
    idx = RAG_PY.find("def warmup_async() -> None")
    block = RAG_PY[idx : idx + 2000]
    assert "global _warmup_started" in block
    assert "_warmup_lock" in block
    assert "if _warmup_started:" in block and "return" in block


def test_warmup_rag_no_warmup_opt_out_preserved():
    """`RAG_NO_WARMUP=1` must still be honored — lightweight commands
    (`rag stats`, `rag session list`) shouldn't pay the warmup cost."""
    idx = RAG_PY.find("def warmup_async() -> None")
    block = RAG_PY[idx : idx + 2000]
    assert 'os.environ.get("RAG_NO_WARMUP") == "1"' in block, (
        "RAG_NO_WARMUP=1 escape hatch must be preserved"
    )


def test_warmup_coordinator_does_not_join():
    """The coordinator `_run` must NOT call `.join()` on the child
    threads — the whole point is fire-and-forget. Joining would
    re-serialize the warmup from the main thread's perspective."""
    idx = RAG_PY.find("def _run() -> None")
    assert idx >= 0
    # Search near the _run() inside warmup_async
    block = RAG_PY[idx : idx + 1000]
    assert ".join(" not in block, (
        "warmup `_run()` must not join child threads — fire-and-forget"
    )
