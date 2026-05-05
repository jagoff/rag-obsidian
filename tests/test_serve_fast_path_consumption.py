"""Tests for fast_path consumption in the WA `serve` endpoint (2026-04-22).

Previously, `retrieve()` set `result["fast_path"]=True` when the adaptive
routing gate fired (baseline: semantic intent + top1>0.6; WA-specific:
top1>0.3 + majority WhatsApp in top3) but the serve endpoint IGNORED the
marker — always used `resolve_chat_model()` (command-r / qwen2.5:7b) +
CHAT_OPTIONS with default num_ctx.

Measured in rag_queries (CLI, last 7d):
  fast_path=1 n=9  avg t_gen=3.1s
  fast_path=0 n=15 avg t_gen=8.5s
  → 2.75× speedup on fast-path hits

Post-fix, serve switches to _LOOKUP_MODEL + _LOOKUP_NUM_CTX when
result["fast_path"] is True. log_query_event now includes fast_path
and gen_model for analytics.

These tests verify the shape of the switch via source grepping and
direct helper-fn validation. Full-pipeline integration is expensive
(requires ollama + a real vault) and covered by the same e2e suite
that validates `rag query` + `rag chat`.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402


# ── Source-level contract ───────────────────────────────────────────────────


def test_serve_consumes_fast_path_from_retrieve_result():
    """The main LLM call inside serve's _handle_query must read
    `result["fast_path"]` and route to _LOOKUP_MODEL accordingly."""
    src = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")
    # Anchor: the `_serve_fast_path` assignment I added should exist.
    # Without it, the whole refactor never happened.
    assert "_serve_fast_path = bool(result.get(" in src, (
        "serve does not read fast_path from retrieve result — "
        "expected `_serve_fast_path = bool(result.get(...))` assignment"
    )
    # And it must be used to pick the gen model
    assert "_serve_gen_model = (" in src, (
        "serve does not conditionally pick gen model by fast_path — "
        "expected `_serve_gen_model = (...)` assignment"
    )
    # Must reference _LOOKUP_MODEL for the fast branch
    idx = src.find("_serve_gen_model = (")
    assert idx >= 0
    nearby = src[idx : idx + 500]
    assert "_LOOKUP_MODEL" in nearby, (
        "_serve_gen_model doesn't switch to _LOOKUP_MODEL on fast_path"
    )
    # Must reference _LOOKUP_NUM_CTX to reduce context for the small model
    idx_opts = src.find("_serve_gen_options = (")
    assert idx_opts >= 0
    nearby_opts = src[idx_opts : idx_opts + 500]
    assert "_LOOKUP_NUM_CTX" in nearby_opts, (
        "_serve_gen_options doesn't reduce num_ctx on fast_path"
    )


def test_serve_logs_fast_path_and_gen_model():
    """log_query_event in the serve main path must surface `fast_path`
    and `gen_model` so the dashboard can measure realised speedup."""
    src = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")
    # Find the serve cmd log event
    idx = src.find('"cmd": "serve", "q": question')
    assert idx >= 0, "serve log_query_event anchor missing"
    nearby = src[idx : idx + 800]
    assert '"fast_path"' in nearby, (
        "serve log_query_event missing fast_path field"
    )
    assert '"gen_model"' in nearby, (
        "serve log_query_event missing gen_model field"
    )


def test_serve_logs_telemetry_keys_parity():
    """log_query_event en serve debe emitir las mismas keys de telemetría
    que to_log_event: mmr_applied, contradiction_penalty_applied,
    anaphora_*, temporal_intent, llm_judge_* y decompose_*.

    Verificación source-level: si el dict pierde una key en un refactor
    futuro, este test falla antes de que llegue a producción.
    """
    src = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")
    idx = src.find('"cmd": "serve", "q": question')
    assert idx >= 0, "serve log_query_event anchor missing"
    # El bloque del dict ocupa ~3150 chars con las nuevas keys.
    block = src[idx : idx + 3300]

    expected_keys = [
        '"mmr_applied"',
        '"contradiction_penalty_applied"',
        '"anaphora_resolved"',
        '"anaphora_original"',
        '"anaphora_rewritten"',
        '"temporal_intent"',
        '"llm_judge_fired"',
        '"llm_judge_ms"',
        '"llm_judge_top_score_before"',
        '"llm_judge_top_score_after"',
        '"llm_judge_parse_failed"',
        '"llm_judge_n_candidates"',
        '"decomposed"',
        '"n_sub_queries"',
        '"decompose_ms"',
    ]
    missing = [k for k in expected_keys if k not in block]
    assert not missing, (
        f"serve log_query_event falta(n) {len(missing)} keys de telemetría: "
        f"{missing}"
    )


# ── Shared infra: _LOOKUP_MODEL + _LOOKUP_NUM_CTX exist with sane values ────


def test_lookup_model_is_smaller_than_chat_default():
    """Sanity: _LOOKUP_MODEL should be a smaller model than the default.
    Otherwise the fast-path would be slower, not faster."""
    # Not asserting the exact string (env override allowed) — just that
    # the module exposes the value.
    assert hasattr(rag, "_LOOKUP_MODEL")
    lookup = rag._LOOKUP_MODEL
    assert isinstance(lookup, str) and lookup
    # Typical values: "qwen2.5:3b", "qwen2.5:1.5b", maybe "llama3.2:3b".
    # We just check it's a smaller-category model name — arbitrary but
    # catches someone setting it to command-r accidentally.
    assert not lookup.startswith("command-r"), (
        f"_LOOKUP_MODEL is {lookup!r} — fast-path should use a small model"
    )


def test_lookup_num_ctx_is_smaller_than_default():
    """Sanity: _LOOKUP_NUM_CTX should be ≤ CHAT_OPTIONS['num_ctx']. The
    point of the fast path is to REDUCE prefill cost on simple lookups;
    a bigger ctx would defeat the purpose."""
    lookup_ctx = getattr(rag, "_LOOKUP_NUM_CTX", None)
    assert lookup_ctx is not None, "_LOOKUP_NUM_CTX missing from rag module"
    default_ctx = rag.CHAT_OPTIONS.get("num_ctx")
    assert default_ctx is not None, "CHAT_OPTIONS['num_ctx'] missing"
    assert lookup_ctx <= default_ctx, (
        f"_LOOKUP_NUM_CTX={lookup_ctx} should be ≤ default {default_ctx}"
    )


# ── Keep-alive respects the model switch ────────────────────────────────────


def test_keep_alive_respects_fast_path_model():
    """`chat_keep_alive(_serve_gen_model)` must be called with the actual
    model so a small model (_LOOKUP_MODEL) doesn't get pinned with a
    large-model keep-alive policy."""
    src = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")
    idx = src.find("_serve_gen_model = (")
    assert idx >= 0, "_serve_gen_model definition missing"
    # Find the ollama.chat call inside the same handler block
    handler_start = src.rfind("def _handle_query(body: dict)", 0, idx)
    assert handler_start >= 0
    # Everything between _serve_gen_model def and the end of the block
    handler_end = src.find("return payload", idx)
    block = src[idx : handler_end]
    # The ollama.chat should use _serve_gen_model for keep_alive too
    assert "chat_keep_alive(_serve_gen_model)" in block, (
        "ollama.chat in serve passes resolve_chat_model()-based keep_alive "
        "instead of the fast-path model — large-model keep-alive on small "
        "model wastes VRAM"
    )


# ── Scoreboard: measuring the diff pre/post ─────────────────────────────────


def test_analytics_query_for_fast_path_speedup():
    """Documents the SQL query to measure the realised speedup post-fix.
    Not a behavioural test — just ensures the query shape is queryable
    given our schema (doesn't depend on data being present)."""
    # Schema contract check: rag_queries.extra_json can be queried with
    # json_extract on 'fast_path' key. This proves the analytics query
    # in the commit message body works — the real speedup measurement
    # happens in production after 24h of traffic.
    query = (
        "SELECT AVG(CASE WHEN json_extract(extra_json,'$.fast_path')=1 "
        "             THEN t_gen END) as fast_avg, "
        "       AVG(CASE WHEN json_extract(extra_json,'$.fast_path')=0 "
        "             THEN t_gen END) as slow_avg "
        "FROM rag_queries WHERE cmd='serve'"
    )
    # Just verify the SQL is syntactically valid by opening a temp DB
    # and running it (should return None/None on empty DB).
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE rag_queries ("
        "  id INTEGER PRIMARY KEY,"
        "  cmd TEXT, t_gen REAL, extra_json TEXT"
        ")"
    )
    cur = conn.execute(query)
    row = cur.fetchone()
    assert row == (None, None)  # Empty table → both aggregates are null
    conn.close()
