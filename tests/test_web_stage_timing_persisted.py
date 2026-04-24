"""Tests that `/api/chat` persists stage-level timing to rag_queries
(2026-04-22).

Pre-fix, TTFT / llm_prefill_ms / llm_decode_ms / total_ms were emitted
to the SSE `done` event for the browser but never persisted to SQL.
Result: the dashboard couldn't distinguish whether the latency problem
was retrieve, LLM prefill, or LLM decode — only the coarse t_retrieve
+ t_gen columns.

Real telemetry (`rag_queries`, 30d) showed:
  web p50 = 9s, p90 = 25s, p99 = 58s
  LLM dominates (~80% of total), but no way to split prefill vs decode

Post-fix, `log_query_event` includes ttft_ms, llm_prefill_ms,
llm_decode_ms, total_ms in extra_json so the analytics stack can
query them with json_extract.

These tests use a partial mock of the internals because `/api/chat`
invokes the full pipeline (retrieve + LLM + post-process). We assert
at the `log_query_event` call site via monkeypatch, which is where
the contract lives.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")


# ── Contract: `log_query_event` in /api/chat must include stage-timing keys ─


def test_extra_json_contract_normal_path():
    """The keys `ttft_ms`, `llm_prefill_ms`, `llm_decode_ms`, `total_ms`
    are present in the normal web-chat log_query_event payload.

    Test reads rag.py's source to verify the contract — a brittle way
    to check but beats spinning up the full pipeline. If someone
    removes the keys, grep fails.
    """
    src = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    # The main `cmd: "web"` log call must carry all four stage keys.
    assert "\"ttft_ms\"" in src, "ttft_ms missing from web log_query_event"
    assert "\"llm_prefill_ms\"" in src, "llm_prefill_ms missing"
    assert "\"llm_decode_ms\"" in src, "llm_decode_ms missing"
    # total_ms is the crosscheck column.
    assert "\"total_ms\": int(_t_total_ms)" in src, (
        "total_ms missing or not wired to _t_total_ms"
    )


def test_extra_json_contract_timing_breakdown_persisted():
    """Post-2026-04-23 el web endpoint también persiste el dict `timing`
    devuelto por retrieve() (embed_ms / sem_ms / bm25_ms / rrf_ms /
    reranker_ms / total_ms). Sin esto no se puede diagnosticar por qué
    `retrieve_ms` es alto en casos warm (4-6s) — si es embed (warmup
    race) o reranker (cold MPS).

    Pre-fix sólo el CLI `query` + `chat` persistían este dict; el web
    emitía el t_retrieve agregado pero no el breakdown."""
    src = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    # Buscamos el bloque log_query_event con cmd="web" (NO bypass).
    # El bloque es largo (~70 líneas) por todos los docstrings inline;
    # scaneamos 6000 bytes desde el marker para cubrirlo completo.
    web_idx = src.find('"cmd": "web",')
    assert web_idx >= 0, "main web log_query_event block not found"
    nearby = src[web_idx : web_idx + 6000]
    # The timing key should land in extra_json via _round_timing_ms
    assert "\"timing\"" in nearby, (
        "timing key missing from web log_query_event payload"
    )
    assert "_round_timing_ms" in nearby, (
        "_round_timing_ms not wired — timing dict must go through "
        "the shared rounder to stay shape-stable with CLI rows"
    )
    # And the import should be in place at the top of the file (in the
    # `from rag import (...)` block — scan the first 5000 bytes to cover
    # the multi-line import list comfortably).
    assert "_round_timing_ms" in src[:5000], (
        "_round_timing_ms import missing from web/server.py top imports"
    )


def test_extra_json_contract_low_conf_bypass():
    """The low_conf_bypass path also persists stage timing so analytics
    can filter `cmd='web.chat.low_conf_bypass'` rows with the same SQL
    shape as normal rows."""
    src = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    # Look for the block that logs low_conf_bypass
    bypass_idx = src.find('"cmd": "web.chat.low_conf_bypass"')
    assert bypass_idx >= 0
    # The stage-timing keys must appear within ~1000 chars of that marker
    nearby = src[bypass_idx : bypass_idx + 1500]
    assert "\"ttft_ms\"" in nearby, "ttft_ms missing from low_conf_bypass path"
    assert "\"llm_prefill_ms\"" in nearby
    assert "\"llm_decode_ms\"" in nearby
    assert "\"total_ms\"" in nearby


# ── Analytics query: the stage keys are readable via json_extract ──────────


def test_stage_keys_queryable_from_rag_queries(tmp_path, monkeypatch):
    """Belt + suspenders: insert a synthetic row with the expected shape,
    read it back via the exact SQL pattern the dashboard would use, and
    assert all four fields come through."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    rag.log_query_event({
        "cmd": "web",
        "q": "dummy",
        "t_retrieve": 2.3,
        "t_gen": 8.1,
        "ttft_ms": 512,
        "llm_prefill_ms": 1200,
        "llm_decode_ms": 6300,
        "total_ms": 10400,
    })

    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT "
            "  json_extract(extra_json,'$.ttft_ms'),"
            "  json_extract(extra_json,'$.llm_prefill_ms'),"
            "  json_extract(extra_json,'$.llm_decode_ms'),"
            "  json_extract(extra_json,'$.total_ms') "
            "FROM rag_queries ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert row == (512, 1200, 6300, 10400)


def test_analytics_p50_over_ttft_ms(tmp_path, monkeypatch):
    """Realistic analytics query: p50 of ttft_ms across the recent
    window. This is what the dashboard would add to surface the
    percepción metric next to t_retrieve / t_gen."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)

    # Seed 10 rows with varying TTFT.
    for i, ttft in enumerate([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900]):
        rag.log_query_event({
            "cmd": "web", "q": f"q{i}",
            "t_retrieve": 1.0, "t_gen": 3.0,
            "ttft_ms": ttft,
            "llm_prefill_ms": 500, "llm_decode_ms": 2500, "total_ms": ttft + 3000,
        })

    with rag._ragvec_state_conn() as conn:
        p50 = conn.execute(
            "WITH o AS ("
            " SELECT CAST(json_extract(extra_json,'$.ttft_ms') AS INTEGER) AS v,"
            "        ROW_NUMBER() OVER (ORDER BY CAST(json_extract(extra_json,'$.ttft_ms') AS INTEGER)) AS rn,"
            "        COUNT(*) OVER () AS n"
            " FROM rag_queries WHERE cmd='web'"
            ") SELECT MIN(CASE WHEN rn >= n*0.5 THEN v END) FROM o"
        ).fetchone()[0]

    # With 10 sorted values [100,300,500,700,900,1100,...,1900] and the
    # SQL `MIN(v) WHERE rn >= n*0.5`, p50 picks the first row where
    # rn >= 5, i.e. v=900. (Nearest-rank percentile, lower-median.)
    assert p50 == 900
