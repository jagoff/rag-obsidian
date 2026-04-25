"""Tests para `rag_health_report()` / `rag stats` health section + startup
warning cuando una feature esta default-ON pero su dependencia falta.

Contexto 2026-04-22:
  - `RAG_EXTRACT_ENTITIES` default ON tras 2026-04-21 (commit 3af64ea).
  - `gliner` es soft-dep (pyproject `[entities]` extra). Cuando falta:
    - Cada proceso nuevo (watch, serve, 4 ingesters, web) re-intenta
      importar -> 28+ `gliner_import_failed` en silent_errors.jsonl/dia.
    - `rag_entities` table queda stale (solo se popula via backfill manual).
    - Usuario no tiene forma de saberlo sin `cat silent_errors.jsonl`.

Cambios 2026-04-22:
  1. Nuevo helper `rag_health_report()` en rag.py -> dict con:
     - `feature_deps`: list de features ON con dep faltante
     - `silent_errors_24h`: dict de {where: count} de las ultimas 24h
     - `sql_state_errors_24h`: idem de sql_state_errors.jsonl
  2. `rag stats` renderea la seccion "Health" usando el helper.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── rag_health_report: feature deps ──────────────────────────────────────────


def test_rag_health_report_exists():
    assert hasattr(rag, "rag_health_report"), \
        "post 2026-04-22 rag.py must expose rag_health_report() to power " \
        "`rag stats` health section + startup warnings"


def test_rag_health_report_returns_dict_with_expected_keys():
    report = rag.rag_health_report()
    assert isinstance(report, dict)
    assert "feature_deps" in report
    assert "silent_errors_24h" in report
    assert "sql_state_errors_24h" in report


def test_rag_health_report_flags_entities_when_gliner_missing(monkeypatch):
    """When RAG_EXTRACT_ENTITIES is ON (default) AND gliner import fails,
    feature_deps must surface the mismatch."""
    monkeypatch.delenv("RAG_EXTRACT_ENTITIES", raising=False)  # default ON
    # Simulate failed load: sticky flag set.
    monkeypatch.setattr(rag, "_gliner_load_failed", True)
    report = rag.rag_health_report()
    fd = report["feature_deps"]
    assert any("entity" in str(item).lower() and "gliner" in str(item).lower()
               for item in fd), f"feature_deps={fd}"


def test_rag_health_report_silent_on_entities_when_disabled(monkeypatch):
    """If the user explicitly disabled the feature, no warning even if dep missing."""
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    monkeypatch.setattr(rag, "_gliner_load_failed", True)
    report = rag.rag_health_report()
    fd = report["feature_deps"]
    assert not any("entity" in str(item).lower() for item in fd), \
        "user opted out via RAG_EXTRACT_ENTITIES=0 — don't warn"


def test_rag_health_report_silent_on_entities_when_gliner_healthy(monkeypatch):
    """If gliner loaded fine, no warning."""
    monkeypatch.delenv("RAG_EXTRACT_ENTITIES", raising=False)  # default ON
    monkeypatch.setattr(rag, "_gliner_load_failed", False)
    report = rag.rag_health_report()
    fd = report["feature_deps"]
    assert not any("entity" in str(item).lower() and "gliner" in str(item).lower()
                   for item in fd)


# ── rag_health_report: silent_errors.jsonl rollup ────────────────────────────


def test_rag_health_report_counts_silent_errors_24h(tmp_path, monkeypatch):
    log_path = tmp_path / "silent_errors.jsonl"
    now = datetime.now()

    entries = [
        {"ts": now.isoformat(timespec="seconds"),
         "where": "gliner_import_failed", "exc_type": "Exception", "exc": "x"},
        {"ts": now.isoformat(timespec="seconds"),
         "where": "gliner_import_failed", "exc_type": "Exception", "exc": "x"},
        {"ts": now.isoformat(timespec="seconds"),
         "where": "ranker_config_load", "exc_type": "Exception", "exc": "x"},
        # Old one (>24h): must be excluded
        {"ts": (now - timedelta(days=2)).isoformat(timespec="seconds"),
         "where": "should_be_skipped", "exc_type": "Exception", "exc": "x"},
    ]
    with log_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", log_path)
    report = rag.rag_health_report()
    se24 = report["silent_errors_24h"]
    assert se24.get("gliner_import_failed") == 2
    assert se24.get("ranker_config_load") == 1
    assert "should_be_skipped" not in se24


def test_rag_health_report_silent_errors_empty_when_log_missing(monkeypatch, tmp_path):
    missing = tmp_path / "nope.jsonl"
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", missing)
    report = rag.rag_health_report()
    assert report["silent_errors_24h"] == {}


def test_rag_health_report_silent_errors_tolerates_corrupt_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "silent_errors.jsonl"
    now = datetime.now()
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"),
            "where": "ok", "exc_type": "x", "exc": "x",
        }) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"),
            "where": "also_ok", "exc_type": "x", "exc": "x",
        }) + "\n")

    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", log_path)
    report = rag.rag_health_report()
    se24 = report["silent_errors_24h"]
    # Corrupt line is skipped; the two valid ones are counted.
    assert se24.get("ok") == 1
    assert se24.get("also_ok") == 1


# ── rag_health_report: sql_state_errors.jsonl rollup ─────────────────────────


def test_rag_health_report_counts_sql_state_errors_24h(tmp_path, monkeypatch):
    log_path = tmp_path / "sql_state_errors.jsonl"
    now = datetime.now()
    entries = [
        {"ts": now.isoformat(timespec="seconds"),
         "event": "semantic_cache_store_failed",
         "err": "OperationalError('database is locked')"},
        {"ts": now.isoformat(timespec="seconds"),
         "event": "semantic_cache_store_failed",
         "err": "OperationalError('database is locked')"},
        {"ts": now.isoformat(timespec="seconds"),
         "event": "queries_sql_write_failed",
         "err": "OperationalError('disk I/O error')"},
        {"ts": (now - timedelta(days=3)).isoformat(timespec="seconds"),
         "event": "old_event", "err": "x"},
    ]
    with log_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", log_path)
    report = rag.rag_health_report()
    sse = report["sql_state_errors_24h"]
    assert sse.get("semantic_cache_store_failed") == 2
    assert sse.get("queries_sql_write_failed") == 1
    assert "old_event" not in sse


# ── rag stats: integration (just check the CLI runs clean) ───────────────────


def test_rag_stats_runs_without_error_even_when_gliner_missing(monkeypatch):
    """Smoke test: the `stats` command must not raise when gliner is absent."""
    monkeypatch.delenv("RAG_EXTRACT_ENTITIES", raising=False)  # default ON
    monkeypatch.setattr(rag, "_gliner_load_failed", True)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["stats"])
    # Command must exit 0; the output must mention the Health section.
    assert result.exit_code == 0, result.output
    # No hard-coded text — just assert the health signal surfaces when missing.
    assert "Health" in result.output or "health" in result.output or \
        "gliner" in result.output.lower()


# ── _warn_feature_dep_once: first-call-per-process stderr warn ───────────────


def test_warn_feature_dep_once_exists():
    assert hasattr(rag, "_warn_feature_dep_once"), \
        "post 2026-04-22 rag.py must expose _warn_feature_dep_once() — called " \
        "from _get_gliner_model and friends the first time a default-ON " \
        "feature's soft-dep fails, so the operator sees ONE line on stderr " \
        "instead of having to grep silent_errors.jsonl"


def test_warn_feature_dep_once_emits_to_stderr_on_first_call(capsys, monkeypatch):
    # Reset the warned-set to simulate a fresh process.
    monkeypatch.setattr(rag, "_WARNED_FEATURE_DEPS", set())
    rag._warn_feature_dep_once("entities", "gliner")
    captured = capsys.readouterr()
    assert "entities" in captured.err
    assert "gliner" in captured.err


def test_warn_feature_dep_once_is_idempotent(capsys, monkeypatch):
    """Second call for the same feature must NOT emit — prevents log spam."""
    monkeypatch.setattr(rag, "_WARNED_FEATURE_DEPS", set())
    rag._warn_feature_dep_once("entities", "gliner")
    capsys.readouterr()  # drain
    rag._warn_feature_dep_once("entities", "gliner")
    captured = capsys.readouterr()
    assert captured.err == "", \
        f"second call for same feature produced stderr: {captured.err!r}"


def test_warn_feature_dep_once_different_features_each_warn_once(capsys, monkeypatch):
    monkeypatch.setattr(rag, "_WARNED_FEATURE_DEPS", set())
    rag._warn_feature_dep_once("entities", "gliner")
    rag._warn_feature_dep_once("nli", "transformers_nli")
    captured = capsys.readouterr()
    assert "entities" in captured.err
    assert "nli" in captured.err
