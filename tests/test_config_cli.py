"""Feature #7 del 2026-04-23 — `rag config` command tests.

Validates:
- _CONFIG_VARS curated list is well-shaped
- _collect_env_var_names_from_source picks up env vars from the source
- CLI output: table rendering, --as-json, --only-set, --filter
- Current values picked up from os.environ correctly
"""
from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

import rag


# ── _CONFIG_VARS shape ────────────────────────────────────────────────────


def test_curated_vars_tuple_shape():
    """Each curated var is a 4-tuple (name, default, type, description)."""
    assert isinstance(rag._CONFIG_VARS, tuple)
    for entry in rag._CONFIG_VARS:
        assert isinstance(entry, tuple)
        assert len(entry) == 4
        name, default, type_, desc = entry
        assert isinstance(name, str) and name
        assert isinstance(default, str)
        assert isinstance(type_, str)
        assert isinstance(desc, str) and desc


def test_curated_names_are_rag_prefixed():
    """All curated names start with RAG_ or OBSIDIAN_RAG_ or OLLAMA_."""
    for name, *_ in rag._CONFIG_VARS:
        assert (
            name.startswith("RAG_")
            or name.startswith("OBSIDIAN_RAG_")
            or name.startswith("OLLAMA_")
        )


def test_curated_names_unique():
    names = [name for name, *_ in rag._CONFIG_VARS]
    assert len(names) == len(set(names))


def test_curated_has_core_vars():
    """Sanity: the most commonly-used vars exist in the curated list."""
    names = {name for name, *_ in rag._CONFIG_VARS}
    assert "OBSIDIAN_RAG_VAULT" in names
    assert "RAG_SCORE_CALIBRATION" in names
    assert "RAG_MMR_DIVERSITY" in names
    assert "RAG_PPR_TOPIC" in names
    assert "RAG_LLM_INTENT" in names
    assert "OLLAMA_KEEP_ALIVE" in names


# ── _collect_env_var_names_from_source ───────────────────────────────────


def test_source_scan_returns_set():
    names = rag._collect_env_var_names_from_source()
    assert isinstance(names, set)


def test_source_scan_picks_up_known_vars():
    """The scan should find at least the curated vars that exist in
    rag.py (they all go through os.environ.get)."""
    from_scan = rag._collect_env_var_names_from_source()
    # Sanity: find some well-known ones.
    assert "RAG_SCORE_CALIBRATION" in from_scan
    assert "RAG_MMR_DIVERSITY" in from_scan
    assert "RAG_PPR_TOPIC" in from_scan


def test_source_scan_prefixed_only():
    """No stray non-RAG/non-OLLAMA vars leak through."""
    names = rag._collect_env_var_names_from_source()
    for n in names:
        assert (
            n.startswith("RAG_")
            or n.startswith("OBSIDIAN_RAG_")
            or n.startswith("OLLAMA_")
        )


# ── CLI: table output ────────────────────────────────────────────────────


def test_cli_config_default_renders_table(monkeypatch):
    """Default output shows the curated table + uncurated section."""
    # Isolate from user's environment: clear all tracked vars.
    for name, *_ in rag._CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config"])
    assert result.exit_code == 0, result.output
    assert "Env vars curadas" in result.output
    assert "OBSIDIAN_RAG_VAULT" in result.output
    assert "Total:" in result.output


def test_cli_config_only_set_hides_unset(monkeypatch):
    """--only-set shows only vars present in env."""
    for name, *_ in rag._CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_LLM_INTENT", "1")

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--only-set"])
    assert result.exit_code == 0, result.output
    assert "RAG_LLM_INTENT" in result.output
    # A var we KNOW is unset should NOT appear.
    assert "RAG_SCORE_CALIBRATION" not in result.output


def test_cli_config_filter_substring(monkeypatch):
    """--filter restricts by substring (case-insensitive)."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--filter", "mmr"])
    assert result.exit_code == 0, result.output
    assert "RAG_MMR_DIVERSITY" in result.output
    # Shouldn't leak unrelated vars.
    assert "OBSIDIAN_RAG_VAULT" not in result.output


def test_cli_config_filter_case_insensitive(monkeypatch):
    """Filter matching is case-insensitive."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--filter", "PPR"])
    assert result.exit_code == 0
    assert "RAG_PPR_TOPIC" in result.output

    result2 = runner.invoke(rag.cli, ["config", "--filter", "ppr"])
    assert result2.exit_code == 0
    assert "RAG_PPR_TOPIC" in result2.output


# ── CLI: JSON output ─────────────────────────────────────────────────────


def test_cli_config_as_json_valid(monkeypatch):
    """--as-json emits parseable JSON."""
    for name, *_ in rag._CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--as-json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    # Each item has the expected shape
    for item in data:
        assert {"name", "default", "type", "description",
                "current", "is_set", "curated"} <= set(item.keys())


def test_cli_config_as_json_reflects_current_env(monkeypatch):
    """When env var is set, 'current' field reflects the value."""
    monkeypatch.setenv("RAG_MMR_DIVERSITY", "1")

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--as-json",
                                      "--filter", "mmr_diversity"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    mmr = next(d for d in data if d["name"] == "RAG_MMR_DIVERSITY")
    assert mmr["current"] == "1"
    assert mmr["is_set"] is True


def test_cli_config_unset_var_has_empty_current(monkeypatch):
    monkeypatch.delenv("RAG_SCORE_CALIBRATION", raising=False)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["config", "--as-json",
                                      "--filter", "score_calibration"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    sc = next(d for d in data if d["name"] == "RAG_SCORE_CALIBRATION")
    assert sc["current"] == ""
    assert sc["is_set"] is False
