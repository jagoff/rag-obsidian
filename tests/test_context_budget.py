"""Feature #11 del 2026-04-23 — Context budget tracking tests.

Validates:
- _estimate_tokens: char count × ratio + 1
- _context_budget_status: ok / warn / over levels
- CLI `rag context estimate` with arg, --file, --as-json
- CLI `rag context budget --session` (mocked session)
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import rag


# ── _estimate_tokens ─────────────────────────────────────────────────────


def test_estimate_empty_is_zero():
    assert rag._estimate_tokens("") == 0
    assert rag._estimate_tokens(None) == 0


def test_estimate_scales_linearly():
    t1 = rag._estimate_tokens("x" * 100)
    t2 = rag._estimate_tokens("x" * 1000)
    # ratio ~0.25 per char + 1 → ~25 vs ~250
    assert t2 > t1 * 8  # linear, not quadratic
    assert t1 >= 25


def test_estimate_tunable_via_env(monkeypatch):
    monkeypatch.setattr(rag, "_TOKENS_PER_CHAR", 0.5)
    assert rag._estimate_tokens("x" * 100) == 51


# ── _context_budget_status ───────────────────────────────────────────────


def test_budget_ok_under_warn_threshold():
    out = rag._context_budget_status(1000, num_ctx=4096)
    assert out["level"] == "ok"
    assert out["ratio"] == pytest.approx(0.244, abs=0.001)


def test_budget_warn_at_80_pct():
    out = rag._context_budget_status(3300, num_ctx=4096)  # ~0.806
    assert out["level"] == "warn"


def test_budget_over_at_100_pct():
    out = rag._context_budget_status(4500, num_ctx=4096)
    assert out["level"] == "over"
    assert out["ratio"] > 1.0


def test_budget_zero_inputs_safe():
    out = rag._context_budget_status(0, num_ctx=4096)
    assert out["level"] == "ok"
    assert out["ratio"] == 0.0


def test_budget_invalid_num_ctx_defaults_to_one():
    """num_ctx=0 should be clamped to 1 to avoid div-by-zero."""
    out = rag._context_budget_status(100, num_ctx=0)
    # tokens(100) / num_ctx(1) = 100 → over
    assert out["level"] == "over"


# ── CLI: context estimate ────────────────────────────────────────────────


def test_cli_estimate_with_arg():
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["context", "estimate", "hola mundo"])
    assert result.exit_code == 0, result.output
    assert "tokens:" in result.output.lower()


def test_cli_estimate_as_json():
    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["context", "estimate", "texto de prueba", "--as-json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert "chars" in data
    assert "tokens_est" in data
    assert "level" in data


def test_cli_estimate_from_file(tmp_path):
    fpath = tmp_path / "prompt.txt"
    fpath.write_text("El quick brown fox " * 100)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["context", "estimate", "--file", str(fpath), "--as-json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["chars"] == len("El quick brown fox " * 100)
    assert data["tokens_est"] > 100


def test_cli_estimate_num_ctx_override():
    runner = CliRunner()
    # Short text with low num_ctx → warn or over.
    result = runner.invoke(
        rag.cli,
        ["context", "estimate", "x" * 400, "--num-ctx", "100", "--as-json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["level"] in ("warn", "over")


# ── CLI: context budget (session-based) ──────────────────────────────────


def test_cli_budget_missing_session_reports_gracefully(monkeypatch):
    monkeypatch.setattr(rag, "last_session_id", lambda: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["context", "budget"])
    assert result.exit_code == 0
    assert "sin sesión" in result.output.lower() or (
        "no encontrada" in result.output.lower()
    )


def test_cli_budget_non_existent_session(monkeypatch):
    monkeypatch.setattr(rag, "load_session", lambda sid: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["context", "budget", "--session", "missing"])
    assert result.exit_code == 0
    assert "no encontrada" in result.output.lower()


def test_cli_budget_active_session_as_json(monkeypatch):
    fake_sess = {
        "turns": [
            {"q": "hola", "a": "chau"},
            {"q": "cómo estás", "a": "bien"},
        ],
        "compressed_history": {"summary": "Conversación corta."},
    }
    monkeypatch.setattr(rag, "load_session", lambda sid: fake_sess)
    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["context", "budget", "--session", "abc123", "--as-json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["session"] == "abc123"
    assert data["n_turns"] == 2
    assert data["tokens_est"] > 0
    assert data["level"] in ("ok", "warn", "over")
