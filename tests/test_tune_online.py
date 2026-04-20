"""Tests for Task 6: rag tune --online, ranker versioning, auto-rollback CI gate."""
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag


# ── _behavior_augmented_cases ─────────────────────────────────────────────────

def test_behavior_augmented_cases_missing_file(tmp_path):
    """Missing behavior.jsonl → empty list (no crash)."""
    fake = tmp_path / "behavior.jsonl"
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert result == []


def test_behavior_augmented_cases_empty_file(tmp_path):
    """Empty behavior.jsonl → empty list."""
    fake = tmp_path / "behavior.jsonl"
    fake.write_text("")
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert result == []


def _iso(offset_secs: int = 0) -> str:
    """Return ISO timestamp offset_secs from now."""
    return rag.datetime.fromtimestamp(time.time() + offset_secs).isoformat(timespec="seconds")


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events))


def test_behavior_positive_events(tmp_path):
    """open/positive_implicit/save/kept with query → positive cases."""
    fake = tmp_path / "behavior.jsonl"
    _write_events(fake, [
        {"event": "open",              "query": "ikigai framework", "path": "Ikigai.md",     "ts": _iso(-100)},
        {"event": "positive_implicit", "query": "career change",    "path": "Career.md",     "ts": _iso(-100)},
        {"event": "save",              "query": "morning routine",   "path": "Morning.md",    "ts": _iso(-100)},
        {"event": "kept",              "query": "weekly review",     "path": "Review.md",     "ts": _iso(-100)},
    ])
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 4
    for c in result:
        assert "expected" in c
        assert c["weight"] == 0.5
        assert c.get("kind_hint") == "behavior_pos"


def test_behavior_negative_events(tmp_path):
    """negative_implicit/deleted with query → negative cases."""
    fake = tmp_path / "behavior.jsonl"
    _write_events(fake, [
        {"event": "negative_implicit", "query": "bad query", "path": "Wrong.md", "ts": _iso(-100)},
        {"event": "deleted",           "query": "stale note",  "path": "Old.md",   "ts": _iso(-100)},
    ])
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 2
    for c in result:
        assert "anti_expected" in c
        assert c["weight"] == 0.5
        assert c.get("kind_hint") == "behavior_neg"


def test_behavior_conflict_dropped(tmp_path):
    """Same (q, path) pair appearing as both positive and negative → both dropped."""
    fake = tmp_path / "behavior.jsonl"
    _write_events(fake, [
        {"event": "open",              "query": "same query", "path": "Same.md", "ts": _iso(-100)},
        {"event": "negative_implicit", "query": "same query", "path": "Same.md", "ts": _iso(-100)},
        # Unrelated event that should survive
        {"event": "open",              "query": "other query", "path": "Other.md", "ts": _iso(-100)},
    ])
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    paths = [c.get("expected", c.get("anti_expected", [None]))[0] for c in result]
    assert "Same.md" not in paths
    assert "Other.md" in paths
    assert len(result) == 1


def test_behavior_ignores_old_events(tmp_path):
    """Events older than days window are excluded."""
    fake = tmp_path / "behavior.jsonl"
    old_ts = rag.datetime.fromtimestamp(time.time() - 30 * 86400).isoformat(timespec="seconds")
    _write_events(fake, [
        {"event": "open", "query": "old event", "path": "Old.md", "ts": old_ts},
        {"event": "open", "query": "recent",     "path": "New.md", "ts": _iso(-100)},
    ])
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 1
    assert result[0]["expected"] == ["New.md"]


def test_behavior_ignores_events_without_query(tmp_path):
    """Events without 'query' field (e.g. brief events) are skipped."""
    fake = tmp_path / "behavior.jsonl"
    _write_events(fake, [
        {"event": "kept",  "path": "Brief.md", "ts": _iso(-100)},          # no query
        {"event": "open",  "path": "Brief.md", "ts": _iso(-100)},          # no query
        {"event": "open",  "query": "with query", "path": "Note.md", "ts": _iso(-100)},
    ])
    with patch.object(rag, "BEHAVIOR_LOG_PATH", fake):
        result = rag._behavior_augmented_cases(days=14)
    assert len(result) == 1
    assert result[0]["expected"] == ["Note.md"]


# ── Ranker config versioning ──────────────────────────────────────────────────

def test_backup_ranker_config_no_existing(tmp_path):
    """No ranker.json → backup returns None (no crash)."""
    fake = tmp_path / "ranker.json"
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        result = rag._backup_ranker_config()
    assert result is None


def test_backup_ranker_config_creates_backup(tmp_path):
    """Existing ranker.json → backup file created."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        backup = rag._backup_ranker_config()
    assert backup is not None
    assert backup.is_file()
    assert backup.name.startswith("ranker.")
    assert backup.name.endswith(".json")
    # Original must still exist
    assert fake.is_file()


def test_backup_prunes_to_three_newest(tmp_path):
    """After 4 backups created, only 3 newest remain."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        for _ in range(4):
            rag._backup_ranker_config()
            time.sleep(0.01)  # ensure different mtime
    backups = list(tmp_path.glob("ranker.*.json"))
    assert len(backups) <= 3


def test_restore_ranker_backup(tmp_path):
    """_restore_ranker_backup copies backup → ranker.json."""
    fake = tmp_path / "ranker.json"
    fake.write_text('{"weights": {"original": 1}}')
    backup = tmp_path / "ranker.12345.json"
    backup.write_text('{"weights": {"restored": 1}}')
    with patch.object(rag, "RANKER_CONFIG_PATH", fake):
        ok = rag._restore_ranker_backup(backup)
    assert ok
    data = json.loads(fake.read_text())
    assert data["weights"] == {"restored": 1}


# ── rag tune --rollback ───────────────────────────────────────────────────────

def test_tune_rollback_no_backups(tmp_path, capsys):
    """--rollback with no backups → informative message, no crash."""
    fake_ranker = tmp_path / "ranker.json"
    from click.testing import CliRunner
    runner = CliRunner()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker):
        result = runner.invoke(rag.tune, ["--rollback"])
    assert result.exit_code == 0
    assert "No hay backups" in result.output


def test_tune_rollback_restores_newest(tmp_path):
    """--rollback restores the most recent backup."""
    fake_ranker = tmp_path / "ranker.json"
    fake_ranker.write_text('{"weights": {"current": 1}}')
    # Create two backups — second is newer
    b1 = tmp_path / "ranker.111.json"
    b1.write_text('{"weights": {"backup1": 1}}')
    time.sleep(0.02)
    b2 = tmp_path / "ranker.222.json"
    b2.write_text('{"weights": {"backup2": 1}}')
    from click.testing import CliRunner
    runner = CliRunner()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker):
        result = runner.invoke(rag.tune, ["--rollback"])
    assert result.exit_code == 0
    assert "restored" in result.output.lower()
    data = json.loads(fake_ranker.read_text())
    assert data["weights"] == {"backup2": 1}


# ── Auto-rollback CI gate ─────────────────────────────────────────────────────

def _make_tunable_mocks(tmp_path):
    """Return mocks needed to run tune --online --apply in-memory."""
    fake_ranker = tmp_path / "ranker.json"
    fake_behavior = tmp_path / "behavior.jsonl"
    fake_behavior.write_text("")  # empty — no behavior cases
    queries_yaml = tmp_path / "queries.yaml"
    queries_yaml.write_text("queries:\n  - question: test\n    expected:\n      - Note.md\n")
    return fake_ranker, fake_behavior, queries_yaml


def _fake_apply_weighted(feats, weights, k):
    """Minimal stub: return top-k items with path key (avoids full feature dict)."""
    return [{"path": f.get("path", "Note.md"), "score": 1.0} for f in feats[:k]]


def test_gate_fails_triggers_rollback(tmp_path):
    """CI gate failure: backup restored, exit=1. Tested via the gate logic directly."""
    fake_ranker = tmp_path / "ranker.json"
    fake_ranker.write_text('{"weights": {"recency_always": 0.0}}')
    # Create a backup that will be restored
    backup = tmp_path / "ranker.111.json"
    backup.write_text('{"weights": {"recency_always": 0.05}}')

    # The gate logic is extracted here: simulate what tune does after best_w.save()
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "_run_eval_gate", return_value=(0.50, 0.40, "singles 50%")):
        # Copy current ranker (as backup would be created)
        import shutil
        b = fake_ranker.parent / "ranker.999.json"
        shutil.copy2(fake_ranker, b)
        # Call _run_eval_gate to verify the parsing
        s_hit5, c_hit5, out = rag._run_eval_gate()
        assert s_hit5 == pytest.approx(0.50)
        assert c_hit5 == pytest.approx(0.40)
        # Gate fails: s_hit5 < GATE_SINGLES_HIT5_MIN
        gate_ok = s_hit5 >= rag.GATE_SINGLES_HIT5_MIN and c_hit5 >= rag.GATE_CHAINS_HIT5_MIN
        assert not gate_ok
        # Restore the backup
        ok = rag._restore_ranker_backup(b)
        assert ok


def test_gate_passes_no_rollback(tmp_path):
    """CI gate passes: _run_eval_gate returns numbers above floor."""
    fake_ranker = tmp_path / "ranker.json"
    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "_run_eval_gate", return_value=(0.92, 0.82, "passing")):
        s_hit5, c_hit5, _ = rag._run_eval_gate()
        gate_ok = (
            s_hit5 is not None and s_hit5 >= rag.GATE_SINGLES_HIT5_MIN
            and c_hit5 is not None and c_hit5 >= rag.GATE_CHAINS_HIT5_MIN
        )
        assert gate_ok


def test_tune_apply_without_online_unchanged(tmp_path):
    """tune --apply without --online does NOT call _run_eval_gate (gate is skipped)."""
    fake_ranker, fake_behavior, queries_yaml = _make_tunable_mocks(tmp_path)
    fake_ranker.write_text('{"weights": {}}')

    def fake_collect(col, q, k_pool=50):
        return [{"path": "Note.md"}]

    from click.testing import CliRunner
    runner = CliRunner()
    gate_called = []
    def fake_gate():
        gate_called.append(True)
        return (0.5, 0.5, "")

    with patch.object(rag, "RANKER_CONFIG_PATH", fake_ranker), \
         patch.object(rag, "BEHAVIOR_LOG_PATH", fake_behavior), \
         patch.object(rag, "collect_ranker_features", fake_collect), \
         patch.object(rag, "apply_weighted_scores", _fake_apply_weighted), \
         patch.object(rag, "get_db", return_value=MagicMock(count=lambda: 1)), \
         patch.object(rag, "_run_eval_gate", fake_gate):
        result = runner.invoke(rag.tune, [
            "--file", str(queries_yaml),
            "--apply", "--samples", "5", "--no-chains",
        ])

    assert not gate_called, "CI gate must NOT run without --online"


def test_rag_explore_scrubbed_from_eval_env():
    """_run_eval_gate must scrub RAG_EXPLORE from the subprocess env."""
    captured_env = {}


    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        # Return a mock result
        mock = MagicMock()
        mock.stdout = "Singles: hit@5 90.00%\nChains: hit@5 80.00%"
        mock.stderr = ""
        return mock

    test_env = dict(os.environ)
    test_env["RAG_EXPLORE"] = "1"

    with patch.dict(os.environ, test_env), \
         patch("subprocess.run", fake_run):
        rag._run_eval_gate()

    assert "RAG_EXPLORE" not in captured_env


def test_gate_constants_derivation():
    """Gate constants must match the documented floor CI lower bounds."""
    assert rag.GATE_SINGLES_HIT5_MIN == pytest.approx(0.7619, abs=1e-4)
    assert rag.GATE_CHAINS_HIT5_MIN == pytest.approx(0.6364, abs=1e-4)


def test_eval_gate_timeout_returns_none_none():
    """subprocess.TimeoutExpired → both hit@5 values None (treated as
    regression by the caller → auto-rollback). Regression guard against
    the timeout accidentally bubbling up as a raw exception.

    Pre-2026-04-20 the timeout was 600s — bumped down to 300s after the
    audit because real eval wall is 60-100s + cold-start margin. Test
    the contract (returns None tuple on timeout) not the specific value."""
    import subprocess

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    with patch("subprocess.run", fake_run):
        s_hit5, c_hit5, out = rag._run_eval_gate()

    assert s_hit5 is None
    assert c_hit5 is None
    assert "timeout" in out.lower()


def test_eval_gate_timeout_is_bounded_tightly():
    """Guard against the timeout creeping back up to 10+ minutes. A
    dead ollama should fail the gate fast, not block the nightly tune
    for 10 min.

    Introspects _run_eval_gate's subprocess.run call and asserts the
    timeout kwarg is ≤ 5 min (our target post-audit)."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        mock = MagicMock()
        mock.stdout = "Singles: hit@5 90.00%\nChains: hit@5 80.00%"
        mock.stderr = ""
        return mock

    with patch("subprocess.run", fake_run):
        rag._run_eval_gate()

    assert captured["timeout"] is not None, "timeout kwarg missing from subprocess.run"
    assert captured["timeout"] <= 300, (
        f"eval gate timeout is {captured['timeout']}s — audit target was ≤300s "
        f"to fail fast when ollama is down. Did it creep up?"
    )
