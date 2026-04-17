"""Tests for the nightly online-tune launchd service and RAG_EXPLORE plists."""
import plistlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag as rag_module

RAG_BIN = "/usr/local/bin/rag"


def _parse_plist(xml: str) -> dict:
    return plistlib.loads(xml.encode())


def test_online_tune_plist_valid_plist():
    xml = rag_module._online_tune_plist(RAG_BIN)
    result = subprocess.run(
        ["plutil", "-lint", "-"],
        input=xml.encode(),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_online_tune_plist_label():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    assert d["Label"] == "com.fer.obsidian-rag-online-tune"


def test_online_tune_plist_schedule_0330():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    cal = d["StartCalendarInterval"]
    assert cal["Hour"] == 3
    assert cal["Minute"] == 30
    assert "Weekday" not in cal  # runs every day, not day-restricted


def test_online_tune_plist_program_arguments():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    args = d["ProgramArguments"]
    assert args[0] == RAG_BIN
    assert "tune" in args
    assert "--online" in args
    assert "--days" in args
    assert "14" in args
    assert "--apply" in args
    assert "--yes" in args


def test_online_tune_plist_no_rag_explore():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_online_tune_plist_one_shot():
    d = _parse_plist(rag_module._online_tune_plist(RAG_BIN))
    assert d.get("KeepAlive") is False
    assert d.get("RunAtLoad") is False


def test_morning_plist_has_rag_explore():
    d = _parse_plist(rag_module._morning_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert env.get("RAG_EXPLORE") == "1"


def test_today_plist_has_rag_explore():
    d = _parse_plist(rag_module._today_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert env.get("RAG_EXPLORE") == "1"


def test_watch_plist_no_rag_explore():
    d = _parse_plist(rag_module._watch_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_digest_plist_no_rag_explore():
    d = _parse_plist(rag_module._digest_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_emergent_plist_no_rag_explore():
    d = _parse_plist(rag_module._emergent_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_patterns_plist_no_rag_explore():
    d = _parse_plist(rag_module._patterns_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_archive_plist_no_rag_explore():
    d = _parse_plist(rag_module._archive_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_wa_tasks_plist_no_rag_explore():
    d = _parse_plist(rag_module._wa_tasks_plist(RAG_BIN))
    env = d.get("EnvironmentVariables", {})
    assert "RAG_EXPLORE" not in env


def test_services_spec_includes_online_tune():
    specs = rag_module._services_spec(RAG_BIN)
    labels = [s[0] for s in specs]
    assert "com.fer.obsidian-rag-online-tune" in labels


def test_services_spec_total_count():
    specs = rag_module._services_spec(RAG_BIN)
    assert len(specs) == 9
