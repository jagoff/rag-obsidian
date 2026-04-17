"""Tests for brief diff signal: record_brief_written + _diff_brief_signal."""
import json
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_sidecars(tmp_path, monkeypatch):
    """Redirect all brief sidecar paths and behavior log to tmp_path."""
    bw = tmp_path / "brief_written.jsonl"
    bs = tmp_path / "brief_state.jsonl"
    bl = tmp_path / "behavior.jsonl"
    monkeypatch.setattr(rag, "BRIEF_WRITTEN_PATH", bw)
    monkeypatch.setattr(rag, "BRIEF_STATE_PATH", bs)
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", bl)
    return {"written": bw, "state": bs, "behavior": bl, "tmp": tmp_path}


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "05-Reviews").mkdir(parents=True)
    (vault / "02-Areas").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_sidecar_entry(path: Path, brief_type: str, brief_rel: str,
                         paths_cited: list[str], age_hours: float = 24.0):
    ts = (datetime.now() - timedelta(hours=age_hours)).isoformat(timespec="seconds")
    rec = {
        "ts": ts,
        "brief_type": brief_type,
        "brief_path": brief_rel,
        "paths_cited": paths_cited,
        "citations_by_section": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ── record_brief_written ───────────────────────────────────────────────────────

def test_record_brief_written_appends_valid_json(tmp_sidecars):
    paths = ["02-Areas/Foo.md", "03-Resources/Bar.md"]
    rag.record_brief_written("morning", Path("05-Reviews/2026-04-17.md"), paths, {"agenda": paths[:1]})
    lines = _read_jsonl(tmp_sidecars["written"])
    assert len(lines) == 1
    rec = lines[0]
    assert rec["brief_type"] == "morning"
    assert rec["paths_cited"] == paths
    assert "ts" in rec
    assert rec["citations_by_section"] == {"agenda": paths[:1]}


def test_record_brief_written_noop_on_io_error(tmp_sidecars, monkeypatch):
    # Point to a path whose parent is a read-only directory so open() raises.
    ro_dir = tmp_sidecars["tmp"] / "ro_dir"
    ro_dir.mkdir()
    os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)  # r-x: can't create files
    bad = ro_dir / "brief_written.jsonl"
    monkeypatch.setattr(rag, "BRIEF_WRITTEN_PATH", bad)
    try:
        # Must not raise
        rag.record_brief_written("morning", Path("05-Reviews/x.md"), ["a.md"], {})
        assert not bad.is_file()
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)  # restore so tmp_path cleanup works


# ── _diff_brief_signal: edge cases ────────────────────────────────────────────

def test_diff_brief_signal_no_sidecar_no_crash(tmp_sidecars, tmp_vault):
    # No brief_written.jsonl at all → should silently return
    rag._diff_brief_signal()
    assert _read_jsonl(tmp_sidecars["behavior"]) == []


def test_diff_brief_signal_entry_too_young_ignored(tmp_sidecars, tmp_vault):
    _write_sidecar_entry(
        tmp_sidecars["written"],
        "morning", "05-Reviews/2026-04-17.md",
        ["02-Areas/Foo.md"], age_hours=5.0,  # < 18h
    )
    rag._diff_brief_signal()
    assert _read_jsonl(tmp_sidecars["behavior"]) == []


def test_diff_brief_signal_entry_too_old_ignored(tmp_sidecars, tmp_vault):
    _write_sidecar_entry(
        tmp_sidecars["written"],
        "morning", "05-Reviews/2026-04-17.md",
        ["02-Areas/Foo.md"], age_hours=48.0,  # > 36h
    )
    rag._diff_brief_signal()
    assert _read_jsonl(tmp_sidecars["behavior"]) == []


# ── _diff_brief_signal: kept vs deleted ───────────────────────────────────────

def test_diff_kept_and_deleted(tmp_sidecars, tmp_vault):
    """2 of 3 cited wikilinks survive → 2 kept + 1 deleted."""
    # Write cited notes (so we can create a brief file that keeps some)
    (tmp_vault / "02-Areas").mkdir(exist_ok=True)
    (tmp_vault / "03-Resources").mkdir(exist_ok=True)
    (tmp_vault / "02-Areas" / "Ikigai.md").write_text("# Ikigai\n")
    (tmp_vault / "02-Areas" / "Goals.md").write_text("# Goals\n")
    (tmp_vault / "03-Resources" / "DeepWork.md").write_text("# DeepWork\n")

    # Simulate a brief that cited all three
    cited = [
        "02-Areas/Ikigai.md",
        "02-Areas/Goals.md",
        "03-Resources/DeepWork.md",
    ]
    brief_rel = "05-Reviews/2026-04-17.md"
    _write_sidecar_entry(tmp_sidecars["written"], "morning", brief_rel, cited, age_hours=24)

    # On-disk brief still has [[Ikigai]] and [[Goals]] but not Deep Work
    brief_file = tmp_vault / "05-Reviews" / "2026-04-17.md"
    brief_file.write_text(
        "# Morning brief — 2026-04-17\n\n"
        "Hoy hay que revisar [[Ikigai]] y los [[Goals]] del mes. Sin DeepWork.\n"
    )

    rag._diff_brief_signal()

    events = _read_jsonl(tmp_sidecars["behavior"])
    assert len(events) == 3
    by_path = {e["path"]: e["event"] for e in events}
    assert by_path["02-Areas/Ikigai.md"] == "kept"
    assert by_path["02-Areas/Goals.md"] == "kept"
    assert by_path["03-Resources/DeepWork.md"] == "deleted"
    assert all(e["source"] == "brief" for e in events)


def test_diff_brief_file_gone_all_deleted(tmp_sidecars, tmp_vault):
    """Brief file no longer on disk → all paths emitted as deleted."""
    cited = ["02-Areas/Foo.md", "02-Areas/Bar.md"]
    brief_rel = "05-Reviews/2026-04-16.md"
    _write_sidecar_entry(tmp_sidecars["written"], "morning", brief_rel, cited, age_hours=24)
    # Do NOT create the brief file on disk

    rag._diff_brief_signal()

    events = _read_jsonl(tmp_sidecars["behavior"])
    assert len(events) == 2
    assert all(e["event"] == "deleted" for e in events)
    assert {e["path"] for e in events} == set(cited)


# ── _diff_brief_signal: dedup ─────────────────────────────────────────────────

def test_diff_dedup_second_call_noop(tmp_sidecars, tmp_vault):
    """Second call emits nothing — brief_state prevents re-emission."""
    cited = ["02-Areas/Note.md"]
    brief_rel = "05-Reviews/2026-04-17.md"
    _write_sidecar_entry(tmp_sidecars["written"], "morning", brief_rel, cited, age_hours=24)

    # Brief file exists but doesn't contain the wikilink → deleted
    brief_file = tmp_vault / "05-Reviews" / "2026-04-17.md"
    brief_file.write_text("# Morning brief\n\nNada aqui.\n")

    rag._diff_brief_signal()
    first = _read_jsonl(tmp_sidecars["behavior"])
    assert len(first) == 1

    rag._diff_brief_signal()
    second = _read_jsonl(tmp_sidecars["behavior"])
    # Still only 1 event total — no new writes
    assert len(second) == 1


# ── _resolve_wikilinks_to_paths: ambiguous ───────────────────────────────────

def test_ambiguous_wikilink_skipped(tmp_sidecars, tmp_vault):
    """Ambiguous title (multiple paths) → skipped, not emitted as kept or deleted."""
    # Two notes with same title
    title_to_paths = {"Ambig": {"02-Areas/Ambig.md", "03-Resources/Ambig.md"}}
    paths, ambig = rag._resolve_wikilinks_to_paths(
        ["Ambig"], title_to_paths, tmp_vault
    )
    assert paths == []
    assert ambig == 1


# ── _resolve_wikilinks_to_paths: unambiguous ─────────────────────────────────

def test_unambiguous_wikilink_resolved(tmp_sidecars, tmp_vault):
    (tmp_vault / "02-Areas" / "Clear.md").write_text("# Clear\n")
    title_to_paths = {"Clear": {"02-Areas/Clear.md"}}
    paths, ambig = rag._resolve_wikilinks_to_paths(
        ["Clear"], title_to_paths, tmp_vault
    )
    assert paths == ["02-Areas/Clear.md"]
    assert ambig == 0
