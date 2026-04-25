"""Feature #12 del 2026-04-23 — `rag session export` tests.

Validates:
- _render_session_to_markdown frontmatter + structure
- CLI: export with SID, default to last session, --stdout, --no-sources
- Overwrite protection (suffix append on collision)
- Empty session handled gracefully
"""
from __future__ import annotations


from click.testing import CliRunner

import rag


# ── _render_session_to_markdown ──────────────────────────────────────────


def _make_session(n_turns: int = 2, sid: str = "abc123") -> dict:
    turns = []
    for i in range(n_turns):
        turns.append({
            "q": f"pregunta {i + 1}",
            "a": f"respuesta {i + 1}",
            "paths": [f"01-Projects/note-{i}.md", f"02-Areas/other-{i}.md"],
        })
    return {
        "id": sid,
        "created_at": "2026-04-23T10:00:00",
        "updated_at": "2026-04-23T10:05:00",
        "mode": "chat",
        "turns": turns,
    }


def test_render_includes_frontmatter():
    sess = _make_session(2, "s123")
    md = rag._render_session_to_markdown(sess)
    assert md.startswith("---")
    assert "session_id: s123" in md
    assert "n_turns: 2" in md
    assert "tags: [conversation, rag-export]" in md


def test_render_has_title_from_first_question():
    sess = _make_session(1, "x")
    md = rag._render_session_to_markdown(sess)
    assert "# pregunta 1" in md


def test_render_renders_all_turns():
    sess = _make_session(3, "y")
    md = rag._render_session_to_markdown(sess)
    assert "## Turno 1" in md
    assert "## Turno 2" in md
    assert "## Turno 3" in md
    assert "pregunta 1" in md
    assert "respuesta 3" in md


def test_render_includes_source_wikilinks():
    sess = _make_session(1, "z")
    md = rag._render_session_to_markdown(sess)
    assert "[[01-Projects/note-0.md]]" in md
    assert "[[02-Areas/other-0.md]]" in md


def test_render_respects_no_sources_flag():
    sess = _make_session(1, "z")
    md = rag._render_session_to_markdown(sess, include_sources=False)
    assert "[[01-Projects/note-0.md]]" not in md
    assert "**Sources:**" not in md


def test_render_handles_empty_answer():
    sess = {
        "id": "empty",
        "created_at": "",
        "updated_at": "",
        "mode": "chat",
        "turns": [{"q": "hola", "a": ""}],
    }
    md = rag._render_session_to_markdown(sess)
    assert "*(sin respuesta)*" in md


def test_render_truncates_long_title():
    long_q = "x" * 200
    sess = {
        "id": "long", "created_at": "", "updated_at": "", "mode": "chat",
        "turns": [{"q": long_q, "a": "ok", "paths": []}],
    }
    md = rag._render_session_to_markdown(sess)
    # Title line: find the "# " start.
    import re
    m = re.search(r"^# (.+)$", md, re.MULTILINE)
    assert m is not None
    assert len(m.group(1)) <= 80


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_export_missing_sid_without_last(monkeypatch):
    monkeypatch.setattr(rag, "last_session_id", lambda: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["session", "export"])
    assert result.exit_code == 0
    assert "No hay sesión" in result.output


def test_cli_export_session_not_found(monkeypatch):
    monkeypatch.setattr(rag, "load_session", lambda sid: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["session", "export", "missing"])
    assert result.exit_code == 0
    assert "no encontrada" in result.output.lower()


def test_cli_export_empty_session(monkeypatch):
    monkeypatch.setattr(
        rag, "load_session",
        lambda sid: {"id": sid, "turns": [], "created_at": "", "updated_at": "", "mode": "chat"},
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["session", "export", "empty1"])
    assert result.exit_code == 0
    assert "vacía" in result.output.lower()


def test_cli_export_to_stdout(monkeypatch):
    monkeypatch.setattr(rag, "load_session", lambda sid: _make_session(2, sid))
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["session", "export", "stdout-test", "--stdout"],
    )
    assert result.exit_code == 0
    # Markdown in stdout
    assert "session_id: stdout-test" in result.output
    assert "## Turno 1" in result.output


def test_cli_export_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "load_session", lambda sid: _make_session(1, sid))
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    # Stub get_db + _index_single_file to avoid touching real DB.
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_index_single_file", lambda col, p: None)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["session", "export", "file-test", "--folder", "sub/dir"],
    )
    assert result.exit_code == 0, result.output
    # File should exist under tmp_path/sub/dir/
    target_dir = tmp_path / "sub" / "dir"
    assert target_dir.exists()
    mds = list(target_dir.glob("*.md"))
    assert len(mds) == 1
    content = mds[0].read_text(encoding="utf-8")
    assert "session_id: file-test" in content


def test_cli_export_overwrite_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "load_session", lambda sid: _make_session(1, sid))
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_index_single_file", lambda col, p: None)

    # Pre-create a file with the expected path.
    runner = CliRunner()
    # First run.
    result1 = runner.invoke(
        rag.cli,
        ["session", "export", "collision", "--folder", "c",
         "--filename", "fixed-name"],
    )
    assert result1.exit_code == 0, result1.output
    # Second run with same filename → should produce a different file.
    result2 = runner.invoke(
        rag.cli,
        ["session", "export", "collision", "--folder", "c",
         "--filename", "fixed-name"],
    )
    assert result2.exit_code == 0, result2.output
    files = list((tmp_path / "c").glob("*.md"))
    assert len(files) == 2


def test_cli_export_force_overwrites(tmp_path, monkeypatch):
    """--force sobreescribe un archivo existente."""
    monkeypatch.setattr(rag, "load_session", lambda sid: _make_session(1, sid))
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_index_single_file", lambda col, p: None)

    runner = CliRunner()
    runner.invoke(rag.cli,
                  ["session", "export", "f", "--folder", "f",
                   "--filename", "fixed", "--force"])
    runner.invoke(rag.cli,
                  ["session", "export", "f", "--folder", "f",
                   "--filename", "fixed", "--force"])
    files = list((tmp_path / "f").glob("*.md"))
    # --force → same path overwritten, only 1 file total.
    assert len(files) == 1
